"""Choosing and building the teacher source: live calls through the budgeted client, or a recorded replay.

A recording is looked up in the workspace first (``$TASKDISTILL_HOME/<task>/teacher_recording.jsonl.gz``), then
among the packaged demo recordings (``taskdistill/_data/demos/<task>/teacher_recording.jsonl.gz``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal, get_args

from taskdistill import paths
from taskdistill.config import TaskSpec
from taskdistill.ledger import Ledger
from taskdistill.teacher.base import TeacherError, TeacherSource
from taskdistill.teacher.cache import ResponseCache
from taskdistill.teacher.client import LiveTeacher
from taskdistill.teacher.pricing import PricingError, PricingSnapshot
from taskdistill.teacher.replay import ReplayTeacher, load_recording

Mode = Literal["live", "replay"]
MODES: tuple[str, ...] = get_args(Mode)
RECORDING_NAME = "teacher_recording.jsonl.gz"
PRICING_SNAPSHOT_NAME = "pricing_snapshot.json"


class TeacherUnavailable(TeacherError):
    """The requested teacher mode cannot be used: no API key for live calls, or no recording to replay."""


def data_root() -> Traversable:
    """The package data directory (``taskdistill/_data``)."""
    return resources.files("taskdistill") / "_data"


def workspace_recording(task: str) -> Path:
    return paths.home() / task / RECORDING_NAME


def packaged_recording(task: str) -> Traversable:
    return data_root() / "demos" / task / RECORDING_NAME


def find_recording(task: str) -> Traversable | Path | None:
    """The workspace recording of ``task`` if present, else the packaged demo recording, else None."""
    local = workspace_recording(task)
    if local.is_file():
        return local
    packaged = packaged_recording(task)
    if packaged.is_file():
        return packaged
    return None


def _key_hint(spec: TaskSpec) -> str:
    names = [spec.teacher.api_key_env]
    if "OPENROUTER_API_KEY" not in names:
        names.append("OPENROUTER_API_KEY")
    return " or ".join(names)


def resolve_mode(spec: TaskSpec, requested: str | None) -> Mode:
    """``live`` or ``replay`` for ``spec``.

    An explicit request wins but must be usable (``live`` needs a teacher API key, ``replay`` a recording).
    Without one: ``live`` when an API key is set, else ``replay`` when a recording exists, else an error.
    """
    if requested is not None and requested not in MODES:
        raise ValueError(f"teacher mode must be one of {', '.join(MODES)}, got {requested!r}")
    has_key = spec.teacher.api_key() is not None
    if requested == "live":
        if not has_key:
            raise TeacherUnavailable(f"live teacher calls need an API key: set {_key_hint(spec)}")
        return "live"
    if requested == "replay":
        if find_recording(spec.task) is None:
            raise TeacherUnavailable(
                f"no teacher recording for task '{spec.task}': expected $TASKDISTILL_HOME/{spec.task}/{RECORDING_NAME} "
                "or a packaged demo recording"
            )
        return "replay"
    if has_key:
        return "live"
    if find_recording(spec.task) is not None:
        return "replay"
    raise TeacherUnavailable(
        f"no teacher API key (set {_key_hint(spec)}) and no recording for task '{spec.task}' "
        f"($TASKDISTILL_HOME/{spec.task}/{RECORDING_NAME})"
    )


def packaged_pricing() -> Traversable:
    return data_root() / PRICING_SNAPSHOT_NAME


def load_pricing(path: Path | str | None = None) -> PricingSnapshot:
    """The pricing snapshot at ``path``; by default the workspace snapshot, else the packaged one.

    The packaged snapshot (OpenRouter, 2026-09-26) covers the DeepSeek and Qwen teacher candidates; run
    ``taskdistill pricing refresh`` for current prices or other models.
    """
    if path is not None:
        return PricingSnapshot.load(path)
    local = paths.pricing_path()
    if local.is_file():
        return PricingSnapshot.load(local)
    source = packaged_pricing()
    try:
        return PricingSnapshot.from_json(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PricingError(f"the packaged pricing snapshot is not readable: {exc}") from exc


def make_teacher(
    spec: TaskSpec,
    *,
    mode: str,
    phase: str,
    run_id: str,
    run_cap: float | None = None,
    use_cache: bool = True,
    concurrency: int = 8,
    pricing: PricingSnapshot | None = None,
    cache: ResponseCache | None = None,
) -> TeacherSource:
    """Build the teacher source for ``mode``.

    ``replay`` answers from the task's recording after checking its manifest against ``spec``. ``live`` is the
    budgeted client on the workspace ledger, with the response cache unless ``use_cache`` is false (``serve``
    never reads the cache), the task cap from ``budget.usd_cap`` and ``run_cap`` from ``--max-usd``; it needs a
    teacher API key. ``cache`` replaces the workspace response cache (the bake-off keeps its own).
    """
    if mode == "replay":
        source = find_recording(spec.task)
        if source is None:
            raise TeacherUnavailable(
                f"no teacher recording for task '{spec.task}': expected $TASKDISTILL_HOME/{spec.task}/{RECORDING_NAME} "
                "or a packaged demo recording"
            )
        return ReplayTeacher(load_recording(source), spec)
    if mode != "live":
        raise ValueError(f"teacher mode must be one of {', '.join(MODES)}, got {mode!r}")
    api_key = spec.teacher.api_key()
    if api_key is None:
        raise TeacherUnavailable(f"live teacher calls need an API key: set {_key_hint(spec)}")
    return LiveTeacher(
        spec.teacher.base_url,
        api_key,
        ledger=Ledger(),
        pricing=pricing if pricing is not None else load_pricing(),
        task=spec.task,
        phase=phase,
        run_id=run_id,
        cache=(cache if cache is not None else ResponseCache()) if use_cache else None,
        run_cap=run_cap,
        task_cap=spec.budget.usd_cap,
        concurrency=concurrency,
    )


def new_run_id(command: str) -> str:
    """``<command>-<UTC yyyymmddThhmmss>``."""
    return f"{command}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
