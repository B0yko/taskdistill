"""Build a teacher recording from the response cache (must-have 13).

The records are the cached teacher answers for a set of request keys: the task's captured requests, the keys
curate sent for labelling, and any extra request bodies (smoke-test queries, the README request). A cached
row counts only when it was produced under the spec's routing and reasoning settings (the cache context), and
a request body must have been built for the spec's teacher model and settings, so a recording never holds
outputs from another model or provider under the spec's manifest. Records hold the output and its metadata,
never inputs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from taskdistill import paths
from taskdistill.config import TaskSpec
from taskdistill.store import Store
from taskdistill.teacher.base import TeacherResult
from taskdistill.teacher.cache import CONTEXT_EXCLUDED, ResponseCache, request_context
from taskdistill.teacher.factory import load_pricing
from taskdistill.teacher.pricing import PricingError
from taskdistill.teacher.replay import RecordingError, expected_manifest, load_recording, write_recording
from taskdistill.teacher.request_key import canonical_json, request_key
from taskdistill.teacher.requests import build_teacher_request

LABELLING_KEYS = "labelling_keys.txt"
#: How many missing keys an error message lists.
SHOWN_MISSING = 5


def spec_context(spec: TaskSpec) -> str:
    """The cache context of every request :func:`build_teacher_request` makes for ``spec`` (input-independent)."""
    return request_context(build_teacher_request(spec, ""))


def _context_fields(body: Mapping[str, Any]) -> dict[str, bytes]:
    return {k: canonical_json(v) for k, v in body.items() if k not in CONTEXT_EXCLUDED and v is not None}


def check_request(spec: TaskSpec, body: Mapping[str, Any], context: str | None = None) -> None:
    """:class:`RecordingError` unless ``body`` asks the spec's teacher model under the spec's settings.

    A body for another provider or reasoning setting has the same request key as the spec's request for the same
    input, so recording it would replay another provider's output as the spec's teacher.
    """
    where = spec.source or spec.task
    key = request_key(body)
    model = body.get("model")
    if model != spec.teacher.model:
        raise RecordingError(
            f"request {key[:12]} is for model {model!r}, but the recording is for teacher.model "
            f"{spec.teacher.model!r} of {where}"
        )
    if request_context(body) == (context if context is not None else spec_context(spec)):
        return
    wanted, given = _context_fields(build_teacher_request(spec, "")), _context_fields(body)
    differing = sorted(name for name in wanted.keys() | given.keys() if wanted.get(name) != given.get(name))
    raise RecordingError(
        f"request {key[:12]} was built with other {', '.join(differing) or 'settings'} than teacher.extra_body of "
        f"{where}; a recording holds only requests made under the spec's settings (its manifest records them)"
    )


def wanted_keys(spec: TaskSpec, keys: Iterable[str] | None, requests: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """The request keys to record, in order of first appearance: those of ``requests`` (each checked with
    :func:`check_request`), then the bare ``keys``."""
    context = spec_context(spec)
    wanted: list[str] = []
    for body in requests or ():
        check_request(spec, body, context)
        wanted.append(request_key(body))
    wanted.extend(key.strip() for key in keys or () if key.strip())
    return list(dict.fromkeys(wanted))


def collect_records(
    cache: ResponseCache, wanted: Iterable[str], context: str
) -> tuple[list[TeacherResult], list[str], int]:
    """Cached results for ``wanted`` under ``context``, the keys that are missing, and how many of those are cached
    under another context (another provider or reasoning setting)."""
    keys = sorted(set(wanted))
    found = {result.key: result for result in cache.iter_results(keys, context)}
    missing = [key for key in keys if key not in found]
    elsewhere = sum(1 for key in missing if key in cache)
    return [found[key] for key in sorted(found)], missing, elsewhere


def _default_pricing_date() -> str:
    try:
        return load_pricing().date
    except PricingError as exc:
        raise RecordingError(
            f"the recording manifest needs the pricing snapshot date, but the snapshot cannot be read ({exc}); "
            "run `taskdistill pricing refresh` or pass the snapshot date explicitly"
        ) from exc


def build_recording(
    spec: TaskSpec,
    *,
    keys: list[str] | None = None,
    requests: list[dict[str, Any]] | None = None,
    out: Path,
    cache: ResponseCache | None = None,
    pricing_date: str | None = None,
) -> dict[str, Any]:
    """Write the recording of ``keys`` and ``requests`` from the response cache to ``out``; return its manifest.

    Request bodies must have been built for the spec's model and settings (:func:`check_request`), and every
    wanted key must be cached under the spec's settings; otherwise :class:`RecordingError` says what differs, or
    how many keys are missing and names the first few. The manifest is :func:`expected_manifest` plus the record
    count and the time of the newest record, so the same cache gives the same bytes. ``pricing_date`` defaults to
    the date of the pricing snapshot in use; an unreadable snapshot is an error.
    """
    wanted = wanted_keys(spec, keys, requests)
    if not wanted:
        raise RecordingError(f"no request keys to record for task '{spec.task}'")
    source = cache if cache is not None else ResponseCache()
    results, missing, elsewhere = collect_records(source, wanted, spec_context(spec))
    if missing:
        shown = ", ".join(missing[:SHOWN_MISSING]) + (", ..." if len(missing) > SHOWN_MISSING else "")
        note = (
            f"; {elsewhere} of them are cached under another provider or reasoning setting than "
            f"teacher.extra_body of {spec.source or spec.task}"
            if elsewhere
            else ""
        )
        raise RecordingError(
            f"{len(missing)} of {len(wanted)} request keys are not in the response cache ({shown}){note}. "
            "Label them with the live teacher first (curate or the demo with --live), then record again."
        )
    manifest = expected_manifest(spec, pricing_date if pricing_date is not None else _default_pricing_date())
    path = write_recording(out, manifest, results)
    return load_recording(path).manifest


def _captured_keys(spec: TaskSpec, store: Store) -> list[str]:
    keys: list[str] = []
    for row in store.iter_captures(spec.task):
        if not row.captured or row.request_key is None or row.status is None or not 200 <= row.status < 300:
            continue
        try:
            body = json.loads(row.request_body or "")
        except ValueError:
            continue
        if isinstance(body, dict) and body.get("model") == spec.teacher.model:
            keys.append(row.request_key)
    return keys


def default_keys(spec: TaskSpec, store: Store) -> list[str]:
    """The task's captured request keys (successful, non-streamed, for the spec's teacher model) plus the keys in
    ``$TASKDISTILL_HOME/<task>/data/labelling_keys.txt``, deduplicated in order of first appearance."""
    keys = _captured_keys(spec, store)
    labelling = paths.home() / spec.task / "data" / LABELLING_KEYS
    if labelling.is_file():
        keys.extend(line.strip() for line in labelling.read_text(encoding="utf-8").splitlines() if line.strip())
    return list(dict.fromkeys(keys))
