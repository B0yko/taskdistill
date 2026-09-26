"""``taskdistill serve``: resolve the run, the threshold and the teacher, then start the cascade server.

- The run is ``--run`` or the one named in ``$TASKDISTILL_HOME/<task>/selected_run.json``; its adapter is
  ``runs/<run>/adapter`` and its base model comes from ``train_log.json``.
- ``--threshold auto`` reads ``$TASKDISTILL_HOME/<task>/threshold.json`` (written by ``eval``, chosen on
  validation), or the run's own ``eval/<run>/threshold.json`` when the task-level file belongs to another run;
  a number is used as given, ``0`` never escalates on confidence and ``inf`` always escalates.
- The teacher is live when an API key is set, else the recording (``--replay`` forces the recording). A live
  teacher must be in the pricing snapshot. ``serve`` never reads the response cache, and says in its banner
  when escalations are replayed.
- Binding to anything but localhost requires ``TASKDISTILL_SERVER_TOKEN``; clients then send it as a bearer token.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from taskdistill import paths
from taskdistill.capture.proxy import TOKEN_ENV, UnsafeBindError, bind_host, is_local_host
from taskdistill.config import TaskSpec
from taskdistill.evaluate.threshold import threshold_from_json
from taskdistill.serve.app import create_app
from taskdistill.serve.worker import ModelWorker
from taskdistill.store import Store
from taskdistill.teacher import factory as teacher_factory
from taskdistill.teacher.pricing import PricingError, PricingSnapshot

if TYPE_CHECKING:
    from taskdistill.backends.base import Backend

__all__ = [
    "BackendFactory",
    "ServeSetupError",
    "ServedRun",
    "UnsafeBindError",
    "build_server",
    "resolve_run",
    "resolve_threshold",
    "run_server",
    "server_token",
    "teacher_pricing",
]

BackendFactory = Callable[[str, str, "str | None"], "Backend"]
SELECTED_RUN = "selected_run.json"
THRESHOLD_FILE = "threshold.json"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class ServeSetupError(RuntimeError):
    """The server cannot start: no run or threshold, a wrong backend, an unpriced live teacher, a failed load."""


@dataclass(frozen=True)
class ServedRun:
    run_id: str
    run_dir: Path
    adapter_path: Path
    base_model: str
    backend: str | None
    source: str  # "given" | "selected_run.json"


@dataclass(frozen=True)
class ResolvedThreshold:
    value: float
    description: str
    warnings: tuple[str, ...] = ()


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ServeSetupError(f"cannot read {what} {paths.relative_to_home(path)}: {exc}") from exc
    if not isinstance(data, dict):
        raise ServeSetupError(f"{what} {paths.relative_to_home(path)} is not a JSON object")
    return data


def server_token(host: str) -> str | None:
    """``TASKDISTILL_SERVER_TOKEN`` when binding beyond localhost (required there), else None."""
    host = bind_host(host)
    if is_local_host(host):
        return None
    token = os.environ.get(TOKEN_ENV) or None
    if token is None:
        raise UnsafeBindError(
            f"refusing to serve on {host or '<all interfaces>'} without {TOKEN_ENV}; set it and send "
            f"'Authorization: Bearer <token>' from clients, or bind to 127.0.0.1"
        )
    return token


def resolve_run(spec: TaskSpec, run_id: str | None = None) -> ServedRun:
    """The run to serve: ``run_id``, else the one in ``selected_run.json``."""
    source = "given"
    if run_id is None:
        selected = paths.task_home(spec.task) / SELECTED_RUN
        if not selected.is_file():
            raise ServeSetupError(
                f"no run given and no {SELECTED_RUN} for task '{spec.task}': pass --run <run-id>, or run "
                f"`taskdistill eval --task {spec.task}` to select one on validation"
            )
        value = _read_json(selected, SELECTED_RUN).get("run_id")
        if not isinstance(value, str) or not value:
            raise ServeSetupError(f"{SELECTED_RUN} for task '{spec.task}' names no run_id")
        run_id, source = value, SELECTED_RUN
    if not _RUN_ID.fullmatch(run_id):
        raise ServeSetupError(f"invalid run id {run_id!r}")
    run_dir = paths.runs_dir(spec.task) / run_id
    log_path = run_dir / "train_log.json"
    if not log_path.is_file():
        raise ServeSetupError(
            f"run {run_id} of task '{spec.task}' not found (no {paths.relative_to_home(log_path)}); "
            f"train one with `taskdistill train --task {spec.task}`"
        )
    train_log = _read_json(log_path, "training log")
    base_model = train_log.get("base_model")
    if not isinstance(base_model, str) or not base_model:
        raise ServeSetupError(f"training log of run {run_id} names no base_model")
    adapter = run_dir / "adapter"
    if not adapter.is_dir():
        raise ServeSetupError(f"run {run_id} has no adapter directory ({paths.relative_to_home(adapter)})")
    backend = train_log.get("backend")
    return ServedRun(
        run_id=run_id,
        run_dir=run_dir,
        adapter_path=adapter,
        base_model=base_model,
        backend=backend if isinstance(backend, str) else None,
        source=source,
    )


def _describe(value: float) -> str:
    if math.isinf(value):
        return "always escalate"
    if value == 0:
        return "0 (never escalates on confidence)"
    return f"{value:.6g}"


def _read_threshold(path: Path, name: str) -> tuple[float, str | None]:
    """The threshold in an eval ``threshold.json`` (null = always escalate) and the run it was chosen for."""
    data = _read_json(path, name)
    if "threshold" not in data:
        raise ServeSetupError(f"{name} has no threshold")
    raw = data["threshold"]
    if raw is not None and (not isinstance(raw, int | float) or isinstance(raw, bool)):
        raise ServeSetupError(f"{name}: threshold must be a number or null")
    owner = data.get("run_id")
    return threshold_from_json(raw), owner if isinstance(owner, str) and owner else None


def resolve_threshold(spec: TaskSpec, value: str | float, run_id: str) -> ResolvedThreshold:
    """``auto`` reads the threshold ``eval`` chose on validation for ``run_id``; else a number.

    ``auto`` takes ``$TASKDISTILL_HOME/<task>/threshold.json`` (the selected run's) when it belongs to
    ``run_id``, else the run's own ``eval/<run_id>/threshold.json``; only when neither belongs to the run is
    the task-level file used, with a warning.
    """
    if isinstance(value, str) and value.strip().lower() == "auto":
        task_file = paths.task_home(spec.task) / THRESHOLD_FILE
        run_name = f"eval/{run_id}/{THRESHOLD_FILE}"
        run_file = paths.task_home(spec.task) / "eval" / run_id / THRESHOLD_FILE
        task_level = _read_threshold(task_file, THRESHOLD_FILE) if task_file.is_file() else None
        owner: str | None
        if task_level is not None and task_level[1] == run_id:
            (threshold, owner), name = task_level, THRESHOLD_FILE
        elif _RUN_ID.fullmatch(run_id) and run_file.is_file():
            (threshold, owner), name = _read_threshold(run_file, run_name), run_name
        elif task_level is not None:
            (threshold, owner), name = task_level, THRESHOLD_FILE
        else:
            raise ServeSetupError(
                f"--threshold auto needs {THRESHOLD_FILE} for task '{spec.task}' and run {run_id}: run "
                f"`taskdistill eval --task {spec.task} --run {run_id}` (it chooses the threshold on validation), "
                "or pass --threshold <number>"
            )
        warnings: tuple[str, ...] = ()
        if owner != run_id:
            warnings = (
                f"{name} was chosen for run {owner or '(unknown)'}, not for the served run {run_id}; "
                f"re-run `taskdistill eval --task {spec.task} --run {run_id}` or pass --threshold",
            )
        source = f"auto: {name}, chosen on validation for run {owner or '(unknown)'}"
        return ResolvedThreshold(threshold, f"{_describe(threshold)} ({source})", warnings)
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ServeSetupError(f"--threshold must be 'auto' or a number, got {value!r}") from exc
    if math.isnan(threshold) or threshold < 0:
        raise ServeSetupError(f"--threshold must be >= 0 ('inf' always escalates), got {value!r}")
    return ResolvedThreshold(threshold, f"{_describe(threshold)} (given)")


def teacher_pricing(spec: TaskSpec) -> PricingSnapshot:
    """The pricing snapshot for live escalations; it must price ``teacher.model``.

    The live client prices every call before sending it, so without a price each low-confidence escalation
    would quietly become a student fallback; serve refuses to start instead.
    """
    try:
        pricing = teacher_factory.load_pricing()
    except PricingError as exc:
        raise ServeSetupError(
            f"cannot price live teacher calls: {exc}; run `taskdistill pricing refresh` or serve with --replay"
        ) from exc
    try:
        pricing.price_for(spec.teacher.model, spec.teacher.provider)
    except KeyError as exc:
        message = str(exc.args[0]) if exc.args else str(exc)
        if "pricing refresh" not in message:
            message += "; run `taskdistill pricing refresh`"
        raise ServeSetupError(f"cannot price live teacher calls: {message}") from exc
    return pricing


def _url(host: str, port: int) -> str:
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def build_server(
    spec: TaskSpec,
    *,
    run_id: str | None = None,
    threshold: str | float = "auto",
    backend: str = "mlx",
    replay: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_usd: float | None = None,
    backend_factory: BackendFactory | None = None,
    store: Store | None = None,
) -> tuple[FastAPI, list[str]]:
    """The cascade app for ``spec`` and the startup banner lines.

    The backend is constructed here (so an unavailable backend fails before anything starts) and loaded on
    the app's model worker thread. ``backend_factory(name, base_model, adapter_path)`` defaults to
    :func:`taskdistill.backends.factory.load_backend`.
    """
    host = bind_host(host)
    token = server_token(host)
    run = resolve_run(spec, run_id)
    if run.backend is not None and run.backend != backend:
        raise ServeSetupError(
            f"run {run.run_id} was trained with the {run.backend} backend; its adapter cannot be loaded by the "
            f"{backend} backend: serve it with --backend {run.backend}"
        )
    chosen = resolve_threshold(spec, threshold, run.run_id)
    mode = teacher_factory.resolve_mode(spec, "replay" if replay else None)
    pricing = teacher_pricing(spec) if mode == "live" else None
    if backend_factory is None:
        from taskdistill.backends.factory import load_backend

        backend_factory = load_backend
    model = backend_factory(backend, run.base_model, str(run.adapter_path))

    teacher = teacher_factory.make_teacher(
        spec,
        mode=mode,
        phase="serve",
        run_id=teacher_factory.new_run_id("serve"),
        run_cap=max_usd,
        use_cache=False,
        pricing=pricing,
    )
    worker = ModelWorker(lambda: model, spec=spec)
    app = create_app(
        spec,
        worker=worker,
        teacher=teacher,
        threshold=chosen.value,
        store=store if store is not None else Store(),
        token=token,
        run_id=run.run_id,
    )

    cascade = spec.cascade
    run_note = "" if run.source == "given" else f", from {SELECTED_RUN}"
    if mode == "replay":
        teacher_line = "teacher: replay (recorded outputs; escalations are not live calls)"
    else:
        teacher_line = f"teacher: live ({spec.teacher.model} via {spec.teacher.base_url})"
    base = paths.relative_to_home(run.base_model) if Path(run.base_model).is_absolute() else run.base_model
    banner = [
        f"task: {spec.task} ({spec.type})",
        f"run: {run.run_id}{run_note} (base {base}, adapter {paths.relative_to_home(run.adapter_path)})",
        f"threshold: {chosen.description}",
        f"backend: {backend}",
        teacher_line,
        f"escalations: {cascade.escalation_response} responses; on teacher error: {cascade.on_teacher_error}",
        f"listening on {_url(host, port)}/v1 (OpenAI-compatible; any model name is accepted)",
    ]
    if mode == "live" and max_usd is not None:
        banner.append(f"spend cap for this server run: ${max_usd:.2f} (--max-usd)")
    if token is not None:
        banner.append(f"auth: 'Authorization: Bearer <{TOKEN_ENV}>' required on every endpoint")
    banner.extend(f"warning: {w}" for w in chosen.warnings)
    return app, banner


def run_server(
    spec: TaskSpec,
    *,
    run_id: str | None = None,
    threshold: str | float = "auto",
    backend: str = "mlx",
    replay: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_usd: float | None = None,
    backend_factory: BackendFactory | None = None,
    echo: Callable[[str], Any] = print,
    log_level: str = "info",
) -> None:
    """Build the server, print the banner, load the model on its worker thread and serve with uvicorn.

    One process with one model worker thread: requests are handled concurrently, generations one at a time.
    """
    import uvicorn

    app, banner = build_server(
        spec,
        run_id=run_id,
        threshold=threshold,
        backend=backend,
        replay=replay,
        host=host,
        port=port,
        max_usd=max_usd,
        backend_factory=backend_factory,
    )
    for line in banner:
        echo(line)
    worker: ModelWorker = app.state.worker
    echo("loading the student model ...")
    worker.start()
    if not worker.wait_ready():
        worker.stop()
        error = worker.load_error
        raise ServeSetupError(f"the student model failed to load: {type(error).__name__}: {error}") from error
    echo(f"model loaded in {(worker.load_ms or 0.0) / 1000.0:.1f} s")
    uvicorn.run(app, host=bind_host(host), port=port, log_level=log_level)
