"""``taskdistill train``: plan a run, prepare its data and train it with the chosen backend.

MLX training runs in a child process (``python -m taskdistill.train.mlx_lora``): mlx-lm's trainer sets
a process-wide wired-memory limit, and gradient checkpointing patches the transformer block class for
the whole process, so neither may leak into the calling process (for example a server). The base
model is resolved (downloaded on first use, with progress) in this process before the child starts.

Training a run id again replaces its directory before anything else is written, so a failed or
interrupted re-run never leaves the earlier ``train_log.json`` next to a different adapter. The earlier
adapter's eval outputs (``eval/<run-id>/`` and a task-level ``threshold.json`` chosen for it) are removed at
the same time, so ``serve --threshold auto`` never applies a threshold chosen on other weights. A run
directory recorded with a different base model, backend, label source or learning-rate schedule (two
configurations that map to one run id) is refused rather than replaced.

A torch run on a Hugging Face base is pinned to the commit the base resolves to when it is trained; the
commit goes to ``train_config.json``, ``train_log.json`` and the adapter's ``adapter_config.json``, and eval
and serve load that commit again.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taskdistill import paths
from taskdistill.backends.factory import TORCH_INSTALL_HINT, mlx_unavailable_reason, torch_unavailable_reason
from taskdistill.config import TaskSpec
from taskdistill.models import is_available_locally, resolve_model_path
from taskdistill.train.common import (
    BACKENDS,
    TrainBackendUnavailable,
    TrainConfig,
    TrainingFailed,
    plan_training,
    portable_path,
    prepare_run_data,
)

MLX_UNAVAILABLE = (
    "MLX training needs Apple Silicon (arm64 macOS 14+). Use --backend torch (install the torch extra: "
    "uvx --from 'taskdistill[torch] @ git+https://github.com/B0yko/taskdistill@v0.1.1' taskdistill ...)."
)
SELECTED_RUN_REASON = "only run so far; re-run eval --select to choose among runs on validation"
# mlx-lm announces its end-of-run save of the final weights; the run keeps the best-validation adapter instead.
_HIDDEN_CHILD_LINES = ("Saved final weights to",)
# What a run id stands for besides profile and seed: a run directory recorded with other values is another model.
_IDENTITY_FIELDS = ("base_model", "backend", "labels", "lr_schedule")
_COMMIT = re.compile(r"[0-9a-f]{40}")
THRESHOLD_FILE = "threshold.json"
SELECTED_RUN_FILE = "selected_run.json"

__all__ = ["TrainBackendUnavailable", "TrainResult", "TrainingFailed", "run_training"]


@dataclass
class TrainResult:
    run_id: str
    run_dir: Path
    log: dict[str, Any]


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}")
    if backend == "mlx" and mlx_unavailable_reason() is not None:
        raise TrainBackendUnavailable(MLX_UNAVAILABLE)
    if backend == "torch":
        reason = torch_unavailable_reason()
        if reason is not None:
            raise TrainBackendUnavailable(f"torch training is not installed ({reason}); {TORCH_INSTALL_HINT}")


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["TASKDISTILL_HOME"] = str(paths.home())
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_DISABLE"] = "1"
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    return env


def _ensure_base_model(cfg: TrainConfig, log: Callable[[str], Any]) -> None:
    """Fetch a missing base here, where download progress is visible; the training child then finds it cached."""
    if is_available_locally(cfg.base_model, cfg.base_revision):
        return
    at = f"@{cfg.base_revision}" if cfg.base_revision else ""
    log(f"downloading {cfg.base_model}{at} (first use; later runs read it from the Hugging Face cache)")
    resolve_model_path(cfg.base_model, cfg.base_revision)


def _pin_torch_revision(cfg: TrainConfig, log: Callable[[str], Any]) -> None:
    """Pin a Hugging Face base of a torch run to the commit it resolves to now (fetching it on first use).

    The torch equivalents of the MLX bases carry no pinned revision, so without this the adapter would be
    trained on whatever the default branch points at today and applied later to whatever it points at then.
    The commit is recorded as ``base_revision``; the trainer loads exactly that commit and writes it into the
    adapter's ``adapter_config.json``, which the torch backend reads. Local directories are left alone.
    """
    if cfg.backend != "torch" or cfg.base_revision is not None or Path(cfg.base_model).expanduser().exists():
        return
    if cfg.base_model.startswith("mlx-community/"):
        return  # MLX weights: the torch trainer refuses them with a clear error
    if not is_available_locally(cfg.base_model, None):
        log(f"downloading {cfg.base_model} (first use; later runs read it from the Hugging Face cache)")
    snapshot = resolve_model_path(cfg.base_model, None)
    if snapshot.parent.name == "snapshots" and _COMMIT.fullmatch(snapshot.name):
        cfg.base_revision = snapshot.name
        log(f"{cfg.base_model}: pinned to revision {snapshot.name}")


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _recorded_config(run_dir: Path) -> dict[str, Any] | None:
    """What an existing run directory says it was trained with: ``train_config.json``, else ``train_log.json``."""
    for name in ("train_config.json", "train_log.json"):
        data = _read_json_object(run_dir / name)
        if data is not None:
            return data
    return None


def _same_base(recorded: str, cfg: TrainConfig) -> bool:
    """Whether a recorded ``base_model`` names the base of ``cfg`` (a hub id, or a local directory in any form)."""
    new = cfg.base_model
    if recorded in (new, portable_path(new)):
        return True
    if not Path(new).is_absolute():
        return False
    candidate = Path(recorded) if Path(recorded).is_absolute() else paths.home() / recorded
    try:
        if candidate.resolve() == Path(new).resolve():
            return True
    except OSError:
        pass
    # Recorded before local bases outside the workspace were kept absolute: a cwd-relative path, or the name only.
    return recorded in (paths.relative_to_home(new), Path(new).name)


def _check_same_run(cfg: TrainConfig, run_dir: Path) -> None:
    """Refuse to replace a run directory that holds another model under the same run id.

    The run id keeps only a short base name (``Qwen2.5-0.5B`` and ``Qwen2.5-0.5B-Instruct-8bit`` share one), so
    two bases can map to one id; replacing one with the other would silently change what ``selected_run.json``
    and the eval outputs of that id point at.
    """
    recorded = _recorded_config(run_dir)
    if recorded is None:
        return
    differences = []
    for key in _IDENTITY_FIELDS:
        old = recorded.get(key)
        if not isinstance(old, str) or not old:
            continue  # older files may lack a field
        new = str(getattr(cfg, key))
        same = _same_base(old, cfg) if key == "base_model" else old == new
        if not same:
            shown = portable_path(new) if key == "base_model" else new
            differences.append(f"{key} {old!r}, not {shown!r}")
    if differences:
        raise TrainingFailed(
            f"run {cfg.run_id} already exists and was trained with a different configuration "
            f"({'; '.join(differences)}); the two map to the same run id. Delete "
            f"{paths.relative_to_home(run_dir)} to replace it, or train this one with another --seed"
        )


def _names_run(path: Path, rid: str) -> bool:
    data = _read_json_object(path) if path.is_file() else None
    return data is not None and data.get("run_id") == rid


def _remove_stale_eval(cfg: TrainConfig, log: Callable[[str], Any]) -> None:
    """Remove the eval outputs of an earlier adapter with this run id; they describe weights that are replaced.

    That is ``eval/<run-id>/`` (eval JSONs, plots, the run's own ``threshold.json``) and the task-level
    ``threshold.json`` when it was chosen for this run id. ``serve --threshold auto`` then asks for a new eval
    instead of applying a threshold chosen on another adapter's confidences.
    """
    removed: list[str] = []
    eval_root = paths.eval_dir(cfg.task)
    run_eval_dir = eval_root / cfg.run_id
    if run_eval_dir.is_dir() and run_eval_dir.resolve().parent == eval_root.resolve():
        shutil.rmtree(run_eval_dir)
        removed.append(paths.relative_to_home(run_eval_dir))
    task_threshold = paths.task_home(cfg.task) / THRESHOLD_FILE
    if _names_run(task_threshold, cfg.run_id):
        task_threshold.unlink()
        removed.append(paths.relative_to_home(task_threshold))
    if removed:
        log(
            f"removed the eval outputs of the earlier {cfg.run_id} adapter ({', '.join(removed)}); "
            f"re-run `taskdistill eval --task {cfg.task} --run {cfg.run_id}` after training "
            "(it chooses the cascade threshold again)"
        )


def _reset_run_dir(cfg: TrainConfig, log: Callable[[str], Any]) -> None:
    """Remove an earlier run with the same id (its log, plot, adapter and cached predictions) and its eval outputs.

    A run directory recorded with a different base model, backend, label source or schedule is refused.
    """
    run_dir = Path(cfg.run_dir)
    if run_dir.is_dir():
        if run_dir.resolve().parent != paths.runs_dir(cfg.task).resolve():
            raise TrainingFailed(f"refusing to replace {run_dir.name}: it is not a run directory of task '{cfg.task}'")
        _check_same_run(cfg, run_dir)
        log(f"run {cfg.run_id} exists; replacing it")
        shutil.rmtree(run_dir)
    _remove_stale_eval(cfg, log)


def _train_mlx_subprocess(cfg: TrainConfig, log: Callable[[str], Any]) -> None:
    run_dir = Path(cfg.run_dir)
    _ensure_base_model(cfg, log)
    tail: deque[str] = deque(maxlen=40)
    with tempfile.TemporaryDirectory(prefix="taskdistill-train-") as scratch:
        # The child gets absolute paths from outside the run directory; the run keeps the portable copy.
        config_path = cfg.save(Path(scratch) / "train_config.json", portable=False)
        command = [sys.executable, "-m", "taskdistill.train.mlx_lora", str(config_path)]
        proc = subprocess.Popen(
            command,
            cwd=run_dir,
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                text = line.rstrip("\n")
                tail.append(text)
                if not text.startswith(_HIDDEN_CHILD_LINES):
                    log(text)
            returncode = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
    if returncode != 0:
        detail = "\n".join(tail)
        raise TrainingFailed(f"MLX training exited with code {returncode}. Last output:\n{detail}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _provisional_non_teacher(target: Path, task: str) -> bool:
    """Whether ``selected_run.json`` is train's provisional entry for a run not trained on teacher labels.

    Earlier versions also wrote it for a gold-label run trained first; a teacher-label run replaces that entry.
    """
    data = _read_json_object(target)
    if data is None or data.get("reason") != SELECTED_RUN_REASON or not isinstance(data.get("run_id"), str):
        return False
    train_log = _read_json_object(paths.runs_dir(task) / data["run_id"] / "train_log.json")
    labels = None if train_log is None else train_log.get("labels")
    return isinstance(labels, str) and labels != "teacher"


def _write_selected_run_if_missing(task: str, rid: str, labels: str) -> None:
    """Name the first teacher-label run in ``selected_run.json`` until ``eval --select`` chooses on validation.

    Gold-label runs (the label-quality ceiling) are never selected: the student is trained on teacher labels.
    """
    if labels != "teacher":
        return
    target = paths.task_home(task) / SELECTED_RUN_FILE
    if target.exists() and not _provisional_non_teacher(target, task):
        return
    payload = {"run_id": rid, "reason": SELECTED_RUN_REASON, "date": datetime.now(UTC).isoformat(timespec="seconds")}
    _write_json_atomic(target, payload)


def _summary(cfg: TrainConfig, train_log: dict[str, Any]) -> str:
    """The closing line, from ``train_log.json`` (examples and epochs as trained, after the length filter)."""
    used, planned = train_log.get("n_train"), train_log.get("n_train_planned", cfg.n_train)
    examples = f"{used} examples" if used == planned else f"{used} of {planned} examples (rest over max_seq_len)"
    epochs = train_log.get("epochs")
    parts = [f"run {cfg.run_id} done in {train_log.get('wall_seconds') or 0:.0f} s: {examples}"]
    if isinstance(epochs, (int, float)):
        parts.append(f"{epochs:.2f} epochs")
    best = train_log.get("best_val_loss")
    if best is not None:
        parts.append(f"best validation loss {best:.4f} at iteration {train_log.get('best_iteration')}")
    return ", ".join(parts) + f"; adapter: {train_log.get('adapter_dir')}"


def run_training(
    spec: TaskSpec,
    *,
    backend: str = "mlx",
    base: str | None = None,
    profile: str | None = None,
    seed: int | None = None,
    labels: str = "teacher",
    log: Callable[[str], Any] = print,
) -> TrainResult:
    """Train one LoRA run and return its id, directory and ``train_log.json`` contents."""
    _check_backend(backend)
    cfg = plan_training(spec, base=base, profile=profile, seed=seed, labels=labels, backend=backend)
    _check_same_run(cfg, Path(cfg.run_dir))  # before anything is fetched or removed
    _pin_torch_revision(cfg, log)
    _reset_run_dir(cfg, log)
    stats = prepare_run_data(spec, cfg)
    cfg.save(Path(cfg.run_dir) / "train_config.json")
    dropped = f", {stats['dropped_no_gold']} dropped without gold" if cfg.labels == "gold" else ""
    log(
        f"run {cfg.run_id}: {cfg.backend}, {cfg.base_model}, {cfg.n_train} train / {cfg.n_valid} valid examples"
        f"{dropped}; {cfg.iters} iterations x batch {cfg.batch_size} ({cfg.epochs_trained:.2f} epochs planned), "
        f"validation every {cfg.steps_per_eval}"
    )
    if backend == "mlx":
        _train_mlx_subprocess(cfg, log)
    else:
        module = importlib.import_module("taskdistill.train.torch_lora")
        module.train_torch(cfg, spec, log=log)
    log_path = Path(cfg.run_dir) / "train_log.json"
    if not log_path.is_file():
        raise TrainingFailed(f"training finished without writing {paths.relative_to_home(log_path)}")
    train_log: dict[str, Any] = json.loads(log_path.read_text(encoding="utf-8"))
    _write_selected_run_if_missing(spec.task, cfg.run_id, cfg.labels)
    log(_summary(cfg, train_log))
    return TrainResult(run_id=cfg.run_id, run_dir=Path(cfg.run_dir), log=train_log)
