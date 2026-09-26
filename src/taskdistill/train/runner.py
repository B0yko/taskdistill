"""``taskdistill train``: plan a run, prepare its data and train it with the chosen backend.

MLX training runs in a child process (``python -m taskdistill.train.mlx_lora``): mlx-lm's trainer sets
a process-wide wired-memory limit, and gradient checkpointing patches the transformer block class for
the whole process, so neither may leak into the calling process (for example a server). The base
model is resolved (downloaded on first use, with progress) in this process before the child starts.

Training a run id again replaces its directory before anything else is written, so a failed or
interrupted re-run never leaves the earlier ``train_log.json`` next to a different adapter.
"""

from __future__ import annotations

import importlib
import json
import os
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
    prepare_run_data,
)

MLX_UNAVAILABLE = (
    "MLX training needs Apple Silicon (arm64 macOS 14+). Use --backend torch (install the torch extra: "
    "uvx --from 'taskdistill[torch] @ git+https://github.com/B0yko/taskdistill' taskdistill ...)."
)
SELECTED_RUN_REASON = "only run so far; re-run eval --select to choose among runs on validation"
# mlx-lm announces its end-of-run save of the final weights; the run keeps the best-validation adapter instead.
_HIDDEN_CHILD_LINES = ("Saved final weights to",)

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


def _reset_run_dir(cfg: TrainConfig, log: Callable[[str], Any]) -> None:
    """Remove an earlier run with the same id (its log, plot, adapter and cached predictions)."""
    run_dir = Path(cfg.run_dir)
    if not run_dir.is_dir():
        return
    if run_dir.resolve().parent != paths.runs_dir(cfg.task).resolve():
        raise TrainingFailed(f"refusing to replace {run_dir.name}: it is not a run directory of task '{cfg.task}'")
    log(f"run {cfg.run_id} exists; replacing it")
    shutil.rmtree(run_dir)


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


def _write_selected_run_if_missing(task: str, rid: str) -> None:
    target = paths.task_home(task) / "selected_run.json"
    if target.exists():
        return
    payload = {"run_id": rid, "reason": SELECTED_RUN_REASON, "date": datetime.now(UTC).isoformat(timespec="seconds")}
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    _write_selected_run_if_missing(spec.task, cfg.run_id)
    log(_summary(cfg, train_log))
    return TrainResult(run_id=cfg.run_id, run_dir=Path(cfg.run_dir), log=train_log)
