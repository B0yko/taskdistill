"""Training plan, run data and the training log, shared by the MLX and torch trainers.

A run lives in ``$TASKDISTILL_HOME/<task>/runs/<run-id>/``:

- ``data/``: the exact training and validation files of the run (with gold completions for
  ``--labels gold``), their ``.meta.jsonl`` sidecars, ``order.json`` and ``data_stats.json``;
- ``adapter/``: the LoRA checkpoint with the lowest validation loss, in the backend's native format;
- ``train_config.json`` (the plan, with workspace-relative paths), ``train_log.json`` and ``loss.png``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import platform
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from taskdistill import __version__, paths
from taskdistill.config import TaskSpec
from taskdistill.evaluate.splits import EvalRecord, TaskType, ValidationSplit
from taskdistill.hardware import hardware_info, load_average
from taskdistill.models import pinned_revision, torch_equivalent

BACKENDS = ("mlx", "torch")
PROFILES = ("quick", "full")
LABEL_SOURCES = ("teacher", "gold")

QUICK_MAX_ITERS = 200
QUICK_MAX_VAL_BATCHES = 10
LORA_SCALE = 20.0  # mlx-lm defaults; the torch trainer uses the same alpha/rank ratio
LORA_DROPOUT = 0.0
VALID_ORDER_SEED = 0  # validation order is the same for every seed, so capped val passes compare like with like

_MODEL_SUFFIXES = ("-instruct", "-4bit", "-8bit", "-3bit", "-6bit", "-bf16", "-fp16", "-mlx")
_VERSION_PACKAGES = {"mlx": ("mlx", "mlx-lm"), "torch": ("torch", "transformers", "peft")}


class TrainDataError(ValueError):
    """The curated data needed for training is missing or inconsistent."""


class TrainBackendUnavailable(RuntimeError):
    """The requested training backend cannot run on this machine or is not installed."""


class TrainingFailed(RuntimeError):
    """The training process did not produce a run."""


_PATH_FIELDS = ("run_dir", "data_dir", "base_model")


def _portable_path(value: str) -> str:
    """An absolute path rendered relative to the workspace (see :func:`taskdistill.paths.relative_to_home`)."""
    return paths.relative_to_home(value) if len(value) > 1 and Path(value).is_absolute() else value


@dataclass
class TrainConfig:
    """Everything a trainer needs; ``run_dir``, ``data_dir`` and a local ``base_model`` are absolute in memory."""

    task: str
    backend: str
    base_model: str
    base_revision: str | None
    profile: str
    seed: int
    labels: str
    lora_rank: int
    lora_layers: int | str
    learning_rate: float
    batch_size: int
    epochs: float
    max_seq_len: int
    n_train: int
    n_valid: int
    iters: int
    steps_per_eval: int
    val_batches: int
    run_id: str
    run_dir: str
    data_dir: str
    task_type: str = "classification"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def to_portable_dict(self) -> dict[str, Any]:
        """``to_dict()`` with absolute paths made workspace-relative, safe to keep in a shared run directory."""
        data = self.to_dict()
        for key in _PATH_FIELDS:
            data[key] = _portable_path(data[key])
        return data

    def save(self, path: Path, *, portable: bool = True) -> Path:
        """Write the config as JSON; ``portable=False`` keeps absolute paths (for a file outside the workspace)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_portable_dict() if portable else self.to_dict()
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        """Read a saved config; relative run/data directories (and a local base) resolve against the workspace."""
        data = json.loads(path.read_text(encoding="utf-8"))
        root = paths.home()
        for key in ("run_dir", "data_dir"):
            value = data.get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                data[key] = str(root / value)
        base = data.get("base_model")
        if isinstance(base, str) and base and not Path(base).is_absolute() and (root / base).exists():
            data["base_model"] = str(root / base)
        return cls.from_dict(data)

    @property
    def epochs_trained(self) -> float:
        return self.iters * self.batch_size / self.n_train if self.n_train else 0.0


# -- naming ----------------------------------------------------------------------------------------


def short_model_name(base: str) -> str:
    """``mlx-community/Qwen2.5-0.5B-Instruct-4bit`` -> ``qwen2.5-0.5b`` (also for local directories)."""
    name = re.split(r"[\\/]", base.rstrip("/\\"))[-1].lower()
    stripped = True
    while stripped:
        stripped = False
        for suffix in _MODEL_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[: -len(suffix)]
                stripped = True
    name = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-.")
    return name or "model"


def run_id(base: str, profile: str, seed: int, labels: str = "teacher", backend: str = "mlx") -> str:
    """Deterministic run id: ``<short>-<profile>-s<seed>[-gold][-torch]``."""
    rid = f"{short_model_name(base)}-{profile}-s{seed}"
    if labels == "gold":
        rid += "-gold"
    if backend == "torch":
        rid += "-torch"
    return rid


# -- schedule --------------------------------------------------------------------------------------


def compute_iterations(epochs: float, n_train: int, batch_size: int, profile: str) -> int:
    """``ceil(epochs * n_train / batch_size)``, capped at QUICK_MAX_ITERS for the quick profile."""
    if n_train <= 0 or batch_size <= 0:
        return 0
    iters = max(1, math.ceil(round(epochs * n_train / batch_size, 9)))
    return min(iters, QUICK_MAX_ITERS) if profile == "quick" else iters


def eval_every(iters: int, profile: str) -> int:
    """Validation every tenth of the run (full) or every quarter (quick)."""
    return max(1, iters // (4 if profile == "quick" else 10))


def val_batch_count(n_valid: int, batch_size: int, profile: str) -> int:
    """Validation batches per pass: -1 (all) for full, at most QUICK_MAX_VAL_BATCHES for quick."""
    if n_valid <= 0:
        return 0
    if profile != "quick":
        return -1
    return min(math.ceil(n_valid / batch_size), QUICK_MAX_VAL_BATCHES)


# -- curated data ----------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainDataError(f"{paths.relative_to_home(path)}:{n}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TrainDataError(f"{paths.relative_to_home(path)}:{n}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_split(task: str, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Rows of the curated ``<split>.jsonl`` and its line-aligned sidecar (None if absent)."""
    source = paths.data_dir(task)
    data_file = source / f"{split}.jsonl"
    if not data_file.is_file():
        if split == "valid":
            return [], None
        raise TrainDataError(
            f"no curated data for task '{task}': {paths.relative_to_home(data_file)} is missing "
            f"(run `taskdistill curate --task {task}` first)"
        )
    rows = read_jsonl(data_file)
    meta_file = source / f"{split}.meta.jsonl"
    metas = read_jsonl(meta_file) if meta_file.is_file() else None
    if metas is not None and len(metas) != len(rows):
        raise TrainDataError(
            f"{paths.relative_to_home(meta_file)} has {len(metas)} lines but {split}.jsonl has {len(rows)}; "
            "re-run curate"
        )
    return rows, metas


def _gold_ok(spec: TaskSpec, gold: Any) -> bool:
    if gold is None:
        return False
    if spec.type == "classification":
        return not spec.labels or str(gold) in spec.labels
    return isinstance(gold, (dict, str))


def _kept_indices(
    spec: TaskSpec, split: str, labels: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    rows, metas = _read_split(spec.task, split)
    if labels == "gold" and rows and metas is None:
        raise TrainDataError(f"--labels gold needs data/{split}.meta.jsonl with gold labels (re-run curate)")
    sidecar = metas if metas is not None else [{} for _ in rows]
    if labels == "gold":
        kept = [i for i, meta in enumerate(sidecar) if _gold_ok(spec, meta.get("gold"))]
    else:
        kept = list(range(len(rows)))
    return rows, sidecar, kept


def gold_completion(spec: TaskSpec, gold: Any) -> str:
    """The assistant text for a gold label: the label itself, or canonical JSON for extraction."""
    if spec.type == "classification":
        return str(gold)
    from taskdistill.tasks.extraction import canonical_output

    obj = json.loads(gold) if isinstance(gold, str) else gold
    return canonical_output(obj, spec.json_schema or {})


def _with_completion(row: dict[str, Any], content: str) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise TrainDataError("training rows must be chat examples ending with an assistant message")
    new_messages = [dict(m) for m in messages]
    new_messages[-1]["content"] = content
    return {**row, "messages": new_messages}


def plan_training(
    spec: TaskSpec,
    *,
    base: str | None = None,
    profile: str | None = None,
    seed: int | None = None,
    labels: str = "teacher",
    backend: str = "mlx",
) -> TrainConfig:
    """Resolve the run's hyperparameters, sizes and schedule from the task spec and the curated data."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {', '.join(BACKENDS)}")
    if labels not in LABEL_SOURCES:
        raise ValueError(f"unknown label source {labels!r}; expected teacher or gold")
    chosen_profile = profile or spec.train.profile
    if chosen_profile not in PROFILES:
        raise ValueError(f"unknown profile {chosen_profile!r}; expected quick or full")
    base_model = base or spec.student.base_model
    if Path(base_model).expanduser().exists():  # a local model directory; the trainer runs elsewhere
        base_model = str(Path(base_model).expanduser().resolve())
    elif backend == "torch":
        base_model = torch_equivalent(base_model)
    chosen_seed = spec.train.seed if seed is None else int(seed)

    _, _, train_kept = _kept_indices(spec, "train", labels)
    _, _, valid_kept = _kept_indices(spec, "valid", labels)
    n_train, n_valid = len(train_kept), len(valid_kept)
    if n_train == 0:
        what = "with a gold label " if labels == "gold" else ""
        raise TrainDataError(f"task '{spec.task}' has no training examples {what}in data/train.jsonl")

    batch_size = max(1, min(spec.train.batch_size, n_train))
    iters = compute_iterations(spec.train.epochs, n_train, batch_size, chosen_profile)
    rid = run_id(base_model, chosen_profile, chosen_seed, labels, backend)
    run_dir = paths.runs_dir(spec.task) / rid
    return TrainConfig(
        task=spec.task,
        backend=backend,
        base_model=base_model,
        base_revision=pinned_revision(base_model),
        profile=chosen_profile,
        seed=chosen_seed,
        labels=labels,
        lora_rank=spec.train.lora_rank,
        lora_layers=spec.train.lora_layers,
        learning_rate=spec.train.learning_rate,
        batch_size=batch_size,
        epochs=spec.train.epochs,
        max_seq_len=spec.train.max_seq_len,
        n_train=n_train,
        n_valid=n_valid,
        iters=iters,
        steps_per_eval=eval_every(iters, chosen_profile),
        val_batches=val_batch_count(n_valid, batch_size, chosen_profile),
        run_id=rid,
        run_dir=str(run_dir),
        data_dir=str(run_dir / "data"),
        task_type=spec.type,
    )


def prepare_run_data(spec: TaskSpec, cfg: TrainConfig) -> dict[str, Any]:
    """Write the run's ``data/{train,valid}.jsonl`` (+ sidecars) in a seeded shuffled order.

    With ``labels == "gold"`` the assistant message is replaced by the canonical gold output from the
    sidecar, and rows without a usable gold label are dropped and counted.
    """
    out = Path(cfg.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"labels": cfg.labels, "splits": {}}
    order: dict[str, Any] = {"train_seed": cfg.seed, "valid_seed": VALID_ORDER_SEED}
    for split, seed in (("train", cfg.seed), ("valid", VALID_ORDER_SEED)):
        rows, sidecar, kept = _kept_indices(spec, split, cfg.labels)
        shuffled = list(kept)
        random.Random(seed).shuffle(shuffled)
        out_rows, out_metas = [], []
        for i in shuffled:
            row = rows[i]
            if cfg.labels == "gold":
                row = _with_completion(row, gold_completion(spec, sidecar[i]["gold"]))
            out_rows.append(row)
            out_metas.append(sidecar[i])
        write_jsonl(out / f"{split}.jsonl", out_rows)
        write_jsonl(out / f"{split}.meta.jsonl", out_metas)
        order[split] = shuffled
        stats["splits"][split] = {
            "source_rows": len(rows),
            "rows": len(out_rows),
            "dropped_no_gold": len(rows) - len(kept),
        }
    n_train, n_valid = stats["splits"]["train"]["rows"], stats["splits"]["valid"]["rows"]
    if (n_train, n_valid) != (cfg.n_train, cfg.n_valid):
        raise TrainDataError("the curated data changed after the run was planned; plan the run again")
    stats["n_train"], stats["n_valid"] = n_train, n_valid
    stats["dropped_no_gold"] = sum(s["dropped_no_gold"] for s in stats["splits"].values())
    (out / "order.json").write_text(json.dumps(order) + "\n", encoding="utf-8")
    (out / "data_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def _user_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def validation_split_for(cfg: TrainConfig) -> ValidationSplit:
    """The run's validation data as a :class:`ValidationSplit` (checkpoint selection reads only this)."""
    task_type: TaskType
    if cfg.task_type == "classification":
        task_type = "classification"
    elif cfg.task_type == "extraction":
        task_type = "extraction"
    else:
        raise ValueError(f"unknown task type {cfg.task_type!r}")
    data = Path(cfg.data_dir)
    rows = read_jsonl(data / "valid.jsonl") if (data / "valid.jsonl").is_file() else []
    meta_file = data / "valid.meta.jsonl"
    metas = read_jsonl(meta_file) if meta_file.is_file() else [{} for _ in rows]
    records = []
    for i, (row, meta) in enumerate(zip(rows, metas, strict=True)):
        extra = meta.get("meta") if isinstance(meta.get("meta"), dict) else {}
        group = extra.get("group") if extra else None
        records.append(
            EvalRecord(
                id=str(meta.get("input_hash") or f"valid-{i}"),
                input=_user_text(row),
                gold=meta.get("gold"),
                teacher=meta.get("teacher"),
                group=None if group is None else str(group),
            )
        )
    return ValidationSplit(records, task_type)


# -- training log ----------------------------------------------------------------------------------


def adapter_size_mb(directory: Path | str) -> float:
    """Total size of the files in an adapter directory, in MB (10^6 bytes)."""
    root = Path(directory)
    if not root.is_dir():
        return 0.0
    return round(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6, 2)


def library_versions(backend: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version(), "taskdistill": __version__}
    for package in _VERSION_PACKAGES.get(backend, ()):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _json_safe(value: Any) -> Any:
    """Non-finite floats become null; absolute paths become workspace-relative."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if len(value) > 1 and Path(value).is_absolute():
            return paths.relative_to_home(value)
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _count_used_examples(record: dict[str, Any], cfg: TrainConfig, log: dict[str, Any]) -> None:
    """``n_train``/``n_valid`` count the examples the trainer used (after its ``max_seq_len`` filter).

    The planned counts stay in ``n_train_planned``/``n_valid_planned``, and ``epochs`` is always
    ``iterations * batch_size / n_train`` over the examples actually trained on.
    """
    dropped = record.get("dropped_too_long")
    for split, planned in (("train", cfg.n_train), ("valid", cfg.n_valid)):
        key = f"n_{split}"
        if key not in log:
            too_long = dropped.get(split) if isinstance(dropped, dict) else None
            record[key] = max(0, planned - int(too_long or 0))
        record[f"{key}_planned"] = planned
    used, iterations, batch_size = record["n_train"], record.get("iterations"), record.get("batch_size")
    if isinstance(iterations, int) and isinstance(batch_size, int):
        record["epochs"] = round(iterations * batch_size / used, 4) if used else 0.0


def write_train_log(cfg: TrainConfig, log: dict[str, Any]) -> Path:
    """Write ``train_log.json`` (backend measurements over the common fields) and ``loss.png``."""
    from taskdistill.evaluate.plots import plot_loss

    run_dir = Path(cfg.run_dir)
    adapter_dir = run_dir / "adapter"
    stats_file = Path(cfg.data_dir) / "data_stats.json"
    data_stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.is_file() else {}
    record: dict[str, Any] = {
        "run_id": cfg.run_id,
        "task": cfg.task,
        "backend": cfg.backend,
        "base_model": cfg.base_model,
        "base_revision": cfg.base_revision,
        "profile": cfg.profile,
        "seed": cfg.seed,
        "labels": cfg.labels,
        "n_train": cfg.n_train,
        "n_valid": cfg.n_valid,
        "dropped_no_gold": 0,
        "iterations": cfg.iters,
        "epochs": round(cfg.epochs_trained, 4),
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "lora_rank": cfg.lora_rank,
        "lora_layers": cfg.lora_layers,
        "max_seq_len": cfg.max_seq_len,
        "wall_seconds": None,
        "train_seconds": None,
        "peak_memory_gb": None,
        "tokens_per_second": None,
        "trained_tokens": None,
        "processed_tokens": None,
        "curve": {"train": [], "val": []},
        "best_iteration": None,
        "best_val_loss": None,
        "final_val_loss": None,
        "adapter_dir": paths.relative_to_home(adapter_dir),
        "adapter_size_mb": adapter_size_mb(adapter_dir),
        "hardware": hardware_info(),
        "versions": library_versions(cfg.backend),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "load_average": load_average(),
    }
    record.update(log)
    if "dropped_no_gold" in data_stats:  # the run's data preparation is the source of truth
        record["dropped_no_gold"] = int(data_stats["dropped_no_gold"])
    _count_used_examples(record, cfg, log)
    if record.get("tokens_per_second") is None and record.get("processed_tokens") and record.get("train_seconds"):
        record["tokens_per_second"] = round(record["processed_tokens"] / record["train_seconds"], 1)
    curve = record.get("curve") or {}
    loss_png = run_dir / "loss.png"
    best = record.get("best_iteration")
    plot_loss(
        curve.get("train", []),
        curve.get("val", []),
        loss_png,
        title=f"{cfg.run_id}: loss on completion tokens",
        best_iteration=best if isinstance(best, int) else None,
    )
    record["loss_plot"] = paths.relative_to_home(loss_png)
    path = run_dir / "train_log.json"
    path.write_text(json.dumps(_json_safe(record), indent=2) + "\n", encoding="utf-8")
    return path
