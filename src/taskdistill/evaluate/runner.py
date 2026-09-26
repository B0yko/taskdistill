"""``taskdistill eval``: score a run against the teacher and the gold labels, and choose its cascade threshold.

Validation data drives every choice: the run (``select``), the threshold (on raw confidence), the isotonic fit and the
TF-IDF ``C``. The reported split (``test`` by default) is only scored, and each test scoring is appended to
``$TASKDISTILL_HOME/<task>/test_access_log.jsonl``. Outputs go to ``$TASKDISTILL_HOME/<task>/eval/<run-id>/``.

``selected_run.json`` is written last, once the selected run's ``threshold.json`` is in place (and, when the selected
run is the one evaluated, once its evaluation is complete), so a failed or interrupted ``eval --select`` never leaves
the selection pointing at a run whose threshold was not chosen. Every ``threshold.json`` names the run it was chosen
for and is stamped with that run's adapter hash and training date.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import shlex
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from taskdistill import paths
from taskdistill.config import TaskSpec
from taskdistill.evaluate.baseline import TfidfBaseline, tune_baseline
from taskdistill.evaluate.bootstrap import Stat, paired_bootstrap
from taskdistill.evaluate.calibration import IsotonicCalibrator, auroc, calibration_report, ece, fit_isotonic
from taskdistill.evaluate.data import data_sha256, load_split, load_train_pairs
from taskdistill.evaluate.metrics import (
    NO_GROUP,
    document_counts,
    encode_labels,
    extraction_scores,
    macro_f1_from_codes,
    micro_f1,
    per_group_scores,
    per_trait_breakdown,
    resolve_fields,
)
from taskdistill.evaluate.plots import plot_reliability, plot_threshold_curve
from taskdistill.evaluate.predictions import (
    LazyBackend,
    MessagesFn,
    adapter_sha256,
    make_header,
    predict_split,
    settings_sha256,
)
from taskdistill.evaluate.selection import MAX_LATENCY_RATIO, MIN_GAIN, choose_base_model, score_runs, select_run
from taskdistill.evaluate.splits import EvalRecord, TaskType, TestSplit, ValidationSplit
from taskdistill.evaluate.threshold import (
    ThresholdResult,
    apply_threshold,
    evaluate_operating_point,
    select_threshold,
    threshold_to_json,
    usable_records,
)
from taskdistill.evaluate.zero_shot import zero_shot_messages
from taskdistill.hardware import hardware_info
from taskdistill.predict import ScoringBackend
from taskdistill.train.common import short_model_name

__all__ = [
    "EvalError",
    "RunInfo",
    "count_test_scorings",
    "read_run",
    "read_selected_run",
    "run_eval",
    "select_on_validation",
]

RESAMPLES: Final = 1000
BOOTSTRAP_SEED: Final = 0
EVAL_SPLITS: Final = ("valid", "test")
SELECTED_RUN_FILE: Final = "selected_run.json"
THRESHOLD_FILE: Final = "threshold.json"
TEST_ACCESS_LOG: Final = "test_access_log.jsonl"
ZERO_SHOT_PREFIX: Final = "zero-shot-"
CLASSIFICATION_HEADLINE: Final = ("accuracy", "macro_f1", "agreement")
EXTRACTION_HEADLINE: Final = ("json_validity", "field_micro_f1", "field_exact_match", "doc_exact_match", "agreement")
GOLD_METRICS: Final = frozenset({"accuracy", "macro_f1", "field_micro_f1", "field_exact_match", "doc_exact_match"})
SEED_RULE: Final = "best validation metric among the seeds of each base model (ties: smallest run id)"
BASE_RULE: Final = (
    "the larger base model only if it gains at least 1 point on the validation metric and its student p95 stays "
    "under 3x the smaller one's; otherwise the smaller"
)
CONFIDENCE_DEFINITIONS: Final = {
    "classification": {
        "primary": "trie-constrained greedy label; product of the renormalised token probabilities",
        "alternative": "free greedy generation; exp(mean token log-prob), scored on the free output's own label",
    },
    "extraction": {
        "primary": "minimum over fields of the product of the value tokens' probabilities (0 for invalid output)",
        "alternative": "exp(mean token log-prob) of the whole output of the same generation",
    },
}
_EXTRACTION_GOLD_KEYS: Final = (
    "field_micro_f1",
    "field_precision",
    "field_recall",
    "field_exact_match",
    "doc_exact_match",
    "per_field_exact",
    "tp",
    "fp",
    "fn",
)
_SIZE = re.compile(r"(\d+(?:\.\d+)?)b(?![a-z0-9])")

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
BackendFactory = Callable[[str, str | None], ScoringBackend]
Log = Callable[[str], Any]


class EvalError(RuntimeError):
    """Evaluation cannot run: a missing run, selection file or split."""


@dataclass(frozen=True)
class RunInfo:
    """What eval needs to know about a run (or the zero-shot pseudo-run)."""

    run_id: str
    run_dir: Path
    base_model: str  # as passed to the backend
    base_label: str  # as reported (never an absolute path)
    labels: str  # teacher | gold | zero-shot
    seed: int | None
    profile: str | None
    backend: str | None
    adapter_dir: Path | None

    @property
    def zero_shot(self) -> bool:
        return self.labels == "zero-shot"

    def describe(self) -> dict[str, Any]:
        return {
            "base_model": self.base_label,
            "labels": self.labels,
            "seed": self.seed,
            "profile": self.profile,
            "backend": self.backend,
        }


# -- small helpers -----------------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read {paths.relative_to_home(path)}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalError(f"{paths.relative_to_home(path)}: expected a JSON object")
    return data


def _clean(value: Any) -> Any:
    """JSON-ready: numpy scalars as Python numbers, non-finite floats as null, paths workspace-relative."""
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return paths.relative_to_home(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` atomically: a reader sees the old file or the new one, never a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(payload), indent=2, ensure_ascii=False, allow_nan=False)
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _display_model(name: str) -> str:
    return paths.relative_to_home(name) if Path(name).is_absolute() else name


def _local_model(name: str) -> str:
    """A local base-model directory stored workspace-relative (as ``train_config.json`` keeps it) made absolute.

    Hugging Face ids and paths that do not exist under the workspace are returned unchanged.
    """
    if name and not Path(name).is_absolute():
        candidate = paths.home() / name
        if candidate.exists():
            return str(candidate)
    return name


def _portable_command(command: str) -> str:
    """``command`` with every absolute path (also in ``--opt=/path``) rendered relative to the workspace.

    Only the paths change; the rest of the string (quoting, spacing) is kept as given.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    found: set[str] = set()
    for token in tokens:
        option, sep, value = token.partition("=")
        candidate = value if sep and option.startswith("-") else token
        if len(candidate) > 1 and Path(candidate).is_absolute():
            found.add(candidate)
    for path in sorted(found, key=len, reverse=True):
        command = command.replace(path, paths.relative_to_home(path))
    return command


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _mean(values: npt.NDArray[Any]) -> float | None:
    return float(values.mean()) if values.size else None


def _latency(values: Sequence[float | None]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if not arr.size:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "n": int(arr.size),
    }


def _base_size(base: str) -> float | None:
    """Parameter count in billions read from the model name (``qwen2.5-0.5b`` -> 0.5), if it has one."""
    found = _SIZE.findall(short_model_name(base))
    return float(found[-1]) if found else None


def _default_factory(backend: str) -> BackendFactory:
    def make(base_model: str, adapter_path: str | None) -> ScoringBackend:
        from taskdistill.backends.factory import load_backend

        return load_backend(backend, base_model, adapter_path)

    return make


def _default_command(
    task: str,
    *,
    run_id: str | None,
    split: str,
    backend: str,
    select: bool,
    zero_shot: bool,
    fast: bool,
    base: str | None = None,
) -> str:
    parts = ["taskdistill", "eval", "--task", task]
    if run_id:
        parts += ["--run", run_id]
    parts += ["--split", split]
    if backend != "mlx":
        parts += ["--backend", backend]
    if select:
        parts.append("--select")
    if zero_shot:
        parts.append("--zero-shot")
    if base:
        parts += ["--base", shlex.quote(base)]
    if fast:
        parts.append("--fast")
    return " ".join(parts)


# -- runs --------------------------------------------------------------------------------------------------------


def selected_run_path(task: str) -> Path:
    return paths.task_home(task) / SELECTED_RUN_FILE


def _commit_selection(spec: TaskSpec, payload: Mapping[str, Any], log: Log) -> None:
    """Write ``selected_run.json``: only once the selected run's ``threshold.json`` has been written."""
    path = _write_json(selected_run_path(spec.task), payload)
    log(f"wrote {paths.relative_to_home(path)}")


def read_selected_run(task: str) -> dict[str, Any]:
    """Contents of ``selected_run.json``; a clear error when there is none."""
    path = selected_run_path(task)
    if not path.is_file():
        raise EvalError(
            f"no run selected for task '{task}': {paths.relative_to_home(path)} is missing. Train a run first, "
            "pass --run <run-id>, or choose among the runs with eval --select"
        )
    data = _read_json(path)
    if not isinstance(data.get("run_id"), str) or not data["run_id"]:
        raise EvalError(f"{paths.relative_to_home(path)} names no run_id")
    return data


def read_run(spec: TaskSpec, run_id: str) -> RunInfo:
    """A trained run: its ``train_log.json`` and adapter directory must exist."""
    run_dir = paths.runs_dir(spec.task) / run_id
    if not run_dir.is_dir():
        raise EvalError(f"run '{run_id}' not found: {paths.relative_to_home(run_dir)} does not exist")
    log_path = run_dir / "train_log.json"
    if not log_path.is_file():
        raise EvalError(f"run '{run_id}' has no train_log.json; it did not finish training")
    adapter = run_dir / "adapter"
    if not adapter.is_dir():
        raise EvalError(f"run '{run_id}' has no adapter directory ({paths.relative_to_home(adapter)})")
    train_log = _read_json(log_path)
    config_path = run_dir / "train_config.json"
    configured = _read_json(config_path).get("base_model") if config_path.is_file() else None
    reported = train_log.get("base_model") or configured or spec.student.base_model
    return RunInfo(
        run_id=run_id,
        run_dir=run_dir,
        base_model=_local_model(str(configured or reported)),
        base_label=_display_model(str(reported)),
        labels=str(train_log.get("labels") or "teacher"),
        seed=_optional_int(train_log.get("seed")),
        profile=_optional_str(train_log.get("profile")),
        backend=_optional_str(train_log.get("backend")),
        adapter_dir=adapter,
    )


def zero_shot_run(spec: TaskSpec, base_model: str, backend: str) -> RunInfo:
    """The pseudo-run ``zero-shot-<short base>[-torch]``: the base model with no adapter.

    The suffix keeps the MLX and torch variants of one base apart (as trained run ids do); the predictions cache
    header also names the base model, so two bases with the same short name never share cached rows.
    """
    base_model = _local_model(base_model)
    if not Path(base_model).is_absolute() and Path(base_model).expanduser().is_dir():
        base_model = str(Path(base_model).expanduser().resolve())  # a local directory given relative to the cwd
    run_id = ZERO_SHOT_PREFIX + short_model_name(base_model) + ("-torch" if backend == "torch" else "")
    return RunInfo(
        run_id=run_id,
        run_dir=paths.runs_dir(spec.task) / run_id,
        base_model=base_model,
        base_label=_display_model(base_model),
        labels="zero-shot",
        seed=None,
        profile=None,
        backend=backend,
        adapter_dir=None,
    )


def discover_candidates(spec: TaskSpec, backend: str, log: Log) -> list[RunInfo]:
    """Runs trained on teacher labels with an adapter and a training log, for ``backend``; gold runs never count."""
    candidates: list[RunInfo] = []
    for run_dir in sorted(p for p in paths.runs_dir(spec.task).iterdir() if p.is_dir()):
        if run_dir.name.startswith(ZERO_SHOT_PREFIX):
            continue
        if not (run_dir / "train_log.json").is_file() or not (run_dir / "adapter").is_dir():
            continue
        try:
            info = read_run(spec, run_dir.name)
        except EvalError as exc:
            log(f"  skipping {run_dir.name}: {exc}")
            continue
        if info.labels != "teacher":
            continue
        if info.backend is not None and info.backend != backend:
            log(f"  skipping {info.run_id}: trained with the {info.backend} backend, evaluating with {backend}")
            continue
        candidates.append(info)
    return candidates


def _selected_id(task: str) -> str | None:
    """The run named in ``selected_run.json``; None without one (or when it cannot be read)."""
    if not selected_run_path(task).is_file():
        return None
    try:
        return str(read_selected_run(task)["run_id"])
    except EvalError:
        return None


def _is_selected(task: str, run_id: str) -> bool:
    """Whether ``run_id`` is the selected run; with no ``selected_run.json`` every evaluated run counts as selected."""
    if not selected_run_path(task).is_file():
        return True
    return _selected_id(task) == run_id


# -- threshold ---------------------------------------------------------------------------------------------------


def _choose_threshold(
    spec: TaskSpec, valid: ValidationSplit, *, labels: Sequence[str], fields: Sequence[str], log: Log
) -> ThresholdResult:
    """The cascade threshold, chosen on validation from raw confidence."""
    cascade = spec.cascade
    threshold = select_threshold(
        valid,
        reference=cascade.reference,
        metric=cascade.metric,
        target=cascade.target,
        max_drop=cascade.max_drop,
        labels=list(labels) or None,
        fields=list(fields) or None,
    )
    chosen = "always escalate" if threshold.always_escalate else f"t = {threshold.threshold:.4f}"
    log(
        f"threshold on valid: {chosen}, escalation {threshold.escalation_rate:.1%}, {cascade.metric} "
        f"{threshold.quality:.4f} vs {cascade.reference} (target {threshold.target_value:.4f})"
    )
    if threshold.warning:
        log(f"warning: {threshold.warning}")
    return threshold


def _run_stamp(info: RunInfo) -> dict[str, Any]:
    """What identifies the weights a threshold was chosen on: the adapter's hash and the run's training date.

    Retraining a run id keeps the id but changes both, so a reader can tell a threshold chosen for earlier weights.
    """
    if info.adapter_dir is None:
        return {}
    log_path = info.run_dir / "train_log.json"
    trained = _read_json(log_path).get("date") if log_path.is_file() else None
    return {"adapter_sha256": adapter_sha256(info.adapter_dir), "train_date": _optional_str(trained)}


def _write_thresholds(
    spec: TaskSpec,
    info: RunInfo,
    threshold: ThresholdResult,
    *,
    date: str,
    log: Log,
    task_level: bool | None = None,
) -> tuple[Path, Path | None]:
    """``eval/<run-id>/threshold.json`` always; the task's ``threshold.json`` only for the selected run.

    The task-level file is what ``serve --threshold auto`` reads for the selected run, so evaluating another run (a
    seed, the larger base, the gold-label ceiling) never replaces it; that run's threshold stays in its own copy,
    which ``serve --run <run-id> --threshold auto`` reads. Zero-shot pseudo-runs are never served. ``task_level``
    True writes the task-level file for a run that is being selected (``selected_run.json`` is written after it);
    None decides from ``selected_run.json``.
    """
    run_id = info.run_id
    payload = {
        **threshold.to_dict(),
        "run_id": run_id,
        "task": spec.task,
        "split": "valid",
        "date": date,
        **_run_stamp(info),
    }
    run_path = _write_json(paths.eval_dir(spec.task) / run_id / THRESHOLD_FILE, payload)
    if info.zero_shot:
        return run_path, None
    if task_level if task_level is not None else _is_selected(spec.task, run_id):
        return run_path, _write_json(paths.task_home(spec.task) / THRESHOLD_FILE, payload)
    log(
        f"{THRESHOLD_FILE} kept for the selected run {_selected_id(spec.task) or '(unreadable selected_run.json)'}; "
        f"the threshold of {run_id} is in {paths.relative_to_home(run_path)} "
        f"(`serve --run {run_id} --threshold auto` reads it)"
    )
    return run_path, None


def _sync_selected_threshold(spec: TaskSpec, info: RunInfo, valid: ValidationSplit, log: Log) -> None:
    """For a selection that is not evaluated in the same call, choose and write the selected run's threshold.

    Both ``eval/<run-id>/threshold.json`` and the task-level file are written; ``selected_run.json`` follows.
    """
    labels = list(spec.labels) if spec.type == "classification" else []
    fields = _fields(spec, [valid]) if spec.type == "extraction" else []
    log(f"threshold for the selected run {info.run_id}:")
    threshold = _choose_threshold(spec, valid, labels=labels, fields=fields, log=log)
    _write_thresholds(spec, info, threshold, date=_now(), log=log, task_level=True)


# -- predictions -------------------------------------------------------------------------------------------------


def _predict[S: (ValidationSplit, TestSplit)](
    spec: TaskSpec,
    info: RunInfo,
    split: S,
    backend: ScoringBackend,
    *,
    alternatives: bool,
    backend_name: str,
    log: Log,
) -> S:
    messages_fn: MessagesFn | None = functools.partial(zero_shot_messages, spec) if info.zero_shot else None
    header = make_header(
        data_sha256=data_sha256(spec.task, split.name),
        adapter_sha256=adapter_sha256(info.adapter_dir),
        alternatives=alternatives,
        backend=backend_name,
        settings_sha256=settings_sha256(spec, messages_fn),
        base_model=info.base_label if info.adapter_dir is None else None,
    )
    return predict_split(
        spec,
        split,
        backend,
        cache_path=info.run_dir / f"preds_{split.name}.jsonl",
        alternatives=alternatives,
        header=header,
        messages_fn=messages_fn,
        log=log,
    )


class _Backends:
    """One lazily created backend per run for the whole eval call."""

    def __init__(self, factory: BackendFactory) -> None:
        self.factory = factory
        self.by_run: dict[str, LazyBackend] = {}

    def get(self, info: RunInfo) -> LazyBackend:
        if info.run_id not in self.by_run:
            adapter = None if info.adapter_dir is None else str(info.adapter_dir)
            self.by_run[info.run_id] = LazyBackend(functools.partial(self.factory, info.base_model, adapter))
        return self.by_run[info.run_id]

    def drop(self, run_id: str) -> None:
        """Free one run's backend (it is created again on next use)."""
        backend = self.by_run.pop(run_id, None)
        if backend is not None:
            backend.close()

    def release(self, keep: str | None = None) -> None:
        for run_id in list(self.by_run):
            if run_id != keep:
                self.by_run.pop(run_id).close()


# -- scoring -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Refs:
    task_type: TaskType
    gold: list[Any]
    teacher: list[Any]
    has_gold: BoolArray
    has_teacher: BoolArray
    labels: list[str]
    fields: list[str]

    @property
    def n(self) -> int:
        return len(self.gold)

    @classmethod
    def of(cls, records: Sequence[EvalRecord], task_type: TaskType, labels: list[str], fields: list[str]) -> _Refs:
        gold = [r.gold for r in records]
        teacher = [r.teacher for r in records]
        return cls(
            task_type=task_type,
            gold=gold,
            teacher=teacher,
            has_gold=np.asarray([g is not None for g in gold], dtype=np.bool_),
            has_teacher=np.asarray([t is not None for t in teacher], dtype=np.bool_),
            labels=labels,
            fields=fields,
        )


class _System:
    """One system's answers on a split: correctness per record, point metrics and resampled statistics."""

    headline: tuple[str, ...] = ()

    def __init__(self, refs: _Refs, answers: Sequence[Any]) -> None:
        if len(answers) != refs.n:
            raise ValueError(f"{len(answers)} answers for {refs.n} records")
        self.refs = refs
        self.answers = list(answers)
        self.correct_gold: BoolArray = np.zeros(refs.n, dtype=np.bool_)
        self.correct_teacher: BoolArray = np.zeros(refs.n, dtype=np.bool_)

    @property
    def primary_correct(self) -> BoolArray:
        """Correct against gold where a record has gold, otherwise against the teacher."""
        return np.asarray(np.where(self.refs.has_gold, self.correct_gold, self.correct_teacher), dtype=np.bool_)

    def available(self) -> list[str]:
        has_gold = bool(self.refs.has_gold.any())
        has_teacher = bool(self.refs.has_teacher.any())
        return [m for m in self.headline if (m not in GOLD_METRICS or has_gold) and (m != "agreement" or has_teacher)]

    def gold_index(self, idx: IntArray) -> IntArray:
        return idx[self.refs.has_gold[idx]]

    def teacher_index(self, idx: IntArray) -> IntArray:
        return idx[self.refs.has_teacher[idx]]

    def stat(self, metric: str, idx: IntArray) -> float | None:
        raise NotImplementedError

    def metrics(self) -> dict[str, Any]:
        raise NotImplementedError

    def all_index(self) -> IntArray:
        return np.arange(self.refs.n, dtype=np.int64)


class _ClassificationSystem(_System):
    headline = CLASSIFICATION_HEADLINE

    def __init__(self, refs: _Refs, answers: Sequence[Any]) -> None:
        super().__init__(refs, answers)
        self.correct_gold = np.asarray(
            [g is not None and a == g for g, a in zip(refs.gold, self.answers, strict=True)], dtype=np.bool_
        )
        self.correct_teacher = np.asarray(
            [t is not None and a == t for t, a in zip(refs.teacher, self.answers, strict=True)], dtype=np.bool_
        )
        self.gold_codes = encode_labels(refs.gold, refs.labels)
        self.pred_codes = encode_labels(self.answers, refs.labels)

    def stat(self, metric: str, idx: IntArray) -> float | None:
        if metric == "agreement":
            return _mean(self.correct_teacher[self.teacher_index(idx)])
        sel = self.gold_index(idx)
        if metric == "accuracy":
            return _mean(self.correct_gold[sel])
        if metric == "macro_f1":
            if not sel.size:
                return None
            return macro_f1_from_codes(self.gold_codes[sel], self.pred_codes[sel], len(self.refs.labels))
        raise ValueError(f"unknown classification metric {metric!r}")

    def metrics(self) -> dict[str, Any]:
        everything = self.all_index()
        out: dict[str, Any] = {m: self.stat(m, everything) for m in self.headline}
        out["n"] = self.refs.n
        out["n_gold"] = int(self.refs.has_gold.sum())
        return out


class _ExtractionSystem(_System):
    headline = EXTRACTION_HEADLINE

    def __init__(self, refs: _Refs, answers: Sequence[Any]) -> None:
        super().__init__(refs, answers)
        gold_docs = [g if isinstance(g, Mapping) else {} for g in refs.gold]
        teacher_docs = [t if isinstance(t, Mapping) else {} for t in refs.teacher]
        self.gold_counts = document_counts(gold_docs, self.answers, refs.fields)
        self.teacher_counts = document_counts(teacher_docs, self.answers, refs.fields)
        self.valid = self.gold_counts.valid
        self.correct_gold = np.asarray(self.gold_counts.doc_exact & refs.has_gold, dtype=np.bool_)
        self.correct_teacher = np.asarray(self.teacher_counts.doc_exact & refs.has_teacher, dtype=np.bool_)

    def stat(self, metric: str, idx: IntArray) -> float | None:
        if metric == "json_validity":
            return _mean(self.valid[idx])
        if metric == "agreement":
            sel = self.teacher_index(idx)
            counts = self.teacher_counts
        else:
            sel = self.gold_index(idx)
            counts = self.gold_counts
        if not sel.size:
            return None
        if metric in ("agreement", "field_micro_f1"):
            return micro_f1(int(counts.tp[sel].sum()), int(counts.fp[sel].sum()), int(counts.fn[sel].sum()))
        if metric == "field_exact_match":
            return _mean(counts.matches[sel])
        if metric == "doc_exact_match":
            return _mean(counts.doc_exact[sel])
        raise ValueError(f"unknown extraction metric {metric!r}")

    def metrics(self) -> dict[str, Any]:
        everything = self.all_index()
        gold_rows = np.flatnonzero(self.refs.has_gold)
        if gold_rows.size:
            scores = extraction_scores(
                [self.refs.gold[i] for i in gold_rows], [self.answers[i] for i in gold_rows], self.refs.fields
            )
        else:
            scores = dict.fromkeys(_EXTRACTION_GOLD_KEYS)
        scores["json_validity"] = self.stat("json_validity", everything)
        scores["agreement"] = self.stat("agreement", everything)
        scores["n"] = self.refs.n
        scores["n_gold"] = int(gold_rows.size)
        return scores

    def per_field(self, reference: str) -> dict[str, float] | None:
        mask = self.refs.has_gold if reference == "gold" else self.refs.has_teacher
        if not mask.any():
            return None
        counts = self.gold_counts if reference == "gold" else self.teacher_counts
        matches = counts.matches[mask]
        return {name: float(matches[:, j].mean()) for j, name in enumerate(counts.fields)}


def _system(refs: _Refs, answers: Sequence[Any]) -> _System:
    if refs.task_type == "classification":
        return _ClassificationSystem(refs, answers)
    return _ExtractionSystem(refs, answers)


def _teacher_metrics(teacher: _System) -> dict[str, Any]:
    """The teacher against gold (``{}`` without gold); agreement with itself is left out."""
    if not teacher.refs.has_gold.any():
        return {}
    out = teacher.metrics()
    out.pop("agreement", None)
    return out


def _fields(spec: TaskSpec, splits: Sequence[ValidationSplit | TestSplit]) -> list[str]:
    if spec.schema_fields:
        return list(spec.schema_fields)
    docs = [v for split in splits for r in split.records for v in (r.gold, r.teacher, r.pred)]
    return resolve_fields(docs, [])


def _confidences(records: Sequence[EvalRecord]) -> FloatArray:
    return np.asarray([0.0 if r.confidence is None else float(r.confidence) for r in records], dtype=np.float64)


def _report_or_none(conf: FloatArray | None, correct: BoolArray, mask: BoolArray) -> dict[str, Any] | None:
    return calibration_report(conf[mask], correct[mask]) if conf is not None and mask.any() else None


def _correctness_label(refs: _Refs) -> str:
    if refs.has_gold.all():
        return "gold"
    if refs.has_gold.any():
        return "gold where present, else teacher"
    return "teacher"


@dataclass(frozen=True)
class _Isotonic:
    """Isotonic maps fitted on validation: one per reference, plus the primary map (gold where present) for the plot."""

    vs_gold: IsotonicCalibrator | None
    vs_teacher: IsotonicCalibrator | None
    primary: IsotonicCalibrator
    primary_reference: str

    def fit_json(self) -> dict[str, Any]:
        def entry(calibrator: IsotonicCalibrator | None) -> dict[str, Any] | None:
            return None if calibrator is None else {"n_fit": calibrator.n_fit, "map": calibrator.to_dict()}

        return {
            "split": "valid",
            "n_fit": self.primary.n_fit,
            "reference": self.primary_reference,
            "map": self.primary.to_dict(),
            "vs_gold": entry(self.vs_gold),
            "vs_teacher": entry(self.vs_teacher),
        }


def _fit_isotonic(valid: ValidationSplit, student: _System) -> _Isotonic:
    """One isotonic map per reference, each fitted on the validation records that have that reference."""
    refs = student.refs

    def fit(mask: BoolArray, correct: BoolArray) -> IsotonicCalibrator | None:
        if not mask.any():
            return None
        if mask.all():
            return fit_isotonic(valid, correct)
        subset = valid.replace_records([r for r, keep in zip(valid.records, mask, strict=True) if keep])
        return fit_isotonic(subset, correct[mask])

    vs_gold = fit(refs.has_gold, student.correct_gold)
    vs_teacher = fit(refs.has_teacher, student.correct_teacher)
    reference = _correctness_label(refs)
    if reference == "gold" and vs_gold is not None:
        primary = vs_gold
    elif reference == "teacher" and vs_teacher is not None:
        primary = vs_teacher
    else:
        primary = fit_isotonic(valid, student.primary_correct)
    return _Isotonic(vs_gold=vs_gold, vs_teacher=vs_teacher, primary=primary, primary_reference=reference)


def _calibration(conf: FloatArray, student: _System, isotonic: _Isotonic) -> dict[str, Any]:
    """Raw and isotonic calibration vs gold and vs the teacher, each isotonic map fitted on its own reference.

    An isotonic block is ``null`` when validation has none of that reference to fit on.
    """
    refs = student.refs
    by_gold = None if isotonic.vs_gold is None else isotonic.vs_gold.apply(conf)
    by_teacher = None if isotonic.vs_teacher is None else isotonic.vs_teacher.apply(conf)
    return {
        "correctness": _correctness_label(refs),
        "raw": {
            "vs_gold": _report_or_none(conf, student.correct_gold, refs.has_gold),
            "vs_teacher": _report_or_none(conf, student.correct_teacher, refs.has_teacher),
        },
        "isotonic": {
            "vs_gold": _report_or_none(by_gold, student.correct_gold, refs.has_gold),
            "vs_teacher": _report_or_none(by_teacher, student.correct_teacher, refs.has_teacher),
        },
        "isotonic_fit": isotonic.fit_json(),
    }


def _score_block(conf: FloatArray, system: _System) -> dict[str, Any]:
    refs = system.refs
    t_mask, g_mask = refs.has_teacher, refs.has_gold

    def area(mask: BoolArray, correct: BoolArray) -> float | None:
        return auroc(conf[mask], correct[mask]) if mask.any() else None

    def calib(mask: BoolArray, correct: BoolArray) -> float | None:
        return ece(conf[mask], correct[mask]) if mask.any() else None

    return {
        "n": refs.n,
        "mean_confidence": _mean(conf),
        "auroc_vs_teacher": area(t_mask, system.correct_teacher),
        "auroc_vs_gold": area(g_mask, system.correct_gold),
        "accuracy_vs_teacher": _mean(system.correct_teacher[t_mask]),
        "accuracy_vs_gold": _mean(system.correct_gold[g_mask]),
        "ece_vs_teacher": calib(t_mask, system.correct_teacher),
        "ece_vs_gold": calib(g_mask, system.correct_gold),
    }


def _confidence_block(split: ValidationSplit | TestSplit, refs: _Refs, reference: str) -> dict[str, Any]:
    """Primary against alternative confidence on one split (AUROC of each for its own answers' correctness)."""
    records = split.records
    primary = _score_block(_confidences(records), _system(refs, [r.pred for r in records]))
    alternative: dict[str, Any] | None = None
    alt_conf = [r.alt_confidence for r in records]
    if records and all(c is not None for c in alt_conf):
        conf = np.asarray([float(c) for c in alt_conf if c is not None], dtype=np.float64)
        alternative = _score_block(conf, _system(refs, [r.alt_pred for r in records]))
    better: str | None = None
    key = f"auroc_vs_{reference}"
    if alternative is not None and primary[key] is not None and alternative[key] is not None:
        a, b = float(primary[key]), float(alternative[key])
        better = "primary" if a > b else "alternative" if b > a else "tie"
    return {"primary": primary, "alternative": alternative, "chosen": "primary", "better_auroc": better}


def _bootstrap(
    systems: Mapping[str, _System],
    conf: FloatArray,
    correct: BoolArray,
    groups: Sequence[str] | None,
) -> dict[str, Any]:
    """Paired bootstrap (and the cluster bootstrap over groups) of every headline metric and difference."""
    stats: dict[str, Stat] = {}
    for name, system in systems.items():
        for metric in system.available():
            stats[f"{name}.{metric}"] = functools.partial(system.stat, metric)

    def calibration_stat(fn: Callable[[FloatArray, BoolArray], float | None]) -> Stat:
        def stat(idx: IntArray) -> float | None:
            return fn(conf[idx], correct[idx])

        return stat

    stats["student.ece"] = calibration_stat(ece)
    stats["student.auroc"] = calibration_stat(auroc)
    diffs: list[tuple[str, str]] = []
    for metric in systems["student"].available():
        for a, b in (("student", "teacher"), ("cascade", "teacher"), ("student", "tfidf")):
            if f"{a}.{metric}" in stats and f"{b}.{metric}" in stats:
                diffs.append((f"{a}.{metric}", f"{b}.{metric}"))
    n = systems["student"].refs.n
    paired = paired_bootstrap(n, stats, diffs, resamples=RESAMPLES, seed=BOOTSTRAP_SEED)
    cluster = None
    if groups is not None:
        cluster = paired_bootstrap(n, stats, diffs, resamples=RESAMPLES, seed=BOOTSTRAP_SEED, groups=groups)
    return {"paired": paired, "cluster": cluster}


def _breakdown_fn(
    records: Sequence[EvalRecord], systems: Mapping[str, _System]
) -> Callable[[Sequence[EvalRecord]], dict[str, Any]]:
    position = {id(r): i for i, r in enumerate(records)}

    def fn(group: Sequence[EvalRecord]) -> dict[str, Any]:
        idx = np.asarray([position[id(r)] for r in group], dtype=np.int64)
        out: dict[str, Any] = {}
        for name, system in systems.items():
            metrics = system.available()
            if name == "teacher":
                metrics = [m for m in metrics if m in GOLD_METRICS]
                if not metrics:
                    continue
            out[name] = {m: system.stat(m, idx) for m in metrics}
        return out

    return fn


def _per_field(systems: Mapping[str, _System]) -> dict[str, Any] | None:
    extraction = {name: s for name, s in systems.items() if isinstance(s, _ExtractionSystem)}
    if not extraction:
        return None
    first = next(iter(extraction.values()))
    vs_gold = {name: s.per_field("gold") for name, s in extraction.items()}
    vs_teacher = {name: s.per_field("teacher") for name, s in extraction.items() if name != "teacher"}
    return {
        "fields": list(first.refs.fields),
        "vs_gold": vs_gold if first.refs.has_gold.any() else None,
        "vs_teacher": vs_teacher,
    }


def _tfidf(
    spec: TaskSpec, valid: ValidationSplit, target: ValidationSplit | TestSplit, log: Log
) -> tuple[TfidfBaseline, list[str]] | None:
    """TF-IDF + logistic regression on the train split's teacher labels, ``C`` tuned on validation."""
    try:
        texts, labels = load_train_pairs(spec)
        baseline = tune_baseline(texts, [str(label) for label in labels], valid)
    except ValueError as exc:
        log(f"  tfidf baseline skipped: {exc}")
        return None
    return baseline, baseline.predict([r.input for r in target.records])


def _ci(bootstrap: Mapping[str, Any], name: str) -> str:
    entry = bootstrap.get("stats", {}).get(name)
    if not entry or entry.get("point") is None:
        return "n/a"
    lo, hi = entry.get("lo"), entry.get("hi")
    if lo is None or hi is None:
        return f"{entry['point']:.4f}"
    return f"{entry['point']:.4f} [{lo:.4f}, {hi:.4f}]"


def _append_test_access(task: str, run_id: str, command: str) -> int:
    path = paths.task_home(task) / TEST_ACCESS_LOG
    line = json.dumps({"date": _now(), "run_id": run_id, "command": command}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return count_test_scorings(task)


def count_test_scorings(task: str) -> int:
    """How many times the test split has been scored for ``task``."""
    path = paths.task_home(task) / TEST_ACCESS_LOG
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


# -- selection ---------------------------------------------------------------------------------------------------


def _select(
    spec: TaskSpec,
    valid: ValidationSplit,
    backends: _Backends,
    *,
    backend_name: str,
    alternatives: bool,
    log: Log,
) -> tuple[dict[str, Any], ValidationSplit, RunInfo]:
    """Choose the run on validation: best seed per base model, then the base-model rule.

    Returns the ``selected_run.json`` payload, the selected run's validation predictions and the run; the caller
    writes the payload (:func:`_commit_selection`) once the selected run's threshold is in place. Each candidate's
    backend is freed once its validation pass is done, so later candidates' latencies are not measured next to
    earlier models.
    """
    candidates = discover_candidates(spec, backend_name, log)
    if not candidates:
        raise EvalError(
            f"no candidate runs for task '{spec.task}': eval --select needs runs trained on teacher labels with an "
            "adapter under runs/ (train one with `taskdistill train`)"
        )
    metric, reference = spec.cascade.metric, spec.cascade.reference
    if reference == "gold" and not valid.has_gold:
        raise EvalError("cascade.reference is gold but the validation split has no gold labels")
    log(f"selecting among {len(candidates)} run(s) on validation ({metric} vs {reference})")
    labels = list(spec.labels) if spec.type == "classification" else None
    fields = _fields(spec, [valid]) if spec.type == "extraction" else None
    predicted: dict[str, ValidationSplit] = {}
    p95: dict[str, float] = {}
    by_run = {info.run_id: info for info in candidates}
    for info in candidates:
        log(f"  candidate {info.run_id} ({info.base_label}, seed {info.seed})")
        predicted[info.run_id] = _predict(
            spec, info, valid, backends.get(info), alternatives=alternatives, backend_name=backend_name, log=log
        )
        backends.drop(info.run_id)
        p95_value = _latency([r.latency_ms for r in predicted[info.run_id].records])["p95"]
        p95[info.run_id] = float(p95_value) if p95_value is not None else math.nan
    scores = score_runs(predicted, metric=metric, reference=reference, labels=labels, fields=fields)
    by_base: dict[str, list[str]] = {}
    for info in candidates:
        by_base.setdefault(info.base_label, []).append(info.run_id)
    best = {
        base: select_run(
            {rid: predicted[rid] for rid in run_ids}, metric=metric, reference=reference, labels=labels, fields=fields
        )
        for base, run_ids in by_base.items()
    }

    def size_key(base: str) -> tuple[float, float, str]:
        size = _base_size(base)
        return (math.inf if size is None else size, p95[best[base]], base)

    order = sorted(by_base, key=size_key)
    current = order[0]
    decisions: list[dict[str, Any]] = []
    for other in order[1:]:
        small_run, large_run = best[current], best[other]
        decision = choose_base_model(
            predicted[small_run],
            predicted[large_run],
            small_p95_ms=p95[small_run],
            large_p95_ms=p95[large_run],
            metric=metric,
            reference=reference,
            labels=labels,
            fields=fields,
        )
        decisions.append(
            {"small_base": current, "large_base": other, "small_run": small_run, "large_run": large_run, **decision}
        )
        log(f"  base rule, {other} vs {current}: {decision['reason']} -> {decision['choice']}")
        if decision["choice"] == "large":
            current = other
    selected = best[current]
    seeds = len(by_base[current])
    reason = f"{selected}: best validation {metric} ({scores[selected]:.4f}) among {seeds} run(s) of {current}; " + (
        "; ".join(f"{d['large_base']} vs {d['small_base']}: {d['reason']}" for d in decisions) or "one base model"
    )
    payload: dict[str, Any] = {
        "run_id": selected,
        "reason": reason,
        "date": _now(),
        **_run_stamp(by_run[selected]),
        "split": "valid",
        "n_valid": len(valid),
        "rule": {
            "metric": metric,
            "reference": reference,
            "seed_rule": SEED_RULE,
            "base_rule": BASE_RULE,
            "min_gain": MIN_GAIN,
            "max_latency_ratio": MAX_LATENCY_RATIO,
            "best_per_base": best,
            "decisions": decisions,
        },
        "candidates": {
            rid: {
                "valid_metric": scores[rid],
                "p95_ms": p95[rid],
                "base_model": by_run[rid].base_label,
                "seed": by_run[rid].seed,
                "profile": by_run[rid].profile,
            }
            for rid in sorted(predicted)
        },
    }
    log(f"selected {reason}")
    return payload, predicted[selected], by_run[selected]


# -- evaluation --------------------------------------------------------------------------------------------------


def _resolve_run(
    spec: TaskSpec,
    *,
    run_id: str | None,
    zero_shot: bool,
    backend: str,
    log: Log,
    selected: str | None = None,
    base: str | None = None,
) -> RunInfo:
    """The run to evaluate. ``selected`` is a selection made in this call (not yet in ``selected_run.json``).

    Zero-shot takes ``base`` when given, else the base of ``run_id``, of the selected run, or of the spec.
    """
    if zero_shot:
        if base:
            return zero_shot_run(spec, base, backend)
        if run_id:
            return zero_shot_run(spec, read_run(spec, run_id).base_model, backend)
        chosen_base = spec.student.base_model
        if selected is not None or selected_run_path(spec.task).is_file():
            try:
                chosen_base = read_run(spec, selected or read_selected_run(spec.task)["run_id"]).base_model
            except EvalError as exc:
                log(f"zero-shot: {exc}; using the spec's base model")
        return zero_shot_run(spec, chosen_base, backend)
    info = read_run(spec, run_id or selected or read_selected_run(spec.task)["run_id"])
    if info.backend is not None and info.backend != backend:
        raise EvalError(
            f"run '{info.run_id}' was trained with the {info.backend} backend; "
            f"evaluate it with --backend {info.backend}"
        )
    return info


def _evaluate(
    spec: TaskSpec,
    info: RunInfo,
    valid: ValidationSplit,
    backend: ScoringBackend,
    *,
    split: str,
    backend_name: str,
    alternatives: bool,
    fast: bool,
    command: str,
    selection: Mapping[str, Any] | None,
    log: Log,
    promote: bool = False,
) -> dict[str, Any]:
    """Evaluate ``info``; ``promote`` marks the run selected in this call (its threshold becomes the task's)."""
    task_type: TaskType = spec.type
    cascade = spec.cascade
    kind = "zero-shot base model" if info.zero_shot else f"{info.labels} labels, seed {info.seed}"
    log(f"eval {info.run_id} ({info.base_label}, {kind}) on {split}")
    if cascade.reference == "gold" and not valid.has_gold:
        raise EvalError("cascade.reference is gold but the validation split has no gold labels")
    test = load_split(spec, "test") if split == "test" else None  # data errors surface before any prediction
    if test is not None and not len(test):
        raise EvalError(f"the test split of task '{spec.task}' is empty")

    valid_pred = _predict(spec, info, valid, backend, alternatives=alternatives, backend_name=backend_name, log=log)
    target: ValidationSplit | TestSplit = valid_pred
    if test is not None:
        target = _predict(spec, info, test, backend, alternatives=alternatives, backend_name=backend_name, log=log)

    labels = list(spec.labels) if task_type == "classification" else []
    # Fields for the choices come from validation only; the reported split may add fields seen only there.
    valid_fields = _fields(spec, [valid_pred]) if task_type == "extraction" else []
    fields = _fields(spec, [valid_pred, target]) if task_type == "extraction" else []

    # Choices, on validation only: threshold (raw confidence) and the isotonic maps.
    threshold = _choose_threshold(spec, valid_pred, labels=labels, fields=valid_fields, log=log)
    valid_refs = _Refs.of(valid_pred.records, task_type, labels, valid_fields)
    isotonic = _fit_isotonic(valid_pred, _system(valid_refs, [r.pred for r in valid_pred.records]))

    # Scoring on the reported split.
    records = target.records
    refs = _Refs.of(records, task_type, labels, fields)
    student = _system(refs, [r.pred for r in records])
    teacher = _system(refs, refs.teacher)
    cascade_answers, escalated = apply_threshold(target, threshold.threshold)
    systems: dict[str, _System] = {
        "student": student,
        "teacher": teacher,
        "cascade": _system(refs, cascade_answers),
    }
    tfidf = None if task_type != "classification" or info.zero_shot else _tfidf(spec, valid_pred, target, log)
    if tfidf is not None:
        systems["tfidf"] = _system(refs, tfidf[1])
    conf = _confidences(records)
    correct = student.primary_correct

    systems_json: dict[str, Any] = {
        "student": {
            "metrics": student.metrics(),
            "latency_ms": _latency([r.latency_ms for r in records]),
            "calibration": _calibration(conf, student, isotonic),
        },
        "teacher": {"metrics": _teacher_metrics(teacher)},
        "cascade": {
            "metrics": systems["cascade"].metrics(),
            "threshold": threshold_to_json(threshold.threshold),
            "escalation_rate": float(escalated.mean()),
        },
    }
    if tfidf is not None:
        systems_json["tfidf"] = {"metrics": systems["tfidf"].metrics(), "C": tfidf[0].C, "table": tfidf[0].table}

    def operating_point(split_obj: ValidationSplit | TestSplit, point_fields: list[str]) -> dict[str, Any]:
        if not usable_records(split_obj.records, cascade.reference):
            _, escalated_here = apply_threshold(split_obj, threshold.threshold)
            reason = f"no {cascade.reference} labels on the {split_obj.name} split"
            log(f"operating point on {split_obj.name} not scored: {reason}")
            return {
                "split": split_obj.name,
                "available": False,
                "reason": reason,
                "threshold": threshold_to_json(threshold.threshold),
                "always_escalate": threshold.always_escalate,
                "n": 0,
                "escalation_rate": float(escalated_here.mean()) if len(split_obj) else None,
                "quality": None,
                "teacher_quality": None,
                "target_value": None,
                "met": None,
            }
        point = evaluate_operating_point(
            split_obj,
            threshold.threshold,
            reference=cascade.reference,
            metric=cascade.metric,
            target=cascade.target,
            max_drop=cascade.max_drop,
            labels=labels or None,
            fields=point_fields or None,
        )
        return {**point, "available": True}

    operating = {"valid": operating_point(valid_pred, valid_fields)}
    comparison: dict[str, Any] = {
        "reference": cascade.reference,
        "definitions": CONFIDENCE_DEFINITIONS[task_type],
        "valid": _confidence_block(valid_pred, valid_refs, cascade.reference),
    }
    if split == "test":
        operating["test"] = operating_point(target, fields)
        comparison["test"] = _confidence_block(target, refs, cascade.reference)

    has_groups = any(r.group is not None for r in records)
    groups = [r.group if r.group is not None else NO_GROUP for r in records] if has_groups else None
    bootstrap = _bootstrap(systems, conf, correct, groups)
    breakdown = _breakdown_fn(records, systems)
    per_group = per_group_scores(records, breakdown) if has_groups else None
    per_trait = per_trait_breakdown(records, breakdown) if any(r.traits for r in records) else None
    per_field = _per_field(systems) if task_type == "extraction" else None

    date = _now()
    scorings = (
        _append_test_access(spec.task, info.run_id, command) if split == "test" else count_test_scorings(spec.task)
    )

    out_dir = paths.eval_dir(spec.task) / info.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    reliability_png = plot_reliability(
        conf,
        correct,
        out_dir / f"reliability_{split}.png",
        calibrated=isotonic.primary.apply(conf),
        title=f"{info.run_id} on {split}: raw vs isotonic ({isotonic.primary_reference})",
    )
    curve_png = plot_threshold_curve(
        threshold.curve,
        threshold.point,
        out_dir / "threshold_curve.png",
        f"{info.run_id}: cascade on validation",
        target=threshold.target_value,
        quality_label=f"{cascade.metric} vs {cascade.reference}",
    )
    run_threshold, task_threshold = _write_thresholds(
        spec, info, threshold, date=date, log=log, task_level=True if promote else None
    )
    selected = task_threshold is not None

    eval_path = out_dir / f"eval_{split}.json"
    result: dict[str, Any] = {
        "task": spec.task,
        "task_type": task_type,
        "run_id": info.run_id,
        "split": split,
        "n": len(records),
        "n_valid": len(valid_pred),
        "date": date,
        "command": command,
        "hardware": hardware_info(),
        "run": info.describe(),
        "systems": systems_json,
        "threshold": threshold.to_dict(),
        "operating_point": operating,
        "confidence_comparison": comparison,
        "bootstrap": bootstrap,
        "per_group": per_group,
        "per_field": per_field,
        "per_trait": per_trait,
        "test_scorings": scorings,
        "profile_fast": fast,
        "selected": selected,
        "selection": selection,
        "files": {
            "eval": paths.relative_to_home(eval_path),
            "reliability": paths.relative_to_home(reliability_png),
            "threshold_curve": paths.relative_to_home(curve_png),
            "threshold": paths.relative_to_home(task_threshold or run_threshold),
            "run_threshold": paths.relative_to_home(run_threshold),
            "predictions": {
                name: paths.relative_to_home(info.run_dir / f"preds_{name}.jsonl")
                for name in dict.fromkeys(("valid", split))
            },
        },
    }
    result = _clean(result)
    _write_json(eval_path, result)
    _log_summary(result, bootstrap["paired"], log)
    log(f"wrote {paths.relative_to_home(eval_path)}")
    return result


def _log_summary(result: Mapping[str, Any], paired: Mapping[str, Any], log: Log) -> None:
    split, n = result["split"], result["n"]
    metric = "field_micro_f1" if result["task_type"] == "extraction" else "accuracy"
    for system in ("student", "cascade", "teacher", "tfidf"):
        parts = [
            f"{m} {_ci(paired, f'{system}.{m}')}" for m in (metric, "agreement") if f"{system}.{m}" in paired["stats"]
        ]
        if parts and system != "teacher":
            log(f"{split} (n={n}) {system}: " + ", ".join(parts))
        elif system == "teacher" and f"teacher.{metric}" in paired["stats"]:
            log(f"{split} (n={n}) teacher: {metric} {_ci(paired, f'teacher.{metric}')}")
    point = result["operating_point"].get(split)
    if point is not None and point.get("met") is not None:
        verdict = "met" if point["met"] else "NOT met"
        log(
            f"operating point on {split}: escalation {point['escalation_rate']:.1%}, quality {point['quality']:.4f} "
            f"(target {point['target_value']:.4f}: {verdict})"
        )


def select_on_validation(
    spec: TaskSpec,
    *,
    backend: str = "mlx",
    fast: bool = False,
    backend_factory: BackendFactory | None = None,
    log: Log = print,
) -> dict[str, Any]:
    """Only the run selection of ``eval --select``: returns (and writes) the ``selected_run.json`` payload.

    The selected run's threshold is chosen on the same validation predictions and written too, so ``threshold.json``
    always follows ``selected_run.json``.
    """
    valid = load_split(spec, "valid")
    if not len(valid):
        raise EvalError(f"the validation split of task '{spec.task}' is empty")
    backends = _Backends(backend_factory or _default_factory(backend))
    try:
        payload, chosen, selected = _select(spec, valid, backends, backend_name=backend, alternatives=not fast, log=log)
        _sync_selected_threshold(spec, selected, chosen, log)
        _commit_selection(spec, payload, log)
        return payload
    finally:
        backends.release()


def run_eval(
    spec: TaskSpec,
    *,
    run_id: str | None = None,
    split: str = "test",
    backend: str = "mlx",
    select: bool = False,
    zero_shot: bool = False,
    fast: bool = False,
    base: str | None = None,
    backend_factory: BackendFactory | None = None,
    log: Log = print,
    command: str | None = None,
) -> dict[str, Any]:
    """Evaluate one run and return the eval JSON (also written to ``eval/<run-id>/eval_<split>.json``).

    ``select`` first chooses the run on validation and writes ``selected_run.json`` (and the selected run's
    ``threshold.json``); ``selected_run.json`` is written only once the selected run's threshold is in place, and
    after its evaluation when the selected run is the one evaluated here. Without ``run_id`` the selected run is
    evaluated. ``zero_shot`` evaluates a base model with no adapter as the pseudo-run
    ``zero-shot-<short base>[-torch]``: ``base`` when given (a Hugging Face id or a local directory), else the base
    of ``run_id`` or of the selected run. ``fast`` (the quick profile) skips the alternative confidence scores; the
    zero-shot baseline is not part of it, so ``zero_shot`` with ``fast`` is refused.
    ``backend_factory(base_model, adapter_path)`` builds the backend, at most once per run and phase (selection
    frees each candidate's backend after its validation pass).
    """
    if split not in EVAL_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(EVAL_SPLITS)}")
    if zero_shot and fast:
        raise EvalError(
            "--zero-shot and --fast cannot be combined: the fast (quick) profile skips the zero-shot baseline; "
            "drop --fast to evaluate it"
        )
    if base is not None and not zero_shot:
        raise EvalError("a base model is given only with --zero-shot (a trained run is evaluated on its own base)")
    factory = backend_factory or _default_factory(backend)
    command = _portable_command(
        command
        or _default_command(
            spec.task,
            run_id=run_id,
            split=split,
            backend=backend,
            select=select,
            zero_shot=zero_shot,
            fast=fast,
            base=base,
        )
    )
    alternatives = not fast
    valid = load_split(spec, "valid")
    if not len(valid):
        raise EvalError(f"the validation split of task '{spec.task}' is empty; eval chooses the threshold on it")
    backends = _Backends(factory)
    try:
        selection: dict[str, Any] | None = None
        chosen: ValidationSplit | None = None
        selected: RunInfo | None = None
        if select:
            selection, chosen, selected = _select(
                spec, valid, backends, backend_name=backend, alternatives=alternatives, log=log
            )
        info = _resolve_run(
            spec,
            run_id=run_id,
            zero_shot=zero_shot,
            backend=backend,
            log=log,
            selected=None if selected is None else selected.run_id,
            base=base,
        )
        promote = selected is not None and selected.run_id == info.run_id
        if selection is not None and chosen is not None and selected is not None and not promote:
            # The selected run is not evaluated in this call (zero-shot, or another --run): its threshold first, then
            # the selection, both before the other run's evaluation reads selected_run.json.
            _sync_selected_threshold(spec, selected, chosen, log)
            _commit_selection(spec, selection, log)
        backends.release(keep=info.run_id)
        result = _evaluate(
            spec,
            info,
            valid,
            backends.get(info),
            split=split,
            backend_name=backend,
            alternatives=alternatives,
            fast=fast,
            command=command,
            selection=selection,
            log=log,
            promote=promote,
        )
        if promote and selection is not None:
            _commit_selection(spec, selection, log)  # after the selected run's threshold.json and eval JSON
        return result
    finally:
        backends.release()
