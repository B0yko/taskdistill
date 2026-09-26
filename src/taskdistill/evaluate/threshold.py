"""Cascade threshold: the student answers when its confidence is at least ``t``, the teacher answers otherwise.

The threshold is chosen on validation data only, over raw confidences. Candidates are the unique student confidences
plus "always escalate" (``math.inf``). The chosen ``t`` has the lowest escalation rate among candidates whose quality
meets the target (ties go to the higher ``t``). When none does, every request is escalated and a warning is set.

The quality curve is computed in one vectorised pass: each metric is a function of per-record additive counts, so
sorting records by confidence and taking cumulative sums gives the counts at every threshold at once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from taskdistill.evaluate.metrics import check_metric, document_counts, encode_labels, resolve_fields, score
from taskdistill.evaluate.splits import EvalRecord, TaskType, TestSplit, ValidationSplit, validation_only

Reference = Literal["teacher", "gold"]
ALWAYS_ESCALATE: Final = math.inf
EPS: Final = 1e-9

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
Records = ValidationSplit | TestSplit | Sequence[EvalRecord]


@dataclass(frozen=True)
class ThresholdResult:
    """The chosen operating point on validation and the full quality/escalation curve."""

    threshold: float
    escalation_rate: float
    quality: float
    target_value: float
    met: bool
    warning: str | None
    curve: list[tuple[float, float, float]] = field(repr=False)
    reference: str = "teacher"
    metric: str = "agreement"
    target: float | None = None
    max_drop: float | None = None
    n: int = 0
    teacher_quality: float = 0.0

    @property
    def always_escalate(self) -> bool:
        return math.isinf(self.threshold)

    @property
    def point(self) -> tuple[float, float, float]:
        """``(threshold, escalation_rate, quality)`` of the chosen point."""
        return (self.threshold, self.escalation_rate, self.quality)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form; the always-escalate threshold is written as ``null``."""
        return {
            "threshold": threshold_to_json(self.threshold),
            "always_escalate": self.always_escalate,
            "escalation_rate": self.escalation_rate,
            "quality": self.quality,
            "target_value": self.target_value,
            "met": self.met,
            "warning": self.warning,
            "reference": self.reference,
            "metric": self.metric,
            "target": self.target,
            "max_drop": self.max_drop,
            "n": self.n,
            "teacher_quality": self.teacher_quality,
            "curve": [
                {"threshold": threshold_to_json(t), "escalation_rate": r, "quality": q} for t, r, q in self.curve
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ThresholdResult:
        return cls(
            threshold=threshold_from_json(data["threshold"]),
            escalation_rate=float(data["escalation_rate"]),
            quality=float(data["quality"]),
            target_value=float(data["target_value"]),
            met=bool(data["met"]),
            warning=data.get("warning"),
            curve=[
                (threshold_from_json(p["threshold"]), float(p["escalation_rate"]), float(p["quality"]))
                for p in data.get("curve", [])
            ],
            reference=str(data.get("reference", "teacher")),
            metric=str(data.get("metric", "agreement")),
            target=data.get("target"),
            max_drop=data.get("max_drop"),
            n=int(data.get("n", 0)),
            teacher_quality=float(data.get("teacher_quality", 0.0)),
        )


def threshold_to_json(threshold: float) -> float | None:
    """``None`` for always-escalate (JSON has no infinity), the float otherwise."""
    return None if math.isinf(threshold) else float(threshold)


def threshold_from_json(value: float | None) -> float:
    """Inverse of :func:`threshold_to_json`."""
    return ALWAYS_ESCALATE if value is None else float(value)


# --- helpers ----------------------------------------------------------------------------------------------------


def _records(split: Records) -> Sequence[EvalRecord]:
    if isinstance(split, (ValidationSplit, TestSplit)):
        return split.records
    return split


def reference_value(record: EvalRecord, reference: Reference) -> Any:
    if reference == "teacher":
        return record.teacher
    if reference == "gold":
        return record.gold
    raise ValueError(f"reference must be 'teacher' or 'gold', got {reference!r}")


def check_reference(metric: str, reference: Reference) -> None:
    """Raise ValueError for an unknown reference, or for ``agreement`` scored against anything but the teacher."""
    if reference not in ("teacher", "gold"):
        raise ValueError(f"reference must be 'teacher' or 'gold', got {reference!r}")
    if metric == "agreement" and reference != "teacher":
        raise ValueError("metric 'agreement' is measured against the teacher; use reference='teacher'")


def usable_records(records: Sequence[EvalRecord], reference: Reference) -> list[EvalRecord]:
    """Records that have a reference value (others cannot be scored)."""
    return [r for r in records if reference_value(r, reference) is not None]


def _task_type(split: Records, task_type: TaskType | None) -> TaskType:
    own = split.task_type if isinstance(split, (ValidationSplit, TestSplit)) else None
    if task_type is None:
        if own is None:
            raise ValueError("task_type is required when passing plain records")
        return own
    if own is not None and own != task_type:
        raise ValueError(f"task_type {task_type!r} does not match the split's {own!r}")
    return task_type


def _check_target(target: float | None, max_drop: float | None, *, required: bool) -> None:
    if target is not None and max_drop is not None:
        raise ValueError("set either target or max_drop, not both")
    if required and target is None and max_drop is None:
        raise ValueError("set target or max_drop")
    if max_drop is not None and max_drop < 0:
        raise ValueError("max_drop must be non-negative")


def confidences(records: Sequence[EvalRecord]) -> FloatArray:
    """Student confidences with a missing or NaN confidence as ``-inf``: such a record always escalates."""
    conf = np.asarray([math.nan if r.confidence is None else r.confidence for r in records], dtype=np.float64)
    conf[np.isnan(conf)] = -math.inf
    return conf


def apply_threshold(split: Records, threshold: float) -> tuple[list[Any], BoolArray]:
    """Cascade answers and the escalation mask at ``threshold`` (reporting only; any split type).

    The student answers only when its confidence is at least the threshold; a missing or NaN confidence escalates.
    """
    if math.isnan(threshold):
        raise ValueError("threshold is NaN")
    records = _records(split)
    conf = confidences(records)
    escalated = np.isneginf(conf) | ~(conf >= threshold)
    answers = [r.teacher if esc else r.pred for r, esc in zip(records, escalated, strict=True)]
    return answers, escalated


class _AdditiveScorer:
    """A metric written as ``finalise(sum of per-record count vectors)``."""

    def __init__(
        self,
        task_type: TaskType,
        metric: str,
        refs: Sequence[Any],
        labels: Sequence[str] | None,
        fields: Sequence[str] | None,
    ) -> None:
        check_metric(task_type, metric)
        self.task_type = task_type
        self.metric = metric
        self.refs = list(refs)
        self.n = len(self.refs)
        self.labels = list(labels) if labels is not None else None
        self.fields = list(fields) if fields is not None else None
        if task_type == "classification" and metric == "macro_f1":
            if self.labels is None:
                raise ValueError("macro_f1 needs the label set")
            self.ref_codes = encode_labels(self.refs, self.labels)
            n_labels = len(self.labels)
            self.gold_counts = np.bincount(self.ref_codes, minlength=n_labels + 1)[:n_labels]

    def resolve_fields(self, *answer_sets: Sequence[Any]) -> None:
        if self.task_type == "extraction" and self.fields is None:
            preds = [doc for answers in answer_sets for doc in answers]
            self.fields = resolve_fields(self.refs, preds)

    def contributions(self, answers: Sequence[Any]) -> IntArray:
        n = self.n
        if self.task_type == "classification":
            if self.metric != "macro_f1":
                eq = [ref is not None and ref == ans for ref, ans in zip(self.refs, answers, strict=True)]
                return np.asarray(eq, dtype=np.int64).reshape(n, 1)
            assert self.labels is not None
            n_labels = len(self.labels)
            codes = encode_labels(answers, self.labels)
            out = np.zeros((n, 2 * (n_labels + 1)), dtype=np.int64)
            rows = np.arange(n)
            hit = codes == self.ref_codes
            out[rows[hit], self.ref_codes[hit]] = 1
            out[rows, n_labels + 1 + codes] = 1
            return out
        assert self.fields is not None
        counts = document_counts(self.refs, answers, self.fields)
        if self.metric in ("agreement", "field_f1"):
            return np.stack([counts.tp, counts.fp, counts.fn], axis=1)
        column = {
            "field_exact_match": counts.fields_equal,
            "doc_exact_match": counts.doc_exact,
            "json_validity": counts.valid,
        }[self.metric]
        return np.asarray(column, dtype=np.int64).reshape(n, 1)

    def finalise(self, totals: IntArray) -> FloatArray:
        n = self.n
        t = totals.astype(np.float64)
        if self.task_type == "classification":
            if self.metric != "macro_f1":
                return t[:, 0] / n
            assert self.labels is not None
            n_labels = len(self.labels)
            tp = t[:, :n_labels]
            denom = self.gold_counts[None, :] + t[:, n_labels + 1 : 2 * n_labels + 1]
            f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
            return np.asarray(f1.mean(axis=1), dtype=np.float64)
        if self.metric in ("agreement", "field_f1"):
            denom = 2.0 * t[:, 0] + t[:, 1] + t[:, 2]
            return np.divide(2.0 * t[:, 0], denom, out=np.zeros_like(denom), where=denom > 0)
        if self.metric == "field_exact_match":
            assert self.fields is not None
            return t[:, 0] / (n * len(self.fields))
        return t[:, 0] / n


def quality_curve(
    records: Sequence[EvalRecord],
    *,
    task_type: TaskType,
    reference: Reference,
    metric: str,
    labels: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Thresholds (ascending, ending with ``inf``), escalation rates and cascade quality at each.

    A record with a missing or NaN confidence escalates at every threshold, as in :func:`apply_threshold`.
    """
    check_reference(metric, reference)
    n = len(records)
    if n == 0:
        raise ValueError("no records with a reference value to select a threshold on")
    scorer = _AdditiveScorer(task_type, metric, [reference_value(r, reference) for r in records], labels, fields)
    preds = [r.pred for r in records]
    teacher = [r.teacher for r in records]
    scorer.resolve_fields(preds, teacher)
    conf = confidences(records)
    order = np.argsort(-conf, kind="stable")
    student = scorer.contributions(preds)[order]
    fallback = scorer.contributions(teacher)[order]
    dims = student.shape[1]
    cum_student = np.zeros((n + 1, dims), dtype=np.int64)
    cum_teacher = np.zeros((n + 1, dims), dtype=np.int64)
    np.cumsum(student, axis=0, out=cum_student[1:])
    np.cumsum(fallback, axis=0, out=cum_teacher[1:])
    thresholds = np.append(np.unique(conf[np.isfinite(conf)]), ALWAYS_ESCALATE)
    answered = n - np.searchsorted(np.sort(conf), thresholds, side="left")
    totals = cum_student[answered] + (cum_teacher[n] - cum_teacher[answered])
    rates = (n - answered).astype(np.float64) / n
    return thresholds.astype(np.float64), rates, scorer.finalise(totals)


def choose_index(thresholds: FloatArray, rates: FloatArray, quality: FloatArray, target_value: float) -> int | None:
    """Index of the lowest escalation rate meeting ``quality >= target_value`` (ties: higher threshold)."""
    meets = np.flatnonzero(quality >= target_value - EPS)
    if meets.size == 0:
        return None
    lowest = rates[meets].min()
    tied = meets[rates[meets] == lowest]
    return int(tied[np.argmax(thresholds[tied])])


@validation_only("val")
def select_threshold(
    val: ValidationSplit,
    *,
    reference: Reference,
    metric: str,
    target: float | None = None,
    max_drop: float | None = None,
    task_type: TaskType | None = None,
    labels: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
) -> ThresholdResult:
    """Choose the cascade threshold on the validation split.

    Quality is ``metric(reference values, cascade answers)``. With ``max_drop``, the target is the teacher's own
    score against the reference minus ``max_drop`` (for reference ``teacher`` and ``agreement``, ``1 - max_drop``).
    Records without a reference value are left out.
    """
    task = _task_type(val, task_type)
    check_metric(task, metric)
    check_reference(metric, reference)
    _check_target(target, max_drop, required=True)
    records = usable_records(val.records, reference)
    thresholds, rates, quality = quality_curve(
        records, task_type=task, reference=reference, metric=metric, labels=labels, fields=fields
    )
    teacher_quality = float(quality[-1])
    target_value = float(target) if target is not None else teacher_quality - float(max_drop or 0.0)
    index = choose_index(thresholds, rates, quality, target_value)
    warning: str | None = None
    met = index is not None
    if index is None:
        index = len(thresholds) - 1
        warning = (
            f"no threshold reaches {metric} >= {target_value:.4f} against the {reference} on validation "
            f"(best {float(quality.max()):.4f}); every request is escalated"
        )
    elif math.isinf(thresholds[index]):
        warning = f"only escalating every request reaches {metric} >= {target_value:.4f} against the {reference}"
    curve = [(float(t), float(r), float(q)) for t, r, q in zip(thresholds, rates, quality, strict=True)]
    return ThresholdResult(
        threshold=float(thresholds[index]),
        escalation_rate=float(rates[index]),
        quality=float(quality[index]),
        target_value=target_value,
        met=met,
        warning=warning,
        curve=curve,
        reference=reference,
        metric=metric,
        target=target,
        max_drop=max_drop,
        n=len(records),
        teacher_quality=teacher_quality,
    )


def evaluate_operating_point(
    split: ValidationSplit | TestSplit,
    threshold: float,
    *,
    reference: Reference,
    metric: str,
    target: float | None = None,
    max_drop: float | None = None,
    task_type: TaskType | None = None,
    labels: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Escalation rate, cascade quality and whether the target held, at a fixed threshold (reporting)."""
    task = _task_type(split, task_type)
    check_metric(task, metric)
    check_reference(metric, reference)
    _check_target(target, max_drop, required=False)
    records = usable_records(split.records, reference)
    if not records:
        raise ValueError("no records with a reference value")
    refs = [reference_value(r, reference) for r in records]
    answers, escalated = apply_threshold(records, threshold)
    teacher = [r.teacher for r in records]
    if task == "extraction" and fields is None:
        fields = resolve_fields(refs, [*answers, *[r.pred for r in records], *teacher])
    quality = score(task, metric, refs, answers, labels, fields)
    teacher_quality = score(task, metric, refs, teacher, labels, fields)
    target_value: float | None = None
    if target is not None:
        target_value = float(target)
    elif max_drop is not None:
        target_value = teacher_quality - float(max_drop)
    return {
        "split": split.name,
        "threshold": threshold_to_json(threshold),
        "always_escalate": math.isinf(threshold),
        "n": len(records),
        "escalation_rate": float(escalated.mean()),
        "quality": quality,
        "teacher_quality": teacher_quality,
        "target_value": target_value,
        "met": None if target_value is None else bool(quality >= target_value - EPS),
    }
