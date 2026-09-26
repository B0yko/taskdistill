"""Quality metrics for classification and extraction.

Classification values are label strings (``None`` or an unknown string is a wrong answer). Extraction values are
parsed JSON objects; anything that is not a mapping (``None`` included) is an invalid document.

Empty inputs score 0.0 rather than raising, so a report over an empty group stays serialisable.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from taskdistill.evaluate.splits import EvalRecord, TaskType

NUMBER_TOLERANCE: Final = 0.005
_FLOAT_SLACK: Final = 1e-9
NO_GROUP: Final = "(none)"

CLASSIFICATION_METRICS: Final = ("agreement", "accuracy", "macro_f1")
EXTRACTION_METRICS: Final = ("agreement", "field_f1", "field_exact_match", "doc_exact_match", "json_validity")

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_DATETIME = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+-]\d{2}(?::?\d{2})?)?)?"
)

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


def _check_aligned(a: Sequence[Any], b: Sequence[Any]) -> None:
    if len(a) != len(b):
        raise ValueError(f"references and predictions differ in length ({len(a)} != {len(b)})")


# --- classification ---------------------------------------------------------------------------------------------


def accuracy(gold: Sequence[Any], pred: Sequence[Any]) -> float:
    """Share of positions where the prediction equals the reference label."""
    _check_aligned(gold, pred)
    if not gold:
        return 0.0
    hits = sum(1 for g, p in zip(gold, pred, strict=True) if g is not None and g == p)
    return hits / len(gold)


def agreement(teacher: Sequence[Any], pred: Sequence[Any]) -> float:
    """Classification agreement: the label equality rate against the teacher."""
    return accuracy(teacher, pred)


def encode_labels(values: Sequence[Any], labels: Sequence[str]) -> IntArray:
    """Map each value to its index in ``labels``; ``None`` and unknown values map to ``len(labels)``."""
    index = _label_index(labels)
    unknown = len(labels)
    return np.fromiter(
        (index.get(v, unknown) if isinstance(v, str) else unknown for v in values), dtype=np.int64, count=len(values)
    )


def _label_index(labels: Sequence[str]) -> dict[str, int]:
    if not labels:
        raise ValueError("the label set is empty")
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError("the label set contains duplicates")
    return index


def macro_f1_from_codes(gold_codes: IntArray, pred_codes: IntArray, n_labels: int) -> float:
    """Macro-F1 over ``n_labels`` classes from codes produced by :func:`encode_labels` (``zero_division=0``)."""
    if gold_codes.shape != pred_codes.shape:
        raise ValueError("gold and predicted codes differ in shape")
    if n_labels <= 0:
        raise ValueError("n_labels must be positive")
    size = n_labels + 1
    tp = np.bincount(gold_codes[gold_codes == pred_codes], minlength=size)[:n_labels]
    denom = np.bincount(gold_codes, minlength=size)[:n_labels] + np.bincount(pred_codes, minlength=size)[:n_labels]
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(n_labels, dtype=np.float64), where=denom > 0)
    return float(f1.mean())


def macro_f1(gold: Sequence[Any], pred: Sequence[Any], labels: Sequence[str]) -> float:
    """Mean F1 over every label in ``labels`` (absent labels score 0), like sklearn with ``zero_division=0``."""
    _check_aligned(gold, pred)
    return macro_f1_from_codes(encode_labels(gold, labels), encode_labels(pred, labels), len(labels))


# --- extraction: value comparison -------------------------------------------------------------------------------


def normalise_string(text: str) -> str:
    """NFKC, trim, casefold and collapse whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def normalise_field_value(value: Any) -> Any:
    """Canonical form of one field value: strings normalised, numbers as float, containers recursively."""
    if value is None or isinstance(value, bool):
        return value
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        return normalise_string(value)
    if isinstance(value, Mapping):
        return {str(k): normalise_field_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise_field_value(v) for v in value]
    return value


def parse_iso_date(value: Any, *, strict: bool = True) -> date | None:
    """Parse an ISO date. Strict accepts only ``YYYY-MM-DD``; lenient also accepts ISO date-times."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        if _ISO_DATE.fullmatch(text):
            return date.fromisoformat(text)
        if not strict and _ISO_DATETIME.fullmatch(text):
            return datetime.fromisoformat(text).date()
    except ValueError:
        return None
    return None


def field_equal(gold: Any, pred: Any) -> bool:
    """Whether a predicted field value matches the reference value.

    Both null is a match. Numbers match within an absolute tolerance of 0.005. When the reference parses as
    ``YYYY-MM-DD`` and the prediction as an ISO date or date-time, the calendar dates are compared. Other strings are
    compared after :func:`normalise_string`. Values of different kinds never match.
    """
    if gold is None or pred is None:
        return gold is None and pred is None
    if isinstance(gold, bool) or isinstance(pred, bool):
        return isinstance(gold, bool) and isinstance(pred, bool) and gold == pred
    if _is_number(gold) or _is_number(pred):
        if not (_is_number(gold) and _is_number(pred)):
            return False
        return abs(float(gold) - float(pred)) <= NUMBER_TOLERANCE + _FLOAT_SLACK
    if isinstance(gold, str) or isinstance(pred, str):
        if not (isinstance(gold, str) and isinstance(pred, str)):
            return False
        gold_date = parse_iso_date(gold, strict=True)
        if gold_date is not None:
            pred_date = parse_iso_date(pred, strict=False)
            if pred_date is not None:
                return gold_date == pred_date
        return normalise_string(gold) == normalise_string(pred)
    if isinstance(gold, Mapping) and isinstance(pred, Mapping):
        keys = list(dict.fromkeys([*gold.keys(), *pred.keys()]))
        return all(field_equal(gold.get(k), pred.get(k)) for k in keys)
    if isinstance(gold, (list, tuple)) and isinstance(pred, (list, tuple)):
        return len(gold) == len(pred) and all(field_equal(g, p) for g, p in zip(gold, pred, strict=True))
    if isinstance(gold, (Mapping, list, tuple)) or isinstance(pred, (Mapping, list, tuple)):
        return False
    return bool(gold == pred)


# --- extraction: document scores --------------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentCounts:
    """Per-document counts behind the extraction metrics (arrays of length n)."""

    fields: tuple[str, ...]
    tp: IntArray
    fp: IntArray
    fn: IntArray
    valid: BoolArray
    matches: BoolArray  # shape (n, n_fields); an invalid document is all False

    @property
    def n(self) -> int:
        return int(self.valid.shape[0])

    @property
    def fields_equal(self) -> IntArray:
        return self.matches.sum(axis=1).astype(np.int64)

    @property
    def doc_exact(self) -> BoolArray:
        return self.matches.all(axis=1) & self.valid


def resolve_fields(refs: Sequence[Any], preds: Sequence[Any]) -> list[str]:
    """Field names in first-appearance order across the reference and predicted documents."""
    names: dict[str, None] = {}
    for doc in (*refs, *preds):
        if isinstance(doc, Mapping):
            names.update(dict.fromkeys(str(k) for k in doc))
    return list(names)


def document_counts(golds: Sequence[Any], preds: Sequence[Any], fields: Sequence[str]) -> DocumentCounts:
    """Count TP/FP/FN and field matches per document.

    Counting per field: reference non-null and prediction equal -> TP; prediction non-null and not equal -> FP, plus
    an FN when the reference is non-null; reference non-null and prediction null -> FN; both null -> not counted.
    An invalid prediction counts every non-null reference field as FN. A reference that is not a mapping is treated as
    all fields null.
    """
    _check_aligned(golds, preds)
    if not fields:
        raise ValueError("extraction metrics need at least one field")
    n, n_fields = len(golds), len(fields)
    tp = np.zeros(n, dtype=np.int64)
    fp = np.zeros(n, dtype=np.int64)
    fn = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=np.bool_)
    equal = np.zeros((n, n_fields), dtype=np.bool_)
    for i, (gold_doc, pred_doc) in enumerate(zip(golds, preds, strict=True)):
        gold_map: Mapping[str, Any] = gold_doc if isinstance(gold_doc, Mapping) else {}
        if not isinstance(pred_doc, Mapping):
            fn[i] = sum(1 for f in fields if gold_map.get(f) is not None)
            continue
        valid[i] = True
        for j, name in enumerate(fields):
            g, p = gold_map.get(name), pred_doc.get(name)
            same = field_equal(g, p)
            equal[i, j] = same
            if p is not None:
                if same:
                    tp[i] += 1
                else:
                    fp[i] += 1
                    if g is not None:
                        fn[i] += 1
            elif g is not None:
                fn[i] += 1
    return DocumentCounts(tuple(fields), tp, fp, fn, valid, equal)


def micro_f1(tp: int, fp: int, fn: int) -> float:
    """``2TP / (2TP + FP + FN)``, 0.0 when nothing was counted."""
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def extraction_scores(golds: Sequence[Any], preds: Sequence[Any], fields: Sequence[str]) -> dict[str, Any]:
    """JSON validity, field micro-F1 (with precision/recall), field and document exact match, per-field exact."""
    counts = document_counts(golds, preds, fields)
    n = counts.n
    tp, fp, fn = int(counts.tp.sum()), int(counts.fp.sum()), int(counts.fn.sum())
    per_field = counts.matches.mean(axis=0) if n else np.zeros(len(fields))
    return {
        "n": n,
        "json_validity": float(counts.valid.mean()) if n else 0.0,
        "field_micro_f1": micro_f1(tp, fp, fn),
        "field_precision": tp / (tp + fp) if tp + fp else 0.0,
        "field_recall": tp / (tp + fn) if tp + fn else 0.0,
        "field_exact_match": float(counts.matches.mean()) if n else 0.0,
        "doc_exact_match": float(counts.doc_exact.mean()) if n else 0.0,
        "per_field_exact": {name: float(per_field[j]) for j, name in enumerate(counts.fields)},
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def extraction_agreement(teacher: Sequence[Any], preds: Sequence[Any], fields: Sequence[str]) -> float:
    """Extraction agreement: field micro-F1 against the teacher's output."""
    counts = document_counts(teacher, preds, fields)
    return micro_f1(int(counts.tp.sum()), int(counts.fp.sum()), int(counts.fn.sum()))


# --- dispatch ---------------------------------------------------------------------------------------------------


def check_metric(task_type: TaskType, metric: str) -> None:
    """Raise ValueError unless ``metric`` applies to ``task_type``."""
    if task_type not in ("classification", "extraction"):
        raise ValueError(f"unknown task type {task_type!r}")
    allowed = CLASSIFICATION_METRICS if task_type == "classification" else EXTRACTION_METRICS
    if metric not in allowed:
        raise ValueError(f"metric {metric!r} does not apply to {task_type}; expected one of {', '.join(allowed)}")


def score(
    task_type: TaskType,
    metric: str,
    refs: Sequence[Any],
    preds: Sequence[Any],
    labels: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
) -> float:
    """One named metric of ``preds`` against ``refs`` (teacher outputs or gold, as the caller chooses).

    Classification: ``agreement``/``accuracy`` (label equality) and ``macro_f1`` (needs ``labels``). Extraction:
    ``agreement``/``field_f1`` (field micro-F1), ``field_exact_match``, ``doc_exact_match``, ``json_validity``.
    Without ``fields``, extraction uses the keys seen in the documents.
    """
    check_metric(task_type, metric)
    _check_aligned(refs, preds)
    if task_type == "classification":
        if metric == "macro_f1":
            if labels is None:
                raise ValueError("macro_f1 needs the label set")
            return macro_f1(refs, preds, labels)
        return accuracy(refs, preds)
    names = list(fields) if fields is not None else resolve_fields(refs, preds)
    result = extraction_scores(refs, preds, names)
    key = {"agreement": "field_micro_f1", "field_f1": "field_micro_f1"}.get(metric, metric)
    return float(result[key])


# --- breakdowns -------------------------------------------------------------------------------------------------

GroupFn = Callable[[Sequence[EvalRecord]], float | Mapping[str, Any]]


def _summarise(records: Sequence[EvalRecord], fn: GroupFn) -> dict[str, Any]:
    value = fn(records)
    if isinstance(value, Mapping):
        return {"n": len(records), **value}
    number = float(value)
    return {"n": len(records), "score": None if math.isnan(number) else number}


def per_group_scores(records: Sequence[EvalRecord], fn: GroupFn) -> dict[str, dict[str, Any]]:
    """Apply ``fn`` to the records of each group (``record.group``; missing groups under ``"(none)"``), sorted."""
    buckets: dict[str, list[EvalRecord]] = {}
    for record in records:
        buckets.setdefault(record.group if record.group is not None else NO_GROUP, []).append(record)
    return {name: _summarise(buckets[name], fn) for name in sorted(buckets)}


def per_trait_breakdown(records: Sequence[EvalRecord], fn: GroupFn) -> dict[str, dict[str, Any]]:
    """Apply ``fn`` to the records carrying each trait; records without traits go under ``"(none)"``."""
    buckets: dict[str, list[EvalRecord]] = {}
    for record in records:
        traits = list(dict.fromkeys(record.traits)) or [NO_GROUP]
        for trait in traits:
            buckets.setdefault(trait, []).append(record)
    return {name: _summarise(buckets[name], fn) for name in sorted(buckets)}
