"""Choices made on validation data only: checkpoint, run and base model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from taskdistill.evaluate.metrics import score
from taskdistill.evaluate.splits import ValidationSplit, validation_only
from taskdistill.evaluate.threshold import Reference, check_reference, reference_value, usable_records

MIN_GAIN: Final = 0.01
MAX_LATENCY_RATIO: Final = 3.0
_TIE: Final = 1e-12


def split_score(
    split: ValidationSplit,
    *,
    metric: str,
    reference: Reference = "teacher",
    labels: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
) -> float:
    """``metric`` of the split's predictions against its reference values (records without one are left out).

    ``agreement`` is always measured against the teacher.
    """
    check_reference(metric, reference)
    records = usable_records(split.records, reference)
    if not records:
        raise ValueError(f"no records with a {reference} reference to score")
    refs = [reference_value(r, reference) for r in records]
    return score(split.task_type, metric, refs, [r.pred for r in records], labels, fields)


@validation_only("val")
def select_checkpoint(val: ValidationSplit, history: Sequence[tuple[int, float]]) -> int:
    """Iteration with the lowest validation loss (ties go to the earlier iteration; NaN losses are ignored).

    ``history`` holds ``(iteration, validation loss)`` pairs measured on ``val``.
    """
    finite = [(int(it), float(loss)) for it, loss in history if not math.isnan(float(loss))]
    if not finite:
        raise ValueError("no validation losses to choose a checkpoint from")
    best_iteration, _ = min(finite, key=lambda item: (item[1], item[0]))
    return best_iteration


@validation_only("candidates")
def score_runs(
    candidates: Mapping[str, ValidationSplit],
    *,
    metric: str,
    reference: Reference = "teacher",
    labels: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
) -> dict[str, float]:
    """Validation ``metric`` per run id, in sorted id order."""
    return {
        run_id: split_score(candidates[run_id], metric=metric, reference=reference, labels=labels, fields=fields)
        for run_id in sorted(candidates)
    }


@validation_only("candidates")
def select_run(
    candidates: Mapping[str, ValidationSplit],
    *,
    metric: str,
    reference: Reference = "teacher",
    labels: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
) -> str:
    """Run id with the best validation ``metric`` (ties go to the lexicographically smallest id)."""
    scores = score_runs(candidates, metric=metric, reference=reference, labels=labels, fields=fields)
    best = max(scores.values())
    return min(run_id for run_id, value in scores.items() if value >= best - _TIE)


@validation_only("small", "large")
def choose_base_model(
    small: ValidationSplit,
    large: ValidationSplit,
    *,
    small_p95_ms: float,
    large_p95_ms: float,
    metric: str,
    reference: Reference = "teacher",
    labels: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
    min_gain: float = MIN_GAIN,
    max_latency_ratio: float = MAX_LATENCY_RATIO,
) -> dict[str, Any]:
    """Pick the base model: the large one only if it gains at least ``min_gain`` (1 point) on the validation target
    metric and its student p95 stays under ``max_latency_ratio`` (3x) the small one's.
    """
    if sorted(r.id for r in small.records) != sorted(r.id for r in large.records):
        raise ValueError("both base models must be scored on the same validation records")
    if small_p95_ms <= 0 or large_p95_ms <= 0:
        raise ValueError("p95 latencies must be positive")
    small_value = split_score(small, metric=metric, reference=reference, labels=labels, fields=fields)
    large_value = split_score(large, metric=metric, reference=reference, labels=labels, fields=fields)
    gain = large_value - small_value
    ratio = large_p95_ms / small_p95_ms
    gains_enough = gain >= min_gain - 1e-9  # 0.57 - 0.56 is 0.00999... in binary floating point
    fast_enough = ratio < max_latency_ratio
    if gains_enough and fast_enough:
        choice = "large"
        reason = (
            f"large gains {gain * 100:.2f} points (>= {min_gain * 100:.2f}) at {ratio:.2f}x the small p95 "
            f"(< {max_latency_ratio:g}x)"
        )
    elif not gains_enough:
        choice = "small"
        reason = f"large gains {gain * 100:.2f} points, below the {min_gain * 100:.2f}-point minimum"
    else:
        choice = "small"
        reason = (
            f"large gains {gain * 100:.2f} points but its p95 is {ratio:.2f}x the small p95 "
            f"(needs < {max_latency_ratio:g}x)"
        )
    return {
        "choice": choice,
        "reason": reason,
        "metric": metric,
        "reference": reference,
        "small": small_value,
        "large": large_value,
        "gain": gain,
        "small_p95_ms": float(small_p95_ms),
        "large_p95_ms": float(large_p95_ms),
        "latency_ratio": ratio,
        "min_gain": min_gain,
        "max_latency_ratio": max_latency_ratio,
        "n": len(small.records),
    }
