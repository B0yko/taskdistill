"""Calibration of student confidence: ECE, Brier score, AUROC and isotonic recalibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from taskdistill.evaluate.splits import EvalRecord, ValidationSplit, validation_only

DEFAULT_BINS: Final = 15
_RANGE_SLACK: Final = 1e-6

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Correctness = Sequence[bool] | npt.NDArray[np.bool_] | Callable[[EvalRecord], bool]


def as_arrays(conf: npt.ArrayLike, correct: npt.ArrayLike) -> tuple[FloatArray, BoolArray]:
    """Validate and flatten confidences (in [0, 1]) and correctness flags of equal length."""
    c = np.asarray(conf, dtype=np.float64).reshape(-1)
    y = np.asarray(correct, dtype=np.bool_).reshape(-1)
    if c.shape != y.shape:
        raise ValueError(f"confidence and correctness differ in length ({c.size} != {y.size})")
    if np.isnan(c).any():
        raise ValueError("confidence contains NaN")
    if c.size and (c.min() < -_RANGE_SLACK or c.max() > 1.0 + _RANGE_SLACK):
        raise ValueError("confidence must lie in [0, 1]")
    return np.clip(c, 0.0, 1.0), y


def bin_index(conf: FloatArray, n_bins: int = DEFAULT_BINS) -> npt.NDArray[np.int64]:
    """Equal-width bin of each confidence; bins are left-closed and the last one includes 1.0."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    return np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)


def reliability_bins(conf: npt.ArrayLike, correct: npt.ArrayLike, n_bins: int = DEFAULT_BINS) -> list[dict[str, Any]]:
    """Per-bin count, mean confidence and accuracy (``None`` for empty bins)."""
    c, y = as_arrays(conf, correct)
    idx = bin_index(c, n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=c, minlength=n_bins)
    hit_sum = np.bincount(idx, weights=y.astype(np.float64), minlength=n_bins)
    bins: list[dict[str, Any]] = []
    for b in range(n_bins):
        n = int(counts[b])
        bins.append(
            {
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "n": n,
                "mean_confidence": float(conf_sum[b] / n) if n else None,
                "accuracy": float(hit_sum[b] / n) if n else None,
            }
        )
    return bins


def ece(conf: npt.ArrayLike, correct: npt.ArrayLike, n_bins: int = DEFAULT_BINS) -> float:
    """Expected calibration error: bin-count-weighted mean of |accuracy - mean confidence| over equal-width bins."""
    c, y = as_arrays(conf, correct)
    if c.size == 0:
        raise ValueError("ECE of an empty set is undefined")
    idx = bin_index(c, n_bins)
    conf_sum = np.bincount(idx, weights=c, minlength=n_bins)
    hit_sum = np.bincount(idx, weights=y.astype(np.float64), minlength=n_bins)
    return float(np.abs(hit_sum - conf_sum).sum() / c.size)


def brier(conf: npt.ArrayLike, correct: npt.ArrayLike) -> float:
    """Mean squared difference between confidence and correctness."""
    c, y = as_arrays(conf, correct)
    if c.size == 0:
        raise ValueError("the Brier score of an empty set is undefined")
    return float(np.mean((c - y.astype(np.float64)) ** 2))


def auroc(conf: npt.ArrayLike, correct: npt.ArrayLike) -> float | None:
    """Area under the ROC curve of confidence for correctness; ``None`` when either class is empty."""
    from sklearn.metrics import roc_auc_score

    c, y = as_arrays(conf, correct)
    if c.size == 0 or bool(y.all()) or not bool(y.any()):
        return None
    return float(roc_auc_score(y, c))


def calibration_report(conf: npt.ArrayLike, correct: npt.ArrayLike, n_bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """ECE, Brier, AUROC, counts and the reliability bins in one JSON-ready dict."""
    c, y = as_arrays(conf, correct)
    n = int(c.size)
    return {
        "n": n,
        "n_correct": int(y.sum()),
        "accuracy": float(y.mean()) if n else None,
        "mean_confidence": float(c.mean()) if n else None,
        "ece": ece(c, y, n_bins) if n else None,
        "brier": brier(c, y) if n else None,
        "auroc": auroc(c, y),
        "n_bins": n_bins,
        "bins": reliability_bins(c, y, n_bins),
    }


@dataclass(frozen=True)
class IsotonicCalibrator:
    """A monotone map from raw to calibrated confidence, clipped outside the fitted range."""

    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]
    n_fit: int

    def apply(self, conf: npt.ArrayLike) -> FloatArray:
        """Calibrated confidences for raw confidences ``conf``."""
        c = np.asarray(conf, dtype=np.float64)
        xp = np.asarray(self.x_thresholds, dtype=np.float64)
        fp = np.asarray(self.y_thresholds, dtype=np.float64)
        return np.asarray(np.interp(c, xp, fp), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"x_thresholds": list(self.x_thresholds), "y_thresholds": list(self.y_thresholds), "n_fit": self.n_fit}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IsotonicCalibrator:
        return cls(
            tuple(float(x) for x in data["x_thresholds"]),
            tuple(float(y) for y in data["y_thresholds"]),
            int(data["n_fit"]),
        )


def _correct_flags(records: Sequence[EvalRecord], correctness: Correctness) -> BoolArray:
    if callable(correctness):
        return np.fromiter((bool(correctness(r)) for r in records), dtype=np.bool_, count=len(records))
    flags = np.asarray(correctness, dtype=np.bool_).reshape(-1)
    if flags.size != len(records):
        raise ValueError(f"correctness has {flags.size} entries for {len(records)} records")
    return flags


@validation_only("val")
def fit_isotonic(val: ValidationSplit, correctness: Correctness) -> IsotonicCalibrator:
    """Fit isotonic regression of correctness on raw confidence over the validation split.

    ``correctness`` is a flag per record (aligned with ``val.records``) or a function of the record.
    """
    from sklearn.isotonic import IsotonicRegression

    records = val.records
    if not records:
        raise ValueError("cannot fit isotonic calibration on an empty split")
    missing = [r.id for r in records if r.confidence is None]
    if missing:
        raise ValueError(f"{len(missing)} record(s) have no confidence, e.g. {missing[0]!r}")
    conf, flags = as_arrays(
        [r.confidence for r in records if r.confidence is not None], _correct_flags(records, correctness)
    )
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(conf, flags.astype(np.float64))
    return IsotonicCalibrator(
        tuple(float(x) for x in model.X_thresholds_),
        tuple(float(y) for y in model.y_thresholds_),
        len(records),
    )
