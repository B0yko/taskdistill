"""PNG plots for evaluation reports (Agg canvas, no pyplot state, no ``Software`` metadata)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy.typing as npt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from taskdistill.evaluate.calibration import DEFAULT_BINS, reliability_bins

DPI = 150


def new_figure(width: float = 6.0, height: float = 4.0) -> Figure:
    """A figure bound to an Agg canvas, independent of the global pyplot backend."""
    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def save_png(fig: Figure, path: str | Path) -> Path:
    """Write ``fig`` as PNG without the ``Software`` text chunk; parent directories are created."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format="png", dpi=DPI, metadata={"Software": None})
    return target


def plot_threshold_curve(
    curve: Sequence[tuple[float, float, float]],
    chosen: tuple[float, float, float] | None,
    path: str | Path,
    title: str,
    *,
    target: float | None = None,
    quality_label: str = "quality",
) -> Path:
    """Cascade quality against escalation rate, with the chosen ``(threshold, rate, quality)`` point marked."""
    fig = new_figure()
    ax = fig.add_subplot()
    points = sorted(curve, key=lambda p: p[1])
    ax.plot([p[1] for p in points], [p[2] for p in points], color="#1f5f99", linewidth=1.5, label="cascade")
    if target is not None:
        ax.axhline(target, color="#888888", linestyle="--", linewidth=1.0, label=f"target {target:.3f}")
    if chosen is not None:
        threshold, rate, quality = chosen
        name = "always escalate" if math.isinf(threshold) else f"t = {threshold:.3f}"
        ax.scatter([rate], [quality], color="#c0392b", zorder=3, label=f"chosen: {name}, {rate:.1%} escalated")
    ax.set_xlabel("escalation rate")
    ax.set_ylabel(quality_label)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize="small")
    return save_png(fig, path)


def plot_reliability(
    conf: npt.ArrayLike,
    correct: npt.ArrayLike,
    path: str | Path,
    calibrated: npt.ArrayLike | None = None,
    *,
    n_bins: int = DEFAULT_BINS,
    title: str = "Reliability diagram",
) -> Path:
    """Accuracy against mean confidence per bin for raw (and optionally calibrated) confidence."""
    fig = new_figure(5.0, 5.0)
    ax = fig.add_subplot()
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#888888", linestyle="--", linewidth=1.0, label="perfect calibration")
    series = [("raw", conf, "#1f5f99", "o")]
    if calibrated is not None:
        series.append(("isotonic", calibrated, "#c0392b", "s"))
    for name, values, colour, marker in series:
        bins = [b for b in reliability_bins(values, correct, n_bins) if b["n"]]
        ax.plot(
            [b["mean_confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            color=colour,
            marker=marker,
            linewidth=1.2,
            label=name,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize="small")
    return save_png(fig, path)


def plot_loss(
    train: Sequence[tuple[int, float]],
    val: Sequence[tuple[int, float]],
    path: str | Path,
    *,
    title: str = "Training loss",
    best_iteration: int | None = None,
) -> Path:
    """Train and validation loss against iteration; the kept checkpoint is marked when given."""
    fig = new_figure()
    ax = fig.add_subplot()
    if train:
        ax.plot([p[0] for p in train], [p[1] for p in train], color="#1f5f99", linewidth=1.2, label="train")
    if val:
        ax.plot(
            [p[0] for p in val], [p[1] for p in val], color="#c0392b", marker="o", linewidth=1.2, label="validation"
        )
    if best_iteration is not None:
        ax.axvline(best_iteration, color="#888888", linestyle="--", linewidth=1.0, label=f"kept: {best_iteration}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if train or val:
        ax.legend(loc="upper right", fontsize="small")
    return save_png(fig, path)
