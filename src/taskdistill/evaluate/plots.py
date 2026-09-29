"""PNG plots for evaluation reports (Agg canvas, no pyplot state, no ``Software`` metadata)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from taskdistill.evaluate.calibration import DEFAULT_BINS, reliability_bins

DPI = 150

# One restrained palette for every chart: ink and muted text, a light grid, one accent and one contrast colour.
INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
ACCENT = "#ea7a22"  # the student (orange in the project's palette)
CONTRAST = "#5b6bbf"  # the teacher / the second series (periwinkle)
REFERENCE = "#94a3b8"


#: Percent ticks without trailing zeros: 97.5%, 98%, 100%.
PERCENT = FuncFormatter(lambda value, _: f"{round(value * 100, 2):g}%")


def new_figure(width: float = 6.4, height: float = 4.0) -> Figure:
    """A figure bound to an Agg canvas, independent of the global pyplot backend."""
    fig = Figure(figsize=(width, height), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _style(ax: Axes, title: str, subtitle: str | None, xlabel: str, ylabel: str) -> None:
    """Left-aligned title and subtitle, no top/right spines, a faint grid behind the data."""
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0, pad=6)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


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
    subtitle: str | None = None,
) -> Path:
    """Cascade quality against escalation rate, with the chosen ``(threshold, rate, quality)`` point marked."""
    fig = new_figure()
    ax = fig.add_subplot()
    points = sorted(curve, key=lambda p: p[1])
    xs, ys = [p[1] for p in points], [p[2] for p in points]
    ax.plot(xs, ys, color=ACCENT, linewidth=1.8, drawstyle="steps-post", solid_joinstyle="miter")
    if xs:
        ax.fill_between(xs, ys, min(ys), step="post", color=ACCENT, alpha=0.07, linewidth=0)
    if target is not None:
        ax.axhline(target, color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.0)
        ax.annotate(
            f"target {target:.1%}", xy=(1.0, target), xycoords=("axes fraction", "data"), xytext=(0, 4),
            textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=MUTED,
        )  # fmt: skip
    if chosen is not None:
        threshold, rate, quality = chosen
        name = "always escalate" if math.isinf(threshold) else f"t = {threshold:.3f}"
        ax.scatter([rate], [quality], s=46, color=CONTRAST, edgecolor="white", linewidth=1.5, zorder=4)
        ax.annotate(
            f"{name}\n{rate:.1%} escalated, {quality:.1%}", xy=(rate, quality), xytext=(14, -26),
            textcoords="offset points", fontsize=8.5, color=INK, va="top",
            arrowprops={"arrowstyle": "-", "color": REFERENCE, "linewidth": 0.8},
        )  # fmt: skip
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(PERCENT)
    ax.yaxis.set_major_formatter(PERCENT)
    _style(ax, title, subtitle, "escalation rate (share of requests sent to the teacher)", quality_label)
    return save_png(fig, path)


def plot_reliability(
    conf: npt.ArrayLike,
    correct: npt.ArrayLike,
    path: str | Path,
    calibrated: npt.ArrayLike | None = None,
    *,
    n_bins: int = DEFAULT_BINS,
    title: str = "Reliability diagram",
    subtitle: str | None = None,
) -> Path:
    """Accuracy against mean confidence per bin for raw (and optionally calibrated) confidence."""
    series = [("raw", reliability_bins(conf, correct, n_bins))]
    if calibrated is not None:
        series.append(("isotonic", reliability_bins(calibrated, correct, n_bins)))
    return plot_reliability_bins(series, path, title=title, subtitle=subtitle)


def plot_reliability_bins(
    series: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    path: str | Path,
    *,
    title: str = "Reliability diagram",
    subtitle: str | None = None,
) -> Path:
    """Reliability diagram from precomputed bins (``n``, ``mean_confidence``, ``accuracy``); marker area ~ ``n``."""
    fig = new_figure(5.2, 5.0)
    ax = fig.add_subplot()
    ax.plot([0.0, 1.0], [0.0, 1.0], color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.0, label="perfect calibration")
    colours = (ACCENT, CONTRAST)
    largest = max((b["n"] for _, bins in series for b in bins if b.get("n")), default=1)
    for i, (name, bins) in enumerate(series):
        kept = [b for b in bins if b.get("n")]
        colour = colours[i % len(colours)]
        xs, ys = [b["mean_confidence"] for b in kept], [b["accuracy"] for b in kept]
        ax.plot(xs, ys, color=colour, linewidth=1.4, label=name)
        sizes = [12 + 110 * math.sqrt(b["n"] / largest) for b in kept]
        ax.scatter(xs, ys, s=sizes, color=colour, edgecolor="white", linewidth=1.0, zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PERCENT)
    ax.yaxis.set_major_formatter(PERCENT)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK)
    _style(ax, title, subtitle, "mean confidence in bin (marker area: examples in bin)", "accuracy in bin")
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
        ax.plot([p[0] for p in train], [p[1] for p in train], color=ACCENT, linewidth=1.2, alpha=0.8, label="train")
    if val:
        ax.plot(
            [p[0] for p in val], [p[1] for p in val], color=CONTRAST, marker="o", markersize=3.5, linewidth=1.4,
            label="validation",
        )  # fmt: skip
    if best_iteration is not None:
        ax.axvline(
            best_iteration, color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.0, label=f"kept: {best_iteration}"
        )
    if train or val:
        ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK)
    _style(ax, title, None, "iteration", "loss")
    return save_png(fig, path)
