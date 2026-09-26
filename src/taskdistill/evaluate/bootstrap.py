"""Paired and cluster bootstrap confidence intervals.

Every statistic is a function of an index array into the evaluation set. Within one resample, all statistics see the
same index draw, so a difference between two systems is paired. With ``groups``, whole groups are resampled with
replacement (cluster bootstrap) and their members concatenated.

The draw is fixed by the seed: resample ``r`` takes ``rng.integers(0, n, size=n)`` (paired) or
``rng.integers(0, k, size=k)`` over the ``k`` sorted group names (cluster), with ``rng = np.random.default_rng(seed)``
shared across resamples in order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

LEVEL: Final = 0.95
PERCENTILES: Final = (2.5, 97.5)

IntArray = npt.NDArray[np.int64]
Stat = Callable[[IntArray], float | None]


def diff_name(a: str, b: str) -> str:
    """The key under which the ``a - b`` difference is reported."""
    return f"{a}_minus_{b}"


def _as_float(value: float | None) -> float:
    if value is None:
        return math.nan
    return float(value)


def _cluster_members(groups: Sequence[Any], n: int) -> tuple[list[str], list[IntArray]]:
    if len(groups) != n:
        raise ValueError(f"groups has {len(groups)} entries for n={n}")
    labels = np.asarray([str(g) for g in groups], dtype=np.str_)
    names, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable").astype(np.int64)
    members = np.split(order, np.cumsum(counts)[:-1])
    return [str(name) for name in names], [np.asarray(m, dtype=np.int64) for m in members]


def _interval(samples: npt.NDArray[np.float64]) -> tuple[float | None, float | None, int]:
    valid = samples[~np.isnan(samples)]
    if valid.size == 0:
        return None, None, 0
    lo, hi = np.percentile(valid, PERCENTILES)
    return float(lo), float(hi), int(valid.size)


def paired_bootstrap(
    n: int,
    stats: Mapping[str, Stat],
    diffs: Sequence[tuple[str, str]] = (),
    resamples: int = 1000,
    seed: int = 0,
    groups: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """95% percentile intervals for each statistic and each ``(a, b)`` difference ``a - b``.

    A statistic may return ``None`` (undefined on a resample, e.g. AUROC with one class); such resamples are left out
    of that interval and ``valid_resamples`` says how many remained.
    """
    if n < 1:
        raise ValueError("the bootstrap needs at least one example")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not stats:
        raise ValueError("no statistics to bootstrap")
    names = list(stats)
    position = {name: j for j, name in enumerate(names)}
    for a, b in diffs:
        for name in (a, b):
            if name not in position:
                raise ValueError(f"difference refers to unknown statistic {name!r}")

    method = "paired" if groups is None else "cluster"
    group_names: list[str] = []
    members: list[IntArray] = []
    if groups is not None:
        group_names, members = _cluster_members(groups, n)

    everything = np.arange(n, dtype=np.int64)
    point = [_as_float(stats[name](everything)) for name in names]

    rng = np.random.default_rng(seed)
    samples = np.full((resamples, len(names)), np.nan, dtype=np.float64)
    for r in range(resamples):
        if groups is None:
            idx = rng.integers(0, n, size=n, dtype=np.int64)
        else:
            picks = rng.integers(0, len(members), size=len(members), dtype=np.int64)
            idx = np.concatenate([members[k] for k in picks])
        for j, name in enumerate(names):
            samples[r, j] = _as_float(stats[name](idx))

    common: dict[str, Any] = {"resamples": resamples, "seed": seed, "method": method}

    def entry(value: float, column: npt.NDArray[np.float64]) -> dict[str, Any]:
        lo, hi, valid = _interval(column)
        return {"point": None if math.isnan(value) else value, "lo": lo, "hi": hi, "valid_resamples": valid, **common}

    out_stats = {name: entry(point[j], samples[:, j]) for j, name in enumerate(names)}
    out_diffs: dict[str, dict[str, Any]] = {}
    for a, b in diffs:
        ja, jb = position[a], position[b]
        out_diffs[diff_name(a, b)] = {"a": a, "b": b, **entry(point[ja] - point[jb], samples[:, ja] - samples[:, jb])}
    return {
        **common,
        "level": LEVEL,
        "n": n,
        "n_groups": len(group_names) if groups is not None else None,
        "stats": out_stats,
        "diffs": out_diffs,
    }
