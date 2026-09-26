from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from taskdistill.evaluate.bootstrap import diff_name, paired_bootstrap
from taskdistill.evaluate.calibration import auroc

IntArray = npt.NDArray[np.int64]


def _mean_of(values: npt.NDArray[np.float64]) -> Callable[[IntArray], float]:
    return lambda idx: float(values[idx].mean())


def _systems(n: int = 200, seed: int = 1) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    student = (rng.random(n) < 0.8).astype(np.float64)
    teacher = (rng.random(n) < 0.9).astype(np.float64)
    return student, teacher


def test_same_seed_gives_identical_results() -> None:
    student, teacher = _systems()
    stats = {"student": _mean_of(student), "teacher": _mean_of(teacher)}
    first = paired_bootstrap(len(student), stats, [("student", "teacher")], resamples=300, seed=7)
    second = paired_bootstrap(len(student), stats, [("student", "teacher")], resamples=300, seed=7)
    assert first == second
    other = paired_bootstrap(len(student), stats, [("student", "teacher")], resamples=300, seed=8)
    assert other["stats"]["student"]["lo"] != first["stats"]["student"]["lo"] or (
        other["stats"]["student"]["hi"] != first["stats"]["student"]["hi"]
    )
    json.dumps(first, allow_nan=False)


def test_interval_contains_the_point_and_reports_metadata() -> None:
    student, teacher = _systems()
    result = paired_bootstrap(
        len(student),
        {"student": _mean_of(student), "teacher": _mean_of(teacher)},
        [("student", "teacher")],
        resamples=1000,
        seed=0,
    )
    assert result["method"] == "paired" and result["resamples"] == 1000 and result["seed"] == 0
    assert result["n"] == 200 and result["n_groups"] is None and result["level"] == 0.95
    for entry in [*result["stats"].values(), *result["diffs"].values()]:
        assert entry["lo"] <= entry["point"] <= entry["hi"]
        assert (entry["resamples"], entry["seed"], entry["method"], entry["valid_resamples"]) == (
            1000,
            0,
            "paired",
            1000,
        )
    diff = result["diffs"]["student_minus_teacher"]
    assert diff["a"] == "student" and diff["b"] == "teacher"
    assert diff["point"] == pytest.approx(student.mean() - teacher.mean())
    assert result["stats"]["student"]["point"] == pytest.approx(student.mean())


def test_identical_systems_have_a_zero_difference_interval() -> None:
    student, _ = _systems()
    result = paired_bootstrap(
        len(student), {"a": _mean_of(student), "b": _mean_of(student.copy())}, [("a", "b")], resamples=200
    )
    diff = result["diffs"][diff_name("a", "b")]
    assert (diff["point"], diff["lo"], diff["hi"]) == (0.0, 0.0, 0.0)


def test_differences_are_paired() -> None:
    # b is a shifted copy of a, so every paired resample gives exactly the same difference.
    a = np.random.default_rng(3).random(50)
    b = a - 0.1
    result = paired_bootstrap(50, {"a": _mean_of(a), "b": _mean_of(b)}, [("a", "b")], resamples=100, seed=5)
    diff = result["diffs"]["a_minus_b"]
    assert diff["lo"] == pytest.approx(0.1) and diff["hi"] == pytest.approx(0.1)
    assert result["stats"]["a"]["hi"] - result["stats"]["a"]["lo"] > 0.05


def test_every_statistic_sees_the_same_documented_index_draw() -> None:
    seen: dict[str, list[IntArray]] = {"x": [], "y": []}

    def recorder(name: str) -> Callable[[IntArray], float]:
        def stat(idx: IntArray) -> float:
            seen[name].append(idx.copy())
            return float(len(idx))

        return stat

    paired_bootstrap(6, {"x": recorder("x"), "y": recorder("y")}, resamples=3, seed=11)
    rng = np.random.default_rng(11)
    expected = [np.arange(6)] + [rng.integers(0, 6, size=6) for _ in range(3)]
    for name in ("x", "y"):
        assert len(seen[name]) == 4
        for got, want in zip(seen[name], expected, strict=True):
            np.testing.assert_array_equal(got, want)


def test_tiny_case_by_hand() -> None:
    values = np.array([0.0, 1.0])
    draw = np.random.default_rng(0).integers(0, 2, size=2)
    result = paired_bootstrap(2, {"m": _mean_of(values)}, resamples=1, seed=0)
    entry = result["stats"]["m"]
    assert entry["point"] == 0.5
    assert entry["lo"] == entry["hi"] == pytest.approx(values[draw].mean())


def test_cluster_bootstrap_resamples_whole_groups() -> None:
    groups = ["t2", "t1", "t2", "t3", "t1", "t2"]  # sorted names: t1 -> [1, 4], t2 -> [0, 2, 5], t3 -> [3]
    members = {"t1": [1, 4], "t2": [0, 2, 5], "t3": [3]}
    seen: list[IntArray] = []

    def stat(idx: IntArray) -> float:
        seen.append(idx.copy())
        return float(len(idx))

    result = paired_bootstrap(6, {"size": stat}, resamples=5, seed=2, groups=groups)
    assert result["method"] == "cluster" and result["n_groups"] == 3
    assert result["stats"]["size"]["method"] == "cluster"
    rng = np.random.default_rng(2)
    names = ["t1", "t2", "t3"]
    for idx in seen[1:]:
        picks = rng.integers(0, 3, size=3)
        want = np.concatenate([members[names[k]] for k in picks])
        np.testing.assert_array_equal(idx, want)
    sizes = {len(idx) for idx in seen[1:]}
    assert sizes <= {3, 4, 5, 6, 7, 8, 9}


def test_cluster_bootstrap_is_deterministic_and_wider_for_correlated_groups() -> None:
    # Ten groups whose members share one outcome: the cluster interval must be wider than the paired one.
    groups = [f"g{i // 20}" for i in range(200)]
    outcome = np.repeat(np.random.default_rng(4).random(10) < 0.5, 20).astype(np.float64)
    stats = {"acc": _mean_of(outcome)}
    cluster = paired_bootstrap(200, stats, resamples=500, seed=0, groups=groups)
    again = paired_bootstrap(200, stats, resamples=500, seed=0, groups=groups)
    paired = paired_bootstrap(200, stats, resamples=500, seed=0)
    assert cluster == again
    width = cluster["stats"]["acc"]["hi"] - cluster["stats"]["acc"]["lo"]
    assert width > paired["stats"]["acc"]["hi"] - paired["stats"]["acc"]["lo"]


def test_undefined_resamples_are_left_out() -> None:
    conf = np.array([0.9, 0.2, 0.8, 0.3])
    correct = np.array([True, False, True, False])
    result = paired_bootstrap(4, {"auroc": lambda idx: auroc(conf[idx], correct[idx])}, resamples=200, seed=0)
    entry = result["stats"]["auroc"]
    assert entry["point"] == 1.0
    assert 0 < entry["valid_resamples"] < 200
    assert entry["lo"] == entry["hi"] == 1.0

    none = paired_bootstrap(3, {"never": lambda idx: None}, resamples=10)
    assert none["stats"]["never"] == {
        "point": None,
        "lo": None,
        "hi": None,
        "valid_resamples": 0,
        "resamples": 10,
        "seed": 0,
        "method": "paired",
    }


def test_arguments_are_validated() -> None:
    stat = {"m": _mean_of(np.zeros(3))}
    with pytest.raises(ValueError, match="unknown statistic"):
        paired_bootstrap(3, stat, [("m", "missing")])
    with pytest.raises(ValueError, match="groups"):
        paired_bootstrap(3, stat, groups=["a", "b"])
    with pytest.raises(ValueError, match="at least one example"):
        paired_bootstrap(0, stat)
    with pytest.raises(ValueError, match="resamples"):
        paired_bootstrap(3, stat, resamples=0)
    with pytest.raises(ValueError, match="no statistics"):
        paired_bootstrap(3, {})
