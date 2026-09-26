from __future__ import annotations

import math

import pytest

from taskdistill.evaluate.selection import choose_base_model, score_runs, select_checkpoint, select_run, split_score
from taskdistill.evaluate.splits import EvalRecord, TestSplit, ValidationSplit


def _split(
    correct: int, n: int = 100, cls: type[ValidationSplit] | type[TestSplit] = ValidationSplit
) -> ValidationSplit:
    """``n`` records whose prediction agrees with the teacher on the first ``correct``."""
    records = [
        EvalRecord(id=f"r{i}", input=f"q{i}", teacher="a", gold="a", pred="a" if i < correct else "b", confidence=0.9)
        for i in range(n)
    ]
    return cls(records, "classification")  # type: ignore[return-value]


# --- checkpoint -------------------------------------------------------------------------------------------------


def test_select_checkpoint_lowest_loss_and_earliest_tie() -> None:
    history = [(100, 0.52), (200, 0.41), (300, 0.41), (400, 0.45)]
    assert select_checkpoint(_split(1, 2), history) == 200


def test_select_checkpoint_ignores_nan_losses() -> None:
    assert select_checkpoint(_split(1, 2), [(50, math.nan), (100, 0.9), (150, 0.7)]) == 150
    with pytest.raises(ValueError, match="no validation losses"):
        select_checkpoint(_split(1, 2), [(50, math.nan)])
    with pytest.raises(ValueError, match="no validation losses"):
        select_checkpoint(_split(1, 2), [])


def test_select_checkpoint_refuses_the_test_split() -> None:
    with pytest.raises(TypeError, match="ValidationSplit"):
        select_checkpoint(_split(1, 2, TestSplit), [(100, 0.5)])


# --- run --------------------------------------------------------------------------------------------------------


def test_select_run_best_metric() -> None:
    runs = {"seed-13": _split(80), "seed-14": _split(84), "seed-15": _split(82)}
    assert select_run(runs, metric="agreement") == "seed-14"
    assert score_runs(runs, metric="agreement") == {"seed-13": 0.8, "seed-14": 0.84, "seed-15": 0.82}


def test_select_run_ties_go_to_the_smallest_id() -> None:
    runs = {"b": _split(80), "a": _split(80), "c": _split(70)}
    assert select_run(runs, metric="accuracy", reference="gold") == "a"


def test_select_run_macro_f1_needs_labels() -> None:
    runs = {"x": _split(50, 4)}
    assert select_run(runs, metric="macro_f1", labels=["a", "b"]) == "x"
    with pytest.raises(ValueError, match="label set"):
        select_run(runs, metric="macro_f1")


def test_select_run_refuses_any_test_split() -> None:
    with pytest.raises(TypeError, match="ValidationSplit"):
        select_run({"a": _split(80), "b": _split(90, cls=TestSplit)}, metric="agreement")
    with pytest.raises(TypeError, match="ValidationSplit"):
        select_run(_split(80, cls=TestSplit), metric="agreement")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ValidationSplit"):
        score_runs({"a": _split(80, cls=TestSplit)}, metric="agreement")
    with pytest.raises(TypeError, match="empty mapping"):
        select_run({}, metric="agreement")


# --- base model -------------------------------------------------------------------------------------------------


def test_large_model_chosen_when_it_gains_enough_and_is_fast_enough() -> None:
    decision = choose_base_model(_split(85), _split(87), small_p95_ms=40.0, large_p95_ms=80.0, metric="agreement")
    assert decision["choice"] == "large"
    assert decision["small"] == pytest.approx(0.85) and decision["large"] == pytest.approx(0.87)
    assert decision["gain"] == pytest.approx(0.02) and decision["latency_ratio"] == pytest.approx(2.0)
    assert "2.00 points" in decision["reason"]


def test_exactly_one_point_counts_as_enough() -> None:
    # 0.57 - 0.56 is 0.00999... in binary floating point; one point must still qualify.
    decision = choose_base_model(_split(56), _split(57), small_p95_ms=40.0, large_p95_ms=100.0, metric="agreement")
    assert decision["choice"] == "large"


def test_small_model_chosen_when_the_gain_is_below_one_point() -> None:
    decision = choose_base_model(
        _split(560, 1000), _split(569, 1000), small_p95_ms=40, large_p95_ms=41, metric="accuracy"
    )
    assert decision["choice"] == "small"
    assert "below the 1.00-point minimum" in decision["reason"]


def test_small_model_chosen_when_the_large_one_is_three_times_slower() -> None:
    decision = choose_base_model(_split(80), _split(95), small_p95_ms=40.0, large_p95_ms=120.0, metric="agreement")
    assert decision["choice"] == "small"  # exactly 3x is not under 3x
    assert "3.00x" in decision["reason"]
    faster = choose_base_model(_split(80), _split(95), small_p95_ms=40.0, large_p95_ms=119.0, metric="agreement")
    assert faster["choice"] == "large"


def test_base_model_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="same validation records"):
        choose_base_model(_split(5, 10), _split(5, 11), small_p95_ms=1, large_p95_ms=1, metric="agreement")
    with pytest.raises(ValueError, match="positive"):
        choose_base_model(_split(5, 10), _split(5, 10), small_p95_ms=0, large_p95_ms=1, metric="agreement")


def test_choose_base_model_refuses_the_test_split() -> None:
    with pytest.raises(TypeError, match="ValidationSplit"):
        choose_base_model(_split(85, cls=TestSplit), _split(87), small_p95_ms=40, large_p95_ms=80, metric="agreement")
    with pytest.raises(TypeError, match="ValidationSplit"):
        choose_base_model(_split(85), _split(87, cls=TestSplit), small_p95_ms=40, large_p95_ms=80, metric="agreement")


def test_split_score_leaves_out_records_without_a_reference() -> None:
    records = [
        EvalRecord(id="1", input="", teacher="a", pred="a"),
        EvalRecord(id="2", input="", teacher=None, pred="b"),
        EvalRecord(id="3", input="", teacher="b", pred="a"),
    ]
    assert split_score(ValidationSplit(records, "classification"), metric="agreement") == 0.5
    with pytest.raises(ValueError, match="no records"):
        split_score(ValidationSplit(records[1:2], "classification"), metric="agreement")


def test_agreement_is_only_measured_against_the_teacher() -> None:
    # gold a / teacher b / student a, and gold b / teacher b / student b: accuracy 1.0, agreement 0.5
    records = [
        EvalRecord(id="1", input="", gold="a", teacher="b", pred="a", confidence=0.9),
        EvalRecord(id="2", input="", gold="b", teacher="b", pred="b", confidence=0.8),
    ]
    split = ValidationSplit(records, "classification")
    with pytest.raises(ValueError, match="measured against the teacher"):
        split_score(split, metric="agreement", reference="gold")
    with pytest.raises(ValueError, match="measured against the teacher"):
        select_run({"a": split}, metric="agreement", reference="gold")
    with pytest.raises(ValueError, match="measured against the teacher"):
        choose_base_model(split, split, small_p95_ms=40, large_p95_ms=80, metric="agreement", reference="gold")
    assert split_score(split, metric="agreement") == 0.5
    assert split_score(split, metric="accuracy", reference="gold") == 1.0
