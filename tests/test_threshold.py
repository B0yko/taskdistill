from __future__ import annotations

import json
import math
import random
import time
from typing import Any

import numpy as np
import pytest

from taskdistill.evaluate.metrics import score
from taskdistill.evaluate.splits import EvalRecord, TestSplit, ValidationSplit
from taskdistill.evaluate.threshold import (
    ThresholdResult,
    apply_threshold,
    choose_index,
    evaluate_operating_point,
    quality_curve,
    select_threshold,
    threshold_from_json,
    threshold_to_json,
)

INF = math.inf


def _rec(i: int, conf: float | None, pred: Any, teacher: Any, gold: Any = None) -> EvalRecord:
    return EvalRecord(id=str(i), input=f"q{i}", gold=gold, teacher=teacher, pred=pred, confidence=conf)


# (confidence, student, teacher): agreement with the teacher at each threshold
#   t=0.5 -> 3/5, t=0.6 -> 4/5, t=0.7 -> 4/5, t=0.8 -> 1, t=0.9 -> 1, always escalate -> 1
FIVE = [
    _rec(1, 0.9, "A", "A"),
    _rec(2, 0.8, "B", "B"),
    _rec(3, 0.7, "C", "D"),
    _rec(4, 0.6, "A", "A"),
    _rec(5, 0.5, "B", "C"),
]


def _val(records: list[EvalRecord], task_type: str = "classification") -> ValidationSplit:
    return ValidationSplit(records, task_type)  # type: ignore[arg-type]


def test_curve_by_hand() -> None:
    result = select_threshold(_val(FIVE), reference="teacher", metric="agreement", target=0.8)
    assert result.curve == pytest.approx(
        [(0.5, 0.0, 0.6), (0.6, 0.2, 0.8), (0.7, 0.4, 0.8), (0.8, 0.6, 1.0), (0.9, 0.8, 1.0), (INF, 1.0, 1.0)]
    )


@pytest.mark.parametrize(
    ("target", "threshold", "rate", "quality"),
    [
        (0.6, 0.5, 0.0, 0.6),  # every record answered by the student
        (0.8, 0.6, 0.2, 0.8),  # 0.6 and 0.7 both reach 0.8; 0.6 escalates less
        (0.9, 0.8, 0.6, 1.0),
        (1.0, 0.8, 0.6, 1.0),
    ],
)
def test_lowest_escalation_rate_meeting_the_target(
    target: float, threshold: float, rate: float, quality: float
) -> None:
    result = select_threshold(_val(FIVE), reference="teacher", metric="agreement", target=target)
    assert (result.threshold, result.escalation_rate, result.quality) == pytest.approx((threshold, rate, quality))
    assert result.met is True and result.warning is None
    assert result.target_value == target and result.n == 5 and result.teacher_quality == 1.0


def test_max_drop_against_the_teacher_is_one_minus_max_drop() -> None:
    result = select_threshold(_val(FIVE), reference="teacher", metric="agreement", max_drop=0.2)
    assert result.target_value == pytest.approx(0.8)
    assert result.threshold == 0.6
    assert result.max_drop == 0.2 and result.target is None


def test_max_drop_against_gold_uses_the_teacher_gold_score() -> None:
    # Against gold the teacher is right on records 1, 2 and 5 (score 0.6), so max_drop 0 means a target of 0.6.
    records = [
        _rec(1, 0.9, "A", "A", gold="A"),
        _rec(2, 0.8, "B", "B", gold="B"),
        _rec(3, 0.7, "C", "D", gold="C"),
        _rec(4, 0.6, "A", "A", gold="B"),
        _rec(5, 0.5, "B", "C", gold="C"),
    ]
    # gold accuracy per threshold: t=0.5 -> 3/5 (1,2,3), t=0.6 -> 4/5 (1,2,3,5), t=0.7 -> 4/5, t=0.8 -> 3/5,
    # t=0.9 -> 3/5, always escalate -> 3/5
    result = select_threshold(_val(records), reference="gold", metric="accuracy", max_drop=0.0)
    assert [q for _, _, q in result.curve] == pytest.approx([0.6, 0.8, 0.8, 0.6, 0.6, 0.6])
    assert result.teacher_quality == pytest.approx(0.6)
    assert result.target_value == pytest.approx(0.6)
    assert result.threshold == 0.5 and result.escalation_rate == 0.0
    stricter = select_threshold(_val(records), reference="gold", metric="accuracy", target=0.8)
    assert stricter.threshold == 0.6 and stricter.quality == pytest.approx(0.8)


def test_no_threshold_meets_the_target() -> None:
    records = [_rec(1, 0.9, "A", "B", gold="A"), _rec(2, 0.4, "B", "B", gold="C")]
    result = select_threshold(_val(records), reference="gold", metric="accuracy", target=0.9)
    assert math.isinf(result.threshold) and result.always_escalate
    assert result.escalation_rate == 1.0 and result.met is False
    assert result.warning is not None and "every request is escalated" in result.warning
    assert result.quality == 0.0  # the teacher is wrong on both


def test_only_always_escalate_meets_the_target() -> None:
    records = [_rec(1, 0.9, "A", "B"), _rec(2, 0.4, "B", "C")]
    result = select_threshold(_val(records), reference="teacher", metric="agreement", target=0.99)
    assert result.always_escalate and result.met is True
    assert result.warning is not None and "only escalating" in result.warning


def test_tied_confidences_escalate_together_and_equality_answers() -> None:
    records = [_rec(1, 0.7, "A", "A"), _rec(2, 0.7, "B", "C"), _rec(3, 0.9, "A", "A"), _rec(4, None, "A", "A")]
    result = select_threshold(_val(records), reference="teacher", metric="agreement", target=0.75)
    # candidates: 0.7 (records 1-3 answered: 3/4 agree), 0.9 (record 3 answered: 4/4), inf. A missing confidence
    # always escalates.
    assert [p[0] for p in result.curve] == [0.7, 0.9, INF]
    assert [p[1] for p in result.curve] == pytest.approx([0.25, 0.75, 1.0])
    assert [p[2] for p in result.curve] == pytest.approx([0.75, 1.0, 1.0])
    assert result.threshold == 0.7
    answers, escalated = apply_threshold(_val(records), 0.7)
    assert answers == ["A", "B", "A", "A"] and escalated.tolist() == [False, False, False, True]


def test_ties_in_escalation_rate_go_to_the_higher_threshold() -> None:
    thresholds = np.array([0.2, 0.3, 0.4, INF])
    rates = np.array([0.5, 0.5, 0.75, 1.0])
    quality = np.array([0.9, 0.9, 0.95, 1.0])
    assert choose_index(thresholds, rates, quality, 0.8) == 1
    assert choose_index(thresholds, rates, quality, 0.95) == 2
    assert choose_index(thresholds, rates, quality, 1.01) is None


def test_macro_f1_threshold_by_hand() -> None:
    labels = ["A", "B", "C"]
    records = [
        _rec(1, 0.9, "A", "A"),
        _rec(2, 0.6, "A", "B"),
        _rec(3, 0.3, "C", "C"),
    ]
    result = select_threshold(_val(records), reference="teacher", metric="macro_f1", target=0.99, labels=labels)
    # t=0.3: A: tp1 fp1 -> 2/3, B: fn1 -> 0, C: tp1 -> 1 => 5/9; t=0.6: record 3 escalates, same answers => 5/9;
    # t=0.9: records 2, 3 escalate => all three correct => 1.
    assert [q for _, _, q in result.curve] == pytest.approx([5 / 9, 5 / 9, 1.0, 1.0])
    assert result.threshold == 0.9 and result.escalation_rate == pytest.approx(2 / 3)


def test_extraction_agreement_threshold_by_hand() -> None:
    fields = ["a", "b"]
    teacher = [{"a": "x", "b": 1}, {"a": "y", "b": None}, {"a": "z", "b": 3}]
    preds = [
        {"a": "X", "b": 1.001},  # TP 2
        {"a": "q", "b": None},  # FP 1, FN 1
        None,  # invalid: FN 2
    ]
    confs = [0.95, 0.5, 0.2]
    records = [_rec(i, c, p, t) for i, (c, p, t) in enumerate(zip(confs, preds, teacher, strict=True))]
    result = select_threshold(
        _val(records, "extraction"), reference="teacher", metric="agreement", target=0.8, fields=fields
    )
    # t=0.2: tp 2, fp 1, fn 3 -> 4/8; t=0.5: record 3 escalates -> tp 4, fp 1, fn 1 -> 8/10;
    # t=0.95: tp 5 -> 1; always escalate -> 1.
    assert [q for _, _, q in result.curve] == pytest.approx([0.5, 0.8, 1.0, 1.0])
    assert result.threshold == 0.5 and result.escalation_rate == pytest.approx(1 / 3)


def _random_split(task_type: str, n: int, seed: int) -> tuple[ValidationSplit, list[str], list[str]]:
    rng = random.Random(seed)
    labels = [f"l{i}" for i in range(5)]
    fields = ["f1", "f2", "f3"]
    records = []
    gold: Any
    teacher: Any
    pred: Any
    for i in range(n):
        conf = rng.choice([None, round(rng.random(), 2)]) if rng.random() < 0.1 else round(rng.random(), 2)
        if task_type == "classification":
            gold = rng.choice(labels)
            teacher = gold if rng.random() < 0.85 else rng.choice(labels)
            pred = teacher if rng.random() < 0.7 else rng.choice([*labels, None, "other"])
        else:
            gold = {f: rng.choice([None, "a", "b", 1, 2.0]) for f in fields}
            teacher = {f: (v if rng.random() < 0.85 else rng.choice([None, "a", 1])) for f, v in gold.items()}
            pred = (
                None
                if rng.random() < 0.1
                else {f: (v if rng.random() < 0.7 else rng.choice([None, "b", 2.0])) for f, v in teacher.items()}
            )
        records.append(_rec(i, conf, pred, teacher, gold))
    return _val(records, task_type), labels, fields


@pytest.mark.parametrize(
    ("task_type", "reference", "metric"),
    [
        ("classification", "teacher", "agreement"),
        ("classification", "gold", "accuracy"),
        ("classification", "gold", "macro_f1"),
        ("classification", "teacher", "macro_f1"),
        ("extraction", "teacher", "agreement"),
        ("extraction", "gold", "field_f1"),
        ("extraction", "gold", "field_exact_match"),
        ("extraction", "gold", "doc_exact_match"),
        ("extraction", "teacher", "json_validity"),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_vectorised_curve_matches_direct_scoring(task_type: str, reference: str, metric: str, seed: int) -> None:
    split, labels, fields = _random_split(task_type, 120, seed)
    result = select_threshold(
        split,
        reference=reference,
        metric=metric,
        target=0.5,
        labels=labels,
        fields=fields,  # type: ignore[arg-type]
    )
    refs = [getattr(r, reference) for r in split.records]
    for threshold, rate, quality in result.curve:
        answers, escalated = apply_threshold(split, threshold)
        assert rate == pytest.approx(float(escalated.mean()))
        assert quality == pytest.approx(score(split.task_type, metric, refs, answers, labels, fields))


def test_records_without_a_reference_are_left_out() -> None:
    records = [*FIVE, _rec(6, 0.99, "A", None)]
    result = select_threshold(_val(records), reference="teacher", metric="agreement", target=0.8)
    assert result.n == 5 and result.threshold == 0.6


def test_selection_is_fast_on_a_thousand_distinct_confidences() -> None:
    rng = np.random.default_rng(0)
    labels = [f"intent_{i}" for i in range(77)]
    conf = rng.permutation(1000) / 1000 + 0.0005
    records = []
    for i in range(1000):
        teacher = labels[int(rng.integers(0, 77))]
        pred = teacher if rng.random() < conf[i] else labels[int(rng.integers(0, 77))]
        records.append(_rec(i, float(conf[i]), pred, teacher, gold=teacher))
    split = _val(records)
    start = time.perf_counter()
    for metric in ("agreement", "macro_f1"):
        result = select_threshold(split, reference="teacher", metric=metric, target=0.97, labels=labels)
        assert len(result.curve) == 1001
    assert time.perf_counter() - start < 5.0


def test_invalid_arguments() -> None:
    split = _val(FIVE)
    with pytest.raises(ValueError, match="either target or max_drop"):
        select_threshold(split, reference="teacher", metric="agreement", target=0.9, max_drop=0.1)
    with pytest.raises(ValueError, match="set target or max_drop"):
        select_threshold(split, reference="teacher", metric="agreement")
    with pytest.raises(ValueError, match="label set"):
        select_threshold(split, reference="teacher", metric="macro_f1", target=0.9)
    with pytest.raises(ValueError, match="does not apply"):
        select_threshold(split, reference="teacher", metric="field_f1", target=0.9)
    with pytest.raises(ValueError, match="reference"):
        select_threshold(split, reference="judge", metric="agreement", target=0.9)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not match"):
        select_threshold(split, reference="teacher", metric="agreement", target=0.9, task_type="extraction")
    with pytest.raises(ValueError, match="no records"):
        select_threshold(_val([]), reference="teacher", metric="agreement", target=0.9)


def test_select_threshold_refuses_the_test_split() -> None:
    with pytest.raises(TypeError, match="ValidationSplit"):
        select_threshold(TestSplit(FIVE, "classification"), reference="teacher", metric="agreement", target=0.8)
    with pytest.raises(TypeError, match="ValidationSplit"):
        select_threshold(list(FIVE), reference="teacher", metric="agreement", target=0.8)  # type: ignore[arg-type]


def test_reporting_on_the_test_split() -> None:
    test = TestSplit(FIVE, "classification")
    answers, escalated = apply_threshold(test, 0.7)
    assert answers == ["A", "B", "C", "A", "C"]
    assert escalated.tolist() == [False, False, False, True, True]
    point = evaluate_operating_point(test, 0.7, reference="teacher", metric="agreement", target=0.9)
    assert point == {
        "split": "test",
        "threshold": 0.7,
        "always_escalate": False,
        "n": 5,
        "escalation_rate": pytest.approx(0.4),
        "quality": pytest.approx(0.8),
        "teacher_quality": 1.0,
        "target_value": 0.9,
        "met": False,
    }
    dropped = evaluate_operating_point(test, INF, reference="teacher", metric="agreement", max_drop=0.05)
    assert dropped["threshold"] is None and dropped["always_escalate"] and dropped["met"] is True
    assert dropped["target_value"] == pytest.approx(0.95)
    open_ended = evaluate_operating_point(test, 0.5, reference="teacher", metric="agreement")
    assert open_ended["met"] is None and open_ended["quality"] == pytest.approx(0.6)


def test_result_serialises_without_infinity() -> None:
    result = select_threshold(_val(FIVE), reference="teacher", metric="agreement", target=0.8)
    text = json.dumps(result.to_dict(), allow_nan=False)
    data = json.loads(text)
    assert data["threshold"] == 0.6 and data["always_escalate"] is False
    assert data["curve"][-1] == {"threshold": None, "escalation_rate": 1.0, "quality": 1.0}
    restored = ThresholdResult.from_dict(data)
    assert restored == result
    assert threshold_to_json(INF) is None and threshold_from_json(None) == INF
    assert threshold_from_json(threshold_to_json(0.25)) == 0.25
    assert result.point == (0.6, pytest.approx(0.2), pytest.approx(0.8))


def test_quality_curve_needs_task_records() -> None:
    thresholds, rates, quality = quality_curve(
        FIVE, task_type="classification", reference="teacher", metric="agreement"
    )
    assert thresholds.tolist() == [0.5, 0.6, 0.7, 0.8, 0.9, INF]
    assert rates.tolist() == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert quality.tolist() == pytest.approx([0.6, 0.8, 0.8, 1.0, 1.0, 1.0])


# (confidence, student, teacher): a NaN confidence escalates like a missing one
#   t=0.6 -> records 2 and 3 answered: 2/3 agree, 1/3 escalated; t=0.9 -> 3/3, 2/3 escalated; inf -> 3/3
WITH_NAN = [_rec(1, math.nan, "b", "a"), _rec(2, 0.9, "a", "a"), _rec(3, 0.6, "c", "b")]


def test_nan_confidence_escalates_at_every_threshold() -> None:
    answers, escalated = apply_threshold(_val(WITH_NAN), 0.5)
    assert answers == ["a", "a", "c"] and escalated.tolist() == [True, False, False]
    answers, escalated = apply_threshold(TestSplit(WITH_NAN, "classification"), INF)
    assert answers == ["a", "a", "b"] and escalated.tolist() == [True, True, True]
    answers, escalated = apply_threshold(_val(WITH_NAN), -INF)
    assert answers == ["a", "a", "c"] and escalated.tolist() == [True, False, False]
    point = evaluate_operating_point(_val(WITH_NAN), 0.5, reference="teacher", metric="agreement")
    assert point["escalation_rate"] == pytest.approx(1 / 3) and point["quality"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="NaN"):
        apply_threshold(_val(WITH_NAN), math.nan)


def test_nan_confidence_selection_matches_reporting() -> None:
    split = _val(WITH_NAN)
    result = select_threshold(split, reference="teacher", metric="agreement", target=0.9)
    assert [p[0] for p in result.curve] == [0.6, 0.9, INF]
    assert [p[1] for p in result.curve] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert [p[2] for p in result.curve] == pytest.approx([2 / 3, 1.0, 1.0])
    assert result.threshold == 0.9 and result.met is True
    for threshold, rate, quality in result.curve:
        point = evaluate_operating_point(split, threshold, reference="teacher", metric="agreement")
        assert point["escalation_rate"] == pytest.approx(rate) and point["quality"] == pytest.approx(quality)


# gold a / teacher b / student a, and gold b / teacher b / student b: accuracy 1.0, agreement with the teacher 0.5
GOLD_VS_TEACHER = [_rec(1, 0.9, "a", "b", gold="a"), _rec(2, 0.8, "b", "b", gold="b")]


def test_agreement_is_only_measured_against_the_teacher() -> None:
    split = _val(GOLD_VS_TEACHER)
    with pytest.raises(ValueError, match="measured against the teacher"):
        select_threshold(split, reference="gold", metric="agreement", target=0.9)
    test = TestSplit(GOLD_VS_TEACHER, "classification")
    with pytest.raises(ValueError, match="measured against the teacher"):
        evaluate_operating_point(test, 0.5, reference="gold", metric="agreement")
    with pytest.raises(ValueError, match="measured against the teacher"):
        quality_curve(GOLD_VS_TEACHER, task_type="classification", reference="gold", metric="agreement")
    with pytest.raises(ValueError, match="measured against the teacher"):
        select_threshold(
            _val([_rec(1, 0.9, {"x": 1}, {"x": 1}, gold={"x": 1})], "extraction"),
            reference="gold",
            metric="agreement",
            target=0.9,
        )
    gold = select_threshold(split, reference="gold", metric="accuracy", target=1.0)
    assert gold.threshold == 0.8 and gold.escalation_rate == 0.0 and gold.quality == 1.0
    teacher = evaluate_operating_point(split, 0.8, reference="teacher", metric="agreement")
    assert teacher["quality"] == pytest.approx(0.5)
