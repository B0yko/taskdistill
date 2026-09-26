from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import pytest
from sklearn.metrics import f1_score

from taskdistill.evaluate.metrics import (
    accuracy,
    agreement,
    document_counts,
    encode_labels,
    extraction_agreement,
    extraction_scores,
    field_equal,
    macro_f1,
    macro_f1_from_codes,
    micro_f1,
    normalise_field_value,
    parse_iso_date,
    per_group_scores,
    per_trait_breakdown,
    resolve_fields,
    score,
)
from taskdistill.evaluate.splits import EvalRecord

# --- classification ---------------------------------------------------------------------------------------------


def test_accuracy_counts_label_equality() -> None:
    assert accuracy(["a", "b", "c", "a"], ["a", "c", "c", None]) == 0.5
    assert accuracy([], []) == 0.0


def test_accuracy_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        accuracy(["a"], ["a", "b"])


def test_agreement_is_label_equality_with_the_teacher() -> None:
    assert agreement(["x", "y", "z"], ["x", "y", "y"]) == pytest.approx(2 / 3)


def test_macro_f1_six_examples_by_hand() -> None:
    # a: TP=1 FP=1 FN=1 -> 2/4; b: TP=2 FP=1 FN=0 -> 4/5; c: TP=1 FP=0 FN=1 -> 2/3; d: absent -> 0.
    gold = ["a", "a", "b", "b", "c", "c"]
    pred = ["a", "b", "b", "b", "a", "c"]
    expected = (1 / 2 + 4 / 5 + 2 / 3 + 0) / 4  # 59/120
    assert macro_f1(gold, pred, ["a", "b", "c", "d"]) == pytest.approx(expected)
    assert expected == pytest.approx(59 / 120)


def test_macro_f1_unknown_and_missing_predictions_are_wrong() -> None:
    assert macro_f1(["a", "b"], [None, "zzz"], ["a", "b"]) == 0.0
    # a: TP=1, FN=0, FP=0 -> 1; b: FN=1 -> 0
    assert macro_f1(["a", "b"], ["a", None], ["a", "b"]) == 0.5


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_macro_f1_matches_sklearn(seed: int) -> None:
    rng = random.Random(seed)
    labels = [f"l{i}" for i in range(7)]
    gold = [rng.choice(labels[:6]) for _ in range(300)]
    pred = [rng.choice([*labels, "unknown"]) if rng.random() < 0.4 else g for g in gold]
    expected = f1_score(gold, pred, labels=labels, average="macro", zero_division=0)
    assert macro_f1(gold, pred, labels) == pytest.approx(expected)


def test_encode_labels_and_codes_path() -> None:
    codes = encode_labels(["b", None, "a", "q"], ["a", "b"])
    assert codes.tolist() == [1, 2, 0, 2]
    gold = encode_labels(["a", "a", "b", "b", "c", "c"], ["a", "b", "c", "d"])
    pred = encode_labels(["a", "b", "b", "b", "a", "c"], ["a", "b", "c", "d"])
    assert macro_f1_from_codes(gold, pred, 4) == pytest.approx(59 / 120)


def test_label_set_must_be_valid() -> None:
    with pytest.raises(ValueError, match="empty"):
        macro_f1(["a"], ["a"], [])
    with pytest.raises(ValueError, match="duplicates"):
        macro_f1(["a"], ["a"], ["a", "a"])


# --- extraction: values -----------------------------------------------------------------------------------------


def test_normalise_field_value_strings_and_numbers() -> None:
    assert normalise_field_value("  \uff21cme\u00a0  GmbH\n") == "acme gmbh"  # fullwidth A, NBSP, newline
    assert normalise_field_value("Straße") == "strasse"
    assert normalise_field_value(12) == 12.0
    assert normalise_field_value(None) is None
    assert normalise_field_value(True) is True
    assert normalise_field_value({"A": " X "}) == {"A": "x"}
    assert normalise_field_value([" Y", 2]) == ["y", 2.0]


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        (100, 100.004, True),
        (100, 100.006, False),
        (1.23, 1.235, True),  # exactly at the tolerance
        (5, 5.0, True),
        (True, 1, False),
        (True, True, True),
        (1234.56, "1234.56", False),
        ("ACME  Ltd", " acme ltd", True),
        ("INV-0042", "INV-0043", False),
        (None, None, True),
        ("x", None, False),
        (None, "x", False),
    ],
)
def test_field_equal_values(gold: object, pred: object, expected: bool) -> None:
    assert field_equal(gold, pred) is expected


@pytest.mark.parametrize(
    ("pred", "expected"),
    [
        ("2026-03-01", True),
        (" 2026-03-01 ", True),
        ("2026-03-01T00:00:00", True),
        ("2026-03-01 00:00", True),
        ("2026-03-01T10:15:00Z", True),
        ("2026-03-01T10:15:00+02:00", True),
        ("2026-03-02", False),
        ("01/03/2026", False),
        ("2026-3-1", False),
        ("20260301", False),
    ],
)
def test_field_equal_iso_dates(pred: str, expected: bool) -> None:
    assert field_equal("2026-03-01", pred) is expected


def test_dates_need_a_strict_reference() -> None:
    # The reference must be YYYY-MM-DD; a date-time reference is compared as a string.
    assert field_equal("2026-03-01T00:00:00", "2026-03-01") is False
    assert parse_iso_date("2026-03-01T00:00:00") is None
    assert str(parse_iso_date("2026-03-01T00:00:00", strict=False)) == "2026-03-01"
    assert parse_iso_date("2026-02-30") is None
    # An impossible date falls back to string comparison.
    assert field_equal("2026-02-30", "2026-02-30") is True


def test_field_equal_nested_values() -> None:
    assert field_equal({"a": "X", "b": None}, {"a": "x"}) is True
    assert field_equal({"a": 1}, {"a": 2}) is False
    assert field_equal([1, "A"], [1.001, "a"]) is True
    assert field_equal([1], [1, 2]) is False
    assert field_equal({"a": 1}, [1]) is False


# --- extraction: documents --------------------------------------------------------------------------------------

FIELDS = ["a", "b", "c", "d"]
GOLDS = [
    {"a": "X", "b": 10, "c": None, "d": "2026-01-05"},
    {"a": "Y", "b": None, "c": "Z", "d": None},
    {"a": "Q", "b": 1, "c": None, "d": None},
    {"a": "W", "b": 2.5, "c": None, "d": "2026-02-01"},
]
PREDS = [
    # a TP, b TP (within tolerance), c both null, d wrong -> FP + FN
    {"a": "x ", "b": 10.001, "c": None, "d": "2026-01-06"},
    # a missing -> FN, b spurious -> FP only, c TP, d both null
    {"a": None, "b": 5, "c": "z", "d": None, "extra": "ignored"},
    # invalid document -> FN for a and b
    None,
    # all equal, date given as a date-time
    {"a": "W", "b": 2.5, "c": None, "d": "2026-02-01T00:00:00"},
]


def test_document_counts_by_hand() -> None:
    counts = document_counts(GOLDS, PREDS, FIELDS)
    assert counts.tp.tolist() == [2, 1, 0, 3]
    assert counts.fp.tolist() == [1, 1, 0, 0]
    assert counts.fn.tolist() == [1, 1, 2, 0]
    assert counts.valid.tolist() == [True, True, False, True]
    assert counts.fields_equal.tolist() == [3, 2, 0, 4]
    assert counts.doc_exact.tolist() == [False, False, False, True]


def test_extraction_scores_by_hand() -> None:
    result = extraction_scores(GOLDS, PREDS, FIELDS)
    assert (result["tp"], result["fp"], result["fn"]) == (6, 2, 4)
    assert result["field_micro_f1"] == pytest.approx(12 / 18)
    assert result["field_precision"] == pytest.approx(6 / 8)
    assert result["field_recall"] == pytest.approx(6 / 10)
    assert result["json_validity"] == pytest.approx(0.75)
    assert result["field_exact_match"] == pytest.approx(9 / 16)
    assert result["doc_exact_match"] == pytest.approx(0.25)
    assert result["per_field_exact"] == pytest.approx({"a": 0.5, "b": 0.5, "c": 0.75, "d": 0.5})
    assert result["n"] == 4


def test_extraction_invalid_document_types() -> None:
    result = extraction_scores([{"a": "x", "b": None}], [["not", "a", "dict"]], ["a", "b"])
    assert result["json_validity"] == 0.0
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)
    # An invalid document is wrong on every field, including the null one.
    assert result["field_exact_match"] == 0.0


def test_extraction_all_null_counts_nothing() -> None:
    result = extraction_scores([{"a": None}], [{"a": None}], ["a"])
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 0)
    assert result["field_micro_f1"] == 0.0
    assert result["field_exact_match"] == 1.0
    assert result["doc_exact_match"] == 1.0


def test_extraction_empty_and_invalid_arguments() -> None:
    empty = extraction_scores([], [], ["a"])
    assert empty["n"] == 0 and empty["json_validity"] == 0.0 and empty["per_field_exact"] == {"a": 0.0}
    with pytest.raises(ValueError, match="at least one field"):
        extraction_scores([{}], [{}], [])


def test_extraction_agreement_is_field_micro_f1_against_the_teacher() -> None:
    assert extraction_agreement(GOLDS, PREDS, FIELDS) == pytest.approx(12 / 18)
    assert micro_f1(6, 2, 4) == pytest.approx(12 / 18)
    assert micro_f1(0, 0, 0) == 0.0


def test_resolve_fields_in_first_appearance_order() -> None:
    assert resolve_fields([{"b": 1, "a": 2}, None], [{"c": 3, "a": 1}]) == ["b", "a", "c"]


# --- dispatch ---------------------------------------------------------------------------------------------------


def test_score_dispatch_classification() -> None:
    refs, preds = ["a", "a", "b", "b", "c", "c"], ["a", "b", "b", "b", "a", "c"]
    assert score("classification", "agreement", refs, preds) == pytest.approx(4 / 6)
    assert score("classification", "accuracy", refs, preds) == pytest.approx(4 / 6)
    assert score("classification", "macro_f1", refs, preds, labels=["a", "b", "c", "d"]) == pytest.approx(59 / 120)
    with pytest.raises(ValueError, match="label set"):
        score("classification", "macro_f1", refs, preds)
    with pytest.raises(ValueError, match="does not apply"):
        score("classification", "field_f1", refs, preds)


def test_score_dispatch_extraction() -> None:
    assert score("extraction", "agreement", GOLDS, PREDS, fields=FIELDS) == pytest.approx(12 / 18)
    assert score("extraction", "field_f1", GOLDS, PREDS, fields=FIELDS) == pytest.approx(12 / 18)
    assert score("extraction", "field_exact_match", GOLDS, PREDS, fields=FIELDS) == pytest.approx(9 / 16)
    assert score("extraction", "doc_exact_match", GOLDS, PREDS, fields=FIELDS) == pytest.approx(0.25)
    assert score("extraction", "json_validity", GOLDS, PREDS, fields=FIELDS) == pytest.approx(0.75)
    # Without fields, the keys seen in the documents are used; "extra" becomes a field predicted without a reference.
    assert score("extraction", "field_f1", GOLDS, PREDS) == pytest.approx(12 / 19)
    with pytest.raises(ValueError, match="does not apply"):
        score("extraction", "macro_f1", GOLDS, PREDS)
    with pytest.raises(ValueError, match="unknown task type"):
        score("regression", "accuracy", [], [])  # type: ignore[arg-type]


# --- breakdowns -------------------------------------------------------------------------------------------------


def _rec(i: int, group: str | None, traits: tuple[str, ...], ok: bool) -> EvalRecord:
    return EvalRecord(id=str(i), input="", gold="a", pred="a" if ok else "b", group=group, traits=traits)


def _acc(records: Sequence[EvalRecord]) -> float:
    return accuracy([r.gold for r in records], [r.pred for r in records])


def test_per_group_scores() -> None:
    records = [
        _rec(0, "t2", (), True),
        _rec(1, "t1", (), False),
        _rec(2, "t1", (), True),
        _rec(3, None, (), True),
    ]
    result = per_group_scores(records, _acc)
    assert list(result) == ["(none)", "t1", "t2"]
    assert result["t1"] == {"n": 2, "score": 0.5}
    assert result["t2"] == {"n": 1, "score": 1.0}
    assert result["(none)"] == {"n": 1, "score": 1.0}


def test_per_group_scores_with_a_mapping_result() -> None:
    records = [_rec(0, "g", (), True), _rec(1, "g", (), False)]
    result = per_group_scores(records, lambda rs: {"accuracy": _acc(rs), "k": len(rs)})
    assert result == {"g": {"n": 2, "accuracy": 0.5, "k": 2}}


def test_per_trait_breakdown() -> None:
    records = [
        _rec(0, None, ("eu_number", "net_terms"), False),
        _rec(1, None, ("eu_number",), True),
        _rec(2, None, (), True),
        _rec(3, None, ("net_terms", "net_terms"), True),  # a repeated trait counts once
    ]
    result = per_trait_breakdown(records, _acc)
    assert list(result) == ["(none)", "eu_number", "net_terms"]
    assert result["eu_number"] == {"n": 2, "score": 0.5}
    assert result["net_terms"] == {"n": 2, "score": 0.5}
    assert result["(none)"] == {"n": 1, "score": 1.0}


def test_macro_f1_is_fast_enough_for_bootstrap() -> None:
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 77, size=3000)
    pred = np.where(rng.random(3000) < 0.8, gold, rng.integers(0, 78, size=3000))
    values = [macro_f1_from_codes(gold[idx], pred[idx], 77) for idx in rng.integers(0, 3000, size=(200, 3000))]
    assert all(0.0 <= v <= 1.0 for v in values)
