from __future__ import annotations

import random
import time

import pytest

from taskdistill.evaluate.baseline import DEFAULT_GRID, TfidfBaseline, tune_baseline
from taskdistill.evaluate.splits import EvalRecord, TestSplit, ValidationSplit

TOPICS = {
    "card_lost": ["lost", "card", "missing", "stolen"],
    "transfer_pending": ["transfer", "pending", "waiting", "arrived"],
    "exchange_rate": ["exchange", "rate", "currency", "conversion"],
}
FILLER = ["please", "help", "my", "the", "why", "is", "today", "still", "account", "hello"]


def _texts(n: int, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        label = sorted(TOPICS)[i % len(TOPICS)]
        words = rng.sample(TOPICS[label], 2) + rng.sample(FILLER, 4)
        rng.shuffle(words)
        texts.append(" ".join(words))
        labels.append(label)
    return texts, labels


def _val(
    n: int = 30, seed: int = 99, cls: type[ValidationSplit] | type[TestSplit] = ValidationSplit
) -> ValidationSplit:
    texts, labels = _texts(n, seed)
    records = [
        EvalRecord(id=str(i), input=t, teacher=lab) for i, (t, lab) in enumerate(zip(texts, labels, strict=True))
    ]
    return cls(records, "classification")  # type: ignore[return-value]


def test_tune_baseline_fits_and_reports_the_table() -> None:
    texts, labels = _texts(90, 1)
    baseline = tune_baseline(texts, labels, _val())
    assert [row["C"] for row in baseline.table] == sorted(DEFAULT_GRID)
    assert all(row["n_val"] == 30 for row in baseline.table)
    best = max(row["val_agreement"] for row in baseline.table)
    tied = [row["C"] for row in baseline.table if row["val_agreement"] == best]
    chosen = baseline.C
    assert chosen == min(tied)  # ties go to the smaller C
    assert baseline.predict(["my card was stolen", "transfer still pending", "exchange rate today"]) == [
        "card_lost",
        "transfer_pending",
        "exchange_rate",
    ]
    assert baseline.predict([]) == []
    assert sorted(baseline.classes) == sorted(TOPICS)


def test_ties_go_to_the_smallest_c() -> None:
    texts, labels = _texts(90, 1)
    baseline = tune_baseline(texts, labels, _val(), grid=(30.0, 3.0, 10.0))
    assert [row["val_agreement"] for row in baseline.table] == [1.0, 1.0, 1.0]
    assert baseline.C == 3.0


def test_tuning_uses_teacher_labels_and_skips_records_without_one() -> None:
    texts, labels = _texts(90, 1)
    val = _val()
    records = list(val.records)
    # The teacher disagrees with the obvious topic on one record; agreement is measured against the teacher.
    records[0] = EvalRecord(id="x", input="lost card stolen", teacher="exchange_rate")
    records.append(EvalRecord(id="y", input="lost card", teacher=None))
    baseline = tune_baseline(texts, labels, ValidationSplit(records, "classification"), grid=(1.0,))
    assert baseline.table == [{"C": 1.0, "val_agreement": pytest.approx(29 / 30), "n_val": 30}]


def test_predict_with_confidence() -> None:
    texts, labels = _texts(90, 1)
    baseline = TfidfBaseline.fit(texts, labels, C=10.0)
    predicted, confidence = baseline.predict_with_confidence(["card lost", "exchange currency"])
    assert predicted == ["card_lost", "exchange_rate"]
    assert confidence.shape == (2,) and all(1 / 3 < c <= 1.0 for c in confidence)
    empty_labels, empty_conf = baseline.predict_with_confidence([])
    assert empty_labels == [] and empty_conf.shape == (0,)


def test_tuning_is_deterministic() -> None:
    texts, labels = _texts(90, 1)
    first = tune_baseline(texts, labels, _val())
    second = tune_baseline(texts, labels, _val())
    assert first.table == second.table and first.C == second.C


def test_invalid_inputs() -> None:
    texts, labels = _texts(9, 1)
    with pytest.raises(ValueError, match="labels"):
        tune_baseline(texts, labels[:-1], _val())
    with pytest.raises(ValueError, match="two distinct"):
        tune_baseline(texts, ["card_lost"] * len(texts), _val())
    with pytest.raises(ValueError, match="grid"):
        tune_baseline(texts, labels, _val(), grid=())
    no_teacher = ValidationSplit([EvalRecord(id="1", input="lost card")], "classification")
    with pytest.raises(ValueError, match="teacher label"):
        tune_baseline(texts, labels, no_teacher)


def test_tune_baseline_refuses_the_test_split() -> None:
    texts, labels = _texts(30, 1)
    with pytest.raises(TypeError, match="ValidationSplit"):
        tune_baseline(texts, labels, _val(cls=TestSplit))


def test_tuning_takes_seconds_on_nine_thousand_short_texts() -> None:
    rng = random.Random(5)
    vocab = [f"w{i}" for i in range(1500)]
    labels = [f"intent_{i}" for i in range(77)]
    keywords = {label: rng.sample(vocab, 6) for label in labels}

    def sample(n: int) -> tuple[list[str], list[str]]:
        out_texts, out_labels = [], []
        for _ in range(n):
            label = rng.choice(labels)
            words = rng.sample(keywords[label], 2) + rng.sample(vocab, 8)
            rng.shuffle(words)
            out_texts.append(" ".join(words))
            out_labels.append(label)
        return out_texts, out_labels

    train_texts, train_labels = sample(9000)
    val_texts, val_labels = sample(1000)
    val = ValidationSplit(
        [
            EvalRecord(id=str(i), input=t, teacher=lab)
            for i, (t, lab) in enumerate(zip(val_texts, val_labels, strict=True))
        ],
        "classification",
    )
    start = time.perf_counter()
    baseline = tune_baseline(train_texts, train_labels, val)
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, f"tuning took {elapsed:.1f}s"
    assert max(row["val_agreement"] for row in baseline.table) > 0.8
