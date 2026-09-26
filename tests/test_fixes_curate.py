"""Regression tests for curate fixes: schema-invalid values after the PII scrub, predefined split values checked at
import, predefined test rows whose recorded outputs are invalid, output-weighted dedupe votes and malformed text
content parts."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from taskdistill.backends.fake import FakeBackend
from taskdistill.capture.importer import ImportFormatError, import_file
from taskdistill.config import InputSpec, TaskSpec
from taskdistill.curate.extract import InputUnparsed, extract_input, input_hash
from taskdistill.curate.io import read_split
from taskdistill.curate.label import label_examples
from taskdistill.curate.merge import (
    CurateError,
    Example,
    PiiStage,
    Record,
    extract_records,
    load_sources,
    merge_records,
    normalise_examples,
    scrub_examples,
)
from taskdistill.curate.pipeline import run_curate
from taskdistill.curate.split import dedupe_split, resolve_cluster, weighted_majority
from taskdistill.serve.app import create_app
from taskdistill.serve.worker import ModelWorker
from taskdistill.store import Store
from taskdistill.tasks.extraction import canonical_output, validate
from test_curate_stages import SCHEMA, FakeTeacher, WordTokenizer, add_capture, example, make_spec
from test_serve import ANSWERS, chat
from test_serve import FakeTeacher as ServeTeacher
from test_serve import make_spec as serve_spec

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*httpx2.*")
    from fastapi.testclient import TestClient


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(path))
    return path


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.sqlite")


def write_jsonl(path: Path, rows: list[Any]) -> Path:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# (1) PII placeholders must not produce schema-invalid targets ----------------------------------------------------

EMAIL_SCHEMA: dict[str, Any] = copy.deepcopy(SCHEMA)
EMAIL_SCHEMA["properties"]["contact"] = {"type": ["string", "null"], "format": "email"}


def email_spec(**overrides: Any) -> TaskSpec:
    spec = make_spec("extraction", **overrides)
    spec.json_schema = copy.deepcopy(EMAIL_SCHEMA)
    return spec


def bill(i: int, contact: str | None) -> dict[str, Any]:
    return {"vendor_name": f"Vendor {i}", "invoice_number": f"A-{i:03d}", "total_amount": 10.0 + i, "contact": contact}


def test_a_scrubbed_teacher_value_that_fails_the_schema_drops_the_example_and_is_counted() -> None:
    spec = email_spec()
    value = bill(1, "ap@example.com")
    bad = example("Invoice A-001 from Vendor 1, questions to ap@example.com", teacher=value, gold=dict(value))
    fine = example("Invoice A-002 from Vendor 2", teacher=bill(2, None), gold=bill(2, None))
    stage = PiiStage(spec)

    kept, counts = scrub_examples(stage, [bad, fine])

    assert kept == [fine]
    assert stage.rejected(bad)
    assert bad.teacher["contact"] == "<EMAIL>"  # scrubbed, but never written: the example is dropped
    assert counts["pii_schema_invalid"] == 1
    assert counts["pii_schema_invalid_gold"] == 1
    assert counts["pii_schema_invalid_fields"] == {
        "contact": {"teacher": 1, "labelled": 0, "gold": 1, "kinds": {"email": 2}}
    }
    assert counts["by_field"]["teacher"]["email"] == 1  # hits are still counted
    warning = stage.schema_warning("teacher", "gold")
    assert warning is not None
    assert "contact (email)" in warning
    assert "1 example(s) dropped" in warning
    assert "1 gold value(s) set to None" in warning
    assert "curate.pii.kinds" in warning


def test_a_scrubbed_gold_that_fails_the_schema_becomes_none_and_the_example_stays() -> None:
    spec = email_spec()
    ex = example("Invoice A-003 from Vendor 3, ap@example.com", teacher=bill(3, None), gold=bill(3, "ap@example.com"))

    kept, counts = scrub_examples(PiiStage(spec), [ex])

    assert kept == [ex]
    assert ex.gold is None
    assert ex.teacher == bill(3, None)
    assert (counts["pii_schema_invalid"], counts["pii_schema_invalid_gold"]) == (0, 1)


def test_an_unconstrained_field_keeps_its_placeholder() -> None:
    spec = make_spec("extraction")  # contact is a plain string: <EMAIL> is valid
    value = bill(4, "ap@example.com")
    ex = example("Invoice A-004, ap@example.com", teacher=value, gold=dict(value))
    stage = PiiStage(spec)

    kept, counts = scrub_examples(stage, [ex])

    assert kept == [ex]
    assert ex.teacher["contact"] == ex.gold["contact"] == "<EMAIL>"
    assert (counts["pii_schema_invalid"], counts["pii_schema_invalid_gold"]) == (0, 0)
    assert counts["pii_schema_invalid_fields"] == {}
    assert stage.schema_warning("teacher", "gold") is None


def test_curate_never_writes_a_schema_invalid_target_after_the_scrub(store: Store, home: Path, tmp_path: Path) -> None:
    spec = email_spec(curate={"split": {"val": 0.2, "test": 0.2, "seed": 3}})
    rows = []
    for i in range(20):
        contact = f"billing{i}@example.com" if i % 4 == 0 else None
        text = f"Invoice A-{i:03d} from Vendor {i} over {10.0 + i:.2f} EUR." + (f" Mail {contact}." if contact else "")
        rows.append({"input": text, "output": json.dumps(bill(i, contact)), "gold": bill(i, contact)})
    import_file(store, spec.task, write_jsonl(tmp_path / "pairs.jsonl", rows), "pairs")
    lines: list[str] = []

    result = run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lines.append)

    written = 0
    for split in ("train", "valid", "test"):
        data = home / spec.task / "data"
        for row, meta in zip(
            read_jsonl(data / f"{split}.jsonl"), read_jsonl(data / f"{split}.meta.jsonl"), strict=True
        ):
            written += 1
            target = row["messages"][2]["content"]
            assert validate(json.loads(target), EMAIL_SCHEMA) == []
            assert validate(meta["teacher"], EMAIL_SCHEMA) == []
            assert meta["gold"] is None or validate(meta["gold"], EMAIL_SCHEMA) == []
            assert target == canonical_output(meta["teacher"], EMAIL_SCHEMA)
    assert written == 15  # the five rows with an e-mail contact are dropped
    stats = result.stats
    assert stats["stages"]["pii"]["pii_schema_invalid"] == 5
    assert stats["stages"]["pii"]["pii_schema_invalid_gold"] == 5
    assert stats["pii"]["pii_schema_invalid_fields"]["contact"]["kinds"] == {"email": 10}
    assert stats["stages"]["pii"]["hits"]["email"] == 15  # input, teacher and gold of each of the five
    assert any(line.startswith("warning: PII placeholders") and "contact (email)" in line for line in lines)
    assert any("scrubbed values failing the JSON Schema: examples dropped 5, golds set to None 5" in ln for ln in lines)


def test_a_labelled_answer_that_fails_the_schema_after_the_scrub_is_dropped(store: Store, home: Path) -> None:
    spec = email_spec()
    texts = [f"Invoice A-{i:03d} from Vendor {i} over {10.0 + i:.2f} EUR, reply ap{i}@example.com" for i in range(3)]
    rows = [{"input": text, "meta": {"split": "train"}} for text in texts]
    plain = [f"Invoice B-{i:03d} from Vendor {i} over {20.0 + i:.2f} EUR" for i in range(3)]
    rows += [{"input": text, "meta": {"split": "train"}} for text in plain]
    rows.append({"input": "Invoice C-001 from Vendor 9, write to ops@example.com", "meta": {"split": "test"}})
    store.add_imports(spec.task, "inputs", rows)

    def answer(text: str) -> str:
        contact = next((word.rstrip(".,") for word in text.split() if "@" in word), None)
        return json.dumps(bill(len(text), contact))

    teacher = FakeTeacher(answer)
    lines: list[str] = []

    result = run_curate(spec, store=store, teacher=teacher, tokenizer=WordTokenizer(), log=lines.append)

    stats = result.stats
    assert stats["labelling"]["requested"] == 7
    assert stats["labelling"]["pii_schema_invalid"] == 4
    assert stats["pii"]["pii_schema_invalid"] == 4
    assert stats["pii"]["pii_schema_invalid_fields"]["contact"]["labelled"] == 4
    assert stats["splits"] == {"train": 3, "valid": 0, "test": 0}
    assert stats["split_sizes"]["label"] == {"train": 3, "valid": 0, "test": 0}
    for row in read_split(spec.task, "train"):
        assert validate(row.teacher, EMAIL_SCHEMA) == []
    assert any(
        line.startswith("warning: PII placeholders") and "4 example(s) dropped" in line and "contact (email)" in line
        for line in lines
    )


# (2) predefined split values are checked at import, and curate names the row -------------------------------------


def test_an_unknown_split_value_is_rejected_at_import_with_its_line_and_nothing_is_written(
    store: Store, tmp_path: Path
) -> None:
    rows = [
        {"input": "Where is my card?", "output": "card arrival", "meta": {"split": "train"}},
        {"input": "Rate for euros?", "output": "exchange rate", "meta": {"split": "Validation"}},
        {"input": "Card was stolen", "output": "lost or stolen card", "meta": {"split": "eval"}},
    ]
    path = write_jsonl(tmp_path / "bad.jsonl", rows)
    with pytest.raises(ImportFormatError, match=r"bad\.jsonl:3: meta\.split is 'eval'; expected train, valid or test"):
        import_file(store, "t", path, "pairs")
    assert list(store.iter_imports("t")) == []

    path = write_jsonl(tmp_path / "num.jsonl", [{"input": "a", "meta": {"split": 3}}])
    with pytest.raises(ImportFormatError, match=r"num\.jsonl:1: meta\.split is 3"):
        import_file(store, "t", path, "inputs")
    assert list(store.iter_imports("t")) == []


def test_split_aliases_blank_and_absent_values_import_and_are_kept_as_given(store: Store, tmp_path: Path) -> None:
    rows = [
        {"input": "one", "meta": {"split": "Dev"}},
        {"input": "two", "meta": {"split": " test "}},
        {"input": "three", "meta": {"split": ""}},
        {"input": "four", "meta": {"split": None}},
        {"input": "five", "meta": {"other": 1}},
        {"input": "six"},
    ]
    counts = import_file(store, "t", write_jsonl(tmp_path / "ok.jsonl", rows), "inputs")
    assert counts["imported"] == 6
    assert [row.meta for row in store.iter_imports("t")] == [
        {"split": "Dev"}, {"split": " test "}, {"split": ""}, {"split": None}, {"other": 1}, {},
    ]  # fmt: skip


def test_the_predefined_path_is_configurable_or_can_be_turned_off(store: Store, tmp_path: Path) -> None:
    nested = write_jsonl(tmp_path / "nested.jsonl", [{"input": "a", "meta": {"info": {"split": "holdout"}}}])
    with pytest.raises(ImportFormatError, match=r"nested\.jsonl:1: meta\.info\.split is 'holdout'"):
        import_file(store, "t", nested, "inputs", predefined="meta.info.split")
    assert import_file(store, "t", nested, "inputs")["imported"] == 1  # meta.split is not set

    other = write_jsonl(tmp_path / "other.jsonl", [{"input": "b", "meta": {"split": "eval"}}])
    assert import_file(store, "u", other, "inputs", predefined=None)["imported"] == 1
    with pytest.raises(ValueError, match="invalid meta path"):
        import_file(store, "u", other, "inputs", predefined="meta.")


def test_curate_names_the_import_row_holding_an_unknown_split(store: Store, home: Path) -> None:
    spec = make_spec()
    store.add_imports(spec.task, "pairs", [{"input": "Where is my card?", "output": "card arrival"}])
    store.add_imports(spec.task, "inputs", [{"input": "My card was stolen", "meta": {"split": "eval"}}])
    bad = next(row for row in store.iter_imports(spec.task) if row.format == "inputs")

    with pytest.raises(CurateError, match=rf"meta\.split is 'eval' in import row {bad.id} \(inputs, imported "):
        run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lambda _: None)

    records = [Record("Card never came", None, None, {"split": "holdout"}, "inputs", None, "import row 7 (inputs)")]
    with pytest.raises(CurateError, match=r"'holdout' in import row 7 \(inputs\); expected"):
        merge_records(records, "meta.split")


# (3) predefined test rows stay when their recorded outputs are invalid or have no majority -----------------------


def _with_outputs(text: str, outputs: list[str], split: str | None = None) -> Example:
    """An example as merge leaves it: recorded outputs, and the predefined split (if any) in ``meta.split``."""
    ex = Example(input_hash=input_hash(text), raw_input=text, text=text, meta={"split": split} if split else {})
    ex.outputs = list(outputs)
    return ex


def test_a_predefined_test_row_without_a_valid_majority_is_kept_without_a_teacher_value() -> None:
    spec = make_spec()
    invalid_test = _with_outputs("Something odd happened", ["no idea"], split="test")
    invalid_test.golds = ["top up failed"]
    tied_test = _with_outputs("Where did my money go?", ["card arrival", "exchange rate"], split="test")
    invalid_train = _with_outputs("Odd thing for training", ["no idea"], split="train")
    tied_valid = _with_outputs("Which one is it?", ["card arrival", "exchange rate"], split="valid")
    unset = _with_outputs("No split at all", ["no idea"])

    kept, counts = normalise_examples(spec, [invalid_test, tied_test, invalid_train, tied_valid, unset])

    assert kept == [invalid_test, tied_test]
    for ex in kept:
        assert ex.teacher is None
        assert ex.teacher_origin == "invalid"
    assert invalid_test.gold == "top_up_failed"
    assert (counts["kept_test_all_invalid"], counts["kept_test_no_majority"]) == (1, 1)
    assert (counts["dropped_all_invalid"], counts["dropped_no_majority"]) == (2, 1)
    assert (counts["kept"], counts["labelled"], counts["unlabelled"]) == (2, 0, 0)


def test_labelling_never_relabels_a_test_row_kept_without_a_teacher_value() -> None:
    spec = make_spec()
    kept_invalid = example("Something odd happened", split="test")
    kept_invalid.teacher_origin = "invalid"
    pending = example("When will my card arrive?", split="test")
    teacher = FakeTeacher(lambda _: "card arrival")

    kept, stats, _ = label_examples(spec, [kept_invalid, pending], teacher, log=lambda _: None)

    assert [body["messages"][-1]["content"] for body in teacher.bodies] == ["When will my card arrive?"]
    assert kept == [kept_invalid, pending]
    assert kept_invalid.teacher is None
    assert kept_invalid.teacher_origin == "invalid"
    assert stats["requested"] == 1


@pytest.mark.parametrize("captured", [["I cannot say"], ["card arrival", "exchange rate"]])
def test_a_captured_gold_test_row_stays_in_test_whatever_the_teacher_answered(
    captured: list[str], store: Store, home: Path
) -> None:
    spec = make_spec()
    rows = [{"input": f"When will card number {i} arrive at my home?", "meta": {"split": "train"}} for i in range(4)]
    hard = "My top up failed twice and the money is gone"
    rows.append({"input": hard, "gold": "top_up_failed", "meta": {"split": "test"}})
    rows.append({"input": "Tell me when the card arrives please", "gold": "card_arrival", "meta": {"split": "test"}})
    store.add_imports(spec.task, "inputs", rows)
    for output in captured:
        add_capture(store, spec, hard, output)
    teacher = FakeTeacher(lambda _: "card arrival")

    result = run_curate(spec, store=store, teacher=teacher, tokenizer=WordTokenizer(), log=lambda _: None)

    assert result.stats["splits"]["test"] == 2
    assert hard not in [body["messages"][-1]["content"] for body in teacher.bodies]  # never relabelled
    row = next(row for row in read_split(spec.task, "test") if row.input == hard)
    assert (row.teacher, row.gold, row.target) == (None, "top_up_failed", "")
    assert result.stats["sources"]["test"]["teacher_invalid"] == 1
    normalise = result.stats["stages"]["normalise"]
    assert normalise["kept_test_all_invalid" if len(captured) == 1 else "kept_test_no_majority"] == 1
    assert normalise["dropped_all_invalid"] == normalise["dropped_no_majority"] == 0


# (4) dedupe votes count the outputs behind each merged example ---------------------------------------------------


def test_normalisation_records_how_many_outputs_support_the_winner() -> None:
    spec = make_spec()
    ex = _with_outputs("Where is my card?", ["card arrival", "Card_Arrival", "no idea", "lost or stolen card",
                                             "card arrival"])  # fmt: skip
    (kept,), _ = normalise_examples(spec, [ex])
    assert (kept.teacher, kept.votes) == ("card_arrival", 3)


def test_weighted_majority_is_strict_over_the_summed_weights() -> None:
    assert weighted_majority([("a", 3), ("b", 1)]) == (True, "a")
    assert weighted_majority([("a", 2), ("b", 1), ("b", 1)]) == (False, None)
    assert weighted_majority([("a", 3), ("b", 1), ("b", 1)]) == (True, "a")
    assert weighted_majority([({"k": 1, "j": 2}, 1), ({"j": 2, "k": 1}, 1), ("x", 1)]) == (True, {"k": 1, "j": 2})
    assert weighted_majority([(None, 5), ("a", 0)]) == (False, None)


def test_a_cluster_member_backed_by_more_outputs_outweighs_the_others() -> None:
    backed = example("Where is my new card? It has not arrived yet.", teacher="card_arrival")
    backed.votes = 3
    lower = example("where is my new card? it has not arrived yet.", teacher="lost_or_stolen_card")
    upper = example("WHERE IS MY NEW CARD? IT HAS NOT ARRIVED YET.", teacher="lost_or_stolen_card")

    keep, _ = resolve_cluster([backed, lower])
    assert keep is backed
    keep, _ = resolve_cluster([lower, upper, backed])
    assert keep is backed  # 3 outputs against 2, although two members carry the minority value

    kept, counts = dedupe_split([lower, upper, backed], 0.9)
    assert kept == [backed]
    assert counts["conflict_clusters"] == 0


def test_case_variant_duplicates_resolve_by_the_output_majority_end_to_end(store: Store, home: Path) -> None:
    spec = make_spec()
    text = "Where is my new card? It has not arrived yet."
    rows = [{"input": text, "output": "card arrival", "meta": {"split": "train"}} for _ in range(3)]
    rows.append({"input": text.lower(), "output": "lost or stolen card", "meta": {"split": "train"}})
    rows += [{"input": f"What rate do you apply for currency {i}?", "output": "exchange rate"} for i in range(3)]
    store.add_imports(spec.task, "pairs", rows)

    result = run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lambda _: None)

    train = {row.input: row.teacher for row in read_split(spec.task, "train")}
    assert train[text] == "card_arrival"
    assert text.lower() not in train
    assert result.stats["dedupe"]["conflict_clusters"] == 0


# (5) malformed text content parts are unparsed input, never an error ---------------------------------------------

LAST_USER = InputSpec.model_validate({"from": "last_user_message"})


def _body(content: Any) -> dict[str, Any]:
    return {"messages": [{"role": "system", "content": "Classify."}, {"role": "user", "content": content}]}


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": None}],
        [{"type": "text", "text": 5}],
        [{"type": "text", "text": ["a"]}],
        [{"type": "text"}],
        [{"type": "text", "text": "ok"}, {"type": "text", "text": None}],
        [{"type": "text", "text": "ok"}, {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}],
        [],
    ],
)
def test_a_text_part_without_string_text_is_unparsed(content: Any) -> None:
    with pytest.raises(InputUnparsed):
        extract_input(_body(content), LAST_USER)


def test_text_parts_with_string_text_are_joined() -> None:
    content = [{"type": "text", "text": "Where is "}, {"type": "text", "text": "my card?"}]
    assert extract_input(_body(content), LAST_USER) == "Where is my card?"


def test_curate_counts_a_capture_with_a_null_text_part_as_unparsed(store: Store) -> None:
    spec = make_spec()
    add_capture(store, spec, "When will my card arrive?", "card arrival")
    body = _body([{"type": "text", "text": None}])
    store.add_capture(task=spec.task, source="proxy", request_key="k", request_body=json.dumps(body),
                      response_body=json.dumps({"choices": [{"message": {"content": "card arrival"}}]}),
                      status=200, captured=True)  # fmt: skip
    captures, imports, _ = load_sources(store, spec.task, spec)

    records, counts = extract_records(spec, captures, imports)

    assert [r.raw_input for r in records] == ["When will my card arrive?"]
    assert records[0].source is not None and records[0].source.startswith("capture ")
    assert counts["input_unparsed"] == 1
    assert input_hash(records[0].raw_input)


@pytest.mark.parametrize("text", [None, 5])
def test_serve_routes_a_null_or_numeric_text_part_to_the_teacher_as_unparsed(text: Any, store: Store) -> None:
    backend = FakeBackend(answers=ANSWERS)
    teacher = ServeTeacher("card_arrival")
    spec = serve_spec()
    app = create_app(
        spec,
        worker=ModelWorker(lambda: backend, spec=spec),
        teacher=teacher,
        threshold=0.8,
        store=store,
        token=None,
        run_id="run-1",
    )
    body = chat()
    body["messages"][-1]["content"] = [{"type": "text", "text": text}]
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-reason"] == "input_unparsed"
    assert backend.calls == []
