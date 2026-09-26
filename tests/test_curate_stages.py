"""Curate stage functions, each on hand-built inputs: load, extraction, merge, normalisation, PII, split, dedupe,
cross-split removal, labelling, the length filter and the dataset card."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

from taskdistill.config import TaskSpec
from taskdistill.curate.card import render_card, unknown_card_keys
from taskdistill.curate.extract import input_hash
from taskdistill.curate.label import label_examples
from taskdistill.curate.length import length_filter, training_messages
from taskdistill.curate.merge import (
    CurateError,
    Example,
    PiiStage,
    Record,
    canonical_split,
    extract_records,
    load_sources,
    merge_records,
    normalise_examples,
    scrub_examples,
)
from taskdistill.curate.split import (
    assign_splits,
    by_split,
    dedupe_split,
    dedupe_within,
    fill_groups,
    remove_cross_split,
)
from taskdistill.store import Store
from taskdistill.teacher.base import ReplayMiss, TeacherResult
from taskdistill.teacher.client import SpendNotConfirmed
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

TEACHER_MODEL = "vendor/teacher-a"
LABELS = ["card_arrival", "lost_or_stolen_card", "exchange_rate", "top_up_failed"]
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor_name": {"type": "string"},
        "invoice_number": {"type": "string"},
        "total_amount": {"type": "number"},
        "contact": {"type": ["string", "null"]},
    },
    "required": ["vendor_name", "invoice_number", "total_amount", "contact"],
    "additionalProperties": False,
}


def make_spec(kind: str = "classification", *, curate: dict[str, Any] | None = None, **overrides: Any) -> TaskSpec:
    cfg: dict[str, Any] = {
        "task": "intents" if kind == "classification" else "bills",
        "type": kind,
        "teacher": {
            "model": TEACHER_MODEL,
            "max_tokens": 24 if kind == "classification" else 256,
            "extra_body": {"provider": {"order": ["alpha/fp8"], "allow_fallbacks": False}},
        },
        "student": {"system_prompt": "Answer briefly.", "base_model": "fake/student"},
        "train": {"max_seq_len": 512},
        "curate": curate or {},
        "cascade": {"target": 0.97},
    }
    if kind == "classification":
        cfg["labels_file"] = "labels.txt"
    else:
        cfg["schema_file"] = "schema.json"
    cfg.update(overrides)
    spec = TaskSpec.model_validate(cfg)
    spec.teacher_prompt = "Label the message with one intent.\n" if kind == "classification" else "Extract fields.\n"
    if kind == "classification":
        spec.labels = list(LABELS)
    else:
        spec.json_schema = copy.deepcopy(SCHEMA)
    spec.source = f"tasks/{spec.task}/task.yaml"
    return spec


def completion(content: str | None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": TEACHER_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 3, "cost": 1e-5},
    }


def add_capture(store: Store, spec: TaskSpec, text: str, output: str | None, **fields: Any) -> None:
    body = build_teacher_request(spec, text)
    row: dict[str, Any] = {
        "task": spec.task,
        "source": "proxy",
        "request_key": request_key(body),
        "request_body": json.dumps(body),
        "response_body": json.dumps(completion(output)),
        "status": 200,
        "captured": True,
    }
    row.update(fields)
    store.add_capture(**row)


def example(text: str, teacher: Any = None, gold: Any = None, split: str | None = None, **meta: Any) -> Example:
    ex = Example(input_hash=input_hash(text), raw_input=text, text=text, meta=dict(meta), gold=gold)
    ex.teacher = teacher
    ex.teacher_origin = "recorded" if teacher is not None else None
    ex.split = split
    return ex


class WordTokenizer:
    """One token per whitespace-separated word, plus two per message; records how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        self.calls.append({"messages": copy.deepcopy(messages), **kwargs})
        return [7] * sum(len(m["content"].split()) + 2 for m in messages)


class FakeTeacher:
    """A teacher source that answers from a function of the user message and records every body it receives."""

    def __init__(
        self,
        answer: Callable[[str], str | None],
        *,
        mode: str = "replay",
        cost: float = 0.0,
        cached: Iterable[str] = (),
        miss: Iterable[str] = (),
        usage_cost: float | None = None,
        provider: str = "Alpha",
    ) -> None:
        self.answer = answer
        self.mode = mode  # type: ignore[assignment]
        self.cost = cost
        self.cached_inputs = set(cached)
        self.miss = set(miss)
        self.usage_cost = usage_cost
        self.provider = provider
        self.bodies: list[dict[str, Any]] = []

    def is_cached(self, body: dict[str, Any]) -> bool:
        return body["messages"][-1]["content"] in self.cached_inputs

    async def complete(self, body: dict[str, Any]) -> TeacherResult:
        self.bodies.append(copy.deepcopy(body))
        text = body["messages"][-1]["content"]
        if text in self.miss:
            raise ReplayMiss(f"replay miss: {request_key(body)}")
        cached = text in self.cached_inputs
        source = "cache" if cached else "live" if self.mode == "live" else "replay"
        usage: dict[str, Any] = {"prompt_tokens": 30, "completion_tokens": 3}
        if self.usage_cost is not None:
            usage["cost"] = self.usage_cost
        return TeacherResult(
            key=request_key(body),
            output=self.answer(text),
            response=completion(self.answer(text)),
            usage=usage,
            latency_ms=12.5,
            provider=self.provider,
            finish_reason="stop",
            created=1790000000.0,
            source=source,  # type: ignore[arg-type]
            cost_usd=0.0 if cached else self.cost,
        )

    async def stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        raise NotImplementedError
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        return None


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.sqlite")


# stage 1-2: load and extraction ---------------------------------------------------------------------------------


def test_load_counts_usable_captures_and_skips_the_rest_by_reason(store: Store) -> None:
    spec = make_spec()
    add_capture(store, spec, "Where is my card?", "card arrival")
    add_capture(store, spec, "Streamed request", None, captured=False, response_body=None, error="stream: not captured")
    add_capture(store, spec, "Server error", None, captured=False, status=500, response_body=None)
    add_capture(store, spec, "Odd status", "card arrival", status=203)
    add_capture(store, spec, "No body kept", None, response_body=None)
    store.add_imports(spec.task, "pairs", [{"input": "My card was stolen", "output": "lost or stolen card"}])
    store.add_imports(spec.task, "inputs", [{"input": "Exchange rate?"}, {"input": "Top up failed", "gold": "x"}])

    captures, imports, counts = load_sources(store, spec.task)

    assert [c.request_body and json.loads(c.request_body)["messages"][-1]["content"] for c in captures] == [
        "Where is my card?"
    ]
    assert len(imports) == 3
    assert counts["captures"]["total"] == 5
    assert counts["captures"]["usable"] == 1
    assert counts["captures"]["skipped"] == {"no_response_body": 1, "not_captured": 2, "status_not_200": 1}
    assert counts["imports"] == {"total": 3, "by_format": {"inputs": 2, "pairs": 1}}
    assert counts["captures"]["by_model"] == {TEACHER_MODEL: 1}
    assert counts["captures"]["other_model"] is None  # no spec given


def test_load_counts_captures_by_request_model_upstream_model_and_system_prompt(store: Store) -> None:
    spec = make_spec()
    add_capture(store, spec, "Where is my card?", "card arrival", upstream_model=TEACHER_MODEL + "-2026")
    add_capture(store, spec, "Is my card lost?", "lost or stolen card", upstream_model=TEACHER_MODEL + "-2026")
    other = make_spec(teacher={"model": "vendor/other-model", "max_tokens": 24})
    other.teacher_prompt = "A different prompt.\n"
    add_capture(store, other, "My card was stolen on the train", "lost or stolen card")
    no_system = {"model": TEACHER_MODEL, "messages": [{"role": "user", "content": "Rate for yen?"}]}
    store.add_capture(
        task=spec.task, source="proxy", request_key=request_key(no_system), request_body=json.dumps(no_system),
        response_body=json.dumps(completion("exchange rate")), status=200,
    )  # fmt: skip

    captures, _, counts = load_sources(store, spec.task, spec)

    assert len(captures) == 4  # captures from another model or prompt are kept, only counted
    c = counts["captures"]
    assert c["by_model"] == {TEACHER_MODEL: 3, "vendor/other-model": 1}
    assert c["by_upstream_model"] == {"(none)": 2, TEACHER_MODEL + "-2026": 2}
    other_sha = hashlib.sha256(b"A different prompt.\n").hexdigest()
    assert c["by_system_prompt"] == {spec.teacher_prompt_sha256: 2, other_sha: 1, "(none)": 1}
    assert c["other_model"] == 1
    assert c["other_prompt"] == 2


def test_extraction_uses_input_from_and_counts_unparsed_and_missing_outputs(store: Store) -> None:
    spec = make_spec(input={"from": "regex", "regex": r"Message: (?P<input>.+)"})
    good = build_teacher_request(spec, "Message: Where is my card?")
    store.add_capture(
        task=spec.task, source="proxy", request_key=request_key(good), request_body=json.dumps(good),
        response_body=json.dumps(completion("card arrival")), status=200,
    )  # fmt: skip
    unparsed = build_teacher_request(spec, "no prefix here")
    store.add_capture(
        task=spec.task, source="proxy", request_key=None, request_body=json.dumps(unparsed),
        response_body=json.dumps(completion("card arrival")), status=200,
    )  # fmt: skip
    missing = build_teacher_request(spec, "Message: Tool call only")
    store.add_capture(
        task=spec.task, source="import", request_key=None, request_body=json.dumps(missing),
        response_body=json.dumps(completion(None)), status=200,
    )  # fmt: skip
    store.add_capture(
        task=spec.task, source="proxy", request_key=None, request_body="{not json",
        response_body=json.dumps(completion("x")), status=200,
    )  # fmt: skip
    store.add_imports(spec.task, "inputs", [{"input": "Exchange rate?", "gold": "exchange_rate"}])

    records, counts = extract_records(spec, *load_sources(store, spec.task)[:2])

    assert [(r.raw_input, r.output, r.origin) for r in records] == [
        ("Where is my card?", "card arrival", "capture"),
        ("Exchange rate?", None, "inputs"),
    ]
    assert counts["input_unparsed"] == 1
    assert counts["missing_output"] == 1
    assert counts["request_invalid"] == 1
    assert counts["from_captures"] == 1
    assert counts["from_imports"] == 1


# stage 3: merge ---------------------------------------------------------------------------------------------------


def test_merge_joins_captured_output_with_imported_gold_by_input_hash() -> None:
    records = [
        Record("Where is my card?", "card arrival", None, {}, "capture"),
        Record("Where  is my card? ", None, "card_arrival", {"split": "train", "source": "crm"}, "inputs"),
        Record("Exchange rate?", None, "exchange_rate", {"split": "valid"}, "inputs"),
    ]
    examples, counts = merge_records(records, "meta.split")

    assert len(examples) == 2
    merged = next(ex for ex in examples if ex.input_hash == input_hash("Where is my card?"))
    assert merged.raw_input == "Where is my card?"  # the first record's raw input
    assert merged.outputs == ["card arrival"]
    assert merged.golds == ["card_arrival"]
    assert merged.meta == {"split": "train", "source": "crm"}
    assert merged.origins == ["capture", "inputs"]
    assert counts["with_both"] == 1
    assert counts["merged_records"] == 1
    assert [ex.input_hash for ex in examples] == sorted(ex.input_hash for ex in examples)


def test_merge_same_hash_pair_split_test_and_train_ends_in_test() -> None:
    records = [
        Record("How do I top up?", "top up failed", "top_up_failed", {"split": "train"}, "pairs"),
        Record("How do I top up?", None, "top_up_failed", {"split": "test"}, "inputs"),
        Record("Rate for euros?", None, None, {"split": "validation"}, "inputs"),
        Record("Rate for euros?", None, None, {"split": "train"}, "inputs"),
        Record("Card never came", None, None, {"split": "test"}, "inputs"),
        Record("Card never came", None, None, {}, "inputs"),
    ]
    examples, counts = merge_records(records, "meta.split")
    splits = {ex.raw_input: ex.meta["split"] for ex in examples}

    assert splits == {"How do I top up?": "test", "Rate for euros?": "valid", "Card never came": "test"}
    assert counts["cross_split_merged"] == 2  # a member without a split does not disagree


def test_merge_counts_exact_repeats_within_one_source_and_split() -> None:
    records = [
        # the same text twice in the train import and captured twice: one repeated input
        Record("Where is my card?", None, "card_arrival", {"split": "train"}, "inputs"),
        Record("Where is my card?", "card arrival", None, {}, "capture"),
        Record("Where is my card?", None, "card_arrival", {"split": "train"}, "inputs"),
        Record("Where is my card?", "card arrival", None, {}, "capture"),
        # train and test copies are a cross-split merge, not a repeat
        Record("How do I top up?", None, "top_up_failed", {"split": "train"}, "inputs"),
        Record("How do I top up?", None, "top_up_failed", {"split": "test"}, "inputs"),
        # a captured output joining imported gold is not a repeat
        Record("Rate for euros?", "exchange rate", None, {}, "capture"),
        Record("Rate for euros?", None, "exchange_rate", {"split": "valid"}, "inputs"),
        # three captures of one query: two repeats
        Record("Card never came", "card arrival", None, {}, "capture"),
        Record("Card never came", "card arrival", None, {}, "capture"),
        Record("Card never came", "card arrival", None, {}, "capture"),
    ]
    examples, counts = merge_records(records, "meta.split")
    repeats = {ex.raw_input: ex.repeats for ex in examples}
    assert repeats == {"Where is my card?": 1, "How do I top up?": 0, "Rate for euros?": 0, "Card never came": 2}
    assert counts["repeated_inputs"] == 3
    assert counts["cross_split_merged"] == 1


def test_merge_meta_conflicts_first_value_wins_and_are_counted() -> None:
    records = [
        Record("Card never came", None, None, {"group": "a", "traits": ["x"]}, "inputs"),
        Record("Card never came", None, None, {"group": "b", "traits": ["x"], "id": 7}, "inputs"),
    ]
    (ex,), counts = merge_records(records, "meta.split")
    assert ex.meta == {"group": "a", "traits": ["x"], "id": 7}
    assert counts["meta_conflicts"] == {"group": 1}


def test_nested_predefined_split_path() -> None:
    records = [
        Record("Card never came", None, None, {"info": {"split": "train", "k": 1}}, "inputs"),
        Record("Card never came", None, None, {"info": {"split": "test", "k": 1}}, "inputs"),
    ]
    (ex,), counts = merge_records(records, "meta.info.split")
    assert ex.meta == {"info": {"split": "test", "k": 1}}
    assert counts["meta_conflicts"] == {}
    assert counts["cross_split_merged"] == 1


def test_unknown_predefined_split_value_is_an_error_naming_it() -> None:
    with pytest.raises(CurateError, match="'holdout'"):
        merge_records([Record("Card never came", None, None, {"split": "holdout"}, "inputs")], "meta.split")
    assert canonical_split(" Dev ") == "valid"
    assert canonical_split("") is None
    with pytest.raises(CurateError, match="3"):
        canonical_split(3)


# stage 4: normalisation -----------------------------------------------------------------------------------------


def _normalised(spec: TaskSpec, outputs: list[str], golds: list[Any] | None = None) -> tuple[list[Example], dict]:
    ex = example("Where is my card?")
    ex.outputs = list(outputs)
    ex.golds = list(golds or [])
    return normalise_examples(spec, [ex])


def test_output_majority_rules() -> None:
    spec = make_spec()
    (ex,), counts = _normalised(spec, ["card arrival", "Card_Arrival.", "lost or stolen card"])
    assert ex.teacher == "card_arrival"
    assert ex.teacher_origin == "recorded"
    assert counts["conflicts_resolved"] == 1

    kept, counts = _normalised(spec, ["card arrival", "exchange rate"])
    assert kept == []
    assert counts["dropped_no_majority"] == 1

    (ex,), counts = _normalised(spec, ["no idea", "exchange rate"])
    assert ex.teacher == "exchange_rate"
    assert counts["invalid_outputs"] == 1


def test_all_invalid_outputs_drop_the_example_instead_of_relabelling_it() -> None:
    kept, counts = _normalised(make_spec(), ["no idea", "something else"])
    assert kept == []
    assert counts["dropped_all_invalid"] == 1
    assert counts["invalid_outputs"] == 2
    assert counts["unlabelled"] == 0


def test_examples_without_outputs_are_kept_for_labelling() -> None:
    (ex,), counts = _normalised(make_spec(), [], ["card arrival"])
    assert ex.teacher is None
    assert ex.gold == "card_arrival"
    assert counts["unlabelled"] == 1


def test_gold_is_normalised_and_conflicts_resolve_by_strict_majority() -> None:
    spec = make_spec()
    (ex,), counts = _normalised(spec, [], ["banana", 3])
    assert ex.gold is None
    assert counts["gold_invalid"] == 2

    (ex,), counts = _normalised(spec, [], ["card_arrival", "exchange rate"])
    assert ex.gold is None
    assert counts["gold_conflicts_unresolved"] == 1

    (ex,), counts = _normalised(spec, [], ["card_arrival", "Card Arrival", "exchange rate"])
    assert ex.gold == "card_arrival"
    assert counts["gold_conflicts_resolved"] == 1


def test_extraction_outputs_are_parsed_validated_and_ordered() -> None:
    spec = make_spec("extraction")
    fenced = (
        '```json\n{"total_amount": 12.5, "vendor_name": "Brala Ltd", "invoice_number": "A-1", "contact": null}\n```'
    )
    invalid = '{"vendor_name": "Brala Ltd"}'
    gold = {"contact": None, "total_amount": 12.5, "invoice_number": "A-1", "vendor_name": "Brala Ltd"}
    (ex,), counts = _normalised(spec, [fenced, invalid], [gold, {"vendor_name": 1}])
    assert list(ex.teacher) == ["vendor_name", "invoice_number", "total_amount", "contact"]
    assert ex.teacher == gold
    assert list(ex.gold) == ["vendor_name", "invoice_number", "total_amount", "contact"]
    assert counts["invalid_outputs"] == 1
    assert counts["gold_invalid"] == 1


# stage 5: PII -----------------------------------------------------------------------------------------------------


def test_pii_scrub_counts_hits_per_kind_and_field() -> None:
    spec = make_spec("extraction")
    value = {"vendor_name": "Brala Ltd", "invoice_number": "A-1", "total_amount": 1.0, "contact": "ap@example.com"}
    ex = example("Bill from ap@example.com, call 555-0142, card 4111 1111 1111 1111", teacher=value, gold=dict(value))
    plain = example("Nothing to hide here", teacher=dict(value, contact=None))
    stage = PiiStage(spec)
    counts = scrub_examples(stage, [ex, plain])

    assert ex.text == "Bill from <EMAIL>, call <PHONE>, card <CARD>"
    assert ex.raw_input.startswith("Bill from ap@example.com")  # the raw input is kept in memory
    assert ex.teacher["contact"] == "<EMAIL>"
    assert ex.gold["contact"] == "<EMAIL>"
    assert counts["by_field"]["input"] == {"email": 1, "iban": 0, "card": 1, "ssn": 0, "phone": 1, "ipv4": 0}
    assert counts["by_field"]["teacher"]["email"] == 1
    assert counts["by_field"]["gold"]["email"] == 1
    assert counts["hits"]["email"] == 3
    assert counts["total"] == 5
    assert counts["examples_changed"] == 1


def test_pii_examples_changed_counts_distinct_examples_across_repeated_scrubs() -> None:
    spec = make_spec("extraction")
    ex = example("Invoice A-1 from Brala Ltd, contact ap@example.com")
    stage = PiiStage(spec)
    assert scrub_examples(stage, [ex])["examples_changed"] == 1
    ex.teacher = {"vendor_name": "Brala Ltd", "invoice_number": "A-1", "total_amount": 1.0, "contact": "ap@example.com"}
    ex.teacher_origin = "labelled"
    hits = stage.scrub(ex, text=False, gold=False)  # the pipeline scrubs labelled values again

    assert hits == {"email": 1}
    assert ex.teacher["contact"] == "<EMAIL>"
    assert stage.counts()["examples_changed"] == 1
    assert stage.counts()["total"] == 2
    assert stage.scrub(example("nothing here"), text=True) == {}
    assert stage.counts()["examples_changed"] == 1


def test_pii_scrub_respects_configured_kinds_and_can_be_disabled() -> None:
    text = "Mail ap@example.com from 192.0.2.10"
    only_ip = make_spec(curate={"pii": {"kinds": ["ipv4"]}})
    ex = example(text)
    counts = scrub_examples(PiiStage(only_ip), [ex])
    assert ex.text == "Mail ap@example.com from <IPV4>"
    assert counts["hits"] == {"ipv4": 1}

    disabled = make_spec(curate={"pii": {"enabled": False}})
    ex = example(text, teacher="card_arrival")
    counts = scrub_examples(PiiStage(disabled), [ex])
    assert ex.text == text
    assert counts["enabled"] is False
    assert counts["total"] == 0


# stage 6: split ---------------------------------------------------------------------------------------------------


def _labelled_pool(n_per_label: int) -> list[Example]:
    pool = [example(f"message {label} number {i}", teacher=label) for label in LABELS for i in range(n_per_label)]
    return sorted(pool, key=lambda ex: ex.input_hash)


def test_predefined_splits_and_aliases_win_over_the_random_split() -> None:
    spec = make_spec()
    rows = [
        example("one two three", teacher="card_arrival", split=None),
        example("four five six", teacher="card_arrival", split=None),
    ]
    rows[0].meta["split"] = "dev"
    rows[1].meta["split"] = "test"
    counts = assign_splits(spec, rows)
    assert [ex.split for ex in rows] == ["valid", "test"]
    assert rows[0].meta["split"] == "valid"
    assert counts["method"] == "predefined"
    assert counts["predefined"] == {"train": 0, "valid": 1, "test": 1}

    rows[0].meta["split"] = "holdout"
    with pytest.raises(CurateError, match="holdout"):
        assign_splits(spec, rows)


def test_stratified_split_is_deterministic_and_proportional_per_label() -> None:
    spec = make_spec(curate={"split": {"val": 0.1, "test": 0.1, "seed": 5}})
    pool = _labelled_pool(100)
    counts = assign_splits(spec, pool)
    first = {ex.input_hash: ex.split for ex in pool}

    assert counts["method"] == "stratified"
    assert counts["splits"] == {"train": 320, "valid": 40, "test": 40}
    for label in LABELS:
        per = [ex.split for ex in pool if ex.teacher == label]
        assert 8 <= per.count("test") <= 12
        assert 8 <= per.count("valid") <= 12

    again = _labelled_pool(100)
    assign_splits(spec, again)
    assert {ex.input_hash: ex.split for ex in again} == first

    other_seed = _labelled_pool(100)
    assign_splits(make_spec(curate={"split": {"seed": 6}}), other_seed)
    assert {ex.input_hash: ex.split for ex in other_seed} != first


def test_stratified_split_falls_back_to_gold_then_unlabelled() -> None:
    spec = make_spec(curate={"split": {"val": 0.25, "test": 0.25}})
    rows = [example(f"gold only {i}", gold="exchange_rate") for i in range(8)]
    rows += [example(f"nothing at all {i}") for i in range(8)]
    assign_splits(spec, rows)
    for group in (rows[:8], rows[8:]):
        assert [ex.split for ex in group].count("test") == 2
        assert [ex.split for ex in group].count("valid") == 2


def test_grouped_split_keeps_whole_groups_and_is_deterministic() -> None:
    spec = make_spec("extraction", curate={"split": {"val": 0.2, "test": 0.2, "group_by": "meta.template"}})

    def pool() -> list[Example]:
        rows = [example(f"document {g} {i}", template=f"t{g:02d}") for g in range(20) for i in range(10)]
        rows.append(example("a document without a template"))
        return sorted(rows, key=lambda ex: ex.input_hash)

    rows = pool()
    counts = assign_splits(spec, rows)
    by_template: dict[str, set[str | None]] = {}
    for ex in rows:
        by_template.setdefault(ex.meta.get("template", ex.input_hash), set()).add(ex.split)
    assert all(len(splits) == 1 for splits in by_template.values())
    assert counts["method"] == "grouped"
    assert counts["groups"] == 21
    assert 30 <= counts["splits"]["test"] <= 50
    assert 30 <= counts["splits"]["valid"] <= 50

    again = pool()
    assign_splits(spec, again)
    assert {ex.input_hash: ex.split for ex in again} == {ex.input_hash: ex.split for ex in rows}


def test_grouped_split_with_few_large_groups_leaves_no_split_empty() -> None:
    spec = make_spec("extraction", curate={"split": {"val": 0.1, "test": 0.1, "group_by": "meta.template"}})
    rows = [example(f"document {g} line {i}", template=f"t{g}") for g in range(5) for i in range(100)]
    counts = assign_splits(spec, rows)
    assert counts["splits"] == {"train": 300, "valid": 100, "test": 100}
    assert all(len({ex.split for ex in rows if ex.meta["template"] == f"t{g}"}) == 1 for g in range(5))

    two = [example(f"document {g} line {i}", template=f"t{g}") for g in range(2) for i in range(50)]
    assert assign_splits(spec, two)["splits"] == {"train": 50, "valid": 0, "test": 50}  # train keeps a group
    one = [example(f"document line {i}", template="t0") for i in range(50)]
    assert assign_splits(spec, one)["splits"] == {"train": 50, "valid": 0, "test": 0}


def test_fill_groups_takes_midpoints_inside_the_target_and_never_nothing() -> None:
    sizes = {"a": 100, "b": 30, "c": 10, "d": 60}
    assert fill_groups(["a", "b", "c", "d"], sizes, 40, keep=1) == ["b", "c"]
    assert fill_groups(["a", "d"], sizes, 20, keep=1) == ["d"]  # none fits: the smallest group
    assert fill_groups(["a", "b"], sizes, 500, keep=1) == ["a"]  # one group is left for train
    assert fill_groups(["a"], sizes, 20, keep=1) == []
    assert fill_groups(["a", "b"], sizes, 0, keep=0) == []


def _pool_with_one_more(kind: str, grouped: bool, extra: int) -> tuple[list[Example], list[Example]]:
    def rows(n: int) -> list[Example]:
        rng = random.Random(1)
        out = []
        for i in range(n):
            label = LABELS[rng.randrange(len(LABELS))]
            meta = {"template": f"t{i % 37}"} if grouped else {}
            teacher = label if kind == "classification" else None
            out.append(example(f"message about {label} number {i}", teacher=teacher, **meta))
        return out

    before, after = rows(1000), rows(1000)
    label = LABELS[extra % len(LABELS)]
    meta = {"template": f"t{extra}"} if grouped else {}
    after.append(example(f"a new message {extra}", teacher=label if kind == "classification" else None, **meta))
    return before, after


@pytest.mark.parametrize(
    ("kind", "extra_cfg", "grouped"),
    [("classification", {}, False), ("extraction", {}, False), ("extraction", {"group_by": "meta.template"}, True)],
    ids=["stratified", "random", "grouped"],
)
def test_adding_a_row_moves_at_most_boundary_rows_and_never_train_to_test(
    kind: str, extra_cfg: dict[str, Any], grouped: bool
) -> None:
    spec = make_spec(kind, curate={"split": {"val": 0.1, "test": 0.1, **extra_cfg}})
    for extra in range(8):
        before, after = _pool_with_one_more(kind, grouped, extra)
        assign_splits(spec, before)
        assign_splits(spec, after)
        old = {ex.input_hash: ex.split for ex in before}
        moves = Counter((old[ex.input_hash], ex.split) for ex in after[:-1] if old[ex.input_hash] != ex.split)
        assert sum(moves.values()) <= 3, moves
        assert moves[("train", "test")] == 0


def test_extraction_without_groups_uses_a_seeded_hash_order() -> None:
    spec = make_spec("extraction", curate={"split": {"val": 0.1, "test": 0.2}})
    rows = [example(f"invoice text number {i}") for i in range(50)]
    counts = assign_splits(spec, rows)
    assert counts["method"] == "random"
    assert counts["splits"] == {"train": 35, "valid": 5, "test": 10}
    assert counts["targets"] == {"valid": 5, "test": 10}
    shuffled = [example(f"invoice text number {i}") for i in reversed(range(50))]
    assign_splits(spec, shuffled)
    assert {ex.input_hash: ex.split for ex in shuffled} == {ex.input_hash: ex.split for ex in rows}


# stage 7: dedupe --------------------------------------------------------------------------------------------------


def test_dedupe_matches_texts_under_three_words_exactly_only() -> None:
    rows = [
        example("Top up", teacher="top_up_failed"),
        example("top  UP", teacher="top_up_failed"),
        example("top up please", teacher="top_up_failed"),
        example("top-up", teacher="top_up_failed"),
    ]
    kept, counts = dedupe_split(rows, 0.9)
    assert [ex.text for ex in kept] == ["Top up", "top up please", "top-up"]
    assert counts["removed_exact"] == 1
    assert counts["removed_near"] == 0

    kept, counts = dedupe_split(rows, 0.9, exact=False)
    assert len(kept) == 4


def test_dedupe_keeps_lowest_index_member_with_the_majority_value_and_majority_gold() -> None:
    base = "my new card has still not arrived after two whole weeks of waiting at home"
    rows = [
        example(base, teacher="lost_or_stolen_card", gold="card_arrival"),
        example(base + " today", teacher="card_arrival", gold="card_arrival"),
        example(base.upper(), teacher="card_arrival", gold=None),
        example(base + " now", teacher=None, gold="lost_or_stolen_card"),
        example("something unrelated entirely about exchange rates", teacher="exchange_rate"),
    ]
    kept, counts = dedupe_split(rows, 0.8)
    assert [ex.text for ex in kept] == [base + " today", "something unrelated entirely about exchange rates"]
    assert kept[0].teacher == "card_arrival"
    assert kept[0].gold == "card_arrival"  # 2 of 3 golds
    assert counts["removed_near"] == 3
    assert counts["removed_exact"] == 0
    assert counts["conflict_clusters"] == 0


def test_dedupe_drops_a_cluster_without_a_majority_and_counts_exact_removals() -> None:
    rows = [
        example("Where is my card?", teacher="card_arrival"),
        example("where is my card?", teacher="lost_or_stolen_card"),
        example("What is the exchange rate?", teacher="exchange_rate"),
        example("WHAT IS THE EXCHANGE RATE?", teacher=None),
    ]
    kept, counts = dedupe_split(rows, 0.9)
    assert [ex.text for ex in kept] == ["What is the exchange rate?"]
    assert counts["conflict_clusters"] == 1
    assert counts["conflict_examples"] == 2
    assert counts["removed_exact"] == 1


def test_dedupe_gold_conflict_without_majority_keeps_the_survivors_own_gold() -> None:
    rows = [
        example("What are the fees for top-ups?", "top_up_failed", gold="exchange_rate"),
        example("What are the fees for top ups?", "top_up_failed", gold="top_up_failed"),
    ]
    kept, counts = dedupe_split(rows, 0.9)
    assert [ex.text for ex in kept] == ["What are the fees for top-ups?"]
    assert kept[0].gold == "exchange_rate"  # a tie: the kept row's own gold, never None
    assert counts["gold_conflicts"] == 1

    rows = [
        example("What are the fees for top-ups?", "top_up_failed", gold=None),
        example("What are the fees for top ups?", "top_up_failed", gold="top_up_failed"),
        example("what are the fees for top-ups?", "top_up_failed", gold="top_up_failed"),
    ]
    kept, counts = dedupe_split(rows, 0.9)
    assert kept[0].gold == "top_up_failed"  # a survivor without gold takes the majority
    assert counts["gold_conflicts"] == 0


def test_dedupe_split_reports_the_repeats_merged_before_it() -> None:
    rows = [example("Where is my card?", "card_arrival"), example("Rate for yen?", "exchange_rate")]
    rows[0].repeats = 2
    _, counts = dedupe_split(rows, 0.9)
    assert counts["repeated_inputs"] == 2


def test_dedupe_of_unlabelled_cluster_keeps_the_first_member() -> None:
    rows = [example("Exchange rate for dollars?", gold="exchange_rate"), example("exchange rate for dollars?")]
    kept, _ = dedupe_split(rows, 0.9)
    assert [ex.text for ex in kept] == ["Exchange rate for dollars?"]
    assert kept[0].gold == "exchange_rate"


def test_dedupe_within_runs_per_split() -> None:
    rows = [example("Where is my card?", "card_arrival", split=s) for s in ("train", "valid", "test")]
    rows.append(example("where is my card?", "card_arrival", split="test"))
    rows[0].repeats = 1
    splits, counts = dedupe_within(by_split(rows), 0.9)
    assert {name: len(v) for name, v in splits.items()} == {"train": 1, "valid": 1, "test": 1}
    assert counts["removed_exact"] == 1
    assert counts["splits"]["test"]["before"] == 2
    assert counts["repeated_inputs"] == 1
    assert counts["splits"]["train"]["repeated_inputs"] == 1


# stage 8: cross-split ---------------------------------------------------------------------------------------------


def test_cross_split_removal_never_touches_test() -> None:
    long = "the card I ordered last month has still not been delivered to my address"
    train = [
        example("Where is my card?", "card_arrival", split="train"),
        example(long + " yet", "card_arrival", split="train"),
        example("I want to top up by bank transfer", "top_up_failed", split="train"),
        example("A train only question about fees", "exchange_rate", split="train"),
    ]
    valid = [
        example("where is my card?", "card_arrival", split="valid"),
        example("What rate do you use for yen", "exchange_rate", split="valid"),
    ]
    test = [
        example(long, "card_arrival", split="test"),
        example("What rate do you use for yen", "exchange_rate", split="test"),
        example("I want to top up by bank transfer", "top_up_failed", split="test"),
    ]
    splits, counts = remove_cross_split({"train": train, "valid": valid, "test": test}, 0.8)

    assert [ex.text for ex in splits["train"]] == ["A train only question about fees"]
    assert splits["valid"] == [valid[0]]
    assert splits["test"] == test
    assert counts["removed"] == {"train": 3, "valid": 1, "test": 0}
    assert counts["pairs"]["train/valid"] == {"exact": 1, "near": 0}
    assert counts["pairs"]["train/test"] == {"exact": 1, "near": 1}
    assert counts["pairs"]["valid/test"] == {"exact": 1, "near": 0}


# stage 9: labelling -----------------------------------------------------------------------------------------------


def _answer(text: str) -> str:
    return "Card Arrival" if "card" in text.lower() else "exchange rate"


def test_labelling_sends_the_raw_input_through_the_shared_request_builder() -> None:
    spec = make_spec()
    raw = "Email ap@example.com: my card has not arrived"
    ex = example(raw, split="train")
    scrub_examples(PiiStage(spec), [ex])
    assert ex.text == "Email <EMAIL>: my card has not arrived"
    done = example("Exchange rate?", teacher="exchange_rate", split="valid")
    teacher = FakeTeacher(_answer)

    kept, stats, keys = label_examples(spec, [ex, done], teacher, log=lambda _: None)

    expected = build_teacher_request(spec, raw)
    assert teacher.bodies == [expected]
    assert keys == [request_key(expected)]
    assert kept == [ex, done]
    assert ex.teacher == "card_arrival"
    assert ex.teacher_origin == "labelled"
    assert stats["requested"] == 1
    assert stats["replayed"] == 1
    assert stats["labelled"] == 1
    assert stats["date"] == "2026-09-21"


def test_projection_gate_refuses_before_sending_more_than_the_sample() -> None:
    spec = make_spec()
    rows = [example(f"card question number {i}", split="train") for i in range(100)]
    teacher = FakeTeacher(_answer, mode="live", cost=0.02)
    lines: list[str] = []

    with pytest.raises(SpendNotConfirmed, match=r"\$2\.00"):
        label_examples(spec, rows, teacher, log=lines.append)
    assert len(teacher.bodies) == 50
    assert any("projected teacher cost (live): $2.0000 for 100 uncached" in line for line in lines)

    teacher = FakeTeacher(_answer, mode="live", cost=0.02)
    _, stats, keys = label_examples(spec, rows, teacher, yes=True, max_usd=3.0, log=lambda _: None)
    assert len(teacher.bodies) == 100
    assert len(keys) == 100
    assert stats["projection"]["projected_usd"] == pytest.approx(2.0)
    assert stats["cost_usd"] == pytest.approx(2.0)
    assert stats["live"] == 100
    assert stats["max_usd"] == 3.0


def test_projection_samples_only_uncached_requests() -> None:
    spec = make_spec()
    rows = [example(f"card question number {i}", split="train") for i in range(40)]
    cached = {ex.raw_input for ex in rows[:20]}
    teacher = FakeTeacher(_answer, mode="live", cost=0.001, cached=cached)

    _, stats, _ = label_examples(spec, rows, teacher, projection_sample=5, log=lambda _: None)

    first = [body["messages"][-1]["content"] for body in teacher.bodies[:5]]
    assert first == [ex.raw_input for ex in rows[20:25]]
    assert stats["projection"]["n_total"] == 20
    assert stats["projection"]["projected_usd"] == pytest.approx(0.02)
    assert stats["cached"] == 20
    assert stats["live"] == 20
    assert stats["cost_usd"] == pytest.approx(0.02)


def test_small_batch_inside_the_sample_needs_no_confirmation() -> None:
    rows = [example(f"card question number {i}", split="train") for i in range(10)]
    teacher = FakeTeacher(_answer, mode="live", cost=0.1)
    _, stats, _ = label_examples(make_spec(), rows, teacher, log=lambda _: None)
    assert stats["projection"]["projected_usd"] == pytest.approx(1.0)
    assert len(teacher.bodies) == 10


def test_invalid_teacher_answers_drop_train_and_valid_rows_but_keep_test_rows() -> None:
    rows = [
        example("Where is my card?", split="test"),
        example("Something odd", split="test"),
        example("Something odd for training", split="train"),
        example("Something odd for validation", split="valid"),
    ]
    teacher = FakeTeacher(lambda text: "card arrival" if "card" in text else "I cannot say")
    kept, stats, keys = label_examples(make_spec(), rows, teacher, log=lambda _: None)
    assert kept == rows[:2]
    assert rows[1].teacher is None
    assert rows[1].teacher_origin == "invalid"
    assert stats["invalid"] == 3
    assert stats["invalid_by_split"] == {"train": 1, "valid": 1, "test": 1}
    assert stats["kept_invalid_test"] == 1
    assert stats["labelled"] == 1
    assert len(keys) == 4


def test_labelling_records_the_original_cost_of_replayed_and_cached_answers_and_the_providers() -> None:
    def rows() -> list[Example]:
        return [example(f"card question number {i}", split="train") for i in range(4)]

    teacher = FakeTeacher(_answer, usage_cost=0.003, provider="Beta")
    _, stats, _ = label_examples(make_spec(), rows(), teacher, log=lambda _: None)
    assert stats["replayed"] == 4
    assert stats["cost_usd"] == 0.0
    assert stats["recorded_cost_usd"] == pytest.approx(0.012)
    assert stats["providers"] == {"Beta": 4}

    cached = FakeTeacher(_answer, mode="live", cached={ex.raw_input for ex in rows()}, usage_cost=0.001)
    _, stats, _ = label_examples(make_spec(), rows(), cached, log=lambda _: None)
    assert stats["cached"] == 4
    assert stats["cost_usd"] == 0.0
    assert stats["recorded_cost_usd"] == pytest.approx(0.004)

    _, stats, _ = label_examples(make_spec(), rows(), FakeTeacher(_answer), log=lambda _: None)
    assert stats["recorded_cost_usd"] is None


def test_labelling_without_a_teacher_is_a_clear_error() -> None:
    rows = [example("Where is my card?", split="test"), example("Exchange?", teacher="exchange_rate", split="train")]
    with pytest.raises(CurateError, match="1 example"):
        label_examples(make_spec(), rows, None)
    kept, stats, keys = label_examples(make_spec(), rows[1:], None)
    assert kept == rows[1:]
    assert stats["requested"] == 0
    assert keys == []


def test_replay_miss_propagates() -> None:
    rows = [example("Where is my card?", split="test")]
    with pytest.raises(ReplayMiss):
        label_examples(make_spec(), rows, FakeTeacher(_answer, miss={"Where is my card?"}), log=lambda _: None)


# stage 10: length -------------------------------------------------------------------------------------------------


def test_length_filter_drops_long_examples_and_never_truncates() -> None:
    spec = make_spec(train={"max_seq_len": 20})
    short = example("Where is my card?", teacher="card_arrival", split="train")
    long = example(" ".join(["word"] * 30), teacher="card_arrival", split="train")
    edge = example(" ".join(["word"] * 11), teacher="card_arrival", split="test")
    tokenizer = WordTokenizer()

    splits, counts = length_filter(spec, {"train": [short, long], "valid": [], "test": [edge]}, tokenizer)

    assert splits == {"train": [short], "valid": [], "test": [edge]}
    assert edge.length == 20
    assert long.length == 39
    assert counts["dropped"] == 1
    assert counts["splits"]["train"] == {
        "n": 2, "kept": 1, "dropped": 1, "p50": 26.0, "p95": 37.7, "max": 39, "max_kept": 13,
    }  # fmt: skip
    assert counts["splits"]["valid"]["p95"] is None
    assert all(call["tokenize"] is True and call["return_dict"] is False for call in tokenizer.calls)
    assert tokenizer.calls[0]["messages"] == training_messages(spec, "Where is my card?", "card_arrival")
    assert tokenizer.calls[1]["messages"][1]["content"] == long.text  # the whole input is measured


def test_a_test_row_without_a_teacher_value_is_measured_with_an_empty_answer() -> None:
    spec = make_spec()
    row = example("Something odd", split="test")
    row.teacher_origin = "invalid"
    tokenizer = WordTokenizer()
    splits, _ = length_filter(spec, {"train": [], "valid": [], "test": [row]}, tokenizer)
    assert splits["test"] == [row]
    assert tokenizer.calls[0]["messages"][2] == {"role": "assistant", "content": ""}


# dataset card -----------------------------------------------------------------------------------------------------


def test_card_uses_card_info_and_defaults() -> None:
    stats = {
        "task": "intents",
        "task_type": "classification",
        "date": "2026-09-26T10:00:00+00:00",
        "splits": {"train": 2, "valid": 1, "test": 1},
        "teacher": {"model": TEACHER_MODEL, "provider": "alpha/fp8", "prompt_sha256": "ab" * 32},
        "labelling": {"requested": 2, "live": 0, "cached": 0, "replayed": 2, "cost_usd": 0.0, "date": "2026-09-21"},
        "leakage": {"ok": True, "exact": 0, "near": 0, "threshold": 0.9},
    }
    card = render_card(stats)
    assert "- Source: user-provided data" in card
    assert "- Licence: not recorded" in card
    assert f"`{TEACHER_MODEL}`" in card
    assert "- Provider: alpha/fp8" in card
    assert "ab" * 32 in card
    assert "2 replayed from a recording" in card

    assert card.startswith("# Dataset card: intents\n")

    info = {
        "name": "Example corpus (demo)",
        "source": "Example corpus",
        "licence": "CC BY 4.0",
        "attribution": "Made by Example Org.",
        "citation": "@misc{x}",
        "split_rule": "official test set -> test; the rest 90/10 train/valid.",
    }
    card = render_card(stats, info)
    assert card.startswith("# Dataset card: Example corpus (demo)\n")
    assert "for task `intents`" in card
    assert "- Licence: CC BY 4.0" in card
    assert "Made by Example Org." in card
    assert "@misc{x}" in card
    assert (
        "Predefined split rule (from the data source): official test set -> test; the rest 90/10 train/valid.\n" in card
    )
    assert unknown_card_keys(info) == []
    assert unknown_card_keys({"licenze": "typo", "source": "x"}) == ["licenze"]
    assert "typo" not in render_card(stats, {"licenze": "typo"})  # unknown keys are ignored


def test_card_lists_split_sizes_repeats_gold_conflicts_and_labelling_cost() -> None:
    stats = {
        "task": "intents",
        "task_type": "classification",
        "date": "2026-09-26T10:00:00+00:00",
        "splits": {"train": 2, "valid": 1, "test": 2},
        "split_sizes": {
            "split": {"train": 4, "valid": 1, "test": 3},
            "dedupe": {"train": 3, "valid": 1, "test": 2},
            "length": {"train": 2, "valid": 1, "test": 2},
        },
        "sources": {"test": {"recorded": 0, "labelled": 1, "teacher_invalid": 1, "with_gold": 2}},
        "dedupe": {
            "threshold": 0.9,
            "repeated_inputs": 4,
            "gold_conflicts": 1,
            "splits": {"train": {"repeated_inputs": 3}, "test": {"repeated_inputs": 1}},
        },
        "stages": {
            "load": {
                "captures": {
                    "usable": 3,
                    "total": 3,
                    "by_model": {TEACHER_MODEL: 2, "vendor/other": 1},
                    "by_upstream_model": {"(none)": 3},
                    "by_system_prompt": {"ab" * 32: 2, "cd" * 32: 1},
                    "other_model": 1,
                    "other_prompt": 1,
                }
            }
        },
        "teacher": {"model": TEACHER_MODEL, "provider": None, "prompt_sha256": "ab" * 32},
        "labelling": {
            "requested": 3,
            "live": 1,
            "cached": 0,
            "replayed": 2,
            "cost_usd": 0.5,
            "recorded_cost_usd": 0.25,
            "providers": {"Alpha": 3},
            "date": "2026-09-21",
            "invalid": 2,
            "kept_invalid_test": 1,
            "truncated": 0,
        },
        "leakage": {"ok": True, "exact": 0, "near": 0, "threshold": 0.9},
    }
    card = render_card(stats)
    assert "| dedupe within splits | 3 | 1 | 2 |" in card
    assert "| length filter | 2 | 1 | 2 |" in card
    assert "| test | 2 | 0 | 1 | 1 | 2 |" in card
    assert "merged into one example: 4 (train 3, valid 0, test 1)" in card
    assert "1 clusters with conflicting gold labels" in card
    assert "cost $0.5000 in this run, originally $0.2500 when the replayed or cached answers were produced" in card
    assert "served by Alpha 3" in card
    assert "2 invalid answers (1 train/valid examples dropped, 1 test examples kept without a teacher value)" in card
    assert "request models: `vendor/teacher-a` 2, `vendor/other` 1" in card
    assert "1 captures asked another model than `vendor/teacher-a`" in card
    assert "1 captures used another system prompt" in card
