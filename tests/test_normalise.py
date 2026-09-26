from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from taskdistill.config import TaskSpec
from taskdistill.curate.normalise import normalise_output
from taskdistill.tasks.classification import label_key, normalise_label
from taskdistill.tasks.extraction import canonical_output, normalise_extraction, parse_json_output, validate

LABELS = [
    "card_arrival",
    "card_arrival_time",
    "cash_withdrawal_charge",
    "beneficiary_not_allowed",
    "Refund_not_showing_up",
    "reverted_card_payment?",
    "top_up_by_card_charge",
    "verify_3ds_payment",
    "why_can't_i_pay",
]

INVOICE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "vendor_name": {"type": "string"},
        "invoice_number": {"type": "string"},
        "invoice_date": {"type": "string", "format": "date"},
        "due_date": {"type": ["string", "null"], "format": "date"},
        "currency": {"enum": ["EUR", "USD", "GBP", "CHF", "PLN"]},
        "total_amount": {"type": "number"},
        "tax_amount": {"type": ["number", "null"]},
        "po_number": {"type": ["string", "null"]},
    },
    "required": [
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "due_date",
        "currency",
        "total_amount",
        "tax_amount",
        "po_number",
    ],
    "additionalProperties": False,
}

INVOICE = {
    "vendor_name": "Brask Lumen GmbH",
    "invoice_number": "INV-2026-0413",
    "invoice_date": "2026-03-14",
    "due_date": None,
    "currency": "EUR",
    "total_amount": 1234.56,
    "tax_amount": 197.11,
    "po_number": "PO-58213",
}
CANONICAL = (
    '{"vendor_name":"Brask Lumen GmbH","invoice_number":"INV-2026-0413","invoice_date":"2026-03-14",'
    '"due_date":null,"currency":"EUR","total_amount":1234.56,"tax_amount":197.11,"po_number":"PO-58213"}'
)
# the same document with keys in reverse order and pretty-printed
REVERSED = json.dumps(dict(reversed(list(INVOICE.items()))), indent=2)


# --- classification ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("card_arrival", "card_arrival"),
        ("Card Arrival", "card_arrival"),
        ("  card   arrival \n", "card_arrival"),
        ("card-arrival", "card_arrival"),
        ("card - arrival", "card_arrival"),
        ('"card arrival"', "card_arrival"),
        ("`card_arrival`", "card_arrival"),
        ("**card arrival**", "card_arrival"),
        ("Label: card arrival.", "card_arrival"),
        ("**Label:** card arrival", "card_arrival"),
        ("**Intent**: `card_arrival`", "card_arrival"),
        ("category: 'card arrival'.", "card_arrival"),
        ("“Card Arrival”", "card_arrival"),
        ("ｃａｒｄ ａｒｒｉｖａｌ", "card_arrival"),  # NFKC folds full-width letters
        ("why can’t i pay", "why_can't_i_pay"),
        ("Reverted card payment?", "reverted_card_payment?"),
        ("VERIFY 3DS PAYMENT", "verify_3ds_payment"),
        ("card arrival..", "card_arrival."),  # only one trailing period is dropped
        ("", ""),
    ],
)
def test_label_key(text: str, key: str) -> None:
    assert label_key(text) == key


@pytest.mark.parametrize(
    "output",
    [
        "card arrival",
        "Card Arrival",
        "card_arrival",
        "`card_arrival`",
        "Label: card arrival.",
        "**card arrival**",
        "card arrival\n",
        '"card arrival"',
        "CARD ARRIVAL",
        "Intent: card-arrival",
        "category: Card_Arrival",
    ],
)
def test_normalise_label_teacher_variants(output: str) -> None:
    assert normalise_label(output, LABELS) == "card_arrival"


@pytest.mark.parametrize(
    ("output", "label"),
    [
        ("card arrival time", "card_arrival_time"),
        ("cash withdrawal charge", "cash_withdrawal_charge"),
        ("Beneficiary not allowed.", "beneficiary_not_allowed"),
        ("refund not showing up", "Refund_not_showing_up"),
        ("Refund_not_showing_up", "Refund_not_showing_up"),
        ("reverted card payment?", "reverted_card_payment?"),
        ("reverted card payment", "reverted_card_payment?"),  # terminal punctuation is not significant
        ("Reverted card payment.", "reverted_card_payment?"),
        ("top up by card charge", "top_up_by_card_charge"),
        ("verify 3ds payment", "verify_3ds_payment"),
        ("why can't I pay", "why_can't_i_pay"),
        ("Why can’t I pay?", "why_can't_i_pay"),
    ],
)
def test_normalise_label_other_labels(output: str, label: str) -> None:
    assert normalise_label(output, LABELS) == label


@pytest.mark.parametrize(
    "output",
    [
        "",
        "   ",
        "card arival",  # no fuzzy matching
        "card",
        "arrival",
        "card arrival please",
        "The label is card arrival",
        "card arrival, card arrival time",
        "card_arrival\nExplanation: the card has not arrived",
        "verify 3d payment",
        "why cant i pay",
        "unknown",
    ],
)
def test_normalise_label_no_match(output: str) -> None:
    assert normalise_label(output, LABELS) is None


def test_normalise_label_requires_a_unique_match() -> None:
    assert normalise_label("card arrival", ["card-arrival", "card_arrival", "other"]) is None
    assert normalise_label("card arrival", ["Card_Arrival", "card_arrival"]) is None
    # an exact key wins over a punctuation-insensitive one
    assert normalise_label("foo", ["foo", "foo?"]) == "foo"
    assert normalise_label("foo?", ["foo", "foo?"]) == "foo?"
    assert normalise_label("foo!", ["foo", "foo?"]) is None


def test_normalise_label_accepts_any_sequence() -> None:
    assert normalise_label("card arrival", tuple(LABELS)) == "card_arrival"
    assert normalise_label("card arrival", []) is None


# --- extraction: parsing -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        CANONICAL,
        REVERSED,
        f"```json\n{REVERSED}\n```",
        f"```\n{CANONICAL}\n```",
        f"Here is the extracted data:\n```json\n{CANONICAL}\n```\nLet me know if you need more.",
        f"Sure! {CANONICAL} Hope this helps.",
        f"Note {{see below}}:\n```json\n{CANONICAL}\n```",
        f"{CANONICAL}\n(Values in {{curly}} braces were inferred.)",
    ],
)
def test_parse_json_output_finds_the_object(text: str) -> None:
    assert parse_json_output(text) == INVOICE


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json here",
        "[1, 2, 3]",
        '"just a string"',
        "{not json}",
        '{"a": 1',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": 1e999}',
        "} {",
    ],
)
def test_parse_json_output_rejects(text: str) -> None:
    assert parse_json_output(text) is None


def test_parse_json_output_keeps_unicode_and_nesting() -> None:
    assert parse_json_output('{"name": "Zürich Ølstue", "n": {"x": [1, 2]}}') == {
        "name": "Zürich Ølstue",
        "n": {"x": [1, 2]},
    }


# --- extraction: validation ----------------------------------------------------------------------


def test_validate_accepts_a_valid_invoice() -> None:
    assert validate(INVOICE, INVOICE_SCHEMA) == []
    assert validate({**INVOICE, "due_date": "2026-04-13", "tax_amount": None, "po_number": None}, INVOICE_SCHEMA) == []
    assert validate({**INVOICE, "total_amount": 1234}, INVOICE_SCHEMA) == []


@pytest.mark.parametrize(
    ("change", "path"),
    [
        ({"invoice_date": "2026-02-30"}, "$.invoice_date"),  # format date enforced: no 30 February
        ({"invoice_date": "14.03.2026"}, "$.invoice_date"),
        ({"invoice_date": "20260314"}, "$.invoice_date"),
        ({"due_date": "Net 30"}, "$.due_date"),
        ({"currency": "JPY"}, "$.currency"),
        ({"currency": "eur"}, "$.currency"),
        ({"total_amount": "1234.56"}, "$.total_amount"),
        ({"total_amount": True}, "$.total_amount"),
        ({"total_amount": None}, "$.total_amount"),
        ({"tax_amount": "197.11"}, "$.tax_amount"),
        ({"po_number": 58213}, "$.po_number"),
        ({"vendor_name": None}, "$.vendor_name"),
    ],
)
def test_validate_reports_field_errors(change: dict[str, Any], path: str) -> None:
    errors = validate({**INVOICE, **change}, INVOICE_SCHEMA)
    assert len(errors) == 1
    assert errors[0].startswith(f"{path}: ")


def test_validate_required_and_additional_properties() -> None:
    doc = {k: v for k, v in INVOICE.items() if k != "po_number"}
    doc["notes"] = "x"
    errors = validate(doc, INVOICE_SCHEMA)
    assert errors == [
        "$: 'po_number' is a required property",
        "$: Additional properties are not allowed ('notes' was unexpected)",
    ]
    assert errors == sorted(errors)


def test_validate_errors_are_sorted() -> None:
    errors = validate({**INVOICE, "total_amount": "x", "currency": "JPY", "invoice_date": "x"}, INVOICE_SCHEMA)
    assert [e.split(":")[0] for e in errors] == ["$.currency", "$.invoice_date", "$.total_amount"]
    assert errors == sorted(errors)


def test_validate_non_object() -> None:
    assert validate([], INVOICE_SCHEMA) == ["$: [] is not of type 'object'"]


# --- extraction: canonical output ----------------------------------------------------------------


def test_canonical_output_orders_by_schema_and_is_compact() -> None:
    assert canonical_output(dict(reversed(list(INVOICE.items()))), INVOICE_SCHEMA) == CANONICAL


def test_canonical_output_drops_unknown_keys_and_keeps_unicode() -> None:
    obj = {"extra": 1, "currency": "CHF", "vendor_name": "Zürich Ølstue AG"}
    assert canonical_output(obj, INVOICE_SCHEMA) == '{"vendor_name":"Zürich Ølstue AG","currency":"CHF"}'


def test_canonical_output_orders_nested_objects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "lines": {
                "type": "array",
                "items": {"type": "object", "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}}},
            },
            "meta": {"type": "object"},
        },
    }
    obj = {"meta": {"z": 1, "a": 2}, "lines": [{"qty": 2, "sku": "A-1", "x": 0}], "id": "7"}
    assert canonical_output(obj, schema) == '{"id":"7","lines":[{"sku":"A-1","qty":2}],"meta":{"z":1,"a":2}}'


def test_normalise_extraction_round_trip() -> None:
    text = f"```json\n{REVERSED}\n```"
    obj, canonical = normalise_extraction(text, INVOICE_SCHEMA)
    assert obj == INVOICE
    assert obj is not None and list(obj) == list(INVOICE_SCHEMA["properties"])
    assert canonical == CANONICAL
    assert normalise_extraction(CANONICAL, INVOICE_SCHEMA) == (INVOICE, CANONICAL)


@pytest.mark.parametrize(
    "text",
    [
        "I could not find an invoice.",
        CANONICAL.replace("2026-03-14", "2026-13-14"),
        CANONICAL.replace('"EUR"', '"EURO"'),
        CANONICAL.replace(',"po_number":"PO-58213"', ""),
        CANONICAL.replace("}", ',"notes":"x"}'),
        CANONICAL[:-1],
    ],
)
def test_normalise_extraction_invalid(text: str) -> None:
    assert normalise_extraction(text, INVOICE_SCHEMA) == (None, None)


def test_schema_is_not_mutated() -> None:
    before = copy.deepcopy(INVOICE_SCHEMA)
    normalise_extraction(CANONICAL, INVOICE_SCHEMA)
    validate({}, INVOICE_SCHEMA)
    assert before == INVOICE_SCHEMA


# --- normalise_output ----------------------------------------------------------------------------


def _spec(task_type: str) -> TaskSpec:
    raw: dict[str, Any] = {
        "task": f"fixture-{task_type}",
        "type": task_type,
        "teacher": {"model": "example/teacher-small"},
        "student": {"system_prompt": "Answer."},
        "cascade": {"target": 0.97},
    }
    if task_type == "classification":
        raw["labels_file"] = "labels.txt"
    else:
        raw["schema_file"] = "schema.json"
    return TaskSpec.model_validate(raw)


def test_normalise_output_classification() -> None:
    spec = _spec("classification")
    spec.labels = list(LABELS)
    assert normalise_output(spec, "Label: Card Arrival.") == ("card_arrival", "card_arrival")
    assert normalise_output(spec, "refund not showing up") == ("Refund_not_showing_up", "Refund_not_showing_up")
    assert normalise_output(spec, "something else") == (None, None)


def test_normalise_output_extraction() -> None:
    spec = _spec("extraction")
    spec.json_schema = INVOICE_SCHEMA
    assert normalise_output(spec, f"```json\n{REVERSED}\n```") == (INVOICE, CANONICAL)
    assert normalise_output(spec, "not json") == (None, None)
    assert normalise_output(spec, CANONICAL.replace("2026-03-14", "14.03.2026")) == (None, None)


def test_normalise_output_needs_loaded_labels_or_schema() -> None:
    with pytest.raises(ValueError, match="labels"):
        normalise_output(_spec("classification"), "card arrival")
    with pytest.raises(ValueError, match="Schema"):
        normalise_output(_spec("extraction"), CANONICAL)
