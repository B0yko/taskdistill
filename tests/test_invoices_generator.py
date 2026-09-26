"""Synthetic invoice generator: determinism, split, traits, gold derivability and length budget."""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import random
import re
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from taskdistill.config import PII_KINDS
from taskdistill.curate.pii import PiiScrubber
from taskdistill.demos import invoices as inv
from taskdistill.demos.invoices import (
    CURRENCIES,
    FIELDS,
    QUICK_SIZES,
    SPLIT_TEMPLATES,
    TEMPLATES,
    TRAIT_RATES,
    TRAITS,
    InvoiceDoc,
    doc_to_record,
    format_date,
    format_number,
    generate,
    quick_subset,
    sample_docs,
    split_of,
)

#: SHA-256 of the canonical JSON of the first 100 documents of ``generate()``. Must hold on Python 3.12 and 3.13.
FIRST_100_SHA256 = "c58fc58d71a1456d941b890513a38331541661387607f5d91ce76387df3515ed"

STUDENT_SYSTEM_PROMPT = "Extract the invoice fields as JSON matching the schema."
STUDENT_BASE = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "CHF": "CHF", "PLN": "zł"}
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
ALLOWED_IBANS = ("GB82 WEST 1234 5698 7654 32", "DE89 3704 0044 0532 0130 00")

# Words that may appear only when the corresponding optional field is present.
DUE_WORDS = re.compile(
    r"\bdue\b|payable (?:by|within)|\bpay by\b|terms|\bnet\s*\d|\d+\s*(?:days?|tage|tagen|jours|dni)\b"
    r"|fällig|zahlungs|échéance|conditions|termin|warunki",
    re.IGNORECASE,
)
TAX_WORDS = re.compile(r"\bvat\b|\btax|tax\b|\bmwst\b|\bust\.|\btva\b", re.IGNORECASE)
PO_WORDS = re.compile(r"\bpo\b|p\.o\.|purchase|bestell|your order|your ref|commande|zamówieni", re.IGNORECASE)
DISTRACTOR_WORDS = re.compile(
    r"subtotal|net amount|amount net of|amount before|net total|net value|\bnet = |zwischensumme|nettobetrag"
    r"|montant ht|total ht|wartość netto|previous balance|earlier invoice|previous invoice|account balance"
    r"|old balance|\bolder\b|previous_balance|saldo|solde|deposit|advance payment|part payment|prepaid"
    r"|payments received|remaining|left to pay|\bleft\b|still open|open_amount|anzahlung|restbetrag|acompte"
    r"|reste |zaliczka|pozostało",
    re.IGNORECASE,
)
DEPOSIT_WORDS = re.compile(
    r"deposit|advance payment|part payment|prepaid|payments received|anzahlung|acompte|zaliczka", re.IGNORECASE
)
# A label row that calls its amount owed: "Amount outstanding (incl. tax): 1,234.56 GBP", "Amount payable:   €9.00".
OWED_ROW = re.compile(r"^\s*([^:\n]*\b(?:outstanding|payable)\b[^:\n]*):\s*(\S.*)$", re.IGNORECASE | re.MULTILINE)
# A word (two or more letters) or a closing ")" / "*" run straight into a currency sign, code or amount.
GLUED_VALUE = re.compile(
    r"(?:[^\W\d_]{2,}|[)*])(?:[€£$]|zł|(?:EUR|USD|GBP|CHF|PLN)\b|(?:\d{1,3}(?:[., ]\d{3})+|\d+)[.,]\d\d(?!\d))"
)
QUOTE_MARKERS = re.compile(r"^(?:-----Original Message-----|Previous correspondence:|On .* wrote:|>.*)$", re.MULTILINE)
NET_DAYS = re.compile(r"\bnet\s*(\d+)|(\d+)\s*(?:days?|tage|tagen|jours|dni)\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def docs() -> list[InvoiceDoc]:
    return generate()


def canonical_sha256(documents: list[InvoiceDoc]) -> str:
    payload = [dataclasses.asdict(d) for d in documents]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def date_renderings(day: dt.date) -> list[str]:
    month = MONTHS[day.month - 1]
    return [
        f"{day.year:04d}-{day.month:02d}-{day.day:02d}",
        f"{day.day} {month} {day.year}",
        f"{day.day:02d} {month} {day.year}",
        f"{month} {day.day}, {day.year}",
        f"{day.day} {month[:3]} {day.year}",
        f"{day.day:02d} {month[:3]} {day.year}",
        f"{day.day:02d}-{month[:3]}-{day.year}",
        f"{day.day:02d}.{day.month:02d}.{day.year}",
        f"{day.month:02d}/{day.day:02d}/{day.year}",
        f"{day.day:02d}/{day.month:02d}/{day.year}",
    ]


def amount_renderings(value: float, eu: bool) -> list[str]:
    cents = round(value * 100)
    units, frac = divmod(cents, 100)
    if eu:
        return [f"{units:,}".replace(",", ".") + f",{frac:02d}", f"{units:,}".replace(",", " ") + f",{frac:02d}"]
    return [f"{units:,}.{frac:02d}", f"{units}.{frac:02d}"]


def has_date(text: str, day: dt.date) -> bool:
    """Whether any rendering of ``day`` stands on its own in ``text`` (so "2 Oct" never matches "22 Oct")."""
    return any(re.search(rf"(?<![0-9A-Za-z]){re.escape(r)}(?![0-9])", text) for r in date_renderings(day))


def has_amount(text: str, value: float, eu: bool) -> bool:
    """Whether ``value`` appears as a whole amount in the document's number format."""
    return any(
        re.search(rf"(?<![0-9.,])(?<![0-9] ){re.escape(r)}(?![0-9])", text) for r in amount_renderings(value, eu)
    )


def main_text(text: str) -> str:
    """The document without its quoted or forwarded older messages."""
    match = QUOTE_MARKERS.search(text)
    return text if match is None else text[: match.start()]


def number_shape(number: str) -> re.Pattern[str]:
    parts = ["\\d" if ch.isdigit() else "[A-Z]" if ch.isalpha() else re.escape(ch) for ch in number]
    return re.compile(r"(?<![A-Za-z0-9])" + "".join(parts) + r"(?![A-Za-z0-9])")


def load_schema() -> dict[str, Any]:
    path = resources.files("taskdistill").joinpath("_data", "tasks", "invoices", "schema.json")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------------------------


def test_first_100_documents_are_pinned(docs: list[InvoiceDoc]) -> None:
    assert canonical_sha256(docs[:100]) == FIRST_100_SHA256


def test_generation_is_repeatable_and_prefix_stable(docs: list[InvoiceDoc]) -> None:
    assert generate() == docs
    assert generate(per_template=3) == docs[: 3 * len(TEMPLATES)]
    assert canonical_sha256(generate(seed=8, per_template=4)) != canonical_sha256(docs[: 4 * len(TEMPLATES)])


def test_generator_source_uses_only_stable_random_methods() -> None:
    tree = ast.parse(Path(inv.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    rng_methods = {
        c.func.attr
        for c in calls
        if isinstance(c.func, ast.Attribute)
        and (
            (isinstance(c.func.value, ast.Name) and c.func.value.id == "rng")
            or (isinstance(c.func.value, ast.Attribute) and c.func.value.attr == "rng")
        )
    }
    assert rng_methods == {"random", "randrange", "choice", "shuffle"}
    builtins_called = {c.func.id for c in calls if isinstance(c.func, ast.Name)}
    assert not builtins_called & {"hash", "set", "frozenset"}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Set | ast.SetComp)]
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"__future__", "datetime", "random", "collections.abc", "dataclasses", "typing"}


# --------------------------------------------------------------------------------------------------------
# Counts and split
# --------------------------------------------------------------------------------------------------------


def test_three_thousand_documents_from_thirty_templates(docs: list[InvoiceDoc]) -> None:
    assert len(docs) == 3000
    assert len(TEMPLATES) == 30
    assert Counter(d.template for d in docs) == dict.fromkeys(TEMPLATES, 100)
    assert Counter(d.kind for d in docs) == {"email": 1500, "layout": 1500}
    assert all(d.template.startswith(d.kind + "-") for d in docs)
    assert len({d.id for d in docs}) == 3000
    assert len({d.text for d in docs}) == 3000
    assert docs[0].id == "email-01-000" and docs[30].id == "email-01-001" and docs[-1].id == "layout-15-099"


def test_split_is_by_template_and_balanced_by_kind(docs: list[InvoiceDoc]) -> None:
    assert Counter(d.split for d in docs) == {"train": 2000, "valid": 400, "test": 600}
    expected_kinds = {"train": (10, 10), "valid": (2, 2), "test": (3, 3)}
    for split, templates in SPLIT_TEMPLATES.items():
        emails = sum(t.startswith("email-") for t in templates)
        layouts = sum(t.startswith("layout-") for t in templates)
        assert (emails, layouts) == expected_kinds[split]
    listed = [t for templates in SPLIT_TEMPLATES.values() for t in templates]
    assert sorted(listed) == sorted(TEMPLATES)
    assert len(listed) == len(set(listed))
    splits_per_template: dict[str, set[str]] = {}
    for d in docs:
        splits_per_template.setdefault(d.template, set()).add(d.split)
        assert d.split == split_of(d.template)
    assert all(len(s) == 1 for s in splits_per_template.values())


def test_split_of() -> None:
    assert split_of("email-01") == "train"
    assert split_of("layout-10") == "train"
    assert split_of("email-12") == "valid"
    assert split_of("layout-11") == "valid"
    assert split_of("email-13") == "test"
    assert split_of("layout-15") == "test"
    with pytest.raises(ValueError, match="unknown invoice template"):
        split_of("email-16")


# --------------------------------------------------------------------------------------------------------
# Traits
# --------------------------------------------------------------------------------------------------------


def test_trait_rates_match_targets(docs: list[InvoiceDoc]) -> None:
    assert TRAIT_RATES == {
        "eu_number_format": 0.25,
        "missing_optional": 0.30,
        "distractor_amounts": 0.50,
        "net_terms_due": 0.15,
        "quoted_reply_chain": 0.20,
        "label_typos": 0.10,
    }
    for subset in (docs, [d for d in docs if d.split == "train"]):
        counts = Counter(t for d in subset for t in d.traits)
        for trait in TRAITS:
            assert abs(counts[trait] / len(subset) - TRAIT_RATES[trait]) <= 0.04, trait
    assert all(list(d.traits) == [t for t in TRAITS if t in d.traits] for d in docs)


def test_missing_optional_fields_are_null_and_absent(docs: list[InvoiceDoc]) -> None:
    for d in docs:
        nulls = [f for f in inv.OPTIONAL_FIELDS if d.gold[f] is None]
        assert bool(nulls) == ("missing_optional" in d.traits), d.id
        if d.gold["due_date"] is None:
            assert DUE_WORDS.search(d.text) is None, d.id
        if d.gold["tax_amount"] is None:
            assert TAX_WORDS.search(d.text) is None, d.id
        if d.gold["po_number"] is None:
            assert PO_WORDS.search(d.text) is None, d.id


def test_net_terms_give_only_the_number_of_days(docs: list[InvoiceDoc]) -> None:
    net_docs = [d for d in docs if "net_terms_due" in d.traits]
    assert net_docs
    for d in net_docs:
        assert d.gold["due_date"] is not None, d.id
        invoice_date = dt.date.fromisoformat(d.gold["invoice_date"])
        due = dt.date.fromisoformat(d.gold["due_date"])
        offsets = {int(a or b) for a, b in NET_DAYS.findall(d.text)}
        assert (due - invoice_date).days in offsets, d.id
        assert not has_date(d.text, due), d.id


def test_distractor_amounts_appear_only_with_the_trait(docs: list[InvoiceDoc]) -> None:
    for d in docs:
        found = DISTRACTOR_WORDS.search(main_text(d.text)) is not None
        assert found == ("distractor_amounts" in d.traits), d.id


def test_quoted_reply_chain_holds_an_older_invoice(docs: list[InvoiceDoc]) -> None:
    for d in docs:
        numbers = number_shape(d.gold["invoice_number"]).findall(d.text)
        others = [n for n in numbers if n != d.gold["invoice_number"]]
        if "quoted_reply_chain" in d.traits:
            assert others, d.id
            assert QUOTE_MARKERS.search(d.text) is not None, d.id
        else:
            assert not others, d.id


def test_label_typos_change_only_documents_with_the_trait(docs: list[InvoiceDoc]) -> None:
    clean = inv._generate(inv.SEED, inv.PER_TEMPLATE, apply_typos=False)
    assert len(clean) == len(docs)
    for with_typos, without in zip(docs, clean, strict=True):
        assert with_typos.gold == without.gold
        assert with_typos.traits == without.traits
        assert (with_typos.text != without.text) == ("label_typos" in with_typos.traits), with_typos.id


class _FixedRng(random.Random):
    """Answers every ``randrange(n)`` with a fixed position (capped at ``n - 1``) and counts the draws."""

    def __init__(self, value: int) -> None:
        super().__init__(0)
        self.value = value
        self.calls = 0

    def randrange(self, start: int, stop: int | None = None, step: int = 1) -> int:
        self.calls += 1
        return min(self.value, start - 1)


@pytest.mark.parametrize(
    ("label", "pos", "expected"),
    [
        ("Invoice No.", 0, "Ivnoice No."),
        ("Invoice No.", 4, "Invoiec No."),
        ("Total", 0, "Ttoal"),
        ("Summe", 1, "Sume"),
        ("Due", 0, "Duee"),
        ("PO", 0, "POO"),
        ("Rechnungsbetrag / Invoice total", 0, "Rcehnungsbetrag / Invoice total"),
        ("**Total**", 1, "**Toatl**"),
    ],
)
def test_typo_misspells_the_longest_word(label: str, pos: int, expected: str) -> None:
    rng = _FixedRng(pos)
    assert inv._typo(label, rng, True) == expected
    unchanged = _FixedRng(pos)
    assert inv._typo(label, unchanged, False) == label
    assert rng.calls == unchanged.calls == 1


# --------------------------------------------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------------------------------------------


def test_gold_validates_against_the_schema(docs: list[InvoiceDoc]) -> None:
    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert list(schema["properties"]) == list(FIELDS)
    assert schema["required"] == list(FIELDS)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["currency"]["enum"] == list(CURRENCIES)
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    for d in docs:
        assert list(d.gold) == list(FIELDS)
        errors = [e.message for e in validator.iter_errors(d.gold)]
        assert errors == [], (d.id, errors)
    bad = dict(docs[0].gold, currency="JPY", invoice_date="14.03.2026")
    assert sorted(e.validator for e in validator.iter_errors(bad)) == ["enum", "format", "pattern"]
    assert list(validator.iter_errors({**docs[0].gold, "extra": 1}))


def test_gold_values_are_well_formed(docs: list[InvoiceDoc]) -> None:
    for d in docs:
        g = d.gold
        invoice_date = dt.date.fromisoformat(g["invoice_date"])
        assert dt.date(2025, 1, 1) <= invoice_date <= dt.date(2026, 12, 31)
        if g["due_date"] is not None:
            assert dt.date.fromisoformat(g["due_date"]) > invoice_date
        assert g["currency"] in CURRENCIES
        assert isinstance(g["total_amount"], float) and g["total_amount"] > 0
        assert round(g["total_amount"], 2) == g["total_amount"]
        if g["tax_amount"] is not None:
            assert isinstance(g["tax_amount"], float) and 0 < g["tax_amount"] < g["total_amount"]
            assert round(g["tax_amount"], 2) == g["tax_amount"]
        assert g["vendor_name"].endswith(("GmbH", "Ltd", "LLC", "S.A.", "sp. z o.o."))
        assert not re.search(r"\d", g["vendor_name"])


def test_every_gold_value_is_present_or_derivable(docs: list[InvoiceDoc]) -> None:
    for d in docs:
        g, text = d.gold, d.text
        eu = "eu_number_format" in d.traits
        assert g["vendor_name"] in text, d.id
        assert g["invoice_number"] in text, d.id
        if g["po_number"] is not None:
            assert g["po_number"] in text, d.id
        assert g["currency"] in text or SYMBOLS[g["currency"]] in text, d.id
        assert has_amount(text, g["total_amount"], eu), d.id
        if g["tax_amount"] is not None:
            assert has_amount(text, g["tax_amount"], eu), d.id
        assert has_date(text, dt.date.fromisoformat(g["invoice_date"])), d.id
        if g["due_date"] is not None and "net_terms_due" not in d.traits:
            assert has_date(text, dt.date.fromisoformat(g["due_date"])), d.id


def test_amounts_follow_one_number_format_per_document(docs: list[InvoiceDoc]) -> None:
    us_thousands = re.compile(r"\d,\d{3}\.\d\d(?!\d)")
    eu_thousands = re.compile(r"\d[. ]\d{3},\d\d(?!\d)")
    for d in docs:
        if "eu_number_format" in d.traits:
            assert us_thousands.search(d.text) is None, d.id
        else:
            assert eu_thousands.search(d.text) is None, d.id


def _old_kv(rows: list[tuple[str, str]], sep: str, width: int) -> list[str]:
    return [f"{label + sep:<{width}}{value}" for label, value in rows]


def _old_right(rows: list[tuple[str, str]], width: int) -> list[str]:
    return [f"{label}{value:>{max(1, width - len(label))}}" for label, value in rows]


@pytest.mark.parametrize(
    ("rows", "sep", "width", "expected"),
    [
        ([("Invoice", "X-1")], ": ", 0, ["Invoice: X-1"]),
        ([("Total", "1.00")], ": ", 12, ["Total:      1.00"]),
        ([("Amount", "5.00")], "", 22, ["Amount                5.00"]),
        ([("Previous balance (not included)", "3211.74")], "", 22, ["Previous balance (not included) 3211.74"]),
        ([("Exactly twenty-two ch.", "7.00")], "", 22, ["Exactly twenty-two ch. 7.00"]),
        ([("Twenty-one characters", "7.00")], "", 22, ["Twenty-one characters 7.00"]),
        ([("A very long label here", "9.99")], ": ", 10, ["A very long label here: 9.99"]),
        ([("Montant", "3,00")], " : ", 12, ["Montant :   3,00"]),
    ],
)
def test_kv_rows_always_separate_label_and_value(
    rows: list[tuple[str, str]], sep: str, width: int, expected: list[str]
) -> None:
    assert inv._kv(rows, sep, width) == expected


@pytest.mark.parametrize(
    ("rows", "width", "expected"),
    [
        ([("TOTAL", "12.50")], 20, ["TOTAL          12.50"]),
        ([("Previous balance (not included)", "2.493,53")], 34, ["Previous balance (not included) 2.493,53"]),
        (
            [("Saldo poprzednie / Previous balance", "2 438,64 EUR")],
            46,
            ["Saldo poprzednie / Previous balance 2 438,64 EUR"],
        ),
        ([("abc", "1")], 3, ["abc 1"]),
        (
            [("Subtotal", "2.441,55"), ("TOTAL", "2.954,28")],
            34,
            [f"Subtotal{'2.441,55':>26}", f"TOTAL{'2.954,28':>29}"],
        ),
    ],
)
def test_right_aligned_rows_always_separate_label_and_value(
    rows: list[tuple[str, str]], width: int, expected: list[str]
) -> None:
    lines = inv._right(rows, width)
    assert lines == expected
    for line, (label, value) in zip(lines, rows, strict=True):
        assert line.startswith(label + " ") and line.endswith(value)


def test_row_helpers_keep_rows_that_already_fit_unchanged() -> None:
    rng = random.Random(11)
    checked = 0
    for _ in range(2000):
        label = "x" * rng.randrange(1, 40)
        value = "9" * rng.randrange(1, 14)
        width = rng.randrange(1, 50)
        sep = rng.choice(("", ": ", " : ", "  "))
        old_kv = _old_kv([(label, value)], sep, width)[0]
        if old_kv[len(label + sep)] == " " or (label + sep).endswith(" "):
            assert inv._kv([(label, value)], sep, width)[0] == old_kv
            checked += 1
        old_right = _old_right([(label, value)], width)[0]
        if old_right[len(label)] == " ":
            assert inv._right([(label, value)], width)[0] == old_right
            checked += 1
    assert checked > 1000


def test_no_label_runs_into_its_amount(docs: list[InvoiceDoc]) -> None:
    for d in docs:
        for line in d.text.splitlines():
            assert GLUED_VALUE.search(line) is None, (d.id, line)
    assert GLUED_VALUE.search("Previous balance (not included)2.493,53")
    assert GLUED_VALUE.search("Saldo poprzednie / Previous balance2 438,64 EUR")
    assert GLUED_VALUE.search("Solde antérieur / Previous balance€3,686.45")
    assert GLUED_VALUE.search("Previous balance (not included)GBP 1,464.93")
    assert GLUED_VALUE.search("**Total**1,234.00")
    assert GLUED_VALUE.search("Steel brackets M8    12  1,376.25") is None
    assert GLUED_VALUE.search("Invoice INV-7K3-20418 for 12.00") is None


def test_known_glue_cases_render_with_a_space(docs: list[InvoiceDoc]) -> None:
    by_id = {d.id: d for d in docs}
    assert "Previous balance (not included) 2.493,53" in by_id["layout-03-002"].text.splitlines()
    assert "Previous balance (not included) 3211.74" in by_id["email-12-022"].text.splitlines()
    assert "Saldo poprzednie / Previous balance 2 438,64 EUR" in by_id["layout-15-005"].text.splitlines()
    assert "Solde antérieur / Previous balance €3,686.45" in by_id["layout-12-015"].text.splitlines()


def test_gross_total_is_never_called_owed_next_to_a_deposit() -> None:
    clean = inv._generate(inv.SEED, inv.PER_TEMPLATE, apply_typos=False)
    owed_totals: Counter[str] = Counter()
    for d in clean:
        body = main_text(d.text)
        eu = "eu_number_format" in d.traits
        has_deposit = DEPOSIT_WORDS.search(body) is not None
        for label, value in OWED_ROW.findall(body):
            if has_amount(value, d.gold["total_amount"], eu):
                assert not has_deposit, (d.id, label)
                owed_totals[d.template] += 1
    assert set(owed_totals) == {"email-06", "layout-08", "layout-14"}
    deposit_docs = [d for d in clean if d.template in owed_totals and DEPOSIT_WORDS.search(main_text(d.text))]
    assert deposit_docs
    assert all(owed_totals[t] + sum(d.template == t for d in deposit_docs) == 100 for t in owed_totals)


def test_deposit_rows_name_the_gross_total_neutrally(docs: list[InvoiceDoc]) -> None:
    by_id = {d.id: d for d in docs}
    email = by_id["email-06-005"]
    assert email.gold["total_amount"] == 26794.26
    lines = email.text.splitlines()
    start = lines.index("Invoice total (incl. tax): 26,794.26 GBP")
    assert lines[start + 1 : start + 3] == ["Deposit received: 13,129.00 GBP", "Balance remaining: 13,665.26 GBP"]
    assert "is not yet paid in full (terms: Net 45)" in email.text
    assert "outstanding" not in email.text.lower()

    service = by_id["layout-08-001"]
    assert service.gold["total_amount"] == 85.6
    assert [line for line in service.text.splitlines() if line.startswith(("Invoice total", "Deposit", "Balance"))] == [
        "Invoice total:                    CHF 85.60",
        "Deposit received:                 CHF 49.00",
        "Balance remaining:                CHF 36.60",
    ]
    assert "payable" not in service.text.lower()

    bill = by_id["layout-14-003"]
    assert bill.gold["total_amount"] == 11292.05
    assert "Bill total:                       €11,292.05" in bill.text.splitlines()
    assert "Balance remaining:                €7,792.05" in bill.text.splitlines()
    assert "payable" not in bill.text.lower()


def test_slash_dates_carry_a_format_hint(docs: list[InvoiceDoc]) -> None:
    for d in docs:
        if re.search(r"\b\d\d/\d\d/\d{4}\b", d.text):
            assert "MM/DD/YYYY" in d.text or "DD/MM/YYYY" in d.text, d.id


def test_invoice_and_po_numbers_cannot_look_like_pii(docs: list[InvoiceDoc]) -> None:
    for d in docs:
        for value in (d.gold["invoice_number"], d.gold["po_number"]):
            if value is None:
                continue
            assert re.search(r"[A-Z]", value) and re.search(r"\d", value), value
            assert re.search(r"\d{7,}", value) is None, value
            assert re.search(r"\d{3}-\d{3}-\d{4}|\d{3}-\d{2}-\d{4}", value) is None, value
            groups = re.split(r"[-./ ]", value)
            for left, right in itertools.pairwise(groups):
                assert not (left.isdigit() and right.isdigit()), value


def test_contact_details_use_reserved_values_only(docs: list[InvoiceDoc]) -> None:
    for d in docs:
        for domain in re.findall(r"[\w.+-]+@([\w.-]+)", d.text):
            assert domain in ("example.com", "example.org") or domain.endswith(".test"), (d.id, domain)
        for phone in re.findall(r"\+?\d[\d ().-]{6,}\d", d.text):
            digits = re.sub(r"\D", "", phone)
            if len(digits) >= 7 and "-" in phone and "555-01" in phone:
                assert re.fullmatch(r"\+1 555-01\d\d", phone), (d.id, phone)
        assert re.search(r"\b\d{3}-\d{3}-\d{4}\b|\b\d{3}-\d{2}-\d{4}\b", d.text) is None, d.id
        assert re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", d.text) is None, d.id
        for iban in re.findall(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?", d.text):
            assert iban in ALLOWED_IBANS, (d.id, iban)
    phones = Counter(p for d in docs for p in re.findall(r"555-01\d\d", d.text))
    assert phones and all(0 <= int(p[-2:]) <= 99 for p in phones)


def test_pii_scrub_changes_no_gold_value(docs: list[InvoiceDoc]) -> None:
    scrubber = PiiScrubber(list(PII_KINDS))
    hits: Counter[str] = Counter()
    for d in docs:
        scrubbed, counts = scrubber.scrub(d.text)
        hits.update(counts)
        eu = "eu_number_format" in d.traits
        for field in ("vendor_name", "invoice_number", "po_number"):
            value = d.gold[field]
            if value is not None:
                assert value in scrubbed, (d.id, field)
        for field in ("total_amount", "tax_amount"):
            if d.gold[field] is not None:
                assert has_amount(scrubbed, d.gold[field], eu), (d.id, field)
        for field in ("invoice_date", "due_date"):
            if d.gold[field] is not None:
                day = dt.date.fromisoformat(d.gold[field])
                assert has_date(scrubbed, day) == has_date(d.text, day), (d.id, field)
    assert sum(hits.values()) > 0


# --------------------------------------------------------------------------------------------------------
# Subsets, records and samples
# --------------------------------------------------------------------------------------------------------


def test_quick_subsets_are_fixed_balanced_subsets_of_the_full_splits(docs: list[InvoiceDoc]) -> None:
    assert QUICK_SIZES == {"train": 400, "valid": 60, "test": 100}
    for split, size in QUICK_SIZES.items():
        subset = quick_subset(docs, split)
        assert len(subset) == size
        assert len({d.id for d in subset}) == size
        full_ids = {d.id for d in docs if d.split == split}
        assert {d.id for d in subset} <= full_ids
        per_template = Counter(d.template for d in subset)
        assert set(per_template) == set(SPLIT_TEMPLATES[split])
        assert max(per_template.values()) - min(per_template.values()) <= 1
        for d in subset:
            assert int(d.id.rsplit("-", 1)[1]) < per_template[d.template]
        assert quick_subset(generate(), split) == subset
    test_counts = Counter(d.template for d in quick_subset(docs, "test"))
    assert test_counts == {
        "email-13": 17,
        "email-14": 17,
        "email-15": 17,
        "layout-13": 17,
        "layout-14": 16,
        "layout-15": 16,
    }
    assert set(Counter(d.template for d in quick_subset(docs, "train")).values()) == {20}
    assert set(Counter(d.template for d in quick_subset(docs, "valid")).values()) == {15}


def test_quick_subset_rejects_unknown_split_and_short_input(docs: list[InvoiceDoc]) -> None:
    with pytest.raises(ValueError, match="unknown split"):
        quick_subset(docs, "validation")
    with pytest.raises(ValueError, match="not enough documents"):
        quick_subset(generate(per_template=5), "train")


def test_doc_to_record(docs: list[InvoiceDoc]) -> None:
    doc = next(d for d in docs if d.split == "test" and d.traits)
    record = doc_to_record(doc)
    assert record == {
        "input": doc.text,
        "gold": doc.gold,
        "meta": {
            "id": doc.id,
            "split": "test",
            "group": doc.template,
            "template": doc.template,
            "kind": doc.kind,
            "traits": list(doc.traits),
        },
    }
    assert json.loads(json.dumps(record, ensure_ascii=False)) == record
    assert {doc_to_record(d)["meta"]["split"] for d in docs} == {"train", "valid", "test"}


def test_sample_docs_cover_both_kinds_and_every_trait(docs: list[InvoiceDoc]) -> None:
    sample = sample_docs(20, docs)
    assert len(sample) == 20
    assert len({d.id for d in sample}) == 20
    assert Counter(d.kind for d in sample) == {"email": 10, "layout": 10}
    assert {d.template for d in sample} == set(SPLIT_TEMPLATES["train"])
    assert {t for d in sample for t in d.traits} == set(TRAITS)
    assert any(not d.traits for d in sample)
    assert sample_docs() == sample


# --------------------------------------------------------------------------------------------------------
# Dedupe safety and length budget
# --------------------------------------------------------------------------------------------------------


def _shingles(text: str) -> frozenset[tuple[str, str, str]]:
    words = re.findall(r"\w+", text.lower())
    return frozenset(zip(words, words[1:], words[2:], strict=False))


def test_documents_of_one_template_are_far_from_near_duplicates(docs: list[InvoiceDoc]) -> None:
    by_template: dict[str, list[frozenset[tuple[str, str, str]]]] = {}
    for d in docs:
        by_template.setdefault(d.template, []).append(_shingles(d.text))
    worst = 0.0
    for shingle_sets in by_template.values():
        for i, a in enumerate(shingle_sets):
            for b in shingle_sets[i + 1 :]:
                worst = max(worst, len(a & b) / len(a | b))
    assert worst < 0.6


def _cached_tokenizer_dir() -> Path | None:
    from huggingface_hub import constants

    snapshots = Path(constants.HF_HUB_CACHE) / f"models--{STUDENT_BASE.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir()):
        if (snapshot / "tokenizer.json").is_file() or (snapshot / "tokenizer_config.json").is_file():
            return snapshot
    return None


@pytest.mark.slow
def test_fewer_than_one_percent_of_training_examples_exceed_1024_tokens(docs: list[InvoiceDoc]) -> None:
    snapshot = _cached_tokenizer_dir()
    if snapshot is None:
        pytest.skip(f"{STUDENT_BASE} tokenizer is not in the local Hugging Face cache")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    lengths: list[int] = []
    for d in docs:
        if d.split != "train":
            continue
        answer = json.dumps(d.gold, separators=(",", ":"), ensure_ascii=False)
        messages = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": d.text},
            {"role": "assistant", "content": answer},
        ]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True)
        ids = encoded["input_ids"] if not isinstance(encoded, list) else encoded
        lengths.append(len(ids))
    lengths.sort()
    over = sum(n > 1024 for n in lengths)
    p50, p95 = lengths[len(lengths) // 2], lengths[int(len(lengths) * 0.95)]
    print(f"student tokens over {len(lengths)} training examples: p50={p50} p95={p95} max={lengths[-1]} over={over}")
    assert len(lengths) == 2000
    assert over / len(lengths) < 0.01


def test_teacher_prompt_names_every_field() -> None:
    path = resources.files("taskdistill").joinpath("_data", "tasks", "invoices", "teacher_prompt.md")
    prompt = path.read_text(encoding="utf-8")
    for field in FIELDS:
        assert f'"{field}"' in prompt
    assert "Net 30" in prompt
    assert "null" in prompt


@pytest.mark.parametrize(
    ("cents", "style", "expected"),
    [
        (123456, "us", "1,234.56"),
        (123456, "plain", "1234.56"),
        (123456, "eu_dot", "1.234,56"),
        (123456, "eu_space", "1 234,56"),
        (5, "us", "0.05"),
        (100000000, "eu_dot", "1.000.000,00"),
        (99999, "eu_space", "999,99"),
    ],
)
def test_format_number(cents: int, style: str, expected: str) -> None:
    assert format_number(cents, style) == expected


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("iso", "2026-03-04"),
        ("dmy_long", "4 March 2026"),
        ("mdy_long", "March 4, 2026"),
        ("dmy_short", "04 Mar 2026"),
        ("dmy_dash", "04-Mar-2026"),
        ("dotted", "04.03.2026"),
        ("us_slash", "03/04/2026"),
        ("uk_slash", "04/03/2026"),
    ],
)
def test_format_date(style: str, expected: str) -> None:
    assert format_date(dt.date(2026, 3, 4), style) == expected


def test_unknown_formats_are_rejected() -> None:
    with pytest.raises(ValueError):
        format_number(1, "swiss")
    with pytest.raises(ValueError):
        format_date(dt.date(2026, 1, 1), "julian")
