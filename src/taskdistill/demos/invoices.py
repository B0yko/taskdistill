"""Synthetic invoices for the extraction demo: the generator, its template split and the quick-profile subsets.

The generator uses the standard library only. All randomness comes from one ``random.Random(seed)`` and only
its ``random``, ``randrange``, ``choice`` and ``shuffle`` methods, which are stable across Python 3.12 and
3.13. It never iterates over a set and never calls ``hash()``, so a seed always gives the same documents.

3,000 documents come from 30 templates, 100 each: 15 email bodies and 15 plain-text invoice layouts. The
split is by template and fixed in code, so the test split measures generalisation to unseen layouts. Gold
values are exact by construction. Company names are syllable combinations; every contact detail uses
reserved values (``example.com``/``example.org``/``.test`` domains, 555-01xx phone numbers, example IBANs).
"""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

Kind = Literal["email", "layout"]
SplitName = Literal["train", "valid", "test"]

SEED = 7
PER_TEMPLATE = 100
FIELDS = (
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "currency",
    "total_amount",
    "tax_amount",
    "po_number",
)
OPTIONAL_FIELDS = ("due_date", "tax_amount", "po_number")
CURRENCIES = ("EUR", "USD", "GBP", "CHF", "PLN")
TRAITS = (
    "eu_number_format",
    "missing_optional",
    "distractor_amounts",
    "net_terms_due",
    "quoted_reply_chain",
    "label_typos",
)
TRAIT_RATES: dict[str, float] = {
    "eu_number_format": 0.25,
    "missing_optional": 0.30,
    "distractor_amounts": 0.50,
    "net_terms_due": 0.15,
    "quoted_reply_chain": 0.20,
    "label_typos": 0.10,
}
QUICK_SIZES: dict[str, int] = {"train": 400, "valid": 60, "test": 100}

_FIRST_DAY = dt.date(2025, 1, 1)
_N_DAYS = (dt.date(2027, 1, 1) - _FIRST_DAY).days
_TERM_DAYS = (7, 10, 14, 15, 21, 30, 30, 30, 45, 60, 90)
_MISSING_SUBSETS: tuple[tuple[str, ...], ...] = (
    ("due_date",),
    ("tax_amount",),
    ("po_number",),
    ("due_date", "tax_amount"),
    ("due_date", "po_number"),
    ("tax_amount", "po_number"),
    ("due_date", "tax_amount", "po_number"),
)
_MISSING_SUBSETS_KEEP_DUE: tuple[tuple[str, ...], ...] = (
    ("tax_amount",),
    ("po_number",),
    ("tax_amount", "po_number"),
)
_TAX_RATES_BP: dict[str, tuple[int, ...]] = {
    "EUR": (1900, 2000, 2100, 2300, 700),
    "USD": (600, 725, 800, 825, 950),
    "GBP": (2000, 2000, 500),
    "CHF": (810, 810, 260),
    "PLN": (2300, 2300, 800),
}
_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "CHF": "CHF", "PLN": "zł"}
_MONTHS = (
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
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I or O: never confused with digits, never forms "PO"
_DEPTS = ("MKT", "SRV", "LGX", "ENG", "OPS", "SUP", "HWD", "FAC")

_START = (
    "Bral", "Cor", "Dren", "Esk", "Fal", "Grel", "Hov", "Ister", "Jorn", "Kel", "Lum", "Mord", "Nev", "Ost",
    "Pell", "Quar", "Ros", "Sil", "Tav", "Ulm", "Vard", "Wex", "Yor", "Zel", "Brin", "Cald", "Dov", "Fen",
    "Gorm", "Hal", "Varn", "Tesk",
)  # fmt: skip
_MID = ("a", "e", "i", "o", "an", "en", "ar", "or", "il", "u")
_END = (
    "dex", "vik", "lune", "rith", "quay", "mere", "var", "ium", "dal", "zen", "orra", "ane", "brook", "gard",
    "holm", "stead", "wyn", "tor",
)  # fmt: skip
_SECTORS = (
    "Logistics", "Systems", "Print", "Textiles", "Analytics", "Foods", "Metalworks", "Consulting", "Labs",
    "Supply", "Components", "Media", "Facility Services", "Engineering", "Software", "Trading", "Packaging",
    "Energy", "Instruments", "Design",
)  # fmt: skip
_SUFFIXES = ("GmbH", "Ltd", "LLC", "S.A.", "sp. z o.o.")
_TOWN_ENDS = ("burg", "dal", "haven", "stad", "ville", "wick", "mouth", "berg", "field", "moor")
_STREET_KINDS = ("Road", "Street", "Lane", "Way", "Park", "Avenue")
_FIRST_NAMES = ("Alex", "Sam", "Robin", "Kim", "Jo", "Charlie", "Toni", "Ari", "Noa", "Mika", "Eli", "Sascha")
_ITEMS = (
    "Consulting services", "Steel brackets M8", "Annual software licence", "Freight and handling",
    "On-site support visit", "Printer toner cartridges", "Cloud hosting, monthly plan", "Packaging material",
    "Translation services", "Maintenance visit", "Ergonomic office chairs", "Network cabling",
    "Design review workshop", "Data migration", "Label printing", "Pallet storage", "Travel expenses",
    "Staff training session", "Safety gloves, box", "Laboratory analysis", "Stainless fasteners",
    "Website maintenance", "Conference room hire", "Catering service", "Hydraulic hose assembly",
    "LED panel lights", "Copy paper A4, box", "Courier delivery", "Server rack rental", "Quality inspection",
)  # fmt: skip
_SERVICES = (
    "Consulting", "Project management", "Software development", "Workshop facilitation", "Technical writing",
    "Data analysis", "Quality assurance", "Training delivery", "Solution design", "On-call support",
)  # fmt: skip
_QTYS = (1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 20)
_PRICE_BANDS = ((500, 5000), (2000, 20000), (5000, 60000), (20000, 250000))

# Reserved example IBANs (valid mod-97 checksums, never real accounts).
EXAMPLE_IBANS = ("GB82 WEST 1234 5698 7654 32", "DE89 3704 0044 0532 0130 00")


@dataclass(frozen=True)
class InvoiceDoc:
    """One synthetic document with its exact gold fields."""

    id: str
    template: str
    kind: Kind
    text: str
    gold: dict[str, Any]
    traits: tuple[str, ...]
    split: SplitName


# ----------------------------------------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------------------------------------


def format_date(day: dt.date, style: str) -> str:
    """Render a date in one of the generator's styles."""
    month = _MONTHS[day.month - 1]
    if style == "iso":
        return day.isoformat()
    if style == "dmy_long":
        return f"{day.day} {month} {day.year}"
    if style == "mdy_long":
        return f"{month} {day.day}, {day.year}"
    if style == "dmy_short":
        return f"{day.day:02d} {month[:3]} {day.year}"
    if style == "dmy_dash":
        return f"{day.day:02d}-{month[:3]}-{day.year}"
    if style == "dotted":
        return f"{day.day:02d}.{day.month:02d}.{day.year}"
    if style == "us_slash":
        return f"{day.month:02d}/{day.day:02d}/{day.year}"
    if style == "uk_slash":
        return f"{day.day:02d}/{day.month:02d}/{day.year}"
    raise ValueError(f"unknown date style {style!r}")


def format_number(cents: int, style: str) -> str:
    """Render cents as ``us`` 1,234.56, ``plain`` 1234.56, ``eu_dot`` 1.234,56 or ``eu_space`` 1 234,56."""
    units, frac = divmod(cents, 100)
    if style == "us":
        return f"{units:,}.{frac:02d}"
    if style == "plain":
        return f"{units}.{frac:02d}"
    if style == "eu_dot":
        return f"{units:,}".replace(",", ".") + f",{frac:02d}"
    if style == "eu_space":
        return f"{units:,}".replace(",", " ") + f",{frac:02d}"
    raise ValueError(f"unknown number style {style!r}")


def _rfc_date(day: dt.date, hour: int, minute: int) -> str:
    return f"{_WEEKDAYS[day.weekday()]}, {day.day} {_MONTHS[day.month - 1][:3]} {day.year} {hour:02d}:{minute:02d}"


def _typo(label: str, rng: random.Random, apply: bool) -> str:
    """Misspell the longest word of a label; draws the same random numbers whether or not it applies."""
    words = label.split(" ")
    counts = [sum(ch.isalpha() for ch in word) for word in words]
    best = counts.index(max(counts))
    word = words[best]
    letters = [i for i, ch in enumerate(word) if ch.isalpha()]
    pos = rng.randrange(max(1, len(letters) - 2))
    if not apply or not letters:
        return label
    if len(letters) >= 4:
        a, b = letters[1 + pos], letters[2 + pos]
        if b == a + 1 and word[a] != word[b]:
            new = word[:a] + word[b] + word[a] + word[b + 1 :]
        else:
            new = word[:a] + word[a + 1 :]
    else:
        last = letters[-1]
        new = word[: last + 1] + word[last] + word[last + 1 :]
    words[best] = new
    return " ".join(words)


def _digits(rng: random.Random, n: int) -> str:
    return f"{rng.randrange(10**n):0{n}d}"


def _letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_LETTERS) for _ in range(n))


# Invoice and PO number formats always mix letters and digits, never hold 7+ consecutive digits and never
# put two all-digit groups next to each other, so no phone, SSN, card or IBAN pattern can match them.


def _num_inv(rng: random.Random, day: dt.date) -> str:
    return f"INV-{_digits(rng, 1)}{_letters(rng, 1)}{_digits(rng, 1)}-{_digits(rng, 5)}"


def _num_re(rng: random.Random, day: dt.date) -> str:
    return f"RE{day.year}/{_letters(rng, 1)}{_digits(rng, 4)}"


def _num_ll(rng: random.Random, day: dt.date) -> str:
    return f"{_letters(rng, 2)}{_digits(rng, 4)}-{_letters(rng, 1)}{_digits(rng, 2)}"


def _num_dept(rng: random.Random, day: dt.date) -> str:
    return f"{day.year % 100:02d}-{rng.choice(_DEPTS)}-{_digits(rng, 4)}"


def _num_fv(rng: random.Random, day: dt.date) -> str:
    return f"FV/{_digits(rng, 4)}/{_letters(rng, 2)}/{day.year % 100:02d}"


def _num_zone(rng: random.Random, day: dt.date) -> str:
    return f"{_letters(rng, 2)}-{day.year}{_letters(rng, 1)}-{_digits(rng, 4)}"


def _num_f(rng: random.Random, day: dt.date) -> str:
    return f"F{day.year % 100:02d}{_letters(rng, 1)}{_digits(rng, 4)}"


def _num_seq(rng: random.Random, day: dt.date) -> str:
    return f"{_digits(rng, 5)}/{_letters(rng, 1)}{day.year % 100:02d}"


def _num_trx(rng: random.Random, day: dt.date) -> str:
    return f"{_letters(rng, 3)}{_digits(rng, 3)}{_letters(rng, 1)}{_digits(rng, 2)}"


def _num_sub(rng: random.Random, day: dt.date) -> str:
    return f"SUB-{_letters(rng, 2)}{_digits(rng, 5)}"


def _po_dash(rng: random.Random) -> str:
    return f"PO-{_digits(rng, 5)}-{_letters(rng, 1)}"


def _po_compact(rng: random.Random) -> str:
    return f"PO{_digits(rng, 4)}{_letters(rng, 2)}"


def _po_prefix(rng: random.Random) -> str:
    return f"{_letters(rng, 2)}-PO-{_digits(rng, 4)}"


def _po_bst(rng: random.Random) -> str:
    return f"BST-{_digits(rng, 4)}-{_letters(rng, 1)}{_digits(rng, 1)}"


def _po_ord(rng: random.Random) -> str:
    return f"ORD/{_letters(rng, 1)}{_digits(rng, 5)}"


def _po_p(rng: random.Random) -> str:
    return f"P{_digits(rng, 3)}-{_letters(rng, 1)}{_digits(rng, 3)}"


def _po_mixed(rng: random.Random) -> str:
    return f"{_digits(rng, 4)}{_letters(rng, 2)}{_digits(rng, 2)}"


# ----------------------------------------------------------------------------------------------------------
# Facts and rendering context
# ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Older:
    number: str
    date: dt.date
    sent: dt.date
    amount: int


@dataclass(frozen=True)
class _Facts:
    vendor: str
    slug: str
    customer: str
    cslug: str
    number: str
    inv_date: dt.date
    due_days: int
    due_date: dt.date | None
    net_only: bool
    currency: str
    items: tuple[tuple[str, int, int], ...]
    subtotal: int
    rate_bp: int
    tax: int | None
    total: int
    po: str | None
    distractors: dict[str, int]
    older: _Older | None
    sent: dt.date
    reminder: dt.date
    delivery: dt.date
    hour: int
    minute: int
    person: str
    colleague: str
    phone: str
    street: str
    city: str
    cstreet: str
    ccity: str
    typo_keys: tuple[str, ...]


@dataclass(frozen=True)
class _Template:
    id: str
    kind: Kind
    render: Callable[[_Ctx], str]
    number: Callable[[random.Random, dt.date], str]
    po: Callable[[random.Random], str]
    date_style: str
    money_style: str = "code_before"
    num_style: str = "us"
    eu_style: str = "eu_dot"
    currencies: tuple[str, ...] = CURRENCIES
    quote_style: str = "note"


class _Ctx:
    """What a template needs to render one document."""

    def __init__(self, rng: random.Random, tpl: _Template, facts: _Facts, eu: bool, apply_typos: bool) -> None:
        self.rng = rng
        self.t = tpl
        self.f = facts
        self.eu = eu
        self._apply_typos = apply_typos
        self._typos: dict[str, str] = {}

    def ch(self, *options: str) -> str:
        return self.rng.choice(options)

    def d(self, day: dt.date) -> str:
        return format_date(day, self.t.date_style)

    def n(self, cents: int) -> str:
        return format_number(cents, self.t.eu_style if self.eu else self.t.num_style)

    def m(self, cents: int) -> str:
        num = self.n(cents)
        cur = self.f.currency
        style = self.t.money_style
        if style == "bare":
            return num
        if style == "code_before":
            return f"{cur} {num}"
        if style == "code_after":
            return f"{num} {cur}"
        if cur == "CHF":
            return f"CHF {num}"
        if cur == "PLN" or self.eu:
            return f"{num} {_SYMBOLS[cur]}"
        return f"{_SYMBOLS[cur]}{num}"

    def rate(self) -> str:
        whole, frac = divmod(self.f.rate_bp, 100)
        if frac == 0:
            return f"{whole}%"
        text = f"{whole}.{frac:02d}".rstrip("0")
        return (text.replace(".", ",") if self.eu else text) + "%"

    def taxname(self) -> str:
        return "sales tax" if self.f.currency == "USD" else "VAT"

    def taxlabel(self) -> str:
        return "Sales tax" if self.f.currency == "USD" else "VAT"

    def lab(self, key: str, text: str) -> str:
        """A field label; misspelt when the document has the label_typos trait and ``key`` was picked."""
        if key not in self.f.typo_keys:
            return text
        if text not in self._typos:
            self._typos[text] = _typo(text, self.rng, self._apply_typos)
        return self._typos[text]

    def rfc(self, day: dt.date) -> str:
        return _rfc_date(day, self.f.hour, self.f.minute)

    def vendor_email(self) -> str:
        return f"billing@{self.f.slug}.test"

    @property
    def pp(self) -> int | None:
        return self.f.distractors.get("partial_payment")

    @property
    def pb(self) -> int | None:
        return self.f.distractors.get("previous_balance")

    @property
    def show_subtotal(self) -> bool:
        return "subtotal" in self.f.distractors


# ----------------------------------------------------------------------------------------------------------
# Shared building blocks
# ----------------------------------------------------------------------------------------------------------


def _greeting(c: _Ctx) -> str:
    return c.ch("Dear Sir or Madam,", "Hello,", "Hi team,", f"Dear {c.f.customer} team,", "Good morning,")


def _closing(c: _Ctx) -> str:
    return c.ch("Kind regards,", "Best regards,", "Many thanks,", "Best wishes,", "Regards,")


def _amount_sentences(c: _Ctx) -> list[str]:
    f = c.f
    total = c.m(f.total)
    out: list[str] = []
    if f.tax is None:
        out.append(
            c.ch(
                f"The {c.lab('total', 'total amount')} is {total}.",
                f"The {c.lab('total', 'invoice total')} comes to {total}.",
            )
        )
    elif c.show_subtotal:
        out.append(
            f"The net amount is {c.m(f.subtotal)}; {c.lab('tax', c.taxname())} at {c.rate()} adds {c.m(f.tax)}, "
            f"so the {c.lab('total', 'invoice total')} is {total}."
        )
    else:
        out.append(
            c.ch(
                f"The {c.lab('total', 'total amount')} is {total}, including {c.m(f.tax)} {c.lab('tax', c.taxname())}.",
                f"The {c.lab('total', 'invoice total')} of {total} includes {c.lab('tax', c.taxname())} "
                f"of {c.m(f.tax)} ({c.rate()}).",
            )
        )
    if c.pp is not None:
        out.append(
            c.ch(
                f"We have already received your deposit of {c.m(c.pp)}, which leaves {c.m(f.total - c.pp)} to pay.",
                f"Thank you for the advance payment of {c.m(c.pp)}; the remaining balance is {c.m(f.total - c.pp)}.",
            )
        )
    if c.pb is not None:
        out.append(
            c.ch(
                f"Please note that {c.m(c.pb)} from an earlier invoice is still open on your account; "
                "it is not included in this invoice.",
                f"Your account also shows a previous balance of {c.m(c.pb)}, which is not part of this invoice.",
            )
        )
    return out


def _due_sentence(c: _Ctx) -> str | None:
    f = c.f
    if f.due_date is None:
        return None
    if f.net_only:
        return c.ch(
            f"Our {c.lab('terms', 'payment terms')} are net {f.due_days} days.",
            f"{c.lab('terms', 'Terms')}: Net {f.due_days}.",
            f"The invoice is {c.lab('terms', 'payable')} within {f.due_days} days of the invoice date.",
        )
    return c.ch(
        f"Payment is {c.lab('due', 'due')} by {c.d(f.due_date)}.",
        f"Please pay by the {c.lab('due', 'due date')}, {c.d(f.due_date)}.",
        f"The {c.lab('due', 'due date')} is {c.d(f.due_date)}.",
    )


def _po_sentence(c: _Ctx) -> str | None:
    po = c.f.po
    if po is None:
        return None
    return c.ch(
        f"Please quote your {c.lab('po', 'purchase order')} {po} with your payment.",
        f"This invoice refers to your {c.lab('po', 'PO')} {po}.",
        f"{c.lab('po', 'Purchase order')}: {po}.",
    )


def _terms_value(c: _Ctx, style: str) -> str:
    n = c.f.due_days
    if style == "net":
        return f"Net {n}"
    if style == "days_net":
        return f"{n} days net"
    if style == "de":
        return f"{n} Tage netto / {n} days net"
    if style == "fr":
        return f"{n} jours net / {n} days net"
    if style == "pl":
        return f"{n} dni / {n} days"
    raise ValueError(style)


def _due_rows(
    c: _Ctx, due_label: str, terms_label: str, terms_style: str = "net", with_terms: bool = False
) -> list[tuple[str, str]]:
    """Label/value rows for the due date: an explicit date, or only the terms when the doc has net_terms_due."""
    f = c.f
    if f.due_date is None:
        return []
    if f.net_only:
        return [(c.lab("terms", terms_label), _terms_value(c, terms_style))]
    rows = [(c.lab("due", due_label), c.d(f.due_date))]
    if with_terms:
        rows.insert(0, (terms_label, _terms_value(c, terms_style)))
    return rows


@dataclass(frozen=True)
class _TotalLabels:
    subtotal: str = "Subtotal"
    tax: str | None = None
    total: str = "Total"
    total_tax: str = "Total"
    paid: str = "Deposit received"
    remaining: str = "Balance remaining"
    previous: str = "Previous balance (not included)"


def _gross_labels(c: _Ctx, total: str, total_tax: str, neutral: str = "Invoice total") -> _TotalLabels:
    """Labels that call the gross total "outstanding" or "payable" only when no deposit leaves a smaller balance."""
    if c.pp is None:
        return _TotalLabels(total=total, total_tax=total_tax)
    return _TotalLabels(total=neutral, total_tax=neutral if total == total_tax else f"{neutral} (incl. tax)")


def _total_rows(c: _Ctx, labels: _TotalLabels) -> list[tuple[str, str]]:
    f = c.f
    rows: list[tuple[str, str]] = []
    if c.show_subtotal:
        rows.append((labels.subtotal, c.m(f.subtotal)))
    if f.tax is not None:
        rows.append((f"{c.lab('tax', labels.tax or c.taxlabel())} {c.rate()}", c.m(f.tax)))
    rows.append((c.lab("total", labels.total if f.tax is None else labels.total_tax), c.m(f.total)))
    if c.pp is not None:
        rows.append((labels.paid, c.m(c.pp)))
        rows.append((labels.remaining, c.m(f.total - c.pp)))
    if c.pb is not None:
        rows.append((labels.previous, c.m(c.pb)))
    return rows


def _kv(rows: Sequence[tuple[str, str]], sep: str = ": ", width: int = 0) -> list[str]:
    """``label<sep>value`` rows, optionally padded to a value column; a label never runs into its value."""
    if not width:
        return [f"{label}{sep}{value}" for label, value in rows]
    out: list[str] = []
    for label, value in rows:
        cell = label + sep
        if len(cell) >= width and not cell.endswith(" "):
            cell += " "
        out.append(f"{cell:<{width}}{value}")
    return out


def _right(rows: Sequence[tuple[str, str]], width: int) -> list[str]:
    """Rows with the value right-aligned to ``width``; at least one space between label and value."""
    return [f"{label} {value:>{max(1, width - len(label) - 1)}}" for label, value in rows]


def _items_table(
    c: _Ctx, style: str, headers: tuple[str, str, str, str] = ("Description", "Qty", "Unit", "Amount")
) -> list[str]:
    rows = [(desc, str(qty), c.n(unit), c.n(qty * unit)) for desc, qty, unit in c.f.items]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(4)]

    def fmt(r: Sequence[str]) -> list[str]:
        return [r[0].ljust(widths[0]), r[1].rjust(widths[1]), r[2].rjust(widths[2]), r[3].rjust(widths[3])]

    if style == "pipes":
        out = ["| " + " | ".join(fmt(headers)) + " |", "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
        out += ["| " + " | ".join(fmt(r)) + " |" for r in rows]
        return out
    if style == "box":
        rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        out = [rule, "| " + " | ".join(fmt(headers)) + " |", rule]
        out += ["| " + " | ".join(fmt(r)) + " |" for r in rows]
        return [*out, rule]
    line = "  ".join(fmt(headers))
    out = [line, "-" * len(line)]
    out += ["  ".join(fmt(r)) for r in rows]
    return out


def _quote_lines(c: _Ctx, style: str) -> list[str]:
    """The quoted older message holding an earlier invoice's number, date and amount."""
    o = c.f.older
    if o is None:
        return []
    when, amount = c.d(o.date), c.m(o.amount)
    if style == "original":
        return [
            "",
            "-----Original Message-----",
            f"From: {c.f.vendor} <{c.vendor_email()}>",
            f"Sent: {c.rfc(o.sent)}",
            f"Subject: Invoice {o.number}",
            "",
            c.ch(
                f"Invoice {o.number} of {when} for {amount} has been paid in full. Thank you!",
                f"Attached is invoice {o.number} dated {when}, amount {amount}.",
            ),
        ]
    if style == "nested":
        return [
            "",
            f"> On {c.d(o.sent)} {c.f.customer} wrote:",
            f"> > We have paid invoice {o.number} ({amount}, issued {when}). Please confirm receipt.",
            "> Confirmed, thank you.",
        ]
    if style == "note":
        return [
            "",
            "Previous correspondence:",
            f"> {c.ch('Payment received with thanks', 'Settled')}: invoice {o.number} dated {when}, {amount}.",
        ]
    return [
        "",
        f"On {c.d(o.sent)}, {c.f.person} <{c.vendor_email()}> wrote:",
        f"> {c.ch('Hello,', 'Hi,', 'Good afternoon,')}",
        f"> please find attached invoice {o.number} dated {when} for {amount}.",
        "> Kind regards",
    ]


def _join(lines: Sequence[str | None]) -> str:
    text = "\n".join(line for line in lines if line is not None)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip("\n") + "\n"


# ----------------------------------------------------------------------------------------------------------
# Email templates
# ----------------------------------------------------------------------------------------------------------


def _email_01(c: _Ctx) -> str:
    f = c.f
    return _join(
        [
            f"Subject: Invoice {f.number} from {f.vendor}",
            "",
            _greeting(c),
            "",
            c.ch("please find attached", "attached please find", "enclosed you will find")
            + f" our {c.lab('number', 'invoice')} {f.number} {c.lab('date', 'dated')} {c.d(f.inv_date)}.",
            *_amount_sentences(c),
            _due_sentence(c),
            _po_sentence(c),
            "",
            c.ch("Do not hesitate to contact us if you have any questions.", "Thank you for your business."),
            "",
            _closing(c),
            f.person,
            f.vendor,
            f"{c.vendor_email()} | Tel. {f.phone}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_02(c: _Ctx) -> str:
    f = c.f
    lines: list[str | None] = [
        f"Subject: Payment reminder - {c.lab('number', 'invoice')} {f.number}",
        f"Date: {c.rfc(f.reminder)}",
        "",
        _greeting(c),
        "",
        f"this is a friendly reminder that {c.lab('number', 'invoice')} {f.number}, "
        f"{c.lab('date', 'issued on')} {c.d(f.inv_date)}, has not been settled yet.",
    ]
    tax = f" (of which {c.m(f.tax)} {c.lab('tax', c.taxname())})" if f.tax is not None else ""
    lines.append(f"The {c.lab('total', 'invoice total')} is {c.m(f.total)}{tax}.")
    if c.show_subtotal:
        lines.append(f"For reference, the amount before {c.taxname()} was {c.m(f.subtotal)}.")
    if f.due_date is not None:
        if f.net_only:
            lines.append(f"Our {c.lab('terms', 'terms')} are Net {f.due_days}, counted from the invoice date.")
        else:
            lines.append(f"It was {c.lab('due', 'due')} on {c.d(f.due_date)}.")
    if f.po is not None:
        lines.append(f"The invoice relates to your {c.lab('po', 'order')} {f.po}.")
    if c.pp is not None:
        lines.append(f"Thank you for the part payment of {c.m(c.pp)}; {c.m(f.total - c.pp)} is still open.")
    if c.pb is not None:
        lines.append(f"In addition, {c.m(c.pb)} from a previous invoice remains outstanding.")
    lines += [
        "",
        "If you have already paid, please disregard this message.",
        "",
        _closing(c),
        f.person,
        f"Credit control, {f.vendor}",
        f"Tel. {f.phone}",
        *_quote_lines(c, c.t.quote_style),
    ]
    return _join(lines)


def _email_03(c: _Ctx) -> str:
    f = c.f
    first = f.colleague.split(" ")[0]
    return _join(
        [
            f"From: {f.colleague} <{first.lower()}@{f.cslug}.test>",
            f"To: accounts-payable@{f.cslug}.test",
            f"Subject: Fwd: Invoice {f.number}",
            "",
            c.ch("Please book and pay this one.", "FYI - for approval and payment.", "Can you process this, please?"),
            "",
            first,
            "",
            "---------- Forwarded message ---------",
            f"From: {f.vendor} <{c.vendor_email()}>",
            f"Date: {c.rfc(f.sent)}",
            f"Subject: Invoice {f.number}",
            f"To: {f.colleague} <{first.lower()}@{f.cslug}.test>",
            "",
            f"Hello {first},",
            "",
            f"attached is {c.lab('number', 'invoice')} {f.number} {c.lab('date', 'dated')} {c.d(f.inv_date)} "
            f"for {c.ch('the goods delivered last week', 'the services provided', 'your recent delivery')}.",
            *_amount_sentences(c),
            _due_sentence(c),
            _po_sentence(c),
            "",
            _closing(c),
            f.person,
            f.vendor,
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_04(c: _Ctx) -> str:
    f = c.f
    parts = [f"{c.lab('number', 'Invoice')} {f.number} {c.lab('date', 'dated')} {c.d(f.inv_date)} attached"]
    total = f"{c.lab('total', 'total')} {c.m(f.total)}"
    if f.tax is not None:
        total += f" incl. {c.m(f.tax)} {c.lab('tax', c.taxname())}"
    parts.append(total)
    if f.po is not None:
        parts.append(f"{c.lab('po', 'PO')} {f.po}")
    tail: list[str] = []
    if f.due_date is not None:
        tail.append(
            f"{c.lab('terms', 'Terms')} net {f.due_days}."
            if f.net_only
            else f"{c.lab('due', 'Due')} {c.d(f.due_date)}."
        )
    if c.show_subtotal:
        tail.append(f"(net amount {c.m(f.subtotal)})")
    if c.pp is not None:
        tail.append(f"Deposit of {c.m(c.pp)} already received, {c.m(f.total - c.pp)} left.")
    if c.pb is not None:
        tail.append(f"Old balance of {c.m(c.pb)} still open, separate.")
    first = f.person.split(" ")[0]
    return _join(
        [
            c.ch("Hi,", "Hi there,", "Hello,"),
            "",
            " - ".join(parts) + ".",
            " ".join(tail) if tail else None,
            "",
            c.ch("Thanks!", "Cheers,", "Thanks a lot,"),
            first,
            f.vendor,
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_05(c: _Ctx) -> str:
    f = c.f
    desc = f.items[0][0].lower()
    body = [
        f"We hereby submit our {c.lab('number', 'invoice no.')} {f.number} {c.lab('date', 'dated')} "
        f"{c.d(f.inv_date)} for {desc} and related items."
    ]
    if f.tax is None:
        body.append(f"The {c.lab('total', 'total amount')} is {c.m(f.total)}.")
    else:
        body.append(
            f"The {c.lab('total', 'total amount')} of {c.m(f.total)} includes {c.lab('tax', c.taxname())} "
            f"of {c.m(f.tax)} at {c.rate()}."
        )
    if c.show_subtotal:
        body.append(f"The amount net of {c.taxname()} is {c.m(f.subtotal)}.")
    if f.due_date is not None:
        body.append(
            f"The amount is {c.lab('terms', 'payable')} within {f.due_days} days of the invoice date."
            if f.net_only
            else f"The amount is payable by {c.d(f.due_date)} ({c.lab('due', 'due date')})."
        )
    if f.po is not None:
        body.append(f"This invoice refers to your {c.lab('po', 'purchase order')} {f.po}.")
    if c.pp is not None:
        body.append(f"Your advance payment of {c.m(c.pp)} has been received; {c.m(f.total - c.pp)} remains to be paid.")
    if c.pb is not None:
        body.append(f"We also note an unpaid balance of {c.m(c.pb)} from a previous invoice.")
    return _join(
        [
            f.vendor,
            f.street,
            f.city,
            "",
            f.customer,
            "Attn: Accounts Payable",
            f.cstreet,
            f.ccity,
            "",
            f"{f.city.split(' ')[-1]}, {c.d(f.sent)}",
            "",
            f"Re: {c.lab('number', 'Invoice No.')} {f.number}",
            "",
            "Dear Sir or Madam,",
            "",
            " ".join(body),
            "",
            "Yours faithfully,",
            "",
            f.person,
            f.vendor,
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_06(c: _Ctx) -> str:
    f = c.f
    unpaid = "remains unpaid" if c.pp is None else "is not yet paid in full"
    if f.due_date is None:
        status = unpaid
    elif f.net_only:
        status = f"{unpaid} ({c.lab('terms', 'terms')}: Net {f.due_days})"
    else:
        status = f"was {c.lab('due', 'due')} on {c.d(f.due_date)} and {unpaid}"
    rows = _total_rows(c, _gross_labels(c, "Amount outstanding", "Amount outstanding (incl. tax)"))
    return _join(
        [
            f"Subject: OVERDUE - {c.lab('number', 'Invoice')} {f.number}",
            f"Date: {c.rfc(f.reminder)}",
            "",
            _greeting(c),
            "",
            f"According to our records, {c.lab('number', 'invoice')} {f.number} {c.lab('date', 'dated')} "
            f"{c.d(f.inv_date)} {status}.",
            "",
            *_kv(rows),
            f"{c.lab('po', 'Your PO')}: {f.po}" if f.po is not None else None,
            "",
            c.ch(
                "Please arrange payment immediately to avoid further action.",
                "Please settle the amount at your earliest convenience.",
            ),
            "If payment has already been made, please send us the remittance details.",
            "",
            _closing(c),
            f"{f.vendor} - Accounts Receivable",
            f"{c.vendor_email()} / {f.phone}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_07(c: _Ctx) -> str:
    f = c.f
    bullets: list[tuple[str, str]] = [
        (c.lab("number", "Invoice number"), f.number),
        (c.lab("date", "Invoice date"), c.d(f.inv_date)),
    ]
    if f.po is not None:
        bullets.append((c.lab("po", "PO number"), f.po))
    bullets += _total_rows(c, _TotalLabels(total="Total", total_tax="Total incl. tax"))
    bullets += _due_rows(c, "Due date", "Payment terms", "days_net")
    return _join(
        [
            f"Subject: {f.vendor} invoice {f.number}",
            "",
            _greeting(c),
            "",
            c.ch("Thank you for your business.", "Thanks again for working with us.")
            + " Here are the details of our latest invoice:",
            "",
            *[f"- {label}: {value}" for label, value in bullets],
            "",
            "The PDF copy is attached for your records.",
            "",
            _closing(c),
            f.person,
            f.vendor,
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_08(c: _Ctx) -> str:
    f = c.f
    rows: list[tuple[str, str]] = [
        (c.lab("number", "Invoice"), f.number),
        (c.lab("date", "Issued"), c.d(f.inv_date)),
        (c.lab("total", "Amount"), c.m(f.total)),
    ]
    if f.tax is not None:
        rows.append((c.lab("tax", "Tax"), c.m(f.tax)))
    rows += _due_rows(c, "Due", "Terms", "net")
    if f.po is not None:
        rows.append((c.lab("po", "Your ref"), f.po))
    extra: list[str] = []
    if c.show_subtotal:
        extra.append(f"Net amount: {c.m(f.subtotal)}")
    if c.pb is not None:
        extra.append(f"Account balance before this invoice: {c.m(c.pb)}")
    if c.pp is not None:
        extra.append(f"Payments received on this invoice: {c.m(c.pp)}")
    return _join(
        [
            "*** This is an automated message - please do not reply ***",
            "",
            f"A new invoice from {f.vendor} is available in your customer portal.",
            "",
            *_kv(rows, ": ", width=12),
            *extra,
            "",
            f"Questions? Contact {c.vendor_email()} or call {f.phone}.",
            f"{f.vendor} - Billing",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_09(c: _Ctx) -> str:
    f = c.f
    rows = _total_rows(
        c,
        _TotalLabels(
            subtotal="Nettobetrag / Net amount",
            tax="MwSt. / VAT",
            total="Rechnungsbetrag / Invoice total",
            total_tax="Rechnungsbetrag / Invoice total",
            paid="Anzahlung erhalten / Deposit received",
            remaining="Restbetrag / Balance remaining",
            previous="Vorheriger Saldo / Previous balance",
        ),
    )
    rows += _due_rows(c, "Fällig am / Due date", "Zahlungsziel / Payment terms", "de")
    if f.po is not None:
        rows.append((c.lab("po", "Ihre Bestellung / Your PO"), f.po))
    return _join(
        [
            f"Betreff / Subject: Rechnung / Invoice {f.number}",
            "",
            "Sehr geehrte Damen und Herren,",
            "Dear Sir or Madam,",
            "",
            f"anbei erhalten Sie unsere Rechnung Nr. {f.number} vom {c.d(f.inv_date)}.",
            f"please find attached our {c.lab('number', 'invoice no.')} {f.number} {c.lab('date', 'dated')} "
            f"{c.d(f.inv_date)}.",
            "",
            *_kv(rows),
            "",
            "Mit freundlichen Grüßen / Kind regards",
            f.person,
            f.vendor,
            f.street,
            f.city,
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_10(c: _Ctx) -> str:
    f = c.f
    plan = c.ch("Business plan", "Team plan", "Professional plan", "Enterprise plan")
    month = f"{_MONTHS[f.inv_date.month - 1]} {f.inv_date.year}"
    lines: list[str | None] = [
        f"Subject: Your {f.vendor} subscription - invoice {f.number}",
        "",
        c.ch("Hello,", "Hi there,"),
        "",
        f"thank you for renewing your subscription with {f.vendor} - your {c.lab('number', 'invoice number')} is "
        f"{f.number} and it was {c.lab('date', 'issued on')} {c.d(f.inv_date)}.",
        "",
        f"Plan: {plan} ({f.items[0][1]} {'seat' if f.items[0][1] == 1 else 'seats'})",
        f"Billing period: {month}",
        *_amount_sentences(c),
        _due_sentence(c),
        _po_sentence(c),
        "",
        "You can download the invoice from your account page at any time.",
        "",
        f"The {f.vendor} team",
        *_quote_lines(c, c.t.quote_style),
    ]
    return _join(lines)


def _email_11(c: _Ctx) -> str:
    f = c.f
    first = f.colleague.split(" ")[0]
    detail = (
        f"{c.lab('number', 'Invoice no.')} {f.number}, {c.lab('date', 'invoice date')} {c.d(f.inv_date)}, "
        f"{c.lab('total', 'total')} {c.m(f.total)}"
    )
    if f.tax is not None:
        detail += f" ({c.lab('tax', c.taxname())} {c.m(f.tax)} included)"
    return _join(
        [
            f"Hi {first},",
            "",
            c.ch("thanks for getting in touch.", "thanks for your message.", "thank you for your question.")
            + " As requested, here are the details of the invoice we sent you:",
            "",
            detail + ".",
            _due_sentence(c),
            _po_sentence(c),
            *_amount_sentences(c)[1:],
            f"The net amount before {c.taxname()} was {c.m(f.subtotal)}." if c.show_subtotal else None,
            "",
            c.ch("Let me know if anything else is needed.", "Happy to help if you need anything else."),
            "",
            "Best,",
            f.person,
            f"{f.vendor} accounts team",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_12(c: _Ctx) -> str:
    f = c.f
    rows: list[tuple[str, str]] = [
        (c.lab("number", "Invoice"), f.number),
        (c.lab("date", "Date"), c.d(f.inv_date)),
    ]
    rows += _due_rows(c, "Due date", "Terms", "net", with_terms=True)
    if f.po is not None:
        rows.append((c.lab("po", "PO"), f.po))
    rows += _total_rows(c, _TotalLabels(total="Amount", total_tax="Amount incl. tax"))
    return _join(
        [
            f"Subject: New invoice {f.number}",
            "",
            f"Hello {f.customer} team,",
            "",
            "a new invoice has been issued for your account:",
            "",
            *_kv(rows, "", width=22),
            "",
            f"Currency: {f.currency}",
            "",
            c.ch("Best regards,", "Regards,"),
            f.vendor,
            f.phone,
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_13(c: _Ctx) -> str:
    f = c.f
    first = f.colleague.split(" ")[0]
    rows: list[tuple[str, str]] = [
        ("Vendor", f.vendor),
        (c.lab("number", "Invoice #"), f.number),
        (c.lab("date", "Invoice date"), c.d(f.inv_date)),
    ]
    rows += _total_rows(c, _TotalLabels(total="Gross amount", total_tax="Gross amount"))
    rows += _due_rows(c, "Due date", "Terms", "days_net")
    if f.po is not None:
        rows.append((c.lab("po", "PO"), f.po))
    return _join(
        [
            f"Hi {first},",
            "",
            f"we received the invoice below from {f.vendor}; could you approve it for payment?",
            "",
            *_kv(rows),
            "",
            "Thanks,",
            f"{f.person.split(' ')[0]} (Accounts Payable)",
            "",
            "-----Original Message-----",
            f"From: {c.vendor_email()}",
            f"Sent: {c.rfc(f.sent)}",
            f"Subject: Invoice {f.number}",
            "",
            c.ch(
                "Please find our invoice attached.", "Attached is our latest invoice.", "Invoice attached, thank you."
            ),
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_14(c: _Ctx) -> str:
    f = c.f
    msg = (
        f"{c.lab('number', 'inv')} {f.number} from {f.vendor} came in, {c.lab('date', 'dated')} {c.d(f.inv_date)}, "
        f"{c.lab('total', 'total')} {c.m(f.total)}"
    )
    if f.tax is not None:
        msg += f", {c.lab('tax', c.taxname())} {c.m(f.tax)}"
    if f.po is not None:
        msg += f", {c.lab('po', 'PO')} {f.po}"
    if f.due_date is not None:
        msg += (
            f", {c.lab('terms', 'terms')} net {f.due_days}"
            if f.net_only
            else f", {c.lab('due', 'due')} by {c.d(f.due_date)}"
        )
    extra: list[str] = []
    if c.show_subtotal:
        extra.append(f"net amount {c.m(f.subtotal)}")
    if c.pp is not None:
        extra.append(f"they already got {c.m(c.pp)} from us as a deposit, so {c.m(f.total - c.pp)} left to pay")
    if c.pb is not None:
        extra.append(f"there is also an older {c.m(c.pb)} on the account, not this one")
    return _join(
        [
            c.ch("hey", "hi", "morning"),
            msg + ".",
            "; ".join(extra) if extra else None,
            c.ch("can you pay it?", "ok to pay?", "pls book it."),
            f"thx {f.person.split(' ')[0]}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _email_15(c: _Ctx) -> str:
    f = c.f
    rows = _total_rows(
        c,
        _TotalLabels(
            subtotal="Montant HT / Net amount",
            tax="TVA / VAT",
            total="Montant total / Total amount",
            total_tax="Montant TTC / Total incl. VAT",
            paid="Acompte reçu / Deposit received",
            remaining="Reste à payer / Balance remaining",
            previous="Solde antérieur / Previous balance",
        ),
    )
    rows += _due_rows(c, "Échéance / Due date", "Conditions de paiement / Payment terms", "fr")
    if f.po is not None:
        rows.append((c.lab("po", "Bon de commande / Purchase order"), f.po))
    return _join(
        [
            f"Objet / Subject : Facture / Invoice {f.number}",
            "",
            "Madame, Monsieur,",
            "Dear Sir or Madam,",
            "",
            f"Veuillez trouver ci-joint notre facture n° {f.number} du {c.d(f.inv_date)}.",
            f"Please find attached our {c.lab('number', 'invoice no.')} {f.number} {c.lab('date', 'dated')} "
            f"{c.d(f.inv_date)}.",
            "",
            *_kv(rows, " : "),
            "",
            "Cordialement / Kind regards,",
            f.person,
            f.vendor,
            *_quote_lines(c, c.t.quote_style),
        ]
    )


# ----------------------------------------------------------------------------------------------------------
# Layout templates
# ----------------------------------------------------------------------------------------------------------


def _layout_01(c: _Ctx) -> str:
    f = c.f
    header: list[tuple[str, str]] = [
        (c.lab("number", "Invoice No."), f.number),
        (c.lab("date", "Invoice Date"), c.d(f.inv_date)),
    ]
    header += _due_rows(c, "Due Date", "Terms", "net")
    if f.po is not None:
        header.append((c.lab("po", "PO Number"), f.po))
    table = _items_table(c, "spaces", ("Description", "Qty", "Unit price", "Amount"))
    return _join(
        [
            "INVOICE",
            "",
            f.vendor,
            f.street,
            f.city,
            f"{c.vendor_email()} | {f.phone}",
            "",
            "Bill to:",
            f.customer,
            f.cstreet,
            f.ccity,
            "",
            *_kv(header, ": ", width=16),
            "",
            *table,
            "",
            *_right(_total_rows(c, _TotalLabels(total_tax="Total incl. tax")), len(table[0])),
            "",
            f"All amounts in {f.currency}. Thank you for your business.",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_02(c: _Ctx) -> str:
    f = c.f
    right: list[str] = [
        "INVOICE",
        f"{c.lab('number', 'Number')}: {f.number}",
        f"{c.lab('date', 'Date')}: {c.d(f.inv_date)}",
    ]
    right += [f"{a}: {b}" for a, b in _due_rows(c, "Due", "Terms", "days_net")]
    if f.po is not None:
        right.append(f"{c.lab('po', 'Customer PO')}: {f.po}")
    left = [f.vendor, f.street, f.city, f"Tel {f.phone}", f"{f.slug}@example.com"]
    top = [
        f"{(left[i] if i < len(left) else ''):<40}{right[i] if i < len(right) else ''}".rstrip()
        for i in range(max(len(left), len(right)))
    ]
    rows = [f"{desc:<34}{qty:>4} x {c.n(unit):>10} = {c.n(qty * unit):>11}" for desc, qty, unit in f.items]
    return _join(
        [
            *top,
            "",
            f"Customer: {f.customer}, {f.ccity}",
            "",
            *rows,
            "",
            *_right(_total_rows(c, _TotalLabels(total="TOTAL", total_tax="TOTAL")), 66),
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_03(c: _Ctx) -> str:
    f = c.f
    w = 34
    lines: list[str | None] = [
        f.vendor.center(w).rstrip(),
        f.street.center(w).rstrip(),
        f.city.center(w).rstrip(),
        "-" * w,
        "INVOICE / RECEIPT".center(w).rstrip(),
        f"{c.lab('number', 'Receipt No')}: {f.number}",
        f"{c.lab('date', 'Date')}: {c.d(f.inv_date)} {f.hour:02d}:{f.minute:02d}",
        "-" * w,
    ]
    for desc, qty, unit in f.items:
        lines.append(f"{qty} x {desc}")
        lines.append(f"{c.n(qty * unit):>{w}}")
    lines.append("-" * w)
    lines += _right(
        _total_rows(c, _TotalLabels(total="TOTAL", total_tax="TOTAL", paid="DEPOSIT", remaining="REMAINING")), w
    )
    lines.append("-" * w)
    lines += _kv(_due_rows(c, "Due", "Terms", "net"))
    if f.po is not None:
        lines.append(f"{c.lab('po', 'PO')}: {f.po}")
    lines += [
        f"Currency: {f.currency}",
        c.ch("Thank you!", "Thank you for your visit!"),
        *_quote_lines(c, c.t.quote_style),
    ]
    return _join(lines)


def _layout_04(c: _Ctx) -> str:
    f = c.f
    rows: list[tuple[str, str]] = [
        (c.lab("number", "Invoice Number"), f.number),
        (c.lab("date", "Invoice Date"), c.d(f.inv_date)),
    ]
    rows += _due_rows(c, "Payment Due", "Payment Terms", "net", with_terms=True)
    if f.po is not None:
        rows.append((c.lab("po", "P.O. Number"), f.po))
    return _join(
        [
            f.vendor,
            f"{f.street}, {f.city}",
            f"Phone {f.phone} - {f.slug}@example.com",
            "",
            "INVOICE (dates MM/DD/YYYY)",
            "",
            *_kv(rows, ": ", width=18),
            "",
            f"Sold to: {f.customer}",
            "",
            *_items_table(c, "spaces", ("Item", "Qty", "Rate", "Amount")),
            "",
            *_kv(_total_rows(c, _TotalLabels(total="Invoice Total", total_tax="Invoice Total")), ": ", width=34),
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_05(c: _Ctx) -> str:
    f = c.f
    meta: list[tuple[str, str]] = [
        (c.lab("number", "Invoice"), f.number),
        (c.lab("date", "Date"), c.d(f.inv_date)),
    ]
    meta += _due_rows(c, "Due", "Terms", "net")
    if f.po is not None:
        meta.append((c.lab("po", "PO"), f.po))
    totals = _total_rows(c, _TotalLabels(total="**Total**", total_tax="**Total**"))
    return _join(
        [
            f"# {f.vendor}",
            "",
            f"**{c.lab('number', 'Invoice')}** {f.number}  ",
            f"Billed to: {f.customer}",
            "",
            *[f"- {a}: {b}" for a, b in meta[1:]],
            "",
            *_items_table(c, "pipes"),
            "",
            *[f"{a}: {b}" for a, b in totals],
            "",
            f"Questions: {c.vendor_email()}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_06(c: _Ctx) -> str:
    f = c.f
    rows: list[tuple[str, str]] = [
        (c.lab("number", "Rechnungsnummer / Invoice No."), f.number),
        (c.lab("date", "Rechnungsdatum / Invoice date"), c.d(f.inv_date)),
        ("Lieferdatum / Delivery date", c.d(f.delivery)),
    ]
    rows += _due_rows(c, "Fällig am / Due date", "Zahlungsbedingungen / Terms", "de")
    if f.po is not None:
        rows.append((c.lab("po", "Bestellnummer / PO number"), f.po))
    table = _items_table(c, "spaces", ("Beschreibung / Description", "Menge", "Preis", "Betrag"))
    totals = _total_rows(
        c,
        _TotalLabels(
            subtotal="Zwischensumme / Subtotal",
            tax="USt. / VAT",
            total="Gesamtbetrag / Total",
            total_tax="Gesamtbetrag / Total",
            paid="Anzahlung / Deposit",
            remaining="Restbetrag / Remaining",
            previous="Saldovortrag / Previous balance",
        ),
    )
    return _join(
        [
            "RECHNUNG / INVOICE",
            "",
            f"{f.vendor} - {f.street} - {f.city}",
            "",
            "An / To:",
            f.customer,
            f.cstreet,
            f.ccity,
            "",
            *_kv(rows, ": ", width=32),
            "",
            *table,
            "",
            *_right(totals, len(table[0])),
            f"Währung / Currency: {f.currency}",
            "",
            f"IBAN: {EXAMPLE_IBANS[1]}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_07(c: _Ctx) -> str:
    f = c.f
    lines: list[str | None] = [
        "# document export",
        "[document]",
        "type = invoice",
        f"{c.lab('number', 'number')} = {f.number}",
        f"{c.lab('date', 'date')} = {c.d(f.inv_date)}",
    ]
    for label, value in _due_rows(c, "due_date", "payment_terms", "net"):
        lines.append(f"{label} = {value}")
    if f.po is not None:
        lines.append(f"{c.lab('po', 'purchase_order')} = {f.po}")
    lines += [
        "",
        "[seller]",
        f"name = {f.vendor}",
        f"email = {c.vendor_email()}",
        "",
        "[buyer]",
        f"name = {f.customer}",
        "",
    ]
    lines.append("[amounts]")
    lines.append(f"currency = {f.currency}")
    if c.show_subtotal:
        lines.append(f"net = {c.n(f.subtotal)}")
    if f.tax is not None:
        lines.append(f"{c.lab('tax', 'tax')} = {c.n(f.tax)}")
        lines.append(f"tax_rate = {c.rate()}")
    lines.append(f"{c.lab('total', 'gross_total')} = {c.n(f.total)}")
    if c.pp is not None:
        lines.append(f"prepaid = {c.n(c.pp)}")
        lines.append(f"open_amount = {c.n(f.total - c.pp)}")
    if c.pb is not None:
        lines.append(f"previous_balance = {c.n(c.pb)}")
    lines += ["", "[lines]"]
    for i, (desc, qty, unit) in enumerate(f.items, start=1):
        lines.append(f"{i} = {desc}; {qty}; {c.n(unit)}")
    lines += _quote_lines(c, c.t.quote_style)
    return _join(lines)


def _layout_08(c: _Ctx) -> str:
    f = c.f
    period = f"{_MONTHS[f.delivery.month - 1]} {f.delivery.year}"
    rows: list[tuple[str, str]] = [
        (c.lab("number", "Invoice number"), f.number),
        (c.lab("date", "Invoice date"), c.d(f.inv_date)),
        ("Service period", period),
    ]
    rows += _due_rows(c, "Due date", "Terms", "days_net")
    if f.po is not None:
        rows.append((c.lab("po", "Client PO"), f.po))
    work = [f"  {c.ch(*_SERVICES):<24}{qty:>3} h @ {c.n(unit)} = {c.n(qty * unit)}" for _, qty, unit in f.items]
    return _join(
        [
            f"{f.vendor} - Professional Services",
            f"{f.street}, {f.city}",
            "",
            f"SERVICE INVOICE for {f.customer}",
            "",
            *_kv(rows, ": ", width=18),
            "",
            "Work performed:",
            *work,
            "",
            *_kv(_total_rows(c, _gross_labels(c, "Amount payable", "Amount payable")), ": ", width=34),
            f"(amounts in {f.currency})",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_09(c: _Ctx) -> str:
    f = c.f
    inner: list[str] = [
        f"INVOICE  {c.lab('number', 'No.')} {f.number}",
        f"{c.lab('date', 'Date')}: {c.d(f.inv_date)}",
    ]
    inner += [f"{a}: {b}" for a, b in _due_rows(c, "Due", "Terms", "net")]
    if f.po is not None:
        inner.append(f"{c.lab('po', 'Purchase order')}: {f.po}")
    inner += ["", f"From: {f.vendor}", f"To:   {f.customer}"]
    width = max(len(s) for s in inner) + 2
    box = ["+" + "-" * width + "+", *[f"| {s:<{width - 1}}|" for s in inner], "+" + "-" * width + "+"]
    table = _items_table(c, "box")
    return _join(
        [
            *box,
            "",
            *table,
            *_right(_total_rows(c, _TotalLabels(total="TOTAL", total_tax="TOTAL")), len(table[0])),
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_10(c: _Ctx) -> str:
    f = c.f
    rows: list[tuple[str, str]] = [
        (c.lab("number", "Invoice #"), f.number),
        (c.lab("date", "Invoice date"), c.d(f.inv_date)),
    ]
    rows += _due_rows(c, "Due date", "Terms", "net", with_terms=True)
    if f.po is not None:
        rows.append((c.lab("po", "P.O. #"), f.po))
    return _join(
        [
            f"BILL FROM {f.vendor}",
            f"Remit to: {f.street}, {f.city}",
            "",
            f"Bill to: {f.customer}",
            f"         {f.cstreet}, {f.ccity}",
            "",
            *_kv(rows, "  ", width=16),
            "",
            *_items_table(c, "spaces", ("Description", "Qty", "Price", "Line total")),
            "",
            *_kv(_total_rows(c, _TotalLabels(total="Invoice amount", total_tax="Invoice amount")), "  ", width=34),
            "",
            f"Make payments to {f.vendor} | Questions: {f.phone}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_11(c: _Ctx) -> str:
    f = c.f
    head = [f.vendor, f"{c.lab('number', 'Inv #')} {f.number}", f"{c.lab('date', 'Date')} {c.d(f.inv_date)}"]
    head += [f"{a} {b}" for a, b in _due_rows(c, "Due", "Terms", "net")]
    if f.po is not None:
        head.append(f"{c.lab('po', 'PO')} {f.po}")
    items = [f"{desc} - {qty} x {c.n(unit)} = {c.n(qty * unit)}" for desc, qty, unit in f.items]
    totals = " | ".join(f"{a} {b}" for a, b in _total_rows(c, _TotalLabels(total="Total", total_tax="Total")))
    return _join(
        [
            " | ".join(head),
            f"Bill to: {f.customer}",
            *items,
            totals,
            f"Currency {f.currency}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_12(c: _Ctx) -> str:
    f = c.f
    rows: list[tuple[str, str]] = [
        (c.lab("number", "Facture n° / Invoice no."), f.number),
        (c.lab("date", "Date de facture / Invoice date"), c.d(f.inv_date)),
    ]
    rows += _due_rows(c, "Échéance / Due date", "Conditions / Terms", "fr")
    if f.po is not None:
        rows.append((c.lab("po", "Bon de commande / PO"), f.po))
    totals = _total_rows(
        c,
        _TotalLabels(
            subtotal="Total HT / Net total",
            tax="TVA / VAT",
            total="Total / Total",
            total_tax="Total TTC / Total incl. VAT",
            paid="Acompte / Deposit",
            remaining="Reste dû / Remaining",
            previous="Solde antérieur / Previous balance",
        ),
    )
    table = _items_table(c, "spaces", ("Désignation / Item", "Qté", "PU", "Montant"))
    return _join(
        [
            "FACTURE / INVOICE",
            "",
            f"Émetteur / Seller: {f.vendor}",
            f"{f.street}, {f.city}",
            f"Client / Customer: {f.customer}",
            "",
            *_kv(rows, " : ", width=34),
            "",
            *table,
            "",
            *_right(totals, len(table[0])),
            f"Devise / Currency : {f.currency}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_13(c: _Ctx) -> str:
    f = c.f
    rows: list[tuple[str, str]] = [
        ("Supplier", f.vendor),
        ("Customer", f.customer),
        (c.lab("number", "Invoice number"), f.number),
        (c.lab("date", "Date of issue"), c.d(f.inv_date)),
    ]
    rows += _due_rows(c, "Due date", "Payment terms", "days_net")
    if f.po is not None:
        rows.append((c.lab("po", "Purchase order"), f.po))
    rows += _total_rows(c, _TotalLabels(total="Invoice total", total_tax="Invoice total incl. tax"))
    return _join(
        [
            "INVOICE SUMMARY",
            "",
            *[f"{label} {'.' * max(3, 30 - len(label))} {value}" for label, value in rows],
            "",
            "Line items:",
            *[f"  {qty} x {desc} at {c.m(unit)}" for desc, qty, unit in f.items],
            "",
            f"Contact: {c.vendor_email()}, {f.phone}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_14(c: _Ctx) -> str:
    f = c.f
    acct = f"AC-{_letters(c.rng, 2)}{_digits(c.rng, 4)}"
    rows: list[tuple[str, str]] = [
        ("Account number", acct),
        (c.lab("number", "Bill number"), f.number),
        (c.lab("date", "Bill date"), c.d(f.inv_date)),
    ]
    rows += _due_rows(c, "Pay by", "Payment terms", "days_net")
    if f.po is not None:
        rows.append((c.lab("po", "Your order ref"), f.po))
    return _join(
        [
            f"{f.vendor} - YOUR BILL",
            "Dates are shown as DD/MM/YYYY.",
            "",
            f"{f.customer}",
            f"{f.cstreet}, {f.ccity}",
            "",
            *_kv(rows, ": ", width=18),
            "",
            "Charges this period",
            *[f"  {desc:<32}{c.n(qty * unit):>12}" for desc, qty, unit in f.items],
            "",
            *_kv(_total_rows(c, _gross_labels(c, "Amount payable", "Amount payable", "Bill total")), ": ", width=34),
            f"All charges in {f.currency}.",
            "",
            f"Pay online or by bank transfer to IBAN {EXAMPLE_IBANS[0]}.",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


def _layout_15(c: _Ctx) -> str:
    f = c.f
    title = "FAKTURA VAT / VAT INVOICE" if f.tax is not None else "FAKTURA / INVOICE"
    rows: list[tuple[str, str]] = [
        (c.lab("number", "Nr faktury / Invoice no."), f.number),
        (c.lab("date", "Data wystawienia / Issue date"), c.d(f.inv_date)),
    ]
    rows += _due_rows(c, "Termin płatności / Due date", "Warunki / Terms", "pl")
    if f.po is not None:
        rows.append((c.lab("po", "Numer zamówienia / PO number"), f.po))
    table = _items_table(c, "spaces", ("Nazwa / Item", "Ilość", "Cena", "Wartość"))
    totals = _total_rows(
        c,
        _TotalLabels(
            subtotal="Wartość netto / Net value",
            tax="VAT",
            total="Razem / Total",
            total_tax="Razem brutto / Total gross",
            paid="Zaliczka / Deposit",
            remaining="Pozostało / Remaining",
            previous="Saldo poprzednie / Previous balance",
        ),
    )
    return _join(
        [
            title,
            "",
            f"Sprzedawca / Seller: {f.vendor}, {f.street}, {f.city}",
            f"Nabywca / Buyer: {f.customer}, {f.ccity}",
            "",
            *_kv(rows, ": ", width=32),
            "",
            *table,
            "",
            *_right(totals, len(table[0])),
            f"Waluta / Currency: {f.currency}",
            *_quote_lines(c, c.t.quote_style),
        ]
    )


# ----------------------------------------------------------------------------------------------------------
# Template registry and split
# ----------------------------------------------------------------------------------------------------------

# fmt: off
_TEMPLATE_LIST: tuple[_Template, ...] = (
    _Template("email-01", "email", _email_01, _num_inv, _po_dash, "dmy_long", "symbol", quote_style="reply"),
    _Template("email-02", "email", _email_02, _num_ll, _po_compact, "iso", quote_style="original"),
    _Template("email-03", "email", _email_03, _num_dept, _po_ord, "mdy_long", "code_after", eu_style="eu_space",
              quote_style="reply"),
    _Template("email-04", "email", _email_04, _num_f, _po_p, "dmy_short", "symbol", num_style="plain",
              quote_style="nested"),
    _Template("email-05", "email", _email_05, _num_re, _po_bst, "dmy_long", quote_style="original"),
    _Template("email-06", "email", _email_06, _num_trx, _po_dash, "iso", "code_after", eu_style="eu_space",
              quote_style="reply"),
    _Template("email-07", "email", _email_07, _num_zone, _po_prefix, "dmy_dash", "symbol", quote_style="reply"),
    _Template("email-08", "email", _email_08, _num_sub, _po_mixed, "mdy_long"),
    _Template("email-09", "email", _email_09, _num_re, _po_bst, "dotted", "code_after", currencies=("EUR", "CHF"),
              quote_style="reply"),
    _Template("email-10", "email", _email_10, _num_sub, _po_compact, "dmy_long", "symbol", eu_style="eu_space",
              quote_style="original"),
    _Template("email-11", "email", _email_11, _num_inv, _po_ord, "mdy_long", quote_style="nested"),
    _Template("email-12", "email", _email_12, _num_seq, _po_prefix, "iso", "bare", num_style="plain",
              eu_style="eu_space", quote_style="reply"),
    _Template("email-13", "email", _email_13, _num_ll, _po_dash, "dmy_short", "code_after"),
    _Template("email-14", "email", _email_14, _num_f, _po_p, "dotted", "symbol", num_style="plain",
              quote_style="reply"),
    _Template("email-15", "email", _email_15, _num_fv, _po_mixed, "dotted", "symbol", eu_style="eu_space",
              currencies=("EUR", "CHF"), quote_style="original"),
    _Template("layout-01", "layout", _layout_01, _num_inv, _po_dash, "iso", "bare"),
    _Template("layout-02", "layout", _layout_02, _num_ll, _po_bst, "dmy_short", eu_style="eu_space"),
    _Template("layout-03", "layout", _layout_03, _num_trx, _po_p, "dotted", "bare", num_style="plain"),
    _Template("layout-04", "layout", _layout_04, _num_zone, _po_compact, "us_slash", "symbol",
              currencies=("USD", "EUR", "GBP")),
    _Template("layout-05", "layout", _layout_05, _num_f, _po_prefix, "dmy_long", "symbol", quote_style="nested"),
    _Template("layout-06", "layout", _layout_06, _num_re, _po_bst, "dotted", "code_after", currencies=("EUR", "CHF")),
    _Template("layout-07", "layout", _layout_07, _num_dept, _po_ord, "iso", "bare", num_style="plain",
              eu_style="eu_space"),
    _Template("layout-08", "layout", _layout_08, _num_sub, _po_mixed, "mdy_long"),
    _Template("layout-09", "layout", _layout_09, _num_seq, _po_dash, "dmy_dash", quote_style="reply"),
    _Template("layout-10", "layout", _layout_10, _num_inv, _po_p, "mdy_long", "symbol", currencies=("USD", "GBP")),
    _Template("layout-11", "layout", _layout_11, _num_trx, _po_compact, "dmy_short"),
    _Template("layout-12", "layout", _layout_12, _num_fv, _po_ord, "dotted", "symbol", eu_style="eu_space",
              currencies=("EUR", "CHF")),
    _Template("layout-13", "layout", _layout_13, _num_zone, _po_bst, "dmy_long"),
    _Template("layout-14", "layout", _layout_14, _num_seq, _po_prefix, "uk_slash", "symbol",
              currencies=("GBP", "EUR")),
    _Template("layout-15", "layout", _layout_15, _num_fv, _po_mixed, "dotted", "code_after", eu_style="eu_space",
              currencies=("PLN", "EUR")),
)
# fmt: on

#: Every template id, 15 email bodies then 15 plain-text layouts.
TEMPLATES: tuple[str, ...] = tuple(t.id for t in _TEMPLATE_LIST)

#: The split is by template and fixed in code, balanced by kind.
SPLIT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "train": (
        *(f"email-{i:02d}" for i in range(1, 11)),
        *(f"layout-{i:02d}" for i in range(1, 11)),
    ),
    "valid": ("email-11", "email-12", "layout-11", "layout-12"),
    "test": ("email-13", "email-14", "email-15", "layout-13", "layout-14", "layout-15"),
}
_SPLIT_OF = {template: split for split, templates in SPLIT_TEMPLATES.items() for template in templates}


def split_of(template: str) -> SplitName:
    """The split a template belongs to."""
    try:
        split = _SPLIT_OF[template]
    except KeyError:
        raise ValueError(f"unknown invoice template {template!r}") from None
    return cast(SplitName, split)


# ----------------------------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------------------------


def _company(rng: random.Random) -> str:
    parts = [rng.choice(_START) + rng.choice(_MID) + rng.choice(_END)]
    if rng.random() < 0.6:
        parts.append(rng.choice(_SECTORS))
    parts.append(rng.choice(_SUFFIXES))
    return " ".join(parts)


def _town(rng: random.Random) -> str:
    return f"{_digits(rng, 5)} {rng.choice(_START)}{rng.choice(_TOWN_ENDS)}"


def _street(rng: random.Random) -> str:
    return f"{rng.randrange(1, 180)} {rng.choice(_START)}{rng.choice(_MID)} {rng.choice(_STREET_KINDS)}"


def _amount_not_in(rng: random.Random, low: int, high: int, taken: Sequence[int]) -> int:
    value = rng.randrange(low, high)
    while value in taken:
        value = rng.randrange(low, high)
    return value


def _sample_facts(rng: random.Random, tpl: _Template, traits: Sequence[str]) -> _Facts:
    currency = rng.choice(tpl.currencies)
    vendor = _company(rng)
    customer = _company(rng)
    while customer.split(" ")[0] == vendor.split(" ")[0]:
        customer = _company(rng)
    inv_date = _FIRST_DAY + dt.timedelta(days=rng.randrange(_N_DAYS))
    net_only = "net_terms_due" in traits
    missing: tuple[str, ...] = ()
    if "missing_optional" in traits:
        missing = rng.choice(_MISSING_SUBSETS_KEEP_DUE if net_only else _MISSING_SUBSETS)
    due_days = rng.choice(_TERM_DAYS)
    due_date = None if "due_date" in missing else inv_date + dt.timedelta(days=due_days)

    items: list[tuple[str, int, int]] = []
    for _ in range(rng.choice((1, 1, 2, 2, 3, 4))):
        low, high = rng.choice(_PRICE_BANDS)
        items.append((rng.choice(_ITEMS), rng.choice(_QTYS), rng.randrange(low, high) // 5 * 5))
    subtotal = sum(qty * unit for _, qty, unit in items)
    rate_bp = rng.choice(_TAX_RATES_BP[currency])
    tax = None if "tax_amount" in missing else (subtotal * rate_bp + 5000) // 10000
    total = subtotal + (tax or 0)

    number = tpl.number(rng, inv_date)
    po_value = tpl.po(rng)
    po = None if "po_number" in missing else po_value

    distractors: dict[str, int] = {}
    if "distractor_amounts" in traits:
        kinds = ["previous_balance", "partial_payment"] + (["subtotal"] if tax is not None else [])
        rng.shuffle(kinds)
        for kind in sorted(kinds[: rng.choice((1, 1, 2))]):
            if kind == "subtotal":
                distractors[kind] = subtotal
            elif kind == "previous_balance":
                distractors[kind] = _amount_not_in(rng, 2000, 400000, [total, subtotal, tax or 0])
            else:
                paid = max(100, total * rng.randrange(20, 61) // 100 // 100 * 100)
                distractors[kind] = paid if paid < total else total // 2

    older = None
    if "quoted_reply_chain" in traits:
        o_date = inv_date - dt.timedelta(days=rng.randrange(20, 150))
        o_number = tpl.number(rng, o_date)
        while o_number == number:
            o_number = tpl.number(rng, o_date)
        taken = [total, subtotal, tax or 0, *distractors.values()]
        older = _Older(
            number=o_number,
            date=o_date,
            sent=o_date + dt.timedelta(days=rng.randrange(0, 3)),
            amount=_amount_not_in(rng, 2000, 1500000, taken) // 5 * 5,
        )

    sent = inv_date + dt.timedelta(days=rng.randrange(0, 4))
    if due_date is not None:
        reminder = due_date + dt.timedelta(days=rng.randrange(3, 25))
    else:
        reminder = inv_date + dt.timedelta(days=rng.randrange(20, 45))
    if sent == due_date:
        sent += dt.timedelta(days=1)
    delivery = inv_date - dt.timedelta(days=rng.randrange(0, 10))

    typo_keys: tuple[str, ...] = ()
    if "label_typos" in traits:
        optional = [key for key, present in (("tax", tax is not None), ("po", po is not None)) if present]
        if due_date is not None:
            optional.append("terms" if net_only else "due")
        keys = [rng.choice(("number", "date", "total"))]
        if optional and rng.random() < 0.5:
            keys.append(rng.choice(optional))
        typo_keys = tuple(keys)

    return _Facts(
        vendor=vendor,
        slug=vendor.split(" ")[0].lower(),
        customer=customer,
        cslug=customer.split(" ")[0].lower(),
        number=number,
        inv_date=inv_date,
        due_days=due_days,
        due_date=due_date,
        net_only=net_only and due_date is not None,
        currency=currency,
        items=tuple(items),
        subtotal=subtotal,
        rate_bp=rate_bp,
        tax=tax,
        total=total,
        po=po,
        distractors=distractors,
        older=older,
        sent=sent,
        reminder=reminder,
        delivery=delivery,
        hour=rng.randrange(7, 19),
        minute=rng.randrange(60),
        person=f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LETTERS)}.",
        colleague=f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LETTERS)}.",
        phone=f"+1 555-01{rng.randrange(100):02d}",
        street=_street(rng),
        city=_town(rng),
        cstreet=_street(rng),
        ccity=_town(rng),
        typo_keys=typo_keys,
    )


def _gold(f: _Facts) -> dict[str, Any]:
    return {
        "vendor_name": f.vendor,
        "invoice_number": f.number,
        "invoice_date": f.inv_date.isoformat(),
        "due_date": f.due_date.isoformat() if f.due_date is not None else None,
        "currency": f.currency,
        "total_amount": f.total / 100,
        "tax_amount": f.tax / 100 if f.tax is not None else None,
        "po_number": f.po,
    }


def _make_doc(rng: random.Random, tpl: _Template, index: int, apply_typos: bool) -> InvoiceDoc:
    traits = tuple(trait for trait in TRAITS if rng.random() < TRAIT_RATES[trait])
    facts = _sample_facts(rng, tpl, traits)
    ctx = _Ctx(rng, tpl, facts, "eu_number_format" in traits, apply_typos)
    return InvoiceDoc(
        id=f"{tpl.id}-{index:03d}",
        template=tpl.id,
        kind=tpl.kind,
        text=tpl.render(ctx),
        gold=_gold(facts),
        traits=traits,
        split=split_of(tpl.id),
    )


def _generate(seed: int, per_template: int, apply_typos: bool) -> list[InvoiceDoc]:
    rng = random.Random(seed)
    docs: list[InvoiceDoc] = []
    for index in range(per_template):
        for tpl in _TEMPLATE_LIST:
            docs.append(_make_doc(rng, tpl, index, apply_typos))
    return docs


def generate(seed: int = SEED, per_template: int = PER_TEMPLATE) -> list[InvoiceDoc]:
    """All documents in a fixed order: document ``i`` of every template, then ``i + 1``.

    A smaller ``per_template`` gives a prefix of the full list.
    """
    return _generate(seed, per_template, apply_typos=True)


def quick_subset(docs: Sequence[InvoiceDoc], split: str) -> list[InvoiceDoc]:
    """The quick-profile subset of one split: the first documents of each of its templates, balanced.

    With the full list from :func:`generate` this is always the same set, a subset of the full split.
    """
    if split not in QUICK_SIZES:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(QUICK_SIZES)}")
    templates = SPLIT_TEMPLATES[split]
    base, extra = divmod(QUICK_SIZES[split], len(templates))
    quota = {template: base + (1 if i < extra else 0) for i, template in enumerate(templates)}
    taken = dict.fromkeys(templates, 0)
    out: list[InvoiceDoc] = []
    for doc in docs:
        if doc.template in quota and taken[doc.template] < quota[doc.template]:
            taken[doc.template] += 1
            out.append(doc)
    if len(out) != QUICK_SIZES[split]:
        raise ValueError(f"not enough documents for the quick {split} subset ({len(out)} < {QUICK_SIZES[split]})")
    return out


_SAMPLE_TRAITS = (None, *TRAITS)


def sample_docs(n: int = 20, docs: Sequence[InvoiceDoc] | None = None) -> list[InvoiceDoc]:
    """A fixed sample for the README: one document per training template, alternating kinds, covering every trait."""
    pool = list(docs) if docs is not None else generate()
    emails = [t for t in SPLIT_TEMPLATES["train"] if t.startswith("email")]
    layouts = [t for t in SPLIT_TEMPLATES["train"] if t.startswith("layout")]
    order = [t for pair in zip(emails, layouts, strict=True) for t in pair]
    chosen: list[InvoiceDoc] = []
    chosen_ids: dict[str, bool] = {}
    for j in range(n):
        template = order[j % len(order)]
        want = _SAMPLE_TRAITS[j % len(_SAMPLE_TRAITS)]
        candidates = [d for d in pool if d.template == template and d.id not in chosen_ids]
        if not candidates:
            continue
        match = [d for d in candidates if (not d.traits if want is None else want in d.traits)]
        pick = (match or candidates)[0]
        chosen.append(pick)
        chosen_ids[pick.id] = True
    return chosen


def doc_to_record(doc: InvoiceDoc) -> dict[str, Any]:
    """An ``inputs`` import row: the text, its gold fields and the metadata that fixes its split and group."""
    return {
        "input": doc.text,
        "gold": dict(doc.gold),
        "meta": {
            "id": doc.id,
            "split": doc.split,
            "group": doc.template,
            "template": doc.template,
            "kind": doc.kind,
            "traits": list(doc.traits),
        },
    }
