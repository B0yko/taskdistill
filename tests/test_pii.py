from __future__ import annotations

import random
from collections import Counter

import pytest

from taskdistill.curate.pii import (
    APPLY_ORDER,
    IBAN_COUNTRIES,
    PII_KINDS,
    PLACEHOLDERS,
    PiiScrubber,
    iban_valid,
    luhn_valid,
    scrub,
)

SCRUBBER = PiiScrubber()


def one(text: str, kinds: tuple[str, ...] = PII_KINDS) -> tuple[str, Counter[str]]:
    return PiiScrubber(kinds).scrub(text)


# --- checksums -----------------------------------------------------------------------------------


def test_luhn_hand_computed() -> None:
    # 4111111111111111, from the right: undoubled 1 x 8 = 8; doubled 1 x 7 -> 2 x 7 = 14 and 4 -> 8; 8 + 14 + 8 = 30.
    assert luhn_valid("4111111111111111")
    # 4111111111111112: same doubled part, undoubled 7 x 1 + 2 = 9; 9 + 14 + 8 = 31.
    assert not luhn_valid("4111111111111112")
    # textbook example: 7 9 9 2 7 3 9 8 7 1 3 -> 70
    assert luhn_valid("79927398713")
    assert not luhn_valid("79927398710")
    for number in ("5555555555554444", "4012888888881881", "378282246310005"):
        assert luhn_valid(number)


def test_iban_hand_computed() -> None:
    # GB82WEST12345698765432 -> WEST12345698765432GB82 -> W=32 E=14 S=28 T=29 ... G=16 B=11
    assert int("3214282912345698765432161182") % 97 == 1
    assert iban_valid("GB82WEST12345698765432")
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert iban_valid("DE89 3704 0044 0532 0130 00")
    assert not iban_valid("GB83 WEST 1234 5698 7654 32")  # wrong check digits
    assert not iban_valid("GB82 WEST 1234 5698 7654 33")  # one BBAN digit changed
    assert not iban_valid("DE88 3704 0044 0532 0130 00")
    assert not iban_valid("GB82WEST1234")  # BBAN shorter than 11
    assert not iban_valid("")


def test_creditor_reference_is_not_an_iban() -> None:
    # RF18539007547034 -> 539007547034RF18 -> R=27 F=15 -> 539007547034271518, remainder 1 like an IBAN,
    # but RF (ISO 11649 creditor reference) is not a country.
    assert int("539007547034271518") % 97 == 1
    assert "RF" not in IBAN_COUNTRIES
    assert not iban_valid("RF18539007547034")
    assert not iban_valid("RF18 5390 0754 7034")
    assert {"GB", "DE", "FR", "NL", "CH", "PL"} <= IBAN_COUNTRIES


# --- positives -----------------------------------------------------------------------------------

POSITIVES = [
    ("email", "jane.doe@example.com"),
    ("email", "billing+invoices@accounts.example.org"),
    ("email", "ops_team-7@mail.example.test"),
    ("email", "jürgen.weiß@example.org"),
    ("iban", "GB82 WEST 1234 5698 7654 32"),
    ("iban", "GB82WEST12345698765432"),
    ("iban", "DE89 3704 0044 0532 0130 00"),
    ("iban", "DE89370400440532013000"),
    ("card", "4111 1111 1111 1111"),
    ("card", "4111-1111-1111-1111"),
    ("card", "4111111111111111"),
    ("card", "5555 5555 5555 4444"),
    ("card", "4012 8888 8888 1881"),
    ("card", "378282246310005"),
    ("card", "3782 822463 10005"),
    ("ssn", "123-45-6789"),
    ("phone", "+1 415 555 0123"),
    ("phone", "+14155550123"),
    ("phone", "(415) 555-0123"),
    ("phone", "415-555-0123"),
    ("phone", "415.555.0123"),
    ("phone", "555-0142"),
    ("phone", "+44 20 7946 0958"),
    ("phone", "+49 30 1234 5678"),
    ("phone", "+44 (0)20 7946 0958"),
    ("phone", "+1 (415) 555-0123"),
    ("phone", "(+44) 20 7946 0958"),
    ("phone", "+33 1 99 00 12 34"),
    ("phone", "+1 415.555.0123"),
    ("phone", "+1.415.555.0123"),
    ("phone", "+442079460958"),
    ("phone", "+4930 1234 5678"),
    ("phone", "+49 30 1234 5678-0"),
    ("phone", "1-415-555-0123"),
    ("phone", "020 7946 0958"),
    ("phone", "(020) 7946 0123"),
    ("phone", "0113 496 0123"),
    ("phone", "020/7946 0123"),
    ("ipv4", "192.0.2.1"),
    ("ipv4", "198.51.100.7"),
    ("ipv4", "203.0.113.255"),
    ("ipv4", "0.0.0.0"),
    ("ipv4", "127.0.0.1"),
]


@pytest.mark.parametrize(("kind", "value"), POSITIVES)
def test_positive_alone_and_in_context(kind: str, value: str) -> None:
    placeholder = PLACEHOLDERS[kind]
    assert SCRUBBER.scrub(value) == (placeholder, Counter({kind: 1}))
    text = f"Reach us at {value}, thanks."
    assert SCRUBBER.scrub(text) == (f"Reach us at {placeholder}, thanks.", Counter({kind: 1}))
    assert SCRUBBER.scrub(f"({value})") == (f"({placeholder})", Counter({kind: 1}))
    assert SCRUBBER.scrub(f"{value}.") == (f"{placeholder}.", Counter({kind: 1}))


# --- negatives -----------------------------------------------------------------------------------

NEGATIVES = [
    # checksums
    "4111 1111 1111 1112",
    "4111111111111112",
    "5555 5555 5555 4445",
    "124111111111111111134",  # 4111111111111111 inside a longer contiguous run
    "4111  1111  1111  1111",  # double spaces do not join groups
    "4111 111 1111 11111",  # not a card layout
    "GB83 WEST 1234 5698 7654 32",
    "XGB82WEST12345698765432",  # glued to a word
    "GB82WEST12345698765432abc",
    "GB82WEST12345698765432X2",  # a longer token fails the checksum
    "RF18 5390 0754 7034",  # creditor reference: mod-97 passes, RF is not a country
    "RF18539007547034",
    "INV-4111111111111111",  # identifier
    "1.4111111111111111",  # decimal part
    # SSN rules
    "000-12-3456",
    "666-12-3456",
    "912-12-3456",
    "123-00-4567",
    "123-45-0000",
    "123-45-67890",
    "INV-123-45-6789",
    # IPv4 rules
    "256.1.1.1",
    "192.0.2.256",
    "1.2.3.4.5",
    "version 10.2.3.4.1",
    "v1.2.3.4",
    "192.168.01.1",
    "192.0.2",
    # dates
    "2026-03-14",
    "14.03.2026",
    "03/14/2026",
    "14/03/2026",
    "2026/03/14",
    "14-03-2026",
    "March 14, 2026",
    # amounts
    "1.234,56",
    "12,345.67",
    "1234.50",
    "1 234,56",
    "1 234 567,89",
    "-1.234,56 €",
    "+1 234 567,89",
    "USD 1,234,567.00",
    "Adjustment +23456.78 EUR",
    "credit +45678.90",
    "balance +234567.89",
    "+23 456.78",
    # times, years, percentages, postal codes
    "14:30",
    "09:15:00",
    "2026",
    "1999-2026",
    "12.5%",
    "15 %",
    "94103",
    "94103-1234",
    "SW1A 1AA",
    # invoice and PO numbers
    "INV-2026-0413",
    "PO-58213",
    "2026/INV/0042",
    "AB-77120-X",
    "INV-555-0142",
    "PO/415-555-0123",
    "INV2026-00413",
    "Net 30",
    "Qty 12 x 45.00",
    # not email addresses
    "jane@localhost",
    "@example.com",
    "jane@",
    "a@b",
    "3 items @ 45.00",
    "jane@@example.com",
    "logo@2x.png",
    "icon@3x.WEBP",
    "banner@img.example.jpg",
    "a@img.2x.png",
]


@pytest.mark.parametrize("text", NEGATIVES)
def test_negative_unchanged(text: str) -> None:
    for sample in (text, f"Ref {text} due."):
        assert SCRUBBER.scrub(sample) == (sample, Counter())


def test_invoice_body_is_untouched() -> None:
    body = (
        "Invoice INV-2026-0413 (PO-58213)\n"
        "Invoice date: 14.03.2026   Due: 2026-04-13 (Net 30)\n"
        "Subtotal 1.037,45 €  VAT 19 % 197,11 €  Total 1.234,56 €\n"
        "Previous balance: 12,345.67 USD, partial payment 1234.50\n"
        "Ref 2026/INV/0042, order AB-77120-X, 3 items at 09:30\n"
    )
    assert SCRUBBER.scrub(body) == (body, Counter())


def test_gold_values_are_untouched() -> None:
    gold = {
        "vendor_name": "Brask Lumen GmbH",
        "invoice_number": "INV-2026-0413",
        "invoice_date": "2026-03-14",
        "due_date": "2026-04-13",
        "currency": "EUR",
        "total_amount": 1234.56,
        "tax_amount": 197.11,
        "po_number": "PO-58213",
    }
    assert SCRUBBER.scrub_value(gold) == (gold, Counter())


# --- precedence, boundaries, counts --------------------------------------------------------------


def test_iban_digits_are_not_a_card() -> None:
    assert SCRUBBER.scrub("DE89 3704 0044 0532 0130 00") == ("<IBAN>", Counter({"iban": 1}))


def test_email_wins_over_digits_inside_it() -> None:
    assert SCRUBBER.scrub("jane.4155550123@example.com") == ("<EMAIL>", Counter({"email": 1}))


def test_card_followed_by_expiry_and_cvv() -> None:
    text = "card 4111 1111 1111 1111 exp 12/28 cvv 123"
    assert SCRUBBER.scrub(text) == ("card <CARD> exp 12/28 cvv 123", Counter({"card": 1}))
    assert SCRUBBER.scrub("4111 1111 1111 1111 123") == ("<CARD> 123", Counter({"card": 1}))


def test_card_between_number_groups() -> None:
    assert SCRUBBER.scrub("order 12 4111 1111 1111 1111") == ("order 12 <CARD>", Counter({"card": 1}))
    assert SCRUBBER.scrub("12 4111111111111111 34") == ("12 <CARD> 34", Counter({"card": 1}))


def test_two_cards_in_one_digit_sequence() -> None:
    text = "4111 1111 1111 1111 5555 5555 5555 4444"
    assert SCRUBBER.scrub(text) == ("<CARD> <CARD>", Counter({"card": 2}))


def test_longest_card_wins_over_an_earlier_overlapping_one() -> None:
    # 4111111111111111 (groups 0-3, 16 digits) is valid. 1111 1111 1111 1111 113 (groups 1-5, 19 digits):
    # from the right 3 undoubled, nine doubled ones = 18, nine undoubled ones = 9; 3 + 18 + 9 = 30, valid.
    text = "4111 1111 1111 1111 1111 113"
    assert SCRUBBER.scrub(text) == ("4111 <CARD>", Counter({"card": 1}))


def test_card_after_an_abbreviation() -> None:
    assert SCRUBBER.scrub("Card No.4111111111111111") == ("Card No.<CARD>", Counter({"card": 1}))
    assert SCRUBBER.scrub("Karte Nr.4111-1111-1111-1111") == ("Karte Nr.<CARD>", Counter({"card": 1}))


@pytest.mark.timeout(30)
def test_many_cards_in_one_digit_sequence() -> None:
    text = "4111 1111 1111 1111 " * 1500
    assert SCRUBBER.scrub(text) == ("<CARD> " * 1500, Counter({"card": 1500}))


@pytest.mark.timeout(30)
def test_long_run_of_random_groups() -> None:
    rng = random.Random(1)
    text = " ".join(str(rng.randrange(1000, 10000)) for _ in range(20000))
    scrubbed, counts = SCRUBBER.scrub(text)
    assert counts["card"] == scrubbed.count("<CARD>") > 1000
    assert set(counts) == {"card"}
    assert SCRUBBER.scrub(scrubbed) == (scrubbed, Counter())


def test_phone_country_code_is_not_part_of_a_card() -> None:
    text = "+1 415 555 0123 4111 1111 1111 1111"
    assert SCRUBBER.scrub(text) == ("<PHONE> <CARD>", Counter({"phone": 1, "card": 1}))


def test_nanp_international_number_stops_at_eleven_digits() -> None:
    assert SCRUBBER.scrub("+1 415 555 0123 2026-03-14") == ("<PHONE> 2026-03-14", Counter({"phone": 1}))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tel +49 30 1234 5678 14.03.2026", "Tel <PHONE> 14.03.2026"),
        ("+33 1 99 00 12 34 12.03.2026", "<PHONE> 12.03.2026"),
        ("Tel +44 20 7946 0958 1.234,56 EUR", "Tel <PHONE> 1.234,56 EUR"),
        ("Mobile +44 7700 900123 3.5%", "Mobile <PHONE> 3.5%"),
        ("+44 20 7946 0958 2026-03-14", "<PHONE> 2026-03-14"),
        ("+49 30 1234 2026-03-14", "<PHONE> 2026-03-14"),
        ("+49 30 1234 5678 14-03-2026", "<PHONE> 14-03-2026"),
        ("+44 20 7946 0958 03/14/2026", "<PHONE> 03/14/2026"),
        ("+33.1.99.00.12.34 12 items", "<PHONE> 12 items"),
        ("call +442079460958 today", "call <PHONE> today"),
        ("+49 30 12345678-1234", "<PHONE>-1234"),  # 16 digits: the extension is left out
    ],
)
def test_international_number_leaves_following_values_intact(text: str, expected: str) -> None:
    assert SCRUBBER.scrub(text) == (expected, Counter({"phone": 1}))


def test_signed_amounts_are_not_international_numbers() -> None:
    for text in ("+12 345 678,90", "+1 234 567 890,12", "+20.5%", "+1.234,56", "+23456.78", "+1234567.89"):
        assert SCRUBBER.scrub(text) == (text, Counter())


def test_iban_does_not_take_following_words() -> None:
    text = "IBAN GB82 WEST 1234 5698 7654 32 EUR, Thanks"
    assert SCRUBBER.scrub(text) == ("IBAN <IBAN> EUR, Thanks", Counter({"iban": 1}))


def test_iban_after_a_failed_candidate() -> None:
    text = "AB12 GB82 WEST 1234 5698 7654 32"
    assert SCRUBBER.scrub(text) == ("AB12 <IBAN>", Counter({"iban": 1}))
    text = "Reference: RF18 5390 0754 7034, IBAN DE89 3704 0044 0532 0130 00"
    assert SCRUBBER.scrub(text) == ("Reference: RF18 5390 0754 7034, IBAN <IBAN>", Counter({"iban": 1}))


def test_email_top_level_label_rules() -> None:
    assert SCRUBBER.scrub("jane@mail.example.test") == ("<EMAIL>", Counter({"email": 1}))
    assert SCRUBBER.scrub("see logo@2x.png and jane@png.example.org") == (
        "see logo@2x.png and <EMAIL>",
        Counter({"email": 1}),
    )
    # a shorter domain is not taken when the full one ends in a file extension
    assert SCRUBBER.scrub("a@img.logo.png") == ("a@img.logo.png", Counter())


def test_ipv4_with_port_and_prefix() -> None:
    text = "hosts 192.0.2.1:8080 and 198.51.100.0/24."
    assert SCRUBBER.scrub(text) == ("hosts <IPV4>:8080 and <IPV4>/24.", Counter({"ipv4": 2}))


def test_hit_counts_per_kind() -> None:
    text = (
        "From jane@example.com to ops@example.org; call +44 20 7946 0958, (415) 555-0123 or 555-0199. "
        "Pay GB82 WEST 1234 5698 7654 32 or card 5555 5555 5555 4444. SSN 123-45-6789. Host 203.0.113.9."
    )
    scrubbed, counts = SCRUBBER.scrub(text)
    assert counts == Counter({"email": 2, "phone": 3, "iban": 1, "card": 1, "ssn": 1, "ipv4": 1})
    assert scrubbed == (
        "From <EMAIL> to <EMAIL>; call <PHONE>, <PHONE> or <PHONE>. Pay <IBAN> or card <CARD>. SSN <SSN>. Host <IPV4>."
    )


# --- idempotency ---------------------------------------------------------------------------------


def test_placeholders_are_not_rematched() -> None:
    text = " ".join(PLACEHOLDERS[k] for k in APPLY_ORDER)
    assert SCRUBBER.scrub(text) == (text, Counter())


@pytest.mark.parametrize("text", [v for _, v in POSITIVES] + NEGATIVES)
def test_idempotent_on_samples(text: str) -> None:
    once, _ = SCRUBBER.scrub(text)
    assert SCRUBBER.scrub(once) == (once, Counter())


def test_idempotent_on_random_strings() -> None:
    rng = random.Random(7)
    pieces = [*"0123456789" * 4, *" -./,+()@:", "GB82", "WEST", "a", "X", "example.com", "<PHONE>", "\n"]
    for _ in range(2000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randrange(5, 50)))
        once, _ = SCRUBBER.scrub(text)
        assert SCRUBBER.scrub(once) == (once, Counter()), text


# --- scrub_value, kinds, module function ---------------------------------------------------------


def test_scrub_value_recurses_and_keeps_keys_and_non_strings() -> None:
    value = {
        "vendor_name": "Brask Lumen GmbH",
        "contact": {"email": "jane@example.com", "phones": ["+44 20 7946 0958", "555-0142"]},
        "jane@example.com": None,
        "total_amount": 4111111111111111,
        "paid": True,
        "items": [{"note": "card 4111 1111 1111 1111"}, 12.5, None],
        "pair": ("192.0.2.1", 3),
    }
    scrubbed, counts = SCRUBBER.scrub_value(value)
    assert scrubbed == {
        "vendor_name": "Brask Lumen GmbH",
        "contact": {"email": "<EMAIL>", "phones": ["<PHONE>", "<PHONE>"]},
        "jane@example.com": None,
        "total_amount": 4111111111111111,
        "paid": True,
        "items": [{"note": "card <CARD>"}, 12.5, None],
        "pair": ("<IPV4>", 3),
    }
    assert counts == Counter({"email": 1, "phone": 2, "card": 1, "ipv4": 1})
    assert value["contact"]["email"] == "jane@example.com"  # input not mutated


def test_scrub_value_on_scalars() -> None:
    assert SCRUBBER.scrub_value("555-0142") == ("<PHONE>", Counter({"phone": 1}))
    assert SCRUBBER.scrub_value(42) == (42, Counter())
    assert SCRUBBER.scrub_value(None) == (None, Counter())


def test_kinds_filter() -> None:
    text = "jane@example.com 555-0142 192.0.2.1 4111 1111 1111 1111"
    assert one(text, ("email",)) == ("<EMAIL> 555-0142 192.0.2.1 4111 1111 1111 1111", Counter({"email": 1}))
    assert one(text, ("phone", "ipv4")) == (
        "jane@example.com <PHONE> <IPV4> 4111 1111 1111 1111",
        Counter({"phone": 1, "ipv4": 1}),
    )
    assert one(text, ()) == (text, Counter())


def test_kinds_run_in_fixed_order() -> None:
    assert PiiScrubber(["ipv4", "phone", "email"]).kinds == ("email", "phone", "ipv4")
    assert PiiScrubber().kinds == APPLY_ORDER
    assert APPLY_ORDER == ("email", "iban", "card", "ssn", "phone", "ipv4")


def test_without_card_kind_a_card_is_not_a_phone() -> None:
    assert one("4111 1111 1111 1111", ("phone", "ssn", "ipv4")) == ("4111 1111 1111 1111", Counter())


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="passport"):
        PiiScrubber(["email", "passport"])


def test_module_level_scrub() -> None:
    assert scrub("mail jane@example.com, call 555-0142") == ("mail <EMAIL>, call <PHONE>", Counter(email=1, phone=1))
    assert scrub("mail jane@example.com, call 555-0142", kinds=["phone"]) == (
        "mail jane@example.com, call <PHONE>",
        Counter(phone=1),
    )


def test_empty_text() -> None:
    assert SCRUBBER.scrub("") == ("", Counter())


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "text",
    [
        "AB12 " + "ABCD " * 20000,
        "1 " * 50000,
        "1234 " * 20000,
        "+1" + " 2" * 50000,
        "a" * 100000 + "@example.com",
        "x@" + "a." * 30000 + "1",
        "123-" * 30000,
        "1." * 50000,
    ],
    ids=["caps", "digits", "groups", "plus", "local-part", "labels", "dashes", "dots"],
)
def test_long_inputs_are_linear_enough(text: str) -> None:
    scrubbed, _ = SCRUBBER.scrub(text)
    assert isinstance(scrubbed, str)
