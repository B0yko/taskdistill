"""Best-effort PII scrub by regex with checksums.

Matches are replaced by typed placeholders (``<EMAIL>``, ``<PHONE>``, ``<IBAN>``, ``<CARD>``, ``<IPV4>``,
``<SSN>``) and counted per kind. Detectors run in a fixed order (email, IBAN, card, SSN, phone, IPv4) so a
longer, checksummed identifier is claimed before a looser pattern can take part of it:

- email: ``local@example.com``-shaped addresses with a letters-only top-level label that is not a file extension;
- IBAN: a country that issues IBANs, two check digits and 11-30 capitals or digits, compact or in groups of
  four separated by single spaces, accepted only when the ISO 13616 mod-97 remainder is 1 (so an ISO 11649
  ``RF`` creditor reference is not an IBAN);
- card: 13-19 digits, contiguous or in groups (4-4-4-4 with an optional short last group, 4-6-5, 4-6-4)
  separated by single spaces or hyphens, accepted only when the Luhn check passes; never a slice of a
  longer contiguous digit run, never after a ``+`` country code;
- SSN: ``ddd-dd-dddd`` with area not 000, 666 or 9xx, group not 00 and serial not 0000;
- phone: ``+`` international numbers (E.164 or grouped, 7-15 digits, exactly 11 for country code 1; a
  trailing group that would exceed that, or that starts a date, amount or percentage, is left out), North
  American ``(415) 555-0123`` / ``415-555-0123`` / ``415.555.0123`` / ``555-0142``, and trunk-prefixed
  national numbers such as ``020 7946 0123``. Dates, amounts, times, years, percentages, ZIP codes and
  identifiers that attach digits to letters (``INV-2026-0413``, ``PO-58213``) do not match;
- IPv4: four octets 0-255 without leading zeros, not inside a longer dotted run such as ``1.2.3.4.5``.

This is not anonymisation: names, addresses and unusual formats pass through unchanged.
"""

from __future__ import annotations

import bisect
import functools
import itertools
import re
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

PII_KINDS = ("email", "phone", "iban", "card", "ipv4", "ssn")
#: The order in which detectors run, independent of the order kinds are configured in.
APPLY_ORDER = ("email", "iban", "card", "ssn", "phone", "ipv4")
PLACEHOLDERS = {kind: f"<{kind.upper()}>" for kind in PII_KINDS}

_MAX_PASSES = 8

# The top-level label is letters only, never a file extension (``logo@2x.png`` is an image name) and never
# followed by another label, so a shorter domain is not taken from a longer one.
_EMAIL = re.compile(
    r"(?<![\w.%+-])[\w%+-][\w.%+-]{0,63}"
    r"@(?:[^\W_](?:[\w-]{0,61}[^\W_])?\.)+"
    r"(?!(?i:png|jpe?g|gif|svg|webp|bmp|ico|tiff?|heic|pdf)(?![\w-]))[^\W\d_]{2,63}"
    r"(?![\w-]|\.[^\W_])"
)

# Prefixes that issue IBANs. ``RF`` (ISO 11649 creditor reference) passes mod-97 but is not one of them.
# fmt: off
IBAN_COUNTRIES = frozenset(
    {
        # SWIFT IBAN registry
        "AD", "AE", "AL", "AT", "AZ", "BA", "BE", "BG", "BH", "BI", "BR", "BY", "CH", "CR", "CY", "CZ", "DE", "DJ",
        "DK", "DO", "EE", "EG", "ES", "FI", "FK", "FO", "FR", "GB", "GE", "GI", "GL", "GR", "GT", "HN", "HR", "HU",
        "IE", "IL", "IQ", "IS", "IT", "JO", "KW", "KZ", "LB", "LC", "LI", "LT", "LU", "LV", "LY", "MC", "MD", "ME",
        "MK", "MN", "MR", "MT", "MU", "NI", "NL", "NO", "OM", "PK", "PL", "PS", "PT", "QA", "RO", "RS", "RU", "SA",
        "SC", "SD", "SE", "SI", "SK", "SM", "SO", "ST", "SV", "TL", "TN", "TR", "UA", "VA", "VG", "XK", "YE",
        # national IBANs outside the registry
        "AO", "BF", "BJ", "CF", "CG", "CI", "CM", "CV", "DZ", "GA", "GQ", "GW", "IR", "KM", "MA", "MG", "ML", "MZ",
        "NE", "SN", "TD", "TG",
        # territories labelled with their own code
        "AX", "BL", "GF", "GG", "GP", "IM", "JE", "MF", "MQ", "NC", "PF", "PM", "RE", "TF", "WF", "YT",
    }
)
# fmt: on

_IBAN_CANDIDATE = re.compile(r"(?<!\w)[A-Z]{2}[0-9]{2}[A-Z0-9]{0,31}(?: [A-Z0-9]+){0,8}")
_IBAN_TOKEN = re.compile(r"[A-Z0-9]+")

# Digits after a letter and ``-`` or ``/`` belong to an identifier, digits after ``1.`` to an amount; a
# number after an abbreviation such as ``No.`` is still a candidate.
_DIGIT_SEQUENCE = re.compile(r"(?<![\w+])(?<!\w[-/])(?<![0-9][.,])(\+)?[0-9]+(?:[ -][0-9]+)*")
_DIGIT_GROUP = re.compile(r"[0-9]+")
_CARD_GROUPED = ((4, 6, 5), (4, 6, 4))

# Digits glued to letters, hyphens or slashes (identifiers) and decimal parts (amounts) are not PII.
_LEFT = r"(?<![\w+/-])(?<![0-9][.,])"
_RIGHT = r"(?![\w-]|[.,/][0-9])"

_SSN = re.compile(_LEFT + r"(?!000|666|9)[0-9]{3}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}" + _RIGHT)

_PHONE_INTL = re.compile(r"(?<![\w+])(?:\(\+[1-9][0-9]{0,2}\)|\+[1-9][0-9]{0,2})(?:[ .-]?(?:\([0-9]{1,4}\)|[0-9]+))*")
_PHONE_NATIONAL = (
    # North American: (415) 555-0123, 415-555-0123, 415.555.0123, 1-415-555-0123
    re.compile(_LEFT + r"(?:1[ .-]?)?(?:\([0-9]{3}\)[ .-]?|[0-9]{3}[ .-])[0-9]{3}[ .-][0-9]{4}" + _RIGHT),
    # trunk prefix 0: 020 7946 0123, 0113 496 0123, (020) 79460123, 020/7946 0123
    re.compile(_LEFT + r"(?:\(0[1-9][0-9]{1,3}\) ?|0[1-9][0-9]{1,3}[ /-])[0-9]{3,4}[ -]?[0-9]{3,5}" + _RIGHT),
    # seven-digit local number: 555-0142
    re.compile(_LEFT + r"[2-9][0-9]{2}-[0-9]{4}" + _RIGHT),
)

_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
_IPV4 = re.compile(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\w|\.[0-9])")

_WORD_AFTER = re.compile(r"\w|[.,][0-9]")
# Dates that a phone number must not end inside: 2026-03-14, 14.03.2026, 03/14/2026, 14-03-2026.
_DATE = re.compile(r"(?<![0-9])(?:[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}|[0-9]{1,2}([./-])[0-9]{1,2}\1[0-9]{2,4})(?![0-9])")


def luhn_valid(digits: str) -> bool:
    """Luhn (mod 10) check of a string of ASCII digits."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_valid(iban: str) -> bool:
    """ISO 13616 check: move the first four characters to the end, map A-Z to 10-35, remainder mod 97 is 1.

    The first two letters must be a country that issues IBANs (:data:`IBAN_COUNTRIES`).
    """
    compact = iban.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}", compact) or compact[:2] not in IBAN_COUNTRIES:
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


def _tainted_end(text: str, end: int) -> bool:
    """True when the character after ``end`` continues a word or a decimal number."""
    return _WORD_AFTER.match(text, end) is not None


def _replace_spans(text: str, spans: list[tuple[int, int]], placeholder: str) -> str:
    out: list[str] = []
    pos = 0
    for start, end in spans:
        out.append(text[pos:start])
        out.append(placeholder)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _regex_spans(pattern: re.Pattern[str]) -> Callable[[str], list[tuple[int, int]]]:
    def find(text: str) -> list[tuple[int, int]]:
        return [m.span() for m in pattern.finditer(text)]

    return find


def _iban_layout(parts: list[str]) -> bool:
    total = sum(len(p) for p in parts)
    if not 15 <= total <= 34:
        return False
    if len(parts) == 1:
        return True
    return all(len(p) == 4 for p in parts[:-1]) and 1 <= len(parts[-1]) <= 4


def _find_ibans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    while (m := _IBAN_CANDIDATE.search(text, pos)) is not None:
        pos = m.start() + 1
        if text[m.start() : m.start() + 2] not in IBAN_COUNTRIES:
            continue
        tokens = [t.span() for t in _IBAN_TOKEN.finditer(text, m.start(), m.end())]
        if _tainted_end(text, m.end()):
            tokens.pop()
        for j in range(len(tokens), 0, -1):
            parts = [text[a:b] for a, b in tokens[:j]]
            if _iban_layout(parts) and iban_valid("".join(parts)):
                spans.append((m.start(), tokens[j - 1][1]))
                pos = tokens[j - 1][1]
                break
    return spans


def _card_layout(lengths: list[int]) -> bool:
    total = sum(lengths)
    if not 13 <= total <= 19:
        return False
    if len(lengths) == 1 or tuple(lengths) in _CARD_GROUPED:
        return True
    return all(n == 4 for n in lengths[:-1]) and 1 <= lengths[-1] <= 4


def _pick_cards(groups: list[tuple[int, int, str]], lo: int, hi: int) -> list[tuple[int, int]]:
    """Card spans among groups[lo:hi], in text order.

    The longest Luhn-valid layout wins (the earliest on ties), then the longest one that overlaps no card
    already taken, and so on. That equals taking the best card of a range and repeating on the groups left
    and right of it, but every window of at most 19 digits is checked once, so the cost stays linear.
    """
    candidates: list[tuple[int, int, int]] = []
    for i in range(lo, hi):
        lengths: list[int] = []
        total = 0
        for j in range(i, hi):
            total += len(groups[j][2])
            if total > 19:
                break
            lengths.append(len(groups[j][2]))
            if total >= 13 and _card_layout(lengths) and luhn_valid("".join(g[2] for g in groups[i : j + 1])):
                candidates.append((-total, i, j))
    candidates.sort()
    taken = [False] * (hi - lo)
    spans: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if not any(taken[i - lo : j - lo + 1]):
            taken[i - lo : j - lo + 1] = [True] * (j - i + 1)
            spans.append((groups[i][0], groups[j][1]))
    return sorted(spans)


def _find_cards(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _DIGIT_SEQUENCE.finditer(text):
        groups = [(g.start(), g.end(), g.group()) for g in _DIGIT_GROUP.finditer(text, m.start(), m.end())]
        lo, hi = (1 if m.group(1) else 0), len(groups)
        if _tainted_end(text, m.end()):
            hi -= 1
        spans.extend(_pick_cards(groups, lo, hi))
    return spans


def _intl_phone_ends(s: str) -> list[int]:
    """Offsets where a ``+`` number matched as ``s`` may end: before a separator, or at the end of ``s``.

    A switch between ``.`` and the other separators ends the number (``+44 7700 900123 3.5%``); the first
    separator after the country code may differ when the next two agree (``+1 415.555.0123``). A first digit
    run longer than a country code that goes on with ``.`` is a signed amount (``+23456.78``), so the number
    may only end with that run.
    """
    cuts = [k for k in range(2, len(s)) if s[k] in " .-(" and s[k - 1] not in " .-("]
    dots = [s[k] == "." for k in cuts]
    first = 1 if len(dots) >= 3 and dots[1] == dots[2] != dots[0] else 0
    switch = next((n for n in range(first + 1, len(dots)) if dots[n] != dots[first]), None)
    ends = [*cuts, len(s)] if switch is None else cuts[: switch + 1]
    plus = s.index("+")
    run_end = len(s) - len(s[plus + 1 :].lstrip("0123456789"))
    if run_end - plus - 1 > 3 and s[run_end : run_end + 1] == ".":
        ends = [k for k in ends if k <= run_end]
    return ends


def _find_intl_phones(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if "+" not in text:
        return spans
    dates = [d.span() for d in _DATE.finditer(text)]
    date_starts = [a for a, _ in dates]
    for m in _PHONE_INTL.finditer(text):
        s = m.group()
        digits = list(itertools.accumulate((ch.isdigit() for ch in s), initial=0))  # digits[k]: digits in s[:k]
        nanp = s[s.index("+") + 1] == "1"  # country code 1 is prefix-free: always 1 + 10 digits
        for k in reversed(_intl_phone_ends(s)):
            end = m.start() + k
            if _tainted_end(text, end):
                continue
            n = bisect.bisect_left(date_starts, end) - 1  # the last date starting before the cut
            if n >= 0 and dates[n][1] > end:
                continue
            if (digits[k] == 11) if nanp else (7 <= digits[k] <= 15):
                spans.append((m.start(), end))
                break
    return spans


def _find_phones(text: str) -> list[tuple[int, int]]:
    spans = _find_intl_phones(text)
    for pattern in _PHONE_NATIONAL:
        for m in pattern.finditer(text):
            if not any(a < m.end() and m.start() < b for a, b in spans):
                spans.append(m.span())
    return sorted(spans)


_FINDERS: dict[str, Callable[[str], list[tuple[int, int]]]] = {
    "email": _regex_spans(_EMAIL),
    "iban": _find_ibans,
    "card": _find_cards,
    "ssn": _regex_spans(_SSN),
    "phone": _find_phones,
    "ipv4": _regex_spans(_IPV4),
}


class PiiScrubber:
    """Replace PII of the configured kinds by typed placeholders and count hits per kind."""

    def __init__(self, kinds: Iterable[str] = PII_KINDS) -> None:
        wanted = list(kinds)
        unknown = sorted(set(wanted) - set(PII_KINDS))
        if unknown:
            raise ValueError(f"unknown PII kind(s) {', '.join(unknown)}; expected a subset of {', '.join(PII_KINDS)}")
        self.kinds: tuple[str, ...] = tuple(kind for kind in APPLY_ORDER if kind in wanted)

    def __repr__(self) -> str:
        return f"PiiScrubber(kinds={self.kinds!r})"

    def _pass(self, text: str, counts: Counter[str]) -> tuple[str, int]:
        hits = 0
        for kind in self.kinds:
            spans = _FINDERS[kind](text)
            if spans:
                text = _replace_spans(text, spans, PLACEHOLDERS[kind])
                counts[kind] += len(spans)
                hits += len(spans)
        return text, hits

    def scrub(self, text: str) -> tuple[str, Counter[str]]:
        """``(scrubbed text, hits per kind)``. Scrubbing is idempotent: placeholders never match again."""
        counts: Counter[str] = Counter()
        for _ in range(_MAX_PASSES):
            text, hits = self._pass(text, counts)
            if not hits:
                break
        return text, counts

    def scrub_value(self, value: Any) -> tuple[Any, Counter[str]]:
        """Scrub every string inside a JSON-like value (dict values, list items); keys and non-strings are kept."""
        counts: Counter[str] = Counter()

        def walk(v: Any) -> Any:
            if isinstance(v, str):
                scrubbed, found = self.scrub(v)
                counts.update(found)
                return scrubbed
            if isinstance(v, dict):
                return {k: walk(item) for k, item in v.items()}
            if isinstance(v, list):
                return [walk(item) for item in v]
            if isinstance(v, tuple):
                return tuple(walk(item) for item in v)
            return v

        return walk(value), counts


@functools.lru_cache(maxsize=16)
def _scrubber(kinds: tuple[str, ...]) -> PiiScrubber:
    return PiiScrubber(kinds)


def scrub(text: str, kinds: Iterable[str] | None = None) -> tuple[str, Counter[str]]:
    """Scrub ``text`` with a shared :class:`PiiScrubber` (all kinds when ``kinds`` is None)."""
    return _scrubber(PII_KINDS if kinds is None else tuple(sorted(set(kinds)))).scrub(text)
