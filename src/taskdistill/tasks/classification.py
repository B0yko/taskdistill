"""Classification outputs: map a free-text model answer onto the canonical label set.

The teacher prompt renders labels with spaces instead of underscores, so a teacher answers
``card arrival``, ``Card Arrival``, ``"card arrival"`` or ``Label: card arrival.`` for the canonical
label ``card_arrival``. :func:`label_key` reduces both sides to one comparison key; there is no fuzzy
matching, so a misspelt or paraphrased answer maps to ``None``.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Sequence

#: Characters that wrap a label in Markdown or quoted answers.
_WRAPPERS = "\"'`*“”‘’„«»"
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "′": "'"})
_PREFIX = re.compile(r"^(?:label|intent|category)[\s*_`]*[:：][\s*_`]*")
_SEPARATORS = re.compile(r"[\s_-]+")
#: Terminal punctuation ignored by the second, punctuation-insensitive lookup.
_TERMINAL_PUNCT = ".?!,;:"


def label_key(text: str) -> str:
    """Comparison key of a label or a model answer.

    NFKC, casefold, strip whitespace and wrapping quotes/backticks/asterisks, drop one trailing period and a
    leading ``label:`` / ``intent:`` / ``category:`` prefix, then collapse runs of whitespace, underscores and
    hyphens into one ``_``.
    """
    s = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).casefold()
    period_dropped = False
    previous = None
    while s != previous:
        previous = s
        s = s.strip().strip(_WRAPPERS).strip()
        if not period_dropped and s.endswith("."):
            s = s[:-1]
            period_dropped = True
        s = _PREFIX.sub("", s)
    return _SEPARATORS.sub("_", s).strip("_")


def _loose(key: str) -> str:
    return key.rstrip(_TERMINAL_PUNCT).rstrip("_")


@functools.lru_cache(maxsize=64)
def _index(labels: tuple[str, ...]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    exact: dict[str, list[str]] = {}
    loose: dict[str, list[str]] = {}
    for label in labels:
        key = label_key(label)
        exact.setdefault(key, []).append(label)
        loose.setdefault(_loose(key), []).append(label)
    return exact, loose


def normalise_label(output: str, labels: Sequence[str]) -> str | None:
    """The unique canonical label whose key equals the key of ``output``, else ``None``.

    An exact key match wins. Otherwise the keys are compared once more without terminal punctuation
    (``reverted card payment`` for the label ``reverted_card_payment?``); that match must also be unique.
    """
    key = label_key(output)
    if not key:
        return None
    exact, loose = _index(tuple(labels))
    hits = exact.get(key)
    if hits is not None:
        return hits[0] if len(hits) == 1 else None
    hits = loose.get(_loose(key))
    if hits is not None and len(hits) == 1:
        return hits[0]
    return None
