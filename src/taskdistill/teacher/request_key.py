"""The request key: one function shared by the response cache, the replay and proxy capture."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

#: Body fields that determine the teacher's output. Base URL, headers and ``stream`` are excluded;
#: other output-changing fields (provider routing, reasoning controls) go in the recording manifest.
KEY_FIELDS = ("model", "messages", "temperature", "max_tokens", "response_format", "top_p", "seed", "stop")


def _canonical_numbers(obj: Any) -> Any:
    """``obj`` with every integral float replaced by the equal int (recursively); NaN and infinities refused."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"canonical JSON has no representation for the number {obj!r} (not valid JSON)")
        return int(obj) if obj.is_integer() else obj
    if isinstance(obj, Mapping):
        return {key: _canonical_numbers(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_canonical_numbers(value) for value in obj]
    return obj


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, no whitespace, UTF-8, one spelling per number.

    The same JSON value always gives the same bytes:

    - object keys are sorted and no whitespace is written;
    - a number with an integral value is written as an integer (``0.0``, ``-0.0`` and ``0`` all give ``0``,
      ``1e20`` gives ``100000000000000000000``), so ``"temperature": 0`` and ``"temperature": 0.0`` share one
      key whichever language serialised the body; other floats are written as Python's shortest round-trip
      repr (``0.7``); ``NaN`` and the infinities are not JSON numbers and raise :class:`ValueError`;
    - strings are UTF-8 (not ``\\u`` escaped), except a lone UTF-16 surrogate (which UTF-8 cannot encode, e.g.
      from a client that cut a string inside an emoji) is written as its JSON escape ``\\udXXX``; that still
      parses back to the same string, so every ``str`` has a key and no two strings share one.
    """
    text = json.dumps(
        _canonical_numbers(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8", errors="backslashreplace")


def request_key(body: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON (:func:`canonical_json`) of the output-determining fields of a
    chat-completions body.

    Absent fields and explicit ``null`` are equivalent. Raises :class:`ValueError` only for a body holding NaN or
    an infinity, which is not JSON.
    """
    subset = {field: body.get(field) for field in KEY_FIELDS}
    return hashlib.sha256(canonical_json(subset)).hexdigest()
