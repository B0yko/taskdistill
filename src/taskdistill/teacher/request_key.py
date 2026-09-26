"""The request key: one function shared by the response cache, the replay and proxy capture."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

#: Body fields that determine the teacher's output. Base URL, headers and ``stream`` are excluded;
#: other output-changing fields (provider routing, reasoning controls) go in the recording manifest.
KEY_FIELDS = ("model", "messages", "temperature", "max_tokens", "response_format", "top_p", "seed", "stop")


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def request_key(body: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the output-determining fields of a chat-completions body.

    Absent fields and explicit ``null`` are equivalent.
    """
    subset = {field: body.get(field) for field in KEY_FIELDS}
    return hashlib.sha256(canonical_json(subset)).hexdigest()
