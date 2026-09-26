"""Pull the raw task input out of a chat-completions request (shared by curate and serve)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from taskdistill.config import InputSpec

_WS = re.compile(r"\s+")


class InputUnparsed(ValueError):
    """The request does not contain an input matching ``input.from``."""


def _content_text(content: Any) -> str | None:
    """The text of a message's content: a string, or OpenAI content parts that are all text parts.

    None when any part is not a text part or its ``text`` is missing or not a string (null, a number, ...), so the
    request is unparsed rather than an error.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:  # OpenAI content parts
        parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "text":
                return None
            text = part.get("text")
            if not isinstance(text, str):
                return None
            parts.append(text)
        return "".join(parts)
    return None


def extract_input(body: Mapping[str, Any], spec: InputSpec) -> str:
    """Return the task input from ``body`` according to ``input.from``; raise InputUnparsed otherwise."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise InputUnparsed("request has no messages")
    last_user = None
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            last_user = _content_text(message.get("content"))
            break
    if last_user is None:
        raise InputUnparsed("request has no text user message")
    if spec.from_ == "last_user_message":
        return last_user
    assert spec.regex is not None
    match = re.search(spec.regex, last_user, re.DOTALL)
    if match is None or match.group("input") is None:
        raise InputUnparsed("input.regex did not match the last user message")
    return match.group("input")


def normalise_text(text: str) -> str:
    """NFKC, collapse whitespace, strip. Used for input hashing and exact dedupe."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def input_hash(text: str) -> str:
    """SHA-256 of the NFKC-normalised, whitespace-collapsed input."""
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()
