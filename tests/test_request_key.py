"""The request key is canonical: one key per JSON value, and a key for every string."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest

from taskdistill.config import load_task
from taskdistill.teacher.cache import request_context
from taskdistill.teacher.factory import packaged_recording
from taskdistill.teacher.replay import ReplayTeacher, load_recording
from taskdistill.teacher.request_key import canonical_json, request_key
from taskdistill.teacher.requests import build_teacher_request

BODY: dict[str, Any] = {
    "model": "vendor/model-a",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.0,
    "max_tokens": 24,
    "top_p": 1.0,
}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, b"0"),
        (-0.0, b"0"),
        (1.0, b"1"),
        (24.0, b"24"),
        (1e20, b"100000000000000000000"),
        (0.7, b"0.7"),
        (1e-7, b"1e-07"),
        (True, b"true"),
        (False, b"false"),
        (None, b"null"),
        ([1, 1.0, True, 2.5], b"[1,1,true,2.5]"),
        ((0.0, "a"), b'[0,"a"]'),
        ({"b": 2.0, "a": {"x": [3.0]}}, b'{"a":{"x":[3]},"b":2}'),
    ],
)
def test_canonical_json_writes_one_spelling_per_number(value: Any, expected: bytes) -> None:
    assert canonical_json(value) == expected


def test_int_and_float_spellings_of_one_request_share_a_key() -> None:
    as_ints = {**BODY, "temperature": 0, "top_p": 1, "max_tokens": 24}
    assert request_key(as_ints) == request_key(BODY)
    nested = {**BODY, "response_format": {"type": "json_schema", "json_schema": {"schema": {"maxItems": 3.0}}}}
    nested_int = {**BODY, "response_format": {"type": "json_schema", "json_schema": {"schema": {"maxItems": 3}}}}
    assert request_key(nested) == request_key(nested_int)
    assert request_context({**BODY, "top_k": 40.0}) == request_context({**BODY, "top_k": 40})
    # values that differ as JSON still differ
    assert request_key({**BODY, "temperature": 0.5}) != request_key(BODY)
    assert request_key({**BODY, "temperature": True}) != request_key({**BODY, "temperature": 1})


def test_a_body_serialised_by_another_language_hits_the_packaged_recording() -> None:
    """``JSON.stringify`` writes 0.0 as 0: the parsed body must still replay (it used to be a replay miss)."""
    from taskdistill.demos.invoices import generate

    spec = load_task("invoices")
    recording = load_recording(packaged_recording("invoices"))
    body = build_teacher_request(spec, generate()[0].text)
    assert isinstance(body["temperature"], float) and body["temperature"] == 0.0
    assert request_key(body) in recording

    from_js = json.loads(json.dumps(body).replace('"temperature": 0.0', '"temperature": 0'))
    assert isinstance(from_js["temperature"], int)
    result = asyncio.run(ReplayTeacher(recording, spec).complete(from_js))
    assert result.source == "replay"
    assert result.key == request_key(body)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nan_and_infinities_are_not_json_and_are_refused(value: float) -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_json({"a": [value]})
    with pytest.raises(ValueError):
        request_key({**BODY, "temperature": value})


def test_a_lone_surrogate_gets_a_key_that_parses_back_to_the_same_string() -> None:
    lone = "I lost my card \ud83d"
    body = {**BODY, "messages": [{"role": "user", "content": lone}]}
    key = request_key(body)  # used to raise UnicodeEncodeError
    assert len(key) == 64
    data = canonical_json(lone)
    assert data == b'"I lost my card \\ud83d"'
    data.decode("utf-8")  # valid UTF-8
    assert json.loads(data) == lone
    # no collision with the escaped text written literally, nor with a well-formed emoji
    assert canonical_json("I lost my card \\ud83d") != data
    assert request_key({**BODY, "messages": [{"role": "user", "content": "I lost my card \U0001f600"}]}) != key
    # well-formed text is still written as UTF-8, not escaped
    assert canonical_json("café \U0001f600") == '"café \U0001f600"'.encode()
