"""Capture proxy: byte-for-byte forwarding, header hygiene, task selection, token auth and failure handling."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import sqlite3
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from taskdistill.capture import proxy as proxy_mod
from taskdistill.capture.export import export_file
from taskdistill.capture.proxy import (
    STREAM_NOT_CAPTURED,
    TOKEN_ENV,
    UnsafeBindError,
    create_proxy_app,
    is_local_host,
    run_proxy,
)
from taskdistill.store import Store
from taskdistill.teacher.request_key import request_key

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*httpx2.*")
    from fastapi.testclient import TestClient

UPSTREAM = "http://upstream.test/api/v1"
CLIENT_KEY = "sk-test-client-key-0001"
PROXY_TOKEN = "proxy-token-for-tests-7f3a"
AUTH = {"Authorization": f"Bearer {CLIENT_KEY}", "Content-Type": "application/json"}

# Odd key order, irregular whitespace, an escaped and a raw non-ASCII character and a trailing newline:
# any re-serialisation on the way would change these bytes.
RAW_REQUEST = (
    b'{ "temperature" : 0,\n  "messages":[{"content":"Where is my card? caf\\u00e9 \xc3\xa9","role":"user"}],\n'
    b'"model":"example/teacher",   "max_tokens":24 }\n'
)
UPSTREAM_BODY = (
    b'{"id":"chatcmpl-1","object":"chat.completion","model":"example/teacher-0731",'
    b'"choices":[{"index":0,"message":{"role":"assistant","content":"card_arrival"},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":42,"completion_tokens":3,"total_tokens":45,"cost":0.000123}}'
)
SSE_BODY = (
    b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"card"}}]}\n\n'
    b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"_arrival"}}]}\n\n'
    b"data: [DONE]\n\n"
)
MODELS_BODY = b'{"object":"list","data":[{"id":"example/teacher","object":"model"}]}'
CHAT = "/t/banking/v1/chat/completions"


def _json_response(body: bytes = UPSTREAM_BODY, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(status, content=body, headers={"content-type": "application/json", **headers})


def _rows(store: Store, task: str = "banking") -> list[Any]:
    return list(store.iter_captures(task))


def _all_values(db: Path) -> list[Any]:
    conn = sqlite3.connect(db)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return [v for t in tables for row in conn.execute(f"SELECT * FROM {t}") for v in row]
    finally:
        conn.close()


def _db_bytes(db: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(db.parent.glob(db.name + "*")))


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.sqlite")


@pytest.fixture
def upstream() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=UPSTREAM, assert_all_called=False) as router:
        yield router


@pytest.fixture
def make_proxy(store: Store) -> Iterator[Callable[..., TestClient]]:
    stack = contextlib.ExitStack()
    clients: list[httpx.AsyncClient] = []

    def factory(**kwargs: Any) -> TestClient:
        http = httpx.AsyncClient()
        clients.append(http)
        app = create_proxy_app(kwargs.pop("store", store), UPSTREAM, client=http, **kwargs)
        return stack.enter_context(TestClient(app))

    yield factory
    stack.close()
    for http in clients:
        asyncio.run(http.aclose())


# forwarding ---------------------------------------------------------------------------------------


def test_body_forwarded_byte_for_byte_with_client_authorization(
    upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    route = upstream.post("/chat/completions").mock(return_value=_json_response())
    resp = make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH)

    assert resp.status_code == 200
    assert resp.content == UPSTREAM_BODY
    assert resp.headers["content-type"] == "application/json"
    sent = route.calls.last.request
    assert sent.content == RAW_REQUEST
    assert sent.headers["authorization"] == f"Bearer {CLIENT_KEY}"
    assert sent.headers["content-type"] == "application/json"
    assert str(sent.url) == f"{UPSTREAM}/chat/completions"


def test_captured_row_values(store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]) -> None:
    upstream.post("/chat/completions").mock(return_value=_json_response())
    make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH)

    [row] = _rows(store)
    assert row.task == "banking"
    assert row.source == "proxy"
    assert row.request_key == request_key(json.loads(RAW_REQUEST))
    assert row.request_body == RAW_REQUEST.decode("utf-8")
    assert row.response_body == UPSTREAM_BODY.decode("utf-8")
    assert row.status == 200
    assert (row.prompt_tokens, row.completion_tokens) == (42, 3)
    assert row.cost_usd == pytest.approx(0.000123)
    assert row.upstream_model == "example/teacher-0731"
    assert row.captured is True
    assert row.error is None
    assert row.latency_ms is not None
    assert row.latency_ms >= 0
    assert store.count_captures("banking") == {"total": 1, "captured": 1, "not_captured": 0}


def test_usage_without_cost_and_bool_tokens_are_ignored(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    body = b'{"model":"m","choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":true,"completion_tokens":7}}'
    upstream.post("/chat/completions").mock(return_value=_json_response(body))
    make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH)

    [row] = _rows(store)
    assert (row.prompt_tokens, row.completion_tokens, row.cost_usd) == (None, 7, None)
    assert row.captured is True


def test_request_key_ignores_whitespace_key_order_and_stream(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    upstream.post("/chat/completions").mock(return_value=_json_response())
    proxy = make_proxy()
    compact = json.dumps(json.loads(RAW_REQUEST), separators=(",", ":")).encode()
    proxy.post(CHAT, content=RAW_REQUEST, headers=AUTH)
    proxy.post(CHAT, content=compact, headers=AUTH)

    keys = {row.request_key for row in _rows(store)}
    assert keys == {request_key(json.loads(RAW_REQUEST))}


def test_only_allowlisted_headers_reach_upstream(
    upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    route = upstream.post("/chat/completions").mock(return_value=_json_response())
    headers = {
        **AUTH,
        "HTTP-Referer": "https://app.example.com",
        "X-Title": "Example App",
        "X-Taskdistill-Task": "other",
        "Cookie": "session=abc",
        "X-Custom": "1",
    }
    make_proxy().post(CHAT, content=RAW_REQUEST, headers=headers)

    sent = route.calls.last.request.headers
    assert sent["http-referer"] == "https://app.example.com"
    assert sent["x-title"] == "Example App"
    for name in ("x-taskdistill-task", "x-taskdistill-token", "cookie", "x-custom"):
        assert name not in sent


def test_upstream_status_and_selected_headers_are_returned(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    error_body = b'{"error":{"message":"Incorrect API key provided: sk-t****0001","code":429}}'
    upstream.post("/chat/completions").mock(
        return_value=_json_response(
            error_body, 429, **{"retry-after": "3", "x-ratelimit-remaining-requests": "0", "set-cookie": "a=b"}
        )
    )
    resp = make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH)

    assert resp.status_code == 429
    assert resp.content == error_body
    assert resp.headers["retry-after"] == "3"
    assert resp.headers["x-ratelimit-remaining-requests"] == "0"
    assert "set-cookie" not in resp.headers
    [row] = _rows(store)
    assert row.status == 429
    assert row.captured is False
    assert row.response_body is None  # error bodies may echo a masked key; they are not stored
    assert row.error == "upstream HTTP 429"


def test_non_json_request_is_forwarded_and_logged_without_key(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    route = upstream.post("/chat/completions").mock(return_value=_json_response(b'{"error":"bad"}', 400))
    resp = make_proxy().post(CHAT, content=b"not json {", headers=AUTH)

    assert resp.status_code == 400
    assert route.calls.last.request.content == b"not json {"
    [row] = _rows(store)
    assert row.request_key is None
    assert row.request_body == "not json {"
    assert row.captured is False


def test_non_json_success_response_is_not_captured(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, content=b"plain text", headers={"content-type": "text/plain"})
    )
    resp = make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH)

    assert resp.content == b"plain text"
    assert resp.headers["content-type"] == "text/plain"
    [row] = _rows(store)
    assert (row.captured, row.response_body, row.error) == (False, "plain text", "response is not a JSON object")


# 2xx bodies that are not chat completions ---------------------------------------------------------

ERROR_IN_200 = "upstream error in a 2xx response"
# OpenRouter reports errors raised after generation started with HTTP 200: a top-level error object, or a choice
# carrying "error" and finish_reason "error" (with the usage of the partial generation).
TOP_LEVEL_ERROR_BODY = b'{"error":{"code":502,"message":"Provider returned error"}}'
CHOICE_ERROR_BODY = (
    b'{"id":"c1","model":"example/teacher-0731","choices":[{"index":0,"message":{"role":"assistant",'
    b'"content":"card_"},"finish_reason":"error","error":{"code":502,"message":"Provider disconnected"}}],'
    b'"usage":{"prompt_tokens":42,"completion_tokens":2,"cost":0.00005}}'
)
FINISH_ERROR_BODY = b'{"model":"m","choices":[{"index":0,"message":{"content":"x"},"finish_reason":"error"}]}'


@pytest.mark.parametrize(
    ("body", "usage"),
    [
        (TOP_LEVEL_ERROR_BODY, (None, None, None, None)),
        (CHOICE_ERROR_BODY, (42, 2, 0.00005, "example/teacher-0731")),
        (FINISH_ERROR_BODY, (None, None, None, "m")),
    ],
)
def test_upstream_error_with_status_200_is_returned_but_not_captured(
    body: bytes,
    usage: tuple[Any, ...],
    store: Store,
    upstream: respx.MockRouter,
    make_proxy: Callable[..., TestClient],
) -> None:
    upstream.post("/chat/completions").mock(return_value=_json_response(body))
    resp = make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH)

    assert (resp.status_code, resp.content) == (200, body)
    [row] = _rows(store)
    assert (row.status, row.captured, row.error) == (200, False, ERROR_IN_200)
    assert row.response_body is None  # error bodies are never stored, whatever the status
    assert row.request_key == request_key(json.loads(RAW_REQUEST))
    assert (row.prompt_tokens, row.completion_tokens, row.cost_usd, row.upstream_model) == usage
    assert store.count_captures("banking") == {"total": 1, "captured": 0, "not_captured": 1}


@pytest.mark.parametrize(
    ("raw", "body", "error"),
    [
        (RAW_REQUEST, b'{"id":"c1","model":"m","choices":[]}', '"response.choices" must be a non-empty array'),
        (RAW_REQUEST, b'{"id":"c1","model":"m"}', '"response.choices" must be a non-empty array'),
        (RAW_REQUEST, b'{"model":"m","choices":["card_arrival"]}', '"response.choices" must hold objects'),
        (b'{"model":"m","prompt":"Where is my card?"}', UPSTREAM_BODY, '"request.messages" must be a non-empty array'),
        (b'{"model":"m","messages":[]}', UPSTREAM_BODY, '"request.messages" must be a non-empty array'),
        (b"[1, 2]", UPSTREAM_BODY, '"request" must be a JSON object'),
    ],
)
def test_2xx_pair_that_is_not_a_chat_completion_is_not_captured(
    raw: bytes,
    body: bytes,
    error: str,
    store: Store,
    upstream: respx.MockRouter,
    make_proxy: Callable[..., TestClient],
) -> None:
    route = upstream.post("/chat/completions").mock(return_value=_json_response(body))
    resp = make_proxy().post(CHAT, content=raw, headers=AUTH)

    assert (resp.status_code, resp.content) == (200, body)
    assert route.calls.last.request.content == raw
    [row] = _rows(store)
    assert (row.captured, row.error, row.status) == (False, error, 200)
    assert row.response_body == body.decode()
    assert row.request_body == raw.decode()


def test_unpaired_surrogate_escapes_are_forwarded_but_not_captured(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient], tmp_path: Path
) -> None:
    # A string cut inside a surrogate pair, as JSON.stringify writes it: valid JSON that UTF-8 cannot encode.
    cut_request = b'{"model":"example/teacher","messages":[{"role":"user","content":"cut \\ud83d"}]}'
    cut_response = UPSTREAM_BODY.replace(b'"card_arrival"', b'"card \\ud83d"')
    paired_request = b'{"model":"example/teacher","messages":[{"role":"user","content":"smile \\ud83d\\ude00"}]}'
    route = upstream.post("/chat/completions").mock(
        side_effect=[_json_response(), _json_response(cut_response), _json_response()]
    )
    proxy = make_proxy()
    first = proxy.post(CHAT, content=cut_request, headers=AUTH)
    second = proxy.post(CHAT, content=RAW_REQUEST, headers=AUTH)
    third = proxy.post(CHAT, content=paired_request, headers=AUTH)

    assert (first.status_code, first.content) == (200, UPSTREAM_BODY)
    assert (second.status_code, second.content) == (200, cut_response)
    assert third.status_code == 200
    assert [c.request.content for c in route.calls] == [cut_request, RAW_REQUEST, paired_request]
    a, b, c = _rows(store)
    assert (a.request_key, a.request_body, a.captured) == (None, cut_request.decode(), False)
    assert a.error == '"request" contains an unpaired UTF-16 surrogate escape'
    assert (b.request_key, b.response_body, b.captured) == (
        request_key(json.loads(RAW_REQUEST)),
        cut_response.decode(),
        False,
    )
    assert b.error == '"response" contains an unpaired UTF-16 surrogate escape'
    assert (c.request_key, c.captured, c.error) == (request_key(json.loads(paired_request)), True, None)
    assert json.loads(paired_request)["messages"][0]["content"] == "smile \U0001f600"
    assert export_file(store, "banking", tmp_path / "e.jsonl") == 1


def test_query_string_is_forwarded(upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]) -> None:
    route = upstream.post("/chat/completions").mock(return_value=_json_response())
    make_proxy().post(CHAT + "?api-version=2024-10-21", content=RAW_REQUEST, headers=AUTH)
    assert route.calls.last.request.url.params["api-version"] == "2024-10-21"


# streaming ----------------------------------------------------------------------------------------


def test_stream_is_passed_through_and_not_captured(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    raw = b'{"model":"example/teacher", "stream": true, "messages":[{"role":"user","content":"hi"}]}'
    route = upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, content=SSE_BODY, headers={"content-type": "text/event-stream"})
    )
    proxy = make_proxy()
    with proxy.stream("POST", CHAT, content=raw, headers=AUTH) as resp:
        received = b"".join(resp.iter_bytes())
        content_type = resp.headers["content-type"]

    assert received == SSE_BODY
    assert content_type.startswith("text/event-stream")
    assert route.calls.last.request.content == raw
    [row] = _rows(store)
    assert row.captured is False
    assert row.response_body is None
    assert row.request_key == request_key(json.loads(raw))
    assert row.request_body == raw.decode()
    assert row.status == 200
    assert row.error == STREAM_NOT_CAPTURED
    assert store.count_captures("banking") == {"total": 1, "captured": 0, "not_captured": 1}


# task selection -----------------------------------------------------------------------------------


def test_task_header_selects_task(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    upstream.post("/chat/completions").mock(return_value=_json_response())
    resp = make_proxy().post(
        "/v1/chat/completions", content=RAW_REQUEST, headers={**AUTH, "X-Taskdistill-Task": "alpha"}
    )
    assert resp.status_code == 200
    assert len(_rows(store, "alpha")) == 1


def test_path_segment_wins_over_header(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    upstream.post("/chat/completions").mock(return_value=_json_response())
    make_proxy().post(
        "/t/beta/v1/chat/completions", content=RAW_REQUEST, headers={**AUTH, "X-Taskdistill-Task": "alpha"}
    )
    assert len(_rows(store, "beta")) == 1
    assert _rows(store, "alpha") == []


def test_default_task_applies_without_path_or_header(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    upstream.post("/chat/completions").mock(return_value=_json_response())
    proxy = make_proxy(default_task="gamma")
    proxy.post("/v1/chat/completions", content=RAW_REQUEST, headers=AUTH)
    proxy.post("/v1/chat/completions", content=RAW_REQUEST, headers={**AUTH, "X-Taskdistill-Task": "alpha"})
    assert len(_rows(store, "gamma")) == 1
    assert len(_rows(store, "alpha")) == 1


def test_no_task_is_a_400_and_nothing_is_forwarded(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    route = upstream.post("/chat/completions").mock(return_value=_json_response())
    resp = make_proxy().post("/v1/chat/completions", content=RAW_REQUEST, headers=AUTH)

    assert resp.status_code == 400
    assert "X-Taskdistill-Task" in resp.json()["error"]["message"]
    assert route.call_count == 0
    assert store.count_captures("banking")["total"] == 0


@pytest.mark.parametrize("task", ["-leading-dash", "a" * 65, "sp ace"])
def test_invalid_task_name_is_a_400(
    task: str, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    route = upstream.post("/chat/completions").mock(return_value=_json_response())
    resp = make_proxy().post("/v1/chat/completions", content=RAW_REQUEST, headers={**AUTH, "X-Taskdistill-Task": task})
    assert resp.status_code == 400
    assert route.call_count == 0


def test_invalid_default_task_and_empty_token_are_rejected(store: Store) -> None:
    with pytest.raises(ValueError, match="default task"):
        create_proxy_app(store, UPSTREAM, default_task="../x")
    with pytest.raises(ValueError, match="token"):
        create_proxy_app(store, UPSTREAM, token="")


# models -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/t/banking/v1/models", "/v1/models"])
def test_models_passed_through_and_not_stored(
    path: str, store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    route = upstream.get("/models").mock(return_value=_json_response(MODELS_BODY))
    resp = make_proxy().get(path, headers={"Authorization": f"Bearer {CLIENT_KEY}"})

    assert resp.status_code == 200
    assert resp.content == MODELS_BODY
    assert route.calls.last.request.headers["authorization"] == f"Bearer {CLIENT_KEY}"
    assert store.count_captures("banking")["total"] == 0


def test_models_upstream_down_is_502(upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]) -> None:
    upstream.get("/models").mock(side_effect=httpx.ConnectError("connection refused"))
    resp = make_proxy().get("/v1/models")
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_error"


# token --------------------------------------------------------------------------------------------


def test_token_required_when_configured(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    route = upstream.post("/chat/completions").mock(return_value=_json_response())
    models = upstream.get("/models").mock(return_value=_json_response(MODELS_BODY))
    proxy = make_proxy(token=PROXY_TOKEN)

    missing = proxy.post(CHAT, content=RAW_REQUEST, headers=AUTH)
    wrong = proxy.post(CHAT, content=RAW_REQUEST, headers={**AUTH, "X-Taskdistill-Token": "nope"})
    models_missing = proxy.get("/v1/models", headers=AUTH)
    assert (missing.status_code, wrong.status_code, models_missing.status_code) == (401, 401, 401)
    assert missing.json()["error"]["type"] == "authentication_error"
    assert route.call_count == 0
    assert models.call_count == 0
    assert store.count_captures("banking")["total"] == 0

    ok = proxy.post(CHAT, content=RAW_REQUEST, headers={**AUTH, "X-Taskdistill-Token": PROXY_TOKEN})
    assert ok.status_code == 200
    assert ok.content == UPSTREAM_BODY
    assert proxy.get("/v1/models", headers={"X-Taskdistill-Token": PROXY_TOKEN}).status_code == 200


def test_token_is_stripped_and_never_stored(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    route = upstream.post("/chat/completions").mock(return_value=_json_response())
    proxy = make_proxy(token=PROXY_TOKEN)
    proxy.post(CHAT, content=RAW_REQUEST, headers={**AUTH, "X-Taskdistill-Token": PROXY_TOKEN})

    sent = route.calls.last.request
    assert "x-taskdistill-token" not in sent.headers
    assert PROXY_TOKEN.encode() not in sent.content
    assert sent.headers["authorization"] == f"Bearer {CLIENT_KEY}"
    for value in _all_values(store.path):
        assert PROXY_TOKEN not in str(value)
        assert CLIENT_KEY not in str(value)
    blob = _db_bytes(store.path)
    assert PROXY_TOKEN.encode() not in blob
    assert CLIENT_KEY.encode() not in blob


# the store never holds headers --------------------------------------------------------------------

HEADER_LIKE = re.compile(r"header|authori[sz]ation|auth|cookie|api_?key|secret|bearer|referer|(^|_)token$")


def test_store_has_no_header_columns(store: Store) -> None:
    conn = sqlite3.connect(store.path)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    finally:
        conn.close()
    tables = [t for t in tables if not t.startswith("sqlite_")]
    assert set(tables) >= {"captures", "imports", "served"}
    for table in tables:
        columns = store.columns(table)
        assert columns
        assert [c for c in columns if HEADER_LIKE.search(c.lower())] == [], table
    assert HEADER_LIKE.search("prompt_tokens") is None  # token counts are not credentials
    assert HEADER_LIKE.search("x_taskdistill_token") is not None


def test_no_header_value_reaches_the_store(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    upstream.post("/chat/completions").mock(return_value=_json_response())
    upstream.get("/models").mock(return_value=_json_response(MODELS_BODY))
    headers = {**AUTH, "HTTP-Referer": "https://referer.example.com", "X-Title": "Title-Marker-5731"}
    proxy = make_proxy()
    proxy.post(CHAT, content=RAW_REQUEST, headers=headers)
    proxy.get("/t/banking/v1/models", headers=headers)

    assert store.count_captures("banking")["total"] == 1
    blob = _db_bytes(store.path)
    for secret in (CLIENT_KEY, "referer.example.com", "Title-Marker-5731", "Bearer"):
        assert secret.encode() not in blob


# failures -----------------------------------------------------------------------------------------


def test_logging_failure_still_returns_upstream_response(
    store: Store,
    upstream: respx.MockRouter,
    make_proxy: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(**_: Any) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "add_capture", broken)
    upstream.post("/chat/completions").mock(return_value=_json_response())
    with caplog.at_level(logging.WARNING, logger="taskdistill.capture.proxy"):
        resp = make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH)

    assert resp.status_code == 200
    assert resp.content == UPSTREAM_BODY
    assert any(
        "capture logging failed" in r.getMessage() and "OperationalError" in r.getMessage() for r in caplog.records
    )


def test_logging_failure_in_record_building_or_stream_is_contained(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_key(_: Any) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(proxy_mod, "request_key", broken_key)
    upstream.post("/chat/completions").mock(
        side_effect=[
            _json_response(),
            httpx.Response(200, content=SSE_BODY, headers={"content-type": "text/event-stream"}),
        ]
    )
    proxy = make_proxy()
    plain = proxy.post(CHAT, content=RAW_REQUEST, headers=AUTH)
    streamed = proxy.post(CHAT, content=b'{"stream":true,"messages":[]}', headers=AUTH)

    assert (plain.status_code, plain.content) == (200, UPSTREAM_BODY)
    assert (streamed.status_code, streamed.content) == (200, SSE_BODY)
    assert _rows(store) == []


def test_upstream_down_is_502_and_logged(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    upstream.post("/chat/completions").mock(side_effect=httpx.ConnectError("connection refused"))
    resp = make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH)

    assert resp.status_code == 502
    assert resp.json()["error"] == {"message": "upstream unreachable (ConnectError)", "type": "upstream_error"}
    [row] = _rows(store)
    assert (row.status, row.captured, row.response_body) == (502, False, None)
    assert row.error == "upstream unreachable (ConnectError)"
    assert row.request_key == request_key(json.loads(RAW_REQUEST))


def test_upstream_timeout_is_504(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient]
) -> None:
    upstream.post("/chat/completions").mock(side_effect=httpx.ReadTimeout("timed out"))
    resp = make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH)
    assert resp.status_code == 504
    [row] = _rows(store)
    assert row.status == 504
    assert row.captured is False


def test_upstream_down_with_broken_logging_is_still_502(
    store: Store, upstream: respx.MockRouter, make_proxy: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(**_: Any) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(store, "add_capture", broken)
    upstream.post("/chat/completions").mock(side_effect=httpx.ConnectError("connection refused"))
    assert make_proxy().post(CHAT, content=RAW_REQUEST, headers=AUTH).status_code == 502


# client lifecycle ---------------------------------------------------------------------------------


def test_client_created_in_lifespan_when_not_injected(store: Store, upstream: respx.MockRouter) -> None:
    upstream.post("/chat/completions").mock(return_value=_json_response())
    app = create_proxy_app(store, UPSTREAM + "/")
    with TestClient(app) as proxy:
        assert proxy.post(CHAT, content=RAW_REQUEST, headers=AUTH).content == UPSTREAM_BODY
    # Without the lifespan (no context manager) a client is created on first use.
    assert TestClient(app).post(CHAT, content=RAW_REQUEST, headers=AUTH).status_code == 200
    assert store.count_captures("banking")["captured"] == 2


# run_proxy and bind safety ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("LOCALHOST", True),
        ("::1", True),
        ("[::1]", True),
        (" [::1] ", True),
        ("0.0.0.0", False),
        ("[::]", False),
        ("", False),
        ("::", False),
        ("192.0.2.10", False),
        ("localhost.example.com", False),
    ],
)
def test_is_local_host(host: str, local: bool) -> None:
    assert is_local_host(host) is local


@pytest.fixture
def captured_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run(app: FastAPI, **kwargs: Any) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(proxy_mod.uvicorn, "run", fake_run)
    return calls


@pytest.mark.parametrize("host", ["0.0.0.0", "", "::", "192.0.2.10"])
@pytest.mark.parametrize("env_value", [None, ""])
def test_run_proxy_refuses_public_bind_without_token(
    host: str,
    env_value: str | None,
    store: Store,
    captured_run: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if env_value is None:
        monkeypatch.delenv(TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(TOKEN_ENV, env_value)
    with pytest.raises(UnsafeBindError, match=TOKEN_ENV):
        run_proxy(host, 0, store=store, upstream_base_url=UPSTREAM)
    assert captured_run == []


def test_run_proxy_on_localhost_needs_no_token(
    store: Store, captured_run: list[dict[str, Any]], upstream: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    upstream.post("/chat/completions").mock(return_value=_json_response())
    run_proxy("127.0.0.1", 0, store=store, upstream_base_url=UPSTREAM, default_task="banking")

    [call] = captured_run
    assert (call["host"], call["port"]) == ("127.0.0.1", 0)
    with TestClient(call["app"]) as proxy:
        assert proxy.post("/v1/chat/completions", content=RAW_REQUEST, headers=AUTH).status_code == 200
    assert store.count_captures("banking")["captured"] == 1


def test_run_proxy_public_bind_enforces_token(
    store: Store, captured_run: list[dict[str, Any]], upstream: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, PROXY_TOKEN)
    route = upstream.post("/chat/completions").mock(return_value=_json_response())
    run_proxy("0.0.0.0", 0, store=store, upstream_base_url=UPSTREAM)

    [call] = captured_run
    assert call["host"] == "0.0.0.0"
    with TestClient(call["app"]) as proxy:
        assert proxy.post(CHAT, content=RAW_REQUEST, headers=AUTH).status_code == 401
        ok = proxy.post(CHAT, content=RAW_REQUEST, headers={**AUTH, "X-Taskdistill-Token": PROXY_TOKEN})
    assert ok.status_code == 200
    assert "x-taskdistill-token" not in route.calls.last.request.headers


@pytest.mark.parametrize(
    ("host", "bound"), [("[::1]", "::1"), (" [::1] ", "::1"), ("::1", "::1"), (" 127.0.0.1 ", "127.0.0.1")]
)
def test_run_proxy_binds_the_bare_host(
    host: str, bound: str, store: Store, captured_run: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    run_proxy(host, 0, store=store, upstream_base_url=UPSTREAM)
    [call] = captured_run
    assert (call["host"], call["port"]) == (bound, 0)


def test_run_proxy_bracketed_wildcard_needs_the_token(
    store: Store, captured_run: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(UnsafeBindError, match=TOKEN_ENV):
        run_proxy("[::]", 0, store=store, upstream_base_url=UPSTREAM)
    assert captured_run == []
    monkeypatch.setenv(TOKEN_ENV, PROXY_TOKEN)
    run_proxy("[::]", 0, store=store, upstream_base_url=UPSTREAM)
    [call] = captured_run
    assert call["host"] == "::"


# over real sockets (ephemeral ports) --------------------------------------------------------------


class _Served:
    """Run an ASGI app with uvicorn on 127.0.0.1 and an ephemeral port in a background thread."""

    def __init__(self, app: FastAPI) -> None:
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.port = 0

    def __enter__(self) -> _Served:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("server did not start")
            time.sleep(0.01)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        return self

    def __exit__(self, *_: object) -> None:
        self.server.should_exit = True
        self.thread.join(10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _echo_upstream(seen: list[tuple[bytes, dict[str, str]]]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.body()
        seen.append((body, dict(request.headers)))
        if b'"stream":true' in body.replace(b" ", b""):

            async def events() -> Any:
                for part in SSE_BODY.split(b"\n\n")[:-1]:
                    yield part + b"\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        return Response(UPSTREAM_BODY, media_type="application/json")

    return app


def test_real_sockets_forward_bytes_and_stream(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    seen: list[tuple[bytes, dict[str, str]]] = []
    stream_raw = b'{"model":"example/teacher","stream":true,"messages":[{"role":"user","content":"hi"}]}'
    with (
        _Served(_echo_upstream(seen)) as up,
        _Served(create_proxy_app(store, up.url + "/v1", token=PROXY_TOKEN)) as proxy,
        httpx.Client(base_url=proxy.url, trust_env=False, timeout=10) as client,
    ):
        headers = {**AUTH, "X-Taskdistill-Token": PROXY_TOKEN}
        plain = client.post(CHAT, content=RAW_REQUEST, headers=headers)
        streamed = client.post(CHAT, content=stream_raw, headers=headers)
        denied = client.post(CHAT, content=RAW_REQUEST, headers=AUTH)

    assert plain.status_code == 200
    assert plain.content == UPSTREAM_BODY
    assert streamed.content == SSE_BODY
    assert denied.status_code == 401
    assert [body for body, _ in seen] == [RAW_REQUEST, stream_raw]
    for _, sent in seen:
        assert sent["authorization"] == f"Bearer {CLIENT_KEY}"
        assert "x-taskdistill-token" not in sent
    rows = _rows(store)
    assert [(r.captured, r.status) for r in rows] == [(True, 200), (False, 200)]
    assert PROXY_TOKEN.encode() not in _db_bytes(store.path)
