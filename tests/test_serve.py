from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import math
import random
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from taskdistill.backends.fake import FakeBackend
from taskdistill.backends.types import Generation
from taskdistill.config import TaskSpec
from taskdistill.ledger import Ledger
from taskdistill.serve.app import ModelLoadError, create_app, render, unsupported_feature
from taskdistill.serve.worker import ModelNotReady, ModelWorker
from taskdistill.store import Store
from taskdistill.teacher.base import (
    BudgetExceeded,
    ReplayMiss,
    TeacherHTTPError,
    TeacherResult,
    TeacherTimeout,
)
from taskdistill.teacher.client import LiveTeacher
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.replay import ReplayTeacher, expected_manifest, write_recording
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

TEACHER = "vendor/teacher-model"
TEACHER_BASE = "https://teacher.example.com/api/v1"
TEACHER_KEY = "sk-test-teacher-key"
CLIENT_KEY = "sk-test-client-key"
LABELS = ["card_arrival", "lost_card", "cash_withdrawal"]
SURE = "My card still has not arrived after two weeks."
UNSURE = "I think I lost it somewhere yesterday."
ANSWERS = {SURE: ("card_arrival", 0.95), UNSURE: ("lost_card", 0.40)}
SCHEMA = {
    "type": "object",
    "properties": {"vendor": {"type": ["string", "null"]}, "total": {"type": ["number", "null"]}},
    "required": ["vendor", "total"],
    "additionalProperties": False,
}


def make_spec(kind: Literal["classification", "extraction"] = "classification", **sections: Any) -> TaskSpec:
    raw: dict[str, Any] = {
        "task": "support-intents" if kind == "classification" else "receipts",
        "type": kind,
        "teacher": {"model": TEACHER, "base_url": TEACHER_BASE, "max_tokens": 24},
        "student": {"system_prompt": "Classify the message."},
        "cascade": {"target": 0.97},
    }
    if kind == "classification":
        raw["labels_file"] = "labels.txt"
    else:
        raw["schema_file"] = "schema.json"
    for name, values in sections.items():
        raw[name] = {**raw.get(name, {}), **values}
    spec = TaskSpec.model_validate(raw)
    spec.teacher_prompt = "Label the customer's message with one intent.\n"
    if kind == "classification":
        spec.labels = list(LABELS)
    else:
        spec.json_schema = copy.deepcopy(SCHEMA)
    return spec


def chat(text: str | None = SURE, **extra: Any) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": "You label support messages."}]
    if text is not None:
        messages.append({"role": "user", "content": text})
    body: dict[str, Any] = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0, "max_tokens": 24}
    body.update(extra)
    return body


class FakeTeacher:
    """A teacher source that records every body it receives."""

    def __init__(
        self,
        output: str | None = "card_arrival",
        *,
        mode: Literal["live", "replay"] = "live",
        error: BaseException | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.mode: Literal["live", "replay"] = mode
        self.output = output
        self.error = error
        self.chunks = chunks or []
        self.bodies: list[dict[str, Any]] = []
        self.stream_bodies: list[dict[str, Any]] = []
        self.closed = False

    def response(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "gen-teacher-1",
            "object": "chat.completion",
            "created": 1790000000,
            "model": body["model"],
            "provider": "Alpha",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": self.output}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 90, "completion_tokens": 4, "total_tokens": 94, "cost": 3e-5},
        }

    async def complete(self, body: dict[str, Any]) -> TeacherResult:
        self.bodies.append(copy.deepcopy(body))
        if self.error is not None:
            raise self.error
        response = self.response(body)
        return TeacherResult(
            key=request_key(body),
            output=self.output,
            response=response,
            usage=response["usage"],
            latency_ms=12.5,
            provider="Alpha",
            finish_reason="stop",
            created=1790000000.0,
            source=self.mode,
            cost_usd=3e-5,
        )

    async def stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        self.stream_bodies.append(copy.deepcopy(body))
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.sqlite")


Serve = Callable[..., AbstractContextManager[TestClient]]


@pytest.fixture
def serve(store: Store) -> Serve:
    @contextmanager
    def _serve(
        spec: TaskSpec,
        teacher: Any,
        *,
        backend: FakeBackend | None = None,
        threshold: float = 0.8,
        token: str | None = None,
        run_id: str | None = "run-1",
        app_store: Any = None,
    ) -> Iterator[TestClient]:
        model = backend if backend is not None else FakeBackend(answers=ANSWERS)
        worker = ModelWorker(lambda: model, spec=spec)
        app = create_app(
            spec,
            worker=worker,
            teacher=teacher,
            threshold=threshold,
            store=app_store if app_store is not None else store,
            token=token,
            run_id=run_id,
        )
        with TestClient(app) as client:
            yield client

    return _serve


def sse_events(text: str) -> list[Any]:
    blocks = [block for block in text.split("\n\n") if block]
    return [block.removeprefix("data: ") for block in blocks]


# routes and headers ------------------------------------------------------------------------------


def test_confident_student_answers(serve: Serve) -> None:
    teacher = FakeTeacher()
    with serve(make_spec(), teacher) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE))
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "student"
    assert resp.headers["x-taskdistill-confidence"] == "0.950000"
    assert "x-taskdistill-reason" not in resp.headers
    assert "x-taskdistill-teacher" not in resp.headers
    data = resp.json()
    assert data["id"].startswith("chatcmpl-td-") and len(data["id"]) == len("chatcmpl-td-") + 32
    assert data["object"] == "chat.completion"
    assert data["model"] == "gpt-4o-mini"
    assert isinstance(data["created"], int)
    assert data["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "card_arrival"}, "finish_reason": "stop"}
    ]
    usage = data["usage"]
    assert usage["completion_tokens"] == 1
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert teacher.bodies == []


def test_low_confidence_escalates_with_only_the_model_replaced(serve: Serve) -> None:
    teacher = FakeTeacher(" Lost Card.")
    body = chat(
        UNSURE,
        top_p=0.9,
        seed=7,
        stop=["\n"],
        user="customer-42",
        metadata={"ticket": "T-1", "nested": {"a": [1, 2.5, None, True]}},
        provider={"order": ["alpha"], "allow_fallbacks": False},
    )
    with serve(make_spec(), teacher) as client:
        resp = client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {CLIENT_KEY}"})
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-reason"] == "low_confidence"
    assert resp.headers["x-taskdistill-teacher"] == "live"
    assert resp.headers["x-taskdistill-confidence"] == "0.400000"
    assert resp.json()["choices"][0]["message"]["content"] == "lost_card"
    assert resp.json()["model"] == "gpt-4o-mini"
    [sent] = teacher.bodies
    expected = {**body, "model": TEACHER}
    assert sent == expected
    assert list(sent) == list(expected)
    assert json.dumps(sent) == json.dumps(expected)


@pytest.mark.parametrize("model", ["gpt-4o-mini", TEACHER, "taskdistill/support-intents", "anything at all", None])
def test_any_model_is_accepted(serve: Serve, model: str | None) -> None:
    body = chat(SURE)
    if model is None:
        del body["model"]
    else:
        body["model"] = model
    with serve(make_spec(), FakeTeacher()) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "student"
    assert resp.json()["model"] == (model or "taskdistill/support-intents")


def test_escalation_with_the_teacher_slug_as_model(serve: Serve) -> None:
    teacher = FakeTeacher("lost_card")
    with serve(make_spec(), teacher) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE, model=TEACHER))
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert teacher.bodies[0]["model"] == TEACHER
    assert resp.json()["model"] == TEACHER


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]},
        {"tool_choice": "auto"},
        {"functions": [{"name": "lookup", "parameters": {"type": "object"}}]},
        {"function_call": "auto"},
        {"n": 2},
        {"logprobs": True},
        {"logprobs": True, "top_logprobs": 3},
    ],
)
def test_unsupported_requests_go_to_the_teacher(serve: Serve, extra: dict[str, Any]) -> None:
    backend = FakeBackend(answers=ANSWERS)
    teacher = FakeTeacher("card_arrival")
    body = chat(SURE, **extra)
    with serve(make_spec(), teacher, backend=backend) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-reason"] == "unsupported"
    assert "x-taskdistill-confidence" not in resp.headers
    assert backend.calls == []
    assert teacher.bodies == [{**body, "model": TEACHER}]


class VerbatimTeacher(FakeTeacher):
    """Answers every request with one fixed ``chat.completion`` (e.g. several choices, logprobs, tool calls)."""

    def __init__(self, response: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(response["choices"][0]["message"].get("content"), **kwargs)
        self.fixed = response

    def response(self, body: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(self.fixed)


def completion(*choices: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-teacher-2",
        "object": "chat.completion",
        "created": 1790000000,
        "model": TEACHER,
        "choices": [{"index": i, **choice} for i, choice in enumerate(choices)],
        "usage": {"prompt_tokens": 90, "completion_tokens": 9, "total_tokens": 99, "cost": 3e-5},
    }


TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "lookup", "arguments": '{"ticket": "T-1"}'},
}
VERBATIM_CASES = [
    (
        {"n": 2},
        completion(
            {"message": {"role": "assistant", "content": "Card Arrival."}, "finish_reason": "stop"},
            {"message": {"role": "assistant", "content": "card arrival"}, "finish_reason": "stop"},
        ),
    ),
    (
        {"logprobs": True, "top_logprobs": 2},
        completion(
            {
                "message": {"role": "assistant", "content": "Card Arrival."},
                "logprobs": {"content": [{"token": "Card", "logprob": -0.1, "bytes": None, "top_logprobs": []}]},
                "finish_reason": "stop",
            }
        ),
    ),
    (
        {"tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]},
        completion(
            {
                "message": {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
                "finish_reason": "tool_calls",
            }
        ),
    ),
]


@pytest.mark.parametrize(("extra", "response"), VERBATIM_CASES)
def test_unsupported_escalations_are_returned_verbatim_in_canonical_mode(
    serve: Serve, extra: dict[str, Any], response: dict[str, Any]
) -> None:
    spec = make_spec()
    assert spec.cascade.escalation_response == "canonical"
    teacher = VerbatimTeacher(response)
    with serve(spec, teacher) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE, **extra))
        metrics = client.get("/metrics").text
    assert resp.headers["x-taskdistill-reason"] == "unsupported"
    assert resp.json() == response
    assert "taskdistill_escalation_unnormalised_total 0.0" in metrics


def test_an_unsupported_stream_passes_the_teacher_stream_through(serve: Serve) -> None:
    call = json.dumps({"index": 0, **TOOL_CALL}, separators=(",", ":")).encode()
    chunks = [
        b'data: {"id":"gen-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant",'
        b'"content":null,"tool_calls":[' + call + b']},"finish_reason":null}]}\n\n',
        b'data: {"id":"gen-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},'
        b'"finish_reason":"tool_calls"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    teacher = FakeTeacher(None, chunks=chunks)
    body = chat(SURE, stream=True, tools=[{"type": "function", "function": {"name": "lookup"}}])
    with serve(make_spec(), teacher) as client:
        resp = client.post("/v1/chat/completions", json=body)
        metrics = client.get("/metrics").text
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-reason"] == "unsupported"
    assert resp.content == b"".join(chunks)
    assert teacher.stream_bodies == [{**body, "model": TEACHER}]
    assert teacher.bodies == []
    assert "taskdistill_escalation_unnormalised_total 0.0" in metrics


@pytest.mark.parametrize("extra", [{"n": 1}, {"logprobs": False}, {"tools": []}, {"tools": None}])
def test_supported_variants_stay_with_the_student(serve: Serve, extra: dict[str, Any]) -> None:
    assert unsupported_feature(chat(SURE, **extra)) is None
    with serve(make_spec(), FakeTeacher()) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE, **extra))
    assert resp.headers["x-taskdistill-route"] == "student"


def test_unparsed_input_goes_to_the_teacher(serve: Serve) -> None:
    backend = FakeBackend(answers=ANSWERS)
    teacher = FakeTeacher("card_arrival")
    with serve(make_spec(), teacher, backend=backend) as client:
        resp = client.post("/v1/chat/completions", json=chat(None))
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-reason"] == "input_unparsed"
    assert "x-taskdistill-confidence" not in resp.headers
    assert backend.calls == []


def test_regex_input_is_extracted_or_escalated(serve: Serve) -> None:
    spec = make_spec(input={"from": "regex", "regex": r"Message:\s*(?P<input>.+)"})
    backend = FakeBackend(answers=ANSWERS)
    teacher = FakeTeacher("card_arrival")
    with serve(spec, teacher, backend=backend) as client:
        matched = client.post("/v1/chat/completions", json=chat(f"Ticket 17\nMessage: {SURE}"))
        unmatched = client.post("/v1/chat/completions", json=chat(SURE))
    assert matched.headers["x-taskdistill-route"] == "student"
    assert backend.calls == [SURE]
    assert unmatched.headers["x-taskdistill-reason"] == "input_unparsed"


def test_unparsed_input_gets_the_teacher_response_verbatim(serve: Serve) -> None:
    spec = make_spec(input={"from": "regex", "regex": r"Message:\s*(?P<input>.+)"})
    teacher = FakeTeacher("Lost card.")
    chunks = [b'data: {"choices":[{"index":0,"delta":{"content":"Lost card."}}]}\n\n', b"data: [DONE]\n\n"]
    streamer = FakeTeacher(chunks=chunks)
    with serve(spec, teacher) as client:
        resp = client.post("/v1/chat/completions", json=chat("Where is my card?"))
        metrics = client.get("/metrics").text
    with serve(spec, streamer) as client:
        streamed = client.post("/v1/chat/completions", json=chat("Where is my card?", stream=True))
    assert resp.headers["x-taskdistill-reason"] == "input_unparsed"
    assert resp.json() == teacher.response({"model": TEACHER})
    assert resp.json()["choices"][0]["message"]["content"] == "Lost card."
    assert "taskdistill_escalation_unnormalised_total 0.0" in metrics
    assert streamed.headers["x-taskdistill-reason"] == "input_unparsed"
    assert streamed.content == b"".join(chunks)


def test_live_teacher_gets_the_teacher_key_never_the_clients(serve: Serve, tmp_path: Path) -> None:
    pricing = PricingSnapshot(
        date="2026-09-26",
        source="test",
        models={TEACHER: {"default": ModelPrice(1e-6, 2e-6), "providers": {}}},
    )
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=5.0)
    teacher = LiveTeacher(
        TEACHER_BASE, TEACHER_KEY, ledger=ledger, pricing=pricing, task="t", phase="serve", run_id="r"
    )
    body = chat(UNSURE)
    upstream = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1790000000,
        "model": TEACHER,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "lost_card"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 3, "cost": 4e-5},
    }
    with respx.mock(base_url=TEACHER_BASE, assert_all_called=True) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=upstream))
        with serve(make_spec(), teacher, token=CLIENT_KEY) as client:
            resp = client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {CLIENT_KEY}"})
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-teacher"] == "live"
    [call] = route.calls
    sent = call.request
    assert sent.headers["authorization"] == f"Bearer {TEACHER_KEY}"
    assert all(CLIENT_KEY not in value for value in sent.headers.values())
    assert json.loads(sent.content) == {**body, "model": TEACHER}
    assert ledger.spent(phase="serve") == pytest.approx(4e-5)


# canonical and raw escalations -------------------------------------------------------------------


def test_canonical_escalation_normalises_the_label(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher("Card Arrival.")) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE))
    data = resp.json()
    assert data["id"].startswith("chatcmpl-td-")
    assert data["model"] == "gpt-4o-mini"
    assert data["choices"][0]["message"]["content"] == "card_arrival"
    assert data["usage"] == {"prompt_tokens": 90, "completion_tokens": 4, "total_tokens": 94}


def test_canonical_escalation_normalises_extraction_json(serve: Serve) -> None:
    spec = make_spec("extraction", student={"system_prompt": "Extract the receipt fields.", "max_tokens": 64})
    backend = FakeBackend(answers={UNSURE: ('{"vendor": "Example Ltd", "total": 12.5}', 0.3)})
    teacher = FakeTeacher('Here you go:\n```json\n{"total": 12.50, "vendor": "Example Ltd"}\n```')
    with serve(spec, teacher, backend=backend) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE))
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.json()["choices"][0]["message"]["content"] == '{"vendor":"Example Ltd","total":12.5}'


def test_canonical_escalation_returns_unnormalisable_output_raw(serve: Serve) -> None:
    teacher = FakeTeacher("I am not sure which intent this is.")
    with serve(make_spec(), teacher) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE))
        metrics = client.get("/metrics").text
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.json() == teacher.response({"model": TEACHER})
    assert "taskdistill_escalation_unnormalised_total 1.0" in metrics


def test_raw_escalation_returns_the_teacher_response_verbatim(serve: Serve) -> None:
    spec = make_spec(cascade={"escalation_response": "raw"})
    teacher = FakeTeacher("Card Arrival.")
    with serve(spec, teacher) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE))
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-teacher"] == "live"
    assert resp.json() == teacher.response({"model": TEACHER})


def test_student_stream_is_one_chunk_then_done(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher()) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE, stream=True))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-taskdistill-route"] == "student"
    assert resp.text.endswith("data: [DONE]\n\n")
    first, done = sse_events(resp.text)
    assert done == "[DONE]"
    chunk = json.loads(first)
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["id"].startswith("chatcmpl-td-")
    assert chunk["model"] == "gpt-4o-mini"
    assert chunk["choices"] == [
        {"index": 0, "delta": {"role": "assistant", "content": "card_arrival"}, "finish_reason": "stop"}
    ]
    assert "usage" not in chunk


def test_stream_usage_is_included_when_asked(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher()) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE, stream=True, stream_options={"include_usage": True}))
    first, _ = sse_events(resp.text)
    assert json.loads(first)["usage"]["completion_tokens"] == 1


def test_canonical_escalation_streams_one_chunk(serve: Serve) -> None:
    teacher = FakeTeacher("Lost card")
    body = chat(UNSURE, stream=True)
    with serve(make_spec(), teacher) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["content-type"].startswith("text/event-stream")
    first, done = sse_events(resp.text)
    assert done == "[DONE]"
    assert json.loads(first)["choices"][0]["delta"] == {"role": "assistant", "content": "lost_card"}
    assert teacher.bodies == [{**body, "model": TEACHER}]
    assert teacher.stream_bodies == []


def test_canonical_stream_of_unnormalisable_output_is_raw_text(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher("no idea")) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE, stream=True))
    first, _ = sse_events(resp.text)
    assert json.loads(first)["choices"][0]["delta"]["content"] == "no idea"


def test_raw_escalation_passes_the_teacher_stream_through(serve: Serve, store: Store) -> None:
    spec = make_spec(cascade={"escalation_response": "raw"})
    chunks = [
        b'data: {"id":"gen-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant",'
        b'"content":"Lost"},"finish_reason":null}]}\n\n',
        b'data: {"id":"gen-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" card"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: {"id":"gen-1","object":"chat.completion.chunk","choices":[],'
        b'"usage":{"prompt_tokens":88,"completion_tokens":2,"cost":2e-5}}\n\n',
        b"data: [DONE]\n\n",
    ]
    teacher = FakeTeacher(chunks=chunks)
    body = chat(UNSURE, stream=True)
    with serve(spec, teacher) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-reason"] == "low_confidence"
    assert resp.headers["x-taskdistill-teacher"] == "live"
    assert resp.content == b"".join(chunks)
    assert teacher.stream_bodies == [{**body, "model": TEACHER}]
    assert teacher.bodies == []
    [row] = store.iter_served("support-intents")
    assert (row.route, row.prompt_tokens, row.completion_tokens) == ("teacher", 88, 2)
    assert row.teacher_cost_usd == pytest.approx(2e-5)


def test_raw_stream_error_before_the_first_byte_falls_back(serve: Serve) -> None:
    spec = make_spec(cascade={"escalation_response": "raw"})
    with serve(spec, FakeTeacher(error=TeacherTimeout("slow"))) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE, stream=True))
    assert resp.headers["x-taskdistill-route"] == "student-fallback"
    first, done = sse_events(resp.text)
    assert (json.loads(first)["choices"][0]["delta"]["content"], done) == ("lost_card", "[DONE]")


class BreakingTeacher(FakeTeacher):
    """Streams ``chunks``, then fails the way a live stream that broke off does."""

    async def stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        self.stream_bodies.append(copy.deepcopy(body))
        for chunk in self.chunks:
            yield chunk
        raise TeacherTimeout("teacher stream broke off (ReadError)")


@pytest.mark.parametrize(
    ("chunks", "separator"),
    [
        ([b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Lost"}}]}\n\n'], b""),
        (
            [
                b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Lost"}}]}\n\n',
                b'data: {"choices":[{"index":0,"delta":{"content":" ca',
            ],
            b"\n\n",
        ),
    ],
)
def test_a_stream_that_breaks_off_ends_with_an_error_event(
    serve: Serve, store: Store, chunks: list[bytes], separator: bytes
) -> None:
    spec = make_spec(cascade={"escalation_response": "raw"})
    with serve(spec, BreakingTeacher(chunks=chunks)) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE, stream=True))
        metrics = client.get("/metrics").text
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "teacher"
    relayed = b"".join(chunks) + separator
    assert resp.content.startswith(relayed)
    tail = resp.content[len(relayed) :]
    assert tail.startswith(b"data: ") and tail.endswith(b"\n\n")
    error = json.loads(tail[len(b"data: ") :])["error"]
    assert (error["type"], error["code"]) == ("teacher_error", 502)
    assert "broke off" in error["message"]
    assert b"[DONE]" not in resp.content
    [row] = store.iter_served(spec.task)
    assert (row.route, row.reason, row.status) == ("teacher", "low_confidence", 502)
    assert 'taskdistill_teacher_errors_total{kind="timeout"} 1.0' in metrics
    assert "taskdistill_teacher_latency_seconds_count 0.0" in metrics


class ClosingTeacher(FakeTeacher):
    """Streams forever until closed; records that its stream was closed (a live stream settles its charge then)."""

    def __init__(self) -> None:
        super().__init__()
        self.stream_closed = False

    async def stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        try:
            while True:
                yield b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'
                await asyncio.sleep(0)
        finally:
            self.stream_closed = True


def test_a_client_leaving_mid_stream_is_still_logged_and_the_teacher_stream_closed(store: Store) -> None:
    spec = make_spec(cascade={"escalation_response": "raw"})
    teacher = ClosingTeacher()
    worker = ModelWorker(lambda: FakeBackend(answers=ANSWERS), spec=spec)
    app = create_app(spec, worker=worker, teacher=teacher, threshold=0.8, store=store)
    payload = json.dumps(chat(UNSURE, stream=True)).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if not sent:
            return {"type": "http.request", "body": payload, "more_body": False}
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        if message["type"] == "http.response.body" and any(m["type"] == "http.response.body" for m in sent):
            raise OSError("the client went away")
        sent.append(message)

    async def call() -> None:
        with contextlib.suppress(Exception):
            await app(scope, receive, send)

    try:
        asyncio.run(call())
    finally:
        worker.stop()
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert dict(start["headers"])[b"x-taskdistill-route"] == b"teacher"
    assert teacher.stream_closed
    [row] = store.iter_served(spec.task)
    assert (row.route, row.reason, row.teacher_mode) == ("teacher", "low_confidence", "live")


# response template ---------------------------------------------------------------------------------


def test_response_template_uses_literal_replacement(serve: Serve) -> None:
    spec = make_spec(student={"system_prompt": "Classify.", "response_template": '{"intent": "{label}"}'})
    with serve(spec, FakeTeacher("Lost card")) as client:
        student = client.post("/v1/chat/completions", json=chat(SURE))
        teacher = client.post("/v1/chat/completions", json=chat(UNSURE))
    assert student.json()["choices"][0]["message"]["content"] == '{"intent": "card_arrival"}'
    assert teacher.json()["choices"][0]["message"]["content"] == '{"intent": "lost_card"}'
    assert json.loads(student.json()["choices"][0]["message"]["content"]) == {"intent": "card_arrival"}


def test_render_never_formats_other_braces() -> None:
    spec = make_spec(student={"system_prompt": "Classify.", "response_template": '{0} {"a": {label}} {label}{x}'})
    assert render(spec, "card_arrival") == '{0} {"a": card_arrival} card_arrival{x}'


# teacher failures ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (TeacherTimeout("teacher did not answer"), "timeout"),
        (BudgetExceeded("the global cap would be crossed"), "budget"),
        (TeacherHTTPError(503, "overloaded"), "http"),
        (httpx.ConnectError("refused"), "transport"),
    ],
)
def test_teacher_failure_falls_back_to_the_student(serve: Serve, error: Exception, kind: str) -> None:
    with serve(make_spec(), FakeTeacher(error=error)) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE))
        metrics = client.get("/metrics").text
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "student-fallback"
    assert resp.headers["x-taskdistill-reason"] == "low_confidence"
    assert resp.headers["x-taskdistill-teacher"] == "live"
    assert resp.headers["x-taskdistill-teacher-error"] == kind
    assert resp.headers["x-taskdistill-confidence"] == "0.400000"
    assert resp.json()["choices"][0]["message"]["content"] == "lost_card"
    assert f'taskdistill_teacher_errors_total{{kind="{kind}"}} 1.0' in metrics


def test_budget_refusal_of_a_live_teacher_falls_back(serve: Serve, tmp_path: Path) -> None:
    pricing = PricingSnapshot(
        date="2026-09-26", source="test", models={TEACHER: {"default": ModelPrice(1e-3, 1e-3), "providers": {}}}
    )
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=5.0)
    teacher = LiveTeacher(
        TEACHER_BASE, TEACHER_KEY, ledger=ledger, pricing=pricing, task="t", phase="serve", run_id="r", run_cap=1e-4
    )
    with respx.mock(base_url=TEACHER_BASE, assert_all_called=False) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(500))
        with serve(make_spec(), teacher) as client:
            resp = client.post("/v1/chat/completions", json=chat(UNSURE))
    assert resp.headers["x-taskdistill-route"] == "student-fallback"
    assert resp.headers["x-taskdistill-teacher-error"] == "budget"
    assert not route.called
    assert ledger.spent() == 0


def test_teacher_failure_without_a_student_answer_is_a_502(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher(error=TeacherTimeout("slow"))) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE, n=2))
    assert resp.status_code == 502
    assert resp.headers["x-taskdistill-route"] == "error"
    assert resp.headers["x-taskdistill-reason"] == "unsupported"
    assert resp.json()["error"]["type"] == "teacher_error"


def test_on_teacher_error_error_returns_502(serve: Serve) -> None:
    spec = make_spec(cascade={"on_teacher_error": "error"})
    with serve(spec, FakeTeacher(error=TeacherHTTPError(500, "upstream body sk-or-v1-abc"))) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE))
    assert resp.status_code == 502
    assert resp.headers["x-taskdistill-route"] == "error"
    assert resp.headers["x-taskdistill-teacher"] == "live"
    assert resp.headers["x-taskdistill-confidence"] == "0.400000"
    assert "HTTP 500" in resp.json()["error"]["message"]
    assert "sk-or-v1" not in resp.text


def test_replay_miss_is_a_502_even_with_student_fallback(serve: Serve) -> None:
    spec = make_spec()
    assert spec.cascade.on_teacher_error == "student"
    teacher = FakeTeacher(mode="replay", error=ReplayMiss("replay miss: request key abc is not in the recording"))
    with serve(spec, teacher) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE))
    assert resp.status_code == 502
    assert resp.headers["x-taskdistill-route"] == "error"
    assert resp.headers["x-taskdistill-teacher"] == "replay"
    assert resp.headers["x-taskdistill-teacher-error"] == "replay_miss"
    error = resp.json()["error"]
    assert error["type"] == "replay_miss"
    assert "not in the teacher recording" in error["message"]
    assert request_key({**chat(UNSURE), "model": TEACHER}) in error["message"]
    assert "without --replay" in error["message"]
    assert "--live" not in error["message"] and ".." not in error["message"]


def test_replay_teacher_answers_recorded_requests_and_refuses_others(serve: Serve, tmp_path: Path) -> None:
    spec = make_spec()
    recorded = build_teacher_request(spec, UNSURE)
    record = {
        "key": request_key(recorded),
        "output": "Lost card",
        "usage": {"prompt_tokens": 80, "completion_tokens": 2, "cost": 1e-5},
        "latency_ms": 350.0,
        "provider": None,
        "finish_reason": "stop",
        "timestamp": 1790000000.0,
    }
    manifest = {**expected_manifest(spec, "2026-09-26"), "created": "2026-09-26T12:00:00+00:00"}
    teacher = ReplayTeacher(write_recording(tmp_path / "rec.jsonl.gz", manifest, [record]), spec)
    other = FakeBackend(answers={**ANSWERS, "Where is the nearest cash machine?": ("cash_withdrawal", 0.2)})
    with serve(spec, teacher, backend=other) as client:
        hit = client.post("/v1/chat/completions", json={**recorded, "model": "gpt-4o-mini"})
        miss = client.post(
            "/v1/chat/completions", json=build_teacher_request(spec, "Where is the nearest cash machine?")
        )
        health = client.get("/healthz").json()
    assert hit.status_code == 200
    assert hit.headers["x-taskdistill-teacher"] == "replay"
    assert hit.json()["choices"][0]["message"]["content"] == "lost_card"
    assert miss.status_code == 502
    assert miss.json()["error"]["type"] == "replay_miss"
    assert health["teacher"] == "replay"


def test_replayed_escalations_are_logged_at_no_cost(serve: Serve, store: Store) -> None:
    chunks = [
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"lost_card"}}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":80,"completion_tokens":2,"cost":1e-5}}\n\n',
        b"data: [DONE]\n\n",
    ]
    with serve(make_spec(), FakeTeacher("lost_card", mode="replay")) as client:
        client.post("/v1/chat/completions", json=chat(UNSURE))
    raw = make_spec(cascade={"escalation_response": "raw"})
    with serve(raw, FakeTeacher(mode="replay", chunks=chunks)) as client:
        client.post("/v1/chat/completions", json=chat(UNSURE, stream=True))
    rows = list(store.iter_served("support-intents"))
    assert [(r.route, r.teacher_mode, r.teacher_cost_usd) for r in rows] == [
        ("teacher", "replay", 0.0),
        ("teacher", "replay", 0.0),
    ]
    assert [(r.prompt_tokens, r.completion_tokens) for r in rows] == [(90, 4), (80, 2)]


# auth, logging, metrics, health --------------------------------------------------------------------


def test_token_is_required_when_configured(serve: Serve, store: Store) -> None:
    with serve(make_spec(), FakeTeacher(), token="serve-token-1") as client:
        missing = client.post("/v1/chat/completions", json=chat(SURE))
        wrong = client.post("/v1/chat/completions", json=chat(SURE), headers={"Authorization": "Bearer nope"})
        basic = client.post("/v1/chat/completions", json=chat(SURE), headers={"Authorization": "Basic serve-token-1"})
        ok = client.post("/v1/chat/completions", json=chat(SURE), headers={"Authorization": "Bearer serve-token-1"})
        others = [client.get(path) for path in ("/v1/models", "/healthz", "/metrics")]
        authed = client.get("/healthz", headers={"Authorization": "bearer serve-token-1"})
    assert [r.status_code for r in (missing, wrong, basic)] == [401, 401, 401]
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.headers["x-taskdistill-route"] == "error"
    assert missing.json()["error"]["type"] == "authentication_error"
    assert ok.status_code == 200
    assert [r.status_code for r in others] == [401, 401, 401]
    assert authed.status_code == 200
    assert [row.status for row in store.iter_served("support-intents")] == [200]


def test_no_token_needed_without_one(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher()) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE), headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 200


def test_every_request_is_logged(serve: Serve, store: Store) -> None:
    spec = make_spec()
    with serve(spec, FakeTeacher("lost_card")) as client:
        client.post("/v1/chat/completions", json=chat(SURE, model="app-model"))
        client.post("/v1/chat/completions", json=chat(UNSURE))
        client.post("/v1/chat/completions", json=chat(SURE, tools=[{"type": "function"}]))
        client.post("/v1/chat/completions", json=chat(None))
        client.post("/v1/chat/completions", content=b"{not json", headers={"content-type": "application/json"})
    rows = list(store.iter_served(spec.task))
    assert [(r.route, r.reason, r.status) for r in rows] == [
        ("student", None, 200),
        ("teacher", "low_confidence", 200),
        ("teacher", "unsupported", 200),
        ("teacher", "input_unparsed", 200),
        ("error", None, 400),
    ]
    student, low, unsupported, _, bad = rows
    assert student.request_model == "app-model"
    assert student.confidence == pytest.approx(0.95)
    assert student.student_ms is not None and student.teacher_ms is None
    assert student.teacher_mode is None and student.teacher_cost_usd is None
    assert low.confidence == pytest.approx(0.40)
    assert low.teacher_ms is not None and low.student_ms is not None
    assert (low.prompt_tokens, low.completion_tokens) == (90, 4)
    assert low.teacher_cost_usd == pytest.approx(3e-5)
    assert low.teacher_mode == "live"
    assert unsupported.confidence is None and unsupported.student_ms is None
    assert bad.request_model is None
    assert all(r.total_ms is not None and r.total_ms >= 0 for r in rows)


class BrokenStore:
    def add_served(self, **row: Any) -> int:
        raise OSError("disk full")


def test_a_logging_failure_never_fails_the_request(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher("lost_card"), app_store=BrokenStore()) as client:
        assert client.post("/v1/chat/completions", json=chat(SURE)).status_code == 200
        assert client.post("/v1/chat/completions", json=chat(UNSURE)).status_code == 200


def test_metrics_expose_the_counters(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher("lost_card")) as client:
        client.post("/v1/chat/completions", json=chat(SURE))
        client.post("/v1/chat/completions", json=chat(UNSURE))
        client.post("/v1/chat/completions", json=chat(SURE, n=3))
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    text = resp.text
    assert 'taskdistill_requests_total{route="student"} 1.0' in text
    assert 'taskdistill_requests_total{route="teacher"} 2.0' in text
    assert 'taskdistill_escalations_total{reason="low_confidence"} 1.0' in text
    assert 'taskdistill_escalations_total{reason="unsupported"} 1.0' in text
    assert 'taskdistill_request_latency_seconds_count{route="student"} 1.0' in text
    assert "taskdistill_student_latency_seconds_count 2.0" in text
    assert "taskdistill_teacher_latency_seconds_count 2.0" in text
    assert "taskdistill_escalation_unnormalised_total 0.0" in text
    assert 'taskdistill_teacher_errors_total{kind="timeout"} 0.0' in text


def test_metrics_are_per_app(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher()) as client:
        client.post("/v1/chat/completions", json=chat(SURE))
    with serve(make_spec(), FakeTeacher()) as client:
        text = client.get("/metrics").text
    assert 'taskdistill_requests_total{route="student"} 0.0' in text


def test_healthz_and_models(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher(mode="replay"), threshold=0.8, run_id="qwen-run-7") as client:
        health = client.get("/healthz")
        models = client.get("/v1/models")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "task": "support-intents",
        "run": "qwen-run-7",
        "threshold": 0.8,
        "teacher": "replay",
        "backend": "fake",
        "escalation_response": "canonical",
        "on_teacher_error": "student",
    }
    assert models.json()["data"][0]["id"] == "taskdistill/support-intents"
    assert [m["id"] for m in models.json()["data"]] == ["taskdistill/support-intents"]


def test_healthz_reports_always_escalate_as_null(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher(), threshold=math.inf) as client:
        assert client.get("/healthz").json()["threshold"] is None


def test_bad_json_is_a_400(serve: Serve) -> None:
    with serve(make_spec(), FakeTeacher()) as client:
        bad = client.post("/v1/chat/completions", content=b"{nope", headers={"content-type": "application/json"})
        array = client.post("/v1/chat/completions", json=[1, 2])
    assert bad.status_code == 400 and array.status_code == 400
    assert bad.headers["x-taskdistill-route"] == "error"


def test_the_teacher_is_closed_at_shutdown(serve: Serve) -> None:
    teacher = FakeTeacher()
    with serve(make_spec(), teacher):
        assert not teacher.closed
    assert teacher.closed


# thresholds ----------------------------------------------------------------------------------------


def test_threshold_zero_never_escalates_on_confidence(serve: Serve) -> None:
    backend = FakeBackend(answers={UNSURE: ("lost_card", 1e-6)})
    teacher = FakeTeacher()
    with serve(make_spec(), teacher, backend=backend, threshold=0.0) as client:
        resp = client.post("/v1/chat/completions", json=chat(UNSURE))
        unsupported = client.post("/v1/chat/completions", json=chat(UNSURE, n=2))
    assert resp.headers["x-taskdistill-route"] == "student"
    assert resp.json()["choices"][0]["message"]["content"] == "lost_card"
    assert unsupported.headers["x-taskdistill-reason"] == "unsupported"
    assert len(teacher.bodies) == 1


def test_threshold_zero_serves_an_invalid_extraction_as_generated(serve: Serve) -> None:
    spec = make_spec("extraction", student={"system_prompt": "Extract.", "max_tokens": 64})
    backend = FakeBackend(answers={SURE: ('{"vendor": "Example Ltd"', 0.9)})
    with serve(spec, FakeTeacher(), backend=backend, threshold=0.0) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE))
    assert resp.headers["x-taskdistill-route"] == "student"
    assert resp.headers["x-taskdistill-confidence"] == "0.000000"
    assert resp.json()["choices"][0]["message"]["content"] == '{"vendor": "Example Ltd"'


def test_invalid_extraction_escalates_and_is_never_a_fallback(serve: Serve) -> None:
    spec = make_spec("extraction", student={"system_prompt": "Extract.", "max_tokens": 64})
    backend = FakeBackend(answers={SURE: ("not json", 0.9)})
    with serve(spec, FakeTeacher(error=TeacherTimeout("slow")), backend=backend) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE))
    assert resp.status_code == 502
    assert resp.headers["x-taskdistill-reason"] == "low_confidence"


def test_always_escalate_threshold_escalates_everything(serve: Serve) -> None:
    backend = FakeBackend(answers={SURE: ("card_arrival", 1.0)})
    teacher = FakeTeacher("card_arrival")
    with serve(make_spec(), teacher, backend=backend, threshold=math.inf) as client:
        resp = client.post("/v1/chat/completions", json=chat(SURE))
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-reason"] == "low_confidence"
    assert resp.headers["x-taskdistill-confidence"] == "1.000000"
    assert len(teacher.bodies) == 1


def test_create_app_rejects_bad_arguments(store: Store) -> None:
    spec = make_spec()
    worker = ModelWorker(lambda: FakeBackend(), spec=spec)
    for threshold in (-0.1, math.nan):
        with pytest.raises(ValueError, match="threshold"):
            create_app(spec, worker=worker, teacher=FakeTeacher(), threshold=threshold, store=store)
    with pytest.raises(ValueError, match="token"):
        create_app(spec, worker=worker, teacher=FakeTeacher(), threshold=0.5, store=store, token="")
    spec.labels = []
    with pytest.raises(ValueError, match="labels"):
        create_app(spec, worker=worker, teacher=FakeTeacher(), threshold=0.5, store=store)


# the model worker ----------------------------------------------------------------------------------


@dataclass
class CountingBackend(FakeBackend):
    """Counts generations in flight to prove they never overlap."""

    active: int = 0
    max_active: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self._count_lock = threading.Lock()

    def generate_with_scores(
        self, messages: list[dict[str, str]], constraint: Any = None, max_tokens: int = 256
    ) -> Generation:
        with self._count_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            return super().generate_with_scores(messages, constraint, max_tokens)
        finally:
            with self._count_lock:
                self.active -= 1


def test_every_backend_call_runs_on_one_thread_and_never_interleaves(serve: Serve) -> None:
    rng = random.Random(3)
    texts = [SURE if rng.random() < 0.5 else UNSURE for _ in range(24)]
    backend = CountingBackend(answers=ANSWERS, delay_s=0.01)
    with serve(make_spec(), FakeTeacher("lost_card"), backend=backend) as client:
        worker: ModelWorker = client.app.state.worker  # type: ignore[attr-defined]
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda t: client.post("/v1/chat/completions", json=chat(t)), texts))
    assert all(r.status_code == 200 for r in responses)
    assert backend.loaded
    assert len(backend.threads) == 1
    assert backend.threads == {worker.thread_id}
    assert worker.thread_id != threading.get_ident()
    assert backend.max_active == 1
    assert sorted(backend.calls) == sorted(texts)


@dataclass
class GatedBackend(FakeBackend):
    """Holds its first generation until released, so later requests queue behind it."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.generating = threading.Event()
        self.release = threading.Event()

    def generate_with_scores(
        self, messages: list[dict[str, str]], constraint: Any = None, max_tokens: int = 256
    ) -> Generation:
        if not self.generating.is_set():
            self.generating.set()
            assert self.release.wait(10)
        return super().generate_with_scores(messages, constraint, max_tokens)


def test_student_ms_is_the_generation_time_without_the_queue_wait(serve: Serve, store: Store) -> None:
    backend = GatedBackend(answers=ANSWERS)
    with serve(make_spec(), FakeTeacher(), backend=backend) as client:
        worker: ModelWorker = client.app.state.worker  # type: ignore[attr-defined]
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post, "/v1/chat/completions", json=chat(SURE))
            assert backend.generating.wait(5)
            second = pool.submit(client.post, "/v1/chat/completions", json=chat(SURE))
            deadline = time.monotonic() + 5
            while worker._jobs.qsize() < 1 and time.monotonic() < deadline:
                time.sleep(0.005)
            time.sleep(0.3)
            backend.release.set()
            assert [first.result(10).status_code, second.result(10).status_code] == [200, 200]
        metrics = client.get("/metrics").text
    held, queued = sorted(store.iter_served("support-intents"), key=lambda r: r.student_ms or 0.0, reverse=True)
    assert held.student_ms is not None and held.student_ms >= 280
    assert queued.student_ms is not None and queued.student_ms < 150
    assert queued.total_ms is not None and queued.total_ms >= 280
    assert "taskdistill_student_latency_seconds_count 2.0" in metrics
    assert "taskdistill_student_queue_seconds_count 2.0" in metrics


def test_worker_runs_jobs_in_order_on_its_thread() -> None:
    spec = make_spec()
    backend = CountingBackend(answers=ANSWERS, delay_s=0.005)
    loads: list[int] = []

    def loader() -> FakeBackend:
        loads.append(threading.get_ident())
        return backend

    worker = ModelWorker(loader, spec=spec)
    worker.start()
    assert worker.wait_ready(5)
    assert worker.status == "ready"
    assert worker.backend_name == "fake"
    assert loads == [worker.thread_id]
    assert worker.trie is not None and worker.trie.labels == LABELS

    async def many() -> list[Any]:
        return await asyncio.gather(*(worker.run(text) for text in [SURE, UNSURE] * 6))

    predictions = asyncio.run(many())
    assert [p.answer for p in predictions] == ["card_arrival", "lost_card"] * 6
    assert predictions[0].confidence == pytest.approx(0.95)
    assert backend.max_active == 1
    assert backend.threads == {worker.thread_id}

    future: Future[int] = worker.submit(threading.get_ident)
    assert future.result(5) == worker.thread_id
    worker.stop()
    assert worker.status == "stopped"
    with pytest.raises(ModelNotReady):
        worker.submit(time.time)


def test_worker_surfaces_a_load_error(store: Store) -> None:
    def loader() -> FakeBackend:
        raise RuntimeError("weights not found")

    spec = make_spec()
    worker = ModelWorker(loader, spec=spec)
    worker.start()
    assert not worker.wait_ready(5)
    assert worker.status == "error"
    assert isinstance(worker.load_error, RuntimeError)
    with pytest.raises(ModelNotReady, match="weights not found"):
        asyncio.run(worker.run(SURE))
    worker.stop()

    teacher = FakeTeacher()
    app = create_app(
        spec, worker=ModelWorker(loader, spec=spec), teacher=teacher, threshold=0.5, store=store, run_id="r"
    )
    with pytest.raises(ModelLoadError, match="weights not found"), TestClient(app):
        pass
    assert teacher.closed
