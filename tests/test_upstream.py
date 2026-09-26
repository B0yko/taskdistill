from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from taskdistill.config import TaskSpec
from taskdistill.ledger import Ledger
from taskdistill.teacher.base import BudgetExceeded, TeacherError, TeacherHTTPError, TeacherResult, TeacherTimeout
from taskdistill.teacher.client import LiveTeacher
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.replay import ReplayTeacher, expected_manifest, write_recording
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request
from taskdistill.teacher.upstream import create_upstream_app

MODEL = "vendor/model-a"
QUERY = "My card still has not arrived after two weeks."


@pytest.fixture
def spec() -> TaskSpec:
    spec = TaskSpec.model_validate(
        {
            "task": "demo-intents",
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": {"model": MODEL, "max_tokens": 24},
            "student": {"system_prompt": "Classify the message."},
            "cascade": {"target": 0.97},
        }
    )
    spec.teacher_prompt = "Label the customer's message with one intent.\n"
    spec.labels = ["card_arrival", "lost_card"]
    return spec


@pytest.fixture
def body(spec: TaskSpec) -> dict[str, Any]:
    return build_teacher_request(spec, QUERY)


@pytest.fixture
def replay(tmp_path: Path, spec: TaskSpec, body: dict[str, Any]) -> ReplayTeacher:
    rec = {
        "key": request_key(body),
        "output": "card_arrival",
        "usage": {"prompt_tokens": 120, "completion_tokens": 3, "cost": 1.2e-5},
        "latency_ms": 410.5,
        "provider": "Alpha",
        "finish_reason": "stop",
        "timestamp": 1790000000.0,
    }
    manifest = {**expected_manifest(spec, "2026-09-26"), "created": "2026-09-26T12:00:00+00:00"}
    return ReplayTeacher(write_recording(tmp_path / "rec.jsonl.gz", manifest, [rec]), spec)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/chat/completions"])
def test_replay_answers_chat_completions(replay: ReplayTeacher, body: dict[str, Any], path: str) -> None:
    with TestClient(create_upstream_app(replay)) as client:
        resp = client.post(path, json=body, headers={"Authorization": "Bearer sk-test-client"})
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-teacher"] == "replay"
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == MODEL
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "card_arrival"}
    assert data["usage"]["completion_tokens"] == 3


def test_replay_miss_is_a_404(replay: ReplayTeacher, spec: TaskSpec) -> None:
    with TestClient(create_upstream_app(replay)) as client:
        resp = client.post("/v1/chat/completions", json=build_teacher_request(spec, "How do I close my account?"))
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["type"] == "replay_miss"
    assert "replay miss" in error["message"]


def test_replay_stream(replay: ReplayTeacher, body: dict[str, Any]) -> None:
    with TestClient(create_upstream_app(replay)) as client:
        resp = client.post("/v1/chat/completions", json={**body, "stream": True})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.endswith("data: [DONE]\n\n")
    blocks = [block for block in resp.text.split("\n\n") if block]
    assert blocks[-1] == "data: [DONE]"
    events = [json.loads(block.removeprefix("data: ")) for block in blocks[:-1]]
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "card_arrival"


def test_replay_stream_miss_is_a_404(replay: ReplayTeacher, spec: TaskSpec) -> None:
    with TestClient(create_upstream_app(replay)) as client:
        resp = client.post("/v1/chat/completions", json={**build_teacher_request(spec, "unrecorded"), "stream": True})
    assert resp.status_code == 404


@pytest.mark.parametrize("path", ["/v1/models", "/models"])
def test_models_lists_the_configured_model(replay: ReplayTeacher, path: str) -> None:
    with TestClient(create_upstream_app(replay)) as client:
        resp = client.get(path)
    assert resp.status_code == 200
    assert [m["id"] for m in resp.json()["data"]] == [MODEL]
    with TestClient(create_upstream_app(replay, models=["a/b", "c/d"])) as client:
        assert [m["id"] for m in client.get(path).json()["data"]] == ["a/b", "c/d"]


def test_invalid_json_is_a_400(replay: ReplayTeacher) -> None:
    with TestClient(create_upstream_app(replay)) as client:
        resp = client.post("/v1/chat/completions", content=b"{not json", headers={"content-type": "application/json"})
        assert resp.status_code == 400
        assert client.post("/v1/chat/completions", json=[1, 2]).status_code == 400


class FailingSource:
    mode: Literal["live", "replay"] = "live"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.closed = False

    async def complete(self, body: dict[str, Any]) -> TeacherResult:
        raise self.error

    async def stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        raise self.error
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (BudgetExceeded("refused: run cap --max-usd=0.50"), 402),
        (TeacherHTTPError(429, "rate limited"), 429),
        (TeacherHTTPError(400, "bad request"), 400),
        (TeacherHTTPError(200, "error object in a 200 body"), 502),
        (TeacherTimeout("no answer"), 504),
        (TeacherError("something else"), 502),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_teacher_errors_map_to_statuses(error: Exception, status: int, stream: bool) -> None:
    source = FailingSource(error)
    with TestClient(create_upstream_app(source)) as client:
        resp = client.post("/v1/chat/completions", json={"model": MODEL, "messages": [], "stream": stream})
    assert resp.status_code == status
    assert resp.json()["error"]["code"] == status
    assert source.closed  # closed with the app


def test_live_teacher_behind_the_upstream_uses_only_the_teacher_key(tmp_path: Path, body: dict[str, Any]) -> None:
    base = "https://teacher.example.com/api/v1"
    pricing = PricingSnapshot(
        date="2026-09-26", source="test", models={MODEL: {"default": ModelPrice(1e-7, 2e-7), "providers": {}}}
    )
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=1.0)
    teacher = LiveTeacher(
        base, "sk-test-teacher", ledger=ledger, pricing=pricing, task="demo-intents", phase="demo", run_id="r1"
    )
    seen: list[httpx.Request] = []
    response = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1790000000,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "card_arrival"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 3, "cost": 1.2e-5},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=response)

    with respx.mock() as mock:
        mock.post(f"{base}/chat/completions").mock(side_effect=handler)
        with TestClient(create_upstream_app(teacher)) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": "Bearer sk-test-client", "X-Custom": "client-only"},
            )
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-teacher"] == "live"
    assert resp.json() == response
    assert seen[0].headers["authorization"] == "Bearer sk-test-teacher"
    assert "x-custom" not in seen[0].headers
    assert ledger.spent() == pytest.approx(1.2e-5)


def test_budget_refusal_through_the_upstream_is_a_402(tmp_path: Path, body: dict[str, Any]) -> None:
    pricing = PricingSnapshot(
        date="2026-09-26", source="test", models={MODEL: {"default": ModelPrice(1e-3, 1e-3), "providers": {}}}
    )
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=0.01)
    teacher = LiveTeacher(
        "https://teacher.example.com/api/v1", "sk-test-teacher", ledger=ledger, pricing=pricing,
        task="demo-intents", phase="demo", run_id="r1", run_cap=0.001,
    )  # fmt: skip
    with TestClient(create_upstream_app(teacher)) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 402
    assert "run cap --max-usd" in resp.json()["error"]["message"]
