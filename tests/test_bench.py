"""Live bench: warm-up exclusion, one request at a time, replay refusal, spend projection and machine state."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from taskdistill import bench as bench_mod
from taskdistill import paths
from taskdistill.bench import BenchError, run_bench
from taskdistill.config import TaskSpec
from taskdistill.curate.extract import input_hash
from taskdistill.ledger import Ledger, worst_case_cost
from taskdistill.store import Store
from taskdistill.teacher.client import SpendNotConfirmed
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.requests import build_teacher_request

TASK = "bench-intents"
URL = "http://127.0.0.1:8000"
SLOW_WARMUP_S = 0.3


def _spec() -> TaskSpec:
    spec = TaskSpec.model_validate(
        {
            "task": TASK,
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": {"model": "vendor/model-a", "max_tokens": 24},
            "student": {"system_prompt": "Classify the message."},
            "cascade": {"target": 0.97},
        }
    )
    spec.teacher_prompt = "Label the customer's message with one intent.\n"
    spec.labels = ["card_arrival", "lost_card", "top_up"]
    return spec


def _query(i: int) -> str:
    return f"Question number {i}: where is my card?"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TASKDISTILL_SERVER_TOKEN", raising=False)
    monkeypatch.delenv("TASKDISTILL_BUDGET_USD", raising=False)
    return paths.home()


def _workspace(n_inputs: int, *, missing: int = 0) -> tuple[TaskSpec, Store, list[str]]:
    """``n_inputs`` test inputs imported into the store (plus ``missing`` test rows without a raw input)."""
    spec = _spec()
    store = Store()
    queries = [_query(i) for i in range(n_inputs)]
    store.add_imports(TASK, "inputs", [{"input": q, "meta": {"split": "test"}} for q in queries])
    hashes = [input_hash(q) for q in queries]
    meta = paths.data_dir(TASK) / "test.meta.jsonl"
    rows = [{"input_hash": h, "gold": None, "teacher": "card_arrival", "meta": {"split": "test"}} for h in hashes]
    rows[1:1] = [{"input_hash": f"{k:064x}", "gold": None, "teacher": "lost_card", "meta": {}} for k in range(missing)]
    meta.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return spec, store, queries


#: (route, reason, HTTP status or None for 200) of the fake server's answer to one input.
Route = Callable[[str], tuple[str, str | None, int | None]]


def _student(_: str) -> tuple[str, str | None, int | None]:
    return "student", None, None


class FakeServer:
    """A MockTransport stand-in for ``taskdistill serve``: headers like the real server, a log of what arrived."""

    def __init__(
        self,
        *,
        threshold: float | None = 0.5,
        teacher: str = "live",
        route: Route = _student,
        slow: set[str] | None = None,
        cost_per_escalation: float = 0.0,
        token: str | None = None,
        task: str | None = TASK,
    ) -> None:
        self.threshold = threshold
        self.teacher = teacher
        self.route = route
        self.slow = slow or set()
        self.cost = cost_per_escalation
        self.token = token
        self.task = task
        self.seen: list[str] = []
        self.auth: list[str | None] = []
        self.ledger = Ledger() if cost_per_escalation else None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth.append(request.headers.get("authorization"))
        if self.token is not None and request.headers.get("authorization") != f"Bearer {self.token}":
            return httpx.Response(401, json={"error": "unauthorised"})
        if request.url.path == "/healthz":
            health = {"status": "ok", "run": "run-a", "threshold": self.threshold, "teacher": "x"}
            if self.task is not None:
                health["task"] = self.task
            return httpx.Response(200, json=health)
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        self.seen.append(text)
        if text in self.slow:
            time.sleep(SLOW_WARMUP_S)
        route, reason, status = self.route(text)
        headers = {"x-taskdistill-route": route, "x-taskdistill-confidence": "0.9"}
        if route != "student" and (reason is not None or route != "error"):  # an "error" without a reason: no student
            headers["x-taskdistill-reason"] = reason or "low_confidence"
            headers["x-taskdistill-teacher"] = self.teacher
            if self.ledger is not None and route == "teacher":
                res = self.ledger.reserve(self.cost, task=TASK, phase="serve", run_id="serve-1", model="vendor/model-a")
                self.ledger.settle(res, self.cost)
        payload = {"object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": "top_up"}}]}
        return httpx.Response(status or 200, json=payload, headers=headers)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def test_warmup_requests_are_excluded_and_timed_inputs_are_distinct(home: Path) -> None:
    spec, store, queries = _workspace(10)
    server = FakeServer(slow=set(queries[:3]))
    out = home / "b.json"
    result = run_bench(URL, spec, n=5, warmup=3, store=store, client=server.client(), out=out)

    assert server.seen == queries[:8]  # data order: 3 warm-up inputs, then 5 new ones
    assert [r["input_hash"] for r in result["requests"]] == [input_hash(q) for q in queries[3:8]]
    assert result["n"] == 5 and result["warmup"] == 3 and result["concurrency"] == 1
    assert result["latency_ms"]["n"] == 5
    assert result["latency_ms"]["p95"] < SLOW_WARMUP_S * 1000 / 2  # the slow warm-up calls are not in the stats
    assert result["mode"] == "cascade" and result["threshold"] == 0.5 and result["run_id"] == "run-a"
    assert json.loads(out.read_text(encoding="utf-8")) == result
    # --out is a copy: the workspace file that `report` reads is always written
    written = sorted((paths.task_home(TASK) / "bench").glob("bench_*.json"))
    assert len(written) == 1 and json.loads(written[0].read_text(encoding="utf-8")) == result


def test_escalation_rate_routes_and_reasons(home: Path) -> None:
    spec, store, queries = _workspace(8)

    def route(text: str) -> tuple[str, str | None, int | None]:
        index = queries.index(text)
        if index in (3, 4):
            return "teacher", "low_confidence", None
        if index == 5:
            return "teacher", "input_unparsed", None
        return "student", None, None

    server = FakeServer(route=route, cost_per_escalation=0.002)
    result = run_bench(URL, spec, n=6, warmup=2, store=store, client=server.client())
    assert result["route_counts"] == {"student": 3, "teacher": 3}
    assert result["reasons"] == {"input_unparsed": 1, "low_confidence": 2}
    assert result["escalation_rate"] == pytest.approx(0.5)
    assert result["escalations"] == 3 and result["failed_escalations"] == 0
    assert result["teacher_modes"] == {"live": 3}
    assert result["spend_usd"] == pytest.approx(0.006) and result["spend_note"] is None


def test_failed_escalations_count_as_escalations(home: Path) -> None:
    spec, store, queries = _workspace(7)

    def route(text: str) -> tuple[str, str | None, int | None]:
        index = queries.index(text)
        if index == 2:
            return "error", "low_confidence", 502  # the teacher call failed and on_teacher_error is "error"
        if index == 3:
            return "student-fallback", "low_confidence", None
        if index == 4:
            return "error", None, 503  # the model is not ready: the student never ran
        return "student", None, None

    result = run_bench(URL, spec, n=6, warmup=1, store=store, client=FakeServer(route=route).client())
    assert result["errors"] == 2 and result["latency_ms"]["n"] == 4
    assert result["escalations"] == 2 and result["failed_escalations"] == 2
    assert result["escalation_rate"] == pytest.approx(2 / 6)
    assert result["route_counts"] == {"error": 2, "student": 3, "student-fallback": 1}
    # failed teacher calls are not live answers: an unchanged ledger is a measured $0
    assert result["live_teacher_answers"] == 0 and result["spend_usd"] == 0.0


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


def test_requests_go_one_at_a_time_over_http(home: Path) -> None:
    spec, store, _ = _workspace(12)
    state = {"in_flight": 0, "max": 0, "count": 0}
    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok", "task": TASK, "threshold": 0.7, "teacher": "live", "run": "run-a"})

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        await request.body()
        state["in_flight"] += 1
        state["max"] = max(state["max"], state["in_flight"])
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        state["count"] += 1
        return JSONResponse({"object": "chat.completion"}, headers={"x-taskdistill-route": "student"})

    with _Served(app) as served:
        result = run_bench(served.url + "/v1", spec, n=8, warmup=4, store=store)
    assert state["count"] == 12
    assert state["max"] == 1
    assert result["n"] == 8 and result["url"] == served.url


def test_replayed_escalation_fails_the_bench(home: Path) -> None:
    spec, store, queries = _workspace(6)

    def route(text: str) -> tuple[str, str | None, int | None]:
        return ("teacher", "low_confidence", None) if text == queries[3] else ("student", None, None)

    server = FakeServer(route=route, teacher="replay")
    with pytest.raises(BenchError, match="replay"):
        run_bench(URL, spec, n=4, warmup=1, store=store, client=server.client())
    assert server.seen == queries[:4]  # stopped at the first replayed escalation
    assert not (paths.task_home(TASK) / "bench").exists()


def test_replayed_escalation_during_warmup_fails_too(home: Path) -> None:
    spec, store, _ = _workspace(6)
    server = FakeServer(route=lambda _: ("student-fallback", "low_confidence", None), teacher="replay")
    with pytest.raises(BenchError, match="replayed teacher"):
        run_bench(URL, spec, n=2, warmup=2, store=store, client=server.client())
    assert len(server.seen) == 1


def test_threshold_zero_is_student_only_and_machine_state_is_taken_before_timing(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, store, _ = _workspace(9)
    server = FakeServer(threshold=0.0, teacher="replay")  # a replay teacher is fine when nothing escalates
    snapshots: list[int] = []

    def fake_state() -> dict[str, Any]:
        snapshots.append(len(server.seen))
        return {"load_average": [1.0, 1.0, 1.0], "memory": {"free_percent": 70}, "seen": len(server.seen)}

    monkeypatch.setattr(bench_mod, "machine_state", fake_state)
    result = run_bench(URL, spec, n=5, warmup=4, store=store, client=server.client())
    assert result["mode"] == "student_only" and result["threshold"] == 0.0
    assert snapshots[0] == 4  # after the warm-up, before the first timed request
    assert result["machine_state"]["seen"] == 4
    assert result["machine_state_after"]["seen"] == 9
    assert result["escalation_rate"] == 0.0
    assert result["hardware"]


def test_always_escalate_threshold_is_cascade_mode(home: Path) -> None:
    spec, store, _ = _workspace(4)
    result = run_bench(URL, spec, n=2, warmup=1, store=store, client=FakeServer(threshold=None).client())
    assert result["mode"] == "cascade" and result["threshold"] is None


def test_projection_needs_yes_above_fifty_cents(home: Path) -> None:
    spec, store, _ = _workspace(60)
    escalate_all = lambda _: ("teacher", "low_confidence", None)  # noqa: E731
    server = FakeServer(route=escalate_all, cost_per_escalation=0.01)
    with pytest.raises(SpendNotConfirmed, match="projected"):
        run_bench(URL, spec, n=58, warmup=2, store=store, client=server.client())
    assert len(server.seen) == 52  # warm-up + the first 50 timed requests, then the projection refused

    server = FakeServer(route=escalate_all, cost_per_escalation=0.01)
    result = run_bench(URL, spec, n=58, warmup=2, yes=True, store=store, client=server.client())
    projection = result["projection"]
    assert projection["after_requests"] == 52
    assert projection["spent_usd"] == pytest.approx(0.52)
    assert projection["projected_usd"] == pytest.approx(0.60)
    assert result["spend_usd"] == pytest.approx(0.60)
    assert result["n"] == 58


def test_projection_below_threshold_needs_no_yes(home: Path) -> None:
    spec, store, queries = _workspace(60)

    def route(text: str) -> tuple[str, str | None, int | None]:
        return ("teacher", "low_confidence", None) if queries.index(text) % 10 == 0 else ("student", None, None)

    server = FakeServer(route=route, cost_per_escalation=0.001)
    result = run_bench(URL, spec, n=58, warmup=2, store=store, client=server.client())
    assert result["projection"]["projected_usd"] < 0.5
    assert result["n"] == 58


def test_server_of_another_task_is_refused(home: Path) -> None:
    spec, store, _ = _workspace(4)
    server = FakeServer(task="other-task")
    with pytest.raises(BenchError, match="serves task 'other-task'"):
        run_bench(URL, spec, n=2, warmup=1, store=store, client=server.client())
    assert server.seen == []
    with pytest.raises(BenchError, match="did not report its task"):
        run_bench(URL, spec, n=2, warmup=1, store=store, client=FakeServer(task=None).client())


def test_unmeasurable_spend_fails_closed(home: Path) -> None:
    """Live escalations with no serve spend in this ledger: the server keeps its ledger in another workspace."""
    spec, store, _ = _workspace(80)
    escalate_all = lambda _: ("teacher", "low_confidence", None)  # noqa: E731

    server = FakeServer(route=escalate_all)  # live teacher answers, nothing in this workspace's ledger
    with pytest.raises(SpendNotConfirmed, match="cannot be projected"):
        run_bench(URL, spec, n=70, warmup=2, store=store, client=server.client())
    assert len(server.seen) == 52  # refused at the projection point

    server = FakeServer(route=escalate_all)
    with pytest.raises(BenchError, match=r"--max-usd 0\.01 cannot be enforced"):
        run_bench(URL, spec, n=70, warmup=2, yes=True, max_usd=0.01, store=store, client=server.client())
    assert len(server.seen) == 1  # the first live answer shows the cap cannot be checked

    result = run_bench(URL, spec, n=70, warmup=2, yes=True, store=store, client=FakeServer(route=escalate_all).client())
    assert result["n"] == 70
    assert result["spend_usd"] is None and "cannot be measured" in result["spend_note"]
    assert result["live_teacher_answers"] == 72
    assert result["projection"]["projected_usd"] is None and result["projection"]["unmeasured"]
    assert any("spend not measured" in note for note in result["notes"])

    short = run_bench(URL, spec, n=5, warmup=1, store=store, client=FakeServer(route=escalate_all).client())
    assert short["spend_usd"] is None and short["projection"] is None  # below the projection sample


def test_max_usd_stops_the_bench(home: Path) -> None:
    spec, store, _ = _workspace(40)
    server = FakeServer(route=lambda _: ("teacher", "low_confidence", None), cost_per_escalation=0.01)
    result = run_bench(URL, spec, n=30, warmup=2, yes=True, max_usd=0.1, store=store, client=server.client())
    assert len(server.seen) == 10
    assert result["n"] == 8
    assert result["spend_usd"] == pytest.approx(0.1)
    assert "max-usd" in result["stopped"]


def test_max_usd_reached_during_warmup_fails(home: Path) -> None:
    spec, store, _ = _workspace(10)
    server = FakeServer(route=lambda _: ("teacher", "low_confidence", None), cost_per_escalation=0.05)
    with pytest.raises(BenchError, match="during the warm-up"):
        run_bench(URL, spec, n=5, warmup=4, yes=True, max_usd=0.1, store=store, client=server.client())
    assert len(server.seen) == 2


def _price_teacher(request_fee: float) -> None:
    """A pricing snapshot in the workspace that bounds every teacher call by ``request_fee``."""
    PricingSnapshot(
        date="2026-09-26",
        source="test",
        models={"vendor/model-a": {"default": ModelPrice(0.0, 0.0, request_fee), "providers": {}}},
    ).save()


def _serve_spend() -> float:
    return Ledger().spent(task=TASK, phase="serve")


def test_max_usd_stops_before_a_request_that_could_cross_it(home: Path) -> None:
    spec, store, _ = _workspace(12)
    _price_teacher(0.03)  # one teacher call costs at most $0.03
    server = FakeServer(route=lambda _: ("teacher", "low_confidence", None), cost_per_escalation=0.03)
    result = run_bench(URL, spec, n=10, warmup=0, yes=True, max_usd=0.05, store=store, client=server.client())
    assert len(server.seen) == 1  # a second request could bring the spend to $0.06
    assert result["n"] == 1
    assert result["spend_usd"] == pytest.approx(0.03) and result["spend_usd"] <= 0.05
    assert result["stopped"].startswith("--max-usd 0.05 would be crossed")
    assert "next request up to $0.030000" in result["stopped"]
    assert "stopped after 1 requests" in result["stopped"]
    assert result["stopped"] in result["notes"]


def test_max_usd_bound_is_the_request_worst_case(home: Path) -> None:
    spec, store, queries = _workspace(6)
    price = ModelPrice(1e-4, 1e-3)
    PricingSnapshot(
        date="2026-09-26", source="test", models={"vendor/model-a": {"default": price, "providers": {}}}
    ).save()
    worst = [worst_case_cost(build_teacher_request(spec, q), price) for q in queries]
    cap = worst[0] + worst[1] - 1e-6  # the first request fits, the second could cross the cap
    server = FakeServer(route=lambda _: ("teacher", "low_confidence", None), cost_per_escalation=worst[0] / 2)
    result = run_bench(URL, spec, n=5, warmup=0, yes=True, max_usd=cap, store=store, client=server.client())
    # after $worst[0]/2, the second request's worst case still fits; after twice that, the third one does not
    assert worst[0] / 2 + worst[1] <= cap < worst[0] + worst[2]
    assert len(server.seen) == 2
    assert result["spend_usd"] == pytest.approx(worst[0]) and result["spend_usd"] <= cap


def test_max_usd_bound_is_at_least_the_most_one_request_has_cost(home: Path) -> None:
    """A request can cost more than one call's worst case (the server retried a call that timed out)."""
    spec, store, _ = _workspace(12)
    _price_teacher(0.01)
    server = FakeServer(route=lambda _: ("teacher", "low_confidence", None), cost_per_escalation=0.03)
    result = run_bench(URL, spec, n=10, warmup=0, yes=True, max_usd=0.05, store=store, client=server.client())
    assert len(server.seen) == 1
    assert result["spend_usd"] == pytest.approx(0.03)


def test_max_usd_without_a_price_uses_the_most_one_request_has_cost(home: Path) -> None:
    spec, store, _ = _workspace(12)  # no snapshot prices vendor/model-a
    server = FakeServer(route=lambda _: ("teacher", "low_confidence", None), cost_per_escalation=0.03)
    result = run_bench(URL, spec, n=10, warmup=0, yes=True, max_usd=0.05, store=store, client=server.client())
    assert len(server.seen) == 1
    assert result["spend_usd"] == pytest.approx(0.03)
    assert any("no price for the teacher model vendor/model-a" in note for note in result["notes"])


def test_max_usd_is_not_crossed_during_the_warm_up(home: Path) -> None:
    spec, store, _ = _workspace(10)
    _price_teacher(0.03)
    server = FakeServer(route=lambda _: ("teacher", "low_confidence", None), cost_per_escalation=0.03)
    with pytest.raises(BenchError, match=r"--max-usd 0\.05 would be crossed .* during the warm-up, after 1 requests"):
        run_bench(URL, spec, n=5, warmup=4, yes=True, max_usd=0.05, store=store, client=server.client())
    assert len(server.seen) == 1
    assert _serve_spend() == pytest.approx(0.03)


def test_max_usd_leaves_student_answers_alone_while_they_cost_nothing(home: Path) -> None:
    spec, store, _ = _workspace(12)
    _price_teacher(0.03)
    result = run_bench(URL, spec, n=10, warmup=2, max_usd=0.05, store=store, client=FakeServer().client())
    assert result["n"] == 10 and result["stopped"] is None


def test_rows_without_raw_input_are_skipped_and_reported(home: Path) -> None:
    spec, store, _ = _workspace(6, missing=2)
    result = run_bench(URL, spec, n=3, warmup=2, store=store, client=FakeServer().client())
    assert result["skipped_no_raw_input"] == 2
    assert any("no raw input" in note for note in result["notes"])
    written = sorted((paths.task_home(TASK) / "bench").glob("bench_*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text(encoding="utf-8"))["n"] == 3


def test_too_few_inputs_and_n_capped(home: Path) -> None:
    spec, store, _ = _workspace(3)
    with pytest.raises(BenchError, match="warm-up"):
        run_bench(URL, spec, n=5, warmup=3, store=store, client=FakeServer().client())
    result = run_bench(URL, spec, n=5, warmup=1, store=store, client=FakeServer().client())
    assert result["n"] == 2 and result["n_requested"] == 5
    assert any("n reduced to 2" in note for note in result["notes"])


def test_token_is_sent_as_bearer(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, store, _ = _workspace(3)
    monkeypatch.setenv("TASKDISTILL_SERVER_TOKEN", "serve-token-for-tests")
    server = FakeServer(token="serve-token-for-tests")
    run_bench(URL, spec, n=2, warmup=1, store=store, client=server.client())
    assert set(server.auth) == {"Bearer serve-token-for-tests"}
    with pytest.raises(BenchError, match="HTTP 401"):
        run_bench(URL, spec, n=2, warmup=1, token="wrong", store=store, client=server.client())


def test_unreachable_server_is_a_bench_error(home: Path) -> None:
    spec, store, _ = _workspace(3)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(BenchError, match="did not answer"):
        run_bench(URL, spec, n=1, warmup=1, store=store, client=httpx.Client(transport=httpx.MockTransport(refuse)))
