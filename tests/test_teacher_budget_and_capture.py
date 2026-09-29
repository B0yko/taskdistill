"""Teacher client budgeting and capture: reservations and per-call charges in the ledger, run caps, recorded answers
and the serve-time timeout and retry policy."""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
import respx

from taskdistill.config import TaskSpec
from taskdistill.ledger import (
    DEFAULT_MAX_TOKENS,
    Ledger,
    UnboundedCompletion,
    prompt_upper_bound,
    reservation_cost,
    worst_case_cost,
)
from taskdistill.store import Store
from taskdistill.teacher import client as client_mod
from taskdistill.teacher import factory, record
from taskdistill.teacher.base import BudgetExceeded, TeacherError, TeacherHTTPError, TeacherResult, TeacherTimeout
from taskdistill.teacher.cache import ResponseCache, request_context
from taskdistill.teacher.client import LiveTeacher, charged_usd
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.record import build_recording, default_keys
from taskdistill.teacher.replay import RECORD_FIELDS, RecordingError, ReplayTeacher, load_recording
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

BASE = "https://teacher.example.com/api/v1"
URL = f"{BASE}/chat/completions"
MODEL = "vendor/model-a"
TASK = "demo-intents"
KEY = "sk-test-teacher"
PRICE = ModelPrice(prompt=1e-6, completion=2e-6)
# make_body(): prompt bound len("hello") + 16 = 21 tokens, max_tokens 8 -> 21e-6 + 8 x 2e-6 = 37e-6
WORST = 37e-6
QUERIES = ["My card still has not arrived.", "I think I lost my card.", "Where is my new card?"]


def make_body(text: str = "hello", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 8,
        "provider": {"order": ["alpha/fp8"], "allow_fallbacks": False},
    }
    body.update(extra)
    return body


def unbounded_body(**extra: Any) -> dict[str, Any]:
    body = make_body(**extra)
    del body["max_tokens"]
    return body


def completion(content: str = "card_arrival", *, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1790000000,
        "model": MODEL,
        "provider": "Alpha",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": usage if usage is not None else {"prompt_tokens": 12, "completion_tokens": 3, "cost": 2.1e-5},
    }


class LimitedSnapshot(PricingSnapshot):
    """A snapshot that knows each model's maximum completion tokens, per endpoint like ``price_for``."""

    limits: dict[str, dict[str | None, int]] = {MODEL: {None: 20_000, "alpha/fp8": 20_000}}  # noqa: RUF012
    asked: list[tuple[str, str | None]] = []  # noqa: RUF012

    def max_completion_tokens(self, model: str, provider: str | None = None) -> int | None:
        self.asked.append((model, provider))
        by_provider = self.limits.get(model, {})
        return by_provider.get(provider, by_provider.get(None))


def snapshot(cls: type[PricingSnapshot] = PricingSnapshot) -> PricingSnapshot:
    return cls(date="2026-09-26", source="test", models={MODEL: {"default": PRICE, "providers": {"alpha/fp8": PRICE}}})


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    for name in ("TASKDISTILL_TEACHER_API_KEY", "OPENROUTER_API_KEY", "TASKDISTILL_BUDGET_USD"):
        monkeypatch.delenv(name, raising=False)
    return workspace


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.sqlite", global_cap=5.0)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(client_mod, "_sleep", fake_sleep)
    return delays


def live(ledger: Ledger, pricing: PricingSnapshot | None = None, **kw: Any) -> LiveTeacher:
    params: dict[str, Any] = {"task": TASK, "phase": "serve", "run_id": "serve-1", "rng": random.Random(7)}
    params.update(kw)
    return LiveTeacher(BASE, KEY, ledger=ledger, pricing=pricing or snapshot(), **params)


def run(teacher: LiveTeacher, body: dict[str, Any]) -> TeacherResult:
    async def go() -> TeacherResult:
        try:
            return await teacher.complete(body)
        finally:
            await teacher.aclose()

    return asyncio.run(go())


def collect(teacher: LiveTeacher, body: dict[str, Any]) -> bytes:
    async def go() -> bytes:
        try:
            return b"".join([chunk async for chunk in teacher.stream(body)])
        finally:
            await teacher.aclose()

    return asyncio.run(go())


def ledger_rows(ledger: Ledger) -> list[tuple[str, str]]:
    with sqlite3.connect(ledger.path) as conn:
        return [(str(r[0]), str(r[1])) for r in conn.execute("SELECT request_key, status FROM ledger ORDER BY id")]


def make_spec(**teacher: Any) -> TaskSpec:
    teacher_cfg: dict[str, Any] = {
        "model": MODEL,
        "base_url": BASE,
        "max_tokens": 24,
        "extra_body": {"provider": {"order": ["alpha/fp8"], "allow_fallbacks": False}, "reasoning": {"enabled": False}},
    }
    teacher_cfg.update(teacher)
    spec = TaskSpec.model_validate(
        {
            "task": TASK,
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": teacher_cfg,
            "student": {"system_prompt": "Classify the message."},
            "cascade": {"target": 0.97},
        }
    )
    spec.teacher_prompt = "Label the customer's message with one intent.\n"
    spec.labels = ["card_arrival", "lost_card"]
    spec.source = f"tasks/{TASK}/task.yaml"
    return spec


# (2) a body without max_tokens is reserved at what can really be sent --------------------------------
def test_an_unbounded_body_is_refused_when_the_model_limit_is_unknown(ledger: Ledger, sleeps: list[float]) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        with pytest.raises(UnboundedCompletion, match="sets no max_tokens") as info:
            run(live(ledger), unbounded_body())
        with pytest.raises(UnboundedCompletion):
            collect(live(ledger), unbounded_body(stream=True))
    assert isinstance(info.value, BudgetExceeded)  # serve reports it as a refused (budget) teacher call
    # the only remedy that works is a limit in the request (`pricing refresh` records no completion limits)
    assert "max_tokens" in str(info.value) and "pricing refresh" not in str(info.value)
    assert charged_usd(info.value) == 0.0
    assert route.call_count == 0
    assert ledger_rows(ledger) == []


def test_an_unbounded_body_is_reserved_at_the_model_maximum_and_sent_unchanged(tmp_path: Path) -> None:
    body = unbounded_body()
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        # the teacher writes far more than the 4096 tokens the ledger used to assume
        return httpx.Response(200, json=completion(usage={"prompt_tokens": 12, "completion_tokens": 19_000}))

    at_model_max = worst_case_cost(body, PRICE, max_completion_tokens=20_000)
    assumed = worst_case_cost(body, PRICE)  # the old 4096-token guess
    assert assumed == pytest.approx(21e-6 + DEFAULT_MAX_TOKENS * 2e-6)
    assert at_model_max == pytest.approx(21e-6 + 20_000 * 2e-6)

    # a cap between the guess and the model maximum: the old reservation fitted, then the call crossed the cap
    tight = Ledger(tmp_path / "tight.sqlite", global_cap=(assumed + at_model_max) / 2)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL).mock(side_effect=handler)
        with pytest.raises(BudgetExceeded, match="TASKDISTILL_BUDGET_USD"):
            run(live(tight, snapshot(LimitedSnapshot)), body)
        assert route.call_count == 0

        roomy = Ledger(tmp_path / "roomy.sqlite", global_cap=at_model_max)
        result = run(live(roomy, snapshot(LimitedSnapshot)), body)
    assert sent == [body]  # only what the caller sent: no max_tokens added
    assert LimitedSnapshot.asked[-1] == (MODEL, "alpha/fp8")  # the pinned endpoint's limit is asked for
    assert result.cost_usd == pytest.approx(12e-6 + 19_000 * 2e-6)
    assert roomy.committed() <= roomy.global_cap


def test_the_ledger_never_reserves_the_projection_guess() -> None:
    body = unbounded_body(n=2)
    with pytest.raises(UnboundedCompletion):
        reservation_cost(body, PRICE, model=MODEL)
    assert reservation_cost(body, PRICE, max_completion_tokens=100) == pytest.approx(21e-6 + 2 * 100 * 2e-6)
    bounded = make_body(max_tokens=5)
    # an explicit limit wins over the model maximum
    assert reservation_cost(bounded, PRICE, max_completion_tokens=100) == pytest.approx(21e-6 + 5 * 2e-6)
    assert reservation_cost(bounded, PRICE) == worst_case_cost(bounded, PRICE)


# (3) run ids are unique per invocation -----------------------------------------------------------
def test_commands_started_in_the_same_second_do_not_share_a_run_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FrozenClock:
        @staticmethod
        def now(tz: Any = None) -> Any:
            from datetime import datetime

            return datetime(2026, 9, 26, 14, 40, 29, tzinfo=tz)

    monkeypatch.setattr(factory, "datetime", FrozenClock)
    first, second = factory.new_run_id("curate"), factory.new_run_id("curate")
    assert first != second
    assert first.startswith("curate-20260926T144029-") and second.startswith("curate-20260926T144029-")

    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=5.0)
    spent = ledger.reserve(0.8, task="banking77", phase="curate-label", run_id=first, model=MODEL, run_cap=1.0)
    ledger.settle(spent, 0.8)
    # the other command has spent nothing: its own --max-usd 1 still has room for 0.3
    ledger.reserve(0.3, task="invoices", phase="curate-label", run_id=second, model=MODEL, run_cap=1.0)


# (4) a failed --fill run never charges requests that were never sent ------------------------------
class _Upstream:
    """Answers 400 for one request (right away) and 200 for the others (after a delay); records what arrived."""

    def __init__(self, failing: str) -> None:
        self.failing = failing
        self.received: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        key = request_key(body)
        self.received.append(key)
        if key == self.failing:
            return httpx.Response(400, json={"error": {"message": "bad request"}})
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=completion())


def _check_nothing_unsent_was_charged(ledger: Ledger, received: list[str]) -> None:
    rows = ledger_rows(ledger)
    phantom = [key for key, status in rows if status == "settled" and key not in received]
    assert phantom == []
    assert ledger.open_reservations() == 0


def test_a_closed_client_releases_the_calls_still_waiting_for_a_slot(ledger: Ledger) -> None:
    bodies = [make_body(f"message {i}") for i in range(12)]
    upstream = _Upstream(request_key(bodies[0]))
    teacher = live(ledger, concurrency=3, transport=httpx.MockTransport(upstream))

    async def without_cancelling() -> list[Any]:
        # the old fill_missing: close the client while the other calls still wait for the semaphore
        tasks = [asyncio.ensure_future(teacher.complete(body)) for body in bodies]
        with pytest.raises(TeacherHTTPError):
            await asyncio.gather(*tasks)
        await teacher.aclose()
        return list(await asyncio.gather(*tasks, return_exceptions=True))

    outcomes = asyncio.run(without_cancelling())
    closed = [o for o in outcomes if isinstance(o, TeacherError) and "client was closed" in str(o)]
    assert closed, "some calls should have found the client closed"
    assert all(charged_usd(o) == 0.0 for o in closed)
    _check_nothing_unsent_was_charged(ledger, upstream.received)


def test_fill_missing_stops_the_waiting_requests_when_one_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec()
    bodies = [build_teacher_request(spec, f"message {i}") for i in range(16)]
    upstream = _Upstream(request_key(bodies[0]))
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=5.0)
    teacher = LiveTeacher(
        BASE,
        KEY,
        ledger=ledger,
        pricing=snapshot(),
        task=TASK,
        phase="record-fill",
        run_id="record-1",
        concurrency=3,
        cache=ResponseCache(tmp_path / "cache.sqlite"),
        transport=httpx.MockTransport(upstream),
    )
    outcomes: list[str] = []
    unfinished_at_close: list[int] = []
    complete, aclose = teacher.complete, teacher.aclose

    async def spy_complete(body: dict[str, Any]) -> TeacherResult:
        try:
            result = await complete(body)
        except asyncio.CancelledError:
            outcomes.append("cancelled")
            raise
        except TeacherError as exc:
            outcomes.append(str(exc))
            raise
        outcomes.append("answered")
        return result

    async def spy_aclose() -> None:
        unfinished_at_close.append(len(bodies) - len(outcomes))
        await aclose()

    monkeypatch.setattr(teacher, "complete", spy_complete)
    monkeypatch.setattr(teacher, "aclose", spy_aclose)
    monkeypatch.setattr(factory, "make_teacher", lambda *a, **k: teacher)
    monkeypatch.setattr(record, "load_pricing", snapshot)
    with pytest.raises(TeacherHTTPError):
        record.fill_missing(spec, bodies, yes=True, cache=ResponseCache(tmp_path / "cache.sqlite"), log=lambda _m: None)
    # every request had ended before the client was closed: none could find it closed, the waiting ones were cancelled
    assert unfinished_at_close == [0]
    assert not [o for o in outcomes if "client was closed" in o]
    assert outcomes.count("cancelled") >= len(bodies) - teacher.concurrency
    # the waiting requests were stopped before they reserved: every ledger row belongs to a request that was sent
    assert {key for key, _ in ledger_rows(ledger)} <= set(upstream.received)
    assert len(upstream.received) < len(bodies)
    _check_nothing_unsent_was_charged(ledger, upstream.received)


# (5) `teacher record --task` records the answers of captured traffic ---------------------------------
def _capture(store: Store, body: dict[str, Any], response: dict[str, Any] | None, **kw: Any) -> None:
    row: dict[str, Any] = {
        "task": TASK,
        "source": "proxy",
        "request_key": request_key(body),
        "request_body": json.dumps(body),
        "response_body": None if response is None else json.dumps(response),
        "status": 200,
        "latency_ms": 321.0,
        "upstream_model": MODEL,
    }
    row.update(kw)
    store.add_capture(**row)


def test_captured_answers_are_recorded_when_the_cache_has_none(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    bodies = [build_teacher_request(spec, q) for q in QUERIES]
    store = Store(tmp_path / "store.sqlite")
    for body, label in zip(bodies, ["card_arrival", "lost_card", "card_arrival"], strict=True):
        _capture(store, body, completion(label))
    _capture(store, bodies[1], completion("card_arrival"))  # a later capture of the same request
    keys = default_keys(spec, store)
    assert keys == [request_key(b) for b in bodies]

    out = tmp_path / "rec.jsonl.gz"
    empty = ResponseCache(tmp_path / "cache.sqlite")
    manifest = build_recording(spec, keys=keys, out=out, cache=empty, store=store, pricing_date="2026-09-26")
    assert manifest["records"] == 3
    recording = load_recording(out)
    record_1 = recording.records[request_key(bodies[1])]
    assert record_1["output"] == "lost_card"  # the first capture of a request wins
    assert record_1["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "cost": 2.1e-5}
    assert (record_1["latency_ms"], record_1["provider"], record_1["finish_reason"]) == (321.0, "Alpha", "stop")
    assert isinstance(record_1["timestamp"], float)
    assert all(set(r) == set(RECORD_FIELDS) for r in recording.records.values())
    text = out.read_bytes()
    assert all(q.encode() not in text for q in QUERIES)
    teacher = ReplayTeacher(out, spec)
    assert [asyncio.run(teacher.complete(b)).output for b in bodies] == ["card_arrival", "lost_card", "card_arrival"]

    # the CLI path (no store argument) reads the workspace store
    workspace = Store()
    for body in bodies:
        _capture(workspace, body, completion("lost_card"))
    manifest = build_recording(spec, keys=keys, out=tmp_path / "cli.jsonl.gz", cache=empty, pricing_date="d")
    assert manifest["records"] == 3


def test_the_cache_wins_and_captures_of_other_settings_or_failures_are_not_recorded(tmp_path: Path) -> None:
    spec = make_spec()
    bodies = [build_teacher_request(spec, q) for q in QUERIES]
    store = Store(tmp_path / "store.sqlite")
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cached = TeacherResult(
        key=request_key(bodies[0]),
        output="lost_card",
        response={},
        usage={},
        latency_ms=1.0,
        provider="Alpha",
        finish_reason="stop",
        created=1.0,
        source="live",
    )
    cache.put(cached, request_context(bodies[0]))
    _capture(store, bodies[0], completion("card_arrival"))
    other = make_spec(extra_body={"provider": {"order": ["beta"], "allow_fallbacks": False}})
    _capture(store, build_teacher_request(other, QUERIES[1]), completion("card_arrival"))
    _capture(store, bodies[2], {"error": {"code": 502, "message": "upstream failed"}})

    keys = [request_key(b) for b in bodies]
    with pytest.raises(RecordingError) as info:
        build_recording(spec, keys=keys, out=tmp_path / "rec.jsonl.gz", cache=cache, store=store, pricing_date="d")
    message = str(info.value)
    assert "2 of 3 request keys are not in the response cache or the task's captured traffic" in message
    assert "1 of them are cached or captured under another provider or reasoning setting" in message

    manifest = build_recording(spec, keys=keys[:1], out=tmp_path / "one.jsonl.gz", cache=cache, store=store)
    assert manifest["records"] == 1
    assert load_recording(tmp_path / "one.jsonl.gz").records[keys[0]]["output"] == "lost_card"


def test_a_key_cached_and_captured_under_other_settings_is_counted_once(tmp_path: Path) -> None:
    spec = make_spec()
    other = make_spec(extra_body={"provider": {"order": ["beta"], "allow_fallbacks": False}})
    bodies = [build_teacher_request(other, q) for q in QUERIES]
    store = Store(tmp_path / "store.sqlite")
    cache = ResponseCache(tmp_path / "cache.sqlite")
    for body in bodies:  # e.g. teacher.extra_body changed after a caching teacher answered the proxy's traffic
        _capture(store, body, completion())
        cache.put(
            TeacherResult(
                key=request_key(body),
                output="card_arrival",
                response={},
                usage={},
                latency_ms=1.0,
                provider="Beta",
                finish_reason="stop",
                created=1.0,
                source="live",
            ),
            request_context(body),
        )
    keys = [request_key(b) for b in bodies]  # the spec's requests for these inputs have the same keys
    assert keys == [request_key(build_teacher_request(spec, q)) for q in QUERIES]
    with pytest.raises(RecordingError) as info:
        build_recording(spec, keys=keys, out=tmp_path / "rec.jsonl.gz", cache=cache, store=store, pricing_date="d")
    assert "3 of 3 request keys are not in the response cache" in str(info.value)
    assert "; 3 of them are cached or captured under another provider or reasoning setting" in str(info.value)


def test_default_keys_recompute_the_key_from_the_captured_body(tmp_path: Path) -> None:
    spec = make_spec()
    body = build_teacher_request(spec, QUERIES[0])
    store = Store(tmp_path / "store.sqlite")
    _capture(store, body, completion(), request_key="0" * 64)  # a key stored by an older version
    assert default_keys(spec, store) == [request_key(body)]


# (6) serve's teacher gives up quickly -----------------------------------------------------------
def test_make_teacher_gives_serve_its_own_timeout_and_retry_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", "sk-test")
    spec = make_spec()
    served = factory.make_teacher(spec, mode="live", phase="serve", run_id="serve-1", use_cache=False)
    assert isinstance(served, LiveTeacher)
    assert (served.timeout, served.max_retries, served.retry_after_max) == (10.0, 1, 2.0)
    # one definition of the serve policy: the knobs serve passes are the ones the factory applies for the phase
    assert dict(factory.SERVE_TEACHER_KNOBS) == {"timeout": 10.0, "max_retries": 1, "retry_after_max": 2.0}
    passed = factory.make_teacher(
        spec, mode="live", phase="serve", run_id="serve-2", use_cache=False, **factory.SERVE_TEACHER_KNOBS
    )
    assert isinstance(passed, LiveTeacher)
    assert (passed.timeout, passed.max_retries, passed.retry_after_max) == (10.0, 1, 2.0)
    # a hung teacher is given up (two timeouts plus the capped backoff) inside serve's deadline for the whole call
    from taskdistill.serve.runner import SERVE_TEACHER_DEADLINE_S

    policy_s = served.timeout * (served.max_retries + 1) + served.retry_after_max * served.max_retries
    assert policy_s < SERVE_TEACHER_DEADLINE_S
    batch = factory.make_teacher(spec, mode="live", phase="curate-label", run_id="curate-1")
    assert isinstance(batch, LiveTeacher)
    assert (batch.timeout, batch.max_retries, batch.retry_after_max) == (60.0, 5, 60.0)
    tuned = factory.make_teacher(
        spec, mode="live", phase="serve", run_id="s", use_cache=False, timeout=5.0, max_retries=0, retry_after_max=0.5
    )
    assert isinstance(tuned, LiveTeacher)
    assert (tuned.timeout, tuned.max_retries, tuned.retry_after_max) == (5.0, 0, 0.5)


def test_a_rate_limited_serve_escalation_fails_within_seconds(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", "sk-test")
    teacher = factory.make_teacher(
        make_spec(), mode="live", phase="serve", run_id="serve-1", use_cache=False, pricing=snapshot()
    )
    assert isinstance(teacher, LiveTeacher)
    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(429, headers={"Retry-After": "60"}))
        with pytest.raises(TeacherHTTPError):
            run(teacher, make_body())
    assert route.call_count == 2  # one retry
    assert sleeps == [2.0]  # Retry-After: 60 capped at 2 s


# (7) a failed call says what the ledger charged for it ------------------------------------------
def test_a_timed_out_call_carries_its_charge(ledger: Ledger, sleeps: list[float]) -> None:
    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=httpx.ReadTimeout("read timed out"))
        with pytest.raises(TeacherTimeout) as info:
            run(live(ledger, max_retries=2), make_body())
    assert charged_usd(info.value) == pytest.approx(3 * WORST)
    assert charged_usd(info.value) == pytest.approx(ledger.spent())


def test_a_call_that_cost_nothing_carries_a_zero_charge(ledger: Ledger, sleeps: list[float], tmp_path: Path) -> None:
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(503, text="overloaded"))
        with pytest.raises(TeacherHTTPError) as http_error:
            run(live(ledger, max_retries=1), make_body())
    assert charged_usd(http_error.value) == 0.0
    tight = Ledger(tmp_path / "tight.sqlite", global_cap=1e-6)
    with respx.mock(assert_all_called=False), pytest.raises(BudgetExceeded) as refused:
        run(live(tight), make_body())
    assert charged_usd(refused.value) == 0.0
    with pytest.raises(TeacherError, match="no price") as unpriced:
        run(live(ledger), make_body(model="vendor/unknown"))
    assert charged_usd(unpriced.value) == 0.0
    assert charged_usd(ValueError("not a teacher error")) is None


def test_a_charge_after_a_timeout_is_part_of_the_result_and_of_a_later_failure(
    ledger: Ledger, sleeps: list[float]
) -> None:
    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=[httpx.ReadTimeout("slow"), httpx.Response(200, json=completion())])
        result = run(live(ledger), make_body())
    # the timed-out attempt may have been billed: the result's cost is everything the ledger charged for the call
    assert result.cost_usd == pytest.approx(WORST + 2.1e-5)
    assert result.cost_usd == pytest.approx(ledger.spent())

    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=[httpx.ReadTimeout("slow"), httpx.Response(400, json={"error": {}})])
        with pytest.raises(TeacherHTTPError) as info:
            run(live(ledger), make_body())
    assert charged_usd(info.value) == pytest.approx(WORST)


def test_a_stream_that_never_started_carries_its_charge(ledger: Ledger, sleeps: list[float]) -> None:
    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=httpx.ReadTimeout("read timed out"))
        with pytest.raises(TeacherTimeout, match="did not start the stream") as info:
            collect(live(ledger, max_retries=1), make_body(stream=True))
    assert charged_usd(info.value) == pytest.approx(2 * WORST)
    assert charged_usd(info.value) == pytest.approx(ledger.spent())


def test_a_body_with_nan_is_a_teacher_error_not_a_crash(ledger: Ledger) -> None:
    with respx.mock(assert_all_called=False), pytest.raises(TeacherError, match="not valid JSON") as info:
        run(live(ledger), make_body(temperature=float("nan")))
    assert charged_usd(info.value) == 0.0


# (7) the charge of a call is known however it ends -------------------------------------------------
class _Slow:
    """An upstream that answers after ``delay`` seconds, JSON or an SSE stream; records what arrived."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.received: list[bytes] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.received.append(request.content)
        await asyncio.sleep(self.delay)
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json=completion())


def test_a_call_cut_off_by_a_deadline_reports_its_charge(ledger: Ledger) -> None:
    body = make_body()
    teacher = live(ledger, transport=httpx.MockTransport(_Slow(5.0)))
    key = request_key(body)
    assert teacher.last_charged_usd(key) is None

    async def go() -> None:
        try:
            with anyio.fail_after(0.05):  # serve's deadline around the teacher call
                await teacher.complete(body)
        finally:
            await teacher.aclose()

    with pytest.raises(TimeoutError):
        asyncio.run(go())
    # the request was on the wire: settled at its worst case, and the teacher says so although nothing carried it
    assert ledger.spent() == pytest.approx(WORST)
    assert teacher.last_charged_usd(key) == pytest.approx(ledger.spent())


def test_a_stream_cut_off_by_a_deadline_reports_its_charge(ledger: Ledger) -> None:
    body = make_body(stream=True)
    teacher = live(ledger, transport=httpx.MockTransport(_Slow(5.0)))

    async def go() -> None:
        chunks = teacher.stream(body)
        try:
            with anyio.fail_after(0.05):
                await anext(chunks, b"")
        finally:
            await chunks.aclose()
            await teacher.aclose()

    with pytest.raises(TimeoutError):
        asyncio.run(go())
    assert ledger.spent() == pytest.approx(WORST)
    assert teacher.last_charged_usd(request_key(body)) == pytest.approx(WORST)


def test_a_call_cancelled_while_waiting_for_a_slot_reports_no_charge(ledger: Ledger) -> None:
    upstream = _Slow(0.2)
    teacher = live(ledger, concurrency=1, transport=httpx.MockTransport(upstream))
    first, waiting = make_body("first"), make_body("waiting")

    async def go() -> TeacherResult:
        try:
            running = asyncio.ensure_future(teacher.complete(first))
            await asyncio.sleep(0.02)
            with pytest.raises(TimeoutError), anyio.fail_after(0.05):
                await teacher.complete(waiting)  # the only slot is taken
            return await running
        finally:
            await teacher.aclose()

    result = asyncio.run(go())
    assert len(upstream.received) == 1
    assert teacher.last_charged_usd(request_key(waiting)) == 0.0
    assert teacher.last_charged_usd(request_key(first)) == pytest.approx(result.cost_usd)
    assert ledger.spent() == pytest.approx(result.cost_usd)


def test_the_reported_charge_matches_answers_and_errors(
    ledger: Ledger, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = live(ledger, max_retries=0)
    answered, failed = make_body("hello"), make_body("world")
    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=[httpx.Response(200, json=completion()), httpx.ReadTimeout("slow")])
        result = run(teacher, answered)
        with pytest.raises(TeacherTimeout) as info:
            run(teacher, failed)
    assert teacher.last_charged_usd(request_key(answered)) == pytest.approx(result.cost_usd)
    assert teacher.last_charged_usd(request_key(failed)) == charged_usd(info.value) == pytest.approx(WORST)
    assert ledger.spent() == pytest.approx(result.cost_usd + WORST)

    # only the most recently ended keys are kept
    monkeypatch.setattr(client_mod, "CHARGES_REMEMBERED", 2)
    remembering = live(ledger)
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        bodies = [make_body(f"m{i}") for i in range(3)]
        for body in bodies:
            run(remembering, body)
    assert remembering.last_charged_usd(request_key(bodies[0])) is None
    assert remembering.last_charged_usd(request_key(bodies[2])) == pytest.approx(2.1e-5)


# (1) a lone surrogate reaches the teacher -----------------------------------------------------------
def test_a_lone_surrogate_is_budgeted_and_sent_as_its_json_escape(ledger: Ledger) -> None:
    text = "cut \ud83d"  # a string cut inside an emoji, as JSON.stringify writes it: valid JSON, not UTF-8
    body = make_body(text)
    # 5 UTF-8 bytes of "cut " + one 3-byte surrogate + 16 per message
    assert prompt_upper_bound(body) == 4 + 3 + 16
    sent: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=b"data: [DONE]\n\n")
        return httpx.Response(200, json=completion())

    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=handler)
        result = run(live(ledger), body)
        streamed = collect(live(ledger), make_body(text, stream=True))
    assert result.output == "card_arrival"
    assert streamed == b"data: [DONE]\n\n"
    assert len(sent) == 2
    assert all(b'"cut \\ud83d"' in raw for raw in sent)  # the JSON escape, in valid UTF-8
    assert [json.loads(raw)["messages"][0]["content"] for raw in sent] == [text, text]
    assert json.loads(sent[0]) == body
    # a body with no surrogate is sent byte for byte as httpx itself encodes JSON
    plain = make_body("héllo")
    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        run(live(ledger), plain)
    assert route.calls[0].request.content == httpx.Request("POST", URL, json=plain).content


def test_a_body_json_cannot_carry_is_refused_before_anything_is_reserved(ledger: Ledger) -> None:
    # NaN outside the key fields: the request key is computed, but the body still cannot be sent
    streamed = make_body(stream=True, top_k=float("nan"))
    tools = make_body(tools=[{"type": "function", "function": {"name": "f", "parameters": {"x": float("inf")}}}])
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        with pytest.raises(TeacherError, match="not valid JSON") as stream_info:
            collect(live(ledger), streamed)
        with pytest.raises(TeacherError, match="not valid JSON") as tools_info:
            run(live(ledger), tools)
    assert route.call_count == 0
    assert charged_usd(stream_info.value) == charged_usd(tools_info.value) == 0.0
    assert ledger_rows(ledger) == []
