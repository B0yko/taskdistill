from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import random
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from taskdistill.ledger import Ledger, prompt_upper_bound, worst_case_cost
from taskdistill.teacher import client as client_mod
from taskdistill.teacher.base import BudgetExceeded, TeacherError, TeacherHTTPError, TeacherResult, TeacherTimeout
from taskdistill.teacher.cache import ResponseCache, request_context
from taskdistill.teacher.client import (
    LiveTeacher,
    SpendNotConfirmed,
    confirm_spend,
    latency_stats,
    project_cost,
    retry_after_seconds,
)
from taskdistill.teacher.pricing import ModelPrice, PricingError, PricingSnapshot, refresh
from taskdistill.teacher.request_key import canonical_json, request_key

BASE = "https://teacher.example.com/api/v1"
URL = f"{BASE}/chat/completions"
MODEL = "vendor/model-a"
KEY = "sk-test-teacher"
ALPHA = ModelPrice(prompt=1e-6, completion=2e-6)
BETA = ModelPrice(prompt=3e-6, completion=4e-6)
# body below: prompt bound = len("hello") + 16 = 21 tokens, max_tokens 8 -> 21e-6 + 8 x 2e-6 = 37e-6
WORST = 37e-6


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


def completion(
    content: Any = "card_arrival",
    *,
    usage: dict[str, Any] | None = None,
    provider: str | None = "Alpha",
    finish_reason: str = "stop",
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1790000000,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": usage if usage is not None else {"prompt_tokens": 12, "completion_tokens": 3, "cost": 2.1e-5},
    }
    if provider is not None:
        data["provider"] = provider
    return data


@pytest.fixture
def pricing() -> PricingSnapshot:
    return PricingSnapshot(
        date="2026-09-26",
        source="test",
        models={MODEL: {"default": ModelPrice(5e-7, 1e-6), "providers": {"alpha/fp8": ALPHA, "beta": BETA}}},
    )


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


def make_teacher(ledger: Ledger, pricing: PricingSnapshot, **kw: Any) -> LiveTeacher:
    params: dict[str, Any] = {"task": "demo", "phase": "label", "run_id": "run-1", "rng": random.Random(7)}
    params.update(kw)
    return LiveTeacher(BASE, params.pop("api_key", KEY), ledger=ledger, pricing=pricing, **params)


async def _complete(teacher: LiveTeacher, body: dict[str, Any]) -> TeacherResult:
    try:
        return await teacher.complete(body)
    finally:
        await teacher.aclose()


def run(teacher: LiveTeacher, body: dict[str, Any]) -> TeacherResult:
    return asyncio.run(_complete(teacher, body))


# happy path and settlement -----------------------------------------------------------------------
def test_success_settles_with_usage_cost_and_sends_only_the_teacher_key(
    ledger: Ledger, pricing: PricingSnapshot
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion())

    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=handler)
        result = run(make_teacher(ledger, pricing), make_body())

    assert result.output == "card_arrival"
    assert result.source == "live"
    assert result.provider == "Alpha"
    assert result.finish_reason == "stop"
    assert result.attempts == 1
    assert result.cost_usd == pytest.approx(2.1e-5)
    assert result.key == request_key(make_body())
    assert not result.truncated
    assert ledger.spent() == pytest.approx(2.1e-5)
    assert ledger.open_reservations() == 0

    request = seen[0]
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert not [h for h in request.headers if h.lower().startswith("x-")]
    assert json.loads(request.content) == make_body()


def test_settlement_falls_back_to_usage_times_price(ledger: Ledger, pricing: PricingSnapshot) -> None:
    usage = {"prompt_tokens": 12, "completion_tokens": 3}
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion(usage=usage)))
        result = run(make_teacher(ledger, pricing), make_body())
    # served by "Alpha" -> alpha/fp8: 12 x 1e-6 + 3 x 2e-6 = 18e-6
    assert result.cost_usd == pytest.approx(18e-6, abs=1e-15)
    assert ledger.spent() == pytest.approx(18e-6, abs=1e-12)


def test_settlement_prices_the_provider_that_served_the_call(ledger: Ledger, pricing: PricingSnapshot) -> None:
    usage = {"prompt_tokens": 12, "completion_tokens": 3}
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion(usage=usage, provider="Beta")))
        result = run(make_teacher(ledger, pricing), make_body())
    # 12 x 3e-6 + 3 x 4e-6 = 48e-6
    assert result.cost_usd == pytest.approx(48e-6, abs=1e-15)


def test_missing_usage_is_charged_at_the_worst_case(ledger: Ledger, pricing: PricingSnapshot) -> None:
    payload = completion()
    del payload["usage"]
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=payload))
        result = run(make_teacher(ledger, pricing), make_body())
    assert result.cost_usd == pytest.approx(WORST)
    assert ledger.spent() == pytest.approx(WORST)


def test_content_parts_are_joined(ledger: Ledger, pricing: PricingSnapshot) -> None:
    parts = [{"type": "text", "text": "card_"}, {"type": "text", "text": "arrival"}]
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion(content=parts)))
        result = run(make_teacher(ledger, pricing), make_body())
    assert result.output == "card_arrival"


def test_max_tokens_default_bounds_an_unbounded_request(tmp_path: Path, pricing: PricingSnapshot) -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    body = make_body()
    del body["max_tokens"]
    # the injected limit bounds the reservation: (5 + 16) x 1e-6 + 64 x 2e-6 = 149e-6 fits a 150e-6 cap;
    # without it the worst case would be 21e-6 + 4096 x 2e-6 = 8213e-6
    tight = Ledger(tmp_path / "ledger.sqlite", global_cap=150e-6)
    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=handler)
        result = run(make_teacher(tight, pricing, max_tokens_default=64), body)
        with pytest.raises(BudgetExceeded):
            run(make_teacher(tight, pricing), body)
    assert len(sent) == 1
    assert sent[0]["max_tokens"] == 64
    # the key is the one capture and replay compute on the body as given, not on the body sent
    assert result.key == request_key(body)
    assert result.key != request_key({**body, "max_tokens": 64})
    usage = {"prompt_tokens": 12, "completion_tokens": 64}
    assert client_mod.is_truncated(sent[0], usage, "stop") is False
    assert client_mod.is_truncated(sent[0], {**usage, "completion_tokens": 65}, "stop") is True


def test_max_tokens_default_is_part_of_the_cache_context(
    tmp_path: Path, ledger: Ledger, pricing: PricingSnapshot
) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    body = make_body()
    del body["max_tokens"]
    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        first = run(make_teacher(ledger, pricing, cache=cache, max_tokens_default=64), body)
        again = run(make_teacher(ledger, pricing, cache=cache, max_tokens_default=64), body)
        assert route.call_count == 1
        wider = run(make_teacher(ledger, pricing, cache=cache, max_tokens_default=128), body)
    assert route.call_count == 2
    assert (first.source, again.source, wider.source) == ("live", "cache", "live")
    assert first.key == again.key == wider.key == request_key(body)


def test_stream_flag_is_dropped_for_complete(ledger: Ledger, pricing: PricingSnapshot) -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=handler)
        run(make_teacher(ledger, pricing), make_body(stream=False, stream_options={"include_usage": True}))
    assert "stream" not in sent[0]
    assert "stream_options" not in sent[0]


# retries -----------------------------------------------------------------------------------------
class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_retries_on_429_503_and_timeouts_then_succeeds(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(client_mod, "_clock", clock)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            clock.now += 0.5
            return httpx.Response(429)
        if calls == 2:
            clock.now += 0.3
            return httpx.Response(503)
        if calls == 3:
            clock.now += 5.0
            raise httpx.ConnectTimeout("connect timed out", request=request)
        clock.now += 0.042
        return httpx.Response(200, json=completion())

    with respx.mock() as mock:
        route = mock.post(URL).mock(side_effect=handler)
        result = run(make_teacher(ledger, pricing), make_body())

    assert route.call_count == 4
    assert result.attempts == 4
    assert result.latency_ms == pytest.approx(42.0)  # the successful attempt only
    assert len(sleeps) == 3
    # nothing was billed by the failed attempts: 429/503 responses and a connection never made
    assert ledger.spent() == pytest.approx(2.1e-5)
    assert ledger.summary()["calls"] == 1
    assert ledger.open_reservations() == 0


def test_backoff_is_exponential_with_full_jitter(ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]) -> None:
    with respx.mock() as mock:
        mock.post(URL).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(502),
                httpx.Response(500),
                httpx.Response(200, json=completion()),
            ]
        )
        result = run(make_teacher(ledger, pricing, rng=random.Random(7), backoff_base=0.5), make_body())
    assert result.attempts == 4
    expected_rng = random.Random(7)
    assert sleeps == [expected_rng.uniform(0.0, cap) for cap in (0.5, 1.0, 2.0)]
    assert all(0.0 <= d <= cap for d, cap in zip(sleeps, (0.5, 1.0, 2.0), strict=True))


def test_retry_after_is_honoured_and_capped(ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]) -> None:
    with respx.mock() as mock:
        mock.post(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "3"}),
                httpx.Response(503, headers={"Retry-After": "600"}),
                httpx.Response(200, json=completion()),
            ]
        )
        result = run(make_teacher(ledger, pricing, retry_after_max=60.0), make_body())
    assert result.attempts == 3
    assert sleeps == [3.0, 60.0]


def test_retry_after_parses_http_dates() -> None:
    past = email.utils.format_datetime(datetime(2000, 1, 1, tzinfo=UTC), usegmt=True)
    assert retry_after_seconds(past) == 0.0
    assert retry_after_seconds("2.5") == 2.5
    assert retry_after_seconds("soon") is None
    assert retry_after_seconds(None) is None


def test_exhausted_5xx_retries_raise_and_release(ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]) -> None:
    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(503, text="overloaded"))
        with pytest.raises(TeacherHTTPError) as info:
            run(make_teacher(ledger, pricing, max_retries=2), make_body())
    assert info.value.status == 503
    assert route.call_count == 3
    assert ledger.spent() == 0.0
    assert ledger.open_reservations() == 0
    assert ledger.summary()["released"] == 1


def test_read_timeouts_are_charged_at_the_worst_case(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]
) -> None:
    with respx.mock() as mock:
        route = mock.post(URL).mock(side_effect=httpx.ReadTimeout("read timed out"))
        with pytest.raises(TeacherTimeout, match="3 attempts"):
            run(make_teacher(ledger, pricing, max_retries=2), make_body())
    assert route.call_count == 3
    # each attempt may have been processed and billed: unknown charge = its worst case
    assert ledger.spent() == pytest.approx(3 * WORST)
    assert ledger.summary()["calls"] == 3
    assert ledger.open_reservations() == 0


def test_connection_failures_cost_nothing(ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]) -> None:
    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(TeacherTimeout):
            run(make_teacher(ledger, pricing, max_retries=1), make_body())
    assert ledger.spent() == 0.0
    assert ledger.open_reservations() == 0


def test_proxy_errors_cost_nothing_and_are_retried(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]
) -> None:
    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=[httpx.ProxyError("tunnel refused"), httpx.Response(200, json=completion())])
        result = run(make_teacher(ledger, pricing), make_body())
    assert result.attempts == 2
    assert ledger.spent() == pytest.approx(2.1e-5)
    assert ledger.summary()["calls"] == 1


@pytest.mark.parametrize("base_url", ["teacher.example.com/api/v1", "htps://teacher.example.com/v1", "https:///v1"])
def test_base_url_without_http_scheme_or_host_is_refused_up_front(
    ledger: Ledger, pricing: PricingSnapshot, base_url: str
) -> None:
    with pytest.raises(TeacherError, match="http:// or https://"):
        LiveTeacher(base_url, KEY, ledger=ledger, pricing=pricing, task="demo", phase="label", run_id="run-1")
    assert ledger.summary()["calls"] == 0


@pytest.mark.parametrize(
    "error",
    [
        httpx.UnsupportedProtocol("Request URL has an unsupported protocol"),
        httpx.LocalProtocolError("Illegal header value b'Bearer sk-test-teacher\\n'"),
        httpx.InvalidURL("Invalid non-printable ASCII character in URL"),
    ],
)
def test_unsendable_requests_are_not_retried_and_cost_nothing(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float], error: Exception
) -> None:
    with respx.mock() as mock:
        route = mock.post(URL).mock(side_effect=error)
        with pytest.raises(TeacherError, match="cannot be sent") as info:
            run(make_teacher(ledger, pricing, max_retries=5), make_body())
    assert not isinstance(info.value, TeacherTimeout)
    assert KEY not in str(info.value)
    assert route.call_count == 1
    assert sleeps == []
    assert ledger.spent() == 0.0
    assert ledger.summary()["released"] == 1
    assert ledger.open_reservations() == 0


def test_unsendable_stream_is_not_retried_and_costs_nothing(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]
) -> None:
    with respx.mock() as mock:
        route = mock.post(URL).mock(side_effect=httpx.UnsupportedProtocol("unsupported protocol"))
        with pytest.raises(TeacherError, match="cannot be sent"):
            asyncio.run(_collect(make_teacher(ledger, pricing), make_body()))
    assert route.call_count == 1
    assert sleeps == []
    assert ledger.spent() == 0.0
    assert ledger.open_reservations() == 0


def test_client_errors_are_not_retried_and_release(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]
) -> None:
    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(400, json={"error": {"message": "bad request"}}))
        with pytest.raises(TeacherHTTPError) as info:
            run(make_teacher(ledger, pricing), make_body())
    assert info.value.status == 400
    assert route.call_count == 1
    assert sleeps == []
    assert ledger.spent() == 0.0
    assert ledger.open_reservations() == 0
    assert ledger.summary()["released"] == 1


def test_error_object_in_a_200_body_is_retried(ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]) -> None:
    upstream_error = {"error": {"code": 503, "message": "provider unavailable"}}
    with respx.mock() as mock:
        mock.post(URL).mock(
            side_effect=[httpx.Response(200, json=upstream_error), httpx.Response(200, json=completion())]
        )
        result = run(make_teacher(ledger, pricing), make_body())
    assert result.attempts == 2
    assert ledger.spent() == pytest.approx(2.1e-5)
    assert ledger.summary()["released"] == 1


def test_budget_refusal_happens_before_any_request(tmp_path: Path, pricing: PricingSnapshot) -> None:
    tight = Ledger(tmp_path / "ledger.sqlite", global_cap=1e-5)  # below the 37e-6 worst case
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        with pytest.raises(BudgetExceeded, match="TASKDISTILL_BUDGET_USD"):
            run(make_teacher(tight, pricing), make_body())
    assert route.call_count == 0


def test_run_and_task_caps_are_passed_to_the_ledger(ledger: Ledger, pricing: PricingSnapshot) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        with pytest.raises(BudgetExceeded, match="run cap --max-usd"):
            run(make_teacher(ledger, pricing, run_cap=1e-5), make_body())
        with pytest.raises(BudgetExceeded, match=r"task cap budget\.usd_cap"):
            run(make_teacher(ledger, pricing, task_cap=1e-5), make_body())


def test_unknown_model_price_is_a_teacher_error(ledger: Ledger, pricing: PricingSnapshot) -> None:
    with respx.mock(), pytest.raises(TeacherError, match="pricing refresh"):
        run(make_teacher(ledger, pricing), make_body(model="vendor/unknown"))


# truncation --------------------------------------------------------------------------------------
def test_finish_reason_length_is_flagged(
    ledger: Ledger, pricing: PricingSnapshot, caplog: pytest.LogCaptureFixture
) -> None:
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion(finish_reason="length")))
        with caplog.at_level(logging.WARNING, logger="taskdistill.teacher"):
            result = run(make_teacher(ledger, pricing), make_body())
    assert result.truncated
    assert "truncated" in caplog.text


def test_completion_tokens_above_max_tokens_are_flagged(ledger: Ledger, pricing: PricingSnapshot) -> None:
    usage = {"prompt_tokens": 12, "completion_tokens": 9, "cost": 2e-5}
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion(usage=usage)))
        result = run(make_teacher(ledger, pricing), make_body())
    assert result.truncated


def test_truncation_allows_max_tokens_per_choice() -> None:
    body = {"max_tokens": 8, "n": 3}
    assert not client_mod.is_truncated(body, {"completion_tokens": 24}, "stop")
    assert client_mod.is_truncated(body, {"completion_tokens": 25}, "stop")
    assert client_mod.is_truncated({"max_completion_tokens": 8}, {"completion_tokens": 9}, "stop")
    assert not client_mod.is_truncated({}, {"completion_tokens": 10_000}, "stop")
    assert client_mod.is_truncated({}, {}, "length")


def test_tools_and_n_are_reserved_so_the_settlement_fits(
    tmp_path: Path, pricing: PricingSnapshot, caplog: pytest.LogCaptureFixture
) -> None:
    tools = [
        {"type": "function", "function": {"name": f"f{i}", "description": "d" * 400, "parameters": {"type": "object"}}}
        for i in range(10)
    ]
    body = make_body("hi", n=3, tools=tools, max_tokens=10)
    price = pricing.price_for(MODEL, "alpha/fp8")
    worst = worst_case_cost(body, price)
    assert prompt_upper_bound(body) == 2 + 16 + len(canonical_json(tools))
    # the teacher bills every tool definition as prompt tokens and 3 choices x 10 completion tokens
    usage = {"prompt_tokens": prompt_upper_bound(body), "completion_tokens": 30}
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=worst * 1.5)
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion(usage=usage)))
        with caplog.at_level(logging.WARNING):
            result = run(make_teacher(ledger, pricing), body)
    assert result.cost_usd == pytest.approx(worst, abs=1e-12)
    assert not result.truncated  # 30 completion tokens = 10 per choice
    assert "above its worst case" not in caplog.text
    assert ledger.committed() <= ledger.global_cap


def test_reasoning_tokens_are_reported(
    ledger: Ledger, pricing: PricingSnapshot, caplog: pytest.LogCaptureFixture
) -> None:
    usage = {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "cost": 2e-5,
        "completion_tokens_details": {"reasoning_tokens": 6},
    }
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion(usage=usage)))
        with caplog.at_level(logging.WARNING, logger="taskdistill.teacher"):
            result = run(make_teacher(ledger, pricing), make_body())
    assert result.extra["reasoning_tokens"] == 6
    assert "reasoning" in caplog.text


# cache -------------------------------------------------------------------------------------------
def test_cache_hit_costs_nothing_and_skips_http(tmp_path: Path, ledger: Ledger, pricing: PricingSnapshot) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        first = run(make_teacher(ledger, pricing, cache=cache), make_body())
        # fields outside the key (stream, user, provider routing) do not change it
        second = run(make_teacher(ledger, pricing, cache=cache), make_body(stream=False, user="u-1"))
    assert route.call_count == 1
    assert first.source == "live"
    assert second.source == "cache"
    assert second.cost_usd == 0.0
    assert second.key == first.key == request_key(make_body())
    assert second.output == first.output
    assert second.response == first.response
    assert ledger.summary()["calls"] == 1
    assert ledger.spent() == pytest.approx(2.1e-5)
    assert len(cache) == 1
    assert [r.key for r in cache.iter_results()] == [first.key]


def test_changed_output_settings_outside_the_key_miss_the_cache(
    tmp_path: Path, ledger: Ledger, pricing: PricingSnapshot
) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    plain = make_body()
    no_reasoning = make_body(reasoning={"enabled": False})
    other_provider = make_body(provider={"order": ["beta"], "allow_fallbacks": False})
    assert request_key(plain) == request_key(no_reasoning) == request_key(other_provider)
    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        assert run(make_teacher(ledger, pricing, cache=cache), plain).source == "live"
        assert run(make_teacher(ledger, pricing, cache=cache), no_reasoning).source == "live"
        assert run(make_teacher(ledger, pricing, cache=cache), no_reasoning).source == "cache"
        assert route.call_count == 2
        # the reasoning-off answer replaced the row, so the old setting is a miss again
        assert run(make_teacher(ledger, pricing, cache=cache), plain).source == "live"
        assert run(make_teacher(ledger, pricing, cache=cache), other_provider).source == "live"
    assert route.call_count == 4
    assert len(cache) == 1
    key = request_key(plain)
    assert cache.get(key, request_context(other_provider)) is not None
    assert cache.get(key, request_context(plain)) is None
    assert cache.get(key) is not None  # no context given: any row for the key
    assert [r.key for r in cache.iter_results([key], context=request_context(other_provider))] == [key]
    assert list(cache.iter_results([key], context=request_context(plain))) == []


def test_request_context_ignores_key_fields_stream_and_user() -> None:
    body = make_body()
    assert request_context(body) == request_context(
        {**body, "messages": [], "temperature": 1.0, "stream": True, "stream_options": {}, "user": "u-2", "seed": None}
    )
    assert request_context(body) != request_context({**body, "reasoning": {"enabled": False}})
    assert request_context(body) != request_context(body, max_tokens_default=64)


def test_cache_from_before_the_context_column_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE responses (key TEXT PRIMARY KEY, output TEXT, response TEXT NOT NULL, usage TEXT NOT NULL, "
            "latency_ms REAL, provider TEXT, finish_reason TEXT, truncated INTEGER NOT NULL DEFAULT 0, "
            "created REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("k" * 64, "card_arrival", "{}", "{}", 10.0, "Alpha", "stop", 0, 1790000000.0),
        )
    conn.close()
    cache = ResponseCache(path)
    old = cache.get("k" * 64)
    assert old is not None and old.output == "card_arrival" and old.source == "cache"
    assert cache.get("k" * 64, request_context(make_body())) is None  # unknown context: a miss


def test_is_cached(tmp_path: Path, ledger: Ledger, pricing: PricingSnapshot) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    teacher = make_teacher(ledger, pricing, cache=cache)
    assert not teacher.is_cached(make_body())
    assert not make_teacher(ledger, pricing).is_cached(make_body())
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, json=completion()))
        run(teacher, make_body())
    assert teacher.is_cached(make_body())
    assert teacher.is_cached(make_body(stream=True, user="u-9"))
    assert not teacher.is_cached(make_body("another input"))
    assert not teacher.is_cached(make_body(reasoning={"enabled": False}))


# concurrency -------------------------------------------------------------------------------------
def test_two_hundred_concurrent_calls_never_cross_the_cap(tmp_path: Path) -> None:
    price = ModelPrice(prompt=1e-6, completion=1e-6)
    pricing = PricingSnapshot(date="2026-09-26", source="test", models={MODEL: {"default": price, "providers": {}}})
    bodies = [
        {"model": MODEL, "messages": [{"role": "user", "content": f"q{i:03d}"}], "temperature": 0, "max_tokens": 30}
        for i in range(200)
    ]
    worst = (4 + 16) * 1e-6 + 30 * 1e-6  # 50e-6 per call
    cap = 50.5 * worst
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=cap)
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.001)
        in_flight -= 1
        body = json.loads(request.content)
        usage = {"prompt_tokens": prompt_upper_bound(body), "completion_tokens": body["max_tokens"]}
        return httpx.Response(200, json=completion(usage=usage, finish_reason="length"))

    async def main() -> list[Any]:
        teacher = LiveTeacher(
            BASE, KEY, ledger=ledger, pricing=pricing, task="demo", phase="label", run_id="run-1", concurrency=8
        )
        try:
            return await asyncio.gather(*(teacher.complete(b) for b in bodies), return_exceptions=True)
        finally:
            await teacher.aclose()

    with respx.mock() as mock:
        route = mock.post(URL).mock(side_effect=handler)
        results = asyncio.run(main())

    done = [r for r in results if isinstance(r, TeacherResult)]
    refused = [r for r in results if isinstance(r, BudgetExceeded)]
    assert len(done) == 50
    assert len(refused) == 150
    assert route.call_count == 50
    assert peak <= 8
    assert ledger.spent() <= cap
    assert ledger.spent() == pytest.approx(50 * worst)
    assert ledger.committed() <= cap
    assert ledger.open_reservations() == 0


# streaming ---------------------------------------------------------------------------------------
SSE = (
    b'data: {"id":"gen-1","provider":"Alpha","choices":[{"index":0,"delta":{"content":"card_"}}]}\n\n'
    b'data: {"id":"gen-1","choices":[{"index":0,"delta":{"content":"arrival"}}]}\n\n'
    b'data: {"id":"gen-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":12,"completion_tokens":3,"cost":0.00002}}\n\n'
    b"data: [DONE]\n\n"
)


async def _collect(teacher: LiveTeacher, body: dict[str, Any]) -> bytes:
    try:
        return b"".join([chunk async for chunk in teacher.stream(body)])
    finally:
        await teacher.aclose()


def test_stream_passes_bytes_through_and_settles_from_the_final_usage(ledger: Ledger, pricing: PricingSnapshot) -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=SSE, headers={"content-type": "text/event-stream"})

    with respx.mock() as mock:
        mock.post(URL).mock(side_effect=handler)
        data = asyncio.run(_collect(make_teacher(ledger, pricing), make_body()))
    assert data == SSE
    assert sent[0]["stream"] is True
    assert ledger.spent() == pytest.approx(2e-5)
    assert ledger.open_reservations() == 0


def test_stream_without_usage_is_charged_at_the_worst_case(ledger: Ledger, pricing: PricingSnapshot) -> None:
    sse = b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\ndata: [DONE]\n\n'
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, content=sse))
        assert asyncio.run(_collect(make_teacher(ledger, pricing), make_body())) == sse
    assert ledger.spent() == pytest.approx(WORST)


def test_stream_client_error_is_not_retried_and_releases(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]
) -> None:
    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(400, text="bad request"))
        with pytest.raises(TeacherHTTPError) as info:
            asyncio.run(_collect(make_teacher(ledger, pricing), make_body()))
    assert info.value.status == 400
    assert route.call_count == 1
    assert sleeps == []
    assert ledger.spent() == 0.0
    assert ledger.open_reservations() == 0
    assert ledger.summary()["released"] == 1


def test_stream_exhausted_5xx_retries_raise_and_release(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]
) -> None:
    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(TeacherHTTPError) as info:
            asyncio.run(_collect(make_teacher(ledger, pricing, max_retries=2), make_body()))
    assert info.value.status == 500
    assert route.call_count == 3
    assert len(sleeps) == 2
    assert ledger.spent() == 0.0
    assert ledger.open_reservations() == 0
    assert ledger.summary()["released"] == 1


def test_stream_retries_429_before_the_first_byte(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]
) -> None:
    with respx.mock() as mock:
        route = mock.post(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "2"}),
                httpx.Response(200, content=SSE, headers={"content-type": "text/event-stream"}),
            ]
        )
        data = asyncio.run(_collect(make_teacher(ledger, pricing), make_body()))
    assert data == SSE
    assert route.call_count == 2
    assert sleeps == [2.0]
    # the 429 cost nothing: one reservation kept across both attempts, settled from the final usage
    assert ledger.spent() == pytest.approx(2e-5)
    assert ledger.summary()["calls"] == 1
    assert ledger.summary()["released"] == 0
    assert ledger.open_reservations() == 0


def test_stream_retries_transport_failures_before_the_first_byte(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]
) -> None:
    with respx.mock() as mock:
        route = mock.post(URL).mock(
            side_effect=[
                httpx.ConnectError("refused"),
                httpx.ReadTimeout("read timed out"),
                httpx.Response(200, content=SSE),
            ]
        )
        data = asyncio.run(_collect(make_teacher(ledger, pricing), make_body()))
    assert data == SSE
    assert route.call_count == 3
    assert len(sleeps) == 2
    # the connect error cost nothing; the read timeout may have been billed (worst case); then the real charge
    assert ledger.spent() == pytest.approx(WORST + 2e-5)
    assert ledger.summary()["calls"] == 2
    assert ledger.open_reservations() == 0


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error

    async def __aiter__(self) -> Any:
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


SSE_CHUNKS = [block + b"\n\n" for block in SSE.split(b"\n\n") if block]


def test_stream_closed_at_done_settles_from_the_captured_usage(ledger: Ledger, pricing: PricingSnapshot) -> None:
    async def read_until_done(teacher: LiveTeacher) -> list[bytes]:
        seen: list[bytes] = []
        gen = teacher.stream(make_body())
        try:
            async for chunk in gen:
                seen.append(chunk)
                if b"[DONE]" in chunk:
                    break
        finally:
            await gen.aclose()  # type: ignore[attr-defined]
            await teacher.aclose()
        return seen

    assert len(SSE_CHUNKS) == 4
    with respx.mock() as mock:
        mock.post(URL).mock(return_value=httpx.Response(200, stream=ChunkedStream(SSE_CHUNKS)))
        seen = asyncio.run(read_until_done(make_teacher(ledger, pricing)))
    assert b"".join(seen) == SSE
    assert ledger.spent() == pytest.approx(2e-5)  # usage.cost of the final chunk, not the 37e-6 worst case
    assert ledger.open_reservations() == 0


def test_stream_failure_after_the_first_byte_is_not_retried(
    ledger: Ledger, pricing: PricingSnapshot, sleeps: list[float]
) -> None:
    stream = ChunkedStream(SSE_CHUNKS[:1], error=httpx.ReadError("connection reset"))
    received: list[bytes] = []

    async def consume(teacher: LiveTeacher) -> None:
        try:
            async for chunk in teacher.stream(make_body()):
                received.append(chunk)
        finally:
            await teacher.aclose()

    with respx.mock() as mock:
        route = mock.post(URL).mock(return_value=httpx.Response(200, stream=stream))
        with pytest.raises(TeacherTimeout, match="broke off"):
            asyncio.run(consume(make_teacher(ledger, pricing)))
    assert received == SSE_CHUNKS[:1]
    assert route.call_count == 1
    assert sleeps == []
    assert ledger.spent() == pytest.approx(WORST)  # no usage seen: unknown charge
    assert ledger.open_reservations() == 0


# batch helpers -----------------------------------------------------------------------------------
def _result(cost: float, latency: float | None, source: str = "live") -> TeacherResult:
    return TeacherResult(
        key="k",
        output="x",
        response={},
        usage={},
        latency_ms=latency,
        provider=None,
        finish_reason="stop",
        created=0.0,
        source=source,  # type: ignore[arg-type]
        cost_usd=cost,
    )


def test_project_cost() -> None:
    sample = [_result(0.01, 10), _result(0.02, 10), _result(0.03, 10)]
    projection = project_cost(sample, 1000)
    assert projection["sample_n"] == 3
    assert projection["live_n"] == 3
    assert projection["n_total"] == 1000
    assert projection["mean_usd"] == pytest.approx(0.02)
    assert projection["projected_usd"] == pytest.approx(20.0)


def test_project_cost_ignores_cache_hits() -> None:
    # (0.01 + 0.03) / 2 live calls = 0.02 per call; the free cache hit does not dilute the mean
    sample = [_result(0.01, 10), _result(0.0, 1, source="cache"), _result(0.03, 10)]
    projection = project_cost(sample, 100)
    assert (projection["sample_n"], projection["live_n"]) == (3, 2)
    assert projection["mean_usd"] == pytest.approx(0.02)
    assert projection["projected_usd"] == pytest.approx(2.0)


def test_project_cost_without_live_calls() -> None:
    all_cached = [_result(0.0, 1, source="cache") for _ in range(50)]
    projection = project_cost(all_cached, 400)
    assert projection == {"sample_n": 50, "live_n": 0, "n_total": 400, "mean_usd": None, "projected_usd": None}
    with pytest.raises(SpendNotConfirmed, match="cannot be projected"):
        confirm_spend(projection["projected_usd"], yes=False)
    confirm_spend(projection["projected_usd"], yes=True)
    assert project_cost([], 400)["projected_usd"] is None
    # nothing left to call, or a replay (which never spends): $0
    assert project_cost(all_cached, 0)["projected_usd"] == 0.0
    assert project_cost([], 0)["projected_usd"] == 0.0
    replayed = [_result(1.2e-5, 400, source="replay") for _ in range(5)]
    assert project_cost(replayed, 400) == {
        "sample_n": 5,
        "live_n": 0,
        "n_total": 400,
        "mean_usd": None,
        "projected_usd": 0.0,
    }


def test_confirm_spend() -> None:
    confirm_spend(0.49, yes=False)
    confirm_spend(0.50, yes=False)
    confirm_spend(12.0, yes=True)
    confirm_spend(None, yes=True)
    with pytest.raises(SpendNotConfirmed, match="--yes") as info:
        confirm_spend(0.51, yes=False)
    assert info.value.exit_code == 1
    assert "$0.51" in info.value.format_message()
    with pytest.raises(SpendNotConfirmed):
        confirm_spend(0.2, yes=False, threshold=0.1)


def test_latency_stats_exclude_cache_hits() -> None:
    results = [_result(0, 10), _result(0, 20), _result(0, 30), _result(0, 40), _result(0, 1000, source="cache")]
    stats = latency_stats(results)
    # numpy linear percentiles of [10, 20, 30, 40]: p50 = 25, p95 = 30 + 0.85 x 10 = 38.5
    assert stats == {"n": 4, "p50_ms": 25.0, "p95_ms": pytest.approx(38.5), "mean_ms": 25.0}
    assert latency_stats([_result(0, 5, source="cache")]) == {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}


# pricing -----------------------------------------------------------------------------------------
MODELS_JSON = {
    "data": [
        {
            "id": "vendor/model-a",
            "canonical_slug": "vendor/model-a-20260731",
            "pricing": {"prompt": "0.00000006", "completion": "0.00000018"},
        },
        {"id": "vendor/model-b", "pricing": {"prompt": "0.0000001", "completion": "0.0000004", "request": "0.0005"}},
        {"id": "router/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
}
ENDPOINTS_A = {
    "data": {
        "id": "vendor/model-a",
        "endpoints": [
            {
                "provider_name": "Alpha",
                "tag": "alpha/fp8",
                "pricing": {"prompt": "0.00000006", "completion": "0.00000018", "discount": 0},
            },
            {
                "provider_name": "Alpha",
                "tag": "alpha/bf16",
                "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            },
            {
                "provider_name": "Gamma",
                "tag": "gamma",
                "pricing": {"prompt": "0.00000014", "completion": "0.00000028", "request": "0.0001"},
            },
        ],
    }
}


@pytest.fixture
def snapshot() -> PricingSnapshot:
    return PricingSnapshot.from_openrouter(MODELS_JSON, {"vendor/model-a": ENDPOINTS_A}, "2026-09-26")


def test_pricing_parses_usd_per_token_strings(snapshot: PricingSnapshot) -> None:
    assert snapshot.date == "2026-09-26"
    assert sorted(snapshot.models) == ["vendor/model-a", "vendor/model-b"]  # the dynamic "-1" router is skipped
    assert snapshot.models["vendor/model-a"]["default"] == ModelPrice(6e-8, 1.8e-7, 0.0)
    assert snapshot.price_for("vendor/model-b") == ModelPrice(1e-7, 4e-7, 5e-4)
    assert snapshot.price_for("vendor/model-b", "alpha") == ModelPrice(1e-7, 4e-7, 5e-4)


def test_pricing_pinned_provider_and_per_provider_max(snapshot: PricingSnapshot) -> None:
    model = "vendor/model-a"
    assert snapshot.price_for(model, "alpha/fp8") == ModelPrice(6e-8, 1.8e-7, 0.0)
    # a base slug or display name covering two endpoints -> their field-wise maximum
    assert snapshot.price_for(model, "alpha") == ModelPrice(1e-7, 2e-7, 0.0)
    assert snapshot.price_for(model, "Alpha") == ModelPrice(1e-7, 2e-7, 0.0)
    assert snapshot.price_for(model, "Gamma") == ModelPrice(1.4e-7, 2.8e-7, 1e-4)
    # unpinned or unknown provider -> the maximum over every known provider
    assert snapshot.price_for(model) == ModelPrice(1.4e-7, 2.8e-7, 1e-4)
    assert snapshot.price_for(model, "delta/fp8") == ModelPrice(1.4e-7, 2.8e-7, 1e-4)


def test_pricing_unknown_model_is_a_clear_key_error(snapshot: PricingSnapshot) -> None:
    with pytest.raises(KeyError, match=r"vendor/nope.*2026-09-26.*pricing refresh"):
        snapshot.price_for("vendor/nope")
    with pytest.raises(KeyError):
        snapshot.price_for("router/auto")


def test_pricing_save_and_load_round_trip(snapshot: PricingSnapshot, tmp_path: Path) -> None:
    path = snapshot.save(tmp_path / "pricing.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["date"] == "2026-09-26"
    assert data["models"]["vendor/model-a"]["providers"]["gamma"] == {
        "prompt": 1.4e-7,
        "completion": 2.8e-7,
        "request": 1e-4,
    }
    assert PricingSnapshot.load(path) == snapshot
    with pytest.raises(PricingError, match="pricing refresh"):
        PricingSnapshot.load(tmp_path / "missing.json")


def test_pricing_refresh_fetches_models_and_endpoints(snapshot: PricingSnapshot) -> None:
    base = "https://router.example.com/api/v1"
    with respx.mock() as mock:
        models = mock.get(f"{base}/models").mock(return_value=httpx.Response(200, json=MODELS_JSON))
        endpoints = mock.get(f"{base}/models/vendor/model-a/endpoints").mock(
            return_value=httpx.Response(200, json=ENDPOINTS_A)
        )
        fresh = refresh(base, api_key="sk-test-pricing", models=["vendor/model-a"], date="2026-09-26")
    assert models.calls.last.request.headers["authorization"] == "Bearer sk-test-pricing"
    assert endpoints.call_count == 1
    assert fresh.models == snapshot.models
    assert fresh.source == f"{base}/models"


def test_pricing_refresh_reports_a_missing_model() -> None:
    base = "https://router.example.com/api/v1"
    with respx.mock() as mock:
        mock.get(f"{base}/models").mock(return_value=httpx.Response(200, json=MODELS_JSON))
        mock.get(f"{base}/models/vendor/gone/endpoints").mock(return_value=httpx.Response(404))
        with pytest.raises(PricingError, match="vendor/gone"):
            refresh(base, models=["vendor/gone"])


def test_pricing_accepts_the_canonical_slug(snapshot: PricingSnapshot, tmp_path: Path) -> None:
    dated = "vendor/model-a-20260731"
    assert snapshot.aliases == {dated: "vendor/model-a"}
    assert dated not in snapshot.models
    assert snapshot.price_for(dated, "alpha/fp8") == ModelPrice(6e-8, 1.8e-7, 0.0)
    assert snapshot.price_for(dated) == ModelPrice(1.4e-7, 2.8e-7, 1e-4)  # max over the id's providers
    assert PricingSnapshot.load(snapshot.save(tmp_path / "pricing.json")).aliases == snapshot.aliases
    # endpoints fetched under the canonical slug land on the id's entry
    by_canonical = PricingSnapshot.from_openrouter(MODELS_JSON, {dated: ENDPOINTS_A}, "2026-09-26")
    assert by_canonical.models == snapshot.models


def test_pricing_refresh_accepts_a_canonical_slug(snapshot: PricingSnapshot) -> None:
    base = "https://router.example.com/api/v1"
    with respx.mock() as mock:
        mock.get(f"{base}/models").mock(return_value=httpx.Response(200, json=MODELS_JSON))
        mock.get(f"{base}/models/vendor/model-a-20260731/endpoints").mock(
            return_value=httpx.Response(200, json=ENDPOINTS_A)
        )
        fresh = refresh(base, models=["vendor/model-a-20260731"], date="2026-09-26")
    assert fresh.models == snapshot.models


def test_pricing_refresh_turns_transport_and_json_failures_into_pricing_errors() -> None:
    base = "https://router.example.com/api/v1"
    with respx.mock() as mock:
        mock.get(f"{base}/models").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(PricingError, match=r"GET https://router\.example\.com/api/v1/models failed: ConnectError"):
            refresh(base)
    with respx.mock() as mock:
        mock.get(f"{base}/models").mock(return_value=httpx.Response(200, text="<html>maintenance</html>"))
        with pytest.raises(PricingError, match="did not return JSON"):
            refresh(base)
    with respx.mock() as mock:
        mock.get(f"{base}/models").mock(return_value=httpx.Response(200, json=MODELS_JSON))
        mock.get(f"{base}/models/vendor/model-a/endpoints").mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(PricingError, match="endpoints failed: ReadTimeout"):
            refresh(base, models=["vendor/model-a"])
    with respx.mock() as mock:
        mock.get(f"{base}/models").mock(return_value=httpx.Response(503))
        with pytest.raises(PricingError, match="HTTP 503"):
            refresh(base)


def test_pinned_provider_without_endpoint_prices_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    snap = PricingSnapshot.from_openrouter(MODELS_JSON, None, "2026-09-26")
    with caplog.at_level(logging.WARNING, logger="taskdistill.pricing"):
        assert snap.price_for("vendor/model-b", "zeta/fp8") == ModelPrice(1e-7, 4e-7, 5e-4)
        assert snap.price_for("vendor/model-b", "zeta/fp8") == ModelPrice(1e-7, 4e-7, 5e-4)
    assert caplog.text.count("no per-provider prices") == 1
