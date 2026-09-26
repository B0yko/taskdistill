"""Live teacher: an async OpenAI-compatible client with bounded concurrency, retries, a cache and a spend ledger.

Every call reserves its worst-case cost in the ledger before it is sent and settles the reservation with
the real charge afterwards (``usage.cost`` when the API returns it, otherwise usage x snapshot price), so
concurrent calls can never cross a cap. Accounting per attempt:

- an HTTP error response (429, 5xx, other 4xx) costs nothing: the reservation is kept for the retry and
  released when the call gives up;
- an attempt whose outcome is unknown (read timeout, dropped connection) may have been billed: it is
  settled at its worst case and the retry reserves again;
- a connection that was never established (connect error or timeout, proxy failure) costs nothing;
- a request that cannot be sent at all (base URL without an http(s) scheme, invalid header value, a client
  closed while the call waited for a slot) is not retried: the reservation is released and
  :class:`TeacherError` raised at once.

Streams are retried the same way until their first byte is passed on; after that a failure ends the stream.

The worst case must bound what the request sent can cost. A body without ``max_tokens`` (and without
``max_completion_tokens``) is reserved at the model's maximum completion length from the pricing snapshot, and
refused with :class:`~taskdistill.ledger.UnboundedCompletion` when the snapshot does not know it; the body is
sent as given, never with a smaller limit than the one reserved. ``max_tokens_default`` instead adds a limit to
such bodies (and reserves that).

Every :class:`TeacherError` a call raises carries ``charged_usd``: what the ledger charged for the call's
attempts before it failed (0.0 when every reservation was released). A result's ``cost_usd`` likewise covers
every attempt of the call, not only the one that answered. A call cancelled by a caller's deadline raises no
:class:`TeacherError` to carry it, so :meth:`LiveTeacher.last_charged_usd` reports, per request key, what the
latest call that ended was charged, however it ended.

The body is sent as the compact UTF-8 JSON httpx itself would write, except that a lone UTF-16 surrogate (a string
cut inside an emoji; UTF-8 cannot encode it) is sent as its JSON escape ``\\udXXX`` rather than failing to encode.

Ledger and cache calls are synchronous SQLite transactions of a few milliseconds; keeping them off worker
threads means a cancelled call can never leave a reservation half-recorded.

The request carries only ``Authorization: Bearer <teacher key>``; no header of the calling application is
ever forwarded.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import math
import random
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any, Literal

import click
import httpx
import numpy as np

from taskdistill.ledger import Ledger, choice_count, completion_limit, reservation_cost
from taskdistill.teacher.base import TeacherError, TeacherHTTPError, TeacherResult, TeacherTimeout
from taskdistill.teacher.cache import ResponseCache, request_context
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.request_key import request_key

log = logging.getLogger("taskdistill.teacher")

#: Statuses worth retrying besides every 5xx.
RETRY_STATUSES = frozenset({408, 429})
#: Transport failures where the request provably never reached the teacher (retried, nothing charged).
_UNSENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError)
#: Failures where the request could not be sent at all: a configuration error, never retried.
_UNSENDABLE = (httpx.UnsupportedProtocol, httpx.LocalProtocolError, httpx.InvalidURL)
DEFAULT_SPEND_THRESHOLD = 0.50
#: Batch defaults (curate, bake-off, demo labelling): seconds per HTTP timeout, retries, longest Retry-After honoured.
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_AFTER_MAX_S = 60.0
#: Attribute a raised :class:`TeacherError` carries: the USD the ledger charged for the call before it failed.
CHARGED_ATTR = "charged_usd"
#: How many request keys :meth:`LiveTeacher.last_charged_usd` remembers (the least recently ended are dropped).
CHARGES_REMEMBERED = 4096

# Indirections so tests can control time without touching the event loop.
_sleep = asyncio.sleep
_clock = time.perf_counter


class SpendNotConfirmed(click.ClickException):
    """A live batch's projected cost is above the threshold and ``--yes`` was not given."""


def charged_usd(exc: BaseException) -> float | None:
    """What the ledger charged for a failed live call (``exc.charged_usd``); None when the error does not say."""
    value = getattr(exc, CHARGED_ATTR, None)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


class _Charge:
    """The USD settled so far for one call, over all its attempts."""

    __slots__ = ("usd",)

    def __init__(self) -> None:
        self.usd = 0.0


def _with_charge(exc: TeacherError, charge: _Charge) -> TeacherError:
    setattr(exc, CHARGED_ATTR, charge.usd)
    return exc


def model_max_completion_tokens(pricing: PricingSnapshot, model: str, provider: str | None = None) -> int | None:
    """The most completion tokens per choice ``model`` can generate, when the pricing snapshot records it.

    Read through ``pricing.max_completion_tokens(model, provider)``, which answers like ``price_for``: the pinned
    provider's limit, else the largest over the endpoints routing may pick. A snapshot without that lookup, or
    without a limit for ``model``, gives None (and the call is refused rather than reserved at a guess).
    """
    lookup = getattr(pricing, "max_completion_tokens", None)
    if not callable(lookup):
        return None
    try:
        value = lookup(model, provider)
    except KeyError:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def encode_body(body: Mapping[str, Any]) -> bytes:
    """The request bytes: compact JSON, UTF-8, a lone surrogate written as its JSON escape (see the module doc).

    The same bytes httpx writes for ``json=body`` whenever that can be encoded. :class:`ValueError` for NaN or
    an infinity, which JSON cannot carry.
    """
    text = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.encode("utf-8", errors="backslashreplace")


def pinned_provider(body: Mapping[str, Any]) -> str | None:
    """The first entry of ``provider.order`` in a request body, if any."""
    provider = body.get("provider")
    if isinstance(provider, Mapping):
        order = provider.get("order")
        if isinstance(order, list) and order:
            return str(order[0])
    return None


def message_text(content: Any) -> str | None:
    """Assistant message content as text (content parts are joined)."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "") for part in content if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return str(content)


def is_truncated(body: Mapping[str, Any], usage: Mapping[str, Any], finish_reason: str | None) -> bool:
    """True when the output was cut off: ``finish_reason == "length"`` or more completion tokens than allowed.

    The allowance is ``max_tokens`` (else ``max_completion_tokens``) per choice, times ``n``.
    """
    if finish_reason == "length":
        return True
    limit = completion_limit(body)
    completion = usage.get("completion_tokens")
    if limit is None or not isinstance(completion, int) or isinstance(completion, bool):
        return False
    return completion > limit * choice_count(body)


def check_base_url(base_url: str) -> str:
    """The base URL without a trailing slash; :class:`TeacherError` unless it is an http(s) URL with a host."""
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL as exc:
        raise TeacherError(f"teacher base URL {base_url!r} is not a valid URL ({exc})") from exc
    if url.scheme not in ("http", "https") or not url.host:
        raise TeacherError(
            f"teacher base URL {base_url!r} must be an http:// or https:// URL with a host, "
            "e.g. https://openrouter.ai/api/v1"
        )
    return base_url.rstrip("/")


def usage_cost(usage: Mapping[str, Any], price: ModelPrice) -> float | None:
    """``usage.cost`` when present, else token counts x price (+ request fee); None when usage has neither."""
    cost = usage.get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
        return float(cost)
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(prompt, int) or isinstance(completion, int):
        return price.cost(int(prompt or 0), int(completion or 0))
    return None


def retry_after_seconds(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (delta seconds or an HTTP date)."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        seconds = when.timestamp() - time.time()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


class _SSEScanner:
    """Picks ``usage`` and ``provider`` out of an SSE byte stream without altering it."""

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, Any] = {}
        self.provider: str | None = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if isinstance(event, dict):
                if isinstance(event.get("usage"), dict):
                    self.usage = event["usage"]
                if isinstance(event.get("provider"), str):
                    self.provider = event["provider"]


class LiveTeacher:
    """The live teacher behind an OpenAI-compatible ``/chat/completions`` endpoint (mode ``live``).

    ``cache`` is for curate, the bake-off and demo labelling only; ``serve`` constructs it without one.
    """

    mode: Literal["live", "replay"] = "live"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        ledger: Ledger,
        pricing: PricingSnapshot,
        task: str,
        phase: str,
        run_id: str,
        concurrency: int = 8,
        cache: ResponseCache | None = None,
        run_cap: float | None = None,
        task_cap: float | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens_default: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        backoff_base: float = 0.5,
        backoff_max: float = 20.0,
        retry_after_max: float = DEFAULT_RETRY_AFTER_MAX_S,
        rng: random.Random | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.base_url = check_base_url(base_url)
        self._api_key = api_key
        self.ledger = ledger
        self.pricing = pricing
        self.task = task
        self.phase = phase
        self.run_id = run_id
        self.concurrency = concurrency
        self.cache = cache
        self.run_cap = run_cap
        self.task_cap = task_cap
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens_default = max_tokens_default
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.retry_after_max = retry_after_max
        self._transport = transport
        self._rng = rng or random.Random()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._charges: OrderedDict[str, float] = OrderedDict()

    # plumbing ----------------------------------------------------------------------------------
    def _bind(self) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        """HTTP client and semaphore for the running event loop (created lazily, recreated for a new loop)."""
        loop = asyncio.get_running_loop()
        if self._client is None or self._semaphore is None or self._loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                transport=self._transport,
                limits=httpx.Limits(max_connections=self.concurrency, max_keepalive_connections=self.concurrency),
            )
            self._semaphore = asyncio.Semaphore(self.concurrency)
            self._loop = loop
        return self._client, self._semaphore

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _prepare(self, body: Mapping[str, Any], *, stream: bool) -> tuple[dict[str, Any], int | None]:
        """The body to send, and the ``max_tokens`` added to it when it set no completion limit (else None)."""
        out = dict(body)
        if stream:
            out["stream"] = True
        else:
            out.pop("stream", None)
            out.pop("stream_options", None)
        injected: int | None = None
        unbounded = out.get("max_tokens") is None and out.get("max_completion_tokens") is None
        if self.max_tokens_default is not None and unbounded:
            injected = out["max_tokens"] = self.max_tokens_default
        return out, injected

    def _cache_ref(self, body: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
        """Request key and cache context of the caller's body (as capture and replay see it), and the body to send.

        :class:`TeacherError` when the body holds a number JSON cannot carry (NaN, an infinity).
        """
        send, injected = self._prepare(body, stream=False)
        try:
            return request_key(body), request_context(body, max_tokens_default=injected), send
        except ValueError as exc:
            raise TeacherError(f"the teacher request is not valid JSON: {exc}") from exc

    @staticmethod
    def _encode(body: Mapping[str, Any]) -> bytes:
        """:func:`encode_body`; a body JSON cannot carry (NaN, an infinity, a non-JSON value) is a
        :class:`TeacherError`. It runs before anything is reserved, so such a body is never charged."""
        try:
            return encode_body(body)
        except (ValueError, TypeError) as exc:
            raise TeacherError(f"the teacher request is not valid JSON: {exc}") from exc

    def _remember(self, key: str | None, charge: _Charge) -> None:
        if key is None:
            return
        self._charges[key] = charge.usd
        self._charges.move_to_end(key)
        while len(self._charges) > CHARGES_REMEMBERED:
            self._charges.popitem(last=False)

    def last_charged_usd(self, key: str) -> float | None:
        """What the ledger charged, over all attempts, for the latest call of request ``key`` that has ended.

        It covers every way a call ends: an answer (its ``cost_usd``), a :class:`TeacherError` (its
        ``charged_usd``), a cancellation (a caller's deadline; an attempt on the wire is settled at its worst
        case, a call still waiting for a slot costs 0.0) and a stream closed early. None when no call of ``key``
        has ended yet (or it is older than the last :data:`CHARGES_REMEMBERED` keys). When calls of one key
        overlap, the one that ended last is reported.
        """
        return self._charges.get(key)

    def is_cached(self, body: Mapping[str, Any]) -> bool:
        """True when :meth:`complete` would answer ``body`` from the cache, i.e. without a teacher call."""
        if self.cache is None:
            return False
        try:
            key, context, _ = self._cache_ref(body)
        except TeacherError:
            return False
        return self.cache.get(key, context) is not None

    def _price(self, body: Mapping[str, Any]) -> tuple[str, str | None, ModelPrice]:
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise TeacherError("the teacher request has no model")
        provider = pinned_provider(body)
        try:
            return model, provider, self.pricing.price_for(model, provider)
        except KeyError as exc:
            raise TeacherError(str(exc.args[0]) if exc.args else str(exc)) from exc

    def _worst_case(self, body: Mapping[str, Any], model: str, pinned: str | None, price: ModelPrice) -> float:
        """The reservation for sending ``body`` as is (:func:`~taskdistill.ledger.reservation_cost`).

        A body without a completion limit is bounded by the model's maximum completion tokens from the pricing
        snapshot, else refused with :class:`UnboundedCompletion`: it is never sent under a smaller reservation.
        A :class:`TeacherError` also when ``tools``, ``functions`` or a message's tool calls hold NaN or an
        infinity (the request key does not cover those fields).
        """
        limit = None if completion_limit(body) is not None else model_max_completion_tokens(self.pricing, model, pinned)
        try:
            return reservation_cost(body, price, model=model, max_completion_tokens=limit)
        except ValueError as exc:
            raise TeacherError(f"the teacher request is not valid JSON: {exc}") from exc

    def _settle_price(self, model: str, served_by: Any, pinned: str | None, reserved: ModelPrice) -> ModelPrice:
        """Price of the provider that served the call when the snapshot knows it, else the reserved price."""
        for provider in (served_by if isinstance(served_by, str) else None, pinned):
            if provider:
                try:
                    return self.pricing.price_for(model, provider)
                except KeyError:
                    break
        return reserved

    def _reserve(self, amount: float, model: str, key: str) -> int:
        return self.ledger.reserve(
            amount,
            task=self.task,
            phase=self.phase,
            run_id=self.run_id,
            model=model,
            run_cap=self.run_cap,
            task_cap=self.task_cap,
            request_key=key,
        )

    def _settle(self, res_id: int, amount: float, charge: _Charge) -> None:
        self.ledger.settle(res_id, amount)
        charge.usd += amount

    @staticmethod
    def _unsendable(url: str, exc: Exception) -> TeacherError:
        # The exception text is left out: for an invalid header value it may quote the API key.
        return TeacherError(
            f"the teacher request to {url} cannot be sent ({type(exc).__name__}); "
            "check the teacher base URL and API key. Nothing was charged and the call was not retried."
        )

    @staticmethod
    def _closed(url: str) -> TeacherError:
        return TeacherError(
            f"the teacher request to {url} was not sent: the teacher client was closed while the call waited "
            "(the batch it belonged to was stopped). Nothing was charged."
        )

    @staticmethod
    def _closed_client_error(client: httpx.AsyncClient, exc: RuntimeError) -> bool:
        """True for httpx's refusal to send on a closed client: raised before any byte leaves the process."""
        return client.is_closed and "client has been closed" in str(exc)

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        """Retry-After when the server sent one (capped), else exponential backoff with full jitter."""
        hinted = retry_after_seconds(retry_after)
        if hinted is not None:
            return min(hinted, self.retry_after_max)
        return self._rng.uniform(0.0, min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1)))

    # TeacherSource -----------------------------------------------------------------------------
    async def complete(self, body: dict[str, Any]) -> TeacherResult:
        """Answer one chat-completions body: from the cache when configured, else a budgeted live call.

        The request key is computed on ``body`` as given, before ``max_tokens_default`` is applied, so it equals
        the key capture and replay compute for the same body. A :class:`TeacherError` carries ``charged_usd``, and
        :meth:`last_charged_usd` reports the charge however the call ends (also when it is cancelled).
        """
        charge = _Charge()
        key: str | None = None
        try:
            key, context, send = self._cache_ref(body)
            if self.cache is not None:
                hit = self.cache.get(key, context)
                if hit is not None:
                    return hit
            model, pinned, price = self._price(send)
            worst = self._worst_case(send, model, pinned, price)
            content = self._encode(send)
            client, semaphore = self._bind()
            async with semaphore:
                result = await self._call(client, send, content, key, model, pinned, price, worst, charge)
        except TeacherError as exc:
            _with_charge(exc, charge)
            raise
        finally:
            self._remember(key, charge)
        if self.cache is not None:
            self.cache.put(result, context)
        return result

    async def _call(
        self,
        client: httpx.AsyncClient,
        body: dict[str, Any],
        content: bytes,
        key: str,
        model: str,
        pinned: str | None,
        price: ModelPrice,
        worst: float,
        charge: _Charge,
    ) -> TeacherResult:
        url = f"{self.base_url}/chat/completions"
        res_id: int | None = None
        in_flight = False
        attempts = 0
        try:
            while True:
                if client.is_closed:
                    raise self._closed(url)
                if res_id is None:
                    res_id = self._reserve(worst, model, key)
                attempts += 1
                start = _clock()
                in_flight = True
                try:
                    resp = await client.post(url, content=content, headers=self._headers())
                except _UNSENDABLE as exc:
                    in_flight = False
                    self.ledger.release(res_id)
                    res_id = None
                    raise self._unsendable(url, exc) from exc
                except httpx.TransportError as exc:
                    in_flight = False
                    if not isinstance(exc, _UNSENT):
                        self._settle(res_id, worst, charge)
                        res_id = None
                    if attempts > self.max_retries:
                        raise TeacherTimeout(
                            f"teacher {self.base_url} did not answer after {attempts} attempts ({type(exc).__name__})"
                        ) from exc
                    await _sleep(self._backoff(attempts, None))
                    continue
                except RuntimeError as exc:
                    if not self._closed_client_error(client, exc):
                        raise
                    in_flight = False
                    raise self._closed(url) from exc
                latency_ms = (_clock() - start) * 1000.0
                in_flight = False
                status = resp.status_code
                if not 200 <= status < 300:
                    if (status in RETRY_STATUSES or status >= 500) and attempts <= self.max_retries:
                        await _sleep(self._backoff(attempts, resp.headers.get("retry-after")))
                        continue
                    raise TeacherHTTPError(status, resp.text)
                try:
                    data = resp.json()
                except ValueError:
                    data = None
                if not isinstance(data, dict):
                    self._settle(res_id, worst, charge)
                    res_id = None
                    raise TeacherHTTPError(status, resp.text)
                raw_choices = data.get("choices")
                choices: list[Any] = raw_choices if isinstance(raw_choices, list) else []
                choice: dict[str, Any] = choices[0] if choices and isinstance(choices[0], dict) else {}
                raw_usage = data.get("usage")
                usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
                error = data.get("error") or choice.get("error")
                settle_price = self._settle_price(model, data.get("provider"), pinned, price)
                cost = usage_cost(usage, settle_price)
                if error:
                    # An error object in a 2xx body: generation may have started (billed by its usage, or at the
                    # worst case when usage is missing); with neither choices nor usage nothing was generated.
                    if cost is None and not choices:
                        self.ledger.release(res_id)
                    else:
                        self._settle(res_id, worst if cost is None else cost, charge)
                    res_id = None
                    code = error.get("code") if isinstance(error, dict) else None
                    code = code if isinstance(code, int) and 400 <= code < 600 else 502
                    if (code in RETRY_STATUSES or code >= 500) and attempts <= self.max_retries:
                        await _sleep(self._backoff(attempts, resp.headers.get("retry-after")))
                        continue
                    raise TeacherHTTPError(code, json.dumps(error, ensure_ascii=False))
                if cost is None:
                    log.warning("teacher response for %s has no usage; charging its worst case", key[:12])
                    cost = worst
                self._settle(res_id, cost, charge)
                res_id = None
                return self._result(body, key, data, choice, usage, latency_ms, charge.usd, attempts)
        finally:
            if res_id is not None:
                # Cancelled while a request was on the wire: its charge is unknown. Otherwise nothing was billed.
                if in_flight:
                    self._settle(res_id, worst, charge)
                else:
                    self.ledger.release(res_id)

    def _result(
        self,
        body: Mapping[str, Any],
        key: str,
        data: dict[str, Any],
        choice: Mapping[str, Any],
        usage: dict[str, Any],
        latency_ms: float,
        cost: float,
        attempts: int,
    ) -> TeacherResult:
        raw_message = choice.get("message")
        message: Mapping[str, Any] = raw_message if isinstance(raw_message, Mapping) else {}
        finish_reason = choice.get("finish_reason")
        finish_reason = finish_reason if isinstance(finish_reason, str) else None
        provider = data.get("provider") if isinstance(data.get("provider"), str) else None
        truncated = is_truncated(body, usage, finish_reason)
        extra: dict[str, Any] = {"model": data.get("model")}
        details = usage.get("completion_tokens_details")
        reasoning = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
        if isinstance(reasoning, int) and reasoning > 0:
            extra["reasoning_tokens"] = reasoning
            log.warning("teacher response %s used %d reasoning tokens; disable reasoning", key[:12], reasoning)
        if truncated:
            log.warning(
                "teacher output truncated for %s (finish_reason=%s, completion_tokens=%s, max_tokens=%s)",
                key[:12],
                finish_reason,
                usage.get("completion_tokens"),
                body.get("max_tokens"),
            )
        return TeacherResult(
            key=key,
            output=message_text(message.get("content")),
            response=data,
            usage=usage,
            latency_ms=latency_ms,
            provider=provider,
            finish_reason=finish_reason,
            created=time.time(),
            source="live",
            cost_usd=cost,
            attempts=attempts,
            truncated=truncated,
            extra=extra,
        )

    async def stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Pass the teacher's SSE bytes through untouched.

        Until the first byte has been passed on, 408/429/5xx responses and transport failures are retried like
        :meth:`complete` (the reservation is kept across attempts that cost nothing). Once bytes have been
        passed on, a failure ends the stream with :class:`TeacherTimeout`. The reservation is settled with the
        ``usage`` of the final chunk when the stream carried one, also when the consumer stops reading early
        (e.g. at ``[DONE]``), else at its worst case. A :class:`TeacherError` carries ``charged_usd``, and once the
        stream has ended or been closed, :meth:`last_charged_usd` reports what it was charged.
        """
        charge = _Charge()
        key: str | None = None
        try:
            try:
                key = request_key(body)
            except ValueError as exc:
                raise TeacherError(f"the teacher request is not valid JSON: {exc}") from exc
            send, _ = self._prepare(body, stream=True)
            model, pinned, price = self._price(send)
            worst = self._worst_case(send, model, pinned, price)
            content = self._encode(send)
            url = f"{self.base_url}/chat/completions"
            client, semaphore = self._bind()
            async with semaphore:
                res_id: int | None = None
                in_flight = False
                yielded = False
                attempts = 0
                scanner = _SSEScanner()
                try:
                    while True:
                        if client.is_closed:
                            raise self._closed(url)
                        if res_id is None:
                            res_id = self._reserve(worst, model, key)
                        attempts += 1
                        scanner = _SSEScanner()
                        status, text, retry_after = 0, "", None
                        in_flight = True
                        try:
                            async with client.stream("POST", url, content=content, headers=self._headers()) as resp:
                                status = resp.status_code
                                if 200 <= status < 300:
                                    async for chunk in resp.aiter_raw():
                                        scanner.feed(chunk)
                                        yielded = True
                                        yield chunk
                                    break
                                in_flight = False
                                retry_after = resp.headers.get("retry-after")
                                text = (await resp.aread()).decode("utf-8", "replace")
                        except _UNSENDABLE as exc:
                            in_flight = False
                            self.ledger.release(res_id)
                            res_id = None
                            raise self._unsendable(url, exc) from exc
                        except httpx.TransportError as exc:
                            if yielded:
                                raise TeacherTimeout(f"teacher stream broke off ({type(exc).__name__})") from exc
                            if in_flight and not isinstance(exc, _UNSENT):
                                self._settle(res_id, worst, charge)
                                res_id = None
                            in_flight = False
                            if attempts > self.max_retries:
                                raise TeacherTimeout(
                                    f"teacher {self.base_url} did not start the stream after {attempts} attempts "
                                    f"({type(exc).__name__})"
                                ) from exc
                            delay = self._backoff(attempts, None)
                        except RuntimeError as exc:
                            if yielded or status or not self._closed_client_error(client, exc):
                                raise
                            in_flight = False
                            raise self._closed(url) from exc
                        else:
                            if not ((status in RETRY_STATUSES or status >= 500) and attempts <= self.max_retries):
                                raise TeacherHTTPError(status, text)
                            delay = self._backoff(attempts, retry_after)
                        await _sleep(delay)
                finally:
                    if res_id is not None:
                        if in_flight:
                            amount = None
                            if scanner.usage:
                                settle_price = self._settle_price(model, scanner.provider, pinned, price)
                                amount = usage_cost(scanner.usage, settle_price)
                            self._settle(res_id, worst if amount is None else amount, charge)
                        else:
                            self.ledger.release(res_id)
        except TeacherError as exc:
            _with_charge(exc, charge)
            raise
        finally:
            self._remember(key, charge)

    async def aclose(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()
        self._semaphore = None
        self._loop = None

    async def __aenter__(self) -> LiveTeacher:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# batch helpers ---------------------------------------------------------------------------------
def project_cost(sample_results: Iterable[TeacherResult], n_total: int) -> dict[str, Any]:
    """Project the cost of a batch's remaining calls: mean cost of the sample's live calls x ``n_total``.

    ``n_total`` is the number of calls still to make that will reach the teacher, i.e. inputs that are not
    cached (:meth:`LiveTeacher.is_cached`). Cache hits cost nothing and say nothing about uncached inputs, so
    only live results enter the mean (``live_n``); sample uncached inputs to get a projection. Without a live
    result the cost cannot be projected and ``mean_usd``/``projected_usd`` are None, except that the projection
    is $0 when nothing is left to call (``n_total`` 0) or the sample came from a replay, which never spends.
    """
    results = list(sample_results)
    live = [float(r.cost_usd) for r in results if r.source == "live"]
    out: dict[str, Any] = {"sample_n": len(results), "live_n": len(live), "n_total": n_total}
    if live:
        mean = sum(live) / len(live)
        return {**out, "mean_usd": mean, "projected_usd": mean * n_total}
    replay_only = bool(results) and all(r.source == "replay" for r in results)
    projected = 0.0 if n_total == 0 or replay_only else None
    return {**out, "mean_usd": None, "projected_usd": projected}


def confirm_spend(projected_usd: float | None, yes: bool, threshold: float = DEFAULT_SPEND_THRESHOLD) -> None:
    """Refuse a projected spend above ``threshold``, or one that could not be projected, unless ``--yes``."""
    if yes:
        return
    if projected_usd is None:
        raise SpendNotConfirmed(
            "the teacher spend cannot be projected because the sample held no live teacher calls; "
            "re-run with --yes to confirm (and --max-usd to cap it)"
        )
    if projected_usd > threshold:
        raise SpendNotConfirmed(
            f"projected teacher spend ${projected_usd:.2f} is above ${threshold:.2f}; "
            "re-run with --yes to confirm (and --max-usd to cap it)"
        )


def latency_stats(results: Iterable[TeacherResult]) -> dict[str, Any]:
    """p50/p95/mean teacher latency in ms over non-cached results (each the successful attempt only)."""
    values = [float(r.latency_ms) for r in results if r.source != "cache" and r.latency_ms is not None]
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
    }
