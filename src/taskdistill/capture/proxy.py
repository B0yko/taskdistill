"""Capture proxy: an OpenAI-compatible reverse proxy that logs chat completions into the store.

The request body is forwarded byte for byte, with the client's own ``Authorization`` header, to
``{upstream}/chat/completions``. The proxy logs bodies, the request key, token usage, latency, status,
upstream model and cost. It never stores a header. ``stream: true`` requests are passed through untouched
and counted as not captured, and so is a 2xx response that is not a complete chat completion (see
``completion_problem``), such as an upstream error reported with HTTP 200. A logging failure never fails the
client's request.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from taskdistill.capture.importer import UPSTREAM_ERROR_IN_2XX, completion_problem
from taskdistill.store import Store
from taskdistill.teacher.request_key import request_key

log = logging.getLogger(__name__)

TASK_HEADER = "x-taskdistill-task"
TOKEN_HEADER = "x-taskdistill-token"
TOKEN_ENV = "TASKDISTILL_SERVER_TOKEN"
LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: Request headers forwarded upstream (an allowlist: X-Taskdistill-* and everything else stay behind).
FORWARD_REQUEST_HEADERS = (
    "authorization",
    "content-type",
    "content-encoding",
    "accept",
    "http-referer",
    "x-title",
    "openai-organization",
    "openai-project",
)
#: Upstream response headers returned to the client besides content-type (rate limits and retry hints).
RETURN_RESPONSE_HEADERS = ("retry-after", "retry-after-ms", "x-request-id")
RETURN_RESPONSE_PREFIXES = ("x-ratelimit-",)

STREAM_NOT_CAPTURED = "stream: not captured"
DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
_TASK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class UnsafeBindError(RuntimeError):
    """Binding beyond localhost was requested without ``TASKDISTILL_SERVER_TOKEN``."""


def bind_host(host: str) -> str:
    """The host as the socket layer takes it: surrounding whitespace and IPv6 brackets (``[::1]``) removed."""
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def is_local_host(host: str) -> bool:
    """True for the loopback names the proxy may bind to without a token."""
    return bind_host(host).lower() in LOCAL_HOSTS


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def _valid_task(task: str) -> bool:
    return _TASK_NAME.fullmatch(task) is not None


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        return None


def _request_key(parsed: Any) -> str | None:
    if not isinstance(parsed, Mapping):
        return None
    try:
        return request_key(parsed)
    except UnicodeEncodeError:
        return None  # an unpaired surrogate escape; completion_problem reports it


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


@dataclass
class _CaptureRecord:
    """Everything the proxy logs for one request. There is deliberately no field for headers."""

    task: str
    request_key: str | None
    request_body: str
    response_body: str | None
    status: int | None
    latency_ms: float | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    upstream_model: str | None = None
    captured: bool = False
    error: str | None = None


def _record(
    task: str,
    raw: bytes,
    parsed: Any,
    *,
    status: int | None,
    latency_ms: float | None,
    response_raw: bytes | None = None,
    error: str | None = None,
) -> _CaptureRecord:
    rec = _CaptureRecord(
        task=task,
        request_key=_request_key(parsed),
        request_body=raw.decode("utf-8", errors="replace"),
        response_body=None,
        status=status,
        latency_ms=latency_ms,
        error=error,
    )
    if response_raw is None or status is None:
        return rec
    if not 200 <= status < 300:
        # Error bodies are not stored: some providers echo a masked API key in them.
        rec.error = error or f"upstream HTTP {status}"
        return rec
    response_text = response_raw.decode("utf-8", errors="replace")
    response = _parse_json(response_raw)
    if not isinstance(response, Mapping):
        rec.response_body = response_text
        rec.error = error or "response is not a JSON object"
        return rec
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        rec.prompt_tokens = _as_int(usage.get("prompt_tokens"))
        rec.completion_tokens = _as_int(usage.get("completion_tokens"))
        rec.cost_usd = _as_float(usage.get("cost"))
    model = response.get("model")
    rec.upstream_model = model if isinstance(model, str) else None
    problem = completion_problem(parsed, response)
    if problem != UPSTREAM_ERROR_IN_2XX:
        # Upstream errors are not stored even when they arrive with a 2xx status; usage and cost still are.
        rec.response_body = response_text
    rec.error = error or problem
    rec.captured = rec.error is None
    return rec


@dataclass
class _ProxyState:
    store: Store
    upstream: str
    default_task: str | None
    token: str | None
    client: httpx.AsyncClient | None
    owns_client: bool = False

    def http(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
            self.owns_client = True
        return self.client

    async def close(self) -> None:
        if self.owns_client and self.client is not None:
            await self.client.aclose()
            self.client = None
            self.owns_client = False

    def authorised(self, request: Request) -> bool:
        if self.token is None:
            return True
        given = request.headers.get(TOKEN_HEADER)
        if given is None:
            return False
        return hmac.compare_digest(given.encode("utf-8"), self.token.encode("utf-8"))

    def resolve_task(self, request: Request, path_task: str | None) -> str | JSONResponse:
        task = path_task or request.headers.get(TASK_HEADER) or self.default_task
        if not task:
            return _error(
                400,
                "no task selected: use http://<proxy>/t/<task>/v1 as the base URL or send X-Taskdistill-Task",
                "invalid_request_error",
            )
        if not _valid_task(task):
            return _error(
                400, "task must be 1-64 characters of letters, digits, '.', '_' or '-'", "invalid_request_error"
            )
        return task

    def url(self, endpoint: str, request: Request) -> str:
        query = request.url.query
        return f"{self.upstream}/{endpoint}" + (f"?{query}" if query else "")

    async def log(self, task: str, raw: bytes, parsed: Any, **fields: Any) -> None:
        """Log one request; any failure is reported as a warning and never reaches the client."""
        try:
            rec = _record(task, raw, parsed, **fields)
            await run_in_threadpool(
                self.store.add_capture,
                task=rec.task,
                source="proxy",
                request_key=rec.request_key,
                request_body=rec.request_body,
                response_body=rec.response_body,
                status=rec.status,
                latency_ms=rec.latency_ms,
                prompt_tokens=rec.prompt_tokens,
                completion_tokens=rec.completion_tokens,
                cost_usd=rec.cost_usd,
                upstream_model=rec.upstream_model,
                captured=rec.captured,
                error=rec.error,
            )
        except Exception as exc:
            log.warning("capture logging failed for task %s: %s: %s", task, type(exc).__name__, exc)


def _forward_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name in FORWARD_REQUEST_HEADERS:
        value = request.headers.get(name)
        if value is not None:
            headers[name] = value
    return headers


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in upstream.headers.items():
        lname = name.lower()
        if (
            lname == "content-type"
            or lname in RETURN_RESPONSE_HEADERS
            or any(lname.startswith(p) for p in RETURN_RESPONSE_PREFIXES)
        ):
            headers[lname] = value
    return headers


def _upstream_failure(exc: httpx.HTTPError) -> tuple[int, str]:
    if isinstance(exc, httpx.TimeoutException):
        return 504, f"upstream timed out ({type(exc).__name__})"
    return 502, f"upstream unreachable ({type(exc).__name__})"


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def create_proxy_app(
    store: Store,
    upstream_base_url: str,
    *,
    default_task: str | None = None,
    token: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Build the capture proxy.

    ``client`` is used for upstream calls when given (and left open); otherwise one is created at startup
    and closed at shutdown. With ``token`` set, every request must carry it in ``X-Taskdistill-Token``.
    """
    if default_task is not None and not _valid_task(default_task):
        raise ValueError(f"invalid default task name: {default_task!r}")
    if token is not None and not token:
        raise ValueError("token must be a non-empty string or None")
    state = _ProxyState(
        store=store,
        upstream=upstream_base_url.rstrip("/"),
        default_task=default_task,
        token=token,
        client=client,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state.http()
        try:
            yield
        finally:
            await state.close()

    app = FastAPI(title="taskdistill capture proxy", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    async def chat(request: Request, path_task: str | None) -> Response:
        if not state.authorised(request):
            return _error(401, "missing or invalid X-Taskdistill-Token", "authentication_error")
        task = state.resolve_task(request, path_task)
        if isinstance(task, JSONResponse):
            return task
        raw = await request.body()
        parsed = _parse_json(raw)
        stream = isinstance(parsed, Mapping) and bool(parsed.get("stream"))
        http = state.http()
        upstream_request = http.build_request(
            "POST", state.url("chat/completions", request), content=raw, headers=_forward_headers(request)
        )
        start = time.perf_counter()
        try:
            upstream = await http.send(upstream_request, stream=stream)
        except httpx.HTTPError as exc:
            status, message = _upstream_failure(exc)
            await state.log(task, raw, parsed, status=status, latency_ms=_elapsed_ms(start), error=message)
            return _error(status, message, "upstream_error")
        latency_ms = _elapsed_ms(start)
        headers = _response_headers(upstream)

        if stream:
            await state.log(
                task, raw, parsed, status=upstream.status_code, latency_ms=latency_ms, error=STREAM_NOT_CAPTURED
            )

            async def body() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                except httpx.HTTPError as exc:
                    log.warning("upstream stream for task %s ended early: %s", task, type(exc).__name__)
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                body(), status_code=upstream.status_code, headers=headers, background=BackgroundTask(upstream.aclose)
            )

        content = upstream.content
        await state.log(task, raw, parsed, status=upstream.status_code, latency_ms=latency_ms, response_raw=content)
        return Response(content=content, status_code=upstream.status_code, headers=headers)

    async def models(request: Request) -> Response:
        if not state.authorised(request):
            return _error(401, "missing or invalid X-Taskdistill-Token", "authentication_error")
        try:
            upstream = await state.http().get(state.url("models", request), headers=_forward_headers(request))
        except httpx.HTTPError as exc:
            status, message = _upstream_failure(exc)
            return _error(status, message, "upstream_error")
        return Response(content=upstream.content, status_code=upstream.status_code, headers=_response_headers(upstream))

    @app.post("/t/{task}/v1/chat/completions")
    async def chat_for_task(task: str, request: Request) -> Response:
        return await chat(request, task)

    @app.post("/v1/chat/completions")
    async def chat_by_header(request: Request) -> Response:
        return await chat(request, None)

    @app.get("/t/{task}/v1/models")
    async def models_for_task(task: str, request: Request) -> Response:
        return await models(request)

    @app.get("/v1/models")
    async def models_plain(request: Request) -> Response:
        return await models(request)

    return app


def run_proxy(
    host: str,
    port: int,
    *,
    store: Store,
    upstream_base_url: str,
    default_task: str | None = None,
) -> None:
    """Serve the capture proxy with uvicorn.

    Binding beyond localhost needs ``TASKDISTILL_SERVER_TOKEN``; clients then send it in ``X-Taskdistill-Token``.
    An IPv6 host may be given in brackets (``[::1]``).
    """
    host = bind_host(host)
    token: str | None = None
    if not is_local_host(host):
        token = os.environ.get(TOKEN_ENV) or None
        if token is None:
            raise UnsafeBindError(
                f"refusing to bind the capture proxy to {host or '<all interfaces>'!s} without {TOKEN_ENV}; "
                f"set it and send it in the X-Taskdistill-Token header, or bind to 127.0.0.1"
            )
    app = create_proxy_app(store, upstream_base_url, default_task=default_task, token=token)
    uvicorn.run(app, host=host, port=port, log_level="info")
