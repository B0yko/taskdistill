"""The cascade server: an OpenAI-compatible endpoint in front of the student and the teacher.

``POST /v1/chat/completions`` accepts any ``model`` value, so an application changes only its base URL:

1. With a token configured, ``Authorization: Bearer <token>`` is required (401 otherwise; not logged).
2. Requests using tools or functions, ``n > 1`` or ``logprobs`` go to the teacher (reason ``unsupported``),
   and so do requests whose input does not match ``input.from`` (reason ``input_unparsed``).
3. Otherwise the student answers on the model worker thread. At or above the threshold it is the answer
   (route ``student``); below it the request escalates (reason ``low_confidence``). ``math.inf`` escalates
   everything and ``0`` never escalates on confidence.
4. An escalation sends the original body, with only ``model`` replaced by ``teacher.model``, through the
   teacher source, which holds the teacher key; the client's headers are never forwarded. With
   ``cascade.escalation_response: canonical`` a ``low_confidence`` escalation is normalised like curate's
   training data and rendered like a student answer (returned raw and counted when it cannot be normalised);
   ``raw`` returns the teacher's response verbatim, and passes its SSE stream through for ``stream: true``.
   ``unsupported`` and ``input_unparsed`` escalations are always raw: the first ask for what a canonical
   answer would drop (tool calls, several choices, log-probabilities), and the second may not be task traffic.
5. A failed teacher call returns the student's answer with route ``student-fallback`` when
   ``cascade.on_teacher_error`` is ``student`` and a student answer exists, else a 502. A replay miss is always
   a 502: the recording does not hold the request, and replay never falls back. A passed-through stream that
   breaks off ends with an SSE ``error`` event and is logged with status 502.

Every response carries ``x-taskdistill-route`` (``student``, ``teacher``, ``student-fallback``, or ``error``),
``x-taskdistill-confidence`` whenever the student ran, and ``x-taskdistill-reason`` and ``x-taskdistill-teacher``
(``live`` or ``replay``) whenever the teacher was involved. Every authenticated request is logged to the store
(401s are only counted in ``/metrics``, so unauthenticated clients cannot grow the store); a logging failure
never fails the request. ``student_ms`` is the generation time, without the wait for the model worker; a
replayed escalation is logged at a teacher cost of 0, since nothing was spent.
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from taskdistill.config import TaskSpec
from taskdistill.curate.extract import InputUnparsed, extract_input
from taskdistill.curate.normalise import normalise_output
from taskdistill.predict import Prediction
from taskdistill.serve.metrics import ServeMetrics
from taskdistill.serve.worker import ModelNotReady, ModelWorker
from taskdistill.store import Store
from taskdistill.teacher.base import (
    BudgetExceeded,
    ReplayMiss,
    TeacherError,
    TeacherHTTPError,
    TeacherResult,
    TeacherSource,
    TeacherTimeout,
)
from taskdistill.teacher.request_key import request_key

log = logging.getLogger("taskdistill.serve")

ROUTE_HEADER = "x-taskdistill-route"
CONFIDENCE_HEADER = "x-taskdistill-confidence"
REASON_HEADER = "x-taskdistill-reason"
TEACHER_HEADER = "x-taskdistill-teacher"
TEACHER_ERROR_HEADER = "x-taskdistill-teacher-error"
#: Body fields that ask for something the student cannot do (function calling).
TOOL_FIELDS = ("tools", "functions", "tool_choice", "function_call")
SSE_HEADERS = {"cache-control": "no-cache"}

Route = Literal["student", "teacher", "student-fallback", "error"]
Reason = Literal["low_confidence", "unsupported", "input_unparsed"]
#: Exceptions a teacher call may end with that the cascade handles (anything else is a server error).
TEACHER_FAILURES: tuple[type[BaseException], ...] = (TeacherError, httpx.HTTPError, TimeoutError)


class ModelLoadError(RuntimeError):
    """The student model could not be loaded, so the server cannot start."""


def unsupported_feature(body: Mapping[str, Any]) -> str | None:
    """The name of the first body field the student cannot honour, or None."""
    for name in TOOL_FIELDS:
        value = body.get(name)
        if value is not None and value != []:
            return name
    n = body.get("n")
    if n is not None and n != 1:
        return "n"
    if body.get("logprobs") or body.get("top_logprobs") is not None:
        return "logprobs"
    return None


def render(spec: TaskSpec, answer: str) -> str:
    """The response content for a canonical answer: ``response_template`` with ``{label}`` replaced literally."""
    template = spec.student.response_template
    return template.replace("{label}", answer) if template else answer


def teacher_error_kind(exc: BaseException) -> str:
    """A short label for a failed teacher call (the ``kind`` of ``taskdistill_teacher_errors_total``)."""
    if isinstance(exc, ReplayMiss):
        return "replay_miss"
    if isinstance(exc, BudgetExceeded):
        return "budget"
    if isinstance(exc, TeacherTimeout | httpx.TimeoutException | TimeoutError):
        return "timeout"
    if isinstance(exc, TeacherHTTPError):
        return "http"
    if isinstance(exc, httpx.HTTPError):
        return "transport"
    return "error"


def _teacher_error_message(exc: BaseException, kind: str, key: str | None = None) -> str:
    # Teacher error bodies are left out: some providers echo a masked API key in them.
    if kind == "replay_miss":
        return (
            f"the request (key {key or 'unknown'}) is not in the teacher recording; replay never falls back to "
            "the student or to a live call. Restart serve with a teacher API key and without --replay for live "
            "escalations."
        )
    if kind == "budget":
        return "the teacher call was refused: it would cross a spend cap (see `taskdistill budget`)"
    if kind == "timeout":
        return "the teacher did not answer in time"
    if isinstance(exc, TeacherHTTPError):
        return f"the teacher returned HTTP {exc.status}"
    return f"the teacher call failed ({type(exc).__name__})"


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _cost(usage: Mapping[str, Any]) -> float | None:
    cost = usage.get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool) and math.isfinite(cost):
        return float(cost)
    return None


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _error_body(status: int, message: str, kind: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "code": status}}


def raw_escalation(spec: TaskSpec, reason: Reason) -> bool:
    """True when an escalation for ``reason`` returns the teacher's response (or SSE stream) verbatim.

    Only ``low_confidence`` escalations are task inputs the student could have answered, so only they are
    rendered canonically under ``escalation_response: canonical``.
    """
    return spec.cascade.escalation_response == "raw" or reason != "low_confidence"


@dataclass
class _Served:
    """What one request did; turned into response headers, metrics and a ``served`` row."""

    started: float
    request_model: str | None = None
    route: Route = "error"
    reason: Reason | None = None
    confidence: float | None = None
    student_ms: float | None = None
    teacher_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    teacher_cost_usd: float | None = None
    teacher_mode: str | None = None
    teacher_error: str | None = None
    status: int = 200
    deferred: bool = False
    finished: bool = False

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {ROUTE_HEADER: self.route}
        if self.confidence is not None:
            headers[CONFIDENCE_HEADER] = f"{self.confidence:.6f}"
        if self.reason is not None:
            headers[REASON_HEADER] = self.reason
        if self.teacher_mode is not None:
            headers[TEACHER_HEADER] = self.teacher_mode
        if self.teacher_error is not None:
            headers[TEACHER_ERROR_HEADER] = self.teacher_error
        return headers


class _SSEUsage:
    """Picks ``usage`` out of an SSE byte stream without altering it."""

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, Any] = {}

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
            if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                self.usage = event["usage"]


class _RelayResponse(StreamingResponse):
    """A pass-through SSE response whose cleanup always runs, also when the client goes away mid-stream."""

    def __init__(
        self, content: AsyncIterator[bytes], cleanup: Callable[[], Awaitable[None]], headers: Mapping[str, str]
    ) -> None:
        super().__init__(content, media_type="text/event-stream", headers=dict(headers))
        self._cleanup = cleanup

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._cleanup()


class _Cascade:
    """Request handling for one app; the endpoint functions below are thin wrappers around it."""

    def __init__(
        self,
        spec: TaskSpec,
        worker: ModelWorker,
        teacher: TeacherSource,
        threshold: float,
        store: Store,
        token: str | None,
        run_id: str | None,
    ) -> None:
        self.spec = spec
        self.worker = worker
        self.teacher = teacher
        self.threshold = threshold
        self.store = store
        self.token = token
        self.run_id = run_id
        self.metrics = ServeMetrics()
        self.model_id = f"taskdistill/{spec.task}"
        #: Replayed escalations cost nothing, so they are logged at a teacher cost of 0 (not the recorded cost).
        self.replaying = teacher.mode == "replay"

    # auth ---------------------------------------------------------------------------------------
    def authorised(self, request: Request) -> bool:
        if self.token is None:
            return True
        scheme, _, given = (request.headers.get("authorization") or "").partition(" ")
        if scheme.lower() != "bearer" or not given.strip():
            return False
        return hmac.compare_digest(given.strip().encode("utf-8"), self.token.encode("utf-8"))

    def unauthorised(self, *, chat: bool = False) -> JSONResponse:
        headers = {"www-authenticate": "Bearer"}
        if chat:
            headers[ROUTE_HEADER] = "error"
        body = _error_body(401, "missing or invalid bearer token", "authentication_error")
        return JSONResponse(body, status_code=401, headers=headers)

    # responses ----------------------------------------------------------------------------------
    def error(self, served: _Served, status: int, message: str, kind: str) -> JSONResponse:
        served.route = "error"
        served.status = status
        return JSONResponse(_error_body(status, message, kind), status_code=status, headers=served.headers())

    def answer(self, served: _Served, body: Mapping[str, Any], content: str, stream: bool) -> Response:
        """A ``chat.completion`` (or one SSE chunk and ``[DONE]``) carrying ``content``."""
        prompt = served.prompt_tokens or 0
        completion = served.completion_tokens or 0
        usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
        base = {
            "id": f"chatcmpl-td-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if stream else "chat.completion",
            "created": int(time.time()),
            "model": served.request_model or self.model_id,
        }
        if not stream:
            message = {"role": "assistant", "content": content}
            data = {**base, "choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": usage}
            return JSONResponse(data, headers=served.headers())
        delta = {"role": "assistant", "content": content}
        chunk: dict[str, Any] = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}]}
        options = body.get("stream_options")
        if isinstance(options, Mapping) and options.get("include_usage"):
            chunk["usage"] = usage
        payload = b"data: " + _json_bytes(chunk) + b"\n\ndata: [DONE]\n\n"
        return Response(payload, media_type="text/event-stream", headers={**served.headers(), **SSE_HEADERS})

    def student_content(self, prediction: Prediction) -> str:
        if prediction.answer is None:
            return prediction.raw_text  # an invalid extraction served at threshold 0: returned as generated
        return render(self.spec, prediction.answer)

    # the request --------------------------------------------------------------------------------
    async def handle(self, request: Request, served: _Served) -> Response:
        raw = await request.body()
        try:
            body: Any = json.loads(raw)
        except (ValueError, RecursionError):
            return self.error(served, 400, "the request body is not valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return self.error(served, 400, "the request body must be a JSON object", "invalid_request_error")
        model = body.get("model")
        served.request_model = model if isinstance(model, str) else None
        stream = bool(body.get("stream"))

        prediction: Prediction | None = None
        reason: Reason
        if unsupported_feature(body) is not None:
            reason = "unsupported"
        else:
            try:
                input_text = extract_input(body, self.spec.input)
            except InputUnparsed:
                reason = "input_unparsed"
            else:
                started = time.perf_counter()
                try:
                    prediction = await self.worker.run(input_text)
                except ModelNotReady as exc:
                    return self.error(served, 503, str(exc), "model_not_ready")
                waited = time.perf_counter() - started
                generation = max(prediction.latency_ms, 0.0) / 1000.0
                served.student_ms = generation * 1000.0
                served.confidence = prediction.confidence
                self.metrics.student_latency.observe(generation)
                self.metrics.student_queue.observe(max(waited - generation, 0.0))
                if prediction.confidence >= self.threshold:
                    served.route = "student"
                    served.prompt_tokens = prediction.prompt_tokens
                    served.completion_tokens = prediction.completion_tokens
                    return self.answer(served, body, self.student_content(prediction), stream)
                reason = "low_confidence"
        return await self.escalate(body, served, reason, prediction, stream)

    async def escalate(
        self, body: dict[str, Any], served: _Served, reason: Reason, prediction: Prediction | None, stream: bool
    ) -> Response:
        served.reason = reason
        served.teacher_mode = self.teacher.mode
        self.metrics.escalations.labels(reason=reason).inc()
        teacher_body = dict(body)
        teacher_body["model"] = self.spec.teacher.model
        raw_mode = raw_escalation(self.spec, reason)
        started = time.perf_counter()
        try:
            if stream and raw_mode:
                chunks = self.teacher.stream(teacher_body)
                first = await anext(chunks, b"")
                served.route = "teacher"
                served.deferred = True
                return self.relay(first, chunks, served, started)
            result = await self.teacher.complete(teacher_body)
        except TEACHER_FAILURES as exc:
            served.teacher_ms = (time.perf_counter() - started) * 1000.0
            key = request_key(teacher_body) if isinstance(exc, ReplayMiss) else None
            return self.teacher_failed(served, exc, body, prediction, stream, key=key)
        elapsed = time.perf_counter() - started
        served.teacher_ms = elapsed * 1000.0
        self.metrics.teacher_latency.observe(elapsed)
        return self.teacher_answer(served, body, result, stream, raw_mode)

    def teacher_answer(
        self, served: _Served, body: Mapping[str, Any], result: TeacherResult, stream: bool, raw_mode: bool
    ) -> Response:
        served.route = "teacher"
        served.prompt_tokens = _int(result.usage.get("prompt_tokens"))
        served.completion_tokens = _int(result.usage.get("completion_tokens"))
        served.teacher_cost_usd = 0.0 if self.replaying or result.source == "replay" else float(result.cost_usd)
        if raw_mode:
            return JSONResponse(result.response, headers=served.headers())
        canonical: str | None = None
        if isinstance(result.output, str):
            _, canonical = normalise_output(self.spec, result.output)
        if canonical is not None:
            return self.answer(served, body, render(self.spec, canonical), stream)
        self.metrics.unnormalised.inc()
        log.warning("teacher output could not be normalised; returned raw")
        if stream:
            return self.answer(served, body, result.output or "", stream)
        return JSONResponse(result.response, headers=served.headers())

    def teacher_failed(
        self,
        served: _Served,
        exc: BaseException,
        body: Mapping[str, Any],
        prediction: Prediction | None,
        stream: bool,
        *,
        key: str | None = None,
    ) -> Response:
        kind = teacher_error_kind(exc)
        served.teacher_error = kind
        self.metrics.teacher_errors.labels(kind=kind).inc()
        log.warning("teacher call failed (%s): %s: %s", kind, type(exc).__name__, str(exc)[:300])
        if (
            prediction is not None
            and prediction.answer is not None
            and kind != "replay_miss"
            and self.spec.cascade.on_teacher_error == "student"
        ):
            served.route = "student-fallback"
            served.prompt_tokens = prediction.prompt_tokens
            served.completion_tokens = prediction.completion_tokens
            return self.answer(served, body, render(self.spec, prediction.answer), stream)
        kind_name = "replay_miss" if kind == "replay_miss" else "teacher_error"
        return self.error(served, 502, _teacher_error_message(exc, kind, key), kind_name)

    def relay(self, first: bytes, rest: AsyncIterator[bytes], served: _Served, started: float) -> Response:
        """Pass the teacher's SSE bytes through; the request is logged once the stream is over.

        The status line has gone out with the first byte, so a teacher failure after it ends the stream with
        an SSE ``error`` event (OpenAI clients raise on it) and is logged with status 502.
        """
        scanner = _SSEUsage()

        async def body() -> AsyncIterator[bytes]:
            tail = b""
            try:
                if first:
                    scanner.feed(first)
                    tail = first[-2:]
                    yield first
                async for chunk in rest:
                    scanner.feed(chunk)
                    if chunk:
                        tail = (tail + chunk)[-2:]
                    yield chunk
            except TEACHER_FAILURES as exc:
                kind = teacher_error_kind(exc)
                served.teacher_error = kind
                served.status = 502
                self.metrics.teacher_errors.labels(kind=kind).inc()
                log.warning("teacher stream ended early (%s): %s", kind, type(exc).__name__)
                message = f"the teacher stream broke off ({kind}); the answer is incomplete"
                event = _error_body(502, message, "teacher_error")
                separator = b"" if tail in (b"", b"\n\n") else b"\n\n"
                yield separator + b"data: " + _json_bytes(event) + b"\n\n"

        stream = body()

        async def cleanup() -> None:
            for generator in (stream, rest):
                aclose = getattr(generator, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception as exc:
                        log.warning("closing the teacher stream failed: %s", type(exc).__name__)
            elapsed = time.perf_counter() - started
            served.teacher_ms = elapsed * 1000.0
            if served.teacher_error is None:
                self.metrics.teacher_latency.observe(elapsed)
            served.prompt_tokens = _int(scanner.usage.get("prompt_tokens"))
            served.completion_tokens = _int(scanner.usage.get("completion_tokens"))
            served.teacher_cost_usd = 0.0 if self.replaying else _cost(scanner.usage)
            await self.finish(served)

        return _RelayResponse(stream, cleanup, {**served.headers(), **SSE_HEADERS})

    async def finish(self, served: _Served) -> None:
        """Count the request and log it to the store (a logging failure is only a warning)."""
        if served.finished:
            return
        served.finished = True
        total = time.perf_counter() - served.started
        self.metrics.requests.labels(route=served.route).inc()
        self.metrics.request_latency.labels(route=served.route).observe(total)
        try:
            await run_in_threadpool(
                self.store.add_served,
                task=self.spec.task,
                route=served.route,
                reason=served.reason,
                confidence=served.confidence,
                student_ms=served.student_ms,
                teacher_ms=served.teacher_ms,
                total_ms=total * 1000.0,
                prompt_tokens=served.prompt_tokens,
                completion_tokens=served.completion_tokens,
                teacher_cost_usd=served.teacher_cost_usd,
                teacher_mode=served.teacher_mode,
                status=served.status,
                request_model=served.request_model,
            )
        except Exception as exc:
            log.warning("serve logging failed: %s: %s", type(exc).__name__, exc)

    def health(self) -> tuple[int, dict[str, Any]]:
        status = self.worker.status
        data = {
            "status": "ok" if status == "ready" else status,
            "task": self.spec.task,
            "run": self.run_id,
            "threshold": None if math.isinf(self.threshold) else self.threshold,
            "teacher": self.teacher.mode,
            "backend": self.worker.backend_name,
            "escalation_response": self.spec.cascade.escalation_response,
            "on_teacher_error": self.spec.cascade.on_teacher_error,
        }
        return (200 if status == "ready" else 503), data


def _check_spec(spec: TaskSpec) -> None:
    if spec.type == "classification" and not spec.labels:
        raise ValueError(f"task {spec.task}: no labels loaded")
    if spec.type == "extraction" and not spec.json_schema:
        raise ValueError(f"task {spec.task}: no JSON Schema loaded")


def create_app(
    spec: TaskSpec,
    *,
    worker: ModelWorker,
    teacher: TeacherSource,
    threshold: float,
    store: Store,
    token: str | None = None,
    run_id: str | None = None,
) -> FastAPI:
    """Build the cascade server.

    The app starts ``worker`` and waits for the model at startup (a load failure fails the startup with
    :class:`ModelLoadError`), and stops the worker and closes ``teacher`` at shutdown. With ``token`` set,
    every endpoint requires ``Authorization: Bearer <token>``.
    """
    _check_spec(spec)
    if math.isnan(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a number >= 0 (math.inf escalates everything), got {threshold!r}")
    if token is not None and not token:
        raise ValueError("token must be a non-empty string or None")
    cascade = _Cascade(spec, worker, teacher, threshold, store, token, run_id)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            worker.start()
            await run_in_threadpool(worker.wait_ready)
            if worker.load_error is not None:
                raise ModelLoadError(
                    f"the student model failed to load: {type(worker.load_error).__name__}: {worker.load_error}"
                ) from worker.load_error
            yield
        finally:
            await run_in_threadpool(worker.stop)
            await teacher.aclose()

    app = FastAPI(
        title=f"taskdistill cascade ({spec.task})", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.cascade = cascade
    app.state.worker = worker
    app.state.metrics = cascade.metrics

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        if not cascade.authorised(request):
            cascade.metrics.requests.labels(route="error").inc()
            return cascade.unauthorised(chat=True)
        served = _Served(started=time.perf_counter())
        try:
            response = await cascade.handle(request, served)
        except Exception as exc:
            log.exception("request failed: %s", type(exc).__name__)
            served.deferred = False
            response = cascade.error(served, 500, "internal server error", "server_error")
        if not served.deferred:
            await cascade.finish(served)
        return response

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        if not cascade.authorised(request):
            return cascade.unauthorised()
        data = [{"id": cascade.model_id, "object": "model", "created": 0, "owned_by": "taskdistill"}]
        return JSONResponse({"object": "list", "data": data})

    @app.get("/healthz")
    async def healthz(request: Request) -> Response:
        if not cascade.authorised(request):
            return cascade.unauthorised()
        status, data = cascade.health()
        return JSONResponse(data, status_code=status)

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        if not cascade.authorised(request):
            return cascade.unauthorised()
        return Response(cascade.metrics.render(), media_type=cascade.metrics.content_type)

    return app
