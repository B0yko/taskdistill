"""An OpenAI-compatible HTTP front for any teacher source.

The demos put the capture proxy in front of it, so proxy traffic is answered by the recording (replay) or by
the budgeted live teacher without touching the proxy's own forwarding code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from taskdistill.teacher.base import (
    BudgetExceeded,
    ReplayMiss,
    TeacherError,
    TeacherHTTPError,
    TeacherSource,
    TeacherTimeout,
)


def _error(status: int, message: str, kind: str, headers: dict[str, str]) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind, "code": status}}, status_code=status, headers=headers
    )


async def _chain(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    if first:
        yield first
    async for chunk in rest:
        yield chunk


def create_upstream_app(source: TeacherSource, *, models: list[str] | None = None) -> FastAPI:
    """``POST /v1/chat/completions`` (and ``/chat/completions``) plus ``GET /v1/models`` (and ``/models``).

    Errors map to HTTP statuses: a replay miss is 404, a refused budget 402, a teacher HTTP error keeps its
    status, a timeout is 504 and any other teacher error 502. The source is closed when the app shuts down.
    """
    model = getattr(source, "model", None)
    listed = list(models) if models is not None else ([model] if isinstance(model, str) and model else [])
    headers: dict[str, str] = {"x-taskdistill-teacher": str(source.mode)}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await source.aclose()

    app = FastAPI(title="taskdistill teacher upstream", lifespan=lifespan, docs_url=None, redoc_url=None)

    async def chat_completions(request: Request) -> Response:
        try:
            body: Any = await request.json()
        except ValueError:
            return _error(400, "the request body is not JSON", "invalid_request_error", headers)
        if not isinstance(body, dict):
            return _error(400, "the request body must be a JSON object", "invalid_request_error", headers)
        try:
            if body.get("stream"):
                chunks = source.stream(body)
                first = await anext(chunks, b"")
                return StreamingResponse(_chain(first, chunks), media_type="text/event-stream", headers=headers)
            result = await source.complete(body)
        except ReplayMiss as exc:
            return _error(404, str(exc), "replay_miss", headers)
        except BudgetExceeded as exc:
            return _error(402, str(exc), "budget_exceeded", headers)
        except TeacherHTTPError as exc:
            status = exc.status if 400 <= exc.status < 600 else 502
            return _error(status, str(exc), "teacher_error", headers)
        except TeacherTimeout as exc:
            return _error(504, str(exc), "teacher_timeout", headers)
        except TeacherError as exc:
            return _error(502, str(exc), "teacher_error", headers)
        return JSONResponse(result.response, headers=headers)

    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "created": 0, "owned_by": "taskdistill"} for m in listed],
        }

    for prefix in ("/v1", ""):
        app.add_api_route(f"{prefix}/chat/completions", chat_completions, methods=["POST"])
        app.add_api_route(f"{prefix}/models", list_models, methods=["GET"])
    return app
