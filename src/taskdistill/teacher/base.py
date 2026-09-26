"""Teacher sources: the live OpenAI-compatible client and the recorded replay share one interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class TeacherError(RuntimeError):
    """Base class for every teacher failure."""


class BudgetExceeded(TeacherError):
    """A call was refused because its worst-case cost would cross a spend cap."""


class ReplayMiss(TeacherError):
    """A request key is not in the recording. Always a hard error, never a silent fallback."""


class ManifestMismatch(TeacherError):
    """A recording's manifest does not match the task spec; the message names the differing field."""


class TeacherHTTPError(TeacherError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"teacher returned HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class TeacherTimeout(TeacherError):
    """The teacher did not answer in time after all retries."""


@dataclass
class TeacherResult:
    key: str
    output: str | None
    response: dict[str, Any]
    usage: dict[str, Any]
    latency_ms: float | None
    provider: str | None
    finish_reason: str | None
    created: float
    source: Literal["live", "cache", "replay"]
    cost_usd: float = 0.0
    attempts: int = 1
    truncated: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class TeacherSource(Protocol):
    """Anything that can answer a chat-completions body like the teacher does."""

    mode: Literal["live", "replay"]

    async def complete(self, body: dict[str, Any]) -> TeacherResult: ...

    def stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Raw SSE bytes for a ``stream: true`` body (live pass-through, or synthesised from a replay)."""
        ...

    async def aclose(self) -> None: ...
