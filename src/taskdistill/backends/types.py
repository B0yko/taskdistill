"""Types shared by backends and :mod:`taskdistill.confidence` (kept separate to avoid import cycles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class DecodeSession(Protocol):
    """Incremental decoding state after a prompt has been prefilled."""

    def logprobs(self) -> np.ndarray:
        """Log-softmax over the vocabulary for the next token (1-D float array)."""
        ...

    def feed(self, token_id: int) -> None:
        """Append ``token_id`` and advance the state."""
        ...


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass
class Generation:
    """A decoded completion with per-token scores.

    ``token_logprobs`` are renormalised over the allowed tokens for constrained decoding, and
    full-vocabulary log-probs for free generation. ``token_spans`` are the character spans of each
    token in ``text``.
    """

    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    token_spans: list[tuple[int, int]]
    finish_reason: str  # "stop" | "length"
    prompt_tokens: int
    elapsed_ms: float = 0.0
    constrained: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
