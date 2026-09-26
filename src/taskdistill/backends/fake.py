"""A scripted backend for tests: answers and confidences are looked up by the input text."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from taskdistill.backends.base import Backend
from taskdistill.backends.types import Generation

Answer = tuple[str, float]


class _WhitespaceTokenizer:
    """Just enough tokenizer for code paths that count tokens."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(w) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join("x" * i for i in ids)

    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


@dataclass
class FakeTrie:
    labels: list[str]


@dataclass
class FakeBackend(Backend):
    """Returns ``answers[input]`` (text, confidence), else ``default``; records every call.

    The generation is a single token spanning the whole text whose log-probability is ``log(confidence)``,
    so both the classification and the extraction confidence definitions return ``confidence``.
    """

    answers: Mapping[str, Answer] | Callable[[str], Answer] = field(default_factory=dict)
    default: Answer = ("", 0.5)
    delay_s: float = 0.0
    base_model: str = "fake/student"
    adapter_path: str | None = None
    calls: list[str] = field(default_factory=list)
    threads: set[int] = field(default_factory=set)
    loaded: bool = False

    def __post_init__(self) -> None:
        Backend.__init__(self, self.base_model, self.adapter_path)
        self.name = "fake"
        self.tokenizer = _WhitespaceTokenizer()
        self.stop_ids = [0]

    def load(self) -> None:
        self.loaded = True
        self.threads.add(threading.get_ident())

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        return self.tokenizer.encode(self.tokenizer.apply_chat_template(messages))

    def label_trie(self, labels: list[str], messages: list[dict[str, str]] | None = None) -> Any:
        return FakeTrie(list(labels))

    def _lookup(self, text: str) -> Answer:
        if callable(self.answers):
            return self.answers(text)
        return self.answers.get(text, self.default)

    def generate_with_scores(
        self,
        messages: list[dict[str, str]],
        constraint: Any = None,
        max_tokens: int = 256,
    ) -> Generation:
        started = time.perf_counter()
        self.threads.add(threading.get_ident())
        user = messages[-1]["content"]
        self.calls.append(user)
        if self.delay_s:
            time.sleep(self.delay_s)
        text, confidence = self._lookup(user)
        if isinstance(constraint, FakeTrie) and text not in constraint.labels:
            text = constraint.labels[0]
        confidence = min(max(confidence, 1e-12), 1.0)
        return Generation(
            text=text,
            token_ids=[1],
            token_logprobs=[math.log(confidence)],
            token_spans=[(0, len(text))],
            finish_reason="stop",
            prompt_tokens=len(self.prompt_ids(messages)),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            constrained=constraint is not None,
        )
