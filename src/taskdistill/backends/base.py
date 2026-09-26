"""Backend interface shared by the MLX backend, the torch backend and the fake backend used in tests.

A backend only has to provide tokenisation and a step-wise decoding session that returns
log-probabilities over the vocabulary. Constrained decoding, span mapping and confidence are
implemented once, in :mod:`taskdistill.confidence`, on top of that interface.
"""

from __future__ import annotations

import time
from typing import Any

from taskdistill.backends.types import DecodeSession, Generation
from taskdistill.confidence import LabelTrie, constrained_greedy, free_greedy

__all__ = ["Backend", "DecodeSession", "Generation"]


class Backend:
    """Base class: subclasses implement ``_load``, ``prompt_ids``, ``start`` and ``tokenizer``."""

    name = "base"

    def __init__(self, base_model: str, adapter_path: str | None = None) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.tokenizer: Any = None
        self.stop_ids: list[int] = []

    # -- to implement -----------------------------------------------------------------------------
    def load(self) -> None:
        raise NotImplementedError

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        """Token ids of the chat template with the generation prompt appended."""
        raise NotImplementedError

    def start(self, prompt_ids: list[int]) -> DecodeSession:
        raise NotImplementedError

    # -- shared -------------------------------------------------------------------------------------
    def label_trie(self, labels: list[str], messages: list[dict[str, str]] | None = None) -> LabelTrie:
        return LabelTrie.build(self.tokenizer, labels, end_ids=self.stop_ids[:1], prompt_ids_fn=self.prompt_ids)

    def generate_with_scores(
        self,
        messages: list[dict[str, str]],
        constraint: LabelTrie | None = None,
        max_tokens: int = 256,
    ) -> Generation:
        """Greedy decoding with per-token log-probs; trie-constrained when ``constraint`` is given."""
        started = time.perf_counter()
        ids = self.prompt_ids(messages)
        session = self.start(ids)
        if constraint is not None:
            gen = constrained_greedy(session, constraint, self.tokenizer)
        else:
            gen = free_greedy(session, self.tokenizer, max_tokens=max_tokens, stop_ids=self.stop_ids)
        gen.prompt_tokens = len(ids)
        gen.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return gen
