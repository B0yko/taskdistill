"""One student prediction with its confidence, shared by ``eval`` and ``serve``."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from taskdistill.backends.types import Generation
from taskdistill.confidence import (
    LabelTrie,
    classification_confidence,
    extraction_confidence,
    mean_logprob_confidence,
)
from taskdistill.config import TaskSpec
from taskdistill.tasks.extraction import canonical_output
from taskdistill.teacher.requests import build_student_messages

__all__ = ["Prediction", "ScoringBackend", "predict"]


class ScoringBackend(Protocol):
    """What :func:`predict` needs from a backend (``taskdistill.backends.base.Backend`` provides it)."""

    def label_trie(self, labels: list[str]) -> LabelTrie: ...

    def generate_with_scores(
        self,
        messages: list[dict[str, str]],
        constraint: LabelTrie | None = None,
        max_tokens: int = 256,
    ) -> Generation: ...


@dataclass
class Prediction:
    """The student's answer and scores for one input.

    ``answer`` is the canonical string (the label, or compact JSON in schema key order) and ``None``
    for an invalid extraction; ``value`` is the label or the parsed object. The ``alt_*`` fields hold
    the alternative score: free generation with mean token log-prob.
    """

    answer: str | None
    value: Any
    confidence: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    field_confidences: dict[str, float] | None = None
    alt_answer: str | None = None
    alt_value: Any = None
    alt_confidence: float | None = None
    raw_text: str = ""


def _exact_label(text: str, labels: list[str]) -> str | None:
    stripped = text.strip()
    return stripped if stripped in labels else None


def predict(
    spec: TaskSpec,
    backend: ScoringBackend,
    input_text: str,
    *,
    trie: LabelTrie | None = None,
    alternatives: bool = False,
    messages: list[dict[str, str]] | None = None,
) -> Prediction:
    """Run the student on ``input_text``.

    Classification decodes under the label trie (built from ``spec.labels`` when not given) and, with
    ``alternatives``, also generates freely: ``alt_value`` is the label the free output spells exactly
    (after trimming whitespace) or ``None``. Extraction generates freely once and scores the same
    generation both ways. ``latency_ms`` covers the primary generation and its scoring.
    """
    if messages is None:
        messages = build_student_messages(spec, input_text)
    max_tokens = spec.student.max_tokens

    if spec.type == "classification":
        if trie is None:
            trie = backend.label_trie(spec.labels)
        started = time.perf_counter()
        gen = backend.generate_with_scores(messages, constraint=trie, max_tokens=max_tokens)
        confidence = classification_confidence(gen)
        latency_ms = (time.perf_counter() - started) * 1000.0
        pred = Prediction(
            answer=gen.text,
            value=gen.text,
            confidence=confidence,
            latency_ms=latency_ms,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=len(gen.token_ids),
            raw_text=gen.text,
        )
        if alternatives:
            free = backend.generate_with_scores(messages, constraint=None, max_tokens=max_tokens)
            pred.alt_answer = free.text
            pred.alt_value = _exact_label(free.text, spec.labels)
            pred.alt_confidence = mean_logprob_confidence(free)
        return pred

    schema = spec.json_schema
    if schema is None:
        raise ValueError(f"task {spec.task}: extraction needs a JSON schema")
    started = time.perf_counter()
    gen = backend.generate_with_scores(messages, constraint=None, max_tokens=max_tokens)
    score = extraction_confidence(gen, schema)
    answer = canonical_output(score.obj, schema) if score.valid and score.obj is not None else None
    latency_ms = (time.perf_counter() - started) * 1000.0
    return Prediction(
        answer=answer,
        value=score.obj,
        confidence=score.doc_confidence,
        latency_ms=latency_ms,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=len(gen.token_ids),
        field_confidences=dict(score.field_confidences),
        alt_answer=answer,
        alt_value=score.obj,
        alt_confidence=mean_logprob_confidence(gen),
        raw_text=gen.text,
    )
