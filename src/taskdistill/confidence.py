"""Constrained decoding, span mapping and confidence scores for the student.

Classification decodes greedily under a :class:`LabelTrie` built over the canonical label strings
as they tokenise after the assistant prefix, each followed by the chat end token, so every output
is a valid label and a label that is a prefix of another is still reachable. Its confidence is the
product of the renormalised probabilities of the chosen tokens, end token included.

Extraction generates freely. Invalid JSON or a schema violation scores 0; otherwise each top-level
value's character span is mapped to the tokens that overlap it, a field's confidence is the product
of those tokens' probabilities and the document's confidence is the minimum over fields.

The alternative score for both task types is the exponentiated mean token log-prob of a free
generation (:func:`mean_logprob_confidence`).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from taskdistill.backends.types import DecodeSession, Generation, Tokenizer

__all__ = [
    "ExtractionScore",
    "LabelTrie",
    "TrieNode",
    "classification_confidence",
    "constrained_greedy",
    "extraction_confidence",
    "free_greedy",
    "json_value_spans",
    "mean_logprob_confidence",
    "token_char_spans",
]

PromptIdsFn = Callable[[list[dict[str, str]]], list[int]]
_PROBE_MESSAGES: list[dict[str, str]] = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
_JSON_WS = " \t\n\r"
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


# -- token spans ------------------------------------------------------------------------------------


def _common_prefix_len(a: str, b: str) -> int:
    if b.startswith(a):
        return len(a)
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def token_char_spans(tokenizer: Tokenizer, ids: Sequence[int]) -> list[tuple[int, int]]:
    """Character span of each token in ``tokenizer.decode(ids)``, by decoding growing prefixes.

    A prefix that decodes to text which is not a prefix of the full text (a partial UTF-8 sequence
    shown as a replacement character, or clean-up of spaces) is cut back to the common prefix. Spans
    are contiguous, monotone and clamped to the full text, so a token that only starts a multi-byte
    character gets an empty span at that character and the token that completes it covers it.
    """
    ids = [int(i) for i in ids]
    full = tokenizer.decode(ids)
    spans: list[tuple[int, int]] = []
    prev = 0
    for k in range(1, len(ids) + 1):
        partial = full if k == len(ids) else tokenizer.decode(ids[:k])
        end = min(max(_common_prefix_len(partial, full), prev), len(full))
        spans.append((prev, end))
        prev = end
    return spans


# -- label trie -------------------------------------------------------------------------------------


@dataclass
class TrieNode:
    """A trie node: children by token id; ``label`` is set on the node reached by a label's end token."""

    children: dict[int, TrieNode] = field(default_factory=dict)
    label: str | None = None


class LabelTrie:
    """Token trie over the label set; every label's sequence ends with the chat end token."""

    def __init__(self, sequences: Mapping[str, Sequence[int]], end_id: int) -> None:
        if not sequences:
            raise ValueError("a label trie needs at least one label")
        self.end_id = int(end_id)
        self.root = TrieNode()
        self.sequences: dict[str, tuple[int, ...]] = {}
        seen: dict[tuple[int, ...], str] = {}
        for label, raw in sequences.items():
            seq = tuple(int(t) for t in raw)
            if len(seq) < 2 or seq[-1] != self.end_id:
                raise ValueError(f"label {label!r}: token sequence must be non-empty and end with the end token")
            if self.end_id in seq[:-1]:
                raise ValueError(f"label {label!r} contains the end token")
            if seq in seen:
                raise ValueError(f"labels {seen[seq]!r} and {label!r} tokenise identically")
            seen[seq] = label
            self.sequences[label] = seq
            node = self.root
            for token in seq:
                node = node.children.setdefault(token, TrieNode())
            node.label = label

    @classmethod
    def build(
        cls,
        tokenizer: Tokenizer,
        labels: Sequence[str],
        end_ids: Sequence[int],
        prompt_ids_fn: PromptIdsFn | None = None,
    ) -> LabelTrie:
        """Tokenise ``labels`` as the model sees them after the assistant prefix and append ``end_ids[0]``.

        With ``prompt_ids_fn`` (the backend's chat template with the generation prompt), each label is
        encoded in context: ``encode(decode(prefix) + label)`` minus the prefix tokens. If the prefix
        does not survive re-encoding, the label is encoded on its own.
        """
        if not end_ids:
            raise ValueError("an end token id (the chat end token) is required")
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate labels")
        prefix: list[int] = []
        prefix_text = ""
        if prompt_ids_fn is not None:
            prefix = [int(t) for t in prompt_ids_fn(_PROBE_MESSAGES)]
            prefix_text = tokenizer.decode(prefix)
        end_id = int(end_ids[0])
        sequences: dict[str, list[int]] = {}
        for label in labels:
            if not label:
                raise ValueError("labels must be non-empty strings")
            tokens: list[int] | None = None
            if prompt_ids_fn is not None:
                full = [int(t) for t in tokenizer.encode(prefix_text + label, add_special_tokens=False)]
                if full[: len(prefix)] == prefix and len(full) > len(prefix):
                    tokens = full[len(prefix) :]
            if tokens is None:
                tokens = [int(t) for t in tokenizer.encode(label, add_special_tokens=False)]
            if not tokens:
                raise ValueError(f"label {label!r} tokenises to no tokens")
            sequences[label] = [*tokens, end_id]
        return cls(sequences, end_id)

    @property
    def labels(self) -> list[str]:
        return list(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def node(self, prefix: Sequence[int]) -> TrieNode:
        """The node reached by ``prefix``; raises ``ValueError`` if the prefix leaves the trie."""
        node = self.root
        for token in prefix:
            child = node.children.get(int(token))
            if child is None:
                raise ValueError(f"token {token} is not allowed after {list(prefix)}")
            node = child
        return node

    def allowed(self, prefix: Sequence[int]) -> list[int]:
        """Token ids allowed after ``prefix`` (ascending); empty once a label is complete."""
        return sorted(self.node(prefix).children)

    def label_of(self, tokens: Sequence[int]) -> str | None:
        """The label spelled by ``tokens`` (end token included), or ``None``."""
        try:
            return self.node(tokens).label
        except ValueError:
            return None


# -- decoding ---------------------------------------------------------------------------------------


def _renormalise(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Log-probs renormalised over ``values`` with log-sum-exp; non-finite maxima share the mass."""
    vals = np.where(np.isnan(values), -np.inf, values)
    top = float(vals.max())
    if not math.isfinite(top):
        mask = vals == top
        share = -math.log(int(mask.sum()))
        return np.where(mask, share, -np.inf)
    lse = top + math.log(float(np.exp(vals - top).sum()))
    out: npt.NDArray[np.float64] = vals - lse
    return out


def _argmax(logprobs: npt.NDArray[Any]) -> int:
    """Index of the largest entry (lowest index on ties), ignoring NaN."""
    idx = int(np.argmax(logprobs))
    if math.isnan(float(logprobs[idx])):
        idx = int(np.nanargmax(logprobs))
    return idx


def constrained_greedy(session: DecodeSession, trie: LabelTrie, tokenizer: Tokenizer) -> Generation:
    """Greedy decoding restricted to the trie; returns the label with renormalised per-token log-probs.

    At each step the log-probs of the allowed tokens are renormalised with log-sum-exp and the argmax
    is taken (lowest token id on ties). A step with a single allowed token has probability 1 after
    renormalisation, so the model is not queried for it. The end token is scored but not fed.
    """
    node = trie.root
    token_ids: list[int] = []
    token_logprobs: list[float] = []
    queried = 0
    while node.label is None:
        allowed = sorted(node.children)
        if not allowed:
            raise RuntimeError("label trie has a dead end")
        if len(allowed) == 1:
            token, logprob = allowed[0], 0.0
        else:
            full = np.asarray(session.logprobs())
            renorm = _renormalise(full[allowed].astype(np.float64))
            queried += 1
            best = int(np.argmax(renorm))
            token, logprob = allowed[best], float(renorm[best])
        token_ids.append(token)
        token_logprobs.append(logprob)
        node = node.children[token]
        if token != trie.end_id:
            session.feed(token)
    label = node.label
    n = len(label)
    spans = [(min(s, n), min(e, n)) for s, e in token_char_spans(tokenizer, token_ids[:-1])]
    spans.append((n, n))
    return Generation(
        text=label,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        token_spans=spans,
        finish_reason="stop",
        prompt_tokens=0,
        constrained=True,
        extra={"queried_steps": queried},
    )


def free_greedy(session: DecodeSession, tokenizer: Tokenizer, max_tokens: int, stop_ids: Sequence[int]) -> Generation:
    """Unconstrained greedy decoding with full-vocabulary log-probs.

    Stops when a stop id is chosen (it is kept in ``token_ids`` and ``token_logprobs`` with an empty
    span at the end of the text, but not in ``text``) or after ``max_tokens`` tokens (``"length"``).
    """
    stops = {int(s) for s in stop_ids}
    content: list[int] = []
    token_ids: list[int] = []
    token_logprobs: list[float] = []
    finish = "length"
    for step in range(max(0, max_tokens)):
        logprobs = np.asarray(session.logprobs())
        token = _argmax(logprobs)
        token_ids.append(token)
        token_logprobs.append(float(logprobs[token]))
        if token in stops:
            finish = "stop"
            break
        content.append(token)
        if step + 1 < max_tokens:
            session.feed(token)
    text = tokenizer.decode(content) if content else ""
    spans = token_char_spans(tokenizer, content) if content else []
    if finish == "stop":
        spans.append((len(text), len(text)))
    return Generation(
        text=text,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        token_spans=spans,
        finish_reason=finish,
        prompt_tokens=0,
        constrained=False,
    )


# -- scores -----------------------------------------------------------------------------------------


def _prob(logprob_sum: float) -> float:
    if math.isnan(logprob_sum):
        return 0.0
    return min(1.0, max(0.0, math.exp(min(logprob_sum, 0.0))))


def classification_confidence(gen: Generation) -> float:
    """Product of the chosen tokens' renormalised probabilities, end token included (0.0 when empty)."""
    if not gen.token_logprobs:
        return 0.0
    return _prob(math.fsum(gen.token_logprobs))


def mean_logprob_confidence(gen: Generation) -> float:
    """``exp`` of the mean token log-prob (0.0 for an empty generation)."""
    if not gen.token_logprobs:
        return 0.0
    return _prob(math.fsum(gen.token_logprobs) / len(gen.token_logprobs))


# -- JSON spans -------------------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not JSON")


def _finite_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        raise ValueError(f"number {literal} overflows a float")
    return value


_DECODER = json.JSONDecoder(parse_constant=_reject_constant, parse_float=_finite_float)


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in _JSON_WS:
        i += 1
    return i


def _scan_object(text: str, i: int) -> tuple[dict[str, tuple[int, int]], int]:
    """Scan the object opening at ``text[i]``; returns value spans per key and the index after ``}``."""
    spans: dict[str, tuple[int, int]] = {}
    i = _skip_ws(text, i + 1)
    if i < len(text) and text[i] == "}":
        return spans, i + 1
    while True:
        if i >= len(text) or text[i] != '"':
            raise ValueError(f"expected a key at {i}")
        key, i = _DECODER.raw_decode(text, i)
        i = _skip_ws(text, i)
        if i >= len(text) or text[i] != ":":
            raise ValueError(f"expected ':' at {i}")
        i = _skip_ws(text, i + 1)
        start = i
        _, i = _DECODER.raw_decode(text, i)
        spans[str(key)] = (start, i)
        i = _skip_ws(text, i)
        if i >= len(text):
            raise ValueError("unterminated object")
        if text[i] == ",":
            i = _skip_ws(text, i + 1)
            continue
        if text[i] == "}":
            return spans, i + 1
        raise ValueError(f"expected ',' or '}}' at {i}")


def json_value_spans(text: str) -> tuple[dict[str, Any] | None, dict[str, tuple[int, int]]]:
    """Parse the first top-level JSON object in ``text`` and locate each top-level value.

    The object is the one :func:`taskdistill.tasks.extraction.parse_json_output` returns: inside the
    first Markdown code fence that contains a ``{`` if there is one, starting at the first ``{``, with
    any text after its closing brace ignored. Spans are ``(start, end)`` character offsets into the
    original ``text``; string values include their quotes and nested values are covered whole. A
    fenced object must close inside its fence. Returns ``(None, {})`` when no valid object starts
    there, including nesting too deep to parse. Non-finite numbers are rejected.
    """
    fence = next((m for m in _FENCE.finditer(text) if "{" in m.group(1)), None)
    if fence is not None:
        scan = text[: fence.end(1)]
        start = scan.find("{", fence.start(1))
    else:
        scan, start = text, text.find("{")
    if start < 0:
        return None, {}
    try:
        spans, end = _scan_object(scan, start)
        obj = _DECODER.decode(scan[start:end])
    except (ValueError, RecursionError):
        return None, {}
    if not isinstance(obj, dict):
        return None, {}
    return obj, spans


# -- extraction -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionScore:
    """Confidence of one extraction: ``doc_confidence`` is 0.0 and ``obj`` is ``None`` when invalid."""

    doc_confidence: float
    field_confidences: dict[str, float]
    obj: dict[str, Any] | None
    valid: bool
    errors: list[str] = field(default_factory=list)


def _overlaps(token: tuple[int, int], value: tuple[int, int]) -> bool:
    ts, te = token
    vs, ve = value
    if te > ts:
        return ts < ve and te > vs
    return vs <= ts < ve


def extraction_confidence(gen: Generation, schema: Mapping[str, Any] | None) -> ExtractionScore:
    """Field and document confidence of a free JSON generation.

    For each schema property present in the output, the tokens whose character span overlaps the
    value's span are collected (a token covering several values counts for each) and the field
    confidence is the product of their probabilities. The document confidence is the minimum over
    those fields; properties absent from the output have no span and are left out. With no schema
    property present, it is the product over all tokens.
    """
    from taskdistill.tasks.extraction import validate

    obj, spans = json_value_spans(gen.text)
    if obj is None:
        return ExtractionScore(0.0, {}, None, False, ["output is not a JSON object"])
    errors = validate(obj, dict(schema or {}))
    if errors:
        return ExtractionScore(0.0, {}, None, False, list(errors))
    properties = (schema or {}).get("properties") or {}
    field_confidences: dict[str, float] = {}
    for name in properties:
        value_span = spans.get(name)
        if value_span is None:
            continue
        selected = [
            lp for span, lp in zip(gen.token_spans, gen.token_logprobs, strict=False) if _overlaps(span, value_span)
        ]
        field_confidences[name] = _prob(math.fsum(selected))
    if field_confidences:
        doc = min(field_confidences.values())
    else:
        doc = _prob(math.fsum(gen.token_logprobs))
    return ExtractionScore(doc, field_confidences, obj, True)
