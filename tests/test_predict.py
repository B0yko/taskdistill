from __future__ import annotations

import math
import string
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from taskdistill.backends.base import Backend
from taskdistill.backends.types import DecodeSession, Generation
from taskdistill.config import TaskSpec
from taskdistill.predict import Prediction, predict

L = math.log
END = 0
PIECES = [
    "<|end|>",
    "<|start|>",
    "system",
    "user",
    "assistant",
    "\n",
    "card",
    "_arr",
    "ival",
    "_link",
    "ing",
    "cash",
    '{"',
    '":',
    ' "',
    '",',
    "vendor",
    "total",
    "Example",
    " Ltd",
    " 12",
    ".5",
    *(string.ascii_letters + string.digits + " _-.,:;{}[]\"'?!"),
]
SCHEMA = {
    "type": "object",
    "properties": {"vendor": {"type": ["string", "null"]}, "total": {"type": ["number", "null"]}},
    "required": ["vendor", "total"],
    "additionalProperties": False,
}


class PieceTokenizer:
    """Greedy longest-match tokenizer over ``PIECES``; the id is the piece's index."""

    def __init__(self) -> None:
        self.ids = {p: i for i, p in enumerate(PIECES)}
        self.by_len = sorted(PIECES, key=len, reverse=True)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            piece = next(p for p in self.by_len if text.startswith(p, i))
            out.append(self.ids[piece])
            i += len(piece)
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(PIECES[i] for i in ids)


TOK = PieceTokenizer()
CARD, ARR, IVAL, LINK, ING, CASH = (TOK.ids[p] for p in ("card", "_arr", "ival", "_link", "ing", "cash"))
Script = Callable[[str, tuple[int, ...]], Mapping[int, float]]


class ScriptedSession:
    def __init__(self, prompt: str, script: Script) -> None:
        self.prompt = prompt
        self.script = script
        self.fed: list[int] = []

    def logprobs(self) -> np.ndarray:
        out = np.full(len(PIECES), L(1e-6), dtype=np.float64)
        for token, value in self.script(self.prompt, tuple(self.fed)).items():
            out[token] = value
        return out

    def feed(self, token_id: int) -> None:
        self.fed.append(token_id)


class ScriptedBackend(Backend):
    """The shared :class:`Backend` logic on top of a scripted decoding session."""

    name = "scripted"

    def __init__(self, script: Script) -> None:
        super().__init__("test/student")
        self.tokenizer = TOK
        self.stop_ids = [END]
        self.script = script
        self.trie_calls = 0
        self.prompts: list[str] = []

    def load(self) -> None:
        return None

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        text = "".join(f"<|start|>{m['role']}\n{m['content']}<|end|>\n" for m in messages)
        return TOK.encode(text + "<|start|>assistant\n")

    def start(self, prompt_ids: list[int]) -> DecodeSession:
        prompt = TOK.decode(prompt_ids)
        self.prompts.append(prompt)
        return ScriptedSession(prompt, self.script)

    def label_trie(self, labels: list[str], messages: list[dict[str, str]] | None = None) -> Any:
        self.trie_calls += 1
        return super().label_trie(labels, messages)


def make_spec(kind: str, *, max_tokens: int = 16) -> TaskSpec:
    raw: dict[str, Any] = {
        "task": "t",
        "type": kind,
        "teacher": {"model": "vendor/model"},
        "student": {"system_prompt": "Answer briefly.", "max_tokens": max_tokens},
        "cascade": {"target": 0.9},
    }
    if kind == "classification":
        raw["labels_file"] = "labels.txt"
    else:
        raw["schema_file"] = "schema.json"
    spec = TaskSpec.model_validate(raw)
    if kind == "classification":
        spec.labels = ["card", "card_arrival", "cash"]
    else:
        spec.json_schema = SCHEMA
    return spec


# -- classification ---------------------------------------------------------------------------------


def classification_script(prompt: str, prefix: tuple[int, ...]) -> Mapping[int, float]:
    if "link" in prompt:
        steps = {(): {CARD: L(0.6), CASH: L(0.2)}, (CARD,): {LINK: L(0.7), END: L(0.1), ARR: L(0.1)}}
        steps |= {(CARD, LINK): {ING: L(0.9)}, (CARD, LINK, ING): {END: L(0.9)}}
        return steps.get(prefix, {})
    steps = {
        (): {CARD: L(0.6), CASH: L(0.2)},
        (CARD,): {ARR: L(0.5), END: L(0.25)},
        (CARD, ARR): {IVAL: L(0.9)},
        (CARD, ARR, IVAL): {END: L(0.8)},
    }
    return steps.get(prefix, {})


def test_classification_uses_trie_and_renormalised_product() -> None:
    backend = ScriptedBackend(classification_script)
    spec = make_spec("classification")
    pred = predict(spec, backend, "has my card arrived?")
    assert isinstance(pred, Prediction)
    assert pred.answer == pred.value == pred.raw_text == "card_arrival"
    # root: .6 / (.6 + .2); after "card": .5 / (.5 + .25); "ival" and the end token are forced
    assert pred.confidence == pytest.approx(0.75 * (2 / 3))
    assert pred.completion_tokens == 4
    assert pred.prompt_tokens == len(
        backend.prompt_ids(
            [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "has my card arrived?"}]
        )
    )
    assert pred.latency_ms > 0
    assert pred.field_confidences is None
    assert (pred.alt_answer, pred.alt_value, pred.alt_confidence) == (None, None, None)
    assert backend.trie_calls == 1
    assert "<|start|>system\nAnswer briefly.<|end|>" in backend.prompts[-1]
    assert "<|start|>user\nhas my card arrived?<|end|>" in backend.prompts[-1]


def test_classification_alternative_free_generation() -> None:
    backend = ScriptedBackend(classification_script)
    pred = predict(make_spec("classification"), backend, "has my card arrived?", alternatives=True)
    assert pred.answer == "card_arrival"
    assert pred.alt_answer == "card_arrival"
    assert pred.alt_value == "card_arrival"
    assert pred.alt_confidence == pytest.approx((0.6 * 0.5 * 0.9 * 0.8) ** 0.25)


def test_classification_alternative_outside_label_set_is_none() -> None:
    backend = ScriptedBackend(classification_script)
    pred = predict(make_spec("classification"), backend, "link my card", alternatives=True)
    # constrained: "_link" is not allowed after "card"; the end token and "_arr" tie, lowest id wins
    assert pred.answer == "card"
    assert pred.confidence == pytest.approx(0.75 * 0.5)
    assert pred.alt_answer == "card_linking"
    assert pred.alt_value is None
    assert pred.alt_confidence == pytest.approx((0.6 * 0.7 * 0.9 * 0.9) ** 0.25)


def test_classification_alternative_trims_whitespace_only() -> None:
    space, nl = TOK.ids[" "], TOK.ids["\n"]

    def script(prompt: str, prefix: tuple[int, ...]) -> Mapping[int, float]:
        steps: dict[tuple[int, ...], dict[int, float]] = {
            (): {space: L(0.9)},
            (space,): {CASH: L(0.9)},
            (space, CASH): {nl: L(0.9)},
            (space, CASH, nl): {END: L(0.9)},
        }
        return steps.get(prefix, {})

    pred = predict(make_spec("classification"), ScriptedBackend(script), "x", alternatives=True)
    assert pred.alt_answer == " cash\n"
    assert pred.alt_value == "cash"


def test_classification_explicit_trie_and_messages() -> None:
    backend = ScriptedBackend(classification_script)
    spec = make_spec("classification")
    trie = backend.label_trie(spec.labels)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "link"}]
    pred = predict(spec, backend, "ignored", trie=trie, messages=messages)
    assert backend.trie_calls == 1
    assert pred.answer == "card"
    assert backend.prompts[-1].startswith("<|start|>system\ns<|end|>\n<|start|>user\nlink<|end|>")


# -- extraction -------------------------------------------------------------------------------------

JSON_TEXT = '{"vendor": "Example Ltd", "total": 12.5}'
JSON_PROBS = [0.99, 0.98, 0.97, 0.9, 0.8, 0.7, 0.95, 0.96, 0.99, 0.9, 0.6, 0.5, 0.9, 0.85]


def planned(text: str, probs: Sequence[float]) -> Script:
    seq = [*TOK.encode(text), END]
    assert len(seq) == len(probs)

    def script(prompt: str, prefix: tuple[int, ...]) -> Mapping[int, float]:
        k = len(prefix)
        return {seq[k]: L(probs[k])} if k < len(seq) else {}

    return script


def test_extraction_field_and_document_confidence() -> None:
    assert TOK.decode(TOK.encode(JSON_TEXT)) == JSON_TEXT
    assert len(TOK.encode(JSON_TEXT)) == 13
    backend = ScriptedBackend(planned(JSON_TEXT, JSON_PROBS))
    pred = predict(make_spec("extraction", max_tokens=64), backend, "Invoice from Example Ltd, total 12.50")
    assert pred.raw_text == JSON_TEXT
    assert pred.value == {"vendor": "Example Ltd", "total": 12.5}
    assert pred.answer == '{"vendor":"Example Ltd","total":12.5}'
    assert pred.field_confidences == pytest.approx({"vendor": 0.9 * 0.8 * 0.7 * 0.95, "total": 0.6 * 0.5})
    assert pred.confidence == pytest.approx(0.3)
    assert pred.completion_tokens == 14
    assert pred.alt_answer == pred.answer
    assert pred.alt_value == pred.value
    assert pred.alt_confidence == pytest.approx(math.exp(sum(L(p) for p in JSON_PROBS) / 14))


def test_extraction_invalid_output_scores_zero() -> None:
    text = '{"vendor": "Example Ltd"'
    probs = [0.99, 0.98, 0.97, 0.9, 0.8, 0.7, 0.95, 0.5]  # ..., ' Ltd', '"', end token
    backend = ScriptedBackend(planned(text, probs))
    pred = predict(make_spec("extraction", max_tokens=64), backend, "doc")
    assert pred.raw_text == text
    assert (pred.answer, pred.value, pred.confidence, pred.field_confidences) == (None, None, 0.0, {})
    assert pred.alt_value is None
    assert pred.alt_confidence == pytest.approx(math.exp(sum(L(p) for p in probs) / 8))


def test_extraction_truncated_by_max_tokens_scores_zero() -> None:
    backend = ScriptedBackend(planned(JSON_TEXT, JSON_PROBS))
    pred = predict(make_spec("extraction", max_tokens=5), backend, "doc")
    assert pred.completion_tokens == 5
    assert pred.raw_text == '{"vendor": "Example'
    assert pred.confidence == 0.0
    assert pred.answer is None


class CannedBackend:
    """Returns one prebuilt generation, for outputs too long to script token by token."""

    def __init__(self, gen: Generation) -> None:
        self.gen = gen

    def label_trie(self, labels: list[str]) -> Any:
        raise AssertionError("extraction never builds a label trie")

    def generate_with_scores(
        self, messages: list[dict[str, str]], constraint: Any = None, max_tokens: int = 256
    ) -> Generation:
        assert constraint is None
        return self.gen


@pytest.mark.parametrize(
    "text",
    [
        '{"vendor": ' + "[" * 20000,  # a runaway generation cut off at max_tokens
        '{"vendor": ' + "[" * 10000 + "]" * 10000 + ', "total": 1}',
        '```json\n{"vendor": "run ```make``` first", "total": 1}\n```',
    ],
)
def test_extraction_degenerate_output_scores_zero(text: str) -> None:
    gen = Generation(text, [1, 2], [L(0.9), L(0.9)], [(0, len(text)), (len(text), len(text))], "length", 7)
    pred = predict(make_spec("extraction", max_tokens=64), CannedBackend(gen), "doc")
    assert (pred.answer, pred.value, pred.confidence, pred.field_confidences) == (None, None, 0.0, {})
    assert pred.alt_confidence == pytest.approx(0.9)
    assert (pred.prompt_tokens, pred.completion_tokens, pred.raw_text) == (7, 2, text)


def test_extraction_without_schema_raises() -> None:
    spec = make_spec("extraction")
    spec.json_schema = None
    with pytest.raises(ValueError, match="schema"):
        predict(spec, ScriptedBackend(planned(JSON_TEXT, JSON_PROBS)), "doc")


# -- the shared fake backend ------------------------------------------------------------------------


def test_predict_with_fake_backend() -> None:
    fake = pytest.importorskip("taskdistill.backends.fake")
    backend = fake.FakeBackend(
        answers={"hello": ("card", 0.8), "doc": ('{"vendor": "Example Ltd", "total": null}', 0.7)}
    )
    pred = predict(make_spec("classification"), backend, "hello", alternatives=True)
    assert (pred.answer, pred.confidence) == ("card", pytest.approx(0.8))
    assert (pred.alt_value, pred.alt_confidence) == ("card", pytest.approx(0.8))
    pred = predict(make_spec("extraction", max_tokens=64), backend, "doc")
    assert pred.answer == '{"vendor":"Example Ltd","total":null}'
    assert pred.field_confidences == pytest.approx({"vendor": 0.7, "total": 0.7})
    assert pred.confidence == pytest.approx(0.7)
