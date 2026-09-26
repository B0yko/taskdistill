from __future__ import annotations

import itertools
import math
import string
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from taskdistill.backends.types import Generation
from taskdistill.confidence import (
    ExtractionScore,
    LabelTrie,
    classification_confidence,
    constrained_greedy,
    extraction_confidence,
    free_greedy,
    json_value_spans,
    mean_logprob_confidence,
    token_char_spans,
)

# -- fakes ------------------------------------------------------------------------------------------

SPECIALS = ["<|end|>", "<|start|>"]
WORDS = [
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
    "▁card",
    "▁cash",
    '{"',
    '":',
    ' "',
    '",',
    "Example",
    " Ltd",
    " 12",
    ".5",
]
CHARS = list(string.ascii_letters + string.digits + " _-.,:;{}[]\"'\\/!?")


class PieceTokenizer:
    """Greedy longest-match tokenizer over a fixed piece list; the id is the piece's index.

    Pieces starting with ``▁`` are word-initial variants: when the text starts with a letter, the first
    token is swapped for its variant (as a SentencePiece dummy prefix would), so a label tokenises
    differently on its own than after the assistant prefix. Variants decode without the marker.
    """

    def __init__(self, pieces: Sequence[str], *, initial_variants: bool = False) -> None:
        self.pieces = list(pieces)
        self.ids = {p: i for i, p in enumerate(self.pieces)}
        self.matchable = sorted((p for p in self.pieces if not p.startswith("▁")), key=len, reverse=True)
        self.variants = (
            {self.ids[p[1:]]: i for i, p in enumerate(self.pieces) if p.startswith("▁") and p[1:] in self.ids}
            if initial_variants
            else {}
        )

    def id(self, piece: str) -> int:
        return self.ids[piece]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for piece in self.matchable:
                if text.startswith(piece, i):
                    out.append(self.ids[piece])
                    i += len(piece)
                    break
            else:
                raise ValueError(f"cannot encode {text[i]!r}")
        if out and text[0].isalpha() and out[0] in self.variants:
            out[0] = self.variants[out[0]]
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[i].removeprefix("▁") for i in ids)


class ByteTokenizer:
    """One token per UTF-8 byte; incomplete sequences decode to a replacement character."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


class CleanupTokenizer(PieceTokenizer):
    """Removes a space before punctuation on decode, like ``clean_up_tokenization_spaces``."""

    def decode(self, ids: list[int]) -> str:
        return super().decode(ids).replace(" .", ".")


class LowerTokenizer(PieceTokenizer):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return super().encode(text.lower(), add_special_tokens)


Script = Callable[[tuple[int, ...]], Mapping[int, float]]


class ScriptedSession:
    """A decoding session whose next-token log-probs are a scripted function of the fed tokens."""

    def __init__(self, vocab_size: int, script: Script, default: float = math.log(1e-6)) -> None:
        self.vocab_size = vocab_size
        self.script = script
        self.default = default
        self.fed: list[int] = []
        self.calls = 0

    def logprobs(self) -> np.ndarray:
        self.calls += 1
        out = np.full(self.vocab_size, self.default, dtype=np.float32)
        for token, value in self.script(tuple(self.fed)).items():
            out[token] = value
        return out

    def feed(self, token_id: int) -> None:
        self.fed.append(int(token_id))


def table(entries: Mapping[tuple[int, ...], Mapping[int, float]]) -> Script:
    return lambda prefix: entries.get(prefix, {})


@pytest.fixture
def tok() -> PieceTokenizer:
    return PieceTokenizer(SPECIALS + WORDS + CHARS)


END = 0
START = 1
L = math.log


def ids(tok: PieceTokenizer, *pieces: str) -> list[int]:
    return [tok.id(p) for p in pieces]


# -- token_char_spans -------------------------------------------------------------------------------


def test_token_char_spans_contiguous(tok: PieceTokenizer) -> None:
    assert token_char_spans(tok, ids(tok, "card", "_arr", "ival")) == [(0, 4), (4, 8), (8, 12)]
    assert token_char_spans(tok, []) == []


def test_token_char_spans_multibyte_partial_token_gets_empty_span() -> None:
    bt = ByteTokenizer()
    tokens = bt.encode("aé!")  # a, 0xC3, 0xA9, !
    assert tokens == [0x61, 0xC3, 0xA9, 0x21]
    assert token_char_spans(bt, tokens) == [(0, 1), (1, 1), (1, 2), (2, 3)]


def test_token_char_spans_monotone_under_decode_cleanup() -> None:
    ct = CleanupTokenizer(SPECIALS + WORDS + CHARS)
    tokens = ct.encode("a .b")
    assert ct.decode(tokens) == "a.b"
    # "a" -> "a"; "a " is not a prefix of "a.b" (common prefix 1); "a." -> 2; "a.b" -> 3
    assert token_char_spans(ct, tokens) == [(0, 1), (1, 1), (1, 2), (2, 3)]


# -- label trie -------------------------------------------------------------------------------------


def test_trie_prefix_label_ends_with_end_token(tok: PieceTokenizer) -> None:
    trie = LabelTrie.build(tok, ["card", "card_arrival", "card_linking"], end_ids=[END])
    card, arr, ival, link, ing = ids(tok, "card", "_arr", "ival", "_link", "ing")
    assert trie.sequences == {
        "card": (card, END),
        "card_arrival": (card, arr, ival, END),
        "card_linking": (card, link, ing, END),
    }
    assert trie.allowed([]) == [card]
    assert trie.allowed([card]) == sorted([END, arr, link])
    assert trie.allowed([card, arr]) == [ival]
    assert trie.allowed([card, arr, ival]) == [END]
    assert trie.allowed([card, END]) == []
    assert trie.label_of([card, END]) == "card"
    assert trie.label_of([card]) is None
    assert trie.label_of([arr]) is None
    assert len(trie) == 3
    with pytest.raises(ValueError, match="not allowed"):
        trie.allowed([arr])


def test_trie_uses_tokens_after_assistant_prefix() -> None:
    tok = PieceTokenizer(SPECIALS + WORDS + CHARS, initial_variants=True)
    assert tok.encode("card") == [tok.id("▁card")]

    def prompt_ids(messages: list[dict[str, str]]) -> list[int]:
        text = "".join(f"<|start|>{m['role']}\n{m['content']}<|end|>\n" for m in messages)
        return tok.encode(text + "<|start|>assistant\n")

    alone = LabelTrie.build(tok, ["card", "cash"], end_ids=[END])
    in_context = LabelTrie.build(tok, ["card", "cash"], end_ids=[END], prompt_ids_fn=prompt_ids)
    assert alone.sequences == {"card": (tok.id("▁card"), END), "cash": (tok.id("▁cash"), END)}
    assert in_context.sequences == {"card": (tok.id("card"), END), "cash": (tok.id("cash"), END)}


def test_trie_falls_back_when_prefix_merges_with_label() -> None:
    tok = PieceTokenizer([*SPECIALS, *WORDS, "\nc", *CHARS])

    def prompt_ids(messages: list[dict[str, str]]) -> list[int]:
        return tok.encode("<|start|>assistant\n")

    # "...\n" + "card" re-encodes as "...", "\nc", "a", "r", "d": the prefix is not preserved
    trie = LabelTrie.build(tok, ["card"], end_ids=[END], prompt_ids_fn=prompt_ids)
    assert trie.sequences["card"] == (tok.id("card"), END)


def test_trie_rejects_identical_tokenisation() -> None:
    lower = LowerTokenizer(SPECIALS + WORDS + CHARS)
    with pytest.raises(ValueError, match="tokenise identically"):
        LabelTrie.build(lower, ["card", "CARD"], end_ids=[END])
    with pytest.raises(ValueError, match="tokenise identically"):
        LabelTrie({"a": [5, END], "b": [5, END]}, end_id=END)


def test_trie_rejects_bad_input(tok: PieceTokenizer) -> None:
    with pytest.raises(ValueError, match="end token"):
        LabelTrie.build(tok, ["card"], end_ids=[])
    with pytest.raises(ValueError, match="duplicate"):
        LabelTrie.build(tok, ["card", "card"], end_ids=[END])
    with pytest.raises(ValueError, match="non-empty"):
        LabelTrie.build(tok, ["card", ""], end_ids=[END])
    with pytest.raises(ValueError, match="contains the end token"):
        LabelTrie.build(tok, ["card<|end|>x"], end_ids=[END])
    with pytest.raises(ValueError, match="at least one"):
        LabelTrie({}, end_id=END)
    with pytest.raises(ValueError, match="end with the end token"):
        LabelTrie({"a": [5]}, end_id=END)


# -- constrained decoding ---------------------------------------------------------------------------


def test_constrained_renormalises_over_allowed_tokens(tok: PieceTokenizer) -> None:
    trie = LabelTrie.build(tok, ["card", "cash"], end_ids=[END])
    card, cash = ids(tok, "card", "cash")
    # 0.7 of the mass sits on tokens outside the trie; renormalised: .2/.3 and .1/.3
    session = ScriptedSession(len(tok.pieces), table({(): {card: L(0.2), cash: L(0.1), tok.id("x"): L(0.7)}}))
    gen = constrained_greedy(session, trie, tok)
    assert gen.text == "card"
    assert gen.token_ids == [card, END]
    assert gen.token_logprobs[0] == pytest.approx(L(2 / 3))
    assert gen.token_logprobs[1] == 0.0  # only the end token is allowed after "card"
    assert classification_confidence(gen) == pytest.approx(0.6667, abs=1e-4)
    assert session.fed == [card]
    assert session.calls == 1
    assert gen.constrained is True
    assert gen.finish_reason == "stop"
    assert gen.token_spans == [(0, 4), (4, 4)]

    other = ScriptedSession(len(tok.pieces), table({(): {card: L(0.1), cash: L(0.2)}}))
    gen = constrained_greedy(other, trie, tok)
    assert gen.text == "cash"
    assert classification_confidence(gen) == pytest.approx(0.6667, abs=1e-4)


def test_constrained_prefix_label_stops_on_end_token(tok: PieceTokenizer) -> None:
    labels = ["card", "card_arrival", "card_linking"]
    trie = LabelTrie.build(tok, labels, end_ids=[END])
    card, arr, link = ids(tok, "card", "_arr", "_link")
    after_card = {END: L(0.5), arr: L(0.3), link: L(0.1)}
    session = ScriptedSession(len(tok.pieces), table({(card,): after_card}))
    gen = constrained_greedy(session, trie, tok)
    assert gen.text == "card"
    assert gen.token_ids == [card, END]
    # the only query is the branch after "card": .5 / (.5 + .3 + .1)
    assert classification_confidence(gen) == pytest.approx(0.5 / 0.9)
    assert session.calls == 1
    assert session.fed == [card]


def test_constrained_longer_label_through_prefix(tok: PieceTokenizer) -> None:
    trie = LabelTrie.build(tok, ["card", "card_arrival", "card_linking"], end_ids=[END])
    card, arr, ival, link = ids(tok, "card", "_arr", "ival", "_link")
    session = ScriptedSession(len(tok.pieces), table({(card,): {END: L(0.2), arr: L(0.6), link: L(0.1)}}))
    gen = constrained_greedy(session, trie, tok)
    assert gen.text == "card_arrival"
    assert gen.token_ids == [card, arr, ival, END]
    assert gen.token_logprobs == pytest.approx([0.0, L(0.6 / 0.9), 0.0, 0.0])
    assert classification_confidence(gen) == pytest.approx(2 / 3)
    assert gen.token_spans == [(0, 4), (4, 8), (8, 12), (12, 12)]
    assert session.fed == [card, arr, ival]
    assert gen.extra["queried_steps"] == 1


def test_constrained_product_includes_end_token(tok: PieceTokenizer) -> None:
    trie = LabelTrie.build(tok, ["card", "card_arrival", "card_linking", "cash"], end_ids=[END])
    card, cash, arr, link = ids(tok, "card", "cash", "_arr", "_link")
    session = ScriptedSession(
        len(tok.pieces),
        table({(): {card: L(0.4), cash: L(0.1)}, (card,): {END: L(0.3), arr: L(0.1), link: L(0.2)}}),
    )
    gen = constrained_greedy(session, trie, tok)
    assert gen.text == "card"
    # .4/.5 for "card" times .3/.6 for the end token
    assert gen.token_logprobs == pytest.approx([L(0.8), L(0.5)])
    assert classification_confidence(gen) == pytest.approx(0.4)
    assert session.calls == 2


def test_constrained_ties_pick_lowest_token_id(tok: PieceTokenizer) -> None:
    card, cash = ids(tok, "card", "cash")
    assert card < cash
    trie = LabelTrie.build(tok, ["cash", "card"], end_ids=[END])
    session = ScriptedSession(len(tok.pieces), table({(): {card: L(0.25), cash: L(0.25)}}))
    gen = constrained_greedy(session, trie, tok)
    assert gen.text == "card"
    assert classification_confidence(gen) == pytest.approx(0.5)


def test_constrained_all_allowed_minus_inf_is_uniform(tok: PieceTokenizer) -> None:
    trie = LabelTrie.build(tok, ["card", "cash"], end_ids=[END])
    card, cash = ids(tok, "card", "cash")
    session = ScriptedSession(len(tok.pieces), table({(): {card: -np.inf, cash: -np.inf}}))
    gen = constrained_greedy(session, trie, tok)
    assert gen.text == "card"
    assert classification_confidence(gen) == pytest.approx(0.5)


def test_constrained_single_label_needs_no_query(tok: PieceTokenizer) -> None:
    trie = LabelTrie.build(tok, ["card_arrival"], end_ids=[END])
    session = ScriptedSession(len(tok.pieces), table({}))
    gen = constrained_greedy(session, trie, tok)
    assert gen.text == "card_arrival"
    assert classification_confidence(gen) == 1.0
    assert session.calls == 0


# -- free decoding ----------------------------------------------------------------------------------


def test_free_greedy_stops_on_stop_id(tok: PieceTokenizer) -> None:
    card, arr, ival = ids(tok, "card", "_arr", "ival")
    script = table(
        {
            (): {card: L(0.9)},
            (card,): {arr: L(0.5), END: L(0.4)},
            (card, arr): {ival: L(0.8)},
            (card, arr, ival): {END: L(0.7)},
        }
    )
    session = ScriptedSession(len(tok.pieces), script)
    gen = free_greedy(session, tok, max_tokens=10, stop_ids=[END])
    assert gen.text == "card_arrival"
    assert gen.token_ids == [card, arr, ival, END]
    assert gen.token_logprobs == pytest.approx([L(0.9), L(0.5), L(0.8), L(0.7)], rel=1e-6)
    assert gen.token_spans == [(0, 4), (4, 8), (8, 12), (12, 12)]
    assert gen.finish_reason == "stop"
    assert gen.constrained is False
    assert session.fed == [card, arr, ival]
    expected = math.exp((L(0.9) + L(0.5) + L(0.8) + L(0.7)) / 4)
    assert mean_logprob_confidence(gen) == pytest.approx(expected, rel=1e-6)


def test_free_greedy_stops_on_any_stop_id(tok: PieceTokenizer) -> None:
    card = tok.id("card")
    session = ScriptedSession(len(tok.pieces), table({(): {card: L(0.9)}, (card,): {START: L(0.6)}}))
    gen = free_greedy(session, tok, max_tokens=10, stop_ids=[END, START])
    assert gen.text == "card"
    assert gen.token_ids == [card, START]
    assert gen.finish_reason == "stop"


def test_free_greedy_length_limit(tok: PieceTokenizer) -> None:
    card, arr = ids(tok, "card", "_arr")
    session = ScriptedSession(len(tok.pieces), table({(): {card: L(0.9)}, (card,): {arr: L(0.5)}}))
    gen = free_greedy(session, tok, max_tokens=2, stop_ids=[END])
    assert gen.text == "card_arr"
    assert gen.token_ids == [card, arr]
    assert gen.finish_reason == "length"
    assert gen.token_spans == [(0, 4), (4, 8)]
    assert session.fed == [card]  # the last token is not fed: no further step needs it

    empty = free_greedy(ScriptedSession(len(tok.pieces), table({})), tok, max_tokens=0, stop_ids=[END])
    assert (empty.text, empty.token_ids, empty.finish_reason) == ("", [], "length")
    assert mean_logprob_confidence(empty) == 0.0


def test_free_greedy_immediate_stop(tok: PieceTokenizer) -> None:
    session = ScriptedSession(len(tok.pieces), table({(): {END: L(0.6)}}))
    gen = free_greedy(session, tok, max_tokens=5, stop_ids=[END])
    assert (gen.text, gen.token_ids, gen.token_spans, gen.finish_reason) == ("", [END], [(0, 0)], "stop")
    assert mean_logprob_confidence(gen) == pytest.approx(0.6)


def test_free_greedy_ties_pick_lowest_id(tok: PieceTokenizer) -> None:
    card, cash = ids(tok, "card", "cash")
    session = ScriptedSession(len(tok.pieces), table({(): {cash: L(0.4), card: L(0.4)}, (card,): {END: 0.0}}))
    gen = free_greedy(session, tok, max_tokens=5, stop_ids=[END])
    assert gen.text == "card"


# -- scores -----------------------------------------------------------------------------------------


def make_gen(pieces: Sequence[str], probs: Sequence[float], *, end_prob: float | None = None) -> Generation:
    """A generation from text pieces with contiguous spans (and an optional end token)."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for piece in pieces:
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
    lps = [math.log(p) for p in probs]
    token_ids = list(range(100, 100 + len(pieces)))
    if end_prob is not None:
        spans.append((pos, pos))
        lps.append(math.log(end_prob))
        token_ids.append(END)
    return Generation(
        text="".join(pieces),
        token_ids=token_ids,
        token_logprobs=lps,
        token_spans=spans,
        finish_reason="stop",
        prompt_tokens=0,
    )


def test_classification_and_mean_confidence_by_hand() -> None:
    gen = make_gen(["a", "b"], [0.5, 0.8], end_prob=0.5)
    assert classification_confidence(gen) == pytest.approx(0.2)
    assert mean_logprob_confidence(gen) == pytest.approx((0.5 * 0.8 * 0.5) ** (1 / 3))
    empty = make_gen([], [])
    assert classification_confidence(empty) == 0.0
    assert mean_logprob_confidence(empty) == 0.0
    tiny_positive = Generation("a", [1], [1e-12], [(0, 1)], "stop", 0)
    assert classification_confidence(tiny_positive) == 1.0
    nan = Generation("a", [1], [float("nan")], [(0, 1)], "stop", 0)
    assert classification_confidence(nan) == 0.0
    assert mean_logprob_confidence(nan) == 0.0


# -- JSON spans -------------------------------------------------------------------------------------


def test_json_value_spans_simple() -> None:
    obj, spans = json_value_spans('{"a": 1, "b": "xy"}')
    assert obj == {"a": 1, "b": "xy"}
    assert spans == {"a": (6, 7), "b": (14, 18)}


def test_json_value_spans_leading_prose_and_fence_keep_original_offsets() -> None:
    text = 'Here you go:\n```json\n{"a": "b"}\n```'
    obj, spans = json_value_spans(text)
    assert obj == {"a": "b"}
    assert text.index("{") == 21
    assert spans == {"a": (27, 30)}
    assert text[27:30] == '"b"'


def test_json_value_spans_escapes_nesting_literals() -> None:
    text = (
        '{"s": "q\\"uo}te\\\\", "n": {"x": [1, {"y": "}"}], "z": null},\n'
        '  "t": true, "f": false, "nul": null, "num": -1.5e3, "arr": [ ], "e": ""}'
    )
    obj, spans = json_value_spans(text)
    assert obj == {
        "s": 'q"uo}te\\',
        "n": {"x": [1, {"y": "}"}], "z": None},
        "t": True,
        "f": False,
        "nul": None,
        "num": -1500.0,
        "arr": [],
        "e": "",
    }
    got = {k: text[s:e] for k, (s, e) in spans.items()}
    assert got == {
        "s": '"q\\"uo}te\\\\"',
        "n": '{"x": [1, {"y": "}"}], "z": null}',
        "t": "true",
        "f": "false",
        "nul": "null",
        "num": "-1.5e3",
        "arr": "[ ]",
        "e": '""',
    }
    assert spans["s"] == (6, 18)


def test_json_value_spans_unicode_offsets_are_characters() -> None:
    text = '{"name": "Zoë Ünal", "city": "\\u00e9t\\u00e9"}'
    obj, spans = json_value_spans(text)
    assert obj == {"name": "Zoë Ünal", "city": "été"}
    assert spans["name"] == (9, 19)
    assert text[slice(*spans["city"])] == '"\\u00e9t\\u00e9"'


def test_json_value_spans_whitespace_everywhere() -> None:
    text = '\n{ \t"a" \n : \r\n 12 , "b":[1,2]\n}\n'
    obj, spans = json_value_spans(text)
    assert obj == {"a": 12, "b": [1, 2]}
    assert text[slice(*spans["a"])] == "12"
    assert text[slice(*spans["b"])] == "[1,2]"


def test_json_value_spans_empty_trailing_and_duplicates() -> None:
    assert json_value_spans("{}") == ({}, {})
    assert json_value_spans(" { } ") == ({}, {})
    obj, spans = json_value_spans('{"a": 1} and then } more')
    assert obj == {"a": 1}
    assert spans == {"a": (6, 7)}
    obj, spans = json_value_spans('{"a": 1, "a": 22}')
    assert obj == {"a": 22}
    assert spans == {"a": (14, 16)}


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"a": }',
        '{"a": 1',
        '{"a": 1,}',
        "{a: 1}",
        '{"a" 1}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": 1 "b": 2}',
        '{"a": "unterminated}',
        '{"a": [1, 2}',
        '{"a": 1e999}',
        "",
    ],
)
def test_json_value_spans_invalid(text: str) -> None:
    assert json_value_spans(text) == (None, {})


def test_json_value_spans_prefers_fenced_object() -> None:
    text = 'Use {braces} carefully.\n```json\n{"a": [1]}\n```\nDone.'
    obj, spans = json_value_spans(text)
    assert obj == {"a": [1]}
    assert text[slice(*spans["a"])] == "[1]"
    assert spans["a"] == (text.index("[1]"), text.index("[1]") + 3)


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1, "b": "x"}',
        'Result: {"a": {"b": [1, 2]}} trailing } brace',
        'Use {braces} carefully.\n```json\n{"a": [1]}\n```\nDone.',
        '```\n{"a": "b"}\n```',
        '{"a": "}", "b": "{"}',
        '{"a": 1} {"b": 2}',
        'prefix {not json} then {"a": 1}',
        '{"a": NaN}',
        '{"a": 1e999}',
        "[1, 2]",
        "",
        'Here it is:\n```json\n{"note": "run ```make``` first"}\n```',
        '```json\n{"a": "x```"}',
        '```json\n{"a": 1}\n```\n{"b": 2}',
        '{"a": ' + "[" * 20000,
        '{"a": ' + "[" * 10000 + "]" * 10000 + "}",
    ],
)
def test_json_value_spans_agrees_with_parse_json_output(text: str) -> None:
    from taskdistill.tasks.extraction import parse_json_output

    assert json_value_spans(text)[0] == parse_json_output(text)


@pytest.mark.parametrize(
    "text", ['Here it is:\n```json\n{"note": "run ```make``` first"}\n```', '```json\n{"a": "x```"}']
)
def test_json_value_spans_fenced_object_must_close_inside_the_fence(text: str) -> None:
    # the fence ends at the first ``` inside the string, so the fenced object is unterminated
    assert json_value_spans(text) == (None, {})


def test_json_value_spans_fence_offsets_unchanged_by_the_fence_limit() -> None:
    text = 'Intro {x}\n```json\n{"a": "```"}\n```\n```json\n{"b": 1}\n```'
    # the first fence holding a "{" ends inside the string: no object, the later fence is not used
    assert json_value_spans(text) == (None, {})
    text = 'Intro\n```json\n{"a": [1, 2], "b": "c"}\n```\ntrailing {"z": 0}'
    obj, spans = json_value_spans(text)
    assert obj == {"a": [1, 2], "b": "c"}
    assert spans == {"a": (20, 26), "b": (33, 36)}
    assert (text[20:26], text[33:36]) == ("[1, 2]", '"c"')


@pytest.mark.parametrize(
    "text",
    ['{"a": ' + "[" * 20000, '{"a": ' + "[" * 10000 + "]" * 10000 + "}", "{" + '"a": {' * 20000],
)
def test_json_value_spans_too_deep_is_invalid_not_an_error(text: str) -> None:
    assert json_value_spans(text) == (None, {})
    schema = {"type": "object", "properties": {"a": {"type": "array"}}}
    score = extraction_confidence(make_gen([text], [0.99]), schema)
    assert (score.doc_confidence, score.valid, score.obj, score.field_confidences) == (0.0, False, None, {})


# -- extraction confidence --------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {"vendor": {"type": ["string", "null"]}, "total": {"type": ["number", "null"]}},
    "required": ["vendor", "total"],
    "additionalProperties": False,
}


def test_extraction_field_product_and_document_min() -> None:
    pieces = ['{"', "vendor", '":', ' "', "Example", " Ltd", '",', ' "', "total", '":', " 12", ".5", "}"]
    probs = [0.99, 0.98, 0.97, 0.9, 0.8, 0.7, 0.95, 0.96, 0.99, 0.9, 0.6, 0.5, 0.9]
    gen = make_gen(pieces, probs, end_prob=0.85)
    assert gen.text == '{"vendor": "Example Ltd", "total": 12.5}'
    score = extraction_confidence(gen, SCHEMA)
    assert isinstance(score, ExtractionScore)
    assert score.valid is True
    assert score.obj == {"vendor": "Example Ltd", "total": 12.5}
    # vendor value '"Example Ltd"' overlaps ' "', 'Example', ' Ltd', '",'
    # total value '12.5' overlaps ' 12', '.5'
    assert score.field_confidences == pytest.approx({"vendor": 0.9 * 0.8 * 0.7 * 0.95, "total": 0.6 * 0.5})
    assert list(score.field_confidences) == ["vendor", "total"]
    assert score.doc_confidence == pytest.approx(0.3)
    whole = math.exp(sum(math.log(p) for p in [*probs, 0.85]) / 14)
    assert mean_logprob_confidence(gen) == pytest.approx(whole)


def test_extraction_token_spanning_two_values_counts_for_both() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}
    gen = make_gen(['{"a":', '1,"b":2', "}"], [0.9, 0.5, 0.8])
    score = extraction_confidence(gen, schema)
    assert score.field_confidences == pytest.approx({"a": 0.5, "b": 0.5})
    assert score.doc_confidence == pytest.approx(0.5)


def test_extraction_empty_span_token_inside_value_counts() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    text = '{"a":"é"}'
    # tokens: '{"a":"' (0,6), partial byte of é (6,6), rest of é (6,7), '"}' (7,9), end (9,9)
    gen = Generation(
        text=text,
        token_ids=[1, 2, 3, 4, END],
        token_logprobs=[L(0.9), L(0.5), L(0.8), L(0.7), L(0.6)],
        token_spans=[(0, 6), (6, 6), (6, 7), (7, 9), (9, 9)],
        finish_reason="stop",
        prompt_tokens=0,
    )
    score = extraction_confidence(gen, schema)
    # value '"é"' is (5, 8): all but the end token overlap it
    assert score.field_confidences["a"] == pytest.approx(0.9 * 0.5 * 0.8 * 0.7)


def test_extraction_invalid_json_scores_zero() -> None:
    gen = make_gen(['{"vendor": ', '"Example"', ", "], [0.99, 0.99, 0.99])
    score = extraction_confidence(gen, SCHEMA)
    assert score.doc_confidence == 0.0
    assert score.obj is None
    assert score.valid is False
    assert score.field_confidences == {}
    assert score.errors


def test_extraction_schema_violation_scores_zero() -> None:
    gen = make_gen(['{"vendor": "Example", ', '"total": "twelve"}'], [0.99, 0.99])
    score = extraction_confidence(gen, SCHEMA)
    assert (score.doc_confidence, score.obj, score.valid, score.field_confidences) == (0.0, None, False, {})
    assert score.errors
    missing = extraction_confidence(make_gen(['{"vendor": null}'], [0.99]), SCHEMA)
    assert (missing.doc_confidence, missing.valid) == (0.0, False)


def test_extraction_absent_optional_field_is_left_out() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}
    gen = make_gen(['{"a": ', "7", "}"], [0.9, 0.4, 0.5], end_prob=0.5)
    score = extraction_confidence(gen, schema)
    assert score.field_confidences == pytest.approx({"a": 0.4})
    assert score.doc_confidence == pytest.approx(0.4)
    none_present = extraction_confidence(make_gen(["{", "}"], [0.9, 0.5], end_prob=0.5), schema)
    assert none_present.valid is True
    assert none_present.field_confidences == {}
    assert none_present.doc_confidence == pytest.approx(0.9 * 0.5 * 0.5)


def test_extraction_leading_prose_offsets(tok: PieceTokenizer) -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    gen = make_gen(["Sure: ", '{"a": ', '"b"', "}"], [0.3, 0.9, 0.6, 0.9])
    score = extraction_confidence(gen, schema)
    assert score.field_confidences == pytest.approx({"a": 0.6})


# -- real tokenizer -------------------------------------------------------------------------------

QWEN_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def _qwen_snapshot() -> Path | None:
    """The pinned Qwen snapshot from the local Hugging Face cache (honours HF_HUB_CACHE / HF_HOME); never downloads."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    from taskdistill.models import KNOWN_REVISIONS

    try:
        path = huggingface_hub.snapshot_download(QWEN_REPO, revision=KNOWN_REVISIONS[QWEN_REPO], local_files_only=True)
    except (LocalEntryNotFoundError, OSError):
        return None
    snapshot = Path(path)
    if (snapshot / "tokenizer.json").is_file() and (snapshot / "tokenizer_config.json").is_file():
        return snapshot
    return None


def test_qwen_snapshot_resolution_honours_hub_cache_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub import constants

    from taskdistill.models import KNOWN_REVISIONS

    revision = KNOWN_REVISIONS[QWEN_REPO]
    repo_dir = tmp_path / f"models--{QWEN_REPO.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    unpinned = repo_dir / "snapshots" / ("0" * 40)  # sorts first, but is not the pinned revision
    unpinned.mkdir()
    (unpinned / "tokenizer.json").write_text("{}")
    (unpinned / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    assert _qwen_snapshot() is None  # the pinned snapshot exists but holds no tokenizer files
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "tokenizer_config.json").write_text("{}")
    assert _qwen_snapshot() == snapshot
    other = tmp_path / "empty"
    other.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(other))
    assert _qwen_snapshot() is None


@pytest.fixture(scope="module")
def qtok() -> Any:
    snapshot = _qwen_snapshot()
    if snapshot is None:
        pytest.skip("the pinned Qwen2.5-0.5B tokenizer is not in the local Hugging Face cache")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)


@pytest.mark.slow
def test_qwen_tokenizer_label_trie(qtok: Any) -> None:
    im_end = 151645
    assert qtok.convert_tokens_to_ids("<|im_end|>") == im_end

    def prompt_ids(messages: list[dict[str, str]]) -> list[int]:
        out = qtok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return list(out if isinstance(out, list) else out["input_ids"])

    labels = ["card", "card_arrival", "cash_withdrawal_charge"]
    trie = LabelTrie.build(qtok, labels, end_ids=[im_end], prompt_ids_fn=prompt_ids)
    for label in labels:
        seq = trie.sequences[label]
        assert seq[-1] == im_end
        assert im_end not in seq[:-1]
        assert qtok.decode(list(seq[:-1])) == label
    card, arrival = trie.sequences["card"], trie.sequences["card_arrival"]
    assert arrival[: len(card) - 1] == card[:-1]  # "card" is a token prefix of "card_arrival"
    assert trie.allowed(card[:-1]) == sorted({im_end, arrival[len(card) - 1]})
    assert trie.allowed([]) == sorted({card[0], trie.sequences["cash_withdrawal_charge"][0]})

    vocab = 151936
    session = ScriptedSession(vocab, table({(card[0],): {im_end: L(0.3), arrival[1]: L(0.6)}, (): {card[0]: L(0.9)}}))
    gen = constrained_greedy(session, trie, qtok)
    assert gen.text == "card_arrival"
    assert gen.token_ids == list(arrival)
    # root: .9 against the "cash" token's 1e-6; after "card": .6 / (.6 + .3); the rest is forced
    assert classification_confidence(gen) == pytest.approx(0.9 / (0.9 + 1e-6) * 0.6 / 0.9, rel=1e-5)
    assert gen.token_spans[-1] == (len("card_arrival"), len("card_arrival"))


@pytest.mark.slow
def test_qwen_tokenizer_spans_cover_multibyte_text(qtok: Any) -> None:
    text = '{"vendor": "Zoë Ünal 龘 Ltd", "total": 12.5}'
    tokens = qtok.encode(text, add_special_tokens=False)
    assert qtok.decode(tokens) == text
    spans = token_char_spans(qtok, tokens)
    assert len(spans) == len(tokens)
    assert any(s == e for s, e in spans)  # "龘" is split over byte-level tokens
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(a[1] == b[0] and a[0] <= a[1] for a, b in itertools.pairwise(spans))
    assert "".join(text[s:e] for s, e in spans) == text

    schema = {"type": "object", "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}}}
    gen = Generation(text, tokens, [L(0.9)] * len(tokens), spans, "stop", 0)
    score = extraction_confidence(gen, schema)
    _, value_spans = json_value_spans(text)
    for name, (vs, ve) in value_spans.items():
        covering = [i for i, (s, e) in enumerate(spans) if (s < ve and e > vs) or (s == e and vs <= s < ve)]
        assert score.field_confidences[name] == pytest.approx(0.9 ** len(covering))
    assert score.doc_confidence == pytest.approx(min(score.field_confidences.values()))
