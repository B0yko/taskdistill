"""The torch backend on CPU with the tiny offline Qwen2 model (no downloads)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from taskdistill.backends import torch_backend  # noqa: E402
from taskdistill.backends.torch_backend import TorchBackend, TorchSession, resolve_torch_base  # noqa: E402
from taskdistill.confidence import LabelTrie, classification_confidence  # noqa: E402
from tiny_model import IM_END_ID, build_tiny_model  # noqa: E402

MESSAGES = [
    {"role": "system", "content": "Classify the customer's banking message into exactly one intent label."},
    {"role": "user", "content": "My card has not arrived yet. When will my new card arrive?"},
]
LABELS = ["card", "card arrival", "cash withdrawal"]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def backend(tiny_dir: Path) -> TorchBackend:
    b = TorchBackend(str(tiny_dir))
    b.load()
    return b


def _normalised(logprobs: np.ndarray) -> bool:
    return math.isclose(float(np.exp(logprobs.astype(np.float64)).sum()), 1.0, abs_tol=1e-4)


def test_tiny_model_loads_on_cpu(backend: TorchBackend) -> None:
    assert backend.loader == "transformers"
    assert backend.device == "cpu"
    assert backend.stop_ids[0] == IM_END_ID
    assert backend.tokenizer.convert_ids_to_tokens(IM_END_ID) == "<|im_end|>"
    assert backend.logits_dtype == "float32"
    head = backend.model.get_output_embeddings()
    assert head.weight is backend.model.get_input_embeddings().weight, "a float32 head is left as it is"


def test_tiny_model_builder_is_deterministic(tmp_path: Path) -> None:
    a = build_tiny_model(tmp_path / "a", seed=3)
    b = build_tiny_model(tmp_path / "b", seed=3)
    for name in ("model.safetensors", "tokenizer.json", "config.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_prompt_ids_end_with_the_assistant_generation_prompt(backend: TorchBackend) -> None:
    ids = backend.prompt_ids(MESSAGES)
    tail = backend.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assert ids[-len(tail) :] == tail
    text = backend.tokenizer.decode(ids)
    assert text.startswith("<|im_start|>system\nClassify")
    assert text.endswith("<|im_end|>\n<|im_start|>assistant\n")


def test_logprobs_are_normalised_and_the_cache_matches_a_full_forward(backend: TorchBackend) -> None:
    prompt = backend.prompt_ids(MESSAGES)
    session = backend.start(prompt)
    first = session.logprobs()
    assert first.dtype == np.float32
    assert first.shape == (backend.model.config.vocab_size,)
    assert _normalised(first)

    fed = [int(np.argmax(first)), 5, 7]
    for token in fed:
        session.feed(token)
    incremental = session.logprobs()
    assert _normalised(incremental)
    fresh = backend.start(prompt + fed).logprobs()
    np.testing.assert_allclose(incremental, fresh, atol=1e-4)


def test_free_greedy_generation_terminates(backend: TorchBackend) -> None:
    gen = backend.generate_with_scores(MESSAGES, max_tokens=12)
    assert gen.finish_reason in {"stop", "length"}
    assert 1 <= len(gen.token_ids) <= 12
    assert len(gen.token_logprobs) == len(gen.token_ids)
    assert all(lp <= 1e-6 for lp in gen.token_logprobs)
    assert gen.prompt_tokens == len(backend.prompt_ids(MESSAGES))
    assert not gen.constrained


def _path_probability(backend: TorchBackend, trie: LabelTrie, label: str) -> float:
    """Product of the renormalised probabilities along ``label``'s path through the trie."""
    session = backend.start(backend.prompt_ids(MESSAGES))
    node, prob = trie.root, 1.0
    for token in trie.sequences[label]:
        allowed = sorted(node.children)
        if len(allowed) > 1:
            lp = session.logprobs().astype(np.float64)[allowed]
            p = np.exp(lp - lp.max())
            prob *= float(p[allowed.index(token)] / p.sum())
        node = node.children[token]
        if token != trie.end_id:
            session.feed(token)
    return prob


def test_generate_with_scores_under_a_label_trie_with_a_prefix_label(backend: TorchBackend) -> None:
    trie = backend.label_trie(LABELS)
    card, arrival = trie.sequences["card"], trie.sequences["card arrival"]
    assert arrival[: len(card) - 1] == card[:-1], "'card' must be a token prefix of 'card arrival'"
    assert all(seq[-1] == IM_END_ID for seq in trie.sequences.values())

    gen = backend.generate_with_scores(MESSAGES, trie)
    assert gen.constrained
    assert gen.text in LABELS
    assert gen.token_ids == list(trie.sequences[gen.text])
    confidence = classification_confidence(gen)
    assert 0.0 <= confidence <= 1.0

    probs = {label: _path_probability(backend, trie, label) for label in LABELS}
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-4)
    assert math.isclose(confidence, probs[gen.text], rel_tol=1e-3)


def test_local_directories_load_as_they_are(tiny_dir: Path) -> None:
    assert resolve_torch_base(str(tiny_dir), "abc") == (str(tiny_dir), None)


def test_mlx_repositories_map_to_their_torch_equivalents() -> None:
    with pytest.warns(UserWarning, match="Qwen/Qwen2.5-0.5B-Instruct"):
        assert resolve_torch_base("mlx-community/Qwen2.5-0.5B-Instruct-4bit", "a" * 40) == (
            "Qwen/Qwen2.5-0.5B-Instruct",
            None,
        )
    with pytest.raises(ValueError, match="no known torch equivalent"):
        resolve_torch_base("mlx-community/SomeModel-4bit")
    assert resolve_torch_base("Qwen/Qwen2.5-1.5B-Instruct", "main") == ("Qwen/Qwen2.5-1.5B-Instruct", "main")


def test_the_backend_loads_lazily_on_first_use(tiny_dir: Path) -> None:
    lazy = TorchBackend(str(tiny_dir))
    assert not lazy.loaded
    gen = lazy.generate_with_scores(MESSAGES, max_tokens=2)
    assert lazy.loaded
    assert gen.prompt_tokens > 0


class _PositionsRequired:
    """Stands in for Unsloth's cached forward, which reads ``position_ids.max()`` whenever a cache is passed."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.positions: list[list[int] | None] = []

    def __call__(self, *, input_ids: Any, past_key_values: Any = None, position_ids: Any = None, **kwargs: Any) -> Any:
        if past_key_values is not None and position_ids is None:
            raise AttributeError("'NoneType' object has no attribute 'max'")
        self.positions.append(None if position_ids is None else position_ids[0].tolist())
        return self.model(input_ids=input_ids, past_key_values=past_key_values, position_ids=position_ids, **kwargs)


def test_decoding_passes_explicit_position_ids(backend: TorchBackend) -> None:
    model = _PositionsRequired(backend.model)
    prompt = backend.prompt_ids(MESSAGES)
    session = TorchSession(model, prompt, backend.device)
    fed = [5, 7, 9]
    for token in fed:
        session.feed(token)
    n = len(prompt)
    assert model.positions == [list(range(n)), [n], [n + 1], [n + 2]]
    np.testing.assert_allclose(session.logprobs(), backend.start(prompt + fed).logprobs(), atol=1e-4)


def test_trie_and_free_decoding_run_on_a_forward_that_needs_position_ids(tiny_dir: Path) -> None:
    b = TorchBackend(str(tiny_dir))
    b.load()
    b.model = _PositionsRequired(b.model)
    gen = b.generate_with_scores(MESSAGES, b.label_trie(LABELS))
    assert gen.text in LABELS
    free = b.generate_with_scores(MESSAGES, max_tokens=4)
    assert 1 <= len(free.token_ids) <= 4
    assert all(p is not None for p in b.model.positions)


def test_a_bfloat16_model_computes_float32_logits(tiny_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_backend, "torch_device", lambda: ("cpu", torch.bfloat16))
    b = TorchBackend(str(tiny_dir))
    b.load()
    model = b.model
    head = model.get_output_embeddings()
    embed = model.get_input_embeddings().weight
    assert b.logits_dtype == "float32"
    assert (head.weight.dtype, embed.dtype) == (torch.float32, torch.bfloat16)
    assert torch.equal(head.weight, embed.float())

    prompt = b.prompt_ids(MESSAGES)
    session = b.start(prompt)
    with torch.no_grad():
        hidden = model.model(input_ids=torch.tensor([prompt])).last_hidden_state[0, -1]
        logits = hidden.float() @ embed.float().T
    np.testing.assert_allclose(session.logprobs(), torch.log_softmax(logits, -1).numpy(), atol=1e-5)
    rounded = torch.log_softmax(logits.to(torch.bfloat16).float(), -1).numpy()
    assert not np.allclose(session.logprobs(), rounded, atol=1e-4), "logits must not be rounded to bfloat16"

    dtypes: list[Any] = []
    handle = head.register_forward_hook(lambda _m, _args, out: dtypes.append(out.dtype))
    try:
        session.feed(5)
    finally:
        handle.remove()
    assert dtypes == [torch.float32]
    assert math.isclose(float(np.exp(session.logprobs().astype(np.float64)).sum()), 1.0, abs_tol=1e-4)
