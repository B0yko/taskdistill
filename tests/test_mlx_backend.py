"""MLX backend and MLX LoRA training on the cached 0.5B student base (``pytest -m mlx``, Apple Silicon only)."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from taskdistill.backends.factory import load_backend
from taskdistill.backends.mlx_backend import MLXBackend
from taskdistill.confidence import classification_confidence, mean_logprob_confidence
from taskdistill.config import load_task
from taskdistill.models import KNOWN_REVISIONS

pytest.importorskip("mlx.core", reason="MLX is not installed on this machine")
pytest.importorskip("mlx_lm", reason="mlx-lm is not installed on this machine")

pytestmark = pytest.mark.mlx

REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
LABELS = ["card", "card arrival", "cash"]  # "card" is a token prefix of "card arrival"
SYSTEM = "Classify the banking message."
MESSAGES = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "My new card still has not arrived."}]
TRAIN_LOG_KEYS = {
    "run_id", "task", "backend", "base_model", "base_revision", "profile", "seed", "labels", "n_train", "n_valid",
    "dropped_no_gold", "iterations", "epochs", "batch_size", "learning_rate", "lora_rank", "lora_layers",
    "max_seq_len", "wall_seconds", "train_seconds", "peak_memory_gb", "tokens_per_second", "trained_tokens",
    "processed_tokens", "curve", "best_iteration", "best_val_loss", "final_val_loss", "adapter_dir",
    "adapter_size_mb", "hardware", "versions", "date", "load_average",
}  # fmt: skip


def _model_cached() -> bool:
    import huggingface_hub

    try:
        path = Path(huggingface_hub.snapshot_download(REPO, revision=KNOWN_REVISIONS[REPO], local_files_only=True))
    except Exception:
        return False
    return any(path.glob("*.safetensors"))


if not _model_cached():
    pytest.skip("the 0.5B student base is not in the local Hugging Face cache", allow_module_level=True)


# -- training through the real subprocess path (runs first, before this process loads a model) --------

QUERIES = {
    "card": [
        "I want to order a replacement card",
        "Can I get a second card for my partner",
        "How do I activate the card you sent",
        "My card was damaged, I need a new card",
        "Please freeze my card for now",
        "Which card types do you offer",
    ],
    "card arrival": [
        "My new card still has not arrived",
        "When will my card be delivered",
        "The card I ordered two weeks ago never came",
        "Is my card in the post yet",
        "How long does card delivery take",
        "Still waiting for the card to arrive",
    ],
    "cash": [
        "The ATM did not give me my cash",
        "How much cash can I take out per day",
        "Is there a fee for cash withdrawals",
        "Where can I withdraw cash abroad",
        "The machine kept my cash",
        "I need cash from an ATM tonight",
    ],
}


def _example(text: str, label: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": label},
        ]
    }


def _write_workspace(root: Path, home: Path) -> Any:
    task_dir = root / "tasks" / "tiny"
    task_dir.mkdir(parents=True)
    (task_dir / "teacher_prompt.md").write_text("Pick one label.\n", encoding="utf-8")
    (task_dir / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    spec = {
        "task": "tiny",
        "type": "classification",
        "labels_file": "labels.txt",
        "teacher": {"model": "example/teacher-model"},
        "student": {"base_model": REPO, "system_prompt": SYSTEM},
        "train": {"profile": "full", "lora_rank": 8, "lora_layers": 8, "learning_rate": 1.0e-4, "batch_size": 8,
                  "epochs": 2.5, "max_seq_len": 256, "seed": 3},
        "cascade": {"target": 0.9},
    }  # fmt: skip
    (task_dir / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    pool = [(text, label) for label, texts in QUERIES.items() for text in texts]
    variants = ["", " please", " today", " thanks"]
    train = [(f"{text}{suffix}", label) for suffix in variants[:2] for text, label in pool][:32]
    valid = [(f"{text}{variants[k % 2 + 2]}", label) for k, (text, label) in enumerate(pool[::2])][:8]
    data = home / "tiny" / "data"
    data.mkdir(parents=True)
    for split, rows in (("train", train), ("valid", valid)):
        with (data / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for text, label in rows:
                fh.write(json.dumps(_example(text, label)) + "\n")
        with (data / f"{split}.meta.jsonl").open("w", encoding="utf-8") as fh:
            for i, (_, label) in enumerate(rows):
                fh.write(json.dumps({"input_hash": f"{split}-{i}", "gold": label, "teacher": label, "meta": {}}) + "\n")
    return load_task(str(task_dir))


def test_train_ten_iterations_then_score_with_the_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from taskdistill.train.runner import run_training

    home = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    spec = _write_workspace(tmp_path, home)
    lines: list[str] = []
    result = run_training(spec, backend="mlx", log=lines.append)

    log = result.log
    assert set(log) >= TRAIN_LOG_KEYS
    assert result.run_id == "qwen2.5-0.5b-full-s3"
    assert (log["n_train"], log["n_valid"], log["iterations"], log["batch_size"]) == (32, 8, 10, 8)
    assert log["epochs"] == pytest.approx(2.5)
    assert log["base_revision"] == KNOWN_REVISIONS[REPO]
    assert log["peak_memory_gb"] > 0 and log["tokens_per_second"] > 0
    assert log["processed_tokens"] > log["trained_tokens"] > 0
    assert log["wall_seconds"] >= log["train_seconds"] > 0
    val_iterations = [it for it, _ in log["curve"]["val"]]
    assert val_iterations[0] == 0 and val_iterations[-1] == 10  # base model first, final weights last
    assert log["final_val_loss"] == log["curve"]["val"][-1][1]
    assert log["best_iteration"] in val_iterations
    assert log["best_val_loss"] == min(loss for _, loss in log["curve"]["val"])
    assert log["curve"]["train"] and all(math.isfinite(loss) for _, loss in log["curve"]["train"])
    assert log["versions"]["mlx"] and log["versions"]["mlx-lm"]
    assert log["adapter_dir"] == "tiny/runs/qwen2.5-0.5b-full-s3/adapter"
    assert log["n_train_planned"] == 32 and log["dropped_too_long"] == {"train": 0, "valid": 0}
    for name in ("train_log.json", "train_config.json", "adapter/adapter_config.json"):
        text = (result.run_dir / name).read_text(encoding="utf-8")
        assert str(tmp_path) not in text and str(Path.home()) not in text, name

    adapter = result.run_dir / "adapter"
    assert sorted(p.name for p in adapter.iterdir()) == ["adapter_config.json", "adapters.safetensors"]
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert (adapter_config["num_layers"], adapter_config["base_model"]) == (8, REPO)
    assert (result.run_dir / "loss.png").is_file()
    assert not (result.run_dir / "_last_adapters.safetensors").exists()
    assert not (result.run_dir / "adapter.partial").exists()
    selected = json.loads((home / "tiny" / "selected_run.json").read_text(encoding="utf-8"))
    assert selected["run_id"] == result.run_id
    assert any("adapter saved" in line for line in lines)

    with pytest.raises(ValueError, match=r"trained on mlx-community/Qwen2\.5-0\.5B-Instruct-4bit"):
        MLXBackend("example-org/another-base-4bit", adapter_path=str(adapter)).load()  # refused before any download
    backend = MLXBackend(REPO, adapter_path=str(adapter))
    backend.load()
    trie = backend.label_trie(LABELS)
    with ThreadPoolExecutor(max_workers=1) as pool:  # loaded here, first run on a worker (as the serve thread does)
        gen = pool.submit(backend.generate_with_scores, MESSAGES, constraint=trie).result(timeout=120)
    assert gen.text in LABELS and gen.constrained
    assert 0.0 <= classification_confidence(gen) <= 1.0
    on_main = backend.generate_with_scores(MESSAGES, constraint=trie)
    assert on_main.token_ids == gen.token_ids and on_main.token_logprobs == pytest.approx(gen.token_logprobs)


# -- backend on the base model ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def backend() -> MLXBackend:
    instance = load_backend("mlx", REPO)
    assert isinstance(instance, MLXBackend) and not instance.loaded
    instance.load()
    return instance


def test_stop_ids_and_vocabulary(backend: MLXBackend) -> None:
    im_end = backend.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert backend.stop_ids == [im_end, 151643] and im_end == 151645  # <|im_end|>, then <|endoftext|>
    assert backend.vocab_size == 151936 > len(backend.tokenizer)


def test_prompt_ids_match_the_chat_template(backend: MLXBackend) -> None:
    ids = backend.prompt_ids(MESSAGES)
    expected = backend.tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, return_dict=False)
    assert ids == list(expected)
    assert backend.tokenizer.decode(ids).endswith("<|im_start|>assistant\n")


def test_logprobs_are_a_float32_distribution_over_the_model_vocab(backend: MLXBackend) -> None:
    session = backend.start(backend.prompt_ids(MESSAGES))
    for _ in range(3):
        lp = session.logprobs()
        assert lp.dtype == np.float32 and lp.shape == (151936,)
        assert abs(float(np.exp(lp.astype(np.float64)).sum()) - 1.0) < 1e-3
        session.feed(int(np.argmax(lp)))


def test_chunked_prefill_matches_a_single_prefill(backend: MLXBackend) -> None:
    ids = backend.prompt_ids(MESSAGES)
    whole = backend.start(ids).logprobs()
    chunked = MLXBackend(REPO, prefill_step=7)
    chunked.model, chunked.tokenizer, chunked._wrapper = backend.model, backend.tokenizer, backend._wrapper
    parts = chunked.start(ids).logprobs()
    assert int(np.argmax(parts)) == int(np.argmax(whole))
    assert float(np.max(np.abs(np.exp(parts) - np.exp(whole)))) < 0.01  # float16 activations differ slightly


def test_free_greedy_matches_mlx_lm(backend: MLXBackend) -> None:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    gen = backend.generate_with_scores(MESSAGES, max_tokens=12)
    assert gen.finish_reason in {"stop", "length"} and gen.text.strip()
    assert len(gen.token_ids) == len(gen.token_logprobs) and gen.prompt_tokens == len(backend.prompt_ids(MESSAGES))
    assert all(lp <= 0.0 for lp in gen.token_logprobs)
    assert 0.0 <= mean_logprob_confidence(gen) <= 1.0

    reference = []
    for token, _ in generate_step(mx.array(backend.prompt_ids(MESSAGES)), backend.model, max_tokens=len(gen.token_ids)):
        reference.append(int(token))
        if token in backend.stop_ids:
            break
    assert gen.token_ids == reference


def test_label_trie_confidence(backend: MLXBackend) -> None:
    trie = backend.label_trie(LABELS)
    gen = backend.generate_with_scores(MESSAGES, constraint=trie)
    assert gen.text in LABELS
    assert gen.token_ids[-1] == backend.stop_ids[0]
    assert 0.0 <= classification_confidence(gen) <= 1.0
    other = backend.generate_with_scores(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "The ATM kept my cash"}], constraint=trie
    )
    assert other.text in LABELS


def test_load_and_generate_on_one_worker_thread() -> None:
    def work() -> tuple[str, float]:
        worker = MLXBackend(REPO)
        worker.load()
        gen = worker.generate_with_scores(MESSAGES, constraint=worker.label_trie(LABELS))
        return gen.text, classification_confidence(gen)

    with ThreadPoolExecutor(max_workers=1) as pool:
        label, confidence = pool.submit(work).result(timeout=120)
    assert label in LABELS and 0.0 <= confidence <= 1.0


def test_sliced_loss_equals_full_loss(backend: MLXBackend) -> None:
    import mlx.core as mx
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset

    from taskdistill.train.mlx_lora import BatchIterator, make_loss, supports_sliced_logits

    rows = [_example(text, label) for label, texts in QUERIES.items() for text in texts[:2]]
    dataset = CacheDataset(ChatDataset(rows, backend._wrapper, mask_prompt=True))
    batch = next(BatchIterator(seed=0)(dataset, batch_size=6, max_seq_length=256))
    assert supports_sliced_logits(backend.model)
    sliced, n_sliced = make_loss(backend.model, sliced=True)(backend.model, *batch)
    full, n_full = make_loss(backend.model, sliced=False)(backend.model, *batch)
    mx.eval(sliced, full, n_sliced, n_full)
    assert int(n_sliced.item()) == int(n_full.item()) > 0
    assert float(sliced.item()) == pytest.approx(float(full.item()), rel=2e-3)
