"""LoRA training on the torch path (transformers + PEFT) on CPU with the tiny offline Qwen2 model."""

from __future__ import annotations

import json
import math
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
safetensors_torch = pytest.importorskip("safetensors.torch")

from taskdistill import paths  # noqa: E402
from taskdistill.backends.torch_backend import TorchBackend  # noqa: E402
from taskdistill.confidence import classification_confidence  # noqa: E402
from taskdistill.train import torch_lora  # noqa: E402
from taskdistill.train.common import plan_training  # noqa: E402
from tiny_model import (  # noqa: E402
    IM_END_ID,
    TOY_LABELS,
    TOY_SYSTEM_PROMPT,
    build_tiny_model,
    write_toy_classification_task,
)

LOG_KEYS = {
    "run_id",
    "task",
    "backend",
    "base_model",
    "base_revision",
    "profile",
    "seed",
    "labels",
    "n_train",
    "n_valid",
    "dropped_no_gold",
    "iterations",
    "epochs",
    "batch_size",
    "learning_rate",
    "lora_rank",
    "lora_layers",
    "max_seq_len",
    "wall_seconds",
    "train_seconds",
    "peak_memory_gb",
    "peak_memory_source",
    "tokens_per_second",
    "trained_tokens",
    "processed_tokens",
    "curve",
    "best_iteration",
    "best_val_loss",
    "final_val_loss",
    "adapter_dir",
    "adapter_size_mb",
    "hardware",
    "versions",
    "date",
    "load_average",
}


def _messages(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": TOY_SYSTEM_PROMPT}, {"role": "user", "content": text}]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Iterator[types.SimpleNamespace]:
    root = tmp_path_factory.mktemp("torch-train")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TASKDISTILL_HOME", str(root / "home"))
        tiny = build_tiny_model(root / "tiny")
        spec = write_toy_classification_task(root / "tasks", str(tiny))
        cfg = plan_training(spec, backend="torch", profile="quick")
        lines: list[str] = []
        log = torch_lora.train_torch(cfg, spec, log=lines.append)
        yield types.SimpleNamespace(root=root, tiny=tiny, spec=spec, cfg=cfg, log=log, lines=lines)


def test_the_plan_matches_the_toy_data(workspace: Any) -> None:
    cfg = workspace.cfg
    assert (cfg.backend, cfg.n_train, cfg.n_valid, cfg.batch_size) == ("torch", 16, 8, 4)
    assert cfg.iters == 8  # ceil(2 epochs * 16 / 4)
    assert cfg.steps_per_eval == 2
    assert cfg.run_id == "tiny-quick-s13-torch"


def test_training_writes_the_adapter_log_and_plot(workspace: Any) -> None:
    run_dir = Path(workspace.cfg.run_dir)
    assert (run_dir / "adapter" / "adapter_config.json").is_file()
    assert (run_dir / "adapter" / "adapter_model.safetensors").is_file()
    assert (run_dir / "loss.png").is_file()
    on_disk = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
    assert on_disk == workspace.log
    assert set(on_disk) >= LOG_KEYS

    adapter_config = json.loads((run_dir / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_config["r"] == 4
    assert adapter_config["lora_alpha"] == 80  # mlx-lm scale 20 x rank 4
    assert adapter_config["target_modules"] == sorted(adapter_config["target_modules"])
    for name in ("train_log.json", "adapter/adapter_config.json", "adapter/README.md"):
        assert str(workspace.root) not in (run_dir / name).read_text(encoding="utf-8"), name


def test_the_log_records_the_run(workspace: Any) -> None:
    log = workspace.log
    assert log["backend"] == "torch"
    assert log["loader"] == "transformers"
    assert log["device"] == "cpu"
    assert (log["iterations"], log["n_train"], log["n_valid"], log["epochs"]) == (8, 16, 8, 2.0)
    assert log["dropped_no_gold"] == 0
    assert log["adapter_dir"] == "toy/runs/tiny-quick-s13-torch/adapter"
    assert log["adapter_size_mb"] > 0
    assert log["peak_memory_gb"] > 0 and "ru_maxrss" in log["peak_memory_source"]
    assert log["processed_tokens"] > log["trained_tokens"] > 0
    assert log["tokens_per_second"] > 0
    assert {"torch", "transformers", "peft", "python", "taskdistill"} <= set(log["versions"])
    assert (log["batch_size"], log["dtype"], log["loss_scaling"]) == (4, "float32", None)

    val = log["curve"]["val"]
    assert [it for it, _ in val] == [0, 2, 4, 6, 8]
    assert [it for it, _ in log["curve"]["train"]] == [2, 4, 6, 8]
    best_iteration, best_loss = min(val, key=lambda p: (p[1], p[0]))
    assert (log["best_iteration"], log["best_val_loss"]) == (best_iteration, best_loss)
    assert log["final_val_loss"] == val[-1][1]
    assert log["best_val_loss"] < val[0][1], "a few steps at lr 5e-3 must lower the validation loss"


def test_the_reloaded_adapter_reproduces_the_logged_best_validation_loss(workspace: Any) -> None:
    backend = TorchBackend(str(workspace.tiny), adapter_path=str(Path(workspace.cfg.run_dir) / "adapter"))
    backend.load()
    rows, _ = torch_lora._encode_split(backend.tokenizer, Path(workspace.cfg.data_dir) / "valid.jsonl", 256)
    batches = [torch_lora._collate(rows[i : i + 4], 0, "cpu") for i in range(0, len(rows), 4)]
    loss = torch_lora._validation_loss(backend.model, batches)
    assert math.isclose(loss, workspace.log["best_val_loss"], rel_tol=1e-4)


def test_the_reloaded_adapter_generates(workspace: Any) -> None:
    backend = TorchBackend(str(workspace.tiny), adapter_path=str(Path(workspace.cfg.run_dir) / "adapter"))
    backend.load()
    assert type(backend.model).__name__.startswith("Peft")
    messages = _messages("When will my card arrive")
    trie = backend.label_trie(list(TOY_LABELS))
    gen = backend.generate_with_scores(messages, trie)
    assert gen.text in TOY_LABELS
    assert 0.0 <= classification_confidence(gen) <= 1.0
    free = backend.generate_with_scores(messages, max_tokens=8)
    assert free.finish_reason in {"stop", "length"}
    assert 1 <= len(free.token_ids) <= 8


def test_completion_only_masking(workspace: Any) -> None:
    backend = TorchBackend(str(workspace.tiny))
    backend.load()
    tokenizer = backend.tokenizer
    prompt = _messages("My new card has not arrived")
    ids, labels = torch_lora.encode_example(tokenizer, [*prompt, {"role": "assistant", "content": "card arrival"}])
    prompt_ids = backend.prompt_ids(prompt)
    n = len(prompt_ids)
    assert ids[:n] == prompt_ids
    assert labels[:n] == [torch_lora.IGNORE_INDEX] * n
    assert labels[n:] == ids[n:]
    label_ids = tokenizer.encode("card arrival", add_special_tokens=False)
    assert ids[n : n + len(label_ids) + 1] == [*label_ids, IM_END_ID]
    assert tokenizer.decode(ids[n:]) == "card arrival<|im_end|>\n"


def test_batch_schedule_uses_full_seeded_batches_that_wrap_into_the_next_epoch() -> None:
    batches = torch_lora.batch_schedule(10, 4, 5, seed=13)
    assert all(len(b) == 4 for b in batches)
    flat = [i for b in batches for i in b]
    assert sorted(flat[:10]) == list(range(10))
    assert sorted(flat[10:]) == list(range(10))
    assert flat[:10] != flat[10:]
    assert batches == torch_lora.batch_schedule(10, 4, 5, seed=13)
    assert batches != torch_lora.batch_schedule(10, 4, 5, seed=14)
    assert torch_lora.batch_schedule(3, 8, 1, seed=0)[0].count(0) >= 2  # batch larger than the data


def test_lora_layer_selection_and_scale() -> None:
    assert torch_lora.layers_to_transform(24, "all") is None
    assert torch_lora.layers_to_transform(24, 8) == list(range(16, 24))
    assert torch_lora.layers_to_transform(24, 24) is None
    assert torch_lora.layers_to_transform(24, 30) is None
    # the MLX trainer's rule, max(1, min(lora_layers, blocks)): zero and negatives train the last block
    assert torch_lora.layers_to_transform(24, 0) == [23]
    assert torch_lora.layers_to_transform(24, -1) == [23]
    assert torch_lora.lora_alpha_for(16) == 320


def test_lora_layers_restricts_adapters_to_the_last_blocks(tmp_path: Path) -> None:
    tiny = build_tiny_model(tmp_path / "tiny")
    cfg = types.SimpleNamespace(
        base_model=str(tiny), base_revision=None, lora_rank=4, lora_layers=1, max_seq_len=128, seed=0
    )
    model, _tokenizer, device, info = torch_lora._load_for_training(cfg, "transformers")
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert device == "cpu"
    assert info["lora_blocks"] == 1
    assert len(names) == 14  # 7 projections x (A, B) in the last block only
    assert all(".layers.1." in name and "lora_" in name for name in names)


def test_training_is_reproducible_for_a_seed(workspace: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    spec = write_toy_classification_task(tmp_path / "tasks", str(workspace.tiny))
    cfg = plan_training(spec, backend="torch", profile="quick")
    log = torch_lora.train_torch(cfg, spec, log=lambda _line: None)
    assert log["curve"] == workspace.log["curve"]
    first = Path(workspace.cfg.run_dir) / "adapter"
    again = Path(cfg.run_dir) / "adapter"
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        assert (again / name).read_bytes() == (first / name).read_bytes(), name

    other = plan_training(spec, backend="torch", profile="quick", seed=14)
    assert other.run_id == "tiny-quick-s14-torch"
    assert torch_lora.train_torch(other, spec, log=lambda _line: None)["curve"] != log["curve"]


def test_the_best_validation_checkpoint_is_restored_before_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    tiny = build_tiny_model(tmp_path / "tiny")
    spec = write_toy_classification_task(tmp_path / "tasks", str(tiny))
    cfg = plan_training(spec, backend="torch", profile="quick")  # validation at 0, 2, 4, 6 and 8
    scripted = iter([(0, 5.0), (2, 3.0), (4, 1.0), (6, 2.0), (8, 4.0)])
    snapshots: dict[int, dict[str, Any]] = {}

    def scripted_loss(model: Any, _batches: Any) -> float:
        iteration, loss = next(scripted)
        snapshots[iteration] = torch_lora._trainable_state(model)
        return loss

    monkeypatch.setattr(torch_lora, "_validation_loss", scripted_loss)
    log = torch_lora.train_torch(cfg, spec, log=lambda _line: None)

    assert log["curve"]["val"] == [[0, 5.0], [2, 3.0], [4, 1.0], [6, 2.0], [8, 4.0]]
    assert (log["best_iteration"], log["best_val_loss"], log["final_val_loss"]) == (4, 1.0, 4.0)
    best, final = snapshots[4], snapshots[8]
    assert any(not torch.equal(best[name], final[name]) for name in best), "training must move the weights"
    saved = safetensors_torch.load_file(str(Path(cfg.run_dir) / "adapter" / "adapter_model.safetensors"))
    expected = {name.replace(".default.", "."): tensor for name, tensor in best.items()}
    assert set(saved) == set(expected)
    for name, tensor in saved.items():
        assert torch.equal(tensor, expected[name]), name


def test_the_batch_size_is_capped_by_the_rows_left_after_the_length_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    tiny = build_tiny_model(tmp_path / "tiny")
    spec = write_toy_classification_task(tmp_path / "tasks", str(tiny), n_train=4, n_valid=4, epochs=2)
    train_file = paths.data_dir("toy") / "train.jsonl"
    rows = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines()]
    for row in rows[1:]:
        row["messages"][1]["content"] = "card " * 300  # longer than max_seq_len (256)
    train_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cfg = plan_training(spec, backend="torch", profile="quick")
    assert (cfg.n_train, cfg.batch_size, cfg.iters) == (4, 4, 2)
    schedules: list[list[list[int]]] = []
    real_schedule = torch_lora.batch_schedule

    def recording_schedule(*args: Any, **kwargs: Any) -> list[list[int]]:
        schedules.append(real_schedule(*args, **kwargs))
        return schedules[-1]

    monkeypatch.setattr(torch_lora, "batch_schedule", recording_schedule)
    log = torch_lora.train_torch(cfg, spec, log=lambda _line: None)

    assert schedules == [[[0], [0]]]
    assert log["dropped_too_long"]["train"] == 3
    assert (log["iterations"], log["batch_size"], log["epochs"]) == (2, 1, 2.0)


def test_a_hub_base_is_recorded_by_repo_id_and_snapshot_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = "0123abcd" * 5
    snapshot = build_tiny_model(tmp_path / "hf" / "models--example-org--tiny" / "snapshots" / sha)
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)  # a working directory above the cache made the recorded base cwd-relative
    monkeypatch.setattr(
        torch_lora, "local_model_dir", lambda source, _rev: str(snapshot) if source == "example-org/tiny" else source
    )
    spec = write_toy_classification_task(tmp_path / "tasks", "example-org/tiny", n_train=8, n_valid=4, epochs=1)
    cfg = plan_training(spec, backend="torch", profile="quick")
    log = torch_lora.train_torch(cfg, spec, log=lambda _line: None)

    adapter = Path(cfg.run_dir) / "adapter"
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    card = (adapter / "README.md").read_text(encoding="utf-8")
    assert (config["base_model_name_or_path"], config["revision"]) == ("example-org/tiny", sha)
    assert "base_model: example-org/tiny" in card
    for text in (json.dumps(config), card):
        assert "snapshots" not in text and str(tmp_path) not in text
    assert (log["torch_base_model"], log["base_model"]) == ("example-org/tiny", "example-org/tiny")


def test_a_base_recorded_as_a_repo_id_is_kept(tmp_path: Path) -> None:
    config = {"base_model_name_or_path": "example-org/tiny-bnb-4bit", "revision": None, "target_modules": ["v", "q"]}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "README.md").write_text("---\nbase_model: example-org/tiny-bnb-4bit\n---\n", encoding="utf-8")
    torch_lora._tidy_adapter_dir(tmp_path, hub_base="example-org/tiny")
    tidied = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert tidied == {**config, "target_modules": ["q", "v"]}
    assert "base_model: example-org/tiny-bnb-4bit" in (tmp_path / "README.md").read_text(encoding="utf-8")
