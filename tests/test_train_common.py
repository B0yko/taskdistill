from __future__ import annotations

import json
import math
import types
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from taskdistill.backends import factory
from taskdistill.backends.factory import BackendUnavailable, load_backend
from taskdistill.config import TaskSpec, load_task
from taskdistill.evaluate.splits import ValidationSplit
from taskdistill.train import common, runner
from taskdistill.train.common import (
    QUICK_MAX_ITERS,
    TrainConfig,
    TrainDataError,
    compute_iterations,
    eval_every,
    plan_training,
    prepare_run_data,
    run_id,
    short_model_name,
    val_batch_count,
    validation_split_for,
    write_train_log,
)
from taskdistill.train.mlx_lora import BatchIterator, collate

BASE_05 = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
BASE_15 = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
LABELS = ["card", "card arrival", "cash"]
SCHEMA = {"type": "object", "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}}}


def write_task(root: Path, *, kind: str = "classification", **train: Any) -> TaskSpec:
    task_dir = root / "tasks" / "tiny"
    task_dir.mkdir(parents=True)
    (task_dir / "teacher_prompt.md").write_text("Pick one label.\n", encoding="utf-8")
    raw: dict[str, Any] = {
        "task": "tiny",
        "type": kind,
        "teacher": {"model": "example/teacher-model"},
        "student": {"base_model": BASE_05, "system_prompt": "Classify the banking message."},
        "train": {"batch_size": 8, "epochs": 2, "seed": 13, **train},
        "cascade": {"target": 0.9},
    }
    if kind == "classification":
        raw["labels_file"] = "labels.txt"
        (task_dir / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    else:
        raw["schema_file"] = "schema.json"
        (task_dir / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    (task_dir / "task.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_task(str(task_dir))


def chat(user: str, answer: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "Classify the banking message."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
    }


def write_split(home: Path, split: str, rows: list[dict[str, Any]], metas: list[dict[str, Any]] | None) -> None:
    data = home / "tiny" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if metas is not None:
        (data / f"{split}.meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas), encoding="utf-8")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    monkeypatch.chdir(tmp_path)
    return workspace


def classification_data(home: Path, n_train: int = 32, n_valid: int = 8, gold_every: int = 1) -> None:
    def rows(n: int, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        out, metas = [], []
        for i in range(n):
            teacher = LABELS[i % 3]
            gold = LABELS[(i + 1) % 3] if i % gold_every == 0 else None
            out.append(chat(f"{prefix} message number {i}", teacher))
            metas.append({"input_hash": f"{prefix}{i:04d}", "gold": gold, "teacher": teacher, "meta": {"group": i % 2}})
        return out, metas

    for split, n in (("train", n_train), ("valid", n_valid)):
        r, m = rows(n, split)
        write_split(home, split, r, m)


# -- schedule --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("epochs", "n_train", "batch_size", "profile", "expected"),
    [
        (2, 9000, 8, "full", 2250),
        (2, 2000, 8, "quick", QUICK_MAX_ITERS),  # 500 capped
        (2, 400, 8, "quick", 100),
        (2.5, 32, 8, "full", 10),
        (1, 10, 8, "full", 2),  # ceil
        (0.1, 8, 8, "full", 1),  # at least one
        (2, 0, 8, "full", 0),
    ],
)
def test_compute_iterations(epochs: float, n_train: int, batch_size: int, profile: str, expected: int) -> None:
    assert compute_iterations(epochs, n_train, batch_size, profile) == expected


def test_eval_schedule_and_val_batches() -> None:
    assert eval_every(2250, "full") == 225
    assert eval_every(200, "quick") == 50
    assert eval_every(3, "full") == 1
    assert eval_every(3, "quick") == 1
    assert val_batch_count(1000, 8, "full") == -1
    assert val_batch_count(300, 8, "quick") == 10
    assert val_batch_count(40, 8, "quick") == 5
    assert val_batch_count(3, 8, "quick") == 1
    assert val_batch_count(0, 8, "full") == 0


# -- naming ----------------------------------------------------------------------------------------


def test_run_id_naming() -> None:
    assert short_model_name(BASE_05) == "qwen2.5-0.5b"
    assert short_model_name("Qwen/Qwen2.5-1.5B-Instruct") == "qwen2.5-1.5b"
    assert short_model_name("/tmp/models/Tiny Qwen2/") == "tiny-qwen2"
    assert run_id(BASE_05, "quick", 13) == "qwen2.5-0.5b-quick-s13"
    assert run_id(BASE_15, "full", 7, "gold") == "qwen2.5-1.5b-full-s7-gold"
    assert run_id("Qwen/Qwen2.5-0.5B-Instruct", "full", 1, "teacher", "torch") == "qwen2.5-0.5b-full-s1-torch"
    assert run_id(BASE_05, "full", 2, "gold", "torch") == "qwen2.5-0.5b-full-s2-gold-torch"


# -- plan ------------------------------------------------------------------------------------------


def test_plan_training_from_spec(home: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=40, n_valid=12)
    cfg = plan_training(spec)
    assert (cfg.task, cfg.backend, cfg.profile, cfg.seed, cfg.labels) == ("tiny", "mlx", "full", 13, "teacher")
    assert cfg.base_model == BASE_05
    assert cfg.base_revision == "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
    assert (cfg.n_train, cfg.n_valid, cfg.batch_size) == (40, 12, 8)
    assert cfg.iters == 10 and cfg.steps_per_eval == 1 and cfg.val_batches == -1
    assert cfg.run_id == "qwen2.5-0.5b-full-s13"
    assert Path(cfg.run_dir) == home / "tiny" / "runs" / cfg.run_id
    assert Path(cfg.data_dir) == Path(cfg.run_dir) / "data"
    assert cfg.task_type == "classification"

    quick = plan_training(spec, profile="quick", seed=4, base=BASE_15)
    assert quick.run_id == "qwen2.5-1.5b-quick-s4"
    assert quick.base_revision == "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
    assert quick.steps_per_eval == max(1, quick.iters // 4)
    assert quick.val_batches == 2

    torch_cfg = plan_training(spec, backend="torch")
    assert torch_cfg.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert torch_cfg.base_revision is None
    assert torch_cfg.run_id == "qwen2.5-0.5b-full-s13-torch"

    restored = TrainConfig.load(cfg.save(home / "cfg.json"))
    assert restored == cfg


def test_plan_clamps_batch_size_for_tiny_data(home: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=5, n_valid=2)
    cfg = plan_training(spec)
    assert cfg.batch_size == 5
    assert cfg.iters == 2  # ceil(2 epochs * 5 / 5)


def test_plan_counts_gold_rows_only_for_gold_labels(home: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=32, n_valid=8, gold_every=2)
    cfg = plan_training(spec, labels="gold")
    assert (cfg.n_train, cfg.n_valid) == (16, 4)
    assert cfg.run_id.endswith("-gold")


def test_plan_requires_curated_data(home: Path) -> None:
    spec = write_task(home.parent)
    with pytest.raises(TrainDataError, match="taskdistill curate --task tiny"):
        plan_training(spec)


def test_plan_rejects_unknown_options(home: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home)
    with pytest.raises(ValueError, match="backend"):
        plan_training(spec, backend="jax")
    with pytest.raises(ValueError, match="label source"):
        plan_training(spec, labels="silver")
    with pytest.raises(ValueError, match="profile"):
        plan_training(spec, profile="medium")


# -- run data --------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prepare_run_data_gold_replacement(home: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=32, n_valid=8, gold_every=2)
    cfg = plan_training(spec, labels="gold")
    stats = prepare_run_data(spec, cfg)
    assert stats["dropped_no_gold"] == 16 + 4
    assert stats["splits"]["train"] == {"source_rows": 32, "rows": 16, "dropped_no_gold": 16}

    data = Path(cfg.data_dir)
    rows = read_jsonl(data / "train.jsonl")
    metas = read_jsonl(data / "train.meta.jsonl")
    assert len(rows) == len(metas) == 16
    for row, meta in zip(rows, metas, strict=True):
        assert meta["gold"] is not None
        assert row["messages"][-1] == {"role": "assistant", "content": meta["gold"]}
        index = int(meta["input_hash"][len("train") :])
        assert row["messages"][1]["content"] == f"train message number {index}"

    order = json.loads((data / "order.json").read_text(encoding="utf-8"))
    assert sorted(order["train"]) == list(range(0, 32, 2))
    assert order["train"] != sorted(order["train"])  # shuffled
    assert order["train_seed"] == 13

    source = read_jsonl(home / "tiny" / "data" / "train.jsonl")
    assert source[0]["messages"][-1]["content"] == "card"  # the curated files are never modified


def test_prepare_run_data_teacher_labels_keep_every_row(home: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=12, n_valid=4, gold_every=3)
    cfg = plan_training(spec)
    stats = prepare_run_data(spec, cfg)
    assert stats["dropped_no_gold"] == 0
    rows = read_jsonl(Path(cfg.data_dir) / "train.jsonl")
    metas = read_jsonl(Path(cfg.data_dir) / "train.meta.jsonl")
    assert Counter(r["messages"][-1]["content"] for r in rows) == Counter(LABELS[i % 3] for i in range(12))
    assert all(r["messages"][-1]["content"] == m["teacher"] for r, m in zip(rows, metas, strict=True))


def test_prepare_run_data_order_depends_on_seed_only(home: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=30, n_valid=10)
    first = plan_training(spec, seed=1)
    prepare_run_data(spec, first)
    again = plan_training(spec, seed=1, profile="quick")
    prepare_run_data(spec, again)
    other = plan_training(spec, seed=2)
    prepare_run_data(spec, other)

    def order(cfg: TrainConfig) -> dict[str, Any]:
        return json.loads((Path(cfg.data_dir) / "order.json").read_text(encoding="utf-8"))

    assert order(first)["train"] == order(again)["train"]
    assert order(first)["train"] != order(other)["train"]
    assert order(first)["valid"] == order(other)["valid"]  # validation order is shared by all seeds


def test_prepare_run_data_extraction_gold_is_canonical(home: Path) -> None:
    spec = write_task(home.parent, kind="extraction")
    rows = [chat(f"document {i}", '{"vendor":"teacher"}') for i in range(4)]
    metas: list[dict[str, Any]] = [
        {"input_hash": "h0", "gold": {"total": 12.5, "vendor": "Example Tools Ltd"}},
        {"input_hash": "h1", "gold": None},
        {"input_hash": "h2", "gold": '{"total": 3, "vendor": "Sample GmbH", "extra": 1}'},
        {"input_hash": "h3", "gold": {"vendor": "Demo LLC", "total": 0}},
    ]
    write_split(home, "train", rows, metas)
    write_split(home, "valid", rows[:1], metas[:1])
    cfg = plan_training(spec, labels="gold")
    prepare_run_data(spec, cfg)
    written = {
        m["input_hash"]: r["messages"][-1]["content"]
        for r, m in zip(
            read_jsonl(Path(cfg.data_dir) / "train.jsonl"),
            read_jsonl(Path(cfg.data_dir) / "train.meta.jsonl"),
            strict=True,
        )
    }
    assert written == {
        "h0": '{"vendor":"Example Tools Ltd","total":12.5}',
        "h2": '{"vendor":"Sample GmbH","total":3}',
        "h3": '{"vendor":"Demo LLC","total":0}',
    }


def test_gold_labels_need_a_sidecar(home: Path) -> None:
    spec = write_task(home.parent)
    write_split(home, "train", [chat("a", "card")], None)
    with pytest.raises(TrainDataError, match=r"meta\.jsonl"):
        plan_training(spec, labels="gold")


def test_misaligned_sidecar_is_refused(home: Path) -> None:
    spec = write_task(home.parent)
    write_split(home, "train", [chat("a", "card"), chat("b", "cash")], [{"gold": "card"}])
    with pytest.raises(TrainDataError, match="re-run curate"):
        plan_training(spec)


def test_validation_split_for(home: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=8, n_valid=6)
    cfg = plan_training(spec)
    prepare_run_data(spec, cfg)
    val = validation_split_for(cfg)
    assert isinstance(val, ValidationSplit)
    assert len(val) == 6 and val.task_type == "classification"
    record = next(r for r in val if r.id == "valid0002")
    assert record.input == "valid message number 2"
    assert (record.gold, record.teacher, record.group) == ("card", "cash", "0")


# -- train log -------------------------------------------------------------------------------------


def test_write_train_log_keys_and_relative_paths(home: Path, tmp_path: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=16, n_valid=4, gold_every=2)
    local_base = tmp_path / "models" / "tiny-base"
    local_base.mkdir(parents=True)
    cfg = plan_training(spec, labels="gold", base="models/tiny-base")
    assert cfg.base_model == str(local_base.resolve())  # relative local paths are made absolute for the trainer
    prepare_run_data(spec, cfg)
    adapter = Path(cfg.run_dir) / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapters.safetensors").write_bytes(b"\0" * 2_500_000)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    path = write_train_log(
        cfg,
        {
            "wall_seconds": 12.5,
            "train_seconds": 10.0,
            "peak_memory_gb": 1.2,
            "processed_tokens": 5000,
            "trained_tokens": 300,
            "curve": {"train": [[1, 2.0], [2, 1.5]], "val": [[0, 3.0], [2, math.nan]]},
            "best_iteration": 0,
            "best_val_loss": 3.0,
            "final_val_loss": math.nan,
            "extra_path": str(Path(cfg.run_dir) / "data" / "train.jsonl"),
        },
    )
    log = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "run_id", "task", "backend", "base_model", "base_revision", "profile", "seed", "labels", "n_train",
        "n_valid", "dropped_no_gold", "iterations", "epochs", "batch_size", "learning_rate", "lora_rank",
        "lora_layers", "max_seq_len", "wall_seconds", "train_seconds", "peak_memory_gb", "tokens_per_second",
        "trained_tokens", "processed_tokens", "curve", "best_iteration", "best_val_loss", "final_val_loss",
        "adapter_dir", "adapter_size_mb", "hardware", "versions", "date", "load_average",
    }  # fmt: skip
    assert required <= set(log)
    assert log["tokens_per_second"] == 500.0
    assert log["dropped_no_gold"] == 10
    assert log["epochs"] == pytest.approx(cfg.iters * cfg.batch_size / cfg.n_train, abs=1e-4)
    assert log["adapter_size_mb"] == pytest.approx(2.5, abs=0.01)
    assert log["adapter_dir"] == f"tiny/runs/{cfg.run_id}/adapter"
    assert log["loss_plot"] == f"tiny/runs/{cfg.run_id}/loss.png"
    assert log["extra_path"] == f"tiny/runs/{cfg.run_id}/data/train.jsonl"
    # Outside the workspace the local base stays absolute: eval and serve load it from there (never a
    # working-directory-relative path or its name only, which they could not resolve).
    assert log["base_model"] == str(local_base.resolve())
    assert log["final_val_loss"] is None and log["curve"]["val"][1] == [2, None]
    assert {"python", "taskdistill", "mlx", "mlx-lm"} <= set(log["versions"])
    assert {"model", "cpu", "memory_gb"} <= set(log["hardware"])
    assert (Path(cfg.run_dir) / "loss.png").is_file()
    text = path.read_text(encoding="utf-8")
    assert text.count(str(tmp_path)) == 1  # the base model is the only absolute path


def test_write_train_log_counts_the_examples_left_after_the_length_filter(home: Path) -> None:
    spec = write_task(home.parent, epochs=2.5)
    classification_data(home, n_train=32, n_valid=8)
    cfg = plan_training(spec)
    prepare_run_data(spec, cfg)
    assert (cfg.iters, cfg.batch_size, cfg.epochs_trained) == (10, 8, 2.5)

    measured = {"iterations": 10, "batch_size": 8, "dropped_too_long": {"train": 2, "valid": 1}}
    log = json.loads(write_train_log(cfg, measured).read_text(encoding="utf-8"))
    assert (log["n_train"], log["n_train_planned"], log["n_valid"], log["n_valid_planned"]) == (30, 32, 7, 8)
    assert log["epochs"] == pytest.approx(10 * 8 / 30, abs=1e-4)  # iterations * batch_size / n_train, as trained

    explicit = json.loads(write_train_log(cfg, {**measured, "n_train": 29}).read_text(encoding="utf-8"))
    assert explicit["n_train"] == 29 and explicit["epochs"] == pytest.approx(80 / 29, abs=1e-4)

    nothing_dropped = json.loads(write_train_log(cfg, {}).read_text(encoding="utf-8"))
    assert (nothing_dropped["n_train"], nothing_dropped["n_train_planned"], nothing_dropped["epochs"]) == (32, 32, 2.5)


def test_saved_train_config_is_workspace_relative(home: Path, tmp_path: Path) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=16, n_valid=4)
    cfg = plan_training(spec)
    saved = cfg.save(Path(cfg.run_dir) / "train_config.json")
    text = saved.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    on_disk = json.loads(text)
    assert on_disk["run_dir"] == f"tiny/runs/{cfg.run_id}" and on_disk["data_dir"] == f"tiny/runs/{cfg.run_id}/data"
    assert TrainConfig.load(saved) == cfg  # resolved against $TASKDISTILL_HOME

    absolute = json.loads(cfg.save(tmp_path / "child.json", portable=False).read_text(encoding="utf-8"))
    assert absolute["run_dir"] == cfg.run_dir and Path(absolute["run_dir"]).is_absolute()

    inside = home / "models" / "tiny-base"
    inside.mkdir(parents=True)
    local = plan_training(spec, base=str(inside))
    record = json.loads(local.save(tmp_path / "local.json").read_text(encoding="utf-8"))
    assert record["base_model"] == "models/tiny-base"
    assert TrainConfig.load(tmp_path / "local.json").base_model == str(inside.resolve())

    outside = tmp_path / "elsewhere" / "tiny-base"
    outside.mkdir(parents=True)
    far = plan_training(spec, base=str(outside))
    far_record = json.loads(far.save(tmp_path / "far.json").read_text(encoding="utf-8"))
    assert far_record["base_model"] == str(outside.resolve())  # outside the workspace: kept absolute, never a name
    assert far_record["run_dir"] == f"tiny/runs/{far.run_id}"
    assert TrainConfig.load(tmp_path / "far.json").base_model == str(outside.resolve())


def test_adapter_size_mb_of_missing_dir(tmp_path: Path) -> None:
    assert common.adapter_size_mb(tmp_path / "nope") == 0.0


def test_mlx_adapter_config_base_keeps_a_local_base_outside_the_workspace(home: Path, tmp_path: Path) -> None:
    """``_portable_base`` follows :func:`common.portable_path`, unlike the display-only ``paths.relative_to_home``."""
    from taskdistill.train.mlx_lora import _portable_base

    assert _portable_base("mlx-community/Qwen2.5-0.5B-Instruct-4bit") == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    inside = home / "models" / "tiny-base"
    inside.mkdir(parents=True)
    assert _portable_base(str(inside)) == "models/tiny-base"

    outside = tmp_path / "elsewhere" / "tiny-base"
    outside.mkdir(parents=True)
    assert _portable_base(str(outside)) == str(outside.resolve())  # never reduced to "tiny-base"


# -- runner orchestration (no model) ---------------------------------------------------------------


def test_run_training_refuses_mlx_off_apple_silicon(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = write_task(home.parent)
    monkeypatch.setattr(runner, "mlx_unavailable_reason", lambda: "linux/x86_64")
    with pytest.raises(common.TrainBackendUnavailable) as excinfo:
        runner.run_training(spec, backend="mlx")
    message = str(excinfo.value)
    assert message.startswith("MLX training needs Apple Silicon (arm64 macOS 14+). Use --backend torch")
    assert "uvx --from 'taskdistill[torch] @ git+https://github.com/B0yko/taskdistill@v0.1.1' taskdistill" in message


def test_run_training_writes_selected_run_once(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=16, n_valid=4)
    monkeypatch.setattr(runner, "mlx_unavailable_reason", lambda: None)
    seen: list[TrainConfig] = []

    def fake_child(cfg: TrainConfig, log: Any) -> None:
        seen.append(cfg)
        assert (Path(cfg.data_dir) / "train.jsonl").is_file()
        write_train_log(cfg, {"wall_seconds": 1.0, "best_iteration": 4, "best_val_loss": 0.5})

    monkeypatch.setattr(runner, "_train_mlx_subprocess", fake_child)
    lines: list[str] = []
    first = runner.run_training(spec, seed=1, log=lines.append)
    assert first.run_id == "qwen2.5-0.5b-full-s1" and first.log["best_val_loss"] == 0.5
    selected = json.loads((home / "tiny" / "selected_run.json").read_text(encoding="utf-8"))
    assert selected["run_id"] == first.run_id and "eval --select" in selected["reason"]

    runner.run_training(spec, seed=2, log=lines.append)
    selected = json.loads((home / "tiny" / "selected_run.json").read_text(encoding="utf-8"))
    assert selected["run_id"] == first.run_id  # never overwritten by a later run
    assert len(seen) == 2 and any("best validation loss 0.5000" in line for line in lines)


class TrainingFailedForTest(RuntimeError):
    pass


def test_rerunning_a_run_id_removes_the_earlier_run_first(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=16, n_valid=4)
    monkeypatch.setattr(runner, "mlx_unavailable_reason", lambda: None)

    def finished(cfg: TrainConfig, log: Any) -> None:
        adapter = Path(cfg.run_dir) / "adapter"
        adapter.mkdir()
        (adapter / "adapters.safetensors").write_bytes(b"first")
        write_train_log(cfg, {"best_iteration": 9, "best_val_loss": 0.5})
        (Path(cfg.run_dir) / "preds_valid.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_train_mlx_subprocess", finished)
    first = runner.run_training(spec, seed=3, log=lambda _line: None)
    assert (first.run_dir / "train_log.json").is_file() and (first.run_dir / "loss.png").is_file()

    def interrupted(cfg: TrainConfig, log: Any) -> None:
        assert (Path(cfg.data_dir) / "train.jsonl").is_file()
        partial = Path(cfg.run_dir) / "adapter.partial"
        partial.mkdir()
        (partial / "adapters.safetensors").write_bytes(b"iteration 0")
        raise TrainingFailedForTest

    monkeypatch.setattr(runner, "_train_mlx_subprocess", interrupted)
    lines: list[str] = []
    with pytest.raises(TrainingFailedForTest):
        runner.run_training(spec, seed=3, log=lines.append)
    left = sorted(p.name for p in first.run_dir.iterdir())
    assert left == ["adapter.partial", "data", "train_config.json"]  # no log, plot, adapter or predictions
    assert any("exists; replacing it" in line for line in lines)


def test_the_base_model_is_fetched_before_the_child_starts(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=16, n_valid=4)
    cfg = plan_training(spec, base=BASE_15)
    fetched: list[tuple[str, str | None]] = []
    monkeypatch.setattr(runner, "is_available_locally", lambda repo, rev: False)
    monkeypatch.setattr(runner, "resolve_model_path", lambda repo, rev: fetched.append((repo, rev)) or home)
    lines: list[str] = []
    runner._ensure_base_model(cfg, lines.append)
    assert fetched == [(BASE_15, "8b403126fc14f14cfc99bb4cfa72ecbc129ea677")]
    assert len(lines) == 1 and lines[0].startswith(f"downloading {BASE_15}@8b403126fc14f14cfc99bb4cfa72ecbc129ea677")

    fetched.clear()
    monkeypatch.setattr(runner, "is_available_locally", lambda repo, rev: True)
    runner._ensure_base_model(cfg, lines.append)
    assert fetched == [] and len(lines) == 1


def test_the_training_child_reads_an_absolute_config_from_outside_the_run(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = write_task(home.parent)
    classification_data(home, n_train=16, n_valid=4)
    cfg = plan_training(spec)
    prepare_run_data(spec, cfg)
    monkeypatch.setattr(runner, "_ensure_base_model", lambda cfg, log: None)
    seen: dict[str, Any] = {}

    class FakeChild:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            config = Path(command[-1])
            seen["config"] = config
            seen["child_cfg"] = json.loads(config.read_text(encoding="utf-8"))
            seen["cwd"] = kwargs["cwd"]
            seen["env"] = kwargs["env"]
            self.stdout = iter(["[taskdistill] training\n", "Saved final weights to x\n"])

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", FakeChild)
    lines: list[str] = []
    runner._train_mlx_subprocess(cfg, lines.append)
    assert lines == ["[taskdistill] training"]
    assert not seen["config"].exists()  # removed once the child has exited
    assert Path(cfg.run_dir) not in seen["config"].parents
    assert seen["child_cfg"]["run_dir"] == cfg.run_dir and Path(cfg.run_dir).is_absolute()
    assert Path(seen["cwd"]) == Path(cfg.run_dir)
    assert seen["env"]["TASKDISTILL_HOME"] == str(home)


# -- the MLX trainer's entry point (no model) ------------------------------------------------------


def test_mlx_trainer_main_accepts_a_config_path(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from taskdistill.train import mlx_lora

    spec = write_task(home.parent)
    classification_data(home, n_train=16, n_valid=4)
    torch_cfg = plan_training(spec, backend="torch")
    path = torch_cfg.save(home / "torch_config.json")
    assert mlx_lora.main(str(path)) == 2
    assert mlx_lora.main(path) == 2
    assert mlx_lora.main([str(path)]) == 2
    assert capsys.readouterr().err.count("for the torch backend, not mlx") == 3
    assert mlx_lora.main([]) == 2
    assert "usage: python -m taskdistill.train.mlx_lora" in capsys.readouterr().err


# -- batching (pure part of the MLX trainer) -------------------------------------------------------


def test_epoch_batches_are_full_and_cover_every_example() -> None:
    batches = BatchIterator(seed=5).epoch_batches(10, 4)
    drawn = [next(batches) for _ in range(5)]  # 20 draws = exactly two epochs
    assert all(len(b) == 4 for b in drawn)
    flat = [i for b in drawn for i in b]
    assert Counter(flat[:10]) == Counter(range(10))
    assert Counter(flat) == Counter(dict.fromkeys(range(10), 2))
    assert flat[:10] != flat[10:]  # a fresh permutation per epoch
    again = BatchIterator(seed=5).epoch_batches(10, 4)
    assert [next(again) for _ in range(5)] == drawn


def test_sequential_batches_keep_the_remainder() -> None:
    assert list(BatchIterator.sequential_batches(10, 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_collate_pads_like_mlx_lm_and_locates_completions() -> None:
    items = [(list(range(1, 41)), 37), (list(range(1, 11)), 6)]
    batch, spans, positions, real, supervised = collate(items, max_seq_length=512)
    assert batch.shape == (2, 65)  # 1 + 32 * ceil(40 / 32)
    assert batch[1, 10:].sum() == 0
    assert spans.tolist() == [[37, 40], [6, 10]]
    assert positions.shape == (2, 32)
    assert positions[0, :3].tolist() == [36, 37, 38]  # target index = token index - 1
    assert positions[1, :4].tolist() == [5, 6, 7, 8]
    assert int(positions.max()) <= batch.shape[1] - 2
    assert (real, supervised) == (50, 3 + 4)


def test_collate_caps_width_at_max_seq_length() -> None:
    batch, spans, _, real, supervised = collate([(list(range(1, 101)), 90)], max_seq_length=64)
    assert batch.shape == (1, 64)
    assert spans.tolist() == [[90, 64]]
    assert (real, supervised) == (64, 0)  # the completion was cut off entirely: no loss tokens


# -- backend factory -------------------------------------------------------------------------------


def test_load_backend_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="expected one of: mlx, torch"):
        load_backend("jax", BASE_05)


def test_load_backend_explains_missing_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "mlx_unavailable_reason", lambda: "this machine is linux/x86_64")
    with pytest.raises(BackendUnavailable, match=r"Apple Silicon.*--backend torch"):
        load_backend("mlx", BASE_05)


def test_load_backend_explains_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "torch_unavailable_reason", lambda: "missing package(s): torch")
    with pytest.raises(BackendUnavailable, match=r"taskdistill\[torch\]"):
        load_backend("torch", BASE_05)


@pytest.mark.skipif(factory.torch_unavailable_reason() is not None, reason="the torch extra is not installed")
def test_load_backend_torch_maps_the_mlx_base_and_defers_loading() -> None:
    backend = load_backend("torch", BASE_05, adapter_path="runs/x/adapter")
    assert type(backend).__name__ == "TorchBackend"
    assert backend.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert backend.adapter_path == "runs/x/adapter"
    assert backend.tokenizer is None  # nothing is loaded until the serving thread calls load()


def test_lr_schedule_warmup_then_cosine() -> None:
    from taskdistill.train.common import lr_at, warmup_iters

    assert warmup_iters(200) == 20 and warmup_iters(2219) == 100 and warmup_iters(200, "constant") == 0
    peak = 1e-4
    assert lr_at(0, peak, 200) == pytest.approx(peak / 20)
    assert lr_at(19, peak, 200) == pytest.approx(peak)
    assert lr_at(20, peak, 200) == pytest.approx(peak)
    assert lr_at(199, peak, 200) == pytest.approx(peak * 0.1, rel=1e-3)
    assert lr_at(110, peak, 200) == pytest.approx(peak * 0.1 + 0.5 * peak * 0.9 * (1 + math.cos(math.pi * 0.5)))
    assert lr_at(57, peak, 200, "constant") == peak


def test_mlx_schedule_matches_lr_at() -> None:
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    from taskdistill.train.common import lr_at
    from taskdistill.train.mlx_lora import lr_schedule

    cfg = types.SimpleNamespace(lr_schedule="warmup_cosine", learning_rate=1e-4, iters=200)
    schedule = lr_schedule(cfg)  # type: ignore[arg-type]
    for step in (0, 5, 19, 20, 100, 199, 250):
        assert float(schedule(mx.array(step))) == pytest.approx(lr_at(step, 1e-4, 200), rel=1e-5)
    constant = types.SimpleNamespace(lr_schedule="constant", learning_rate=1e-4, iters=200)
    assert lr_schedule(constant) == 1e-4  # type: ignore[arg-type]
