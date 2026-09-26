"""Regressions for training and evaluation: retrained run ids, local bases, selection and base revisions."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from taskdistill import paths
from taskdistill.backends.fake import FakeBackend
from taskdistill.config import TaskSpec
from taskdistill.evaluate import predictions
from taskdistill.evaluate import runner as eval_runner
from taskdistill.serve.runner import ServeSetupError, resolve_threshold
from taskdistill.train import runner as train_runner
from taskdistill.train.common import TrainConfig, TrainingFailed, write_train_log
from tiny_model import TOY_LABELS, TOY_SYSTEM_PROMPT, toy_example, write_toy_classification_task

MLX_05 = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MLX_05_8BIT = "mlx-community/Qwen2.5-0.5B-Instruct-8bit"
N_TRAIN, N_VALID, N_TEST = 12, 8, 6
VALID_IDS = range(N_TRAIN, N_TRAIN + N_VALID)
TEST_IDS = range(N_TRAIN + N_VALID, N_TRAIN + N_VALID + N_TEST)


# -- helpers -----------------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_runner, "RESAMPLES", 50)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    monkeypatch.chdir(tmp_path)
    return workspace


def toy_spec(root: Path, base: str = MLX_05) -> TaskSpec:
    return write_toy_classification_task(root / "tasks", base, n_train=N_TRAIN, n_valid=N_VALID)


def write_test_split(spec: TaskSpec, ids: range = TEST_IDS) -> None:
    """``data/test.jsonl`` and its sidecar in the toy format (an empty range writes an empty split)."""
    data = paths.data_dir(spec.task)
    rows, metas = [], []
    for i in ids:
        text, label = toy_example(i)
        messages = [
            {"role": "system", "content": TOY_SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": label},
        ]
        rows.append(json.dumps({"messages": messages}) + "\n")
        metas.append(json.dumps({"input_hash": f"{i:064x}", "gold": label, "teacher": label, "meta": {}}) + "\n")
    (data / "test.jsonl").write_text("".join(rows), encoding="utf-8")
    (data / "test.meta.jsonl").write_text("".join(metas), encoding="utf-8")


def fake_trainer(monkeypatch: pytest.MonkeyPatch) -> list[TrainConfig]:
    """Replace the MLX child with one that writes a new adapter and the training log (no model is loaded)."""
    monkeypatch.setattr(train_runner, "mlx_unavailable_reason", lambda: None)
    trained: list[TrainConfig] = []

    def child(cfg: TrainConfig, log: Any) -> None:
        trained.append(cfg)
        adapter = Path(cfg.run_dir) / "adapter"
        adapter.mkdir()
        (adapter / "adapters.safetensors").write_bytes(f"weights {len(trained)}".encode())
        (adapter / "adapter_config.json").write_text(json.dumps({"base_model": cfg.base_model}), encoding="utf-8")
        write_train_log(cfg, {"wall_seconds": 1.0, "best_iteration": 2, "best_val_loss": 0.5})

    monkeypatch.setattr(train_runner, "_train_mlx_subprocess", child)
    return trained


def script(*, wrong: int = 1, confidence: float = 0.9) -> dict[str, tuple[str, float]]:
    """Answers for the validation and test inputs: the first ``wrong`` validation inputs get another label."""
    answers: dict[str, tuple[str, float]] = {}
    for n, i in enumerate([*VALID_IDS, *TEST_IDS]):
        text, label = toy_example(i)
        if n < wrong:
            other = next(x for x in TOY_LABELS if x != label)
            answers[text] = (other, 0.3)
        else:
            answers[text] = (label, confidence - 0.01 * n)
    return answers


class Factory:
    """``backend_factory`` answering from a script per run (by adapter directory) or per base model (no adapter)."""

    def __init__(self, scripts: dict[str, Any]) -> None:
        self.scripts = scripts
        self.calls: list[tuple[str, str]] = []

    def __call__(self, base_model: str, adapter_path: str | None) -> FakeBackend:
        key = Path(adapter_path).parent.name if adapter_path else f"base:{base_model}"
        self.calls.append((base_model, key))
        return FakeBackend(answers=self.scripts.get(key, {}), base_model=base_model)


def make_run(spec: TaskSpec, run_id: str, *, base: str = MLX_05, seed: int = 13, labels: str = "teacher") -> Path:
    run_dir = paths.runs_dir(spec.task) / run_id
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "adapter" / "adapters.safetensors").write_bytes(run_id.encode())
    log = {
        "run_id": run_id,
        "task": spec.task,
        "backend": "mlx",
        "base_model": base,
        "profile": "full",
        "seed": seed,
        "labels": labels,
        "date": "2026-01-01T00:00:00+00:00",
    }
    (run_dir / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    return run_dir


def read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def quiet(_line: str) -> None:
    return None


# -- retraining a run id drops the earlier adapter's eval outputs ------------------------------------------------------


def test_retraining_a_run_id_removes_its_stale_threshold_and_eval_outputs(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = toy_spec(tmp_path)
    fake_trainer(monkeypatch)
    first = train_runner.run_training(spec, profile="quick", seed=13, log=quiet)
    other = train_runner.run_training(spec, profile="quick", seed=14, log=quiet)
    rid = first.run_id
    factory = Factory({rid: script(wrong=1), other.run_id: script(wrong=0)})
    eval_runner.run_eval(spec, run_id=rid, split="valid", backend_factory=factory, log=quiet)
    eval_runner.run_eval(spec, run_id=other.run_id, split="valid", backend_factory=factory, log=quiet)

    task_threshold = paths.task_home(spec.task) / "threshold.json"
    assert read(task_threshold)["run_id"] == rid  # the selected run (train named the first teacher run)
    assert resolve_threshold(spec, "auto", rid).warnings == ()
    assert (paths.eval_dir(spec.task) / rid / "eval_valid.json").is_file()

    lines: list[str] = []
    again = train_runner.run_training(spec, profile="quick", seed=13, log=lines.append)
    assert again.run_id == rid
    assert (again.run_dir / "adapter" / "adapters.safetensors").read_bytes() == b"weights 3"
    assert not (paths.eval_dir(spec.task) / rid).exists()
    assert not task_threshold.exists()
    assert any("re-run `taskdistill eval --task toy --run " + rid in line for line in lines)
    with pytest.raises(ServeSetupError, match="--threshold auto needs"):
        resolve_threshold(spec, "auto", rid)  # no threshold chosen on other weights is applied
    # Another run's eval outputs are untouched.
    assert (paths.eval_dir(spec.task) / other.run_id / "threshold.json").is_file()
    assert resolve_threshold(spec, "auto", other.run_id).warnings == ()

    # Evaluating the retrained run chooses its threshold again.
    eval_runner.run_eval(spec, run_id=rid, split="valid", backend_factory=factory, log=quiet)
    assert read(task_threshold)["run_id"] == rid
    assert resolve_threshold(spec, "auto", rid).warnings == ()


def test_thresholds_are_stamped_with_the_adapter_they_were_chosen_for(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = toy_spec(tmp_path)
    fake_trainer(monkeypatch)
    first = train_runner.run_training(spec, profile="quick", log=quiet)
    factory = Factory({first.run_id: script()})
    eval_runner.run_eval(spec, split="valid", backend_factory=factory, log=quiet)
    first_sha = predictions.adapter_sha256(first.run_dir / "adapter")
    for path in (
        paths.task_home(spec.task) / "threshold.json",
        paths.eval_dir(spec.task) / first.run_id / "threshold.json",
    ):
        stored = read(path)
        assert (stored["adapter_sha256"], stored["train_date"]) == (first_sha, first.log["date"])

    again = train_runner.run_training(spec, profile="quick", log=quiet)  # same run id, new weights
    eval_runner.run_eval(spec, split="valid", backend_factory=factory, log=quiet)
    stored = read(paths.task_home(spec.task) / "threshold.json")
    assert stored["run_id"] == again.run_id == first.run_id
    assert stored["adapter_sha256"] == predictions.adapter_sha256(again.run_dir / "adapter") != first_sha


def test_retraining_keeps_a_task_threshold_chosen_for_another_run(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = toy_spec(tmp_path)
    fake_trainer(monkeypatch)
    first = train_runner.run_training(spec, profile="quick", seed=13, log=quiet)
    second = train_runner.run_training(spec, profile="quick", seed=14, log=quiet)
    factory = Factory({first.run_id: script(), second.run_id: script()})
    eval_runner.run_eval(spec, run_id=first.run_id, split="valid", backend_factory=factory, log=quiet)
    train_runner.run_training(spec, profile="quick", seed=14, log=quiet)
    assert read(paths.task_home(spec.task) / "threshold.json")["run_id"] == first.run_id
    assert (paths.eval_dir(spec.task) / first.run_id / "threshold.json").is_file()


# -- a local base outside the workspace --------------------------------------------------------------------------------


@pytest.fixture
def torch_available() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")


def test_a_local_base_outside_the_workspace_and_cwd_loads_for_eval_and_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_available: None
) -> None:
    import huggingface_hub.constants as hf_constants

    from taskdistill.backends.factory import load_backend
    from taskdistill.serve.runner import resolve_run
    from tiny_model import build_tiny_model

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)  # a base that cannot be found is never fetched
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("TASKDISTILL_HOME", str(project / ".taskdistill"))
    base = build_tiny_model(tmp_path / "models" / "my-student-base").resolve()
    spec = write_toy_classification_task(project / "tasks", "acme/unused-base", n_train=8, n_valid=4)

    result = train_runner.run_training(spec, backend="torch", base=str(base), profile="quick", log=quiet)
    assert read(result.run_dir / "train_config.json")["base_model"] == str(base)
    assert result.log["base_model"] == str(base)
    assert read(result.run_dir / "train_config.json")["run_dir"] == f"toy/runs/{result.run_id}"

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # eval and serve run from another directory
    info = eval_runner.read_run(spec, result.run_id)
    assert info.base_model == str(base)
    backend = eval_runner._default_factory("torch")(info.base_model, str(info.adapter_dir))
    backend.load()
    assert backend.source == str(base)

    served = resolve_run(spec, result.run_id)
    assert served.base_model == str(base)
    serving = load_backend("torch", served.base_model, str(served.adapter_path))
    serving.load()
    assert serving.tokenizer is not None


def test_a_base_inside_the_workspace_stays_workspace_relative(home: Path, tmp_path: Path) -> None:
    from taskdistill.train.common import plan_training, portable_path

    spec = toy_spec(tmp_path)
    inside = home / "models" / "tiny-base"
    inside.mkdir(parents=True)
    cfg = plan_training(spec, base=str(inside))
    assert cfg.to_portable_dict()["base_model"] == "models/tiny-base"
    outside = tmp_path / "far" / "tiny-base"
    outside.mkdir(parents=True)
    assert portable_path(str(outside)) == str(outside)  # never the name only, never relative to the cwd
    assert portable_path("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"


# -- only teacher-label runs become the provisional selection ----------------------------------------------------------


def test_a_gold_label_run_trained_first_is_never_selected(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = toy_spec(tmp_path)
    fake_trainer(monkeypatch)
    selected = paths.task_home(spec.task) / "selected_run.json"
    gold = train_runner.run_training(spec, profile="quick", labels="gold", log=quiet)
    assert gold.run_id.endswith("-gold")
    assert not selected.exists()

    teacher = train_runner.run_training(spec, profile="quick", log=quiet)
    assert read(selected)["run_id"] == teacher.run_id


def test_a_provisional_gold_selection_is_replaced_by_a_teacher_run(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = toy_spec(tmp_path)
    fake_trainer(monkeypatch)
    gold = train_runner.run_training(spec, profile="quick", labels="gold", log=quiet)
    selected = paths.task_home(spec.task) / "selected_run.json"
    # What earlier versions wrote for a gold run trained first.
    selected.write_text(json.dumps({"run_id": gold.run_id, "reason": train_runner.SELECTED_RUN_REASON}))
    teacher = train_runner.run_training(spec, profile="quick", log=quiet)
    assert read(selected)["run_id"] == teacher.run_id

    # A selection made by eval --select is never replaced by training.
    selected.write_text(json.dumps({"run_id": gold.run_id, "reason": "chosen on validation"}))
    train_runner.run_training(spec, profile="quick", seed=99, log=quiet)
    assert read(selected)["run_id"] == gold.run_id


# -- distinct bases under one run id -----------------------------------------------------------------------------------


def test_a_different_base_with_the_same_run_id_is_refused(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = toy_spec(tmp_path)
    trained = fake_trainer(monkeypatch)
    first = train_runner.run_training(spec, profile="quick", base=MLX_05, log=quiet)
    assert first.run_id == "qwen2.5-0.5b-quick-s13"
    adapter = (first.run_dir / "adapter" / "adapters.safetensors").read_bytes()

    with pytest.raises(TrainingFailed) as excinfo:
        train_runner.run_training(spec, profile="quick", base=MLX_05_8BIT, log=quiet)
    message = str(excinfo.value)
    assert "qwen2.5-0.5b-quick-s13" in message and MLX_05 in message and MLX_05_8BIT in message
    assert "Delete toy/runs/qwen2.5-0.5b-quick-s13" in message and "another --seed" in message
    assert len(trained) == 1  # refused before anything was trained, fetched or removed
    assert (first.run_dir / "adapter" / "adapters.safetensors").read_bytes() == adapter
    assert read(first.run_dir / "train_log.json")["base_model"] == MLX_05

    other_seed = train_runner.run_training(spec, profile="quick", base=MLX_05_8BIT, seed=14, log=quiet)
    assert other_seed.run_id == "qwen2.5-0.5b-quick-s14"
    again = train_runner.run_training(spec, profile="quick", base=MLX_05, log=quiet)  # the same base: replaced
    assert again.run_id == first.run_id


def test_local_bases_with_one_short_name_are_refused_but_the_same_directory_is_replaced(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = toy_spec(tmp_path)
    fake_trainer(monkeypatch)
    one = tmp_path / "a" / "Qwen2.5-0.5B-Instruct"
    two = tmp_path / "b" / "Qwen2.5-0.5B"
    one.mkdir(parents=True)
    two.mkdir(parents=True)
    first = train_runner.run_training(spec, profile="quick", base=str(one), log=quiet)
    with pytest.raises(TrainingFailed, match="different configuration"):
        train_runner.run_training(spec, profile="quick", base=str(two), log=quiet)

    # A run recorded before absolute paths were kept (the directory's name only) is the same base.
    config = first.run_dir / "train_config.json"
    config.write_text(json.dumps({**read(config), "base_model": one.name}), encoding="utf-8")
    assert train_runner.run_training(spec, profile="quick", base=str(one), log=quiet).run_id == first.run_id


# -- eval --select writes selected_run.json last ---------------------------------------------------------------------


RUN_A, RUN_B = "qwen2.5-0.5b-full-s13", "qwen2.5-0.5b-full-s14"


@pytest.fixture
def two_runs(home: Path, tmp_path: Path) -> Iterator[TaskSpec]:
    """Run A selected and evaluated (its threshold is the task's); run B added later and better on validation."""
    spec = toy_spec(tmp_path)
    write_test_split(spec)
    make_run(spec, RUN_A, seed=13)
    (paths.task_home(spec.task) / "selected_run.json").write_text(json.dumps({"run_id": RUN_A}), encoding="utf-8")
    eval_runner.run_eval(spec, split="valid", backend_factory=Factory({RUN_A: script(wrong=2)}), log=quiet)
    assert read(paths.task_home(spec.task) / "threshold.json")["run_id"] == RUN_A
    make_run(spec, RUN_B, seed=14)
    yield spec


def assert_still_run_a(spec: TaskSpec) -> None:
    assert read(paths.task_home(spec.task) / "selected_run.json")["run_id"] == RUN_A
    assert read(paths.task_home(spec.task) / "threshold.json")["run_id"] == RUN_A
    assert resolve_threshold(spec, "auto", RUN_A).warnings == ()


def test_a_failed_select_eval_leaves_the_selection_and_threshold_in_step(two_runs: TaskSpec) -> None:
    spec = two_runs
    factory = Factory({RUN_A: script(wrong=2), RUN_B: script(wrong=0)})
    write_test_split(spec, range(0))  # an empty test split fails the evaluation after the selection
    with pytest.raises(eval_runner.EvalError, match=r"test split .* is empty"):
        eval_runner.run_eval(spec, select=True, backend_factory=factory, log=quiet)
    assert_still_run_a(spec)


def test_an_interrupted_select_eval_leaves_the_selection_and_threshold_in_step(two_runs: TaskSpec) -> None:
    spec = two_runs
    answers = script(wrong=0)
    test_texts = {toy_example(i)[0] for i in TEST_IDS}

    def interrupted(text: str) -> tuple[str, float]:
        if text in test_texts:
            raise KeyboardInterrupt
        return answers[text]

    factory = Factory({RUN_A: script(wrong=2), RUN_B: interrupted})
    with pytest.raises(KeyboardInterrupt):
        eval_runner.run_eval(spec, select=True, backend_factory=factory, log=quiet)
    assert_still_run_a(spec)


@pytest.mark.parametrize("evaluate", ["selected", "zero-shot"])
def test_selected_run_json_is_written_after_the_selected_runs_threshold(
    two_runs: TaskSpec, monkeypatch: pytest.MonkeyPatch, evaluate: str
) -> None:
    spec = two_runs
    factory = Factory({RUN_A: script(wrong=2), RUN_B: script(wrong=0), f"base:{MLX_05}": script(wrong=0)})
    commit: Callable[..., None] = eval_runner._commit_selection
    seen: list[tuple[str, bool]] = []

    def spy(spec: TaskSpec, payload: dict[str, Any], log: Any) -> None:
        owner = read(paths.task_home(spec.task) / "threshold.json")["run_id"]
        evaluated = (paths.eval_dir(spec.task) / RUN_B / "eval_test.json").is_file()
        seen.append((owner, evaluated))
        commit(spec, payload, log)

    monkeypatch.setattr(eval_runner, "_commit_selection", spy)
    result = eval_runner.run_eval(
        spec, select=True, zero_shot=evaluate == "zero-shot", backend_factory=factory, log=quiet
    )
    # The threshold is in place when the selection is written; when the selected run is evaluated, so is its eval.
    assert seen == [(RUN_B, evaluate == "selected")]
    assert read(paths.task_home(spec.task) / "selected_run.json")["run_id"] == RUN_B
    stored = read(paths.task_home(spec.task) / "threshold.json")
    assert stored["run_id"] == RUN_B
    assert stored["adapter_sha256"] == predictions.adapter_sha256(paths.runs_dir(spec.task) / RUN_B / "adapter")
    assert read(paths.eval_dir(spec.task) / RUN_B / "threshold.json")["run_id"] == RUN_B
    assert read(paths.task_home(spec.task) / "selected_run.json")["adapter_sha256"] == stored["adapter_sha256"]
    assert result["run_id"] == (RUN_B if evaluate == "selected" else "zero-shot-qwen2.5-0.5b")
    assert resolve_threshold(spec, "auto", RUN_B).warnings == ()


def test_select_on_validation_writes_the_threshold_before_the_selection(
    two_runs: TaskSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = two_runs
    factory = Factory({RUN_A: script(wrong=2), RUN_B: script(wrong=0)})
    commit: Callable[..., None] = eval_runner._commit_selection
    owners: list[str] = []

    def spy(spec: TaskSpec, payload: dict[str, Any], log: Any) -> None:
        owners.append(read(paths.task_home(spec.task) / "threshold.json")["run_id"])
        commit(spec, payload, log)

    monkeypatch.setattr(eval_runner, "_commit_selection", spy)
    payload = eval_runner.select_on_validation(spec, backend_factory=factory, log=quiet)
    assert owners == [RUN_B] and payload["run_id"] == RUN_B


# -- zero-shot base ----------------------------------------------------------------------------------------------------


def test_zero_shot_takes_an_explicit_base(home: Path, tmp_path: Path) -> None:
    spec = toy_spec(tmp_path)
    large = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    make_run(spec, "qwen2.5-1.5b-full-s13", base=large)
    (paths.task_home(spec.task) / "selected_run.json").write_text(json.dumps({"run_id": "qwen2.5-1.5b-full-s13"}))
    factory = Factory({f"base:{MLX_05}": script(), f"base:{large}": script()})

    default = eval_runner.run_eval(spec, zero_shot=True, split="valid", backend_factory=factory, log=quiet)
    assert default["run_id"] == "zero-shot-qwen2.5-1.5b"  # unchanged: the selected run's base

    pinned = eval_runner.run_eval(spec, zero_shot=True, base=MLX_05, split="valid", backend_factory=factory, log=quiet)
    assert pinned["run_id"] == "zero-shot-qwen2.5-0.5b" and pinned["run"]["base_model"] == MLX_05
    assert factory.calls[-1] == (MLX_05, f"base:{MLX_05}")
    assert "--base" in pinned["command"]

    local = tmp_path / "models" / "Tiny-Base"
    local.mkdir(parents=True)
    local = local.resolve()
    factory.scripts[f"base:{local}"] = script()
    by_path = eval_runner.run_eval(
        spec, zero_shot=True, base="models/Tiny-Base", split="valid", backend_factory=factory, log=quiet
    )
    assert by_path["run_id"] == "zero-shot-tiny-base"
    assert factory.calls[-1] == (str(local), f"base:{local}")  # a cwd-relative directory made absolute

    with pytest.raises(eval_runner.EvalError, match="only with --zero-shot"):
        eval_runner.run_eval(spec, base=MLX_05, split="valid", backend_factory=factory, log=quiet)


# -- torch runs are pinned to the base commit they trained on ----------------------------------------------------------


def test_torch_runs_load_the_base_revision_they_were_trained_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_available: None
) -> None:
    import huggingface_hub.constants as hf_constants
    import numpy as np

    from taskdistill.backends.factory import load_backend
    from taskdistill.backends.torch_backend import TorchBackend
    from taskdistill.serve.runner import resolve_run
    from tiny_model import build_tiny_model

    cache = tmp_path / "hf-cache"
    repo = cache / "models--acme--tiny-base"
    sha_a, sha_b = "a" * 40, "b" * 40
    build_tiny_model(repo / "snapshots" / sha_a, seed=0)
    build_tiny_model(repo / "snapshots" / sha_b, seed=7)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(sha_a)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    spec = write_toy_classification_task(tmp_path / "tasks", "acme/tiny-base", n_train=8, n_valid=4)
    lines: list[str] = []
    result = train_runner.run_training(spec, backend="torch", profile="quick", log=lines.append)
    assert result.log["base_revision"] == result.log["torch_base_revision"] == sha_a
    assert read(result.run_dir / "train_config.json")["base_revision"] == sha_a
    adapter_config = read(result.run_dir / "adapter" / "adapter_config.json")
    assert (adapter_config["base_model_name_or_path"], adapter_config["revision"]) == ("acme/tiny-base", sha_a)
    assert any(f"pinned to revision {sha_a}" in line for line in lines)

    (repo / "refs" / "main").write_text(sha_b)  # the default branch moves after training
    adapter = str(result.run_dir / "adapter")
    messages = [{"role": "system", "content": TOY_SYSTEM_PROMPT}, {"role": "user", "content": "My card is broken"}]

    def logprobs(backend: Any) -> Any:
        backend.load()
        return backend.start(backend.prompt_ids(messages)).logprobs()

    evaluated = eval_runner._default_factory("torch")("acme/tiny-base", adapter)
    first = logprobs(evaluated)
    assert evaluated.source_revision == sha_a
    assert Path(evaluated.model.get_base_model().config._name_or_path).name == sha_a

    served = resolve_run(spec, result.run_id)
    serving = load_backend("torch", served.base_model, str(served.adapter_path))
    np.testing.assert_allclose(logprobs(serving), first, atol=1e-6)
    assert serving.source_revision == sha_a

    moved = TorchBackend("acme/tiny-base", adapter, revision=sha_b)  # an explicit revision still wins
    assert not np.allclose(logprobs(moved), first, atol=1e-3)


def test_adapter_base_revision_reads_only_a_record_for_the_same_base(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from taskdistill.backends.torch_backend import adapter_base_revision

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    assert adapter_base_revision(str(adapter), "acme/base") is None  # no adapter_config.json
    config = {"base_model_name_or_path": "acme/base", "revision": "c" * 40}
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    assert adapter_base_revision(str(adapter), "acme/base") == "c" * 40
    assert adapter_base_revision(str(adapter), "acme/other") is None
    assert adapter_base_revision(None, "acme/base") is None
