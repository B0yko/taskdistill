"""Loader dispatch for the torch path: Unsloth only when it is importable and CUDA is available.

Unsloth is never installed or run here; a fake ``unsloth`` module with a mocked ``FastLanguageModel``
stands in for it, and CUDA availability is patched.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from taskdistill.backends import torch_backend  # noqa: E402
from taskdistill.backends.torch_backend import LORA_TARGET_MODULES, TorchBackend, select_loader  # noqa: E402
from taskdistill.train import torch_lora  # noqa: E402
from taskdistill.train.common import plan_training  # noqa: E402
from tiny_model import IM_END_ID, build_tiny_model, build_tokenizer, write_toy_classification_task  # noqa: E402


@pytest.fixture
def fake_unsloth(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    module = types.ModuleType("unsloth")
    module.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader=None)
    module.__version__ = "0.0.test"  # type: ignore[attr-defined]
    fast = mock.MagicMock(name="FastLanguageModel")
    module.FastLanguageModel = fast  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", module)
    return fast


@pytest.fixture
def cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _hide_unsloth(monkeypatch: pytest.MonkeyPatch, spec: Any) -> None:
    real = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> Any:
        return spec if name == "unsloth" else real(name, package)

    monkeypatch.delitem(sys.modules, "unsloth", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_unsloth_is_selected_when_importable_and_cuda_is_available(fake_unsloth: mock.MagicMock, cuda: None) -> None:
    assert select_loader() == "unsloth"


def test_transformers_is_selected_without_cuda(fake_unsloth: mock.MagicMock, no_cuda: None) -> None:
    assert select_loader() == "transformers"


def test_transformers_is_selected_without_unsloth(monkeypatch: pytest.MonkeyPatch, cuda: None) -> None:
    _hide_unsloth(monkeypatch, None)
    assert select_loader() == "transformers"


def test_an_unsloth_import_failure_falls_back_to_transformers(monkeypatch: pytest.MonkeyPatch, cuda: None) -> None:
    _hide_unsloth(monkeypatch, importlib.machinery.ModuleSpec("unsloth", loader=None))
    monkeypatch.setitem(sys.modules, "unsloth", None)  # makes ``import unsloth`` raise ImportError
    with pytest.warns(RuntimeWarning, match="unsloth is installed but could not be imported"):
        assert select_loader() == "transformers"


def test_backend_load_goes_through_fast_language_model(
    fake_unsloth: mock.MagicMock, cuda: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = mock.MagicMock(name="base")
    base.generation_config.eos_token_id = IM_END_ID
    wrapped = mock.MagicMock(name="with_adapter")
    fake_unsloth.from_pretrained.return_value = (base, build_tokenizer())
    fake_unsloth.for_inference.side_effect = lambda model: model
    from_pretrained = mock.MagicMock(return_value=wrapped)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", from_pretrained)

    adapter = tmp_path / "adapter"
    backend = TorchBackend("mlx-community/Qwen2.5-0.5B-Instruct-4bit", adapter_path=str(adapter))
    with pytest.warns(UserWarning, match="holds MLX weights"):
        backend.load()

    fake_unsloth.from_pretrained.assert_called_once_with(
        model_name="Qwen/Qwen2.5-0.5B-Instruct", max_seq_length=2048, dtype=None, load_in_4bit=True
    )
    from_pretrained.assert_called_once_with(base, str(adapter))
    fake_unsloth.for_inference.assert_called_once_with(wrapped)
    assert backend.loader == "unsloth"
    assert backend.model is wrapped
    assert backend.stop_ids[0] == IM_END_ID
    assert backend.logits_dtype is None  # a mocked model has no real output head to inspect


def test_backend_load_uses_transformers_without_cuda(
    fake_unsloth: mock.MagicMock, no_cuda: None, tmp_path: Path
) -> None:
    backend = TorchBackend(str(build_tiny_model(tmp_path / "tiny")))
    backend.load()
    assert backend.loader == "transformers"
    fake_unsloth.from_pretrained.assert_not_called()


def _peft_stand_in(model: Any, **kwargs: Any) -> Any:
    """What ``FastLanguageModel.get_peft_model`` does, minus the CUDA kernels: wrap with PEFT LoRA."""
    config = peft.LoraConfig(
        r=kwargs["r"],
        lora_alpha=kwargs["lora_alpha"],
        lora_dropout=kwargs["lora_dropout"],
        target_modules=kwargs["target_modules"],
        layers_to_transform=kwargs.get("layers_to_transform"),
        task_type="CAUSAL_LM",
    )
    return peft.get_peft_model(model, config)


def _fake_from_pretrained(**kwargs: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = kwargs["model_name"]
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32), AutoTokenizer.from_pretrained(path)


def _fake_float16_from_pretrained(**kwargs: Any) -> tuple[Any, Any]:
    """``dtype=None`` on a GPU without bfloat16 (T4, V100): Unsloth loads the base in float16."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = kwargs["model_name"]
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16), AutoTokenizer.from_pretrained(path)


def test_training_loads_through_unsloth_with_the_expected_lora_arguments(
    fake_unsloth: mock.MagicMock, cuda: None, tmp_path: Path
) -> None:
    tiny = build_tiny_model(tmp_path / "tiny")
    fake_unsloth.from_pretrained.side_effect = _fake_from_pretrained
    fake_unsloth.get_peft_model.side_effect = _peft_stand_in
    cfg = types.SimpleNamespace(
        base_model=str(tiny), base_revision=None, lora_rank=4, lora_layers=1, max_seq_len=128, seed=7
    )

    loader = select_loader()
    model, _tokenizer, device, info = torch_lora._load_for_training(cfg, loader)

    assert loader == "unsloth"
    fake_unsloth.from_pretrained.assert_called_once_with(
        model_name=str(tiny), max_seq_length=128, dtype=None, load_in_4bit=True
    )
    fake_unsloth.get_peft_model.assert_called_once()
    kwargs = fake_unsloth.get_peft_model.call_args.kwargs
    assert kwargs["r"] == 4
    assert kwargs["lora_alpha"] == torch_lora.lora_alpha_for(4) == 80
    assert kwargs["lora_dropout"] == 0
    assert kwargs["target_modules"] == LORA_TARGET_MODULES
    assert kwargs["layers_to_transform"] == [1]
    assert kwargs["use_gradient_checkpointing"] == "unsloth"
    assert kwargs["random_state"] == 7
    assert device == "cpu"
    assert info["loader"] == "unsloth"
    lora_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert lora_names and all(".layers.1." in name for name in lora_names)


def test_train_torch_runs_through_the_unsloth_branch(
    fake_unsloth: mock.MagicMock, cuda: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    tiny = build_tiny_model(tmp_path / "tiny")
    spec = write_toy_classification_task(tmp_path / "tasks", str(tiny), n_train=8, n_valid=4, epochs=1)
    fake_unsloth.from_pretrained.side_effect = _fake_from_pretrained
    fake_unsloth.get_peft_model.side_effect = _peft_stand_in
    cfg = plan_training(spec, backend="torch", profile="quick")

    log = torch_lora.train_torch(cfg, spec, log=lambda _line: None)

    fake_unsloth.from_pretrained.assert_called_once_with(
        model_name=str(tiny), max_seq_length=cfg.max_seq_len, dtype=None, load_in_4bit=True
    )
    kwargs = fake_unsloth.get_peft_model.call_args.kwargs
    assert (kwargs["r"], kwargs["lora_alpha"], kwargs["lora_dropout"]) == (4, 80, 0)
    assert kwargs["target_modules"] == LORA_TARGET_MODULES
    assert kwargs["layers_to_transform"] is None  # lora_layers: all
    assert kwargs["use_gradient_checkpointing"] == "unsloth"
    assert kwargs["random_state"] == 13
    assert log["loader"] == "unsloth"
    assert log["versions"]["unsloth"] == "0.0.test"
    assert (log["dtype"], log["loss_scaling"]) == ("float32 compute, 4-bit base (load_in_4bit)", None)
    assert (Path(cfg.run_dir) / "adapter" / "adapter_model.safetensors").is_file()


def test_select_loader_is_shared_by_backend_and_training() -> None:
    assert torch_lora.select_loader is torch_backend.select_loader


def test_a_float16_base_trains_with_loss_scaling(
    fake_unsloth: mock.MagicMock, cuda: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    tiny = build_tiny_model(tmp_path / "tiny")
    spec = write_toy_classification_task(tmp_path / "tasks", str(tiny), n_train=8, n_valid=4, epochs=1)
    models: list[Any] = []

    def get_peft_model(model: Any, **kwargs: Any) -> Any:
        models.append(_peft_stand_in(model, **kwargs))
        return models[-1]

    fake_unsloth.from_pretrained.side_effect = _fake_float16_from_pretrained
    fake_unsloth.get_peft_model.side_effect = get_peft_model
    scalers: list[Any] = []
    real_scaler: Any = torch.amp.GradScaler

    class RecordingScaler(real_scaler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.scaled = 0
            scalers.append(self)

        def scale(self, outputs: Any) -> Any:
            self.scaled += 1
            return super().scale(outputs)

    monkeypatch.setattr(torch.amp, "GradScaler", RecordingScaler)
    cfg = plan_training(spec, backend="torch", profile="quick")
    log = torch_lora.train_torch(cfg, spec, log=lambda _line: None)

    (scaler,) = scalers
    assert scaler.scaled == cfg.iters
    assert log["dtype"] == "float16 compute, 4-bit base (load_in_4bit)"
    assert log["loss_scaling"] == "torch.amp.GradScaler (float16 compute)"
    (model,) = models
    causal_lm = model.get_base_model()
    assert model.accelerator_scaler is scaler  # read by Unsloth's fused cross-entropy, as its trainer sets it
    assert causal_lm.accelerator_scaler is scaler and causal_lm.model.accelerator_scaler is scaler
    assert all(math.isfinite(loss) for _, loss in log["curve"]["train"])
    assert log["best_val_loss"] < log["curve"]["val"][0][1]
