"""LoRA fine-tuning on PyTorch: ``taskdistill train --backend torch``.

Hugging Face ``transformers`` + PEFT by default; Unsloth's ``FastLanguageModel`` when
:func:`taskdistill.backends.torch_backend.select_loader` picks it (Unsloth importable and CUDA
available). The training contract is the MLX trainer's (:mod:`taskdistill.train.mlx_lora`):

- the loss covers the completion only: the tokens of ``messages[:-1]`` plus the generation prompt are
  labelled ``-100``, the prefix mlx-lm's ``mask_prompt`` masks, so the loss tokens are the answer,
  ``<|im_end|>`` and the newline after it; examples longer than ``max_seq_len`` are dropped and counted;
- LoRA on ``q, k, v, o, gate, up, down`` of the last ``lora_layers`` blocks (every block for ``"all"``),
  rank ``lora_rank``, no dropout;
- LoRA scale: mlx-lm adds ``scale * x A B`` with ``scale = 20``, PEFT adds ``(lora_alpha / r) * x A B``,
  so ``lora_alpha = 20 * lora_rank`` gives the same update (both start ``A`` uniform in
  ``±1/sqrt(fan_in)`` and ``B`` at zero, so the adapter starts as a no-op);
- AdamW with weight decay 0 (the same update as the MLX trainer's Adam) at a constant learning rate;
  a float16 base (Unsloth on GPUs without bfloat16) adds ``torch.amp.GradScaler`` loss scaling;
- full batches from a seeded permutation of the training rows per epoch (``random.Random(seed)``, as
  the MLX trainer), an epoch's last batch completed from the next permutation; right padding; the
  batch size is capped at the rows left after the length filter, as in the MLX trainer;
- validation loss (token-weighted mean over the loss tokens, validation rows in file order) before the
  first step, every ``steps_per_eval`` steps and after the last step; the checkpoint with the lowest
  validation loss (:func:`taskdistill.evaluate.selection.select_checkpoint`) is written to
  ``run_dir/adapter`` with PEFT's ``save_pretrained``.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from taskdistill.backends.torch_backend import (
    LORA_TARGET_MODULES,
    chat_ids,
    load_causal_lm,
    local_model_dir,
    model_device,
    resolve_torch_base,
    select_loader,
    torch_device,
)
from taskdistill.hardware import load_average
from taskdistill.paths import relative_to_home
from taskdistill.train import common

__all__ = [
    "IGNORE_INDEX",
    "batch_schedule",
    "encode_example",
    "layers_to_transform",
    "lora_alpha_for",
    "train_torch",
]

IGNORE_INDEX = -100
REPORT_EVERY = 10

Example = tuple[list[int], list[int]]
Logger = Callable[[str], None]


def lora_alpha_for(rank: int) -> int:
    """PEFT ``lora_alpha`` for mlx-lm's LoRA scale (PEFT scales the update by ``alpha / r``)."""
    return round(common.LORA_SCALE * rank)


def layers_to_transform(num_layers: int, lora_layers: int | str) -> list[int] | None:
    """Indices of the last ``lora_layers`` blocks, or ``None`` for all of them (mlx-lm's ``num_layers``).

    An integer is clamped to ``1..num_layers`` exactly as the MLX trainer does, so the same task spec
    trains the same blocks on both backends.
    """
    if lora_layers == "all":
        return None
    n = max(1, min(int(lora_layers), num_layers))
    if n == num_layers:
        return None
    return list(range(num_layers - n, num_layers))


def encode_example(tokenizer: Any, messages: Sequence[dict[str, str]]) -> Example:
    """``(input_ids, labels)`` of one chat example, with the prompt tokens labelled ``IGNORE_INDEX``.

    The prompt is ``messages[:-1]`` rendered with the generation prompt when the last message is the
    assistant's (mlx-lm's ``mask_prompt`` rule).
    """
    messages = list(messages)
    if len(messages) < 2:
        raise ValueError("a training example needs a prompt and a completion message")
    full = chat_ids(tokenizer, messages, add_generation_prompt=False)
    prompt = chat_ids(tokenizer, messages[:-1], add_generation_prompt=messages[-1].get("role") == "assistant")
    offset = min(len(prompt), len(full))
    return full, [IGNORE_INDEX] * offset + full[offset:]


def batch_schedule(n: int, batch_size: int, iters: int, seed: int) -> list[list[int]]:
    """Row indices of every training batch: a seeded permutation per epoch, wrapping into the next one.

    Every batch is full; an epoch's last batch is completed from the next epoch's permutation. The
    permutations come from ``random.Random(seed).shuffle``, as in the MLX trainer.
    """
    if n <= 0:
        raise ValueError("no training examples")
    rng = random.Random(seed)
    order: list[int] = []
    position = 0
    batches: list[list[int]] = []
    for _ in range(iters):
        batch: list[int] = []
        while len(batch) < batch_size:
            if position >= len(order):
                order = list(range(n))
                rng.shuffle(order)
                position = 0
            take = min(batch_size - len(batch), len(order) - position)
            batch.extend(order[position : position + take])
            position += take
        batches.append(batch)
    return batches


def _encode_split(tokenizer: Any, path: Path, max_seq_len: int) -> tuple[list[Example], dict[str, int]]:
    """Encoded examples of a run data file; examples longer than ``max_seq_len`` are dropped and counted."""
    rows = common.read_jsonl(path) if path.is_file() else []
    encoded = [encode_example(tokenizer, row["messages"]) for row in rows]
    kept = [ex for ex in encoded if len(ex[0]) <= max_seq_len]
    stats = {
        "rows": len(rows),
        "dropped_too_long": len(encoded) - len(kept),
        "max_tokens": max((len(ex[0]) for ex in kept), default=0),
    }
    return kept, stats


def _collate(examples: Sequence[Example], pad_id: int, device: str) -> dict[str, Any]:
    """Right-padded ``input_ids``, ``attention_mask`` and ``labels`` (``IGNORE_INDEX`` on padding)."""
    import torch

    width = max(len(ids) for ids, _ in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((len(examples), width), dtype=torch.long)
    for row, (ids, labs) in enumerate(examples):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[row, : len(labs)] = torch.tensor(labs, dtype=torch.long)
        attention[row, : len(ids)] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": attention.to(device), "labels": labels.to(device)}


def _loss_tokens(batch: dict[str, Any]) -> int:
    """Tokens that contribute to the loss (the model shifts the labels by one internally)."""
    return int((batch["labels"][:, 1:] != IGNORE_INDEX).sum().item())


def _chunks(items: Sequence[Example], size: int) -> Iterator[Sequence[Example]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _peak_memory_gb(device: str) -> tuple[float | None, str]:
    """Peak memory in GB and how it was measured."""
    import torch

    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9, "torch.cuda.max_memory_allocated"
    try:
        import resource
    except ImportError:  # Windows
        return None, "unavailable"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0  # bytes on macOS, KiB on Linux
    return rss * scale / 1e9, "process peak RSS (resource.getrusage ru_maxrss): CPU run, whole Python process"


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _compute_dtype(model: Any) -> Any:
    """Dtype of the frozen floating-point weights: the base model's compute dtype."""
    import torch

    for param in model.parameters():
        if not param.requires_grad and param.is_floating_point():
            return param.dtype
    return torch.float32


def _grad_scaler(model: Any, device: str, loader: str) -> Any:
    """A ``GradScaler`` for a float16 base (Unsloth picks float16 on GPUs without bfloat16), else None.

    Without loss scaling, activation gradients through frozen float16 blocks can underflow to zero.
    For Unsloth the scaler is also attached as ``accelerator_scaler`` down the ``.model`` chain, as its
    trainer patch does, because its fused cross-entropy reads it.
    """
    import torch

    if _compute_dtype(model) != torch.float16:
        return None
    scaler = torch.amp.GradScaler(device)
    current = model
    while loader == "unsloth":
        current.accelerator_scaler = scaler
        inner = getattr(current, "model", None)
        if not isinstance(inner, torch.nn.Module) or inner is current:
            break
        current = inner
    return scaler


def _load_for_training(cfg: Any, loader: str) -> tuple[Any, Any, str, dict[str, Any]]:
    """The base model wrapped with fresh LoRA adapters, its tokenizer, the device and facts for the log."""
    source, revision = resolve_torch_base(cfg.base_model, cfg.base_revision)
    alpha = lora_alpha_for(cfg.lora_rank)
    if loader == "unsloth":
        from unsloth import FastLanguageModel

        kwargs: dict[str, Any] = {} if revision is None else {"revision": revision}
        base, tokenizer = FastLanguageModel.from_pretrained(
            model_name=source, max_seq_length=cfg.max_seq_len, dtype=None, load_in_4bit=True, **kwargs
        )
        n_layers = int(base.config.num_hidden_layers)
        model = FastLanguageModel.get_peft_model(
            base,
            r=cfg.lora_rank,
            target_modules=list(LORA_TARGET_MODULES),
            lora_alpha=alpha,
            lora_dropout=0,
            bias="none",
            layers_to_transform=layers_to_transform(n_layers, cfg.lora_layers),
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )
        device = model_device(model, default="cuda")
        dtype_name = f"{_dtype_name(_compute_dtype(model))} compute, 4-bit base (load_in_4bit)"
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer

        device, dtype = torch_device()
        path = local_model_dir(source, revision)
        base = load_causal_lm(path, dtype)
        base.to(device)
        tokenizer = AutoTokenizer.from_pretrained(path)
        n_layers = int(base.config.num_hidden_layers)
        lora = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=alpha,
            lora_dropout=common.LORA_DROPOUT,
            bias="none",
            target_modules=list(LORA_TARGET_MODULES),
            layers_to_transform=layers_to_transform(n_layers, cfg.lora_layers),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)
        dtype_name = _dtype_name(dtype)
    blocks = layers_to_transform(n_layers, cfg.lora_layers) or list(range(n_layers))
    info = {
        "loader": loader,
        "device": device,
        "dtype": dtype_name,
        "torch_base_model": source,
        "torch_base_revision": revision,
        "lora_alpha": alpha,
        "lora_scale": alpha / cfg.lora_rank,
        "lora_dropout": common.LORA_DROPOUT,
        "lora_blocks": len(blocks),
        "target_modules": list(LORA_TARGET_MODULES),
    }
    return model, tokenizer, device, info


def _versions(loader: str) -> dict[str, str | None]:
    versions = common.library_versions("torch")
    if loader == "unsloth":
        import unsloth

        versions["unsloth"] = str(getattr(unsloth, "__version__", None) or "unknown")
    return versions


def _trainable_state(model: Any) -> dict[str, Any]:
    return {name: p.detach().to("cpu", copy=True) for name, p in model.named_parameters() if p.requires_grad}


def _restore_state(model: Any, state: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name].to(param.device))


def _validation_loss(model: Any, batches: Sequence[dict[str, Any]]) -> float:
    """Token-weighted mean cross-entropy over the loss tokens of ``batches``."""
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            ntoks = _loss_tokens(batch)
            if ntoks:
                total += float(model(**batch).loss.float().item()) * ntoks
                count += ntoks
    model.train()
    return total / count if count else float("nan")


def _snapshot_revision(path: str) -> str | None:
    """The commit of a Hugging Face cache snapshot directory (``.../snapshots/<sha>``), else None."""
    p = Path(path)
    if p.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", p.name):
        return p.name
    return None


def _tidy_adapter_dir(adapter_dir: Path, *, hub_base: str | None = None, revision: str | None = None) -> None:
    """Make PEFT's files reproducible and free of absolute paths.

    ``target_modules`` is saved from a set, whose order changes between processes, so it is sorted. PEFT
    records the directory the base was loaded from: for a hub base (``hub_base``) that is a cache
    snapshot, so it is replaced by the repo id and the snapshot's commit goes to ``revision``, which is
    what ``AutoPeftModelForCausalLM`` loads; a local base directory is made workspace-relative. A repo id
    recorded as such (Unsloth's pre-quantised repos) is kept. ``README.md`` gets the same base.
    """
    path = adapter_dir / "adapter_config.json"
    if not path.is_file():
        return
    config = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(config.get("target_modules"), list):
        config["target_modules"] = sorted(config["target_modules"])
    base = config.get("base_model_name_or_path")
    if isinstance(base, str) and Path(base).is_absolute():
        if hub_base is not None:
            new_base = hub_base
            config["revision"] = revision or config.get("revision") or _snapshot_revision(base)
        else:
            new_base = relative_to_home(base)
        config["base_model_name_or_path"] = new_base
        card = adapter_dir / "README.md"
        if card.is_file():
            card.write_text(card.read_text(encoding="utf-8").replace(base, new_base), encoding="utf-8")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_torch(cfg: common.TrainConfig, spec: Any, *, log: Logger = print) -> dict[str, Any]:
    """Train the LoRA adapter planned in ``cfg`` and return the train log.

    Writes the run's data (``prepare_run_data``, idempotent), ``run_dir/adapter`` (the checkpoint with
    the lowest validation loss; the final weights when there is no validation data), ``train_log.json``
    and ``loss.png``.
    """
    import torch

    from taskdistill.evaluate.selection import select_checkpoint

    started = time.perf_counter()
    load_before = load_average()
    common.prepare_run_data(spec, cfg)
    data_dir = Path(cfg.data_dir)

    _seed_everything(cfg.seed)
    loader = select_loader()
    model, tokenizer, device, info = _load_for_training(cfg, loader)
    model.train()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    train_rows, train_stats = _encode_split(tokenizer, data_dir / "train.jsonl", cfg.max_seq_len)
    valid_rows, valid_stats = _encode_split(tokenizer, data_dir / "valid.jsonl", cfg.max_seq_len)
    if not train_rows:
        raise common.TrainDataError(
            f"no training example fits max_seq_len={cfg.max_seq_len} in {relative_to_home(data_dir / 'train.jsonl')}"
        )
    batch_size = min(cfg.batch_size, len(train_rows))  # as the MLX trainer, after the length filter
    valid_batches = [_collate(chunk, pad_id, device) for chunk in _chunks(valid_rows, batch_size)]
    if cfg.val_batches > 0:
        valid_batches = valid_batches[: cfg.val_batches]
    val_split = common.validation_split_for(cfg)

    schedule = batch_schedule(len(train_rows), batch_size, cfg.iters, cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    scaler = _grad_scaler(model, device, loader)
    n_trainable = sum(p.numel() for p in params)
    log(
        f"{cfg.run_id}: {len(train_rows)} train / {len(valid_rows)} valid examples, {cfg.iters} iterations x batch "
        f"{batch_size}, LoRA rank {cfg.lora_rank} (alpha {info['lora_alpha']}) on {info['lora_blocks']} blocks "
        f"({n_trainable / 1e6:.2f}M parameters), {loader} on {device}, {info['dtype']}"
        + (", float16 loss scaling" if scaler is not None else "")
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_curve: list[list[float]] = []
    val_curve: list[list[float]] = []
    history: list[tuple[int, float]] = []
    best_state: dict[str, Any] | None = None
    val_seconds = 0.0

    def validate(iteration: int) -> None:
        nonlocal best_state, val_seconds
        if not valid_batches:
            return
        tic = time.perf_counter()
        loss = _validation_loss(model, valid_batches)
        val_curve.append([iteration, loss])
        if math.isfinite(loss):
            history.append((iteration, loss))
            if select_checkpoint(val_split, history) == iteration:
                best_state = _trainable_state(model)
        val_seconds += time.perf_counter() - tic
        log(f"iteration {iteration}: validation loss {loss:.4f}")

    validate(0)
    train_seconds = 0.0
    processed_tokens = trained_tokens = 0
    report_every = max(1, min(cfg.steps_per_eval, REPORT_EVERY))
    window_loss, window_tokens = 0.0, 0
    for it, indices in enumerate(schedule, start=1):
        tic = time.perf_counter()
        batch = _collate([train_rows[i] for i in indices], pad_id, device)
        ntoks = _loss_tokens(batch)
        loss = model(**batch).loss
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        value = float(loss.detach().float().item())  # synchronises the device
        train_seconds += time.perf_counter() - tic
        processed_tokens += int(batch["attention_mask"].sum().item())
        trained_tokens += ntoks
        window_loss += value * ntoks
        window_tokens += ntoks
        if it % report_every == 0 or it == cfg.iters:
            mean = window_loss / window_tokens if window_tokens else float("nan")
            train_curve.append([it, mean])
            log(f"iteration {it}: train loss {mean:.4f}")
            window_loss, window_tokens = 0.0, 0
        if it % cfg.steps_per_eval == 0 or it == cfg.iters:
            validate(it)

    if history:
        best_iteration = select_checkpoint(val_split, history)
        best_val: float | None = dict(history)[best_iteration]
        if best_state is not None:
            _restore_state(model, best_state)
    else:
        best_iteration, best_val = cfg.iters, None
    adapter_dir = Path(cfg.run_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    source = str(info["torch_base_model"])
    _tidy_adapter_dir(
        adapter_dir, hub_base=None if Path(source).is_dir() else source, revision=info["torch_base_revision"]
    )

    peak_gb, peak_source = _peak_memory_gb(device)
    measurements: dict[str, Any] = {
        "backend": "torch",
        "iterations": cfg.iters,
        "epochs": round(cfg.iters * batch_size / len(train_rows), 4),
        "batch_size": batch_size,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "train_seconds": round(train_seconds, 2),
        "val_seconds": round(val_seconds, 2),
        "peak_memory_gb": None if peak_gb is None else round(peak_gb, 3),
        "peak_memory_source": peak_source,
        "processed_tokens": processed_tokens,
        "trained_tokens": trained_tokens,
        "tokens_per_second": round(processed_tokens / train_seconds, 1) if train_seconds > 0 else None,
        "curve": {"train": train_curve, "val": val_curve},
        "best_iteration": best_iteration,
        "best_val_loss": best_val,
        "final_val_loss": val_curve[-1][1] if val_curve else None,
        "load_average": load_before,
        "versions": _versions(loader),
        "optimizer": "adamw (weight_decay 0)",
        "loss_scaling": None if scaler is None else "torch.amp.GradScaler (float16 compute)",
        "loss": "completion tokens only",
        "trainable_parameters": n_trainable,
        "steps_per_eval": cfg.steps_per_eval,
        "val_batches": cfg.val_batches,
        "max_tokens": {"train": train_stats["max_tokens"], "valid": valid_stats["max_tokens"]},
        "dropped_too_long": {"train": train_stats["dropped_too_long"], "valid": valid_stats["dropped_too_long"]},
        **info,
    }
    path = common.write_train_log(cfg, measurements)
    train_log: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    best_text = "" if best_val is None else f", best validation loss {best_val:.4f}"
    log(
        f"{cfg.run_id}: done in {train_log['wall_seconds']} s (training {train_log['train_seconds']} s, "
        f"{train_log['tokens_per_second']} tokens/s){best_text} at iteration {best_iteration}; "
        f"adapter in {train_log['adapter_dir']}"
    )
    return train_log
