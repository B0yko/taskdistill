"""LoRA fine-tuning of a 4-bit base with mlx-lm: ``python -m taskdistill.train.mlx_lora <train_config.json>``.

The run uses mlx-lm's trainer with three replacements:

- the loss covers completion tokens only, with ``<`` where mlx-lm 0.31.3's ``default_loss`` has ``<=``
  (that version also trains on the first padding token), and the vocabulary projection is applied only
  to the completion positions ("sliced logits"), which cuts memory for long prompts;
- the batch iterator draws a fresh seeded permutation every epoch and completes the last batch of an
  epoch from the next one, so every example is used (mlx-lm fixes batch composition once and never
  trains on the remainder), and it pads like mlx-lm (to 1 + a multiple of 32 tokens);
- a training callback keeps the adapter with the lowest validation loss (chosen by
  ``taskdistill.evaluate.selection.select_checkpoint``), and the final weights are validated after
  ``train()`` returns, since mlx-lm never validates after the last step.

Checkpoints go to ``adapter.partial/``, which becomes ``adapter/`` only when training has finished;
``train_log.json`` is written last and marks a complete run.

Optimiser: Adam, as in ``mlx_lm lora``; weight decay has little to act on in a few hundred LoRA steps,
and the torch trainer's AdamW runs with weight decay 0, which is the same update.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np

from taskdistill.backends.factory import configure_mlx
from taskdistill.hardware import load_average
from taskdistill.models import resolve_model_path
from taskdistill.paths import relative_to_home
from taskdistill.train.common import (
    LORA_DROPOUT,
    LORA_SCALE,
    TrainConfig,
    TrainDataError,
    library_versions,
    read_jsonl,
    validation_split_for,
    write_train_log,
)

PAD_TO = 32
GRAD_CHECKPOINT_TOKENS = 2048  # batch_size x longest example above which activations are recomputed
SLICED_LOSS_MODEL_TYPES = frozenset({"qwen2", "qwen3", "llama"})
LAST_ADAPTER_FILE = "_last_adapters.safetensors"
PARTIAL_ADAPTER_DIR = "adapter.partial"
NEVER = 10**9


def _say(message: str) -> None:
    print(f"[taskdistill] {message}", flush=True)


# -- batching --------------------------------------------------------------------------------------


def collate(
    items: list[tuple[list[int], int]], max_seq_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Pad a batch of ``(tokens, completion_offset)`` pairs.

    Returns the token matrix (padded with 0 to 1 + a multiple of 32, capped at ``max_seq_length``),
    ``(offset, length)`` rows, the positions of the completion targets (width rounded up to a multiple
    of 32, clipped at the last target), the real token count and the completion token count.
    """
    lengths = [min(len(tokens), max_seq_length) for tokens, _ in items]
    offsets = [int(offset) for _, offset in items]
    width = max(2, min(1 + PAD_TO * math.ceil(max(lengths) / PAD_TO), max_seq_length))
    batch = np.zeros((len(items), width), dtype=np.int32)
    for row, ((tokens, _), length) in enumerate(zip(items, lengths, strict=True)):
        batch[row, :length] = tokens[:length]
    starts = [max(offset, 1) for offset in offsets]
    counts = [max(0, length - start) for length, start in zip(lengths, starts, strict=True)]
    window = max(1, min(PAD_TO * math.ceil(max(counts) / PAD_TO), width - 1))
    positions = np.minimum(np.asarray(starts)[:, None] - 1 + np.arange(window)[None, :], width - 2)
    spans = np.asarray(list(zip(offsets, lengths, strict=True)), dtype=np.int32)
    return batch, spans, positions.astype(np.int32), int(sum(lengths)), int(sum(counts))


class BatchIterator:
    """Drop-in replacement for mlx-lm's ``iterate_batches`` (same call signature).

    ``loop=True`` (training): a new seeded permutation every epoch; a batch that runs past the end of
    an epoch is completed from the next permutation, so all batches are full and every example is seen
    once per epoch. ``loop=False`` (validation): the dataset in order; the last batch may be smaller.
    Training batches are counted in ``processed_tokens`` (real, unpadded) and ``trained_tokens``.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.processed_tokens = 0
        self.trained_tokens = 0
        self.train_batches = 0

    def epoch_batches(self, n: int, batch_size: int) -> Iterator[list[int]]:
        order: list[int] = []
        position = 0
        while True:
            batch: list[int] = []
            while len(batch) < batch_size:
                if position >= len(order):
                    order = list(range(n))
                    self._rng.shuffle(order)
                    position = 0
                take = min(batch_size - len(batch), len(order) - position)
                batch.extend(order[position : position + take])
                position += take
            yield batch

    @staticmethod
    def sequential_batches(n: int, batch_size: int) -> Iterator[list[int]]:
        for start in range(0, n, batch_size):
            yield list(range(start, min(start + batch_size, n)))

    def __call__(
        self,
        dataset: Any,
        batch_size: int,
        max_seq_length: int,
        loop: bool = False,
        seed: int | None = None,
        comm_group: Any = None,
    ) -> Iterator[tuple[Any, Any, Any]]:
        import mlx.core as mx

        if comm_group is not None and comm_group.size() > 1:
            raise ValueError("distributed training is not supported")
        n = len(dataset)
        if n == 0:
            return
        groups = self.epoch_batches(n, batch_size) if loop else self.sequential_batches(n, batch_size)
        for indices in groups:
            batch, spans, positions, real, supervised = collate([dataset[i] for i in indices], max_seq_length)
            if loop:
                self.processed_tokens += real
                self.trained_tokens += supervised
                self.train_batches += 1
            yield mx.array(batch), mx.array(spans), mx.array(positions)


# -- loss ------------------------------------------------------------------------------------------


def supports_sliced_logits(model: Any) -> bool:
    """Architectures whose ``__call__`` is exactly ``head(model.model(inputs))`` (tied or separate head)."""
    return getattr(model, "model_type", None) in SLICED_LOSS_MODEL_TYPES and hasattr(model, "model")


def make_loss(model: Any, *, sliced: bool | None = None) -> Callable[..., Any]:
    """Mean cross-entropy over completion tokens: targets at positions ``offset .. length - 1``."""
    import mlx.core as mx
    import mlx.nn as nn

    use_sliced = supports_sliced_logits(model) if sliced is None else sliced

    def head(m: Any, hidden: Any) -> Any:
        if getattr(m.args, "tie_word_embeddings", False):
            return m.model.embed_tokens.as_linear(hidden)
        return m.lm_head(hidden)

    def completion_only_loss(m: Any, batch: Any, spans: Any, positions: Any = None) -> tuple[Any, Any]:
        inputs, targets = batch[:, :-1], batch[:, 1:]
        start = mx.maximum(spans[:, 0:1], 1)
        if use_sliced and positions is not None:
            hidden = mx.take_along_axis(m.model(inputs), positions[..., None], axis=1)
            targets = mx.take_along_axis(targets, positions, axis=1)
            logits = head(m, hidden)
            mask = mx.arange(positions.shape[1])[None, :] < (spans[:, 1:] - start)
        else:
            logits = m(inputs)
            steps = mx.arange(1, targets.shape[1] + 1)[None, :]
            mask = mx.logical_and(steps >= start, steps < spans[:, 1:])
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), targets) * mask
        ntoks = mask.sum()
        return ce.sum() / mx.maximum(ntoks, 1), ntoks

    return completion_only_loss


# -- data ------------------------------------------------------------------------------------------


def build_dataset(rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> tuple[Any, dict[str, int]]:
    """ChatDataset(mask_prompt=True) in a CacheDataset; examples longer than ``max_seq_len`` are dropped."""
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset

    dataset = CacheDataset(ChatDataset(rows, tokenizer, mask_prompt=True))
    lengths = [len(dataset[i][0]) for i in range(len(dataset))]
    keep = [i for i, n in enumerate(lengths) if n <= max_seq_len]
    stats = {
        "rows": len(rows),
        "dropped_too_long": len(rows) - len(keep),
        "max_tokens": max(lengths, default=0),
        "tokens": sum(lengths[i] for i in keep),
    }
    if len(keep) < len(rows):
        dataset = CacheDataset(ChatDataset([rows[i] for i in keep], tokenizer, mask_prompt=True))
        stats["max_tokens"] = max((lengths[i] for i in keep), default=0)
    return dataset, stats


# -- training --------------------------------------------------------------------------------------


def flat_parameters(tree: Any) -> dict[str, Any]:
    """``{"model.layers.0.self_attn.q_proj.lora_a": array, ...}``, the adapter file layout."""
    from mlx.utils import tree_flatten

    return dict(cast(list[tuple[str, Any]], tree_flatten(tree)))


def _seed_everything(seed: int, mx: Any) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    mx.random.seed(seed)


def _clear_outputs(run_dir: Path) -> None:
    """Remove what an earlier run with this id left, so no old log or plot sits beside the new adapter."""
    for name in ("train_log.json", "loss.png", LAST_ADAPTER_FILE):
        (run_dir / name).unlink(missing_ok=True)
    for name in ("adapter", PARTIAL_ADAPTER_DIR):
        if (run_dir / name).is_dir():
            shutil.rmtree(run_dir / name)


def _publish_adapter(partial: Path, final: Path) -> None:
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)


def _portable_base(base_model: str) -> str:
    """A local base directory is recorded relative to the workspace; a Hub id as is."""
    return relative_to_home(base_model) if Path(base_model).is_absolute() else base_model


def train_mlx(cfg: TrainConfig) -> dict[str, Any]:
    """Train the run described by ``cfg``; writes the adapter, ``train_log.json`` and ``loss.png``."""
    started = time.perf_counter()
    load_before = load_average()
    mx = configure_mlx()
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, evaluate, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from taskdistill.evaluate.selection import select_checkpoint

    _seed_everything(cfg.seed, mx)
    mx.reset_peak_memory()
    run_dir = Path(cfg.run_dir)
    data_dir = Path(cfg.data_dir)
    _clear_outputs(run_dir)
    partial_dir = run_dir / PARTIAL_ADAPTER_DIR
    partial_dir.mkdir(parents=True)
    best_file = partial_dir / "adapters.safetensors"
    if not (data_dir / "train.jsonl").is_file():
        raise TrainDataError(f"missing {relative_to_home(data_dir / 'train.jsonl')}; prepare the run data first")

    loaded = load(str(resolve_model_path(cfg.base_model, cfg.base_revision)))
    model, tokenizer = loaded[0], loaded[1]
    model.freeze()
    blocks = len(model.layers)
    num_layers = blocks if cfg.lora_layers == "all" else max(1, min(int(cfg.lora_layers), blocks))
    lora_parameters = {"rank": cfg.lora_rank, "scale": LORA_SCALE, "dropout": LORA_DROPOUT}
    linear_to_lora_layers(model, num_layers, lora_parameters)
    trainable = sum(v.size for v in flat_parameters(model.trainable_parameters()).values())
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_parameters,
        "base_model": _portable_base(cfg.base_model),
        "base_revision": cfg.base_revision,
    }
    (partial_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")

    train_set, train_stats = build_dataset(read_jsonl(data_dir / "train.jsonl"), tokenizer, cfg.max_seq_len)
    valid_rows = read_jsonl(data_dir / "valid.jsonl") if (data_dir / "valid.jsonl").is_file() else []
    valid_set, valid_stats = build_dataset(valid_rows, tokenizer, cfg.max_seq_len)
    if len(train_set) == 0:
        raise TrainDataError(f"no training example fits max_seq_len={cfg.max_seq_len}")
    batch_size = min(cfg.batch_size, len(train_set))
    grad_checkpoint = batch_size * train_stats["max_tokens"] > GRAD_CHECKPOINT_TOKENS
    sliced = supports_sliced_logits(model)
    loss = make_loss(model, sliced=sliced)
    batches = BatchIterator(cfg.seed)
    val_split = validation_split_for(cfg)
    report_every = max(1, min(10, cfg.iters // 20))

    def save_adapter() -> None:
        mx.save_safetensors(str(best_file), flat_parameters(model.trainable_parameters()))

    class BestValidation(TrainingCallback):
        """Records the loss curves and saves the adapter whenever validation loss is the best so far."""

        def __init__(self) -> None:
            self.train_curve: list[list[float]] = []
            self.val_curve: list[list[float]] = []
            self.history: list[tuple[int, float]] = []
            self.best: tuple[int, float] | None = None
            self.val_seconds = 0.0

        def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
            self.train_curve.append([int(train_info["iteration"]), float(train_info["train_loss"])])

        def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
            self.val_seconds += float(val_info.get("val_time", 0.0))
            self.consider(int(val_info["iteration"]), float(val_info["val_loss"]))

        def consider(self, iteration: int, value: float) -> None:
            self.val_curve.append([iteration, value])
            if not math.isfinite(value):
                _say(f"iteration {iteration}: validation loss is not finite; not a checkpoint candidate")
                return
            self.history.append((iteration, value))
            if select_checkpoint(val_split, list(self.history)) == iteration:
                save_adapter()
                self.best = (iteration, value)
                _say(f"iteration {iteration}: best validation loss so far ({value:.4f}); adapter saved")

    recorder = BestValidation()
    in_run_dir = Path.cwd().resolve() == run_dir.resolve()
    args = TrainingArgs(
        batch_size=batch_size,
        iters=cfg.iters,
        val_batches=cfg.val_batches if len(valid_set) else 0,
        steps_per_report=report_every,
        steps_per_eval=cfg.steps_per_eval,
        steps_per_save=NEVER,
        max_seq_length=cfg.max_seq_len,
        adapter_file=LAST_ADAPTER_FILE if in_run_dir else str(run_dir / LAST_ADAPTER_FILE),
        grad_checkpoint=grad_checkpoint,
    )
    _say(
        f"{cfg.run_id}: {len(train_set)} train / {len(valid_set)} valid examples, {cfg.iters} iterations x batch "
        f"{batch_size}, LoRA rank {cfg.lora_rank} on {num_layers}/{blocks} blocks ({trainable / 1e6:.2f}M "
        f"parameters), longest example {train_stats['max_tokens']} tokens, "
        f"gradient checkpointing {'on' if grad_checkpoint else 'off'}"
    )

    train_started = time.perf_counter()
    train(
        model=model,
        optimizer=optim.Adam(learning_rate=cfg.learning_rate),
        train_dataset=train_set,
        val_dataset=valid_set if len(valid_set) else None,
        args=args,
        loss=loss,
        iterate_batches=batches,
        training_callback=recorder,
    )
    train_seconds = time.perf_counter() - train_started - recorder.val_seconds

    final_val: float | None = None
    if len(valid_set):
        tic = time.perf_counter()
        final_val = float(
            evaluate(
                model=model,
                dataset=valid_set,
                batch_size=batch_size,
                num_batches=cfg.val_batches,
                max_seq_length=cfg.max_seq_len,
                loss=loss,
                iterate_batches=batches,
            )
        )
        recorder.val_seconds += time.perf_counter() - tic
        _say(f"iteration {cfg.iters}: validation loss {final_val:.4f} (final weights)")
        recorder.consider(cfg.iters, final_val)
    if recorder.best is None:
        save_adapter()
        recorder.best = (cfg.iters, final_val if final_val is not None else math.nan)
    (run_dir / LAST_ADAPTER_FILE).unlink(missing_ok=True)
    _publish_adapter(partial_dir, run_dir / "adapter")

    peak_gb = mx.get_peak_memory() / 1e9
    wall = time.perf_counter() - started
    log: dict[str, Any] = {
        "n_train": len(train_set),
        "n_valid": len(valid_set),
        "iterations": cfg.iters,
        "epochs": round(cfg.iters * batch_size / len(train_set), 4),
        "batch_size": batch_size,
        "wall_seconds": round(wall, 2),
        "train_seconds": round(train_seconds, 2),
        "val_seconds": round(recorder.val_seconds, 2),
        "peak_memory_gb": round(peak_gb, 3),
        "processed_tokens": batches.processed_tokens,
        "trained_tokens": batches.trained_tokens,
        "tokens_per_second": round(batches.processed_tokens / train_seconds, 1) if train_seconds > 0 else None,
        "curve": {"train": recorder.train_curve, "val": recorder.val_curve},
        "best_iteration": recorder.best[0],
        "best_val_loss": recorder.best[1] if len(valid_set) else None,
        "final_val_loss": final_val,
        "load_average": load_before,
        "versions": library_versions("mlx"),
        "optimizer": "adam",
        "loss": "completion tokens only" + (", sliced logits" if sliced else ""),
        "grad_checkpoint": grad_checkpoint,
        "lora_scale": LORA_SCALE,
        "lora_dropout": LORA_DROPOUT,
        "lora_blocks": num_layers,
        "trainable_parameters": int(trainable),
        "steps_per_eval": cfg.steps_per_eval,
        "val_batches": cfg.val_batches,
        "max_tokens": {"train": train_stats["max_tokens"], "valid": valid_stats["max_tokens"]},
        "dropped_too_long": {"train": train_stats["dropped_too_long"], "valid": valid_stats["dropped_too_long"]},
    }
    write_train_log(cfg, log)
    _say(
        f"{cfg.run_id}: done in {wall:.1f} s (training {train_seconds:.1f} s, "
        f"{log['tokens_per_second']} tokens/s), peak memory {peak_gb:.2f} GB, "
        f"best validation loss at iteration {recorder.best[0]}"
    )
    return log


def main(config: str | Path | list[str] | None = None) -> int:
    """Entry point: ``main(config_json_path)``, or ``main(argv)`` / ``main()`` (``sys.argv``) from the command line."""
    if isinstance(config, (str, Path)):
        args = [str(config)]
    else:
        args = sys.argv[1:] if config is None else list(config)
    if len(args) != 1:
        print("usage: python -m taskdistill.train.mlx_lora <train_config.json>", file=sys.stderr)
        return 2
    cfg = TrainConfig.load(Path(args[0]))
    if cfg.backend != "mlx":
        print(f"train_config.json is for the {cfg.backend} backend, not mlx", file=sys.stderr)
        return 2
    os.chdir(cfg.run_dir)
    train_mlx(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
