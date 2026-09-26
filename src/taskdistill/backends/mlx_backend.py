"""MLX backend: mlx-lm models (4-bit base plus an optional LoRA adapter) on Apple Silicon.

MLX binds unevaluated graphs and explicitly created streams to the thread that made them. ``load()``
therefore evaluates every parameter before it returns (mlx-lm reads adapter weights lazily), so a
backend loaded on one thread can run on another; each decoding session stays on the thread that
started it. Every decoding step is evaluated before it returns, and the session hands out NumPy arrays
only.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from taskdistill.backends.base import Backend
from taskdistill.backends.factory import configure_mlx
from taskdistill.confidence import LabelTrie
from taskdistill.models import resolve_model_path

FUSE_ENV = "TASKDISTILL_MLX_FUSE"
PREFILL_STEP = 2048


class MLXDecodeSession:
    """KV-cached greedy decoding state; ``logprobs()`` is a float32 log-softmax over the model's vocabulary."""

    def __init__(self, model: Any, prompt_ids: list[int], *, prefill_step: int = PREFILL_STEP) -> None:
        if not prompt_ids:
            raise ValueError("cannot start decoding from an empty prompt")
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        self._mx = mx
        self._model = model
        self._cache = make_prompt_cache(model)
        tokens = mx.array(prompt_ids, dtype=mx.int32)
        while tokens.size > prefill_step:
            model(tokens[:prefill_step][None], cache=self._cache)
            mx.eval([c.state for c in self._cache])
            tokens = tokens[prefill_step:]
        self._logprobs = self._last_logprobs(model(tokens[None], cache=self._cache))
        self.length = len(prompt_ids)

    def _last_logprobs(self, logits: Any) -> np.ndarray:
        mx = self._mx
        last = logits[0, -1, :].astype(mx.float32)  # 4-bit Qwen models return float16 logits
        logprobs = last - mx.logsumexp(last)
        mx.eval(logprobs)
        return np.array(logprobs, dtype=np.float32)

    def logprobs(self) -> np.ndarray:
        return self._logprobs

    def feed(self, token_id: int) -> None:
        step = self._mx.array([[int(token_id)]], dtype=self._mx.int32)
        self._logprobs = self._last_logprobs(self._model(step, cache=self._cache))
        self.length += 1


class MLXBackend(Backend):
    """An mlx-lm model; loads on the calling thread (lazily on first use if ``load()`` was not called)."""

    name = "mlx"

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        *,
        revision: str | None = None,
        prefill_step: int = PREFILL_STEP,
        fuse: bool | None = None,
    ) -> None:
        super().__init__(base_model, None if adapter_path is None else str(adapter_path))
        self.revision = revision
        self.prefill_step = prefill_step
        if fuse is None:
            fuse = os.environ.get(FUSE_ENV, "1").strip().lower() not in ("0", "false", "no")
        self.fuse = fuse
        self.fused = False
        self.model: Any = None
        self.vocab_size: int | None = None
        self._wrapper: Any = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        with self._load_lock:
            if self.model is not None:
                return
            mx = configure_mlx()
            from mlx_lm import load as mlx_load

            if self.adapter_path is not None:
                check_adapter_base(self.adapter_path, self.base_model)
            path = resolve_model_path(self.base_model, self.revision)
            loaded = mlx_load(str(path), adapter_path=self.adapter_path)
            model, wrapper = loaded[0], loaded[1]
            if self.adapter_path is not None and self.fuse:
                self.fused = _fuse_lora(model) > 0
            model.eval()
            mx.eval(model.parameters())  # lazily loaded adapter weights would stay bound to this thread
            tokenizer = getattr(wrapper, "_tokenizer", wrapper)
            self._wrapper = wrapper
            self.tokenizer = tokenizer
            self.stop_ids = _stop_ids(tokenizer, getattr(wrapper, "eos_token_ids", None))
            args = getattr(model, "args", None)
            self.vocab_size = getattr(args, "vocab_size", None)
            self.model = model

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load()

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        """Chat template with the generation prompt, rendered as in training, then tokenised."""
        self._ensure_loaded()
        text = self._wrapper.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def start(self, prompt_ids: list[int]) -> MLXDecodeSession:
        self._ensure_loaded()
        return MLXDecodeSession(self.model, prompt_ids, prefill_step=self.prefill_step)

    def label_trie(self, labels: list[str], messages: list[dict[str, str]] | None = None) -> LabelTrie:
        self._ensure_loaded()
        return super().label_trie(labels, messages)


def _fuse_lora(model: Any) -> int:
    """Merge LoRA weights into the (re-quantised) base layers in memory; returns the number fused.

    An unfused adapter runs two extra matmuls per projection and roughly doubles decoding latency. The merge
    is the same operation as ``mlx_lm fuse`` without writing anything to disk; eval and serve both load
    through this backend, so thresholds are chosen on the same fused model that serves.
    """
    from mlx.utils import tree_unflatten

    fused = [(name, module.fuse(dequantize=False)) for name, module in model.named_modules() if hasattr(module, "fuse")]
    if fused:
        model.update_modules(tree_unflatten(fused))
    return len(fused)


def check_adapter_base(adapter_path: str, base_model: str) -> None:
    """Refuse an adapter trained on a different base (mlx-lm loads adapter weights non-strictly)."""
    config_file = Path(adapter_path) / "adapter_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"no adapter_config.json in the adapter directory {Path(adapter_path).name}")
    trained_on = json.loads(config_file.read_text(encoding="utf-8")).get("base_model")
    if not trained_on or trained_on == base_model or Path(trained_on).exists() or Path(base_model).exists():
        return
    raise ValueError(f"this adapter was trained on {trained_on}, not {base_model}; pass --base {trained_on}")


def _special_id(tokenizer: Any, token: str) -> int | None:
    value = tokenizer.convert_tokens_to_ids(token)
    if isinstance(value, int) and value >= 0 and value != getattr(tokenizer, "unk_token_id", None):
        return value
    return None


def _stop_ids(tokenizer: Any, eos_ids: Any) -> list[int]:
    """The chat end token (``<|im_end|>``) first, then every end-of-sequence id and ``<|endoftext|>``."""
    ids: list[int] = []
    candidates = [_special_id(tokenizer, "<|im_end|>"), *sorted(eos_ids or [])]
    candidates += [getattr(tokenizer, "eos_token_id", None), _special_id(tokenizer, "<|endoftext|>")]
    for value in candidates:
        if isinstance(value, int) and value >= 0 and value not in ids:
            ids.append(value)
    return ids
