"""PyTorch backend: Hugging Face ``transformers`` + PEFT, or Unsloth's loader on CUDA when it is installed.

The loader is chosen by :func:`select_loader`, which training uses as well. Unsloth is used only when
it is importable **and** CUDA is available; everywhere else (CPU, Apple Silicon, CUDA without Unsloth)
the model loads with ``AutoModelForCausalLM`` in float32 on CPU or bfloat16 on CUDA, and a LoRA
adapter directory written by PEFT's ``save_pretrained`` is applied with ``PeftModel.from_pretrained``.
The output projection runs in float32 with either loader (:func:`float32_output_head`), so logits
are not rounded to the bfloat16 grid before the float32 log-softmax.

MLX 4-bit repositories (the default student bases) have no torch weights; they are mapped to their
full-precision ``Qwen/...`` equivalents through :data:`taskdistill.models.TORCH_EQUIVALENTS`, with a
warning. Local directories load as they are. A Hugging Face base with no revision given loads the commit
the adapter was trained on (``revision`` in its ``adapter_config.json``), not whatever the default branch
points at now.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np

from taskdistill.backends.base import Backend
from taskdistill.confidence import LabelTrie

__all__ = [
    "LORA_TARGET_MODULES",
    "TorchBackend",
    "TorchSession",
    "adapter_base_revision",
    "chat_ids",
    "float32_output_head",
    "load_causal_lm",
    "local_model_dir",
    "model_device",
    "resolve_torch_base",
    "select_loader",
    "stop_token_ids",
    "torch_device",
]


Loader = Literal["unsloth", "transformers"]

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_MAX_SEQ_LEN = 2048


def _unsloth_installed() -> bool:
    if "unsloth" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("unsloth") is not None
    except (ImportError, ValueError):
        return False


def select_loader() -> Loader:
    """``"unsloth"`` when Unsloth is importable and CUDA is available, else ``"transformers"``.

    Unsloth is imported here: it patches transformers and PEFT, so it should come first, and it raises
    on hosts without a supported GPU, in which case the loader falls back to ``"transformers"``.
    """
    if not _unsloth_installed():
        return "transformers"
    import torch

    if not torch.cuda.is_available():
        return "transformers"
    try:
        import unsloth  # noqa: F401
    except Exception as exc:
        warnings.warn(
            f"unsloth is installed but could not be imported ({type(exc).__name__}: {exc}); "
            "using transformers + PEFT instead",
            RuntimeWarning,
            stacklevel=2,
        )
        return "transformers"
    return "unsloth"


def torch_device() -> tuple[str, Any]:
    """The device and weight dtype for the transformers loader: CUDA/bfloat16 or CPU/float32."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def resolve_torch_base(base_model: str, revision: str | None = None) -> tuple[str, str | None]:
    """Map a student base to something torch can load: ``(local dir or hub repo id, revision)``.

    Existing local directories are returned unchanged. MLX repositories listed in
    ``TORCH_EQUIVALENTS`` are replaced by their torch equivalent (the pinned MLX revision does not
    apply to it, so the revision is dropped) with a warning.
    """
    if Path(base_model).expanduser().is_dir():
        return str(Path(base_model).expanduser()), None
    from taskdistill.models import TORCH_EQUIVALENTS

    target = TORCH_EQUIVALENTS.get(base_model)
    if target is not None:
        warnings.warn(
            f"{base_model} holds MLX weights; the torch backend uses {target} instead",
            UserWarning,
            stacklevel=2,
        )
        return target, None
    if base_model.startswith("mlx-community/"):
        raise ValueError(
            f"{base_model} holds MLX weights and has no known torch equivalent; pass --base with a "
            "transformers checkpoint (a Hugging Face repo id or a local directory)"
        )
    return base_model, revision


def adapter_base_revision(adapter_path: str | None, source: str) -> str | None:
    """The base commit a PEFT adapter directory records (``revision`` in ``adapter_config.json``), else None.

    Only a record for ``source`` itself counts (``base_model_name_or_path`` equal to it): the training run
    writes the Hugging Face repo id there together with the commit of the snapshot it trained on.
    """
    if not adapter_path:
        return None
    try:
        config = json.loads((Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict) or config.get("base_model_name_or_path") != source:
        return None
    revision = config.get("revision")
    return revision if isinstance(revision, str) and revision else None


def local_model_dir(source: str, revision: str | None) -> str:
    if Path(source).is_dir():
        return source
    from taskdistill.models import resolve_model_path

    return str(resolve_model_path(source, revision=revision))


def load_causal_lm(path: str, dtype: Any) -> Any:
    """``AutoModelForCausalLM.from_pretrained(path, dtype=dtype)`` (``dtype=``, not the deprecated ``torch_dtype=``)."""
    from transformers import AutoModelForCausalLM

    auto: Any = AutoModelForCausalLM
    return auto.from_pretrained(path, dtype=dtype)


def model_device(model: Any, default: str = "cpu") -> str:
    """Device type of the model's first parameter (``default`` when it has none)."""
    try:
        return str(next(iter(model.parameters())).device.type)
    except (StopIteration, AttributeError, TypeError):
        return default


def chat_ids(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    """Render the chat template as text, then encode it without adding special tokens."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]


def stop_token_ids(tokenizer: Any, model: Any = None) -> list[int]:
    """``<|im_end|>`` first (the chat end token the label trie terminates with), then every EOS id."""
    ids: list[int] = []
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        ids.append(int(vocab["<|im_end|>"]))
    candidates: list[Any] = [tokenizer.eos_token_id]
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        candidates.append(generation_config.eos_token_id)
    for value in candidates:
        for token in value if isinstance(value, (list, tuple)) else [value]:
            if token is not None and int(token) not in ids:
                ids.append(int(token))
    if not ids:
        raise ValueError("the tokenizer defines neither <|im_end|> nor an EOS token")
    return ids


def _float32_input(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
    return (args[0].float(), *args[1:])


def float32_output_head(model: Any) -> str | None:
    """Make the output projection produce float32 logits; returns the logits dtype (None when unknown).

    A bfloat16 ``lm_head`` rounds the logits to bfloat16 (a spacing of 0.125 between 16 and 32) before
    the log-softmax, which coarsens the label probabilities that confidence scores come from. A head
    that is not float32 is replaced by a float32 copy whose input is cast to float32; that costs
    ``vocab x hidden x 4`` bytes and happens only for half-precision models (CUDA). The copy is a plain
    ``nn.Linear``, so Unsloth's single-token path (``torch.mv`` on ``lm_head.weight``) uses it too.
    """
    import torch

    inner = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    get_head = getattr(inner, "get_output_embeddings", None)
    head = get_head() if callable(get_head) else None
    if type(head) is not torch.nn.Linear or not head.weight.is_floating_point():
        return None
    if head.weight.dtype == torch.float32:
        return "float32"
    fp32 = torch.nn.Linear(head.in_features, head.out_features, bias=head.bias is not None, device="meta")
    fp32.weight = torch.nn.Parameter(head.weight.detach().to(torch.float32), requires_grad=False)
    if head.bias is not None:
        fp32.bias = torch.nn.Parameter(head.bias.detach().to(torch.float32), requires_grad=False)
    fp32.register_forward_pre_hook(_float32_input)
    inner.set_output_embeddings(fp32)
    return "float32"


class TorchSession:
    """Incremental decoding over the model's KV cache (``past_key_values``), batch size 1.

    Position ids are always passed explicitly: Unsloth's cached single-token forward reads them and
    fails without them (``generate`` normally supplies them); for transformers they match the defaults.
    """

    def __init__(self, model: Any, prompt_ids: list[int], device: str) -> None:
        if not prompt_ids:
            raise ValueError("empty prompt")
        self._model = model
        self._device = device
        self._past: Any = None
        self._length = 0
        self._logprobs = self._forward(prompt_ids)

    def _forward(self, ids: list[int]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
            positions = torch.arange(self._length, self._length + len(ids), dtype=torch.long, device=self._device)
            out = self._model(
                input_ids=input_ids,
                position_ids=positions[None],
                past_key_values=self._past,
                use_cache=True,
                logits_to_keep=1,
            )
            self._past = out.past_key_values
            self._length += len(ids)
            logits = out.logits[0, -1].float()
            logprobs = torch.log_softmax(logits, dim=-1)
        return np.asarray(logprobs.cpu().numpy(), dtype=np.float32)

    def logprobs(self) -> np.ndarray:
        return self._logprobs

    def feed(self, token_id: int) -> None:
        self._logprobs = self._forward([int(token_id)])


class TorchBackend(Backend):
    """Student backend on PyTorch (CPU or CUDA); loads on the calling thread, lazily on first use."""

    name = "torch"

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        *,
        revision: str | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        super().__init__(base_model, None if adapter_path is None else str(adapter_path))
        self.revision = revision
        self.max_seq_len = max_seq_len
        self.model: Any = None
        self.loader: Loader | None = None
        self.device = "cpu"
        self.source: str | None = None
        self.source_revision: str | None = None  # the base commit loaded (None: a local directory or the default)
        self.logits_dtype: str | None = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        with self._load_lock:
            if self.model is not None:
                return
            loader = select_loader()
            source, revision = resolve_torch_base(self.base_model, self.revision)
            if revision is None and not Path(source).is_dir():
                revision = adapter_base_revision(self.adapter_path, source)  # the commit the adapter was trained on
            if loader == "unsloth":
                model, tokenizer = self._load_unsloth(source, revision)
                self.device = model_device(model, default="cuda")
            else:
                model, tokenizer = self._load_transformers(source, revision)
            self.logits_dtype = float32_output_head(model)
            self.loader = loader
            self.source = source
            self.source_revision = revision
            self.tokenizer = tokenizer
            self.stop_ids = stop_token_ids(tokenizer, model)
            self.model = model

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load()

    def _load_transformers(self, source: str, revision: str | None) -> tuple[Any, Any]:
        from transformers import AutoTokenizer

        device, dtype = torch_device()
        path = local_model_dir(source, revision)
        model = load_causal_lm(path, dtype)
        model.to(device)
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(path)
        self.device = device
        return model, tokenizer

    def _load_unsloth(self, source: str, revision: str | None) -> tuple[Any, Any]:
        from unsloth import FastLanguageModel

        kwargs: dict[str, Any] = {}
        if revision is not None:
            kwargs["revision"] = revision
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=source,
            max_seq_length=self.max_seq_len,
            dtype=None,
            load_in_4bit=True,
            **kwargs,
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model = FastLanguageModel.for_inference(model) or model
        return model, tokenizer

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        """Chat template with the generation prompt, rendered as text as in training, then tokenised."""
        self._ensure_loaded()
        return chat_ids(self.tokenizer, messages, add_generation_prompt=True)

    def start(self, prompt_ids: list[int]) -> TorchSession:
        self._ensure_loaded()
        return TorchSession(self.model, prompt_ids, self.device)

    def label_trie(self, labels: list[str], messages: list[dict[str, str]] | None = None) -> LabelTrie:
        self._ensure_loaded()
        return super().label_trie(labels, messages)
