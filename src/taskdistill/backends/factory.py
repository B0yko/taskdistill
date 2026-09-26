"""Construct a backend by name, importing its heavy dependencies only when it is chosen."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import sys
from typing import TYPE_CHECKING, Any

from taskdistill.models import torch_equivalent

if TYPE_CHECKING:
    from taskdistill.backends.base import Backend

BACKENDS = ("mlx", "torch")
TORCH_INSTALL_HINT = (
    "install the torch extra: "
    "uvx --from 'taskdistill[torch] @ git+https://github.com/B0yko/taskdistill' taskdistill ..."
)


class BackendUnavailable(RuntimeError):
    """The requested backend cannot run on this machine or is not installed."""


DEVICE_ENV = "TASKDISTILL_MLX_DEVICE"


def configure_mlx() -> Any:
    """Import ``mlx.core`` and prepare the device (shared by the MLX backend and the MLX trainer).

    ``TASKDISTILL_MLX_DEVICE=cpu`` selects the CPU device, for machines without a usable Metal device.
    mlx-lm 0.31.3 reads ``mx.device_info()["max_recommended_working_set_size"]`` whenever Metal is
    present, which raises ``KeyError`` when the default device is the CPU; a shim answers that query
    with the GPU's info.
    """
    import mlx.core as mx

    if os.environ.get(DEVICE_ENV, "").strip().lower() == "cpu":
        mx.set_default_device(mx.cpu)
    _patch_device_info(mx)
    return mx


def _patch_device_info(mx: Any) -> None:
    if getattr(mx.device_info, "_taskdistill_shim", False):
        return
    try:
        if not mx.metal.is_available() or "max_recommended_working_set_size" in mx.device_info():
            return
    except Exception:
        return
    original = mx.device_info

    def device_info(d: Any = None) -> Any:
        return original(mx.gpu) if d is None else original(d)

    device_info._taskdistill_shim = True  # type: ignore[attr-defined]
    mx.device_info = device_info


def mlx_unavailable_reason() -> str | None:
    """None when MLX can run here, else a short reason."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return f"this machine is {sys.platform}/{platform.machine()}, not Apple Silicon macOS"
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except Exception as exc:  # ImportError, or a wheel that cannot load its Metal library
        return f"mlx / mlx-lm cannot be imported ({exc})"
    return None


def torch_unavailable_reason() -> str | None:
    missing = [name for name in ("torch", "transformers", "peft") if importlib.util.find_spec(name) is None]
    if missing:
        return f"missing package(s): {', '.join(missing)}"
    return None


def load_backend(name: str, base_model: str, adapter_path: str | None = None) -> Backend:
    """A backend for ``base_model`` (plus a LoRA adapter directory, if given).

    The model is not loaded yet: call ``.load()`` on the thread that will run it (backends also load
    lazily on first use). For ``torch``, an MLX 4-bit repo id is mapped to its full-precision equivalent.
    """
    if name == "mlx":
        reason = mlx_unavailable_reason()
        if reason is not None:
            raise BackendUnavailable(
                f"the mlx backend needs Apple Silicon (arm64 macOS 14+): {reason}. "
                f"Use --backend torch ({TORCH_INSTALL_HINT})."
            )
        from taskdistill.backends.mlx_backend import MLXBackend

        return MLXBackend(base_model, adapter_path)
    if name == "torch":
        reason = torch_unavailable_reason()
        if reason is not None:
            raise BackendUnavailable(f"the torch backend is not installed ({reason}); {TORCH_INSTALL_HINT}")
        module = importlib.import_module("taskdistill.backends.torch_backend")
        backend: Backend = module.TorchBackend(torch_equivalent(base_model), adapter_path)
        return backend
    raise ValueError(f"unknown backend {name!r}; expected one of: {', '.join(BACKENDS)}")
