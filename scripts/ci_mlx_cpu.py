"""CI probe: can MLX run a tiny Qwen2-architecture model on the CPU device of a hosted macOS runner?

Metal is not a reliable target on GitHub-hosted runners, so CI only tries the CPU device. The result is
informative (the job step is allowed to fail) and is recorded in the README.
"""

from __future__ import annotations

import time

import mlx.core as mx

mx.set_default_device(mx.cpu)  # must happen before mlx_lm is imported

from mlx_lm.models import qwen2  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=512,
        tie_word_embeddings=True,
    )
    mx.random.seed(0)
    model = qwen2.Model(args)
    mx.eval(model.parameters())
    cache = make_prompt_cache(model)
    logits = model(mx.array([[1, 2, 3, 4, 5]]), cache=cache)[:, -1, :]
    tokens = []
    for _ in range(8):
        token = int(mx.argmax(logits, axis=-1).item())
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)[:, -1, :]
    mx.eval(logits)
    print(f"mlx {mx.__version__} on {mx.default_device()}: decoded {tokens} in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
