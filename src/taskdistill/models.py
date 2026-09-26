"""Base models: pinned revisions, local-first snapshot resolution and tokenizer loading.

The student base models are pinned to the Hugging Face commits verified for this release, so a
re-run trains on exactly the same weights. A snapshot already in the local Hugging Face cache is used
without any network call; only a missing snapshot is downloaded.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

KNOWN_REVISIONS: dict[str, str] = {
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit": "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit": "8b403126fc14f14cfc99bb4cfa72ecbc129ea677",
}

TORCH_EQUIVALENTS: dict[str, str] = {
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit": "Qwen/Qwen2.5-0.5B-Instruct",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit": "Qwen/Qwen2.5-1.5B-Instruct",
}

# Weights, configs and tokenizer files; no repository code (*.py) is ever fetched.
MODEL_PATTERNS: tuple[str, ...] = (
    "*.json",
    "model*.safetensors",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
)

TOKENIZER_PATTERNS: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "*.tiktoken",
    "chat_template.jinja",
    "chat_template.json",
)

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json")


class ModelNotAvailable(RuntimeError):
    """A base model is neither a local directory, nor cached, nor downloadable."""


def pinned_revision(repo: str) -> str | None:
    return KNOWN_REVISIONS.get(repo)


def torch_equivalent(repo: str) -> str:
    """The full-precision Hugging Face repo for an MLX 4-bit repo (unchanged if unknown)."""
    return TORCH_EQUIVALENTS.get(repo, repo)


def _wanted(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _is_complete(path: Path, patterns: Sequence[str]) -> bool:
    """A cached snapshot folder can exist with only some files; check the ones the patterns ask for."""
    if not path.is_dir():
        return False
    if _wanted("config.json", patterns) and not (path / "config.json").is_file():
        return False
    if _wanted("model.safetensors", patterns) and not any(path.glob("*.safetensors")):
        return False
    wants_tokenizer = any(_wanted(name, patterns) for name in _TOKENIZER_FILES)
    return not wants_tokenizer or any((path / name).is_file() for name in _TOKENIZER_FILES)


def _cached_snapshot(repo: str, revision: str | None, patterns: Sequence[str]) -> Path | None:
    """The complete snapshot in the local Hugging Face cache, or None (never touches the network)."""
    from huggingface_hub import snapshot_download

    try:
        local = Path(snapshot_download(repo, revision=revision, allow_patterns=list(patterns), local_files_only=True))
    except Exception:  # not cached (LocalEntryNotFoundError and friends)
        return None
    return local if _is_complete(local, patterns) else None


def _snapshot(repo: str, revision: str | None, patterns: Sequence[str]) -> Path:
    from huggingface_hub import snapshot_download

    local = _cached_snapshot(repo, revision, patterns)
    if local is not None:
        return local
    try:
        fetched = Path(snapshot_download(repo, revision=revision, allow_patterns=list(patterns)))
    except Exception as exc:
        at = f"@{revision}" if revision else ""
        raise ModelNotAvailable(
            f"base model {repo}{at} is not in the local Hugging Face cache and could not be downloaded: {exc}"
        ) from exc
    if not _is_complete(fetched, patterns):
        raise ModelNotAvailable(f"base model {repo}: the downloaded snapshot is missing required files")
    return fetched


def resolve_model_path(repo: str, revision: str | None = None, allow_patterns: Sequence[str] | None = None) -> Path:
    """Local directory for ``repo``.

    An existing path is returned as is. Otherwise the Hugging Face snapshot at ``revision`` (default: the
    pinned revision, else the default branch) is taken from the local cache, or downloaded if missing.
    """
    local = Path(repo).expanduser()
    if local.exists():
        return local
    patterns = tuple(allow_patterns) if allow_patterns is not None else MODEL_PATTERNS
    return _snapshot(repo, revision or KNOWN_REVISIONS.get(repo), patterns)


def is_available_locally(repo: str, revision: str | None = None, allow_patterns: Sequence[str] | None = None) -> bool:
    """True when :func:`resolve_model_path` would need no download (a local path, or a complete cached snapshot)."""
    if Path(repo).expanduser().exists():
        return True
    patterns = tuple(allow_patterns) if allow_patterns is not None else MODEL_PATTERNS
    return _cached_snapshot(repo, revision or KNOWN_REVISIONS.get(repo), patterns) is not None


def tokenizer_path(repo: str, revision: str | None = None) -> Path:
    """Local directory holding ``repo``'s tokenizer files and config.json (no weights are fetched)."""
    return resolve_model_path(repo, revision, allow_patterns=TOKENIZER_PATTERNS)


def load_tokenizer(repo: str, revision: str | None = None) -> Any:
    """A ``transformers`` tokenizer for ``repo`` (weights are never loaded on this path)."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_path(repo, revision)))
