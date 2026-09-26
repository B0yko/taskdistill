from __future__ import annotations

from pathlib import Path
from typing import Any

import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from taskdistill import models
from taskdistill.models import (
    KNOWN_REVISIONS,
    MODEL_PATTERNS,
    TOKENIZER_PATTERNS,
    TORCH_EQUIVALENTS,
    ModelNotAvailable,
    resolve_model_path,
    tokenizer_path,
    torch_equivalent,
)

REPO_05 = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
REPO_15 = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


class FakeHub:
    """Stands in for ``huggingface_hub.snapshot_download``: a local cache plus an optional network."""

    def __init__(self, root: Path, *, cached: set[str] | None = None, network: bool = True) -> None:
        self.root = root
        self.cached = set(cached or ())
        self.network = network
        self.calls: list[dict[str, Any]] = []

    def snapshot(self, files: set[str]) -> Path:
        folder = self.root / "snapshot"
        folder.mkdir(parents=True, exist_ok=True)
        for name in files:
            (folder / name).write_text("{}", encoding="utf-8")
        return folder

    def __call__(self, repo_id: str, **kwargs: Any) -> str:
        self.calls.append({"repo_id": repo_id, **kwargs})
        if kwargs.get("local_files_only"):
            if not self.cached:
                raise LocalEntryNotFoundError("not cached")
            return str(self.snapshot(self.cached))
        if not self.network:
            raise OSError("network is unreachable")
        return str(self.snapshot({"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}))


def install(monkeypatch: pytest.MonkeyPatch, hub: FakeHub) -> FakeHub:
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    return hub


def test_local_directory_is_returned_as_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = install(monkeypatch, FakeHub(tmp_path / "hub", network=False))
    local = tmp_path / "my-model"
    local.mkdir()
    assert resolve_model_path(str(local)) == local
    assert tokenizer_path(str(local)) == local
    assert hub.calls == []


def test_cached_snapshot_uses_pinned_revision_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = install(
        monkeypatch,
        FakeHub(tmp_path, cached={"config.json", "model.safetensors", "tokenizer.json"}, network=False),
    )
    path = resolve_model_path(REPO_05)
    assert path == tmp_path / "snapshot"
    assert len(hub.calls) == 1
    call = hub.calls[0]
    assert call["repo_id"] == REPO_05
    assert call["revision"] == KNOWN_REVISIONS[REPO_05]
    assert call["local_files_only"] is True
    assert call["allow_patterns"] == list(MODEL_PATTERNS)
    assert not any(p.endswith(".py") for p in call["allow_patterns"])


def test_explicit_revision_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = install(monkeypatch, FakeHub(tmp_path, cached={"config.json", "model.safetensors", "tokenizer.json"}))
    resolve_model_path(REPO_15, revision="abc123")
    assert hub.calls[0]["revision"] == "abc123"


def test_unknown_repo_resolves_the_default_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = install(monkeypatch, FakeHub(tmp_path))
    resolve_model_path("example-org/other-model")
    assert [c["revision"] for c in hub.calls] == [None, None]
    assert [c.get("local_files_only", False) for c in hub.calls] == [True, False]


def test_missing_snapshot_is_downloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = install(monkeypatch, FakeHub(tmp_path))
    path = resolve_model_path(REPO_15)
    assert (path / "model.safetensors").is_file()
    assert [c.get("local_files_only", False) for c in hub.calls] == [True, False]
    assert hub.calls[1]["revision"] == KNOWN_REVISIONS[REPO_15]


def test_partial_snapshot_triggers_a_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A snapshot folder holding only tokenizer files must not pass for the full model."""
    hub = install(monkeypatch, FakeHub(tmp_path, cached={"config.json", "tokenizer.json"}))
    resolve_model_path(REPO_05)
    assert len(hub.calls) == 2 and hub.calls[1].get("local_files_only") is None


def test_tokenizer_path_fetches_no_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = install(monkeypatch, FakeHub(tmp_path, cached={"config.json", "tokenizer.json"}, network=False))
    assert tokenizer_path(REPO_05) == tmp_path / "snapshot"
    patterns = hub.calls[0]["allow_patterns"]
    assert patterns == list(TOKENIZER_PATTERNS)
    assert "config.json" in patterns
    assert not any("safetensors" in p for p in patterns)


def test_unavailable_model_has_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeHub(tmp_path, network=False))
    with pytest.raises(ModelNotAvailable, match=r"Qwen2\.5-1\.5B-Instruct-4bit@8b403126.*not in the local"):
        resolve_model_path(REPO_15)


def test_is_available_locally_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = tmp_path / "my-model"
    local.mkdir()
    hub = install(monkeypatch, FakeHub(tmp_path / "hub"))
    assert models.is_available_locally(str(local))
    assert not models.is_available_locally(REPO_15)
    assert [(c["revision"], c.get("local_files_only")) for c in hub.calls] == [(KNOWN_REVISIONS[REPO_15], True)]

    install(monkeypatch, FakeHub(tmp_path / "partial", cached={"config.json", "tokenizer.json"}))
    assert not models.is_available_locally(REPO_05)  # tokenizer files only: the weights would be downloaded
    install(monkeypatch, FakeHub(tmp_path / "full", cached={"config.json", "model.safetensors", "tokenizer.json"}))
    assert models.is_available_locally(REPO_05)


def test_torch_equivalents() -> None:
    assert TORCH_EQUIVALENTS == {
        REPO_05: "Qwen/Qwen2.5-0.5B-Instruct",
        REPO_15: "Qwen/Qwen2.5-1.5B-Instruct",
    }
    assert torch_equivalent(REPO_15) == "Qwen/Qwen2.5-1.5B-Instruct"
    assert torch_equivalent("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert set(KNOWN_REVISIONS) == set(TORCH_EQUIVALENTS)
    assert all(len(sha) == 40 for sha in KNOWN_REVISIONS.values())


def _cached_05() -> Path | None:
    try:
        path = Path(
            huggingface_hub.snapshot_download(
                REPO_05,
                revision=KNOWN_REVISIONS[REPO_05],
                allow_patterns=list(TOKENIZER_PATTERNS),
                local_files_only=True,
            )
        )
    except Exception:
        return None
    return path if (path / "tokenizer_config.json").is_file() else None


@pytest.mark.skipif(_cached_05() is None, reason="the 0.5B student base is not in the local Hugging Face cache")
def test_load_tokenizer_from_the_cache() -> None:
    tokenizer = models.load_tokenizer(REPO_05)
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == 151645
    messages = [{"role": "system", "content": "Classify."}, {"role": "user", "content": "My card is lost"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    assert text.endswith("<|im_start|>assistant\n")
