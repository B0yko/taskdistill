"""Pricing snapshot completion limits, the ablation script workspace and the README's serve escalation settings."""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

from taskdistill.teacher import client as client_mod
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
MODELS_JSON = {
    "data": [
        {
            "id": "vendor/model-a",
            "canonical_slug": "vendor/model-a-20260101",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "top_provider": {"context_length": 131072, "max_completion_tokens": 65536},
            "context_length": 131072,
        },
        {
            "id": "vendor/model-b",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "context_length": 8192,  # no top_provider: falls back to context_length
        },
        {
            "id": "vendor/model-c",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            # neither top_provider nor context_length: no known limit
        },
    ]
}
ENDPOINTS_A = {
    "data": {
        "endpoints": [
            {
                "tag": "alpha/fp8",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "max_completion_tokens": 20_000,
            },
            {"tag": "beta", "pricing": {"prompt": "0.000002", "completion": "0.000003"}, "context_length": 4_096},
            {"tag": "gamma", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},  # no limit of its own
        ]
    }
}


def _load_script(name: str) -> types.ModuleType:
    """Import a ``scripts/*.py`` file as a module, without adding ``scripts/`` to ``sys.path``."""
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- ablation_lr_schedule.py never mixes with an inherited workspace ------------------------


def test_ablation_home_ignores_an_inherited_taskdistill_home(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("ablation_lr_schedule.py")
    monkeypatch.delenv("ABLATION_HOME", raising=False)
    monkeypatch.setenv("TASKDISTILL_HOME", "/some/other/workspace")
    assert module.ablation_home() == str(module.DEFAULT_ABLATION_HOME)

    monkeypatch.setenv("ABLATION_HOME", "/tmp/custom-ablation-home")
    assert module.ablation_home() == "/tmp/custom-ablation-home"


# -- PricingSnapshot.max_completion_tokens and the limits it is built and persisted from ----


def test_from_openrouter_parses_the_models_list_and_per_endpoint_limits() -> None:
    snapshot = PricingSnapshot.from_openrouter(MODELS_JSON, {"vendor/model-a": ENDPOINTS_A}, "2026-09-26")
    # No provider pinned: the largest known limit for the model (here, its top_provider.max_completion_tokens,
    # larger than either endpoint's own limit -- routing could pick any of them).
    assert snapshot.max_completion_tokens("vendor/model-a") == 65_536
    assert snapshot.max_completion_tokens("vendor/model-a", "alpha/fp8") == 20_000  # a pinned endpoint's own limit
    assert snapshot.max_completion_tokens("vendor/model-a", "beta") == 4_096
    assert snapshot.max_completion_tokens("vendor/model-a", "Alpha/FP8") == 20_000  # case-insensitive
    # gamma is not a known endpoint tag: the largest known limit for the model wins (here, the model-level
    # default from top_provider.max_completion_tokens, larger than either of the other two endpoints' own).
    assert snapshot.max_completion_tokens("vendor/model-a", "gamma") == 65_536
    # model-a's dated canonical_slug resolves like price_for.
    assert snapshot.max_completion_tokens("vendor/model-a-20260101", "alpha/fp8") == 20_000
    # model-b has no top_provider: context_length is used as its limit.
    assert snapshot.max_completion_tokens("vendor/model-b") == 8_192
    # model-c has neither: no worst case can be bounded for it.
    assert snapshot.max_completion_tokens("vendor/model-c") is None
    assert snapshot.max_completion_tokens("vendor/unknown") is None  # unlike price_for, not an error


def test_pricing_snapshot_persists_limits_and_loads_older_snapshots_without_them() -> None:
    snapshot = PricingSnapshot.from_openrouter(MODELS_JSON, {"vendor/model-a": ENDPOINTS_A}, "2026-09-26")
    data = snapshot.to_json()
    assert data["limits"]["vendor/model-a"] == {
        "default": 65_536,
        "providers": {"alpha/fp8": 20_000, "beta": 4_096},
    }
    assert data["limits"]["vendor/model-b"] == {"default": 8_192, "providers": {}}
    assert "vendor/model-c" not in data["limits"]  # never a bare {"default": null, "providers": {}} entry

    reloaded = PricingSnapshot.from_json(json.loads(json.dumps(data)))
    assert reloaded == snapshot
    assert reloaded.max_completion_tokens("vendor/model-a", "beta") == 4_096

    del data["limits"]  # a snapshot saved before this feature existed
    older = PricingSnapshot.from_json(data)
    assert older.completion_limits == {}
    assert older.max_completion_tokens("vendor/model-a") is None
    assert older.price_for("vendor/model-a") == snapshot.price_for("vendor/model-a")  # pricing itself is unaffected


def test_the_live_client_asks_a_real_pricing_snapshot_for_the_model_maximum() -> None:
    """``client.model_max_completion_tokens`` against the real :class:`PricingSnapshot`, not a test stub."""
    snapshot = PricingSnapshot.from_openrouter(MODELS_JSON, {"vendor/model-a": ENDPOINTS_A}, "2026-09-26")
    assert client_mod.model_max_completion_tokens(snapshot, "vendor/model-a", "alpha/fp8") == 20_000
    assert client_mod.model_max_completion_tokens(snapshot, "vendor/model-a", None) == 65_536
    assert client_mod.model_max_completion_tokens(snapshot, "vendor/model-c") is None
    assert client_mod.model_max_completion_tokens(snapshot, "vendor/unknown") is None

    # a snapshot with no max_completion_tokens lookup at all (e.g. a Protocol-only stand-in) is handled the same
    class NoLimits:
        def price_for(self, model: str, provider: str | None = None) -> ModelPrice:
            return ModelPrice(1e-6, 2e-6)

    assert client_mod.model_max_completion_tokens(NoLimits(), "vendor/model-a") is None  # type: ignore[arg-type]


def test_the_packaged_pricing_snapshot_carries_limits_for_its_bundled_models() -> None:
    from taskdistill.teacher.factory import load_pricing

    snapshot = load_pricing()
    assert snapshot.date == "2026-09-26"
    known = [slug for slug in snapshot.models if snapshot.max_completion_tokens(slug) is not None]
    assert known  # at least one bundled model carries a usable completion limit
    for slug in known:
        assert isinstance(snapshot.max_completion_tokens(slug), int)


# -- the README documents serve's actual escalation deadline and retry policy --------------


def test_readme_states_serves_real_escalation_deadline_and_retry_policy() -> None:
    from taskdistill.serve.runner import SERVE_TEACHER_DEADLINE_S
    from taskdistill.teacher.factory import SERVE_MAX_RETRIES, SERVE_RETRY_AFTER_MAX_S, SERVE_TIMEOUT_S

    readme = Path(__file__).resolve().parent.parent / "README.md"
    lines = readme.read_text(encoding="utf-8").splitlines()
    row = next(ln for ln in lines if ln.startswith("| `cascade.on_teacher_error`"))
    assert f"{SERVE_TIMEOUT_S:g} s" in row
    assert f"{SERVE_MAX_RETRIES}" in row
    assert f"{SERVE_RETRY_AFTER_MAX_S:g} s" in row
    assert f"{SERVE_TEACHER_DEADLINE_S:g} s" in row
