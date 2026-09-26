from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from taskdistill.config import TaskSpec
from taskdistill.ledger import Ledger
from taskdistill.teacher import factory
from taskdistill.teacher.base import ManifestMismatch
from taskdistill.teacher.cache import ResponseCache
from taskdistill.teacher.client import LiveTeacher
from taskdistill.teacher.factory import (
    RECORDING_NAME,
    TeacherUnavailable,
    find_recording,
    load_pricing,
    make_teacher,
    new_run_id,
    packaged_pricing,
    resolve_mode,
    workspace_recording,
)
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.replay import ReplayTeacher, expected_manifest, write_recording
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

MODEL = "vendor/model-a"
TASK = "demo-intents"
QUERY = "My card still has not arrived after two weeks."
KEY_ENVS = ("TASKDISTILL_TEACHER_API_KEY", "OPENROUTER_API_KEY")


def make_spec(**teacher: Any) -> TaskSpec:
    teacher_cfg: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 24,
        "extra_body": {"provider": {"order": ["alpha/fp8"], "allow_fallbacks": False}},
    }
    teacher_cfg.update(teacher)
    spec = TaskSpec.model_validate(
        {
            "task": TASK,
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": teacher_cfg,
            "student": {"system_prompt": "Classify the message."},
            "cascade": {"target": 0.97},
            "budget": {"usd_cap": 2.0},
        }
    )
    spec.teacher_prompt = "Label the customer's message with one intent.\n"
    spec.labels = ["card_arrival", "lost_card"]
    spec.source = f"tasks/{TASK}/task.yaml"
    return spec


def write_demo_recording(path: Path, spec: TaskSpec) -> dict[str, Any]:
    body = build_teacher_request(spec, QUERY)
    record = {
        "key": request_key(body),
        "output": "card_arrival",
        "usage": {"prompt_tokens": 120, "completion_tokens": 3, "cost": 1.2e-5},
        "latency_ms": 410.5,
        "provider": "Alpha",
        "finish_reason": "stop",
        "timestamp": 1790000000.25,
    }
    write_recording(path, expected_manifest(spec, "2026-09-26"), [record])
    return body


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    monkeypatch.setenv("TASKDISTILL_BUDGET_USD", "5")
    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    return workspace


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in for the package data directory."""
    root = tmp_path / "package"
    root.mkdir()
    monkeypatch.setattr(factory, "data_root", lambda: root)
    return root


# find_recording -------------------------------------------------------------------------------
def test_find_recording_prefers_the_workspace_then_the_package(home: Path, package: Path) -> None:
    spec = make_spec()
    assert find_recording(TASK) is None

    packaged = package / "demos" / TASK / RECORDING_NAME
    write_demo_recording(packaged, spec)
    assert find_recording(TASK) == packaged

    local = home / TASK / RECORDING_NAME
    write_demo_recording(local, spec)
    assert workspace_recording(TASK) == local
    assert find_recording(TASK) == local
    assert find_recording("other-task") is None


# resolve_mode ---------------------------------------------------------------------------------
def test_resolve_mode_without_a_request_prefers_live_then_replay(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = make_spec()
    with pytest.raises(TeacherUnavailable) as err:
        resolve_mode(spec, None)
    message = str(err.value)
    assert "no teacher API key" in message
    assert "TASKDISTILL_TEACHER_API_KEY or OPENROUTER_API_KEY" in message
    assert "no recording" in message

    write_demo_recording(home / TASK / RECORDING_NAME, spec)
    assert resolve_mode(spec, None) == "replay"

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert resolve_mode(spec, None) == "live"


def test_an_explicit_mode_wins_but_must_be_usable(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = make_spec()
    with pytest.raises(TeacherUnavailable, match="TASKDISTILL_TEACHER_API_KEY or OPENROUTER_API_KEY"):
        resolve_mode(spec, "live")
    with pytest.raises(TeacherUnavailable, match="no teacher recording for task 'demo-intents'"):
        resolve_mode(spec, "replay")

    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", "sk-test")
    assert resolve_mode(spec, "live") == "live"
    with pytest.raises(TeacherUnavailable):
        resolve_mode(spec, "replay")

    write_demo_recording(home / TASK / RECORDING_NAME, spec)
    assert resolve_mode(spec, "replay") == "replay"
    assert resolve_mode(spec, None) == "live"
    with pytest.raises(ValueError, match="live, replay"):
        resolve_mode(spec, "offline")


def test_the_key_hint_names_the_spec_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = make_spec(api_key_env="MY_TEACHER_KEY")
    with pytest.raises(TeacherUnavailable, match="MY_TEACHER_KEY or OPENROUTER_API_KEY"):
        resolve_mode(spec, "live")
    monkeypatch.setenv("MY_TEACHER_KEY", "sk-test")
    assert resolve_mode(spec, "live") == "live"


# make_teacher ---------------------------------------------------------------------------------
def test_make_teacher_replays_the_recording(home: Path) -> None:
    spec = make_spec()
    body = write_demo_recording(home / TASK / RECORDING_NAME, spec)
    teacher = make_teacher(spec, mode="replay", phase="serve", run_id="serve-1")
    assert isinstance(teacher, ReplayTeacher)
    assert teacher.mode == "replay"
    result = asyncio.run(teacher.complete(body))
    assert result.output == "card_arrival"
    assert result.source == "replay"


def test_make_teacher_refuses_a_recording_of_another_spec(home: Path) -> None:
    write_demo_recording(home / TASK / RECORDING_NAME, make_spec())
    other = make_spec(extra_body={"provider": {"order": ["beta"], "allow_fallbacks": False}})
    with pytest.raises(ManifestMismatch, match=r"provider is 'alpha/fp8' in the recording but 'beta'"):
        make_teacher(other, mode="replay", phase="serve", run_id="serve-1")


def test_make_teacher_replay_without_a_recording_is_a_clear_error() -> None:
    with pytest.raises(TeacherUnavailable, match="no teacher recording"):
        make_teacher(make_spec(), mode="replay", phase="serve", run_id="serve-1")
    with pytest.raises(ValueError, match="live, replay"):
        make_teacher(make_spec(), mode="offline", phase="serve", run_id="serve-1")


def test_make_teacher_live_wires_the_ledger_cache_caps_and_pricing(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", "sk-test")
    spec = make_spec(base_url="https://teacher.example.com/api/v1/")
    teacher = make_teacher(spec, mode="live", phase="label", run_id="curate-1", run_cap=0.75, concurrency=3)
    assert isinstance(teacher, LiveTeacher)
    assert teacher.mode == "live"
    assert teacher.base_url == "https://teacher.example.com/api/v1"
    assert (teacher.task, teacher.phase, teacher.run_id) == (TASK, "label", "curate-1")
    assert teacher.run_cap == 0.75
    assert teacher.task_cap == 2.0
    assert teacher.concurrency == 3
    assert isinstance(teacher.ledger, Ledger)
    assert teacher.ledger.path == home.resolve() / "ledger.sqlite"
    assert isinstance(teacher.cache, ResponseCache)
    assert teacher.cache.path == home.resolve() / "cache.sqlite"
    assert teacher.pricing.date == "2026-09-26"  # the packaged snapshot: the workspace has none

    uncached = make_teacher(spec, mode="live", phase="serve", run_id="serve-1", use_cache=False)
    assert isinstance(uncached, LiveTeacher)
    assert uncached.cache is None

    snapshot = PricingSnapshot("2030-01-01", "test", {MODEL: {"default": ModelPrice(1e-6, 2e-6), "providers": {}}})
    own_cache = ResponseCache(home / "other_cache.sqlite")
    given = make_teacher(spec, mode="live", phase="bakeoff", run_id="bakeoff-1", pricing=snapshot, cache=own_cache)
    assert isinstance(given, LiveTeacher)
    assert given.pricing is snapshot
    assert given.cache is own_cache
    no_cache = make_teacher(spec, mode="live", phase="serve", run_id="serve-2", use_cache=False, cache=own_cache)
    assert isinstance(no_cache, LiveTeacher) and no_cache.cache is None


def test_make_teacher_live_without_an_api_key_is_a_clear_error() -> None:
    with pytest.raises(TeacherUnavailable, match="live teacher calls need an API key: set TASKDISTILL_TEACHER_API_KEY"):
        make_teacher(make_spec(), mode="live", phase="label", run_id="curate-1")


# pricing --------------------------------------------------------------------------------------
def test_load_pricing_prefers_the_workspace_snapshot(home: Path) -> None:
    packaged = load_pricing()
    assert packaged.date == "2026-09-26"
    local = PricingSnapshot("2030-01-01", "test", {MODEL: {"default": ModelPrice(1e-6, 2e-6), "providers": {}}})
    local.save()
    assert load_pricing().date == "2030-01-01"
    assert load_pricing(home / "pricing.json").models.keys() == {MODEL}


def test_the_packaged_pricing_snapshot_loads_and_prices_the_candidates(tmp_path: Path) -> None:
    source = packaged_pricing()
    raw = source.read_bytes()
    assert len(raw) < 50_000
    copy = tmp_path / "pricing.json"
    with source.open("rb") as src, copy.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    snapshot = PricingSnapshot.load(copy)
    assert snapshot.date == "2026-09-26"
    assert snapshot.source == "https://openrouter.ai/api/v1/models"
    assert PricingSnapshot.from_json(snapshot.to_json()) == snapshot
    assert not any(slug.endswith((":free", ":batch")) for slug in snapshot.models)
    assert all(slug.startswith(("deepseek/", "qwen/")) for slug in snapshot.models)
    assert {"deepseek/deepseek-v4-flash-0731", "qwen/qwen3-235b-a22b-2507"} <= snapshot.models.keys()

    flash = snapshot.price_for("deepseek/deepseek-v4-flash-0731", "deepinfra/fp8")
    assert flash.prompt == pytest.approx(0.06e-6)
    assert flash.completion == pytest.approx(0.18e-6)
    by_canonical = snapshot.price_for("deepseek/deepseek-v4-flash-20260731", "deepinfra/fp8")
    assert by_canonical == flash
    assert snapshot.aliases["deepseek/deepseek-v4.1-flash-20260910"] == "deepseek/deepseek-v4.1-flash"

    data = json.loads(raw)
    assert set(data["capabilities"]) == set(snapshot.models)


def test_new_run_id_is_the_command_a_utc_timestamp_and_a_random_suffix() -> None:
    run_id = new_run_id("bakeoff")
    assert re.fullmatch(r"bakeoff-\d{8}T\d{6}-[0-9a-f]{6}", run_id)
