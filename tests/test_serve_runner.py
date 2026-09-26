from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import uvicorn
from fastapi.testclient import TestClient

from taskdistill import paths
from taskdistill.backends.fake import FakeBackend
from taskdistill.config import TaskSpec
from taskdistill.serve import runner
from taskdistill.serve.runner import (
    ServeSetupError,
    UnsafeBindError,
    build_server,
    resolve_run,
    resolve_threshold,
    run_server,
)
from taskdistill.serve.worker import ModelWorker
from taskdistill.store import Store
from taskdistill.teacher.client import LiveTeacher
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.replay import ReplayTeacher, expected_manifest, write_recording
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

TASK = "support-intents"
TEACHER = "vendor/teacher-model"
TEACHER_BASE = "https://teacher.example.com/api/v1"
TEACHER_KEY = "sk-test-teacher-key"
KEY_ENV = "SUPPORT_TEACHER_KEY"
BASE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
RUN = "qwen2.5-0.5b-full-s13-teacher-mlx"
SURE = "My card still has not arrived after two weeks."
UNSURE = "I think I lost it somewhere yesterday."
ANSWERS = {SURE: ("card_arrival", 0.95), UNSURE: ("lost_card", 0.40)}
REPLAY_LINE = "teacher: replay (recorded outputs; escalations are not live calls)"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    monkeypatch.setenv("TASKDISTILL_BUDGET_USD", "5")
    for name in ("TASKDISTILL_TEACHER_API_KEY", "OPENROUTER_API_KEY", "TASKDISTILL_SERVER_TOKEN", KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    return workspace


@pytest.fixture
def spec(home: Path) -> TaskSpec:
    spec = TaskSpec.model_validate(
        {
            "task": TASK,
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": {"model": TEACHER, "base_url": TEACHER_BASE, "api_key_env": KEY_ENV, "max_tokens": 24},
            "student": {"system_prompt": "Classify the message."},
            "cascade": {"target": 0.97},
        }
    )
    spec.teacher_prompt = "Label the customer's message with one intent.\n"
    spec.labels = ["card_arrival", "lost_card", "cash_withdrawal"]
    return spec


class Factory:
    """Stands in for ``load_backend``: records its arguments and returns a FakeBackend."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(self, name: str, base_model: str, adapter_path: str | None) -> FakeBackend:
        self.calls.append((name, base_model, adapter_path))
        return FakeBackend(answers=ANSWERS, base_model=base_model, adapter_path=adapter_path)


def make_run(run_id: str = RUN, *, backend: str = "mlx", base: str = BASE_MODEL) -> Path:
    run_dir = paths.runs_dir(TASK) / run_id
    (run_dir / "adapter").mkdir(parents=True)
    log = {"run_id": run_id, "task": TASK, "backend": backend, "base_model": base}
    (run_dir / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    return run_dir


def select_run(run_id: str = RUN) -> None:
    data = {"run_id": run_id, "reason": "best validation agreement", "date": "2026-09-26T12:00:00+00:00"}
    (paths.task_home(TASK) / "selected_run.json").write_text(json.dumps(data), encoding="utf-8")


def write_threshold(value: float | None, run_id: str = RUN) -> None:
    data = {"threshold": value, "always_escalate": value is None, "escalation_rate": 0.2, "run_id": run_id}
    (paths.task_home(TASK) / "threshold.json").write_text(json.dumps(data), encoding="utf-8")


def record(spec: TaskSpec, outputs: dict[str, str]) -> Path:
    records = [
        {
            "key": request_key(build_teacher_request(spec, text)),
            "output": output,
            "usage": {"prompt_tokens": 80, "completion_tokens": 2, "cost": 1e-5},
            "latency_ms": 350.0,
            "provider": None,
            "finish_reason": "stop",
            "timestamp": 1790000000.0,
        }
        for text, output in outputs.items()
    ]
    manifest = {**expected_manifest(spec, "2026-09-26"), "created": "2026-09-26T12:00:00+00:00"}
    return write_recording(paths.task_home(TASK) / "teacher_recording.jsonl.gz", manifest, records)


@pytest.fixture
def ready(spec: TaskSpec) -> Path:
    """A selected run, a threshold chosen for it, and a teacher recording."""
    run_dir = make_run()
    select_run()
    write_threshold(0.8)
    record(spec, {UNSURE: "Lost card"})
    return run_dir


def app_request(spec: TaskSpec, text: str) -> dict[str, Any]:
    return {**build_teacher_request(spec, text), "model": "gpt-4o-mini"}


def pricing_snapshot() -> None:
    PricingSnapshot(
        date="2026-09-26", source="test", models={TEACHER: {"default": ModelPrice(1e-6, 2e-6), "providers": {}}}
    ).save()


# replay and the banner ------------------------------------------------------------------------------


def test_replay_server_from_the_selected_run(spec: TaskSpec, ready: Path, tmp_path: Path) -> None:
    factory = Factory()
    store = Store(tmp_path / "store.sqlite")
    app, banner = build_server(spec, replay=True, backend_factory=factory, store=store)
    assert factory.calls == [("mlx", BASE_MODEL, str(ready / "adapter"))]
    assert isinstance(app.state.cascade.teacher, ReplayTeacher)
    assert banner[0] == f"task: {TASK} (classification)"
    assert banner[1].startswith(f"run: {RUN}, from selected_run.json (base {BASE_MODEL}")
    assert banner[2] == f"threshold: 0.8 (auto: threshold.json, chosen on validation for run {RUN})"
    assert "backend: mlx" in banner
    assert REPLAY_LINE in banner
    assert "listening on http://127.0.0.1:8000/v1 (OpenAI-compatible; any model name is accepted)" in banner
    assert not any(line.startswith(("warning:", "auth:")) for line in banner)

    with TestClient(app) as client:
        health = client.get("/healthz").json()
        student = client.post("/v1/chat/completions", json=app_request(spec, SURE))
        escalated = client.post("/v1/chat/completions", json=app_request(spec, UNSURE))
    assert health["teacher"] == "replay"
    assert (health["run"], health["threshold"], health["backend"]) == (RUN, 0.8, "fake")
    assert student.headers["x-taskdistill-route"] == "student"
    assert escalated.headers["x-taskdistill-route"] == "teacher"
    assert escalated.headers["x-taskdistill-teacher"] == "replay"
    assert escalated.json()["choices"][0]["message"]["content"] == "lost_card"
    rows = list(store.iter_served(TASK))
    assert [(r.route, r.teacher_mode) for r in rows] == [("student", None), ("teacher", "replay")]
    assert rows[1].teacher_cost_usd == 0.0
    assert (rows[1].prompt_tokens, rows[1].completion_tokens) == (80, 2)


def test_without_an_api_key_the_recording_is_used(spec: TaskSpec, ready: Path, tmp_path: Path) -> None:
    app, banner = build_server(spec, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    assert REPLAY_LINE in banner
    assert app.state.cascade.teacher.mode == "replay"


def test_replay_is_forced_even_with_an_api_key(
    spec: TaskSpec, ready: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, TEACHER_KEY)
    pricing_snapshot()
    _, live = build_server(spec, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    _, replayed = build_server(spec, replay=True, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    assert f"teacher: live ({TEACHER} via {TEACHER_BASE})" in live
    assert REPLAY_LINE in replayed


# the live teacher -----------------------------------------------------------------------------------


def test_live_teacher_uses_the_key_from_api_key_env_and_never_the_cache(
    spec: TaskSpec, ready: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, TEACHER_KEY)
    pricing_snapshot()
    app, banner = build_server(spec, max_usd=0.5, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    teacher = app.state.cascade.teacher
    assert isinstance(teacher, LiveTeacher)
    assert teacher.cache is None
    assert (teacher.phase, teacher.run_cap) == ("serve", 0.5)
    assert teacher.run_id.startswith("serve-")
    assert f"teacher: live ({TEACHER} via {TEACHER_BASE})" in banner
    assert "spend cap for this server run: $0.50 (--max-usd)" in banner

    upstream = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1790000000,
        "model": TEACHER,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "lost_card"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 3, "cost": 4e-5},
    }
    with respx.mock(base_url=TEACHER_BASE, assert_all_called=True) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=upstream))
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=app_request(spec, UNSURE),
                headers={"Authorization": "Bearer sk-test-client-key"},
            )
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-teacher"] == "live"
    assert route.calls[0].request.headers["authorization"] == f"Bearer {TEACHER_KEY}"


def test_a_ledger_cap_refuses_the_call_and_the_student_answers(
    spec: TaskSpec, ready: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, TEACHER_KEY)
    pricing_snapshot()
    app, _ = build_server(spec, max_usd=1e-6, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    with respx.mock(base_url=TEACHER_BASE, assert_all_called=False) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(500))
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=app_request(spec, UNSURE))
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "student-fallback"
    assert resp.headers["x-taskdistill-teacher-error"] == "budget"
    assert not route.called


def test_a_live_teacher_without_a_price_refuses_to_start(
    spec: TaskSpec, ready: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, TEACHER_KEY)
    PricingSnapshot(
        date="2026-09-26",
        source="test",
        models={"vendor/other-model": {"default": ModelPrice(1e-6, 2e-6), "providers": {}}},
    ).save()
    factory = Factory()
    with pytest.raises(ServeSetupError, match=r"no price for teacher model 'vendor/teacher-model'.*pricing refresh"):
        build_server(spec, backend_factory=factory)
    assert factory.calls == []
    _, banner = build_server(spec, replay=True, backend_factory=factory, store=Store(paths.home() / "s.sqlite"))
    assert REPLAY_LINE in banner


def test_an_unreadable_pricing_snapshot_refuses_a_live_server(
    spec: TaskSpec, ready: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, TEACHER_KEY)
    paths.pricing_path().write_text("{not json", encoding="utf-8")
    with pytest.raises(ServeSetupError, match="pricing refresh"):
        build_server(spec, backend_factory=Factory())


def test_the_teacher_factory_is_asked_for_an_uncached_serve_teacher(
    spec: TaskSpec, ready: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real = runner.teacher_factory.make_teacher

    def spy(spec: TaskSpec, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(spec, **kwargs)

    monkeypatch.setattr(runner.teacher_factory, "make_teacher", spy)
    build_server(spec, replay=True, max_usd=2.0, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    assert seen["mode"] == "replay"
    assert seen["phase"] == "serve"
    assert seen["use_cache"] is False
    assert seen["run_cap"] == 2.0


# a slow teacher: the serve retry policy and the deadline ------------------------------------------------


def test_the_serve_teacher_gets_the_serve_retry_policy(
    spec: TaskSpec, ready: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real = runner.teacher_factory.make_teacher

    def spy(spec: TaskSpec, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(spec, **kwargs)

    monkeypatch.setattr(runner.teacher_factory, "make_teacher", spy)
    monkeypatch.setenv(KEY_ENV, TEACHER_KEY)
    pricing_snapshot()
    app, _ = build_server(spec, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    knobs = dict(runner.teacher_factory.SERVE_TEACHER_KNOBS)
    assert seen["phase"] == "serve"
    assert {name: seen[name] for name in knobs} == knobs
    teacher = app.state.cascade.teacher
    assert isinstance(teacher, LiveTeacher)
    assert (teacher.timeout, teacher.max_retries, teacher.retry_after_max) == (
        knobs["timeout"],
        knobs["max_retries"],
        knobs["retry_after_max"],
    )
    # a hung or rate-limited teacher gives up before the deadline, so the teacher's own error is reported
    policy_s = teacher.timeout * (teacher.max_retries + 1) + teacher.retry_after_max * teacher.max_retries
    assert policy_s < runner.SERVE_TEACHER_DEADLINE_S


def test_the_app_gives_the_teacher_the_serve_deadline(spec: TaskSpec, ready: Path, tmp_path: Path) -> None:
    app, banner = build_server(spec, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    assert app.state.cascade.teacher_deadline_s == runner.SERVE_TEACHER_DEADLINE_S
    assert f"teacher deadline: {runner.SERVE_TEACHER_DEADLINE_S:g} s per escalation, then a teacher timeout" in banner
    app, banner = build_server(
        spec, teacher_deadline_s=None, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite")
    )
    assert app.state.cascade.teacher_deadline_s is None
    assert not any(line.startswith("teacher deadline") for line in banner)
    for bad in (0.0, -5.0, math.nan, math.inf):
        with pytest.raises(ServeSetupError, match="teacher deadline"):
            build_server(spec, teacher_deadline_s=bad, backend_factory=Factory())


def test_a_rate_limited_live_teacher_falls_back_to_the_student_quickly(
    spec: TaskSpec, ready: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A teacher answering 429 with ``Retry-After: 60`` costs the client at most the deadline, not minutes."""
    monkeypatch.setenv(KEY_ENV, TEACHER_KEY)
    pricing_snapshot()
    store = Store(tmp_path / "s.sqlite")
    app, _ = build_server(spec, teacher_deadline_s=0.5, backend_factory=Factory(), store=store)
    limited = httpx.Response(429, headers={"retry-after": "60"}, json={"error": {"message": "rate limited"}})
    with respx.mock(base_url=TEACHER_BASE, assert_all_called=True) as router:
        route = router.post("/chat/completions").mock(return_value=limited)
        with TestClient(app) as client:
            started = time.perf_counter()
            resp = client.post("/v1/chat/completions", json=app_request(spec, UNSURE))
            elapsed = time.perf_counter() - started
    assert resp.status_code == 200
    assert resp.headers["x-taskdistill-route"] == "student-fallback"
    assert resp.headers["x-taskdistill-teacher-error"] == "timeout"
    assert resp.json()["choices"][0]["message"]["content"] == "lost_card"
    assert route.called
    assert elapsed < 5.0
    ledger = app.state.cascade.teacher.ledger
    assert ledger.open_reservations() == 0 and ledger.spent() == 0  # a 429 is never charged
    [row] = store.iter_served(TASK)
    assert (row.route, row.reason, row.teacher_mode) == ("student-fallback", "low_confidence", "live")


# binding and auth -----------------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.10", "::", "example.test"])
def test_binding_beyond_localhost_needs_a_token(spec: TaskSpec, host: str) -> None:
    with pytest.raises(UnsafeBindError, match="TASKDISTILL_SERVER_TOKEN"):
        build_server(spec, host=host, backend_factory=Factory())


def test_a_non_local_bind_enforces_the_token(
    spec: TaskSpec, ready: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TASKDISTILL_SERVER_TOKEN", "serve-token-1")
    app, banner = build_server(spec, host="0.0.0.0", port=8123, backend_factory=Factory(), store=Store(tmp_path / "s"))
    assert "listening on http://0.0.0.0:8123/v1 (OpenAI-compatible; any model name is accepted)" in banner
    assert any(line.startswith("auth: ") for line in banner)
    with TestClient(app) as client:
        denied = client.post("/v1/chat/completions", json=app_request(spec, SURE))
        allowed = client.post(
            "/v1/chat/completions", json=app_request(spec, SURE), headers={"Authorization": "Bearer serve-token-1"}
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200


@pytest.mark.parametrize(("host", "url"), [("localhost", "http://localhost:8000"), ("[::1]", "http://[::1]:8000")])
def test_localhost_needs_no_token(
    spec: TaskSpec, ready: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str, url: str
) -> None:
    monkeypatch.setenv("TASKDISTILL_SERVER_TOKEN", "serve-token-1")
    app, banner = build_server(spec, host=host, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    assert f"listening on {url}/v1 (OpenAI-compatible; any model name is accepted)" in banner
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=app_request(spec, SURE)).status_code == 200


# threshold ------------------------------------------------------------------------------------------


def test_auto_threshold_needs_threshold_json(spec: TaskSpec) -> None:
    make_run()
    with pytest.raises(ServeSetupError, match=r"threshold\.json"):
        build_server(spec, run_id=RUN, backend_factory=Factory())


def test_a_threshold_from_another_run_is_a_warning(spec: TaskSpec, ready: Path, tmp_path: Path) -> None:
    make_run("qwen2.5-1.5b-full-s13-teacher-mlx")
    app, banner = build_server(
        spec, run_id="qwen2.5-1.5b-full-s13-teacher-mlx", backend_factory=Factory(), store=Store(tmp_path / "s")
    )
    warnings = [line for line in banner if line.startswith("warning:")]
    assert len(warnings) == 1 and RUN in warnings[0]
    assert app.state.cascade.threshold == 0.8


def test_a_stale_adapter_sha_is_a_warning_even_when_the_run_id_still_matches(spec: TaskSpec, tmp_path: Path) -> None:
    """A threshold.json stamped for different weights than the served adapter warns, not only a different run."""
    from taskdistill.evaluate.predictions import adapter_sha256

    run_dir = make_run()
    (run_dir / "adapter" / "adapters.safetensors").write_bytes(b"weights v1")
    select_run()
    record(spec, {UNSURE: "Lost card"})
    write_threshold(0.8)  # no adapter_sha256 stamped (an older eval, or a hand-written file): no warning
    assert resolve_threshold(spec, "auto", RUN, run_dir / "adapter").warnings == ()

    threshold_path = paths.task_home(TASK) / "threshold.json"
    data = json.loads(threshold_path.read_text(encoding="utf-8"))
    data["adapter_sha256"] = "0" * 64  # stale: the run was retrained after this threshold was chosen
    threshold_path.write_text(json.dumps(data), encoding="utf-8")
    resolved = resolve_threshold(spec, "auto", RUN, run_dir / "adapter")
    assert len(resolved.warnings) == 1
    assert "different adapter" in resolved.warnings[0] and RUN in resolved.warnings[0]

    app, banner = build_server(spec, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    assert any("different adapter" in line for line in banner if line.startswith("warning:"))
    assert app.state.cascade.threshold == 0.8  # the stale threshold is still used, only with a warning

    data["adapter_sha256"] = adapter_sha256(run_dir / "adapter")  # matches the currently served adapter again
    threshold_path.write_text(json.dumps(data), encoding="utf-8")
    assert resolve_threshold(spec, "auto", RUN, run_dir / "adapter").warnings == ()


def write_run_threshold(value: float | None, run_id: str) -> None:
    data = {"threshold": value, "always_escalate": value is None, "escalation_rate": 0.1, "run_id": run_id}
    run_eval = paths.eval_dir(TASK) / run_id
    run_eval.mkdir(parents=True, exist_ok=True)
    (run_eval / "threshold.json").write_text(json.dumps(data), encoding="utf-8")


def test_another_run_uses_its_own_eval_threshold(spec: TaskSpec, ready: Path, tmp_path: Path) -> None:
    other = "qwen2.5-1.5b-full-s13-teacher-mlx"
    make_run(other)
    write_run_threshold(0.6, other)
    write_run_threshold(0.8, RUN)
    app, banner = build_server(spec, run_id=other, backend_factory=Factory(), store=Store(tmp_path / "s"))
    assert app.state.cascade.threshold == 0.6
    assert banner[2] == f"threshold: 0.6 (auto: eval/{other}/threshold.json, chosen on validation for run {other})"
    assert not any(line.startswith("warning:") for line in banner)
    selected, _ = build_server(spec, backend_factory=Factory(), store=Store(tmp_path / "s"))
    assert selected.state.cascade.threshold == 0.8


def test_the_run_threshold_is_used_without_a_task_level_file(spec: TaskSpec, tmp_path: Path) -> None:
    make_run()
    record(spec, {UNSURE: "Lost card"})
    write_run_threshold(None, RUN)
    assert resolve_threshold(spec, "auto", RUN).value == math.inf
    app, _ = build_server(spec, run_id=RUN, backend_factory=Factory(), store=Store(tmp_path / "s"))
    assert app.state.cascade.threshold == math.inf


def test_a_null_threshold_always_escalates(spec: TaskSpec, ready: Path, tmp_path: Path) -> None:
    write_threshold(None)
    app, banner = build_server(spec, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    assert banner[2].startswith("threshold: always escalate")
    with TestClient(app) as client:
        assert client.get("/healthz").json()["threshold"] is None
        resp = client.post("/v1/chat/completions", json=app_request(spec, UNSURE))
    assert resp.headers["x-taskdistill-route"] == "teacher"


def test_threshold_zero_never_escalates_on_confidence(spec: TaskSpec, ready: Path, tmp_path: Path) -> None:
    app, banner = build_server(spec, threshold=0, backend_factory=Factory(), store=Store(tmp_path / "s.sqlite"))
    assert banner[2] == "threshold: 0 (never escalates on confidence) (given)"
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=app_request(spec, UNSURE))
    assert resp.headers["x-taskdistill-route"] == "student"


@pytest.mark.parametrize(
    ("value", "expected"), [(0.0, 0.0), ("0.75", 0.75), (1, 1.0), ("inf", math.inf), ("AUTO", 0.8)]
)
def test_resolve_threshold_values(spec: TaskSpec, ready: Path, value: str | float, expected: float) -> None:
    assert resolve_threshold(spec, value, RUN).value == expected


@pytest.mark.parametrize("value", [-0.5, "nan", "abc", "-1"])
def test_resolve_threshold_rejects_bad_values(spec: TaskSpec, value: str | float) -> None:
    with pytest.raises(ServeSetupError, match="--threshold"):
        resolve_threshold(spec, value, RUN)


# the run --------------------------------------------------------------------------------------------


def test_an_explicit_run_needs_no_selection(spec: TaskSpec) -> None:
    run_dir = make_run("qwen2.5-0.5b-quick-s7-teacher-mlx")
    run = resolve_run(spec, "qwen2.5-0.5b-quick-s7-teacher-mlx")
    assert (run.run_id, run.adapter_path, run.base_model, run.backend, run.source) == (
        "qwen2.5-0.5b-quick-s7-teacher-mlx",
        run_dir / "adapter",
        BASE_MODEL,
        "mlx",
        "given",
    )


def test_the_banner_shows_a_local_base_model_relative_to_the_workspace(
    spec: TaskSpec, home: Path, tmp_path: Path
) -> None:
    local_base = paths.home() / "models" / "qwen-local"
    make_run(base=str(local_base))
    select_run()
    write_threshold(0.8)
    record(spec, {UNSURE: "Lost card"})
    factory = Factory()
    _, banner = build_server(spec, replay=True, backend_factory=factory, store=Store(tmp_path / "s.sqlite"))
    assert banner[1].startswith(f"run: {RUN}, from selected_run.json (base models/qwen-local, adapter ")
    assert str(home) not in "\n".join(banner)
    assert factory.calls[0][1] == str(local_base)


def test_run_resolution_errors(spec: TaskSpec) -> None:
    with pytest.raises(ServeSetupError, match=r"selected_run\.json"):
        resolve_run(spec)
    with pytest.raises(ServeSetupError, match="not found"):
        resolve_run(spec, "no-such-run")
    with pytest.raises(ServeSetupError, match="invalid run id"):
        resolve_run(spec, "../escape")
    run_dir = make_run()
    (run_dir / "adapter").rmdir()
    with pytest.raises(ServeSetupError, match="no adapter"):
        resolve_run(spec, RUN)


def test_a_run_trained_on_another_backend_is_refused(spec: TaskSpec) -> None:
    make_run("qwen2.5-0.5b-full-s13-teacher-torch", backend="torch", base="Qwen/Qwen2.5-0.5B-Instruct")
    write_threshold(0.8, "qwen2.5-0.5b-full-s13-teacher-torch")
    with pytest.raises(ServeSetupError, match="--backend torch"):
        build_server(spec, run_id="qwen2.5-0.5b-full-s13-teacher-torch", backend="mlx", backend_factory=Factory())


# run_server -----------------------------------------------------------------------------------------


def test_run_server_loads_the_model_then_starts_uvicorn(
    spec: TaskSpec, ready: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        worker: ModelWorker = app.state.worker
        started.update(kwargs, ready=worker.ready, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    lines: list[str] = []
    run_server(spec, replay=True, port=8765, backend_factory=Factory(), echo=lines.append)
    assert started["ready"] is True
    assert (started["host"], started["port"]) == ("127.0.0.1", 8765)
    assert REPLAY_LINE in lines
    assert lines[-1].startswith("model loaded in ")
    started["app"].state.worker.stop()


def test_run_server_reports_a_model_that_fails_to_load(
    spec: TaskSpec, ready: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(name: str, base_model: str, adapter_path: str | None) -> FakeBackend:
        backend = FakeBackend()

        def load() -> None:
            raise OSError("adapter weights are missing")

        backend.load = load  # type: ignore[method-assign]
        return backend

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("uvicorn must not start"))
    with pytest.raises(ServeSetupError, match="adapter weights are missing"):
        run_server(spec, replay=True, backend_factory=broken, echo=lambda line: None)
