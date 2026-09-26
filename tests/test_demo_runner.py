"""`taskdistill demo` spend rules: one demo invocation is one command run.

Every live teacher the demo builds (the capture upstream, curate's labelling) is charged to one ledger run, so
``--max-usd`` caps them together; the serve smoke test, whose teacher ``build_server`` charges to a run of its own,
gets only what is left of the cap. The $0.50 ``--yes`` gate projects the capture and the labelling once.

The live teacher is stood in for by the packaged invoices recording (``make_teacher`` is wrapped to record what the
demo asks for and to answer from the replay), prices come from a test snapshot, and curate's length filter counts
words, so no test here loads a model, reaches the network or has a real API key.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from taskdistill.config import load_task
from taskdistill.demos import runner as demo_runner
from taskdistill.demos.runner import DemoContext, DemoData, run_demo
from taskdistill.ledger import Ledger
from taskdistill.serve import runner as serve_runner
from taskdistill.store import Store
from taskdistill.teacher import factory
from taskdistill.teacher.client import SpendNotConfirmed
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*httpx2.*")
    from fastapi import FastAPI

KEY_ENV = ("OPENROUTER_API_KEY", "TASKDISTILL_TEACHER_API_KEY")
SPEC_ENV = ("TASKDISTILL_TEACHER_MODEL", "TASKDISTILL_TEACHER_BASE_URL")
#: The invoices quick profile: 120 training inputs go through the capture, 24 + 36 are labelled by curate.
CAPTURED, LABELLED = 120, 60


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty workspace, the bundled specs' defaults and a placeholder teacher key (live mode needs one set)."""
    for name in (*KEY_ENV, *SPEC_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", "sk-test-demo-runner")
    workspace = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    monkeypatch.chdir(tmp_path)
    return workspace


class WordTokenizer:
    """One token per whitespace-separated word, plus two per message (curate's length filter only counts)."""

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        return [7] * sum(len(m["content"].split()) + 2 for m in messages)


@pytest.fixture(autouse=True)
def word_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    import taskdistill.models

    monkeypatch.setattr(taskdistill.models, "load_tokenizer", lambda *args, **kwargs: WordTokenizer())


def price_per_request(monkeypatch: pytest.MonkeyPatch, fee: float) -> None:
    """Price the invoices teacher at a flat ``fee`` per request, so a request's worst case is exactly ``fee``."""
    spec = load_task("invoices")
    price = ModelPrice(0.0, 0.0, fee)
    providers = {spec.teacher.provider: price} if spec.teacher.provider else {}
    snapshot = PricingSnapshot(
        date="2026-09-26", source="test", models={spec.teacher.model: {"default": price, "providers": providers}}
    )
    monkeypatch.setattr(factory, "load_pricing", lambda *args, **kwargs: snapshot)


@pytest.fixture
def teachers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every ``make_teacher`` call (its phase, run id and run cap); the teacher answers from the recording."""
    calls: list[dict[str, Any]] = []
    real = factory.make_teacher

    def recording_teacher(spec: Any, *, mode: str, phase: str, run_id: str, **kwargs: Any) -> Any:
        calls.append({"mode": mode, "phase": phase, "run_id": run_id, "run_cap": kwargs.get("run_cap")})
        return real(spec, mode="replay", phase=phase, run_id=run_id)

    monkeypatch.setattr(factory, "make_teacher", recording_teacher)
    return calls


def live_demo(until: str, *, yes: bool = False, max_usd: float | None = None) -> list[str]:
    lines: list[str] = []
    run_demo("invoices", profile="quick", live=True, yes=yes, until=until, max_usd=max_usd, backend="torch",
             log=lines.append)  # fmt: skip
    return lines


# one ledger run ----------------------------------------------------------------------------------------------------
@pytest.mark.slow  # runs the capture proxy and curate on the quick invoices data
def test_a_live_demo_charges_capture_and_labelling_to_one_ledger_run(
    home: Path, teachers: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    price_per_request(monkeypatch, 0.001)

    lines = live_demo("curate", max_usd=0.25)

    assert [call["phase"] for call in teachers] == ["demo-capture", "curate-label"]
    assert all(call["mode"] == "live" for call in teachers)
    run_ids = {call["run_id"] for call in teachers}
    assert len(run_ids) == 1, f"each stage got its own ledger run, so --max-usd capped each one: {run_ids}"
    (run_id,) = run_ids
    assert run_id.startswith("demo-invoices-")
    assert all(call["run_cap"] == 0.25 for call in teachers)
    assert any(f"ledger run {run_id}" in line and "--max-usd $0.25 for the whole run" in line for line in lines)
    captures = [row for row in Store(home / "store.sqlite").iter_captures("invoices") if row.captured]
    assert len(captures) == CAPTURED


@pytest.mark.slow  # runs the capture proxy and curate on the quick invoices data
def test_a_second_demo_invocation_is_a_new_ledger_run(
    teachers: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    price_per_request(monkeypatch, 0.001)
    minted = iter(["demo-invoices-first", "demo-invoices-second"])
    monkeypatch.setattr(factory, "new_run_id", lambda command: next(minted))

    live_demo("capture")
    live_demo("curate")  # everything is captured: only the labelling teacher is built

    assert [(call["phase"], call["run_id"]) for call in teachers] == [
        ("demo-capture", "demo-invoices-first"),
        ("curate-label", "demo-invoices-second"),
    ]


# the $0.50 --yes gate ----------------------------------------------------------------------------------------------
@pytest.mark.slow  # runs the capture proxy and curate on the quick invoices data
def test_the_yes_gate_projects_capture_and_labelling_together(
    home: Path, teachers: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """$0.36 of capture and $0.18 of labelling each stay under $0.50; together they need --yes."""
    price_per_request(monkeypatch, 0.003)

    with pytest.raises(SpendNotConfirmed, match=r"projected teacher spend \$0\.54 is above \$0\.50"):
        live_demo("curate")

    assert teachers == []
    assert list(Store(home / "store.sqlite").iter_captures("invoices")) == []

    lines = live_demo("curate", yes=True)
    assert [call["phase"] for call in teachers] == ["demo-capture", "curate-label"]
    assert any(f"{CAPTURED + LABELLED} uncached live requests" in line for line in lines)


@pytest.mark.slow  # runs the capture proxy and curate on the quick invoices data
def test_the_gate_counts_only_the_stages_the_run_reaches(
    home: Path, teachers: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    price_per_request(monkeypatch, 0.003)

    lines = live_demo("capture")  # $0.36 of capture alone: no --yes needed
    assert any(f"{CAPTURED} uncached live requests for this demo run ({CAPTURED} capture, 0 labelling)" in line
               for line in lines)  # fmt: skip

    lines = live_demo("curate")  # resumed: only the $0.18 of labelling is left
    assert any(f"{LABELLED} uncached live requests for this demo run (0 capture, {LABELLED} labelling)" in line
               for line in lines)  # fmt: skip
    assert [call["phase"] for call in teachers] == ["demo-capture", "curate-label"]


@pytest.mark.slow  # runs the capture proxy and curate on the quick invoices data
def test_labelling_answered_by_the_response_cache_is_not_projected(
    home: Path, teachers: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from taskdistill.teacher.base import TeacherResult
    from taskdistill.teacher.cache import ResponseCache, request_context
    from taskdistill.teacher.request_key import request_key
    from taskdistill.teacher.requests import build_teacher_request

    price_per_request(monkeypatch, 0.003)
    spec = load_task("invoices")
    data = demo_runner.load_demo_data("invoices", "quick")
    cache = ResponseCache(home / "cache.sqlite")
    for row in data.records["valid"] + data.records["test"]:
        body = build_teacher_request(spec, row["input"])
        result = TeacherResult(key=request_key(body), output="{}", response={}, usage={}, latency_ms=None,
                               provider=None, finish_reason="stop", created=0.0, source="live")  # fmt: skip
        cache.put(result, request_context(body))

    lines = live_demo("curate")  # $0.36 of capture; the labelling costs nothing

    assert any(f"{CAPTURED} uncached live requests for this demo run ({CAPTURED} capture, 0 labelling)" in line
               for line in lines)  # fmt: skip


# the serve smoke test ----------------------------------------------------------------------------------------------
def smoke_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    def answer() -> dict[str, Any]:
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}}]}

    return app


def serve_context(mode: str, max_usd: float | None, log: Callable[[str], Any]) -> DemoContext:
    return DemoContext(
        name="invoices", profile="quick", spec=load_task("invoices"), mode=mode, yes=False, max_usd=max_usd,
        backend="torch", base=None, seed=None, log=log, data=DemoData(records={}, card_info={}, smoke=["a", "b"]),
        ledger_run_id="demo-invoices-20260926T120000",
    )  # fmt: skip


@pytest.fixture
def build_server_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_build_server(spec: Any, **kwargs: Any) -> tuple[FastAPI, list[str]]:
        calls.append(copy.deepcopy(kwargs))
        return smoke_app(), []

    monkeypatch.setattr(serve_runner, "build_server", fake_build_server)
    return calls


def test_the_smoke_test_gets_only_what_is_left_of_the_demo_cap(
    home: Path, build_server_calls: list[dict[str, Any]]
) -> None:
    ledger = Ledger()
    spent = ledger.reserve(0.30, task="invoices", phase="curate-label", run_id="demo-invoices-20260926T120000",
                           model="m")  # fmt: skip
    ledger.settle(spent, 0.30)
    ledger.reserve(0.05, task="invoices", phase="demo-capture", run_id="demo-invoices-20260926T120000", model="m")
    other = ledger.reserve(0.20, task="invoices", phase="curate-label", run_id="curate-20260926T110000", model="m")
    ledger.settle(other, 0.20)
    lines: list[str] = []

    demo_runner.stage_serve(serve_context("live", 1.0, lines.append))

    (call,) = build_server_calls
    assert call["max_usd"] == pytest.approx(0.65), "the smoke test must not get a fresh --max-usd of its own"
    assert "smoke-test spend cap: $0.6500 left of --max-usd $1.00" in "\n".join(lines)


def test_a_spent_cap_leaves_the_smoke_test_nothing(home: Path, build_server_calls: list[dict[str, Any]]) -> None:
    ledger = Ledger()
    spent = ledger.reserve(0.5, task="invoices", phase="curate-label", run_id="demo-invoices-20260926T120000",
                           model="m")  # fmt: skip
    ledger.settle(spent, 0.5)

    demo_runner.stage_serve(serve_context("live", 0.5, lambda line: None))

    assert build_server_calls[0]["max_usd"] == 0.0


@pytest.mark.parametrize(("mode", "max_usd"), [("replay", 1.0), ("live", None)])
def test_without_live_spend_to_subtract_the_cap_is_passed_through(
    mode: str, max_usd: float | None, home: Path, build_server_calls: list[dict[str, Any]]
) -> None:
    demo_runner.stage_serve(serve_context(mode, max_usd, lambda line: None))

    assert build_server_calls[0]["max_usd"] == max_usd
    assert not (home / "ledger.sqlite").exists()
