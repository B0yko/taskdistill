from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from taskdistill.ledger import (
    DEFAULT_MAX_TOKENS,
    Ledger,
    choice_count,
    completion_limit,
    completion_upper_bound,
    default_global_cap,
    prompt_upper_bound,
    worst_case_cost,
)
from taskdistill.teacher.base import BudgetExceeded
from taskdistill.teacher.pricing import ModelPrice


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.sqlite"


def _reserve(ledger: Ledger, amount: float, **kw: Any) -> int:
    params: dict[str, Any] = {"task": "t", "phase": "label", "run_id": "run-1", "model": "vendor/model-a"}
    params.update(kw)
    return ledger.reserve(amount, **params)


# worst case ----------------------------------------------------------------------------------------
def test_worst_case_cost_hand_computed() -> None:
    body = {
        "model": "vendor/model-a",
        "messages": [
            {"role": "system", "content": "abc"},  # 3 bytes
            {"role": "user", "content": "héllo"},  # 6 bytes (é is two bytes)
        ],
        "max_tokens": 10,
    }
    price = ModelPrice(prompt=1e-6, completion=2e-6, request=0.001)
    # prompt bound = 3 + 6 + 2 x 16 = 41 tokens; 41 x 1e-6 + 10 x 2e-6 + 0.001 = 0.001061
    assert prompt_upper_bound(body) == 41
    assert completion_upper_bound(body) == 10
    assert worst_case_cost(body, price) == pytest.approx(0.001061, abs=1e-15)


def test_worst_case_counts_text_parts_only_and_falls_back_to_4096() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ab"},
                    {"type": "image_url", "image_url": {"url": "http://192.0.2.10/scan.png"}},
                    {"type": "text", "text": "€"},  # 3 bytes
                ],
            },
            {"role": "assistant", "content": None},
        ]
    }
    price = ModelPrice(prompt=1e-6, completion=1e-6)
    # prompt bound = (2 + 3 + 16) + (0 + 16) = 37; completion bound = 4096
    assert prompt_upper_bound(body) == 37
    assert completion_upper_bound(body) == DEFAULT_MAX_TOKENS == 4096
    assert worst_case_cost(body, price) == pytest.approx(4133e-6, abs=1e-15)


def test_completion_bound_uses_max_completion_tokens_when_max_tokens_is_absent() -> None:
    assert completion_upper_bound({"max_completion_tokens": 50}) == 50
    assert completion_upper_bound({"max_tokens": None, "max_completion_tokens": 50}) == 50
    assert completion_upper_bound({"max_tokens": 7, "max_completion_tokens": 50}) == 7


def test_worst_case_counts_tool_definitions_tool_calls_and_n() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        "max_tokens": 10,
        "n": 3,
    }
    price = ModelPrice(prompt=1e-6, completion=2e-6)
    # tools, canonical JSON: [{"function":{"name":"f","parameters":{"type":"object"}},"type":"function"}] = 76 bytes
    # tool_calls: [{"function":{"arguments":"{}","name":"f"},"id":"c1","type":"function"}] = 72 bytes
    # messages: (2 + 16) + (0 + 16 + 72) + (2 + 16) = 124; prompt bound = 124 + 76 = 200
    # completion bound = 10 per choice x n=3 = 30; cost = 200 x 1e-6 + 30 x 2e-6 = 260e-6
    assert prompt_upper_bound(body) == 200
    assert completion_upper_bound(body) == 30
    assert worst_case_cost(body, price) == pytest.approx(260e-6, abs=1e-15)


def test_worst_case_counts_legacy_functions() -> None:
    body = {
        "messages": [{"role": "assistant", "content": None, "function_call": {"name": "g", "arguments": "{}"}}],
        "functions": [{"name": "g"}],
    }
    # functions: [{"name":"g"}] = 14 bytes; function_call: {"arguments":"{}","name":"g"} = 29 bytes
    # prompt bound = 14 + (0 + 16 + 29) = 59
    assert prompt_upper_bound(body) == 59


def test_completion_bound_multiplies_by_n_and_ignores_invalid_n() -> None:
    assert completion_upper_bound({"max_tokens": 5, "n": 4}) == 20
    assert completion_upper_bound({"n": 2}) == 2 * DEFAULT_MAX_TOKENS
    for invalid in (0, -1, True, "3", None):
        assert completion_upper_bound({"max_tokens": 5, "n": invalid}) == 5
    assert choice_count({}) == 1
    assert completion_limit({}) is None
    assert completion_limit({"max_tokens": 0, "max_completion_tokens": 9}) == 9


# caps --------------------------------------------------------------------------------------------
def test_global_cap_defaults_to_five_dollars(monkeypatch: pytest.MonkeyPatch, ledger_path: Path) -> None:
    monkeypatch.delenv("TASKDISTILL_BUDGET_USD", raising=False)
    assert default_global_cap() == 5.00
    assert Ledger(ledger_path).global_cap == 5.00


def test_global_cap_is_read_from_the_environment_and_named(monkeypatch: pytest.MonkeyPatch, ledger_path: Path) -> None:
    monkeypatch.setenv("TASKDISTILL_BUDGET_USD", "0.30")
    ledger = Ledger(ledger_path)
    assert ledger.global_cap == 0.30
    _reserve(ledger, 0.20)
    with pytest.raises(BudgetExceeded, match=r"global cap TASKDISTILL_BUDGET_USD=0\.30"):
        _reserve(ledger, 0.11)


def test_invalid_budget_environment_is_rejected(monkeypatch: pytest.MonkeyPatch, ledger_path: Path) -> None:
    monkeypatch.setenv("TASKDISTILL_BUDGET_USD", "lots")
    with pytest.raises(ValueError, match="TASKDISTILL_BUDGET_USD"):
        Ledger(ledger_path)


def test_task_cap_is_named_and_scoped_to_the_task(ledger_path: Path) -> None:
    ledger = Ledger(ledger_path, global_cap=10.0)
    _reserve(ledger, 0.30, task="alpha", task_cap=0.50)
    with pytest.raises(BudgetExceeded, match=r"task cap budget\.usd_cap=0\.50") as info:
        _reserve(ledger, 0.30, task="alpha", task_cap=0.50)
    assert "global cap" not in str(info.value)
    assert "run cap" not in str(info.value)
    _reserve(ledger, 0.30, task="beta", task_cap=0.50)  # another task has its own allowance


def test_run_cap_is_named_and_scoped_to_the_run(ledger_path: Path) -> None:
    ledger = Ledger(ledger_path, global_cap=10.0)
    _reserve(ledger, 0.20, run_id="r1", run_cap=0.25)
    with pytest.raises(BudgetExceeded, match=r"run cap --max-usd=0\.25") as info:
        _reserve(ledger, 0.10, run_id="r1", run_cap=0.25)
    assert "task cap" not in str(info.value)
    _reserve(ledger, 0.20, run_id="r2", run_cap=0.25)


def test_every_crossed_cap_is_named(ledger_path: Path) -> None:
    ledger = Ledger(ledger_path, global_cap=1.0)
    _reserve(ledger, 0.9)
    with pytest.raises(BudgetExceeded) as info:
        _reserve(ledger, 0.2, task_cap=0.5, run_cap=0.5)
    message = str(info.value)
    assert "global cap TASKDISTILL_BUDGET_USD=1.00" in message
    assert "task cap budget.usd_cap=0.50" in message
    assert "run cap --max-usd=0.50" in message


def test_open_reservations_count_until_released_or_settled(ledger_path: Path) -> None:
    ledger = Ledger(ledger_path, global_cap=1.0)
    first = _reserve(ledger, 0.6)
    with pytest.raises(BudgetExceeded):
        _reserve(ledger, 0.5)
    assert ledger.committed() == pytest.approx(0.6)
    ledger.release(first)
    second = _reserve(ledger, 0.5)
    ledger.settle(second, 0.1)  # the real charge replaces the worst case
    _reserve(ledger, 0.9)
    assert ledger.committed() == pytest.approx(1.0)
    assert ledger.spent() == pytest.approx(0.1)


def test_sums_are_exact_at_the_cap(ledger_path: Path) -> None:
    # 0.1 + 0.2 > 0.3 in binary floating point; the ledger counts nano-dollars and accepts both.
    ledger = Ledger(ledger_path, global_cap=0.3)
    ledger.settle(_reserve(ledger, 0.1), 0.1)
    ledger.settle(_reserve(ledger, 0.2), 0.2)
    assert ledger.spent() == 0.3
    with pytest.raises(BudgetExceeded):
        _reserve(ledger, 1e-9)


def test_settle_and_release_validate_their_reservation(ledger_path: Path) -> None:
    ledger = Ledger(ledger_path, global_cap=1.0)
    res = _reserve(ledger, 0.1)
    ledger.settle(res, 0.05)
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(res, 0.05)
    with pytest.raises(ValueError, match="already settled"):
        ledger.release(res)
    with pytest.raises(KeyError):
        ledger.release(9999)
    with pytest.raises(ValueError):
        ledger.settle(_reserve(ledger, 0.1), -1.0)
    with pytest.raises(ValueError):
        _reserve(ledger, float("nan"))


# reporting ---------------------------------------------------------------------------------------
def _populate(ledger: Ledger) -> None:
    ledger.settle(_reserve(ledger, 0.5, task="a", phase="label", run_id="r1"), 0.1)
    ledger.settle(_reserve(ledger, 0.5, task="a", phase="bakeoff", run_id="r2"), 0.2)
    ledger.settle(_reserve(ledger, 0.5, task="b", phase="label", run_id="r3"), 0.3)
    ledger.release(_reserve(ledger, 0.5, task="b", phase="label", run_id="r3"))
    _reserve(ledger, 0.25, task="b", phase="bench", run_id="r4")  # left open


def test_spent_filters(ledger_path: Path) -> None:
    ledger = Ledger(ledger_path, global_cap=5.0)
    _populate(ledger)
    assert ledger.spent() == pytest.approx(0.6)
    assert ledger.spent(task="a") == pytest.approx(0.3)
    assert ledger.spent(phase="label") == pytest.approx(0.4)
    assert ledger.spent(task="a", phase="bakeoff") == pytest.approx(0.2)
    assert ledger.spent(run_id="r3") == pytest.approx(0.3)
    assert ledger.spent(task="b", phase="bench") == 0.0
    assert ledger.committed() == pytest.approx(0.85)
    assert ledger.open_reservations() == 1


def test_summary_and_export(ledger_path: Path, tmp_path: Path) -> None:
    ledger = Ledger(ledger_path, global_cap=5.0)
    _populate(ledger)
    summary = ledger.summary()
    assert summary["total"] == pytest.approx(0.6)
    assert summary["by_task"] == pytest.approx({"a": 0.3, "b": 0.3})
    assert summary["by_phase"] == pytest.approx({"bakeoff": 0.2, "label": 0.4})
    assert summary["by_task_phase"] == {"a": {"bakeoff": 0.2, "label": 0.1}, "b": {"label": 0.3}}
    assert summary["calls"] == 3
    assert summary["released"] == 1
    assert summary["open_reservations"] == 1
    assert summary["open_reserved"] == pytest.approx(0.25)
    assert summary["cap"] == 5.0

    out = tmp_path / "reports" / "spend.json"
    ledger.export_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total"] == pytest.approx(0.6)
    assert data["by_task_phase"]["b"]["label"] == pytest.approx(0.3)
    assert len(data["exported"]) == 10  # an ISO date, no clock time and no path
    assert str(tmp_path) not in out.read_text(encoding="utf-8")


# concurrency -------------------------------------------------------------------------------------
def test_threads_never_cross_the_cap(ledger_path: Path) -> None:
    ledger = Ledger(ledger_path, global_cap=1.0)
    accepted: list[int] = []
    barrier = threading.Barrier(16)

    def worker(n: int) -> None:
        barrier.wait()
        for _ in range(20):
            try:
                res = _reserve(ledger, 0.01, run_id=f"thread-{n}")
            except BudgetExceeded:
                continue
            accepted.append(res)
            if res % 2:
                ledger.settle(res, 0.01)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(accepted) == 100
    assert ledger.committed() == pytest.approx(1.0)
    assert ledger.committed() <= 1.0


def _process_worker(path: str, start: Any, out: Any, attempts: int) -> None:
    ledger = Ledger(path, global_cap=0.37)
    start.wait(60)
    accepted = 0
    for i in range(attempts):
        try:
            res = ledger.reserve(0.01, task="t", phase="label", run_id=f"proc-{os.getpid()}", model="m")
        except BudgetExceeded:
            continue
        accepted += 1
        if i % 2:
            ledger.settle(res, 0.01)
    out.put(accepted)


@pytest.mark.timeout(120)
def test_processes_never_cross_the_cap(ledger_path: Path) -> None:
    Ledger(ledger_path, global_cap=0.37)  # create the schema first
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    out = ctx.Queue()
    procs = [ctx.Process(target=_process_worker, args=(str(ledger_path), start, out, 30)) for _ in range(4)]
    for p in procs:
        p.start()
    start.set()
    counts = [out.get(timeout=90) for _ in procs]
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    assert sum(counts) == 37
    ledger = Ledger(ledger_path, global_cap=0.37)
    assert ledger.committed() == pytest.approx(0.37)
    assert ledger.committed() <= 0.37
