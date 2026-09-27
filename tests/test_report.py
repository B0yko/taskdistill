"""Report: quality rows, operating point, composed cost/latency, break-even, bench cross-check and served traffic."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pytest

from taskdistill import paths
from taskdistill.config import CostSpec, TaskSpec
from taskdistill.curate.extract import input_hash
from taskdistill.evaluate.bootstrap import diff_name
from taskdistill.evaluate.plots import plot_reliability, plot_threshold_curve
from taskdistill.evaluate.threshold import ThresholdResult
from taskdistill.ledger import Ledger
from taskdistill.report.builder import (
    ReportError,
    build_break_even,
    build_report,
    escalated,
    latency_summary,
    local_cost_usd,
    wilson_interval,
)
from taskdistill.report.render import interval, pct, render_markdown, usd
from taskdistill.store import Store
from taskdistill.teacher.base import TeacherResult
from taskdistill.teacher.cache import ResponseCache, request_context
from taskdistill.teacher.factory import workspace_recording
from taskdistill.teacher.replay import expected_manifest, write_recording
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

TASK = "intents"
BASE_05 = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
BASE_15 = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SELECTED = "qwen2.5-0.5b-full-s13"
QUERIES = {
    "q1": "My card still has not arrived after two weeks.",
    "q2": "I think I lost my card yesterday evening.",
    "q3": "How do I top up my account by bank transfer?",
    "q4": "Card not here yet, reach me at jo@example.com please.",
}
TRAIN = ["Top up failed twice this morning.", "Where can I see my card delivery status?"]
VALID = ["My card was stolen on the train."]
# Recorded teacher latency (ms) and cost (USD) per test request; q1/q2 come from the cache, q3/q4 from the recording.
TEACHER = {"q1": (400.0, 1e-4), "q2": (500.0, 2e-4), "q3": (600.0, 3e-4), "q4": (800.0, 4e-4)}
LABELLING_COST = {"train": 5e-4, "valid": 6e-4}
# Student confidence and in-process latency (ms) from the selected run's test predictions cache.
PREDS = {"q1": (0.95, 50.0), "q2": (0.40, 60.0), "q3": (0.90, 70.0), "q4": (0.20, 80.0)}
THRESHOLD = 0.5
WALL_SECONDS = 600.0


def make_spec(cost: dict[str, float] | None = None, task: str = TASK) -> TaskSpec:
    spec = TaskSpec.model_validate(
        {
            "task": task,
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": {
                "model": "vendor/model-a",
                "max_tokens": 24,
                "extra_body": {"provider": {"order": ["alpha"], "allow_fallbacks": False}},
            },
            "student": {"system_prompt": "Classify the message.", "base_model": BASE_05},
            "cascade": {"target": 0.97},
            "cost": cost or {},
        }
    )
    spec.teacher_prompt = "Label the customer's message with one intent.\n"
    spec.labels = ["card_arrival", "lost_card", "top_up", "stolen_card"]
    return spec


def _stat(point: float, lo: float, hi: float) -> dict[str, Any]:
    return {"point": point, "lo": lo, "hi": hi, "valid_resamples": 1000, "resamples": 1000, "seed": 0}


def _calibration(ece: float, auroc: float) -> dict[str, Any]:
    report = {"n": 4, "ece": ece, "brier": 0.1, "auroc": auroc, "bins": []}
    return {
        "raw": {"vs_gold": report, "vs_teacher": {**report, "ece": ece + 0.01}},
        "isotonic": {"vs_gold": {**report, "ece": ece / 2}, "vs_teacher": report},
        "isotonic_fit": {"n_fit": 50, "reference": "gold"},
    }


def make_eval(
    run_id: str,
    *,
    base: str,
    labels: str,
    seed: int | None,
    student: tuple[float, float, float],
    ece: float = 0.05,
    auroc: float = 0.8,
    selected: bool = False,
    task_type: str = "classification",
) -> dict[str, Any]:
    acc, f1, agree = student
    systems: dict[str, Any] = {
        "student": {
            "metrics": {"accuracy": acc, "macro_f1": f1, "agreement": agree},
            "latency_ms": {"p50": 65.0, "p95": 78.5, "mean": 65.0, "n": 4},
            "calibration": _calibration(ece, auroc),
        },
        "teacher": {"metrics": {"accuracy": 0.95, "macro_f1": 0.94}},
        "cascade": {
            "metrics": {"accuracy": 0.94, "macro_f1": 0.93, "agreement": 0.975},
            "threshold": THRESHOLD,
            "escalation_rate": 0.5,
        },
    }
    if labels != "zero-shot":
        systems["tfidf"] = {"metrics": {"accuracy": 0.80, "macro_f1": 0.78, "agreement": 0.82}, "C": 3.0}
    stats = {
        "student.accuracy": _stat(acc, acc - 0.05, acc + 0.05),
        "student.macro_f1": _stat(f1, f1 - 0.05, f1 + 0.05),
        "student.agreement": _stat(agree, agree - 0.04, agree + 0.04),
        "student.ece": _stat(ece, ece - 0.02, ece + 0.03),
        "student.auroc": _stat(auroc, auroc - 0.05, auroc + 0.05),
        "teacher.accuracy": _stat(0.95, 0.92, 0.98),
        "teacher.agreement": _stat(1.0, 1.0, 1.0),
        "teacher.macro_f1": _stat(0.94, 0.90, 0.97),
        "cascade.accuracy": _stat(0.94, 0.90, 0.97),
        "cascade.macro_f1": _stat(0.93, 0.89, 0.96),
        "cascade.agreement": _stat(0.975, 0.95, 1.0),
        "tfidf.accuracy": _stat(0.80, 0.75, 0.85),
    }
    diffs = {
        diff_name("cascade.accuracy", "teacher.accuracy"): {
            "a": "cascade.accuracy",
            "b": "teacher.accuracy",
            **_stat(-0.01, -0.03, 0.01),
        },
        diff_name("cascade.agreement", "teacher.agreement"): {
            "a": "cascade.agreement",
            "b": "teacher.agreement",
            **_stat(-0.025, -0.05, 0.0),
        },
    }
    threshold = ThresholdResult(
        threshold=THRESHOLD,
        escalation_rate=0.45,
        quality=0.972,
        target_value=0.97,
        met=True,
        warning=None,
        curve=[(0.2, 0.2, 0.93), (THRESHOLD, 0.45, 0.972), (math.inf, 1.0, 1.0)],
        reference="teacher",
        metric="agreement",
        target=0.97,
        n=20,
        teacher_quality=1.0,
    )
    return {
        "task": TASK,
        "task_type": task_type,
        "run_id": run_id,
        "split": "test",
        "n": 4,
        "date": "2026-09-26T10:00:00+00:00",
        "command": f"taskdistill eval --task {TASK} --run {run_id}",
        "hardware": {"model": "Mac17,4", "cpu": "Apple M5", "memory_gb": 24.0},
        "run": {"base_model": base, "labels": labels, "seed": seed, "profile": "full", "backend": "mlx"},
        "systems": systems,
        "threshold": threshold.to_dict(),
        "operating_point": {
            "valid": {"escalation_rate": 0.45, "quality": 0.972, "target_value": 0.97, "met": True},
            "test": {
                "escalation_rate": 0.5,
                "quality": 0.975,
                "teacher_quality": 1.0,
                "target_value": 0.97,
                "met": True,
            },
        },
        "confidence_comparison": {
            "reference": "teacher",
            "definitions": {"primary": "trie-constrained label probability", "alternative": "mean token log-prob"},
            "valid": {
                "primary": {"auroc_vs_teacher": 0.81, "auroc_vs_gold": 0.8, "accuracy_vs_gold": 0.9},
                "alternative": {"auroc_vs_teacher": 0.7, "auroc_vs_gold": 0.69, "accuracy_vs_gold": 0.88},
                "chosen": "primary",
                "better_auroc": "primary",
            },
        },
        "bootstrap": {"paired": {"method": "paired", "n": 4, "stats": stats, "diffs": diffs}, "cluster": None},
        "per_group": None,
        "per_field": None,
        "per_trait": None,
        "test_scorings": 2 if selected else 1,
        "profile_fast": False,
        "files": {
            "threshold_curve": f"{TASK}/eval/{run_id}/threshold_curve.png",
            "reliability": f"{TASK}/eval/{run_id}/reliability_test.png",
        },
    }


EVALS: dict[str, dict[str, Any]] = {
    SELECTED: {
        "base": BASE_05, "labels": "teacher", "seed": 13, "student": (0.90, 0.85, 0.92), "ece": 0.05, "auroc": 0.80,
    },
    "qwen2.5-0.5b-full-s14": {
        "base": BASE_05, "labels": "teacher", "seed": 14, "student": (0.86, 0.83, 0.90), "ece": 0.07, "auroc": 0.76,
    },
    "qwen2.5-0.5b-full-s15": {
        "base": BASE_05, "labels": "teacher", "seed": 15, "student": (0.88, 0.84, 0.91), "ece": 0.06, "auroc": 0.78,
    },
    "qwen2.5-1.5b-full-s13": {"base": BASE_15, "labels": "teacher", "seed": 13, "student": (0.91, 0.88, 0.93)},
    "qwen2.5-0.5b-full-s13-gold": {"base": BASE_05, "labels": "gold", "seed": 13, "student": (0.93, 0.90, 0.89)},
    "zero-shot-qwen2.5-0.5b": {"base": BASE_05, "labels": "zero-shot", "seed": None, "student": (0.40, 0.30, 0.41)},
}  # fmt: skip


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _record(key: str, latency: float, cost: float) -> dict[str, Any]:
    usage = {"prompt_tokens": 120, "completion_tokens": 3, "cost": cost}
    return {"key": key, "output": "card_arrival", "usage": usage, "latency_ms": latency, "provider": "Alpha",
            "finish_reason": "stop", "timestamp": 1790000000.0}  # fmt: skip


def make_workspace(
    spec: TaskSpec,
    *,
    ledger: bool = True,
    curate_cost: bool = True,
    bench: bool = False,
    serve_rows: bool = False,
    unrecorded: tuple[str, ...] = (),
) -> Store:
    """Evals, selection, predictions, train log, curated data, teacher records and PNGs under TASKDISTILL_HOME."""
    task = spec.task
    store = Store()
    splits = {"train": TRAIN, "valid": VALID, "test": list(QUERIES.values())}
    data = paths.data_dir(task)
    for split, texts in splits.items():
        store.add_imports(task, "inputs", [{"input": t, "gold": "top_up", "meta": {"split": split}} for t in texts])
        _jsonl(data / f"{split}.jsonl", [{"messages": [{"role": "user", "content": t}]} for t in texts])
        sidecar = [{"input_hash": input_hash(t), "gold": "top_up", "teacher": "top_up", "meta": {}} for t in texts]
        _jsonl(data / f"{split}.meta.jsonl", sidecar)

    cache = ResponseCache()
    recorded = []
    for name in ("q1", "q2"):
        body = build_teacher_request(spec, QUERIES[name])
        latency, cost = TEACHER[name]
        result = TeacherResult(
            key=request_key(body), output="card_arrival", response={}, usage={"cost": cost}, latency_ms=latency,
            provider="Alpha", finish_reason="stop", created=1790000000.0, source="live",
        )  # fmt: skip
        cache.put(result, request_context(body))
    for name in ("q3", "q4"):
        if name not in unrecorded:
            recorded.append(_record(request_key(build_teacher_request(spec, QUERIES[name])), *TEACHER[name]))
    for split in ("train", "valid"):
        for text in splits[split]:
            recorded.append(_record(request_key(build_teacher_request(spec, text)), 300.0, LABELLING_COST[split]))
    manifest = {**expected_manifest(spec, "2026-09-26"), "created": "2026-09-26T12:00:00+00:00"}
    write_recording(workspace_recording(task), manifest, recorded)

    for run_id, cfg in EVALS.items():
        ev = make_eval(run_id, selected=run_id == SELECTED, **cfg)
        target = paths.eval_dir(task) / run_id / "eval_test.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ev), encoding="utf-8")
    selection = {
        "run_id": SELECTED,
        "reason": "best validation agreement",
        "date": "2026-09-26T09:00:00+00:00",
        "rule": {"metric": "agreement"},
        "candidates": {rid: {"valid_metric": 0.9, "p95_ms": 80.0, "base_model": BASE_05} for rid in EVALS},
    }
    (paths.task_home(task) / "selected_run.json").write_text(json.dumps(selection), encoding="utf-8")

    run_dir = paths.runs_dir(task) / SELECTED
    header = {"type": "header", "data_sha256": "0" * 64, "adapter_sha256": None, "alternatives": True, "backend": "mlx"}
    rows = [
        {"id": input_hash(QUERIES[name]), "answer": "card_arrival", "value": "card_arrival", "confidence": conf,
         "alt_value": "card_arrival", "alt_confidence": 0.99, "latency_ms": latency, "prompt_tokens": 20,
         "completion_tokens": 2, "field_confidences": None}
        for name, (conf, latency) in reversed(list(PREDS.items()))  # cache order need not follow the data
    ]  # fmt: skip
    _jsonl(run_dir / "preds_test.jsonl", [header, *rows])
    for rid, base, seed in ((SELECTED, BASE_05, 13), ("qwen2.5-1.5b-full-s13", BASE_15, 13)):
        log = {
            "run_id": rid, "base_model": base, "backend": "mlx", "profile": "full", "seed": seed, "labels": "teacher",
            "n_train": 2, "iterations": 100, "epochs": 2.0, "wall_seconds": WALL_SECONDS if rid == SELECTED else 1800.0,
            "peak_memory_gb": 3.2, "tokens_per_second": 1500.0, "adapter_size_mb": 17.5,
            "adapter_dir": f"{task}/runs/{rid}/adapter",
        }  # fmt: skip
        (paths.runs_dir(task) / rid).mkdir(parents=True, exist_ok=True)
        (paths.runs_dir(task) / rid / "train_log.json").write_text(json.dumps(log), encoding="utf-8")

    stats: dict[str, Any] = {
        "splits": {"train": 2, "valid": 1, "test": 4},
        "length": {"max_seq_len": 512, "dropped": 3, "splits": {"train": {"n": 4, "kept": 2, "dropped": 2}}},
        "labelling": {"requested": 5, "concurrency": 8, "cost_usd": 0.5 if curate_cost else 0.0},
        "pii": {"hits": {"email": 1}},
    }
    (paths.data_dir(task) / "curate_stats.json").write_text(json.dumps(stats), encoding="utf-8")

    eval_dir = paths.eval_dir(task) / SELECTED
    curve = [(0.2, 0.2, 0.93), (0.5, 0.45, 0.972)]
    plot_threshold_curve(curve, curve[1], eval_dir / "threshold_curve.png", "quality against escalation")
    plot_reliability([0.95, 0.4, 0.9, 0.2], [1, 0, 1, 0], eval_dir / "reliability_test.png")
    plot_reliability([0.9, 0.5], [1, 0], eval_dir / "reliability_valid.png")

    if ledger:
        book = Ledger()
        for phase, amount in (("curate-label", 0.012), ("demo-capture", 0.003), ("bakeoff", 0.1), ("serve", 0.05)):
            res = book.reserve(amount, task=task, phase=phase, run_id=f"{phase}-1", model="vendor/model-a")
            book.settle(res, amount)
    if bench:
        _write_bench(task, "bench_20260926T120000Z.json", mode="student_only", run_id=SELECTED)
        _write_bench(task, "bench_20260926T130000Z.json", mode="cascade", run_id=SELECTED)
        _write_bench(task, "bench_20260926T140000Z.json", mode="student_only", run_id="other-run")
    return store


BENCH_STUDENT_MS = {"q1": 40.0, "q2": 45.0, "q3": 55.0, "q4": 60.0}


def _write_bench(task: str, name: str, *, mode: str, run_id: str, queries: tuple[str, ...] | None = None) -> None:
    requests = [
        {"i": i, "input_hash": input_hash(QUERIES[q]), "latency_ms": ms, "status": 200, "route": "student"}
        for i, (q, ms) in enumerate(BENCH_STUDENT_MS.items())
        if queries is None or q in queries
    ]
    stamp = name.removeprefix("bench_").removesuffix(".json")
    bench = {
        "task": task,
        "mode": mode,
        "date": f"2026-09-26T{stamp[9:11]}:00:00+00:00",
        "run_id": run_id,
        "threshold": 0.0 if mode == "student_only" else THRESHOLD,
        "teacher": "live",
        "n": 4,
        "warmup": 20,
        "latency_ms": {"p50": 50.0, "p95": 59.25, "mean": 50.0, "n": 4}
        if mode == "student_only"
        else {"p50": 320.0, "p95": 850.0, "mean": 380.0, "n": 4},
        "escalation_rate": 0.0 if mode == "student_only" else 0.48,
        "spend_usd": 0.0 if mode == "student_only" else 0.0123,
        "machine_state": {"load_average": [1.2, 1.1, 1.0]},
        "hardware": {"model": "Mac17,4", "cpu": "Apple M5", "memory_gb": 24.0, "os": "macOS 26.0"},
        "command": f"taskdistill bench --url http://127.0.0.1:8000 --task {task} --n 4 --warmup 20",
        "requests": requests if mode == "student_only" else [],
    }
    if queries is not None:
        bench["n"] = bench["latency_ms"]["n"] = len(requests)
    target = paths.task_home(task) / "bench" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bench), encoding="utf-8")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TASKDISTILL_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TASKDISTILL_BUDGET_USD", raising=False)
    return paths.home()


def _row(report: dict[str, Any], system: str, name_part: str = "") -> dict[str, Any]:
    rows = [r for r in report["quality"]["rows"] if r["system"] == system and name_part in r["name"]]
    assert len(rows) == 1, [r["name"] for r in report["quality"]["rows"]]
    return rows[0]


# -- formulas ----------------------------------------------------------------------------------------------------


def test_local_cost_is_energy_without_amortisation() -> None:
    cost = CostSpec()
    # 20 W for one hour is 0.02 kWh; at $0.30/kWh that is $0.006.
    assert local_cost_usd(3_600_000.0, cost) == pytest.approx(0.006)
    assert local_cost_usd(65.0, cost) == pytest.approx(0.02 * 65 / 3_600_000 * 0.30)


def test_local_cost_adds_amortisation_only_when_enabled() -> None:
    cost = CostSpec(local_watts=20, usd_per_kwh=0.30, hardware_usd=1800, amortisation_hours=9000)
    assert cost.amortisation_enabled
    # energy $0.006 per hour plus $1800 / 9000 h = $0.20 per hour of wall time
    assert local_cost_usd(3_600_000.0, cost) == pytest.approx(0.206)
    assert local_cost_usd(1_800_000.0, cost) == pytest.approx(0.103)


def test_escalation_rule_and_helpers() -> None:
    assert escalated(0.4, 0.5) and not escalated(0.5, 0.5) and not escalated(0.9, 0.5)
    assert escalated(0.99, None) and escalated(None, 0.5)
    assert latency_summary([]) == {"p50": None, "p95": None, "mean": None, "n": 0}
    assert latency_summary([10.0, 20.0, None]) == {"p50": 15.0, "p95": pytest.approx(19.5), "mean": 15.0, "n": 2}
    lo, hi = wilson_interval(2, 9) or (0.0, 0.0)
    assert lo == pytest.approx(0.0632, abs=1e-3) and hi == pytest.approx(0.5474, abs=1e-3)
    assert wilson_interval(0, 0) is None


# -- quality and operating point ---------------------------------------------------------------------------------


def test_quality_rows(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    rows = report["quality"]["rows"]
    assert [r["system"] for r in rows] == ["teacher", "zero-shot", "tfidf", "student", "student", "student", "cascade"]
    assert report["quality"]["n"] == 4

    teacher = _row(report, "teacher")
    assert teacher["name"] == "teacher (vendor/model-a)" and teacher["recorded"] is True
    assert teacher["metrics"] == {"accuracy": 0.95, "macro_f1": 0.94, "agreement": None}
    assert teacher["ci"]["accuracy"] == [0.92, 0.98]

    seeds = _row(report, "student", "qwen2.5-0.5b (teacher labels, 3 seeds)")
    assert seeds["n_seeds"] == 3
    assert [s["seed"] for s in seeds["per_seed"]] == [13, 14, 15]
    assert seeds["metrics"]["accuracy"] == pytest.approx(0.88)
    assert seeds["std"]["accuracy"] == pytest.approx(0.02)  # sample std of 0.90, 0.86, 0.88
    assert seeds["metrics"]["ece"] == pytest.approx(0.06) and seeds["std"]["ece"] == pytest.approx(0.01)
    assert seeds["metrics"]["auroc"] == pytest.approx(0.78) and seeds["std"]["auroc"] == pytest.approx(0.02)
    assert seeds["selected_run"] == SELECTED
    assert seeds["ci"]["accuracy"] == pytest.approx([0.85, 0.95])  # the validation-selected seed's interval
    assert seeds["calibration"]["reference"] == "gold"

    large = _row(report, "student", "qwen2.5-1.5b")
    assert large["n_seeds"] == 1 and large["std"]["accuracy"] is None
    assert large["metrics"]["accuracy"] == pytest.approx(0.91)
    gold = _row(report, "student", "gold labels")
    assert gold["labels"] == "gold" and gold["metrics"]["accuracy"] == pytest.approx(0.93)
    zero = _row(report, "zero-shot")
    assert zero["metrics"]["accuracy"] == pytest.approx(0.40) and zero["run_ids"] == ["zero-shot-qwen2.5-0.5b"]
    tfidf = _row(report, "tfidf")
    assert tfidf["metrics"]["macro_f1"] == pytest.approx(0.78) and tfidf["C"] == 3.0
    cascade = _row(report, "cascade")
    assert cascade["metrics"]["agreement"] == pytest.approx(0.975)
    assert cascade["threshold"] == THRESHOLD and cascade["escalation_rate"] == 0.5
    assert cascade["ci"]["accuracy"] == [0.90, 0.97]
    assert report["quality"]["test_scorings"] == 2


def test_operating_point_and_pngs(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    out = tmp_path / "out"
    op = build_report(spec, out_dir=out, store=store)["operating_point"]
    assert op["reference"] == "teacher" and op["metric"] == "agreement" and op["target"] == 0.97
    assert op["threshold"] == THRESHOLD and not op["always_escalate"]
    assert op["escalation_rate"] == {"valid": 0.45, "test": 0.5}
    assert op["target_met_on_test"] is True
    target = op["cascade_minus_teacher"]["target_metric"]
    assert target["metric"] == "agreement" and target["source"] == "bootstrap"
    assert target["point"] == pytest.approx(-0.025)
    assert target["ci"] == pytest.approx([-0.05, 0.0])
    gold = op["cascade_minus_teacher"]["gold_metric"]
    assert gold["metric"] == "accuracy" and gold["point"] == pytest.approx(-0.01)
    assert gold["ci"] == pytest.approx([-0.03, 0.01])

    assert op["curve_png"] == "threshold_curve.png"
    assert op["reliability_png"] == ["reliability_test.png"]
    assert not (out / "reliability_valid.png").exists()  # only the files the test evaluation lists
    source = paths.eval_dir(TASK) / SELECTED
    for name in ("threshold_curve.png", "reliability_test.png"):
        copied = (out / name).read_bytes()
        assert copied == (source / name).read_bytes()
        assert b"Software" not in copied


# -- cost and latency --------------------------------------------------------------------------------------------


def test_cascade_latency_and_cost_composed_per_request(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    cost = build_report(spec, out_dir=tmp_path / "out", store=store)["cost_latency"]

    teacher = cost["teacher"]
    assert teacher["usd_per_request"] == pytest.approx(2.5e-4)
    assert teacher["usd_per_1k"] == pytest.approx(0.25)
    assert teacher["latency_ms"]["p50"] == pytest.approx(550.0)
    assert teacher["latency_ms"]["p95"] == pytest.approx(770.0)
    assert teacher["record_sources"] == {"cache": 2, "recording": 2}
    assert teacher["concurrency"] == 8 and teacher["recorded"] is True

    student = cost["student"]
    assert student["source"] == "eval" and student["flagged"] is True
    assert student["latency_ms"]["p50"] == 65.0 and student["latency_ms"]["p95"] == 78.5
    assert student["usd_per_request"] == pytest.approx(0.02 * 65 / 3_600_000 * 0.30)

    # q2 (0.40) and q4 (0.20) escalate at t = 0.5: 50, 60 + 500, 70, 80 + 800 ms.
    cascade = cost["cascade"]
    assert cascade["n"] == 4 and cascade["escalation_rate"] == 0.5
    assert cascade["latency_ms"]["p50"] == pytest.approx(315.0)
    assert cascade["latency_ms"]["p95"] == pytest.approx(832.0)
    assert cascade["latency_ms"]["mean"] == pytest.approx(390.0)
    energy = 0.02 * (50 + 60 + 70 + 80) / 3_600_000 * 0.30
    assert cascade["usd_per_request"] == pytest.approx((2e-4 + 4e-4 + energy) / 4)
    assert cost["assumptions"]["text"] == "local cost = 20 W x wall time x $0.3/kWh; hardware amortisation off"


def test_amortisation_changes_local_costs(home: Path, tmp_path: Path) -> None:
    spec = make_spec({"hardware_usd": 1800, "amortisation_hours": 9000})
    store = make_workspace(spec)
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    per_hour = 0.02 * 0.30 + 1800 / 9000
    assert report["cost_latency"]["student"]["usd_per_request"] == pytest.approx(per_hour * 65 / 3_600_000)
    assert report["break_even"]["training_usd"] == pytest.approx(per_hour * WALL_SECONDS / 3600)
    assert "hardware amortisation $1800 / 9000 h" in report["assumptions"]["text"]
    assert report["assumptions"]["amortisation_enabled"] is True


def test_student_latency_from_bench_and_cross_check(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec, bench=True)
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    cost = report["cost_latency"]
    assert cost["student"]["source"] == "bench" and cost["student"]["flagged"] is False
    assert cost["student"]["latency_ms"]["p50"] == 50.0  # the other run's newer bench is not used
    # bench student latency per request: 40, 45 + 500, 55, 60 + 800 ms
    assert cost["cascade"]["latency_ms"]["p50"] == pytest.approx(300.0)
    assert cost["cascade"]["latency_ms"]["p95"] == pytest.approx(812.75)
    assert cost["cascade"]["student_latency_source"] == "bench"
    bench = report["live_bench"]
    assert bench["student_only"]["file"] == "bench_20260926T120000Z.json"
    assert bench["cascade"]["spend_usd"] == 0.0123
    check = bench["cross_check"]["cascade"]
    assert check["composed_p50"] == pytest.approx(300.0) and check["measured_p50"] == 320.0
    assert check["ratio_p50"] == pytest.approx(320 / 300)
    assert check["test_escalation_rate"] == 0.5 and check["measured_escalation_rate"] == 0.48
    assert bench["cross_check"]["student_only"]["eval_p50"] == 65.0


# -- break-even --------------------------------------------------------------------------------------------------


def test_break_even_volume(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    be = report["break_even"]
    assert report["labelling_cost"]["source"] == "ledger"
    assert report["labelling_cost"]["by_phase"] == {"curate-label": 0.012, "demo-capture": 0.003}
    assert be["labelling_usd"] == pytest.approx(0.015)  # bake-off and serve spend are not labelling
    assert be["training_usd"] == pytest.approx(0.02 * WALL_SECONDS / 3600 * 0.30)
    energy = 0.02 * 260 / 3_600_000 * 0.30
    savings = 2.5e-4 - (6e-4 + energy) / 4
    assert be["savings_usd_per_request"] == pytest.approx(savings)
    assert be["volume_exact"] == pytest.approx((0.015 + 0.001) / savings)
    assert be["volume"] == 161 and be["requests"] == 161  # "requests" is the key the CLI and the demo read
    assert be["reason"] is None and be["lower_bound"] is False and be["notes"] == []


def test_break_even_is_null_without_savings(home: Path, tmp_path: Path) -> None:
    spec = make_spec({"local_watts": 5_000_000})
    store = make_workspace(spec)
    be = build_report(spec, out_dir=tmp_path / "out", store=store)["break_even"]
    assert be["savings_usd_per_request"] < 0
    assert be["volume"] is None and be["volume_exact"] is None
    assert "saves nothing" in be["reason"]


def test_labelling_cost_fallbacks(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec, ledger=False)
    report = build_report(spec, out_dir=tmp_path / "a", store=store)
    assert report["labelling_cost"] == {"usd": 0.5, "source": "curate_stats.json"}
    assert report["spend"]["ledger"] is False

    stats_path = paths.data_dir(TASK) / "curate_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["labelling"]["cost_usd"] = 0.0
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    report = build_report(spec, out_dir=tmp_path / "b", store=store)
    labelling = report["labelling_cost"]
    # recorded usage.cost of every curated example's request: test 1e-3, train 2 x 5e-4, valid 6e-4
    assert labelling["source"] == "recorded"
    assert labelling["usd"] == pytest.approx(1e-3 + 1e-3 + 6e-4)
    assert labelling["n_requests"] == 7 and labelling["n_priced"] == 7


# -- served traffic ----------------------------------------------------------------------------------------------


def _served(store: Store, route: str, reason: str | None, *, ts: float | None = None) -> None:
    teacher = route != "student"
    store.add_served(
        task=TASK,
        route=route,
        reason=reason,
        confidence=None if reason == "unsupported" else (0.3 if teacher else 0.9),
        student_ms=None if reason == "unsupported" else 50.0,
        teacher_ms=500.0 if teacher else None,
        total_ms=560.0 if teacher else 55.0,
        teacher_cost_usd=2e-4 if teacher else None,
        teacher_mode="live" if teacher else None,
        status=200,
        **({"ts": ts} if ts is not None else {}),
    )


def test_serve_log_figures_and_drift(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    for _ in range(7):
        _served(store, "student", None)
    _served(store, "teacher", "low_confidence")
    _served(store, "teacher", "low_confidence")
    _served(store, "teacher", "unsupported")
    _served(store, "teacher", "low_confidence", ts=time.time() - 100_000)  # outside the window
    report = build_report(spec, from_serve_log=True, since_s=3600, out_dir=tmp_path / "out", store=store)
    log = report["serve_log"]
    assert log["requests"] == 10
    assert log["route_counts"] == {"student": 7, "teacher": 3}
    assert log["route_shares"] == {"student": pytest.approx(0.7), "teacher": pytest.approx(0.3)}
    assert log["escalation_rate"] == pytest.approx(0.3)
    assert log["confidence_escalation_rate"] == pytest.approx(2 / 9)
    assert log["latency_ms_by_route"]["teacher"]["p50"] == pytest.approx(560.0)
    assert log["latency_ms_by_route"]["student"]["p95"] == pytest.approx(55.0)
    assert log["teacher_spend_usd"] == pytest.approx(6e-4)
    energy = 9 * 0.02 * 50 / 3_600_000 * 0.30
    assert log["usd_per_1k"] == pytest.approx((6e-4 + energy) / 10 * 1000)
    drift = log["drift"]
    assert drift["expected_escalation_rate"] == 0.5 and drift["possible_drift"] is False
    assert report["command"].startswith(f"taskdistill report --task {TASK} --from-serve-log --since 1h")

    for _ in range(40):
        _served(store, "student", None)
    log = build_report(spec, from_serve_log=True, out_dir=tmp_path / "out2", store=store)["serve_log"]
    assert log["requests"] == 51 and log["drift"]["possible_drift"] is True
    assert "drifted" in log["drift"]["note"]


# -- files -------------------------------------------------------------------------------------------------------


def test_report_files_default_location_and_no_absolute_paths(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec()
    store = make_workspace(spec, bench=True)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    report = build_report(spec, store=store, from_serve_log=True, command="taskdistill report --task intents")
    out = cwd / "reports" / TASK
    text_json = (out / "report.json").read_text(encoding="utf-8")
    text_md = (out / "report.md").read_text(encoding="utf-8")
    assert json.loads(text_json) == report
    assert (out / "threshold_curve.png").is_file()
    for text in (text_json, text_md):
        for root in {str(tmp_path), str(tmp_path.resolve()), str(home), str(Path.home())}:
            assert root not in text
    assert report["command"] == "taskdistill report --task intents"
    assert report["hardware"] and report["date"]


def test_training_dataset_and_spend(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    training = {row["run_id"]: row for row in report["training"]}
    assert set(training) == {SELECTED, "qwen2.5-1.5b-full-s13"}
    row = training[SELECTED]
    assert row["examples"] == 2 and row["dropped_by_length_filter"] == 2
    assert row["iterations"] == 100 and row["epochs"] == 2.0
    assert row["wall_minutes"] == pytest.approx(10.0)
    assert row["peak_memory_gb"] == 3.2 and row["tokens_per_second"] == 1500.0 and row["adapter_mb"] == 17.5
    assert report["dataset"]["splits"] == {"train": 2, "valid": 1, "test": 4}
    assert report["spend"]["task_total"] == pytest.approx(0.165)
    assert report["spend"]["task_by_phase"]["bakeoff"] == pytest.approx(0.1)
    assert report["selected_run"]["run_id"] == SELECTED


def test_markdown_formatting(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec, bench=True)
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert md == render_markdown(report)
    assert md.startswith(f"# taskdistill report: {TASK}\n")
    assert "88.0% ± 2.0 [85.0, 95.0]" in md  # seed mean ± std with the selected seed's interval
    assert "95.0% [92.0, 98.0]" in md
    assert "| reference |" in md or "reference" in md
    assert "0.780 ± 0.020" in md  # AUROC is not a percentage
    assert "6.0% ± 1.0 [3.0, 8.0]" in md  # ECE with the selected seed's bootstrap interval
    assert "| valid | primary (chosen) | 0.810 | 0.800 | 90.0% |" in md
    assert "-2.5 pts [-5.0, +0.0]" in md
    assert "![Cascade quality against escalation rate](threshold_curve.png)" in md
    assert "| Teacher only | $0.250 | 550 | 770 |" in md
    assert "local cost = 20 W x wall time x $0.3/kWh; hardware amortisation off" in md
    assert "**161 requests**" in md
    assert "## Live bench" in md and "## Training" in md


def test_render_helpers() -> None:
    assert pct(0.8766) == "87.7%" and pct(None) == "—"
    assert interval([0.8512, 0.9]) == "[85.1, 90.0]"
    assert interval([0.81, 0.9], "auroc") == "[0.810, 0.900]"
    assert interval(None) == ""
    assert usd(0.25) == "$0.250" and usd(1.5e-4) == "$0.000150" and usd(12.346) == "$12.35" and usd(0) == "$0"


def test_report_needs_an_evaluation(home: Path, tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="taskdistill eval"):
        build_report(make_spec(), out_dir=tmp_path / "out", store=Store())


def test_extraction_columns_cluster_intervals_and_breakdowns(home: Path, tmp_path: Path) -> None:
    spec = TaskSpec.model_validate(
        {
            "task": "docs",
            "type": "extraction",
            "schema_file": "schema.json",
            "teacher": {"model": "vendor/model-a"},
            "student": {"system_prompt": "Extract the fields as JSON."},
            "cascade": {"target": 0.95},
        }
    )
    spec.teacher_prompt = "Extract the invoice fields.\n"
    spec.json_schema = {"type": "object", "properties": {"vendor_name": {"type": "string"}}}
    metrics = {"json_validity": 1.0, "field_micro_f1": 0.9, "field_exact_match": 0.85, "doc_exact_match": 0.6}
    stats = {"student.field_micro_f1": _stat(0.9, 0.85, 0.94), "cascade.field_micro_f1": _stat(0.95, 0.9, 0.98)}
    cluster = {"student.field_micro_f1": _stat(0.9, 0.7, 0.97), "cascade.field_micro_f1": _stat(0.95, 0.8, 0.99)}
    diff = {"a": "cascade.field_micro_f1", "b": "teacher.field_micro_f1", **_stat(-0.02, -0.04, 0.0)}
    ev = {
        "task": "docs",
        "task_type": "extraction",
        "run_id": "qwen2.5-0.5b-full-s13",
        "n": 6,
        "run": {"base_model": BASE_05, "labels": "teacher", "seed": 13, "profile": "full", "backend": "mlx"},
        "systems": {
            "student": {"metrics": {**metrics, "agreement": 0.93}, "calibration": _calibration(0.04, 0.9)},
            "teacher": {"metrics": {**metrics, "field_micro_f1": 0.97}},
            "cascade": {"metrics": {**metrics, "field_micro_f1": 0.95, "agreement": 0.97}, "threshold": 0.7},
        },
        "threshold": {"threshold": 0.7, "escalation_rate": 0.3, "metric": "field_f1", "reference": "gold"},
        "bootstrap": {
            "paired": {"stats": stats, "diffs": {diff_name("cascade.field_micro_f1", "teacher.field_micro_f1"): diff}},
            "cluster": {"n_groups": 6, "stats": cluster, "diffs": {}},
        },
        "per_group": {
            "email-11": {"n": 3, "student": {"field_micro_f1": 0.8}, "teacher": {"field_micro_f1": 0.9}},
            "layout-11": {"n": 3, "student": {"field_micro_f1": 1.0}, "teacher": {"field_micro_f1": 1.0}},
        },
        "per_field": {
            "fields": ["vendor_name"],
            "vs_gold": {"student": {"vendor_name": 0.9}, "teacher": {"vendor_name": 1.0}},
            "vs_teacher": {"student": {"vendor_name": 0.95}},
        },
        "per_trait": {"eu_number_format": {"n": 2, "student": {"field_micro_f1": 0.75}}},
    }
    target = paths.eval_dir("docs") / "qwen2.5-0.5b-full-s13" / "eval_test.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(ev), encoding="utf-8")
    report = build_report(spec, out_dir=tmp_path / "out", store=Store())
    assert report["quality"]["columns"] == [
        "json_validity", "field_micro_f1", "field_exact_match", "doc_exact_match", "agreement",
    ]  # fmt: skip
    student = _row(report, "student")
    assert student["ci"]["field_micro_f1"] == [0.85, 0.94]
    assert student["ci_cluster"]["field_micro_f1"] == [0.7, 0.97]
    assert report["quality"]["cluster_bootstrap"] and report["quality"]["n_groups"] == 6
    op = report["operating_point"]
    assert op["metric"] == "field_f1" and op["cascade_minus_teacher"]["target_metric"]["point"] == pytest.approx(-0.02)
    assert op["cascade_minus_teacher"]["gold_metric"]["metric"] == "field_micro_f1"
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Field micro-F1" in md and "Cluster bootstrap over 6 groups" in md
    assert "<summary>Per-group (template) scores</summary>" in md and "| email-11 | 3 | 80.0% |" in md
    assert "| Group | n | Field micro-F1 |" in md and "teacher:" in md
    assert "| `vendor_name` | 90.0% | 100.0% |" in md and "Against the teacher:" in md
    assert "| eu_number_format | 2 | 75.0% |" in md
    assert "<summary>Per-trait breakdown</summary>" in md
    assert "No ledger in this workspace" in md


# -- regressions -------------------------------------------------------------------------------------------------


def _patch_selected_eval(change: Any) -> None:
    path = paths.eval_dir(TASK) / SELECTED / "eval_test.json"
    ev = json.loads(path.read_text(encoding="utf-8"))
    change(ev)
    path.write_text(json.dumps(ev), encoding="utf-8")


def test_target_metric_follows_the_reference(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)

    # accuracy against the teacher is the eval's agreement, not the gold accuracy difference (-1.0 pts)
    _patch_selected_eval(lambda ev: ev["threshold"].update(metric="accuracy"))
    op = build_report(spec, out_dir=tmp_path / "a", store=store)["operating_point"]
    target = op["cascade_minus_teacher"]["target_metric"]
    assert op["metric"] == "accuracy" and op["metric_key"] == "agreement"
    assert target["metric"] == "agreement" and target["target_metric"] == "accuracy"
    assert target["reference"] == "teacher" and target["source"] == "bootstrap"
    assert target["point"] == pytest.approx(-0.025) and target["ci"] == pytest.approx([-0.05, 0.0])
    assert op["cascade_minus_teacher"]["gold_metric"]["point"] == pytest.approx(-0.01)

    # macro-F1 against the teacher has no bootstrap difference: the test operating point, without an interval
    def macro(ev: dict[str, Any]) -> None:
        ev["threshold"]["metric"] = "macro_f1"
        ev["operating_point"]["test"].update(quality=0.96, teacher_quality=0.99)

    _patch_selected_eval(macro)
    report = build_report(spec, out_dir=tmp_path / "b", store=store)
    target = report["operating_point"]["cascade_minus_teacher"]["target_metric"]
    assert target["point"] == pytest.approx(-0.03) and target["ci"] is None
    assert "operating point" in target["source"]
    assert any("without a confidence interval" in note for note in report["operating_point"]["notes"])
    md = (tmp_path / "b" / "report.md").read_text(encoding="utf-8")
    assert "| Cascade − teacher, Macro-F1 against the teacher (target) | -3.0 pts |" in md

    # an eval without the agreement difference: cascade agreement - 1 from the bootstrap stat
    def no_diff(ev: dict[str, Any]) -> None:
        ev["threshold"]["metric"] = "agreement"
        del ev["bootstrap"]["paired"]["diffs"][diff_name("cascade.agreement", "teacher.agreement")]

    _patch_selected_eval(no_diff)
    target = build_report(spec, out_dir=tmp_path / "c", store=store)["operating_point"]["cascade_minus_teacher"]
    assert target["target_metric"]["source"].startswith("bootstrap (teacher agreement is 1")
    assert target["target_metric"]["point"] == pytest.approx(-0.025)
    assert target["target_metric"]["ci"] == pytest.approx([-0.05, 0.0])

    # against gold, the metric's own difference
    _patch_selected_eval(lambda ev: ev["threshold"].update(metric="accuracy", reference="gold"))
    target = build_report(spec, out_dir=tmp_path / "d", store=store)["operating_point"]["cascade_minus_teacher"]
    assert target["target_metric"]["metric"] == "accuracy"
    assert target["target_metric"]["point"] == pytest.approx(-0.01)


def test_commands_carry_no_absolute_path(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = make_spec()
    store = make_workspace(spec, bench=True)
    _patch_selected_eval(lambda ev: ev.update(command=f"taskdistill eval --task {TASK} --out {tmp_path / 'evals'}"))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    roots = {str(tmp_path), str(tmp_path.resolve()), str(home), str(Path.home())}

    out = tmp_path / "elsewhere" / "rep"
    report = build_report(spec, store=store, out_dir=out, command=f"taskdistill report --task {TASK} --out {out}")
    for text in ((out / "report.json").read_text(encoding="utf-8"), (out / "report.md").read_text(encoding="utf-8")):
        for root in roots:
            assert root not in text
    assert report["command"] == f"taskdistill report --task {TASK} --out rep"
    selected = next(p for p in report["quality"]["provenance"] if p["run_id"] == SELECTED)
    assert selected["command"] == f"taskdistill eval --task {TASK} --out evals"

    inside = cwd / "reports" / "x"
    report = build_report(spec, store=store, out_dir=inside, command=f"taskdistill report --task {TASK} --out={inside}")
    assert report["command"] == f"taskdistill report --task {TASK} --out=reports/x"
    report = build_report(spec, store=store, out_dir=out, command=f"taskdistill report --task {TASK} --out {home}/r")
    assert report["command"] == f"taskdistill report --task {TASK} --out $TASKDISTILL_HOME/r"
    report = build_report(spec, store=store, out_dir=inside)
    assert report["command"] == f"taskdistill report --task {TASK} --out reports/x"


def test_escalated_requests_without_teacher_records_stay_in_the_cascade(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec, unrecorded=("q4",))  # q4 escalates (0.20 < 0.5) and has no teacher record
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    cost = report["cost_latency"]
    assert cost["teacher"]["n"] == 3
    cascade = cost["cascade"]
    assert cascade["n"] == 4 and cascade["n_escalated"] == 2 and cascade["escalation_rate"] == 0.5
    assert cascade["imputed"] == {"teacher_cost": 1, "teacher_latency": 1}
    # q4 gets the mean recorded teacher latency (400, 500, 600 ms -> 500) and cost ($1e-4..3e-4 -> $2e-4)
    assert cascade["latency_ms"]["mean"] == pytest.approx((50 + 560 + 70 + 580) / 4)
    assert cascade["usd_per_request"] == pytest.approx(local_cost_usd(65.0, spec.cost) + (2e-4 + 2e-4) / 4)
    assert any("mean recorded teacher cost" in note for note in cost["notes"])
    assert any("no recorded teacher cost" in note for note in report["break_even"]["notes"])
    assert "1 teacher records imputed" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")

    # nothing recorded at all: the cascade cost cannot be composed and the break-even says why
    workspace_recording(TASK).unlink()
    paths.cache_path().unlink()
    report = build_report(spec, out_dir=tmp_path / "out2", store=store)
    cascade = report["cost_latency"]["cascade"]
    assert cascade["n"] == 4 and cascade["escalation_rate"] == 0.5
    assert cascade["usd_per_request"] is None and cascade["latency_ms"]["n"] == 0
    assert cascade["incomplete"] and report["break_even"]["volume"] is None
    assert report["break_even"]["reason"] == "the teacher or cascade cost per request is unknown"


def test_cascade_covers_the_whole_test_split_with_a_partial_bench(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    _write_bench(TASK, "bench_20260926T120000Z.json", mode="student_only", run_id=SELECTED, queries=("q1", "q2"))
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    cost = report["cost_latency"]
    assert cost["student"]["source"] == "bench"
    cascade = cost["cascade"]
    # escalation and cost over all four test requests, as on the operating point
    assert cascade["n"] == 4 and cascade["escalation_rate"] == report["operating_point"]["escalation_rate"]["test"]
    assert cascade["usd_per_request"] == pytest.approx(local_cost_usd(50.0, spec.cost) + (2e-4 + 4e-4) / 4)
    # latency only where the bench timed the student: q1 40 ms, q2 45 + 500 ms
    assert cascade["latency_n"] == 2 and cascade["latency_escalation_rate"] == 0.5
    assert cascade["latency_ms"]["p50"] == pytest.approx((40 + 545) / 2)
    assert any("composed over the 2 benched test requests" in note for note in cost["notes"])
    savings = 2.5e-4 - cascade["usd_per_request"]
    assert report["break_even"]["volume_exact"] == pytest.approx((0.015 + 0.001) / savings)
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "50.0% escalated; latency over 2 benched test requests (50.0% escalated)" in md


def test_live_bench_of_another_run_is_not_cross_checked(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    _write_bench(TASK, "bench_20260926T130000Z.json", mode="cascade", run_id="other-run")
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    bench = report["live_bench"]
    assert bench["cascade"]["matches_selected_run"] is False
    assert bench["cross_check"] == {}
    assert any("not the selected run" in note for note in bench["notes"])
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "cascade (run `other-run`, not the selected run)" in md and "Cross-check" not in md


def test_provenance_of_evaluations_and_benches(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec, bench=True)
    report = build_report(spec, out_dir=tmp_path / "out", store=store)
    selected = next(p for p in report["quality"]["provenance"] if p["run_id"] == SELECTED)
    assert selected["date"] == "2026-09-26T10:00:00+00:00" and selected["hardware"]["cpu"] == "Apple M5"
    assert selected["command"] == f"taskdistill eval --task {TASK} --run {SELECTED}"
    assert {p["run_id"] for p in report["quality"]["provenance"]} == set(EVALS)
    assert report["operating_point"]["provenance"]["file"] == f"eval/{SELECTED}/eval_test.json"
    student = report["cost_latency"]["student"]["provenance"]
    assert student["file"] == "bench/bench_20260926T120000Z.json"
    assert student["machine_state"]["load_average"] == [1.2, 1.1, 1.0]
    assert report["live_bench"]["cascade"]["hardware"]["model"] == "Mac17,4"
    assert report["cost_latency"]["teacher"]["recorded_between"] == ["2026-09-21", "2026-09-21"]
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "<summary>Evaluation dates, hardware and commands</summary>" in md
    assert "| Evaluated | 2026-09-26T10:00:00+00:00 on Mac17,4, Apple M5, 24 GB, `taskdistill eval" in md
    assert "load average 1.20 / 1.10 / 1.00 before timing" in md
    assert "- Report built: " in md


def test_serve_log_counts_failed_escalations(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    for _ in range(6):
        _served(store, "student", None)
    _served(store, "teacher", "low_confidence")
    _served(store, "student-fallback", "low_confidence")  # the teacher failed, the student answered
    _served(store, "teacher", "input_unparsed")
    store.add_served(
        task=TASK,
        route="error",
        reason="low_confidence",
        confidence=0.3,
        student_ms=50.0,
        total_ms=900.0,
        teacher_mode="live",
        status=502,
    )  # on_teacher_error: error
    store.add_served(task=TASK, route="error", reason=None, total_ms=5.0, status=503)  # model still loading
    log = build_report(spec, from_serve_log=True, out_dir=tmp_path / "out", store=store)["serve_log"]
    assert log["requests"] == 11
    assert log["escalations"] == 4 and log["escalation_rate"] == pytest.approx(4 / 11)
    assert log["failed_escalations"] == 2
    assert log["failed_escalation_routes"] == {"error": 1, "student-fallback": 1}
    # the student's own decisions: 6 answers and 3 low-confidence escalations, whatever the teacher did next
    assert log["confidence_decisions"] == 9
    assert log["confidence_escalation_rate"] == pytest.approx(3 / 9)
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "| Failed escalations (teacher error or replay miss) | 2 |" in md


def test_break_even_without_labelling_cost_is_a_lower_bound(home: Path) -> None:
    spec = make_spec()
    run_dir = paths.runs_dir(TASK) / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_log.json").write_text(json.dumps({"wall_seconds": 3600.0}), encoding="utf-8")
    cost_latency = {"teacher": {"usd_per_request": 2e-4}, "cascade": {"usd_per_request": 1e-4}}
    be = build_break_even(spec, "r1", cost_latency, {"usd": 0.0, "source": "none found", "n_requests": 0})
    assert be["training_usd"] == pytest.approx(0.006)
    assert be["volume"] == be["requests"] == 60 and be["lower_bound"] is True and be["reason"] is None
    assert any("lower bound" in note for note in be["notes"])
    md = render_markdown({"task": TASK, "break_even": be})
    assert "**at least 60 requests**" in md and "Note: no teacher labelling cost was found" in md

    be = build_break_even(spec, None, cost_latency, {"usd": 0.01, "source": "ledger"})
    assert be["volume"] == 100 and be["lower_bound"] is True  # no train log: training energy left out
    assert any("train_log.json" in note for note in be["notes"])


def test_report_for_another_run(home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    store = make_workspace(spec)
    default = build_report(spec, out_dir=tmp_path / "a", store=store)
    other = next(
        r["run_ids"][0] for r in default["quality"]["rows"] if r["system"] == "student" and "1.5b" in r["name"]
    )
    report = build_report(spec, out_dir=tmp_path / "b", store=store, run_id=other)
    assert report["selected_run"]["run_id"] == other
    assert "chosen with --run" in report["selected_run"]["reason"]
    assert any("--run" in note for note in report["notes"])
    with pytest.raises(ReportError, match="no test evaluation"):
        build_report(spec, out_dir=tmp_path / "c", store=store, run_id="no-such-run")
