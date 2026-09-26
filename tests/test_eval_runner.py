from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from sklearn.metrics import f1_score

from taskdistill import paths
from taskdistill.backends.fake import FakeBackend
from taskdistill.backends.types import Generation
from taskdistill.config import TaskSpec
from taskdistill.curate.extract import input_hash
from taskdistill.evaluate import data as eval_data
from taskdistill.evaluate import predictions, runner
from taskdistill.evaluate.bootstrap import diff_name
from taskdistill.evaluate.calibration import IsotonicCalibrator, calibration_report
from taskdistill.evaluate.splits import TestSplit, ValidationSplit
from taskdistill.evaluate.threshold import select_threshold
from taskdistill.evaluate.zero_shot import LABEL_INSTRUCTION, zero_shot_messages
from taskdistill.serve.runner import resolve_threshold
from taskdistill.tasks.extraction import canonical_output

LABELS = ["card_arrival", "cash_withdrawal", "exchange_rate"]
CA, CW, ER = LABELS
SMALL, LARGE = "fake/qwen-0.5b", "fake/qwen-1.5b"
RUN = "qwen-0.5b-full-s13"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"vendor": {"type": ["string", "null"]}, "total": {"type": ["number", "null"]}},
    "required": ["vendor", "total"],
    "additionalProperties": False,
}

TRAIN = [
    ("when will my card arrive", CA),
    ("my new card has not arrived", CA),
    ("card delivery is late", CA),
    ("track my card delivery", CA),
    ("withdraw cash at the atm", CW),
    ("cash machine took my money", CW),
    ("atm withdrawal limit", CW),
    ("how to take out cash", CW),
    ("what is the exchange rate", ER),
    ("euro exchange rate today", ER),
    ("rate for converting dollars", ER),
    ("currency exchange fee rate", ER),
]
# (input, teacher = gold, student answer, student confidence); the student is wrong only at low confidence.
VALID = [
    ("is my card on its way", CA, CA, 0.95),
    ("card not delivered yet", CA, CA, 0.9),
    ("where is the card i ordered", CA, CA, 0.85),
    ("card arrival date", CA, CA, 0.8),
    ("atm gave no cash", CW, CW, 0.75),
    ("cash withdrawal abroad", CW, ER, 0.3),
    ("take money from the atm", CW, CW, 0.7),
    ("withdraw cash limit", CW, CW, 0.65),
    ("exchange rate for pounds", ER, ER, 0.6),
    ("which rate do you apply", ER, ER, 0.55),
    ("dollar to euro rate", ER, CW, 0.4),
    ("rate used for conversion", ER, ER, 0.5),
]
# (input, teacher, gold, student answer, student confidence)
TEST = [
    ("where is my new card", CA, CA, CA, 0.92),
    ("card still not here", CA, CA, CA, 0.88),
    ("how do i get cash out", CW, CW, CW, 0.81),
    ("atm cash limit today", CW, CW, CW, 0.77),
    ("what rate do you use for euros", ER, ER, ER, 0.66),
    ("dollar conversion rate please", ER, ER, ER, 0.58),
    ("card delivery tracking", CW, CA, CW, 0.52),
    ("card fee for exchange", CA, ER, CA, 0.47),
    ("withdraw money abroad", CW, CW, CW, 0.45),
    ("exchange fee for pounds", ER, ER, CW, 0.35),
]


# -- fixtures and helpers ----------------------------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    return workspace


def classification_spec() -> TaskSpec:
    spec = TaskSpec.model_validate(
        {
            "task": "intents",
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": {"model": "vendor/teacher-model"},
            "student": {"system_prompt": "Classify the banking message.", "base_model": SMALL, "max_tokens": 8},
            "cascade": {"reference": "teacher", "metric": "agreement", "target": 0.9},
        }
    )
    spec.labels = list(LABELS)
    return spec


def extraction_spec() -> TaskSpec:
    spec = TaskSpec.model_validate(
        {
            "task": "bills",
            "type": "extraction",
            "schema_file": "schema.json",
            "teacher": {"model": "vendor/teacher-model"},
            "student": {"system_prompt": "Extract the fields as JSON.", "base_model": SMALL, "max_tokens": 64},
            "curate": {"split": {"group_by": "meta.template", "stratify": False}},
            "cascade": {"reference": "teacher", "metric": "agreement", "target": 0.9},
        }
    )
    spec.json_schema = SCHEMA
    return spec


def write_split(spec: TaskSpec, split: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """``rows``: dicts with input, teacher, gold and meta, written as curate writes them."""
    data_dir = paths.data_dir(spec.task)
    with (
        (data_dir / f"{split}.jsonl").open("w", encoding="utf-8") as data_fh,
        (data_dir / f"{split}.meta.jsonl").open("w", encoding="utf-8") as meta_fh,
    ):
        for row in rows:
            teacher = row["teacher"]
            completion = teacher if spec.type == "classification" else canonical_output(teacher, SCHEMA)
            messages = [
                {"role": "system", "content": spec.student.system_prompt},
                {"role": "user", "content": row["input"]},
                {"role": "assistant", "content": completion},
            ]
            data_fh.write(json.dumps({"messages": messages}) + "\n")
            meta = {
                "input_hash": input_hash(row["input"]),
                "gold": row.get("gold"),
                "teacher": teacher,
                "meta": row.get("meta", {}),
            }
            meta_fh.write(json.dumps(meta) + "\n")


def write_classification_data(spec: TaskSpec) -> None:
    write_split(spec, "train", [{"input": text, "teacher": label, "gold": label} for text, label in TRAIN])
    write_split(spec, "valid", [{"input": text, "teacher": label, "gold": label} for text, label, _, _ in VALID])
    write_split(spec, "test", [{"input": t, "teacher": teacher, "gold": gold} for t, teacher, gold, _, _ in TEST])


def make_run(
    spec: TaskSpec,
    run_id: str,
    *,
    base: str = SMALL,
    seed: int = 13,
    labels: str = "teacher",
    backend: str = "mlx",
) -> Path:
    run_dir = paths.runs_dir(spec.task) / run_id
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "adapter" / "adapters.safetensors").write_bytes(run_id.encode())
    log = {
        "run_id": run_id,
        "task": spec.task,
        "backend": backend,
        "base_model": base,
        "profile": "full",
        "seed": seed,
        "labels": labels,
    }
    (run_dir / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    return run_dir


def classification_script() -> dict[str, tuple[str, float]]:
    script = {text: (answer, conf) for text, _, answer, conf in VALID}
    script.update({text: (answer, conf) for text, _, _, answer, conf in TEST})
    return script


class RecordingBackend(FakeBackend):
    """A fake backend that also records the system prompt of every generation."""

    systems: list[str]

    def generate_with_scores(
        self, messages: list[dict[str, str]], constraint: Any = None, max_tokens: int = 256
    ) -> Generation:
        if not hasattr(self, "systems"):
            self.systems = []
        self.systems.append(messages[0]["content"])
        return super().generate_with_scores(messages, constraint, max_tokens)


class Factory:
    """``backend_factory`` that scripts answers per run (by adapter directory) or per base model (no adapter)."""

    def __init__(self, scripts: Mapping[str, Mapping[str, tuple[str, float]]]) -> None:
        self.scripts = {key: dict(script) for key, script in scripts.items()}
        self.calls: list[tuple[str, str]] = []
        self.backends: list[RecordingBackend] = []

    def __call__(self, base_model: str, adapter_path: str | None) -> RecordingBackend:
        key = Path(adapter_path).parent.name if adapter_path else f"base:{base_model}"
        self.calls.append((base_model, key))
        backend = RecordingBackend(answers=dict(self.scripts.get(key, {})), base_model=base_model)
        self.backends.append(backend)
        return backend

    @property
    def generations(self) -> list[str]:
        return [text for backend in self.backends for text in backend.calls]


def run(spec: TaskSpec, factory: Factory, **kwargs: Any) -> dict[str, Any]:
    logs: list[str] = []
    kwargs.setdefault("log", logs.append)
    return runner.run_eval(spec, backend_factory=factory, **kwargs)


@pytest.fixture
def fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "RESAMPLES", 50)


def walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)


# -- data and prompts --------------------------------------------------------------------------------------------------


def test_load_split_types_groups_and_traits(home: Path) -> None:
    spec = extraction_spec()
    rows: list[dict[str, Any]] = [
        {
            "input": "Invoice 1 from Bralvik Ltd",
            "teacher": {"vendor": "Bralvik Ltd", "total": 10.0},
            "gold": {"vendor": "Bralvik Ltd", "total": 10.0},
            "meta": {"template": "email-01", "group": "ignored", "traits": ["eu_number_format"]},
        },
        {"input": "Invoice 2", "teacher": {"vendor": None, "total": 2.5}, "gold": None, "meta": {}},
    ]
    write_split(spec, "valid", rows)
    write_split(spec, "test", rows)
    valid = eval_data.load_split(spec, "valid")
    test = eval_data.load_split(spec, "test")
    assert type(valid) is ValidationSplit
    assert type(test) is TestSplit
    first, second = valid.records
    assert first.id == input_hash("Invoice 1 from Bralvik Ltd")
    assert first.input == "Invoice 1 from Bralvik Ltd"
    assert first.group == "email-01"  # group_by wins over meta.group
    assert first.traits == ("eu_number_format",)
    assert first.teacher == {"vendor": "Bralvik Ltd", "total": 10.0}
    assert second.gold is None and second.group is None and second.traits == ()

    spec.curate.split.group_by = None
    assert eval_data.load_split(spec, "valid").records[0].group == "ignored"  # falls back to meta.group
    with pytest.raises(eval_data.EvalDataError, match="curate"):
        eval_data.load_split(spec.model_copy(update={"task": "missing"}), "valid")


def test_zero_shot_messages() -> None:
    spec = classification_spec()
    system, user = zero_shot_messages(spec, "where is my card")
    assert user == {"role": "user", "content": "where is my card"}
    lines = system["content"].splitlines()
    assert all(label in lines for label in LABELS)  # one label per line
    assert system["content"].startswith("Classify the banking message.")
    assert system["content"].endswith(LABEL_INSTRUCTION)

    extraction = zero_shot_messages(extraction_spec(), "Invoice text")[0]["content"]
    assert f"matching this JSON Schema: {json.dumps(SCHEMA)}. Output JSON only." in extraction


# -- the eval JSON -----------------------------------------------------------------------------------------------------


def test_classification_eval_json_follows_the_contract(home: Path) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    factory = Factory({RUN: classification_script()})
    result = run(spec, factory, run_id=RUN, command="taskdistill eval --task intents")

    top = {
        "task", "task_type", "run_id", "split", "n", "date", "command", "hardware", "run", "systems", "threshold",
        "operating_point", "confidence_comparison", "bootstrap", "per_group", "per_field", "per_trait",
        "test_scorings", "profile_fast",
    }  # fmt: skip
    assert top <= set(result)
    assert (result["task"], result["task_type"], result["run_id"], result["split"]) == (
        "intents",
        "classification",
        RUN,
        "test",
    )
    assert result["n"] == len(TEST)
    assert result["command"] == "taskdistill eval --task intents"
    assert result["run"] == {"base_model": SMALL, "labels": "teacher", "seed": 13, "profile": "full", "backend": "mlx"}
    assert result["profile_fast"] is False
    assert result["test_scorings"] == 1

    systems = result["systems"]
    assert set(systems) == {"student", "teacher", "cascade", "tfidf"}
    student = systems["student"]
    assert set(student) == {"metrics", "latency_ms", "calibration"}
    assert {"accuracy", "macro_f1", "agreement"} <= set(student["metrics"])
    assert set(student["latency_ms"]) == {"p50", "p95", "mean", "n"} and student["latency_ms"]["n"] == len(TEST)
    calibration = student["calibration"]
    assert set(calibration) >= {"raw", "isotonic", "isotonic_fit"}
    for kind in ("raw", "isotonic"):
        assert set(calibration[kind]) == {"vs_gold", "vs_teacher"}
        assert {"ece", "brier", "auroc", "bins"} <= set(calibration[kind]["vs_gold"])
    assert calibration["isotonic_fit"]["n_fit"] == len(VALID)
    assert calibration["isotonic_fit"]["reference"] == "gold"
    assert set(systems["cascade"]) == {"metrics", "threshold", "escalation_rate"}
    assert {"metrics", "C"} <= set(systems["tfidf"])
    assert {"accuracy", "macro_f1", "agreement"} <= set(systems["tfidf"]["metrics"])

    assert {"threshold", "escalation_rate", "quality", "met", "curve", "reference", "metric"} <= set(
        result["threshold"]
    )
    assert set(result["operating_point"]) == {"valid", "test"}
    assert {"escalation_rate", "quality", "target_value", "met"} <= set(result["operating_point"]["test"])
    comparison = result["confidence_comparison"]
    for split in ("valid", "test"):
        assert comparison[split]["chosen"] == "primary"
        for which in ("primary", "alternative"):
            assert {"auroc_vs_teacher", "auroc_vs_gold", "accuracy_vs_gold"} <= set(comparison[split][which])

    paired = result["bootstrap"]["paired"]
    assert (paired["resamples"], paired["seed"], paired["method"]) == (1000, 0, "paired")
    for system in ("student", "teacher", "cascade", "tfidf"):
        for metric in ("accuracy", "macro_f1", "agreement"):
            assert f"{system}.{metric}" in paired["stats"]
    for metric in ("accuracy", "macro_f1", "agreement"):
        assert diff_name(f"student.{metric}", f"teacher.{metric}") in paired["diffs"]
        assert diff_name(f"cascade.{metric}", f"teacher.{metric}") in paired["diffs"]
    assert result["bootstrap"]["cluster"] is None
    assert result["per_group"] is None and result["per_field"] is None and result["per_trait"] is None

    out = paths.eval_dir("intents") / RUN
    on_disk = json.loads((out / "eval_test.json").read_text(encoding="utf-8"))
    assert on_disk == result
    assert (out / "reliability_test.png").is_file()
    assert (out / "threshold_curve.png").is_file()
    threshold_file = json.loads((paths.task_home("intents") / "threshold.json").read_text(encoding="utf-8"))
    assert threshold_file["run_id"] == RUN
    assert threshold_file["threshold"] == pytest.approx(result["threshold"]["threshold"])


def test_metrics_student_teacher_cascade_by_hand(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    result = run(spec, Factory({RUN: classification_script()}), run_id=RUN)
    gold = [g for _, _, g, _, _ in TEST]
    teacher = [t for _, t, _, _, _ in TEST]
    student = [a for _, _, _, a, _ in TEST]
    systems = result["systems"]

    assert systems["teacher"]["metrics"]["accuracy"] == pytest.approx(0.8)  # teacher labels from the sidecar
    expected_f1 = f1_score(gold, teacher, labels=LABELS, average="macro", zero_division=0)
    assert systems["teacher"]["metrics"]["macro_f1"] == pytest.approx(expected_f1)
    assert systems["student"]["metrics"]["accuracy"] == pytest.approx(0.7)
    assert systems["student"]["metrics"]["agreement"] == pytest.approx(0.9)
    student_f1 = f1_score(gold, student, labels=LABELS, average="macro", zero_division=0)
    assert systems["student"]["metrics"]["macro_f1"] == pytest.approx(student_f1)

    # Threshold 0.4 on validation: only the last test record (0.35) escalates, and the teacher answers it.
    assert result["threshold"]["threshold"] == pytest.approx(0.4)
    assert systems["cascade"]["escalation_rate"] == pytest.approx(0.1)
    assert systems["cascade"]["metrics"]["agreement"] == pytest.approx(1.0)
    assert systems["cascade"]["metrics"]["accuracy"] == pytest.approx(0.8)
    paired = result["bootstrap"]["paired"]
    diff = paired["diffs"][diff_name("student.accuracy", "teacher.accuracy")]
    assert diff["point"] == pytest.approx(-0.1)
    assert result["operating_point"]["test"]["met"] is True


def test_choices_read_validation_only(home: Path, monkeypatch: pytest.MonkeyPatch, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN, seed=13)
    make_run(spec, "qwen-0.5b-full-s14", seed=14)
    seen: dict[str, list[Any]] = {}

    def spy(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            first = args[0] if name != "tune_baseline" else args[2]
            values = list(first.values()) if isinstance(first, Mapping) else [first]
            seen.setdefault(name, []).extend(values)
            return fn(*args, **kwargs)

        return wrapper

    for name in ("select_threshold", "fit_isotonic", "tune_baseline", "select_run", "score_runs", "choose_base_model"):
        monkeypatch.setattr(runner, name, spy(name, getattr(runner, name)))
    script = classification_script()
    result = run(spec, Factory({RUN: script, "qwen-0.5b-full-s14": script}), select=True)

    assert {"select_threshold", "fit_isotonic", "tune_baseline", "select_run", "score_runs"} <= set(seen)
    for name, splits in seen.items():
        assert splits and all(type(s) is ValidationSplit for s in splits), name
    assert all(r.confidence is not None for r in seen["select_threshold"][0].records)
    # The chosen threshold is a validation confidence; no test confidence is near 0.4.
    assert result["threshold"]["threshold"] == pytest.approx(0.4)
    assert all(abs(conf - 0.4) > 0.01 for *_, conf in TEST)


# -- selection ---------------------------------------------------------------------------------------------------------


def test_select_picks_the_best_seed_and_skips_gold_runs(
    home: Path, fast_bootstrap: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    good = {text: (label, 0.9) for text, label, _, _ in VALID}
    worse = dict(good, **{VALID[0][0]: (CW, 0.9), VALID[1][0]: (CW, 0.9)})
    medium = dict(good, **{VALID[0][0]: (CW, 0.9)})
    make_run(spec, "qwen-0.5b-full-s13", seed=13)
    make_run(spec, "qwen-0.5b-full-s14", seed=14)
    make_run(spec, "qwen-0.5b-full-s15", seed=15)
    make_run(spec, "qwen-0.5b-full-s13-gold", seed=13, labels="gold")
    factory = Factory(
        {
            "qwen-0.5b-full-s13": worse,
            "qwen-0.5b-full-s14": good,
            "qwen-0.5b-full-s15": medium,
            "qwen-0.5b-full-s13-gold": good,
        }
    )
    closed_at: list[int] = []
    close = predictions.LazyBackend.close

    def recording_close(self: predictions.LazyBackend) -> None:
        if self.loaded:
            closed_at.append(len(factory.calls))
        close(self)

    monkeypatch.setattr(predictions.LazyBackend, "close", recording_close)
    result = run(spec, factory, select=True, split="valid")
    assert closed_at == [1, 2, 3]  # each candidate's model is freed before the next one loads

    selected = json.loads((paths.task_home("intents") / "selected_run.json").read_text(encoding="utf-8"))
    assert {"run_id", "reason", "date", "rule", "candidates"} <= set(selected)
    assert selected["run_id"] == "qwen-0.5b-full-s14" == result["run_id"]
    assert set(selected["candidates"]) == {"qwen-0.5b-full-s13", "qwen-0.5b-full-s14", "qwen-0.5b-full-s15"}
    for entry in selected["candidates"].values():
        assert {"valid_metric", "p95_ms", "base_model", "seed"} <= set(entry)
    assert selected["candidates"]["qwen-0.5b-full-s14"]["valid_metric"] == pytest.approx(1.0)
    assert selected["candidates"]["qwen-0.5b-full-s13"]["valid_metric"] == pytest.approx(10 / 12)
    assert all(key != "qwen-0.5b-full-s13-gold" for _, key in factory.calls)
    assert result["selection"]["run_id"] == "qwen-0.5b-full-s14"
    # One backend per run: the selected run's validation predictions are reused for its evaluation.
    assert sorted(key for _, key in factory.calls) == ["qwen-0.5b-full-s13", "qwen-0.5b-full-s14", "qwen-0.5b-full-s15"]


def write_cached_valid(spec: TaskSpec, run_id: str, answers: Sequence[str], latency_ms: float) -> None:
    """Pre-computed validation predictions with a fixed latency (the cache the runner would have written)."""
    run_dir = paths.runs_dir(spec.task) / run_id
    header = predictions.make_header(
        data_sha256=eval_data.data_sha256(spec.task, "valid"),
        adapter_sha256=predictions.adapter_sha256(run_dir / "adapter"),
        alternatives=True,
        backend="mlx",
        settings_sha256=predictions.settings_sha256(spec),
    )
    lines = [json.dumps(header)]
    for record, answer in zip(eval_data.load_split(spec, "valid").records, answers, strict=True):
        row = {
            "id": record.id,
            "answer": answer,
            "value": answer,
            "confidence": 0.9,
            "alt_value": answer,
            "alt_confidence": 0.9,
            "latency_ms": latency_ms,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "field_confidences": None,
        }
        lines.append(json.dumps(row))
    (run_dir / "preds_valid.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("large_correct", "large_p95", "expected"),
    [
        (12, 20.0, "qwen-1.5b-full-s13"),  # +8.3 points at 2x the p95: the larger base
        (12, 40.0, "qwen-0.5b-full-s14"),  # +8.3 points but 4x the p95: the 3x guard keeps the smaller base
        (11, 12.0, "qwen-0.5b-full-s14"),  # no gain: the smaller base
    ],
)
def test_select_applies_the_base_model_rule(
    home: Path, fast_bootstrap: None, large_correct: int, large_p95: float, expected: str
) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    teacher = [label for _, label, _, _ in VALID]

    def answers(correct: int) -> list[str]:
        return [label if i < correct else (CW if label != CW else CA) for i, label in enumerate(teacher)]

    for run_id, base, seed, correct, latency in (
        ("qwen-0.5b-full-s13", SMALL, 13, 10, 10.0),
        ("qwen-0.5b-full-s14", SMALL, 14, 11, 10.0),
        ("qwen-1.5b-full-s13", LARGE, 13, large_correct, large_p95),
        ("qwen-1.5b-full-s13-gold", LARGE, 13, 12, 5.0),
    ):
        make_run(spec, run_id, base=base, seed=seed, labels="gold" if run_id.endswith("gold") else "teacher")
        write_cached_valid(spec, run_id, answers(correct), latency)
    factory = Factory({})
    result = run(spec, factory, select=True, split="valid")

    selected = json.loads((paths.task_home("intents") / "selected_run.json").read_text(encoding="utf-8"))
    assert selected["run_id"] == expected == result["run_id"]
    assert "qwen-1.5b-full-s13-gold" not in selected["candidates"]
    decision = selected["rule"]["decisions"][0]
    assert (decision["small_run"], decision["large_run"]) == ("qwen-0.5b-full-s14", "qwen-1.5b-full-s13")
    assert decision["latency_ratio"] == pytest.approx(large_p95 / 10.0)
    assert selected["candidates"]["qwen-1.5b-full-s13"]["p95_ms"] == pytest.approx(large_p95)
    assert factory.calls == []  # every prediction came from the cache


def test_missing_selected_run_is_a_clear_error(home: Path) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    with pytest.raises(runner.EvalError, match=r"selected_run\.json is missing"):
        run(spec, Factory({}))


def test_evaluates_the_run_named_in_selected_run(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    (paths.task_home("intents") / "selected_run.json").write_text(json.dumps({"run_id": RUN}), encoding="utf-8")
    result = run(spec, Factory({RUN: classification_script()}), split="valid")
    assert result["run_id"] == RUN and result["selected"] is True


# -- test access, cache, zero-shot -------------------------------------------------------------------------------------


def test_test_access_log_is_appended_only_for_the_test_split(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    factory = Factory({RUN: classification_script()})
    log_path = paths.task_home("intents") / "test_access_log.jsonl"

    on_valid = run(spec, factory, run_id=RUN, split="valid")
    assert not log_path.exists()
    assert on_valid["test_scorings"] == 0
    assert set(on_valid["operating_point"]) == {"valid"}
    assert set(on_valid["confidence_comparison"]) >= {"valid"} and "test" not in on_valid["confidence_comparison"]
    assert (paths.eval_dir("intents") / RUN / "eval_valid.json").is_file()

    run(spec, factory, run_id=RUN, command="first")
    second = run(spec, factory, run_id=RUN, command="second")
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [e["command"] for e in entries] == ["first", "second"]
    assert all(set(e) == {"date", "run_id", "command"} and e["run_id"] == RUN for e in entries)
    assert second["test_scorings"] == 2


def test_prediction_cache_is_reused_and_invalidated_by_data_changes(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    factory = Factory({RUN: classification_script()})

    first = run(spec, factory, run_id=RUN)
    assert len(factory.calls) == 1  # one backend for both splits
    generated = len(factory.generations)
    # Warm-up (3 per split) plus a constrained and a free generation per record.
    assert generated == 3 + 2 * len(VALID) + 3 + 2 * len(TEST)
    header = json.loads((paths.runs_dir("intents") / RUN / "preds_test.jsonl").read_text().splitlines()[0])
    assert header["type"] == "header" and header["alternatives"] is True and header["backend"] == "mlx"
    assert header["data_sha256"] == eval_data.data_sha256("intents", "test")

    second = run(spec, factory, run_id=RUN)
    assert len(factory.calls) == 1 and len(factory.generations) == generated  # nothing recomputed
    assert second["systems"]["student"]["metrics"] == first["systems"]["student"]["metrics"]

    changed = [(f"{text} please", teacher, gold, answer, conf) for text, teacher, gold, answer, conf in TEST]
    write_split(spec, "test", [{"input": t, "teacher": te, "gold": g} for t, te, g, _, _ in changed])
    factory.scripts[RUN].update({text: (answer, conf) for text, _, _, answer, conf in changed})
    run(spec, factory, run_id=RUN)
    assert len(factory.calls) == 2
    fresh = factory.backends[-1].calls
    assert len(fresh) == 3 + 2 * len(TEST)
    assert all(text.endswith("please") for text in fresh)  # validation stayed cached


def test_zero_shot_pseudo_run(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN, base=SMALL)
    (paths.task_home("intents") / "selected_run.json").write_text(json.dumps({"run_id": RUN}), encoding="utf-8")
    factory = Factory({f"base:{SMALL}": classification_script()})

    with pytest.raises(runner.EvalError, match="--zero-shot and --fast"):
        run(spec, factory, zero_shot=True, fast=True)
    assert factory.calls == []

    result = run(spec, factory, zero_shot=True)
    assert result["run_id"] == "zero-shot-qwen-0.5b"
    assert result["run"]["labels"] == "zero-shot" and result["run"]["base_model"] == SMALL
    assert factory.calls == [(SMALL, f"base:{SMALL}")]  # the base model without an adapter
    assert "tfidf" not in result["systems"]
    prompts = set(factory.backends[0].systems)
    assert prompts == {zero_shot_messages(spec, "")[0]["content"]}
    pseudo = paths.runs_dir("intents") / "zero-shot-qwen-0.5b"
    assert (pseudo / "preds_valid.jsonl").is_file() and (pseudo / "preds_test.jsonl").is_file()
    assert (paths.eval_dir("intents") / "zero-shot-qwen-0.5b" / "eval_test.json").is_file()
    assert not (paths.task_home("intents") / "threshold.json").exists()  # serving never uses the zero-shot threshold


def test_fast_profile_skips_alternatives(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    factory = Factory({RUN: classification_script()})
    result = run(spec, factory, run_id=RUN, fast=True)
    assert result["profile_fast"] is True
    assert result["confidence_comparison"]["test"]["alternative"] is None
    assert len(factory.generations) == 3 + len(VALID) + 3 + len(TEST)  # no free generations
    assert "tfidf" in result["systems"] and result["bootstrap"]["paired"]["stats"]


# -- extraction --------------------------------------------------------------------------------------------------------


def bill(vendor: str | None, total: float | None) -> dict[str, Any]:
    return {"vendor": vendor, "total": total}


def test_extraction_breakdowns_and_cluster_bootstrap(home: Path, fast_bootstrap: None) -> None:
    spec = extraction_spec()
    valid_docs = [
        ("Bill from Bralvik Ltd, total 10.00", bill("Bralvik Ltd", 10.0), "layout-a", 0.95),
        ("Bill from Corlune GmbH, total 20.50", bill("Corlune GmbH", 20.5), "layout-a", 0.9),
        ("Bill from Dremere LLC, total 30.00", bill("Dremere LLC", 30.0), "layout-a", 0.85),
        ("Bill from Eskdal S.A., total 41.00", bill("Eskdal S.A.", 41.0), "layout-b", 0.8),
        ("Bill from Falzen Ltd, total 55.10", bill("Falzen Ltd", 55.1), "layout-b", 0.75),
        ("Bill from Grelholm GmbH, total 60.00", bill("Grelholm GmbH", 60.0), "layout-b", 0.3),
    ]
    test_docs = [
        ("Invoice Hovgard Ltd 1.234,56 EUR", bill("Hovgard Ltd", 1234.56), bill("Hovgard Ltd", 1234.56),
         "email-x", ["eu_number_format"], bill("Hovgard Ltd", 1234.56), 0.9),
        ("Invoice Isterwyn GmbH 99.00", bill("Isterwyn GmbH", 99.0), bill("Isterwyn GmbH", 99.0),
         "email-x", [], bill("Isterwyn GmbH", 99.0), 0.88),
        ("Invoice Jornvik LLC 12.00", bill("Jornvik LLC", 12.0), bill("Jornvik LLC", 12.0),
         "email-x", ["missing_optional"], bill("Jornvik LLC", 21.0), 0.4),
        ("Invoice Kelmere Ltd 5,00", bill("Kelmere Ltd", 5.0), bill("Kelmere Ltd", 5.0),
         "layout-y", ["eu_number_format"], None, 0.9),
        ("Invoice Lumstead S.A. 7.50", bill("Lumstead S.A.", 7.5), bill("Lumstead S.A.", 7.5),
         "layout-y", [], bill("Lumstead S.A.", 7.5), 0.85),
        ("Invoice Mordtor GmbH 8.25", bill("Mordtor GmbH", 8.25), bill("Mordtor GmbH", 8.25),
         "layout-y", ["missing_optional"], bill("Mordtor GmbH", 8.25), 0.8),
        ("Invoice Nevbrook Ltd 3.10", bill("Nevbrook Ltd", 3.1), bill("Nevbrook Ltd", 3.1),
         "layout-z", [], bill("Nevbrook Ltd", 3.1), 0.95),
        ("Invoice Ostholm LLC 4.40", bill("Ostholm LLC", 4.4), bill("Ostholm LLC", 4.0),
         "layout-z", ["eu_number_format"], bill("Ostholm LLC", 4.4), 0.7),
        ("Invoice Pellane GmbH 6.60", bill("Pellane GmbH", 6.6), bill("Pellane GmbH", 6.6),
         "layout-z", [], bill("Pellane GmbH", 6.6), 0.65),
    ]  # fmt: skip
    write_split(
        spec,
        "valid",
        [{"input": t, "teacher": d, "gold": d, "meta": {"template": g}} for t, d, g, _ in valid_docs],
    )
    write_split(
        spec,
        "test",
        [
            {"input": t, "teacher": teacher, "gold": gold, "meta": {"template": g, "traits": traits}}
            for t, teacher, gold, g, traits, _, _ in test_docs
        ],
    )
    script: dict[str, tuple[str, float]] = {}
    for text, doc, _, conf in valid_docs:
        script[text] = (json.dumps(doc if conf > 0.5 else bill(doc["vendor"], 0.0)), conf)
    for text, _, _, _, _, answer, conf in test_docs:
        script[text] = ("not json" if answer is None else json.dumps(answer), conf)
    make_run(spec, RUN)
    result = run(spec, Factory({RUN: script}), run_id=RUN)

    systems = result["systems"]
    assert "tfidf" not in systems
    metrics = systems["student"]["metrics"]
    assert {"json_validity", "field_micro_f1", "field_exact_match", "doc_exact_match", "per_field_exact"} <= set(
        metrics
    )
    assert metrics["json_validity"] == pytest.approx(8 / 9)
    assert set(metrics["per_field_exact"]) == {"vendor", "total"}
    assert "agreement" in metrics
    assert systems["teacher"]["metrics"]["doc_exact_match"] == pytest.approx(8 / 9)  # the teacher misreads one total

    per_field = result["per_field"]
    assert per_field["fields"] == ["vendor", "total"]
    assert set(per_field["vs_gold"]) == {"student", "teacher", "cascade"}
    assert per_field["vs_gold"]["teacher"]["vendor"] == pytest.approx(1.0)
    assert per_field["vs_gold"]["teacher"]["total"] == pytest.approx(8 / 9)
    assert set(result["per_trait"]) == {"eu_number_format", "missing_optional", "(none)"}
    assert result["per_trait"]["eu_number_format"]["n"] == 3
    assert set(result["per_group"]) == {"email-x", "layout-y", "layout-z"}
    assert result["per_group"]["layout-y"]["student"]["json_validity"] == pytest.approx(2 / 3)
    cluster = result["bootstrap"]["cluster"]
    assert cluster is not None and cluster["method"] == "cluster" and cluster["n_groups"] == 3
    assert "student.field_micro_f1" in cluster["stats"]
    assert diff_name("cascade.agreement", "teacher.agreement") in cluster["diffs"]
    alternative = result["confidence_comparison"]["test"]["alternative"]
    assert alternative is not None and alternative["n"] == len(test_docs)


# -- paths -------------------------------------------------------------------------------------------------------------


def test_outputs_carry_no_absolute_paths(home: Path, fast_bootstrap: None, tmp_path: Path) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    make_run(spec, "qwen-0.5b-full-s14", seed=14)
    script = classification_script()
    result = run(spec, Factory({RUN: script, "qwen-0.5b-full-s14": script}), select=True)

    documents: list[Any] = [result]
    for path in [*home.rglob("eval_*.json"), *home.rglob("threshold.json"), *home.rglob("selected_run.json")]:
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    log_path = paths.task_home("intents") / "test_access_log.jsonl"
    documents.extend(json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines())
    assert len(documents) >= 5
    for document in documents:
        for text in walk_strings(document):
            assert str(tmp_path) not in text
            assert not text.startswith("/"), text
    assert result["files"]["eval"] == f"intents/eval/{result['run_id']}/eval_test.json"
    assert not math.isnan(result["systems"]["student"]["latency_ms"]["p95"])


def test_backend_mismatch_and_no_candidates_are_clear_errors(home: Path) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, "qwen-0.5b-full-s13-torch", backend="torch")
    with pytest.raises(runner.EvalError, match="--backend torch"):
        run(spec, Factory({}), run_id="qwen-0.5b-full-s13-torch")
    make_run(spec, "qwen-0.5b-full-s13-gold", labels="gold")
    with pytest.raises(runner.EvalError, match="no candidate runs"):
        runner.select_on_validation(spec, backend_factory=Factory({}), log=lambda _: None)


# -- local base models, zero-shot identity, commands ------------------------------------------------------------------


def test_local_base_model_directory_resolves_from_any_cwd(
    home: Path, fast_bootstrap: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    model_dir = paths.home() / "models" / "tiny"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    run_dir = make_run(spec, RUN, base="models/tiny")  # train_log.json and train_config.json keep it portable
    (run_dir / "train_config.json").write_text(json.dumps({"base_model": "models/tiny"}), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    info = runner.read_run(spec, RUN)
    assert info.base_model == str(model_dir) and Path(info.base_model).is_dir()
    factory = Factory({RUN: classification_script(), f"base:{model_dir}": classification_script()})
    result = run(spec, factory, run_id=RUN, split="valid")
    assert factory.calls == [(str(model_dir), RUN)]
    assert result["run"]["base_model"] == "models/tiny"

    zero = run(spec, factory, run_id=RUN, split="valid", zero_shot=True)
    assert factory.calls[-1] == (str(model_dir), f"base:{model_dir}")
    assert zero["run_id"] == "zero-shot-tiny" and zero["run"]["base_model"] == "models/tiny"
    header = json.loads((paths.runs_dir("intents") / "zero-shot-tiny" / "preds_valid.jsonl").read_text().split("\n")[0])
    assert header["base_model"] == "models/tiny" and header["adapter_sha256"] is None


def test_zero_shot_runs_are_kept_apart_by_base_model_and_backend(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    script = classification_script()
    four_bit, eight_bit = "org/Qwen2.5-0.5B-Instruct-4bit", "org/Qwen2.5-0.5B-Instruct-8bit"
    factory = Factory({f"base:{four_bit}": script, f"base:{eight_bit}": script})

    spec.student.base_model = four_bit
    first = run(spec, factory, zero_shot=True, split="valid")
    spec.student.base_model = eight_bit
    second = run(spec, factory, zero_shot=True, split="valid")
    assert first["run_id"] == second["run_id"] == "zero-shot-qwen2.5-0.5b"
    assert [key for _, key in factory.calls] == [f"base:{four_bit}", f"base:{eight_bit}"]  # no reuse across bases

    torch = run(spec, factory, zero_shot=True, split="valid", backend="torch")
    assert torch["run_id"] == "zero-shot-qwen2.5-0.5b-torch"
    assert (paths.eval_dir("intents") / "zero-shot-qwen2.5-0.5b-torch" / "eval_valid.json").is_file()


def test_command_paths_are_made_workspace_relative(home: Path, fast_bootstrap: None, tmp_path: Path) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    task_file = home / "tasks" / "intents.yaml"
    command = f"taskdistill eval --task {task_file} --report={tmp_path / 'out.json'} --run {RUN}"
    result = run(spec, Factory({RUN: classification_script()}), run_id=RUN, command=command)
    assert result["command"] == f"taskdistill eval --task tasks/intents.yaml --report=out.json --run {RUN}"
    log_path = paths.task_home("intents") / "test_access_log.jsonl"
    entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["command"] == result["command"] and str(tmp_path) not in entry["command"]
    demo = "taskdistill demo banking77 --profile quick (eval --select)"
    assert runner._portable_command(demo) == demo  # commands without paths are kept verbatim


def test_missing_sidecar_is_an_error(home: Path) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    (paths.data_dir("intents") / "test.meta.jsonl").unlink()
    factory = Factory({RUN: classification_script()})
    with pytest.raises(eval_data.EvalDataError, match=r"test\.meta\.jsonl is missing.*curate"):
        run(spec, factory, run_id=RUN)
    assert factory.calls == []  # refused before any prediction


# -- threshold.json ----------------------------------------------------------------------------------------------------


def test_task_threshold_follows_the_selected_run(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    other = "qwen-0.5b-full-s14"
    make_run(spec, RUN, seed=13)
    make_run(spec, other, seed=14)
    better = {text: (label, 0.9) for text, label, _, _ in VALID}  # all right at 0.9: threshold 0.9
    factory = Factory({RUN: classification_script(), other: better})
    task_threshold = paths.task_home("intents") / "threshold.json"
    (paths.task_home("intents") / "selected_run.json").write_text(json.dumps({"run_id": RUN}), encoding="utf-8")

    run(spec, factory, run_id=RUN, split="valid")
    assert json.loads(task_threshold.read_text())["run_id"] == RUN

    logs: list[str] = []
    result = run(spec, factory, run_id=other, split="valid", log=logs.append)
    stored = json.loads(task_threshold.read_text())
    assert (stored["run_id"], stored["threshold"]) == (RUN, pytest.approx(0.4))  # the selected run's, kept
    own = json.loads((paths.eval_dir("intents") / other / "threshold.json").read_text())
    assert (own["run_id"], own["threshold"]) == (other, pytest.approx(0.9))
    assert result["selected"] is False and result["files"]["run_threshold"] == f"intents/eval/{other}/threshold.json"
    assert any(f"intents/eval/{other}/threshold.json" in line and f"serve --run {other}" in line for line in logs)
    # serve --threshold auto finds each run's own threshold, with no warning.
    for run_id, value in ((RUN, 0.4), (other, 0.9)):
        resolved = resolve_threshold(spec, "auto", run_id)
        assert resolved.value == pytest.approx(value) and resolved.warnings == ()

    # A new selection that is not evaluated in the same call (here with --zero-shot) still moves threshold.json.
    run(spec, factory, select=True, zero_shot=True, split="valid")
    assert json.loads((paths.task_home("intents") / "selected_run.json").read_text())["run_id"] == other
    stored = json.loads(task_threshold.read_text())
    assert (stored["run_id"], stored["threshold"]) == (other, pytest.approx(0.9))


def test_select_on_validation_writes_the_selected_threshold(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    write_classification_data(spec)
    make_run(spec, RUN)
    factory = Factory({RUN: classification_script()})
    payload = runner.select_on_validation(spec, backend_factory=factory, log=lambda _: None)
    stored = json.loads((paths.task_home("intents") / "threshold.json").read_text())
    assert payload["run_id"] == stored["run_id"] == RUN
    assert stored["threshold"] == pytest.approx(0.4) and stored["split"] == "valid"


def test_always_escalate_when_no_threshold_reaches_the_target(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    spec.cascade.target = 1.0
    write_classification_data(spec)
    make_run(spec, RUN)
    script = classification_script()
    script[VALID[0][0]] = (CW, 0.99)  # the most confident validation answer is wrong
    logs: list[str] = []
    result = run(spec, Factory({RUN: script}), run_id=RUN, log=logs.append)

    assert result["threshold"]["threshold"] is None and result["threshold"]["always_escalate"] is True
    cascade = result["systems"]["cascade"]
    assert cascade["threshold"] is None and cascade["escalation_rate"] == pytest.approx(1.0)
    assert cascade["metrics"]["agreement"] == pytest.approx(1.0)
    assert cascade["metrics"]["accuracy"] == pytest.approx(result["systems"]["teacher"]["metrics"]["accuracy"])
    assert result["operating_point"]["test"]["escalation_rate"] == pytest.approx(1.0)
    assert any(line.startswith("warning:") and "escalat" in line for line in logs)
    assert json.loads((paths.task_home("intents") / "threshold.json").read_text())["threshold"] is None


# -- no gold, gold reference, isotonic per reference ------------------------------------------------------------------


def write_bills(spec: TaskSpec, *, gold: bool, test_note: bool = False) -> dict[str, tuple[str, float]]:
    """A small extraction task (two templates per split); returns the student script.

    ``test_note`` adds a ``note`` field that only the test documents (and answers) carry.
    """
    valid = [
        ("Bill A1 Bralvik Ltd 10.00", bill("Bralvik Ltd", 10.0), "t1", bill("Bralvik Ltd", 10.0), 0.95),
        ("Bill A2 Corlune GmbH 20.50", bill("Corlune GmbH", 20.5), "t1", bill("Corlune GmbH", 20.5), 0.9),
        ("Bill A3 Dremere LLC 30.00", bill("Dremere LLC", 30.0), "t2", bill("Dremere LLC", 3.0), 0.4),
        ("Bill A4 Eskdal S.A. 41.00", bill("Eskdal S.A.", 41.0), "t2", bill("Eskdal S.A.", 41.0), 0.8),
    ]
    test = [
        ("Bill B1 Falzen Ltd 55.10", bill("Falzen Ltd", 55.1), "t3", bill("Falzen Ltd", 55.1), 0.9),
        ("Bill B2 Grelholm GmbH 60.00", bill("Grelholm GmbH", 60.0), "t3", bill("Grelholm GmbH", 6.0), 0.3),
        ("Bill B3 Hovgard Ltd 7.25", bill("Hovgard Ltd", 7.25), "t4", bill("Hovgard Ltd", 7.25), 0.85),
        ("Bill B4 Isterwyn GmbH 8.00", bill("Isterwyn GmbH", 8.0), "t4", bill("Isterwyn GmbH", 8.0), 0.7),
    ]
    if test_note:
        test = [(text, {**doc, "note": "paid"}, g, {**ans, "note": "paid"}, c) for text, doc, g, ans, c in test]
    for split, rows in (("valid", valid), ("test", test)):
        write_split(
            spec,
            split,
            [
                {"input": text, "teacher": doc, "gold": doc if gold else None, "meta": {"template": group}}
                for text, doc, group, _, _ in rows
            ],
        )
    return {text: (json.dumps(answer), conf) for text, _, _, answer, conf in [*valid, *test]}


@pytest.mark.parametrize("task_type", ["classification", "extraction"])
def test_no_gold_reports_agreement_only(home: Path, fast_bootstrap: None, task_type: str) -> None:
    spec = classification_spec() if task_type == "classification" else extraction_spec()
    if task_type == "classification":
        write_split(spec, "train", [{"input": text, "teacher": label} for text, label in TRAIN])
        write_split(spec, "valid", [{"input": text, "teacher": label} for text, label, _, _ in VALID])
        write_split(spec, "test", [{"input": t, "teacher": teacher} for t, teacher, _, _, _ in TEST])
        script = classification_script()
        gold_metric = "accuracy"
    else:
        script = write_bills(spec, gold=False)
        gold_metric = "field_micro_f1"
    make_run(spec, RUN)
    result = run(spec, Factory({RUN: script}), run_id=RUN)

    systems = result["systems"]
    assert systems["teacher"]["metrics"] == {}
    student = systems["student"]["metrics"]
    assert student[gold_metric] is None and student["n_gold"] == 0
    assert student["agreement"] is not None
    calibration = systems["student"]["calibration"]
    assert calibration["correctness"] == "teacher"
    assert calibration["raw"]["vs_gold"] is None and calibration["raw"]["vs_teacher"] is not None
    assert calibration["isotonic"]["vs_gold"] is None and calibration["isotonic"]["vs_teacher"] is not None
    assert calibration["isotonic_fit"]["reference"] == "teacher" and calibration["isotonic_fit"]["vs_gold"] is None
    paired = result["bootstrap"]["paired"]
    assert f"student.{gold_metric}" not in paired["stats"] and "student.agreement" in paired["stats"]
    assert diff_name("student.agreement", "teacher.agreement") in paired["diffs"]
    assert not any(gold_metric in name for name in paired["diffs"])
    assert result["operating_point"]["test"]["available"] is True
    assert result["operating_point"]["test"]["met"] is not None
    if task_type == "classification":
        assert systems["tfidf"]["metrics"]["accuracy"] is None and systems["tfidf"]["metrics"]["agreement"] is not None
    else:
        assert result["per_field"]["vs_gold"] is None and result["per_field"]["vs_teacher"]["student"]
        assert result["bootstrap"]["cluster"] is not None and result["bootstrap"]["cluster"]["n_groups"] == 2


@pytest.mark.parametrize("test_gold", [True, False])
def test_gold_reference_with_max_drop(home: Path, fast_bootstrap: None, test_gold: bool) -> None:
    spec = classification_spec()
    spec.cascade.reference, spec.cascade.metric = "gold", "accuracy"
    spec.cascade.target, spec.cascade.max_drop = None, 0.05
    write_classification_data(spec)
    if not test_gold:
        write_split(spec, "test", [{"input": t, "teacher": teacher} for t, teacher, _, _, _ in TEST])
    make_run(spec, RUN)
    result = run(spec, Factory({RUN: classification_script()}), run_id=RUN)

    threshold = result["threshold"]
    assert (threshold["reference"], threshold["metric"], threshold["max_drop"]) == ("gold", "accuracy", 0.05)
    assert threshold["target_value"] == pytest.approx(0.95)  # the teacher is right on every validation record
    assert threshold["threshold"] == pytest.approx(0.5)  # escalates the two wrong answers (0.4 and 0.3)
    point = result["operating_point"]["test"]
    if test_gold:
        assert point["available"] is True
        assert point["target_value"] == pytest.approx(point["teacher_quality"] - 0.05)
        assert point["met"] is (point["quality"] >= point["target_value"])
    else:
        assert point["available"] is False and point["met"] is None
        assert point["reason"] == "no gold labels on the test split"
        assert point["escalation_rate"] == pytest.approx(0.3)  # the 0.47, 0.45 and 0.35 records
        assert result["systems"]["student"]["metrics"]["accuracy"] is None
        assert (paths.eval_dir("intents") / RUN / "eval_test.json").is_file()


def test_isotonic_maps_are_fitted_per_reference(home: Path, fast_bootstrap: None) -> None:
    spec = classification_spec()
    other = {CA: ER, CW: CA, ER: CA}
    valid_rows = []
    for i, (text, label, _, _) in enumerate(VALID):
        gold = other[label] if i in (1, 4, 7, 10) else None if i in (2, 5) else label
        valid_rows.append({"input": text, "teacher": label, "gold": gold})
    write_split(spec, "train", [{"input": text, "teacher": label, "gold": label} for text, label in TRAIN])
    write_split(spec, "valid", valid_rows)
    write_split(spec, "test", [{"input": t, "teacher": te, "gold": g} for t, te, g, _, _ in TEST])
    # The student always agrees with the teacher, at the scripted confidences.
    script = {text: (label, conf) for text, label, _, conf in VALID}
    script.update({text: (teacher, conf) for text, teacher, _, _, conf in TEST})
    make_run(spec, RUN)
    result = run(spec, Factory({RUN: script}), run_id=RUN)

    calibration = result["systems"]["student"]["calibration"]
    fit = calibration["isotonic_fit"]
    assert fit["reference"] == "gold where present, else teacher" and fit["n_fit"] == len(VALID)
    assert fit["vs_gold"]["n_fit"] == len(VALID) - 2 and fit["vs_teacher"]["n_fit"] == len(VALID)
    assert set(fit["vs_teacher"]["map"]["y_thresholds"]) == {1.0}  # always right against the teacher
    # vs_teacher uses the teacher-fitted map: the student agrees with the teacher everywhere, so ECE is 0.
    assert calibration["isotonic"]["vs_teacher"]["ece"] == pytest.approx(0.0)
    assert calibration["raw"]["vs_teacher"]["ece"] > 0.0
    gold_map = IsotonicCalibrator.from_dict(fit["vs_gold"]["map"])
    conf = [conf for *_, conf in TEST]
    correct = [teacher == gold for _, teacher, gold, _, _ in TEST]
    expected = calibration_report(gold_map.apply(conf), correct)
    assert calibration["isotonic"]["vs_gold"]["ece"] == pytest.approx(expected["ece"])


def test_extraction_choices_use_validation_fields_only(
    home: Path, fast_bootstrap: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = extraction_spec()
    spec.json_schema = {"type": "object"}  # no properties: the fields come from the documents
    script = write_bills(spec, gold=True, test_note=True)
    make_run(spec, RUN)
    seen: list[Any] = []
    select = select_threshold

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("fields"))
        return select(*args, **kwargs)

    monkeypatch.setattr(runner, "select_threshold", spy)
    result = run(spec, Factory({RUN: script}), run_id=RUN)
    assert seen == [["vendor", "total"]]  # the test-only field never shapes the threshold
    assert result["per_field"]["fields"] == ["vendor", "total", "note"]  # but it is reported
