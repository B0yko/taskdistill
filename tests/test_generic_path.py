"""The generic path: a task that is not a demo goes through the whole product.

A support-ticket extraction task (``tests/fixtures/generic_task/``) runs through ``init -> capture --import ->
curate -> train --backend torch -> eval -> serve -> report`` using only public entry points: the Typer app through
``CliRunner``, and the FastAPI app that ``serve`` builds (``build_server``) through ``TestClient``. The teacher API
is faked with respx on its base URL (pricing, curate labelling, serve escalations); the student base is a tiny
random-weight model built in the test.

The fixture holds the application's captured traffic for the training tickets (``captured.jsonl``, the
``openai`` format ``capture --export`` writes), every ticket with its gold fields and predefined split
(``inputs.jsonl``), and the teacher's answers to the tickets that were never captured (``teacher_answers.jsonl``).
Deliberate quirks: one ticket was sent twice, one captured answer violates the schema, one forwarded thread is too
long for ``train.max_seq_len``, some tickets carry an email address or a phone number, the teacher misreads the
severity of two tickets, the validation and test splits differ in size (so a threshold chosen on the wrong split
shows), and the teacher's answer to the ticket sent to the server is fenced JSON with its keys in another order (so a
verbatim pass-through shows).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

pytest.importorskip("torch")
pytest.importorskip("peft")

from taskdistill.cli import app
from taskdistill.config import load_task
from taskdistill.serve.runner import build_server
from taskdistill.tasks.extraction import canonical_output, parse_json_output
from tiny_model import build_tiny_model

FIXTURE = Path(__file__).parent / "fixtures" / "generic_task"
TASK = "support-tickets"
TEACHER_BASE = "https://llm.example.com/api/v1"
TEACHER_MODEL = "example/ticket-extractor"
TEACHER_KEY = "teacher-key-for-tests"
KEY_ENV = "TASKDISTILL_TEACHER_API_KEY"
CLIENT_KEY = "client-key-never-forwarded"
APP_MODEL = "ticket-triage-prod"
STUDENT_PROMPT = "Extract the ticket fields as JSON."
PROMPT_PRICE, COMPLETION_PRICE = 2e-7, 8e-7
PROMPT_TOKENS, COMPLETION_TOKENS = 160, 32
CALL_COST = PROMPT_TOKENS * PROMPT_PRICE + COMPLETION_TOKENS * COMPLETION_PRICE  # usage.cost of every fake answer
RUN_ID = "tiny-student-full-s13-torch"
N_TRAIN, N_VALID, N_TEST = 48, 14, 12  # tickets per split in inputs.jsonl; only the training tickets were captured
N_LABELLED = N_VALID + N_TEST  # never captured, so curate has the teacher label them
NEW_TICKET = (
    "Ticket ST-3190: the speaker hums loudly whenever it is idle. It still plays music. Please refund 39.00 for the "
    "trouble."
)
ENV_VARS = (
    KEY_ENV,
    "OPENROUTER_API_KEY",
    "TASKDISTILL_TEACHER_BASE_URL",
    "TASKDISTILL_TEACHER_MODEL",
    "TASKDISTILL_SERVER_TOKEN",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_user_message(body: dict[str, Any]) -> str:
    return next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")


class FakeTeacher:
    """The teacher API: answers each ticket from the fixture and records every chat-completions call."""

    def __init__(self) -> None:
        self.answers: dict[str, str] = {}
        for row in read_jsonl(FIXTURE / "captured.jsonl"):
            self.answers[last_user_message(row["request"])] = row["response"]["choices"][0]["message"]["content"]
        for row in read_jsonl(FIXTURE / "teacher_answers.jsonl"):
            self.answers[row["input"]] = row["output"]
        self.calls: list[tuple[str | None, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((request.headers.get("authorization"), body))
        if request.headers.get("authorization") != f"Bearer {TEACHER_KEY}":
            return httpx.Response(401, json={"error": {"message": "unknown API key", "code": 401}})
        answer = self.answers.get(last_user_message(body))
        if answer is None:
            return httpx.Response(400, json={"error": {"message": "unexpected input", "code": 400}})
        n = len(self.calls)
        usage = {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            "cost": CALL_COST,
        }
        return httpx.Response(
            200,
            json={
                "id": f"gen-test-{n}",
                "object": "chat.completion",
                "created": 1790000000 + n,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": usage,
            },
        )


@contextmanager
def teacher_api(fake: FakeTeacher) -> Iterator[respx.MockRouter]:
    """The teacher's base URL; any other HTTP request made through httpx fails the test."""
    pricing = {"prompt": f"{PROMPT_PRICE:.7f}", "completion": f"{COMPLETION_PRICE:.7f}"}
    models = {"data": [{"id": TEACHER_MODEL, "pricing": pricing}]}
    with respx.mock(base_url=TEACHER_BASE, assert_all_called=False, assert_all_mocked=True) as router:
        router.get("/models").mock(return_value=httpx.Response(200, json=models))
        router.post("/chat/completions").mock(side_effect=fake)
        yield router


def invoke(runner: CliRunner, *args: str) -> str:
    result = runner.invoke(app, list(args))
    detail = f"taskdistill {' '.join(args)} exited with {result.exit_code}:\n{result.output}"
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        detail += f"\n{type(result.exception).__name__}: {result.exception}"
    assert result.exit_code == 0, detail
    return result.output


def customise_task(task_dir: Path, base_model: Path) -> None:
    """What a user does after ``init``: write the schema and prompt, name the teacher and the student, size training."""
    for name in ("schema.json", "teacher_prompt.md"):
        (task_dir / name).write_text((FIXTURE / name).read_text(encoding="utf-8"), encoding="utf-8")
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    spec["teacher"].update(base_url=TEACHER_BASE, model=TEACHER_MODEL, max_tokens=128)
    spec["student"].update(base_model=str(base_model), system_prompt=STUDENT_PROMPT, max_tokens=80)
    spec["train"].update(epochs=2, batch_size=4, max_seq_len=256, lora_rank=4, learning_rate=5e-3)
    (task_dir / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def canonical(answer: str) -> str:
    schema = json.loads((FIXTURE / "schema.json").read_text(encoding="utf-8"))
    parsed = parse_json_output(answer)
    assert parsed is not None
    return canonical_output(parsed, schema)


@pytest.mark.slow
def test_a_new_extraction_task_runs_from_init_to_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / ".taskdistill"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TASKDISTILL_HOME", str(home))
    monkeypatch.setenv("TASKDISTILL_BUDGET_USD", "5")
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    base_model = build_tiny_model(tmp_path / "tiny-student")
    runner = CliRunner()
    fake = FakeTeacher()
    task_home = home / TASK
    inputs = read_jsonl(FIXTURE / "inputs.jsonl")
    captured = read_jsonl(FIXTURE / "captured.jsonl")

    with teacher_api(fake):
        # init: scaffold the spec, then make it this task's
        output = invoke(runner, "init", TASK, "--type", "extraction")
        task_dir = tmp_path / "tasks" / TASK
        assert f"created tasks/{TASK}/" in output
        assert sorted(p.name for p in task_dir.iterdir()) == ["schema.json", "task.yaml", "teacher_prompt.md"]
        customise_task(task_dir, base_model)
        spec = load_task(TASK)
        assert (spec.type, spec.schema_fields) == ("extraction", ["ticket_id", "product", "severity", "refund_amount"])

        # the live teacher is priced from its API's model list
        invoke(runner, "pricing", "refresh", "--base-url", TEACHER_BASE)
        assert TEACHER_MODEL in json.loads((home / "pricing.json").read_text(encoding="utf-8"))["models"]

        # capture --import: the application's traffic, then every ticket with its gold fields and split
        output = invoke(runner, "capture", "--task", TASK, "--import", str(FIXTURE / "captured.jsonl"))
        assert f"imported {len(captured)} of {len(captured)} rows (openai)" in output
        output = invoke(
            runner, "capture", "--task", TASK, "--import", str(FIXTURE / "inputs.jsonl"), "--format", "inputs"
        )
        assert f"imported {len(inputs)} of {len(inputs)} rows (inputs)" in output
        last_import = json.loads((task_home / "last_import.json").read_text(encoding="utf-8"))
        assert (last_import["file"], last_import["imported"]) == ("inputs.jsonl", len(inputs))
        exported = tmp_path / "exported.jsonl"
        invoke(runner, "capture", "--task", TASK, "--export", str(exported))
        assert read_jsonl(exported) == captured  # the fixture is exactly what `capture --export` writes

        # curate: the valid and test tickets were never captured, so the live (fake) teacher labels them
        monkeypatch.setenv(KEY_ENV, TEACHER_KEY)
        output = invoke(runner, "curate", "--task", TASK, "--max-usd", "0.10")
        monkeypatch.delenv(KEY_ENV)
        assert f"teacher: live ({TEACHER_MODEL})" in output
        check_curate(task_home, inputs, captured, fake)

        # train: LoRA on the tiny student with the torch backend
        output = invoke(runner, "train", "--task", TASK, "--backend", "torch")
        assert f"run {RUN_ID}:" in output
        run_dir = task_home / "runs" / RUN_ID
        train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
        assert train_log["backend"] == "torch"
        assert (train_log["n_train"], train_log["n_valid"]) == (N_TRAIN - 2, N_VALID)  # schema violation, long thread
        iteration, initial_val_loss = train_log["curve"]["val"][0]
        assert iteration == 0
        # the curated tickets teach the adapter: at lr 5e-3 validation loss falls far below the untrained model's
        assert train_log["best_val_loss"] < 0.8 * initial_val_loss
        assert (run_dir / "adapter" / "adapter_config.json").is_file()
        assert (run_dir / "loss.png").is_file()

        # eval: choose the run and the threshold on validation, report on test
        invoke(runner, "eval", "--task", TASK, "--backend", "torch", "--select")
        check_eval(task_home)
        check_student_predictions(run_dir)
        assert len(fake.calls) == N_LABELLED  # eval scores the teacher from its recorded answers, never live

        # serve: the app serve builds, with the live (fake) teacher behind it
        monkeypatch.setenv(KEY_ENV, TEACHER_KEY)
        check_serve(fake)
        monkeypatch.delenv(KEY_ENV)

        # report, including the traffic just served
        out_dir = tmp_path / "report-out"
        output = invoke(runner, "report", "--task", TASK, "--from-serve-log", "--since", "1h", "--out", str(out_dir))
        assert "report.md" in output
        check_report(out_dir)


def check_curate(
    task_home: Path, inputs: list[dict[str, Any]], captured: list[dict[str, Any]], fake: FakeTeacher
) -> None:
    data = task_home / "data"
    stats = json.loads((data / "curate_stats.json").read_text(encoding="utf-8"))
    stages = stats["stages"]
    by_split = {name: sum(1 for row in inputs if row["meta"]["split"] == name) for name in ("train", "valid", "test")}
    assert by_split == {"train": N_TRAIN, "valid": N_VALID, "test": N_TEST}

    load = stages["load"]
    assert (load["captures"]["usable"], load["imports"]["total"]) == (len(captured), len(inputs))
    assert (load["captures"]["other_model"], load["captures"]["other_prompt"]) == (0, 0)
    merge = stages["merge"]
    assert (merge["examples"], merge["with_teacher_output"], merge["with_both"]) == (len(inputs), N_TRAIN, N_TRAIN)
    assert merge["repeated_inputs"] == 1  # one ticket was sent twice
    normalise = stages["normalise"]
    assert (normalise["invalid_outputs"], normalise["dropped_all_invalid"]) == (1, 1)  # severity "critical"
    assert (normalise["labelled"], normalise["unlabelled"]) == (N_TRAIN - 1, N_LABELLED)

    emails = sum(1 for row in inputs if "@example.com" in row["input"])
    phones = sum(1 for row in inputs if "555-01" in row["input"])
    assert emails and phones
    assert (stages["pii"]["hits"]["email"], stages["pii"]["hits"]["phone"]) == (emails, phones)
    assert stages["split"]["splits"] == {"train": N_TRAIN - 1, "valid": N_VALID, "test": N_TEST}
    assert (stages["dedupe"]["removed_exact"], stages["dedupe"]["removed_near"]) == (0, 0)
    assert sum(stages["cross_split"]["removed"].values()) == 0

    label = stages["label"]
    assert (label["requested"], label["labelled"], label["live"], label["invalid"]) == (N_LABELLED,) * 3 + (0,)
    assert label["cost_usd"] == pytest.approx(N_LABELLED * CALL_COST)
    assert stages["length"]["dropped"] == 1  # the forwarded thread
    assert stages["leakage"]["ok"] is True
    assert stats["splits"] == {"train": N_TRAIN - 2, "valid": N_VALID, "test": N_TEST}

    assert len(fake.calls) == N_LABELLED
    for authorization, body in fake.calls:
        assert authorization == f"Bearer {TEACHER_KEY}"
        assert body["model"] == TEACHER_MODEL
        assert body["messages"][0]["content"] == (FIXTURE / "teacher_prompt.md").read_text(encoding="utf-8")
    labelled = {last_user_message(body) for _, body in fake.calls}
    assert labelled == {row["input"] for row in inputs if row["meta"]["split"] != "train"}

    rows = read_jsonl(data / "train.jsonl")
    assert len(rows) == N_TRAIN - 2
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    assert {row["messages"][0]["content"] for row in rows} == {STUDENT_PROMPT}
    for row in rows:
        target = row["messages"][2]["content"]
        assert target == canonical(target)  # compact JSON in schema order, whatever the teacher's formatting
        assert "@example.com" not in row["messages"][1]["content"]
    assert any("<EMAIL>" in row["messages"][1]["content"] for row in rows)
    assert len(read_jsonl(data / "valid.meta.jsonl")) == N_VALID
    assert len(read_jsonl(data / "test.meta.jsonl")) == N_TEST
    assert (data / "dataset_card.md").is_file()


def check_eval(task_home: Path) -> None:
    selected = json.loads((task_home / "selected_run.json").read_text(encoding="utf-8"))
    assert (selected["run_id"], selected["split"], selected["n_valid"]) == (RUN_ID, "valid", N_VALID)
    result = json.loads((task_home / "eval" / RUN_ID / "eval_test.json").read_text(encoding="utf-8"))
    assert (result["task_type"], result["split"], result["n"], result["n_valid"]) == (
        "extraction",
        "test",
        N_TEST,
        N_VALID,
    )
    systems = result["systems"]
    assert set(systems) == {"student", "teacher", "cascade"}
    headline = ("json_validity", "field_micro_f1", "field_exact_match", "doc_exact_match")
    for name in ("student", "teacher", "cascade"):
        metrics = systems[name]["metrics"]
        assert (metrics["n"], metrics["n_gold"]) == (N_TEST, N_TEST)
        for key in headline + (() if name == "teacher" else ("agreement",)):
            assert 0.0 <= metrics[key] <= 1.0, (name, key)
    teacher = systems["teacher"]["metrics"]
    assert teacher["json_validity"] == 1.0
    assert (teacher["tp"], teacher["fp"], teacher["fn"]) == (40, 1, 1)  # one misread severity among the test tickets
    assert teacher["doc_exact_match"] == pytest.approx(11 / 12)
    assert {"raw", "isotonic"} <= set(systems["student"]["calibration"])

    # The tiny student cannot reach 0.97 agreement with the teacher on validation, so eval chose to always escalate
    # (threshold null) and the cascade on test is exactly the teacher.
    threshold = result["threshold"]
    assert (threshold["reference"], threshold["metric"], threshold["target"]) == ("teacher", "agreement", 0.97)
    assert (threshold["threshold"], threshold["always_escalate"], threshold["met"]) == (None, True, True)
    assert threshold["n"] == N_VALID != N_TEST  # chosen on the validation tickets, not the test ones
    cascade = systems["cascade"]
    assert (cascade["threshold"], cascade["escalation_rate"]) == (None, 1.0)
    assert cascade["metrics"]["agreement"] == 1.0
    assert (cascade["metrics"]["tp"], cascade["metrics"]["fp"], cascade["metrics"]["fn"]) == (40, 1, 1)
    assert cascade["metrics"]["doc_exact_match"] == teacher["doc_exact_match"]
    point = result["operating_point"]
    assert set(point) == {"valid", "test"}
    assert (point["valid"]["n"], point["valid"]["threshold"], point["valid"]["escalation_rate"]) == (N_VALID, None, 1.0)
    assert (point["test"]["n"], point["test"]["threshold"], point["test"]["escalation_rate"]) == (N_TEST, None, 1.0)
    chosen = json.loads((task_home / "threshold.json").read_text(encoding="utf-8"))
    assert (chosen["run_id"], chosen["split"], chosen["threshold"], chosen["n"]) == (RUN_ID, "valid", None, N_VALID)
    assert (task_home / "eval" / RUN_ID / "threshold_curve.png").is_file()


def check_student_predictions(run_dir: Path) -> None:
    """The torch student generated text for every validation and test ticket, not just an end token."""
    for split, n in (("valid", N_VALID), ("test", N_TEST)):
        header, *rows = read_jsonl(run_dir / f"preds_{split}.jsonl")
        assert (header["type"], header["backend"]) == ("header", "torch")
        assert len(rows) == n
        for row in rows:
            assert row["completion_tokens"] > 1, (split, row)  # the count includes the end token when there is one
            assert 0.0 <= row["confidence"] <= 1.0


def check_serve(fake: FakeTeacher) -> None:
    """``serve`` as eval left it (threshold auto: always escalate), then at threshold 0 (the student answers)."""
    spec = load_task(TASK)
    prompt = (FIXTURE / "teacher_prompt.md").read_text(encoding="utf-8")
    request = {
        "model": APP_MODEL,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": NEW_TICKET}],
        "temperature": 0,
        "max_tokens": 128,
    }
    auth = {"Authorization": f"Bearer {CLIENT_KEY}"}

    server, banner = build_server(spec, backend="torch")
    assert f"teacher: live ({TEACHER_MODEL} via {TEACHER_BASE})" in banner
    assert "backend: torch" in banner
    assert any(line.startswith(f"run: {RUN_ID}, from selected_run.json") for line in banner)
    # eval chose "always escalate" (threshold null) on validation, and serve took it from threshold.json
    threshold_line = f"threshold: always escalate (auto: threshold.json, chosen on validation for run {RUN_ID})"
    assert threshold_line in banner
    assert not any(line.startswith("warning:") for line in banner)
    before = len(fake.calls)
    with TestClient(server) as client:
        health = client.get("/healthz")
        models = client.get("/v1/models").json()
        resp = client.post("/v1/chat/completions", json=request, headers=auth)
    assert health.status_code == 200
    assert {k: health.json()[k] for k in ("status", "task", "run", "threshold", "teacher", "backend")} == {
        "status": "ok",
        "task": TASK,
        "run": RUN_ID,
        "threshold": None,
        "teacher": "live",
        "backend": "torch",
    }
    assert [m["id"] for m in models["data"]] == [f"taskdistill/{TASK}"]

    assert resp.status_code == 200, resp.text
    assert 0.0 <= float(resp.headers["x-taskdistill-confidence"]) <= 1.0  # the student ran before escalating
    assert resp.headers["x-taskdistill-route"] == "teacher"
    assert resp.headers["x-taskdistill-reason"] == "low_confidence"
    assert resp.headers["x-taskdistill-teacher"] == "live"
    body = resp.json()
    assert (body["object"], body["model"]) == ("chat.completion", APP_MODEL)
    content = body["choices"][0]["message"]["content"]
    assert content == canonical(fake.answers[NEW_TICKET])  # normalised like curate's training targets
    assert content != fake.answers[NEW_TICKET]

    assert len(fake.calls) == before + 1
    authorization, sent = fake.calls[-1]
    assert authorization == f"Bearer {TEACHER_KEY}"
    assert sent == {**request, "model": TEACHER_MODEL}  # only the model is replaced

    # at threshold 0 the torch student answers the same ticket itself, whatever its confidence
    server, banner = build_server(spec, backend="torch", threshold=0)
    assert "threshold: 0 (never escalates on confidence) (given)" in banner
    with TestClient(server) as client:
        assert client.get("/healthz").json()["threshold"] == 0
        resp = client.post("/v1/chat/completions", json=request, headers=auth)
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-taskdistill-route"] == "student"
    assert 0.0 <= float(resp.headers["x-taskdistill-confidence"]) <= 1.0
    assert "x-taskdistill-reason" not in resp.headers
    assert "x-taskdistill-teacher" not in resp.headers
    body = resp.json()
    assert (body["object"], body["model"]) == ("chat.completion", APP_MODEL)
    assert body["choices"][0]["message"]["content"].strip()  # the student's own generation
    assert body["usage"]["completion_tokens"] > 1
    assert len(fake.calls) == before + 1  # the teacher was not asked


def check_report(out_dir: Path) -> None:
    assert TASK in (out_dir / "report.md").read_text(encoding="utf-8")
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert (report["task"], report["task_type"], report["notes"]) == (TASK, "extraction", [])
    assert report["selected_run"]["run_id"] == RUN_ID
    assert report["teacher"]["model"] == TEACHER_MODEL
    quality = report["quality"]
    assert (quality["split"], quality["n"]) == ("test", N_TEST)
    assert {"teacher", "student", "cascade"} <= {row["system"] for row in quality["rows"]}
    point = report["operating_point"]
    assert point["run_id"] == RUN_ID
    assert (point["reference"], point["metric"], point["target"]) == ("teacher", "agreement", 0.97)
    assert set(point["escalation_rate"]) == {"valid", "test"}
    costs = report["cost_latency"]
    assert costs["teacher"]["usd_per_request"] == pytest.approx(CALL_COST)  # the recorded usage.cost
    assert costs["student"]["latency_ms"]["n"] == costs["cascade"]["n"] == N_TEST
    assert report["labelling_cost"]["usd"] == pytest.approx(N_LABELLED * CALL_COST)
    assert report["break_even"]["labelling_usd"] == pytest.approx(N_LABELLED * CALL_COST)
    serve_log = report["serve_log"]
    assert (serve_log["requests"], serve_log["route_counts"]) == (2, {"student": 1, "teacher": 1})
    assert (serve_log["reasons"], serve_log["teacher_modes"]) == ({"low_confidence": 1}, {"live": 1})
    assert (serve_log["escalations"], serve_log["confidence_decisions"]) == (1, 2)
    assert serve_log["teacher_spend_usd"] == pytest.approx(CALL_COST)
