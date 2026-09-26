"""CLI smoke tests: every command runs through Typer's CliRunner in a temporary workspace, without network."""

from __future__ import annotations

import importlib
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import typer.main
import typer.rich_utils
from typer.core import TyperGroup
from typer.testing import CliRunner, Result

from taskdistill import __version__
from taskdistill.backends.factory import torch_unavailable_reason
from taskdistill.cli import app
from taskdistill.store import Store

TASK = "tickets"
TEST_API_KEY = "sk-test-not-a-real-key"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
BOX = "│|╭╮╰╯─┃━┏┓┗┛ \t"
ENV_VARS = (
    "OPENROUTER_API_KEY",
    "TASKDISTILL_TEACHER_API_KEY",
    "TASKDISTILL_TEACHER_BASE_URL",
    "TASKDISTILL_TEACHER_MODEL",
    "TASKDISTILL_BUDGET_USD",
    "TASKDISTILL_SERVER_TOKEN",
)
TOP_LEVEL_COMMANDS = (
    "init",
    "capture",
    "curate",
    "train",
    "eval",
    "serve",
    "report",
    "bench",
    "demo",
    "budget",
    "teacher",
    "pricing",
)
MONEY_COMMANDS = (
    ("curate",),
    ("demo",),
    ("bench",),
    ("teacher", "bakeoff"),
    ("teacher", "record"),
    ("serve",),
)
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
MODELS_PAYLOAD: dict[str, Any] = {
    "data": [
        {
            "id": "vendor-a/instruct-small",
            "canonical_slug": "vendor-a/instruct-small-0601",
            "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
        },
        {
            "id": "vendor-b/instruct-large",
            "pricing": {"prompt": "0.000002", "completion": "0.000008", "request": "0.001"},
        },
        {"id": "vendor-c/router", "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
}


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh working directory and ``$TASKDISTILL_HOME``, with no API keys or teacher overrides set."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "workspace"
    monkeypatch.setenv("TASKDISTILL_HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return home


@pytest.fixture(autouse=True)
def wide_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typer reads ``TERMINAL_WIDTH`` once, at import, and prefers it to ``COLUMNS``; a narrow value would cut
    option names in the help panels, so the tests ignore it and ``invoke`` sets the width."""
    monkeypatch.setattr(typer.rich_utils, "MAX_WIDTH", None)


@pytest.fixture(autouse=True)
def http() -> Iterator[respx.MockRouter]:
    """Every httpx request fails unless a test routes it, so no command can reach the network."""
    with respx.mock(assert_all_called=False) as router:
        yield router


def invoke(*args: str) -> Result:
    """Run the CLI with a wide terminal so help panels are never wrapped."""
    result = CliRunner().invoke(app, list(args), env={"COLUMNS": "200"})
    return result


def text(result: Result) -> str:
    return ANSI.sub("", result.output)


def assert_ok(result: Result) -> str:
    assert result.exit_code == 0, f"exit {result.exit_code}\n{result.output}\n{result.exception!r}"
    return text(result)


def assert_fails(result: Result, *fragments: str) -> str:
    """Exit code 1 through the CLI's own error path (not a crash), with every fragment in the message."""
    assert result.exit_code == 1, f"exit {result.exit_code}\n{result.output}"
    assert isinstance(result.exception, SystemExit), f"unexpected crash: {result.exception!r}"
    message = ANSI.sub("", result.stderr)
    assert message.startswith("error: "), message
    for fragment in fragments:
        assert fragment in message, f"{fragment!r} not in {message!r}"
    return message


def first_tokens(help_text: str) -> set[str]:
    """The first word of every help line, box-drawing characters stripped: the command names of a Commands panel."""
    tokens = set()
    for line in help_text.splitlines():
        stripped = line.strip(BOX)
        if stripped:
            tokens.add(stripped.split()[0])
    return tokens


def has_option(help_text: str, option: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(option)}(?![\w-])", help_text) is not None


def command_at(*path: str) -> Any:
    """The command behind ``taskdistill <path>``, read from Typer's command tree rather than rendered help."""
    command: Any = typer.main.get_command(app)
    for name in path:
        assert isinstance(command, TyperGroup), f"{command.name} has no subcommands"
        assert name in command.commands, f"no command {name!r} under {command.name}"
        command = command.commands[name]
    return command


def subcommand_names(*path: str) -> set[str]:
    command = command_at(*path)
    assert isinstance(command, TyperGroup), f"{command.name} has no subcommands"
    return set(command.commands)


def option_names(*path: str) -> set[str]:
    return {opt for param in command_at(*path).params for opt in (*param.opts, *param.secondary_opts)}


def route_every_request(http: respx.MockRouter) -> respx.Route:
    """Answer any request with 503. Unrouted requests raise without being recorded; routed ones land in ``calls``."""
    return http.route().mock(return_value=httpx.Response(503, text="unavailable"))


def init_task(name: str = TASK, task_type: str = "classification") -> Path:
    assert_ok(invoke("init", name, "--type", task_type))
    return Path.cwd() / "tasks" / name


def write_jsonl(path: Path, rows: list[dict[str, Any] | None]) -> Path:
    """One JSON object per line; ``None`` writes a blank line."""
    lines = ["" if row is None else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def chat_pair(message: str, label: str) -> dict[str, Any]:
    request = {
        "model": "vendor-a/instruct-small",
        "messages": [
            {"role": "system", "content": "Classify the support message into one label."},
            {"role": "user", "content": message},
        ],
        "temperature": 0,
        "max_tokens": 24,
    }
    response = {
        "id": f"cmpl-{label}",
        "object": "chat.completion",
        "model": "vendor-a/instruct-small",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": label}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 31, "completion_tokens": 1, "total_tokens": 32},
    }
    return {"request": request, "response": response}


# ------------------------------------------------------------------------------------------- version, help
def test_version() -> None:
    assert assert_ok(invoke("--version")) == f"taskdistill {__version__}\n"


def test_help_lists_every_command() -> None:
    assert set(TOP_LEVEL_COMMANDS) <= subcommand_names()
    assert {"bakeoff", "record"} <= subcommand_names("teacher")
    assert "refresh" in subcommand_names("pricing")

    root = first_tokens(assert_ok(invoke("--help")))
    assert set(TOP_LEVEL_COMMANDS) <= root, sorted(set(TOP_LEVEL_COMMANDS) - root)
    assert {"bakeoff", "record"} <= first_tokens(assert_ok(invoke("teacher", "--help")))
    assert "refresh" in first_tokens(assert_ok(invoke("pricing", "--help")))


def test_no_arguments_shows_help() -> None:
    result = invoke()
    assert result.exit_code in (0, 2)
    assert set(TOP_LEVEL_COMMANDS) <= first_tokens(text(result))


@pytest.mark.parametrize("command", MONEY_COMMANDS, ids="-".join)
def test_money_spending_commands_take_yes_and_max_usd(command: tuple[str, ...]) -> None:
    assert {"--yes", "--max-usd"} <= option_names(*command)
    help_text = assert_ok(invoke(*command, "--help"))
    assert has_option(help_text, "--yes"), help_text
    assert has_option(help_text, "--max-usd"), help_text


# -------------------------------------------------------------------------------------------------- init
def test_init_creates_task_and_refuses_to_overwrite() -> None:
    out = assert_ok(invoke("init", TASK, "--type", "classification"))
    directory = Path.cwd() / "tasks" / TASK
    assert sorted(p.name for p in directory.iterdir()) == ["labels.txt", "task.yaml", "teacher_prompt.md"]
    assert f"tasks/{TASK}/" in out
    assert "labels.txt" in out

    prompt = directory / "teacher_prompt.md"
    prompt.write_text("Edited prompt.\n", encoding="utf-8")
    assert_fails(invoke("init", TASK, "--type", "classification"), "already exists", "--force")
    assert prompt.read_text(encoding="utf-8") == "Edited prompt.\n"

    assert_ok(invoke("init", TASK, "--type", "classification", "--force"))
    assert prompt.read_text(encoding="utf-8") != "Edited prompt.\n"


def test_init_extraction_writes_schema() -> None:
    assert_ok(invoke("init", "orders", "--type", "extraction"))
    directory = Path.cwd() / "tasks" / "orders"
    assert sorted(p.name for p in directory.iterdir()) == ["schema.json", "task.yaml", "teacher_prompt.md"]
    assert json.loads((directory / "schema.json").read_text(encoding="utf-8"))["type"] == "object"


def test_init_unknown_type_names_the_key() -> None:
    assert_fails(invoke("init", TASK, "--type", "regression"), "type", "regression")
    assert not (Path.cwd() / "tasks" / TASK).exists()


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        ("target: 0.97", 'target: "high"', "cascade.target"),
        ("metric: agreement", "metric: agreement\n  thresold: 0.5", "cascade.thresold"),
        ("max_tokens: 24", "max_tokens: plenty", "teacher.max_tokens"),
    ],
    ids=["wrong-type", "unknown-key", "teacher-key"],
)
def test_broken_task_yaml_names_the_offending_key(old: str, new: str, key: str) -> None:
    spec_file = init_task() / "task.yaml"
    source = spec_file.read_text(encoding="utf-8")
    assert source.count(old) == 1
    spec_file.write_text(source.replace(old, new), encoding="utf-8")
    message = assert_fails(invoke("curate", "--task", TASK), key)
    assert "task.yaml" in message


def test_unknown_task_says_how_to_create_it() -> None:
    assert_fails(invoke("curate", "--task", "missing"), "missing", "taskdistill init missing")


# ----------------------------------------------------------------------------------------------- capture
@pytest.mark.parametrize("fmt", ["pairs", "inputs"])
def test_capture_import_prints_summary_and_writes_json(fmt: str, workspace: Path) -> None:
    init_task()
    rows: list[dict[str, Any] | None] = [
        {"input": "My parcel has not arrived after ten days.", "gold": "delivery", "meta": {"split": "train"}},
        {"input": "I was charged twice for order 1042.", "meta": {"split": "valid"}},
        None,
        {"input": "How do I send back a jacket that does not fit?", "gold": "returns"},
    ]
    if fmt == "pairs":
        outputs = iter(["delivery", "billing", "returns"])
        rows = [None if row is None else {**row, "output": next(outputs)} for row in rows]
    source = write_jsonl(Path.cwd() / f"{fmt}.jsonl", rows)

    out = assert_ok(invoke("capture", "--task", TASK, "--import", str(source), "--format", fmt))
    assert f"imported 3 of 3 rows ({fmt}) into task {TASK}" in out
    assert f"{TASK}/last_import.json" in out

    summary = json.loads((workspace / TASK / "last_import.json").read_text(encoding="utf-8"))
    assert summary == {"task": TASK, "file": f"{fmt}.jsonl", "read": 3, "imported": 3, "blank": 1, "format": fmt}
    stored = list(Store().iter_imports(TASK))
    assert [row.input for row in stored] == [row["input"] for row in rows if row is not None]


def test_capture_import_bad_line_names_it() -> None:
    source = write_jsonl(Path.cwd() / "bad.jsonl", [{"input": "Where is my refund?"}, {"text": "no input key"}])
    assert_fails(invoke("capture", "--task", TASK, "--import", str(source), "--format", "inputs"), "bad.jsonl:2")
    assert list(Store().iter_imports(TASK)) == []


def test_capture_export_writes_openai_jsonl() -> None:
    pairs = [
        chat_pair("My parcel has not arrived after ten days.", "delivery"),
        chat_pair("I was charged twice for order 1042.", "billing"),
    ]
    source = write_jsonl(Path.cwd() / "traffic.jsonl", [*pairs])
    out = assert_ok(invoke("capture", "--task", TASK, "--import", str(source), "--format", "openai"))
    assert "imported 2 of 2 rows (openai)" in out

    target = Path.cwd() / "exported" / "traffic.jsonl"
    out = assert_ok(invoke("capture", "--task", TASK, "--export", str(target)))
    assert "exported 2 captured requests" in out
    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == pairs

    out = assert_ok(invoke("capture", "--task", "copy", "--import", str(target), "--format", "openai"))
    assert "imported 2 of 2 rows (openai) into task copy" in out


def test_capture_import_and_export_are_exclusive(tmp_path: Path) -> None:
    source = write_jsonl(tmp_path / "in.jsonl", [{"input": "Where is my refund?"}])
    result = invoke("capture", "--task", TASK, "--import", str(source), "--export", str(tmp_path / "out.jsonl"))
    assert_fails(result, "--import", "--export")


# ------------------------------------------------------------------------------------------------ budget
def test_budget_on_empty_workspace(workspace: Path) -> None:
    out = assert_ok(invoke("budget"))
    assert "total $0.0000 of cap $5.00 (0 calls)" in out

    target = Path.cwd() / "spend" / "budget.json"
    out = assert_ok(invoke("budget", "--json", str(target)))
    assert f"wrote {target}" in out
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["total"] == 0
    assert data["calls"] == 0
    assert data["by_task"] == {}
    assert data["cap"] == pytest.approx(5.0)
    assert (workspace / "ledger.sqlite").is_file()


def test_budget_cap_follows_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_BUDGET_USD", "2.5")
    assert "total $0.0000 of cap $2.50 (0 calls)" in assert_ok(invoke("budget"))


# -------------------------------------------------------------------------------------------------- demo
def test_demo_unknown_name(workspace: Path) -> None:
    assert_fails(invoke("demo", "weather"), "unknown demo 'weather'", "banking77", "invoices")
    assert not (workspace / "weather").exists()


def test_demo_unknown_until_stage(workspace: Path) -> None:
    assert_fails(invoke("demo", "invoices", "--until", "bogus"), "--until", "curate", "report")
    assert not (workspace / "invoices").exists()


def test_demo_unknown_profile() -> None:
    assert_fails(invoke("demo", "invoices", "--profile", "huge"), "--profile", "quick", "full")


# ------------------------------------------------------------------------------------ eval, report, train
def test_eval_rejects_unknown_split() -> None:
    init_task()
    assert_fails(invoke("eval", "--task", TASK, "--split", "bogus", "--backend", "torch"), "--split", "test", "valid")


def test_report_without_evaluation() -> None:
    init_task()
    assert_fails(invoke("report", "--task", TASK), "no test evaluation", f"taskdistill eval --task {TASK}")
    assert not (Path.cwd() / "reports").exists()


def test_report_rejects_bad_since() -> None:
    init_task()
    assert_fails(invoke("report", "--task", TASK, "--from-serve-log", "--since", "soon"), "duration", "24h")


def test_train_without_curated_data(workspace: Path) -> None:
    """With the torch extra installed the missing data is reported; without it, the missing extra comes first."""
    init_task()
    result = invoke("train", "--task", TASK, "--backend", "torch")
    if torch_unavailable_reason() is None:
        assert_fails(result, "no curated data", "train.jsonl", f"taskdistill curate --task {TASK}")
    else:
        assert_fails(result, "torch training is not installed", "taskdistill[torch]")
    assert not (workspace / TASK / "runs").exists()


def test_train_torch_backend_without_torch(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    init_task()
    # Import the trainer while torch is visible, so no module imported under the patch remembers it as missing.
    importlib.import_module("taskdistill.train.runner")
    for name in ("torch", "peft"):
        monkeypatch.setitem(sys.modules, name, None)
    assert torch_unavailable_reason() is not None
    assert_fails(
        invoke("train", "--task", TASK, "--backend", "torch"),
        "torch training is not installed",
        "peft",
        "taskdistill[torch]",
    )
    assert not (workspace / TASK / "runs").exists()


def test_train_rejects_unknown_backend() -> None:
    init_task()
    assert_fails(invoke("train", "--task", TASK, "--backend", "tpu"), "unknown backend 'tpu'")


# ----------------------------------------------------------------------------------------------- teacher
def test_bakeoff_rejects_free_variant(http: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a key set, the model lookup and teacher calls would go out; the rejection must come before them."""
    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", TEST_API_KEY)
    route_every_request(http)
    init_task()
    result = invoke("teacher", "bakeoff", "--task", TASK, "--models", "vendor-a/instruct-small:free", "--yes")
    assert not http.calls, [str(call.request.url) for call in http.calls]
    assert_fails(result, "vendor-a/instruct-small:free", "':free' variants are rejected")


def test_bakeoff_rejects_free_variant_among_paid_ones(http: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKDISTILL_TEACHER_API_KEY", TEST_API_KEY)
    route_every_request(http)
    init_task()
    models = "vendor-b/instruct-large,vendor-a/instruct-small:free@host-a/fp8"
    result = invoke("teacher", "bakeoff", "--task", TASK, "--models", models)
    assert not http.calls, [str(call.request.url) for call in http.calls]
    assert_fails(result, "vendor-a/instruct-small:free", "':free' variants are rejected")


# ----------------------------------------------------------------------------------------------- pricing
def test_pricing_refresh_writes_snapshot(http: respx.MockRouter, workspace: Path) -> None:
    route = http.get(OPENROUTER_MODELS).mock(return_value=httpx.Response(200, json=MODELS_PAYLOAD))
    out = assert_ok(invoke("pricing", "refresh"))
    assert route.call_count == 1
    assert "authorization" not in route.calls.last.request.headers

    data = json.loads((workspace / "pricing.json").read_text(encoding="utf-8"))
    assert f"pricing snapshot {data['date']}: 2 models -> pricing.json" in out
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["date"])
    assert data["source"] == OPENROUTER_MODELS
    assert sorted(data["models"]) == ["vendor-a/instruct-small", "vendor-b/instruct-large"]
    assert data["models"]["vendor-b/instruct-large"]["default"] == pytest.approx(
        {"prompt": 2e-6, "completion": 8e-6, "request": 0.001}
    )
    assert data["aliases"] == {"vendor-a/instruct-small-0601": "vendor-a/instruct-small"}


def test_pricing_refresh_with_provider_endpoints(http: respx.MockRouter, workspace: Path) -> None:
    http.get(OPENROUTER_MODELS).mock(return_value=httpx.Response(200, json=MODELS_PAYLOAD))
    endpoints = {
        "data": {
            "endpoints": [
                {"tag": "host-a/fp8", "pricing": {"prompt": "0.0000002", "completion": "0.0000005"}},
                {"tag": "host-b", "pricing": {"prompt": "0.0000003", "completion": "0.0000006"}},
            ]
        }
    }
    route = http.get(f"{OPENROUTER_MODELS}/vendor-a/instruct-small/endpoints").mock(
        return_value=httpx.Response(200, json=endpoints)
    )
    assert_ok(invoke("pricing", "refresh", "--models", "vendor-a/instruct-small"))
    assert route.call_count == 1
    providers = json.loads((workspace / "pricing.json").read_text(encoding="utf-8"))["models"][
        "vendor-a/instruct-small"
    ]["providers"]
    assert sorted(providers) == ["host-a/fp8", "host-b"]
    assert providers["host-a/fp8"]["prompt"] == pytest.approx(2e-7)


def test_pricing_refresh_http_error(http: respx.MockRouter, workspace: Path) -> None:
    http.get(OPENROUTER_MODELS).mock(return_value=httpx.Response(503, text="unavailable"))
    assert_fails(invoke("pricing", "refresh"), "HTTP 503")
    assert not (workspace / "pricing.json").exists()
