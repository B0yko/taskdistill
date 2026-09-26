"""Offline replay: the packaged recordings hold every request the demos send, under one request key everywhere.

The spec's request-key tests: the cache, the replay and proxy capture compute the same key; a manifest mismatch is
refused; replay coverage is 100% for both demos in both profiles. The coverage test runs the demo's own data,
capture and serve stages with only their network edges stubbed, so the requests come from the demo's code, not from
a copy of it; the end-to-end tests run the real servers. The Banking77 cases download the pinned CSVs from GitHub
(public data, never a paid API) once per session. No test here has a teacher API key.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from taskdistill.capture.proxy import create_proxy_app
from taskdistill.cli import app
from taskdistill.config import TaskSpec, load_task
from taskdistill.curate.io import raw_inputs_by_hash
from taskdistill.demos import banking77, invoices
from taskdistill.demos import runner as demo_runner
from taskdistill.demos.runner import SMOKE_QUERIES, DemoContext, DemoData
from taskdistill.ledger import Ledger
from taskdistill.serve import runner as serve_runner
from taskdistill.store import Store
from taskdistill.teacher import factory
from taskdistill.teacher.base import ManifestMismatch, TeacherResult
from taskdistill.teacher.cache import ResponseCache
from taskdistill.teacher.client import LiveTeacher
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.replay import (
    Recording,
    ReplayTeacher,
    check_manifest,
    expected_manifest,
    load_recording,
    write_recording,
)
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*httpx2.*")
    from fastapi.testclient import TestClient

KEY_ENV = ("OPENROUTER_API_KEY", "TASKDISTILL_TEACHER_API_KEY")
#: Variables the bundled specs expand; unset, the specs use the values the recordings were made with.
SPEC_ENV = ("TASKDISTILL_TEACHER_MODEL", "TASKDISTILL_TEACHER_BASE_URL")
TEACHER_URL = "https://teacher.example.com/api/v1"
#: The spec's dataset sizes: Banking77 has 10,003 train and 3,080 test queries, the invoices 30 templates x 100.
FULL_ROWS = {"banking77": 13_083, "invoices": 3_000}
#: The quick-profile subsets the demo modules state.
QUICK_ROWS = {"banking77": banking77.QUICK_SIZES, "invoices": invoices.QUICK_SIZES}

fetches_banking77 = pytest.mark.network  # downloads the pinned Banking77 CSVs


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No teacher key, the bundled specs' defaults, an empty workspace and a working directory without tasks/."""
    for name in (*KEY_ENV, *SPEC_ENV):
        monkeypatch.delenv(name, raising=False)
    workspace = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    monkeypatch.chdir(tmp_path)
    return workspace


@pytest.fixture(scope="session")
def dataset_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A workspace shared by the session, so the pinned Banking77 CSVs are downloaded at most once."""
    return tmp_path_factory.mktemp("replay-datasets")


@pytest.fixture(scope="session")
def demo_data(dataset_home: Path) -> Callable[[str, str], DemoData]:
    """``load_demo_data(name, profile)``, memoised for the session and cached in ``dataset_home``."""
    loaded: dict[tuple[str, str], DemoData] = {}

    def get(name: str, profile: str) -> DemoData:
        if (name, profile) not in loaded:
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("TASKDISTILL_HOME", str(dataset_home))
                loaded[name, profile] = demo_runner.load_demo_data(name, profile)
        return loaded[name, profile]

    return get


def seed_datasets(name: str, home: Path, dataset_home: Path) -> None:
    """Copy the session's downloaded dataset into ``home``, so the demo reads it from its cache."""
    if name == "banking77":
        shutil.copytree(dataset_home / "_datasets" / "banking77", home / "_datasets" / "banking77")


def completion(model: str, output: str | None) -> dict[str, Any]:
    return {
        "id": "gen-key-agreement",
        "object": "chat.completion",
        "created": 1790000000,
        "model": model,
        "provider": "Example Host",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 80, "cost": 2.5e-4},
    }


@contextlib.contextmanager
def unstarted_server(app: Any, port: int) -> Iterator[str]:
    """Stands in for ``background_server``: the URL only; respx answers what is sent to it."""
    yield f"http://127.0.0.1:{port}"


def demo_traffic(
    ctx: DemoContext, recording: Recording, monkeypatch: pytest.MonkeyPatch
) -> dict[str, list[dict[str, Any]]]:
    """Every teacher request the demo sends, by origin, taken from the demo's own stages in the current workspace.

    ``stage_data`` imports every split with the importer; ``stage_capture`` sends the training inputs and
    ``stage_serve`` the smoke queries, and writes ``request.json``. No server starts: respx takes each request the
    application sends and answers from the recording. Curate labels from the raw inputs it reads back from the store
    (``raw_inputs_by_hash``); all imported inputs are a superset of the ones it labels.
    """
    spec = ctx.spec
    sent: dict[str, list[dict[str, Any]]] = {"capture": [], "smoke test": []}

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent["capture" if request.url.path.startswith(f"/t/{spec.task}/") else "smoke test"].append(body)
        record = recording.get(request_key(body)) or {"output": ""}
        return httpx.Response(
            200, json=completion(spec.teacher.model, record["output"]), headers={"x-taskdistill-route": "teacher"}
        )

    monkeypatch.setattr(demo_runner, "background_server", unstarted_server)
    monkeypatch.setattr(serve_runner, "build_server", lambda *args, **kwargs: (None, []))
    demo_runner.stage_data(ctx)
    with respx.mock() as mock:
        mock.post(path__regex=r"/chat/completions$").mock(side_effect=answer)
        demo_runner.stage_capture(ctx)
        demo_runner.stage_serve(ctx)
    raw_inputs = raw_inputs_by_hash(Store(), spec)
    return {
        "capture": sent["capture"],
        "labelling": [build_teacher_request(spec, text) for text in raw_inputs.values()],
        "smoke test": sent["smoke test"],
        "README request": [json.loads(Path("request.json").read_text(encoding="utf-8"))],
    }


# replay coverage ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "profile"),
    [
        pytest.param("banking77", "quick", marks=fetches_banking77),
        pytest.param("banking77", "full", marks=fetches_banking77),
        pytest.param("invoices", "quick"),
        pytest.param("invoices", "full"),
    ],
)
def test_the_packaged_recording_covers_every_request_the_demo_sends(
    name: str,
    profile: str,
    home: Path,
    dataset_home: Path,
    demo_data: Callable[[str, str], DemoData],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_task(name)
    source = factory.find_recording(name)
    assert source is not None
    assert source == factory.packaged_recording(name)
    recording = load_recording(source)
    check_manifest(recording, spec)
    ReplayTeacher(recording, spec)
    assert recording.manifest["records"] == len(recording.records) == FULL_ROWS[name]

    quick_test = {row["input"] for row in demo_data(name, "quick").records["test"]}
    seed_datasets(name, home, dataset_home)
    ctx = DemoContext(
        name=name, profile=profile, spec=spec, mode=factory.resolve_mode(spec, None), yes=False, max_usd=None,
        backend="torch", base=None, seed=None, log=lambda line: None,
    )  # fmt: skip
    assert ctx.mode == "replay"
    traffic = demo_traffic(ctx, recording, monkeypatch)

    imported = Counter(str(row.meta.get("split")) for row in Store().iter_imports(spec.task))
    if profile == "full":
        assert imported.total() == FULL_ROWS[name]
    else:
        assert imported == QUICK_ROWS[name]
    capture_keys = {request_key(body) for body in traffic["capture"]}
    assert len(traffic["capture"]) == len(capture_keys) == imported["train"]
    assert 0 < len(traffic["labelling"]) <= imported.total()
    assert ctx.data is not None
    assert len(ctx.data.smoke) == SMOKE_QUERIES
    assert [body["messages"][-1]["content"] for body in traffic["smoke test"]] == ctx.data.smoke
    assert set(ctx.data.smoke) <= quick_test
    assert request_key(traffic["README request"][0]) == request_key(traffic["smoke test"][0])

    for origin, bodies in traffic.items():
        missing = [body["messages"][-1]["content"] for body in bodies if request_key(body) not in recording]
        assert not missing, (
            f"{name} {profile}: {len(missing)} of {len(bodies)} {origin} requests are not in the recording, "
            f"e.g. {missing[0][:80]!r}"
        )


@pytest.mark.parametrize("name", [pytest.param("banking77", marks=fetches_banking77), "invoices"])
def test_a_demo_re_recording_holds_exactly_the_shipped_requests(
    name: str, demo_data: Callable[[str, str], DemoData], dataset_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``teacher record --demo`` records ``demo_requests``: one request per full-profile row, the shipped key set.

    With the coverage test above, a re-recording therefore keeps every request of both profiles.
    """
    demo_data(name, "full")
    monkeypatch.setenv("TASKDISTILL_HOME", str(dataset_home))
    spec = load_task(name)
    recorded = [request_key(body) for body in demo_runner.demo_requests(name, spec)]
    assert len(recorded) == len(set(recorded)) == FULL_ROWS[name]
    assert set(recorded) == set(load_recording(factory.packaged_recording(name)).records)


# end to end -----------------------------------------------------------------------------------------
@pytest.mark.network  # the student tokenizer for the length filter, and the Banking77 CSVs
@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "profile"),
    [("invoices", "quick"), ("invoices", "full"), ("banking77", "quick")],
)
def test_demo_until_curate_runs_on_the_recording_alone(
    name: str,
    profile: str,
    home: Path,
    dataset_home: Path,
    demo_data: Callable[[str, str], DemoData],
) -> None:
    data = demo_data(name, profile)
    seed_datasets(name, home, dataset_home)

    args = ["demo", name, "--profile", profile, "--until", "curate", "--backend", "torch"]
    result = CliRunner().invoke(app, args, catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "teacher replay" in result.output
    stats = json.loads((home / name / "data" / "curate_stats.json").read_text(encoding="utf-8"))
    labelling = stats["labelling"]
    assert labelling["mode"] == "replay"
    assert stats["teacher"]["mode"] == "replay"
    assert labelling["requested"] > 0
    assert labelling["replayed"] == labelling["requested"]
    assert labelling["live"] == 0
    assert labelling["cached"] == 0
    assert labelling["cost_usd"] == 0

    recording = load_recording(factory.packaged_recording(name))
    captures = list(Store(home / "store.sqlite").iter_captures(name))
    assert len(captures) == len(data.records["train"])
    assert all(row.captured for row in captures)
    assert all(row.request_key in recording for row in captures)
    labelling_keys = (home / name / "data" / "labelling_keys.txt").read_text(encoding="utf-8").split()
    assert labelling_keys
    assert set(labelling_keys) <= set(recording.records)
    timing = json.loads((home / name / f"demo_timing_{profile}.json").read_text(encoding="utf-8"))
    assert timing["mode"] == "replay"


def test_the_demo_refuses_a_recording_made_for_another_teacher(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = load_recording(factory.packaged_recording("invoices")).manifest["teacher_model"]
    monkeypatch.setenv("TASKDISTILL_TEACHER_MODEL", "vendor/other-model")

    result = CliRunner().invoke(app, ["demo", "invoices", "--until", "capture", "--backend", "torch"])

    assert result.exit_code == 1
    assert f"recording manifest mismatch: teacher_model is {recorded!r}" in result.output
    assert "'vendor/other-model'" in result.output
    assert list(Store(home / "store.sqlite").iter_captures("invoices")) == []


# request key agreement ------------------------------------------------------------------------------
async def complete_live(spec: TaskSpec, cache: ResponseCache, workspace: Path, body: dict[str, Any]) -> TeacherResult:
    """One labelling call through the live client and its response cache (the only writer of the cache)."""
    price = ModelPrice(1e-7, 2e-7)
    providers = {spec.teacher.provider: price} if spec.teacher.provider else {}
    pricing = PricingSnapshot(
        date="2026-09-26", source="test", models={spec.teacher.model: {"default": price, "providers": providers}}
    )
    teacher = LiveTeacher(
        TEACHER_URL,
        "sk-test-teacher-0001",
        ledger=Ledger(workspace / "ledger.sqlite", global_cap=1.0),
        pricing=pricing,
        task=spec.task,
        phase="curate-label",
        run_id="key-agreement",
        cache=cache,
    )
    try:
        return await teacher.complete(body)
    finally:
        await teacher.aclose()


def test_cache_replay_and_proxy_capture_compute_the_same_request_key(
    tmp_path: Path, demo_data: Callable[[str, str], DemoData]
) -> None:
    spec = load_task("invoices")
    body = build_teacher_request(spec, demo_data("invoices", "quick").smoke[0])  # the demo's README request
    key = request_key(body)
    packaged = load_recording(factory.packaged_recording("invoices"))
    recorded = packaged.get(key)
    assert recorded is not None
    # The application's own serialisation: other key order, indentation, ASCII escapes and "stream": false.
    wire = json.dumps({"stream": False, **dict(reversed(list(body.items())))}, indent=2).encode("utf-8")
    assert json.loads(wire) != body

    cache = ResponseCache(tmp_path / "cache.sqlite")
    store = Store(tmp_path / "store.sqlite")
    with respx.mock() as mock:
        route = mock.post(f"{TEACHER_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=completion(spec.teacher.model, recorded["output"]))
        )
        live = asyncio.run(complete_live(spec, cache, tmp_path, body))
        again = asyncio.run(complete_live(spec, cache, tmp_path, json.loads(wire)))
        with TestClient(create_proxy_app(store, TEACHER_URL, default_task=spec.task)) as client:
            proxied = client.post(
                f"/t/{spec.task}/v1/chat/completions",
                content=wire,
                headers={"Authorization": "Bearer sk-test-application-0001", "Content-Type": "application/json"},
            )
        assert route.call_count == 2  # the live call and the proxied one; the second labelling call hit the cache

    assert live.source == "live"
    assert again.source == "cache"
    cached = cache.get(key)
    assert cached is not None

    assert proxied.status_code == 200
    (capture,) = list(store.iter_captures(spec.task))
    assert capture.captured
    assert capture.request_body == wire.decode("utf-8")

    path = write_recording(
        tmp_path / "teacher_recording.jsonl.gz", expected_manifest(spec, "2026-09-26"), cache.iter_results()
    )
    replayed = asyncio.run(ReplayTeacher(path, spec).complete(json.loads(wire)))
    shipped = asyncio.run(ReplayTeacher(packaged, spec).complete(json.loads(capture.request_body or "")))

    assert live.key == again.key == cached.key == capture.request_key == replayed.key == shipped.key == key
    assert replayed.output == shipped.output == live.output == recorded["output"]


# manifest mismatch ----------------------------------------------------------------------------------
def with_teacher(spec: TaskSpec, **update: Any) -> TaskSpec:
    return spec.model_copy(update={"teacher": spec.teacher.model_copy(update=update)})


def with_extra_body(spec: TaskSpec, **update: Any) -> TaskSpec:
    return with_teacher(spec, extra_body={**spec.teacher.extra_body, **update})


MISMATCHES = [
    pytest.param(lambda s: with_teacher(s, model="vendor/other-model"), "teacher_model", id="teacher-model"),
    pytest.param(
        lambda s: with_extra_body(s, provider={**s.teacher.extra_body["provider"], "order": ["otherhost/bf16"]}),
        "provider",
        id="provider",
    ),
    pytest.param(
        lambda s: s.model_copy(update={"teacher_prompt": s.teacher_prompt + "\nAnswer briefly.\n"}),
        "teacher_prompt_sha256",
        id="prompt",
    ),
    pytest.param(lambda s: with_teacher(s, temperature=0.7), "generation.temperature", id="temperature"),
    pytest.param(lambda s: with_teacher(s, max_tokens=s.teacher.max_tokens + 1), "generation.max_tokens", id="max"),
    pytest.param(
        lambda s: with_extra_body(s, reasoning={"enabled": True}),
        "generation.extra_body.reasoning.enabled",
        id="reasoning",
    ),
]


@pytest.mark.parametrize("name", ["banking77", "invoices"])
@pytest.mark.parametrize(("change", "field"), MISMATCHES)
def test_a_manifest_mismatch_is_refused_and_names_the_field(
    name: str, change: Callable[[TaskSpec], TaskSpec], field: str
) -> None:
    spec = load_task(name)
    recording = load_recording(factory.packaged_recording(name))
    check_manifest(recording, spec)
    other = change(spec)
    assert spec.teacher.model == recording.manifest["teacher_model"]  # the change did not leak into the bundled spec

    pattern = rf"recording manifest mismatch: {re.escape(field)} is "
    with pytest.raises(ManifestMismatch, match=pattern):
        check_manifest(recording, other)
    with pytest.raises(ManifestMismatch, match=pattern):
        ReplayTeacher(recording, other)
    with pytest.raises(ManifestMismatch, match=pattern):
        factory.make_teacher(other, mode="replay", phase="curate-label", run_id="mismatch")
