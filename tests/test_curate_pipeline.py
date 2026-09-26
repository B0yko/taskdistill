"""run_curate end to end: proxy-captured outputs joined with imported gold, replayed labelling of the rest, the
files written under the workspace, the spend gate, the leakage guard and the 13k-example time budget."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import random
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from taskdistill.capture.importer import import_file
from taskdistill.capture.proxy import create_proxy_app
from taskdistill.config import TaskSpec
from taskdistill.curate import pipeline
from taskdistill.curate.extract import input_hash
from taskdistill.curate.io import CuratedDataError, raw_inputs_by_hash, read_split
from taskdistill.curate.pipeline import STAGES, CurateError, run_curate
from taskdistill.ledger import Ledger
from taskdistill.store import Store
from taskdistill.tasks.extraction import canonical_output
from taskdistill.teacher.cache import ResponseCache
from taskdistill.teacher.client import LiveTeacher, SpendNotConfirmed
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.replay import Recording, ReplayTeacher, expected_manifest
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request
from test_curate_stages import (
    SCHEMA,
    TEACHER_MODEL,
    FakeTeacher,
    WordTokenizer,
    add_capture,
    completion,
    make_spec,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*httpx2.*")
    from fastapi.testclient import TestClient

UPSTREAM = "http://upstream.test/v1"

# (text, gold, teacher answer as the upstream phrases it)
TRAIN = [
    ("Where is my new card? It has not arrived yet.", "card_arrival", "card arrival"),
    ("My card still has not been delivered after ten days", "card_arrival", "Card Arrival"),
    ("I think someone stole my card at the station", "lost_or_stolen_card", "card arrival"),
    ("I lost my card yesterday on the bus to work", "lost_or_stolen_card", "lost or stolen card"),
    ("What exchange rate do you apply for dollars?", "exchange_rate", "exchange rate"),
    ("Why did my top up fail again this morning?", "top_up_failed", "top up failed"),
    ("My top up was declined, what happened?", "top_up_failed", "Label: top_up_failed"),
    ("Please call me on 555-0142 about my missing card", "card_arrival", "card arrival"),
    ("How do I get euros at a good rate?", "exchange_rate", "exchange rate"),
]
VALID = [
    ("I lost my card yesterday on the bus to work.", "lost_or_stolen_card"),
    ("Is the rate for pounds fixed today?", "exchange_rate"),
    ("The top up from my bank card keeps failing", "top_up_failed"),
]
TEST = [
    ("How do I get euros at a good rate?", "exchange_rate"),
    ("What is the exchange rate?", "exchange_rate"),
    ("WHAT IS THE EXCHANGE RATE?", "exchange_rate"),
    ("Tell me when my card will arrive", "card_arrival"),
    ("Email ops@example.com: my card was stolen", "lost_or_stolen_card"),
]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(path))
    return path


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.sqlite")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay_for(spec: TaskSpec, answers: dict[str, str]) -> ReplayTeacher:
    """A real ReplayTeacher over an in-memory recording answering the given raw inputs."""
    records = {}
    for text, output in answers.items():
        key = request_key(build_teacher_request(spec, text))
        records[key] = {
            "key": key,
            "output": output,
            "usage": {"prompt_tokens": 40, "completion_tokens": 3, "cost": 2e-5},
            "latency_ms": 321.0,
            "provider": "Alpha",
            "finish_reason": "stop",
            "timestamp": 1790000000.0,
        }
    return ReplayTeacher(Recording(manifest=expected_manifest(spec), records=records), spec)


@contextlib.contextmanager
def proxy(store: Store, answers: dict[str, str]) -> Iterator[TestClient]:
    """The real capture proxy in front of a respx upstream that answers by the user message."""

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=completion(answers[body["messages"][-1]["content"]]))

    http = httpx.AsyncClient()
    try:
        with respx.mock(base_url=UPSTREAM, assert_all_called=False) as router:
            router.post("/chat/completions").mock(side_effect=upstream)
            with TestClient(create_proxy_app(store, UPSTREAM, client=http)) as client:
                yield client
    finally:
        asyncio.run(http.aclose())


def classification_run(
    store: Store, tmp_path: Path, **kwargs: Any
) -> tuple[TaskSpec, pipeline.CurateResult, list[str], ReplayTeacher]:
    spec = make_spec()
    with proxy(store, {text: answer for text, _, answer in TRAIN}) as client:
        for text, _, _ in TRAIN:
            resp = client.post(
                f"/t/{spec.task}/v1/chat/completions",
                json=build_teacher_request(spec, text),
                headers={"Authorization": "Bearer sk-test-client"},
            )
            assert resp.status_code == 200
    rows = [{"input": t, "gold": g, "meta": {"split": "train"}} for t, g, _ in TRAIN]
    rows += [{"input": t, "gold": g, "meta": {"split": "validation"}} for t, g in VALID]
    rows += [{"input": t, "gold": g, "meta": {"split": "test"}} for t, g in TEST]
    import_file(store, spec.task, write_jsonl(tmp_path / "gold.jsonl", rows), "inputs")
    teacher = replay_for(spec, {t: g.replace("_", " ") for t, g in VALID + TEST})
    lines: list[str] = []
    result = run_curate(spec, store=store, teacher=teacher, tokenizer=WordTokenizer(), log=lines.append, **kwargs)
    return spec, result, lines, teacher


def test_classification_end_to_end(store: Store, home: Path, tmp_path: Path) -> None:
    spec, result, lines, teacher = classification_run(store, tmp_path)
    data = home / spec.task / "data"

    stage_lines = [line for line in lines if line.startswith("[")]
    assert [line.split("]")[0] for line in stage_lines] == [f"[{k}/11" for k in range(1, 12)]
    assert [line.split(": ")[0].split("] ")[1] for line in stage_lines] == [title for _, title in STAGES]
    assert list(result.stats["stages"]) == [key for key, _ in STAGES]

    stats = json.loads((data / "curate_stats.json").read_text())
    assert stats["splits"] == {"train": 7, "valid": 3, "test": 4}
    assert stats["stages"]["merge"]["cross_split_merged"] == 1
    assert stats["stages"]["merge"]["with_both"] == len(TRAIN)
    assert stats["cross_split"]["removed"] == {"train": 1, "valid": 0, "test": 0}
    assert stats["cross_split"]["pairs"]["train/valid"] == {"exact": 0, "near": 1}
    assert stats["dedupe"]["splits"]["test"] == {**stats["dedupe"]["splits"]["test"], "before": 5, "after": 4}
    assert stats["dedupe"]["removed_exact"] == 1
    assert stats["pii"]["hits"]["phone"] == 1
    assert stats["pii"]["hits"]["email"] == 1
    assert stats["labelling"]["requested"] == 6
    assert stats["labelling"]["replayed"] == 6
    assert stats["labelling"]["mode"] == "replay"
    assert stats["leakage"]["ok"] is True
    assert stats["teacher"] == {
        "model": TEACHER_MODEL,
        "provider": "alpha/fp8",
        "prompt_sha256": spec.teacher_prompt_sha256,
        "temperature": 0.0,
        "max_tokens": 24,
        "mode": "replay",
    }
    assert stats["sources"]["test"] == {"recorded": 1, "labelled": 3, "teacher_invalid": 0, "with_gold": 4}
    assert stats["split_sizes"] == {
        "split": {"train": 8, "valid": 3, "test": 5},
        "dedupe": {"train": 8, "valid": 3, "test": 4},
        "cross_split": {"train": 7, "valid": 3, "test": 4},
        "label": {"train": 7, "valid": 3, "test": 4},
        "length": {"train": 7, "valid": 3, "test": 4},
    }
    assert stats["labelling"]["recorded_cost_usd"] == pytest.approx(6 * 2e-5)
    assert stats["labelling"]["providers"] == {"Alpha": 6}
    assert stats["stages"]["load"]["captures"]["by_model"] == {TEACHER_MODEL: len(TRAIN)}
    assert stats["stages"]["load"]["captures"]["other_model"] == 0
    assert stats["stages"]["load"]["captures"]["other_prompt"] == 0
    names = [f"{s}{suffix}" for s in ("train", "valid", "test") for suffix in (".jsonl", ".meta.jsonl")]
    assert stats["files"] == {
        name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in [*names, "labelling_keys.txt"]
    }

    train = read_jsonl(data / "train.jsonl")
    train_meta = read_jsonl(data / "train.meta.jsonl")
    assert len(train) == len(train_meta) == 7
    by_text = {row["messages"][1]["content"]: (row, meta) for row, meta in zip(train, train_meta, strict=True)}
    for row, _ in by_text.values():
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert row["messages"][0]["content"] == spec.student.system_prompt
        assert row["messages"][2]["content"] in spec.labels
    # The proxy-captured teacher output is joined with the imported gold by input hash.
    row, meta = by_text["I think someone stole my card at the station"]
    assert row["messages"][2]["content"] == "card_arrival"
    assert meta == {
        "input_hash": input_hash("I think someone stole my card at the station"),
        "gold": "lost_or_stolen_card",
        "teacher": "card_arrival",
        "meta": {"split": "train"},
    }
    assert "Please call me on <PHONE> about my missing card" in by_text
    assert "How do I get euros at a good rate?" not in by_text  # moved to test with its captured output
    assert "I lost my card yesterday on the bus to work" not in by_text  # near-duplicate of a valid text

    valid_meta = read_jsonl(data / "valid.meta.jsonl")
    assert {m["meta"]["split"] for m in valid_meta} == {"valid"}
    test_rows = read_jsonl(data / "test.jsonl")
    test_meta = read_jsonl(data / "test.meta.jsonl")
    test_texts = [row["messages"][1]["content"] for row in test_rows]
    assert "How do I get euros at a good rate?" in test_texts
    assert "Email <EMAIL>: my card was stolen" in test_texts
    assert all(m["gold"] == m["teacher"] for m in test_meta)

    raw = raw_inputs_by_hash(store, spec)
    assert raw[input_hash("Please call me on 555-0142 about my missing card")].endswith(
        "555-0142 about my missing card"
    )
    labelled = [
        m for m in valid_meta + test_meta if m["input_hash"] != input_hash("How do I get euros at a good rate?")
    ]
    expected = sorted(request_key(build_teacher_request(spec, raw[m["input_hash"]])) for m in labelled)
    keys = (data / "labelling_keys.txt").read_text().split()
    assert keys == expected
    assert request_key(build_teacher_request(spec, "Email ops@example.com: my card was stolen")) in keys
    assert teacher.recording.records.keys() >= set(keys)

    card = (data / "dataset_card.md").read_text()
    assert str(home) not in card and str(tmp_path) not in card
    assert "- Source: user-provided data" in card and "- Licence: not recorded" in card
    assert f"`{TEACHER_MODEL}`" in card and "- Provider: alpha/fp8" in card
    assert spec.teacher_prompt_sha256 in card
    assert "## Label distribution (teacher labels)" in card
    assert "| train | 7 |" in card
    assert "6 replayed from a recording" in card
    assert "cost $0.0000 in this run, originally $0.0001 when the replayed or cached answers were produced" in card
    assert "labelled on 2026-09-21" in card
    assert "| cross-split removal | 7 | 3 | 4 |" in card
    assert not any(line.startswith("warning") for line in lines)
    assert set(result.paths) >= {"train", "valid_meta", "stats", "card", "labelling_keys"}


def test_read_split_returns_rows_joined_with_their_sidecar(store: Store, home: Path, tmp_path: Path) -> None:
    spec, _, _, _ = classification_run(store, tmp_path)
    rows = read_split(spec.task, "test")
    assert len(rows) == 4
    assert {row.input for row in rows} >= {"Tell me when my card will arrive", "Email <EMAIL>: my card was stolen"}
    assert all(row.target == row.teacher for row in rows)
    assert all(row.meta == {"split": "test"} for row in rows)
    with pytest.raises(CuratedDataError, match="taskdistill curate --task other"):
        read_split("other", "train")
    assert not (home / "other").exists()
    with pytest.raises(ValueError, match="unknown split"):
        read_split(spec.task, "dev")


def test_rerun_writes_identical_data(store: Store, home: Path, tmp_path: Path) -> None:
    spec, _, _, teacher = classification_run(store, tmp_path)
    data = home / spec.task / "data"
    names = [f"{s}{suffix}" for s in ("train", "valid", "test") for suffix in (".jsonl", ".meta.jsonl")]
    first = {name: (data / name).read_bytes() for name in [*names, "labelling_keys.txt"]}
    run_curate(spec, store=store, teacher=teacher, tokenizer=WordTokenizer(), log=lambda _: None)
    assert {name: (data / name).read_bytes() for name in first} == first


def test_extraction_end_to_end_with_grouped_split(store: Store, home: Path, tmp_path: Path) -> None:
    spec = make_spec("extraction", curate={"split": {"val": 0.2, "test": 0.2, "group_by": "meta.template", "seed": 3}})
    docs = []
    for t in range(6):
        for i in range(5):
            contact = f"billing{t}@example.com" if i % 2 == 0 else None
            gold = {"vendor_name": f"Vendor {t}", "invoice_number": f"T{t}-{i:03d}", "total_amount": 10.5 * (i + 1),
                    "contact": contact}  # fmt: skip
            text = (
                f"Layout {t}. Invoice T{t}-{i:03d} issued by Vendor {t} for goods delivered in week {i + 7}.\n"
                f"Amount payable: {gold['total_amount']:.2f} EUR." + (f" Questions go to {contact}." if contact else "")
            )
            docs.append((t, i, text, gold))
    pairs = []
    inputs = []
    answers = {}
    for t, i, text, gold in docs:
        meta = {"template": f"layout-{t}", "id": f"{t}-{i}"}
        if i < 3:
            output = json.dumps(dict(reversed(list(gold.items()))))
            pairs.append(
                {"input": text, "output": f"```json\n{output}\n```" if i == 0 else output, "gold": gold, "meta": meta}
            )
        else:
            inputs.append({"input": text, "gold": gold, "meta": meta})
            answers[text] = "Here you go: " + json.dumps(gold)
    import_file(store, spec.task, write_jsonl(tmp_path / "pairs.jsonl", pairs), "pairs")
    import_file(store, spec.task, write_jsonl(tmp_path / "inputs.jsonl", inputs), "inputs")

    result = run_curate(
        spec, store=store, teacher=replay_for(spec, answers), tokenizer=WordTokenizer(), log=lambda _: None
    )
    data = home / spec.task / "data"

    templates: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        rows = read_jsonl(data / f"{split}.jsonl")
        metas = read_jsonl(data / f"{split}.meta.jsonl")
        for row, meta in zip(rows, metas, strict=True):
            templates.setdefault(meta["meta"]["template"], set()).add(split)
            target = row["messages"][2]["content"]
            assert target == canonical_output(meta["teacher"], SCHEMA)
            assert json.loads(target) == meta["teacher"]
            assert list(meta["teacher"]) == list(SCHEMA["properties"])
            assert meta["teacher"] == meta["gold"]
            if "Questions go to" in row["messages"][1]["content"]:
                assert "<EMAIL>" in row["messages"][1]["content"]
                assert meta["teacher"]["contact"] == "<EMAIL>"
                assert meta["gold"]["contact"] == "<EMAIL>"
    assert all(len(splits) == 1 for splits in templates.values())
    assert len(templates) == 6
    stats = result.stats
    assert stats["splits"] == {"train": 20, "valid": 5, "test": 5}
    assert stats["stages"]["split"]["method"] == "grouped"
    assert stats["labelling"]["requested"] == 12
    assert stats["distribution"]["kind"] == "field"
    assert sum(stats["distribution"]["teacher"][s]["contact"] for s in ("train", "valid", "test")) == 18
    assert stats["pii"]["by_field"]["teacher"]["email"] == 18
    assert stats["pii"]["by_field"]["gold"]["email"] == 18
    assert stats["pii"]["by_field"]["input"]["email"] == 18
    assert stats["pii"]["examples_changed"] == 18  # distinct examples, although labelled values are scrubbed again
    assert stats["labelling"]["pii_hits"] == {"email": 6}
    card = (data / "dataset_card.md").read_text()
    assert "## Field distribution (non-null teacher values)" in card
    assert "30 rows were split by group (`meta.template`) with seed 3 (valid 0.2, test 0.2);" in card
    assert "| contact |" in card


def test_no_training_example_is_truncated(store: Store, home: Path) -> None:
    spec = make_spec(train={"max_seq_len": 40})
    texts = [f"Where is card number {i} that I ordered?" for i in range(6)]
    long = "My card " + " ".join(f"word{i}" for i in range(60))
    rows = [{"input": t, "output": "card arrival", "meta": {"split": "train"}} for t in [*texts, long]]
    store.add_imports(spec.task, "pairs", rows)
    tokenizer = WordTokenizer()

    result = run_curate(spec, store=store, teacher=None, tokenizer=tokenizer, log=lambda _: None)

    written = read_jsonl(home / spec.task / "data" / "train.jsonl")
    assert sorted(row["messages"][1]["content"] for row in written) == sorted(texts)
    for row in written:
        assert tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=False).__len__() <= 40
    assert result.stats["length"]["dropped"] == 1
    assert result.stats["length"]["splits"]["train"]["max"] == 71
    assert result.stats["length"]["splits"]["train"]["max_kept"] <= 40


def test_leakage_failure_raises_and_writes_nothing(
    store: Store, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _, _, teacher = classification_run(store, tmp_path)
    data = home / spec.task / "data"
    before = {path.name: path.read_bytes() for path in data.iterdir()}
    store.add_imports(
        spec.task,
        "pairs",
        [
            {
                "input": "where is my new card? it has not arrived yet.",
                "output": "card arrival",
                "meta": {"split": "test"},
            }
        ],
    )

    def keep_everything(splits: dict[str, list[Any]], threshold: float) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        return {k: list(v) for k, v in splits.items()}, {"threshold": threshold, "removed": {"train": 0, "valid": 0}}

    monkeypatch.setattr(pipeline, "remove_cross_split", keep_everything)
    with pytest.raises(CurateError, match=r"leakage check failed: 1 exact .*train/test 1"):
        run_curate(spec, store=store, teacher=teacher, tokenizer=WordTokenizer(), log=lambda _: None)
    assert {path.name: path.read_bytes() for path in data.iterdir()} == before

    fresh = make_spec()
    fresh.task = "fresh"
    store.add_imports("fresh", "pairs", [
        {"input": "Where is my card?", "output": "card arrival", "meta": {"split": "train"}},
        {"input": "WHERE IS MY CARD?", "output": "card arrival", "meta": {"split": "test"}},
    ])  # fmt: skip
    with pytest.raises(CurateError, match="leakage"):
        run_curate(fresh, store=store, teacher=None, tokenizer=WordTokenizer(), log=lambda _: None)
    assert not (home / "fresh" / "data").exists()


def test_spend_gate_stops_after_the_sample_and_writes_nothing(store: Store, home: Path) -> None:
    spec = make_spec()
    store.add_imports(spec.task, "inputs", [{"input": f"card question number {i}"} for i in range(100)])
    teacher = FakeTeacher(lambda _: "card arrival", mode="live", cost=0.02)
    lines: list[str] = []

    with pytest.raises(SpendNotConfirmed):
        run_curate(spec, store=store, teacher=teacher, tokenizer=WordTokenizer(), log=lines.append)
    assert len(teacher.bodies) == 50
    assert any("$2.0000 for 100 uncached" in line for line in lines)
    assert not (home / spec.task / "data").exists()

    teacher = FakeTeacher(lambda _: "card arrival", mode="live", cost=0.02)
    result = run_curate(spec, store=store, teacher=teacher, yes=True, max_usd=2.5, tokenizer=WordTokenizer(),
                        log=lambda _: None)  # fmt: skip
    assert len(teacher.bodies) == 100
    assert result.stats["labelling"]["cost_usd"] == pytest.approx(2.0)
    assert result.stats["labelling"]["max_usd"] == 2.5
    assert sum(result.stats["splits"].values()) == 100


def test_live_labelling_goes_through_the_budgeted_cached_client(store: Store, home: Path, tmp_path: Path) -> None:
    spec = make_spec()
    texts = [f"When does card number {i} arrive at my flat?" for i in range(12)]
    store.add_imports(spec.task, "inputs", [{"input": t, "meta": {"split": "train"}} for t in texts])
    price = ModelPrice(prompt=1e-6, completion=2e-6)
    pricing = PricingSnapshot(
        date="2026-09-26", source="test", models={TEACHER_MODEL: {"default": price, "providers": {"alpha/fp8": price}}}
    )
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=5.0)
    cache = ResponseCache(tmp_path / "cache.sqlite")
    base = "https://teacher.example.com/api/v1"

    def teacher() -> LiveTeacher:
        return LiveTeacher(
            base, "sk-test-teacher", ledger=ledger, pricing=pricing, task=spec.task, phase="curate",
            run_id="curate-test", cache=cache, concurrency=4,
        )  # fmt: skip

    with respx.mock(base_url=base) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion("card arrival")))
        live = teacher()
        first = run_curate(spec, store=store, teacher=live, tokenizer=WordTokenizer(), log=lambda _: None)
        asyncio.run(live.aclose())  # the caller may still close it from a new event loop
        assert route.call_count == 12
        sent = [json.loads(call.request.content)["messages"][-1]["content"] for call in route.calls]
        assert sorted(sent) == sorted(texts)
        assert first.stats["labelling"]["live"] == 12
        assert first.stats["labelling"]["cost_usd"] == pytest.approx(12e-5)
        assert ledger.spent(task=spec.task, phase="curate") == pytest.approx(12e-5)

        second = run_curate(spec, store=store, teacher=teacher(), tokenizer=WordTokenizer(), log=lambda _: None)
        assert route.call_count == 12
        assert second.stats["labelling"]["cached"] == 12
        assert second.stats["labelling"]["projection"]["n_total"] == 0
        assert second.stats["labelling"]["cost_usd"] == 0.0


def test_missing_teacher_and_empty_store_are_clear_errors(store: Store, home: Path) -> None:
    spec = make_spec()
    with pytest.raises(CurateError, match="no usable captured or imported data"):
        run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lambda _: None)
    store.add_imports(spec.task, "inputs", [{"input": "Where is my card?", "gold": "card_arrival"}])
    with pytest.raises(CurateError, match="no teacher is available"):
        run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lambda _: None)


def test_captured_outputs_need_no_teacher(store: Store, home: Path) -> None:
    spec = make_spec(curate={"split": {"val": 0.25, "test": 0.25}})
    for i in range(8):
        add_capture(store, spec, f"When will card number {i} arrive at my home?", "card arrival")
    result = run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lambda _: None)
    assert result.stats["splits"] == {"train": 4, "valid": 2, "test": 2}
    assert result.stats["labelling"]["requested"] == 0
    assert (home / spec.task / "data" / "labelling_keys.txt").read_text() == ""


def _demo_card_info(name: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The card_info dict the demo runner builds (Banking77 on a small offline stand-in for the CSVs)."""
    from taskdistill.demos import banking77 as b77
    from taskdistill.demos.runner import load_demo_data

    if name == "banking77":
        labels = ["card_arrival", "exchange_rate"]
        data = b77.BankingData(
            train=[(f"train text {i} about {label}", label) for i in range(20) for label in labels],
            test=[(f"test text {i} about {label}", label) for i in range(5) for label in labels],
            labels=labels,
        )
        monkeypatch.setattr(b77, "fetch_banking77", lambda *args, **kwargs: data)
        monkeypatch.setattr(b77, "profile_splits", lambda splits, profile: {s: list(splits[s]) for s in splits})
    return load_demo_data(name, "quick").card_info


@pytest.mark.parametrize("demo", ["banking77", "invoices"])
def test_the_demo_runners_card_info_is_accepted(
    demo: str, store: Store, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _demo_card_info(demo, monkeypatch)
    spec = make_spec()
    rows = [{"input": f"Where is card number {i} that I ordered?", "output": "card arrival"} for i in range(8)]
    store.add_imports(spec.task, "pairs", rows)
    lines: list[str] = []

    result = run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lines.append, card_info=info)

    card = result.paths["card"].read_text()
    assert card.startswith(f"# Dataset card: {info['name']}\n")
    assert f"- Source: {info['source']}" in card
    assert f"- Licence: {info['licence']}" in card
    assert info["attribution"] in card
    assert f"Predefined split rule (from the data source): {info['split_rule'].rstrip('.')}." in card
    assert result.stats["source"]["name"] == info["name"]
    assert not any("ignored" in line for line in lines)

    lines.clear()
    run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lines.append,
               card_info={**info, "homepage": "https://example.org"})  # fmt: skip
    assert any("card_info key(s) homepage ignored" in line for line in lines)


def test_empty_train_is_an_error_naming_the_stage_and_nothing_is_sent_or_written(store: Store, home: Path) -> None:
    spec = make_spec()
    store.add_imports(spec.task, "inputs", [
        {"input": f"Where is card number {i} that I ordered?", "meta": {"split": "test"}} for i in range(5)
    ])  # fmt: skip
    teacher = FakeTeacher(lambda _: "card arrival")
    with pytest.raises(CurateError, match=r"train split of task 'intents' is empty after split assignment"):
        run_curate(spec, store=store, teacher=teacher, tokenizer=WordTokenizer(), log=lambda _: None)
    assert teacher.bodies == []
    assert not (home / spec.task / "data").exists()

    other = make_spec()
    other.task = "shadowed"
    text = "Where is my new card? It has not arrived yet."
    store.add_imports("shadowed", "pairs", [
        {"input": text, "output": "card arrival", "meta": {"split": "train"}},
        {"input": text.lower().rstrip("."), "output": "card arrival", "meta": {"split": "test"}},
        {"input": "What rate do you apply for dollars?", "output": "exchange rate", "meta": {"split": "valid"}},
    ])  # fmt: skip
    with pytest.raises(CurateError, match="empty after cross-split removal: every row duplicates"):
        run_curate(other, store=store, teacher=None, tokenizer=WordTokenizer(), log=lambda _: None)

    long = make_spec(train={"max_seq_len": 10})
    long.task = "long"
    store.add_imports("long", "pairs", [
        {"input": f"Where is card number {i} that I ordered last month?", "output": "card arrival"} for i in range(4)
    ])  # fmt: skip
    with pytest.raises(CurateError, match=r"empty after length filter: every row is longer than train\.max_seq_len"):
        run_curate(long, store=store, teacher=None, tokenizer=WordTokenizer(), log=lambda _: None)
    assert not (home / "long" / "data").exists()


def test_empty_valid_or_test_split_is_a_warning(store: Store, home: Path) -> None:
    spec = make_spec()
    rows = [{"input": f"Where is card number {i} that I ordered?", "output": "card arrival"} for i in range(4)]
    rows.append({"input": "What rate do you use for yen today?", "output": "exchange rate", "meta": {"split": "test"}})
    store.add_imports(spec.task, "pairs", rows)
    lines: list[str] = []

    result = run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lines.append)

    assert result.stats["splits"] == {"train": 4, "valid": 0, "test": 1}
    warnings_ = [line for line in lines if line.startswith("warning:")]
    assert warnings_ == [
        "warning: the valid split is empty after split assignment: no row was assigned to it (every row has a "
        "predefined split elsewhere, or there are too few rows for the valid/test fractions); training has no "
        "validation loss and eval cannot choose a threshold"
    ]
    assert (home / spec.task / "data" / "valid.jsonl").read_text() == ""


def test_test_rows_with_an_invalid_teacher_answer_stay_in_test(store: Store, home: Path) -> None:
    spec = make_spec()
    rows = [{"input": f"When will card number {i} arrive at my home?", "meta": {"split": "train"}} for i in range(3)]
    rows.append({"input": "Something odd for training purposes", "meta": {"split": "train"}})
    rows.append({"input": "Something odd about my account today", "gold": "exchange_rate", "meta": {"split": "test"}})
    rows.append({"input": "Tell me when the card arrives please", "gold": "card_arrival", "meta": {"split": "test"}})
    store.add_imports(spec.task, "inputs", rows)
    teacher = FakeTeacher(lambda text: "I cannot say" if "odd" in text else "card arrival")

    result = run_curate(spec, store=store, teacher=teacher, tokenizer=WordTokenizer(), log=lambda _: None)

    stats = result.stats
    assert stats["splits"] == {"train": 3, "valid": 0, "test": 2}
    assert stats["split_sizes"]["cross_split"] == {"train": 4, "valid": 0, "test": 2}
    assert stats["split_sizes"]["label"] == stats["split_sizes"]["length"] == {"train": 3, "valid": 0, "test": 2}
    assert stats["labelling"]["invalid_by_split"] == {"train": 1, "valid": 0, "test": 1}
    assert stats["sources"]["test"]["teacher_invalid"] == 1
    test = read_split(spec.task, "test")
    odd = next(row for row in test if row.input.startswith("Something odd"))
    assert odd.teacher is None
    assert odd.gold == "exchange_rate"
    assert odd.target == ""
    assert "Something odd for training purposes" not in {row.input for row in read_split(spec.task, "train")}


def test_captures_from_another_model_or_prompt_are_counted_and_listed(store: Store, home: Path) -> None:
    spec = make_spec(curate={"split": {"val": 0.25, "test": 0.25}})
    for i in range(6):
        add_capture(store, spec, f"When will card number {i} arrive at my home?", "card arrival")
    other = make_spec(teacher={"model": "vendor/other-model", "max_tokens": 24})
    other.teacher_prompt = "A different prompt.\n"
    add_capture(store, other, "My card was stolen on the train today", "lost or stolen card", upstream_model="x/y")
    lines: list[str] = []

    result = run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lines.append)

    assert sum(result.stats["splits"].values()) == 7
    load = result.stats["stages"]["load"]["captures"]
    assert load["other_model"] == 1
    assert load["other_prompt"] == 1
    assert "(from another model 1, another system prompt 1 than the spec's teacher" in lines[0]
    card = result.paths["card"].read_text()
    assert "request models: `vendor/teacher-a` 6, `vendor/other-model` 1" in card
    assert "models named in the responses: none 6, `x/y` 1" in card
    assert "1 captures asked another model than `vendor/teacher-a`" in card
    assert "1 captures used another system prompt than the teacher prompt" in card


@pytest.mark.slow
def test_thirteen_thousand_examples_curate_well_under_a_minute(store: Store, home: Path) -> None:
    spec = make_spec()
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(400)] + ["card", "rate", "top", "up", "transfer", "fee", "refund", "pin"]
    labels = spec.labels
    rows = []
    for i in range(13000):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randint(4, 14)))
        split = "test" if i < 3000 else "train"
        rows.append({"input": text, "output": rng.choice(labels).replace("_", " "), "gold": rng.choice(labels),
                     "meta": {"split": split}})  # fmt: skip
    rows += [dict(rows[i], input=rows[i]["input"].upper()) for i in range(0, 13000, 200)]
    store.add_imports(spec.task, "pairs", rows)

    start = time.perf_counter()
    result = run_curate(spec, store=store, teacher=None, tokenizer=WordTokenizer(), log=lambda _: None)
    elapsed = time.perf_counter() - start

    assert sum(result.stats["splits"].values()) >= 12900
    assert result.stats["leakage"]["ok"]
    assert elapsed < 45, f"curate took {elapsed:.1f}s"
