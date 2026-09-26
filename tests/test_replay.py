from __future__ import annotations

import asyncio
import gzip
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from taskdistill.config import TaskSpec
from taskdistill.ledger import Ledger
from taskdistill.teacher.base import ManifestMismatch, ReplayMiss, TeacherResult
from taskdistill.teacher.cache import ResponseCache
from taskdistill.teacher.client import LiveTeacher
from taskdistill.teacher.pricing import ModelPrice, PricingSnapshot
from taskdistill.teacher.replay import (
    RECORD_FIELDS,
    Recording,
    RecordingError,
    ReplayTeacher,
    check_manifest,
    expected_manifest,
    load_recording,
    write_recording,
)
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

MODEL = "vendor/model-a"
QUERY = "My card still has not arrived after two weeks."


def make_spec(prompt: str = "Label the customer's message with one intent.\n", **teacher: Any) -> TaskSpec:
    teacher_cfg: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 24,
        "extra_body": {"provider": {"order": ["alpha/fp8"], "allow_fallbacks": False}},
    }
    teacher_cfg.update(teacher)
    spec = TaskSpec.model_validate(
        {
            "task": "demo-intents",
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": teacher_cfg,
            "student": {"system_prompt": "Classify the message."},
            "cascade": {"target": 0.97},
        }
    )
    spec.teacher_prompt = prompt
    spec.labels = ["card_arrival", "lost_card"]
    spec.source = "tasks/demo-intents/task.yaml"
    return spec


def record(key: str, output: str = "card_arrival", **extra: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "key": key,
        "output": output,
        "usage": {"prompt_tokens": 120, "completion_tokens": 3, "cost": 1.2e-5},
        "latency_ms": 410.5,
        "provider": "Alpha",
        "finish_reason": "stop",
        "timestamp": 1790000000.25,
    }
    rec.update(extra)
    return rec


def manifest_for(spec: TaskSpec) -> dict[str, Any]:
    return {**expected_manifest(spec, "2026-09-26"), "created": "2026-09-26T12:00:00+00:00"}


@pytest.fixture
def spec() -> TaskSpec:
    return make_spec()


@pytest.fixture
def body(spec: TaskSpec) -> dict[str, Any]:
    return build_teacher_request(spec, QUERY)


@pytest.fixture
def recording_path(tmp_path: Path, spec: TaskSpec, body: dict[str, Any]) -> Path:
    other = build_teacher_request(spec, "I lost my card yesterday.")
    records = [record(request_key(body)), record(request_key(other), "lost_card")]
    return write_recording(tmp_path / "teacher_recording.jsonl.gz", manifest_for(spec), records)


# format ------------------------------------------------------------------------------------------
def test_round_trip(recording_path: Path, spec: TaskSpec, body: dict[str, Any]) -> None:
    rec = load_recording(recording_path)
    assert len(rec) == 2
    assert rec.manifest["type"] == "manifest"
    assert rec.manifest["schema_version"] == 1
    assert rec.manifest["records"] == 2
    assert rec.manifest["task"] == "demo-intents"
    assert rec.manifest["teacher_model"] == MODEL
    assert rec.manifest["provider"] == "alpha/fp8"
    assert rec.manifest["teacher_prompt_sha256"] == spec.teacher_prompt_sha256
    assert rec.manifest["pricing_snapshot_date"] == "2026-09-26"
    assert rec.manifest["generation"] == {
        "temperature": 0.0,
        "max_tokens": 24,
        "response_format": None,
        "extra_body": {"provider": {"order": ["alpha/fp8"], "allow_fallbacks": False}},
    }
    assert rec.records[request_key(body)] == record(request_key(body))


def test_gzip_header_has_no_filename_and_zero_mtime(recording_path: Path) -> None:
    raw = recording_path.read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert raw[2] == 8  # deflate
    assert raw[3] & 0x08 == 0  # FNAME flag not set
    assert raw[4:8] == b"\x00\x00\x00\x00"  # MTIME = 0
    assert b"teacher_recording" not in raw[:64]


def test_bytes_are_deterministic_and_records_sorted(tmp_path: Path, spec: TaskSpec) -> None:
    records = [record("c" * 64), record("a" * 64), record("b" * 64)]
    first = write_recording(tmp_path / "one.jsonl.gz", manifest_for(spec), records)
    second = write_recording(tmp_path / "two.jsonl.gz", manifest_for(spec), list(reversed(records)))
    assert first.read_bytes() == second.read_bytes()
    lines = gzip.decompress(first.read_bytes()).decode("utf-8").splitlines()
    assert [json.loads(line)["key"][0] for line in lines[1:]] == ["a", "b", "c"]


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85", "\r", "\x0b", "\x1c"])
def test_outputs_with_unicode_line_separators_round_trip(tmp_path: Path, spec: TaskSpec, separator: str) -> None:
    output = '{"vendor": "Example' + separator + 'Ltd"}'
    records = [record("a" * 64, output), record("b" * 64)]
    path = write_recording(tmp_path / "sep.jsonl.gz", manifest_for(spec), records)
    rec = load_recording(path)
    assert len(rec) == 2
    assert rec.records["a" * 64]["output"] == output
    assert rec.records["b" * 64]["output"] == "card_arrival"


def test_created_defaults_to_the_newest_record_not_the_clock(tmp_path: Path, spec: TaskSpec) -> None:
    manifest = expected_manifest(spec, "2026-09-26")
    assert "created" not in manifest
    records = [record("a" * 64), record("b" * 64, timestamp=1790000001.0), record("c" * 64, timestamp=None)]
    first = write_recording(tmp_path / "one.jsonl.gz", manifest, records)
    second = write_recording(tmp_path / "two.jsonl.gz", manifest, records)
    assert first.read_bytes() == second.read_bytes()
    # newest timestamp 1790000001.0 s = 20717 days + 51201 s after the epoch
    assert load_recording(first).manifest["created"] == "2026-09-21T14:13:21+00:00"
    untimed = write_recording(tmp_path / "three.jsonl.gz", manifest, [record("a" * 64, timestamp=None)])
    assert load_recording(untimed).manifest["created"] is None


def test_explicit_created_is_kept(recording_path: Path) -> None:
    assert load_recording(recording_path).manifest["created"] == "2026-09-26T12:00:00+00:00"


def test_records_never_carry_inputs_or_headers(tmp_path: Path, spec: TaskSpec, body: dict[str, Any]) -> None:
    dirty = record(
        request_key(body),
        messages=body["messages"],
        input=QUERY,
        headers={"authorization": "Bearer sk-test-secret"},
        request=body,
    )
    path = write_recording(tmp_path / "rec.jsonl.gz", manifest_for(spec), [dirty])
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    assert QUERY not in text
    assert "sk-test-secret" not in text
    assert spec.teacher_prompt.strip() not in text
    for line in text.splitlines()[1:]:
        assert tuple(sorted(json.loads(line))) == tuple(sorted(RECORD_FIELDS))


def test_teacher_results_become_records(tmp_path: Path, spec: TaskSpec) -> None:
    result = TeacherResult(
        key="d" * 64,
        output="lost_card",
        response={"id": "gen-1", "choices": []},
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        latency_ms=12.5,
        provider="Alpha",
        finish_reason="stop",
        created=1790000001.0,
        source="live",
        cost_usd=1e-6,
    )
    rec = load_recording(write_recording(tmp_path / "rec.jsonl.gz", manifest_for(spec), [result]))
    assert rec.records["d" * 64] == {
        "key": "d" * 64,
        "output": "lost_card",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "latency_ms": 12.5,
        "provider": "Alpha",
        "finish_reason": "stop",
        "timestamp": 1790000001.0,
    }


def _write_raw(path: Path, lines: list[dict[str, Any]]) -> Path:
    payload = "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8")
    path.write_bytes(gzip.compress(payload, mtime=0))
    return path


def test_load_refuses_records_with_extra_fields(tmp_path: Path, spec: TaskSpec) -> None:
    head = {**manifest_for(spec), "type": "manifest", "schema_version": 1, "records": 1}
    path = _write_raw(tmp_path / "bad.jsonl.gz", [head, {**record("e" * 64), "input": QUERY}])
    with pytest.raises(RecordingError, match="unexpected fields input"):
        load_recording(path)


def test_load_refuses_a_truncated_recording(tmp_path: Path, spec: TaskSpec) -> None:
    head = {**manifest_for(spec), "type": "manifest", "schema_version": 1, "records": 3}
    path = _write_raw(tmp_path / "short.jsonl.gz", [head, record("e" * 64)])
    with pytest.raises(RecordingError, match="declares 3 records, found 1"):
        load_recording(path)


def test_load_refuses_an_unknown_schema_version(tmp_path: Path, spec: TaskSpec) -> None:
    head = {**manifest_for(spec), "type": "manifest", "schema_version": 2, "records": 0}
    path = _write_raw(tmp_path / "v2.jsonl.gz", [head])
    with pytest.raises(ManifestMismatch, match="schema_version is 2 in the recording"):
        load_recording(path)


def test_load_refuses_a_file_without_manifest(tmp_path: Path) -> None:
    path = _write_raw(tmp_path / "nomanifest.jsonl.gz", [record("e" * 64)])
    with pytest.raises(RecordingError, match="manifest"):
        load_recording(path)


def test_load_refuses_an_empty_recording(tmp_path: Path) -> None:
    with pytest.raises(RecordingError, match="is empty"):
        load_recording(_write_raw(tmp_path / "empty.jsonl.gz", []))


def test_write_refuses_unknown_manifest_fields(tmp_path: Path, spec: TaskSpec) -> None:
    with pytest.raises(RecordingError, match="unknown recording manifest fields: headers"):
        write_recording(tmp_path / "x.jsonl.gz", {**manifest_for(spec), "headers": {}}, [])


def test_load_from_a_traversable(tmp_path: Path, recording_path: Path) -> None:
    archive = tmp_path / "pkg.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(recording_path, "taskdistill/_data/demos/demo/teacher_recording.jsonl.gz")
    with zipfile.ZipFile(archive) as zf:
        traversable = zipfile.Path(zf, "taskdistill/_data/demos/demo/teacher_recording.jsonl.gz")
        rec = load_recording(traversable)
    assert len(rec) == 2
    assert rec.name == "teacher_recording.jsonl.gz"


# manifest checks ---------------------------------------------------------------------------------
def test_matching_manifest_is_accepted(recording_path: Path, spec: TaskSpec) -> None:
    check_manifest(load_recording(recording_path), spec)
    check_manifest(load_recording(recording_path), make_spec())


def test_manifest_mismatch_names_the_teacher_model(recording_path: Path) -> None:
    with pytest.raises(ManifestMismatch) as info:
        check_manifest(load_recording(recording_path), make_spec(model="vendor/model-b"))
    assert str(info.value).startswith(
        "recording manifest mismatch: teacher_model is 'vendor/model-a' in the recording "
        "but 'vendor/model-b' in tasks/demo-intents/task.yaml"
    )


def test_manifest_mismatch_names_a_generation_field(recording_path: Path) -> None:
    with pytest.raises(ManifestMismatch, match=r"generation\.max_tokens is 24 in the recording but 32 in"):
        check_manifest(load_recording(recording_path), make_spec(max_tokens=32))
    with pytest.raises(ManifestMismatch, match=r"generation\.temperature is 0\.0 in the recording but 0\.2 in"):
        check_manifest(load_recording(recording_path), make_spec(temperature=0.2))
    with pytest.raises(ManifestMismatch, match=r"generation\.response_format is null in the recording"):
        check_manifest(load_recording(recording_path), make_spec(response_format={"type": "json_object"}))


def test_manifest_mismatch_names_the_provider_and_extra_body(recording_path: Path) -> None:
    pinned_elsewhere = {"provider": {"order": ["beta"], "allow_fallbacks": False}}
    with pytest.raises(ManifestMismatch, match="provider is 'alpha/fp8' in the recording but 'beta' in"):
        check_manifest(load_recording(recording_path), make_spec(extra_body=pinned_elsewhere))
    reasoning_off = {
        "provider": {"order": ["alpha/fp8"], "allow_fallbacks": False},
        "reasoning": {"enabled": False},
    }
    with pytest.raises(ManifestMismatch, match=r"generation\.extra_body\.reasoning is absent in the recording"):
        check_manifest(load_recording(recording_path), make_spec(extra_body=reasoning_off))


def test_manifest_mismatch_names_the_prompt_hash(recording_path: Path) -> None:
    with pytest.raises(ManifestMismatch, match="teacher_prompt_sha256 is"):
        check_manifest(load_recording(recording_path), make_spec(prompt="A different prompt.\n"))


def test_replay_teacher_refuses_a_mismatched_spec(recording_path: Path) -> None:
    with pytest.raises(ManifestMismatch, match="teacher_model"):
        ReplayTeacher(recording_path, make_spec(model="vendor/model-b"))


# replay ------------------------------------------------------------------------------------------
def test_replay_complete_reconstructs_the_response(recording_path: Path, spec: TaskSpec, body: dict[str, Any]) -> None:
    teacher = ReplayTeacher(recording_path, spec)
    assert teacher.mode == "replay"
    assert teacher.model == MODEL
    result = asyncio.run(teacher.complete(body))
    key = request_key(body)
    assert result.source == "replay"
    assert result.key == key
    assert result.output == "card_arrival"
    assert result.cost_usd == pytest.approx(1.2e-5)
    assert result.latency_ms == 410.5
    assert not result.truncated
    assert result.response == {
        "id": "replay-" + key[:16],
        "object": "chat.completion",
        "created": 1790000000,
        "model": MODEL,
        "provider": "Alpha",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "card_arrival"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 3, "cost": 1.2e-5},
    }
    assert teacher.hits == 1


def test_replay_cost_is_zero_without_recorded_cost(tmp_path: Path, spec: TaskSpec, body: dict[str, Any]) -> None:
    rec = record(request_key(body), usage={"prompt_tokens": 120, "completion_tokens": 30}, finish_reason="length")
    teacher = ReplayTeacher(write_recording(tmp_path / "r.jsonl.gz", manifest_for(spec), [rec]))
    result = asyncio.run(teacher.complete(body))
    assert result.cost_usd == 0.0
    assert result.truncated


def test_replay_miss_is_a_hard_error(recording_path: Path, spec: TaskSpec) -> None:
    teacher = ReplayTeacher(recording_path, spec)
    unknown = build_teacher_request(spec, "How do I close my account?")
    with pytest.raises(ReplayMiss) as info:
        asyncio.run(teacher.complete(unknown))
    assert request_key(unknown) in str(info.value)
    assert "never falls back" in str(info.value)
    assert teacher.misses == 1


def test_a_body_that_cannot_be_keyed_is_a_replay_miss_not_a_crash(recording_path: Path, spec: TaskSpec) -> None:
    """request_key() raises ValueError on NaN/Infinity; replay reports it as a miss like any other unkeyable body."""
    teacher = ReplayTeacher(recording_path, spec)
    unkeyable = {**build_teacher_request(spec, "How do I close my account?"), "temperature": float("nan")}
    with pytest.raises(ReplayMiss, match="cannot be keyed"):
        asyncio.run(teacher.complete(unkeyable))
    assert teacher.misses == 1
    with pytest.raises(ReplayMiss, match="cannot be keyed"):
        asyncio.run(_collect(teacher.stream(unkeyable)))
    assert teacher.misses == 2


def test_replay_ignores_fields_outside_the_key(recording_path: Path, spec: TaskSpec, body: dict[str, Any]) -> None:
    teacher = ReplayTeacher(recording_path, spec)
    result = asyncio.run(teacher.complete({**body, "stream": False, "user": "u-1", "provider": {"order": ["x"]}}))
    assert result.output == "card_arrival"


async def _collect(stream: Any) -> list[bytes]:
    return [chunk async for chunk in stream]


def test_replay_stream_synthesises_sse(recording_path: Path, spec: TaskSpec, body: dict[str, Any]) -> None:
    teacher = ReplayTeacher(recording_path, spec)
    chunks = asyncio.run(_collect(teacher.stream({**body, "stream": True})))
    assert chunks[-1] == b"data: [DONE]\n\n"
    events = [json.loads(c.decode("utf-8").removeprefix("data: ")) for c in chunks[:-1]]
    assert all(c.startswith(b"data: ") and c.endswith(b"\n\n") for c in chunks)
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "card_arrival"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"]["cost"] == pytest.approx(1.2e-5)


def test_replay_stream_miss_raises(recording_path: Path, spec: TaskSpec) -> None:
    teacher = ReplayTeacher(recording_path, spec)
    with pytest.raises(ReplayMiss):
        asyncio.run(_collect(teacher.stream(build_teacher_request(spec, "unrecorded"))))


def test_replay_accepts_an_in_memory_recording(spec: TaskSpec, body: dict[str, Any]) -> None:
    rec = Recording(manifest=manifest_for(spec), records={request_key(body): record(request_key(body))})
    assert asyncio.run(ReplayTeacher(rec).complete(body)).output == "card_arrival"


# request key agreement ---------------------------------------------------------------------------
def test_cache_and_replay_compute_the_same_request_key(tmp_path: Path, spec: TaskSpec, body: dict[str, Any]) -> None:
    base = "https://teacher.example.com/api/v1"
    pricing = PricingSnapshot(
        date="2026-09-26", source="test", models={MODEL: {"default": ModelPrice(1e-7, 2e-7), "providers": {}}}
    )
    cache = ResponseCache(tmp_path / "cache.sqlite")
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=1.0)
    response = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1790000000,
        "model": MODEL,
        "provider": "Alpha",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "card_arrival"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 3, "cost": 1.2e-5},
    }

    async def label() -> TeacherResult:
        teacher = LiveTeacher(
            base, "sk-test-teacher", ledger=ledger, pricing=pricing, task=spec.task, phase="label", run_id="r1",
            cache=cache,
        )  # fmt: skip
        try:
            return await teacher.complete(body)
        finally:
            await teacher.aclose()

    with respx.mock() as mock:
        mock.post(f"{base}/chat/completions").mock(return_value=httpx.Response(200, json=response))
        live = asyncio.run(label())

    cached = cache.get(request_key(body))
    assert cached is not None
    path = write_recording(tmp_path / "rec.jsonl.gz", manifest_for(spec), [live])
    replayed = asyncio.run(ReplayTeacher(path, spec).complete(build_teacher_request(spec, QUERY)))
    assert live.key == cached.key == replayed.key == request_key(body)
    assert replayed.output == live.output == cached.output


def test_key_agreement_holds_when_the_client_adds_max_tokens(
    tmp_path: Path, spec: TaskSpec, body: dict[str, Any]
) -> None:
    base = "https://teacher.example.com/api/v1"
    pricing = PricingSnapshot(
        date="2026-09-26", source="test", models={MODEL: {"default": ModelPrice(1e-7, 2e-7), "providers": {}}}
    )
    cache = ResponseCache(tmp_path / "cache.sqlite")
    ledger = Ledger(tmp_path / "ledger.sqlite", global_cap=1.0)
    unbounded = {k: v for k, v in body.items() if k != "max_tokens"}
    key = request_key(unbounded)  # what proxy capture stores for this body
    sent: list[dict[str, Any]] = []
    response = {
        "id": "gen-2",
        "object": "chat.completion",
        "created": 1790000000,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "card_arrival"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 3, "cost": 1.2e-5},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response)

    async def label() -> TeacherResult:
        teacher = LiveTeacher(
            base, "sk-test-teacher", ledger=ledger, pricing=pricing, task=spec.task, phase="demo", run_id="r1",
            cache=cache, max_tokens_default=64,
        )  # fmt: skip
        try:
            return await teacher.complete(unbounded)
        finally:
            await teacher.aclose()

    with respx.mock() as mock:
        mock.post(f"{base}/chat/completions").mock(side_effect=handler)
        live = asyncio.run(label())

    assert sent[0]["max_tokens"] == 64
    assert live.key == key
    rows = list(cache.iter_results([key]))  # keys taken from the capture store
    assert [r.key for r in rows] == [key]
    path = write_recording(tmp_path / "rec.jsonl.gz", manifest_for(spec), rows)
    replayed = asyncio.run(ReplayTeacher(path, spec).complete(unbounded))
    assert replayed.key == key
    assert replayed.output == "card_arrival"
