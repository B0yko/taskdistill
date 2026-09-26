from __future__ import annotations

import asyncio
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from taskdistill.config import TaskSpec
from taskdistill.store import Store
from taskdistill.teacher.base import TeacherResult
from taskdistill.teacher.cache import ResponseCache, request_context
from taskdistill.teacher.record import LABELLING_KEYS, build_recording, default_keys, spec_context
from taskdistill.teacher.replay import RECORD_FIELDS, RecordingError, ReplayTeacher, load_recording
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request

MODEL = "vendor/model-a"
TASK = "demo-intents"
PROMPT = "Label the customer's message with one intent.\n"
QUERIES = [
    "My card still has not arrived after two weeks.",
    "I think I lost my card on the train.",
    "Where is the card you sent me?",
]
T0 = 1790000000.0


def make_spec(**teacher: Any) -> TaskSpec:
    teacher_cfg: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 24,
        "extra_body": {
            "provider": {"order": ["alpha/fp8"], "allow_fallbacks": False},
            "reasoning": {"enabled": False},
        },
    }
    teacher_cfg.update(teacher)
    spec = TaskSpec.model_validate(
        {
            "task": TASK,
            "type": "classification",
            "labels_file": "labels.txt",
            "teacher": teacher_cfg,
            "student": {"system_prompt": "Classify the message."},
            "cascade": {"target": 0.97},
        }
    )
    spec.teacher_prompt = PROMPT
    spec.labels = ["card_arrival", "lost_card"]
    spec.source = f"tasks/{TASK}/task.yaml"
    return spec


def result_for(body: dict[str, Any], output: str, index: int) -> TeacherResult:
    return TeacherResult(
        key=request_key(body),
        output=output,
        # The raw response echoes the request here only to prove that nothing but the record fields is written.
        response={"model": MODEL, "echo": body["messages"], "choices": []},
        usage={"prompt_tokens": 120 + index, "completion_tokens": 3, "cost": 1.2e-5},
        latency_ms=400.0 + index,
        provider="Alpha",
        finish_reason="stop",
        created=T0 + index,
        source="live",
        cost_usd=1.2e-5,
    )


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "home"
    monkeypatch.setenv("TASKDISTILL_HOME", str(workspace))
    return workspace


@pytest.fixture
def spec() -> TaskSpec:
    return make_spec()


@pytest.fixture
def bodies(spec: TaskSpec) -> list[dict[str, Any]]:
    return [build_teacher_request(spec, q) for q in QUERIES]


@pytest.fixture
def cache(tmp_path: Path, bodies: list[dict[str, Any]]) -> ResponseCache:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    for i, (body, output) in enumerate(zip(bodies, ["card_arrival", "lost_card", "card_arrival"], strict=True)):
        cache.put(result_for(body, output, i), request_context(body))
    return cache


def test_a_recording_built_from_the_cache_replays_every_request(
    tmp_path: Path, spec: TaskSpec, bodies: list[dict[str, Any]], cache: ResponseCache
) -> None:
    out = tmp_path / "rec" / "teacher_recording.jsonl.gz"
    manifest = build_recording(spec, requests=bodies, out=out, cache=cache, pricing_date="2026-09-26")
    assert manifest["records"] == 3
    assert manifest["task"] == TASK
    assert manifest["teacher_model"] == MODEL
    assert manifest["provider"] == "alpha/fp8"
    assert manifest["pricing_snapshot_date"] == "2026-09-26"
    assert manifest["generation"]["extra_body"] == spec.teacher.extra_body
    assert manifest["created"] == datetime.fromtimestamp(T0 + 2, UTC).isoformat(timespec="seconds")

    teacher = ReplayTeacher(out, spec)
    outputs = [asyncio.run(teacher.complete(body)).output for body in bodies]
    assert outputs == ["card_arrival", "lost_card", "card_arrival"]


def test_bare_keys_use_the_spec_context_and_give_the_same_bytes(
    tmp_path: Path, spec: TaskSpec, bodies: list[dict[str, Any]], cache: ResponseCache
) -> None:
    by_requests = tmp_path / "a.jsonl.gz"
    by_keys = tmp_path / "b.jsonl.gz"
    build_recording(spec, requests=bodies, out=by_requests, cache=cache, pricing_date="2026-09-26")
    keys = [request_key(b) for b in reversed(bodies)] + [request_key(bodies[0])]
    build_recording(spec, keys=keys, out=by_keys, cache=cache, pricing_date="2026-09-26")
    assert by_requests.read_bytes() == by_keys.read_bytes()
    assert spec_context(spec) == request_context(bodies[0])


def test_missing_keys_are_counted_and_the_first_few_named(tmp_path: Path, spec: TaskSpec, cache: ResponseCache) -> None:
    missing = [build_teacher_request(spec, f"never labelled {i}") for i in range(7)]
    wanted = [build_teacher_request(spec, QUERIES[0]), *missing]
    out = tmp_path / "rec.jsonl.gz"
    with pytest.raises(RecordingError) as err:
        build_recording(spec, requests=wanted, out=out, cache=cache, pricing_date="2026-09-26")
    message = str(err.value)
    assert "7 of 8 request keys are not in the response cache" in message
    shown = sorted(request_key(b) for b in missing)
    assert all(key in message for key in shown[:5])
    assert shown[5] not in message and ", ..." in message
    assert not out.exists()


def test_rows_cached_under_another_provider_are_skipped(tmp_path: Path, spec: TaskSpec) -> None:
    other = make_spec(extra_body={"provider": {"order": ["beta"], "allow_fallbacks": False}})
    body_other = build_teacher_request(other, QUERIES[0])
    assert request_key(body_other) == request_key(build_teacher_request(spec, QUERIES[0]))
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(result_for(body_other, "card_arrival", 0), request_context(body_other))

    with pytest.raises(RecordingError, match="1 of them are cached under another provider or reasoning setting"):
        build_recording(spec, keys=[request_key(body_other)], out=tmp_path / "rec.jsonl.gz", cache=cache)
    manifest = build_recording(
        other, keys=[request_key(body_other)], out=tmp_path / "other.jsonl.gz", cache=cache, pricing_date="x"
    )
    assert manifest["records"] == 1


def test_a_request_built_for_another_provider_is_refused(tmp_path: Path, spec: TaskSpec) -> None:
    other = make_spec(extra_body={"provider": {"order": ["beta"], "allow_fallbacks": False}})
    body_other = build_teacher_request(other, QUERIES[0])
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(result_for(body_other, "lost_card", 0), request_context(body_other))
    out = tmp_path / "rec.jsonl.gz"
    with pytest.raises(RecordingError, match=r"built with other provider, reasoning than teacher\.extra_body"):
        build_recording(spec, requests=[body_other], out=out, cache=cache, pricing_date="2026-09-26")
    assert not out.exists()


def test_a_request_for_another_model_is_refused(tmp_path: Path, spec: TaskSpec, cache: ResponseCache) -> None:
    body = {**build_teacher_request(spec, QUERIES[0]), "model": "vendor/other"}
    with pytest.raises(RecordingError, match=r"is for model 'vendor/other', but the recording is for teacher\.model"):
        build_recording(spec, requests=[body], out=tmp_path / "rec.jsonl.gz", cache=cache, pricing_date="x")


def test_client_only_fields_do_not_count_as_other_settings(
    tmp_path: Path, spec: TaskSpec, bodies: list[dict[str, Any]], cache: ResponseCache
) -> None:
    streamed = [{**body, "stream": True, "user": "u-1"} for body in bodies]
    manifest = build_recording(spec, requests=streamed, out=tmp_path / "rec.jsonl.gz", cache=cache, pricing_date="x")
    assert manifest["records"] == 3


def test_records_hold_no_inputs_and_the_gzip_header_is_reproducible(
    tmp_path: Path, spec: TaskSpec, bodies: list[dict[str, Any]], cache: ResponseCache
) -> None:
    out = tmp_path / "teacher_recording.jsonl.gz"
    build_recording(spec, requests=bodies, out=out, cache=cache, pricing_date="2026-09-26")
    raw = out.read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert raw[3] & 0x08 == 0  # FNAME flag: no file name stored
    assert raw[4:8] == b"\x00\x00\x00\x00"  # mtime 0

    text = gzip.decompress(raw).decode("utf-8")
    for query in QUERIES:
        assert query not in text
    assert PROMPT.strip() not in text
    lines = [json.loads(line) for line in text.splitlines()]
    assert lines[0]["type"] == "manifest"
    assert all(set(record) == set(RECORD_FIELDS) for record in lines[1:])
    assert [r["key"] for r in lines[1:]] == sorted(request_key(b) for b in bodies)

    again = tmp_path / "again.jsonl.gz"
    build_recording(spec, requests=bodies, out=again, cache=cache, pricing_date="2026-09-26")
    assert again.read_bytes() == raw


def test_the_pricing_date_defaults_to_the_snapshot_in_use(
    tmp_path: Path, spec: TaskSpec, bodies: list[dict[str, Any]], cache: ResponseCache
) -> None:
    manifest = build_recording(spec, requests=bodies, out=tmp_path / "rec.jsonl.gz", cache=cache)
    assert manifest["pricing_snapshot_date"] == "2026-09-26"
    assert load_recording(tmp_path / "rec.jsonl.gz").manifest == manifest


def test_an_unreadable_pricing_snapshot_is_an_error_not_a_null_date(
    home: Path, tmp_path: Path, spec: TaskSpec, bodies: list[dict[str, Any]], cache: ResponseCache
) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "pricing.json").write_text("{not json", encoding="utf-8")
    out = tmp_path / "rec.jsonl.gz"
    with pytest.raises(RecordingError, match="needs the pricing snapshot date"):
        build_recording(spec, requests=bodies, out=out, cache=cache)
    assert not out.exists()
    manifest = build_recording(spec, requests=bodies, out=out, cache=cache, pricing_date="2026-09-26")
    assert manifest["pricing_snapshot_date"] == "2026-09-26"


def test_nothing_to_record_is_an_error(tmp_path: Path, spec: TaskSpec, cache: ResponseCache) -> None:
    with pytest.raises(RecordingError, match="no request keys"):
        build_recording(spec, keys=[], out=tmp_path / "rec.jsonl.gz", cache=cache)


def test_default_keys_are_the_captured_requests_and_the_labelling_keys(
    home: Path, tmp_path: Path, spec: TaskSpec, bodies: list[dict[str, Any]]
) -> None:
    store = Store(tmp_path / "store.sqlite")

    def capture(body: dict[str, Any], **kw: Any) -> None:
        row: dict[str, Any] = {
            "task": TASK,
            "source": "proxy",
            "request_key": request_key(body),
            "request_body": json.dumps(body),
            "response_body": "{}",
            "status": 200,
        }
        row.update(kw)
        store.add_capture(**row)

    capture(bodies[0])
    capture(bodies[1])
    capture(bodies[0])  # the same request again
    capture({**bodies[2], "model": "vendor/other"})
    capture(build_teacher_request(spec, "upstream failed"), status=502)
    capture(build_teacher_request(spec, "streamed"), captured=False, response_body=None)
    store.add_capture(
        task="another-task",
        source="proxy",
        request_key="f" * 64,
        request_body=json.dumps(bodies[2]),
        response_body="{}",
        status=200,
    )
    labelling = home / TASK / "data" / LABELLING_KEYS
    labelling.parent.mkdir(parents=True)
    labelling.write_text(f"{request_key(bodies[1])}\n\n{'a' * 64}\n  {'b' * 64}  \n", encoding="utf-8")

    assert default_keys(spec, store) == [request_key(bodies[0]), request_key(bodies[1]), "a" * 64, "b" * 64]


def test_default_keys_without_a_labelling_file(tmp_path: Path, spec: TaskSpec) -> None:
    assert default_keys(spec, Store(tmp_path / "store.sqlite")) == []


class _FillingTeacher:
    """Stands in for the live teacher: answers and writes to the cache like LiveTeacher does."""

    mode = "live"

    def __init__(self, cache: ResponseCache) -> None:
        self.cache = cache
        self.bodies: list[dict[str, Any]] = []
        self.closed = False

    async def complete(self, body: dict[str, Any]) -> TeacherResult:
        self.bodies.append(body)
        result = result_for(body, "card_arrival", len(self.bodies))
        self.cache.put(result, request_context(body))
        return result

    async def aclose(self) -> None:
        self.closed = True


def test_fill_missing_asks_only_for_uncached_requests(
    tmp_path: Path, spec: TaskSpec, bodies: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from taskdistill.teacher import factory, record

    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(result_for(bodies[0], "card_arrival", 0), request_context(bodies[0]))
    fake = _FillingTeacher(cache)
    monkeypatch.setattr(factory, "make_teacher", lambda *a, **k: fake)
    monkeypatch.setattr(record, "load_pricing", lambda: _price_table())
    sent = record.fill_missing(spec, bodies, yes=True, cache=cache, log=lambda _m: None)
    assert sent == 2
    assert [request_key(b) for b in fake.bodies] == [request_key(b) for b in bodies[1:]]
    assert fake.closed
    assert record.fill_missing(spec, bodies, yes=True, cache=cache, log=lambda _m: None) == 0
    manifest = build_recording(spec, requests=bodies, out=tmp_path / "rec.jsonl.gz", cache=cache, pricing_date="d")
    assert manifest["records"] == 3


def _price_table() -> Any:
    from taskdistill.teacher.pricing import ModelPrice

    class _Pricing:
        def price_for(self, model: str, provider: str | None = None) -> ModelPrice:
            return ModelPrice(prompt=1e-7, completion=1e-7, request=0.0)

    return _Pricing()
