"""Offline replay of recorded teacher outputs.

A recording is gzip-compressed JSON Lines, written with an empty gzip filename and mtime 0 so its bytes are
reproducible. Line 1 is the manifest::

    {"type": "manifest", "schema_version": 1, "task", "teacher_model", "provider", "teacher_prompt_sha256",
     "generation": {"temperature", "max_tokens", "response_format", "extra_body"},
     "pricing_snapshot_date", "created", "records"}

Every further line is one record, sorted by request key, with exactly these fields and nothing else (no
inputs, no messages, no headers)::

    {"key", "output", "usage", "latency_ms", "provider", "finish_reason", "timestamp"}

A request whose key is not in the recording raises :class:`ReplayMiss`; replay never falls back to a live call.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Literal

from taskdistill.config import TaskSpec
from taskdistill.teacher.base import ManifestMismatch, ReplayMiss, TeacherError, TeacherResult
from taskdistill.teacher.client import is_truncated
from taskdistill.teacher.request_key import request_key

SCHEMA_VERSION = 1
RECORD_FIELDS = ("key", "output", "usage", "latency_ms", "provider", "finish_reason", "timestamp")
GENERATION_FIELDS = ("temperature", "max_tokens", "response_format", "extra_body")
MANIFEST_FIELDS = (
    "type",
    "schema_version",
    "task",
    "teacher_model",
    "provider",
    "teacher_prompt_sha256",
    "generation",
    "pricing_snapshot_date",
    "created",
    "records",
)
#: Manifest fields that must equal the task spec's before a recording may be replayed.
COMPARED_FIELDS = ("schema_version", "task", "teacher_model", "provider", "teacher_prompt_sha256", "generation")
_MISSING = object()


class RecordingError(TeacherError):
    """A recording file is malformed."""


@dataclass
class Recording:
    manifest: dict[str, Any]
    records: dict[str, dict[str, Any]]
    name: str = "teacher_recording.jsonl.gz"

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, key: object) -> bool:
        return key in self.records

    def get(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def record_from_result(result: TeacherResult) -> dict[str, Any]:
    """The replay record of a teacher result: output, usage, latency, provider, finish reason and timestamp."""
    return {
        "key": result.key,
        "output": result.output,
        "usage": copy.deepcopy(result.usage),
        "latency_ms": result.latency_ms,
        "provider": result.provider,
        "finish_reason": result.finish_reason,
        "timestamp": result.created,
    }


def _clean_record(record: Mapping[str, Any] | TeacherResult) -> dict[str, Any]:
    if isinstance(record, TeacherResult):
        return record_from_result(record)
    out = {field: copy.deepcopy(record.get(field)) for field in RECORD_FIELDS}
    if not isinstance(out["key"], str) or not out["key"]:
        raise RecordingError("every recording record needs a request key")
    return out


def _latest_timestamp(records: Iterable[Mapping[str, Any]]) -> str | None:
    """ISO 8601 UTC time of the newest record (None when no record has a timestamp); never the wall clock."""
    stamps = [
        float(r["timestamp"])
        for r in records
        if isinstance(r.get("timestamp"), int | float)
        and not isinstance(r.get("timestamp"), bool)
        and math.isfinite(r["timestamp"])
    ]
    if not stamps:
        return None
    return datetime.fromtimestamp(max(stamps), UTC).isoformat(timespec="seconds")


def write_recording(
    path: Path | str, manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any] | TeacherResult]
) -> Path:
    """Write a recording: the manifest line, then the records sorted by key. Same inputs give the same bytes.

    Record fields other than the seven listed in the module docstring are dropped, so inputs and headers can
    never reach a recording. A manifest without ``created`` gets the time of the newest record, so the bytes
    never depend on when the file was written.
    """
    unknown = sorted(set(manifest) - set(MANIFEST_FIELDS))
    if unknown:
        raise RecordingError(f"unknown recording manifest fields: {', '.join(unknown)}")
    missing = [f for f in ("task", "teacher_model", "teacher_prompt_sha256", "generation") if f not in manifest]
    if missing:
        raise RecordingError(f"recording manifest lacks {', '.join(missing)}")
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        clean = _clean_record(record)
        by_key[clean["key"]] = clean
    head = {field: copy.deepcopy(manifest.get(field)) for field in MANIFEST_FIELDS}
    head["type"] = "manifest"
    head["schema_version"] = SCHEMA_VERSION
    head["records"] = len(by_key)
    if head["created"] is None:
        head["created"] = _latest_timestamp(by_key.values())
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        gz.write(_dumps(head) + b"\n")
        for key in sorted(by_key):
            gz.write(_dumps(by_key[key]) + b"\n")
    tmp.replace(out)
    return out


def load_recording(src: Path | str | Traversable) -> Recording:
    """Read a recording from a path or an ``importlib.resources`` traversable (package data)."""
    source: Path | Traversable = Path(src) if isinstance(src, str) else src
    name = source.name
    try:
        with source.open("rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as gz:
            # Split on "\n" only: str.splitlines() also breaks on U+2028, U+2029 and U+0085, which JSON strings
            # written with ensure_ascii=False may contain.
            lines = gz.read().decode("utf-8").split("\n")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise RecordingError(f"cannot read teacher recording {name}: {exc}") from exc
    if not any(line.strip() for line in lines):
        raise RecordingError(f"teacher recording {name} is empty")
    try:
        manifest = json.loads(lines[0])
    except ValueError as exc:
        raise RecordingError(f"teacher recording {name}: the manifest line is not JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("type") != "manifest":
        raise RecordingError(f"teacher recording {name} does not start with a manifest line")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestMismatch(
            f"recording manifest mismatch: schema_version is {manifest.get('schema_version')!r} in the recording "
            f"but this version of taskdistill reads {SCHEMA_VERSION}"
        )
    records: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise RecordingError(f"teacher recording {name}, line {number}: not JSON") from exc
        if not isinstance(record, dict) or not isinstance(record.get("key"), str):
            raise RecordingError(f"teacher recording {name}, line {number}: a record needs a key")
        extra = sorted(set(record) - set(RECORD_FIELDS))
        if extra:
            raise RecordingError(
                f"teacher recording {name}, line {number}: unexpected fields {', '.join(extra)} "
                "(records hold only the output and its metadata, never inputs or headers)"
            )
        if record["key"] in records:
            raise RecordingError(f"teacher recording {name}, line {number}: duplicate key {record['key']}")
        records[record["key"]] = {field: record.get(field) for field in RECORD_FIELDS}
    declared = manifest.get("records")
    if declared is not None and declared != len(records):
        raise RecordingError(f"teacher recording {name}: manifest declares {declared} records, found {len(records)}")
    return Recording(manifest=manifest, records=records, name=name)


def expected_manifest(spec: TaskSpec, pricing_date: str | None = None) -> dict[str, Any]:
    """The manifest fields a recording must carry to be replayed for ``spec``."""
    manifest: dict[str, Any] = {
        "type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "task": spec.task,
        "teacher_model": spec.teacher.model,
        "provider": spec.teacher.provider,
        "teacher_prompt_sha256": spec.teacher_prompt_sha256,
        "generation": {
            "temperature": spec.teacher.temperature,
            "max_tokens": spec.teacher.max_tokens,
            "response_format": copy.deepcopy(spec.teacher.response_format),
            "extra_body": copy.deepcopy(spec.teacher.extra_body),
        },
    }
    if pricing_date is not None:
        manifest["pricing_snapshot_date"] = pricing_date
    return manifest


def _show(value: Any) -> str:
    if value is _MISSING:
        return "absent"
    if isinstance(value, str):
        return repr(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _first_difference(recorded: Any, expected: Any, path: str) -> tuple[str, Any, Any] | None:
    if isinstance(recorded, dict) and isinstance(expected, dict):
        order = [k for k in GENERATION_FIELDS if k in recorded or k in expected]
        order += sorted((set(recorded) | set(expected)) - set(order))
        for key in order:
            found = _first_difference(recorded.get(key, _MISSING), expected.get(key, _MISSING), f"{path}.{key}")
            if found is not None:
                return found
        return None
    if recorded is _MISSING or expected is _MISSING:
        return None if recorded is expected else (path, recorded, expected)
    return None if recorded == expected else (path, recorded, expected)


def check_manifest(recording: Recording, spec: TaskSpec) -> None:
    """Refuse a recording made for a different teacher, prompt or generation setting than ``spec``."""
    expected = json.loads(_dumps(expected_manifest(spec)))
    where = spec.source or f"tasks/{spec.task}/task.yaml"
    for field in COMPARED_FIELDS:
        found = _first_difference(recording.manifest.get(field, _MISSING), expected[field], field)
        if found is not None:
            path, got, want = found
            raise ManifestMismatch(
                f"recording manifest mismatch: {path} is {_show(got)} in the recording but {_show(want)} in "
                f"{where} ({recording.name}); restore the spec the recording was made with, or re-record with --live"
            )


class ReplayTeacher:
    """Answers teacher requests from a recording (mode ``replay``). A miss is always a hard error."""

    mode: Literal["live", "replay"] = "replay"

    def __init__(self, recording: Recording | Path | str | Traversable, spec: TaskSpec | None = None) -> None:
        self.recording = recording if isinstance(recording, Recording) else load_recording(recording)
        if spec is not None:
            check_manifest(self.recording, spec)
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str | None:
        model = self.recording.manifest.get("teacher_model")
        return model if isinstance(model, str) else None

    def _lookup(self, body: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        try:
            key = request_key(body)
        except ValueError as exc:
            self.misses += 1
            raise ReplayMiss(f"replay miss: the request cannot be keyed ({exc})") from exc
        record = self.recording.records.get(key)
        if record is None:
            self.misses += 1
            raise ReplayMiss(
                f"replay miss: request key {key} (model {body.get('model')!r}) is not in the teacher recording "
                f"{self.recording.name}. Replay never falls back to a live call: build the request with "
                "build_teacher_request() so its key matches the recording, or set a teacher API key and use --live."
            )
        self.hits += 1
        return key, record

    def _base(self, body: Mapping[str, Any], key: str, record: Mapping[str, Any], obj: str) -> dict[str, Any]:
        timestamp = record.get("timestamp")
        return {
            "id": "replay-" + key[:16],
            "object": obj,
            "created": int(timestamp) if isinstance(timestamp, int | float) else 0,
            "model": body.get("model") or self.model,
            "provider": record.get("provider"),
        }

    def response_for(self, body: Mapping[str, Any], key: str, record: Mapping[str, Any]) -> dict[str, Any]:
        """The recorded answer as a ``chat.completion`` response."""
        response = self._base(body, key, record, "chat.completion")
        response["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": record.get("output")},
                "finish_reason": record.get("finish_reason"),
            }
        ]
        response["usage"] = copy.deepcopy(record.get("usage") or {})
        return response

    async def complete(self, body: dict[str, Any]) -> TeacherResult:
        key, record = self._lookup(body)
        usage = copy.deepcopy(record.get("usage") or {})
        cost = usage.get("cost")
        timestamp = record.get("timestamp")
        return TeacherResult(
            key=key,
            output=record.get("output"),
            response=self.response_for(body, key, record),
            usage=usage,
            latency_ms=record.get("latency_ms"),
            provider=record.get("provider"),
            finish_reason=record.get("finish_reason"),
            created=float(timestamp) if isinstance(timestamp, int | float) else 0.0,
            source="replay",
            cost_usd=float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else 0.0,
            attempts=1,
            truncated=is_truncated(body, usage, record.get("finish_reason")),
        )

    async def stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """SSE synthesised from the recorded output: one content chunk, one final chunk with usage, [DONE]."""
        key, record = self._lookup(body)
        base = self._base(body, key, record, "chat.completion.chunk")
        first = {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": record.get("output") or ""},
                    "finish_reason": None,
                }
            ],
        }
        last = {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": record.get("finish_reason")}],
            "usage": copy.deepcopy(record.get("usage") or {}),
        }
        for event in (first, last):
            yield b"data: " + _dumps(event) + b"\n\n"
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        return None
