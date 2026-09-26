"""Student predictions over an evaluation split, cached per run and split.

The cache is ``runs/<run-id>/preds_<split>.jsonl``: a header line (the SHA-256 of ``data/<split>.jsonl``, the adapter's
SHA-256, the base model when there is no adapter, whether alternative scores were computed, the backend and a hash of
the prediction settings), then one row per record. Rows are reused only when the header matches, so changed data, a
retrained adapter, another base model or a different prompt invalidates them. The first predictions of a fresh pass
are warm-up calls and are not timed or kept.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from taskdistill.backends.types import Generation
from taskdistill.confidence import LabelTrie
from taskdistill.config import TaskSpec
from taskdistill.evaluate.data import file_sha256
from taskdistill.evaluate.splits import TestSplit, ValidationSplit
from taskdistill.predict import Prediction, ScoringBackend, predict
from taskdistill.teacher.requests import build_student_messages

WARMUP: Final = 3
HEADER_TYPE: Final = "header"
ROW_FIELDS: Final = (
    "id",
    "answer",
    "value",
    "confidence",
    "alt_value",
    "alt_confidence",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "field_confidences",
)
_PROGRESS_STEPS: Final = 10
_PROGRESS_MIN: Final = 100

MessagesFn = Callable[[str], list[dict[str, str]]]
Log = Callable[[str], Any]


# -- header ------------------------------------------------------------------------------------------------------


def adapter_sha256(adapter_dir: Path | None) -> str | None:
    """SHA-256 over an adapter directory's files (relative path and content, in sorted order); None without one."""
    if adapter_dir is None:
        return None
    root = Path(adapter_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {root.name}")
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(file_sha256(path).encode("ascii") + b"\0")
    return digest.hexdigest()


def settings_sha256(spec: TaskSpec, messages_fn: MessagesFn | None = None) -> str:
    """Hash of everything besides the input and the weights that shapes a prediction: prompt, limits, labels, schema."""
    template = messages_fn("") if messages_fn is not None else build_student_messages(spec, "")
    payload = {
        "type": spec.type,
        "messages": template,
        "max_tokens": spec.student.max_tokens,
        "labels": list(spec.labels) if spec.type == "classification" else None,
        "schema": spec.json_schema if spec.type == "extraction" else None,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def make_header(
    *,
    data_sha256: str,
    adapter_sha256: str | None,
    alternatives: bool,
    backend: str,
    settings_sha256: str | None = None,
    base_model: str | None = None,
) -> dict[str, Any]:
    """The cache header line; ``base_model`` identifies the weights when there is no adapter (zero-shot)."""
    return {
        "type": HEADER_TYPE,
        "data_sha256": data_sha256,
        "adapter_sha256": adapter_sha256,
        "base_model": base_model,
        "alternatives": bool(alternatives),
        "backend": backend,
        "settings_sha256": settings_sha256,
    }


def header_compatible(cached: Mapping[str, Any], wanted: Mapping[str, Any]) -> bool:
    """Whether rows cached under ``cached`` serve ``wanted``: all fields equal; alternatives may be a superset."""
    keys = (set(cached) | set(wanted)) - {"alternatives"}
    if any(cached.get(k) != wanted.get(k) for k in keys):
        return False
    return bool(cached.get("alternatives")) or not bool(wanted.get("alternatives"))


# -- cache file --------------------------------------------------------------------------------------------------


def read_cache(path: Path) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """The header and the rows by record id; ``(None, {})`` when there is no usable cache.

    A truncated last line (an interrupted pass) is skipped.
    """
    if not path.is_file():
        return None, {}
    header: dict[str, Any] | None = None
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if number == 0:
                if item.get("type") != HEADER_TYPE:
                    return None, {}
                header = item
                continue
            if isinstance(item.get("id"), str) and "confidence" in item and "value" in item:
                rows[item["id"]] = item
    return header, rows


def prediction_row(record_id: str, pred: Prediction) -> dict[str, Any]:
    """One cache row."""
    return {
        "id": record_id,
        "answer": pred.answer,
        "value": pred.value,
        "confidence": float(pred.confidence),
        "alt_value": pred.alt_value,
        "alt_confidence": None if pred.alt_confidence is None else float(pred.alt_confidence),
        "latency_ms": float(pred.latency_ms),
        "prompt_tokens": int(pred.prompt_tokens),
        "completion_tokens": int(pred.completion_tokens),
        "field_confidences": pred.field_confidences,
    }


def _dumps(item: Mapping[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))


# -- backend -----------------------------------------------------------------------------------------------------


class LazyBackend:
    """Creates (and loads) the backend on first use, so a fully cached pass never loads a model."""

    def __init__(self, factory: Callable[[], ScoringBackend]) -> None:
        self._factory = factory
        self._backend: ScoringBackend | None = None
        self.created = 0

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def get(self) -> ScoringBackend:
        if self._backend is None:
            backend = self._factory()
            load = getattr(backend, "load", None)
            if callable(load):
                load()
            self._backend = backend
            self.created += 1
        return self._backend

    def label_trie(self, labels: list[str]) -> LabelTrie:
        return self.get().label_trie(labels)

    def generate_with_scores(
        self,
        messages: list[dict[str, str]],
        constraint: LabelTrie | None = None,
        max_tokens: int = 256,
    ) -> Generation:
        return self.get().generate_with_scores(messages, constraint=constraint, max_tokens=max_tokens)

    def close(self) -> None:
        """Drop the backend so its weights can be freed (and empty MLX's buffer cache when MLX is in use)."""
        if self._backend is None:
            return
        self._backend = None
        gc.collect()
        mx = sys.modules.get("mlx.core")
        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()


# -- prediction --------------------------------------------------------------------------------------------------


def _attach[S: (ValidationSplit, TestSplit)](split: S, rows: Mapping[str, Mapping[str, Any]]) -> S:
    ordered = [rows[r.id] for r in split.records]

    def optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    out = split.with_predictions(
        [row.get("value") for row in ordered],
        [float(row["confidence"]) for row in ordered],
        latencies_ms=[optional_float(row.get("latency_ms")) for row in ordered],
        alt_preds=[row.get("alt_value") for row in ordered],
        alt_confidences=[optional_float(row.get("alt_confidence")) for row in ordered],
    )
    records = [
        dataclasses.replace(
            record,
            extra={
                **record.extra,
                "answer": row.get("answer"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "field_confidences": row.get("field_confidences"),
            },
        )
        for record, row in zip(out.records, ordered, strict=True)
    ]
    return out.replace_records(records)


def predict_split[S: (ValidationSplit, TestSplit)](
    spec: TaskSpec,
    split: S,
    backend: ScoringBackend,
    *,
    cache_path: Path,
    alternatives: bool,
    header: Mapping[str, Any],
    messages_fn: MessagesFn | None = None,
    log: Log | None = None,
) -> S:
    """``split`` with the student's predictions, confidences and latencies attached (same split class).

    Cached rows are reused when the cache header matches ``header``; the rest are predicted (after ``WARMUP``
    untimed warm-up calls) and the cache is rewritten in split order. For classification the label trie is built
    once per pass.
    """
    wanted = {**header, "type": HEADER_TYPE, "alternatives": bool(alternatives)}
    cached_header, cached_rows = read_cache(cache_path)
    rows: dict[str, dict[str, Any]] = {}
    use_alternatives = bool(alternatives)
    if cached_header is not None and header_compatible(cached_header, wanted):
        rows = cached_rows
        if cached_header.get("alternatives"):
            use_alternatives = True
            wanted["alternatives"] = True
    missing = [r for r in split.records if r.id not in rows]
    if log is not None:
        log(f"  {split.name}: {len(split)} examples, {len(split) - len(missing)} cached")
    if not missing:
        return _attach(split, rows)

    trie = backend.label_trie(list(spec.labels)) if spec.type == "classification" else None

    def run(text: str, alts: bool) -> Prediction:
        messages = messages_fn(text) if messages_fn is not None else None
        return predict(spec, backend, text, trie=trie, alternatives=alts, messages=messages)

    for record in missing[:WARMUP]:
        run(record.input, False)
    step = max(1, len(missing) // _PROGRESS_STEPS)
    done = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        fh.write(_dumps(wanted) + "\n")
        for record in split.records:
            row = rows.get(record.id)
            if row is None:
                row = prediction_row(record.id, run(record.input, use_alternatives))
                rows[record.id] = row
                done += 1
                if log is not None and len(missing) >= _PROGRESS_MIN and (done % step == 0 or done == len(missing)):
                    log(f"  {split.name}: {done}/{len(missing)} predicted")
            fh.write(_dumps(row) + "\n")
            fh.flush()
    return _attach(split, rows)
