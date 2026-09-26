"""Curated evaluation data: ``data/<split>.jsonl`` and its line-aligned ``.meta.jsonl`` sidecar as typed splits.

A record's input is the (PII-scrubbed) user message the student sees; ``gold`` and ``teacher`` come from the sidecar,
which curate always writes (a missing sidecar is an error, never a silent fallback to teacher-only metrics).
The group is the value at ``curate.split.group_by`` when the spec sets it, else ``meta.group``; traits are
``meta.traits``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, overload

from taskdistill import paths
from taskdistill.config import TaskSpec
from taskdistill.evaluate.splits import EvalRecord, TaskType, TestSplit, ValidationSplit

SPLIT_NAMES: Final = ("train", "valid", "test")
EVAL_SPLITS: Final = ("valid", "test")
_CHUNK: Final = 1 << 20


class EvalDataError(ValueError):
    """The curated data needed for evaluation is missing or inconsistent."""


def data_file(task: str, split: str) -> Path:
    return paths.data_dir(task) / f"{split}.jsonl"


def meta_file(task: str, split: str) -> Path:
    return paths.data_dir(task) / f"{split}.meta.jsonl"


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_sha256(task: str, split: str) -> str:
    """SHA-256 of ``data/<split>.jsonl``: the predictions cache is valid only for these exact inputs."""
    path = data_file(task, split)
    if not path.is_file():
        raise EvalDataError(_missing(task, split))
    return file_sha256(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSON Lines file (blank lines skipped); every row must be an object."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalDataError(f"{paths.relative_to_home(path)}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise EvalDataError(f"{paths.relative_to_home(path)}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def _message(row: Mapping[str, Any], role: str) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def user_content(row: Mapping[str, Any]) -> str:
    """The last user message of a chat example (the scrubbed task input)."""
    return _message(row, "user") or ""


def assistant_content(row: Mapping[str, Any]) -> str | None:
    """The last assistant message of a chat example (the canonical teacher output)."""
    return _message(row, "assistant")


def meta_path_value(row: Mapping[str, Any], path: str) -> Any:
    """Value at a dotted path such as ``meta.template`` inside a sidecar row (None when any step is missing)."""
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _missing(task: str, split: str) -> str:
    rel = paths.relative_to_home(data_file(task, split))
    return f"no curated {split} split for task '{task}': {rel} is missing (run `taskdistill curate --task {task}`)"


def _as_value(task_type: TaskType, value: Any) -> Any:
    """Classification values as label strings; extraction values as objects (a JSON string is parsed)."""
    if value is None:
        return None
    if task_type == "classification":
        return str(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, Mapping) else None


def _teacher_from_row(task_type: TaskType, row: Mapping[str, Any]) -> Any:
    text = assistant_content(row)
    if text is None:
        return None
    return _as_value(task_type, text.strip() if task_type == "classification" else text)


def _group(spec: TaskSpec, sidecar: Mapping[str, Any]) -> str | None:
    group_by = spec.curate.split.group_by
    value = meta_path_value(sidecar, group_by) if group_by else meta_path_value(sidecar, "meta.group")
    if value is None or isinstance(value, (Mapping, list)):
        return None
    return str(value)


def _traits(sidecar: Mapping[str, Any]) -> tuple[str, ...]:
    value = meta_path_value(sidecar, "meta.traits")
    if isinstance(value, (list, tuple)):
        return tuple(str(t) for t in value)
    if isinstance(value, str) and value:
        return (value,)
    return ()


def _rows(spec: TaskSpec, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = data_file(spec.task, split)
    if not path.is_file():
        raise EvalDataError(_missing(spec.task, split))
    rows = read_jsonl(path)
    sidecar_path = meta_file(spec.task, split)
    if not sidecar_path.is_file():
        raise EvalDataError(
            f"{paths.relative_to_home(sidecar_path)} is missing: without it the gold labels, input hashes and groups "
            f"of the {split} split are unknown; re-run `taskdistill curate --task {spec.task}`"
        )
    metas = read_jsonl(sidecar_path)
    if len(metas) != len(rows):
        raise EvalDataError(
            f"{paths.relative_to_home(sidecar_path)} has {len(metas)} lines but {split}.jsonl has {len(rows)}; "
            "re-run curate"
        )
    return rows, metas


def load_records(spec: TaskSpec, split: str) -> list[EvalRecord]:
    """The records of any curated split (train, valid or test), in file order."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(SPLIT_NAMES)}")
    task_type: TaskType = spec.type
    rows, metas = _rows(spec, split)
    records: list[EvalRecord] = []
    for i, (row, sidecar) in enumerate(zip(rows, metas, strict=True)):
        teacher = (
            _as_value(task_type, sidecar["teacher"]) if "teacher" in sidecar else _teacher_from_row(task_type, row)
        )
        records.append(
            EvalRecord(
                id=str(sidecar.get("input_hash") or f"{split}-{i}"),
                input=user_content(row),
                gold=_as_value(task_type, sidecar.get("gold")),
                teacher=teacher,
                group=_group(spec, sidecar),
                traits=_traits(sidecar),
                extra={"line": i},
            )
        )
    return records


@overload
def load_split(spec: TaskSpec, split: Literal["valid"]) -> ValidationSplit: ...


@overload
def load_split(spec: TaskSpec, split: Literal["test"]) -> TestSplit: ...


@overload
def load_split(spec: TaskSpec, split: str) -> ValidationSplit | TestSplit: ...


def load_split(spec: TaskSpec, split: str) -> ValidationSplit | TestSplit:
    """``valid`` as a :class:`ValidationSplit`, ``test`` as a :class:`TestSplit`."""
    if split == "valid":
        return ValidationSplit(load_records(spec, "valid"), spec.type)
    if split == "test":
        return TestSplit(load_records(spec, "test"), spec.type)
    raise ValueError(f"unknown evaluation split {split!r}; expected valid or test")


def load_train_pairs(spec: TaskSpec) -> tuple[list[str], list[Any]]:
    """Training inputs and their teacher labels (records without a teacher label are left out)."""
    texts: list[str] = []
    labels: list[Any] = []
    for record in load_records(spec, "train"):
        if record.teacher is not None:
            texts.append(record.input)
            labels.append(record.teacher)
    return texts, labels
