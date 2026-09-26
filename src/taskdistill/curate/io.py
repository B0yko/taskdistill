"""Read curated data back: ``data/<split>.jsonl`` with its line-aligned ``<split>.meta.jsonl`` sidecar.

The raw (unscrubbed) input of an example is never written to ``data/``; :func:`raw_inputs_by_hash` recovers it
from the store with the same load and extraction rules curate uses, so a hash maps to the exact input curate
labelled with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taskdistill import paths
from taskdistill.config import TaskSpec
from taskdistill.curate.extract import input_hash
from taskdistill.curate.merge import SPLITS, extract_records, load_sources
from taskdistill.store import Store


class CuratedDataError(FileNotFoundError):
    """Curated data is missing or malformed; the message names the file and says how to fix it."""


@dataclass(frozen=True)
class CuratedRow:
    """One curated example: the training chat plus its sidecar fields."""

    messages: list[dict[str, str]]
    input_hash: str
    gold: Any
    teacher: Any
    meta: dict[str, Any]

    @property
    def input(self) -> str:
        """The PII-scrubbed input (the user message)."""
        return next(m["content"] for m in self.messages if m.get("role") == "user")

    @property
    def target(self) -> str:
        """The canonical teacher output (the assistant message)."""
        return next(m["content"] for m in reversed(self.messages) if m.get("role") == "assistant")


def split_files(task: str, split: str) -> tuple[Path, Path]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")
    directory = paths.home() / task / "data"  # reading never creates the task's directories
    return directory / f"{split}.jsonl", directory / f"{split}.meta.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise CuratedDataError(f"{paths.relative_to_home(path)}:{number}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise CuratedDataError(f"{paths.relative_to_home(path)}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def read_split(task: str, split: str) -> list[CuratedRow]:
    """The curated rows of ``split`` for ``task``, joined with their sidecar by line."""
    data_file, meta_file = split_files(task, split)
    for path in (data_file, meta_file):
        if not path.is_file():
            raise CuratedDataError(
                f"no curated data for task '{task}': {paths.relative_to_home(path)} is missing; "
                f"run `taskdistill curate --task {task}`"
            )
    rows, metas = _read_jsonl(data_file), _read_jsonl(meta_file)
    if len(rows) != len(metas):
        raise CuratedDataError(
            f"{paths.relative_to_home(meta_file)} has {len(metas)} rows but {data_file.name} has {len(rows)}; "
            f"re-run `taskdistill curate --task {task}`"
        )
    out: list[CuratedRow] = []
    for row, meta in zip(rows, metas, strict=True):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise CuratedDataError(f"{paths.relative_to_home(data_file)}: a row has no messages")
        out.append(
            CuratedRow(
                messages=messages,
                input_hash=str(meta.get("input_hash")),
                gold=meta.get("gold"),
                teacher=meta.get("teacher"),
                meta=meta.get("meta") or {},
            )
        )
    return out


def raw_inputs_by_hash(store: Store, spec: TaskSpec) -> dict[str, str]:
    """``{input_hash: raw input}`` from the store's captures (via ``input.from``) and imports.

    Records are read in curate's order and the first raw input of each hash wins, as in curate's merge.
    """
    captures, imports, _ = load_sources(store, spec.task)
    records, _ = extract_records(spec, captures, imports)
    out: dict[str, str] = {}
    for record in records:
        out.setdefault(input_hash(record.raw_input), record.raw_input)
    return out
