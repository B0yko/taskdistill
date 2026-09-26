"""Curate stages 1-5: load the store, extract inputs, merge by input hash, normalise outputs, scrub PII.

Every stage function returns its counts as a JSON-ready dict; the pipeline logs and records them. Examples are
ordered by input hash, so the result never depends on the order in which traffic was captured or imported.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from taskdistill.config import TaskSpec
from taskdistill.curate.dedupe import majority_vote
from taskdistill.curate.extract import InputUnparsed, extract_input, input_hash, normalise_text
from taskdistill.curate.normalise import normalise_output
from taskdistill.curate.pii import PiiScrubber
from taskdistill.store import CaptureRow, ImportRow, Store
from taskdistill.tasks.classification import normalise_label
from taskdistill.tasks.extraction import canonical_output, normalise_extraction
from taskdistill.teacher.client import message_text

SPLITS = ("train", "valid", "test")
#: Accepted spellings of a predefined split value.
SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "dev": "valid",
    "test": "test",
}
#: How strongly a split is protected: when merged records disagree, the most protected split wins.
SPLIT_RANK = {"train": 0, "valid": 1, "test": 2}


class CurateError(RuntimeError):
    """Curate cannot produce a dataset; the message says why and what to do."""


@dataclass
class Record:
    """One observation of a task input: a captured request/response, or an imported row."""

    raw_input: str
    output: str | None
    gold: Any
    meta: dict[str, Any]
    origin: str  # capture | pairs | inputs
    ts: float | None = None


@dataclass
class Example:
    """All records that share one input hash, carried through the later stages.

    ``raw_input`` (the unscrubbed input) stays in memory only: teacher labelling sends it, nothing writes it.
    ``text`` is the PII-scrubbed input the student trains on.
    """

    input_hash: str
    raw_input: str
    outputs: list[str] = field(default_factory=list)
    golds: list[Any] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    origins: list[str] = field(default_factory=list)
    repeats: int = 0
    teacher: Any = None
    teacher_origin: str | None = None  # recorded | labelled | invalid (a test row whose labelling answer was invalid)
    gold: Any = None
    text: str = ""
    split: str | None = None
    length: int | None = None


# meta paths -------------------------------------------------------------------------------------
def meta_keys(path: str | None) -> list[str]:
    """``"meta.a.b"`` -> ``["a", "b"]``; an empty list for None."""
    if not path:
        return []
    keys = path.split(".")
    if keys[0] == "meta":
        keys = keys[1:]
    if not keys or not all(keys):
        raise CurateError(f"invalid meta path {path!r}; expected e.g. meta.split")
    return keys


def meta_get(meta: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value: Any = meta
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _meta_pop(meta: dict[str, Any], keys: Sequence[str]) -> Any:
    parent: Any = meta
    for key in keys[:-1]:
        parent = parent.get(key) if isinstance(parent, dict) else None
    if isinstance(parent, dict):
        return parent.pop(keys[-1], None)
    return None


def meta_set(meta: dict[str, Any], keys: Sequence[str], value: Any) -> None:
    parent = meta
    for key in keys[:-1]:
        child = parent.get(key)
        if not isinstance(child, dict):
            child = parent[key] = {}
        parent = child
    parent[keys[-1]] = value


def canonical_split(value: Any, path: str = "meta.split") -> str | None:
    """``train``/``valid``/``test`` for a predefined split value (aliases accepted); None when unset."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    name = SPLIT_ALIASES.get(value.strip().lower()) if isinstance(value, str) else None
    if name is None:
        raise CurateError(
            f"{path} is {value!r}; expected train, valid or test (aliases: validation, val, dev for valid)"
        )
    return name


# stage 1: load ---------------------------------------------------------------------------------
NO_VALUE = "(none)"


def _request_teacher(request_body: str | None) -> tuple[str, str]:
    """The request's ``model`` and the SHA-256 of its system prompt (:data:`NO_VALUE` when absent)."""
    try:
        body = json.loads(request_body or "")
    except ValueError:
        return NO_VALUE, NO_VALUE
    if not isinstance(body, Mapping):
        return NO_VALUE, NO_VALUE
    model = body.get("model")
    prompt = NO_VALUE
    messages = body.get("messages")
    for message in messages if isinstance(messages, list) else []:
        if isinstance(message, Mapping) and message.get("role") in ("system", "developer"):
            text = message_text(message.get("content"))
            if text is not None:
                prompt = hashlib.sha256(text.encode("utf-8")).hexdigest()
            break
    return (model if isinstance(model, str) and model else NO_VALUE), prompt


def load_sources(
    store: Store, task: str, spec: TaskSpec | None = None
) -> tuple[list[CaptureRow], list[ImportRow], dict[str, Any]]:
    """Usable captures (captured, HTTP 200, request and response stored) and every import row of ``task``.

    Usable captures are counted by request model, upstream (served) model and system-prompt SHA-256; with
    ``spec``, those from another model or with another system prompt than the spec's teacher are counted too.
    Such captures are kept (an application may name the model differently from ``teacher.model``), and the
    dataset card lists them.
    """
    captures: list[CaptureRow] = []
    skipped: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_upstream: Counter[str] = Counter()
    by_prompt: Counter[str] = Counter()
    total = 0
    for row in store.iter_captures(task):
        total += 1
        if not row.captured:
            skipped["not_captured"] += 1
        elif row.status != 200:
            skipped["status_not_200"] += 1
        elif not row.request_body:
            skipped["no_request_body"] += 1
        elif not row.response_body:
            skipped["no_response_body"] += 1
        else:
            captures.append(row)
            by_source[row.source] += 1
            model, prompt = _request_teacher(row.request_body)
            by_model[model] += 1
            by_prompt[prompt] += 1
            by_upstream[row.upstream_model or NO_VALUE] += 1
    imports = list(store.iter_imports(task))
    costs = [row.cost_usd for row in captures if row.cost_usd is not None]
    stamps = [row.ts for row in captures]
    counts: dict[str, Any] = {
        "captures": {
            "total": total,
            "usable": len(captures),
            "by_source": dict(sorted(by_source.items())),
            "skipped": dict(sorted(skipped.items())),
            "by_model": dict(sorted(by_model.items())),
            "by_upstream_model": dict(sorted(by_upstream.items())),
            "by_system_prompt": dict(sorted(by_prompt.items())),
            "other_model": None if spec is None else len(captures) - by_model[spec.teacher.model],
            "other_prompt": None if spec is None else len(captures) - by_prompt[spec.teacher_prompt_sha256],
            "recorded_cost_usd": sum(costs) if costs else None,
            "first_ts": min(stamps) if stamps else None,
            "last_ts": max(stamps) if stamps else None,
        },
        "imports": {"total": len(imports), "by_format": dict(sorted(Counter(r.format for r in imports).items()))},
    }
    return captures, imports, counts


# stage 2: input extraction -------------------------------------------------------------------------
def _response_output(response_body: str) -> str | None:
    try:
        response = json.loads(response_body)
    except ValueError:
        return None
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return None
    return message_text(message.get("content"))


def extract_records(
    spec: TaskSpec, captures: Iterable[CaptureRow], imports: Iterable[ImportRow]
) -> tuple[list[Record], dict[str, Any]]:
    """Records from captures (input via ``input.from``, output = ``choices[0].message.content``) and imports."""
    records: list[Record] = []
    counts: Counter[str] = Counter()
    for row in captures:
        try:
            body = json.loads(row.request_body or "")
        except ValueError:
            counts["request_invalid"] += 1
            continue
        if not isinstance(body, Mapping):
            counts["request_invalid"] += 1
            continue
        try:
            raw = extract_input(body, spec.input)
        except InputUnparsed:
            counts["input_unparsed"] += 1
            continue
        if not normalise_text(raw):
            counts["empty_input"] += 1
            continue
        output = _response_output(row.response_body or "")
        if output is None:
            counts["missing_output"] += 1
            continue
        records.append(Record(raw, output, None, {}, "capture", row.ts))
        counts["from_captures"] += 1
    for imp in imports:
        if not normalise_text(imp.input):
            counts["empty_input"] += 1
            continue
        output = imp.output if imp.format == "pairs" else None
        meta = imp.meta if isinstance(imp.meta, dict) else {}
        records.append(Record(imp.input, output, imp.gold, meta, imp.format, imp.ts))
        counts["from_imports"] += 1
    out = {
        "records": len(records),
        "from_captures": counts["from_captures"],
        "from_imports": counts["from_imports"],
        "request_invalid": counts["request_invalid"],
        "input_unparsed": counts["input_unparsed"],
        "missing_output": counts["missing_output"],
        "empty_input": counts["empty_input"],
    }
    return records, out


# stage 3: merge ---------------------------------------------------------------------------------
def merge_records(records: Iterable[Record], split_path: str | None) -> tuple[list[Example], dict[str, Any]]:
    """One example per input hash of the raw input, ordered by hash.

    Teacher outputs (captures and ``pairs`` outputs) and golds (imports) are collected for later votes. Meta dicts
    are merged with the first value winning (conflicts counted per key); the predefined split field is resolved
    apart: when members disagree the most protected split wins (test > valid > train), counted as
    ``cross_split_merged``. Records of one origin kind with the same predefined split are exact repeats of the input
    (a captured record joining an imported one is not); ``example.repeats`` holds the largest such count minus one,
    so the dedupe stage can report them per split.
    """
    split_keys = meta_keys(split_path)
    groups: dict[str, list[Record]] = {}
    n_records = 0
    for record in records:
        n_records += 1
        groups.setdefault(input_hash(record.raw_input), []).append(record)

    examples: list[Example] = []
    conflicts: Counter[str] = Counter()
    cross_split = 0
    for key in sorted(groups):
        members = groups[key]
        example = Example(input_hash=key, raw_input=members[0].raw_input, text=members[0].raw_input)
        splits: list[str] = []
        conflicted: set[str] = set()
        copies: Counter[tuple[str, str | None]] = Counter()
        for record in members:
            example.origins.append(record.origin)
            if record.output is not None:
                example.outputs.append(record.output)
            if record.gold is not None:
                example.golds.append(record.gold)
            meta = copy.deepcopy(record.meta)
            name = None
            if split_keys:
                name = canonical_split(_meta_pop(meta, split_keys), split_path or "meta.split")
                if name is not None:
                    splits.append(name)
            copies[(record.origin, name)] += 1
            for meta_key, value in meta.items():
                if meta_key not in example.meta:
                    example.meta[meta_key] = value
                elif example.meta[meta_key] != value:
                    conflicted.add(meta_key)
        conflicts.update(conflicted)
        per_origin: Counter[str] = Counter()
        for (origin, _), n in copies.items():
            per_origin[origin] += n - 1
        example.repeats = max(per_origin.values())
        if splits:
            distinct = set(splits)
            if len(distinct) > 1:
                cross_split += 1
            meta_set(example.meta, split_keys, max(distinct, key=SPLIT_RANK.__getitem__))
        examples.append(example)

    counts = {
        "records": n_records,
        "examples": len(examples),
        "merged_records": n_records - len(examples),
        "with_teacher_output": sum(1 for ex in examples if ex.outputs),
        "with_gold": sum(1 for ex in examples if ex.golds),
        "with_both": sum(1 for ex in examples if ex.outputs and ex.golds),
        "cross_split_merged": cross_split,
        "repeated_inputs": sum(ex.repeats for ex in examples),
        "meta_conflicts": dict(sorted(conflicts.items())),
    }
    return examples, counts


# stage 4: output normalisation --------------------------------------------------------------------
def normalise_gold(spec: TaskSpec, gold: Any) -> Any:
    """The canonical label (classification) or the schema-valid object (extraction); None when invalid."""
    if spec.type == "classification":
        return normalise_label(gold, spec.labels) if isinstance(gold, str) else None
    assert spec.json_schema is not None
    if isinstance(gold, str):
        text = gold
    elif isinstance(gold, dict):
        try:
            text = json.dumps(gold, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return None
    else:
        return None
    obj, _ = normalise_extraction(text, spec.json_schema)
    return obj


def target_text(spec: TaskSpec, value: Any) -> str:
    """The assistant message the student trains on: the label, or compact JSON in schema property order."""
    if spec.type == "classification":
        return str(value)
    assert spec.json_schema is not None
    return canonical_output(value, spec.json_schema)


def normalise_examples(spec: TaskSpec, examples: Iterable[Example]) -> tuple[list[Example], dict[str, Any]]:
    """Normalise teacher outputs and golds; resolve each by strict majority.

    Invalid outputs are dropped and counted. An example whose outputs were all invalid is dropped (never
    relabelled), and so is one whose valid outputs have no strict majority. Examples without any output are kept
    for teacher labelling. Invalid golds become None; conflicting golds resolve by strict majority, else None.
    """
    if spec.type == "classification" and not spec.labels:
        raise CurateError(f"task {spec.task}: no labels loaded")
    if spec.type == "extraction" and not spec.json_schema:
        raise CurateError(f"task {spec.task}: no JSON Schema loaded")
    counts: Counter[str] = Counter()
    kept: list[Example] = []
    for ex in examples:
        counts["examples"] += 1
        gold_values = []
        for gold in ex.golds:
            value = normalise_gold(spec, gold)
            if value is None:
                counts["gold_invalid"] += 1
            else:
                gold_values.append(value)
        if gold_values:
            ok, winner = majority_vote(gold_values)
            distinct = {json.dumps(v, sort_keys=True, ensure_ascii=False) for v in gold_values}
            if len(distinct) > 1:
                counts["gold_conflicts_resolved" if ok else "gold_conflicts_unresolved"] += 1
            ex.gold = winner if ok else None
        if not ex.outputs:
            counts["unlabelled"] += 1
            kept.append(ex)
            continue
        values = []
        for output in ex.outputs:
            counts["outputs"] += 1
            value, _ = normalise_output(spec, output)
            if value is None:
                counts["invalid_outputs"] += 1
            else:
                values.append(value)
        if not values:
            counts["dropped_all_invalid"] += 1
            continue
        ok, winner = majority_vote(values)
        if not ok:
            counts["dropped_no_majority"] += 1
            continue
        if len({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in values}) > 1:
            counts["conflicts_resolved"] += 1
        ex.teacher = winner
        ex.teacher_origin = "recorded"
        kept.append(ex)
    names = (
        "examples", "outputs", "invalid_outputs", "dropped_all_invalid", "dropped_no_majority", "conflicts_resolved",
        "unlabelled", "gold_invalid", "gold_conflicts_resolved", "gold_conflicts_unresolved",
    )  # fmt: skip
    out: dict[str, Any] = {name: counts[name] for name in names}
    out["kept"] = len(kept)
    out["labelled"] = sum(1 for ex in kept if ex.teacher is not None)
    out["with_gold"] = sum(1 for ex in kept if ex.gold is not None)
    return kept, out


# stage 5: PII scrub -------------------------------------------------------------------------------
class PiiStage:
    """Scrubs inputs, teacher values and golds with the task's PII settings and keeps hits per kind.

    Classification values are labels from the task's own label set, not user data, so they are left as they are.
    ``examples_changed`` counts distinct examples (by input hash), however often an example is scrubbed.
    """

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        pii = spec.curate.pii
        self.scrubber = PiiScrubber(pii.kinds) if pii.enabled else None
        self.hits: dict[str, Counter[str]] = {"input": Counter(), "teacher": Counter(), "gold": Counter()}
        self.changed: set[str] = set()

    @property
    def examples_changed(self) -> int:
        return len(self.changed)

    def _value(self, value: Any, found: Counter[str]) -> Any:
        if self.scrubber is None or value is None or self.spec.type == "classification":
            return value
        scrubbed, hits = self.scrubber.scrub_value(value)
        found.update(hits)
        return scrubbed

    def scrub(self, ex: Example, *, text: bool = True, teacher: bool = True, gold: bool = True) -> Counter[str]:
        """Scrub the selected fields of ``ex`` in place; returns this call's hits per kind."""
        found: dict[str, Counter[str]] = {name: Counter() for name in self.hits}
        if text:
            ex.text = ex.raw_input
            if self.scrubber is not None:
                ex.text, hits = self.scrubber.scrub(ex.raw_input)
                found["input"].update(hits)
        if teacher:
            ex.teacher = self._value(ex.teacher, found["teacher"])
        if gold:
            ex.gold = self._value(ex.gold, found["gold"])
        total: Counter[str] = Counter()
        for name, hits in found.items():
            self.hits[name].update(hits)
            total.update(hits)
        if total:
            self.changed.add(ex.input_hash)
        return total

    def counts(self) -> dict[str, Any]:
        total: Counter[str] = Counter()
        for bucket in self.hits.values():
            total.update(bucket)
        kinds = list(self.scrubber.kinds) if self.scrubber is not None else []
        return {
            "enabled": self.scrubber is not None,
            "kinds": kinds,
            "hits": {kind: total[kind] for kind in kinds},
            "by_field": {name: {kind: bucket[kind] for kind in kinds} for name, bucket in self.hits.items()},
            "total": sum(total.values()),
            "examples_changed": self.examples_changed,
        }


def scrub_examples(stage: PiiStage, examples: Iterable[Example]) -> dict[str, Any]:
    """Stage 5: scrub every example's input, teacher value and gold."""
    for ex in examples:
        stage.scrub(ex)
    return stage.counts()
