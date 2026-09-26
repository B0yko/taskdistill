"""``taskdistill curate``: turn captured and imported traffic into train/valid/test chat data.

Eleven stages run in order, each logged as ``[k/11] <stage>: <counts>`` and recorded under
``curate_stats.json["stages"]``: load, input extraction, merge by input hash, output normalisation, PII scrub, split
assignment, dedupe inside each split, cross-split removal, teacher labelling, the length filter and an independent
leakage check. Nothing is written unless every stage succeeds and the train split is not empty; the files are then
written atomically (temporary file, then rename) under ``$TASKDISTILL_HOME/<task>/data/``. The split sizes after
each stage that can remove rows are recorded under ``curate_stats.json["split_sizes"]``, and the SHA-256 of every
data file under ``["files"]``, so a later step can tell which data a run was trained on.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taskdistill import paths
from taskdistill.config import TaskSpec
from taskdistill.curate.card import CARD_INFO_KEYS, SIZE_STAGES, card_source, render_card, unknown_card_keys
from taskdistill.curate.dedupe import leakage_check
from taskdistill.curate.label import DEFAULT_PROJECTION_SAMPLE, label_examples
from taskdistill.curate.length import example_messages, length_filter, model_label
from taskdistill.curate.merge import (
    SPLITS,
    CurateError,
    Example,
    PiiStage,
    extract_records,
    load_sources,
    merge_records,
    normalise_examples,
    scrub_examples,
)
from taskdistill.curate.split import assign_splits, by_split, dedupe_within, remove_cross_split
from taskdistill.store import Store
from taskdistill.teacher.base import TeacherSource

__all__ = ["STAGES", "CurateError", "CurateResult", "run_curate"]

#: Stage keys (as recorded in ``stats["stages"]``) and the names they are logged under.
STAGES: tuple[tuple[str, str], ...] = (
    ("load", "load"),
    ("extract", "input extraction"),
    ("merge", "merge"),
    ("normalise", "output normalisation"),
    ("pii", "PII scrub"),
    ("split", "split"),
    ("dedupe", "dedupe"),
    ("cross_split", "cross-split removal"),
    ("label", "teacher labelling"),
    ("length", "length filter"),
    ("leakage", "leakage check"),
)


@dataclass
class CurateResult:
    stats: dict[str, Any]
    paths: dict[str, Path]


def _kinds(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {n}" for name, n in counts.items() if n)


def _summary(name: str, c: Mapping[str, Any]) -> str:
    if name == "load":
        captures, imports = c["captures"], c["imports"]
        skipped = _kinds(captures["skipped"])
        formats = _kinds(imports["by_format"])
        other = _kinds(
            {"another model": captures["other_model"] or 0, "another system prompt": captures["other_prompt"] or 0}
        )
        return (
            f"{captures['usable']} usable captures of {captures['total']}"
            + (f" (skipped: {skipped})" if skipped else "")
            + (f" (from {other} than the spec's teacher; kept, listed in the card)" if other else "")
            + f", {imports['total']} imported rows"
            + (f" ({formats})" if formats else "")
        )
    if name == "extract":
        problems = _kinds({k: c[k] for k in ("input_unparsed", "missing_output", "request_invalid", "empty_input")})
        return f"{c['records']} records ({c['from_captures']} from captures, {c['from_imports']} from imports); " + (
            problems or "no input unparsed, no output missing"
        )
    if name == "merge":
        return (
            f"{c['examples']} examples from {c['records']} records ({c['with_teacher_output']} with teacher output, "
            f"{c['with_gold']} with gold, {c['with_both']} with both); {c['repeated_inputs']} repeated inputs, "
            f"{c['cross_split_merged']} cross-split merged, {sum(c['meta_conflicts'].values())} meta conflicts"
        )
    if name == "normalise":
        kept_test = c["kept_test_all_invalid"] + c["kept_test_no_majority"]
        return (
            f"{c['kept']} kept ({c['labelled']} with teacher output, {c['unlabelled']} to label"
            + (f", {kept_test} predefined test rows without a teacher value" if kept_test else "")
            + f"); {c['invalid_outputs']} invalid outputs, dropped {c['dropped_all_invalid']} all-invalid and "
            f"{c['dropped_no_majority']} without a majority; {c['gold_invalid']} invalid gold"
        )
    if name == "pii":
        if not c["enabled"]:
            return "disabled"
        invalid = _kinds(
            {"examples dropped": c["pii_schema_invalid"], "golds set to None": c["pii_schema_invalid_gold"]}
        )
        return f"{c['total']} hits ({_kinds(c['hits']) or 'none'}) in {c['examples_changed']} examples" + (
            f"; scrubbed values failing the JSON Schema: {invalid}" if invalid else ""
        )
    if name == "split":
        predefined, randomly = sum(c["predefined"].values()), sum(c["random"].values())
        detail = "all predefined" if not randomly else f"{predefined} predefined, {randomly} {c['method']}"
        return ", ".join(f"{s} {c['splits'][s]}" for s in SPLITS) + f" ({detail})"
    if name == "dedupe":
        return (
            f"removed {c['removed_exact']} exact and {c['removed_near']} near-duplicates, "
            f"{c['conflict_clusters']} conflicting clusters dropped ({c['conflict_examples']} examples), "
            f"{c['gold_conflicts']} gold conflicts; " + ", ".join(f"{s} {c['splits'][s]['after']}" for s in SPLITS)
        )
    if name == "cross_split":
        removed = c["removed"]
        return f"removed train {removed['train']}, valid {removed['valid']} (test is never changed)"
    if name == "label":
        if not c["requested"]:
            return "nothing to label"
        return (
            f"{c['labelled']} labelled of {c['requested']} ({c['live']} live, {c['cached']} cached, "
            f"{c['replayed']} replayed; ${c['cost_usd']:.4f}); {c['invalid']} invalid "
            f"({c['kept_invalid_test']} test rows kept without a teacher value), {c['truncated']} truncated"
            + (
                f"; {c['pii_schema_invalid']} dropped, their scrubbed answer fails the JSON Schema"
                if c.get("pii_schema_invalid")
                else ""
            )
        )
    if name == "length":
        parts = []
        for s in SPLITS:
            split = c["splits"][s]
            if split["n"]:
                parts.append(f"{s} p50 {split['p50']:g} p95 {split['p95']:g} max {split['max']}")
        return f"dropped {c['dropped']} over {c['max_seq_len']} tokens" + (f"; {', '.join(parts)}" if parts else "")
    if name == "leakage":
        verdict = "ok" if c["ok"] else "FAILED"
        return f"{verdict} ({c['exact']} exact, {c['near']} near-duplicate pairs across train/valid/test)"
    raise KeyError(name)


def _iso_date(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, UTC).date().isoformat()


def _distribution(spec: TaskSpec, splits: Mapping[str, Sequence[Example]]) -> dict[str, Any]:
    if spec.type == "classification":

        def labels(values: list[Any]) -> dict[str, int]:
            counts = Counter(str(v) for v in values if v is not None)
            ordered = {label: counts.pop(label, 0) for label in spec.labels}
            ordered.update(sorted(counts.items()))
            return ordered

        return {
            "kind": "label",
            "teacher": {s: labels([ex.teacher for ex in splits[s]]) for s in SPLITS},
            "gold": {s: labels([ex.gold for ex in splits[s]]) for s in SPLITS},
        }
    fields = spec.schema_fields

    def non_null(values: list[Any]) -> dict[str, int]:
        return {f: sum(1 for v in values if isinstance(v, dict) and v.get(f) is not None) for f in fields}

    return {
        "kind": "field",
        "fields": fields,
        "teacher": {s: non_null([ex.teacher for ex in splits[s]]) for s in SPLITS},
        "gold": {s: non_null([ex.gold for ex in splits[s]]) for s in SPLITS},
    }


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def _sizes(splits: Mapping[str, Sequence[Example]]) -> dict[str, int]:
    return {name: len(splits.get(name, [])) for name in SPLITS}


#: Why a stage can leave a split empty, for the error and warning messages.
EMPTIED_BY = {
    "split": "no row was assigned to it (every row has a predefined split elsewhere, or there are too few rows "
    "for the valid/test fractions)",
    "dedupe": "dedupe within splits removed every row (duplicate clusters with conflicting outputs are dropped)",
    "cross_split": "every row duplicates a row of a more protected split",
    "label": "every row's teacher answer was invalid, or failed the JSON Schema after the PII scrub",
    "length": "every row is longer than train.max_seq_len",
}


def _emptied_at(sizes: Mapping[str, Mapping[str, int]], name: str) -> tuple[str, str] | None:
    """The first stage after which split ``name`` was empty, as ``(title, reason)``."""
    for key, title in SIZE_STAGES:
        if key in sizes and sizes[key][name] == 0:
            return title, EMPTIED_BY[key]
    return None


def _require_train(spec: TaskSpec, sizes: Mapping[str, Mapping[str, int]]) -> None:
    emptied = _emptied_at(sizes, "train")
    if emptied is not None:
        title, reason = emptied
        raise CurateError(
            f"the train split of task '{spec.task}' is empty after {title}: {reason}; nothing was written. Add "
            f"training data (captures, or imports without a predefined valid/test split) and re-run "
            f"`taskdistill curate --task {spec.task}`"
        )


def _warn_empty(sizes: Mapping[str, Mapping[str, int]], log: Callable[[str], None]) -> None:
    consequence = {
        "valid": "training has no validation loss and eval cannot choose a threshold",
        "test": "eval has nothing to report on the test split",
    }
    for name, why in consequence.items():
        emptied = _emptied_at(sizes, name)
        if emptied is not None:
            log(f"warning: the {name} split is empty after {emptied[0]}: {emptied[1]}; {why}")


def _write_atomic(files: Mapping[Path, bytes]) -> None:
    """Write every file to a temporary sibling first, then rename them all into place."""
    staged: list[tuple[Path, Path]] = []
    try:
        for path, content in files.items():
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_bytes(content)
            staged.append((tmp, path))
        for tmp, path in staged:
            tmp.replace(path)
    except BaseException:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise


def output_paths(task: str) -> dict[str, Path]:
    directory = paths.data_dir(task)
    out: dict[str, Path] = {"data_dir": directory}
    for split in SPLITS:
        out[split] = directory / f"{split}.jsonl"
        out[f"{split}_meta"] = directory / f"{split}.meta.jsonl"
    out["stats"] = directory / "curate_stats.json"
    out["card"] = directory / "dataset_card.md"
    out["labelling_keys"] = directory / "labelling_keys.txt"
    return out


def run_curate(
    spec: TaskSpec,
    *,
    store: Store,
    teacher: TeacherSource | None,
    yes: bool = False,
    max_usd: float | None = None,
    tokenizer: Any = None,
    log: Callable[[str], None] = print,
    projection_sample: int = DEFAULT_PROJECTION_SAMPLE,
    card_info: Mapping[str, Any] | None = None,
) -> CurateResult:
    """Curate the store's data for ``spec`` and write ``data/``; see the module docstring for the stages.

    ``teacher`` labels examples without a recorded output (required only when there are some). ``max_usd`` is only
    recorded: the caller sets the run cap on the teacher. ``tokenizer`` defaults to the student base model's
    tokenizer. ``card_info`` (:data:`~taskdistill.curate.card.CARD_INFO_KEYS`: ``name``, ``source``, ``licence``,
    ``attribution``, ``citation``, ``split_rule``) goes into the dataset card; other keys are ignored with a
    warning. Labelling releases the teacher's connections when its batch ends; the caller still owns the teacher.
    Raises :class:`CurateError` (and writes nothing) when the train split ends up empty; an empty valid or test
    split is a logged warning.
    """
    started = time.perf_counter()
    ignored = unknown_card_keys(card_info)
    if ignored:
        log(
            f"warning: card_info key(s) {', '.join(ignored)} ignored; the dataset card uses {', '.join(CARD_INFO_KEYS)}"
        )
    stages: dict[str, Any] = {}
    numbers = {key: k for k, (key, _) in enumerate(STAGES, start=1)}
    titles = dict(STAGES)

    def done(key: str, counts: dict[str, Any]) -> None:
        stages[key] = counts
        log(f"[{numbers[key]}/{len(STAGES)}] {titles[key]}: {_summary(key, counts)}")

    captures, imports, counts = load_sources(store, spec.task, spec)
    done("load", counts)
    records, counts = extract_records(spec, captures, imports)
    done("extract", counts)
    if not records:
        raise CurateError(
            f"no usable captured or imported data for task '{spec.task}': capture traffic with "
            f"`taskdistill capture --task {spec.task}` or import a file with "
            f"`taskdistill capture --task {spec.task} --import <file.jsonl> --format openai|pairs|inputs`"
        )
    examples, counts = merge_records(records, spec.curate.split.predefined)
    done("merge", counts)
    examples, counts = normalise_examples(spec, examples)
    done("normalise", counts)
    pii = PiiStage(spec)
    examples, counts = scrub_examples(pii, examples)
    done("pii", counts)
    warning = pii.schema_warning("teacher", "gold")
    if warning:
        log(warning)
    done("split", assign_splits(spec, examples))
    sizes: dict[str, dict[str, int]] = {"split": _sizes(by_split(examples))}

    threshold = spec.curate.dedupe.near_dup_jaccard
    splits, dedupe_counts = dedupe_within(by_split(examples), threshold, exact=spec.curate.dedupe.exact)
    done("dedupe", dedupe_counts)
    sizes["dedupe"] = _sizes(splits)
    splits, cross_counts = remove_cross_split(splits, threshold)
    done("cross_split", cross_counts)
    sizes["cross_split"] = _sizes(splits)
    _require_train(spec, sizes)  # before any teacher spend

    ordered = [ex for name in SPLITS for ex in splits[name]]
    kept, label_stats, keys = label_examples(
        spec, ordered, teacher, yes=yes, max_usd=max_usd, projection_sample=projection_sample, log=log
    )
    labelled_hits: Counter[str] = Counter()
    for ex in kept:
        if ex.teacher_origin == "labelled":
            labelled_hits.update(pii.scrub(ex, text=False, gold=False))
    label_stats["pii_hits"] = dict(sorted(labelled_hits.items()))
    before = len(kept)
    kept = [ex for ex in kept if not pii.rejected(ex)]  # the scrubbed answer fails the JSON Schema
    label_stats["pii_schema_invalid"] = before - len(kept)
    done("label", label_stats)
    warning = pii.schema_warning("labelled")
    if warning:
        log(warning)

    splits = by_split(kept)
    sizes["label"] = _sizes(splits)
    if tokenizer is None and kept:
        from taskdistill.models import load_tokenizer

        tokenizer = load_tokenizer(spec.student.base_model)
    splits, length_counts = length_filter(spec, splits, tokenizer)
    done("length", length_counts)
    sizes["length"] = _sizes(splits)

    report = leakage_check({name: [ex.text for ex in splits[name]] for name in SPLITS}, threshold)
    done("leakage", {"ok": report.ok, "exact": report.exact, "near": report.near, "sizes": dict(report.sizes)})
    if not report.ok:
        pairs = ", ".join(f"{name} {n}" for name, n in report.by_split_pair().items() if n)
        raise CurateError(
            f"leakage check failed: {report.exact} exact and {report.near} near-duplicate pairs across splits "
            f"({pairs}); nothing was written"
        )
    _require_train(spec, sizes)
    _warn_empty(sizes, log)

    load = stages["load"]["captures"]
    stats: dict[str, Any] = {
        "task": spec.task,
        "task_type": spec.type,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "splits": {name: len(splits[name]) for name in SPLITS},
        "split_sizes": sizes,
        "sources": {
            name: {
                "recorded": sum(1 for ex in splits[name] if ex.teacher_origin == "recorded"),
                "labelled": sum(1 for ex in splits[name] if ex.teacher_origin == "labelled"),
                "teacher_invalid": sum(1 for ex in splits[name] if ex.teacher is None),
                "with_gold": sum(1 for ex in splits[name] if ex.gold is not None),
            }
            for name in SPLITS
        },
        "distribution": _distribution(spec, splits),
        "pii": pii.counts(),
        "dedupe": dedupe_counts,
        "cross_split": cross_counts,
        "labelling": label_stats,
        "length": length_counts,
        "leakage": report.to_dict(),
        "captured_from": _iso_date(load["first_ts"]),
        "captured_to": _iso_date(load["last_ts"]),
        "teacher": {
            "model": spec.teacher.model,
            "provider": spec.teacher.provider,
            "prompt_sha256": spec.teacher_prompt_sha256,
            "temperature": spec.teacher.temperature,
            "max_tokens": spec.teacher.max_tokens,
            "mode": teacher.mode if teacher is not None else None,
        },
        "student": {
            "base_model": model_label(spec.student.base_model),
            "system_prompt": spec.student.system_prompt,
            "max_seq_len": spec.train.max_seq_len,
        },
        "source": {k: v for k, v in card_source(card_info).items() if k in ("name", "source", "licence", "split_rule")},
        "stages": stages,
    }

    out = output_paths(spec.task)
    files: dict[Path, bytes] = {}
    for name in SPLITS:
        files[out[name]] = _jsonl([{"messages": example_messages(spec, ex)} for ex in splits[name]])
        files[out[f"{name}_meta"]] = _jsonl(
            [
                {"input_hash": ex.input_hash, "gold": ex.gold, "teacher": ex.teacher, "meta": ex.meta}
                for ex in splits[name]
            ]
        )
    files[out["labelling_keys"]] = "".join(key + "\n" for key in keys).encode("utf-8")
    stats["files"] = {path.name: hashlib.sha256(content).hexdigest() for path, content in files.items()}
    stats["elapsed_s"] = round(time.perf_counter() - started, 3)
    files[out["card"]] = render_card(stats, card_info).encode("utf-8")
    files[out["stats"]] = (json.dumps(stats, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _write_atomic(files)
    return CurateResult(stats=stats, paths=out)
