"""``dataset_card.md``: a Markdown card generated from ``curate_stats.json``.

It covers the source and licence, counts per split (and after each stage that removes rows), the label or field
distribution, dedupe, cross-split and PII counts, length statistics, the teacher (slug, provider, prompt SHA-256,
and the models and prompts the captures actually used) and the labelling cost and date. It never contains an
absolute path. ``card_info`` comes from the caller (the demos pass their name, attribution and split rule); keys
outside :data:`CARD_INFO_KEYS` are ignored, and the pipeline logs them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from taskdistill.curate.merge import NO_VALUE, SPLITS

DEFAULT_SOURCE = "user-provided data"
DEFAULT_LICENCE = "not recorded"
#: ``card_info`` keys: ``name`` titles the card, ``split_rule`` describes how the caller assigned predefined splits.
CARD_INFO_KEYS = ("name", "source", "licence", "attribution", "citation", "split_rule")
#: Stages after which split sizes are recorded (``stats["split_sizes"]``), with the names the card uses.
SIZE_STAGES = (
    ("split", "split assignment"),
    ("dedupe", "dedupe within splits"),
    ("cross_split", "cross-split removal"),
    ("label", "teacher labelling"),
    ("length", "length filter"),
)


def _n(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.1f}" if not value.is_integer() else f"{int(value):,}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table(header: list[str], rows: list[list[Any]], align_left: int = 1) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" if i < align_left else "---:" for i in range(len(header))) + " |")
    lines.extend("| " + " | ".join(_n(cell) for cell in row) + " |" for row in rows)
    return lines


def _usd(value: Any) -> str:
    return "-" if value is None else f"${float(value):.4f}"


def unknown_card_keys(card_info: Mapping[str, Any] | None) -> list[str]:
    """``card_info`` keys the card does not use."""
    return sorted(set(card_info or {}) - set(CARD_INFO_KEYS))


def card_source(card_info: Mapping[str, Any] | None) -> dict[str, str | None]:
    """The card's source facts; the source and licence fall back to "user-provided data" / "not recorded"."""
    info = dict(card_info or {})

    def optional(key: str) -> str | None:
        return str(info[key]) if info.get(key) else None

    return {
        "name": optional("name"),
        "source": str(info.get("source") or DEFAULT_SOURCE),
        "licence": str(info.get("licence") or DEFAULT_LICENCE),
        "attribution": optional("attribution"),
        "citation": optional("citation"),
        "split_rule": optional("split_rule"),
    }


def _counts(counts: Mapping[str, Any], limit: int = 8) -> str:
    """``a 3, b 1`` for the largest counts, with the rest summed."""
    ordered = sorted(((str(k), int(v)) for k, v in counts.items() if v), key=lambda item: (-item[1], item[0]))
    shown = [f"`{k}` {_n(v)}" if k != NO_VALUE else f"none {_n(v)}" for k, v in ordered[:limit]]
    rest = sum(v for _, v in ordered[limit:])
    return ", ".join(shown + ([f"{len(ordered) - limit} others {_n(rest)}"] if rest else []))


def _split_sentence(stage: Mapping[str, Any]) -> str:
    field = stage.get("predefined_field")
    predefined = sum(stage.get("predefined", {}).values())
    randomly = sum(stage.get("random", {}).values())
    fractions = stage.get("fractions", {})
    how = {"grouped": f"by group (`{stage.get('group_by')}`)", "stratified": "stratified by label"}.get(
        str(stage.get("method")), "at random"
    )
    parts = []
    if predefined:
        parts.append(f"{_n(predefined)} rows kept their predefined split (`{field}`).")
    if randomly:
        parts.append(
            f"{_n(randomly)} rows were split {how} with seed {stage.get('seed')} "
            f"(valid {fractions.get('val')}, test {fractions.get('test')}); rows are ordered by a seeded hash of "
            "their input (or group), so adding data later moves only rows at a split boundary."
        )
    return " ".join(parts)


def render_card(stats: Mapping[str, Any], card_info: Mapping[str, Any] | None = None) -> str:
    """The dataset card for one curate run."""
    source = card_source(card_info)
    stages: Mapping[str, Any] = stats.get("stages", {})
    teacher: Mapping[str, Any] = stats.get("teacher", {})
    labelling: Mapping[str, Any] = stats.get("labelling", {})
    splits: Mapping[str, int] = stats.get("splits", {})
    sources: Mapping[str, Mapping[str, int]] = stats.get("sources", {})
    out: list[str] = [f"# Dataset card: {source['name'] or stats.get('task')}", ""]
    out.append(
        f"Curated by `taskdistill curate` for task `{stats.get('task')}` on {str(stats.get('date', ''))[:10]} (UTC). "
        f"Task type: {stats.get('task_type')}. The student trains on teacher outputs; gold labels, where present, "
        "are used only for evaluation."
    )
    out += ["", "## Source and licence", "", f"- Source: {source['source']}", f"- Licence: {source['licence']}"]
    if source["attribution"]:
        out += ["", source["attribution"]]
    if source["citation"]:
        out += ["", "```bibtex", source["citation"], "```"]

    out += ["", "## Splits", ""]
    rows = [
        [
            name,
            splits.get(name, 0),
            sources.get(name, {}).get("recorded", 0),
            sources.get(name, {}).get("labelled", 0),
            sources.get(name, {}).get("teacher_invalid", 0),
            sources.get(name, {}).get("with_gold", 0),
        ]
        for name in SPLITS
    ]
    header = ["split", "examples", "teacher output recorded", "labelled by curate", "no valid teacher output"]
    out += _table([*header, "with gold"], rows)
    split_stage = stages.get("split", {})
    if split_stage:
        out += ["", _split_sentence(split_stage)]
    if source["split_rule"]:
        out += ["", f"Predefined split rule (from the data source): {source['split_rule'].rstrip('.')}."]
    sizes: Mapping[str, Mapping[str, int]] = stats.get("split_sizes", {})
    if sizes:
        out += ["", "Rows per split after each stage that can remove rows:", ""]
        rows = [[title, *(sizes[key].get(name, 0) for name in SPLITS)] for key, title in SIZE_STAGES if key in sizes]
        out += _table(["after", *SPLITS], rows)
        out += [
            "",
            "Test rows are never removed by cross-split removal or labelling: a test row whose teacher answer is "
            "invalid is kept without a teacher value (it counts against the teacher in evaluation). After dedupe "
            "within splits, only the length filter can remove test rows.",
        ]

    distribution: Mapping[str, Any] = stats.get("distribution", {})
    teacher_dist: Mapping[str, Mapping[str, int]] = distribution.get("teacher", {})
    if distribution.get("kind") == "label":
        out += ["", "## Label distribution (teacher labels)", ""]
        names: list[str] = []
        for name in SPLITS:
            names += [label for label in teacher_dist.get(name, {}) if label not in names]
        rows = [[label, *(teacher_dist.get(name, {}).get(label, 0) for name in SPLITS)] for label in names]
        out += _table(["label", *SPLITS], rows)
    elif distribution.get("kind") == "field":
        out += ["", "## Field distribution (non-null teacher values)", ""]
        fields: list[str] = list(distribution.get("fields", []))
        rows = [[f, *(teacher_dist.get(name, {}).get(f, 0) for name in SPLITS)] for f in fields]
        out += _table(["field", *SPLITS], rows)

    out += ["", "## Curation", ""]
    load = stages.get("load", {})
    captures, imports = load.get("captures", {}), load.get("imports", {})
    extract = stages.get("extract", {})
    merge = stages.get("merge", {})
    norm = stages.get("normalise", {})
    dedupe = stats.get("dedupe", {})
    cross = stats.get("cross_split", {})
    out += [
        f"- Loaded {_n(captures.get('usable'))} usable captures (of {_n(captures.get('total'))}) and "
        f"{_n(imports.get('total'))} imported rows.",
        f"- Input extraction: {_n(extract.get('records'))} records; {_n(extract.get('input_unparsed'))} inputs "
        f"unparsed, {_n(extract.get('missing_output'))} responses without an output.",
        f"- Merge by input hash: {_n(merge.get('examples'))} examples; {_n(merge.get('cross_split_merged'))} merged "
        "across predefined splits (kept in the most protected split).",
        f"- Output normalisation: {_n(norm.get('invalid_outputs'))} invalid outputs dropped; "
        f"{_n(norm.get('dropped_all_invalid'))} examples dropped with no valid output, "
        f"{_n(norm.get('dropped_no_majority'))} with no majority; {_n(norm.get('gold_invalid'))} invalid gold values"
        + (
            f" ({_n(norm.get('kept_test_all_invalid', 0) + norm.get('kept_test_no_majority', 0))} predefined test "
            "rows kept without a teacher value instead, so the test split never depends on the teacher's answers)"
            if norm.get("kept_test_all_invalid") or norm.get("kept_test_no_majority")
            else ""
        )
        + ".",
        f"- Exact repeats of an input (same input hash within one source and split) merged into one example: "
        f"{_n(dedupe.get('repeated_inputs', 0))} ("
        + ", ".join(f"{name} {_n(dedupe.get('splits', {}).get(name, {}).get('repeated_inputs', 0))}" for name in SPLITS)
        + ").",
        f"- Dedupe within splits (Jaccard >= {dedupe.get('threshold')} on word 3-shingles; texts under 3 words "
        f"{'by exact match' if dedupe.get('exact', True) else 'not deduplicated'}): "
        f"{_n(dedupe.get('removed_exact'))} exact (same text up to case and whitespace) and "
        f"{_n(dedupe.get('removed_near'))} near-duplicates removed; {_n(dedupe.get('conflict_clusters'))} clusters "
        f"with conflicting outputs dropped ({_n(dedupe.get('conflict_examples'))} examples); "
        f"{_n(dedupe.get('gold_conflicts', 0))} clusters with conflicting gold labels (the majority gold is kept, "
        "else the kept row's own).",
    ]
    removed = cross.get("removed", {})
    pairs = cross.get("pairs", {})

    def pair(name: str) -> str:
        found = pairs.get(name, {})
        return f"{_n(found.get('exact', 0))} exact, {_n(found.get('near', 0))} near"

    out.append(
        f"- Cross-split duplicates removed: train {_n(removed.get('train', 0))} (vs valid: {pair('train/valid')}; "
        f"vs test: {pair('train/test')}), valid {_n(removed.get('valid', 0))} (vs test: {pair('valid/test')}); "
        "test is never changed."
    )

    pii = stats.get("pii", {})
    out += ["", "### PII scrub", ""]
    if pii.get("enabled"):
        out.append(
            f"Regex with checksums; matches are replaced by typed placeholders. {_n(pii.get('examples_changed'))} "
            "examples changed."
        )
        out.append("")
        by_field = pii.get("by_field", {})
        rows = [
            [kind, *(by_field.get(f, {}).get(kind, 0) for f in ("input", "teacher", "gold")), count]
            for kind, count in pii.get("hits", {}).items()
        ]
        out += _table(["kind", "input", "teacher", "gold", "total"], rows)
        invalid, invalid_gold = pii.get("pii_schema_invalid", 0), pii.get("pii_schema_invalid_gold", 0)
        if invalid or invalid_gold:
            out += [
                "",
                f"A scrubbed placeholder can break a field's `format`, `pattern`, `enum` or length constraint: "
                f"{_n(invalid)} examples were dropped (their scrubbed teacher or curate-labelled answer fails the "
                f"task's JSON Schema), and {_n(invalid_gold)} gold values were set to `null` instead.",
            ]
            invalid_fields: Mapping[str, Mapping[str, Any]] = pii.get("pii_schema_invalid_fields", {})
            if invalid_fields:
                rows = [
                    [
                        name,
                        hit.get("teacher", 0),
                        hit.get("labelled", 0),
                        hit.get("gold", 0),
                        _counts(hit.get("kinds", {})),
                    ]
                    for name, hit in invalid_fields.items()
                ]
                out += ["", *_table(["field", "teacher", "labelled", "gold", "PII kinds found there"], rows)]
    else:
        out.append("Disabled for this task.")

    length = stats.get("length", {})
    out += [
        "",
        "## Length statistics",
        "",
        f"Student tokens of the full chat (system prompt, input, output) with the `{length.get('tokenizer')}` "
        f"chat template. Examples over {_n(length.get('max_seq_len'))} tokens are dropped, never truncated.",
        "",
    ]
    rows = [
        [name, s.get("n"), s.get("p50"), s.get("p95"), s.get("max"), s.get("dropped")]
        for name, s in ((name, length.get("splits", {}).get(name, {})) for name in SPLITS)
    ]
    out += _table(["split", "examples", "p50", "p95", "max", "dropped"], rows)

    out += ["", "## Teacher", ""]
    provider = teacher.get("provider")
    out += [
        f"- Model: `{teacher.get('model')}`",
        f"- Provider: {provider if provider else 'not pinned'}",
        f"- Teacher prompt SHA-256: `{teacher.get('prompt_sha256')}`",
    ]
    if captures.get("usable"):
        first, last = stats.get("captured_from"), stats.get("captured_to")
        span = f" between {first} and {last}" if first and last else ""
        out.append(f"- Captured teacher outputs: {_n(captures.get('usable'))}{span}")
        out.append(f"  - request models: {_counts(captures.get('by_model', {}))}")
        out.append(f"  - models named in the responses: {_counts(captures.get('by_upstream_model', {}))}")
        other_model, other_prompt = captures.get("other_model"), captures.get("other_prompt")
        if other_model:
            out.append(
                f"  - {_n(other_model)} captures asked another model than `{teacher.get('model')}`; their outputs "
                "are used as teacher labels like the others."
            )
        if other_prompt:
            prompts = captures.get("by_system_prompt", {})
            out.append(
                f"  - {_n(other_prompt)} captures used another system prompt than the teacher prompt (system prompt "
                f"SHA-256: {_counts({k[:12] if k != NO_VALUE else k: v for k, v in prompts.items()})})."
            )
    if labelling.get("requested"):
        recorded = labelling.get("recorded_cost_usd")
        original = (
            f", originally {_usd(recorded)} when the replayed or cached answers were produced" if recorded else ""
        )
        providers = labelling.get("providers") or {}
        served = f"; served by {', '.join(f'{k} {_n(v)}' for k, v in providers.items())}" if providers else ""
        kept = labelling.get("kept_invalid_test", 0)
        pii_invalid = labelling.get("pii_schema_invalid", 0)
        out.append(
            f"- Labelling by curate: {_n(labelling.get('requested'))} requests ({_n(labelling.get('live'))} live, "
            f"{_n(labelling.get('cached'))} cached, {_n(labelling.get('replayed'))} replayed from a recording), "
            f"cost {_usd(labelling.get('cost_usd'))} in this run{original}, labelled on "
            f"{labelling.get('date') or '-'}{served}; {_n(labelling.get('invalid'))} invalid answers "
            f"({_n(labelling.get('invalid', 0) - kept)} train/valid examples dropped, {_n(kept)} test examples kept "
            f"without a teacher value), {_n(labelling.get('truncated'))} truncated"
            + (
                f"; {_n(pii_invalid)} dropped, their scrubbed answer fails the task's JSON Schema"
                if pii_invalid
                else ""
            )
            + "."
        )
    else:
        out.append("- Labelling by curate: none needed (every example had a recorded teacher output); cost $0.0000.")

    leakage = stats.get("leakage", {})
    out += [
        "",
        "## Leakage check",
        "",
        f"An independent pass over the final splits found {_n(leakage.get('exact', 0))} exact and "
        f"{_n(leakage.get('near', 0))} near-duplicate pairs across train/valid/test "
        f"(Jaccard >= {leakage.get('threshold')}): {'ok' if leakage.get('ok') else 'FAILED'}.",
        "",
    ]
    return "\n".join(out)
