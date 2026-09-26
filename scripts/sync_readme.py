"""Rewrite the README blocks that hold numbers from the committed reports/*.json files.

A block is delimited by ``<!-- sync:NAME -->`` and ``<!-- /sync:NAME -->`` (or ``<!-- /sync -->`` for inline
blocks). Every block NAME must have a renderer below; the renderer reads only files under reports/.

    uv run python scripts/sync_readme.py          # rewrite README.md
    uv run python scripts/sync_readme.py --check  # exit 1 if README.md differs from what the reports say

Two environment variables help development and testing without touching the real files:

    SYNC_README_REPORTS=path/to/reports   # read report JSON from here instead of reports/
    SYNC_README_PATH=path/to/README.md    # rewrite/check this file instead of README.md
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
README = Path(os.environ.get("SYNC_README_PATH") or ROOT / "README.md")
REPORTS = Path(os.environ.get("SYNC_README_REPORTS") or ROOT / "reports")
BLOCK = re.compile(r"(<!-- sync:(?P<name>[a-z0-9-]+) -->)(?P<body>.*?)(<!-- /sync(?::(?P=name))? -->)", re.DOTALL)

Renderer = Callable[[], str]
RENDERERS: dict[str, Renderer] = {}


def renderer(name: str) -> Callable[[Renderer], Renderer]:
    def register(fn: Renderer) -> Renderer:
        RENDERERS[name] = fn
        return fn

    return register


def load(relative: str) -> Any:
    """Read a required report file under the reports directory; a clear error if it is missing."""
    path = REPORTS / relative
    if not path.is_file():
        raise SystemExit(f"sync_readme: missing {relative} under {REPORTS}; run scripts/reproduce.sh first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional(relative: str) -> Any | None:
    """Like :func:`load`, but ``None`` when the file does not exist (an optional report, not yet produced)."""
    path = REPORTS / relative
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        fn = RENDERERS.get(name)
        if fn is None:
            raise SystemExit(f"sync_readme: no renderer for block '{name}'")
        body = fn()
        inline = "\n" not in match.group("body") and "\n" not in body
        if inline:
            return f"{match.group(1)}{body}{match.group(4)}"
        return f"{match.group(1)}\n{body.strip()}\n{match.group(4)}"

    return BLOCK.sub(replace, text)


# == formatting helpers ================================================================================
# Same conventions as taskdistill.report.render (percentages to one decimal, 95% intervals as [lo, hi],
# differences in points, AUROC with three decimals, dollars with three significant digits, latency in ms),
# reimplemented here with no import from the package so this script stays dependency-free.

DASH = "—"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def pct(value: Any, decimals: int = 1) -> str:
    number = _num(value)
    return DASH if number is None else f"{number * 100:.{decimals}f}%"


def pts(value: Any, decimals: int = 1) -> str:
    number = _num(value)
    return DASH if number is None else f"{number * 100:+.{decimals}f} pts"


def plain(value: Any, digits: int = 3) -> str:
    number = _num(value)
    return DASH if number is None else f"{number:.{digits}f}"


def interval(ci: Any, as_plain: bool = False, decimals: int = 1) -> str:
    """``[lo, hi]`` in percent, or with three decimals for an AUROC-like plain metric."""
    if not isinstance(ci, list | tuple) or len(ci) != 2:
        return ""
    lo, hi = _num(ci[0]), _num(ci[1])
    if lo is None or hi is None:
        return ""
    if as_plain:
        return f"[{lo:.3f}, {hi:.3f}]"
    return f"[{lo * 100:.{decimals}f}, {hi * 100:.{decimals}f}]"


def signed_interval(ci: Any, decimals: int = 1) -> str:
    if not isinstance(ci, list | tuple) or len(ci) != 2:
        return ""
    lo, hi = _num(ci[0]), _num(ci[1])
    if lo is None or hi is None:
        return ""
    return f"[{lo * 100:+.{decimals}f}, {hi * 100:+.{decimals}f}]"


def ms(value: Any) -> str:
    number = _num(value)
    if number is None:
        return DASH
    return f"{number:,.0f}" if abs(number) >= 100 else f"{number:.1f}"


def usd(value: Any) -> str:
    """Dollars with three significant digits and no exponent (``$0.0000783``, ``$0.412``, ``$12.35``)."""
    number = _num(value)
    if number is None:
        return DASH
    if number == 0:
        return "$0"
    if abs(number) >= 1:
        return f"${number:,.2f}"
    decimals = max(2, -math.floor(math.log10(abs(number))) + 2)
    return f"${number:.{decimals}f}"


def integer(value: Any) -> str:
    number = _num(value)
    return DASH if number is None else f"{number:,.0f}"


def hardware_text(hardware: Any) -> str:
    """``Mac17,4, Apple M5, 24 GB, macOS 26.6.2`` from a hardware-info dict."""
    if not isinstance(hardware, Mapping):
        return DASH
    memory = _num(hardware.get("memory_gb"))
    parts = [hardware.get("model"), hardware.get("cpu"), f"{memory:g} GB" if memory else None, hardware.get("os")]
    return ", ".join(str(part) for part in parts if part) or DASH


def load_average_text(values: Any) -> str:
    if not isinstance(values, list | tuple) or not values:
        return DASH
    numbers = [_num(v) for v in values]
    return "/".join(f"{n:.2f}" if n is not None else DASH for n in numbers)


def fmt_seconds(value: Any) -> str:
    """``3.1 s`` under a minute, ``3.5 min`` at or above it."""
    number = _num(value)
    if number is None:
        return DASH
    return f"{number:.1f} s" if number < 60 else f"{number / 60:.1f} min"


def md_table(header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |" for row in rows]
    return lines


# == quality-row helpers ================================================================================

COLUMN_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "agreement": "Agreement",
    "json_validity": "JSON validity",
    "field_micro_f1": "Field micro-F1",
    "field_exact_match": "Field EM",
    "doc_exact_match": "Doc EM",
    "ece": "ECE",
    "auroc": "AUROC",
}


def col_label(metric: str) -> str:
    return COLUMN_LABELS.get(metric, metric.replace("_", " ").title())


def find_row(
    rows: Sequence[Mapping[str, Any]], system: str, labels: Any = "__any__", base_hint: str | None = None
) -> Mapping[str, Any] | None:
    """The first quality row matching ``system`` (and, when given, ``labels`` and a substring of ``base_model``)."""
    for row in rows:
        if row.get("system") != system:
            continue
        if labels != "__any__" and row.get("labels") != labels:
            continue
        if base_hint is not None and base_hint not in (row.get("base_model") or ""):
            continue
        return row
    return None


def metric_cell(row: Mapping[str, Any], metric: str) -> str:
    metrics = row.get("metrics") or {}
    if metric not in metrics:
        return DASH
    value = metrics.get(metric)
    if value is None:
        return "reference" if row.get("system") == "teacher" and metric == "agreement" else DASH
    text = plain(value) if metric == "auroc" else pct(value)
    std = _num((row.get("std") or {}).get(metric))
    if (row.get("n_seeds") or 0) > 1 and std is not None:
        text += f" ± {std:.3f}" if metric == "auroc" else f" ± {std * 100:.1f}"
    ci = interval((row.get("ci") or {}).get(metric), as_plain=(metric == "auroc"))
    return f"{text} {ci}" if ci else text


def quality_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[str]:
    body = [[row.get("name"), integer(row.get("n"))] + [metric_cell(row, m) for m in columns] for row in rows]
    return md_table(["System", "n", *[col_label(m) for m in columns]], body)


def breakdown_tables(block: Mapping[str, Any], row_label: str) -> list[str]:
    """Per-group or per-trait scores (``{name: {"n", <system>: {<metric>: value}}}``): one table per system."""
    entries = [(name, entry) for name, entry in block.items() if isinstance(entry, Mapping)]
    systems: list[str] = []
    for _, entry in entries:
        systems += [key for key, value in entry.items() if isinstance(value, Mapping) and key not in systems]
    lines: list[str] = []
    for system in systems:
        metrics: list[str] = []
        for _, entry in entries:
            scores = entry.get(system)
            if isinstance(scores, Mapping):
                metrics += [m for m in scores if m not in metrics]
        rows = [
            [name, integer(entry.get("n"))] + [pct((entry.get(system) or {}).get(m)) for m in metrics]
            for name, entry in entries
        ]
        lines += [f"{system}:", "", *md_table([row_label, "n", *[col_label(m) for m in metrics]], rows), ""]
    return lines


def per_field_tables(block: Mapping[str, Any]) -> list[str]:
    """Per-field exact match (``{"fields", "vs_gold": {<system>: {<field>: value}}, "vs_teacher": ...}``)."""
    lines: list[str] = []
    fields = [str(f) for f in block.get("fields") or []]
    for reference in ("vs_gold", "vs_teacher"):
        by_system = block.get(reference)
        if not isinstance(by_system, Mapping) or not by_system:
            continue
        systems = list(by_system.keys())
        rows = [[f"`{field}`"] + [pct(by_system[system].get(field)) for system in systems] for field in fields]
        title = reference.removeprefix("vs_")
        lines += [f"Against the {title}:", "", *md_table(["Field", *systems], rows), ""]
    return lines


def operating_point_block(report: Mapping[str, Any]) -> list[str]:
    """Target, chosen threshold, escalation rate, cascade-minus-teacher differences, whether the target held."""
    op = report.get("operating_point") or {}
    if not op.get("available"):
        return ["No operating point yet: no test evaluation of the selected run.", ""]
    reference = op.get("reference") or "teacher"
    metric_label = col_label(str(op.get("metric") or ""))
    target = op.get("target")
    if _num(target) is not None:
        target_text = f"{metric_label} ≥ {pct(target)} against the {reference}"
    else:
        target_text = f"{metric_label} against the {reference}"
    threshold = op.get("threshold")
    threshold_text = "always escalate" if threshold is None else f"{_num(threshold):.4f}"
    rates = op.get("escalation_rate") or {}
    diffs = op.get("cascade_minus_teacher") or {}
    target_diff, gold_diff = diffs.get("target_metric") or {}, diffs.get("gold_metric") or {}
    met = op.get("target_met_on_test")

    def diff_text(entry: Mapping[str, Any]) -> str:
        if not entry:
            return DASH
        text = pts(entry.get("point"))
        paired, cluster = signed_interval(entry.get("ci")), signed_interval(entry.get("ci_cluster"))
        if paired:
            text += f" {paired}"
        if cluster:
            text += f", cluster {cluster}"
        return text

    rows = [
        ["Target", target_text],
        ["Chosen threshold (on validation)", threshold_text],
        ["Escalation rate, valid / test", f"{pct(rates.get('valid'))} / {pct(rates.get('test'))}"],
        [f"Cascade − teacher, {metric_label} against the {reference} (target)", diff_text(target_diff)],
        [f"Cascade − teacher, {col_label(str(gold_diff.get('metric') or 'gold metric'))} (gold)",
         diff_text(gold_diff)],
        ["Target held on test", DASH if met is None else ("yes" if met else "no")],
    ]  # fmt: skip
    return [*md_table(["", "Value"], rows), ""]


def teacher_latency_phrase(report: Mapping[str, Any]) -> str:
    """``about 594 ms at p50 and 1,242 ms at p95 (Banking77 labelling calls, 2026-09-26)``."""
    teacher = (report.get("cost_latency") or {}).get("teacher") or {}
    latency = teacher.get("latency_ms") or {}
    recorded = teacher.get("recorded_between") or []
    date = recorded[-1] if recorded else teacher.get("pricing_snapshot_date") or DASH
    p50, p95 = ms(latency.get("p50")), ms(latency.get("p95"))
    return f"about {p50} ms at p50 and {p95} ms at p95 (Banking77 labelling calls, {date})"


# == teacher-latency (inline) and quickstart-timing =====================================================


@renderer("teacher-latency")
def render_teacher_latency() -> str:
    return teacher_latency_phrase(load("banking77/report.json"))


@renderer("quickstart-timing")
def render_quickstart_timing() -> str:
    timing = load("demo_timing.json")
    runs = timing.get("runs") or []
    if not runs:
        return "Timing not measured yet."
    lines = [f"Measured on {hardware_text(timing.get('hardware'))}, {timing.get('date') or DASH}:", ""]
    rows = [
        [
            run.get("demo", DASH), run.get("profile", DASH), run.get("cache", DASH),
            fmt_seconds(run.get("install_s")), fmt_seconds(run.get("model_download_s")),
            fmt_seconds(run.get("demo_s")), fmt_seconds(run.get("total_s")),
        ]
        for run in runs
    ]  # fmt: skip
    lines += md_table(["Demo", "Profile", "Cache", "Install", "Model download", "Demo", "Total"], rows)
    lines.append("")
    totals = [_num(run.get("total_s")) for run in runs if _num(run.get("total_s")) is not None]
    cold = [run for run in runs if run.get("cache") == "cold"]
    warm = [run for run in runs if run.get("cache") == "warm"]
    sentences = []
    if cold:
        worst = max(cold, key=lambda run: _num(run.get("total_s")) or 0)
        sentences.append(
            f"A cold `taskdistill demo {worst.get('demo')}` (empty `uv` cache, empty Hugging Face cache) took "
            f"{fmt_seconds(worst.get('total_s'))} in total."
        )
    if warm:
        best = min(warm, key=lambda run: _num(run.get("total_s")) or 0)
        warm_time = fmt_seconds(best.get("total_s"))
        sentences.append(f"With a warm cache, `taskdistill demo {best.get('demo')}` took {warm_time}.")
    if totals and max(totals) < 300:
        sentences.append("The quickstart above finishes in under five minutes, even on a cold cache.")
    elif totals:
        sentences.append(f"The slowest of these runs took {fmt_seconds(max(totals))}.")
    lines.append(" ".join(sentences))
    return "\n".join(lines)


# == results: per-section builders ======================================================================


def _provenance_paragraph(banking: Mapping[str, Any], invoices: Mapping[str, Any]) -> list[str]:
    hardware = hardware_text(banking.get("hardware"))
    report_date = banking.get("date") or DASH
    teacher_recorded = ((banking.get("cost_latency") or {}).get("teacher") or {}).get("recorded_between") or []
    recorded_date = teacher_recorded[-1] if teacher_recorded else DASH
    access_bits = []
    for name, report in (("Banking77", banking), ("Invoices", invoices)):
        count = _num((report.get("test_access") or {}).get("count"))
        if count is not None:
            access_bits.append(f"{name} {integer(count)} time{'' if count == 1 else 's'}")
    access_text = "; ".join(access_bits) if access_bits else DASH
    para = (
        f"Numbers below come from `scripts/reproduce.sh` (full profile, recorded teacher outputs) on {report_date} "
        f"on {hardware}; teacher outputs were recorded on {recorded_date}. The test split was scored {access_text} "
        "in this workspace, and nothing was chosen with it: every choice (base model, seed, threshold, isotonic "
        "calibration) was made on the validation split alone. Figures marked “recorded” below (the "
        "teacher and cascade rows) replay the teacher outputs captured then, not a live call. The live bench and "
        "latency numbers are dated measurements and are not expected to reproduce exactly on different hardware or "
        "under different load. Unless noted otherwise, every score has a 95% paired bootstrap interval in brackets "
        "(1,000 resamples)."
    )
    return [para, ""]


def _seeded_row_table(row: Mapping[str, Any], columns: Sequence[str]) -> list[str]:
    rows = [
        [f"`{seed.get('run_id')}`", seed.get("seed")]
        + [plain((seed.get("metrics") or {}).get(m)) if m == "auroc" else pct((seed.get("metrics") or {}).get(m))
           for m in columns]
        for seed in row.get("per_seed") or []
    ]  # fmt: skip
    lines = [f"Per-seed scores (selected: `{row.get('selected_run')}`):", ""]
    lines += md_table(["Run", "Seed", *[col_label(m) for m in columns]], rows)
    lines.append("")
    return lines


def _quality_section_banking77(report: Mapping[str, Any]) -> list[str]:
    quality = report.get("quality") or {}
    all_rows = quality.get("rows") or []
    columns = list(quality.get("student_columns") or quality.get("columns") or [])
    order = [
        find_row(all_rows, "teacher"),
        find_row(all_rows, "zero-shot"),
        find_row(all_rows, "tfidf"),
        find_row(all_rows, "student", labels="teacher", base_hint="0.5B"),
        find_row(all_rows, "student", labels="teacher", base_hint="1.5B"),
        find_row(all_rows, "student", labels="gold"),
        find_row(all_rows, "cascade"),
    ]
    rows = [row for row in order if row is not None]
    lines = [
        "### Banking77 (77 intents)",
        "",
        f"Test split, n = {integer(quality.get('n'))}. Cells are the score and its 95% paired bootstrap interval; "
        "the 3-seed student row shows mean ± sample standard deviation, with the interval of the "
        "validation-selected seed. ECE and AUROC use raw (pre-isotonic) student confidence.",
        "",
        *quality_table(rows, columns),
        "",
    ]
    seeded = find_row(all_rows, "student", labels="teacher", base_hint="0.5B")
    if seeded and (seeded.get("n_seeds") or 0) > 1:
        lines += _seeded_row_table(seeded, columns)
    lines += operating_point_block(report)
    curve = (report.get("operating_point") or {}).get("curve_png")
    if curve:
        lines += [f"![Banking77 cascade quality against escalation rate](reports/banking77/{curve})", ""]
    return lines


def _quality_section_invoices(report: Mapping[str, Any]) -> list[str]:
    quality = report.get("quality") or {}
    all_rows = quality.get("rows") or []
    columns = list(quality.get("student_columns") or quality.get("columns") or [])
    order = [
        find_row(all_rows, "teacher"),
        find_row(all_rows, "zero-shot"),
        find_row(all_rows, "tfidf"),
        find_row(all_rows, "student", labels="teacher", base_hint="0.5B"),
        find_row(all_rows, "student", labels="teacher", base_hint="1.5B"),
        find_row(all_rows, "student", labels="gold"),
        find_row(all_rows, "cascade"),
    ]
    rows = [row for row in order if row is not None]
    n_groups = quality.get("n_groups")
    layouts = (
        f" The test set is {integer(n_groups)} layouts never seen in training, so the effective sample is "
        f"{integer(n_groups)} layouts; the template-cluster bootstrap below says how wide that makes the "
        "uncertainty."
        if _num(n_groups)
        else ""
    )
    lines = [
        "### Invoices (8-field JSON extraction)",
        "",
        f"Test split, n = {integer(quality.get('n'))}.{layouts} Cells are the score and its 95% paired bootstrap "
        "interval (document-level).",
        "",
        *quality_table(rows, columns),
        "",
    ]
    if quality.get("cluster_bootstrap") and _num(n_groups):
        cluster_rows = [
            [row.get("name")] + [interval((row.get("ci_cluster") or {}).get(m)) or DASH for m in columns]
            for row in rows
            if row.get("ci_cluster")
        ]
        lines += [
            f"Template-cluster bootstrap over {integer(n_groups)} groups (one per layout): 95% intervals.",
            "",
            *md_table(["System", *[col_label(m) for m in columns]], cluster_rows),
            "",
        ]
    per_group = quality.get("per_group")
    if isinstance(per_group, Mapping) and per_group:
        lines += ["Per-template scores:", "", *breakdown_tables(per_group, "Template")]
    per_field = quality.get("per_field")
    if isinstance(per_field, Mapping) and per_field:
        lines += ["<details>", "<summary>Per-field exact match</summary>", "", *per_field_tables(per_field),
                  "</details>", ""]  # fmt: skip
    per_trait = quality.get("per_trait")
    if isinstance(per_trait, Mapping) and per_trait:
        lines += ["<details>", "<summary>Per-trait breakdown</summary>", "", *breakdown_tables(per_trait, "Trait"),
                  "</details>", ""]  # fmt: skip
    lines += operating_point_block(report)
    curve = (report.get("operating_point") or {}).get("curve_png")
    if curve:
        lines += [f"![Invoices cascade quality against escalation rate](reports/invoices/{curve})", ""]
    return lines


def _cost_latency_task_table(report: Mapping[str, Any]) -> list[str]:
    cost = report.get("cost_latency") or {}
    teacher, student, cascade = cost.get("teacher") or {}, cost.get("student") or {}, cost.get("cascade") or {}
    teacher_source = (
        f"recorded live labelling calls at concurrency {teacher.get('concurrency', DASH)} "
        "(cache hits and retries excluded)"
    )
    student_source = "bench against `serve --threshold 0`" if student.get("source") == "bench" \
        else "eval in-process latency (flagged: not end-to-end through `serve`)"  # fmt: skip
    cascade_source = f"composed per request, {pct(cascade.get('escalation_rate'))} escalated"
    rows = [
        ["Teacher only", usd(teacher.get("usd_per_1k")), usd(teacher.get("list_price_usd_per_1k")),
         ms((teacher.get("latency_ms") or {}).get("p50")), ms((teacher.get("latency_ms") or {}).get("p95")),
         teacher_source],
        ["Student only", usd(student.get("usd_per_1k")), DASH,
         ms((student.get("latency_ms") or {}).get("p50")), ms((student.get("latency_ms") or {}).get("p95")),
         student_source],
        ["Cascade", usd(cascade.get("usd_per_1k")), usd(cascade.get("list_price_usd_per_1k")),
         ms((cascade.get("latency_ms") or {}).get("p50")), ms((cascade.get("latency_ms") or {}).get("p95")),
         cascade_source],
    ]  # fmt: skip
    return md_table(["System", "$/1k recorded", "$/1k list price", "p50 ms", "p95 ms", "Source"], rows)


def _break_even_sentence(report: Mapping[str, Any]) -> str | None:
    be = report.get("break_even") or {}
    if be.get("volume") is None:
        return None
    list_price = be.get("list_price") or {}
    text = f"Break-even at {integer(be['volume'])} requests at the recorded teacher cost"
    if list_price.get("volume") is not None:
        text += f", {integer(list_price['volume'])} requests at the list price without prompt caching"
    return text + "."


def _cost_and_latency_section(banking: Mapping[str, Any], invoices: Mapping[str, Any]) -> list[str]:
    lines = ["### Cost and latency", ""]
    assumptions = (banking.get("assumptions") or {}).get("text") or DASH
    lines += [f"Energy assumptions: {assumptions}.", ""]
    for label, report in (("Banking77", banking), ("Invoices", invoices)):
        lines += [f"**{label}**", "", *_cost_latency_task_table(report), ""]
        sentence = _break_even_sentence(report)
        if sentence:
            lines += [sentence, ""]
    lines += [
        f"The Why section above states the teacher answered in {teacher_latency_phrase(banking)}; that is this "
        "same recorded Banking77 teacher latency.",
        "",
    ]
    return lines


def _live_bench_section(banking: Mapping[str, Any], invoices: Mapping[str, Any]) -> list[str]:
    lines = ["### Live bench cross-check", ""]
    any_measured = False
    for label, report in (("Banking77", banking), ("Invoices", invoices)):
        bench = report.get("live_bench") or {}
        student_only, cascade = bench.get("student_only"), bench.get("cascade")
        if not student_only and not cascade:
            lines += [f"**{label}**: not measured yet.", ""]
            continue
        any_measured = True
        lines += [f"**{label}**", ""]
        rows = []
        for mode_label, run in (("Student only", student_only), ("Cascade", cascade)):
            if not run:
                continue
            latency = run.get("latency_ms") or {}
            rows.append([
                mode_label, run.get("date") or DASH, integer(run.get("n")),
                ms(latency.get("p50")), ms(latency.get("p95")), pct(run.get("escalation_rate")),
                usd(run.get("spend_usd")), load_average_text(run.get("load_average")),
            ])  # fmt: skip
        lines += md_table(["Mode", "Date", "n", "p50 ms", "p95 ms", "Escalated", "Spend", "Load average"], rows)
        lines.append("")
        check = (bench.get("cross_check") or {}).get("cascade")
        if check:
            lines += [
                f"Composed (from the test split) vs measured: p50 {ms(check.get('composed_p50'))} vs "
                f"{ms(check.get('measured_p50'))} ms, p95 {ms(check.get('composed_p95'))} vs "
                f"{ms(check.get('measured_p95'))} ms.",
                "",
            ]
    if not any_measured:
        lines += [
            "Composed cost and latency above come from the test split, not a live run through `taskdistill "
            "serve`; the live bench will replace this with a measured cross-check (dated, not expected to "
            "reproduce exactly).",
            "",
        ]
    return lines


def _task_label(task: str) -> str:
    return {"banking77": "Banking77", "invoices": "Invoices"}.get(task, task)


def _training_section(training: Mapping[str, Any] | None) -> list[str]:
    lines = ["### Training on the M5", ""]
    if not training or not training.get("table"):
        lines += ["Not available yet: `reports/training.json` (run `scripts/collect_reports.py`).", ""]
        return lines
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in training["table"]:
        by_task.setdefault(str(row.get("task")), []).append(row)
    loads: list[float] = []
    for task in sorted(by_task):
        rows = []
        for r in by_task[task]:
            rows.append([
                f"`{r.get('run_id')}`", r.get("base_model") or DASH, integer(r.get("n_train")),
                integer(r.get("curate_dropped_by_length")), integer(r.get("iterations")), plain(r.get("epochs"), 2),
                plain((_num(r.get("wall_seconds")) or 0) / 60, 1), plain(r.get("peak_memory_gb"), 2),
                integer(r.get("tokens_per_second")), plain(r.get("adapter_size_mb"), 1),
            ])  # fmt: skip
            loads += [n for n in (_num(v) for v in r.get("load_average") or []) if n is not None]
        header = [
            "Run", "Base", "Examples", "Dropped (length)", "Iterations", "Epochs", "Wall min", "Peak GB",
            "Tokens/s", "Adapter MB",
        ]  # fmt: skip
        lines += [f"**{_task_label(task)}**", "", *md_table(header, rows), ""]
    if loads:
        lines += [
            f"Load average recorded alongside these runs ranged up to {max(loads):.2f}. The MacBook Air is "
            "fanless and can throttle under sustained load; that is why the load average is recorded next to "
            "every timing rather than assumed away.",
            "",
        ]
    return lines


def _confidence_comparison_rows(report: Mapping[str, Any], task_label: str) -> list[list[Any]]:
    comparison = (report.get("quality") or {}).get("confidence_comparison") or {}
    rows: list[list[Any]] = []
    for split in ("valid", "test"):
        entry = comparison.get(split) or {}
        chosen = entry.get("chosen")
        for name in ("primary", "alternative"):
            scores = entry.get(name)
            if not isinstance(scores, Mapping):
                continue
            label = f"{name} (chosen)" if chosen == name else name
            rows.append([
                task_label, split, label, integer(scores.get("n")),
                plain(scores.get("auroc_vs_teacher")), plain(scores.get("auroc_vs_gold")),
                pct(scores.get("ece_vs_teacher")), pct(scores.get("ece_vs_gold")),
            ])  # fmt: skip
    return rows


def _calibration_section(banking: Mapping[str, Any], invoices: Mapping[str, Any]) -> list[str]:
    lines = ["### Calibration", ""]
    for png in (banking.get("operating_point") or {}).get("reliability_png") or []:
        lines += [f"![Banking77 reliability diagram, raw vs isotonic](reports/banking77/{png})", ""]
    rows = _confidence_comparison_rows(banking, "Banking77") + _confidence_comparison_rows(invoices, "Invoices")
    if rows:
        lines += [
            "Two confidence definitions were compared on validation, before either was used at test time: the "
            "primary is the trie-constrained greedy label's own renormalised token-probability product; the "
            "alternative is free greedy generation scored by the mean per-token log-probability. AUROC is against "
            "the teacher and the gold label; ECE against each.",
            "",
            *md_table(
                ["Task", "Split", "Confidence", "n", "AUROC/teacher", "AUROC/gold", "ECE/teacher", "ECE/gold"], rows
            ),
            "",
        ]
    for label, report in (("Banking77", banking), ("Invoices", invoices)):
        row = find_row((report.get("quality") or {}).get("rows") or [], "student", labels="teacher")
        calibration = (row or {}).get("calibration") or {}
        if calibration.get("ece") is not None and calibration.get("isotonic_ece") is not None:
            lines += [
                f"{label} student, ECE on test: {pct(calibration.get('ece'))} raw vs "
                f"{pct(calibration.get('isotonic_ece'))} after isotonic calibration.",
                "",
            ]
    return lines


def _decision_sentence(decision: Mapping[str, Any]) -> str:
    gain, ratio = _num(decision.get("gain")), _num(decision.get("latency_ratio"))
    min_gain, max_ratio = _num(decision.get("min_gain")), _num(decision.get("max_latency_ratio"))
    if gain is None or ratio is None:
        return ""
    gain_word = "meets" if min_gain is not None and gain >= min_gain else "is below"
    ratio_word = "under" if max_ratio is not None and ratio < max_ratio else "at or above"
    verdict = "the smaller base was kept" if decision.get("choice") == "small" else "the larger base was selected"
    # Two decimals here (rather than the usual one) so a near-threshold gain such as 0.97 vs a 1.00 minimum
    # cannot round to the same digits as the threshold it is being compared against.
    return (
        f"Rule: use the larger base only if it gains at least {pct(min_gain, 2)} on the validation metric and its "
        f"p95 latency stays under {plain(max_ratio, 1)}x the smaller model's. Here the gain {gain_word} the "
        f"{pct(min_gain, 2)} minimum ({pts(gain, 2)}) and the p95 ratio ({plain(ratio, 2)}x) is {ratio_word} the "
        f"{plain(max_ratio, 1)}x limit, so {verdict}."
    )


def _base_model_section(banking: Mapping[str, Any], invoices: Mapping[str, Any]) -> list[str]:
    lines = ["### Choosing the base model", ""]
    for label, report in (("Banking77", banking), ("Invoices", invoices)):
        selected = report.get("selected_run") or {}
        rule = selected.get("rule") or {}
        candidates = selected.get("candidates") or {}
        by_base: dict[str, list[Mapping[str, Any]]] = {}
        for run_id, candidate in candidates.items():
            by_base.setdefault(str(candidate.get("base_model")), []).append({**candidate, "run_id": run_id})
        lines += [f"**{label}**", ""]
        if len(by_base) < 2:
            base = next(iter(by_base), DASH)
            text = selected.get("reason") or f"Only one base model was trained ({base}); no size comparison made"
            lines += [f"{text}.", ""]
            continue
        rows = []
        for base, candidate_list in sorted(by_base.items()):
            best = max(candidate_list, key=lambda c: _num(c.get("valid_metric")) or float("-inf"))
            rows.append([base, pct(best.get("valid_metric")), ms(best.get("p95_ms"))])
        lines += [*md_table(["Base model", "Validation metric", "p95 ms"], rows), ""]
        decisions = rule.get("decisions") or []
        if decisions:
            sentence = _decision_sentence(decisions[0])
            if sentence:
                lines += [sentence, ""]
        elif selected.get("reason"):
            lines += [f"{selected['reason']}.", ""]
    return lines


def _lr_ablation_bullet(ablation: Mapping[str, Any] | None) -> str | None:
    if not ablation:
        return None
    summary = ablation.get("summary") or {}
    constant, warm = summary.get("constant") or {}, summary.get("warmup_cosine") or {}
    if not constant or not warm:
        return None
    runs = [r for r in ablation.get("runs") or [] if r.get("lr_schedule") == "constant"]
    losses = [(r.get("seed"), _num(r.get("best_val_loss"))) for r in runs]
    losses = [(seed, loss) for seed, loss in losses if loss is not None]
    diverged_seed = None
    if len(losses) >= 3:
        ordered = sorted(losses, key=lambda item: item[1])
        rest = [loss for _, loss in ordered[:-1]]
        if ordered[-1][1] > 2 * max(rest or [0.0]):
            diverged_seed = ordered[-1][0]
    divergence = f" one seed (seed {diverged_seed}) diverged to a much higher training loss than the other two;" \
        if diverged_seed is not None else ""  # fmt: skip
    return (
        f"Learning-rate schedule ({ablation.get('profile', 'quick')} profile, {integer(len(runs))}-seed Banking77 "
        f"validation): a constant learning rate reached {pct(constant.get('valid_agreement_mean'))} ± "
        f"{pct(constant.get('valid_agreement_std'))} mean agreement with the teacher, against "
        f"{pct(warm.get('valid_agreement_mean'))} ± {pct(warm.get('valid_agreement_std'))} for linear "
        f"warm-up + cosine decay;{divergence} the warm-up schedule is the default."
    )


def _size_ceiling_bullet(banking: Mapping[str, Any]) -> str | None:
    decisions = ((banking.get("selected_run") or {}).get("rule") or {}).get("decisions") or []
    if not decisions:
        return None
    decision = decisions[0]
    gain, ratio = _num(decision.get("gain")), _num(decision.get("latency_ratio"))
    min_gain, max_ratio = _num(decision.get("min_gain")), _num(decision.get("max_latency_ratio"))
    if gain is None or ratio is None:
        return None
    verdict = "below" if min_gain is not None and gain < min_gain else "at or above"
    kept = "kept the 0.5B base" if decision.get("choice") == "small" else "chose the 1.5B base"
    return (
        f"The bigger Banking77 student (1.5B vs 0.5B, teacher labels) gained {pts(gain, 2)} on validation "
        f"agreement, {verdict} the {pct(min_gain, 2)} minimum the base-model rule requires, for "
        f"{plain(ratio, 2)}x the p95 latency (limit {plain(max_ratio, 1)}x); the rule {kept}."
    )


def _label_ceiling_bullet(banking: Mapping[str, Any]) -> str | None:
    rows = (banking.get("quality") or {}).get("rows") or []
    teacher_row = find_row(rows, "teacher")
    gold_row = find_row(rows, "student", labels="gold")
    student_row = find_row(rows, "student", labels="teacher", base_hint="0.5B")
    if not (teacher_row and gold_row and student_row):
        return None
    teacher_acc = (teacher_row.get("metrics") or {}).get("accuracy")
    gold_acc = (gold_row.get("metrics") or {}).get("accuracy")
    student_acc = (student_row.get("selected_metrics") or student_row.get("metrics") or {}).get("accuracy")
    return (
        f"A Banking77 student trained on the gold labels reached {pct(gold_acc)} accuracy against gold, above "
        f"both the same student trained on teacher labels ({pct(student_acc)}) and the teacher itself "
        f"({pct(teacher_acc)}); part of that gap is Banking77's own label noise (Ying and Thomas, 2022, flag "
        "about 14% of the training utterances as potential label errors), which caps how high any model's "
        "accuracy against gold can go."
    )


def _prompt_variant_bullet(base: Mapping[str, Any] | None, variants: Mapping[str, Any | None]) -> str | None:
    if not base or not base.get("chosen"):
        return None
    chosen_id = (base.get("chosen") or {}).get("candidate")
    base_candidate = next((c for c in base.get("candidates") or [] if c.get("candidate") == chosen_id), None)
    if not base_candidate:
        return None
    base_acc = _num((base_candidate.get("metrics") or {}).get("accuracy"))
    base_tokens = _num((base_candidate.get("tokens") or {}).get("prompt"))
    base_n = _num(base_candidate.get("n"))
    base_cost = _num(base_candidate.get("cost_per_1k_usd"))
    bits = []
    for name, variant in variants.items():
        candidates = (variant or {}).get("candidates") or []
        if not candidates:
            continue
        candidate = candidates[0]
        acc = _num((candidate.get("metrics") or {}).get("accuracy"))
        tokens = _num((candidate.get("tokens") or {}).get("prompt"))
        n = _num(candidate.get("n"))
        cost = _num(candidate.get("cost_per_1k_usd"))
        if not all(v is not None and v for v in (base_acc, acc, base_tokens, tokens, base_n, n, base_cost, cost)):
            continue
        gain = (acc - base_acc) * 100
        token_ratio = (tokens / n) / (base_tokens / base_n)
        cost_ratio = cost / base_cost
        bits.append(
            f"{name} ({gain:+.1f} pts accuracy at {token_ratio:.1f}x the prompt tokens, "
            f"{cost_ratio:.1f}x the cost per 1k)"
        )
    if not bits:
        return None
    return (
        "Teacher prompt variants (same 200 Banking77 validation queries, same teacher model): "
        + "; ".join(bits)
        + ". The spec prompt was kept for the recorded run."
    )


def _confidence_bullet(banking: Mapping[str, Any], invoices: Mapping[str, Any]) -> str | None:
    bits = []
    for label, report in (("Banking77", banking), ("Invoices", invoices)):
        comparison = ((report.get("quality") or {}).get("confidence_comparison") or {}).get("test") or {}
        if comparison.get("chosen") != "primary" or not comparison.get("alternative"):
            continue
        primary, alternative = comparison.get("primary") or {}, comparison["alternative"]
        bits.append(
            f"{label}: AUROC {plain(primary.get('auroc_vs_teacher'))} vs {plain(alternative.get('auroc_vs_teacher'))}"
            f", ECE {pct(primary.get('ece_vs_teacher'))} vs {pct(alternative.get('ece_vs_teacher'))}"
        )
    if not bits:
        return None
    return (
        "The alternative confidence definition (free greedy generation, mean per-token log-probability) did not "
        "improve on the primary (trie-constrained token-probability product) on test — "
        + "; ".join(bits)
        + " — so the primary definition was kept."
    )


def _spend_and_downloads_section(spend: Mapping[str, Any] | None, downloads: Mapping[str, Any] | None) -> list[str]:
    lines = ["### Spend and downloads", ""]
    if spend:
        rows = []
        for task, phases in sorted((spend.get("by_task_phase") or {}).items()):
            labelling = sum(_num(phases.get(p)) or 0 for p in ("demo-capture", "curate-label", "record-fill"))
            bakeoff = _num(phases.get("bakeoff")) or 0
            live_bench = usd(phases.get("serve")) if "serve" in phases else DASH
            rows.append([_task_label(task), usd(labelling), usd(bakeoff), live_bench])
        lines += [*md_table(["Task", "Teacher labelling", "Bake-off", "Live bench"], rows), ""]
        lines += [
            f"Total spend for the whole build: {usd(spend.get('total'))} of a {usd(spend.get('cap'))} global cap "
            f"({integer(spend.get('calls'))} teacher calls).",
            "",
        ]
    else:
        lines += ["Not available yet: no ledger export (`reports/spend.json`).", ""]
    if downloads:
        total_gb, budget_gb = _num(downloads.get("total_gb")), _num(downloads.get("budget_gb"))
        within = "within" if total_gb is not None and budget_gb is not None and total_gb <= budget_gb else "over"
        parts = [
            f"{d.get('purpose') or d.get('repo')}: {plain(d.get('gb'), 2)} GB" for d in downloads.get("downloads") or []
        ]
        lines += [
            f"Model downloads: {plain(total_gb, 2)} GB, {within} the {plain(budget_gb, 1)} GB budget"
            + (f" ({'; '.join(parts)})" if parts else "")
            + ".",
            "",
        ]
    else:
        lines += ["Not available yet: no download log (`reports/downloads.json`).", ""]
    return lines


@renderer("results")
def render_results() -> str:
    banking = load("banking77/report.json")
    invoices = load("invoices/report.json")
    training = load_optional("training.json")
    spend = load_optional("spend.json")
    downloads = load_optional("downloads.json")
    ablation = load_optional("ablations/lr_schedule.json")
    bakeoff_banking = load_optional("bakeoff/banking77.json")
    bakeoff_variants = {
        "snake-case labels": load_optional("bakeoff/banking77.labels-snake-case.json"),
        "labels with examples": load_optional("bakeoff/banking77.labels-with-examples.json"),
    }

    lines: list[str] = ["## Results", ""]
    lines += _provenance_paragraph(banking, invoices)
    lines += _quality_section_banking77(banking)
    lines += _quality_section_invoices(invoices)
    lines += _cost_and_latency_section(banking, invoices)
    lines += _live_bench_section(banking, invoices)
    lines += _training_section(training)
    lines += _calibration_section(banking, invoices)
    lines += _base_model_section(banking, invoices)

    lines += ["### What didn't work", ""]
    bullets = [
        _lr_ablation_bullet(ablation),
        _size_ceiling_bullet(banking),
        _label_ceiling_bullet(banking),
        _prompt_variant_bullet(bakeoff_banking, bakeoff_variants),
        _confidence_bullet(banking, invoices),
    ]
    lines += [text for bullet in bullets if bullet for text in (f"- {bullet}", "")]

    lines += _spend_and_downloads_section(spend, downloads)
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    args = parser.parse_args(argv)
    current = README.read_text(encoding="utf-8")
    updated = render(current)
    if args.check:
        if updated != current:
            diff = difflib.unified_diff(
                current.splitlines(), updated.splitlines(), "README.md", "README.md (from reports)", lineterm=""
            )
            print("\n".join(list(diff)[:200]))
            print("sync_readme: README.md differs from reports/*.json; run scripts/sync_readme.py", file=sys.stderr)
            return 1
        print("sync_readme: README.md matches reports/*.json")
        return 0
    if updated != current:
        README.write_text(updated, encoding="utf-8")
        print("sync_readme: README.md updated")
    else:
        print("sync_readme: README.md already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
