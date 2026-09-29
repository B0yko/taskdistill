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


def points(value: Any, decimals: int = 1) -> str:
    """Like :func:`pts`, but spelled out for prose about a gain: ``+0.97 points`` -- the base-model rule is a
    difference in points, never a percentage of anything."""
    number = _num(value)
    if number is None:
        return DASH
    return f"{number * 100:+.{decimals}f} points"


def min_gain_points(value: Any) -> str:
    """The base-model rule's minimum gain, spelled out without a forced decimal count: ``1 point`` for exactly
    one (not ``1.00 points``), ``1.5 points`` when it is fractional -- singular only for exactly one point."""
    number = _num(value)
    if number is None:
        return DASH
    text = f"{number * 100:g}"
    return f"{text} point" if text == "1" else f"{text} points"


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


#: Product names for the model identifiers of the machines behind the committed reports; an identifier not
#: listed here is shown as it is.
PRODUCT_NAMES = {"Mac17,4": "MacBook Air", "Mac16,9": "Mac Studio"}


def product_name(hardware: Any) -> str | None:
    model = hardware.get("model") if isinstance(hardware, Mapping) else None
    return PRODUCT_NAMES.get(str(model)) if model else None


def machine_text(hardware: Any) -> str:
    """``MacBook Air, Apple M5, 24 GB`` from a hardware-info dict (the identifier when the product is unknown)."""
    if not isinstance(hardware, Mapping):
        return DASH
    memory = _num(hardware.get("memory_gb"))
    parts = [product_name(hardware) or hardware.get("model"), hardware.get("cpu"), f"{memory:g} GB" if memory else None]
    return ", ".join(str(part) for part in parts if part) or DASH


def hardware_text(hardware: Any) -> str:
    """``MacBook Air, Apple M5, 24 GB (Mac17,4), macOS 26.6.2`` from a hardware-info dict."""
    if not isinstance(hardware, Mapping):
        return DASH
    text = machine_text(hardware)
    if product_name(hardware):
        text += f" ({hardware.get('model')})"
    return ", ".join(str(part) for part in (text, hardware.get("os")) if part and part != DASH) or DASH


def date_only(value: Any) -> str:
    """The ``YYYY-MM-DD`` prefix of an ISO date or datetime string."""
    text = str(value) if value else ""
    return text.split("T", 1)[0] if text else DASH


def times_text(count: Any) -> str:
    """``8 times``, or ``1 time`` for exactly one."""
    number = _num(count)
    if number is None:
        return DASH
    return f"{integer(number)} time" if number == 1 else f"{integer(number)} times"


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


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*B", re.IGNORECASE)


def base_size(base_model: Any) -> tuple[float, str]:
    """``(0.5, "0.5B")`` parsed out of a base-model id such as ``mlx-community/Qwen2.5-0.5B-Instruct-4bit``."""
    match = _SIZE_RE.search(str(base_model or ""))
    if not match:
        return (math.inf, str(base_model or DASH))
    return (float(match.group(1)), f"{match.group(1)}B")


def zero_shot_rows(rows: Sequence[Mapping[str, Any]], task_type: str) -> list[Mapping[str, Any]]:
    """Every zero-shot row, smallest base first, labelled ``base <size> zero-shot``.

    Only the first row names what the zero-shot prompt includes (the label list for classification, the JSON
    Schema for extraction); later rows are the same setup at a different base size, so the note is not repeated.
    """
    matches = sorted(
        (row for row in rows if row.get("system") == "zero-shot"), key=lambda row: base_size(row.get("base_model"))[0]
    )
    note = "labels in the prompt" if task_type == "classification" else "schema in the prompt"
    out = []
    for i, row in enumerate(matches):
        _, size = base_size(row.get("base_model"))
        label = f"base {size} zero-shot" + (f" ({note})" if i == 0 else "")
        out.append({**row, "name": label})
    return out


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
        # A row aggregating more than one seed shows mean +/- sample standard deviation only: its own "ci" is
        # one particular seed's bootstrap interval, not an interval of the mean, so it is never shown here --
        # see the per-seed table instead, which attaches it to that one seed.
        return text + (f" ± {std:.3f}" if metric == "auroc" else f" ± {std * 100:.1f}")
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
    lines = [*md_table(["", "Value"], rows), ""]
    outcome = _operating_point_outcome(op, metric_label, reference)
    if outcome:
        lines += [outcome, ""]
    return lines


def _operating_point_outcome(op: Mapping[str, Any], metric_label: str, reference: str) -> str | None:
    """A plain sentence on whether the validation-chosen threshold met the target on test, with the numbers that
    made it true (or not) -- the table above says "yes"/"no"; this says why, so the "no" case cannot read as if
    the target had held."""
    met = op.get("target_met_on_test")
    if met is None:
        return None
    quality = op.get("quality") or {}
    rates = op.get("escalation_rate") or {}
    target_text, valid_q, test_q = pct(op.get("target")), pct(quality.get("valid")), pct(quality.get("test"))
    valid_esc, test_esc = pct(rates.get("valid")), pct(rates.get("test"))
    if met:
        return (
            f"Target held on test: yes — {metric_label} against the {reference} was {test_q} on test (target "
            f"{target_text}) at a {test_esc} escalation rate, close to the {valid_q} on validation at {valid_esc}."
        )
    return (
        f"Target held on test: no — {metric_label} against the {reference} met the {target_text} target on "
        f"validation ({valid_q} at {valid_esc} escalation) but fell to {test_q} on test ({test_esc} escalation)."
    )


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

    def machine(run: Mapping[str, Any]) -> str:
        return machine_text(run.get("hardware") or timing.get("hardware") or {})

    def load_before(run: Mapping[str, Any]) -> str:
        state = run.get("state_before") or {}
        return load_average_text(state.get("load_average"))

    lines = [
        "Measured quick-profile demos in replay mode, each in a fresh workspace and directory "
        f"({timing.get('date') or DASH}; `reports/demo_timing.json`):",
        "",
    ]
    rows = [
        [
            machine(run), f"`{run.get('demo', DASH)}`", run.get("cache", DASH), fmt_seconds(run.get("install_s")),
            fmt_seconds(run.get("demo_s")), fmt_seconds(run.get("total_s")), load_before(run),
        ]
        for run in runs
    ]  # fmt: skip
    lines += md_table(["Machine", "Demo", "Cache", "Install", "Demo", "Total", "Load average before"], rows)
    lines.append("")
    sentences = []
    for run in [r for r in runs if r.get("cache") == "cold"]:
        size = _num(run.get("model_download_bytes"))
        downloaded = f", including the {size / 1e9:.2f} GB base-model download" if size else ""
        sentences.append(
            f"The cold run (empty `uv` cache and Hugging Face cache on the {machine(run)}) took "
            f"{fmt_seconds(run.get('total_s'))}: {fmt_seconds(run.get('install_s'))} to install and "
            f"{fmt_seconds(run.get('demo_s'))} for the demo{downloaded}."
        )
    totals = [_num(run.get("total_s")) for run in runs if _num(run.get("total_s")) is not None]
    if totals and max(totals) < 300:
        sentences.append(
            "Every measured run finished in under five minutes; slower Macs and slower connections will take longer."
        )
    elif totals:
        sentences.append(f"The slowest of these runs took {fmt_seconds(max(totals))}.")
    lines.append(" ".join(sentences))
    return "\n".join(lines)


@renderer("headline")
def render_headline() -> str:
    """Three headline numbers under the title, plus the target that did not hold, when one did not."""
    banking = load("banking77/report.json")
    invoices = load("invoices/report.json")
    studio = load_optional("banking77/report_mac_studio.json") or banking
    spend = load_optional("spend.json") or {}
    op = banking.get("operating_point") or {}
    latency = studio.get("cost_latency") or {}
    teacher_p50 = _num(((latency.get("teacher") or {}).get("latency_ms") or {}).get("p50"))
    student_p50 = _num(((latency.get("student") or {}).get("latency_ms") or {}).get("p50"))
    hardware = studio.get("hardware") or {}
    machine = " ".join(
        str(part) for part in (product_name(hardware), str(hardware.get("cpu") or "").replace("Apple ", "")) if part
    )
    cells = [
        (
            pct((op.get("quality") or {}).get("test")),
            "agreement with the teacher",
            f"Banking77 test set, {pct((op.get('escalation_rate') or {}).get('test'))} of requests escalated",
        )
    ]
    if teacher_p50 and student_p50:
        cells.append(
            (
                f"{teacher_p50 / student_p50:.0f}× faster",
                "student vs teacher API at p50",
                f"{ms(student_p50)} ms vs {ms(teacher_p50)} ms{', ' + machine if machine else ''}",
            )
        )
    if spend.get("total") is not None:
        cells.append(
            (f"${_num(spend['total']) or 0:.2f}", "total API spend",
             f"{integer(spend.get('calls'))} teacher calls for the whole build")
        )  # fmt: skip
    row = "".join(
        f'<td align="center" width="{100 // len(cells)}%"><h3>{big}</h3>{label}<br><sub>{detail}</sub></td>'
        for big, label, detail in cells
    )
    lines = ["<table>", f"<tr>{row}</tr>", "</table>"]
    missed = []
    unseen = " (layouts never seen in training)"
    for label, report, where in (("Banking77", banking, ""), ("invoices", invoices, unseen)):
        rop = report.get("operating_point") or {}
        if rop.get("target_met_on_test") is False:
            missed.append(
                f"on {label} the cascade missed its {pct(rop.get('target'))} target on the test set{where}, "
                f"reaching {pct((rop.get('quality') or {}).get('test'))}"
            )
    if missed:
        lines += [
            "",
            f'<sub>Not every target held: {"; ".join(missed)}. Details in <a href="#results">Results</a>.</sub>',
        ]
    return "\n".join(lines)


# == results: per-section builders ======================================================================


def _extra_zero_shot_phrase(task_label: str, report: Mapping[str, Any]) -> str | None:
    """``"the invoices table also shows the 1.5B zero-shot base"`` when a task's report has more than one
    zero-shot row (a size beyond the first, which the provenance paragraph already covers as the norm)."""
    rows = [row for row in (report.get("quality") or {}).get("rows") or [] if row.get("system") == "zero-shot"]
    if len(rows) <= 1:
        return None
    extra = sorted(rows, key=lambda row: base_size(row.get("base_model"))[0])[1:]
    sizes = " and ".join(base_size(row.get("base_model"))[1] for row in extra)
    plural = "s" if len(extra) > 1 else ""
    return f"the {task_label.lower()} table also shows the {sizes} zero-shot base{plural}"


def _machines_paragraph(banking: Mapping[str, Any], studio: Mapping[str, Any] | None) -> list[str]:
    """Which machine produced which numbers, stated once before any table uses the second one."""
    if not studio:
        return []
    trainer = machine_text(banking.get("hardware"))
    return [
        f"Two machines were used. Quality, calibration and training numbers come from the {trainer} "
        "that trained the students; it is fanless and shared with other work. Cost, latency, the live bench and the "
        f"cross-machine reproduction come from a {machine_text(studio.get('hardware'))}, used only for those runs. "
        "Each table names its machine.",
        "",
    ]


def _bottom_line(banking: Mapping[str, Any], invoices: Mapping[str, Any]) -> list[str]:
    """The two headline outcomes (did the cascade hold its target on test), before the tables that show them."""
    items = []
    for label, report in (("Banking77", banking), ("Invoices", invoices)):
        op = report.get("operating_point") or {}
        outcome = _operating_point_outcome(op, col_label(str(op.get("metric") or "")), op.get("reference") or "teacher")
        if outcome:
            items.append(f"- **{label}.** {outcome}")
    if not items:
        return []
    return ["**Bottom line.** Each cascade's threshold was chosen on validation; on test:", "", *items, ""]


METRIC_DEFINITIONS = (
    "Metrics: **agreement** is how often a system gives the teacher's answer (for invoices, field micro-F1 against "
    "the teacher's JSON); **accuracy** and **macro-F1** (F1 averaged over the 77 intents, each counted equally) are "
    "against the gold labels; **ECE** (expected calibration error, lower is better) is the average gap between the "
    "student's stated confidence and how often it is right; **AUROC** (0.5 = chance, 1.0 = perfect) is how well that "
    "confidence separates right answers from wrong ones. For invoices, **JSON validity** is the share of outputs "
    "that parse as a JSON object, **field micro-F1** and **field EM** (exact match) score the 8 fields against the "
    "gold, and **Doc EM** is the share of documents with all 8 fields right."
)


def _provenance_paragraph(banking: Mapping[str, Any], invoices: Mapping[str, Any]) -> list[str]:
    hardware = hardware_text(banking.get("hardware"))
    report_date = date_only(banking.get("date"))
    teacher_recorded = ((banking.get("cost_latency") or {}).get("teacher") or {}).get("recorded_between") or []
    recorded_date = teacher_recorded[-1] if teacher_recorded else DASH
    extra_bits = [
        phrase
        for label, report in (("Banking77", banking), ("Invoices", invoices))
        for phrase in [_extra_zero_shot_phrase(label, report)]
        if phrase
    ]
    extra_text = f", plus one extra zero-shot evaluation ({'; '.join(extra_bits)})" if extra_bits else ""
    banking_times = times_text((banking.get("test_access") or {}).get("count"))
    invoices_times = times_text((invoices.get("test_access") or {}).get("count"))
    para = (
        f"Numbers below come from `scripts/reproduce.sh` (full profile, recorded teacher outputs) on {report_date} "
        f"on a {hardware}{extra_text}; teacher outputs were recorded on {recorded_date}. The test splits were scored "
        f"{banking_times} (Banking77) and {invoices_times} (invoices), each time by an evaluation shown in these "
        "tables; no choice used them: every choice (base model, seed, threshold, isotonic calibration) was made on "
        "the validation split alone. Figures marked “recorded” below (the "
        "teacher and cascade rows) replay the teacher outputs captured then, not a live call. The live bench and "
        "latency numbers are dated measurements and are not expected to reproduce exactly on different hardware or "
        "under different load. Unless noted otherwise, every score has a 95% paired bootstrap interval in brackets "
        "(1,000 resamples)."
    )
    return [para, ""]


def _seeded_row_table(row: Mapping[str, Any], columns: Sequence[str]) -> list[str]:
    """Every seed's own score, with an interval where one is available for that seed.

    Only the validation-selected seed has one here: the row's own ``ci`` is the bootstrap interval of that one
    seed's evaluation (not of the seeds' mean), so it is attached to that seed's row and to no other.
    """
    selected_run, row_ci = row.get("selected_run"), row.get("ci") or {}
    rows = []
    for seed in row.get("per_seed") or []:
        run_id, selected = seed.get("run_id"), seed.get("run_id") == row.get("selected_run")
        seed_metrics, seed_ci = seed.get("metrics") or {}, seed.get("ci") or (row_ci if selected else {})
        cells = []
        for m in columns:
            value = plain(seed_metrics.get(m)) if m == "auroc" else pct(seed_metrics.get(m))
            ci_text = interval(seed_ci.get(m), as_plain=(m == "auroc"))
            cells.append(f"{value} {ci_text}" if ci_text else value)
        label = f"`{run_id}`" + (" (validation-selected)" if selected else "")
        rows.append([label, seed.get("seed"), *cells])
    lines = [f"Per-seed scores for `{row.get('name')}` (selected run: `{selected_run}`):", ""]
    lines += md_table(["Run", "Seed", *[col_label(m) for m in columns]], rows)
    lines.append("")
    return lines


def _seeded_rows_tables(all_rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[str]:
    """A per-seed table for every student (teacher-labels) row -- at either base size -- with more than one
    seed, so the aggregate row's mean +/- std (see :func:`metric_cell`) always has its detail nearby."""
    lines: list[str] = []
    for hint in ("0.5B", "1.5B"):
        row = find_row(all_rows, "student", labels="teacher", base_hint=hint)
        if row and (row.get("n_seeds") or 0) > 1:
            lines += _seeded_row_table(row, columns)
    return ["<details>", "<summary>Per-seed scores</summary>", "", *lines, "</details>", ""] if lines else []


def _ordered_quality_rows(all_rows: Sequence[Mapping[str, Any]], task_type: str) -> list[Mapping[str, Any]]:
    """Teacher, every zero-shot row (smallest base first), TF-IDF, each student base (teacher labels), the
    gold-label student, then the cascade -- whichever of these exist in this task's report."""
    order = [
        find_row(all_rows, "teacher"),
        *zero_shot_rows(all_rows, task_type),
        find_row(all_rows, "tfidf"),
        find_row(all_rows, "student", labels="teacher", base_hint="0.5B"),
        find_row(all_rows, "student", labels="teacher", base_hint="1.5B"),
        find_row(all_rows, "student", labels="gold"),
        find_row(all_rows, "cascade"),
    ]
    return [row for row in order if row is not None]


def _quality_section_banking77(report: Mapping[str, Any]) -> list[str]:
    quality = report.get("quality") or {}
    all_rows = quality.get("rows") or []
    columns = list(quality.get("student_columns") or quality.get("columns") or [])
    rows = _ordered_quality_rows(all_rows, "classification")
    lines = [
        "### Banking77 (77 intents)",
        "",
        f"Test split, n = {integer(quality.get('n'))}. Cells are the score and its 95% paired bootstrap interval; "
        "a row aggregating more than one seed shows mean ± sample standard deviation only (see the per-seed "
        "table below for each seed's own score, and interval where the eval recorded one). ECE and AUROC use "
        "raw (pre-isotonic) student confidence.",
        "",
        *quality_table(rows, columns),
        "",
        *_seeded_rows_tables(all_rows, columns),
    ]
    lines += operating_point_block(report)
    curve = (report.get("operating_point") or {}).get("curve_png")
    if curve:
        lines += [f"![Banking77 cascade quality against escalation rate](reports/banking77/{curve})", ""]
    return lines


def _quality_section_invoices(report: Mapping[str, Any]) -> list[str]:
    quality = report.get("quality") or {}
    all_rows = quality.get("rows") or []
    columns = list(quality.get("student_columns") or quality.get("columns") or [])
    rows = _ordered_quality_rows(all_rows, "extraction")
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
        "interval (document-level). A row aggregating more than one seed shows mean ± sample standard "
        "deviation only (see the per-seed table below for each seed's own score, and interval where the eval "
        "recorded one).",
        "",
        *quality_table(rows, columns),
        "",
        *_seeded_rows_tables(all_rows, columns),
    ]
    if quality.get("cluster_bootstrap") and _num(n_groups):
        cluster_rows = [
            [row.get("name")]
            + [interval((row.get("ci_cluster") or {}).get(m), as_plain=(m == "auroc")) or DASH for m in columns]
            for row in rows
            if row.get("ci_cluster")
        ]
        lines += [
            "<details>",
            f"<summary>Template-cluster bootstrap over {integer(n_groups)} layouts</summary>",
            "",
            f"Template-cluster bootstrap over {integer(n_groups)} groups (one per layout): 95% intervals.",
            "",
            *md_table(["System", *[col_label(m) for m in columns]], cluster_rows),
            "",
            "</details>",
            "",
        ]
    per_group = quality.get("per_group")
    if isinstance(per_group, Mapping) and per_group:
        lines += ["<details>", "<summary>Per-template scores (student, teacher, cascade)</summary>", "",
                  *breakdown_tables(per_group, "Template"), "</details>", ""]  # fmt: skip
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


def _cost_and_latency_section(
    banking: Mapping[str, Any],
    invoices: Mapping[str, Any],
    banking_studio: Mapping[str, Any] | None = None,
    invoices_studio: Mapping[str, Any] | None = None,
) -> list[str]:
    lines = ["### Cost and latency", ""]
    assumptions = (banking.get("assumptions") or {}).get("text") or DASH
    lines += [f"Energy assumptions: {assumptions}.", ""]
    for label, report, studio in (
        ("Banking77", banking, banking_studio),
        ("Invoices", invoices, invoices_studio),
    ):
        lines += [f"**{label}**", ""]
        if studio:
            lines += [
                f"Measured on a {hardware_text(studio.get('hardware'))}, {date_only(studio.get('date'))}:",
                "",
                *_cost_latency_task_table(studio),
                "",
            ]
            mba_latency = ((report.get("cost_latency") or {}).get("student") or {}).get("latency_ms") or {}
            lines += [
                f"For comparison, the MacBook Air's student latency was {ms(mba_latency.get('p50'))} ms p50 / "
                f"{ms(mba_latency.get('p95'))} ms p95 (in-process eval, not through the server).",
                "",
            ]
            sentence = _break_even_sentence(studio)
        else:
            lines += [*_cost_latency_task_table(report), ""]
            sentence = _break_even_sentence(report)
        if sentence:
            lines += [sentence, ""]
    lines += [
        f"The Why section above states the teacher answered in {teacher_latency_phrase(banking)}; that is this "
        "same recorded Banking77 teacher latency.",
        "",
    ]
    return lines


def _live_bench_section(
    banking: Mapping[str, Any],
    invoices: Mapping[str, Any],
    banking_studio: Mapping[str, Any] | None = None,
    invoices_studio: Mapping[str, Any] | None = None,
) -> list[str]:
    lines = ["### Live bench cross-check", ""]
    any_measured = False
    for label, report, studio in (
        ("Banking77", banking, banking_studio),
        ("Invoices", invoices, invoices_studio),
    ):
        source = studio or report
        bench = source.get("live_bench") or {}
        student_only, cascade = bench.get("student_only"), bench.get("cascade")
        if not student_only and not cascade:
            lines += [f"**{label}**: not measured yet.", ""]
            continue
        any_measured = True
        header = f"**{label}**"
        if studio:
            header += f" (measured on a {hardware_text(studio.get('hardware'))}, {date_only(studio.get('date'))})"
        lines += [header, ""]
        rows = []
        for mode_label, run in (("Student only", student_only), ("Cascade", cascade)):
            if not run:
                continue
            latency = run.get("latency_ms") or {}
            load_average = (run.get("machine_state") or {}).get("load_average")
            rows.append([
                mode_label, run.get("run_id") or DASH, run.get("date") or DASH, integer(run.get("n")),
                ms(latency.get("p50")), ms(latency.get("p95")), pct(run.get("escalation_rate")),
                usd(run.get("spend_usd")), load_average_text(load_average),
            ])  # fmt: skip
        lines += md_table(["Mode", "Run", "Date", "n", "p50 ms", "p95 ms", "Escalated", "Spend", "Load average"], rows)
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
    lines = ["### Training on the MacBook Air", ""]
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


#: The one metric ``scripts/compare_reports.py`` rows are compacted to per task: accuracy for classification,
#: field micro-F1 for extraction, matching the metric each task's own quality table leads with.
MAIN_METRIC_BY_TASK = {"banking77": "accuracy", "invoices": "field_micro_f1"}


def _gain_phrase(reason: Any) -> str | None:
    """The tail of a base-model selection reason that states the gain, e.g. ``large gains 0.97 points, below
    the 1.00-point minimum`` -- quoted alone rather than the whole reason, which also names the run and its
    validation score."""
    match = re.search(r"(?:large|small) gains?.*$", str(reason or ""))
    return match.group(0) if match else None


def _reproducibility_section(reproduction: Mapping[str, Any] | None) -> list[str]:
    """``reports/reproduction.json``, written by ``scripts/compare_reports.py`` after ``scripts/reproduce.sh`` ran
    again on a second machine: optional, so nothing is rendered until it exists."""
    if not reproduction:
        return []
    reference, rerun = reproduction.get("reference") or {}, reproduction.get("rerun") or {}
    tasks = reproduction.get("tasks") or {}
    tolerance = _num(reproduction.get("tolerance_pts"))
    tolerance_text = f"{tolerance:.1f} points" if tolerance is not None else DASH
    lines = [
        "### Reproducibility on a second machine",
        "",
        f"`scripts/reproduce.sh` (full profile) was run again on a second machine and compared with "
        f"`scripts/compare_reports.py`; tolerance {tolerance_text} on each test-split metric.",
        "",
        f"Reference: {hardware_text(reference.get('hardware'))}, {date_only(reference.get('date'))}. Rerun: "
        f"{hardware_text(rerun.get('hardware'))}, {date_only(rerun.get('date'))}.",
        "",
    ]
    task_rows: list[list[Any]] = []
    metric_rows: list[list[Any]] = []
    selected_sentences: list[str] = []
    for task in sorted(tasks):
        result = tasks[task] or {}
        max_diff = _num(result.get("max_abs_diff_pts"))
        task_rows.append([
            _task_label(task), f"{max_diff:.2f} pts" if max_diff is not None else DASH,
            "yes" if result.get("all_within") else "no",
        ])  # fmt: skip
        main_metric = MAIN_METRIC_BY_TASK.get(task, "accuracy")
        for row in result.get("rows") or []:
            name = str(row.get("row") or DASH)
            if row.get("metric") != main_metric or name.lower().startswith("zero-shot"):
                continue  # the compact table is student/teacher/tfidf only, at each task's main metric
            diff = _num(row.get("diff_pts"))
            metric_rows.append([
                _task_label(task), name, col_label(main_metric), pct(row.get("reference")), pct(row.get("rerun")),
                f"{diff:+.2f} pts" if diff is not None else DASH,
            ])  # fmt: skip
        sel = result.get("selected_run") or {}
        if not sel:
            continue
        if sel.get("same"):
            selected_sentences.append(
                f"{_task_label(task)}: the selected run matched on both machines (`{sel.get('reference')}`)."
            )
        else:
            ref_gain = _gain_phrase(sel.get("reference_reason")) or sel.get("reference_reason") or DASH
            rerun_gain = _gain_phrase(sel.get("rerun_reason")) or sel.get("rerun_reason") or DASH
            selected_sentences.append(
                f"{_task_label(task)}: the selected run differed — `{sel.get('reference')}` on the reference "
                f"machine vs `{sel.get('rerun')}` on the rerun (reference: “{ref_gain}”; rerun: “{rerun_gain}”)."
            )
    lines += [*md_table(["Task", "Max abs difference", "All rows within tolerance"], task_rows), ""]
    if metric_rows:
        lines += ["<details>", "<summary>Main metric of every row, both machines</summary>", "",
                  *md_table(["Task", "Row", "Metric", "Reference", "Rerun", "Diff"], metric_rows), "",
                  "</details>", ""]  # fmt: skip
    if selected_sentences:
        lines += [" ".join(selected_sentences), ""]
    outside = []
    for task, result in (reproduction.get("tasks") or {}).items():
        for row in result.get("rows") or []:
            if not row.get("within"):
                diff = _num(row.get("diff_pts")) or 0.0
                outside.append(
                    f"{_task_label(task)} {row.get('row')}, {col_label(str(row.get('metric')))}: "
                    f"{pct(row.get('reference'))} vs {pct(row.get('rerun'))} ({diff:+.2f} points)"
                )
    overall = "yes" if reproduction.get("all_within") else "no"
    lines += [f"All rows within tolerance across every task: {overall}.", ""]
    if outside:
        lines += ["Outside the tolerance:", "", *[f"- {item}" for item in outside], ""]
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
        row = _selected_student_row(report)
        calibration = (row or {}).get("calibration") or {}
        ece, isotonic_ece = calibration.get("ece"), calibration.get("isotonic_ece")
        if ece is None or isotonic_ece is None:
            continue
        run_id = (report.get("selected_run") or {}).get("run_id") or (row or {}).get("selected_run") or DASH
        # calibration.ece is the same basis as the quality table's ECE column for this row (see calibration's
        # own "reference"); isotonic_ece shares one "reference" key with it in this schema, but the check
        # below still says so explicitly if a future report ever gives the isotonic figure its own basis.
        reference = calibration.get("reference") or "teacher"
        isotonic_reference = calibration.get("isotonic_reference") or reference
        basis_note = f" (isotonic figure is vs {isotonic_reference})" if isotonic_reference != reference else ""
        lines += [
            f"{label} student `{run_id}`, ECE on test vs {reference}: {pct(ece)} raw vs {pct(isotonic_ece)} "
            f"after isotonic calibration{basis_note}.",
            "",
        ]
    return lines


def _selected_student_row(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The quality row for the report's selected run (student, teacher labels) -- not just the first student
    row, which can be a different base size than the one actually chosen and served (invoices selects 1.5B,
    but the 0.5B teacher-labels row lists first)."""
    run_id = (report.get("selected_run") or {}).get("run_id")
    rows = (report.get("quality") or {}).get("rows") or []
    if run_id is not None:
        for row in rows:
            if row.get("system") == "student" and run_id in (row.get("run_ids") or []):
                return row
    return find_row(rows, "student", labels="teacher")


def _decision_sentence(decision: Mapping[str, Any]) -> str:
    gain, ratio = _num(decision.get("gain")), _num(decision.get("latency_ratio"))
    min_gain, max_ratio = _num(decision.get("min_gain")), _num(decision.get("max_latency_ratio"))
    if gain is None or ratio is None:
        return ""
    gain_word = "meets" if min_gain is not None and gain >= min_gain else "is below"
    ratio_word = "under" if max_ratio is not None and ratio < max_ratio else "at or above"
    verdict = "the smaller base was kept" if decision.get("choice") == "small" else "the larger base was selected"
    # Two decimals on the gain (rather than the usual one) so a near-threshold gain such as 0.97 vs a 1-point
    # minimum cannot round to the same digits as the threshold it is being compared against. In points, not
    # percent: the rule is a difference between two scores, not a share of anything.
    return (
        f"Rule: use the larger base only if it gains at least {min_gain_points(min_gain)} on the validation "
        f"metric and its p95 latency stays under {plain(max_ratio, 1)}x the smaller model's. Here the gain "
        f"{gain_word} the {min_gain_points(min_gain)} minimum ({points(gain, 2)}) and the p95 ratio "
        f"({plain(ratio, 2)}x) is {ratio_word} the {plain(max_ratio, 1)}x limit, so {verdict}."
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
    lines += [
        "The rule compares point estimates on the validation split (the best seed of each base); no interval is "
        "computed for this decision, so a gain close to the minimum can go either way on a rerun (see "
        "Reproducibility above).",
        "",
    ]
    return lines


DIVERGED_AGREEMENT = 0.10  # a quick run whose validation agreement stays below this counts as diverged


def _ablation_summary(ablation: Mapping[str, Any]) -> dict[str, Any] | None:
    summary = ablation.get("summary") or {}
    constant, warm = summary.get("constant") or {}, summary.get("warmup_cosine") or {}
    if not constant or not warm:
        return None
    runs = ablation.get("runs") or []

    def diverged(schedule: str) -> tuple[int, int]:
        values = [_num(r.get("valid_agreement")) for r in runs if r.get("lr_schedule") == schedule]
        values = [v for v in values if v is not None]
        return sum(v < DIVERGED_AGREEMENT for v in values), len(values)

    return {
        "constant": constant,
        "warm": warm,
        "div_constant": diverged("constant"),
        "div_warm": diverged("warmup_cosine"),
    }


def _cpu_name(report: Mapping[str, Any] | None) -> str:
    """``MacBook Air (Apple M5)``, or the chip alone when the product is unknown."""
    hardware = (report or {}).get("hardware") or {}
    cpu = str(hardware.get("cpu") or "second machine")
    product = product_name(hardware)
    return f"{product} ({cpu})" if product else cpu


def _early_checkpoints(training: Mapping[str, Any] | None, task: str) -> tuple[int, int]:
    """(runs whose best checkpoint is at or before the end of warm-up, all teacher-label 0.5B runs) for a task."""
    runs = [
        r
        for r in (training or {}).get("runs") or []
        if r.get("task") == task and r.get("labels") == "teacher" and "0.5B" in str(r.get("base_model"))
    ]
    early = [
        r
        for r in runs
        if _num(r.get("best_iteration")) is not None
        and _num(r.get("warmup_iterations")) is not None
        and _num(r.get("best_iteration")) <= _num(r.get("warmup_iterations"))
    ]
    return len(early), len(runs)


def _lr_ablation_bullet(
    ablation: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    second_training: Mapping[str, Any] | None = None,
) -> str | None:
    if not ablation:
        return None
    first = _ablation_summary(ablation)
    if first is None:
        return None
    n_seeds = len(ablation.get("seeds") or []) or first["div_warm"][1]

    def machine_text(summary: Mapping[str, Any], name: str) -> str:
        return (
            f"on the {name} a constant rate reached {pct(summary['constant'].get('valid_agreement_mean'))} ± "
            f"{pct(summary['constant'].get('valid_agreement_std'))} mean validation agreement with the teacher and "
            f"linear warm-up + cosine decay {pct(summary['warm'].get('valid_agreement_mean'))} ± "
            f"{pct(summary['warm'].get('valid_agreement_std'))}, with {summary['div_constant'][0]} and "
            f"{summary['div_warm'][0]} of {summary['div_warm'][1]} seeds diverging (agreement below "
            f"{pct(DIVERGED_AGREEMENT, 0)})"
        )

    parts = [machine_text(first, _cpu_name(ablation))]
    second_summary = _ablation_summary(second) if second else None
    if second_summary is not None:
        parts.append(machine_text(second_summary, _cpu_name(second)))
    text = (
        f"Learning-rate schedule at the spec's peak rate (quick profile, 200 iterations, {n_seeds} seeds, Banking77): "
        + "; ".join(parts)
        + ". Warm-up + cosine is the default and did better on average, but it does not make short runs at this "
        "peak rate reliable."
    )
    early = [(_cpu_name(ablation), _early_checkpoints(training, "invoices"))]
    if second_training is not None:
        early.append((_cpu_name(second), _early_checkpoints(second_training, "invoices")))
    early = [(name, counts) for name, counts in early if counts[1]]
    if early:
        described = " and ".join(f"{counts[0]} of {counts[1]} on the {name}" for name, counts in early)
        text += (
            f" In the full profile, Banking77 converged on every seed; on invoices, the best validation checkpoint of "
            f"{described} 0.5B seeds came at or before the end of warm-up, so those students stopped early."
        )
    return text


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
        f"The bigger Banking77 student (1.5B vs 0.5B, teacher labels) gained {points(gain, 2)} on validation "
        f"agreement, {verdict} the {min_gain_points(min_gain)} minimum the base-model rule requires, for "
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


#: AUROC (0-1 scale) and ECE (fraction) differences at or under these are called "about equal" in prose below,
#: rather than crediting either side with a real advantage on a gap that is noise-sized.
AUROC_TIE = 0.01
ECE_TIE = 0.003


def _direction_note(
    primary: Any, alternative: Any, fmt: Callable[[Any], str], *, higher_is_better: bool, tie: float
) -> str:
    """``0.846 vs 0.858 (higher for the alternative)`` -- the comparison word is derived only from comparing the
    two numbers (against a small tie tolerance in the metric's own units), never assumed from which definition
    was actually chosen; that is what let the old text claim "did not improve" on a task where it did."""
    p, a = _num(primary), _num(alternative)
    text = f"{fmt(primary)} vs {fmt(alternative)}"
    if p is None or a is None:
        return text
    if abs(p - a) <= tie:
        return f"{text} (about equal)"
    a_is_better = (a > p) if higher_is_better else (a < p)
    word = "higher" if higher_is_better else "lower"
    return f"{text} ({word} for the {'alternative' if a_is_better else 'primary'})"


def _auroc_note(primary: Any, alternative: Any) -> str:
    return _direction_note(primary, alternative, plain, higher_is_better=True, tie=AUROC_TIE)


def _ece_note(primary: Any, alternative: Any) -> str:
    return _direction_note(primary, alternative, pct, higher_is_better=False, tie=ECE_TIE)


def _ece_ratio_phrase(primary: Any, alternative: Any) -> str:
    """``2.8% vs 9.2%, about 3.3x lower for the primary`` -- the ratio that motivated keeping the primary
    definition on validation, computed from the two values rather than asserted as "about three times" from
    memory."""
    p, a = _num(primary), _num(alternative)
    text = f"{pct(primary)} vs {pct(alternative)}"
    if p is None or a is None or p <= 0 or a <= p:
        return text
    return f"{text}, about {a / p:.1f}x lower for the primary"


def _confidence_bullet(banking: Mapping[str, Any], invoices: Mapping[str, Any]) -> str | None:
    valid_bits, test_bits = [], []
    for label, report in (("Banking77", banking), ("Invoices", invoices)):
        comparison = (report.get("quality") or {}).get("confidence_comparison") or {}
        valid, test = comparison.get("valid") or {}, comparison.get("test") or {}
        if valid.get("chosen") != "primary" or not valid.get("alternative") or not test.get("alternative"):
            continue
        v_primary, v_alt = valid.get("primary") or {}, valid["alternative"]
        t_primary, t_alt = test.get("primary") or {}, test["alternative"]
        v_auroc = _auroc_note(v_primary.get("auroc_vs_teacher"), v_alt.get("auroc_vs_teacher"))
        v_ece = _ece_ratio_phrase(v_primary.get("ece_vs_teacher"), v_alt.get("ece_vs_teacher"))
        valid_bits.append(f"{label} AUROC {v_auroc}, ECE {v_ece}")
        t_auroc = _auroc_note(t_primary.get("auroc_vs_teacher"), t_alt.get("auroc_vs_teacher"))
        t_ece = _ece_note(t_primary.get("ece_vs_teacher"), t_alt.get("ece_vs_teacher"))
        test_bits.append(f"{label} AUROC {t_auroc}, ECE {t_ece}")
    if not valid_bits:
        return None
    return (
        "Two confidence definitions were compared on validation, before either was used at test time: "
        + "; ".join(valid_bits)
        + "; so the primary (trie-constrained token-probability product) was kept over the alternative (free "
        "greedy generation, mean per-token log-probability). Reported honestly, on test: " + "; ".join(test_bits) + "."
    )


def _threshold_transfer_bullet(label: str, report: Mapping[str, Any], group_noun: str) -> str | None:
    """Only when the validation-chosen threshold missed the target on test: the numbers that show it did not
    transfer, so this never claims a miss the report does not have."""
    op = report.get("operating_point") or {}
    if op.get("target_met_on_test") is not False:
        return None
    quality, rates = op.get("quality") or {}, op.get("escalation_rate") or {}
    metric_label, reference = col_label(str(op.get("metric") or "")), op.get("reference") or "teacher"
    return (
        f"{label}: the threshold chosen on the validation {group_noun} did not transfer to the unseen test "
        f"{group_noun} — {metric_label} against the {reference} met the {pct(op.get('target'))} target on "
        f"validation ({pct(quality.get('valid'))}, {pct(rates.get('valid'))} escalation) but fell to "
        f"{pct(quality.get('test'))} on test ({pct(rates.get('test'))} escalation)."
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
    banking_studio = load_optional("banking77/report_mac_studio.json")
    invoices_studio = load_optional("invoices/report_mac_studio.json")
    training = load_optional("training.json")
    spend = load_optional("spend.json")
    downloads = load_optional("downloads.json")
    ablation = load_optional("ablations/lr_schedule.json")
    ablation_second = load_optional("ablations/lr_schedule.second_machine.json")
    training_second = load_optional("reproduction/training.json")
    reproduction = load_optional("reproduction.json")
    bakeoff_banking = load_optional("bakeoff/banking77.json")
    bakeoff_variants = {
        "snake-case labels": load_optional("bakeoff/banking77.labels-snake-case.json"),
        "labels with examples": load_optional("bakeoff/banking77.labels-with-examples.json"),
    }

    lines: list[str] = ["## Results", ""]
    lines += _provenance_paragraph(banking, invoices)
    lines += _machines_paragraph(banking, banking_studio)
    lines += _bottom_line(banking, invoices)
    lines += [METRIC_DEFINITIONS, ""]
    lines += _quality_section_banking77(banking)
    lines += _quality_section_invoices(invoices)
    lines += _cost_and_latency_section(banking, invoices, banking_studio, invoices_studio)
    lines += _live_bench_section(banking, invoices, banking_studio, invoices_studio)
    lines += _training_section(training)
    lines += _reproducibility_section(reproduction)
    lines += _calibration_section(banking, invoices)
    lines += _base_model_section(banking, invoices)

    lines += ["### What didn't work", ""]
    bullets = [
        _lr_ablation_bullet(ablation, ablation_second, training, training_second),
        _size_ceiling_bullet(banking),
        _label_ceiling_bullet(banking),
        _prompt_variant_bullet(bakeoff_banking, bakeoff_variants),
        _confidence_bullet(banking, invoices),
        _threshold_transfer_bullet("Banking77", banking, "queries"),
        _threshold_transfer_bullet("Invoices", invoices, "layouts"),
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
