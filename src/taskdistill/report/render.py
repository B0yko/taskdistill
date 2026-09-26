"""Render a report JSON (``report.builder.build_report``) as Markdown.

Formatting is the same everywhere: rates and scores as percentages with one decimal, 95% intervals as ``[lo, hi]`` in
the same unit, differences in percentage points, AUROC with three decimals, latency in ms and dollars with three
significant digits. Only relative file names from the report are printed, never an absolute path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

DASH = "—"
LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "agreement": "Agreement",
    "json_validity": "JSON validity",
    "field_micro_f1": "Field micro-F1",
    "field_exact_match": "Field EM",
    "doc_exact_match": "Doc EM",
    "ece": "ECE",
    "auroc": "AUROC",
    "field_f1": "field micro-F1",
    "auroc_vs_teacher": "AUROC vs teacher",
    "auroc_vs_gold": "AUROC vs gold",
    "accuracy_vs_teacher": "Accuracy vs teacher",
    "accuracy_vs_gold": "Accuracy vs gold",
    "ece_vs_teacher": "ECE vs teacher",
    "ece_vs_gold": "ECE vs gold",
    "mean_confidence": "Mean confidence",
}
#: Metrics shown as plain numbers (not percentages).
PLAIN = frozenset({"auroc", "brier"})
_COUNT_KEYS = frozenset({"n", "tp", "fp", "fn", "count", "requests", "records", "n_groups"})


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def pct(value: Any) -> str:
    number = _num(value)
    return DASH if number is None else f"{number * 100:.1f}%"


def points(value: Any) -> str:
    number = _num(value)
    return DASH if number is None else f"{number * 100:+.1f} pts"


def plain(value: Any, digits: int = 3) -> str:
    number = _num(value)
    return DASH if number is None else f"{number:.{digits}f}"


def interval(ci: Any, metric: str = "") -> str:
    """``[lo, hi]`` in percent (one decimal), or with three decimals for plain metrics."""
    if not isinstance(ci, list | tuple) or len(ci) != 2:
        return ""
    lo, hi = _num(ci[0]), _num(ci[1])
    if lo is None or hi is None:
        return ""
    if metric in PLAIN:
        return f"[{lo:.3f}, {hi:.3f}]"
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def metric_value(value: Any, metric: str) -> str:
    return plain(value) if metric in PLAIN else pct(value)


def ms(value: Any) -> str:
    number = _num(value)
    if number is None:
        return DASH
    return f"{number:,.0f}" if abs(number) >= 100 else f"{number:.1f}"


def usd(value: Any) -> str:
    """Dollars with three significant digits and no exponent (``$0.000167``, ``$0.412``, ``$12.35``)."""
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
    """``Mac17,4, Apple M5, 24 GB, macOS 26.1`` from a ``hardware_info()`` dict."""
    if not isinstance(hardware, Mapping):
        return DASH
    memory = _num(hardware.get("memory_gb"))
    parts = [hardware.get("model"), hardware.get("cpu"), f"{memory:g} GB" if memory else None, hardware.get("os")]
    return ", ".join(str(x) for x in parts if x) or DASH


def _load_text(state: Any) -> str:
    load = state.get("load_average") if isinstance(state, Mapping) else None
    if not isinstance(load, list | tuple) or not load:
        return ""
    return "load average " + " / ".join(f"{_num(x):.2f}" if _num(x) is not None else DASH for x in load)


def _provenance_text(prov: Any) -> str:
    """``<date> on <hardware>, `<command>` `` for an eval or bench source."""
    if not isinstance(prov, Mapping):
        return ""
    parts = [str(prov.get("date") or DASH), f"on {hardware_text(prov.get('hardware'))}"]
    load = _load_text(prov.get("machine_state"))
    text = " ".join(parts) + (f" ({load} before timing)" if load else "")
    if prov.get("command"):
        text += f", `{prov['command']}`"
    return text


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(_cell(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _label(metric: str) -> str:
    return LABELS.get(metric, metric.replace("_", " "))


# -- sections --------------------------------------------------------------------------------------


def _metric_cell(row: Mapping[str, Any], metric: str) -> str:
    metrics = row.get("metrics") or {}
    if metric not in metrics:
        return DASH
    value = metrics.get(metric)
    if value is None:
        return "reference" if row.get("system") == "teacher" and metric == "agreement" else DASH
    text = metric_value(value, metric)
    std = _num((row.get("std") or {}).get(metric))
    if (row.get("n_seeds") or 0) > 1 and std is not None:
        text += f" ± {std:.3f}" if metric in PLAIN else f" ± {std * 100:.1f}"
    ci = interval((row.get("ci") or {}).get(metric), metric)
    return f"{text} {ci}" if ci else text


def _quality(report: Mapping[str, Any]) -> list[str]:
    quality = report.get("quality") or {}
    rows = quality.get("rows") or []
    if not rows:
        return ["## Quality", "", "No test evaluation yet.", ""]
    columns = list(quality.get("student_columns") or quality.get("columns") or [])
    lines = [
        f"## Quality (test split, n = {integer(quality.get('n'))})",
        "",
        "Cells are the score and its 95% paired bootstrap interval (1,000 resamples). Rows over several seeds show "
        "mean ± sample standard deviation; their interval is that of the validation-selected seed. Teacher and "
        "cascade numbers come from recorded teacher outputs. ECE and AUROC use raw student confidence.",
        "",
    ]
    body = []
    for row in rows:
        body.append(
            [row.get("name"), integer(row.get("n"))]
            + [_metric_cell(row, m) for m in columns]
            + [", ".join(f"`{r}`" for r in row.get("run_ids") or [])]
        )
    lines += table(["System", "n", *[_label(m) for m in columns], "Runs"], body)
    lines.append("")
    if quality.get("cluster_bootstrap"):
        groups = quality.get("n_groups")
        lines += [
            f"Cluster bootstrap over {integer(groups)} groups (templates): 95% intervals.",
            "",
        ]
        cluster_rows = [
            [row.get("name")] + [interval((row.get("ci_cluster") or {}).get(m), m) or DASH for m in columns]
            for row in rows
            if row.get("ci_cluster")
        ]
        lines += [*table(["System", *[_label(m) for m in columns]], cluster_rows), ""]
    seeds = [row for row in rows if (row.get("n_seeds") or 0) > 1]
    for row in seeds:
        lines += [f"Per-seed values, {row.get('name')} (selected: `{row.get('selected_run')}`):", ""]
        per_seed = [
            [f"`{seed.get('run_id')}`", seed.get("seed")]
            + [metric_value((seed.get("metrics") or {}).get(m), m) for m in columns]
            for seed in row.get("per_seed") or []
        ]
        lines += [*table(["Run", "Seed", *[_label(m) for m in columns]], per_seed), ""]
    for key, title in (
        ("per_group", "Per-group (template) scores"),
        ("per_field", "Per-field exact match"),
        ("per_trait", "Per-trait breakdown"),
    ):
        block = quality.get(key)
        if isinstance(block, Mapping) and block:
            row_title = "Trait" if key == "per_trait" else "Group"
            details = _per_field(block) if key == "per_field" else _breakdown(block, row_title)
            lines += ["<details>", f"<summary>{title}</summary>", "", *details, "</details>", ""]
    comparison = quality.get("confidence_comparison")
    if isinstance(comparison, Mapping) and comparison:
        lines += ["<details>", "<summary>Confidence definitions compared</summary>", "", *_comparison(comparison)]
        lines += ["</details>", ""]
    provenance = [prov for prov in quality.get("provenance") or [] if isinstance(prov, Mapping)]
    if provenance:
        prov_rows = [
            [f"`{prov.get('run_id')}`", prov.get("date") or DASH, hardware_text(prov.get("hardware")),
             f"`{prov['command']}`" if prov.get("command") else DASH]
            for prov in provenance
        ]  # fmt: skip
        lines += ["<details>", "<summary>Evaluation dates, hardware and commands</summary>", ""]
        lines += [*table(["Run", "Date", "Hardware", "Command"], prov_rows), "", "</details>", ""]
    scorings = _num(quality.get("test_scorings"))
    if scorings is not None:
        lines += [f"The test split has been scored {integer(scorings)} time{'' if scorings == 1 else 's'}.", ""]
    return lines


def _leaf(key: str, value: Any, *, percent: bool = True) -> str:
    """One scalar: dollars for costs, integers for counts, AUROC/Brier plain, other fractions as percentages."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = _num(value)
    if number is None:
        return DASH if value is None else str(value)
    if "usd" in key or "cost" in key:
        return usd(number)
    if isinstance(value, int) or key in _COUNT_KEYS:
        return integer(number)
    if key in PLAIN or "auroc" in key or "brier" in key:
        return plain(number)
    return pct(number) if percent and -1.0 <= number <= 1.0 else f"{number:,.4g}"


def _is_leaf(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _nested(block: Any, depth: int = 0) -> list[str]:
    """A dict of scalars as a key/value table, a dict of such dicts as one table, deeper dicts per sub-heading."""
    if not isinstance(block, Mapping) or not block:
        return [str(block), ""]
    values = list(block.values())
    if all(_is_leaf(v) for v in values):
        return [*table(["Key", "Value"], [[k, _leaf(k, v)] for k, v in block.items()]), ""]
    if all(isinstance(v, Mapping) for v in values) and all(
        any(_is_leaf(x) for x in v.values()) for v in values if isinstance(v, Mapping)
    ):
        columns: list[str] = []
        for v in values:
            for k, x in v.items():
                if _is_leaf(x) and k not in columns:
                    columns.append(k)
        rows = [[name] + [_leaf(c, (row or {}).get(c)) for c in columns] for name, row in block.items()]
        return [*table(["", *[_label(c) for c in columns]], rows), ""]
    lines: list[str] = []
    for name, value in block.items():
        if _is_leaf(value):
            lines += [f"- {name}: {_leaf(name, value)}", ""]
            continue
        lines += [f"{'#' * min(6, 4 + depth)} {name}", "", *_nested(value, depth + 1)]
    return lines


def _breakdown(block: Mapping[str, Any], title: str) -> list[str]:
    """Per-group or per-trait scores (``{group: {"n", <system>: {<metric>: value}}}``): one table per system."""
    entries = [(name, entry) for name, entry in block.items() if isinstance(entry, Mapping)]
    systems: list[str] = []
    for _, entry in entries:
        systems += [k for k, v in entry.items() if isinstance(v, Mapping) and k not in systems]
    if not systems:
        return _nested(block)
    lines: list[str] = []
    for system in systems:
        metrics: list[str] = []
        for _, entry in entries:
            scores = entry.get(system)
            if isinstance(scores, Mapping):
                metrics += [m for m, v in scores.items() if _is_leaf(v) and m not in metrics]
        rows = [
            [name, _leaf("n", entry.get("n"))] + [_leaf(m, (entry.get(system) or {}).get(m)) for m in metrics]
            for name, entry in entries
        ]
        lines += [f"{system}:", "", *table([title, "n", *[_label(m) for m in metrics]], rows), ""]
    return lines


def _per_field(block: Mapping[str, Any]) -> list[str]:
    """Per-field exact match (``{"fields", "vs_gold": {<system>: {<field>: value}}, "vs_teacher": ...}``)."""
    lines: list[str] = []
    for reference in ("vs_gold", "vs_teacher"):
        by_system = block.get(reference)
        if not isinstance(by_system, Mapping) or not by_system:
            continue
        systems = [name for name, scores in by_system.items() if isinstance(scores, Mapping)]
        fields = [str(f) for f in block.get("fields") or []]
        for name in systems:
            fields += [f for f in by_system[name] if f not in fields]
        rows = [[f"`{f}`"] + [pct(by_system[name].get(f)) for name in systems] for f in fields]
        lines += [f"Against the {reference.removeprefix('vs_')}:", "", *table(["Field", *systems], rows), ""]
    return lines or _nested(block)


def _comparison(block: Mapping[str, Any]) -> list[str]:
    """The primary and alternative confidence definitions per split, the chosen one marked."""
    columns: list[str] = []
    rows: list[list[Any]] = []
    entries = [(split, entry) for split, entry in block.items() if isinstance(entry, Mapping)]
    for _, entry in entries:
        for name in ("primary", "alternative"):
            scores = entry.get(name)
            if isinstance(scores, Mapping):
                columns += [k for k, v in scores.items() if _is_leaf(v) and k not in columns]
    for split, entry in entries:
        for name in ("primary", "alternative"):
            scores = entry.get(name)
            if isinstance(scores, Mapping):
                label = f"{name} (chosen)" if entry.get("chosen") == name else name
                rows.append([split, label] + [_leaf(c, scores.get(c)) for c in columns])
    if not rows:
        return _nested(block)
    return [*table(["Split", "Confidence", *[_label(c) for c in columns]], rows), ""]


def _target_text(op: Mapping[str, Any]) -> str:
    metric = _label(str(op.get("metric") or ""))
    reference = op.get("reference")
    if _num(op.get("target")) is not None:
        return f"{metric} ≥ {pct(op.get('target'))} against the {reference}"
    if _num(op.get("max_drop")) is not None:
        return f"{metric} drop ≤ {points(op.get('max_drop')).lstrip('+')} against the teacher's {reference} score"
    return f"{metric} against the {reference}"


def _signed_interval(ci: Any) -> str:
    if not isinstance(ci, list | tuple) or len(ci) != 2:
        return ""
    lo, hi = _num(ci[0]), _num(ci[1])
    return "" if lo is None or hi is None else f"[{lo * 100:+.1f}, {hi * 100:+.1f}]"


def _difference(entry: Any) -> str:
    """A difference in percentage points with its paired (and cluster) 95% interval."""
    if not isinstance(entry, Mapping):
        return DASH
    text = points(entry.get("point"))
    paired, cluster = _signed_interval(entry.get("ci")), _signed_interval(entry.get("ci_cluster"))
    if paired:
        text += f" {paired}"
    if cluster:
        text += f", cluster {cluster}"
    return text


def _operating_point(report: Mapping[str, Any]) -> list[str]:
    op = report.get("operating_point") or {}
    lines = ["## Operating point", ""]
    if not op.get("available"):
        return [*lines, "No test evaluation of the selected run.", ""]
    threshold = op.get("threshold")
    threshold_text = "always escalate" if threshold is None else f"{threshold:.4f}"
    rates = op.get("escalation_rate") or {}
    quality = op.get("quality") or {}
    diffs = op.get("cascade_minus_teacher") or {}
    target_diff = diffs.get("target_metric") or {}
    gold_diff = diffs.get("gold_metric") or {}
    met = op.get("target_met_on_test")
    rows = [
        ["Selected run", f"`{op.get('run_id')}`"],
        ["Target", _target_text(op)],
        ["Chosen threshold (on validation)", threshold_text],
        ["Escalation rate, validation / test", f"{pct(rates.get('valid'))} / {pct(rates.get('test'))}"],
        ["Cascade quality, validation / test", f"{pct(quality.get('valid'))} / {pct(quality.get('test'))}"],
        [
            f"Cascade − teacher, {_label(str(op.get('metric')))} against the {op.get('reference')} (target)",
            _difference(target_diff),
        ],
        [f"Cascade − teacher, {_label(str(gold_diff.get('metric') or 'gold metric'))} (gold)", _difference(gold_diff)],
        ["Target held on test", DASH if met is None else ("yes" if met else "no")],
    ]
    if op.get("provenance"):
        rows.append(["Evaluated", _provenance_text(op["provenance"])])
    lines += [*table(["", "Value"], rows), ""]
    if isinstance(target_diff, Mapping) and target_diff.get("ci") is None and target_diff.get("source"):
        lines += [f"The target difference comes from the {target_diff['source']}.", ""]
    if op.get("warning"):
        lines += [f"Warning: {op['warning']}", ""]
    if op.get("curve_png"):
        lines += [f"![Cascade quality against escalation rate]({op['curve_png']})", ""]
    for png in op.get("reliability_png") or []:
        lines += [f"![Reliability diagram]({png})", ""]
    return lines


def _cost_latency(report: Mapping[str, Any]) -> list[str]:
    cost = report.get("cost_latency") or {}
    if not cost:
        return []
    teacher, student, cascade = cost.get("teacher") or {}, cost.get("student") or {}, cost.get("cascade") or {}
    assumptions = (report.get("assumptions") or cost.get("assumptions") or {}).get("text", "")
    concurrency = teacher.get("concurrency")
    student_source = "bench" if student.get("source") == "bench" else "eval (flagged)"
    rows = [
        [
            "Teacher only",
            usd(teacher.get("usd_per_1k")),
            ms((teacher.get("latency_ms") or {}).get("p50")),
            ms((teacher.get("latency_ms") or {}).get("p95")),
            integer(teacher.get("n")),
            f"recorded live labelling calls, concurrency {concurrency}",
        ],
        [
            "Student only",
            usd(student.get("usd_per_1k")),
            ms((student.get("latency_ms") or {}).get("p50")),
            ms((student.get("latency_ms") or {}).get("p95")),
            integer((student.get("latency_ms") or {}).get("n")),
            student_source,
        ],
        [
            "Cascade",
            usd(cascade.get("usd_per_1k")),
            ms((cascade.get("latency_ms") or {}).get("p50")),
            ms((cascade.get("latency_ms") or {}).get("p95")),
            integer(cascade.get("n")),
            _cascade_source(cascade),
        ],
    ]
    lines = ["## Cost and latency", ""]
    lines += [*table(["System", "$ / 1k requests", "p50 ms", "p95 ms", "n", "Source"], rows), ""]
    lines += [
        f"Assumptions: {assumptions}. Teacher cost is the mean recorded `usage.cost` of the test requests; teacher "
        "latency is the recorded latency of the live labelling calls (cache hits and retried attempts excluded)"
        + (_recorded_between(teacher.get("recorded_between")))
        + ".",
        "",
        f"Student latency: {student.get('note', '')}."
        + (f" Measured {_provenance_text(student['provenance'])}." if student.get("provenance") else ""),
        "",
        "Cascade latency is composed per test request as student latency plus the recorded teacher latency when the "
        "request escalates at the chosen threshold; cascade cost is the student energy plus the teacher cost of "
        "escalated requests.",
        "",
    ]
    teacher_list, cascade_list = teacher.get("list_price_usd_per_1k"), cascade.get("list_price_usd_per_1k")
    if teacher_list is not None and cascade_list is not None:
        lines += [
            f"At the list price without the provider's prompt cache (the same token counts), the teacher costs "
            f"{usd(teacher_list)} and the cascade {usd(cascade_list)} per 1k requests.",
            "",
        ]
    return lines


def _recorded_between(dates: Any) -> str:
    if not isinstance(dates, list | tuple) or len(dates) != 2:
        return ""
    return f", recorded on {dates[0]}" if dates[0] == dates[1] else f", recorded {dates[0]} to {dates[1]}"


def _cascade_source(cascade: Mapping[str, Any]) -> str:
    text = f"composed per request, {pct(cascade.get('escalation_rate'))} escalated"
    latency_n, n = _num(cascade.get("latency_n")), _num(cascade.get("n"))
    if latency_n and n is not None and latency_n != n:
        text += (
            f"; latency over {integer(latency_n)} {cascade.get('latency_scope') or 'requests'} "
            f"({pct(cascade.get('latency_escalation_rate'))} escalated)"
        )
    imputed = cascade.get("imputed") or {}
    count = max(_num(imputed.get("teacher_cost")) or 0, _num(imputed.get("teacher_latency")) or 0)
    if count:
        text += f"; {integer(count)} teacher records imputed"
    return text


def _break_even(report: Mapping[str, Any]) -> list[str]:
    be = report.get("break_even") or {}
    if not be:
        return []
    lines = ["## Break-even volume", ""]
    labelling = f"labelling {usd(be.get('labelling_usd'))} ({be.get('labelling_source')})"
    training = f"training energy {usd(be.get('training_usd'))}"
    savings = f"savings {usd(be.get('savings_usd_per_request'))} per request"
    if be.get("volume") is not None:
        bound = "at least " if be.get("lower_bound") else ""
        lines.append(f"**{bound}{integer(be['volume'])} requests** = ({labelling} + {training}) / {savings}.")
    else:
        lines.append(f"Not reached: {be.get('reason')}. ({labelling}; {training}; {savings}.)")
    list_price = be.get("list_price") or {}
    if list_price.get("volume") is not None:
        lines.append(
            f"At the list price without prompt caching: {integer(list_price['volume'])} requests "
            f"(savings {usd(list_price.get('savings_usd_per_request'))} per request; labelling cost as paid)."
        )
    for note in be.get("notes") or []:
        lines.append(f"Note: {note}.")
    lines += ["", f"Assumptions: {be.get('assumptions')}.", ""]
    return lines


def _live_bench(report: Mapping[str, Any]) -> list[str]:
    bench = report.get("live_bench") or {}
    runs: list[Mapping[str, Any]] = [bench[mode] for mode in ("student_only", "cascade") if bench.get(mode)]
    if not runs:
        return []
    rows = []
    for run in runs:
        latency = run.get("latency_ms") or {}
        mode = str(run.get("mode")).replace("_", "-")
        if run.get("matches_selected_run") is False:
            mode += f" (run `{run.get('run_id')}`, not the selected run)"
        spend = (
            usd(run.get("spend_usd"))
            if run.get("spend_usd") is not None or not run.get("spend_note")
            else "not measured"
        )
        rows.append(
            [
                mode,
                run.get("date") or DASH,
                integer(run.get("n")),
                integer(run.get("warmup")),
                ms(latency.get("p50")),
                ms(latency.get("p95")),
                pct(run.get("escalation_rate")),
                spend,
                run.get("teacher") or DASH,
            ]
        )
    lines = ["## Live bench", ""]
    header = ["Mode", "Date", "n", "Warm-up", "p50 ms", "p95 ms", "Escalated", "Spend", "Teacher"]
    lines += [*table(header, rows), ""]
    lines += [
        "Measured end to end through `taskdistill serve`, one request at a time, warm-up requests excluded. These are "
        "dated measurements and are not expected to reproduce exactly.",
        "",
    ]
    for run in runs:
        text = f"{hardware_text(run.get('hardware'))}"
        load = _load_text(run.get("machine_state"))
        if load:
            text += f", {load} before timing"
        if run.get("command"):
            text += f"; `{run['command']}`"
        lines.append(f"- {str(run.get('mode')).replace('_', '-')} (`{run.get('file')}`): {text}.")
    lines.append("")
    check = (bench.get("cross_check") or {}).get("cascade")
    if check:
        lines += [
            f"Cross-check: composed cascade p50/p95 {ms(check.get('composed_p50'))}/{ms(check.get('composed_p95'))} "
            f"ms against measured {ms(check.get('measured_p50'))}/{ms(check.get('measured_p95'))} ms; escalation "
            f"{pct(check.get('test_escalation_rate'))} on test against {pct(check.get('measured_escalation_rate'))} "
            "measured.",
            "",
        ]
    for note in bench.get("notes") or []:
        lines += [f"Note: {note}.", ""]
    for run in runs:
        if run.get("spend_note"):
            lines += [
                f"Spend of the {str(run.get('mode')).replace('_', '-')} bench not measured: {run['spend_note']}.",
                "",
            ]
    return lines


def _serve_log(report: Mapping[str, Any]) -> list[str]:
    log = report.get("serve_log")
    if not log:
        return []
    window = log.get("since_seconds")
    title = "## Served traffic" + (f" (last {window / 3600:g} h)" if _num(window) is not None else "")
    lines = [title, ""]
    rows = [
        ["Requests", integer(log.get("requests"))],
        [
            "Route shares",
            ", ".join(f"{route} {pct(share)}" for route, share in (log.get("route_shares") or {}).items()) or DASH,
        ],
        ["Observed escalation rate", pct(log.get("escalation_rate"))],
        ["Failed escalations (teacher error or replay miss)", integer(log.get("failed_escalations"))],
        ["Low-confidence escalation rate", pct(log.get("confidence_escalation_rate"))],
        ["Latency p50 / p95 ms", f"{ms((log.get('latency_ms') or {}).get('p50'))} / "
         f"{ms((log.get('latency_ms') or {}).get('p95'))}"],
        ["Teacher spend", usd(log.get("teacher_spend_usd"))],
        ["$ / 1k requests", usd(log.get("usd_per_1k"))],
    ]  # fmt: skip
    for route, latency in (log.get("latency_ms_by_route") or {}).items():
        rows.append([f"Latency p50 / p95 ms, {route}", f"{ms(latency.get('p50'))} / {ms(latency.get('p95'))}"])
    lines += [*table(["", "Value"], rows), ""]
    drift = log.get("drift") or {}
    if drift.get("note"):
        lines += [f"Drift: {drift['note']}.", ""]
    lines += [f"Assumptions: {log.get('assumptions')}.", ""]
    return lines


def _training(report: Mapping[str, Any]) -> list[str]:
    runs = report.get("training") or []
    if not runs:
        return []
    rows = [
        [
            f"`{r.get('run_id')}`",
            r.get("base_model") or DASH,
            r.get("labels") or DASH,
            integer(r.get("examples")),
            integer(r.get("dropped_by_length_filter")),
            integer(r.get("iterations")),
            plain(r.get("epochs"), 2),
            plain(r.get("wall_minutes"), 1),
            plain(r.get("peak_memory_gb"), 2),
            integer(r.get("tokens_per_second")),
            plain(r.get("adapter_mb"), 1),
        ]
        for r in runs
    ]
    header = [
        "Run", "Base", "Labels", "Examples", "Dropped (length)", "Iterations", "Epochs", "Wall min", "Peak GB",
        "Tokens/s", "Adapter MB",
    ]  # fmt: skip
    return ["## Training", "", *table(header, rows), ""]


def _flatten(stats: Mapping[str, Any], prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key, value in stats.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and depth < 1:
            items += _flatten(value, f"{name}.", depth + 1)
        elif _is_leaf(value):
            items.append((name, value))
    return items


def _dataset(report: Mapping[str, Any]) -> list[str]:
    dataset = report.get("dataset") or {}
    splits = dataset.get("splits") or {}
    if not any(v is not None for v in splits.values()):
        return []
    lines = ["## Dataset", ""]
    lines += [*table(["Split", "Examples"], [[s, integer(n)] for s, n in splits.items()]), ""]
    stats = dataset.get("curate_stats")
    if isinstance(stats, Mapping) and stats:
        numbers = [(k, v) for k, v in _flatten(stats) if not isinstance(v, str) or len(v) < 80]
        if numbers:
            rows = [[k, _leaf(k.rsplit(".", 1)[-1], v, percent=False)] for k, v in numbers]
            lines += ["<details>", "<summary>Curation statistics</summary>", "", *table(["Key", "Value"], rows)]
            lines += ["", "</details>", ""]
    if dataset.get("dataset_card"):
        lines += ["The dataset card is written next to the curated data (`dataset_card.md`).", ""]
    return lines


def _spend(report: Mapping[str, Any]) -> list[str]:
    spend = report.get("spend") or {}
    if not spend.get("ledger"):
        return ["## Spend", "", "No ledger in this workspace: no teacher API spend recorded.", ""]
    rows = [[phase, usd(value)] for phase, value in sorted((spend.get("task_by_phase") or {}).items())]
    rows.append(["total (this task)", usd(spend.get("task_total"))])
    rows.append(["total (workspace)", usd(spend.get("workspace_total"))])
    return ["## Spend", "", *table(["Phase", "USD"], rows), ""]


def _notes(report: Mapping[str, Any]) -> list[str]:
    notes = list(report.get("notes") or [])
    for section in ("quality", "operating_point", "cost_latency"):
        notes += list((report.get(section) or {}).get("notes") or [])
    if not notes:
        return []
    return ["## Notes", "", *[f"- {note}" for note in dict.fromkeys(notes)], ""]


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as Markdown: quality, operating point, cost/latency, break-even, bench, serve log, training."""
    selected = report.get("selected_run") or {}
    teacher = report.get("teacher") or {}
    lines = [
        f"# taskdistill report: {report.get('task')}",
        "",
        f"- Report built: {report.get('date')} on {hardware_text(report.get('hardware'))}",
        f"- Command: `{report.get('command')}`",
        "- Each measured number carries the date and hardware of the evaluation or bench that produced it.",
        f"- Teacher: `{teacher.get('model')}`" + (f" via {teacher['provider']}" if teacher.get("provider") else ""),
        f"- Selected run: `{selected.get('run_id')}`" + (f" ({selected['reason']})" if selected.get("reason") else ""),
        "",
    ]
    for section in (
        _quality,
        _operating_point,
        _cost_latency,
        _break_even,
        _live_bench,
        _serve_log,
        _training,
        _dataset,
        _spend,
        _notes,
    ):
        lines += section(report)
    return "\n".join(lines).rstrip() + "\n"
