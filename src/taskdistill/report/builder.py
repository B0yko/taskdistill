"""``taskdistill report``: quality table, operating point, cost and latency, break-even and live figures.

Everything is read from the workspace: ``eval/*/eval_test.json``, ``selected_run.json``, the selected run's test
predictions cache and ``train_log.json``, the curated data, the response cache and the teacher recording (recorded
teacher cost and latency), bench JSONs, the ledger and, with ``--from-serve-log``, the served-request log. The report
is written as ``report.json`` and ``report.md`` (plus copies of the evaluation PNGs); neither file carries an absolute
path.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import shutil
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from taskdistill import paths
from taskdistill.config import CostSpec, TaskSpec
from taskdistill.curate.io import raw_inputs_by_hash
from taskdistill.evaluate.bootstrap import diff_name
from taskdistill.hardware import hardware_info
from taskdistill.ledger import Ledger
from taskdistill.report.render import render_markdown
from taskdistill.store import Store
from taskdistill.teacher.cache import ResponseCache, request_context
from taskdistill.teacher.client import usage_cost
from taskdistill.teacher.factory import find_recording, load_pricing
from taskdistill.teacher.pricing import ModelPrice, PricingError
from taskdistill.teacher.replay import Recording, RecordingError, load_recording
from taskdistill.teacher.request_key import request_key
from taskdistill.teacher.requests import build_teacher_request
from taskdistill.train.common import short_model_name

#: Ledger phases whose spend is teacher labelling (curate, the demo's captured traffic); any other phase whose
#: name contains "label" or "capture" counts too.
LABELLING_PHASES = ("curate-label", "demo-capture", "label", "labelling", "capture")
#: Ledger phase of the cascade server's teacher calls (served traffic and the live bench).
SERVE_PHASE = "serve"
#: The live teacher client's default concurrency, stated next to recorded teacher latency.
DEFAULT_TEACHER_CONCURRENCY = 8
COLUMNS: dict[str, tuple[str, ...]] = {
    "classification": ("accuracy", "macro_f1", "agreement"),
    "extraction": ("json_validity", "field_micro_f1", "field_exact_match", "doc_exact_match", "agreement"),
}
CALIBRATION_COLUMNS = ("ece", "auroc")
GOLD_METRIC = {"classification": "accuracy", "extraction": "field_micro_f1"}
#: ``cascade.metric`` names that differ from the metric keys of an eval JSON (against gold).
METRIC_KEYS = {"field_f1": "field_micro_f1"}
#: ``cascade.metric`` against the teacher, as eval metric keys: label accuracy and field micro-F1 against the teacher's
#: output are what the eval reports as ``agreement``; macro-F1 against the teacher has no eval key.
TEACHER_METRIC_KEYS = {"agreement": "agreement", "accuracy": "agreement", "field_f1": "agreement"}
_Z95 = 1.959963984540054


class ReportError(RuntimeError):
    """The workspace holds nothing a report can be built from."""


# -- small helpers ---------------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """A finite float, or None for anything else (booleans included)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _task_dir(task: str) -> Path:
    """``$TASKDISTILL_HOME/<task>`` without creating it: building a report only reads the workspace."""
    return paths.home() / task


def latency_summary(values: Iterable[float | None]) -> dict[str, Any]:
    """p50/p95/mean in ms (numpy's linear percentiles) and the count; None fields when there is no value."""
    clean = [v for v in (_num(x) for x in values) if v is not None]
    if not clean:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "n": len(clean),
    }


def local_cost_usd(latency_ms: float, cost: CostSpec) -> float:
    """Local cost of ``latency_ms`` of compute: energy, plus hardware amortisation only when it is enabled.

    Energy = ``local_watts`` x wall time x ``usd_per_kwh``; amortisation = ``hardware_usd / amortisation_hours``
    per hour of wall time.
    """
    hours = max(0.0, latency_ms) / 3_600_000.0
    usd = cost.local_watts / 1000.0 * hours * cost.usd_per_kwh
    if cost.amortisation_enabled:
        usd += cost.hardware_usd / cost.amortisation_hours * hours
    return usd


def cost_assumptions(cost: CostSpec) -> dict[str, Any]:
    """The local-cost assumptions, as numbers and as the sentence printed next to every local cost."""
    text = f"local cost = {cost.local_watts:g} W x wall time x ${cost.usd_per_kwh:g}/kWh"
    if cost.amortisation_enabled:
        text += (
            f" + hardware amortisation ${cost.hardware_usd:g} / {cost.amortisation_hours:g} h "
            f"(${cost.hardware_usd / cost.amortisation_hours:.4g} per hour of wall time)"
        )
    else:
        text += "; hardware amortisation off"
    return {
        "local_watts": cost.local_watts,
        "usd_per_kwh": cost.usd_per_kwh,
        "hardware_usd": cost.hardware_usd,
        "amortisation_hours": cost.amortisation_hours,
        "amortisation_enabled": cost.amortisation_enabled,
        "text": text,
    }


def wilson_interval(successes: int, n: int) -> tuple[float, float] | None:
    """95% Wilson score interval of a proportion, or None when ``n`` is 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + _Z95**2 / n
    centre = (p + _Z95**2 / (2 * n)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / n + _Z95**2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


_PATH_ROOTS = frozenset({"Users", "home", "tmp", "private", "var", "Volumes", "opt", "root", "mnt"})
#: An absolute path inside a longer string: ``/`` at the start or after whitespace, ``=``, ``:``, a quote or bracket.
_EMBEDDED_PATH = re.compile(r"(?<![^\s=:'\"(\[,])/[^\s'\"()\[\],;]+")


def _looks_like_path(value: str) -> bool:
    if len(value) < 2 or not value.startswith("/") or "\n" in value:
        return False
    first = value[1:].split("/", 1)[0]
    if first in _PATH_ROOTS:
        return True
    try:
        return Path(value).exists()
    except OSError:
        return False


def display_path(value: Path | str) -> str:
    """A path as a report prints it: relative to the current directory, else ``$TASKDISTILL_HOME/...``, else its name.

    Never absolute, so a committed report carries no user name or machine layout.
    """
    path = Path(value)
    with contextlib.suppress(OSError, RuntimeError):
        path = path.resolve()
    for base, prefix in ((Path.cwd(), "."), (paths.home(), "$TASKDISTILL_HOME")):
        with contextlib.suppress(OSError, RuntimeError):
            base = base.resolve()
        try:
            rel = path.relative_to(base)
        except ValueError:
            continue
        text = rel.as_posix()
        if prefix == ".":
            return text if text != "." else "."
        return prefix if text == "." else f"{prefix}/{text}"
    return path.name or "."


def sanitise_command(command: str) -> str:
    """A command line with every absolute path argument (``/x``, ``--opt=/x``, quoted) made relative."""
    words = []
    for word in command.split():
        option, sep, value = word.partition("=") if word.startswith("-") else ("", "", word)
        quote = value[:1] if value[:1] in ("'", '"') else ""
        inner = value[len(quote) :].rstrip(quote) if quote else value
        if len(inner) > 1 and inner.startswith("/"):
            value = f"{quote}{display_path(inner)}{quote}"
        words.append(f"{option}{sep}{value}")
    return " ".join(words)


def _scrub_paths(text: str) -> str:
    """Absolute paths embedded in a longer string (a note, an eval's command) made relative."""

    def replace(match: re.Match[str]) -> str:
        found = match.group(0)
        return display_path(found) if _looks_like_path(found) else found

    return _EMBEDDED_PATH.sub(replace, text)


def _relative_paths(value: Any) -> Any:
    """Absolute path strings become workspace- or CWD-relative, so a report never carries one."""
    if isinstance(value, str):
        if _looks_like_path(value):
            return paths.relative_to_home(value)
        return _scrub_paths(value) if "/" in value else value
    if isinstance(value, dict):
        return {str(k): _relative_paths(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_relative_paths(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# -- evaluation results ----------------------------------------------------------------------------


def _load_evals(task: str) -> dict[str, dict[str, Any]]:
    evals: dict[str, dict[str, Any]] = {}
    for path in sorted((_task_dir(task) / "eval").glob("*/eval_test.json")):
        data = _read_json(path)
        if isinstance(data, dict):
            evals[str(data.get("run_id") or path.parent.name)] = data
    return evals


def _run(ev: Mapping[str, Any]) -> dict[str, Any]:
    return _dict(ev.get("run"))


def _labels(ev: Mapping[str, Any]) -> str:
    return str(_run(ev).get("labels") or "teacher")


def _system(ev: Mapping[str, Any], name: str) -> dict[str, Any]:
    return _dict(_dict(ev.get("systems")).get(name))


def _metrics(ev: Mapping[str, Any], system: str, columns: Sequence[str]) -> dict[str, float | None]:
    metrics = _dict(_system(ev, system).get("metrics"))
    return {column: _num(metrics.get(column)) for column in columns}


def _calibration(ev: Mapping[str, Any]) -> dict[str, Any]:
    """Raw-confidence ECE/AUROC of the student: against gold when gold exists, else against the teacher."""
    calibration = _dict(_system(ev, "student").get("calibration"))
    raw = _dict(calibration.get("raw"))
    reference = "gold" if raw.get("vs_gold") else "teacher"
    report = _dict(raw.get(f"vs_{reference}"))
    isotonic = _dict(_dict(calibration.get("isotonic")).get(f"vs_{reference}"))
    return {
        "reference": reference if report else None,
        "ece": _num(report.get("ece")),
        "auroc": _num(report.get("auroc")),
        "brier": _num(report.get("brier")),
        "isotonic_ece": _num(isotonic.get("ece")),
    }


def _bootstrap(ev: Mapping[str, Any], kind: str) -> dict[str, Any]:
    return _dict(_dict(ev.get("bootstrap")).get(kind))


def _interval(entry: Any) -> list[float] | None:
    entry = _dict(entry)
    lo, hi = _num(entry.get("lo")), _num(entry.get("hi"))
    return [lo, hi] if lo is not None and hi is not None else None


def _cis(ev: Mapping[str, Any], system: str, columns: Sequence[str], kind: str = "paired") -> dict[str, Any] | None:
    stats = _dict(_bootstrap(ev, kind).get("stats"))
    if not stats:
        return None
    return {column: _interval(stats.get(f"{system}.{column}")) for column in columns}


def _model_name(base: Any) -> str:
    return short_model_name(str(base)) if base else "unknown"


def _base_row(system: str, name: str, ev: Mapping[str, Any], run_ids: list[str]) -> dict[str, Any]:
    run = _run(ev)
    return {
        "system": system,
        "name": name,
        "base_model": run.get("base_model"),
        "labels": run.get("labels"),
        "run_ids": run_ids,
        "n": ev.get("n"),
    }


def _mean_std(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if len(clean) >= 2 else None
    return mean, std


def _student_metrics(ev: Mapping[str, Any], columns: Sequence[str]) -> dict[str, float | None]:
    calibration = _calibration(ev)
    return {**_metrics(ev, "student", columns), "ece": calibration["ece"], "auroc": calibration["auroc"]}


def _student_group_row(
    key: tuple[str, str, str, str],
    runs: list[tuple[str, dict[str, Any]]],
    columns: Sequence[str],
    selected: str | None,
    candidates: Mapping[str, Any],
) -> dict[str, Any]:
    base, labels, profile, backend = key
    all_columns = (*columns, *CALIBRATION_COLUMNS)
    runs = sorted(runs, key=lambda item: (_num(_run(item[1]).get("seed")) or 0.0, item[0]))
    seed_metrics = [_student_metrics(ev, columns) for _, ev in runs]
    per_seed = [
        {"run_id": rid, "seed": _run(ev).get("seed"), "n": ev.get("n"), "metrics": metrics}
        for (rid, ev), metrics in zip(runs, seed_metrics, strict=True)
    ]
    ids = [rid for rid, _ in runs]
    if selected in ids:
        chosen = selected
    else:
        scored = [(_num(_dict(candidates.get(rid)).get("valid_metric")), rid) for rid in ids]
        ranked = sorted(((s, rid) for s, rid in scored if s is not None), key=lambda x: (-x[0], x[1]))
        chosen = ranked[0][1] if ranked else ids[0]
    chosen_ev = dict(runs)[chosen]
    mean: dict[str, float | None] = {}
    std: dict[str, float | None] = {}
    for column in all_columns:
        mean[column], std[column] = _mean_std([metrics[column] for metrics in seed_metrics])
    label_text = {"teacher": "teacher labels", "gold": "gold labels"}.get(labels, f"{labels} labels")
    extras = [label_text]
    if profile and profile != "full":
        extras.append(f"{profile} profile")
    if backend and backend != "mlx":
        extras.append(backend)
    if len(runs) > 1:
        extras.append(f"{len(runs)} seeds")
    name = f"student {_model_name(base)} ({', '.join(extras)})"
    row = _base_row("student", name, chosen_ev, ids)
    row.update(
        {
            "profile": profile or None,
            "backend": backend or None,
            "n_seeds": len(runs),
            "metrics": mean,
            "std": std,
            "per_seed": per_seed,
            "selected_run": chosen,
            "selected_seed": _run(chosen_ev).get("seed"),
            "selected_metrics": _student_metrics(chosen_ev, columns),
            "ci": _cis(chosen_ev, "student", all_columns),
            "ci_cluster": _cis(chosen_ev, "student", all_columns, "cluster"),
            "calibration": _calibration(chosen_ev),
        }
    )
    return row


def build_quality(
    spec: TaskSpec, evals: Mapping[str, dict[str, Any]], selected: str | None, candidates: Mapping[str, Any]
) -> dict[str, Any]:
    """Quality rows: teacher, zero-shot, TF-IDF, student groups (seeds aggregated) and the cascade."""
    columns = COLUMNS[spec.type]
    student_columns = (*columns, *CALIBRATION_COLUMNS)
    sel_ev = evals.get(selected) if selected else None
    rows: list[dict[str, Any]] = []
    notes: list[str] = []

    teacher_ev = sel_ev
    if teacher_ev is None:
        teacher_ev = next((ev for _, ev in sorted(evals.items()) if _labels(ev) != "zero-shot"), None)
        if teacher_ev is not None:
            notes.append("teacher row taken from another run's evaluation: the selected run has no test evaluation")
    if teacher_ev is not None:
        metrics = _metrics(teacher_ev, "teacher", columns)
        metrics["agreement"] = None  # the reference itself
        row = _base_row("teacher", f"teacher ({spec.teacher.model})", teacher_ev, [str(teacher_ev.get("run_id"))])
        row.update(
            {
                "base_model": spec.teacher.model,
                "labels": None,
                "metrics": metrics,
                "ci": _cis(teacher_ev, "teacher", columns),
                "ci_cluster": _cis(teacher_ev, "teacher", columns, "cluster"),
                "recorded": True,
                "has_gold": bool(_dict(_system(teacher_ev, "teacher").get("metrics"))),
            }
        )
        rows.append(row)

    for rid, ev in sorted(evals.items()):
        if _labels(ev) != "zero-shot":
            continue
        row = _base_row("zero-shot", f"zero-shot {_model_name(_run(ev).get('base_model'))}", ev, [rid])
        row.update(
            {
                "metrics": _student_metrics(ev, columns),
                "ci": _cis(ev, "student", student_columns),
                "ci_cluster": _cis(ev, "student", student_columns, "cluster"),
                "calibration": _calibration(ev),
            }
        )
        rows.append(row)

    if spec.type == "classification":
        tfidf_ev = sel_ev if sel_ev is not None and _system(sel_ev, "tfidf") else None
        if tfidf_ev is None:
            tfidf_ev = next((ev for _, ev in sorted(evals.items()) if _system(ev, "tfidf")), None)
        if tfidf_ev is not None:
            row = _base_row(
                "tfidf", "TF-IDF + logistic regression (teacher labels)", tfidf_ev, [str(tfidf_ev.get("run_id"))]
            )
            row.update(
                {
                    "base_model": None,
                    "labels": "teacher",
                    "metrics": _metrics(tfidf_ev, "tfidf", columns),
                    "ci": _cis(tfidf_ev, "tfidf", columns),
                    "ci_cluster": _cis(tfidf_ev, "tfidf", columns, "cluster"),
                    "C": _num(_system(tfidf_ev, "tfidf").get("C")),
                }
            )
            rows.append(row)

    groups: dict[tuple[str, str, str, str], list[tuple[str, dict[str, Any]]]] = {}
    for rid, ev in evals.items():
        labels = _labels(ev)
        if labels == "zero-shot":
            continue
        run = _run(ev)
        key = (str(run.get("base_model") or ""), labels, str(run.get("profile") or ""), str(run.get("backend") or ""))
        groups.setdefault(key, []).append((rid, ev))
    order = {"teacher": 0, "gold": 1}
    for key in sorted(groups, key=lambda k: (order.get(k[1], 2), k[0], k[2], k[3])):
        rows.append(_student_group_row(key, groups[key], columns, selected, candidates))

    if sel_ev is not None and _system(sel_ev, "cascade"):
        cascade = _system(sel_ev, "cascade")
        threshold = _num(cascade.get("threshold"))
        where = "always escalate" if threshold is None else f"t = {threshold:.4g}"
        row = _base_row("cascade", f"cascade ({selected}, {where})", sel_ev, [str(selected)])
        row.update(
            {
                "metrics": _metrics(sel_ev, "cascade", columns),
                "ci": _cis(sel_ev, "cascade", columns),
                "ci_cluster": _cis(sel_ev, "cascade", columns, "cluster"),
                "threshold": threshold,
                "escalation_rate": _num(cascade.get("escalation_rate")),
                "recorded": True,
            }
        )
        rows.append(row)

    source = sel_ev or teacher_ev or {}
    used = dict.fromkeys(rid for row in rows for rid in row.get("run_ids") or [])
    provenance = [_eval_provenance(evals[rid], rid) for rid in used if rid in evals]
    return {
        "task_type": spec.type,
        "provenance": provenance,
        "split": "test",
        "n": source.get("n"),
        "columns": list(columns),
        "student_columns": list(student_columns),
        "rows": rows,
        "cluster_bootstrap": bool(_bootstrap(source, "cluster")),
        "n_groups": _bootstrap(source, "cluster").get("n_groups"),
        "per_group": source.get("per_group"),
        "per_field": source.get("per_field"),
        "per_trait": source.get("per_trait"),
        "confidence_comparison": source.get("confidence_comparison"),
        "test_scorings": source.get("test_scorings"),
        "profile_fast": source.get("profile_fast"),
        "notes": notes,
    }


# -- operating point -------------------------------------------------------------------------------


def _cascade_minus_teacher(ev: Mapping[str, Any], metric: str) -> dict[str, Any] | None:
    """``cascade.<metric> - teacher.<metric>`` with its paired-bootstrap CI (and the cluster CI when present)."""
    out: dict[str, Any] = {"metric": metric, "point": None, "ci": None, "ci_cluster": None, "source": None}
    for kind in ("paired", "cluster"):
        boot = _bootstrap(ev, kind)
        diffs, stats = _dict(boot.get("diffs")), _dict(boot.get("stats"))
        entry = _dict(diffs.get(diff_name(f"cascade.{metric}", f"teacher.{metric}")))
        ci_key = "ci" if kind == "paired" else "ci_cluster"
        if entry:
            if kind == "paired":
                out["point"], out["source"] = _num(entry.get("point")), "bootstrap"
            out[ci_key] = _interval(entry)
        elif metric == "agreement" and stats.get("cascade.agreement"):
            # The teacher agrees with itself on every resample, so the difference is cascade agreement - 1.
            stat = _dict(stats["cascade.agreement"])
            point = _num(stat.get("point"))
            ci = _interval(stat)
            if kind == "paired":
                out["point"] = None if point is None else point - 1.0
                out["source"] = "bootstrap (teacher agreement is 1 by definition)"
            out[ci_key] = None if ci is None else [ci[0] - 1.0, ci[1] - 1.0]
    if out["point"] is None:
        cascade = _num(_dict(_system(ev, "cascade").get("metrics")).get(metric))
        teacher = 1.0 if metric == "agreement" else _num(_dict(_system(ev, "teacher").get("metrics")).get(metric))
        if cascade is None or teacher is None:
            return None
        out["point"], out["source"] = cascade - teacher, "point estimate (no bootstrap difference in the eval)"
    return out


def target_metric_key(reference: str, metric: str) -> str | None:
    """The eval metric key that scores ``metric`` against ``reference``; None when the eval has none."""
    if reference == "teacher":
        return TEACHER_METRIC_KEYS.get(metric)
    return METRIC_KEYS.get(metric, metric)


def _operating_point_difference(test: Mapping[str, Any], metric: str, reference: str) -> dict[str, Any] | None:
    """Cascade minus teacher from the test operating point (quality against the reference), without an interval."""
    quality = _num(test.get("quality"))
    teacher = _num(test.get("teacher_quality"))
    if quality is None:
        return None
    if teacher is None:
        teacher = 1.0
    return {
        "metric": metric,
        "point": quality - teacher,
        "ci": None,
        "ci_cluster": None,
        "source": f"test operating point ({metric} against the {reference}); the eval has no bootstrap difference",
    }


def _find_pngs(task: str, run_id: str | None, ev: Mapping[str, Any] | None) -> dict[str, list[Path]]:
    """The threshold-curve and reliability PNGs the evaluation lists, else those found next to it."""
    files = _dict(ev.get("files")) if ev is not None else {}
    listed = {
        kind: paths.home() / str(files[key])
        for kind, key in (("curve", "threshold_curve"), ("reliability", "reliability"))
        if files.get(key)
    }
    if listed and all(path.is_file() for path in listed.values()):
        return {kind: [listed[kind]] if kind in listed else [] for kind in ("curve", "reliability")}
    places = [_task_dir(task) / "eval" / run_id] if run_id else []
    found: dict[str, list[Path]] = {"curve": [], "reliability": []}
    for place in [*places, _task_dir(task)]:
        for png in sorted(place.glob("*.png")) if place.is_dir() else []:
            name = png.name.lower()
            if "threshold" in name or "curve" in name:
                found["curve"].append(png)
            elif "reliab" in name:
                found["reliability"].append(png)
        if found["curve"] or found["reliability"]:
            break
    return found


def build_operating_point(
    spec: TaskSpec, ev: Mapping[str, Any] | None, selected: str | None, out_dir: Path
) -> dict[str, Any]:
    """Target, threshold, escalation rates, cascade-minus-teacher differences and the copied PNGs."""
    op: dict[str, Any] = {"run_id": selected, "available": ev is not None}
    threshold = _dict(ev.get("threshold")) if ev is not None else {}
    if not threshold:
        stored = _read_json(_task_dir(spec.task) / "threshold.json")
        if isinstance(stored, dict) and (stored.get("run_id") in (None, selected)):
            threshold = stored
    metric = str(threshold.get("metric") or spec.cascade.metric)
    reference = str(threshold.get("reference") or spec.cascade.reference)
    metric_key = target_metric_key(reference, metric)
    points = _dict(ev.get("operating_point")) if ev is not None else {}
    valid, test = _dict(points.get("valid")), _dict(points.get("test"))
    t_value = threshold.get("threshold") if threshold else None
    op.update(
        {
            "reference": reference,
            "metric": metric,
            "metric_key": metric_key,
            "target": threshold.get("target", spec.cascade.target),
            "max_drop": threshold.get("max_drop", spec.cascade.max_drop),
            "target_value": _num(threshold.get("target_value")),
            "threshold": _num(t_value),
            "always_escalate": bool(threshold) and t_value is None,
            "warning": threshold.get("warning"),
            "escalation_rate": {
                "valid": _num(valid.get("escalation_rate", threshold.get("escalation_rate"))),
                "test": _num(test.get("escalation_rate")),
            },
            "quality": {
                "valid": _num(valid.get("quality", threshold.get("quality"))),
                "test": _num(test.get("quality")),
            },
            "target_met_on_test": test.get("met") if isinstance(test.get("met"), bool) else None,
            "cascade_minus_teacher": {"target_metric": None, "gold_metric": None},
            "provenance": _eval_provenance(ev, selected),
            "curve_png": None,
            "reliability_png": [],
            "notes": [],
        }
    )
    if ev is None:
        op["notes"].append("no test evaluation of the selected run: run `taskdistill eval --task <task>`")
    else:
        target_diff = _cascade_minus_teacher(ev, metric_key) if metric_key is not None else None
        if target_diff is None:
            target_diff = _operating_point_difference(test, metric, reference)
            if target_diff is not None:
                op["notes"].append(
                    f"cascade minus teacher on {metric} against the {reference} comes from the test operating point, "
                    "without a confidence interval: the eval has no bootstrap difference for it"
                )
        if target_diff is not None:
            target_diff["reference"] = reference
            target_diff["target_metric"] = metric
        op["cascade_minus_teacher"]["target_metric"] = target_diff
        gold_metric = GOLD_METRIC[spec.type]
        if _dict(_system(ev, "teacher").get("metrics")):
            op["cascade_minus_teacher"]["gold_metric"] = _cascade_minus_teacher(ev, gold_metric)
        else:
            op["notes"].append(f"no gold labels on the test split: cascade minus teacher on {gold_metric} is undefined")
    pngs = _find_pngs(spec.task, selected, ev)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, files in pngs.items():
        for png in files:
            shutil.copyfile(png, out_dir / png.name)
        names = [png.name for png in files]
        if kind == "curve":
            op["curve_png"] = names[0] if names else None
        else:
            op["reliability_png"] = names
    return op


# -- teacher records, predictions and cost/latency -------------------------------------------------


@dataclass
class TeacherRecord:
    """Recorded cost and latency of one teacher request (from the response cache or the recording)."""

    cost_usd: float | None
    latency_ms: float | None
    source: str
    created: float | None = None


class TeacherRecords:
    """Recorded teacher responses by request key: the workspace response cache first, then the teacher recording.

    A response without ``usage.cost`` is priced from its token counts with the pricing snapshot.
    """

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        self.cache = ResponseCache() if paths.cache_path().is_file() else None
        self.recording: Recording | None = None
        source = find_recording(spec.task)
        if source is not None:
            with contextlib.suppress(RecordingError):
                self.recording = load_recording(source)
        self.price: ModelPrice | None = None
        self.pricing_date: str | None = None
        try:
            snapshot = load_pricing()
            self.price = snapshot.price_for(spec.teacher.model, spec.teacher.provider)
            self.pricing_date = snapshot.date
        except (PricingError, KeyError, OSError, ValueError):
            self.price = None
        self.cost_sources: Counter[str] = Counter()
        self._memo: dict[str, TeacherRecord | None] = {}

    def _cost(self, usage: Mapping[str, Any]) -> float | None:
        cost = _num(usage.get("cost"))
        if cost is not None and cost >= 0:
            self.cost_sources["usage.cost"] += 1
            return cost
        priced = usage_cost(usage, self.price) if self.price is not None else None
        if priced is not None:
            self.cost_sources["pricing snapshot"] += 1
        return priced

    def get(self, raw_input: str) -> TeacherRecord | None:
        """The recorded teacher response to the application's request for ``raw_input``, or None."""
        if raw_input in self._memo:
            return self._memo[raw_input]
        found = self._lookup(raw_input)
        self._memo[raw_input] = found
        return found

    def _lookup(self, raw_input: str) -> TeacherRecord | None:
        body = build_teacher_request(self.spec, raw_input)
        key = request_key(body)
        if self.cache is not None:
            result = self.cache.get(key, request_context(body))
            if result is not None:
                return TeacherRecord(
                    self._cost(result.usage or {}), _num(result.latency_ms), "cache", _num(result.created)
                )
        if self.recording is not None:
            record = self.recording.get(key)
            if record is not None:
                usage = _dict(record.get("usage"))
                return TeacherRecord(
                    self._cost(usage), _num(record.get("latency_ms")), "recording", _num(record.get("timestamp"))
                )
        return None


def _split_hashes(task: str, split: str) -> list[str]:
    return [str(row.get("input_hash") or "") for row in _read_jsonl(_task_dir(task) / "data" / f"{split}.meta.jsonl")]


def _predictions(task: str, run_id: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Header and rows of ``runs/<run>/preds_test.jsonl`` (empty when absent)."""
    if not run_id:
        return {}, []
    rows = _read_jsonl(_task_dir(task) / "runs" / run_id / "preds_test.jsonl")
    header = rows[0] if rows and rows[0].get("type") == "header" else {}
    return header, [row for row in rows if row.get("type") != "header"]


def _confidence_field(ev: Mapping[str, Any] | None) -> str:
    comparison = _dict(_dict(ev.get("confidence_comparison")).get("valid")) if ev else {}
    return "alt_confidence" if comparison.get("chosen") == "alternative" else "confidence"


def escalated(confidence: float | None, threshold: float | None) -> bool:
    """The cascade escalates when the threshold is always-escalate (None), confidence is missing, or below it."""
    if threshold is None or confidence is None:
        return True
    return not confidence >= threshold


def _bench_matches(bench: Mapping[str, Any], selected: str | None) -> bool:
    run_id = bench.get("run_id")
    return not run_id or not selected or run_id == selected


def _latest_benches(task: str, selected: str | None) -> dict[str, tuple[str, dict[str, Any]]]:
    """The newest bench JSON per mode (``student_only``, ``cascade``) with its file name.

    Benches of the selected run (or of an unknown run) win over newer benches of another run.
    """
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    directory = _task_dir(task) / "bench"
    if not directory.is_dir():
        return latest
    found = []
    for path in directory.glob("*.json"):
        data = _read_json(path)
        if isinstance(data, dict) and data.get("mode") in ("student_only", "cascade"):
            found.append((_bench_matches(data, selected), str(data.get("date") or ""), path.name, data))
    for _, _, name, data in sorted(found, key=lambda item: (item[0], item[1], item[2])):
        latest[str(data["mode"])] = (name, data)
    return latest


def _bench_summary(name: str, bench: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file": name,
        "date": bench.get("date"),
        "mode": bench.get("mode"),
        "run_id": bench.get("run_id"),
        "threshold": bench.get("threshold"),
        "teacher": bench.get("teacher"),
        "n": bench.get("n"),
        "warmup": bench.get("warmup"),
        "concurrency": bench.get("concurrency", 1),
        "latency_ms": bench.get("latency_ms"),
        "escalation_rate": bench.get("escalation_rate"),
        "route_counts": bench.get("route_counts"),
        "reasons": bench.get("reasons"),
        "spend_usd": bench.get("spend_usd"),
        "spend_note": bench.get("spend_note"),
        "errors": bench.get("errors"),
        "failed_escalations": bench.get("failed_escalations"),
        "stopped": bench.get("stopped"),
        "machine_state": bench.get("machine_state"),
        "hardware": bench.get("hardware"),
        "command": sanitise_command(str(bench["command"])) if bench.get("command") else None,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _iso_date(ts: float | None) -> str | None:
    if ts is None or ts <= 0:
        return None
    with contextlib.suppress(OverflowError, OSError, ValueError):
        return datetime.fromtimestamp(ts, UTC).date().isoformat()
    return None


def _eval_provenance(ev: Mapping[str, Any] | None, run_id: str | None) -> dict[str, Any] | None:
    """Where an evaluation's numbers come from: its file, date, command and hardware."""
    if ev is None:
        return None
    rid = str(ev.get("run_id") or run_id or "")
    return {
        "source": "eval",
        "run_id": rid or None,
        "file": f"eval/{rid}/eval_test.json" if rid else None,
        "date": ev.get("date"),
        "command": sanitise_command(str(ev["command"])) if ev.get("command") else None,
        "hardware": ev.get("hardware"),
    }


def _bench_provenance(name: str, bench: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": "bench",
        "file": f"bench/{name}",
        "date": bench.get("date"),
        "command": sanitise_command(str(bench["command"])) if bench.get("command") else None,
        "hardware": bench.get("hardware"),
        "machine_state": bench.get("machine_state"),
    }


def build_cost_latency(
    spec: TaskSpec,
    ev: Mapping[str, Any] | None,
    selected: str | None,
    *,
    raw_inputs: Mapping[str, str],
    records: TeacherRecords,
    benches: Mapping[str, tuple[str, dict[str, Any]]],
    curate_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Teacher-only, student-only and cascade: $/1k requests and latency p50/p95, with their sources.

    The cascade's escalation rate and cost cover every test request with a prediction of the selected run; its
    latency is composed per request wherever a per-request student latency exists (every prediction for eval
    latencies, the benched requests for bench latencies). An escalated request without a recorded teacher cost or
    latency gets the mean recorded one, and the number of such imputations is reported.
    """
    cost = spec.cost
    notes: list[str] = []
    hashes = [h for h in _split_hashes(spec.task, "test") if h]
    with_raw = [h for h in hashes if h in raw_inputs]
    if len(with_raw) < len(hashes):
        notes.append(f"{len(hashes) - len(with_raw)} of {len(hashes)} test inputs have no raw input in the store")
    teacher_by_hash: dict[str, TeacherRecord] = {}
    for h in with_raw:
        record = records.get(raw_inputs[h])
        if record is not None:
            teacher_by_hash[h] = record
    missing = len(with_raw) - len(teacher_by_hash)
    if missing:
        notes.append(f"{missing} test requests have no recorded teacher response (cache or recording)")
    teacher_costs = [r.cost_usd for r in teacher_by_hash.values() if r.cost_usd is not None]
    teacher_latencies = [r.latency_ms for r in teacher_by_hash.values() if r.latency_ms is not None]
    teacher_cost = _mean(teacher_costs)
    teacher_latency_mean = _mean(teacher_latencies)
    concurrency = _labelling_concurrency(curate_stats)
    dates = sorted(d for d in (_iso_date(r.created) for r in teacher_by_hash.values()) if d)
    teacher_row = {
        "usd_per_request": teacher_cost,
        "usd_per_1k": None if teacher_cost is None else teacher_cost * 1000,
        "latency_ms": latency_summary(teacher_latencies),
        "n": len(teacher_by_hash),
        "n_cost": len(teacher_costs),
        "cost_sources": dict(records.cost_sources),
        "pricing_snapshot_date": records.pricing_date,
        "record_sources": dict(Counter(r.source for r in teacher_by_hash.values())),
        "recorded_between": [dates[0], dates[-1]] if dates else None,
        "concurrency": concurrency["value"],
        "concurrency_source": concurrency["source"],
        "recorded": True,
        "source": "recorded live labelling calls on the test split (cache hits and retried attempts excluded)",
    }

    header, preds = _predictions(spec.task, selected)
    field = _confidence_field(ev)
    by_id = {str(p.get("id")): p for p in preds}
    aligned = {h: by_id[h] for h in hashes if h in by_id}
    if len(aligned) < len(hashes):
        notes.append(
            f"{len(hashes) - len(aligned)} of {len(hashes)} test inputs have no prediction of the selected run "
            f"(runs/{selected}/preds_test.jsonl); re-run eval"
        )

    threshold_dict = _dict(ev.get("threshold")) if ev else {}
    threshold = _num(threshold_dict.get("threshold")) if threshold_dict else None
    if not threshold_dict:
        notes.append("no chosen threshold for the selected run: the cascade is composed as always-escalate")

    bench = benches.get("student_only")
    if bench is not None and not _bench_matches(bench[1], selected):
        notes.append(f"bench {bench[0]} measured run {bench[1].get('run_id')}, not the selected run; ignored")
        bench = None
    student_per_request: dict[str, float] = {}
    eval_latency = _dict(_system(ev, "student").get("latency_ms")) if ev else {}
    eval_latency = {k: eval_latency.get(k) for k in ("p50", "p95", "mean", "n")}
    provenance: dict[str, Any] | None
    if bench is not None:
        student_source = "bench"
        name, data = bench
        student_latency = dict(_dict(data.get("latency_ms")))
        for req in data.get("requests") or []:
            req = _dict(req)
            latency = _num(req.get("latency_ms"))
            if req.get("route") == "student" and latency is not None and req.get("input_hash"):
                student_per_request[str(req["input_hash"])] = latency
        student_note = f"measured by `taskdistill bench` against `serve --threshold 0` ({name}): warm, concurrency 1"
        student_latency_n = student_latency.get("n", data.get("n"))
        provenance = _bench_provenance(name, data)
    else:
        student_source = "eval"
        for h, p in aligned.items():
            latency = _num(p.get("latency_ms"))
            if latency is not None:
                student_per_request[h] = latency
        if _num(eval_latency.get("p50")) is not None:
            student_latency = dict(eval_latency)
        else:
            student_latency = latency_summary(student_per_request.values())
        student_latency_n = student_latency.get("n")
        student_note = (
            "in-process generation latency from `taskdistill eval` (flagged: not end-to-end through `serve`; run "
            "`taskdistill bench` against `serve --threshold 0` for the measured figure)"
        )
        provenance = _eval_provenance(ev, selected)
    student_mean = _num(student_latency.get("mean"))
    if student_mean is None and student_per_request:
        student_mean = float(np.mean(list(student_per_request.values())))
    student_cost = None if student_mean is None else local_cost_usd(student_mean, cost)
    student_row = {
        "usd_per_request": student_cost,
        "usd_per_1k": None if student_cost is None else student_cost * 1000,
        "latency_ms": {k: _num(student_latency.get(k)) for k in ("p50", "p95", "mean")} | {"n": student_latency_n},
        "source": student_source,
        "flagged": student_source == "eval",
        "note": student_note,
        "cost_basis": "local energy estimate over the mean student latency",
        "eval_latency_ms": eval_latency,
        "provenance": provenance,
    }

    escalated_by_hash = {h: escalated(_num(p.get(field)), threshold) for h, p in aligned.items()}
    n_full = len(escalated_by_hash)
    n_escalated = sum(escalated_by_hash.values())
    imputed = {"teacher_cost": 0, "teacher_latency": 0}
    incomplete: list[str] = []

    def teacher_value(h: str, kind: str) -> float | None:
        record = teacher_by_hash.get(h)
        value = None if record is None else (record.cost_usd if kind == "teacher_cost" else record.latency_ms)
        if value is None:
            value = teacher_cost if kind == "teacher_cost" else teacher_latency_mean
            if value is not None:
                imputed[kind] += 1
        return value

    escalated_costs = [teacher_value(h, "teacher_cost") for h, esc in escalated_by_hash.items() if esc]
    cascade_cost: float | None = None
    if not n_full:
        incomplete.append("no test predictions of the selected run")
    elif student_cost is None:
        incomplete.append("no student latency to estimate the student energy from")
    elif any(c is None for c in escalated_costs):
        incomplete.append("escalated test requests have no recorded teacher cost, and there is none to impute from")
    else:
        cascade_cost = student_cost + sum(c for c in escalated_costs if c is not None) / n_full

    latency_hashes = [h for h in hashes if h in student_per_request and h in escalated_by_hash]
    composed_latency: list[float] = []
    latency_escalated = 0
    for h in latency_hashes:
        total_ms = student_per_request[h]
        if escalated_by_hash[h]:
            latency_escalated += 1
            teacher_ms = teacher_value(h, "teacher_latency")
            if teacher_ms is None:
                incomplete.append(
                    "escalated test requests have no recorded teacher latency, and there is none to impute from"
                )
                composed_latency = []
                break
            total_ms += teacher_ms
        composed_latency.append(total_ms)
    n_latency = len(latency_hashes)
    if imputed["teacher_cost"] or imputed["teacher_latency"]:
        notes.append(
            f"escalated test requests without a recorded teacher response were given the mean recorded teacher cost "
            f"({imputed['teacher_cost']}) and latency ({imputed['teacher_latency']})"
        )
    for reason in dict.fromkeys(incomplete):
        notes.append(f"cascade figures incomplete: {reason}")
    if student_source == "bench" and n_latency < n_full:
        notes.append(
            f"cascade latency is composed over the {n_latency} benched test requests "
            f"({latency_escalated / n_latency:.1%} escalated); the escalation rate and cost cover all {n_full}"
            if n_latency
            else "the student-only bench holds no per-request latency of a test request: no cascade latency"
        )
    cascade_row = {
        "usd_per_request": cascade_cost,
        "usd_per_1k": None if cascade_cost is None else cascade_cost * 1000,
        "latency_ms": latency_summary(composed_latency),
        "n": n_full,
        "n_escalated": n_escalated,
        "escalation_rate": n_escalated / n_full if n_full else None,
        "latency_n": len(composed_latency),
        "latency_escalation_rate": latency_escalated / n_latency if n_latency and composed_latency else None,
        "latency_scope": "benched test requests" if student_source == "bench" else "test split",
        "imputed": imputed,
        "incomplete": list(dict.fromkeys(incomplete)) or None,
        "threshold": threshold,
        "confidence": field,
        "student_latency_source": student_source,
        "source": "composed per test request: student latency, plus the recorded teacher latency when escalated",
        "cost_basis": "student energy for every request plus the recorded teacher cost of escalated requests",
    }
    return {
        "teacher": teacher_row,
        "student": student_row,
        "cascade": cascade_row,
        "assumptions": cost_assumptions(cost),
        "predictions_backend": header.get("backend"),
        "notes": notes,
    }


def _labelling_concurrency(curate_stats: Mapping[str, Any]) -> dict[str, Any]:
    """The live labelling concurrency recorded by curate, else the teacher client's default (8)."""
    value = _dict(curate_stats.get("labelling")).get("concurrency")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return {"value": value, "source": "curate_stats.json"}
    return {"value": DEFAULT_TEACHER_CONCURRENCY, "source": "default of the live teacher client"}


# -- break-even ------------------------------------------------------------------------------------


def labelling_cost(
    spec: TaskSpec,
    curate_stats: Mapping[str, Any],
    *,
    raw_inputs: Mapping[str, str],
    records: TeacherRecords,
) -> dict[str, Any]:
    """Teacher labelling cost: the ledger's labelling phases, else ``curate_stats.json``, else recorded costs.

    The recorded fallback sums the recorded ``usage.cost`` of the teacher requests of every curated example, which
    is what labelling costs when a reproduction replays the recording and the ledger holds nothing.
    """
    if paths.ledger_path().is_file():
        spent = _dict(_dict(Ledger().summary().get("by_task_phase")).get(spec.task))
        by_phase = {
            phase: float(usd)
            for phase, usd in sorted(spent.items())
            if phase in LABELLING_PHASES or "label" in phase or "capture" in phase
        }
        total = sum(by_phase.values())
        if total > 0:
            return {"usd": total, "source": "ledger", "by_phase": {k: v for k, v in by_phase.items() if v > 0}}
    stats_cost = _num(_dict(curate_stats.get("labelling")).get("cost_usd"))
    if stats_cost is not None and stats_cost > 0:
        return {"usd": stats_cost, "source": "curate_stats.json"}
    costs: list[float] = []
    n = 0
    for split in ("train", "valid", "test"):
        for h in _split_hashes(spec.task, split):
            if h not in raw_inputs:
                continue
            n += 1
            record = records.get(raw_inputs[h])
            if record is not None and record.cost_usd is not None:
                costs.append(record.cost_usd)
    if costs:
        return {"usd": float(sum(costs)), "source": "recorded", "n_requests": n, "n_priced": len(costs)}
    return {"usd": stats_cost if stats_cost is not None else 0.0, "source": "none found", "n_requests": n}


def _train_log(task: str, run_id: str | None) -> dict[str, Any]:
    if not run_id:
        return {}
    return _dict(_read_json(_task_dir(task) / "runs" / run_id / "train_log.json"))


def build_break_even(
    spec: TaskSpec,
    selected: str | None,
    cost_latency: Mapping[str, Any],
    labelling: Mapping[str, Any],
) -> dict[str, Any]:
    """Break-even volume = (labelling cost + training cost) / (teacher - cascade cost per request).

    ``reason`` says why the volume is null; ``notes`` list what it leaves out (then it is a lower bound). The
    volume is also under ``requests``, the key the CLI and the demo read.
    """
    log = _train_log(spec.task, selected)
    wall = _num(log.get("wall_seconds"))
    training = None if wall is None else local_cost_usd(wall * 1000.0, spec.cost)
    teacher = _num(_dict(cost_latency.get("teacher")).get("usd_per_request"))
    cascade_row = _dict(cost_latency.get("cascade"))
    cascade = _num(cascade_row.get("usd_per_request"))
    labelling_usd = _num(labelling.get("usd")) or 0.0
    notes: list[str] = []
    if labelling.get("source") == "none found":
        notes.append(
            "no teacher labelling cost was found (ledger, curate_stats.json or recorded costs): the volume leaves it "
            "out and is a lower bound"
        )
    if training is None:
        notes.append("the selected run has no train_log.json wall time: the volume leaves the training energy out")
    imputed = _dict(cascade_row.get("imputed"))
    if _num(imputed.get("teacher_cost")):
        notes.append(
            f"{imputed['teacher_cost']} escalated test requests have no recorded teacher cost; the mean recorded "
            "cost was used for them"
        )
    out: dict[str, Any] = {
        "volume": None,
        "requests": None,
        "volume_exact": None,
        "lower_bound": bool(labelling.get("source") == "none found" or training is None),
        "reason": None,
        "notes": notes,
        "labelling_usd": labelling_usd,
        "labelling_source": labelling.get("source"),
        "training_usd": training,
        "training_wall_seconds": wall,
        "training_run": selected,
        "teacher_usd_per_request": teacher,
        "cascade_usd_per_request": cascade,
        "savings_usd_per_request": None,
        "formula": "(teacher labelling cost + training energy cost) / (teacher - cascade cost per request)",
        "assumptions": cost_assumptions(spec.cost)["text"],
    }
    if teacher is None or cascade is None:
        out["reason"] = "the teacher or cascade cost per request is unknown"
        return out
    savings = teacher - cascade
    out["savings_usd_per_request"] = savings
    if savings <= 0:
        out["reason"] = "the cascade saves nothing per request (it costs at least as much as the teacher)"
        return out
    exact = (labelling_usd + (training or 0.0)) / savings
    out["volume_exact"] = exact
    out["volume"] = out["requests"] = math.ceil(exact - 1e-9)
    return out


# -- live bench cross-check and served traffic -----------------------------------------------------


def _ratio(a: Any, b: Any) -> float | None:
    x, y = _num(a), _num(b)
    return x / y if x is not None and y not in (None, 0.0) else None


def build_live_bench(
    benches: Mapping[str, tuple[str, dict[str, Any]]],
    cost_latency: Mapping[str, Any],
    operating_point: Mapping[str, Any],
    selected: str | None,
) -> dict[str, Any]:
    """The latest bench per mode next to the composed figures; only a bench of the selected run is cross-checked."""
    out: dict[str, Any] = {"student_only": None, "cascade": None, "cross_check": {}, "notes": []}
    for mode in ("student_only", "cascade"):
        if mode in benches:
            name, data = benches[mode]
            summary = _bench_summary(name, data)
            summary["matches_selected_run"] = _bench_matches(data, selected)
            out[mode] = summary
            if not summary["matches_selected_run"]:
                out["notes"].append(
                    f"the {mode.replace('_', '-')} bench {name} measured run {data.get('run_id')}, not the selected "
                    f"run {selected}: it is not cross-checked against the composed figures"
                )
    cascade = out["cascade"]
    if cascade is not None and cascade["matches_selected_run"]:
        cascade_row = _dict(cost_latency.get("cascade"))
        composed = _dict(cascade_row.get("latency_ms"))
        measured = _dict(cascade.get("latency_ms"))
        out["cross_check"]["cascade"] = {
            "composed_p50": composed.get("p50"),
            "composed_p95": composed.get("p95"),
            "composed_n": cascade_row.get("latency_n"),
            "measured_p50": measured.get("p50"),
            "measured_p95": measured.get("p95"),
            "measured_n": cascade.get("n"),
            "ratio_p50": _ratio(measured.get("p50"), composed.get("p50")),
            "ratio_p95": _ratio(measured.get("p95"), composed.get("p95")),
            "test_escalation_rate": _dict(operating_point.get("escalation_rate")).get("test"),
            "measured_escalation_rate": cascade.get("escalation_rate"),
        }
    student = out["student_only"]
    student_row = _dict(cost_latency.get("student"))
    if student is not None and student["matches_selected_run"]:
        measured = _dict(student.get("latency_ms"))
        eval_latency = _dict(student_row.get("eval_latency_ms"))
        out["cross_check"]["student_only"] = {
            "measured_p50": measured.get("p50"),
            "measured_p95": measured.get("p95"),
            "eval_p50": eval_latency.get("p50"),
            "eval_p95": eval_latency.get("p95"),
        }
    return out


def build_serve_log(
    spec: TaskSpec, store: Store, since_s: float | None, operating_point: Mapping[str, Any]
) -> dict[str, Any]:
    """Figures from the served-request log: routes, observed escalation, latency by route, spend and drift."""
    since = time.time() - since_s if since_s is not None else None
    rows = list(store.iter_served(spec.task, since))
    n = len(rows)
    routes = Counter(r.route for r in rows)
    reasons = Counter(r.reason for r in rows if r.reason)
    # Escalated: a reason and no student answer. ``student-fallback`` and ``error`` rows with a reason are
    # escalations whose teacher call failed (or missed the recording).
    escalated_rows = [r for r in rows if r.reason and r.route != "student"]
    failed = [r for r in escalated_rows if r.route != "teacher"]
    # The student's own decisions: it answered, or it escalated on low confidence (whatever the teacher did next).
    # Rows it never decided on (unsupported, unparsed input, errors before the student) are left out.
    confidence_rows = [r for r in rows if (r.route == "student" and not r.reason) or r.reason == "low_confidence"]
    low_confidence = sum(1 for r in confidence_rows if r.reason == "low_confidence")
    teacher_spend = float(sum(_num(r.teacher_cost_usd) or 0.0 for r in rows))
    energy = float(sum(local_cost_usd(_num(r.student_ms) or 0.0, spec.cost) for r in rows))
    by_route = {route: latency_summary(r.total_ms for r in rows if r.route == route) for route in sorted(routes)}
    out: dict[str, Any] = {
        "since_seconds": since_s,
        "requests": n,
        "route_counts": dict(sorted(routes.items())),
        "route_shares": {route: count / n for route, count in sorted(routes.items())} if n else {},
        "reasons": dict(sorted(reasons.items())),
        "escalations": len(escalated_rows),
        "escalation_rate": len(escalated_rows) / n if n else None,
        "failed_escalations": len(failed),
        "failed_escalation_routes": dict(sorted(Counter(r.route for r in failed).items())),
        "confidence_decisions": len(confidence_rows),
        "confidence_escalation_rate": low_confidence / len(confidence_rows) if confidence_rows else None,
        "latency_ms": latency_summary(r.total_ms for r in rows),
        "latency_ms_by_route": by_route,
        "teacher_spend_usd": teacher_spend,
        "teacher_modes": dict(Counter(r.teacher_mode for r in rows if r.teacher_mode)),
        "local_energy_usd": energy,
        "usd_per_1k": (teacher_spend + energy) / n * 1000 if n else None,
        "assumptions": cost_assumptions(spec.cost)["text"],
        "drift": None,
    }
    expected = _num(_dict(operating_point.get("escalation_rate")).get("test"))
    interval = wilson_interval(low_confidence, len(confidence_rows))
    if expected is not None and interval is not None:
        observed = low_confidence / len(confidence_rows)
        lo, hi = interval
        outside = not lo <= expected <= hi
        verdict = (
            "outside the interval: the served inputs may have drifted from the training data"
            if outside
            else "consistent with the operating point"
        )
        out["drift"] = {
            "expected_escalation_rate": expected,
            "observed_escalation_rate": observed,
            "difference": observed - expected,
            "interval": [lo, hi],
            "possible_drift": outside,
            "note": (
                f"observed low-confidence escalation {observed:.1%} over {len(confidence_rows)} requests "
                f"(95% CI {lo:.1%}-{hi:.1%}) against {expected:.1%} on the test split at the operating point: "
                f"{verdict}"
            ),
        }
    elif n == 0:
        out["drift"] = {"note": "no served requests in the window"}
    return out


# -- training, dataset, spend ----------------------------------------------------------------------


def _length_dropped(curate_stats: Mapping[str, Any]) -> int | None:
    """Training examples the length filter dropped (``curate_stats.json["length"]["splits"]["train"]``)."""
    section = _dict(curate_stats.get("length"))
    train = _dict(_dict(section.get("splits")).get("train"))
    value = train.get("dropped")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def build_training(task: str, curate_stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per training run: examples, length-filter drops, iterations, epochs, time, memory, speed, size."""
    dropped = _length_dropped(curate_stats)
    rows = []
    for path in sorted((_task_dir(task) / "runs").glob("*/train_log.json")):
        log = _dict(_read_json(path))
        if not log:
            continue
        wall = _num(log.get("wall_seconds"))
        rows.append(
            {
                "run_id": log.get("run_id") or path.parent.name,
                "base_model": log.get("base_model"),
                "backend": log.get("backend"),
                "profile": log.get("profile"),
                "labels": log.get("labels"),
                "seed": log.get("seed"),
                "examples": log.get("n_train"),
                "dropped_by_length_filter": dropped,
                "dropped_no_gold": log.get("dropped_no_gold"),
                "iterations": log.get("iterations"),
                "epochs": log.get("epochs"),
                "wall_minutes": None if wall is None else wall / 60.0,
                "peak_memory_gb": _num(log.get("peak_memory_gb")),
                "tokens_per_second": _num(log.get("tokens_per_second")),
                "adapter_mb": _num(log.get("adapter_size_mb")),
                "best_iteration": log.get("best_iteration"),
                "best_val_loss": _num(log.get("best_val_loss")),
            }
        )
    return rows


def _count_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def build_dataset(task: str, curate_stats: Mapping[str, Any]) -> dict[str, Any]:
    data = _task_dir(task) / "data"
    return {
        "splits": {split: _count_lines(data / f"{split}.jsonl") for split in ("train", "valid", "test")},
        "dataset_card": "dataset_card.md" if (data / "dataset_card.md").is_file() else None,
        "curate_stats": dict(curate_stats) or None,
    }


def build_spend(task: str) -> dict[str, Any]:
    if not paths.ledger_path().is_file():
        return {"task_total": 0.0, "task_by_phase": {}, "workspace_total": 0.0, "global_cap": None, "ledger": False}
    summary = Ledger().summary()
    return {
        "task_total": _dict(summary.get("by_task")).get(task, 0.0),
        "task_by_phase": _dict(_dict(summary.get("by_task_phase")).get(task)),
        "workspace_total": summary.get("total"),
        "global_cap": summary.get("cap"),
        "open_reservations": summary.get("open_reservations"),
        "ledger": True,
    }


# -- entry point -----------------------------------------------------------------------------------


def _since_text(since_s: float) -> str:
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if since_s >= size and since_s % size == 0:
            return f"{int(since_s // size)}{unit}"
    return f"{since_s:g}s"


def _default_command(spec: TaskSpec, from_serve_log: bool, since_s: float | None, out_dir: Path | None) -> str:
    parts = ["taskdistill", "report", "--task", spec.task]
    if from_serve_log:
        parts.append("--from-serve-log")
        if since_s is not None:
            parts += ["--since", _since_text(since_s)]
    if out_dir is not None:
        parts += ["--out", display_path(out_dir) if out_dir.is_absolute() else out_dir.as_posix()]
    return " ".join(parts)


def _selected_run(task: str, evals: Mapping[str, dict[str, Any]]) -> tuple[str | None, dict[str, Any], list[str]]:
    notes: list[str] = []
    data = _dict(_read_json(_task_dir(task) / "selected_run.json"))
    run_id = data.get("run_id")
    if isinstance(run_id, str) and run_id:
        if run_id not in evals:
            notes.append(f"the selected run {run_id} has no test evaluation (eval/{run_id}/eval_test.json)")
        return run_id, data, notes
    students = sorted(rid for rid, ev in evals.items() if _labels(ev) == "teacher")
    if len(students) == 1:
        notes.append("no selected_run.json: using the only teacher-label run")
        return students[0], data, notes
    if students:
        notes.append("no selected_run.json and several runs: the operating point and cascade are not reported")
    return None, data, notes


def build_report(
    spec: TaskSpec,
    *,
    from_serve_log: bool = False,
    since_s: float | None = None,
    out_dir: Path | str | None = None,
    store: Store | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Build the report, write ``<out_dir>/report.json`` and ``report.md`` and return the JSON.

    ``out_dir`` defaults to ``reports/<task>`` relative to the current directory.
    """
    target = Path(out_dir) if out_dir is not None else Path("reports") / spec.task
    evals = _load_evals(spec.task)
    if not evals and not from_serve_log:
        raise ReportError(
            f"no test evaluation for task '{spec.task}': run `taskdistill eval --task {spec.task}` first "
            "(or use --from-serve-log)"
        )
    selected, selection, notes = _selected_run(spec.task, evals)
    sel_ev = evals.get(selected) if selected else None
    candidates = _dict(selection.get("candidates"))
    curate_stats = _dict(_read_json(_task_dir(spec.task) / "data" / "curate_stats.json"))
    store = store if store is not None else Store()
    records = TeacherRecords(spec)
    raw_inputs = raw_inputs_by_hash(store, spec)
    benches = _latest_benches(spec.task, selected)

    quality = build_quality(spec, evals, selected, candidates)
    operating_point = build_operating_point(spec, sel_ev, selected, target)
    cost_latency = build_cost_latency(
        spec, sel_ev, selected, raw_inputs=raw_inputs, records=records, benches=benches, curate_stats=curate_stats
    )
    labelling = labelling_cost(spec, curate_stats, raw_inputs=raw_inputs, records=records)
    report: dict[str, Any] = {
        "task": spec.task,
        "task_type": spec.type,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": sanitise_command(
            command or _default_command(spec, from_serve_log, since_s, Path(out_dir) if out_dir else None)
        ),
        "hardware": hardware_info(),
        "teacher": {"model": spec.teacher.model, "provider": spec.teacher.provider},
        "selected_run": {
            "run_id": selected,
            "reason": selection.get("reason"),
            "date": selection.get("date"),
            "rule": selection.get("rule"),
        },
        "quality": quality,
        "operating_point": operating_point,
        "cost_latency": cost_latency,
        "assumptions": cost_assumptions(spec.cost),
        "break_even": build_break_even(spec, selected, cost_latency, labelling),
        "labelling_cost": labelling,
        "live_bench": build_live_bench(benches, cost_latency, operating_point, selected),
        "serve_log": None,
        "training": build_training(spec.task, curate_stats),
        "dataset": build_dataset(spec.task, curate_stats),
        "spend": build_spend(spec.task),
        "notes": notes,
    }
    if from_serve_log:
        report["serve_log"] = build_serve_log(spec, store, since_s, operating_point)
    report = _relative_paths(report)
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report
