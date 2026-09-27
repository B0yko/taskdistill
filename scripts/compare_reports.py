"""Compare the reports of two full reproductions and write reports/reproduction.json.

    uv run python scripts/compare_reports.py --reference reports --rerun path/to/rerun/reports [--tolerance 2.0]

Quality rows are matched by system, base model and labels; the test-split metrics of each row are compared in
percentage points against the tolerance. The selected run (base model and seed) of each task is compared too:
it is a decision on validation data, so it is reported rather than held to a tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("banking77", "invoices")
METRICS = {
    "classification": ("accuracy", "macro_f1", "agreement"),
    "extraction": ("json_validity", "field_micro_f1", "field_exact_match", "doc_exact_match", "agreement"),
}


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('system')}|{row.get('base_model')}|{row.get('labels')}"


def compare_task(reference: dict[str, Any], rerun: dict[str, Any], tolerance_pts: float) -> dict[str, Any]:
    kind = reference.get("task_type", "classification")
    ref_rows = {row_key(r): r for r in reference["quality"]["rows"]}
    new_rows = {row_key(r): r for r in rerun["quality"]["rows"]}
    rows = []
    for key, ref in ref_rows.items():
        new = new_rows.get(key)
        if new is None or ref.get("system") == "cascade":
            continue  # the cascade row depends on the selected run, compared separately below
        for metric in METRICS[kind]:
            a, b = ref["metrics"].get(metric), new["metrics"].get(metric)
            if not isinstance(a, int | float) or not isinstance(b, int | float):
                continue
            diff = (b - a) * 100
            rows.append(
                {
                    "row": ref.get("name"),
                    "metric": metric,
                    "reference": a,
                    "rerun": b,
                    "diff_pts": round(diff, 3),
                    "within": abs(diff) <= tolerance_pts,
                }
            )
    ref_sel, new_sel = reference.get("selected_run") or {}, rerun.get("selected_run") or {}
    cascades = {}
    for name, report in (("reference", reference), ("rerun", rerun)):
        row = next((r for r in report["quality"]["rows"] if r.get("system") == "cascade"), None)
        op = report.get("operating_point") or {}
        cascades[name] = {
            "run_id": ref_sel.get("run_id") if name == "reference" else new_sel.get("run_id"),
            "metrics": None if row is None else row.get("metrics"),
            "escalation_rate_test": (op.get("escalation_rate") or {}).get("test"),
            "target_met_on_test": op.get("target_met_on_test"),
        }
    diffs = [abs(r["diff_pts"]) for r in rows]
    return {
        "rows": rows,
        "max_abs_diff_pts": max(diffs) if diffs else None,
        "all_within": all(r["within"] for r in rows),
        "selected_run": {
            "reference": ref_sel.get("run_id"),
            "rerun": new_sel.get("run_id"),
            "same": ref_sel.get("run_id") == new_sel.get("run_id"),
            "reference_reason": ref_sel.get("reason"),
            "rerun_reason": new_sel.get("reason"),
        },
        "cascade": cascades,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=ROOT / "reports")
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2.0, help="percentage points (default 2.0)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "reproduction.json")
    args = parser.parse_args()
    out: dict[str, Any] = {"tolerance_pts": args.tolerance, "tasks": {}}
    for task in TASKS:
        reference = json.loads((args.reference / task / "report.json").read_text(encoding="utf-8"))
        rerun = json.loads((args.rerun / task / "report.json").read_text(encoding="utf-8"))
        out.setdefault("reference", {"date": reference.get("date"), "hardware": reference.get("hardware")})
        out.setdefault("rerun", {"date": rerun.get("date"), "hardware": rerun.get("hardware")})
        out["tasks"][task] = compare_task(reference, rerun, args.tolerance)
    out["all_within"] = all(t["all_within"] for t in out["tasks"].values())
    out["command"] = "scripts/reproduce.sh on a second machine, then scripts/compare_reports.py"
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for task, result in out["tasks"].items():
        sel = result["selected_run"]
        print(
            f"{task}: max |diff| {result['max_abs_diff_pts']} pts, all within {args.tolerance}: "
            f"{result['all_within']}; selected {sel['reference']} vs {sel['rerun']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
