"""Collect the cross-task report files from a workspace: reports/training.json.

    uv run python scripts/collect_reports.py   # reads $TASKDISTILL_HOME (default ./.taskdistill-reproduce)

The training table has one row per base model x task (the validation-selected seed for the base that has
several seeds), plus every other run for reference. Paths are relative; hardware comes from sysctl.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("banking77", "invoices")
FIELDS = (
    "run_id", "task", "backend", "base_model", "profile", "seed", "labels", "n_train", "n_valid", "dropped_no_gold",
    "dropped_too_long", "iterations", "epochs", "batch_size", "learning_rate", "lr_schedule", "warmup_iterations",
    "lora_rank", "lora_layers", "max_seq_len", "grad_checkpoint", "wall_seconds", "train_seconds", "peak_memory_gb",
    "tokens_per_second", "processed_tokens", "best_iteration", "best_val_loss", "adapter_size_mb", "date",
    "load_average", "hardware",
)  # fmt: skip


def main() -> int:
    home = Path(os.environ.get("TASKDISTILL_HOME") or ROOT / ".taskdistill-reproduce")
    runs: list[dict[str, Any]] = []
    selected: dict[str, str | None] = {}
    for task in TASKS:
        task_dir = home / task
        sel = task_dir / "selected_run.json"
        selected[task] = json.loads(sel.read_text())["run_id"] if sel.is_file() else None
        stats_file = task_dir / "data" / "curate_stats.json"
        length_dropped = None
        if stats_file.is_file():
            length_dropped = json.loads(stats_file.read_text()).get("length", {}).get("dropped")
        for log_file in sorted((task_dir / "runs").glob("*/train_log.json")):
            log = json.loads(log_file.read_text())
            row = {field: log.get(field) for field in FIELDS}
            row["curate_dropped_by_length"] = length_dropped
            row["selected"] = log.get("run_id") == selected[task]
            runs.append(row)
    table = []
    for task in TASKS:
        for base in sorted({r["base_model"] for r in runs if r["task"] == task and r["labels"] == "teacher"}):
            candidates = [r for r in runs if r["task"] == task and r["base_model"] == base and r["labels"] == "teacher"]
            chosen = next((r for r in candidates if r["selected"]), None) or min(candidates, key=lambda r: r["seed"])
            table.append({**chosen, "seeds_trained": sorted(r["seed"] for r in candidates)})
    out = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "scripts/reproduce.sh (then scripts/collect_reports.py)",
        "selected_runs": selected,
        "table": table,
        "runs": runs,
    }
    target = ROOT / "reports" / "training.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
