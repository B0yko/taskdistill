"""Ablation: constant learning rate vs linear warm-up + cosine decay, Banking77 quick profile, 3 seeds.

Everything is scored on the VALIDATION split only (no test access). Runs in replay mode in its own workspace,
``$ABLATION_HOME`` (default ``<repo>/.taskdistill-ablation``), never whatever ``TASKDISTILL_HOME`` an enclosing
shell (e.g. its own working workspace) already has set:

    uv run python scripts/ablation_lr_schedule.py            # writes reports/ablations/lr_schedule.json
    ABLATION_HOME=/tmp/ablation uv run python scripts/ablation_lr_schedule.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "ablations" / "lr_schedule.json"
SEEDS = (13, 14, 15)
SCHEDULES = ("constant", "warmup_cosine")
DEFAULT_ABLATION_HOME = ROOT / ".taskdistill-ablation"


def ablation_home() -> str:
    """The ablation's own workspace: ``$ABLATION_HOME`` if set, else a repo-local default.

    Never an inherited ``$TASKDISTILL_HOME`` (reproduce.sh nests the ablation's workspace under its own
    ``$REPRODUCE_HOME`` by passing ``ABLATION_HOME``; run directly, without it, the ablation still gets its own
    workspace rather than mixing with whatever workspace the caller's shell already has set).
    """
    return os.environ.get("ABLATION_HOME") or str(DEFAULT_ABLATION_HOME)


def main() -> int:
    os.environ["TASKDISTILL_HOME"] = ablation_home()
    for name in ("OPENROUTER_API_KEY", "TASKDISTILL_TEACHER_API_KEY"):
        os.environ.pop(name, None)  # replay only

    from taskdistill.config import load_task
    from taskdistill.demos.runner import run_demo
    from taskdistill.evaluate.runner import run_eval
    from taskdistill.hardware import hardware_info
    from taskdistill.train.runner import run_training

    run_demo("banking77", profile="quick", until="curate")
    base_spec = load_task("banking77")
    runs = []
    for schedule in SCHEDULES:
        spec = base_spec.model_copy(deep=True)
        spec.train.lr_schedule = schedule  # type: ignore[assignment]
        for seed in SEEDS:
            result = run_training(spec, profile="quick", seed=seed, log=lambda _line: None)
            ev = run_eval(spec, run_id=result.run_id, split="valid", fast=True, log=lambda _line: None)
            metrics = ev["systems"]["student"]["metrics"]
            log = result.log
            runs.append(
                {
                    "run_id": result.run_id,
                    "lr_schedule": schedule,
                    "seed": seed,
                    "iterations": log.get("iterations"),
                    "warmup_iterations": log.get("warmup_iterations"),
                    "best_val_loss": log.get("best_val_loss"),
                    "val_loss_curve": log.get("curve", {}).get("val"),
                    "valid_accuracy": metrics.get("accuracy"),
                    "valid_agreement": metrics.get("agreement"),
                    "n_valid": ev.get("n"),
                }
            )
            print(f"{schedule} seed {seed}: valid agreement {metrics.get('agreement'):.3f}", file=sys.stderr)
    summary = {}
    for schedule in SCHEDULES:
        values = [r["valid_agreement"] for r in runs if r["lr_schedule"] == schedule]
        accs = [r["valid_accuracy"] for r in runs if r["lr_schedule"] == schedule]
        summary[schedule] = {
            "valid_agreement_mean": statistics.mean(values),
            "valid_agreement_std": statistics.stdev(values),
            "valid_accuracy_mean": statistics.mean(accs),
            "valid_accuracy_std": statistics.stdev(accs),
        }
    report = {
        "ablation": "learning-rate schedule (peak 1e-4, LoRA rank 16, scale 20, all layers, batch 8)",
        "task": "banking77",
        "profile": "quick",
        "split": "valid",
        "seeds": list(SEEDS),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": hardware_info(),
        "command": "uv run python scripts/ablation_lr_schedule.py",
        "runs": runs,
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
