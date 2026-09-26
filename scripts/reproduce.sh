#!/usr/bin/env bash
# Regenerate every replay-based reports/*.json from scratch: no API key, recorded teacher outputs only.
#
#   scripts/reproduce.sh                 # both tasks, then the learning-rate ablation (several hours on a MacBook Air M5)
#   scripts/reproduce.sh banking77       # one task (the ablation is Banking77-only and runs with it)
#   REPRODUCE_HOME=/tmp/repro scripts/reproduce.sh
#
# It uses its own workspace, $REPRODUCE_HOME (default ./.taskdistill-reproduce), so it never mixes with other runs.
# An inherited TASKDISTILL_HOME is ignored on purpose: that variable names your working workspace (its ledger,
# cache, runs and selected run), and the reproduction retrains fixed run ids and rewrites selected_run.json and
# threshold.json, which must not happen there. The ablation gets a workspace of its own under $REPRODUCE_HOME too,
# passed as $ABLATION_HOME (ablation_lr_schedule.py ignores any inherited TASKDISTILL_HOME the same way).
# Training on Metal is not bit-for-bit deterministic, so reruns can differ slightly; README states the tolerance used.
set -euo pipefail
start_dir=$PWD
cd "$(dirname "$0")/.."

REPRODUCE_HOME="${REPRODUCE_HOME:-$PWD/.taskdistill-reproduce}"
case $REPRODUCE_HOME in
  /*) ;;
  *) REPRODUCE_HOME="$start_dir/$REPRODUCE_HOME" ;;  # relative to where the script was started
esac
if [[ -n "${TASKDISTILL_HOME:-}" && "$TASKDISTILL_HOME" != "$REPRODUCE_HOME" ]]; then
  echo "note: ignoring TASKDISTILL_HOME=$TASKDISTILL_HOME; the reproduction uses REPRODUCE_HOME=$REPRODUCE_HOME"
fi
export TASKDISTILL_HOME="$REPRODUCE_HOME"
ABLATION_HOME="$REPRODUCE_HOME/_ablation_lr_schedule"
unset OPENROUTER_API_KEY TASKDISTILL_TEACHER_API_KEY
export TRANSFORMERS_VERBOSITY=error
echo "workspace: $TASKDISTILL_HOME"

td() { uv run --quiet taskdistill "$@"; }
SMALL=mlx-community/Qwen2.5-0.5B-Instruct-4bit
LARGE=mlx-community/Qwen2.5-1.5B-Instruct-4bit

selected_base() {
  uv run --quiet python -c "
import json, os, sys
from pathlib import Path
home = Path(os.environ['TASKDISTILL_HOME'])
run = json.loads((home / sys.argv[1] / 'selected_run.json').read_text())['run_id']
print(json.loads((home / sys.argv[1] / 'runs' / run / 'train_log.json').read_text())['base_model'])
" "$1"
}

reproduce_task() {
  local task=$1
  echo "== $task: data, capture (replay), curate, first 0.5B run, selection, smoke test, report"
  td demo "$task" --profile full
  echo "== $task: two more 0.5B seeds and the 1.5B run"
  td train --task "$task" --profile full --seed 14
  td train --task "$task" --profile full --seed 15
  td train --task "$task" --profile full --base "$LARGE" --seed 13
  echo "== $task: choose the run on validation (seed, then base-model rule), score it on test"
  td eval --task "$task" --select
  for run in qwen2.5-0.5b-full-s13 qwen2.5-0.5b-full-s14 qwen2.5-0.5b-full-s15 qwen2.5-1.5b-full-s13; do
    td eval --task "$task" --run "$run"
  done
  echo "== $task: gold-label ceiling at the chosen size; zero-shot baseline on the 0.5B base (whichever size is chosen)"
  local base
  base=$(selected_base "$task")
  td train --task "$task" --profile full --base "$base" --seed 13 --labels gold
  td eval --task "$task" --run "$(uv run --quiet python -c "from taskdistill.train.common import run_id; print(run_id('$base', 'full', 13, 'gold'))")"
  td eval --task "$task" --zero-shot --base "$SMALL"
  echo "== $task: report"
  td report --task "$task" --out "reports/$task"
  cp "$TASKDISTILL_HOME/$task/data/dataset_card.md" "reports/$task/dataset_card.md"
}

tasks=("${@:-banking77 invoices}")
ablation=0
for task in ${tasks[@]}; do
  reproduce_task "$task"
  if [[ $task == banking77 ]]; then
    ablation=1
  fi
done
uv run --quiet python scripts/collect_reports.py
if [[ $ablation == 1 ]]; then
  echo "== ablation: constant vs warm-up + cosine learning rate (Banking77 quick, validation only)"
  ABLATION_HOME="$ABLATION_HOME" uv run --quiet python scripts/ablation_lr_schedule.py
fi
echo "done: reports/ regenerated; run scripts/sync_readme.py to update README.md"
