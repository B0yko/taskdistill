#!/usr/bin/env bash
# Regenerate every replay-based reports/*.json from scratch: no API key, recorded teacher outputs only.
#
#   scripts/reproduce.sh                 # both tasks (several hours on a MacBook Air M5)
#   scripts/reproduce.sh banking77       # one task
#
# It uses its own workspace (default ./.taskdistill-reproduce) so it never mixes with other runs. Training on
# Metal is not bit-for-bit deterministic, so reruns can differ slightly; README states the tolerance used.
set -euo pipefail
cd "$(dirname "$0")/.."

export TASKDISTILL_HOME="${TASKDISTILL_HOME:-$PWD/.taskdistill-reproduce}"
unset OPENROUTER_API_KEY TASKDISTILL_TEACHER_API_KEY
export TRANSFORMERS_VERBOSITY=error

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
  echo "== $task: gold-label ceiling at the chosen size, zero-shot base"
  local base
  base=$(selected_base "$task")
  td train --task "$task" --profile full --base "$base" --seed 13 --labels gold
  td eval --task "$task" --run "$(uv run --quiet python -c "from taskdistill.train.common import run_id; print(run_id('$base', 'full', 13, 'gold'))")"
  td eval --task "$task" --zero-shot
  echo "== $task: report"
  td report --task "$task" --out "reports/$task"
  cp "$TASKDISTILL_HOME/$task/data/dataset_card.md" "reports/$task/dataset_card.md"
}

tasks=("${@:-banking77 invoices}")
for task in ${tasks[@]}; do
  reproduce_task "$task"
done
uv run --quiet python scripts/collect_reports.py
echo "done: reports/ regenerated; run scripts/sync_readme.py to update README.md"
