# ADR 0003: Every choice is made on validation data, enforced by type

- Status: accepted
- Date: 2026-09-26

## Context

A distillation report is only credible if the test split was never used to choose anything: not the base
model, the seed, the checkpoint, the threshold, the calibration map or the baseline's hyperparameters. Such
leaks are easy to introduce by accident when the same helper functions serve both reporting and selection.

## Decision

- `taskdistill.evaluate.splits` defines `ValidationSplit` and `TestSplit` as two distinct classes (neither
  an alias nor a subclass of the other).
- Every selection function (`select_threshold`, `fit_isotonic`, `select_checkpoint`, `select_run`,
  `choose_base_model`, `tune_baseline`) is decorated with `@validation_only(...)`, which raises `TypeError` at
  call time when it receives a `TestSplit`, or a mapping or sequence that contains one. A unit test asserts
  this for each function.
- The run used for the cascade, the operating point and serving is chosen by `taskdistill eval --select` on
  validation and declared in `$TASKDISTILL_HOME/<task>/selected_run.json`.
- The threshold reference defaults to agreement with the teacher (`cascade.reference: teacher`), because that
  works on real traffic with no gold labels. `cascade.reference: gold` with `max_drop` is available when gold
  labels exist.
- `eval --split test` appends to `$TASKDISTILL_HOME/<task>/test_access_log.jsonl`, so the number of times the
  test split was scored is recorded rather than remembered.

## Consequences

- Test-set numbers can be reported by any code, but no code path can choose with them without failing loudly.
- The threshold chosen on validation can miss its target on test; the report states whether it held.
