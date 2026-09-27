# ADR 0006: Base model choice, 0.5B against 1.5B, per task

- Status: accepted
- Date: 2026-09-27

## Context

A larger student is usually more accurate and always slower. The cascade exists to cut latency and cost, so the
larger base has to earn its latency on the task at hand, and the decision must not look at the test split.

## Decision

`taskdistill eval --select` applies one rule per task on the validation split: the seed with the best validation
metric is chosen within each base model, and the 1.5B base replaces the 0.5B base only if it raises the validation
target metric (agreement with the teacher) by at least 1 point **and** its student p95 latency stays under 3 times
the 0.5B p95. The run is declared in `selected_run.json`; the bundled demo specs keep the 0.5B default for the quick
profile.

Outcome of the full-profile runs on the MacBook Air M5 (`reports/<task>/report.json`, `selected_run.candidates`):

| Task | 0.5B (best seed) | 1.5B | Gain | p95 ratio | Chosen |
|---|---|---|---|---|---|
| Banking77 | 88.2% agreement, p95 75 ms | 89.1%, p95 163 ms | +0.97 pts | 2.2x | 0.5B |
| Invoices | 94.7% agreement, p95 1.10 s | 98.5%, p95 1.97 s | +3.79 pts | 1.8x | 1.5B |

## Consequences

- The two tasks end up with different students, as intended: the 1.5B base pays for itself on extraction, not on
  intent classification.
- The Banking77 decision sits on the edge of the rule. The reproduction on a second machine (Apple M4 Max) measured a
  +1.26-point gain and chose the 1.5B base; a knife-edge rule should be read together with its measured gain, which
  the report prints next to the decision.
- p95 comes from the in-process validation predictions of each run, measured on the machine that ran eval; the live
  latency through the server is measured separately with `taskdistill bench`.
