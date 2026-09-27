# taskdistill report: banking77

- Report built: 2026-09-27T00:49:45+00:00 on Mac17,4, Apple M5, 24 GB, macOS 26.6.2
- Command: `taskdistill report --task banking77 --out reports/banking77`
- Each measured number carries the date and hardware of the evaluation or bench that produced it.
- Teacher: `deepseek/deepseek-v4.1-flash` via deepinfra/fp8
- Selected run: `qwen2.5-0.5b-full-s13` (qwen2.5-0.5b-full-s13: best validation agreement (0.8816) among 3 run(s) of mlx-community/Qwen2.5-0.5B-Instruct-4bit; mlx-community/Qwen2.5-1.5B-Instruct-4bit vs mlx-community/Qwen2.5-0.5B-Instruct-4bit: large gains 0.97 points, below the 1.00-point minimum)

## Quality (test split, n = 3,075)

Cells are the score and its 95% paired bootstrap interval (1,000 resamples). Rows over several seeds show mean ± sample standard deviation; their interval is that of the validation-selected seed. Teacher and cascade numbers come from recorded teacher outputs. ECE and AUROC use raw student confidence.

| System | n | Accuracy | Macro-F1 | Agreement | ECE | AUROC | Runs |
|---|---|---|---|---|---|---|---|
| teacher (deepseek/deepseek-v4.1-flash) | 3,075 | 75.8% [74.2, 77.4] | 74.9% [73.0, 76.1] | reference | — | — | `qwen2.5-0.5b-full-s13` |
| zero-shot qwen2.5-0.5b | 3,075 | 23.0% [21.6, 24.5] | 20.5% [19.0, 21.6] | 25.6% [24.1, 27.0] | 35.3% [33.9, 36.7] | 0.738 [0.718, 0.761] | `zero-shot-qwen2.5-0.5b` |
| TF-IDF + logistic regression (teacher labels) | 3,075 | 72.5% [70.8, 74.0] | 71.4% [69.7, 72.5] | 83.5% [82.2, 84.8] | — | — | `qwen2.5-0.5b-full-s13` |
| student qwen2.5-0.5b (teacher labels, 3 seeds) | 3,075 | 75.7% ± 0.5 [73.8, 77.0] | 74.6% ± 0.7 [72.7, 75.6] | 88.1% ± 0.2 [86.9, 89.3] | 15.2% ± 0.3 [13.7, 16.5] | 0.801 ± 0.003 [0.781, 0.816] | `qwen2.5-0.5b-full-s13`, `qwen2.5-0.5b-full-s14`, `qwen2.5-0.5b-full-s15` |
| student qwen2.5-1.5b (teacher labels) | 3,075 | 76.5% [74.9, 78.0] | 75.5% [73.9, 76.7] | 89.2% [88.1, 90.4] | 15.4% [14.1, 16.9] | 0.797 [0.779, 0.814] | `qwen2.5-1.5b-full-s13` |
| student qwen2.5-0.5b (gold labels) | 3,075 | 92.0% [91.0, 92.9] | 92.0% [91.0, 92.9] | 75.8% [74.2, 77.4] | 1.4% [1.0, 2.4] | 0.915 [0.897, 0.931] | `qwen2.5-0.5b-full-s13-gold` |
| cascade (qwen2.5-0.5b-full-s13, t = 0.874) | 3,075 | 76.1% [74.5, 77.7] | 75.0% [73.3, 76.2] | 97.6% [97.0, 98.1] | — | — | `qwen2.5-0.5b-full-s13` |

Per-seed values, student qwen2.5-0.5b (teacher labels, 3 seeds) (selected: `qwen2.5-0.5b-full-s13`):

| Run | Seed | Accuracy | Macro-F1 | Agreement | ECE | AUROC |
|---|---|---|---|---|---|---|
| `qwen2.5-0.5b-full-s13` | 13 | 75.5% | 74.4% | 88.1% | 15.0% | 0.799 |
| `qwen2.5-0.5b-full-s14` | 14 | 75.3% | 74.1% | 87.9% | 15.6% | 0.801 |
| `qwen2.5-0.5b-full-s15` | 15 | 76.3% | 75.4% | 88.2% | 15.0% | 0.804 |

<details>
<summary>Confidence definitions compared</summary>

| Split | Confidence | n | Mean confidence | AUROC vs teacher | AUROC vs gold | Accuracy vs teacher | Accuracy vs gold | ECE vs teacher | ECE vs gold |
|---|---|---|---|---|---|---|---|---|---|
| valid | primary (chosen) | 1,030 | 90.2% | 0.877 | 0.799 | 88.2% | 73.2% | 2.8% | 17.0% |
| valid | alternative | 1,030 | 97.3% | 0.874 | 0.811 | 88.2% | 73.2% | 9.2% | 24.1% |
| test | primary (chosen) | 3,075 | 90.5% | 0.903 | 0.799 | 88.1% | 75.5% | 3.0% | 15.0% |
| test | alternative | 3,075 | 97.3% | 0.901 | 0.807 | 88.0% | 75.4% | 9.3% | 21.8% |

</details>

<details>
<summary>Evaluation dates, hardware and commands</summary>

| Run | Date | Hardware | Command |
|---|---|---|---|
| `qwen2.5-0.5b-full-s13` | 2026-09-26T16:53:29+00:00 | Mac17,4, Apple M5, 24 GB, macOS 26.6.2 | `taskdistill eval --task banking77 --run qwen2.5-0.5b-full-s13` |
| `zero-shot-qwen2.5-0.5b` | 2026-09-26T18:00:33+00:00 | Mac17,4, Apple M5, 24 GB, macOS 26.6.2 | `taskdistill eval --task banking77 --zero-shot` |
| `qwen2.5-0.5b-full-s14` | 2026-09-26T16:58:08+00:00 | Mac17,4, Apple M5, 24 GB, macOS 26.6.2 | `taskdistill eval --task banking77 --run qwen2.5-0.5b-full-s14` |
| `qwen2.5-0.5b-full-s15` | 2026-09-26T17:02:12+00:00 | Mac17,4, Apple M5, 24 GB, macOS 26.6.2 | `taskdistill eval --task banking77 --run qwen2.5-0.5b-full-s15` |
| `qwen2.5-1.5b-full-s13` | 2026-09-26T17:13:55+00:00 | Mac17,4, Apple M5, 24 GB, macOS 26.6.2 | `taskdistill eval --task banking77 --run qwen2.5-1.5b-full-s13` |
| `qwen2.5-0.5b-full-s13-gold` | 2026-09-26T17:45:47+00:00 | Mac17,4, Apple M5, 24 GB, macOS 26.6.2 | `taskdistill eval --task banking77 --run qwen2.5-0.5b-full-s13-gold` |

</details>

The test split has been scored 8 times in this workspace (`test_access_log.jsonl`), every time by an evaluation listed in the table above; nothing was chosen with it.

## Operating point

|  | Value |
|---|---|
| Selected run | `qwen2.5-0.5b-full-s13` |
| Target | Agreement ≥ 97.0% against the teacher |
| Chosen threshold (on validation) | 0.8740 |
| Escalation rate, validation / test | 23.4% / 23.7% |
| Cascade quality, validation / test | 97.1% / 97.6% |
| Cascade − teacher, Agreement against the teacher (target) | -2.4 pts [-3.0, -1.9] |
| Cascade − teacher, Accuracy (gold) | +0.3 pts [-0.1, +0.8] |
| Target held on test | yes |
| Evaluated | 2026-09-26T16:53:29+00:00 on Mac17,4, Apple M5, 24 GB, macOS 26.6.2, `taskdistill eval --task banking77 --run qwen2.5-0.5b-full-s13` |

![Cascade quality against escalation rate](threshold_curve.png)

![Reliability diagram](reliability_test.png)

## Cost and latency

| System | $ / 1k requests | p50 ms | p95 ms | n | Source |
|---|---|---|---|---|---|
| Teacher only | $0.00593 | 594 | 1,242 | 3,075 | recorded live labelling calls, concurrency 8 |
| Student only | $0.0000783 | 44.5 | 68.2 | 3,075 | eval (flagged) |
| Cascade | $0.00151 | 49.6 | 892 | 3,075 | composed per request, 23.8% escalated |

Assumptions: local cost = 20 W x wall time x $0.3/kWh; hardware amortisation off. Teacher cost is the mean recorded `usage.cost` of the test requests; teacher latency is the recorded latency of the live labelling calls (cache hits and retried attempts excluded), recorded on 2026-09-26.

Student latency: in-process generation latency from `taskdistill eval` (flagged: not end-to-end through `serve`; run `taskdistill bench` against `serve --threshold 0` for the measured figure). Measured 2026-09-26T16:53:29+00:00 on Mac17,4, Apple M5, 24 GB, macOS 26.6.2, `taskdistill eval --task banking77 --run qwen2.5-0.5b-full-s13`.

Cascade latency is composed per test request as student latency plus the recorded teacher latency when the request escalates at the chosen threshold; cascade cost is the student energy plus the teacher cost of escalated requests.

At the list price without the provider's prompt cache (the same token counts), the teacher costs $0.0581 and the cascade $0.0139 per 1k requests.

## Break-even volume

**18,565 requests** = (labelling $0.0795 (ledger) + training energy $0.00251) / savings $0.00000442 per request.
At the list price without prompt caching: 1,858 requests (savings $0.0000442 per request; labelling cost as paid).

Assumptions: local cost = 20 W x wall time x $0.3/kWh; hardware amortisation off.

## Training

| Run | Base | Labels | Examples | Dropped (length) | Iterations | Epochs | Wall min | Peak GB | Tokens/s | Adapter MB |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5-0.5b-full-s13` | mlx-community/Qwen2.5-0.5B-Instruct-4bit | teacher | 8,874 | 0 | 2,219 | 2.00 | 25.1 | 3.45 | 668 | 35.2 |
| `qwen2.5-0.5b-full-s13-gold` | mlx-community/Qwen2.5-0.5B-Instruct-4bit | gold | 8,874 | 0 | 2,219 | 2.00 | 25.0 | 3.45 | 673 | 35.2 |
| `qwen2.5-0.5b-full-s14` | mlx-community/Qwen2.5-0.5B-Instruct-4bit | teacher | 8,874 | 0 | 2,219 | 2.00 | 28.2 | 3.38 | 592 | 35.2 |
| `qwen2.5-0.5b-full-s15` | mlx-community/Qwen2.5-0.5B-Instruct-4bit | teacher | 8,874 | 0 | 2,219 | 2.00 | 20.7 | 3.45 | 810 | 35.2 |
| `qwen2.5-1.5b-full-s13` | mlx-community/Qwen2.5-1.5B-Instruct-4bit | teacher | 8,874 | 0 | 2,219 | 2.00 | 59.1 | 6.16 | 280 | 73.9 |

## Dataset

| Split | Examples |
|---|---|
| train | 8,874 |
| valid | 1,030 |
| test | 3,075 |

<details>
<summary>Curation statistics</summary>

| Key | Value |
|---|---|
| task | banking77 |
| task_type | classification |
| date | 2026-09-26T14:26:16+00:00 |
| splits.train | 8,874 |
| splits.valid | 1,030 |
| splits.test | 3,075 |
| distribution.kind | label |
| pii.enabled | yes |
| pii.total | 0 |
| pii.examples_changed | 0 |
| dedupe.threshold | 0.9 |
| dedupe.exact | yes |
| dedupe.repeated_inputs | 5 |
| dedupe.removed_exact | 0 |
| dedupe.removed_near | 31 |
| dedupe.conflict_clusters | 0 |
| dedupe.conflict_examples | 0 |
| dedupe.gold_conflicts | 2 |
| cross_split.threshold | 0.9 |
| labelling.requested | 4,102 |
| labelling.mode | replay |
| labelling.max_usd | — |
| labelling.live | 0 |
| labelling.cached | 0 |
| labelling.replayed | 4,102 |
| labelling.cost_usd | $0 |
| labelling.recorded_cost_usd | $0.0246 |
| labelling.truncated | 0 |
| labelling.invalid | 12 |
| labelling.kept_invalid_test | 8 |
| labelling.labelled | 4,090 |
| labelling.date | 2026-09-26 |
| labelling.concurrency | — |
| length.max_seq_len | 512 |
| length.tokenizer | mlx-community/Qwen2.5-0.5B-Instruct-4bit |
| length.dropped | 0 |
| leakage.ok | yes |
| leakage.threshold | 0.9 |
| leakage.exact | 0 |
| leakage.near | 0 |
| captured_from | 2026-09-26 |
| captured_to | 2026-09-26 |
| teacher.model | deepseek/deepseek-v4.1-flash |
| teacher.provider | deepinfra/fp8 |
| teacher.prompt_sha256 | 0d8430f72f5be434d4d343651d41f46bd7a18fcc318b4c53db6b84417a4c8fca |
| teacher.temperature | 0 |
| teacher.max_tokens | 24 |
| teacher.mode | replay |
| student.base_model | mlx-community/Qwen2.5-0.5B-Instruct-4bit |
| student.system_prompt | Classify the customer's banking message into exactly one intent label. |
| student.max_seq_len | 512 |
| source.name | Banking77 (demo) |
| source.licence | CC BY 4.0 |
| files.train.jsonl | 42820af6f7d9662312a7cbe20d70eb062b29aa33d5c2c52af4c1465a5b0b922a |
| files.train.meta.jsonl | 4f3cf2cf42d3607339678da641778f78b76024ca3a6a2fb63a52851c76a7b4d6 |
| files.valid.jsonl | c7e0ccb6defdad5ecaedc57585d9cf19dfb66e31268661f154b2fbadb97debda |
| files.valid.meta.jsonl | d24f7fb25ab45468bdd9c31793bd0155616964104614c0aa4fad9a4914fda11f |
| files.test.jsonl | 65f671279e09db114d1a9419b2fe772571d4c4c67026e02bbed774d2a529efe2 |
| files.test.meta.jsonl | ddcf7363629552b73fc1fbd6f74693fb7cfb4dc99779f3e79a54fdaf9096d2d2 |
| files.labelling_keys.txt | 6c1d29048bb0408e16ee26f52d80202ad3c447d0f50dabfff71013143b5ec845 |
| elapsed_s | 5.734 |

</details>

The dataset card is written next to the curated data (`dataset_card.md`).

## Spend

| Phase | USD |
|---|---|
| bakeoff | $0.0211 |
| curate-label | $0.0246 |
| demo-capture | $0.0549 |
| record-fill | $0.0000881 |
| total (this task) | $0.101 |
| total (workspace) | $0.251 |
