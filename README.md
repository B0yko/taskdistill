<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="taskdistill" src="docs/assets/logo-light.svg" width="352">
</picture>

**Replace an expensive LLM API call on a narrow task with a small model on your Mac, and prove it with numbers.**

[![CI](https://github.com/B0yko/taskdistill/actions/workflows/ci.yml/badge.svg)](https://github.com/B0yko/taskdistill/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/taskdistill?color=ff9543)](https://pypi.org/project/taskdistill/)
[![Python](https://img.shields.io/pypi/pyversions/taskdistill?color=ff9543)](https://pypi.org/project/taskdistill/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX-0e1117?logo=apple&logoColor=white)](#limitations)
[![License](https://img.shields.io/badge/license-Apache--2.0-0e1117)](LICENSE)

[Quickstart](#quickstart) · [How it works](#how-it-works) · [Your own call](#use-it-on-your-own-call) · [Results](#results) · [Limitations](#limitations)

<!-- sync:headline -->
<table>
<tr><td align="center" width="33%"><h3>97.6%</h3>agreement with the teacher<br><sub>Banking77 test set, 23.7% of requests escalated</sub></td><td align="center" width="33%"><h3>30× faster</h3>student vs teacher API at p50<br><sub>19.9 ms vs 594 ms, Mac Studio M4 Max</sub></td><td align="center" width="33%"><h3>$0.25</h3>total API spend<br><sub>17,302 teacher calls in total</sub></td></tr>
</table>

<sub>Not every target held: on invoices the cascade missed its 97.0% target on the test set (layouts never seen in training), reaching 94.1%. Details in <a href="#results">Results</a>.</sub>
<!-- /sync -->

</div>

taskdistill captures the traffic your application already sends, curates it into training data, LoRA fine-tunes a
0.5B–1.5B Qwen2.5 student with MLX on Apple Silicon, evaluates it against the original model on held-out data, and
serves an OpenAI-compatible cascade: the student answers first and hands the request to the original model (the
teacher) when its confidence is below a threshold chosen on validation data. Your application changes only its base
URL.

![taskdistill demo banking77, then a request to the cascade server](docs/demo.svg)

## Why

Many production LLM calls are narrow: route a support message to one of N intents, pull eight fields out of an invoice
e-mail. They ship on a mid-tier API model because that is the fastest way to launch. At volume the call becomes a steady
cost and a latency floor (in this project the teacher answered in <!-- sync:teacher-latency -->about 594 ms at p50 and 1,242 ms at p95 (Banking77 labelling calls, 2026-09-26)<!-- /sync -->), and it
sends every input to a third party. A small fine-tuned model could take most of that traffic, but teams hold back for
three reasons: nobody has the logged data in trainable shape, nobody trusts the small model's quality, and nobody knows
which requests it will get wrong.

taskdistill is for the engineers who own such a call. It gives you a reproducible way to

1. collect training data from the traffic you already have (a capture proxy, or an import of existing logs),
2. measure the student against the teacher on held-out data, with baselines and confidence intervals, and
3. run a cascade whose quality/cost trade-off you pick explicitly, on validation data, instead of hoping for it.

### How it relates to other tools

<details>
<summary>OpenPipe, Predibase / LoRAX, RouteLLM, FrugalGPT, distilabel and LiteLLM, compared</summary>

| Project | What it does | What it does not do here |
|---|---|---|
| [OpenPipe](https://openpipe.ai) | Hosted fine-tuning of smaller models from logged requests; acquired by CoreWeave in 2025, its platform stopped new training and inference on 30 July 2026 and moved to Weights & Biases. | Not local; its open-source repository has been paused since 2024. |
| [Predibase](https://www.rubrik.com/company/newsroom/press-releases/25/rubrik-to-acquire-predibase-to-accelerate-agentic-ai-adoption) / [LoRAX](https://github.com/predibase/lorax) | Managed fine-tuning and serving (Predibase, acquired by Rubrik in 2025); LoRAX serves many LoRA adapters on one NVIDIA GPU. | LoRAX serves only; it needs an NVIDIA GPU on Linux and does not capture, train or evaluate. |
| [RouteLLM](https://github.com/lm-sys/RouteLLM) | Routers that send each query to a strong or a weak existing general-purpose model. | It does not train a task-specific student from your traffic. |
| [FrugalGPT](https://arxiv.org/abs/2305.05176) | Research method and code for a cascade over a sequence of paid LLM APIs. | It cascades between existing API models; no local student, no capture proxy. |
| [distilabel](https://github.com/argilla-io/distilabel) | Pipelines that generate synthetic data and AI feedback and emit datasets. | It neither trains nor serves models. |
| [LiteLLM](https://github.com/BerriAI/litellm) | Gateway exposing 100+ LLM APIs in the OpenAI format, with routing, cost tracking and logging; fine-tuning passes through to hosted APIs. | It does not build training sets from its logs or distil locally. |

</details>

The gap taskdistill fills: one local, open-source pipeline from captured traffic to a cascade with a
validation-chosen threshold on Apple Silicon, with an evaluation report you can reproduce offline. (Statements checked against each project's own site or
repository on 2026-09-26.)

## Quickstart

On an Apple Silicon Mac (macOS 14 or later) with [uv](https://docs.astral.sh/uv/getting-started/installation/)
installed. No API key is needed: without one, the demo replays the recorded teacher outputs that ship with the
package.

```bash
uvx --from git+https://github.com/B0yko/taskdistill taskdistill demo banking77
```

The same release is on PyPI: `uvx taskdistill demo banking77` runs it without cloning anything.

<!-- sync:quickstart-timing -->
Measured quick-profile demos in replay mode, each in a fresh workspace and directory (2026-09-27; `reports/demo_timing.json`):

| Machine | Demo | Cache | Install | Demo | Total | Load average before |
|---|---|---|---|---|---|---|
| MacBook Air, Apple M5, 24 GB | `banking77` | warm | 12.9 s | 2.4 min | 2.7 min | 1.90/1.98/2.02 |
| MacBook Air, Apple M5, 24 GB | `invoices` | warm | 0.2 s | 2.1 min | 2.1 min | 2.62/2.48/2.23 |
| Mac Studio, Apple M4 Max, 128 GB | `banking77` | warm | 3.1 s | 1.4 min | 1.5 min | 7.43/7.80/8.05 |
| Mac Studio, Apple M4 Max, 128 GB | `invoices` | warm | 0.2 s | 1.1 min | 1.1 min | 12.43/10.15/8.98 |
| Mac Studio, Apple M4 Max, 128 GB | `banking77` | cold | 7.1 s | 1.6 min | 1.7 min | 9.31/9.69/8.89 |

The cold run (empty `uv` cache and Hugging Face cache on the Mac Studio, Apple M4 Max, 128 GB) took 1.7 min: 7.1 s to install and 1.6 min for the demo, including the 0.29 GB base-model download. Every measured run finished in under five minutes; slower Macs and slower connections will take longer.
<!-- /sync:quickstart-timing -->

The demo runs the whole pipeline on [Banking77](#data-and-licences) and leaves a trained student, a report in
`reports/banking77/` and a `request.json` in the current directory. Serve the cascade and call it:

```bash
uvx --from git+https://github.com/B0yko/taskdistill taskdistill serve --task banking77
```

```bash
curl -s -D - http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
```

The response is a standard `chat.completion`; the `x-taskdistill-route` header says whether the student or the teacher
answered and `x-taskdistill-confidence` carries the student's confidence. `taskdistill demo invoices` does the same for
JSON extraction on synthetic invoices. Both demos use the `quick` profile, sized to finish in a few minutes:
Banking77 trains on 2,000 examples (300 validation, 500 test, 200 iterations) and invoices on 120 documents (24
validation, 36 test, 6 per layout, 30 iterations), cut down from 400/60/100 so the demo stays within the time
budget. Use `--profile full` for the configuration behind the numbers below (it takes tens of minutes to hours on a
laptop).

## How it works

```mermaid
flowchart LR
  subgraph today[Your application today]
    A[App] -->|base URL| P[capture proxy]
    P --> T[(teacher API)]
  end
  P --> S[(SQLite store)]
  S --> C[curate] --> D[data/*.jsonl] --> TR[train: LoRA with MLX] --> AD[adapter]
  AD --> E[eval]
  E -->|validation: run, threshold, calibration| TH[selected_run.json + threshold.json]
  E -->|test: report only| R[report]
  subgraph cascade[Your application after]
    A2[App] -->|base URL| CS[cascade server]
    CS -->|confidence >= t| ST[student on the Mac]
    CS -->|confidence < t| T2[(teacher API)]
  end
  TH --> CS
  CS --> S
  S --> R
```

| Step | Command | What it does |
|---|---|---|
| Capture | `taskdistill capture --task T` | OpenAI-compatible reverse proxy: forwards `POST /chat/completions` unchanged with the client's own `Authorization`, and stores request and response bodies, usage, latency and status. It never stores a header. `--import` loads existing logs instead. |
| Curate | `taskdistill curate --task T` | Extracts the task input, merges records by input hash (this is how gold labels reach captured inputs), normalises teacher outputs, scrubs PII, assigns splits, removes duplicates and near-duplicates inside and across splits, labels unlabelled inputs with the teacher, drops (never truncates) over-long examples, and fails unless an independent leakage check finds zero duplicates across splits. |
| Train | `taskdistill train --task T` | LoRA on the 4-bit base with mlx-lm, loss on completion tokens only, keeping the checkpoint with the best validation loss. `--backend torch` uses transformers + PEFT instead. |
| Eval | `taskdistill eval --task T --select` | Chooses the run and the cascade threshold on validation, then scores student, teacher, cascade and baselines on test with paired bootstrap confidence intervals and calibration metrics. |
| Serve | `taskdistill serve --task T` | FastAPI server with `/v1/chat/completions`, `/v1/models`, `/healthz` and `/metrics`. Low-confidence, unsupported and unparsable requests go to the teacher with the teacher's key, never the client's. |
| Report | `taskdistill report --task T` | Quality, operating point, $/1k requests, latency and break-even volume, optionally from real served traffic (`--from-serve-log`). |

**Confidence.** Classification decodes greedily under a trie of the canonical labels, so every answer is a valid
label; confidence is the product of the renormalised probabilities of the chosen tokens (the end token included, so a
label that is a prefix of another is handled). Extraction generates JSON freely; each field's confidence is the
product of the probabilities of the tokens spanning its value, and the document's confidence is the minimum over
fields (0 for invalid JSON or a schema violation).

**Selection on validation only.** Every choice (base model, seed, checkpoint, threshold, isotonic calibration,
baseline hyperparameters) is made by a function that accepts only a `ValidationSplit`; passing a `TestSplit` raises
`TypeError`. The threshold is the one with the lowest escalation rate that meets the target on validation, and the
test split then reports whether the target held.

## Use it on your own call

Say your application classifies support tickets with a paid API model.

1. **Install the command and create a task spec.**

   ```bash
   uv tool install git+https://github.com/B0yko/taskdistill
   ```

   ```bash
   taskdistill init tickets --type classification
   ```

   Edit `tasks/tickets/teacher_prompt.md` (the system prompt your application sends today), `labels.txt`, and in
   `task.yaml` the teacher's `model`, `base_url` and, for OpenRouter, the pinned provider in `extra_body`.
   Everything below runs in the directory that contains `tasks/`; the workspace defaults to `./.taskdistill`.

2. **Capture traffic.** Start the proxy and change only the base URL of your OpenAI client:

   ```bash
   export TASKDISTILL_TEACHER_BASE_URL=https://openrouter.ai/api/v1   # where the proxy forwards
   taskdistill capture --task tickets
   ```

   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:8787/t/tickets/v1", api_key=YOUR_KEY)  # the key is passed through
   ```

   Or import logs you already have (`--format openai` for request/response pairs, `pairs` for input/output,
   `inputs` for unlabelled inputs that the teacher will label). Gold labels, if you have any, are imported with
   `--format inputs` and joined to captured traffic by input hash:

   ```bash
   taskdistill capture --task tickets --import gold.jsonl --format inputs
   ```

3. **Curate, train and evaluate.** Curate prints a projected cost from a 50-request sample before labelling anything
   and asks for `--yes` above $0.50; `--max-usd` caps the run.

   ```bash
   export TASKDISTILL_TEACHER_API_KEY=...    # or OPENROUTER_API_KEY
   taskdistill curate --task tickets --max-usd 2
   taskdistill train --task tickets
   taskdistill eval --task tickets --select
   ```

   Without gold labels, eval reports agreement with the teacher, and the default cascade target (97% agreement) needs
   no gold at all.

4. **Serve the cascade** and point your client at it. The model string your application sends can stay as it is.

   ```bash
   taskdistill serve --task tickets
   ```

   ```python
   client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
   ```

5. **Watch it.** `taskdistill report --task tickets --from-serve-log --since 24h` recomputes the figures from real
   served traffic, including the observed escalation rate; a rate far above the validation estimate means the traffic
   has drifted away from the training data.

<!-- sync:results -->
## Results

Numbers below come from `scripts/reproduce.sh` (full profile, recorded teacher outputs) on 2026-09-27 on a MacBook Air, Apple M5, 24 GB (Mac17,4), macOS 26.6.2, plus one extra zero-shot evaluation (the invoices table also shows the 1.5B zero-shot base); teacher outputs were recorded on 2026-09-26. The test splits were scored 8 times (Banking77) and 9 times (invoices), each time by an evaluation shown in these tables; no choice used them: every choice (base model, seed, threshold, isotonic calibration) was made on the validation split alone. Figures marked “recorded” below (the teacher and cascade rows) replay the teacher outputs captured then, not a live call. The live bench and latency numbers are dated measurements and are not expected to reproduce exactly on different hardware or under different load. Unless noted otherwise, every score has a 95% paired bootstrap interval in brackets (1,000 resamples).

Two machines were used. Quality, calibration and training numbers come from the MacBook Air, Apple M5, 24 GB that trained the students; it is fanless and shared with other work. Cost, latency, the live bench and the cross-machine reproduction come from a Mac Studio, Apple M4 Max, 128 GB, used only for those runs. Each table names its machine.

**Bottom line.** Each cascade's threshold was chosen on validation; on test:

- **Banking77.** Target held on test: yes — Agreement against the teacher was 97.6% on test (target 97.0%) at a 23.7% escalation rate, close to the 97.1% on validation at 23.4%.
- **Invoices.** Target held on test: no — Agreement against the teacher met the 97.0% target on validation (98.5% at 0.0% escalation) but fell to 94.1% on test (1.7% escalation).

Metrics: **agreement** is how often a system gives the teacher's answer (for invoices, field micro-F1 against the teacher's JSON); **accuracy** and **macro-F1** (F1 averaged over the 77 intents, each counted equally) are against the gold labels; **ECE** (expected calibration error, lower is better) is the average gap between the student's stated confidence and how often it is right; **AUROC** (0.5 = chance, 1.0 = perfect) is how well that confidence separates right answers from wrong ones. For invoices, **JSON validity** is the share of outputs that parse as a JSON object, **field micro-F1** and **field EM** (exact match) score the 8 fields against the gold, and **Doc EM** is the share of documents with all 8 fields right.

### Banking77 (77 intents)

Test split, n = 3,075. Cells are the score and its 95% paired bootstrap interval; a row aggregating more than one seed shows mean ± sample standard deviation only (see the per-seed table below for each seed's own score, and interval where the eval recorded one). ECE and AUROC use raw (pre-isotonic) student confidence.

| System | n | Accuracy | Macro-F1 | Agreement | ECE | AUROC |
|---|---|---|---|---|---|---|
| teacher (deepseek/deepseek-v4.1-flash) | 3,075 | 75.8% [74.2, 77.4] | 74.9% [73.0, 76.1] | reference | — | — |
| base 0.5B zero-shot (labels in the prompt) | 3,075 | 23.0% [21.6, 24.5] | 20.5% [19.0, 21.6] | 25.6% [24.1, 27.0] | 35.3% [33.9, 36.7] | 0.738 [0.718, 0.761] |
| TF-IDF + logistic regression (teacher labels) | 3,075 | 72.5% [70.8, 74.0] | 71.4% [69.7, 72.5] | 83.5% [82.2, 84.8] | — | — |
| student qwen2.5-0.5b (teacher labels, 3 seeds) | 3,075 | 75.7% ± 0.5 | 74.6% ± 0.7 | 88.1% ± 0.2 | 15.2% ± 0.3 | 0.801 ± 0.003 |
| student qwen2.5-1.5b (teacher labels) | 3,075 | 76.5% [74.9, 78.0] | 75.5% [73.9, 76.7] | 89.2% [88.1, 90.4] | 15.4% [14.1, 16.9] | 0.797 [0.779, 0.814] |
| student qwen2.5-0.5b (gold labels) | 3,075 | 92.0% [91.0, 92.9] | 92.0% [91.0, 92.9] | 75.8% [74.2, 77.4] | 1.4% [1.0, 2.4] | 0.915 [0.897, 0.931] |
| cascade (qwen2.5-0.5b-full-s13, t = 0.874) | 3,075 | 76.1% [74.5, 77.7] | 75.0% [73.3, 76.2] | 97.6% [97.0, 98.1] | — | — |

<details>
<summary>Per-seed scores</summary>

Per-seed scores for `student qwen2.5-0.5b (teacher labels, 3 seeds)` (selected run: `qwen2.5-0.5b-full-s13`):

| Run | Seed | Accuracy | Macro-F1 | Agreement | ECE | AUROC |
|---|---|---|---|---|---|---|
| `qwen2.5-0.5b-full-s13` (validation-selected) | 13 | 75.5% [73.8, 77.0] | 74.4% [72.7, 75.6] | 88.1% [86.9, 89.3] | 15.0% [13.7, 16.5] | 0.799 [0.781, 0.816] |
| `qwen2.5-0.5b-full-s14` | 14 | 75.3% | 74.1% | 87.9% | 15.6% | 0.801 |
| `qwen2.5-0.5b-full-s15` | 15 | 76.3% | 75.4% | 88.2% | 15.0% | 0.804 |

</details>

|  | Value |
|---|---|
| Target | Agreement ≥ 97.0% against the teacher |
| Chosen threshold (on validation) | 0.8740 |
| Escalation rate, valid / test | 23.4% / 23.7% |
| Cascade − teacher, Agreement against the teacher (target) | -2.4 pts [-3.0, -1.9] |
| Cascade − teacher, Accuracy (gold) | +0.3 pts [-0.1, +0.8] |
| Target held on test | yes |

Target held on test: yes — Agreement against the teacher was 97.6% on test (target 97.0%) at a 23.7% escalation rate, close to the 97.1% on validation at 23.4%.

![Banking77 cascade quality against escalation rate](reports/banking77/threshold_curve.png)

### Invoices (8-field JSON extraction)

Test split, n = 600. The test set is 6 layouts never seen in training, so the effective sample is 6 layouts; the template-cluster bootstrap below says how wide that makes the uncertainty. Cells are the score and its 95% paired bootstrap interval (document-level). A row aggregating more than one seed shows mean ± sample standard deviation only (see the per-seed table below for each seed's own score, and interval where the eval recorded one).

| System | n | JSON validity | Field micro-F1 | Field EM | Doc EM | Agreement | ECE | AUROC |
|---|---|---|---|---|---|---|---|---|
| teacher (deepseek/deepseek-v4-flash-0731) | 600 | 100.0% [100.0, 100.0] | 99.5% [99.3, 99.7] | 99.5% [99.3, 99.7] | 96.0% [94.5, 97.5] | reference | — | — |
| base 0.5B zero-shot (schema in the prompt) | 600 | 42.3% [38.7, 46.2] | 37.0% [34.3, 40.0] | 24.8% [22.5, 27.4] | 1.8% [0.8, 3.0] | 36.9% [34.1, 39.9] | 15.9% [14.2, 17.8] | 0.770 [0.717, 0.824] |
| base 1.5B zero-shot | 600 | 62.3% [58.2, 66.2] | 73.3% [70.2, 76.1] | 58.9% [54.9, 62.6] | 40.3% [36.3, 44.3] | 73.3% [70.2, 76.1] | 9.7% [8.2, 11.5] | 0.981 [0.970, 0.989] |
| student qwen2.5-0.5b (teacher labels, 3 seeds) | 600 | 97.1% ± 4.3 | 87.2% ± 4.4 | 85.9% ± 4.7 | 39.2% ± 21.2 | 86.8% ± 4.5 | 15.6% ± 13.3 | 0.874 ± 0.113 |
| student qwen2.5-1.5b (teacher labels) | 600 | 99.2% [98.3, 99.8] | 94.1% [93.3, 94.9] | 94.0% [93.0, 94.9] | 67.5% [63.8, 71.3] | 93.5% [92.6, 94.4] | 14.1% [11.9, 17.6] | 0.846 [0.813, 0.879] |
| student qwen2.5-1.5b (gold labels) | 600 | 95.0% [93.0, 96.7] | 93.8% [92.5, 94.9] | 91.4% [89.3, 93.1] | 78.2% [74.8, 81.5] | 93.5% [92.1, 94.6] | 7.8% [5.8, 10.6] | 0.902 [0.869, 0.931] |
| cascade (qwen2.5-1.5b-full-s13, t = 0.1524) | 600 | 100.0% [100.0, 100.0] | 94.6% [93.9, 95.3] | 94.9% [94.3, 95.6] | 69.2% [65.3, 72.8] | 94.1% [93.3, 94.9] | — | — |

<details>
<summary>Per-seed scores</summary>

Per-seed scores for `student qwen2.5-0.5b (teacher labels, 3 seeds)` (selected run: `qwen2.5-0.5b-full-s13`):

| Run | Seed | JSON validity | Field micro-F1 | Field EM | Doc EM | Agreement | ECE | AUROC |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5-0.5b-full-s13` (validation-selected) | 13 | 100.0% [100.0, 100.0] | 90.9% [89.9, 91.8] | 91.3% [90.4, 92.2] | 55.3% [51.3, 59.3] | 90.4% [89.3, 91.4] | 8.4% [6.7, 11.8] | 0.913 [0.888, 0.935] |
| `qwen2.5-0.5b-full-s14` | 14 | 92.2% | 88.5% | 84.2% | 47.2% | 88.3% | 7.3% | 0.962 |
| `qwen2.5-0.5b-full-s15` | 15 | 99.2% | 82.3% | 82.3% | 15.2% | 81.8% | 31.0% | 0.747 |

</details>

<details>
<summary>Template-cluster bootstrap over 6 layouts</summary>

Template-cluster bootstrap over 6 groups (one per layout): 95% intervals.

| System | JSON validity | Field micro-F1 | Field EM | Doc EM | Agreement | ECE | AUROC |
|---|---|---|---|---|---|---|---|
| teacher (deepseek/deepseek-v4-flash-0731) | [100.0, 100.0] | [98.4, 100.0] | [98.5, 100.0] | [88.0, 100.0] | [100.0, 100.0] | — | — |
| base 0.5B zero-shot (schema in the prompt) | [29.0, 55.7] | [23.9, 50.3] | [15.4, 36.0] | [0.0, 5.2] | [23.5, 50.3] | [11.1, 21.3] | [0.662, 0.916] |
| base 1.5B zero-shot | [32.6, 86.8] | [46.7, 89.1] | [30.4, 82.3] | [19.7, 58.8] | [46.7, 89.1] | [5.0, 14.5] | [0.967, 0.993] |
| student qwen2.5-0.5b (teacher labels, 3 seeds) | [100.0, 100.0] | [82.3, 96.3] | [83.2, 96.4] | [30.7, 73.3] | [80.8, 96.3] | [5.0, 19.5] | [0.873, 0.943] |
| student qwen2.5-1.5b (teacher labels) | [97.8, 100.0] | [87.1, 98.5] | [86.9, 98.4] | [38.7, 89.3] | [85.5, 98.5] | [4.2, 41.6] | [0.696, 0.977] |
| student qwen2.5-1.5b (gold labels) | [87.3, 99.0] | [83.2, 98.8] | [77.8, 98.3] | [47.0, 94.7] | [82.1, 98.8] | [4.1, 28.7] | [0.871, 0.991] |
| cascade (qwen2.5-1.5b-full-s13, t = 0.1524) | [100.0, 100.0] | [88.3, 98.7] | [89.1, 98.7] | [41.0, 89.8] | [86.7, 98.7] | — | — |

</details>

<details>
<summary>Per-template scores (student, teacher, cascade)</summary>

student:

| Template | n | JSON validity | Field micro-F1 | Field EM | Doc EM | Agreement |
|---|---|---|---|---|---|---|
| email-13 | 100 | 100.0% | 99.2% | 99.2% | 94.0% | 99.2% |
| email-14 | 100 | 100.0% | 95.9% | 96.1% | 69.0% | 95.9% |
| email-15 | 100 | 100.0% | 98.8% | 98.9% | 91.0% | 98.8% |
| layout-13 | 100 | 99.0% | 98.8% | 98.4% | 94.0% | 98.8% |
| layout-14 | 100 | 96.0% | 77.1% | 76.9% | 0.0% | 73.8% |
| layout-15 | 100 | 100.0% | 94.2% | 94.5% | 57.0% | 94.2% |

teacher:

| Template | n | Field micro-F1 | Field EM | Doc EM |
|---|---|---|---|---|
| email-13 | 100 | 100.0% | 100.0% | 100.0% |
| email-14 | 100 | 100.0% | 100.0% | 100.0% |
| email-15 | 100 | 100.0% | 100.0% | 100.0% |
| layout-13 | 100 | 100.0% | 100.0% | 100.0% |
| layout-14 | 100 | 96.8% | 97.0% | 76.0% |
| layout-15 | 100 | 100.0% | 100.0% | 100.0% |

cascade:

| Template | n | JSON validity | Field micro-F1 | Field EM | Doc EM | Agreement |
|---|---|---|---|---|---|---|
| email-13 | 100 | 100.0% | 99.2% | 99.2% | 94.0% | 99.2% |
| email-14 | 100 | 100.0% | 96.0% | 96.2% | 70.0% | 96.0% |
| email-15 | 100 | 100.0% | 98.8% | 98.9% | 91.0% | 98.8% |
| layout-13 | 100 | 100.0% | 99.3% | 99.4% | 95.0% | 99.3% |
| layout-14 | 100 | 100.0% | 79.5% | 80.9% | 4.0% | 76.2% |
| layout-15 | 100 | 100.0% | 94.7% | 95.0% | 61.0% | 94.7% |

</details>

<details>
<summary>Per-field exact match</summary>

Against the gold:

| Field | student | teacher | cascade |
|---|---|---|---|
| `vendor_name` | 88.7% | 96.0% | 90.2% |
| `invoice_number` | 83.0% | 100.0% | 83.8% |
| `invoice_date` | 94.8% | 100.0% | 95.7% |
| `due_date` | 88.8% | 100.0% | 89.8% |
| `currency` | 99.2% | 100.0% | 100.0% |
| `total_amount` | 99.2% | 100.0% | 100.0% |
| `tax_amount` | 99.2% | 100.0% | 100.0% |
| `po_number` | 99.2% | 100.0% | 100.0% |

Against the teacher:

| Field | student | cascade |
|---|---|---|
| `vendor_name` | 84.7% | 86.2% |
| `invoice_number` | 83.0% | 83.8% |
| `invoice_date` | 94.8% | 95.7% |
| `due_date` | 88.8% | 89.8% |
| `currency` | 99.2% | 100.0% |
| `total_amount` | 99.2% | 100.0% |
| `tax_amount` | 99.2% | 100.0% |
| `po_number` | 99.2% | 100.0% |

</details>

<details>
<summary>Per-trait breakdown</summary>

student:

| Trait | n | JSON validity | Field micro-F1 | Field EM | Doc EM | Agreement |
|---|---|---|---|---|---|---|
| (none) | 91 | 98.9% | 94.3% | 93.8% | 70.3% | 93.4% |
| distractor_amounts | 306 | 99.0% | 93.9% | 93.8% | 67.6% | 93.4% |
| eu_number_format | 156 | 98.7% | 95.0% | 94.7% | 73.1% | 94.7% |
| label_typos | 59 | 98.3% | 94.5% | 93.6% | 71.2% | 94.3% |
| missing_optional | 177 | 99.4% | 92.5% | 93.6% | 63.3% | 91.8% |
| net_terms_due | 93 | 97.8% | 87.8% | 87.5% | 35.5% | 87.1% |
| quoted_reply_chain | 126 | 100.0% | 94.4% | 94.7% | 68.3% | 94.2% |

teacher:

| Trait | n | Field micro-F1 | Field EM | Doc EM |
|---|---|---|---|---|
| (none) | 91 | 99.0% | 99.0% | 92.3% |
| distractor_amounts | 306 | 99.6% | 99.6% | 96.7% |
| eu_number_format | 156 | 99.7% | 99.7% | 97.4% |
| label_typos | 59 | 99.8% | 99.8% | 98.3% |
| missing_optional | 177 | 99.4% | 99.5% | 96.0% |
| net_terms_due | 93 | 99.3% | 99.3% | 94.6% |
| quoted_reply_chain | 126 | 99.8% | 99.8% | 98.4% |

cascade:

| Trait | n | JSON validity | Field micro-F1 | Field EM | Doc EM | Agreement |
|---|---|---|---|---|---|---|
| (none) | 91 | 100.0% | 94.9% | 94.9% | 71.4% | 94.0% |
| distractor_amounts | 306 | 100.0% | 94.5% | 94.8% | 69.0% | 94.0% |
| eu_number_format | 156 | 100.0% | 95.8% | 96.1% | 75.0% | 95.5% |
| label_typos | 59 | 100.0% | 95.1% | 95.3% | 72.9% | 94.8% |
| missing_optional | 177 | 100.0% | 92.9% | 94.3% | 65.0% | 92.3% |
| net_terms_due | 93 | 100.0% | 89.6% | 90.2% | 41.9% | 88.9% |
| quoted_reply_chain | 126 | 100.0% | 94.4% | 94.7% | 68.3% | 94.2% |

</details>

|  | Value |
|---|---|
| Target | Agreement ≥ 97.0% against the teacher |
| Chosen threshold (on validation) | 0.1524 |
| Escalation rate, valid / test | 0.0% / 1.7% |
| Cascade − teacher, Agreement against the teacher (target) | -5.9 pts [-6.7, -5.1], cluster [-13.3, -1.3] |
| Cascade − teacher, Field micro-F1 (gold) | -4.8 pts [-5.5, -4.2], cluster [-10.1, -1.3] |
| Target held on test | no |

Target held on test: no — Agreement against the teacher met the 97.0% target on validation (98.5% at 0.0% escalation) but fell to 94.1% on test (1.7% escalation).

![Invoices cascade quality against escalation rate](reports/invoices/threshold_curve.png)

### Cost and latency

Energy assumptions: local cost = 20 W x wall time x $0.3/kWh; hardware amortisation off.

**Banking77**

Measured on a Mac Studio, Apple M4 Max, 128 GB (Mac16,9), macOS 26.5.2, 2026-09-27:

| System | $/1k recorded | $/1k list price | p50 ms | p95 ms | Source |
|---|---|---|---|---|---|
| Teacher only | $0.00593 | $0.0581 | 594 | 1,242 | recorded live labelling calls at concurrency 8 (cache hits and retries excluded) |
| Student only | $0.0000351 | — | 19.9 | 31.7 | bench against `serve --threshold 0` |
| Cascade | $0.00136 | $0.0129 | 21.5 | 854 | composed per request, 22.0% escalated |

For comparison, the MacBook Air's student latency was 44.5 ms p50 / 68.2 ms p95 (in-process eval, not through the server).

Break-even at 17,419 requests at the recorded teacher cost, 1,762 requests at the list price without prompt caching.

**Invoices**

Measured on a Mac Studio, Apple M4 Max, 128 GB (Mac16,9), macOS 26.5.2, 2026-09-27:

| System | $/1k recorded | $/1k list price | p50 ms | p95 ms | Source |
|---|---|---|---|---|---|
| Teacher only | $0.0462 | $0.0577 | 1,354 | 2,427 | recorded live labelling calls at concurrency 8 (cache hits and retries excluded) |
| Student only | $0.000729 | — | 441 | 498 | bench against `serve --threshold 0` |
| Cascade | $0.00119 | $0.00130 | 442 | 499 | composed per request, 1.0% escalated |

For comparison, the MacBook Air's student latency was 1,391 ms p50 / 1,988 ms p95 (in-process eval, not through the server).

Break-even at 3,140 requests at the recorded teacher cost, 2,505 requests at the list price without prompt caching.

The Why section above states the teacher answered in about 594 ms at p50 and 1,242 ms at p95 (Banking77 labelling calls, 2026-09-26); that is this same recorded Banking77 teacher latency.

### Live bench cross-check

**Banking77** (measured on a Mac Studio, Apple M4 Max, 128 GB (Mac16,9), macOS 26.5.2, 2026-09-27)

| Mode | Run | Date | n | p50 ms | p95 ms | Escalated | Spend | Load average |
|---|---|---|---|---|---|---|---|---|
| Student only | qwen2.5-0.5b-full-s13 | 2026-09-27T02:51:24+00:00 | 300 | 19.9 | 31.7 | 0.0% | $0 | 15.01/12.64/10.21 |
| Cascade | qwen2.5-0.5b-full-s13 | 2026-09-27T02:52:20+00:00 | 300 | 31.3 | 815 | 21.3% | $0.000500 | 14.16/12.56/10.22 |

Composed (from the test split) vs measured: p50 21.5 vs 31.3 ms, p95 854 vs 815 ms.

**Invoices** (measured on a Mac Studio, Apple M4 Max, 128 GB (Mac16,9), macOS 26.5.2, 2026-09-27)

| Mode | Run | Date | n | p50 ms | p95 ms | Escalated | Spend | Load average |
|---|---|---|---|---|---|---|---|---|
| Student only | qwen2.5-1.5b-full-s13 | 2026-09-27T02:54:45+00:00 | 300 | 441 | 498 | 0.0% | $0 | 11.04/11.93/10.15 |
| Cascade | qwen2.5-1.5b-full-s13 | 2026-09-27T02:57:14+00:00 | 300 | 445 | 505 | 1.3% | $0.000199 | 9.45/10.91/10.00 |

Composed (from the test split) vs measured: p50 442 vs 445 ms, p95 499 vs 505 ms.

### Training on the MacBook Air

**Banking77**

| Run | Base | Examples | Dropped (length) | Iterations | Epochs | Wall min | Peak GB | Tokens/s | Adapter MB |
|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5-0.5b-full-s13` | mlx-community/Qwen2.5-0.5B-Instruct-4bit | 8,874 | 0 | 2,219 | 2.00 | 25.1 | 3.45 | 668 | 35.2 |
| `qwen2.5-1.5b-full-s13` | mlx-community/Qwen2.5-1.5B-Instruct-4bit | 8,874 | 0 | 2,219 | 2.00 | 59.1 | 6.16 | 280 | 73.9 |

**Invoices**

| Run | Base | Examples | Dropped (length) | Iterations | Epochs | Wall min | Peak GB | Tokens/s | Adapter MB |
|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5-0.5b-full-s13` | mlx-community/Qwen2.5-0.5B-Instruct-4bit | 2,000 | 0 | 500 | 2.00 | 34.1 | 4.17 | 792 | 35.2 |
| `qwen2.5-1.5b-full-s13` | mlx-community/Qwen2.5-1.5B-Instruct-4bit | 2,000 | 0 | 500 | 2.00 | 96.7 | 5.20 | 276 | 73.9 |

Load average recorded alongside these runs ranged up to 4.27. The MacBook Air is fanless and can throttle under sustained load; that is why the load average is recorded next to every timing rather than assumed away.

### Reproducibility on a second machine

`scripts/reproduce.sh` (full profile) was run again on a second machine and compared with `scripts/compare_reports.py`; tolerance 2.0 points on each test-split metric.

Reference: MacBook Air, Apple M5, 24 GB (Mac17,4), macOS 26.6.2, 2026-09-27. Rerun: Mac Studio, Apple M4 Max, 128 GB (Mac16,9), macOS 26.5.2, 2026-09-27.

| Task | Max abs difference | All rows within tolerance |
|---|---|---|
| Banking77 | 0.78 pts | yes |
| Invoices | 10.17 pts | no |

<details>
<summary>Main metric of every row, both machines</summary>

| Task | Row | Metric | Reference | Rerun | Diff |
|---|---|---|---|---|---|
| Banking77 | teacher (deepseek/deepseek-v4.1-flash) | Accuracy | 75.8% | 75.8% | +0.00 pts |
| Banking77 | TF-IDF + logistic regression (teacher labels) | Accuracy | 72.5% | 72.5% | +0.00 pts |
| Banking77 | student qwen2.5-0.5b (teacher labels, 3 seeds) | Accuracy | 75.7% | 75.9% | +0.25 pts |
| Banking77 | student qwen2.5-1.5b (teacher labels) | Accuracy | 76.5% | 76.3% | -0.16 pts |
| Invoices | teacher (deepseek/deepseek-v4-flash-0731) | Field micro-F1 | 99.5% | 99.5% | +0.00 pts |
| Invoices | student qwen2.5-0.5b (teacher labels, 3 seeds) | Field micro-F1 | 87.2% | 88.2% | +0.96 pts |
| Invoices | student qwen2.5-1.5b (teacher labels) | Field micro-F1 | 94.1% | 94.5% | +0.46 pts |
| Invoices | student qwen2.5-1.5b (gold labels) | Field micro-F1 | 93.8% | 91.5% | -2.37 pts |

</details>

Banking77: the selected run differed — `qwen2.5-0.5b-full-s13` on the reference machine vs `qwen2.5-1.5b-full-s13` on the rerun (reference: “large gains 0.97 points, below the 1.00-point minimum”; rerun: “large gains 1.26 points (>= 1.00) at 1.87x the small p95 (< 3x)”). Invoices: the selected run matched on both machines (`qwen2.5-1.5b-full-s13`).

All rows within tolerance across every task: no.

Outside the tolerance:

- Invoices student qwen2.5-0.5b (teacher labels, 3 seeds), Doc EM: 39.2% vs 41.6% (+2.33 points)
- Invoices student qwen2.5-1.5b (teacher labels), Doc EM: 67.5% vs 74.3% (+6.83 points)
- Invoices student qwen2.5-1.5b (gold labels), JSON validity: 95.0% vs 98.2% (+3.17 points)
- Invoices student qwen2.5-1.5b (gold labels), Field micro-F1: 93.8% vs 91.5% (-2.37 points)
- Invoices student qwen2.5-1.5b (gold labels), Doc EM: 78.2% vs 68.0% (-10.17 points)
- Invoices student qwen2.5-1.5b (gold labels), Agreement: 93.5% vs 91.0% (-2.43 points)

### Calibration

![Banking77 reliability diagram, raw vs isotonic](reports/banking77/reliability_test.png)

Two confidence definitions were compared on validation, before either was used at test time: the primary is the trie-constrained greedy label's own renormalised token-probability product; the alternative is free greedy generation scored by the mean per-token log-probability. AUROC is against the teacher and the gold label; ECE against each.

| Task | Split | Confidence | n | AUROC/teacher | AUROC/gold | ECE/teacher | ECE/gold |
|---|---|---|---|---|---|---|---|
| Banking77 | valid | primary (chosen) | 1,030 | 0.877 | 0.799 | 2.8% | 17.0% |
| Banking77 | valid | alternative | 1,030 | 0.874 | 0.811 | 9.2% | 24.1% |
| Banking77 | test | primary (chosen) | 3,075 | 0.903 | 0.799 | 3.0% | 15.0% |
| Banking77 | test | alternative | 3,075 | 0.901 | 0.807 | 9.3% | 21.8% |
| Invoices | valid | primary (chosen) | 400 | 0.904 | 0.913 | 3.7% | 4.3% |
| Invoices | valid | alternative | 400 | 0.903 | 0.908 | 10.0% | 9.3% |
| Invoices | test | primary (chosen) | 600 | 0.846 | 0.846 | 14.1% | 14.1% |
| Invoices | test | alternative | 600 | 0.858 | 0.858 | 32.1% | 32.1% |

Banking77 student `qwen2.5-0.5b-full-s13`, ECE on test vs gold: 15.0% raw vs 3.7% after isotonic calibration.

Invoices student `qwen2.5-1.5b-full-s13`, ECE on test vs gold: 14.1% raw vs 15.6% after isotonic calibration.

### Choosing the base model

**Banking77**

| Base model | Validation metric | p95 ms |
|---|---|---|
| mlx-community/Qwen2.5-0.5B-Instruct-4bit | 88.2% | 74.8 |
| mlx-community/Qwen2.5-1.5B-Instruct-4bit | 89.1% | 163 |

Rule: use the larger base only if it gains at least 1 point on the validation metric and its p95 latency stays under 3.0x the smaller model's. Here the gain is below the 1 point minimum (+0.97 points) and the p95 ratio (2.18x) is under the 3.0x limit, so the smaller base was kept.

**Invoices**

| Base model | Validation metric | p95 ms |
|---|---|---|
| mlx-community/Qwen2.5-0.5B-Instruct-4bit | 94.7% | 1,102 |
| mlx-community/Qwen2.5-1.5B-Instruct-4bit | 98.5% | 1,974 |

Rule: use the larger base only if it gains at least 1 point on the validation metric and its p95 latency stays under 3.0x the smaller model's. Here the gain meets the 1 point minimum (+3.79 points) and the p95 ratio (1.79x) is under the 3.0x limit, so the larger base was selected.

The rule compares point estimates on the validation split (the best seed of each base); no interval is computed for this decision, so a gain close to the minimum can go either way on a rerun (see Reproducibility above).

### What didn't work

- Learning-rate schedule at the spec's peak rate (quick profile, 200 iterations, 3 seeds, Banking77): on the MacBook Air (Apple M5) a constant rate reached 32.1% ± 31.8% mean validation agreement with the teacher and linear warm-up + cosine decay 68.9% ± 8.3%, with 1 and 0 of 3 seeds diverging (agreement below 10%); on the Mac Studio (Apple M4 Max) a constant rate reached 23.3% ± 38.0% mean validation agreement with the teacher and linear warm-up + cosine decay 33.9% ± 32.9%, with 2 and 1 of 3 seeds diverging (agreement below 10%). Warm-up + cosine is the default and did better on average, but it does not make short runs at this peak rate reliable. In the full profile, Banking77 converged on every seed; on invoices, the best validation checkpoint of 1 of 3 on the MacBook Air (Apple M5) and 1 of 3 on the Mac Studio (Apple M4 Max) 0.5B seeds came at or before the end of warm-up, so those students stopped early.

- The bigger Banking77 student (1.5B vs 0.5B, teacher labels) gained +0.97 points on validation agreement, below the 1 point minimum the base-model rule requires, for 2.18x the p95 latency (limit 3.0x); the rule kept the 0.5B base.

- A Banking77 student trained on the gold labels reached 92.0% accuracy against gold, above both the same student trained on teacher labels (75.5%) and the teacher itself (75.8%); part of that gap is Banking77's own label noise (Ying and Thomas, 2022, flag about 14% of the training utterances as potential label errors), which caps how high any model's accuracy against gold can go.

- Teacher prompt variants (same 200 Banking77 validation queries, same teacher model): snake-case labels (+0.5 pts accuracy at 1.3x the prompt tokens, 3.3x the cost per 1k); labels with examples (+4.0 pts accuracy at 4.3x the prompt tokens, 2.9x the cost per 1k). The spec prompt was kept for the recorded run.

- Two confidence definitions were compared on validation, before either was used at test time: Banking77 AUROC 0.877 vs 0.874 (about equal), ECE 2.8% vs 9.2%, about 3.3x lower for the primary; Invoices AUROC 0.904 vs 0.903 (about equal), ECE 3.7% vs 10.0%, about 2.7x lower for the primary; so the primary (trie-constrained token-probability product) was kept over the alternative (free greedy generation, mean per-token log-probability). Reported honestly, on test: Banking77 AUROC 0.903 vs 0.901 (about equal), ECE 3.0% vs 9.3% (lower for the primary); Invoices AUROC 0.846 vs 0.858 (higher for the alternative), ECE 14.1% vs 32.1% (lower for the primary).

- Invoices: the threshold chosen on the validation layouts did not transfer to the unseen test layouts — Agreement against the teacher met the 97.0% target on validation (98.5%, 0.0% escalation) but fell to 94.1% on test (1.7% escalation).

### Spend and downloads

| Task | Teacher labelling | Bake-off | Live bench |
|---|---|---|---|
| Banking77 | $0.0796 | $0.0211 | $0.000500 |
| Invoices | $0.138 | $0.0116 | $0.000199 |

Total spend across all runs: $0.251 of a $14.00 global cap (17,302 teacher calls).

Model downloads: 1.17 GB, within the 1.5 GB budget (student base 0.5B: 0.29 GB; student base 1.5B: 0.88 GB).
<!-- /sync:results -->

## Configuration reference

A task lives in `tasks/<task>/task.yaml` next to its teacher prompt and its labels file or JSON Schema. String values
can use `${NAME}` or `${NAME:-default}` environment references. Unknown keys are errors, and every validation error
names the offending key.

<details>
<summary>Every <code>task.yaml</code> key</summary>

| Key | Default | Meaning |
|---|---|---|
| `task` | (required) | Task name: 1–64 letters, digits, `.`, `_` or `-`. Data lives in `$TASKDISTILL_HOME/<task>/`. |
| `type` | (required) | `classification` or `extraction`. |
| `labels_file` | — | Classification: one canonical label per line (required). |
| `schema_file` | — | Extraction: a JSON Schema with `type: object` and `properties` (required). |
| `input.from` | `last_user_message` | Where the task input is in each request: the last user message, or `regex`. |
| `input.regex` | `null` | With `from: regex`: a pattern over the last user message with a named group `input`. Requests that do not match go to the teacher (`input_unparsed`). |
| `teacher.base_url` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint. |
| `teacher.api_key_env` | `TASKDISTILL_TEACHER_API_KEY` | Environment variable holding the teacher key; falls back to `OPENROUTER_API_KEY`. |
| `teacher.model` | (required) | The slug your application calls today. |
| `teacher.prompt_file` | `teacher_prompt.md` | The teacher's system prompt, sent verbatim. |
| `teacher.temperature` | `0` | Teacher sampling temperature. |
| `teacher.max_tokens` | `256` | Teacher completion cap; outputs that hit it are flagged as truncated. |
| `teacher.response_format` | `null` | For example `{type: json_object}`, when the provider supports it. |
| `teacher.extra_body` | `{}` | Extra request fields: provider pinning (`provider: {order: [...], allow_fallbacks: false}`), reasoning controls. Recorded in the replay manifest, not in the request key. |
| `student.base_model` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | Hugging Face id or local directory of the student base; `--base` overrides it. |
| `student.system_prompt` | (required) | The student's short system prompt, used in training and serving. |
| `student.max_tokens` | `16` | Student generation cap. |
| `student.response_template` | `null` | Renders student answers and canonical escalations, e.g. `'{"intent": "{label}"}'` (literal replacement of `{label}`). |
| `train.profile` | `full` | `quick` caps training at 200 iterations and validates less often; `--profile` overrides it. |
| `train.lora_rank` | `16` | LoRA rank (scale 20, as mlx-lm's default). |
| `train.lora_layers` | `all` | Number of top transformer blocks to adapt, or `all`. |
| `train.learning_rate` | `1.0e-4` | Peak learning rate (Adam). |
| `train.batch_size` | `8` | Examples per step. |
| `train.epochs` | `2` | Passes over the training split. |
| `train.max_seq_len` | `512` | Longer examples are dropped by curate and training, never truncated. |
| `train.seed` | `13` | Training seed (`--seed` overrides it). |
| `train.lr_schedule` | `warmup_cosine` | Linear warm-up over 10% of the iterations (at most 100), then cosine decay to 10% of the peak; or `constant`. |
| `curate.dedupe.exact` | `true` | Exact duplicates on the normalised input (NFKC, whitespace-collapsed, case-folded). |
| `curate.dedupe.near_dup_jaccard` | `0.9` | MinHash LSH threshold on word 3-shingles (0.5–1). Inputs under 3 words are compared exactly. |
| `curate.pii.enabled` | `true` | Regex scrub with checksums before dedupe; matches become `<EMAIL>`, `<PHONE>`, `<IBAN>`, `<CARD>`, `<IPV4>`, `<SSN>`. |
| `curate.pii.kinds` | all six | Any subset of `email, phone, iban, card, ipv4, ssn`. |
| `curate.split.predefined` | `meta.split` | Meta field whose value (`train`, `valid`, `test`) fixes a row's split; `null` to ignore. |
| `curate.split.val`, `curate.split.test` | `0.1`, `0.1` | Fractions for rows without a predefined split. |
| `curate.split.stratify` | `true` | Stratify the random split by label (classification). |
| `curate.split.group_by` | `null` | Meta field whose groups stay in one split (for example `meta.template`); also the clusters of the cluster bootstrap. Without it, eval uses `meta.group` when present. |
| `curate.split.seed` | `13` | Split seed. |
| `cascade.reference` | `teacher` | What the target is measured against: `teacher` (works with no gold labels) or `gold`. |
| `cascade.metric` | `agreement` | `agreement` (label equality, or field micro-F1 against the teacher for extraction), `accuracy` or `macro_f1` (classification), `field_f1` (extraction). |
| `cascade.target` | — | Minimum cascade quality on validation, e.g. `0.97`. Set exactly one of `target` and `max_drop`. |
| `cascade.max_drop` | — | Maximum drop below the teacher's own score on the reference, usually with `reference: gold`. |
| `cascade.on_teacher_error` | `student` | On a teacher timeout or error: return the student's answer (`student-fallback`) or an HTTP 502 (`error`). A replay miss is always an error. `serve`'s escalation gives up quickly rather than retrying like a batch job: a 10 s HTTP timeout, at most 1 retry, `Retry-After` honoured up to 2 s, all within a 30 s deadline for the whole call (wait for a free connection included), so a hung or rate-limited teacher falls back within seconds, not minutes. |
| `cascade.escalation_response` | `canonical` | `canonical` normalises and renders the teacher's answer like a student answer; `raw` returns it verbatim. |
| `cost.local_watts` | `20` | Power draw assumed for local inference and training. |
| `cost.usd_per_kwh` | `0.30` | Electricity price. |
| `cost.hardware_usd`, `cost.amortisation_hours` | `0`, `0` | Hardware amortisation, applied only when both are above 0. |
| `budget.usd_cap` | `null` | Optional per-task spend cap; `TASKDISTILL_BUDGET_USD` always applies. |

</details>

**Environment** (see `.env.example`): `TASKDISTILL_TEACHER_BASE_URL`, `TASKDISTILL_TEACHER_API_KEY` (falls back to
`OPENROUTER_API_KEY`), `TASKDISTILL_TEACHER_MODEL`, `TASKDISTILL_BUDGET_USD` (global spend cap, default 5.00),
`TASKDISTILL_SERVER_TOKEN` (required to bind `capture` or `serve` beyond localhost) and `TASKDISTILL_HOME` (the
workspace: store, response cache, ledger, curated data and runs; default `./.taskdistill`).

**Spend control.** Every command that can call the teacher takes `--max-usd` (a cap for that run) and `--yes`. Before
each call the ledger reserves the worst case (UTF-8 bytes of the messages plus 16 per message as prompt tokens, plus
`max_tokens` of completion) and settles it with the `usage.cost` the API returns, or with the token counts times the
pricing snapshot when the response has no cost, so concurrent calls can never cross a cap. Batches first run a 50-request sample and need `--yes` when the projection exceeds $0.50. `taskdistill budget`
prints the spend by task and phase.

## Limitations

- **Platforms.** The MLX path needs Apple Silicon and macOS 14 or later (the minimum of the pinned `mlx` wheels). On
  other machines `train --backend mlx` stops with a message pointing to `--backend torch`. The torch path
  (transformers + PEFT, Unsloth when importable with CUDA) is tested on CPU with a tiny model in CI; **it has not been
  run on a GPU**, and the Unsloth branch is covered only by a dispatch test with a mock.
- **PII scrub is best-effort.** It is a regex scrub with checksums for e-mail addresses, phone numbers, IBANs, card
  numbers, IPv4 addresses and US SSNs. It does not find names, street addresses or anything else, and it is not an
  anonymisation guarantee. It also creates a train/serve skew: the student trains on placeholders such as `<EMAIL>`
  but sees raw text when it serves.
- **Training on Metal is not bit-for-bit deterministic.** Two runs with the same seed can differ slightly; the
  results report the spread over seeds and the tolerance used when checking that README numbers reproduce.
- **Scope.** Single-turn classification and JSON extraction only: no summaries or chat replies, no multi-turn or
  tool-calling tasks, no multi-label classification. Streamed responses are forwarded but not captured. The server
  generates one request at a time (no batching) and serves one task per process. Escalations for unsupported or
  unparsable requests are returned verbatim; only low-confidence escalations are normalised.
- **Evaluation limits.** The invoice test set is 600 documents from 6 layouts never seen in training, so its effective
  sample is 6 layouts; the template-cluster bootstrap intervals say how wide that makes the uncertainty. The invoices
  are synthetic and cleaner than real mail. Banking77's gold labels are noisy (Ying and Thomas, 2022, flag about 14%
  of the training utterances as potential label errors), which caps any model's measured accuracy against gold.
- **Cost figures are estimates where they are not measured.** Local cost is an energy estimate (20 W at $0.30/kWh by
  default, both configurable), not a power measurement. With an inexpensive teacher whose provider caches the shared
  prompt, the money saved per request is small; the main gains are latency and keeping inputs on the machine.
- **Latency numbers are dated measurements from two machines**, and each table names its machine. Training and the
  in-process student latency ran on a fanless MacBook Air (Apple M5, 24 GB) that throttles under sustained load and
  is shared with other work; the cost and latency tables, the live bench and the break-even volumes come from a Mac
  Studio (Apple M4 Max, 128 GB) used only for those runs. The load average is recorded next to every timing, and live
  numbers are not expected to reproduce exactly on other hardware or under other load.
- **Adapters are merged into the 4-bit base in memory** for evaluation and serving, which avoids the extra LoRA
  matrix multiplications an unmerged adapter runs at every decoding step. The re-quantisation shifts confidences
  slightly, which is why the threshold is chosen on the same merged model that serves.
- **Teacher terms.** Some commercial APIs forbid using their outputs to train other models. Check your provider's terms
  before distilling its outputs; see [ADR 5](docs/adr/0005-teacher-model-choice.md) for the checks made here.

## Roadmap

- Shadow mode and canary routing for the cascade, so a new student can be compared on live traffic before it answers.
- Scheduled retraining and active learning from escalated requests (the served log already records them).
- Drift alerts on the observed escalation rate (today `report --from-serve-log` shows drift; nothing alerts).
- Capture of streamed responses, multi-turn and tool-calling tasks.
- Constrained JSON decoding for extraction, and request batching in the server.

## Data and licences

- **taskdistill** is Apache-2.0 (Copyright 2026 Andrii Boiko). The synthetic invoice generator and its output are part
  of the repository and share that licence. A 20-document sample of the generated invoices (one per training layout,
  covering every trait, with their gold fields) is in [`examples/invoices/`](examples/invoices/); `scripts/make_examples.py`
  regenerates it.
- **Banking77** (Casanueva et al., 2020) is licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The demo downloads the CSVs at runtime from a pinned
  commit of [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)
  and checks their SHA-256. If GitHub is unavailable it falls back to the Parquet mirror `legacy-datasets/banking77`
  on the Hugging Face Hub (install the `hub` extra); `PolyAI/banking77` itself is a script-based Hub repository with no
  Parquet conversion. This repository does not redistribute the CSVs: it ships only the teacher's outputs keyed by
  request hash. Modifications: the official training set is split 90/10 into train and validation (stratified, seed
  13), and duplicates are removed as described above.

  ```bibtex
  @inproceedings{casanueva-etal-2020-efficient,
      title = "Efficient Intent Detection with Dual Sentence Encoders",
      author = "Casanueva, I{\~n}igo and Tem{\v{c}}inas, Tadas and Gerz, Daniela and Henderson, Matthew and Vuli{\'c}, Ivan",
      booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI",
      year = "2020",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2020.nlp4convai-1.5/",
      doi = "10.18653/v1/2020.nlp4convai-1.5",
      pages = "38--45"
  }
  ```
- **Student bases**: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and
  [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) are Apache-2.0; the demos use the 4-bit MLX
  conversions `mlx-community/Qwen2.5-0.5B-Instruct-4bit` and `mlx-community/Qwen2.5-1.5B-Instruct-4bit` at pinned
  revisions. No trained adapters are published.
- **Teacher outputs**: DeepSeek V4.1 Flash (Banking77) and DeepSeek V4 Flash 0731 (invoices) through OpenRouter, pinned
  to the DeepInfra provider. The DeepSeek V4 weights are MIT-licensed, DeepSeek's platform terms allow training other
  models on outputs, and DeepInfra's terms do not restrict the use of outputs
  ([ADR 5](docs/adr/0005-teacher-model-choice.md)). The outputs were recorded on 2026-09-26.

## Licence

Apache-2.0. See [LICENSE](LICENSE). Copyright 2026 Andrii Boiko.
