# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-27

First release.

### Added

- `taskdistill init` scaffolds a classification or extraction task spec (`task.yaml`, teacher prompt, labels or JSON
  Schema) from packaged templates; specs are validated with Pydantic and errors name the offending key.
- `taskdistill capture`: an OpenAI-compatible reverse proxy that forwards chat completions unchanged and stores request
  and response bodies (never headers) in SQLite; import and export of existing logs (`openai`, `pairs`, `inputs`).
- `taskdistill curate`: input extraction, merge by input hash, output normalisation, regex PII scrub with checksums,
  seeded stratified or grouped splits, exact and MinHash near-duplicate removal inside and across splits, budgeted
  teacher labelling, a length filter that drops rather than truncates, an independent leakage check and a dataset card.
- `taskdistill train`: LoRA on 4-bit Qwen2.5 bases with mlx-lm on Apple Silicon (completion-only loss, best-validation
  checkpoint, warm-up and cosine learning-rate schedule), and a transformers + PEFT path (`--backend torch`, Unsloth
  when available with CUDA) tested on CPU.
- Trie-constrained decoding for classification and span-mapped field confidence for extraction, on both backends.
- `taskdistill eval`: accuracy, macro-F1, teacher agreement, extraction field and document metrics, ECE, Brier, AUROC,
  isotonic calibration, paired and cluster bootstrap intervals, a TF-IDF baseline, zero-shot baselines, and run and
  threshold selection that accept only a `ValidationSplit`.
- `taskdistill serve`: an OpenAI-compatible cascade server with a single model worker thread, canonical or raw
  escalations, SSE, Prometheus metrics, token auth beyond localhost and a served-request log.
- `taskdistill report` (quality, operating point, cost, latency and break-even volume, also from served traffic) and
  `taskdistill bench` (live end-to-end latency through a running server).
- A cost ledger with worst-case reservations and global, per-task and per-run caps; `taskdistill budget`;
  `taskdistill pricing refresh`; `taskdistill teacher bakeoff` and `taskdistill teacher record`.
- Offline replay of recorded teacher outputs for both demos (`taskdistill demo banking77`, `taskdistill demo
  invoices`), with a manifest check and a hard error on a miss.
- Reports, ADRs and the scripts that regenerate every README number (`scripts/reproduce.sh`,
  `scripts/sync_readme.py`).
- A 20-document sample of the synthetic invoices under `examples/invoices/` (`scripts/make_examples.py`).
