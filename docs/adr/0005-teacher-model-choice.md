# ADR 0005: Teacher model choice

- Status: accepted
- Date: 2026-09-26

## Context

The demos need a teacher: an inexpensive, capable API model whose outputs the students learn from and whose
answers the cascade falls back to. Three constraints shape the choice.

1. **Quality per dollar.** The teacher labels about 13,000 Banking77 queries (each request carries a system
   prompt listing all 77 intents) and 3,000 invoice documents.
2. **No hidden reasoning.** With `max_tokens: 24` a thinking model is cut off mid-thought, and reasoning tokens
   would multiply the cost. The teacher must be a non-reasoning model or have reasoning disabled per request.
3. **Terms that allow distillation and publishing.** Some commercial APIs forbid using their outputs to train
   another model. The recording of teacher outputs ships with this repository.

A model slug can change behaviour behind the same name, and on OpenRouter the same slug is served by several
providers with different quantisations.

## Decision

**Bake-off.** `taskdistill teacher bakeoff` scored three DeepSeek and Qwen candidates on OpenRouter, each pinned
to one provider, on validation inputs with gold labels: 200 Banking77 queries and 50 invoice documents. The
rule is the cheapest model (by measured cost per 1,000 requests) within 2 points of the best gold score:
accuracy for Banking77, and field micro-F1 and JSON validity for invoices. Candidates with truncated outputs,
reasoning tokens or more than 1% failed requests are not eligible. `:free` variants are rejected. The results
are committed in `reports/bakeoff/`.

| Task | Chosen teacher | Provider (pinned) | List price, $/M tokens in / out | Gold score on validation |
|---|---|---|---|---|
| Banking77 | `deepseek/deepseek-v4.1-flash` | DeepInfra, `deepinfra/fp8` | 0.14 / 0.42 | accuracy 0.740 (n = 200) |
| Invoices | `deepseek/deepseek-v4-flash-0731` | DeepInfra, `deepinfra/fp8` | 0.06 / 0.18 | field micro-F1 0.992, JSON validity 1.000 (n = 50) |

Prices are from the pricing snapshot of 2026-09-26 (`taskdistill pricing refresh`); the v4.1-flash price
included a 30% promotional discount on that date. The ledger settles every call with the `usage.cost` that
OpenRouter returns, which was below the list price for Banking77 because the provider cached the shared system
prompt. Different teachers per task are allowed by design: each task pins its own.

**Pinning.** Each bundled task spec pins the slug and the provider in `teacher.extra_body`:
`provider: {order: [deepinfra/fp8], allow_fallbacks: false, require_parameters: true, data_collection: deny}`,
plus `reasoning: {enabled: false}` because both DeepSeek V4 Flash models are hybrid reasoning models with
reasoning on by default. The bake-off verified zero reasoning tokens and zero truncated outputs for both.
Invoices request `response_format: {type: json_object}`, which the pinned endpoint lists in its supported
parameters. Every raw teacher output is stored (the response cache while building, the recording in the
package), with the date of labelling in the recording manifest.

**Terms.** Checked on 2026-09-26:
- OpenRouter's terms defer output ownership to the model's terms and do not restrict training on outputs.
- DeepInfra's terms do not restrict the customer's use of outputs, and state that DeepInfra does not train on
  customer data. `data_collection: deny` keeps requests on endpoints that do not retain prompts.
- The DeepSeek V4 model weights are MIT-licensed, and DeepSeek's own platform terms explicitly allow using
  outputs to train other models, including distillation.
- Providers whose terms were ambiguous or restrictive about training on outputs (for example Alibaba Model
  Studio, Parasail, SiliconFlow, Fireworks) were excluded from the candidates.

## Consequences

- The recordings in `src/taskdistill/_data/demos/` hold DeepSeek V4 Flash outputs obtained through DeepInfra;
  publishing them is allowed under the terms above.
- Replay refuses a recording whose manifest does not match the spec's slug, provider, prompt or generation
  parameters, so changing the teacher means re-recording, not silently mixing outputs.
- Teams using taskdistill on their own call must check their own provider's terms: some commercial APIs
  forbid training models on their outputs.
