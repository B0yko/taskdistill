# ADR 0004: Recorded teacher replay for offline reproducibility

- Status: accepted
- Date: 2026-09-26

## Context

The demos and the README numbers depend on teacher outputs from a paid API. A reader without an API key must
still be able to run the whole pipeline, and a rerun must see exactly the outputs the numbers were computed
from, even if the provider changes the model behind the same slug.

## Decision

- **Request key.** One function, `teacher.request_key.request_key`, computes the SHA-256 of the canonical JSON
  (sorted keys, no whitespace, UTF-8) of `{model, messages, temperature, max_tokens, response_format, top_p,
  seed, stop}`. The response cache, the replay and proxy capture all call it, and a test asserts they agree.
  Base URL, headers and `stream` are excluded. Other output-changing body fields (provider routing, reasoning
  controls) are recorded in the recording manifest instead.
- **Recording.** Each demo ships `teacher_recording.jsonl.gz` as package data: a manifest line (schema version,
  teacher slug, pinned provider, teacher-prompt SHA-256, generation parameters including the extra body,
  pricing snapshot date), then one record per request key with the output, usage, latency, provider, finish
  reason and timestamp. Never the inputs, headers or keys. gzip is written with an empty file name and mtime 0.
- **Replay.** A replay upstream implements the same interface as the live client. Replay refuses a manifest
  that does not match the task spec and names the differing field. A key that is not in the recording is a
  hard error, never a silent fallback to a live call or to the student.
- **Labelling sends the raw input.** Curate's teacher labelling builds the request with the same function the
  application uses (`teacher.requests.build_teacher_request`) from the raw extracted input, so its key equals
  the application's own request, a cascade escalation and the recording. The student trains on the
  PII-scrubbed text; the teacher sees what the application already sends it in production.
- **The cache is for building data, not for serving.** The on-disk response cache is read only by curate, the
  bake-off and demo labelling, so re-runs cost nothing. `serve` never reads it: a cascade that answered
  escalations from a cache would report latencies and costs that production never sees. `serve` may use the
  replay (for the demo and offline smoke tests), and then says so in its banner, in `/healthz` and in the
  `x-taskdistill-teacher: replay` header. `taskdistill bench` refuses any replayed escalation.

## Consequences

- `taskdistill demo banking77` and `demo invoices` run end to end with no API key, and a CI test asserts that
  every request either demo sends, in both profiles, is in the recording.
- A teacher slug change or a prompt edit invalidates the recording loudly instead of producing silently
  different data.
- Recordings grow with the number of requests (about 13k for Banking77) but stay small because they hold only
  outputs and metadata.
