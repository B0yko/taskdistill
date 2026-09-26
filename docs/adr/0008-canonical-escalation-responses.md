# ADR 0008: Canonical escalation responses by default

- Status: accepted
- Date: 2026-09-26

## Context

When the cascade escalates, the teacher's raw answer can differ in form from the student's: "Card Arrival."
instead of `card_arrival`, or JSON wrapped in a Markdown fence. The application would then have to handle two
output formats depending on a routing decision it cannot see.

## Decision

`cascade.escalation_response` defaults to `canonical`: the teacher output goes through the same normaliser
curate uses to build training data (label mapping for classification; fence stripping, JSON parsing and
schema validation for extraction) and the same rendering as student answers (`student.response_template`),
so both routes return one format. An output that cannot be normalised is returned raw and counted in
`/metrics`. `raw` returns the teacher's response verbatim, including its SSE stream for `stream: true`.

## Consequences

- Applications see one response format regardless of the route; the `x-taskdistill-route` header tells them
  which model answered if they care.
- In canonical mode a streamed request receives one SSE chunk followed by `[DONE]`, not token-by-token
  streaming.
