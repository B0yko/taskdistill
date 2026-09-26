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

Canonical mode applies to `low_confidence` escalations: requests in the task's shape that the student could
have answered. The other two escalation reasons are returned verbatim in both modes (a stream is passed
through), because the request is not in the task's shape, so there is no student-format answer to match:

- `unsupported` (tools or functions, `n > 1`, `logprobs`): the client asked for something a canonical answer
  would drop (tool calls, extra choices, log-probabilities), so the teacher's response is passed through whole.
- `input_unparsed` (the request does not match `input.from`): the server cannot tell task traffic from other
  calls sent to the same base URL, and rewriting an unrelated call's answer into a task label or schema object
  would corrupt it. These answers are not counted as unnormalised outputs; `x-taskdistill-reason` says which
  path a response took.

## Consequences

- Applications see one response format for task requests regardless of the route; the
  `x-taskdistill-route` header tells them which model answered if they care.
- An application whose prompt drifts so that `input.regex` stops matching gets the teacher's raw text for
  those requests (reason `input_unparsed`) instead of the canonical format. The rate of `input_unparsed`
  escalations in `/metrics` and the serve log is the signal that the input rule needs updating.
- In canonical mode a streamed request receives one SSE chunk followed by `[DONE]`, not token-by-token
  streaming.
