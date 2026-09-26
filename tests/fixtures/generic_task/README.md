# Generic-path fixture: support-ticket extraction

A small task that is not one of the demos, used by `tests/test_generic_path.py` to run
`init -> capture --import -> curate -> train -> eval -> serve -> report` end to end.

- `schema.json`, `teacher_prompt.md`: the task, written over the files `taskdistill init --type extraction` scaffolds.
  Four fields: `ticket_id`, `product`, `severity`, `refund_amount`.
- `captured.jsonl`: the application's traffic for the 48 training tickets in the `openai` import format
  (`{"request", "response"}`, what `taskdistill capture --export` writes). One ticket was sent twice, and one answer
  violates the schema (severity `critical`).
- `inputs.jsonl`: all 74 tickets in the `inputs` format with their gold fields and `meta.split`
  (48 train, 14 valid, 12 test; the two held-out splits differ in size so a threshold chosen on the wrong one shows).
  Curate joins them to the captures by input hash; the valid and test tickets were never captured, so the teacher
  labels them.
- `teacher_answers.jsonl`: the fake teacher's replies to the tickets that were never captured, plus one new ticket
  sent to the cascade server. The teacher misreads the severity of one valid and one test ticket. Its reply to the new
  ticket is fenced JSON with the keys out of schema order and `39.00`, so the server's canonical rendering differs
  from the raw reply.

One training ticket is a long forwarded thread that the length filter drops at `max_seq_len: 256`. Some tickets carry
an email address or a phone number for the PII scrub. All data is synthetic: `example.com` addresses, 555-01xx phone
numbers and invented ticket references.
