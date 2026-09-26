# ADR 0001: Generative student with trie-constrained decoding

- Status: accepted
- Date: 2026-09-26

## Context

taskdistill distils two kinds of narrow calls: classification (one label out of N) and JSON extraction. A
classification call could be distilled into an encoder with a classification head, which is cheap to train and
gives a softmax over labels for free. Extraction needs a generative model anyway.

## Decision

Both task types use the same student: a small instruction-tuned causal LM (Qwen2.5 0.5B or 1.5B) fine-tuned
with LoRA to produce the teacher's answer as text, with the loss on the completion tokens only.

- Classification decodes greedily under a trie built over the canonical label strings as they tokenise after
  the assistant prefix. Every label ends with the chat end token (`<|im_end|>`), so a label that is a prefix of
  another ("card" and "card_arrival") is still decodable, and every output is a valid label.
- Extraction generates freely; the output is parsed and validated against the task's JSON Schema.

## Consequences

- One training path, one serving path and one confidence module for both task types; a new task type that
  can be written as "text in, short text out" needs no new model code.
- Adding or renaming a label needs no new head: the trie is rebuilt from `labels.txt` at load time (the
  student still has to be retrained to learn the new label).
- The student is larger and slower per request than a classification head on a small encoder would be. The
  README reports its measured latency, and the TF-IDF + logistic regression baseline shows how much of the
  quality an LLM-free model already gets.
- Decoding under the trie needs per-step log-probabilities over the vocabulary, so both backends expose a
  step-wise decoding session instead of relying on a library `generate()` call.
