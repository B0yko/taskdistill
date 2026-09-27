# ADR 0002: Confidence definitions

- Status: accepted
- Date: 2026-09-27

## Context

The cascade escalates a request when the student's confidence is below a threshold chosen on validation data. The
confidence must separate right from wrong answers (AUROC) and, ideally, be calibrated (ECE), so that the threshold
means the same thing on new traffic. Several scores are cheap to compute from one decoding pass.

## Decision

Both backends expose `generate_with_scores(messages, constraint)`, which returns the text and per-token log-probs.

- **Classification (primary):** greedy decoding under a trie of the canonical labels, each ending with the chat end
  token; confidence is the product of the chosen tokens' probabilities, renormalised over the tokens the trie allows
  at each step (the end token included).
- **Extraction (primary):** free generation; invalid JSON or a schema violation gives 0. Otherwise each schema key's
  value is mapped to the tokens that span it, the field's confidence is the product of their probabilities, and the
  document's confidence is the minimum over fields.
- **Alternatives, computed on the same data:** for classification, free greedy generation scored by the exponential
  of the mean token log-prob (an output that is not exactly a label counts as wrong); for extraction, the exponential
  of the mean token log-prob of the whole output of the same generation.

The definitions were compared on the validation split before either was used on test
(`confidence_comparison` in `reports/<task>/report.json`, full profile, selected runs):

| Task | Confidence | AUROC vs teacher | ECE vs teacher |
|---|---|---|---|
| Banking77 | primary (trie product) | 0.877 | 2.8% |
| Banking77 | alternative (free generation, mean log-prob) | 0.874 | 9.2% |
| Invoices | primary (min over fields) | 0.904 | 3.7% |
| Invoices | alternative (whole-output mean log-prob) | 0.903 | 10.0% |

The two separate right from wrong about equally well, but the primary definitions are better calibrated (about 3.3x
lower ECE on Banking77 and 2.7x on invoices), so they drive the cascade. Isotonic calibration, fitted on validation, is reported for calibration only;
the threshold is chosen on raw confidence.

## Consequences

- Classification never returns an invalid label, and a label that is a prefix of another is still decodable.
- The document-level minimum makes one uncertain field enough to escalate, which is what an extraction consumer
  needs; it also means long schemas escalate more often.
- On the invoice test set, drawn from layouts never seen in training, the primary score's AUROC (0.846) was slightly
  below the alternative's (0.858); the choice was made on validation and is not revisited on test.
