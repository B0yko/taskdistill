# Dataset card: Banking77 (demo)

Curated by `taskdistill curate` for task `banking77` on 2026-09-26 (UTC). Task type: classification. The student trains on teacher outputs; gold labels, where present, are used only for evaluation.

## Source and licence

- Source: https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data
- Licence: CC BY 4.0

BANKING77 by Casanueva et al. (2020), from PolyAI's task-specific-datasets repository (https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

```bibtex
@inproceedings{casanueva-etal-2020-efficient,
    title = "Efficient Intent Detection with Dual Sentence Encoders",
    author = "Casanueva, I{\~n}igo  and
      Tem{\v{c}}inas, Tadas  and
      Gerz, Daniela  and
      Henderson, Matthew  and
      Vuli{\'c}, Ivan",
    booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.nlp4convai-1.5/",
    doi = "10.18653/v1/2020.nlp4convai-1.5",
    pages = "38--45"
}

```

## Splits

| split | examples | teacher output recorded | labelled by curate | no valid teacher output | with gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 8,874 | 8,874 | 0 | 0 | 8,874 |
| valid | 1,030 | 0 | 1,030 | 0 | 1,030 |
| test | 3,075 | 7 | 3,060 | 8 | 3,075 |

13,042 rows kept their predefined split (`meta.split`).

Predefined split rule (from the data source): official test set -> test; official train set -> 90/10 train/valid, stratified by gold label with seed 13.

Rows per split after each stage that can remove rows:

| after | train | valid | test |
| --- | ---: | ---: | ---: |
| split assignment | 8,926 | 1,037 | 3,079 |
| dedupe within splits | 8,900 | 1,036 | 3,075 |
| cross-split removal | 8,874 | 1,034 | 3,075 |
| teacher labelling | 8,874 | 1,030 | 3,075 |
| length filter | 8,874 | 1,030 | 3,075 |

Test rows are never removed by cross-split removal or labelling: a test row whose teacher answer is invalid is kept without a teacher value (it counts against the teacher in evaluation). After dedupe within splits, only the length filter can remove test rows.

## Label distribution (teacher labels)

| label | train | valid | test |
| --- | ---: | ---: | ---: |
| Refund_not_showing_up | 160 | 16 | 43 |
| activate_my_card | 166 | 21 | 43 |
| age_limit | 104 | 13 | 42 |
| apple_pay_or_google_pay | 102 | 12 | 36 |
| atm_support | 87 | 10 | 40 |
| automatic_top_up | 120 | 12 | 40 |
| balance_not_updated_after_bank_transfer | 72 | 6 | 14 |
| balance_not_updated_after_cheque_or_cash_deposit | 159 | 18 | 36 |
| beneficiary_not_allowed | 67 | 9 | 11 |
| cancel_transfer | 148 | 21 | 39 |
| card_about_to_expire | 93 | 10 | 33 |
| card_acceptance | 60 | 6 | 45 |
| card_arrival | 29 | 3 | 15 |
| card_delivery_estimate | 250 | 31 | 83 |
| card_linking | 97 | 4 | 35 |
| card_not_working | 125 | 15 | 47 |
| card_payment_fee_charged | 161 | 22 | 52 |
| card_payment_not_recognised | 99 | 15 | 19 |
| card_payment_wrong_exchange_rate | 84 | 12 | 30 |
| card_swallowed | 55 | 7 | 41 |
| cash_withdrawal_charge | 179 | 22 | 44 |
| cash_withdrawal_not_recognised | 69 | 8 | 24 |
| change_pin | 159 | 16 | 67 |
| compromised_card | 156 | 19 | 70 |
| contactless_not_working | 25 | 3 | 37 |
| country_support | 172 | 22 | 63 |
| declined_card_payment | 220 | 27 | 72 |
| declined_cash_withdrawal | 150 | 16 | 44 |
| declined_transfer | 122 | 12 | 33 |
| direct_debit_payment_not_recognised | 92 | 11 | 20 |
| disposable_card_limits | 115 | 14 | 40 |
| edit_personal_details | 109 | 14 | 41 |
| exchange_charge | 106 | 9 | 32 |
| exchange_rate | 265 | 31 | 98 |
| exchange_via_app | 62 | 8 | 15 |
| extra_charge_on_statement | 206 | 19 | 53 |
| failed_transfer | 151 | 20 | 55 |
| fiat_currency_support | 154 | 14 | 50 |
| get_disposable_virtual_card | 75 | 10 | 41 |
| get_physical_card | 78 | 7 | 25 |
| getting_spare_card | 72 | 7 | 21 |
| getting_virtual_card | 63 | 7 | 35 |
| lost_or_stolen_card | 82 | 13 | 44 |
| lost_or_stolen_phone | 101 | 11 | 39 |
| order_physical_card | 42 | 3 | 8 |
| passcode_forgotten | 139 | 17 | 59 |
| pending_card_payment | 159 | 17 | 39 |
| pending_cash_withdrawal | 130 | 18 | 44 |
| pending_top_up | 124 | 17 | 40 |
| pending_transfer | 77 | 9 | 26 |
| pin_blocked | 89 | 12 | 31 |
| receiving_money | 87 | 10 | 32 |
| request_refund | 170 | 16 | 47 |
| reverted_card_payment? | 83 | 5 | 19 |
| supported_cards_and_currencies | 55 | 8 | 23 |
| terminate_account | 97 | 11 | 41 |
| top_up_by_bank_transfer_charge | 45 | 12 | 17 |
| top_up_by_card_charge | 92 | 11 | 32 |
| top_up_by_cash_or_cheque | 98 | 11 | 40 |
| top_up_failed | 166 | 23 | 53 |
| top_up_limits | 99 | 13 | 46 |
| top_up_reverted | 87 | 7 | 32 |
| topping_up_by_card | 65 | 11 | 19 |
| transaction_charged_twice | 149 | 18 | 42 |
| transfer_fee_charged | 156 | 16 | 47 |
| transfer_into_account | 107 | 10 | 45 |
| transfer_not_received_by_recipient | 167 | 27 | 47 |
| transfer_timing | 296 | 29 | 94 |
| unable_to_verify_identity | 124 | 11 | 51 |
| verify_my_identity | 114 | 10 | 48 |
| verify_source_of_funds | 65 | 8 | 34 |
| verify_top_up | 97 | 12 | 40 |
| virtual_card_not_working | 34 | 5 | 32 |
| visa_or_mastercard | 110 | 12 | 36 |
| why_verify_identity | 69 | 12 | 26 |
| wrong_amount_of_cash_received | 160 | 17 | 41 |
| wrong_exchange_rate_for_cash_withdrawal | 101 | 9 | 29 |

## Curation

- Loaded 8,965 usable captures (of 8,965) and 13,083 imported rows.
- Input extraction: 22,048 records; 0 inputs unparsed, 0 responses without an output.
- Merge by input hash: 13,071 examples; 7 merged across predefined splits (kept in the most protected split).
- Output normalisation: 29 invalid outputs dropped; 29 examples dropped with no valid output, 0 with no majority; 0 invalid gold values.
- Exact repeats of an input (same input hash within one source and split) merged into one example: 5 (train 3, valid 1, test 1).
- Dedupe within splits (Jaccard >= 0.9 on word 3-shingles; texts under 3 words by exact match): 0 exact (same text up to case and whitespace) and 31 near-duplicates removed; 0 clusters with conflicting outputs dropped (0 examples); 2 clusters with conflicting gold labels (the majority gold is kept, else the kept row's own).
- Cross-split duplicates removed: train 26 (vs valid: 0 exact, 6 near; vs test: 0 exact, 20 near), valid 2 (vs test: 0 exact, 2 near); test is never changed.

### PII scrub

Regex with checksums; matches are replaced by typed placeholders. 0 examples changed.

| kind | input | teacher | gold | total |
| --- | ---: | ---: | ---: | ---: |
| email | 0 | 0 | 0 | 0 |
| iban | 0 | 0 | 0 | 0 |
| card | 0 | 0 | 0 | 0 |
| ssn | 0 | 0 | 0 | 0 |
| phone | 0 | 0 | 0 | 0 |
| ipv4 | 0 | 0 | 0 | 0 |

## Length statistics

Student tokens of the full chat (system prompt, input, output) with the `mlx-community/Qwen2.5-0.5B-Instruct-4bit` chat template. Examples over 512 tokens are dropped, never truncated.

| split | examples | p50 | p95 | max | dropped |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 8,874 | 43 | 65.4 | 122 | 0 |
| valid | 1,030 | 43 | 67 | 129 | 0 |
| test | 3,075 | 42 | 62 | 112 | 0 |

## Teacher

- Model: `deepseek/deepseek-v4.1-flash`
- Provider: deepinfra/fp8
- Teacher prompt SHA-256: `0d8430f72f5be434d4d343651d41f46bd7a18fcc318b4c53db6b84417a4c8fca`
- Captured teacher outputs: 8,965 between 2026-09-26 and 2026-09-26
  - request models: `deepseek/deepseek-v4.1-flash` 8,965
  - models named in the responses: `deepseek/deepseek-v4.1-flash` 8,965
- Labelling by curate: 4,102 requests (0 live, 0 cached, 4,102 replayed from a recording), cost $0.0000 in this run, originally $0.0246 when the replayed or cached answers were produced, labelled on 2026-09-26; served by DeepInfra 4,102; 12 invalid answers (4 train/valid examples dropped, 8 test examples kept without a teacher value), 0 truncated.

## Leakage check

An independent pass over the final splits found 0 exact and 0 near-duplicate pairs across train/valid/test (Jaccard >= 0.9): ok.
