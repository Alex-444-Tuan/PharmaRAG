# Grounding eval (AT-2, AT-2b), test split

- Timestamp: 2026-10-23T21:39:41Z
- Writer LLM (claim generation): claude-sonnet-5-5
- NLI model: cross-encoder/nli-deberta-v3-small   Embedding model: BAAI/bge-small-en-v1.5
- corpus_version: 24bdf02ae716c922
- config/thresholds.yaml commit: e928fb5e21b57484ed4ff7e145b7fe0fee1c23c3 (uncommitted changes: False)
- Grounding thresholds (config/thresholds.yaml): {'entail_min': 0.7, 'contradict_min': 0.8, 'support_ratio_min': 0.8, 'candidate_spans': 3, 'nearest_spans': 3, 'premise_context': 'cascade', 'premise_heading': 'sub'}
- Verifier as run: {'context': 'cascade', 'heading': 'sub', 'candidate_spans': 3, 'nearest_spans': 3, 'entail_min': 0.7, 'contradict_min': 0.8}
- Claims: grounding_claims_test.jsonl sha256 d9780f73f936220f
- Other datasets: {'grounding_labels_test.jsonl': 'c909f080e7612cbb935980ec13f10d266b7c5f3575a4936c7120b49099fb9380', 'grounding_topup_test.jsonl': 'c0aef093945348ac793e7a4470bbb84eeca2a9f3316fe5b4b6a65cbf89639df3'}
- Runs: 1 (The verifier is deterministic, so one run; there is no LLM in scoring.)
- Natural vs perturbed claims: {'S:natural': 30, 'H:natural': 1, 'H:perturbed': 9}

## Result

| metric | value | counts | target |
|---|---|---|---|
| AT-2 reduction (B - S) / B | 90.0% | B=10, S=1 | at least 73% |
| AT-2b false-reject rate | 10.0% | 3 of 30 supported claims rejected | at most 10% |
| precision of `supported` | 96.4% | 27/28 = 96.4% | |
| recall of `supported` | 90.0% | 27/30 = 90.0% | |

AT-2b guard: PASS. AT-2: MET. AT-2 counts as a result only when the guard passes: valid.

## By claim kind

| label:kind | claims | passed as supported |
|---|---|---|
| H:frequency | 1 | 0 |
| H:natural | 1 | 1 |
| H:negation | 2 | 0 |
| H:numeric | 1 | 0 |
| H:other_drug | 4 | 0 |
| H:population | 1 | 0 |
| S:natural | 30 | 27 |

Rejection reasons: {'low_entailment': 7, 'nli_contradiction': 3, 'numeric_mismatch': 2}

Labels: an LLM judge (Claude Opus 5.5) marked each natural claim S or H, and a pharmacist reviewed all 40 labels and approved them unchanged against the cited evidence shown in the worksheet (rubric at the top of reports/grounding_label_worksheet.md). Top-up fired: 9 supported claims were replaced by perturbed claims labelled H by construction. Baseline (verifier off) returns every sentence, so all B reach the user.
