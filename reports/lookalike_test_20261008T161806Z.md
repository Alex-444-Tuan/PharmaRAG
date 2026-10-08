# Look-alike eval, test split, full graph

- Timestamp: 2026-10-08T16:18:06Z
- Model ID: claude-sonnet-5-5
- corpus_version: 24bdf02ae716c922
- config/thresholds.yaml commit: 980367d8eae4348b179e53e98b66ecc53082caf7 (uncommitted changes: False)
- Runs: 5   Queries: 15   Kinds: {'misspelling': 8, 'brand': 4, 'correct_partner': 3}
- Dataset: lookalike_test.jsonl sha256 71f96645e017c573
- Resolver thresholds: {'resolved_min': 0.9, 'margin': 0.05, 'ambiguous_min': 0.8, 'trigram_min': 0.4, 'trigram_k': 5, 'max_candidates': 3}

## System

| outcome | count (last run) |
|---|---|
| correct | 12 |
| clarified | 3 |
| other | 0 |
| wrong_drug | 0 |
| not_recognised | 0 |
| not_covered | 0 |
| clarify_without_gold | 0 |
| **misinterpreted** | **0** of 15 |

Across 5 run(s): misinterpreted [0, 0, 0, 0, 0], clarified [3, 3, 3, 3, 3].

## Baseline (resolver off)

Misinterpreted per run: [2, 1, 1, 2, 2] of 15.

## AT-1 metric

(B - S) / B with B, S the misinterpreted counts of baseline, system

- Baseline misinterpreted per run: [2, 1, 1, 2, 2]
- System misinterpreted per run: [0, 0, 0, 0, 0]
- System clarified per run (counted separately): [3, 3, 3, 3, 3]
- Reduction per run: [1.0, 1.0, 1.0, 1.0, 1.0]
- Reduction of the mean counts: 1.0
