# Look-alike eval, test split, resolver-only

- Timestamp: 2026-10-06T19:30:04Z
- Model ID: none (resolver-only, no LLM)
- corpus_version: 24bdf02ae716c922
- config/thresholds.yaml commit: 980367d8eae4348b179e53e98b66ecc53082caf7 (uncommitted changes: False)
- Runs: 3   Queries: 15   Kinds: {'misspelling': 8, 'brand': 4, 'correct_partner': 3}
- Dataset: lookalike_test.jsonl sha256 71f96645e017c573
- Resolver thresholds: {'resolved_min': 0.9, 'margin': 0.05, 'ambiguous_min': 0.8, 'trigram_min': 0.4, 'trigram_k': 5, 'max_candidates': 3}
- RxNav: live

## System (resolver only)

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

Across 3 run(s): misinterpreted [0, 0, 0], clarified [3, 3, 3].
Rows resolved in degraded mode (RxNav down): [0, 0, 0].

### Rows (last run)

| mention | kind | outcome | status | match | resolved_name | score |
|---|---|---|---|---|---|---|
| hydroxazine | misspelling | clarified | ambiguous | None | None | 0.9636 |
| quinidene | misspelling | clarified | ambiguous | None | None | 0.9306 |
| glypuride | misspelling | correct | resolved | approximate | glyburide | 0.9481 |
| predisolone | misspelling | clarified | ambiguous | None | None | 0.9436 |
| carbamezapine | misspelling | correct | resolved | approximate | carbamazepine | 0.9846 |
| lorazapam | misspelling | correct | resolved | approximate | lorazepam | 0.9306 |
| tizanadine | misspelling | correct | resolved | approximate | tizanidine | 0.9378 |
| nifedapine | misspelling | correct | resolved | approximate | nifedipine | 0.9378 |
| Celebrex | brand | correct | resolved | brand | celecoxib | 1.0 |
| Lamictel | brand | correct | resolved | approximate | lamotrigine | 0.95 |
| Toprol | brand | correct | resolved | brand | metoprolol | 1.0 |
| Paxill | brand | correct | resolved | approximate | paroxetine | 0.9667 |
| tiagabine | correct_partner | correct | resolved | exact | tiagabine | 1.0 |
| clomipramine | correct_partner | correct | resolved | exact | clomipramine | 1.0 |
| azacitidine | correct_partner | correct | resolved | exact | azacitidine | 1.0 |

## Baseline proxy

**PROXY, not the AT-1 number. This baseline runs unfiltered dense and lexical retrieval on the question, fuses the lists, and checks which drug the top evidence belongs to. No answer is written, so it does not say whether a writer would have cited that chunk.**

- Embedding model: BAAI/bge-small-en-v1.5
- Top evidence chunk belongs to another drug: **7** of 15
- Any of the top 8 evidence chunks belongs to another drug: **15** of 15
- No evidence returned: 0
