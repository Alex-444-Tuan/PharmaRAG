# Look-alike eval, test split, resolver-only

- Timestamp: 2026-10-06T18:47:50Z
- Model ID: none (resolver-only, no LLM)
- corpus_version: f8f0f8f32f1155fd
- config/thresholds.yaml commit: 5deed9d171077285557696752cddcef31b151474 (uncommitted changes: False)
- Runs: 1   Queries: 15   Kinds: {'misspelling': 8, 'brand': 4, 'correct_partner': 3}
- Dataset: lookalike_test.jsonl sha256 8637c9e78d0cb9ed
- Resolver thresholds: {'resolved_min': 0.9, 'margin': 0.05, 'ambiguous_min': 0.8, 'trigram_min': 0.4, 'trigram_k': 5, 'max_candidates': 3}
- RxNav: live

## System (resolver only)

| outcome | count (last run) |
|---|---|
| correct | 14 |
| clarified | 1 |
| other | 0 |
| wrong_drug | 0 |
| not_recognised | 0 |
| not_covered | 0 |
| clarify_without_gold | 0 |
| **misinterpreted** | **0** of 15 |

Across 1 run(s): misinterpreted [0], clarified [1].
Rows resolved in degraded mode (RxNav down): [0].

### Rows (last run)

| mention | kind | outcome | status | match | resolved_name | score |
|---|---|---|---|---|---|---|
| lamotragine | misspelling | correct | resolved | approximate | lamotrigine | 0.9436 |
| lamivadine | misspelling | correct | resolved | approximate | lamivudine | 0.96 |
| fluoxitine | misspelling | correct | resolved | approximate | fluoxetine | 0.9378 |
| paroxitin | misspelling | correct | resolved | approximate | paroxetine | 0.9128 |
| alprazolm | misspelling | correct | resolved | approximate | alprazolam | 0.98 |
| azathiaprene | misspelling | correct | resolved | approximate | azathioprine | 0.9133 |
| sumatriptin | misspelling | correct | resolved | approximate | sumatriptan | 0.9636 |
| clomiphine | misspelling | clarified | ambiguous | None | None | 0.9378 |
| Lamictal | brand | correct | resolved | brand | lamotrigine | 1.0 |
| Cozaar | brand | correct | resolved | brand | losartan | 1.0 |
| Paxil | brand | correct | resolved | brand | paroxetine | 1.0 |
| Tegretol | brand | correct | resolved | brand | carbamazepine | 1.0 |
| tiagabine | correct_partner | correct | resolved | exact | tiagabine | 1.0 |
| quinidine | correct_partner | correct | resolved | exact | quinidine | 1.0 |
| azacitidine | correct_partner | correct | resolved | exact | azacitidine | 1.0 |

## Baseline proxy

**PROXY, not the AT-1 number. This baseline runs unfiltered dense and lexical retrieval on the question, fuses the lists, and checks which drug the top evidence belongs to. No answer is written, so it does not say whether a writer would have cited that chunk.**

- Embedding model: BAAI/bge-small-en-v1.5
- Top evidence chunk belongs to another drug: **5** of 15
- Any of the top 8 evidence chunks belongs to another drug: **15** of 15
- No evidence returned: 0
