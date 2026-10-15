# Grounding threshold tuning, dev split (synthetic claims)

- Timestamp: 2026-10-15T22:39:12Z
- NLI model: cross-encoder/nli-deberta-v3-small   Embedding model: BAAI/bge-small-en-v1.5
- Dev claims: grounding_dev.jsonl sha256 2cd8d438c8232c22; S=66, H=48
- config/thresholds.yaml commit: 2d429da86c4567ff9671b7d5192c4493a8c6be48 (uncommitted changes: False)
- Current grounding thresholds: {'entail_min': 0.7, 'contradict_min': 0.8, 'support_ratio_min': 0.8, 'candidate_spans': 3, 'nearest_spans': 3, 'premise_context': 'cascade', 'premise_heading': 'sub'}
- Candidate spans used in this run: 3   Heading in premise: sub
- Dev claims are synthetic (rule-based rewrites and perturbations of real label sentences). The test claims are natural writer sentences labelled by a human.
- Precision and recall are for the verdict `supported` against label S: precision = supported verdicts that are S; recall = S claims given `supported`.
- `contradict_min` changes only the reason in the debug payload: `unsupported` and `contradicted` are both removed.

## Cascade check

`cascade` (span plus neighbour first, the span alone only on a miss) at the proposed `both` thresholds {'entail_min': 0.6, 'contradict_min': 0.8}: same verdict as `both` on 114/114 claims; precision 63/64 = 98.4%, recall 63/66 = 95.5%, false-reject 3/66 = 4.5%, reduction 47/48 = 97.9%.

## Premise: neighbour

| thresholds | precision | recall | false-reject | reduction |
|---|---|---|---|---|
| current (0.7/0.8) | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| proposed (0.6/0.8) | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |

Per kind, claims passed as `supported` (S kinds should pass, H kinds should not):

| kind | claims | passed at current | passed at proposed |
|---|---|---|---|
| H:frequency | 5 | 0 | 0 |
| H:negation | 8 | 0 | 0 |
| H:numeric | 10 | 0 | 0 |
| H:other_drug | 8 | 0 | 0 |
| H:overclaim | 6 | 1 | 1 |
| H:population | 5 | 0 | 0 |
| H:row_numeric | 6 | 0 | 0 |
| S:combine | 8 | 0 | 1 |
| S:heading | 10 | 10 | 10 |
| S:join | 6 | 5 | 5 |
| S:rewrite | 14 | 13 | 13 |
| S:row_restate | 8 | 7 | 7 |
| S:tallman | 6 | 6 | 6 |
| S:verbatim | 14 | 12 | 12 |

Without the `heading` kind (104 claims; that kind was built to need the heading):

| thresholds | precision | recall | false-reject | reduction |
|---|---|---|---|---|
| current (0.7) | 43/44 = 97.7% | 43/56 = 76.8% | 13/56 = 23.2% | 47/48 = 97.9% |
| proposed above (0.6) | 44/45 = 97.8% | 44/56 = 78.6% | 12/56 = 21.4% | 47/48 = 97.9% |
| own proposal (0.6) | 44/45 = 97.8% | 44/56 = 78.6% | 12/56 = 21.4% | 47/48 = 97.9% |

entail_min sweep:

| entail_min | precision | recall | false-reject | reduction |
|---|---|---|---|---|
| 0.05 | 55/56 = 98.2% | 55/66 = 83.3% | 11/66 = 16.7% | 47/48 = 97.9% |
| 0.1 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.15 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.2 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.25 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.3 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.35 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.4 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.45 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.5 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.55 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.6 | 54/55 = 98.2% | 54/66 = 81.8% | 12/66 = 18.2% | 47/48 = 97.9% |
| 0.65 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.7 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.75 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.8 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.85 | 52/53 = 98.1% | 52/66 = 78.8% | 14/66 = 21.2% | 47/48 = 97.9% |
| 0.9 | 52/53 = 98.1% | 52/66 = 78.8% | 14/66 = 21.2% | 47/48 = 97.9% |
| 0.95 | 52/53 = 98.1% | 52/66 = 78.8% | 14/66 = 21.2% | 47/48 = 97.9% |
| 0.97 | 47/48 = 97.9% | 47/66 = 71.2% | 19/66 = 28.8% | 47/48 = 97.9% |
| 0.99 | 30/31 = 96.8% | 30/66 = 45.5% | 36/66 = 54.5% | 47/48 = 97.9% |

contradict_min sweep (at the proposed entail_min):

| contradict_min | contradicted verdicts | on supported claims | precision |
|---|---|---|---|
| 0.5 | 22 | 1 | 21/22 = 95.5% |
| 0.55 | 22 | 1 | 21/22 = 95.5% |
| 0.6 | 22 | 1 | 21/22 = 95.5% |
| 0.65 | 22 | 1 | 21/22 = 95.5% |
| 0.7 | 22 | 1 | 21/22 = 95.5% |
| 0.75 | 21 | 1 | 20/21 = 95.2% |
| 0.8 | 21 | 1 | 20/21 = 95.2% |
| 0.85 | 19 | 1 | 18/19 = 94.7% |
| 0.9 | 19 | 1 | 18/19 = 94.7% |
| 0.95 | 15 | 0 | 15/15 = 100.0% |
| 0.97 | 15 | 0 | 15/15 = 100.0% |
| 0.99 | 14 | 0 | 14/14 = 100.0% |

## Premise: span

| thresholds | precision | recall | false-reject | reduction |
|---|---|---|---|---|
| current (0.7/0.8) | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| proposed (0.7/0.8) | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |

Per kind, claims passed as `supported` (S kinds should pass, H kinds should not):

| kind | claims | passed at current | passed at proposed |
|---|---|---|---|
| H:frequency | 5 | 0 | 0 |
| H:negation | 8 | 0 | 0 |
| H:numeric | 10 | 0 | 0 |
| H:other_drug | 8 | 0 | 0 |
| H:overclaim | 6 | 1 | 1 |
| H:population | 5 | 0 | 0 |
| H:row_numeric | 6 | 0 | 0 |
| S:combine | 8 | 1 | 1 |
| S:heading | 10 | 10 | 10 |
| S:join | 6 | 0 | 0 |
| S:rewrite | 14 | 14 | 14 |
| S:row_restate | 8 | 8 | 8 |
| S:tallman | 6 | 6 | 6 |
| S:verbatim | 14 | 14 | 14 |

Without the `heading` kind (104 claims; that kind was built to need the heading):

| thresholds | precision | recall | false-reject | reduction |
|---|---|---|---|---|
| current (0.7) | 43/44 = 97.7% | 43/56 = 76.8% | 13/56 = 23.2% | 47/48 = 97.9% |
| proposed above (0.7) | 43/44 = 97.7% | 43/56 = 76.8% | 13/56 = 23.2% | 47/48 = 97.9% |
| own proposal (0.7) | 43/44 = 97.7% | 43/56 = 76.8% | 13/56 = 23.2% | 47/48 = 97.9% |

entail_min sweep:

| entail_min | precision | recall | false-reject | reduction |
|---|---|---|---|---|
| 0.05 | 53/56 = 94.6% | 53/66 = 80.3% | 13/66 = 19.7% | 45/48 = 93.8% |
| 0.1 | 53/55 = 96.4% | 53/66 = 80.3% | 13/66 = 19.7% | 46/48 = 95.8% |
| 0.15 | 53/55 = 96.4% | 53/66 = 80.3% | 13/66 = 19.7% | 46/48 = 95.8% |
| 0.2 | 53/55 = 96.4% | 53/66 = 80.3% | 13/66 = 19.7% | 46/48 = 95.8% |
| 0.25 | 53/55 = 96.4% | 53/66 = 80.3% | 13/66 = 19.7% | 46/48 = 95.8% |
| 0.3 | 53/55 = 96.4% | 53/66 = 80.3% | 13/66 = 19.7% | 46/48 = 95.8% |
| 0.35 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.4 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.45 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.5 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.55 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.6 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.65 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.7 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.75 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.8 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.85 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.9 | 53/54 = 98.1% | 53/66 = 80.3% | 13/66 = 19.7% | 47/48 = 97.9% |
| 0.95 | 51/52 = 98.1% | 51/66 = 77.3% | 15/66 = 22.7% | 47/48 = 97.9% |
| 0.97 | 50/51 = 98.0% | 50/66 = 75.8% | 16/66 = 24.2% | 47/48 = 97.9% |
| 0.99 | 28/28 = 100.0% | 28/66 = 42.4% | 38/66 = 57.6% | 48/48 = 100.0% |

contradict_min sweep (at the proposed entail_min):

| contradict_min | contradicted verdicts | on supported claims | precision |
|---|---|---|---|
| 0.5 | 22 | 1 | 21/22 = 95.5% |
| 0.55 | 22 | 1 | 21/22 = 95.5% |
| 0.6 | 22 | 1 | 21/22 = 95.5% |
| 0.65 | 22 | 1 | 21/22 = 95.5% |
| 0.7 | 21 | 1 | 20/21 = 95.2% |
| 0.75 | 21 | 1 | 20/21 = 95.2% |
| 0.8 | 21 | 1 | 20/21 = 95.2% |
| 0.85 | 21 | 1 | 20/21 = 95.2% |
| 0.9 | 21 | 1 | 20/21 = 95.2% |
| 0.95 | 18 | 1 | 17/18 = 94.4% |
| 0.97 | 17 | 1 | 16/17 = 94.1% |
| 0.99 | 15 | 1 | 14/15 = 93.3% |

## Premise: both

| thresholds | precision | recall | false-reject | reduction |
|---|---|---|---|---|
| current (0.7/0.8) | 62/63 = 98.4% | 62/66 = 93.9% | 4/66 = 6.1% | 47/48 = 97.9% |
| proposed (0.6/0.8) | 63/64 = 98.4% | 63/66 = 95.5% | 3/66 = 4.5% | 47/48 = 97.9% |

Per kind, claims passed as `supported` (S kinds should pass, H kinds should not):

| kind | claims | passed at current | passed at proposed |
|---|---|---|---|
| H:frequency | 5 | 0 | 0 |
| H:negation | 8 | 0 | 0 |
| H:numeric | 10 | 0 | 0 |
| H:other_drug | 8 | 0 | 0 |
| H:overclaim | 6 | 1 | 1 |
| H:population | 5 | 0 | 0 |
| H:row_numeric | 6 | 0 | 0 |
| S:combine | 8 | 5 | 6 |
| S:heading | 10 | 10 | 10 |
| S:join | 6 | 5 | 5 |
| S:rewrite | 14 | 14 | 14 |
| S:row_restate | 8 | 8 | 8 |
| S:tallman | 6 | 6 | 6 |
| S:verbatim | 14 | 14 | 14 |

Without the `heading` kind (104 claims; that kind was built to need the heading):

| thresholds | precision | recall | false-reject | reduction |
|---|---|---|---|---|
| current (0.7) | 52/53 = 98.1% | 52/56 = 92.9% | 4/56 = 7.1% | 47/48 = 97.9% |
| proposed above (0.6) | 53/54 = 98.1% | 53/56 = 94.6% | 3/56 = 5.4% | 47/48 = 97.9% |
| own proposal (0.1) | 54/56 = 96.4% | 54/56 = 96.4% | 2/56 = 3.6% | 46/48 = 95.8% |

entail_min sweep:

| entail_min | precision | recall | false-reject | reduction |
|---|---|---|---|---|
| 0.05 | 64/68 = 94.1% | 64/66 = 97.0% | 2/66 = 3.0% | 44/48 = 91.7% |
| 0.1 | 64/66 = 97.0% | 64/66 = 97.0% | 2/66 = 3.0% | 46/48 = 95.8% |
| 0.15 | 63/65 = 96.9% | 63/66 = 95.5% | 3/66 = 4.5% | 46/48 = 95.8% |
| 0.2 | 63/65 = 96.9% | 63/66 = 95.5% | 3/66 = 4.5% | 46/48 = 95.8% |
| 0.25 | 63/65 = 96.9% | 63/66 = 95.5% | 3/66 = 4.5% | 46/48 = 95.8% |
| 0.3 | 63/65 = 96.9% | 63/66 = 95.5% | 3/66 = 4.5% | 46/48 = 95.8% |
| 0.35 | 63/64 = 98.4% | 63/66 = 95.5% | 3/66 = 4.5% | 47/48 = 97.9% |
| 0.4 | 63/64 = 98.4% | 63/66 = 95.5% | 3/66 = 4.5% | 47/48 = 97.9% |
| 0.45 | 63/64 = 98.4% | 63/66 = 95.5% | 3/66 = 4.5% | 47/48 = 97.9% |
| 0.5 | 63/64 = 98.4% | 63/66 = 95.5% | 3/66 = 4.5% | 47/48 = 97.9% |
| 0.55 | 63/64 = 98.4% | 63/66 = 95.5% | 3/66 = 4.5% | 47/48 = 97.9% |
| 0.6 | 63/64 = 98.4% | 63/66 = 95.5% | 3/66 = 4.5% | 47/48 = 97.9% |
| 0.65 | 62/63 = 98.4% | 62/66 = 93.9% | 4/66 = 6.1% | 47/48 = 97.9% |
| 0.7 | 62/63 = 98.4% | 62/66 = 93.9% | 4/66 = 6.1% | 47/48 = 97.9% |
| 0.75 | 62/63 = 98.4% | 62/66 = 93.9% | 4/66 = 6.1% | 47/48 = 97.9% |
| 0.8 | 62/63 = 98.4% | 62/66 = 93.9% | 4/66 = 6.1% | 47/48 = 97.9% |
| 0.85 | 62/63 = 98.4% | 62/66 = 93.9% | 4/66 = 6.1% | 47/48 = 97.9% |
| 0.9 | 62/63 = 98.4% | 62/66 = 93.9% | 4/66 = 6.1% | 47/48 = 97.9% |
| 0.95 | 61/62 = 98.4% | 61/66 = 92.4% | 5/66 = 7.6% | 47/48 = 97.9% |
| 0.97 | 58/59 = 98.3% | 58/66 = 87.9% | 8/66 = 12.1% | 47/48 = 97.9% |
| 0.99 | 40/41 = 97.6% | 40/66 = 60.6% | 26/66 = 39.4% | 47/48 = 97.9% |

contradict_min sweep (at the proposed entail_min):

| contradict_min | contradicted verdicts | on supported claims | precision |
|---|---|---|---|
| 0.5 | 22 | 0 | 22/22 = 100.0% |
| 0.55 | 22 | 0 | 22/22 = 100.0% |
| 0.6 | 22 | 0 | 22/22 = 100.0% |
| 0.65 | 22 | 0 | 22/22 = 100.0% |
| 0.7 | 22 | 0 | 22/22 = 100.0% |
| 0.75 | 21 | 0 | 21/21 = 100.0% |
| 0.8 | 21 | 0 | 21/21 = 100.0% |
| 0.85 | 21 | 0 | 21/21 = 100.0% |
| 0.9 | 21 | 0 | 21/21 = 100.0% |
| 0.95 | 17 | 0 | 17/17 = 100.0% |
| 0.97 | 16 | 0 | 16/16 = 100.0% |
| 0.99 | 14 | 0 | 14/14 = 100.0% |
