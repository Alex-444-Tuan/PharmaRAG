# Verifier per-sentence latency (CPU, real corpus evidence)

- Timestamp: 2026-10-15T23:05:09Z   Hardware: macOS-26.6.2-arm64-arm-64bit arm, CPU only
- NLI model: cross-encoder/nli-deberta-v3-xsmall   Embedding model: BAAI/bge-small-en-v1.5
- Claims: 154 (dev set evidence), repeats: 3, one untimed warm-up pass
- Budget: at most 150 ms p95 per sentence (spec section 9)
- `verify_sentence` = thread hop + sentence embedding + candidate selection + number check + one NLI batch. `prepare` (span embeddings) is separate: in stream mode it overlaps the LLM's time to first token.

| variant | p50 ms | p95 ms | NLI-path p50 | NLI-path p95 | verdict supported p50 | verdict supported p95 | prepare p50 | budget (all sentences) |
|---|---|---|---|---|---|---|---|---|
| spec: neighbour, no heading, 5 candidates | 68.5 | 114.5 | 73.5 | 116.5 | 70.9 | 139.4 | 165.5 | met |
| neighbour + heading, 3 candidates | 52.6 | 86.1 | 55.5 | 92.9 | 55.3 | 95.5 | 162.2 | met |
| cascade + heading, 5 candidates | 87.6 | 230.2 | 98.2 | 233.0 | 81.4 | 174.9 | 160.4 | EXCEEDED |
| shipped: cascade + heading, 3 candidates | 61.7 | 151.3 | 69.1 | 153.4 | 58.2 | 127.4 | 169.8 | EXCEEDED |

## verify_many, batches of 1 to 4 sentences ({'context': 'cascade', 'heading': 'sub', 'candidate_spans': 3})

One NLI batch per stage for the whole burst. Batches are the first n sentences of one answer (one evidence set). The ms columns are for the whole batch.

| sentences | answers | batch p50 ms | batch p95 ms | per-sentence p50 | per-sentence p95 |
|---|---|---|---|---|---|
| 1 | 124 | 60.6 | 153.9 | 60.6 | 153.9 |
| 2 | 10 | 150.7 | 226.6 | 75.3 | 113.3 |
| 3 | 9 | 231.7 | 279.7 | 77.2 | 93.2 |
| 4 | 8 | 306.5 | 429.1 | 76.6 | 107.3 |
