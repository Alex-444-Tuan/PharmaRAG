# Verifier per-sentence latency (CPU, real corpus evidence)

- Timestamp: 2026-10-15T22:44:46Z   Hardware: macOS-26.6.2-arm64-arm-64bit arm, CPU only
- NLI model: cross-encoder/nli-deberta-v3-small   Embedding model: BAAI/bge-small-en-v1.5
- Claims: 154 (dev set evidence), repeats: 5, one untimed warm-up pass
- Budget: at most 150 ms p95 per sentence (spec section 9)
- `verify_sentence` = thread hop + sentence embedding + candidate selection + number check + one NLI batch. `prepare` (span embeddings) is separate: in stream mode it overlaps the LLM's time to first token.

| variant | p50 ms | p95 ms | NLI-path p50 | NLI-path p95 | verdict supported p50 | verdict supported p95 | prepare p50 | budget (all sentences) |
|---|---|---|---|---|---|---|---|---|
| spec: neighbour, no heading, 5 candidates | 93.4 | 155.0 | 101.2 | 155.9 | 97.9 | 181.6 | 163.8 | EXCEEDED |
| neighbour + heading, 3 candidates | 69.9 | 115.2 | 74.6 | 125.0 | 74.6 | 128.7 | 162.0 | met |
| cascade + heading, 5 candidates | 118.8 | 312.6 | 128.3 | 310.2 | 111.5 | 227.6 | 162.1 | EXCEEDED |
| shipped: cascade + heading, 3 candidates | 82.3 | 207.9 | 96.0 | 210.2 | 79.5 | 174.4 | 171.7 | EXCEEDED |

## verify_many, batches of 1 to 4 sentences ({'context': 'cascade', 'heading': 'sub', 'candidate_spans': 3})

One NLI batch per stage for the whole burst. Batches are the first n sentences of one answer (one evidence set). The ms columns are for the whole batch.

| sentences | answers | batch p50 ms | batch p95 ms | per-sentence p50 | per-sentence p95 |
|---|---|---|---|---|---|
| 1 | 124 | 85.7 | 210.9 | 85.7 | 210.9 |
| 2 | 10 | 190.1 | 331.2 | 95.0 | 165.6 |
| 3 | 9 | 294.1 | 385.9 | 98.0 | 128.6 |
| 4 | 8 | 426.6 | 596.4 | 106.7 | 149.1 |
