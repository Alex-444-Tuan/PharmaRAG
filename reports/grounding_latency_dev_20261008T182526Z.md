# Verifier per-sentence latency (CPU, real corpus evidence)

- Timestamp: 2026-10-08T18:25:26Z   Hardware: macOS-26.6.2-arm64-arm-64bit arm, CPU only
- NLI model: cross-encoder/nli-deberta-v3-xsmall   Embedding model: BAAI/bge-small-en-v1.5
- Claims: 126 (dev set evidence), repeats: 5, one untimed warm-up pass
- Budget: at most 150 ms p95 per sentence (spec section 9)
- `verify_sentence` = thread hop + sentence embedding + candidate selection + number check + one NLI batch. `prepare` (span embeddings) is separate: in stream mode it overlaps the LLM's time to first token.

| variant | p50 ms | p95 ms | NLI-path p50 | NLI-path p95 | verdict supported p50 | verdict supported p95 | prepare p50 | budget (all sentences) |
|---|---|---|---|---|---|---|---|---|
| spec: neighbour, no heading, 5 candidates | 119.9 | 322.1 | 125.8 | 334.6 | 126.4 | 348.7 | 186.0 | EXCEEDED |
| neighbour + heading, 3 candidates | 78.9 | 174.3 | 82.0 | 179.3 | 83.0 | 181.5 | 254.9 | EXCEEDED |
| cascade + heading, 5 candidates | 113.8 | 249.5 | 120.5 | 256.4 | 107.3 | 215.8 | 210.5 | EXCEEDED |
| cascade + heading, 3 candidates | 71.7 | 140.9 | 77.5 | 142.5 | 65.0 | 105.2 | 210.7 | met |
