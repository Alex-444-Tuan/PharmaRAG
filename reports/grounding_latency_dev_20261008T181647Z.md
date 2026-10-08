# Verifier per-sentence latency (CPU, real corpus evidence)

- Timestamp: 2026-10-08T18:16:47Z   Hardware: macOS-26.6.2-arm64-arm-64bit arm, CPU only
- NLI model: cross-encoder/nli-deberta-v3-small   Embedding model: BAAI/bge-small-en-v1.5
- Claims: 126 (dev set evidence), repeats: 5, one untimed warm-up pass
- Budget: at most 150 ms p95 per sentence (spec section 9)
- `verify_sentence` = thread hop + sentence embedding + candidate selection + number check + one NLI batch. `prepare` (span embeddings) is separate: in stream mode it overlaps the LLM's time to first token.

| variant | p50 ms | p95 ms | NLI-path p50 | NLI-path p95 | verdict supported p50 | verdict supported p95 | prepare p50 | budget (all sentences) |
|---|---|---|---|---|---|---|---|---|
| spec: neighbour, no heading, 5 candidates | 131.2 | 265.0 | 137.0 | 267.6 | 135.8 | 262.0 | 177.3 | EXCEEDED |
| neighbour + heading, 3 candidates | 84.2 | 128.7 | 86.2 | 132.2 | 85.9 | 130.2 | 196.1 | met |
| cascade + heading, 5 candidates | 136.2 | 282.7 | 142.3 | 288.0 | 128.1 | 242.7 | 203.8 | EXCEEDED |
| cascade + heading, 3 candidates | 95.4 | 193.4 | 103.4 | 194.5 | 88.0 | 163.8 | 200.4 | EXCEEDED |
