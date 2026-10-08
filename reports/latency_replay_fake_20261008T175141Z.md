# Latency eval: replay_fake (2026-10-08T17:51:41Z)

- corpus_version `24bdf02ae716c922`, recorded with model `claude-sonnet-5-5`
- thresholds commit `980367d8eae4348b179e53e98b66ecc53082caf7` (dirty: False), git HEAD `3f13de235607940babe26544884839b5bb35f659`
- verifier: fake: 30-80 ms per sentence, 150 ms prepare
- 30 questions x 5 reps x 3 modes, first rep discarded, 480 measured requests, 0 errors
- Replay: recorded analyze reply and writer tokens with recorded timing; resolver on the local alias table; no RxNav facts; no answer cache exists in this build.

## Headline: requests without regeneration (120 paired requests)

| mode | n | p50 ms | p95 ms | mean ms |
| --- | --- | --- | --- | --- |
| off | 120 | 4123.9 | 6417.23 | 4424.69 |
| sequential | 120 | 4643.96 | 7063.11 | 4874.21 |
| stream | 120 | 4240.56 | 6496.72 | 4488.14 |

Stream minus off, bootstrap 95% interval (questions resampled):

| statistic | difference ms | 95% CI ms | 95% CI relative |
| --- | --- | --- | --- |
| p50 | 116.66 | [-3.2, 173.94] | [-0.0007, 0.0422] |
| p95 | 79.49 | [-275.02, 721.73] | [-0.046, 0.1122] |

Sequential minus off at p50: 520.07 ms (what the verifier would cost without the in-stream design).

**AT-3 PASS**: stream p50 is +2.8% of off (limit +5%), stream p95 is +1.2% (limit +10%).

## Answer node and regeneration

| mode | answer node p50 ms | answer node p95 ms | requests | regenerated | regeneration rate | regenerated p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| off | 2438.3 | 4675.14 | 120 | 0 | 0.0 | None |
| sequential | 2862.26 | 5317.96 | 120 | 0 | 0.0 | None |
| stream | 2504.31 | 4768.96 | 120 | 0 | 0.0 | None |

## Time to first visible sentence (SSE, stream mode)

Time to first visible sentence in `stream` mode over SSE. Not part of AT-3.

| metric | n | p50 ms | p95 ms |
| --- | --- | --- | --- |
| first `sentence` event | 120 | 3197.12 | 4588.63 |
| `done` event | 120 | 4238.3 | 6498.86 |
