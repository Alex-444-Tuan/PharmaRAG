# Latency eval: replay_real (2026-10-16T00:26:48Z)

- corpus_version `24bdf02ae716c922`, recorded with model `claude-sonnet-5-5`
- thresholds commit `e928fb5e21b57484ed4ff7e145b7fe0fee1c23c3` (dirty: False), git HEAD `e928fb5e21b57484ed4ff7e145b7fe0fee1c23c3`
- verifier: real (NLI)
- 30 questions x 5 reps x 3 modes, first rep discarded, 648 measured requests, 0 errors
- Replay: recorded analyze reply and writer tokens with recorded timing; resolver on the local alias table (no RxNav); rx_facts=None, so the RxNorm facts branch returns no facts and the writer prompt has no [r] blocks; no answer cache exists in this build.

## Headline: all recorded questions, no regeneration (80 paired requests, 20 questions)

| mode | n | p50 ms | p95 ms | mean ms |
| --- | --- | --- | --- | --- |
| off | 80 | 4150.34 | 6434.58 | 4497.65 |
| sequential | 80 | 4625.95 | 7550.38 | 5196.22 |
| stream | 80 | 4306.39 | 6635.89 | 4621.51 |

Stream minus off, bootstrap 95% interval (questions resampled):

| statistic | difference ms | 95% CI ms | 95% CI relative |
| --- | --- | --- | --- |
| p50 | 156.05 | [74.7, 238.59] | [0.0167, 0.0524] |
| p95 | 201.31 | [-2.9, 282.31] | [-0.0005, 0.0402] |

Sequential minus off at p50: 475.61 ms (what the verifier would cost without the in-stream design).

**AT-3 PASS (point estimate)**: stream p50 is +3.8% of off (limit +5%), stream p95 is +3.1% (limit +10%).
- **CI upper bound exceeds the limit at p50**: +5.2% > +5%. The data cannot rule out a failure at this percentile.

## Headline without answers cut at the output token cap (5 questions excluded) (60 paired requests, 15 questions)

| mode | n | p50 ms | p95 ms | mean ms |
| --- | --- | --- | --- | --- |
| off | 60 | 4020.42 | 7013.22 | 4113.9 |
| sequential | 60 | 4576.38 | 7533.63 | 4654.48 |
| stream | 60 | 4138.28 | 7285.31 | 4258.31 |

Stream minus off, bootstrap 95% interval (questions resampled):

| statistic | difference ms | 95% CI ms | 95% CI relative |
| --- | --- | --- | --- |
| p50 | 117.86 | [57.75, 213.43] | [0.0151, 0.0523] |
| p95 | 272.08 | [77.85, 282.37] | [0.0174, 0.0526] |

Sequential minus off at p50: 555.96 ms (what the verifier would cost without the in-stream design).

**AT-3 PASS (point estimate)**: stream p50 is +2.9% of off (limit +5%), stream p95 is +3.9% (limit +10%).
- **CI upper bound exceeds the limit at p50**: +5.2% > +5%. The data cannot rule out a failure at this percentile.

Answers that hit the cap lose their last sentence, so they leave less to verify at the end of the stream, which flatters the stream mode. Both headlines are reported.

## Answer node and regeneration (AT-3 question set)

| mode | answer node p50 ms | answer node p95 ms | requests | regenerated | regeneration rate | regenerated p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| off | 2436.14 | 4674.87 | 120 | 0 | 0.0 | None |
| sequential | 2929.63 | 5928.55 | 120 | 40 | 0.3333 | 8317.44 |
| stream | 2512.39 | 4926.73 | 120 | 40 | 0.3333 | 7081.07 |

## Regeneration on the grounding questions (spec section 11.3)

Grounding questions replayed in the three modes. A regenerated request replays the first recording a second time (the second attempt's prompt differs, its text is not recorded), so its latency is the cost of a second writer call, not a measurement of the strict prompt.

| mode | requests | regenerated | rate | regenerated p50 ms | not regenerated p50 ms |
| --- | --- | --- | --- | --- | --- |
| off | 56 | 0 | 0.0 | None | 4271.75 |
| sequential | 56 | 20 | 0.3571 | 8831.96 | 4431.74 |
| stream | 56 | 20 | 0.3571 | 7665.02 | 4059.58 |

## Time to first visible sentence (SSE, stream mode)

Time to first visible sentence in `stream` mode over SSE. Not part of AT-3.

| metric | n | p50 ms | p95 ms |
| --- | --- | --- | --- |
| first `sentence` event | 120 | 3399.21 | 4663.36 |
| `done` event | 120 | 4276.99 | 6602.83 |
