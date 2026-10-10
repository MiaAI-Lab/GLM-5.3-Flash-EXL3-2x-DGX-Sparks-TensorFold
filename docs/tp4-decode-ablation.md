# Isolating the additional TP4 preset (2026-10-10)

This comparison measures the additions **beyond the BadAd84 backports**, not
the complete branch versus untouched upstream. The Q4 no-cluster reduction
and TP4-adapted CUDA attention stay enabled in every case. The same image,
weights, transport, precision, context and memory settings are held fixed.
No code or preset behavior was changed after collecting these results.

Source: `47bc74c022658b6f611ea36bf35f21775e8989ab`, patch hash `19ec06e95e57`,
upstream `tp4` base `d22003070575c2774de0f010062aa57bae06f3c7`.
[Machine-readable results](benchmarks/tp4-ablation-20261010.json) include exact
settings, model revisions, per-run rates, token counts and token hashes.

## Protocol

- Four GB10 Sparks, switchless ring, upstream NCCL/RoCE transport.
- Same 4bpw Ablit checkpoint and DFlash2 drafter; 500,000 maximum context,
  eight slots, FP8 KV / Q4 dense KV, 32 GiB KV pool, 20 GiB memory reserve.
- Bundled `tools/tp4_prose_prompts.json`, temperature zero, thinking off,
  2,048 output tokens per request. These are varied prose, not structured-output
  or repeated-template prompts. Same prompts are intentionally reused across
  configurations for matched comparison.
- One C8 warmup with a 32-token cap per boot, then three C4 measurements.
  Opening baseline, combined preset and closing baseline also get two C8 runs.
- Each candidate is a separate boot. No concurrent benchmark or outside client
  traffic; request/output counter deltas match every measured workload.
- Decode uses the existing helper's full-concurrency steady window. Prefill,
  startup compilation, warmup, and low-concurrency tails are excluded.
- Cached-input counts match across compared requests. These are warm,
  short-prompt decode tests, not cold-prefill or full-context measurements.
- Execution order: baseline, PRMT only, scheduler only, launch order only,
  side-stream off only, combined, baseline again.

Profiling remains off throughout. The baseline uses `streams` sampling,
`policy` draft depth, synchronous messages, expert launch order 0, PRMT off,
and shared-expert side stream on. Each single-factor row changes only the
listed setting(s); all other baseline values remain fixed.

## C4: individual changes

Rates are **aggregate steady decode tok/s**. Percentages use the mean of the
opening and closing baseline means: **126.488 tok/s**.

| Change from backports-only baseline | Three-run mean | Run range | Change |
| --- | ---: | ---: | ---: |
| Opening baseline | 126.527 | 126.352–126.661 | — |
| PRMT only: `TF_GLM_EXL3_DEC_PRMT=1` | 125.517 | 123.721–126.734 | -0.77% |
| Scheduler group: packed / joint / async | 128.205 | 128.100–128.334 | +1.36% |
| Expert launch order 2 only | 128.388 | 128.278–128.508 | +1.50% |
| Shared-expert side stream off only | 125.907 | 125.652–126.114 | -0.46% |
| Combined preset (all four changes) | 128.104 | 127.949–128.307 | +1.28% |
| Closing baseline | 126.450 | 126.300–126.668 | — |

The opening/closing baseline means differ by -0.06%. Versus the faster of the
two baseline means, scheduler, launch order, and combined gains are +1.33%,
+1.47%, and +1.25% respectively. PRMT's low second pass is retained, not
discarded as an outlier.

## C8: combined preset only

Individual switches were **not** isolated at C8.

| Configuration | Two-run mean | Run range |
| --- | ---: | ---: |
| Opening backports-only baseline | 162.545 | 162.480–162.611 |
| Combined preset | 167.942 | 167.880–168.005 |
| Closing backports-only baseline | 161.874 | 161.831–161.917 |

Combined gain: **+3.53%** over the bracketed baseline mean (162.210 tok/s),
or **+3.32%** over the faster baseline. Baseline drift was -0.41%.

## Interpretation and limits

The observed C4 improvements are from selecting existing scheduler options and
the existing expert launch order. This screen does **not** demonstrate a
standalone PRMT speedup or a benefit from disabling the side stream. Their
negative observed changes are not proof of a general regression either,
particularly given PRMT's spread. Do not advertise the earlier combined
PRMT-plus-launch-order measurement as a PRMT-only gain.

Effects are conditional on the backported-kernel baseline and are not additive.
This is not a full factorial study or a leave-one-out test of the combined
preset: it cannot establish the benefit of removing PRMT from that preset,
or assign its C8 improvement to individual switches.

All **132 measured requests / 270,336 output tokens** matched reference token
hashes and prompt/token-count checks. Containers retained their identities
within each case, with no OOMs or restarts during measurements. These are
within-boot repeats, not multiple independent boots of every candidate, so
there is no confidence interval or long-term stability/quality claim.

The [earlier whole-branch comparison](tp4-decode-tuning.md#measured-results-2026-10-10)
used a different C4 output cap (4,096 versus 2,048 here). Do not subtract these
percentages from that comparison to infer a separate backport contribution.

### Suggested PR performance wording

> With both backported kernels held enabled, the additional opt-in preset
> measured +1.28% aggregate steady C4 decode and +3.53% at C8 on matched greedy
> prose. At C4, scheduler-only and launch-order-only changes measured +1.36%
> and +1.50%; PRMT-only and side-stream-off showed no gain in this screen.
> Baseline runs bracketed the candidates, all 132 measured output token hashes
> matched, and results are bounded within-boot measurements rather than a
> claim of statistical confidence or universal speedup.
