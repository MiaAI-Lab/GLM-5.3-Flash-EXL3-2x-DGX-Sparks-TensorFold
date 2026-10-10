# Opt-in TP4 scheduler and expert launch-order preset

This change selects four existing runtime options. It adds no kernel patches,
PRMT dispatch, QMM/attention backports, private NCCL build or transport changes.
The shared-expert side stream, profiling default, precision, checkpoint, context,
memory limits, parallelism and prefill code remain unchanged.

Configure the workers normally, then:

```bash
TP4_DECODE_TUNING=1 ./start-tp4-switchless.sh
# Restart with the preset disabled:
TP4_DECODE_TUNING=0 ./start-tp4-switchless.sh restart
```

The preset requires TP=4 and is off by default. Per-setting overrides win, under
the recipe's existing environment/local-config precedence. Workers receive the
same exported settings through the normal launcher. The published image and
patch hash are unchanged; no image rebuild is required.

| Setting | Preset | Existing implementation |
| --- | --- | --- |
| `TF_GLM_MULTI_SAMPLER` | `packed` | Packed sampling collectives |
| `TF_GLM_MULTI_DEPTH` | `joint` | Joint draft-depth allocation |
| `TF_GLM_MULTI_ASYNC` | `1` | Asynchronous control messages |
| `TF_GLM_EXL3_DEC_ORDER` | `2` | Expert block launch order, patch 0090 |

This is a tested configuration choice, not a newly implemented scheduler or
expert kernel. Existing authorship is retained; see [CREDITS](../CREDITS.md).
Two- and three-Spark defaults are unchanged. Measurements on a switchless ring
do not establish a gain on a switched fabric or other hardware/workloads.

## Validation and reproduction

```bash
IMAGE=<published-image> scripts/test-cpu.sh -k tp4_decode_tuning
```

Tests check that the preset changes exactly four settings, preserves defaults
and individual overrides, and rejects invalid preset/TP combinations.

Run the baseline (preset off) and candidate (preset on) serially, on an idle
server, with the same weights/settings and identical warmup/cache preparation.
Use the bundled benchmark helper; the output caps below match the original
upstream-versus-fork screen:

```bash
python3 tools/tp4_decode_benchmark.py --concurrency 1 --max-tokens 2048 --label c1 --output c1.json
python3 tools/tp4_decode_benchmark.py --concurrency 4 --max-tokens 4096 --label c4 --output c4.json
python3 tools/tp4_decode_benchmark.py --concurrency 8 --max-tokens 2048 --label c8 --output c8.json
```

Use unique output names per repetition and boot; `--base-url` and `--model`
override endpoint defaults. The helper uses bundled varied prose, temperature
zero and thinking off. It measures aggregate health-counter decode rate while
all requested streams are decoding, excluding the first three and last two
seconds of their common span. SSE TTFT is recorded separately. It rejects
contaminated request/output counters and saves hashes rather than prompt/output
text. Compare token hashes, token counts and cache state across configurations.

## Exact-preset results

Measured 2026-10-10, source `336811f8c2c19e13bca2157dd4c7330d5a520b31`, against
upstream `tp4` at `d22003070575c2774de0f010062aa57bae06f3c7`. Both used the same
published image, patch hash `7a37454d3238`, with no additional kernel patches.
Exactly the four settings above differ; the shared-expert side stream stays on.

Four GB10 Sparks on a switchless ring, upstream NCCL/RoCE, the same 4bpw Ablit
checkpoint and DFlash2 drafter, 500,000 context, eight slots, FP8 KV / Q4 dense
KV, 32 GiB KV pool and 20 GiB memory reserve. Exact revisions, image digest,
counts and output token hashes are in the
[machine-readable results](benchmarks/tp4-scheduler-order-20261010.json).

| Concurrency | Upstream aggregate decode tok/s | Preset aggregate decode tok/s | Change | Mean TTFT, upstream / preset |
| --- | ---: | ---: | ---: | ---: |
| C1 | 65.03 | 65.60 | +0.88% | 0.363 / 0.365 s |
| C4 | 123.17 | 125.48 | +1.88% | 0.643 / 0.629 s |
| C8 | 156.14 | 162.54 | +4.10% | 0.570 / 0.574 s |

This is **one additional candidate boot**, compared with the retained upstream
run from earlier the same day, not a fresh bracketed A/B or a repeated
statistical study. Both used the same API/vision/tools/eight-stream warmup and
the same C1, C4, C8 sequence. All 13 measured requests matched prompt hashes,
output token hashes, usage and cached-input counts. Actual output totals were
2,048 / 11,746 / 16,384 tokens; partial cached input was 0 / 64 / 256 tokens.
Counter deltas matched the exact workload. No OOMs/restarts occurred during
measurement, and the 14 configuration tests passed.

These results support a modest, workload-specific observed benefit for selecting
the existing options together. They do not prove a universal speedup, improved
prefill, long-context quality, or a gain from a new kernel. Earlier broader
experimental-branch results included backports and other settings and are not
the baseline here; they remain in Git history, not in this proposed change.
