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

Validation of this reduced preset against the unmodified upstream TP4 image
is pending. Earlier experimental-branch measurements included kernel backports
and other settings and must not be attributed to this reduced change.
