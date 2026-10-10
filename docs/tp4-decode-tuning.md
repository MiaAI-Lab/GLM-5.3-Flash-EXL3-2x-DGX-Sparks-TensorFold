# Opt-in TP4 decode tuning

Based on the `tp4` branch at `d22003070575c2774de0f010062aa57bae06f3c7`.
This is a small, portable subset of measured decode experiments, integrated as
numbered TensorFold patches rather than runtime source rewriting or monkey
patches. It keeps this branch's prefill code and native NCCL/RoCE ring transport.
It does not bundle a separate NCCL build, topology-specific relay, machine
addresses, model paths, or service launchers. Results from a different transport
must not be represented as measurements of this branch.

## Usage and scope

Configure the workers normally, then:

```bash
TP4_DECODE_TUNING=1 ./start-tp4-switchless.sh
# Later, restart with the opt-in disabled:
TP4_DECODE_TUNING=0 ./start-tp4-switchless.sh restart
```

The preset requires `TP=4`. It does not change `PARALLEL`, context, checkpoint,
KV precision, memory limits, network topology, or port. Environment/local config
precedence remains the recipe's. Each individual override below wins over the
preset. Unset/0 leaves the upstream defaults intact, including on TP2/TP3.

| Setting | Preset value | Purpose |
| --- | --- | --- |
| `TF_GLM_QMM_CLUSTERS` | `0` | Partial-buffer reduction for <=64 rows and <28 Mi weight values; `1` restores clusters |
| `TF_GLM_SEG_CHUNKS_CUDA` | `1` | CUDA chunk attention at 16 heads, FP8 KV, BF16 queries, 1–64 rows; native merge |
| `TF_GLM_EXL3_DEC_PRMT` | `1` | Existing PRMT unpack helper in decode; separate compiled cache identity |
| `TF_GLM_EXL3_DEC_ORDER` | `2` | Existing exact-arithmetic expert block ordering |
| `TF_GLM_MULTI_SAMPLER` | `packed` | Existing packed sampler |
| `TF_GLM_MULTI_DEPTH` | `joint` | Existing calibrated joint draft-depth choice |
| `TF_GLM_MULTI_ASYNC` | `1` | Existing asynchronous control messages |
| `TF_GLM_MULTI_PROFILE` | `0` | No per-round diagnostic profiling |
| `TF_GLM_SIDE` | `0` | Shared expert on the original stream for this preset |

The attention replacement is qualified for GB10 / Triton3.7.1 only. It loads
before graph capture, keeps the source kernel's padded24-head layout and
arithmetic, and admits only TP4's16-head shape. Explicit head-tile overrides,
other shapes/precisions, an unsupported runtime, or a build failure keep Triton.
The Q4 reduction choice is fixed for the process lifetime; restart to change it.
PRMT builds use distinct extension names for on/off, never a stale cached kernel.

These are opt-in changes, not a blanket recommendation for a switch, TP2, TP3,
BF16 KV, other GPUs, or long-context quality. Synthetic equality and greedy
prose equality are regression checks, not a substitute for those evaluations.

## Provenance

- `0110` ports **BadAd84**'s main-branch `0103-glm-qmm-decode-noclusters.patch`
  from `11619a191999398c6392e8079fe4dcc545cc3f1a`; the heuristic and arithmetic are
  unchanged, but its default is inverted here so it remains opt-in.
- `0111` ports the CUDA kernel from **BadAd84**'s main-branch
  `0108-glm-seg-chunks-cuda.patch` (same source revision), retaining its original
  warp roles and partials/merge contract. This adaptation changes the head guard
  to16 and supplies standalone integration for `tp4`, without importing the
  main branch's unrelated prompt/indexer patches.
- `0112` reuses TensorFold/MiaAI's existing `decode_tile_p` helper in the decode
  kernel's two load paths. The helper, expert arithmetic, and prompt kernels
  are not rewritten. The selectable decode dispatch and TP4 integration are
  contributed by virtualkevin.

See [CREDITS](../CREDITS.md) and [NOTICE](../NOTICE). Existing attribution is kept.

## Checks and benchmark protocol

```bash
IMAGE=<locally-built-image> scripts/test-cpu.sh -k tp4_decode_tuning
IMAGE=<locally-built-image> scripts/test-gpu.sh -k tp4
```

Stop inference before GPU tests. Checks cover default preservation, override
precedence, build-cache identities, fallback on unsupported Triton, bitwise
PRMT operands, Q4 cluster/buffer equivalence across the size/row thresholds,
and exact attention outputs **and all chunk partials**, including dense/sparse
boundaries, omitted token lists and replay with new query data.

Run the following on a healthy, idle server for the baseline and the candidate,
serially, with the same weights and settings. No other client may generate.

```bash
python3 tools/tp4_decode_benchmark.py --concurrency 1 --max-tokens 2048 --label baseline-c1 --output baseline-c1.json
python3 tools/tp4_decode_benchmark.py --concurrency 4 --max-tokens 4096 --label baseline-c4 --output baseline-c4.json
python3 tools/tp4_decode_benchmark.py --concurrency 8 --max-tokens 2048 --label baseline-c8 --output baseline-c8.json
```

Change `--label`/`--output` for the candidate; use `--base-url` and `--model` for
an endpoint with non-default configuration. Requests use temperature0, thinking
off, and distinct free-form prose prompts in `tools/tp4_prose_prompts.json`.
Natural end-of-message is allowed; report actual output counts. Compare returned
token hashes, not just response validity. No endpoint, prompt or output text is
written by the helper; it keeps hashes, timing, usage and counter evidence.

Decode is the aggregate `/health` output-counter slope while **all** requested
streams are decoding, starting3s after the last first token and ending2s before
the earliest last token. No sufficiently long window means a null score, not an
estimated maximum. TTFT is client-observed SSE latency. Counter deltas must
match the test's exact request/output totals; otherwise the run is contaminated.
For prefill, use `--prompts` with a JSON list of fresh long prompts, confirm zero
cached tokens, and report active prefill separately from TTFT. Summing request
prefill seconds does not yield aggregate cluster throughput when work is shared.

Benchmark results for the exact contribution build will be appended after validation.
