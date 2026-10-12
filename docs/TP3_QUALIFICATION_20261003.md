# Independent TP3 qualification and receipt checklist

This is an opt-in **measured profile**, not a change to v1.5 defaults or production
certification. The [full independent report on PR #42](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/42#issuecomment-5971355725)
records the results, sample counts, negatives and audit hashes from 2026-10-03.
It covers three DGX Sparks/GB10 connected as a direct CX7 triangle, TP3/NCCL,
local weights, eight actual decoder slots and a shared FP8 KV pool.

## Measured profile

Recipe commit: `1576746a04983b6eded0551dbf22512ee9e95654` (v1.5).
Target revision: `078455ffe6472f9a52fbc1139f58b9db2881b25c`.
DFlash2 revision: `bf582e4eacc1810f76656d1811693ff6c6737d2a`.
Published image: `sha256:ef83797d791fef96c4605e8d37367aca6de5aeac7bb672792cb682e2e55d4237`.

After configuring the workers and fabric as described in [3 Sparks](../README.md#3-sparks-experimental),
these are the measured settings; keep your own addresses, paths and container name
out of a public report. `restart` interrupts requests: inspect `/health` and arrange
a maintenance window before applying any profile.

Put this block in `scripts/local.sh`, which the launcher sources; plain unexported
assignments pasted into a shell are not passed to a child launcher.

```bash
TP=3
COMM=nccl
PARALLEL=8
CONTEXT=1048576
KV=fp8
DENSE=q4
DRAFTER=dflash2
VISION=1
VISION_URLS=0
WORKER_WEIGHTS=copy
WORKER_WEIGHTS2=copy
KV_POOL_GIB=22
MEMORY_RESERVE_GIB=18.3
FILL_BUDGET_MS=200
FILL_DRAFTS=1
STREAM_SMOOTH=1
STREAM_SMOOTH_MS=400
export TF_GLM_MULTI_WINDOW=32
export TF_GLM_MULTI_LONE=0
export TF_GLM_CACHE_ENTRIES=32
export TF_GLM_KEEP_REASONING=0
```

The tested image additionally had local HTTP model-admission and host-owned
queued-caller cancellation guards, **not CUDA/kernel/graph/throughput patches**.
Thus the settings above alone do not reproduce those guard checks on stock v1.5.
The exact derived image ID and guard checks are in the linked report. The report
does not establish that lowering the default W64 window to W32 improves speed or
reliability: W48 never reached readiness, W64 had both ready and not-ready starts,
and their wait site/cause was not identified.

The observed shared pool was **4,202,496 logical tokens**, not eight reserved
million-token contexts. Head available RAM reached **12.491 GiB** in sampled
saturated tests: above the operator's 8 GiB hard floor, but below the 13 GiB steady
target. Do not copy a memory profile without measuring all three ranks under load.
Sampled head SM clocks were **2,398–2,411 MHz**, not normalized to the README's
2,200 MHz benchmark. These are not matched speedup measurements.

## Receipts that make comparisons useful

For each candidate and control, retain:

- Exact recipe/TensorFold/patch/image/checkpoint/drafter pins, full nonsecret
  settings, topology and per-rank clock policy; record local patches separately.
- Prompt catalog bytes/hash, seed, temperature, top-p, reasoning mode, reply cap,
  warmups, sample count, restart order and any overlapping work. Use equal sample
  counts and alternate candidate/control boots to expose boot-to-boot drift.
- Separate prose, numeric structured and actual CODE workloads. A 400-token code
  completion is a speed probe, not proof that the code executes or is correct.
- Per-stream actual prompt/completion/reasoning/cached tokens, HTTP status,
  finish reason, SSE/DONE state and first/last content timestamps relative to one
  shared monotonic origin. An absent cache count means unknown, not cold.
- The aggregate denominator and TTFT statistic. SparkDash's native decode measure
  excludes the first token and uses summed remaining tokens / earliest-first to
  latest-last content window; it is not summed per-stream rates or full wall rate.
  Wave-mean TTFT and pooled per-stream median TTFT are different statistics.
- `/health` before/during/after, owned request-counter deltas, available RAM on
  every rank, GPU clocks/temperature/throttle state, container restarts/OOM state,
  errors and observer cleanup. Missing telemetry remains unknown, not zero.

The linked three-repeat receipts lack absolute first/last timestamps and
per-stream cache counts. Per-stream rates and summaries were independently
checked, but the saved receipts alone cannot reconstruct the multi-stream
aggregate window. Future collectors should retain those fields.

## Bounded qualification, not just fastest decode

1. Check private/public health, model ID, complete streaming/nonstreaming replies,
   explicit JSON/schema, typed tools and bounded vision. Keep transport timeout
   behavior separate from engine timings.
2. Prove actual active slots, not merely submitted clients. Eight slots served a
   16-client wave by queueing; that does not establish 16-way active decoding.
3. Compare drafted-alone/batch with `draft: false` on the **same engine**, prompt,
   seed and sampling settings, in both orders. This is not BF16 quality parity.
4. Exercise a queued ninth-caller disconnect while eight holders remain unfinished,
   an active disconnect, and new cold prefill while incumbents still decode;
   cancel only owned calls, join observers and require final drain.
5. Test unique cold prompts and a full cached next turn separately. One retrieval
   request used 999,999 prompt tokens, cache 0, TTFT 912.093 s; its next turn used
   1,000,066 prompt tokens with 999,936 cached, TTFT 2.183 s. All four supplied facts
   were returned, but this is not broad document comprehension or simultaneous
   million-token capacity. The cold wait exceeds common 600/900-second budgets.
6. Keep startup failures, memory-floor misses and output-quality failures in the
   report. Same-profile restart and rollback checks do not prove host-reboot
   recovery. Plain-language "JSON only" and language quality need separate tests
   from schema-constrained outputs; do not normalize failed replies into passes.

Do not publish credentials, deployment addresses, private documents or raw user
prompts. Synthetic fixtures and nonsecret receipts are sufficient for this guide.
