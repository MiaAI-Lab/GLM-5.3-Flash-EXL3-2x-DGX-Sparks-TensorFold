# Matched GLM v1.5 measurements

Actual private synthetic runs on two NVIDIA DGX Sparks (GB10), not dashboard screenshots or vendor headline rates. Both variants used TensorFold v0.6.0, the v1.5 recipe, TP=2, PARALLEL=4, configured context 1,048,576, DFlash2, temperature zero, 4,096-token reply budgets and an 18.5 GiB host reserve.

Recipe base: `1576746a04983b6eded0551dbf22512ee9e95654`.
Image: `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:ef83797d791fef96c4605e8d37367aca6de5aeac7bb672792cb682e2e55d4237`.
Exact target/drafter pins are in [setup](hermes-agent.md). Both hosts hash-verified the latest standard snapshot/drafter (103 files) and the abliterated snapshot before use.

## Concurrency and functional checks

Separate short counting requests proved 1/2/4 actual simultaneous decoders. Aggregate end-to-end rates (total returned tokens/group wall time) were:

| Requests | Standard tok/s | Abliterated tok/s |
|---|---:|---:|
| 1 | 97.38 | 95.60 |
| 2 | 127.28 | 133.80 |
| 4 | 184.97 | 190.47 |

These are short-counting aggregate rates, not long-form per-user sustained decode. Both variants passed a two-turn function-call loop, an arithmetic reasoning answer, a synthetic red-image query, incremental SSE completion and actual-model identity checks. The abliterated model's generated LRU implementation and accompanying tests were extracted and executed locally: 16 tests passed. This is one code-quality example, not a comprehensive intelligence or refusal benchmark.

GPU memory totals/usage were unavailable through NVML on these integrated GPUs. MemAvailable is host telemetry, not a GPU-allocation measurement, and the reported minimum covers the head only. Kernels/drivers stayed unchanged between variants: 6.17.0-1032-nvidia/580.178.04 on the head; 6.17.0-1029-nvidia/580.173.02 on the worker.

Additional abliterated retrieval checks passed with thinking enabled and a 4,096-token reply budget: 47,863 prompt tokens uncached, and 248,443 prompt tokens with 47,808 cached. Both returned the exact requested record values. These are two correctness probes, not a broad long-context accuracy benchmark or full-window stress test.

## Scope and attribution

No full-window 1M stress claim, broad language benchmark, safety/refusal evaluation, commercial-license clearance or independent external review is implied. The optional abliterated weights are not made the upstream default. No kernel, serving-argument or network changes are part of this contribution.

Credit: Mia's AI Lab for the deployment recipe and standard quantization; Ash Hart for TensorFold; bullerwins for the abliterated checkpoint; incoai for DFlash2.

## Eight matched single-request runs per variant

| Metric | Standard | Abliterated |
|---|---:|---:|
| Median client decode tok/s | 57.43 | 55.30 |
| Mean client decode tok/s | 57.85 | 55.09 |
| Fastest complete measured run tok/s | 69.48 | 66.56 |
| Peak 0.5-second counter sample tok/s | 101.47 | 103.68 |
| Median time to first generated token, seconds | 0.1768 | 0.1774 |
| Minimum head MemAvailable, GiB | 15.76 | 14.34 |

Prompts: bicycle-maintenance prose and executable Python LRU/TTL code, each repeated three times; one synthetic 30,350-token record prompt; one reasoning/code prompt. The long prompt was warm in both variants (30,336 cached tokens). All accepted timing runs showed one decoder and counter-token deltas equal to their own usage. A contaminated preliminary run was excluded and the suite rerun under exclusive admission.

Client decode = (completion tokens - 1)/(last-event time - first generated-token time). The mean and median aggregate per-run rates, not pooled throughput. A short counter-window peak is NOT sustained decode. The complex reasoning prompt exhausted all 4,096 output tokens in reasoning in both variants: it tests the budget/path, not a successful final-code answer.
