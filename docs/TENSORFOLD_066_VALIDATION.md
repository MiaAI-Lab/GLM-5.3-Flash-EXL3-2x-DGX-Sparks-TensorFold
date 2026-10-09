# TensorFold v0.6.6 validation

This draft ports the public v1.10 recipe to TensorFold commit `cb2ebf0540f42604e2759b2ddef497861e928248`. Only `patches/v066/0001-glm-recipe-compat.patch` is applied; the historical v0.6.0 patches remain inactive. No private checkpoint or private abliteration implementation is included.

## Current merged runtime

The upstream merge includes the v1.9/v1.10 sparse-kernel loops, display backend, spill tier, kept-state limits, decode launch order, loop guard, draft candidate checks, Responses include values, streamed admission, effort-tail and cache-share options. Anthropic routes and chunked request framing are native to the pinned TensorFold source. The active patch identity is `0a998e123165`. This runtime differs from the earlier tested image. A fresh image was built from the tracked source archive of merge commit `2486f456bb69482182ee763aa3fa373631e48905`, reusing cached dependency layers (not an uncached rebuild), and exercised with the stock Mia and Mia Ablit checkpoints. The upstream second-rail probe, repeated-character smoke check, defaults, credits and license documentation are retained.

## Current Mia GPU observations

Both checkpoints loaded on the same tensor-parallel pair with four lanes, the configured 1,048,576-token context limit, q4 dense weights, FP8 KV, split prefill and nucleus-union sampling enabled. Both ranks matched the new image and the recipe's pinned checkpoint for each test. No context fallback or container restart occurred.

| Observation | Stock Mia | Mia Ablit |
| --- | ---: | ---: |
| Functional checks | 10/11 | 10/11 |
| Rank 0 startup estimate | 88.09 GiB | 88.09 GiB |
| Shared KV pool at startup | 2,535,424 tokens | 2,582,528 tokens |
| Model/kernel load | 429.8 s (cold) | 115.4 s (warm) |
| API readiness | 441.49 s | 132.3 s |
| 32k-tier needle | 33,046 tokens, 24.643 s, pass | 33,031 tokens, 23.193 s, pass |
| 128k-tier needle | 133,416 tokens, 96.106 s, pass | 133,494 tokens, 95.784 s, pass |

The load timings compare cold and warm starts and do not establish a speed difference between checkpoints. Pool capacity depends on startup memory. The configured maximum context was accepted at startup; retrieval at that maximum remains unverified.

Both checkpoints passed chat, SSE, tool roundtrip, Responses, Anthropic messages/count_tokens, vision, prefix-cache, and four-lane queue/cancellation checks. Valid Responses include fields and sampled decoding with `top_k=0`, temperature 0.7, top_p 0.9, min_p 0.05 and seed 2026 also passed bounded checks.

The single failing functional check was `reasoning_on_off`: its arithmetic prompt expected 34, but thinking disabled returned 40; thinking enabled returned 34 with reasoning content. This happened on both checkpoints. A focused stock Mia comparison with `draft: false` (policy 0) returned the same incorrect 40 as the drafted request, with identical output hashes and no cached prompt tokens. The cause remains unassigned, and this diagnostic does not establish general drafted-versus-serial correctness.

## Earlier GPU observations (prior runtime)

These observations apply to the earlier recipe build identity `eda0320be7fc`, before the v1.9/v1.10 merge. They do not validate the current active patch.

One Docker image built from this Mia recipe was run on two independent tensor-parallel pairs (four DGX Sparks total). This is one build tested on two clusters, not four independent builds. Both used a compatible EXL3 checkpoint, q4 dense weights, FP8 KV, four lanes, and DFlash2 revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`. These historical observations do not establish default Mia checkpoint behavior.

- Both clusters passed 11/11 functional checks: health/model discovery, chat, SSE, thinking on/off, typed tool roundtrip, Responses, Anthropic messages/count_tokens, vision, prefix reuse, and four-lane queue/cancellation recovery. Thinking content emission was observed on the second pair. This does not establish compatibility with every API client.
- Prefix-cache counters changed from 0 to 1,664 tokens. Four requests ran concurrently; a fifth queued, and cancellation recovered the lane.
- Pair A recovered the needle at 33,046 and 133,494 prompt tokens. Pair B recovered it at 33,061 and 133,416 tokens (25.349 and 100.049 seconds).
- Each shared KV pool reported 2,203,648 tokens at startup. Available memory and configuration determine this capacity. Both clusters remained healthy with zero container restarts after validation.
- Cold model/kernel loading took 433.5 and 440.6 seconds; API readiness took approximately 435 and 444 seconds.

A 949,712-token probe on the first pair reached the client's 1,100-second timeout without an answer. Other traffic overlapped, so the cause is undetermined. The server remained healthy, released its reservation, and answered a subsequent small request in 0.173 seconds without a restart. Retrieval at the configured 1,048,576-token maximum is unverified. A small single-trial performance comparison had mixed results; no overall speed improvement is claimed.

## Local checks

The merged candidate passed exact-commit patch application without offsets/fuzz and a reverse dry-run, parsing of 420 TensorFold Python files and all public Python tools/tests, 27 dependency-free scheduler/cancellation/capacity/shutdown/spill regressions with zero device calls, all recipe shell syntax checks, loop-guard, Responses include, chunked-body and streamed-admission regressions. The sampled nucleus-union CPU test matched 180 draws over 60 cases. Effort-tail template checks passed across five settings with chat and tool histories. These source checks do not establish GPU behavior. Earlier preparation also recorded build identity and nonstreamed Responses/Anthropic Retry-After checks.

To repeat the bounded runtime checks on an idle test server:

```sh
python3 -B tools/validate_tf066_runtime.py --self-check
python3 -B tools/validate_tf066_runtime.py --base-url http://127.0.0.1:8888 --functional
python3 -B tools/validate_tf066_runtime.py --base-url http://127.0.0.1:8888 --long-context
```

The runtime tool prints scalar results, not prompts or replies. Long-context mode is limited to the explicit 32k and 128k tiers. These commands send inference requests except for `--self-check`.

## Remaining gates

- Investigate the non-thinking arithmetic failure on both Mia checkpoints; overall default-checkpoint verification remains incomplete.
- A maintainer-published image if prebuilt distribution is desired. The image built from the clean tracked source used cached dependency layers. The default remains `PULL=0`; no legacy digest is reused.
- General drafted-versus-serial/bitwise correctness and memory behavior at maximum concurrent context.
- Near-limit retrieval and a repeatable isolated performance comparison.
- Broader Responses/Anthropic streaming-client coverage, TP above two, alternative dense/KV/drafter configurations, and GPU equivalence checks for the sparse-kernel and spill/display options.
