# TensorFold v0.6.6 validation

This draft ports the public v1.10 recipe to TensorFold commit `cb2ebf0540f42604e2759b2ddef497861e928248`. Only `patches/v066/0001-glm-recipe-compat.patch` is applied; the historical v0.6.0 patches remain inactive. No private checkpoint or private abliteration implementation is included.

## Current merged runtime

The upstream merge includes the v1.9/v1.10 sparse-kernel loops, display backend, spill tier, kept-state limits, decode launch order, loop guard, draft candidate checks, Responses include values, streamed admission, effort-tail and cache-share options. Anthropic routes and chunked request framing are native to the pinned TensorFold source. The active patch identity is `0a998e123165`. This runtime differs from the earlier tested image and needs a fresh build and GPU validation. The upstream second-rail probe, repeated-character smoke check, defaults, credits and license documentation are retained.

## Earlier GPU observations (prior runtime)

These observations apply to the earlier recipe build identity `eda0320be7fc`, before the v1.9/v1.10 merge. They do not validate the current active patch.

One Docker image built from this Mia recipe was run on two independent tensor-parallel pairs (four DGX Sparks total). This is one build tested on two clusters, not four independent builds. Both used a compatible EXL3 checkpoint, q4 dense weights, FP8 KV, four lanes, and DFlash2 revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`. The default Mia checkpoints still need GPU smoke checks.

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

- Fresh merged-runtime build and stock Mia/gated Mia Ablit checkpoint GPU smoke checks, including sampled decoding with the new nucleus-union default.
- A full Docker build from a clean recipe checkout, followed by a maintainer-published image if prebuilt distribution is desired. The default remains `PULL=0`; no legacy digest is reused.
- Near-limit retrieval and a repeatable isolated performance comparison.
- Broader Responses/Anthropic streaming-client coverage, TP above two, alternative dense/KV/drafter configurations, and GPU equivalence checks for the sparse-kernel and spill/display options.
