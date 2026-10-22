# Registered-memory RDMA: qualification and NCCL comparison

**Four-host ring GPU qualification passed. Switched RDMA is implemented and code-tested; switched hardware is unverified.**

Native source `be94e919e35c628261bd38364b1743a2e4353a26`, patch0014 SHA `46111e72fa8432f5477a935a326ab4d8c7122cb7e3d4b29998e5791bb4d6dab2`, engine SHA `42408ed8447654e9018d5550db6d4ebde703cf858d03215219323504372bf33a`. Build receipts pin the exact compiled source and toolchain. The submission contains the same production source; documentation and receipt packaging do not change the tested binary.

## Performance

Same binary, hardware, model and serving configuration; the declared transport changes from NCCL to registered RDMA. Unmodified Grill 0.3.0 default C1/exact400, eight acquisitions and 32 requests per capture:

| Transport | Median acquisition tokens/s |
|---|---:|
| NCCL |157.09|
| Registered RDMA |184.41|

Official log-mean effect: **+16.99%, nominal model-based 95% interval [15.79%,18.21%]**. Displayed rates are medians of acquisition medians, a different estimator. Sequential captures on a short synthetic workload do not establish causality, equivalence, a global optimum or general serving speed. TTFT and settlement are included. MiaAI's unpublished 17/4/428/56-request suites remain unavailable; no absolute-speed reproduction or Python-versus-Zig claim.

## Evidence

- [Build receipt](build/build-receipt-rdma-be94e91.json), [embedded-byte proof](build/embedded-proof-be94e91.json), [four-host artifact equality](build/focused-image-artifacts-be94e91.json). Real NVCC 13.0.88/SM121, 2,889 source files checked, exact 56 exported fatbins, 51 reachable engine embeds and probe `fn_roce` match. Whole-image fingerprint, engine/probe and AOT agree across hosts.
- [49 executed native tests](native-test-names.log), [ARM64 semantic build](native-linux-semantic.log), [15 offline Bash5 launcher tests](launcher-bash5.log), [recipe reconstruction](reconstruction.log). The semantic build uses nondeployable placeholders. GPU qualification uses genuine NVCC outputs.
- [NCCL and registered probe summary](gpu/probe-result.json), raw per-rank logs and loaded-library observations under [gpu](gpu). Every rank passes eager bytes/rank/guards/NCCL oracle and 64 mixed graph replays. Registered replay advances 128 own operations and 256 relay stripes per rank with unchanged host enqueue counts.
- [NCCL capture](gpu/captures/focused-nccl), [registered-RDMA capture](gpu/captures/focused-registered-from-nccl). Original Grill request/response/capture/report bytes are unchanged. [Standalone verifier](gpu/verify.py) and [analysis](gpu/analysis.json) validate these inputs without external comparison datasets.
- Published fixture JSONL covers NCCL, registered graph and eager. Projection omits only the top-level presentation label; request/output fields and token arrays are unchanged. [Projection hashes](gpu/derivation.json) bind the source and published bytes. The profile has 40/40 exact output comparisons with matching returned dimensions. Qualifier JSONL does not retain full request bodies or seed; prompt hashes, draft flags, maximum output and returned dimensions are checked.
- [Final acceptance/load analysis](gpu/final-serving.json), [raw acceptance](gpu/captures/accept-focused-final), [load capture](gpu/captures/bench-focused-final-90s), [eager/graph settings](gpu/eager-graph-effective-mode.json). The 40-fixture runs qualify the same profile as the final serving containers; final IDs have fresh acceptance and load. Eager mode is verified by immutable config plus pinned source semantics, not a kernel trace.
- [Both-key LAN smokes](gpu/final-lan-smoke.jsonl) and [SSH/sudo/gateway observations](gpu/postopen-identity-gateways.json). Credentials remain in protected host files; published outputs contain labels and results only.
- [File manifest](manifest.json) lists every published file except itself.

## Serving and topology limits

Final four-client load: 57 complete generations, 14,592 tokens, zero worker/metadata errors. Pinned clipped 90-second client-window model: 156.51 tokens/s. Full-drain 93.455328 seconds: 156.14 tokens/s, a different denominator. TTFT p50/p95: 0.182/0.302 seconds. Acceptance has 31 settled cases and 12/12 exact comparisons. A 32,752-token needle finds the value in all four outputs; both reuse calls report 32,704 cached tokens. Continued role/thinking markers limit this to presence/parity. Cancellation proves client closure and exact follow-up, not backend retirement timing.

GPU evidence covers a two-rail-per-neighbor, dual-PCI-domain GB10 ring, context 131072, parallel 8, max output 32768, DFlash2/FP8 KV, CLI cache 3 GiB, env cache 32 GiB and L2PF1/SIDE1. Eligible gathers use registered 512 KiB slots; larger/ineligible gathers use NCCL. Processes actually load separately patched NCCL 2.30.7 SHA `1cdf5f3e2d617d5c07a87903eb8849d5bd7758f6c900455d613a2e0eb289d63e`; the flag does not install or qualify that library. All 16 HCA snapshots show traffic and zero error/congestion deltas. The snapshots include oracle traffic and do not establish QP endpoints.

Direct mode on the physical ring aborts with `NoSharedSubnet` then `RequiredRoceUnavailable`. A generic fallback diagnostic precedes that required-mode abort. This proves refusal on missing direct routes, not switched-hardware success.

Unqualified: switched hardware, single-rail ring hardware, GPU mixed-profile rejection, allocated relay, TP1/2/3, Flash Next, production peer-loss injection, full-window quality, vision, long-duration reliability, reboot, tailnet inference and private-assistant role. The matched TP4 NCCL profile passed; other hardware/settings require their own qualification.

The retained serving report pins its analyzer and benchmark/acceptance clients by hash, but their source files are not included here. The load-rate and TTFT figures are retained observations; the standalone verifier does not reproduce that load analysis.

No weights, serving binaries, credential values, gateway configurations or deployment controllers are included. Nonsecret labels, producer paths and hardware names/addresses remain provenance. Original synthetic raw captures retain their recorded paths. **BertholomusAI (Albert Lee, @bertholomus)** is credited for the registered-memory/DeepSeek RDMA work and benchmark client; existing project credits remain in CREDITS/NOTICE.
