## Registered-memory RDMA for Zig TP4

Extend the Zig TP4 ring support merged in [#200](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/200). This submission targets MiaAI’s `zig-engine` branch at `a6c7e08ae864239125d45fc15146d26ec74c7f6d`; it adds an optional transport profile to that branch. The implementation retains GLM’s existing collective placement, rank-major gather layout and relay/direct routing schedules.

Add `TP4_NETWORK=registered-rdma` as an opt-in networking profile. Eligible gathers up to **512 KiB per rank** use registered host memory with GLM's existing ring relay or direct all-peer RoCE routes. Larger/ineligible gathers retain NCCL. The default remains `TP4_NETWORK=upstream`. CUDA compute kernels and model arithmetic are unchanged.

**Four-host ring GPU qualification passed. Switched support is implemented and code-tested; switched hardware is unverified.**

Configuration follows the existing `.env` / `scripts/local.sh`, launcher wrappers and fabric discovery. `FABRIC_EXPECT=ring|mesh` can assert the topology; discovery supplies `TF_ROCE_RING=1|0`. Conflicting flags, unsupported engine/TP/split combinations, missing workers and incorrect ring order fail closed. The launcher supplies internal `TF_GLM_TP4_REGISTERED_RDMA=1`, requires RoCE and 512 KiB slots, and preserves existing L2PF/SIDE overrides. It does not force context or cache budgets.

Startup settings must agree across all ranks before NCCL opens. Registered setup is required; failures cannot silently benchmark a fallback backend. Runtime transport failures are sticky/fatal. Default backend selection remains unchanged, while startup validation and failure checks are strengthened for all callers of the modified communicator. All ranks must run the same build and settings.

## Why this implementation

The [transport/configuration experiments](https://github.com/swbiggart/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/1) drove the transport and configuration choices:

- **Registered memory with GLM’s relay:** the initial four-candidate C1/exact400 screen observed 180.88 tokens/s for registered relay with 512 KiB slots, versus 178.74 for allocated-memory relay and 178.41 for the DeepSeek-style two-stage neighbor transport. This supports transferring Bertholomus’s registered-memory approach while retaining GLM’s existing gather schedule and arithmetic.
- **512 KiB cutoff:** registered 1024 KiB slots measured 180.40 tokens/s and showed no material advantage in the short serving screen. We retained the existing 512 KiB cutoff, with NCCL handling larger/ineligible gathers.
- **L2 prefetch and shared-expert side stream:** the 2×2 L2PF/SIDE screen had its highest observed rates with both enabled. The small SIDE margin is descriptive, not an established independent gain. The measured profile uses `TF_GLM_L2PF=1` and `TF_GLM_SIDE=1`; the installer preserves explicit overrides.

The linked research PR retains the alternatives, configuration screens, raw receipts and limitations. These experiments explain the selection; they do not prove a global optimum. The NCCL comparison and qualification below identify the source and binary submitted in this PR.

## NCCL comparison and qualification

[Receipts and file hashes](https://github.com/swbiggart/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/921ce7e7ba8ce7ad6767fb0d8a99ef5c05a88cf1/docs/tp4-rdma-focused) · [Raw captures](https://github.com/swbiggart/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/921ce7e7ba8ce7ad6767fb0d8a99ef5c05a88cf1/docs/tp4-rdma-focused/gpu/captures)

Same binary, hardware, model and serving configuration; only the declared transport changes. Unmodified Grill 0.3.0 default C1/exact400, eight acquisitions and 32 requests per capture:

| Transport | Median acquisition tokens/s |
|---|---:|
| NCCL |157.09|
| Registered RDMA |184.41|

Official effect: **+16.99%, nominal model-based 95% interval [15.79%,18.21%]**. The effect uses log means, rather than the ratio of displayed medians. This is a sequential comparison on one short synthetic workload, including TTFT and settlement; no causal, equivalence or general serving-speed claim. This comparison does not reproduce the separate 17/4/428/56-request suites reported for #200, its Python-versus-Zig comparison or its absolute speed.

- Real NVCC 13.0.88/SM121 build; 2,889 source files checked; exact 56 exported fatbins, 51 reachable engine embeds and probe `fn_roce` match. Matching image content, engine/probe and AOT across four hosts. Native source `be94e919e35c628261bd38364b1743a2e4353a26`, engine SHA `42408ed8447654e9018d5550db6d4ebde703cf858d03215219323504372bf33a`.
- 49 native host tests, 9/9 ARM64 semantic build steps, 15 offline launcher tests. Semantic placeholders are nondeployable; GPU qualification uses genuine NVCC outputs.
- NCCL and registered probes pass on all four ranks: eager byte/rank/guard/NCCL-oracle checks and 64 changing-input mixed graph replays. Registered replay advances 128 own operations and 256 relay stripes per rank with unchanged host enqueue counts.
- Direct mode on the physical ring aborts with `NoSharedSubnet` then `RequiredRoceUnavailable`; this negative test does not qualify switched hardware.
- 40 exact fixture outputs match across NCCL, registered graph and eager with matching returned dimensions. The qualification’s final containers used that graph profile and passed fresh acceptance and load.
- Final 90-second four-client load: 57 complete generations, 14,592 tokens, zero errors; 156.51 tokens/s under the client's clipped-window model, or 156.14 tokens/s over the 93.455-second full drain. Acceptance: 31 settled cases, 12/12 exact comparisons. A 32,752-token needle checks presence/parity; role-marker continuation limits clean-answer claims. Cancellation proves client closure and exact follow-up, not backend retirement timing.

## Deployment scope

GPU evidence covers a two-rail-per-neighbor, dual-PCI-domain GB10 ring with DFlash2/FP8 KV, context 131072, parallel 8, max output 32768, CLI cache 3 GiB and L2PF1/SIDE1. Switched direct routing and single-rail ring routing have CPU coverage only. Direct mode requires compatible subnet routes to all three peers; arbitrary routed IP reachability is unsupported.

NCCL remains necessary for larger/ineligible gathers and the engine’s other NCCL collectives. The measured fleet loads separately patched NCCL 2.30.7 SHA `1cdf5f3e…`; recipe stock dependency pins are unchanged. This flag does not install or qualify that library or physical wiring. Switched byte/oracle, graph, exact-token and performance gates remain outstanding. GPU mixed-profile rejection, allocated relay, TP1/2/3, Flash Next, production peer-loss injection, full-window quality, vision, long-duration reliability and reboot remain unqualified.

Credit to **BertholomusAI (Albert Lee, @bertholomus)** for the DeepSeek RDMA/registered-memory work and benchmark client. The implementation retains GLM's gather schedule and arithmetic. Existing MiaAI, Hector Isaza, TensorFold, Jürgen Schmied and b12x credits remain.

**Maintainer preference:** Would you prefer merging this as an opt-in addition to `zig-engine`, as implemented here, or developing it into a replacement for the current TP4 default after broader qualification? This submission keeps the existing default and NCCL fallback.

No model weights, serving binaries, credential values, gateway configurations or deployment controllers are included. Nonsecret labels, hostnames/addresses and producer paths may appear in receipts.
