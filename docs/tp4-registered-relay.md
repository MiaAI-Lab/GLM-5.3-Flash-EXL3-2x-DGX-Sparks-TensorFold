# Optional registered-memory TP4 RDMA

`TP4_NETWORK=registered-rdma` enables registered-host-memory GLM collectives for the Zig engine on four hosts: the existing neighbor relay on a physical ring, or direct all-peer RoCE on a switched mesh. **The default is `TP4_NETWORK=upstream`**, preserving the existing TP4 ring/NCCL launch path. Switched registered RDMA is implemented and code-tested only; **it is not hardware-verified**. The profile uses the existing configuration files and fabric discovery.

```sh
# Physical ring, workers in cable order:
TP4_NETWORK=registered-rdma ENGINE=zig SPLIT=0 ./start-tp4-switchless.sh
# Switched mesh: code-tested only, not hardware-verified:
TP4_NETWORK=registered-rdma ENGINE=zig SPLIT=0 ./start-tp4.sh
# Existing fabric discovery may select either mesh or ring:
TP4_NETWORK=registered-rdma ENGINE=zig TP=4 SPLIT=0 ./start.sh
```

The profile defaults an unset `COMM` to `roce` on either topology and requires `ENGINE=zig`, `TP=4`, `SPLIT=0` and 512 KiB slots. `FABRIC_EXPECT=mesh` or `ring` checks an expected topology; unset allows the existing fabric discovery to choose. Explicit incompatible settings are rejected before host discovery, preparation, SSH or stopping a service. Use `TP4_NETWORK` rather than setting the internal native boolean `TF_GLM_TP4_REGISTERED_RDMA` directly. All ranks must run the same build and agree startup settings; mixed binaries are unsupported.

The setting can also be saved in `scripts/local.sh` or `.env` with the other required values, then used through `scripts/prepare.sh` and `./start.sh`. Both wrappers retain their existing environment override behavior; use `./start.sh` for a configuration selected entirely through local.sh/.env. Explicit `TF_GLM_L2PF` and `TF_GLM_SIDE` values are preserved; the measured ring profile used both at 1. No context or cache budget is forced by this networking flag.

## Startup agreement and failure handling

Default **backend selection semantics** are retained; startup validation and failure checks are strengthened for this native source. Native startup exchanges profile agreement before opening NCCL. Every rank must run the same build and agree on RoCE-enable, ring and prefetch settings. Runtime failure checks also cover optional allocated RoCE callers (sampler, snapshots, spill and in-place gathers). The shared region helper changes setup order, and the proxy diagnostic counter is atomic. The patch changes image identity even when opting out; default TP2/TP3 and Flash Next GPU regression checks remain outstanding. The matched TP4 NCCL control below passed at the tested profile; other profiles remain unqualified. The host launcher test compares selected effective settings, not complete binary behavior.

Select `TP4_NETWORK=upstream` or `TP4_NETWORK=registered-rdma` through the launcher. It supplies the internal native boolean and required RoCE settings, selecting ring or direct mode through existing fabric discovery. The library, discovered NIC/GID settings, model/arena and ports still require qualification for the intended deployment.

## Behavior and topology

This retains GLM's existing rank-major FP32 gather with its ring-relay or direct all-peer schedule, using anonymous host memory registered with CUDA instead of `cuMemHostAlloc`. It is GPU staging through host memory and CPU RDMA proxies, not direct GPU-memory RDMA. Custom transport handles eligible gathers up to **512 KiB per rank**; larger/ineligible gathers still use NCCL. At 4096 FP32 values per row, this covers 32 rows; 33–64-row verification windows remain NCCL. Setup is required for the opt-in: failure aborts startup rather than silently benchmarking NCCL. Once a custom operation has started, transport errors are sticky and fatal.

The launcher uses existing `detect_links`/`FABRIC_EXPECT` logic: a ring gets per-rank `TF_ROCE_RING=1`, while a switched mesh gets `TF_ROCE_RING=0`. Global ring overrides are excluded from the shared rank environment. Optional explicit `TF_ROCE_RING` accepts only 0 or 1: a conflict with explicit `FABRIC_EXPECT` fails before host calls, and a conflict with autodiscovered fabric fails before stopping the previous service. The wrong wrapper, missing peers and workers outside physical ring order fail closed. There is no independent topology selector.

Ring mode retains the existing subnet, device and GID discovery. **Both one-rail and two-rail-per-neighbor layouts are supported by routing code**; host tests cover both, shuffled per-rank device ordering, non-/24 subnet masks and missing-neighbor refusal. Direct mode uses the existing all-peer RoCE routes and requires a compatible subnet route to every other rank. It does not infer a physical switch or support arbitrary routed IP reachability. CPU tests cover direct routing with 1, 2 and 4 rails per rank, shared and multiple subnets, shuffled HCA order and /20 through /23 prefixes; they check symmetric queue-pair planning and reject missing direct peers. Launcher fixtures exercise both wrappers and autodiscovery, two/four local HCAs, uniform mesh CIDRs, crossed ring device ordering and all four rank environments.

No private hostname, rail name, subnet, GID or port is baked into this profile. No CUDA compute kernel, model arithmetic, collective byte class, prefetch or stream-ordering change is introduced by exposing direct registered routing. **GPU evidence covers a two-rail-per-neighbor, dual-PCI-domain GB10 ring.** Switched hardware remains unverified and needs its own byte/graph and model gates.

A known-working NCCL library is a prerequisite on both topologies for fallback collectives. The benchmark ring fleet used separately patched **NCCL 2.30.7**, SHA-256 `1cdf5f3e2d617d5c07a87903eb8849d5bd7758f6c900455d613a2e0eb289d63e`; see [public sparkring patches](https://github.com/FujitsuPolycom/sparkring/tree/4b9b6a0a213456b96400c5e1a1cf59c20ff892c8/spark_transport/nccl). The recipe's stock NCCL wheel and dependency pins remain unchanged. This flag configures neither physical wiring nor that library; a version string alone is not compatibility proof. The ring result does not qualify the stock wheel for that setup or any switched deployment. Both launcher and native direct startup warn about the switched qualification limit.

## Rationale and attribution

Registered host memory is the transferable improvement from the DeepSeek RDMA work by **BertholomusAI (Albert Lee, [bertholomus](https://github.com/bertholomus))**. GLM already has ring relay and direct all-peer schedules with full rank-major gathers. This profile changes host-memory allocation/registration while preserving GLM's collective placement, compute math, prefetch hooks and consumers. See [CREDITS](../CREDITS.md), [NOTICE](../NOTICE) and [source provenance](../zig/TRANSPORT_SOURCE.json). Separate [design exploration](https://github.com/swbiggart/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/1) records alternative approaches.

## NCCL comparison

The same binary and serving configuration measured **157.09 NCCL versus 184.41 registered-RDMA median acquisition tokens/s** with unmodified Grill 0.3.0 default C1/exact400. Each capture completed eight acquisitions and 32 requests. The official log-mean effect is **+16.99%, nominal model-based 95% interval [15.79%,18.21%]**. This includes TTFT and settlement and describes sequential captures on one short synthetic workload; it is not a causal, equivalence, global-optimum or general serving-speed guarantee. [Raw captures and standalone verifier](tp4-rdma-focused/README.md).

## Qualification

Native source `be94e919e35c628261bd38364b1743a2e4353a26`, recipe patch0014 SHA `46111e72fa8432f5477a935a326ab4d8c7122cb7e3d4b29998e5791bb4d6dab2`, engine SHA `42408ed8447654e9018d5550db6d4ebde703cf858d03215219323504372bf33a`. The build receipt records the exact compiled source, recipe and toolchain; the submission retains the same production source. Patch0014 applies to MiaAI zig-engine `a6c7e08ae864239125d45fc15146d26ec74c7f6d` after its 13 existing patches. The source verifier compares 2,889 files and explicitly accounts for six upstream executable-mode differences.

The real NVCC 13.0.88/SM121 build exports exactly 56 fatbins; all 51 reachable engine embeds and the probe's `fn_roce` match exported bytes. Whole-image fingerprint, engine/probe and AOT hashes agree across four hosts. Native host tests: 49; ARM64 semantic build steps: 9/9; offline Bash5 launcher tests: 15. Semantic placeholders are nondeployable; GPU qualification uses genuine NVCC outputs.

NCCL and registered probes pass eager byte/rank/guard/oracle checks and 64 changing-input mixed graph replays on every rank. Registered replay advances 128 own operations and 256 relay stripes per rank, with unchanged host enqueue counters. Uniform direct mode on the physical ring aborts with `NoSharedSubnet` then `RequiredRoceUnavailable`; a generic fallback diagnostic precedes that required-mode abort. It does not qualify switched hardware.

The profile passes 40 exact fixture outputs across NCCL, registered graph and eager with equal returned dimensions. Final containers of the same graph profile pass fresh acceptance and a 90-second four-client load: 57 complete generations, zero errors. The clipped client-window model reports 156.51 tokens/s; full-drain rate is 156.14 tokens/s. Acceptance has 31 settled cases and 12/12 exact comparisons. A 32,752-token needle checks value presence/parity, with continued role markers; cancellation proves client closure and exact follow-up, not backend retirement. Detailed limits and raw receipts are in the [evidence index](tp4-rdma-focused/README.md).

The tested serving profile uses context 131072, parallel 8, max output 32768, DFlash2/FP8 KV, CLI cache 3 GiB, L2PF1/SIDE1. Switched and single-rail ring hardware remain unverified. Mixed-profile rejection, allocated relay, TP1/2/3 and Flash Next remain GPU-unqualified. No production peer-loss injection, full-window quality, vision, long-duration reliability or reboot qualification. MiaAI's unpublished 17/4/428/56-request fixtures remain unavailable; no Python-versus-Zig or absolute-speed reproduction claim.

## Weight-free probe before model qualification

Build the Linux CUDA/libibverbs `glm-transport-probe` target with genuine fatbins. Launch one `tf-glm-transport-probe RANK MASTER_IP PORT 64` per rank under an external fleet deadline on the intended hardware and NIC/GID configuration. For a ring, all ranks require:

```sh
TF_GLM_TP4_REGISTERED_RDMA=1 TF_GLM_ROCE=1 TF_ROCE_RING=1 TF_ROCE_MAX_KB=512 TF_GLM_L2PF=0
```

For switched direct mode, use the same flags with `TF_ROCE_RING=0` and compatible subnet routes to every peer. This is a qualification procedure; switched hardware has not passed it.

For an NCCL reference, use `TF_GLM_TP4_REGISTERED_RDMA=0 TF_GLM_ROCE=0` on every rank. The probe compares bytes against NCCL, stamps changing inputs, checks rank placement/guards and captures custom/NCCL/custom graph calls with execution-counter barriers. Ring mode requires relay progress; direct mode requires zero relay stripes and advancing own-operation counts. Host enqueue counts must stay unchanged during replay. The probe uses empty prefetch work; serving math and nonempty prefetch/side-stream interaction need model-level exact-token/eager-graph gates. The external supervisor must bound peer hangs; the probe alone does not qualify production peer-loss handling.
