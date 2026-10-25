# Optional registered-memory TP4 ring relay

`TP4_NETWORK=registered-rdma` selects registered host memory for GLM's existing four-rank physical-ring relay. **Ring only:** the launcher refuses discovered mesh, while native startup requires relay mode and rejects advertised diagonal subnet routes. Neither verifies physical cabling or isolation. The default `TP4_NETWORK=upstream` leaves this branch's prerequisite image unchanged. This local revision includes pinned, unmerged PR #214; it therefore has different shared image inputs from bare upstream `697d8e9`, independently of the transport selection.

```sh
# Workers must follow the physical cable order around the ring.
TP4_NETWORK=registered-rdma ENGINE=zig SPLIT=0 ./start-tp4-switchless.sh
# Discover the fabric, then refuse anything except a physical ring.
TP4_NETWORK=registered-rdma ENGINE=zig TP=4 SPLIT=0 ./start.sh
```

Save the same settings in `scripts/local.sh` or `.env` for `./start.sh` and `scripts/prepare.sh`. The profile requires `ENGINE=zig`, `TP=4`, `SPLIT=0`, `COMM=roce` (the opt-in default) and fixed 512 KiB slots. `FABRIC_EXPECT=ring` asserts the topology; unset permits discovery followed by ring validation. `FABRIC_EXPECT=mesh`, the switched wrapper and `TF_ROCE_RING=0` fail before host calls. A discovered mesh, missing peer or incorrect ring order fails before stopping an existing service. Standalone `./stop.sh` remains usable with stale launch settings.

Select the public `TP4_NETWORK` setting; the launcher supplies `TF_GLM_TP4_REGISTERED_RDMA=1`, `TF_GLM_ROCE=1` and per-rank `TF_ROCE_RING=1`. Contradictory native flags fail closed. Direct invocation of the patched binary also requires `TF_ROCE_RING=1`. After exchanging identical route rows, every rank rejects any advertised subnet route between opposite ranks before connecting/starting the RDMA proxy. This blocks advertised meshes and partial diagonals even when relay mode is requested. Subnet data cannot distinguish physical cables from a logical ring built on switched hardware; direct binary invocation on such hardware remains unsupported. Operators must verify physical cabling and isolation independently. Existing discovery supplies the rank-local HCA order, GIDs, addresses and NCCL graphs. One- and two-rail-per-neighbor route planning has CPU coverage; historical GPU evidence covers two rails per neighbor across two PCI domains.

## Fabric isolation is required

Run only among trusted fleet hosts on dedicated, nonrouted point-to-point RDMA links. Do not bridge or route those links into client, management-LAN or tailnet networks, or attach untrusted hosts. Protect the bootstrap/control listener (`MASTER_ADDR`/`MASTER_PORT`) with host/network access controls permitting only the intended ranks; the existing TP4 listener may bind a wildcard address. The authenticated inference gateway does not protect this control listener or the RDMA fabric.

The peers exchange RDMA addresses and remote keys for remotely writable pinned host regions. Those are write capabilities: disclosure or an untrusted peer can compromise transport memory and inference integrity. The settings digest and protocol tag provide compatibility checking, not peer authentication or encryption. IP reachability alone is not proof of access to an arbitrary registered region. This profile does not install firewall rules, VLANs, credentials or isolation; verify those controls before any deployment. A shared switched fabric is outside this profile's scope.

## Selected image and startup protocol

The optional patch is `zig/transports/0024-zig-glm-tp4-registered-relay.patch`, outside the shared numbered `zig/patches` series. `scripts/prepare.sh` stages it after the shared patches only for `registered-rdma`, using the same `zig/Dockerfile`. The image hash covers that selected patch. Its `0024` number follows the re-exported #214 series (`0017–0023`). With `upstream`, staging, shared patch bytes, Dockerfile, AOT, pins and image hash exactly match the pinned prerequisite stack identified below; only selecting `registered-rdma` adds this transport patch. Before stopping any service, launch requires the expected selected `tf.patches` image label on all ranks, including with `PREPARE=0` or an `IMAGE` override. A missing, stale or mismatched label fails closed; labels do not attest to a maliciously forged image.

Every rank in the selected image uses the existing first settings exchange before NCCL. Its first settings row is `GLMT` + little-endian protocol version **3** + SHA-256 of the original caller settings and a fixed-size transport-profile suffix. `Comm.init` owns this exchange; production and the qualifier cannot omit it. There is no additional pre-NCCL agreement round. The startup profile and #214 cache-feature admission results are captured once, including invalid inputs. The existing profile suffix includes whether local cache admission succeeded, so even a single rank missing an optional kernel capability stops every rank. Agreement completes before captured errors are returned or NCCL is opened; no extra round is added.

New ranks reject incompatible first-row versions with `IncompatibleTransportProtocol` and a message naming the expected version. The inspected stock and previous PR images send a plain 32-byte digest. The same settings/verdict frame order lets responsive old and new ranks complete a negative verdict and exit before their startup sequences diverge. Followers always reply and consume the verdict; the head reads every reply before reporting refusal. Legacy ranks report their existing `RanksDisagree` error. CPU stream-socket tests cover all mixed old/new role combinations at TP2/3/4, version/settings mismatches and invalid profiles. This does not qualify arbitrary older protocols or production peer-loss deadlines.

Use the same selected image and settings on every rank, including when using that binary for an NCCL reference run. The unselected prerequisite image retains its existing startup and failure behavior. #214's KV sharding and spill v2 still require TP=2, and are refused before NCCL if requested with this TP4 profile.

## Collective behavior and prerequisites

Registered anonymous host memory replaces CUDA-allocated staging for this profile. GLM's collective placement, rank-major FP32 gather, neighbor-relay schedule, prefetch hooks and model arithmetic are retained. This is host-memory staging with CPU RDMA proxies. Eligible gathers up to **512 KiB per rank** use the relay; larger/ineligible gathers use NCCL. At 4096 FP32 values per row, 32 rows fit; 33–64-row windows retain NCCL. Required setup failure aborts startup; runtime transport failures are sticky and fatal.

Explicit `TF_GLM_L2PF` and `TF_GLM_SIDE` settings are preserved; both were 1 in the historical measured ring profile. No context or cache budget is forced.

A known-working NCCL library is required for fallback collectives. The historical ring used separately patched **NCCL 2.30.7**, SHA-256 `1cdf5f3e2d617d5c07a87903eb8849d5bd7758f6c900455d613a2e0eb289d63e`; see the [public sparkring patches](https://github.com/FujitsuPolycom/sparkring/tree/4b9b6a0a213456b96400c5e1a1cf59c20ff892c8/spark_transport/nccl). The stock wheel and dependency pins are unchanged. This flag neither installs that library nor qualifies the stock wheel, cabling or isolation; a version string alone is insufficient.

## Revision and historical evidence

The current native source is `a6645bea11fd9a837116c7aafefaa86af2628d9b`; its optional patch SHA-256 is `e1b5440713c403fcaec07cee56c00698805661ab11bc0365aa8bb923dbe7fe61`. [Source provenance](../zig/TRANSPORT_SOURCE.json) separates the revised source and CPU checks from historical qualification. The [2026-10-25 bounded hardware checks](tp4-registered-relay-20261025/README.md) passed a full NVCC build, four-rank NCCL and registered-ring probes, and short matched GLM checks at context131072/parallel8. All 12 cross-mode exact-token comparisons matched; both keys and eight-client gateway checks passed. The same source completed a [matched timing matrix](tp4-registered-relay-20261025/README.md): all 312 requests were valid, with three measured waves per cell and observed RDMA throughput gains of 1.96%–16.95%. The report states the sequential order and timing limits. Timestamped cleanup receipts record stopped/preserved containers and maintenance gateways; deployment acceptance and full-window quality remain outstanding.

The [historical compact benchmark and qualification](https://github.com/swbiggart/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/d62da68a86a30209480e51c533e7cb87cff5b952/docs/tp4-rdma-focused) remain unchanged at the previous public PR head. They describe `be94e919` and its binary, and do not qualify this revision. They are linked rather than duplicated in this compact update.

Historical byte/oracle/graph, exact-token and load results remain attributable only to their recorded source/binaries. The archived direct-on-ring negative test is historical; revised native parsing refuses direct mode earlier. Full-window quality, production peer loss, long-duration reliability and reboot remain unqualified. See the compact records for complete proof limits and immutable raw-evidence links.

## Integration and offline checks

This local revision combines merged MiaAI `zig-engine` `697d8e9cc84f5b7e45d7cdeb9517bec93f4c6cfb` with explicitly selected, unmerged PR #214 at `2d36adc7d12c547b86af8c15f01f260fe927eea8`. It addresses the [maintainer's re-export request](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/204#issuecomment-6395325650). The local prerequisite recipe commit is `cb4f4e661fe28a588b3dc06059d80ea5609aca3b`; TensorFold remains pinned at v1.0.7 (`e7c523eb`).

| Layer | Included source | Result |
| --- | --- | --- |
| Merged base | #211–#213, #186 and #187 at `697d8e9` | Memory, chat, display and prompt-cache changes retained; all 18 handwritten patch bytes unchanged |
| Pinned #214 | `2d36adc7`; original native feature commits through `98152c56` | Re-exported by `scripts/port-patches.sh` as `0017–0023` after the merged `0015`/`0016` display fixes |
| Selected transport | `0024-zig-glm-tp4-registered-relay.patch` | Applies after all 25 shared patches; added only for `registered-rdma` |

The original #214 native commit diffs were verified byte-for-byte against its pinned patches before cherry-picking. Its one display-cache conflict keeps both `.shard = sh` and per-registration zeroing. The transport startup conflict keeps #214's captured cache admission after the versioned first agreement and before NCCL. CPU checks cover identical invalid cache reads, mismatched settings, one-rank capability refusal, TP2-only feature restrictions, and the actual serving-entry signature.

These checks qualify only this pinned local stack. If #214 lands with different source or numbering, re-export against that landed stack before publication. The earlier open-PR conflict table is superseded; it is not a current conflict claim.

```sh
BASH=/path/to/bash python3 -m unittest discover -s tests -p test_tp4_network.py -v
python3 tests/verify_tp4_native_patch.py --upstream-repo <pinned-TensorFold-checkout> --native-repo <revised-native-checkout> --zig <Zig-0.17.0>
```

The first command uses local fixtures without SSH, Docker or GPUs, and compares unselected image inputs/hash with the pinned prerequisite recipe. The second reconstructs the full source, compares frozen file bytes/modes/hashes and runs CPU transport/probe checks. A local frozen native checkout is required for its exact-byte comparison. Neither command qualifies RDMA hardware or model output.

Registered-memory ownership is adapted from **BertholomusAI (Albert Lee, [bertholomus](https://github.com/bertholomus))**. Existing MiaAI, Hector Isaza, Ash Hart, TensorFold, Jürgen Schmied and b12x/RoCEnante attribution remains in [CREDITS](../CREDITS.md) and [NOTICE](../NOTICE). The separate [research PR](https://github.com/swbiggart/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/1) preserves the design alternatives.
