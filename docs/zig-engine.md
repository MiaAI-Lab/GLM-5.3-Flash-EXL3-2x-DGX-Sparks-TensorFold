# The Zig engine (`ENGINE=zig`, work in progress)

`ENGINE=zig ./start.sh` serves the same checkpoint with TensorFold v1.0.7's Zig engine (`tensorfold-native`) instead of
the Python engine (TensorFold v0.6.0 plus `patches/`). The scripts, ports, pinned revisions, NFS worker weights, banner,
`DRY_RUN`, `./stop.sh` and the health check are the same. `ENGINE=zig` is the default. `ENGINE=python` selects the Python engine.

Status: the image builds and the commands print (`DRY_RUN=1`); the server has not been qualified through these scripts
yet. The Zig engine's greedy replies equal the Python engine's on two prompts at TP=2 (drafts off and on).

## What is in the image

`zig/Dockerfile`, built by `scripts/prepare.sh` (no GPU needed):

| Part | Source |
| --- | --- |
| `tensorfold-native` and `tensorfold` | TensorFold v1.0.7 (`ZIG_TF_REF`, commit `e7c523eb`) plus `zig/patches/0001-0014`, built with Zig 0.17.0 and nvcc (sm_121 fatbins are embedded in the binary) |
| captured Triton kernel set | `zig/aot/glm-aot-sm121.tar.xz` (aot.json and 311 cubins, captured from the Python engine on a GB10; pinned by its sha256), at `TENSORFOLD_CUDA_KERNELS` |
| runtime libraries | `nvidia/cuda:13.0.2-runtime-ubuntu24.04` (cuBLAS, cuBLASLt), NCCL 2.30.7 (the pip wheel; the Python image runs the same), libibverbs and the mlx5 provider |

`zig/patches/0001-0005` establishes the Zig port against TensorFold v1.0.7: shared engine changes (lane host, native
CUDA registry, server), Flash Next CUDA support, GLM kernels, the GLM engine, and tests. Patches 0006-0013 retain the
recipe's subsequent fixes; 0014 adds the opt-in registered relay. `zig/TRANSPORT_SOURCE.json` records its exact native
source, patch/file checksums and reconstruction limits. `tools/make-zig-patches.sh` refreshes the original port set;
`tests/verify_tp4_native_patch.py` checks all fourteen patches against the frozen focused source.
The image tag is `zig-v1.0.7-<hash>` of the patches, the kernel set, the Dockerfile and the pins (`IMAGE_TAG`,
`IMAGE_DIGEST` pin a published one; none yet, so `prepare.sh` builds it, about 10 minutes the first time).

## Settings

Every `scripts/config.sh` setting, and what the Zig engine does with it. TODO: the engine does not read it yet; the
recipe keeps the setting and `./start.sh` refuses a value the engine cannot honour (rather than ignoring it) where that
would change what is served.

| Setting | Zig engine |
| --- | --- |
| `TP`, `WORKER`, `WORKER2/3`, `FABRIC_PEER*`, `MASTER_PORT`, `MASTER_ADDR`, `SOCKET_IFNAME` | `--tp`, `--rank`, `--master`, `--master-port`; NCCL variables as the Python engine's. TP=2, TP=3 on a triangle with `SPLIT=0 COMM=nccl` and TP=4 on a ring of four with `SPLIT=0 COMM=nccl` (both experimental, see below; TP=3 also takes `COMM=roce`). TP=4 on a switch is code-tested only, not hardware-verified, and requires the explicit `TP4_NETWORK=registered-rdma` profile below; `COMM=roce` at TP=4 also requires that profile |
| `MODEL_ID`, `MODEL_REVISION`, `ABLIT`, `DFLASH2_ID`, `DFLASH2_REVISION`, `HF_CACHE`, `WORKER_WEIGHTS*`, `NFS_*` | unchanged: the snapshots are mounted at `/root/.cache/huggingface` and passed as `MODEL` and `--drafter` (`ABLIT=1` has the same architecture; not tried yet) |
| `SERVED_NAME`, `HOST`, `PORT` | `--name`, `--host`, `--port` |
| `CONTEXT`, `PARALLEL` | `--context`, `--parallel`. Defaults 131072 and 8 under `ENGINE=zig`: the captured kernels are keyed on that arena; other values need more captures (TODO; a warning says so) |
| `MAX_TOKENS`, `THINKING` | `--max-tokens`, `--thinking` / `--no-thinking` |
| `DRAFTER`, `TF_GLM_MTP`, `TF_GLM_DRAFT_QUANT` | `dflash2` -> `--drafter DIR`. `mtp` -> no `--drafter`; the container gets `DRAFTER=mtp` (PARALLEL=1), which the Zig engine reads from its environment. The MTP head is in `zig/patches`; `ZIG_HAS` defaults to `mtp displaykv` |
| `KV`, `DENSE` | fixed at `fp8` and `q4`; other values refused (TODO) |
| `TP4_NETWORK` | `upstream` (default) retains existing transport-selection semantics. `registered-rdma` requires Zig, TP=4, SPLIT=0 and COMM=roce on a discovered mesh or ring; maps to `TF_GLM_TP4_REGISTERED_RDMA=1`. See [the focused profile](tp4-registered-relay.md) |
| `COMM`, `TF_ROCE_MAX_KB` | `TF_GLM_ROCE` (1 for `roce`, 0 for `nccl`), `TF_ROCE_MAX_KB` (fixed at 512 for the opt-in profile) |
| `TF_GLM_L2PF` | `TF_GLM_L2PF` (0 / off: off; anything else: 1) |
| `MEMORY_RESERVE_GIB`, `KV_POOL_GIB` | `TENSORFOLD_MEMORY_RESERVE_GIB` and `TF_GLM_CACHE_GIB`, passed as for Python; the memory-sized pool is in branch `r107-cx-displaykv` (see `DISPLAY_KV_*` below); the 19.6 GiB default is the Python engine's measurement |
| `TF_GLM_EFFORT_TAIL`, `TF_GLM_CLEAR_THINKING`, `STREAM_SMOOTH`, `STREAM_SMOOTH_MS`, `TF_GLM_LOOP_GUARD`, `MAX_QUEUED` (`TF_GLM_MAX_QUEUED`), `TF_GLM_KEEP_REASONING`, `TF_GLM_KEEP_REASONING_MB`, `TF_GLM_ASSISTANT_ENDS`, `TF_GLM_PROMPT_STOP` | implemented (`zig/src/server/recipe.zig`, branch `r107-int`): the ranks read the recipe's own names, passed unchanged |
| `KV_POOL_GIB`, `TF_GLM_CACHE_ENTRIES`, `TF_GLM_CACHE_SHARE_PCT`, `TF_GLM_KEPT_BYTES_GIB`, `TF_GLM_KEEP_PER_CHAT`, `SHARED_PREFIX` | the engine keeps prompt states. `./start.sh` passes `--prompt-cache-gib 0` (the qualified path); raise it to keep states (`./start.sh --prompt-cache-gib 8`: the last flag wins). The budget is the asked size, at most what is free at start minus the reserve (`MEMORY_RESERVE_GIB`) and the keep headroom, `TF_GLM_KEEP_HEADROOM_GIB` (2; patch 0012): the kept states stop that far above the reserve, so the prompts' buffers can grow into it. A keep that would take the machine under the reserve is skipped at that mark on every rank (a warning; earlier and later states are unaffected), and a save the engine still refuses is skipped the same way on every rank, the prompt going on from the rows it ran, instead of ending the server |
| `SPILL_*` | implemented: the same `--spill-gib` / `--snapshot-dir` / `--spill-*` flags as the Python path. Default `SPILL_GIB=0` |
| `VISION`, `VISION_URLS` | implemented (`--vision --vision-backend zig`, `--vision-urls`). Default `VISION=0` under `ENGINE=zig` so the text server keeps the tower's memory; `VISION=1` loads the native tower on rank 0 (the image carries cuDNN; there is no Python helper) |
| `DISPLAY_KV_MIB`, `DISPLAY_KV_BACKEND` (`TF_GLM_DISPLAY_KV_MIB`, `_BACKEND`, `_CARD`) | implemented. Default off. `ZIG_HAS` includes `displaykv` |
| `COPY`, `COPY_MAX`, `COPY_CODE`, `DRAFT_POLICY` | TODO: the engine's copy drafts and DFlash2 policy are built in, not settable |
| `TF_GLM_SIDE` | shared expert on a second stream during decode; passed unchanged. The existing recipe defaults to 1 at TP=4 and 0 at TP=2/3. The registered-rdma profile respects explicit values; its measured configuration used 1 |
| `SPLIT` (`TF_GLM_HC_SPLIT`, `TF_GLM_PREFILL_OVERLAP`), `TF_GLM_PROMPT_OWN_FRONT`, `TF_GLM_PROMPT_MICROBATCH`, `TF_GLM_PREFILL_ROWS`, `TF_GLM_PROMPT_PARTIALS`, `TF_GLM_MSA` | split prefill, overlap, own-front and microbatch run at TP=2 (`r107-tpf`, on `r107-int`). `TF_GLM_PREFILL_ROWS` sets the chunk size. Past two ranks the chunk runs unsplit (no TP=3/4 ring exchange in this project) |
| `KDA_CHUNKED` | the chunked kernel is built in |
| `MULTI_PREFILL`, `FILL_BUDGET_MS`, `FILL_DRAFTS`, `TF_GLM_MULTI_WINDOW`, `TF_GLM_MULTI_LONE` | TODO: prompts fill one at a time; the verify window is fixed (64 rows shared, 16 a stream) |
| `STREAM_SMOOTH`, `STREAM_SMOOTH_MS` | TODO |
| `MAX_QUEUED` | TODO (no 429 admission yet) |
| `NUCLEUS_UNION` | not needed (the Zig sampler has its own) |
| `TF_ROCE_WAIT_S`, `TF_ROCE_IB_TIMEOUT`, `TF_ROCE_RING` | native RoCE wait timeout, QP ACK timeout, and ring relay selection. The registered-rdma profile supplies 1 per rank for a discovered ring and 0 for a switched mesh; an explicit value must agree with the fabric |
| `TF_GLM_EXL3_LOADS`, `TF_GLM_EXL3_DEC_ORDER` | TODO (its kernels are its own) |
| `TF_GLM_CLEAR_THINKING`, `TF_GLM_EFFORT_TAIL` | TODO (earlier turns keep their reasoning, as with the Python defaults) |
| `LOG_*`, `FOREGROUND`, `WAIT_TIMEOUT`, `STOP_TIMEOUT`, `CONTAINER_NAME`, `KERNEL_CACHE`, `STATE_DIR` | unchanged |
| `HF_HUB_OFFLINE`, `TENSORFOLD_API_KEY` | read as before |

A Zig rank gets what a Python rank gets: every `TENSORFOLD_*`, `TF_GLM_*` and `TF_ROCE_*` variable that
`scripts/config.sh` exports or you set (variables the engine does not read are ignored; a malformed value of one it does
read fails the start), plus `TF_GLM_ROCE` from `COMM`, `TF_GLM_L2PF` as 0 or 1, and `TF_CUDA_DEVICE`, `TF_GRAPH_LOG`,
`TF_ROUND_LOG`, `TF_TP_*` when you set them. `ZIG_HAS` (space-separated `mtp`, `displaykv`) lifts the refusal of a feature
once its port is in `zig/patches`.

## Prefill capture coverage

The SM121 archive includes dense `_merge` specializations for NCH 1–37 and
all three row classes (constant 1, aligned runtime, unaligned runtime), plus
`_dense_chunks` FP8 HBT32/R=1 and `_hc_partial_mma` R=1. These add the HC R=1 capture
reported in #176 and dense merge coverage relevant to #177. All 210 original cubins are
preserved byte-for-byte; 101 captures are appended. Kernel arithmetic and
native dispatch are unchanged.

The archive passed 549 strict direct-cubin GPU comparisons against the
original JIT kernels on GB10. The same archive was also used in a TP2 build
with capture-coverage tests and a behavior-preserving extraction of the dense
attention block: boundary, serial/drafted parity, eight
concurrent requests, tools/vision and 20 benchmark requests passed at 128K
context with DFlash2. This does not qualify arbitrary MTP configurations,
1M context, or every other missing-capture path. No speedup is claimed.

## Known limits (not the scripts')

- Wider decode `_select_split` cubins are in the AOT set (the long-context branch). Prompts past the captured buckets still fail with `MissingTritonVariant`.
- TP=2 is what this project runs by default. TP=3 runs on three Sparks cabled as a triangle and TP=4 on four cabled as
  a ring (below); split prefill past two ranks stays unsplit.

- Qual on this head (`zig-notes/r107-int.md`): 58 of 61 cases identical to Python. `tools_required` differs after the call opener. Decode at 8 streams is still under the Python engine.

### TP=3 (experimental)

Three Sparks, each cabled to the other two (a triangle). The tested start:

```bash
TP=3 ENGINE=zig SPLIT=0 COMM=nccl DRAFTER=dflash2 PARALLEL=8 CONTEXT=131072 MEMORY_RESERVE_GIB=24 ./start.sh
```

`COMM=roce` also works at TP=3: b12x's RoCEnante transport carries the small all-gathers over the direct routes
(patch 0010); larger collectives stay on NCCL by design. On the tested triangle, Grill's default C1 workload measured
+12.2% against `COMM=nccl`, with the same replies, on an image that also contained patch 0011. start.sh
refuses other SPLIT/COMM values for the Zig engine at TP=3. `MEMORY_RESERVE_GIB=24` is not the default: rank 0
holds the most (22 heads and the server), and with 19.6 its free memory dropped under 8 GiB during a long sweep.

- Network: start.sh gives a triangle the ring's NCCL graph files (rank 0 -> 1 -> 2 -> 0, each rank listening toward the
  previous one) with NCCL's NIC fusion off (`scripts/nodes.sh`). The Zig engine uses only main-communicator
  collectives at TP=3; the Python engine also uses TF_NCCL_RING's two-rank links. Without the graph files NCCL
  connected through ports cabled to the other peer and timed out; raw point-to-point on the main communicator did not
  connect on the tested triangle, and neither engine uses it at TP=3. `RING_GRAPH=0`, another `NCCL_ALGO` or changed
  routing settings leave the tested setup. Paths with spaces or quotes in STATE_DIR or a worker's home are not
  supported by the graph-file writing.
- Kernel set: `zig/aot/glm-aot-sm121.tar.xz` keeps the TP=2 set's 311 entries first and unchanged, then 186 TP=3
  entries captured from the Python engine at TP=3 (ranks of 22/21/21 heads), then 12 `_conv_shift` /
  `_conv_shift_seg` variants compiled offline with the image's Triton for a layer stride that is not a multiple of 16
  (the Zig engine sizes that buffer by the prompt's rows; at TP=3 the packed KDA stride, 8726 / 8341, is not a
  multiple of 16). Its `aot.json` records each step (`tp3_completion`); its top-level `tp: 2` is the base set's
  metadata, not a selection key. Patch 0009 agrees the KV pool over every rank.
- Checked, for this configuration (DFlash2 drafts, text only): the same requests on the Zig and Python engines at TP=3
  gave identical replies (17 parity cases, 4 chat-resume cases, 428 sweep cases, by text, counts and token digest),
  with no missing kernel variant on any rank. At `CONTEXT=1048576` (`TF_TP_TIMEOUT_S=1500`), 262k, 512k and
  1,000,031-token prompts were served (HTTP 200; the ~1M prompt took 962 s with drafts off and 961 s on, with the GPU
  clock capped at 2200 MHz). TP=2 with this kernel set gave the same replies as before. MTP drafts (`DRAFTER=mtp
  PARALLEL=1`) at TP=3 gave the same replies as the Python engine (one more kernel, `_residual_add` at WORLD=3);
  their acceptance depends on the sampler fix in patch 0011 (separate PR). Not qualified at TP=3: vision, other
  arenas, the prompt cache, and TP=3 against TP=2 numerically.

### TP=4 (experimental)

Four Sparks cabled as a ring, each to its two neighbours (README "4 Sparks", without a switch), the workers in the
ring's order in `scripts/local.sh`. The tested start:

```bash
ENGINE=zig SPLIT=0 COMM=nccl DRAFTER=dflash2 PARALLEL=8 CONTEXT=131072 MEMORY_RESERVE_GIB=24 ./start-tp4-switchless.sh
```

With `TP4_NETWORK=upstream` (the default), start.sh still requires `COMM=nccl`
for the Zig engine at TP=4. A switch and `SPLIT=1` remain refused. This preserves
transport selection, not every startup or failure behavior: the newly built image
adds strict profile agreement before NCCL and transport failure checks, including
for default selection. Mixed settings are rejected. Use the same rebuilt image
on every rank; every rank must agree on the current startup profile.

The opt-in registered RDMA profile supports both existing TP4 wrappers:

```bash
# Physical ring, workers in cable order:
TP4_NETWORK=registered-rdma ENGINE=zig SPLIT=0 ./start-tp4-switchless.sh
# Switched mesh: code-tested only, NOT hardware-verified:
TP4_NETWORK=registered-rdma ENGINE=zig SPLIT=0 ./start-tp4.sh
# Let existing fabric discovery select mesh or ring:
TP4_NETWORK=registered-rdma ENGINE=zig TP=4 SPLIT=0 ./start.sh
```

The same settings may be set in `scripts/local.sh` or `.env` for `./start.sh`
and `scripts/prepare.sh`. `FABRIC_EXPECT=mesh` or `ring` checks an expected
fabric; unset permits discovery. There is no separate topology selector. The
wrappers retain their existing `COMM` environment override behavior: use
`./start.sh` when selecting all settings through local.sh/.env. An unset `COMM`
defaults to `roce` for this opt-in on either fabric; explicit `COMM=nccl` conflicts.
The default upstream selection and its Zig TP4 ring/NCCL restriction are unchanged.

Static profile conflicts fail in start/restart and prepare before host discovery,
preparation, SSH or stopping an existing service. Discovery rejects the wrong
wrapper, missing peers and workers outside physical ring order. `TF_ROCE_RING`
is optional; only 0 or 1 is accepted. An explicit mismatch with `FABRIC_EXPECT`
is rejected before host calls; with unset `FABRIC_EXPECT`, a mismatch with the
discovered fabric fails before stopping the previous service. The launcher
supplies `TF_ROCE_RING=1` per rank on a ring and `0` on a mesh, excluding any
global override from the shared rank environment. Per-rank HCA names, GIDs,
rail order, NCCL graph assignments and addresses still come from existing
fabric discovery. Unreachable-worker placeholders cannot qualify this profile.
Standalone `./stop.sh` skips transport-profile validation so stale launch settings
do not prevent shutdown.

This is a first-install API: `TP4_NETWORK` accepts `upstream` or
`registered-rdma`. The native flag is internal; select the profile with
`TP4_NETWORK`. Contradictory `TF_GLM_TP4_REGISTERED_RDMA` values are rejected.
Research-only switches are outside the supported configuration.

On a ring this profile uses the existing neighbour relay protocol; on a switched
mesh it uses direct all-peer RoCE. Both use registered host memory, fixed 512 KiB
slots, required setup, and NCCL for larger/ineligible byte counts. Automatic
cutoff growth is skipped only for this opt-in. `TF_GLM_L2PF` and `TF_GLM_SIDE`
keep explicit values; the measured ring configuration used both at 1. No context
or cache setting is forced. The switched path has offline code coverage only;
CPU fixture tests establish recipe/discovery/forwarding behavior, not RDMA,
NCCL, GPU correctness or performance on switched hardware. The launcher emits
this limit when a switched mesh is discovered. Run the fixture suite with Bash 4+
(the configuration already uses associative arrays):

```bash
BASH=/path/to/bash python3 -m unittest discover -s tests -p test_tp4_network.py -v
```

Every opt-in path requires a known-working NCCL library for fallback collectives.
The accepted ring fleet used a separately patched NCCL 2.30.7 library; the stock
wheel in this Dockerfile is not qualified by that result, and a version string
alone is not proof. This profile does not install or patch NCCL. Use the same
image on every rank with consistent startup settings.

The following historical TP4 checks describe the upstream NCCL configuration:

- Kernel set: the TP=3 set's 510 entries first and unchanged, then 72 TP=4 entries captured from the Python engine at
  TP=4 on a ring (16 heads on every rank; the four ranks compiled byte-identical sets). No variant had to be compiled
  offline. Its `aot.json` records the step in `tp3_completion` (the merge tool's list of appended steps).
- Checked, for this configuration (DFlash2 drafts, text only): the same requests on the Zig and Python engines at TP=4
  gave identical replies (17 parity cases, 4 chat-resume cases, 428 sweep cases, and the step-3 main set's 56
  requests including eight at once, drafts off and on), with no missing kernel variant on any rank. At
  `CONTEXT=1048576` (`TF_TP_TIMEOUT_S=1500`), 262k, 512k and 1,000,031-token prompts were served (HTTP 200; the ~1M
  prompt took 889 s, drafts off and on). TP=2 with this kernel set gave the same replies as before. The measured
  428-case sweep took much longer on Zig: 974 s against 85 s on Python, with the same replies (observed workload
  times, not a controlled attribution: the Python run had a capture hook and its default memory reserve, Zig 24 GiB).
  Not qualified at TP=4: vision, MTP drafts, other arenas, the prompt cache, and TP=4 against TP=2 or TP=3 numerically.

## Build and test the Zig side

Offline configuration/forwarding checks (Bash 4+):

```sh
python3 -m unittest discover -s tests -p test_tp4_network.py -v
python3 tests/verify_tp4_native_patch.py --upstream-repo <TensorFold-checkout> --native-repo <focused-checkout> --zig <Zig-0.17.0>
```

The reconstruction check compares all source-file bytes, accounts explicitly for
six pre-existing executable-mode differences in the frozen baseline, and runs
host transport tests when `--zig` is supplied. These checks do not build CUDA
kernels or qualify GPU execution.


```sh
tools/make-zig-patches.sh <Zig port checkout> [branch] [base]   # refresh zig/patches and zig/aot
ENGINE=zig scripts/prepare.sh                                    # build (or pull) the image on both Sparks
ENGINE=zig DRY_RUN=1 ./start.sh                                  # print every rank's docker command
ENGINE=zig ./start.sh                                            # start it
```

Credits: TensorFold's Zig engine is by Ash Hart ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)) and its
contributors; the native CUDA server, registry and rank mirror are jschmied's
([PR #443](https://github.com/ashhart/TensorFold/pull/443)); the one-shot RoCE all-gather is
[b12x / RoCEnante](https://github.com/local-inference-lab) by local-inference-lab. See [`CREDITS.md`](../CREDITS.md).
