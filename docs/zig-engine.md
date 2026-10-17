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
| `tensorfold-native` and `tensorfold` | TensorFold v1.0.7 (`ZIG_TF_REF`, commit `e7c523eb`) plus `zig/patches/0001-0005`, built with Zig 0.17.0 and nvcc (sm_121 fatbins are embedded in the binary) |
| captured Triton kernel set | `zig/aot/glm-aot-sm121.tar.xz` (aot.json and 311 cubins, captured from the Python engine on a GB10; pinned by its sha256), at `TENSORFOLD_CUDA_KERNELS` |
| runtime libraries | `nvidia/cuda:13.0.2-runtime-ubuntu24.04` (cuBLAS, cuBLASLt), NCCL 2.30.7 (the pip wheel; the Python image runs the same), libibverbs and the mlx5 provider |

`zig/patches/` is the Zig port against TensorFold v1.0.7, in five patches: the shared engine changes (lane host, native
CUDA registry, server), the Flash Next Zig CUDA engine the GLM engine builds on, the GLM kernels, the GLM engine, and its
tests. `tools/make-zig-patches.sh <Zig port checkout>` writes them and checks that they reproduce the branch exactly.
The image tag is `zig-v1.0.7-<hash>` of the patches, the kernel set, the Dockerfile and the pins (`IMAGE_TAG`,
`IMAGE_DIGEST` pin a published one; none yet, so `prepare.sh` builds it, about 10 minutes the first time).

## Settings

Every `scripts/config.sh` setting, and what the Zig engine does with it. TODO: the engine does not read it yet; the
recipe keeps the setting and `./start.sh` refuses a value the engine cannot honour (rather than ignoring it) where that
would change what is served.

| Setting | Zig engine |
| --- | --- |
| `TP`, `WORKER`, `WORKER2/3`, `FABRIC_PEER*`, `MASTER_PORT`, `MASTER_ADDR`, `SOCKET_IFNAME` | `--tp`, `--rank`, `--master`, `--master-port`; NCCL variables as the Python engine's. TP=2 only: `--tp 3` is refused by the parser and TP=4 has no split, ring or relay code yet (TODO: start refuses TP=3/4; `DRY_RUN=1` still prints them) |
| `MODEL_ID`, `MODEL_REVISION`, `ABLIT`, `DFLASH2_ID`, `DFLASH2_REVISION`, `HF_CACHE`, `WORKER_WEIGHTS*`, `NFS_*` | unchanged: the snapshots are mounted at `/root/.cache/huggingface` and passed as `MODEL` and `--drafter` (`ABLIT=1` has the same architecture; not tried yet) |
| `SERVED_NAME`, `HOST`, `PORT` | `--name`, `--host`, `--port` |
| `CONTEXT`, `PARALLEL` | `--context`, `--parallel`. Defaults 131072 and 8 under `ENGINE=zig`: the captured kernels are keyed on that arena; other values need more captures (TODO; a warning says so) |
| `MAX_TOKENS`, `THINKING` | `--max-tokens`, `--thinking` / `--no-thinking` |
| `DRAFTER`, `TF_GLM_MTP`, `TF_GLM_DRAFT_QUANT` | `dflash2` -> `--drafter DIR`. `mtp` -> no `--drafter`; the container gets `DRAFTER=mtp` (PARALLEL=1), which the Zig engine reads from its environment. The MTP head is in `zig/patches`; `ZIG_HAS` defaults to `mtp displaykv` |
| `KV`, `DENSE` | fixed at `fp8` and `q4`; other values refused (TODO) |
| `COMM`, `TF_ROCE_MAX_KB` | `TF_GLM_ROCE` (1 for `roce`, 0 for `nccl`), `TF_ROCE_MAX_KB` |
| `TF_GLM_L2PF` | `TF_GLM_L2PF` (0 / off: off; anything else: 1) |
| `MEMORY_RESERVE_GIB`, `KV_POOL_GIB` | `TENSORFOLD_MEMORY_RESERVE_GIB` and `TF_GLM_CACHE_GIB`, passed as for Python; the memory-sized pool is in branch `r107-cx-displaykv` (see `DISPLAY_KV_*` below); the 19.6 GiB default is the Python engine's measurement |
| `TF_GLM_EFFORT_TAIL`, `TF_GLM_CLEAR_THINKING`, `STREAM_SMOOTH`, `STREAM_SMOOTH_MS`, `TF_GLM_LOOP_GUARD`, `MAX_QUEUED` (`TF_GLM_MAX_QUEUED`), `TF_GLM_KEEP_REASONING`, `TF_GLM_KEEP_REASONING_MB`, `TF_GLM_ASSISTANT_ENDS`, `TF_GLM_PROMPT_STOP` | implemented (`zig/src/server/recipe.zig`, branch `r107-int`): the ranks read the recipe's own names, passed unchanged |
| `KV_POOL_GIB`, `TF_GLM_CACHE_ENTRIES`, `TF_GLM_CACHE_SHARE_PCT`, `TF_GLM_KEPT_BYTES_GIB`, `TF_GLM_KEEP_PER_CHAT`, `SHARED_PREFIX` | the engine keeps prompt states. `./start.sh` passes `--prompt-cache-gib $PROMPT_CACHE_GIB`, default 0 (the qualified path); raise it to keep states. The budget is at most MemAvailable above `MEMORY_RESERVE_GIB` once the pool is up, so at the defaults it is 0: lower `KV_POOL_GIB` to leave room (`scripts/config.sh`) |
| `SPILL_*` | implemented: the same `--spill-gib` / `--snapshot-dir` / `--spill-*` flags as the Python path. Default `SPILL_GIB=0` |
| `VISION`, `VISION_URLS` | implemented (`--vision --vision-backend zig`, `--vision-urls`). Default `VISION=0` under `ENGINE=zig` so the text server keeps the tower's memory; `VISION=1` loads the native tower on rank 0 (the image carries cuDNN; there is no Python helper) |
| `DISPLAY_KV_MIB`, `DISPLAY_KV_BACKEND` (`TF_GLM_DISPLAY_KV_MIB`, `_BACKEND`, `_CARD`) | implemented. Default off. `ZIG_HAS` includes `displaykv` |
| `COPY`, `COPY_MAX`, `COPY_CODE`, `DRAFT_POLICY` | TODO: the engine's copy drafts and DFlash2 policy are built in, not settable |
| `TF_GLM_SIDE` | ported (shared expert on a second stream, decode windows; off by default as in Python), branch `r107-tpf`; passed unchanged |
| `SPLIT` (`TF_GLM_HC_SPLIT`, `TF_GLM_PREFILL_OVERLAP`), `TF_GLM_PROMPT_OWN_FRONT`, `TF_GLM_PROMPT_MICROBATCH`, `TF_GLM_PREFILL_ROWS`, `TF_GLM_PROMPT_PARTIALS`, `TF_GLM_MSA` | split prefill, overlap, own-front and microbatch run at TP=2 (`r107-tpf`, on `r107-int`). `TF_GLM_PREFILL_ROWS` sets the chunk size. Past two ranks the chunk runs unsplit (no TP=3/4 ring exchange in this project) |
| `KDA_CHUNKED` | the chunked kernel is built in |
| `MULTI_PREFILL`, `FILL_BUDGET_MS`, `FILL_DRAFTS`, `TF_GLM_MULTI_WINDOW`, `TF_GLM_MULTI_LONE` | TODO: prompts fill one at a time; the verify window is fixed (64 rows shared, 16 a stream) |
| `STREAM_SMOOTH`, `STREAM_SMOOTH_MS` | TODO |
| `MAX_QUEUED` | TODO (no 429 admission yet) |
| `NUCLEUS_UNION` | not needed (the Zig sampler has its own) |
| `TF_ROCE_WAIT_S`, `TF_ROCE_IB_TIMEOUT`, `TF_ROCE_RING` | TODO |
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
- TP=2 is what this project runs. The TP=3/4 control plane is in the engine; spark3 and spark4 are not in this project, and split prefill past two ranks stays unsplit.
- Qual on this head (`zig-notes/r107-int.md`): 58 of 61 cases identical to Python. `tools_required` differs after the call opener. Decode at 8 streams is still under the Python engine.

## Build and test the Zig side

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
