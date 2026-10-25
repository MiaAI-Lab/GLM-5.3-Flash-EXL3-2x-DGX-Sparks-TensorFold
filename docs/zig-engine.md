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
| `tensorfold-native` and `tensorfold` | TensorFold v1.0.7 (`ZIG_TF_REF`, commit `e7c523eb`) plus `zig/patches/` (`0001`-`0023`, three of them `0014`), built with Zig 0.17.0 and nvcc (sm_121 fatbins are embedded in the binary) |
| captured Triton kernel set | `zig/aot/glm-aot-sm121.tar.xz` (aot.json and 311 cubins, captured from the Python engine on a GB10; pinned by its sha256), at `TENSORFOLD_CUDA_KERNELS` |
| runtime libraries | `nvidia/cuda:13.0.2-runtime-ubuntu24.04` (cuBLAS, cuBLASLt), NCCL 2.30.7 (the pip wheel; the Python image runs the same), libibverbs and the mlx5 provider |

`zig/patches/` is the Zig port against TensorFold v1.0.7: five patches (the shared engine changes (lane host, native
CUDA registry, server), the Flash Next Zig CUDA engine the GLM engine builds on, the GLM kernels, the GLM engine, and its
tests), which `tools/make-zig-patches.sh <Zig port checkout>` writes and checks against the branch, then the hand-written
patches `0006`-`0013`, then Robert's three base fixes as hand-written `0014`s (the parked pages, a long prompt's preparation,
the shared-cut chat id), ezoushen's display KV fixes `0015`-`0016` (#186), and Robert's port as `0017`-`0023` (continued
chats, spill v2 and KV sharding: one patch per feature), exported by `scripts/port-patches.sh` from the port's engine branch. [`zig/patches/README.md`](../zig/patches/README.md) lists them and the ordering
rules; `tools/make-zig-patches.sh` refuses while patches after `0005` are there (`--force` deletes them).
The image tag is `zig-v1.0.7-<hash>` of the patches, the kernel set, the Dockerfile and the pins (`IMAGE_TAG`,
`IMAGE_DIGEST` pin a published one; none yet, so `prepare.sh` builds it, about 10 minutes the first time).

## Settings

Every `scripts/config.sh` setting, and what the Zig engine does with it. TODO: the engine does not read it yet; the
recipe keeps the setting and `./start.sh` refuses a value the engine cannot honour (rather than ignoring it) where that
would change what is served.

| Setting | Zig engine |
| --- | --- |
| `TP`, `WORKER`, `WORKER2/3`, `FABRIC_PEER*`, `MASTER_PORT`, `MASTER_ADDR`, `SOCKET_IFNAME` | `--tp`, `--rank`, `--master`, `--master-port`; NCCL variables as the Python engine's. TP=2, TP=3 on a triangle with `SPLIT=0 COMM=nccl` and TP=4 on a ring of four with `SPLIT=0 COMM=nccl` (both experimental, see below; TP=3 also takes `COMM=roce`). TP=4 on a switch is untested and refused; the RoCE ring relay is not ported (start refuses `COMM=roce` at TP=4) |
| `MODEL_ID`, `MODEL_REVISION`, `ABLIT`, `DFLASH2_ID`, `DFLASH2_REVISION`, `HF_CACHE`, `WORKER_WEIGHTS*`, `NFS_*` | unchanged: the snapshots are mounted at `/root/.cache/huggingface` and passed as `MODEL` and `--drafter` (`ABLIT=1` has the same architecture; not tried yet) |
| `SERVED_NAME`, `HOST`, `PORT` | `--name`, `--host`, `--port` |
| `CONTEXT`, `PARALLEL` | `--context`, `--parallel`. Defaults 131072 and 8 under `ENGINE=zig`: the captured kernels are keyed on that arena; other values need more captures (TODO; a warning says so) |
| `MAX_TOKENS`, `THINKING` | `--max-tokens`, `--thinking` / `--no-thinking` |
| `DRAFTER`, `TF_GLM_MTP`, `TF_GLM_DRAFT_QUANT` | `dflash2` -> `--drafter DIR`. `mtp` -> no `--drafter`; the container gets `DRAFTER=mtp` (PARALLEL=1), which the Zig engine reads from its environment. The MTP head is in `zig/patches`; `ZIG_HAS` defaults to `mtp displaykv` |
| `KV`, `DENSE` | fixed at `fp8` and `q4`; other values refused (TODO) |
| `COMM`, `TF_ROCE_MAX_KB` | `TF_GLM_ROCE` (1 for `roce`, 0 for `nccl`), `TF_ROCE_MAX_KB` |
| `TF_GLM_L2PF` | `TF_GLM_L2PF` (0 / off: off; anything else: 1) |
| `MEMORY_RESERVE_GIB`, `KV_POOL_GIB` | `TENSORFOLD_MEMORY_RESERVE_GIB` and `TF_GLM_CACHE_GIB`, passed as for Python; the memory-sized pool is in branch `r107-cx-displaykv` (see `DISPLAY_KV_*` below); the 19.6 GiB default is the Python engine's measurement |
| `TENSORFOLD_PREPARE_FLOOR_GIB` | `zig/patches/0014-zig-prepare-memory.patch`: the MemAvailable rank 0 keeps while several prompts are rendered and tokenized at once. Each prepare reserves 48 bytes a text byte and waits while the ones in flight and its own would take MemAvailable under the floor. Waiters go in arrival order (a large prepare is not starved by small ones), a prepare that finds none in flight and none waiting goes at once, as before; a waiter queued behind prepares in flight waits for memory at most `TENSORFOLD_PREPARE_WAIT_MS` (default 5000; 0: at once) and then goes as soon as none is in flight, so it only queues concurrent prepares and never changes a reply; past `TENSORFOLD_PREPARE_MAX_WAITING` waiters (default 64; 0: no bound) one more is refused at once with 429 and `Retry-After: 5`, as a request past the lanes is. Default 3 GiB; 0 turns the gate off; a malformed value of any of the three refuses the start. Passed to every rank like every `TENSORFOLD_*` variable (only rank 0 prepares) |
| `TF_GLM_EFFORT_TAIL`, `TF_GLM_CLEAR_THINKING`, `STREAM_SMOOTH`, `STREAM_SMOOTH_MS`, `TF_GLM_LOOP_GUARD`, `MAX_QUEUED` (`TF_GLM_MAX_QUEUED`), `TF_GLM_KEEP_REASONING`, `TF_GLM_KEEP_REASONING_MB`, `TF_GLM_ASSISTANT_ENDS`, `TF_GLM_PROMPT_STOP` | implemented (`zig/src/server/recipe.zig`, branch `r107-int`): the ranks read the recipe's own names, passed unchanged |
| `PROMPT_CACHE_GIB`, `KV_POOL_GIB`, `TF_GLM_CACHE_ENTRIES`, `TF_GLM_CACHE_SHARE_PCT`, `TF_GLM_KEPT_BYTES_GIB`, `TF_GLM_KEEP_PER_CHAT`, `SHARED_PREFIX` | the engine keeps prompt states. `./start.sh` passes `--prompt-cache-gib $PROMPT_CACHE_GIB`, default 0 (the qualified path); raise it to keep states (`PROMPT_CACHE_GIB=8`, or `./start.sh --prompt-cache-gib 8`: the last flag wins; ezoushen's setting, Mia PR #187). The budget is the asked size, at most what is free at start minus the reserve (`MEMORY_RESERVE_GIB`) and the keep headroom, `TF_GLM_KEEP_HEADROOM_GIB` (2; patch 0012): the kept states stop that far above the reserve, so the prompts' buffers can grow into it. The pool is sized first, so at the defaults the budget comes out 0 (`prompt cache: off (budget 0)`, measured in #187): lower `KV_POOL_GIB` to leave it room. **Measured on the pair** (the port's round-1 image): at the default `KV_POOL_GIB` (12.5) the budget was 0, the memory-sized pool taking the room, so `PROMPT_CACHE_GIB` alone does nothing; `KV_POOL_GIB=4` with `PROMPT_CACHE_GIB=8` gave 6.2 GiB for kept states (at most 32, fixed 102 MiB each) and a pool of 825,344 tokens, and 39/39 greedy replies identical to the base engine by token hash with 16 cases resumed from kept states; with spill v2 on (round 2's image) the same sizing gave 5.1 GiB, its 1,066 MiB of pinned staging being taken before the pool. So `./start.sh` warns when `PROMPT_CACHE_GIB` (or `--prompt-cache-gib`) is above 0 while `KV_POOL_GIB` is not set (`scripts/port.sh`: `port_pool_check`). Spill v2 and KV sharding's carries keep and restore these states, so they need the same room (spill v2 refuses to start on a budget of 0). MTP drafts turn the prompt cache off. A keep that would take the machine under the reserve is skipped at that mark on every rank (a warning; earlier and later states are unaffected), and a save the engine still refuses is skipped the same way on every rank, the prompt going on from the rows it ran, instead of ending the server |
| parked pages (no setting) | `zig/patches/0014-zig-parked-pages.patch`: the KV pool's room and the kept prompt states' budget count MemAvailable plus the pages parked on the kernel's per-CPU free lists (on a unified GPU, read once at start; a line per rank from 0.25 GiB), so a start right after an image copy is not budgeted several GiB short. `./start.sh`'s memory check counts them the same way, here and on each worker (`scripts/memavail.sh`; `scripts/memavail_test.sh`), only for an engine that does (`ZIG_HAS` includes `parked`) |
| `SPILL_GIB`, `SPILL_DIR`, `SPILL_MIN_TOKENS`, `SPILL_MIN_FREE_GIB`, `SPILL_HIGHWATER`, `SPILL_FLUSH_S` | the v1 tier (`cuda_spill.zig`, in `zig/patches`): the same `--spill-gib` / `--snapshot-dir` / `--spill-*` flags as the Python path and `TF_SPILL_FLUSH_S`, mapped in `scripts/port.sh`. Default `SPILL_GIB=0`. It writes the prompt cache's kept states, so it **needs `PROMPT_CACHE_GIB` above 0** (and room for it, above) and DFlash2 drafts: `./start.sh` refuses `SPILL_GIB` above 0 with `PROMPT_CACHE_GIB=0` or `DRAFTER=mtp`, where before the tier silently never opened. `SPILL_DIR` is checked on every Spark before anything is stopped (created as you when missing, writable, on a local disk, outside `HF_CACHE` and any NFS export, `SPILL_MIN_FREE_GIB` free counting the tier's own folders as room, since the engine frees those itself; a folder the check cannot read only warns). The ranks run as root, and this tier writes its `tfspill-<hash>` folders and files as root, readable by root only (0700 / 0600): removing them takes `sudo` (`tools/spill_check.py --sudo`, operator-only). `--spill-gib` as an extra argument is refused (the folder would not be checked or mounted), and with the tier on so are `--snapshot-dir`, `--spill-highwater`, `--spill-min-tokens` and `--spill-min-free-gib` (`start.sh` passes them itself, after the extra arguments: set the `SPILL_*` setting). Each rank opens its tier from its own prompt cache budget: after the start, a rank whose log shows no `spill tier:` line is warned about |
| shared-cut chat id (no setting) | `zig/patches/0014-zig-chat-id.patch`, a base-engine fix that acts only with a prompt cache: a kept shared system-prompt cut (`SHARED_PREFIX`) no longer gives its conversation to the conversations behind it, so a second conversation behind the same system prompt is a conversation of its own and `TF_GLM_KEEP_PER_CHAT` no longer counts both together (measured on the pair: zig-engine resumed the second conversation at the shared cut, the patch at its own state). Two side effects, neither a reply: `/health`'s `kept_mix` lists a shared cut as a conversation of its own (one state), and the v1 spill tier (`SPILL_GIB` above 0) no longer counts such a cut as superseded by the turn kept after it, so it writes the cut to disk when it is evicted (one more spilled state a shared prompt) |
| `SPILL_V2` (`TF_GLM_SPILL_V2`), `SPILL_QUOTA` (`TF_GLM_SPILL_QUOTA`, a share), `SPILL_TURNS` (`TF_GLM_SPILL_TURNS`), `TF_GLM_SPILL_READ_BLOCKS`, `TF_GLM_SPILL_STAGE_MIB`, `TF_GLM_SPILL_LOAD_MS`, `TF_SPILL_DEFER_MS` | implemented (`zig/patches/0018`, `0019`, `0021`; Robert's GLM patches 0088/0110, `ZIG_HAS=spill2`): with `SPILL_GIB` above 0, `SPILL_V2=1` writes every kept prompt state to each Spark's disk when it is kept, in 2,048-token blocks a conversation's turns share, and reads it back beside decoding, also after a restart or a crash (a kill keeps every published point). Default `SPILL_V2=0`: v1, which stays the rollback (the two never run together). It needs the prompt cache (above: `PROMPT_CACHE_GIB` and a lower `KV_POOL_GIB`), two Sparks and DFlash2 drafts; the ranks agree on an identity at start (`scripts/port.sh` exports the engine, series, checkpoint and drafter ids) and its pinned staging (1,066 MiB measured) is taken before the pool (the kept-state budget measured 5.1 GiB at `KV_POOL_GIB=4` with `PROMPT_CACHE_GIB=8`, against 6.2 GiB with the tier off). Defaults: `SPILL_QUOTA` 0 (none), `SPILL_TURNS` 4, `TF_GLM_SPILL_READ_BLOCKS` 16, `TF_GLM_SPILL_STAGE_MIB` 256, `TF_GLM_SPILL_LOAD_MS` 40, `TF_SPILL_DEFER_MS` 5000; `SPILL_HIGHWATER` is v1's (not passed). Its folders under `SPILL_DIR` are given to `SPILL_DIR`'s owner (the files in them stay root's). **Measured on the pair** (round 2's image): 39/39 greedy replies identical by token hash; the port's spill check passed 20 of 22 main cases (restores from disk exact, damaged and missing files refused with exact replies) and 4 of 4 restart cases; its dual-restore case failed there; that was the client's (its D1 item kept its point inside its own question; the check now pads such an item), and with the fixed client two concurrent restores both came from disk in 2.28 s while a stream decoded (round 3's image, KV sharding and spill v2 on, 2026-10-24). Without the v2 tier these settings do nothing (a warning) |
| `KV_SHARD` (`TF_GLM_KV_SHARD`), `TF_GLM_KV_SHARD_CARRY`, `TF_GLM_KV_SHARD_TWINS`, `TF_GLM_KV_SHARD_CHECK` | implemented (`zig/patches/0020`, `0022`; Robert's GLM patch 0118, `ZIG_HAS=kvshard`): each conversation's latent rows stored once, on the Spark that owns its part of the pool, so the pool holds more tokens in the same memory; decode computed by the owner, a prompt's history fetched or a fresh prompt borrowing the other part's free rows (`TF_GLM_KV_SHARD_TWINS`, 1), kept states carried to a full part's other side (`TF_GLM_KV_SHARD_CARRY`, 1), the ranks' layout digests compared every `TF_GLM_KV_SHARD_CHECK` decode rounds (1024). With the spill tier: v2 only (`0023`, `ZIG_HAS=spillshard`). Refused at TP other than 2, `PARALLEL=1`, `DRAFTER=mtp` and `TF_GLM_MULTI_LONE=1`. Default 0. **Measured on the pair** (round 2's image): 39/39 greedy replies identical (8 streams, a 99,899-token prompt; also with `TF_GLM_KV_SHARD_TWINS=0`); **1.83x the tokens in the same memory** (a pool of 1,511,424 tokens in two parts against 825,344 unsharded); decode 1.4-3.8% below unsharded at 1 to 8 streams (not interleaved; repetitions varied up to 4%). Spill v2 with KV sharding (`0023`), round 3's image: 39/39 identical and the same spill checks |
| continued chats (no setting) | implemented (`zig/patches/0017`; Robert's GLM patch 0111): a request that continues a conversation without resuming its own turn state keeps that conversation (`chats_continued` and `chats_continued_prefill_tokens` on `/health` and `/metrics` with a prompt cache). With it the lane host names a shared cut's conversation by the request again, so the v1 spill tier counts the cut as superseded by its conversation's later turn, as the base engine did (the shared-cut chat id row above; spill v2 writes at keep and is not affected) |
| `PREFIX_WAIT`, `TF_GLM_PREFIX_WAIT` | not ported, and not needed: the Zig engine admits and fills one prompt at a time, so no request prefills a prefix another is still filling. Set to anything but 0, it warns and does nothing |
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
`TF_ROUND_LOG`, `TF_TP_*` when you set them. `ZIG_HAS` (space-separated; default `mtp displaykv parked spill2
kvshard spillshard`: `parked` for 0119's port, `0014-zig-parked-pages`, and `spill2`, `kvshard` and `spillshard` for
Robert's spill v2, KV sharding and the two together, `zig/patches/0018`-`0023`) names what the image serves: a narrower `ZIG_HAS` (an
image built without those patches) refuses their settings, and without `parked` the start memory check reads
`MemAvailable` alone.

Robert's GLM features (the spill tier, spill v2, KV sharding) are mapped to the ranks in one place, `scripts/port.sh`
(`port_map`): with all of them off (the default) it adds no flag, variable, mount or label. Each setting is also taken
from its `TF_GLM_` name when only that one is set; the two switches (`TF_GLM_KV_SHARD`, `TF_GLM_SPILL_V2`) reach the
ranks only through it. A fork can turn their defaults on (`PORT_SPILL_DEFAULT_GIB`, `PORT_SPILL_V2_DEFAULT`,
`PORT_SHARD_DEFAULT`, `PORT_SPILL_QUOTA_DEFAULT`): a default that cannot run is turned off with a warning, where the
same setting given explicitly refuses. Every rank's container carries `tf.recipe.port` (what was mapped; none while all
is off), and a `./start.sh` that finds the server running refuses when that label differs from what it is asked for
(`./start.sh restart` applies it). Tests: `scripts/port_test.sh` (bash 3.2, so it runs on a Mac: `scripts/config.sh`
and `scripts/nodes.sh` use `declare -A`, so it sources `scripts/port.sh` and `scripts/memavail.sh` alone).

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

start.sh refuses the Zig engine at TP=4 on a switch (`./start-tp4.sh`, not tested), with `SPLIT=1`, or with
`COMM=roce`: the ring's RoCE relay to the Spark across (`TF_ROCE_RING=1`, the Python engine's patch 0099) is not
ported, so every collective goes over NCCL's ring channels and its graph files (`scripts/nodes.sh`, as for the Python
engine on a ring).

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

```sh
tools/make-zig-patches.sh <Zig port checkout> [branch] [base]   # refresh zig/patches/0001-0005 and zig/aot
scripts/port-patches.sh --dry-run                                # check 0017+ from the port's engine branch (no write)
scripts/port_patches_test.sh; scripts/port_test.sh               # the recipe's port checks (a Mac is enough)
python3 -I tools/selftest.py                                     # the client tools against a fake server
ENGINE=zig scripts/prepare.sh                                    # build (or pull) the image on both Sparks
ENGINE=zig DRY_RUN=1 ./start.sh                                  # print every rank's docker command
ENGINE=zig ./start.sh                                            # start it
```

Credits: TensorFold's Zig engine is by Ash Hart ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)) and its
contributors; the native CUDA server, registry and rank mirror are jschmied's
([PR #443](https://github.com/ashhart/TensorFold/pull/443)); the one-shot RoCE all-gather is
[b12x / RoCEnante](https://github.com/local-inference-lab) by local-inference-lab. See [`CREDITS.md`](../CREDITS.md).
