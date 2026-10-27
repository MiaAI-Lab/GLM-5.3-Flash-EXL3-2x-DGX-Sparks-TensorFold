# zig/patches

Patches to TensorFold at `ZIG_TF_REF` (v1.0.7, `e7c523eb`; `scripts/config.sh`), applied in filename order by
`zig/Dockerfile` (`git apply --check`, then `git apply`, in the checkout's root). The image tag (`zig-v1.0.7-<hash>`)
changes with them, the captured kernel set `zig/aot/glm-aot-sm121.tar.xz`, the Dockerfile and the pins, so the next
`scripts/prepare.sh` builds (or pulls) a new image. This README is not a patch: the Dockerfile and the hash take
`*.patch` only.

| Patch | By | What it does |
| --- | --- | --- |
| `0001-zig-shared-engine.patch` | Mia's AI Lab | the shared runtime: every TensorFold file the port modifies (lane host, `native/cuda`, the server, the build), 80 files. With 0002-0005, the Zig port at `r107-int` `a10c8647` against v1.0.7 (`zig/PORT_REF`), cut by path by `tools/make-zig-patches.sh` |
| `0002-zig-flashnext-cuda.patch` | Mia's AI Lab, after Ash Hart and the TensorFold contributors | the Flash Next Zig CUDA engine and its kernels, which the GLM engine builds on: a port of TensorFold's Python Flash Next CUDA engine (71 files) |
| `0003-zig-glm-kernels.patch` | Mia's AI Lab, after Ash Hart and the TensorFold contributors | the GLM CUDA kernels, `zig/kernels/cuda/glm_*` (EXL3 experts and the rest TensorFold's GLM engine runs as CUDA C++; 11 files) |
| `0004-zig-glm-engine.patch` | Mia's AI Lab | the GLM engine: `zig/src/families/glm_cuda`, the GLM CLI and host (81 files); it replays the captured Triton kernels (`zig/aot/`) |
| `0005-zig-glm-tests.patch` | Mia's AI Lab | the rest under `zig/`: host tests and fixtures (1,118 files). The captured kernel set is not in it: it ships as `zig/aot/glm-aot-sm121.tar.xz` |
| `0006-zig-glm-scores-rows-g128.patch` | Hector Isaza ([hecisaza](https://github.com/hecisaza)), PR #170 | fix: the prompt-tail indexer scores (`_scores_rows`) launch at G=128, which the kernel set has in every argument class (a 20,995-token prompt needed G=88: `MissingTritonVariant`); G only splits the work, the scores are unchanged. No switch |
| `0007-zig-flashnext-record-before-long-batch.patch` | Hector Isaza, PR #174 | fix: a long batch's record goes on the TP control link before the batch, so a prompt larger than the socket buffers (~262K tokens and up) cannot deadlock the ranks; with a 16 MB-prompt regression test. No switch |
| `0008-zig-aot-log-missing-variant.patch` | Hector Isaza, PR #181 | diagnostics: a failed captured-kernel lookup also logs its constexprs and a bounded `MISSING-VARIANT` line. Logging only: selection and the error are unchanged |
| `0009-zig-glm-tpn-pool.patch` | Hector Isaza, PR #190 | TP=3/4: the KV pool is agreed over every rank (the all-gather was sized for two). Two ranks unchanged |
| `0010-zig-glm-tpn-roce-probe.patch` | Hector Isaza, PRs #192 and #194 | TP=3/4 direct RoCE (`COMM=roce`): every rail's subnet is probed beyond two ranks, kept only when the probed GID is the fixed index's own (`railAddr`); a pair of ranks left without a shared subnet keeps NCCL. Host tests |
| `0011-zig-glm-greedy-draft-probs.patch` | Hector Isaza, PR #191 | sampler fix: a greedy draft that asks for its probability gets a full top-k row (the one-candidate fast path left the draft ids stale and MTP drafts were almost never accepted; replies stayed exact). No switch |
| `0012-zig-glm-cache-reserve.patch` | Hector Isaza, PR #202 | the prompt cache's keep floor and refused saves: `TF_GLM_KEEP_HEADROOM_GIB` (2 GiB; 0 turns it off) above the memory reserve, the ranks agree on room before a keep and on the save after it (a refused save is kept by none, the prompt goes on from its rows), and the cache stays off with an MTP head |
| `0013-zig-glm-round-batching.patch` | Hector Isaza, PR #202 | a round's draws and commits in one call each with several streams (one sampler readback, one commit table, one tap pass): eight streams 90.5 tok/s with drafts (89.1 before), replies identical to the per-window engine's. No switch |
| `0014-zig-chat-id.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X), PR #213 | fix: a kept shared system-prompt cut no longer gives its conversation to the conversations behind it, so `TF_GLM_KEEP_PER_CHAT` counts each conversation on its own (with a prompt cache only; `docs/zig-engine.md` has its two side effects). No switch |
| `0014-zig-parked-pages.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X), PR #212 | the startup budgets (the KV pool's room and the kept states' budget) count the pages parked on the per-CPU free lists (GLM 0119's rule); `./start.sh`'s memory check too. No switch; sizing only |
| `0014-zig-prepare-memory.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X), PR #211 | fix: a long prompt's preparation no longer runs rank 0 out of memory (scratch given back, kept reasoning signed in one pass, the connection arena capped, and a memory gate in arrival order: `TENSORFOLD_PREPARE_FLOOR_GIB` 3, `TENSORFOLD_PREPARE_WAIT_MS` 5000, `TENSORFOLD_PREPARE_MAX_WAITING` 64) |
| `0017-zig-core-chat-continued.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | core prompt cache + lane host: a request keeps its conversation when it continues it without its own turn state (port of GLM 0111) (engine `glm/r8b` @ `ff464227b3d5`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0018-zig-core-spill-store.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | core/spill: the disk block store of spill v2 (port of GLM 0110's store, lifted from the DeepSeek port) (engine `glm/r8b` @ `3a03212607f6`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0019-zig-core-cache-tier.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | core prompt cache + lane host: spill v2's Tier, restore jobs and the restore service beside decoding (port of GLM 0110's host side) (engine `glm/r8b` @ `d7a85d9bdc1d`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0020-zig-glm-kv-shard-host.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: KV shard host side: the part planner and pool parts, switches and their agreement, storage, placement check and telemetry (port of GLM 0118, parts 1-2) (engine `glm/r8b` @ `8d1796cc3ab1`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0021-zig-glm-spill-blocks.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | glm_cuda: spill v2 for GLM: kept-state packing, the rank protocol, the compatibility fingerprint and the GLM tier behind TF_GLM_SPILL_V2=1 (port of GLM 0110) (engine `glm/r8b` @ `3022ee57d656`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0022-zig-glm-kv-shard.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: KV shard device paths: owner-computed decode and glm_shard.cu, a prompt chunk's history fetch, prompt borrowing and kept states' carries; TF_GLM_KV_SHARD=1 on (port of GLM 0118, parts 3-5) (engine `glm/r8b` @ `069c49ffd2dc`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0023-zig-glm-spill-kv-shard.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | glm_cuda: spill v2 with KV sharding: each rank keeps and restores its own view of a kept state under its home part (port of GLM 0118's spill blocks) (engine `glm/r8b` @ `1ea862b590e4`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0024-zig-glm-1m-coverage.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: an offline test that every Triton lookup at TP=2, up to a 1,048,576-token window, is in the captured kernel set; TF_GLM_MULTI_LONE=1 with KV sharding warns instead of refusing (engine `glm/r8b` @ `95526fcaa650`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0025-zig-glm-mtp-cache.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: MTP drafts with the prompt cache (one stream): the head absorbs one row behind the target and kept states carry its section; TF_GLM_MTP_DIGEST (engine `glm/r8b` @ `863b5f903e44`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0026-zig-glm-fill-steps.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: prompts fill a 2048-row chunk a step between decode rounds behind TF_GLM_FILL=1 (concurrent fills, step A) (engine `glm/r8b` @ `2cb4f1d2ccf8`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0027-zig-glm-pictures.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | core + server + glm_cuda lanes: a prompt with pictures is kept and resumed under its content key; a pass attaches its pictures once, every mark's segment injects them, a resumed pass runs the tower only for the pictures past its cut (TF_GLM_IMAGE_CACHE, TF_GLM_IMAGE_MARK_SNAP) (engine `glm/r8b` @ `f4bdb3247840`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0028-zig-glm-spill-pictures.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | glm_cuda: spill v2 for prompts with pictures: points with pictures before their cut written as format v3 on every rank, found only for the same pictures in the same places, with KV sharding too (port of the DeepSeek port's spill-pictures) (engine `glm/r8b` @ `7ffb37f42b63`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0029-zig-glm-tower-workspace.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: the vision tower's workspace for its largest pass is taken at start, before the staging and the pool are sized (engine `glm/r8b` @ `4bab856b97f4`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0030-zig-glm-prefix-wait.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | lane host: a sliced fill waits for the shared cut another fill is computing (TF_GLM_PREFIX_WAIT, wojo's GLM 0110) (engine `glm/r8b` @ `1305bda37155`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0031-zig-glm-prompt-scores-cuda.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: a prompt chunk's DSA indexer in CUDA, pool scores and one-pass top-K pools (TF_GLM_SCORES_CUDA, default on; ported from BadAd84's Python patches 0101/0105) (engine `glm/r8b` @ `fe2f060f7929`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0032-zig-glm-decode-speed.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: decode clusters, draft prelaunch and narrow prompt tiles (TF_GLM_QMM_CLUSTERS, TF_GLM_DRAFT_PRELAUNCH, TF_GLM_PROMPT_TILE; after BadAd84's GLM 0103, 0100, 0102) (engine `glm/r8b` @ `0e0518f341f2`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0033-zig-glm-multi-prefill.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: several waiting prompts fill a chunk each a step (TF_GLM_MULTI_PREFILL, default off; with TF_GLM_FILL=1) (engine `glm/r8b` @ `1571e7dfdd29`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0034-zig-glm-nucleus-race.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: a sampled row with no top_p cut and no min_p draws by a GPU race over every token (TF_GLM_NUCLEUS_RACE, default on) (engine `glm/r8b` @ `3155e0ea134a`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0035-zig-glm-nucleus-union.patch` | Hector Isaza | glm_cuda: read TENSORFOLD_NUCLEUS_UNION so top_p draws stop falling back to whole-shard sorts (engine `glm/r8b` @ `f190863a4e08`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0036-zig-glm-nucleus-agree.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: the ranks agree on TF_GLM_NUCLEUS_RACE with TENSORFOLD_NUCLEUS_UNION (one word, bit 1) (engine `glm/r8b` @ `6cc5c62a69ae`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0037-zig-glm-nucleus-race-check.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: the race's margin is relative past 1, and TF_GLM_NUCLEUS_RACE_CHECK=1 draws every raced row again from whole shards (engine `glm/r8b` @ `3c8b05d8bbb7`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0038-zig-glm-batched-fill.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: several prompts' prefill chunks in one forward (TF_GLM_MULTI_PREFILL=2, with TF_GLM_FILL=1; default 0, as today) (engine `glm/r8b` @ `70fd9f5b8d6d`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |

## Spill v2, continued chats and KV sharding (0017-0023)

The GLM-5.3-Flash Python patches by
Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X), continued chats
(0111, [#128](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/128)), spill v2 (0088/0110,
[#78](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/78) and
[#124](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/124)) and KV sharding (0118,
[#152](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/152)), ported by him to this
engine, after the three hand-written `0014` base fixes (#211-#213): one patch per feature, in the order they build on
each other, each the diff of one commit of the engine branch `glm/series` (Robert Wojciechowski's
[`wojo/TensorFold` `glm-zig/series`](https://github.com/wojo/TensorFold/tree/glm-zig/series), rebased by Mia's AI Lab onto
ezoushen's `0015` and `0016` and published as
[`MiaAI-Lab/TensorFold` `glm-zig/series-on-186`](https://github.com/MiaAI-Lab/TensorFold/tree/glm-zig/series-on-186), from
[`glm-zig/engine-fixes-186`](https://github.com/MiaAI-Lab/TensorFold/tree/glm-zig/engine-fixes-186) = `ZIG_TF_REF` plus
`0001`-`0013`, the three `0014`s and `0015`-`0016`; the one change to those commits, in the spill store's, is the tests'
scripted partial write, see its header) with a header that says what it does, its switches and defaults, what was measured on the pair and
whose work it is (spill v2's block store design is [BadAd84](https://github.com/BadAd84)'s, #118). Rounds 4 and 5 (`0024`-`0030`,
from the parity plan with his Python GLM: a 1M window's kernel coverage, MTP drafts with the prompt cache, sliced
prompt fills, prompts with pictures kept, resumed and spilled, the vision tower's workspace at start, and the prefix
wait for sliced fills) continue the series on Robert's `glm/pub-r5-on-186`, published as
[`wojo/TensorFold` `glm-zig/series-r5-on-186`](https://github.com/wojo/TensorFold/tree/glm-zig/series-r5-on-186).

| Patches | Switch (both ranks) | Default |
| --- | --- | --- |
| `0017` continued chats | none (chat ids drive only the cache's eviction policy and `/health`) | on, with a prompt cache only |
| `0018`, `0019`, `0021` spill v2 | `TF_GLM_SPILL_V2=1` with `--spill-gib` and `--snapshot-dir` (`SPILL_GIB`), and the identity variables (`docs/zig-engine.md`) | off; v1 stays the rollback |
| `0020`, `0022` KV sharding | `TF_GLM_KV_SHARD=1`, `_CARRY`, `_TWINS`, `_CHECK` | off |
| `0023` spill v2 with KV sharding | both of the above | off |
| `0024` 1M kernel coverage | none (a host test); `TF_GLM_MULTI_LONE=1` with KV sharding now only warns | - |
| `0025` MTP drafts with the prompt cache | none (`TF_GLM_MTP_DIGEST=1`: a debug log of the MTP head's state) | on wherever a prompt cache and MTP drafts are (one stream); spill and KV sharding still refuse MTP |
| `0026` sliced prompt fills | `TF_GLM_FILL=1` (recipe: `FILL`), refused with MTP drafts | off |
| `0027`, `0028` prompts with pictures kept, resumed and spilled | `TF_GLM_IMAGE_CACHE` (`0`: none kept), `TF_GLM_IMAGE_MARK_SNAP=1` (rank 0: no state cut inside a picture) | on with a prompt cache (text prompts unchanged); mark snap off |
| `0029` the vision tower's workspace at start | none (vision only, rank 0) | on |
| `0030` the prefix wait for sliced fills | `TF_GLM_PREFIX_WAIT` (recipe: `PREFIX_WAIT`, 1 with `FILL=1`), `TF_GLM_PREFIX_WAIT_MS` | on in the engine (`1`), acting only with `TF_GLM_FILL=1`; this recipe passes it only with `FILL=1` |

The spill v2 and KV shard switches are off in the engine and byte-identical when off. Not everything here is a switch:
continued chats (`0017`) is on whenever a prompt cache is, and the three `0014` base fixes (the prepare gate, parked
pages, the chat id) are on by default. This recipe sets no switch: they are usable as
environment variables (every rank gets each `TF_GLM_*` variable), and the recipe settings that check and map them
(`SPILL_V2`, `KV_SHARD`, `scripts/port.sh`) come separately. `ZIG_HAS`'s default names `spill2`, `kvshard` and
`spillshard`. Measured on the pair (two GB10s, TP=2, `PARALLEL=8`, `CONTEXT=131072`, DFlash2, 2026-10-24, these
patches as `0017`-`0023` of #214, the same bodies): `tools/exact.py`'s 39 cases identical to the base engine with every
feature off, with the prompt cache on (`KV_POOL_GIB=4`, `--prompt-cache-gib 8`), with KV sharding (the pool 1,511,424
tokens vs 825,344 in the same memory), with spill v2 and with both; `tools/spill_check.py` (in the later recipe PR)
passed its evict/restore, corruption and restart cases once two client bugs were fixed in it; two concurrent restores
from disk while a stream decoded took 2.28 s, the stream's worst gap 0.13 s. Those images had `0014-zig-prepare-memory`'s
gate before its arrival order (#211's review), which no measurement here depends on.

Round 4 and 5's: `0026`'s `TF_GLM_FILL` is off (recipe `FILL=0`); `0025` and `0027`-`0028` change nothing without a
prompt cache (`PROMPT_CACHE_GIB=0`, the default) and with one change only MTP and picture prompts, which no state was
kept for before; `0029` takes the tower's workspace at the start (vision only); `0030` acts only with `FILL=1`. Measured
on the pair (round 4's image, these patches as `0024`-`0029` of the `zig-r4` recipe): every `tools/exact.py` run 39/39
identical to the base engine, with every feature off, the cache on, `FILL=1` with KV sharding and spill v2, MTP with the
prompt cache, and `VISION=1` with KV sharding and spill v2 (picture checks 19/19); a decoder's worst stall behind a
36k-token cold prompt 17.2 s with `FILL` off, 1.07 s with `FILL=1`. `0030` is not run on the pair yet.

They are made by `scripts/port-patches.sh`, never by hand: the engine branch (default `glm/pub-r5-on-186`) is exported relative
to `glm/zig-engine-fixes-on-186`, which must be `ZIG_TF_REF` plus every hand-written patch here exactly (with the captured kernel
set at `zig/tests/cuda/glm_p1/aot`, which no patch carries), numbered after the last hand-written one; each first-parent
commit becomes one patch (named with `--name REV=SLUG`, else the name of the patch already exported from that commit),
its message the patch's header. It refuses unless the patches here reproduce the base on a clean `ZIG_TF_REF` and every
new patch applies after them and gives the branch's tree (outside the kernel set and the port's notes, `docs/glm-port/`).
An exported patch's first line is `# port-patches.sh: ...`; the script replaces those and rewrites only the table rows it
owns (marked `<!-- port-patches -->`). `scripts/port-patches.sh --dry-run` checks without writing (with no arguments it
reproduces `0017`-`0030` from `glm/pub-r5-on-186`); `scripts/port_patches_test.sh` tests the script on a throwaway clone of the engine.

## Ordering rules

- Filename order is apply order. Each patch applies on a clean `ZIG_TF_REF` after every patch before it, and none after
  it may be needed by it. The three `0014`s are independent of each other.
- A new hand-written patch takes the number after the last hand-written one (`0016` today), even when an exported patch
  already has it: then rebase the engine branch on a base that contains it and run `scripts/port-patches.sh` again,
  which numbers its patches after the last hand-written one. Never renumber an exported patch by hand (its header names
  the commits it came from).
- Remove exported patches from the last one back, never one in the middle: each builds on those before it.
- Never edit an exported patch: change the engine commit and export again.
