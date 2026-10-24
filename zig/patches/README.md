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
| `0014-zig-glm-parked-pages.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: the startup budgets count the pages parked on the per-CPU free lists (port of GLM 0119) (engine `glm/series-r5` @ `d9d631748d76`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0015-zig-server-prepare-memory.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | server + core: a long prompt's preparation no longer runs rank 0 out of memory (port of DeepSeek 0023, GLM's three renders) (engine `glm/series-r5` @ `2a16b14a4903`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0016-zig-core-chat-id.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | core prompt cache: a shared cut no longer joins two conversations (engine `glm/series-r5` @ `b42a552e7693`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0017-zig-core-chat-continued.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | core prompt cache + lane host: a request keeps its conversation when it continues it without its own turn state (port of GLM 0111) (engine `glm/series-r5` @ `b54a7d4048bb`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0018-zig-core-spill-store.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | core/spill: the disk block store of spill v2 (port of GLM 0110's store, lifted from the DeepSeek port) (engine `glm/series-r5` @ `e4fb5b76cb1b`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0019-zig-core-cache-tier.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | core prompt cache + lane host: spill v2's Tier, restore jobs and the restore service beside decoding (port of GLM 0110's host side) (engine `glm/series-r5` @ `f139cada683a`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0020-zig-glm-kv-shard-host.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: KV shard host side: the part planner and pool parts, switches and their agreement, storage, placement check and telemetry (port of GLM 0118, parts 1-2) (engine `glm/series-r5` @ `8455b2cef6de`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0021-zig-glm-spill-blocks.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | glm_cuda: spill v2 for GLM: kept-state packing, the rank protocol, the compatibility fingerprint and the GLM tier behind TF_GLM_SPILL_V2=1 (port of GLM 0110) (engine `glm/series-r5` @ `68854c0fff08`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0022-zig-glm-kv-shard.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: KV shard device paths: owner-computed decode and glm_shard.cu, a prompt chunk's history fetch, prompt borrowing and kept states' carries; TF_GLM_KV_SHARD=1 on (port of GLM 0118, parts 3-5) (engine `glm/series-r5` @ `d0b25f0c94ec`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0023-zig-glm-spill-kv-shard.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | glm_cuda: spill v2 with KV sharding: each rank keeps and restores its own view of a kept state under its home part (port of GLM 0118's spill blocks) (engine `glm/series-r5` @ `15e06e463cc9`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0024-zig-glm-1m-coverage.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: an offline test that every Triton lookup at TP=2, up to a 1,048,576-token window, is in the captured kernel set; TF_GLM_MULTI_LONE=1 with KV sharding warns instead of refusing (engine `glm/series-r5` @ `e01a4efb4f66`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0025-zig-glm-mtp-cache.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: MTP drafts with the prompt cache (one stream): the head absorbs one row behind the target and kept states carry its section; TF_GLM_MTP_DIGEST (engine `glm/series-r5` @ `853d13b036ef`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0026-zig-glm-fill-steps.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: prompts fill a 2048-row chunk a step between decode rounds behind TF_GLM_FILL=1 (concurrent fills, step A) (engine `glm/series-r5` @ `2387efe46102`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0027-zig-glm-pictures.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | core + server + glm_cuda lanes: a prompt with pictures is kept and resumed under its content key; a pass attaches its pictures once, every mark's segment injects them, a resumed pass runs the tower only for the pictures past its cut (TF_GLM_IMAGE_CACHE, TF_GLM_IMAGE_MARK_SNAP) (engine `glm/series-r5` @ `8acebf74c7af`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0028-zig-glm-spill-pictures.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | glm_cuda: spill v2 for prompts with pictures: points with pictures before their cut written as format v3 on every rank, found only for the same pictures in the same places, with KV sharding too (port of the DeepSeek port's spill-pictures) (engine `glm/series-r5` @ `4483a96ecf21`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0029-zig-glm-tower-workspace.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) | glm_cuda: the vision tower's workspace for its largest pass is taken at start, before the staging and the pool are sized (engine `glm/series-r5` @ `e3ae8a62096d`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |
| `0030-zig-glm-prefix-wait.patch` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84](https://github.com/BadAd84) (#118) | lane host: a sliced fill waits for the shared cut another fill is computing (TF_GLM_PREFIX_WAIT, wojo's GLM 0110) (engine `glm/series-r5` @ `e1352d709873`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |

## The port of Robert Wojciechowski's GLM features (0014 and later)

The GLM-5.3-Flash Python patches by
Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X), spill v2 (0088/0110,
[PR #78](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/78) and
[#124](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/124)), continued chats (0111,
[#128](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/128)), KV sharding (0118,
[#152](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/152)) and the parked pages
(0119, [#149](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/149)), ported by him to
this engine, with two fixes to the base engine he found on the way (a long prompt's preparation running rank 0 out of
memory, the DeepSeek recipe's 0023, and a shared prompt cut joining two conversations), arrive here as `0014` to
`0023`: one patch per feature, in the order they build on each other, each the diff of one commit of the engine branch
`glm/series` (published as [`wojo/TensorFold` `glm-zig/series`](https://github.com/wojo/TensorFold/tree/glm-zig/series), from
`glm-zig/engine-base`) with a header that says what it does,
its switches and defaults, what was measured on the pair and whose work it is (spill v2's block store design is
[BadAd84](https://github.com/BadAd84)'s, #118). Round 4 (`0024`-`0029`, from the parity plan with his Python GLM: a
1M window's kernel coverage, MTP drafts with the prompt cache, sliced prompt fills, prompts with pictures kept, resumed
and spilled, the vision tower's workspace at start) continues the series on `glm/series-r4` (not published yet). After
`0029` the engine is the port's integration branch (`glm/integration-r4`, round 4) without the port's notes.

| Patches | Switch (both ranks unless noted) | Default (engine; this recipe) |
| --- | --- | --- |
| `0014` parked pages | none (sizing only) | on |
| `0015` prepare memory | `TENSORFOLD_PREPARE_FLOOR_GIB` (rank 0; GiB, `0` turns the gate off) | on (3 GiB) |
| `0016` shared-cut chat id | none | on |
| `0017` continued chats | none (chat ids drive only the cache's eviction policy and `/health`) | on |
| `0018`, `0019`, `0021` spill v2 | `TF_GLM_SPILL_V2=1` with `--spill-gib` and `--snapshot-dir` (recipe: `SPILL_GIB`, `SPILL_V2`, `scripts/port.sh`) | off; v1 stays the rollback |
| `0020`, `0022` KV sharding | `TF_GLM_KV_SHARD=1` (recipe: `KV_SHARD`), `_CARRY`, `_TWINS`, `_CHECK` | off |
| `0023` spill v2 with KV sharding | both of the above | off |
| `0024` 1M kernel coverage | none (a host test); `TF_GLM_MULTI_LONE=1` with KV sharding now only warns | - |
| `0025` MTP drafts with the prompt cache | none (`TF_GLM_MTP_DIGEST=1`: a debug log of the MTP head's state) | on wherever a prompt cache and MTP drafts are (one stream); spill and KV sharding still refuse MTP |
| `0026` sliced prompt fills | `TF_GLM_FILL=1` (recipe: `FILL`), refused with MTP drafts | off |
| `0027`, `0028` prompts with pictures kept, resumed and spilled | `TF_GLM_IMAGE_CACHE` (`0`: none kept), `TF_GLM_IMAGE_MARK_SNAP=1` (rank 0: no state cut inside a picture) | on with a prompt cache (text prompts unchanged); mark snap off |
| `0029` the vision tower's workspace at start | none (vision only, rank 0) | on |

Spill v2's and KV sharding's switches (`0018`-`0023`) are off in the engine and byte-identical when off, and off in
this recipe (`scripts/port.sh`'s defaults). Round 4's: `0026`'s `TF_GLM_FILL` is off (recipe `FILL=0`); `0025` and
`0027`-`0028` change nothing without a prompt cache (`PROMPT_CACHE_GIB=0`, the default) and with one change only MTP
and picture prompts, which no state was kept for before; `0029` takes the tower's workspace at the start (vision
only); none of them has run on the pair yet. The base fixes are on by default, as fixes are: `0014`'s parked pages (no
switch; sizing only, the same replies) and `0015`'s preparation gate (`TENSORFOLD_PREPARE_FLOOR_GIB`, 3 GiB; `0` turns
the gate off, while the rest of `0015`, the scratch given back and the one-pass signer, has no switch). `0016` and
`0017` have no switch either: they change which conversation a kept state belongs to, so which states the cache evicts,
and what `/health`, `/metrics` and rank 0's log report with a prompt cache, never a reply. `ZIG_HAS`'s default lists
`spill2`, `kvshard`, `spillshard` and `parked` now that they are here (`docs/zig-engine.md`). Measured on the pair
(round 1 and round 2's images, the port as one patch): every feature off, `tools/exact.py` 39/39 identical to the base
engine; the prompt cache on (`KV_POOL_GIB=4`, `PROMPT_CACHE_GIB=8`: a 6.2 GiB budget for kept states, 5.1 GiB with
spill v2 on, whose 1,066 MiB of pinned staging comes first), 39/39; KV sharding on, 39/39, the pool 1.83x the tokens in
the same memory; spill v2 on, 39/39, `tools/spill_check.py` 20 of 22 main cases and 4 of 4 restart cases (each header
says which). Round 3 as this series (`0014`-`0023`, 2026-10-24): every feature off 39/39; the cache on 39/39; KV sharding
39/39 (fan-out one fill); spill v2 39/39 and spill v2 with KV sharding (`0023`) 39/39, each passing
`tools/spill_check.py`'s main and restart cases except two the client got wrong (dual-restore's D1 kept its point inside
its question, and "no temporaries" listed the folders while a state was being written again; both fixed in `tools/`)
and cancel (not exercised: the restore finished first); two concurrent restores (`0019`), checked against a running server with
KV sharding and spill v2 both on, both came from disk in 2.28 s while a stream decoded (its worst gap 0.13 s). Decode on
the pair (`--parallel 8`, short prompts, aggregate tok/s at 1/2/4/8 streams): every feature off 51.4/72.0/91.9/118.2,
KV sharding + spill v2 + the cache 49.7/65.4/92.5/115.4 (both `PARALLEL=8`, `CONTEXT=131072`); `ENGINE=python` at the
same settings 50.7/72.0/93.0/126.0 (with the 1M window 50.5/72.7/94.2/123.6). Cold prefill about 2,000
tok/s on all three at 13k and 51k tokens. `0014`, `0015` and `0016` are also offered alone as pull requests
against this branch, with the same diff: whichever lands first, the other copy is an exact duplicate (`git apply
--reverse --check` passes). Their headers are the series' messages, not the standalones': `0016`'s leaves out a note
of `glm/pr-chat-id`'s. With that fix alone, a shared cut and the turn kept after it get different conversation ids, so
the v1 spill tier no longer counts the cut as superseded by that turn and writes it to disk when it is evicted (as the
Python fork's `_superseded` exempts shared cuts). From `0017` on the lane host keeps both in the request's
conversation (as Python 0089 does), which `0016`'s header says instead: with the whole series v1 counts the cut as
superseded by its conversation's later turn again, as the base engine did (spill v2 writes at keep and is not affected).

They are made by `scripts/port-patches.sh`, never by hand: the engine branch (default `glm/series-r4`) is exported
relative to `glm/zig-engine-base`, which must be `ZIG_TF_REF` plus `0001`-`0013` exactly (with the captured kernel set
at `zig/tests/cuda/glm_p1/aot`, which no patch carries); each first-parent commit becomes one patch, named with
`--name REV=SLUG` (else the name of the patch already exported from that commit, else from its subject), its message
the patch's header. It refuses unless the patches here reproduce the base on a clean `ZIG_TF_REF` and every new patch
applies after them and gives the branch's tree (outside the kernel set and the port's notes, `docs/glm-port/`). An
exported patch's first line is `# port-patches.sh: ...`; the script replaces those and rewrites only the table rows it
owns (marked `<!-- port-patches -->`), after the last hand-written row above. `scripts/port-patches.sh --dry-run`
checks without writing (with no arguments it reproduces `0014`-`0029` from `glm/series-r4`); `--branch
glm/zig-engine-base` checks the base alone.

## Ordering rules

- Filename order is apply order. Each patch applies on a clean `ZIG_TF_REF` after every patch before it, and none
  after it may be needed by it.
- `0001`-`0005` are regenerated together by `tools/make-zig-patches.sh`, which removes every `*.patch` first: it
  refuses while any patch after `0005` is here (they would be lost) unless `--force`. Move them aside and back, or
  rebase them on the new `0001`-`0005`.
- A new hand-written patch takes the number after the last hand-written one (`0014` today), even when an exported
  patch already has it: then rebase the engine branch on a base that contains it and run `scripts/port-patches.sh`
  again, which numbers its patches after the last hand-written one. Never renumber an exported patch by hand (its
  header names the commits it came from).
- Remove exported patches from the last one back, never one in the middle: each builds on those before it.
- Never edit an exported patch: change the engine commit and export again.
