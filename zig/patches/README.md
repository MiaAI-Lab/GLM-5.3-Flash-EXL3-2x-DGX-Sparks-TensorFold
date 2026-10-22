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

## The port of Robert Wojciechowski's GLM features (0014 and later)

Robert Wojciechowski's ([wojo](https://github.com/wojo)) GLM-5.3-Flash Python patches, spill v2 (0088/0110,
[PR #78](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/78) and
[#124](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/124)), continued chats (0111,
[#128](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/128)), KV sharding (0118,
[#152](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/152)) and the parked pages
(0119, [#149](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/149)), are being ported to
this engine (planned, in progress: none is here yet). They arrive as `0014-zig-<name>.patch` and later, one patch per
feature, each the diff of one engine commit with a header that says what it does, its switches and defaults, what was
measured and whose work it is (spill v2's block store design is [BadAd84](https://github.com/BadAd84)'s, #118). Every
switch is off in the engine and byte-identical when off; `scripts/port.sh` has the recipe's settings for them (all off
by default) and `ZIG_HAS` lifts their refusals once they are here (`docs/zig-engine.md`).

They are made by `scripts/port-patches.sh`, never by hand: the engine branch (default `glm/integration`, in the port's
TensorFold repository) is exported relative to `glm/zig-engine-base`, which must be `ZIG_TF_REF` plus `0001`-`0013`
exactly (with the captured kernel set at `zig/tests/cuda/glm_p1/aot`, which no patch carries); each first-parent commit
becomes one patch, named with `--name REV=SLUG` (else from its subject), its message the patch's header. It refuses
unless the patches here reproduce the base on a clean `ZIG_TF_REF` and every new patch applies after them and gives the
branch's tree (outside the kernel set and the port's notes, `docs/glm-port/`). An exported patch's first line is
`# port-patches.sh: ...`; the script replaces those and rewrites only the table rows it owns (marked
`<!-- port-patches -->`), after the last hand-written row above. `scripts/port-patches.sh --dry-run` checks without
writing; `--branch glm/zig-engine-base` checks the base alone.

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
