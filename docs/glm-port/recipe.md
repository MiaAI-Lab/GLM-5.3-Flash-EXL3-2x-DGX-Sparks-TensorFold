# Recipe: settings for Robert's GLM features on the Zig engine (step A)

Branch `zig-port` of Mia's GLM recipe (from `zig-engine` a6c7e08). What the recipe needs before the engine pieces land:
the prompt cache as a setting, one place that maps the ported features to the ranks, refusals where a feature cannot
run (instead of the silent no-ops before), and the parked pages in the start memory check. Step B (the patch export
and the fork's defaults once the engine patches exist) comes later in this worktree.

## What was ported from where

| Piece | From | Here |
| --- | --- | --- |
| `PROMPT_CACHE_GIB` | Mia PR #187 (ezoushen, open, conflicting): `config.sh` setting, `start.sh` check | the same setting and check, plus the spill tier's dependency on it (below) and a warning after the start when rank 0 logs `prompt cache: off` with the tier on |
| `scripts/port.sh` | the DeepSeek recipe's `scripts/port.sh` (R, bdfc9e4): one `port_map`, explicit refuses / fork default degrades, `PORT_*_DEFAULT`, `PORT_MAPPED_RE`, the `tf.recipe.port` label, `spill_dir_probe` / `port_spill_judge` / `port_spill_check` | the same structure, cut down to GLM (see the differences below) |
| setting names and aliases | the GLM Python fork's `scripts/config.sh` (P, 8641c90): `SPILL_QUOTA` (a share, also `TF_GLM_SPILL_QUOTA`), `SPILL_TURNS` (`TF_GLM_SPILL_TURNS`), `TF_GLM_KV_SHARD` default 1 only at TP=2, `PARALLEL>1`, `TF_GLM_MULTI_LONE=0`, `TF_GLM_KV_SHARD_CARRY`, `PREFIX_WAIT` | the same names; `KV_SHARD` and `SPILL_V2` added as the recipe names of `TF_GLM_KV_SHARD` / `TF_GLM_SPILL_V2`; all defaults off |
| engine names | the settings contract in `PORT-STATUS.md` (2026-10-22) | `port_map` (comment lists each name) |
| `scripts/memavail.sh` | R's `scripts/memavail.sh` (`mem_counting`, the engine's 0119 `parkedBytes` rule in awk), its synthetic zoneinfo cases from R's `port_test.sh` | copied; `mem_avail_gb` added for `start.sh`'s whole-GiB check |

## Wiring in `start.sh`

- `scripts/config.sh` sources `scripts/port.sh` where the spill block was (the `SPILL_*` defaults moved there).
- Zig branch: `--prompt-cache-gib "$PROMPT_CACHE_GIB"` replaces the hard-coded 0; both engines' copies of the spill flags
  are gone (port_map makes them). The `ZIG_HAS` comment names the three new tokens; the default stays `mtp displaykv`.
- `port_check_extra "$@"` before the extra arguments join `SERVE_ARGS` (`--spill-*` / `--snapshot-dir` as extras would
  write inside the container: refused).
- `port_validate "$(arg_value --parallel)" "$(arg_value --prompt-cache-gib)"` right after `arg_value`: the effective
  values, so `./start.sh --prompt-cache-gib 8` (documented) satisfies the spill check. All the gates are in it.
- After `need_workers`: `port_spill_check` (every Spark, before anything is stopped), `port_map`, `PORT_SERVE` appended
  to `SERVE_ARGS`. Before the "already running" test, so a degraded default is already reflected in the label.
- "Already running": each rank's `tf.recipe.port` label (none = `off`) must equal `port_identity`, else refuse and say
  `./start.sh restart`.
- Memory check: `mem_avail_gb` here and on each worker (`declare -f mem_counting mem_avail_gb` over ssh), the parked
  GiB noted in the log line when there are any.
- `zig_env` / `env_args`: the passthrough drops `PORT_MAPPED_RE` names, then adds `PORT_ENV` (re-run by the SPLIT retry).
- `RUN_ARGS`: the `/spill` mount when the tier is on; `--label tf.recipe.port=...` only when something is on.
- After "Server answered": `port_check_started` (spill on only).

Off is byte-identical: with nothing set, `port_map` returns empty arrays, `port_identity` is `off` (no label), the
passthrough filter drops only `TF_GLM_KV_SHARD` / `TF_GLM_SPILL_V2` (unset by default), `--prompt-cache-gib 0` as
before. Behaviour that does change with nothing set: the memory check's figure (parked pages counted: no switch, by
the contract), one `docker inspect` per worker when the server already runs, and `--spill-*` extras refused.

## How it differs

From the DeepSeek recipe's `port.sh`:
- No pool setting or derivation (`POOL`, `port_shard_pool`): GLM's pool is memory-sized (`KV_POOL_GIB`), and KV
  sharding changes what fits in it, not its geometry.
- No spill identity variables (`TF_DS_ENGINE_ID`, `TF_DS_RECIPE_PATCHES`, `TF_DS_MODEL_REVISION`, rank weight digests):
  the GLM v2 fingerprint is round 2's (glm-spill); whether the recipe must supply such names is open (below).
- No `WORKER_SPILL_DIR`: this recipe has always used one `SPILL_DIR` on every Spark (mounted at `/spill`).
- No `spill_fix_owner` (handing root-owned spill files back), no `RANK1_EXTRA_ENV`, no display-reserve default.
- Gating by `ZIG_HAS` tokens (the GLM recipe's mechanism), not by an engine branch name.
- The fork's default KV shard is conditional and silent outside its conditions (the Python fork's `_kvs` rule), where
  DeepSeek warns at TP!=2; a missing capability does warn.
- The spill tier has an extra requirement here: the Zig engine opens it only with a prompt cache budget above 0
  (`cuda_native.zig`: `spill = if (p.budget > 0) openSpill(...)`) and not with MTP drafts (`cacheFor` returns null).

From the Python fork's recipe:
- Everything defaults off (the fork has spill 64 GiB, quota 0.25, KV shard 1). A fork sets `PORT_SPILL_DEFAULT_GIB=64`,
  `PORT_SPILL_V2_DEFAULT=1`, `PORT_SHARD_DEFAULT=1`, `PORT_SPILL_QUOTA_DEFAULT=0.25` (the tests' `PT_FORK=1` does that).
- The switches are not exported by `config.sh`; they reach the ranks only through `port_map`, after the checks.
- The parked-page rule is the engine's (count - high_min, 512 MiB rest without high_min), not the fork's
  `_parked_zone` (high_min + 256 MiB of slack): one rule for the recipe and the engine.
- `PREFIX_WAIT` is not ported (the Zig engine fills one prompt at a time; GAP.md): it warns.
- The fork's `parked_pages_check` CI step: this recipe has no CI; `scripts/port_test.sh` runs the synthetic cases, and
  the engine's own 0119 tests come with `glm/base-fixes`.

## Tests

`scripts/port_test.sh` (bash 3.2; run `/bin/bash scripts/port_test.sh` on the Mac): 92 cases. Defaults off; v1 mapping;
the `PROMPT_CACHE_GIB` and MTP refusals; v2 and KV-shard refusals naming the missing patch, with `ENGINE=python` too;
aliases and conflicts; forms; fork defaults (`PT_FORK=1`) degrading in each combination; spill + shard resolution;
`PREFIX_WAIT`; the label; `port_spill_judge` on synthetic lines; `spill_dir_probe` on temporary folders (also through
`sh -c`); `port_spill_check` with the worker's ssh command run locally; the post-start budget-0 warning; the parked
pages on synthetic `/proc` files and `mem_avail_gb` through `bash -c` / `sh -c`; `bash -n` of every script. A
mutation pass (12 single-line breaks of `port.sh` / `memavail.sh`) made it fail each time.

Not run here: `start.sh` itself (bash 4 for `declare -A`, docker, flock, ssh). A `DRY_RUN=1 ./start.sh` on a Spark,
before and after this branch, is the end-to-end check of the off path (it stops nothing, but it is a Spark run).

## Open issues

1. The engine names (`TF_GLM_SPILL_V2`, `_QUOTA`, `_TURNS`, `_READ_BLOCKS`, `_STAGE_MIB`, `_LOAD_MS`,
   `TF_SPILL_DEFER_MS`, `TF_GLM_KV_SHARD`, `_CARRY`, `_TWINS`, `_CHECK`) are the contract's: check `port_map` against
   the readers once `glm/cache-core` / `glm/shard-host` and round 2 land (units: quota a share; `_CHECK` passed only
   when set, its default the engine's), and whether v2 needs identity variables from the recipe.
2. `ZIG_HAS` gains `spill2`, `kvshard`, `spillshard` only when the patches are in `zig/patches` (step B / round 3).
3. A `PROMPT_CACHE_GIB` above 0 can still get a budget of 0 at runtime (the pool is sized first; #187's measurement at
   the defaults): only the post-start warning; `KV_POOL_GIB`'s default is unchanged.
4. Spill files are written by root in the containers; this recipe's comment says they take `SPILL_DIR`'s owner (the
   Python engine's 0088 behaviour). Check the Zig v1/v2 tiers do the same, else port DeepSeek's `spill_fix_owner`.
5. `tf.recipe.port` covers the ported features only: a running server with another `PROMPT_CACHE_GIB` (or any other
   setting) is still "already running", as before.
6. KV sharding with the display reserve (owner planes first on it) has no recipe rule yet.
7. bash 3.2: `scripts/config.sh` (its local.sh path) and `scripts/nodes.sh` use `declare -A`; not made worse (the new
   files are 3.2-safe), not fixed either.
