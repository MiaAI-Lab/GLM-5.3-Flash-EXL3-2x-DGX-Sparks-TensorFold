# Recipe: settings, patch delivery and tools for Robert's GLM features on the Zig engine

Branch `zig-port` of Mia's GLM recipe (from `zig-engine` a6c7e08). What the recipe needs before the engine pieces land:
the prompt cache as a setting, one place that maps the ported features to the ranks, refusals where a feature cannot
run (instead of the silent no-ops before), and the parked pages in the start memory check. Step B (the patch export,
the guard on `tools/make-zig-patches.sh`, `zig/patches/README.md` and the client tools) is at the end; the fork's
defaults and the `ZIG_HAS` tokens wait for the engine patches.

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
- `port_check_extra "$@"` before the extra arguments join `SERVE_ARGS`: `--spill-gib` as an extra would turn on a tier
  writing inside the container (refused); with the tier on, `--snapshot-dir`, `--spill-highwater` (v1),
  `--spill-min-tokens` and `--spill-min-free-gib` are refused too, since `port_map`'s copies come after the extras and
  win. Every other flag passes as before (review R4).
- `port_validate "$(arg_value --parallel)" "$(arg_value --prompt-cache-gib)"` right after `arg_value`: the effective
  values, so `./start.sh --prompt-cache-gib 8` (documented) satisfies the spill check. All the gates are in it.
- After `need_workers`: `port_spill_check` (every Spark, before anything is stopped), `port_map`, `PORT_SERVE` appended
  to `SERVE_ARGS`. Before the "already running" test, so a degraded default is already reflected in the label.
- "Already running": each rank's `tf.recipe.port` label (none = `off`) must equal `port_identity`, else refuse and say
  `./start.sh restart`.
- Memory check: `mem_avail_gb "$PARKED"` here and on each worker (`declare -f mem_counting mem_avail_gb` over ssh), the
  parked GiB noted in the log line when there are any. `PARKED` is 1 only when `port_parked` says the engine's budget
  counts them (`ZIG_HAS` with `parked`, or `patches/0119-cuda-budget-parked-pages.patch`); 0 reads `free -g` as before
  (review R3).
- `zig_env` / `env_args`: the passthrough drops `PORT_MAPPED_RE` names, then adds `PORT_ENV` (re-run by the SPLIT retry).
- `RUN_ARGS`: the `/spill` mount when the tier is on; `--label tf.recipe.port=...` only when something is on.
- After the smoke test (every rank followed a request, so each has set up its prompt cache): `port_check_started` with
  each rank's `prompt cache:` / `spill tier:` / `spill: cannot open` lines (spill on only; review R5).

Off is byte-identical: with nothing set, `port_map` returns empty arrays, `port_identity` is `off` (no label), the
passthrough filter drops only `TF_GLM_KV_SHARD` / `TF_GLM_SPILL_V2` (unset by default), `--prompt-cache-gib 0` as
before, and the memory check reads `free -g` as before (no `ZIG_HAS parked`). Behaviour that does change with nothing
set: one `docker inspect` per worker when the server already runs, and a `--spill-gib` extra refused.

## How it differs

From the DeepSeek recipe's `port.sh`:
- No pool setting or derivation (`POOL`, `port_shard_pool`): GLM's pool is memory-sized (`KV_POOL_GIB`), and KV
  sharding changes what fits in it, not its geometry.
- No spill identity variables (`TF_DS_ENGINE_ID`, `TF_DS_RECIPE_PATCHES`, `TF_DS_MODEL_REVISION`, rank weight digests):
  the GLM v2 fingerprint is round 2's (glm-spill); whether the recipe must supply such names is open (below).
- No `WORKER_SPILL_DIR`: this recipe has always used one `SPILL_DIR` on every Spark (mounted at `/spill`).
- No `spill_fix_owner` (handing root-owned spill files back; why: review R2 below), no `RANK1_EXTRA_ENV`, no
  display-reserve default.
- The free-disk floor check counts the tiers' own folders as room (`du` of `tfspill-*`, `native-spill-*`, `spill-*`),
  and a shortfall it cannot measure (root's folders) only warns; DeepSeek's compares `df` alone (review R1).
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
4. Settled by the review (R2): the Zig v1 tier does not hand its files to `SPILL_DIR`'s owner (root, 0700 / 0600);
   the comments and docs now say so. Handing them back is not done yet (see the review section).
5. `tf.recipe.port` covers the ported features only: a running server with another `PROMPT_CACHE_GIB` (or any other
   setting) is still "already running", as before.
6. KV sharding with the display reserve (owner planes first on it) has no recipe rule yet.
7. bash 3.2: `scripts/config.sh` (its local.sh path) and `scripts/nodes.sh` use `declare -A`; not made worse (the new
   files are 3.2-safe), not fixed either.

# Step B: patch delivery and client tools

How the port's engine patches reach `zig/patches` (none yet: the engine pieces are being written on `glm/*` in
`~/GitHub/tensorfold-glm`), a guard on the tool that would delete them, the patches' README, and the DeepSeek port's
client tools adapted to GLM.

## What was ported from where

| Piece | From | Here |
| --- | --- | --- |
| `scripts/port-patches.sh` | R's `scripts/port-patches.sh` (bdfc9e4): build the patches in a temporary folder, check them on a clean `TF_REF` with the hand-written patches first, write only then; `newreadme` (rows marked `<!-- port-patches -->`); `PORT_PATCHES_LIB=1` for the tests | GLM pins (`ZIG_TF_REF`, read with leading whitespace: it sits inside config.sh's zig block), engine `~/GitHub/tensorfold-glm`, base `glm/zig-engine-base`, branch `glm/integration`; per-commit patches by default; GLM-style headers; the kernel set excluded and guarded (below) |
| `scripts/port_patches_test.sh` | new (R tests only `newreadme`) | 13 cases against the engine repository in a throwaway `git clone --shared` (no commit on its branches): base alone, one commit, two, a merge, squash, five refusals, writes on a copy of the recipe |
| `scripts/port_test.sh` | R's `newreadme` case | `newreadme` (prose kept, idempotent, rows after the last hand-written row), `slug`, `patchlike` |
| `tools/make-zig-patches.sh` guard | new | refuses while `zig/patches` holds a patch numbered above 0005 (it `rm -f`s every patch, then writes 0001-0005); `--force` (first argument) keeps the old behaviour |
| `zig/patches/README.md` | new; R's README for the generated-row convention | a row per 0001-0013 (from their headers, the recipe's log and its merged PRs: #170, #174, #181, #190, #192/#194, #191, #202), how 0014+ are made, ordering rules |
| `tools/checklib.py`, `exact.py`, `spill_check.py`, `resume_bench.py`, `prefix_fanout.py`, `selftest.py`, `fake_server.py`, `test_checklib_credentials.py` | R's `tools/` (bdfc9e4) | adapted (below); not taken: `kv_shard_check_e2e.py`, `pair_check.py`, `gpu_validate.sh`, `bench.py`, `smoke.py`, `validate.py` (DeepSeek-engine checks or runbooks). No clash with this recipe's tools (`client.py`, `needle.py`, `prompt_reuse.py`, ... stay as they are) |

## port-patches.sh: how it differs from DeepSeek's

- `--mode commits` is the default (R: squash): one patch a first-parent commit of `BASE..BRANCH`, named
  `NNNN-zig-<slug>.patch` (`--name REV=SLUG`, else the subject's slug), numbered after the last hand-written patch. A
  merge is one patch (its diff against its first parent) whose header lists the commits it brings in. Squash stays.
- The header is the commit's message as written (GLM's 0012/0013 style: plain text, not R's `# ` lines), after a first
  line `# port-patches.sh: <branch> @ <sha> over <base> @ <sha> (<mode> <from>..<to>)` that marks it as exported, then a
  `Credits:` line (the commits' authors, `Robert Wojciechowski (GitHub @wojo, X @rwojo)` for Robert; BadAd84's
  block store design, #118, when the patch's paths or its message say spill; none added when the message has its own
  `Credits:` line) and GLM's `Apply from the checkout root with: git apply <file>` line. `git apply` reads from the
  first line that looks like a patch, so a message with such a line (`diff `, `--- `, `+++ `, `@@ `, ...) is refused,
  not silently mangled.
- Excluded from the patches and the comparisons: `zig/tests/cuda/glm_p1/aot` (the baseline carries the captured kernel
  set there; it ships as `zig/aot/glm-aot-sm121.tar.xz`) and `docs/glm-port/` (these notes). A branch that changes the
  kernel set is refused (checked after the base, so a wrong base is named as such): the image would lack those kernels.
- Nothing to export is not an error: `--branch glm/zig-engine-base` checks the base alone and exits 0, and so does a
  range that touches only the excluded paths; earlier exports are then left as they are (said).
- README rows: `| patch | By | What it does |` (author markdown, subject, engine ref and sha), where R generated its
  switch and default columns from `scripts/port.sh`: GLM's patches carry their switches and defaults in the header.

## The tools: how they differ from DeepSeek's

- `checklib.py`: `DEFAULT_MODEL` `GLM-5.3-Flash-EXL3` (`SERVED_NAME`); the API key from `API_KEY`, else the first of
  `TENSORFOLD_API_KEY`'s comma-separated keys (what this recipe passes to the server; it writes no key file), else
  `--api-key-file` (still no default path, no `API_KEY_FILE`); `default_context()` (`CONTEXT`, else 131,072, the
  Zig default), `fit(context, reply)` (90 % of the window less the reply), `capped`, `over_window`; `SPILL_FIELDS`
  without the picture points' two; no DeepSeek KV-shard field lists (`kv_shard{...}` and `kv_shard_*` are flattened as
  they come); `SPILL_MARKER` `.glm53-tensorfold-spill` (no marker is written yet, open issue 6).
- `spill_check.py`: no picture items or checks (GLM's lane host keeps no picture prompts); GLM's store numbers (6,204 B
  a position, a 64-token keep grid with tails, `estimate` counts whole blocks and the tail); the small state is a 1 MiB
  lower bound (low is the floor phase's safe side); `record --context`, the long items (`--cancel-tokens`,
  `--kill-tokens`, `--floor-tokens`: R's 500,000 / 200,000 / 200,000) default to the most the window allows, and any
  item past it is refused before a request.
- `exact.py`: checklib's key rule and headers (R read `~/.local/state/dsv41-tensorfold/api-key` by default); `--context`
  leaves out cold cases past the window, listed in `meta.over_window`, which `compare` notes instead of calling a
  mismatch (at 131,072 every case fits).
- `resume_bench.py`: sizes 32,000 / 64,000 / 100,000 (R: 100,000-250,000), `--context`, sizes past the window refused.
- `prefix_fanout.py`: `--grid 64` (GLM's keep grid; R 2048), `--context`; its docstring says what it proves here: the
  Zig engine fills one prompt at a time, so a shared prefix is computed once, which is why the prefix wait is not ported.
- `fake_server.py`: no KV-shard simulation, `/fake/kill-rank1` or picture points; GLM's name and a 131,072 window.
- `selftest.py`: `compile`, `creds` (runs `test_checklib_credentials.py`), `kinds`, `counters`, `exact` (plus a record
  at a smaller window), `spill` (no pictures), `resume` (new: seed, restart, resume, report, and `restore-wrong`
  caught), `prefix` (the fake's 2048 grid); no `kv`, `pair` or `runbook`.

## Tests (step B)

- `/bin/bash scripts/port_test.sh`: 96 of 96 (bash 3.2.57, this Mac).
- `/bin/bash scripts/port_patches_test.sh`: 13 of 13 (~18 s, against `~/GitHub/tensorfold-glm`); a mutation pass of
  `port-patches.sh` (the tree comparison always true, the hand-written patches not applied, earlier exports kept, no
  credits line, no kernel-set refusal, numbering from 0) failed it each time.
- `scripts/port-patches.sh --dry-run --branch glm/zig-engine-base`: `ZIG_TF_REF` + 0001-0013 = `0db3e2f6` outside the
  kernel set (583 files there) and the notes, nothing to export. In a throwaway clone with one trivial commit on the
  base: `0014-zig-glm-trivial.patch` written, applies, gives the branch's tree.
- `tools/make-zig-patches.sh`: refuses on this tree (0006-0013 listed), nothing changed; in a temporary copy with only
  0001-0005 and with `--force` it goes past the guard (both then stop at the missing checkout).
- `python3 -I tools/selftest.py`: PASS, 83 expectations (~85 s, Python 3.15); `tools/test_checklib_credentials.py`
  7 of 7.
- No engine code changed in this step: no cross-build or Zig host tests.

## Which checks need the Sparks

On any machine: `scripts/port_test.sh`, `scripts/port_patches_test.sh` (needs the engine repository),
`scripts/port-patches.sh --dry-run`, `tools/selftest.py`, `tools/test_checklib_credentials.py`, `exact.py compare`,
`spill_check.py estimate`, `resume_bench.py report`. On the Sparks (a GPU run, Robert's go): `exact.py record` on the
image before and after a change; `prefix_fanout.py --expect-one-fill` on today's image with `PROMPT_CACHE_GIB` above 0
(the check that the prefix wait is not needed); `spill_check.py record / run / drop` and `resume_bench.py seed /
resume` once the spill v2 patches are in `zig/patches` (`SPILL_V2=1`, `ZIG_HAS` with `spill2`).

## Open issues (step B)

1. Counter names. The tools read the DeepSeek port's cache-tier names (`prompt_cache.memory_hits`,
   `disk_hits_committed`, `restored_tokens`, `entries`, `restores_*`, `parked`, and `spill` as an array of per-rank
   objects). The GLM Zig server today reports `prompt_cache{hits, misses, evicted, restored_from_disk}`, `kept_*` at the
   top and v1's `spill{entries, bytes, writes, write_bytes, reads, read_bytes, skipped, failed}` as one object
   (`status_routes.zig keptFields`). `glm/cache-core` decides the v2 names: either it reports cache-tier's or
   `checklib` maps them. Until then `spill_check.py` and `resume_bench.py` have nothing to read; `exact.py` and
   `prefix_fanout.py` work against today's server (`tensorfold.token_sha`, `usage...cached_tokens`, `/health live`
   are in v1.0.7's server).
2. `spill_check.py`'s small-state size and keep window are estimates until round 2's packer fixes them.
3. `TOKENS_PER_WORD` (1.15) is DeepSeek's tokenizer's; GLM's is not measured (`fit` leaves 10 % of the window).
4. The engine commits' messages are the patches' headers: round 3 has to write them as such (what, switches and
   defaults, measured, credits) and give `--name`s. Exported from `glm/integration`, each merge of a piece is one patch;
   a series branch with one commit a feature (as DeepSeek's `port/series-370daf0`) gives cleaner patches.
5. Numbering: 0014 and later assume no one else takes 0014 first (open PR #186 already collides on 0008); a new
   hand-written patch moves the export after it on the next run (tested), but the numbers should be agreed with Mia and
   hecisaza.
6. `SPILL_MARKER`: this recipe writes no marker at the spill root (step A's issue 4, `spill_fix_owner`).
7. `tools/make-zig-patches.sh` needs bash 4 (`mapfile`) and GNU tar: only its guard was run here.

# Review (round 1): findings and outcomes

The independent review of a2271bf..ea0d00d found one major issue and five smaller ones; the fixes are one commit on
`zig-port` (after ea0d00d). Tests after it, on this Mac (bash 3.2.57): `scripts/port_test.sh` 108 of 108 (96 before;
written first, 15 cases failed on the unfixed code, each for its finding's reason, and the 2 judge cases added after
use the new probe line; a mutation pass of 8 single-line reversals of the fixes failed it each time),
`scripts/port_patches_test.sh` 13 of 13, `bash -n` of every script. No engine code changed and the recipe has no
`build.zig`: no cross-build. `tools/` did not change: `tools/selftest.py` not re-run.

| Finding | Outcome |
| --- | --- |
| R1 (major): the free-disk floor check counts the tier's own files as used, so a full tier, or an old build's folder the engine prunes at open, refuses the start (or turns a fork's default tier off for good) | **Fixed.** `spill_dir_probe` measures the tiers' own folders in `SPILL_DIR` with `du -sk` as you (`tfspill-*` Zig v1, `native-spill-*` v2/core-spill, `spill-*` Python 0088; not symlinks) and says whether one could not be read; `port_spill_judge` adds them to the free space for the floor and for the cap warning. A folder it cannot read (the Zig tier's are root's 0700, so on Zig every start after the first) makes a shortfall a warning, not a refusal, and skips the cap warning. The count is generous on purpose (0088 and core/spill keep the newest other build): a start the tier could run is worse refused than let through, and the tier reports its skipped writes. A running server's folders are unreadable, so `./start.sh` on it now says "nothing to do" again. |
| R2 (minor): comments and docs say the spill files are yours | **Partly fixed.** `port.sh`, `start.sh`, README, `docs/zig-engine.md`, CHANGELOG and the "Spill tier:" log line (ENGINE=zig) now say the Zig tier's folders and files are root's (0700 / 0600; removing them takes `sudo`, `tools/spill_check.py --sudo`), the Python 0088 tier's SPILL_DIR's owner's. **Not ported:** DeepSeek's `spill_fix_owner`. It runs a root container with `CAP_CHOWN` on every start and stop; the better fix is the one 0088 made in the engine itself (each folder and file given the spill root's owner: no privileged container), and the files that matter from round 2 on are the v2 tier's, which the glm-spill piece writes. Flagged for glm-spill; if it declines, port `spill_fix_owner` (maintainer-reviewed on Mia's DeepSeek repo, #30) in round 3 together with the marker (`checklib.SPILL_MARKER`, step B issue 6). With the files handed back, R1's count becomes exact on Zig too. |
| R3 (minor): the memory check counts parked pages that neither shipped engine budgets yet | **Fixed.** `mem_avail_gb <parked 0|1>`: 0 (the default) is `free -g`'s available column, exactly as before the port; 1 is `mem_counting`. `start.sh` passes 1 only when `port_parked` says the engine counts them: `ZIG_HAS` with `parked` (added to the default when the 0119 port is exported into `zig/patches`), or `patches/0119-cuda-budget-parked-pages.patch` for ENGINE=python (PR #149). Comments, `docs/zig-engine.md` and the CHANGELOG corrected. f4db50d's message still says the engine counts them; this commit's message corrects it (history not rewritten). |
| R4 (minor): every `--spill-*` / `--snapshot-dir` extra argument refused | **Fixed.** Refused: `--spill-gib` always (it would turn on a tier with no checked or mounted folder); with the tier on, `--snapshot-dir`, `--spill-highwater` (v1 only), `--spill-min-tokens`, `--spill-min-free-gib`, which `port_map` passes after the extras (so the extra would be silently ignored), naming the setting to use. Everything else passes as before: `--spill-keep-builds` (0088's, no setting), `--snapshot-dir` with the tier off (TensorFold's own snapshot folder). `=`-forms handled. |
| R5 (nit): only rank 0's log is read after the start | **Fixed in the recipe, flagged for the engine.** `port_check_started` takes every rank's `prompt cache:` / `spill tier:` / `spill: cannot open` lines (workers over ssh; `(unread)` when ssh fails) and warns for each rank with no `spill tier:` line, quoting the engine's reason. It now runs after the smoke test: `cacheFor` is the last step of each rank's setup, and rank 0 can answer `/v1/models` before rank 1 has reached it; the smoke test needs every rank. The engine gap stays: `cacheFor` opens each rank's tier from its own budget with no agreement, and a rank without one answers a disk restore with `error.NoSpillTier`. The fix (agree the budget across ranks, or open the tier on every rank when rank 0's budget is above 0) belongs to cache-core / glm-spill. |
| R6 (nit): "has no the spill v2 port" | **Fixed:** "lacks the spill v2 port" (and "patches/ lack ..."); the tests' patterns follow. |

## Round 3: the port in `zig/patches` (`0014`-`0023`)

The engine's per-feature series (`glm/series` in `~/GitHub/tensorfold-glm`, ten commits on `glm/zig-engine-base`, its
last tree `glm/integration`'s after round 3's merge outside `docs/glm-port`; the engine's own notes:
`docs/glm-port/series.md` there) exported with `scripts/port-patches.sh` as `0014-zig-glm-parked-pages` ...
`0023-zig-glm-spill-kv-shard`. What changed here with it:

| Where | Change |
| --- | --- |
| `zig/patches/0014`-`0023`, `README.md` | the exported patches and their rows; the port's section says what each brings, its switch and default, what was measured on the pair (g1/g2 only), and that `0014`-`0016` are also offered alone |
| `start.sh` | `ZIG_HAS`'s default is `mtp displaykv spill2 kvshard spillshard parked`: `SPILL_V2=1` and `KV_SHARD=1` start, the memory check counts the parked pages |
| `scripts/port.sh` | `port_missing` says a token is missing from `ZIG_HAS` (not that the port is absent); spill v2 beyond two ranks is refused before anything stops (a fork's default degrades: v1 for an explicit `SPILL_GIB`, else off), as the engine refuses it at start; the probe counts the v2 tier's own `glm-spill-<hash>` folders as room (core/spill under GLM's prefix: the probe knew only `native-spill-`, so a full v2 tier counted as used and could refuse the start) |
| `scripts/port-patches.sh` | the default branch is `glm/series`; a commit with no `--name` keeps the name of the patch already exported from it (so `--dry-run` with no arguments reproduces `0014`-`0023`); a message with its own `Credits:` paragraph is credited as it says (the series' KV-shard patch touches spill v2's tests, which made the file-name rule credit BadAd84's design) |
| `scripts/port_test.sh` | the cases run with the new default `ZIG_HAS`; the ones about a missing port name an image without it (`mtp displaykv`); new: the default names a feature exactly when `zig/patches` carries it (by content: `parkedBytes`, `cuda_spill_tier.zig`, `gpu_paths = true`, `KvShardHomeFull`), spill v2 at TP=3, the `glm-spill-` probe |
| `scripts/port_patches_test.sh` | starts from the hand-written patches (the copy drops this recipe's exports); new: the credits paragraph, a re-export keeping names |
| `docs/zig-engine.md`, `README.md`, `CHANGELOG.md`, `CREDITS.md`, `NOTICE` | the settings' rows as implemented (`SPILL_V2`, `KV_SHARD`, `PROMPT_CACHE_GIB` with the GPU finding, parked pages, `TENSORFOLD_PREPARE_FLOOR_GIB`, continued chats), the credits for `0014`-`0023` |

Robert's fork defaults are a further branch, `fork-main` (one commit, "Robert's defaults"): see its own section there.

### Re-export after merge round 3's follow-ups (2026-10-23)

The engine fixed two of round 3's leftovers (`docs/glm-port/merge-r3.md` there, "Follow-ups": `KvShardHomeFull` is
one of the prompt cache's `point_errors`, so it is never deferred; `Tier.VTable.check`, asked before the store makes
room, so a restore refused for its point evicts no kept state) and folded them into the series: `0019`, `0021` and
`0023` changed, `0020` and `0022` were re-applied unchanged, `0014`-`0018` are untouched. `scripts/port-patches.sh`
re-exported `glm/series` (3181fe20) with its defaults, plus `--name` for the five rewritten commits (`f139cada` to
`core-cache-tier`, `8455b2ce` to `glm-kv-shard-host`, `68854c0f` to `glm-spill-blocks`, `d0b25f0c` to
`glm-kv-shard`, `dbe61751` to `glm-spill-kv-shard`): a re-export keeps a patch's name only for the commit it came
from, and a rewritten commit is a new one. The files and their names are as before; every header's `glm/series` sha
moved, and `0019`, `0021` and `0023` carry the fixes. Nothing else here changed (no setting, no default; `ZIG_HAS`
as before). Tests: `scripts/port_test.sh` 113/113, `scripts/port_patches_test.sh` 15/15, `tools/selftest.py` 83/83.
