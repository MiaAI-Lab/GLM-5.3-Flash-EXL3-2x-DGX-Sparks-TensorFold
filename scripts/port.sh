# Robert's (wojo's) GLM features on the Zig engine: their settings, their checks, and the ONE place they map to the
# engine's flags and variables (port_map). The features are the spill tier (v1: cuda_spill.zig, already in zig/patches;
# v2: his GLM Python fork's patches 0088/0110/0111), KV sharding (0118) and the parked pages (0119: the start memory
# check, scripts/memavail.sh). Sourced by scripts/config.sh after the serving settings; start.sh calls the functions.
# Every feature is OFF by default (PORT_SPILL_DEFAULT_GIB, PORT_SPILL_V2_DEFAULT, PORT_SHARD_DEFAULT below): with them
# off nothing below adds a flag, a variable, a mount or a label, and every rank runs as before. A fork may turn a default
# on by changing those values (and PORT_SPILL_QUOTA_DEFAULT); a default that cannot run here is turned off with a
# warning, while the same setting given explicitly refuses.
# The Zig engine serves the v2 tier and KV sharding from zig/patches/0017-0023 (the port's engine series); ZIG_HAS
# (start.sh, whose default lists them) names them: spill2, kvshard, and spillshard for the two together. Round 4's
# patches 0024-0029 add mtpcache (kept prompt states with MTP drafts, 0025), fill (sliced prompt fills, FILL, 0026) and
# pictures (prompts with pictures kept, resumed and spilled, 0027-0028). An image without them (a narrower ZIG_HAS)
# refuses an explicit setting and names the missing patch. ENGINE=python (patches/)
# has the v1 tier (0088) only. Likewise the start memory check counts the parked pages only where the engine does
# (scripts/memavail.sh's parked_counted: ZIG_HAS parked, zig/patches/0014-zig-parked-pages; or patches/'s 0119).
# Kept to bash 3.2 syntax (with scripts/memavail.sh) so that scripts/port_test.sh runs on a Mac too. scripts/config.sh
# and scripts/nodes.sh use `declare -A`, which bash 3.2 lacks; the tests source this file alone.

# ---------------------------------------------------------------- the settings
# Spill tier: kept prompt states written to each Spark's local disk and read back instead of a new prefill, also after
# a clean restart (README: Spill tier). ENGINE=zig spills the prompt cache's kept states, so it needs PROMPT_CACHE_GIB
# above 0 (scripts/config.sh) and DFlash2 drafts (with MTP drafts the kept states also hold the MTP head's state, which
# no spill tier stores yet: they stay in memory, ZIG_HAS mtpcache).
#   SPILL_GIB          the cap a Spark (GiB; 0: off, the default)
#   SPILL_DIR          the same absolute path on every Spark, mounted at /spill: a folder you can write, on a local disk,
#                      outside HF_CACHE and outside any NFS export (WORKER_WEIGHTS=nfs serves HF_CACHE to the workers);
#                      start.sh checks it on every Spark and creates it (as you) when missing. The ranks run as root: the
#                      Zig engine's v1 tier (cuda_spill.zig) writes its tfspill-<hash> folders and files as root, readable
#                      by root only (0700 / 0600), so removing them takes sudo; its v2 tier
#                      hands its glm-spill-<hash> folders to SPILL_DIR's owner (the files in them stay root's); the
#                      Python engine's (0088) hands its files to SPILL_DIR's owner
#   SPILL_MIN_TOKENS   shorter kept states are not written (8192)
#   SPILL_MIN_FREE_GIB free disk the tier leaves on SPILL_DIR's file system: past it writes are refused (50)
#   SPILL_FLUSH_S      what a clean stop gives the tier to write what still waits (60; the ranks' TF_SPILL_FLUSH_S);
#                      STOP_TIMEOUT grows with it
#   SPILL_HIGHWATER    v1 only: past this fraction of the pool, the states eviction would take next are written early
#                      (0.70)
#   SPILL_V2=1         (TF_GLM_SPILL_V2) the v2 tier: shared 2,048-token blocks written when a state is kept, read back
#                      after a crash too; 0 (the default): v1, which stays the rollback. ENGINE=zig with ZIG_HAS spill2
#   v2 only, the Python fork's names (each also taken from its TF_GLM_ name when only that one is set):
#   SPILL_QUOTA        (TF_GLM_SPILL_QUOTA) the share of SPILL_GIB one conversation may hold before its own old turns
#                      and forks go first, 0 to 1 (0, the default: none; the fork runs 0.25)
#   SPILL_TURNS        (TF_GLM_SPILL_TURNS) the newest states kept on a conversation's path (4; 0: every turn)
#   TF_GLM_SPILL_READ_BLOCKS, TF_GLM_SPILL_STAGE_MIB, TF_GLM_SPILL_LOAD_MS, TF_SPILL_DEFER_MS: the engine's own tuning,
#                      passed as set (unset: the engine's defaults)
# KV sharding (0118): each conversation's latent cache stored once, on the Spark that owns it, instead of on both
# (~1.9x the pool's tokens in the same memory on the Python engine; 1.83x measured on the Zig engine: 1,511,424 tokens
# against 825,344 at KV_POOL_GIB=4). Two Sparks, PARALLEL above 1, DFlash2 drafts; ENGINE=zig with ZIG_HAS kvshard
# (with the spill tier: v2 and ZIG_HAS spillshard). TF_GLM_MULTI_LONE=1 only warns: it does nothing on the Zig engine.
#   KV_SHARD=1               (TF_GLM_KV_SHARD) on; 0 (the default) off. A fork's default of 1 applies only where the
#                            Python fork turned it on (TP=2, PARALLEL above 1, TF_GLM_MULTI_LONE=0), silently off elsewhere
#   TF_GLM_KV_SHARD_CARRY    1 (default): a kept conversation whose part is full is carried into the other part; 0: not
#   TF_GLM_KV_SHARD_TWINS    1 (default): a prompt that runs from 0 borrows the other part's free rows; 0: it fetches
#   TF_GLM_KV_SHARD_CHECK    decode rounds between the ranks' placement digest checks (unset: the engine's default)
# Sliced prompt fills (zig/patches/0026): FILL=1 (TF_GLM_FILL on every rank) fills a prompt a 2048-row chunk a step
# between decode rounds, so a long prompt no longer stalls the other streams for its whole pass; 0 (the default): whole
# passes, as before. ENGINE=zig with ZIG_HAS fill; refused with DRAFTER=mtp (one MTP head serves every stream).
# scripts/config.sh sets its default; the line below is for the tests, which source this file alone.
FILL="${FILL:-${TF_GLM_FILL:-0}}"
# The prefix wait (the Python fork's 0110, on the Zig engine's lane host from glm/prefix-wait; see below for the patch):
# with FILL=1, requests that arrive together on one system prompt all start filling at once, and each computed it (g6,
# prefix_fanout: 31,488 / 85,440 / 199,360 duplicate tokens at 2 / 4 / 8 at once on a 28k-token prompt). With the wait a
# request whose prompt shares a cut another prompt is still filling waits for that prompt to keep it, holding no lane,
# then resumes from it: the prefix is computed once. Whole passes (FILL=0) never need it: each ends before the next
# request is admitted, which finds its cut kept.
#   PREFIX_WAIT=1           (TF_GLM_PREFIX_WAIT) on, the default; 0 off. Passed only with FILL=1, and it needs the prompt
#                           cache (PROMPT_CACHE_GIB above 0: the cut is a kept prompt state)
#   TF_GLM_PREFIX_WAIT_MS   the wait's bound in ms (unset: the engine's 60000; 0 off), passed as set with FILL=1
# An image whose engine predates the wait ignores both (it reads neither): FILL=1 then runs as before.
PORT_PW_EXPLICIT=0; [[ -z "${PREFIX_WAIT:-}${TF_GLM_PREFIX_WAIT:-}" ]] || PORT_PW_EXPLICIT=1
PREFIX_WAIT="${PREFIX_WAIT:-${TF_GLM_PREFIX_WAIT:-1}}"
PORT_SPILL_DEFAULT_GIB=0   # SPILL_GIB when it is not set (0: off). Robert's fork: 64
PORT_SPILL_V2_DEFAULT=0    # SPILL_V2 when it is not set (0: v1). Robert's fork: 1
PORT_SHARD_DEFAULT=0       # KV_SHARD when it is not set, where it can run (0: off). Robert's fork: 1
PORT_SPILL_QUOTA_DEFAULT=0 # SPILL_QUOTA when it is not set (v2 only; 0: none). Robert's fork: 0.25
# Whether a switch was set (the environment, scripts/local.sh or .env) or is the recipe's default: a default that cannot
# run degrades, an explicit one refuses.
PORT_SPILL_EXPLICIT=0; [[ -z "${SPILL_GIB:-}" ]] || PORT_SPILL_EXPLICIT=1
PORT_V2_EXPLICIT=0; [[ -z "${SPILL_V2:-}${TF_GLM_SPILL_V2:-}" ]] || PORT_V2_EXPLICIT=1
PORT_SHARD_EXPLICIT=0; [[ -z "${KV_SHARD:-}${TF_GLM_KV_SHARD:-}" ]] || PORT_SHARD_EXPLICIT=1
# Whether KV_POOL_GIB was set: scripts/config.sh records it before it fills in the default (port_pool_check)
PORT_POOL_EXPLICIT=${PORT_POOL_EXPLICIT:-0}
# The v2 and KV-shard settings that were set: they do nothing without their feature (port_validate warns)
PORT_V2_SET=""
for _n in SPILL_QUOTA TF_GLM_SPILL_QUOTA SPILL_TURNS TF_GLM_SPILL_TURNS TF_GLM_SPILL_READ_BLOCKS TF_GLM_SPILL_STAGE_MIB \
          TF_GLM_SPILL_LOAD_MS TF_SPILL_DEFER_MS; do [[ -z "${!_n:-}" ]] || PORT_V2_SET+=" $_n"; done
PORT_SHARD_SET=""
for _n in TF_GLM_KV_SHARD_CARRY TF_GLM_KV_SHARD_TWINS TF_GLM_KV_SHARD_CHECK; do [[ -z "${!_n:-}" ]] || PORT_SHARD_SET+=" $_n"; done
unset _n
SPILL_GIB="${SPILL_GIB:-$PORT_SPILL_DEFAULT_GIB}"
SPILL_DIR="${SPILL_DIR:-$HOME/.cache/tensorfold-spill}"
SPILL_HIGHWATER="${SPILL_HIGHWATER:-0.70}"
SPILL_MIN_TOKENS="${SPILL_MIN_TOKENS:-8192}"
SPILL_MIN_FREE_GIB="${SPILL_MIN_FREE_GIB:-50}"
SPILL_FLUSH_S="${SPILL_FLUSH_S:-60}"
SPILL_V2="${SPILL_V2:-${TF_GLM_SPILL_V2:-$PORT_SPILL_V2_DEFAULT}}"
SPILL_QUOTA="${SPILL_QUOTA:-${TF_GLM_SPILL_QUOTA:-$PORT_SPILL_QUOTA_DEFAULT}}"
SPILL_TURNS="${SPILL_TURNS:-${TF_GLM_SPILL_TURNS:-4}}"
KV_SHARD="${KV_SHARD:-${TF_GLM_KV_SHARD:-$PORT_SHARD_DEFAULT}}"
# Where each rank sees its spill directory.
PORT_SPILL_MOUNT=/spill

port_num()  { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
port_gt0()  { awk -v x="$1" 'BEGIN {exit !(x + 0 > 0)}'; }
port_spill_on() { port_num "$SPILL_GIB" && port_gt0 "$SPILL_GIB"; }
port_v2_on() { port_spill_on && [[ "$SPILL_V2" == 1 ]]; }
# port_has <feature>: the Zig engine in this image serves it (ZIG_HAS, start.sh: spill2, kvshard, spillshard, parked,
# mtpcache, fill, pictures)
port_has() { [[ "$ENGINE" == zig && " ${ZIG_HAS:-} " == *" $1 "* ]]; }
# port_missing <feature>: what is missing for it, for a refusal or a warning
port_missing() {
  local what
  case "$1" in
    spill2)     what="the spill v2 port (Robert's GLM patches 0110/0111)" ;;
    kvshard)    what="the KV-shard port (Robert's GLM patch 0118)" ;;
    spillshard) what="the port of spill v2 and KV sharding together" ;;
    fill)       what="the sliced prompt fills (zig/patches/0026)" ;;
    pictures)   what="the kept prompts with pictures (zig/patches/0027-0028)" ;;
  esac
  if [[ "$ENGINE" == zig ]]; then
    echo "this image's tensorfold-native lacks $what: ZIG_HAS does not name it (zig/patches/0017-0030 bring spill2, kvshard, spillshard, mtpcache, fill, pictures and prefixwait, which start.sh's default ZIG_HAS lists; now: ${ZIG_HAS:-empty})"
  else
    echo "the Python engine's patches/ lack $what (Robert's fork, wojo/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, has them for ENGINE=python)"
  fi
}
# A clean stop writes what still waits within SPILL_FLUSH_S: give the ranks that long and 30 s (the stop without the
# tier), as before the port. Unchanged with the tier off.
PORT_STOP_TIMEOUT_SET=0
if port_spill_on && [[ -z "${STOP_TIMEOUT:-}" && "$SPILL_FLUSH_S" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  STOP_TIMEOUT=$(( ${SPILL_FLUSH_S%.*} + 30 )); PORT_STOP_TIMEOUT_SET=1
fi

# ---------------------------------------------------------------- turning a default off
# port_spill_default_off <reason>: the spill tier was on by default only and cannot run here: say so loudly and start
# without it on every rank (port_map then gives none of them a spill flag, variable or mount)
port_spill_default_off() {
  warn "SPILL TIER OFF for this start: $1. The spill tier is on by default here (SPILL_GIB=$SPILL_GIB a Spark); the server starts without it (kept prompt states stay in memory only, and are gone after a restart). Set SPILL_GIB=0 to silence this, or SPILL_GIB explicitly to make it a refusal"
  SPILL_GIB=0
  if (( PORT_STOP_TIMEOUT_SET )); then unset STOP_TIMEOUT; PORT_STOP_TIMEOUT_SET=0; fi
}
# port_v2_default_off <reason>: SPILL_V2=1 was the default only: an explicit SPILL_GIB keeps the v1 tier, a default one
# goes off with it
port_v2_default_off() {
  SPILL_V2=0
  if (( PORT_SPILL_EXPLICIT )); then
    warn "SPILL V2 OFF for this start: $1. SPILL_V2=1 is the default here; the tier runs as v1 (SPILL_GIB=$SPILL_GIB). Set SPILL_V2=0 to silence this, or SPILL_V2=1 explicitly to make it a refusal"
  else
    port_spill_default_off "$1 (spill v2 is the default, and v1 is not turned on in its place)"
  fi
}
port_shard_default_off() {
  warn "KV SHARDING OFF for this start: $1. KV_SHARD=1 is the default here; the latent cache is replicated on every Spark, as before. Set KV_SHARD=0 to silence this, or KV_SHARD=1 explicitly to make it a refusal"
  KV_SHARD=0
}
# port_spill_refuse <reason>: a spill check failed. Explicit SPILL_GIB: refuse (die). Default: PORT_SPILL_WHY keeps the
# reason and it returns 1 (the caller then turns the tier off).
PORT_SPILL_WHY=""
port_spill_refuse() { (( PORT_SPILL_EXPLICIT )) || { PORT_SPILL_WHY=$1; return 1; }; die "$1"; }

# port_pool_check <prompt cache GiB>: ENGINE=zig with DFlash2 (or MTP drafts, kept where ZIG_HAS names mtpcache), PROMPT_CACHE_GIB or
# --prompt-cache-gib above 0 while KV_POOL_GIB was not set. The engine sizes the KV pool first and gives the kept prompt
# states only what is left above MEMORY_RESERVE_GIB: at the default pool that is nothing (the pair, TP=2 at 12.5:
# "prompt cache: off (budget 0)"), so no state would be kept. Warns; the pool stays as it is. The pair's sizing:
# KV_POOL_GIB=4 with PROMPT_CACHE_GIB=8, a 6.2 GiB budget (g1b), 5.1 GiB with spill v2 on (g2: its 1,066 MiB of pinned
# staging is taken before the pool).
port_pool_check() {
  [[ "$ENGINE" == zig ]] && { [[ "$DRAFTER" == dflash2 ]] || port_has mtpcache; } && port_gt0 "$1" && (( ! PORT_POOL_EXPLICIT )) || return 0
  warn "PROMPT_CACHE_GIB=$1 with KV_POOL_GIB at its default ($KV_POOL_GIB): the engine sizes the KV pool first and gives the kept prompt states only what is left above MEMORY_RESERVE_GIB, and at the default pool that left nothing on the pair (KV_POOL_GIB=12.5: \"prompt cache: off (budget 0)\"), so no prompt state would be kept. Set KV_POOL_GIB lower with it; the pair ran KV_POOL_GIB=4 with PROMPT_CACHE_GIB=8: a 6.2 GiB budget for kept states, 5.1 GiB with spill v2 on (its 1,066 MiB of pinned staging is taken before the pool)"
}

# port_vision_check <prompt cache GiB>: ENGINE=zig with VISION=1 (the default at TP=2) and PROMPT_CACHE_GIB or
# --prompt-cache-gib above 0. Rank 0's vision tower (~1.05 GiB of bf16 weights, and its workspace for the largest pass,
# ~0.4 GiB, which the engine takes at start) is loaded before the pool and the kept-state budget are sized, so the
# budget for kept prompt states on rank 0 is ~1.4 GiB smaller than with VISION=0 (estimated from the tower's sizes,
# not measured on the pair). Warns; nothing changes.
# An image whose ZIG_HAS does not name pictures keeps no prompt with pictures (only text prompts): said too.
port_vision_check() {
  [[ "$ENGINE" == zig && "${VISION:-0}" == 1 ]] && port_gt0 "$1" || return 0
  warn "VISION=1 with PROMPT_CACHE_GIB=$1: rank 0's vision tower (~1.05 GiB of weights and ~0.4 GiB of workspace, taken at start) comes out of the memory the kept prompt states are given, so their budget on rank 0 is ~1.4 GiB smaller (estimated); VISION=0 keeps it for text"
  port_has pictures ||
    warn "VISION=1 with PROMPT_CACHE_GIB=$1: $(port_missing pictures), so a prompt with pictures is never kept or resumed (text prompts are)"
}

# ---------------------------------------------------------------- checks (no ssh, nothing changed)
# port_validate <parallel> <prompt cache GiB>: every setting's form, and the combinations the engine cannot run, before
# anything is stopped. The two arguments are the effective --parallel and --prompt-cache-gib (the command line's extra
# arguments win: start.sh's arg_value); default PARALLEL and PROMPT_CACHE_GIB.
port_validate() {
  local par=${1:-$PARALLEL} cache=${2:-${PROMPT_CACHE_GIB:-0}} x why
  # the switches' aliases: the recipe's name wins
  for x in "KV_SHARD TF_GLM_KV_SHARD" "FILL TF_GLM_FILL" "PREFIX_WAIT TF_GLM_PREFIX_WAIT" "SPILL_V2 TF_GLM_SPILL_V2" "SPILL_QUOTA TF_GLM_SPILL_QUOTA" "SPILL_TURNS TF_GLM_SPILL_TURNS"; do
    set -- $x
    [[ -z "${!1:-}" || -z "${!2:-}" || "${!1}" == "${!2}" ]] || warn "$2=${!2} is overridden by $1=${!1} (the recipe's name wins)"
  done
  [[ "$KV_SHARD" =~ ^[01]$ ]] || die "KV_SHARD (TF_GLM_KV_SHARD) is 0 or 1, not $KV_SHARD"
  [[ "$FILL" =~ ^[01]$ ]] || die "FILL (TF_GLM_FILL) is 0 or 1, not $FILL"
  [[ "$PREFIX_WAIT" =~ ^[01]$ ]] || die "PREFIX_WAIT (TF_GLM_PREFIX_WAIT) is 0 or 1, not $PREFIX_WAIT"
  [[ -z "${TF_GLM_PREFIX_WAIT_MS:-}" || "$TF_GLM_PREFIX_WAIT_MS" =~ ^[0-9]{1,9}$ ]] ||
    die "TF_GLM_PREFIX_WAIT_MS is a whole number of milliseconds (0: no wait), not $TF_GLM_PREFIX_WAIT_MS"
  [[ "$SPILL_V2" =~ ^[01]$ ]] || die "SPILL_V2 (TF_GLM_SPILL_V2) is 0 or 1, not $SPILL_V2"
  port_num "$SPILL_GIB" || die "SPILL_GIB is a number of GiB (0: off), not $SPILL_GIB"
  [[ -z "${TF_GLM_KV_SHARD_CARRY:-}" || "$TF_GLM_KV_SHARD_CARRY" =~ ^[01]$ ]] || die "TF_GLM_KV_SHARD_CARRY is 0 or 1, not $TF_GLM_KV_SHARD_CARRY"
  [[ -z "${TF_GLM_KV_SHARD_TWINS:-}" || "$TF_GLM_KV_SHARD_TWINS" =~ ^[01]$ ]] || die "TF_GLM_KV_SHARD_TWINS is 0 or 1, not $TF_GLM_KV_SHARD_TWINS"
  [[ -z "${TF_GLM_KV_SHARD_CHECK:-}" || "$TF_GLM_KV_SHARD_CHECK" =~ ^[0-9]{1,9}$ ]] ||
    die "TF_GLM_KV_SHARD_CHECK is a count of decode rounds, not $TF_GLM_KV_SHARD_CHECK"
  # the spill tier's own settings, when it is on
  if port_spill_on; then
    [[ "$SPILL_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] ||
      { port_spill_refuse "SPILL_DIR must be an absolute path of A-Z a-z 0-9 . _ / - only (the same on every Spark), not '$SPILL_DIR'" ||
        port_spill_default_off "$PORT_SPILL_WHY"; }
  fi
  if port_spill_on; then
    [[ "$SPILL_MIN_TOKENS" =~ ^[0-9]{1,9}$ ]] || die "SPILL_MIN_TOKENS is a token count, not $SPILL_MIN_TOKENS"
    port_num "$SPILL_MIN_FREE_GIB" || die "SPILL_MIN_FREE_GIB is a number of GiB, not $SPILL_MIN_FREE_GIB"
    port_num "$SPILL_FLUSH_S" || die "SPILL_FLUSH_S is a number of seconds, not $SPILL_FLUSH_S"
    port_num "$SPILL_HIGHWATER" && port_gt0 "$SPILL_HIGHWATER" && awk -v h="$SPILL_HIGHWATER" 'BEGIN {exit !(h <= 1)}' ||
      die "SPILL_HIGHWATER is a fraction of the pool above 0, at most 1, not $SPILL_HIGHWATER"
  fi
  if port_v2_on; then
    [[ "$SPILL_QUOTA" =~ ^(0|1|0?[.][0-9]+|1[.]0+)$ ]] || die "SPILL_QUOTA (TF_GLM_SPILL_QUOTA) is a share of SPILL_GIB, 0 to 1, not $SPILL_QUOTA"
    [[ "$SPILL_TURNS" =~ ^[0-9]{1,6}$ ]] || die "SPILL_TURNS (TF_GLM_SPILL_TURNS) is a count of turns (0: every turn), not $SPILL_TURNS"
    for x in TF_GLM_SPILL_READ_BLOCKS TF_GLM_SPILL_STAGE_MIB; do
      [[ -z "${!x:-}" || "${!x}" =~ ^[1-9][0-9]{0,5}$ ]] || die "$x is a whole number from 1, not ${!x}"
    done
    [[ -z "${TF_GLM_SPILL_LOAD_MS:-}" ]] || { port_num "$TF_GLM_SPILL_LOAD_MS" && port_gt0 "$TF_GLM_SPILL_LOAD_MS"; } ||
      die "TF_GLM_SPILL_LOAD_MS is a number of milliseconds above 0, not $TF_GLM_SPILL_LOAD_MS"
    [[ -z "${TF_SPILL_DEFER_MS:-}" || "$TF_SPILL_DEFER_MS" =~ ^[0-9]{1,9}$ ]] || die "TF_SPILL_DEFER_MS is a whole number of milliseconds, not $TF_SPILL_DEFER_MS"
  fi
  # a fork's default KV_SHARD=1 is the Python fork's: two Sparks, more than one stream, TF_GLM_MULTI_LONE=0 (and DFlash2,
  # which PARALLEL above 1 implies); off elsewhere without a word. Explicit, the checks below refuse.
  if [[ "$KV_SHARD" == 1 ]] && (( ! PORT_SHARD_EXPLICIT )); then
    [[ "$TP" == 2 && "$par" =~ ^[0-9]{1,4}$ ]] && (( par > 1 )) && [[ "$DRAFTER" == dflash2 && "${TF_GLM_MULTI_LONE:-0}" == 0 ]] ||
      KV_SHARD=0
  fi
  if [[ "$ENGINE" == zig ]]; then
    # MTP drafts: kept prompt states hold the MTP head's state too, where the image has that port (else the engine
    # turns its prompt cache off with them); no spill tier stores it yet, so their states stay in memory
    if [[ "$DRAFTER" != dflash2 ]] && port_gt0 "$cache" && ! port_has mtpcache; then
      warn "PROMPT_CACHE_GIB=$cache with DRAFTER=$DRAFTER: this image's engine keeps no prompt state with MTP drafts (ZIG_HAS does not name mtpcache, the port that keeps the MTP head's state with them; now: ${ZIG_HAS:-empty}), so its prompt cache is off. Set DRAFTER=dflash2 to keep prompt states"
    fi
    if port_spill_on && [[ "$DRAFTER" != dflash2 ]]; then
      port_spill_refuse "SPILL_GIB=$SPILL_GIB with DRAFTER=$DRAFTER: the Zig engine's spill tier does not store the MTP head's state yet (with MTP drafts the kept prompt states stay in memory), so it would write nothing. Set DRAFTER=dflash2, or SPILL_GIB=0" ||
        port_spill_default_off "$PORT_SPILL_WHY"
    fi
    if port_spill_on && ! port_gt0 "$cache"; then
      port_spill_refuse "SPILL_GIB=$SPILL_GIB with PROMPT_CACHE_GIB=0: the Zig engine spills the prompt cache's kept states, and --prompt-cache-gib 0 keeps none, so the spill tier would never open (a silent no-op before this check). Set PROMPT_CACHE_GIB (GiB of kept prompt states; it needs memory above MEMORY_RESERVE_GIB once the pool is up, so lower KV_POOL_GIB too: docs/zig-engine.md), or SPILL_GIB=0" ||
        port_spill_default_off "$PORT_SPILL_WHY"
    fi
    if port_v2_on && ! port_has spill2; then
      (( ! PORT_V2_EXPLICIT )) || die "SPILL_V2=1: $(port_missing spill2). Set SPILL_V2=0 for the v1 tier"
      port_v2_default_off "$(port_missing spill2)"
    fi
    # the engine refuses spill v2 beyond two ranks (its rank protocol and identity are the pair's): before anything stops
    if port_v2_on && [[ "$TP" != 2 ]]; then
      (( ! PORT_V2_EXPLICIT )) || die "SPILL_V2=1: spill v2 runs on two Sparks only for now (TP=2), not TP=$TP. Set SPILL_V2=0 for the v1 tier"
      port_v2_default_off "spill v2 runs on two Sparks only for now (TP=2), not TP=$TP"
    fi
    if [[ "$KV_SHARD" == 1 ]]; then
      why=""
      if [[ "$TP" != 2 ]]; then why="it is written for two Sparks (TP=2), not TP=$TP"
      elif [[ ! "$par" =~ ^[0-9]{1,4}$ ]] || (( par < 2 )); then why="it needs more than one stream (PARALLEL above 1), not --parallel $par"
      elif [[ "$DRAFTER" != dflash2 ]]; then why="it does not run with DRAFTER=$DRAFTER (MTP drafts)"
      fi
      [[ -z "$why" ]] || die "KV_SHARD=1: $why. Set KV_SHARD=0"
      # the Python fork refuses its lone-stream decode with sharding; the Zig engine has none, so it only warns
      [[ "${TF_GLM_MULTI_LONE:-0}" == 0 ]] ||
        warn "TF_GLM_MULTI_LONE=$TF_GLM_MULTI_LONE does nothing on the Zig engine (it has no lone-stream decode); KV_SHARD=1 runs as without it"
      if ! port_has kvshard; then
        (( ! PORT_SHARD_EXPLICIT )) || die "KV_SHARD=1: $(port_missing kvshard). Set KV_SHARD=0"
        port_shard_default_off "$(port_missing kvshard)"
      fi
    fi
    # both: v2 only (v1 does not know the owner parts), and the two ported together
    if [[ "$KV_SHARD" == 1 ]] && port_spill_on; then
      why=""
      if [[ "$SPILL_V2" != 1 ]]; then why="the v1 spill tier does not know the owner parts (SPILL_V2=1 is the tier for it)"
      elif ! port_has spillshard; then why=$(port_missing spillshard)
      fi
      if [[ -n "$why" ]]; then
        if (( PORT_SHARD_EXPLICIT && PORT_SPILL_EXPLICIT )); then die "KV_SHARD=1 with SPILL_GIB=$SPILL_GIB: $why. Turn one off"
        elif (( PORT_SHARD_EXPLICIT )); then port_spill_default_off "KV_SHARD=1 is set and $why"
        else port_shard_default_off "the spill tier is on and $why"; fi
      fi
    fi
    # sliced prompt fills: one MTP head serves every stream, so a fill cannot run beside another stream's drafts
    if [[ "$FILL" == 1 ]]; then
      [[ "$DRAFTER" == dflash2 ]] || die "FILL=1: sliced prompt fills do not run with DRAFTER=$DRAFTER (one MTP head serves every stream; the engine refuses it too). Set FILL=0"
      port_has fill || die "FILL=1: $(port_missing fill). Set FILL=0"
      # the prefix wait resumes a cut another prompt keeps: a kept prompt state
      [[ "$PREFIX_WAIT" != 1 ]] || (( ! PORT_PW_EXPLICIT )) || port_gt0 "$cache" ||
        warn "PREFIX_WAIT=1 does nothing without a prompt cache (PROMPT_CACHE_GIB=$cache): the cut a request waits for is a kept prompt state"
    else
      # whole passes fill one prompt at a time: a request finds the cut of the one before it kept, nothing to wait for
      [[ "$PREFIX_WAIT" != 1 ]] || (( ! PORT_PW_EXPLICIT )) ||
        warn "PREFIX_WAIT=1 does nothing without FILL=1: whole prompt passes fill one prompt at a time, so a request already finds the shared cut of the one before it kept"
      [[ -z "${TF_GLM_PREFIX_WAIT_MS:-}" ]] || warn "TF_GLM_PREFIX_WAIT_MS=$TF_GLM_PREFIX_WAIT_MS does nothing without FILL=1"
    fi
    # a prompt cache the default pool leaves no room for, and the vision tower's share of what it gets
    port_pool_check "$cache"
    port_vision_check "$cache"
  else
    # the Python engine (patches/): the v1 tier only
    if port_v2_on; then
      (( ! PORT_V2_EXPLICIT )) || die "SPILL_V2=1: $(port_missing spill2). Set SPILL_V2=0 for the v1 tier"
      port_v2_default_off "$(port_missing spill2)"
    fi
    if [[ "$KV_SHARD" == 1 ]]; then
      (( ! PORT_SHARD_EXPLICIT )) || die "KV_SHARD=1: $(port_missing kvshard). Set KV_SHARD=0"
      port_shard_default_off "$(port_missing kvshard)"
    fi
    ! port_gt0 "$cache" || warn "PROMPT_CACHE_GIB=$cache does nothing with ENGINE=python, which keeps prompt states in the KV pool (KV_POOL_GIB)"
    [[ "$PREFIX_WAIT" != 1 ]] || (( ! PORT_PW_EXPLICIT )) ||
      warn "PREFIX_WAIT=1 does nothing with ENGINE=python (this recipe's v0.6.0 has no prefix wait; the Zig engine's needs FILL=1)"
    if [[ "$FILL" == 1 ]]; then
      warn "FILL=1 does nothing with ENGINE=python (the Zig engine's sliced fills); the Python engine fills beside decoding in layer slices (FILL_BUDGET_MS)"
      FILL=0
    fi
  fi
  # settings of a feature that is off
  [[ -z "$PORT_SHARD_SET" || "$KV_SHARD" == 1 ]] || warn "${PORT_SHARD_SET# } do nothing without KV_SHARD=1"
  [[ -z "$PORT_V2_SET" ]] || port_v2_on || warn "${PORT_V2_SET# } do nothing without the v2 spill tier (SPILL_GIB above 0, SPILL_V2=1)"
  [[ "$SPILL_V2" == 0 ]] || (( ! PORT_V2_EXPLICIT )) || port_spill_on || warn "SPILL_V2=1 does nothing without SPILL_GIB above 0"
  return 0
}

# port_check_extra ARGS...: the command line's extra serve arguments (./start.sh [restart] ARGS) may not turn the spill
# tier on: its folder is checked and mounted only through the settings (SPILL_GIB, SPILL_DIR), so a --spill-gib given
# here would write inside the container, and be lost with it. With the tier on, the flags port_map passes come after the
# extra arguments and win, so one given here too would be ignored: refused, naming its setting. Other flags pass as
# before (--snapshot-dir with the tier off is TensorFold's own snapshot folder; 0088's --spill-keep-builds has no setting).
port_check_extra() {
  local a f s
  for a in "$@"; do
    f=${a%%=*}
    case "$f" in
      --spill-gib)
        die "the extra argument $a: the spill tier goes on through SPILL_GIB (with SPILL_DIR; scripts/port.sh), which checks and mounts its folder on every Spark" ;;
      --snapshot-dir) s="SPILL_DIR (mounted at $PORT_SPILL_MOUNT on every rank)" ;;
      --spill-highwater) [[ "$SPILL_V2" != 1 ]] || continue; s=SPILL_HIGHWATER ;;   # v1's: not passed under v2
      --spill-min-tokens) s=SPILL_MIN_TOKENS ;;
      --spill-min-free-gib) s=SPILL_MIN_FREE_GIB ;;
      *) continue ;;
    esac
    ! port_spill_on ||
      die "the extra argument $a: with the spill tier on (SPILL_GIB=$SPILL_GIB) start.sh passes $f itself, after the extra arguments, so this one would be ignored. Set $s instead"
  done
  return 0
}

# port_check_started <rank 0's log> [<rank 1's log>...]: with the spill tier on (ENGINE=zig), every rank must have opened
# it (each log: its lines matching PORT_TIER_LOG_RE; "(unread)" when it could not be read). The engine's lines
# (glm_cuda/cuda_native.zig): v1 opens with "spill tier: <dir>, up to ..." and fails with "spill: cannot open <dir>
# (<error>); no spill tier"; v2 (SPILL_V2=1) opens with "rank N: spill v2 tier: <dir>, up to ..." and fails with "rank N:
# spill v2's store under <dir> did not open (<error>)". The tier asked for is the one looked for: a rank that logged the
# other version's line warns too. Rank 0 first needs a prompt cache, which the tier writes from: its budget is what
# MemAvailable leaves above the reserve once the pool is up, so it can come out 0 even with PROMPT_CACHE_GIB above 0
# (Mia PR #187), and no tier opens. Each rank opens its v1 tier from its own budget (cuda_native.zig cacheFor), with no
# agreement between the ranks: a Spark with a little less memory free can come out with none while rank 0 writes and
# restores states it cannot follow (v2 agrees one budget and fails the start on every rank when a store does not open,
# so there a missing line means a log that did not get that far). Warns; the server runs on.
PORT_TIER_LOG_RE="prompt cache:|spill tier: |spill v2 tier: |spill: cannot open|spill v2's store under .* did not open"
port_check_started() {
  [[ "$ENGINE" == zig ]] && port_spill_on || return 0
  local line l r=0 want='spill tier: ' other='spill v2 tier: ' failed='spill: cannot open.*' v=v1 ov=v2 name='spill tier'
  if [[ "$SPILL_V2" == 1 ]]; then
    want='spill v2 tier: ' other='spill tier: ' failed="spill v2's store under .* did not open.*" v=v2 ov=v1 name='spill v2 tier'
  fi
  line=$(grep -m1 -o 'prompt cache: off[^"]*' <<<"$1" || true)
  if [[ -n "$line" ]]; then
    warn "the spill tier is not open: rank 0 logged \"$line\", and the tier writes only kept prompt states. Lower KV_POOL_GIB (the pool takes that memory first) or MEMORY_RESERVE_GIB, then ./start.sh restart"
    return 0
  fi
  for l in "$@"; do
    if [[ "$l" == '(unread)' ]]; then warn "could not read rank $r's log: whether its spill tier opened is not checked"
    elif grep -q -- "$want" <<<"$l"; then :
    elif grep -q -- "$other" <<<"$l"; then
      warn "rank $r opened the $ov spill tier, not $v (SPILL_V2=$SPILL_V2): this image's engine did not take the setting; ./start.sh stop, then check ZIG_HAS and the image"
    else
      line=$(grep -m1 -o -- "$failed" <<<"$l" || true)
      if (( r == 0 )); then warn "rank 0 opened no $name${line:+ (it logged \"$line\")}: no kept state is written to disk"
      else warn "rank $r opened no $name${line:+ (it logged \"$line\")}: rank 0 writes and restores kept states that rank $r cannot follow (a restore from disk fails there). Each rank's prompt cache budget is its own: free memory on that Spark (or lower KV_POOL_GIB), then ./start.sh restart"; fi
    fi
    r=$((r + 1))
  done
  return 0
}

# ---------------------------------------------------------------- the mapping (the one place)
# port_map fills:
#   PORT_SERVE      the serve flags every rank gets (the same SERVE_ARGS on every rank; checked against
#                   `tensorfold-native capabilities --json` by scripts/zig_flags.py before anything stops)
#   PORT_ENV        NAME=value pairs every rank gets, identical (docker -e)
#   PORT_MAPPED_RE  the variable names port_map owns: an exported variable of these names is not passed a second time.
#                   The three switches (TF_GLM_KV_SHARD, TF_GLM_SPILL_V2, TF_GLM_FILL) always: a switch reaches the ranks only on,
#                   and only through here, so a hand-exported one can never turn a feature on past the checks
# Engine names (the settings contract, sparks/deployments/2026-10-22-glm-zig-port/PORT-STATUS.md): the v1 flags
# --spill-gib, --snapshot-dir, --spill-highwater, --spill-min-tokens, --spill-min-free-gib (server/cli.zig) and
# TF_SPILL_FLUSH_S (glm_cuda/cuda_native.zig openSpill); v2 keeps the flags (--spill-highwater is v1's: not passed)
# and adds TF_GLM_SPILL_V2, TF_GLM_SPILL_QUOTA (a share), TF_GLM_SPILL_TURNS, TF_GLM_SPILL_READ_BLOCKS,
# TF_GLM_SPILL_STAGE_MIB, TF_GLM_SPILL_LOAD_MS, TF_SPILL_DEFER_MS; KV sharding TF_GLM_KV_SHARD, _CARRY, _TWINS, _CHECK;
# sliced fills TF_GLM_FILL (cuda_native.zig: 1, or 0/unset; in the ranks' settings text only when on); the prefix wait
# TF_GLM_PREFIX_WAIT (1 default, 0 off) and TF_GLM_PREFIX_WAIT_MS (native/cuda.zig serve: rank 0's admission alone).
PORT_MAPPED_BASE='TF_GLM_KV_SHARD|TF_GLM_SPILL_V2|TF_GLM_FILL|TF_GLM_PREFIX_WAIT|TF_GLM_PREFIX_WAIT_MS'
PORT_MAPPED_SPILL='TF_SPILL_FLUSH_S'
PORT_MAPPED_V2='TF_GLM_SPILL_QUOTA|TF_GLM_SPILL_TURNS|TF_GLM_SPILL_READ_BLOCKS|TF_GLM_SPILL_STAGE_MIB|TF_GLM_SPILL_LOAD_MS|TF_SPILL_DEFER_MS'
PORT_MAPPED_SHARD='TF_GLM_KV_SHARD_CARRY|TF_GLM_KV_SHARD_TWINS|TF_GLM_KV_SHARD_CHECK'
PORT_MAPPED_RE=$PORT_MAPPED_BASE
PORT_ENV=(); PORT_SERVE=()
port_map() {
  local x
  PORT_ENV=(); PORT_SERVE=()
  PORT_MAPPED_RE=$PORT_MAPPED_BASE
  if port_spill_on; then
    PORT_SERVE+=(--spill-gib "$SPILL_GIB" --snapshot-dir "$PORT_SPILL_MOUNT")
    [[ "$SPILL_V2" == 1 ]] || PORT_SERVE+=(--spill-highwater "$SPILL_HIGHWATER")
    PORT_SERVE+=(--spill-min-tokens "$SPILL_MIN_TOKENS" --spill-min-free-gib "$SPILL_MIN_FREE_GIB")
    PORT_ENV+=("TF_SPILL_FLUSH_S=$SPILL_FLUSH_S")
    PORT_MAPPED_RE+="|$PORT_MAPPED_SPILL"
    if [[ "$SPILL_V2" == 1 ]]; then
      PORT_ENV+=("TF_GLM_SPILL_V2=1" "TF_GLM_SPILL_QUOTA=$SPILL_QUOTA" "TF_GLM_SPILL_TURNS=$SPILL_TURNS")
      for x in TF_GLM_SPILL_READ_BLOCKS TF_GLM_SPILL_STAGE_MIB TF_GLM_SPILL_LOAD_MS TF_SPILL_DEFER_MS; do
        [[ -z "${!x:-}" ]] || PORT_ENV+=("$x=${!x}")
      done
      PORT_MAPPED_RE+="|$PORT_MAPPED_V2"
    fi
  fi
  if [[ "$KV_SHARD" == 1 ]]; then
    PORT_ENV+=("TF_GLM_KV_SHARD=1" "TF_GLM_KV_SHARD_CARRY=${TF_GLM_KV_SHARD_CARRY:-1}" "TF_GLM_KV_SHARD_TWINS=${TF_GLM_KV_SHARD_TWINS:-1}")
    [[ -z "${TF_GLM_KV_SHARD_CHECK:-}" ]] || PORT_ENV+=("TF_GLM_KV_SHARD_CHECK=$TF_GLM_KV_SHARD_CHECK")
    PORT_MAPPED_RE+="|$PORT_MAPPED_SHARD"
  fi
  if [[ "$FILL" == 1 ]]; then
    PORT_ENV+=("TF_GLM_FILL=1" "TF_GLM_PREFIX_WAIT=$PREFIX_WAIT")
    [[ -z "${TF_GLM_PREFIX_WAIT_MS:-}" ]] || PORT_ENV+=("TF_GLM_PREFIX_WAIT_MS=$TF_GLM_PREFIX_WAIT_MS")
  fi
  return 0
}

# port_spill_identity: spill v2's identity (glm_cuda/cuda_spill_identity.zig), the same on every rank or the engine
# refuses to start ("spill v2 never stores state an unidentified engine made"): the engine (TensorFold's pin and this
# image's hash: zig/patches, the kernel set, the Dockerfile, the pins), the zig/patches series, the checkpoint and
# DFlash2 by id and pinned revision. Fills PORT_ID_ENV (start.sh passes it with PORT_ENV); nothing unless SPILL_V2=1 on
# the Zig engine. Called after port_map (it needs config.sh's image_hash).
PORT_ID_ENV=()
PORT_ID_NAMES='TF_GLM_ENGINE_ID|TF_GLM_RECIPE_PATCHES|TF_GLM_MODEL_ID|TF_GLM_MODEL_REVISION|TF_GLM_DFLASH2_ID|TF_GLM_DFLASH2_REVISION'
port_spill_identity() {
  local series
  PORT_ID_ENV=()
  [[ "$ENGINE" == zig ]] && port_v2_on || return 0
  [[ -n "${MODEL_REVISION:-}" ]] || die "SPILL_V2=1 needs a pinned MODEL_REVISION: spill v2 never stores state that weights it cannot identify made"
  [[ "$DRAFTER" != dflash2 || -n "${DFLASH2_REVISION:-}" ]] ||
    die "SPILL_V2=1 with DRAFTER=dflash2 needs a pinned DFLASH2_REVISION: spill v2 never stores state that a drafter it cannot identify made"
  series=$(cd "$_cfg_root" && cat zig/patches/*.patch 2>/dev/null | sha256sum | cut -c1-16)
  PORT_ID_ENV=("TF_GLM_ENGINE_ID=tensorfold-${ZIG_TF_REF:0:12}+$(image_hash)" "TF_GLM_RECIPE_PATCHES=sha256:$series"
               "TF_GLM_MODEL_ID=$MODEL_ID" "TF_GLM_MODEL_REVISION=$MODEL_REVISION")
  [[ "$DRAFTER" != dflash2 ]] || PORT_ID_ENV+=("TF_GLM_DFLASH2_ID=$DFLASH2_ID" "TF_GLM_DFLASH2_REVISION=$DFLASH2_REVISION")
  PORT_MAPPED_RE+="|$PORT_ID_NAMES"
}

# port_identity: what port_map gave the ranks, as one word (the containers' tf.recipe.port label; start.sh compares a
# running server's with it): "off" when nothing is on, as containers started before the port (no label) read. Else the
# features and a checksum of every flag, variable and SPILL_DIR (another folder is another set of kept states).
# FILL=1 adds ",fill=1" (a label without it is the one before: fills off).
port_identity() {
  port_spill_on || [[ "$KV_SHARD" == 1 || "$FILL" == 1 ]] || { echo off; return 0; }
  local sp=0 sum
  ! port_spill_on || { sp=v1; [[ "$SPILL_V2" != 1 ]] || sp=v2; }
  sum=$(printf '%s\n' ${PORT_SERVE[@]+"${PORT_SERVE[@]}"} ${PORT_ENV[@]+"${PORT_ENV[@]}"} "$SPILL_DIR" | cksum | awk '{print $1}')
  echo "spill=$sp,shard=$KV_SHARD$( [[ "$FILL" != 1 ]] || echo ",fill=1" ),sum=$sum"
}
# port_summary: the words for what is on (empty when nothing is)
port_summary() {
  local s=""
  ! port_spill_on || s+=" · spill $( [[ "$SPILL_V2" == 1 ]] && echo v2 || echo v1) $SPILL_GIB GiB"
  [[ "$KV_SHARD" != 1 ]] || s+=" · KV shard"
  [[ "$FILL" != 1 ]] || s+=" · sliced fills$( [[ "$PREFIX_WAIT" == 1 ]] || echo " (no prefix wait)" )"
  echo "$s"
}

# ---------------------------------------------------------------- each Spark's spill directory
# spill_dir_probe <create 0|1> <dir> [<path it must not overlap>...]: one line about <dir> on this node, self-contained
# (start.sh sends it to the workers over ssh; bash or sh, Linux or a Mac). With create=1 a missing directory is made (as
# you, so docker never creates it as root); with 0 (DRY_RUN) the nearest existing parent is described. Prints
# "OK|NEW <writable 0|1> <fs type> <free bytes> <tier bytes> <tier unread 0|1> <exported 0|1> <overlapped path or ->
# <home-or-root 0|1> <canonical path>", or "ERR <reason>". Exported: under a path that /var/lib/nfs/etab, /etc/exports or
# /etc/exports.d/*.exports lists. Tier bytes: the disk the spill tiers' own folders in <dir> take (du), which the engine
# frees itself (it drops its oldest files to keep the floor, and removes other builds' folders at open): tfspill-<hash>
# (the Zig v1 tier), glm-spill-<hash> (the Zig v2 tier, zig/patches/0021: core/spill under GLM's prefix; its folders are
# SPILL_DIR's owner's), native-spill-<hash> (core/spill's other families) and spill-<hash> (the Python engine's 0088); not
# symlinks. Tier unread 1: one of them could not be read through (the Zig v1 tier writes them as root, 0700), so it
# counts short.
spill_dir_probe() {
  local create=$1 d=$2 st=OK p real rest w=0 fs="" avail exp=0 e x f fr over=- top=0 home tb=0 tu=0 k
  shift 2
  if [ ! -d "$d" ]; then
    if [ "$create" = 1 ]; then mkdir -p -- "$d" 2>/dev/null || { echo "ERR cannot create $d"; return 0; }
    else st=NEW; fi
  fi
  p=$d; while [ ! -d "$p" ]; do p=$(dirname -- "$p"); done
  real=$(cd -P -- "$p" 2>/dev/null && pwd -P) || { echo "ERR cannot enter $p"; return 0; }
  if [ "$p" != "$d" ]; then rest=${d#"$p"}; real="${real%/}/${rest#/}"; fi
  [ ! -w "$p" ] || w=1
  if command -v findmnt >/dev/null 2>&1; then fs=$(findmnt -n -o FSTYPE -T "$p" 2>/dev/null | head -1); fi
  [ -n "$fs" ] || fs=$(stat -f -c %T "$p" 2>/dev/null) || fs=""
  avail=$(df -Pk "$p" 2>/dev/null | awk 'NR == 2 { printf "%.0f", $4 * 1024 }')
  if [ "$st" = OK ]; then
    for x in "$d"/tfspill-* "$d"/glm-spill-* "$d"/native-spill-* "$d"/spill-*; do
      [ -d "$x" ] && [ ! -L "$x" ] || continue
      k=$(du -sk -- "$x" 2>/dev/null) || tu=1
      k=${k%%[!0-9]*}
      tb=$((tb + ${k:-0}))
    done
  fi
  for x in /var/lib/nfs/etab /etc/exports /etc/exports.d/*.exports; do
    [ -r "$x" ] || continue
    while read -r e _; do
      e=${e#\"}; e=${e%\"}
      case "$e" in /*) e=${e%/}; case "$real/" in "$e"/*) exp=1 ;; esac ;; esac
    done < "$x"
  done
  for f in "$@"; do
    [ -n "$f" ] || continue
    fr=$f; [ ! -d "$f" ] || fr=$(cd -P -- "$f" 2>/dev/null && pwd -P) || fr=$f
    fr=${fr%/}
    case "$real/" in "$fr"/*) over=$fr; break ;; esac
    case "$fr/" in "$real"/*) over=$fr; break ;; esac
  done
  home=$(cd -P -- "$HOME" 2>/dev/null && pwd -P) || home=$HOME
  [ "$real" != / ] && [ "$real" != "$home" ] || top=1
  echo "$st $w ${fs:-unknown} ${avail:-0} $((tb * 1024)) $tu $exp $(printf '%s' "$over" | tr ' \t' '__') $top $real"
}

# port_spill_judge <where> <probe line>: refuse a spill directory that is not a writable folder of its own on this
# Spark's local disk, outside the shared model files, with SPILL_MIN_FREE_GIB free. Free counts the tiers' own folders
# too (the engine frees them itself; an estimate on the generous side, as 0088 and v2 keep the newest other build); when
# one of them cannot be read (root's), a shortfall only warns. Each refusal goes through port_spill_refuse: with
# SPILL_GIB explicit it dies; with the default it returns 1 and PORT_SPILL_WHY says why.
port_spill_judge() {
  local where=$1 st w fs avail tier tu exp over top real floor_b want_b room has
  read -r st w fs avail tier tu exp over top real <<<"$2"
  [[ "$st" == OK || "$st" == NEW ]] || { port_spill_refuse "SPILL_DIR on $where: ${2#ERR }"; return 1; }
  case "$fs" in
    nfs|nfs4|cifs|smb|smb2|smb3|smbfs|fuse.sshfs|sshfs|9p|ceph|fuse.ceph*|glusterfs|fuse.glusterfs|lustre|afs|davfs|fuse.rclone|fuse.s3fs|fuse.juicefs)
      port_spill_refuse "SPILL_DIR $real on $where is on a network file system ($fs): the spill tier needs each Spark's own local disk"; return 1 ;;
    tmpfs|ramfs|devtmpfs|overlay|overlayfs|squashfs|zram)
      port_spill_refuse "SPILL_DIR $real on $where is on $fs (memory, a container layer or a read-only image): the spill tier needs each Spark's own local disk"; return 1 ;;
  esac
  [[ "$exp" == 0 ]] || { port_spill_refuse "SPILL_DIR $real on $where is inside an NFS export (/var/lib/nfs/etab, /etc/exports or /etc/exports.d): keep the spill tier out of the model files the workers read (WORKER_WEIGHTS=nfs)"; return 1; }
  [[ "$over" == - ]] || { port_spill_refuse "SPILL_DIR $real on $where overlaps $over (the Hugging Face cache, or the NFS export the workers read): give the spill tier a folder of its own"; return 1; }
  [[ "$top" == 0 ]] || { port_spill_refuse "SPILL_DIR on $where is / or your home folder itself: give the spill tier a folder of its own"; return 1; }
  if [[ "$w" != 1 ]]; then
    if [[ "$st" == NEW ]]; then port_spill_refuse "SPILL_DIR $real on $where does not exist, and you cannot create it there (its nearest existing parent is not writable)"
    else port_spill_refuse "SPILL_DIR $real on $where is not writable by you (docker may have created it as root): make it yours there (sudo chown \$(id -u) $real)"; fi
    return 1
  fi
  floor_b=$(awk -v g="$SPILL_MIN_FREE_GIB" 'BEGIN {printf "%.0f", g * 1073741824}')
  want_b=$(awk -v g="$SPILL_GIB" -v f="$floor_b" 'BEGIN {printf "%.0f", g * 1073741824 + f}')
  room=$(awk -v a="$avail" -v t="$tier" 'BEGIN {printf "%.0f", a + t}')
  has=$(awk -v a="$avail" -v t="$tier" 'BEGIN {printf "%d GiB free", a / 1073741824; if (t + 0 > 0) printf " (and %.1f GiB in the spill tiers\047 own folders, which the tier frees itself)", t / 1073741824}')
  if ! awk -v r="$room" -v f="$floor_b" 'BEGIN {exit !(r + 0 >= f + 0)}'; then
    if [[ "$tu" == 1 ]]; then
      warn "SPILL_DIR $real on $where has $(awk -v a="$avail" 'BEGIN {printf "%d", a / 1073741824}') GiB free, under SPILL_MIN_FREE_GIB ($SPILL_MIN_FREE_GIB), and holds spill folders this check cannot read (the Zig engine writes them as root): the tier frees its own files to keep the floor, and skips a write when they are not enough (spill: skipped in /health)"
    else
      port_spill_refuse "SPILL_DIR $real on $where has $has, under SPILL_MIN_FREE_GIB ($SPILL_MIN_FREE_GIB): the tier would refuse every write. Free the disk or lower SPILL_MIN_FREE_GIB"; return 1
    fi
  elif [[ "$tu" == 0 ]] && awk -v r="$room" -v w="$want_b" 'BEGIN {exit !(r + 0 < w + 0)}'; then
    warn "SPILL_DIR $real on $where has $has, less than SPILL_GIB ($SPILL_GIB) + SPILL_MIN_FREE_GIB ($SPILL_MIN_FREE_GIB): the tier stops writing at the floor before it reaches its cap"
  fi
  [[ "$st" != NEW ]] || log "SPILL_DIR $real on $where does not exist yet: start.sh creates it (as you)"
  return 0
}

# port_spill_check: SPILL_DIR on every Spark, before anything is stopped (start.sh, after need_workers): created when
# missing (as you; DRY_RUN=1 creates nothing), then judged. With SPILL_GIB explicit a failed check refuses the start;
# with a default one it turns the tier off on every rank (port_spill_default_off) when any Spark's folder fails.
port_spill_check() {
  port_spill_on || return 0
  local create=1 line i h
  [[ "${DRY_RUN:-0}" != 1 ]] || create=0
  PORT_SPILL_WHY=""
  line=$(spill_dir_probe "$create" "$SPILL_DIR" "$HF_CACHE" "$(port_nfs_path)")
  port_spill_judge "this Spark" "$line" || { port_spill_default_off "$PORT_SPILL_WHY"; return 0; }
  for i in $(worker_ids); do
    if [[ -n "${WORKER_DOWN[$i]:-}" ]]; then warn "DRY_RUN: $(wvar WORKER "$i") cannot be reached: SPILL_DIR is not checked there"; continue; fi
    # that worker's Hugging Face cache (WORKER_HF_CACHE / WORKER_HF_CACHE<i>, else its HF_HOME there)
    h=$(wval WORKER_HF_CACHE "$i")
    if [[ -n "$h" ]]; then h=$(printf '%q' "$h"); else h='"${HF_HOME:-$HOME/.cache/huggingface}"'; fi
    if ! line=$(worker "$i" "$(declare -f spill_dir_probe); spill_dir_probe $create '$SPILL_DIR' $h" 2>/dev/null) || [[ -z "$line" ]]; then
      port_spill_refuse "could not check SPILL_DIR on $(worker_host "$i")" || port_spill_default_off "$PORT_SPILL_WHY"
      return 0
    fi
    port_spill_judge "$(worker_host "$i")" "$line" || { port_spill_default_off "$PORT_SPILL_WHY"; return 0; }
  done
  log "Spill tier: $( [[ "$SPILL_V2" == 1 ]] && echo v2 || echo v1), up to $SPILL_GIB GiB a Spark in $SPILL_DIR (on every Spark), keeping $SPILL_MIN_FREE_GIB GiB of disk free$( [[ "$ENGINE" != zig ]] || echo "; the engine writes its files there as root (removing them takes sudo)")"
}
# NFS_PATH when it is a real path ("/" is an NFSv4 pseudo-root, not a folder of this Spark)
port_nfs_path() { [[ "${NFS_PATH:-/}" == / ]] || echo "$NFS_PATH"; }

# ---- long contexts on the Zig engine: TF_TP_TIMEOUT_S and what the kernel set is known to cover
# zig/aot/glm-aot-coverage.txt records the engine's offline host test (cuda_aot_coverage_test: every Triton lookup the
# engine makes at TP=2, for the contexts, stream counts and KV pools it names, against the captured set) and the
# contexts a GPU run has served at TP=2 (gpu=). key=value lines; aot= is the sha256 of the set it checked.
PORT_COVERAGE="${PORT_COVERAGE:-${_cfg_root:-.}/zig/aot/glm-aot-coverage.txt}"
port_coverage() {
  [[ -f "$PORT_COVERAGE" ]] || return 0
  sed -n "s/^$1=\([^#]*\).*/\1/p" "$PORT_COVERAGE" | head -1 | sed 's/[[:space:]]*$//'
}
# port_long_context <context> <parallel> (the effective --context and --parallel), ENGINE=zig: past 262,144 tokens
# TF_TP_TIMEOUT_S=1500 unless set (a rank waits that long for the other's next call; 300 s by default is less than one
# long prompt's pass: TP=3/4 served ~1M prompts at 1500), and a warning naming what is unverified at this window
port_long_context() {
  local ctx=$1 par=$2 set_sha aot_sha top c gpu=0 known=0 streams=0
  [[ "$ENGINE" == zig ]] || return 0
  if [[ "$ctx" =~ ^[0-9]+$ ]] && (( ctx > 262144 )) && [[ -z "${TF_TP_TIMEOUT_S:-}" ]]; then
    export TF_TP_TIMEOUT_S=1500
    log "CONTEXT=$ctx: TF_TP_TIMEOUT_S=1500 (a long prompt's pass outlasts the ranks' default 300 s wait)"
  fi
  if [[ "$TP" != 2 ]]; then
    [[ "$ctx" == 131072 && "$par" == 8 ]] ||
      warn "ENGINE=zig: the captured kernel set is keyed on the 8-stream, 131072-token arena; other CONTEXT / PARALLEL can fail with MissingTritonVariant"
    return 0
  fi
  aot_sha=$(cut -c1-64 "${_cfg_root:-.}/zig/aot/glm-aot-sm121.tar.xz.sha256" 2>/dev/null)
  set_sha=$(port_coverage aot)
  if [[ -z "$set_sha" || "$set_sha" != "$aot_sha" ]]; then
    warn "ENGINE=zig: no coverage record for this kernel set (zig/aot/glm-aot-coverage.txt$( [[ -z "$set_sha" ]] || echo " is for set ${set_sha:0:12}, this one is ${aot_sha:0:12}")): CONTEXT=$ctx PARALLEL=$par is unchecked and can fail with MissingTritonVariant"
    return 0
  fi
  for c in $(port_coverage gpu); do [[ "$c" == "$ctx" ]] && gpu=1; done
  for c in $(port_coverage parallel); do [[ "$c" == "$par" ]] && streams=1; done
  top=0; for c in $(port_coverage contexts); do (( c > top )) && top=$c; done
  [[ "$ctx" =~ ^[0-9]+$ ]] && (( ctx >= 1 && ctx <= top )) && known=1
  if (( ! known || ! streams )); then
    warn "ENGINE=zig: CONTEXT=$ctx PARALLEL=$par is outside what the kernel set's coverage test checked (contexts up to $top, PARALLEL $(port_coverage parallel), KV pools of $(port_coverage pools) rows: zig/aot/glm-aot-coverage.txt): it can fail with MissingTritonVariant"
  elif (( ! gpu )) || [[ "$par" != "$(port_coverage gpu_parallel)" ]]; then
    warn "ENGINE=zig: CONTEXT=$ctx PARALLEL=$par: the kernel set holds every Triton variant this window looks up (checked offline: zig/aot/glm-aot-coverage.txt, for KV pools of $(port_coverage pools) rows), but no GPU run has served it yet (GPU-checked: CONTEXT $(port_coverage gpu) at PARALLEL $(port_coverage gpu_parallel)). Unverified: a long prompt's memory on each Spark, TF_TP_TIMEOUT_S=${TF_TP_TIMEOUT_S:-300}, and the replies"
  fi
}
