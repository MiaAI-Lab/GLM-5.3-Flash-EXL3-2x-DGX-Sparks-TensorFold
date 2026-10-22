#!/usr/bin/env bash
# Checks of scripts/port.sh and scripts/memavail.sh without the Sparks: the defaults (everything off: no flag, variable,
# mount or label), a fork's defaults turned on (PT_FORK=1: spill v2 64 GiB, KV sharding) degrading where an explicit
# setting refuses, the ZIG_HAS gates (spill2, kvshard, spillshard), PROMPT_CACHE_GIB as the spill tier's requirement,
# the aliases, the mapping to the engine's flags and variables, the refusals (port_validate), the spill folder rules
# (port_spill_judge on synthetic probe lines, spill_dir_probe and port_spill_check on temporary folders, the worker's
# probe through its ssh command run locally), the parked pages on synthetic /proc files, and `bash -n` of every script.
# bash 3.2 or later (runs on a Mac); no docker, no ssh, no GPU.
# Usage: scripts/port_test.sh      (exits 1 when a check fails)
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
fails=0; n=0
tmp=$(mktemp -d "${TMPDIR:-/tmp}/port-test.XXXXXX")
trap 'chmod -R u+w "$tmp" 2>/dev/null; rm -rf -- "$tmp"' EXIT
# PT_FORK=1 in a case's first script: port.sh with a fork's defaults on (Robert's: spill v2 64 GiB a Spark with a quarter
# of it a conversation, KV sharding), the configuration the degrade paths exist for
fork_sh=$tmp/port-fork.sh
sed -e 's/^PORT_SPILL_DEFAULT_GIB=[0-9.]*/PORT_SPILL_DEFAULT_GIB=64/' -e 's/^PORT_SPILL_V2_DEFAULT=[01]/PORT_SPILL_V2_DEFAULT=1/' \
    -e 's/^PORT_SHARD_DEFAULT=[01]/PORT_SHARD_DEFAULT=1/' -e 's/^PORT_SPILL_QUOTA_DEFAULT=[0-9.]*/PORT_SPILL_QUOTA_DEFAULT=0.25/' \
    "$here/port.sh" > "$fork_sh"
grep -q '^PORT_SPILL_DEFAULT_GIB=64 ' "$fork_sh" && grep -q '^PORT_SPILL_V2_DEFAULT=1 ' "$fork_sh" && grep -q '^PORT_SHARD_DEFAULT=1 ' "$fork_sh" &&
  grep -q '^PORT_SPILL_QUOTA_DEFAULT=0.25 ' "$fork_sh" ||
  { echo "FAIL the fork variant of port.sh (its default lines moved?)"; exit 1; }
ALL_HAS="mtp displaykv spill2 kvshard spillshard"

# case_ <name> <expected exit 0|1> <script before port.sh> <script after>: runs both in a subshell over a fresh port.sh,
# with the serving settings of the two-Spark Zig default (TP 2, PARALLEL 8, DFlash2, PROMPT_CACHE_GIB 0, ZIG_HAS as
# start.sh's default) and nothing of the caller's environment; die/warn/log stubbed (die exits 1)
case_() {
  local name=$1 want=$2 got out
  n=$((n + 1))
  out=$( (
    for v in $(compgen -v | grep -E '^(SPILL_|KV_SHARD|TF_GLM_|TF_SPILL_|PREFIX_WAIT|STOP_TIMEOUT|PROMPT_CACHE_GIB|ZIG_HAS|WORKER)'); do
      unset "$v"
    done
    ENGINE=zig TP=2 PARALLEL=8 DRAFTER=dflash2 TF_GLM_MULTI_LONE=0 PROMPT_CACHE_GIB=0 ZIG_HAS="mtp displaykv"
    HOME=/home/u HF_CACHE=/home/u/.cache/huggingface NFS_PATH=/home/u/.cache/huggingface DRY_RUN=0
    die()  { echo "DIE: $*"; exit 1; }
    warn() { echo "WARN: $*"; }
    log()  { echo "LOG: $*"; }
    expect() { [[ "$1" == "$2" ]] || { echo "got  [$1]"; echo "want [$2]"; exit 1; }; }
    # expect_die <pattern> <command...>: the command refuses, with a message matching the pattern
    expect_die() {
      local p=$1 o; shift
      o=$( ( "$@" ) 2>&1 ) && { echo "no refusal from $*: $o"; exit 1; }
      grep -q "DIE: .*$p" <<<"$o" || { echo "refused, but not with [$p]:"; echo "$o"; exit 1; }
    }
    # expect_out <pattern> <command...>: the command succeeds and prints a line matching the pattern (no: !<pattern>)
    expect_out() {
      local p=$1 o; shift
      o=$( ( "$@" ) 2>&1 ) || { echo "failed: $*"; echo "$o"; exit 1; }
      if [[ "$p" == '!'* ]]; then ! grep -q -- "${p#!}" <<<"$o" || { echo "unwanted [${p#!}]:"; echo "$o"; exit 1; }
      else grep -q -- "$p" <<<"$o" || { echo "missing [$p]:"; echo "$o"; exit 1; }; fi
    }
    eval "$3"
    # shellcheck source=scripts/port.sh
    if [[ "${PT_FORK:-0}" == 1 ]]; then source "$fork_sh"; else source "$here/port.sh"; fi
    eval "$4"
  ) 2>&1 ); got=$?
  if [[ "$got" == "$want" ]]; then printf 'ok   %s\n' "$name"
  else fails=$((fails + 1)); printf 'FAIL %s (exit %s, wanted %s)\n%s\n' "$name" "$got" "$want" "$(sed 's/^/     /' <<<"$out")"; fi
}

# ---- the defaults: everything off
case_ "defaults: everything off, no flag, variable or label; SPILL_DIR beside the Hugging Face cache" 0 '' '
  expect "$PORT_SPILL_EXPLICIT $PORT_V2_EXPLICIT $PORT_SHARD_EXPLICIT $SPILL_GIB $SPILL_V2 $KV_SHARD" "0 0 0 0 0 0"
  port_validate 8 0; port_map
  expect "${#PORT_ENV[@]} ${#PORT_SERVE[@]}" "0 0"
  expect "$(port_identity)" off
  expect "$(port_summary)" ""
  expect "${STOP_TIMEOUT:-unset}" unset
  expect "$PORT_MAPPED_RE" "$PORT_MAPPED_BASE"
  expect "$SPILL_DIR" /home/u/.cache/tensorfold-spill'
case_ "defaults: the ranks' passthrough drops only the two switches (TF_GLM_KV_SHARD, TF_GLM_SPILL_V2)" 0 '' '
  port_validate 8 0; port_map
  expect "$(printf "%s\n" TF_GLM_KV_SHARD=0 TF_GLM_KV_SHARD_CARRY=0 TF_GLM_SPILL_V2=1 TF_GLM_SPILL_QUOTA=0.25 TF_GLM_CACHE_GIB=12.5 TF_SPILL_FLUSH_S=60 |
    grep -vE "^($PORT_MAPPED_RE)=" | paste -sd" " -)" "TF_GLM_KV_SHARD_CARRY=0 TF_GLM_SPILL_QUOTA=0.25 TF_GLM_CACHE_GIB=12.5 TF_SPILL_FLUSH_S=60"'
case_ "defaults with ENGINE=python: off as well" 0 'ENGINE=python' '
  port_validate "" ""; port_map; expect "${#PORT_ENV[@]} ${#PORT_SERVE[@]} $(port_identity)" "0 0 off"'

# ---- the spill tier (v1) and PROMPT_CACHE_GIB
case_ "refuse: SPILL_GIB>0 with PROMPT_CACHE_GIB=0 (zig: the tier would never open)" 0 'SPILL_GIB=64' '
  expect_die "SPILL_GIB=64 with PROMPT_CACHE_GIB=0.*silent no-op" port_validate 8 0'
case_ "SPILL_GIB>0 with the effective --prompt-cache-gib above 0 (an extra argument wins)" 0 'SPILL_GIB=64' '
  port_validate 8 8; port_map; expect "${#PORT_SERVE[@]}" 10'
case_ "spill v1: the flags, TF_SPILL_FLUSH_S, STOP_TIMEOUT, the label" 0 'SPILL_GIB=64 PROMPT_CACHE_GIB=8' '
  port_validate 8 8; port_map
  expect "${PORT_SERVE[*]}" "--spill-gib 64 --snapshot-dir /spill --spill-highwater 0.70 --spill-min-tokens 8192 --spill-min-free-gib 50"
  expect "${PORT_ENV[*]}" "TF_SPILL_FLUSH_S=60"
  expect "$STOP_TIMEOUT" 90
  [[ "$(port_identity)" =~ ^spill=v1,shard=0,sum=[0-9]+$ ]] || { port_identity; exit 1; }
  expect "$(port_summary)" " · spill v1 64 GiB"
  [[ "TF_SPILL_FLUSH_S=60" =~ ^($PORT_MAPPED_RE)= ]] || exit 1'
case_ "spill v1 settings reach the flags" 0 'SPILL_GIB=8 SPILL_HIGHWATER=1 SPILL_MIN_TOKENS=4096 SPILL_MIN_FREE_GIB=20 SPILL_FLUSH_S=5' '
  port_validate 8 8; port_map
  expect "${PORT_SERVE[*]} ${PORT_ENV[*]} $STOP_TIMEOUT" "--spill-gib 8 --snapshot-dir /spill --spill-highwater 1 --spill-min-tokens 4096 --spill-min-free-gib 20 TF_SPILL_FLUSH_S=5 35"'
case_ "ENGINE=python: spill v1 without PROMPT_CACHE_GIB (it keeps states in the pool)" 0 'ENGINE=python SPILL_GIB=64' '
  port_validate "" ""; port_map
  expect "${PORT_SERVE[*]}" "--spill-gib 64 --snapshot-dir /spill --spill-highwater 0.70 --spill-min-tokens 8192 --spill-min-free-gib 50"'
case_ "ENGINE=python: PROMPT_CACHE_GIB above 0 warns that it does nothing" 0 'ENGINE=python PROMPT_CACHE_GIB=8' '
  expect_out "WARN: PROMPT_CACHE_GIB=8 does nothing with ENGINE=python" port_validate "" ""'
case_ "refuse: spill with MTP drafts (zig: no prompt cache)" 0 'SPILL_GIB=64 DRAFTER=mtp PARALLEL=1 PROMPT_CACHE_GIB=8' '
  expect_die "DRAFTER=mtp" port_validate 1 8'
case_ "refuse: a relative SPILL_DIR, SPILL_HIGHWATER past 1, SPILL_MIN_TOKENS 8k, SPILL_GIB abc" 0 'PROMPT_CACHE_GIB=8' '
  ( SPILL_GIB=8 SPILL_DIR=spill; PORT_SPILL_EXPLICIT=1; expect_die "SPILL_DIR must be an absolute path" port_validate 8 8 ) || exit 1
  ( SPILL_GIB=8 SPILL_HIGHWATER=1.5; expect_die SPILL_HIGHWATER port_validate 8 8 ) || exit 1
  ( SPILL_GIB=8 SPILL_MIN_TOKENS=8k; expect_die SPILL_MIN_TOKENS port_validate 8 8 ) || exit 1
  ( SPILL_GIB=abc; expect_die "SPILL_GIB is a number" port_validate 8 8 ) || exit 1'
case_ "SPILL_GIB=0.0 is off" 0 'SPILL_GIB=0.0' '
  port_validate 8 0; port_map; expect "${#PORT_SERVE[@]} $(port_identity) ${STOP_TIMEOUT:-unset}" "0 off unset"'
case_ "extra serve arguments: --spill-* and --snapshot-dir refused, --prompt-cache-gib accepted" 0 '' '
  port_check_extra --prompt-cache-gib 8 --reasoning-effort high || exit 1
  expect_die "the extra argument --spill-gib" port_check_extra --spill-gib 8
  expect_die "the extra argument --snapshot-dir=/x" port_check_extra --snapshot-dir=/x
  expect_die "the extra argument --snapshot-dir" port_check_extra --snapshot-dir /x'

# ---- spill v2
case_ "refuse: SPILL_V2=1 without ZIG_HAS spill2, naming the missing patch" 0 'SPILL_GIB=64 SPILL_V2=1 PROMPT_CACHE_GIB=8' '
  expect_die "SPILL_V2=1: this image.s tensorfold-native has no the spill v2 port (Robert.s GLM patches 0110/0111).*ZIG_HAS" port_validate 8 8'
case_ "refuse: TF_GLM_SPILL_V2=1 exported counts as explicit" 0 'SPILL_GIB=64 PROMPT_CACHE_GIB=8 TF_GLM_SPILL_V2=1' '
  expect "$PORT_V2_EXPLICIT $SPILL_V2" "1 1"; expect_die "SPILL_V2=1" port_validate 8 8'
case_ "refuse: SPILL_V2=1 with ENGINE=python (patches/ has v1 only)" 0 'ENGINE=python SPILL_GIB=64 SPILL_V2=1' '
  expect_die "Python engine.s patches/ have no the spill v2 port" port_validate "" ""'
case_ "spill v2: the flags (no --spill-highwater), the Python fork's variables, the engine's tuning as set" 0 '
  ZIG_HAS="mtp displaykv spill2" SPILL_GIB=64 SPILL_V2=1 SPILL_QUOTA=0.25 SPILL_TURNS=2 PROMPT_CACHE_GIB=8
  TF_GLM_SPILL_READ_BLOCKS=32 TF_GLM_SPILL_STAGE_MIB=256 TF_GLM_SPILL_LOAD_MS=40 TF_SPILL_DEFER_MS=500' '
  port_validate 8 8; port_map
  expect "${PORT_SERVE[*]}" "--spill-gib 64 --snapshot-dir /spill --spill-min-tokens 8192 --spill-min-free-gib 50"
  expect "${PORT_ENV[*]}" "TF_SPILL_FLUSH_S=60 TF_GLM_SPILL_V2=1 TF_GLM_SPILL_QUOTA=0.25 TF_GLM_SPILL_TURNS=2 TF_GLM_SPILL_READ_BLOCKS=32 TF_GLM_SPILL_STAGE_MIB=256 TF_GLM_SPILL_LOAD_MS=40 TF_SPILL_DEFER_MS=500"
  for v in TF_GLM_SPILL_V2 TF_GLM_SPILL_QUOTA TF_GLM_SPILL_TURNS TF_GLM_SPILL_READ_BLOCKS TF_GLM_SPILL_STAGE_MIB TF_GLM_SPILL_LOAD_MS; do
    [[ "$v=x" =~ ^($PORT_MAPPED_RE)= ]] || { echo "not mapped: $v"; exit 1; }
  done
  [[ "$(port_identity)" =~ ^spill=v2,shard=0,sum= ]] || exit 1
  expect "$(port_summary)" " · spill v2 64 GiB"'
case_ "spill v2 defaults: no quota, 4 turns" 0 'ZIG_HAS=spill2 SPILL_GIB=8 SPILL_V2=1 PROMPT_CACHE_GIB=8' '
  port_validate 8 8; port_map; expect "${PORT_ENV[*]}" "TF_SPILL_FLUSH_S=60 TF_GLM_SPILL_V2=1 TF_GLM_SPILL_QUOTA=0 TF_GLM_SPILL_TURNS=4"'
case_ "aliases: TF_GLM_SPILL_QUOTA / _TURNS alone are taken; with both set the recipe name wins, with a warning" 0 '
  TF_GLM_SPILL_QUOTA=0.3 TF_GLM_SPILL_TURNS=8' '
  expect "$SPILL_QUOTA $SPILL_TURNS" "0.3 8"
  SPILL_QUOTA=0.5; expect_out "WARN: TF_GLM_SPILL_QUOTA=0.3 is overridden by SPILL_QUOTA=0.5" port_validate 8 0'
case_ "refuse: SPILL_QUOTA 1.5, SPILL_TURNS all, READ_BLOCKS 0, STAGE_MIB 1M, LOAD_MS 0, DEFER_MS 1.5" 0 'ZIG_HAS=spill2 SPILL_GIB=8 SPILL_V2=1' '
  ( SPILL_QUOTA=1.5; expect_die "SPILL_QUOTA .TF_GLM_SPILL_QUOTA. is a share" port_validate 8 8 ) || exit 1
  ( SPILL_TURNS=all; expect_die SPILL_TURNS port_validate 8 8 ) || exit 1
  ( TF_GLM_SPILL_READ_BLOCKS=0; expect_die TF_GLM_SPILL_READ_BLOCKS port_validate 8 8 ) || exit 1
  ( TF_GLM_SPILL_STAGE_MIB=1M; expect_die TF_GLM_SPILL_STAGE_MIB port_validate 8 8 ) || exit 1
  ( TF_GLM_SPILL_LOAD_MS=0; expect_die TF_GLM_SPILL_LOAD_MS port_validate 8 8 ) || exit 1
  ( TF_SPILL_DEFER_MS=1.5; expect_die TF_SPILL_DEFER_MS port_validate 8 8 ) || exit 1'
case_ "SPILL_QUOTA forms: 0, 1, .25, 0.25, 1.0 pass" 0 'ZIG_HAS=spill2 SPILL_GIB=8 SPILL_V2=1' '
  for q in 0 1 .25 0.25 1.0; do ( SPILL_QUOTA=$q; port_validate 8 8 >/dev/null ) || { echo "refused $q"; exit 1; }; done'
case_ "v2 settings without the v2 tier warn and map nothing" 0 'SPILL_QUOTA=0.25 TF_GLM_SPILL_LOAD_MS=40' '
  expect_out "WARN: SPILL_QUOTA TF_GLM_SPILL_LOAD_MS do nothing without the v2 spill tier" port_validate 8 0
  port_map; expect "${#PORT_ENV[@]}" 0'
case_ "SPILL_V2=1 without SPILL_GIB warns" 0 'SPILL_V2=1' '
  expect_out "WARN: SPILL_V2=1 does nothing without SPILL_GIB" port_validate 8 0'

# ---- KV sharding
case_ "refuse: KV_SHARD=1 without ZIG_HAS kvshard, naming the missing patch" 0 'KV_SHARD=1' '
  expect_die "KV_SHARD=1: this image.s tensorfold-native has no the KV-shard port (Robert.s GLM patch 0118)" port_validate 8 0'
case_ "refuse: TF_GLM_KV_SHARD=1 exported counts as explicit" 0 'TF_GLM_KV_SHARD=1' '
  expect "$PORT_SHARD_EXPLICIT $KV_SHARD" "1 1"; expect_die "KV-shard port" port_validate 8 0'
case_ "KV shard: the variables, carry and twins on by default, the check passed only when set" 0 'ZIG_HAS=kvshard KV_SHARD=1' '
  port_validate 8 0; port_map
  expect "${PORT_ENV[*]} ${#PORT_SERVE[@]}" "TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=1 TF_GLM_KV_SHARD_TWINS=1 0"
  [[ "TF_GLM_KV_SHARD_CARRY=x" =~ ^($PORT_MAPPED_RE)= ]] || exit 1
  [[ "$(port_identity)" =~ ^spill=0,shard=1,sum= ]] || exit 1
  expect "$(port_summary)" " · KV shard"
  TF_GLM_KV_SHARD_CARRY=0 TF_GLM_KV_SHARD_CHECK=1024; port_validate 8 0; port_map
  expect "${PORT_ENV[*]}" "TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=0 TF_GLM_KV_SHARD_TWINS=1 TF_GLM_KV_SHARD_CHECK=1024"'
case_ "refuse: KV_SHARD=1 at TP=3, --parallel 1, DRAFTER=mtp, TF_GLM_MULTI_LONE=1, ENGINE=python" 0 'ZIG_HAS=kvshard KV_SHARD=1' '
  ( TP=3; expect_die "two Sparks .TP=2., not TP=3" port_validate 8 0 ) || exit 1
  expect_die "more than one stream .PARALLEL above 1., not --parallel 1" port_validate 1 0
  ( DRAFTER=mtp; expect_die "DRAFTER=mtp" port_validate 8 0 ) || exit 1
  ( TF_GLM_MULTI_LONE=1; expect_die "TF_GLM_MULTI_LONE=1" port_validate 8 0 ) || exit 1
  ( ENGINE=python; expect_die "Python engine.s patches/ have no the KV-shard port" port_validate "" "" ) || exit 1
  port_validate 16 0 >/dev/null || exit 1'
case_ "refuse: KV_SHARD=2, TF_GLM_KV_SHARD_CARRY=yes, TF_GLM_KV_SHARD_CHECK=x" 0 'ZIG_HAS=kvshard' '
  ( KV_SHARD=2; expect_die "KV_SHARD .TF_GLM_KV_SHARD. is 0 or 1" port_validate 8 0 ) || exit 1
  ( TF_GLM_KV_SHARD_CARRY=yes; expect_die TF_GLM_KV_SHARD_CARRY port_validate 8 0 ) || exit 1
  ( TF_GLM_KV_SHARD_CHECK=x; expect_die TF_GLM_KV_SHARD_CHECK port_validate 8 0 ) || exit 1'
case_ "KV-shard settings without KV_SHARD=1 warn (and pass to the ranks as before)" 0 'TF_GLM_KV_SHARD_TWINS=0' '
  expect_out "WARN: TF_GLM_KV_SHARD_TWINS do nothing without KV_SHARD=1" port_validate 8 0
  port_map; expect "${#PORT_ENV[@]}" 0; [[ ! "TF_GLM_KV_SHARD_TWINS=0" =~ ^($PORT_MAPPED_RE)= ]] || exit 1'
case_ "KV_SHARD wins over TF_GLM_KV_SHARD, with a warning" 0 'ZIG_HAS=kvshard KV_SHARD=1 TF_GLM_KV_SHARD=0' '
  expect_out "WARN: TF_GLM_KV_SHARD=0 is overridden by KV_SHARD=1" port_validate 8 0; expect "$KV_SHARD" 1'

# ---- spill and KV sharding together
case_ "refuse: KV_SHARD=1 with spill v1 (both explicit)" 0 'ZIG_HAS="kvshard spillshard" KV_SHARD=1 SPILL_GIB=8' '
  expect_die "the v1 spill tier does not know the owner parts.*Turn one off" port_validate 8 8'
case_ "refuse: KV_SHARD=1 with spill v2 without ZIG_HAS spillshard (both explicit)" 0 'ZIG_HAS="spill2 kvshard" KV_SHARD=1 SPILL_GIB=8 SPILL_V2=1' '
  expect_die "spill v2 and KV sharding together" port_validate 8 8'
case_ "KV shard and spill v2 together with ZIG_HAS spillshard" 0 "ZIG_HAS='$ALL_HAS' KV_SHARD=1 SPILL_GIB=8 SPILL_V2=1" '
  port_validate 8 8; port_map
  expect "${PORT_ENV[*]}" "TF_SPILL_FLUSH_S=60 TF_GLM_SPILL_V2=1 TF_GLM_SPILL_QUOTA=0 TF_GLM_SPILL_TURNS=4 TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=1 TF_GLM_KV_SHARD_TWINS=1"
  [[ "$(port_identity)" =~ ^spill=v2,shard=1,sum= ]] || exit 1'

# ---- a fork's defaults (PT_FORK=1: spill v2 64 GiB, KV sharding)
case_ "fork defaults with every port in the image: spill v2 and KV shard on" 0 "PT_FORK=1 ZIG_HAS='$ALL_HAS' PROMPT_CACHE_GIB=8" '
  expect "$PORT_SPILL_EXPLICIT $PORT_V2_EXPLICIT $PORT_SHARD_EXPLICIT $SPILL_GIB $SPILL_V2 $KV_SHARD $STOP_TIMEOUT" "0 0 0 64 1 1 90"
  expect_out "!WARN" port_validate 8 8
  port_validate 8 8; port_map
  expect "${PORT_SERVE[*]}" "--spill-gib 64 --snapshot-dir /spill --spill-min-tokens 8192 --spill-min-free-gib 50"
  expect "${PORT_ENV[*]}" "TF_SPILL_FLUSH_S=60 TF_GLM_SPILL_V2=1 TF_GLM_SPILL_QUOTA=0.25 TF_GLM_SPILL_TURNS=4 TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=1 TF_GLM_KV_SHARD_TWINS=1"'
case_ "fork defaults on the stock image (no spill2, kvshard): both off with warnings, as before the port" 0 "PT_FORK=1 PROMPT_CACHE_GIB=8" '
  o=$(port_validate 8 8); grep -q "WARN: SPILL TIER OFF for this start: .*spill v2 port" <<<"$o" && grep -q "WARN: KV SHARDING OFF for this start: .*KV-shard port" <<<"$o" || { echo "$o"; exit 1; }
  port_validate 8 8 >/dev/null; port_map
  expect "${#PORT_ENV[@]} ${#PORT_SERVE[@]} $(port_identity) ${STOP_TIMEOUT:-unset}" "0 0 off unset"'
case_ "fork defaults with PROMPT_CACHE_GIB=0: the spill tier off (warned), KV shard on" 0 "PT_FORK=1 ZIG_HAS='$ALL_HAS'" '
  expect_out "WARN: SPILL TIER OFF for this start: SPILL_GIB=64 with PROMPT_CACHE_GIB=0" port_validate 8 0
  port_validate 8 0 >/dev/null; port_map; expect "${PORT_ENV[*]}" "TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=1 TF_GLM_KV_SHARD_TWINS=1"'
case_ "fork default KV shard where the Python fork had it off (TP=3, --parallel 1, MULTI_LONE): off without a word" 0 "PT_FORK=1 ZIG_HAS='$ALL_HAS' PROMPT_CACHE_GIB=8" '
  ( TP=3; expect_out "!KV SHARDING" port_validate 8 8; port_validate 8 8 >/dev/null; expect "$KV_SHARD" 0 ) || exit 1
  ( expect_out "!KV SHARDING" port_validate 1 8; port_validate 1 8 >/dev/null; expect "$KV_SHARD" 0 ) || exit 1
  ( TF_GLM_MULTI_LONE=1; port_validate 8 8 >/dev/null; expect "$KV_SHARD" 0 ) || exit 1'
case_ "fork defaults, SPILL_GIB explicit, no spill2: the tier runs as v1 (warned), the default KV shard goes off" 0 "PT_FORK=1 ZIG_HAS='mtp kvshard spillshard' SPILL_GIB=32 PROMPT_CACHE_GIB=8" '
  o=$(port_validate 8 8); grep -q "WARN: SPILL V2 OFF for this start" <<<"$o" && grep -q "WARN: KV SHARDING OFF for this start: the spill tier is on and the v1 spill tier" <<<"$o" || { echo "$o"; exit 1; }
  port_validate 8 8 >/dev/null; port_map
  expect "${PORT_SERVE[*]} ${PORT_ENV[*]}" "--spill-gib 32 --snapshot-dir /spill --spill-highwater 0.70 --spill-min-tokens 8192 --spill-min-free-gib 50 TF_SPILL_FLUSH_S=60"'
case_ "fork defaults, KV_SHARD=1 explicit, no spillshard: the default spill tier goes off (warned)" 0 "PT_FORK=1 ZIG_HAS='spill2 kvshard' KV_SHARD=1 PROMPT_CACHE_GIB=8" '
  expect_out "WARN: SPILL TIER OFF for this start: KV_SHARD=1 is set and this image" port_validate 8 8
  port_validate 8 8 >/dev/null; expect "$SPILL_GIB $KV_SHARD" "0 1"'
case_ "fork defaults, both default, no spillshard: KV shard off (warned), spill v2 kept" 0 "PT_FORK=1 ZIG_HAS='spill2 kvshard' PROMPT_CACHE_GIB=8" '
  expect_out "WARN: KV SHARDING OFF for this start: the spill tier is on" port_validate 8 8
  port_validate 8 8 >/dev/null; expect "$SPILL_GIB $SPILL_V2 $KV_SHARD" "64 1 0"'
case_ "fork defaults with ENGINE=python: both off with warnings" 0 "PT_FORK=1 ENGINE=python" '
  o=$(port_validate "" ""); grep -q "SPILL TIER OFF" <<<"$o" && grep -q "KV SHARDING OFF" <<<"$o" || { echo "$o"; exit 1; }'
case_ "fork default spill with an unusable SPILL_DIR name: off with a warning" 0 "PT_FORK=1 ZIG_HAS='$ALL_HAS' PROMPT_CACHE_GIB=8 SPILL_DIR='/a b'" '
  expect_out "WARN: SPILL TIER OFF for this start: SPILL_DIR must be an absolute path" port_validate 8 8'
case_ "fork default spill with MTP drafts: off with a warning" 0 "PT_FORK=1 ZIG_HAS='$ALL_HAS' PROMPT_CACHE_GIB=8 DRAFTER=mtp PARALLEL=1" '
  expect_out "WARN: SPILL TIER OFF for this start: SPILL_GIB=64 with DRAFTER=mtp" port_validate 1 8'

# ---- prefix wait: not ported
case_ "PREFIX_WAIT=1 / TF_GLM_PREFIX_WAIT=1 warn and do nothing; 0 is silent" 0 '' '
  ( PREFIX_WAIT=1; expect_out "WARN: PREFIX_WAIT=1 does nothing: the prefix wait .* is not ported" port_validate 8 0 ) || exit 1
  ( TF_GLM_PREFIX_WAIT=1; expect_out "WARN: TF_GLM_PREFIX_WAIT=1 does nothing" port_validate 8 0 ) || exit 1
  ( PREFIX_WAIT=0 TF_GLM_PREFIX_WAIT=0; expect_out "!WARN" port_validate 8 0 ) || exit 1
  port_map; expect "${#PORT_ENV[@]}" 0'

# ---- after the start: rank 0 opened no prompt cache, so the spill tier is not open
case_ "after the start: a prompt cache budget of 0 with the spill tier on warns; kept states, or no tier, say nothing" 0 'SPILL_GIB=8 PROMPT_CACHE_GIB=8' '
  expect_out "WARN: the spill tier is not open: rank 0 logged \"prompt cache: off (budget 0)\"" port_check_started "info: prompt cache: off (budget 0)"
  expect_out "!WARN" port_check_started "info: prompt cache: 2.0 GiB for kept states, at most 14 (fixed 102 MiB each)"
  ( SPILL_GIB=0; expect_out "!WARN" port_check_started "info: prompt cache: off (budget 0)" ) || exit 1
  ( ENGINE=python; expect_out "!WARN" port_check_started "prompt cache: off (budget 0)" ) || exit 1'

# ---- the label (tf.recipe.port)
case_ "the label: the same settings give the same word; another folder, quota or carry another" 0 "ZIG_HAS='$ALL_HAS' SPILL_GIB=8 SPILL_V2=1 KV_SHARD=1 PROMPT_CACHE_GIB=8" '
  port_validate 8 8; port_map; a=$(port_identity); port_map; expect "$(port_identity)" "$a"
  ( SPILL_DIR=/data/spill; port_map; [[ "$(port_identity)" != "$a" ]] ) || exit 1
  ( SPILL_QUOTA=0.25; port_map; [[ "$(port_identity)" != "$a" ]] ) || exit 1
  ( TF_GLM_KV_SHARD_CARRY=0; port_map; [[ "$(port_identity)" != "$a" ]] ) || exit 1
  [[ "$a" =~ ^[A-Za-z0-9=,.]+$ ]] || { echo "not one word: $a"; exit 1; }'

# ---- the spill folder: port_spill_judge on synthetic probe lines (explicit SPILL_GIB: refusals die)
G=1073741824
case_ "judge: a local, writable folder with room passes" 0 'SPILL_GIB=8' "
  port_spill_judge here 'OK 1 ext4 $((500 * G)) 0 - 0 /data/spill' || exit 1
  expect_out '!WARN' port_spill_judge here 'OK 1 ext4 $((500 * G)) 0 - 0 /data/spill'"
case_ "judge: refusals (network, memory, export, overlap, home, not writable, no room, ERR)" 0 'SPILL_GIB=8' "
  expect_die 'network file system (nfs4)' port_spill_judge w1 'OK 1 nfs4 $((500 * G)) 0 - 0 /mnt/spill'
  expect_die 'on tmpfs' port_spill_judge w1 'OK 1 tmpfs $((500 * G)) 0 - 0 /dev/shm/spill'
  expect_die 'inside an NFS export' port_spill_judge here 'OK 1 ext4 $((500 * G)) 1 - 0 /home/u/.cache/huggingface/spill'
  expect_die 'overlaps /home/u/.cache/huggingface' port_spill_judge here 'OK 1 ext4 $((500 * G)) 0 /home/u/.cache/huggingface 0 /home/u/.cache/huggingface/s'
  expect_die '/ or your home folder' port_spill_judge here 'OK 1 ext4 $((500 * G)) 0 - 1 /home/u'
  expect_die 'not writable by you' port_spill_judge here 'OK 0 ext4 $((500 * G)) 0 - 0 /data/spill'
  expect_die 'you cannot create it' port_spill_judge here 'NEW 0 ext4 $((500 * G)) 0 - 0 /data/spill'
  expect_die 'under SPILL_MIN_FREE_GIB' port_spill_judge here 'OK 1 ext4 $((40 * G)) 0 - 0 /data/spill'
  expect_die 'SPILL_DIR on w1: cannot create /x' port_spill_judge w1 'ERR cannot create /x'"
case_ "judge: less room than SPILL_GIB + the floor warns; a folder to create is logged" 0 'SPILL_GIB=64' "
  expect_out 'WARN: .*less than SPILL_GIB (64) + SPILL_MIN_FREE_GIB' port_spill_judge here 'OK 1 ext4 $((100 * G)) 0 - 0 /data/spill'
  expect_out 'LOG: SPILL_DIR /data/spill on here does not exist yet' port_spill_judge here 'NEW 1 ext4 $((500 * G)) 0 - 0 /data/spill'"
case_ "judge: a default tier's failure returns 1 with the reason (no refusal)" 0 "PT_FORK=1" "
  port_spill_judge w1 'OK 1 nfs $((500 * G)) 0 - 0 /mnt/spill' && exit 1
  [[ \"\$PORT_SPILL_WHY\" == *'network file system (nfs)'* ]] || { echo \"\$PORT_SPILL_WHY\"; exit 1; }"

# ---- spill_dir_probe on temporary folders (this machine: Linux or a Mac)
mkdir -p "$tmp/hf/hub" "$tmp/ro"
tmpr=$(cd -P "$tmp" && pwd -P)
case_ "probe: creates a missing folder (create=1) and describes it" 0 '' "
  l=\$(spill_dir_probe 1 '$tmp/a/spill' '$tmp/hf'); [[ -d '$tmp/a/spill' ]] || exit 1
  read -r st w fs avail exp over top real <<<\"\$l\"
  expect \"\$st \$w \$exp \$over \$top \$real\" 'OK 1 0 - 0 $tmpr/a/spill'
  [[ \"\$avail\" =~ ^[0-9]+\$ && \"\$avail\" -gt 0 ]] || { echo \"\$l\"; exit 1; }"
case_ "probe: DRY_RUN (create=0) creates nothing and says NEW" 0 '' "
  l=\$(spill_dir_probe 0 '$tmp/b/spill'); [[ ! -e '$tmp/b' ]] || exit 1
  read -r st w fs avail exp over top real <<<\"\$l\"; expect \"\$st \$w \$real\" 'NEW 1 $tmpr/b/spill'"
case_ "probe: a folder inside (or around) the Hugging Face cache overlaps it" 0 '' "
  l=\$(spill_dir_probe 1 '$tmp/hf/spill' '$tmp/hf'); read -r st w fs avail exp over top real <<<\"\$l\"; expect \"\$over\" '$tmpr/hf'
  l=\$(spill_dir_probe 1 '$tmp' '$tmp/hf'); read -r st w fs avail exp over top real <<<\"\$l\"; expect \"\$over\" '$tmpr/hf'"
case_ "probe: the same line through sh -c (a worker's login shell may be sh)" 0 '' "
  expect \"\$(sh -c \"\$(declare -f spill_dir_probe); spill_dir_probe 1 '$tmp/s/spill' '$tmp/hf'\" | cut -d' ' -f1,2,5-)\" 'OK 1 0 - 0 $tmpr/s/spill'"
case_ "probe: the home folder itself" 0 '' "
  HOME='$tmp/a'; l=\$(spill_dir_probe 1 '$tmp/a'); read -r st w fs avail exp over top real <<<\"\$l\"; expect \"\$top\" 1"
if [[ "$(id -u)" != 0 ]]; then
  chmod 555 "$tmp/ro"
  case_ "probe: a folder that cannot be created is an error; an unwritable one says so" 0 '' "
    l=\$(spill_dir_probe 1 '$tmp/ro/spill'); [[ \"\$l\" == 'ERR cannot create $tmp/ro/spill' ]] || { echo \"\$l\"; exit 1; }
    l=\$(spill_dir_probe 1 '$tmp/ro'); read -r st w rest <<<\"\$l\"; expect \"\$st \$w\" 'OK 0'"
fi

# ---- port_spill_check: this Spark and each worker (the worker's ssh command run here, through bash -c)
stubs="worker_ids() { echo 1; }; wvar() { if (( \$2 == 1 )); then echo \"\$1\"; else echo \"\$1\$2\"; fi; }
  wval() { local v; v=\$(wvar \"\$1\" \"\$2\"); echo \"\${!v:-}\"; }; worker_host() { echo w1; }; WORKER_DOWN=()
  worker() { shift; bash -c \"\$1\"; }"
case_ "spill check: both Sparks' folders created and judged, the worker's through its ssh command" 0 "SPILL_GIB=0.001 SPILL_MIN_FREE_GIB=0 PROMPT_CACHE_GIB=8 SPILL_DIR='$tmp/c/spill' HF_CACHE='$tmp/hf' NFS_PATH='$tmp/hf'" "$stubs
  expect_out 'LOG: Spill tier: v1, up to 0.001 GiB a Spark in $tmp/c/spill' port_spill_check
  [[ -d '$tmp/c/spill' ]] || exit 1"
case_ "spill check: refuse a worker folder overlapping its WORKER_HF_CACHE" 0 "SPILL_GIB=0.001 SPILL_MIN_FREE_GIB=0 SPILL_DIR='$tmp/d/spill' HF_CACHE='$tmp/hf' WORKER_HF_CACHE='$tmp/d'" "$stubs
  expect_die 'SPILL_DIR $tmpr/d/spill on w1 overlaps $tmpr/d' port_spill_check"
case_ "spill check: DRY_RUN creates nothing" 0 "DRY_RUN=1 SPILL_GIB=0.001 SPILL_MIN_FREE_GIB=0 SPILL_DIR='$tmp/e/spill' HF_CACHE='$tmp/hf'" "$stubs
  expect_out 'does not exist yet' port_spill_check; [[ ! -e '$tmp/e' ]] || exit 1"
case_ "spill check: a worker that cannot be checked refuses an explicit tier" 0 "SPILL_GIB=0.001 SPILL_MIN_FREE_GIB=0 SPILL_DIR='$tmp/f/spill' HF_CACHE='$tmp/hf'" "$stubs
  worker() { return 255; }; expect_die 'could not check SPILL_DIR on w1' port_spill_check"
case_ "spill check: a fork's default tier whose worker folder fails goes off everywhere (warned)" 0 "PT_FORK=1 ZIG_HAS='$ALL_HAS' PROMPT_CACHE_GIB=8 SPILL_MIN_FREE_GIB=0 SPILL_DIR='$tmp/g/spill' HF_CACHE='$tmp/hf' WORKER_HF_CACHE='$tmp/g'" "$stubs
  port_validate 8 8 >/dev/null
  expect_out 'WARN: SPILL TIER OFF for this start: SPILL_DIR .* on w1 overlaps' port_spill_check
  port_spill_check >/dev/null; port_map; expect \"\$SPILL_GIB \${STOP_TIMEOUT:-unset}\" '0 unset'
  expect \"\${PORT_ENV[*]}\" 'TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=1 TF_GLM_KV_SHARD_TWINS=1'"

# ---- scripts/memavail.sh: the start check counts the pages parked on the per-CPU free lists, as the engine's 0119
# does (native/cuda_memory.zig parkedBytes): its Zig tests' synthetic /proc/zoneinfo cases, as in the DeepSeek recipe
# zone <node> <name> <high_min or -> [counts...]: one zone's block as the kernel prints it (no counts: no pagesets)
zone() {
  local node=$1 name=$2 hm=$3 c n=0; shift 3
  printf 'Node %s, zone %8s\n  pages free     10\n        managed  1000\n      nr_free_cma  0\n' "$node" "$name"
  if (( $# )); then
    printf '  pagesets\n'
    for c in "$@"; do
      printf '    cpu: %s\n              count:    %s\n              high:     5000\n              batch:    63\n' "$n" "$c"
      [[ "$hm" == - ]] || printf '              high_min: %s\n' "$hm"
      printf '              high_max: 198283\n  vm stats threshold: 110\n'
      n=$((n + 1))
    done
  fi
  printf '  start_pfn:           4096\n'
}
mt=$tmp/mem; mkdir -p "$mt"
printf 'MemTotal:       %s kB\nMemFree:        1000 kB\nMemAvailable:   %s kB\n' $((128 * 1048576)) $((100 * 1048576)) > "$mt/meminfo"
printf 'MemTotal:       %s kB\nMemAvailable:   %s kB\n' $((128 * 1048576)) $((125 * 1048576)) > "$mt/meminfo-high"
printf 'MemTotal:       1 kB\n' > "$mt/meminfo-none"
{ zone 0 DMA 2146 200000 3143 2332 17; zone 0 DMA32 2146; zone 0 Normal 2146 100000 100000 100000; zone 1 Normal 2146 400000; zone 1 Movable 2146; } > "$mt/z-many"
six=$(( (6 << 30) / 4096 / 4 + 2146 ))
zone 0 Normal 2146 $six $six $six $six > "$mt/z-six"
zone 0 Normal 2146 $(for i in $(seq 20); do printf '%s ' $((2146 + 1440)); done) > "$mt/z-warm"
zone 0 Normal 2146 $(for i in $(seq 20); do printf '100 '; done) > "$mt/z-below"
half=$(( ((1 << 30) + (1 << 29)) / 4096 / 2 ))
zone 0 Normal - $half $half > "$mt/z-old"
zone 0 Normal - 1000 1000 1000 1000 > "$mt/z-old-small"
{ printf '  count: 5\n      count: 6\n'; zone 0 Normal 2146 $((2146 + 10))
  printf '      count:    99999\n      high_min: 7\n'
  printf 'Node 0, zone Other\n      count:    77777\n  pagesets\n    cpu: 0\n              high:     1\n'; } > "$mt/z-only"
{ zone 0 Normal 2146 500000; printf '  pagesets\n    cpu: 0\n              count:    abc\n'; } > "$mt/z-bad"
printf '  pagesets\n    cpu: 0\n              count:    600000\r\n              high_min: 2146\r\n' > "$mt/z-crlf"
# mc <name> <meminfo> <zoneinfo> <want: avail GiB, parked GiB, parked bytes>
mc() { case_ "parked pages: $1" 0 "source '$here/memavail.sh'" "expect \"\$(mem_counting '$mt/$2' '$mt/$3' 4096)\" '$4'"; }
many=$(( (200000 + 3143 + 2332 + 17 + 300000 + 400000 - 8 * 2146) * 4096 ))
mc "every CPU, zone and node, less each list's resting level (count - high_min)" meminfo z-many "$(awk -v b=$many 'BEGIN {printf "%.1f %.1f %.0f", 100 + b / 2^30, b / 2^30, b}')"
mc "an image copy's 6 GiB raise the budget by 6 GiB" meminfo z-six "106.0 6.0 6442450944"
mc "a warm start counts ~nothing (under 0.25 GiB)" meminfo z-warm "$(awk -v b=$((20 * 1440 * 4096)) 'BEGIN {printf "%.1f %.1f %.0f", 100 + b / 2^30, b / 2^30, b}')"
mc "below the resting level: nothing" meminfo z-below "100.0 0.0 0"
mc "no high_min (before 6.7) rests the first 512 MiB" meminfo z-old "101.0 1.0 1073741824"
mc "no high_min, under 512 MiB: nothing" meminfo z-old-small "100.0 0.0 0"
mc "only the count: and high_min: lines under pagesets" meminfo z-only "100.0 0.0 40960"
mc "an unparsable count counts nothing" meminfo z-bad "100.0 0.0 0"
mc "carriage returns are blanks" meminfo z-crlf "$(awk -v b=$(( (600000 - 2146) * 4096 )) 'BEGIN {printf "%.1f %.1f %.0f", 100 + b / 2^30, b / 2^30, b}')"
mc "capped at MemTotal" meminfo-high z-six "128.0 6.0 6442450944"
mc "no zoneinfo: MemAvailable alone" meminfo missing "100.0 0.0 0"
case_ "parked pages: no MemAvailable fails and prints nothing" 0 "source '$here/memavail.sh'" "
  out=\$(mem_counting '$mt/meminfo-none' '$mt/z-many' 4096) && exit 1; [[ -z \"\$out\" ]]"
case_ "parked pages: the page size from getconf by default" 0 "source '$here/memavail.sh'" "
  expect \"\$(mem_counting '$mt/meminfo' '$mt/z-six')\" \"\$(mem_counting '$mt/meminfo' '$mt/z-six' \$(getconf PAGESIZE))\""
case_ "start check: mem_avail_gb is whole GiB (rounded down) and the parked GiB" 0 "source '$here/memavail.sh'" "
  mem_counting() { echo '109.9 6.0 6442450944'; }; expect \"\$(mem_avail_gb)\" '109 6.0'"
case_ "start check: mem_avail_gb on the synthetic Spark (100 GiB available, 6 GiB parked)" 0 "source '$here/memavail.sh'" "
  eval \"\$(declare -f mem_counting | sed '1s/mem_counting/mem_counting_file/')\"
  mem_counting() { mem_counting_file '$mt/meminfo' '$mt/z-six' 4096; }
  expect \"\$(mem_avail_gb)\" '106 6.0'"
case_ "start check: mem_avail_gb runs on a worker as start.sh sends it (declare -f over ssh, here bash -c and sh -c)" 0 "source '$here/memavail.sh'" "
  for sh in bash sh; do
    expect \"\$(\$sh -c \"\$(declare -f mem_counting); mem_counting '$mt/meminfo' '$mt/z-six' 4096\")\" '106.0 6.0 6442450944'
  done
  l=\$(bash -c \"\$(declare -f mem_counting mem_avail_gb); mem_avail_gb\")
  if [[ -r /proc/meminfo ]]; then [[ \"\$l\" =~ ^[0-9]+\\ [0-9]+\\.[0-9]\$ ]] || { echo \"[\$l]\"; exit 1; }; fi"
if [[ -r /proc/meminfo ]]; then
  case_ "parked pages: this host's /proc/meminfo and /proc/zoneinfo" 0 "source '$here/memavail.sh'" "
    read -r a p b <<<\"\$(mem_counting)\"; [[ \"\$a\" =~ ^[0-9]+\\.[0-9]\$ && \"\$b\" =~ ^[0-9]+\$ ]] || { echo \"[\$a \$p \$b]\"; exit 1; }"
fi

# ---- every script parses (bash -n, with this bash)
for f in "$root"/start.sh "$root"/start-tp3.sh "$root"/start-tp4.sh "$root"/start-tp4-switchless.sh "$root"/stop.sh "$here"/*.sh; do
  [[ -f "$f" ]] || continue
  n=$((n + 1))
  if out=$(bash -n "$f" 2>&1); then printf 'ok   bash -n %s\n' "${f#$root/}"
  else fails=$((fails + 1)); printf 'FAIL bash -n %s\n%s\n' "${f#$root/}" "$out"; fi
done

echo "$((n - fails)) of $n passed (bash $BASH_VERSION)"
(( fails == 0 ))
