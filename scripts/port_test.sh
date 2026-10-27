#!/usr/bin/env bash
# Checks of scripts/port.sh and scripts/memavail.sh without the Sparks: the defaults (everything off: no flag, variable,
# mount or label), a fork's defaults turned on (PT_FORK=1: spill v2 64 GiB, KV sharding) degrading where an explicit
# setting refuses, the ZIG_HAS gates (spill2, kvshard, spillshard), PROMPT_CACHE_GIB as the spill tier's requirement,
# the aliases, the mapping to the engine's flags and variables, the refusals (port_validate), the extra serve arguments
# (port_check_extra), the spill folder rules (port_spill_judge on synthetic probe lines, spill_dir_probe and
# port_spill_check on temporary folders, the worker's probe through its ssh command run locally, the free-disk floor with
# the tiers' own folders counted, df stubbed), every rank's tier after the start (port_check_started), the parked pages
# on synthetic /proc files and their gate (parked_counted, mem_avail_gb), scripts/port-patches.sh's README rewrite and
# helpers (its runs against the engine: scripts/port_patches_test.sh), and `bash -n` of every script.
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
# start.sh's default ZIG_HAS (the image with zig/patches/0014-0030), and an image without the port's patches
DEF_HAS="mtp displaykv parked spill2 kvshard spillshard mtpcache fill pictures prefixwait"
NO_PORT="mtp displaykv"

# case_ <name> <expected exit 0|1> <script before port.sh> <script after>: runs both in a subshell over a fresh port.sh,
# with the serving settings of the two-Spark Zig default (TP 2, PARALLEL 8, DFlash2, PROMPT_CACHE_GIB 0, KV_POOL_GIB
# at its default 12.5, not set (PORT_POOL_EXPLICIT 0), ZIG_HAS as start.sh's default: $DEF_HAS) and nothing of the
# caller's environment; die/warn/log stubbed (die exits 1)
case_() {
  local name=$1 want=$2 got out
  n=$((n + 1))
  out=$( (
    for v in $(compgen -v | grep -E '^(SPILL_|KV_SHARD|KV_POOL_GIB|PORT_|TF_GLM_|TF_SPILL_|PREFIX_WAIT|STOP_TIMEOUT|PROMPT_CACHE_GIB|ZIG_HAS|WORKER|FILL$)'); do
      unset "$v"
    done
    ENGINE=zig TP=2 PARALLEL=8 DRAFTER=dflash2 TF_GLM_MULTI_LONE=0 PROMPT_CACHE_GIB=0 KV_POOL_GIB=12.5 ZIG_HAS="$DEF_HAS"
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

# ---- spill v2's identity (port_spill_identity): the engine refuses SPILL_V2=1 without it
ID_STUBS='image_hash() { echo 0123456789ab; }; _cfg_root=$tmp/root; mkdir -p "$tmp/root/zig/patches"; printf x > "$tmp/root/zig/patches/0014-a.patch"
  ZIG_TF_REF=e7c523ebfa16b8094cf9e35e3773a7d25befcc5b MODEL_ID=m/glm MODEL_REVISION=078455ff DFLASH2_ID=d/df DFLASH2_REVISION=bf582e4e'
case_ "identity: SPILL_V2=1 passes the engine, series, model and drafter on every rank" 0 "tmp=$tmp; $ID_STUBS"' SPILL_GIB=64 PROMPT_CACHE_GIB=8 SPILL_V2=1 ZIG_HAS="$ALL_HAS"' '
  port_validate 8 8; port_map; port_spill_identity
  expect "${PORT_ID_ENV[*]}" "TF_GLM_ENGINE_ID=tensorfold-e7c523ebfa16+0123456789ab TF_GLM_RECIPE_PATCHES=sha256:$(printf x | sha256sum | cut -c1-16) TF_GLM_MODEL_ID=m/glm TF_GLM_MODEL_REVISION=078455ff TF_GLM_DFLASH2_ID=d/df TF_GLM_DFLASH2_REVISION=bf582e4e"
  [[ "TF_GLM_ENGINE_ID" =~ ^($PORT_MAPPED_RE)$ ]] || { echo "TF_GLM_ENGINE_ID not in PORT_MAPPED_RE"; exit 1; }'
case_ "identity: none with v1, with the spill off, or with the Python engine" 0 "tmp=$tmp; $ID_STUBS"' ZIG_HAS="$ALL_HAS"' '
  SPILL_GIB=64 PROMPT_CACHE_GIB=8 SPILL_V2=0; port_validate 8 8; port_map; port_spill_identity; expect "${#PORT_ID_ENV[@]}" 0
  SPILL_GIB=0 SPILL_V2=1; port_map; port_spill_identity; expect "${#PORT_ID_ENV[@]}" 0
  ENGINE=python SPILL_GIB=64 SPILL_V2=1; port_spill_identity; expect "${#PORT_ID_ENV[@]}" 0'
case_ "identity: SPILL_V2=1 refuses an unpinned checkpoint or drafter" 0 "tmp=$tmp; $ID_STUBS"' SPILL_GIB=64 PROMPT_CACHE_GIB=8 SPILL_V2=1 ZIG_HAS="$ALL_HAS"' '
  port_validate 8 8; port_map
  MODEL_REVISION= expect_die "pinned MODEL_REVISION" port_spill_identity
  DFLASH2_REVISION= expect_die "pinned DFLASH2_REVISION" port_spill_identity
  DRAFTER=mtp DFLASH2_REVISION= port_spill_identity'

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
case_ "defaults: the ranks' passthrough drops only the switches (TF_GLM_KV_SHARD, TF_GLM_SPILL_V2, TF_GLM_FILL, TF_GLM_PREFIX_WAIT and its bound)" 0 '' '
  port_validate 8 0; port_map
  expect "$(printf "%s\n" TF_GLM_KV_SHARD=0 TF_GLM_FILL=1 TF_GLM_PREFIX_WAIT=1 TF_GLM_PREFIX_WAIT_MS=9 TF_GLM_KV_SHARD_CARRY=0 TF_GLM_SPILL_V2=1 TF_GLM_SPILL_QUOTA=0.25 TF_GLM_CACHE_GIB=12.5 TF_SPILL_FLUSH_S=60 |
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
  expect_out "WARN: PROMPT_CACHE_GIB=8 does nothing with ENGINE=python" port_validate "" ""
  expect_out "!KV_POOL_GIB at its default" port_validate "" ""'
# The pool is sized first: at the default KV_POOL_GIB the kept-state budget came out 0 on the pair (g1)
case_ "PROMPT_CACHE_GIB (or --prompt-cache-gib) above 0 with KV_POOL_GIB not set warns: at the default pool the budget is 0" 0 'PROMPT_CACHE_GIB=8' '
  expect_out "WARN: PROMPT_CACHE_GIB=8 with KV_POOL_GIB at its default (12.5): .*prompt cache: off (budget 0)" port_validate 8 8
  expect_out "KV_POOL_GIB=4 with PROMPT_CACHE_GIB=8: a 6.2 GiB budget for kept states, 5.1 GiB with spill v2 on" port_validate 8 8
  PROMPT_CACHE_GIB=0; expect_out "WARN: PROMPT_CACHE_GIB=16 with KV_POOL_GIB at its default (12.5)" port_validate 8 16
  ( TP=3 KV_POOL_GIB=32; expect_out "WARN: PROMPT_CACHE_GIB=8 with KV_POOL_GIB at its default (32)" port_validate 8 8 ) || exit 1'
case_ "no pool warning with KV_POOL_GIB set (even to 12.5), no cache, MTP drafts on an image that keeps no state with them, or ENGINE=python" 0 'PROMPT_CACHE_GIB=8' '
  ( PORT_POOL_EXPLICIT=1 KV_POOL_GIB=4; expect_out "!KV_POOL_GIB at its default" port_validate 8 8 ) || exit 1
  ( PORT_POOL_EXPLICIT=1; expect_out "!KV_POOL_GIB at its default" port_validate 8 8 ) || exit 1
  ( PROMPT_CACHE_GIB=0; expect_out "!WARN" port_validate 8 0 ) || exit 1
  ( DRAFTER=mtp ZIG_HAS=$NO_PORT; expect_out "!KV_POOL_GIB at its default" port_validate 1 8 ) || exit 1
  ( ENGINE=python; expect_out "!KV_POOL_GIB at its default" port_validate "" 8 ) || exit 1'
# MTP drafts keep prompt states where the engine has the port (ZIG_HAS mtpcache: kept states with the MTP head's section)
case_ "MTP drafts with ZIG_HAS mtpcache keep prompt states (the pool warning as with DFlash2); without it, a warning that none is kept" 0 'PROMPT_CACHE_GIB=8 DRAFTER=mtp PARALLEL=1' '
  ( expect_out "WARN: PROMPT_CACHE_GIB=8 with KV_POOL_GIB at its default (12.5)" port_validate 1 8
    expect_out "!keeps no prompt state" port_validate 1 8
    PORT_POOL_EXPLICIT=1 KV_POOL_GIB=4; expect_out "!WARN" port_validate 1 8 ) || exit 1
  ( ZIG_HAS=$NO_PORT; expect_out "WARN: PROMPT_CACHE_GIB=8 with DRAFTER=mtp: this image.s engine keeps no prompt state with MTP drafts .*mtpcache" port_validate 1 8 ) || exit 1
  ( PROMPT_CACHE_GIB=0; expect_out "!WARN" port_validate 1 0 ) || exit 1
  ( DRAFTER=dflash2 PORT_POOL_EXPLICIT=1; expect_out "!keeps no prompt state" port_validate 8 8 ) || exit 1'
case_ "scripts/config.sh records whether KV_POOL_GIB was set, before it fills in the default" 0 '' "
  a=\$(grep -nF 'PORT_POOL_EXPLICIT=0; [[ -z \"\${KV_POOL_GIB:-}\" ]] || PORT_POOL_EXPLICIT=1' '$here/config.sh' | cut -d: -f1)
  b=\$(grep -nF 'KV_POOL_GIB=\"\${KV_POOL_GIB:-\$_pool}\"' '$here/config.sh' | cut -d: -f1)
  [[ -n \"\$a\" && -n \"\$b\" ]] && (( a < b )) || { echo \"config.sh: recorded at [\$a], defaulted at [\$b]\"; exit 1; }"
case_ "VISION=1 with PROMPT_CACHE_GIB above 0 warns that rank 0's kept-state budget is ~1.4 GiB smaller; not without vision, a cache, or with ENGINE=python" 0 'PROMPT_CACHE_GIB=8 PORT_POOL_EXPLICIT=1 KV_POOL_GIB=4' '
  ( VISION=1; expect_out "WARN: VISION=1 with PROMPT_CACHE_GIB=8: .*~1.4 GiB smaller" port_validate 8 8 ) || exit 1
  ( VISION=1; expect_out "!never kept or resumed" port_validate 8 8 ) || exit 1
  ( VISION=1 ZIG_HAS=$NO_PORT; expect_out "WARN: VISION=1 with PROMPT_CACHE_GIB=8: .*lacks the kept prompts with pictures .*never kept or resumed" port_validate 8 8 ) || exit 1
  ( VISION=1; expect_out "WARN: VISION=1 with PROMPT_CACHE_GIB=2.5: " port_validate 8 2.5 ) || exit 1
  ( VISION=0; expect_out "!VISION=1 with" port_validate 8 8 ) || exit 1
  ( unset VISION; expect_out "!VISION=1 with" port_validate 8 8 ) || exit 1
  ( VISION=1 PROMPT_CACHE_GIB=0; expect_out "!WARN" port_validate 8 0 ) || exit 1
  ( VISION=1 ENGINE=python; expect_out "!VISION=1 with" port_validate "" 8 ) || exit 1'
case_ "scripts/config.sh: VISION defaults to 1 for ENGINE=zig at TP=2 and ENGINE=python, 0 for ENGINE=zig at TP=3/4; a set VISION stays" 0 '' "
  line=\$(grep -E '^VISION=\"\\\$\\{VISION:-' '$here/config.sh')
  [[ -n \"\$line\" ]] || { echo 'config.sh: no VISION default line'; exit 1; }
  for c in 'zig 2 1' 'zig 3 0' 'zig 4 0' 'python 2 1' 'python 3 1'; do
    set -- \$c
    got=\$( unset VISION; ENGINE=\$1 TP=\$2; eval \"\$line\"; echo \"\$VISION\" )
    [[ \"\$got\" == \"\$3\" ]] || { echo \"ENGINE=\$1 TP=\$2: VISION=\$got, wanted \$3\"; exit 1; }
  done
  got=\$( VISION=0 ENGINE=zig TP=2; eval \"\$line\"; echo \"\$VISION\" ); [[ \"\$got\" == 0 ]] || { echo \"a set VISION=0 became \$got\"; exit 1; }"
case_ "refuse: spill with MTP drafts, also where their states are kept (the tier does not store the MTP head yet)" 0 'SPILL_GIB=64 DRAFTER=mtp PARALLEL=1 PROMPT_CACHE_GIB=8' '
  expect_die "DRAFTER=mtp: .*does not store the MTP head" port_validate 1 8
  ( ZIG_HAS="$DEF_HAS mtpcache"; expect_die "DRAFTER=mtp: .*does not store the MTP head" port_validate 1 8 ) || exit 1
  ( ZIG_HAS="$DEF_HAS mtpcache" SPILL_V2=1; expect_die "DRAFTER=mtp" port_validate 1 8 ) || exit 1'
case_ "refuse: KV_SHARD=1 with MTP drafts, also where their states are kept" 0 'ZIG_HAS="$DEF_HAS mtpcache" KV_SHARD=1 DRAFTER=mtp' '
  expect_die "KV_SHARD=1: it .*DRAFTER=mtp" port_validate 8 0'
# ---- sliced prompt fills (FILL, zig/patches/0026)
case_ "FILL=1: TF_GLM_FILL=1 on every rank, the label and summary say so; off (the default) adds nothing" 0 '' '
  expect "$FILL" 0
  port_validate 8 0; port_map; expect "${#PORT_ENV[@]} $(port_identity)" "0 off"
  FILL=1; port_validate 8 0; port_map
  expect "${PORT_ENV[*]}" "TF_GLM_FILL=1 TF_GLM_PREFIX_WAIT=1"
  [[ "$(port_identity)" =~ ^spill=0,shard=0,fill=1,sum=[0-9]+$ ]] || { port_identity; exit 1; }
  expect "$(port_summary)" " · sliced fills"
  [[ "TF_GLM_FILL=1" =~ ^($PORT_MAPPED_RE)= ]] || exit 1'
case_ "FILL from TF_GLM_FILL when only that is set; FILL wins over it, with a warning" 0 'TF_GLM_FILL=1' '
  expect "$FILL" 1
  ( FILL=0; expect_out "WARN: TF_GLM_FILL=1 is overridden by FILL=0" port_validate 8 0 ) || exit 1'
case_ "FILL=1 with KV sharding and spill v2: every variable together, the label names all three" 0 "ZIG_HAS='$ALL_HAS fill' FILL=1 KV_SHARD=1 SPILL_GIB=8 SPILL_V2=1" '
  port_validate 8 8; port_map
  [[ " ${PORT_ENV[*]} " == *" TF_GLM_KV_SHARD=1 "* && " ${PORT_ENV[*]} " == *" TF_GLM_SPILL_V2=1 "* && " ${PORT_ENV[*]} " == *" TF_GLM_FILL=1 TF_GLM_PREFIX_WAIT=1 "* ]] || { echo "${PORT_ENV[*]}"; exit 1; }
  [[ "$(port_identity)" =~ ^spill=v2,shard=1,fill=1,sum= ]] || { port_identity; exit 1; }'
case_ "refuse: FILL=2, FILL=1 with DRAFTER=mtp (also with mtpcache), FILL=1 without ZIG_HAS fill (naming the patch)" 0 '' '
  ( FILL=2; expect_die "FILL .TF_GLM_FILL. is 0 or 1, not 2" port_validate 8 0 ) || exit 1
  ( FILL=1 DRAFTER=mtp; expect_die "FILL=1: sliced prompt fills do not run with DRAFTER=mtp" port_validate 1 0 ) || exit 1
  ( FILL=1 DRAFTER=mtp PROMPT_CACHE_GIB=8; expect_die "FILL=1: .*DRAFTER=mtp" port_validate 1 8 ) || exit 1
  ( FILL=1 ZIG_HAS=$NO_PORT; expect_die "FILL=1: this image.s tensorfold-native lacks the sliced prompt fills .zig/patches/0026.*ZIG_HAS" port_validate 8 0 ) || exit 1'
case_ "FILL=1 with ENGINE=python: a warning that it does nothing, and nothing passed" 0 'ENGINE=python FILL=1' '
  expect_out "WARN: FILL=1 does nothing with ENGINE=python" port_validate "" ""
  port_validate "" "" >/dev/null; port_map; expect "${#PORT_ENV[@]} $(port_identity)" "0 off"'
case_ "scripts/config.sh: FILL defaults to 0, from TF_GLM_FILL when only that is set" 0 '' "
  line=\$(grep -E '^FILL=\"\\\$\\{FILL:-' '$here/config.sh')
  [[ -n \"\$line\" ]] || { echo 'config.sh: no FILL default line'; exit 1; }
  got=\$( unset FILL TF_GLM_FILL; eval \"\$line\"; echo \"\$FILL\" ); [[ \"\$got\" == 0 ]] || { echo \"default FILL=\$got\"; exit 1; }
  got=\$( unset FILL; TF_GLM_FILL=1; eval \"\$line\"; echo \"\$FILL\" ); [[ \"\$got\" == 1 ]] || { echo \"from TF_GLM_FILL: \$got\"; exit 1; }"
case_ "refuse: a relative SPILL_DIR, SPILL_HIGHWATER past 1, SPILL_MIN_TOKENS 8k, SPILL_GIB abc" 0 'PROMPT_CACHE_GIB=8' '
  ( SPILL_GIB=8 SPILL_DIR=spill; PORT_SPILL_EXPLICIT=1; expect_die "SPILL_DIR must be an absolute path" port_validate 8 8 ) || exit 1
  ( SPILL_GIB=8 SPILL_HIGHWATER=1.5; expect_die SPILL_HIGHWATER port_validate 8 8 ) || exit 1
  ( SPILL_GIB=8 SPILL_MIN_TOKENS=8k; expect_die SPILL_MIN_TOKENS port_validate 8 8 ) || exit 1
  ( SPILL_GIB=abc; expect_die "SPILL_GIB is a number" port_validate 8 8 ) || exit 1'
case_ "SPILL_GIB=0.0 is off" 0 'SPILL_GIB=0.0' '
  port_validate 8 0; port_map; expect "${#PORT_SERVE[@]} $(port_identity) ${STOP_TIMEOUT:-unset}" "0 off unset"'
case_ "extra serve arguments, the tier off: --spill-gib refused; --snapshot-dir (TensorFold's own snapshots) and tuning flags pass" 0 '' '
  port_check_extra --prompt-cache-gib 8 --reasoning-effort high || exit 1
  port_check_extra --snapshot-dir none --snapshot-dir=/x --spill-min-tokens 4096 --spill-keep-builds 2 || exit 1
  expect_die "the extra argument --spill-gib: the spill tier goes on through SPILL_GIB" port_check_extra --spill-gib 8
  expect_die "the extra argument --spill-gib=8" port_check_extra --spill-gib=8'
case_ "extra serve arguments, the tier on: a flag the recipe passes itself is refused, naming its setting; others pass" 0 'SPILL_GIB=8 PROMPT_CACHE_GIB=8' '
  expect_die "the extra argument --snapshot-dir: .*SPILL_DIR" port_check_extra --snapshot-dir /x
  expect_die "the extra argument --snapshot-dir=/x" port_check_extra --snapshot-dir=/x
  expect_die "the extra argument --spill-min-free-gib: .*SPILL_MIN_FREE_GIB" port_check_extra --spill-min-free-gib 20
  expect_die "the extra argument --spill-min-tokens=1: .*SPILL_MIN_TOKENS" port_check_extra --spill-min-tokens=1
  expect_die "the extra argument --spill-highwater: .*SPILL_HIGHWATER" port_check_extra --spill-highwater 0.5
  expect_die "the extra argument --spill-gib" port_check_extra --spill-gib 16
  port_check_extra --spill-keep-builds 2 --prompt-cache-gib 8 || exit 1'
case_ "extra serve arguments with ENGINE=python and the tier on: --spill-keep-builds (0088's, no setting) passes" 0 'ENGINE=python SPILL_GIB=64' '
  port_check_extra --spill-keep-builds 3 || exit 1'

# ---- spill v2
case_ "refuse: SPILL_V2=1 without ZIG_HAS spill2 (an image without the port), naming the missing patch" 0 'ZIG_HAS=$NO_PORT SPILL_GIB=64 SPILL_V2=1 PROMPT_CACHE_GIB=8' '
  expect_die "SPILL_V2=1: this image.s tensorfold-native lacks the spill v2 port (Robert.s GLM patches 0110/0111).*ZIG_HAS" port_validate 8 8'
case_ "refuse: TF_GLM_SPILL_V2=1 exported counts as explicit" 0 'ZIG_HAS=$NO_PORT SPILL_GIB=64 PROMPT_CACHE_GIB=8 TF_GLM_SPILL_V2=1' '
  expect "$PORT_V2_EXPLICIT $SPILL_V2" "1 1"; expect_die "SPILL_V2=1" port_validate 8 8'
case_ "refuse: SPILL_V2=1 with ENGINE=python (patches/ has v1 only)" 0 'ENGINE=python SPILL_GIB=64 SPILL_V2=1' '
  expect_die "Python engine.s patches/ lack the spill v2 port" port_validate "" ""'
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
case_ "refuse: SPILL_V2=1 at TP=3; a fork's default there: v1 for an explicit SPILL_GIB, else the tier off (warned)" 0 'ZIG_HAS=spill2 SPILL_GIB=64 SPILL_V2=1 PROMPT_CACHE_GIB=8 TP=3' '
  expect_die "SPILL_V2=1: spill v2 runs on two Sparks only for now .TP=2., not TP=3" port_validate 8 8
  ( PORT_V2_EXPLICIT=0; expect_out "WARN: SPILL V2 OFF for this start: spill v2 runs on two Sparks only" port_validate 8 8
    port_validate 8 8 >/dev/null; expect "$SPILL_GIB $SPILL_V2" "64 0" ) || exit 1
  ( PORT_V2_EXPLICIT=0 PORT_SPILL_EXPLICIT=0; expect_out "WARN: SPILL TIER OFF for this start: spill v2 runs on two Sparks only" port_validate 8 8
    port_validate 8 8 >/dev/null; expect "$SPILL_GIB" 0 ) || exit 1'
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
case_ "refuse: KV_SHARD=1 without ZIG_HAS kvshard (an image without the port), naming the missing patch" 0 'ZIG_HAS=$NO_PORT KV_SHARD=1' '
  expect_die "KV_SHARD=1: this image.s tensorfold-native lacks the KV-shard port (Robert.s GLM patch 0118)" port_validate 8 0'
case_ "refuse: TF_GLM_KV_SHARD=1 exported counts as explicit" 0 'ZIG_HAS=$NO_PORT TF_GLM_KV_SHARD=1' '
  expect "$PORT_SHARD_EXPLICIT $KV_SHARD" "1 1"; expect_die "KV-shard port" port_validate 8 0'
case_ "KV shard: the variables, carry and twins on by default, the check passed only when set" 0 'ZIG_HAS=kvshard KV_SHARD=1' '
  port_validate 8 0; port_map
  expect "${PORT_ENV[*]} ${#PORT_SERVE[@]}" "TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=1 TF_GLM_KV_SHARD_TWINS=1 0"
  [[ "TF_GLM_KV_SHARD_CARRY=x" =~ ^($PORT_MAPPED_RE)= ]] || exit 1
  [[ "$(port_identity)" =~ ^spill=0,shard=1,sum= ]] || exit 1
  expect "$(port_summary)" " · KV shard"
  TF_GLM_KV_SHARD_CARRY=0 TF_GLM_KV_SHARD_CHECK=1024; port_validate 8 0; port_map
  expect "${PORT_ENV[*]}" "TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=0 TF_GLM_KV_SHARD_TWINS=1 TF_GLM_KV_SHARD_CHECK=1024"'
case_ "refuse: KV_SHARD=1 at TP=3, --parallel 1, DRAFTER=mtp, ENGINE=python" 0 'ZIG_HAS=kvshard KV_SHARD=1' '
  ( TP=3; expect_die "two Sparks .TP=2., not TP=3" port_validate 8 0 ) || exit 1
  expect_die "more than one stream .PARALLEL above 1., not --parallel 1" port_validate 1 0
  ( DRAFTER=mtp; expect_die "DRAFTER=mtp" port_validate 8 0 ) || exit 1
  ( ENGINE=python; expect_die "Python engine.s patches/ lack the KV-shard port" port_validate "" "" ) || exit 1
  port_validate 16 0 >/dev/null || exit 1'
case_ "KV_SHARD=1 with TF_GLM_MULTI_LONE=1: a warning (it does nothing on the Zig engine), sharding on" 0 'ZIG_HAS=kvshard KV_SHARD=1 TF_GLM_MULTI_LONE=1' '
  expect_out "WARN: TF_GLM_MULTI_LONE=1 does nothing on the Zig engine" port_validate 8 0
  port_validate 8 0 >/dev/null; port_map; expect "$KV_SHARD ${PORT_ENV[*]}" "1 TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=1 TF_GLM_KV_SHARD_TWINS=1"
  ( TF_GLM_MULTI_LONE=0; expect_out "!MULTI_LONE" port_validate 8 0 ) || exit 1
  ( KV_SHARD=0; expect_out "!MULTI_LONE" port_validate 8 0 ) || exit 1'
case_ "refuse: KV_SHARD=2, TF_GLM_KV_SHARD_CARRY=yes, TF_GLM_KV_SHARD_CHECK=x" 0 'ZIG_HAS=kvshard' '
  ( KV_SHARD=2; expect_die "KV_SHARD .TF_GLM_KV_SHARD. is 0 or 1" port_validate 8 0 ) || exit 1
  ( TF_GLM_KV_SHARD_CARRY=yes; expect_die TF_GLM_KV_SHARD_CARRY port_validate 8 0 ) || exit 1
  ( TF_GLM_KV_SHARD_CHECK=x; expect_die TF_GLM_KV_SHARD_CHECK port_validate 8 0 ) || exit 1'
case_ "KV-shard settings without KV_SHARD=1 warn (and pass to the ranks as before)" 0 'TF_GLM_KV_SHARD_TWINS=0' '
  expect_out "WARN: TF_GLM_KV_SHARD_TWINS do nothing without KV_SHARD=1" port_validate 8 0
  port_map; expect "${#PORT_ENV[@]}" 0; [[ ! "TF_GLM_KV_SHARD_TWINS=0" =~ ^($PORT_MAPPED_RE)= ]] || exit 1'
case_ "KV_SHARD wins over TF_GLM_KV_SHARD, with a warning" 0 'ZIG_HAS=kvshard KV_SHARD=1 TF_GLM_KV_SHARD=0' '
  expect_out "WARN: TF_GLM_KV_SHARD=0 is overridden by KV_SHARD=1" port_validate 8 0; expect "$KV_SHARD" 1'

# ---- long contexts on the Zig engine (port_long_context, zig/aot/glm-aot-coverage.txt)
LC_PRE="_cfg_root=$root"
case_ "long context: past 262,144 tokens TF_TP_TIMEOUT_S=1500 unless set; not at 262,144, not with ENGINE=python" 0 "$LC_PRE" '
  port_long_context 1048576 8 >/dev/null; expect "${TF_TP_TIMEOUT_S:-unset}" 1500
  ( unset TF_TP_TIMEOUT_S; port_long_context 262144 8 >/dev/null; expect "${TF_TP_TIMEOUT_S:-unset}" unset ) || exit 1
  ( unset TF_TP_TIMEOUT_S; ENGINE=python; port_long_context 1048576 4 >/dev/null; expect "${TF_TP_TIMEOUT_S:-unset}" unset ) || exit 1
  ( TF_TP_TIMEOUT_S=900; port_long_context 1048576 8 >/dev/null; expect "$TF_TP_TIMEOUT_S" 900 ) || exit 1
  ( unset TF_TP_TIMEOUT_S; TP=3; port_long_context 1048576 8 >/dev/null; expect "$TF_TP_TIMEOUT_S" 1500 ) || exit 1'
case_ "long context: the GPU-checked window is silent, one the coverage test checked warns what is unverified, one past it can miss" 0 "$LC_PRE" '
  expect_out "!WARN" port_long_context 131072 8
  expect_out "!WARN" port_long_context 1048576 4
  expect_out "!WARN" port_long_context 1048576 8
  expect_out "WARN: ENGINE=zig: CONTEXT=524288 PARALLEL=8: the kernel set holds every Triton variant.*no GPU run has served it yet.*Unverified: a long prompt.s memory on each Spark, TF_TP_TIMEOUT_S=1500" port_long_context 524288 8
  expect_out "no GPU run has served it yet" port_long_context 131072 4
  expect_out "no GPU run has served it yet" port_long_context 200000 8
  expect_out "outside what the kernel set.s coverage test checked (contexts up to 1048576" port_long_context 2097152 8
  expect_out "outside what the kernel set.s coverage test checked" port_long_context 131072 2'
case_ "long context: a coverage record of another kernel set, or none, leaves every window unchecked; TP=3 keeps the arena warning" 0 "$LC_PRE; tmp=$tmp" '
  printf "aot=%064d\ncontexts=1048576\nparallel=8\ngpu=131072:8\n" 0 > "$tmp/cov-other.txt"
  ( PORT_COVERAGE=$tmp/cov-other.txt; expect_out "is for set 000000000000, this one is 29ad991945a3.*unchecked" port_long_context 131072 8 ) || exit 1
  ( PORT_COVERAGE=$tmp/none.txt; expect_out "no coverage record for this kernel set.*unchecked" port_long_context 131072 8 ) || exit 1
  ( TP=3; expect_out "keyed on the 8-stream, 131072-token arena" port_long_context 1048576 8; expect_out "!WARN" port_long_context 131072 8 ) || exit 1
  grep -q "^aot=$(cut -c1-64 "$root/zig/aot/glm-aot-sm121.tar.xz.sha256")$" "$root/zig/aot/glm-aot-coverage.txt" || { echo "the coverage record is not for zig/aot/glm-aot-sm121.tar.xz"; exit 1; }'
LC_DEFAULT='ZIG_CONTEXT_1M="${ZIG_CONTEXT_1M:-1}"'
LC_PAR='[[ "$ENGINE" != zig || "$DRAFTER" != dflash2 ]] || _par=8'
LC_CTX='if [[ "$ENGINE" == zig ]]; then _ctx=131072; [[ "$ZIG_CONTEXT_1M" == 1 && "$TP" == 2 ]] && _ctx=1048576; fi'
case_ "scripts/config.sh: ENGINE=zig defaults to 1,048,576 x 8 at TP=2, 131072 x 8 with ZIG_CONTEXT_1M=0 or at TP=3" 0 '' '
  grep -qxF -- "$LC_DEFAULT" "$here/config.sh" || { echo "no ZIG_CONTEXT_1M default of 1"; exit 1; }
  grep -qxF -- "$LC_PAR" "$here/config.sh" || { echo "the ENGINE=zig PARALLEL default moved"; exit 1; }
  grep -qxF -- "$LC_CTX" "$here/config.sh" || { echo "the ENGINE=zig CONTEXT default moved"; exit 1; }
  DRAFTER=dflash2
  for ENGINE in zig python; do for TP in 2 3; do for ZIG_CONTEXT_1M in 0 1; do
    _ctx=python; _par=python; eval "$LC_CTX"; eval "$LC_PAR"; want=python; wpar=python
    [[ "$ENGINE" == zig ]] && { want=131072; wpar=8; }
    [[ "$ENGINE" == zig && "$ZIG_CONTEXT_1M" == 1 && "$TP" == 2 ]] && want=1048576
    expect "$ENGINE $TP $ZIG_CONTEXT_1M $_ctx $_par" "$ENGINE $TP $ZIG_CONTEXT_1M $want $wpar"
  done; done; done
  ! grep -q "no prompts past ~32k tokens" "$here/config.sh" || { echo "the stale 32k comment is back"; exit 1; }'

case_ "scripts/config.sh: the reserve grows past 4 requests except on the Zig engine at TP=2" 0 '' '
  for e in zig python; do for p in 4 8; do
    got=$(env -u MEMORY_RESERVE_GIB -u TENSORFOLD_MEMORY_RESERVE_GIB -u TF_GLM_MULTI_WINDOW ENGINE=$e PARALLEL=$p TP=2 bash -c "source \"$here/config.sh\" >/dev/null 2>&1; echo \$MEMORY_RESERVE_GIB")
    want=14.5; [[ $e == python && $p == 8 ]] && want=19.6  # 0.95 x 4 requests + 0.04 x 32 rows (its window is 64 at 8)
    expect "$e $p $got" "$e $p $want"
  done; done
  got=$(env -u MEMORY_RESERVE_GIB -u TENSORFOLD_MEMORY_RESERVE_GIB -u TF_GLM_MULTI_WINDOW ENGINE=zig PARALLEL=8 TP=3 bash -c "source \"$here/config.sh\" >/dev/null 2>&1; echo \$MEMORY_RESERVE_GIB")
  awk -v g="$got" "BEGIN { exit !(g > 14.5) }" || { echo "zig at TP=3: the reserve should still grow (got $got)"; exit 1; }'

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
case_ "fork defaults with every port in the image: spill v2 and KV shard on" 0 "PT_FORK=1 ZIG_HAS='$ALL_HAS' PROMPT_CACHE_GIB=8 KV_POOL_GIB=4 PORT_POOL_EXPLICIT=1" '
  expect "$PORT_SPILL_EXPLICIT $PORT_V2_EXPLICIT $PORT_SHARD_EXPLICIT $SPILL_GIB $SPILL_V2 $KV_SHARD $STOP_TIMEOUT" "0 0 0 64 1 1 90"
  expect_out "!WARN" port_validate 8 8
  port_validate 8 8; port_map
  expect "${PORT_SERVE[*]}" "--spill-gib 64 --snapshot-dir /spill --spill-min-tokens 8192 --spill-min-free-gib 50"
  expect "${PORT_ENV[*]}" "TF_SPILL_FLUSH_S=60 TF_GLM_SPILL_V2=1 TF_GLM_SPILL_QUOTA=0.25 TF_GLM_SPILL_TURNS=4 TF_GLM_KV_SHARD=1 TF_GLM_KV_SHARD_CARRY=1 TF_GLM_KV_SHARD_TWINS=1"'
case_ "fork defaults on an image without the port (no spill2, kvshard): both off with warnings, as before the port" 0 "PT_FORK=1 ZIG_HAS='$NO_PORT' PROMPT_CACHE_GIB=8" '
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
# ---- the prefix wait (PREFIX_WAIT, the Python fork's 0110; the Zig engine's lane host with FILL=1)
case_ "PREFIX_WAIT with FILL=1: on by default, TF_GLM_PREFIX_WAIT to every rank (0 when off), TF_GLM_PREFIX_WAIT_MS as set" 0 'FILL=1' '
  expect "$PREFIX_WAIT" 1
  port_validate 8 8; port_map
  expect "${PORT_ENV[*]}" "TF_GLM_FILL=1 TF_GLM_PREFIX_WAIT=1"
  expect "$(port_summary)" " · sliced fills"
  a=$(port_identity)
  ( PREFIX_WAIT=0; port_validate 8 8; port_map; expect "${PORT_ENV[*]}" "TF_GLM_FILL=1 TF_GLM_PREFIX_WAIT=0"
    expect "$(port_summary)" " · sliced fills (no prefix wait)"; [[ "$(port_identity)" != "$a" ]] ) || exit 1
  ( TF_GLM_PREFIX_WAIT_MS=5000; port_validate 8 8; port_map
    expect "${PORT_ENV[*]}" "TF_GLM_FILL=1 TF_GLM_PREFIX_WAIT=1 TF_GLM_PREFIX_WAIT_MS=5000" ) || exit 1
  for x in TF_GLM_PREFIX_WAIT TF_GLM_PREFIX_WAIT_MS; do [[ "$x=1" =~ ^($PORT_MAPPED_RE)= ]] || { echo "$x not mapped"; exit 1; }; done'
case_ "PREFIX_WAIT from TF_GLM_PREFIX_WAIT when only that is set; PREFIX_WAIT wins over it, with a warning" 0 'FILL=1 TF_GLM_PREFIX_WAIT=0' '
  expect "$PREFIX_WAIT" 0
  port_validate 8 8 >/dev/null; port_map; expect "${PORT_ENV[*]}" "TF_GLM_FILL=1 TF_GLM_PREFIX_WAIT=0"
  ( PREFIX_WAIT=1; expect_out "WARN: TF_GLM_PREFIX_WAIT=0 is overridden by PREFIX_WAIT=1" port_validate 8 8 ) || exit 1'
case_ "PREFIX_WAIT without FILL=1: nothing passed (whole passes fill one prompt at a time); set, it warns; unset or 0 silent" 0 '' '
  expect_out "!WARN" port_validate 8 0; port_map; expect "${#PORT_ENV[@]}" 0
  ( PREFIX_WAIT=0; expect_out "!WARN" port_validate 8 0 ) || exit 1
  ( PORT_PW_EXPLICIT=1 PREFIX_WAIT=1; expect_out "WARN: PREFIX_WAIT=1 does nothing without FILL=1" port_validate 8 0; port_map; expect "${#PORT_ENV[@]}" 0 ) || exit 1
  ( TF_GLM_PREFIX_WAIT_MS=5000; expect_out "WARN: TF_GLM_PREFIX_WAIT_MS=5000 does nothing without FILL=1" port_validate 8 0 ) || exit 1'
case_ "PREFIX_WAIT=1 set with FILL=1 and no prompt cache warns (nothing is kept to resume); the default does not" 0 'FILL=1 PREFIX_WAIT=1' '
  expect_out "WARN: PREFIX_WAIT=1 does nothing without a prompt cache" port_validate 8 0
  ( PORT_PW_EXPLICIT=0; expect_out "!WARN" port_validate 8 0 ) || exit 1'
case_ "refuse: PREFIX_WAIT=2, TF_GLM_PREFIX_WAIT_MS not a whole number of milliseconds" 0 'FILL=1' '
  ( PREFIX_WAIT=2; expect_die "PREFIX_WAIT .TF_GLM_PREFIX_WAIT. is 0 or 1, not 2" port_validate 8 8 ) || exit 1
  ( TF_GLM_PREFIX_WAIT_MS=1s; expect_die "TF_GLM_PREFIX_WAIT_MS is a whole number of milliseconds" port_validate 8 8 ) || exit 1
  ( TF_GLM_PREFIX_WAIT_MS=-5; expect_die "TF_GLM_PREFIX_WAIT_MS is a whole number of milliseconds" port_validate 8 8 ) || exit 1'
case_ "PREFIX_WAIT=1 with ENGINE=python: a warning that it does nothing, and nothing passed" 0 'ENGINE=python PREFIX_WAIT=1' '
  expect_out "WARN: PREFIX_WAIT=1 does nothing with ENGINE=python" port_validate "" ""
  port_validate "" "" >/dev/null; port_map; expect "${#PORT_ENV[@]}" 0'

# ---- after the start: each rank opened its spill tier (rank 0's prompt cache budget above 0; every rank's own budget
# above 0 too: each rank opens its tier from its own, with no agreement between them)
r0_ok='info: prompt cache: 2.0 GiB for kept states, at most 14 (fixed 102 MiB each)
info: spill tier: /spill/tfspill-00000000000000aa/rank0, up to 8 GiB, 0 states (0 MiB) found'
r1_ok='info: spill tier: /spill/tfspill-00000000000000aa/rank1, up to 8 GiB, 0 states (0 MiB) found'
case_ "after the start: a prompt cache budget of 0 with the spill tier on warns; every rank's tier open, or no tier, say nothing" 0 'SPILL_GIB=8 PROMPT_CACHE_GIB=8' "
  expect_out 'WARN: the spill tier is not open: rank 0 logged \"prompt cache: off (budget 0)\"' port_check_started 'info: prompt cache: off (budget 0)' '$r1_ok'
  expect_out '!WARN' port_check_started '$r0_ok' '$r1_ok'
  expect_out '!WARN' port_check_started '$r0_ok'
  ( SPILL_GIB=0; expect_out '!WARN' port_check_started 'info: prompt cache: off (budget 0)' '' ) || exit 1
  ( ENGINE=python; expect_out '!WARN' port_check_started 'prompt cache: off (budget 0)' '' ) || exit 1"
case_ "after the start: a rank that opened no spill tier while rank 0 did warns, naming the rank and the engine's reason" 0 'SPILL_GIB=8 PROMPT_CACHE_GIB=8' "
  expect_out 'WARN: rank 1 opened no spill tier' port_check_started '$r0_ok' ''
  expect_out 'WARN: rank 2 opened no spill tier' port_check_started '$r0_ok' '$r1_ok' 'info: prompt cache: something'
  expect_out 'WARN: rank 1 opened no spill tier (it logged \"spill: cannot open /spill (AccessDenied); no spill tier\")' \
    port_check_started '$r0_ok' 'error: spill: cannot open /spill (AccessDenied); no spill tier'
  expect_out 'WARN: rank 0 opened no spill tier' port_check_started 'info: prompt cache: 2.0 GiB for kept states' '$r1_ok'
  expect_out 'WARN: could not read rank 1.s log' port_check_started '$r0_ok' '(unread)'
  expect_out '!rank 1 opened' port_check_started '$r0_ok' '(unread)'"
# spill v2's start lines, as glm_cuda/cuda_native.zig openTier logs them (g3 and g6 with SPILL_V2=1 printed exactly these,
# and the check above, which knew v1's "spill tier:" only, warned "rank 0 opened no spill tier" for both ranks), and its
# failure line (one rank's store that did not open fails the start on every rank, so a log that has it has no tier line)
v2_r0="info: prompt cache: 5.1 GiB for kept states, at most 14 (fixed 102 MiB each)
info: rank 0: spill v2 tier: /spill, up to 16 GiB, 0 points (0 MiB) found; each kept state written at keep from 8192 tokens, keeping 20 GiB of disk free; its folders given to the spill directory's owner"
v2_r1="info: rank 1: spill v2 tier: /spill, up to 16 GiB, 3 points (812 MiB) found; each kept state written at keep from 8192 tokens, keeping 20 GiB of disk free; owner-sharded (TF_GLM_KV_SHARD: a point's latent rows on its home part's rank only)"
v2_bad="error: rank 1: spill v2's store under /spill did not open (AccessDenied)"
case_ "after the start, spill v2: both ranks' \"spill v2 tier:\" lines say nothing; a missing one warns, with the engine's reason" 0 'SPILL_GIB=16 SPILL_V2=1 PROMPT_CACHE_GIB=8' '
  port_validate 8 8
  expect_out "!WARN" port_check_started "$v2_r0" "$v2_r1"
  expect_out "!WARN" port_check_started "$v2_r0"
  expect_out "WARN: rank 1 opened no spill v2 tier" port_check_started "$v2_r0" ""
  expect_out "WARN: rank 0 opened no spill v2 tier" port_check_started "info: prompt cache: 5.1 GiB for kept states" "$v2_r1"
  expect_out "WARN: rank 1 opened no spill v2 tier (it logged \"spill v2.s store under /spill did not open (AccessDenied)\")" \
    port_check_started "$v2_r0" "$v2_bad"
  expect_out "WARN: the spill tier is not open: rank 0 logged \"prompt cache: off (budget 0)\"" port_check_started "info: prompt cache: off (budget 0)" "$v2_r1"'
case_ "after the start: a rank that opened the other version of the tier than asked warns, naming both" 0 'SPILL_GIB=16 PROMPT_CACHE_GIB=8' '
  ( SPILL_V2=1; port_validate 8 8; expect_out "WARN: rank 1 opened the v1 spill tier, not v2 (SPILL_V2=1)" port_check_started "$v2_r0" "$r1_ok" ) || exit 1
  ( port_validate 8 8; expect_out "WARN: rank 0 opened the v2 spill tier, not v1 (SPILL_V2=0)" port_check_started "$v2_r0" "$r1_ok" ) || exit 1
  ( port_validate 8 8; expect_out "!WARN" port_check_started "$r0_ok" "$r1_ok" ) || exit 1'
case_ "start.sh reads each rank's tier lines through port.sh's pattern, which keeps v1's, v2's and their failures" 0 '' '
  for l in "$r0_ok" "$v2_r0" "$v2_r1" "$v2_bad" "error: spill: cannot open /spill (AccessDenied); no spill tier"; do
    while IFS= read -r x; do grep -qE -- "$PORT_TIER_LOG_RE" <<<"$x" || { echo "the pattern drops: $x"; exit 1; }; done <<<"$l"
  done
  ! grep -qE -- "$PORT_TIER_LOG_RE" <<<"info: rank 0: kv shard: two parts" || { echo "the pattern keeps an unrelated line"; exit 1; }
  grep -qF -- "grep -E \"\$PORT_TIER_LOG_RE\"" "'"$root"'/start.sh" || { echo "start.sh does not grep with PORT_TIER_LOG_RE"; exit 1; }
  [[ "$(grep -c -- "PORT_TIER_LOG_RE" "'"$root"'/start.sh")" == 2 ]] || { echo "start.sh: rank 0 and the workers both through it"; exit 1; }'

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
  port_spill_judge here 'OK 1 ext4 $((500 * G)) 0 0 0 - 0 /data/spill' || exit 1
  expect_out '!WARN' port_spill_judge here 'OK 1 ext4 $((500 * G)) 0 0 0 - 0 /data/spill'"
case_ "judge: refusals (network, memory, export, overlap, home, not writable, no room, ERR)" 0 'SPILL_GIB=8' "
  expect_die 'network file system (nfs4)' port_spill_judge w1 'OK 1 nfs4 $((500 * G)) 0 0 0 - 0 /mnt/spill'
  expect_die 'on tmpfs' port_spill_judge w1 'OK 1 tmpfs $((500 * G)) 0 0 0 - 0 /dev/shm/spill'
  expect_die 'inside an NFS export' port_spill_judge here 'OK 1 ext4 $((500 * G)) 0 0 1 - 0 /home/u/.cache/huggingface/spill'
  expect_die 'overlaps /home/u/.cache/huggingface' port_spill_judge here 'OK 1 ext4 $((500 * G)) 0 0 0 /home/u/.cache/huggingface 0 /home/u/.cache/huggingface/s'
  expect_die '/ or your home folder' port_spill_judge here 'OK 1 ext4 $((500 * G)) 0 0 0 - 1 /home/u'
  expect_die 'not writable by you' port_spill_judge here 'OK 0 ext4 $((500 * G)) 0 0 0 - 0 /data/spill'
  expect_die 'you cannot create it' port_spill_judge here 'NEW 0 ext4 $((500 * G)) 0 0 0 - 0 /data/spill'
  expect_die 'under SPILL_MIN_FREE_GIB' port_spill_judge here 'OK 1 ext4 $((40 * G)) 0 0 0 - 0 /data/spill'
  expect_die 'SPILL_DIR on w1: cannot create /x' port_spill_judge w1 'ERR cannot create /x'"
case_ "judge: less room than SPILL_GIB + the floor warns; a folder to create is logged" 0 'SPILL_GIB=64' "
  expect_out 'WARN: .*less than SPILL_GIB (64) + SPILL_MIN_FREE_GIB' port_spill_judge here 'OK 1 ext4 $((100 * G)) 0 0 0 - 0 /data/spill'
  expect_out 'LOG: SPILL_DIR /data/spill on here does not exist yet' port_spill_judge here 'NEW 1 ext4 $((500 * G)) 0 0 0 - 0 /data/spill'"
case_ "judge: the tiers' own folders count as room, for the floor and for the cap" 0 'SPILL_GIB=64' "
  expect_out '!WARN' port_spill_judge here 'OK 1 ext4 $((45 * G)) $((70 * G)) 0 0 - 0 /data/spill'
  expect_out 'WARN: .*has 45 GiB free (and 20.0 GiB in the spill tiers. own folders, which the tier frees itself), less than SPILL_GIB (64)' \
    port_spill_judge here 'OK 1 ext4 $((45 * G)) $((20 * G)) 0 0 - 0 /data/spill'
  expect_die 'has 40 GiB free (and 2.0 GiB in the spill tiers. own folders, which the tier frees itself), under SPILL_MIN_FREE_GIB (50)' \
    port_spill_judge here 'OK 1 ext4 $((40 * G)) $((2 * G)) 0 0 - 0 /data/spill'"
case_ "judge: folders it cannot read: under the floor warns (no refusal), and no cap warning" 0 'SPILL_GIB=64' "
  expect_out 'WARN: SPILL_DIR /data/spill on here has 40 GiB free, under SPILL_MIN_FREE_GIB (50), and holds spill folders this check cannot read' \
    port_spill_judge here 'OK 1 ext4 $((40 * G)) 4096 1 0 - 0 /data/spill'
  expect_out '!WARN' port_spill_judge here 'OK 1 ext4 $((60 * G)) 4096 1 0 - 0 /data/spill'"
case_ "judge: a default tier's failure returns 1 with the reason (no refusal)" 0 "PT_FORK=1" "
  port_spill_judge w1 'OK 1 nfs $((500 * G)) 0 0 0 - 0 /mnt/spill' && exit 1
  [[ \"\$PORT_SPILL_WHY\" == *'network file system (nfs)'* ]] || { echo \"\$PORT_SPILL_WHY\"; exit 1; }"

# ---- spill_dir_probe on temporary folders (this machine: Linux or a Mac)
mkdir -p "$tmp/hf/hub" "$tmp/ro"
tmpr=$(cd -P "$tmp" && pwd -P)
case_ "probe: creates a missing folder (create=1) and describes it" 0 '' "
  l=\$(spill_dir_probe 1 '$tmp/a/spill' '$tmp/hf'); [[ -d '$tmp/a/spill' ]] || exit 1
  read -r st w fs avail tier tu exp over top real <<<\"\$l\"
  expect \"\$st \$w \$exp \$over \$top \$real\" 'OK 1 0 - 0 $tmpr/a/spill'
  [[ \"\$avail\" =~ ^[0-9]+\$ && \"\$avail\" -gt 0 ]] || { echo \"\$l\"; exit 1; }"
case_ "probe: DRY_RUN (create=0) creates nothing and says NEW" 0 '' "
  l=\$(spill_dir_probe 0 '$tmp/b/spill'); [[ ! -e '$tmp/b' ]] || exit 1
  read -r st w fs avail tier tu exp over top real <<<\"\$l\"; expect \"\$st \$w \$real\" 'NEW 1 $tmpr/b/spill'"
case_ "probe: a folder inside (or around) the Hugging Face cache overlaps it" 0 '' "
  l=\$(spill_dir_probe 1 '$tmp/hf/spill' '$tmp/hf'); read -r st w fs avail tier tu exp over top real <<<\"\$l\"; expect \"\$over\" '$tmpr/hf'
  l=\$(spill_dir_probe 1 '$tmp' '$tmp/hf'); read -r st w fs avail tier tu exp over top real <<<\"\$l\"; expect \"\$over\" '$tmpr/hf'"
case_ "probe: the same line through sh -c (a worker's login shell may be sh)" 0 '' "
  expect \"\$(sh -c \"\$(declare -f spill_dir_probe); spill_dir_probe 1 '$tmp/s/spill' '$tmp/hf'\" | cut -d' ' -f1,2,7-)\" 'OK 1 0 - 0 $tmpr/s/spill'"
case_ "probe: the home folder itself" 0 '' "
  HOME='$tmp/a'; l=\$(spill_dir_probe 1 '$tmp/a'); read -r st w fs avail tier tu exp over top real <<<\"\$l\"; expect \"\$top\" 1"
mkdir -p "$tmp/t/spill/other"
for x in tfspill-00000000000000aa glm-spill-00000000000000ee native-spill-00000000000000bb spill-00000000000000cc; do
  mkdir -p "$tmp/t/spill/$x/rank0"; head -c 1048576 /dev/zero > "$tmp/t/spill/$x/rank0/s"
done
head -c 1048576 /dev/zero > "$tmp/t/spill/other/s"; head -c 1048576 /dev/zero > "$tmp/t/spill/spill-file"
ln -s "$tmp/t/spill/other" "$tmp/t/spill/tfspill-00000000000000dd"
case_ "probe: measures the spill tiers' own folders (tfspill-, glm-spill-, native-spill-, spill-), not other entries or links" 0 '' "
  for sh in bash sh; do
    l=\$(\$sh -c \"\$(declare -f spill_dir_probe); spill_dir_probe 1 '$tmp/t/spill'\"); read -r st w fs avail tier tu rest <<<\"\$l\"
    [[ \"\$tier\" =~ ^[0-9]+\$ ]] && (( tier >= 4 * 1048576 && tier < 4 * 1048576 + 262144 )) || { echo \"\$sh: \$l\"; exit 1; }
    expect \"\$tu\" 0
  done"
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

# ---- the free-disk floor: the tier's own folders count as room, since the engine frees them itself (it drops its oldest
# files to keep the floor, and removes other builds' folders at open). df is stubbed (the probe's only df call), here and
# in the worker's shell; the floor is 1 GiB and the tier's folder holds 8 MiB.
# dfstub <free KiB>: the function, as text for a case's script
dfstub() { printf 'df() { printf "Filesystem 1024-blocks Used Available Capacity Mounted\\n/dev/x 2097152 1 %s 50%%%% /\\n"; }; worker() { shift; bash -c "$(declare -f df); $1"; }' "$1"; }
tierdir() { mkdir -p "$1/tfspill-00000000000000aa/rank0" && head -c 8388608 /dev/zero > "$1/tfspill-00000000000000aa/rank0/00000000000000aa.state"; }
tierdir "$tmp/r1/spill"; tierdir "$tmp/r2/spill"; tierdir "$tmp/r3/spill"
case_ "spill check: under the floor by less than the tier's own files: no refusal (it frees them itself)" 0 "SPILL_GIB=0.001 SPILL_MIN_FREE_GIB=1 PROMPT_CACHE_GIB=8 SPILL_DIR='$tmp/r1/spill' HF_CACHE='$tmp/hf'" "$stubs
  $(dfstub $((1048576 - 4096)))
  expect_out 'LOG: Spill tier: v1, up to 0.001 GiB' port_spill_check"
case_ "spill check: under the floor by more than the tier's own files: refused, naming both" 0 "SPILL_GIB=0.001 SPILL_MIN_FREE_GIB=1 PROMPT_CACHE_GIB=8 SPILL_DIR='$tmp/r2/spill' HF_CACHE='$tmp/hf'" "$stubs
  $(dfstub $((1048576 - 16384)))
  expect_die 'has 0 GiB free (and 0.0 GiB in the spill tiers. own folders, which the tier frees itself), under SPILL_MIN_FREE_GIB (1)' port_spill_check"
if [[ "$(id -u)" != 0 ]]; then
  chmod 000 "$tmp/r3/spill/tfspill-00000000000000aa"
  case_ "spill check: a tier folder it cannot read (the Zig engine's, root's 0700) under the floor: a warning, not a refusal" 0 "SPILL_GIB=0.001 SPILL_MIN_FREE_GIB=1 PROMPT_CACHE_GIB=8 SPILL_DIR='$tmp/r3/spill' HF_CACHE='$tmp/hf'" "$stubs
    $(dfstub $((1048576 - 16384)))
    expect_out 'WARN: SPILL_DIR .* on this Spark has 0 GiB free, under SPILL_MIN_FREE_GIB (1), and holds spill folders this check cannot read' port_spill_check
    expect_out 'LOG: Spill tier: v1' port_spill_check"
  chmod 700 "$tmp/r3/spill/tfspill-00000000000000aa"
fi

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
case_ "start check: mem_avail_gb 1 is whole GiB (rounded down) and the parked GiB" 0 "source '$here/memavail.sh'" "
  mem_counting() { echo '109.9 6.0 6442450944'; }; expect \"\$(mem_avail_gb 1)\" '109 6.0'"
case_ "start check: mem_avail_gb 1 on the synthetic Spark (100 GiB available, 6 GiB parked)" 0 "source '$here/memavail.sh'" "
  eval \"\$(declare -f mem_counting | sed '1s/mem_counting/mem_counting_file/')\"
  mem_counting() { mem_counting_file '$mt/meminfo' '$mt/z-six' 4096; }
  expect \"\$(mem_avail_gb 1)\" '106 6.0'"
# free -g on a Spark right after an image copy (available 103), for the check of an engine that counts no parked pages
freestub="free() { printf '               total        used        free      shared  buff/cache   available\\nMem:             119          12           3           0         101         103\\nSwap:             15           0          15\\n'; }"
case_ "start check: mem_avail_gb 0 (the engine counts no parked pages) is free -g's available column alone, as before" 0 "source '$here/memavail.sh'" "
  mem_counting() { echo '109.9 6.0 6442450944'; }; $freestub
  expect \"\$(mem_avail_gb 0)\" '103 0'
  expect \"\$(mem_avail_gb)\" '103 0'
  expect \"\$(bash -c \"\$(declare -f free mem_counting mem_avail_gb); mem_avail_gb 0\")\" '103 0'"
case_ "start check: mem_avail_gb runs on a worker as start.sh sends it (declare -f over ssh, here bash -c and sh -c)" 0 "source '$here/memavail.sh'" "
  for sh in bash sh; do
    expect \"\$(\$sh -c \"\$(declare -f mem_counting); mem_counting '$mt/meminfo' '$mt/z-six' 4096\")\" '106.0 6.0 6442450944'
  done
  l=\$(bash -c \"\$(declare -f mem_counting mem_avail_gb); mem_avail_gb 1\")
  if [[ -r /proc/meminfo ]]; then [[ \"\$l\" =~ ^[0-9]+\\ [0-9]+\\.[0-9]\$ ]] || { echo \"[\$l]\"; exit 1; }; fi"
case_ "start check: parked_counted (scripts/memavail.sh; scripts/memavail_test.sh has its cases) with these cases' default" 0 "source '$here/memavail.sh'" "
  ( ZIG_HAS='mtp displaykv'; ! parked_counted ) || { echo 'counted without the token'; exit 1; }
  parked_counted || { echo 'not counted with start.sh.s default'; exit 1; }"
check_start() { grep -qF -- "$1" "$root/start.sh" || { echo "start.sh lacks: $1"; return 1; }; }
case_ "start.sh: the memory check passes the gate here and to each worker; the post-start check reads every rank" 0 '' "
  $(declare -f check_start); root='$root'
  check_start 'PARKED=0; ! parked_counted || PARKED=1' || exit 1
  check_start 'read -r here_gb here_parked <<<\"\$(mem_avail_gb \"\$PARKED\")\"' || exit 1
  check_start 'declare -f mem_counting mem_avail_gb); mem_avail_gb \$PARKED' || exit 1
  check_start 'port_check_started \"\${PORT_LOGS[@]}\"' || exit 1"
case_ "start.sh's default ZIG_HAS is these cases' default, and names each of the port's features exactly when zig/patches carries it (by content)" 0 '' "
  $(declare -f check_start); root='$root'
  check_start 'ZIG_HAS=\"\${ZIG_HAS:-$DEF_HAS}\"' || exit 1
  carries() { cat '$root'/zig/patches/*.patch | grep -qF -- \"\$1\"; }
  for f in 'parked:+pub fn parkedBytes(' 'spill2:+++ b/zig/src/families/glm_cuda/cuda_spill_tier.zig' 'kvshard:+pub const gpu_paths = true;' 'spillshard:return error.KvShardHomeFull;' \
           'mtpcache:+++ b/zig/src/families/glm_cuda/cuda_mtp_keep.zig' 'fill:return error.FillWithMtp;' 'pictures:+++ b/zig/src/core/prompt_media.zig' 'prefixwait:+++ b/zig/src/core/lane_host_prefix_test.zig'; do
    t=\${f%%:*}; m=\${f#*:}
    if [[ ' $DEF_HAS ' == *\" \$t \"* ]]; then carries \"\$m\" || { echo \"ZIG_HAS's default names \$t, but no zig/patches patch adds: \$m\"; exit 1; }
    else ! carries \"\$m\" || { echo \"zig/patches carries \$t (\$m), but ZIG_HAS's default does not name it\"; exit 1; }; fi
  done
  # and the start check's notes say which engine counts the parked pages (once zig/patches carries the patch)
  if carries '+pub fn parkedBytes('; then
    grep -q \"zig/patches' parked-pages patch\" '$here/memavail.sh' && ! grep -qi 'neither engine' '$here/memavail.sh' '$root/CHANGELOG.md' ||
      { echo 'memavail.sh or the CHANGELOG still says no engine here counts the parked pages'; exit 1; }
  fi"
if [[ -r /proc/meminfo ]]; then
  case_ "parked pages: this host's /proc/meminfo and /proc/zoneinfo" 0 "source '$here/memavail.sh'" "
    read -r a p b <<<\"\$(mem_counting)\"; [[ \"\$a\" =~ ^[0-9]+\\.[0-9]\$ && \"\$b\" =~ ^[0-9]+\$ ]] || { echo \"[\$a \$p \$b]\"; exit 1; }"
fi

# ---- scripts/port-patches.sh's helpers (its runs against the engine repository: scripts/port_patches_test.sh)
# check_ <name> <script>: the script in a subshell; exit 0 passes
check_() {
  local name=$1 out got
  n=$((n + 1))
  out=$( ( eval "$2" ) 2>&1 ); got=$?
  if [[ "$got" == 0 ]]; then printf 'ok   %s\n' "$name"
  else fails=$((fails + 1)); printf 'FAIL %s (exit %s)\n%s\n' "$name" "$got" "$(sed 's/^/     /' <<<"$out")"; fi
}
check_ "port-patches: newreadme keeps the prose and is idempotent, its rows after the last hand-written row" '
  PORT_PATCHES_LIB=1; source "$here/port-patches.sh"
  t=$(mktemp -d "${TMPDIR:-/tmp}/port-test.XXXXXX"); trap "rm -rf -- \"$t\"" EXIT
  readme="$here/../zig/patches/README.md"
  prose=$(grep -F "<!-- port-patches -->" "$readme" | grep -v "^|" | head -1)
  [[ -n "$prose" ]] || { echo "the README has no prose line naming the marker: the test needs one"; exit 1; }
  printf "| \`0014-zig-glm-port.patch\` | x | y <!-- port-patches --> |\n" > "$t/rows"
  newreadme "$readme" "$t/rows" > "$t/a"; newreadme "$t/a" "$t/rows" > "$t/b"
  cmp -s "$t/a" "$t/b" || { echo "a second run changed the README:"; diff "$t/a" "$t/b"; exit 1; }
  grep -qxF "$prose" "$t/a" || { echo "the prose line was dropped: $prose"; exit 1; }
  [[ "$(grep -cE "$ROW_RE" "$t/a")" == 1 ]] || { echo "not exactly one generated row"; exit 1; }
  [[ "$(grep -vE "$ROW_RE" "$t/a")" == "$(grep -vE "$ROW_RE" "$readme")" ]] || { echo "hand-written lines changed"; exit 1; }
  last=$(grep -nE "^[|] \`[0-9]+-" "$t/a" | grep -v "port-patches" | tail -1 | cut -d: -f1)
  [[ "$(sed -n "$((last + 1))p" "$t/a")" == "$(cat "$t/rows")" ]] || { echo "the row is not after the last hand-written row"; exit 1; }'
check_ "port-patches: slug and patchlike" '
  PORT_PATCHES_LIB=1; source "$here/port-patches.sh"
  [[ "$(slug "glm_cuda: the startup budgets count the pages parked on the per-CPU free lists (port of GLM 0119)")" == \
     glm-cuda-the-startup-budgets-count-the-pages ]] || { slug "glm_cuda: the startup budgets count the pages parked"; exit 1; }
  [[ "$(slug "ZIG: Parked pages!")" == parked-pages ]] || exit 1
  w4="abcdefghij abcdefghij abcdefghij abcdefghij"; d4=abcdefghij-abcdefghij-abcdefghij-abcdefghij
  [[ "$(slug "$w4 abcd tail")" == "$d4-abcd" && "$(slug "$w4 abcdefg tail")" == "$d4" ]] || { slug "$w4 abcdefg tail"; exit 1; }
  [[ "$(slug "$(printf "x%.0s" {1..60})")" == "$(printf "x%.0s" {1..48})" ]] || exit 1
  t=$(mktemp "${TMPDIR:-/tmp}/port-test.XXXXXX"); trap "rm -f -- \"$t\"" EXIT
  printf "what it does\n  --- indented is fine\n- a list item\n" > "$t"; ! patchlike "$t" >/dev/null || exit 1
  for l in "diff --git a/x b/x" "--- a/x" "+++ b/x" "@@ -1 +1 @@" "GIT binary patch"; do
    printf "text\n%s\n" "$l" > "$t"; patchlike "$t" >/dev/null || { echo "not caught: $l"; exit 1; }
  done'

# ---- Robert's credit is never broken across lines (plain "Robert Wojciechowski (GitHub @wojo, X @rwojo)", Markdown
# "Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X)"): every text
# file that names him, the exported patches' headers (their engine commits' messages) included
# credit_split <file>...: "<file>:<line>" for each credit that starts on a line and ends on the next (the next line's
# indentation and comment marks dropped: # // //! * -- > and a patch's leading +)
credit_split() {
  awk '
    function norm(s) { sub(/^[ \t]*[+]?[ \t]*(#+|\/\/[\/!]?|[*]|--|>)?[ \t]*/, "", s); return s }
    FNR == 1 { prev = "" }
    {
      cur = $0; sub(/[ \t\r]+$/, "", cur)
      if (FNR > 1 && prev != "") {
        j = prev " " norm(cur); b = length(prev)
        for (k = 1; k <= 3; k++) {
          s = j; off = 0
          while (match(s, pat[k])) {
            st = off + RSTART; en = st + RLENGTH - 1
            if (st <= b && en > b) { print FILENAME ":" FNR - 1; break }
            off += RSTART + RLENGTH - 1; s = substr(s, RSTART + RLENGTH)
          }
        }
      }
      prev = cur
    }
    BEGIN {
      pat[1] = "Robert Wojciechowski \\(GitHub @wojo, X @rwojo\\)"
      pat[2] = "Robert Wojciechowski \\(\\[wojo\\]\\(https://github[.]com/wojo\\) on GitHub, \\[@rwojo\\]\\(https://x[.]com/rwojo\\) on X\\)"
      pat[3] = "Robert Wojciechowski \\((GitHub|\\[wojo\\])"
    }' "$@" | sort -u
}
check_ "credits: credit_split finds a credit broken anywhere in it, and only that" '
  t=$(mktemp -d "${TMPDIR:-/tmp}/port-test.XXXXXX"); trap "rm -rf -- \"$t\"" EXIT
  printf "%s\n" "# by Robert Wojciechowski (GitHub" "# @wojo, X @rwojo), ported" > "$t/a"
  printf "%s\n" "  (patch 0088, #78 by Robert Wojciechowski" "  ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X))" > "$t/b"
  printf "%s\n" "+//! the spill tier, by Robert" "+//! Wojciechowski (GitHub @wojo, X @rwojo)." > "$t/c"
  printf "%s\n" "Credits: Robert Wojciechowski (GitHub @wojo, X @rwojo); spill v2" "design: BadAd84" "by Robert Wojciechowski" "(https://github.com/wojo), not a credit of this form" > "$t/d"
  [[ "$(credit_split "$t/a" "$t/b" "$t/c" "$t/d" | tr "\n" " ")" == "$t/a:1 $t/b:1 $t/c:1 " ]] ||
    { credit_split "$t/a" "$t/b" "$t/c" "$t/d"; exit 1; }'
check_ "credits: no file here breaks Robert's credit across lines (zig/patches' exported headers included)" '
  files=$(cd "$root" && grep -rlI --exclude-dir=.git "Wojciechowski" . | sed "s|^\./||")
  [[ -n "$files" ]] || { echo "no file names him: the scan found nothing to check"; exit 1; }
  bad=$(cd "$root" && credit_split $files)
  [[ -z "$bad" ]] || { echo "broken across lines (keep each credit on one line):"; echo "$bad"; exit 1; }'

# ---- every script parses (bash -n, with this bash)
for f in "$root"/start.sh "$root"/start-tp3.sh "$root"/start-tp4.sh "$root"/start-tp4-switchless.sh "$root"/stop.sh "$here"/*.sh; do
  [[ -f "$f" ]] || continue
  n=$((n + 1))
  if out=$(bash -n "$f" 2>&1); then printf 'ok   bash -n %s\n' "${f#$root/}"
  else fails=$((fails + 1)); printf 'FAIL bash -n %s\n%s\n' "${f#$root/}" "$out"; fi
done

echo "$((n - fails)) of $n passed (bash $BASH_VERSION)"
(( fails == 0 ))
