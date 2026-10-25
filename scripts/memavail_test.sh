#!/usr/bin/env bash
# Checks of scripts/memavail.sh without the Sparks: mem_counting on synthetic /proc/meminfo and /proc/zoneinfo files (the
# zoneinfo cases of the engine's parked-pages tests, as in the DeepSeek recipe), mem_avail_gb as start.sh calls it here
# and sends it to the workers, its gate (parked_counted), what start.sh and scripts/nodes.sh wire up, and `bash -n`.
# bash 3.2 or later (runs on a Mac); no docker, no ssh, no GPU.
# Usage: scripts/memavail_test.sh      (exits 1 when a check fails)
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
fails=0; n=0
tmp=$(mktemp -d "${TMPDIR:-/tmp}/memavail-test.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT

# case_ <name> <script>: runs it in a subshell over a fresh scripts/memavail.sh, with expect available
case_() {
  local name=$1 out
  n=$((n + 1))
  out=$( (
    expect() { [[ "$1" == "$2" ]] || { echo "got  [$1]"; echo "want [$2]"; exit 1; }; }
    # shellcheck source=scripts/memavail.sh
    source "$here/memavail.sh"
    eval "$2"
  ) 2>&1 ) && { echo "ok   $name"; return; }
  fails=$((fails + 1))
  echo "FAIL $name"
  sed 's/^/     /' <<<"$out"
}

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

# ---- mem_counting: "<available GiB> <parked GiB> <parked bytes>"
# mc <name> <meminfo> <zoneinfo> <want>
mc() { case_ "parked pages: $1" "expect \"\$(mem_counting '$mt/$2' '$mt/$3' 4096)\" '$4'"; }
gib() { awk -v b="$1" -v a="${2:-100}" 'BEGIN {printf "%.1f %.1f %.0f", a + b / 2^30, b / 2^30, b}'; }
mc "every CPU, zone and node, less each list's resting level (count - high_min)" meminfo z-many "$(gib $(( (200000 + 3143 + 2332 + 17 + 300000 + 400000 - 8 * 2146) * 4096 )))"
mc "an image copy's 6 GiB raise the budget by 6 GiB" meminfo z-six "106.0 6.0 6442450944"
mc "a warm start counts ~nothing (under 0.25 GiB)" meminfo z-warm "$(gib $((20 * 1440 * 4096)))"
mc "below the resting level: nothing" meminfo z-below "100.0 0.0 0"
mc "no high_min (before 6.7) rests the first 512 MiB" meminfo z-old "101.0 1.0 1073741824"
mc "no high_min, under 512 MiB: nothing" meminfo z-old-small "100.0 0.0 0"
mc "only the count: and high_min: lines under pagesets" meminfo z-only "100.0 0.0 40960"
mc "an unparsable count counts nothing" meminfo z-bad "100.0 0.0 0"
mc "carriage returns are blanks" meminfo z-crlf "$(gib $(( (600000 - 2146) * 4096 )))"
mc "capped at MemTotal" meminfo-high z-six "128.0 6.0 6442450944"
mc "no zoneinfo: MemAvailable alone" meminfo missing "100.0 0.0 0"
case_ "parked pages: no MemAvailable fails and prints nothing" "
  out=\$(mem_counting '$mt/meminfo-none' '$mt/z-many' 4096) && exit 1; [[ -z \"\$out\" ]]"
case_ "parked pages: the page size from getconf by default" "
  expect \"\$(mem_counting '$mt/meminfo' '$mt/z-six')\" \"\$(mem_counting '$mt/meminfo' '$mt/z-six' \$(getconf PAGESIZE))\""

# ---- mem_avail_gb: what start.sh's check reads, here and on each worker
case_ "start check: mem_avail_gb 1 is whole GiB (rounded down) and the parked GiB" "
  mem_counting() { echo '109.9 6.0 6442450944'; }; expect \"\$(mem_avail_gb 1)\" '109 6.0'"
case_ "start check: mem_avail_gb 1 on the synthetic Spark (100 GiB available, 6 GiB parked)" "
  eval \"\$(declare -f mem_counting | sed '1s/mem_counting/mem_counting_file/')\"
  mem_counting() { mem_counting_file '$mt/meminfo' '$mt/z-six' 4096; }
  expect \"\$(mem_avail_gb 1)\" '106 6.0'"
# free -g on a Spark right after an image copy (available 103), for the check of an engine that counts no parked pages
freestub="free() { printf '               total        used        free      shared  buff/cache   available\\nMem:             119          12           3           0         101         103\\nSwap:             15           0          15\\n'; }"
case_ "start check: mem_avail_gb 0 (the engine counts no parked pages) is free -g's available column alone, as before" "
  mem_counting() { echo '109.9 6.0 6442450944'; }; $freestub
  expect \"\$(mem_avail_gb 0)\" '103 0'
  expect \"\$(mem_avail_gb)\" '103 0'
  expect \"\$(bash -c \"\$(declare -f free mem_counting mem_avail_gb); mem_avail_gb 0\")\" '103 0'"
case_ "start check: mem_avail_gb runs on a worker as start.sh sends it (declare -f over ssh, here bash -c and sh -c)" "
  for sh in bash sh; do
    expect \"\$(\$sh -c \"\$(declare -f mem_counting); mem_counting '$mt/meminfo' '$mt/z-six' 4096\")\" '106.0 6.0 6442450944'
  done
  l=\$(bash -c \"\$(declare -f mem_counting mem_avail_gb); mem_avail_gb 1\")
  if [[ -r /proc/meminfo ]]; then [[ \"\$l\" =~ ^[0-9]+\\ [0-9]+\\.[0-9]\$ ]] || { echo \"[\$l]\"; exit 1; }; fi"

# ---- parked_counted: only an engine whose budget counts the parked pages
case_ "start check: parked_counted is the zig engine with parked in ZIG_HAS, the python one with patches/'s 0119" "
  mkdir -p '$tmp/r-none/patches' '$tmp/r-py/patches'; : > '$tmp/r-py/patches/0119-cuda-budget-parked-pages.patch'
  ( ENGINE=zig; ZIG_HAS='mtp displaykv parked'; parked_counted ) || { echo 'zig not counted with parked'; exit 1; }
  ( ENGINE=zig; ZIG_HAS=parked; parked_counted ) || { echo 'zig not counted with parked alone'; exit 1; }
  ( ENGINE=zig; ZIG_HAS='mtp displaykv'; ! parked_counted ) || { echo 'zig counted without parked'; exit 1; }
  ( ENGINE=zig; ZIG_HAS='mtp parked2'; ! parked_counted ) || { echo 'zig counted by another name'; exit 1; }
  ( ENGINE=zig; unset ZIG_HAS; ! parked_counted ) || { echo 'zig counted without ZIG_HAS'; exit 1; }
  ( ENGINE=zig; ZIG_HAS='mtp displaykv'; _cfg_root='$tmp/r-py'; ! parked_counted ) || { echo 'zig counted by the python patch'; exit 1; }
  ( ENGINE=python; ZIG_HAS=parked; _cfg_root='$tmp/r-none'; ! parked_counted ) || { echo 'python counted by ZIG_HAS'; exit 1; }
  ( ENGINE=python; _cfg_root='$tmp/r-py'; parked_counted ) || { echo 'python not counted with 0119'; exit 1; }"
# start.sh's default ZIG_HAS (its line ZIG_HAS="${ZIG_HAS:-...}")
zig_has_default() { sed -n 's/^ *ZIG_HAS="\${ZIG_HAS:-\(.*\)}"$/\1/p' "$1/start.sh"; }
case_ "start check: this recipe's zig engine counts them: start.sh's default ZIG_HAS names parked, and a zig/patches patch adds the engine's reading (by content, whatever its number or name)" "
  $(declare -f zig_has_default)
  d=\$(zig_has_default '$root')
  [[ -n \"\$d\" ]] || { echo 'start.sh sets no ZIG_HAS default'; exit 1; }
  ( ENGINE=zig; ZIG_HAS=\$d; parked_counted ) || { echo \"start.sh's default ZIG_HAS (\$d) lacks parked\"; exit 1; }
  grep -q '^+pub fn parkedBytes(' '$root'/zig/patches/*.patch || { echo 'no zig/patches patch adds parkedBytes'; exit 1; }"

# ---- what start.sh and scripts/nodes.sh wire up
check_in() { grep -qF -- "$2" "$root/$1" || { echo "$1 lacks: $2"; return 1; }; }
case_ "start.sh: the memory check passes the gate here and to each worker; nodes.sh sources memavail.sh" "
  $(declare -f check_in); root='$root'
  check_in start.sh 'PARKED=0; ! parked_counted || PARKED=1'
  check_in start.sh 'read -r here_gb here_parked <<<\"\$(mem_avail_gb \"\$PARKED\")\"'
  check_in start.sh '\"\$(declare -f mem_counting mem_avail_gb); mem_avail_gb \$PARKED\"'
  check_in scripts/nodes.sh '/scripts/memavail.sh\"'
  ! grep -v '^ *#' '$root/start.sh' | grep -q 'free -g' || { echo 'start.sh still reads free -g itself'; exit 1; }"
case_ "bash -n: start.sh, scripts/nodes.sh, scripts/memavail.sh" "
  for f in start.sh scripts/nodes.sh scripts/memavail.sh scripts/memavail_test.sh; do bash -n '$root/'\$f || exit 1; done"

echo "$((n - fails)) of $n passed"
(( fails == 0 ))
