# The memory a Spark can give a starting rank, counted the way the engine's startup budget counts it: MemAvailable plus
# the pages parked on the kernel's per-CPU free lists (wojo's GLM patch 0119, PR #149; the Zig engine's
# native/cuda_memory.zig parkedBytes, as ported to the DeepSeek recipe's scripts/memavail.sh, from which this is copied).
# start.sh counts them only for an engine that does (scripts/port.sh: port_parked), and reads `free -g` as before
# otherwise: ENGINE=zig does with zig/patches/0014 (0119's port, glm_cuda/cuda_cache_plan.zig parkedBytes; ZIG_HAS
# parked, which start.sh's default lists), ENGINE=python only with patches/0119-cuda-budget-parked-pages.patch, which
# patches/ does not carry.
# Sourced by scripts/nodes.sh; start.sh sends mem_counting to each worker over ssh (self-contained: awk and getconf only).
# bash 3.2 and POSIX awk (mawk, gawk, BSD awk), so that scripts/port_test.sh runs it on a Mac against synthetic /proc
# files.
#
# A bulk free (prepare.sh's docker save | docker load of the image, a compiler's exit, the first start of a new image)
# leaves 2-9 GiB on the per-CPU free lists for a minute or two. MemFree and MemAvailable do not count them, so a check
# right after an image copy reads several GiB less than an engine with 0119 (which counts them) has.
#
# One rule, the same as the engine's 0119 (and not the Python fork's _parked_zone, which rested each list 256 MiB above
# high_min): in /proc/zoneinfo, under each zone's `pagesets` line (the indented lines that follow it: four spaces, or the
# "  vm stats threshold" line), sum every `count:` (pages on a CPU's list) and every `high_min:` (that list's resting
# level); parked = count - high_min pages, or count pages less 512 MiB when the kernel prints no high_min (before 6.7);
# never below 0. Any unparsable count or high_min counts nothing (0), as the engine does. Then available = MemAvailable
# + parked, at most MemTotal. /proc/meminfo is read first, /proc/zoneinfo right after (pages that drain in between are
# counted by neither, never twice).

# mem_counting [meminfo] [zoneinfo] [page bytes]: "<available GiB> <parked GiB> <parked bytes>" (GiB with one
# decimal), from /proc/meminfo, /proc/zoneinfo and getconf PAGESIZE by default. A missing zoneinfo counts no parked
# pages. Prints nothing (and fails) when the meminfo has no MemAvailable.
mem_counting() {
  local mi=${1:-/proc/meminfo} zi=${2:-/proc/zoneinfo} pg=${3:-}
  [ -n "$pg" ] || pg=$(getconf PAGESIZE 2>/dev/null)
  case "$pg" in ''|*[!0-9]*) pg=4096 ;; esac
  [ -r "$zi" ] || zi=/dev/null
  awk -v page="$pg" -v rest_default=536870912 -v zfile="$zi" '
    FILENAME != zfile {
      if ($1 == "MemTotal:") total = $2 * 1024
      else if ($1 == "MemAvailable:") { avail = $2 * 1024; seen = 1 }
      next
    }
    {
      row = $0
      gsub(/\r/, " ", row)
      t = row; sub(/^[ \t]+/, "", t); sub(/[ \t]+$/, "", t)
      if (t == "pagesets") { listed = 1; next }
      if (listed && substr($0, 1, 4) != "    " && substr($0, 1, 10) != "  vm stats") { listed = 0; next }
      if (!listed) next
      n = split(t, w, /[ \t]+/)
      if (w[1] != "count:" && w[1] != "high_min:") next
      if (n < 2 || w[2] !~ /^[0-9]+$/) { bad = 1; next }
      if (w[1] == "count:") count += w[2]; else rest += w[2]
    }
    END {
      if (!seen) exit 1
      parked = 0
      if (!bad && count > 0) {
        resting = rest > 0 ? rest * page : rest_default
        parked = count * page - resting
        if (parked < 0) parked = 0
      }
      a = avail + parked
      if (total > 0 && a > total) a = total
      printf "%.1f %.1f %.0f\n", a / 1073741824, parked / 1073741824, parked
    }' "$mi" "$zi"
}

# mem_avail_gb [<parked 0|1>]: "<whole GiB available> <parked GiB>" on this node for start.sh's memory check. With 1
# (the engine counts the parked pages: port_parked) mem_counting's figure, rounded down as `free -g` shows it; with 0
# (the default), or when /proc/meminfo has no MemAvailable, `free -g`'s available column and 0, as before the port.
# Self-contained with mem_counting (start.sh sends both to the workers).
mem_avail_gb() {
  local m
  if [ "${1:-0}" = 1 ] && m=$(mem_counting 2>/dev/null) && [ -n "$m" ]; then
    echo "$m" | awk '{ printf "%d %s\n", $1, $2 }'
  else
    free -g 2>/dev/null | awk '/^Mem:/ { print $7, 0 }'
  fi
}
