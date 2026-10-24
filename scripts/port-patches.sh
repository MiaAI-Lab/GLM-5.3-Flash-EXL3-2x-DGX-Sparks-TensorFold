#!/usr/bin/env bash
# Export the GLM engine's port commits (the GLM-5.3-Flash patches 0088/0110/0111/0118/0119 by
# Robert Wojciechowski (GitHub @wojo, X @rwojo), spill v2, continued chats, KV sharding and parked pages, ported to this
# Zig engine) as zig/patches/0014-zig-<name>.patch and later, numbered after the last hand-written patch, which
# zig/Dockerfile applies with `git apply` after 0001-0013 in filename order. Nothing is written unless every step holds:
#   1. a clean checkout of ZIG_TF_REF (the engine pin, scripts/config.sh) gets the hand-written zig/patches in filename
#      order, exactly as the Dockerfile applies them, and must then equal BASE outside the excluded paths
#   2. the patches: one per first-parent commit of BASE..BRANCH (--mode commits, the default; a merge is its whole diff
#      against its first parent), or one for the range (--mode squash), without the excluded paths. A patch's header is
#      its commit's message (what it does, its switches and defaults, what was measured) and a credits line
#   3. each new patch passes `git apply --check` there after the ones before it and applies, and the result must equal
#      BRANCH outside the excluded paths
#   4. only then (not with --dry-run): the previously exported patches (first line "# port-patches.sh") are replaced and
#      zig/patches/README.md's generated table rows (marked <!-- port-patches -->) rewritten
# Excluded: the captured kernel set zig/tests/cuda/glm_p1/aot (it ships as zig/aot/glm-aot-sm121.tar.xz, written by
# tools/make-zig-patches.sh; a BRANCH that changes it is refused) and the port's notes, docs/glm-port/. BRANCH equal to
# BASE checks step 1 and exports nothing. The image tag (scripts/config.sh, image_hash) changes with the patches, so the
# next scripts/prepare.sh builds a new image.
#
# Usage: scripts/port-patches.sh [--dry-run] [--engine DIR] [--branch REF] [--base REF] [--mode commits|squash]
#                                [--name [REV=]SLUG]... [--title TEXT] [--exclude PATH]... [--keep]
#   --dry-run         build and check the patches in a temporary folder, print what would change, write nothing
#   --engine DIR      the engine repository (default ~/GitHub/tensorfold-glm)
#   --branch REF      what to export (default glm/series-r4: the port's per-feature commits, rounds 1-4)
#   --base REF        what it is relative to: ZIG_TF_REF plus the hand-written patches (default glm/zig-engine-base)
#   --mode            commits (default: one patch per first-parent commit) or squash (one patch for BASE..BRANCH)
#   --name REV=SLUG   commits: the patch of commit REV is NNNN-zig-SLUG.patch (repeatable; a commit without one keeps
#                     the name of the patch already exported from it, else is named from its subject); squash: --name
#                     SLUG (default glm-port)
#   --title TEXT      squash: the patch's first header line (default: the port's summary)
#   --exclude PATH    leave PATH out of the patches and the comparisons as well (repeatable)
#   --keep            keep the temporary folder (printed) for inspection
# Env: ZIG_TF_REF, TF_REPO (default: scripts/config.sh's pins). ZIG_TF_REF is taken from the engine repository when it
# has the commit, else fetched from TF_REPO (network). Runs on Linux and macOS (bash 3.2 or later, git).
[[ "${PORT_PATCHES_LIB:-0}" == 1 ]] || set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
die()  { printf '[port-patches] ERROR: %s\n' "$*" >&2; exit 1; }
log()  { printf '[port-patches] %s\n' "$*"; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
# A generated row: a table row (it starts with "| ") that carries the marker. Prose that names the marker is kept.
ROW_RE='^[|] .*<!-- port-patches -->'   # [|], not \|: awk -v would read \| as an escape
newreadme() {  # <readme> <rows file>: the README with the generated rows replaced, after the table's last hand-written row
  awk -v rows="$2" -v re="$ROW_RE" '
    $0 ~ re { next }
    { lines[++n] = $0 }
    END {
      last = 0
      for (i = 1; i <= n; i++) if (lines[i] ~ /^\| `[0-9]+-/) last = i
      for (i = 1; i <= n; i++) {
        print lines[i]
        if (i == last) while ((getline r < rows) > 0) print r
      }
    }' "$1"
}
# patchlike <file>: the lines of a header that git apply could read as part of a patch (it parses from the first of them)
patchlike() { grep -nE '^(diff |--- |[+][+][+] |@@ |Index: |GIT binary patch|Binary files )' "$1"; }
# slug <subject>: a file name part from a commit subject (lowercase, a-z 0-9 -, at most 48, cut after a whole word),
# without a leading "zig-"
slug() {
  local s
  s=$(printf '%s\n' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/^zig-//')
  if (( ${#s} > 48 )); then
    if [[ "${s:48:1}" == - ]]; then s=${s:0:48}; else s=${s:0:48}; s=${s%-*}; fi
  fi
  printf '%s\n' "${s%-}"
}
# Sourced with PORT_PATCHES_LIB=1 (scripts/port_test.sh): define the helpers above and stop here
[[ "${PORT_PATCHES_LIB:-0}" != 1 ]] || return 0

engine="$HOME/GitHub/tensorfold-glm"; branch=glm/series-r4; base=glm/zig-engine-base; mode=commits
title="GLM-5.3-Flash patches 0088/0110/0111/0118/0119 (spill v2, continued chats, KV sharding, parked pages) by Robert Wojciechowski (GitHub @wojo, X @rwojo), ported by him to the GLM Zig engine"
dry=0; keep=0; excludes=(zig/tests/cuda/glm_p1/aot docs/glm-port); names=()
AOT=zig/tests/cuda/glm_p1/aot
while (( $# )); do
  case "$1" in
    --dry-run) dry=1 ;;
    --keep) keep=1 ;;
    --engine) engine=${2:?--engine needs a directory}; shift ;;
    --branch) branch=${2:?--branch needs a ref}; shift ;;
    --base) base=${2:?--base needs a ref}; shift ;;
    --mode) mode=${2:?--mode needs commits or squash}; shift ;;
    --name) names+=("${2:?--name needs [REV=]SLUG}"); shift ;;
    --title) title=${2:?--title needs text}; shift ;;
    --exclude) excludes+=("${2:?--exclude needs a path}"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (--help)" ;;
  esac
  shift
done
[[ "$mode" == commits || "$mode" == squash ]] || die "--mode is commits or squash, not $mode"

# the pins, as scripts/config.sh sets them (read, not sourced: config.sh needs bash 4 and a Spark's tools)
pin() { sed -n "s/^[[:space:]]*$1=\"\\\${$1:-\\([^}]*\\)}\".*/\\1/p" "$root/scripts/config.sh" | head -1; }
TF_REF=${ZIG_TF_REF:-$(pin ZIG_TF_REF)}; TF_REPO=${TF_REPO:-$(pin TF_REPO)}
[[ "$TF_REF" =~ ^[0-9a-f]{40}$ ]] || die "ZIG_TF_REF is not a full commit hash: '$TF_REF'"

[[ -d "$engine" ]] && engine=$(cd "$engine" && pwd)
git -C "$engine" rev-parse --git-dir >/dev/null 2>&1 || die "$engine is not a git repository (--engine)"
branch_sha=$(git -C "$engine" rev-parse --verify -q "$branch^{commit}") || die "no $branch in $engine (--branch)"
base_sha=$(git -C "$engine" rev-parse --verify -q "$base^{commit}") || die "no $base in $engine (--base)"
git -C "$engine" merge-base --is-ancestor "$base_sha" "$branch_sha" || die "$base is not an ancestor of $branch: rebase or merge it first"

spec=(.)
for x in "${excludes[@]}"; do spec+=(":(exclude)$x"); done
diffopts=(--binary --full-index --no-renames --no-color --no-ext-diff --src-prefix=a/ --dst-prefix=b/)

# the --name pairs: commit -> slug (bash 3.2 has no associative arrays: two lists)
name_shas=(); name_slugs=(); bare=""
for x in ${names[@]+"${names[@]}"}; do
  s=${x##*=}; s=${s#zig-}
  [[ "$s" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "--name: the slug is lowercase a-z 0-9 -, not '$s'"
  if [[ "$x" == *=* ]]; then
    [[ "$mode" == commits ]] || die "--name REV=SLUG is for --mode commits; squash takes --name SLUG"
    c=$(git -C "$engine" rev-parse --verify -q "${x%%=*}^{commit}") || die "--name: no commit ${x%%=*} in $engine"
    name_shas+=("$c"); name_slugs+=("$s")
  else
    bare=$s
  fi
done
old_name() {  # <commit>: the slug of the exported patch already here that came from it, or nothing
  local i
  for i in ${old_shas[@]+"${!old_shas[@]}"}; do [[ "$1" != "${old_shas[$i]}"* ]] || { echo "${old_slugs[$i]}"; return; }; done
}
name_of() {  # <commit>: its --name slug, or nothing
  local i
  for i in ${name_shas[@]+"${!name_shas[@]}"}; do [[ "${name_shas[$i]}" != "$1" ]] || { echo "${name_slugs[$i]}"; return; }; done
}

# the patches already there: hand-written ones stay and set the next number; exported ones (first line) are replaced
generated() { head -n 1 "$1" 2>/dev/null | grep -q '^# port-patches.sh:'; }
hand=(); old=(); last=0; old_shas=(); old_slugs=()
for p in $(cd "$root/zig/patches" && LC_ALL=C ls -1 -- *.patch 2>/dev/null); do
  if generated "$root/zig/patches/$p"; then
    old+=("$p")
    # the commit it came from (its first line's "(commits FROM..TO)"): a re-export keeps the patch's name for it
    c=$(head -n 1 "$root/zig/patches/$p" | sed -n 's/.*(commits [0-9a-f]*\.\.\([0-9a-f]*\)).*/\1/p')
    s=${p#[0-9][0-9][0-9][0-9]-zig-}; s=${s%.patch}
    [[ -z "$c" ]] || { old_shas+=("$c"); old_slugs+=("$s"); }
    continue
  fi
  hand+=("$p")
  [[ "$p" =~ ^([0-9]+)- ]] && (( 10#${BASH_REMATCH[1]} > last )) && last=$((10#${BASH_REMATCH[1]}))
done
(( ${#hand[@]} )) || die "no hand-written patches in zig/patches: refusing to guess the base"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/port-patches.XXXXXX")
if (( keep )); then trap 'log "kept $tmp"' EXIT; else trap 'rm -rf -- "$tmp"' EXIT; fi
mkdir -p "$tmp/out"

# ---------------------------------------------------------------- 1. the patches
# message <out file> <subject or empty> <commit>...: the header text of one patch (the commits' messages, or the subject
# and the commits' subjects), with a credits line unless it has one. Its authors: the commits' (a merge's: those it
# brings in); BadAd84's block store design is credited when the patch or its message is about spill, or, for a message
# with its own credits paragraph, when that paragraph names him (the README row follows it).
message() {
  local out=$1 subject=$2 c who by md spill=0 authors=() a seen
  shift 2
  {
    if [[ -n "$subject" ]]; then
      printf '%s\n\n' "$subject"
      for c in "$@"; do printf -- '- %s\n' "$(git -C "$engine" log -1 --format=%s "$c")"; done
    else
      git -C "$engine" log -1 --format=%B "$1" | awk '{ l[++n] = $0 } END { while (n && l[n] ~ /^[[:space:]]*$/) n--; for (i = 1; i <= n; i++) print l[i] }'
      if (( $# > 1 )); then
        printf '\nCommits:\n'
        for c in "${@:2}"; do printf -- '- %s\n' "$(git -C "$engine" log -1 --format=%s "$c")"; done
      fi
    fi
  } > "$out"
  for c in "$@"; do
    a=$(git -C "$engine" log -1 --format=%an "$c"); seen=0
    for who in ${authors[@]+"${authors[@]}"}; do [[ "$who" != "$a" ]] || seen=1; done
    (( seen )) || authors+=("$a")
  done
  grep -qi spill "$out" && spill=1
  for c in "$@"; do git -C "$engine" diff --name-only "$c^1" "$c" -- "${spec[@]}" | grep -qi spill && spill=1; done
  # a message with a credits paragraph of its own is credited as it says: BadAd84's design only where it names him
  if grep -q '^Credits:' "$out"; then spill=0; sed -n '/^Credits:/,/^[[:space:]]*$/p' "$out" | grep -q BadAd84 && spill=1; fi
  by=""; md=""
  for a in ${authors[@]+"${authors[@]}"}; do
    if [[ "$a" == "Robert Wojciechowski" ]]; then
      by+="${by:+, }Robert Wojciechowski (GitHub @wojo, X @rwojo)"
      md+="${md:+, }Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X)"
    else by+="${by:+, }$a"; md+="${md:+, }$a"; fi
  done
  if (( spill )); then
    by+="; spill v2's block store design (2,048-token blocks, lineage, the write-at-keep ring, release and fsync order): BadAd84 (https://github.com/BadAd84), MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold#118"
    md+="; block store design: [BadAd84](https://github.com/BadAd84) (#118)"
  fi
  grep -q '^Credits:' "$out" || printf '\nCredits: %s.\n' "$by" >> "$out"
  by_md=$md
}
num=$last; new=(); subjects=(); bys=(); shas=()
emit() {  # <from> <to> <slug> <message file> <subject>
  local f diff=$tmp/diff
  git -C "$engine" diff "${diffopts[@]}" "$1" "$2" -- "${spec[@]}" > "$diff"
  if ! grep -q '^diff --git ' "$diff"; then
    log "skip $(git -C "$engine" rev-parse --short=12 "$2") ($5): nothing outside the excluded paths"; return 0
  fi
  if patchlike "$4" >"$tmp/bad"; then
    sed 's/^/    /' "$tmp/bad" >&2
    die "the message of $(git -C "$engine" rev-parse --short=12 "$2") ($5) has lines git apply would read as a patch (above): reword them (indent them, say), then export again"
  fi
  num=$((num + 1)); f=$(printf '%04d-zig-%s.patch' "$num" "$3")
  {
    printf '# port-patches.sh: %s @ %s over %s @ %s (%s %s..%s); regenerate with scripts/port-patches.sh, do not edit\n' \
      "$branch" "${branch_sha:0:12}" "$base" "${base_sha:0:12}" "$mode" \
      "$(git -C "$engine" rev-parse --short=12 "$1")" "$(git -C "$engine" rev-parse --short=12 "$2")"
    cat "$4"
    printf 'Apply from the checkout root with: git apply %s\n\n' "$f"
    cat "$diff"
  } > "$tmp/out/$f"
  new+=("$f"); subjects+=("$5"); bys+=("$by_md"); shas+=("$(git -C "$engine" rev-parse --short=12 "$2")")
}
if [[ "$mode" == squash ]]; then
  if [[ "$base_sha" != "$branch_sha" ]]; then
    message "$tmp/msg" "$title" $(git -C "$engine" rev-list --reverse --no-merges "$base_sha..$branch_sha")
    emit "$base_sha" "$branch_sha" "${bare:-glm-port}" "$tmp/msg" "$title"
  fi
else
  commits=($(git -C "$engine" rev-list --first-parent --reverse "$base_sha..$branch_sha"))
  [[ -z "$bare" ]] || (( ${#commits[@]} == 1 )) || die "--name $bare names no commit: with --mode commits use --name REV=SLUG (${#commits[@]} commits)"
  for c in ${name_shas[@]+"${name_shas[@]}"}; do
    [[ " ${commits[*]-} " == *" $c "* ]] || die "--name: ${c:0:12} is not one of the commits exported ($base..$branch, first parent)"
  done
  for c in ${commits[@]+"${commits[@]}"}; do
    s=$(git -C "$engine" log -1 --format=%s "$c")
    if [[ -n "$(git -C "$engine" rev-list --no-walk --merges "$c")" ]]; then   # a merge: the commits it brings in
      message "$tmp/msg" "" "$c" $(git -C "$engine" rev-list --reverse --no-merges "$c^1..$c")
    else
      message "$tmp/msg" "" "$c"
    fi
    n=$(name_of "$c"); [[ -n "$n" ]] || n=$(old_name "$c"); [[ -n "$n" ]] || n=${bare:-$(slug "$s")}
    [[ -n "$n" ]] || n=commit-$(git -C "$engine" rev-parse --short=8 "$c")
    emit "$c^1" "$c" "$n" "$tmp/msg" "$s"
  done
fi

# ---------------------------------------------------------------- 2. a clean ZIG_TF_REF with the hand-written patches = BASE
log "Checking on a clean checkout of ZIG_TF_REF ${TF_REF:0:12}"
if git -C "$engine" cat-file -e "$TF_REF^{commit}" 2>/dev/null; then
  git clone -q --no-checkout --shared "$engine" "$tmp/tf"
else
  log "ZIG_TF_REF is not in $engine: fetching it from $TF_REPO"
  git init -q "$tmp/tf"; git -C "$tmp/tf" remote add origin "$TF_REPO"
  git -C "$tmp/tf" fetch -q --depth 1 origin "$TF_REF" || die "cannot fetch ZIG_TF_REF from $TF_REPO"
  # BASE and BRANCH from the engine repository, for the comparisons
  git -C "$tmp/tf" fetch -q "$engine" "$base_sha" "$branch_sha" || die "cannot read $base and $branch from $engine"
fi
git -C "$tmp/tf" -c advice.detachedHead=false checkout -q --detach "$TF_REF"
[[ "$(git -C "$tmp/tf" rev-parse HEAD)" == "$TF_REF" ]] || die "the checkout is not at ZIG_TF_REF"
apply() {  # <patch file> <label>: as zig/Dockerfile does (git apply --check, then git apply, in the checkout's root)
  ( cd "$tmp/tf" && git apply --check "$1" && git apply "$1" ) 2>"$tmp/apply.err" ||
    { sed 's/^/    /' "$tmp/apply.err" >&2; die "$2 does not apply on a clean ZIG_TF_REF (git apply --check): nothing was written"; }
}
same_as() {  # <commit>: the checkout (untracked and ignored files too: -f) equals <commit> outside the excludes
  git -C "$tmp/tf" add -A -f >/dev/null
  git -C "$tmp/tf" diff --cached --quiet "$1" -- "${spec[@]}"
}
for p in "${hand[@]}"; do apply "$root/zig/patches/$p" "zig/patches/$p"; done
if ! same_as "$base_sha"; then
  git -C "$tmp/tf" diff --cached --stat "$base_sha" -- "${spec[@]}" | tail -15 >&2
  die "$base is not ZIG_TF_REF plus zig/patches ${hand[*]} (the differences above, outside ${excludes[*]}): export relative to the base the image is built from"
fi
log "ZIG_TF_REF + ${#hand[@]} hand-written patches = $base (${base_sha:0:12}) outside ${excludes[*]}"
git -C "$engine" diff --quiet "$base_sha" "$branch_sha" -- "$AOT" ||
  die "$branch changes the captured kernel set ($AOT), which no patch carries: it ships as zig/aot/glm-aot-sm121.tar.xz (tools/make-zig-patches.sh writes it); export the kernel set first, then the code"
if (( ${#new[@]} == 0 )); then
  if [[ "$base_sha" == "$branch_sha" ]]; then log "$branch is $base: nothing to export, zig/patches unchanged"
  else log "$base..$branch changes nothing outside ${excludes[*]}: nothing to export, zig/patches unchanged"; fi
  (( ${#old[@]} == 0 )) || log "the patches it exported before stay as they are: ${old[*]} (remove them by hand if $branch no longer has them)"
  exit 0
fi

# ---------------------------------------------------------------- 3. the new patches apply, and reproduce BRANCH
for p in "${new[@]}"; do apply "$tmp/out/$p" "$p"; log "  $p applies ($(grep -c '^diff --git ' "$tmp/out/$p") files, $(wc -c <"$tmp/out/$p" | tr -d ' ') bytes)"; done
if ! same_as "$branch_sha"; then
  git -C "$tmp/tf" diff --cached --stat "$branch_sha" -- "${spec[@]}" | tail -15 >&2
  die "the exported patches applied but do not reproduce $branch (above): nothing was written"
fi
log "ZIG_TF_REF + all patches = $branch (${branch_sha:0:12}) outside ${excludes[*]}"

# ---------------------------------------------------------------- 4. write (not with --dry-run)
readme="$root/zig/patches/README.md"
[[ -f "$readme" ]] || die "no $readme to list the patches in"
rows() {
  local i
  for i in "${!new[@]}"; do
    printf '| `%s` | %s | %s (engine `%s` @ `%s`; its switches, defaults and measurements: the header; exported by `scripts/port-patches.sh`) <!-- port-patches --> |\n' \
      "${new[$i]}" "${bys[$i]}" "${subjects[$i]//|/\\|}" "$branch" "${shas[$i]}"
  done
}
rows > "$tmp/rows"
newreadme "$readme" "$tmp/rows" > "$tmp/README.md"
grep -qE "$ROW_RE" "$tmp/README.md" || die "found no patch table in $readme to add the rows to"
if (( dry )); then
  log "DRY RUN: would remove ${old[*]:-nothing} and write ${new[*]} into zig/patches, and update its README:"
  diff -u "$readme" "$tmp/README.md" | sed 's/^/    /' || true
  exit 0
fi
for p in ${old[@]+"${old[@]}"}; do rm -f -- "$root/zig/patches/$p"; done
for p in "${new[@]}"; do cp -- "$tmp/out/$p" "$root/zig/patches/$p"; done
cp -- "$tmp/README.md" "$readme"
log "Wrote ${new[*]} (removed ${old[*]:-none}); zig/patches/README.md updated. The image tag changes: scripts/prepare.sh builds it"
