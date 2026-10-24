#!/usr/bin/env bash
# Checks of scripts/port-patches.sh against the port's engine repository, without the Sparks: every engine commit used
# here is made in a throwaway clone (git clone --shared, removed at the end), never on the repository's own branches.
# The base alone (nothing to export), one commit (its header: the message, the credits with BadAd84's for spill, the
# apply line), several commits with --name, a merge as one patch, a squash, the refusals (the kernel set changed, a
# message line git apply would read, a base that is not ZIG_TF_REF plus the hand-written patches, a hand-written patch
# that no longer applies), and the writes on a copy of this recipe (README rows, earlier exports replaced, numbering
# after a new hand-written patch).
# bash 3.2 or later (runs on a Mac), git. ~1 minute.
# Usage: scripts/port_patches_test.sh [engine repository, default $PORT_PATCHES_ENGINE or ~/GitHub/tensorfold-glm]
#        (exits 1 when a check fails; 0 with a SKIP line when the repository or its base branch is not there)
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
engine=${1:-${PORT_PATCHES_ENGINE:-$HOME/GitHub/tensorfold-glm}}
base_ref=glm/zig-engine-base
if ! git -C "$engine" rev-parse --verify -q "$base_ref^{commit}" >/dev/null 2>&1; then
  echo "SKIP port-patches: no $base_ref in $engine (the port's engine repository)"; exit 0
fi
fails=0; n=0
tmp=$(mktemp -d "${TMPDIR:-/tmp}/port-patches-test.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT

# the throwaway engine: a shared clone, test branches from the base
eng=$tmp/engine
git clone -q --shared --no-checkout "$engine" "$eng" || { echo "FAIL cannot clone $engine"; exit 1; }
g() { git -C "$eng" -c user.name="Robert Wojciechowski" -c user.email=port-patches-test@example.invalid -c advice.detachedHead=false "$@"; }
B=origin/$base_ref
g checkout -q -b t/base "$B" || { echo "FAIL cannot check out $B"; exit 1; }
F=zig/src/families/glm_cuda/cuda_pool_capacity.zig
edit() { printf '\n// port-patches test: %s\n' "$1" >> "$eng/$F"; g add "$F"; }
g checkout -q -b t/one "$B"; edit one; g commit -q -m "glm_cuda: spill v2's test commit

What it does: a comment. Switches: none; off is byte-identical."
g checkout -q -b t/two "$B"; edit a; g commit -q -m "glm_cuda: first of two"; edit b; g commit -q -m "glm_cuda: second of two"
g checkout -q -b t/side "$B"; edit side1; g commit -q -m "side: one"; edit side2; g commit -q -m "side: two"
g checkout -q -b t/merge "$B"; g merge -q --no-ff -m "Merge branch 't/side' into t/merge" t/side
g checkout -q -b t/aot "$B"; mkdir -p "$eng/zig/tests/cuda/glm_p1/aot"; echo x > "$eng/zig/tests/cuda/glm_p1/aot/extra.json"
g add zig/tests/cuda/glm_p1/aot/extra.json; edit aot; g commit -q -m "glm_cuda: a new kernel capture"
g checkout -q -b t/notes "$B"; mkdir -p "$eng/docs/glm-port"; echo notes > "$eng/docs/glm-port/x.md"; g add docs/glm-port/x.md
g commit -q -m "docs/glm-port: notes only"
g checkout -q -b t/credits "$B"; edit credits
g commit -q -m "glm_cuda: a shard commit beside spill v2's tests

Touches nothing of the spill design.

Credits: Robert Wojciechowski (GitHub @wojo, X @rwojo); GLM patch 0118."
g checkout -q -b t/msg "$B"; edit msg; g commit -q -m "glm_cuda: a message that quotes a diff

--- a/x
+++ b/x"

# a copy of this recipe's script, pins and patches: the writes land there
rec=$tmp/recipe; mkdir -p "$rec/scripts" "$rec/zig"
cp "$here/port-patches.sh" "$here/config.sh" "$rec/scripts/"; cp -R "$root/zig/patches" "$rec/zig/"
# without the patches this recipe exported (0014 and later): the checks start from the hand-written ones
for f in "$rec"/zig/patches/*.patch; do head -n 1 "$f" | grep -q '^# port-patches.sh:' && rm -f -- "$f"; done
grep -v '^[|] .*<!-- port-patches -->' "$root/zig/patches/README.md" > "$rec/zig/patches/README.md"
pp() { /bin/bash "$rec/scripts/port-patches.sh" --engine "$eng" --base "$B" "$@" 2>&1; }

# check_ <name> <script>: the script in a subshell (with $out from it); exit 0 passes
check_() {
  local name=$1 o got
  n=$((n + 1))
  o=$( ( eval "$2" ) 2>&1 ); got=$?
  if [[ "$got" == 0 ]]; then printf 'ok   %s\n' "$name"
  else fails=$((fails + 1)); printf 'FAIL %s (exit %s)\n%s\n' "$name" "$got" "$(printf '%s\n' "$o" | tail -25 | sed 's/^/     /')"; fi
}
kept() { sed -n 's/^\[port-patches\] kept //p' <<<"$1"; }
has() { grep -qF -- "$2" <<<"$1" || { echo "missing: $2"; echo "$1"; exit 1; }; }

check_ "the base alone: ZIG_TF_REF + 0001-0013 = $base_ref, nothing to export (exit 0)" '
  out=$(pp --dry-run --branch t/base) || { echo "$out"; exit 1; }
  has "$out" "13 hand-written patches = $B"; has "$out" "nothing to export"'
check_ "one commit: 0014, its header (marker, message, credits with BadAd84 for spill, apply line), reproduces the branch" '
  out=$(pp --dry-run --keep --branch t/one --name t/one=glm-test) || { echo "$out"; exit 1; }
  d=$(kept "$out"); trap "rm -rf -- \"$d\"" EXIT; p=$d/out/0014-zig-glm-test.patch
  [[ -f "$p" ]] || { echo "no $p: $(ls "$d/out")"; exit 1; }
  has "$out" "ZIG_TF_REF + all patches = t/one"
  head -1 "$p" | grep -q "^# port-patches.sh: t/one @ " || { head -1 "$p"; exit 1; }
  sed -n 2p "$p" | grep -qx "glm_cuda: spill v2.s test commit" || { sed -n 2p "$p"; exit 1; }
  grep -q "^Credits: Robert Wojciechowski (GitHub @wojo, X @rwojo); spill v2.s block store design .*BadAd84" "$p" || { grep Credits "$p"; exit 1; }
  grep -qx "Apply from the checkout root with: git apply 0014-zig-glm-test.patch" "$p"
  [[ "$(grep -c "^diff --git " "$p")" == 1 ]]
  has "$out" "| \`0014-zig-glm-test.patch\` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X); block store design: [BadAd84]"'
check_ "a message with its own credits paragraph: kept as written, no second line, BadAd84 only where it names him" '
  out=$(pp --dry-run --keep --branch t/credits --name t/credits=credits) || { echo "$out"; exit 1; }
  d=$(kept "$out"); trap "rm -rf -- \"$d\"" EXIT; p=$d/out/0014-zig-credits.patch
  [[ "$(grep -c "^Credits:" "$p")" == 1 ]] || { grep Credits "$p"; exit 1; }
  ! grep -q BadAd84 "$p" || exit 1
  grep "<!-- port-patches -->" <<<"$out" | grep -q BadAd84 && { echo "$out"; exit 1; }
  has "$out" "| \`0014-zig-credits.patch\` | Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) |"'
check_ "two commits: 0014 and 0015, one named, one from its subject; no BadAd84 without spill" '
  out=$(pp --dry-run --keep --branch t/two --name t/two~1=first) || { echo "$out"; exit 1; }
  d=$(kept "$out"); trap "rm -rf -- \"$d\"" EXIT
  [[ "$(cd "$d/out" && ls)" == "0014-zig-first.patch
0015-zig-glm-cuda-second-of-two.patch" ]] || { ls "$d/out"; exit 1; }
  ! grep -q BadAd84 "$d/out/0014-zig-first.patch"'
check_ "a merge is one patch (its whole diff), listing the commits it brings in" '
  out=$(pp --dry-run --keep --branch t/merge --name t/merge=side) || { echo "$out"; exit 1; }
  d=$(kept "$out"); trap "rm -rf -- \"$d\"" EXIT; p=$d/out/0014-zig-side.patch
  [[ "$(ls "$d/out" | wc -l | tr -d " ")" == 1 ]] && grep -q "^Commits:" "$p" && grep -q "^- side: one" "$p" && grep -q "^- side: two" "$p"'
check_ "squash: one patch for the range, named by --name, its first line the default title (credit included)" '
  out=$(pp --dry-run --keep --branch t/two --mode squash --name glm-port-test) || { echo "$out"; exit 1; }
  d=$(kept "$out"); trap "rm -rf -- \"$d\"" EXIT
  [[ "$(cd "$d/out" && ls)" == 0014-zig-glm-port-test.patch ]] && grep -q "^- glm_cuda: first of two" "$d/out/0014-zig-glm-port-test.patch" || exit 1
  sed -n 2p "$d/out/0014-zig-glm-port-test.patch" | grep -qF "by Robert Wojciechowski (GitHub @wojo, X @rwojo), ported by him" ||
    { sed -n 2p "$d/out/0014-zig-glm-port-test.patch"; exit 1; }'
check_ "refused: a branch that changes the captured kernel set" '
  out=$(pp --dry-run --branch t/aot) && { echo "$out"; exit 1; }; has "$out" "changes the captured kernel set"'
check_ "notes only (docs/glm-port): nothing to export, exit 0" '
  out=$(pp --dry-run --branch t/notes) || { echo "$out"; exit 1; }; has "$out" "changes nothing outside"'
check_ "refused: a message line git apply would read as a patch" '
  out=$(pp --dry-run --branch t/msg) && { echo "$out"; exit 1; }; has "$out" "lines git apply would read as a patch"'
check_ "refused: a base that is not ZIG_TF_REF plus the hand-written patches" '
  ref=$(sed -n "s/^[[:space:]]*ZIG_TF_REF=\"\\\${ZIG_TF_REF:-\\([0-9a-f]*\\)}\".*/\\1/p" "$rec/scripts/config.sh")
  out=$(/bin/bash "$rec/scripts/port-patches.sh" --engine "$eng" --base "$ref" --branch t/one --dry-run 2>&1) && { echo "$out"; exit 1; }
  has "$out" "is not ZIG_TF_REF plus zig/patches"'
check_ "refused: --name for a commit outside the range, a bare --name with several commits, a bad slug" '
  out=$(pp --dry-run --branch t/two --name t/one=x) && exit 1; has "$out" "is not one of the commits exported"
  out=$(pp --dry-run --branch t/two --name x) && exit 1; has "$out" "names no commit"
  out=$(pp --dry-run --branch t/one --name "t/one=Bad_Slug") && exit 1; has "$out" "the slug is lowercase"'
check_ "refused: a hand-written patch that no longer applies (nothing written)" '
  cp -R "$rec" "$tmp/broken"; f=$tmp/broken/zig/patches/0013-zig-glm-round-batching.patch
  awk "/^@@/ && !done { print \"@@ -1,1 +1,1 @@\"; print \"-no such line\"; print \"+x\"; done = 1 } { print }" "$f" > "$f.new" && mv "$f.new" "$f"
  before=$(ls "$tmp/broken/zig/patches")
  out=$(/bin/bash "$tmp/broken/scripts/port-patches.sh" --engine "$eng" --base "$B" --branch t/one 2>&1) && { echo "$out"; exit 1; }
  has "$out" "does not apply on a clean ZIG_TF_REF"; [[ "$(ls "$tmp/broken/zig/patches")" == "$before" ]]'
check_ "write: one export, then a second replaces it (README rows rewritten, hand-written rows and prose kept)" '
  readme=$rec/zig/patches/README.md; cp "$readme" "$tmp/readme.orig"
  out=$(pp --branch t/one --name t/one=glm-test) || { echo "$out"; exit 1; }
  [[ -f "$rec/zig/patches/0014-zig-glm-test.patch" ]] || exit 1
  [[ "$(grep -c "<!-- port-patches -->" "$readme")" == "$(( $(grep -c "<!-- port-patches -->" "$tmp/readme.orig") + 1 ))" ]] || { grep -n port-patches "$readme"; exit 1; }
  out=$(pp --branch t/two) || { echo "$out"; exit 1; }
  has "$out" "removed 0014-zig-glm-test.patch"
  [[ "$(cd "$rec/zig/patches" && ls 001[4-9]*)" == "0014-zig-glm-cuda-first-of-two.patch
0015-zig-glm-cuda-second-of-two.patch" ]] || { ls "$rec/zig/patches"; exit 1; }
  [[ "$(grep -c "^| \`001[45]-.*<!-- port-patches -->" "$readme")" == 2 ]] || exit 1
  [[ "$(grep -v "<!-- port-patches -->" "$readme")" == "$(grep -v "<!-- port-patches -->" "$tmp/readme.orig")" ]] || { diff "$tmp/readme.orig" "$readme"; exit 1; }
  # the base alone again: nothing exported, the earlier export stays (and says so)
  out=$(pp --branch t/base) || { echo "$out"; exit 1; }; has "$out" "stay as they are"
  [[ -f "$rec/zig/patches/0015-zig-glm-cuda-second-of-two.patch" ]]'
check_ "write: a re-export without --name keeps each commit's patch name" '
  out=$(pp --branch t/two --name t/two~1=first-one --name t/two=second-one) || { echo "$out"; exit 1; }
  out=$(pp --branch t/two) || { echo "$out"; exit 1; }
  [[ "$(cd "$rec/zig/patches" && ls 001[4-9]*)" == "0014-zig-first-one.patch
0015-zig-second-one.patch" ]] || { ls "$rec/zig/patches"; exit 1; }'
check_ "write: a new hand-written 0014 moves the export after it (once the base contains it)" '
  # the base with a hand-written 0014: one more commit on the base, and its diff as the patch
  g checkout -q -b t/base2 "$B"; edit hand; g commit -q -m "hand-written fix"
  { echo "hand-written fix"; echo; git -C "$eng" diff --binary --full-index --no-renames "$B" t/base2; } > "$rec/zig/patches/0014-zig-hand-fix.patch"
  g checkout -q -b t/after t/base2; edit after; g commit -q -m "glm_cuda: after the hand fix"
  out=$(/bin/bash "$rec/scripts/port-patches.sh" --engine "$eng" --base t/base2 --branch t/after 2>&1) || { echo "$out"; exit 1; }
  [[ "$(cd "$rec/zig/patches" && ls 001[4-9]*)" == "0014-zig-hand-fix.patch
0015-zig-glm-cuda-after-the-hand-fix.patch" ]] || { ls "$rec/zig/patches"; exit 1; }
  grep -q "^| \`0014-zig-hand-fix" "$rec/zig/patches/README.md" && exit 1   # not a row of its own: the script owns only its rows
  grep -q "^| \`0015-zig-glm-cuda-after-the-hand-fix.patch\` .*<!-- port-patches -->" "$rec/zig/patches/README.md"'

echo "$((n - fails)) of $n passed (bash $BASH_VERSION, engine $engine)"
(( fails == 0 ))
