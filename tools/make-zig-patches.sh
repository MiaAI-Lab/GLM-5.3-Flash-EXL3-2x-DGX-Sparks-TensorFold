#!/usr/bin/env bash
# Maintainers: write zig/patches/*.patch and zig/aot/glm-aot-sm121.tar.xz from the Zig GLM port's branch, then check that
# the patches, applied in filename order with git apply to a fresh TensorFold TF_REF tree, give exactly the branch's files
# (minus the notes, capture tools and kernel set listed below).
#   tools/make-zig-patches.sh [--force] <Zig port checkout> [branch, default r107-int] [base ref, default v1.0.7]
# The port is TensorFold's Zig engine (tensorfold-native, ashhart/TensorFold) plus the Flash Next Zig CUDA base and the
# GLM engine on top. Each file lands in exactly one patch, by path:
#   0001 shared runtime   every file the port modifies (lane host, native/cuda, server, build)
#   0002 Flash Next CUDA  the Flash Next Zig CUDA engine and kernels the GLM engine builds on
#   0003 GLM kernels      zig/kernels/cuda/glm_*
#   0004 GLM engine       zig/src/families/glm_cuda, the GLM CLI and host
#   0005 GLM tests        the rest under zig/ (tests, fixtures)
# Not shipped: notes-glm/, tools/ (capture and reference harnesses), the port's HANDOFF/REPORT files, and the captured
# Triton kernel set zig/tests/cuda/glm_p1/aot, which goes out as zig/aot/glm-aot-sm121.tar.xz (cubins captured on a GB10
# from the Python engine; it cannot be rebuilt without a GPU and the reference engine).
# It writes 0001-0005 only and removes every zig/patches/*.patch first, so it refuses while any patch after 0005 is there
# (0006-0016, the three 0014 base fixes among them, are hand-written patches; 0017 and later are scripts/port-patches.sh's
# export of the engine's port commits): --force deletes them too.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
force=0
if [[ "${1:-}" == --force ]]; then force=1; shift; fi
later=$(cd zig/patches && LC_ALL=C ls -1 -- *.patch 2>/dev/null | awk -F- '$1 ~ /^[0-9]+$/ && $1 + 0 > 5' | tr '\n' ' ') || true
if [[ -n "$later" ]] && (( ! force )); then
  echo "zig/patches holds patches after 0005: ${later% }. This tool writes 0001-0005 only and removes every *.patch first, so it would delete them (0006-0016, the three 0014 base fixes among them, are hand-written patches; 0017 and later are scripts/port-patches.sh's export). Move them aside and put them back after (rebased on the new 0001-0005), or pass --force to delete them; nothing was changed" >&2
  exit 1
fi
repo=$(readlink -f "${1:?usage: tools/make-zig-patches.sh [--force] <Zig port checkout> [branch] [base ref]}")
branch=${2:-r107-int}
base=${3:-v1.0.7}
out=$PWD/zig
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git -C "$repo" rev-parse --verify -q "$base^{commit}" >/dev/null || { echo "no $base in $repo" >&2; exit 1; }
git -C "$repo" rev-parse --verify -q "$branch^{commit}" >/dev/null || { echo "no $branch in $repo" >&2; exit 1; }
aot=zig/tests/cuda/glm_p1/aot
git -C "$repo" diff --name-status --no-renames "$base" "$branch" > "$tmp/status"
excluded='^(notes-glm/|tools/|HANDOFF-ZIG[.]md$|REPORT-P5[.]md$|'"$aot"'/)'
awk -v re="$excluded" '$2 !~ re { print $1 "\t" $2 }' "$tmp/status" > "$tmp/files"
group() {  # <number> <name> <awk condition on $1 (status) and $2 (path)>
  awk -F'\t' "$3 { print \$2 }" "$tmp/files" | LC_ALL=C sort > "$tmp/g$1"
  cat "$tmp/g$1" >> "$tmp/all"
}
: > "$tmp/all"
group 1 shared  '$1 == "M"'
group 2 flashnext '$1 == "A" && ($2 ~ /^zig\/src\/families\/flashnext\// || $2 ~ /^zig\/src\/cli\/flashnext_cuda[.]zig$/ || $2 ~ /^zig\/kernels\/cuda\// && $2 !~ /\/glm_/ || $2 ~ /^zig\/tests\/cuda\/flashnext/ )'
group 3 glmk '$1 == "A" && $2 ~ /^zig\/kernels\/cuda\/glm_/'
group 4 glm '$1 == "A" && ($2 ~ /^zig\/src\/families\/glm_cuda\// || $2 ~ /^zig\/src\/cli\/glm_cuda[.]zig$/ || $2 ~ /^zig\/src\/native\/glm_host[.]zig$/)'
LC_ALL=C sort "$tmp/all" > "$tmp/all.sorted"
awk -F'\t' '$1 == "A" { print $2 }' "$tmp/files" | LC_ALL=C sort | LC_ALL=C comm -23 - "$tmp/all.sorted" > "$tmp/g5"
stray=$(awk -F'\t' '$1 != "A" && $1 != "M"' "$tmp/files"); [[ -z "$stray" ]] || { echo "unexpected changes (deleted or renamed files): $stray" >&2; exit 1; }
rm -f "$out"/patches/*.patch
names=(shared-engine flashnext-cuda glm-kernels glm-engine glm-tests)
for n in 1 2 3 4 5; do
  f=$(printf '%s/patches/%04d-zig-%s.patch' "$out" "$n" "${names[n-1]}")
  [[ -s "$tmp/g$n" ]] || { echo "group $n is empty" >&2; exit 1; }
  { printf 'TensorFold %s..%s (%s): %s\nApply from the checkout root with: git apply %s\n\n' "$base" "$(git -C "$repo" rev-parse --short "$branch")" "$branch" "${names[n-1]}" "$(basename "$f")"
    mapfile -t paths < "$tmp/g$n"
    git -C "$repo" diff --binary --no-renames --src-prefix=a/ --dst-prefix=b/ "$base" "$branch" -- "${paths[@]}"; } > "$f"
  echo "$f ($(wc -l < "$tmp/g$n") files)"
done
# the captured kernel set: a deterministic tarball (sorted names, fixed owner and time)
mkdir -p "$out/aot"
git -C "$repo" archive "$branch" "$aot" | tar -x -C "$tmp"
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-10-11 00:00:00' -C "$tmp/$aot" -cf - . | xz -9e -T1 > "$out/aot/glm-aot-sm121.tar.xz"
sha256sum "$out/aot/glm-aot-sm121.tar.xz" | tee "$out/aot/glm-aot-sm121.tar.xz.sha256"
printf '%s %s %s\n' "$(git -C "$repo" rev-parse "$base")" "$(git -C "$repo" rev-parse "$branch")" "$branch" > "$out/PORT_REF"
# check: base + patches == the branch's files (minus the excluded ones)
mkdir "$tmp/tree"; git -C "$repo" archive "$base" | tar -x -C "$tmp/tree"
(cd "$tmp/tree" && for p in "$out"/patches/*.patch; do git apply --whitespace=nowarn "$p" || { echo "FAILED: $p" >&2; exit 1; }; done)
mkdir "$tmp/want"; git -C "$repo" archive "$branch" | tar -x -C "$tmp/want"
bad=$(cd "$tmp/want" && while IFS=$'\t' read -r _ p; do cmp -s "$p" "$tmp/tree/$p" || echo "$p"; done < "$tmp/files")
[[ -z "$bad" ]] || { echo "differs after the patches: $bad" >&2; exit 1; }
echo "OK: zig/patches/*.patch on $base give the branch's files exactly ($(wc -l < "$tmp/files") files)"
