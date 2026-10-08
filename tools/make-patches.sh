#!/usr/bin/env bash
# Maintainers: write patches/NNNN-*.patch from a TensorFold development branch whose commits are named after the
# patches (one commit a patch, on top of TensorFold v0.6.0), then check that patches/*.patch, applied in filename order
# with patch -p0 to a fresh v0.6.0 tree, give exactly the branch's src/.
#   tools/make-patches.sh <TensorFold checkout> [first patch number, default 0099]
# Each patch is `git diff --no-prefix --relative=src <commit>~1 <commit> -- src`; earlier patches are left as they are.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
tf=${1:?usage: tools/make-patches.sh <TensorFold checkout> [first patch number]}
first=${2:-0099}
base=$(git -C "$tf" rev-list -n1 v0.6.0)
for c in $(git -C "$tf" rev-list --reverse "$base"..HEAD); do
  name=$(git -C "$tf" log -1 --format=%s "$c")
  [[ "$name" =~ ^[0-9]{4}-[a-z0-9-]+$ ]] || { echo "commit $c is not named NNNN-name: $name" >&2; exit 1; }
  [[ "${name:0:4}" < "$first" ]] && continue
  git -C "$tf" diff --no-prefix --relative=src "$c~1" "$c" -- src > "patches/$name.patch"
  echo "patches/$name.patch"
done
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git -C "$tf" archive "$base" src | tar -x -C "$tmp"
(cd "$tmp/src" && for p in "$OLDPWD"/patches/*.patch; do patch -p0 -s -f < "$p" >/dev/null || { echo "FAILED: $p" >&2; exit 1; }; done)
git -C "$tf" archive HEAD src | tar -x -C "$tmp/want" --one-top-level 2>/dev/null || { mkdir -p "$tmp/want"; git -C "$tf" archive HEAD src | tar -x -C "$tmp/want"; }
find "$tmp/src" -name '*.orig' -delete
diff -r -q "$tmp/src" "$tmp/want/src" && echo "OK: patches/*.patch on v0.6.0 give the branch's src/ exactly"
