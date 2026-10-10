#!/usr/bin/env bash
# The CPU tests (tests/cpu): the patched engine's new code paths on CPU tensors, Triton kernels in Triton's
# interpreter, several ranks as processes over gloo (two, three and four). No GPU; a memory-capped container of the
# image (MEM, default 3g; CPUS, default 6; pytest is pip-installed into it).
#   scripts/test-cpu.sh [pytest args]
#   TF_SRC=<patched TensorFold src/> scripts/test-cpu.sh   tests that tree instead of the image's
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing: run scripts/prepare.sh first"
args=(--rm --memory "${MEM:-3g}" --cpus "${CPUS:-6}" -v "$PWD/tests:/tests:ro" -w /tests
      -v "$PWD/scripts/config.sh:/recipe/scripts/config.sh:ro")
[[ -z "${TF_SRC:-}" ]] || args+=(-v "$(readlink -f "$TF_SRC"):/tf-src:ro" -e TF_SRC=/tf-src)
# pytest is not in the image: a throwaway layer adds it
exec docker run "${args[@]}" --entrypoint bash "$IMAGE" -c \
  'pip install -q --no-cache-dir pytest==8.3.5 >/dev/null 2>&1 || { echo "cannot install pytest (no network?)"; exit 2; }; exec python -m pytest -q -p no:cacheprovider cpu "$@"' _ "$@"
