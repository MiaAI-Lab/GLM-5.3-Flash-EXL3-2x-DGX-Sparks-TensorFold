#!/usr/bin/env bash
# The GPU tests (tests/gpu) on this Spark's GPU: the new CUDA kernels on synthetic tensors at a rank's real shapes,
# against references and their own bits (any block of rows, repeated runs). Small (a few GiB); never while the server
# runs on this Spark: it refuses then.
#   scripts/test-gpu.sh [pytest args]
#   TF_SRC=<patched TensorFold src/> scripts/test-gpu.sh   tests that tree instead of the image's
#   TEST_GPU_GIB=4 scripts/test-gpu.sh                   caps the GPU memory the tests take (a Spark serving meanwhile)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing: run scripts/prepare.sh first"
[[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != true ]] ||
  die "$CONTAINER_NAME runs on this Spark: stop it first (./stop.sh)"
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE")
mkdir -p "$KERNEL_CACHE/$KCACHE"
args=(--rm --gpus all --ipc=host --memory "${MEM:-12g}" -v "$PWD/tests:/tests:ro" -w /tests
      -v "$KERNEL_CACHE/$KCACHE:/cache")
[[ -z "${TF_SRC:-}" ]] || args+=(-v "$(readlink -f "$TF_SRC"):/tf-src:ro" -e TF_SRC=/tf-src)
[[ -z "${TEST_GPU_GIB:-}" ]] || args+=(-e TEST_GPU_GIB="$TEST_GPU_GIB")
exec docker run "${args[@]}" --entrypoint bash "$IMAGE" -c \
  'pip install -q --no-cache-dir pytest==8.3.5 >/dev/null 2>&1 || { echo "cannot install pytest (no network?)"; exit 2; }; exec python -m pytest -q -p no:cacheprovider gpu "$@"' _ "$@"
