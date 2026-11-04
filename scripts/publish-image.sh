#!/usr/bin/env bash
# Push the image scripts/prepare.sh built to GitHub Container Registry as $GHCR_IMAGE:<TensorFold version>-<image hash>
# and :latest, labelled with this repository (so the package takes the repository's visibility and access).
# The image hash is config.sh's image_hash (patches/*.patch plus IMAGE_EXTRAS), the same value prepare.sh writes into
# the image's tf.patches label and looks for when it pulls: prepare.sh pulls $GHCR_IMAGE:<TensorFold version>-<hash>.
# Run it on the head Spark (the one that runs ./start.sh); the worker gets the image from the head or the registry.
# Needs a GitHub token with write:packages: `gh auth login -s write:packages`, or GHCR_TOKEN / GITHUB_TOKEN set.
# Usage: scripts/publish-image.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

REPO_URL="${REPO_URL:-https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold}"
ctr image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing, run scripts/prepare.sh first"
hash=$(ctr image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE")
[[ "$hash" == "$(image_hash)" ]] ||
  die "$IMAGE was built from other patches or extras ($hash, not $(image_hash)); run scripts/prepare.sh first"
tag="${TF_VERSION}-${hash}"

token="${GHCR_TOKEN:-${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}}"
[[ -n "$token" ]] || die "no GitHub token: run gh auth login -s write:packages, or set GHCR_TOKEN"
user="${GHCR_USER:-$(gh api user --jq .login 2>/dev/null || echo "${GHCR_IMAGE#ghcr.io/}" | cut -d/ -f1)}"

log "Labelling $IMAGE as $GHCR_IMAGE:$tag"
ctr build $(rt_build_net) -q -t "$GHCR_IMAGE:$tag" -t "$GHCR_IMAGE:latest" \
  --label org.opencontainers.image.source="$REPO_URL" \
  --label org.opencontainers.image.licenses=Apache-2.0 \
  --label org.opencontainers.image.description="GLM-5.3-Flash EXL3 on two DGX Sparks: TensorFold $TF_VERSION with patches $hash" \
  - <<<"FROM $IMAGE" >/dev/null
# the labels above sit on top of the built image: its tf.patches label must come through unchanged
[[ "$(ctr image inspect -f '{{index .Config.Labels "tf.patches"}}' "$GHCR_IMAGE:$tag")" == "$hash" ]] ||
  die "$GHCR_IMAGE:$tag lost its tf.patches label"
log "Pushing $GHCR_IMAGE:$tag and :latest (~25 GB uncompressed; only changed layers upload)"
# log in with a throwaway credential store, so no registry credential stays behind after the push: docker reads
# DOCKER_CONFIG, podman --authfile (podman's DOCKER_CONFIG equivalent only exists for its docker-compatible mode)
login_dir=$(mktemp -d)
trap 'rm -rf -- "$login_dir"' EXIT
if [[ "$CONTAINER_RT" == podman ]]; then
  auth=(--authfile "$login_dir/auth.json")
  echo "$token" | ctr login ghcr.io -u "$user" --password-stdin "${auth[@]}" >/dev/null
  ctr push "${auth[@]}" "$GHCR_IMAGE:$tag"
  ctr push "${auth[@]}" "$GHCR_IMAGE:latest"
else
  echo "$token" | DOCKER_CONFIG="$login_dir" ctr login ghcr.io -u "$user" --password-stdin >/dev/null
  DOCKER_CONFIG="$login_dir" ctr push "$GHCR_IMAGE:$tag"
  DOCKER_CONFIG="$login_dir" ctr push "$GHCR_IMAGE:latest"
fi
digest=$(ctr image inspect -f '{{range .RepoDigests}}{{println .}}{{end}}' "$GHCR_IMAGE:$tag" | grep -m1 "^$GHCR_IMAGE@" |
         cut -d@ -f2)
log "Done: $CONTAINER_RT pull $GHCR_IMAGE:$tag"
log "Pin it in scripts/config.sh: IMAGE_TAG=$tag IMAGE_DIGEST=$digest"
