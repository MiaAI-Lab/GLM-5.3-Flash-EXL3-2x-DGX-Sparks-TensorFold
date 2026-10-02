#!/usr/bin/env bash
# Prepare both Sparks to serve Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw with TensorFold (two ranks):
#   1. preflight checks: docker and the GPU on both nodes, key-based ssh to the worker, the RoCE link, disk space
#   2. the image on the head: TensorFold plus patches/*.patch on NVIDIA's PyTorch container, pulled prebuilt from
#      $GHCR_IMAGE when a matching tag is reachable (PULL=0 skips that), else built locally
#   3. the same image on the worker: pulled there, else streamed from the head (docker save | ssh docker load)
#   4. download the checkpoint on the head into the Hugging Face cache (~164 GiB, resumable), and DFlash2 too when
#      DRAFTER=dflash2, at their pinned revisions (MODEL_REVISION, DFLASH2_REVISION)
#   5. verify the checkpoint with `tensorfold info`
#   6. the same files on the worker, copied from the head over the Sparks' link (rsync), checked file by file
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."     # the repository root
source ./scripts/config.sh
source ./scripts/nodes.sh

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$KERNEL_CACHE" "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running (it holds the download locks); wait for it or stop it: pgrep -af prepare.sh"
log "Preflight checks on both Sparks"
command -v docker >/dev/null || die "docker is not installed"
command -v rsync >/dev/null || die "rsync is not installed on this node (sudo apt install rsync)"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
if ! command -v nvidia-smi >/dev/null; then warn "nvidia-smi not found on this node"
elif ! nvidia-smi -L >/dev/null 2>&1; then warn "nvidia-smi failed on this node: is the NVIDIA driver working?"; fi
docker info 2>/dev/null | grep -qi nvidia || warn "docker does not list an nvidia runtime on this node; --gpus all may fail"
need_worker
worker 'docker info >/dev/null 2>&1' || die "the worker ($WORKER) cannot talk to its docker daemon (docker group?)"
worker 'nvidia-smi -L >/dev/null 2>&1' || warn "nvidia-smi failed on the worker: is the NVIDIA driver working?"
worker 'docker info 2>/dev/null | grep -qi nvidia' || warn "docker does not list an nvidia runtime on the worker; --gpus all may fail"
worker 'command -v rsync >/dev/null' || die "rsync is not installed on the worker (sudo apt install rsync)"
detect_link
log "Link: head $HEAD_ADDR ($HEAD_DEV, $HEAD_HCA, GID $HEAD_GID) <-> worker $WORKER_ADDR ($WORKER_DEV, $WORKER_HCA, GID $WORKER_GID)"
WORKER_HF=$(worker_hf_cache)
[[ "$WORKER_WEIGHTS" == nfs ]] || worker "mkdir -p '$WORKER_HF/hub' && test -w '$WORKER_HF/hub'" ||
  die "the worker's $WORKER_HF/hub is not writable (left root-owned by a container? fix its ownership there)"

PATCHES_HASH=$(image_hash)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
worker_free_gb() { worker "df -BG --output=avail '$1' | tail -1 | tr -dc '0-9'"; }
# Disk on the head: the checkpoint download (unless its snapshot is here) and the image (unless built from these
# patches), both on one filesystem when Docker's root shares it with HF_CACHE
DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
need_ckpt=0; [[ -d "$(model_cache_dir)/snapshots/$(snapshot_rev "$MODEL_ID")" ]] || need_ckpt=$MIN_FREE_GB
need_img=0; [[ $REBUILD -eq 0 && "$built_hash" == "$PATCHES_HASH" ]] || need_img=$IMAGE_FREE_GB
if [[ "$(stat -c %d "$HF_CACHE")" == "$(stat -c %d "$DOCKER_ROOT" 2>/dev/null)" ]]; then
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt + need_img )) ||
    die "only ${have} GB free under $HF_CACHE (also Docker's root); the checkpoint and the image need ~$((need_ckpt + need_img)) GB (MIN_FREE_GB, IMAGE_FREE_GB)"
else
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt )) ||
    die "only ${have} GB free under $HF_CACHE; the checkpoint needs ~${need_ckpt} GB (MIN_FREE_GB)"
  have=$(free_gb "$DOCKER_ROOT"); (( have >= need_img )) ||
    die "only ${have} GB free under Docker's root ($DOCKER_ROOT); the image needs ~${need_img} GB (IMAGE_FREE_GB)"
fi
WORKER_DOCKER_ROOT=$(worker "docker info -f '{{.DockerRootDir}}'" 2>/dev/null || echo /var/lib/docker)
log "Disk: $(free_gb "$HF_CACHE") GB free under $HF_CACHE here, $(worker_free_gb "$WORKER_HF") GB under $WORKER_HF on the worker"

# ---------------------------------------------------------------- 2. image (head)
# Local fixes in ./patches (unified diffs against site-packages, applied with patch -p0) are baked into the image.
# The image is rebuilt when they (or IMAGE_EXTRAS) change; the TensorFold install layer stays cached, so that takes seconds.
prebuilt=$(prebuilt_image)            # the pinned digest (config.sh's IMAGE_TAG / IMAGE_DIGEST), else the hash's tag
if [[ $REBUILD -eq 0 && "${PULL:-1}" == 1 && "$built_hash" != "$PATCHES_HASH" ]]; then
  log "Pulling the prebuilt image $prebuilt (PULL=0 builds instead)"
  if docker pull "$prebuilt" &&
     [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$prebuilt")" == "$PATCHES_HASH" ]]; then
    docker tag "$prebuilt" "$IMAGE"; built_hash=$PATCHES_HASH
    log "Using $prebuilt as $IMAGE"
  else
    warn "could not pull $prebuilt (no image for these patches, the package is not public, or no network): building it locally"
  fi
fi
if [[ $REBUILD -eq 1 || "$built_hash" != "$PATCHES_HASH" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 && [[ $REBUILD -eq 0 ]] || { log "Pulling base image $BASE_IMAGE"; docker pull "$BASE_IMAGE"; }
  log "Building $IMAGE (TensorFold $TF_VERSION, patches $PATCHES_HASH: $(compgen -G 'patches/*.patch' | wc -l) patches, plus $IMAGE_EXTRAS)"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  docker build "${nocache[@]}" -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_SPEC="git+${TF_REPO}@${TF_VERSION}" --build-arg PATCHES_HASH="$PATCHES_HASH" --build-arg EXTRAS="$IMAGE_EXTRAS" \
    -f - patches <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG TF_SPEC
ARG EXTRAS
RUN pip install --no-cache-dir --upgrade "${TF_SPEC}" && pip install --no-cache-dir ${EXTRAS} && tensorfold --version
COPY . /opt/tf-patches
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done && \
    python -c "import tensorfold.cuda.server, tensorfold.families.glm5_next.cuda.engine, tensorfold.vision.glm, av, xgrammar"
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH}
ENV HF_HOME=/root/.cache/huggingface TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace
DOCKERFILE
else
  log "Image $IMAGE already built with patches $PATCHES_HASH (use --rebuild to force)"
fi
docker run --rm --entrypoint tensorfold "$IMAGE" --version 2>/dev/null | tail -1

# ---------------------------------------------------------------- 3. image (worker)
image_id=$(image_ident "$IMAGE")                    # by content: .Id differs between image stores (issue #8)
if [[ "$(worker_image_ident "$IMAGE")" != "$image_id" ]]; then
  wfree=$(worker_free_gb "$WORKER_DOCKER_ROOT")
  (( wfree >= IMAGE_FREE_GB )) ||
    die "only ${wfree} GB free under the worker's Docker root ($WORKER_DOCKER_ROOT); the image needs ~${IMAGE_FREE_GB} GB (IMAGE_FREE_GB)"
  if [[ "${PULL:-1}" == 1 ]] && worker docker pull "$prebuilt" >/dev/null 2>&1 &&
     [[ "$(worker_image_ident "$prebuilt")" == "$image_id" ]]; then
    worker docker tag "$prebuilt" "$IMAGE"
    log "Using $prebuilt as $IMAGE on the worker"
  else
    log "Copying $IMAGE to the worker (docker save | docker load; only missing layers are stored)"
    docker save "$IMAGE" | worker docker load >/dev/null
  fi
  [[ "$(worker_image_ident "$IMAGE")" == "$image_id" ]] || die "the worker's $IMAGE differs from the head's"
fi
log "Image $IMAGE identical on both Sparks"

# ---------------------------------------------------------------- 4. download (head)
# One revision on both ranks: MODEL_REVISION / DFLASH2_REVISION (config.sh; empty: what the Hub calls main when first
# downloaded, then kept).
command -v hf >/dev/null || warn "host 'hf' CLI not found, downloading from inside the container"
download() {  # <repo id> <revision or empty>
  if command -v hf >/dev/null; then
    # Host CLI: resumable, parallel, writes the standard HF cache layout.
    hf download "$1" ${2:+--revision "$2"} --cache-dir "$HF_CACHE/hub" >/dev/null
  else
    # Keep downloads owned by the host user, in the same HF_CACHE/hub layout as the host CLI.
    docker run --rm --user "$(id -u):$(id -g)" --network host --entrypoint python ${HF_TOKEN:+-e HF_TOKEN} \
      -v "$HF_CACHE":/hf -e HF_HOME=/hf -e HOME=/tmp "$IMAGE" -c \
      'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None)' "$1" "$2"
  fi
}
models=("$MODEL_ID"); [[ "$DRAFTER" == dflash2 ]] && models+=("$DFLASH2_ID")
for id in "${models[@]}"; do
  pin=$(model_revision "$id"); dir=$(model_cache_dir "$id")
  log "Downloading $id${pin:+ @ ${pin:0:8}} into $HF_CACHE/hub (resumes if interrupted)"
  if ! download "$id" "$pin"; then
    # no network: a pinned snapshot already here is enough (serving reads the local cache only)
    [[ -n "$pin" && -f "$dir/snapshots/$pin/config.json" ]] || die "$id: the download failed"
    warn "$id: could not reach Hugging Face; using the snapshot already here (${pin:0:8})"
  fi
  # a download by commit sha writes no refs/main: name the pin there when nothing else is (tools that take repo ids)
  [[ -z "$pin" || -f "$dir/refs/main" ]] || { mkdir -p "$dir/refs"; printf %s "$pin" > "$dir/refs/main"; }
  rev=$(snapshot_rev "$id")
  [[ -n "$rev" && -d "$dir/snapshots/$rev" ]] || die "$id: no snapshot${rev:+ $rev} after the download"
  log "Checkpoint: $dir/snapshots/$rev ($(du -shL "$dir/snapshots/$rev" | cut -f1))"
done

# ---------------------------------------------------------------- 5. verify (head)
# before the copy: the worker gets only a checkpoint TensorFold reads
log "Verifying checkpoint with tensorfold info"
# (without its "EXL3 support is experimental" note: this recipe serves the EXL3 checkpoint on purpose)
info=$(docker run --rm --entrypoint tensorfold -e HF_HUB_OFFLINE=1 -v "$HF_CACHE":/root/.cache/huggingface "$IMAGE" \
  info "/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev "$MODEL_ID")" 2>&1) ||
  die "tensorfold info cannot read the checkpoint: $info"
printf '%s\n' "$info" | grep -v "EXL3 support is experimental" || true

# ---------------------------------------------------------------- 6. the same files on the worker
if [[ "$WORKER_WEIGHTS" == nfs ]]; then         # no copy: rank 1 reads the head's cache over NFS
  ensure_nfs_volume
  for id in "${models[@]}"; do
    dir=$(model_cache_dir "$id"); rev=$(snapshot_rev "$id")
    manifest=$(cd "$dir/snapshots/$rev" && find -L . -type f -printf '%P %s\n' | sort)
    have=$(worker_nfs find -L "/hf/hub/${dir##*/}/snapshots/$rev" -type f -printf '%P %s\n' 2>/dev/null | sort || true)
    [[ "$have" == "$manifest" ]] ||
      die "the worker does not see $id @ ${rev:0:8} over NFS ($NFS_VOLUME: :$NFS_PATH from ${NFS_SERVER:-$HEAD_ADDR}); is HF_CACHE exported to it? (README: Worker weights over NFS)"
    log "Worker reads $id @ ${rev:0:8} from the head over NFS ($NFS_VOLUME)"
  done
fi
for id in "${models[@]}"; do
  [[ "$WORKER_WEIGHTS" == nfs ]] && break
  dir=$(model_cache_dir "$id"); rev=$(snapshot_rev "$id")
  # every file of the snapshot, with its size (links followed), as the head has it
  manifest=$(cd "$dir/snapshots/$rev" && find -L . -type f -printf '%P %s\n' | sort)
  wdir="$WORKER_HF/hub/${dir##*/}"
  have=$(worker "cd '$wdir/snapshots/$rev' 2>/dev/null && find -L . -type f -printf '%P %s\n' | sort" || true)
  if [[ "$have" == "$manifest" ]]; then
    log "Worker has $id @ ${rev:0:8}"
    continue
  fi
  need=$(( $(du -sLB1G "$dir/snapshots/$rev" | cut -f1) + 5 ))
  wfree=$(worker_free_gb "$WORKER_HF")
  (( wfree >= need )) || die "only ${wfree} GB free under $WORKER_HF on the worker, $id needs ~${need} GB"
  log "Copying $id @ ${rev:0:8} to the worker over the Sparks' link (~${need} GB, resumes)"
  worker "mkdir -p '$wdir/refs' '$wdir/snapshots'"
  # the blobs this revision uses, then its snapshot links and refs/main (the cache layout huggingface_hub keeps)
  (cd "$dir" && find "snapshots/$rev" -type l -printf '%l\n' | sed 's#^\(\.\./\)*##' | sort -u) > "$STATE_DIR/blobs"
  # -L: a blob may itself be a link into the cache root's blobs/ (huggingface_hub's xet backend, issue #15)
  rsync -a -L --partial --files-from="$STATE_DIR/blobs" "$dir/" "$WORKER:$wdir/" \
    -e "ssh -o BatchMode=yes" ${RSYNC_OPTS:-}
  rsync -a "$dir/snapshots/$rev" "$WORKER:$wdir/snapshots/" -e "ssh -o BatchMode=yes"
  # refs/main as the head has it (with a pin: only when the worker has none)
  if [[ -z "$(model_revision "$id")" ]]; then worker "printf %s '$rev' > '$wdir/refs/main'"
  else worker "test -f '$wdir/refs/main' || printf %s '$rev' > '$wdir/refs/main'"; fi
  have=$(worker "cd '$wdir/snapshots/$rev' && find -L . -type f -printf '%P %s\n' | sort")
  [[ "$have" == "$manifest" ]] || die "the worker's copy of $id differs from the head's after the copy"
done

prepared_state > "$PREPARED_MARKER"
log "Done: both Sparks are ready. Start the server with ./start.sh (port $PORT)."
log "The first start compiles CUDA kernels for GB10 (a few minutes); they are cached in $KERNEL_CACHE/<image hash> here and under ~/.cache/tensorfold-glm53 on the worker (a folder per image)."
