#!/usr/bin/env bash
# Prepare both Sparks to serve Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (with ABLIT=1, the gated
# Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit, which needs HF_TOKEN) with TensorFold (two ranks; with TP=3, both
# workers the same way, each per its own WORKER_WEIGHTS / WORKER_WEIGHTS2):
#   1. preflight checks: docker and the GPU on both nodes, key-based ssh to the worker, the RoCE link, disk space
#   2. the image on the head: TensorFold plus patches/*.patch on NVIDIA's PyTorch container, pulled prebuilt from
#      $GHCR_IMAGE when a matching tag is reachable (PULL=0 skips that), else built locally
#   3. the same image on the worker: pulled there, else streamed from the head (docker save | ssh docker load)
#   4. download the checkpoint on the head into the Hugging Face cache (~164 GiB, resumable), and DFlash2 too when
#      DRAFTER=dflash2, at their pinned revisions (MODEL_REVISION, DFLASH2_REVISION)
#   5. verify the checkpoint with `tensorfold info`
#   6. the same files on the worker, copied from the head over the Sparks' link (rsync), checked file by file; or, with
#      WORKER_WEIGHTS=nfs, a read-only NFS volume of the head's cache on the worker, checked the same way
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
need_hf_token                        # ABLIT=1 (gated weights) without HF_TOKEN: say so before anything else

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$KERNEL_CACHE" "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running (it holds the download locks); wait for it or stop it: pgrep -af prepare.sh"
SPARKS="both Sparks"; (( TP == 2 )) || SPARKS="all $TP Sparks"
log "Preflight checks on $SPARKS"
# The gated Ablit weights (ABLIT=1): HF_TOKEN must reach their files, which takes the token's account having accepted
# the terms on the model's page. Asked for one small file before the image work, so a refusal says why at once instead
# of failing the download later. The header goes to curl on stdin, keeping the token out of the process list. No
# answer from the Hub (offline): left to the download, which falls back on a snapshot already here.
if [[ "$MODEL_ID" == "$ABLIT_ID" ]]; then
  _url="https://huggingface.co/$MODEL_ID/resolve/${MODEL_REVISION:-main}/config.json"
  code=$(printf 'Authorization: Bearer %s\n' "$HF_TOKEN" |
    curl -s -o /dev/null -w '%{http_code}' --max-time 20 -H @- "$_url" || true)
  case "$code" in
    2??|3??) log "HF_TOKEN reaches the gated $MODEL_ID" ;;
    401) die "Hugging Face does not accept HF_TOKEN (401): $MODEL_ID is gated; create a token with read access at
    https://huggingface.co/settings/tokens and set HF_TOKEN=hf_... in scripts/local.sh, .env or the environment" ;;
    403) die "HF_TOKEN may not read $MODEL_ID yet (403): it is gated; open https://huggingface.co/$MODEL_ID, log in
    with the token's account and agree to its terms (a fine-grained token also needs read access to public gated
    repositories), then run this again" ;;
    404) die "Hugging Face has no $MODEL_ID @ ${MODEL_REVISION:-main} (404): check MODEL_REVISION" ;;
    000) warn "could not reach Hugging Face to check HF_TOKEN against the gated $MODEL_ID; trying the download" ;;
    *) warn "Hugging Face answered HTTP $code when checking HF_TOKEN against the gated $MODEL_ID; trying the download" ;;
  esac
fi
command -v docker >/dev/null || die "docker is not installed"
command -v rsync >/dev/null || die "rsync is not installed on this node (sudo apt install rsync)"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
if ! command -v nvidia-smi >/dev/null; then warn "nvidia-smi not found on this node"
elif ! nvidia-smi -L >/dev/null 2>&1; then warn "nvidia-smi failed on this node: is the NVIDIA driver working?"; fi
docker info 2>/dev/null | grep -qi nvidia || warn "docker does not list an nvidia runtime on this node; --gpus all may fail"
check_workers
# "the worker" at TP=2, "worker 2 (user@host)" beyond
wname() { if (( TP == 2 )); then echo "the worker"; else echo "worker $1 ($(worker_host "$1"))"; fi; }
declare -a WORKER_HF=()
for i in $(worker_ids); do
  need_worker "$i"
  w=$(wname "$i")
  worker "$i" 'docker info >/dev/null 2>&1' || die "$w ($(worker_host "$i")) cannot talk to its docker daemon (docker group?)"
  worker "$i" 'nvidia-smi -L >/dev/null 2>&1' || warn "nvidia-smi failed on $w: is the NVIDIA driver working?"
  worker "$i" 'docker info 2>/dev/null | grep -qi nvidia' || warn "docker does not list an nvidia runtime on $w; --gpus all may fail"
  [[ "$(worker_weights "$i")" == nfs ]] || worker "$i" 'command -v rsync >/dev/null' ||
    die "rsync is not installed on $w (sudo apt install rsync)"
done
detect_links
if (( TP == 2 )); then
  log "Link: head $HEAD_ADDR ($HEAD_DEV, $HEAD_HCA, GID $HEAD_GID) <-> worker $WORKER_ADDR ($WORKER_DEV, $WORKER_HCA, GID $WORKER_GID)"
else
  for i in $(worker_ids); do
    log "Link to worker $i: head ${LINK_HEAD_ADDR[i]:-?} <-> ${LINK_WORKER_ADDR[i]:-?}; RoCE ${NODE_HCAS[i]}"
  done
fi
for i in $(worker_ids); do
  WORKER_HF[i]=$(worker_hf_cache "$i")
  [[ "$(worker_weights "$i")" == nfs ]] || worker "$i" "mkdir -p '${WORKER_HF[i]}/hub' && test -w '${WORKER_HF[i]}/hub'" ||
    die "$(wname "$i")'s ${WORKER_HF[i]}/hub is not writable (left root-owned by a container? fix its ownership there)"
done

PATCHES_HASH=$(image_hash)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
worker_free_gb() { worker "$1" "df -BG --output=avail '$2' | tail -1 | tr -dc '0-9'"; }
models=("$MODEL_ID"); [[ "$DRAFTER" == dflash2 ]] && models+=("$DFLASH2_ID")
# hub_missing_gb: GB (GiB, as df counts) the download still needs on the head: the files of each revision to serve
# whose blob is not in the cache yet (by name, the LFS sha256 or else the git blob id, and by size), from the Hub's
# file list. A new pin that shares its blobs with a cached revision needs next to nothing (issue #24). Asked with
# huggingface_hub from this host's python3 or the hf CLI's own; fails without either, or without the Hub.
HUB_MISSING_PY='
import math, os, sys
from huggingface_hub import HfApi
hub, args, missing = sys.argv[1], sys.argv[2:], 0
for repo, rev in zip(args[::2], args[1::2]):
    blobs = os.path.join(hub, "models--" + repo.replace("/", "--"), "blobs")
    for f in HfApi().model_info(repo, revision=rev or None, files_metadata=True).siblings:
        lfs = f.lfs
        name = (getattr(lfs, "sha256", None) or lfs["sha256"]) if lfs else f.blob_id
        size = (getattr(lfs, "size", None) or lfs["size"]) if lfs else (f.size or 0)
        p = os.path.join(blobs, name)
        missing += 0 if os.path.isfile(p) and os.path.getsize(p) == size else size
print(math.ceil(missing / 2**30))'
hub_missing_gb() {
  local py id args=() out tried=""
  for id in "${models[@]}"; do args+=("$id" "$(model_revision "$id")"); done
  for py in "$(command -v python3)" "$(sed -n '1s/^#! *\(\/[^ ]*python[0-9.]*\)$/\1/p' "$(command -v hf || echo /dev/null)" 2>/dev/null)"; do
    [[ -n "$py" && -x "$py" && "$py" != "${tried:-}" ]] || continue
    tried=$py
    out=$(timeout 30 "$py" -c "$HUB_MISSING_PY" "$HF_CACHE/hub" "${args[@]}" 2>/dev/null) && [[ "$out" =~ ^[0-9]+$ ]] &&
      { echo "$out"; return 0; }
  done
  return 1
}
# Disk on the head: what the download still needs (plus 5 GB), and the image (unless built from these patches), both
# on one filesystem when Docker's root shares it with HF_CACHE. Without the Hub's file list: MIN_FREE_GB, unless every
# snapshot to serve is already here (as before).
DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
if missing=$(hub_missing_gb); then
  need_ckpt=0; (( missing == 0 )) || need_ckpt=$((missing + 5))
  ckpt_what="the download needs ~${missing} GB that the cache does not hold yet"
else
  need_ckpt=0
  for id in "${models[@]}"; do [[ -d "$(model_cache_dir "$id")/snapshots/$(snapshot_rev "$id")" ]] || need_ckpt=$MIN_FREE_GB; done
  ckpt_what="the checkpoint needs MIN_FREE_GB (the Hub's file list could not be read to count what is missing)"
fi
(( need_ckpt )) || ckpt_what="nothing to download"
need_img=0; [[ $REBUILD -eq 0 && "$built_hash" == "$PATCHES_HASH" ]] || need_img=$IMAGE_FREE_GB
if [[ "$(stat -c %d "$HF_CACHE")" == "$(stat -c %d "$DOCKER_ROOT" 2>/dev/null)" ]]; then
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt + need_img )) ||
    die "only ${have} GB free under $HF_CACHE (also Docker's root), ~$((need_ckpt + need_img)) GB needed: $ckpt_what; the image needs ${need_img} GB (IMAGE_FREE_GB)"
else
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt )) ||
    die "only ${have} GB free under $HF_CACHE, ~${need_ckpt} GB needed: $ckpt_what"
  have=$(free_gb "$DOCKER_ROOT"); (( have >= need_img )) ||
    die "only ${have} GB free under Docker's root ($DOCKER_ROOT); the image needs ~${need_img} GB (IMAGE_FREE_GB)"
fi
declare -a WORKER_DOCKER_ROOT=()
disk=""
for i in $(worker_ids); do
  WORKER_DOCKER_ROOT[i]=$(worker "$i" "docker info -f '{{.DockerRootDir}}'" 2>/dev/null || echo /var/lib/docker)
  if [[ "$(worker_weights "$i")" == nfs ]]; then d=${WORKER_DOCKER_ROOT[i]}; else d=${WORKER_HF[i]}; fi
  if (( TP == 2 )); then disk+=", $(worker_free_gb "$i" "$d") GB under $d on the worker"
  else disk+=", $(worker_free_gb "$i" "$d") GB under $d on worker $i"; fi
done
log "Disk: $(free_gb "$HF_CACHE") GB free under $HF_CACHE here$disk"

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

# ---------------------------------------------------------------- 3. image (workers)
image_id=$(image_ident "$IMAGE")                    # by content: .Id differs between image stores (issue #8)
for i in $(worker_ids); do
  w=$(wname "$i")
  [[ "$(worker_image_ident "$i" "$IMAGE")" != "$image_id" ]] || continue
  wfree=$(worker_free_gb "$i" "${WORKER_DOCKER_ROOT[i]}")
  (( wfree >= IMAGE_FREE_GB )) ||
    die "only ${wfree} GB free under $w's Docker root (${WORKER_DOCKER_ROOT[i]}); the image needs ~${IMAGE_FREE_GB} GB (IMAGE_FREE_GB)"
  if [[ "${PULL:-1}" == 1 ]] && worker "$i" docker pull "$prebuilt" >/dev/null 2>&1 &&
     [[ "$(worker_image_ident "$i" "$prebuilt")" == "$image_id" ]]; then
    worker "$i" docker tag "$prebuilt" "$IMAGE"
    log "Using $prebuilt as $IMAGE on $w"
  else
    log "Copying $IMAGE to $w (docker save | docker load; only missing layers are stored)"
    docker save "$IMAGE" | worker "$i" docker load >/dev/null
  fi
  [[ "$(worker_image_ident "$i" "$IMAGE")" == "$image_id" ]] || die "$w's $IMAGE differs from the head's"
done
log "Image $IMAGE identical on $SPARKS"

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

# ---------------------------------------------------------------- 6. the same files on the workers
# The manifests below ("<file> <size>" a line) are sorted in byte order on both sides (LC_ALL=C, also inside the
# commands sent over ssh, which carries no locale of ours): each host's own collation would order the same files
# differently (issue #21). On a mismatch, the first differing lines show what differs.
manifest_diff() {  # <head's manifest> <worker's manifest>
  warn "first differences (< head, > worker):"
  diff <(printf '%s\n' "$1") <(printf '%s\n' "$2") | head -20 >&2 || true
}
for i in $(worker_ids); do
  [[ "$(worker_weights "$i")" == nfs ]] || continue  # no copy: rank i reads the head's cache over NFS
  w=$(wname "$i"); server=$(nfs_server "$i")
  ensure_nfs_volume "$i"
  # the mount first: a refused export says so here, not as a missing file below
  if ! out=$(worker_nfs "$i" true 2>&1); then
    die "$w cannot mount the head's :$NFS_PATH from $server over NFS ($NFS_VOLUME): $(tail -1 <<<"$out")
    The head must export $NFS_PATH to that worker's address on its link (${LINK_WORKER_ADDR[i]:-its CX7 address}, or its subnet), read-only;
    or set $(wvar NFS_SERVER "$i") to a head address the export allows, or $(wvar WORKER_WEIGHTS "$i")=copy (README: Worker weights over NFS)"
  fi
  for id in "${models[@]}"; do
    dir=$(model_cache_dir "$id"); rev=$(snapshot_rev "$id")
    manifest=$(cd "$dir/snapshots/$rev" && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort)
    have=$(worker_nfs "$i" find -L "/hf/hub/${dir##*/}/snapshots/$rev" -type f -printf '%P %s\n' 2>/dev/null | LC_ALL=C sort || true)
    [[ "$have" == "$manifest" ]] || { manifest_diff "$manifest" "$have"
      die "$w does not see $id @ ${rev:0:8} over NFS ($NFS_VOLUME: :$NFS_PATH from $server); is HF_CACHE exported to it? (README: Worker weights over NFS)"; }
    if (( TP == 2 )); then log "Worker reads $id @ ${rev:0:8} from the head over NFS ($NFS_VOLUME)"
    else log "Worker $i reads $id @ ${rev:0:8} from the head over NFS ($NFS_VOLUME from $server)"; fi
  done
done
for i in $(worker_ids); do
for id in "${models[@]}"; do
  [[ "$(worker_weights "$i")" == nfs ]] && break
  w=$(wname "$i"); WHOST=$(worker_host "$i"); WHF=${WORKER_HF[i]}
  dir=$(model_cache_dir "$id"); rev=$(snapshot_rev "$id")
  # every file of the snapshot, with its size (links followed), as the head has it
  manifest=$(cd "$dir/snapshots/$rev" && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort)
  wdir="$WHF/hub/${dir##*/}"
  have=$(worker "$i" "cd '$wdir/snapshots/$rev' 2>/dev/null && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort" || true)
  if [[ "$have" == "$manifest" ]]; then
    if (( TP == 2 )); then log "Worker has $id @ ${rev:0:8}"; else log "Worker $i has $id @ ${rev:0:8}"; fi
    continue
  fi
  # the blobs this revision uses (the cache layout huggingface_hub keeps), and how much rsync must send: the blobs (and
  # any plain file in the snapshot) that the worker lacks or holds at another size (issue #24; an earlier revision
  # may have brought it the rest)
  (cd "$dir" && find "snapshots/$rev" -type l -printf '%l\n' | sed 's#^\(\.\./\)*##' | LC_ALL=C sort -u) > "$STATE_DIR/blobs"
  (cat "$STATE_DIR/blobs"; cd "$dir" && find "snapshots/$rev" -type f) > "$STATE_DIR/files"
  send=$(awk 'FILENAME == ARGV[1] { w[$2] = $1; next } w[$2] != $1 { s += $1 } END { printf "%d", (s + 2^30 - 1) / 2^30 }' \
           <(worker "$i" "cd '$wdir' 2>/dev/null && xargs -r stat -Lc '%s %n' 2>/dev/null; true" < "$STATE_DIR/files") \
           <(cd "$dir" && xargs -r stat -Lc '%s %n' < "$STATE_DIR/files"))
  need=0; (( send == 0 )) || need=$((send + 5))
  wfree=$(worker_free_gb "$i" "$WHF")
  (( wfree >= need )) ||
    die "only ${wfree} GB free under $WHF on $w; $id needs ~${need} GB there (~${send} GB to send; or set $(wvar WORKER_WEIGHTS "$i")=nfs)"
  log "Copying $id @ ${rev:0:8} to $w over the Sparks' link (~${send} GB to send, resumes)"
  worker "$i" "mkdir -p '$wdir/refs' '$wdir/snapshots'"
  # the blobs, then the snapshot's links and refs/main
  # -L: a blob may itself be a link into the cache root's blobs/ (huggingface_hub's xet backend, issue #15)
  rsync -a -L --partial --files-from="$STATE_DIR/blobs" "$dir/" "$WHOST:$wdir/" \
    -e "ssh -o BatchMode=yes" ${RSYNC_OPTS:-}
  rsync -a "$dir/snapshots/$rev" "$WHOST:$wdir/snapshots/" -e "ssh -o BatchMode=yes"
  # refs/main as the head has it (with a pin: only when the worker has none)
  if [[ -z "$(model_revision "$id")" ]]; then worker "$i" "printf %s '$rev' > '$wdir/refs/main'"
  else worker "$i" "test -f '$wdir/refs/main' || printf %s '$rev' > '$wdir/refs/main'"; fi
  have=$(worker "$i" "cd '$wdir/snapshots/$rev' && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort")
  [[ "$have" == "$manifest" ]] ||
    { manifest_diff "$manifest" "$have"; die "$w's copy of $id differs from the head's after the copy"; }
done
done

prepared_state > "$PREPARED_MARKER"
if (( TP == 2 )); then log "Done: both Sparks are ready. Start the server with ./start.sh (port $PORT)."
else log "Done: all $TP Sparks are ready. Start the server with ./start-tp$TP.sh (port $PORT)."; fi
log "The first start compiles CUDA kernels for GB10 (a few minutes); they are cached in $KERNEL_CACHE/<image hash> here and under ~/.cache/tensorfold-glm53 on $( (( TP == 2 )) && echo "the worker" || echo "each worker") (a folder per image)."
