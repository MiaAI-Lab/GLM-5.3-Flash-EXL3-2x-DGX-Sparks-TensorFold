#!/usr/bin/env bash
# Serve GLM-5.3 Flash EXL3 (Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) with TensorFold on two DGX Sparks, end to end:
# runs scripts/prepare.sh on both Sparks when the image or the checkpoint is not ready yet (first run, or after patches
# change), starts rank 1 on the worker and rank 0 here, which serves the API on port 8888, waits until the OpenAI API
# answers, then runs a smoke test. Stop it with ./stop.sh. (3 Sparks: ./start-tp3.sh, experimental.)
#
# Usage: ./start.sh [restart] [extra tensorfold serve args]
#   ./start.sh                         # scripts/config.sh defaults: 4 requests at once, a 1,048,576-token window,
#                                      # FP8 KV cache, DFlash2 drafts, 4-bit dense weights, images and video
#                                      # (if the server already runs, says so and leaves it alone)
#   ./start.sh restart                 # stop both ranks (./stop.sh), then start them again, e.g. to apply changed
#                                      # settings or patches; the new arguments are checked before stopping
#   ./start.sh restart --parallel 2 --context 524288
#   CONTEXT=131072 ./start.sh restart
#   KV=bf16 ./start.sh restart         # the exact bf16 KV cache (and a 196,608 window)
#   DRAFTER=mtp ./start.sh restart     # the checkpoint's own MTP head instead of DFlash2 (one request at a time)
#   DRY_RUN=1 ./start.sh               # print every rank's docker command and exit, stopping and starting nothing
# Extra arguments go to both ranks after the defaults, so they win (the last value of a flag counts).
# Setup: WORKER=user@<worker address> in scripts/local.sh (key-based ssh).
# Settings, from the environment, scripts/local.sh or ./.env (defaults and measured effects in scripts/config.sh):
#   serving  CONTEXT, PARALLEL, KV, DENSE, DRAFTER, DRAFT_POLICY, COPY, COPY_MAX, COPY_CODE, SPLIT, KDA_CHUNKED,
#            SHARED_PREFIX, MULTI_PREFILL, STREAM_SMOOTH, STREAM_SMOOTH_MS, FILL_BUDGET_MS, FILL_DRAFTS, FILL_ROWS,
#            KV_POOL_GIB, MEMORY_RESERVE_GIB, MAX_TOKENS, THINKING, VISION, VISION_URLS, COMM, SERVED_NAME, HOST, PORT
#   nodes    WORKER, FABRIC_PEER, WORKER_HF_CACHE, MASTER_PORT, NCCL_RAILS (1: one RoCE device), NCCL_CHANNELS,
#            NCCL_DEBUG; TP (2), WORKER2, FABRIC_PEER2, WORKER_HF_CACHE2, MASTER_ADDR, SOCKET_IFNAME (3 Sparks)
#   files    ABLIT (1: the gated Ablit weights, needs HF_TOKEN), MODEL_ID, MODEL_REVISION, DFLASH2_ID, DFLASH2_REVISION,
#            HF_CACHE (default: HF_HOME), KERNEL_CACHE,
#            WORKER_WEIGHTS (copy | nfs: rank 1 reads the head's HF_CACHE over NFS), NFS_PATH, NFS_SERVER, NFS_VOLUME,
#            WORKER_WEIGHTS2, NFS_SERVER2,
#            STATE_DIR, HF_HUB_OFFLINE=0 (let TensorFold reach the Hub; default serves from the local cache only)
#   image    IMAGE, TF_VERSION, TF_REPO, BASE_IMAGE, GHCR_IMAGE, IMAGE_TAG / IMAGE_DIGEST (the pinned published
#            image), CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), PULL, MIN_FREE_GB, IMAGE_FREE_GB, RSYNC_OPTS, HF_TOKEN (prepare.sh's downloads;
#            required with ABLIT=1);
#            FOREGROUND=1 (stay attached to rank 0's log, exit with its code); WAIT_TIMEOUT (seconds, default 1800);
#            DRY_RUN=1 (print the docker commands, change nothing);
#            STOP_TIMEOUT (stop.sh); LOG_DIR, LOG_KEEP (saved server logs)
#   decode   TF_GLM_L2PF (1), TF_GLM_EXL3_LOADS (nc), TF_ROCE_MAX_KB (512), TF_GLM_MULTI_LONE (0),
#            TF_GLM_MULTI_WINDOW (32)
#   ranks    every TENSORFOLD_*, TF_GLM_* and TF_ROCE_* variable goes to every rank (TP>2: TF_ROCE_HCA per node, found),
#            e.g.
#            TENSORFOLD_GLM_IMAGE_TOKENS, TENSORFOLD_MEMORY_RESERVE_GIB, TF_GLM_KEEP_REASONING=0
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done
need_hf_token                        # ABLIT=1 (gated weights) without HF_TOKEN: say so before anything else

# The serve arguments both ranks share: scripts/config.sh's defaults first, then the command line's (argparse keeps
# the last value). --drafter goes in front after the setup step, which knows DFlash2's snapshot.
SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS")
[[ "$DRAFTER" =~ ^(mtp|dflash2)$ ]] || die "DRAFTER is mtp or dflash2, not $DRAFTER"
[[ "$DENSE" =~ ^(bf16|fp8|q4)$ ]] || die "DENSE is bf16, fp8 or q4, not $DENSE"
[[ "$COMM" =~ ^(nccl|roce)$ ]] || die "COMM is nccl or roce, not $COMM"
check_workers
[[ "$KV" =~ ^(bf16|fp8)$ ]] || die "KV is bf16 or fp8, not $KV"
[[ "$CONTEXT" =~ ^[0-9]+$ && "$CONTEXT" -le 1048576 ]] || die "CONTEXT is a token count up to 1048576 (0: the largest that fits), not $CONTEXT"
[[ "$PARALLEL" =~ ^[1-8]$ ]] || die "PARALLEL is 1 to 8, not $PARALLEL"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS is a token count, not $MAX_TOKENS"
[[ "$DRAFTER" == dflash2 || "$PARALLEL" == 1 ]] || die "PARALLEL=$PARALLEL needs DRAFTER=dflash2 (mtp serves one request at a time: PARALLEL=1)"
for v in SPLIT SHARED_PREFIX KDA_CHUNKED COPY_CODE MULTI_PREFILL STREAM_SMOOTH; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$DISPLAY_KV_MIB" =~ ^(0|[1-9][0-9]{0,3})$ ]] && (( DISPLAY_KV_MIB % 16 == 0 && DISPLAY_KV_MIB <= 2032 )) ||
  die "DISPLAY_KV_MIB is a multiple of 16 from 0 to 2032, not $DISPLAY_KV_MIB"
(( DISPLAY_KV_MIB == 0 || PARALLEL > 1 )) || die "DISPLAY_KV_MIB adds to the shared pool, which needs PARALLEL above 1"
[[ "$DISPLAY_KV_BACKEND" == drm || "$DISPLAY_KV_BACKEND" == dispram ]] ||
  die "DISPLAY_KV_BACKEND is drm or dispram, not $DISPLAY_KV_BACKEND"
if (( DISPLAY_KV_MIB )) && [[ "$DISPLAY_KV_BACKEND" == dispram ]]; then
  # kindling spark-os: dispramd owns the reservation (its nvidia_drm has no dumb buffers); every rank gets its socket
  # and python client (RUN_ARGS below), and the rank fails closed without them
  [[ -S /run/dispram/dispram.sock ]] || die "DISPLAY_KV_BACKEND=dispram needs dispramd (/run/dispram/dispram.sock), which this Spark lacks"
  [[ -d /opt/kindling/dispram/python ]] || die "DISPLAY_KV_BACKEND=dispram needs the dispram client (/opt/kindling/dispram/python)"
elif (( DISPLAY_KV_MIB )); then        # headless only: a monitor's framebuffer lives in the reservation
  [[ -e /dev/dri/card0 ]] || die "DISPLAY_KV_MIB needs /dev/dri/card0, which this Spark lacks"
  # no outputs under card0 at all: nvidia_drm runs without modeset, which has no dumb buffers for the span
  compgen -G '/sys/class/drm/card0-*/status' >/dev/null ||
    die "DISPLAY_KV_MIB needs nvidia_drm with modeset=1, and card0 shows no display outputs (an /etc/modprobe.d file may set modeset=0); set it to 0"
  _shown=$(grep -lx connected /sys/class/drm/card0-*/status 2>/dev/null | sed 's|.*/\(card0-[^/]*\)/status|\1|' | paste -sd, - || true)
  [[ -z "$_shown" ]] || die "DISPLAY_KV_MIB is for headless Sparks, and $_shown has a display connected; set it to 0"
fi
[[ "$WORKER_WEIGHTS" == copy || "$WORKER_WEIGHTS" == nfs ]] || die "WORKER_WEIGHTS is copy or nfs, not $WORKER_WEIGHTS"
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1
[[ "$KV_POOL_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "KV_POOL_GIB is a number of GiB, not $KV_POOL_GIB"
[[ "$MEMORY_RESERVE_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "MEMORY_RESERVE_GIB is a number of GiB, not $MEMORY_RESERVE_GIB"
[[ "$COPY_MAX" =~ ^([1-9]|1[0-5])$ ]] || die "COPY_MAX is 1 to 15, not $COPY_MAX"
[[ "$DRAFT_POLICY" == f* ]] || die "DRAFT_POLICY is a DFlash2 policy (fc5:0.3, fnc7:0.3, fcost7:noisy ...), not $DRAFT_POLICY"
# decode settings the ranks read at load (scripts/config.sh): checked here so a typo fails before anything is stopped
[[ "$TF_GLM_MULTI_LONE" =~ ^[01]$ ]] || die "TF_GLM_MULTI_LONE is 0 or 1, not $TF_GLM_MULTI_LONE"
[[ "$TF_GLM_MULTI_WINDOW" =~ ^(16|24|32|40|48|56|64)$ ]] || die "TF_GLM_MULTI_WINDOW is 16 to 64 rows in steps of 8, not $TF_GLM_MULTI_WINDOW"
[[ "$TF_GLM_CLEAR_THINKING" =~ ^[01]$ ]] || die "TF_GLM_CLEAR_THINKING is 0 or 1, not $TF_GLM_CLEAR_THINKING"
[[ "$TF_GLM_EFFORT_TAIL" =~ ^[01]$ ]] || die "TF_GLM_EFFORT_TAIL is 0 or 1, not $TF_GLM_EFFORT_TAIL"
[[ "$TF_GLM_L2PF" =~ ^(0|off|1|bulk|lines|touch)$ ]] || die "TF_GLM_L2PF is 0, 1 (bulk), lines or touch, not $TF_GLM_L2PF"
[[ "$TF_GLM_EXL3_LOADS" =~ ^(0|ldg|1|nc|nc1|nc2|nc4)$ ]] || die "TF_GLM_EXL3_LOADS is 0, nc, nc2 or nc4, not $TF_GLM_EXL3_LOADS"
[[ "$TF_GLM_EXL3_DEC_ORDER" =~ ^[012]$ ]] || die "TF_GLM_EXL3_DEC_ORDER is 0, 1 or 2, not $TF_GLM_EXL3_DEC_ORDER"
[[ "$TF_ROCE_MAX_KB" =~ ^[1-9][0-9]*$ ]] || die "TF_ROCE_MAX_KB is a size in KiB (512: up to 32-row windows over RoCE), not $TF_ROCE_MAX_KB"
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$VISION" == 1 ]] && SERVE_ARGS+=(--vision)
[[ "$VISION" == 1 && "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
if [[ "$SPILL_GIB" != 0 ]]; then                   # patch 0088: the spill tier (scripts/config.sh)
  [[ "$SPILL_DIR" == /* ]] || die "SPILL_DIR must be an absolute path (the same on every Spark), not $SPILL_DIR"
  SERVE_ARGS+=(--spill-gib "$SPILL_GIB" --snapshot-dir /spill --spill-highwater "$SPILL_HIGHWATER"
               --spill-min-tokens "$SPILL_MIN_TOKENS" --spill-min-free-gib "$SPILL_MIN_FREE_GIB")
fi
SERVE_ARGS+=("$@")
# The effective value of a flag (its last occurrence, as --flag value or --flag=value).
arg_value() {
  local flag=$1 value="" i
  for (( i = 0; i < ${#SERVE_ARGS[@]}; i++ )); do
    case "${SERVE_ARGS[i]}" in
      "$flag") value="${SERVE_ARGS[i + 1]:-}" ;;
      "$flag="*) value="${SERVE_ARGS[i]#*=}" ;;
    esac
  done
  echo "$value"
}
# Where to reach the server from this machine: a wildcard bind answers on loopback.
API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
# Words for the nodes: "both Sparks" / "the worker" at TP=2, "all 3 Sparks" / "rank 2 (user@host)" beyond.
SPARKS="both Sparks"; (( TP == 2 )) || SPARKS="all $TP Sparks"
wname() { if (( TP == 2 )); then echo "the worker"; else echo "rank $1 ($(worker_host "$1"))"; fi; }
served_name() {
  curl -s --max-time 5 "$URL/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

# ---------------------------------------------------------------- banner and progress
B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner                                             # the TensorFold ribbon and MIA AI LAB (terminals only)
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · %s x DGX Spark · %s at once · %s-token window · %s KV · %s drafts · port %s%s\n\n' "$D" "$MODEL_ID" "$TP" \
  "$(arg_value --parallel)" "$(arg_value --context)" "$KV" "$DRAFTER" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$KERNEL_CACHE" "$STATE_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_workers
(( DRY )) && log "DRY_RUN=1: printing the docker commands; nothing is stopped, started or prepared"

# ---------------------------------------------------------------- already running?
all_running() { local i; running_here || return 1; for i in $(worker_ids); do running_worker "$i" || return 1; done; }
if (( ! DRY )) && [[ "$MODE" == start ]] && all_running; then
  log "$CONTAINER_NAME is already running on $SPARKS (model: $(served_name || echo "not answering yet"), port $PORT): nothing to do."
  log "Use ./start.sh restart to restart it (e.g. with new settings), or ./stop.sh to stop it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
# scripts/prepare.sh (image and checkpoint on both Sparks) runs whenever what it last prepared differs from now: the
# first run, new patches, another model, drafter or worker. PREPARE=1 forces it, PREPARE=0 skips it.
step 1 "Setup: image and checkpoint on $SPARKS"
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now (not ready yet)"
  else
    log "Not ready yet: running scripts/prepare.sh (the first time this pulls the image, downloads ~166 GiB and copies both to the worker)"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE (patches $(image_hash)) and $MODEL_ID on $SPARKS${PREPARE:+ (PREPARE=$PREPARE)}"
fi
why="scripts/prepare.sh did not"; [[ "${PREPARE:-auto}" == 0 ]] && why="PREPARE=0 skipped scripts/prepare.sh, which would"
(( DRY )) && why="DRY_RUN: scripts/prepare.sh would"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing: $why build it"
for i in $(worker_ids); do
  [[ -z "${WORKER_DOWN[$i]:-}" ]] || continue
  worker "$i" docker image inspect "$IMAGE" >/dev/null 2>&1 || { (( DRY )) && warn "image $IMAGE missing on $(wname "$i"): $why copy it there"; } ||
    die "image $IMAGE missing on $(wname "$i"): $why copy it there"
done
# Compiled kernels are kept per image (its patches hash): a build is found by its extension's name, so another image's
# build of the same name, older or newer, must never be the one loaded
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
[[ "$KCACHE" =~ ^[0-9a-f]{12}$ ]] || KCACHE=$(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -d: -f2 | cut -c1-12)
# (docker creates the folder at the mount: the cache is the containers', so it may not be writable from here)
# The snapshots every rank serves (config.sh's pins, else refs/main), as paths under the containers' cache mount: the
# ranks read them offline, whatever the Hub's main is now. A worker's cache is its own HF_HOME (prepare.sh copies
# into the same place), or the head's over NFS (WORKER_WEIGHTS / WORKER_WEIGHTS<i>=nfs).
declare -a WORKER_HF=() WORKER_MOUNT=()
for i in $(worker_ids); do
  WORKER_HF[i]=$(worker_hf_cache "$i")
  WORKER_MOUNT[i]="${WORKER_HF[i]}:/root/.cache/huggingface"       # rank i's cache: its own copy, or the head's (nfs)
  [[ "$(worker_weights "$i")" == nfs ]] && WORKER_MOUNT[i]="$NFS_VOLUME:/root/.cache/huggingface:ro"
done
snapshot() {  # <repo id>: its snapshot path in the container, checked on every Spark (DRY_RUN: here only)
  local id=$1 rev sub i
  rev=$(snapshot_rev "$id")
  [[ -n "$rev" ]] || die "$id not in $HF_CACHE: $why download it"
  sub="hub/models--${id//\//--}/snapshots/$rev"
  [[ -f "$HF_CACHE/$sub/config.json" ]] || die "$id @ ${rev:0:8} not in $HF_CACHE: $why download it"
  for i in $(worker_ids); do
    (( DRY )) && break
    if [[ "$(worker_weights "$i")" == nfs ]]; then
      worker_nfs "$i" test -f "/hf/$sub/config.json" ||
        die "$(wname "$i") does not see $id @ ${rev:0:8} over NFS ($NFS_VOLUME): $why set it up"
    else
      worker "$i" "test -f '${WORKER_HF[i]}/$sub/config.json'" ||
        die "$id @ ${rev:0:8} not on $(wname "$i") (${WORKER_HF[i]}): $why copy it there"
    fi
  done
  echo "/root/.cache/huggingface/$sub"
}
MODEL_ARG=$(snapshot "$MODEL_ID")
if [[ "$DRAFTER" == dflash2 ]]; then DRAFTER_ARG=$(snapshot "$DFLASH2_ID")
else DRAFTER_ARG=none; fi                           # the checkpoint's MTP head, even when DFlash2 is downloaded
SERVE_ARGS=(--drafter "$DRAFTER_ARG" "${SERVE_ARGS[@]}")   # a --drafter on the command line comes later and wins

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: arguments, link, previous server, port, memory"
# tensorfold's own parser, in a throwaway container without the GPU: a typo fails here, before anything is stopped
docker run --rm --entrypoint python "$IMAGE" -c \
  'import sys; from tensorfold.cli import build_parser; build_parser().parse_args(sys.argv[1:])' \
  serve "$MODEL_ARG" --tp "$TP" --rank 0 --master 127.0.0.1 --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" >/dev/null 2>"$STATE_DIR/args.err" ||
  if (( DRY )); then warn "DRY_RUN: tensorfold serve in $IMAGE rejects these arguments: $(tail -1 "$STATE_DIR/args.err")"
  else cat "$STATE_DIR/args.err" >&2; die "tensorfold serve rejects these arguments (see above); nothing was changed"; fi
detect_links
if (( TP == 2 )); then log "Link: $HEAD_ADDR ($HEAD_DEV) <-> $WORKER_ADDR ($WORKER_DEV), RoCE $HEAD_HCAS / $WORKER_HCAS (GID $HEAD_GID / $WORKER_GID; $(tr ',' '\n' <<<"$HEAD_HCAS" | wc -l) rail(s) -> NCCL_IB_HCA, TF_ROCE_HCA)"
else
  log "Rendezvous: $MASTER_ADDR:$MASTER_PORT; NCCL bootstrap over ${NODE_DEV[*]} (rank 0 to $((TP - 1)))"
  for r in 0 $(worker_ids); do
    log "  rank $r: RoCE ${NODE_HCAS[r]} (GID ${NODE_GID[r]:-per device})$( (( r == 0 )) || echo ", link to the head ${LINK_WORKER_ADDR[r]:-?} <-> ${LINK_HEAD_ADDR[r]:-?}")"
  done
fi
here_up=0; running_here && here_up=1
declare -a worker_up=() left_worker=() there_gb=()
any_worker_up=0
for i in $(worker_ids); do worker_up[i]=0; running_worker "$i" && { worker_up[i]=1; any_worker_up=1; }; done
if (( DRY )); then
  (( here_up || any_worker_up )) && log "DRY_RUN: $CONTAINER_NAME is running; a real start would stop it first (./stop.sh)"
elif (( here_up || any_worker_up )); then             # after the setup and the checks: down only while restarting,
  if [[ "$MODE" == start ]]; then                     # or when a start that failed halfway left some ranks up
    if (( TP > 2 )); then log "Only some ranks are running: stopping them, then starting all $TP ranks"
    elif (( here_up )); then log "Only rank 0 is running (here): stopping it, then starting both ranks"
    else log "Only rank 1 is running (on $WORKER): stopping it, then starting both ranks"; fi
  fi
  ./stop.sh
fi
left_here=0; docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && left_here=1
any_left=0
for i in $(worker_ids); do
  left_worker[i]=0
  worker "$i" "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'" 2>/dev/null && { left_worker[i]=1; any_left=1; }
done
if (( ! DRY && ( left_here || any_left ) )); then
  if (( TP == 2 )); then
    where="on both Sparks"; (( left_worker[1] )) || where="here"; (( left_here )) || where="on the worker"
  else
    where=$(if (( left_here )); then echo "here"; fi; for i in $(worker_ids); do if (( left_worker[i] )); then echo "on $(worker_host "$i")"; fi; done)
    where=$(paste -sd, <<<"$where" | sed 's/,/, /g')
  fi
  log "Removing the previous (stopped) container $CONTAINER_NAME $where, its log saved first (a crash's evidence)"
  if (( left_here )); then
    saved=$(save_log "$LOG_DIR" 0 "$CONTAINER_NAME" "$LOG_KEEP") || warn "could not save rank 0's log to $LOG_DIR"
    [[ -z "${saved:-}" ]] || log "Rank 0's log: $saved"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
  for i in $(worker_ids); do
    (( left_worker[i] )) || continue
    saved=$(worker_save_log "$i") || warn "could not save rank $i's log on $(worker_host "$i")"
    [[ -z "${saved:-}" ]] || log "Rank $i's log (on $(worker_host "$i")): $saved"
    worker "$i" "docker rm -f '$CONTAINER_NAME' >/dev/null"
  done
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  (( DRY )) && log "DRY_RUN: port $PORT is in use now" ||
    die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi
# TensorFold budgets each Spark's free memory minus MEMORY_RESERVE_GIB (14.5); the defaults want ~110 GiB free at start on each.
# With less, rank 0 refuses the window and names one that fits, which the load below then starts again with.
here_gb=$(free -g | awk '/^Mem:/ {print $7}')
low=0; (( here_gb >= 110 )) || low=1
for i in $(worker_ids); do
  there_gb[i]=$(worker "$i" "free -g | awk '/^Mem:/ {print \$7}'" 2>/dev/null || echo 0)
  (( there_gb[i] >= 110 )) || low=1
done
if (( ! low )); then
  if (( TP == 2 )); then log "Arguments OK, port $PORT free, ${here_gb} GiB memory available here, ${there_gb[1]} GiB on the worker"
  else log "Arguments OK, port $PORT free, ${here_gb} GiB memory available here, $(for i in $(worker_ids); do printf '%s on rank %s, ' "${there_gb[i]}" "$i"; done | sed 's/, $//')"; fi
else
  (( here_gb >= 110 )) ||
    warn "only ${here_gb} GiB memory available here (the default needs ~110): stop other GPU workloads (docker ps), or lower CONTEXT"
  for i in $(worker_ids); do
    [[ -z "${WORKER_DOWN[$i]:-}" ]] || continue
    (( there_gb[i] >= 110 )) ||
      warn "only ${there_gb[i]} GiB memory available on $(wname "$i") (the default needs ~110): stop other GPU workloads there (ssh $(worker_host "$i") docker ps), or lower CONTEXT"
  done
fi

# TensorFold's own switches (TENSORFOLD_*, TF_GLM_*, TF_ROCE_*) reach every rank with the same values: the workers'
# docker commands run over ssh, where this shell's environment does not reach. None of them is a secret. At TP>2,
# TF_ROCE_HCA names a node's own devices, so each rank gets its own (scripts/nodes.sh, rank_nccl_env); at TP=2 it
# passes as set, as in v1.4 (unset: the ranks take NCCL_IB_HCA's devices).
env_args() {
  local _skip='^$' name
  ENV_ARGS=(-e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}")
  (( TP == 2 )) || _skip='^TF_ROCE_HCA='
  while IFS='=' read -r name _; do ENV_ARGS+=(-e "$name=${!name}"); done < <(env | grep -E '^(TENSORFOLD|TF_GLM|TF_ROCE)_[A-Z0-9_]+=' | grep -v "$_skip" || true)
}
env_args
RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK
          --ulimit memlock=-1 --ulimit stack=67108864)
# DISPLAY_KV_BACKEND=dispram (patch 0087): dispramd's socket and its python client in every rank
if (( DISPLAY_KV_MIB )) && [[ "$DISPLAY_KV_BACKEND" == dispram ]]; then
  RUN_ARGS+=(-v /run/dispram:/run/dispram -v /opt/kindling/dispram/python:/opt/dispram:ro -e PYTHONPATH=/opt/dispram)
fi
if [[ "$SPILL_GIB" != 0 ]]; then                   # the spill tier's directory on every rank, files owned by you
  mkdir -p "$SPILL_DIR"
  for i in $(worker_ids); do worker "$i" "mkdir -p '$SPILL_DIR'" || die "could not create SPILL_DIR on worker $i"; done
  # (files take the owner of SPILL_DIR on each Spark. No --init on these containers: a rank past 0 runs as PID 1
  # without a SIGTERM handler, so a stop leaves it serving rank 0's flush before both exit)
  RUN_ARGS+=(-v "$SPILL_DIR":/spill -e TF_SPILL_FLUSH_S="$SPILL_FLUSH_S")
fi

# ---------------------------------------------------------------- 3. launch, 4. load (a second try when the window does not fit)
# No token goes into the containers: the ranks read only the local cache (HF_HUB_OFFLINE=1), and with HF_HUB_OFFLINE=0
# huggingface_hub finds the token file in the mounted cache.
# Ranks TP-1 .. 1 on their workers first, then rank 0 here, which serves the API. DRY_RUN=1 prints each rank's
# command exactly as it would run (a worker's as the string ssh sends) and exits.
launch() {
  local rank0 rankw worker_cmd remote a i
  local -a here_cmd
  for (( i = TP - 1; i >= 1; i-- )); do
    rankw=(tensorfold serve "$MODEL_ARG" --tp "$TP" --rank "$i" --master "${LINK_HEAD_ADDR[i]:-$MASTER_ADDR}" --master-port "$MASTER_PORT" "${SERVE_ARGS[@]}")
    log "Rank $i on $(worker_host "$i"): ${rankw[*]}"
    worker_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}"
                $(rank_nccl_env "$i")
                -v "${WORKER_MOUNT[i]}" -v "\$HOME/.cache/tensorfold-glm53/$KCACHE:/cache"
                "$IMAGE" "${rankw[@]}")
    remote=""; for a in "${worker_cmd[@]}"; do
      case "$a" in '$HOME'*) remote+=" \"$a\"" ;; *) remote+=" $(printf '%q' "$a")" ;; esac
    done
    if (( DRY )); then
      printf '[dry-run] rank %s on %s:\n  %s\n' "$i" "$(worker_host "$i")" "mkdir -p \$HOME/.cache/tensorfold-glm53 &&$remote"
      continue
    fi
    worker "$i" "mkdir -p \$HOME/.cache/tensorfold-glm53 &&$remote" >/dev/null || die "could not start rank $i on $(worker_host "$i")"
  done
  rank0=(tensorfold serve "$MODEL_ARG" --tp "$TP" --rank 0 --master "$MASTER_ADDR" --master-port "$MASTER_PORT"
         --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}")
  log "Rank 0 here: ${rank0[*]}"
  here_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}"
            $(rank_nccl_env 0)
            -v "$HF_CACHE":/root/.cache/huggingface -v "$KERNEL_CACHE/$KCACHE":/cache
            "$IMAGE" "${rank0[@]}")
  if (( DRY )); then
    printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${here_cmd[@]}" | sed 's/ $//')"
    exit 0
  fi
  "${here_cmd[@]}" >/dev/null
}
# FOREGROUND=1: stay attached to rank 0's log and exit with its code (systemd's Restart=on-failure). Either rank ending
# takes the other one down: a lone rank would wait for its peer forever.
foreground() {
  local watch w code i
  trap './stop.sh; exit 130' INT TERM
  ( exec 8>&-                                        # not holding start.sh's lock once start.sh has exited
    while sleep 30; do
      running_here || exit 0
      for i in $(worker_ids); do
        worker "$i" true 2>/dev/null || continue    # a worker out of reach for a moment says nothing about its rank
        running_worker "$i" && continue
        warn "rank $i on $(worker_host "$i") exited: stopping rank 0"
        docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1
        exit 1
      done
    done ) &
  watch=$!
  docker logs -f "$CONTAINER_NAME" || true
  kill "$watch" 2>/dev/null || true
  w=0; wait "$watch" || w=$?
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)
  [[ "$w" != 1 || "$code" != 0 ]] || code=1         # a worker's rank failed, even if rank 0 then shut down cleanly
  for i in $(worker_ids); do worker "$i" "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME'" >/dev/null 2>&1 || true; done
  exit "$code"
}
# NVIDIA's container banner, without its license notice (GOVERNING TERMS ...), which stays visible
NOISE='^\s*$|EXL3 support is experimental|^=+$|^== PyTorch ==|^NVIDIA Release|Copyright|All rights reserved|PyTorch Version|Various files include|NOTE: CUDA Forward|Using CUDA|cuda-compatibility|Container image|torch/utils/_pytree\.py.*register_constant'
LOGS_PID=""
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
# GPU memory a container's processes hold so far (GiB); on the worker through ssh, with this definition
gpu_gib() {
  local pids
  pids=$(docker top "$1" -eo pid 2>/dev/null | tail -n +2 | paste -sd'|')
  [[ -n "$pids" ]] || { echo 0; return; }
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' -v re="^($pids)$" '$1 ~ re { s += $2 } END { printf "%.1f", s / 1024 }'
}
fail() {
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.5
  printf '\n%s── rank 0 (here): last server log lines ──%s\n' "$D" "$R"
  docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  for i in $(worker_ids); do
    printf '%s── rank %s (%s): last server log lines ──%s\n' "$D" "$i" "$(worker_host "$i")" "$R"
    worker "$i" docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  done
  die "$1"
}
# Issue #36: on some pairs a rank's first NCCL connection fails with SPLIT=1, seconds into the start and before any
# weights load (NCCL error 2, ibv_reg_mr: Cannot allocate memory), about half the time; a start that comes up stays up.
# Such a start is tried once more as it was, then with SPLIT=0 (prompt chunks unsplit: the same replies, long prompts
# fill slower), and says so.
split_nccl_failed() {
  local i
  [[ "$SPLIT" == 1 ]] || return 1
  { docker logs "$CONTAINER_NAME" 2>&1 || true
    for i in $(worker_ids); do worker "$i" docker logs "$CONTAINER_NAME" 2>&1 || true; done
  } | grep 'NCCL error 2: unhandled system error' >/dev/null      # not -q: an early exit fails the pipe (pipefail)
}
split_tries=0; refitted=0
for attempt in 1 2 3 4; do
  if (( TP == 2 )); then step 3 "Launch: container $CONTAINER_NAME, rank 1 on $WORKER, then rank 0 here"
  else step 3 "Launch: container $CONTAINER_NAME, ranks $((TP - 1)) to 1 on the workers, then rank 0 here"; fi
  launch
  [[ "${FOREGROUND:-0}" == 1 ]] && foreground
  case "$TP" in
    3) step 4 "Loading: the weights on each Spark (~60 GiB on rank 0, ~50 on the others) (2-6 min; the very first start also compiles CUDA kernels)" ;;
    *) step 4 "Loading: ~80 GiB of weights on each Spark (2-6 min; the very first start also compiles CUDA kernels)" ;;
  esac
  # docker logs is the background job, so killing it ends the whole pipeline (no orphaned `docker logs -f`)
  docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE" | sed -u "s/^/  ${D}│${R} /") 2>&1 &
  LOGS_PID=$!
  start=$SECONDS; next_beat=15; refit=""; retry=""
  until curl -sf --max-time 5 "$URL/v1/models" >/dev/null 2>&1; do
    if ! running_here; then
      # the memory at this start holds a smaller window than asked: once, take the largest one TensorFold names
      refit=$(docker logs "$CONTAINER_NAME" 2>&1 | sed -n 's/.*largest fitting prompt-plus-reply window: \([0-9]*\) tokens.*/\1/p' | tail -1)
      [[ -n "$refit" && $refitted == 0 ]] && break
      refit=""
      (( split_tries < 2 )) && split_nccl_failed && { retry=split; break; }
      fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
    fi
    (( SECONDS - start < WAIT_TIMEOUT )) ||
      fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); the ranks are still running: docker logs -f $CONTAINER_NAME"
    if (( SECONDS - start >= next_beat )); then
      for i in $(worker_ids); do
        running_worker "$i" && continue
        (( split_tries < 2 )) && split_nccl_failed && { retry=split; break 2; }
        fail "rank $i on $(worker_host "$i") exited (code $(worker "$i" docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo "?")) before the server was ready"
      done
      estimate=$(docker logs "$CONTAINER_NAME" 2>&1 | sed -n 's/.*startup estimate \([0-9.]*\) GiB.*/\1/p' | tail -1)
      there=""
      for i in $(worker_ids); do
        g=$(worker "$i" "$(declare -f gpu_gib); gpu_gib $CONTAINER_NAME" 2>/dev/null || echo "?")
        if (( TP == 2 )); then there+=", $g on the worker"; else there+=", $g on rank $i"; fi
      done
      printf '  %s⋯ %ss elapsed, %s%s GiB on the GPU here%s%s\n' "$D" "$((SECONDS - start))" \
        "$(gpu_gib "$CONTAINER_NAME")" "${estimate:+ of ~$estimate}" "$there" "$R"
      next_beat=$((next_beat + 15))
    fi
    sleep 3
  done
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.3
  if [[ "$retry" == split ]]; then
    split_tries=$((split_tries + 1))
    if (( split_tries == 1 )); then
      warn "a rank's first NCCL connection failed with SPLIT=1 (NCCL error 2, issue #36): starting again as it was"
    else
      warn "it failed again: starting with SPLIT=0 (prompt chunks unsplit: the same replies, long prompts fill slower); SPLIT=0 in .env skips these tries (issue #36)"
      SPLIT=0
      export TF_GLM_HC_SPLIT=0 TF_GLM_PREFILL_OVERLAP=0
      env_args
    fi
    ./stop.sh >/dev/null
    continue
  fi
  [[ -z "$refit" ]] && break
  refitted=1
  warn "this start's memory budget holds a ${refit}-token window, not $(arg_value --context): starting again with --context $refit"
  CONTEXT=$refit
  SERVE_ARGS+=(--context "$refit")
  ./stop.sh >/dev/null
done
log "Server answered after $((SECONDS - start))s"

# ---------------------------------------------------------------- 5. smoke test
# Thinking off and greedy, so that a short reply has text (the model thinks first otherwise); no text, or one character repeated (a collapsed model: issues #76, #81, #86), fails the start.
step 5 "Smoke test: one chat completion through $( (( TP == 2 )) && echo "both" || echo "all $TP") ranks"
SERVED=$(served_name || echo "$SERVED_NAME")
if smoke=$(curl -s --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED\", \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); assert len(c.strip()) < 8 or len(set(c.strip())) > 2, "one repeated character"; print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens,", r.get("tensorfold", {}).get("decode_s"), "s")' 2>/dev/null); then
  log "OK: $smoke"
else
  fail "the smoke test request failed (no reply text, or one character repeated: the weights load but the model does not answer; see the README's Ablit weights); the ranks are still running"
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED)
    Window   $(arg_value --context) tokens$( (( TP == 2 )) || echo " · $TP Sparks (experimental)") · $(arg_value --parallel) at once · $KV KV · $DRAFTER drafts · $DENSE dense weights$( [[ "$VISION" == 1 ]] && echo " · images and video")$( [[ "$COMM" == roce ]] && echo " · RoCE all-gathers")$( [[ "$SPLIT" == 1 ]] && echo " · split prefill")$( [[ "$KDA_CHUNKED" == 1 ]] && echo " · chunked KDA")
    Drafts   $DRAFT_POLICY · copy drafts $( [[ "$COPY" == 1 ]] && echo "up to $COPY_MAX$( [[ "$COPY_CODE" == 1 ]] && echo ", code rules")" || echo off) · shared system prompts $( [[ "$SHARED_PREFIX" == 1 ]] && echo on || echo off)
    Logs     docker logs -f $CONTAINER_NAME$(for i in $(worker_ids); do printf '   (rank %s: ssh %s docker logs -f %s)' "$i" "$(worker_host "$i")" "$CONTAINER_NAME"; done)
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
