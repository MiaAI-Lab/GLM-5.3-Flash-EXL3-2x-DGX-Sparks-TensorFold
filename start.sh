#!/usr/bin/env bash
# Serve GLM-5.3 Flash EXL3 (Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) with TensorFold on two DGX Sparks, end to end:
# runs scripts/prepare.sh on both Sparks when the image or the checkpoint is not ready yet (first run, or after patches
# change), starts rank 1 on the worker and rank 0 here, which serves the API on port 8888, waits until the OpenAI API
# answers, then runs a smoke test. Stop it with ./stop.sh.
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
# Extra arguments go to both ranks after the defaults, so they win (the last value of a flag counts).
# Setup: WORKER=user@<worker address> in scripts/local.sh (key-based ssh).
# Settings, from the environment, scripts/local.sh or ./.env (defaults and measured effects in scripts/config.sh):
#   serving  CONTEXT, PARALLEL, KV, DENSE, DRAFTER, DRAFT_POLICY, COPY, COPY_MAX, COPY_CODE, SPLIT, KDA_CHUNKED,
#            SHARED_PREFIX, MULTI_PREFILL, KV_POOL_GIB, MEMORY_RESERVE_GIB, MAX_TOKENS, THINKING, VISION, VISION_URLS, COMM, SERVED_NAME, HOST, PORT
#   nodes    WORKER, FABRIC_PEER, MASTER_PORT, NCCL_RAILS (1: one CX7 port), NCCL_CHANNELS, NCCL_DEBUG
#   files    MODEL_ID, MODEL_REVISION, DFLASH2_ID, DFLASH2_REVISION, HF_CACHE (default: HF_HOME), KERNEL_CACHE,
#            WORKER_WEIGHTS (copy | nfs: rank 1 reads the head's HF_CACHE over NFS), NFS_PATH, NFS_SERVER, NFS_VOLUME,
#            STATE_DIR, HF_HUB_OFFLINE=0 (let TensorFold reach the Hub; default serves from the local cache only)
#   image    IMAGE, TF_VERSION, TF_REPO, BASE_IMAGE, GHCR_IMAGE, IMAGE_TAG / IMAGE_DIGEST (the pinned published
#            image), CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), PULL, MIN_FREE_GB, IMAGE_FREE_GB, RSYNC_OPTS, HF_TOKEN (prepare.sh's downloads);
#            FOREGROUND=1 (stay attached to rank 0's log, exit with its code); WAIT_TIMEOUT (seconds, default 1800);
#            STOP_TIMEOUT (stop.sh)
#   decode   TF_GLM_L2PF (1), TF_GLM_EXL3_LOADS (nc), TF_ROCE_MAX_KB (512), TF_GLM_MULTI_LONE (1)
#   ranks    every TENSORFOLD_*, TF_GLM_* and TF_ROCE_* variable goes to both ranks, e.g.
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

# The serve arguments both ranks share: scripts/config.sh's defaults first, then the command line's (argparse keeps
# the last value). --drafter goes in front after the setup step, which knows DFlash2's snapshot.
SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS")
[[ "$DRAFTER" =~ ^(mtp|dflash2)$ ]] || die "DRAFTER is mtp or dflash2, not $DRAFTER"
[[ "$DENSE" =~ ^(bf16|fp8|q4)$ ]] || die "DENSE is bf16, fp8 or q4, not $DENSE"
[[ "$COMM" =~ ^(nccl|roce)$ ]] || die "COMM is nccl or roce, not $COMM"
[[ "$KV" =~ ^(bf16|fp8)$ ]] || die "KV is bf16 or fp8, not $KV"
[[ "$CONTEXT" =~ ^[0-9]+$ && "$CONTEXT" -le 1048576 ]] || die "CONTEXT is a token count up to 1048576 (0: the largest that fits), not $CONTEXT"
[[ "$PARALLEL" =~ ^[1-4]$ ]] || die "PARALLEL is 1 to 4, not $PARALLEL"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS is a token count, not $MAX_TOKENS"
[[ "$DRAFTER" == dflash2 || "$PARALLEL" == 1 ]] || die "PARALLEL=$PARALLEL needs DRAFTER=dflash2 (mtp serves one request at a time: PARALLEL=1)"
for v in SPLIT SHARED_PREFIX KDA_CHUNKED COPY_CODE MULTI_PREFILL; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$WORKER_WEIGHTS" == copy || "$WORKER_WEIGHTS" == nfs ]] || die "WORKER_WEIGHTS is copy or nfs, not $WORKER_WEIGHTS"
[[ "$KV_POOL_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "KV_POOL_GIB is a number of GiB, not $KV_POOL_GIB"
[[ "$MEMORY_RESERVE_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "MEMORY_RESERVE_GIB is a number of GiB, not $MEMORY_RESERVE_GIB"
[[ "$COPY_MAX" =~ ^([1-9]|1[0-5])$ ]] || die "COPY_MAX is 1 to 15, not $COPY_MAX"
[[ "$DRAFT_POLICY" == f* ]] || die "DRAFT_POLICY is a DFlash2 policy (fc5:0.3, fnc7:0.3, fcost7:noisy ...), not $DRAFT_POLICY"
# decode settings the ranks read at load (scripts/config.sh): checked here so a typo fails before anything is stopped
[[ "$TF_GLM_MULTI_LONE" =~ ^[01]$ ]] || die "TF_GLM_MULTI_LONE is 0 or 1, not $TF_GLM_MULTI_LONE"
[[ "$TF_GLM_CLEAR_THINKING" =~ ^[01]$ ]] || die "TF_GLM_CLEAR_THINKING is 0 or 1, not $TF_GLM_CLEAR_THINKING"
[[ "$TF_GLM_L2PF" =~ ^(0|off|1|bulk|lines|touch)$ ]] || die "TF_GLM_L2PF is 0, 1 (bulk), lines or touch, not $TF_GLM_L2PF"
[[ "$TF_GLM_EXL3_LOADS" =~ ^(0|ldg|1|nc|nc1|nc2|nc4)$ ]] || die "TF_GLM_EXL3_LOADS is 0, nc, nc2 or nc4, not $TF_GLM_EXL3_LOADS"
[[ "$TF_ROCE_MAX_KB" =~ ^[1-9][0-9]*$ ]] || die "TF_ROCE_MAX_KB is a size in KiB (512: up to 32-row windows over RoCE), not $TF_ROCE_MAX_KB"
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$VISION" == 1 ]] && SERVE_ARGS+=(--vision)
[[ "$VISION" == 1 && "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
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
running_worker() { [[ "$(worker docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
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
printf '%s  %s · 2 x DGX Spark · %s at once · %s-token window · %s KV · %s drafts · port %s%s\n\n' "$D" "$MODEL_ID" \
  "$(arg_value --parallel)" "$(arg_value --context)" "$KV" "$DRAFTER" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$KERNEL_CACHE" "$STATE_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_worker

# ---------------------------------------------------------------- already running?
if [[ "$MODE" == start ]] && running_here && running_worker; then
  log "$CONTAINER_NAME is already running on both Sparks (model: $(served_name || echo "not answering yet"), port $PORT): nothing to do."
  log "Use ./start.sh restart to restart it (e.g. with new settings), or ./stop.sh to stop it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
# scripts/prepare.sh (image and checkpoint on both Sparks) runs whenever what it last prepared differs from now: the
# first run, new patches, another model, drafter or worker. PREPARE=1 forces it, PREPARE=0 skips it.
step 1 "Setup: image and checkpoint on both Sparks"
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  log "Not ready yet: running scripts/prepare.sh (the first time this pulls the image, downloads ~166 GiB and copies both to the worker)"
  ./scripts/prepare.sh
else
  log "Ready: $IMAGE (patches $(image_hash)) and $MODEL_ID on both Sparks${PREPARE:+ (PREPARE=$PREPARE)}"
fi
why="scripts/prepare.sh did not"; [[ "${PREPARE:-auto}" == 0 ]] && why="PREPARE=0 skipped scripts/prepare.sh, which would"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing: $why build it"
worker docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing on the worker: $why copy it there"
# Compiled kernels are kept per image (its patches hash): a build is found by its extension's name, so another image's
# build of the same name, older or newer, must never be the one loaded
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
[[ "$KCACHE" =~ ^[0-9a-f]{12}$ ]] || KCACHE=$(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -d: -f2 | cut -c1-12)
# (docker creates the folder at the mount: the cache is the containers', so it may not be writable from here)
# The snapshots both ranks serve (config.sh's pins, else refs/main), as paths under the containers' cache mount: the
# ranks read them offline, whatever the Hub's main is now. The worker's cache is its own HF_HOME (prepare.sh copies
# into the same place).
WORKER_HF=$(worker_hf_cache)
WORKER_MOUNT="$WORKER_HF:/root/.cache/huggingface"                 # rank 1's cache: its own copy, or the head's (nfs)
[[ "$WORKER_WEIGHTS" == nfs ]] && WORKER_MOUNT="$NFS_VOLUME:/root/.cache/huggingface:ro"
snapshot() {  # <repo id>: its snapshot path in the container, checked on both Sparks
  local id=$1 rev sub
  rev=$(snapshot_rev "$id")
  [[ -n "$rev" ]] || die "$id not in $HF_CACHE: $why download it"
  sub="hub/models--${id//\//--}/snapshots/$rev"
  [[ -f "$HF_CACHE/$sub/config.json" ]] || die "$id @ ${rev:0:8} not in $HF_CACHE: $why download it"
  if [[ "$WORKER_WEIGHTS" == nfs ]]; then
    worker_nfs test -f "/hf/$sub/config.json" ||
      die "the worker does not see $id @ ${rev:0:8} over NFS ($NFS_VOLUME): $why set it up"
  else
    worker "test -f '$WORKER_HF/$sub/config.json'" || die "$id @ ${rev:0:8} not on the worker ($WORKER_HF): $why copy it there"
  fi
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
  serve "$MODEL_ARG" --tp 2 --rank 0 --master 127.0.0.1 --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" >/dev/null ||
  die "tensorfold serve rejects these arguments (see above); nothing was changed"
detect_link
log "Link: $HEAD_ADDR ($HEAD_DEV) <-> $WORKER_ADDR ($WORKER_DEV), RoCE $HEAD_HCAS / $WORKER_HCAS"
here_up=0; running_here && here_up=1
worker_up=0; running_worker && worker_up=1
if (( here_up || worker_up )); then                   # after the setup and the checks: down only while restarting,
  if [[ "$MODE" == start ]]; then                     # or when a start that failed halfway left one rank up
    if (( here_up )); then log "Only rank 0 is running (here): stopping it, then starting both ranks"
    else log "Only rank 1 is running (on $WORKER): stopping it, then starting both ranks"; fi
  fi
  ./stop.sh
fi
left_here=0; docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && left_here=1
left_worker=0; worker "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'" 2>/dev/null && left_worker=1
if (( left_here || left_worker )); then
  where="on both Sparks"; (( left_worker )) || where="here"; (( left_here )) || where="on the worker"
  log "Removing the previous (stopped) container $CONTAINER_NAME $where"
  (( left_here == 0 )) || docker rm -f "$CONTAINER_NAME" >/dev/null
  (( left_worker == 0 )) || worker "docker rm -f '$CONTAINER_NAME' >/dev/null"
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi
# TensorFold budgets each Spark's free memory minus MEMORY_RESERVE_GIB (14.5); the defaults want ~110 GiB free at start on each.
# With less, rank 0 refuses the window and names one that fits, which the load below then starts again with.
here_gb=$(free -g | awk '/^Mem:/ {print $7}')
there_gb=$(worker "free -g | awk '/^Mem:/ {print \$7}'")
if (( here_gb >= 110 && there_gb >= 110 )); then
  log "Arguments OK, port $PORT free, ${here_gb} GiB memory available here, ${there_gb} GiB on the worker"
else
  (( here_gb >= 110 )) ||
    warn "only ${here_gb} GiB memory available here (the default needs ~110): stop other GPU workloads (docker ps), or lower CONTEXT"
  (( there_gb >= 110 )) ||
    warn "only ${there_gb} GiB memory available on the worker (the default needs ~110): stop other GPU workloads there (ssh $WORKER docker ps), or lower CONTEXT"
fi

# TensorFold's own switches (TENSORFOLD_*, TF_GLM_*, TF_ROCE_*) reach both ranks with their values: rank 1's docker
# command runs over ssh, where this shell's environment does not reach. None of them is a secret.
ENV_ARGS=(-e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}")
while IFS='=' read -r name _; do ENV_ARGS+=(-e "$name=${!name}"); done < <(env | grep -E '^(TENSORFOLD|TF_GLM|TF_ROCE)_[A-Z0-9_]+=' || true)
RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK
          --ulimit memlock=-1 --ulimit stack=67108864)

# ---------------------------------------------------------------- 3. launch, 4. load (a second try when the window does not fit)
# No token goes into the containers: the ranks read only the local cache (HF_HUB_OFFLINE=1), and with HF_HUB_OFFLINE=0
# huggingface_hub finds the token file in the mounted cache.
launch() {
  local rank1 rank0 worker_cmd remote a
  rank1=(tensorfold serve "$MODEL_ARG" --tp 2 --rank 1 --master "$HEAD_ADDR" --master-port "$MASTER_PORT" "${SERVE_ARGS[@]}")
  rank0=(tensorfold serve "$MODEL_ARG" --tp 2 --rank 0 --master "$HEAD_ADDR" --master-port "$MASTER_PORT"
         --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}")
  log "Rank 1 on $WORKER: ${rank1[*]}"
  log "Rank 0 here: ${rank0[*]}"
  worker_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}"
              $(nccl_env "$WORKER_DEV" "$WORKER_HCAS" "$WORKER_GID")
              -v "$WORKER_MOUNT" -v "\$HOME/.cache/tensorfold-glm53/$KCACHE:/cache"
              "$IMAGE" "${rank1[@]}")
  remote=""; for a in "${worker_cmd[@]}"; do
    case "$a" in '$HOME'*) remote+=" \"$a\"" ;; *) remote+=" $(printf '%q' "$a")" ;; esac
  done
  worker "mkdir -p \$HOME/.cache/tensorfold-glm53 &&$remote" >/dev/null || die "could not start rank 1 on $WORKER"
  docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}" \
    $(nccl_env "$HEAD_DEV" "$HEAD_HCAS" "$HEAD_GID") \
    -v "$HF_CACHE":/root/.cache/huggingface -v "$KERNEL_CACHE/$KCACHE":/cache \
    "$IMAGE" "${rank0[@]}" >/dev/null
}
# FOREGROUND=1: stay attached to rank 0's log and exit with its code (systemd's Restart=on-failure). Either rank ending
# takes the other one down: a lone rank would wait for its peer forever.
foreground() {
  local watch w code
  trap './stop.sh; exit 130' INT TERM
  ( exec 8>&-                                        # not holding start.sh's lock once start.sh has exited
    while sleep 30; do
      running_here || exit 0
      worker true 2>/dev/null || continue           # the worker out of reach for a moment says nothing about rank 1
      running_worker && continue
      warn "rank 1 on $WORKER exited: stopping rank 0"
      docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1
      exit 1
    done ) &
  watch=$!
  docker logs -f "$CONTAINER_NAME" || true
  kill "$watch" 2>/dev/null || true
  w=0; wait "$watch" || w=$?
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)
  [[ "$w" != 1 || "$code" != 0 ]] || code=1         # rank 1 failed, even if rank 0 then shut down cleanly
  worker "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME'" >/dev/null 2>&1 || true
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
  printf '%s── rank 1 (%s): last server log lines ──%s\n' "$D" "$WORKER" "$R"
  worker docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  die "$1"
}
for attempt in 1 2; do
  step 3 "Launch: container $CONTAINER_NAME, rank 1 on $WORKER, then rank 0 here"
  launch
  [[ "${FOREGROUND:-0}" == 1 ]] && foreground
  step 4 "Loading: ~80 GiB of weights on each Spark (2-6 min; the very first start also compiles CUDA kernels)"
  # docker logs is the background job, so killing it ends the whole pipeline (no orphaned `docker logs -f`)
  docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE" | sed -u "s/^/  ${D}│${R} /") 2>&1 &
  LOGS_PID=$!
  start=$SECONDS; next_beat=15; refit=""
  until curl -sf --max-time 5 "$URL/v1/models" >/dev/null 2>&1; do
    if ! running_here; then
      # the memory at this start holds a smaller window than asked: once, take the largest one TensorFold names
      refit=$(docker logs "$CONTAINER_NAME" 2>&1 | sed -n 's/.*largest fitting prompt-plus-reply window: \([0-9]*\) tokens.*/\1/p' | tail -1)
      [[ -n "$refit" && $attempt == 1 ]] && break
      fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
    fi
    (( SECONDS - start < WAIT_TIMEOUT )) ||
      fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); both ranks are still running: docker logs -f $CONTAINER_NAME"
    if (( SECONDS - start >= next_beat )); then
      running_worker ||
        fail "rank 1 on $WORKER exited (code $(worker docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo "?")) before the server was ready"
      estimate=$(docker logs "$CONTAINER_NAME" 2>&1 | sed -n 's/.*startup estimate \([0-9.]*\) GiB.*/\1/p' | tail -1)
      printf '  %s⋯ %ss elapsed, %s%s GiB on the GPU here, %s on the worker%s\n' "$D" "$((SECONDS - start))" \
        "$(gpu_gib "$CONTAINER_NAME")" "${estimate:+ of ~$estimate}" \
        "$(worker "$(declare -f gpu_gib); gpu_gib $CONTAINER_NAME" 2>/dev/null || echo "?")" "$R"
      next_beat=$((next_beat + 15))
    fi
    sleep 3
  done
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.3
  [[ -z "$refit" ]] && break
  warn "this start's memory budget holds a ${refit}-token window, not $(arg_value --context): starting again with --context $refit"
  CONTEXT=$refit
  SERVE_ARGS+=(--context "$refit")
  ./stop.sh >/dev/null
done
log "Server answered after $((SECONDS - start))s"

# ---------------------------------------------------------------- 5. smoke test
# Thinking off and greedy, so that a short reply has text (the model thinks first otherwise); no text fails the start.
step 5 "Smoke test: one chat completion through both ranks"
SERVED=$(served_name || echo "$SERVED_NAME")
if smoke=$(curl -s --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED\", \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens,", r.get("tensorfold", {}).get("decode_s"), "s")' 2>/dev/null); then
  log "OK: $smoke"
else
  fail "the smoke test request failed (no reply text); both ranks are still running"
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED)
    Window   $(arg_value --context) tokens · $(arg_value --parallel) at once · $KV KV · $DRAFTER drafts · $DENSE dense weights$( [[ "$VISION" == 1 ]] && echo " · images and video")$( [[ "$COMM" == roce ]] && echo " · RoCE all-gathers")$( [[ "$SPLIT" == 1 ]] && echo " · split prefill")$( [[ "$KDA_CHUNKED" == 1 ]] && echo " · chunked KDA")
    Drafts   $DRAFT_POLICY · copy drafts $( [[ "$COPY" == 1 ]] && echo "up to $COPY_MAX$( [[ "$COPY_CODE" == 1 ]] && echo ", code rules")" || echo off) · shared system prompts $( [[ "$SHARED_PREFIX" == 1 ]] && echo on || echo off)
    Logs     docker logs -f $CONTAINER_NAME   (rank 1: ssh $WORKER docker logs -f $CONTAINER_NAME)
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
