#!/usr/bin/env bash
# Stop the server that ./start.sh (or ./start-tp3.sh, ./start-tp4.sh) started and remove its containers on every Spark,
# freeing their GPU memory: here (rank 0) and on every configured worker (WORKER, WORKER2, WORKER3), whatever TP is.
# The ranks get STOP_TIMEOUT seconds (default 30) to shut down; requests still running are cut off (it does not drain
# them), so stop.sh says when there are any. Each rank's log is saved first (docker rm deletes it), gzipped, in LOG_DIR
# here (~/.cache/tensorfold-glm53/logs) and ~/.cache/tensorfold-glm53/logs on each worker; the newest LOG_KEEP (10) of
# each rank stay.
# Usage: ./stop.sh      Env: CONTAINER_NAME, PORT, WORKER, WORKER2 (see scripts/config.sh), STOP_TIMEOUT,
#                       LOG_DIR, LOG_KEEP
#   DRY_RUN=1 ./stop.sh   says what it would stop, and on which Spark, and stops nothing
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
source ./scripts/nodes.sh

# Where the containers are: here (rank 0) and on the workers (rank i on worker i)
here=0; docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && here=1
workers=$(configured_workers)
nw=$(wc -w <<<"$workers")
declare -a there=()
any=0
[[ -n "$workers" ]] || warn "WORKER is not set (scripts/local.sh): rank 1 was left alone"
for i in $workers; do
  there[i]=0
  if ! worker "$i" true 2>/dev/null; then
    warn "cannot reach the worker ($(worker_host "$i")) over ssh: rank $i was left alone"
  elif worker "$i" "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'"; then
    there[i]=1; any=1
  fi
done
if (( here == 0 && any == 0 )); then
  if (( nw <= 1 )); then log "No container named $CONTAINER_NAME on either Spark: nothing to stop"
  else log "No container named $CONTAINER_NAME on any Spark: nothing to stop"; fi
  exit 0
fi
if (( nw <= 1 )); then
  where="on both Sparks"; (( any )) || where="here"; (( here )) || where="on the worker"
else
  # (if, not &&: a last false test would be the substitution's status and stop the script under set -e)
  where=$(if (( here )); then echo "here"; fi; for i in $workers; do if (( there[i] )); then echo "on $(worker_host "$i")"; fi; done)
  where=$(paste -sd, <<<"$where" | sed 's/,/, /g')
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; then
  api_host="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && api_host=127.0.0.1
  [[ "$api_host" == *:* ]] && api_host="[$api_host]"
  busy=$(curl -s --max-time 3 "http://$api_host:$PORT/health" 2>/dev/null |
         python3 -c 'import json,sys; print(json.load(sys.stdin).get("requests_running", 0))' 2>/dev/null || echo 0)
  (( busy == 0 )) || warn "$busy request(s) still running will be cut off"
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  log "DRY_RUN=1: would stop and remove $CONTAINER_NAME $where (up to ${STOP_TIMEOUT:-30}s); nothing is stopped"
  if (( here )); then printf '[dry-run] rank 0 here:\n  docker stop -t %s %s; (its log saved to %s); docker rm -f %s\n' "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" "$LOG_DIR" "$CONTAINER_NAME"; fi
  for i in $workers; do
    if (( there[i] )); then printf '[dry-run] rank %s on %s:\n  docker stop -t %s %s; (its log saved to ~/.cache/tensorfold-glm53/logs there); docker rm -f %s\n' "$i" "$(worker_host "$i")" "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" "$CONTAINER_NAME"; fi
  done
  exit 0
fi
log "Stopping $CONTAINER_NAME $where (up to ${STOP_TIMEOUT:-30}s)"

# each rank: stop, save its log (with the shutdown lines), then remove the container
stop_here() {
  local saved
  docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1 || true
  saved=$(save_log "$LOG_DIR" 0 "$CONTAINER_NAME" "$LOG_KEEP") || warn "could not save rank 0's log to $LOG_DIR"
  docker rm -f "$CONTAINER_NAME" >/dev/null
  log "Stopped and removed rank 0 here${saved:+; its log: $saved}"
}
stop_worker() {
  local saved h
  h=$(worker_host "$1")
  worker "$1" "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME' >/dev/null 2>&1" || true
  saved=$(worker_save_log "$1") || warn "could not save rank $1's log on $h"
  worker "$1" "docker rm -f '$CONTAINER_NAME' >/dev/null" || { warn "could not stop rank $1 on $h"; return 1; }
  log "Stopped and removed rank $1 on $h${saved:+; its log there: $saved}"
}
wpids=()
for i in $workers; do if (( there[i] )); then stop_worker "$i" & wpids+=($!); fi; done
if (( here )); then stop_here; fi
failed=0
for p in "${wpids[@]}"; do wait "$p" || failed=1; done
(( failed == 0 )) || exit 0                             # a warning above said what is left
log "Stopped and removed $CONTAINER_NAME $where; its GPU memory is free again"
