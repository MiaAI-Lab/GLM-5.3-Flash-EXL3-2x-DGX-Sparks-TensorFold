#!/usr/bin/env bash
# Serve GLM-5.3 Flash EXL3 on four DGX Sparks (TP=4, experimental): ./start.sh with TP=4, COMM=nccl.
# Needs the TP-N GLM engine (patches 0066-0068, in the same published image), WORKER, WORKER2 and WORKER3 in
# scripts/local.sh, and the Sparks cabled to each other directly: a ring without a switch (each Spark's two CX7 ports to
# its two neighbours, one subnet per cable) with WORKER .. WORKER3 in the ring's order, or every pair on a subnet.
# README: "4 Sparks (experimental)".
#
# Usage: ./start-tp4.sh [restart] [extra tensorfold serve args]   (as ./start.sh; ./stop.sh stops it)
#   DRY_RUN=1 ./start-tp4.sh            # print the links found and every rank's docker command, change nothing
# On a ring NCCL carries every all-gather (COMM=nccl) as a ring in rank order, so each rank sends only to its
# neighbour; COMM=roce is refused there.
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; echo; echo "./start.sh's options:"; echo ;;
esac
export TP=4 COMM="${COMM:-nccl}"
exec "$(dirname "$(readlink -f "$0")")/start.sh" "$@"
