#!/usr/bin/env bash
# Serve GLM-5.3 Flash EXL3 on four DGX Sparks without a switch (TP=4, experimental): ./start.sh with TP=4, COMM=nccl.
# Needs the TP-N GLM engine (patches 0066-0068), the ring exchange (patch 0098) and the RoCE relay (patch 0099), in
# the same image, WORKER, WORKER2 and WORKER3 in scripts/local.sh, and the Sparks cabled as a ring: each Spark's two
# CX7 ports to its two neighbours, one subnet per cable, with WORKER .. WORKER3 in the ring's order. Four Sparks on a
# switch: ./start-tp4.sh; this script stops there and says so.
# README: "4 Sparks (experimental)".
#
# Usage: ./start-tp4-switchless.sh [restart] [extra tensorfold serve args]   (as ./start.sh; ./stop.sh stops it)
#   DRY_RUN=1 ./start-tp4-switchless.sh   # print the ring found and every rank's docker command, change nothing
# On the ring NCCL carries every all-gather as a ring in rank order, so each rank sends only to its neighbour.
# COMM=roce ./start-tp4-switchless.sh sends the small ones over RoCE to the neighbours, which relay to the Spark across
# (patch 0099; not measured on the Sparks yet). COMM in scripts/local.sh or .env is not read here: set it on the
# command line.
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; echo; echo "./start.sh's options:"; echo ;;
esac
export TP=4 COMM="${COMM:-nccl}" FABRIC_EXPECT=ring
exec "$(dirname "$(readlink -f "$0")")/start.sh" "$@"
