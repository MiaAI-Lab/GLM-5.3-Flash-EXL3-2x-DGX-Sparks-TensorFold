#!/usr/bin/env bash
# Serve GLM-5.3 Flash EXL3 on four DGX Sparks on a switch (TP=4, experimental): ./start.sh with TP=4, COMM=nccl by
# default. Needs the TP-N GLM engine (patches 0066-0068, in the same image), WORKER, WORKER2 and WORKER3 in
# scripts/local.sh, and every pair of Sparks on one RoCE subnet: their CX7 ports on one switch. Four Sparks cabled to
# each other without a switch (a ring): ./start-tp4-switchless.sh; this script stops there and says so.
# README: "4 Sparks (experimental)".
#
# Usage: ./start-tp4.sh [restart] [extra tensorfold serve args]   (as ./start.sh; ./stop.sh stops it)
#   DRY_RUN=1 ./start-tp4.sh            # print the links found and every rank's docker command, change nothing
# COMM defaults to nccl here (NCCL for every all-gather); COMM=roce ./start-tp4.sh sends the small ones over RoCE,
# each peer on the devices that share its subnet (needs the TP-N engine's RoCE). COMM in scripts/local.sh or .env is
# not read here: set it on the command line.
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; echo; echo "./start.sh's options:"; echo ;;
esac
export TP=4 COMM="${COMM:-nccl}" FABRIC_EXPECT=mesh
exec "$(dirname "$(readlink -f "$0")")/start.sh" "$@"
