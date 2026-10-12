#!/usr/bin/env bash
# Serve GLM-5.3 Flash EXL3 on three DGX Sparks (TP=3, experimental): ./start.sh with TP=3, COMM=nccl by default.
# Needs the TP-N GLM engine (patches 0066-0068; prepare.sh builds the image on the first start), WORKER and WORKER2 in
# scripts/local.sh, and a triangle of direct CX7 cables, one per pair (one subnet per cable).
# README: "3 Sparks (experimental)".
#
# Usage: ./start-tp3.sh [restart] [extra tensorfold serve args]   (as ./start.sh; ./stop.sh stops it)
#   DRY_RUN=1 ./start-tp3.sh            # print every rank's docker command, change nothing
# COMM defaults to nccl here (NCCL for every all-gather); COMM=roce ./start-tp3.sh sends the small ones over RoCE,
# each peer on the devices that share its subnet (needs the TP-N engine's RoCE). COMM in scripts/local.sh or .env is
# not read here: set it on the command line.
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; echo; echo "./start.sh's options:"; echo ;;
esac
export TP=3 COMM="${COMM:-nccl}"
exec "$(dirname "$(readlink -f "$0")")/start.sh" "$@"
