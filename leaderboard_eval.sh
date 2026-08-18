#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
  pwd
)"
cd "$SCRIPT_DIR"

if [[ $# -ge 1 ]]; then
  LEADERBOARD_ROOT="$1"
else
  LEADERBOARD_ROOT="$SCRIPT_DIR/../Data/leaderboard"
fi

NET_NAME="${2:-promptmr8_metric_aligned_50ep_v1}"

if [[ $# -ge 3 ]]; then
  RECON_ROOT="$3"
else
  RECON_ROOT="$SCRIPT_DIR/../result/$NET_NAME/reconstructions_leaderboard"
fi

python3 "$SCRIPT_DIR/leaderboard_eval.py" \
  -lp "$LEADERBOARD_ROOT" \
  -yp "$RECON_ROOT"
