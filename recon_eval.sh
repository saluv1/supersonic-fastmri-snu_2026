set -euo pipefail
cd "$(dirname "$0")"
 
DATA_ROOT="${1:-../Data/leaderboard}"
NET_NAME="${2:-promptmr8_metric_aligned_50ep_v1}"
 
python3 recon_eval.py \
  -n "$NET_NAME" \
  -p "$DATA_ROOT"
