#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# retrain_daily.sh — Daily retrain for the AQI forecasting model
#
# Run manually or via cron / GitHub Actions every day at 05:00 UTC.
# Usage:
#   ./ml/retrain_daily.sh              # default: 8-week lookback
#   ./ml/retrain_daily.sh 4            # custom lookback in weeks
#   ./ml/retrain_daily.sh --no-reload  # skip restart reminder
#
# Cron example (every day at 05:00 UTC):
#   0 5 * * * cd /path/to/repo && ./ml/retrain_daily.sh >> logs/retrain.log 2>&1
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
LOOKBACK=${1:-52}           # weeks of training data (default 52 weeks = 1 year rolling)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/retrain_$(date -u +%Y%m%d_%H%M%S).log"

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# Activate virtual environment
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
else
  echo "ERROR: No .venv found. Create one with: python -m venv .venv && pip install -e ." >&2
  exit 1
fi

# ── Run ──────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  AQI Daily Retrain" | tee -a "$LOG_FILE"
echo "  Started : $TIMESTAMP" | tee -a "$LOG_FILE"
echo "  Lookback: last $LOOKBACK weeks" | tee -a "$LOG_FILE"
echo "  Log     : $LOG_FILE" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

python ml/train.py --lookback-weeks "$LOOKBACK" 2>&1 | tee -a "$LOG_FILE"

FINISH=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
echo "" | tee -a "$LOG_FILE"
echo "✓ Retrain complete — $FINISH" | tee -a "$LOG_FILE"
echo "  New model saved to: ml/model-registry/model.ubj" | tee -a "$LOG_FILE"

# ── Hot-reload the running API (if it's up) ───────────────────────────────
API_URL="${API_URL:-http://localhost:8000}"
echo "" | tee -a "$LOG_FILE"
echo "  Attempting hot-reload at $API_URL/reload-model ..." | tee -a "$LOG_FILE"
if curl -sf -X POST "$API_URL/reload-model" -o /dev/null; then
  echo "  ✓ API model reloaded — no restart needed" | tee -a "$LOG_FILE"
else
  echo "  ⚠ API not reachable. Restart manually:" | tee -a "$LOG_FILE"
  echo "    uvicorn app.main:app --reload" | tee -a "$LOG_FILE"
fi
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
