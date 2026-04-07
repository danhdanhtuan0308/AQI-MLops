#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# retrain_daily.sh — Daily retrain for the AQI forecasting model
#
# Run manually or via cron / GitHub Actions every day at 05:00 UTC.
# Rolling window: trains on the most recent 365 days (1 full year).
# Each daily run shifts the window forward by 1 day automatically because
# train.py uses current_timestamp in the Athena query at runtime.
#   e.g. run on 03/21/2026 → trains on 03/21/2025–03/21/2026
#
# Usage:
#   ./ml/retrain_daily.sh              # default: 365 days (1 year rolling)
#   ./ml/retrain_daily.sh 180          # custom lookback in days
#   ./ml/retrain_daily.sh --no-reload  # skip restart reminder
#
# Cron example (every day at 05:00 UTC):
#   0 5 * * * cd /path/to/repo && ./ml/retrain_daily.sh >> logs/retrain.log 2>&1
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
LOOKBACK=${1:-365}          # days of training data (default 365 days = 1 full year rolling)
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
echo "  Lookback: last $LOOKBACK days (rolling 1-year window)" | tee -a "$LOG_FILE"
echo "  Log     : $LOG_FILE" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

python ml/train.py --lookback-days "$LOOKBACK" 2>&1 | tee -a "$LOG_FILE"

FINISH=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
echo "" | tee -a "$LOG_FILE"
echo "✓ Retrain complete — $FINISH" | tee -a "$LOG_FILE"
echo "  New model saved to: ml/model-registry/model.ubj" | tee -a "$LOG_FILE"

# ── Hot-reload the running API (if it's up) ───────────────────────────────
# Restart the container so uvicorn picks up any code changes from the volume mount,
# then call /reload-model to swap in the new model artifacts and re-warm the cache.
API_URL="${API_URL:-http://localhost:8000}"
echo "" | tee -a "$LOG_FILE"
echo "  Restarting API container to load new code + model ..." | tee -a "$LOG_FILE"
if docker compose -f "$REPO_ROOT/docker-compose.yml" restart api 2>/dev/null; then
  echo "  ✓ Container restarted" | tee -a "$LOG_FILE"
  # Wait for the API to be healthy before calling reload-model
  for i in $(seq 1 12); do
    if curl -sf "$API_URL/health" -o /dev/null; then
      break
    fi
    sleep 5
  done
  if curl -sf -X POST "$API_URL/reload-model" -o /dev/null; then
    echo "  ✓ Model reloaded — cache re-warming in background" | tee -a "$LOG_FILE"
  else
    echo "  ⚠ /reload-model failed after restart — check API logs" | tee -a "$LOG_FILE"
  fi
else
  echo "  ⚠ Docker not available / container not running. Restart manually:" | tee -a "$LOG_FILE"
  echo "    cd $REPO_ROOT && docker compose restart api" | tee -a "$LOG_FILE"
fi
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
