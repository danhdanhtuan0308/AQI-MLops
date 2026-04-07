#!/usr/bin/env bash
# =============================================================================
# deploy/redeploy.sh — Pull latest code and restart the API on EC2
#
# Run this on the EC2 instance after pushing new code to GitHub:
#   bash ~/AQI-MLops/deploy/redeploy.sh
#
# What it does:
#   1. git pull origin main  (gets new app/, ml/, deploy/ etc.)
#   2. docker compose restart api  (restarts uvicorn so new app/ code is loaded)
#      - No rebuild needed: app/ is bind-mounted into the container
#   3. POST /reload-model  (hot-swaps model artifacts in memory + re-warms cache)
#
# For model-only retrains: retrain_daily.sh handles this automatically.
# =============================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
API_URL="${API_URL:-http://localhost:8000}"

echo "════════════════════════════════════════════════════════"
echo "  AQI-MLops Redeploy  —  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "  Dir: $APP_DIR"
echo "════════════════════════════════════════════════════════"

cd "$APP_DIR"

# ── 1. Pull latest code ───────────────────────────────────────────────────────
echo ""
echo "── Pulling latest code from GitHub ─────────────────────"
git fetch origin main
git reset --hard origin/main
echo "   ✓ Now at $(git log --oneline -1)"

# ── 2. Restart the API container (picks up new app/ code via volume mount) ───
echo ""
echo "── Restarting API container ─────────────────────────────"
docker compose restart api
echo "   ✓ Container restarted"

# ── 3. Wait for the API to become healthy ─────────────────────────────────────
echo ""
echo "── Waiting for API to be healthy ────────────────────────"
for i in $(seq 1 12); do
    if curl -sf "$API_URL/health" -o /dev/null; then
        echo "   ✓ API is up"
        break
    fi
    if [ "$i" -eq 12 ]; then
        echo "   ✗ API did not come up within 60s — check: docker compose logs api"
        exit 1
    fi
    echo "   ... waiting (${i}/12)"
    sleep 5
done

# ── 4. Hot-reload model artifacts (re-caches feature importance + predictions) ─
echo ""
echo "── Hot-reloading model artifacts ────────────────────────"
RESPONSE=$(curl -sf -X POST "$API_URL/reload-model" 2>&1 || echo "FAILED")
if echo "$RESPONSE" | grep -q '"ok"'; then
    echo "   ✓ Model reloaded — cache re-warming in background"
else
    echo "   ⚠ /reload-model call failed or API not reachable: $RESPONSE"
fi

# ── 5. Verify new feature count ───────────────────────────────────────────────
echo ""
echo "── Verifying deployment ─────────────────────────────────"
N_FEATS=$(curl -sf "$API_URL/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('n_features','?'))" 2>/dev/null || echo "?")
echo "   n_features : $N_FEATS  (expect 20)"
echo "   model path : $(curl -sf "$API_URL/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_path','?'))" 2>/dev/null || echo "?")"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Redeploy complete!"
echo "════════════════════════════════════════════════════════"
