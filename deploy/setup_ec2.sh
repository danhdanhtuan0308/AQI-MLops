#!/usr/bin/env bash
# =============================================================================
# deploy/setup_ec2.sh
# One-time bootstrap for AQI-MLops on Ubuntu 24.04 ARM64 (t4g.small / Graviton)
#
# Run once after launching a fresh EC2 instance:
#   chmod +x setup_ec2.sh && ./setup_ec2.sh
#
# After this script completes:
#   1. Edit ~/AQI-MLops/.env with your real credentials
#   2. sudo systemctl start aqi-api
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/danhdanhtuan0308/AQI-MLops.git"
APP_DIR="$HOME/AQI-MLops"

echo "════════════════════════════════════════════════════════"
echo "  AQI-MLops EC2 Bootstrap  —  $(uname -m)  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "════════════════════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "── Installing system packages ───────────────────────────"
sudo apt-get update -y -qq
sudo apt-get install -y -qq git curl nginx

# ── 2. Docker ─────────────────────────────────────────────────────────────────
echo "── Installing Docker ────────────────────────────────────"
sudo apt-get install -y -qq docker.io docker-compose-plugin
sudo systemctl enable docker
sudo systemctl start docker
# Allow ubuntu user to run docker without sudo
sudo usermod -aG docker "$USER"
echo "   Docker $(docker --version) installed"

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
echo "── Cloning repository ───────────────────────────────────"
if [ -d "$APP_DIR/.git" ]; then
    echo "   Repo already exists — hard-resetting to main"
    git -C "$APP_DIR" fetch origin main
    git -C "$APP_DIR" reset --hard origin/main
    git -C "$APP_DIR" clean -fd
else
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# ── 4. .env file ──────────────────────────────────────────────────────────────
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo ""
    echo "  ⚠  IMPORTANT — fill in real credentials before starting:"
    echo "     nano $APP_DIR/.env"
    echo ""
fi

# ── 5. nginx reverse proxy ────────────────────────────────────────────────────
echo "── Configuring nginx ────────────────────────────────────"
sudo cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/aqi-api
sudo ln -sf /etc/nginx/sites-available/aqi-api /etc/nginx/sites-enabled/aqi-api
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

# ── 6. systemd service (manages docker compose lifecycle) ────────────────────
echo "── Installing systemd service ───────────────────────────"
sudo cp "$APP_DIR/deploy/aqi-api.service" /etc/systemd/system/aqi-api.service
sudo sed -i "s|__USER__|$USER|g"       /etc/systemd/system/aqi-api.service
sudo sed -i "s|__APP_DIR__|$APP_DIR|g" /etc/systemd/system/aqi-api.service
sudo systemctl daemon-reload
sudo systemctl enable aqi-api

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. nano $APP_DIR/.env          ← add OWM_API_KEY, AWS credentials"
echo "    2. newgrp docker               ← activate docker group in current shell"
echo "    3. cd $APP_DIR && docker compose build"
echo "    4. sudo systemctl start aqi-api"
echo "    5. curl http://localhost:8000/health"
echo ""
echo "  Daily redeploy (after git push from laptop):"
echo "    bash $APP_DIR/deploy/redeploy.sh"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status aqi-api"
echo "    sudo journalctl -u aqi-api -f"
echo "    docker compose logs -f"
echo "    sudo systemctl status nginx"
echo "════════════════════════════════════════════════════════"
