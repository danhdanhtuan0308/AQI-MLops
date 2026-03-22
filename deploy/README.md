# deploy/ — EC2 Deployment Files

Infrastructure-as-code for running the AQI API on an AWS EC2 t4g.small (ARM64 Graviton, Ubuntu 24.04). Includes a one-time bootstrap script, a systemd service unit, and an nginx reverse-proxy config.

---

## Directory Structure

```
deploy/
├── setup_ec2.sh      # One-time bootstrap: install deps, clone repo, configure nginx + systemd
├── aqi-api.service   # systemd unit — runs uvicorn, auto-restarts on failure
└── nginx.conf        # nginx reverse proxy: port 80 → 127.0.0.1:8000
```

---

## Instance Details

| Property | Value |
|---|---|
| Instance ID | `i-028aa0d0c1c305362` |
| Instance type | `t4g.small` (2 vCPU, 2 GB RAM, ARM64 Graviton2) |
| AMI | `ami-0bc0f64eea5d47edf` (Ubuntu 24.04 LTS ARM64, 2026-03-21) |
| Region | `us-east-1` |
| Public IP | `3.94.115.44` |
| Public DNS | `ec2-3-94-115-44.compute-1.amazonaws.com` |
| Security group | `sg-0ff863b76ab5d0102` — ports 22 (SSH) and 80 (HTTP) open |
| Key pair | `aqi-mlops-key` |

**Access URLs:**
- `http://3.94.115.44/` — Dashboard
- `http://3.94.115.44/health` — Health check
- `http://3.94.115.44/metrics/{city_slug}` — Live F1/Precision/Recall

---

## setup_ec2.sh

One-time bootstrap script. Run once after launching a fresh EC2 instance.

**What it does (in order):**
1. `apt-get` installs: `git`, `curl`, `nginx`, `build-essential`, `libssl-dev`
2. Installs `uv` (ARM64-native Python package manager) to `~/.local/bin`
3. Clones `danhdanhtuan0308/AQI-MLops` from GitHub (or pulls if already cloned)
4. Runs `uv sync` to install all Python dependencies into `.venv`
5. Copies `.env.example` → `.env` if `.env` doesn't exist yet
6. Installs nginx config: copies `deploy/nginx.conf` → `/etc/nginx/sites-available/aqi-api`, symlinks into `sites-enabled`, removes default site, tests and restarts nginx
7. Installs systemd service: copies `deploy/aqi-api.service`, substitutes `__USER__` and `__APP_DIR__` tokens, runs `systemctl daemon-reload` and `systemctl enable aqi-api`

**After the script:** fill in `.env` with real credentials, then `sudo systemctl start aqi-api`.

```bash
# Run on a fresh instance:
chmod +x setup_ec2.sh && ./setup_ec2.sh
```

**The script does NOT start the service** — this is intentional so you have time to fill in `.env` first.

---

## aqi-api.service

systemd unit file for the FastAPI application.

**Key settings:**
- Runs: `uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1`
- Loads environment from: `__APP_DIR__/.env` (substituted to `/home/ubuntu/AQI-MLops/.env`)
- `Restart=on-failure`, `RestartSec=10s` — auto-restarts if the process crashes
- `After=network.target` — waits for network before starting

The service listens on `127.0.0.1:8000` (localhost only). nginx forwards port 80 traffic to it.

**Useful commands:**
```bash
sudo systemctl status aqi-api
sudo systemctl restart aqi-api
sudo journalctl -u aqi-api -f       # tail logs
```

---

## nginx.conf

Reverse proxy configuration. Listens on port 80 and forwards all requests to the uvicorn process at `127.0.0.1:8000`.

**Security headers included:**
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`

`proxy_read_timeout 120s` — allows Athena queries (cold start) up to 2 minutes.

---

## CD Pipeline Integration

The `cd.yml` workflow SSH-deploys to this instance on every push to `main` that touches `app/**`, `ml/model-registry/**`, `pyproject.toml`, or `uv.lock`. It:
1. Runs the CI gate (`ci.yml`) first — deploy is blocked if tests fail
2. SSH: `git pull origin main`
3. SSH: `uv sync`
4. SSH: `sudo systemctl restart aqi-api`
5. SSH: `curl http://localhost:8000/health` — fails the workflow if the service doesn't respond

Required GitHub Secrets: `EC2_SSH_KEY`, `EC2_HOST`, `EC2_USER`.

---

## Manual Deployment

```bash
# Copy credentials
scp -i ~/.ssh/aqi-mlops-key.pem .env ubuntu@3.94.115.44:~/AQI-MLops/.env

# Copy model artifacts (first time)
scp -i ~/.ssh/aqi-mlops-key.pem ml/model-registry/* ubuntu@3.94.115.44:~/AQI-MLops/ml/model-registry/

# Restart the service
ssh -i ~/.ssh/aqi-mlops-key.pem ubuntu@3.94.115.44 "sudo systemctl restart aqi-api"

# Verify
curl http://3.94.115.44/health
```
