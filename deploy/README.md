# deploy/ — EC2 Deployment Files

Infrastructure files for running the AQI API on an AWS EC2 t4g.small (ARM64 Graviton2, Ubuntu 24.04). The service runs inside Docker and is managed by a systemd unit that wraps docker compose. All HTTP traffic goes through nginx which forwards requests from port 80 to the container on port 8000.

---

## Directory Structure

```
deploy/
  setup_ec2.sh      One-time bootstrap: installs Docker, clones repo, configures nginx and systemd
  aqi-api.service   systemd unit file that manages the docker compose lifecycle
  nginx.conf        nginx reverse proxy: port 80 to 127.0.0.1:8000
```

---

## Instance Details

| Property | Value |
|----------|-------|
| Instance ID | i-028aa0d0c1c305362 |
| Instance type | t4g.small (2 vCPU, 2 GB RAM, ARM64 Graviton2) |
| AMI | ami-0bc0f64eea5d47edf (Ubuntu 24.04 LTS ARM64) |
| Region | us-east-1 |
| Public IP | 3.94.115.44 |
| Security group | sg-0ff863b76ab5d0102, ports 22 (SSH) and 80 (HTTP) open |
| Key pair | aqi-mlops-key |

Access URLs:
- http://3.94.115.44/ Dashboard
- http://3.94.115.44/health Health check
- http://3.94.115.44/cache/status Redis cache status

---

## setup_ec2.sh

One-time bootstrap script. Run once after launching a fresh EC2 instance.

What it does in order:
1. Installs git, curl, and nginx via apt-get
2. Installs Docker Engine using the official get.docker.com convenience script, which includes the docker compose plugin
3. Adds the ubuntu user to the docker group
4. Clones the AQI-MLops repository from GitHub (or hard-resets if already present)
5. Copies .env.example to .env if .env does not exist yet
6. Installs the nginx configuration: copies deploy/nginx.conf to /etc/nginx/sites-available/aqi-api, creates a symlink in sites-enabled, removes the default site, tests and restarts nginx
7. Installs the systemd service: copies deploy/aqi-api.service, substitutes __USER__ and __APP_DIR__ tokens, then runs systemctl daemon-reload and systemctl enable aqi-api

The script does not build the Docker image or start the service. Fill in .env first.

```bash
chmod +x setup_ec2.sh && ./setup_ec2.sh
```

After the script completes:

```bash
nano ~/AQI-MLops/.env           # fill in credentials
newgrp docker                   # activate docker group in current shell
cd ~/AQI-MLops && docker compose build
sudo systemctl start aqi-api
curl http://localhost:8000/health
```

---

## Environment Variables (.env)

The .env file on EC2 must contain these variables:

| Variable | Description |
|----------|-------------|
| OWM_API_KEY | OpenWeatherMap API key for Lambda A (not used by FastAPI directly) |
| AWS_ACCESS_KEY_ID | AWS credentials for Athena queries |
| AWS_SECRET_ACCESS_KEY | AWS credentials for Athena queries |
| AWS_DEFAULT_REGION | AWS region, typically us-east-1 |
| REDIS_URL | Redis Cloud connection string: redis://default:...@host:port |

The REDIS_URL is the most important addition. Without it, Redis caching is disabled and every dashboard request hits Athena directly.

---

## aqi-api.service

systemd unit file that manages the Docker Compose lifecycle for the FastAPI container.

Key settings:
- ExecStart runs docker compose up to start the container defined in docker-compose.yml
- ExecStop runs docker compose down for a graceful shutdown
- After and Requires ensure it waits for the Docker daemon to be ready
- Restart=on-failure with RestartSec=10s auto-restarts the service if it exits unexpectedly
- WorkingDirectory is set to the repo root

The container binds to 127.0.0.1:8000 (localhost only). nginx forwards port 80 traffic to it.

Useful commands:

```bash
sudo systemctl status aqi-api
sudo systemctl restart aqi-api
sudo journalctl -u aqi-api -f
docker compose logs -f
```

---

## nginx.conf

Reverse proxy configuration. Listens on port 80 and forwards all requests to the Docker container at 127.0.0.1:8000.

Security headers included:
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block

---

## CD Pipeline

The cd.yml GitHub Actions workflow deploys automatically when code is pushed to main and touches the app/ or ml/ directories. It:

1. SSHes into EC2
2. Runs git pull to fetch the latest code
3. Runs docker compose build to rebuild the image
4. Runs docker compose up -d --force-recreate --remove-orphans to restart the container cleanly
5. Waits for /health to return HTTP 200

The --force-recreate flag ensures the new container always replaces the old one even if the compose config did not change.

---

## Directory Structure

```
deploy/
├── setup_ec2.sh      # One-time bootstrap: install Docker, clone repo, configure nginx + systemd
├── aqi-api.service   # systemd unit — manages docker compose lifecycle, auto-restarts on failure
└── nginx.conf        # nginx reverse proxy: port 80 → 127.0.0.1:8000 (Docker container)
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
1. `apt-get` installs: `git`, `curl`, `nginx`
2. Installs Docker Engine via the official convenience script (`get.docker.com`) — includes `docker compose` plugin
3. Adds `ubuntu` to the `docker` group (`sudo usermod -aG docker ubuntu`)
4. Clones `danhdanhtuan0308/AQI-MLops` from GitHub (or hard-resets if already cloned)
5. Copies `.env.example` → `.env` if `.env` doesn't exist yet
6. Installs nginx config: copies `deploy/nginx.conf` → `/etc/nginx/sites-available/aqi-api`, symlinks into `sites-enabled`, removes default site, tests and restarts nginx
7. Installs systemd service: copies `deploy/aqi-api.service`, substitutes `__USER__` and `__APP_DIR__` tokens, runs `systemctl daemon-reload` and `systemctl enable aqi-api`

**After the script:**
```bash
nano ~/AQI-MLops/.env              # fill in OWM_API_KEY, AWS credentials
newgrp docker                      # activate docker group in current shell
cd ~/AQI-MLops && docker compose build
sudo systemctl start aqi-api
curl http://localhost:8000/health
```

**The script does NOT build the image or start the service** — fill in `.env` first.

```bash
# Run on a fresh instance:
chmod +x setup_ec2.sh && ./setup_ec2.sh
```

---

## aqi-api.service

systemd unit file — manages the Docker Compose lifecycle for the FastAPI container.

**Key settings:**
- `ExecStart=/usr/bin/docker compose up` — starts the container defined in `docker-compose.yml`
- `ExecStop=/usr/bin/docker compose down` — graceful shutdown
- `After=network.target docker.service`, `Requires=docker.service` — waits for Docker daemon
- `Restart=on-failure`, `RestartSec=10s` — auto-restarts if compose exits unexpectedly
- `WorkingDirectory` is set to the repo root (substituted from `__APP_DIR__`)

The container binds port `127.0.0.1:8000` (localhost only). nginx forwards port 80 traffic to it.

**Useful commands:**
```bash
sudo systemctl status aqi-api
sudo systemctl restart aqi-api
sudo journalctl -u aqi-api -f      # tail systemd logs
docker compose logs -f             # tail container logs directly
```

---

## nginx.conf

Reverse proxy configuration. Listens on port 80 and forwards all requests to the Docker container at `127.0.0.1:8000`.

**Security headers included:**
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`

`proxy_read_timeout 120s` — allows Athena queries (cold start) up to 2 minutes.

---

## CD Pipeline Integration

The `cd.yml` workflow SSH-deploys to this instance on every push to `main` that touches `app/**`, `ml/model-registry/**`, `pyproject.toml`, or `uv.lock`. It:
1. Runs the CI gate (`ci.yml`) first — deploy is blocked if tests fail
2. SSH: `git fetch origin main && git reset --hard origin/main && git clean -fd` — hard reset (never blocked by local changes)
3. SSH: `docker compose build` — rebuilds the image from the updated code
4. SSH: `docker compose up -d` — replaces running container with the new image
5. SSH: `curl http://localhost:8000/health` — fails the workflow if the service doesn't respond

Required GitHub Secrets: `EC2_SSH_KEY`, `EC2_HOST`, `EC2_USER`.

---

## Manual Deployment

```bash
# Copy credentials
scp -i ~/.ssh/aqi-mlops-key.pem .env ubuntu@3.94.115.44:~/AQI-MLops/.env

# Copy model artifacts (first time)
scp -i ~/.ssh/aqi-mlops-key.pem ml/model-registry/* ubuntu@3.94.115.44:~/AQI-MLops/ml/model-registry/

# Restart the container
ssh -i ~/.ssh/aqi-mlops-key.pem ubuntu@3.94.115.44 "cd ~/AQI-MLops && docker compose up -d"

# Verify
curl http://3.94.115.44/health
```
