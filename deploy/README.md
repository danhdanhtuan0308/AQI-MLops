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
| GRAFANA_OTLP_ENDPOINT | OTLP gateway URL (e.g. https://otlp-gateway-prod-us-east-2.grafana.net/otlp) |
| GRAFANA_OTLP_INSTANCE_ID | Grafana Cloud instance ID for OTLP Basic auth |
| GRAFANA_API_KEY | Grafana Cloud API key (used for metrics, traces, and logs) |
| GRAFANA_LOKI_URL | Loki push URL (e.g. https://logs-prod-036.grafana.net/loki/api/v1/push) |
| GRAFANA_LOKI_USER | Loki instance user ID |
| OTEL_SERVICE_NAME | OpenTelemetry service name tag (default: aqi-api) |
| OTEL_SERVICE_VERSION | OpenTelemetry service version tag (default: 1.0) |

The REDIS_URL is critical. Without it, Redis caching is disabled and every dashboard request hits Athena directly.

The GRAFANA_* variables enable observability. If any are missing, the corresponding signal (metrics, traces, or logs) is silently disabled and the service runs without it.

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
