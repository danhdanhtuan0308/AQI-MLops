# AQI-MLops

End-to-end MLOps pipeline for real-time Air Quality Index (AQI) prediction across 99 global cities.

**Live dashboard:** http://3.94.115.44

---

## What it does

- **Fetches** live AQI + pollutant data from OpenWeatherMap every hour (AWS Lambda)
- **Stores** raw data in S3, merges into a unified Apache Iceberg table via Athena
- **Trains** an XGBoost classifier (multi:softprob, 5 AQI classes) on the full 12-month history
- **Serves** next-hour AQI predictions via a FastAPI app running inside Docker on EC2
- **Monitors** drift weekly (pollutant z-scores, AQI class distribution shift)
- **Retrains** every Monday automatically via GitHub Actions

---


## Repository Structure

```
├── app/                    # FastAPI prediction service
│   ├── main.py             # All routes (/predict, /history, /drift, /metrics, /health)
│   ├── inference.py        # Feature engineering + XGBoost prediction helpers
│   └── templates/          # Single-page dashboard (Alpine.js + Chart.js)
│
├── data-pipeline/          # Hourly live data ingestion
│   ├── lambda/handler.py   # AWS Lambda: fetch AQI → upload to S3
│   ├── fetch_weather.py    # Historical backfill (one-off)
│   └── config.yaml         # City list (99 cities), S3 bucket, API URL config
│
├── lakehouse/              # S3 → Iceberg (Bronze → Silver)
│   ├── merge_handler.py    # Athena MERGE / INSERT for deduplication
│   └── setup.sql           # DDL: create Iceberg table in Glue Data Catalog
│
├── ml/                     # Model training + registry
│   ├── train.py            # XGBoost pipeline (Athena → features → train → MLflow)
│   ├── best-config.yaml    # Winning hyperparameters from random/bayes search
│   ├── retrain_weekly.sh   # Shell wrapper for weekly retrains + hot-reload
│   └── model-registry/     # model.ubj, median.json, features.json (gitignored)
│
├── tests/                  # 52 pytest tests (no AWS, Athena fully mocked)
│
├── deploy/                 # EC2 infrastructure files
│   ├── setup_ec2.sh        # One-time bootstrap: Docker, nginx, git clone
│   ├── aqi-api.service     # systemd unit: manages docker compose lifecycle
│   └── nginx.conf          # Reverse proxy: port 80 → 127.0.0.1:8000
│
├── Dockerfile              # Multi-stage build (python:3.12-slim + uv, non-root)
├── docker-compose.yml      # Production compose: model-registry volume, restart policy
├── .dockerignore           # Excludes .env, *.pem, tests/, deploy/, ml/mlruns/
└── .github/workflows/      # CI (lint+test), CD (Docker deploy), weekly retrain
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Interactive dashboard |
| `GET` | `/cities` | All 99 cities with coordinates + timezone |
| `GET` | `/predict/{city_slug}` | Next-hour AQI prediction + per-class probabilities |
| `GET` | `/history/{city_slug}` | Predicted vs actual AQI (up to 7 days) |
| `GET` | `/drift/{city_slug}` | Feature drift z-scores (current week vs prior week) |
| `GET` | `/metrics/{city_slug}` | Live weighted F1 / Precision / Recall from Athena data |
| `GET` | `/health` | Service health + model metadata |
| `POST` | `/reload-model` | Hot-swap model artifacts without restarting the container |

---

## Local Development

```bash
# Clone and install
git clone https://github.com/danhdanhtuan0308/AQI-MLops.git
cd AQI-MLops
uv sync --group dev

# Copy and fill in credentials
cp .env.example .env
# edit .env: OWM_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

# Run locally (without Docker)
uv run uvicorn app.main:app --reload

# Or with Docker
docker compose up --build
```

### Run tests
```bash
uv run ruff check app/ ml/ tests/   # lint
uv run pytest tests/ -v             # 52 tests, no AWS needed
```

---

## Model

- **Algorithm:** XGBoost `multi:softprob`, 5 AQI classes (Good / Fair / Moderate / Poor / Very Poor)
- **Features (12):** `aqi_lag1`, `pm10_lag1`, `hour_sin`, `hour_cos`, `pm25_ratio`, `co`, `no`, `no2`, `o3`, `so2`, `nh3`, `pm10`
- **Target:** AQI class at T+1 (next hour)
- **Training data:** 12-month history from `aqi_db.aqi_unified` (Iceberg on S3)
- **Class weighting:** `compute_sample_weight("balanced")` — prevents bias toward majority class (Fair/Good)
- **Retrain cadence:** every Monday, last 8 weeks of data, auto hot-reload via `/reload-model`

---

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR | ruff lint + 52 pytest tests |
| `cd.yml` | push to `main` (app/ or ml/ changes) | `git reset --hard` → `docker compose build` → `docker compose up -d` → health check |
| `weekly_retrain.yml` | Every Monday 02:00 UTC | CI gate → retrain on Athena → smoke test → hot-reload |

---

## Infrastructure

| Resource | Value |
|---|---|
| EC2 | `t4g.small` — 2 vCPU, 2 GB RAM, ARM64 Graviton2, `us-east-1` |
| Public IP | `3.94.115.44` |
| Runtime | Docker (python:3.12-slim), nginx reverse proxy |
| Data store | S3 (`weather-bulk`) + Athena + Apache Iceberg |
| Model artifacts | `ml/model-registry/` — bind-mounted into Docker container (not baked into image) |
