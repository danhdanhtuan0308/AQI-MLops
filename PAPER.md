# AQI-MLops: An End-to-End Production MLOps Pipeline for Real-Time Air Quality Index Prediction Across 99 Global Cities

**Daniel Lai**  
Data Science Capstone Project — Team 1  
Drexel University, College of Computing & Informatics  
May 2026

**GitHub Repository:** [https://github.com/danhdanhtuan0308/AQI-MLops](https://github.com/danhdanhtuan0308/AQI-MLops)  
**Live Dashboard:** [http://3.94.115.44](http://3.94.115.44)

---

## Abstract

Air quality affects the health of billions of people, yet accessible real-time forecasts at global scale remain scarce. This paper presents **AQI-MLops**, a fully automated, cloud-native machine learning operations (MLOps) system that collects live air quality sensor readings every hour for 99 cities worldwide, stores them in a medallion data lakehouse on AWS, predicts the next-hour Air Quality Index (AQI) class using a tuned XGBoost classifier, and serves those predictions through a web dashboard with sub-millisecond response times. The system achieves a cross-validated macro F1 score of **0.9425** and a validation macro F1 score of **0.9535** across five AQI classes. The full pipeline — from raw API ingest to model retraining and observability — runs continuously and autonomously using AWS Lambda, S3, Athena (Apache Iceberg), Redis, FastAPI, Docker, and Grafana Cloud, with CI/CD orchestrated by GitHub Actions.

---

## 1. Introduction

The World Health Organization estimates that 99% of the global population breathes air that exceeds recommended quality limits [1]. Timely, city-level AQI forecasts can drive behavioral decisions — whether to exercise outdoors, whether to issue public health alerts, or whether to activate industrial emission controls. Despite the importance of this information, most publicly available AQI data is either delayed batch reports or single-city solutions that do not generalize.

This work addresses the challenge by building a production-grade MLOps system with three primary properties:

1. **Scale**: 99 geographically diverse cities spanning Asia, Europe, the Americas, and Africa.
2. **Freshness**: Hourly ingest, hourly prediction refresh, and automated daily model retraining.
3. **Reliability**: ACID-consistent data storage, Redis-backed sub-millisecond serving, self-healing cache warming, and a full observability stack.

The system is implemented in Python and deployed entirely on AWS, with all infrastructure described as configuration rather than hand-operated scripts. The model, data pipeline, serving layer, and observability components are released as open source at [https://github.com/danhdanhtuan0308/AQI-MLops](https://github.com/danhdanhtuan0308/AQI-MLops).

---

## 2. Related Work

Traditional air quality forecasting relies on numerical weather prediction models (e.g., CMAQ, WRF-Chem) that are computationally expensive and require domain expertise to configure [2]. Data-driven approaches using gradient-boosted trees and recurrent neural networks have demonstrated competitive accuracy on city-level datasets [3, 4], but published systems rarely describe the end-to-end engineering infrastructure required to keep such models fresh and serving at production quality.

MLOps as a discipline formalizes the practices needed to operationalize ML models [5]. Systems such as Uber Michelangelo [6] and Meta FBLearner Flow [7] demonstrate industrial-scale MLOps but are not open-source and operate at a scale impractical for academic projects. AQI-MLops occupies a middle ground: a real production system with real traffic, open-source code, and a total infrastructure cost within a research budget.

---

## 3. System Architecture

The system is composed of five sequential pipeline stages executed across three compute substrates (AWS Lambda, AWS EC2, and GitHub Actions). Figure 1 provides the full architecture diagram.

```
OpenWeatherMap API
        │
        ├─── fetch_weather.py (bulk backfill, 1 year × 99 cities)
        │         │
        │         └──► S3 Bronze: raw_pipeline (Parquet/SNAPPY)
        │
        └─── Lambda A (hourly EventBridge cron)
                  │
                  └──► S3 Bronze: raw_hourly (Parquet, Hive-partitioned)
                            │
                        Lambda B: MERGE INTO aqi_unified (Iceberg, ACID)
                            │
                        Silver: aqi_db.aqi_unified
                            │
                        FastAPI /warm-cache
                            │
                        XGBoost inference (19 features) ──► Redis (TTL 2h)
                            │
                        Lambda C (hourly + 5 min offset)
                            │
                        Gold: predictions Iceberg table
                            │
                        GitHub Actions (daily)
                            │
                        ml/train.py ──► MLflow ──► model-registry/model.ubj
```

### 3.1 Data Ingestion

**Batch Backfill.** The script `data-pipeline/fetch_weather.py` seeds the lakehouse with one year of historical data (2025-03-20 → 2026-03-20) for all 99 cities. The date range is split into seven-day chunks per city to respect the OpenWeatherMap (OWM) free-tier pagination limit. A 0.5-second inter-request sleep bounds throughput to ≈2 req/s, comfortably within OWM free-tier rate limits. Each chunk is parsed into a Pandas DataFrame, serialized as SNAPPY-compressed Parquet using PyArrow, and uploaded directly to `s3://weather-bulk/data-pipeline/` via the AWS SDK.

**Live Hourly Ingest (Lambda A).** An AWS EventBridge rule fires Lambda A every hour. Lambda A calls the OWM Current Air Pollution API for all 99 cities in sequence and writes a single Parquet file to a Hive-partitioned path: `s3://weather-bulk/hourly/year=YYYY/month=MM/day=DD/`. On success, Lambda A invokes Lambda B asynchronously, passing the S3 key of the new file.

The raw data schema captures eleven air quality measurements alongside city metadata:

| Column | Type | Description |
|---|---|---|
| `timestamp` | TIMESTAMP (UTC) | Observation time |
| `city` / `city_slug` | STRING | City name and URL-safe slug |
| `aqi` | INT (1–5) | Overall AQI class |
| `co`, `no`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`, `nh3` | DOUBLE (µg/m³) | Pollutant concentrations |

### 3.2 Data Lakehouse (Medallion Architecture)

The storage layer is built on AWS S3 with Apache Iceberg tables registered in AWS Glue Data Catalog and queried via AWS Athena.

- **Bronze Layer.** Two read-only external tables: `aqi_db.raw_pipeline` (bulk batch) and `aqi_db.raw_hourly` (live hourly, Hive-partitioned).
- **Silver Layer.** Lambda B executes an Athena `MERGE INTO aqi_db.aqi_unified` statement keyed on `(city_slug, timestamp)`. This provides ACID-safe deduplication and ensures idempotency across overlapping pipeline runs. A `source` column (`'pipeline'` or `'live'`) is written for data lineage. The Iceberg table format enables time-travel queries and schema evolution without table rewrites.
- **Gold Layer.** Aggregated analytics tables stored at `s3://weather-bulk/process/BI/`, used by Grafana for business-intelligence dashboards.

### 3.3 Inference and Serving

When Lambda B completes the merge, it sends a `POST /warm-cache` request to the FastAPI service running on EC2. On receiving this call, FastAPI:

1. Queries Athena for the last 340 hours of sensor data across all 99 cities in a single query.
2. Builds 19-dimensional feature vectors for each city (see Section 4.2).
3. Runs batch XGBoost inference.
4. Writes all 99 city predictions to Redis with a 2-hour TTL.
5. Persists predictions to `aqi:pred_buffer` (a Redis list) for Lambda C to flush to Iceberg.

User-facing `GET /predict/{city_slug}` requests are served entirely from Redis — no Athena query occurs at request time, yielding sub-millisecond response latency. A background thread calls `warm_cache()` every 3,600 seconds independently of Lambda B, ensuring cache freshness even if Lambda B is unavailable.

### 3.4 Prediction Persistence (Lambda C)

Lambda C runs at 5 minutes past every hour via EventBridge. It reads the `aqi:pred_buffer` list from Redis, writes the records as Parquet to `s3://weather-bulk/predictions/`, and runs `INSERT INTO` on a dedicated Iceberg `predictions` table. This builds a permanent, queryable record of every prediction the model has ever made, enabling retrospective drift analysis.

### 3.5 Model Retraining and CI/CD

A GitHub Actions workflow (`daily_retrain.yml`) runs every day. It invokes `ml/train.py`, which reads hyperparameters from `ml/best-config.yaml`, trains XGBoost on the full Athena dataset, logs all parameters and metrics to MLflow, and writes the new model artifact to `ml/model-registry/model.ubj`. The FastAPI `GET /reload-model` endpoint hot-swaps the in-memory model without restarting the container. Deployment is fully automated: on every merge to `main`, `cd.yml` rebuilds the Docker image and redeploys to EC2 via `deploy/redeploy.sh`.

---

## 4. Data Science

### 4.1 Exploratory Data Analysis

The training dataset (after feature engineering) contains 186 samples from a single representative city subset used for development, expanding to the full multi-city corpus for production training. The AQI class distribution is:

| Class | Label | Proportion |
|---|---|---|
| 1 | Good | 11.8% |
| 2 | Fair | 38.7% |
| 3 | Moderate | 22.6% |
| 4 | Poor | 14.0% |
| 5 | Very Poor | 12.9% |

Classes 2 and 3 dominate, reflecting that most cities spend most hours in mild-to-moderate air quality conditions. Classes 4 and 5, while minority, are the most safety-relevant and were addressed through class-weighted training.

**Data Quality Issues Encountered:**
- *AQI class 6*: OWM occasionally returns 6 for extreme pollution events outside their documented scale. These labels are clamped to 5 during data loading, as XGBoost's internal 0-indexed representation cannot handle out-of-range targets.
- *Missing leading rows*: Lag features require at least three prior hours of data per city. The first three rows of each city's time series have `NaN` lags and are dropped before training; these rows are inherently untrainable.
- *Day-boundary gaps*: If a city is missing one hour of data, the lag chain breaks. These rows are also dropped.

### 4.2 Feature Engineering

AQI exhibits strong temporal autocorrelation — the next hour's air quality class is highly predictable from recent history. The feature set (19 features total) is designed around three principles:

**Momentum Features** (temporal autocorrelation):

| Feature | Formula | Semantic |
|---|---|---|
| `pm10_lag1` | PM10 at T−1 | Historical particulate level |
| `aqi_delta_1h` | AQI(T) − AQI(T−1) | Short-term AQI trend |
| `aqi_delta_2h` | AQI(T) − AQI(T−2) | Medium-term AQI momentum |
| `aqi_delta_3h` | AQI(T) − AQI(T−3) | Sustained AQI trend |
| `pm10_delta_1h` | PM10(T) − PM10(T−1) | Particulate momentum |
| `pm25_delta_1h` | PM2.5(T) − PM2.5(T−1) | Fine particulate momentum |
| `aqi_roll_std4` | σ(AQI T−3..T) | 4-hour AQI volatility |

**Cyclical Time Encoding** (boundary-safe temporal features):

Raw hour (0–23) and month (1–12) integers break at natural boundaries — hour 23 and hour 0 are adjacent in time but numerically distant. Sine/cosine encoding resolves this:

$$\text{hour\_sin} = \sin\!\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\!\left(\frac{2\pi \cdot h}{24}\right)$$

$$\text{month\_sin} = \sin\!\left(\frac{2\pi \cdot m}{12}\right), \quad \text{month\_cos} = \cos\!\left(\frac{2\pi \cdot m}{12}\right)$$

**Point-in-Time Pollutants**: `co`, `no`, `no2`, `o3`, `so2`, `nh3`, `pm10`, `pm25_ratio` (PM2.5 normalized by total pollutant mass).

Missing values remaining after row-dropping are filled with per-feature medians computed on the training set and persisted to `ml/model-registry/median.json` for use at inference time. This avoids distribution leakage and ensures the serving path can impute missing sensor readings deterministically.

### 4.3 Model Selection and Hyperparameter Tuning

Three model families were benchmarked in `experiement/model_benchmark.ipynb`: Logistic Regression (baseline), CatBoost, and XGBoost. XGBoost consistently achieved the highest macro F1 across cross-validation folds and was selected for production.

Hyperparameter optimization used Random Search (20 trials, 3-fold time-series cross-validation) and Bayesian Search (20 trials) as implemented in `experiement/feature_comparison.ipynb`. Results:

| Method | Best CV F1 (macro) | Trials |
|---|---|---|
| Random Search | **0.9425** | 20 |
| Bayesian Search | 0.9410 | 20 |

Random Search was selected as the production strategy. Best hyperparameters:

```
colsample_bytree : 0.5997      learning_rate    : 0.0810
gamma            : 0.1516      max_delta_step   : 1
max_depth        : 6           min_child_weight  : 8
n_estimators     : 542         reg_alpha        : 0.9359
reg_lambda       : 6.4496      subsample        : 0.6599
```

**Class-Weighted Training.** To handle class imbalance and improve prediction of safety-critical classes 4 and 5, a custom sample weight scheme was applied:
- *Sustained AQI transitions* (new class held for ≥2 consecutive hours): 5× weight boost.
- *Single-hour AQI blips* (new class reverts next hour): 2× weight boost.

**Validation F1 (macro): 0.9535**. All 11 training runs are tracked in MLflow under the experiment `AQI-Classification`, with the best model registered in the `AQI-XGBoost` model registry.

---

## 5. MLOps and Production Engineering

### 5.1 Containerization and Deployment

The FastAPI service is packaged as a Docker image built from the project `Dockerfile` and managed by `docker-compose.yml`. The application source and model artifacts are mounted as volumes, decoupling code updates (git pull) from image rebuilds. Nginx proxies port 80 to the internal FastAPI port 8000. A `systemd` unit file (`deploy/aqi-api.service`) ensures the container restarts automatically on crash or machine reboot.

### 5.2 Observability

The system pushes all telemetry signals directly to **Grafana Cloud** without a self-hosted collection layer, using Basic auth via an OTLP gateway:

- **Metrics → Grafana Mimir** (via OpenTelemetry SDK): prediction request count, prediction latency histogram, cache hit/miss rate, model drift scores per city, system CPU/memory.
- **Logs → Grafana Loki** (via async queue handler): structured JSON logs from all service components.
- **Traces → Grafana Tempo** (via OTLP HTTP exporter): distributed request traces from `FastAPIInstrumentor`, with excluded URLs for health and metrics endpoints to reduce noise.

Alerts are configured with threshold-based rules that push notifications to Teams, Slack, or Discord when drift or error rates exceed defined bounds.

### 5.3 Testing and CI

The test suite under `tests/` covers three layers:

- `test_api.py`: HTTP endpoint tests via FastAPI `TestClient`.
- `test_inference.py`: feature pipeline and model inference correctness.
- `test_training_logic.py`: training logic validation including class weight computation and feature array shapes.

GitHub Actions `ci.yml` runs linting and the full test suite on every pull request. Merges to `main` trigger `cd.yml` which rebuilds and redeploys the service to EC2.

---

## 6. Results

| Metric | Value |
|---|---|
| Cross-validated macro F1 | 0.9425 |
| Validation macro F1 | 0.9535 |
| Cities covered | 99 |
| Features used at inference | 19 |
| Prediction latency (p50) | < 1 ms (Redis-served) |
| Athena warm-cache query latency | ~3–8 s |
| Cache TTL | 2 hours |
| Model retraining frequency | Daily (GitHub Actions) |
| MLflow registered model versions | 11+ |

The dominant error cases are transitions between adjacent classes (e.g., class 2 ↔ class 3), which is expected given their similar pollutant concentrations. The class-weighting scheme substantially improved recall for classes 4 and 5 relative to the unweighted baseline.

---

## 7. Discussion

**Design Decisions.** The choice to pre-compute and cache all 99 city predictions at ingest time (rather than computing at request time) was the single most impactful engineering decision for serving quality. It shifts the latency cost to the data pipeline layer — where it is acceptable — and makes user-facing latency independent of model complexity or Athena query time.

**Limitations.** The current feature set does not include wind direction or speed, which are known drivers of pollutant dispersion [2]. The training corpus is a single year (2025–2026); longer histories would improve seasonal generalization. The AQI class 6 edge case highlights a broader data quality risk: relying on a single upstream API means undocumented API behaviors propagate silently unless caught at ingestion.

**Future Work.** Planned extensions include: (1) incorporating meteorological features (wind, precipitation) to improve accuracy for classes 4 and 5; (2) adding a geographic embedding to capture inter-city correlation; (3) replacing the Redis TTL-based cache with a streaming architecture (Kafka) to push predictions to clients in real time; and (4) implementing automated A/B testing between model versions in the registry.

---

## 8. Conclusion

AQI-MLops demonstrates that a full production MLOps system — covering data engineering, feature engineering, model training, experiment tracking, containerized deployment, Redis-backed serving, and cloud-native observability — can be built and maintained by a small team as an academic capstone project. The system achieves 94%+ macro F1 on a five-class air quality classification task across 99 global cities, with fully automated daily retraining and sub-millisecond prediction serving. All code, data pipeline definitions, model configurations, and infrastructure scripts are publicly available at the repository below.

---

## Repository

**GitHub:** [https://github.com/danhdanhtuan0308/AQI-MLops](https://github.com/danhdanhtuan0308/AQI-MLops)

The repository is organized as follows:

```
AQI-MLops/
├── app/                    # FastAPI service, inference helpers, telemetry
│   ├── main.py             # API routes, cache warming, background thread
│   ├── inference.py        # Feature building, XGBoost predict, AQI metadata
│   └── telemetry.py        # OTel tracing, Mimir metrics, Loki logging
├── data-pipeline/          # Batch backfill and live ingest
│   ├── fetch_weather.py    # Bulk backfill: OWM API → S3 Parquet
│   ├── config.yaml         # Cities, date range, S3 config, API URLs
│   └── lambda/handler.py   # Lambda A: hourly live ingest
├── lakehouse/              # Data lakehouse DDL and merge logic
│   ├── setup.sql           # Athena DDL: Bronze + Silver + Gold Iceberg tables
│   ├── merge_handler.py    # Lambda B: MERGE INTO aqi_unified
│   └── prediction_flush_handler.py  # Lambda C: Redis buffer → Iceberg
├── ml/                     # Training pipeline and model registry
│   ├── train.py            # Feature engineering, XGBoost train, MLflow logging
│   ├── best-config.yaml    # Persisted best hyperparameters
│   └── model-registry/     # model.ubj, features.json, median.json
├── experiement/            # EDA and model benchmarking notebooks
│   ├── data_identification.ipynb
│   ├── feature_comparison.ipynb
│   └── model_benchmark.ipynb
├── tests/                  # Unit and integration tests
├── deploy/                 # EC2 setup, Nginx config, systemd service
├── .github/workflows/      # CI, CD, and daily retraining GitHub Actions
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

Setup instructions, environment variable reference, and deployment guides are documented in the repository README and in each subdirectory's own `README.md`.

---

## References

[1] World Health Organization. *WHO Global Air Quality Guidelines.* Geneva: WHO, 2021.

[2] Byun, D., & Schere, K. L. (2006). Review of the governing equations, computational algorithms, and other components of the Models-3 Community Multiscale Air Quality (CMAQ) modeling system. *Applied Mechanics Reviews*, 59(2), 51–77.

[3] Zhao, R., Gu, X., Xue, B., Zhang, J., & Ren, W. (2018). Short period PM2.5 prediction based on multivariate linear regression model. *PLOS ONE*, 13(7), e0201011.

[4] Qi, Z., Wang, T., Song, G., Hu, W., Li, X., & Zhang, Z. (2019). Deep air learning: Interpolation, prediction, and feature analysis of fine-grained air quality. *IEEE Transactions on Knowledge and Data Engineering*, 32(12), 2285–2297.

[5] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., ... & Dennison, D. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems*, 28.

[6] Hermann, J., & Del Balso, M. (2017). Meet Michelangelo: Uber's Machine Learning Platform. *Uber Engineering Blog*.

[7] Dunn, J. (2016). Introducing FBLearner Flow: Facebook's AI backbone. *Facebook Engineering Blog*.

---

*Submitted for DS Capstone — Drexel University, May 2026.*
