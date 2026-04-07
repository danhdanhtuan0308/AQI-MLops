# app/ — FastAPI Prediction Service

The production web service that serves AQI predictions, history charts, drift analysis, and live model metrics. Deployed behind nginx on an EC2 t4g.small (ARM64 Graviton2). All prediction endpoints are served from Redis and respond in under 1 millisecond. Athena is only queried during the /warm-cache batch run, which is triggered by Lambda B after every hourly merge.

**Current model: Approach A — XGBoost 19-feature model with 5× transition sample boost, trained on a rolling 365-day window.**

---

## Directory Structure

```
app/
  __init__.py           Package marker
  main.py               FastAPI app: all routes, Redis helpers, warm-cache logic
  inference.py          Feature engineering, predict_single, batch_predict, load_artifacts
  telemetry.py          Observability: traces → Tempo, metrics → Mimir, logs → Loki
  templates/
    index.html          Single-page dashboard (Alpine.js + Chart.js, Tailwind CDN)
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Serve the interactive dashboard |
| GET | /cities | All 99 cities with coordinates and timezone |
| GET | /predict/{city_slug} | Next-hour AQI prediction and per-class probabilities (from Redis) |
| GET | /history/{city_slug} | Predicted vs actual AQI for the last 24, 48, 72, or 168 hours |
| GET | /drift/{city_slug} | Per-city feature drift z-scores: today vs yesterday (1d) or this week vs prior week (7d) |
| GET | /drift | Global drift across all 99 cities pooled — model-level input distribution shift |
| GET | /model-metrics | Global weighted F1, Precision, Recall pooled across all 99 cities (24/48/72/168 h window) |
| GET | /accuracy-trend | Global day-by-day accuracy trend for concept drift detection (2–12 week look-back) |
| GET | /feature-importance | XGBoost feature importances (weight, gain, cover) — model-level |
| GET | /feature-importance/history | Feature importance trend across the last 30 daily retrains |
| GET | /health | Service health and model metadata |
| GET | /cache/status | Redis connection status, number of cached keys, and memory used |
| POST | /warm-cache | Query Athena once for all 99 cities, run batch inference, populate Redis |
| POST | /reload-model | Hot-reload model artifacts from disk without restarting the container |

---

## Redis Caching

All four data-serving routes (predict, history, drift, metrics) check Redis before going to Athena. If a key exists in Redis, the cached value is returned immediately. If not, Athena is queried and the result is stored in Redis for future requests.

Cache keys and their TTL:

| Key pattern | Contents | TTL |
|-------------|----------|-----|
| aqi:predict:{slug} | Next-hour prediction payload | 2 hours |
| aqi:history:{slug}:{hours} | History rows for charts — windows: 24, 48, 72, 168 | 2 hours |
| aqi:drift:{slug}:{window} | Per-city drift z-scores — windows: 1d, 7d | 2 hours |
| aqi:drift:global:{window} | Global (all cities) drift z-scores — windows: 1d, 7d | 2 hours |
| aqi:model-metrics:global:{hours} | Global F1, precision, recall — windows: 24, 48, 72, 168 | 2 hours |
| aqi:accuracy_trend:global:{weeks} | Global day-by-day accuracy trend | 2 hours |
| aqi:pred_buffer | Redis list of prediction records for Lambda C | No TTL (drained hourly) |

The /warm-cache endpoint replaces the cache for all 99 cities at once using a single Athena query. Lambda B calls this endpoint after every hourly data merge, so the cache is always fresh within minutes of new data arriving.

The /reload-model endpoint clears all aqi:* keys so the new model's predictions are served immediately after a retrain.

---

## /warm-cache Explained

Calling POST /warm-cache triggers the following steps:

1. Query Athena for the last 340 hours of sensor data for all 99 cities in a single SQL query.
2. Group the rows by city.
3. For each city, build a 19-feature vector from the last four rows and run predict_single.
4. Cache the prediction result under aqi:predict:{slug} (2-hour TTL).
5. Cache four history windows under aqi:history:{slug}:24, :48, :72, and :168.
6. Cache per-city drift payloads under aqi:drift:{slug}:1d and aqi:drift:{slug}:7d.
7. Compute and cache global metrics (aqi:model-metrics:global:{hours}) and global drift (aqi:drift:global:{window}).
8. Append a prediction record to aqi:pred_buffer so Lambda C can persist it to S3 and Athena.
9. All Redis writes are batched into a single pipeline flush — one TCP round-trip instead of ~1200 sequential SETEX calls.

This approach means Athena is queried only once per hour total, regardless of how many users are viewing the dashboard.

The /warm-cache endpoint is called automatically by Lambda B after every hourly data merge. In addition, a background thread started at service startup calls warm_cache() independently every hour (default interval 3600 s, configurable via WARM_INTERVAL_SEC) so the cache stays hot even if Lambda B is unavailable.

---

## inference.py

Core ML inference logic, isolated from FastAPI so it can be tested independently.

| Function | Description |
|----------|-------------|
| load_artifacts() | Loads model.ubj, median.json, and features.json from ml/model-registry/ |
| build_feature_vector(row_t3, row_t2, row_prev, row_curr, median) | Engineers 19 features from four consecutive sensor rows (T-3, T-2, T-1, T) |
| predict_single(model, X) | Returns predicted AQI class (1-5), label, color, and probabilities dict |
| batch_predict(model, median, rows) | Runs windowed predictions over a time-sorted list of rows (minimum 5 rows), returns predicted and actual pairs |

### How batch_predict generates ground truth comparisons

For each position i (starting at 3) in the rows list:
- Features come from rows at i-3, i-2, i-1, and i (time T-3, T-2, T-1, T) — the 3-hour lookback window required by Approach A
- The prediction is the model output for time T+1
- The actual value compared against is rows[i+1]["aqi"], which is the real observed AQI at T+1

This means every prediction in the history chart was a genuine forward-in-time forecast. The model only saw data up to time T when making each prediction.

### Feature list (19 total — Approach A)

| Feature | Source | Description |
|---------|--------|-------------|
| pm10_lag1 | T-1 row | PM10 from the previous hour |
| aqi_delta_1h | T-1 → T | AQI change from T-1 to T (momentum) |
| pm10_delta_1h | T-1 → T | PM10 change from T-1 to T |
| aqi_delta_2h | T-2 → T | AQI change over 2 hours (medium-term momentum) ← NEW |
| aqi_delta_3h | T-3 → T | AQI change over 3 hours (sustained directional trend) ← NEW |
| aqi_roll_std4 | T-3…T | 4-hour rolling std of AQI (neighbourhood volatility) ← NEW |
| pm25_delta_1h | T-1 → T | PM2.5 change from T-1 to T (PM2.5 often leads AQI transitions) ← NEW |
| hour_sin | T | sin(2π × hour / 24), daily cycle |
| hour_cos | T | cos(2π × hour / 24), daily cycle |
| month_sin | T | sin(2π × month / 12), seasonal cycle |
| month_cos | T | cos(2π × month / 12), seasonal cycle |
| pm25_ratio | T | PM2.5 / (PM2.5 + PM10 + NO2 + O3 + SO2) |
| co | T | Carbon monoxide |
| no | T | Nitric oxide |
| no2 | T | Nitrogen dioxide |
| o3 | T | Ozone |
| so2 | T | Sulphur dioxide |
| nh3 | T | Ammonia |
| pm10 | T | Particulate matter under 10 micrometers |

### Approach A — transition boost

Training uses a **5× sample weight** on transition rows (rows where the AQI class changes from T to T+1). These rows represent only ~6.5% of training data but are the hardest cases to predict and the most important operationally. The boost is applied on top of balanced class weights via `sample_weight` in `model.fit()`.

---

## Dashboard

Single-file SPA using Alpine.js for state management and Chart.js for visualizations. No build step required.

| Tab | Contents |
|-----|----------|
| Forecast | City picker, next-hour prediction card, probability bar chart, historical AQI line chart |
| Data Drift | Per-city feature drift z-scores bar chart (1d / 7d window), AQI class distribution comparison, prediction distribution drift |
| System | Global model metrics (F1, Precision, Recall), per-class breakdown, global accuracy trend, feature importances, data freshness, auto-refresh countdown |

Metric color thresholds: green for 80% or above, yellow for 60% to 79%, red for below 60%.

---

## Running Locally

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Requires:
- ml/model-registry/model.ubj, median.json, and features.json (run ml/train.py once to generate)
- A .env file with AWS credentials for Athena queries
- Optional: REDIS_URL in .env for Redis caching during local development

---

## Observability (telemetry.py)

All three observability signals are pushed directly from the FastAPI container to Grafana Cloud. No agents or sidecars are needed.

| Signal | Destination | How |
|--------|-------------|-----|
| Traces | Grafana Cloud Tempo | OTLPSpanExporter with BatchSpanProcessor, pushed per-request to /v1/traces |
| Metrics | Grafana Cloud Mimir | OTLPMetricExporter with PeriodicExportingMetricReader, pushed every 30 seconds to /v1/metrics |
| System metrics | Grafana Cloud Mimir | CPU utilization and memory usage only (limited for Free tier cardinality) |
| Logs | Grafana Cloud Loki | Custom _LokiHandler with non-blocking QueueListener, pushed via HTTP POST |

### Initialization Order

1. `setup_tracing(app)` is called at **module level** immediately after `app = FastAPI(...)` — this is required because FastAPIInstrumentor adds ASGI middleware, and the middleware stack freezes once the first request is processed.
2. `setup_metrics_push()`, `setup_system_metrics()`, and `setup_logging()` are called in the startup event handler.

### Application Metrics

| Metric | Type | Description |
|--------|------|-------------|
| aqi.predictions.total | Counter | Total AQI predictions served |
| aqi.cache.ops.total | Counter | Redis cache hits and misses |
| aqi.warm_cache.seconds | Histogram | Time to run /warm-cache |
| aqi.athena.queries.total | Counter | Total Athena queries executed |

### Environment Variables

| Variable | Description |
|----------|-------------|
| GRAFANA_OTLP_ENDPOINT | OTLP gateway URL (e.g. https://otlp-gateway-prod-us-east-2.grafana.net/otlp) |
| GRAFANA_OTLP_INSTANCE_ID | Grafana Cloud instance ID |
| GRAFANA_API_KEY | Grafana Cloud API key (used for all three signals) |
| GRAFANA_LOKI_URL | Loki push URL (e.g. https://logs-prod-036.grafana.net/loki/api/v1/push) |
| GRAFANA_LOKI_USER | Loki instance user ID |
| OTEL_SERVICE_NAME | Service name tag (default: aqi-api) |
| OTEL_SERVICE_VERSION | Service version tag (default: 1.0) |
