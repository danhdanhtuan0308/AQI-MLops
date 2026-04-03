# app/ — FastAPI Prediction Service

The production web service that serves AQI predictions, history charts, drift analysis, and live model metrics. Deployed behind nginx on an EC2 t4g.small (ARM64 Graviton2). All prediction endpoints are served from Redis and respond in under 1 millisecond. Athena is only queried during the /warm-cache batch run, which is triggered by Lambda B after every hourly merge.

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
| GET | /history/{city_slug} | Predicted vs actual AQI for the last 24, 48, 168, or 720 hours |
| GET | /drift/{city_slug} | Feature drift z-scores comparing the current week to the previous week |
| GET | /metrics/{city_slug} | Live weighted F1, Precision, Recall computed from Athena ground truth data |
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
| aqi:history:{slug}:{hours} | Raw history rows for charts | 2 hours |
| aqi:drift:{slug} | Drift z-scores | 2 hours |
| aqi:metrics:{slug}:{hours} | F1, precision, recall | 2 hours |
| aqi:pred_buffer | Redis list of prediction records for Lambda C | No TTL (drained hourly) |

The /warm-cache endpoint replaces the cache for all 99 cities at once using a single Athena query. Lambda B calls this endpoint after every hourly data merge, so the cache is always fresh within minutes of new data arriving.

The /reload-model endpoint clears all aqi:* keys so the new model's predictions are served immediately after a retrain.

---

## /warm-cache Explained

Calling POST /warm-cache triggers the following steps:

1. Query Athena for the last 722 hours of sensor data for all 99 cities in a single SQL query.
2. Group the rows by city.
3. For each city, build a 15-feature vector from the last two rows and run predict_single.
4. Cache the prediction result under aqi:predict:{slug} (2-hour TTL).
5. Cache four time windows of raw history rows under aqi:history:{slug}:24, :48, :168, and :720.
6. Append a prediction record to aqi:pred_buffer so Lambda C can persist it to S3 and Athena.

This approach means Athena is queried only once per hour total, regardless of how many users are viewing the dashboard.

---

## inference.py

Core ML inference logic, isolated from FastAPI so it can be tested independently.

| Function | Description |
|----------|-------------|
| load_artifacts() | Loads model.ubj, median.json, and features.json from ml/model-registry/ |
| build_feature_vector(prev_row, row, median) | Engineers 15 features from two consecutive sensor rows |
| predict_single(model, X) | Returns predicted AQI class (1-5), label, color, and probabilities dict |
| batch_predict(model, median, rows) | Runs windowed predictions over a time-sorted list of rows, returns predicted and actual pairs |

### How batch_predict generates ground truth comparisons

For each position i in the rows list:
- Features come from rows at i-1 and i (time T-1 and T)
- The prediction is the model output for time T+1
- The actual value compared against is rows[i+2]["aqi"], which is the real observed AQI at T+1

This means every prediction in the history chart was a genuine forward-in-time forecast. The model only saw data up to time T when making each prediction.

### Feature list (16 total)

| Feature | Source | Description |
|---------|--------|-------------|
| aqi_lag1 | Previous row (T-1) | AQI class from the previous hour |
| pm10_lag1 | Previous row (T-1) | PM10 from the previous hour |
| aqi_delta_1h | T-1 → T | AQI change from T-1 to T (positive = rising, negative = falling) |
| pm10_delta_1h | T-1 → T | PM10 change from T-1 to T, captures PM10 momentum |
| hour_sin | Current row (T) | sin(2 * pi * hour / 24), daily cycle |
| hour_cos | Current row (T) | cos(2 * pi * hour / 24), daily cycle |
| month_sin | Current row (T) | sin(2 * pi * month / 12), seasonal cycle |
| month_cos | Current row (T) | cos(2 * pi * month / 12), seasonal cycle |
| pm25_ratio | Current row (T) | PM2.5 divided by sum of all major pollutants |
| co | Current row (T) | Carbon monoxide |
| no | Current row (T) | Nitric oxide |
| no2 | Current row (T) | Nitrogen dioxide |
| o3 | Current row (T) | Ozone |
| so2 | Current row (T) | Sulphur dioxide |
| nh3 | Current row (T) | Ammonia |
| pm10 | Current row (T) | Particulate matter under 10 micrometers |

---

## Dashboard

Single-file SPA using Alpine.js for state management and Chart.js for visualizations. No build step required.

| Tab | Contents |
|-----|----------|
| Forecast | City picker, next-hour prediction card, probability bar chart, historical AQI line chart |
| Data Drift | Feature drift z-scores bar chart, AQI class distribution comparison |
| System | Model metadata, live F1 and Precision and Recall cards, per-class breakdown, data freshness, auto-refresh countdown |

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
