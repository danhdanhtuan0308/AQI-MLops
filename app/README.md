# app/ — FastAPI Prediction Service

The production web service that serves AQI predictions, drift analysis, and live model metrics. Built with FastAPI + Jinja2, deployed behind nginx on EC2 t4g.small (ARM64 Graviton).

---

## Directory Structure

```
app/
├── __init__.py           # Package marker
├── main.py               # FastAPI app — all routes defined here
├── inference.py          # Feature engineering, predict_single, batch_predict, load_artifacts
└── templates/
    └── index.html        # Single-page dashboard (Alpine.js + Chart.js + Tailwind CDN)
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve the interactive dashboard (index.html) |
| `GET` | `/cities` | List all 99+ cities with coordinates and timezone |
| `GET` | `/predict/{city_slug}` | Next-hour AQI prediction + per-class probabilities |
| `GET` | `/history/{city_slug}` | Last N hours of predicted vs actual AQI |
| `GET` | `/drift/{city_slug}` | Feature drift (recent week vs reference week, z-scores) |
| `GET` | `/metrics/{city_slug}` | **Live** weighted F1, Precision, Recall (last 7 days by default) |
| `GET` | `/health` | Service health + model metadata (model type, n_features, n_cities) |
| `POST` | `/reload-model` | Hot-reload model artifacts from disk without restarting the server |

### `/metrics/{city_slug}` — Live Production Metrics

This is the only metrics endpoint. It uses **Athena ground-truth data** (real observed AQI values) compared against the model's T+1 predictions. It does **not** touch the training set in any way.

Query params:
- `hours` (int, default=168, max=336) — look-back window in hours. 168 = last 7 days.

Returns:
```json
{
  "city": "Tokyo",
  "window_hours": 168,
  "n_predictions": 165,
  "weighted": { "f1": 0.874, "precision": 0.881, "recall": 0.874 },
  "per_class": {
    "1": { "f1": 0.91, "precision": 0.93, "recall": 0.89, "support": 42 },
    ...
  },
  "computed_at": "2026-03-22 21:00 UTC"
}
```

Requires ≥ 10 labelled rows in Athena; returns HTTP 422 otherwise (insufficient live data).

---

## inference.py

Core ML inference logic — isolated from FastAPI so it can be tested independently.

| Function | Description |
|---|---|
| `load_artifacts()` | Loads `model.ubj`, `median.json`, `features.json` from `ml/model-registry/` |
| `build_feature_vector(row, prev_row, median)` | Engineers 12 features from two consecutive rows |
| `predict_single(model, median, row, prev_row)` | Returns predicted AQI + per-class probabilities |
| `batch_predict(model, median, rows)` | Runs predictions over a list of rows, returns predicted+actual pairs |

### Feature list (12 total)

`aqi_lag1`, `pm10_lag1`, `hour_sin`, `hour_cos`, `pm25_ratio`, `co`, `no`, `no2`, `o3`, `so2`, `nh3`, `pm10`

---

## Dashboard — index.html

Single-file SPA using Alpine.js (CDN) for reactivity and Chart.js for charts. No build step.

### Tabs

| Tab | Contents |
|---|---|
| **Forecast** | City picker, next-hour prediction card, probability bar chart, historical AQI line chart |
| **Data Drift** | Feature drift scores (z-scores, bar chart), AQI class distribution shift |
| **System** | API status, model metadata, **Live F1 / Precision / Recall cards**, per-class breakdown table, data freshness, drift summary, auto-refresh countdown, retrain guidance |

### Live Metrics on System Tab

The System tab calls `/metrics/{city_slug}?hours=168` when:
- The System tab is opened
- "Refresh Now" is clicked
- The hourly auto-refresh fires

Metrics display green ≥ 80%, yellow ≥ 60%, red < 60%.

---

## Running locally

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000`.

Requires `ml/model-registry/model.ubj`, `median.json`, `features.json` and a `.env` file with `AWS_*` variables for Athena queries.
