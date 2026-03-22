# ML Module — AQI Next-Hour Prediction

XGBoost classifier that predicts the next-hour AQI class (1–5) for any city in the dataset, trained on 12 months of OpenWeatherMap air quality data stored in AWS Athena.

---

## Directory Structure

```
ml/
├── best-config.yaml      # Locked hyper-parameters from search experiments
├── train.py              # Production training pipeline
├── model-registry/       # Saved model artifacts (created by train.py)
│   ├── model.ubj         # XGBoost native binary model
│   ├── features.json     # Feature list + target schema
│   └── median.json       # Per-feature medians for NaN imputation
└── mlruns/               # MLflow local tracking store
```

---

## Data

| Property | Value |
|---|---|
| Source | AWS Athena `aqi_db.aqi_unified` (S3-backed Parquet) |
| Time range | 2025-03-20 → 2026-03-20 (12 months) |
| Cities | 100 world cities (hourly readings) |
| Rows | ~876,000 after feature engineering |
| Target | `aqi_next` — AQI class at T+1 (values 1–5) |

### AQI Scale

| Class | Label | Meaning |
|---|---|---|
| 1 | Good | Air quality is satisfactory |
| 2 | Fair | Acceptable; some pollutants may cause minor concern |
| 3 | Moderate | Sensitive groups may experience health effects |
| 4 | Poor | Everyone may experience health effects |
| 5 | Very Poor | Emergency conditions; health alert |

---

## Feature Engineering

Features are computed per-city — no cross-city leakage.

| Feature | Source time | Description |
|---|---|---|
| `aqi_lag1` | T−1 | Previous-hour AQI (baseline state) |
| `pm10_lag1` | T−1 | Previous-hour PM10 (historical trend) |
| `hour_sin` | T | `sin(2π·hour/24)` — temporal cycle encoding |
| `hour_cos` | T | `cos(2π·hour/24)` — temporal cycle encoding |
| `pm25_ratio` | T | PM2.5 / Σ(major pollutants) — pollution burden share |
| `co` | T | Carbon monoxide |
| `no` | T | Nitric oxide |
| `no2` | T | Nitrogen dioxide |
| `o3` | T | Ozone |
| `so2` | T | Sulphur dioxide |
| `nh3` | T | Ammonia |
| `pm10` | T | Particulate matter ≤10 µm |

**Target:** `aqi_next` = AQI class at T+1. Internally stored as 0-indexed (0–4) for XGBoost; displayed as 1–5 everywhere else.

---

## Model

| Property | Value |
|---|---|
| Algorithm | `XGBClassifier` (XGBoost) |
| Objective | `multi:softprob` |
| Classes | 5 (AQI 1–5) |
| Eval metric | `mlogloss` |
| Search method | `RandomizedSearchCV` (20 trials, 3-fold CV) |
| CV F1 (weighted) | 0.9425 |
| Validation F1 | 0.9535 |

### Locked Hyper-parameters (`random_search`)

```yaml
colsample_bytree: 0.5997
gamma:            0.1516
learning_rate:    0.0810
max_delta_step:   1
max_depth:        6
min_child_weight: 8
n_estimators:     542
reg_alpha:        0.9359
reg_lambda:       6.4496
subsample:        0.6599
```

---

## Training Pipeline (`train.py`)

The pipeline has three stages:

```
load_data()           Pull all rows from Athena (no date filter)
      ↓
engineer_features()   Compute lag features, sin/cos time, pm25_ratio, target
      ↓
build_arrays()        Float32 numpy arrays; NaN → dataset median; labels 0-indexed
      ↓
XGBClassifier.fit()   Train on 100% of data (hyperparams locked)
      ↓
MLflow run            Log params, reference metrics, artifacts, register model
      ↓
model-registry/       Write model.ubj + features.json + median.json
```

### Run Training

```bash
# from project root, with .venv activated:
source .venv/bin/activate

python ml/train.py

# optional override:
MLFLOW_TRACKING_URI=http://0.0.0.0:5000 python ml/train.py
```

Requires AWS credentials in `.env`:
```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_DEFAULT_REGION=us-east-1
```

---

## MLflow Tracking

Open the tracking UI after training:

```bash
mlflow ui --backend-store-uri ml/mlruns
# → http://127.0.0.1:5000
```

### Logged Items

**Params:** all 10 XGBoost hyperparams, `selected_model`, `search_strategy`, `n_features`, `features` (JSON list), `total_rows`

**Metrics:** `cv_f1` (0.9425), `val_f1_notebook` (0.9535)

**Tags:** `model_type`, `target`, `dataset`, `search_type`

**Artifacts:**
- `config/best-config.yaml`
- `preprocessing/features.json`
- `preprocessing/median.json`
- `model/` — MLflow-packaged XGBoost model (also registered in Model Registry as `AQI-XGBoost`)

---

## Model Registry (`ml/model-registry/`)

After training, three files are written here for dependency-free inference:

| File | Description |
|---|---|
| `model.ubj` | XGBoost native binary — load with `model.load_model()` |
| `features.json` | `{"features": [...12 names...], "target": "aqi_next"}` |
| `median.json` | Per-feature median values for NaN imputation at inference time |

These files are the **single source of truth** for the inference service. No MLflow connection needed at prediction time.

---

## Inference

The FastAPI backend (`app/`) loads these artifacts on startup and serves real-time predictions. See [`app/README.md`](../app/README.md) or run:

```bash
uvicorn app.main:app --reload
```
