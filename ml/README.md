# ml/ — Model Training and Registry

XGBoost classifier that predicts the next-hour AQI class (1 through 5) for any city in the dataset, trained on 12 months of hourly air quality data stored in Athena.

---

## Directory Structure

```
ml/
  best-config.yaml      Locked hyperparameters from random search experiments
  train.py              Production training pipeline
  model-registry/       Saved artifacts written by train.py (not tracked in git)
    model.ubj           XGBoost native binary model
    features.json       Feature list and target schema
    median.json         Per-feature medians used for NaN imputation at inference time
  mlruns/               MLflow local tracking store (not tracked in git)
```

---

## Training Data

| Property | Value |
|----------|-------|
| Source | AWS Athena aqi_db.aqi_unified (S3-backed Parquet via Iceberg) |
| Time range | 2025-03-20 to 2026-03-20 (12 months) |
| Cities | 99 world cities, hourly readings |
| Total rows | Approximately 876,000 after feature engineering |
| Target | aqi_next: AQI class at T+1, values 1 through 5 |

### AQI Scale

| Class | Label | Meaning |
|-------|-------|---------|
| 1 | Good | Air quality is satisfactory |
| 2 | Fair | Acceptable; some pollutants may cause minor concern for sensitive people |
| 3 | Moderate | Sensitive groups may experience health effects |
| 4 | Poor | Everyone may begin to experience health effects |
| 5 | Very Poor | Emergency conditions; serious health alert |

---

## Features (14 total)

Features are computed per city so there is no cross-city data leakage. Each prediction uses two consecutive rows from the same city.

| Feature | Source time | Description |
|---------|-------------|-------------|
| aqi_lag1 | T-1 | AQI class from the previous hour |
| pm10_lag1 | T-1 | PM10 reading from the previous hour |
| hour_sin | T | sin(2 * pi * hour / 24), encodes the daily time cycle |
| hour_cos | T | cos(2 * pi * hour / 24), encodes the daily time cycle |
| month_sin | T | sin(2 * pi * month / 12), encodes the seasonal cycle |
| month_cos | T | cos(2 * pi * month / 12), encodes the seasonal cycle |
| pm25_ratio | T | PM2.5 divided by the sum of all major pollutants |
| co | T | Carbon monoxide |
| no | T | Nitric oxide |
| no2 | T | Nitrogen dioxide |
| o3 | T | Ozone |
| so2 | T | Sulphur dioxide |
| nh3 | T | Ammonia |
| pm10 | T | Particulate matter under 10 micrometers |

The target aqi_next is the AQI class at T+1. Internally it is stored as 0-indexed (0 through 4) for XGBoost and converted back to 1-indexed (1 through 5) everywhere it is displayed.

---

## Model

| Property | Value |
|----------|-------|
| Algorithm | XGBClassifier |
| Objective | multi:softprob |
| Number of classes | 5 |
| Evaluation metric | mlogloss |
| Hyperparameter search | RandomizedSearchCV, 20 trials, 3-fold cross-validation |
| Cross-validation F1 (weighted) | 0.9425 |
| Validation F1 (weighted) | 0.9535 |
| Class weighting | compute_sample_weight("balanced") to prevent bias toward majority classes |

### Locked Hyperparameters

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

## Training Pipeline

The pipeline runs in five stages:

```
load_data()
  Query all rows from Athena aqi_db.aqi_unified

engineer_features()
  Compute lag features, sin/cos time encodings, pm25_ratio, and the target column

build_arrays()
  Convert to float32 numpy arrays
  Replace NaN values with per-feature dataset medians
  Convert target labels to 0-indexed integers

XGBClassifier.fit()
  Train on 100 percent of data using locked hyperparameters

model-registry/
  Write model.ubj, features.json, and median.json to disk
  Log all params, metrics, and artifacts to MLflow
```

### Running Training

```bash
source .venv/bin/activate
python ml/train.py
```

Requires AWS credentials in .env:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

---

## MLflow Tracking

```bash
mlflow ui --backend-store-uri ml/mlruns
# Open http://127.0.0.1:5000
```

**What gets logged per run:**

- Parameters: all 10 XGBoost hyperparameters, feature list, total row count, search strategy
- Metrics: cv_f1 (0.9425) and val_f1_notebook (0.9535)
- Artifacts: best-config.yaml, features.json, median.json, packaged XGBoost model
- Model registered in the MLflow Model Registry as AQI-XGBoost

---

## Model Registry Files

Three files are written to ml/model-registry/ after training. The FastAPI service loads these at startup and at hot-reload time. No MLflow connection is needed at inference time.

| File | Description |
|------|-------------|
| model.ubj | XGBoost native binary format, loaded with model.load_model() |
| features.json | List of 14 feature names and the target column name |
| median.json | Per-feature median values used to fill missing sensor readings at inference time |

These files are bind-mounted into the Docker container at runtime and are not baked into the image. This allows the weekly retrain to update the model on disk and hot-reload it without rebuilding the container.

---

## Weekly Retrain

Every Monday at 02:00 UTC, GitHub Actions runs weekly_retrain.yml which:

1. Runs the CI gate (ruff lint and pytest) to verify code health
2. SSHes into EC2 and runs python ml/train.py against the latest Athena data
3. Runs a smoke test to verify the new model produces valid predictions
4. Calls POST /reload-model on the API to load the new artifacts into memory
5. The reload also clears all Redis cache keys so fresh predictions are computed on the next /warm-cache call

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
