"""
AQI XGBoost — Production Training Pipeline
============================================
Reads hyper-parameters from ml/best-config.yaml, trains an XGBoost
classifier on the full dataset to predict AQI at T+1 (next hour),
logs every artefact to MLflow, and registers the model in the
Model Registry.

Usage
-----
    # from project root, with .venv activated:
    python ml/train.py

    # optional overrides via env vars:
    MLFLOW_TRACKING_URI=http://0.0.0.0:5000 python ml/train.py

    # launch the tracking UI after training:
    mlflow ui --backend-store-uri ml/mlruns
"""

import argparse
import json
import logging
import os
from pathlib import Path

import boto3
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent          # project root
CONFIG_PATH   = Path(__file__).parent / "best-config.yaml"
MLRUNS_DIR    = Path(__file__).parent / "mlruns"      # local tracking store
REGISTRY_DIR  = Path(__file__).parent / "model-registry"  # portable model artifacts

# ── Load hyper-parameters from best-config.yaml ───────────────────────────────
load_dotenv(ROOT / ".env")

with open(CONFIG_PATH) as fh:
    cfg = yaml.safe_load(fh)

SELECTED_NAME: str = cfg["selected_model"]["name"]   # "XGBoost Random Search"
SEARCH_KEY: str    = "random_search" if "Random" in SELECTED_NAME else "bayes_search"
BEST_PARAMS: dict  = cfg[SEARCH_KEY]["best_params"]
CV_F1: float       = float(cfg[SEARCH_KEY]["best_cv_f1"])
VAL_F1: float      = float(cfg["selected_model"]["validation_f1"])

# ── Feature / target schema ───────────────────────────────────────────────────
FEATURES = [
    "pm10_lag1",                                       # Historical Trend (T-1)
    "aqi_delta_1h",                                    # AQI momentum    (T - T-1)
    "pm10_delta_1h",                                   # PM10 momentum   (T - T-1)
    "hour_sin", "hour_cos",                            # Temporal Cycle  (T)
    "month_sin", "month_cos",                          # Seasonal Cycle  (T)
    "pm25_ratio", "co", "no", "no2", "o3", "so2", "nh3", "pm10",  # Point-in-time (T)
]
TARGET = "aqi_next"   # AQI 1-5 at T+1 (0-indexed → 0-4 for XGBoost internally)

# ── MLflow setup ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME      = "AQI-Classification"
REGISTERED_MODEL_NAME = "AQI-XGBoost"

_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri())
mlflow.set_tracking_uri(_tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)


# ── 1. Data loading ───────────────────────────────────────────────────────────

def load_data(lookback_days: int | None = None) -> pd.DataFrame:
    """Pull raw AQI data from AWS Athena.

    Parameters
    ----------
    lookback_days:
        When set, restrict training data to the most recent N days.
        Use ``None`` (default) to train on all data ever ingested.
        Example: ``--lookback-days 365`` for a true rolling 1-year retrain.
        Each daily run shifts the window forward by 1 day automatically
        because the query uses ``current_timestamp`` at runtime.
    """
    import awswrangler as wr

    session = boto3.Session(
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )

    where = (
        f"WHERE timestamp >= current_timestamp - interval '{lookback_days}' day"
        if lookback_days
        else ""
    )
    sql = f"""
        SELECT
            timestamp, city, city_slug,
            aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3, source
        FROM aqi_db.aqi_unified
        {where}
        ORDER BY city_slug ASC, timestamp ASC
    """
    raw = wr.athena.read_sql_query(
        sql=sql,
        database="aqi_db",
        s3_output="s3://weather-bulk/athena-results/",
        boto3_session=session,
        ctas_approach=False,
    )

    raw["aqi"] = pd.to_numeric(raw["aqi"], errors="coerce").astype("Int64").clip(upper=5)
    n_class6 = int((raw["aqi"] == 6).sum())
    log.info("Loaded %d rows from Athena  (%d cities)  [class-6 merged: %d]",
             len(raw), raw["city_slug"].nunique(), n_class6)
    return raw


# ── 2. Feature engineering ────────────────────────────────────────────────────

def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-city lag / lead features and temporal-cycle encodings.

    Feature diagram
    ---------------
    pm10_lag1             (T-1)  Historical PM10 level
    aqi_delta_1h          (T)    AQI momentum (T − T-1): rising (+) or falling (−)
    pm10_delta_1h         (T)    PM10 momentum (T − T-1)
    hour_sin, hour_cos    (T)    Temporal cycle — time of day
    month_sin, month_cos  (T)    Seasonal cycle — month of year
    pm25_ratio            (T)    PM2.5 share of total pollution burden
    co, no, no2, o3,      (T)    Point-in-time pollutants
      so2, nh3, pm10
    ─────────────────────────────────────────────────────────────────
    TARGET  aqi_next      (T+1)  Next-hour AQI class (1–5)
    """
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    # PM2.5 burden ratio
    pollutant_sum    = raw[["pm2_5", "pm10", "no2", "o3", "so2"]].sum(axis=1)
    raw["pm25_ratio"] = (raw["pm2_5"] / (pollutant_sum + 1e-9)).round(4)

    # Temporal cycle — encode hour as sine/cosine pair
    raw["hour_sin"] = np.sin(2 * np.pi * raw["timestamp"].dt.hour / 24).round(6)
    raw["hour_cos"] = np.cos(2 * np.pi * raw["timestamp"].dt.hour / 24).round(6)

    # Seasonal cycle — encode month as sine/cosine pair
    raw["month_sin"] = np.sin(2 * np.pi * raw["timestamp"].dt.month / 12).round(6)
    raw["month_cos"] = np.cos(2 * np.pi * raw["timestamp"].dt.month / 12).round(6)

    # Sort so shift() is correct within each city
    raw = raw.sort_values(["city_slug", "timestamp"]).reset_index(drop=True)

    # Per-city lag / lead — prevents cross-city data leakage
    grp               = raw.groupby("city_slug", sort=False)
    raw["aqi_lag1"]   = grp["aqi"].shift(1)         # AQI  @ T-1
    raw["pm10_lag1"]  = grp["pm10"].shift(1)        # PM10 @ T-1
    raw["aqi_next"]   = grp["aqi"].shift(-1)        # AQI  @ T+1  ← target
    # Delta features: momentum (rate of change from T-1 → T)
    raw["aqi_delta_1h"]  = raw["aqi"].astype(float) - raw["aqi_lag1"]
    raw["pm10_delta_1h"] = raw["pm10"].astype(float) - raw["pm10_lag1"]

    # Drop first/last row of each city (lag/lead undefined)
    feat_df = (
        raw.dropna(subset=["pm10_lag1", "aqi_next"])
           .copy()
           .reset_index(drop=True)
    )
    feat_df["aqi_next"] = feat_df["aqi_next"].astype(int)

    log.info("After feature engineering: %d rows  (%d features)",
             len(feat_df), len(FEATURES))
    return feat_df


# ── 3. Array preparation ──────────────────────────────────────────────────────

def build_arrays(feat_df: pd.DataFrame) -> tuple:
    """
    Build float32 numpy arrays from the full dataset.
    NaN is imputed with the dataset median.
    XGBoost requires 0-indexed labels (0–4); values are shifted by -1.
    Returns (X, y, median).
    """
    median = feat_df[FEATURES].median()
    X = feat_df[FEATURES].fillna(median).to_numpy(dtype="float32")
    y = feat_df[TARGET].astype(int).to_numpy() - 1   # 0-indexed for XGBoost
    return X, y, median


# ── 4. Main pipeline ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AQI XGBoost training pipeline")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        metavar="N",
        help=(
            "Train on the most recent N days of data (default: 365 = 1 full year). "
            "Each daily retrain shifts the window forward by 1 day automatically "
            "because the SQL uses current_timestamp at runtime. "
            "e.g. run on 03/21 → trains on 03/21/2025–03/21/2026."
        ),
    )
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("  AQI XGBoost Training Pipeline")
    log.info("  Config  : %s", CONFIG_PATH.relative_to(ROOT))
    log.info("  Model   : %s", SELECTED_NAME)
    log.info("  CV F1   : %.6f  |  Val F1 (notebook): %.6f", CV_F1, VAL_F1)
    log.info(
        "  Lookback: %s",
        f"last {args.lookback_days} days (rolling 1-year window)",
    )
    log.info("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────────────────
    raw     = load_data(args.lookback_days)
    feat_df = engineer_features(raw)
    X, y, median = build_arrays(feat_df)

    # Log class distribution so imbalance is visible in MLflow
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        log.info("  Class %d (AQI %d): %d rows  (%.1f%%)", cls, cls + 1, cnt, 100 * cnt / len(y))

    # ── Train ─────────────────────────────────────────────────────────────────────
    log.info(
        "Training on %d rows (%s)",
        len(feat_df),
        f"last {args.lookback_days} days",
    )
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        **BEST_PARAMS,
    )
    model.fit(X, y, sample_weight=compute_sample_weight("balanced", y), verbose=False)
    log.info("Training complete")

    # ── MLflow run ─────────────────────────────────────────────────────────────
    run_name = SELECTED_NAME.replace(" ", "_")
    with mlflow.start_run(run_name=run_name) as run:

        # --- hyper-parameters -------------------------------------------------
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("selected_model",  SELECTED_NAME)
        mlflow.log_param("search_strategy", SEARCH_KEY)
        mlflow.log_param("n_features",      len(FEATURES))
        mlflow.log_param("features",        json.dumps(FEATURES))
        mlflow.log_param("total_rows",      len(X))
        mlflow.log_param("lookback_days",   args.lookback_days or "full")

        # --- reference metrics from hyper-parameter search --------------------
        mlflow.log_metric("cv_f1",           CV_F1)
        mlflow.log_metric("val_f1_notebook",  VAL_F1)

        # --- tags -------------------------------------------------------------
        mlflow.set_tag("model_type",   "XGBoost")
        mlflow.set_tag("target",       "aqi_next_T+1")
        mlflow.set_tag("dataset",      "aqi_db.aqi_unified")
        mlflow.set_tag("search_type",  SEARCH_KEY)

        # --- artefacts --------------------------------------------------------
        # config file
        mlflow.log_artifact(str(CONFIG_PATH), artifact_path="config")

        # feature list (for downstream inference)
        feat_artifact = Path("/tmp/features.json")
        feat_artifact.write_text(json.dumps({"features": FEATURES, "target": TARGET}))
        mlflow.log_artifact(str(feat_artifact), artifact_path="preprocessing")

        # dataset median used for NaN imputation at inference time
        median_artifact = Path("/tmp/median.json")
        median.to_json(median_artifact)
        mlflow.log_artifact(str(median_artifact), artifact_path="preprocessing")

        # --- register model ---------------------------------------------------
        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X[:5],
        )

        run_id = run.info.run_id

    # ── Save model + artifacts to ml/model-registry/ ─────────────────────────
    # These files are the single source of truth for inference — no MLflow
    # dependency required at prediction time.
    REGISTRY_DIR.mkdir(exist_ok=True)
    model.save_model(str(REGISTRY_DIR / "model.ubj"))
    (REGISTRY_DIR / "features.json").write_text(
        json.dumps({"features": FEATURES, "target": TARGET}, indent=2)
    )
    median.to_json(REGISTRY_DIR / "median.json")
    log.info("Saved model artifacts to  %s", REGISTRY_DIR.relative_to(ROOT))

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("  MLflow run ID : %s", run_id)
    log.info("  Experiment    : %s", EXPERIMENT_NAME)
    log.info("  Registry name : %s", REGISTERED_MODEL_NAME)
    log.info("  Model URI     : %s", model_info.model_uri)
    log.info("  Tracking URI  : %s", _tracking_uri)
    log.info("  Trained on    : %d rows", len(X))
    log.info("")
    log.info("  Model files   : %s/", REGISTRY_DIR.relative_to(ROOT))
    log.info("    model.ubj  — XGBoost native model")
    log.info("    features.json — feature list for inference")
    log.info("    median.json   — NaN imputation values")
    log.info("  Run inference : python ml/predict.py")
    log.info("")
    log.info("  Launch UI:  mlflow ui --backend-store-uri %s", MLRUNS_DIR)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
