"""
FastAPI backend — AQI real-time prediction service.

Routes
------
GET  /                        → serve dashboard HTML
GET  /cities                  → list all available cities
GET  /predict/{city_slug}     → next-hour AQI prediction for a city
GET  /history/{city_slug}     → predicted vs actual history (for dashboard chart)

Run
---
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

import boto3
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from sklearn.metrics import precision_recall_fscore_support

from .inference import (
    AQI_META,
    CITY_TIMEZONES,
    batch_predict,
    build_feature_vector,
    load_artifacts,
    predict_single,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# Known cities from data-pipeline config — used to validate user input
with open(ROOT / "data-pipeline" / "config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

KNOWN_CITIES: dict[str, dict] = {
    c[1]: {"name": c[0], "slug": c[1], "lat": c[2], "lon": c[3]}
    for c in _cfg["cities"]
}

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AQI Prediction API", version="1.0")

_model = None
_median: dict = {}


@app.on_event("startup")
def _startup() -> None:
    global _model, _median
    _model, _median = load_artifacts()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_city(city_slug: str) -> str:
    """Return the validated slug or raise 404. Prevents SQL injection."""
    if city_slug not in KNOWN_CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: '{city_slug}'. Call /cities for the full list.")
    return city_slug


def _athena(sql: str) -> pd.DataFrame:
    import awswrangler as wr  # lazy import — keeps startup fast

    session = boto3.Session(
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )
    return wr.athena.read_sql_query(
        sql=sql,
        database="aqi_db",
        s3_output="s3://weather-bulk/athena-results/",
        boto3_session=session,
        ctas_approach=False,
    )


# ── Routes

@app.get("/cities", summary="List all cities with their coordinates")
def list_cities() -> list[dict]:
    return [
        {**city, "timezone": CITY_TIMEZONES.get(city["slug"], "UTC")}
        for city in KNOWN_CITIES.values()
    ]


@app.get("/predict/{city_slug}", summary="Predict next-hour AQI for a city")
def predict_city(city_slug: str) -> dict:
    """
    Fetches the two most recent rows for the city from Athena, builds
    the lag features, and returns the predicted AQI class for T+1.
    """
    slug = _validate_city(city_slug)
    df = _athena(f"""
        SELECT timestamp, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
        FROM aqi_db.aqi_unified
        WHERE city_slug = '{slug}'
        ORDER BY timestamp DESC
        LIMIT 2
    """)
    if len(df) < 2:
        raise HTTPException(422, "Not enough historical rows to compute lag features")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").clip(upper=5)
    rows = df.to_dict("records")

    X      = build_feature_vector(rows[0], rows[1], _median)
    result = predict_single(_model, X)

    current_aqi = int(rows[1]["aqi"])
    return {
        "city":          KNOWN_CITIES[slug]["name"],
        "city_slug":     slug,
        "timezone":      CITY_TIMEZONES.get(slug, "UTC"),
        "as_of":         str(rows[1]["timestamp"]),
        "current_aqi":   current_aqi,
        "current_label": AQI_META.get(current_aqi, {}).get("label", "—"),
        "current_color": AQI_META.get(current_aqi, {}).get("color", "#ccc"),
        "next_hour":     result,
    }


@app.get("/history/{city_slug}", summary="Predicted vs actual AQI history")
def city_history(
    city_slug: str,
    hours: int = Query(default=48, ge=1, le=168, description="Look-back window in hours (max 168 = 7 days)"),
) -> list[dict]:
    """
    Returns a time-ordered list of {timestamp, predicted, actual, current_aqi}.
    `actual` is the ground-truth AQI for the hour after `timestamp`.
    """
    slug  = _validate_city(city_slug)
    limit = hours + 2   # extra rows needed for lag computation

    df = _athena(f"""
        SELECT timestamp, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
        FROM aqi_db.aqi_unified
        WHERE city_slug = '{slug}'
        ORDER BY timestamp DESC
        LIMIT {limit}
    """)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").clip(upper=5)
    rows = df.to_dict("records")
    return batch_predict(_model, _median, rows)


@app.get("/drift/{city_slug}", summary="Rolling weekly drift: current 7 days vs prior 7 days")
def city_drift(city_slug: str) -> dict:
    """
    Rolling weekly drift monitor.

    Compares the distribution of raw pollutant features between two
    non-overlapping 7-day windows:
      - reference : days 14 → 7 ago  (prior week)
      - recent    : last 7 days       (current week)

    Returns per-feature z-score drift and AQI class distribution shift.
    Retrain with ``python ml/train.py --lookback-weeks N`` when |z| > 1.0
    or the AQI distribution has shifted significantly.
    """
    slug = _validate_city(city_slug)

    df = _athena(f"""
        SELECT timestamp, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
        FROM aqi_db.aqi_unified
        WHERE city_slug = '{slug}'
          AND timestamp >= current_timestamp - interval '14' day
        ORDER BY timestamp ASC
    """)
    if len(df) < 50:
        raise HTTPException(422, "Not enough data for drift analysis (need ≥ 50 rows over the last 14 days)")

    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["aqi"] = df["aqi"].clip(upper=5)

    # Fixed rolling split: reference = days 14→7 ago, recent = last 7 days
    cutoff    = df["timestamp"].max() - pd.Timedelta(days=7)
    ref_df    = df[df["timestamp"] <  cutoff]
    recent_df = df[df["timestamp"] >= cutoff]

    if len(ref_df) < 10 or len(recent_df) < 10:
        raise HTTPException(422, "One of the 7-day windows has too few rows for drift analysis")

    feature_cols = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
    features_out: dict = {}
    for col in feature_cols:
        ref_v = ref_df[col].dropna()
        rec_v = recent_df[col].dropna()
        if len(ref_v) < 2:
            continue
        ref_mean = float(ref_v.mean())
        ref_std  = max(float(ref_v.std()), 1e-9)
        rec_mean = float(rec_v.mean())
        features_out[col] = {
            "ref_mean":    round(ref_mean, 4),
            "recent_mean": round(rec_mean, 4),
            "drift_score": round((rec_mean - ref_mean) / ref_std, 3),
        }

    ref_dist = ref_df["aqi"].dropna().astype(int).value_counts(normalize=True)
    rec_dist = recent_df["aqi"].dropna().astype(int).value_counts(normalize=True)
    aqi_dist = {
        str(cls): {
            "ref_pct":    round(float(ref_dist.get(cls, 0)) * 100, 1),
            "recent_pct": round(float(rec_dist.get(cls, 0)) * 100, 1),
        }
        for cls in [1, 2, 3, 4, 5]
    }

    return {
        "city":             KNOWN_CITIES[slug]["name"],
        "timezone":         CITY_TIMEZONES.get(slug, "UTC"),
        "ref_rows":         len(ref_df),
        "recent_rows":      len(recent_df),
        "ref_window":       f"{ref_df['timestamp'].min()} → {ref_df['timestamp'].max()}",
        "recent_window":    f"{recent_df['timestamp'].min()} → {recent_df['timestamp'].max()}",
        "features":         features_out,
        "aqi_distribution": aqi_dist,
    }


@app.get("/metrics/{city_slug}", summary="Online F1 / Precision / Recall from recent production data")
def city_metrics(
    city_slug: str,
    hours: int = Query(default=168, ge=24, le=336, description="Look-back window in hours (max 336=14 days)"),
) -> dict:
    """
    Computes Precision, Recall, F1 by comparing model T+1 predictions against
    real Athena ground-truth labels for the most recent N hours.
    Pure online / production data — the training set is never touched.
    Requires >= 10 rows with known actuals.
    """
    slug  = _validate_city(city_slug)
    limit = hours + 2

    df = _athena(f"""
        SELECT timestamp, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
        FROM aqi_db.aqi_unified
        WHERE city_slug = '{slug}'
        ORDER BY timestamp DESC
        LIMIT {limit}
    """)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").clip(upper=5)
    rows = df.to_dict("records")
    predictions = batch_predict(_model, _median, rows)

    valid = [
        (r["predicted"], r["actual"])
        for r in predictions
        if r["actual"] is not None
    ]
    if len(valid) < 10:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(valid)} labelled rows — need ≥ 10 to compute metrics.",
        )

    y_pred  = [p for p, _ in valid]
    y_true  = [a for _, a in valid]
    labels  = [1, 2, 3, 4, 5]

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )
    prec_cls, rec_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    return {
        "city":          KNOWN_CITIES[slug]["name"],
        "window_hours":  hours,
        "n_predictions": len(valid),
        "weighted": {
            "f1":        round(float(f1_w),   4),
            "precision": round(float(prec_w), 4),
            "recall":    round(float(rec_w),  4),
        },
        "per_class": {
            str(cls): {
                "f1":        round(float(f1_cls[i]),   4),
                "precision": round(float(prec_cls[i]), 4),
                "recall":    round(float(rec_cls[i]),  4),
                "support":   int(sup_cls[i]),
            }
            for i, cls in enumerate(labels)
        },
        "computed_at": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


@app.post("/reload-model", summary="Hot-reload model artifacts without restarting the server")
def reload_model() -> dict:
    """
    Re-reads model.ubj and median.json from ml/model-registry/ into memory.
    Call this after running `python ml/train.py` to apply the new model immediately.
    """
    global _model, _median
    _model, _median = load_artifacts()
    return {"status": "ok", "message": "Model reloaded successfully"}


@app.get("/health", summary="System and model metadata (no Athena)")
def health() -> dict:
    """
    Returns model registry info and system stats without calling Athena.
    Used by the Dashboard tab for instant status display.
    """
    model_path = ROOT / "ml" / "model-registry" / "model.ubj"
    if model_path.exists():
        mtime = datetime.datetime.fromtimestamp(model_path.stat().st_mtime, tz=datetime.timezone.utc)
        trained_on = mtime.strftime("%Y-%m-%d %H:%M UTC")
    else:
        # try the directory with trailing space (legacy)
        alt = ROOT / "ml" / "model-registry " / "model.ubj"
        if alt.exists():
            mtime = datetime.datetime.fromtimestamp(alt.stat().st_mtime, tz=datetime.timezone.utc)
            trained_on = mtime.strftime("%Y-%m-%d %H:%M UTC")
        else:
            trained_on = "Not found — run python ml/train.py"

    return {
        "status":         "ok",
        "model_loaded":   _model is not None,
        "model_type":     "XGBoost multi:softprob",
        "n_classes":      5,
        "n_features":     12,
        "target":         "AQI @ T+1 (next hour)",
        "model_trained_on": trained_on,
        "n_cities":       len(KNOWN_CITIES),
        "server_time_utc": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    html = (Path(__file__).parent / "templates" / "index.html").read_text()
    return HTMLResponse(html)
