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
import json
import os
import threading
import time
from pathlib import Path

import boto3
import pandas as pd
import redis as redis_lib
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sklearn.metrics import precision_recall_fscore_support

from . import telemetry as _tel
from .inference import (
    AQI_META,
    CITY_TIMEZONES,
    FEATURES,
    batch_predict,
    build_feature_vector,
    load_artifacts,
    predict_single,
)
from .telemetry import setup_tracing as _setup_tracing

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

# ── Observability: tracing (must be before ASGI stack builds) ────────────────
# FastAPIInstrumentor.instrument_app() adds middleware — must run at import time,
# NOT inside a startup event handler (middleware is frozen by then).
_setup_tracing(app)

# ── Observability: Prometheus metrics ─────────────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    # HTTP request metrics (latency, count, status) — auto-instrumented per route
    # Still exposed on /metrics for local debugging
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/favicon.ico"],
    ).instrument(app).expose(app, include_in_schema=False)
except ImportError:
    pass

# ML-specific custom metrics — pushed to Grafana Cloud via OTLP (see telemetry.py).
# Access via module attribute so setup_metrics_push() can swap _Noop → real instruments.

_model  = None
_median: dict = {}
_redis: "redis_lib.Redis | None" = None

CACHE_TTL = 7_200   # 2 hours — safety net; Redis is proactively refreshed every hour by Lambda B via /warm-cache


def _cache_get(key: str):
    if _redis is None:
        return None
    try:
        val = _redis.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


def _cache_set(key: str, value) -> bool:
    """Returns True on success, False on failure (also logs the error)."""
    if _redis is None:
        return False
    try:
        _redis.setex(key, CACHE_TTL, json.dumps(value, default=str))
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Redis _cache_set failed for %s: %s", key, e)
        return False


def _cache_clear(pattern: str = "aqi:*") -> None:
    if _redis is None:
        return
    try:
        keys = list(_redis.scan_iter(pattern))
        if keys:
            _redis.delete(*keys)
    except Exception:
        pass


@app.on_event("startup")
def _startup() -> None:
    global _model, _median, _redis
    _model, _median = load_artifacts()
    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url:
        try:
            _redis = redis_lib.Redis.from_url(
                redis_url, decode_responses=True, socket_connect_timeout=2
            )
            _redis.ping()
        except Exception:
            _redis = None
    _cache_feature_importance()

    # Observability — direct push to Grafana Cloud (no-op when env vars not set)
    from .telemetry import setup_logging, setup_metrics_push, setup_system_metrics
    setup_metrics_push()
    setup_system_metrics()   # CPU, memory — must come after setup_metrics_push()
    setup_logging()

    # Self-warming background loop — keeps Redis populated independently of Lambda B.
    # Runs every WARM_INTERVAL seconds (default 3600 = 1 hour).
    # First warm happens 10s after startup so the cache is hot immediately.
    _warm_interval = int(os.environ.get("WARM_INTERVAL_SEC", "3600"))
    threading.Thread(target=_warm_loop, args=(_warm_interval,), daemon=True).start()


def _warm_loop(interval: int) -> None:
    """Periodically call warm_cache() to keep Redis populated."""
    time.sleep(10)  # let startup finish, model load, etc.
    while True:
        try:
            print("[self-warm] triggering warm_cache()…", flush=True)
            result = warm_cache()
            print(f"[self-warm] done — {result.get('cities_cached')} cities, {result.get('keys_written')} keys", flush=True)
        except Exception as e:
            print(f"[self-warm] failed — {e}", flush=True)
        time.sleep(interval)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cache_feature_importance() -> None:
    """Compute feature importance from the loaded model and write to Redis."""
    if _model is None:
        return
    try:
        booster = _model.get_booster()
        weight  = booster.get_score(importance_type="weight")
        gain    = booster.get_score(importance_type="gain")
        cover   = booster.get_score(importance_type="cover")
        # XGBoost uses internal names f0, f1, … when feature_names aren't set.
        # Map them back to the canonical FEATURES list by index.
        fmap = {f"f{i}": name for i, name in enumerate(FEATURES)}
        def _remap(d: dict) -> dict:
            return {fmap.get(k, k): v for k, v in d.items()}
        weight = _remap(weight)
        gain   = _remap(gain)
        cover  = _remap(cover)
        # Normalise each type to 0-1 so charts are comparable
        def _norm(d: dict) -> dict:
            total = sum(d.values()) or 1
            return {k: round(v / total, 4) for k, v in d.items()}
        payload = {
            "features":    FEATURES,
            "weight":      _norm(weight),
            "gain":        _norm(gain),
            "cover":       _norm(cover),
            "computed_at": __import__("datetime").datetime.now(
                tz=__import__("datetime").timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
        }
        _cache_set("aqi:feature_importance", payload)
        # Append snapshot to history list (capped at 30 entries = ~1 month of daily retrains)
        if _redis is not None:
            try:
                _redis.rpush("aqi:feature_importance:history", json.dumps(payload, default=str))
                _redis.ltrim("aqi:feature_importance:history", -30, -1)
            except Exception:
                pass
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("feature importance cache failed: %s", e)


def _validate_city(city_slug: str) -> str:
    """Return the validated slug or raise 404. Prevents SQL injection."""
    if city_slug not in KNOWN_CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: '{city_slug}'. Call /cities for the full list.")
    return city_slug


def _athena(sql: str) -> pd.DataFrame:
    import awswrangler as wr  # lazy import — keeps startup fast

    _tel.METRIC_ATHENA_QUERIES.add(1)
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
def predict_city(city_slug: str) -> JSONResponse:
    """
    Fetches the two most recent rows for the city from Athena, builds
    the lag features, and returns the predicted AQI class for T+1.
    Result is cached in Redis for 2 hours (data only updates hourly).
    """
    slug      = _validate_city(city_slug)
    cache_key = f"aqi:predict:{slug}"
    cached    = _cache_get(cache_key)
    if cached:
        _tel.METRIC_CACHE_OPS.add(1, {"endpoint": "predict", "result": "hit"})
        _tel.METRIC_PREDICTIONS.add(1, {"city": slug, "predicted_class": str(cached.get("next_hour", {}).get("predicted_aqi", 0)), "cache": "hit"})
        return JSONResponse(cached, headers={"X-Cache": "HIT"})

    _tel.METRIC_CACHE_OPS.add(1, {"endpoint": "predict", "result": "miss"})
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

    current_aqi  = int(rows[1]["aqi"])
    as_of_ts     = pd.Timestamp(rows[1]["timestamp"])
    forecast_for = as_of_ts + pd.Timedelta(hours=1)
    payload = {
        "city":          KNOWN_CITIES[slug]["name"],
        "city_slug":     slug,
        "timezone":      CITY_TIMEZONES.get(slug, "UTC"),
        "as_of":         str(as_of_ts),
        "forecast_for":  str(forecast_for),
        "current_aqi":   current_aqi,
        "current_label": AQI_META.get(current_aqi, {}).get("label", "—"),
        "current_color": AQI_META.get(current_aqi, {}).get("color", "#ccc"),
        "next_hour":     result,
    }
    _tel.METRIC_PREDICTIONS.add(1, {"city": slug, "predicted_class": str(result["predicted_aqi"]), "cache": "miss"})
    _cache_set(cache_key, payload)
    return JSONResponse(payload, headers={"X-Cache": "MISS"})


@app.get("/history/{city_slug}", summary="Predicted vs actual AQI history")
def city_history(
    city_slug: str,
    hours: int = Query(default=48, ge=1, le=168, description="Look-back window in hours (max 168 = 7 days)"),
) -> list[dict]:
    """
    Returns a time-ordered list of {timestamp, predicted, actual, current_aqi}.
    `actual` is the ground-truth AQI for the hour after `timestamp`.
    Result is cached in Redis for 5 hours.
    """
    slug      = _validate_city(city_slug)
    cache_key = f"aqi:history:{slug}:{hours}"
    cached    = _cache_get(cache_key)
    if cached:
        return JSONResponse(cached, headers={"X-Cache": "HIT"})

    fetch = hours + 2   # extra rows needed for lag computation
    df = _athena(f"""
        SELECT timestamp, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
        FROM aqi_db.aqi_unified
        WHERE city_slug = '{slug}'
          AND timestamp >= current_timestamp - interval '{fetch}' hour
        ORDER BY timestamp DESC
    """)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").clip(upper=5)
    rows   = df.to_dict("records")
    result = batch_predict(_model, _median, rows)
    _cache_set(cache_key, result)
    return JSONResponse(result, headers={"X-Cache": "MISS"})


def _build_drift_payload(slug: str, df: pd.DataFrame, window_days: int = 1,
                         model=None, median=None,
                         display_name: str | None = None) -> dict | None:
    """
    Compute drift from a pre-fetched DataFrame.
    `window_days=1` compares today vs yesterday; `window_days=7` compares this week vs prior week.
    `df` must include columns: timestamp, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3.
    Pass `model` and `median` to also compute prediction distribution drift.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["aqi"] = df["aqi"].clip(upper=5)
    # Computed feature: pm25_ratio (same formula used at inference time)
    df["pm25_ratio"] = df["pm2_5"] / (df["pm2_5"] + df["pm10"] + df["no2"] + df["o3"] + df["so2"] + 1e-9)
    # Lag features: match exact model inputs (aqi_lag1, pm10_lag1 = T-1 values)
    df["aqi_lag1"]     = df["aqi"].shift(1)
    df["pm10_lag1"]    = df["pm10"].shift(1)
    # Delta features: momentum (T - T-1)
    df["aqi_delta_1h"]  = df["aqi"]  - df["aqi_lag1"]
    df["pm10_delta_1h"] = df["pm10"] - df["pm10_lag1"]

    if len(df) < 10:
        return None

    cutoff    = df["timestamp"].max() - pd.Timedelta(days=window_days)
    ref_df    = df[df["timestamp"] <  cutoff]
    recent_df = df[df["timestamp"] >= cutoff]

    if len(ref_df) < 5 or len(recent_df) < 5:
        return None

    ref_label    = "yesterday"    if window_days == 1 else "prior 7 days"
    recent_label = "today"        if window_days == 1 else "last 7 days"

    # Track driftable model features.
    # Delta features (aqi_delta_1h, pm10_delta_1h) are intentionally excluded:
    # their mean is always ~0 (mean-reverting by definition), so z-score drift
    # is always ~0 regardless of actual distribution changes. They are model
    # features but not drift-monitorable via z-score on means.
    # hour/month sin/cos omitted — deterministic, cannot drift.
    feature_cols = ["pm10_lag1", "pm25_ratio", "co", "no", "no2", "o3", "so2", "nh3", "pm10"]
    features_out: dict = {}
    for col in feature_cols:
        ref_v = ref_df[col].dropna()
        rec_v = recent_df[col].dropna()
        if len(ref_v) < 2:
            continue
        ref_mean = float(ref_v.mean())
        # Use a relative floor (1% of scale) so near-constant distributions
        # don't produce astronomically large z-scores via a near-zero std.
        ref_std  = max(float(ref_v.std()), 0.01 * max(abs(ref_mean), 1.0))
        rec_mean = float(rec_v.mean())
        raw_score = (rec_mean - ref_mean) / ref_std
        features_out[col] = {
            "ref_mean":    round(ref_mean, 4),
            "recent_mean": round(rec_mean, 4),
            "drift_score": round(max(-10.0, min(10.0, raw_score)), 3),
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

    # Prediction distribution drift
    pred_dist = None
    if model is not None and median is not None:
        from collections import Counter
        ref_rows_list = ref_df.sort_values("timestamp").to_dict("records")
        rec_rows_list = recent_df.sort_values("timestamp").to_dict("records")
        ref_preds = batch_predict(model, median, ref_rows_list) if len(ref_rows_list) >= 3 else []
        rec_preds = batch_predict(model, median, rec_rows_list) if len(rec_rows_list) >= 3 else []
        if ref_preds and rec_preds:
            ref_counts = Counter(r["predicted"] for r in ref_preds)
            rec_counts = Counter(r["predicted"] for r in rec_preds)
            pred_dist = {
                str(cls): {
                    "ref_pct":    round(ref_counts.get(cls, 0) / len(ref_preds) * 100, 1),
                    "recent_pct": round(rec_counts.get(cls, 0) / len(rec_preds) * 100, 1),
                }
                for cls in [1, 2, 3, 4, 5]
            }

    return {
        "city":                    display_name if display_name else KNOWN_CITIES[slug]["name"],
        "timezone":                "UTC" if display_name else CITY_TIMEZONES.get(slug, "UTC"),
        "ref_rows":                len(ref_df),
        "recent_rows":             len(recent_df),
        "ref_window":              f"{ref_df['timestamp'].min()} to {ref_df['timestamp'].max()} ({ref_label})",
        "recent_window":           f"{recent_df['timestamp'].min()} to {recent_df['timestamp'].max()} ({recent_label})",
        "features":                features_out,
        "aqi_distribution":        aqi_dist,
        "prediction_distribution": pred_dist,
    }


@app.get("/drift/{city_slug}", summary="Drift monitor: today vs yesterday (1d) or this week vs prior week (7d)")
def city_drift(
    city_slug: str,
    window: str = Query(default="1d", pattern="^(1d|7d)$", description="Comparison window: '1d' (today vs yesterday) or '7d' (this week vs prior week)"),
) -> dict:
    """
    Drift monitor comparing two non-overlapping windows.

    window=1d: today (last 24h) vs yesterday (24–48h ago)
    window=7d: last 7 days vs prior 7 days

    Returns per-feature z-score drift and AQI class distribution shift.
    Retrain when |z| > 1.0 or the AQI distribution has shifted significantly.
    Result is served from Redis when pre-loaded by /warm-cache.
    """
    slug        = _validate_city(city_slug)
    cache_key   = f"aqi:drift:{slug}:{window}"
    cached      = _cache_get(cache_key)
    if cached:
        return JSONResponse(cached, headers={"X-Cache": "HIT"})

    window_days = 1 if window == "1d" else 7
    interval    = "2" if window == "1d" else "14"
    df = _athena(f"""
        SELECT timestamp, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
        FROM aqi_db.aqi_unified
        WHERE city_slug = '{slug}'
          AND timestamp >= current_timestamp - interval '{interval}' day
        ORDER BY timestamp ASC
    """)

    payload = _build_drift_payload(slug, df, window_days=window_days, model=_model, median=_median)
    if payload is None:
        raise HTTPException(422, "Not enough data for drift analysis (need >= 10 rows and >= 5 rows per window)")

    _cache_set(cache_key, payload)
    return JSONResponse(payload, headers={"X-Cache": "MISS"})


@app.get("/drift", summary="Global data drift — all cities pooled (model-level drift monitoring)")
def global_drift(
    window: str = Query(default="1d", pattern="^(1d|7d)$",
                        description="Comparison window: '1d' (today vs yesterday) or '7d' (this week vs prior week)"),
) -> dict:
    """
    Computes data drift across ALL 99 cities pooled together.
    Since there is one global model (not per-city), this is the correct level
    to detect model-level input distribution shift.
    Result is served from Redis when pre-loaded by /warm-cache.
    """
    cache_key = f"aqi:drift:global:{window}"
    cached    = _cache_get(cache_key)
    if cached:
        return JSONResponse(cached, headers={"X-Cache": "HIT"})

    window_days = 1 if window == "1d" else 7
    interval    = "2" if window == "1d" else "14"
    df = _athena(f"""
        SELECT timestamp, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
        FROM aqi_db.aqi_unified
        WHERE timestamp >= current_timestamp - interval '{interval}' day
        ORDER BY timestamp ASC
    """)

    payload = _build_drift_payload(
        "global", df, window_days=window_days,
        model=_model, median=_median,
        display_name="Global — All 99 Cities",
    )
    if payload is None:
        raise HTTPException(422, "Not enough global data for drift analysis (need >= 10 rows and >= 5 rows per window)")

    _cache_set(cache_key, payload)
    return JSONResponse(payload, headers={"X-Cache": "MISS"})


@app.get("/model-metrics", summary="Global online F1 / Precision / Recall — all 99 cities pooled")
def global_metrics(
    hours: int = Query(default=168, ge=24, le=168, description="Look-back window in hours (24h/48h/72h/168h=7d)"),
) -> dict:
    """
    Computes Precision, Recall, F1 by comparing model T+1 predictions against
    real Athena ground-truth labels for the most recent N hours — pooled across
    all 99 cities.  One model, one score.
    Served from Redis warm-cache; falls back to Athena on miss.
    """
    cache_key = f"aqi:model-metrics:global:{hours}"
    cached    = _cache_get(cache_key)
    if cached:
        return JSONResponse(cached, headers={"X-Cache": "HIT"})

    limit = hours + 2
    try:
        df = _athena(f"""
            SELECT timestamp, city_slug, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
            FROM aqi_db.aqi_unified
            WHERE timestamp >= current_timestamp - interval '{limit}' hour
            ORDER BY city_slug, timestamp ASC
        """)
    except Exception:
        raise HTTPException(status_code=503, detail="Athena query failed — try again shortly.")

    if df.empty:
        raise HTTPException(status_code=422, detail="No data available.")

    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").clip(upper=5)

    all_preds: list[dict] = []
    for _slug, city_df in df.groupby("city_slug"):
        rows = city_df.sort_values("timestamp").to_dict("records")
        if len(rows) >= 3:
            all_preds.extend(batch_predict(_model, _median, rows))

    valid = [
        (r["predicted"], r["actual"])
        for r in all_preds
        if r["actual"] is not None
    ]
    if len(valid) < 10:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(valid)} labelled rows globally — need ≥ 10 to compute metrics.",
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

    metrics_payload = {
        "city":          "Global — All 99 Cities",
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
    _cache_set(cache_key, metrics_payload)
    return JSONResponse(metrics_payload, headers={"X-Cache": "MISS"})


@app.get("/accuracy-trend", summary="Global accuracy trend — all 99 cities pooled (concept drift detection)")
def global_accuracy_trend(
    weeks: int = Query(default=8, ge=2, le=12, description="Number of weeks to look back"),
) -> dict:
    """Global week-over-week accuracy trend pooling all 99 cities.
    Mirrors /drift (global) for concept drift at the model level.
    Served from Redis warm-cache; falls back to Athena on miss.
    """
    cache_key = f"aqi:accuracy_trend:global:{weeks}"
    cached    = _cache_get(cache_key)
    if cached:
        return JSONResponse(cached, headers={"X-Cache": "HIT"})

    hours = weeks * 7 * 24 + 2
    try:
        df = _athena(f"""
            SELECT timestamp, city_slug, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
            FROM aqi_db.aqi_unified
            WHERE timestamp >= current_timestamp - interval '{hours}' hour
            ORDER BY city_slug, timestamp ASC
        """)
    except Exception:
        raise HTTPException(status_code=503, detail="Athena query failed — try again shortly.")

    if df.empty or len(df) < 10:
        raise HTTPException(status_code=422, detail="Not enough global historical data.")

    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").clip(upper=5)

    all_preds: list[dict] = []
    for _slug, city_df in df.groupby("city_slug"):
        rows = city_df.sort_values("timestamp").to_dict("records")
        if len(rows) >= 3:
            all_preds.extend(batch_predict(_model, _median, rows))

    valid = [r for r in all_preds if r["actual"] is not None and r.get("timestamp")]
    if not valid:
        raise HTTPException(status_code=422, detail="No labelled rows available.")

    import collections
    daily: dict = collections.defaultdict(lambda: {"total": 0, "correct": 0})
    for r in valid:
        try:
            day = str(pd.Timestamp(r["timestamp"]).date())
        except Exception:
            continue
        daily[day]["total"]   += 1
        daily[day]["correct"] += int(r["predicted"] == r["actual"])

    if len(daily) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 days of data.")

    trend_rows = sorted(
        [{"day": d, "total": v["total"], "accuracy": round(v["correct"] / v["total"], 4)}
         for d, v in daily.items()],
        key=lambda x: x["day"],
    )
    accuracies  = [r["accuracy"] for r in trend_rows]
    last_week   = float(pd.Series(accuracies[-7:]).mean())
    prior_week  = float(pd.Series(accuracies[-14:-7]).mean()) if len(accuracies) >= 14 else None
    wow_delta   = round(last_week - prior_week, 4) if prior_week is not None else None

    payload = {
        "city":               "Global — All 99 Cities",
        "weeks":              weeks,
        "trend":              trend_rows,
        "last_7d_accuracy":   round(last_week, 4),
        "prior_7d_accuracy":  round(prior_week, 4) if prior_week is not None else None,
        "wow_delta":          wow_delta,
        "concept_drift_flag": wow_delta is not None and wow_delta < -0.10,
        "computed_at": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    _cache_set(cache_key, payload)
    return JSONResponse(payload, headers={"X-Cache": "MISS"})


@app.post("/warm-cache", summary="Bulk-load 7-day history for all cities into Redis (called by Lambda B after each hourly merge)")
def warm_cache() -> dict:
    """
    Single Athena query for ALL cities (last 170 hours) → batch_predict per city
    → populate Redis for history (24h/48h/72h/168h windows) and predict.
    Called automatically by Lambda B after each hourly MERGE so the dashboard
    is always served from Redis, never from on-demand Athena queries.

    All Redis writes are batched into a single pipeline flush to avoid
    ~1200 individual round-trips (would be ~26s at 22ms RTT each).
    """
    if _redis is None:
        return {"status": "skipped", "reason": "Redis not connected"}

    _t0 = time.monotonic()
    df = _athena("""
        SELECT timestamp, city_slug, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3
        FROM aqi_db.aqi_unified
        WHERE timestamp >= current_timestamp - interval '340' hour
        ORDER BY city_slug, timestamp ASC
    """)
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").clip(upper=5)

    # Collect all (key, serialised_value) pairs in memory — no Redis calls yet.
    # One pipeline flush at the end replaces ~1200 sequential SETEX round-trips.
    pipe_entries: list[tuple[str, str]] = []   # (key, json_str) for SETEX
    pred_buf_entries: list[str]         = []   # rows for RPUSH aqi:pred_buffer

    cached, skipped = [], []
    for slug, city_df in df.groupby("city_slug"):
        if slug not in KNOWN_CITIES:
            continue
        rows = city_df.sort_values("timestamp").to_dict("records")

        # History windows
        if len(rows) >= 3:
            history = batch_predict(_model, _median, rows)
            for h in (24, 48, 72, 168):
                window = history[-h:] if len(history) > h else history
                pipe_entries.append((f"aqi:history:{slug}:{h}", json.dumps(window, default=str)))

        # Predict
        if len(rows) >= 2:
            X = build_feature_vector(rows[-2], rows[-1], _median)
            result = predict_single(_model, X)
            current_aqi = max(1, min(5, int(float(rows[-1]["aqi"] or 1))))
            as_of_ts    = pd.Timestamp(rows[-1]["timestamp"])
            payload = {
                "city":          KNOWN_CITIES[slug]["name"],
                "city_slug":     slug,
                "timezone":      CITY_TIMEZONES.get(slug, "UTC"),
                "as_of":         str(as_of_ts),
                "forecast_for":  str(as_of_ts + pd.Timedelta(hours=1)),
                "current_aqi":   current_aqi,
                "current_label": AQI_META.get(current_aqi, {}).get("label", "—"),
                "current_color": AQI_META.get(current_aqi, {}).get("color", "#ccc"),
                "next_hour":     result,
            }
            pipe_entries.append((f"aqi:predict:{slug}", json.dumps(payload, default=str)))
            pred_buf_entries.append(json.dumps({
                "city_slug":    slug,
                "as_of":        str(as_of_ts),
                "forecast_for": str(as_of_ts + pd.Timedelta(hours=1)),
                "predicted":    result["predicted_aqi"],
                "confidence":   result["probabilities"][str(result["predicted_aqi"])],
            }, default=str))
            cached.append(slug)
        else:
            skipped.append(slug)

        # Drift windows (1d and 7d) from already-fetched data
        max_ts = city_df["timestamp"].max()
        for w_days, w_label in [(1, "1d"), (7, "7d")]:
            drift_df = city_df[city_df["timestamp"] >= max_ts - pd.Timedelta(days=w_days * 2)].copy()
            drift_payload = _build_drift_payload(slug, drift_df, window_days=w_days, model=_model, median=_median)
            if drift_payload is not None:
                pipe_entries.append((f"aqi:drift:{slug}:{w_label}", json.dumps(drift_payload, default=str)))

        # (per-city metrics removed — global metrics computed below after city loop)

    # Global metrics — pool ALL cities (one model, one score per window)
    from sklearn.metrics import precision_recall_fscore_support as _prfs
    for h in (24, 48, 72, 168):
        all_preds_h: list[tuple] = []
        for slug3, city_df3 in df.groupby("city_slug"):
            if slug3 not in KNOWN_CITIES:
                continue
            rows3 = city_df3.sort_values("timestamp").to_dict("records")
            limit3 = h + 2
            rows3 = rows3[-limit3:] if len(rows3) > limit3 else rows3
            if len(rows3) >= 3:
                preds3 = batch_predict(_model, _median, rows3)
                all_preds_h.extend(
                    (r["predicted"], r["actual"]) for r in preds3 if r["actual"] is not None
                )
        if len(all_preds_h) >= 10:
            y_pred_h = [p for p, _ in all_preds_h]
            y_true_h = [a for _, a in all_preds_h]
            labels_g = [1, 2, 3, 4, 5]
            prec_w_g, rec_w_g, f1_w_g, _ = _prfs(y_true_h, y_pred_h, average="weighted", labels=labels_g, zero_division=0)
            prec_cls_g, rec_cls_g, f1_cls_g, sup_cls_g = _prfs(y_true_h, y_pred_h, labels=labels_g, zero_division=0)
            global_metrics_payload = {
                "city":          "Global — All 99 Cities",
                "window_hours":  h,
                "n_predictions": len(all_preds_h),
                "weighted": {
                    "f1":        round(float(f1_w_g),   4),
                    "precision": round(float(prec_w_g), 4),
                    "recall":    round(float(rec_w_g),  4),
                },
                "per_class": {
                    str(cls): {
                        "f1":        round(float(f1_cls_g[i]),   4),
                        "precision": round(float(prec_cls_g[i]), 4),
                        "recall":    round(float(rec_cls_g[i]),  4),
                        "support":   int(sup_cls_g[i]),
                    }
                    for i, cls in enumerate(labels_g)
                },
                "computed_at": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
            pipe_entries.append((f"aqi:model-metrics:global:{h}", json.dumps(global_metrics_payload, default=str)))

    # Global accuracy trend — pool ALL cities (concept drift at model level)
    all_global_preds: list[dict] = []
    for slug2, city_df2 in df.groupby("city_slug"):
        if slug2 not in KNOWN_CITIES:
            continue
        rows2 = city_df2.sort_values("timestamp").to_dict("records")
        if len(rows2) >= 3:
            all_global_preds.extend(batch_predict(_model, _median, rows2))
    valid_global = [r for r in all_global_preds if r["actual"] is not None and r.get("timestamp")]
    if len(valid_global) >= 10:
        import collections as _col2
        g_daily: dict = _col2.defaultdict(lambda: {"total": 0, "correct": 0})
        for r in valid_global:
            try:
                day = str(pd.Timestamp(r["timestamp"]).date())
            except Exception:
                continue
            g_daily[day]["total"]   += 1
            g_daily[day]["correct"] += int(r["predicted"] == r["actual"])
        if len(g_daily) >= 2:
            g_trend = sorted(
                [{"day": d, "total": v["total"], "accuracy": round(v["correct"] / v["total"], 4)}
                 for d, v in g_daily.items()],
                key=lambda x: x["day"],
            )
            g_accs      = [t["accuracy"] for t in g_trend]
            g_last      = float(pd.Series(g_accs[-7:]).mean())
            g_prior     = float(pd.Series(g_accs[-14:-7]).mean()) if len(g_accs) >= 14 else None
            g_wow       = round(g_last - g_prior, 4) if g_prior is not None else None
            g_payload   = {
                "city":               "Global — All 99 Cities",
                "weeks":              8,
                "trend":              g_trend,
                "last_7d_accuracy":   round(g_last, 4),
                "prior_7d_accuracy":  round(g_prior, 4) if g_prior is not None else None,
                "wow_delta":          g_wow,
                "concept_drift_flag": g_wow is not None and g_wow < -0.10,
                "computed_at": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
            pipe_entries.append(("aqi:accuracy_trend:global:8", json.dumps(g_payload, default=str)))

    # Global drift — pool ALL cities together (model has one global training set;
    # this is the correct level to detect input distribution shift, not per-city).
    # The warm-cache Athena query (340h ≈ 14 days) covers both 1d and 7d windows.
    for w_days, w_label in [(1, "1d"), (7, "7d")]:
        global_drift_payload = _build_drift_payload(
            "global", df, window_days=w_days,
            model=_model, median=_median,
            display_name="Global — All 99 Cities",
        )
        if global_drift_payload is not None:
            pipe_entries.append((f"aqi:drift:global:{w_label}", json.dumps(global_drift_payload, default=str)))

    # Single pipeline flush — one TCP round-trip for all SETEX + RPUSH commands
    cache_errors = 0
    try:
        pipe = _redis.pipeline(transaction=False)
        for key, val in pipe_entries:
            pipe.setex(key, CACHE_TTL, val)
        if pred_buf_entries:
            pipe.rpush("aqi:pred_buffer", *pred_buf_entries)
        pipe.execute()
    except Exception as e:
        cache_errors = len(pipe_entries)
        import logging
        logging.getLogger(__name__).warning("warm-cache pipeline flush failed: %s", e)

    _tel.METRIC_WARM_CACHE.record(time.monotonic() - _t0)
    return {
        "status":         "ok" if cache_errors == 0 else "partial",
        "cities_cached":  len(cached),
        "cities_skipped": len(skipped),
        "rows_fetched":   len(df),
        "keys_written":   len(pipe_entries),
        "cache_errors":   cache_errors,
    }


@app.get("/feature-importance/history", summary="Feature importance trend — one snapshot per daily retrain, last 30 days")
def feature_importance_history() -> dict:
    """
    Returns up to 30 daily snapshots of feature importance (gain) so the dashboard
    can plot how each feature's influence changes over time after each retrain.
    """
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not connected")
    try:
        raw = _redis.lrange("aqi:feature_importance:history", 0, -1)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    snapshots = [json.loads(r) for r in raw]
    return {"snapshots": snapshots, "count": len(snapshots)}


@app.get("/feature-importance", summary="XGBoost feature importances (weight / gain / cover) — model-level, not city-specific")
def feature_importance() -> dict:
    """
    Returns normalised feature importance scores for all three XGBoost importance
    types (weight = split frequency, gain = avg loss reduction, cover = avg sample
    coverage). Served from Redis; computed once per model load/reload.
    """
    cached = _cache_get("aqi:feature_importance")
    if cached:
        return cached
    # Cold path — compute on the fly (e.g. first call before warm-cache)
    _cache_feature_importance()
    result = _cache_get("aqi:feature_importance")
    if result:
        return result
    raise HTTPException(status_code=503, detail="Model not loaded yet")


@app.post("/reload-model", summary="Hot-reload model artifacts without restarting the server")
def reload_model() -> dict:
    """
    Re-reads model.ubj and median.json from ml/model-registry/ into memory.
    Also flushes all Redis cache keys so the new model's predictions are served immediately.
    """
    global _model, _median
    _model, _median = load_artifacts()
    _cache_clear("aqi:*")  # invalidate all cached predictions/history
    _cache_feature_importance()  # recompute with new model weights
    # Re-warm the full cache in the background so requests aren't cold after a reload
    threading.Thread(target=warm_cache, daemon=True).start()
    return {"status": "ok", "message": "Model reloaded — cache re-warming in background"}


@app.get("/cache/status", summary="Redis cache info (ops use)")
def cache_status() -> dict:
    if _redis is None:
        return {"redis": "disabled", "reason": "REDIS_URL not set or connection failed"}
    try:
        info = _redis.info("memory")
        keys = _redis.dbsize()
        return {
            "redis":       "connected",
            "keys":        keys,
            "used_memory": info.get("used_memory_human", "?"),
            "ttl_seconds": CACHE_TTL,
        }
    except Exception as e:
        return {"redis": "error", "detail": str(e)}


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
        "n_features":     len(FEATURES),
        "target":         "AQI @ T+1 (next hour)",
        "model_trained_on": trained_on,
        "n_cities":       len(KNOWN_CITIES),
        "server_time_utc": datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    html = (Path(__file__).parent / "templates" / "index.html").read_text()
    cities = [
        {**city, "timezone": CITY_TIMEZONES.get(city["slug"], "UTC")}
        for city in KNOWN_CITIES.values()
    ]
    cities_json = json.dumps(cities)
    city_options = "\n".join(
        f'            <option value="{c["slug"]}">{c["name"]}</option>'
        for c in cities
    )
    html = html.replace('"__CITIES__"', cities_json)
    html = html.replace('            __CITY_OPTIONS__', city_options)
    return HTMLResponse(html)
