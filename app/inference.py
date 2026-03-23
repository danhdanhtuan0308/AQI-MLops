"""
Model loading and feature-building helpers for inference.
Loaded once at FastAPI startup; all functions are stateless after that.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
REGISTRY_DIR = Path(__file__).parent.parent / "ml" / "model-registry"

# ── Feature schema (must match ml/train.py exactly) ───────────────────────────
FEATURES: list[str] = [
    "aqi_lag1",    # AQI at T-1
    "pm10_lag1",   # PM10 at T-1
    "hour_sin",    # sin(2π·hour/24)
    "hour_cos",    # cos(2π·hour/24)
    "month_sin",   # sin(2π·month/12)
    "month_cos",   # cos(2π·month/12)
    "pm25_ratio",  # PM2.5 / Σ pollutants at T
    "co", "no", "no2", "o3", "so2", "nh3", "pm10",  # point-in-time pollutants at T
]

# ── AQI class metadata ────────────────────────────────────────────────────────
AQI_META: dict[int, dict] = {
    1: {"label": "Good",      "color": "#50f0e6"},
    2: {"label": "Fair",      "color": "#50ccaa"},
    3: {"label": "Moderate",  "color": "#f0e641"},
    4: {"label": "Poor",      "color": "#ff5050"},
    5: {"label": "Very Poor", "color": "#960032"},
}

# ── City → IANA timezone mapping ─────────────────────────────────────────────
CITY_TIMEZONES: dict[str, str] = {
    "tokyo":            "Asia/Tokyo",
    "delhi":            "Asia/Kolkata",
    "shanghai":         "Asia/Shanghai",
    "sao-paulo":        "America/Sao_Paulo",
    "mexico-city":      "America/Mexico_City",
    "cairo":            "Africa/Cairo",
    "mumbai":           "Asia/Kolkata",
    "beijing":          "Asia/Shanghai",
    "dhaka":            "Asia/Dhaka",
    "osaka":            "Asia/Tokyo",
    "new-york":         "America/New_York",
    "karachi":          "Asia/Karachi",
    "buenos-aires":     "America/Argentina/Buenos_Aires",
    "chongqing":        "Asia/Shanghai",
    "istanbul":         "Europe/Istanbul",
    "kolkata":          "Asia/Kolkata",
    "manila":           "Asia/Manila",
    "lagos":            "Africa/Lagos",
    "rio-de-janeiro":   "America/Sao_Paulo",
    "tianjin":          "Asia/Shanghai",
    "kinshasa":         "Africa/Kinshasa",
    "guangzhou":        "Asia/Shanghai",
    "los-angeles":      "America/Los_Angeles",
    "moscow":           "Europe/Moscow",
    "shenzhen":         "Asia/Shanghai",
    "lahore":           "Asia/Karachi",
    "bangalore":        "Asia/Kolkata",
    "paris":            "Europe/Paris",
    "bogota":           "America/Bogota",
    "jakarta":          "Asia/Jakarta",
    "chennai":          "Asia/Kolkata",
    "lima":             "America/Lima",
    "bangkok":          "Asia/Bangkok",
    "seoul":            "Asia/Seoul",
    "nagoya":           "Asia/Tokyo",
    "hyderabad":        "Asia/Kolkata",
    "london":           "Europe/London",
    "tehran":           "Asia/Tehran",
    "chicago":          "America/Chicago",
    "chengdu":          "Asia/Shanghai",
    "nanjing":          "Asia/Shanghai",
    "wuhan":            "Asia/Shanghai",
    "ho-chi-minh-city": "Asia/Ho_Chi_Minh",
    "luanda":           "Africa/Luanda",
    "ahmedabad":        "Asia/Kolkata",
    "kuala-lumpur":     "Asia/Kuala_Lumpur",
    "xian":             "Asia/Shanghai",
    "hong-kong":        "Asia/Hong_Kong",
    "dongguan":         "Asia/Shanghai",
    "hangzhou":         "Asia/Shanghai",
    "foshan":           "Asia/Shanghai",
    "shenyang":         "Asia/Shanghai",
    "riyadh":           "Asia/Riyadh",
    "baghdad":          "Asia/Baghdad",
    "santiago":         "America/Santiago",
    "surat":            "Asia/Kolkata",
    "madrid":           "Europe/Madrid",
    "suzhou":           "Asia/Shanghai",
    "pune":             "Asia/Kolkata",
    "harbin":           "Asia/Shanghai",
    "houston":          "America/Chicago",
    "dallas":           "America/Chicago",
    "toronto":          "America/Toronto",
    "dar-es-salaam":    "Africa/Dar_es_Salaam",
    "miami":            "America/New_York",
    "belo-horizonte":   "America/Sao_Paulo",
    "singapore":        "Asia/Singapore",
    "philadelphia":     "America/New_York",
    "atlanta":          "America/New_York",
    "fukuoka":          "Asia/Tokyo",
    "khartoum":         "Africa/Khartoum",
    "barcelona":        "Europe/Madrid",
    "johannesburg":     "Africa/Johannesburg",
    "saint-petersburg": "Europe/Moscow",
    "qingdao":          "Asia/Shanghai",
    "jeddah":           "Asia/Riyadh",
    "abidjan":          "Africa/Abidjan",
    "zhengzhou":        "Asia/Shanghai",
    "nairobi":          "Africa/Nairobi",
    "alexandria":       "Africa/Cairo",
    "casablanca":       "Africa/Casablanca",
    "kabul":            "Asia/Kabul",
    "accra":            "Africa/Accra",
    "cape-town":        "Africa/Johannesburg",
    # cities below were missing from the original dict — would have fallen back to UTC
    "sydney":           "Australia/Sydney",
    "melbourne":        "Australia/Melbourne",
    "rome":             "Europe/Rome",
    "berlin":           "Europe/Berlin",
    "addis-ababa":      "Africa/Addis_Ababa",
    "yangon":           "Asia/Rangoon",
    "kathmandu":        "Asia/Kathmandu",
    "ankara":           "Europe/Istanbul",
    "athens":           "Europe/Athens",
    "taipei":           "Asia/Taipei",
    "amsterdam":        "Europe/Amsterdam",
    "dubai":            "Asia/Dubai",
    "caracas":          "America/Caracas",
    "guadalajara":      "America/Mexico_City",
    "monterrey":        "America/Monterrey",
}


# ── Artifact loading ──────────────────────────────────────────────────────────

def load_artifacts() -> tuple[XGBClassifier, dict]:
    """Load model and per-feature medians from ml/model-registry/."""
    if not (REGISTRY_DIR / "model.ubj").exists():
        raise FileNotFoundError(
            f"Model not found at {REGISTRY_DIR / 'model.ubj'}. "
            "Run `python ml/train.py` to train and save the model first."
        )
    model = XGBClassifier()
    model.load_model(str(REGISTRY_DIR / "model.ubj"))
    median: dict = json.loads((REGISTRY_DIR / "median.json").read_text())
    return model, median


# ── Feature building ──────────────────────────────────────────────────────────

def _safe(val, fallback: float = 0.0) -> float:
    """Coerce val to float; return fallback on NaN / None / error."""
    try:
        v = float(val)
        return v if not np.isnan(v) else fallback
    except (TypeError, ValueError):
        return fallback


def build_feature_vector(
    row_prev: dict,
    row_curr: dict,
    median: dict,
) -> np.ndarray:
    """
    Build the 14-feature (1, 14) float32 array for a single (T-1, T) pair.

    row_prev  — Athena row at T-1: needs 'aqi', 'pm10'
    row_curr  — Athena row at T:   needs 'timestamp', pollutant columns
    median    — fallback dict keyed by feature name
    """
    ts    = row_curr["timestamp"]
    hour  = ts.hour  if hasattr(ts, "hour")  else int(str(ts)[11:13])
    month = ts.month if hasattr(ts, "month") else int(str(ts)[5:7])

    pm2_5 = _safe(row_curr.get("pm2_5"), median.get("pm25_ratio", 0))
    pm10  = _safe(row_curr.get("pm10"),  median.get("pm10", 0))
    no2   = _safe(row_curr.get("no2"),   median.get("no2", 0))
    o3    = _safe(row_curr.get("o3"),    median.get("o3", 0))
    so2   = _safe(row_curr.get("so2"),   median.get("so2", 0))

    pm25_ratio = pm2_5 / (pm2_5 + pm10 + no2 + o3 + so2 + 1e-9)

    feat = {
        "aqi_lag1":   _safe(row_prev.get("aqi"),  median.get("aqi_lag1", 2)),
        "pm10_lag1":  _safe(row_prev.get("pm10"), median.get("pm10_lag1", 0)),
        "hour_sin":   float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos":   float(np.cos(2 * np.pi * hour / 24)),
        "month_sin":  float(np.sin(2 * np.pi * month / 12)),
        "month_cos":  float(np.cos(2 * np.pi * month / 12)),
        "pm25_ratio": round(pm25_ratio, 4),
        "co":  _safe(row_curr.get("co"),  median.get("co", 0)),
        "no":  _safe(row_curr.get("no"),  median.get("no", 0)),
        "no2": no2,
        "o3":  o3,
        "so2": so2,
        "nh3": _safe(row_curr.get("nh3"), median.get("nh3", 0)),
        "pm10": pm10,
    }
    return np.array([[feat[f] for f in FEATURES]], dtype="float32")


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_single(model: XGBClassifier, X: np.ndarray) -> dict:
    """
    Run the model on a (1, 12) feature array.
    Returns predicted class (1-5), label, color, and per-class probabilities.
    """
    proba = model.predict_proba(X)[0]           # shape (5,)
    cls   = int(np.argmax(proba)) + 1           # 1-indexed
    meta  = AQI_META[cls]
    return {
        "predicted_aqi": cls,
        "label":         meta["label"],
        "color":         meta["color"],
        "probabilities": {str(i + 1): round(float(p), 4) for i, p in enumerate(proba)},
    }


def batch_predict(
    model: XGBClassifier,
    median: dict,
    rows: list[dict],
) -> list[dict]:
    """
    Given a time-sorted list of raw Athena rows, build the feature matrix
    for every consecutive (T-1, T) window and return predicted vs actual AQI.

    Prediction at position i  →  predicted AQI for rows[i+1]
    Ground truth               →  rows[i+2]["aqi"]  (actual T+1)

    Returns list of {timestamp, predicted, actual, current_aqi}.
    """
    if len(rows) < 3:
        return []

    # Build the full feature matrix in one pass
    vectors = [
        build_feature_vector(rows[i - 1], rows[i], median)[0]
        for i in range(1, len(rows) - 1)
    ]
    X_batch     = np.array(vectors, dtype="float32")
    proba_batch = model.predict_proba(X_batch)          # (N, 5)
    preds       = np.argmax(proba_batch, axis=1) + 1    # 1-indexed

    results = []
    for i, pred in enumerate(preds):
        row_curr = rows[i + 1]
        row_next = rows[i + 2]
        try:
            actual = max(1, min(5, int(float(row_next["aqi"]))))
        except (TypeError, ValueError):
            actual = None
        try:
            current = max(1, min(5, int(float(row_curr["aqi"]))))
        except (TypeError, ValueError):
            current = None

        results.append({
            "timestamp":   str(row_next["timestamp"]),  # T+1: the hour being predicted
            "predicted":   int(pred),
            "actual":      actual,
            "current_aqi": current,
        })
    return results
