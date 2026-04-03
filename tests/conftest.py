"""
Shared pytest fixtures.

`ensure_test_artifacts` auto-creates a tiny trained model in
ml/model-registry/ when running in CI (where the real model file is absent).
This keeps all tests hermetic — no AWS, no real training data needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_test_artifacts():
    """
    If ml/model-registry/model.ubj is missing, build a minimal XGBoost model
    (100 synthetic rows, 5 classes, 15 features) so FastAPI startup and
    inference tests pass without the real production model.
    """
    reg = Path("ml/model-registry")
    needs_artifacts = not all(
        (reg / f).exists() for f in ("model.ubj", "median.json", "features.json")
    )
    if needs_artifacts:
        from xgboost import XGBClassifier

        reg.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        X = rng.random((100, 15)).astype("float32")
        y = rng.integers(0, 5, 100)

        m = XGBClassifier(
            n_estimators=10,
            max_depth=2,
            objective="multi:softprob",
            num_class=5,
            random_state=42,
            n_jobs=1,
        )
        m.fit(X, y)
        m.save_model(str(reg / "model.ubj"))

        features = [
            "pm10_lag1", "aqi_delta_1h", "pm10_delta_1h",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "pm25_ratio", "co", "no", "no2", "o3", "so2", "nh3", "pm10",
        ]
        (reg / "median.json").write_text(json.dumps({f: 1.0 for f in features}))
        (reg / "features.json").write_text(
            json.dumps({"features": features, "target": "aqi_next"}, indent=2)
        )
