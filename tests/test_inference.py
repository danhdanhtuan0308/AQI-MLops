"""
Unit tests for app/inference.py.

All tests are pure Python — no AWS, no Athena, no disk I/O beyond
the tiny model created by the conftest session fixture.
"""
from __future__ import annotations

import datetime

import numpy as np
import pytest

from app.inference import (
    FEATURES,
    batch_predict,
    build_feature_vector,
    load_artifacts,
    predict_single,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_MEDIAN = {f: 1.0 for f in FEATURES}


def _row(hour: int = 12, aqi: int = 2, pm10: float = 20.0, pm2_5: float = 10.0) -> dict:
    ts = datetime.datetime(2026, 3, 1, hour, 0, tzinfo=datetime.timezone.utc)
    return {
        "timestamp": ts, "aqi": aqi, "pm10": pm10, "pm2_5": pm2_5,
        "co": 300.0, "no": 2.0, "no2": 15.0, "o3": 60.0, "so2": 5.0, "nh3": 3.0,
    }


# ── build_feature_vector ──────────────────────────────────────────────────────

class TestBuildFeatureVector:
    def test_output_shape(self):
        X = build_feature_vector(_row(), _row(), _MEDIAN)
        assert X.shape == (1, len(FEATURES))

    def test_dtype_float32(self):
        X = build_feature_vector(_row(), _row(), _MEDIAN)
        assert X.dtype == np.float32

    def test_no_nan(self):
        X = build_feature_vector(_row(), _row(), _MEDIAN)
        assert not np.any(np.isnan(X))

    def test_hour_sin_cos_at_midnight(self):
        X = build_feature_vector(_row(hour=0), _row(hour=0), _MEDIAN)
        sin_idx = FEATURES.index("hour_sin")
        cos_idx = FEATURES.index("hour_cos")
        assert abs(X[0, sin_idx]) < 1e-5          # sin(0) ≈ 0
        assert abs(X[0, cos_idx] - 1.0) < 1e-5    # cos(0) ≈ 1

    def test_hour_sin_cos_at_noon(self):
        X = build_feature_vector(_row(hour=12), _row(hour=12), _MEDIAN)
        sin_idx = FEATURES.index("hour_sin")
        cos_idx = FEATURES.index("hour_cos")
        assert abs(X[0, sin_idx]) < 1e-5           # sin(π) ≈ 0
        assert abs(X[0, cos_idx] + 1.0) < 1e-5     # cos(π) ≈ -1

    def test_pm25_ratio_bounded(self):
        X = build_feature_vector(_row(), _row(), _MEDIAN)
        idx = FEATURES.index("pm25_ratio")
        assert 0.0 <= X[0, idx] <= 1.0

    def test_none_value_falls_back_to_median(self):
        row = _row()
        row["co"] = None
        X = build_feature_vector(_row(), row, _MEDIAN)
        assert not np.any(np.isnan(X))

    def test_nan_value_falls_back_to_median(self):
        row = _row()
        row["no2"] = float("nan")
        X = build_feature_vector(_row(), row, _MEDIAN)
        assert not np.any(np.isnan(X))


# ── predict_single ────────────────────────────────────────────────────────────

class TestPredictSingle:
    @pytest.fixture(scope="class")
    def model_median(self):
        return load_artifacts()

    def test_predicted_aqi_in_range(self, model_median):
        model, median = model_median
        X = build_feature_vector(_row(), _row(), median)
        result = predict_single(model, X)
        assert result["predicted_aqi"] in [1, 2, 3, 4, 5]

    def test_probabilities_sum_to_one(self, model_median):
        model, median = model_median
        X = build_feature_vector(_row(), _row(), median)
        result = predict_single(model, X)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_five_probability_keys(self, model_median):
        model, median = model_median
        X = build_feature_vector(_row(), _row(), median)
        result = predict_single(model, X)
        assert set(result["probabilities"].keys()) == {"1", "2", "3", "4", "5"}

    def test_label_and_color_present(self, model_median):
        model, median = model_median
        X = build_feature_vector(_row(), _row(), median)
        result = predict_single(model, X)
        assert isinstance(result["label"], str) and len(result["label"]) > 0
        assert result["color"].startswith("#")


# ── batch_predict ─────────────────────────────────────────────────────────────

class TestBatchPredict:
    @pytest.fixture(scope="class")
    def model_median(self):
        return load_artifacts()

    def test_too_few_rows_returns_empty(self, model_median):
        model, median = model_median
        assert batch_predict(model, median, []) == []
        assert batch_predict(model, median, [_row()]) == []
        assert batch_predict(model, median, [_row(), _row()]) == []

    def test_output_length(self, model_median):
        model, median = model_median
        n = 8
        rows = [_row(hour=i % 24, aqi=(i % 5) + 1) for i in range(n)]
        results = batch_predict(model, median, rows)
        assert len(results) == n - 2

    def test_result_schema(self, model_median):
        model, median = model_median
        rows = [_row(hour=i % 24, aqi=(i % 5) + 1) for i in range(6)]
        results = batch_predict(model, median, rows)
        for r in results:
            assert "timestamp" in r
            assert "predicted" in r
            assert "actual" in r
            assert "current_aqi" in r
            assert r["predicted"] in [1, 2, 3, 4, 5]

    def test_actual_matches_next_row_aqi(self, model_median):
        """actual at position i should equal aqi of rows[i+2]."""
        model, median = model_median
        rows = [_row(aqi=(i % 5) + 1) for i in range(5)]
        results = batch_predict(model, median, rows)
        for j, r in enumerate(results):
            expected_actual = (j + 2) % 5 + 1
            assert r["actual"] == expected_actual
