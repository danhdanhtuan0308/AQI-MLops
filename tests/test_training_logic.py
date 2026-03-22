"""
Unit tests for ml/train.py — only pure data-transformation functions.

No Athena, no AWS, no model training invoked here.
`load_data()` and `main()` are excluded from tests because they require
live AWS credentials.  Everything else (feature engineering, array
preparation) is tested on synthetic in-memory DataFrames.
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from ml.train import FEATURES, TARGET, build_arrays, engineer_features


# ── Synthetic dataset factory ─────────────────────────────────────────────────

def _make_raw(n_cities: int = 2, n_hours: int = 24) -> pd.DataFrame:
    """Minimal DataFrame that mimics what Athena returns."""
    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    rows = []
    for c in range(n_cities):
        slug = f"city-{c}"
        for h in range(n_hours):
            rows.append({
                "timestamp": base + datetime.timedelta(hours=h),
                "city":      slug.replace("-", " ").title(),
                "city_slug": slug,
                "aqi":       (h % 5) + 1,
                "co":        300.0, "no": 2.0, "no2": 15.0,
                "o3":        60.0,  "so2": 5.0, "pm2_5": 10.0,
                "pm10":      20.0,  "nh3": 3.0,
                "source":    "test",
            })
    df = pd.DataFrame(rows)
    df["aqi"] = df["aqi"].astype("Int64")
    return df


# ── engineer_features ─────────────────────────────────────────────────────────

class TestEngineerFeatures:
    def test_all_model_features_present(self):
        feat_df = engineer_features(_make_raw())
        for col in FEATURES + [TARGET]:
            assert col in feat_df.columns, f"Missing column: {col}"

    def test_no_nan_in_feature_columns(self):
        feat_df = engineer_features(_make_raw())
        assert not feat_df[FEATURES].isnull().any().any()

    def test_no_nan_in_target(self):
        feat_df = engineer_features(_make_raw())
        assert not feat_df[TARGET].isnull().any()

    def test_target_is_integer_1_to_5(self):
        feat_df = engineer_features(_make_raw())
        assert feat_df[TARGET].between(1, 5).all()

    def test_hour_sin_cos_in_unit_circle(self):
        feat_df = engineer_features(_make_raw())
        assert feat_df["hour_sin"].between(-1.0, 1.0).all()
        assert feat_df["hour_cos"].between(-1.0, 1.0).all()

    def test_pm25_ratio_bounded(self):
        feat_df = engineer_features(_make_raw())
        assert feat_df["pm25_ratio"].between(0.0, 1.0).all()

    def test_no_cross_city_leakage(self):
        """
        After engineering, city A's lag should never borrow from city B.
        Verified by checking that the first timestamp per city was dropped
        (lag was undefined there and must have been removed).
        """
        raw = _make_raw(n_cities=3, n_hours=15)
        feat_df = engineer_features(raw)
        for slug in feat_df["city_slug"].unique():
            city_rows = feat_df[feat_df["city_slug"] == slug]
            raw_first = raw[raw["city_slug"] == slug]["timestamp"].min()
            # The very first raw row should not appear in the engineered set
            assert city_rows["timestamp"].min() > raw_first

    def test_row_count_reduced_by_two_per_city(self):
        """engineer_features drops first and last row per city."""
        n_cities, n_hours = 2, 20
        raw = _make_raw(n_cities=n_cities, n_hours=n_hours)
        feat_df = engineer_features(raw)
        # Each city loses 2 rows (lag + lead), so total = n_cities * (n_hours - 2)
        assert len(feat_df) == n_cities * (n_hours - 2)


# ── build_arrays ──────────────────────────────────────────────────────────────

class TestBuildArrays:
    @pytest.fixture(scope="class")
    def arrays(self):
        return build_arrays(engineer_features(_make_raw()))

    def test_X_shape(self, arrays):
        X, y, _ = arrays
        assert X.ndim == 2
        assert X.shape[1] == len(FEATURES)

    def test_y_shape_matches_X(self, arrays):
        X, y, _ = arrays
        assert len(y) == len(X)

    def test_X_dtype_float32(self, arrays):
        X, _, _ = arrays
        assert X.dtype == np.float32

    def test_y_is_zero_indexed(self, arrays):
        """XGBoost requires 0-4 labels, not 1-5."""
        _, y, _ = arrays
        assert y.min() >= 0
        assert y.max() <= 4

    def test_no_nan_in_X(self, arrays):
        X, _, _ = arrays
        assert not np.any(np.isnan(X))

    def test_median_keys_match_features(self, arrays):
        _, _, median = arrays
        assert set(median.index) == set(FEATURES)
