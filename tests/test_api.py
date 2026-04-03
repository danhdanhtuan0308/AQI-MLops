"""
Integration-style tests for the FastAPI endpoints.

Athena is fully mocked out — no AWS calls are made.
The model is loaded from ml/model-registry/ (created by conftest if absent).
"""
from __future__ import annotations

import datetime
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_athena_df(n: int = 15, aqi: int = 2) -> pd.DataFrame:
    """Minimal DataFrame that mimics _athena() output."""
    base = datetime.datetime(2026, 3, 1, 12, tzinfo=datetime.timezone.utc)
    return pd.DataFrame([
        {
            "timestamp": base + datetime.timedelta(hours=i),
            "aqi": aqi, "co": 300.0, "no": 2.0, "no2": 15.0,
            "o3": 60.0, "so2": 5.0, "pm2_5": 10.0, "pm10": 20.0, "nh3": 3.0,
        }
        for i in range(n)
    ])


@pytest.fixture(scope="module")
def client(ensure_test_artifacts):   # noqa: F811 — autouse ensures model exists first
    """
    TestClient with Athena stubbed out.
    The real model from ml/model-registry/ is loaded at startup — no mock needed.
    """
    import app.main as mod

    with patch("app.main._athena", side_effect=lambda sql: _make_athena_df()):
        with TestClient(mod.app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_ok(self, client):
        assert client.get("/health").status_code == 200

    def test_schema(self, client):
        d = client.get("/health").json()
        for key in ("status", "model_loaded", "model_type", "n_features", "n_classes", "n_cities"):
            assert key in d, f"Missing key: {key}"

    def test_model_is_loaded(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_n_features(self, client):
        assert client.get("/health").json()["n_features"] == 15


# ── /cities ───────────────────────────────────────────────────────────────────

class TestCities:
    def test_returns_list(self, client):
        r = client.get("/cities")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert len(r.json()) > 0

    def test_schema(self, client):
        first = client.get("/cities").json()[0]
        assert "slug" in first
        assert "name" in first

    def test_tokyo_present(self, client):
        slugs = [c["slug"] for c in client.get("/cities").json()]
        assert "tokyo" in slugs


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def test_unknown_city_404(self, client):
        assert client.get("/predict/not-a-real-city").status_code == 404

    def test_valid_city_200(self, client):
        assert client.get("/predict/tokyo").status_code == 200

    def test_predicted_aqi_in_range(self, client):
        d = client.get("/predict/tokyo").json()
        assert d["next_hour"]["predicted_aqi"] in [1, 2, 3, 4, 5]

    def test_probabilities_sum_to_one(self, client):
        d = client.get("/predict/tokyo").json()
        total = sum(d["next_hour"]["probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_five_probability_keys(self, client):
        d = client.get("/predict/tokyo").json()
        assert set(d["next_hour"]["probabilities"].keys()) == {"1", "2", "3", "4", "5"}


# ── /history ─────────────────────────────────────────────────────────────────

class TestHistory:
    def test_unknown_city_404(self, client):
        assert client.get("/history/atlantis").status_code == 404

    def test_returns_list(self, client):
        r = client.get("/history/tokyo?hours=24")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_result_schema(self, client):
        results = client.get("/history/tokyo?hours=24").json()
        if results:
            r = results[0]
            assert "timestamp" in r and "predicted" in r and "actual" in r


# ── /metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_unknown_city_404(self, client):
        assert client.get("/metrics/atlantis").status_code == 404

    def test_valid_city_200(self, client):
        assert client.get("/metrics/tokyo").status_code == 200

    def test_weighted_schema(self, client):
        d = client.get("/metrics/tokyo").json()
        assert "weighted" in d
        w = d["weighted"]
        assert "f1" in w and "precision" in w and "recall" in w

    def test_per_class_schema(self, client):
        d = client.get("/metrics/tokyo").json()
        for cls in ["1", "2", "3", "4", "5"]:
            assert cls in d["per_class"]
            c = d["per_class"][cls]
            assert "f1" in c and "precision" in c and "recall" in c and "support" in c

    def test_metrics_in_valid_range(self, client):
        d = client.get("/metrics/tokyo").json()
        w = d["weighted"]
        for v in (w["f1"], w["precision"], w["recall"]):
            assert 0.0 <= v <= 1.0, f"Metric out of range: {v}"

    def test_n_predictions_positive(self, client):
        d = client.get("/metrics/tokyo").json()
        assert d["n_predictions"] >= 10

    def test_computed_at_present(self, client):
        d = client.get("/metrics/tokyo").json()
        assert "computed_at" in d and "UTC" in d["computed_at"]
