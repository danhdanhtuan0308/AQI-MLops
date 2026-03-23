# tests/ — Test Suite

Automated tests covering the full ML pipeline: feature engineering, training logic, and all API endpoints. 52 tests total. No real AWS calls are made anywhere in the suite. Athena is fully mocked.

---

## Directory Structure

```
tests/
  __init__.py               Package marker
  conftest.py               Shared fixtures: auto-creates minimal model artifacts in CI if absent
  test_inference.py         16 tests for feature engineering and prediction functions
  test_training_logic.py    14 tests for data preparation and array building (no AWS)
  test_api.py               22 tests for all FastAPI endpoints (Athena mocked)
```

---

## Running Tests

```bash
uv sync --group dev

uv run pytest tests/ -v --tb=short

uv run pytest tests/test_api.py -v

uv run ruff check app/ ml/ tests/
```

---

## conftest.py

Contains a session-scoped autouse fixture called ensure_test_artifacts. It checks whether all three model artifact files exist at ml/model-registry/model.ubj, median.json, and features.json. If any are missing (as in CI where the model-registry/ directory is gitignored), it creates a minimal XGBoost model:

- 100 synthetic rows, 5 AQI classes, 14 features
- 10 estimators with max_depth 2 for fast construction
- Writes all three artifact files before any test runs

This makes the full test suite hermetic. No production model or AWS connection is required.

---

## test_inference.py — 16 tests

Tests for functions in app/inference.py.

| Test class | What it covers |
|------------|----------------|
| TestBuildFeatureVector | Output shape is (14,), dtype is float32, no NaN in output, hour_sin and hour_cos are correct at midnight and noon, month_sin and month_cos are correct, pm25_ratio is non-negative, None and NaN inputs fall back to the median value |
| TestPredictSingle | Predicted AQI is in range 1 through 5, probabilities sum to 1.0, five probability keys are present, response contains label and color fields |
| TestBatchPredict | Returns empty list on fewer than 3 rows, output length equals number of rows minus 2, schema is correct, actual field matches rows at position i+2 |

---

## test_training_logic.py — 14 tests

Tests for feature engineering and array building in ml/train.py. Runs entirely in memory with no network calls.

| Test class | What it covers |
|------------|----------------|
| TestEngineerFeatures | All 14 features are present in the output, no NaN in features or target, target values are in range 1 through 5, hour_sin and hour_cos are bounded between -1 and 1, month_sin and month_cos are bounded between -1 and 1, pm25_ratio is non-negative, no cross-city data leakage in lag features, row count equals number of cities times hours minus 2 |
| TestBuildArrays | X shape is (n_samples, 14), y length matches X, dtype is float32, y is zero-indexed from 0 to 4, no NaN in X, median keys match the feature list |

---

## test_api.py — 22 tests

Integration tests for all FastAPI routes. The client fixture patches app.main._athena to return synthetic DataFrames so no AWS calls are made.

| Test class | What it covers |
|------------|----------------|
| TestHealth | Returns HTTP 200, all expected schema keys are present, model_loaded is True, n_features is 14 |
| TestCities | Returns a list, each item has slug, name, lat, and lon fields, tokyo is in the list |
| TestPredict | Returns 404 for an unknown city, returns 200 for a valid city, predicted AQI is in range 1 through 5, probabilities sum to 1, five probability keys are present |
| TestHistory | Returns 404 for an unknown city, returns a list of records, each record has timestamp, predicted, and actual fields |
| TestMetrics | Returns 404 for an unknown city, returns 200 for a valid city, weighted schema has f1 and precision and recall, per-class schema has f1 and precision and recall and support, all metric values are between 0 and 1, n_predictions is at least 10, computed_at string ends with UTC |

---

## CI Integration

Tests run automatically on every push and pull request in ci.yml. The weekly retrain workflow also runs test_inference.py and test_training_logic.py as a gate before any retraining begins. If tests fail, no model update is deployed.

Ruff lint runs before tests so the pipeline fails fast on style errors before spending time on the test suite.

---

## Dependencies

Installed via uv sync --group dev:

| Package | Version | Purpose |
|---------|---------|--------|
| pytest | 9.0 or higher | Test runner |
| httpx | 0.27 or higher | Async HTTP client required by FastAPI TestClient |
| ruff | 0.4 or higher | Linter and style checker |

---

## Directory Structure

```
tests/
├── __init__.py               # Package marker
├── conftest.py               # Shared fixtures — auto-creates minimal model in CI if absent
├── test_inference.py         # 16 tests — feature engineering + prediction functions
├── test_training_logic.py    # 14 tests — engineer_features + build_arrays (no AWS)
└── test_api.py               # 22 tests — all FastAPI endpoints (Athena mocked)
```

---

## Running tests

```bash
# Install dev dependencies
uv sync --group dev

# Run all 52 tests
uv run pytest tests/ -v --tb=short

# Run a single file
uv run pytest tests/test_api.py -v

# Lint first (ruff)
uv run ruff check app/ ml/ tests/
```

---

## File-by-file Overview

### conftest.py

**Session-scoped `ensure_test_artifacts` fixture (autouse)**

Checks whether all three model artifacts exist: `ml/model-registry/model.ubj`, `median.json`, and `features.json`. If any are missing (as in CI where `*.json` is gitignored), it creates a minimal XGBoost model:
- 100 synthetic rows, 5 AQI classes, 12 features
- 10 estimators, max_depth=2 — fast to build
- Writes all three artifact files before any test runs

This makes all tests fully hermetic — no production model or AWS needed.

---

### test_inference.py — 16 tests

Tests for `app/inference.py` functions.

| Class | Tests |
|---|---|
| `TestBuildFeatureVector` | Output shape (12,), dtype float32, no NaN, `hour_sin`/`hour_cos` correct at midnight and noon, `pm25_ratio` bounded [0, ∞), None/NaN inputs fall back to median |
| `TestPredictSingle` | Predicted AQI in [1..5], probabilities sum to 1.0, 5 probability keys, response contains `label` and `color` |
| `TestBatchPredict` | Empty return on < 3 rows, output length = n−2, schema correct, `actual` matches `rows[i+2]` |

---

### test_training_logic.py — 14 tests

Tests for `ml/train.py` feature engineering and array building. Runs entirely in-memory — no Athena, no AWS.

| Class | Tests |
|---|---|
| `TestEngineerFeatures` | All 12 features present, no NaN in features or target, target values in [1..5], `hour_sin`/`hour_cos` bounded [−1, 1], `pm25_ratio` ≥ 0, no cross-city data leakage, row count = n_cities × (n_hours − 2) |
| `TestBuildArrays` | X shape = (n_samples, 12), y length matches X, dtype float32, y is zero-indexed (0–4), no NaN in X, median keys match the feature list |

---

### test_api.py — 22 tests

Integration tests for all FastAPI routes. The `client` fixture patches `app.main._athena` to return synthetic DataFrames — no AWS calls at all.

| Class | Tests |
|---|---|
| `TestHealth` | HTTP 200, all expected schema keys present, `model_loaded=True`, `n_features=12` |
| `TestCities` | Returns a list, each item has `slug`/`name`/`lat`/`lon`, `tokyo` is present |
| `TestPredict` | 404 for unknown city, 200 for valid city, AQI in [1..5], probabilities sum to 1, 5 probability keys |
| `TestHistory` | 404 for unknown city, returns a list, schema has `timestamp`/`predicted`/`actual` |
| `TestMetrics` | 404 for unknown city, 200 for valid city, weighted schema (f1/precision/recall), per-class schema (f1/precision/recall/support), all metric values in [0,1], `n_predictions ≥ 10`, `computed_at` ends with UTC |

---

## CI Integration

Tests run automatically on every push and pull request via `.github/workflows/ci.yml`. The weekly retrain workflow also runs `test_inference.py` and `test_training_logic.py` as a CI gate before any retraining begins.

Lint (ruff) runs before tests — the pipeline fails fast on style errors.

---

## Dependencies

Installed via `uv sync --group dev`:

| Package | Version | Purpose |
|---|---|---|
| pytest | ≥ 9.0 | Test runner |
| httpx | ≥ 0.27 | Async HTTP client (required by FastAPI TestClient) |
| ruff | ≥ 0.4 | Linter / style checker |
