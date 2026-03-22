# .github/ — CI/CD Workflows

Three GitHub Actions workflows that automate linting, testing, deployment, and weekly model retraining.

---

## Directory Structure

```
.github/
└── workflows/
    ├── ci.yml                # Lint + unit tests on every push / PR
    ├── cd.yml                # Auto-deploy to EC2 on push to main
    └── weekly_retrain.yml    # Scheduled weekly retraining with CI gate
```

---

## ci.yml — Continuous Integration

**Triggers:** push to `main`, `dev`, `feature/**`; any PR targeting `main`; also callable as a reusable workflow (`workflow_call`) by `cd.yml` and `weekly_retrain.yml`.

**Runner:** `ubuntu-latest`

**Steps:**
1. Checkout code
2. Install `uv` + Python 3.12 via `astral-sh/setup-uv@v4`
3. `uv sync --group dev` — installs all dependencies including pytest, httpx, ruff
4. `uv run ruff check app/ ml/ tests/` — lint; fails on any E/F/W violation
5. `uv run pytest tests/ -v --tb=short -q` — runs all 52 tests

The `conftest.py` fixture auto-builds a minimal model in CI (because `*.json` artifacts are gitignored), so all 52 tests pass without any real AWS credentials or production model.

---

## cd.yml — Continuous Deployment

**Triggers:** push to `main` when files in `app/**`, `ml/model-registry/**`, `pyproject.toml`, or `uv.lock` change. Also `workflow_dispatch` for manual deploys.

**Concurrency:** `group: deploy-production`, `cancel-in-progress: false` — only one deploy runs at a time; queued deploys are never cancelled.

**Jobs:**

### 1. `ci` (gate)
Calls `ci.yml` as a reusable workflow. The `deploy` job is blocked until this passes.

### 2. `deploy`
Runs on `ubuntu-latest`, requires the `production` GitHub environment.

Steps:
1. Writes the SSH private key from `EC2_SSH_KEY` secret to `~/.ssh/ec2.pem`
2. `ssh-keyscan` adds the host to `known_hosts` (no interactive prompts)
3. SSH into EC2 and runs:
   - `git fetch origin main && git reset --hard origin/main && git clean -fd` — hard reset (never blocked by local changes)
   - `docker compose build` — rebuilds the image using the updated code + `uv.lock`
   - `docker compose up -d` — replaces the running container with the new image
   - `sleep 8` then `curl http://localhost:8000/health` — workflow fails if health check doesn't return `"status": "ok"`
   - On failure: `docker compose logs --tail=50` is printed before exiting

**Required Secrets:**

| Secret | Value |
|---|---|
| `EC2_SSH_KEY` | PEM private key (contents of `~/.ssh/aqi-mlops-key.pem`) |
| `EC2_HOST` | `ec2-3-94-115-44.compute-1.amazonaws.com` |
| `EC2_USER` | `ubuntu` |
| `AWS_ACCESS_KEY_ID` | AWS credential (for Athena queries at runtime) |
| `AWS_SECRET_ACCESS_KEY` | AWS credential |
| `AWS_REGION` | `us-east-1` |

**Required Variable:**

| Variable | Value |
|---|---|
| `API_URL` | `http://ec2-3-94-115-44.compute-1.amazonaws.com` |

---

## weekly_retrain.yml — Scheduled Retraining

**Triggers:** cron `0 2 * * 1` (every Monday at 02:00 UTC); also `workflow_dispatch`.

**Jobs:**

### 1. `ci-gate`
Runs a focused subset of tests before any retraining:
- `ruff check app/ ml/ tests/` — lint
- `pytest tests/test_inference.py tests/test_training_logic.py -v --tb=short` — inference + training logic tests only (fast, no AWS)

If this fails, the `retrain` job never starts — protecting production from a broken codebase triggering a model replacement.

### 2. `retrain` (needs `ci-gate`)

Steps:
1. Configure AWS credentials from secrets
2. `uv run python ml/train.py --lookback-weeks 8` — fetches 8 weeks of data from Athena, trains XGBoost, writes artifacts to `ml/model-registry/`
3. Python smoke test: loads the new model, checks proba shape (5 classes), proba sum ≈ 1.0, median dict has correct number of keys
4. Upload `ml/model-registry/` as a workflow artifact (retained 90 days)
5. `curl -X POST ${{ vars.API_URL }}/reload-model` — hot-reloads the new model on the running EC2 instance without a restart

---

## Secrets & Variables Summary

All configured in the GitHub repo → Settings → Secrets and variables → Actions.

| Name | Type | Purpose |
|---|---|---|
| `EC2_SSH_KEY` | Secret | SSH private key for EC2 access |
| `EC2_HOST` | Secret | EC2 public DNS |
| `EC2_USER` | Secret | SSH username (`ubuntu`) |
| `AWS_ACCESS_KEY_ID` | Secret | AWS credential for Athena |
| `AWS_SECRET_ACCESS_KEY` | Secret | AWS credential for Athena |
| `AWS_REGION` | Secret | `us-east-1` |
| `API_URL` | Variable | Public EC2 URL for hot-reload |
