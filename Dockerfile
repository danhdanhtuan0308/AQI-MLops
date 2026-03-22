# =============================================================================
# Dockerfile — AQI MLOps FastAPI service
# Target: python:3.12-slim  (multi-arch: amd64 + arm64 / Graviton)
# =============================================================================

# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM python:3.12-slim AS deps

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install only production deps first (layer is cached unless pyproject.toml changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: final runtime image ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the pre-built virtualenv from deps stage
COPY --from=deps /app/.venv /app/.venv

# Copy application source
COPY app/ ./app/
COPY data-pipeline/config.yaml ./data-pipeline/config.yaml
COPY ml/__init__.py ./ml/__init__.py

# model-registry is bind-mounted at runtime — just create the directory
RUN mkdir -p ml/model-registry

# Activate venv on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
