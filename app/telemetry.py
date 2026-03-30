"""
Observability — push traces, metrics, and logs directly to Grafana Cloud APIs.

All 3 signals use the same OTLP gateway endpoint — no separate agents on EC2.

Metrics → Grafana Cloud Mimir  (OTLP HTTP, pushed every 30s)
Traces  → Grafana Cloud Tempo  (OTLP HTTP, per request)
Logs    → Grafana Cloud Loki   (HTTP POST, async queue)

Required env vars (from Grafana Cloud → Connections → OpenTelemetry → Direct):
    GRAFANA_API_KEY           access-policy token
    GRAFANA_OTLP_ENDPOINT     e.g. https://otlp-gateway-prod-us-east-2.grafana.net/otlp
    GRAFANA_OTLP_INSTANCE_ID  numeric instance ID shown on the OTel setup page

Optional (for logs):
    GRAFANA_LOKI_URL          e.g. https://logs-prod-XXX.grafana.net/loki/api/v1/push
    GRAFANA_LOKI_USER         numeric Loki user ID

No-op (safe) when env vars are absent or packages are not installed.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from base64 import b64encode
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

log = logging.getLogger(__name__)


def _otlp_auth_headers() -> dict[str, str] | None:
    """Return Basic auth headers for the OTLP gateway, or None if not configured."""
    instance_id = os.environ.get("GRAFANA_OTLP_INSTANCE_ID", "").strip()
    api_key     = os.environ.get("GRAFANA_API_KEY", "").strip()
    if not all([instance_id, api_key]):
        return None
    token = b64encode(f"{instance_id}:{api_key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _otlp_endpoint() -> str:
    return os.environ.get("GRAFANA_OTLP_ENDPOINT", "").strip().rstrip("/")


# ── Traces → Grafana Cloud Tempo (via OTLP gateway) ───────────────────────────

def setup_tracing(app) -> None:
    """Instrument FastAPI with OTel and push spans to Grafana Cloud via OTLP."""
    endpoint = _otlp_endpoint()
    headers  = _otlp_auth_headers()

    if not endpoint or headers is None:
        log.info("OTel tracing disabled (GRAFANA_OTLP_ENDPOINT/INSTANCE_ID/API_KEY not set)")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name":           os.environ.get("OTEL_SERVICE_NAME",    "aqi-api"),
            "service.version":        os.environ.get("OTEL_SERVICE_VERSION", "1.0"),
            "deployment.environment": "production",
        })

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", headers=headers)
            )
        )
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=provider,
            excluded_urls="health,metrics,cache/status,favicon.ico",
        )
        log.info("OTel tracing enabled → %s/v1/traces", endpoint)

    except ImportError:
        log.warning("opentelemetry packages not installed — tracing disabled")


# ── Metrics → Grafana Cloud Mimir (via OTLP gateway) ──────────────────────────

class _Noop:
    """No-op stub so metric call-sites never need an if-guard."""
    def add(self, *_, **__): pass
    def record(self, *_, **__): pass

METRIC_PREDICTIONS:    "_Noop" = _Noop()
METRIC_CACHE_OPS:      "_Noop" = _Noop()
METRIC_WARM_CACHE:     "_Noop" = _Noop()
METRIC_ATHENA_QUERIES: "_Noop" = _Noop()


def setup_metrics_push() -> None:
    """Create OTel metrics and push them to Grafana Cloud every 30 s via OTLP."""
    global METRIC_PREDICTIONS, METRIC_CACHE_OPS, METRIC_WARM_CACHE, METRIC_ATHENA_QUERIES

    endpoint = _otlp_endpoint()
    headers  = _otlp_auth_headers()

    if not endpoint or headers is None:
        log.info("Metrics push disabled (GRAFANA_OTLP_ENDPOINT/INSTANCE_ID/API_KEY not set)")
        return

    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({
            "service.name":           os.environ.get("OTEL_SERVICE_NAME",    "aqi-api"),
            "service.version":        os.environ.get("OTEL_SERVICE_VERSION", "1.0"),
            "deployment.environment": "production",
        })

        exporter = OTLPMetricExporter(
            endpoint=f"{endpoint}/v1/metrics",
            headers=headers,
        )
        reader   = PeriodicExportingMetricReader(exporter, export_interval_millis=30_000)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        otel_metrics.set_meter_provider(provider)

        meter = provider.get_meter("aqi-api", "1.0")
        METRIC_PREDICTIONS    = meter.create_counter("aqi.predictions.total",    description="Total AQI predictions served")
        METRIC_CACHE_OPS      = meter.create_counter("aqi.cache.ops.total",      description="Redis cache hits and misses")
        METRIC_WARM_CACHE     = meter.create_histogram("aqi.warm_cache.seconds", description="Time to run /warm-cache", unit="s")
        METRIC_ATHENA_QUERIES = meter.create_counter("aqi.athena.queries.total", description="Total Athena queries executed")

        log.info("Metrics push enabled → %s/v1/metrics", endpoint)

    except ImportError:
        log.warning("opentelemetry packages not installed — metrics push disabled")


# ── Logs → Grafana Cloud Loki ─────────────────────────────────────────────────

class _LokiHandler(logging.Handler):
    """Async logging handler that pushes log entries directly to Grafana Cloud Loki."""

    def __init__(self, url: str, user: str, api_key: str):
        super().__init__()
        self._url = url
        token = b64encode(f"{user}:{api_key}".encode()).decode()
        self._headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Basic {token}",
        }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = str(int(record.created * 1e9))
            payload = {
                "streams": [{
                    "stream": {
                        "job":    "aqi-api",
                        "env":    "production",
                        "level":  record.levelname.lower(),
                        "logger": record.name,
                    },
                    "values": [[ts, self.format(record)]],
                }]
            }
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(self._url, data=data, headers=self._headers, method="POST")
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # Never let Loki failures crash the app


def setup_logging() -> None:
    """Attach a non-blocking Loki handler to the root logger."""
    loki_url  = os.environ.get("GRAFANA_LOKI_URL",  "").strip()
    loki_user = os.environ.get("GRAFANA_LOKI_USER", "").strip()
    api_key   = os.environ.get("GRAFANA_API_KEY",   "").strip()

    if not all([loki_url, loki_user, api_key]):
        log.info("Loki logging disabled (GRAFANA_LOKI_URL/USER/API_KEY not set)")
        return

    loki_handler = _LokiHandler(loki_url, loki_user, api_key)
    loki_handler.setLevel(logging.INFO)
    loki_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))

    log_queue = Queue(-1)
    listener  = QueueListener(log_queue, loki_handler, respect_handler_level=True)
    listener.start()

    logging.getLogger().addHandler(QueueHandler(log_queue))
    log.info("Loki logging enabled → %s", loki_url)
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from base64 import b64encode
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

log = logging.getLogger(__name__)


# ── Traces → Grafana Cloud Tempo ──────────────────────────────────────────────

def setup_tracing(app) -> None:
    """Instrument FastAPI with OTel and push spans to Grafana Cloud Tempo."""
    tempo_url = os.environ.get("GRAFANA_TEMPO_URL", "").strip()
    tempo_user = os.environ.get("GRAFANA_TEMPO_USER", "").strip()
    api_key = os.environ.get("GRAFANA_API_KEY", "").strip()

    if not all([tempo_url, tempo_user, api_key]):
        log.info("OTel tracing disabled (GRAFANA_TEMPO_URL/USER/API_KEY not set)")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name":           os.environ.get("OTEL_SERVICE_NAME",    "aqi-api"),
            "service.version":        os.environ.get("OTEL_SERVICE_VERSION", "1.0"),
            "deployment.environment": "production",
        })

        # Basic auth header: username = GRAFANA_TEMPO_USER, password = GRAFANA_API_KEY
        auth_bytes = b64encode(f"{tempo_user}:{api_key}".encode()).decode()
        headers = {"Authorization": f"Basic {auth_bytes}"}

        otlp_endpoint = f"{tempo_url.rstrip('/')}/v1/traces"

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=otlp_endpoint, headers=headers)
            )
        )
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=provider,
            excluded_urls="health,metrics,cache/status,favicon.ico",
        )
        log.info("OTel tracing enabled → %s", tempo_url)

    except ImportError:
        log.warning("opentelemetry packages not installed — tracing disabled")


# ── Metrics → Grafana Cloud Mimir ─────────────────────────────────────────────

# Module-level metric instruments — initialised by setup_metrics_push().
# Before that, they are _Noop stubs (safe to call .add / .record).

class _Noop:
    """No-op stub so metric call-sites never need an if-guard."""
    def add(self, *_, **__): pass
    def record(self, *_, **__): pass

METRIC_PREDICTIONS:  "Counter | _Noop"    = _Noop()
METRIC_CACHE_OPS:    "Counter | _Noop"    = _Noop()
METRIC_WARM_CACHE:   "Histogram | _Noop"  = _Noop()
METRIC_ATHENA_QUERIES: "Counter | _Noop"  = _Noop()


def setup_metrics_push() -> None:
    """Create OTel metrics and push them to Grafana Cloud every 30 s via OTLP."""
    global METRIC_PREDICTIONS, METRIC_CACHE_OPS, METRIC_WARM_CACHE, METRIC_ATHENA_QUERIES

    otlp_endpoint = os.environ.get("GRAFANA_OTLP_ENDPOINT", "").strip()
    instance_id   = os.environ.get("GRAFANA_OTLP_INSTANCE_ID", "").strip()
    api_key       = os.environ.get("GRAFANA_API_KEY", "").strip()

    if not all([otlp_endpoint, instance_id, api_key]):
        log.info("Metrics push disabled (GRAFANA_OTLP_ENDPOINT/INSTANCE_ID/API_KEY not set)")
        return

    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({
            "service.name":           os.environ.get("OTEL_SERVICE_NAME",    "aqi-api"),
            "service.version":        os.environ.get("OTEL_SERVICE_VERSION", "1.0"),
            "deployment.environment": "production",
        })

        auth_bytes = b64encode(f"{instance_id}:{api_key}".encode()).decode()
        headers = {"Authorization": f"Basic {auth_bytes}"}

        exporter = OTLPMetricExporter(
            endpoint=f"{otlp_endpoint.rstrip('/')}/v1/metrics",
            headers=headers,
        )
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=30_000)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        otel_metrics.set_meter_provider(provider)

        meter = provider.get_meter("aqi-api", "1.0")

        METRIC_PREDICTIONS = meter.create_counter(
            "aqi.predictions.total",
            description="Total AQI predictions served",
        )
        METRIC_CACHE_OPS = meter.create_counter(
            "aqi.cache.ops.total",
            description="Redis cache hits and misses per endpoint",
        )
        METRIC_WARM_CACHE = meter.create_histogram(
            "aqi.warm_cache.seconds",
            description="Time taken to execute /warm-cache",
            unit="s",
        )
        METRIC_ATHENA_QUERIES = meter.create_counter(
            "aqi.athena.queries.total",
            description="Total Athena queries executed",
        )

        log.info("Metrics push enabled → %s", otlp_endpoint)

    except ImportError:
        log.warning("opentelemetry packages not installed — metrics push disabled")


# ── Logs → Grafana Cloud Loki ─────────────────────────────────────────────────

class _LokiHandler(logging.Handler):
    """Logging handler that pushes log entries to Grafana Cloud Loki via HTTP."""

    def __init__(self, url: str, user: str, api_key: str):
        super().__init__()
        self._url = url
        auth_bytes = b64encode(f"{user}:{api_key}".encode()).decode()
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth_bytes}",
        }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = str(int(record.created * 1e9))  # nanosecond epoch
            payload = {
                "streams": [{
                    "stream": {
                        "job": "aqi-api",
                        "env": "production",
                        "level": record.levelname.lower(),
                        "logger": record.name,
                    },
                    "values": [[ts, self.format(record)]],
                }]
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                self._url, data=data, headers=self._headers, method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            # Never let Loki failures crash the app
            pass


def setup_logging() -> None:
    """Attach a Loki log handler to the root logger (async via queue)."""
    loki_url = os.environ.get("GRAFANA_LOKI_URL", "").strip()
    loki_user = os.environ.get("GRAFANA_LOKI_USER", "").strip()
    api_key = os.environ.get("GRAFANA_API_KEY", "").strip()

    if not all([loki_url, loki_user, api_key]):
        log.info("Loki logging disabled (GRAFANA_LOKI_URL/USER/API_KEY not set)")
        return

    loki_handler = _LokiHandler(loki_url, loki_user, api_key)
    loki_handler.setLevel(logging.INFO)
    loki_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))

    # Use a queue so HTTP posts don't block the request thread
    log_queue: Queue = Queue(-1)
    queue_handler = QueueHandler(log_queue)
    listener = QueueListener(log_queue, loki_handler, respect_handler_level=True)
    listener.start()

    root = logging.getLogger()
    root.addHandler(queue_handler)
    log.info("Loki logging enabled → %s", loki_url)
