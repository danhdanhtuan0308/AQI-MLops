
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


# ── System Metrics (CPU, memory, network, disk) ────────────────────────────────

def setup_system_metrics() -> None:
    """Push host CPU/memory metrics via OTel (uses global meter provider).

    Intentionally limited to CPU utilization + memory usage only to avoid
    per-core/per-disk/per-NIC cardinality explosion on Grafana Cloud Free tier.
    Must be called AFTER setup_metrics_push() so the global MeterProvider is set.
    """
    try:
        from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor

        # Only collect the two lowest-cardinality metrics to stay within Free tier limits.
        # Omitting: system.cpu.time, disk.*, network.* (high series count per device).
        minimal_config = {
            "system.cpu.utilization": ["user", "system", "idle"],
            "system.memory.usage":    ["used", "free", "available"],
        }
        SystemMetricsInstrumentor(config=minimal_config).instrument()
        log.info("System metrics instrumentation enabled (CPU utilization, memory usage)")
    except ImportError:
        log.warning("opentelemetry-instrumentation-system-metrics not installed — system metrics disabled")


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
