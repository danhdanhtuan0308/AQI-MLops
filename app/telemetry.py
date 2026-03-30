"""
OpenTelemetry tracing — sends spans to Grafana Alloy (OTLP HTTP) → Grafana Cloud Tempo.

Enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set in .env, e.g.:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy:4318

No-op (safe) when the env var is absent — the app starts normally even if the
opentelemetry packages are not installed.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def setup_tracing(app) -> None:
    """Instrument the FastAPI app with OTel traces.

    Skips silently when OTEL_EXPORTER_OTLP_ENDPOINT is not set or the
    opentelemetry packages are missing.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        log.info("OTel tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
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
                OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
            )
        )
        trace.set_tracer_provider(provider)

        # Exclude internal / operational endpoints from trace noise
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=provider,
            excluded_urls="health,metrics,cache/status,favicon.ico",
        )
        log.info("OTel tracing enabled → %s", endpoint)

    except ImportError:
        log.warning("opentelemetry packages not installed — tracing disabled")
