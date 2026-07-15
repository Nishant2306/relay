"""OpenTelemetry setup (SPEC C9).

Spans are created unconditionally via the OTel API (near-zero cost no-ops
until a provider is installed); `setup_tracing()` installs a real
TracerProvider when OTEL_EXPORTER_OTLP_ENDPOINT is configured, else a
console exporter when RELAY_TRACE_CONSOLE=1, else leaves the no-op API.
"""

from __future__ import annotations

import os


def setup_tracing(service_name: str = "relay") -> None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    console = os.environ.get("RELAY_TRACE_CONSOLE", "") == "1"
    if not endpoint and not console:
        return

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except ImportError:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
