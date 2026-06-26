"""Tracing abstraction.

A minimal span API (`start_span` as a context manager) decoupled from any
backend. The default :class:`StructlogTracer` emits structured span events
through the existing logging pipeline — zero extra dependencies. Because call
sites depend only on the :class:`Tracer` interface, a Langfuse or OpenTelemetry
exporter can be added later by implementing one class, without touching the API
or services.

The Langfuse / OpenTelemetry adapters are intentionally left as documented
extension points (see ``docs/deployment.md``) rather than half-wired
dependencies.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from utils.logging import get_logger

logger = get_logger("tracing")


class Span(ABC):
    """A single unit of traced work."""

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a key/value attribute to the span."""
        raise NotImplementedError


class Tracer(ABC):
    """Creates spans. Implementations decide where spans are recorded."""

    @abstractmethod
    @contextmanager
    def start_span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """Start a span as a context manager, yielding the :class:`Span`."""
        raise NotImplementedError


class _StructlogSpan(Span):
    def __init__(self, name: str, attributes: dict[str, Any]) -> None:
        self.name = name
        self.attributes = attributes

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class StructlogTracer(Tracer):
    """Records spans as structured log events, including duration."""

    @contextmanager
    def start_span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = _StructlogSpan(name, dict(attributes))
        started = time.perf_counter()
        try:
            yield span
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info("span", name=name, duration_ms=duration_ms, **span.attributes)


class NoOpTracer(Tracer):
    """Tracer that records nothing (useful in tests)."""

    @contextmanager
    def start_span(self, name: str, **attributes: Any) -> Iterator[Span]:
        yield _StructlogSpan(name, dict(attributes))


def build_tracer(backend: str = "structlog") -> Tracer:
    """Return a tracer for ``backend`` (``structlog`` default, ``none`` for no-op).

    Future backends (``langfuse``, ``opentelemetry``) implement :class:`Tracer`
    and are selected here.
    """
    mapping: Mapping[str, type[Tracer]] = {
        "structlog": StructlogTracer,
        "none": NoOpTracer,
    }
    return mapping.get(backend, StructlogTracer)()
