"""In-process metrics registry.

A lightweight, dependency-free registry that accumulates request counts, success
/ error counts, and per-operation latency summaries. Exposed as JSON at
``GET /v1/metrics``. For a multi-process or scrape-based setup, swap this for a
Prometheus client behind the same record calls.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class _LatencyStat:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


@dataclass
class MetricsRegistry:
    """Thread-safe accumulator for request and latency metrics."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _counters: dict[str, int] = field(default_factory=dict)
    _latencies: dict[str, _LatencyStat] = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe_latency(self, operation: str, ms: float) -> None:
        """Record a latency sample for ``operation``."""
        with self._lock:
            self._latencies.setdefault(operation, _LatencyStat()).record(ms)

    def record_request(self, success: bool) -> None:
        """Record a request outcome (success / error)."""
        self.increment("requests_total")
        self.increment("requests_success" if success else "requests_error")

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable view of all metrics."""
        with self._lock:
            total = self._counters.get("requests_total", 0)
            success = self._counters.get("requests_success", 0)
            errors = self._counters.get("requests_error", 0)
            return {
                "counters": dict(self._counters),
                "success_rate": success / total if total else 0.0,
                "error_rate": errors / total if total else 0.0,
                "latency_ms": {
                    op: {
                        "count": stat.count,
                        "avg_ms": round(stat.avg_ms, 2),
                        "max_ms": round(stat.max_ms, 2),
                    }
                    for op, stat in self._latencies.items()
                },
            }
