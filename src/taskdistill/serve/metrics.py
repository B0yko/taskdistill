"""Prometheus metrics of one cascade server.

Every app gets its own :class:`~prometheus_client.CollectorRegistry`, so two servers in one process (as in the
tests) never share counters, and ``/metrics`` shows only this server's series.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

ROUTES = ("student", "teacher", "student-fallback", "error")
REASONS = ("low_confidence", "unsupported", "input_unparsed")
TEACHER_ERROR_KINDS = ("timeout", "budget", "http", "transport", "replay_miss", "error")
#: Seconds; a small student answers in tens of milliseconds, a remote teacher in about a second.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


class ServeMetrics:
    """The server's counters and latency histograms, in a registry of their own."""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests: Any = Counter(
            "taskdistill_requests_total",
            "Chat completion requests by the route that answered them.",
            ["route"],
            registry=self.registry,
        )
        self.escalations: Any = Counter(
            "taskdistill_escalations_total",
            "Requests escalated to the teacher, by reason.",
            ["reason"],
            registry=self.registry,
        )
        self.request_latency: Any = Histogram(
            "taskdistill_request_latency_seconds",
            "End-to-end latency of chat completion requests, by route.",
            ["route"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.student_latency: Any = Histogram(
            "taskdistill_student_latency_seconds",
            "Student generation time (without the wait for the model worker).",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.student_queue: Any = Histogram(
            "taskdistill_student_queue_seconds",
            "Time requests waited for the model worker before their generation started.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.teacher_latency: Any = Histogram(
            "taskdistill_teacher_latency_seconds",
            "Latency of successful teacher calls (a stream until its last byte).",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.unnormalised: Any = Counter(
            "taskdistill_escalation_unnormalised_total",
            "Canonical low-confidence escalations whose teacher output could not be normalised (returned raw).",
            registry=self.registry,
        )
        self.teacher_errors: Any = Counter(
            "taskdistill_teacher_errors_total",
            "Failed teacher calls, by kind.",
            ["kind"],
            registry=self.registry,
        )
        # Pre-create the known label values so every series is exported from the start, at zero.
        for route in ROUTES:
            self.requests.labels(route=route)
            self.request_latency.labels(route=route)
        for reason in REASONS:
            self.escalations.labels(reason=reason)
        for kind in TEACHER_ERROR_KINDS:
            self.teacher_errors.labels(kind=kind)

    def render(self) -> bytes:
        """The registry in the Prometheus text exposition format."""
        data: bytes = generate_latest(self.registry)
        return data
