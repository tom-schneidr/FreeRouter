"""Typed Sentinel evidence records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ProbeStatus = Literal["pass", "warn", "fail", "error"]
ReadinessStatus = Literal["ready", "limited", "blocked"]


@dataclass(frozen=True)
class SentinelProbeResult:
    check_id: str
    label: str
    status: ProbeStatus
    score: int
    latency_ms: int
    evidence: str
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SentinelEvaluation:
    run_id: str
    route_id: str
    provider_name: str
    model_id: str
    score: int
    readiness: ReadinessStatus
    started_at: int
    completed_at: int
    duration_ms: int
    summary: str
    probes: list[SentinelProbeResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "probes": [probe.to_dict() for probe in self.probes],
        }
