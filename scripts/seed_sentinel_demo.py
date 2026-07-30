"""Seed deterministic Sentinel evidence into an explicit demo database."""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from pathlib import Path

from app.model_catalog import ModelCatalog
from app.sentinel_store import SentinelStore
from app.sentinel_types import SentinelEvaluation, SentinelProbeResult

CHECKS = (
    ("tool_call", "Tool-call conformance"),
    ("structured_json", "Structured JSON"),
    ("streaming", "Streaming readiness"),
    ("canary", "Canary privacy"),
)


def _evaluation(route, index: int) -> SentinelEvaluation:
    now = int(time.time()) - index * 3800
    if index == 0:
        statuses = ("pass", "pass", "pass", "pass")
        scores = (25, 25, 25, 25)
        readiness = "ready"
        summary = "Ready for agent workloads with current evidence."
    elif index == 1:
        statuses = ("pass", "fail", "warn", "pass")
        scores = (25, 0, 15, 25)
        readiness = "limited"
        summary = "Usable with constraints; review warning checks before agent use."
    else:
        statuses = ("fail", "pass", "pass", "pass")
        scores = (0, 25, 25, 25)
        readiness = "blocked"
        summary = "Not trusted for autonomous agent use yet."
    probes = [
        SentinelProbeResult(
            check_id=check_id,
            label=label,
            status=status,
            score=score,
            latency_ms=140 + index * 35,
            evidence=(
                "Deterministic demo evidence passed."
                if status == "pass"
                else "Demo response did not meet the required contract."
            ),
            remediation=(
                ""
                if status == "pass"
                else "Choose a route with native support, then run Sentinel again."
            ),
        )
        for (check_id, label), status, score in zip(CHECKS, statuses, scores, strict=True)
    ]
    return SentinelEvaluation(
        run_id=f"demo-{uuid.uuid4().hex}",
        route_id=route.route_id,
        provider_name=route.provider_name,
        model_id=route.model_id,
        score=sum(scores),
        readiness=readiness,
        started_at=now - 4,
        completed_at=now,
        duration_ms=2150 + index * 300,
        summary=summary,
        probes=probes,
    )


async def seed(database_path: Path, catalog_path: Path) -> None:
    catalog = ModelCatalog(str(catalog_path))
    catalog.initialize()
    store = SentinelStore(str(database_path))
    await store.initialize()
    routes = catalog.enabled_routes()[:3]
    for index, route in enumerate(routes):
        await store.save(_evaluation(route, index))
    print(f"Seeded {len(routes)} Sentinel demo evaluations in {database_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(seed(args.database.resolve(), args.catalog.resolve()))


if __name__ == "__main__":
    main()
