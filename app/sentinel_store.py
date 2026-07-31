"""SQLite persistence for route-level Sentinel evidence."""

from __future__ import annotations

import json

import aiosqlite

from app.sentinel_types import SentinelEvaluation, SentinelProbeResult


class SentinelStore:
    def __init__(self, database_path: str, *, busy_timeout_ms: int = 5000) -> None:
        self.database_path = database_path
        self.busy_timeout_ms = busy_timeout_ms

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.database_path) as db:
            await self._configure(db)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sentinel_evaluations (
                    run_id TEXT PRIMARY KEY,
                    route_id TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    readiness TEXT NOT NULL,
                    started_at INTEGER NOT NULL,
                    completed_at INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    summary TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sentinel_probe_results (
                    run_id TEXT NOT NULL,
                    check_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    evidence TEXT NOT NULL,
                    remediation TEXT NOT NULL,
                    PRIMARY KEY (run_id, check_id),
                    FOREIGN KEY (run_id) REFERENCES sentinel_evaluations(run_id)
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sentinel_route_completed
                ON sentinel_evaluations (route_id, completed_at DESC)
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sentinel_consumer_receipts (
                    run_id TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    consumer_id TEXT,
                    profile_id TEXT,
                    status TEXT NOT NULL,
                    policy_verdict TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    attempts INTEGER NOT NULL,
                    fallback_used INTEGER NOT NULL,
                    fallback_reason TEXT NOT NULL,
                    stream INTEGER NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    tool_policy TEXT NOT NULL,
                    request_path TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sentinel_receipts_created
                ON sentinel_consumer_receipts (created_at DESC)
                """
            )
            await db.commit()

    async def save(self, evaluation: SentinelEvaluation) -> None:
        async with aiosqlite.connect(self.database_path) as db:
            await self._configure(db)
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                """
                INSERT INTO sentinel_evaluations (
                    run_id, route_id, provider_name, model_id, score, readiness,
                    started_at, completed_at, duration_ms, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation.run_id,
                    evaluation.route_id,
                    evaluation.provider_name,
                    evaluation.model_id,
                    evaluation.score,
                    evaluation.readiness,
                    evaluation.started_at,
                    evaluation.completed_at,
                    evaluation.duration_ms,
                    evaluation.summary,
                ),
            )
            await db.executemany(
                """
                INSERT INTO sentinel_probe_results (
                    run_id, check_id, label, status, score, latency_ms, evidence, remediation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        evaluation.run_id,
                        probe.check_id,
                        probe.label,
                        probe.status,
                        probe.score,
                        probe.latency_ms,
                        probe.evidence,
                        probe.remediation,
                    )
                    for probe in evaluation.probes
                ],
            )
            await db.commit()

    async def latest_for_routes(
        self, route_ids: list[str]
    ) -> dict[str, SentinelEvaluation]:
        if not route_ids:
            return {}
        placeholders = ",".join("?" for _ in route_ids)
        async with aiosqlite.connect(self.database_path) as db:
            await self._configure(db)
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"""
                SELECT e.*
                FROM sentinel_evaluations e
                JOIN (
                    SELECT route_id, MAX(completed_at) AS completed_at
                    FROM sentinel_evaluations
                    WHERE route_id IN ({placeholders})
                    GROUP BY route_id
                ) latest
                ON latest.route_id = e.route_id
                AND latest.completed_at = e.completed_at
                ORDER BY e.completed_at DESC, e.run_id DESC
                """,
                route_ids,
            )
            rows = await cursor.fetchall()
            evaluations: dict[str, SentinelEvaluation] = {}
            for row in rows:
                if row["route_id"] in evaluations:
                    continue
                evaluations[row["route_id"]] = await self._row_to_evaluation(db, row)
            return evaluations

    async def history(self, route_id: str, *, limit: int = 10) -> list[SentinelEvaluation]:
        async with aiosqlite.connect(self.database_path) as db:
            await self._configure(db)
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM sentinel_evaluations
                WHERE route_id = ?
                ORDER BY completed_at DESC, run_id DESC
                LIMIT ?
                """,
                (route_id, max(1, min(limit, 50))),
            )
            rows = await cursor.fetchall()
            return [await self._row_to_evaluation(db, row) for row in rows]

    async def save_receipt(self, receipt: dict) -> None:
        """Persist only the safe metadata produced by consumer_contracts.safe_receipt."""
        async with aiosqlite.connect(self.database_path) as db:
            await self._configure(db)
            await db.execute(
                """
                INSERT OR REPLACE INTO sentinel_consumer_receipts (
                    run_id, created_at, consumer_id, profile_id, status, policy_verdict,
                    provider_name, route_id, model_id, latency_ms, attempts,
                    fallback_used, fallback_reason, stream, capabilities_json,
                    tool_policy, request_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt["run_id"],
                    receipt["created_at"],
                    receipt.get("consumer_id"),
                    receipt.get("profile_id"),
                    receipt["status"],
                    receipt["policy_verdict"],
                    receipt.get("provider_name", ""),
                    receipt.get("route_id", ""),
                    receipt.get("model_id", ""),
                    receipt.get("latency_ms", 0),
                    receipt.get("attempts", 0),
                    int(bool(receipt.get("fallback_used"))),
                    receipt.get("fallback_reason", ""),
                    int(bool(receipt.get("stream"))),
                    json.dumps(receipt.get("capabilities", []), separators=(",", ":")),
                    receipt.get("tool_policy", "unmanaged"),
                    receipt.get("request_path", ""),
                ),
            )
            await db.commit()

    async def recent_receipts(self, *, limit: int = 12) -> list[dict]:
        async with aiosqlite.connect(self.database_path) as db:
            await self._configure(db)
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM sentinel_consumer_receipts
                ORDER BY created_at DESC, run_id DESC
                LIMIT ?
                """,
                (max(1, min(limit, 50)),),
            )
            rows = await cursor.fetchall()
        receipts: list[dict] = []
        for row in rows:
            item = dict(row)
            item["fallback_used"] = bool(item["fallback_used"])
            item["stream"] = bool(item["stream"])
            item["capabilities"] = json.loads(item.pop("capabilities_json"))
            receipts.append(item)
        return receipts


    async def _row_to_evaluation(
        self, db: aiosqlite.Connection, row: aiosqlite.Row
    ) -> SentinelEvaluation:
        cursor = await db.execute(
            """
            SELECT * FROM sentinel_probe_results
            WHERE run_id = ?
            ORDER BY rowid
            """,
            (row["run_id"],),
        )
        probe_rows = await cursor.fetchall()
        probes = [
            SentinelProbeResult(
                check_id=probe["check_id"],
                label=probe["label"],
                status=probe["status"],
                score=probe["score"],
                latency_ms=probe["latency_ms"],
                evidence=probe["evidence"],
                remediation=probe["remediation"],
            )
            for probe in probe_rows
        ]
        return SentinelEvaluation(
            run_id=row["run_id"],
            route_id=row["route_id"],
            provider_name=row["provider_name"],
            model_id=row["model_id"],
            score=row["score"],
            readiness=row["readiness"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            duration_ms=row["duration_ms"],
            summary=row["summary"],
            probes=probes,
        )

    async def _configure(self, db: aiosqlite.Connection) -> None:
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute("PRAGMA synchronous = NORMAL")
        await db.execute(f"PRAGMA busy_timeout = {max(0, int(self.busy_timeout_ms))}")
        await db.execute("PRAGMA foreign_keys = ON")


def evidence_json(evaluation: SentinelEvaluation | None) -> str:
    """Stable JSON representation useful to local scripts and diagnostics."""
    return json.dumps(evaluation.to_dict() if evaluation else None, indent=2, sort_keys=True)
