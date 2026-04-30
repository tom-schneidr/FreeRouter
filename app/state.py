from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import time
from typing import Mapping

import aiosqlite


@dataclass(frozen=True)
class ProviderQuota:
    """Static rate-limit configuration for a single provider."""
    name: str
    tokens_per_day: int | None
    requests_per_day: int | None
    requests_per_minute: int | None


@dataclass(frozen=True)
class ProviderState:
    """Snapshot of a provider's current usage counters and cooldown status."""
    provider_name: str
    tokens_used_today: int
    requests_today: int
    requests_this_minute: int
    minute_window_start: int
    cooldown_until: int
    day: str


@dataclass(frozen=True)
class Availability:
    """Result of a quota availability check."""
    available: bool
    reason: str | None = None
    retry_after_seconds: int | None = None


@dataclass(frozen=True)
class RouteState:
    """Health state for one provider/model route."""

    route_id: str
    provider_name: str
    model_id: str
    status: str
    status_reason: str | None
    consecutive_failures: int
    rate_limited_until: int
    next_probe_at: int
    updated_at: int


class StateManager:
    """SQLite-backed local quota tracker.

    Provider rate limits are checked locally before network calls. Successful response headers and
    usage payloads then reconcile the local view with the provider's authoritative counters.
    """

    def __init__(self, database_path: str, quotas: list[ProviderQuota]) -> None:
        self.database_path = database_path
        self.quotas = {quota.name: quota for quota in quotas}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        directory = os.path.dirname(self.database_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_state (
                    provider_name TEXT PRIMARY KEY,
                    tokens_used_today INTEGER NOT NULL DEFAULT 0,
                    requests_today INTEGER NOT NULL DEFAULT 0,
                    requests_this_minute INTEGER NOT NULL DEFAULT 0,
                    minute_window_start INTEGER NOT NULL DEFAULT 0,
                    cooldown_until INTEGER NOT NULL DEFAULT 0,
                    day TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    status_code INTEGER,
                    created_at INTEGER NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS route_state (
                    route_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    status_reason TEXT,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    rate_limited_until INTEGER NOT NULL DEFAULT 0,
                    next_probe_at INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            today = self._today()
            now = self._now()
            for quota in self.quotas.values():
                await db.execute(
                    """
                    INSERT OR IGNORE INTO provider_state (
                        provider_name,
                        minute_window_start,
                        day
                    ) VALUES (?, ?, ?)
                    """,
                    (quota.name, now, today),
                )
            await db.commit()

    async def get_state(self, provider_name: str) -> ProviderState:
        await self._ensure_current_windows(provider_name)
        async with aiosqlite.connect(self.database_path) as db:
            row = await self._fetch_state(db, provider_name)
        return self._row_to_state(row)

    async def check_available(
        self,
        provider_name: str,
        estimated_tokens: int = 0,
    ) -> Availability:
        await self._ensure_current_windows(provider_name)
        async with aiosqlite.connect(self.database_path) as db:
            row = await self._fetch_state(db, provider_name)
        return self._availability_for_state(
            self.quotas[provider_name],
            self._row_to_state(row),
            estimated_tokens,
        )

    async def try_reserve_request(
        self,
        provider_name: str,
        estimated_tokens: int = 0,
    ) -> Availability:
        """Atomically check quota windows and reserve one outbound request.

        This prevents concurrent requests from all observing the same available RPM slot before
        incrementing the local counter.
        """
        async with self._lock:
            async with aiosqlite.connect(self.database_path) as db:
                await self._ensure_current_windows_on_connection(db, provider_name)
                row = await self._fetch_state(db, provider_name)
                state = self._row_to_state(row)
                availability = self._availability_for_state(
                    self.quotas[provider_name],
                    state,
                    estimated_tokens,
                )
                if not availability.available:
                    return availability

                await db.execute(
                    """
                    UPDATE provider_state
                    SET requests_today = requests_today + 1,
                        requests_this_minute = requests_this_minute + 1
                    WHERE provider_name = ?
                    """,
                    (provider_name,),
                )
                await db.commit()

        return Availability(available=True)

    async def check_route_available(
        self,
        route_id: str,
        provider_name: str,
        model_id: str,
        *,
        allow_rate_limit_probe: bool = False,
    ) -> Availability:
        async with aiosqlite.connect(self.database_path) as db:
            await self._ensure_route_state_on_connection(db, route_id, provider_name, model_id)
            row = await self._fetch_route_state(db, route_id)
        state = self._row_to_route_state(row)
        now = self._now()

        if state.status == "potentially_outdated":
            return Availability(available=False, reason="potentially_outdated")

        if state.status in {"rate_limited", "too_slow"}:
            retry_after = max(1, max(state.rate_limited_until, state.next_probe_at) - now)
            if not allow_rate_limit_probe or state.next_probe_at > now:
                reason = "route_rate_limited" if state.status == "rate_limited" else "route_too_slow"
                return Availability(
                    available=False,
                    reason=reason,
                    retry_after_seconds=retry_after,
                )
            reason = "rate_limit_probe" if state.status == "rate_limited" else "too_slow_probe"
            return Availability(available=True, reason=reason)

        return Availability(available=True)

    async def get_route_state(
        self,
        route_id: str,
        provider_name: str,
        model_id: str,
    ) -> RouteState:
        async with aiosqlite.connect(self.database_path) as db:
            await self._ensure_route_state_on_connection(db, route_id, provider_name, model_id)
            row = await self._fetch_route_state(db, route_id)
        return self._row_to_route_state(row)

    async def record_route_success(
        self,
        route_id: str,
        provider_name: str,
        model_id: str,
    ) -> None:
        async with self._lock:
            async with aiosqlite.connect(self.database_path) as db:
                await self._ensure_route_state_on_connection(db, route_id, provider_name, model_id)
                await db.execute(
                    """
                    UPDATE route_state
                    SET status = 'active',
                        status_reason = NULL,
                        consecutive_failures = 0,
                        rate_limited_until = 0,
                        next_probe_at = 0,
                        updated_at = ?
                    WHERE route_id = ?
                    """,
                    (self._now(), route_id),
                )
                await db.commit()

    async def clear_route_health(
        self,
        route_id: str,
        provider_name: str,
        model_id: str,
    ) -> RouteState:
        async with self._lock:
            async with aiosqlite.connect(self.database_path) as db:
                await self._ensure_route_state_on_connection(db, route_id, provider_name, model_id)
                await db.execute(
                    """
                    UPDATE route_state
                    SET status = 'active',
                        status_reason = NULL,
                        consecutive_failures = 0,
                        rate_limited_until = 0,
                        next_probe_at = 0,
                        updated_at = ?
                    WHERE route_id = ?
                    """,
                    (self._now(), route_id),
                )
                await db.commit()
                row = await self._fetch_route_state(db, route_id)
        return self._row_to_route_state(row)

    async def mark_route_rate_limited(
        self,
        route_id: str,
        provider_name: str,
        model_id: str,
        headers: Mapping[str, str] | None = None,
        status_code: int | None = None,
    ) -> RouteState:
        headers = headers or {}
        async with self._lock:
            async with aiosqlite.connect(self.database_path) as db:
                await self._ensure_route_state_on_connection(db, route_id, provider_name, model_id)
                row = await self._fetch_route_state(db, route_id)
                state = self._row_to_route_state(row)
                failures = state.consecutive_failures + 1 if state.status == "rate_limited" else 1
                fallback_cooldown = min(3600, 60 * (2 ** min(failures - 1, 5)))
                rate_limited_until = self._cooldown_from_headers(headers) or self._now() + fallback_cooldown
                next_probe_at = rate_limited_until + self._route_probe_jitter(route_id)
                await db.execute(
                    """
                    UPDATE route_state
                    SET status = 'rate_limited',
                        status_reason = ?,
                        consecutive_failures = ?,
                        rate_limited_until = ?,
                        next_probe_at = ?,
                        updated_at = ?
                    WHERE route_id = ?
                    """,
                    (
                        f"provider_{status_code or 429}",
                        failures,
                        rate_limited_until,
                        next_probe_at,
                        self._now(),
                        route_id,
                    ),
                )
                await db.commit()
                row = await self._fetch_route_state(db, route_id)
        return self._row_to_route_state(row)

    async def mark_route_not_found(
        self,
        route_id: str,
        provider_name: str,
        model_id: str,
        *,
        status_code: int | None = None,
        threshold: int = 2,
    ) -> RouteState:
        async with self._lock:
            async with aiosqlite.connect(self.database_path) as db:
                await self._ensure_route_state_on_connection(db, route_id, provider_name, model_id)
                row = await self._fetch_route_state(db, route_id)
                state = self._row_to_route_state(row)
                failures = state.consecutive_failures + 1 if (state.status_reason or "").startswith("model_not_found") else 1
                status = "potentially_outdated" if failures >= threshold else "active"
                await db.execute(
                    """
                    UPDATE route_state
                    SET status = ?,
                        status_reason = ?,
                        consecutive_failures = ?,
                        rate_limited_until = 0,
                        next_probe_at = 0,
                        updated_at = ?
                    WHERE route_id = ?
                    """,
                    (
                        status,
                        f"model_not_found_{status_code or 'unknown'}",
                        failures,
                        self._now(),
                        route_id,
                    ),
                )
                await db.commit()
                row = await self._fetch_route_state(db, route_id)
        return self._row_to_route_state(row)

    async def mark_route_timeout(
        self,
        route_id: str,
        provider_name: str,
        model_id: str,
        *,
        threshold: int = 2,
    ) -> RouteState:
        async with self._lock:
            async with aiosqlite.connect(self.database_path) as db:
                await self._ensure_route_state_on_connection(db, route_id, provider_name, model_id)
                row = await self._fetch_route_state(db, route_id)
                state = self._row_to_route_state(row)
                failures = state.consecutive_failures + 1 if state.status_reason == "request_timeout" else 1
                status = "too_slow" if failures >= threshold else "active"
                cooldown_seconds = min(3600, 120 * (2 ** max(0, failures - threshold)))
                next_probe_at = self._now() + cooldown_seconds + self._route_probe_jitter(route_id)
                await db.execute(
                    """
                    UPDATE route_state
                    SET status = ?,
                        status_reason = 'request_timeout',
                        consecutive_failures = ?,
                        rate_limited_until = ?,
                        next_probe_at = ?,
                        updated_at = ?
                    WHERE route_id = ?
                    """,
                    (
                        status,
                        failures,
                        next_probe_at if status == "too_slow" else 0,
                        next_probe_at if status == "too_slow" else 0,
                        self._now(),
                        route_id,
                    ),
                )
                await db.commit()
                row = await self._fetch_route_state(db, route_id)
        return self._row_to_route_state(row)

    def _availability_for_state(
        self,
        quota: ProviderQuota,
        state: ProviderState,
        estimated_tokens: int = 0,
    ) -> Availability:
        now = self._now()

        if state.cooldown_until > now:
            return Availability(
                available=False,
                reason="cooldown",
                retry_after_seconds=state.cooldown_until - now,
            )

        if quota.tokens_per_day is not None:
            if state.tokens_used_today + max(estimated_tokens, 0) > quota.tokens_per_day:
                return Availability(available=False, reason="daily_token_limit")

        if quota.requests_per_day is not None and state.requests_today >= quota.requests_per_day:
            return Availability(available=False, reason="daily_request_limit")

        if (
            quota.requests_per_minute is not None
            and state.requests_this_minute >= quota.requests_per_minute
        ):
            retry_after = max(1, 60 - (now - state.minute_window_start))
            return Availability(
                available=False,
                reason="rpm_limit",
                retry_after_seconds=retry_after,
            )

        return Availability(available=True)



    async def record_success(
        self,
        provider_name: str,
        usage: Mapping[str, int] | None,
        headers: Mapping[str, str],
        status_code: int,
    ) -> None:
        usage = usage or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)

        async with self._lock:
            async with aiosqlite.connect(self.database_path) as db:
                await self._ensure_current_windows_on_connection(db, provider_name)
                await db.execute(
                    """
                    UPDATE provider_state
                    SET tokens_used_today = tokens_used_today + ?
                    WHERE provider_name = ?
                    """,
                    (total_tokens, provider_name),
                )
                await db.execute(
                    """
                    INSERT INTO provider_events (
                        provider_name,
                        event_type,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        status_code,
                        created_at
                    ) VALUES (?, 'success', ?, ?, ?, ?, ?)
                    """,
                    (
                        provider_name,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        status_code,
                        self._now(),
                    ),
                )

                cooldown_until = self._cooldown_from_headers(headers)
                remaining_tokens = self._int_header(headers, "x-ratelimit-remaining-tokens")
                if remaining_tokens is not None and remaining_tokens <= 0:
                    await db.execute(
                        "UPDATE provider_state SET cooldown_until = ? WHERE provider_name = ?",
                        (cooldown_until or self._now() + 60, provider_name),
                    )
                elif cooldown_until is not None:
                    remaining_requests = self._int_header(
                        headers,
                        "x-ratelimit-remaining-requests",
                    )
                    if remaining_requests is not None and remaining_requests <= 0:
                        await db.execute(
                            "UPDATE provider_state SET cooldown_until = ? WHERE provider_name = ?",
                            (cooldown_until, provider_name),
                        )

                await db.commit()

    async def mark_exhausted(
        self,
        provider_name: str,
        headers: Mapping[str, str] | None = None,
        cooldown_seconds: int = 60,
        status_code: int | None = None,
    ) -> None:
        headers = headers or {}
        cooldown_until = self._cooldown_from_headers(headers) or self._now() + cooldown_seconds
        async with self._lock:
            async with aiosqlite.connect(self.database_path) as db:
                await self._ensure_current_windows_on_connection(db, provider_name)
                await db.execute(
                    """
                    UPDATE provider_state
                    SET cooldown_until = ?
                    WHERE provider_name = ?
                    """,
                    (cooldown_until, provider_name),
                )
                await db.execute(
                    """
                    INSERT INTO provider_events (
                        provider_name,
                        event_type,
                        status_code,
                        created_at
                    ) VALUES (?, 'exhausted', ?, ?)
                    """,
                    (provider_name, status_code, self._now()),
                )
                await db.commit()

    async def _ensure_current_windows(self, provider_name: str) -> None:
        async with aiosqlite.connect(self.database_path) as db:
            await self._ensure_current_windows_on_connection(db, provider_name)

    async def _ensure_current_windows_on_connection(
        self,
        db: aiosqlite.Connection,
        provider_name: str,
    ) -> None:
        today = self._today()
        now = self._now()
        row = await self._fetch_state(db, provider_name)
        state = self._row_to_state(row)

        if state.day != today:
            await db.execute(
                """
                UPDATE provider_state
                SET tokens_used_today = 0,
                    requests_today = 0,
                    requests_this_minute = 0,
                    minute_window_start = ?,
                    cooldown_until = 0,
                    day = ?
                WHERE provider_name = ?
                """,
                (now, today, provider_name),
            )
        elif now - state.minute_window_start >= 60:
            await db.execute(
                """
                UPDATE provider_state
                SET requests_this_minute = 0,
                    minute_window_start = ?
                WHERE provider_name = ?
                """,
                (now, provider_name),
            )
        await db.commit()

    async def _fetch_state(self, db: aiosqlite.Connection, provider_name: str) -> aiosqlite.Row:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT provider_name,
                   tokens_used_today,
                   requests_today,
                   requests_this_minute,
                   minute_window_start,
                   cooldown_until,
                   day
            FROM provider_state
            WHERE provider_name = ?
            """,
            (provider_name,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Unknown provider: {provider_name}")
        return row

    async def _ensure_route_state_on_connection(
        self,
        db: aiosqlite.Connection,
        route_id: str,
        provider_name: str,
        model_id: str,
    ) -> None:
        await db.execute(
            """
            INSERT OR IGNORE INTO route_state (
                route_id,
                provider_name,
                model_id,
                updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (route_id, provider_name, model_id, self._now()),
        )
        await db.execute(
            """
            UPDATE route_state
            SET provider_name = ?,
                model_id = ?
            WHERE route_id = ?
            """,
            (provider_name, model_id, route_id),
        )
        await db.commit()

    async def _fetch_route_state(self, db: aiosqlite.Connection, route_id: str) -> aiosqlite.Row:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT route_id,
                   provider_name,
                   model_id,
                   status,
                   status_reason,
                   consecutive_failures,
                   rate_limited_until,
                   next_probe_at,
                   updated_at
            FROM route_state
            WHERE route_id = ?
            """,
            (route_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Unknown route: {route_id}")
        return row

    @staticmethod
    def _row_to_state(row: aiosqlite.Row) -> ProviderState:
        return ProviderState(
            provider_name=row["provider_name"],
            tokens_used_today=row["tokens_used_today"],
            requests_today=row["requests_today"],
            requests_this_minute=row["requests_this_minute"],
            minute_window_start=row["minute_window_start"],
            cooldown_until=row["cooldown_until"],
            day=row["day"],
        )

    @staticmethod
    def _row_to_route_state(row: aiosqlite.Row) -> RouteState:
        return RouteState(
            route_id=row["route_id"],
            provider_name=row["provider_name"],
            model_id=row["model_id"],
            status=row["status"],
            status_reason=row["status_reason"],
            consecutive_failures=row["consecutive_failures"],
            rate_limited_until=row["rate_limited_until"],
            next_probe_at=row["next_probe_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _now() -> int:
        return int(time())

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).date().isoformat()

    @staticmethod
    def _route_probe_jitter(route_id: str) -> int:
        return sum(ord(char) for char in route_id) % 60

    @classmethod
    def _cooldown_from_headers(cls, headers: Mapping[str, str]) -> int | None:
        retry_after = cls._int_header(headers, "retry-after")
        if retry_after is not None:
            return cls._now() + max(1, retry_after)

        for header_name in (
            "x-ratelimit-reset",
            "x-ratelimit-reset-tokens",
            "x-ratelimit-reset-requests",
        ):
            value = cls._header(headers, header_name)
            if not value:
                continue
            parsed = cls._parse_reset_header(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_reset_header(value: str) -> int | None:
        normalized = value.strip().lower()
        if normalized.isdigit():
            numeric = int(normalized)
            return numeric if numeric > 2_000_000_000 else int(time()) + numeric

        try:
            return int(parsedate_to_datetime(value).timestamp())
        except (TypeError, ValueError, IndexError):
            pass

        unit_seconds = {
            "ms": 0.001,
            "s": 1,
            "sec": 1,
            "second": 1,
            "seconds": 1,
            "m": 60,
            "min": 60,
            "minute": 60,
            "minutes": 60,
            "h": 3600,
            "hour": 3600,
            "hours": 3600,
        }
        parts = normalized.split()
        if len(parts) == 2:
            try:
                amount = float(parts[0])
            except ValueError:
                return None
            unit = unit_seconds.get(parts[1])
            if unit is not None:
                return int(time() + amount * unit)
        return None

    @classmethod
    def _int_header(cls, headers: Mapping[str, str], name: str) -> int | None:
        value = cls._header(headers, name)
        if value is None:
            return None
        try:
            return int(float(value.strip()))
        except ValueError:
            return None

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str | None:
        lower_name = name.lower()
        for key, value in headers.items():
            if key.lower() == lower_name:
                return value
        return None
