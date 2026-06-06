from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from app.request_limiter import GatewayRequestLimiter
from app.state import ProviderQuota, StateManager


@pytest.mark.asyncio
async def test_concurrency_limiter_queue_timeout():
    """Verify that GatewayRequestLimiter properly times out requests when slot is held."""
    # Max 1 concurrent request, queue timeout 0.05 seconds
    limiter = GatewayRequestLimiter(max_concurrent_requests=1, queue_timeout_seconds=0.05)

    # First request acquires immediately
    assert await limiter.acquire() is True

    # Second request tries to acquire but should timeout and return False
    acquired_second = await limiter.acquire()
    assert acquired_second is False

    # Release the slot and verify a new request can acquire
    limiter.release()
    assert await limiter.acquire() is True
    limiter.release()


@pytest.mark.asyncio
async def test_concurrency_limiter_max_waiting_requests():
    """Verify that GatewayRequestLimiter rejects requests immediately when wait queue is full."""
    # Max 1 concurrent request, max 1 waiting request
    limiter = GatewayRequestLimiter(
        max_concurrent_requests=1,
        queue_timeout_seconds=2.0,
        max_waiting_requests=1
    )

    # 1. Acquire the sole concurrent slot
    assert await limiter.acquire() is True

    # Start a background task to queue up as the 1 waiting request
    waiter_task = asyncio.create_task(limiter.acquire())
    # Yield control to let waiter_task start and increment waiting requests count
    await asyncio.sleep(0.01)

    # 2. This request should fail fast immediately because max_waiting_requests (1) is already reached
    immediate_reject = await limiter.acquire()
    assert immediate_reject is False

    # 3. Clean up the waiter task
    limiter.release()  # release the first slot, letting the waiter acquire it
    assert await waiter_task is True
    limiter.release()


@pytest.mark.asyncio
async def test_sqlite_busy_timeout_concurrency(tmp_path):
    """Verify SQLite busy timeout settings by performing heavy concurrent writes on the StateManager database."""
    db_file = str(tmp_path / "busy_test.sqlite3")
    
    # Create StateManager with small busy timeout to test robustness or default 5000ms
    quotas = [
        ProviderQuota("groq", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)
    ]
    state = StateManager(db_file, quotas, busy_timeout_ms=5000)
    await state.initialize()

    # Generate heavy concurrent writes on the same database
    async def run_concurrent_writes(worker_id: int):
        for i in range(20):
            # Record success or mark rate limited, both do writes to the DB
            await state.record_route_success(
                route_id=f"route-{worker_id}",
                provider_name="groq",
                model_id="llama3",
                usage={"total_tokens": 10},
                status_code=200
            )
            # Sleep briefly to interleave tasks
            await asyncio.sleep(0.001)

    # Run 10 concurrent tasks doing writes
    tasks = [run_concurrent_writes(i) for i in range(10)]
    # All tasks should complete without SQLite3 OperationalError (database is locked)
    await asyncio.gather(*tasks)

    # Verify connection mode and busy timeout pragma directly
    async with aiosqlite.connect(db_file) as db:
        async with db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
            assert row[0].lower() == "wal"

        async with db.execute("PRAGMA busy_timeout") as cursor:
            row = await cursor.fetchone()
            # SQLite might return slightly different numbers or exact setting depending on OS/version
            assert row[0] >= 5000
