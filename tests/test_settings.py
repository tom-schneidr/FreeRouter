from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.settings import Settings


def test_settings_reject_invalid_local_operational_values():
    with pytest.raises(ValidationError):
        Settings(max_concurrent_requests=0)

    with pytest.raises(ValidationError):
        Settings(request_timeout_seconds=-1)

    with pytest.raises(ValidationError):
        Settings(request_queue_max_waiting_requests=0)


def test_settings_accept_valid_local_operational_values():
    settings = Settings(
        max_concurrent_requests=1,
        request_timeout_seconds=1,
        request_queue_max_waiting_requests=None,
        sse_chunk_replay_sleep_seconds=0,
    )

    assert settings.max_concurrent_requests == 1
    assert settings.request_queue_max_waiting_requests is None
