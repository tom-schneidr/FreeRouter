from app.api.stream_monitor import StreamMonitorTracker
from app.router import RouteStreamDiag


def _diag(event_type: str, **kwargs) -> RouteStreamDiag:
    return RouteStreamDiag(
        event_type=event_type,
        provider_name=kwargs.get("provider_name", "demo"),
        route_id=kwargs.get("route_id", "route-a"),
        model_id=kwargs.get("model_id", "model-a"),
        reason=kwargs.get("reason"),
        status_code=kwargs.get("status_code"),
        usage=kwargs.get("usage"),
    )


def test_stream_monitor_tracker_collects_attempts_and_text() -> None:
    tracker = StreamMonitorTracker(requested_model="auto")
    tracker.record_diag(_diag("route_trying"))
    tracker.record_diag(_diag("route_skipped", reason="rate_limited"))
    tracker.record_diag(
        _diag(
            "route_selected",
            provider_name="openrouter",
            route_id="r1",
            model_id="gpt-test",
        )
    )
    tracker.record_openai_sse('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n')
    tracker.record_openai_sse('data: {"choices":[{"delta":{"content":" world"}}]}\n\n')
    tracker.record_diag(_diag("usage_summary", usage={"prompt_tokens": 3, "completion_tokens": 2}))

    payload = tracker.completed_payload(status_code=200, latency_ms=42)

    assert payload["assistant_text"] == "Hello world"
    assert payload["provider_name"] == "openrouter"
    assert payload["route_id"] == "r1"
    assert payload["model_id"] == "gpt-test"
    assert payload["attempts"] == 3
    assert [attempt["status"] for attempt in payload["attempts_detail"]] == [
        "trying",
        "rate_limited",
        "selected",
    ]
    assert payload["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert payload["response_body"]["choices"][0]["message"]["content"] == "Hello world"


def test_stream_monitor_tracker_flushes_final_sse_line_without_blank_line() -> None:
    tracker = StreamMonitorTracker(requested_model="auto")
    tracker.record_openai_sse('data: {"choices":[{"delta":{"content":"tail"}}]}\n')

    payload = tracker.completed_payload(status_code=200, latency_ms=1)

    assert payload["assistant_text"] == "tail"
