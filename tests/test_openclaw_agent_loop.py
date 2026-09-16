from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.model_catalog import ModelCatalog
from app.providers.base import ProviderResponse
from app.router import WaterfallRouter
from app.state import StateManager
from app.state_types import ProviderQuota


class ScriptedAgentProvider:
    api_key = "test-key"
    max_context_tokens = 128_000

    def __init__(self, name: str, responses: list[dict[str, Any]]) -> None:
        self.name = name
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    @property
    def is_configured(self) -> bool:
        return True

    async def chat_completion(self, client, payload, target_model=None):
        self.payloads.append(payload)
        return ProviderResponse(self.name, 200, {}, self.responses.pop(0))


async def _state(tmp_path, *provider_names: str) -> StateManager:
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota(name, None, None, None) for name in provider_names],
    )
    await state.initialize()
    return state


def _catalog(tmp_path, routes: list[tuple[str, str, int]]) -> ModelCatalog:
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": route_id,
                "provider_name": provider_name,
                "model_id": f"{provider_name}/model",
                "display_name": provider_name,
                "rank": rank,
                "enabled": True,
                "context_window": 128_000,
                "tags": ["text", "tool-use"],
                "capabilities": {
                    "text": {
                        "tag": "text",
                        "status": "supported",
                        "source": "probe",
                        "confidence": "high",
                        "checked_at": 1,
                    },
                    "tool-use": {
                        "tag": "tool-use",
                        "status": "supported",
                        "source": "probe",
                        "confidence": "high",
                        "checked_at": 1,
                    },
                },
            }
            for route_id, provider_name, rank in routes
        ]
    )
    return catalog


def _write_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    }


async def test_openclaw_style_tool_loop_completes_real_file_work(tmp_path):
    target = tmp_path / "openclaw-result.txt"
    provider = ScriptedAgentProvider(
        "primary",
        [
            {
                "id": "first",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_write_1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": json.dumps(
                                            {"path": str(target), "content": "done"}
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {
                "id": "second",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The file was written and verified.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        ],
    )
    state = await _state(tmp_path, "primary")
    router = WaterfallRouter(
        [provider],
        _catalog(tmp_path, [("primary-route", "primary", 1)]),
        state,
        request_timeout_seconds=5,
        reject_initial_action_promise=True,
    )
    user_message = {"role": "user", "content": "Write done to the requested file."}
    first_payload = {
        "model": "auto",
        "messages": [user_message],
        "tools": [_write_tool()],
        "tool_choice": "auto",
    }

    first = await router.route_chat_completion(first_payload)
    assistant = first.body["choices"][0]["message"]
    call = assistant["tool_calls"][0]
    arguments = json.loads(call["function"]["arguments"])
    Path(arguments["path"]).write_text(arguments["content"], encoding="utf-8")
    tool_result = {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": json.dumps({"ok": True, "bytes": target.stat().st_size}),
    }

    second = await router.route_chat_completion(
        {
            "model": "auto",
            "messages": [user_message, assistant, tool_result],
            "tools": [_write_tool()],
            "tool_choice": "auto",
        }
    )

    assert target.read_text(encoding="utf-8") == "done"
    assert second.body["choices"][0]["message"]["content"].endswith("verified.")
    assert len(provider.payloads) == 2
    reliability = await state.get_route_tool_reliability(["primary-route"])
    assert reliability["primary-route"].successes == 16


async def test_tool_request_prefers_empirically_reliable_route_over_catalog_rank(tmp_path):
    unreliable = ScriptedAgentProvider("unreliable", [])
    reliable_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_ok",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path":"x","content":"ok"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    reliable = ScriptedAgentProvider("reliable", [reliable_response])
    state = await _state(tmp_path, "unreliable", "reliable")
    catalog = _catalog(
        tmp_path,
        [("unreliable-route", "unreliable", 1), ("reliable-route", "reliable", 2)],
    )
    for _ in range(8):
        await state.record_route_tool_outcome(
            "unreliable-route", "unreliable", "unreliable/model", "schema_invalid"
        )
        await state.record_route_tool_outcome(
            "reliable-route", "reliable", "reliable/model", "valid_call"
        )
    router = WaterfallRouter(
        [unreliable, reliable],
        catalog,
        state,
        request_timeout_seconds=5,
    )

    result = await router.route_chat_completion(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "write it"}],
            "tools": [_write_tool()],
            "tool_choice": "required",
        }
    )

    assert result.provider_name == "reliable"
    assert not unreliable.payloads
