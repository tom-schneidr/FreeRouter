from app.request_sizing import estimate_prompt_tokens, estimate_total_request_tokens


def test_estimate_prompt_tokens_includes_tools_and_tool_calls():
    payload = {
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{\"command\":\"ls\"}"},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "exec",
                    "description": "Run a shell command",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ],
        "max_completion_tokens": 4096,
    }

    prompt_only = estimate_prompt_tokens({"messages": payload["messages"]})
    with_tools = estimate_prompt_tokens(payload)
    prompt_tokens, total_tokens = estimate_total_request_tokens(payload)

    assert with_tools > prompt_only
    assert total_tokens == prompt_tokens + 4096
