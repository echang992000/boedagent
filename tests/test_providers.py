from __future__ import annotations

from boed_agent.providers.claude_provider import ClaudeProvider
from boed_agent.providers.openai_provider import OpenAIProvider
from boed_agent.tools.registry import ToolDefinition


def make_tool() -> ToolDefinition:
    return ToolDefinition(
        name="list_backends",
        description="List backends",
        input_schema={"type": "object", "properties": {}},
        handler=lambda _: {"backends": []},
    )


def test_openai_build_request_and_parse_response() -> None:
    provider = OpenAIProvider(model="gpt-test", client=object())
    request = provider.build_request(
        messages=[],
        tools=[make_tool().to_openai_schema()],
        system_prompt="system prompt",
        state=None,
    )

    assert request["model"] == "gpt-test"
    assert request["tools"][0]["type"] == "function"
    assert request["input"][0]["role"] == "system"

    response = provider.parse_response(
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Need clarification."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "list_backends",
                    "arguments": "{}",
                },
            ],
            "status": "completed",
        }
    )

    assert response.text == "Need clarification."
    assert response.tool_calls[0].name == "list_backends"


def test_claude_build_request_and_parse_response() -> None:
    provider = ClaudeProvider(model="claude-test", client=object())
    request = provider.build_request(
        messages=[],
        tools=[make_tool().to_claude_schema()],
        system_prompt="system prompt",
        state=None,
    )

    assert request["model"] == "claude-test"
    assert request["tools"][0]["name"] == "list_backends"

    response = provider.parse_response(
        {
            "content": [
                {"type": "text", "text": "I need more information."},
                {"type": "tool_use", "id": "tool_1", "name": "list_backends", "input": {}},
            ],
            "stop_reason": "tool_use",
        }
    )

    assert response.text == "I need more information."
    assert response.tool_calls[0].id == "tool_1"
