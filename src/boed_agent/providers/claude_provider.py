"""Anthropic Claude Messages API adapter."""

from __future__ import annotations

import json
from typing import Any

from boed_agent.models import Message, ProviderResponse, ToolCall
from boed_agent.providers.base import LLMProvider


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str, api_key: str | None = None, client: Any | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self._client = client

    def build_request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system_prompt: str,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system_prompt,
            "messages": [],
            "tools": tools,
            "max_tokens": 1024,
        }
        for message in messages:
            if message.role == "tool":
                payload["messages"].append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.content,
                            }
                        ],
                    }
                )
            else:
                payload["messages"].append(
                    {
                        "role": message.role,
                        "content": [{"type": "text", "text": message.content}],
                    }
                )
        return payload

    def parse_response(self, response: Any) -> ProviderResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content = getattr(response, "content", None)
        if content is None and isinstance(response, dict):
            content = response.get("content", [])
        for block in content or []:
            block_type = getattr(block, "type", None) or block.get("type")
            if block_type == "text":
                text_parts.append(str(getattr(block, "text", None) or block.get("text", "")))
            elif block_type == "tool_use":
                tool_calls.append(self.normalize_tool_call(block))
        stop_reason = getattr(response, "stop_reason", None) or (
            response.get("stop_reason") if isinstance(response, dict) else None
        )
        return ProviderResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=response,
        )

    def normalize_tool_call(self, raw_call: Any) -> ToolCall:
        call_id = getattr(raw_call, "id", None) or raw_call.get("id")
        name = getattr(raw_call, "name", None) or raw_call.get("name")
        arguments = getattr(raw_call, "input", None) or raw_call.get("input", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        return ToolCall(id=str(call_id), name=str(name), arguments=dict(arguments))

    def continue_with_tool_results(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system_prompt: str,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.build_request(messages, tools, system_prompt, state)

    def generate(self, request: dict[str, Any]) -> Any:
        client = self._client
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "Claude support requires the optional `anthropic` dependency. Install with `.[anthropic]`."
                ) from exc
            client = Anthropic(api_key=self.api_key)
            self._client = client
        return client.messages.create(**request)
