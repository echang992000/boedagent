"""OpenAI Responses API adapter."""

from __future__ import annotations

import json
from typing import Any

from boed_agent.models import Message, ProviderResponse, ToolCall
from boed_agent.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

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
        payload: dict[str, Any] = {"model": self.model, "input": [], "tools": tools}
        if system_prompt:
            payload["input"].append({"role": "system", "content": system_prompt})
        for message in messages:
            if message.role == "tool":
                payload["input"].append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
            else:
                payload["input"].append({"role": message.role, "content": message.content})
        return payload

    def parse_response(self, response: Any) -> ProviderResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        output_items = getattr(response, "output", None)
        if output_items is None and isinstance(response, dict):
            output_items = response.get("output", [])
        for item in output_items or []:
            item_type = getattr(item, "type", None) or item.get("type")
            if item_type == "function_call":
                tool_calls.append(self.normalize_tool_call(item))
                continue
            if item_type == "message":
                content = getattr(item, "content", None) or item.get("content", [])
                for block in content:
                    block_type = getattr(block, "type", None) or block.get("type")
                    if block_type in {"output_text", "text"}:
                        text = getattr(block, "text", None) or block.get("text", "")
                        if isinstance(text, dict):
                            text = text.get("value", "")
                        text_parts.append(str(text))
            elif item_type in {"output_text", "text"}:
                text_parts.append(str(getattr(item, "text", None) or item.get("text", "")))
        stop_reason = getattr(response, "status", None) or (response.get("status") if isinstance(response, dict) else None)
        return ProviderResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=response,
        )

    def normalize_tool_call(self, raw_call: Any) -> ToolCall:
        call_id = getattr(raw_call, "call_id", None) or raw_call.get("call_id") or raw_call.get("id")
        name = getattr(raw_call, "name", None) or raw_call.get("name")
        arguments = getattr(raw_call, "arguments", None) or raw_call.get("arguments", {})
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
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "OpenAI support requires the optional `openai` dependency. Install with `.[openai]`."
                ) from exc
            client = OpenAI(api_key=self.api_key)
            self._client = client
        return client.responses.create(**request)
