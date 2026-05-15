"""Provider adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from boed_agent.models import Message, ProviderResponse, ToolCall


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def build_request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system_prompt: str,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a provider request payload."""

    @abstractmethod
    def parse_response(self, response: Any) -> ProviderResponse:
        """Normalize a provider response."""

    @abstractmethod
    def normalize_tool_call(self, raw_call: Any) -> ToolCall:
        """Normalize a provider-specific tool call."""

    @abstractmethod
    def continue_with_tool_results(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system_prompt: str,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a follow-up request after tool execution."""

    @abstractmethod
    def generate(self, request: dict[str, Any]) -> Any:
        """Execute the provider request."""
