from __future__ import annotations

import json

from boed_agent.backends.registry import BackendRegistry
from boed_agent.clarification.planner import ClarificationPlanner
from boed_agent.core.runtime import AgentRuntime
from boed_agent.models import Message, ProviderResponse, ToolCall
from boed_agent.providers.base import LLMProvider
from boed_agent.tools.registry import build_default_tool_registry


class DummyProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        self.calls = 0

    def build_request(self, messages, tools, system_prompt, state=None):
        return {"messages": messages, "tools": tools, "system_prompt": system_prompt}

    def parse_response(self, response):
        return response

    def normalize_tool_call(self, raw_call):
        return raw_call

    def continue_with_tool_results(self, messages, tools, system_prompt, state=None):
        return {"messages": messages, "tools": tools, "system_prompt": system_prompt}

    def generate(self, request):
        self.calls += 1
        if self.calls == 1:
            return ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="optimize_design",
                        arguments={
                            "spec": {
                                "backend": "pyro",
                                "model_ref": "boed_agent.demo.pyro_linear:pyro_linear_model"
                            }
                        },
                    )
                ],
            )
        return ProviderResponse(text="Clarification required before optimization.")


def test_optimize_tool_requests_clarification_instead_of_guessing() -> None:
    backend_registry = BackendRegistry.default()
    planner = ClarificationPlanner(backend_registry)
    tools = build_default_tool_registry(backend_registry, planner)

    payload = tools.execute(
        "optimize_design",
        {
            "spec": {
                "backend": "pyro",
                "model_ref": "boed_agent.demo.pyro_linear:pyro_linear_model",
            }
        },
    )

    assert payload["status"] == "needs_clarification"
    fields = [item["field"] for item in payload["clarification_questions"]]
    assert "design_variables" in fields
    assert "observation_labels" in fields
    assert "target_latent_labels" in fields


def test_runtime_executes_tool_and_returns_follow_up_text() -> None:
    backend_registry = BackendRegistry.default()
    planner = ClarificationPlanner(backend_registry)
    tools = build_default_tool_registry(backend_registry, planner)
    runtime = AgentRuntime(provider=DummyProvider(), tools=tools)

    text, history = runtime.run_turn("Optimize the experiment.")

    assert text == "Clarification required before optimization."
    assert any(message.role == "tool" for message in history)
    tool_payloads = [json.loads(message.content) for message in history if message.role == "tool"]
    assert tool_payloads[0]["status"] == "needs_clarification"
