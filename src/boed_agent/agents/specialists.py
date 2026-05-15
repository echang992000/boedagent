"""Specialist agents used by the OpenAI Agents SDK manager agent."""

from __future__ import annotations

from boed_agent.agents.guardrails import extract_agent_tool_output
from boed_agent.agents.schemas import (
    BackendAdvisorOutput,
    ClarificationSpecialistOutput,
    ManagerResponse,
    ResultInterpreterOutput,
)
from boed_agent.agents.sdk_imports import import_agents_sdk
from boed_agent.tools.registry import ToolRegistry


def build_clarification_specialist(model: str, tools: ToolRegistry):
    sdk = import_agents_sdk()
    specialist_tools = tools.as_openai_sdk_tools(selected_names=["validate_experiment_spec", "clarify_missing_fields"])
    return sdk.Agent(
        name="Clarification Specialist",
        model=model,
        instructions=(
            "You convert incomplete BOED specs into an ordered set of blocking questions. "
            "Always validate first, then return blocking_fields and explicit user-facing questions."
        ),
        tools=specialist_tools,
        output_type=ClarificationSpecialistOutput,
    )


def build_backend_advisor_specialist(model: str, tools: ToolRegistry):
    sdk = import_agents_sdk()
    specialist_tools = tools.as_openai_sdk_tools(
        selected_names=["list_backends", "describe_backend", "validate_experiment_spec"]
    )
    return sdk.Agent(
        name="Backend Advisor Specialist",
        model=model,
        instructions=(
            "You recommend the most appropriate BOED backend. "
            "Explain whether Pyro or LFIAX fits the user request and list the next required inputs."
        ),
        tools=specialist_tools,
        output_type=BackendAdvisorOutput,
    )


def build_result_interpreter_specialist(model: str, tools: ToolRegistry):
    sdk = import_agents_sdk()
    specialist_tools = tools.as_openai_sdk_tools(selected_names=["summarize_result"])
    return sdk.Agent(
        name="Result Interpreter Specialist",
        model=model,
        instructions=(
            "You convert BOED tool outputs into a concise summary, caveats, and next action. "
            "If the result includes optimized_design_histories because trajectory recreation was requested, preserve them in the output."
        ),
        tools=specialist_tools,
        output_type=ResultInterpreterOutput,
    )


def specialist_tools_for_manager(model: str, tools: ToolRegistry):
    clarification = build_clarification_specialist(model, tools)
    backend_advisor = build_backend_advisor_specialist(model, tools)
    result_interpreter = build_result_interpreter_specialist(model, tools)
    return [
        clarification.as_tool(
            tool_name="clarification_specialist",
            tool_description="Turn missing-field BOED validation results into explicit user questions.",
            custom_output_extractor=extract_agent_tool_output,
        ),
        backend_advisor.as_tool(
            tool_name="backend_advisor_specialist",
            tool_description="Recommend the appropriate BOED backend and required next inputs.",
            custom_output_extractor=extract_agent_tool_output,
        ),
        result_interpreter.as_tool(
            tool_name="result_interpreter_specialist",
            tool_description="Summarize BOED execution results, caveats, and next actions.",
            custom_output_extractor=extract_agent_tool_output,
        ),
    ]
