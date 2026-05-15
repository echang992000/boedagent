"""Agent engine abstractions for manual and SDK-backed chat runtimes."""

from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from boed_agent.agents.manager import build_boed_manager_agent
from boed_agent.models import (
    AgentTurnResult,
    GuardrailEvent,
    Message,
    SessionConfig,
    ToolEvent,
    TraceMetadata,
)
from boed_agent.providers.base import LLMProvider
from boed_agent.tools.registry import ToolRegistry


DEFAULT_SYSTEM_PROMPT = """
You are a Bayesian optimal experimental design (BOED) agent.

Use the provided tools instead of guessing backend capabilities.
Before any BOED execution, validate the experiment spec and ask for clarification if required fields are missing.
Never invent observation labels, latent targets, design bounds, estimator choices, or compute budgets.
Treat literature preferences as part of the normalized BOED spec when the user asks for literature-informed priors.
If literature is enabled, run the literature dry-run tool before suggesting execution.
Literature outputs are advisory only in v1 and are not auto-applied to backend execution.
""".strip()


class AgentEngine(ABC):
    @abstractmethod
    def run_turn(
        self,
        prompt: str,
        history_or_session: Any = None,
        context: dict[str, Any] | None = None,
        max_loops: int = 8,
    ) -> AgentTurnResult:
        """Execute one chat turn."""


class ManualAgentEngine(AgentEngine):
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt

    def run_turn(
        self,
        prompt: str,
        history_or_session: list[Message] | None = None,
        context: dict[str, Any] | None = None,
        max_loops: int = 8,
    ) -> AgentTurnResult:
        _ = context
        messages = list(history_or_session or [])
        messages.append(Message(role="user", content=prompt))
        loops = 0
        final_text = ""
        tool_events: list[ToolEvent] = []

        while loops < max_loops:
            loops += 1
            request = self.provider.build_request(
                messages,
                self._provider_tools(),
                self.system_prompt,
                state=None,
            )
            response = self.provider.generate(request)
            parsed = self.provider.parse_response(response)

            if parsed.text:
                final_text = parsed.text
                messages.append(Message(role="assistant", content=parsed.text))

            if not parsed.tool_calls:
                break

            for call in parsed.tool_calls:
                result = self.tools.execute(call.name, call.arguments)
                tool_events.append(
                    ToolEvent(
                        name=call.name,
                        arguments=call.arguments,
                        output=result,
                        call_id=call.id,
                    )
                )
                messages.append(
                    Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=json.dumps(result),
                    )
                )

            follow_up = self.provider.continue_with_tool_results(
                messages,
                self._provider_tools(),
                self.system_prompt,
                state=None,
            )
            response = self.provider.generate(follow_up)
            parsed = self.provider.parse_response(response)
            if parsed.text:
                final_text = parsed.text
                messages.append(Message(role="assistant", content=parsed.text))
            if not parsed.tool_calls:
                break

        return AgentTurnResult(
            text=final_text,
            history=messages,
            tool_events=tool_events,
            session_metadata=SessionConfig(
                session_id="manual",
                db_path="",
                tracing_enabled=False,
                runtime_mode="manual",
                resumed=bool(history_or_session),
            ),
        )

    def _provider_tools(self) -> list[dict[str, Any]]:
        if self.provider.name == "openai":
            return self.tools.as_openai_tools()
        if self.provider.name == "claude":
            return self.tools.as_claude_tools()
        raise ValueError(f"Unsupported provider '{self.provider.name}'.")


class OpenAIAgentsSdkEngine(AgentEngine):
    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        session_config: SessionConfig,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.session_config = session_config
        self.system_prompt = system_prompt
        self.api_key = api_key
        self._manager_agent: Any | None = None

    def run_turn(
        self,
        prompt: str,
        history_or_session: Any = None,
        context: dict[str, Any] | None = None,
        max_loops: int = 8,
    ) -> AgentTurnResult:
        _ = history_or_session
        sdk = self._import_sdk()
        if self.api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)

        sdk.set_tracing_disabled(not self.session_config.tracing_enabled)
        db_path = self.session_config.db_path
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        session = sdk.SQLiteSession(self.session_config.session_id, db_path)
        trace_id = uuid.uuid4().hex
        run_config = sdk.RunConfig(
            workflow_name="boed-agent-chat",
            trace_id=trace_id,
            group_id=self.session_config.session_id,
            trace_metadata={
                "runtime_mode": "agents-sdk",
                "session_id": self.session_config.session_id,
            },
            tracing_disabled=not self.session_config.tracing_enabled,
        )

        try:
            result = sdk.Runner.run_sync(
                self._get_manager_agent(),
                prompt,
                context=context or {},
                max_turns=max_loops,
                run_config=run_config,
                session=session,
            )
        except sdk.InputGuardrailTripwireTriggered as exc:
            guardrail_event = self._guardrail_event_from_tripwire(exc, stage="input")
            return AgentTurnResult(
                text=self._guardrail_message(guardrail_event),
                guardrail_events=[guardrail_event],
                session_metadata=self.session_config,
                trace_metadata=TraceMetadata(
                    trace_id=trace_id,
                    group_id=self.session_config.session_id,
                    workflow_name="boed-agent-chat",
                    tracing_enabled=self.session_config.tracing_enabled,
                ),
                raw=getattr(exc, "result", None),
            )
        except sdk.OutputGuardrailTripwireTriggered as exc:
            guardrail_event = self._guardrail_event_from_tripwire(exc, stage="output")
            return AgentTurnResult(
                text=self._guardrail_message(guardrail_event),
                guardrail_events=[guardrail_event],
                session_metadata=self.session_config,
                trace_metadata=TraceMetadata(
                    trace_id=trace_id,
                    group_id=self.session_config.session_id,
                    workflow_name="boed-agent-chat",
                    tracing_enabled=self.session_config.tracing_enabled,
                ),
                raw=getattr(exc, "result", None),
            )

        final_output = getattr(result, "final_output", "")
        text = getattr(final_output, "message", None) or str(final_output)
        guardrail_events = self._extract_guardrail_events(result)
        tool_events = self._extract_tool_events(result)
        trace_metadata = TraceMetadata(
            trace_id=trace_id,
            group_id=self.session_config.session_id,
            workflow_name="boed-agent-chat",
            tracing_enabled=self.session_config.tracing_enabled,
        )
        return AgentTurnResult(
            text=text,
            tool_events=tool_events,
            guardrail_events=guardrail_events,
            session_metadata=self.session_config,
            trace_metadata=trace_metadata,
            raw=final_output,
        )

    def _get_manager_agent(self) -> Any:
        if self._manager_agent is None:
            self._manager_agent = build_boed_manager_agent(
                model=self.model,
                tools=self.tools,
                system_prompt=self.system_prompt,
            )
        return self._manager_agent

    def _import_sdk(self) -> Any:
        try:
            import agents as sdk
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "OpenAI Agents SDK support requires the optional `openai-agents` dependency. "
                "Install with `pip install -e \".[agents]\"`."
            ) from exc
        return sdk

    def _guardrail_event_from_tripwire(self, exc: Exception, stage: str) -> GuardrailEvent:
        result = getattr(exc, "result", None)
        guardrail = getattr(result, "guardrail", None)
        output = getattr(result, "output", None)
        details = getattr(output, "output_info", None) or {}
        if not isinstance(details, dict):
            details = {"info": details}
        return GuardrailEvent(
            name=getattr(guardrail, "name", None) or getattr(guardrail, "get_name", lambda: "guardrail")(),
            stage=stage,
            decision="blocked",
            details=details,
        )

    def _guardrail_message(self, event: GuardrailEvent) -> str:
        reason = event.details.get("reason")
        if reason == "prompt_injection":
            return "I can’t help with prompt extraction or instruction-bypass requests. Ask a BOED-specific question instead."
        if reason == "off_topic":
            return "This agent is scoped to BOED workflow setup and execution. Ask about experiment specs, backend selection, validation, EIG estimation, or optimization."
        if "message" in event.details:
            return str(event.details["message"])
        return "The request was blocked by an agent guardrail."

    def _extract_guardrail_events(self, result: Any) -> list[GuardrailEvent]:
        events: list[GuardrailEvent] = []
        guardrail_groups = [
            ("input", getattr(result, "input_guardrail_results", [])),
            ("output", getattr(result, "output_guardrail_results", [])),
            ("tool_input", getattr(result, "tool_input_guardrail_results", [])),
            ("tool_output", getattr(result, "tool_output_guardrail_results", [])),
        ]
        for stage, results in guardrail_groups:
            for item in results or []:
                guardrail = getattr(item, "guardrail", None)
                output = getattr(item, "output", None)
                details = getattr(output, "output_info", None) or {}
                if not isinstance(details, dict):
                    details = {"info": details}
                decision = "allow"
                behavior = getattr(output, "behavior", None)
                if isinstance(behavior, dict):
                    decision = behavior.get("type", decision)
                elif getattr(output, "tripwire_triggered", False):
                    decision = "blocked"
                events.append(
                    GuardrailEvent(
                        name=getattr(guardrail, "name", None) or getattr(guardrail, "get_name", lambda: "guardrail")(),
                        stage=stage,
                        decision=decision,
                        details=details,
                    )
                )
        return events

    def _extract_tool_events(self, result: Any) -> list[ToolEvent]:
        events: list[ToolEvent] = []
        for item in getattr(result, "new_items", []) or []:
            raw_item = getattr(item, "raw_item", None)
            source = raw_item or item
            event_type = getattr(source, "type", None) or getattr(item, "type", None)
            if event_type not in {"function_call", "tool_call", "function_call_output", "tool_result"}:
                continue
            events.append(
                ToolEvent(
                    name=getattr(source, "name", None) or getattr(item, "name", None) or "tool",
                    arguments=self._coerce_jsonish(
                        getattr(source, "arguments", None) or getattr(item, "arguments", None) or {}
                    ),
                    output=getattr(source, "output", None) or getattr(item, "output", None),
                    status="completed",
                    call_id=getattr(source, "call_id", None) or getattr(item, "call_id", None),
                )
            )
        return events

    def _coerce_jsonish(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return {"raw": payload}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return {}
