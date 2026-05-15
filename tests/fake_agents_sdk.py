from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any


def make_fake_agents_sdk() -> ModuleType:
    module = ModuleType("agents")
    module.tracing_disabled = False

    @dataclass
    class GuardrailFunctionOutput:
        output_info: Any
        tripwire_triggered: bool

    @dataclass
    class ToolGuardrailFunctionOutput:
        output_info: Any = None
        behavior: dict[str, Any] = field(default_factory=lambda: {"type": "allow"})

        @classmethod
        def allow(cls, output_info: Any = None):
            return cls(output_info=output_info, behavior={"type": "allow"})

        @classmethod
        def reject_content(cls, message: str):
            return cls(
                output_info={"message": message},
                behavior={"type": "reject_content", "message": message},
            )

    @dataclass
    class FakeGuardrail:
        guardrail_function: Any
        name: str | None = None

        def get_name(self) -> str:
            return self.name or self.guardrail_function.__name__

    def input_guardrail(func=None, *, name=None, run_in_parallel=True):
        _ = run_in_parallel
        def decorator(f):
            return FakeGuardrail(guardrail_function=f, name=name or f.__name__)
        return decorator(func) if func else decorator

    def output_guardrail(func=None, *, name=None):
        def decorator(f):
            return FakeGuardrail(guardrail_function=f, name=name or f.__name__)
        return decorator(func) if func else decorator

    def tool_input_guardrail(func=None, *, name=None):
        def decorator(f):
            return FakeGuardrail(guardrail_function=f, name=name or f.__name__)
        return decorator(func) if func else decorator

    def tool_output_guardrail(func=None, *, name=None):
        def decorator(f):
            return FakeGuardrail(guardrail_function=f, name=name or f.__name__)
        return decorator(func) if func else decorator

    @dataclass
    class FunctionTool:
        func: Any
        name_override: str | None = None
        description_override: str | None = None
        tool_input_guardrails: list[Any] = field(default_factory=list)
        tool_output_guardrails: list[Any] = field(default_factory=list)
        needs_approval: bool = False
        strict_mode: bool = True

        @property
        def name(self) -> str:
            return self.name_override or self.func.__name__

        def invoke(self, payload: dict[str, Any] | None = None):
            return self.func(**(payload or {}))

    def function_tool(func=None, **kwargs):
        def decorator(f):
            return FunctionTool(func=f, **kwargs)
        return decorator(func) if func else decorator

    @dataclass
    class FakeAgentTool:
        agent: Any
        tool_name: str
        tool_description: str
        custom_output_extractor: Any = None

    class Agent:
        def __init__(self, name: str, instructions: str = "", tools=None, **kwargs):
            self.name = name
            self.instructions = instructions
            self.tools = tools or []
            self.kwargs = kwargs

        def as_tool(
            self,
            tool_name: str | None,
            tool_description: str | None,
            custom_output_extractor=None,
            **kwargs,
        ):
            _ = kwargs
            return FakeAgentTool(
                agent=self,
                tool_name=tool_name or self.name,
                tool_description=tool_description or "",
                custom_output_extractor=custom_output_extractor,
            )

    @dataclass
    class SQLiteSession:
        session_id: str
        db_path: str | None = None

    @dataclass
    class RunConfig:
        workflow_name: str | None = None
        trace_id: str | None = None
        group_id: str | None = None
        trace_metadata: dict[str, Any] | None = None
        tracing_disabled: bool = False

    class InputGuardrailTripwireTriggered(Exception):
        def __init__(self, result):
            super().__init__("input guardrail triggered")
            self.result = result

    class OutputGuardrailTripwireTriggered(Exception):
        def __init__(self, result):
            super().__init__("output guardrail triggered")
            self.result = result

    class Runner:
        next_result = None
        next_exception = None
        last_call = None

        @classmethod
        def run_sync(cls, agent, prompt, **kwargs):
            cls.last_call = {"agent": agent, "prompt": prompt, **kwargs}
            if cls.next_exception is not None:
                exc = cls.next_exception
                cls.next_exception = None
                raise exc
            if callable(cls.next_result):
                return cls.next_result(agent, prompt, kwargs)
            return cls.next_result

    def set_tracing_disabled(flag: bool):
        module.tracing_disabled = flag

    module.Agent = Agent
    module.FunctionTool = FunctionTool
    module.Runner = Runner
    module.RunConfig = RunConfig
    module.SQLiteSession = SQLiteSession
    module.GuardrailFunctionOutput = GuardrailFunctionOutput
    module.ToolGuardrailFunctionOutput = ToolGuardrailFunctionOutput
    module.InputGuardrailTripwireTriggered = InputGuardrailTripwireTriggered
    module.OutputGuardrailTripwireTriggered = OutputGuardrailTripwireTriggered
    module.function_tool = function_tool
    module.input_guardrail = input_guardrail
    module.output_guardrail = output_guardrail
    module.tool_input_guardrail = tool_input_guardrail
    module.tool_output_guardrail = tool_output_guardrail
    module.set_tracing_disabled = set_tracing_disabled
    module.SimpleNamespace = SimpleNamespace
    return module
