"""Provider-neutral tool registry for the BOED agent."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from boed_agent.agents.sdk_imports import import_agents_sdk

from boed_agent.backends.registry import BackendRegistry
from boed_agent.clarification.planner import ClarificationPlanner
from boed_agent.literature.advisory import (
    prepare_literature_spec,
    run_literature_dry_run,
    validate_literature_spec,
)
from boed_agent.models import ExperimentSpec, field_is_missing, to_jsonable


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    risk_level: str = "low"
    sdk_enabled: bool = True

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }

    def to_claude_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai_sdk_tool(
        self,
        tool_input_guardrails: list[Any] | None = None,
        tool_output_guardrails: list[Any] | None = None,
    ) -> Any:
        sdk = import_agents_sdk()
        wrapped = self._build_sdk_callable()
        kwargs: dict[str, Any] = {
            "name_override": self.name,
            "description_override": self.description,
            "strict_mode": False,
        }
        if tool_input_guardrails:
            kwargs["tool_input_guardrails"] = tool_input_guardrails
        if tool_output_guardrails:
            kwargs["tool_output_guardrails"] = tool_output_guardrails
        if self.risk_level == "high":
            kwargs["needs_approval"] = True
        return sdk.function_tool(wrapped, **kwargs)

    def _build_sdk_callable(self) -> Callable[..., str]:
        properties = OrderedDict(self.input_schema.get("properties", {}))
        required = set(self.input_schema.get("required", []))
        params: list[str] = []
        assignments: list[str] = []
        for name, schema in properties.items():
            annotation = _python_type_annotation(schema)
            if name in required:
                params.append(f"{name}: {annotation}")
            else:
                params.append(f"{name}: {annotation} = None")
            assignments.append(
                "    if {name} is not None:\n        args['{name}'] = {name}".format(name=name)
            )
        signature = ", ".join(params)
        function_name = self.name.replace("-", "_")
        source = [
            f"def {function_name}({signature}) -> str:",
            f'    """{self.description}"""',
            "    args = {}",
        ]
        source.extend(assignments)
        source.extend(
            [
                "    result = _handler(args)",
                "    return json.dumps(_to_jsonable(result))",
            ]
        )
        namespace = {"_handler": self.handler, "_to_jsonable": to_jsonable, "json": json, "Any": Any}
        exec("\n".join(source), namespace)
        func = namespace[function_name]
        func.__name__ = function_name
        return func


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'.")
        return self._tools[name].handler(arguments)

    def as_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def as_claude_tools(self) -> list[dict[str, Any]]:
        return [tool.to_claude_schema() for tool in self._tools.values()]

    def as_openai_sdk_tools(
        self,
        selected_names: list[str] | None = None,
        tool_guardrail_builder: Callable[[ToolDefinition], dict[str, list[Any]]] | None = None,
    ) -> list[Any]:
        names = set(selected_names or self._tools.keys())
        sdk_tools: list[Any] = []
        for tool in self._tools.values():
            if tool.name not in names or not tool.sdk_enabled:
                continue
            guardrail_kwargs = tool_guardrail_builder(tool) if tool_guardrail_builder else {}
            sdk_tools.append(tool.to_openai_sdk_tool(**guardrail_kwargs))
        return sdk_tools

    def list(self) -> list[str]:
        return sorted(self._tools)


def build_default_tool_registry(
    backend_registry: BackendRegistry,
    planner: ClarificationPlanner,
    *,
    literature_provider_name: str | None = None,
    literature_model: str | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    def _load_spec(args: dict[str, Any]) -> ExperimentSpec:
        payload = args.get("spec")
        if not isinstance(payload, dict):
            raise ValueError("Tool expects `spec` to be an object.")
        return ExperimentSpec.from_dict(payload)

    def _validate(args: dict[str, Any]) -> dict[str, Any]:
        spec = _load_spec(args)
        try:
            backend = backend_registry.select_backend(spec)
            report = backend.validate(spec)
        except KeyError:
            missing = ["backend"] if field_is_missing(spec, "backend") else []
            report = {
                "valid": False,
                "backend": None,
                "errors": [{"path": "backend", "message": "Unable to infer backend."}],
                "warnings": [],
                "missing_fields": missing,
            }
            return {
                "validation": report,
                "clarification_questions": [question.to_dict() for question in planner.plan(spec)],
            }
        return {
            "validation": report.to_dict(),
            "clarification_questions": [question.to_dict() for question in planner.plan(spec)],
        }

    def _clarify(args: dict[str, Any]) -> dict[str, Any]:
        spec = _load_spec(args)
        return {
            "questions": [question.to_dict() for question in planner.plan(spec)],
        }

    def _estimate(args: dict[str, Any]) -> dict[str, Any]:
        spec = _load_spec(args)
        backend = backend_registry.select_backend(spec)
        report = backend.validate(spec)
        if not report.valid:
            return {
                "status": "needs_clarification",
                "validation": report.to_dict(),
                "clarification_questions": [question.to_dict() for question in planner.plan(spec)],
            }
        result = backend.estimate_eig(spec, design=args.get("design"))
        return result.to_dict()

    def _optimize(args: dict[str, Any]) -> dict[str, Any]:
        spec = _load_spec(args)
        backend = backend_registry.select_backend(spec)
        report = backend.validate(spec)
        if not report.valid:
            return {
                "status": "needs_clarification",
                "validation": report.to_dict(),
                "clarification_questions": [question.to_dict() for question in planner.plan(spec)],
            }
        result = backend.optimize(spec)
        return result.to_dict()

    def _literature_dry_run(args: dict[str, Any]) -> dict[str, Any]:
        spec = prepare_literature_spec(_load_spec(args))
        validation = validate_literature_spec(spec)
        if not validation.valid:
            return {
                "status": "needs_clarification",
                "validation": validation.to_dict(),
                "clarification_questions": [question.to_dict() for question in planner.plan(spec)],
            }
        result, warnings = run_literature_dry_run(
            spec,
            provider_name=literature_provider_name,
            model=literature_model,
        )
        payload = result.to_dict()
        payload["advisory_only"] = True
        if warnings:
            payload["warnings"] = list(warnings)
        return payload

    def _describe_backend(args: dict[str, Any]) -> dict[str, Any]:
        backend_name = args.get("backend_name")
        if not backend_name:
            raise ValueError("`backend_name` is required.")
        return backend_registry.get(str(backend_name)).describe().to_dict()

    def _list_backends(_: dict[str, Any]) -> dict[str, Any]:
        return {"backends": [descriptor.to_dict() for descriptor in backend_registry.list_backends()]}

    def _summarize(args: dict[str, Any]) -> dict[str, Any]:
        result = args.get("result", {})
        if not isinstance(result, dict):
            raise ValueError("`result` must be an object.")
        summary = summarize_result_payload(result)
        payload = {"summary": summary}
        histories = _extract_optimized_design_histories(result)
        if histories:
            payload["optimized_design_histories"] = histories
        history_summaries = _extract_optimized_design_history_summaries(result)
        if history_summaries:
            payload["optimized_design_history_summaries"] = history_summaries
        return payload

    registry.register(
        ToolDefinition(
            name="list_backends",
            description="List BOED backends and their capabilities.",
            input_schema={"type": "object", "properties": {}},
            handler=_list_backends,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="describe_backend",
            description="Describe a BOED backend and its required fields.",
            input_schema={
                "type": "object",
                "properties": {"backend_name": {"type": "string"}},
                "required": ["backend_name"],
            },
            handler=_describe_backend,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="validate_experiment_spec",
            description="Validate an experiment spec against the selected or inferred backend.",
            input_schema=_spec_schema(),
            handler=_validate,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="clarify_missing_fields",
            description="List clarification questions for an incomplete experiment spec.",
            input_schema=_spec_schema(),
            handler=_clarify,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="estimate_eig",
            description="Estimate the expected information gain for a BOED experiment spec.",
            input_schema={
                "type": "object",
                "properties": {
                    "spec": {"type": "object"},
                    "design": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "required": ["spec"],
            },
            handler=_estimate,
            risk_level="medium",
        )
    )
    registry.register(
        ToolDefinition(
            name="optimize_design",
            description="Optimize the design for a BOED experiment spec.",
            input_schema=_spec_schema(),
            handler=_optimize,
            risk_level="medium",
        )
    )
    registry.register(
        ToolDefinition(
            name="run_literature_dry_run",
            description=(
                "Run an advisory-only literature dry-run for a BOED experiment spec. "
                "Returns literature-derived prior suggestions, backend hints, and reasoning trace "
                "without executing a backend or auto-applying priors."
            ),
            input_schema=_spec_schema(),
            handler=_literature_dry_run,
            risk_level="medium",
        )
    )
    registry.register(
        ToolDefinition(
            name="summarize_result",
            description="Summarize a validation or BOED execution result.",
            input_schema={
                "type": "object",
                "properties": {"result": {"type": "object"}},
                "required": ["result"],
            },
            handler=_summarize,
            risk_level="low",
        )
    )
    return registry


def summarize_result_payload(result: dict[str, Any]) -> str:
    status = result.get("status")
    if status == "needs_clarification":
        questions = result.get("clarification_questions", [])
        return f"Execution blocked pending clarification for {len(questions)} fields."
    if result.get("advisory_only"):
        backend = result.get("chosen_backend")
        has_report = bool(result.get("literature_report"))
        return (
            f"Literature advisory completed with suggested backend `{backend}`. "
            f"Literature report {'available' if has_report else 'not available'}."
        )
    if "validation" in result:
        validation = result["validation"]
        if validation.get("valid"):
            return f"Spec is valid for backend `{validation.get('backend')}`."
        return (
            f"Spec is invalid for backend `{validation.get('backend')}` with "
            f"{len(validation.get('errors', []))} error(s)."
        )
    if "eig" in result:
        summary = (
            f"Backend `{result.get('backend')}` returned status `{status}` with "
            f"EIG={result.get('eig') or result.get('value')}."
        )
        histories = _extract_optimized_design_histories(result)
        compressed = _extract_optimized_design_history_summaries(result)
        plot_path = _extract_design_trajectory_plot(result)
        if histories:
            parts = [
                f"{summary} Trajectory recreation was requested; "
                f"{len(histories)} optimized design history(s) are available with "
                f"{len(histories[0])} step(s) in the first trajectory."
            ]
            if compressed:
                parts.append(
                    f"Compressed trajectory summaries are available for {len(compressed)} history(s)."
                )
            if plot_path:
                parts.append(f"Trajectory plot saved to `{plot_path}`.")
            return " ".join(parts)
        return summary
    return json.dumps(to_jsonable(result), indent=2)


def _spec_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"spec": {"type": "object"}},
        "required": ["spec"],
    }


def _python_type_annotation(schema: dict[str, Any]) -> str:
    json_type = schema.get("type")
    if json_type == "string":
        return "str"
    if json_type == "integer":
        return "int"
    if json_type == "number":
        return "float"
    if json_type == "boolean":
        return "bool"
    if json_type == "array":
        return "list"
    if json_type == "object":
        return "dict"
    return "Any"


def _extract_optimized_design_histories(result: dict[str, Any]) -> list[Any]:
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return []
    histories = artifacts.get("optimized_design_histories", [])
    return histories if isinstance(histories, list) else []


def _extract_optimized_design_history_summaries(result: dict[str, Any]) -> list[Any]:
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return []
    summaries = artifacts.get("optimized_design_history_summaries", [])
    return summaries if isinstance(summaries, list) else []


def _extract_design_trajectory_plot(result: dict[str, Any]) -> str | None:
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    plot_path = artifacts.get("design_trajectory_plot")
    return plot_path if isinstance(plot_path, str) and plot_path else None
