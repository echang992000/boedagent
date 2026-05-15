"""Shared data models used across the BOED agent runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


JSONValue = Any


@dataclass
class DesignVariable:
    name: str
    lower: float
    upper: float
    initial: float | None = None
    dtype: str = "float"
    description: str | None = None
    shape: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignVariable":
        return cls(
            name=str(data["name"]),
            lower=float(data["lower"]),
            upper=float(data["upper"]),
            initial=None if data.get("initial") is None else float(data["initial"]),
            dtype=str(data.get("dtype", "float")),
            description=data.get("description"),
            shape=int(data.get("shape", 1)),
        )


@dataclass
class ComputeBudget:
    num_outer_samples: int | None = None
    num_inner_samples: int | None = None
    num_optimization_steps: int | None = None
    guide_training_steps: int | None = None
    max_runtime_seconds: int | None = None
    design_learning_rate: float | None = None
    flow_learning_rate: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ComputeBudget":
        data = data or {}
        return cls(
            num_outer_samples=_optional_int(data.get("num_outer_samples")),
            num_inner_samples=_optional_int(data.get("num_inner_samples")),
            num_optimization_steps=_optional_int(data.get("num_optimization_steps")),
            guide_training_steps=_optional_int(data.get("guide_training_steps")),
            max_runtime_seconds=_optional_int(data.get("max_runtime_seconds")),
            design_learning_rate=_optional_float(data.get("design_learning_rate")),
            flow_learning_rate=_optional_float(data.get("flow_learning_rate")),
        )


@dataclass
class ObjectiveSpec:
    name: str = "expected_information_gain"
    estimator: str | None = None
    mode: str | None = "variational"
    maximize: bool = True
    estimator_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ObjectiveSpec":
        data = data or {}
        return cls(
            name=str(data.get("name", "expected_information_gain")),
            estimator=data.get("estimator"),
            mode=data.get("mode", "variational"),
            maximize=bool(data.get("maximize", True)),
            estimator_kwargs=dict(data.get("estimator_kwargs", {})),
        )


@dataclass
class ArtifactSettings:
    output_dir: str = "artifacts"
    save_normalized_spec: bool = True
    save_transcript: bool = True
    save_backend_summary: bool = True
    save_result_payload: bool = True
    save_trajectory_plot: bool = False
    trajectory_plot_filename: str = "design_trajectory.png"
    trajectory_summary_filename: str = "trajectory_summary.json"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ArtifactSettings":
        data = data or {}
        return cls(
            output_dir=str(data.get("output_dir", "artifacts")),
            save_normalized_spec=bool(data.get("save_normalized_spec", True)),
            save_transcript=bool(data.get("save_transcript", True)),
            save_backend_summary=bool(data.get("save_backend_summary", True)),
            save_result_payload=bool(data.get("save_result_payload", True)),
            save_trajectory_plot=bool(data.get("save_trajectory_plot", False)),
            trajectory_plot_filename=str(
                data.get("trajectory_plot_filename", "design_trajectory.png")
            ),
            trajectory_summary_filename=str(
                data.get("trajectory_summary_filename", "trajectory_summary.json")
            ),
        )


@dataclass
class ExperimentSpec:
    problem_summary: str | None = None
    backend: str | None = None
    provider: str | None = None
    use_literature: bool | None = None
    literature_source_mode: str | None = None
    literature_corpus_dir: str | None = None
    recreate_trajectory: bool = False
    design_variables: list[DesignVariable] = field(default_factory=list)
    observation_labels: list[str] = field(default_factory=list)
    target_latent_labels: list[str] = field(default_factory=list)
    compute_budget: ComputeBudget = field(default_factory=ComputeBudget)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    artifacts: ArtifactSettings = field(default_factory=ArtifactSettings)
    backend_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    model_ref: str | None = None
    guide_ref: str | None = None
    loss_ref: str | None = None
    optim_ref: str | None = None
    simulator_ref: str | None = None
    prior_sampler_ref: str | None = None
    latent_sampler_ref: str | None = None
    differentiable: bool | None = None
    surrogate: dict[str, Any] = field(default_factory=dict)
    initial_design: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            problem_summary=data.get("problem_summary"),
            backend=data.get("backend"),
            provider=data.get("provider"),
            use_literature=(
                None
                if "use_literature" not in data
                else (
                    None if data.get("use_literature") is None else bool(data.get("use_literature"))
                )
            ),
            literature_source_mode=(
                None
                if data.get("literature_source_mode") is None
                else str(data.get("literature_source_mode"))
            ),
            literature_corpus_dir=(
                None
                if data.get("literature_corpus_dir") is None
                else str(data.get("literature_corpus_dir"))
            ),
            recreate_trajectory=bool(
                data.get(
                    "recreate_trajectory",
                    data.get("metadata", {}).get("recreate_trajectory", False),
                )
            ),
            design_variables=[
                DesignVariable.from_dict(item) for item in data.get("design_variables", [])
            ],
            observation_labels=[str(item) for item in data.get("observation_labels", [])],
            target_latent_labels=[
                str(item) for item in data.get("target_latent_labels", [])
            ],
            compute_budget=ComputeBudget.from_dict(data.get("compute_budget")),
            objective=ObjectiveSpec.from_dict(data.get("objective")),
            artifacts=ArtifactSettings.from_dict(data.get("artifacts")),
            backend_options=dict(data.get("backend_options", {})),
            metadata=dict(data.get("metadata", {})),
            model_ref=data.get("model_ref"),
            guide_ref=data.get("guide_ref"),
            loss_ref=data.get("loss_ref"),
            optim_ref=data.get("optim_ref"),
            simulator_ref=data.get("simulator_ref"),
            prior_sampler_ref=data.get("prior_sampler_ref"),
            latent_sampler_ref=data.get("latent_sampler_ref"),
            differentiable=data.get("differentiable"),
            surrogate=dict(data.get("surrogate", {})),
            initial_design=[
                float(value) for value in data.get("initial_design", []) if value is not None
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective_initial_design(self) -> list[float]:
        if self.initial_design:
            return list(self.initial_design)
        values: list[float] = []
        for variable in self.design_variables:
            if variable.initial is not None:
                values.append(variable.initial)
            else:
                values.append((variable.lower + variable.upper) / 2.0)
        return values

    def wants_recreated_trajectory(self) -> bool:
        return bool(self.recreate_trajectory)

    def wants_literature(self) -> bool:
        return bool(self.use_literature)


@dataclass
class ValidationIssue:
    path: str
    message: str
    severity: str = "error"


@dataclass
class ValidationReport:
    valid: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClarificationQuestion:
    field: str
    prompt: str
    reason: str
    backend: str | None = None
    choices: list[str] = field(default_factory=list)
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackendDescriptor:
    name: str
    description: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    status: str = "available"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EIGEstimate:
    backend: str
    estimator: str | None
    design: list[float]
    value: float | None
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationStep:
    step: int
    design: list[float]
    eig: float | None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    backend: str
    estimator: str | None
    status: str
    design: list[float]
    eig: float | None
    history: list[OptimizationStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Message:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    raw: Any = None


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    status: str = "completed"
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuardrailEvent:
    name: str
    stage: str
    decision: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionConfig:
    session_id: str
    db_path: str = "artifacts/agent_sessions.sqlite"
    tracing_enabled: bool = True
    runtime_mode: str = "manual"
    resumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceMetadata:
    trace_id: str | None = None
    group_id: str | None = None
    workflow_name: str | None = None
    tracing_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTurnResult:
    text: str
    history: list[Message] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    guardrail_events: list[GuardrailEvent] = field(default_factory=list)
    trace_metadata: TraceMetadata | None = None
    session_metadata: SessionConfig | None = None
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "history": [to_jsonable(item) for item in self.history],
            "tool_events": [item.to_dict() for item in self.tool_events],
            "guardrail_events": [item.to_dict() for item in self.guardrail_events],
            "trace_metadata": None if self.trace_metadata is None else self.trace_metadata.to_dict(),
            "session_metadata": None if self.session_metadata is None else self.session_metadata.to_dict(),
            "raw": to_jsonable(self.raw),
        }


def get_field_value(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if current is None:
            return None
        if is_dataclass(current):
            current = getattr(current, part, None)
            continue
        if isinstance(current, dict):
            current = current.get(part)
            continue
        return None
    return current


def field_is_missing(obj: Any, path: str) -> bool:
    value = get_field_value(obj, path)
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
