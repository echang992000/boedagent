"""BOED agent package."""

from boed_agent.agent import (
    AgentRunResult,
    BOEDAgent,
    DesignBuilder,
    DesignSpace,
    DryRunResult,
    PostSynthesisValidationError,
)
from boed_agent.backends.registry import BackendRegistry
from boed_agent.classifier import DataClassifier, ClassifierResult
from boed_agent.clarification.planner import ClarificationPlanner
from boed_agent.literature import (
    LiteratureReport,
    LiteratureSearchModule,
    ReasoningStep,
    ReasoningTrace,
    TokenBudget,
)
from boed_agent.prior_builder import AugmentedPrior, DistributionSpec, PriorBuilder
from boed_agent.simulator_choice import BackendChoice, SimulatorChoiceModule
from boed_agent.simulator_protocol import (
    ParameterInfo,
    SimpleSimulator,
    Simulator,
    SimulatorMetadata,
)
from boed_agent.tools.registry import build_default_tool_registry

__all__ = [
    "AgentRunResult",
    "AugmentedPrior",
    "BOEDAgent",
    "BackendChoice",
    "BackendRegistry",
    "ClarificationPlanner",
    "ClassifierResult",
    "DataClassifier",
    "DesignBuilder",
    "DesignSpace",
    "DistributionSpec",
    "DryRunResult",
    "LiteratureReport",
    "LiteratureSearchModule",
    "ParameterInfo",
    "PostSynthesisValidationError",
    "PriorBuilder",
    "ReasoningStep",
    "ReasoningTrace",
    "SimpleSimulator",
    "Simulator",
    "SimulatorChoiceModule",
    "SimulatorMetadata",
    "TokenBudget",
    "build_default_tool_registry",
]
