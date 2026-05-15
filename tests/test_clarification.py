from __future__ import annotations

from boed_agent.backends.registry import BackendRegistry
from boed_agent.clarification.planner import ClarificationPlanner
from boed_agent.models import ExperimentSpec


def test_clarification_orders_core_pyro_questions() -> None:
    planner = ClarificationPlanner(BackendRegistry.default())
    spec = ExperimentSpec.from_dict({"backend": "pyro", "model_ref": "demo.module:model"})

    questions = planner.plan(spec)
    fields = [question.field for question in questions]

    assert fields[:4] == [
        "design_variables",
        "observation_labels",
        "target_latent_labels",
        "objective.estimator",
    ]


def test_clarification_handles_simulator_backend() -> None:
    planner = ClarificationPlanner(BackendRegistry.default())
    spec = ExperimentSpec.from_dict(
        {
            "backend": "lfiax",
            "simulator_ref": "demo.sim:simulate",
            "objective": {"estimator": "lf_pce_eig"},
        }
    )

    questions = planner.plan(spec)
    fields = [question.field for question in questions]

    assert "design_variables" in fields
    assert "backend_options.design_mode" in fields
    assert "differentiable" in fields
    assert "prior_sampler_ref" in fields


def test_clarification_without_literature_opt_in_skips_literature_questions() -> None:
    planner = ClarificationPlanner(BackendRegistry.default())
    spec = ExperimentSpec.from_dict({"backend": "pyro", "model_ref": "demo.module:model"})

    questions = planner.plan(spec)
    fields = [question.field for question in questions]

    assert "literature_source_mode" not in fields
    assert "literature_corpus_dir" not in fields


def test_clarification_literature_opt_in_requires_source_mode() -> None:
    planner = ClarificationPlanner(BackendRegistry.default())
    spec = ExperimentSpec.from_dict(
        {
            "backend": "pyro",
            "model_ref": "demo.module:model",
            "use_literature": True,
        }
    )

    questions = planner.plan(spec)
    fields = [question.field for question in questions]

    assert "literature_source_mode" in fields


def test_clarification_local_literature_requires_corpus_dir() -> None:
    planner = ClarificationPlanner(BackendRegistry.default())
    spec = ExperimentSpec.from_dict(
        {
            "backend": "pyro",
            "model_ref": "demo.module:model",
            "use_literature": True,
            "literature_source_mode": "both",
        }
    )

    questions = planner.plan(spec)
    fields = [question.field for question in questions]

    assert "literature_corpus_dir" in fields
