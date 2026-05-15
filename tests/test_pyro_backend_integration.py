from __future__ import annotations

import pytest

pyro = pytest.importorskip("pyro")
pytest.importorskip("torch")

from boed_agent.backends.pyro_backend import PyroBackend
from boed_agent.demo.foster_variational import build_foster_experiment_spec
from boed_agent.models import ExperimentSpec


@pytest.mark.parametrize(
    ("estimator", "guide_ref", "extra_budget"),
    [
        ("marginal_eig", "boed_agent.demo.pyro_linear:pyro_linear_marginal_guide", {}),
        ("posterior_eig", "boed_agent.demo.pyro_linear:pyro_linear_posterior_guide", {}),
        (
            "vnmc_eig",
            "boed_agent.demo.pyro_linear:pyro_linear_posterior_guide",
            {"num_inner_samples": 4},
        ),
    ],
)
def test_pyro_backend_estimate_and_optimize_smoke(
    estimator: str,
    guide_ref: str,
    extra_budget: dict[str, int],
) -> None:
    backend = PyroBackend()
    spec = ExperimentSpec.from_dict(
        {
            "backend": "pyro",
            "design_variables": [{"name": "x", "lower": -1.0, "upper": 1.0, "initial": 0.1}],
            "observation_labels": ["y"],
            "target_latent_labels": ["theta"],
            "compute_budget": {
                "num_outer_samples": 4,
                "guide_training_steps": 2,
                "num_optimization_steps": 1,
                **extra_budget,
            },
            "objective": {"estimator": estimator},
            "model_ref": "boed_agent.demo.pyro_linear:pyro_linear_model",
            "guide_ref": guide_ref,
            "optim_ref": "boed_agent.demo.pyro_linear:make_pyro_adam",
            "backend_options": {"optimization_strategy": "random_search"},
        }
    )

    report = backend.validate(spec)
    estimate = backend.estimate_eig(spec)
    result = backend.optimize(spec)

    assert report.valid is True
    assert estimate.status == "ok"
    assert estimate.value is not None
    assert result.status in {"completed", "completed_with_warnings"}
    assert len(result.design) == 1


@pytest.mark.parametrize(
    ("experiment_key", "estimator"),
    [
        ("ab_test_linear", "posterior_eig"),
        ("revealed_preference", "marginal_eig"),
    ],
)
def test_foster_paper_specs_smoke_with_candidate_grid(
    experiment_key: str,
    estimator: str,
) -> None:
    backend = PyroBackend()
    spec = build_foster_experiment_spec(experiment_key, estimator)
    spec.compute_budget.num_outer_samples = 2
    spec.compute_budget.guide_training_steps = 2
    spec.compute_budget.num_optimization_steps = 2
    spec.backend_options["candidate_designs"] = spec.backend_options["candidate_designs"][:2]

    if estimator == "vnmc_eig":
        spec.compute_budget.num_inner_samples = 2

    report = backend.validate(spec)
    estimate = backend.estimate_eig(spec, design=spec.backend_options["candidate_designs"][0])
    result = backend.optimize(spec)

    assert report.valid is True
    assert estimate.value is not None
    assert result.status == "completed"
    assert len(result.history) == 2
    assert result.artifacts["optimization_strategy"] == "candidate_grid"
