from __future__ import annotations

from boed_agent.backends.pyro_backend import PyroBackend
from boed_agent.demo.foster_variational import build_foster_experiment_spec
from boed_agent.models import ExperimentSpec


def test_experiment_spec_round_trip_and_initial_design() -> None:
    spec = ExperimentSpec.from_dict(
        {
            "backend": "pyro",
            "use_literature": True,
            "literature_source_mode": "local",
            "literature_corpus_dir": "/tmp/demo-corpus",
            "recreate_trajectory": True,
            "compute_budget": {
                "flow_learning_rate": 0.001,
            },
            "design_variables": [
                {"name": "x", "lower": -1.0, "upper": 3.0},
                {"name": "y", "lower": 2.0, "upper": 6.0, "initial": 5.0},
            ],
        }
    )

    assert spec.effective_initial_design() == [1.0, 5.0]
    assert spec.to_dict()["backend"] == "pyro"
    assert spec.to_dict()["use_literature"] is True
    assert spec.to_dict()["literature_source_mode"] == "local"
    assert spec.to_dict()["literature_corpus_dir"] == "/tmp/demo-corpus"
    assert spec.to_dict()["compute_budget"]["flow_learning_rate"] == 0.001
    assert spec.wants_recreated_trajectory() is True
    assert spec.wants_literature() is True


def test_pyro_validation_reports_missing_fields() -> None:
    backend = PyroBackend()
    spec = ExperimentSpec.from_dict({"backend": "pyro"})

    report = backend.validate(spec)

    assert report.valid is False
    assert "model_ref" in report.missing_fields
    assert "design_variables" in report.missing_fields


def test_foster_ab_test_default_spec_matches_paper_defaults() -> None:
    spec = build_foster_experiment_spec("ab_test_linear")

    assert spec.objective.estimator == "posterior_eig"
    assert spec.compute_budget.num_outer_samples == 10
    assert spec.compute_budget.guide_training_steps == 2500
    assert spec.compute_budget.num_optimization_steps == 11
    assert spec.backend_options["optimization_strategy"] == "candidate_grid"
    assert len(spec.backend_options["candidate_designs"]) == 11
    assert spec.backend_options["final_num_samples"] == 500
    assert spec.guide_ref.endswith(":foster_ab_test_linear_posterior_guide")


def test_foster_revealed_preference_vi_spec_uses_paper_vi_defaults() -> None:
    spec = build_foster_experiment_spec("revealed_preference", estimator="vi_eig")

    assert spec.objective.estimator == "vi_eig"
    assert spec.loss_ref.endswith(":make_foster_trace_elbo_loss")
    assert spec.compute_budget.num_outer_samples == 1
    assert spec.compute_budget.guide_training_steps == 5000
    assert spec.backend_options["optimization_strategy"] == "candidate_grid"
    assert len(spec.backend_options["candidate_designs"]) == 20
    assert spec.guide_ref.endswith(":foster_revealed_preference_vi_guide")
