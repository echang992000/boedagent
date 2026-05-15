from __future__ import annotations

from boed_agent.backends.lfiax_backend import LFIAXBackend
from boed_agent.models import ExperimentSpec


def test_lfiax_backend_validate_surfaces_cli_payload_and_required_fields(monkeypatch) -> None:
    backend = LFIAXBackend()
    spec = ExperimentSpec.from_dict(
        {
            "backend": "lfiax",
            "simulator_ref": "demo.simulators:simulate",
            "prior_sampler_ref": "demo.simulators:sample_prior",
            "objective": {"estimator": "lf_pce_eig_scan"},
        }
    )

    monkeypatch.setattr(
        backend,
        "_run_cli_json",
        lambda args, spec=None, extra_args=None: {  # noqa: ARG005
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_fields": [],
            "backend": "lfiax",
        },
    )

    report = backend.validate(spec)

    assert report.valid is False
    assert "design_variables" in report.missing_fields
    assert "backend_options.design_mode" in report.missing_fields
    assert "differentiable" in report.missing_fields


def test_lfiax_backend_optimize_maps_cli_output_and_recreates_trajectory(monkeypatch) -> None:
    backend = LFIAXBackend()
    spec = ExperimentSpec.from_dict(
        {
            "backend": "lfiax",
            "recreate_trajectory": True,
            "simulator_ref": "demo.simulators:simulate",
            "prior_sampler_ref": "demo.simulators:sample_prior",
            "differentiable": False,
            "design_variables": [{"name": "x", "lower": -1.0, "upper": 1.0, "initial": 0.0}],
            "objective": {"estimator": "lf_pce_eig_scan"},
            "backend_options": {"design_mode": "distribution"},
        }
    )

    monkeypatch.setattr(
        backend,
        "_run_cli_json",
        lambda args, spec=None, extra_args=None: {  # noqa: ARG005
            "status": "completed",
            "backend": "lfiax",
            "estimator": "lf_pce_eig_scan",
            "execution_path": "black_box",
            "design": [0.2],
            "eig": 0.5,
            "xi_mu": [0.2],
            "xi_stddev": [0.1],
            "history": [
                {"step": 0, "design": [0.0], "eig": 0.1},
                {"step": 1, "xi_mu": [0.2], "xi_stddev": [0.1], "eig": 0.5},
            ],
            "warnings": [],
            "artifacts": {
                "likelihood_checkpoint": "/tmp/lfiax.pkl",
                "sigma_history": [[0.4], [0.1]],
            },
        },
    )

    result = backend.optimize(spec)

    assert result.status == "completed"
    assert result.design == [0.2]
    assert len(result.history) == 2
    assert result.artifacts["execution_path"] == "black_box"
    assert result.artifacts["trajectory_recreated"] is True
    assert len(result.artifacts["optimized_design_histories"]) == 1


def test_lfiax_backend_estimate_eig_returns_error_when_cli_is_unavailable(monkeypatch) -> None:
    backend = LFIAXBackend()
    spec = ExperimentSpec.from_dict(
        {
            "backend": "lfiax",
            "simulator_ref": "demo.simulators:simulate",
            "prior_sampler_ref": "demo.simulators:sample_prior",
            "differentiable": True,
            "design_variables": [{"name": "x", "lower": -1.0, "upper": 1.0, "initial": 0.0}],
            "objective": {"estimator": "lf_pce_eig_scan"},
            "backend_options": {"design_mode": "point"},
        }
    )

    def fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("cli-anything-lfiax not found")

    monkeypatch.setattr(backend, "_run_cli_json", fail)

    estimate = backend.estimate_eig(spec, design=[0.1])

    assert estimate.status == "error"
    assert estimate.value is None
    assert "cli-anything-lfiax not found" in estimate.warnings[0]
