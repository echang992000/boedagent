from __future__ import annotations

import json
from pathlib import Path

from boed_agent.backends.lfiax_backend import LFIAXBackend
from boed_agent.backends.pyro_backend import PyroBackend
from boed_agent.backends.registry import BackendRegistry
from boed_agent.clarification.interactive import (
    DEMO_FOSTER_TEMPLATE,
    DEMO_LINEAR_MODEL_REF,
    DEMO_LINEAR_OPTIM_REF,
    DEMO_LINEAR_SUMMARY,
    DEMO_LINEAR_DESIGN_VARIABLE,
    collect_interactive_spec,
)
from boed_agent.clarification.planner import ClarificationPlanner
from boed_agent.cli import main
from boed_agent.literature import advisory as literature_advisory
from boed_agent.literature.llm_client import RecordingLLMClient
from boed_agent.models import ExperimentSpec, OptimizationResult, OptimizationStep, ValidationReport
from boed_agent.tools import registry as tool_registry_module


def write_spec(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_cli_list_backends(capsys) -> None:
    exit_code = main(["list-backends"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "pyro" in captured.out
    assert "lfiax" in captured.out


def test_cli_validate_and_run_lfiax_backend(tmp_path: Path, capsys, monkeypatch) -> None:
    spec_path = write_spec(
        tmp_path / "lfiax.json",
        {
            "backend": "lfiax",
            "simulator_ref": "demo.simulators:simulate",
            "prior_sampler_ref": "demo.simulators:sample_prior",
            "differentiable": False,
            "design_variables": [{"name": "x", "lower": -1.0, "upper": 1.0}],
            "objective": {"estimator": "lf_pce_eig"},
            "artifacts": {"output_dir": str(tmp_path / "artifacts")},
        },
    )

    def fake_validate(self: LFIAXBackend, spec: ExperimentSpec) -> ValidationReport:
        _ = self, spec
        return ValidationReport(valid=True, backend="lfiax")

    def fake_optimize(self: LFIAXBackend, spec: ExperimentSpec) -> OptimizationResult:
        return OptimizationResult(
            backend="lfiax",
            estimator=spec.objective.estimator,
            status="completed",
            design=spec.effective_initial_design(),
            eig=0.25,
            artifacts={"execution_path": "black_box"},
        )

    monkeypatch.setattr(LFIAXBackend, "validate", fake_validate)
    monkeypatch.setattr(LFIAXBackend, "optimize", fake_optimize)

    validate_exit = main(["validate", str(spec_path)])
    validate_output = json.loads(capsys.readouterr().out)
    run_exit = main(["run", str(spec_path)])
    run_output = json.loads(capsys.readouterr().out)

    assert validate_exit == 0
    assert validate_output["validation"]["valid"] is True
    assert run_exit == 0
    assert run_output["status"] == "completed"


def test_cli_run_condenses_history_and_saves_npz(tmp_path: Path, capsys, monkeypatch) -> None:
    spec_path = write_spec(
        tmp_path / "lfiax.json",
        {
            "backend": "lfiax",
            "simulator_ref": "demo.simulators:simulate",
            "prior_sampler_ref": "demo.simulators:sample_prior",
            "differentiable": False,
            "design_variables": [{"name": "x", "lower": -1.0, "upper": 1.0}],
            "objective": {"estimator": "lf_pce_eig"},
            "artifacts": {"output_dir": str(tmp_path / "artifacts")},
        },
    )

    def fake_validate(self: LFIAXBackend, spec: ExperimentSpec) -> ValidationReport:
        _ = self, spec
        return ValidationReport(valid=True, backend="lfiax")

    def fake_optimize(self: LFIAXBackend, spec: ExperimentSpec) -> OptimizationResult:
        return OptimizationResult(
            backend="lfiax",
            estimator=spec.objective.estimator,
            status="completed",
            design=spec.effective_initial_design(),
            eig=0.25,
            history=[
                OptimizationStep(step=index, design=[index / 10.0], eig=0.1 * index)
                for index in range(8)
            ],
            artifacts={"sigma_history": [[0.0]] * 8},
        )

    monkeypatch.setattr(LFIAXBackend, "validate", fake_validate)
    monkeypatch.setattr(LFIAXBackend, "optimize", fake_optimize)

    run_exit = main(["run", str(spec_path)])
    run_output = json.loads(capsys.readouterr().out)

    assert run_exit == 0
    assert run_output["status"] == "completed"
    assert run_output["history_summary"]["num_steps"] == 8
    assert "history" not in run_output
    assert "checkpoints" not in run_output["history_summary"]
    assert "sigma_history" not in run_output["artifacts"]
    if "optimization_history_npz" in run_output["artifacts"]:
        assert Path(run_output["artifacts"]["optimization_history_npz"]).exists()
    else:
        assert any("numpy" in warning for warning in run_output["warnings"])


def test_cli_run_returns_clarification_for_invalid_spec(tmp_path: Path, capsys) -> None:
    spec_path = write_spec(
        tmp_path / "invalid.json",
        {
            "backend": "pyro",
            "model_ref": "boed_agent.demo.pyro_linear:pyro_linear_model",
            "artifacts": {"output_dir": str(tmp_path / "artifacts")},
        },
    )

    exit_code = main(["run", str(spec_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["status"] == "needs_clarification"


def test_cli_literature_dry_run_returns_advisory_payload(tmp_path: Path, capsys, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "paper.md").write_text(
        "# Demo prior note\n\n"
        "We estimate alpha with a Beta(2,5) prior. "
        "Variational Bayesian experimental design performs well here.\n"
    )
    spec_path = write_spec(
        tmp_path / "literature.json",
        {
            "backend": "pyro",
            "model_ref": "demo.module:model",
            "problem_summary": "Toy BOED literature prior for alpha",
            "use_literature": True,
            "literature_source_mode": "local",
            "literature_corpus_dir": str(corpus_dir),
            "target_latent_labels": ["alpha"],
            "observation_labels": ["y"],
            "design_variables": [{"name": "dose", "lower": 0.0, "upper": 1.0}],
        },
    )

    def responder(prompt, tier):
        if "For each sentence below" in prompt:
            return json.dumps(
                [
                    {
                        "id": 0,
                        "type": "prior_distribution",
                        "value": {
                            "parameter": "alpha",
                            "distribution": "Beta",
                            "params": {"a": 2, "b": 5},
                        },
                    }
                ]
            )
        if "propose a prior" in prompt:
            return json.dumps(
                {
                    "distribution": "Beta",
                    "params": {"a": 2, "b": 5},
                    "reasoning": "local corpus supports Beta",
                    "cited_papers": ["title:15b3fd123f4f8fbf"],
                }
            )
        if "rank the candidate" in prompt:
            return json.dumps(
                {
                    "ranked": ["PyroVI", "MINEBED"],
                    "reasoning": "explicit-model literature points to PyroVI",
                    "cited_papers": ["title:15b3fd123f4f8fbf"],
                }
            )
        return "{}"

    monkeypatch.setattr(
        tool_registry_module,
        "run_literature_dry_run",
        lambda spec, **kwargs: literature_advisory.run_literature_dry_run(
            spec,
            llm=RecordingLLMClient(responder=responder),
        ),
    )

    exit_code = main(["literature-dry-run", str(spec_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["advisory_only"] is True
    assert output["literature_report"] is not None
    assert output["reasoning_trace"] is not None
    assert output["backend_choice"]["backend"] == "pyro"
    assert "alpha" in output["prior_used"]["distributions"]


def test_cli_run_warns_that_literature_is_advisory_only(tmp_path: Path, capsys, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    spec_path = write_spec(
        tmp_path / "lfiax_with_literature.json",
        {
            "backend": "lfiax",
            "simulator_ref": "demo.simulators:simulate",
            "prior_sampler_ref": "demo.simulators:sample_prior",
            "differentiable": False,
            "use_literature": True,
            "literature_source_mode": "local",
            "literature_corpus_dir": str(corpus_dir),
            "design_variables": [{"name": "x", "lower": -1.0, "upper": 1.0}],
            "objective": {"estimator": "lf_pce_eig"},
            "artifacts": {"output_dir": str(tmp_path / "artifacts")},
        },
    )

    def fake_validate(self: LFIAXBackend, spec: ExperimentSpec) -> ValidationReport:
        _ = self, spec
        return ValidationReport(valid=True, backend="lfiax")

    def fake_optimize(self: LFIAXBackend, spec: ExperimentSpec) -> OptimizationResult:
        return OptimizationResult(
            backend="lfiax",
            estimator=spec.objective.estimator,
            status="completed",
            design=spec.effective_initial_design(),
            eig=0.25,
        )

    monkeypatch.setattr(LFIAXBackend, "validate", fake_validate)
    monkeypatch.setattr(LFIAXBackend, "optimize", fake_optimize)

    exit_code = main(["run", str(spec_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert any("advisory only" in warning.lower() for warning in output["warnings"])


def test_collect_interactive_spec_populates_pyro_linear_demo() -> None:
    planner = ClarificationPlanner(BackendRegistry.default())
    answers = iter(["1"] * 10)
    output_lines: list[str] = []

    spec, transcript = collect_interactive_spec(
        ExperimentSpec.from_dict({}),
        planner,
        input_fn=lambda _: next(answers),
        output_fn=output_lines.append,
    )

    assert spec.backend == "pyro"
    assert spec.problem_summary == DEMO_LINEAR_SUMMARY
    assert spec.model_ref == DEMO_LINEAR_MODEL_REF
    assert spec.guide_ref == "boed_agent.demo.pyro_linear:pyro_linear_marginal_guide"
    assert spec.optim_ref == DEMO_LINEAR_OPTIM_REF
    assert spec.observation_labels == ["y"]
    assert spec.target_latent_labels == ["theta"]
    assert spec.objective.estimator == "marginal_eig"
    assert spec.compute_budget.num_outer_samples == 4
    assert spec.compute_budget.guide_training_steps == 2
    assert len(transcript) == 10
    assert any(line == "Options:" for line in output_lines)
    assert any("marginal_eig" in line for line in output_lines)
    assert any(DEMO_LINEAR_MODEL_REF in line for line in output_lines)


def test_collect_interactive_spec_rejects_raw_budget_value_at_selection_step() -> None:
    planner = ClarificationPlanner(BackendRegistry.default())
    answers = iter(["4", "1", "1"])
    output_lines: list[str] = []

    spec, transcript = collect_interactive_spec(
        ExperimentSpec.from_dict(
            {
                "backend": "pyro",
                "model_ref": DEMO_LINEAR_MODEL_REF,
                "guide_ref": "boed_agent.demo.pyro_linear:pyro_linear_marginal_guide",
                "optim_ref": DEMO_LINEAR_OPTIM_REF,
                "design_variables": [
                    {
                        "name": DEMO_LINEAR_DESIGN_VARIABLE.name,
                        "lower": DEMO_LINEAR_DESIGN_VARIABLE.lower,
                        "upper": DEMO_LINEAR_DESIGN_VARIABLE.upper,
                        "initial": DEMO_LINEAR_DESIGN_VARIABLE.initial,
                    }
                ],
                "observation_labels": ["y"],
                "target_latent_labels": ["theta"],
                "objective": {"estimator": "marginal_eig"},
            }
        ),
        planner,
        input_fn=lambda _: next(answers),
        output_fn=output_lines.append,
    )

    assert spec.compute_budget.num_outer_samples == 4
    assert spec.compute_budget.guide_training_steps == 2
    assert transcript[0]["field"] == "compute_budget.num_outer_samples"
    assert any("Enter one of the option numbers shown above." in line for line in output_lines)
    assert any("custom/manual option number" in line for line in output_lines)


def test_collect_interactive_spec_populates_foster_paper_demo() -> None:
    planner = ClarificationPlanner(BackendRegistry.default())
    answers = iter(["2", "4"])
    output_lines: list[str] = []

    spec, transcript = collect_interactive_spec(
        ExperimentSpec.from_dict({"metadata": {"demo_template": DEMO_FOSTER_TEMPLATE}}),
        planner,
        input_fn=lambda _: next(answers),
        output_fn=output_lines.append,
    )

    assert spec.backend == "pyro"
    assert spec.metadata["paper_experiment"] == "revealed_preference"
    assert spec.objective.estimator == "vi_eig"
    assert spec.loss_ref == "boed_agent.demo.foster_variational:make_foster_trace_elbo_loss"
    assert spec.compute_budget.num_outer_samples == 1
    assert spec.compute_budget.guide_training_steps == 5000
    assert transcript[0]["field"] == "metadata.paper_experiment"
    assert transcript[1]["field"] == "objective.estimator"
    assert any("A/B Test (Linear)" in line for line in output_lines)
    assert any("Revealed Preference" in line for line in output_lines)
    assert any("vi_eig" in line for line in output_lines)


def test_cli_run_interactive_empty_pyro_spec(tmp_path: Path, capsys, monkeypatch) -> None:
    spec_path = write_spec(tmp_path / "empty.json", {})
    answers = iter(["1"] * 10)

    def fake_validate(self: PyroBackend, spec: ExperimentSpec) -> ValidationReport:
        _ = spec
        return ValidationReport(valid=True, backend="pyro")

    def fake_optimize(self: PyroBackend, spec: ExperimentSpec) -> OptimizationResult:
        return OptimizationResult(
            backend="pyro",
            estimator=spec.objective.estimator,
            status="ok",
            design=spec.effective_initial_design(),
            eig=0.42,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(PyroBackend, "validate", fake_validate)
    monkeypatch.setattr(PyroBackend, "optimize", fake_optimize)

    exit_code = main(["run", str(spec_path), "--interactive"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"backend": "pyro"' in output
    assert '"status": "ok"' in output


def test_cli_run_interactive_foster_spec(tmp_path: Path, capsys, monkeypatch) -> None:
    spec_path = write_spec(
        tmp_path / "foster_interactive.json",
        {
            "metadata": {"demo_template": DEMO_FOSTER_TEMPLATE},
            "artifacts": {"output_dir": str(tmp_path / "artifacts")},
        },
    )
    answers = iter(["1", "1"])

    def fake_validate(self: PyroBackend, spec: ExperimentSpec) -> ValidationReport:
        assert spec.metadata["paper_experiment"] == "ab_test_linear"
        assert spec.objective.estimator == "posterior_eig"
        return ValidationReport(valid=True, backend="pyro")

    def fake_optimize(self: PyroBackend, spec: ExperimentSpec) -> OptimizationResult:
        return OptimizationResult(
            backend="pyro",
            estimator=spec.objective.estimator,
            status="ok",
            design=[5.0],
            eig=0.77,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(PyroBackend, "validate", fake_validate)
    monkeypatch.setattr(PyroBackend, "optimize", fake_optimize)

    exit_code = main(["run", str(spec_path), "--interactive"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"backend": "pyro"' in output
    assert '"status": "ok"' in output
