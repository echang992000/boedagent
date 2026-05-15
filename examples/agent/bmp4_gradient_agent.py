"""BMP4 gradient example: literature-informed fitting plus next-design BOED."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from boed_agent import BOEDAgent, SimpleSimulator, TokenBudget
from boed_agent.backends.pyro_backend import PyroBackend
from boed_agent.literature.advisory import build_literature_llm_client
from boed_agent.literature.clients import LocalCorpusClient
from boed_agent.literature.extraction import ExtractionConfig
from boed_agent.literature.llm_client import LLMClient, NullLLMClient
from boed_agent.literature.reasoning import ReasoningConfig
from boed_agent.literature.search import (
    LiteratureSearchConfig,
    LiteratureSearchModule,
    SourceBundle,
)
from boed_agent.prior_builder import AugmentedPrior, DistributionSpec
from examples.cases.bmp4_gradient.data import (
    build_joint_bmp4_gradient_data,
    load_bmp4_gradient_data,
    make_log_spaced_grid,
)
from examples.cases.bmp4_gradient.inference import (
    fit_variational_model,
    optimize_empirical_eig,
    optimize_selector_dose_empirical_eig,
    summarize_posterior_samples,
)
from examples.cases.bmp4_gradient.promisys_hyperparams import (
    PromisysHyperparams,
    coerce_promisys_hyperparams,
)
from examples.cases.bmp4_gradient import promisys_onestep as promisys_onestep_model
from examples.cases.bmp4_gradient import promisys_twostep as promisys_twostep_model
from examples.cases.bmp4_gradient.pyro import multireceptor_hierarchical as multireceptor_hierarchical_model
from examples.cases.bmp4_gradient.pyro import postfit_boed as bmp4_postfit_boed
from examples.cases.bmp4_gradient.plotting import (
    save_eig_optimization_plot,
    save_joint_posterior_predictive_plot,
    save_prior_posterior_comparison_plot,
    save_posterior_predictive_plot,
)
from examples.cases.bmp4_gradient.registry import get_model_family_registry


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "bmp4_gradient"
DEFAULT_PROBLEM_PATH = REPO_ROOT / "examples" / "cases" / "bmp4_gradient" / "problem.json"
DEFAULT_EIG_ESTIMATOR = "empirical"


def run_bmp4_gradient_example(
    *,
    provider: str | None = None,
    model: str | None = None,
    llm_client: LLMClient | None = None,
    problem_path: str | Path = DEFAULT_PROBLEM_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cell_lines: list[str] | None = None,
    families: list[str] | None = None,
    literature_prior_jsons: list[str] | None = None,
    fit_steps: int = 250,
    fit_learning_rate: float = 0.02,
    posterior_samples: int = 256,
    eig_estimator: str = DEFAULT_EIG_ESTIMATOR,
    eig_steps: int = 100,
    eig_guide_steps: int = 250,
    eig_learning_rate: float = 0.05,
    eig_guide_learning_rate: float = 0.05,
    eig_outer_samples: int = 64,
    eig_inner_samples: int | None = None,
    infonce_lambda: float = promisys_onestep_model.DEFAULT_INFONCE_LAMBDA,
    design_dist_init_std: float = promisys_onestep_model.DEFAULT_DESIGN_DIST_INIT_STD,
    design_temperature_scale: float = promisys_onestep_model.DEFAULT_DESIGN_TEMPERATURE_SCALE,
    selector_temperature_final: float = promisys_onestep_model.DEFAULT_SELECTOR_TEMPERATURE_FINAL,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    normalizer_data_dir: str | Path = promisys_onestep_model.DEFAULT_NORMALIZER_DATA_DIR,
    receptor_noise_log_sd: float = 0.05,
    snpe_steps: int = 500,
    snpe_simulations: int = 512,
    snpe_learning_rate: float = 1e-3,
    mcmc_warmup: int = 200,
    mcmc_samples: int = 256,
    promisys_hyperparams: PromisysHyperparams | dict[str, Any] | str | Path | None = None,
    promisys_hyperparams_json: str | Path | None = None,
) -> dict[str, Any]:
    if promisys_hyperparams is not None and promisys_hyperparams_json is not None:
        raise ValueError("Pass only one of promisys_hyperparams or promisys_hyperparams_json.")
    resolved_promisys_hyperparams = coerce_promisys_hyperparams(
        promisys_hyperparams_json if promisys_hyperparams_json is not None else promisys_hyperparams
    )
    problem = _load_problem_bundle(problem_path)
    data_bundle = load_bmp4_gradient_data(problem["data"]["observed_data_ref"])
    registry = get_model_family_registry()
    selected_cell_lines = cell_lines or list(problem["metadata"]["cell_lines"])
    selected_families = families or list(registry)
    unknown_cell_lines = sorted(set(selected_cell_lines) - set(data_bundle))
    if unknown_cell_lines:
        raise ValueError(f"Unknown BMP4 cell lines requested: {unknown_cell_lines}")
    unknown_families = sorted(set(selected_families) - set(registry))
    if unknown_families:
        raise ValueError(f"Unknown BMP4 model families requested: {unknown_families}")
    prior_overrides = _resolve_literature_prior_overrides(
        literature_prior_jsons,
        selected_families=selected_families,
    )

    literature_llm, llm_warnings = _resolve_llm_client(
        llm_client=llm_client,
        provider=provider,
        model=model,
    )
    shared_family_priors = _build_family_priors(
        problem=problem,
        registry=registry,
        selected_families=selected_families,
        receptor_names=next(iter(data_bundle.values())).receptor_names,
        llm=literature_llm,
        prior_overrides=prior_overrides,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, Any]] = []
    promisys_workflow_modules = {
        "promisys_onestep": promisys_onestep_model,
        "promisys_twostep": promisys_twostep_model,
    }

    for family_name in selected_families:
        family_spec = registry[family_name]
        family_prior_payload = shared_family_priors[family_name]
        if family_spec.execution_mode in promisys_workflow_modules:
            workflow_module = promisys_workflow_modules[family_spec.execution_mode]
            workflow_fn_name = (
                "run_promisys_twostep_workflow"
                if family_name == "promisys_twostep"
                else "run_promisys_onestep_workflow"
            )
            workflow_fn = getattr(workflow_module, workflow_fn_name)
            print(
                f"[bmp4_gradient_agent] Starting {family_name} family run "
                f"for {', '.join(selected_cell_lines)}",
                flush=True,
            )
            joint_data = build_joint_bmp4_gradient_data(
                data_bundle,
                cell_lines=selected_cell_lines,
            )
            run_label = _joint_run_label(joint_data.cell_lines)
            prior_output_label = _prior_output_label(family_prior_payload)
            run_dir = _run_dir_for_prior(
                output_root=output_root,
                family_name=family_name,
                run_label=run_label,
                prior_output_label=prior_output_label,
            )
            eig_result = workflow_fn(
                joint_data=joint_data,
                run_dir=run_dir,
                normalizer_data_dir=normalizer_data_dir,
                receptor_noise_log_sd=receptor_noise_log_sd,
                snpe_steps=snpe_steps,
                snpe_simulations=snpe_simulations,
                snpe_learning_rate=snpe_learning_rate,
                likelihood_steps=fit_steps,
                likelihood_learning_rate=fit_learning_rate,
                posterior_sample_count=posterior_samples,
                eig_steps=eig_steps,
                eig_outer_samples=eig_outer_samples,
                eig_inner_samples=eig_inner_samples,
                eig_learning_rate=eig_learning_rate,
                infonce_lambda=infonce_lambda,
                design_dist_init_std=design_dist_init_std,
                design_temperature_scale=design_temperature_scale,
                selector_temperature_final=selector_temperature_final,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                mcmc_warmup=mcmc_warmup,
                mcmc_samples=mcmc_samples,
                literature_prior=family_prior_payload["prior_used"],
                promisys_hyperparams=resolved_promisys_hyperparams,
            )

            literature_path = run_dir / "literature_prior.json"
            literature_payload = {
                "family": family_name,
                "cell_lines": list(joint_data.cell_lines),
                "prior_output_label": prior_output_label,
                "literature_prior_source": family_prior_payload.get("source_path"),
                "literature_report": family_prior_payload["literature_report"],
                "prior_used": family_prior_payload["prior_used"].to_dict(),
                "translated_prior": {"family": family_name, "sites": {}, "warnings": []},
                "snpe_prior_mode": "shared_amortized_posterior",
            }
            with literature_path.open("w", encoding="utf-8") as handle:
                json.dump(literature_payload, handle, indent=2)
            _annotate_promisys_summaries(
                run_dir=run_dir,
                prior_output_label=prior_output_label,
                literature_prior_source=family_prior_payload.get("source_path"),
            )

            run_records.append(
                {
                    **eig_result,
                    "prior_output_label": prior_output_label,
                    "llm_warnings": llm_warnings,
                    "literature_prior_source": family_prior_payload.get("source_path"),
                    "literature_prior_path": str(literature_path),
                }
            )
            continue

        if family_spec.execution_mode == "joint_cell_lines":
            joint_data = build_joint_bmp4_gradient_data(
                data_bundle,
                cell_lines=selected_cell_lines,
            )
            translated_prior = family_spec.translate_prior(
                family_prior_payload["prior_used"],
                joint_data,
                joint_data.receptor_names,
            )
            fit_model = family_spec.make_fit_model(
                translated_prior,
                joint_data,
                joint_data.receptor_names,
            )

            concentration = torch.tensor(joint_data.bmp4_conc, dtype=torch.float32)
            responses = torch.tensor(joint_data.x_obs, dtype=torch.float32)
            fit_result = fit_variational_model(
                fit_model,
                concentration,
                responses,
                num_steps=fit_steps,
                learning_rate=fit_learning_rate,
                num_posterior_samples=posterior_samples,
            )

            plot_grid = torch.tensor(make_log_spaced_grid(joint_data), dtype=torch.float32)
            plot_design = plot_grid.unsqueeze(0).repeat(len(joint_data.cell_lines), 1)
            predictive = family_spec.predictive_draws(
                plot_design,
                fit_result.posterior_samples,
                joint_data,
                joint_data.receptor_names,
            )
            if eig_estimator == "empirical" and family_name == "multireceptor_hierarchical":
                log10_dose_min, log10_dose_max = multireceptor_hierarchical_model.make_log10_dose_bounds(
                    min_positive_concentration=joint_data.min_positive_concentration,
                    max_concentration=joint_data.max_concentration,
                )
                eig_result = optimize_selector_dose_empirical_eig(
                    posterior_samples=fit_result.posterior_samples,
                    scalar_log_likelihood=lambda y_value, design, posterior_samples, receptor_names: family_spec.scalar_log_likelihood(
                        y_value,
                        design,
                        posterior_samples,
                        joint_data,
                        receptor_names,
                        log10_dose_min=log10_dose_min,
                        log10_dose_max=log10_dose_max,
                    ),
                    receptor_names=joint_data.receptor_names,
                    design_decoder=lambda raw_design: multireceptor_hierarchical_model.decode_selector_dose_design(
                        raw_design,
                        cell_line_names=joint_data.cell_lines,
                        log10_dose_min=log10_dose_min,
                        log10_dose_max=log10_dose_max,
                    ),
                    cell_line_count=len(joint_data.cell_lines),
                    dose_count=1,
                    log10_dose_min=log10_dose_min,
                    log10_dose_max=log10_dose_max,
                    num_steps=eig_steps,
                    learning_rate=eig_learning_rate,
                    outer_samples=eig_outer_samples,
                )
            elif eig_estimator == "empirical":
                eig_result = optimize_empirical_eig(
                    posterior_samples=fit_result.posterior_samples,
                    scalar_log_likelihood=lambda y_value, concentration, posterior_samples, receptor_names: family_spec.scalar_log_likelihood(
                        y_value,
                        concentration,
                        posterior_samples,
                        joint_data,
                        receptor_names,
                    ),
                    receptor_names=joint_data.receptor_names,
                    lower=float(problem["shared"]["design_variables"][0]["lower"]),
                    upper=float(problem["shared"]["design_variables"][0]["upper"]),
                    min_positive_lower=joint_data.min_positive_concentration * 0.5,
                    num_steps=eig_steps,
                    learning_rate=eig_learning_rate,
                    outer_samples=eig_outer_samples,
                )
            else:
                eig_result = _run_variational_boed(
                    family_name=family_name,
                    posterior_samples=fit_result.posterior_samples,
                    receptor_names=joint_data.receptor_names,
                    cell_line_names=joint_data.cell_lines,
                    min_positive_concentration=joint_data.min_positive_concentration,
                    max_concentration=joint_data.max_concentration,
                    eig_estimator=eig_estimator,
                    eig_steps=eig_steps,
                    eig_guide_steps=eig_guide_steps,
                    eig_learning_rate=eig_learning_rate,
                    eig_guide_learning_rate=eig_guide_learning_rate,
                    eig_outer_samples=eig_outer_samples,
                    eig_inner_samples=eig_inner_samples,
                )

            run_label = _joint_run_label(joint_data.cell_lines)
            prior_output_label = _prior_output_label(family_prior_payload)
            run_dir = _run_dir_for_prior(
                output_root=output_root,
                family_name=family_name,
                run_label=run_label,
                prior_output_label=prior_output_label,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            literature_path = run_dir / "literature_prior.json"
            fit_summary_path = run_dir / "fit_summary.json"
            eig_summary_path = run_dir / "eig_optimization_summary.json"
            posterior_samples_path = run_dir / "posterior_samples.pt"
            posterior_predictive_path = run_dir / "posterior_predictive.pt"
            posterior_plot_path = run_dir / "posterior_predictive.png"
            prior_posterior_plot_path = run_dir / "prior_posterior_comparison.png"
            prior_posterior_positive_plot_path = run_dir / "prior_posterior_comparison_positive.png"
            eig_plot_path = run_dir / "eig_optimization.png"

            literature_payload = {
                "family": family_name,
                "cell_lines": list(joint_data.cell_lines),
                "prior_output_label": prior_output_label,
                "literature_report": family_prior_payload["literature_report"],
                "prior_used": family_prior_payload["prior_used"].to_dict(),
                "translated_prior": translated_prior.to_dict(),
            }
            fit_summary = {
                "family": family_name,
                "cell_lines": list(joint_data.cell_lines),
                "prior_output_label": prior_output_label,
                "candidate_name": family_spec.candidate_name,
                "loss_history": fit_result.loss_history,
                "posterior_summary": summarize_posterior_samples(fit_result.posterior_samples),
                "translated_prior": translated_prior.to_dict(),
                "q_obs_mode": "joint_qpcr_measurement_layer",
            }

            with literature_path.open("w", encoding="utf-8") as handle:
                json.dump(literature_payload, handle, indent=2)
            with fit_summary_path.open("w", encoding="utf-8") as handle:
                json.dump(fit_summary, handle, indent=2)
            with eig_summary_path.open("w", encoding="utf-8") as handle:
                json.dump(eig_result, handle, indent=2)

            torch.save(fit_result.posterior_samples, posterior_samples_path)
            torch.save(
                {
                    "cell_lines": list(joint_data.cell_lines),
                    "grid_concentration": plot_design.detach().cpu(),
                    "predictive_mu": predictive["mu"].detach().cpu(),
                    "predictive_y": predictive["y"].detach().cpu(),
                },
                posterior_predictive_path,
            )
            save_joint_posterior_predictive_plot(
                observed_concentration=joint_data.bmp4_conc,
                observed_response=joint_data.x_obs,
                grid_concentration=plot_design.detach().cpu().numpy(),
                predictive_draws=predictive["y"].detach().cpu().numpy(),
                cell_lines=joint_data.cell_lines,
                output_path=posterior_plot_path,
                title=(
                    f"Posterior predictive BMP4 dose-response across selected cell lines: "
                    f"{_display_family_name(family_name)}"
                ),
            )
            save_prior_posterior_comparison_plot(
                translated_prior=translated_prior,
                posterior_samples=fit_result.posterior_samples,
                output_path=prior_posterior_plot_path,
                title=f"Prior vs posterior parameter comparison: {_display_family_name(family_name)}",
                cell_lines=list(joint_data.cell_lines),
                receptor_names=list(joint_data.receptor_names),
                kd_prior_shift=joint_data.kd_prior_shift,
            )
            save_prior_posterior_comparison_plot(
                translated_prior=translated_prior,
                posterior_samples=fit_result.posterior_samples,
                output_path=prior_posterior_positive_plot_path,
                title=(
                    f"Prior vs posterior parameter comparison on positive scale: "
                    f"{_display_family_name(family_name)}"
                ),
                cell_lines=list(joint_data.cell_lines),
                receptor_names=list(joint_data.receptor_names),
                kd_prior_shift=joint_data.kd_prior_shift,
                scale="positive",
            )
            save_eig_optimization_plot(
                eig_result["history"],
                eig_plot_path,
                title=f"Next-design EIG optimization: {_display_family_name(family_name)}",
            )

            run_records.append(
                {
                    "family": family_name,
                    "cell_lines": list(joint_data.cell_lines),
                    "prior_output_label": prior_output_label,
                    "run_dir": str(run_dir),
                    "best_design": eig_result["best_design"],
                    "best_dose": eig_result.get("best_dose"),
                    "best_log10_dose": eig_result.get("best_log10_dose"),
                    "best_cell_line": eig_result.get("best_cell_line"),
                    "best_selector_probs": eig_result.get("best_selector_probs"),
                    "best_eig": eig_result["best_eig"],
                    "eig_estimator": eig_result.get("estimator", eig_estimator),
                    "posterior_samples_path": str(posterior_samples_path),
                    "posterior_predictive_path": str(posterior_predictive_path),
                    "posterior_predictive_plot": str(posterior_plot_path),
                    "prior_posterior_plot": str(prior_posterior_plot_path),
                    "prior_posterior_positive_plot": str(prior_posterior_positive_plot_path),
                    "eig_optimization_plot": str(eig_plot_path),
                    "llm_warnings": llm_warnings,
                    "literature_prior_source": family_prior_payload.get("source_path"),
                }
            )
            continue

        for cell_line in selected_cell_lines:
            cell_data = data_bundle[cell_line]
            translated_prior = family_spec.translate_prior(
                family_prior_payload["prior_used"],
                cell_data,
                cell_data.receptor_names,
            )
            fit_model = family_spec.make_fit_model(
                translated_prior,
                cell_data,
                cell_data.receptor_names,
            )

            concentration = torch.tensor(cell_data.bmp4_conc, dtype=torch.float32)
            responses = torch.tensor(cell_data.x_obs, dtype=torch.float32)
            fit_result = fit_variational_model(
                fit_model,
                concentration,
                responses,
                num_steps=fit_steps,
                learning_rate=fit_learning_rate,
                num_posterior_samples=posterior_samples,
            )

            plot_grid = torch.tensor(make_log_spaced_grid(cell_data), dtype=torch.float32)
            predictive = family_spec.predictive_draws(
                plot_grid,
                fit_result.posterior_samples,
                cell_data,
                cell_data.receptor_names,
            )
            if eig_estimator == "empirical":
                eig_result = optimize_empirical_eig(
                    posterior_samples=fit_result.posterior_samples,
                    scalar_log_likelihood=lambda y_value, concentration, posterior_samples, receptor_names: family_spec.scalar_log_likelihood(
                        y_value,
                        concentration,
                        posterior_samples,
                        cell_data,
                        receptor_names,
                    ),
                    receptor_names=cell_data.receptor_names,
                    lower=float(problem["shared"]["design_variables"][0]["lower"]),
                    upper=float(problem["shared"]["design_variables"][0]["upper"]),
                    min_positive_lower=cell_data.min_positive_concentration * 0.5,
                    num_steps=eig_steps,
                    learning_rate=eig_learning_rate,
                    outer_samples=eig_outer_samples,
                )
            else:
                eig_result = _run_variational_boed(
                    family_name=family_name,
                    posterior_samples=fit_result.posterior_samples,
                    receptor_names=cell_data.receptor_names,
                    cell_line_names=(cell_line,),
                    min_positive_concentration=cell_data.min_positive_concentration,
                    max_concentration=cell_data.max_concentration,
                    eig_estimator=eig_estimator,
                    eig_steps=eig_steps,
                    eig_guide_steps=eig_guide_steps,
                    eig_learning_rate=eig_learning_rate,
                    eig_guide_learning_rate=eig_guide_learning_rate,
                    eig_outer_samples=eig_outer_samples,
                    eig_inner_samples=eig_inner_samples,
                )

            run_dir = output_root / family_name / cell_line
            run_dir.mkdir(parents=True, exist_ok=True)
            literature_path = run_dir / "literature_prior.json"
            fit_summary_path = run_dir / "fit_summary.json"
            eig_summary_path = run_dir / "eig_optimization_summary.json"
            posterior_samples_path = run_dir / "posterior_samples.pt"
            posterior_predictive_path = run_dir / "posterior_predictive.pt"
            posterior_plot_path = run_dir / "posterior_predictive.png"
            prior_posterior_plot_path = run_dir / "prior_posterior_comparison.png"
            prior_posterior_positive_plot_path = run_dir / "prior_posterior_comparison_positive.png"
            eig_plot_path = run_dir / "eig_optimization.png"

            literature_payload = {
                "family": family_name,
                "cell_line": cell_line,
                "literature_report": family_prior_payload["literature_report"],
                "prior_used": family_prior_payload["prior_used"].to_dict(),
                "translated_prior": translated_prior.to_dict(),
            }
            fit_summary = {
                "family": family_name,
                "cell_line": cell_line,
                "candidate_name": family_spec.candidate_name,
                "loss_history": fit_result.loss_history,
                "posterior_summary": summarize_posterior_samples(fit_result.posterior_samples),
                "translated_prior": translated_prior.to_dict(),
                "abundance_prior_mode": (
                    "cell_line_qpcr_truncated_lognormal"
                    if family_name == "multireceptor"
                    else None
                ),
            }

            with literature_path.open("w", encoding="utf-8") as handle:
                json.dump(literature_payload, handle, indent=2)
            with fit_summary_path.open("w", encoding="utf-8") as handle:
                json.dump(fit_summary, handle, indent=2)
            with eig_summary_path.open("w", encoding="utf-8") as handle:
                json.dump(eig_result, handle, indent=2)

            torch.save(fit_result.posterior_samples, posterior_samples_path)
            torch.save(
                {
                    "grid_concentration": plot_grid.detach().cpu(),
                    "predictive_mu": predictive["mu"].detach().cpu(),
                    "predictive_y": predictive["y"].detach().cpu(),
                },
                posterior_predictive_path,
            )
            save_posterior_predictive_plot(
                observed_concentration=cell_data.bmp4_conc,
                observed_response=cell_data.x_obs,
                grid_concentration=plot_grid.detach().cpu().numpy(),
                predictive_draws=predictive["y"].detach().cpu().numpy(),
                output_path=posterior_plot_path,
                title=(
                    f"Posterior predictive BMP4 dose-response: {cell_line} / "
                    f"{_display_family_name(family_name)}"
                ),
            )
            save_prior_posterior_comparison_plot(
                translated_prior=translated_prior,
                posterior_samples=fit_result.posterior_samples,
                output_path=prior_posterior_plot_path,
                title=(
                    f"Prior vs posterior parameter comparison: {cell_line} / "
                    f"{_display_family_name(family_name)}"
                ),
                cell_lines=[cell_line],
                receptor_names=list(cell_data.receptor_names),
                kd_prior_shift=None,
            )
            save_prior_posterior_comparison_plot(
                translated_prior=translated_prior,
                posterior_samples=fit_result.posterior_samples,
                output_path=prior_posterior_positive_plot_path,
                title=(
                    f"Prior vs posterior parameter comparison on positive scale: {cell_line} / "
                    f"{_display_family_name(family_name)}"
                ),
                cell_lines=[cell_line],
                receptor_names=list(cell_data.receptor_names),
                kd_prior_shift=None,
                scale="positive",
            )
            save_eig_optimization_plot(
                eig_result["history"],
                eig_plot_path,
                title=f"Next-design EIG optimization: {cell_line} / {_display_family_name(family_name)}",
            )

            run_record = {
                "family": family_name,
                "cell_line": cell_line,
                "run_dir": str(run_dir),
                "best_design": eig_result["best_design"],
                "best_dose": eig_result.get("best_dose"),
                "best_log10_dose": eig_result.get("best_log10_dose"),
                "best_eig": eig_result["best_eig"],
                "eig_estimator": eig_result.get("estimator", eig_estimator),
                "posterior_samples_path": str(posterior_samples_path),
                "posterior_predictive_path": str(posterior_predictive_path),
                "posterior_predictive_plot": str(posterior_plot_path),
                "prior_posterior_plot": str(prior_posterior_plot_path),
                "prior_posterior_positive_plot": str(prior_posterior_positive_plot_path),
                "eig_optimization_plot": str(eig_plot_path),
                "llm_warnings": llm_warnings,
                "literature_prior_source": family_prior_payload.get("source_path"),
            }
            run_records.append(run_record)

    summary = {
        "problem_summary": problem["problem_summary"],
        "selected_cell_lines": selected_cell_lines,
        "selected_families": selected_families,
        "eig_estimator": eig_estimator,
        "runs": run_records,
        "llm_warnings": llm_warnings,
        "promisys_hyperparams_path": str(promisys_hyperparams_json)
        if promisys_hyperparams_json is not None
        else None,
        "promisys_hyperparams": resolved_promisys_hyperparams.to_dict()
        if resolved_promisys_hyperparams is not None
        else None,
    }
    summary_paths = [output_root / "run_summary.json"]
    prior_labels = sorted(
        {
            str(record["prior_output_label"])
            for record in run_records
            if record.get("prior_output_label")
        }
    )
    if prior_labels:
        summary_paths.append(output_root / f"run_summary__{'__'.join(prior_labels)}.json")
    for summary_path in summary_paths:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    summary["summary_path"] = str(summary_paths[-1])
    summary["summary_paths"] = [str(path) for path in summary_paths]
    return summary


def _run_variational_boed(
    *,
    family_name: str,
    posterior_samples: dict[str, torch.Tensor],
    receptor_names: tuple[str, ...],
    cell_line_names: tuple[str, ...],
    min_positive_concentration: float,
    max_concentration: float,
    eig_estimator: str,
    eig_steps: int,
    eig_guide_steps: int,
    eig_learning_rate: float,
    eig_guide_learning_rate: float,
    eig_outer_samples: int,
    eig_inner_samples: int | None,
) -> dict[str, Any]:
    if eig_estimator == DEFAULT_EIG_ESTIMATOR:
        raise ValueError("Variational BOED helper should not be used for the empirical estimator.")

    log10_dose_min, log10_dose_max = multireceptor_hierarchical_model.make_log10_dose_bounds(
        min_positive_concentration=min_positive_concentration,
        max_concentration=max_concentration,
    )
    context = bmp4_postfit_boed.build_postfit_context(
        family_name=family_name,
        posterior_samples=posterior_samples,
        receptor_names=receptor_names,
        cell_line_names=cell_line_names,
        log10_dose_min=log10_dose_min,
        log10_dose_max=log10_dose_max,
    )
    bmp4_postfit_boed.set_current_context(context)
    try:
        spec = bmp4_postfit_boed.build_postfit_boed_spec(
            context=context,
            estimator=eig_estimator,
            guide_training_steps=eig_guide_steps,
            num_outer_samples=eig_outer_samples,
            num_optimization_steps=eig_steps,
            design_learning_rate=eig_learning_rate,
            guide_learning_rate=eig_guide_learning_rate,
            num_inner_samples=eig_inner_samples,
        )
        backend = PyroBackend()
        validation = backend.validate(spec)
        if not validation.valid:
            raise ValueError(f"BMP4 variational BOED spec is invalid: {validation.to_dict()}")
        result = backend.optimize(spec)
        payload = bmp4_postfit_boed.format_optimization_result(result)
        payload["validation_warnings"] = [issue.to_dict() for issue in validation.warnings]
        payload["guide_ref"] = spec.guide_ref
        payload["model_ref"] = spec.model_ref
        payload["loss_ref"] = spec.loss_ref
        payload["optim_ref"] = spec.optim_ref
        payload["guide_training_steps"] = spec.compute_budget.guide_training_steps
        payload["num_outer_samples"] = spec.compute_budget.num_outer_samples
        payload["num_inner_samples"] = spec.compute_budget.num_inner_samples
        return payload
    finally:
        bmp4_postfit_boed.clear_current_context()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--problem-path", default=str(DEFAULT_PROBLEM_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cell-line", action="append", dest="cell_lines")
    parser.add_argument("--family", action="append", dest="families")
    parser.add_argument(
        "--literature-prior-json",
        action="append",
        dest="literature_prior_jsons",
        help=(
            "Path to a saved literature prior JSON. With one selected family, a plain path is accepted. "
            "For multiple families, use FAMILY=PATH and repeat the flag."
        ),
    )
    parser.add_argument("--fit-steps", type=int, default=250)
    parser.add_argument("--fit-learning-rate", type=float, default=0.02)
    parser.add_argument("--posterior-samples", type=int, default=256)
    parser.add_argument(
        "--eig-estimator",
        default=DEFAULT_EIG_ESTIMATOR,
        choices=["empirical", "posterior_eig", "marginal_eig", "vnmc_eig", "vi_eig"],
    )
    parser.add_argument("--eig-steps", type=int, default=100)
    parser.add_argument("--eig-guide-steps", type=int, default=250)
    parser.add_argument("--eig-learning-rate", type=float, default=0.05)
    parser.add_argument("--eig-guide-learning-rate", type=float, default=0.05)
    parser.add_argument("--eig-outer-samples", type=int, default=64)
    parser.add_argument("--eig-inner-samples", type=int, default=None)
    parser.add_argument(
        "--infonce-lambda",
        type=float,
        default=promisys_onestep_model.DEFAULT_INFONCE_LAMBDA,
        help="Deprecated compatibility knob; local Promisys objective now uses unregularized contrastive EIG.",
    )
    parser.add_argument(
        "--design-dist-init-std",
        type=float,
        default=promisys_onestep_model.DEFAULT_DESIGN_DIST_INIT_STD,
        help="Deprecated compatibility knob; design std is now set by --design-temperature-scale.",
    )
    parser.add_argument(
        "--design-temperature-scale",
        type=float,
        default=promisys_onestep_model.DEFAULT_DESIGN_TEMPERATURE_SCALE,
        help=(
            "Final raw-dose design-distribution std in BMP4 concentration units; "
            "1.0 means the final BOED step targets dose_std=1."
        ),
    )
    parser.add_argument(
        "--selector-temperature-final",
        type=float,
        default=promisys_onestep_model.DEFAULT_SELECTOR_TEMPERATURE_FINAL,
        help=(
            "Final softmax temperature for the cell-line selector; smaller values "
            "make the final selector probability closer to one-hot."
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--normalizer-data-dir",
        default=str(promisys_onestep_model.DEFAULT_NORMALIZER_DATA_DIR),
        help="Directory containing noised_Ls_4k.npy and sim_x_fat_Rs_noised_Ls_4k.npy.",
    )
    parser.add_argument("--receptor-noise-log-sd", type=float, default=0.05)
    parser.add_argument("--snpe-steps", type=int, default=500)
    parser.add_argument("--snpe-simulations", type=int, default=512)
    parser.add_argument("--snpe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mcmc-warmup", type=int, default=200)
    parser.add_argument("--mcmc-samples", type=int, default=256)
    parser.add_argument(
        "--promisys-hyperparams-json",
        default=None,
        help="Optional JSON config overriding promisys SNPE, NSF, BOED, and MCMC knobs.",
    )
    args = parser.parse_args(argv)

    summary = run_bmp4_gradient_example(
        provider=args.provider,
        model=args.model,
        problem_path=args.problem_path,
        output_dir=args.output_dir,
        cell_lines=args.cell_lines,
        families=args.families,
        literature_prior_jsons=args.literature_prior_jsons,
        fit_steps=args.fit_steps,
        fit_learning_rate=args.fit_learning_rate,
        posterior_samples=args.posterior_samples,
        eig_estimator=args.eig_estimator,
        eig_steps=args.eig_steps,
        eig_guide_steps=args.eig_guide_steps,
        eig_learning_rate=args.eig_learning_rate,
        eig_guide_learning_rate=args.eig_guide_learning_rate,
        eig_outer_samples=args.eig_outer_samples,
        eig_inner_samples=args.eig_inner_samples,
        infonce_lambda=args.infonce_lambda,
        design_dist_init_std=args.design_dist_init_std,
        design_temperature_scale=args.design_temperature_scale,
        selector_temperature_final=args.selector_temperature_final,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        normalizer_data_dir=args.normalizer_data_dir,
        receptor_noise_log_sd=args.receptor_noise_log_sd,
        snpe_steps=args.snpe_steps,
        snpe_simulations=args.snpe_simulations,
        snpe_learning_rate=args.snpe_learning_rate,
        mcmc_warmup=args.mcmc_warmup,
        mcmc_samples=args.mcmc_samples,
        promisys_hyperparams_json=args.promisys_hyperparams_json,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _build_family_priors(
    *,
    problem: dict[str, Any],
    registry: dict[str, Any],
    selected_families: list[str],
    receptor_names: tuple[str, ...],
    llm: LLMClient,
    prior_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    shared: dict[str, dict[str, Any]] = {}
    prior_overrides = dict(prior_overrides or {})
    local_corpus_dir = REPO_ROOT / problem["metadata"]["local_corpus_dir"]
    source_bundle = SourceBundle(
        extra=[("bmp4_local", LocalCorpusClient(corpus_dir=local_corpus_dir, source_name="bmp4_local"))]
    )

    for family_name in selected_families:
        if family_name in prior_overrides:
            shared[family_name] = dict(prior_overrides[family_name])
            continue
        family_spec = registry[family_name]
        if not family_spec.literature_parameters:
            shared[family_name] = {
                "literature_report": {
                    "source": "not_applicable",
                    "reasoning": (
                        "This model family uses an SNPE posterior fitted from BMP4 data "
                        "as its starting prior instead of literature-synthesized priors."
                    ),
                },
                "prior_object": AugmentedPrior(),
                "source_path": None,
            }
            continue
        literature_module = LiteratureSearchModule(
            sources=source_bundle,
            llm=llm,
            token_budget=TokenBudget(),
            config=LiteratureSearchConfig(
                available_backends=("PyroVI", "LFIAX"),
                prior_only=True,
                include_design_reasoning=False,
                include_backend_reasoning=False,
                extraction=ExtractionConfig(prior_synthesis_mode=True),
                reasoning=ReasoningConfig(min_sources_for_llm=1),
                verbose=True,
            ),
        )
        metadata = family_spec.build_metadata(receptor_names)
        simulator = SimpleSimulator(
            fn=lambda theta, xi: 0.0,
            metadata=metadata,
            is_explicit=True,
            is_differentiable=True,
            name=family_name,
        )
        agent = BOEDAgent(
            simulator=simulator,
            design_distribution={
                problem["shared"]["design_variables"][0]["name"]: {
                    "lower": problem["shared"]["design_variables"][0]["lower"],
                    "upper": problem["shared"]["design_variables"][0]["upper"],
                }
            },
            problem_description=family_spec.build_problem_description(
                problem["problem_summary"],
                receptor_names,
            ),
            use_literature=True,
            literature_module=literature_module,
        )
        dry_run = agent.run(dry_run=True)
        shared[family_name] = {
            "literature_report": (
                None if dry_run.literature_report is None else dry_run.literature_report.to_dict()
            ),
            "prior_object": dry_run.prior_used,
            "source_path": None,
        }
    return {
        family_name: {
            "literature_report": payload["literature_report"],
            "prior_used": payload["prior_object"],
            "source_path": payload.get("source_path"),
        }
        for family_name, payload in shared.items()
    }


def _load_problem_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    with bundle_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not Path(payload["data"]["observed_data_ref"]).is_absolute():
        payload["data"]["observed_data_ref"] = str(REPO_ROOT / payload["data"]["observed_data_ref"])
    return payload


def _resolve_llm_client(
    *,
    llm_client: LLMClient | None,
    provider: str | None,
    model: str | None,
) -> tuple[LLMClient, list[str]]:
    if llm_client is not None:
        return llm_client, []
    resolved, warnings = build_literature_llm_client(provider, model)
    return resolved or NullLLMClient(), warnings


def _resolve_literature_prior_overrides(
    raw_values: list[str] | None,
    *,
    selected_families: list[str],
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    if not raw_values:
        return overrides

    multiple_families = len(selected_families) != 1
    for raw_value in raw_values:
        family_name: str | None = None
        path_text = raw_value
        if "=" in raw_value:
            maybe_family, maybe_path = raw_value.split("=", 1)
            if maybe_family:
                family_name = maybe_family.strip()
                path_text = maybe_path.strip()

        if family_name is None:
            if multiple_families:
                raise ValueError(
                    "When multiple model families are selected, --literature-prior-json must use FAMILY=PATH."
                )
            family_name = selected_families[0]

        if family_name not in selected_families:
            raise ValueError(
                f"Literature prior override provided for unselected family {family_name!r}."
            )
        if family_name in overrides:
            raise ValueError(f"Duplicate literature prior override for family {family_name!r}.")

        path = Path(path_text)
        if not path.is_absolute():
            path = REPO_ROOT / path
        overrides[family_name] = _load_literature_prior_override(
            path,
            expected_family=family_name,
        )
    return overrides


def _load_literature_prior_override(
    path: str | Path,
    *,
    expected_family: str,
) -> dict[str, Any]:
    prior_path = Path(path)
    with prior_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    family_name = payload.get("family")
    family_reused = isinstance(family_name, str) and family_name != expected_family

    if isinstance(payload.get("prior_used"), dict):
        prior_used = _augmented_prior_from_augmented_payload(payload["prior_used"])
        literature_report = payload.get("literature_report")
        if isinstance(literature_report, dict):
            literature_report = dict(literature_report)
            if family_reused:
                literature_report.setdefault("reported_family", family_name)
                literature_report.setdefault("reused_for_family", expected_family)
                literature_report.setdefault("family_reused_across_models", True)
    elif isinstance(payload.get("priors"), dict):
        prior_used = _augmented_prior_from_codex_payload(payload)
        literature_report = {
            "source": "external_prior_json",
            "path": str(prior_path),
            "family": expected_family,
            "reported_family": family_name,
            "family_reused_across_models": family_reused,
            "notes": list(payload.get("notes") or []),
            "corpus_scope": payload.get("corpus_scope"),
        }
    else:
        raise ValueError(
            f"Unsupported literature prior JSON format in {prior_path}."
        )

    return {
        "literature_report": literature_report,
        "prior_object": prior_used,
        "source_path": str(prior_path),
    }


def _augmented_prior_from_augmented_payload(payload: dict[str, Any]) -> AugmentedPrior:
    distributions: dict[str, DistributionSpec] = {}
    raw_distributions = payload.get("distributions") or {}
    if not isinstance(raw_distributions, dict):
        raise ValueError("Saved prior_used payload must contain a distributions mapping.")
    for name, spec in raw_distributions.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Saved prior distribution for {name!r} must be a mapping.")
        distributions[str(name)] = DistributionSpec(
            name=spec.get("name"),
            params=dict(spec.get("params") or {}),
            source=str(spec.get("source", "literature_json")),
            reasoning=str(spec.get("reasoning", "")),
            cited_papers=list(spec.get("cited_papers") or []),
            fallback=bool(spec.get("fallback", False)),
        )
    return AugmentedPrior(
        distributions=distributions,
        warnings=list(payload.get("warnings") or []),
        notes=list(payload.get("notes") or []),
    )


def _augmented_prior_from_codex_payload(payload: dict[str, Any]) -> AugmentedPrior:
    distributions: dict[str, DistributionSpec] = {}
    priors = payload.get("priors") or {}
    if not isinstance(priors, dict):
        raise ValueError("Codex prior JSON must contain a priors mapping.")
    for name, spec in priors.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Codex prior for {name!r} must be a mapping.")
        distributions[str(name)] = DistributionSpec(
            name=spec.get("distribution"),
            params=dict(spec.get("params") or {}),
            source="literature_json",
            reasoning=str(spec.get("reasoning", "")),
            cited_papers=list(spec.get("cited_papers") or []),
            fallback=bool(spec.get("fallback", False)),
        )
    return AugmentedPrior(
        distributions=distributions,
        notes=list(payload.get("notes") or []),
    )


def _joint_run_label(cell_lines: tuple[str, ...]) -> str:
    if not cell_lines:
        return "joint"
    return "joint__" + "__".join(cell_lines)


def _prior_output_label(family_prior_payload: dict[str, Any]) -> str | None:
    source_path = family_prior_payload.get("source_path")
    if not source_path:
        return None
    path = Path(str(source_path))
    stem = _slugify_label(path.stem)
    if stem in {"literature_prior", "prior"}:
        context = _prior_source_context_label(path)
        if context:
            return _slugify_label(f"{context}_{stem}")
    return stem


def _prior_source_context_label(path: Path) -> str | None:
    parts = list(path.parts)
    if "bmp4_gradient" in parts:
        index = parts.index("bmp4_gradient")
        if index + 1 < len(parts) and parts[index + 1] != "priors":
            return parts[index + 1]
    if path.parent.name.startswith("joint__") and path.parent.parent.name:
        return path.parent.parent.name
    if path.parent.name and path.parent.name not in {"priors", ".", ""}:
        return path.parent.name
    return None


def _annotate_promisys_summaries(
    *,
    run_dir: Path,
    prior_output_label: str | None,
    literature_prior_source: str | None,
) -> None:
    for filename in ("fit_summary.json", "snpe_fit_summary.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["prior_output_label"] = prior_output_label
        payload["literature_prior_source"] = literature_prior_source
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _run_dir_for_prior(
    *,
    output_root: Path,
    family_name: str,
    run_label: str,
    prior_output_label: str | None,
) -> Path:
    if prior_output_label:
        return output_root / family_name / prior_output_label / run_label
    return output_root / family_name / run_label


def _slugify_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "prior"


def _display_family_name(family_name: str) -> str:
    return {
        "hill": "Hill dose-response model",
        "multireceptor": "Multireceptor dose-response model",
        "multireceptor_hierarchical": "Hierarchical multireceptor BMP4 model",
        "promisys_onestep": "Promisys one-step BMP4 model",
        "promisys_twostep": "Promisys two-step BMP4 model",
    }.get(family_name, family_name.replace("_", " "))


if __name__ == "__main__":
    raise SystemExit(main())
