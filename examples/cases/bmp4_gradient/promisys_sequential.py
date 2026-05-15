"""Sequential retrospective BOED workflow for BMP4 Promisys models."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

from examples.cases.bmp4_gradient import promisys_onestep
from examples.cases.bmp4_gradient import promisys_twostep
from examples.cases.bmp4_gradient.plotting import (
    save_joint_posterior_predictive_plot,
    save_sequential_acquisition_plot,
)
from examples.cases.bmp4_gradient.promisys_hyperparams import (
    PromisysHyperparams,
    coerce_promisys_hyperparams,
    effective_promisys_hyperparams,
)


try:  # pragma: no cover - exercised by tests/examples when numpy is installed
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


FAMILY_MODULES = {
    "promisys_onestep": promisys_onestep,
    "promisys_twostep": promisys_twostep,
}


def run_promisys_sequential_workflow(
    *,
    family_name: str,
    joint_data: Any,
    run_dir: str | Path,
    prior_mode: str = "default",
    literature_prior: Any | None = None,
    normalizer_data_dir: str | Path | None = None,
    receptor_noise_log_sd: float = 0.05,
    likelihood_steps: int = 500,
    likelihood_learning_rate: float = 1e-3,
    posterior_sample_count: int = 256,
    eig_steps: int = 100,
    eig_outer_samples: int = 64,
    eig_inner_samples: int | None = None,
    eig_learning_rate: float = 0.05,
    infonce_lambda: float | None = None,
    design_dist_init_std: float | None = None,
    design_temperature_scale: float | None = None,
    selector_temperature_final: float | None = None,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 0.0,
    mcmc_warmup: int = 200,
    mcmc_samples: int = 256,
    rounds: int = 1,
    batch_size: int = 1,
    seed: int = 0,
    lfiax_root: str | Path | None = None,
    promisys_hyperparams: PromisysHyperparams | dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Run greedy retrospective sequential BOED over the observed BMP4 grid."""
    if np is None:
        raise RuntimeError("The BMP4 sequential Promisys workflow requires numpy.")
    import torch

    if int(rounds) < 0:
        raise ValueError("rounds must be nonnegative.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1.")
    module = _family_module(family_name)
    module._validate_joint_data(joint_data)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))

    normalizer_dir = Path(normalizer_data_dir or module.DEFAULT_NORMALIZER_DATA_DIR)
    module._require_promisys(lfiax_root if lfiax_root is not None else module.DEFAULT_LFIAX_ROOT)
    normalizer = module.Bmp4Normalizer.from_source_dir(normalizer_dir)
    effective = _resolve_effective_settings(
        module=module,
        hyperparams=promisys_hyperparams,
        likelihood_steps=likelihood_steps,
        likelihood_learning_rate=likelihood_learning_rate,
        posterior_sample_count=posterior_sample_count,
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
    )

    current_samples, theta_prior = initialize_promisys_prior_samples(
        family_name=family_name,
        joint_data=joint_data,
        sample_count=int(effective["posterior_sample_count"]),
        prior_mode=prior_mode,
        literature_prior=literature_prior,
        seed=seed,
    )
    initial_samples = {
        cell_line: np.asarray(samples, dtype=np.float32).copy()
        for cell_line, samples in current_samples.items()
    }
    collected: dict[str, dict[str, list[Any]]] = {
        str(cell_line): {"dose_index": [], "x_obs": [], "x_obs_norm": [], "bmp4_conc": [], "bmp4_conc_norm": []}
        for cell_line in joint_data.cell_lines
    }
    used_designs: set[tuple[str, int]] = set()
    trace: list[dict[str, Any]] = []
    likelihood_states: list[Any] = []
    all_boed_history: list[dict[str, Any]] = []
    total_acquisitions = max(int(rounds), 0) * max(int(batch_size), 0)

    for acquisition_index in range(total_acquisitions):
        round_index = acquisition_index // max(int(batch_size), 1)
        batch_index = acquisition_index % max(int(batch_size), 1)
        step_dir = run_path / f"step_{acquisition_index + 1:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        likelihood_state, eig_result = module._run_joint_multicontext_boed(
            joint_data=joint_data,
            normalizer=normalizer,
            snpe_samples=current_samples,
            steps=int(effective["likelihood_steps"]),
            outer_samples=int(effective["eig_outer_samples"]),
            inner_samples=int(effective["eig_inner_samples"]),
            flow_learning_rate=float(effective["likelihood_learning_rate"]),
            design_learning_rate=float(effective["eig_learning_rate"]),
            infonce_lambda=float(effective["infonce_lambda"]),
            design_dist_init_std=float(effective["design_dist_init_std"]),
            design_temperature_scale=float(effective["design_temperature_scale"]),
            selector_temperature_final=float(effective["selector_temperature_final"]),
            receptor_noise_log_sd=float(receptor_noise_log_sd),
            flow_config=effective["flow_config"],
            rng=rng,
            seed=seed + 1000 + acquisition_index,
            lfiax_root=lfiax_root if lfiax_root is not None else module.DEFAULT_LFIAX_ROOT,
            early_stopping_patience=effective["early_stopping_patience"],
            early_stopping_min_delta=effective["early_stopping_min_delta"],
        )
        eig_result[f"eig_steps_ignored_for_{family_name}"] = True
        eig_result["eig_steps_requested"] = int(eig_steps)

        snapped = select_snapped_design(
            joint_data=joint_data,
            eig_result=eig_result,
            used_designs=used_designs,
        )
        cell_line = str(snapped["cell_line"])
        cell_index = int(snapped["cell_line_index"])
        dose_index = int(snapped["dose_index"])
        used_designs.add((cell_line, dose_index))

        collected[cell_line]["dose_index"].append(dose_index)
        collected[cell_line]["x_obs"].append(float(joint_data.x_obs[cell_index, dose_index]))
        collected[cell_line]["x_obs_norm"].append(float(joint_data.x_obs_norm[cell_index, dose_index]))
        collected[cell_line]["bmp4_conc"].append(float(joint_data.bmp4_conc[cell_index, dose_index]))
        collected[cell_line]["bmp4_conc_norm"].append(float(joint_data.bmp4_conc_norm[cell_index, dose_index]))

        current_samples = _update_samples_from_collected_observations(
            module=module,
            joint_data=joint_data,
            likelihood_state=likelihood_state,
            current_samples=current_samples,
            collected=collected,
            warmup=int(effective["mcmc_warmup"]),
            sample_count=int(effective["mcmc_samples"]),
            proposal_scale=float(effective["mcmc_proposal_scale"]),
            prior_std_floor=float(effective["mcmc_prior_std_floor"]),
            rng=rng,
        )
        likelihood_states.append(likelihood_state)
        all_boed_history.extend(likelihood_state.history)

        step_record = {
            "acquisition": int(acquisition_index + 1),
            "round": int(round_index + 1),
            "batch_index": int(batch_index + 1),
            "prior_mode": prior_mode,
            "boed": {
                "best_cell_line": eig_result["best_cell_line"],
                "best_bmp4_norm_mu": eig_result["best_bmp4_norm_mu"],
                "best_dose_mu": eig_result["best_dose_mu"],
                "best_eig": eig_result["best_eig"],
                "best_per_cell": eig_result.get("best_per_cell", []),
            },
            "snapped_design": snapped,
            "observation": {
                "x_obs": float(joint_data.x_obs[cell_index, dose_index]),
                "x_obs_norm": float(joint_data.x_obs_norm[cell_index, dose_index]),
            },
            "collected_count_by_cell": {
                name: len(payload["dose_index"])
                for name, payload in collected.items()
            },
            "used_designs": [
                {"cell_line": name, "dose_index": int(index)}
                for name, index in sorted(used_designs)
            ],
            "artifacts": _write_step_artifacts(
                module=module,
                step_dir=step_dir,
                normalizer=normalizer,
                joint_data=joint_data,
                likelihood_state=likelihood_state,
                eig_result=eig_result,
                posterior_samples=current_samples,
                theta_prior=theta_prior,
                metadata={
                    "family": family_name,
                    "prior_mode": prior_mode,
                    "acquisition": int(acquisition_index + 1),
                    "snapped_design": snapped,
                    "promisys_hyperparams": effective["promisys_hyperparams"],
                },
            ),
        }
        trace.append(step_record)

    final_likelihood_state = likelihood_states[-1] if likelihood_states else None
    posterior_predictive = None
    if final_likelihood_state is not None:
        posterior_predictive = module._posterior_predictive_from_likelihood(
            joint_data=joint_data,
            normalizer=normalizer,
            mcmc_samples_by_cell_line=current_samples,
            grid_points=128,
            rng=rng,
            lfiax_root=lfiax_root if lfiax_root is not None else module.DEFAULT_LFIAX_ROOT,
        )

    artifacts = _write_final_artifacts(
        module=module,
        run_path=run_path,
        normalizer=normalizer,
        joint_data=joint_data,
        initial_samples=initial_samples,
        posterior_samples=current_samples,
        theta_prior=theta_prior,
        trace=trace,
        posterior_predictive=posterior_predictive,
        boed_history=all_boed_history,
        metadata={
            "family": family_name,
            "prior_mode": prior_mode,
            "rounds": int(rounds),
            "batch_size": int(batch_size),
            "normalizer_data_dir": str(normalizer_dir),
            "receptor_noise_log_sd": float(receptor_noise_log_sd),
            "requested_promisys_hyperparams": effective["requested_promisys_hyperparams"],
            "promisys_hyperparams": effective["promisys_hyperparams"],
        },
    )

    return {
        "family": family_name,
        "mode": "sequential_retrospective",
        "prior_mode": prior_mode,
        "cell_lines": list(joint_data.cell_lines),
        "run_dir": str(run_path),
        "rounds": int(rounds),
        "batch_size": int(batch_size),
        "acquisition_count": len(trace),
        "trace_path": artifacts["trace_path"],
        "fit_summary_path": artifacts["fit_summary_path"],
        "posterior_samples_path": artifacts["posterior_samples_path"],
        "mcmc_posterior_samples_path": artifacts["mcmc_posterior_samples_path"],
        "posterior_predictive_path": artifacts.get("posterior_predictive_path"),
        "posterior_predictive_plot": artifacts.get("posterior_predictive_plot"),
        "eig_optimization_plot": artifacts.get("eig_optimization_plot"),
        "final_posterior_summary": module._summarize_samples(current_samples),
        "collected_count_by_cell": {
            name: len(payload["dose_index"])
            for name, payload in collected.items()
        },
        "used_design_count": len(used_designs),
        "theta_prior": theta_prior,
        "promisys_hyperparams": effective["promisys_hyperparams"],
    }


def initialize_promisys_prior_samples(
    *,
    family_name: str,
    joint_data: Any,
    sample_count: int,
    prior_mode: str = "default",
    literature_prior: Any | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw initial theta samples directly from a Promisys theta-prior config."""
    if np is None:
        raise RuntimeError("BMP4 Promisys prior sampling requires numpy.")
    if prior_mode not in {"default", "literature"}:
        raise ValueError("prior_mode must be 'default' or 'literature'.")
    module = _family_module(family_name)
    build_prior = (
        module._build_onestep_theta_prior
        if family_name == "promisys_onestep"
        else module._build_twostep_theta_prior
    )
    theta_prior = build_prior(literature_prior if prior_mode == "literature" else None)
    rng = np.random.default_rng(int(seed))
    samples: dict[str, Any] = {}
    for cell_line in joint_data.cell_lines:
        draws = [
            module.sample_theta_norm_from_prior_config(rng, theta_prior)
            for _ in range(max(int(sample_count), 1))
        ]
        samples[str(cell_line)] = np.asarray(draws, dtype=np.float32)
    return samples, theta_prior


def select_snapped_design(
    *,
    joint_data: Any,
    eig_result: dict[str, Any],
    used_designs: set[tuple[str, int]],
) -> dict[str, Any]:
    """Pick the highest-utility cell line with remaining measured designs and snap dose."""
    per_cell = list(eig_result.get("best_per_cell") or [])
    if not per_cell:
        per_cell = [eig_result.get("final_next_experiment") or eig_result]
    per_cell.sort(key=lambda item: float(item.get("utility", item.get("best_eig", -float("inf")))), reverse=True)
    cell_index_by_name = {str(name): index for index, name in enumerate(joint_data.cell_lines)}
    for proposal in per_cell:
        cell_line = str(proposal.get("cell_line") or proposal.get("best_cell_line"))
        if cell_line not in cell_index_by_name:
            continue
        cell_index = cell_index_by_name[cell_line]
        if all((cell_line, index) in used_designs for index in range(joint_data.bmp4_conc.shape[1])):
            continue
        return snap_to_nearest_unused_design(
            joint_data=joint_data,
            cell_line_index=cell_index,
            proposed_bmp4_norm=float(proposal.get("bmp4_norm_mu", proposal.get("best_bmp4_norm_mu"))),
            proposed_log10_dose=float(proposal.get("log10_dose", proposal.get("best_log10_dose", 0.0))),
            used_designs=used_designs,
            utility=float(proposal.get("utility", proposal.get("best_eig", float("nan")))),
        )
    raise RuntimeError("No unused BMP4 designs remain for the selected cell lines.")


def snap_to_nearest_unused_design(
    *,
    joint_data: Any,
    cell_line_index: int,
    proposed_bmp4_norm: float,
    proposed_log10_dose: float,
    used_designs: set[tuple[str, int]],
    utility: float | None = None,
) -> dict[str, Any]:
    """Snap a continuous normalized BMP4 design to the nearest unused observed dose."""
    if np is None:
        raise RuntimeError("BMP4 design snapping requires numpy.")
    cell_line = str(joint_data.cell_lines[int(cell_line_index)])
    norm_values = np.asarray(joint_data.bmp4_conc_norm[int(cell_line_index)], dtype=np.float64)
    raw_values = np.asarray(joint_data.bmp4_conc[int(cell_line_index)], dtype=np.float64)
    candidates: list[tuple[float, float, int]] = []
    for dose_index, norm_value in enumerate(norm_values):
        if (cell_line, int(dose_index)) in used_designs:
            continue
        raw = max(float(raw_values[dose_index]), 1e-30)
        candidates.append(
            (
                abs(float(norm_value) - float(proposed_bmp4_norm)),
                abs(math.log10(raw) - float(proposed_log10_dose)),
                int(dose_index),
            )
        )
    if not candidates:
        raise RuntimeError(f"No unused BMP4 designs remain for {cell_line}.")
    _, _, dose_index = min(candidates)
    dose = float(raw_values[dose_index])
    norm = float(norm_values[dose_index])
    return {
        "cell_line": cell_line,
        "cell_line_index": int(cell_line_index),
        "dose_index": int(dose_index),
        "dose": dose,
        "log10_dose": float(math.log10(max(dose, 1e-30))),
        "bmp4_norm_design": norm,
        "proposed_bmp4_norm": float(proposed_bmp4_norm),
        "proposed_log10_dose": float(proposed_log10_dose),
        "norm_distance": abs(norm - float(proposed_bmp4_norm)),
        "log10_distance": abs(math.log10(max(dose, 1e-30)) - float(proposed_log10_dose)),
        "utility": None if utility is None else float(utility),
    }


def _update_samples_from_collected_observations(
    *,
    module: Any,
    joint_data: Any,
    likelihood_state: Any,
    current_samples: dict[str, Any],
    collected: dict[str, dict[str, list[Any]]],
    warmup: int,
    sample_count: int,
    proposal_scale: float,
    prior_std_floor: float,
    rng: Any,
) -> dict[str, Any]:
    updated: dict[str, Any] = {}
    for cell_index, cell_line_value in enumerate(joint_data.cell_lines):
        cell_line = str(cell_line_value)
        payload = collected[cell_line]
        if not payload["dose_index"]:
            updated[cell_line] = np.asarray(current_samples[cell_line], dtype=np.float32)
            continue
        updated[cell_line] = module._run_cell_line_mcmc(
            likelihood_state=likelihood_state,
            prior_samples=np.asarray(current_samples[cell_line], dtype=np.float32),
            y_norm=np.asarray(payload["x_obs_norm"], dtype=np.float32),
            r_norm=np.asarray(joint_data.Rs_norm[cell_index], dtype=np.float32),
            xi_norm=np.asarray(payload["bmp4_conc_norm"], dtype=np.float32),
            warmup=warmup,
            sample_count=sample_count,
            proposal_scale=proposal_scale,
            prior_std_floor=prior_std_floor,
            rng=rng,
            label=f"{cell_line} sequential",
        )
    return updated


def _resolve_effective_settings(
    *,
    module: Any,
    hyperparams: PromisysHyperparams | dict[str, Any] | str | Path | None,
    likelihood_steps: int,
    likelihood_learning_rate: float,
    posterior_sample_count: int,
    eig_outer_samples: int,
    eig_inner_samples: int | None,
    eig_learning_rate: float,
    infonce_lambda: float | None,
    design_dist_init_std: float | None,
    design_temperature_scale: float | None,
    selector_temperature_final: float | None,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    mcmc_warmup: int,
    mcmc_samples: int,
) -> dict[str, Any]:
    resolved = coerce_promisys_hyperparams(hyperparams)
    posterior_hp = resolved.posterior_net if resolved is not None else None
    objective_hp = resolved.objective if resolved is not None else None
    mcmc_hp = resolved.mcmc if resolved is not None else None
    effective_posterior_sample_count = int(
        posterior_hp.posterior_samples
        if posterior_hp is not None and posterior_hp.posterior_samples is not None
        else posterior_sample_count
    )
    effective_likelihood_steps = int(
        objective_hp.fit_steps
        if objective_hp is not None and objective_hp.fit_steps is not None
        else likelihood_steps
    )
    effective_likelihood_learning_rate = float(
        objective_hp.flow_learning_rate
        if objective_hp is not None and objective_hp.flow_learning_rate is not None
        else likelihood_learning_rate
    )
    effective_eig_outer_samples = int(
        objective_hp.eig_outer_samples
        if objective_hp is not None and objective_hp.eig_outer_samples is not None
        else eig_outer_samples
    )
    effective_eig_inner_samples = int(
        objective_hp.eig_inner_samples
        if objective_hp is not None and objective_hp.eig_inner_samples is not None
        else (eig_inner_samples or module.DEFAULT_INFONCE_NEGATIVES)
    )
    effective_eig_learning_rate = float(
        objective_hp.design_learning_rate
        if objective_hp is not None and objective_hp.design_learning_rate is not None
        else eig_learning_rate
    )
    effective_infonce_lambda = float(
        objective_hp.infonce_lambda
        if objective_hp is not None and objective_hp.infonce_lambda is not None
        else (module.DEFAULT_INFONCE_LAMBDA if infonce_lambda is None else infonce_lambda)
    )
    effective_design_dist_init_std = float(
        objective_hp.design_dist_init_std
        if objective_hp is not None and objective_hp.design_dist_init_std is not None
        else (module.DEFAULT_DESIGN_DIST_INIT_STD if design_dist_init_std is None else design_dist_init_std)
    )
    effective_design_temperature_scale = float(
        objective_hp.design_temperature_scale
        if objective_hp is not None and objective_hp.design_temperature_scale is not None
        else (
            module.DEFAULT_DESIGN_TEMPERATURE_SCALE
            if design_temperature_scale is None
            else design_temperature_scale
        )
    )
    effective_selector_temperature_final = float(
        objective_hp.selector_temperature_final
        if objective_hp is not None and objective_hp.selector_temperature_final is not None
        else (
            module.DEFAULT_SELECTOR_TEMPERATURE_FINAL
            if selector_temperature_final is None
            else selector_temperature_final
        )
    )
    effective_early_stopping_patience = (
        int(objective_hp.early_stopping_patience)
        if objective_hp is not None and objective_hp.early_stopping_patience is not None
        else (None if early_stopping_patience is None else int(early_stopping_patience))
    )
    effective_early_stopping_min_delta = float(
        objective_hp.early_stopping_min_delta
        if objective_hp is not None and objective_hp.early_stopping_min_delta is not None
        else early_stopping_min_delta
    )
    effective_mcmc_warmup = int(
        mcmc_hp.warmup if mcmc_hp is not None and mcmc_hp.warmup is not None else mcmc_warmup
    )
    effective_mcmc_samples = int(
        mcmc_hp.samples if mcmc_hp is not None and mcmc_hp.samples is not None else mcmc_samples
    )
    effective_mcmc_proposal_scale = float(
        mcmc_hp.proposal_scale
        if mcmc_hp is not None and mcmc_hp.proposal_scale is not None
        else module.DEFAULT_MCMC_PROPOSAL_SCALE
    )
    effective_mcmc_prior_std_floor = float(
        mcmc_hp.prior_std_floor
        if mcmc_hp is not None and mcmc_hp.prior_std_floor is not None
        else module.DEFAULT_MCMC_PRIOR_STD_FLOOR
    )
    flow_config = resolved.flow_config(module.JAX_FLOW_CONFIG) if resolved is not None else dict(module.JAX_FLOW_CONFIG)
    effective_hyperparams = effective_promisys_hyperparams(
        base_flow_config=module.JAX_FLOW_CONFIG,
        hyperparams=resolved,
        snpe_steps=0,
        snpe_simulations=0,
        snpe_learning_rate=0.0,
        posterior_sample_count=effective_posterior_sample_count,
        likelihood_steps=effective_likelihood_steps,
        likelihood_learning_rate=effective_likelihood_learning_rate,
        eig_outer_samples=effective_eig_outer_samples,
        eig_inner_samples=effective_eig_inner_samples,
        eig_learning_rate=effective_eig_learning_rate,
        infonce_lambda=effective_infonce_lambda,
        design_dist_init_std=effective_design_dist_init_std,
        design_temperature_scale=effective_design_temperature_scale,
        selector_temperature_final=effective_selector_temperature_final,
        early_stopping_patience=effective_early_stopping_patience,
        early_stopping_min_delta=effective_early_stopping_min_delta,
        mcmc_warmup=effective_mcmc_warmup,
        mcmc_samples=effective_mcmc_samples,
        mcmc_proposal_scale=effective_mcmc_proposal_scale,
        mcmc_prior_std_floor=effective_mcmc_prior_std_floor,
        posterior_hidden_dim=(
            posterior_hp.hidden_dim
            if posterior_hp is not None and posterior_hp.hidden_dim is not None
            else module.DEFAULT_POSTERIOR_NET_HIDDEN_DIM
        ),
        posterior_layers=(
            posterior_hp.layers
            if posterior_hp is not None and posterior_hp.layers is not None
            else module.DEFAULT_POSTERIOR_NET_LAYERS
        ),
        posterior_activation=(
            posterior_hp.activation
            if posterior_hp is not None and posterior_hp.activation is not None
            else module.DEFAULT_POSTERIOR_NET_ACTIVATION
        ),
        posterior_batch_size=(
            posterior_hp.batch_size
            if posterior_hp is not None and posterior_hp.batch_size is not None
            else module.DEFAULT_POSTERIOR_NET_BATCH_SIZE
        ),
    )
    effective_hyperparams["posterior_net"]["source"] = "direct_theta_prior_samples"
    return {
        "posterior_sample_count": effective_posterior_sample_count,
        "likelihood_steps": effective_likelihood_steps,
        "likelihood_learning_rate": effective_likelihood_learning_rate,
        "eig_outer_samples": effective_eig_outer_samples,
        "eig_inner_samples": effective_eig_inner_samples,
        "eig_learning_rate": effective_eig_learning_rate,
        "infonce_lambda": effective_infonce_lambda,
        "design_dist_init_std": effective_design_dist_init_std,
        "design_temperature_scale": effective_design_temperature_scale,
        "selector_temperature_final": effective_selector_temperature_final,
        "early_stopping_patience": effective_early_stopping_patience,
        "early_stopping_min_delta": effective_early_stopping_min_delta,
        "mcmc_warmup": effective_mcmc_warmup,
        "mcmc_samples": effective_mcmc_samples,
        "mcmc_proposal_scale": effective_mcmc_proposal_scale,
        "mcmc_prior_std_floor": effective_mcmc_prior_std_floor,
        "flow_config": flow_config,
        "requested_promisys_hyperparams": resolved.to_dict() if resolved is not None else None,
        "promisys_hyperparams": effective_hyperparams,
    }


def _write_step_artifacts(
    *,
    module: Any,
    step_dir: Path,
    normalizer: Any,
    joint_data: Any,
    likelihood_state: Any,
    eig_result: dict[str, Any],
    posterior_samples: dict[str, Any],
    theta_prior: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, str]:
    import torch

    eig_path = step_dir / "eig_optimization_summary.json"
    mcmc_path = step_dir / "mcmc_posterior_samples.pt"
    checkpoint_path = step_dir / "likelihood_checkpoint.pkl"
    _write_json(eig_path, eig_result)
    torch.save(module._samples_payload(posterior_samples, theta_prior=theta_prior), mcmc_path)
    _write_likelihood_checkpoint(
        module=module,
        path=checkpoint_path,
        normalizer=normalizer,
        joint_data=joint_data,
        likelihood_state=likelihood_state,
        theta_prior=theta_prior,
        metadata=metadata,
    )
    return {
        "eig_summary_path": str(eig_path),
        "mcmc_posterior_samples_path": str(mcmc_path),
        "likelihood_checkpoint": str(checkpoint_path),
    }


def _write_final_artifacts(
    *,
    module: Any,
    run_path: Path,
    normalizer: Any,
    joint_data: Any,
    initial_samples: dict[str, Any],
    posterior_samples: dict[str, Any],
    theta_prior: dict[str, Any],
    trace: list[dict[str, Any]],
    posterior_predictive: dict[str, Any] | None,
    boed_history: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, str]:
    import torch

    initial_path = run_path / "initial_prior_samples.pt"
    posterior_path = run_path / "posterior_samples.pt"
    mcmc_path = run_path / "mcmc_posterior_samples.pt"
    trace_path = run_path / "sequential_trace.json"
    fit_summary_path = run_path / "fit_summary.json"
    posterior_predictive_path = run_path / "posterior_predictive.pt"
    posterior_plot_path = run_path / "posterior_predictive.png"
    eig_plot_path = run_path / "eig_optimization.png"
    prior_plot_path = run_path / "prior_posterior_comparison.png"
    prior_positive_plot_path = run_path / "prior_posterior_comparison_positive.png"

    torch.save(module._samples_payload(initial_samples, theta_prior=theta_prior), initial_path)
    posterior_payload = module._samples_payload(posterior_samples, theta_prior=theta_prior)
    torch.save(posterior_payload, posterior_path)
    torch.save(posterior_payload, mcmc_path)
    _write_json(trace_path, {"trace": trace})

    artifacts = {
        "initial_prior_samples_path": str(initial_path),
        "posterior_samples_path": str(posterior_path),
        "mcmc_posterior_samples_path": str(mcmc_path),
        "trace_path": str(trace_path),
        "fit_summary_path": str(fit_summary_path),
    }
    module._save_theta_comparison_plot(
        initial_samples,
        posterior_samples,
        prior_plot_path,
        positive=False,
    )
    module._save_theta_comparison_plot(
        initial_samples,
        posterior_samples,
        prior_positive_plot_path,
        positive=True,
    )
    artifacts["prior_posterior_plot"] = str(prior_plot_path)
    artifacts["prior_posterior_positive_plot"] = str(prior_positive_plot_path)
    if posterior_predictive is not None:
        torch.save(
            {
                "cell_lines": list(joint_data.cell_lines),
                "grid_concentration": torch.as_tensor(posterior_predictive["grid_concentration"]),
                "predictive_mu": torch.as_tensor(posterior_predictive["predictive_mu"]),
                "predictive_y": torch.as_tensor(posterior_predictive["predictive_y"]),
            },
            posterior_predictive_path,
        )
        save_joint_posterior_predictive_plot(
            observed_concentration=joint_data.bmp4_conc,
            observed_response=joint_data.x_obs,
            grid_concentration=posterior_predictive["grid_concentration"],
            predictive_draws=posterior_predictive["predictive_y"],
            cell_lines=joint_data.cell_lines,
            output_path=posterior_plot_path,
            title=f"Sequential posterior predictive BMP4 dose-response: {metadata['family']}",
        )
        artifacts["posterior_predictive_path"] = str(posterior_predictive_path)
        artifacts["posterior_predictive_plot"] = str(posterior_plot_path)
    if trace:
        eig_plot_written = save_sequential_acquisition_plot(
            trace,
            eig_plot_path,
            title=f"Sequential acquired-design EIG: {metadata['family']}",
        )
        artifacts["eig_optimization_plot"] = str(eig_plot_written)

    fit_summary = {
        **metadata,
        "candidate_name": f"{metadata['family']}_sequential_lfiax",
        "cell_lines": list(joint_data.cell_lines),
        "theta_names": list(module.THETA_NAMES),
        "complex_names": list(module.COMPLEX_NAMES),
        "receptor_names": list(module.TARGET_RECEPTOR_NAMES),
        "theta_raw_prior": theta_prior,
        "initial_prior_summary": module._summarize_samples(initial_samples),
        "mcmc_posterior_summary": module._summarize_samples(posterior_samples),
        "mcmc_sample_count_by_cell": module._sample_counts(posterior_samples),
        "collected_count_by_cell": {
            str(cell_line): sum(
                1
                for item in trace
                if item["snapped_design"]["cell_line"] == str(cell_line)
            )
            for cell_line in joint_data.cell_lines
        },
        "acquisition_count": len(trace),
        "normalizer": normalizer.to_dict(),
        "artifacts": artifacts,
    }
    _write_json(fit_summary_path, fit_summary)
    return artifacts


def _write_likelihood_checkpoint(
    *,
    module: Any,
    path: Path,
    normalizer: Any,
    joint_data: Any,
    likelihood_state: Any,
    theta_prior: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    import jax

    bounds = module._bmp4_design_bounds(joint_data, normalizer)
    final_design_summary = module._summarize_bmp4_design_distribution(
        normalizer=normalizer,
        bmp4_norm_mu=likelihood_state.bmp4_norm_mu,
        bmp4_norm_log_std=likelihood_state.bmp4_norm_log_std,
        bounds=bounds,
        sample_key=jax.random.PRNGKey(0),
        sample_count=512,
    )
    final_log10_doses = np.log10(np.clip(final_design_summary["dose_mu"], 1e-30, None)).astype("float32")
    with path.open("wb") as handle:
        pickle.dump(
            {
                "jax_likelihood": {
                    "flow_params": module._jax_tree_to_numpy(likelihood_state.flow_params),
                    "flow_config": likelihood_state.flow_config,
                    "bmp4_norm_mu": final_design_summary["bmp4_norm_mu"],
                    "bmp4_norm_log_std": final_design_summary["bmp4_norm_log_std"],
                    "bmp4_norm_std": final_design_summary["bmp4_norm_std"],
                    "dose_mu": final_design_summary["dose_mu"],
                    "dose_std": final_design_summary["dose_std"],
                    "log10_doses": final_log10_doses,
                    "infonce_lambda": float(likelihood_state.infonce_lambda),
                    "infonce_negatives": int(likelihood_state.infonce_negatives),
                },
                "metadata": {
                    **metadata,
                    "theta_names": list(module.THETA_NAMES),
                    "complex_names": list(module.COMPLEX_NAMES),
                    "receptor_names": list(module.TARGET_RECEPTOR_NAMES),
                    "theta_raw_prior": theta_prior,
                    "normalizer": normalizer.to_dict(),
                    "likelihood_backend": "jax_lfiax_nsf",
                    "workflow": "sequential_retrospective",
                },
            },
            handle,
        )


def _family_module(family_name: str) -> Any:
    try:
        return FAMILY_MODULES[family_name]
    except KeyError as exc:
        raise ValueError(
            "family_name must be one of: " + ", ".join(sorted(FAMILY_MODULES))
        ) from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)


def _json_ready(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return [_json_ready(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "initialize_promisys_prior_samples",
    "run_promisys_sequential_workflow",
    "select_snapped_design",
    "snap_to_nearest_unused_design",
]
