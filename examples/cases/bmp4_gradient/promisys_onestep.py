from __future__ import annotations

import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from examples.cases.bmp4_gradient.promisys_hyperparams import (
    PromisysHyperparams,
    coerce_promisys_hyperparams,
    effective_promisys_hyperparams,
)


try:  # pragma: no cover - exercised in tests/examples when numpy is available
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


TARGET_RECEPTOR_NAMES = ("ACVR1", "BMPR1A", "ACVR2A", "ACVR2B", "BMPR2")
TYPE_I_RECEPTORS = ("ACVR1", "BMPR1A")
TYPE_II_RECEPTORS = ("ACVR2A", "ACVR2B", "BMPR2")
LIGAND_NAME = "BMP4"
MODEL_SIZE = (1, 2, 3)
DEFAULT_NORMALIZER_DATA_DIR = Path("/Users/vincentzaballa/Development/bmp_simformer/data")
DEFAULT_LFIAX_ROOT = Path("/Users/vincentzaballa/Development/lfiax")
COMPLEX_NAMES = tuple(
    f"{LIGAND_NAME}_{type_i}_{type_ii}"
    for type_i in TYPE_I_RECEPTORS
    for type_ii in TYPE_II_RECEPTORS
)
BINDING_PARAMETER_NAMES = tuple(f"K_{name}" for name in COMPLEX_NAMES)
EFFICIENCY_PARAMETER_NAMES = tuple(f"e_{name}" for name in COMPLEX_NAMES)
OBSERVATION_NOISE_PARAMETER_NAME = "sigma_y_norm"
BIOPHYSICAL_PARAMETER_NAMES = BINDING_PARAMETER_NAMES + EFFICIENCY_PARAMETER_NAMES
BIOPHYSICAL_THETA_DIM = len(BIOPHYSICAL_PARAMETER_NAMES)
THETA_NAMES = BIOPHYSICAL_PARAMETER_NAMES + (OBSERVATION_NOISE_PARAMETER_NAME,)
THETA_DIM = len(THETA_NAMES)
RAW_PARAMETER_LOW = 1e-4
RAW_PARAMETER_HIGH = 1e2
THETA_NORMALIZATION_SCALE = 1.81
OBSERVATION_NOISE_LOG_LOC = -1.5
OBSERVATION_NOISE_LOG_SCALE = 0.5
OBSERVATION_NOISE_MIN = 1e-4
OBSERVATION_NOISE_MAX = 5.0
PROGRESS_PRINT_INTERVAL = 250
BOED_PROGRESS_PRINT_INTERVAL = 1
DEFAULT_INFONCE_LAMBDA = 0.5
DEFAULT_INFONCE_NEGATIVES = 10
DEFAULT_DESIGN_DIST_INIT_STD = 0.3
DEFAULT_DESIGN_TEMPERATURE_SCALE = 1.0
DEFAULT_SELECTOR_TEMPERATURE_FINAL = 0.02
DEFAULT_POSTERIOR_NET_HIDDEN_DIM = 96
DEFAULT_POSTERIOR_NET_LAYERS = 2
DEFAULT_POSTERIOR_NET_ACTIVATION = "gelu"
DEFAULT_POSTERIOR_NET_BATCH_SIZE = 64
DEFAULT_MCMC_PROPOSAL_SCALE = 0.03
DEFAULT_MCMC_PRIOR_STD_FLOOR = 0.03
JAX_FLOW_CONFIG = {
    "event_shape": (1,),
    "num_layers": 4,
    "hidden_sizes": (96, 96),
    "num_bins": 8,
    "standardize_theta": False,
    "use_resnet": True,
    "conditional": True,
    "base_dist": "gaussian",
    "activation": "gelu",
    "dropout_rate": 0.0,
}


def _progress(message: str) -> None:
    print(f"[promisys_onestep] {message}", flush=True)


def _should_report_progress(
    step_index: int,
    total: int,
    *,
    interval: int = PROGRESS_PRINT_INTERVAL,
) -> bool:
    total = max(int(total), 1)
    current = int(step_index) + 1
    interval = max(int(interval), 1)
    return current == 1 or current == total or current % interval == 0


@dataclass(frozen=True)
class GammaTransform:
    shape: float
    loc: float
    scale: float

    def to_dict(self) -> dict[str, float]:
        return {"shape": self.shape, "loc": self.loc, "scale": self.scale}


@dataclass(frozen=True)
class Bmp4Normalizer:
    ligand_transforms: tuple[GammaTransform, ...]
    response_transform: GammaTransform
    receptor_mu: float = 0.75
    receptor_sigma: float = 1.5
    receptor_high: float = 5.0

    @classmethod
    def from_source_dir(
        cls,
        source_dir: str | Path = DEFAULT_NORMALIZER_DATA_DIR,
        *,
        max_fit_samples: int = 200_000,
    ) -> "Bmp4Normalizer":
        if np is None:
            raise RuntimeError("BMP4 normalization requires numpy.")
        source = Path(source_dir)
        ligand_path = source / "noised_Ls_4k.npy"
        response_path = source / "sim_x_fat_Rs_noised_Ls_4k.npy"
        missing = [str(path) for path in (ligand_path, response_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing BMP4 normalization source arrays: " + ", ".join(missing)
            )

        ligand_source = np.load(ligand_path, mmap_mode="r")
        response_source = np.load(response_path, mmap_mode="r")
        if ligand_source.ndim != 3 or ligand_source.shape[1] < 1:
            raise ValueError(
                f"Expected noised_Ls_4k.npy shape (N, ligands, conditions); got {ligand_source.shape}."
            )
        ligand_transforms = tuple(
            _fit_gamma_transform(
                ligand_source[:, ligand_index, :],
                max_fit_samples=max_fit_samples,
            )
            for ligand_index in range(ligand_source.shape[1])
        )
        response_transform = _fit_gamma_transform(
            response_source,
            max_fit_samples=max_fit_samples,
        )
        return cls(
            ligand_transforms=ligand_transforms,
            response_transform=response_transform,
        )

    def normalize_bmp4(self, value: Any) -> Any:
        return _gamma_to_gauss(value, self.ligand_transforms[0])

    def denormalize_bmp4(self, value: Any) -> Any:
        return _gauss_to_gamma(value, self.ligand_transforms[0])

    def normalize_response(self, value: Any) -> Any:
        return _gamma_to_gauss(value, self.response_transform)

    def denormalize_response(self, value: Any) -> Any:
        return _gauss_to_gamma(value, self.response_transform)

    def normalize_receptors(self, value: Any) -> Any:
        if np is None:
            raise RuntimeError("BMP4 receptor normalization requires numpy.")
        from scipy import stats

        raw = np.asarray(value, dtype=np.float64)
        clipped = np.clip(raw, 1e-12, self.receptor_high - 1e-12)
        z = (np.log(clipped) - self.receptor_mu) / self.receptor_sigma
        high_z = (math.log(self.receptor_high) - self.receptor_mu) / self.receptor_sigma
        cdf = stats.norm.cdf(z) / max(stats.norm.cdf(high_z), 1e-12)
        return stats.norm.ppf(np.clip(cdf, 1e-8, 1.0 - 1e-8)).astype("float32")

    def denormalize_receptors(self, value: Any) -> Any:
        if np is None:
            raise RuntimeError("BMP4 receptor denormalization requires numpy.")
        from scipy import stats

        z = np.asarray(value, dtype=np.float64)
        high_z = (math.log(self.receptor_high) - self.receptor_mu) / self.receptor_sigma
        cdf = stats.norm.cdf(z) * max(stats.norm.cdf(high_z), 1e-12)
        raw_z = stats.norm.ppf(np.clip(cdf, 1e-12, 1.0 - 1e-12))
        return np.clip(
            np.exp(self.receptor_mu + self.receptor_sigma * raw_z),
            1e-12,
            self.receptor_high,
        ).astype("float32")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ligand_transforms": [item.to_dict() for item in self.ligand_transforms],
            "response_transform": self.response_transform.to_dict(),
            "receptor_mu": self.receptor_mu,
            "receptor_sigma": self.receptor_sigma,
            "receptor_high": self.receptor_high,
        }


class GaussianMLP:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = DEFAULT_POSTERIOR_NET_HIDDEN_DIM,
        layers: int = DEFAULT_POSTERIOR_NET_LAYERS,
        activation: str = DEFAULT_POSTERIOR_NET_ACTIVATION,
    ) -> None:
        import torch

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.activation = str(activation)
        modules: list[Any] = []
        in_dim = self.input_dim
        for _ in range(self.layers):
            modules.append(torch.nn.Linear(in_dim, self.hidden_dim))
            modules.append(_torch_activation(self.activation))
            in_dim = self.hidden_dim
        modules.append(torch.nn.Linear(in_dim, 2 * self.output_dim))
        self.model = torch.nn.Sequential(*modules)

    def distribution(self, features: Any) -> Any:
        import torch

        features_t = torch.as_tensor(features, dtype=torch.float32)
        raw = self.model(features_t)
        loc, raw_scale = raw[..., : self.output_dim], raw[..., self.output_dim :]
        scale = torch.nn.functional.softplus(raw_scale) + 1e-4
        return torch.distributions.Normal(loc, scale)

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "activation": self.activation,
            "model_state_dict": self.model.state_dict(),
        }


def _torch_activation(name: str) -> Any:
    import torch

    normalized = str(name).lower()
    if normalized == "gelu":
        return torch.nn.GELU()
    if normalized == "relu":
        return torch.nn.ReLU()
    if normalized == "silu":
        return torch.nn.SiLU()
    if normalized == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"Unsupported posterior network activation: {name!r}")


@dataclass(frozen=True)
class JaxLikelihoodState:
    flow_params: Any
    flow_config: dict[str, Any]
    bmp4_norm_mu: Any
    bmp4_norm_log_std: Any
    selector_logits: Any
    selector_temperature: float
    history: list[dict[str, Any]]
    loss_history: list[float]
    gradient_diagnostics: list[dict[str, Any]]
    infonce_lambda: float
    infonce_negatives: int


def run_promisys_onestep_workflow(
    *,
    joint_data: Any,
    run_dir: str | Path,
    normalizer_data_dir: str | Path = DEFAULT_NORMALIZER_DATA_DIR,
    receptor_noise_log_sd: float = 0.05,
    snpe_steps: int = 500,
    snpe_simulations: int = 512,
    snpe_learning_rate: float = 1e-3,
    likelihood_steps: int = 500,
    likelihood_learning_rate: float = 1e-3,
    posterior_sample_count: int = 256,
    eig_steps: int = 100,
    eig_outer_samples: int = 64,
    eig_inner_samples: int | None = None,
    eig_learning_rate: float = 0.05,
    infonce_lambda: float = DEFAULT_INFONCE_LAMBDA,
    design_dist_init_std: float = DEFAULT_DESIGN_DIST_INIT_STD,
    design_temperature_scale: float = DEFAULT_DESIGN_TEMPERATURE_SCALE,
    selector_temperature_final: float = DEFAULT_SELECTOR_TEMPERATURE_FINAL,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    mcmc_warmup: int = 200,
    mcmc_samples: int = 256,
    seed: int = 0,
    lfiax_root: str | Path | None = DEFAULT_LFIAX_ROOT,
    literature_prior: Any | None = None,
    promisys_hyperparams: PromisysHyperparams | dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    if np is None:
        raise RuntimeError("The promisys_onestep workflow requires numpy.")
    import torch

    _progress("Starting BMP4 promisys one-step workflow")
    _validate_joint_data(joint_data)
    _progress(
        "Using cell lines: "
        + ", ".join(str(cell_line) for cell_line in joint_data.cell_lines)
    )
    _progress("Checking promisys simulator dependencies")
    _require_promisys(lfiax_root)

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    _progress(f"Artifacts will be written to {run_path}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))
    hyperparams = coerce_promisys_hyperparams(promisys_hyperparams)
    posterior_hp = hyperparams.posterior_net if hyperparams is not None else None
    objective_hp = hyperparams.objective if hyperparams is not None else None
    mcmc_hp = hyperparams.mcmc if hyperparams is not None else None
    effective_snpe_steps = int(
        posterior_hp.steps if posterior_hp is not None and posterior_hp.steps is not None else snpe_steps
    )
    effective_snpe_simulations = int(
        posterior_hp.simulations
        if posterior_hp is not None and posterior_hp.simulations is not None
        else snpe_simulations
    )
    effective_snpe_learning_rate = float(
        posterior_hp.learning_rate
        if posterior_hp is not None and posterior_hp.learning_rate is not None
        else snpe_learning_rate
    )
    effective_posterior_sample_count = int(
        posterior_hp.posterior_samples
        if posterior_hp is not None and posterior_hp.posterior_samples is not None
        else posterior_sample_count
    )
    posterior_hidden_dim = int(
        posterior_hp.hidden_dim
        if posterior_hp is not None and posterior_hp.hidden_dim is not None
        else DEFAULT_POSTERIOR_NET_HIDDEN_DIM
    )
    posterior_layers = int(
        posterior_hp.layers
        if posterior_hp is not None and posterior_hp.layers is not None
        else DEFAULT_POSTERIOR_NET_LAYERS
    )
    posterior_activation = str(
        posterior_hp.activation
        if posterior_hp is not None and posterior_hp.activation is not None
        else DEFAULT_POSTERIOR_NET_ACTIVATION
    )
    posterior_batch_size = int(
        posterior_hp.batch_size
        if posterior_hp is not None and posterior_hp.batch_size is not None
        else DEFAULT_POSTERIOR_NET_BATCH_SIZE
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
        else (eig_inner_samples or DEFAULT_INFONCE_NEGATIVES)
    )
    effective_eig_learning_rate = float(
        objective_hp.design_learning_rate
        if objective_hp is not None and objective_hp.design_learning_rate is not None
        else eig_learning_rate
    )
    effective_infonce_lambda = float(
        objective_hp.infonce_lambda
        if objective_hp is not None and objective_hp.infonce_lambda is not None
        else infonce_lambda
    )
    effective_design_dist_init_std = float(
        objective_hp.design_dist_init_std
        if objective_hp is not None and objective_hp.design_dist_init_std is not None
        else design_dist_init_std
    )
    effective_design_temperature_scale = float(
        objective_hp.design_temperature_scale
        if objective_hp is not None and objective_hp.design_temperature_scale is not None
        else design_temperature_scale
    )
    effective_selector_temperature_final = float(
        objective_hp.selector_temperature_final
        if objective_hp is not None and objective_hp.selector_temperature_final is not None
        else selector_temperature_final
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
        else DEFAULT_MCMC_PROPOSAL_SCALE
    )
    effective_mcmc_prior_std_floor = float(
        mcmc_hp.prior_std_floor
        if mcmc_hp is not None and mcmc_hp.prior_std_floor is not None
        else DEFAULT_MCMC_PRIOR_STD_FLOOR
    )
    flow_config = hyperparams.flow_config(JAX_FLOW_CONFIG) if hyperparams is not None else dict(JAX_FLOW_CONFIG)
    effective_hyperparams = effective_promisys_hyperparams(
        base_flow_config=JAX_FLOW_CONFIG,
        hyperparams=hyperparams,
        snpe_steps=effective_snpe_steps,
        snpe_simulations=effective_snpe_simulations,
        snpe_learning_rate=effective_snpe_learning_rate,
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
        posterior_hidden_dim=posterior_hidden_dim,
        posterior_layers=posterior_layers,
        posterior_activation=posterior_activation,
        posterior_batch_size=posterior_batch_size,
    )
    if hyperparams is not None:
        _progress(f"Using Promisys hyperparameter overrides: {hyperparams.to_dict()}")

    _progress(f"Loading normalizer data from {normalizer_data_dir}")
    normalizer = Bmp4Normalizer.from_source_dir(normalizer_data_dir)
    theta_prior = _build_onestep_theta_prior(literature_prior)
    if theta_prior["mode"] == "expert_mapped":
        mapped = [
            item["parameter"]
            for item in theta_prior["parameter_priors"]
            if item["source"] == "literature_prior"
        ]
        _progress(
            "Using mapped expert prior for onestep SNPE K parameters: "
            + ", ".join(mapped)
        )
    else:
        _progress("Using default broad log-uniform SNPE theta prior")
    snpe_train_count = max(int(effective_snpe_simulations), len(joint_data.cell_lines))
    _progress(
        f"Simulating SNPE training data: {snpe_train_count} trajectories "
        f"(receptor_noise_log_sd={float(receptor_noise_log_sd):.4g})"
    )
    posterior_features, posterior_targets, simulation_summary = _simulate_snpe_training_data(
        joint_data=joint_data,
        normalizer=normalizer,
        count=snpe_train_count,
        receptor_noise_log_sd=float(receptor_noise_log_sd),
        theta_prior=theta_prior,
        rng=rng,
        lfiax_root=lfiax_root,
    )
    _progress(
        "Fitting SNPE posterior network: "
        f"{int(effective_snpe_steps)} steps on {posterior_features.shape[0]} simulations"
    )
    posterior_net = GaussianMLP(
        posterior_features.shape[1],
        THETA_DIM,
        hidden_dim=posterior_hidden_dim,
        layers=posterior_layers,
        activation=posterior_activation,
    )
    snpe_loss_history = _train_gaussian_net(
        posterior_net,
        posterior_features,
        posterior_targets,
        steps=effective_snpe_steps,
        learning_rate=effective_snpe_learning_rate,
        batch_size=posterior_batch_size,
        seed=seed,
        label="SNPE posterior",
    )

    _progress(
        f"Sampling SNPE posterior priors: {int(effective_posterior_sample_count)} samples per cell line"
    )
    snpe_samples = _sample_cell_line_posteriors(
        posterior_net=posterior_net,
        joint_data=joint_data,
        sample_count=int(effective_posterior_sample_count),
    )

    _progress(
        "Running fixed-posterior multi-context JAX BOED: "
        f"{int(effective_likelihood_steps)} joint steps, {int(effective_eig_outer_samples)} posterior samples, "
        f"{int(effective_eig_inner_samples)} contrastives"
    )
    likelihood_state, eig_result = _run_joint_multicontext_boed(
        joint_data=joint_data,
        normalizer=normalizer,
        snpe_samples=snpe_samples,
        steps=int(effective_likelihood_steps),
        outer_samples=int(effective_eig_outer_samples),
        inner_samples=int(effective_eig_inner_samples),
        flow_learning_rate=float(effective_likelihood_learning_rate),
        design_learning_rate=float(effective_eig_learning_rate),
        infonce_lambda=float(effective_infonce_lambda),
        design_dist_init_std=float(effective_design_dist_init_std),
        design_temperature_scale=float(effective_design_temperature_scale),
        selector_temperature_final=float(effective_selector_temperature_final),
        receptor_noise_log_sd=float(receptor_noise_log_sd),
        flow_config=flow_config,
        rng=rng,
        seed=seed + 1,
        lfiax_root=lfiax_root,
        early_stopping_patience=effective_early_stopping_patience,
        early_stopping_min_delta=effective_early_stopping_min_delta,
    )
    eig_result["eig_steps_ignored_for_promisys_onestep"] = True
    eig_result["eig_steps_requested"] = int(eig_steps)
    _progress(
        "Final next experiment: "
        f"cell_line={eig_result['best_cell_line']}, "
        f"dose_mu={eig_result['best_dose']:.6g}, "
        f"dose_std={eig_result['best_dose_std']:.3g}, "
        f"eig={eig_result['best_eig']:.4f}"
    )
    _progress(
        f"Running MCMC posterior update: warmup={int(effective_mcmc_warmup)}, "
        f"samples={int(effective_mcmc_samples)} per cell line"
    )
    mcmc_samples_by_cell_line = _run_all_cell_line_mcmc(
        joint_data=joint_data,
        likelihood_state=likelihood_state,
        snpe_samples=snpe_samples,
        warmup=int(effective_mcmc_warmup),
        sample_count=int(effective_mcmc_samples),
        proposal_scale=float(effective_mcmc_proposal_scale),
        prior_std_floor=float(effective_mcmc_prior_std_floor),
        rng=rng,
    )

    _progress("Generating posterior predictive summaries")
    posterior_predictive = _posterior_predictive_from_likelihood(
        joint_data=joint_data,
        normalizer=normalizer,
        mcmc_samples_by_cell_line=mcmc_samples_by_cell_line,
        grid_points=128,
        rng=rng,
        lfiax_root=lfiax_root,
    )

    _progress("Writing tensors, summaries, checkpoints, and plots")
    artifacts = _write_promisys_artifacts(
        run_path=run_path,
        normalizer=normalizer,
        posterior_net=posterior_net,
        likelihood_state=likelihood_state,
        joint_data=joint_data,
        snpe_samples=snpe_samples,
        mcmc_samples_by_cell_line=mcmc_samples_by_cell_line,
        posterior_predictive=posterior_predictive,
        eig_result=eig_result,
        snpe_loss_history=snpe_loss_history,
        likelihood_loss_history=likelihood_state.loss_history,
        simulation_summary=simulation_summary,
        normalizer_data_dir=normalizer_data_dir,
        receptor_noise_log_sd=receptor_noise_log_sd,
        eig_steps=eig_steps,
        requested_hyperparams=hyperparams.to_dict() if hyperparams is not None else None,
        effective_hyperparams=effective_hyperparams,
    )
    _progress("Promisys one-step workflow complete")
    return {
        "family": "promisys_onestep",
        "cell_lines": list(joint_data.cell_lines),
        "run_dir": str(run_path),
        "best_design": eig_result["best_design"],
        "best_dose": eig_result["best_dose"],
        "best_dose_mu": eig_result["best_dose_mu"],
        "best_log10_dose": eig_result["best_log10_dose"],
        "best_bmp4_norm_design": eig_result["best_bmp4_norm_design"],
        "best_bmp4_norm_mu": eig_result["best_bmp4_norm_mu"],
        "best_bmp4_norm_std": eig_result["best_bmp4_norm_std"],
        "best_dose_std": eig_result["best_dose_std"],
        "best_cell_line": eig_result["best_cell_line"],
        "best_selector_probs": eig_result["best_selector_probs"],
        "best_eig": eig_result["best_eig"],
        "eig_estimator": "lf_pce_infonce",
        "posterior_samples_path": artifacts["posterior_samples_path"],
        "snpe_posterior_samples_path": artifacts["snpe_posterior_samples_path"],
        "mcmc_posterior_samples_path": artifacts["mcmc_posterior_samples_path"],
        "posterior_predictive_path": artifacts["posterior_predictive_path"],
        "posterior_predictive_plot": artifacts["posterior_predictive_plot"],
        "prior_posterior_plot": artifacts["prior_posterior_plot"],
        "prior_posterior_positive_plot": artifacts["prior_posterior_positive_plot"],
        "eig_optimization_plot": artifacts["eig_optimization_plot"],
        "likelihood_checkpoint": artifacts["likelihood_checkpoint"],
        "snpe_fit_summary_path": artifacts["snpe_fit_summary_path"],
        "fit_summary_path": artifacts["fit_summary_path"],
        "promisys_hyperparams": effective_hyperparams,
    }


def theta_norm_to_raw(theta_norm: Any) -> Any:
    if np is None:
        raise RuntimeError("theta conversion requires numpy.")
    theta = np.asarray(theta_norm, dtype=np.float64)
    if theta.shape[-1] != THETA_DIM:
        return _biophysical_theta_norm_to_raw(theta)
    biophysical_raw = _biophysical_theta_norm_to_raw(theta[..., :BIOPHYSICAL_THETA_DIM])
    sigma_y = _observation_noise_latent_to_positive(theta[..., BIOPHYSICAL_THETA_DIM])
    return np.concatenate([biophysical_raw, np.expand_dims(sigma_y, axis=-1)], axis=-1).astype("float32")


def theta_raw_to_norm(theta_raw: Any) -> Any:
    if np is None:
        raise RuntimeError("theta conversion requires numpy.")
    theta = np.asarray(theta_raw, dtype=np.float64)
    if theta.shape[-1] != THETA_DIM:
        return _biophysical_theta_raw_to_norm(theta)
    biophysical_norm = _biophysical_theta_raw_to_norm(theta[..., :BIOPHYSICAL_THETA_DIM])
    sigma_latent = _observation_noise_positive_to_latent(theta[..., BIOPHYSICAL_THETA_DIM])
    return np.concatenate([biophysical_norm, np.expand_dims(sigma_latent, axis=-1)], axis=-1).astype("float32")


def theta_norm_to_observation_noise(theta_norm: Any) -> Any:
    if np is None:
        raise RuntimeError("theta conversion requires numpy.")
    theta = np.asarray(theta_norm, dtype=np.float64)
    if theta.shape[-1] != THETA_DIM:
        raise ValueError(f"Expected theta dimension {THETA_DIM}, got {theta.shape[-1]}.")
    return _observation_noise_latent_to_positive(theta[..., BIOPHYSICAL_THETA_DIM])


def _biophysical_theta_norm_to_raw(theta_norm: Any) -> Any:
    theta = np.asarray(theta_norm, dtype=np.float64)
    logits = np.clip(theta * THETA_NORMALIZATION_SCALE, -40.0, 40.0)
    unit = 1.0 / (1.0 + np.exp(-logits))
    log_low = math.log(RAW_PARAMETER_LOW)
    log_high = math.log(RAW_PARAMETER_HIGH)
    return np.exp(log_low + unit * (log_high - log_low)).astype("float32")


def _biophysical_theta_raw_to_norm(theta_raw: Any) -> Any:
    theta = np.asarray(theta_raw, dtype=np.float64)
    log_low = math.log(RAW_PARAMETER_LOW)
    log_high = math.log(RAW_PARAMETER_HIGH)
    unit = (np.log(np.clip(theta, RAW_PARAMETER_LOW, RAW_PARAMETER_HIGH)) - log_low) / (
        log_high - log_low
    )
    unit = np.clip(unit, 1e-6, 1.0 - 1e-6)
    return (np.log(unit) - np.log1p(-unit)).astype("float32") / THETA_NORMALIZATION_SCALE


def _observation_noise_latent_to_positive(latent: Any) -> Any:
    values = np.asarray(latent, dtype=np.float64)
    sigma = np.exp(OBSERVATION_NOISE_LOG_LOC + OBSERVATION_NOISE_LOG_SCALE * values)
    return np.clip(sigma, OBSERVATION_NOISE_MIN, OBSERVATION_NOISE_MAX).astype("float32")


def _observation_noise_positive_to_latent(sigma: Any) -> Any:
    values = np.asarray(sigma, dtype=np.float64)
    clipped = np.clip(values, OBSERVATION_NOISE_MIN, OBSERVATION_NOISE_MAX)
    return ((np.log(clipped) - OBSERVATION_NOISE_LOG_LOC) / OBSERVATION_NOISE_LOG_SCALE).astype("float32")


def sample_theta_norm_prior(rng: Any, size: int | tuple[int, ...]) -> Any:
    shape = (int(size),) if isinstance(size, int) else tuple(int(item) for item in size)
    if shape and shape[-1] == THETA_DIM:
        biophysical = _sample_biophysical_theta_norm_prior(rng, shape[:-1] + (BIOPHYSICAL_THETA_DIM,))
        sigma_latent = np.asarray(rng.normal(0.0, 1.0, size=shape[:-1]), dtype=np.float32)
        return np.concatenate([biophysical, np.expand_dims(sigma_latent, axis=-1)], axis=-1).astype("float32")
    return _sample_biophysical_theta_norm_prior(rng, shape)


def _sample_biophysical_theta_norm_prior(rng: Any, size: int | tuple[int, ...]) -> Any:
    unit = rng.uniform(1e-6, 1.0 - 1e-6, size=size)
    return (np.log(unit) - np.log1p(-unit)).astype("float32") / THETA_NORMALIZATION_SCALE


def sample_theta_norm_from_prior_config(rng: Any, theta_prior: dict[str, Any]) -> Any:
    if theta_prior.get("mode") != "expert_mapped":
        return sample_theta_norm_prior(rng, THETA_DIM)
    raw_values = []
    for item in theta_prior["parameter_priors"]:
        if item["distribution"] == "LogNormal":
            params = item["params"]
            value = rng.lognormal(
                mean=float(params["loc"]),
                sigma=float(params["scale"]),
            )
            raw_values.append(float(np.clip(value, RAW_PARAMETER_LOW, RAW_PARAMETER_HIGH)))
        else:
            raw_values.append(float(_biophysical_theta_norm_to_raw(_sample_biophysical_theta_norm_prior(rng, 1))[0]))
    biophysical = _biophysical_theta_raw_to_norm(np.asarray(raw_values, dtype=np.float32))
    sigma_latent = np.asarray([rng.normal(0.0, 1.0)], dtype=np.float32)
    return np.concatenate([biophysical, sigma_latent], axis=0).astype("float32")


def _build_onestep_theta_prior(literature_prior: Any | None) -> dict[str, Any]:
    distributions = getattr(literature_prior, "distributions", None)
    if not distributions:
        return _default_theta_prior_config()
    parameter_priors = []
    used = 0
    for name in BIOPHYSICAL_PARAMETER_NAMES:
        receptor_name = _onestep_receptor_prior_name_for_parameter(name)
        spec, source_parameter = _lookup_kd_prior_spec(distributions, receptor_name)
        if spec is not None:
            parameter_priors.append(
                {
                    "parameter": name,
                    "source": "literature_prior",
                    "source_parameter": source_parameter,
                    "distribution": "LogNormal",
                    "params": _coerce_lognormal_params(spec),
                    "raw_low": RAW_PARAMETER_LOW,
                    "raw_high": RAW_PARAMETER_HIGH,
                }
            )
            used += 1
        else:
            parameter_priors.append(_default_biophysical_parameter_prior(name))
    return {
        "mode": "expert_mapped" if used else "default_loguniform",
        "mapped_parameter_count": int(used),
        "parameter_priors": parameter_priors,
        "observation_noise_prior": _observation_noise_prior_config(),
        "mapping": (
            "Onestep trimer affinities K_BMP4_typeI_typeII use kd_typeII priors. "
            "If receptor-specific kd_* priors are absent, a generic kd prior is reused. "
            "Efficiency parameters and sigma_y_norm use the default promisys implicit priors."
        ),
    }


def _lookup_kd_prior_spec(
    distributions: Any,
    receptor_name: str | None,
) -> tuple[Any | None, str | None]:
    if receptor_name is None:
        return None, None
    if receptor_name:
        source_parameter = f"kd_{receptor_name}"
        spec = distributions.get(source_parameter)
        if spec is not None and _is_lognormal_distribution(spec):
            return spec, source_parameter
    spec = distributions.get("kd")
    if spec is not None and _is_lognormal_distribution(spec):
        return spec, "kd"
    return None, None


def _is_lognormal_distribution(spec: Any) -> bool:
    name = str(_distribution_name(spec) or "")
    normalized = name.replace("-", "").replace("_", "").lower()
    return normalized == "lognormal"


def _coerce_lognormal_params(spec: Any) -> dict[str, float]:
    params = _distribution_params(spec)
    loc = _first_float(params, ("loc", "mu", "mu_log", "log_loc"))
    if loc is None:
        median = _first_float(params, ("median", "median_K_eqtk", "center"))
        loc = math.log(max(median, RAW_PARAMETER_LOW)) if median is not None else 0.0
    scale = _first_float(params, ("scale", "sigma", "sigma_log", "log_scale"))
    if scale is None:
        scale = 1.0
    return {"loc": float(loc), "scale": float(scale)}


def _first_float(params: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in params:
            continue
        try:
            return float(params[key])
        except (TypeError, ValueError):
            continue
    return None


def _default_theta_prior_config() -> dict[str, Any]:
    return {
        "mode": "default_loguniform",
        "mapped_parameter_count": 0,
        "parameter_priors": [
            _default_biophysical_parameter_prior(name) for name in BIOPHYSICAL_PARAMETER_NAMES
        ],
        "observation_noise_prior": _observation_noise_prior_config(),
    }


def _default_biophysical_parameter_prior(name: str) -> dict[str, Any]:
    return {
        "parameter": name,
        "source": "default",
        "distribution": "LogUniform",
        "low": RAW_PARAMETER_LOW,
        "high": RAW_PARAMETER_HIGH,
    }


def _observation_noise_prior_config() -> dict[str, Any]:
    return {
        "distribution": "LogNormal",
        "parameter_name": OBSERVATION_NOISE_PARAMETER_NAME,
        "space": "normalized_response",
        "loc": OBSERVATION_NOISE_LOG_LOC,
        "scale": OBSERVATION_NOISE_LOG_SCALE,
        "low": OBSERVATION_NOISE_MIN,
        "high": OBSERVATION_NOISE_MAX,
    }


def _onestep_receptor_prior_name_for_parameter(parameter_name: str) -> str | None:
    if parameter_name in EFFICIENCY_PARAMETER_NAMES:
        return None
    prefix = f"K_{LIGAND_NAME}_"
    if not parameter_name.startswith(prefix):
        return None
    receptor_part = parameter_name.removeprefix(prefix)
    pieces = receptor_part.split("_")
    if len(pieces) == 2:
        return pieces[1]
    return None


def _distribution_name(spec: Any) -> str | None:
    return getattr(spec, "name", None) or getattr(spec, "distribution", None)


def _distribution_params(spec: Any) -> dict[str, Any]:
    return dict(getattr(spec, "params", {}) or {})


def split_theta_raw(theta_raw: Any) -> tuple[Any, Any, Any]:
    if np is None:
        raise RuntimeError("theta splitting requires numpy.")
    theta = np.asarray(theta_raw, dtype=np.float32)
    if theta.shape[-1] != THETA_DIM:
        raise ValueError(f"Expected theta dimension {THETA_DIM}, got {theta.shape[-1]}.")
    return (
        theta[..., : len(COMPLEX_NAMES)],
        theta[..., len(COMPLEX_NAMES) : BIOPHYSICAL_THETA_DIM],
        theta[..., BIOPHYSICAL_THETA_DIM],
    )


def check_promisys_dependencies(lfiax_root: str | Path | None = DEFAULT_LFIAX_ROOT) -> list[str]:
    missing: list[str] = []
    try:
        import eqtk  # noqa: F401
    except ImportError:
        missing.append("eqtk")

    if lfiax_root is not None:
        root = Path(lfiax_root)
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        import bmp_simulator.promisys.bmp_util  # noqa: F401
    except ImportError:
        missing.append("bmp_simulator.promisys.bmp_util")
    return missing


def simulate_promisys_onestep_raw(
    *,
    bmp4_concentrations: Any,
    receptors: Any,
    theta_norm: Any,
    receptor_noise_log_sd: float = 0.0,
    rng: Any | None = None,
    lfiax_root: str | Path | None = DEFAULT_LFIAX_ROOT,
) -> Any:
    if np is None:
        raise RuntimeError("Promisys simulation requires numpy.")
    bmp_util = _require_promisys(lfiax_root)
    rng = np.random.default_rng() if rng is None else rng
    doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
    base_receptors = np.asarray(receptors, dtype=np.float32).reshape(-1)
    if base_receptors.shape[0] != len(TARGET_RECEPTOR_NAMES):
        raise ValueError(
            f"Expected {len(TARGET_RECEPTOR_NAMES)} receptors, got {base_receptors.shape[0]}."
        )
    theta = np.asarray(theta_norm, dtype=np.float32)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.shape[-1] != THETA_DIM:
        raise ValueError(f"Expected theta dimension {THETA_DIM}, got {theta.shape[-1]}.")

    outputs = np.zeros((theta.shape[0], doses.shape[0]), dtype=np.float32)
    for sample_index, theta_sample in enumerate(theta):
        theta_raw = theta_norm_to_raw(theta_sample)
        affinities, efficiencies, _ = split_theta_raw(theta_raw)
        receptors_i = np.array(base_receptors, dtype=np.float32)
        if receptor_noise_log_sd > 0.0:
            receptors_i = receptors_i * rng.lognormal(
                mean=0.0,
                sigma=float(receptor_noise_log_sd),
                size=receptors_i.shape,
            ).astype("float32")
        receptors_i = np.clip(receptors_i, 1e-8, 5.0)
        for dose_index, dose in enumerate(doses):
            signal = bmp_util.sim_S_LAB(
                MODEL_SIZE,
                np.array([max(float(dose), 1e-12)], dtype=np.float32),
                receptors_i,
                affinities,
                efficiencies,
                model="onestep",
                fixed_receptor=False,
            )
            outputs[sample_index, dose_index] = float(np.asarray(signal).reshape(-1)[0])
    return outputs


def simulate_promisys_onestep_paired_raw(
    *,
    bmp4_concentrations: Any,
    receptors: Any,
    theta_norm: Any,
    receptor_noise_log_sd: float = 0.0,
    rng: Any | None = None,
    lfiax_root: str | Path | None = DEFAULT_LFIAX_ROOT,
) -> Any:
    if np is None:
        raise RuntimeError("Promisys simulation requires numpy.")
    bmp_util = _require_promisys(lfiax_root)
    rng = np.random.default_rng() if rng is None else rng
    doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
    base_receptors = np.asarray(receptors, dtype=np.float32).reshape(-1)
    theta = np.asarray(theta_norm, dtype=np.float32)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.shape[0] != doses.shape[0]:
        raise ValueError(
            "Paired promisys simulation requires one BMP4 concentration per theta sample; "
            f"got {doses.shape[0]} doses and {theta.shape[0]} theta samples."
        )
    outputs = np.zeros((theta.shape[0],), dtype=np.float32)
    for sample_index, (theta_sample, dose) in enumerate(zip(theta, doses, strict=False)):
        theta_raw = theta_norm_to_raw(theta_sample)
        affinities, efficiencies, _ = split_theta_raw(theta_raw)
        receptors_i = np.array(base_receptors, dtype=np.float32)
        if receptor_noise_log_sd > 0.0:
            receptors_i = receptors_i * rng.lognormal(
                mean=0.0,
                sigma=float(receptor_noise_log_sd),
                size=receptors_i.shape,
            ).astype("float32")
        receptors_i = np.clip(receptors_i, 1e-8, 5.0)
        signal = bmp_util.sim_S_LAB(
            MODEL_SIZE,
            np.array([max(float(dose), 1e-12)], dtype=np.float32),
            receptors_i,
            affinities,
            efficiencies,
            model="onestep",
            fixed_receptor=False,
        )
        outputs[sample_index] = float(np.asarray(signal).reshape(-1)[0])
    return outputs


def problem_description(problem_summary: str, receptor_names: Sequence[str]) -> str:
    return (
        f"{problem_summary} Candidate model family: promisys one-step BMP4 trimeric "
        f"model over receptors {', '.join(receptor_names)}. Infer six binding "
        "affinities, six phosphorylation efficiencies, and one normalized "
        "observation-noise scale with simulator-based posterior and likelihood "
        "surrogates."
    )


def metadata_parameters() -> list[dict[str, str]]:
    parameters = [
        {
            "name": name,
            "description": "Normalized one-step BMP4 binding affinity"
            if name.startswith("K_")
            else "Normalized one-step BMP4 phosphorylation efficiency",
        }
        for name in BIOPHYSICAL_PARAMETER_NAMES
    ]
    parameters.append(
        {
            "name": OBSERVATION_NOISE_PARAMETER_NAME,
            "description": "Positive observation-noise scale in normalized BMP4 response space.",
        }
    )
    return parameters


def _fit_gamma_transform(values: Any, *, max_fit_samples: int) -> GammaTransform:
    if np is None:
        raise RuntimeError("Gamma fitting requires numpy.")
    from scipy import stats

    flat = np.asarray(values).reshape(-1)
    finite = flat[np.isfinite(flat)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return GammaTransform(shape=1.0, loc=0.0, scale=1.0)
    if positive.size > max_fit_samples:
        indices = np.linspace(0, positive.size - 1, max_fit_samples, dtype=np.int64)
        positive = positive[indices]
    positive = np.clip(positive.astype(np.float64), 1e-12, None)
    shape, loc, scale = stats.gamma.fit(positive, floc=0.0)
    return GammaTransform(float(shape), float(loc), float(scale))


def _gamma_to_gauss(value: Any, transform: GammaTransform) -> Any:
    if np is None:
        raise RuntimeError("Gamma normalization requires numpy.")
    from scipy import stats

    raw = np.asarray(value, dtype=np.float64)
    cdf = stats.gamma.cdf(
        np.clip(raw, 1e-12, None),
        a=transform.shape,
        loc=transform.loc,
        scale=transform.scale,
    )
    return stats.norm.ppf(np.clip(cdf, 1e-8, 1.0 - 1e-8)).astype("float32")


def _gauss_to_gamma(value: Any, transform: GammaTransform) -> Any:
    if np is None:
        raise RuntimeError("Gamma denormalization requires numpy.")
    from scipy import stats

    z = np.asarray(value, dtype=np.float64)
    cdf = stats.norm.cdf(z)
    return stats.gamma.ppf(
        np.clip(cdf, 1e-8, 1.0 - 1e-8),
        a=transform.shape,
        loc=transform.loc,
        scale=transform.scale,
    ).astype("float32")


def _validate_joint_data(joint_data: Any) -> None:
    receptor_names = tuple(joint_data.receptor_names)
    if receptor_names != TARGET_RECEPTOR_NAMES:
        raise ValueError(
            "promisys_onestep requires receptor order "
            f"{TARGET_RECEPTOR_NAMES}; got {receptor_names}."
        )
    required_attrs = ("x_obs_norm", "Rs_norm", "bmp4_conc_norm")
    missing = [name for name in required_attrs if getattr(joint_data, name, None) is None]
    if missing:
        raise ValueError(f"Joint BMP4 data is missing normalized fields: {missing}.")


def _require_promisys(lfiax_root: str | Path | None) -> Any:
    missing = check_promisys_dependencies(lfiax_root)
    if missing:
        raise RuntimeError(
            "promisys_onestep requires optional simulator dependencies that are "
            f"not importable: {', '.join(missing)}. Install `eqtk` in the active "
            "environment and ensure /Users/vincentzaballa/Development/lfiax is importable."
        )
    from bmp_simulator.promisys import bmp_util

    return bmp_util


def _sample_normalized_observations_from_mean_raw(
    *,
    normalizer: Bmp4Normalizer,
    mean_raw: Any,
    theta_norm: Any,
    rng: Any,
) -> Any:
    mean_raw_arr = np.asarray(mean_raw, dtype=np.float32)
    mean_norm = np.asarray(normalizer.normalize_response(mean_raw_arr), dtype=np.float32)
    theta_arr = np.asarray(theta_norm, dtype=np.float32)
    if theta_arr.ndim == 1:
        theta_arr = theta_arr[None, :]
    sigma_y = np.asarray(theta_norm_to_observation_noise(theta_arr), dtype=np.float32)
    sigma_shape = sigma_y.shape + (1,) * max(mean_norm.ndim - sigma_y.ndim, 0)
    noise = rng.normal(
        loc=0.0,
        scale=np.reshape(sigma_y, sigma_shape),
        size=mean_norm.shape,
    ).astype("float32")
    return (mean_norm + noise).astype("float32")


def _sample_raw_observations_from_mean_raw(
    *,
    normalizer: Bmp4Normalizer,
    mean_raw: Any,
    theta_norm: Any,
    rng: Any,
) -> Any:
    y_norm = _sample_normalized_observations_from_mean_raw(
        normalizer=normalizer,
        mean_raw=mean_raw,
        theta_norm=theta_norm,
        rng=rng,
    )
    return np.asarray(normalizer.denormalize_response(y_norm), dtype=np.float32)


def _simulate_snpe_training_data(
    *,
    joint_data: Any,
    normalizer: Bmp4Normalizer,
    count: int,
    receptor_noise_log_sd: float,
    theta_prior: dict[str, Any] | None,
    rng: Any,
    lfiax_root: str | Path | None,
) -> tuple[Any, Any, dict[str, Any]]:
    features: list[Any] = []
    targets: list[Any] = []
    cell_lines = tuple(joint_data.cell_lines)
    total = int(count)
    theta_prior = _default_theta_prior_config() if theta_prior is None else theta_prior
    for simulation_index in range(total):
        cell_index = int(rng.integers(0, len(cell_lines)))
        theta = sample_theta_norm_from_prior_config(rng, theta_prior)
        receptors_raw = np.asarray(joint_data.q_obs[cell_index], dtype=np.float32)
        receptors_for_feature = np.array(receptors_raw, copy=True)
        if receptor_noise_log_sd > 0.0:
            receptors_for_feature *= rng.lognormal(
                mean=0.0,
                sigma=float(receptor_noise_log_sd),
                size=receptors_for_feature.shape,
            ).astype("float32")
        y_raw = simulate_promisys_onestep_raw(
            bmp4_concentrations=joint_data.bmp4_conc[cell_index],
            receptors=receptors_for_feature,
            theta_norm=theta,
            receptor_noise_log_sd=0.0,
            rng=rng,
            lfiax_root=lfiax_root,
        )[0]
        y_norm = _sample_normalized_observations_from_mean_raw(
            normalizer=normalizer,
            mean_raw=y_raw,
            theta_norm=theta,
            rng=rng,
        )
        r_norm = normalizer.normalize_receptors(receptors_for_feature)
        xi_norm = normalizer.normalize_bmp4(joint_data.bmp4_conc[cell_index])
        features.append(np.concatenate([y_norm, r_norm, xi_norm]).astype("float32"))
        targets.append(theta)
        if _should_report_progress(simulation_index, total):
            _progress(f"SNPE simulations: {simulation_index + 1}/{total}")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        {
            "snpe_simulations": int(count),
            "feature_order": ["y_norm_trajectory", "R_norm", "xi_norm_trajectory"],
            "theta_names": list(THETA_NAMES),
            "complex_names": list(COMPLEX_NAMES),
            "theta_raw_prior": theta_prior,
        },
    )


def _train_gaussian_net(
    network: GaussianMLP,
    features: Any,
    targets: Any,
    *,
    steps: int,
    learning_rate: float,
    batch_size: int = DEFAULT_POSTERIOR_NET_BATCH_SIZE,
    seed: int,
    label: str,
) -> list[float]:
    import torch

    features_t = torch.as_tensor(features, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    if targets_t.ndim == 1:
        targets_t = targets_t[:, None]
    generator = torch.Generator().manual_seed(int(seed))
    optimizer = torch.optim.Adam(network.model.parameters(), lr=float(learning_rate))
    losses: list[float] = []
    batch_size = min(int(batch_size), features_t.shape[0])
    total_steps = max(int(steps), 0)
    for step_index in range(total_steps):
        indices = torch.randint(0, features_t.shape[0], (batch_size,), generator=generator)
        dist = network.distribution(features_t[indices])
        loss = -dist.log_prob(targets_t[indices]).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if _should_report_progress(step_index, total_steps):
            _progress(
                f"{label} fit: {step_index + 1}/{total_steps} "
                f"(loss={losses[-1]:.4f})"
            )
    if not losses:
        dist = network.distribution(features_t)
        losses.append(float((-dist.log_prob(targets_t).sum(dim=-1).mean()).detach().cpu()))
        _progress(f"{label} fit: 0 training steps requested (loss={losses[-1]:.4f})")
    return losses


def _sample_cell_line_posteriors(
    *,
    posterior_net: GaussianMLP,
    joint_data: Any,
    sample_count: int,
) -> dict[str, Any]:
    import torch

    samples: dict[str, Any] = {}
    with torch.no_grad():
        for cell_index, cell_line in enumerate(joint_data.cell_lines):
            feature = np.concatenate(
                [
                    np.asarray(joint_data.x_obs_norm[cell_index], dtype=np.float32),
                    np.asarray(joint_data.Rs_norm[cell_index], dtype=np.float32),
                    np.asarray(joint_data.bmp4_conc_norm[cell_index], dtype=np.float32),
                ]
            )
            dist = posterior_net.distribution(torch.as_tensor(feature[None, :], dtype=torch.float32))
            draws = dist.sample((int(sample_count),)).squeeze(1)
            samples[str(cell_line)] = draws.cpu().numpy().astype("float32")
            _progress(
                f"SNPE posterior samples for {cell_line}: "
                f"{samples[str(cell_line)].shape[0]}"
            )
    return samples


def _run_joint_multicontext_boed(
    *,
    joint_data: Any,
    normalizer: Bmp4Normalizer,
    snpe_samples: dict[str, Any],
    steps: int,
    outer_samples: int,
    inner_samples: int,
    flow_learning_rate: float,
    design_learning_rate: float,
    infonce_lambda: float,
    design_dist_init_std: float,
    design_temperature_scale: float,
    selector_temperature_final: float,
    receptor_noise_log_sd: float,
    flow_config: dict[str, Any],
    rng: Any,
    seed: int,
    lfiax_root: str | Path | None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> tuple[JaxLikelihoodState, dict[str, Any]]:
    _ensure_lfiax_importable(lfiax_root)
    import jax
    import jax.numpy as jnp
    import optax

    log_prob_transform = _make_jax_likelihood_transform(flow_config)
    master_key = jax.random.PRNGKey(int(seed))
    init_key, design_base_key = jax.random.split(master_key)
    flow_params = log_prob_transform.init(
        init_key,
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.zeros((1, THETA_DIM), dtype=jnp.float32),
        jnp.zeros((1, len(TARGET_RECEPTOR_NAMES) + 1), dtype=jnp.float32),
    )
    bounds = _bmp4_design_bounds(joint_data, normalizer)
    min_design_std, max_design_std = _design_std_bounds(bounds)
    init_design_std = min(max(float(design_dist_init_std), min_design_std), max_design_std)
    design_params = _clip_bmp4_design_params(
        {
            "bmp4_norm_mu": jnp.full(
                (len(joint_data.cell_lines),),
                (bounds["bmp4_norm_min"] + bounds["bmp4_norm_max"]) / 2.0,
                dtype=jnp.float32,
            ),
            "bmp4_norm_log_std": jnp.full(
                (len(joint_data.cell_lines),),
                jnp.log(jnp.asarray(init_design_std, dtype=jnp.float32)),
                dtype=jnp.float32,
            ),
            "selector_logits": jnp.zeros((len(joint_data.cell_lines),), dtype=jnp.float32),
        },
        bounds,
    )
    r_norm = jnp.asarray(joint_data.Rs_norm, dtype=jnp.float32)

    flow_optimizer = optax.adam(float(flow_learning_rate))
    flow_opt_state = flow_optimizer.init(flow_params)

    def log_prob_fn(params: Any, y: Any, theta: Any, xi: Any) -> Any:
        return log_prob_transform.apply(params, y, theta, xi)

    def loss_fn(
        params: Any,
        current_design_params: dict[str, Any],
        design_key: Any,
        batch: dict[str, Any],
        selector_temperature: Any,
    ) -> tuple[Any, dict[str, Any]]:
        return _joint_multicontext_infonce_loss(
            params,
            current_design_params,
            batch,
            design_key=design_key,
            log_prob_fn=log_prob_fn,
            bounds=bounds,
            selector_temperature=selector_temperature,
            infonce_lambda=float(infonce_lambda),
            inner_samples=int(inner_samples),
        )

    value_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    history: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    loss_history: list[float] = []
    best_by_cell = [
        {
            "cell_line": str(cell_line),
            "utility": -float("inf"),
            "bmp4_norm_mu": None,
            "bmp4_norm_std": None,
            "bmp4_norm_design": None,
            "log10_dose": None,
            "dose_mu": None,
            "dose_std": None,
            "dose": None,
            "step": None,
        }
        for cell_line in joint_data.cell_lines
    ]

    total_steps = max(int(steps), 1)
    patience = None if early_stopping_patience is None else int(early_stopping_patience)
    min_delta = max(float(early_stopping_min_delta), 0.0)
    best_objective = -float("inf")
    best_objective_step: int | None = None
    steps_without_improvement = 0
    early_stopped = False
    early_stopping_reason: str | None = None
    for step in range(total_steps):
        step_start = time.perf_counter()
        design_key = jax.random.fold_in(design_base_key, int(step + 1))
        selector_temperature = _selector_temperature(
            step_index=step,
            total_steps=total_steps,
            final_temperature=float(selector_temperature_final),
        )
        temperature_norm_std = _design_temperature_norm_std_schedule(
            normalizer=normalizer,
            bmp4_norm_mu=np.asarray(design_params["bmp4_norm_mu"], dtype=np.float32),
            step_index=step,
            total_steps=total_steps,
            final_dose_std=float(design_temperature_scale),
            bounds=bounds,
        )
        design_params = {
            "bmp4_norm_mu": design_params["bmp4_norm_mu"],
            "bmp4_norm_log_std": jnp.log(jnp.asarray(temperature_norm_std, dtype=jnp.float32)),
            "selector_logits": design_params["selector_logits"],
        }
        design_params = _clip_bmp4_design_params(design_params, bounds)
        step_design_params = {
            "bmp4_norm_mu": np.asarray(design_params["bmp4_norm_mu"], dtype=np.float32),
            "bmp4_norm_log_std": np.asarray(design_params["bmp4_norm_log_std"], dtype=np.float32),
        }
        step_selector_logits = np.asarray(design_params["selector_logits"], dtype=np.float64)
        step_bmp4_norm_mu = np.asarray(step_design_params["bmp4_norm_mu"], dtype=np.float64)
        step_bmp4_norm_std = np.asarray(
            _decode_bmp4_norm_std(step_design_params["bmp4_norm_log_std"], bounds),
            dtype=np.float64,
        )
        sampled_bmp4_norm_designs = np.asarray(
            _sample_bmp4_norm_design_distribution(
                design_params=design_params,
                design_key=design_key,
                sample_count=int(outer_samples),
                bounds=bounds,
            ),
            dtype=np.float32,
        )
        simulation_start = time.perf_counter()
        batch_np, sim_summary = _simulate_joint_boed_batch(
            joint_data=joint_data,
            normalizer=normalizer,
            snpe_samples=snpe_samples,
            bmp4_norm_design_samples=sampled_bmp4_norm_designs,
            outer_samples=int(outer_samples),
            receptor_noise_log_sd=float(receptor_noise_log_sd),
            rng=rng,
            lfiax_root=lfiax_root,
        )
        simulation_seconds = time.perf_counter() - simulation_start
        prep_start = time.perf_counter()
        batch = {
            "theta": jnp.asarray(batch_np["theta"], dtype=jnp.float32),
            "y": jnp.asarray(batch_np["y"], dtype=jnp.float32),
            "r_norm": r_norm,
        }
        prep_seconds = time.perf_counter() - prep_start
        grad_start = time.perf_counter()
        (loss, aux), (flow_grads, design_grads) = value_and_grad(
            flow_params,
            design_params,
            design_key,
            batch,
            jnp.asarray(selector_temperature, dtype=jnp.float32),
        )
        loss.block_until_ready()
        grad_seconds = time.perf_counter() - grad_start
        mu_gradients_np = np.asarray(design_grads["bmp4_norm_mu"], dtype=np.float64)
        std_gradients_np = np.asarray(design_grads["bmp4_norm_log_std"], dtype=np.float64)
        selector_gradients_np = np.asarray(design_grads["selector_logits"], dtype=np.float64)
        update_start = time.perf_counter()
        flow_updates, flow_opt_state = flow_optimizer.update(flow_grads, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)
        design_params = _apply_design_sgd_update(
            design_params,
            design_grads,
            bounds,
            learning_rate=float(design_learning_rate),
        )
        design_params["bmp4_norm_mu"].block_until_ready()
        update_seconds = time.perf_counter() - update_start

        utilities = np.asarray(aux["utilities"], dtype=np.float64)
        current_dose_samples = np.asarray(
            _decode_bmp4_norm_designs(normalizer, sampled_bmp4_norm_designs),
            dtype=np.float64,
        )
        current_dose_mu = np.asarray(np.mean(current_dose_samples, axis=1), dtype=np.float64)
        current_dose_std = np.asarray(np.std(current_dose_samples, axis=1), dtype=np.float64)
        temperature_dose_std = float(np.mean(current_dose_std))
        current_log10 = np.log10(np.clip(current_dose_mu, 1e-30, None))
        selector_probs_np = np.asarray(aux["selector_probs"], dtype=np.float64)
        selected_index = int(np.argmax(selector_probs_np))
        selector_probs = selector_probs_np.tolist()
        objective = float(aux["objective"])
        improved = objective > best_objective + min_delta
        if improved:
            best_objective = objective
            best_objective_step = int(step + 1)
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        loss_value = float(loss)
        loss_history.append(loss_value)
        total_seconds = time.perf_counter() - step_start
        timing_record = {
            "simulate": float(simulation_seconds),
            "prepare": float(prep_seconds),
            "grad": float(grad_seconds),
            "update": float(update_seconds),
            "total": float(total_seconds),
        }
        diagnostic_record = {
            "step": int(step + 1),
            "phi_grad_norm": _tree_l2_norm(flow_grads),
            "mu_gradients": mu_gradients_np.tolist(),
            "std_gradients": std_gradients_np.tolist(),
            "selector_gradients": selector_gradients_np.tolist(),
            "mu_grad_norm": float(np.linalg.norm(mu_gradients_np)),
            "std_grad_norm": float(np.linalg.norm(std_gradients_np)),
            "selector_grad_norm": float(np.linalg.norm(selector_gradients_np)),
            "design_update": "sgd_mean_and_selector",
            "selector_temperature": float(selector_temperature),
            "design_temperature": float(temperature_dose_std),
            "design_temperature_dose_std": float(temperature_dose_std),
            "design_temperature_norm_std_mean": float(np.mean(step_bmp4_norm_std)),
            "timings_seconds": timing_record,
        }
        diagnostics.append(diagnostic_record)
        for cell_index, utility in enumerate(utilities):
            if float(utility) > float(best_by_cell[cell_index]["utility"]):
                best_by_cell[cell_index] = {
                    "cell_line": str(joint_data.cell_lines[cell_index]),
                    "utility": float(utility),
                    "bmp4_norm_mu": float(step_bmp4_norm_mu[cell_index]),
                    "bmp4_norm_std": float(step_bmp4_norm_std[cell_index]),
                    "bmp4_norm_design": float(step_bmp4_norm_mu[cell_index]),
                    "log10_dose": float(current_log10[cell_index]),
                    "dose_mu": float(current_dose_mu[cell_index]),
                    "dose_std": float(current_dose_std[cell_index]),
                    "dose": float(current_dose_mu[cell_index]),
                    "step": int(step),
                }
        record = {
            "step": float(step + 1),
            "objective": objective,
            "loss": loss_value,
            "eig": objective,
            "per_cell_utilities": {
                str(cell_line): float(utilities[index])
                for index, cell_line in enumerate(joint_data.cell_lines)
            },
            "per_cell_bmp4_norm_mu": {
                str(cell_line): float(step_bmp4_norm_mu[index])
                for index, cell_line in enumerate(joint_data.cell_lines)
            },
            "per_cell_bmp4_norm_std": {
                str(cell_line): float(step_bmp4_norm_std[index])
                for index, cell_line in enumerate(joint_data.cell_lines)
            },
            "per_cell_bmp4_norm_designs": {
                str(cell_line): float(step_bmp4_norm_mu[index])
                for index, cell_line in enumerate(joint_data.cell_lines)
            },
            "per_cell_log10_doses": {
                str(cell_line): float(current_log10[index])
                for index, cell_line in enumerate(joint_data.cell_lines)
            },
            "per_cell_dose_mu": {
                str(cell_line): float(current_dose_mu[index])
                for index, cell_line in enumerate(joint_data.cell_lines)
            },
            "per_cell_dose_std": {
                str(cell_line): float(current_dose_std[index])
                for index, cell_line in enumerate(joint_data.cell_lines)
            },
            "per_cell_doses": {
                str(cell_line): float(current_dose_mu[index])
                for index, cell_line in enumerate(joint_data.cell_lines)
            },
            "cell_line_names": list(joint_data.cell_lines),
            "selector_probs": selector_probs,
            "selector_temperature": float(selector_temperature),
            "selector_logits": step_selector_logits.tolist(),
            "selected_cell_line_index": selected_index,
            "selected_cell_line": str(joint_data.cell_lines[selected_index]),
            "best_objective_so_far": float(best_objective),
            "best_objective_step": best_objective_step,
            "steps_without_eig_improvement": int(steps_without_improvement),
            "early_stopping_patience": patience,
            "early_stopping_min_delta": float(min_delta),
            "dose": float(current_dose_mu[selected_index]),
            "dose_mu": float(current_dose_mu[selected_index]),
            "dose_std": float(current_dose_std[selected_index]),
            "doses": [float(current_dose_mu[selected_index])],
            "bmp4_norm_design": float(step_bmp4_norm_mu[selected_index]),
            "bmp4_norm_mu": float(step_bmp4_norm_mu[selected_index]),
            "bmp4_norm_std": float(step_bmp4_norm_std[selected_index]),
            "bmp4_norm_designs": [float(step_bmp4_norm_mu[selected_index])],
            "log10_dose": float(current_log10[selected_index]),
            "log10_doses": [float(current_log10[selected_index])],
            "design": float(current_dose_mu[selected_index]),
            "simulated_y_mean_by_cell": sim_summary["y_mean_by_cell"],
            "timings_seconds": timing_record,
        }
        history.append(record)

        if _should_report_progress(step, total_steps, interval=BOED_PROGRESS_PRINT_INTERVAL):
            utilities_text = ", ".join(
                f"{cell_line}={utilities[index]:.3f}"
                for index, cell_line in enumerate(joint_data.cell_lines)
            )
            dose_mu_text = ", ".join(
                f"{cell_line}={current_dose_mu[index]:.3g}"
                for index, cell_line in enumerate(joint_data.cell_lines)
            )
            dose_std_text = ", ".join(
                f"{cell_line}={current_dose_std[index]:.3g}"
                for index, cell_line in enumerate(joint_data.cell_lines)
            )
            selector_text = ", ".join(
                f"{cell_line}={selector_probs[index]:.3f}"
                for index, cell_line in enumerate(joint_data.cell_lines)
            )
            _progress(
                f"Joint BOED: {step + 1}/{total_steps} "
                f"(J={objective:.4f}, selected={record['selected_cell_line']}, "
                f"dose_mu={record['dose_mu']:.6g}, dose_std={record['dose_std']:.3g}, "
                f"U=[{utilities_text}], selector=[{selector_text}], "
                f"dose_mu=[{dose_mu_text}], dose_std=[{dose_std_text}], "
                f"grad_norms=phi:{diagnostic_record['phi_grad_norm']:.3g} "
                f"mu:{diagnostic_record['mu_grad_norm']:.3g} "
                f"std:{diagnostic_record['std_grad_norm']:.3g} "
                f"selector:{diagnostic_record['selector_grad_norm']:.3g}, "
                f"t=sim:{simulation_seconds:.2f}s prep:{prep_seconds:.2f}s "
                f"grad:{grad_seconds:.2f}s update:{update_seconds:.2f}s total:{total_seconds:.2f}s)"
            )
        if patience is not None and patience > 0 and steps_without_improvement >= patience:
            early_stopped = True
            early_stopping_reason = (
                f"no EIG improvement greater than {min_delta:g} for {patience} steps"
            )
            _progress(
                f"Joint BOED early stopping at step {step + 1}/{total_steps}: "
                f"{early_stopping_reason}"
            )
            break

    best_record = history[-1] if history else {}
    best_index = int(best_record.get("selected_cell_line_index", 0))
    best_selector = list(best_record.get("selector_probs", [1.0 / len(joint_data.cell_lines)] * len(joint_data.cell_lines)))
    final_best = {
        "cell_line": str(joint_data.cell_lines[best_index]),
        "utility": float(best_record.get("per_cell_utilities", {}).get(str(joint_data.cell_lines[best_index]), best_record.get("objective", float("nan")))),
        "objective": float(best_record.get("objective", float("nan"))),
        "bmp4_norm_mu": float(best_record.get("per_cell_bmp4_norm_mu", {}).get(str(joint_data.cell_lines[best_index]), best_record.get("bmp4_norm_mu", float("nan")))),
        "bmp4_norm_std": float(best_record.get("per_cell_bmp4_norm_std", {}).get(str(joint_data.cell_lines[best_index]), best_record.get("bmp4_norm_std", float("nan")))),
        "log10_dose": float(best_record.get("per_cell_log10_doses", {}).get(str(joint_data.cell_lines[best_index]), best_record.get("log10_dose", float("nan")))),
        "dose_mu": float(best_record.get("per_cell_dose_mu", {}).get(str(joint_data.cell_lines[best_index]), best_record.get("dose_mu", float("nan")))),
        "dose_std": float(best_record.get("per_cell_dose_std", {}).get(str(joint_data.cell_lines[best_index]), best_record.get("dose_std", float("nan")))),
    }
    eig_result = {
        "history": history,
        "best_design": float(final_best["dose_mu"]),
        "best_dose": float(final_best["dose_mu"]),
        "best_dose_mu": float(final_best["dose_mu"]),
        "best_dose_std": float(final_best["dose_std"]),
        "best_doses": [float(final_best["dose_mu"])],
        "best_bmp4_norm_design": float(final_best["bmp4_norm_mu"]),
        "best_bmp4_norm_mu": float(final_best["bmp4_norm_mu"]),
        "best_bmp4_norm_std": float(final_best["bmp4_norm_std"]),
        "best_bmp4_norm_designs": [float(final_best["bmp4_norm_mu"])],
        "best_log10_dose": float(final_best["log10_dose"]),
        "best_log10_doses": [float(final_best["log10_dose"])],
        "best_cell_line": str(final_best["cell_line"]),
        "best_cell_line_index": best_index,
        "best_selector_probs": best_selector,
        "final_selector_probs": best_selector,
        "best_eig": float(final_best["objective"]),
        "best_utility": float(final_best["utility"]),
        "best_per_cell": best_by_cell,
        "final_next_experiment": {
            "cell_line": str(final_best["cell_line"]),
            "cell_line_index": best_index,
            "bmp4_norm_design": float(final_best["bmp4_norm_mu"]),
            "bmp4_norm_mu": float(final_best["bmp4_norm_mu"]),
            "bmp4_norm_std": float(final_best["bmp4_norm_std"]),
            "log10_dose": float(final_best["log10_dose"]),
            "dose": float(final_best["dose_mu"]),
            "dose_mu": float(final_best["dose_mu"]),
            "dose_std": float(final_best["dose_std"]),
            "utility": float(final_best["utility"]),
            "objective": float(final_best["objective"]),
            "selector_probs": best_selector,
        },
        "design_type": "cell_line_selector_plus_bmp4_dose",
        "dose_count": 1,
        "estimator": "lf_pce_infonce",
        "objective": "fixed_posterior_multi_context_lfiax",
        "fit_steps_requested_for_joint_boed": int(steps),
        "fit_steps_used_for_joint_boed": len(history),
        "early_stopping": {
            "enabled": bool(patience is not None and patience > 0),
            "patience": patience,
            "min_delta": float(min_delta),
            "stopped": bool(early_stopped),
            "reason": early_stopping_reason,
            "best_objective": float(best_objective),
            "best_objective_step": best_objective_step,
            "steps_without_improvement": int(steps_without_improvement),
        },
        "eig_steps_ignored_for_promisys_onestep": True,
        "infonce_lambda": float(infonce_lambda),
        "infonce_negatives": int(inner_samples),
        "design_temperature_scale": float(design_temperature_scale),
        "design_temperature_schedule": "geometric_norm_std_anneal_initial_1_to_final_raw_dose_std_scale",
        "design_temperature_units": "raw BMP4 concentration",
        "design_temperature_initial_norm_std": 1.0,
        "selector_optimization": "softmax_logits",
        "selector_temperature_initial": 1.0,
        "selector_temperature_final": float(selector_temperature_final),
        "selector_temperature_schedule": "geometric_decay_to_final",
        "initial_selector_probs": {
            str(cell_line): float(1.0 / len(joint_data.cell_lines))
            for cell_line in joint_data.cell_lines
        },
        "gradient_diagnostics": diagnostics,
        "optimization_bounds": {
            "bmp4_norm_min": float(bounds["bmp4_norm_min"]),
            "bmp4_norm_max": float(bounds["bmp4_norm_max"]),
            "log10_dose_min": float(bounds["log10_dose_min"]),
            "log10_dose_max": float(bounds["log10_dose_max"]),
            "dose_min": float(bounds["dose_min"]),
            "dose_max": float(bounds["dose_max"]),
        },
    }
    likelihood_state = JaxLikelihoodState(
        flow_params=flow_params,
        flow_config=dict(flow_config),
        bmp4_norm_mu=np.asarray(design_params["bmp4_norm_mu"], dtype=np.float32),
        bmp4_norm_log_std=np.asarray(design_params["bmp4_norm_log_std"], dtype=np.float32),
        selector_logits=np.asarray(design_params["selector_logits"], dtype=np.float32),
        selector_temperature=float(history[-1].get("selector_temperature", selector_temperature_final)),
        history=history,
        loss_history=loss_history,
        gradient_diagnostics=diagnostics,
        infonce_lambda=float(infonce_lambda),
        infonce_negatives=int(inner_samples),
    )
    return likelihood_state, eig_result


def _simulate_joint_boed_batch(
    *,
    joint_data: Any,
    normalizer: Bmp4Normalizer,
    snpe_samples: dict[str, Any],
    bmp4_norm_design_samples: Any,
    outer_samples: int,
    receptor_noise_log_sd: float,
    rng: Any,
    lfiax_root: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    theta_by_cell: list[Any] = []
    y_by_cell: list[Any] = []
    y_means: dict[str, float] = {}
    sample_count = max(int(outer_samples), 1)
    for cell_index, cell_line in enumerate(joint_data.cell_lines):
        posterior = np.asarray(snpe_samples[str(cell_line)], dtype=np.float32)
        indices = rng.choice(posterior.shape[0], size=sample_count, replace=posterior.shape[0] < sample_count)
        theta = posterior[indices]
        design_samples = np.asarray(bmp4_norm_design_samples[cell_index], dtype=np.float32)
        dose = np.asarray(_decode_bmp4_norm_designs(normalizer, design_samples), dtype=np.float32)
        receptors_raw = np.asarray(joint_data.q_obs[cell_index], dtype=np.float32)
        y_raw = simulate_promisys_onestep_paired_raw(
            bmp4_concentrations=dose,
            receptors=receptors_raw,
            theta_norm=theta,
            receptor_noise_log_sd=float(receptor_noise_log_sd),
            rng=rng,
            lfiax_root=lfiax_root,
        )
        y_norm = _sample_normalized_observations_from_mean_raw(
            normalizer=normalizer,
            mean_raw=y_raw,
            theta_norm=theta,
            rng=rng,
        ).reshape(-1, 1)
        theta_by_cell.append(theta.astype("float32"))
        y_by_cell.append(y_norm.astype("float32"))
        y_means[str(cell_line)] = float(np.mean(y_raw))
    return (
        {
            "theta": np.stack(theta_by_cell, axis=0).astype("float32"),
            "y": np.stack(y_by_cell, axis=0).astype("float32"),
        },
        {"y_mean_by_cell": y_means},
    )


def _joint_multicontext_infonce_loss(
    flow_params: Any,
    design_params: dict[str, Any],
    batch: dict[str, Any],
    *,
    design_key: Any,
    log_prob_fn: Any,
    bounds: dict[str, float],
    selector_temperature: Any,
    infonce_lambda: float,
    inner_samples: int,
) -> tuple[Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    utilities = _joint_multicontext_utilities(
        flow_params,
        design_params,
        batch,
        design_key=design_key,
        log_prob_fn=log_prob_fn,
        bounds=bounds,
        infonce_lambda=infonce_lambda,
        inner_samples=inner_samples,
    )
    temperature = jnp.maximum(jnp.asarray(selector_temperature, dtype=jnp.float32), 1e-6)
    selector_probs = jax.nn.softmax(jnp.asarray(design_params["selector_logits"], dtype=jnp.float32) / temperature)
    objective = jnp.sum(selector_probs * utilities)
    return -objective, {
        "objective": objective,
        "utilities": utilities,
        "selector_probs": selector_probs,
    }


def _joint_multicontext_utilities(
    flow_params: Any,
    design_params: dict[str, Any],
    batch: dict[str, Any],
    *,
    design_key: Any,
    log_prob_fn: Any,
    bounds: dict[str, float],
    infonce_lambda: float,
    inner_samples: int,
) -> Any:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    theta_by_cell = batch["theta"]
    y_by_cell = batch["y"]
    r_norm_by_cell = batch["r_norm"]
    bmp4_norm_design_samples = _sample_bmp4_norm_design_distribution(
        design_params=design_params,
        design_key=design_key,
        sample_count=theta_by_cell.shape[1],
        bounds=bounds,
    )

    def utility_one(theta: Any, y: Any, r_norm: Any, bmp4_norm_design: Any) -> Any:
        sample_count = theta.shape[0]
        r_broadcast = jnp.broadcast_to(r_norm[None, :], (sample_count, r_norm.shape[0]))
        xi = jnp.concatenate([r_broadcast, bmp4_norm_design[:, None]], axis=1)
        conditional_lp = log_prob_fn(flow_params, y, theta, xi)
        contrastive_lps = [conditional_lp]
        for shift in range(int(inner_samples)):
            theta_negative = jnp.roll(theta, shift + 1, axis=0)
            contrastive_lps.append(log_prob_fn(flow_params, y, theta_negative, xi))
        stacked = jnp.stack(contrastive_lps, axis=0)
        marginal_lp = logsumexp(stacked, axis=0) - jnp.log(stacked.shape[0])
        eig = jnp.mean(conditional_lp - marginal_lp)
        return eig

    return jax.vmap(utility_one, in_axes=(0, 0, 0, 0))(
        theta_by_cell,
        y_by_cell,
        r_norm_by_cell,
        bmp4_norm_design_samples,
    )


def _ensure_lfiax_importable(lfiax_root: str | Path | None) -> None:
    if lfiax_root is not None:
        root = Path(lfiax_root)
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


def _make_jax_likelihood_transform(flow_config: dict[str, Any]) -> Any:
    _ensure_lfiax_importable(DEFAULT_LFIAX_ROOT)
    import haiku as hk
    import jax
    from lfiax.flows.nsf import make_nsf

    def log_prob(y: Any, theta: Any, xi: Any) -> Any:
        model = make_nsf(**flow_config)
        return model.log_prob(y, theta, xi)

    transform = hk.transform(log_prob)

    class _DeterministicApplyTransform:
        def init(self, key: Any, *args: Any) -> Any:
            return transform.init(key, *args)

        def apply(self, params: Any, *args: Any) -> Any:
            return transform.apply(params, jax.random.PRNGKey(0), *args)

    return _DeterministicApplyTransform()


def _tree_l2_norm(tree: Any) -> float:
    import jax
    import jax.numpy as jnp

    leaves = [jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(tree) if leaf is not None]
    if not leaves:
        return 0.0
    total = jnp.sum(jnp.concatenate([leaf * leaf for leaf in leaves]))
    return float(jnp.sqrt(total))


def _jax_tree_to_numpy(tree: Any) -> Any:
    import jax

    return jax.tree_util.tree_map(lambda value: np.asarray(value), tree)


def _run_all_cell_line_mcmc(
    *,
    joint_data: Any,
    likelihood_state: JaxLikelihoodState,
    snpe_samples: dict[str, Any],
    warmup: int,
    sample_count: int,
    proposal_scale: float,
    prior_std_floor: float,
    rng: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for cell_index, cell_line in enumerate(joint_data.cell_lines):
        _progress(f"MCMC for {cell_line}: starting")
        output[str(cell_line)] = _run_cell_line_mcmc(
            likelihood_state=likelihood_state,
            prior_samples=np.asarray(snpe_samples[str(cell_line)], dtype=np.float32),
            y_norm=np.asarray(joint_data.x_obs_norm[cell_index], dtype=np.float32),
            r_norm=np.asarray(joint_data.Rs_norm[cell_index], dtype=np.float32),
            xi_norm=np.asarray(joint_data.bmp4_conc_norm[cell_index], dtype=np.float32),
            warmup=warmup,
            sample_count=sample_count,
            proposal_scale=proposal_scale,
            prior_std_floor=prior_std_floor,
            rng=rng,
            label=str(cell_line),
        )
        _progress(f"MCMC for {cell_line}: kept {output[str(cell_line)].shape[0]} samples")
    return output


def _run_cell_line_mcmc(
    *,
    likelihood_state: JaxLikelihoodState,
    prior_samples: Any,
    y_norm: Any,
    r_norm: Any,
    xi_norm: Any,
    warmup: int,
    sample_count: int,
    proposal_scale: float,
    prior_std_floor: float,
    rng: Any,
    label: str,
) -> Any:
    log_prob_apply = _make_jax_likelihood_transform(likelihood_state.flow_config).apply
    prior_mean = np.mean(prior_samples, axis=0)
    prior_std = np.std(prior_samples, axis=0) + float(prior_std_floor)
    current = prior_mean.copy().astype("float32")
    current_lp = _mcmc_log_prob(
        likelihood_state,
        log_prob_apply,
        current,
        prior_mean,
        prior_std,
        y_norm,
        r_norm,
        xi_norm,
    )
    kept: list[Any] = []
    total = max(int(warmup), 0) + max(int(sample_count), 1)
    for step in range(total):
        proposed = current + rng.normal(0.0, float(proposal_scale), size=current.shape).astype("float32")
        proposed_lp = _mcmc_log_prob(
            likelihood_state,
            log_prob_apply,
            proposed,
            prior_mean,
            prior_std,
            y_norm,
            r_norm,
            xi_norm,
        )
        if math.log(float(rng.uniform())) < proposed_lp - current_lp:
            current = proposed
            current_lp = proposed_lp
        if step >= warmup:
            kept.append(current.copy())
        if _should_report_progress(step, total):
            phase = "warmup" if step < warmup else "sampling"
            _progress(
                f"MCMC {label}: {step + 1}/{total} "
                f"({phase}, kept={len(kept)})"
            )
    return np.asarray(kept, dtype=np.float32)


def _mcmc_log_prob(
    likelihood_state: JaxLikelihoodState,
    log_prob_apply: Any,
    theta: Any,
    prior_mean: Any,
    prior_std: Any,
    y_norm: Any,
    r_norm: Any,
    xi_norm: Any,
) -> float:
    import jax.numpy as jnp

    theta_np = np.asarray(theta, dtype=np.float32)
    if not np.all(np.isfinite(theta_np)):
        return -float("inf")
    z = (theta_np - prior_mean) / prior_std
    prior_lp = -0.5 * float(np.sum(z * z + np.log(2.0 * np.pi * prior_std * prior_std)))
    theta_t = jnp.asarray(np.repeat(theta_np[None, :], len(y_norm), axis=0), dtype=jnp.float32)
    r_np = np.asarray(r_norm, dtype=np.float32).reshape(1, -1)
    xi_np = np.asarray(xi_norm, dtype=np.float32).reshape(-1, 1)
    xi_t = jnp.asarray(np.concatenate([np.repeat(r_np, len(y_norm), axis=0), xi_np], axis=1), dtype=jnp.float32)
    y_t = jnp.asarray(np.asarray(y_norm, dtype=np.float32).reshape(-1, 1), dtype=jnp.float32)
    ll = log_prob_apply(likelihood_state.flow_params, y_t, theta_t, xi_t).sum()
    return prior_lp + float(ll)


def _posterior_predictive_from_likelihood(
    *,
    joint_data: Any,
    normalizer: Bmp4Normalizer,
    mcmc_samples_by_cell_line: dict[str, Any],
    grid_points: int,
    rng: Any,
    lfiax_root: str | Path | None,
) -> dict[str, Any]:
    _ = normalizer
    lower, upper = _log10_dose_bounds(joint_data)
    grid = np.geomspace(10.0**lower, 10.0**upper, int(grid_points)).astype("float32")
    draws_by_cell: list[Any] = []
    mu_by_cell: list[Any] = []
    for cell_index, cell_line in enumerate(joint_data.cell_lines):
        theta_samples = np.asarray(mcmc_samples_by_cell_line[str(cell_line)], dtype=np.float32)
        if theta_samples.shape[0] == 0:
            theta_samples = np.asarray([np.zeros(THETA_DIM, dtype=np.float32)])
        chosen = theta_samples[
            rng.choice(theta_samples.shape[0], size=min(theta_samples.shape[0], 128), replace=False)
        ]
        _progress(
            f"Posterior predictive for {cell_line}: "
            f"{chosen.shape[0]} theta draws x {len(grid)} doses"
        )
        cell_draws: list[Any] = []
        cell_mu: list[Any] = []
        for theta_index, theta in enumerate(chosen):
            mu_raw = simulate_promisys_onestep_raw(
                bmp4_concentrations=grid,
                receptors=np.asarray(joint_data.q_obs[cell_index], dtype=np.float32),
                theta_norm=theta,
                receptor_noise_log_sd=0.0,
                rng=rng,
                lfiax_root=lfiax_root,
            )[0]
            y_raw = _sample_raw_observations_from_mean_raw(
                normalizer=normalizer,
                mean_raw=mu_raw,
                theta_norm=theta,
                rng=rng,
            )
            cell_draws.append(y_raw.astype("float32"))
            cell_mu.append(np.asarray(mu_raw, dtype=np.float32))
            if _should_report_progress(theta_index, len(chosen)):
                _progress(
                    f"Posterior predictive {cell_line}: "
                    f"{theta_index + 1}/{len(chosen)} theta draws"
                )
        cell_draws_arr = np.asarray(cell_draws, dtype=np.float32)
        cell_mu_arr = np.asarray(cell_mu, dtype=np.float32)
        draws_by_cell.append(cell_draws_arr)
        mu_by_cell.append(cell_mu_arr)
    return {
        "grid_concentration": np.repeat(grid[None, :], len(joint_data.cell_lines), axis=0),
        "predictive_y": np.stack(draws_by_cell, axis=1),
        "predictive_mu": np.stack(mu_by_cell, axis=1),
    }


def _selector_temperature(
    *,
    step_index: int,
    total_steps: int,
    final_temperature: float,
    initial_temperature: float = 1.0,
) -> float:
    initial = max(float(initial_temperature), 1e-6)
    final = max(float(final_temperature), 1e-6)
    if int(total_steps) <= 1:
        return final
    progress = min(max(int(step_index) / max(int(total_steps) - 1, 1), 0.0), 1.0)
    return float(initial * (final / initial) ** progress)


def _selector_probs_from_logits(selector_logits: Any, *, selector_temperature: float = 1.0) -> Any:
    logits = np.asarray(selector_logits, dtype=np.float64)
    temperature = max(float(selector_temperature), 1e-6)
    shifted = logits / temperature
    shifted = shifted - np.max(shifted)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float32)


def _estimate_snpe_prior_eig_baseline(
    *,
    posterior_net: GaussianMLP,
    joint_data: Any,
    snpe_samples: dict[str, Any],
    theta_prior: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not theta_prior or theta_prior.get("mode") != "expert_mapped":
        return None
    if int(theta_prior.get("mapped_parameter_count") or 0) <= 0:
        return None

    import torch

    by_cell: dict[str, Any] = {}
    pooled: list[np.ndarray] = []
    with torch.no_grad():
        for cell_index, cell_line in enumerate(joint_data.cell_lines):
            cell_key = str(cell_line)
            theta = torch.as_tensor(snpe_samples[cell_key], dtype=torch.float32)
            feature = np.concatenate(
                [
                    np.asarray(joint_data.x_obs_norm[cell_index], dtype=np.float32),
                    np.asarray(joint_data.Rs_norm[cell_index], dtype=np.float32),
                    np.asarray(joint_data.bmp4_conc_norm[cell_index], dtype=np.float32),
                ]
            )
            dist = posterior_net.distribution(torch.as_tensor(feature[None, :], dtype=torch.float32))
            log_posterior = dist.log_prob(theta).sum(dim=-1).cpu().numpy()
            log_prior = _theta_norm_prior_log_prob(theta.cpu().numpy(), theta_prior)
            log_ratio = log_posterior - log_prior
            pooled.append(log_ratio)
            by_cell[cell_key] = {
                "mean": float(np.mean(log_ratio)),
                "stderr": float(np.std(log_ratio) / math.sqrt(max(log_ratio.size, 1))),
                "n": int(log_ratio.size),
                "mean_log_posterior": float(np.mean(log_posterior)),
                "mean_log_prior": float(np.mean(log_prior)),
            }

    if not pooled:
        return None
    pooled_ratio = np.concatenate(pooled, axis=0)
    return {
        "enabled": True,
        "estimator": "mean_SNPE_posterior_log_q_minus_log_prior",
        "units": "nats",
        "mean": float(np.mean(pooled_ratio)),
        "stderr": float(np.std(pooled_ratio) / math.sqrt(max(pooled_ratio.size, 1))),
        "n": int(pooled_ratio.size),
        "by_cell": by_cell,
        "theta_prior_mode": str(theta_prior.get("mode")),
        "mapped_parameter_count": int(theta_prior.get("mapped_parameter_count") or 0),
    }


def _theta_norm_prior_log_prob(theta_norm: Any, theta_prior: dict[str, Any]) -> Any:
    theta = np.asarray(theta_norm, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.shape[-1] != THETA_DIM:
        raise ValueError(f"Expected theta dimension {THETA_DIM}, got {theta.shape[-1]}.")

    parameter_priors = theta_prior.get("parameter_priors") or [
        _default_biophysical_parameter_prior(name) for name in BIOPHYSICAL_PARAMETER_NAMES
    ]
    if len(parameter_priors) != BIOPHYSICAL_THETA_DIM:
        raise ValueError(
            f"Expected {BIOPHYSICAL_THETA_DIM} biophysical prior entries, got {len(parameter_priors)}."
        )

    logits = np.clip(theta[:, :BIOPHYSICAL_THETA_DIM] * THETA_NORMALIZATION_SCALE, -40.0, 40.0)
    unit = 1.0 / (1.0 + np.exp(-logits))
    log_low = math.log(RAW_PARAMETER_LOW)
    log_high = math.log(RAW_PARAMETER_HIGH)
    default_delta = log_high - log_low
    raw = np.exp(log_low + default_delta * unit)
    log_prior = np.zeros(theta.shape[0], dtype=np.float64)
    for index, prior in enumerate(parameter_priors):
        distribution = str(prior.get("distribution", "LogUniform")).replace("_", "").replace("-", "").lower()
        raw_i = np.clip(raw[:, index], RAW_PARAMETER_LOW, RAW_PARAMETER_HIGH)
        log_jacobian = (
            np.log(raw_i)
            + math.log(default_delta * THETA_NORMALIZATION_SCALE)
            + np.log(np.clip(unit[:, index], 1e-300, 1.0))
            + np.log(np.clip(1.0 - unit[:, index], 1e-300, 1.0))
        )
        if distribution == "lognormal":
            params = prior.get("params", {})
            loc = float(params.get("loc", params.get("mu", 0.0)))
            scale = max(float(params.get("scale", params.get("sigma", 1.0))), 1e-12)
            log_raw = np.log(np.clip(raw_i, 1e-300, None))
            log_raw_density = (
                -log_raw
                - math.log(scale)
                - 0.5 * math.log(2.0 * math.pi)
                - 0.5 * ((log_raw - loc) / scale) ** 2
            )
        elif distribution == "loguniform":
            low = max(float(prior.get("low", prior.get("raw_low", RAW_PARAMETER_LOW))), RAW_PARAMETER_LOW)
            high = max(float(prior.get("high", prior.get("raw_high", RAW_PARAMETER_HIGH))), low * (1.0 + 1e-12))
            log_raw_density = -np.log(np.clip(raw_i, 1e-300, None)) - math.log(math.log(high) - math.log(low))
        else:
            raise ValueError(f"Unsupported theta prior distribution: {prior.get('distribution')!r}")
        log_prior += log_raw_density + log_jacobian

    sigma_latent = theta[:, BIOPHYSICAL_THETA_DIM]
    log_prior += -0.5 * (sigma_latent**2 + math.log(2.0 * math.pi))
    return log_prior


def _add_prior_eig_baseline(eig_result: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if not baseline:
        return eig_result
    prior_eig = float(baseline["mean"])
    eig_result["prior_eig_baseline"] = baseline
    for item in eig_result.get("history", []):
        incremental = float(item["eig"])
        item["eig_incremental"] = incremental
        item["prior_eig_baseline"] = prior_eig
        item["estimated_total_eig"] = incremental + prior_eig
        if "objective" in item:
            item["objective_incremental"] = float(item["objective"])
            item["estimated_total_objective"] = float(item["objective"]) + prior_eig
    if "best_eig" in eig_result:
        eig_result["best_eig_incremental"] = float(eig_result["best_eig"])
        eig_result["best_eig"] = float(eig_result["best_eig"]) + prior_eig
    if "best_utility" in eig_result:
        eig_result["best_utility_incremental"] = float(eig_result["best_utility"])
        eig_result["best_utility"] = float(eig_result["best_utility"]) + prior_eig
    if isinstance(eig_result.get("final_next_experiment"), dict):
        final = eig_result["final_next_experiment"]
        if "objective" in final:
            final["objective_incremental"] = float(final["objective"])
            final["objective"] = float(final["objective"]) + prior_eig
        if "utility" in final:
            final["utility_incremental"] = float(final["utility"])
            final["utility"] = float(final["utility"]) + prior_eig
    return eig_result


def _write_promisys_artifacts(
    *,
    run_path: Path,
    normalizer: Bmp4Normalizer,
    posterior_net: GaussianMLP,
    likelihood_state: JaxLikelihoodState,
    joint_data: Any,
    snpe_samples: dict[str, Any],
    mcmc_samples_by_cell_line: dict[str, Any],
    posterior_predictive: dict[str, Any],
    eig_result: dict[str, Any],
    snpe_loss_history: list[float],
    likelihood_loss_history: list[float],
    simulation_summary: dict[str, Any],
    normalizer_data_dir: str | Path,
    receptor_noise_log_sd: float,
    eig_steps: int,
    requested_hyperparams: dict[str, Any] | None,
    effective_hyperparams: dict[str, Any],
) -> dict[str, str]:
    import torch
    from examples.cases.bmp4_gradient.plotting import (
        save_eig_optimization_plot,
        save_joint_posterior_predictive_plot,
    )

    snpe_path = run_path / "snpe_posterior_samples.pt"
    posterior_samples_path = run_path / "posterior_samples.pt"
    mcmc_path = run_path / "mcmc_posterior_samples.pt"
    posterior_predictive_path = run_path / "posterior_predictive.pt"
    likelihood_checkpoint_path = run_path / "likelihood_checkpoint.pkl"
    snpe_fit_summary_path = run_path / "snpe_fit_summary.json"
    fit_summary_path = run_path / "fit_summary.json"
    eig_summary_path = run_path / "eig_optimization_summary.json"
    posterior_plot_path = run_path / "posterior_predictive.png"
    prior_plot_path = run_path / "prior_posterior_comparison.png"
    prior_positive_plot_path = run_path / "prior_posterior_comparison_positive.png"
    eig_plot_path = run_path / "eig_optimization.png"

    snpe_payload = _samples_payload(
        snpe_samples,
        theta_prior=simulation_summary.get("theta_raw_prior"),
    )
    mcmc_payload = _samples_payload(
        mcmc_samples_by_cell_line,
        theta_prior=simulation_summary.get("theta_raw_prior"),
    )
    _progress("Writing posterior sample tensors")
    torch.save(snpe_payload, snpe_path)
    torch.save(snpe_payload, posterior_samples_path)
    torch.save(mcmc_payload, mcmc_path)
    _progress("Writing posterior predictive tensor")
    torch.save(
        {
            "cell_lines": list(joint_data.cell_lines),
            "grid_concentration": torch.as_tensor(posterior_predictive["grid_concentration"]),
            "predictive_mu": torch.as_tensor(posterior_predictive["predictive_mu"]),
            "predictive_y": torch.as_tensor(posterior_predictive["predictive_y"]),
        },
        posterior_predictive_path,
    )
    _progress("Writing likelihood checkpoint")
    import jax

    bounds = _bmp4_design_bounds(joint_data, normalizer)
    final_design_summary = _summarize_bmp4_design_distribution(
        normalizer=normalizer,
        bmp4_norm_mu=likelihood_state.bmp4_norm_mu,
        bmp4_norm_log_std=likelihood_state.bmp4_norm_log_std,
        bounds=bounds,
        sample_key=jax.random.PRNGKey(0),
        sample_count=2048,
    )
    final_selector_probs = _selector_probs_from_logits(
        likelihood_state.selector_logits,
        selector_temperature=likelihood_state.selector_temperature,
    )
    final_design_summary["selector_logits"] = np.asarray(likelihood_state.selector_logits, dtype=np.float32)
    final_design_summary["selector_probs"] = final_selector_probs
    final_design_summary["selector_temperature"] = float(likelihood_state.selector_temperature)
    final_log10_doses = np.log10(np.clip(final_design_summary["dose_mu"], 1e-30, None)).astype("float32")
    prior_eig_baseline = _estimate_snpe_prior_eig_baseline(
        posterior_net=posterior_net,
        joint_data=joint_data,
        snpe_samples=snpe_samples,
        theta_prior=simulation_summary.get("theta_raw_prior"),
    )
    eig_result = _add_prior_eig_baseline(eig_result, prior_eig_baseline)
    with likelihood_checkpoint_path.open("wb") as handle:
        pickle.dump(
            {
                "posterior_network": posterior_net.state_dict(),
                "jax_likelihood": {
                    "flow_params": _jax_tree_to_numpy(likelihood_state.flow_params),
                    "flow_config": likelihood_state.flow_config,
                    "bmp4_norm_mu": final_design_summary["bmp4_norm_mu"],
                    "bmp4_norm_log_std": final_design_summary["bmp4_norm_log_std"],
                    "bmp4_norm_std": final_design_summary["bmp4_norm_std"],
                    "dose_mu": final_design_summary["dose_mu"],
                    "dose_std": final_design_summary["dose_std"],
                    "selector_logits": final_design_summary["selector_logits"],
                    "selector_probs": final_design_summary["selector_probs"],
                    "selector_temperature": float(likelihood_state.selector_temperature),
                    "log10_doses": final_log10_doses,
                    "infonce_lambda": float(likelihood_state.infonce_lambda),
                    "infonce_negatives": int(likelihood_state.infonce_negatives),
                },
                "metadata": {
                    "family": "promisys_onestep",
                    "theta_names": list(THETA_NAMES),
                    "complex_names": list(COMPLEX_NAMES),
                    "receptor_names": list(TARGET_RECEPTOR_NAMES),
                    "snpe_theta_prior": simulation_summary.get("theta_raw_prior"),
                    "observation_noise_prior": {
                        "distribution": "LogNormal",
                        "parameter_name": OBSERVATION_NOISE_PARAMETER_NAME,
                        "space": "normalized_response",
                        "loc": OBSERVATION_NOISE_LOG_LOC,
                        "scale": OBSERVATION_NOISE_LOG_SCALE,
                        "low": OBSERVATION_NOISE_MIN,
                        "high": OBSERVATION_NOISE_MAX,
                    },
                    "normalizer": normalizer.to_dict(),
                    "normalizer_data_dir": str(normalizer_data_dir),
                    "likelihood_backend": "jax_lfiax_nsf",
                    "promisys_hyperparams": effective_hyperparams,
                },
            },
            handle,
        )

    summary = {
        "family": "promisys_onestep",
        "candidate_name": "promisys_onestep_lfiax",
        "cell_lines": list(joint_data.cell_lines),
        "theta_names": list(THETA_NAMES),
        "complex_names": list(COMPLEX_NAMES),
        "receptor_names": list(TARGET_RECEPTOR_NAMES),
        "snpe_loss_history": snpe_loss_history,
        "likelihood_loss_history": likelihood_loss_history,
        "joint_boed_history": likelihood_state.history,
        "gradient_diagnostics": likelihood_state.gradient_diagnostics,
        "infonce_lambda": float(likelihood_state.infonce_lambda),
        "infonce_negatives": int(likelihood_state.infonce_negatives),
        "final_design_distribution": final_design_summary,
        "fit_steps_used_for_joint_boed": len(likelihood_state.history),
        "eig_steps_ignored_for_promisys_onestep": True,
        "eig_steps_requested": int(eig_steps),
        "posterior_summary": _summarize_samples(snpe_samples),
        "mcmc_posterior_summary": _summarize_samples(mcmc_samples_by_cell_line),
        "snpe_sample_count_by_cell": _sample_counts(snpe_samples),
        "mcmc_sample_count_by_cell": _sample_counts(mcmc_samples_by_cell_line),
        "normalizer": normalizer.to_dict(),
        "normalizer_data_dir": str(normalizer_data_dir),
        "receptor_noise_log_sd": float(receptor_noise_log_sd),
        "simulation_summary": simulation_summary,
        "prior_eig_baseline": prior_eig_baseline,
        "requested_promisys_hyperparams": requested_hyperparams,
        "promisys_hyperparams": effective_hyperparams,
    }
    _progress("Writing JSON summaries")
    _write_json(snpe_fit_summary_path, summary)
    _write_json(fit_summary_path, summary)
    _write_json(eig_summary_path, eig_result)
    _progress("Writing posterior predictive plot")
    save_joint_posterior_predictive_plot(
        observed_concentration=joint_data.bmp4_conc,
        observed_response=joint_data.x_obs,
        grid_concentration=posterior_predictive["grid_concentration"],
        predictive_draws=posterior_predictive["predictive_y"],
        cell_lines=joint_data.cell_lines,
        output_path=posterior_plot_path,
        title="Posterior predictive BMP4 dose-response: promisys one-step",
    )
    _progress("Writing prior/posterior comparison plots")
    _save_theta_comparison_plot(snpe_samples, mcmc_samples_by_cell_line, prior_plot_path, positive=False)
    _save_theta_comparison_plot(snpe_samples, mcmc_samples_by_cell_line, prior_positive_plot_path, positive=True)
    _progress("Writing EIG optimization plot")
    save_eig_optimization_plot(
        eig_result["history"],
        eig_plot_path,
        title="Next-design EIG optimization: promisys one-step",
    )
    return {
        "snpe_posterior_samples_path": str(snpe_path),
        "posterior_samples_path": str(posterior_samples_path),
        "mcmc_posterior_samples_path": str(mcmc_path),
        "posterior_predictive_path": str(posterior_predictive_path),
        "posterior_predictive_plot": str(posterior_plot_path),
        "prior_posterior_plot": str(prior_plot_path),
        "prior_posterior_positive_plot": str(prior_positive_plot_path),
        "eig_optimization_plot": str(eig_plot_path),
        "likelihood_checkpoint": str(likelihood_checkpoint_path),
        "snpe_fit_summary_path": str(snpe_fit_summary_path),
        "fit_summary_path": str(fit_summary_path),
        "eig_summary_path": str(eig_summary_path),
    }


def _log10_dose_bounds(joint_data: Any) -> tuple[float, float]:
    positive = np.asarray(joint_data.bmp4_conc, dtype=np.float32)
    positive = positive[positive > 0.0]
    low = max(float(positive.min()) if positive.size else 1e-6, 1e-8)
    high = max(float(np.asarray(joint_data.bmp4_conc).max()), low * 10.0)
    return math.log10(low) - 0.25, math.log10(high) + 0.25


def _bmp4_design_bounds(joint_data: Any, normalizer: Bmp4Normalizer) -> dict[str, float]:
    log10_min, log10_max = _log10_dose_bounds(joint_data)
    dose_min = float(10.0**log10_min)
    dose_max = float(10.0**log10_max)
    norm = np.asarray(
        normalizer.normalize_bmp4(np.asarray([dose_min, dose_max], dtype=np.float32)),
        dtype=np.float64,
    )
    norm_min = float(np.min(norm))
    norm_max = float(np.max(norm))
    if not math.isfinite(norm_min) or not math.isfinite(norm_max) or norm_min >= norm_max:
        raise ValueError("Invalid normalized BMP4 design bounds.")
    return {
        "bmp4_norm_min": norm_min,
        "bmp4_norm_max": norm_max,
        "log10_dose_min": float(log10_min),
        "log10_dose_max": float(log10_max),
        "dose_min": dose_min,
        "dose_max": dose_max,
    }


def _design_std_bounds(bounds: dict[str, float]) -> tuple[float, float]:
    max_std = max(float(bounds["bmp4_norm_max"] - bounds["bmp4_norm_min"]), 1e-3)
    return 1e-3, max_std


def _decode_bmp4_norm_std(log_std: Any, bounds: dict[str, float]) -> Any:
    import jax.numpy as jnp

    min_std, max_std = _design_std_bounds(bounds)
    return jnp.clip(jnp.exp(jnp.asarray(log_std, dtype=jnp.float32)), min_std, max_std)


def _design_temperature_dose_std(
    *,
    step_index: int,
    total_steps: int,
    temperature_scale: float,
    bounds: dict[str, float],
) -> float:
    max_std = max((float(bounds["dose_max"]) - float(bounds["dose_min"])) / 2.0, 1e-3)
    progress = (int(step_index) + 1) / max(int(total_steps), 1)
    return min(max(float(temperature_scale) * progress, 1e-6), max_std)


def _design_temperature_norm_std_schedule(
    *,
    normalizer: Bmp4Normalizer,
    bmp4_norm_mu: Any,
    step_index: int,
    total_steps: int,
    final_dose_std: float,
    bounds: dict[str, float],
    initial_norm_std: float = 1.0,
) -> Any:
    min_std, max_std = _design_std_bounds(bounds)
    start = min(max(float(initial_norm_std), min_std), max_std)
    final = _design_temperature_norm_std(
        normalizer=normalizer,
        bmp4_norm_mu=bmp4_norm_mu,
        target_dose_std=float(final_dose_std),
        bounds=bounds,
    )
    if int(total_steps) <= 1:
        return final
    progress = min(max(int(step_index) / max(int(total_steps) - 1, 1), 0.0), 1.0)
    final = np.clip(np.asarray(final, dtype=np.float64), min_std, max_std)
    scheduled = np.exp((1.0 - progress) * math.log(start) + progress * np.log(final))
    return np.asarray(scheduled, dtype=np.float32)


def _design_temperature_norm_std(
    *,
    normalizer: Bmp4Normalizer,
    bmp4_norm_mu: Any,
    target_dose_std: float,
    bounds: dict[str, float],
    quantile_count: int = 257,
) -> Any:
    from scipy import stats

    mu_values = np.asarray(bmp4_norm_mu, dtype=np.float64).reshape(-1)
    min_std, max_std = _design_std_bounds(bounds)
    target = max(float(target_dose_std), 0.0)
    lower = float(bounds["bmp4_norm_min"])
    upper = float(bounds["bmp4_norm_max"])
    quantiles = (np.arange(int(quantile_count), dtype=np.float64) + 0.5) / int(quantile_count)

    def dose_std_for(mu: float, norm_std: float) -> float:
        std = max(float(norm_std), min_std)
        a = (lower - mu) / std
        b = (upper - mu) / std
        eps = stats.truncnorm.ppf(quantiles, a, b)
        norm_samples = mu + std * eps
        dose_samples = np.asarray(normalizer.denormalize_bmp4(norm_samples), dtype=np.float64)
        return float(np.std(dose_samples))

    norm_std_values = []
    for mu in mu_values:
        low = min_std
        high = max_std
        if dose_std_for(float(mu), low) >= target:
            norm_std_values.append(low)
            continue
        if dose_std_for(float(mu), high) <= target:
            norm_std_values.append(high)
            continue
        for _ in range(24):
            mid = 0.5 * (low + high)
            if dose_std_for(float(mu), mid) < target:
                low = mid
            else:
                high = mid
        norm_std_values.append(high)
    return np.asarray(norm_std_values, dtype=np.float32)


def _clip_bmp4_design_params(design_params: dict[str, Any], bounds: dict[str, float]) -> dict[str, Any]:
    import jax.numpy as jnp

    min_std, max_std = _design_std_bounds(bounds)
    clipped = {
        "bmp4_norm_mu": jnp.clip(
            jnp.asarray(design_params["bmp4_norm_mu"], dtype=jnp.float32),
            float(bounds["bmp4_norm_min"]),
            float(bounds["bmp4_norm_max"]),
        ),
        "bmp4_norm_log_std": jnp.clip(
            jnp.asarray(design_params["bmp4_norm_log_std"], dtype=jnp.float32),
            math.log(min_std),
            math.log(max_std),
        ),
    }
    if "selector_logits" in design_params:
        logits = jnp.asarray(design_params["selector_logits"], dtype=jnp.float32)
        logits = logits - jnp.mean(logits)
        clipped["selector_logits"] = jnp.clip(logits, -20.0, 20.0)
    return clipped


def _apply_design_sgd_update(
    design_params: dict[str, Any],
    design_grads: dict[str, Any],
    bounds: dict[str, float],
    *,
    learning_rate: float,
) -> dict[str, Any]:
    import jax.numpy as jnp

    next_params = {
        "bmp4_norm_mu": (
            jnp.asarray(design_params["bmp4_norm_mu"], dtype=jnp.float32)
            - float(learning_rate)
            * jnp.asarray(design_grads["bmp4_norm_mu"], dtype=jnp.float32)
        ),
        "bmp4_norm_log_std": (
            jnp.asarray(design_params["bmp4_norm_log_std"], dtype=jnp.float32)
        ),
    }
    if "selector_logits" in design_params:
        next_params["selector_logits"] = (
            jnp.asarray(design_params["selector_logits"], dtype=jnp.float32)
            - float(learning_rate) * jnp.asarray(design_grads["selector_logits"], dtype=jnp.float32)
        )
    return _clip_bmp4_design_params(next_params, bounds)


def _sample_bmp4_norm_design_distribution(
    *,
    design_params: dict[str, Any],
    design_key: Any,
    sample_count: int,
    bounds: dict[str, float],
) -> Any:
    import jax
    import jax.numpy as jnp

    mu = jnp.asarray(design_params["bmp4_norm_mu"], dtype=jnp.float32)
    std = _decode_bmp4_norm_std(design_params["bmp4_norm_log_std"], bounds)
    keys = jax.random.split(design_key, mu.shape[0])
    lower = float(bounds["bmp4_norm_min"])
    upper = float(bounds["bmp4_norm_max"])

    def sample_one(key: Any, mu_value: Any, std_value: Any) -> Any:
        a = (lower - mu_value) / std_value
        b = (upper - mu_value) / std_value
        eps = jax.random.truncated_normal(
            key,
            lower=a,
            upper=b,
            shape=(int(sample_count),),
            dtype=jnp.float32,
        )
        return mu_value + std_value * eps

    return jax.vmap(sample_one)(keys, mu, std)


def _summarize_bmp4_design_distribution(
    *,
    normalizer: Bmp4Normalizer,
    bmp4_norm_mu: Any,
    bmp4_norm_log_std: Any,
    bounds: dict[str, float],
    sample_key: Any,
    sample_count: int,
) -> dict[str, Any]:
    design_params = {
        "bmp4_norm_mu": bmp4_norm_mu,
        "bmp4_norm_log_std": bmp4_norm_log_std,
    }
    bmp4_norm_samples = np.asarray(
        _sample_bmp4_norm_design_distribution(
            design_params=design_params,
            design_key=sample_key,
            sample_count=int(sample_count),
            bounds=bounds,
        ),
        dtype=np.float32,
    )
    dose_samples = np.asarray(_decode_bmp4_norm_designs(normalizer, bmp4_norm_samples), dtype=np.float32)
    min_std, max_std = _design_std_bounds(bounds)
    bmp4_norm_mu_np = np.asarray(bmp4_norm_mu, dtype=np.float32)
    bmp4_norm_log_std_np = np.asarray(bmp4_norm_log_std, dtype=np.float32)
    bmp4_norm_std_np = np.clip(np.exp(bmp4_norm_log_std_np), min_std, max_std).astype("float32")
    return {
        "bmp4_norm_mu": bmp4_norm_mu_np,
        "bmp4_norm_log_std": bmp4_norm_log_std_np,
        "bmp4_norm_std": bmp4_norm_std_np,
        "dose_mu": np.asarray(np.mean(dose_samples, axis=1), dtype=np.float32),
        "dose_std": np.asarray(np.std(dose_samples, axis=1), dtype=np.float32),
    }


def _decode_bmp4_norm_designs(normalizer: Bmp4Normalizer, bmp4_norm_designs: Any) -> Any:
    raw = np.asarray(normalizer.denormalize_bmp4(bmp4_norm_designs), dtype=np.float64)
    return np.clip(raw, 1e-30, None)


def _samples_payload(
    samples_by_cell_line: dict[str, Any],
    *,
    theta_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    return {
        "theta_norm": {
            cell_line: torch.as_tensor(samples, dtype=torch.float32)
            for cell_line, samples in samples_by_cell_line.items()
        },
        "theta_names": list(THETA_NAMES),
        "complex_names": list(COMPLEX_NAMES),
        "receptor_names": list(TARGET_RECEPTOR_NAMES),
        "theta_coordinate": (
            "First 12 coordinates are Gaussianized log-uniform biophysical "
            "parameters; final coordinate is the latent log-normal observation-noise "
            "coordinate for normalized-response space."
        ),
        "theta_normalization_scale": THETA_NORMALIZATION_SCALE,
        "theta_raw_prior": theta_prior
        if theta_prior is not None
        else {
            "distribution": "LogUniform",
            "low": RAW_PARAMETER_LOW,
            "high": RAW_PARAMETER_HIGH,
        },
        "observation_noise_prior": {
            "distribution": "LogNormal",
            "parameter_name": OBSERVATION_NOISE_PARAMETER_NAME,
            "space": "normalized_response",
            "loc": OBSERVATION_NOISE_LOG_LOC,
            "scale": OBSERVATION_NOISE_LOG_SCALE,
            "low": OBSERVATION_NOISE_MIN,
            "high": OBSERVATION_NOISE_MAX,
        },
    }


def _summarize_samples(samples_by_cell_line: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cell_line, samples in samples_by_cell_line.items():
        arr = np.asarray(samples, dtype=np.float32)
        summary[cell_line] = {
            name: {
                "mean": float(np.mean(arr[:, index])),
                "std": float(np.std(arr[:, index])),
                "q05": float(np.quantile(arr[:, index], 0.05)),
                "q50": float(np.quantile(arr[:, index], 0.5)),
                "q95": float(np.quantile(arr[:, index], 0.95)),
            }
            for index, name in enumerate(THETA_NAMES)
        }
    return summary


def _sample_counts(samples_by_cell_line: dict[str, Any]) -> dict[str, int]:
    return {
        str(cell_line): int(np.asarray(samples).shape[0])
        for cell_line, samples in samples_by_cell_line.items()
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


def _json_safe(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_theta_comparison_plot(
    snpe_samples: dict[str, Any],
    mcmc_samples_by_cell_line: dict[str, Any],
    output_path: Path,
    *,
    positive: bool,
) -> str:
    import matplotlib.pyplot as plt

    cell_line = next(iter(snpe_samples))
    snpe = np.asarray(snpe_samples[cell_line], dtype=np.float32)
    mcmc = np.asarray(mcmc_samples_by_cell_line[cell_line], dtype=np.float32)
    if positive:
        snpe = theta_norm_to_raw(snpe)
        mcmc = theta_norm_to_raw(mcmc)
    panel_count = THETA_DIM
    n_cols = 4
    n_rows = (panel_count + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.6 * n_rows), squeeze=False)
    for index in range(panel_count):
        axis = axes[index // n_cols][index % n_cols]
        bins: int | Any = 24
        if positive:
            if index < BIOPHYSICAL_THETA_DIM:
                bins = np.geomspace(RAW_PARAMETER_LOW, RAW_PARAMETER_HIGH, 25)
            else:
                bins = np.geomspace(OBSERVATION_NOISE_MIN, OBSERVATION_NOISE_MAX, 25)
        axis.hist(snpe[:, index], bins=bins, alpha=0.45, label="SNPE prior")
        axis.hist(mcmc[:, index], bins=bins, alpha=0.45, label="MCMC posterior")
        axis.set_title(THETA_NAMES[index], fontsize=8)
        if positive:
            axis.set_xscale("log")
            if index < BIOPHYSICAL_THETA_DIM:
                axis.set_xlim(RAW_PARAMETER_LOW, RAW_PARAMETER_HIGH)
            else:
                axis.set_xlim(OBSERVATION_NOISE_MIN, OBSERVATION_NOISE_MAX)
    for index in range(panel_count, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        f"promisys_onestep parameter comparison ({cell_line}, {'positive' if positive else 'normalized'} scale)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


__all__ = [
    "BIOPHYSICAL_PARAMETER_NAMES",
    "BIOPHYSICAL_THETA_DIM",
    "BINDING_PARAMETER_NAMES",
    "Bmp4Normalizer",
    "COMPLEX_NAMES",
    "DEFAULT_INFONCE_LAMBDA",
    "DEFAULT_LFIAX_ROOT",
    "DEFAULT_NORMALIZER_DATA_DIR",
    "EFFICIENCY_PARAMETER_NAMES",
    "MODEL_SIZE",
    "RAW_PARAMETER_HIGH",
    "RAW_PARAMETER_LOW",
    "TARGET_RECEPTOR_NAMES",
    "OBSERVATION_NOISE_LOG_LOC",
    "OBSERVATION_NOISE_LOG_SCALE",
    "OBSERVATION_NOISE_MAX",
    "OBSERVATION_NOISE_MIN",
    "OBSERVATION_NOISE_PARAMETER_NAME",
    "THETA_DIM",
    "THETA_NAMES",
    "THETA_NORMALIZATION_SCALE",
    "check_promisys_dependencies",
    "metadata_parameters",
    "problem_description",
    "run_promisys_onestep_workflow",
    "sample_theta_norm_prior",
    "simulate_promisys_onestep_raw",
    "split_theta_raw",
    "theta_norm_to_observation_noise",
    "theta_raw_to_norm",
    "theta_norm_to_raw",
]
