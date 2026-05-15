from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

from boed_agent.models import DesignVariable, ExperimentSpec
from boed_agent.utils.imports import register_callable

from .multireceptor_hierarchical import SELECTOR_TEMPERATURE


MODEL_REF = "bmp4.postfit_boed.model"
POSTERIOR_GUIDE_REF = "bmp4.postfit_boed.posterior_guide"
MARGINAL_GUIDE_REF = "bmp4.postfit_boed.marginal_guide"
VI_GUIDE_REF = "bmp4.postfit_boed.vi_guide"
OPTIM_REF = "boed_agent.demo.pyro_linear:make_pyro_adam"
LOSS_REF = "boed_agent.demo.pyro_linear:make_trace_elbo_loss"

TARGET_LABEL = "theta"
OBSERVATION_LABEL = "y_next"
SELECTOR_LOGIT_BOUND = 8.0
MIN_THETA_SCALE = 1e-4
MIN_GUIDE_SCALE = 1e-4


@dataclass(frozen=True)
class PostfitBoedContext:
    family_name: str
    theta_loc: torch.Tensor
    theta_scale: torch.Tensor
    theta_names: tuple[str, ...]
    receptor_names: tuple[str, ...] = ()
    cell_line_names: tuple[str, ...] = ()
    selector_temperature: float = SELECTOR_TEMPERATURE
    log10_dose_min: float = -4.0
    log10_dose_max: float = 4.0
    observation_positive: bool = False


_CURRENT_CONTEXT: PostfitBoedContext | None = None


def set_current_context(context: PostfitBoedContext) -> None:
    global _CURRENT_CONTEXT
    _CURRENT_CONTEXT = context


def clear_current_context() -> None:
    global _CURRENT_CONTEXT
    _CURRENT_CONTEXT = None


def build_postfit_context(
    *,
    family_name: str,
    posterior_samples: dict[str, torch.Tensor],
    receptor_names: Sequence[str] = (),
    cell_line_names: Sequence[str] = (),
    log10_dose_min: float,
    log10_dose_max: float,
    selector_temperature: float = SELECTOR_TEMPERATURE,
) -> PostfitBoedContext:
    packed, theta_names = _pack_posterior_samples(
        family_name=family_name,
        posterior_samples=posterior_samples,
        receptor_names=tuple(receptor_names),
        cell_line_names=tuple(cell_line_names),
    )
    loc = packed.mean(dim=0)
    scale = packed.std(dim=0, unbiased=False).clamp(min=MIN_THETA_SCALE)
    return PostfitBoedContext(
        family_name=family_name,
        theta_loc=loc.detach().clone(),
        theta_scale=scale.detach().clone(),
        theta_names=theta_names,
        receptor_names=tuple(receptor_names),
        cell_line_names=tuple(cell_line_names),
        selector_temperature=float(selector_temperature),
        log10_dose_min=float(log10_dose_min),
        log10_dose_max=float(log10_dose_max),
        observation_positive=(family_name == "multireceptor_hierarchical"),
    )


def build_postfit_boed_spec(
    *,
    context: PostfitBoedContext,
    estimator: str,
    guide_training_steps: int,
    num_outer_samples: int,
    num_optimization_steps: int,
    design_learning_rate: float,
    guide_learning_rate: float,
    num_inner_samples: int | None = None,
) -> ExperimentSpec:
    allowed = {"posterior_eig", "marginal_eig", "vnmc_eig", "vi_eig"}
    if estimator not in allowed:
        raise ValueError(f"Unsupported BMP4 BOED estimator '{estimator}'.")

    payload: dict[str, Any] = {
        "backend": "pyro",
        "design_variables": [
            {
                "name": item.name,
                "lower": item.lower,
                "upper": item.upper,
                "initial": item.initial,
                "dtype": item.dtype,
                "description": item.description,
                "shape": item.shape,
            }
            for item in _design_variables_for_context(context)
        ],
        "observation_labels": [OBSERVATION_LABEL],
        "target_latent_labels": [TARGET_LABEL],
        "compute_budget": {
            "num_outer_samples": int(num_outer_samples),
            "guide_training_steps": int(guide_training_steps),
            "num_optimization_steps": int(num_optimization_steps),
            "design_learning_rate": float(design_learning_rate),
        },
        "objective": {
            "estimator": estimator,
            "mode": "variational",
        },
        "backend_options": {
            "optimization_strategy": "gradient",
            "guide_learning_rate": float(guide_learning_rate),
        },
        "metadata": {
            "family_name": context.family_name,
            "theta_names": list(context.theta_names),
        },
        "model_ref": MODEL_REF,
        "guide_ref": _guide_ref_for_estimator(estimator),
        "optim_ref": OPTIM_REF,
    }
    spec = ExperimentSpec.from_dict(payload)
    if estimator == "vi_eig":
        spec.loss_ref = LOSS_REF
    if estimator == "vnmc_eig":
        spec.compute_budget.num_inner_samples = int(num_inner_samples or 8)
    return spec


def decode_design(design: Sequence[float] | torch.Tensor) -> dict[str, Any]:
    context = _require_context()
    design_t = _coerce_design_tensor(design)
    if context.family_name == "multireceptor_hierarchical":
        logits = design_t[:-1]
        selector_probs = torch.softmax(logits / max(context.selector_temperature, 1e-6), dim=-1)
        log10_dose = float(design_t[-1].detach().cpu())
        dose = float(torch.pow(design_t.new_tensor(10.0), design_t[-1]).detach().cpu())
        selected_index = int(torch.argmax(selector_probs).detach().cpu())
        return {
            "raw_design_vector": design_t.detach().cpu().tolist(),
            "cell_line_names": list(context.cell_line_names),
            "selector_temperature": float(context.selector_temperature),
            "selector_probs": selector_probs.detach().cpu().tolist(),
            "selected_cell_line_index": selected_index,
            "selected_cell_line": context.cell_line_names[selected_index],
            "dose_count": 1,
            "doses": [dose],
            "dose": dose,
            "log10_doses": [log10_dose],
            "log10_dose": log10_dose,
        }
    log10_dose = float(design_t[-1].detach().cpu())
    dose = float(torch.pow(design_t.new_tensor(10.0), design_t[-1]).detach().cpu())
    return {
        "raw_design_vector": design_t.detach().cpu().tolist(),
        "dose_count": 1,
        "doses": [dose],
        "dose": dose,
        "log10_doses": [log10_dose],
        "log10_dose": log10_dose,
        "design": dose,
    }


def format_optimization_result(result: Any) -> dict[str, Any]:
    decoded_best = decode_design(result.design)
    history: list[dict[str, Any]] = []
    for step in result.history:
        decoded = decode_design(step.design)
        decoded.update(
            {
                "step": float(step.step),
                "eig": None if step.eig is None else float(step.eig),
                "design": float(decoded["dose"]) if decoded.get("dose") is not None else float(decoded["doses"][0]),
            }
        )
        history.append(decoded)

    output = {
        "backend": result.backend,
        "estimator": result.estimator,
        "status": result.status,
        "warnings": list(result.warnings),
        "history": history,
        "best_design": float(decoded_best["dose"]) if decoded_best.get("dose") is not None else float(decoded_best["doses"][0]),
        "best_dose": decoded_best.get("dose"),
        "best_doses": decoded_best.get("doses"),
        "best_log10_dose": decoded_best.get("log10_dose"),
        "best_log10_doses": decoded_best.get("log10_doses"),
        "best_eig": None if result.eig is None else float(result.eig),
        "optimization_bounds": _optimization_bounds_from_context(_require_context()),
    }
    if "selected_cell_line" in decoded_best:
        output.update(
            {
                "design_type": "cell_line_selector_plus_bmp4_dose",
                "best_cell_line": decoded_best.get("selected_cell_line"),
                "best_cell_line_index": decoded_best.get("selected_cell_line_index"),
                "best_selector_probs": decoded_best.get("selector_probs"),
                "dose_count": int(decoded_best.get("dose_count", 1)),
            }
        )
    else:
        output["design_type"] = "bmp4_dose_only"
    return output


def bmp4_postfit_model(design: Any) -> torch.Tensor:
    context = _require_context()
    design_t = _coerce_design_tensor(design)
    batch_shape = design_t.shape[:-1]
    theta_loc = context.theta_loc.to(device=design_t.device, dtype=design_t.dtype)
    theta_scale = context.theta_scale.to(device=design_t.device, dtype=design_t.dtype)
    theta = pyro.sample(
        TARGET_LABEL,
        dist.Normal(
            theta_loc.expand(batch_shape + theta_loc.shape),
            theta_scale.expand(batch_shape + theta_scale.shape),
        ).to_event(1),
    )
    mu, sigma = _forward_response_from_theta(theta, design_t, context)
    if context.observation_positive:
        return pyro.sample(
            OBSERVATION_LABEL,
            dist.LogNormal(
                torch.log(torch.clamp(mu, min=1e-8)),
                sigma,
            ).to_event(1),
        )
    return pyro.sample(
        OBSERVATION_LABEL,
        dist.Normal(mu, sigma).to_event(1),
    )


def bmp4_postfit_posterior_guide(
    y_dict: dict[str, Any],
    design: Any,
    observation_labels: list[str],
    target_labels: list[str],
) -> None:
    context = _require_context()
    _ = target_labels
    design_features = _design_features(design, context)
    y_value = _coerce_observation(y_dict[observation_labels[0]], context)
    obs_features = torch.log(torch.clamp(y_value, min=1e-8)) if context.observation_positive else y_value
    features = torch.cat([design_features, obs_features], dim=-1)
    theta_dim = int(context.theta_loc.shape[0])
    feature_dim = int(features.shape[-1])
    weight = pyro.param(
        "bmp4_postfit_posterior_weight",
        torch.zeros(theta_dim, feature_dim, dtype=features.dtype, device=features.device),
    )
    bias = pyro.param(
        "bmp4_postfit_posterior_bias",
        context.theta_loc.to(device=features.device, dtype=features.dtype).clone(),
    )
    scale = pyro.param(
        "bmp4_postfit_posterior_scale",
        context.theta_scale.to(device=features.device, dtype=features.dtype).clone(),
        constraint=constraints.positive,
    )
    loc = torch.einsum("...f,tf->...t", features, weight) + bias
    pyro.sample(TARGET_LABEL, dist.Normal(loc, scale).to_event(1))


def bmp4_postfit_marginal_guide(
    design: Any,
    observation_labels: list[str],
    target_labels: list[str],
) -> None:
    context = _require_context()
    _ = target_labels
    design_features = _design_features(design, context)
    feature_dim = int(design_features.shape[-1])
    bias = pyro.param(
        "bmp4_postfit_marginal_bias",
        torch.zeros(1, dtype=design_features.dtype, device=design_features.device),
    )
    weight = pyro.param(
        "bmp4_postfit_marginal_weight",
        torch.zeros(1, feature_dim, dtype=design_features.dtype, device=design_features.device),
    )
    scale = pyro.param(
        "bmp4_postfit_marginal_scale",
        0.5 * torch.ones(1, dtype=design_features.dtype, device=design_features.device),
        constraint=constraints.positive,
    )
    loc = torch.einsum("...f,of->...o", design_features, weight) + bias
    if context.observation_positive:
        pyro.sample(observation_labels[0], dist.LogNormal(loc, scale).to_event(1))
        return
    pyro.sample(observation_labels[0], dist.Normal(loc, scale).to_event(1))


def bmp4_postfit_vi_guide(design: Any) -> None:
    context = _require_context()
    design_t = _coerce_design_tensor(design)
    loc = pyro.param(
        "bmp4_postfit_vi_loc",
        context.theta_loc.to(device=design_t.device, dtype=design_t.dtype).clone(),
    )
    scale = pyro.param(
        "bmp4_postfit_vi_scale",
        context.theta_scale.to(device=design_t.device, dtype=design_t.dtype).clone(),
        constraint=constraints.positive,
    )
    pyro.sample(TARGET_LABEL, dist.Normal(loc, scale).to_event(1))


def _pack_posterior_samples(
    *,
    family_name: str,
    posterior_samples: dict[str, torch.Tensor],
    receptor_names: tuple[str, ...],
    cell_line_names: tuple[str, ...],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    if family_name == "hill":
        bottom = _as_scalar_samples(posterior_samples["bottom"])
        top = _as_scalar_samples(posterior_samples["top"])
        packed = torch.stack(
            [
                bottom,
                top,
                torch.log(torch.clamp(_as_scalar_samples(posterior_samples["ec50"]), min=1e-8)),
                torch.log(torch.clamp(_as_scalar_samples(posterior_samples["hill_n"]), min=1e-8)),
                torch.log(torch.clamp(_as_scalar_samples(posterior_samples["sigma"]), min=1e-8)),
            ],
            dim=-1,
        )
        return packed, ("bottom", "top", "log_ec50", "log_hill_n", "log_sigma")

    if family_name == "multireceptor":
        samples = [
            _as_scalar_samples(posterior_samples["bottom"]),
            _as_scalar_samples(posterior_samples["top"]),
            torch.log(torch.clamp(_as_scalar_samples(posterior_samples["s50"]), min=1e-8)),
            torch.log(torch.clamp(_as_scalar_samples(posterior_samples["response_hill"]), min=1e-8)),
            torch.log(torch.clamp(_as_scalar_samples(posterior_samples["sigma_y"]), min=1e-8)),
        ]
        names = ["bottom", "top", "log_s50", "log_response_hill", "log_sigma_y"]
        for prefix in ("kd", "abundance", "weight"):
            for receptor_name in receptor_names:
                samples.append(
                    torch.log(
                        torch.clamp(
                            _as_scalar_samples(posterior_samples[f"{prefix}_{receptor_name}"]),
                            min=1e-8,
                        )
                    )
                )
                names.append(f"log_{prefix}_{receptor_name}")
        return torch.stack(samples, dim=-1), tuple(names)

    if family_name == "multireceptor_hierarchical":
        log_kd = _flatten_posterior_site(posterior_samples["log_kd"])
        log_weight = _flatten_posterior_site(posterior_samples["log_weight"])
        log_R = _flatten_posterior_site(posterior_samples["log_R"])
        log_bottom = torch.log(torch.clamp(_flatten_posterior_site(posterior_samples["bottom"]), min=1e-8))
        log_top = torch.log(torch.clamp(_flatten_posterior_site(posterior_samples["top"]), min=1e-8))
        log_s50 = _flatten_posterior_site(posterior_samples["log_s50"])
        log_response_hill = torch.log(
            torch.clamp(_flatten_posterior_site(posterior_samples["response_hill"]), min=1e-8)
        )
        log_sigma_y = torch.log(torch.clamp(_flatten_posterior_site(posterior_samples["sigma_y"]), min=1e-8))
        packed = torch.cat(
            [
                log_kd,
                log_weight,
                log_R,
                log_bottom,
                log_top,
                log_s50,
                log_response_hill,
                log_sigma_y,
            ],
            dim=-1,
        )
        names: list[str] = []
        names.extend(f"log_kd::{name}" for name in receptor_names)
        names.extend(f"log_weight::{name}" for name in receptor_names)
        names.extend(
            f"log_R::{cell_line}::{receptor_name}"
            for cell_line in cell_line_names
            for receptor_name in receptor_names
        )
        names.extend(f"log_bottom::{cell_line}" for cell_line in cell_line_names)
        names.extend(f"log_top::{cell_line}" for cell_line in cell_line_names)
        names.append("log_s50")
        names.append("log_response_hill")
        names.extend(f"log_sigma_y::{cell_line}" for cell_line in cell_line_names)
        return packed, tuple(names)

    raise ValueError(f"Unsupported BMP4 family '{family_name}'.")


def _forward_response_from_theta(
    theta: torch.Tensor,
    design: torch.Tensor,
    context: PostfitBoedContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.family_name == "hill":
        bottom = theta[..., 0]
        top = theta[..., 1]
        ec50 = torch.exp(theta[..., 2])
        hill_n = torch.exp(theta[..., 3])
        sigma = torch.exp(theta[..., 4])
        log10_dose = design[..., -1]
        dose = torch.pow(design.new_tensor(10.0), log10_dose)
        mu = bottom + (top - bottom) * (dose**hill_n) / (ec50**hill_n + dose**hill_n)
        return mu.unsqueeze(-1), sigma.unsqueeze(-1)

    if context.family_name == "multireceptor":
        receptor_count = len(context.receptor_names)
        index = 0
        bottom = theta[..., index]
        index += 1
        top = theta[..., index]
        index += 1
        s50 = torch.exp(theta[..., index])
        index += 1
        response_hill = torch.exp(theta[..., index])
        index += 1
        sigma_y = torch.exp(theta[..., index])
        index += 1
        kd = torch.exp(theta[..., index : index + receptor_count])
        index += receptor_count
        abundance = torch.exp(theta[..., index : index + receptor_count])
        index += receptor_count
        weight = torch.exp(theta[..., index : index + receptor_count])
        log10_dose = design[..., -1]
        dose = torch.pow(design.new_tensor(10.0), log10_dose)
        dose = torch.clamp(dose, min=1e-12)
        occupancy = dose.unsqueeze(-1) / (kd + dose.unsqueeze(-1))
        signal = torch.sum(occupancy * abundance * weight, dim=-1)
        signal = torch.clamp(signal, min=1e-12)
        mu = bottom + (top - bottom) * (signal**response_hill) / (s50**response_hill + signal**response_hill)
        return mu.unsqueeze(-1), sigma_y.unsqueeze(-1)

    if context.family_name == "multireceptor_hierarchical":
        receptor_count = len(context.receptor_names)
        cell_line_count = len(context.cell_line_names)
        index = 0
        log_kd = theta[..., index : index + receptor_count]
        index += receptor_count
        log_weight = theta[..., index : index + receptor_count]
        index += receptor_count
        log_R = theta[..., index : index + cell_line_count * receptor_count].reshape(
            theta.shape[:-1] + (cell_line_count, receptor_count)
        )
        index += cell_line_count * receptor_count
        bottom = torch.exp(theta[..., index : index + cell_line_count])
        index += cell_line_count
        top = torch.exp(theta[..., index : index + cell_line_count])
        index += cell_line_count
        log_s50 = theta[..., index]
        index += 1
        response_hill = torch.exp(theta[..., index])
        index += 1
        sigma_y = torch.exp(theta[..., index : index + cell_line_count])

        logits = design[..., :cell_line_count]
        log10_dose = design[..., -1]
        alpha = torch.softmax(logits / max(context.selector_temperature, 1e-6), dim=-1)
        dose = torch.pow(design.new_tensor(10.0), log10_dose)
        kd = torch.exp(log_kd)
        weight = torch.exp(log_weight)
        receptor_amount = torch.exp(log_R)

        receptor_selected = torch.sum(alpha.unsqueeze(-1) * receptor_amount, dim=-2)
        bottom_selected = torch.sum(alpha * bottom, dim=-1)
        top_selected = torch.sum(alpha * top, dim=-1)
        sigma_selected = torch.sum(alpha * sigma_y, dim=-1)

        theta_next = dose.unsqueeze(-1) / (dose.unsqueeze(-1) + kd)
        signal = torch.sum(theta_next * receptor_selected * weight, dim=-1)
        signal = torch.clamp(signal, min=1e-12)
        s50 = torch.exp(log_s50)
        effective_top = torch.maximum(top_selected, bottom_selected + 1e-6)
        mu = bottom_selected + (effective_top - bottom_selected) * (
            signal**response_hill / (s50**response_hill + signal**response_hill)
        )
        return mu.unsqueeze(-1), sigma_selected.unsqueeze(-1)

    raise ValueError(f"Unsupported BMP4 family '{context.family_name}'.")


def _design_variables_for_context(context: PostfitBoedContext) -> list[DesignVariable]:
    variables: list[DesignVariable] = []
    if context.family_name == "multireceptor_hierarchical":
        for cell_line_name in context.cell_line_names:
            variables.append(
                DesignVariable(
                    name=f"selector_logit_{cell_line_name}",
                    lower=-SELECTOR_LOGIT_BOUND,
                    upper=SELECTOR_LOGIT_BOUND,
                    initial=0.0,
                    description=f"Relaxed selector logit for the {cell_line_name} cell line.",
                )
            )
    variables.append(
        DesignVariable(
            name="log10_bmp4_concentration",
            lower=context.log10_dose_min,
            upper=context.log10_dose_max,
            initial=(context.log10_dose_min + context.log10_dose_max) / 2.0,
            description="Candidate BMP4 concentration in log10 ng/mL.",
        )
    )
    return variables


def _optimization_bounds_from_context(context: PostfitBoedContext) -> dict[str, Any]:
    bounds = {
        "log10_dose_min": float(context.log10_dose_min),
        "log10_dose_max": float(context.log10_dose_max),
        "dose_min": float(10.0 ** context.log10_dose_min),
        "dose_max": float(10.0 ** context.log10_dose_max),
    }
    if context.family_name == "multireceptor_hierarchical":
        bounds["selector_logit_bound"] = float(SELECTOR_LOGIT_BOUND)
    return bounds


def _guide_ref_for_estimator(estimator: str) -> str:
    if estimator == "marginal_eig":
        return MARGINAL_GUIDE_REF
    if estimator == "vi_eig":
        return VI_GUIDE_REF
    return POSTERIOR_GUIDE_REF


def _coerce_design_tensor(design: Sequence[float] | torch.Tensor) -> torch.Tensor:
    design_t = torch.as_tensor(design, dtype=torch.float32)
    if design_t.ndim == 0:
        return design_t.reshape(1)
    return design_t


def _coerce_observation(value: Any, context: PostfitBoedContext) -> torch.Tensor:
    y = torch.as_tensor(value, dtype=torch.float32)
    if y.ndim == 0:
        return y.reshape(1)
    if y.shape[-1] != 1:
        return y.reshape(y.shape[:-1] + (1,))
    return y


def _design_features(design: Any, context: PostfitBoedContext) -> torch.Tensor:
    design_t = _coerce_design_tensor(design)
    if context.family_name == "multireceptor_hierarchical":
        logits = design_t[..., : len(context.cell_line_names)]
        selector_probs = torch.softmax(logits / max(context.selector_temperature, 1e-6), dim=-1)
        log10_dose = design_t[..., -1:].reshape(design_t.shape[:-1] + (1,))
        return torch.cat([selector_probs, log10_dose], dim=-1)
    return design_t.reshape(design_t.shape[:-1] + (1,))


def _flatten_posterior_site(site: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(site, dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    sample_count = int(tensor.shape[0])
    return tensor.reshape(sample_count, -1)


def _as_scalar_samples(site: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(site, dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor
    return tensor.reshape(tensor.shape[0], -1).squeeze(-1)


def _require_context() -> PostfitBoedContext:
    if _CURRENT_CONTEXT is None:
        raise RuntimeError("No active BMP4 post-fit BOED context has been configured.")
    return _CURRENT_CONTEXT


register_callable(MODEL_REF, bmp4_postfit_model)
register_callable(POSTERIOR_GUIDE_REF, bmp4_postfit_posterior_guide)
register_callable(MARGINAL_GUIDE_REF, bmp4_postfit_marginal_guide)
register_callable(VI_GUIDE_REF, bmp4_postfit_vi_guide)


__all__ = [
    "LOSS_REF",
    "MARGINAL_GUIDE_REF",
    "MODEL_REF",
    "OBSERVATION_LABEL",
    "OPTIM_REF",
    "POSTERIOR_GUIDE_REF",
    "PostfitBoedContext",
    "TARGET_LABEL",
    "VI_GUIDE_REF",
    "build_postfit_boed_spec",
    "build_postfit_context",
    "clear_current_context",
    "decode_design",
    "format_optimization_result",
    "set_current_context",
]
