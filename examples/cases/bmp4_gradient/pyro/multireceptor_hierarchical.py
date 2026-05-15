from __future__ import annotations

import math
from typing import Sequence

import pyro
import pyro.distributions as dist
import torch

from ..priors import TranslatedPrior, make_distribution


SELECTOR_TEMPERATURE = 0.10
DEFAULT_LOG10_DOSE_MARGIN_DECADES = 0.25
DEFAULT_LOG10_DOSE_MIN = -4.0
DEFAULT_LOG10_DOSE_MAX = 4.0
FIT_PARAMETER_NAMES = (
    "log_kd",
    "log_weight",
    "base_log_R",
    "qpcr_intercept",
    "qpcr_slope",
    "sigma_q",
    "sigma_R",
    "log_s50",
    "response_hill",
    "bottom",
    "top",
    "sigma_y",
)
LITERATURE_PARAMETER_NAMES = (
    "kd",
    "weight",
    "bottom",
    "top",
    "s50",
    "response_hill",
    "sigma_y",
)


def make_fit_model(
    prior: TranslatedPrior,
    *,
    q_obs: torch.Tensor,
    kd_prior_shift: torch.Tensor,
):
    q_obs_t = torch.as_tensor(q_obs, dtype=torch.float32)
    kd_prior_shift_t = torch.as_tensor(kd_prior_shift, dtype=torch.float32)

    def model(design: torch.Tensor, y_obs: torch.Tensor | None = None) -> torch.Tensor:
        design_t = torch.as_tensor(design, dtype=torch.float32)
        y_obs_t = None if y_obs is None else torch.as_tensor(y_obs, dtype=torch.float32)
        return _bmp4_multireceptor_model(
            design_t,
            q_obs=q_obs_t,
            y_obs=y_obs_t,
            kd_prior_shift=kd_prior_shift_t,
            prior=prior,
        )

    return model


def predictive_draws(
    concentration: torch.Tensor,
    posterior_samples: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    sample_count = _sample_count_from_samples(posterior_samples)
    cell_line_count = _cell_line_count_from_samples(posterior_samples, sample_count=sample_count)
    receptor_count = _receptor_count_from_samples(posterior_samples, sample_count=sample_count)
    design = _coerce_design_matrix(
        concentration,
        cell_line_count=cell_line_count,
    )
    log_kd = _reshape_site(posterior_samples["log_kd"], sample_count, receptor_count)
    log_weight = _reshape_site(posterior_samples["log_weight"], sample_count, receptor_count)
    log_R = _reshape_site(posterior_samples["log_R"], sample_count, cell_line_count, receptor_count)
    bottom = _reshape_site(posterior_samples["bottom"], sample_count, cell_line_count)
    top = _reshape_site(posterior_samples["top"], sample_count, cell_line_count)
    sigma_y = _reshape_site(posterior_samples["sigma_y"], sample_count, cell_line_count)
    log_s50 = _reshape_site(posterior_samples["log_s50"], sample_count)
    response_hill = _reshape_site(posterior_samples["response_hill"], sample_count)

    kd = torch.exp(log_kd)
    weight = torch.exp(log_weight)
    receptor_amount = torch.exp(log_R)
    mu = _response_mean(
        design=design,
        kd=kd,
        weight=weight,
        receptor_amount=receptor_amount,
        bottom=bottom,
        top=top,
        log_s50=log_s50,
        response_hill=response_hill,
    )
    y = dist.LogNormal(torch.log(torch.clamp(mu, min=1e-8)), sigma_y.unsqueeze(-1)).rsample()
    return {"mu": mu, "y": y}


def scalar_log_likelihood(
    y_value: torch.Tensor,
    concentration: torch.Tensor,
    posterior_samples: dict[str, torch.Tensor],
    *,
    selector_temperature: float = SELECTOR_TEMPERATURE,
    log10_dose_min: float | None = None,
    log10_dose_max: float | None = None,
) -> torch.Tensor:
    sample_count = _sample_count_from_samples(posterior_samples)
    cell_line_count = _cell_line_count_from_samples(posterior_samples, sample_count=sample_count)
    receptor_count = _receptor_count_from_samples(posterior_samples, sample_count=sample_count)
    log_kd = _reshape_site(posterior_samples["log_kd"], sample_count, receptor_count)
    log_weight = _reshape_site(posterior_samples["log_weight"], sample_count, receptor_count)
    log_R = _reshape_site(posterior_samples["log_R"], sample_count, cell_line_count, receptor_count)
    bottom = _reshape_site(posterior_samples["bottom"], sample_count, cell_line_count)
    top = _reshape_site(posterior_samples["top"], sample_count, cell_line_count)
    sigma_y = _reshape_site(posterior_samples["sigma_y"], sample_count, cell_line_count)
    log_s50 = _reshape_site(posterior_samples["log_s50"], sample_count)
    response_hill = _reshape_site(posterior_samples["response_hill"], sample_count)

    design_t = torch.as_tensor(concentration, dtype=torch.float32)
    if _is_selector_dose_design(design_t, cell_line_count=cell_line_count):
        doses, alpha, _ = _selector_dose_design_components(
            design_t,
            cell_line_count=cell_line_count,
            selector_temperature=selector_temperature,
            log10_dose_min=log10_dose_min,
            log10_dose_max=log10_dose_max,
        )
        mu = _selector_dose_response_mean(
            doses=doses,
            alpha=alpha,
            kd=torch.exp(log_kd),
            weight=torch.exp(log_weight),
            receptor_amount=torch.exp(log_R),
            bottom=bottom,
            top=top,
            log_s50=log_s50,
            response_hill=response_hill,
        )
        y_vector = _coerce_future_observation_vector(y_value, dose_count=doses.shape[0])
        sigma_selected = _selector_reduce(alpha, sigma_y)
        return dist.LogNormal(
            torch.log(torch.clamp(mu, min=1e-8)),
            sigma_selected.unsqueeze(-1),
        ).log_prob(y_vector.unsqueeze(0)).sum(dim=-1)

    design = _coerce_design_matrix(
        design_t,
        cell_line_count=cell_line_count,
    )
    y_matrix = _coerce_observation_matrix(
        y_value,
        cell_line_count=design.shape[0],
        design_points=design.shape[1],
    )
    mu = _response_mean(
        design=design,
        kd=torch.exp(log_kd),
        weight=torch.exp(log_weight),
        receptor_amount=torch.exp(log_R),
        bottom=bottom,
        top=top,
        log_s50=log_s50,
        response_hill=response_hill,
    )
    return dist.LogNormal(
        torch.log(torch.clamp(mu, min=1e-8)),
        sigma_y.unsqueeze(-1),
    ).log_prob(y_matrix).sum(dim=(-1, -2))


def problem_description(problem_summary: str, receptor_names: Sequence[str]) -> str:
    receptor_text = ", ".join(receptor_names)
    return (
        f"{problem_summary} Candidate model family: hierarchical multireceptor BMP4 "
        f"model over receptors {receptor_text}. Infer shared receptor affinities and "
        "signaling weights across cell lines, with joint qPCR and response layers. "
        "For receptor affinity priors from SPR dissociation constants, convert to the "
        "EQTK affinity scale used by this BMP4 code as K_eqtk = 1e-8 / K_d, where K_d "
        "is in molar units; equivalently K_eqtk = 10 / K_d_nM or K_eqtk = 10000 / K_d_pM. "
        "Use K_eqtk, not raw K_d, as the positive-scale prior center for kd_* parameters."
    )


def decode_selector_dose_design(
    design: torch.Tensor,
    *,
    cell_line_names: Sequence[str],
    selector_temperature: float = SELECTOR_TEMPERATURE,
    log10_dose_min: float | None = None,
    log10_dose_max: float | None = None,
) -> dict[str, object]:
    design_t = torch.as_tensor(design, dtype=torch.float32)
    cell_line_count = len(tuple(cell_line_names))
    doses, alpha, log10_doses = _selector_dose_design_components(
        design_t,
        cell_line_count=cell_line_count,
        selector_temperature=selector_temperature,
        log10_dose_min=log10_dose_min,
        log10_dose_max=log10_dose_max,
    )
    selector_probs = alpha.detach().cpu().tolist()
    selected_index = int(torch.argmax(alpha).detach().cpu())
    dose_list = doses.detach().cpu().tolist()
    log10_dose_list = log10_doses.detach().cpu().tolist()
    return {
        "raw_design_vector": design_t.detach().cpu().tolist(),
        "cell_line_names": list(tuple(cell_line_names)),
        "selector_temperature": float(selector_temperature),
        "selector_probs": selector_probs,
        "selected_cell_line_index": selected_index,
        "selected_cell_line": str(tuple(cell_line_names)[selected_index]),
        "dose_count": int(doses.shape[0]),
        "doses": dose_list,
        "dose": float(dose_list[0]) if len(dose_list) == 1 else None,
        "log10_doses": log10_dose_list,
        "log10_dose": float(log10_dose_list[0]) if len(log10_dose_list) == 1 else None,
    }


def _bmp4_multireceptor_model(
    design: torch.Tensor,
    *,
    q_obs: torch.Tensor | None,
    y_obs: torch.Tensor | None,
    kd_prior_shift: torch.Tensor | None,
    prior: TranslatedPrior,
) -> torch.Tensor:
    design = design.float()
    n_cell_lines, _ = design.shape

    if q_obs is not None:
        q_obs = q_obs.float()
        n_receptors = q_obs.shape[1]
    else:
        n_receptors = make_distribution(prior.sites["log_kd"]).batch_shape[0]

    if kd_prior_shift is None:
        kd_prior_shift = torch.zeros(n_cell_lines, n_receptors, dtype=torch.float32)
    else:
        kd_prior_shift = kd_prior_shift.float()

    with pyro.plate("receptors", n_receptors, dim=-1):
        log_kd = pyro.sample("log_kd", make_distribution(prior.sites["log_kd"]))
        log_weight = pyro.sample("log_weight", make_distribution(prior.sites["log_weight"]))
        base_log_R = pyro.sample("base_log_R", make_distribution(prior.sites["base_log_R"]))
        qpcr_intercept = pyro.sample("qpcr_intercept", make_distribution(prior.sites["qpcr_intercept"]))
        qpcr_slope = pyro.sample("qpcr_slope", make_distribution(prior.sites["qpcr_slope"]))
        sigma_q = pyro.sample("sigma_q", make_distribution(prior.sites["sigma_q"]))

    sigma_R = pyro.sample("sigma_R", make_distribution(prior.sites["sigma_R"]))
    log_s50 = pyro.sample("log_s50", make_distribution(prior.sites["log_s50"]))
    response_hill = pyro.sample("response_hill", make_distribution(prior.sites["response_hill"]))

    with pyro.plate("cell_lines_for_response", n_cell_lines, dim=-1):
        bottom = pyro.sample("bottom", make_distribution(prior.sites["bottom"]))
        top = pyro.sample("top", make_distribution(prior.sites["top"]))
        sigma_y = pyro.sample("sigma_y", make_distribution(prior.sites["sigma_y"]))

    mean_log_R = base_log_R.unsqueeze(0) + kd_prior_shift

    with pyro.plate("cell_lines", n_cell_lines, dim=-2):
        with pyro.plate("receptors_for_abundance", n_receptors, dim=-1):
            log_R = pyro.sample("log_R", dist.Normal(mean_log_R, sigma_R))
            if q_obs is not None:
                pyro.sample(
                    "q_obs",
                    dist.Normal(qpcr_intercept + qpcr_slope * log_R, sigma_q),
                    obs=q_obs,
                )

    kd = torch.exp(log_kd)
    weight = torch.exp(log_weight)
    receptor_amount = torch.exp(log_R)
    mu = _response_mean(
        design=design,
        kd=kd.unsqueeze(0),
        weight=weight.unsqueeze(0),
        receptor_amount=receptor_amount.unsqueeze(0),
        bottom=bottom.unsqueeze(0),
        top=top.unsqueeze(0),
        log_s50=log_s50.reshape(1),
        response_hill=response_hill.reshape(1),
    ).squeeze(0)

    if y_obs is not None:
        pyro.sample(
            "y",
            dist.LogNormal(
                torch.log(torch.clamp(mu, min=1e-8)),
                sigma_y.unsqueeze(-1),
            ).to_event(2),
            obs=y_obs,
        )
    else:
        pyro.sample(
            "y",
            dist.LogNormal(
                torch.log(torch.clamp(mu, min=1e-8)),
                sigma_y.unsqueeze(-1),
            ).to_event(2),
        )
    return mu


def _response_mean(
    *,
    design: torch.Tensor,
    kd: torch.Tensor,
    weight: torch.Tensor,
    receptor_amount: torch.Tensor,
    bottom: torch.Tensor,
    top: torch.Tensor,
    log_s50: torch.Tensor,
    response_hill: torch.Tensor,
) -> torch.Tensor:
    design = torch.as_tensor(design, dtype=torch.float32)
    theta = design.unsqueeze(0).unsqueeze(-1) / (
        design.unsqueeze(0).unsqueeze(-1) + kd[:, None, None, :]
    )
    signal = torch.sum(
        theta * receptor_amount[:, :, None, :] * weight[:, None, None, :],
        dim=-1,
    )
    signal = torch.clamp(signal, min=1e-12)
    s50 = torch.exp(log_s50).reshape(-1, 1, 1)
    response_hill = response_hill.reshape(-1, 1, 1)
    effective_top = _effective_top(bottom, top)
    return bottom.unsqueeze(-1) + (effective_top - bottom).unsqueeze(-1) * (
        signal**response_hill / (s50**response_hill + signal**response_hill)
    )


def _selector_dose_design_components(
    design: torch.Tensor,
    *,
    cell_line_count: int,
    selector_temperature: float,
    log10_dose_min: float | None,
    log10_dose_max: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    design_t = torch.as_tensor(design, dtype=torch.float32)
    if design_t.ndim != 1 or design_t.shape[0] <= cell_line_count:
        raise ValueError(
            f"Expected selector+dose design vector of length > {cell_line_count}; got shape {tuple(design_t.shape)}."
        )
    class_logits = design_t[:cell_line_count]
    raw_doses = design_t[cell_line_count:]
    alpha = torch.softmax(class_logits / max(float(selector_temperature), 1e-6), dim=0)
    log10_doses, doses = transform_raw_dose_design_to_concentration(
        raw_doses,
        log10_dose_min=log10_dose_min,
        log10_dose_max=log10_dose_max,
    )
    return doses, alpha, log10_doses


def transform_raw_dose_design_to_concentration(
    raw_doses: torch.Tensor,
    *,
    log10_dose_min: float | None = None,
    log10_dose_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_doses_t = torch.as_tensor(raw_doses, dtype=torch.float32)
    lower, upper = _resolve_log10_dose_bounds(
        log10_dose_min=log10_dose_min,
        log10_dose_max=log10_dose_max,
    )
    log10_doses = lower + (upper - lower) * torch.sigmoid(raw_doses_t)
    doses = torch.pow(raw_doses_t.new_tensor(10.0), log10_doses)
    return log10_doses, doses


def make_log10_dose_bounds(
    *,
    min_positive_concentration: float,
    max_concentration: float,
    margin_decades: float = DEFAULT_LOG10_DOSE_MARGIN_DECADES,
) -> tuple[float, float]:
    positive_floor = max(float(min_positive_concentration), 1e-8)
    positive_ceiling = max(float(max_concentration), positive_floor * 10.0)
    margin = max(float(margin_decades), 0.0)
    log10_lower = math.log10(positive_floor) - margin
    log10_upper = math.log10(positive_ceiling) + margin
    if log10_upper <= log10_lower:
        log10_upper = log10_lower + 1.0
    return log10_lower, log10_upper


def _resolve_log10_dose_bounds(
    *,
    log10_dose_min: float | None,
    log10_dose_max: float | None,
) -> tuple[float, float]:
    lower = DEFAULT_LOG10_DOSE_MIN if log10_dose_min is None else float(log10_dose_min)
    upper = DEFAULT_LOG10_DOSE_MAX if log10_dose_max is None else float(log10_dose_max)
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _selector_dose_response_mean(
    *,
    doses: torch.Tensor,
    alpha: torch.Tensor,
    kd: torch.Tensor,
    weight: torch.Tensor,
    receptor_amount: torch.Tensor,
    bottom: torch.Tensor,
    top: torch.Tensor,
    log_s50: torch.Tensor,
    response_hill: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.as_tensor(alpha, dtype=torch.float32)
    doses_t = torch.as_tensor(doses, dtype=torch.float32)
    receptor_selected = _selector_reduce(alpha, receptor_amount)
    bottom_selected = _selector_reduce(alpha, bottom)
    top_selected = _selector_reduce(alpha, top)

    theta = doses_t.unsqueeze(0).unsqueeze(-1) / (doses_t.unsqueeze(0).unsqueeze(-1) + kd[:, None, :])
    signal = torch.sum(
        theta * receptor_selected[:, None, :] * weight[:, None, :],
        dim=-1,
    )
    signal = torch.clamp(signal, min=1e-12)
    s50 = torch.exp(log_s50).reshape(-1, 1)
    response_hill = response_hill.reshape(-1, 1)
    effective_top = _effective_top(bottom_selected, top_selected)
    return bottom_selected.unsqueeze(-1) + (effective_top - bottom_selected).unsqueeze(-1) * (
        signal**response_hill / (s50**response_hill + signal**response_hill)
    )


def _selector_reduce(alpha: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32)
    tensor_t = torch.as_tensor(tensor, dtype=torch.float32)
    if tensor_t.ndim == 3:
        return torch.einsum("c,scr->sr", alpha_t, tensor_t)
    if tensor_t.ndim == 2:
        return torch.einsum("c,sc->s", alpha_t, tensor_t)
    raise ValueError(f"Unsupported selector reduction tensor shape {tuple(tensor_t.shape)}.")


def _coerce_design_matrix(concentration: torch.Tensor, *, cell_line_count: int) -> torch.Tensor:
    design = torch.as_tensor(concentration, dtype=torch.float32)
    if design.ndim == 0:
        return design.reshape(1, 1).expand(cell_line_count, 1)
    if design.ndim == 1:
        return design.reshape(1, -1).expand(cell_line_count, -1)
    if design.ndim == 2:
        return design
    raise ValueError(f"Expected scalar, vector, or matrix design; got shape {tuple(design.shape)}.")


def _is_selector_dose_design(design: torch.Tensor, *, cell_line_count: int) -> bool:
    design_t = torch.as_tensor(design, dtype=torch.float32)
    return design_t.ndim == 1 and int(design_t.shape[0]) > int(cell_line_count)


def _coerce_observation_matrix(
    y_value: torch.Tensor,
    *,
    cell_line_count: int,
    design_points: int,
) -> torch.Tensor:
    y = torch.as_tensor(y_value, dtype=torch.float32)
    if y.ndim == 0:
        return y.reshape(1, 1).expand(cell_line_count, design_points)
    if y.ndim == 1:
        if y.shape[0] == cell_line_count and design_points == 1:
            return y.reshape(cell_line_count, 1)
        return y.reshape(1, -1).expand(cell_line_count, -1)
    if y.ndim == 2:
        return y
    raise ValueError(f"Expected scalar, vector, or matrix response; got shape {tuple(y.shape)}.")


def _coerce_future_observation_vector(y_value: torch.Tensor, *, dose_count: int) -> torch.Tensor:
    y = torch.as_tensor(y_value, dtype=torch.float32)
    if y.ndim == 0:
        return y.reshape(1).expand(dose_count)
    if y.ndim == 1:
        if y.shape[0] != dose_count:
            raise ValueError(
                f"Expected future observation vector of length {dose_count}; got shape {tuple(y.shape)}."
            )
        return y
    raise ValueError(f"Expected scalar or vector future response; got shape {tuple(y.shape)}.")


def _sample_count_from_samples(posterior_samples: dict[str, torch.Tensor]) -> int:
    tensor = torch.as_tensor(posterior_samples["response_hill"], dtype=torch.float32)
    if tensor.ndim == 0:
        return 1
    return int(tensor.shape[0])


def _cell_line_count_from_samples(
    posterior_samples: dict[str, torch.Tensor],
    *,
    sample_count: int,
) -> int:
    tensor = torch.as_tensor(posterior_samples["bottom"], dtype=torch.float32)
    return int(tensor.numel() // sample_count)


def _receptor_count_from_samples(
    posterior_samples: dict[str, torch.Tensor],
    *,
    sample_count: int,
) -> int:
    tensor = torch.as_tensor(posterior_samples["log_kd"], dtype=torch.float32)
    return int(tensor.numel() // sample_count)


def _reshape_site(samples: torch.Tensor, *shape: int) -> torch.Tensor:
    tensor = torch.as_tensor(samples, dtype=torch.float32)
    return tensor.reshape(shape)


def _effective_top(bottom: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
    return torch.maximum(top, bottom + 1e-6)


__all__ = [
    "DEFAULT_LOG10_DOSE_MARGIN_DECADES",
    "SELECTOR_TEMPERATURE",
    "decode_selector_dose_design",
    "FIT_PARAMETER_NAMES",
    "LITERATURE_PARAMETER_NAMES",
    "make_fit_model",
    "make_log10_dose_bounds",
    "predictive_draws",
    "problem_description",
    "scalar_log_likelihood",
    "transform_raw_dose_design_to_concentration",
]
