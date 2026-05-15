from __future__ import annotations

from typing import Sequence

import pyro
import pyro.distributions as dist
import torch

from ..priors import TranslatedPrior, make_distribution
from ..simulators.differentiable.core import simple_multireceptor_dose_response


FIT_PARAMETER_NAMES = ("bottom", "top", "s50", "response_hill", "sigma_y")
LITERATURE_PARAMETER_NAMES = ("kd", "weight", "bottom", "top", "s50", "response_hill", "sigma_y")


def make_fit_model(
    prior: TranslatedPrior,
    *,
    receptor_names: Sequence[str],
):
    receptor_names = tuple(receptor_names)

    def model(concentration: torch.Tensor, y_obs: torch.Tensor | None = None) -> torch.Tensor:
        concentration_t = torch.as_tensor(concentration, dtype=torch.float32)

        bottom = pyro.sample("bottom", make_distribution(prior.sites["bottom"]))
        top = pyro.sample("top", make_distribution(prior.sites["top"]))
        s50 = pyro.sample("s50", make_distribution(prior.sites["s50"]))
        response_hill = pyro.sample(
            "response_hill",
            make_distribution(prior.sites["response_hill"]),
        )
        sigma_y = pyro.sample("sigma_y", make_distribution(prior.sites["sigma_y"]))

        kd = torch.stack(
            [
                pyro.sample(f"kd_{name}", make_distribution(prior.sites[f"kd_{name}"]))
                for name in receptor_names
            ]
        )
        abundance = torch.stack(
            [
                pyro.sample(
                    f"abundance_{name}",
                    make_distribution(prior.sites[f"abundance_{name}"]),
                )
                for name in receptor_names
            ]
        )
        weight = torch.stack(
            [
                pyro.sample(
                    f"weight_{name}",
                    make_distribution(prior.sites[f"weight_{name}"]),
                )
                for name in receptor_names
            ]
        )

        mu = simple_multireceptor_dose_response(
            concentration=concentration_t,
            kd=kd,
            receptor_abundance=abundance,
            receptor_weight=weight,
            top=top,
            bottom=bottom,
            s50=s50,
            response_hill=response_hill,
        )
        with pyro.plate("data", concentration_t.shape[0]):
            pyro.sample("y", dist.Normal(mu, sigma_y), obs=y_obs)
        return mu

    return model


def predictive_draws(
    concentration: torch.Tensor,
    posterior_samples: dict[str, torch.Tensor],
    *,
    receptor_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    concentration_t = torch.as_tensor(concentration, dtype=torch.float32).reshape(1, -1)
    kd = _stack_receptor_samples("kd", posterior_samples, receptor_names)
    abundance = _stack_receptor_samples("abundance", posterior_samples, receptor_names)
    weight = _stack_receptor_samples("weight", posterior_samples, receptor_names)
    top = _as_scalar_samples(posterior_samples["top"]).reshape(-1, 1)
    bottom = _as_scalar_samples(posterior_samples["bottom"]).reshape(-1, 1)
    s50 = _as_scalar_samples(posterior_samples["s50"]).reshape(-1, 1)
    response_hill = _as_scalar_samples(posterior_samples["response_hill"]).reshape(-1, 1)
    sigma_y = _as_scalar_samples(posterior_samples["sigma_y"]).reshape(-1, 1)

    occupancy = concentration_t[:, :, None] / (kd[:, None, :] + concentration_t[:, :, None])
    signal = (occupancy * abundance[:, None, :] * weight[:, None, :]).sum(dim=-1)
    signal = torch.clamp(signal, min=1e-12)
    mu = bottom + (top - bottom) * (signal**response_hill) / (s50**response_hill + signal**response_hill)
    y = dist.Normal(mu, sigma_y).rsample()
    return {"mu": mu, "y": y}


def scalar_log_likelihood(
    y_value: torch.Tensor,
    concentration: torch.Tensor,
    posterior_samples: dict[str, torch.Tensor],
    *,
    receptor_names: Sequence[str],
) -> torch.Tensor:
    concentration_t = torch.as_tensor(concentration, dtype=torch.float32)
    kd = _stack_receptor_samples("kd", posterior_samples, receptor_names)
    abundance = _stack_receptor_samples("abundance", posterior_samples, receptor_names)
    weight = _stack_receptor_samples("weight", posterior_samples, receptor_names)
    top = _as_scalar_samples(posterior_samples["top"])
    bottom = _as_scalar_samples(posterior_samples["bottom"])
    s50 = _as_scalar_samples(posterior_samples["s50"])
    response_hill = _as_scalar_samples(posterior_samples["response_hill"])
    sigma_y = _as_scalar_samples(posterior_samples["sigma_y"])

    occupancy = concentration_t / (kd + concentration_t)
    signal = (occupancy * abundance * weight).sum(dim=-1)
    signal = torch.clamp(signal, min=1e-12)
    mu = bottom + (top - bottom) * (signal**response_hill) / (s50**response_hill + signal**response_hill)
    return dist.Normal(mu, sigma_y).log_prob(y_value)


def problem_description(problem_summary: str, receptor_names: Sequence[str]) -> str:
    receptor_text = ", ".join(receptor_names)
    return (
        f"{problem_summary} Candidate model family: multireceptor BMP4 dose-response "
        f"model over receptors {receptor_text}. Infer shared binding strengths and "
        "signaling weights across the receptor set with cell-line-specific abundance priors. "
        "For receptor affinity priors from SPR dissociation constants, convert to the "
        "EQTK affinity scale used by this BMP4 code as K_eqtk = 1e-8 / K_d, where K_d "
        "is in molar units; equivalently K_eqtk = 10 / K_d_nM or K_eqtk = 10000 / K_d_pM. "
        "Use K_eqtk, not raw K_d, as the positive-scale prior center for kd_* parameters."
    )


def _stack_receptor_samples(
    prefix: str,
    posterior_samples: dict[str, torch.Tensor],
    receptor_names: Sequence[str],
) -> torch.Tensor:
    return torch.stack(
        [_as_scalar_samples(posterior_samples[f"{prefix}_{name}"]) for name in receptor_names],
        dim=-1,
    )


def _as_scalar_samples(samples: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(samples, dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim > 1 and all(size == 1 for size in tensor.shape[1:]):
        return tensor.reshape(tensor.shape[0])
    return tensor


__all__ = [
    "FIT_PARAMETER_NAMES",
    "LITERATURE_PARAMETER_NAMES",
    "make_fit_model",
    "predictive_draws",
    "problem_description",
    "scalar_log_likelihood",
]
