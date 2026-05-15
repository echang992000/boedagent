from __future__ import annotations

from typing import Any

import pyro
import pyro.distributions as dist
import torch

from ..priors import TranslatedPrior, make_distribution
from ..simulators.differentiable.core import hill_function


FIT_PARAMETER_NAMES = ("bottom", "top", "ec50", "hill_n", "sigma")
LITERATURE_PARAMETER_NAMES = ("bottom", "top", "ec50", "hill_n", "sigma")


def make_fit_model(prior: TranslatedPrior):
    def model(concentration: torch.Tensor, y_obs: torch.Tensor | None = None) -> torch.Tensor:
        concentration_t = torch.as_tensor(concentration, dtype=torch.float32)
        bottom = pyro.sample("bottom", make_distribution(prior.sites["bottom"]))
        top = pyro.sample("top", make_distribution(prior.sites["top"]))
        ec50 = pyro.sample("ec50", make_distribution(prior.sites["ec50"]))
        hill_n = pyro.sample("hill_n", make_distribution(prior.sites["hill_n"]))
        sigma = pyro.sample("sigma", make_distribution(prior.sites["sigma"]))
        mu = hill_function(
            concentration=concentration_t,
            top=top,
            bottom=bottom,
            ec50=ec50,
            hill_n=hill_n,
        )
        with pyro.plate("data", concentration_t.shape[0]):
            pyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)
        return mu

    return model


def predictive_draws(concentration: torch.Tensor, posterior_samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    concentration_t = torch.as_tensor(concentration, dtype=torch.float32)
    grid = concentration_t.reshape(1, -1)
    top = posterior_samples["top"].reshape(-1, 1)
    bottom = posterior_samples["bottom"].reshape(-1, 1)
    ec50 = posterior_samples["ec50"].reshape(-1, 1)
    hill_n = posterior_samples["hill_n"].reshape(-1, 1)
    sigma = posterior_samples["sigma"].reshape(-1, 1)
    mu = bottom + (top - bottom) * (grid**hill_n) / (ec50**hill_n + grid**hill_n)
    y = dist.Normal(mu, sigma).rsample()
    return {"mu": mu, "y": y}


def scalar_log_likelihood(
    y_value: torch.Tensor,
    concentration: torch.Tensor,
    posterior_samples: dict[str, torch.Tensor],
) -> torch.Tensor:
    concentration_t = torch.as_tensor(concentration, dtype=torch.float32)
    top = posterior_samples["top"]
    bottom = posterior_samples["bottom"]
    ec50 = posterior_samples["ec50"]
    hill_n = posterior_samples["hill_n"]
    sigma = posterior_samples["sigma"]
    mu = bottom + (top - bottom) * (concentration_t**hill_n) / (ec50**hill_n + concentration_t**hill_n)
    return dist.Normal(mu, sigma).log_prob(y_value)


def problem_description(problem_summary: str) -> str:
    return (
        f"{problem_summary} Candidate model family: 4-parameter Hill dose-response "
        "curve with observation noise. Infer bottom, top, EC50, Hill slope, and noise."
    )


__all__ = [
    "FIT_PARAMETER_NAMES",
    "LITERATURE_PARAMETER_NAMES",
    "make_fit_model",
    "predictive_draws",
    "problem_description",
    "scalar_log_likelihood",
]
