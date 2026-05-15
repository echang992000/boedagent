from __future__ import annotations

from typing import Optional

import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints

from ..simulators.differentiable.core import (
    hill_function,
    simple_multireceptor_dose_response,
)


TensorLike = torch.Tensor | float


def pyro_hill_model(
    concentration: torch.Tensor,
    y_obs: Optional[torch.Tensor] = None,
) -> None:
    """
    Bayesian 4-parameter Hill model.

    concentration: [N]
    y_obs: [N] or None
    """
    concentration = concentration.float()

    bottom = pyro.sample("bottom", dist.Normal(0.0, 10.0))
    top = pyro.sample("top", dist.Normal(1.0, 10.0))
    ec50 = pyro.sample("ec50", dist.LogNormal(0.0, 2.0))
    hill_n = pyro.sample("hill_n", dist.LogNormal(0.0, 1.0))
    sigma = pyro.sample("sigma", dist.LogNormal(-2.0, 1.0))

    mu = hill_function(
        concentration=concentration,
        top=top,
        bottom=bottom,
        ec50=ec50,
        hill_n=hill_n,
    )

    with pyro.plate("data", concentration.shape[0]):
        pyro.sample("y", dist.Normal(mu, sigma), obs=y_obs)


def pyro_multireceptor_dose_response_model(
    concentration: torch.Tensor,
    n_receptors: int,
    y_obs: Optional[torch.Tensor] = None,
    kd_obs: Optional[torch.Tensor] = None,
    qpcr_obs: Optional[torch.Tensor] = None,
    fix_kd_to_observed: bool = False,
) -> None:
    """
    Bayesian multireceptor dose-response model for a single ligand titration curve.

    concentration: [N]
    y_obs: [N] observed response
    kd_obs: [R] optional observed SPR Kd values
    qpcr_obs: [R] optional noisy proxy for receptor abundance (e.g. log-scale qPCR)
    fix_kd_to_observed:
        If True, use observed kd values directly.
        If False, treat Kd as latent and optionally anchor with kd_obs.

    Generative structure
    --------------------
    theta_i(C) = C / (Kd_i + C)
    S(C) = sum_i abundance_i * weight_i * theta_i(C)
    Y(C) = bottom + (top - bottom) * S(C)^m / (S50^m + S(C)^m)

    Optional measurement models
    ---------------------------
    log(Kd_obs_i) ~ Normal(log(Kd_i), sigma_kd)
    qpcr_obs_i ~ Normal(qpcr_intercept + qpcr_slope * log(abundance_i), sigma_qpcr)
    """
    concentration = concentration.float()
    R = int(n_receptors)

    if kd_obs is not None:
        kd_obs = kd_obs.float()
        if kd_obs.shape != (R,):
            raise ValueError(f"kd_obs must have shape ({R},)")

    if qpcr_obs is not None:
        qpcr_obs = qpcr_obs.float()
        if qpcr_obs.shape != (R,):
            raise ValueError(f"qpcr_obs must have shape ({R},)")

    # Shared response parameters
    bottom = pyro.sample("bottom", dist.Normal(0.0, 10.0))
    top = pyro.sample("top", dist.Normal(1.0, 10.0))
    s50 = pyro.sample("s50", dist.LogNormal(0.0, 2.0))
    response_hill = pyro.sample("response_hill", dist.LogNormal(0.0, 1.0))
    sigma_y = pyro.sample("sigma_y", dist.LogNormal(-2.0, 1.0))

    # Receptor-specific latent variables
    with pyro.plate("receptors", R):
        if fix_kd_to_observed:
            if kd_obs is None:
                raise ValueError("kd_obs must be provided when fix_kd_to_observed=True")
            kd = pyro.deterministic("kd", kd_obs)
        else:
            kd = pyro.sample("kd", dist.LogNormal(0.0, 2.0))
            if kd_obs is not None:
                sigma_kd = pyro.sample("sigma_kd", dist.LogNormal(-2.0, 1.0))
                pyro.sample(
                    "kd_obs",
                    dist.LogNormal(torch.log(kd), sigma_kd),
                    obs=kd_obs,
                )

        abundance = pyro.sample("abundance", dist.LogNormal(0.0, 1.0))
        weight = pyro.sample("weight", dist.LogNormal(0.0, 1.0))

        if qpcr_obs is not None:
            # Simple noisy measurement model:
            # qPCR proxy is modeled on the log-abundance scale.
            qpcr_intercept = pyro.sample("qpcr_intercept", dist.Normal(0.0, 5.0))
            qpcr_slope = pyro.sample("qpcr_slope", dist.LogNormal(0.0, 0.5))
            sigma_qpcr = pyro.sample("sigma_qpcr", dist.LogNormal(-1.0, 1.0))
            pyro.sample(
                "qpcr",
                dist.Normal(qpcr_intercept + qpcr_slope * torch.log(abundance), sigma_qpcr),
                obs=qpcr_obs,
            )

    mu = simple_multireceptor_dose_response(
        concentration=concentration,
        kd=kd,
        receptor_abundance=abundance,
        receptor_weight=weight,
        top=top,
        bottom=bottom,
        s50=s50,
        response_hill=response_hill,
    )

    with pyro.plate("data", concentration.shape[0]):
        pyro.sample("y", dist.Normal(mu, sigma_y), obs=y_obs)


# ---------------------------------------------------------------------
# Optional guide for SVI
# ---------------------------------------------------------------------

def pyro_hill_guide(
    concentration: torch.Tensor,
    y_obs: Optional[torch.Tensor] = None,
) -> None:
    """
    Mean-field guide for the Hill model.
    """
    bottom_loc = pyro.param("bottom_loc", torch.tensor(0.0))
    bottom_scale = pyro.param("bottom_scale", torch.tensor(1.0), constraint=constraints.positive)

    top_loc = pyro.param("top_loc", torch.tensor(1.0))
    top_scale = pyro.param("top_scale", torch.tensor(1.0), constraint=constraints.positive)

    ec50_loc = pyro.param("ec50_loc", torch.tensor(0.0))
    ec50_scale = pyro.param("ec50_scale", torch.tensor(1.0), constraint=constraints.positive)

    hill_n_loc = pyro.param("hill_n_loc", torch.tensor(0.0))
    hill_n_scale = pyro.param("hill_n_scale", torch.tensor(0.5), constraint=constraints.positive)

    sigma_loc = pyro.param("sigma_loc", torch.tensor(-2.0))
    sigma_scale = pyro.param("sigma_scale", torch.tensor(0.5), constraint=constraints.positive)

    pyro.sample("bottom", dist.Normal(bottom_loc, bottom_scale))
    pyro.sample("top", dist.Normal(top_loc, top_scale))
    pyro.sample("ec50", dist.LogNormal(ec50_loc, ec50_scale))
    pyro.sample("hill_n", dist.LogNormal(hill_n_loc, hill_n_scale))
    pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))


def pyro_multireceptor_guide(
    concentration: torch.Tensor,
    n_receptors: int,
    y_obs: Optional[torch.Tensor] = None,
    kd_obs: Optional[torch.Tensor] = None,
    qpcr_obs: Optional[torch.Tensor] = None,
    fix_kd_to_observed: bool = False,
) -> None:
    """
    Mean-field guide for the multireceptor model.
    """
    R = int(n_receptors)

    pyro.sample(
        "bottom",
        dist.Normal(
            pyro.param("bottom_loc", torch.tensor(0.0)),
            pyro.param("bottom_scale", torch.tensor(1.0), constraint=constraints.positive),
        ),
    )
    pyro.sample(
        "top",
        dist.Normal(
            pyro.param("top_loc", torch.tensor(1.0)),
            pyro.param("top_scale", torch.tensor(1.0), constraint=constraints.positive),
        ),
    )
    pyro.sample(
        "s50",
        dist.LogNormal(
            pyro.param("s50_loc", torch.tensor(0.0)),
            pyro.param("s50_scale", torch.tensor(1.0), constraint=constraints.positive),
        ),
    )
    pyro.sample(
        "response_hill",
        dist.LogNormal(
            pyro.param("response_hill_loc", torch.tensor(0.0)),
            pyro.param("response_hill_scale", torch.tensor(0.5), constraint=constraints.positive),
        ),
    )
    pyro.sample(
        "sigma_y",
        dist.LogNormal(
            pyro.param("sigma_y_loc", torch.tensor(-2.0)),
            pyro.param("sigma_y_scale", torch.tensor(0.5), constraint=constraints.positive),
        ),
    )

    if not fix_kd_to_observed:
        pyro.sample(
            "sigma_kd",
            dist.LogNormal(
                pyro.param("sigma_kd_loc", torch.tensor(-2.0)),
                pyro.param("sigma_kd_scale", torch.tensor(0.5), constraint=constraints.positive),
            ),
        )

    if qpcr_obs is not None:
        pyro.sample(
            "qpcr_intercept",
            dist.Normal(
                pyro.param("qpcr_intercept_loc", torch.tensor(0.0)),
                pyro.param("qpcr_intercept_scale", torch.tensor(1.0), constraint=constraints.positive),
            ),
        )
        pyro.sample(
            "qpcr_slope",
            dist.LogNormal(
                pyro.param("qpcr_slope_loc", torch.tensor(0.0)),
                pyro.param("qpcr_slope_scale", torch.tensor(0.3), constraint=constraints.positive),
            ),
        )
        pyro.sample(
            "sigma_qpcr",
            dist.LogNormal(
                pyro.param("sigma_qpcr_loc", torch.tensor(-1.0)),
                pyro.param("sigma_qpcr_scale", torch.tensor(0.5), constraint=constraints.positive),
            ),
        )

    with pyro.plate("receptors", R):
        if not fix_kd_to_observed:
            pyro.sample(
                "kd",
                dist.LogNormal(
                    pyro.param("kd_loc", torch.zeros(R)),
                    pyro.param("kd_scale", torch.ones(R), constraint=constraints.positive),
                ),
            )

        pyro.sample(
            "abundance",
            dist.LogNormal(
                pyro.param("abundance_loc", torch.zeros(R)),
                pyro.param("abundance_scale", torch.ones(R), constraint=constraints.positive),
            ),
        )
        pyro.sample(
            "weight",
            dist.LogNormal(
                pyro.param("weight_loc", torch.zeros(R)),
                pyro.param("weight_scale", torch.ones(R), constraint=constraints.positive),
            ),
        )
