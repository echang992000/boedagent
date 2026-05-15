from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import pyro.distributions as dist


@dataclass
class VariationalFitResult:
    loss_history: list[float]
    posterior_samples: dict[str, torch.Tensor]
    fit_predictive: dict[str, torch.Tensor]


def fit_variational_model(
    model: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    concentration: torch.Tensor,
    y_obs: torch.Tensor,
    *,
    num_steps: int,
    learning_rate: float,
    num_posterior_samples: int,
) -> VariationalFitResult:
    import pyro
    from pyro.infer import Predictive, SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDiagonalNormal

    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    svi = SVI(
        model=model,
        guide=guide,
        optim=pyro.optim.Adam({"lr": learning_rate}),
        loss=Trace_ELBO(),
    )

    losses: list[float] = []
    for _ in range(int(num_steps)):
        loss = svi.step(concentration, y_obs)
        losses.append(float(loss))

    predictive = Predictive(model, guide=guide, num_samples=int(num_posterior_samples))
    posterior_bundle = predictive(concentration, None)
    posterior_samples = {
        name: value.detach().cpu()
        for name, value in posterior_bundle.items()
        if name not in {"y", "q_obs", "_RETURN"}
    }
    fit_predictive = {
        name: value.detach().cpu()
        for name, value in posterior_bundle.items()
    }
    return VariationalFitResult(
        loss_history=losses,
        posterior_samples=posterior_samples,
        fit_predictive=fit_predictive,
    )


def summarize_posterior_samples(samples: dict[str, torch.Tensor]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, tensor in samples.items():
        flat = tensor.detach().cpu().reshape(-1).float()
        summary[name] = {
            "mean": float(flat.mean()),
            "std": float(flat.std(unbiased=False)),
            "q05": float(torch.quantile(flat, 0.05)),
            "q50": float(torch.quantile(flat, 0.5)),
            "q95": float(torch.quantile(flat, 0.95)),
        }
    return summary


def optimize_empirical_eig(
    *,
    posterior_samples: dict[str, torch.Tensor],
    scalar_log_likelihood: Callable[[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], Sequence[str]], torch.Tensor],
    receptor_names: Sequence[str],
    lower: float,
    upper: float,
    min_positive_lower: float,
    num_steps: int,
    learning_rate: float,
    outer_samples: int,
) -> dict[str, Any]:
    log_lower = math.log(max(lower, min_positive_lower, 1e-6))
    log_upper = math.log(max(upper, math.exp(log_lower) * 10.0))
    raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    history: list[dict[str, float]] = []

    for step in range(int(num_steps)):
        optimizer.zero_grad()
        design = torch.exp(log_lower + (log_upper - log_lower) * torch.sigmoid(raw))
        eig = _estimate_empirical_eig(
            design=design,
            posterior_samples=posterior_samples,
            scalar_log_likelihood=scalar_log_likelihood,
            receptor_names=receptor_names,
            outer_samples=outer_samples,
        )
        loss = -eig
        loss.backward()
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "design": float(design.detach().cpu()),
                "eig": float(eig.detach().cpu()),
            }
        )

    best = max(history, key=lambda item: item["eig"])
    return {
        "history": history,
        "best_design": best["design"],
        "best_eig": best["eig"],
        "optimization_bounds": {
            "raw_lower": float(lower),
            "raw_upper": float(upper),
            "positive_lower_used": float(math.exp(log_lower)),
        },
    }


def optimize_selector_dose_empirical_eig(
    *,
    posterior_samples: dict[str, torch.Tensor],
    scalar_log_likelihood: Callable[[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], Sequence[str]], torch.Tensor],
    receptor_names: Sequence[str],
    design_decoder: Callable[[torch.Tensor], dict[str, Any]],
    cell_line_count: int,
    dose_count: int,
    log10_dose_min: float | None,
    log10_dose_max: float | None,
    num_steps: int,
    learning_rate: float,
    outer_samples: int,
) -> dict[str, Any]:
    raw_design = torch.zeros(cell_line_count + dose_count, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw_design], lr=learning_rate)
    history: list[dict[str, Any]] = []

    for step in range(int(num_steps)):
        optimizer.zero_grad()
        eig = _estimate_empirical_eig(
            design=raw_design,
            posterior_samples=posterior_samples,
            scalar_log_likelihood=scalar_log_likelihood,
            receptor_names=receptor_names,
            outer_samples=outer_samples,
            selector_log10_dose_min=log10_dose_min,
            selector_log10_dose_max=log10_dose_max,
        )
        loss = -eig
        loss.backward()
        optimizer.step()

        decoded = dict(design_decoder(raw_design.detach()))
        decoded.update(
            {
                "step": float(step),
                "eig": float(eig.detach().cpu()),
                "design": float(decoded["dose"]) if decoded.get("dose") is not None else float(decoded["doses"][0]),
            }
        )
        history.append(decoded)

    best = max(history, key=lambda item: item["eig"])
    return {
        "history": history,
        "best_design": best["design"],
        "best_dose": best.get("dose"),
        "best_doses": best.get("doses"),
        "best_log10_dose": best.get("log10_dose"),
        "best_log10_doses": best.get("log10_doses"),
        "best_cell_line": best.get("selected_cell_line"),
        "best_cell_line_index": best.get("selected_cell_line_index"),
        "best_selector_probs": best.get("selector_probs"),
        "best_eig": best["eig"],
        "design_type": "cell_line_selector_plus_bmp4_dose",
        "dose_count": int(dose_count),
        "optimization_bounds": (
            {
                "log10_dose_min": float(log10_dose_min),
                "log10_dose_max": float(log10_dose_max),
                "dose_min": float(10.0 ** float(log10_dose_min)),
                "dose_max": float(10.0 ** float(log10_dose_max)),
            }
            if log10_dose_min is not None and log10_dose_max is not None
            else None
        ),
    }


def _estimate_empirical_eig(
    *,
    design: torch.Tensor,
    posterior_samples: dict[str, torch.Tensor],
    scalar_log_likelihood: Callable[[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], Sequence[str]], torch.Tensor],
    receptor_names: Sequence[str],
    outer_samples: int,
    selector_log10_dose_min: float | None = None,
    selector_log10_dose_max: float | None = None,
) -> torch.Tensor:
    noise_key = "sigma_y" if "sigma_y" in posterior_samples else "sigma"
    likelihood_count = posterior_samples[noise_key].shape[0]
    sample_count = min(int(outer_samples), int(likelihood_count))
    indices = torch.randperm(likelihood_count)[:sample_count]
    log_cond_terms: list[torch.Tensor] = []
    log_marg_terms: list[torch.Tensor] = []

    for index in indices:
        single_sample = {
            name: value[index]
            for name, value in posterior_samples.items()
        }
        y_draw = _rsample_observation(
            single_sample,
            design,
            receptor_names,
            selector_log10_dose_min=selector_log10_dose_min,
            selector_log10_dose_max=selector_log10_dose_max,
        )
        log_cond = scalar_log_likelihood(
            y_draw,
            design,
            _expand_single_sample(single_sample),
            receptor_names,
        ).reshape(-1)[0]
        log_all = scalar_log_likelihood(y_draw, design, posterior_samples, receptor_names)
        log_marg = torch.logsumexp(log_all, dim=0) - math.log(likelihood_count)
        log_cond_terms.append(log_cond)
        log_marg_terms.append(log_marg)

    return torch.stack(log_cond_terms).mean() - torch.stack(log_marg_terms).mean()


def _rsample_observation(
    sample: dict[str, torch.Tensor],
    design: torch.Tensor,
    receptor_names: Sequence[str],
    *,
    selector_log10_dose_min: float | None = None,
    selector_log10_dose_max: float | None = None,
) -> torch.Tensor:
    if "sigma_y" in sample:
        sigma = sample["sigma_y"]
    else:
        sigma = sample["sigma"]

    if {"bottom", "top", "ec50", "hill_n"}.issubset(sample):
        mu = sample["bottom"] + (sample["top"] - sample["bottom"]) * (design**sample["hill_n"]) / (
            sample["ec50"]**sample["hill_n"] + design**sample["hill_n"]
        )
        return mu + sigma * torch.randn_like(mu)

    if {"log_kd", "log_weight", "log_R", "log_s50", "response_hill"}.issubset(sample):
        log_kd = torch.as_tensor(sample["log_kd"], dtype=torch.float32).squeeze()
        log_weight = torch.as_tensor(sample["log_weight"], dtype=torch.float32).squeeze()
        log_R = torch.as_tensor(sample["log_R"], dtype=torch.float32).squeeze()
        bottom = torch.as_tensor(sample["bottom"], dtype=torch.float32).squeeze()
        top = torch.as_tensor(sample["top"], dtype=torch.float32).squeeze()
        sigma_y = torch.as_tensor(sample["sigma_y"], dtype=torch.float32).squeeze()
        kd = torch.exp(log_kd)
        weight = torch.exp(log_weight)
        receptor_amount = torch.exp(log_R)
        s50 = torch.exp(torch.as_tensor(sample["log_s50"], dtype=torch.float32).squeeze())
        response_hill = torch.as_tensor(sample["response_hill"], dtype=torch.float32).squeeze()
        design_t = torch.as_tensor(design, dtype=torch.float32)

        if design_t.ndim == 1 and design_t.shape[0] > bottom.shape[0]:
            from .pyro.multireceptor_hierarchical import (
                SELECTOR_TEMPERATURE,
                transform_raw_dose_design_to_concentration,
            )

            class_logits = design_t[: bottom.shape[0]]
            raw_doses = design_t[bottom.shape[0] :]
            alpha = torch.softmax(class_logits / SELECTOR_TEMPERATURE, dim=0)
            _, doses = transform_raw_dose_design_to_concentration(
                raw_doses,
                log10_dose_min=selector_log10_dose_min,
                log10_dose_max=selector_log10_dose_max,
            )

            receptor_selected = torch.einsum("c,cr->r", alpha, receptor_amount)
            bottom_selected = torch.dot(alpha, bottom)
            top_selected = torch.dot(alpha, top)
            sigma_selected = torch.dot(alpha, sigma_y)

            theta = doses.unsqueeze(-1) / (doses.unsqueeze(-1) + kd.unsqueeze(0))
            signal = torch.sum(theta * receptor_selected.unsqueeze(0) * weight.unsqueeze(0), dim=-1)
            signal = torch.clamp(signal, min=1e-12)
            effective_top = torch.maximum(top_selected, bottom_selected + 1e-6)
            mu = bottom_selected + (effective_top - bottom_selected) * (
                signal**response_hill / (s50**response_hill + signal**response_hill)
            )
            return dist.LogNormal(
                torch.log(torch.clamp(mu, min=1e-8)),
                sigma_selected,
            ).rsample()

        design_matrix = design_t.reshape(1, 1).expand(bottom.shape[0], 1)
        theta = design_matrix.unsqueeze(-1) / (design_matrix.unsqueeze(-1) + kd.unsqueeze(0).unsqueeze(0))
        signal = torch.sum(theta * receptor_amount.unsqueeze(1) * weight.unsqueeze(0).unsqueeze(0), dim=-1)
        signal = torch.clamp(signal, min=1e-12)
        effective_top = torch.maximum(top, bottom + 1e-6)
        mu = bottom.unsqueeze(-1) + (effective_top - bottom).unsqueeze(-1) * (
            signal**response_hill / (s50**response_hill + signal**response_hill)
        )
        return dist.LogNormal(
            torch.log(torch.clamp(mu, min=1e-8)),
            sigma_y.unsqueeze(-1),
        ).rsample().squeeze(-1)

    ordered_receptors = [
        name for name in receptor_names if f"kd_{name}" in sample
    ]
    kd = torch.stack([sample[f"kd_{name}"] for name in ordered_receptors])
    abundance = torch.stack([sample[f"abundance_{name}"] for name in ordered_receptors])
    weight = torch.stack([sample[f"weight_{name}"] for name in ordered_receptors])
    occupancy = design / (kd + design)
    signal = (occupancy * abundance * weight).sum()
    signal = torch.clamp(signal, min=1e-12)
    mu = sample["bottom"] + (sample["top"] - sample["bottom"]) * (signal**sample["response_hill"]) / (
        sample["s50"]**sample["response_hill"] + signal**sample["response_hill"]
    )
    return mu + sigma * torch.randn_like(mu)


def _expand_single_sample(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    expanded: dict[str, torch.Tensor] = {}
    for name, value in sample.items():
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            expanded[name] = tensor.reshape(1)
        else:
            expanded[name] = tensor.unsqueeze(0)
    return expanded


__all__ = [
    "VariationalFitResult",
    "fit_variational_model",
    "optimize_empirical_eig",
    "optimize_selector_dose_empirical_eig",
    "summarize_posterior_samples",
]
