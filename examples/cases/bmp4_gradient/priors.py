from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pyro.distributions as pyro_dist
import torch
import torch.distributions as torch_dist

from boed_agent.prior_builder import AugmentedPrior, DistributionSpec


class TruncatedLogNormal(pyro_dist.TorchDistribution):
    arg_constraints = {
        "loc": torch_dist.constraints.real,
        "scale": torch_dist.constraints.positive,
        "low": torch_dist.constraints.nonnegative,
        "high": torch_dist.constraints.positive,
    }
    support = torch_dist.constraints.nonnegative
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
        low: torch.Tensor | float,
        high: torch.Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc = torch.as_tensor(loc, dtype=torch.float32)
        self.scale = torch.as_tensor(scale, dtype=torch.float32)
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        base_shape = torch.broadcast_shapes(
            self.loc.shape,
            self.scale.shape,
            self.low.shape,
            self.high.shape,
        )
        super().__init__(batch_shape=base_shape, validate_args=validate_args)
        self._base = torch_dist.Normal(self.loc, self.scale)
        self._effective_low = torch.clamp(self.low, min=1e-12)
        self._log_low = torch.log(self._effective_low)
        self._log_high = torch.log(self.high)
        self._cdf_low = self._base.cdf(self._log_low)
        self._cdf_high = self._base.cdf(self._log_high)
        self._normalizer = torch.clamp(self._cdf_high - self._cdf_low, min=1e-12)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        uniform = torch_dist.Uniform(self._cdf_low, self._cdf_high)
        u = uniform.rsample(sample_shape)
        z = self._base.icdf(u)
        return torch.exp(z)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32)
        in_bounds = (value >= self.low) & (value <= self.high)
        log_value = torch.log(torch.clamp(value, min=1e-12))
        base_log_prob = self._base.log_prob(log_value) - log_value
        adjusted = base_log_prob - torch.log(self._normalizer)
        return torch.where(
            in_bounds,
            adjusted,
            torch.full_like(adjusted, float("-inf")),
        )


@dataclass(frozen=True)
class PriorSiteConfig:
    name: str
    distribution: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "fallback"
    reasoning: str = ""
    cited_papers: list[str] = field(default_factory=list)
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "params": dict(self.params),
            "source": self.source,
            "reasoning": self.reasoning,
            "cited_papers": list(self.cited_papers),
            "fallback": self.fallback,
        }


@dataclass
class TranslatedPrior:
    family: str
    sites: dict[str, PriorSiteConfig]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "sites": {name: config.to_dict() for name, config in self.sites.items()},
            "warnings": list(self.warnings),
        }


def build_hill_prior(augmented: AugmentedPrior) -> TranslatedPrior:
    sites = {
        "bottom": _coerce_site(
            augmented.distributions,
            parameter_name="bottom",
            default=PriorSiteConfig(
                name="bottom",
                distribution="Normal",
                params={"loc": 0.0, "scale": 10.0},
                source="fallback",
                reasoning="Default weak prior for baseline response.",
                fallback=True,
            ),
            positive=False,
        ),
        "top": _coerce_site(
            augmented.distributions,
            parameter_name="top",
            default=PriorSiteConfig(
                name="top",
                distribution="Normal",
                params={"loc": 1.0, "scale": 10.0},
                source="fallback",
                reasoning="Default weak prior for upper response plateau.",
                fallback=True,
            ),
            positive=False,
        ),
        "ec50": _coerce_site(
            augmented.distributions,
            parameter_name="ec50",
            default=PriorSiteConfig(
                name="ec50",
                distribution="LogNormal",
                params={"loc": 0.0, "scale": 2.0},
                source="fallback",
                reasoning="Default weak prior for half-maximal concentration.",
                fallback=True,
            ),
            positive=True,
        ),
        "hill_n": _coerce_site(
            augmented.distributions,
            parameter_name="hill_n",
            default=PriorSiteConfig(
                name="hill_n",
                distribution="LogNormal",
                params={"loc": 0.0, "scale": 1.0},
                source="fallback",
                reasoning="Default weak prior for Hill slope.",
                fallback=True,
            ),
            positive=True,
        ),
        "sigma": _coerce_site(
            augmented.distributions,
            parameter_name="sigma",
            default=PriorSiteConfig(
                name="sigma",
                distribution="LogNormal",
                params={"loc": -2.0, "scale": 1.0},
                source="fallback",
                reasoning="Default weak prior for observation noise.",
                fallback=True,
            ),
            positive=True,
        ),
    }
    return TranslatedPrior(family="hill", sites=sites)


def build_multireceptor_prior(
    augmented: AugmentedPrior,
    *,
    receptor_names: Sequence[str],
    receptor_qpcr: Sequence[float],
) -> TranslatedPrior:
    sites: dict[str, PriorSiteConfig] = {
        "bottom": _coerce_site(
            augmented.distributions,
            parameter_name="bottom",
            default=PriorSiteConfig(
                name="bottom",
                distribution="Normal",
                params={"loc": 0.0, "scale": 10.0},
                source="fallback",
                reasoning="Default weak prior for baseline response.",
                fallback=True,
            ),
            positive=False,
        ),
        "top": _coerce_site(
            augmented.distributions,
            parameter_name="top",
            default=PriorSiteConfig(
                name="top",
                distribution="Normal",
                params={"loc": 1.0, "scale": 10.0},
                source="fallback",
                reasoning="Default weak prior for maximal response.",
                fallback=True,
            ),
            positive=False,
        ),
        "s50": _coerce_site(
            augmented.distributions,
            parameter_name="s50",
            default=PriorSiteConfig(
                name="s50",
                distribution="LogNormal",
                params={"loc": 0.0, "scale": 2.0},
                source="fallback",
                reasoning="Default weak prior for signaling half-max.",
                fallback=True,
            ),
            positive=True,
        ),
        "response_hill": _coerce_site(
            augmented.distributions,
            parameter_name="response_hill",
            default=PriorSiteConfig(
                name="response_hill",
                distribution="LogNormal",
                params={"loc": 0.0, "scale": 1.0},
                source="fallback",
                reasoning="Default weak prior for signaling Hill slope.",
                fallback=True,
            ),
            positive=True,
        ),
        "sigma_y": _coerce_site(
            augmented.distributions,
            parameter_name="sigma_y",
            default=PriorSiteConfig(
                name="sigma_y",
                distribution="LogNormal",
                params={"loc": -2.0, "scale": 1.0},
                source="fallback",
                reasoning="Default weak prior for observation noise.",
                fallback=True,
            ),
            positive=True,
        ),
    }

    for index, receptor_name in enumerate(receptor_names):
        sites[f"kd_{receptor_name}"] = _coerce_site(
            augmented.distributions,
            parameter_name="kd",
            default=PriorSiteConfig(
                name=f"kd_{receptor_name}",
                distribution="LogNormal",
                params={"loc": 0.0, "scale": 2.0},
                source="fallback",
                reasoning=f"Default weak binding prior for {receptor_name}.",
                fallback=True,
            ),
            positive=True,
            alias=f"kd_{receptor_name}",
        )
        sites[f"weight_{receptor_name}"] = _coerce_site(
            augmented.distributions,
            parameter_name="weight",
            default=PriorSiteConfig(
                name=f"weight_{receptor_name}",
                distribution="LogNormal",
                params={"loc": 0.0, "scale": 1.0},
                source="fallback",
                reasoning=f"Default weak signaling-weight prior for {receptor_name}.",
                fallback=True,
            ),
            positive=True,
            alias=f"weight_{receptor_name}",
        )
        qpcr_value = float(receptor_qpcr[index])
        center = float(min(max(qpcr_value, 1e-3), 5.0 - 1e-3))
        sites[f"abundance_{receptor_name}"] = PriorSiteConfig(
            name=f"abundance_{receptor_name}",
            distribution="TruncatedLogNormal",
            params={
                "loc": math.log(center),
                "scale": 1.0,
                "low": 0.0,
                "high": 5.0,
                "center": center,
                "qpcr_value": qpcr_value,
            },
            source="cell_line_qpcr",
            reasoning=(
                f"Cell-line-specific abundance prior centered on qPCR proxy for {receptor_name}."
            ),
            cited_papers=[],
            fallback=False,
        )

    return TranslatedPrior(family="multireceptor", sites=sites)


def build_multireceptor_hierarchical_prior(
    augmented: AugmentedPrior,
    *,
    receptor_names: Sequence[str],
) -> TranslatedPrior:
    receptor_names = tuple(receptor_names)
    receptor_count = len(receptor_names)

    sites: dict[str, PriorSiteConfig] = {
        "log_kd": _coerce_receptor_log_space_site(
            augmented.distributions,
            parameter_name="kd",
            receptor_names=receptor_names,
            default=PriorSiteConfig(
                name="log_kd",
                distribution="Normal",
                params=_vector_params(loc=0.0, scale=2.0, count=receptor_count),
                source="fallback",
                reasoning="Default weak log-affinity prior shared across receptors.",
                fallback=True,
            ),
        ),
        "log_weight": _coerce_log_space_site(
            augmented.distributions,
            parameter_name="weight",
            default=PriorSiteConfig(
                name="log_weight",
                distribution="Normal",
                params=_vector_params(loc=0.0, scale=1.0, count=receptor_count),
                source="fallback",
                reasoning="Default weak log signaling-weight prior shared across receptors.",
                fallback=True,
            ),
        ),
        "base_log_R": PriorSiteConfig(
            name="base_log_R",
            distribution="Normal",
            params=_vector_params(loc=0.0, scale=1.5, count=receptor_count),
            source="fallback",
            reasoning="Default weak parent-like receptor abundance prior.",
            fallback=True,
        ),
        "qpcr_intercept": PriorSiteConfig(
            name="qpcr_intercept",
            distribution="Normal",
            params=_vector_params(loc=0.0, scale=2.0, count=receptor_count),
            source="fallback",
            reasoning="Default weak qPCR intercept prior for each receptor channel.",
            fallback=True,
        ),
        "qpcr_slope": PriorSiteConfig(
            name="qpcr_slope",
            distribution="LogNormal",
            params=_vector_params(loc=0.0, scale=0.35, count=receptor_count),
            source="fallback",
            reasoning="Default positive qPCR slope prior for each receptor channel.",
            fallback=True,
        ),
        "sigma_q": PriorSiteConfig(
            name="sigma_q",
            distribution="LogNormal",
            params=_vector_params(loc=-1.0, scale=0.5, count=receptor_count),
            source="fallback",
            reasoning="Default qPCR measurement-noise prior for each receptor channel.",
            fallback=True,
        ),
        "sigma_R": PriorSiteConfig(
            name="sigma_R",
            distribution="LogNormal",
            params={"loc": -0.5, "scale": 0.5},
            source="fallback",
            reasoning="Default cross-cell-line receptor-abundance dispersion prior.",
            fallback=True,
        ),
        "log_s50": _coerce_log_space_site(
            augmented.distributions,
            parameter_name="s50",
            default=PriorSiteConfig(
                name="log_s50",
                distribution="Normal",
                params={"loc": 0.0, "scale": 2.0},
                source="fallback",
                reasoning="Default weak log half-signal prior.",
                fallback=True,
            ),
        ),
        "response_hill": _coerce_site(
            augmented.distributions,
            parameter_name="response_hill",
            default=PriorSiteConfig(
                name="response_hill",
                distribution="LogNormal",
                params={"loc": 0.0, "scale": 0.5},
                source="fallback",
                reasoning="Default weak prior for response nonlinearity.",
                fallback=True,
            ),
            positive=True,
        ),
        "bottom": _coerce_site(
            augmented.distributions,
            parameter_name="bottom",
            default=PriorSiteConfig(
                name="bottom",
                distribution="LogNormal",
                params={"loc": -2.0, "scale": 1.0},
                source="fallback",
                reasoning="Default weak positive baseline-response prior for each cell line.",
                fallback=True,
            ),
            positive=True,
        ),
        "top": _coerce_site(
            augmented.distributions,
            parameter_name="top",
            default=PriorSiteConfig(
                name="top",
                distribution="LogNormal",
                params={"loc": -0.25, "scale": 1.0},
                source="fallback",
                reasoning="Default weak positive maximal-response prior for each cell line.",
                fallback=True,
            ),
            positive=True,
        ),
        "sigma_y": _coerce_site(
            augmented.distributions,
            parameter_name="sigma_y",
            default=PriorSiteConfig(
                name="sigma_y",
                distribution="LogNormal",
                params={"loc": -1.5, "scale": 0.5},
                source="fallback",
                reasoning="Default weak response-noise prior for each cell line.",
                fallback=True,
            ),
            positive=True,
        ),
    }

    return TranslatedPrior(family="multireceptor_hierarchical", sites=sites)


def make_distribution(config: PriorSiteConfig) -> pyro_dist.TorchDistribution:
    params = config.params
    name = config.distribution.lower()
    if name == "normal":
        return pyro_dist.Normal(
            torch.tensor(params.get("loc", params.get("mu", 0.0)), dtype=torch.float32),
            torch.tensor(params.get("scale", params.get("sigma", 1.0)), dtype=torch.float32),
        )
    if name == "lognormal":
        return pyro_dist.LogNormal(
            torch.tensor(params.get("loc", params.get("mu", 0.0)), dtype=torch.float32),
            torch.tensor(params.get("scale", params.get("sigma", 1.0)), dtype=torch.float32),
        )
    if name == "gamma":
        alpha = params.get("alpha", params.get("concentration", 2.0))
        beta = params.get("beta", params.get("rate", 1.0))
        return pyro_dist.Gamma(
            torch.tensor(alpha, dtype=torch.float32),
            torch.tensor(beta, dtype=torch.float32),
        )
    if name == "uniform":
        return pyro_dist.Uniform(
            torch.tensor(params.get("low", 0.0), dtype=torch.float32),
            torch.tensor(params.get("high", 1.0), dtype=torch.float32),
        )
    if name == "beta":
        return pyro_dist.Beta(
            torch.tensor(params.get("a", params.get("alpha", 2.0)), dtype=torch.float32),
            torch.tensor(params.get("b", params.get("beta", 2.0)), dtype=torch.float32),
        )
    if name == "truncatedlognormal":
        return TruncatedLogNormal(
            loc=torch.tensor(params["loc"], dtype=torch.float32),
            scale=torch.tensor(params["scale"], dtype=torch.float32),
            low=torch.tensor(params["low"], dtype=torch.float32),
            high=torch.tensor(params["high"], dtype=torch.float32),
        )
    raise ValueError(f"Unsupported prior distribution {config.distribution!r}.")


def _coerce_site(
    distributions: Mapping[str, DistributionSpec],
    *,
    parameter_name: str,
    default: PriorSiteConfig,
    positive: bool,
    alias: str | None = None,
) -> PriorSiteConfig:
    spec = _match_distribution_spec(distributions, parameter_name, alias=alias)
    if spec is None:
        return default

    translated = _translate_distribution_spec(
        spec,
        default=default,
        positive=positive,
    )
    return PriorSiteConfig(
        name=default.name,
        distribution=translated.distribution,
        params=translated.params,
        source=translated.source,
        reasoning=translated.reasoning,
        cited_papers=translated.cited_papers,
        fallback=translated.fallback,
    )


def _match_distribution_spec(
    distributions: Mapping[str, DistributionSpec],
    parameter_name: str,
    *,
    alias: str | None = None,
) -> DistributionSpec | None:
    lowered = {key.lower(): value for key, value in distributions.items()}
    candidates = [name for name in (alias, parameter_name) if name]
    for candidate in candidates:
        exact = lowered.get(candidate.lower())
        if exact is not None:
            return exact
    return None


def _translate_distribution_spec(
    spec: DistributionSpec,
    *,
    default: PriorSiteConfig,
    positive: bool,
) -> PriorSiteConfig:
    raw_name = (spec.name or "").strip()
    lowered = raw_name.lower()
    params = dict(spec.params)

    if lowered in {"normal", "lognormal", "gamma", "uniform", "beta"}:
        if positive and lowered == "normal":
            return default
        return PriorSiteConfig(
            name=default.name,
            distribution=raw_name or default.distribution,
            params=_normalize_params(lowered, params),
            source=spec.source,
            reasoning=spec.reasoning,
            cited_papers=list(spec.cited_papers),
            fallback=spec.fallback,
        )
    return default


def _coerce_log_space_site(
    distributions: Mapping[str, DistributionSpec],
    *,
    parameter_name: str,
    default: PriorSiteConfig,
    alias: str | None = None,
) -> PriorSiteConfig:
    spec = _match_distribution_spec(distributions, parameter_name, alias=alias)
    if spec is None:
        return default

    lowered = (spec.name or "").strip().lower()
    if lowered not in {"normal", "lognormal"}:
        return default

    normalized = _normalize_params(lowered, spec.params)
    return PriorSiteConfig(
        name=default.name,
        distribution="Normal",
        params={
            "loc": _broadcast_like_default(normalized.get("loc", 0.0), default.params.get("loc", 0.0)),
            "scale": _broadcast_like_default(
                normalized.get("scale", 1.0),
                default.params.get("scale", 1.0),
            ),
        },
        source=spec.source,
        reasoning=spec.reasoning,
        cited_papers=list(spec.cited_papers),
        fallback=spec.fallback,
    )


def _coerce_receptor_log_space_site(
    distributions: Mapping[str, DistributionSpec],
    *,
    parameter_name: str,
    receptor_names: Sequence[str],
    default: PriorSiteConfig,
) -> PriorSiteConfig:
    base = _coerce_log_space_site(
        distributions,
        parameter_name=parameter_name,
        default=default,
    )
    receptor_specific: list[tuple[str, DistributionSpec, dict[str, Any]]] = []
    for receptor_name in receptor_names:
        spec = _match_distribution_spec(
            distributions,
            f"{parameter_name}_{receptor_name}",
        )
        if spec is None:
            continue
        lowered = (spec.name or "").strip().lower()
        if lowered not in {"normal", "lognormal"}:
            continue
        receptor_specific.append((receptor_name, spec, _normalize_params(lowered, spec.params)))

    if not receptor_specific:
        return base

    specific_by_receptor = {
        receptor_name: (spec, normalized)
        for receptor_name, spec, normalized in receptor_specific
    }
    locs: list[float] = []
    scales: list[float] = []
    sources: list[str] = []
    reasons: list[str] = []
    cited: list[str] = []
    fallback = False
    for index, receptor_name in enumerate(receptor_names):
        receptor_spec = specific_by_receptor.get(receptor_name)
        if receptor_spec is None:
            locs.append(_indexed_param(base.params.get("loc", 0.0), index))
            scales.append(_indexed_param(base.params.get("scale", 1.0), index))
            sources.append(base.source)
            fallback = fallback or base.fallback
            continue

        spec, normalized = receptor_spec
        locs.append(_indexed_param(normalized.get("loc", 0.0), 0))
        scales.append(_indexed_param(normalized.get("scale", 1.0), 0))
        sources.append(spec.source)
        fallback = fallback or spec.fallback
        if spec.reasoning:
            reasons.append(f"{receptor_name}: {spec.reasoning}")
        cited.extend(spec.cited_papers)

    source = sources[0] if len(set(sources)) == 1 else "mixed_receptor_specific"
    return PriorSiteConfig(
        name=default.name,
        distribution="Normal",
        params={"loc": locs, "scale": scales},
        source=source,
        reasoning="; ".join(reasons) or base.reasoning,
        cited_papers=sorted(set(cited)),
        fallback=fallback,
    )


def _normalize_params(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    if name == "normal":
        return {
            "loc": _float_or_float_list(params.get("loc", params.get("mu", 0.0))),
            "scale": _float_or_float_list(params.get("scale", params.get("sigma", 1.0))),
        }
    if name == "lognormal":
        return {
            "loc": _float_or_float_list(params.get("loc", params.get("mu", 0.0))),
            "scale": _float_or_float_list(params.get("scale", params.get("sigma", 1.0))),
        }
    if name == "gamma":
        return {
            "alpha": float(params.get("alpha", params.get("concentration", 2.0))),
            "beta": float(params.get("beta", params.get("rate", 1.0))),
        }
    if name == "uniform":
        return {
            "low": float(params.get("low", 0.0)),
            "high": float(params.get("high", 1.0)),
        }
    if name == "beta":
        return {
            "a": float(params.get("a", params.get("alpha", 2.0))),
            "b": float(params.get("b", params.get("beta", 2.0))),
        }
    return {
        key: [float(item) for item in value]
        if isinstance(value, (list, tuple))
        else float(value)
        for key, value in params.items()
    }


def _vector_params(*, loc: float, scale: float, count: int) -> dict[str, list[float]]:
    return {
        "loc": [float(loc)] * int(count),
        "scale": [float(scale)] * int(count),
    }


def _indexed_param(value: Any, index: int) -> float:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Prior parameter lists must not be empty.")
        return float(value[min(index, len(value) - 1)])
    return float(value)


def _float_or_float_list(value: Any) -> float | list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return float(value)


def _broadcast_like_default(value: Any, reference: Any) -> Any:
    if isinstance(reference, list):
        if isinstance(value, (list, tuple)):
            if len(value) == len(reference):
                return [float(item) for item in value]
            if len(value) == 1:
                return [float(value[0])] * len(reference)
        return [float(value)] * len(reference)
    return float(value)


__all__ = [
    "PriorSiteConfig",
    "TranslatedPrior",
    "TruncatedLogNormal",
    "build_hill_prior",
    "build_multireceptor_hierarchical_prior",
    "build_multireceptor_prior",
    "make_distribution",
]
