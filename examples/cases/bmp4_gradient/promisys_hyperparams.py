"""Config surface for BMP4 promisys autoresearch runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ACTIVATIONS = {"gelu", "relu", "silu", "tanh"}
_TOP_LEVEL_KEYS = {"posterior_net", "flow", "objective", "mcmc"}


@dataclass(frozen=True)
class PosteriorNetHyperparams:
    hidden_dim: int | None = None
    layers: int | None = None
    activation: str | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    steps: int | None = None
    simulations: int | None = None
    posterior_samples: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PosteriorNetHyperparams":
        data = _expect_mapping("posterior_net", data)
        _reject_unknown("posterior_net", data, set(cls.__dataclass_fields__))
        return cls(
            hidden_dim=_optional_positive_int(data, "hidden_dim"),
            layers=_optional_positive_int(data, "layers"),
            activation=_optional_activation(data, "activation"),
            batch_size=_optional_positive_int(data, "batch_size"),
            learning_rate=_optional_positive_float(data, "learning_rate"),
            steps=_optional_nonnegative_int(data, "steps"),
            simulations=_optional_positive_int(data, "simulations"),
            posterior_samples=_optional_positive_int(data, "posterior_samples"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "hidden_dim": self.hidden_dim,
                "layers": self.layers,
                "activation": self.activation,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "steps": self.steps,
                "simulations": self.simulations,
                "posterior_samples": self.posterior_samples,
            }
        )


@dataclass(frozen=True)
class FlowHyperparams:
    num_layers: int | None = None
    hidden_sizes: tuple[int, ...] | None = None
    num_bins: int | None = None
    activation: str | None = None
    use_resnet: bool | None = None
    dropout_rate: float | None = None
    standardize_theta: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FlowHyperparams":
        data = _expect_mapping("flow", data)
        _reject_unknown("flow", data, set(cls.__dataclass_fields__))
        return cls(
            num_layers=_optional_positive_int(data, "num_layers"),
            hidden_sizes=_optional_positive_int_tuple(data, "hidden_sizes"),
            num_bins=_optional_min_int(data, "num_bins", minimum=2),
            activation=_optional_activation(data, "activation"),
            use_resnet=_optional_bool(data, "use_resnet"),
            dropout_rate=_optional_probability(data, "dropout_rate"),
            standardize_theta=_optional_bool(data, "standardize_theta"),
        )

    def overlay(self, base_config: dict[str, Any]) -> dict[str, Any]:
        config = dict(base_config)
        for key, value in self.to_dict().items():
            if key == "hidden_sizes":
                config[key] = tuple(int(item) for item in value)
            else:
                config[key] = value
        return config

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "num_layers": self.num_layers,
                "hidden_sizes": list(self.hidden_sizes) if self.hidden_sizes is not None else None,
                "num_bins": self.num_bins,
                "activation": self.activation,
                "use_resnet": self.use_resnet,
                "dropout_rate": self.dropout_rate,
                "standardize_theta": self.standardize_theta,
            }
        )


@dataclass(frozen=True)
class ObjectiveHyperparams:
    fit_steps: int | None = None
    flow_learning_rate: float | None = None
    design_learning_rate: float | None = None
    eig_outer_samples: int | None = None
    eig_inner_samples: int | None = None
    infonce_lambda: float | None = None
    design_dist_init_std: float | None = None
    design_temperature_scale: float | None = None
    selector_temperature_final: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ObjectiveHyperparams":
        data = _expect_mapping("objective", data)
        _reject_unknown("objective", data, set(cls.__dataclass_fields__))
        return cls(
            fit_steps=_optional_nonnegative_int(data, "fit_steps"),
            flow_learning_rate=_optional_positive_float(data, "flow_learning_rate"),
            design_learning_rate=_optional_positive_float(data, "design_learning_rate"),
            eig_outer_samples=_optional_positive_int(data, "eig_outer_samples"),
            eig_inner_samples=_optional_nonnegative_int(data, "eig_inner_samples"),
            infonce_lambda=_optional_float(data, "infonce_lambda"),
            design_dist_init_std=_optional_positive_float(data, "design_dist_init_std"),
            design_temperature_scale=_optional_positive_float(data, "design_temperature_scale"),
            selector_temperature_final=_optional_positive_float(data, "selector_temperature_final"),
            early_stopping_patience=_optional_nonnegative_int(data, "early_stopping_patience"),
            early_stopping_min_delta=_optional_nonnegative_float(data, "early_stopping_min_delta"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "fit_steps": self.fit_steps,
                "flow_learning_rate": self.flow_learning_rate,
                "design_learning_rate": self.design_learning_rate,
                "eig_outer_samples": self.eig_outer_samples,
                "eig_inner_samples": self.eig_inner_samples,
                "infonce_lambda": self.infonce_lambda,
                "design_dist_init_std": self.design_dist_init_std,
                "design_temperature_scale": self.design_temperature_scale,
                "selector_temperature_final": self.selector_temperature_final,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
            }
        )


@dataclass(frozen=True)
class MCMCHyperparams:
    warmup: int | None = None
    samples: int | None = None
    proposal_scale: float | None = None
    prior_std_floor: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MCMCHyperparams":
        data = _expect_mapping("mcmc", data)
        _reject_unknown("mcmc", data, set(cls.__dataclass_fields__))
        return cls(
            warmup=_optional_nonnegative_int(data, "warmup"),
            samples=_optional_positive_int(data, "samples"),
            proposal_scale=_optional_positive_float(data, "proposal_scale"),
            prior_std_floor=_optional_positive_float(data, "prior_std_floor"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "warmup": self.warmup,
                "samples": self.samples,
                "proposal_scale": self.proposal_scale,
                "prior_std_floor": self.prior_std_floor,
            }
        )


@dataclass(frozen=True)
class PromisysHyperparams:
    posterior_net: PosteriorNetHyperparams = PosteriorNetHyperparams()
    flow: FlowHyperparams = FlowHyperparams()
    objective: ObjectiveHyperparams = ObjectiveHyperparams()
    mcmc: MCMCHyperparams = MCMCHyperparams()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PromisysHyperparams":
        data = _expect_mapping("promisys_hyperparams", data)
        _reject_unknown("promisys_hyperparams", data, _TOP_LEVEL_KEYS)
        return cls(
            posterior_net=PosteriorNetHyperparams.from_dict(data.get("posterior_net")),
            flow=FlowHyperparams.from_dict(data.get("flow")),
            objective=ObjectiveHyperparams.from_dict(data.get("objective")),
            mcmc=MCMCHyperparams.from_dict(data.get("mcmc")),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PromisysHyperparams":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    def flow_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        return self.flow.overlay(base_config)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(
            {
                "posterior_net": self.posterior_net.to_dict(),
                "flow": self.flow.to_dict(),
                "objective": self.objective.to_dict(),
                "mcmc": self.mcmc.to_dict(),
            }
        )


def coerce_promisys_hyperparams(value: Any) -> PromisysHyperparams | None:
    if value is None:
        return None
    if isinstance(value, PromisysHyperparams):
        return value
    if isinstance(value, (str, Path)):
        return PromisysHyperparams.from_json(value)
    if isinstance(value, dict):
        return PromisysHyperparams.from_dict(value)
    raise TypeError(
        "promisys_hyperparams must be a PromisysHyperparams, dict, path, or None; "
        f"got {type(value).__name__}."
    )


def effective_promisys_hyperparams(
    *,
    base_flow_config: dict[str, Any],
    hyperparams: PromisysHyperparams | None,
    snpe_steps: int,
    snpe_simulations: int,
    snpe_learning_rate: float,
    posterior_sample_count: int,
    likelihood_steps: int,
    likelihood_learning_rate: float,
    eig_outer_samples: int,
    eig_inner_samples: int,
    eig_learning_rate: float,
    infonce_lambda: float,
    design_dist_init_std: float,
    design_temperature_scale: float,
    selector_temperature_final: float,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    mcmc_warmup: int,
    mcmc_samples: int,
    mcmc_proposal_scale: float,
    mcmc_prior_std_floor: float,
    posterior_hidden_dim: int,
    posterior_layers: int,
    posterior_activation: str,
    posterior_batch_size: int,
) -> dict[str, Any]:
    flow_config = dict(base_flow_config if hyperparams is None else hyperparams.flow_config(base_flow_config))
    flow_config["hidden_sizes"] = list(flow_config.get("hidden_sizes", ()))
    return {
        "posterior_net": {
            "hidden_dim": int(posterior_hidden_dim),
            "layers": int(posterior_layers),
            "activation": str(posterior_activation),
            "batch_size": int(posterior_batch_size),
            "learning_rate": float(snpe_learning_rate),
            "steps": int(snpe_steps),
            "simulations": int(snpe_simulations),
            "posterior_samples": int(posterior_sample_count),
        },
        "flow": flow_config,
        "objective": {
            "fit_steps": int(likelihood_steps),
            "flow_learning_rate": float(likelihood_learning_rate),
            "design_learning_rate": float(eig_learning_rate),
            "eig_outer_samples": int(eig_outer_samples),
            "eig_inner_samples": int(eig_inner_samples),
            "infonce_lambda": float(infonce_lambda),
            "design_dist_init_std": float(design_dist_init_std),
            "design_temperature_scale": float(design_temperature_scale),
            "selector_temperature_final": float(selector_temperature_final),
            "early_stopping_patience": (
                None if early_stopping_patience is None else int(early_stopping_patience)
            ),
            "early_stopping_min_delta": float(early_stopping_min_delta),
        },
        "mcmc": {
            "warmup": int(mcmc_warmup),
            "samples": int(mcmc_samples),
            "proposal_scale": float(mcmc_proposal_scale),
            "prior_std_floor": float(mcmc_prior_std_floor),
        },
    }


def _expect_mapping(section: str, data: dict[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"`{section}` must be a JSON object.")
    return data


def _reject_unknown(section: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown `{section}` hyperparameter keys: {unknown}")


def _optional_bool(data: dict[str, Any], key: str) -> bool | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(f"`{key}` must be a boolean.")
    return value


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    if key not in data:
        return None
    value = float(data[key])
    if not math.isfinite(value):
        raise ValueError(f"`{key}` must be finite.")
    return value


def _optional_positive_float(data: dict[str, Any], key: str) -> float | None:
    value = _optional_float(data, key)
    if value is not None and value <= 0.0:
        raise ValueError(f"`{key}` must be positive.")
    return value


def _optional_nonnegative_float(data: dict[str, Any], key: str) -> float | None:
    value = _optional_float(data, key)
    if value is not None and value < 0.0:
        raise ValueError(f"`{key}` must be nonnegative.")
    return value


def _optional_probability(data: dict[str, Any], key: str) -> float | None:
    value = _optional_float(data, key)
    if value is not None and not 0.0 <= value < 1.0:
        raise ValueError(f"`{key}` must be in [0, 1).")
    return value


def _optional_positive_int(data: dict[str, Any], key: str) -> int | None:
    value = _optional_int(data, key)
    if value is not None and value <= 0:
        raise ValueError(f"`{key}` must be positive.")
    return value


def _optional_min_int(data: dict[str, Any], key: str, *, minimum: int) -> int | None:
    value = _optional_int(data, key)
    if value is not None and value < minimum:
        raise ValueError(f"`{key}` must be >= {minimum}.")
    return value


def _optional_nonnegative_int(data: dict[str, Any], key: str) -> int | None:
    value = _optional_int(data, key)
    if value is not None and value < 0:
        raise ValueError(f"`{key}` must be non-negative.")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, bool):
        raise TypeError(f"`{key}` must be an integer.")
    int_value = int(value)
    if int_value != value:
        raise ValueError(f"`{key}` must be an integer.")
    return int_value


def _optional_positive_int_tuple(data: dict[str, Any], key: str) -> tuple[int, ...] | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, list):
        raise TypeError(f"`{key}` must be a JSON array.")
    output = tuple(int(item) for item in value)
    if not output or any(item <= 0 for item in output):
        raise ValueError(f"`{key}` must contain one or more positive integers.")
    return output


def _optional_activation(data: dict[str, Any], key: str) -> str | None:
    if key not in data:
        return None
    value = str(data[key]).lower()
    if value not in _ACTIVATIONS:
        raise ValueError(f"`{key}` must be one of {sorted(_ACTIVATIONS)}.")
    return value


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _drop_empty(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value}
