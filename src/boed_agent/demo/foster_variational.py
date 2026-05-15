"""Paper-aligned BOED demos based on Foster et al. (NeurIPS 2019)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from boed_agent.models import DesignVariable, ExperimentSpec
from boed_agent.utils.imports import register_callable


PAPER_TITLE = "Variational Bayesian Optimal Experimental Design"
PAPER_URL = "https://papers.nips.cc/paper_files/paper/2019/file/d55cbf210f175f4a37916eafe6c04f0d-Paper.pdf"
REPRO_URL = "https://github.com/ae-foster/pyro/blob/vboed-reproduce/NEURIPS.md"


@dataclass(frozen=True)
class FosterExperimentDefinition:
    key: str
    label: str
    description: str
    default_estimator: str
    estimator_options: tuple[str, ...]
    build_spec: Any


def list_foster_experiment_definitions() -> list[FosterExperimentDefinition]:
    return [
        FosterExperimentDefinition(
            key="ab_test_linear",
            label="A/B Test (Linear)",
            description="Gaussian linear A/B testing benchmark from the paper's EIG-accuracy study.",
            default_estimator="posterior_eig",
            estimator_options=("posterior_eig", "marginal_eig", "vnmc_eig", "vi_eig"),
            build_spec=build_foster_ab_test_linear_spec,
        ),
        FosterExperimentDefinition(
            key="revealed_preference",
            label="Revealed Preference",
            description="Sigmoid preference benchmark from the paper's explicit-likelihood experiments.",
            default_estimator="posterior_eig",
            estimator_options=("posterior_eig", "marginal_eig", "vnmc_eig", "vi_eig"),
            build_spec=build_foster_revealed_preference_spec,
        ),
    ]


def get_foster_experiment_definition(key: str) -> FosterExperimentDefinition:
    for definition in list_foster_experiment_definitions():
        if definition.key == key:
            return definition
    available = ", ".join(item.key for item in list_foster_experiment_definitions())
    raise KeyError(f"Unknown Foster experiment '{key}'. Available experiments: {available}.")


def build_foster_experiment_spec(key: str, estimator: str | None = None) -> ExperimentSpec:
    definition = get_foster_experiment_definition(key)
    chosen_estimator = estimator or definition.default_estimator
    return definition.build_spec(chosen_estimator)


def build_foster_ab_test_linear_spec(estimator: str = "posterior_eig") -> ExperimentSpec:
    allowed = {"posterior_eig", "marginal_eig", "vnmc_eig", "vi_eig"}
    if estimator not in allowed:
        raise ValueError(f"Unsupported estimator '{estimator}' for the Foster A/B test demo.")

    spec = ExperimentSpec.from_dict(
        {
            "problem_summary": (
                "Foster et al. A/B testing benchmark with a Gaussian linear model and a discrete "
                "candidate grid over the A-group allocation."
            ),
            "backend": "pyro",
            "design_variables": [
                {
                    "name": "group_a_size",
                    "lower": 0.0,
                    "upper": 10.0,
                    "initial": 5.0,
                    "description": "Number of participants assigned to group A out of 10.",
                }
            ],
            "observation_labels": ["y"],
            "target_latent_labels": ["w"],
            "compute_budget": {
                "num_outer_samples": 10,
                "guide_training_steps": 2500,
                "num_optimization_steps": 11,
            },
            "objective": {
                "name": "expected_information_gain",
                "estimator": estimator,
                "mode": "variational",
            },
            "artifacts": {
                "output_dir": "artifacts/foster_variational",
            },
            "backend_options": {
                "guide_learning_rate": 0.05,
                "optimization_strategy": "candidate_grid",
                "candidate_designs": [[float(n)] for n in range(11)],
            },
            "metadata": {
                "paper_title": PAPER_TITLE,
                "paper_url": PAPER_URL,
                "reproduction_url": REPRO_URL,
                "paper_experiment": "ab_test_linear",
            },
            "model_ref": "boed_agent.demo.foster_variational:foster_ab_test_linear_model",
            "guide_ref": _foster_ab_test_guide_ref(estimator),
            "optim_ref": "boed_agent.demo.foster_variational:make_foster_paper_adam",
        }
    )
    if estimator == "vi_eig":
        spec.loss_ref = "boed_agent.demo.foster_variational:make_foster_trace_elbo_loss"
        spec.compute_budget.num_outer_samples = 1
        spec.compute_budget.guide_training_steps = 5000
        spec.backend_options["final_num_samples"] = None
    elif estimator == "marginal_eig":
        spec.compute_budget.guide_training_steps = 1800
        spec.backend_options["final_num_samples"] = 500
    elif estimator == "vnmc_eig":
        spec.compute_budget.num_inner_samples = 1
        spec.compute_budget.guide_training_steps = 1400
        spec.backend_options["final_num_samples"] = [500, 1]
    else:
        spec.backend_options["final_num_samples"] = 500
    return spec


def build_foster_revealed_preference_spec(estimator: str = "posterior_eig") -> ExperimentSpec:
    allowed = {"posterior_eig", "marginal_eig", "vnmc_eig", "vi_eig"}
    if estimator not in allowed:
        raise ValueError(
            f"Unsupported estimator '{estimator}' for the Foster revealed-preference demo."
        )

    candidate_designs = [[float(value)] for value in _linspace(-80.0, 80.0, 20)]
    spec = ExperimentSpec.from_dict(
        {
            "problem_summary": (
                "Foster et al. revealed-preference benchmark with a scalar location latent and a "
                "sigmoid-transformed Gaussian observation model."
            ),
            "backend": "pyro",
            "design_variables": [
                {
                    "name": "offer_location",
                    "lower": -80.0,
                    "upper": 80.0,
                    "initial": 0.0,
                    "description": "Stimulus or offer location shown to the agent.",
                }
            ],
            "observation_labels": ["y"],
            "target_latent_labels": ["loc"],
            "compute_budget": {
                "num_outer_samples": 10,
                "guide_training_steps": 450,
                "num_optimization_steps": 20,
            },
            "objective": {
                "name": "expected_information_gain",
                "estimator": estimator,
                "mode": "variational",
            },
            "artifacts": {
                "output_dir": "artifacts/foster_variational",
            },
            "backend_options": {
                "guide_learning_rate": 0.05,
                "optimization_strategy": "candidate_grid",
                "candidate_designs": candidate_designs,
            },
            "metadata": {
                "paper_title": PAPER_TITLE,
                "paper_url": PAPER_URL,
                "reproduction_url": REPRO_URL,
                "paper_experiment": "revealed_preference",
            },
            "model_ref": "boed_agent.demo.foster_variational:foster_revealed_preference_model",
            "guide_ref": _foster_preference_guide_ref(estimator),
            "optim_ref": "boed_agent.demo.foster_variational:make_foster_paper_adam",
        }
    )
    if estimator == "vi_eig":
        spec.loss_ref = "boed_agent.demo.foster_variational:make_foster_trace_elbo_loss"
        spec.compute_budget.num_outer_samples = 1
        spec.compute_budget.guide_training_steps = 5000
    elif estimator == "marginal_eig":
        spec.compute_budget.guide_training_steps = 1000
        spec.backend_options["final_num_samples"] = 500
    elif estimator == "vnmc_eig":
        spec.compute_budget.num_inner_samples = 1
        spec.compute_budget.guide_training_steps = 200
        spec.backend_options["final_num_samples"] = [100, 50]
    else:
        spec.backend_options["final_num_samples"] = 500
    return spec


def foster_ab_test_linear_model(design: Any) -> Any:
    from pyro.contrib.oed.glmm import group_assignment_matrix, known_covariance_linear_model

    matrix_design = _ab_test_design_matrix(design)
    model = known_covariance_linear_model(
        coef_means=0.0,
        coef_sds=_tensor([10.0, 1.0 / 0.55]),
        observation_sd=_tensor(1.0),
        coef_labels="w",
        observation_label="y",
    )
    return model(matrix_design)


foster_ab_test_linear_model.observation_label = "y"
foster_ab_test_linear_model.w_sizes = {"w": 2}


def foster_revealed_preference_model(design: Any) -> Any:
    import pyro
    import pyro.distributions as dist
    import torch
    from torch.distributions.transforms import SigmoidTransform

    design_tensor = _scalar_design(design)
    batch_shape = design_tensor.shape[:-1]
    loc = pyro.sample(
        "loc",
        dist.Normal(
            -20.0 * torch.ones(batch_shape, dtype=design_tensor.dtype, device=design_tensor.device),
            20.0 * torch.ones(batch_shape, dtype=design_tensor.dtype, device=design_tensor.device),
        ),
    )
    latent_mean = design_tensor.squeeze(-1) - loc
    base_dist = dist.Normal(latent_mean, 0.25 * torch.ones_like(latent_mean))
    response_dist = dist.TransformedDistribution(base_dist, [SigmoidTransform()])
    return pyro.sample("y", response_dist)


foster_revealed_preference_model.observation_label = "y"
foster_revealed_preference_model.w_sizes = {"loc": 1}


def foster_ab_test_linear_posterior_guide(
    y_dict: dict[str, Any],
    design: Any,
    observation_labels: list[str],
    target_labels: list[str],
) -> None:
    import pyro
    import pyro.distributions as dist
    import torch
    from torch.distributions import constraints

    _ = design, observation_labels, target_labels
    y = y_dict["y"]
    if y.ndim == 1:
        y = y.unsqueeze(0)
    regressor = pyro.param(
        "foster_ab_post_regressor",
        -3.0 * torch.ones(2, y.shape[-1], dtype=y.dtype, device=y.device),
    )
    bias = pyro.param(
        "foster_ab_post_bias",
        torch.zeros(2, dtype=y.dtype, device=y.device),
    )
    scale_tril = pyro.param(
        "foster_ab_post_scale_tril",
        torch.tensor([[10.0, 0.0], [0.0, 1.0 / 0.55]], dtype=y.dtype, device=y.device),
        constraint=constraints.lower_cholesky,
    )
    mu = torch.einsum("...n,pn->...p", y, torch.nn.functional.softplus(regressor)) + bias
    pyro.sample(
        "w",
        dist.MultivariateNormal(mu, scale_tril=scale_tril.expand(mu.shape[:-1] + (2, 2))),
    )


def foster_ab_test_linear_marginal_guide(
    design: Any,
    observation_labels: list[str],
    target_labels: list[str],
) -> None:
    import pyro
    import pyro.distributions as dist
    import torch
    from torch.distributions import constraints

    _ = design, target_labels
    mean = pyro.param("foster_ab_marginal_mean", torch.zeros(10))
    scale = pyro.param(
        "foster_ab_marginal_scale",
        3.0 * torch.ones(10),
        constraint=constraints.positive,
    )
    pyro.sample(
        observation_labels[0],
        dist.Normal(mean, scale).to_event(1),
    )


def foster_ab_test_linear_vi_guide(design: Any) -> None:
    import pyro
    import pyro.distributions as dist
    import torch
    from torch.distributions import constraints

    _ = design
    mean = pyro.param("foster_ab_vi_mean", torch.zeros(2))
    scale_tril = pyro.param(
        "foster_ab_vi_scale_tril",
        torch.tensor([[10.0, 0.0], [0.0, 1.0 / 0.55]]),
        constraint=constraints.lower_cholesky,
    )
    pyro.sample("w", dist.MultivariateNormal(mean, scale_tril=scale_tril))


def foster_revealed_preference_posterior_guide(
    y_dict: dict[str, Any],
    design: Any,
    observation_labels: list[str],
    target_labels: list[str],
) -> None:
    import pyro
    import pyro.distributions as dist
    import torch
    from torch.distributions import constraints

    _ = observation_labels, target_labels
    y = y_dict["y"]
    design_tensor = _scalar_design(design).squeeze(-1)
    y_clamped = y.clamp(1e-6, 1.0 - 1e-6)
    logit_y = torch.log(y_clamped) - torch.log1p(-y_clamped)
    bias = pyro.param("foster_pref_post_bias", torch.tensor(-20.0))
    coef_design = pyro.param("foster_pref_post_coef_design", torch.tensor(1.0))
    coef_logit = pyro.param("foster_pref_post_coef_logit", torch.tensor(-1.0))
    scale = pyro.param(
        "foster_pref_post_scale",
        torch.tensor(20.0),
        constraint=constraints.positive,
    )
    loc = bias + coef_design * design_tensor + coef_logit * logit_y
    pyro.sample("loc", dist.Normal(loc, scale))


def foster_revealed_preference_marginal_guide(
    design: Any,
    observation_labels: list[str],
    target_labels: list[str],
) -> None:
    import pyro
    import pyro.distributions as dist
    import torch
    from torch.distributions import constraints
    from torch.distributions.transforms import SigmoidTransform

    _ = target_labels
    design_tensor = _scalar_design(design).squeeze(-1)
    bias = pyro.param("foster_pref_marginal_bias", torch.tensor(0.0))
    coef_design = pyro.param("foster_pref_marginal_coef_design", torch.tensor(0.0))
    scale = pyro.param(
        "foster_pref_marginal_scale",
        torch.tensor(20.0),
        constraint=constraints.positive,
    )
    base_dist = dist.Normal(bias + coef_design * design_tensor, scale)
    pyro.sample(
        observation_labels[0],
        dist.TransformedDistribution(base_dist, [SigmoidTransform()]),
    )


def foster_revealed_preference_vi_guide(design: Any) -> None:
    import pyro
    import pyro.distributions as dist
    from torch.distributions import constraints
    import torch

    _ = design
    mean = pyro.param("foster_pref_vi_mean", torch.tensor(-20.0))
    scale = pyro.param(
        "foster_pref_vi_scale",
        torch.tensor(20.0),
        constraint=constraints.positive,
    )
    pyro.sample("loc", dist.Normal(mean, scale))


def make_foster_paper_adam(spec: Any | None = None) -> Any:
    import pyro

    learning_rate = 0.05
    if spec is not None:
        learning_rate = spec.backend_options.get("guide_learning_rate", learning_rate)
    return pyro.optim.Adam({"lr": learning_rate})


def make_foster_trace_elbo_loss(spec: Any | None = None) -> Any:
    import pyro

    _ = spec
    return pyro.infer.Trace_ELBO().differentiable_loss


def _ab_group_design(count: int) -> Any:
    from pyro.contrib.oed.glmm import group_assignment_matrix

    n_a = max(0, min(10, int(round(count))))
    return group_assignment_matrix(_tensor([float(n_a), float(10 - n_a)]))


def _ab_test_design_matrix(design: Any) -> Any:
    scalar_design = _scalar_design(design).squeeze(-1)
    flat = scalar_design.reshape(-1)
    matrices = [_ab_group_design(int(value.item())) for value in flat]
    output = _stack_tensors(matrices)
    return output.reshape(scalar_design.shape + (10, 2))


def _scalar_design(design: Any) -> Any:
    import torch

    tensor = design if isinstance(design, torch.Tensor) else torch.as_tensor(design, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.shape[-1] != 1:
        tensor = tensor.reshape(tensor.shape[:-1] + (1,))
    return tensor


def _stack_tensors(items: list[Any]) -> Any:
    import torch

    return torch.stack(items, dim=0)


def _tensor(value: Any) -> Any:
    import torch

    return torch.as_tensor(value, dtype=torch.float32)


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / float(num - 1)
    return [start + index * step for index in range(num)]


def _foster_ab_test_guide_ref(estimator: str) -> str:
    if estimator == "marginal_eig":
        return "boed_agent.demo.foster_variational:foster_ab_test_linear_marginal_guide"
    if estimator == "vi_eig":
        return "boed_agent.demo.foster_variational:foster_ab_test_linear_vi_guide"
    return "boed_agent.demo.foster_variational:foster_ab_test_linear_posterior_guide"


def _foster_preference_guide_ref(estimator: str) -> str:
    if estimator == "marginal_eig":
        return "boed_agent.demo.foster_variational:foster_revealed_preference_marginal_guide"
    if estimator == "vi_eig":
        return "boed_agent.demo.foster_variational:foster_revealed_preference_vi_guide"
    return "boed_agent.demo.foster_variational:foster_revealed_preference_posterior_guide"


register_callable("demo.foster_ab_test_linear_model", foster_ab_test_linear_model)
register_callable("demo.foster_ab_test_linear_posterior_guide", foster_ab_test_linear_posterior_guide)
register_callable("demo.foster_ab_test_linear_marginal_guide", foster_ab_test_linear_marginal_guide)
register_callable("demo.foster_ab_test_linear_vi_guide", foster_ab_test_linear_vi_guide)
register_callable("demo.foster_revealed_preference_model", foster_revealed_preference_model)
register_callable(
    "demo.foster_revealed_preference_posterior_guide",
    foster_revealed_preference_posterior_guide,
)
register_callable(
    "demo.foster_revealed_preference_marginal_guide",
    foster_revealed_preference_marginal_guide,
)
register_callable("demo.foster_revealed_preference_vi_guide", foster_revealed_preference_vi_guide)
register_callable("demo.make_foster_paper_adam", make_foster_paper_adam)
register_callable("demo.make_foster_trace_elbo_loss", make_foster_trace_elbo_loss)
