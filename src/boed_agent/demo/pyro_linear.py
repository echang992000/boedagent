"""Small Pyro demo model and guide functions used in examples and tests."""

from __future__ import annotations

from typing import Any

from boed_agent.utils.imports import register_callable


def pyro_linear_model(design: Any) -> None:
    import pyro
    import pyro.distributions as dist

    design_tensor = _as_design_tensor(design)
    theta = pyro.sample(
        "theta",
        dist.Normal(_zeros_like(design_tensor), _ones_like(design_tensor)).to_event(1),
    )
    loc = design_tensor * theta
    pyro.sample("y", dist.Normal(loc, _ones_like(loc)).to_event(1))


def pyro_linear_marginal_guide(design: Any, observation_labels: list[str], target_labels: list[str]) -> None:
    import pyro
    import pyro.distributions as dist
    import torch
    from pyro.distributions import constraints

    _ = target_labels
    design_tensor = _as_design_tensor(design)
    event_shape = _event_shape(design_tensor)
    loc = pyro.param("marginal_loc", torch.zeros(event_shape, dtype=design_tensor.dtype, device=design_tensor.device))
    scale = pyro.param(
        "marginal_scale",
        torch.ones(event_shape, dtype=design_tensor.dtype, device=design_tensor.device),
        constraint=constraints.positive,
    )
    pyro.sample(
        observation_labels[0],
        dist.Normal(loc.expand(design_tensor.shape), scale.expand(design_tensor.shape)).to_event(1),
    )


def pyro_linear_posterior_guide(
    y_dict: dict[str, Any],
    design: Any,
    observation_labels: list[str],
    target_labels: list[str],
) -> None:
    import pyro
    import pyro.distributions as dist
    import torch
    from pyro.distributions import constraints

    y = y_dict[observation_labels[0]]
    design_tensor = _as_design_tensor(design)
    event_shape = _event_shape(design_tensor)
    denom = torch.clamp(design_tensor.abs(), min=1.0)
    bias = pyro.param(
        "posterior_loc",
        torch.zeros(event_shape, dtype=design_tensor.dtype, device=design_tensor.device),
    )
    loc = bias.expand(design_tensor.shape) + y / denom
    scale = pyro.param(
        "posterior_scale",
        0.5 * torch.ones(event_shape, dtype=design_tensor.dtype, device=design_tensor.device),
        constraint=constraints.positive,
    )
    pyro.sample(
        target_labels[0],
        dist.Normal(loc, scale.expand(design_tensor.shape)).to_event(1),
    )


def pyro_linear_vi_guide(design: Any) -> None:
    import pyro
    import pyro.distributions as dist
    import torch
    from pyro.distributions import constraints

    design_tensor = _as_design_tensor(design)
    event_shape = _event_shape(design_tensor)
    loc = pyro.param("vi_loc", torch.zeros(event_shape, dtype=design_tensor.dtype, device=design_tensor.device))
    scale = pyro.param(
        "vi_scale",
        torch.ones(event_shape, dtype=design_tensor.dtype, device=design_tensor.device),
        constraint=constraints.positive,
    )
    pyro.sample(
        "theta",
        dist.Normal(loc.expand(design_tensor.shape), scale.expand(design_tensor.shape)).to_event(1),
    )


def make_trace_elbo_loss(spec: Any | None = None) -> Any:
    import pyro

    _ = spec
    return pyro.infer.Trace_ELBO().differentiable_loss


def make_pyro_adam(spec: Any | None = None) -> Any:
    import pyro

    learning_rate = 0.05
    if spec is not None:
        learning_rate = spec.backend_options.get("guide_learning_rate", learning_rate)
    return pyro.optim.Adam({"lr": learning_rate})


def _as_design_tensor(design: Any) -> Any:
    import torch

    tensor = design if isinstance(design, torch.Tensor) else torch.tensor(design, dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    return tensor


def _event_shape(tensor: Any) -> tuple[int, ...]:
    return (tensor.shape[-1],)


def _zeros_like(tensor: Any) -> Any:
    import torch

    return torch.zeros_like(tensor)


def _ones_like(tensor: Any) -> Any:
    import torch

    return torch.ones_like(tensor)


register_callable("demo.pyro_linear_model", pyro_linear_model)
register_callable("demo.pyro_linear_marginal_guide", pyro_linear_marginal_guide)
register_callable("demo.pyro_linear_posterior_guide", pyro_linear_posterior_guide)
register_callable("demo.pyro_linear_vi_guide", pyro_linear_vi_guide)
register_callable("demo.make_trace_elbo_loss", make_trace_elbo_loss)
register_callable("demo.make_pyro_adam", make_pyro_adam)
