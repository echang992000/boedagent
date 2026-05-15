"""Small LFIAX linear demo problem and checkpoint helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable


def prior(key: Any, n: int) -> Any:
    import jax.random as jrandom

    return jrandom.normal(key, shape=(int(n), 1))


def simulator(theta: Any, xi: Any, key: Any) -> Any:
    import jax.numpy as jnp
    import jax.random as jrandom

    theta_arr = jnp.asarray(theta)
    xi_arr = jnp.asarray(xi)
    if theta_arr.ndim == 1:
        theta_arr = theta_arr[:, None]
    if xi_arr.ndim == 1:
        xi_arr = jnp.broadcast_to(xi_arr, (theta_arr.shape[0], xi_arr.shape[0]))
    noise = 0.1 * jrandom.normal(key, shape=(theta_arr.shape[0], xi_arr.shape[-1]))
    return theta_arr * xi_arr + noise


def load_likelihood_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def build_log_prob_from_checkpoint(path: str | Path) -> Callable[[Any, Any, Any], Any]:
    checkpoint = load_likelihood_checkpoint(path)
    metadata = checkpoint["metadata"]
    flow_params = checkpoint["flow_params"]

    import haiku as hk
    import jax.numpy as jnp
    from lfiax.flows.nsf import make_nsf

    flow_cfg = metadata["flow_config"]
    y_dim = int(metadata["y_dim"])

    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(data: Any, context_theta: Any, context_xi: Any) -> Any:
        model = make_nsf(
            event_shape=(y_dim,),
            num_layers=int(flow_cfg["num_layers"]),
            hidden_sizes=list(flow_cfg["hidden_sizes"]),
            num_bins=int(flow_cfg["num_bins"]),
            conditional=True,
        )
        return model.log_prob(data, context_theta, context_xi)

    def log_prob(y: Any, theta: Any, xi: Any) -> Any:
        return log_prob_fn.apply(flow_params, jnp.asarray(y), jnp.asarray(theta), jnp.asarray(xi))

    return log_prob


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    checkpoint = load_likelihood_checkpoint(path)
    return dict(checkpoint.get("metadata", {}))
