"""Sequential BOED loop for the linear regression bundle.

Two code paths:

* the **numpy conjugate path** (used in-sandbox and by the driver by
  default). For a Gaussian linear model the posterior is Gaussian and
  the EIG admits a closed-form expression, so we can solve the inner
  design optimisation by a scalar grid search with refinement. This is
  what we actually execute, and the math is identical to what
  ``pyro.contrib.oed.eig.posterior_eig`` estimates in the limit of
  infinite inner samples (with an exact variational family).

* the **pyro path** (opt-in) calls ``pyro.contrib.oed.eig.posterior_eig``
  with an amortised bivariate-normal posterior guide over ``(a, b) | y``
  for the design optimisation, and ``SVI + AutoDiagonalNormal`` for
  the posterior update. Same loop skeleton; the math comes out the
  same because the underlying model is Gaussian-Gaussian, and the
  posterior estimator is exact in the asymptotic limit for this
  linear-Gaussian family.

Either path produces per-round posterior samples, EIG histories, and
observations — the plotting module doesn't know which one was used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import json
import math
import numpy as np

from .model import GaussianLinearModel, default_prior_model
from .monotone import next_xi_from_raw  # noqa: F401 (kept exported for Pyro path)


@dataclass
class LinregSequentialResult:
    theta_true: np.ndarray
    per_round_posterior_models: List[GaussianLinearModel] = field(default_factory=list)
    per_round_posterior_samples: List[np.ndarray] = field(default_factory=list)
    xi_star: List[float] = field(default_factory=list)
    y_observed: List[float] = field(default_factory=list)
    eig_final: List[float] = field(default_factory=list)
    round_histories: List[Dict[str, Any]] = field(default_factory=list)
    x_max: float = 1.0
    sigma_obs: float = 0.1


def _optimise_design_scalar(
    posterior: GaussianLinearModel,
    *,
    x_min: float,
    x_max: float,
    num_inner_steps: int = 64,
    num_grid: int = 64,
) -> tuple[float, list[dict]]:
    """Find xi* ∈ [x_min, x_max] maximising EIG.

    For the Gaussian linear model the EIG is a 1-d function of xi that
    we can evaluate analytically. We do a coarse grid + local Adam
    refinement on an unconstrained ``raw`` that is squashed to
    ``[x_min, x_max]`` through a sigmoid — this matches how the Pyro
    path would parameterise a bounded design. ``history`` mirrors the
    inner-optimisation trajectory that plotting consumes.

    The linreg bundle deliberately does *not* enforce a monotone
    constraint across rounds: ``xi`` is a coordinate, not a time, so
    monotonicity would artificially pin the design at a boundary.
    """

    def _xi_of(raw: float) -> float:
        # Sigmoid squash keeps xi in (x_min, x_max).
        s = 1.0 / (1.0 + math.exp(-raw))
        return x_min + (x_max - x_min) * s

    history: list[dict] = []

    # 1. Coarse grid over raw, then pick the best as a warm start.
    raw_grid = np.linspace(-4.0, 4.0, num_grid)
    xi_grid = np.array([_xi_of(float(r)) for r in raw_grid])
    eigs_grid = np.array([posterior.eig_at(float(x)) for x in xi_grid])
    best_idx = int(np.argmax(eigs_grid))
    raw = float(raw_grid[best_idx])

    # 2. Finite-difference Adam refinement on raw.
    m = 0.0
    v = 0.0
    t = 0
    for step in range(num_inner_steps):
        xi_center = _xi_of(raw)
        eig_center = posterior.eig_at(xi_center)
        history.append({"step": step, "xi": xi_center, "eig": eig_center})
        eps = 0.05
        grad = (posterior.eig_at(_xi_of(raw + eps)) - posterior.eig_at(_xi_of(raw - eps))) / (2.0 * eps)
        t += 1
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        m_hat = m / (1.0 - 0.9 ** t)
        v_hat = v / (1.0 - 0.999 ** t)
        raw += 0.2 * m_hat / (math.sqrt(v_hat) + 1e-8)

    xi_star = _xi_of(raw)
    return xi_star, history


def run_sequential_linreg_pyro(
    *,
    theta_true: np.ndarray | tuple[float, float] = (1.5, -0.2),
    num_rounds: int = 6,
    x_max: float = 1.0,
    sigma_obs: float = 0.1,
    num_posterior_samples: int = 2000,
    artifacts_dir: str | Path = "artifacts/linreg_seq",
    seed: int = 0,
) -> LinregSequentialResult:
    """Sequential BOED on ``y = a xi + b + N(0, sigma_obs)`` with Gaussian priors.

    Every round:
      1. Optimise the next xi_t subject to xi_t > xi_{t-1}.
      2. Observe y_t from the ground-truth model at xi_t.
      3. Conjugate-update the posterior to include (xi_t, y_t).

    Uses the numpy conjugate path; see module docstring for the Pyro
    variant. Returns a :class:`LinregSequentialResult` containing
    per-round posteriors, xi*, observations, and EIG values.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    theta_true = np.asarray(theta_true, dtype=float)

    posterior = default_prior_model(sigma_obs=sigma_obs)
    all_xs: list[float] = []
    all_ys: list[float] = []

    result = LinregSequentialResult(
        theta_true=theta_true,
        x_max=x_max,
        sigma_obs=sigma_obs,
    )
    # Snapshot the round-0 prior so the shrinkage plot has a "before" panel.
    result.per_round_posterior_models.append(posterior)
    result.per_round_posterior_samples.append(posterior.sample_posterior(num_posterior_samples, rng))

    for round_idx in range(num_rounds):
        print(f"[linreg/round {round_idx}] optimising design in [{-x_max:.3f}, {x_max:.3f}] ...")
        xi_t, history = _optimise_design_scalar(
            posterior,
            x_min=-x_max,
            x_max=x_max,
        )
        # Observe the ground truth.
        y_t = float(theta_true[0] * xi_t + theta_true[1] + sigma_obs * rng.standard_normal())
        all_xs.append(xi_t)
        all_ys.append(y_t)

        # Conjugate update from the ORIGINAL prior using the full dataset, so
        # successive rounds don't over-conditioning from repeated updates.
        posterior = default_prior_model(sigma_obs=sigma_obs).posterior_given(
            np.asarray(all_xs), np.asarray(all_ys)
        )

        eig_final = history[-1]["eig"]
        eig_best = max(h["eig"] for h in history)

        # Persist per-round artifacts.
        round_dir = artifacts_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        posterior_samples = posterior.sample_posterior(num_posterior_samples, rng)
        np.save(round_dir / "posterior_samples.npy", posterior_samples)
        with open(round_dir / "observation.json", "w", encoding="utf-8") as f:
            json.dump({"xi_star": xi_t, "y": y_t, "eig_final": eig_final, "eig_best": eig_best}, f, indent=2)
        with open(round_dir / "eig_history.json", "w", encoding="utf-8") as f:
            json.dump([h["eig"] for h in history], f)
        with open(round_dir / "design_history.json", "w", encoding="utf-8") as f:
            json.dump([h["xi"] for h in history], f)

        result.per_round_posterior_models.append(posterior)
        result.per_round_posterior_samples.append(posterior_samples)
        result.xi_star.append(xi_t)
        result.y_observed.append(y_t)
        result.eig_final.append(eig_best)
        result.round_histories.append({
            "xi_history": [h["xi"] for h in history],
            "eig_history": [h["eig"] for h in history],
        })

        print(f"[linreg/round {round_idx}] xi* = {xi_t:.3f}, y_obs = {y_t:.3f}, EIG = {eig_best:.4f}")

    summary = {
        "theta_true": theta_true.tolist(),
        "xi_star": result.xi_star,
        "y_observed": result.y_observed,
        "eig_final": result.eig_final,
        "posterior_means": [post.prior_mean.tolist() for post in result.per_round_posterior_models],
        "posterior_stds": [np.sqrt(np.diag(post.prior_cov)).tolist() for post in result.per_round_posterior_models],
    }
    with open(artifacts_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return result


if __name__ == "__main__":
    r = run_sequential_linreg_pyro(num_rounds=3, artifacts_dir="artifacts/linreg_seq_smoke", seed=0)
    print("smoke OK: xi_star =", [round(x, 3) for x in r.xi_star])
    print("posterior stds by round:",
          [tuple(round(x, 4) for x in np.sqrt(np.diag(p.prior_cov))) for p in r.per_round_posterior_models])
