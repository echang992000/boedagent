"""Sequential BOED loop on the SIR model, LFIAX-style, pure numpy.

This file implements the machinery described in ``PLAN.md § 2`` without
needing the external ``cli-anything-lfiax`` harness or a JAX/Flax stack.
The underlying idea is the same as LFIAX:

  1. Train a conditional density ``q_φ(y | θ, ξ)`` on simulated
     ``(θ, ξ, y)`` triplets drawn from the current prior × a design
     distribution.
  2. Use ``q_φ`` to estimate EIG via the LF-PCE objective (Foster et al.
     2020 sPCE / Kleinegesse & Gutmann 2021):

         EIG(ξ) ≈ E_{θ ~ p, y ~ p(y|θ,ξ)} [ log q_φ(y | θ, ξ)
                  - log (1/L) Σ_ℓ q_φ(y | θ_ℓ, ξ) ]

  3. Jointly optimise ``φ`` and ``ξ`` in the same loop — we minimise
     ``-log q`` w.r.t. ``φ`` and maximise the contrastive EIG surrogate
     w.r.t. ``ξ``.

  4. After round ``t``, observe ``y_t`` from the real system at ``ξ_t*``
     and update the posterior by importance-reweighting the current
     particle cloud under the accumulated log-likelihood surrogate.

We use a small-enough surrogate (conditional Gaussian with a 2-layer
MLP for the mean, scalar log-scale) that everything runs in seconds on
CPU. A real LFIAX run would use a normalising flow — the loop
structure is identical, only the surrogate class changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import json
import math
import numpy as np

from .monotone import next_xi_from_raw, next_xi_grad_wrt_raw
from .prior import ParticleCloud, sample_prior, weighted_resample
from .simulator import (
    DEFAULT_I0,
    DEFAULT_N,
    DEFAULT_SIGMA_OBS,
    DEFAULT_T_MAX,
    batch_simulate,
    simulate_sir_trajectory,
)


# -----------------------------------------------------------------------------
# Trajectory cache: amortises the Gillespie cost inside one round
# -----------------------------------------------------------------------------


class TrajectoryCache:
    """One Gillespie trajectory per particle; O(1) observation queries.

    Real LFIAX runs a GPU-batched simulator inside the training loop;
    we emulate that amortisation in pure numpy by pre-simulating one
    trajectory per theta at round start, then doing step-function
    lookups for every subsequent (theta, xi) query.

    Observations are returned in fraction-of-population space (y / N)
    so the downstream surrogate operates on targets in [0, 1]
    regardless of the absolute population size. The noise standard
    deviation ``sigma_obs`` is interpreted in the same fraction space.
    """

    def __init__(
        self,
        thetas: np.ndarray,
        *,
        N: int,
        I0: int,
        t_max: float,
        sigma_obs: float,
        seed: int,
    ) -> None:
        self.thetas = np.asarray(thetas, dtype=float)
        self.N = N
        self.I0 = I0
        self.t_max = t_max
        self.sigma_obs = sigma_obs  # in fraction-of-population space
        self._rng = np.random.default_rng(seed)
        self._trajectories = []
        for i in range(self.thetas.shape[0]):
            child_seed = int(self._rng.integers(0, 2**31 - 1))
            self._trajectories.append(
                simulate_sir_trajectory(
                    self.thetas[i], N=N, I0=I0, t_max=t_max, seed=child_seed
                )
            )

    def observe(self, theta_idx: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """Noisy I(xi)/N for each (theta_idx, xi) pair (fraction space)."""
        theta_idx = np.asarray(theta_idx, dtype=int).reshape(-1)
        xi = np.asarray(xi, dtype=float).reshape(-1)
        clean = np.empty(theta_idx.shape[0], dtype=float)
        for i, (idx, x) in enumerate(zip(theta_idx, xi)):
            clean[i] = self._trajectories[idx].query(np.array([x]))[0]
        clean /= float(self.N)
        noise = self.sigma_obs * self._rng.standard_normal(clean.shape)
        return clean + noise

    def observe_paired(self, theta_idx: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`observe`."""
        return self.observe(theta_idx, xi)


# -----------------------------------------------------------------------------
# Conditional Gaussian likelihood surrogate
#
#   q_φ(y | θ, ξ) = Normal(y; μ_φ(θ, ξ), σ_φ(θ, ξ)^2)
#
# Parameterisation: a 2-hidden-layer MLP on the 3-dim input
# (log beta, log gamma, xi / t_max) with tanh activations. The final
# head emits (mu, log_sigma). Trained with Adam on the simulator.
# -----------------------------------------------------------------------------


@dataclass
class SurrogateParams:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray

    def flat_arrays(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]


@dataclass
class AdamState:
    m: list[np.ndarray]
    v: list[np.ndarray]
    t: int = 0


def _init_params(hidden: int = 64, seed: int = 0, *, mu_init: float = 0.1,
                 sigma_init: float = 0.1) -> SurrogateParams:
    """Initialise a 2-layer MLP surrogate for ``q(y/N | θ, ξ)``.

    Targets are in fraction-of-population space ``y / N ∈ [0, 1]`` so
    the default output bias sits at ``μ=0.1`` (a typical infected
    fraction mid-epidemic) with a broad ``σ=0.1``. With ``hidden=64``
    the network has enough capacity to capture the non-trivial
    ``(β, γ, ξ)`` surface without over-fitting at our batch sizes.
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(hidden)
    return SurrogateParams(
        W1=rng.standard_normal((3, hidden)) * scale,
        b1=np.zeros(hidden),
        W2=rng.standard_normal((hidden, hidden)) * scale,
        b2=np.zeros(hidden),
        W3=rng.standard_normal((hidden, 2)) * scale,
        b3=np.array([mu_init, math.log(sigma_init)]),
    )


def _init_adam(params: SurrogateParams) -> AdamState:
    arrs = params.flat_arrays()
    return AdamState(m=[np.zeros_like(a) for a in arrs], v=[np.zeros_like(a) for a in arrs])


def _featurise(theta: np.ndarray, xi: np.ndarray, *, t_max: float) -> np.ndarray:
    theta = np.atleast_2d(theta).astype(float)
    xi = np.atleast_1d(xi).astype(float)
    if xi.shape[0] != theta.shape[0]:
        if xi.size == 1:
            xi = np.full(theta.shape[0], float(xi.reshape(-1)[0]))
        else:
            raise ValueError(f"featurise shape mismatch: theta {theta.shape}, xi {xi.shape}")
    log_theta = np.log(np.clip(theta, 1e-8, None))
    return np.stack([log_theta[:, 0], log_theta[:, 1], xi / t_max], axis=1)


def _forward(params: SurrogateParams, X: np.ndarray):
    """Return (mu, log_sigma, cache) with cache for backprop.

    ``log_sigma`` is clipped to ``[-6, 1]`` which covers σ ∈ [0.0025, 2.7]
    — wide enough for fraction-space targets in ``[0, 1]`` and narrow
    enough to prevent runaway density evaluations.
    """
    h1 = np.tanh(X @ params.W1 + params.b1)
    h2 = np.tanh(h1 @ params.W2 + params.b2)
    out = h2 @ params.W3 + params.b3
    mu = out[:, 0]
    log_sigma = np.clip(out[:, 1], -6.0, 1.0)
    return mu, log_sigma, (X, h1, h2)


# Heteroscedastic variance floor in fraction space (Fix B2).
# σ_floor² = max(y(1-y)/N, σ²_obs). The binomial term is the CLT
# approximation of the Gillespie counting noise (Wald / DeMoivre;
# Kypraios et al. 2017 for CTMC ABC); the σ_obs² term is the explicit
# measurement-noise floor from problem.json. Without the σ_obs² floor
# the net can claim σ ≈ 0.002 near y=0, *tighter than the actual
# observation noise* — that overconfidence is precisely what lets a
# miscalibrated surrogate catastrophically concentrate the posterior
# on the wrong (β, γ) ridge (Hermans et al. 2021 "Averting a Crisis
# in SBI"). The floor is applied in both _log_prob and the gradient
# path so training and inference see the same likelihood.
_LIKELIHOOD_N = float(DEFAULT_N)
_LIKELIHOOD_SIGMA_OBS = float(DEFAULT_SIGMA_OBS)


def _binomial_var_floor(y: np.ndarray) -> np.ndarray:
    y_clip = np.clip(y, 0.0, 1.0)
    binom = np.maximum(y_clip * (1.0 - y_clip), 1.0 / _LIKELIHOOD_N) / _LIKELIHOOD_N
    sigma_obs2 = _LIKELIHOOD_SIGMA_OBS ** 2
    return np.maximum(binom, sigma_obs2)


def _log_prob(mu: np.ndarray, log_sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    sigma_net2 = np.exp(2.0 * log_sigma)
    sigma_eff2 = sigma_net2 + _binomial_var_floor(y)
    log_sigma_eff = 0.5 * np.log(sigma_eff2)
    return -0.5 * np.log(2.0 * math.pi) - log_sigma_eff - 0.5 * (y - mu) ** 2 / sigma_eff2


def _neg_log_prob_grads(
    params: SurrogateParams, X: np.ndarray, y: np.ndarray
) -> tuple[float, list[np.ndarray]]:
    """Batch-averaged NLL and gradients w.r.t. params. Pure numpy backprop."""
    mu, log_sigma, (X_, h1, h2) = _forward(params, X)
    sigma_net2 = np.exp(2.0 * log_sigma)
    floor = _binomial_var_floor(y)
    sigma_eff2 = sigma_net2 + floor
    # Loss per sample: -log N(y | mu, sigma_eff^2)
    #   = 0.5*log(sigma_eff²) + 0.5*(y-mu)²/sigma_eff² + 0.5*log(2π)
    resid2 = (y - mu) ** 2 / sigma_eff2
    loss = 0.5 * np.log(sigma_eff2) + 0.5 * resid2
    nll = float(loss.mean()) + 0.5 * math.log(2.0 * math.pi)

    B = X.shape[0]
    # d loss / d mu = -(y - mu) / sigma_eff²
    # d loss / d log_sigma_net = (sigma_net² / sigma_eff²) * (1 - resid²)
    # because d sigma_eff² / d log_sigma_net = 2 * sigma_net².
    d_mu = (-(y - mu) / sigma_eff2) / B
    d_log_sigma = ((sigma_net2 / sigma_eff2) * (1.0 - resid2)) / B

    d_out = np.stack([d_mu, d_log_sigma], axis=1)  # (B, 2)
    # Back through W3, b3
    dW3 = h2.T @ d_out
    db3 = d_out.sum(axis=0)
    dh2 = d_out @ params.W3.T
    # Through tanh
    dpre2 = dh2 * (1.0 - h2 ** 2)
    dW2 = h1.T @ dpre2
    db2 = dpre2.sum(axis=0)
    dh1 = dpre2 @ params.W2.T
    dpre1 = dh1 * (1.0 - h1 ** 2)
    dW1 = X_.T @ dpre1
    db1 = dpre1.sum(axis=0)
    return nll, [dW1, db1, dW2, db2, dW3, db3]


def _adam_step(
    params: SurrogateParams,
    grads: list[np.ndarray],
    state: AdamState,
    *,
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    state.t += 1
    arrs = params.flat_arrays()
    for i, (p, g) in enumerate(zip(arrs, grads)):
        state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g
        state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g
        m_hat = state.m[i] / (1.0 - beta1 ** state.t)
        v_hat = state.v[i] / (1.0 - beta2 ** state.t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# -----------------------------------------------------------------------------
# EIG estimation and design optimisation
# -----------------------------------------------------------------------------


def _antithetic_indices(
    rng: np.random.Generator, *, pool_size: int, num_samples: int, sort_key: np.ndarray,
) -> np.ndarray:
    """Draw ``num_samples`` indices as antithetic pairs along ``sort_key``.

    Sort the pool by ``sort_key`` (e.g., log R₀), draw half the samples
    uniformly, and add their reflected complements (k ↔ pool_size-1-k).
    Pairs are negatively correlated in the identified direction, which
    reduces variance of Monte-Carlo averages whose integrand depends
    primarily on R₀ (Foster 2021; Iollo et al. 2024).
    """
    order = np.argsort(sort_key)
    half = num_samples // 2
    if half == 0 or num_samples == 1:
        return rng.integers(0, pool_size, size=num_samples)
    base = rng.integers(0, pool_size, size=half)
    # Reflect across the sorted ordering: if base particle is at rank r
    # in the sorted order, its antithetic pair is at rank pool_size-1-r.
    inv_order = np.empty(pool_size, dtype=np.int64)
    inv_order[order] = np.arange(pool_size)
    base_ranks = inv_order[base]
    mirror_ranks = pool_size - 1 - base_ranks
    mirror = order[mirror_ranks]
    paired = np.concatenate([base, mirror])
    if paired.shape[0] < num_samples:
        # Odd num_samples: one extra uniform sample.
        extra = rng.integers(0, pool_size, size=num_samples - paired.shape[0])
        paired = np.concatenate([paired, extra])
    return paired


def estimate_eig(
    params: SurrogateParams,
    *,
    xi_scalar: float,
    prior_thetas: np.ndarray,
    cache: "TrajectoryCache",
    t_max: float,
    outer_samples: int,
    inner_samples: int,
    rng: np.random.Generator,
) -> float:
    """LF-PCE / sPCE estimator using the trained surrogate q.

    Uses a cached trajectory pool (one Gillespie run per particle) so
    this estimator is O(outer + inner) lookups rather than O(outer)
    fresh simulations. Outer and inner indices are drawn as antithetic
    pairs along log R₀ to reduce Monte-Carlo variance.
    """
    # Sort key = log R₀ = log β − log γ (the identifiable direction).
    log_r0 = np.log(np.clip(prior_thetas[:, 0] / prior_thetas[:, 1], 1e-8, None))
    pool_size = prior_thetas.shape[0]

    idx_outer = _antithetic_indices(
        rng, pool_size=pool_size, num_samples=outer_samples, sort_key=log_r0,
    )
    theta_outer = prior_thetas[idx_outer]
    # Observe y at the common xi_scalar for each outer theta via the cache.
    xi_vec = np.full(outer_samples, xi_scalar)
    y_outer = cache.observe(idx_outer, xi_vec)

    # log q(y | theta_outer, xi)
    X_outer = _featurise(theta_outer, xi_vec, t_max=t_max)
    mu_o, ls_o, _ = _forward(params, X_outer)
    log_q_outer = _log_prob(mu_o, ls_o, y_outer)

    # Contrastive: L inner thetas from the prior (also antithetic).
    idx_inner = _antithetic_indices(
        rng, pool_size=pool_size, num_samples=inner_samples, sort_key=log_r0,
    )
    theta_inner = prior_thetas[idx_inner]

    # For each outer sample, evaluate q(y_outer | theta_inner_j, xi)
    # for all j in inner. We vectorise over outer × inner grid.
    X_grid = _featurise(
        np.repeat(theta_inner, outer_samples, axis=0),
        np.tile(xi_vec, inner_samples),
        t_max=t_max,
    )
    mu_g, ls_g, _ = _forward(params, X_grid)
    y_tile = np.tile(y_outer, inner_samples)
    log_q_grid = _log_prob(mu_g, ls_g, y_tile).reshape(inner_samples, outer_samples)
    # Add the outer term to the contrastive set so the estimator is
    # positively biased but consistent (Foster et al. 2020).
    log_q_all = np.concatenate([log_q_grid, log_q_outer[None, :]], axis=0)
    log_marg = _logsumexp(log_q_all, axis=0) - math.log(log_q_all.shape[0])
    eig = float(np.mean(log_q_outer - log_marg))
    return eig


def _logsumexp(x: np.ndarray, axis: int = 0) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis=axis)


def estimate_eig_weighted(
    params: SurrogateParams,
    *,
    xi_scalar: float,
    prior_thetas: np.ndarray,
    weights: np.ndarray,
    cache: "TrajectoryCache",
    t_max: float,
    outer_samples: int,
    inner_samples: int,
    rng: np.random.Generator,
) -> float:
    """Same LF-PCE estimator as :func:`estimate_eig` but draws particles
    by ``weights`` instead of uniformly. Used by the 1-step lookahead
    acquisition (option 3 — Foster 2021 DAD / Iollo 2024 spirit) where
    we need EIG under a *hypothetical* posterior obtained by importance-
    reweighting the current particles by a future observation y_t."""
    pool_size = prior_thetas.shape[0]
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    idx_outer = rng.choice(pool_size, size=outer_samples, p=w, replace=True)
    theta_outer = prior_thetas[idx_outer]
    xi_vec = np.full(outer_samples, xi_scalar)
    y_outer = cache.observe(idx_outer, xi_vec)

    X_outer = _featurise(theta_outer, xi_vec, t_max=t_max)
    mu_o, ls_o, _ = _forward(params, X_outer)
    log_q_outer = _log_prob(mu_o, ls_o, y_outer)

    idx_inner = rng.choice(pool_size, size=inner_samples, p=w, replace=True)
    theta_inner = prior_thetas[idx_inner]

    X_grid = _featurise(
        np.repeat(theta_inner, outer_samples, axis=0),
        np.tile(xi_vec, inner_samples),
        t_max=t_max,
    )
    mu_g, ls_g, _ = _forward(params, X_grid)
    y_tile = np.tile(y_outer, inner_samples)
    log_q_grid = _log_prob(mu_g, ls_g, y_tile).reshape(inner_samples, outer_samples)
    log_q_all = np.concatenate([log_q_grid, log_q_outer[None, :]], axis=0)
    log_marg = _logsumexp(log_q_all, axis=0) - math.log(log_q_all.shape[0])
    return float(np.mean(log_q_outer - log_marg))


def lookahead_score(
    params: SurrogateParams,
    *,
    xi_scalar: float,
    prior_thetas: np.ndarray,
    cache: "TrajectoryCache",
    xi_grid_next: np.ndarray,
    t_max: float,
    eig_outer: int,
    eig_inner: int,
    num_y_samples: int,
    rng: np.random.Generator,
) -> float:
    """1-step lookahead bonus: E_{y_t}[max_{ξ'} EIG(ξ' | hypothetical posterior after y_t)].

    Approximates Foster 2021 DAD's non-myopic objective with a 1-step
    rollout. For each sampled y_t, importance-reweight the particles
    under q(y_t|θ,ξ_t) and compute the maximum EIG over a coarse grid
    of next designs. Average over y-samples.

    Cost: ``O(num_y_samples · |xi_grid_next| · (eig_outer + eig_inner))``.
    Use a coarser grid and fewer y-samples than the primary acquisition.
    """
    pool_size = prior_thetas.shape[0]
    log_r0 = np.log(np.clip(prior_thetas[:, 0] / prior_thetas[:, 1], 1e-8, None))
    # Sample y_t from the prior-predictive at xi_t (antithetic for variance).
    idx_y = _antithetic_indices(
        rng, pool_size=pool_size, num_samples=num_y_samples, sort_key=log_r0,
    )
    xi_vec = np.full(num_y_samples, xi_scalar)
    y_samples = cache.observe(idx_y, xi_vec)

    bonuses: list[float] = []
    for y_t in y_samples:
        # Importance reweight all particles under the surrogate at (θ_i, xi_t, y_t).
        X_all = _featurise(prior_thetas, np.full(pool_size, xi_scalar), t_max=t_max)
        mu, ls, _ = _forward(params, X_all)
        log_w = _log_prob(mu, ls, np.full(pool_size, float(y_t)))
        # Subtract max for numerical stability; normalise.
        log_w -= log_w.max()
        w = np.exp(log_w)
        w_sum = float(w.sum())
        if w_sum < 1e-12 or not np.isfinite(w_sum):
            # Degenerate y_t: skip this rollout.
            continue
        w /= w_sum
        # EIG over the coarse next-design grid under reweighted particles.
        eigs_next = np.array([
            estimate_eig_weighted(
                params,
                xi_scalar=float(x),
                prior_thetas=prior_thetas,
                weights=w,
                cache=cache,
                t_max=t_max,
                outer_samples=eig_outer,
                inner_samples=eig_inner,
                rng=rng,
            )
            for x in xi_grid_next
        ])
        bonuses.append(float(np.max(eigs_next)))
    if not bonuses:
        return 0.0
    return float(np.mean(bonuses))


# -----------------------------------------------------------------------------
# Joint (φ, ξ) optimisation inside one round
# -----------------------------------------------------------------------------


@dataclass
class RoundResult:
    xi_star: float
    eig_final: float
    xi_history: List[float]
    eig_history: List[float]
    surrogate_nll_history: List[float]


def compute_prior_predictive_peak_cap(
    prior_thetas: np.ndarray,
    *,
    N: int,
    I0: int,
    t_max: float,
    seed: int,
    peak_quantile: float = 0.75,
    peak_multiplier: float = 1.5,
    num_draws: int = 256,
    min_takeoff_multiplier: float = 3.0,
    # Back-compat aliases accepted but ignored — older code paths passed an
    # end-of-outbreak quantile / tail-threshold. The cap semantics have
    # changed to "peak-time × multiplier" (see rationale in the docstring).
    quantile: float | None = None,
    tail_threshold_frac: float | None = None,
) -> float:
    """Cap on the design space based on the prior-predictive peak time.

    Rationale (Fix B1). The SIR surrogate ``q_φ(y|θ,ξ)`` is pathologically
    over-confident when ξ sits in the post-outbreak tail where I(ξ) ≈ 0 for
    almost every θ. Greedy EIG argmax combined with monotone ξ can march
    the design into that regime and never recover. We cap the feasible
    design interval at ``peak_multiplier × Q_peak_quantile(peak_time)`` — a
    few characteristic recovery windows past the typical posterior-
    predictive peak. Past this point the prior expects most identifying
    signal to have vanished.

    An end-of-outbreak quantile does *not* work for a broad log-normal
    prior on R₀: the slow-supercritical tail (R₀ ≈ 1.1–1.5) produces
    outbreaks that take longer than ``t_max`` to end, so any end-time
    quantile saturates at ``t_max``. Peak time is bounded by ``t_max`` by
    construction and concentrates as the posterior concentrates, so the
    cap actually tightens across rounds.

    The takeoff filter (``peak I >= min_takeoff_multiplier × I0``) skips
    subcritical draws whose argmax(I) is at t = 0.

    Returns a float in ``(0.3 × t_max, t_max]``.
    """
    _ = quantile, tail_threshold_frac  # deprecated kwargs accepted for BC
    rng = np.random.default_rng(seed)
    k = min(num_draws, prior_thetas.shape[0])
    idx = rng.choice(prior_thetas.shape[0], size=k, replace=False)
    peak_times: list[float] = []
    for i in idx:
        child_seed = int(rng.integers(0, 2**31 - 1))
        traj = simulate_sir_trajectory(
            prior_thetas[i], N=N, I0=I0, t_max=t_max, seed=child_seed
        )
        I_peak = float(np.max(traj.I))
        if I_peak < min_takeoff_multiplier * I0:
            # Never took off; skip — not representative of informative designs.
            continue
        j_peak = int(np.argmax(traj.I))
        peak_times.append(float(traj.times[j_peak]))
    if not peak_times:
        # No draw took off — prior pathological. Fall back to t_max.
        return float(t_max)
    q_peak = float(np.quantile(np.asarray(peak_times), peak_quantile))
    cap = peak_multiplier * q_peak
    cap = min(cap, float(t_max))
    # Floor at 30% of t_max so very-fast-outbreak priors don't cripple
    # the run entirely.
    cap = max(cap, 0.3 * float(t_max))
    return float(cap)


def optimize_round(
    params: SurrogateParams,
    adam_state: AdamState,
    *,
    prior_thetas: np.ndarray,
    cache: "TrajectoryCache",
    xi_prev: float,
    t_max: float,
    num_steps: int,
    batch_size: int,
    eig_outer: int,
    eig_inner: int,
    design_lr: float,  # kept for API compatibility; unused by the grid-refine path
    raw_init: float,   # kept for API compatibility; initial grid covers the interval
    seed: int,
    grid_points: int = 49,
    acquisition_temperature: float | None = None,
    t_max_eff: float | None = None,
    use_lookahead: bool = False,
    lookahead_grid_points: int = 11,
    lookahead_y_samples: int = 3,
    lookahead_outer: int = 64,
    lookahead_inner: int = 32,
    lookahead_weight: float = 0.5,
    previous_designs: np.ndarray | None = None,
    min_spacing: float = 0.0,
    xi_prior_log_weight_mu: float | None = None,
    xi_prior_log_weight_sigma: float | None = None,
) -> RoundResult:
    """Train the surrogate on a *uniform* design distribution, then pick
    ``ξ_t`` by grid argmax of EIG (with a small finite-difference refine).

    The earlier version jointly trained ``(φ, ξ)`` with an Adam random
    walk on a "raw" unconstrained design parameter. That coupled the
    surrogate training with a noisy design-gradient signal and — if the
    surrogate was poorly calibrated for the first few steps — the raw
    walk could drift to uninformative corners of the feasible interval
    (ξ near 0 or near ``t_max``), and the "best-EIG-seen-during-training"
    tracker would lock that spurious high in. We separate the two
    concerns:

      1. Train ``q_φ`` on uniformly sampled ``ξ ~ U(ξ_prev, t_max)`` so
         the surrogate is calibrated everywhere in the feasible interval.
      2. Evaluate EIG on a dense grid using the trained surrogate.
      3. Take ``argmax`` and refine with a few local steps.

    This tracks the LFIAX/sPCE practice of first learning a density and
    then optimising the design under the frozen density.
    """
    rng = np.random.default_rng(seed)
    xi_history: list[float] = []
    eig_history: list[float] = []
    nll_history: list[float] = []
    xi_low = xi_prev + 1e-3
    # Fix B1: cap the design-space ceiling at the prior-predictive outbreak
    # horizon (passed via ``t_max_eff``) so greedy EIG cannot march ξ into
    # the post-outbreak tail. ``t_max`` itself stays unchanged for the
    # featurisation normaliser (xi / t_max) and for simulator bookkeeping.
    _cap = float(t_max) if t_max_eff is None else float(t_max_eff)
    _cap = min(_cap, float(t_max))
    # If xi_prev has already passed the cap (late rounds on a tightened,
    # narrow posterior), fall back to a *tight window* above xi_prev
    # rather than resetting all the way to t_max. Resetting to t_max
    # reopens the post-outbreak tail and undoes B1. A 15%-of-t_max window
    # keeps the next design within one characteristic recovery timescale
    # of xi_prev, consistent with the monotonicity constraint but still
    # well above the cap's tightened floor.
    if _cap <= xi_low:
        xi_high = min(xi_low + 0.15 * float(t_max), float(t_max))
        print(
            f"[sir/optimize_round] xi_prev={xi_prev:.2f} has passed the "
            f"B1 cap={_cap:.2f}; tight-window fallback xi_high={xi_high:.2f} "
            f"(vs t_max={t_max:.2f})."
        )
    else:
        xi_high = _cap
    assert xi_high > xi_low, (xi_prev, t_max, t_max_eff)

    # --- Stage 1: train q_phi on uniform (theta, xi, y) triples ---
    # A uniform design distribution gives the surrogate coverage of the
    # whole feasible interval, which is what the grid-argmax needs to be
    # well-posed. We still mix in a little biased sampling near the
    # running argmax (after a brief burn-in) so the region actually
    # visited gets extra attention.
    running_best_xi = 0.5 * (xi_low + xi_high)
    for step in range(num_steps):
        idx = rng.integers(0, prior_thetas.shape[0], size=batch_size)
        theta_batch = prior_thetas[idx]
        if step < max(num_steps // 3, 10):
            # Pure uniform early — get the surrogate roughly right everywhere.
            xi_batch = rng.uniform(xi_low, xi_high, size=batch_size)
        else:
            # 70% uniform + 30% around running best for fine-tuning near the peak.
            mask = rng.random(batch_size) < 0.7
            uniform = rng.uniform(xi_low, xi_high, size=batch_size)
            half = 0.15 * (xi_high - xi_low)
            around = np.clip(
                running_best_xi + rng.uniform(-half, half, size=batch_size),
                xi_low, xi_high,
            )
            xi_batch = np.where(mask, uniform, around)
        y_batch = cache.observe(idx, xi_batch)
        X_batch = _featurise(theta_batch, xi_batch, t_max=t_max)
        nll, grads = _neg_log_prob_grads(params, X_batch, y_batch)
        _adam_step(params, grads, adam_state, lr=5e-3)
        nll_history.append(nll)

        # Every so often, snapshot a coarse EIG curve so the history shows the
        # design optimiser's progress — and update running_best_xi for biased
        # sampling in the next chunk of steps.
        if step % max(num_steps // 10, 1) == 0 and step > 0:
            coarse = np.linspace(xi_low, xi_high, 17)
            eigs_coarse = np.array([
                estimate_eig(
                    params,
                    xi_scalar=float(x),
                    prior_thetas=prior_thetas,
                    cache=cache,
                    t_max=t_max,
                    outer_samples=max(eig_outer // 2, 32),
                    inner_samples=eig_inner,
                    rng=rng,
                )
                for x in coarse
            ])
            running_best_xi = float(coarse[int(np.argmax(eigs_coarse))])
            xi_history.append(running_best_xi)
            eig_history.append(float(np.max(eigs_coarse)))

    # --- Stage 2: dense grid argmax under the trained surrogate ---
    grid = np.linspace(xi_low, xi_high, grid_points)
    eigs = np.array([
        estimate_eig(
            params,
            xi_scalar=float(x),
            prior_thetas=prior_thetas,
            cache=cache,
            t_max=t_max,
            outer_samples=eig_outer,
            inner_samples=eig_inner,
            rng=rng,
        )
        for x in grid
    ])

    # Option 3 — 1-step lookahead bonus:
    #   acquisition(ξ_t) = EIG(ξ_t) + λ · E_{y_t}[max_{ξ'} EIG(ξ' | y_t)]
    # Approximates the non-myopic objective from Foster 2021 DAD with a
    # 1-step rollout (cheaper than full DAD policy training).
    if use_lookahead:
        lookahead_grid = np.linspace(xi_low, xi_high, lookahead_grid_points)
        lookahead_bonuses = np.array([
            lookahead_score(
                params,
                xi_scalar=float(x),
                prior_thetas=prior_thetas,
                cache=cache,
                xi_grid_next=lookahead_grid,
                t_max=t_max,
                eig_outer=lookahead_outer,
                eig_inner=lookahead_inner,
                num_y_samples=lookahead_y_samples,
                rng=rng,
            )
            for x in grid
        ])
        eigs_with_lookahead = eigs + lookahead_weight * lookahead_bonuses
    else:
        eigs_with_lookahead = eigs

    # Minimum-spacing penalty: forbid ξ_t within ``min_spacing`` of any
    # already-chosen design. Repeated near-identical measurements add
    # zero new information but each round's resample-with-jitter still
    # inflates σ — observed empirically as posterior-σ growth on seeds
    # whose acquisition collapses on a single ξ region. The penalty is
    # a hard veto (-inf score) so the optimiser cannot pick those points
    # at all; if every grid point is forbidden (e.g. min_spacing ≥
    # interval), we fall back to no penalty.
    # Optional Gaussian "log-prior on ξ" — adds (-0.5 * ((ξ-μ)/σ)²) to the
    # EIG grid score. With EIG in nats, this is the natural modified-BOED
    # objective: argmax [E_θ KL(p(θ|y,ξ)‖p(θ)) + log p(ξ)] with a Gaussian
    # prior p(ξ). Used at round 0 only by run_sequential_sir_lfiax to bias
    # the first design toward the rising limb (Cook et al. 2008 — both
    # limbs are needed to break the R₀ ridge; once xi_1 lands post-peak,
    # monotonicity locks future designs into the uninformative tail).
    if (xi_prior_log_weight_mu is not None
            and xi_prior_log_weight_sigma is not None
            and xi_prior_log_weight_sigma > 0):
        log_prior_xi = -0.5 * ((grid - float(xi_prior_log_weight_mu))
                               / float(xi_prior_log_weight_sigma)) ** 2
        eigs_with_lookahead = eigs_with_lookahead + log_prior_xi

    if previous_designs is not None and min_spacing > 0 and len(previous_designs) > 0:
        prev = np.asarray(previous_designs, dtype=float)
        gaps = np.min(np.abs(grid[:, None] - prev[None, :]), axis=1)
        forbidden = gaps < min_spacing
        if forbidden.all():
            print(
                f"[sir/optimize_round] min_spacing={min_spacing:.2f} forbids "
                f"every grid point; ignoring spacing penalty for this round."
            )
        else:
            eigs_with_lookahead = np.where(forbidden, -np.inf, eigs_with_lookahead)

    # Selection criterion uses lookahead-augmented score; reported
    # eig_final is always the *point* EIG at the chosen ξ_t (no
    # lookahead bonus baked in) so cross-method comparisons stay
    # apples-to-apples.
    if acquisition_temperature is None or acquisition_temperature <= 0:
        # Greedy argmax + local refine — the original behaviour.
        best_idx = int(np.argmax(eigs_with_lookahead))
        xi_star = float(grid[best_idx])
        eig_star = float(eigs[best_idx])  # point-EIG only

        # --- Stage 3: local refine via a fine grid around the argmax ---
        # Only refine when not using lookahead (the lookahead bonus is
        # smooth-but-noisy, so a tiny refinement window around the
        # argmax of the augmented score doesn't help and would mix
        # point-EIG with lookahead).
        if not use_lookahead:
            half = 0.5 * (xi_high - xi_low) / (grid_points - 1)
            fine = np.linspace(max(xi_low, xi_star - half), min(xi_high, xi_star + half), 9)
            fine_eigs = np.array([
                estimate_eig(
                    params,
                    xi_scalar=float(x),
                    prior_thetas=prior_thetas,
                    cache=cache,
                    t_max=t_max,
                    outer_samples=eig_outer,
                    inner_samples=eig_inner,
                    rng=rng,
                )
                for x in fine
            ])
            f_idx = int(np.argmax(fine_eigs))
            if fine_eigs[f_idx] > eig_star:
                xi_star = float(fine[f_idx])
                eig_star = float(fine_eigs[f_idx])
    else:
        # Boltzmann / softmax acquisition: ξ ~ softmax(score(ξ)/τ).
        # ``score`` includes the lookahead bonus when enabled.
        tau = float(acquisition_temperature)
        log_p = eigs_with_lookahead / tau
        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()
        sampled_idx = int(rng.choice(grid_points, p=p))
        xi_star = float(grid[sampled_idx])
        eig_star = float(eigs[sampled_idx])  # point-EIG only

    xi_history.append(xi_star)
    eig_history.append(eig_star)

    return RoundResult(
        xi_star=xi_star,
        eig_final=eig_star,
        xi_history=xi_history,
        eig_history=eig_history,
        surrogate_nll_history=nll_history,
    )


# -----------------------------------------------------------------------------
# Posterior update between rounds (importance reweighting).
# -----------------------------------------------------------------------------


def posterior_update(
    current_particles: ParticleCloud,
    *,
    params: SurrogateParams,
    xi_observed: np.ndarray,
    y_observed: np.ndarray,
    t_max: float,
    num_out: int,
    jitter_scale: float,
    seed: int,
) -> ParticleCloud:
    """Rescore particles under Σ_k log q(y_k | θ, xi_k), resample."""
    log_w = current_particles.log_weights.copy()
    # Add accumulated log-likelihoods under the current surrogate.
    for xi_k, y_k in zip(np.atleast_1d(xi_observed), np.atleast_1d(y_observed)):
        X = _featurise(current_particles.theta, np.full(current_particles.num_particles, float(xi_k)), t_max=t_max)
        mu, log_sigma, _ = _forward(params, X)
        log_w = log_w + _log_prob(mu, log_sigma, np.full(current_particles.num_particles, float(y_k)))
    cloud = ParticleCloud(theta=current_particles.theta, log_weights=log_w)
    return weighted_resample(cloud, num_out=num_out, seed=seed, jitter_scale=jitter_scale)


# -----------------------------------------------------------------------------
# Top-level sequential driver
# -----------------------------------------------------------------------------


@dataclass
class SequentialResult:
    theta_true: np.ndarray
    true_trajectory: Any  # SIRTrajectory from the reference run
    per_round_particles: List[np.ndarray] = field(default_factory=list)
    xi_star: List[float] = field(default_factory=list)
    y_observed: List[float] = field(default_factory=list)
    eig_final: List[float] = field(default_factory=list)
    round_histories: List[Dict[str, Any]] = field(default_factory=list)
    t_max: float = DEFAULT_T_MAX
    N: int = DEFAULT_N
    I0: int = DEFAULT_I0
    sigma_obs: float = DEFAULT_SIGMA_OBS


def run_sequential_sir_lfiax(
    *,
    theta_true: np.ndarray | tuple[float, float] = (0.35, 0.12),
    num_rounds: int = 6,
    t_max: float = DEFAULT_T_MAX,
    N: int = DEFAULT_N,
    I0: int = DEFAULT_I0,
    sigma_obs: float = DEFAULT_SIGMA_OBS,
    num_particles: int = 512,
    num_optimization_steps: int = 120,
    batch_size: int = 128,
    eig_outer: int = 96,
    eig_inner: int = 64,
    design_lr: float = 0.3,
    jitter_scale: float = 0.02,
    artifacts_dir: str | Path = "artifacts/sir_seq",
    seed: int = 0,
    monotone: bool = True,
    acquisition_temperature_start: float | None = 0.1,
    acquisition_temperature_end: float | None = 0.01,
    use_lookahead: bool = False,
    lookahead_grid_points: int = 11,
    lookahead_y_samples: int = 3,
    lookahead_outer: int = 64,
    lookahead_inner: int = 32,
    lookahead_weight: float = 0.5,
    min_design_spacing: float = 0.0,
    first_design_bias_mu: float | None = None,
    first_design_bias_sigma: float | None = None,
) -> SequentialResult:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    theta_true = np.asarray(theta_true, dtype=float)
    # Sample one "reality" trajectory once and reuse it across rounds.
    # A stochastic SIR started from I0=2 can go extinct early with
    # probability ≈ (1/R0)^I0 — the paper handles this by filtering
    # trajectories with ys[:,:,1].mean(0) >= 1 (epidemic_simulate_data.py
    # in desi-ivanova/idad). We do the same: keep resampling the truth
    # seed until we get an outbreak that actually takes off (peak I >=
    # 10 * I0) so the data has identifying signal.
    truth_seed = seed + 1
    for _ in range(64):
        candidate = simulate_sir_trajectory(
            theta_true, N=N, I0=I0, t_max=t_max, seed=truth_seed
        )
        if float(candidate.I.max()) >= max(10.0 * I0, 20.0):
            truth_traj = candidate
            break
        truth_seed += 1
    else:
        # Give up after 64 attempts and use the last candidate anyway —
        # this would only happen for a near-subcritical truth (R0 ≲ 1).
        truth_traj = candidate
    print(
        f"[sir] truth outbreak peak I = {float(truth_traj.I.max()):.0f} at t = "
        f"{float(truth_traj.times[int(np.argmax(truth_traj.I))]):.1f} (seed={truth_seed})"
    )

    # Round-0 prior particles.
    prior_thetas = sample_prior(num_particles, seed=seed)
    particles = ParticleCloud(theta=prior_thetas, log_weights=np.full(num_particles, -math.log(num_particles)))

    # Fix B1 — posterior-adaptive horizon cap. Recomputed each round from
    # the current particle cloud: the 95th percentile of end-of-outbreak
    # times (I(t) < 5% × I_peak after peaking). Prevents greedy EIG from
    # marching ξ into the post-outbreak tail where the homoscedastic-
    # Gaussian surrogate is over-confident on y ≈ 0 and can collapse the
    # posterior onto a spurious high-R₀ mode. In round 0 the cap is
    # typically loose (prior still admits slow-supercritical draws); it
    # tightens automatically once the first observation concentrates the
    # posterior away from that corner. See Hermans et al. (2021)
    # "Averting a Crisis in SBI", Lueckmann et al. (2021) SBI benchmarks.
    t_max_eff_history: list[float] = []

    # Persistent surrogate — keep training it across rounds so it builds up.
    # Operating in fraction-of-population space (y/N) so ``mu_init=0.1``
    # matches a typical mid-epidemic infected fraction.
    params = _init_params(hidden=64, seed=seed)
    adam_state = _init_adam(params)

    result = SequentialResult(
        theta_true=theta_true,
        true_trajectory=truth_traj,
        t_max=t_max,
        N=N,
        I0=I0,
        sigma_obs=sigma_obs,
    )
    # Snapshot round-0 particles (= the prior) so plots see it.
    result.per_round_particles.append(particles.theta.copy())

    xi_prev = 0.0
    xi_observed: list[float] = []
    y_observed: list[float] = []
    ess_history: list[float] = []

    for round_idx in range(num_rounds):
        # Lower bound for the feasible design interval.
        # ``monotone=True`` preserves the physical assay-time monotonicity
        # from Zaballa & Hui (2025). For Gillespie SIR, trajectories are
        # queryable at any t, so the constraint is artificial — and it
        # blocks the acquisition function from revisiting the growth
        # phase after the first design lands late (Cook et al. 2008 show
        # both limbs are needed to break the R₀ ridge). We default to
        # ``monotone=False``.
        xi_low_for_round = xi_prev if monotone else 0.0
        print(
            f"[sir/round {round_idx}] optimising design with xi > {xi_low_for_round:.2f} "
            f"(monotone={monotone}) ..."
        )
        # Build a per-round Gillespie cache: one trajectory per particle.
        # This is the amortisation trick — all subsequent EIG evaluations
        # and surrogate-training batches become step-function lookups.
        cache = TrajectoryCache(
            particles.theta,
            N=N,
            I0=I0,
            t_max=t_max,
            sigma_obs=sigma_obs,
            seed=seed + 1000 * (round_idx + 1),
        )
        # Geometric annealing between τ_start and τ_end (both > 0 ⇒ Boltzmann
        # sampling; set τ_start=None to fall back to greedy argmax). Warm
        # early rounds explore; cool later rounds exploit.
        if (
            acquisition_temperature_start is not None
            and acquisition_temperature_end is not None
            and acquisition_temperature_start > 0
            and acquisition_temperature_end > 0
        ):
            if num_rounds > 1:
                frac = round_idx / (num_rounds - 1)
            else:
                frac = 0.0
            tau_round = float(
                acquisition_temperature_start
                * (acquisition_temperature_end / acquisition_temperature_start) ** frac
            )
        else:
            tau_round = None

        # Fix B1: posterior-adaptive horizon cap. Recompute the design-space
        # ceiling each round from the current particle cloud so the cap
        # tightens as the posterior concentrates. This prevents greedy EIG
        # from marching ξ into the post-outbreak tail where q_φ(y|θ,ξ) is
        # over-confident on y ≈ 0 (Hermans et al. 2021). We use the 80th
        # percentile of end-of-outbreak times rather than the 95th so the
        # cap actually bites on broad posteriors that still admit slow-
        # supercritical draws. The monotonicity constraint already ensures
        # coverage of late times *if* early-round designs land there; the
        # cap's job is to block them from running off the right edge.
        t_max_eff = compute_prior_predictive_peak_cap(
            particles.theta,
            N=N,
            I0=I0,
            t_max=t_max,
            seed=seed + 2000 * (round_idx + 1),
            peak_quantile=0.75,
            peak_multiplier=1.5,
            num_draws=256,
        )
        t_max_eff_history.append(t_max_eff)
        print(
            f"[sir/round {round_idx}] posterior-adaptive design cap = "
            f"{t_max_eff:.2f} (of t_max={t_max:.2f})"
        )

        # Disable lookahead on the FINAL round — there is no ξ_{t+1} to
        # plan for. Otherwise lookahead spends budget computing a
        # bonus that's irrelevant to the picked design.
        round_use_lookahead = use_lookahead and (round_idx < num_rounds - 1)
        round_out = optimize_round(
            params,
            adam_state,
            prior_thetas=particles.theta,
            cache=cache,
            xi_prev=xi_low_for_round,
            t_max=t_max,
            num_steps=num_optimization_steps,
            batch_size=batch_size,
            eig_outer=eig_outer,
            eig_inner=eig_inner,
            design_lr=design_lr,
            raw_init=-1.0,
            seed=seed + 10 * (round_idx + 1),
            acquisition_temperature=tau_round,
            t_max_eff=t_max_eff,
            use_lookahead=round_use_lookahead,
            lookahead_grid_points=lookahead_grid_points,
            lookahead_y_samples=lookahead_y_samples,
            lookahead_outer=lookahead_outer,
            lookahead_inner=lookahead_inner,
            lookahead_weight=lookahead_weight,
            previous_designs=np.asarray(xi_observed, dtype=float) if xi_observed else None,
            min_spacing=min_design_spacing,
            xi_prior_log_weight_mu=(first_design_bias_mu if round_idx == 0 else None),
            xi_prior_log_weight_sigma=(first_design_bias_sigma if round_idx == 0 else None),
        )

        xi_t = round_out.xi_star
        # Observe the ground-truth system at xi_t. Surrogate and cache
        # operate in fraction space (y / N), so noise is sampled in
        # fraction space too.
        clean_count = float(truth_traj.query(np.array([xi_t]))[0])
        noise_frac = sigma_obs * float(rng.standard_normal())
        y_frac = clean_count / float(N) + noise_frac  # fraction space
        y_count = y_frac * float(N)  # count space for human-readable summary
        xi_observed.append(xi_t)
        y_observed.append(y_frac)  # stored in fraction space for the surrogate
        if monotone:
            assert xi_t > xi_prev - 1e-9, (
                f"monotonicity violated at round {round_idx}: "
                f"xi_t={xi_t:.3f} not > xi_prev={xi_prev:.3f}"
            )

        # Update posterior using *only* the newest observation. The
        # incoming `particles` are already a sample from
        # p(θ | y_{1..t-1}) from the previous round's resample, so
        # multiplying in log q(y_k | θ, ξ_k) for every historical k
        # double-counts the old observations under the latest surrogate.
        # One update, one likelihood factor per round — standard SMC.
        _pre_update = particles
        _X_ess = _featurise(
            _pre_update.theta,
            np.full(_pre_update.num_particles, float(xi_t)),
            t_max=t_max,
        )
        _mu_ess, _ls_ess, _ = _forward(params, _X_ess)
        _log_incr = _log_prob(
            _mu_ess, _ls_ess,
            np.full(_pre_update.num_particles, float(y_observed[-1])),
        )
        _log_w = _pre_update.log_weights + _log_incr
        _w = np.exp(_log_w - _log_w.max())
        _w /= _w.sum()
        _ess = float(1.0 / np.sum(_w * _w))
        ess_history.append(_ess)
        particles = posterior_update(
            particles,
            params=params,
            xi_observed=np.array([xi_t]),
            y_observed=np.array([y_observed[-1]]),
            t_max=t_max,
            num_out=num_particles,
            jitter_scale=jitter_scale,
            seed=seed + 100 * (round_idx + 1),
        )

        # Persist per-round artifacts (store y in *counts* for readability).
        round_dir = artifacts_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        np.save(round_dir / "posterior_samples.npy", particles.theta)
        with open(round_dir / "observation.json", "w", encoding="utf-8") as f:
            json.dump({"xi_star": xi_t, "y": y_count, "eig_final": round_out.eig_final}, f, indent=2)
        with open(round_dir / "xi_history.json", "w", encoding="utf-8") as f:
            json.dump(round_out.xi_history, f)
        with open(round_dir / "eig_history.json", "w", encoding="utf-8") as f:
            json.dump(round_out.eig_history, f)

        result.per_round_particles.append(particles.theta.copy())
        result.xi_star.append(xi_t)
        result.y_observed.append(y_count)  # report in counts for humans
        result.eig_final.append(round_out.eig_final)
        result.round_histories.append({
            "xi_history": round_out.xi_history,
            "eig_history": round_out.eig_history,
            "surrogate_nll_history": round_out.surrogate_nll_history,
        })

        print(
            f"[sir/round {round_idx}] xi* = {xi_t:.2f}, y_obs = {y_count:.1f}, "
            f"EIG = {round_out.eig_final:.3f}, ESS = {_ess:.1f} / {num_particles}"
        )
        xi_prev = xi_t

    posterior_R0_means = [
        float(np.mean(particles_t[:, 0] / particles_t[:, 1]))
        for particles_t in result.per_round_particles
    ]
    posterior_R0_stds = [
        float(np.std(particles_t[:, 0] / particles_t[:, 1]))
        for particles_t in result.per_round_particles
    ]
    R0_truth = float(theta_true[0] / theta_true[1])
    R0_bias_final = posterior_R0_means[-1] - R0_truth
    R0_bias_over_std_final = (
        abs(R0_bias_final) / posterior_R0_stds[-1]
        if posterior_R0_stds[-1] > 0 else float("inf")
    )

    # Fix 6: flag designs placed in the low-information tail (Cook 2008,
    # Pagendam & Ross 2013). Fisher information is ~0 after I(t) drops
    # below 5% of its peak; chosen ξ in that regime should yield low EIG.
    I0_peak = float(truth_traj.I.max())
    low_info_threshold = 0.05 * I0_peak
    I_at_xi = [
        float(truth_traj.query(np.array([x]))[0]) for x in result.xi_star
    ]
    designs_in_low_info_regime = [
        bool(i < low_info_threshold) for i in I_at_xi
    ]
    for r, (xi_t, i_val, eig_val, is_low) in enumerate(
        zip(result.xi_star, I_at_xi, result.eig_final, designs_in_low_info_regime)
    ):
        if is_low:
            print(
                f"[sir/round {r}] WARNING: chosen ξ in low-information tail "
                f"(I(ξ)={i_val:.1f} < {low_info_threshold:.1f}); "
                f"EIG at this design is {eig_val:.3f}"
            )

    summary = {
        "theta_true": theta_true.tolist(),
        "xi_star": result.xi_star,
        "y_observed": result.y_observed,
        "eig_final": result.eig_final,
        "ess_per_round": ess_history,
        "posterior_means": [
            [float(np.mean(particles_t[:, 0])), float(np.mean(particles_t[:, 1]))]
            for particles_t in result.per_round_particles
        ],
        "posterior_stds": [
            [float(np.std(particles_t[:, 0])), float(np.std(particles_t[:, 1]))]
            for particles_t in result.per_round_particles
        ],
        "posterior_R0_means": posterior_R0_means,
        "posterior_R0_stds": posterior_R0_stds,
        # Fix 5: R₀ bias diagnostics.
        "R0_truth": R0_truth,
        "R0_posterior_bias": R0_bias_final,
        "R0_posterior_bias_over_std": R0_bias_over_std_final,
        "R0_posterior_mean_trajectory": posterior_R0_means,
        # Fix 6: deep-tail design diagnostic.
        "designs_in_low_info_regime": designs_in_low_info_regime,
        "I_at_xi": I_at_xi,
        "I_peak_truth": I0_peak,
        # Fix B1: per-round posterior-adaptive design horizon cap.
        "t_max_eff_history": t_max_eff_history,
    }
    with open(artifacts_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Stash diagnostics on the result object so downstream plotters can
    # annotate without re-reading summary.json.
    result.R0_bias_over_std = R0_bias_over_std_final  # type: ignore[attr-defined]
    result.ess_per_round = list(ess_history)  # type: ignore[attr-defined]
    result.num_particles_config = num_particles  # type: ignore[attr-defined]

    return result


# -----------------------------------------------------------------------------
# Uniform-grid baseline (Fix 4): same pipeline, non-adaptive design schedule.
# -----------------------------------------------------------------------------


def run_sequential_sir_lfiax_uniform(
    *,
    theta_true: np.ndarray | tuple[float, float] = (0.3332, 0.1103),
    num_rounds: int = 6,
    t_max: float = DEFAULT_T_MAX,
    N: int = DEFAULT_N,
    I0: int = DEFAULT_I0,
    sigma_obs: float = DEFAULT_SIGMA_OBS,
    num_particles: int = 512,
    num_optimization_steps: int = 120,
    batch_size: int = 128,
    eig_outer: int = 96,
    eig_inner: int = 64,
    jitter_scale: float = 0.02,
    artifacts_dir: str | Path = "artifacts/sir_seq_uniform",
    seed: int = 0,
    fixed_schedule: list[float] | None = None,
) -> SequentialResult:
    """Baseline with a fixed (non-adaptive) design schedule.

    Default schedule is evenly-spaced over (0, t_max] (Kleinegesse &
    Gutmann 2021 Table 1 style). Pass ``fixed_schedule`` to override
    with a hand-picked set of T design times — e.g. a physics-aware
    schedule that puts more designs on the rising limb where R₀ is
    identifiable. This is the natural "optimal non-adaptive batch"
    baseline that DAD-style amortised policies beat (Foster 2021).
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    theta_true = np.asarray(theta_true, dtype=float)

    # Match the BOED driver's "require an outbreak" filter so both methods
    # see the same latent truth when given the same seed.
    truth_seed = seed + 1
    for _ in range(64):
        candidate = simulate_sir_trajectory(
            theta_true, N=N, I0=I0, t_max=t_max, seed=truth_seed
        )
        if float(candidate.I.max()) >= max(10.0 * I0, 20.0):
            truth_traj = candidate
            break
        truth_seed += 1
    else:
        truth_traj = candidate
    print(
        f"[uniform] truth outbreak peak I = {float(truth_traj.I.max()):.0f} at t = "
        f"{float(truth_traj.times[int(np.argmax(truth_traj.I))]):.1f} (seed={truth_seed})"
    )

    # Fixed design schedule. Default is evenly-spaced on (0, t_max].
    # If ``fixed_schedule`` is given, use that exact list — e.g. a
    # physics-aware schedule like [8, 18, 28, 45, 65, 85] that puts
    # 3 designs on the rising/peak phase and 3 on the decline.
    if fixed_schedule is not None and len(fixed_schedule) > 0:
        if len(fixed_schedule) != num_rounds:
            raise ValueError(
                f"fixed_schedule length {len(fixed_schedule)} != num_rounds "
                f"{num_rounds}"
            )
        xi_schedule = np.asarray(fixed_schedule, dtype=float)
        print(f"[uniform] custom xi schedule: {[round(float(x), 2) for x in xi_schedule]}")
    else:
        xi_schedule = np.linspace(
            t_max / (num_rounds + 1),
            num_rounds * t_max / (num_rounds + 1),
            num_rounds,
        )
        print(f"[uniform] xi schedule: {[round(float(x), 2) for x in xi_schedule]}")

    prior_thetas = sample_prior(num_particles, seed=seed)
    particles = ParticleCloud(
        theta=prior_thetas,
        log_weights=np.full(num_particles, -math.log(num_particles)),
    )
    params = _init_params(hidden=64, seed=seed)
    adam_state = _init_adam(params)

    result = SequentialResult(
        theta_true=theta_true,
        true_trajectory=truth_traj,
        t_max=t_max,
        N=N,
        I0=I0,
        sigma_obs=sigma_obs,
    )
    result.per_round_particles.append(particles.theta.copy())

    xi_prev = 0.0
    xi_observed: list[float] = []
    y_observed: list[float] = []
    ess_history: list[float] = []

    for round_idx in range(num_rounds):
        xi_t = float(xi_schedule[round_idx])
        print(f"[uniform/round {round_idx}] xi = {xi_t:.2f}")

        cache = TrajectoryCache(
            particles.theta,
            N=N,
            I0=I0,
            t_max=t_max,
            sigma_obs=sigma_obs,
            seed=seed + 1000 * (round_idx + 1),
        )

        # Train surrogate on uniform (theta, xi, y) batches — same
        # schedule as the BOED driver's Stage 1. We skip the Stage 2/3
        # EIG maximisation since the design is predetermined, but we
        # still want a calibrated surrogate for the posterior update
        # and for EIG reporting at xi_t.
        xi_low = 1e-3
        xi_high = float(t_max)
        local_rng = np.random.default_rng(seed + 10 * (round_idx + 1))
        for _step in range(num_optimization_steps):
            idx = local_rng.integers(0, prior_thetas.shape[0], size=batch_size)
            theta_batch = prior_thetas[idx]
            xi_batch = local_rng.uniform(xi_low, xi_high, size=batch_size)
            y_batch = cache.observe(idx, xi_batch)
            X_batch = _featurise(theta_batch, xi_batch, t_max=t_max)
            _nll, grads = _neg_log_prob_grads(params, X_batch, y_batch)
            _adam_step(params, grads, adam_state, lr=5e-3)

        eig_at_xi = estimate_eig(
            params,
            xi_scalar=xi_t,
            prior_thetas=prior_thetas,
            cache=cache,
            t_max=t_max,
            outer_samples=eig_outer,
            inner_samples=eig_inner,
            rng=np.random.default_rng(seed + 5000 * (round_idx + 1)),
        )

        clean_count = float(truth_traj.query(np.array([xi_t]))[0])
        noise_frac = sigma_obs * float(rng.standard_normal())
        y_frac = clean_count / float(N) + noise_frac
        y_count = y_frac * float(N)
        xi_observed.append(xi_t)
        y_observed.append(y_frac)

        # Diagnostic ESS before resample.
        X_ess = _featurise(
            particles.theta,
            np.full(particles.num_particles, float(xi_t)),
            t_max=t_max,
        )
        mu_ess, ls_ess, _ = _forward(params, X_ess)
        log_incr = _log_prob(
            mu_ess, ls_ess,
            np.full(particles.num_particles, float(y_frac)),
        )
        log_w = particles.log_weights + log_incr
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        ess = float(1.0 / np.sum(w * w))
        ess_history.append(ess)

        particles = posterior_update(
            particles,
            params=params,
            xi_observed=np.array([xi_t]),
            y_observed=np.array([y_frac]),
            t_max=t_max,
            num_out=num_particles,
            jitter_scale=jitter_scale,
            seed=seed + 100 * (round_idx + 1),
        )

        round_dir = artifacts_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        np.save(round_dir / "posterior_samples.npy", particles.theta)
        with open(round_dir / "observation.json", "w", encoding="utf-8") as f:
            json.dump({"xi_star": xi_t, "y": y_count, "eig_final": eig_at_xi}, f, indent=2)

        result.per_round_particles.append(particles.theta.copy())
        result.xi_star.append(xi_t)
        result.y_observed.append(y_count)
        result.eig_final.append(eig_at_xi)
        result.round_histories.append({
            "xi_history": [xi_t],
            "eig_history": [eig_at_xi],
            "surrogate_nll_history": [],
        })

        print(
            f"[uniform/round {round_idx}] xi = {xi_t:.2f}, y_obs = {y_count:.1f}, "
            f"EIG(xi) = {eig_at_xi:.3f}, ESS = {ess:.1f} / {num_particles}"
        )
        xi_prev = xi_t

    posterior_R0_means = [
        float(np.mean(particles_t[:, 0] / particles_t[:, 1]))
        for particles_t in result.per_round_particles
    ]
    posterior_R0_stds = [
        float(np.std(particles_t[:, 0] / particles_t[:, 1]))
        for particles_t in result.per_round_particles
    ]
    R0_truth = float(theta_true[0] / theta_true[1])
    R0_bias_final = posterior_R0_means[-1] - R0_truth
    R0_bias_over_std_final = (
        abs(R0_bias_final) / posterior_R0_stds[-1]
        if posterior_R0_stds[-1] > 0 else float("inf")
    )

    I0_peak = float(truth_traj.I.max())
    low_info_threshold = 0.05 * I0_peak
    I_at_xi = [
        float(truth_traj.query(np.array([x]))[0]) for x in result.xi_star
    ]
    designs_in_low_info_regime = [bool(i < low_info_threshold) for i in I_at_xi]

    summary = {
        "theta_true": theta_true.tolist(),
        "xi_star": result.xi_star,
        "y_observed": result.y_observed,
        "eig_final": result.eig_final,
        "ess_per_round": ess_history,
        "posterior_means": [
            [float(np.mean(particles_t[:, 0])), float(np.mean(particles_t[:, 1]))]
            for particles_t in result.per_round_particles
        ],
        "posterior_stds": [
            [float(np.std(particles_t[:, 0])), float(np.std(particles_t[:, 1]))]
            for particles_t in result.per_round_particles
        ],
        "posterior_R0_means": posterior_R0_means,
        "posterior_R0_stds": posterior_R0_stds,
        "R0_truth": R0_truth,
        "R0_posterior_bias": R0_bias_final,
        "R0_posterior_bias_over_std": R0_bias_over_std_final,
        "R0_posterior_mean_trajectory": posterior_R0_means,
        "designs_in_low_info_regime": designs_in_low_info_regime,
        "I_at_xi": I_at_xi,
        "I_peak_truth": I0_peak,
        "method": "uniform_grid",
    }
    with open(artifacts_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    result.ess_per_round = list(ess_history)  # type: ignore[attr-defined]
    result.num_particles_config = num_particles  # type: ignore[attr-defined]
    return result


if __name__ == "__main__":
    # Smaller-scale smoke — proves the loop runs.
    result = run_sequential_sir_lfiax(
        num_rounds=3,
        num_optimization_steps=30,
        batch_size=64,
        eig_outer=32,
        eig_inner=32,
        num_particles=256,
        artifacts_dir="artifacts/sir_seq_smoke",
        seed=0,
    )
    print("smoke OK, xi_star sequence:", [round(x, 2) for x in result.xi_star])
