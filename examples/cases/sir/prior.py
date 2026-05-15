"""Prior over SIR parameters and particle-based posterior helpers.

Log-normal priors on ``beta`` and ``gamma``, with a truncation that
keeps ``R0 = beta / gamma`` inside a sensible range so the Gillespie
simulator doesn't spend time on obvious non-outbreaks.

Between sequential BOED rounds the posterior is represented as a
weighted particle cloud. The public API has two pieces:

* :func:`sample_prior` — draw a batch from the round-0 prior.
* :func:`weighted_resample` — importance resample particles under a
  vector of log-weights (from the learned LFIAX-style likelihood
  surrogate), producing an equally-weighted particle cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# Round-0 prior hyperparameters in the *identifiable* parameterisation
# (log R0, log gamma). For early-outbreak SIR the likelihood is
# principally a function of R0 = beta/gamma (Wearing et al. 2005;
# Roosa & Chowell 2019); putting the prior directly on R0 aligns the
# geometry with the posterior and avoids manufactured (β, γ) bimodality.
#   p(log R0)    = Normal(log 3.0, 0.5^2)      # epidemiologically plausible
#   p(log gamma) = Normal(log 0.1, 0.5^2)
#   beta = R0 * gamma (derived)
LOG_R0_MU = np.log(3.0)
LOG_R0_SIGMA = 0.5
LOG_GAMMA_MU = np.log(0.1)
LOG_GAMMA_SIGMA = 0.5
# Admissible R0 range. The paper does not truncate, but for a particle
# filter we keep a broad guard so ~1% of prior mass at the extremes
# doesn't contaminate the cache with "no outbreak" / "instant outbreak"
# trajectories. With the LogN(log 0.5, 0.5) / LogN(log 0.1, 0.5) prior
# this keeps > 99% of the prior mass.
R0_LOW = 0.5
R0_HIGH = 20.0


@dataclass
class ParticleCloud:
    """Weighted particles representing a (possibly posterior) distribution."""

    theta: np.ndarray  # (M, 2)
    log_weights: np.ndarray  # (M,), normalised so exp(log_weights).sum() == 1

    @property
    def num_particles(self) -> int:
        return int(self.theta.shape[0])

    def effective_sample_size(self) -> float:
        w = np.exp(self.log_weights - self.log_weights.max())
        w /= w.sum()
        return float(1.0 / np.sum(w * w))


def sample_prior(num_samples: int, *, seed: int | None = None) -> np.ndarray:
    """Draw ``num_samples`` prior samples as an ``(M, 2)`` array ``[beta, gamma]``.

    Rejection-truncated on R0 ∈ [R0_LOW, R0_HIGH] so we never hand LFIAX
    a trivial "no outbreak" theta.
    """
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    while sum(x.shape[0] for x in kept) < num_samples:
        batch = _sample_untruncated(rng, size=max(num_samples, 1024))
        r0 = batch[:, 0] / batch[:, 1]
        mask = (r0 >= R0_LOW) & (r0 <= R0_HIGH)
        kept.append(batch[mask])
    out = np.concatenate(kept, axis=0)[:num_samples]
    return out


def _sample_untruncated(rng: np.random.Generator, *, size: int) -> np.ndarray:
    log_R0 = LOG_R0_MU + LOG_R0_SIGMA * rng.standard_normal(size)
    log_gamma = LOG_GAMMA_MU + LOG_GAMMA_SIGMA * rng.standard_normal(size)
    R0 = np.exp(log_R0)
    gamma = np.exp(log_gamma)
    beta = R0 * gamma
    return np.stack([beta, gamma], axis=1)


def log_prior(theta: np.ndarray) -> np.ndarray:
    """Log-density of the (truncated) prior, up to a constant.

    Prior is defined in (log R0, log gamma) space:
        log R0    ~ Normal(LOG_R0_MU, LOG_R0_SIGMA^2)
        log gamma ~ Normal(LOG_GAMMA_MU, LOG_GAMMA_SIGMA^2)
    The density in (beta, gamma) coords carries the Jacobian |det d(R0,γ)/d(β,γ)| = 1/γ.
    """
    theta = np.atleast_2d(theta)
    beta = theta[:, 0]
    gamma = theta[:, 1]
    r0 = beta / gamma
    log_r0 = np.log(r0)
    log_gamma = np.log(gamma)
    lp_r = -0.5 * ((log_r0 - LOG_R0_MU) / LOG_R0_SIGMA) ** 2 - log_r0
    lp_g = -0.5 * ((log_gamma - LOG_GAMMA_MU) / LOG_GAMMA_SIGMA) ** 2 - log_gamma
    in_support = (r0 >= R0_LOW) & (r0 <= R0_HIGH)
    # Jacobian from (log R0, log γ) to (β, γ): |d(log R0, log γ)/d(β, γ)| = 1/(β γ).
    # We drop the β·γ constant — posterior use treats log_prior as "up to a constant".
    out = lp_r + lp_g
    out[~in_support] = -np.inf
    return out


def weighted_resample(
    particles: ParticleCloud,
    *,
    num_out: int,
    seed: int | None = None,
    jitter_scale: float = 0.0,
) -> ParticleCloud:
    """Systematic resample from weighted particles; optional log-space jitter."""
    rng = np.random.default_rng(seed)
    log_w = particles.log_weights - particles.log_weights.max()
    w = np.exp(log_w)
    w /= w.sum()
    # Systematic resampling keeps low-variance draws.
    positions = (np.arange(num_out) + rng.random()) / num_out
    cumulative = np.cumsum(w)
    idx = np.searchsorted(cumulative, positions)
    idx = np.clip(idx, 0, particles.num_particles - 1)
    fresh = particles.theta[idx].copy()
    if jitter_scale > 0.0:
        # Jitter in (log R0, log gamma) space so the identifiable
        # direction (R0) is perturbed independently of the ridge.
        beta = fresh[:, 0]
        gamma = fresh[:, 1]
        log_r0 = np.log(beta / gamma) + jitter_scale * rng.standard_normal(beta.shape)
        log_gamma = np.log(gamma) + jitter_scale * rng.standard_normal(gamma.shape)
        gamma = np.exp(log_gamma)
        beta = np.exp(log_r0) * gamma
        fresh = np.stack([beta, gamma], axis=1)
    return ParticleCloud(theta=fresh, log_weights=np.full(num_out, -np.log(num_out)))


def _smoke_test() -> None:
    samples = sample_prior(4096, seed=0)
    assert samples.shape == (4096, 2)
    r0 = samples[:, 0] / samples[:, 1]
    assert (r0 >= R0_LOW).all() and (r0 <= R0_HIGH).all()
    print(
        "prior samples beta in [",
        round(samples[:, 0].min(), 3),
        ",",
        round(samples[:, 0].max(), 3),
        "], gamma in [",
        round(samples[:, 1].min(), 3),
        ",",
        round(samples[:, 1].max(), 3),
        "]",
    )
    # Resample smoke.
    log_w = np.zeros(4096)
    log_w[: 4096 // 4] = 5.0  # heavily favour first quarter
    cloud = ParticleCloud(theta=samples, log_weights=log_w)
    resampled = weighted_resample(cloud, num_out=2048, seed=1, jitter_scale=0.05)
    assert resampled.theta.shape == (2048, 2)
    print("resample ESS pre =", round(cloud.effective_sample_size(), 1), "post =", resampled.num_particles)


if __name__ == "__main__":
    _smoke_test()
