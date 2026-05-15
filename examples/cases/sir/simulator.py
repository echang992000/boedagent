"""Stochastic (Gillespie) SIR simulator.

A minimal continuous-time Markov chain on compartments (S, I, R) with a
closed population of size ``N``. Two latent parameters:

* ``beta``  — contact rate (units 1/time),
* ``gamma`` — recovery rate (units 1/time).

The simulator emits one trajectory per call. Observations at pre-
selected times are obtained by step-function lookup, so we can "observe
the same outbreak at progressively denser times" across sequential BOED
rounds without re-simulating history.

The main boed_agent-facing callable is :func:`simulate` — it takes
``(theta, xi)`` where ``theta = [beta, gamma]`` and ``xi`` is a length-T
vector of measurement times, and returns a length-T vector of observed
infected counts. This matches the LFIAX simulator-callable signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Defaults match Ivanova et al. (2021) / Zaballa & Hui (2025):
# total population N = 500, initial infected I0 = 2, horizon T = 100.
# (Paper uses Euler–Maruyama on the SDE; this bundle uses the exact
#  Gillespie CTMC with the same (N, I0, T) so the observed I(t) curve
#  matches the paper's setup in distribution.)
DEFAULT_N = 500
DEFAULT_I0 = 2
DEFAULT_T_MAX = 100.0
# Additive Gaussian noise on the *fraction* y / N. The paper uses a
# noiseless y = I(ξ); we keep a small positive σ (~0.5% of population)
# so the surrogate q_φ(y|θ,ξ) has a well-defined density.
# This value is interpreted in fraction-of-population space everywhere
# in the LFIAX pipeline (see TrajectoryCache.observe in
# sequential_lfiax.py). The auxiliary :func:`simulate_sir_at_times` and
# :func:`batch_simulate` functions in this module also work in fraction
# space for consistency — callers that want raw counts should multiply
# by ``N``.
DEFAULT_SIGMA_OBS = 0.005


@dataclass
class SIRTrajectory:
    """Dense step-function representation of one Gillespie run."""

    times: np.ndarray  # jump times, shape (M,)
    S: np.ndarray  # S(t-) just after each jump
    I: np.ndarray  # I(t-) just after each jump
    R: np.ndarray  # R(t-) just after each jump
    theta: np.ndarray  # (2,) the beta, gamma used

    def query(self, times: np.ndarray | Sequence[float]) -> np.ndarray:
        """Step-function lookup of I at arbitrary query times."""
        query = np.asarray(times, dtype=float).reshape(-1)
        # searchsorted returns the insertion index; we want the last jump whose
        # time <= query, so use 'right' and subtract 1.
        idx = np.clip(np.searchsorted(self.times, query, side="right") - 1, 0, len(self.I) - 1)
        return self.I[idx].astype(float)


def simulate_sir_trajectory(
    theta: np.ndarray | Sequence[float],
    *,
    N: int = DEFAULT_N,
    I0: int = DEFAULT_I0,
    t_max: float = DEFAULT_T_MAX,
    seed: int | None = None,
    max_events: int = 200_000,
) -> SIRTrajectory:
    """Run one Gillespie SIR outbreak until ``t >= t_max`` or I == 0."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    beta, gamma = float(theta[0]), float(theta[1])
    rng = np.random.default_rng(seed)

    S = N - I0
    I = I0
    R = 0
    t = 0.0

    times = [0.0]
    S_hist = [S]
    I_hist = [I]
    R_hist = [R]

    for _ in range(max_events):
        if I <= 0 or t >= t_max:
            break
        rate_inf = beta * S * I / N
        rate_rec = gamma * I
        rate_total = rate_inf + rate_rec
        if rate_total <= 0.0:
            break
        # Time until next event (exponential)
        dt = rng.exponential(1.0 / rate_total)
        t_new = t + dt
        if t_new >= t_max:
            # Lock the trajectory at t_max with the current state.
            times.append(t_max)
            S_hist.append(S)
            I_hist.append(I)
            R_hist.append(R)
            t = t_max
            break
        # Which event?
        if rng.random() < rate_inf / rate_total:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1
        t = t_new
        times.append(t)
        S_hist.append(S)
        I_hist.append(I)
        R_hist.append(R)
    else:
        # We burnt through max_events without finishing — lock at t_max.
        times.append(t_max)
        S_hist.append(S)
        I_hist.append(I)
        R_hist.append(R)

    # Ensure the trajectory covers [0, t_max] even when the outbreak dies early.
    if times[-1] < t_max:
        times.append(t_max)
        S_hist.append(S)
        I_hist.append(I)
        R_hist.append(R)

    return SIRTrajectory(
        times=np.asarray(times, dtype=float),
        S=np.asarray(S_hist, dtype=float),
        I=np.asarray(I_hist, dtype=float),
        R=np.asarray(R_hist, dtype=float),
        theta=theta.copy(),
    )


def simulate_sir_at_times(
    theta: np.ndarray | Sequence[float],
    xi: np.ndarray | Sequence[float],
    *,
    N: int = DEFAULT_N,
    I0: int = DEFAULT_I0,
    t_max: float = DEFAULT_T_MAX,
    sigma_obs: float = DEFAULT_SIGMA_OBS,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate one outbreak and observe y = I(xi_k)/N + N(0, sigma_obs^2) at each ``xi_k``.

    Returns observations in *fraction-of-population space* to match the
    downstream LFIAX pipeline (see TrajectoryCache.observe). Multiply
    by ``N`` if raw counts are needed.
    """
    traj = simulate_sir_trajectory(theta, N=N, I0=I0, t_max=t_max, seed=seed)
    clean = traj.query(xi) / float(N)
    rng = np.random.default_rng(seed)
    noisy = clean + sigma_obs * rng.standard_normal(clean.shape)
    return noisy


def simulate(theta: np.ndarray | Sequence[float], xi: np.ndarray | Sequence[float]) -> np.ndarray:
    """LFIAX-compatible signature: (theta, xi) -> observation vector.

    Default hyperparameters are baked in so this matches the
    ``simulator_ref`` contract.
    """
    return simulate_sir_at_times(theta, xi)


def batch_simulate(
    thetas: np.ndarray,
    xis: np.ndarray,
    *,
    N: int = DEFAULT_N,
    I0: int = DEFAULT_I0,
    t_max: float = DEFAULT_T_MAX,
    sigma_obs: float = DEFAULT_SIGMA_OBS,
    seed: int | None = None,
) -> np.ndarray:
    """Run one Gillespie trajectory per theta and observe at the matching xi.

    ``thetas`` has shape (B, 2); ``xis`` has shape (B,) or (B, T). Each row
    gets its own outbreak. Used by the LFIAX-style surrogate trainer.
    """
    thetas = np.asarray(thetas, dtype=float)
    xis = np.asarray(xis, dtype=float)
    if xis.ndim == 1:
        xis = xis.reshape(-1, 1)
    B = thetas.shape[0]
    assert xis.shape[0] == B, (thetas.shape, xis.shape)
    rng = np.random.default_rng(seed)
    out = np.empty(xis.shape, dtype=float)
    for b in range(B):
        child_seed = int(rng.integers(0, 2**31 - 1))
        traj = simulate_sir_trajectory(thetas[b], N=N, I0=I0, t_max=t_max, seed=child_seed)
        clean = traj.query(xis[b]) / float(N)
        noise = sigma_obs * rng.standard_normal(clean.shape)
        out[b] = clean + noise
    return out


def _smoke_test() -> None:
    rng = np.random.default_rng(0)
    peaks_I: list[float] = []
    peaks_t: list[float] = []
    for _ in range(30):
        seed = int(rng.integers(0, 2**31 - 1))
        traj = simulate_sir_trajectory([0.3, 0.1], seed=seed)
        idx = int(np.argmax(traj.I))
        peaks_I.append(float(traj.I[idx]))
        peaks_t.append(float(traj.times[idx]))
    mean_peak_I = float(np.mean(peaks_I))
    mean_peak_t = float(np.mean(peaks_t))
    print(f"30-run mean peak I = {mean_peak_I:.1f}, peak t = {mean_peak_t:.2f}")
    assert 100.0 < mean_peak_I < 600.0, mean_peak_I
    assert 5.0 < mean_peak_t < 40.0, mean_peak_t

    # Dense observation check.
    xi = np.linspace(1.0, 99.0, 20)
    y = simulate(np.array([0.3, 0.1]), xi)
    assert y.shape == (20,)
    print("observation at xi:", np.round(y[:5], 1).tolist(), "...")


if __name__ == "__main__":
    _smoke_test()
