"""Uniform-baseline driver: same SIR / surrogate / posterior machinery as
``sir_sequential_agent.py`` but with designs fixed on an evenly-spaced
grid instead of chosen by EIG optimisation.

Purpose: isolate the contribution of the BOED acquisition function from
the rest of the likelihood-free pipeline (truth trajectory, surrogate
training, importance-resample posterior update).

Usage::

    python examples/agent/sir_uniform_baseline.py --rounds 6 --seed 0 \\
        --artifacts artifacts/sir_seq_uniform_s0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifacts", type=str, default="artifacts/sir_seq_uniform")
    parser.add_argument("--theta-true", type=float, nargs=2, default=(0.3332, 0.1103),
                        metavar=("BETA", "GAMMA"))
    parser.add_argument("--num-optimization-steps", type=int, default=120)
    parser.add_argument("--num-particles", type=int, default=512)
    parser.add_argument("--eig-outer", type=int, default=96)
    parser.add_argument("--eig-inner", type=int, default=64)
    args = parser.parse_args()

    from examples.cases.sir import plotting
    from examples.cases.sir.prior import ParticleCloud, sample_prior
    from examples.cases.sir.sequential_lfiax import (
        DEFAULT_I0,
        DEFAULT_N,
        DEFAULT_SIGMA_OBS,
        DEFAULT_T_MAX,
        SequentialResult,
        TrajectoryCache,
        _adam_step,
        _featurise,
        _forward,
        _init_adam,
        _init_params,
        _log_prob,
        _neg_log_prob_grads,
        estimate_eig,
        posterior_update,
    )
    from examples.cases.sir.simulator import simulate_sir_trajectory

    theta_true = np.asarray(args.theta_true, dtype=float)
    t_max = DEFAULT_T_MAX
    N = DEFAULT_N
    I0 = DEFAULT_I0
    sigma_obs = DEFAULT_SIGMA_OBS
    num_particles = args.num_particles
    num_steps = args.num_optimization_steps
    batch_size = 128
    num_rounds = args.rounds

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- Truth trajectory sampled exactly as in the BOED driver so the two
    # methods see the same latent outbreak when given the same seed.
    truth_seed = args.seed + 1
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

    # --- Evenly-spaced uniform design schedule over (0, t_max]. --------------
    xi_schedule = [t_max * (t + 1) / (num_rounds + 1) for t in range(num_rounds)]
    print(f"[uniform] xi schedule: {[round(x, 2) for x in xi_schedule]}")

    prior_thetas = sample_prior(num_particles, seed=args.seed)
    particles = ParticleCloud(
        theta=prior_thetas,
        log_weights=np.full(num_particles, -math.log(num_particles)),
    )

    params = _init_params(hidden=64, seed=args.seed)
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
    y_observed: list[float] = []
    ess_history: list[float] = []

    for round_idx in range(num_rounds):
        xi_t = float(xi_schedule[round_idx])
        print(f"[uniform/round {round_idx}] xi = {xi_t:.2f} (prev {xi_prev:.2f})")

        # Build a per-round Gillespie cache (same bookkeeping as the BOED run).
        cache = TrajectoryCache(
            particles.theta,
            N=N,
            I0=I0,
            t_max=t_max,
            sigma_obs=sigma_obs,
            seed=args.seed + 1000 * (round_idx + 1),
        )

        # Train the surrogate on uniform (theta, xi, y) triples — same
        # schedule as sequential_lfiax.optimize_round stage 1. We skip
        # stage-2/3 EIG maximisation since the design is predetermined,
        # but we still want a calibrated surrogate for the posterior
        # update and for the EIG diagnostic at xi_t.
        xi_low = xi_prev + 1e-3
        xi_high = float(t_max)
        local_rng = np.random.default_rng(args.seed + 10 * (round_idx + 1))
        for step in range(num_steps):
            idx = local_rng.integers(0, prior_thetas.shape[0], size=batch_size)
            theta_batch = prior_thetas[idx]
            xi_batch = local_rng.uniform(xi_low, xi_high, size=batch_size)
            y_batch = cache.observe(idx, xi_batch)
            X_batch = _featurise(theta_batch, xi_batch, t_max=t_max)
            _nll, grads = _neg_log_prob_grads(params, X_batch, y_batch)
            _adam_step(params, grads, adam_state, lr=5e-3)

        # Evaluate EIG at the scheduled xi_t using the same LF-PCE estimator.
        eig_at_xi = estimate_eig(
            params,
            xi_scalar=xi_t,
            prior_thetas=prior_thetas,
            cache=cache,
            t_max=t_max,
            outer_samples=args.eig_outer,
            inner_samples=args.eig_inner,
            rng=np.random.default_rng(args.seed + 5000 * (round_idx + 1)),
        )

        # Observe the truth in fraction space.
        clean_count = float(truth_traj.query(np.array([xi_t]))[0])
        noise_frac = sigma_obs * float(rng.standard_normal())
        y_frac = clean_count / float(N) + noise_frac
        y_count = y_frac * float(N)
        y_observed.append(y_frac)

        # Diagnostic ESS before the resample.
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
            jitter_scale=0.02,
            seed=args.seed + 100 * (round_idx + 1),
        )

        round_dir = artifacts / f"round_{round_idx:02d}"
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
        "posterior_R0_means": [
            float(np.mean(particles_t[:, 0] / particles_t[:, 1]))
            for particles_t in result.per_round_particles
        ],
        "posterior_R0_stds": [
            float(np.std(particles_t[:, 0] / particles_t[:, 1]))
            for particles_t in result.per_round_particles
        ],
        "method": "uniform_grid",
    }
    with open(artifacts / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plotting.save_posterior_shrinkage_plot(
        result.per_round_particles,
        result.theta_true,
        output_path=artifacts / "posterior_shrinkage.png",
    )
    plotting.save_design_on_curve_plot(
        truth_trajectory=result.true_trajectory,
        xi_star=result.xi_star,
        y_observed=result.y_observed,
        output_path=artifacts / "design_on_curve.png",
        t_max=result.t_max,
        N=result.N,
        I0=result.I0,
        seed=args.seed,
    )

    print("Uniform-baseline SIR done.")
    print(f"  summary.json -> {artifacts / 'summary.json'}")


if __name__ == "__main__":
    main()
