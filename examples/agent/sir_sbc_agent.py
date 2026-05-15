"""Simulation-based calibration (SBC; Talts et al. 2018) for sequential BOED.

For each of ``--num-truths`` truths drawn from the prior, simulate the
full sequential BOED loop and record the final posterior cloud. The
rank of each truth within its own posterior samples (across β, γ, R₀)
should be Uniform(0, L) under a well-calibrated inference scheme.

The resulting rank histograms are emitted via
``plotting.save_sbc_rank_histogram`` with a Binomial(num_truths, 1/B)
99% envelope, so reviewers can read off: (a) uniformity ≈ calibrated,
(b) peaked middle ≈ overconfidence, (c) U-shape ≈ underconfidence,
(d) slope ≈ bias.

Usage::

    python examples/agent/sir_sbc_agent.py \
        --num-truths 50 --rounds 4 --num-bins 20 \
        --artifacts artifacts/sir_sbc

Tuning for wall-clock: each truth is an independent sequential run.
For 50 truths at rounds=4, num_optimization_steps=60 this fits in a
single workstation overnight; reviewers typically accept 30–100 truths
for SBC diagnostic plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _compute_ranks(
    truth_beta: float,
    truth_gamma: float,
    posterior_theta: np.ndarray,
) -> tuple[int, int, int]:
    """Rank of θ* among L posterior samples for β, γ, R₀.

    Following Talts et al. 2018: rank = #{θ_ℓ < θ*}. A well-calibrated
    posterior has ranks ~ Uniform(0, L).
    """
    beta_samples = posterior_theta[:, 0]
    gamma_samples = posterior_theta[:, 1]
    r0_samples = beta_samples / gamma_samples
    truth_r0 = truth_beta / truth_gamma
    rank_beta = int((beta_samples < truth_beta).sum())
    rank_gamma = int((gamma_samples < truth_gamma).sum())
    rank_r0 = int((r0_samples < truth_r0).sum())
    return rank_beta, rank_gamma, rank_r0


def _thin_to_num_bins(
    rank_in_L: int, L: int, num_bins: int, rng: np.random.Generator
) -> int:
    """Project rank ∈ [0, L] onto [0, num_bins) via a uniform draw.

    Talts et al. §4: when num_bins < L, we subsample with ties broken
    randomly so the target distribution remains Uniform(0, num_bins).
    """
    if L <= 0:
        return 0
    # Map rank to fraction in [0, 1] then bin-index in [0, num_bins).
    # Add a half-cell jitter to avoid pile-up at integer fractions.
    frac = (rank_in_L + rng.random()) / (L + 1)
    idx = int(np.floor(frac * num_bins))
    return max(0, min(idx, num_bins - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-truths", type=int, default=50,
                        help="Number of θ* drawn from the prior.")
    parser.add_argument("--rounds", type=int, default=4,
                        help="Sequential BOED rounds per truth.")
    parser.add_argument("--num-bins", type=int, default=20,
                        help="SBC rank-histogram bins (Talts et al. recommend ≥20).")
    parser.add_argument("--num-particles", type=int, default=256)
    parser.add_argument("--num-optimization-steps", type=int, default=60)
    parser.add_argument("--eig-outer", type=int, default=64)
    parser.add_argument("--eig-inner", type=int, default=48)
    parser.add_argument("--artifacts", type=str, default="artifacts/sir_sbc")
    parser.add_argument("--base-seed", type=int, default=10_000,
                        help="Seeds per truth are base_seed + i.")
    parser.add_argument("--monotone", dest="monotone", action="store_true",
                        help="Pass the monotonicity constraint to the BOED loop.")
    parser.add_argument("--no-monotone", dest="monotone", action="store_false")
    parser.set_defaults(monotone=False)
    args = parser.parse_args()

    from examples.cases.sir import plotting
    from examples.cases.sir.prior import sample_prior
    from examples.cases.sir.sequential_lfiax import run_sequential_sir_lfiax

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Draw truths from the prior (the prior is truncated on R0 ∈ [0.5, 20]
    # so no degenerate truths sneak into the SBC budget).
    prior_seed = args.base_seed - 1
    truths = sample_prior(args.num_truths, seed=prior_seed)
    print(f"[sbc] drew {args.num_truths} truths from the prior "
          f"(R0 range: [{(truths[:,0]/truths[:,1]).min():.2f}, "
          f"{(truths[:,0]/truths[:,1]).max():.2f}])")

    ranks_raw: list[tuple[int, int, int]] = []
    ranks_binned: list[tuple[int, int, int]] = []
    L_per_truth: list[int] = []

    per_truth_records: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.base_seed)

    for i, theta_star in enumerate(truths):
        seed_i = args.base_seed + i
        run_dir = artifacts / f"truth_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[sbc] truth {i+1}/{args.num_truths}  "
              f"β*={theta_star[0]:.3f} γ*={theta_star[1]:.3f} "
              f"R0*={theta_star[0]/theta_star[1]:.2f}  seed={seed_i}")

        try:
            result = run_sequential_sir_lfiax(
                theta_true=tuple(theta_star),
                num_rounds=args.rounds,
                num_optimization_steps=args.num_optimization_steps,
                num_particles=args.num_particles,
                eig_outer=args.eig_outer,
                eig_inner=args.eig_inner,
                artifacts_dir=str(run_dir),
                seed=seed_i,
                monotone=args.monotone,
            )
        except Exception as e:
            print(f"[sbc] truth {i} raised {type(e).__name__}: {e} — skipping")
            continue

        # Final posterior particles: (L, 2)
        final = result.per_round_particles[-1]
        theta_samples = final.theta if hasattr(final, "theta") else np.asarray(final)
        L = int(theta_samples.shape[0])
        rank_raw = _compute_ranks(float(theta_star[0]), float(theta_star[1]), theta_samples)
        rank_binned = tuple(
            _thin_to_num_bins(r, L, args.num_bins, rng) for r in rank_raw
        )
        ranks_raw.append(rank_raw)
        ranks_binned.append(rank_binned)
        L_per_truth.append(L)
        per_truth_records.append({
            "truth_index": i,
            "seed": seed_i,
            "beta_star": float(theta_star[0]),
            "gamma_star": float(theta_star[1]),
            "R0_star": float(theta_star[0] / theta_star[1]),
            "L": L,
            "rank_raw": list(rank_raw),
            "rank_binned": list(rank_binned),
            "posterior_R0_mean": float(theta_samples[:, 0].mean() / theta_samples[:, 1].mean())
                                if theta_samples.shape[0] else None,
        })

    if not ranks_binned:
        print("[sbc] no truths completed successfully — aborting.")
        return

    ranks_arr = np.asarray(ranks_binned, dtype=int)
    print(f"\n[sbc] collected {ranks_arr.shape[0]} rank rows")

    # Persist the raw record so downstream analysis can re-bin if needed.
    with open(artifacts / "sbc_ranks.json", "w") as f:
        json.dump({
            "num_truths_requested": args.num_truths,
            "num_truths_completed": int(ranks_arr.shape[0]),
            "num_bins": args.num_bins,
            "L_per_truth": L_per_truth,
            "per_truth": per_truth_records,
        }, f, indent=2)

    out_path = plotting.save_sbc_rank_histogram(
        ranks=ranks_arr,
        num_bins=args.num_bins,
        output_path=artifacts / "sbc_rank_histograms.png",
    )
    print(f"[sbc] rank-histogram figure -> {out_path}")
    print(f"[sbc] ranks JSON            -> {artifacts / 'sbc_ranks.json'}")


if __name__ == "__main__":
    main()
