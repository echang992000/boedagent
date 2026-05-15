"""Driver: sequential BOED on linear regression.

Runs the numpy conjugate path end-to-end (which matches what Pyro's
``marginal_eig`` + ``AutoDiagonalNormal`` would compute for this model)
and emits the three plots.

Usage::

    python examples/agent/linreg_sequential_agent.py --rounds 6 --seed 0 \\
        --artifacts artifacts/linreg_seq
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--artifacts", type=str, default="artifacts/linreg_seq")
    parser.add_argument("--theta-true", type=float, nargs=2, default=(1.5, -0.2),
                        metavar=("A", "B"))
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--sigma-obs", type=float, default=0.1)
    parser.add_argument("--num-posterior-samples", type=int, default=2000)
    args = parser.parse_args()

    from examples.cases.linreg_sequential import plotting
    from examples.cases.linreg_sequential.sequential_pyro import run_sequential_linreg_pyro

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    result = run_sequential_linreg_pyro(
        theta_true=tuple(args.theta_true),
        num_rounds=args.rounds,
        x_max=args.x_max,
        sigma_obs=args.sigma_obs,
        num_posterior_samples=args.num_posterior_samples,
        artifacts_dir=str(artifacts),
        seed=args.seed,
    )

    shrinkage_path = plotting.save_posterior_shrinkage_plot(
        result.per_round_posterior_samples,
        np.asarray(args.theta_true),
        output_path=artifacts / "posterior_shrinkage.png",
    )
    eig_path = plotting.save_eig_per_round_plot(
        result.round_histories,
        result.eig_final,
        result.xi_star,
        output_path=artifacts / "eig_per_round.png",
    )
    overlay_path = plotting.save_fit_overlay_plot(
        theta_true=np.asarray(args.theta_true),
        xi_star=result.xi_star,
        y_observed=result.y_observed,
        per_round_posterior_samples=result.per_round_posterior_samples,
        x_range=(-args.x_max, args.x_max),
        output_path=artifacts / "linreg_fit_overlay.png",
    )

    print("Linreg sequential BOED done.")
    print(f"  posterior shrinkage plot -> {shrinkage_path}")
    print(f"  per-round EIG plot       -> {eig_path}")
    print(f"  linreg fit overlay plot  -> {overlay_path}")
    print(f"  summary.json             -> {artifacts / 'summary.json'}")


if __name__ == "__main__":
    main()
