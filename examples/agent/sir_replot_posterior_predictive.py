"""Regenerate the SIR posterior-predictive diagnostic from saved artifacts.

Reads ``<artifacts_dir>/summary.json`` and
``<artifacts_dir>/round_{NN}/posterior_samples.npy`` so we can rebuild
the posterior-predictive plot without re-running the BOED loop.

Usage::

    python examples/agent/sir_replot_posterior_predictive.py \\
        --artifacts artifacts/sir_seq_v4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=str, default="artifacts/sir_seq_v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-draws", type=int, default=80)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    if not artifacts.exists():
        raise SystemExit(f"artifacts dir not found: {artifacts}")

    from examples.cases.sir import plotting
    from examples.cases.sir.simulator import (
        DEFAULT_I0,
        DEFAULT_N,
        DEFAULT_T_MAX,
        simulate_sir_trajectory,
    )

    summary = json.loads((artifacts / "summary.json").read_text())
    theta_true = np.asarray(summary["theta_true"], dtype=float)
    xi_star = list(map(float, summary["xi_star"]))
    y_observed = list(map(float, summary["y_observed"]))

    # Collect per-round particles in order.
    round_dirs = sorted(d for d in artifacts.iterdir() if d.is_dir() and d.name.startswith("round_"))
    per_round_particles = [np.load(d / "posterior_samples.npy") for d in round_dirs]
    print(f"loaded {len(per_round_particles)} rounds from {artifacts}")

    # Rebuild the same truth trajectory the loop used: rejection-sample
    # seed until peak I >= max(10*I0, 20). This matches the rule in
    # run_sequential_sir_lfiax.
    truth_seed = args.seed + 1
    for _ in range(64):
        candidate = simulate_sir_trajectory(theta_true, seed=truth_seed)
        if float(candidate.I.max()) >= max(10.0 * DEFAULT_I0, 20.0):
            truth_traj = candidate
            break
        truth_seed += 1
    else:
        truth_traj = candidate
    peak_I = float(truth_traj.I.max())
    peak_t = float(truth_traj.times[int(np.argmax(truth_traj.I))])
    print(f"truth: peak I = {peak_I:.0f} at t = {peak_t:.1f} (seed={truth_seed})")

    out = plotting.save_posterior_predictive_plot(
        per_round_particles=per_round_particles,
        xi_star=xi_star,
        y_observed=y_observed,
        truth_trajectory=truth_traj,
        theta_true=theta_true,
        output_path=artifacts / "posterior_predictive.png",
        t_max=DEFAULT_T_MAX,
        N=DEFAULT_N,
        I0=DEFAULT_I0,
        num_draws=args.num_draws,
        seed=args.seed,
    )
    print(f"posterior predictive -> {out}")


if __name__ == "__main__":
    main()
