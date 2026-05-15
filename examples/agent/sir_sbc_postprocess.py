"""Post-process an SBC run: read ``truth_XXX/round_YY/posterior_samples.npy``
and ``truth_XXX/summary.json`` for each completed truth, compute SBC ranks
for β, γ, R₀, and emit the rank-histogram figure.

Useful when the main SBC driver was interrupted before writing the final
aggregate artefacts — the per-truth particles are durable on disk, so we
can recover the calibration plot without re-running the BOED loops.

Usage::

    python examples/agent/sir_sbc_postprocess.py \
        --sbc-dir artifacts/sir_sbc --num-bins 15
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


def _thin(rank_in_L: int, L: int, num_bins: int, rng: np.random.Generator) -> int:
    if L <= 0:
        return 0
    frac = (rank_in_L + rng.random()) / (L + 1)
    return max(0, min(int(np.floor(frac * num_bins)), num_bins - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbc-dir", type=str, required=True)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from examples.cases.sir import plotting

    sbc_dir = Path(args.sbc_dir)
    rng = np.random.default_rng(args.seed)

    ranks_binned: list[tuple[int, int, int]] = []
    ranks_raw: list[tuple[int, int, int]] = []
    records = []

    for truth_dir in sorted(sbc_dir.glob("truth_*")):
        summary_path = truth_dir / "summary.json"
        if not summary_path.exists():
            continue
        # Find the last-round posterior samples.
        round_dirs = sorted(truth_dir.glob("round_*"))
        if not round_dirs:
            continue
        last_round = round_dirs[-1]
        particles_path = last_round / "posterior_samples.npy"
        if not particles_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        theta_star = np.asarray(summary["theta_true"], dtype=float)
        particles = np.load(particles_path)
        if particles.ndim != 2 or particles.shape[1] != 2 or particles.shape[0] == 0:
            continue
        beta_s = particles[:, 0]
        gamma_s = particles[:, 1]
        r0_s = beta_s / gamma_s
        L = int(particles.shape[0])
        rank_beta = int((beta_s < theta_star[0]).sum())
        rank_gamma = int((gamma_s < theta_star[1]).sum())
        rank_r0 = int((r0_s < (theta_star[0] / theta_star[1])).sum())
        rb = _thin(rank_beta, L, args.num_bins, rng)
        rg = _thin(rank_gamma, L, args.num_bins, rng)
        rr = _thin(rank_r0, L, args.num_bins, rng)
        ranks_raw.append((rank_beta, rank_gamma, rank_r0))
        ranks_binned.append((rb, rg, rr))
        records.append({
            "truth_dir": truth_dir.name,
            "beta_star": float(theta_star[0]),
            "gamma_star": float(theta_star[1]),
            "R0_star": float(theta_star[0] / theta_star[1]),
            "L": L,
            "rank_raw": [rank_beta, rank_gamma, rank_r0],
            "rank_binned": [rb, rg, rr],
        })

    if not ranks_binned:
        print("[sbc-post] no completed truths found — aborting.")
        return

    ranks_arr = np.asarray(ranks_binned, dtype=int)
    print(f"[sbc-post] reconstructed ranks for {len(ranks_binned)} truths "
          f"(L per truth range: {min(r['L'] for r in records)}-{max(r['L'] for r in records)})")

    with open(sbc_dir / "sbc_ranks.json", "w") as f:
        json.dump({
            "num_truths_completed": len(ranks_binned),
            "num_bins": args.num_bins,
            "per_truth": records,
        }, f, indent=2)

    out = plotting.save_sbc_rank_histogram(
        ranks=ranks_arr,
        num_bins=args.num_bins,
        output_path=sbc_dir / "sbc_rank_histograms.png",
    )
    print(f"[sbc-post] figure -> {out}")
    print(f"[sbc-post] ranks  -> {sbc_dir / 'sbc_ranks.json'}")


if __name__ == "__main__":
    main()
