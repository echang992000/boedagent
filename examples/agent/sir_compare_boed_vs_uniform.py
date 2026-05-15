"""Aggregate BOED vs uniform-baseline runs and print a comparison table
plus save a side-by-side figure.

Reads summary.json from artifacts/sir_seq_boed_s{0,1,2} and
artifacts/sir_seq_uniform_s{0,1,2} and produces:

  - artifacts/compare_boed_vs_uniform/comparison.json
  - artifacts/compare_boed_vs_uniform/comparison_plot.png
  - prints a per-round table to stdout
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SEEDS = [0, 1, 2]
THETA_TRUE = (0.3332, 0.1103)
R0_TRUE = THETA_TRUE[0] / THETA_TRUE[1]  # ≈ 3.02


def load_runs(stem: str):
    runs = []
    for s in SEEDS:
        path = REPO_ROOT / "artifacts" / f"{stem}_s{s}" / "summary.json"
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def stack(runs, key):
    """Stack a per-round scalar/vector field across seeds -> array[seed, round(, dim)]."""
    return np.array([r[key] for r in runs], dtype=float)


def main() -> None:
    boed = load_runs("sir_seq_boed")
    uni = load_runs("sir_seq_uniform")

    # Fields:
    #   posterior_means[round][β|γ]  has num_rounds + 1 entries (round 0 = prior)
    #   posterior_stds[round][β|γ]
    #   posterior_R0_means[round]
    #   posterior_R0_stds[round]
    #   eig_final[round]              has num_rounds entries
    #   xi_star[round]
    n_rounds = len(boed[0]["eig_final"])

    boed_means = stack(boed, "posterior_means")       # [3, T+1, 2]
    boed_stds = stack(boed, "posterior_stds")         # [3, T+1, 2]
    boed_R0m = stack(boed, "posterior_R0_means")      # [3, T+1]
    boed_R0s = stack(boed, "posterior_R0_stds")       # [3, T+1]
    boed_eig = stack(boed, "eig_final")               # [3, T]
    boed_xi = stack(boed, "xi_star")                  # [3, T]

    uni_means = stack(uni, "posterior_means")
    uni_stds = stack(uni, "posterior_stds")
    uni_R0m = stack(uni, "posterior_R0_means")
    uni_R0s = stack(uni, "posterior_R0_stds")
    uni_eig = stack(uni, "eig_final")
    uni_xi = stack(uni, "xi_star")

    # --- Print comparison table -------------------------------------------
    print(f"\nθ* = (β*={THETA_TRUE[0]:.4f}, γ*={THETA_TRUE[1]:.4f}, R₀*={R0_TRUE:.3f})")
    print(f"Seeds: {SEEDS}\n")

    header = (
        f"{'rd':>2} | "
        f"{'β BOED (mean±sd)':>22} | {'β Uni  (mean±sd)':>22} | "
        f"{'γ BOED (mean±sd)':>22} | {'γ Uni  (mean±sd)':>22} | "
        f"{'R₀ BOED':>14} | {'R₀ Uni':>14} | "
        f"{'EIG BOED':>10} | {'EIG Uni':>10}"
    )
    print(header)
    print("-" * len(header))
    for t in range(n_rounds + 1):
        beta_b_m = boed_means[:, t, 0].mean()
        beta_b_s = boed_means[:, t, 0].std()
        beta_u_m = uni_means[:, t, 0].mean()
        beta_u_s = uni_means[:, t, 0].std()
        gam_b_m = boed_means[:, t, 1].mean()
        gam_b_s = boed_means[:, t, 1].std()
        gam_u_m = uni_means[:, t, 1].mean()
        gam_u_s = uni_means[:, t, 1].std()
        R0_b_m = boed_R0m[:, t].mean()
        R0_b_s = boed_R0m[:, t].std()
        R0_u_m = uni_R0m[:, t].mean()
        R0_u_s = uni_R0m[:, t].std()
        if t == 0:
            eig_cells = f"{'prior':>10} | {'prior':>10}"
        else:
            eig_cells = f"{boed_eig[:, t-1].mean():>10.3f} | {uni_eig[:, t-1].mean():>10.3f}"
        print(
            f"{t:>2} | "
            f"{beta_b_m:>8.3f} ± {beta_b_s:<5.3f}    | "
            f"{beta_u_m:>8.3f} ± {beta_u_s:<5.3f}    | "
            f"{gam_b_m:>8.3f} ± {gam_b_s:<5.3f}    | "
            f"{gam_u_m:>8.3f} ± {gam_u_s:<5.3f}    | "
            f"{R0_b_m:>5.2f} ± {R0_b_s:<5.2f} | "
            f"{R0_u_m:>5.2f} ± {R0_u_s:<5.2f} | "
            f"{eig_cells}"
        )

    # Per-round posterior std (shrinkage diagnostic)
    print("\nPosterior std (across seeds) — lower is tighter")
    print(f"{'rd':>2} | {'sd(β) BOED':>12} | {'sd(β) Uni':>12} | {'sd(γ) BOED':>12} | {'sd(γ) Uni':>12}")
    for t in range(n_rounds + 1):
        print(
            f"{t:>2} | "
            f"{boed_stds[:, t, 0].mean():>12.4f} | {uni_stds[:, t, 0].mean():>12.4f} | "
            f"{boed_stds[:, t, 1].mean():>12.4f} | {uni_stds[:, t, 1].mean():>12.4f}"
        )

    # ξ sequences
    print("\nξ_t sequences (mean across seeds)")
    print("  BOED   :", [f"{v:.2f}" for v in boed_xi.mean(axis=0)])
    print("  Uniform:", [f"{v:.2f}" for v in uni_xi.mean(axis=0)])
    print("  BOED per-seed:")
    for i, s in enumerate(SEEDS):
        print(f"    seed {s}: ", [f"{v:.2f}" for v in boed_xi[i]])

    # Bias to truth in (β, γ) for the final round
    print("\nFinal-round bias / 'σ-away-from-truth'")
    for label, m, sd in [("BOED", boed_means[:, -1], boed_stds[:, -1]),
                         ("Uni ", uni_means[:, -1], uni_stds[:, -1])]:
        mean_b = m[:, 0].mean()
        mean_g = m[:, 1].mean()
        sd_b = sd[:, 0].mean()
        sd_g = sd[:, 1].mean()
        print(
            f"  {label}: β = {mean_b:.3f} (truth {THETA_TRUE[0]:.3f},  "
            f"{(mean_b - THETA_TRUE[0]) / max(sd_b, 1e-9):+.2f}σ); "
            f"γ = {mean_g:.3f} (truth {THETA_TRUE[1]:.3f},  "
            f"{(mean_g - THETA_TRUE[1]) / max(sd_g, 1e-9):+.2f}σ)"
        )

    # --- Save comparison JSON ---------------------------------------------
    out_dir = REPO_ROOT / "artifacts" / "compare_boed_vs_uniform"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "seeds": SEEDS,
        "theta_true": list(THETA_TRUE),
        "R0_true": R0_TRUE,
        "boed": {
            "xi_star": boed_xi.tolist(),
            "eig_final": boed_eig.tolist(),
            "posterior_beta_mean": boed_means[:, :, 0].tolist(),
            "posterior_beta_std": boed_stds[:, :, 0].tolist(),
            "posterior_gamma_mean": boed_means[:, :, 1].tolist(),
            "posterior_gamma_std": boed_stds[:, :, 1].tolist(),
            "posterior_R0_mean": boed_R0m.tolist(),
            "posterior_R0_std": boed_R0s.tolist(),
        },
        "uniform": {
            "xi_star": uni_xi.tolist(),
            "eig_final": uni_eig.tolist(),
            "posterior_beta_mean": uni_means[:, :, 0].tolist(),
            "posterior_beta_std": uni_stds[:, :, 0].tolist(),
            "posterior_gamma_mean": uni_means[:, :, 1].tolist(),
            "posterior_gamma_std": uni_stds[:, :, 1].tolist(),
            "posterior_R0_mean": uni_R0m.tolist(),
            "posterior_R0_std": uni_R0s.tolist(),
        },
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  wrote {out_dir / 'comparison.json'}")

    # --- Comparison figure ------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rounds_axis = np.arange(n_rounds + 1)

    for ax, (title, b_mean, b_std, u_mean, u_std, truth) in zip(
        axes[0],
        [
            ("β posterior", boed_means[:, :, 0], boed_stds[:, :, 0],
             uni_means[:, :, 0], uni_stds[:, :, 0], THETA_TRUE[0]),
            ("γ posterior", boed_means[:, :, 1], boed_stds[:, :, 1],
             uni_means[:, :, 1], uni_stds[:, :, 1], THETA_TRUE[1]),
            ("R₀ posterior", boed_R0m, boed_R0s, uni_R0m, uni_R0s, R0_TRUE),
        ],
    ):
        bm = b_mean.mean(axis=0)
        bs = b_std.mean(axis=0)
        um = u_mean.mean(axis=0)
        us = u_std.mean(axis=0)
        ax.plot(rounds_axis, bm, "-o", color="C3", label="BOED mean")
        ax.fill_between(rounds_axis, bm - bs, bm + bs, color="C3", alpha=0.2,
                        label="BOED ±1σ")
        ax.plot(rounds_axis, um, "-s", color="C0", label="Uniform mean")
        ax.fill_between(rounds_axis, um - us, um + us, color="C0", alpha=0.2,
                        label="Uniform ±1σ")
        ax.axhline(truth, color="orange", ls="--", lw=1.5, label="truth")
        ax.set_xlabel("round")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Per-round EIG
    ax = axes[1, 0]
    rounds_eig = np.arange(1, n_rounds + 1)
    ax.plot(rounds_eig, boed_eig.mean(axis=0), "-o", color="C3", label="BOED")
    ax.fill_between(rounds_eig,
                    boed_eig.mean(axis=0) - boed_eig.std(axis=0),
                    boed_eig.mean(axis=0) + boed_eig.std(axis=0),
                    color="C3", alpha=0.2)
    ax.plot(rounds_eig, uni_eig.mean(axis=0), "-s", color="C0", label="Uniform")
    ax.fill_between(rounds_eig,
                    uni_eig.mean(axis=0) - uni_eig.std(axis=0),
                    uni_eig.mean(axis=0) + uni_eig.std(axis=0),
                    color="C0", alpha=0.2)
    ax.set_xlabel("round")
    ax.set_ylabel("EIG(ξ_t)")
    ax.set_title("per-round EIG at chosen ξ")
    ax.grid(alpha=0.3)
    ax.legend()

    # ξ_t sequences overlaid on outbreak peak time reference
    ax = axes[1, 1]
    for i, s in enumerate(SEEDS):
        ax.plot(rounds_eig, boed_xi[i], "-o", color="C3", alpha=0.6,
                label=f"BOED seed {s}" if i == 0 else None)
        ax.plot(rounds_eig, uni_xi[i], "-s", color="C0", alpha=0.6,
                label=f"Uniform seed {s}" if i == 0 else None)
    ax.axhspan(20, 26, color="orange", alpha=0.15, label="true peak window")
    ax.set_xlabel("round")
    ax.set_ylabel("ξ_t")
    ax.set_title("chosen designs ξ_t per round")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Cumulative EIG
    ax = axes[1, 2]
    ax.plot(rounds_eig, np.cumsum(boed_eig.mean(axis=0)), "-o", color="C3", label="BOED")
    ax.plot(rounds_eig, np.cumsum(uni_eig.mean(axis=0)), "-s", color="C0", label="Uniform")
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative EIG")
    ax.set_title("cumulative EIG (surrogate estimate)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"Sequential BOED vs uniform-grid baseline — stochastic SIR (seeds {SEEDS})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_plot.png", dpi=120)
    print(f"  wrote {out_dir / 'comparison_plot.png'}")


if __name__ == "__main__":
    main()
