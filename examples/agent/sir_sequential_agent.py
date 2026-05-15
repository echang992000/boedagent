"""Driver: sequential BOED on the stochastic SIR via the LFIAX-style loop.

Runs :func:`run_sequential_sir_lfiax` and emits the three plots.

Single-seed usage::

    python examples/agent/sir_sequential_agent.py --rounds 6 --seed 0 \\
        --artifacts artifacts/sir_seq

Multi-seed usage (Fix 3 — ≥3 seeds for workshop submissions)::

    python examples/agent/sir_sequential_agent.py --rounds 6 --seeds 0,1,2 \\
        --artifacts artifacts/sir_seq_fixed
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


def _parse_seeds(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def _aggregate_and_plot(
    base_dir: Path,
    seeds: list[int],
    *,
    num_rounds: int,
    num_particles: int,
    theta_true: tuple[float, float],
) -> None:
    """Read per-seed summary.json files and write ``summary_aggregated.json``
    + a 1×2 seeds-level diagnostic figure to ``base_dir``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_seed_summaries = []
    for s in seeds:
        with open(base_dir / f"seed_{s}" / "summary.json") as f:
            per_seed_summaries.append(json.load(f))

    R0_truth = float(theta_true[0] / theta_true[1])
    R0_final_means = [float(s["posterior_R0_means"][-1]) for s in per_seed_summaries]
    R0_final_stds = [float(s["posterior_R0_stds"][-1]) for s in per_seed_summaries]
    prior_R0_stds = [float(s["posterior_R0_stds"][0]) for s in per_seed_summaries]
    R0_bias = [m - R0_truth for m in R0_final_means]

    eig_per_seed = np.array([s["eig_final"] for s in per_seed_summaries])  # (n_seeds, T)
    R0_stds_per_seed = np.array([s["posterior_R0_stds"] for s in per_seed_summaries])  # (n_seeds, T+1)

    aggregated = {
        "seeds": list(seeds),
        "posterior_R0_mean_per_seed": R0_final_means,
        "posterior_R0_std_per_seed": R0_final_stds,
        "R0_truth": R0_truth,
        "R0_bias_per_seed": R0_bias,
        "R0_posterior_shrinkage": [
            float(np.mean(prior_R0_stds)),
            float(np.mean(R0_final_stds)),
        ],
        "eig_trajectory_mean_per_round": [float(x) for x in eig_per_seed.mean(axis=0)],
        "eig_trajectory_std_per_round": [float(x) for x in eig_per_seed.std(axis=0)],
        "R0_posterior_std_trajectory_mean": [float(x) for x in R0_stds_per_seed.mean(axis=0)],
        "R0_posterior_std_trajectory_std": [float(x) for x in R0_stds_per_seed.std(axis=0)],
    }

    agg_path = base_dir / "summary_aggregated.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n[aggregate] {agg_path}")
    print(
        f"  R0 final  : mean {np.mean(R0_final_means):.3f} (truth {R0_truth:.3f}), "
        f"bias {np.mean(R0_bias):+.3f}, σ {np.mean(R0_final_stds):.3f}"
    )
    print(
        f"  R0 contracts: σ {aggregated['R0_posterior_shrinkage'][0]:.3f} → "
        f"{aggregated['R0_posterior_shrinkage'][1]:.3f}"
    )

    # Seed-variance health check (worklog: STOP if std/mean > 0.5 on R0 σ).
    r0_std_mean = np.mean(R0_final_stds)
    r0_std_std = np.std(R0_final_stds)
    if r0_std_mean > 0 and r0_std_std / r0_std_mean > 0.5:
        print(
            f"[aggregate] WARNING: seed-variance on R₀ posterior σ is high: "
            f"std/mean = {r0_std_std / r0_std_mean:.2f} (> 0.5 threshold)"
        )

    # Seed-level summary figure: (a) box-plot of R₀ posterior σ per round
    # across seeds, (b) per-round EIG mean ± std across seeds.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    ax_box, ax_eig = axes
    rounds_axis = list(range(R0_stds_per_seed.shape[1]))  # 0..T
    ax_box.boxplot(
        [R0_stds_per_seed[:, t] for t in rounds_axis],
        positions=rounds_axis,
        widths=0.5,
    )
    ax_box.set_xlabel("round")
    ax_box.set_ylabel("σ(R₀)")
    ax_box.set_title("(a) R₀ posterior σ per round (across seeds)")
    ax_box.grid(True, alpha=0.3)

    eig_mean = eig_per_seed.mean(axis=0)
    eig_sd = eig_per_seed.std(axis=0)
    rounds_eig = np.arange(1, eig_per_seed.shape[1] + 1)
    ax_eig.plot(rounds_eig, eig_mean, "-o", color="C3", label="mean")
    ax_eig.fill_between(rounds_eig, eig_mean - eig_sd, eig_mean + eig_sd,
                        color="C3", alpha=0.2, label="±1 seed-σ")
    ax_eig.set_xlabel("round")
    ax_eig.set_ylabel("EIG(ξ_t)")
    ax_eig.set_title("(b) per-round EIG (across seeds)")
    ax_eig.grid(True, alpha=0.3)
    ax_eig.legend()

    fig.suptitle(f"Seeds summary ({len(seeds)} seeds)", fontsize=11)
    fig_path = base_dir / "seeds_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate] {fig_path}")


def _plot_boed_vs_uniform(
    *,
    boed_root: Path,
    uniform_root: Path,
    seeds: list[int],
    out_path: Path,
) -> None:
    """Overlay R₀ posterior σ per round for BOED vs uniform-grid baseline.
    Both curves show the mean ± 1 seed-std band. If BOED σ(R₀) at the
    final round exceeds uniform σ(R₀), print a warning (worklog: STOP
    and escalate)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _stack(root: Path, field: str) -> np.ndarray:
        per_seed = []
        for s in seeds:
            with open(root / f"seed_{s}" / "summary.json") as f:
                per_seed.append(json.load(f)[field])
        return np.array(per_seed, dtype=float)

    boed_r0 = _stack(boed_root, "posterior_R0_stds")
    uni_r0 = _stack(uniform_root, "posterior_R0_stds")
    rounds_axis = np.arange(boed_r0.shape[1])

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    for label, arr, color in [("BOED (optimised)", boed_r0, "C3"),
                              ("uniform-grid", uni_r0, "C0")]:
        m = arr.mean(axis=0)
        sd = arr.std(axis=0)
        ax.plot(rounds_axis, m, "-o", color=color, label=label)
        ax.fill_between(rounds_axis, m - sd, m + sd, color=color, alpha=0.2)
    ax.set_xlabel("round")
    ax.set_ylabel("posterior σ(R₀)")
    ax.set_title("R₀ posterior contraction: BOED (optimised) vs. uniform-grid designs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate] {out_path}")

    boed_final = float(boed_r0[:, -1].mean())
    uni_final = float(uni_r0[:, -1].mean())
    print(
        f"[aggregate] final σ(R₀): BOED {boed_final:.3f} vs uniform {uni_final:.3f} "
        f"(Δ = {boed_final - uni_final:+.3f})"
    )
    if boed_final > uni_final:
        print(
            "[aggregate] WARNING: BOED σ(R₀) exceeds uniform σ(R₀) at final round. "
            "This is a scientific finding, not a bug — report it in the paper "
            "rather than tuning it away."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seed list (e.g. '0,1,2'). If set, overrides --seed and aggregates.",
    )
    parser.add_argument("--artifacts", type=str, default="artifacts/sir_seq")
    # Default truth is the middle of iDAD's three reference eval points,
    # [0.1977, 0.1521], [0.3332, 0.1103], [0.7399, 0.0924]
    # (plot_epidemic_posterior.py in desi-ivanova/idad), which sits near
    # the prior mode β≈0.5, γ≈0.1 and gives R0≈3 — a moderate outbreak.
    parser.add_argument("--theta-true", type=float, nargs=2, default=(0.3332, 0.1103),
                        metavar=("BETA", "GAMMA"))
    parser.add_argument("--num-optimization-steps", type=int, default=120)
    parser.add_argument("--num-particles", type=int, default=512)
    parser.add_argument("--eig-outer", type=int, default=96)
    parser.add_argument("--eig-inner", type=int, default=64)
    parser.add_argument(
        "--submission-figure",
        type=str,
        default="",
        help="If set, path stem for a single PDF+PNG combo figure (shrinkage + designs-on-curve), e.g. artifacts/icml26/sir_combo",
    )
    parser.add_argument(
        "--no-monotone",
        dest="monotone",
        action="store_false",
        help="Disable the assay-time monotonicity constraint xi_{t+1} > xi_t. On by default.",
    )
    parser.set_defaults(monotone=True)
    parser.add_argument(
        "--uniform-artifacts",
        type=str,
        default="",
        help="If set (multi-seed only), run the uniform-grid baseline for the same seeds "
             "and write a BOED-vs-uniform comparison figure to <args.artifacts>/boed_vs_uniform.png.",
    )
    parser.add_argument(
        "--lookahead",
        action="store_true",
        help="Enable 1-step lookahead acquisition (Foster 2021 DAD-spirit, "
             "Iollo 2024). Adds λ·E_y[max EIG(ξ_{t+1}|y_t)] to the score.",
    )
    parser.add_argument("--lookahead-weight", type=float, default=0.5)
    parser.add_argument(
        "--jitter-scale",
        type=float,
        default=0.02,
        help="Log-space MCMC jitter applied at each resample (default 0.02). "
             "Set lower to reduce per-round variance inflation when info gain "
             "per round is small.",
    )
    parser.add_argument(
        "--min-design-spacing",
        type=float,
        default=0.0,
        help="Forbid xi_t within Δ of any previous design (default 0 = off). "
             "Use ~5 (5%% of t_max=100) to break design-diversity collapse.",
    )
    parser.add_argument(
        "--first-design-bias-mu",
        type=float,
        default=None,
        help="Round-0-only Gaussian log-prior on xi, mean. Pulls xi_1 toward "
             "the rising limb to break the (β,γ) ridge under monotonicity "
             "(Cook et al. 2008). Suggested: 12.0 for t_max=100.",
    )
    parser.add_argument(
        "--first-design-bias-sigma",
        type=float,
        default=None,
        help="Round-0 Gaussian log-prior std (paired with --first-design-bias-mu). "
             "Suggested: 8.0 for t_max=100.",
    )
    parser.add_argument(
        "--uniform-fixed-schedule",
        type=str,
        default="",
        help="Comma-separated custom xi schedule for the uniform baseline "
             "(must have length == --rounds). E.g. '8,18,28,45,65,85' for a "
             "physics-aware non-adaptive batch (Fix C / DAD-batch baseline).",
    )
    parser.add_argument("--lookahead-grid-points", type=int, default=11)
    parser.add_argument("--lookahead-y-samples", type=int, default=3)
    parser.add_argument("--lookahead-outer", type=int, default=64)
    parser.add_argument("--lookahead-inner", type=int, default=32)
    args = parser.parse_args()

    from examples.cases.sir import plotting
    from examples.cases.sir.sequential_lfiax import run_sequential_sir_lfiax

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed]
    multi_seed = len(seeds) > 1

    # Collect per-seed results so the final representative combo can be
    # annotated with aggregate statistics (reviewer request).
    per_seed_results: dict[int, Any] = {}

    for s in seeds:
        seed_dir = artifacts / f"seed_{s}" if multi_seed else artifacts
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== seed {s} -> {seed_dir} =====")

        result = run_sequential_sir_lfiax(
            theta_true=tuple(args.theta_true),
            num_rounds=args.rounds,
            num_optimization_steps=args.num_optimization_steps,
            num_particles=args.num_particles,
            eig_outer=args.eig_outer,
            eig_inner=args.eig_inner,
            artifacts_dir=str(seed_dir),
            seed=s,
            monotone=args.monotone,
            use_lookahead=args.lookahead,
            lookahead_weight=args.lookahead_weight,
            lookahead_grid_points=args.lookahead_grid_points,
            lookahead_y_samples=args.lookahead_y_samples,
            lookahead_outer=args.lookahead_outer,
            lookahead_inner=args.lookahead_inner,
            jitter_scale=args.jitter_scale,
            min_design_spacing=args.min_design_spacing,
            first_design_bias_mu=args.first_design_bias_mu,
            first_design_bias_sigma=args.first_design_bias_sigma,
        )
        per_seed_results[s] = result

        shrinkage_path = plotting.save_posterior_shrinkage_plot(
            result.per_round_particles,
            result.theta_true,
            output_path=seed_dir / "posterior_shrinkage.png",
        )
        eig_path = plotting.save_eig_per_round_plot(
            result.round_histories,
            result.eig_final,
            result.xi_star,
            output_path=seed_dir / "eig_per_round.png",
        )
        curve_path = plotting.save_design_on_curve_plot(
            truth_trajectory=result.true_trajectory,
            xi_star=result.xi_star,
            y_observed=result.y_observed,
            output_path=seed_dir / "design_on_curve.png",
            t_max=result.t_max,
            N=result.N,
            I0=result.I0,
            seed=s,
        )
        post_pred_path = plotting.save_posterior_predictive_plot(
            per_round_particles=result.per_round_particles,
            xi_star=result.xi_star,
            y_observed=result.y_observed,
            truth_trajectory=result.true_trajectory,
            theta_true=result.theta_true,
            output_path=seed_dir / "posterior_predictive.png",
            t_max=result.t_max,
            N=result.N,
            I0=result.I0,
            seed=s,
        )

        print("SIR sequential BOED done.")
        print(f"  posterior shrinkage plot -> {shrinkage_path}")
        print(f"  per-round EIG plot       -> {eig_path}")
        print(f"  design-on-curve plot     -> {curve_path}")
        print(f"  posterior predictive     -> {post_pred_path}")
        print(f"  summary.json             -> {seed_dir / 'summary.json'}")

        # For single-seed runs the combo figure lives at the user path;
        # for multi-seed, it lives per-seed inside seed_dir and also at
        # the user path for the first seed (representative run).
        if args.submission_figure:
            combo_path_this_seed = (
                args.submission_figure
                if not multi_seed
                else str(seed_dir / "submission_combo")
            )
            combo = plotting.save_submission_combo_figure(
                per_round_particles=result.per_round_particles,
                theta_true=result.theta_true,
                truth_trajectory=result.true_trajectory,
                xi_star=result.xi_star,
                y_observed=result.y_observed,
                output_path=combo_path_this_seed,
                t_max=result.t_max,
                N=result.N,
                I0=result.I0,
                seed=s,
                ess_per_round=getattr(result, "ess_per_round", None),
                num_particles=getattr(result, "num_particles_config", None),
            )
            print(f"  submission combo figure  -> {combo} (+ matching .png)")

    # Compute aggregate statistics across seeds (used both by the
    # representative submission figure annotation and by the published
    # summary_aggregated.json).
    aggregate_stats: dict[str, Any] | None = None
    if multi_seed:
        R0_truth = float(args.theta_true[0] / args.theta_true[1])

        def _final_r0_stats(r: Any) -> tuple[float, float]:
            """Return (R₀ posterior mean, R₀ posterior std) for the
            final round. Prefers r.posterior_R0_means / r.posterior_R0_stds
            (already computed inside run_sequential_sir_lfiax); falls
            back to particles or ratio-of-(β,γ)-arrays if absent."""
            m = getattr(r, "posterior_R0_means", None)
            sd = getattr(r, "posterior_R0_stds", None)
            if m is not None and sd is not None and len(m) > 0 and len(sd) > 0:
                return float(m[-1]), float(sd[-1])
            particles = r.per_round_particles[-1]
            theta = particles.theta if hasattr(particles, "theta") else particles
            r0 = theta[:, 0] / theta[:, 1]
            return float(np.mean(r0)), float(np.std(r0))

        R0_means_by_seed = {}
        R0_stds_by_seed = {}
        for s, r in per_seed_results.items():
            m, sd = _final_r0_stats(r)
            R0_means_by_seed[s] = m
            R0_stds_by_seed[s] = sd

        ess_final_by_seed = {}
        for s, r in per_seed_results.items():
            ess_list = getattr(r, "ess_per_round", None)
            if ess_list is not None and len(ess_list) > 0:
                ess_final_by_seed[s] = float(ess_list[-1])
        R0_means = list(R0_means_by_seed.values())
        R0_stds = list(R0_stds_by_seed.values())
        R0_bias = [abs(m - R0_truth) for m in R0_means]
        bias_over_sigma = [
            (b / sd) if sd > 0 else float("inf")
            for b, sd in zip(R0_bias, R0_stds)
        ]
        aggregate_stats = {
            "n_seeds": len(seeds),
            "R0_truth": R0_truth,
            "R0_mean_across_seeds": float(np.mean(R0_means)),
            "R0_std_across_seeds": float(np.std(R0_means)),
            "bias_over_sigma_range": [
                float(np.min(bias_over_sigma)),
                float(np.max(bias_over_sigma)),
            ],
            "ess_final_range": (
                [float(np.min(list(ess_final_by_seed.values()))),
                 float(np.max(list(ess_final_by_seed.values())))]
                if ess_final_by_seed else None
            ),
        }

        # Pick a "representative" seed: the one whose final R0 mean is
        # closest to the across-seed average — more defensible than
        # always taking seeds[0].
        rep_seed = min(
            seeds,
            key=lambda s: abs(R0_means_by_seed[s] - aggregate_stats["R0_mean_across_seeds"]),
        )
        print(
            f"\n[aggregate] representative seed = {rep_seed} "
            f"(R0={R0_means_by_seed[rep_seed]:.3f} vs across-seed mean "
            f"{aggregate_stats['R0_mean_across_seeds']:.3f})"
        )

        if args.submission_figure:
            rep_result = per_seed_results[rep_seed]
            combo = plotting.save_submission_combo_figure(
                per_round_particles=rep_result.per_round_particles,
                theta_true=rep_result.theta_true,
                truth_trajectory=rep_result.true_trajectory,
                xi_star=rep_result.xi_star,
                y_observed=rep_result.y_observed,
                output_path=args.submission_figure,
                t_max=rep_result.t_max,
                N=rep_result.N,
                I0=rep_result.I0,
                seed=rep_seed,
                ess_per_round=getattr(rep_result, "ess_per_round", None),
                num_particles=getattr(rep_result, "num_particles_config", None),
                aggregate_stats=aggregate_stats,
            )
            print(f"  representative combo     -> {combo}")

        _aggregate_and_plot(artifacts, seeds, num_rounds=args.rounds,
                            num_particles=args.num_particles,
                            theta_true=tuple(args.theta_true))

    # Fix 4: optional uniform-grid baseline over the same seeds.
    if args.uniform_artifacts and multi_seed:
        from examples.cases.sir.sequential_lfiax import run_sequential_sir_lfiax_uniform
        uniform_root = Path(args.uniform_artifacts)
        uniform_root.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            seed_dir = uniform_root / f"seed_{s}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n===== uniform baseline seed {s} -> {seed_dir} =====")
            fixed_schedule = (
                [float(x) for x in args.uniform_fixed_schedule.split(",")]
                if args.uniform_fixed_schedule else None
            )
            run_sequential_sir_lfiax_uniform(
                theta_true=tuple(args.theta_true),
                num_rounds=args.rounds,
                num_optimization_steps=args.num_optimization_steps,
                num_particles=args.num_particles,
                eig_outer=args.eig_outer,
                eig_inner=args.eig_inner,
                artifacts_dir=str(seed_dir),
                seed=s,
                fixed_schedule=fixed_schedule,
            )
        _aggregate_and_plot(uniform_root, seeds, num_rounds=args.rounds,
                            num_particles=args.num_particles,
                            theta_true=tuple(args.theta_true))
        _plot_boed_vs_uniform(
            boed_root=artifacts, uniform_root=uniform_root, seeds=seeds,
            out_path=artifacts / "boed_vs_uniform.png",
        )
        # Three-panel reviewer-oriented comparison (σ shrinkage, EIG,
        # |bias| boxplot). Lives in plotting.save_boed_vs_uniform_comparison.
        R0_truth_agg = float(args.theta_true[0] / args.theta_true[1])

        def _stack_field(root: Path, field: str) -> np.ndarray:
            rows = []
            for s in seeds:
                with open(root / f"seed_{s}" / "summary.json") as f:
                    rows.append(json.load(f)[field])
            return np.asarray(rows, dtype=float)

        try:
            boed_R0_stds = _stack_field(artifacts, "posterior_R0_stds")
            uni_R0_stds = _stack_field(uniform_root, "posterior_R0_stds")
            boed_eig = _stack_field(artifacts, "eig_final")
            uni_eig = _stack_field(uniform_root, "eig_final")
            boed_R0_mean_final = _stack_field(artifacts, "posterior_R0_means")[:, -1]
            uni_R0_mean_final = _stack_field(uniform_root, "posterior_R0_means")[:, -1]
            boed_R0_bias = np.abs(boed_R0_mean_final - R0_truth_agg)
            uni_R0_bias = np.abs(uni_R0_mean_final - R0_truth_agg)
            plotting.save_boed_vs_uniform_comparison(
                boed_R0_stds=boed_R0_stds,
                uniform_R0_stds=uni_R0_stds,
                boed_eig=boed_eig,
                uniform_eig=uni_eig,
                boed_R0_bias=boed_R0_bias,
                uniform_R0_bias=uni_R0_bias,
                output_path=artifacts / "boed_vs_uniform_3panel.png",
            )
        except AttributeError:
            print("[aggregate] note: save_boed_vs_uniform_comparison "
                  "not available in plotting module — skipping 3-panel figure.")


if __name__ == "__main__":
    main()
