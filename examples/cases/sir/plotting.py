"""Plots for the sequential SIR / LFIAX run.

Three figures, matching PLAN.md § 5:

* ``posterior_shrinkage.png`` — per-parameter KDE per round, colormap
  viridis, truth marked.
* ``eig_per_round.png`` — final EIG per round on top, inner
  optimisation traces on the bottom.
* ``design_on_curve.png`` — reproduces the reference layout: true
  infected curve (orange), prior predictive band (blue / grey), and
  vertical rules at xi_1, xi_2, ..., xi_T.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import json
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .simulator import (
    DEFAULT_I0,
    DEFAULT_N,
    DEFAULT_T_MAX,
    simulate_sir_trajectory,
)
from .prior import sample_prior


def _kde_1d(samples: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    z = (grid[:, None] - samples[None, :]) / bandwidth
    k = np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    return k.mean(axis=1) / bandwidth


def save_posterior_shrinkage_plot(
    per_round_particles: List[np.ndarray],
    theta_true: np.ndarray,
    *,
    output_path: str | Path,
    param_names=("beta", "gamma"),
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    cmap = plt.get_cmap("viridis")
    n_rounds = len(per_round_particles)

    for k, (ax, name) in enumerate(zip(axes[:2], param_names)):
        # Set a common grid per parameter based on prior range + truth.
        all_vals = np.concatenate([p[:, k] for p in per_round_particles])
        lo = float(np.quantile(all_vals, 0.01))
        hi = float(np.quantile(all_vals, 0.99))
        lo = min(lo, float(theta_true[k]) * 0.7)
        hi = max(hi, float(theta_true[k]) * 1.3)
        grid = np.linspace(lo, hi, 400)
        for t, particles in enumerate(per_round_particles):
            samples = particles[:, k]
            bw = max(np.std(samples) * 1.06 * samples.shape[0] ** (-1 / 5), 1e-4)
            density = _kde_1d(samples, grid, bandwidth=bw)
            frac = t / max(n_rounds - 1, 1)
            color = cmap(frac)
            label = f"round {t}" if t in {0, n_rounds - 1} else None
            ax.plot(grid, density, color=color, linewidth=2.0 if label else 1.0, label=label, alpha=0.9)
        ax.axvline(float(theta_true[k]), color="orange", linestyle="--", linewidth=1.5, label="truth")
        ax.set_xlabel(name)
        ax.set_ylabel("posterior density")
        ax.set_title(f"Posterior over {name} across rounds")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Third panel: R₀ = β / γ. This is the identifiable combination
    # for early-outbreak SIR (Roosa & Chowell 2019; Wearing 2005) and
    # is typically much better constrained than β or γ alone.
    ax_r = axes[2]
    R0_per_round = [p[:, 0] / np.clip(p[:, 1], 1e-6, None) for p in per_round_particles]
    all_R0 = np.concatenate(R0_per_round)
    r0_true = float(theta_true[0]) / float(theta_true[1])
    lo_r = min(float(np.quantile(all_R0, 0.01)), r0_true * 0.7)
    hi_r = max(float(np.quantile(all_R0, 0.99)), r0_true * 1.3)
    grid_r = np.linspace(lo_r, hi_r, 400)
    for t, samples in enumerate(R0_per_round):
        bw = max(np.std(samples) * 1.06 * samples.shape[0] ** (-1 / 5), 1e-4)
        density = _kde_1d(samples, grid_r, bandwidth=bw)
        frac = t / max(n_rounds - 1, 1)
        color = cmap(frac)
        label = f"round {t}" if t in {0, n_rounds - 1} else None
        ax_r.plot(grid_r, density, color=color, linewidth=2.0 if label else 1.0, label=label, alpha=0.9)
    ax_r.axvline(r0_true, color="orange", linestyle="--", linewidth=1.5, label="truth")
    ax_r.set_xlabel("R0 = beta / gamma")
    ax_r.set_ylabel("posterior density")
    ax_r.set_title("Posterior over R0 across rounds")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="best", fontsize=8)

    fig.suptitle("SIR sequential BOED — posterior shrinkage", y=1.02)
    fig.tight_layout()
    return _save(fig, output_path)


def save_eig_per_round_plot(
    round_histories: List[dict],
    eig_final: List[float],
    xi_star: List[float],
    *,
    output_path: str | Path,
) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5))
    rounds = np.arange(len(eig_final))
    axes[0].plot(rounds, eig_final, marker="o", color="#ae2012", linewidth=2.0)
    for r, (xi_t, e) in enumerate(zip(xi_star, eig_final)):
        axes[0].annotate(f"xi={xi_t:.1f}", (r, e), xytext=(0, 6), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("final EIG")
    axes[0].set_title("Per-round EIG at the chosen xi*")
    axes[0].grid(True, alpha=0.3)

    cmap = plt.get_cmap("viridis")
    for t, hist in enumerate(round_histories):
        frac = t / max(len(round_histories) - 1, 1)
        axes[1].plot(hist["eig_history"], color=cmap(frac), alpha=0.7, linewidth=1.2, label=f"round {t}" if t in {0, len(round_histories) - 1} else None)
    axes[1].set_xlabel("inner step")
    axes[1].set_ylabel("running EIG estimate")
    axes[1].set_title("Inner optimisation trajectories")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _save(fig, output_path)


def save_design_on_curve_plot(
    *,
    truth_trajectory,
    xi_star: List[float],
    y_observed: List[float],
    output_path: str | Path,
    t_max: float = DEFAULT_T_MAX,
    num_prior_draws: int = 300,
    N: int = DEFAULT_N,
    I0: int = DEFAULT_I0,
    seed: int = 0,
) -> str:
    """Reproduces the reference figure layout with xi_t vertical rules."""
    grid = np.linspace(0.0, t_max, 400)
    # Prior predictive band.
    thetas = sample_prior(num_prior_draws, seed=seed)
    rng = np.random.default_rng(seed + 1)
    I_grid = np.zeros((num_prior_draws, grid.shape[0]))
    for i in range(num_prior_draws):
        traj = simulate_sir_trajectory(thetas[i], N=N, I0=I0, t_max=t_max, seed=int(rng.integers(0, 2**31 - 1)))
        I_grid[i] = traj.query(grid)
    mean = I_grid.mean(axis=0)
    lo = np.quantile(I_grid, 0.05, axis=0)
    hi = np.quantile(I_grid, 0.95, axis=0)

    # True curve on the same grid.
    true_I = truth_trajectory.query(grid)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(grid, lo, hi, color="lightgray", alpha=0.8, label="prior predictive 90%")
    ax.plot(grid, mean, color="#1f77b4", linewidth=2.0, label="prior predictive mean")
    ax.plot(grid, true_I, color="orange", linewidth=2.0, label="true SIR trajectory")

    cmap = plt.get_cmap("viridis")
    for t, (xi_t, y_t) in enumerate(zip(xi_star, y_observed)):
        frac = t / max(len(xi_star) - 1, 1)
        color = cmap(frac)
        ax.axvline(float(xi_t), color=color, alpha=0.55, linewidth=1.2)
        ax.scatter([xi_t], [y_t], color=color, s=45, zorder=5, edgecolors="black", linewidths=0.5,
                   label=f"xi_{t+1}={xi_t:.1f}")
    ax.set_xlim(0.0, t_max)
    ax.set_xlabel("Measurement Time")
    ax.set_ylabel("Number Infected")
    ax.set_title("Sequential designs on the SIR process")
    ax.grid(True, alpha=0.3)
    # Many xi handles — stack outside.
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, output_path)


def save_posterior_predictive_plot(
    *,
    per_round_particles: List[np.ndarray],
    xi_star: List[float],
    y_observed: List[float],
    truth_trajectory,
    theta_true: np.ndarray,
    output_path: str | Path,
    t_max: float = DEFAULT_T_MAX,
    N: int = DEFAULT_N,
    I0: int = DEFAULT_I0,
    num_draws: int = 80,
    seed: int = 0,
) -> str:
    """Posterior predictive diagnostic.

    Top row: prior-predictive vs final-posterior-predictive trajectories
    overlaid with the observed (xi_t, y_t) scatter and the true outbreak.
    Bottom row: posterior-predictive 90% band per round — lets the user
    see whether the band tightens around the observations (a good Bayes
    answer) or whether it misses them (surrogate / misspecification).
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, t_max, 300)
    true_I = truth_trajectory.query(grid)

    prior_thetas = per_round_particles[0]
    post_thetas = per_round_particles[-1]
    n_rounds = len(per_round_particles)

    def simulate_band(thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k = min(num_draws, thetas.shape[0])
        idx = rng.choice(thetas.shape[0], size=k, replace=False)
        picked = thetas[idx]
        curves = np.zeros((k, grid.shape[0]))
        for i in range(k):
            s = int(rng.integers(0, 2**31 - 1))
            traj = simulate_sir_trajectory(picked[i], N=N, I0=I0, t_max=t_max, seed=s)
            curves[i] = traj.query(grid)
        mean = curves.mean(axis=0)
        lo = np.quantile(curves, 0.05, axis=0)
        hi = np.quantile(curves, 0.95, axis=0)
        return curves, mean, lo, hi

    _, prior_mean, prior_lo, prior_hi = simulate_band(prior_thetas)
    post_curves, post_mean, post_lo, post_hi = simulate_band(post_thetas)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1.2, 1.0]})

    ax = axes[0]
    ax.fill_between(grid, prior_lo, prior_hi, color="lightgray", alpha=0.55, label="prior pred. 90%")
    ax.plot(grid, prior_mean, color="gray", linewidth=1.4, linestyle=":", label="prior pred. mean")
    ax.fill_between(grid, post_lo, post_hi, color="#1f77b4", alpha=0.25, label=f"round {n_rounds - 1} post. pred. 90%")
    # Thin individual posterior-predictive samples to show the structure.
    for i in range(min(25, post_curves.shape[0])):
        ax.plot(grid, post_curves[i], color="#1f77b4", alpha=0.15, linewidth=0.7)
    ax.plot(grid, post_mean, color="#1f77b4", linewidth=2.2, label="post. pred. mean")
    ax.plot(grid, true_I, color="orange", linewidth=2.2, label="true trajectory")
    ax.scatter(list(xi_star), list(y_observed), color="red", s=55, zorder=6, edgecolors="black",
               linewidths=0.6, label="observed y_t")
    ax.set_xlabel("time")
    ax.set_ylabel("number infected")
    beta_t, gamma_t = float(theta_true[0]), float(theta_true[1])
    ax.set_title(
        f"Posterior predictive vs observations — "
        f"truth β={beta_t:.3f}, γ={gamma_t:.3f}, R0={beta_t / gamma_t:.2f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(0.0, t_max)

    # Bottom panel: posterior-predictive band per round.
    cmap = plt.get_cmap("viridis")
    ax2 = axes[1]
    ax2.plot(grid, true_I, color="orange", linewidth=2.0, label="true trajectory", zorder=3)
    for t, particles in enumerate(per_round_particles):
        _, mean_t, lo_t, hi_t = simulate_band(particles)
        frac = t / max(n_rounds - 1, 1)
        color = cmap(frac)
        label = f"round {t}" if t in {0, n_rounds // 2, n_rounds - 1} else None
        ax2.fill_between(grid, lo_t, hi_t, color=color, alpha=0.12)
        ax2.plot(grid, mean_t, color=color, linewidth=1.6, label=label, alpha=0.9)
    ax2.scatter(list(xi_star), list(y_observed), color="red", s=40, zorder=5, edgecolors="black",
                linewidths=0.5, label="observed y_t")
    ax2.set_xlabel("time")
    ax2.set_ylabel("number infected")
    ax2.set_title("Posterior-predictive band tightening across rounds")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(0.0, t_max)

    fig.tight_layout()
    return _save(fig, output_path)


def save_submission_combo_figure(
    *,
    per_round_particles: List[np.ndarray],
    theta_true: np.ndarray,
    truth_trajectory,
    xi_star: List[float],
    y_observed: List[float],
    output_path: str | Path,
    t_max: float = DEFAULT_T_MAX,
    N: int = DEFAULT_N,
    I0: int = DEFAULT_I0,
    num_prior_draws: int = 300,
    num_predictive_draws: int = 120,
    seed: int = 0,
    dpi: int = 300,
    ess_per_round: List[float] | None = None,
    num_particles: int | None = None,
    aggregate_stats: Dict[str, Any] | None = None,
) -> str:
    """Single two-row figure: posterior shrinkage (β, γ) + sequential designs on the SIR curve.

    Intended for workshop / competition submissions (e.g. ICML AI for Science).
    """
    stem = Path(output_path)
    if stem.suffix.lower() in {".pdf", ".png"}:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig = plt.figure(figsize=(8.4, 5.6), constrained_layout=False)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.12], hspace=0.36, left=0.07, right=0.98, top=0.93, bottom=0.08)
    # Top row: β, γ, R₀ — the R₀ panel is the cleanly-identifiable one
    # and pre-empts the (β, γ) non-identifiability concern (Roosa &
    # Chowell 2019; Wearing et al. 2005).
    gs_top = gs[0].subgridspec(1, 3, wspace=0.32)
    ax_b = fig.add_subplot(gs_top[0, 0])
    ax_g = fig.add_subplot(gs_top[0, 1])
    ax_r = fig.add_subplot(gs_top[0, 2])
    ax_curve = fig.add_subplot(gs[1])

    cmap = plt.get_cmap("viridis")
    n_rounds = len(per_round_particles)
    param_names = ("β", "γ")
    for ax, k, name, panel in ((ax_b, 0, param_names[0], "(a)"), (ax_g, 1, param_names[1], "(b)")):
        # Tighten axis to the final-round 99.5% envelope so panels don't waste
        # space on prior tails that the posterior has already pruned.
        final_vals = per_round_particles[-1][:, k]
        lo = float(np.quantile(final_vals, 0.005))
        hi = float(np.quantile(final_vals, 0.995))
        # Pad with the truth and a small fraction of prior spread so the
        # |θ − θ*| shift is visible.
        lo = min(lo, float(theta_true[k]) * 0.8)
        hi = max(hi, float(theta_true[k]) * 1.2)
        span = hi - lo
        lo -= 0.05 * span
        hi += 0.05 * span
        grid = np.linspace(lo, hi, 400)
        for t, particles in enumerate(per_round_particles):
            samples = particles[:, k]
            bw = max(np.std(samples) * 1.06 * samples.shape[0] ** (-1 / 5), 1e-4)
            density = _kde_1d(samples, grid, bandwidth=bw)
            frac = t / max(n_rounds - 1, 1)
            color = cmap(frac)
            is_endpoint = t in {0, n_rounds - 1}
            ax.plot(
                grid, density, color=color, linewidth=2.0 if is_endpoint else 1.1,
                alpha=0.92,
            )
        ax.axvline(float(theta_true[k]), color="#d95f02", linestyle="--", linewidth=1.4, label="θ*")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.set_title(f"{panel} Posterior on {name}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.92)

    # Annotation: (β, γ) live on the R₀ ridge (Roosa & Chowell 2019; Wearing
    # 2005). Neither is identifiable alone from an SIR outbreak time series
    # — only R₀ = β/γ is. A visible offset in (a)/(b) that collapses in (c)
    # is the expected behaviour, not a calibration failure.
    ax_b.text(
        0.02, 0.97,
        "β, γ jointly live on\nthe R₀ ridge — see (c)",
        transform=ax_b.transAxes,
        ha="left", va="top",
        fontsize=6.5, style="italic", color="#555555",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    # Panel (c) — R₀ = β/γ, the identifiable combination. See Roosa &
    # Chowell (2019), Wearing et al. (2005): in early-outbreak SIR
    # only R₀ is well constrained; β and γ sit on a ridge.
    R0_per_round = [p[:, 0] / np.clip(p[:, 1], 1e-6, None) for p in per_round_particles]
    final_R0_vals = R0_per_round[-1]
    r0_true = float(theta_true[0]) / float(theta_true[1])
    lo_r = float(np.quantile(final_R0_vals, 0.005))
    hi_r = float(np.quantile(final_R0_vals, 0.995))
    lo_r = min(lo_r, r0_true * 0.8)
    hi_r = max(hi_r, r0_true * 1.2)
    span_r = hi_r - lo_r
    lo_r -= 0.05 * span_r
    hi_r += 0.05 * span_r
    grid_r = np.linspace(lo_r, hi_r, 400)
    for t, samples in enumerate(R0_per_round):
        bw = max(np.std(samples) * 1.06 * samples.shape[0] ** (-1 / 5), 1e-4)
        density = _kde_1d(samples, grid_r, bandwidth=bw)
        frac = t / max(n_rounds - 1, 1)
        color = cmap(frac)
        is_endpoint = t in {0, n_rounds - 1}
        ax_r.plot(
            grid_r, density, color=color, linewidth=2.0 if is_endpoint else 1.1,
            alpha=0.92,
        )
    ax_r.axvline(r0_true, color="#d95f02", linestyle="--", linewidth=1.4, label="R₀*")
    ax_r.set_xlabel("R₀ = β/γ")
    ax_r.set_ylabel("density")
    ax_r.set_title("(c) Posterior on R₀")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="upper right", framealpha=0.92)

    # Fix 5: final-round |bias|/σ on R₀. A calibrated posterior has this
    # near 0 (~1 at worst for well-informed updates); larger values flag
    # overconfidence on the wrong side of truth.
    final_R0 = R0_per_round[-1]
    R0_mean_final = float(np.mean(final_R0))
    R0_std_final = float(np.std(final_R0))
    if R0_std_final > 0:
        bias_over_sigma = abs(R0_mean_final - r0_true) / R0_std_final
    else:
        bias_over_sigma = float("inf")
    # Representative-seed |bias|/σ plus aggregate across seeds if provided.
    bias_lines = [f"|bias|/σ = {bias_over_sigma:.2f}"]
    if aggregate_stats is not None:
        agg_mean = aggregate_stats.get("R0_mean_across_seeds")
        agg_std = aggregate_stats.get("R0_std_across_seeds")
        agg_bias_range = aggregate_stats.get("bias_over_sigma_range")
        n_seeds = aggregate_stats.get("n_seeds")
        if agg_mean is not None and agg_std is not None:
            bias_lines.append(f"{n_seeds} seeds: R₀={agg_mean:.2f}±{agg_std:.2f}")
        if agg_bias_range is not None:
            bias_lines.append(
                f"|bias|/σ ∈ [{agg_bias_range[0]:.2f}, {agg_bias_range[1]:.2f}]"
            )
    ax_r.text(
        0.98, 0.72,
        "\n".join(bias_lines),
        transform=ax_r.transAxes,
        ha="right", va="top",
        fontsize=6.8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="#888888"),
    )

    # Fix 7: ESS — show range across seeds if aggregate given.
    if ess_per_round is not None and num_particles is not None and len(ess_per_round) > 0:
        if aggregate_stats is not None and "ess_final_range" in aggregate_stats:
            ess_lo, ess_hi = aggregate_stats["ess_final_range"]
            ess_str = f"ESS final ∈ [{ess_lo:.0f}, {ess_hi:.0f}] / {num_particles}"
        else:
            ess_str = f"ESS: {ess_per_round[-1]:.0f} / {num_particles}"
        ax_r.text(
            0.98, 0.04,
            ess_str,
            transform=ax_r.transAxes,
            ha="right", va="bottom",
            fontsize=6.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="#888888"),
        )

    # Shared colorbar for rounds 0..T across (a), (b), (c).
    # Replaces the cluttered per-round legend entries.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_rounds - 1))
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=(ax_b, ax_g, ax_r),
        orientation="horizontal", fraction=0.045, pad=0.18, aspect=45,
    )
    cbar.set_label("round (0 = prior)", fontsize=8)
    cbar.set_ticks(list(range(n_rounds)))
    cbar_labels = ["prior"] + [str(i) for i in range(1, n_rounds)]
    cbar.set_ticklabels(cbar_labels)
    cbar.ax.tick_params(labelsize=7)

    # --- designs on outbreak curve (panel d) ---
    grid = np.linspace(0.0, t_max, 400)
    thetas = sample_prior(num_prior_draws, seed=seed)
    rng = np.random.default_rng(seed + 1)
    I_grid = np.zeros((num_prior_draws, grid.shape[0]))
    for i in range(num_prior_draws):
        traj = simulate_sir_trajectory(
            thetas[i], N=N, I0=I0, t_max=t_max, seed=int(rng.integers(0, 2**31 - 1))
        )
        I_grid[i] = traj.query(grid)
    mean = I_grid.mean(axis=0)
    lo = np.quantile(I_grid, 0.05, axis=0)
    hi = np.quantile(I_grid, 0.95, axis=0)
    true_I = truth_trajectory.query(grid)

    # Posterior predictive band at the final round, built by
    # forward-simulating the Gillespie model on posterior particles.
    # Same pattern as save_posterior_predictive_plot. Makes the
    # identifiability pathology visible: a narrow band that misses the
    # true orange curve signals posterior concentration on the wrong
    # (β, γ). Foster 2020; Kleinegesse & Gutmann 2021.
    final_particles = per_round_particles[-1]
    n_post = min(int(num_predictive_draws), final_particles.shape[0])
    post_idx = np.random.default_rng(seed + 7).choice(
        final_particles.shape[0], size=n_post, replace=False,
    )
    I_post = np.zeros((n_post, grid.shape[0]))
    rng_post = np.random.default_rng(seed + 101)
    for i, idx in enumerate(post_idx):
        traj = simulate_sir_trajectory(
            final_particles[idx], N=N, I0=I0, t_max=t_max,
            seed=int(rng_post.integers(0, 2**31 - 1)),
        )
        I_post[i] = traj.query(grid)
    post_mean_curve = I_post.mean(axis=0)
    post_lo = np.quantile(I_post, 0.05, axis=0)
    post_hi = np.quantile(I_post, 0.95, axis=0)

    # Diagnostic: posterior-predictive 95% upper should not exceed
    # the prior-predictive 95% upper — that would signal surrogate
    # miscalibration or posterior widening under a pathological
    # observation likelihood.
    exceed_mask = post_hi > 1.1 * hi
    exceed_range: tuple[float, float] | None = None
    if np.any(exceed_mask):
        exceed_t = grid[exceed_mask]
        exceed_range = (float(exceed_t.min()), float(exceed_t.max()))
        print(
            "[sir/plot] posterior-pred band exceeds prior-pred on "
            f"[{exceed_range[0]:.1f}, {exceed_range[1]:.1f}]; "
            "check surrogate calibration"
        )

    ax_curve.fill_between(grid, lo, hi, color="lightgray", alpha=0.75, label="prior predictive 90%")
    ax_curve.fill_between(
        grid, post_lo, post_hi, color="#2ca02c", alpha=0.22,
        label=f"posterior pred. 90% (Gillespie, round {n_rounds - 1})",
    )
    ax_curve.plot(grid, mean, color="#1f77b4", linewidth=1.6, linestyle=":", label="prior pred. mean")
    ax_curve.plot(grid, post_mean_curve, color="#2ca02c", linewidth=1.8, label="posterior pred. mean")
    ax_curve.plot(grid, true_I, color="#d95f02", linewidth=2.0, label="true trajectory")

    # Shade the x-range where posterior-pred 95% exceeds prior-pred 95%
    # by >10% — an honest annotation of residual surrogate miscalibration
    # (Hermans et al. 2021 "Averting a Crisis in SBI").
    if exceed_range is not None:
        ax_curve.axvspan(
            exceed_range[0], exceed_range[1],
            color="#ad1457", alpha=0.06, zorder=1.5,
        )
        ax_curve.text(
            0.5 * (exceed_range[0] + exceed_range[1]),
            ax_curve.get_ylim()[1] * 0.96 if ax_curve.get_ylim()[1] > 0 else 0,
            "surrogate\nmiscalibration",
            ha="center", va="top", fontsize=6.2, color="#ad1457",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.82, edgecolor="#ad1457", linewidth=0.5),
            zorder=7,
        )

    # Stagger t=i annotation offsets vertically so tightly-clustered designs
    # don't collide. Offsets cycle through ±y multiples tuned to look
    # clean for up to ~8 designs.
    offset_cycle = [(0, 10), (0, -14), (0, 20), (0, -24), (0, 30), (0, -34),
                     (0, 40), (0, -44)]
    for t, (xi_t, y_t) in enumerate(zip(xi_star, y_observed)):
        frac = t / max(len(xi_star) - 1, 1)
        color = cmap(frac)
        ax_curve.axvline(float(xi_t), color=color, alpha=0.45, linewidth=1.0, zorder=2)
        ax_curve.scatter(
            [xi_t], [y_t], color=color, s=38, zorder=5, edgecolors="0.2", linewidths=0.45,
        )
        dx, dy = offset_cycle[t % len(offset_cycle)]
        ax_curve.annotate(
            f"t={t + 1}",
            xy=(float(xi_t), float(y_t)),
            xycoords="data",
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=6.5,
            zorder=6,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.6) if abs(dy) > 15 else None,
        )
    # Top-left box listing the monotone ξ-sequence so reviewers don't
    # have to visually match colors to round indices.
    ax_curve.text(
        0.02, 0.97,
        f"ξ-sequence: {', '.join(f'{x:.1f}' for x in xi_star)}",
        transform=ax_curve.transAxes,
        verticalalignment="top",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax_curve.set_xlim(0.0, t_max)
    ax_curve.set_xlabel("measurement time")
    ax_curve.set_ylabel("infected count")
    ax_curve.set_title("(d) Sequential BOED designs on the latent outbreak")
    ax_curve.grid(True, alpha=0.3)
    h_leg, _ = ax_curve.get_legend_handles_labels()
    round_patch = Patch(facecolor=cmap(0.5), edgecolor="0.3", linewidth=0.5, label="design rounds (color)")
    ax_curve.legend(
        handles=list(h_leg) + [round_patch],
        loc="upper right",
        framealpha=0.95,
        fontsize=7.5,
    )

    fig.suptitle(
        "Sequential Bayesian optimal experimental design — stochastic SIR (Gillespie)",
        fontsize=10.5, y=0.995,
    )

    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", format="png")
    plt.close(fig)
    return str(pdf_path)


def save_boed_vs_uniform_comparison(
    *,
    boed_R0_stds: np.ndarray,  # (n_seeds, n_rounds + 1)
    uniform_R0_stds: np.ndarray,  # (n_seeds, n_rounds + 1)
    boed_eig: np.ndarray,  # (n_seeds, n_rounds)
    uniform_eig: np.ndarray,  # (n_seeds, n_rounds)
    boed_R0_bias: np.ndarray,  # (n_seeds,) final-round bias
    uniform_R0_bias: np.ndarray,  # (n_seeds,)
    output_path: str | Path,
) -> str:
    """BOED-vs-uniform comparison (Kleinegesse & Gutmann 2021, Table 1).

    Three panels: σ(R₀) shrinkage per round, per-round EIG trajectory, final-round
    |bias| distribution. All panels show mean ± seed-σ bands where applicable.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    ax_sig, ax_eig, ax_bias = axes
    r_sigma = np.arange(boed_R0_stds.shape[1])
    b_mean = boed_R0_stds.mean(axis=0)
    b_sd = boed_R0_stds.std(axis=0)
    u_mean = uniform_R0_stds.mean(axis=0)
    u_sd = uniform_R0_stds.std(axis=0)
    ax_sig.plot(r_sigma, b_mean, "-o", color="#ae2012", linewidth=2.0, label="BOED (ours)")
    ax_sig.fill_between(r_sigma, b_mean - b_sd, b_mean + b_sd, color="#ae2012", alpha=0.18)
    ax_sig.plot(r_sigma, u_mean, "-s", color="#2a9d8f", linewidth=2.0, label="uniform grid")
    ax_sig.fill_between(r_sigma, u_mean - u_sd, u_mean + u_sd, color="#2a9d8f", alpha=0.18)
    ax_sig.set_xlabel("round (0 = prior)")
    ax_sig.set_ylabel("σ(R₀)")
    ax_sig.set_title("(a) R₀ posterior σ — BOED vs uniform")
    ax_sig.grid(True, alpha=0.3)
    ax_sig.legend(loc="best", fontsize=8)

    r_eig = np.arange(1, boed_eig.shape[1] + 1)
    ax_eig.plot(r_eig, boed_eig.mean(axis=0), "-o", color="#ae2012", linewidth=2.0, label="BOED")
    ax_eig.fill_between(
        r_eig,
        boed_eig.mean(axis=0) - boed_eig.std(axis=0),
        boed_eig.mean(axis=0) + boed_eig.std(axis=0),
        color="#ae2012", alpha=0.18,
    )
    ax_eig.plot(r_eig, uniform_eig.mean(axis=0), "-s", color="#2a9d8f", linewidth=2.0, label="uniform grid")
    ax_eig.fill_between(
        r_eig,
        uniform_eig.mean(axis=0) - uniform_eig.std(axis=0),
        uniform_eig.mean(axis=0) + uniform_eig.std(axis=0),
        color="#2a9d8f", alpha=0.18,
    )
    ax_eig.set_xlabel("round")
    ax_eig.set_ylabel("EIG at ξ_t")
    ax_eig.set_title("(b) Per-round EIG — BOED vs uniform")
    ax_eig.grid(True, alpha=0.3)
    ax_eig.legend(loc="best", fontsize=8)

    data = [np.abs(boed_R0_bias), np.abs(uniform_R0_bias)]
    bp = ax_bias.boxplot(
        data, labels=["BOED", "uniform"], widths=0.45, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
    )
    colors = ["#ae2012", "#2a9d8f"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    # Overlay the raw per-seed values so small-n boxes don't hide the
    # underlying distribution.
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        xs = rng.normal(loc=i, scale=0.04, size=arr.shape[0])
        ax_bias.scatter(xs, arr, color=colors[i - 1], edgecolors="black",
                        s=28, linewidth=0.5, alpha=0.85, zorder=3)
    ax_bias.set_ylabel("|R₀ posterior bias|")
    ax_bias.set_title("(c) Final-round |bias| across seeds")
    ax_bias.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Sequential BOED vs non-adaptive uniform grid — SIR Gillespie",
        fontsize=11, y=1.04,
    )
    return _save(fig, output_path)


def save_sbc_rank_histogram(
    *,
    ranks: np.ndarray,  # (n_truths, 3) — rank of each truth in its posterior for β, γ, R₀
    num_bins: int,
    output_path: str | Path,
    param_names: tuple[str, str, str] = ("β", "γ", "R₀"),
) -> str:
    """Simulation-based calibration (Talts et al. 2018) rank histogram.

    For a well-calibrated posterior the rank of the truth within its
    posterior samples should be Uniform(0, L). A U-shape indicates
    underconfidence (too much mass in the tails), a peaked histogram
    indicates overconfidence, and a slope indicates bias. Dashed lines
    mark the 99% uniform envelope under Binomial(num_truths, 1/num_bins).
    """
    fig, axes = plt.subplots(1, ranks.shape[1], figsize=(4.5 * ranks.shape[1], 3.6), constrained_layout=True)
    if ranks.shape[1] == 1:
        axes = [axes]
    n_truths = ranks.shape[0]
    expected = n_truths / num_bins
    # 99% CI for a uniform histogram bin count. Prefer the exact Binomial
    # quantile via scipy; fall back to a Normal approximation if scipy is
    # not installed (only marginally wider for small n_truths).
    try:
        from scipy.stats import binom  # type: ignore
        lo = binom.ppf(0.005, n_truths, 1.0 / num_bins)
        hi = binom.ppf(0.995, n_truths, 1.0 / num_bins)
    except Exception:  # pragma: no cover — scipy fallback
        p = 1.0 / num_bins
        sd = math.sqrt(max(n_truths * p * (1.0 - p), 1e-9))
        # z_{0.005} ≈ 2.576 for a 99% two-sided CI.
        lo = expected - 2.576 * sd
        hi = expected + 2.576 * sd

    for k, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(
            ranks[:, k], bins=num_bins, range=(0, num_bins),
            color="#457b9d", edgecolor="white", alpha=0.85,
        )
        ax.axhspan(lo, hi, color="#2a9d8f", alpha=0.22, label="99% uniform envelope")
        ax.axhline(expected, color="#2a9d8f", linestyle="--", linewidth=1.2, label="uniform mean")
        ax.set_xlabel(f"rank of θ* among posterior samples — {name}")
        ax.set_ylabel("count")
        ax.set_title(f"SBC rank histogram — {name}")
        ax.grid(True, alpha=0.3, axis="y")
        if k == 0:
            ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"Simulation-based calibration ({n_truths} truths × {num_bins} rank bins, Talts et al. 2018)",
        fontsize=11, y=1.06,
    )
    return _save(fig, output_path)


def _save(fig, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
