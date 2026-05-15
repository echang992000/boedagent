"""Plots for the sequential linear-regression BOED run."""

from __future__ import annotations

from pathlib import Path
from typing import List

import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _kde_1d(samples: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    z = (grid[:, None] - samples[None, :]) / bandwidth
    k = np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    return k.mean(axis=1) / bandwidth


def save_posterior_shrinkage_plot(
    per_round_samples: List[np.ndarray],
    theta_true: np.ndarray,
    *,
    output_path: str | Path,
    param_names=("a (slope)", "b (intercept)"),
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    n_rounds = len(per_round_samples)
    for k, (ax, name) in enumerate(zip(axes, param_names)):
        all_vals = np.concatenate([s[:, k] for s in per_round_samples])
        lo = float(np.quantile(all_vals, 0.002))
        hi = float(np.quantile(all_vals, 0.998))
        lo = min(lo, float(theta_true[k]) - 0.5)
        hi = max(hi, float(theta_true[k]) + 0.5)
        grid = np.linspace(lo, hi, 400)
        for t, samples in enumerate(per_round_samples):
            bw = max(np.std(samples[:, k]) * 1.06 * samples.shape[0] ** (-1 / 5), 1e-3)
            density = _kde_1d(samples[:, k], grid, bandwidth=bw)
            frac = t / max(n_rounds - 1, 1)
            color = cmap(frac)
            label = f"round {t}" if t in {0, n_rounds - 1} else None
            ax.plot(grid, density, color=color, linewidth=2.0 if label else 1.0, label=label, alpha=0.9)
        ax.axvline(float(theta_true[k]), color="orange", linestyle="--", linewidth=1.5, label="truth")
        ax.set_xlabel(name)
        ax.set_ylabel("posterior density")
        ax.set_title(f"Posterior over {name}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Linear regression sequential BOED — posterior shrinkage", y=1.02)
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
        axes[0].annotate(f"xi={xi_t:.2f}", (r, e), xytext=(0, 6), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("best EIG this round")
    axes[0].set_title("Per-round EIG at the chosen xi*")
    axes[0].grid(True, alpha=0.3)

    cmap = plt.get_cmap("viridis")
    for t, hist in enumerate(round_histories):
        frac = t / max(len(round_histories) - 1, 1)
        axes[1].plot(hist["eig_history"], color=cmap(frac), alpha=0.7, linewidth=1.2, label=f"round {t}" if t in {0, len(round_histories) - 1} else None)
    axes[1].set_xlabel("inner step")
    axes[1].set_ylabel("EIG estimate")
    axes[1].set_title("Inner design-optimisation trajectories")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _save(fig, output_path)


def save_fit_overlay_plot(
    *,
    theta_true: np.ndarray,
    xi_star: List[float],
    y_observed: List[float],
    per_round_posterior_samples: List[np.ndarray],
    x_range: tuple[float, float],
    output_path: str | Path,
) -> str:
    """y vs xi overlay — true line + per-round posterior predictive bands."""
    xs = np.linspace(x_range[0], x_range[1], 200)
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    for t, samples in enumerate(per_round_posterior_samples):
        y_draws = samples[:, 0:1] * xs[None, :] + samples[:, 1:2]
        mean = y_draws.mean(axis=0)
        lo = np.quantile(y_draws, 0.05, axis=0)
        hi = np.quantile(y_draws, 0.95, axis=0)
        frac = t / max(len(per_round_posterior_samples) - 1, 1)
        ax.fill_between(xs, lo, hi, alpha=0.15, color=cmap(frac))
        if t in {0, len(per_round_posterior_samples) - 1}:
            ax.plot(xs, mean, color=cmap(frac), linewidth=1.6, label=f"round {t} predictive mean")

    true_y = theta_true[0] * xs + theta_true[1]
    ax.plot(xs, true_y, color="orange", linewidth=2.2, label=f"truth (a={theta_true[0]}, b={theta_true[1]})")

    for t, (xi_t, y_t) in enumerate(zip(xi_star, y_observed)):
        frac = t / max(len(xi_star) - 1, 1)
        color = cmap(frac)
        ax.scatter([xi_t], [y_t], color=color, s=45, zorder=5, edgecolors="black", linewidths=0.5,
                   label=f"xi_{t+1}={xi_t:.2f}")
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_xlabel("xi")
    ax.set_ylabel("y")
    ax.set_title("Linear regression sequential BOED — observations and posterior")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, output_path)


def _save(fig, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
