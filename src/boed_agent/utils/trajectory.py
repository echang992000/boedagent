"""Utilities for compact trajectory summaries and optional plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from boed_agent.models import DesignVariable, OptimizationStep


def summarize_optimized_design_histories(
    histories: list[list[OptimizationStep]],
    design_variables: list[DesignVariable],
) -> list[dict[str, Any]]:
    return [
        summarize_optimized_design_history(history, design_variables, history_index=index)
        for index, history in enumerate(histories)
    ]


def summarize_optimized_design_history(
    history: list[OptimizationStep],
    design_variables: list[DesignVariable],
    history_index: int = 0,
) -> dict[str, Any]:
    if not history:
        return {
            "history_index": history_index,
            "num_steps": 0,
            "design_dimensions": [],
            "checkpoints": [],
        }

    eig_values = [step.eig for step in history if step.eig is not None]
    best_step = max(
        (step for step in history if step.eig is not None),
        key=lambda step: step.eig,
        default=history[-1],
    )
    first = history[0]
    last = history[-1]

    dimension_summaries: list[dict[str, Any]] = []
    num_dims = len(last.design)
    for dim_index in range(num_dims):
        values = [step.design[dim_index] for step in history]
        name = (
            design_variables[dim_index].name
            if dim_index < len(design_variables)
            else f"design_{dim_index}"
        )
        dimension_summaries.append(
            {
                "name": name,
                "start": values[0],
                "end": values[-1],
                "min": min(values),
                "max": max(values),
                "delta": values[-1] - values[0],
            }
        )

    checkpoint_indices = _checkpoint_indices(len(history))
    checkpoints = [history[index].to_dict() for index in checkpoint_indices]

    return {
        "history_index": history_index,
        "num_steps": len(history),
        "start_design": first.design,
        "end_design": last.design,
        "best_design": best_step.design,
        "best_step": best_step.step,
        "best_eig": best_step.eig,
        "eig_min": min(eig_values) if eig_values else None,
        "eig_max": max(eig_values) if eig_values else None,
        "design_dimensions": dimension_summaries,
        "checkpoints": checkpoints,
    }


def compact_optimized_design_history_summary(summary: dict[str, Any]) -> dict[str, Any]:
    compact = dict(summary)
    compact.pop("checkpoints", None)
    return compact


def compact_optimized_design_history_summaries(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [compact_optimized_design_history_summary(summary) for summary in summaries]


def save_design_trajectory_plot(
    histories: list[list[OptimizationStep]],
    design_variables: list[DesignVariable],
    output_path: str | Path,
) -> str:
    if not histories:
        raise ValueError("No histories available for plotting.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Trajectory plotting requires the optional `matplotlib` dependency. "
            "Install with `pip install -e \".[plot]\"`."
        ) from exc

    first_history = histories[0]
    num_dims = len(first_history[0].design) if first_history else 0
    has_eig = any(step.eig is not None for history in histories for step in history)
    num_rows = max(num_dims, 1) + (1 if has_eig else 0)
    fig, axes = plt.subplots(num_rows, 1, figsize=(8, max(3 * num_rows, 4)), sharex=True)
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    elif isinstance(axes, (list, tuple)):
        axes = list(axes)
    else:
        axes = [axes]

    for dim_index in range(max(num_dims, 1)):
        axis = axes[dim_index]
        for history_index, history in enumerate(histories):
            steps = [step.step for step in history]
            values = [step.design[dim_index] for step in history] if num_dims else []
            label = f"history_{history_index}"
            axis.plot(steps, values, marker="o", linewidth=1.5, label=label)
        name = (
            design_variables[dim_index].name
            if dim_index < len(design_variables)
            else f"design_{dim_index}"
        )
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.3)

    if has_eig:
        eig_axis = axes[-1]
        for history_index, history in enumerate(histories):
            steps = [step.step for step in history]
            eigs = [step.eig for step in history]
            eig_axis.plot(steps, eigs, linewidth=1.5, label=f"history_{history_index}")
        eig_axis.set_ylabel("EIG")
        eig_axis.set_xlabel("Step")
        eig_axis.grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel("Step")

    if len(histories) > 1:
        axes[0].legend(loc="best")
    fig.suptitle("Optimized Design Trajectory")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def save_optimization_history_npz(
    history: list[OptimizationStep],
    output_path: str | Path,
    sigma_history: list[list[float]] | None = None,
) -> str:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Compressed trajectory export requires `numpy` in the active environment."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([step.step for step in history], dtype=np.int64)
    designs = np.asarray([step.design for step in history], dtype=np.float32)
    eigs = np.asarray(
        [np.nan if step.eig is None else float(step.eig) for step in history],
        dtype=np.float32,
    )
    arrays: dict[str, Any] = {
        "steps": steps,
        "designs": designs,
        "eig": eigs,
    }
    if sigma_history is not None:
        arrays["sigma_history"] = np.asarray(sigma_history, dtype=np.float32)
    np.savez_compressed(output, **arrays)
    return str(output)


def _checkpoint_indices(num_steps: int) -> list[int]:
    if num_steps <= 1:
        return [0] if num_steps == 1 else []
    indices = {0, num_steps - 1, num_steps // 2}
    if num_steps >= 4:
        indices.add(num_steps // 3)
        indices.add((2 * num_steps) // 3)
    return sorted(indices)
