"""Post-run analysis for BMP4 sequential Promisys comparison artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from examples.cases.bmp4_gradient import promisys_onestep
from examples.cases.bmp4_gradient import promisys_twostep
from examples.cases.bmp4_gradient.data import (
    DEFAULT_DATA_PATH,
    build_joint_bmp4_gradient_data,
    load_bmp4_gradient_data,
)
from examples.cases.bmp4_gradient.plotting import save_sequential_acquisition_plot


try:  # pragma: no cover - exercised by tests/examples when numpy is installed
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


FAMILY_MODULES = {
    "promisys_onestep": promisys_onestep,
    "promisys_twostep": promisys_twostep,
}


def analyze_sequential_run(
    run_dir: str | Path,
    normalizer_data_dir: str | Path | None = None,
    max_posterior_draws: int = 128,
    data_path: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Analyze one ``sequential_*`` run directory and write metrics/plots."""
    if np is None:
        raise RuntimeError("Sequential BMP4 analysis requires numpy.")

    run_path = Path(run_dir)
    fit_summary = _load_json(run_path / "fit_summary.json")
    trace_payload = _load_json(run_path / "sequential_trace.json")
    trace = list(trace_payload.get("trace") or [])
    family_name = str(fit_summary["family"])
    module = _family_module(family_name)
    cell_lines = [str(item) for item in fit_summary["cell_lines"]]
    data_bundle = load_bmp4_gradient_data(data_path or DEFAULT_DATA_PATH)
    joint_data = build_joint_bmp4_gradient_data(data_bundle, cell_lines=cell_lines)
    normalizer_dir = Path(normalizer_data_dir or fit_summary.get("normalizer_data_dir") or module.DEFAULT_NORMALIZER_DATA_DIR)
    normalizer = module.Bmp4Normalizer.from_source_dir(normalizer_dir)

    metrics = _metrics_from_trace(
        run_path=run_path,
        trace=trace,
        module=module,
        joint_data=joint_data,
        normalizer=normalizer,
        max_posterior_draws=max_posterior_draws,
        seed=seed,
    )

    metrics_json = run_path / "sequential_metrics.json"
    metrics_tsv = run_path / "sequential_metrics.tsv"
    diagnostics_png = run_path / "sequential_diagnostics.png"
    _write_json(
        metrics_json,
        {
            "run_dir": str(run_path),
            "family": family_name,
            "prior_mode": fit_summary.get("prior_mode"),
            "cell_lines": cell_lines,
            "normalizer_data_dir": str(normalizer_dir),
            "max_posterior_draws": int(max_posterior_draws),
            "metrics": metrics,
        },
    )
    _write_tsv(metrics_tsv, metrics)
    plot_sequential_diagnostics(
        metrics=metrics,
        cell_lines=cell_lines,
        output_path=diagnostics_png,
        title=f"{family_name} sequential diagnostics ({fit_summary.get('prior_mode')})",
    )
    return {
        "run_dir": str(run_path),
        "family": family_name,
        "prior_mode": fit_summary.get("prior_mode"),
        "cell_lines": cell_lines,
        "metric_count": len(metrics),
        "metrics_json": str(metrics_json),
        "metrics_tsv": str(metrics_tsv),
        "diagnostics_plot": str(diagnostics_png),
        "metrics": metrics,
    }


def analyze_comparison_run(
    comparison_dir: str | Path,
    normalizer_data_dir: str | Path | None = None,
    max_posterior_draws: int = 128,
    data_path: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Analyze every ``sequential_*`` child in a comparison directory."""
    root = Path(comparison_dir)
    if (root / "sequential_trace.json").exists():
        return analyze_sequential_run(
            root,
            normalizer_data_dir=normalizer_data_dir,
            max_posterior_draws=max_posterior_draws,
            data_path=data_path,
            seed=seed,
        )

    sequential_dirs = sorted(path for path in root.glob("sequential_*") if path.is_dir())
    if not sequential_dirs:
        raise FileNotFoundError(f"No sequential_* directories found in {root}.")

    analyses = [
        analyze_sequential_run(
            path,
            normalizer_data_dir=normalizer_data_dir,
            max_posterior_draws=max_posterior_draws,
            data_path=data_path,
            seed=seed,
        )
        for path in sequential_dirs
    ]
    comparison_payload = {
        "comparison_dir": str(root),
        "max_posterior_draws": int(max_posterior_draws),
        "runs": [
            {
                "prior_mode": item.get("prior_mode"),
                "run_dir": item["run_dir"],
                "cell_lines": item["cell_lines"],
                "metrics_json": item["metrics_json"],
                "metrics_tsv": item["metrics_tsv"],
                "diagnostics_plot": item["diagnostics_plot"],
                "metrics": item["metrics"],
            }
            for item in analyses
        ],
    }
    comparison_json = root / "comparison_metrics.json"
    _write_json(comparison_json, comparison_payload)
    comparison_plot = root / "comparison_diagnostics.png"
    plot_comparison_diagnostics(comparison_payload["runs"], comparison_plot)
    return {
        "comparison_dir": str(root),
        "run_count": len(analyses),
        "comparison_metrics": str(comparison_json),
        "comparison_plot": str(comparison_plot),
        "runs": analyses,
    }


def plot_trace_acquisition_history(path: str | Path) -> dict[str, Any]:
    """Regenerate trace-only EIG/design plots without posterior predictive analysis."""
    root = Path(path)
    if (root / "sequential_trace.json").exists():
        return _plot_one_trace_acquisition_history(root)

    sequential_dirs = sorted(child for child in root.glob("sequential_*") if child.is_dir())
    if not sequential_dirs:
        raise FileNotFoundError(f"No sequential_trace.json or sequential_* directories found in {root}.")
    runs = [_plot_one_trace_acquisition_history(child) for child in sequential_dirs]
    return {
        "comparison_dir": str(root),
        "run_count": len(runs),
        "runs": runs,
    }


def plot_sequential_posterior_comparison(path: str | Path) -> dict[str, Any]:
    """Regenerate initial-prior vs final-posterior plots for sequential runs."""
    root = Path(path)
    if (root / "sequential_trace.json").exists():
        return _plot_one_sequential_posterior_comparison(root)

    sequential_dirs = sorted(child for child in root.glob("sequential_*") if child.is_dir())
    if not sequential_dirs:
        raise FileNotFoundError(f"No sequential_trace.json or sequential_* directories found in {root}.")
    runs = [_plot_one_sequential_posterior_comparison(child) for child in sequential_dirs]
    return {
        "comparison_dir": str(root),
        "run_count": len(runs),
        "runs": runs,
    }


def _plot_one_sequential_posterior_comparison(run_path: Path) -> dict[str, Any]:
    fit_summary = _load_json(run_path / "fit_summary.json")
    module = _family_module(str(fit_summary["family"]))
    initial_samples = _load_theta_samples(run_path / "initial_prior_samples.pt")
    posterior_samples = _load_theta_samples(run_path / "posterior_samples.pt")
    normalized_path = run_path / "prior_posterior_comparison.png"
    positive_path = run_path / "prior_posterior_comparison_positive.png"
    module._save_theta_comparison_plot(
        initial_samples,
        posterior_samples,
        normalized_path,
        positive=False,
    )
    module._save_theta_comparison_plot(
        initial_samples,
        posterior_samples,
        positive_path,
        positive=True,
    )
    return {
        "run_dir": str(run_path),
        "family": fit_summary.get("family"),
        "prior_mode": fit_summary.get("prior_mode"),
        "prior_posterior_plot": str(normalized_path),
        "prior_posterior_positive_plot": str(positive_path),
    }


def _plot_one_trace_acquisition_history(run_path: Path) -> dict[str, Any]:
    fit_summary = _load_json(run_path / "fit_summary.json")
    trace_payload = _load_json(run_path / "sequential_trace.json")
    trace = list(trace_payload.get("trace") or [])
    output = run_path / "eig_optimization.png"
    plot_path = save_sequential_acquisition_plot(
        trace,
        output,
        title=(
            f"Sequential acquired-design EIG: {fit_summary.get('family')} "
            f"({fit_summary.get('prior_mode')})"
        ),
    )
    return {
        "run_dir": str(run_path),
        "family": fit_summary.get("family"),
        "prior_mode": fit_summary.get("prior_mode"),
        "acquisition_count": len(trace),
        "eig_optimization_plot": plot_path,
    }


def plot_sequential_diagnostics(
    *,
    metrics: list[dict[str, Any]],
    cell_lines: list[str],
    output_path: str | Path,
    title: str,
) -> str:
    """Write a four-panel sequential diagnostic plot."""
    import matplotlib.pyplot as plt

    if not metrics:
        raise ValueError("No sequential metrics to plot.")

    colors = _cell_line_colors(cell_lines)
    x = np.asarray([row["acquisition"] for row in metrics], dtype=float)
    eig = np.asarray([row["best_eig"] for row in metrics], dtype=float)
    dose = np.asarray([row["dose_ng_ml"] for row in metrics], dtype=float)
    median_norm = np.asarray([row["median_predictive_abs_error_norm"] for row in metrics], dtype=float)
    q25_norm = np.asarray([row["q25_predictive_abs_error_norm"] for row in metrics], dtype=float)
    q75_norm = np.asarray([row["q75_predictive_abs_error_norm"] for row in metrics], dtype=float)

    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.1, 1.2, 1.5], hspace=0.35)
    ax_eig = fig.add_subplot(gs[0, 0])
    ax_dose = fig.add_subplot(gs[1, 0], sharex=ax_eig)
    ax_dist = fig.add_subplot(gs[2, 0], sharex=ax_eig)
    ax_heat = fig.add_subplot(gs[3, 0])

    for cell_line in cell_lines:
        mask = np.asarray([row["selected_cell_line"] == cell_line for row in metrics], dtype=bool)
        ax_eig.scatter(x[mask], eig[mask], color=colors[cell_line], s=78, label=cell_line, zorder=3)
        ax_dose.scatter(x[mask], dose[mask], color=colors[cell_line], s=78, label=cell_line, zorder=3)
    ax_eig.set_ylabel("Final EIG")
    ax_eig.set_title("Final EIG after each acquisition optimization")
    ax_eig.grid(True, alpha=0.25)
    ax_eig.legend(loc="best", ncols=min(len(cell_lines), 4))

    ax_dose.set_yscale("log")
    ax_dose.set_ylabel("Final chosen BMP4 (ng/mL)")
    ax_dose.set_title("Actual snapped experimental design")
    ax_dose.grid(True, alpha=0.25)
    for row in metrics:
        ax_dose.annotate(
            str(row["dose_index"]),
            (row["acquisition"], row["dose_ng_ml"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#2f3e46",
        )

    ax_dist.plot(x, median_norm, color="#ae2012", linewidth=2.0, marker="o")
    ax_dist.fill_between(x, q25_norm, q75_norm, color="#ee9b00", alpha=0.25, label="q25-q75")
    ax_dist.set_ylabel("Median abs. predictive error\n(normalized response)")
    ax_dist.set_xlabel("Acquisition")
    ax_dist.grid(True, alpha=0.25)
    ax_dist.legend(loc="best")

    _plot_coverage_heatmap(ax_heat, metrics, cell_lines)
    fig.suptitle(title, fontsize=15)
    return _save_figure(fig, output_path)


def plot_comparison_diagnostics(runs: list[dict[str, Any]], output_path: str | Path) -> str:
    """Write root-level default/literature comparison diagnostics."""
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("No analyzed runs to compare.")
    labels = [str(run.get("prior_mode") or Path(run["run_dir"]).name) for run in runs]
    fig = plt.figure(figsize=(7.5 * max(len(runs), 2), 11))
    gs = fig.add_gridspec(3, len(runs), height_ratios=[1.0, 1.0, 1.4], hspace=0.35)

    ax_eig = fig.add_subplot(gs[0, :])
    ax_dist = fig.add_subplot(gs[1, :], sharex=ax_eig)
    palette = ["#005f73", "#ae2012", "#0a9396", "#9b2226"]
    for index, run in enumerate(runs):
        metrics = run["metrics"]
        x = [row["acquisition"] for row in metrics]
        ax_eig.scatter(
            x,
            [row["best_eig"] for row in metrics],
            s=58,
            color=palette[index % len(palette)],
            label=labels[index],
        )
        ax_dist.scatter(
            x,
            [row["median_predictive_abs_error_norm"] for row in metrics],
            s=58,
            color=palette[index % len(palette)],
            label=labels[index],
        )
    ax_eig.set_ylabel("Final EIG")
    ax_eig.set_title("One final EIG per acquired experiment")
    ax_eig.grid(True, alpha=0.25)
    ax_eig.legend(loc="best")
    ax_dist.set_ylabel("Median abs. predictive error\n(normalized response)")
    ax_dist.set_xlabel("Acquisition")
    ax_dist.grid(True, alpha=0.25)
    ax_dist.legend(loc="best")

    for index, run in enumerate(runs):
        ax_heat = fig.add_subplot(gs[2, index])
        metrics = run["metrics"]
        cell_lines = list(run.get("cell_lines") or _cell_lines_from_metrics(metrics))
        _plot_coverage_heatmap(ax_heat, metrics, cell_lines)
        ax_heat.set_title(f"Coverage: {labels[index]}")

    fig.suptitle("Sequential BMP4 comparison diagnostics", fontsize=16)
    return _save_figure(fig, output_path)


def _metrics_from_trace(
    *,
    run_path: Path,
    trace: list[dict[str, Any]],
    module: Any,
    joint_data: Any,
    normalizer: Any,
    max_posterior_draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    cell_index_by_name = {str(name): index for index, name in enumerate(joint_data.cell_lines)}
    acquired_by_cell: dict[str, list[int]] = {str(name): [] for name in joint_data.cell_lines}
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))

    for index, item in enumerate(trace):
        snapped = dict(item["snapped_design"])
        cell_line = str(snapped["cell_line"])
        dose_index = int(snapped["dose_index"])
        acquired_by_cell[cell_line].append(dose_index)
        posterior_path = _posterior_path_for_step(run_path, item, index + 1)
        posterior_samples = _load_theta_samples(posterior_path)
        distance = _posterior_predictive_distance(
            module=module,
            joint_data=joint_data,
            normalizer=normalizer,
            posterior_samples=posterior_samples,
            acquired_by_cell=acquired_by_cell,
            max_posterior_draws=max_posterior_draws,
            rng=rng,
        )
        boed = dict(item.get("boed") or {})
        row: dict[str, Any] = {
            "acquisition": int(item.get("acquisition", index + 1)),
            "round": int(item.get("round", index + 1)),
            "batch_index": int(item.get("batch_index", 1)),
            "prior_mode": item.get("prior_mode"),
            "selected_cell_line": cell_line,
            "cell_line_index": int(snapped.get("cell_line_index", cell_index_by_name[cell_line])),
            "dose_index": dose_index,
            "dose_ng_ml": float(snapped["dose"]),
            "log10_dose": float(snapped["log10_dose"]),
            "bmp4_norm_design": float(snapped["bmp4_norm_design"]),
            "proposed_bmp4_norm": float(snapped["proposed_bmp4_norm"]),
            "snap_norm_distance": float(snapped["norm_distance"]),
            "snap_log10_distance": float(snapped["log10_distance"]),
            "best_eig": float(boed.get("best_eig", snapped.get("utility", float("nan")))),
            "best_cell_line": boed.get("best_cell_line"),
            "best_dose_mu": _optional_float(boed.get("best_dose_mu")),
            "best_bmp4_norm_mu": _optional_float(boed.get("best_bmp4_norm_mu")),
            "n_acquired_points": int(sum(len(values) for values in acquired_by_cell.values())),
            "candidate_dose_count": int(joint_data.bmp4_conc.shape[1]),
            **{
                f"collected_count_{name}": int(len(values))
                for name, values in acquired_by_cell.items()
            },
            **distance,
        }
        row.update(_per_cell_utility_columns(boed.get("best_per_cell") or []))
        rows.append(row)
    return rows


def _posterior_predictive_distance(
    *,
    module: Any,
    joint_data: Any,
    normalizer: Any,
    posterior_samples: dict[str, Any],
    acquired_by_cell: dict[str, list[int]],
    max_posterior_draws: int,
    rng: Any,
) -> dict[str, float]:
    raw_abs: list[Any] = []
    norm_abs: list[Any] = []
    for cell_index, cell_line_value in enumerate(joint_data.cell_lines):
        cell_line = str(cell_line_value)
        dose_indices = list(acquired_by_cell.get(cell_line) or [])
        if not dose_indices:
            continue
        theta_samples = np.asarray(posterior_samples[cell_line], dtype=np.float32)
        theta_samples = _choose_rows(theta_samples, max_posterior_draws, rng)
        doses = np.asarray(joint_data.bmp4_conc[cell_index, dose_indices], dtype=np.float32)
        mean_raw = _simulate_raw(module, cell_index, joint_data, doses, theta_samples)
        y_raw = module._sample_raw_observations_from_mean_raw(
            normalizer=normalizer,
            mean_raw=mean_raw,
            theta_norm=theta_samples,
            rng=rng,
        )
        y_norm = normalizer.normalize_response(y_raw)
        obs_raw = np.asarray(joint_data.x_obs[cell_index, dose_indices], dtype=np.float32).reshape(1, -1)
        obs_norm = np.asarray(joint_data.x_obs_norm[cell_index, dose_indices], dtype=np.float32).reshape(1, -1)
        raw_abs.append(np.abs(np.asarray(y_raw, dtype=np.float32) - obs_raw).reshape(-1))
        norm_abs.append(np.abs(np.asarray(y_norm, dtype=np.float32) - obs_norm).reshape(-1))

    if not raw_abs:
        return {
            "median_predictive_abs_error_raw": float("nan"),
            "q25_predictive_abs_error_raw": float("nan"),
            "q75_predictive_abs_error_raw": float("nan"),
            "median_predictive_abs_error_norm": float("nan"),
            "q25_predictive_abs_error_norm": float("nan"),
            "q75_predictive_abs_error_norm": float("nan"),
        }
    raw = np.concatenate(raw_abs)
    norm = np.concatenate(norm_abs)
    return {
        "median_predictive_abs_error_raw": float(np.median(raw)),
        "q25_predictive_abs_error_raw": float(np.quantile(raw, 0.25)),
        "q75_predictive_abs_error_raw": float(np.quantile(raw, 0.75)),
        "median_predictive_abs_error_norm": float(np.median(norm)),
        "q25_predictive_abs_error_norm": float(np.quantile(norm, 0.25)),
        "q75_predictive_abs_error_norm": float(np.quantile(norm, 0.75)),
    }


def _simulate_raw(module: Any, cell_index: int, joint_data: Any, doses: Any, theta_samples: Any) -> Any:
    if module is promisys_onestep:
        return module.simulate_promisys_onestep_raw(
            bmp4_concentrations=doses,
            receptors=np.asarray(joint_data.q_obs[cell_index], dtype=np.float32),
            theta_norm=theta_samples,
            receptor_noise_log_sd=0.0,
        )
    return module.simulate_promisys_twostep_raw(
        bmp4_concentrations=doses,
        receptors=np.asarray(joint_data.q_obs[cell_index], dtype=np.float32),
        theta_norm=theta_samples,
        receptor_noise_log_sd=0.0,
    )


def _posterior_path_for_step(run_path: Path, item: dict[str, Any], acquisition: int) -> Path:
    artifacts = item.get("artifacts") or {}
    path = artifacts.get("mcmc_posterior_samples_path")
    if path:
        return Path(path)
    return run_path / f"step_{acquisition:03d}" / "mcmc_posterior_samples.pt"


def _load_theta_samples(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    theta_norm = payload.get("theta_norm")
    if not isinstance(theta_norm, dict):
        raise ValueError(f"Posterior sample file missing theta_norm mapping: {path}")
    return {str(name): _tensor_to_numpy(value) for name, value in theta_norm.items()}


def _tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _choose_rows(samples: Any, max_rows: int, rng: Any) -> Any:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.shape[0] <= int(max_rows):
        return arr
    indices = rng.choice(arr.shape[0], size=int(max_rows), replace=False)
    return arr[indices]


def _per_cell_utility_columns(best_per_cell: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in best_per_cell:
        cell_line = str(item.get("cell_line"))
        if not cell_line or cell_line == "None":
            continue
        values[f"per_cell_utility_{cell_line}"] = _optional_float(item.get("utility"))
    return values


def _plot_coverage_heatmap(ax: Any, metrics: list[dict[str, Any]], cell_lines: list[str]) -> None:
    max_index = max(
        max(int(row["dose_index"]), int(row.get("candidate_dose_count", 1)) - 1)
        for row in metrics
    ) if metrics else 0
    matrix = np.full((len(cell_lines), max_index + 1), np.nan, dtype=float)
    row_index = {name: index for index, name in enumerate(cell_lines)}
    for row in metrics:
        matrix[row_index[row["selected_cell_line"]], int(row["dose_index"])] = float(row["acquisition"])
    masked = np.ma.masked_invalid(matrix)
    cmap = ax.figure.get_cmap("viridis").copy()
    cmap.set_bad("#f1f3f5")
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_yticks(range(len(cell_lines)))
    ax.set_yticklabels(cell_lines)
    ax.set_xticks(range(max_index + 1))
    ax.set_xticklabels([str(index) for index in range(max_index + 1)])
    ax.set_xlabel("Measured dose index")
    ax.set_ylabel("Cell line")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, str(int(matrix[y, x])), ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Design coverage (cell value = acquisition #)")
    ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)


def _cell_lines_from_metrics(metrics: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for row in metrics:
        cell_line = str(row["selected_cell_line"])
        if cell_line not in ordered:
            ordered.append(cell_line)
    return ordered


def _cell_line_colors(cell_lines: list[str]) -> dict[str, str]:
    palette = ["#005f73", "#ae2012", "#0a9396", "#ca6702", "#6a4c93", "#2f3e46"]
    return {cell_line: palette[index % len(palette)] for index, cell_line in enumerate(cell_lines)}


def _family_module(family_name: str) -> Any:
    try:
        return FAMILY_MODULES[family_name]
    except KeyError as exc:
        raise ValueError("Unsupported BMP4 Promisys family: " + str(family_name)) from exc


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)


def _save_figure(fig: Any, output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # pragma: no cover - best-effort cleanup
        pass
    return str(output)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_ready(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "analyze_comparison_run",
    "analyze_sequential_run",
    "plot_comparison_diagnostics",
    "plot_sequential_diagnostics",
    "plot_sequential_posterior_comparison",
    "plot_trace_acquisition_history",
]
