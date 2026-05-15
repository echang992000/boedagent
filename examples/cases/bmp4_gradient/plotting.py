from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any


def save_posterior_predictive_plot(
    *,
    observed_concentration: Any,
    observed_response: Any,
    grid_concentration: Any,
    predictive_draws: Any,
    output_path: str | Path,
    title: str,
) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "BMP4 plotting requires `matplotlib` and `numpy` in the active environment."
        ) from exc

    observed_concentration = np.asarray(observed_concentration, dtype=float)
    observed_response = np.asarray(observed_response, dtype=float)
    grid_concentration = np.asarray(grid_concentration, dtype=float)
    predictive_draws = np.asarray(predictive_draws, dtype=float)

    mean = predictive_draws.mean(axis=0)
    low = np.quantile(predictive_draws, 0.05, axis=0)
    high = np.quantile(predictive_draws, 0.95, axis=0)

    fig, axis = plt.subplots(figsize=(8, 5))
    plot_obs, zero_proxy = _safe_positive_x(observed_concentration)
    plot_grid, _ = _safe_positive_x(grid_concentration, zero_proxy=zero_proxy)
    axis.scatter(plot_obs, observed_response, color="black", label="observed", zorder=3)
    axis.plot(plot_grid, mean, color="#005f73", linewidth=2.0, label="posterior mean")
    axis.fill_between(plot_grid, low, high, color="#94d2bd", alpha=0.35, label="90% interval")
    axis.set_xscale("log")
    axis.set_xlabel("BMP4 concentration (ng/mL)")
    axis.set_ylabel("Response")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    if zero_proxy is not None:
        axis.text(
            0.02,
            0.02,
            "Zero concentration points plotted at half the minimum positive dose.",
            transform=axis.transAxes,
            fontsize=8,
            alpha=0.7,
        )
    return _save_figure(fig, output_path)


def save_joint_posterior_predictive_plot(
    *,
    observed_concentration: Any,
    observed_response: Any,
    grid_concentration: Any,
    predictive_draws: Any,
    cell_lines: list[str] | tuple[str, ...],
    output_path: str | Path,
    title: str,
) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "BMP4 plotting requires `matplotlib` and `numpy` in the active environment."
        ) from exc

    observed_concentration = np.asarray(observed_concentration, dtype=float)
    observed_response = np.asarray(observed_response, dtype=float)
    grid_concentration = np.asarray(grid_concentration, dtype=float)
    predictive_draws = np.asarray(predictive_draws, dtype=float)
    if grid_concentration.ndim == 1:
        grid_concentration = np.repeat(grid_concentration[None, :], observed_concentration.shape[0], axis=0)

    n_cell_lines = len(cell_lines)
    n_cols = min(2, max(n_cell_lines, 1))
    n_rows = (n_cell_lines + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 4.5 * n_rows), squeeze=False)

    for index, cell_line in enumerate(cell_lines):
        axis = axes[index // n_cols][index % n_cols]
        draws = predictive_draws[:, index, :]
        mean = draws.mean(axis=0)
        low = np.quantile(draws, 0.05, axis=0)
        high = np.quantile(draws, 0.95, axis=0)

        plot_obs, zero_proxy = _safe_positive_x(observed_concentration[index])
        plot_grid, _ = _safe_positive_x(grid_concentration[index], zero_proxy=zero_proxy)
        axis.scatter(plot_obs, observed_response[index], color="black", label="observed", zorder=3)
        axis.plot(plot_grid, mean, color="#005f73", linewidth=2.0, label="posterior mean")
        axis.fill_between(plot_grid, low, high, color="#94d2bd", alpha=0.35, label="90% interval")
        axis.set_xscale("log")
        axis.set_title(cell_line)
        axis.set_xlabel("BMP4 concentration (ng/mL)")
        axis.set_ylabel("Response")
        axis.grid(True, alpha=0.3)
        if zero_proxy is not None:
            axis.text(
                0.02,
                0.02,
                "Zero concentration plotted at half the minimum positive dose.",
                transform=axis.transAxes,
                fontsize=8,
                alpha=0.7,
            )

    for index in range(n_cell_lines, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    return _save_figure(fig, output_path)


def save_eig_optimization_plot(
    history: list[dict[str, float]],
    output_path: str | Path,
    *,
    title: str,
) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("BMP4 plotting requires `matplotlib`.") from exc

    if history and "selector_probs" in history[0]:
        steps = [item["step"] for item in history]
        doses = [item["dose"] if item.get("dose") is not None else item["doses"][0] for item in history]
        eigs = [_eig_plot_value(item) for item in history]
        eig_baseline = _eig_plot_baseline(history)
        selector = np.asarray([item["selector_probs"] for item in history], dtype=float)
        cell_line_names = history[0].get("cell_line_names") or [
            f"cell_line_{index}" for index in range(selector.shape[1])
        ]

        fig = plt.figure(figsize=(8, 10))
        grid = fig.add_gridspec(3, 1, height_ratios=(1.15, 1.15, 1.25), hspace=0.22)
        dose_axis = fig.add_subplot(grid[0])
        selector_axis = fig.add_subplot(grid[1], sharex=dose_axis)
        dose_axis.plot(steps, doses, color="#0a9396", marker="o", linewidth=1.8)
        dose_axis.set_ylabel("Dose (ng/mL)")
        dose_axis.set_yscale("log")
        dose_axis.grid(True, alpha=0.3)
        plt.setp(dose_axis.get_xticklabels(), visible=False)

        for index, cell_line_name in enumerate(cell_line_names):
            selector_axis.plot(
                steps,
                selector[:, index],
                marker="o",
                linewidth=1.6,
                label=str(cell_line_name),
            )
        selector_axis.set_ylabel("Selector prob.")
        selector_axis.set_ylim(0.0, 1.05)
        selector_axis.grid(True, alpha=0.3)
        selector_axis.legend(loc="best")
        plt.setp(selector_axis.get_xticklabels(), visible=False)

        _add_eig_axis(
            fig,
            grid[2],
            sharex_axis=dose_axis,
            steps=steps,
            eigs=eigs,
            baseline=eig_baseline,
            xlabel="Optimization step",
        )
        fig.suptitle(title)
        return _save_figure(fig, output_path, tight_layout=False)

    steps = [item["step"] for item in history]
    designs = [item["design"] for item in history]
    eigs = [_eig_plot_value(item) for item in history]
    eig_baseline = _eig_plot_baseline(history)

    fig = plt.figure(figsize=(8, 7))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.1), hspace=0.18)
    design_axis = fig.add_subplot(grid[0])
    design_axis.plot(steps, designs, color="#0a9396", marker="o", linewidth=1.8)
    design_axis.set_ylabel("Design (ng/mL)")
    design_axis.set_xscale("linear")
    design_axis.grid(True, alpha=0.3)
    plt.setp(design_axis.get_xticklabels(), visible=False)

    _add_eig_axis(
        fig,
        grid[1],
        sharex_axis=design_axis,
        steps=steps,
        eigs=eigs,
        baseline=eig_baseline,
        xlabel="Optimization step",
    )
    fig.suptitle(title)
    return _save_figure(fig, output_path, tight_layout=False)


def _eig_plot_value(item: dict[str, Any]) -> float:
    return float(item.get("estimated_total_eig", item["eig"]))


def _eig_plot_baseline(history: list[dict[str, Any]]) -> float | None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("BMP4 plotting requires `numpy`.") from exc

    candidates = [
        float(item["prior_eig_baseline"])
        for item in history
        if item.get("prior_eig_baseline") is not None
    ]
    if not candidates:
        return None
    finite = np.asarray(candidates, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite[0])


def _add_eig_axis(
    fig: Any,
    spec: Any,
    *,
    sharex_axis: Any,
    steps: list[float],
    eigs: list[float],
    baseline: float | None,
    xlabel: str,
) -> Any:
    ranges = _eig_axis_break_ranges(eigs, baseline=baseline)
    if ranges is None:
        axis = fig.add_subplot(spec, sharex=sharex_axis)
        _plot_positive_eig(axis, steps, eigs, baseline=baseline)
        axis.set_xlabel(xlabel)
        return axis

    lower_range, upper_range = ranges
    subgrid = spec.subgridspec(2, 1, height_ratios=(3.2, 1.0), hspace=0.05)
    upper_axis = fig.add_subplot(subgrid[0], sharex=sharex_axis)
    lower_axis = fig.add_subplot(subgrid[1], sharex=sharex_axis)
    _plot_positive_eig(
        upper_axis,
        steps,
        eigs,
        baseline=baseline,
        ylim=upper_range,
        show_omitted_notice=False,
    )
    _plot_positive_eig(
        lower_axis,
        steps,
        eigs,
        baseline=baseline,
        ylim=lower_range,
        ylabel=False,
    )
    upper_axis.spines["bottom"].set_visible(False)
    lower_axis.spines["top"].set_visible(False)
    upper_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    lower_axis.set_xlabel(xlabel)
    _draw_axis_break_marks(upper_axis, lower_axis)
    return lower_axis


def _eig_axis_break_ranges(
    eigs: list[float],
    *,
    baseline: float | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("BMP4 plotting requires `numpy`.") from exc

    values = np.asarray(eigs, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size < 3:
        return None
    lowest = float(np.min(positive))
    tolerance = max(abs(lowest) * 1e-6, 1e-9)
    upper_values = positive[positive > lowest + tolerance]
    if upper_values.size < 2:
        return None

    upper_low = float(np.min(upper_values))
    upper_high = float(np.max(upper_values))
    upper_span = max(upper_high - upper_low, abs(upper_high) * 0.005, 1e-6)
    gap = upper_low - lowest
    if gap <= max(2.0 * upper_span, abs(upper_low) * 0.02):
        return None

    lower_pad = max(gap * 0.035, abs(lowest) * 0.003, 1e-6)
    upper_pad = max(upper_span * 0.12, abs(upper_high) * 0.002, 1e-6)
    lower_range = (
        max(np.nextafter(0.0, 1.0), lowest - lower_pad),
        lowest + lower_pad,
    )
    upper_floor = upper_low - upper_pad
    if baseline is not None and np.isfinite(float(baseline)) and float(baseline) > 0.0:
        baseline_value = float(baseline)
        if baseline_value >= upper_low - gap * 0.15:
            upper_floor = min(upper_floor, baseline_value - upper_pad)
    upper_range = (
        max(np.nextafter(0.0, 1.0), upper_floor),
        upper_high + upper_pad,
    )
    if lower_range[1] >= upper_range[0]:
        return None
    return lower_range, upper_range


def _draw_axis_break_marks(upper_axis: Any, lower_axis: Any) -> None:
    diagonal = 0.012
    kwargs = {"color": "black", "clip_on": False, "linewidth": 1.0}
    upper_axis.plot((-diagonal, +diagonal), (-diagonal, +diagonal), transform=upper_axis.transAxes, **kwargs)
    upper_axis.plot((1 - diagonal, 1 + diagonal), (-diagonal, +diagonal), transform=upper_axis.transAxes, **kwargs)
    lower_axis.plot((-diagonal, +diagonal), (1 - diagonal, 1 + diagonal), transform=lower_axis.transAxes, **kwargs)
    lower_axis.plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), transform=lower_axis.transAxes, **kwargs)


def _plot_positive_eig(
    axis: Any,
    steps: list[float],
    eigs: list[float],
    *,
    baseline: float | None = None,
    ylim: tuple[float, float] | None = None,
    ylabel: bool = True,
    show_omitted_notice: bool = True,
) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("BMP4 plotting requires `numpy`.") from exc

    steps_arr = np.asarray(steps, dtype=float)
    eigs_arr = np.asarray(eigs, dtype=float)
    positive = np.isfinite(eigs_arr) & (eigs_arr > 0.0)
    if np.any(positive):
        axis.plot(
            steps_arr[positive],
            eigs_arr[positive],
            color="#ae2012",
            marker="o",
            linewidth=1.8,
        )
        if baseline is not None and np.isfinite(float(baseline)) and float(baseline) > 0.0:
            axis.axhline(float(baseline), color="#6c757d", linestyle="--", linewidth=1.0, alpha=0.65)
        if ylim is not None:
            axis.set_ylim(*ylim)
        else:
            upper = float(np.max(eigs_arr[positive]))
            lower = float(np.min(eigs_arr[positive]))
            if baseline is not None and np.isfinite(float(baseline)) and float(baseline) > 0.0:
                baseline_value = float(baseline)
                min_positive = float(np.min(eigs_arr[positive]))
                lower = baseline_value if min_positive >= baseline_value else min_positive
            span = max(upper - lower, abs(upper) * 0.02, 1e-6)
            axis.set_ylim(max(0.0, lower - span * 0.05), upper + max(span * 0.08, 1e-6))
    else:
        axis.text(
            0.5,
            0.5,
            "No positive EIG values",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_ylim(1e-3, 1.0)
    omitted = int(np.size(eigs_arr) - int(np.sum(positive)))
    if omitted and show_omitted_notice:
        axis.text(
            0.02,
            0.04,
            f"{omitted} non-positive EIG point{'s' if omitted != 1 else ''} omitted",
            transform=axis.transAxes,
            fontsize=8,
            alpha=0.75,
        )
    axis.set_yscale("linear")
    if ylabel:
        axis.set_ylabel("Estimated EIG")
    axis.grid(True, alpha=0.3)


def save_sequential_acquisition_plot(
    trace: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
) -> str:
    """Plot one final BOED result per actually acquired sequential experiment."""
    if not trace:
        raise ValueError("No sequential acquisitions to plot.")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(trace):
        snapped = dict(item.get("snapped_design") or {})
        boed = dict(item.get("boed") or {})
        if not snapped:
            continue
        rows.append(
            {
                "acquisition": int(item.get("acquisition", index + 1)),
                "cell_line": str(snapped.get("cell_line", "")),
                "dose": float(snapped.get("dose", float("nan"))),
                "dose_index": int(snapped.get("dose_index", -1)),
                "eig": float(boed.get("best_eig", snapped.get("utility", float("nan")))),
            }
        )
    if not rows:
        raise ValueError("Sequential trace contains no snapped designs to plot.")

    cell_lines = list(dict.fromkeys(row["cell_line"] for row in rows))
    palette = ["#005f73", "#ae2012", "#0a9396", "#ca6702", "#6a4c93", "#2f3e46"]
    colors = {cell_line: palette[index % len(palette)] for index, cell_line in enumerate(cell_lines)}

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:  # pragma: no cover - optional dependency fallback
        return _save_sequential_acquisition_svg(rows, output_path, title=title, colors=colors)

    x = np.asarray([row["acquisition"] for row in rows], dtype=float)
    eig = np.asarray([row["eig"] for row in rows], dtype=float)
    log10_dose = np.log10(np.clip(np.asarray([row["dose"] for row in rows], dtype=float), 1e-30, None))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    for cell_line in cell_lines:
        mask = np.asarray([row["cell_line"] == cell_line for row in rows], dtype=bool)
        axes[0].scatter(x[mask], eig[mask], color=colors[cell_line], s=78, label=cell_line, zorder=3)
        axes[1].scatter(x[mask], log10_dose[mask], color=colors[cell_line], s=78, label=cell_line, zorder=3)

    axes[0].set_ylabel("Final EIG")
    axes[0].set_title("Final EIG after each acquisition optimization")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", ncols=min(len(cell_lines), 4))

    axes[1].set_ylabel("Final chosen design\nlog10(BMP4 ng/mL)")
    axes[1].set_xlabel("Acquisition")
    axes[1].set_title("Actual snapped experimental design")
    if len(log10_dose):
        log_low = math.floor(float(np.nanmin(log10_dose)))
        log_high = math.ceil(float(np.nanmax(log10_dose)))
        axes[1].set_yticks(list(range(log_low, log_high + 1)))
    axes[1].grid(True, alpha=0.25)
    for row in rows:
        axes[1].annotate(
            str(row["dose_index"]),
            (row["acquisition"], math.log10(max(float(row["dose"]), 1e-30))),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#2f3e46",
        )

    fig.suptitle(title)
    return _save_figure(fig, output_path)


def _save_sequential_acquisition_svg(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
    colors: dict[str, str],
) -> str:
    output = Path(output_path)
    if output.suffix.lower() != ".svg":
        output = output.with_suffix(".svg")
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 980, 760
    left, right = 92, 32
    top = 82
    panel_h = 250
    gap = 78
    plot_w = width - left - right
    x_values = [float(row["acquisition"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    eig_values = [float(row["eig"]) for row in rows if math.isfinite(float(row["eig"]))]
    dose_values = [float(row["dose"]) for row in rows if float(row["dose"]) > 0.0]
    eig_min, eig_max = _expanded_range(eig_values)
    log_values = [math.log10(value) for value in dose_values]
    log_min = math.floor(min(log_values)) if log_values else -4.0
    log_max = math.ceil(max(log_values)) if log_values else 4.0
    if log_min == log_max:
        log_min -= 1.0
        log_max += 1.0

    def sx(value: float) -> float:
        return left + (float(value) - x_min) / (x_max - x_min) * plot_w

    def sy(value: float, y_min: float, y_max: float, panel_top: float) -> float:
        return panel_top + panel_h - (float(value) - y_min) / (y_max - y_min) * panel_h

    eig_top = top
    dose_top = top + panel_h + gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>',
    ]
    parts.extend(_svg_axes(left, eig_top, plot_w, panel_h, "Final EIG", "Final EIG after each acquisition optimization"))
    parts.extend(_svg_y_ticks(left, eig_top, plot_w, panel_h, eig_min, eig_max, _format_svg_number))
    parts.extend(
        _svg_x_ticks(
            left,
            eig_top,
            plot_w,
            panel_h,
            x_min,
            x_max,
            lambda value: str(int(round(value))),
        )
    )
    parts.extend(_svg_axes(left, dose_top, plot_w, panel_h, "Final chosen design log10(BMP4 ng/mL)", "Actual snapped experimental design"))
    parts.extend(
        _svg_y_ticks(
            left,
            dose_top,
            plot_w,
            panel_h,
            log_min,
            log_max,
            _format_svg_number,
            values=list(range(int(log_min), int(log_max) + 1)),
        )
    )
    parts.extend(
        _svg_x_ticks(
            left,
            dose_top,
            plot_w,
            panel_h,
            x_min,
            x_max,
            lambda value: str(int(round(value))),
        )
    )

    for row in rows:
        x = sx(float(row["acquisition"]))
        eig_y = sy(float(row["eig"]), eig_min, eig_max, eig_top)
        log_dose = math.log10(max(float(row["dose"]), 1e-30))
        dose_y = sy(log_dose, log_min, log_max, dose_top)
        color = colors.get(str(row["cell_line"]), "#2f3e46")
        parts.append(f'<circle cx="{x:.2f}" cy="{eig_y:.2f}" r="6" fill="{color}"/>')
        parts.append(f'<circle cx="{x:.2f}" cy="{dose_y:.2f}" r="6" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 8:.2f}" y="{dose_y - 8:.2f}" font-family="Arial" font-size="11" fill="#2f3e46">'
            f'{int(row["dose_index"])}</text>'
        )

    legend_x = left
    legend_y = height - 42
    for index, (cell_line, color) in enumerate(colors.items()):
        x = legend_x + index * 150
        parts.append(f'<circle cx="{x:.2f}" cy="{legend_y:.2f}" r="6" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 12:.2f}" y="{legend_y + 4:.2f}" font-family="Arial" font-size="12" fill="#2f3e46">'
            f'{html.escape(cell_line)}</text>'
        )

    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return str(output)


def _expanded_range(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    low, high = min(finite), max(finite)
    if low == high:
        pad = max(abs(low) * 0.05, 0.05)
    else:
        pad = (high - low) * 0.08
    return low - pad, high + pad


def _svg_axes(left: int, top: int, width: int, height: int, ylabel: str, title: str) -> list[str]:
    bottom = top + height
    return [
        f'<text x="{left + width / 2:.1f}" y="{top - 18}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="#2f3e46">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{left + width}" y2="{bottom}" stroke="#495057" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#495057" stroke-width="1"/>',
        f'<text x="{left + width / 2:.1f}" y="{bottom + 38}" text-anchor="middle" font-family="Arial" font-size="13" fill="#2f3e46">Acquisition</text>',
        f'<text x="{left - 58}" y="{top + height / 2:.1f}" transform="rotate(-90 {left - 58} {top + height / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="13" fill="#2f3e46">{html.escape(ylabel)}</text>',
    ]


def _svg_y_ticks(
    left: int,
    top: int,
    width: int,
    height: int,
    low: float,
    high: float,
    formatter: Any,
    *,
    count: int = 5,
    values: list[float] | None = None,
) -> list[str]:
    tick_values = values
    if tick_values is None:
        if high == low:
            tick_values = [low]
        else:
            tick_values = [low + (high - low) * index / max(count - 1, 1) for index in range(count)]
    parts: list[str] = []
    for value in tick_values:
        y = top + height - (value - low) / (high - low if high != low else 1.0) * height
        parts.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#495057" stroke-width="1"/>')
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}" stroke="#d8dee4" stroke-width="0.6" opacity="0.55"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11" fill="#495057">'
            f'{html.escape(str(formatter(value)))}</text>'
        )
    return parts


def _svg_x_ticks(
    left: int,
    top: int,
    width: int,
    height: int,
    low: float,
    high: float,
    formatter: Any,
    *,
    count: int = 6,
) -> list[str]:
    if high == low:
        values = [low]
    else:
        values = [low + (high - low) * index / max(count - 1, 1) for index in range(count)]
    parts: list[str] = []
    bottom = top + height
    used_labels: set[str] = set()
    for value in values:
        label = str(formatter(value))
        if label in used_labels:
            continue
        used_labels.add(label)
        x = left + (value - low) / (high - low if high != low else 1.0) * width
        parts.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 5}" stroke="#495057" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{bottom + 20}" text-anchor="middle" font-family="Arial" font-size="11" fill="#495057">'
            f'{html.escape(label)}</text>'
        )
    return parts


def _format_svg_number(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "nan"
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude < 0.01 or magnitude >= 10000.0):
        return f"{number:.2e}"
    if magnitude < 1.0:
        return f"{number:.3g}"
    if magnitude < 100.0:
        return f"{number:.3g}"
    return f"{number:.0f}"


def save_prior_posterior_comparison_plot(
    *,
    translated_prior: Any,
    posterior_samples: dict[str, Any],
    output_path: str | Path,
    title: str,
    cell_lines: list[str] | tuple[str, ...] | None = None,
    receptor_names: list[str] | tuple[str, ...] | None = None,
    kd_prior_shift: Any | None = None,
    scale: str = "native",
) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("BMP4 plotting requires `matplotlib` and `numpy`.") from exc

    prior_samples = _build_prior_samples_for_comparison(
        translated_prior=translated_prior,
        posterior_samples=posterior_samples,
        kd_prior_shift=kd_prior_shift,
    )
    records = _build_prior_posterior_records(
        prior_samples=prior_samples,
        posterior_samples=posterior_samples,
        translated_prior=translated_prior,
        cell_lines=cell_lines,
        receptor_names=receptor_names,
    )
    transformed_records = [
        transformed
        for transformed in (
            _transform_prior_posterior_record(record, scale=scale)
            for record in records
        )
        if transformed is not None
    ]
    if not transformed_records:
        raise ValueError("No overlapping prior/posterior parameter samples available for comparison plotting.")

    n_panels = len(transformed_records)
    n_cols = min(4, max(1, n_panels))
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.3 * n_cols, 2.7 * n_rows),
        squeeze=False,
    )

    for index, record in enumerate(transformed_records):
        axis = axes[index // n_cols][index % n_cols]
        prior = np.asarray(record["prior"], dtype=float)
        posterior = np.asarray(record["posterior"], dtype=float)
        combined = np.concatenate([prior.reshape(-1), posterior.reshape(-1)])
        finite = combined[np.isfinite(combined)]
        if finite.size == 0:
            axis.set_visible(False)
            continue

        low = float(np.quantile(finite, 0.005))
        high = float(np.quantile(finite, 0.995))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            center = float(np.median(finite))
            low = center - 0.5
            high = center + 0.5
        plot_range = (low, high)
        use_log_x = scale == "positive" and _is_binding_affinity_site(str(record["site_name"]))
        if use_log_x:
            plot_range = (1e-4, 1e2)

        _plot_histogram(axis, prior, plot_range=plot_range, color="#e9c46a", log_x=use_log_x)
        _plot_histogram(axis, posterior, plot_range=plot_range, color="#0a9396", log_x=use_log_x)
        prior_finite = prior[np.isfinite(prior)]
        if prior_finite.size > 0:
            axis.axvline(
                float(np.mean(prior_finite)),
                color="#111111",
                linewidth=1.5,
                linestyle="--",
            )
        if use_log_x:
            axis.set_xscale("log")
            axis.set_xlim(1e-4, 1e2)
        axis.set_title(record["label"], fontsize=9)
        axis.grid(True, alpha=0.25)
        axis.tick_params(labelsize=8)

    for index in range(n_panels, n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    legend_handles = [
        Patch(facecolor="#e9c46a", alpha=0.45, label="prior"),
        Patch(facecolor="#0a9396", alpha=0.45, label="posterior"),
        Line2D([0], [0], color="#111111", linewidth=1.5, linestyle="--", label="prior mean/value"),
    ]
    fig.legend(handles=legend_handles, loc="upper right")
    return _save_figure(fig, output_path)


def _safe_positive_x(values: Any, *, zero_proxy: float | None = None) -> tuple[Any, float | None]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    positive = arr[arr > 0]
    if positive.size == 0:
        return arr, zero_proxy
    if zero_proxy is None:
        zero_proxy = float(positive.min()) * 0.5
    adjusted = np.where(arr > 0, arr, zero_proxy)
    return adjusted, zero_proxy


def _plot_histogram(
    axis: Any,
    values: Any,
    *,
    plot_range: tuple[float, float],
    color: str,
    log_x: bool = False,
) -> None:
    import numpy as np

    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    if np.allclose(arr, arr[0]):
        return
    if log_x:
        arr = arr[arr > 0]
        if arr.size == 0:
            return
        bins = np.logspace(np.log10(plot_range[0]), np.log10(plot_range[1]), 29)
    else:
        bins = 28
    axis.hist(
        arr,
        bins=bins,
        range=None if log_x else plot_range,
        density=True,
        alpha=0.45,
        color=color,
    )


def _build_prior_samples_for_comparison(
    *,
    translated_prior: Any,
    posterior_samples: dict[str, Any],
    kd_prior_shift: Any | None,
) -> dict[str, Any]:
    import torch

    from .priors import make_distribution

    prior_samples: dict[str, Any] = {}
    for site_name, config in translated_prior.sites.items():
        if site_name not in posterior_samples:
            continue
        target = torch.as_tensor(posterior_samples[site_name], dtype=torch.float32)
        prior_dist = make_distribution(config)
        batch_shape = tuple(prior_dist.batch_shape)
        target_shape = tuple(target.shape)
        if batch_shape and len(target_shape) >= len(batch_shape) and target_shape[-len(batch_shape):] == batch_shape:
            sample_shape = target_shape[:-len(batch_shape)]
        else:
            sample_shape = target_shape
        prior_samples[site_name] = prior_dist.sample(sample_shape).detach().cpu()

    if (
        "log_R" in posterior_samples
        and kd_prior_shift is not None
        and {"base_log_R", "sigma_R"}.issubset(prior_samples)
    ):
        posterior_log_r = torch.as_tensor(posterior_samples["log_R"], dtype=torch.float32)
        sample_count, cell_count, receptor_count = posterior_log_r.shape
        base_log_r = torch.as_tensor(prior_samples["base_log_R"], dtype=torch.float32).reshape(sample_count, receptor_count)
        sigma_r = torch.as_tensor(prior_samples["sigma_R"], dtype=torch.float32).reshape(sample_count, 1, 1)
        shift = torch.as_tensor(kd_prior_shift, dtype=torch.float32).reshape(1, cell_count, receptor_count)
        mean_log_r = base_log_r[:, None, :] + shift
        prior_samples["log_R"] = torch.distributions.Normal(mean_log_r, sigma_r).sample().detach().cpu()

    return prior_samples


def _build_prior_posterior_records(
    *,
    prior_samples: dict[str, Any],
    posterior_samples: dict[str, Any],
    translated_prior: Any,
    cell_lines: list[str] | tuple[str, ...] | None,
    receptor_names: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    ordered_sites = [name for name in translated_prior.sites if name in prior_samples and name in posterior_samples]
    if "log_R" in prior_samples and "log_R" in posterior_samples:
        ordered_sites.append("log_R")

    records: list[dict[str, Any]] = []
    for site_name in ordered_sites:
        prior_tensor = prior_samples[site_name]
        posterior_tensor = posterior_samples[site_name]
        site_labels = _site_labels(
            site_name,
            prior_tensor,
            cell_lines=cell_lines,
            receptor_names=receptor_names,
        )
        for index, label in site_labels:
            records.append(
                {
                    "site_name": site_name,
                    "distribution": None
                    if translated_prior.sites.get(site_name) is None
                    else translated_prior.sites[site_name].distribution,
                    "label": label,
                    "prior": _extract_indexed_series(prior_tensor, index),
                    "posterior": _extract_indexed_series(posterior_tensor, index),
                }
            )
    return records


def _transform_prior_posterior_record(record: dict[str, Any], *, scale: str) -> dict[str, Any] | None:
    import numpy as np

    if scale == "native":
        return record
    if scale != "positive":
        raise ValueError(f"Unsupported comparison-plot scale {scale!r}.")

    site_name = str(record["site_name"])
    distribution = "" if record.get("distribution") is None else str(record["distribution"]).lower()
    transform = _positive_transform(site_name, distribution)
    if transform is None:
        return None

    return {
        **record,
        "label": _positive_scale_label(site_name, str(record["label"])),
        "prior": transform(np.asarray(record["prior"], dtype=float)),
        "posterior": transform(np.asarray(record["posterior"], dtype=float)),
    }


def _positive_transform(site_name: str, distribution: str) -> Any | None:
    import numpy as np

    if site_name in {"log_kd", "log_weight", "log_s50"}:
        return lambda values: np.exp(values)
    if site_name in {"base_log_R", "log_R"}:
        return lambda values: np.clip(np.exp(values), 0.0, 5.0)
    if distribution in {"lognormal", "gamma", "beta", "truncatedlognormal"}:
        return lambda values: values
    return None


def _positive_scale_label(site_name: str, label: str) -> str:
    descriptive_replacements = {
        "Log binding affinity": "Binding affinity",
        "Log receptor signaling weight": "Receptor signaling weight",
        "Log baseline receptor abundance": "Baseline receptor abundance",
        "Log receptor abundance": "Receptor abundance",
        "Log half-signal level": "Half-signal level",
    }
    for source, target in descriptive_replacements.items():
        if label.startswith(source):
            return target + label[len(source):]

    replacements = {
        "log_kd": "kd",
        "log_weight": "weight",
        "base_log_R": "base_R",
        "log_s50": "s50",
        "log_R": "R",
    }
    replacement = replacements.get(site_name)
    if replacement is None:
        return label
    return replacement + label[len(site_name):]


def _site_labels(
    site_name: str,
    samples: Any,
    *,
    cell_lines: list[str] | tuple[str, ...] | None,
    receptor_names: list[str] | tuple[str, ...] | None,
) -> list[tuple[tuple[int, ...] | None, str]]:
    tensor = _normalize_site_tensor(samples)
    trailing_shape = tuple(tensor.shape[1:])
    if not trailing_shape:
        return [(None, _pretty_site_label(site_name))]

    if (
        site_name in {"log_kd", "log_weight", "base_log_R", "qpcr_intercept", "qpcr_slope", "sigma_q"}
        and len(trailing_shape) == 1
        and receptor_names
        and trailing_shape[0] == len(receptor_names)
    ):
        return [((idx,), _pretty_site_label(site_name, receptor_name=name)) for idx, name in enumerate(receptor_names)]

    if (
        site_name in {"bottom", "top", "sigma_y"}
        and len(trailing_shape) == 1
        and cell_lines
        and trailing_shape[0] == len(cell_lines)
    ):
        return [((idx,), _pretty_site_label(site_name, cell_line=name)) for idx, name in enumerate(cell_lines)]

    if (
        site_name == "log_R"
        and len(trailing_shape) == 2
        and cell_lines
        and receptor_names
        and trailing_shape == (len(cell_lines), len(receptor_names))
    ):
        labels: list[tuple[tuple[int, ...], str]] = []
        for cell_idx, cell_name in enumerate(cell_lines):
            for receptor_idx, receptor_name in enumerate(receptor_names):
                labels.append(
                    (
                        (cell_idx, receptor_idx),
                        _pretty_site_label(site_name, cell_line=cell_name, receptor_name=receptor_name),
                    )
                )
        return labels

    labels = []
    for index in _ndindex(trailing_shape):
        labels.append((index, _pretty_site_label(site_name, fallback_index=index)))
    return labels


def _extract_indexed_series(samples: Any, index: tuple[int, ...] | None) -> Any:
    tensor = _normalize_site_tensor(samples)
    if index is None:
        return tensor.reshape(-1).detach().cpu().numpy()
    slices = (slice(None),) + index
    return tensor[slices].reshape(-1).detach().cpu().numpy()


def _normalize_site_tensor(samples: Any) -> Any:
    import torch

    tensor = torch.as_tensor(samples, dtype=torch.float32)
    if tensor.ndim <= 1:
        return tensor
    sample_dim = tensor.shape[0]
    trailing = [size for size in tensor.shape[1:] if size != 1]
    if not trailing:
        return tensor.reshape(sample_dim)
    return tensor.reshape((sample_dim, *trailing))


def _pretty_site_label(
    site_name: str,
    *,
    receptor_name: str | None = None,
    cell_line: str | None = None,
    fallback_index: tuple[int, ...] | None = None,
) -> str:
    ligand = "BMP4"
    if site_name == "log_kd" and receptor_name is not None:
        return f"Log binding affinity (Kd): {ligand}-{receptor_name}"
    if site_name.startswith("kd_"):
        return f"Binding affinity (Kd): {ligand}-{site_name.removeprefix('kd_')}"
    if site_name == "log_weight" and receptor_name is not None:
        return f"Log receptor signaling weight: {ligand}-{receptor_name}"
    if site_name.startswith("weight_"):
        return f"Receptor signaling weight: {ligand}-{site_name.removeprefix('weight_')}"
    if site_name == "base_log_R" and receptor_name is not None:
        return f"Log baseline receptor abundance: {receptor_name}"
    if site_name.startswith("abundance_"):
        return f"Receptor abundance prior: {site_name.removeprefix('abundance_')}"
    if site_name == "log_R" and receptor_name is not None and cell_line is not None:
        return f"Log receptor abundance: {cell_line} / {receptor_name}"
    if site_name == "qpcr_intercept" and receptor_name is not None:
        return f"qPCR intercept: {receptor_name}"
    if site_name == "qpcr_slope" and receptor_name is not None:
        return f"qPCR slope: {receptor_name}"
    if site_name == "sigma_q" and receptor_name is not None:
        return f"qPCR noise scale: {receptor_name}"
    if site_name == "bottom" and cell_line is not None:
        return f"Baseline response: {cell_line}"
    if site_name == "top" and cell_line is not None:
        return f"Max response: {cell_line}"
    if site_name == "sigma_y" and cell_line is not None:
        return f"Response noise scale: {cell_line}"
    if site_name == "sigma_R":
        return "Cell-line receptor abundance dispersion"
    if site_name == "log_s50":
        return "Log half-signal level (s50)"
    if site_name == "s50":
        return "Half-signal level (s50)"
    if site_name == "response_hill":
        return "Response Hill coefficient"
    if site_name == "ec50":
        return "Half-maximal concentration (EC50)"
    if site_name == "hill_n":
        return "Hill coefficient"
    if site_name == "sigma":
        return "Observation noise scale"
    if fallback_index is not None:
        suffix = ",".join(str(item) for item in fallback_index)
        return f"{site_name}[{suffix}]"
    return site_name


def _is_binding_affinity_site(site_name: str) -> bool:
    return site_name == "log_kd" or site_name.startswith("kd_")


def _ndindex(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    import numpy as np

    return [tuple(int(item) for item in index) for index in np.ndindex(shape)]


def _save_figure(fig: Any, output_path: str | Path, *, tight_layout: bool = True) -> str:
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.clf()
    plt.close(fig)
    return str(output)


__all__ = [
    "save_eig_optimization_plot",
    "save_joint_posterior_predictive_plot",
    "save_prior_posterior_comparison_plot",
    "save_posterior_predictive_plot",
    "save_sequential_acquisition_plot",
]
