from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.cases.bmp4_gradient.data import build_joint_bmp4_gradient_data, load_bmp4_gradient_data


DEFAULT_CELL_LINES = ("NMuMG", "BMPR2_KD", "ACVR1_KD", "BMPR1A_KD")
DEFAULT_RUNS = {
    "no_prior": Path(
        "artifacts/bmp4_gradient/promisys_twostep/"
        "joint__NMuMG__BMPR2_KD__ACVR1_KD__BMPR1A_KD"
    ),
    "expert_prior": Path(
        "artifacts/bmp4_gradient/promisys_twostep/expert_prior/"
        "joint__NMuMG__BMPR2_KD__ACVR1_KD__BMPR1A_KD"
    ),
    "literature_prior": Path(
        "artifacts/bmp4_gradient/promisys_twostep/literature_prior/"
        "joint__NMuMG__BMPR2_KD__ACVR1_KD__BMPR1A_KD"
    ),
    "multireceptor_hierarchical_literature_prior": Path(
        "artifacts/bmp4_gradient/promisys_twostep/multireceptor_hierarchical_literature_prior/"
        "joint__NMuMG__BMPR2_KD__ACVR1_KD__BMPR1A_KD"
    ),
}


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must be LABEL=ARTIFACT_DIR")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, Path(path)


def _metrics(run: str, prediction: str, cell_line: str, error: np.ndarray) -> dict[str, float | int | str]:
    abs_error = np.abs(error)
    return {
        "run": run,
        "prediction": prediction,
        "cell_line": cell_line,
        "n": int(error.size),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_abs_error": float(np.median(abs_error)),
        "bias": float(error.mean()),
        "max_abs_error": float(abs_error.max()),
    }


def _summary_rows(
    rows: list[dict[str, float | int | str]],
    *,
    baseline_run: str,
) -> list[dict[str, float | int | str]]:
    all_rows = [
        row
        for row in rows
        if row["cell_line"] == "ALL"
    ]
    baseline_by_prediction = {
        str(row["prediction"]): row
        for row in all_rows
        if row["run"] == baseline_run
    }
    summary: list[dict[str, float | int | str]] = []
    for row in all_rows:
        item = dict(row)
        baseline = baseline_by_prediction.get(str(row["prediction"]))
        if baseline is not None:
            item["delta_mae_vs_baseline"] = float(row["mae"]) - float(baseline["mae"])
            item["delta_rmse_vs_baseline"] = float(row["rmse"]) - float(baseline["rmse"])
            item["mae_ratio_vs_baseline"] = _safe_ratio(float(row["mae"]), float(baseline["mae"]))
            item["rmse_ratio_vs_baseline"] = _safe_ratio(float(row["rmse"]), float(baseline["rmse"]))
        else:
            item["delta_mae_vs_baseline"] = float("nan")
            item["delta_rmse_vs_baseline"] = float("nan")
            item["mae_ratio_vs_baseline"] = float("nan")
            item["rmse_ratio_vs_baseline"] = float("nan")
        summary.append(item)
    return summary


def _safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return float("nan")
    return value / baseline


def _interpolate_to_observed_doses(
    *,
    grid_concentration: np.ndarray,
    draws: np.ndarray,
    observed_concentration: np.ndarray,
) -> np.ndarray:
    if grid_concentration.ndim == 1:
        grid_concentration = np.repeat(grid_concentration[None, :], observed_concentration.shape[0], axis=0)
    prediction = np.empty_like(observed_concentration, dtype=float)
    mean_curve = draws.mean(axis=0)
    for cell_index in range(observed_concentration.shape[0]):
        order = np.argsort(grid_concentration[cell_index])
        grid = grid_concentration[cell_index, order]
        curve = mean_curve[cell_index, order]
        prediction[cell_index] = np.interp(observed_concentration[cell_index], grid, curve)
    return prediction


def compare_prediction_errors(
    *,
    runs: dict[str, Path],
    cell_lines: tuple[str, ...],
    data_path: Path | None,
    baseline_run: str,
) -> dict[str, object]:
    joint = build_joint_bmp4_gradient_data(load_bmp4_gradient_data(data_path), cell_lines=cell_lines)
    observed_concentration = np.asarray(joint.bmp4_conc, dtype=float)
    observed_response = np.asarray(joint.x_obs, dtype=float)

    rows: list[dict[str, float | int | str]] = []
    payload: dict[str, object] = {
        "metric_scale": "raw response",
        "cell_lines": list(cell_lines),
        "baseline_run": baseline_run,
        "runs": {},
        "rows": rows,
        "summary_rows": [],
    }

    for label, root in runs.items():
        path = root / "posterior_predictive.pt"
        if not path.exists():
            payload["runs"][label] = {
                "artifact_dir": str(root),
                "missing_path": str(path),
            }
            continue
        posterior_predictive = torch.load(path, map_location="cpu")
        grid = np.asarray(posterior_predictive["grid_concentration"], dtype=float)
        run_predictions = {
            "posterior_predictive_y_mean": np.asarray(posterior_predictive["predictive_y"], dtype=float),
            "latent_mu_mean": np.asarray(posterior_predictive["predictive_mu"], dtype=float),
        }
        payload["runs"][label] = {"artifact_dir": str(root)}
        for prediction_name, draws in run_predictions.items():
            predicted = _interpolate_to_observed_doses(
                grid_concentration=grid,
                draws=draws,
                observed_concentration=observed_concentration,
            )
            error = predicted - observed_response
            rows.append(_metrics(label, prediction_name, "ALL", error))
            for cell_index, cell_line in enumerate(cell_lines):
                rows.append(_metrics(label, prediction_name, cell_line, error[cell_index]))

    payload["summary_rows"] = _summary_rows(rows, baseline_run=baseline_run)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BMP4 posterior predictive errors between twostep artifact directories."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        default=None,
        help="Run to compare as LABEL=ARTIFACT_DIR. Defaults to the known twostep no-prior/lit-prior dirs.",
    )
    parser.add_argument("--cell-line", action="append", default=None, help="Cell line to include.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--baseline-run",
        default="no_prior",
        help="Run label used for delta/ratio columns in the summary table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/bmp4_gradient/promisys_twostep"),
    )
    args = parser.parse_args()

    runs = dict(args.run) if args.run else DEFAULT_RUNS
    cell_lines = tuple(args.cell_line or DEFAULT_CELL_LINES)
    payload = compare_prediction_errors(
        runs=runs,
        cell_lines=cell_lines,
        data_path=args.data_path,
        baseline_run=args.baseline_run,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "prediction_error_comparison.json"
    tsv_path = args.output_dir / "prediction_error_comparison.tsv"
    summary_tsv_path = args.output_dir / "prediction_error_summary.tsv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "run",
            "prediction",
            "cell_line",
            "n",
            "mae",
            "rmse",
            "median_abs_error",
            "bias",
            "max_abs_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(payload["rows"])
    with summary_tsv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "run",
            "prediction",
            "cell_line",
            "n",
            "mae",
            "rmse",
            "median_abs_error",
            "bias",
            "max_abs_error",
            "delta_mae_vs_baseline",
            "delta_rmse_vs_baseline",
            "mae_ratio_vs_baseline",
            "rmse_ratio_vs_baseline",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(payload["summary_rows"])

    print(f"Wrote {tsv_path}")
    print(f"Wrote {summary_tsv_path}")
    print(f"Wrote {json_path}")
    print()
    for label, info in payload["runs"].items():
        if isinstance(info, dict) and info.get("missing_path"):
            print(f"Skipped {label}: missing {info['missing_path']}")
    if any(isinstance(info, dict) and info.get("missing_path") for info in payload["runs"].values()):
        print()
    print(f"ALL-cell metrics relative to {args.baseline_run}:")
    for row in payload["summary_rows"]:
        if row["cell_line"] == "ALL":
            print(
                f"{row['run']:45s} "
                f"{row['prediction']:28s} "
                f"MAE={row['mae']:.4g} "
                f"RMSE={row['rmse']:.4g} "
                f"dMAE={row['delta_mae_vs_baseline']:+.4g} "
                f"dRMSE={row['delta_rmse_vs_baseline']:+.4g} "
                f"medAE={row['median_abs_error']:.4g} "
                f"bias={row['bias']:.4g} "
                f"maxAE={row['max_abs_error']:.4g}"
            )


if __name__ == "__main__":
    main()
