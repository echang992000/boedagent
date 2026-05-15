from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.cases.bmp4_gradient.data import build_joint_bmp4_gradient_data, load_bmp4_gradient_data
from examples.cases.bmp4_gradient.promisys_twostep import (
    BIOPHYSICAL_THETA_DIM,
    DEFAULT_POSTERIOR_NET_ACTIVATION,
    DEFAULT_POSTERIOR_NET_HIDDEN_DIM,
    DEFAULT_POSTERIOR_NET_LAYERS,
    RAW_PARAMETER_HIGH,
    RAW_PARAMETER_LOW,
    THETA_DIM,
    THETA_NORMALIZATION_SCALE,
    GaussianMLP,
)


DEFAULT_RUNS = {
    "no_prior": Path(
        "artifacts/bmp4_gradient/promisys_twostep/"
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


def _load_posterior_net(run_dir: Path) -> GaussianMLP:
    with (run_dir / "likelihood_checkpoint.pkl").open("rb") as handle:
        checkpoint = pickle.load(handle)
    state = checkpoint["posterior_network"]
    network = GaussianMLP(
        int(state.get("input_dim", 29)),
        int(state.get("output_dim", THETA_DIM)),
        hidden_dim=int(state.get("hidden_dim", DEFAULT_POSTERIOR_NET_HIDDEN_DIM)),
        layers=int(state.get("layers", DEFAULT_POSTERIOR_NET_LAYERS)),
        activation=str(state.get("activation", DEFAULT_POSTERIOR_NET_ACTIVATION)),
    )
    network.model.load_state_dict(state["model_state_dict"])
    network.model.eval()
    return network


def _observed_feature_by_cell(
    *,
    cell_lines: tuple[str, ...],
    data_path: Path | None,
) -> dict[str, torch.Tensor]:
    joint = build_joint_bmp4_gradient_data(load_bmp4_gradient_data(data_path), cell_lines=cell_lines)
    features: dict[str, torch.Tensor] = {}
    for cell_index, cell_line in enumerate(joint.cell_lines):
        feature = np.concatenate(
            [
                np.asarray(joint.x_obs_norm[cell_index], dtype=np.float32),
                np.asarray(joint.Rs_norm[cell_index], dtype=np.float32),
                np.asarray(joint.bmp4_conc_norm[cell_index], dtype=np.float32),
            ]
        )
        features[str(cell_line)] = torch.as_tensor(feature[None, :], dtype=torch.float32)
    return features


def _log_prior_theta_norm(theta_norm: np.ndarray, theta_prior: dict[str, Any] | None) -> np.ndarray:
    theta = np.asarray(theta_norm, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.shape[-1] != THETA_DIM:
        raise ValueError(f"Expected theta dimension {THETA_DIM}, got {theta.shape[-1]}.")

    theta_prior = theta_prior or {}
    parameter_priors = theta_prior.get("parameter_priors") or [
        {"distribution": "LogUniform", "low": RAW_PARAMETER_LOW, "high": RAW_PARAMETER_HIGH}
        for _ in range(BIOPHYSICAL_THETA_DIM)
    ]
    if len(parameter_priors) != BIOPHYSICAL_THETA_DIM:
        raise ValueError(
            f"Expected {BIOPHYSICAL_THETA_DIM} biophysical prior entries, got {len(parameter_priors)}."
        )

    logits = np.clip(theta[:, :BIOPHYSICAL_THETA_DIM] * THETA_NORMALIZATION_SCALE, -40.0, 40.0)
    unit = 1.0 / (1.0 + np.exp(-logits))
    log_low = math.log(RAW_PARAMETER_LOW)
    log_high = math.log(RAW_PARAMETER_HIGH)
    default_delta = log_high - log_low

    log_prior = np.zeros(theta.shape[0], dtype=np.float64)
    for index, prior in enumerate(parameter_priors):
        low = float(prior.get("low", prior.get("raw_low", RAW_PARAMETER_LOW)))
        high = float(prior.get("high", prior.get("raw_high", RAW_PARAMETER_HIGH)))
        low = max(low, RAW_PARAMETER_LOW)
        high = max(high, low * (1.0 + 1e-12))
        parameter_log_low = math.log(low)
        parameter_log_high = math.log(high)
        parameter_delta = parameter_log_high - parameter_log_low

        raw = np.exp(log_low + default_delta * unit[:, index])
        raw = np.clip(raw, low, high)
        distribution = str(prior.get("distribution", "LogUniform")).replace("_", "").replace("-", "").lower()
        log_jacobian = (
            np.log(raw)
            + math.log(default_delta * THETA_NORMALIZATION_SCALE)
            + np.log(np.clip(unit[:, index], 1e-300, 1.0))
            + np.log(np.clip(1.0 - unit[:, index], 1e-300, 1.0))
        )
        if distribution == "lognormal":
            params = prior.get("params", {})
            loc = float(params.get("loc", params.get("mu", 0.0)))
            scale = max(float(params.get("scale", params.get("sigma", 1.0))), 1e-12)
            log_raw = np.log(np.clip(raw, 1e-300, None))
            log_raw_density = (
                -log_raw
                - math.log(scale)
                - 0.5 * math.log(2.0 * math.pi)
                - 0.5 * ((log_raw - loc) / scale) ** 2
            )
        elif distribution == "loguniform":
            log_raw_density = -np.log(np.clip(raw, 1e-300, None)) - math.log(parameter_delta)
        else:
            raise ValueError(f"Unsupported theta prior distribution: {prior.get('distribution')!r}")
        log_prior += log_raw_density + log_jacobian

    sigma_latent = theta[:, BIOPHYSICAL_THETA_DIM]
    log_prior += -0.5 * (sigma_latent**2 + math.log(2.0 * math.pi))
    return log_prior


def _summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "stderr": float(np.std(values) / math.sqrt(max(int(values.size), 1))),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.5)),
        "q95": float(np.quantile(values, 0.95)),
    }


def estimate_prior_eig(
    *,
    runs: dict[str, Path],
    cell_lines: tuple[str, ...],
    data_path: Path | None,
) -> dict[str, Any]:
    features_by_cell = _observed_feature_by_cell(cell_lines=cell_lines, data_path=data_path)
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "estimator": "mean over SNPE posterior samples of log q_SNPE(theta|observed) - log p_prior(theta)",
        "units": "nats",
        "rows": rows,
    }

    for run_label, run_dir in runs.items():
        sample_path = run_dir / "snpe_posterior_samples.pt"
        checkpoint_path = run_dir / "likelihood_checkpoint.pkl"
        if not sample_path.exists() or not checkpoint_path.exists():
            continue
        network = _load_posterior_net(run_dir)
        sample_payload = torch.load(sample_path, map_location="cpu")
        theta_by_cell = sample_payload["theta_norm"]
        theta_prior = sample_payload.get("theta_raw_prior")

        pooled: list[np.ndarray] = []
        for cell_line in cell_lines:
            if cell_line not in theta_by_cell or cell_line not in features_by_cell:
                continue
            theta = torch.as_tensor(theta_by_cell[cell_line], dtype=torch.float32)
            with torch.no_grad():
                posterior_dist = network.distribution(features_by_cell[cell_line])
                log_posterior = posterior_dist.log_prob(theta).sum(dim=-1).cpu().numpy()
            log_prior = _log_prior_theta_norm(theta.cpu().numpy(), theta_prior)
            log_ratio = log_posterior - log_prior
            pooled.append(log_ratio)
            row = {
                "run": run_label,
                "artifact_dir": str(run_dir),
                "cell_line": cell_line,
                **_summarize(log_ratio),
                "mean_log_posterior": float(np.mean(log_posterior)),
                "mean_log_prior": float(np.mean(log_prior)),
            }
            rows.append(row)
        if pooled:
            pooled_ratio = np.concatenate(pooled, axis=0)
            rows.append(
                {
                    "run": run_label,
                    "artifact_dir": str(run_dir),
                    "cell_line": "ALL",
                    **_summarize(pooled_ratio),
                    "mean_log_posterior": None,
                    "mean_log_prior": None,
                }
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate BMP4 prior information gain from SNPE posterior/prior density ratios."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        default=None,
        help="Run to evaluate as LABEL=ARTIFACT_DIR. Defaults to the known twostep artifact dirs.",
    )
    parser.add_argument(
        "--cell-line",
        action="append",
        default=None,
        help="Cell line to include. Defaults to NMuMG, BMPR2_KD, ACVR1_KD, BMPR1A_KD.",
    )
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/bmp4_gradient/promisys_twostep"),
    )
    args = parser.parse_args()

    runs = dict(args.run) if args.run else DEFAULT_RUNS
    cell_lines = tuple(args.cell_line or ("NMuMG", "BMPR2_KD", "ACVR1_KD", "BMPR1A_KD"))
    payload = estimate_prior_eig(runs=runs, cell_lines=cell_lines, data_path=args.data_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "prior_eig_estimate.json"
    tsv_path = args.output_dir / "prior_eig_estimate.tsv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "run",
            "artifact_dir",
            "cell_line",
            "n",
            "mean",
            "std",
            "stderr",
            "q05",
            "q50",
            "q95",
            "mean_log_posterior",
            "mean_log_prior",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(payload["rows"])

    print(f"Wrote {tsv_path}")
    print(f"Wrote {json_path}")
    print()
    print("Pooled prior EIG estimates in nats:")
    for row in payload["rows"]:
        if row["cell_line"] == "ALL":
            print(
                f"{row['run']:45s} "
                f"mean={row['mean']:.4g} "
                f"stderr={row['stderr']:.4g} "
                f"q50={row['q50']:.4g} "
                f"n={row['n']}"
            )


if __name__ == "__main__":
    main()
