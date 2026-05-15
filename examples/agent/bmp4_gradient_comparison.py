"""Compare all-data and sequential BMP4 Promisys BOED workflows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from boed_agent.prior_builder import AugmentedPrior, DistributionSpec  # noqa: E402
from examples.cases.bmp4_gradient.data import (  # noqa: E402
    build_joint_bmp4_gradient_data,
    load_bmp4_gradient_data,
)
from examples.cases.bmp4_gradient.promisys_hyperparams import (  # noqa: E402
    coerce_promisys_hyperparams,
)
from examples.cases.bmp4_gradient.promisys_sequential import (  # noqa: E402
    run_promisys_sequential_workflow,
)


DEFAULT_PROBLEM_PATH = REPO_ROOT / "examples" / "cases" / "bmp4_gradient" / "problem.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "bmp4_gradient" / "comparisons"
DEFAULT_LITERATURE_PRIOR = (
    REPO_ROOT
    / "artifacts"
    / "bmp4_gradient"
    / "multireceptor_hierarchical"
    / "joint__NMuMG__BMPR2_KD__ACVR1_KD__BMPR1A_KD"
    / "literature_prior.json"
)


def run_bmp4_gradient_example(**kwargs: Any) -> dict[str, Any]:
    from examples.agent.bmp4_gradient_agent import run_bmp4_gradient_example as _run

    return _run(**kwargs)


def run_bmp4_promisys_comparison(
    *,
    family: str,
    problem_path: str | Path = DEFAULT_PROBLEM_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    run_label: str | None = None,
    cell_lines: list[str] | None = None,
    prior_mode: str = "default",
    literature_prior_json: str | Path | None = None,
    normalizer_data_dir: str | Path | None = None,
    promisys_hyperparams_json: str | Path | None = None,
    rounds: int = 1,
    batch_size: int = 1,
    no_baseline: bool = False,
    receptor_noise_log_sd: float = 0.05,
    fit_steps: int = 50,
    fit_learning_rate: float = 0.02,
    posterior_samples: int = 256,
    eig_steps: int = 100,
    eig_outer_samples: int = 64,
    eig_inner_samples: int | None = None,
    eig_learning_rate: float = 0.05,
    infonce_lambda: float | None = None,
    design_dist_init_std: float | None = None,
    design_temperature_scale: float | None = None,
    selector_temperature_final: float | None = None,
    early_stopping_patience: int | None = 10,
    early_stopping_min_delta: float = 0.0,
    mcmc_warmup: int = 200,
    mcmc_samples: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    if family not in {"promisys_onestep", "promisys_twostep"}:
        raise ValueError("family must be 'promisys_onestep' or 'promisys_twostep'.")
    if int(rounds) < 0:
        raise ValueError("rounds must be nonnegative.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1.")
    prior_modes = _resolve_prior_modes(prior_mode)
    needs_literature_prior = any(mode == "literature" for mode in prior_modes)
    if needs_literature_prior and literature_prior_json is None:
        literature_prior_json = DEFAULT_LITERATURE_PRIOR
        if not Path(literature_prior_json).exists():
            raise ValueError(
                "--literature-prior-json is required for prior-mode literature or both "
                f"because the default prior was not found at {DEFAULT_LITERATURE_PRIOR}."
            )
    promisys_hyperparams_path = None
    resolved_promisys_hyperparams = None
    if promisys_hyperparams_json is not None:
        promisys_hyperparams_path = Path(promisys_hyperparams_json)
        if not promisys_hyperparams_path.is_absolute():
            promisys_hyperparams_path = REPO_ROOT / promisys_hyperparams_path
        resolved_promisys_hyperparams = coerce_promisys_hyperparams(promisys_hyperparams_path)

    problem = _load_problem_bundle(problem_path)
    data_bundle = load_bmp4_gradient_data(problem["data"]["observed_data_ref"])
    selected_cell_lines = list(cell_lines or problem["metadata"]["cell_lines"])
    joint_data = build_joint_bmp4_gradient_data(data_bundle, cell_lines=selected_cell_lines)

    label = run_label or _default_run_label(family, selected_cell_lines)
    root = Path(output_dir) / _slugify_label(label)
    root.mkdir(parents=True, exist_ok=True)

    literature_prior_payload = None
    literature_prior_path = None
    if literature_prior_json is not None:
        literature_prior_path = Path(literature_prior_json)
        if not literature_prior_path.is_absolute():
            literature_prior_path = REPO_ROOT / literature_prior_path
        literature_prior_payload = _load_literature_prior_override(
            literature_prior_path,
            expected_family=family,
        )

    baseline: dict[str, Any] = {}
    sequential: dict[str, Any] = {}
    for mode in prior_modes:
        literature_prior = (
            literature_prior_payload["prior_object"]
            if mode == "literature" and literature_prior_payload is not None
            else None
        )
        literature_prior_jsons = [str(literature_prior_path)] if mode == "literature" else None
        if not no_baseline:
            baseline_dir = root / "baseline_all_data" / mode
            baseline[mode] = run_bmp4_gradient_example(
                problem_path=problem_path,
                output_dir=baseline_dir,
                cell_lines=selected_cell_lines,
                families=[family],
                literature_prior_jsons=literature_prior_jsons,
                fit_steps=fit_steps,
                fit_learning_rate=fit_learning_rate,
                posterior_samples=posterior_samples,
                eig_steps=eig_steps,
                eig_outer_samples=eig_outer_samples,
                eig_inner_samples=eig_inner_samples,
                eig_learning_rate=eig_learning_rate,
                infonce_lambda=infonce_lambda
                if infonce_lambda is not None
                else _default_infonce_lambda(family),
                design_dist_init_std=design_dist_init_std
                if design_dist_init_std is not None
                else _default_design_dist_init_std(family),
                design_temperature_scale=design_temperature_scale
                if design_temperature_scale is not None
                else _default_design_temperature_scale(family),
                selector_temperature_final=selector_temperature_final,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                normalizer_data_dir=normalizer_data_dir
                if normalizer_data_dir is not None
                else _default_normalizer_data_dir(family),
                receptor_noise_log_sd=receptor_noise_log_sd,
                mcmc_warmup=mcmc_warmup,
                mcmc_samples=mcmc_samples,
                promisys_hyperparams=resolved_promisys_hyperparams,
            )

        sequential_dir = root / f"sequential_{mode}"
        sequential[mode] = run_promisys_sequential_workflow(
            family_name=family,
            joint_data=joint_data,
            run_dir=sequential_dir,
            prior_mode=mode,
            literature_prior=literature_prior,
            normalizer_data_dir=normalizer_data_dir,
            receptor_noise_log_sd=receptor_noise_log_sd,
            likelihood_steps=fit_steps,
            likelihood_learning_rate=fit_learning_rate,
            posterior_sample_count=posterior_samples,
            eig_steps=eig_steps,
            eig_outer_samples=eig_outer_samples,
            eig_inner_samples=eig_inner_samples,
            eig_learning_rate=eig_learning_rate,
            infonce_lambda=infonce_lambda,
            design_dist_init_std=design_dist_init_std,
            design_temperature_scale=design_temperature_scale,
            selector_temperature_final=selector_temperature_final,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            mcmc_warmup=mcmc_warmup,
            mcmc_samples=mcmc_samples,
            rounds=rounds,
            batch_size=batch_size,
            seed=seed,
            promisys_hyperparams=resolved_promisys_hyperparams,
        )

    summary = {
        "family": family,
        "cell_lines": selected_cell_lines,
        "prior_mode": prior_mode,
        "prior_modes_run": prior_modes,
        "run_dir": str(root),
        "baseline_enabled": not no_baseline,
        "literature_prior_json": str(literature_prior_path) if literature_prior_path is not None else None,
        "normalizer_data_dir": str(normalizer_data_dir) if normalizer_data_dir is not None else None,
        "promisys_hyperparams_json": str(promisys_hyperparams_path) if promisys_hyperparams_path else None,
        "rounds": int(rounds),
        "batch_size": int(batch_size),
        "fit_steps": int(fit_steps),
        "early_stopping_patience": (
            None if early_stopping_patience is None else int(early_stopping_patience)
        ),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "baseline": baseline,
        "sequential": sequential,
    }
    comparison_path = root / "comparison_summary.json"
    summary["comparison_summary_path"] = str(comparison_path)
    _write_json(comparison_path, summary)
    for mode, result in sequential.items():
        _write_json(Path(result["run_dir"]) / "comparison_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["promisys_onestep", "promisys_twostep"])
    parser.add_argument("--problem-path", default=str(DEFAULT_PROBLEM_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--cell-line", action="append", dest="cell_lines")
    parser.add_argument("--prior-mode", default="default", choices=["default", "literature", "both"])
    parser.add_argument(
        "--literature-prior-json",
        default=None,
        help=(
            "Optional saved literature prior. If omitted with --prior-mode literature/both, "
            f"uses {DEFAULT_LITERATURE_PRIOR}."
        ),
    )
    parser.add_argument("--promisys-hyperparams-json", default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--normalizer-data-dir", default=None)
    parser.add_argument("--output-summary-json", default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--fit-steps", type=int, default=50)
    parser.add_argument("--fit-learning-rate", type=float, default=0.02)
    parser.add_argument("--posterior-samples", type=int, default=256)
    parser.add_argument("--eig-steps", type=int, default=100)
    parser.add_argument("--eig-outer-samples", type=int, default=64)
    parser.add_argument("--eig-inner-samples", type=int, default=None)
    parser.add_argument("--eig-learning-rate", type=float, default=0.05)
    parser.add_argument("--infonce-lambda", type=float, default=None)
    parser.add_argument("--design-dist-init-std", type=float, default=None)
    parser.add_argument(
        "--design-temperature-scale",
        type=float,
        default=None,
        help="Final raw-dose design-distribution std in BMP4 concentration units.",
    )
    parser.add_argument(
        "--selector-temperature-final",
        type=float,
        default=None,
        help="Final softmax temperature for the cell-line selector.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--receptor-noise-log-sd", type=float, default=0.05)
    parser.add_argument("--mcmc-warmup", type=int, default=200)
    parser.add_argument("--mcmc-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    summary = run_bmp4_promisys_comparison(
        family=args.family,
        problem_path=args.problem_path,
        output_dir=args.output_dir,
        run_label=args.run_label,
        cell_lines=args.cell_lines,
        prior_mode=args.prior_mode,
        literature_prior_json=args.literature_prior_json,
        normalizer_data_dir=args.normalizer_data_dir,
        promisys_hyperparams_json=args.promisys_hyperparams_json,
        rounds=args.rounds,
        batch_size=args.batch_size,
        no_baseline=args.no_baseline,
        receptor_noise_log_sd=args.receptor_noise_log_sd,
        fit_steps=args.fit_steps,
        fit_learning_rate=args.fit_learning_rate,
        posterior_samples=args.posterior_samples,
        eig_steps=args.eig_steps,
        eig_outer_samples=args.eig_outer_samples,
        eig_inner_samples=args.eig_inner_samples,
        eig_learning_rate=args.eig_learning_rate,
        infonce_lambda=args.infonce_lambda,
        design_dist_init_std=args.design_dist_init_std,
        design_temperature_scale=args.design_temperature_scale,
        selector_temperature_final=args.selector_temperature_final,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        mcmc_warmup=args.mcmc_warmup,
        mcmc_samples=args.mcmc_samples,
        seed=args.seed,
    )
    if args.output_summary_json:
        _write_json(Path(args.output_summary_json), summary)
    print(json.dumps(summary, indent=2))
    return 0


def _resolve_prior_modes(prior_mode: str) -> list[str]:
    if prior_mode == "both":
        return ["default", "literature"]
    if prior_mode in {"default", "literature"}:
        return [prior_mode]
    raise ValueError("prior_mode must be 'default', 'literature', or 'both'.")


def _load_problem_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    with bundle_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not Path(payload["data"]["observed_data_ref"]).is_absolute():
        payload["data"]["observed_data_ref"] = str(REPO_ROOT / payload["data"]["observed_data_ref"])
    return payload


def _load_literature_prior_override(path: str | Path, *, expected_family: str) -> dict[str, Any]:
    prior_path = Path(path)
    with prior_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    family_name = payload.get("family")
    family_reused = isinstance(family_name, str) and family_name != expected_family
    if isinstance(payload.get("prior_used"), dict):
        prior_used = _augmented_prior_from_augmented_payload(payload["prior_used"])
        literature_report = payload.get("literature_report")
        if isinstance(literature_report, dict):
            literature_report = dict(literature_report)
            if family_reused:
                literature_report.setdefault("reported_family", family_name)
                literature_report.setdefault("reused_for_family", expected_family)
                literature_report.setdefault("family_reused_across_models", True)
    elif isinstance(payload.get("priors"), dict):
        prior_used = _augmented_prior_from_codex_payload(payload)
        literature_report = {
            "source": "external_prior_json",
            "path": str(prior_path),
            "family": expected_family,
            "reported_family": family_name,
            "family_reused_across_models": family_reused,
            "notes": list(payload.get("notes") or []),
            "corpus_scope": payload.get("corpus_scope"),
        }
    else:
        raise ValueError(f"Unsupported literature prior JSON format in {prior_path}.")
    return {
        "literature_report": literature_report,
        "prior_object": prior_used,
        "source_path": str(prior_path),
    }


def _augmented_prior_from_augmented_payload(payload: dict[str, Any]) -> AugmentedPrior:
    distributions: dict[str, DistributionSpec] = {}
    raw_distributions = payload.get("distributions") or {}
    if not isinstance(raw_distributions, dict):
        raise ValueError("Saved prior_used payload must contain a distributions mapping.")
    for name, spec in raw_distributions.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Saved prior distribution for {name!r} must be a mapping.")
        distributions[str(name)] = DistributionSpec(
            name=spec.get("name"),
            params=dict(spec.get("params") or {}),
            source=str(spec.get("source", "literature_json")),
            reasoning=str(spec.get("reasoning", "")),
            cited_papers=list(spec.get("cited_papers") or []),
            fallback=bool(spec.get("fallback", False)),
        )
    return AugmentedPrior(
        distributions=distributions,
        warnings=list(payload.get("warnings") or []),
        notes=list(payload.get("notes") or []),
    )


def _augmented_prior_from_codex_payload(payload: dict[str, Any]) -> AugmentedPrior:
    distributions: dict[str, DistributionSpec] = {}
    priors = payload.get("priors") or {}
    if not isinstance(priors, dict):
        raise ValueError("Codex prior JSON must contain a priors mapping.")
    for name, spec in priors.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Codex prior for {name!r} must be a mapping.")
        distributions[str(name)] = DistributionSpec(
            name=spec.get("distribution") or spec.get("name"),
            params=dict(spec.get("params") or {}),
            source=str(spec.get("source", "literature_json")),
            reasoning=str(spec.get("reasoning", "")),
            cited_papers=list(spec.get("cited_papers") or []),
            fallback=bool(spec.get("fallback", False)),
        )
    return AugmentedPrior(
        distributions=distributions,
        warnings=list(payload.get("warnings") or []),
        notes=list(payload.get("notes") or []),
    )


def _default_run_label(family: str, cell_lines: list[str]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{family}__{'__'.join(cell_lines)}__{stamp}"


def _default_normalizer_data_dir(family: str) -> Path:
    if family == "promisys_onestep":
        from examples.cases.bmp4_gradient import promisys_onestep as module
    else:
        from examples.cases.bmp4_gradient import promisys_twostep as module
    return module.DEFAULT_NORMALIZER_DATA_DIR


def _default_infonce_lambda(family: str) -> float:
    if family == "promisys_onestep":
        from examples.cases.bmp4_gradient import promisys_onestep as module
    else:
        from examples.cases.bmp4_gradient import promisys_twostep as module
    return float(module.DEFAULT_INFONCE_LAMBDA)


def _default_design_dist_init_std(family: str) -> float:
    if family == "promisys_onestep":
        from examples.cases.bmp4_gradient import promisys_onestep as module
    else:
        from examples.cases.bmp4_gradient import promisys_twostep as module
    return float(module.DEFAULT_DESIGN_DIST_INIT_STD)


def _default_design_temperature_scale(family: str) -> float:
    if family == "promisys_onestep":
        from examples.cases.bmp4_gradient import promisys_onestep as module
    else:
        from examples.cases.bmp4_gradient import promisys_twostep as module
    return float(module.DEFAULT_DESIGN_TEMPERATURE_SCALE)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)


def _slugify_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "comparison"


def _json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None  # type: ignore[assignment]
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


if __name__ == "__main__":
    raise SystemExit(main())
