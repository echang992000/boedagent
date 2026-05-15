#!/usr/bin/env python3
"""Autoresearch-style runner for BMP4 promisys hyperparameter trials."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_DIR = REPO_ROOT / "examples" / "autoresearch" / "bmp4_promisys"
DEFAULT_BASELINE_CONFIG = PROGRAM_DIR / "baseline.json"
DEFAULT_TRIAL_CONFIG = PROGRAM_DIR / "trial.json"
DEFAULT_AUTORESEARCH_ROOT = REPO_ROOT / "artifacts" / "bmp4_gradient" / "autoresearch"
DEFAULT_LITERATURE_PRIOR = (
    REPO_ROOT
    / "artifacts"
    / "bmp4_gradient"
    / "multireceptor_hierarchical"
    / "joint__NMuMG__BMPR2_KD__ACVR1_KD__BMPR1A_KD"
    / "literature_prior.json"
)
DEFAULT_NORMALIZER_DATA_DIR = Path("/Users/vincentzaballa/Development/bmp_simformer/data")
RESULTS_HEADER = [
    "trial_id",
    "timestamp",
    "config",
    "best_eig",
    "best_cell_line",
    "best_dose",
    "final_snpe_loss",
    "final_likelihood_loss",
    "runtime_seconds",
    "status",
    "decision",
    "description",
    "run_dir",
    "reasons",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize an autoresearch tag directory.")
    _add_common_tag_args(init_parser)
    init_parser.set_defaults(func=_cmd_init)

    run_parser = subparsers.add_parser("run", help="Run one BMP4 promisys trial and append results.tsv.")
    _add_common_tag_args(run_parser)
    run_parser.add_argument("--trial-id", default=None)
    run_parser.add_argument("--config", default=str(DEFAULT_TRIAL_CONFIG))
    run_parser.add_argument("--baseline", action="store_true", help="Use baseline.json and keep if valid.")
    run_parser.add_argument("--description", default="")
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--family", default="promisys_twostep")
    run_parser.add_argument("--cell-line", action="append", dest="cell_lines")
    run_parser.add_argument("--literature-prior-json", default=str(DEFAULT_LITERATURE_PRIOR))
    run_parser.add_argument("--normalizer-data-dir", default=str(DEFAULT_NORMALIZER_DATA_DIR))
    run_parser.add_argument("--snpe-simulations", type=int, default=128)
    run_parser.add_argument("--snpe-steps", type=int, default=200)
    run_parser.add_argument("--posterior-samples", type=int, default=64)
    run_parser.add_argument("--fit-steps", type=int, default=50)
    run_parser.add_argument("--early-stopping-patience", type=int, default=10)
    run_parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    run_parser.add_argument("--eig-outer-samples", type=int, default=32)
    run_parser.add_argument("--eig-inner-samples", type=int, default=5)
    run_parser.add_argument("--mcmc-warmup", type=int, default=50)
    run_parser.add_argument("--mcmc-samples", type=int, default=64)
    run_parser.add_argument("--timeout-seconds", type=int, default=900)
    run_parser.add_argument(
        "--confirmation",
        action="store_true",
        help="Promote the run budget to the production confirmation defaults.",
    )
    run_parser.set_defaults(func=_cmd_run)

    score_parser = subparsers.add_parser("score", help="Score an existing trial directory.")
    score_parser.add_argument("run_dir")
    score_parser.add_argument("--timeout-seconds", type=float, default=None)
    score_parser.add_argument("--json", action="store_true")
    score_parser.set_defaults(func=_cmd_score)

    loop_parser = subparsers.add_parser(
        "loop",
        help="Autonomously generate and run config variants until a stop criterion is reached.",
    )
    _add_common_tag_args(loop_parser)
    loop_parser.add_argument("--max-trials", type=int, default=10)
    loop_parser.add_argument("--patience", type=int, default=4)
    loop_parser.add_argument("--max-runtime-seconds", type=int, default=None)
    loop_parser.add_argument("--seed", type=int, default=0)
    loop_parser.add_argument("--seed-config", default=str(DEFAULT_TRIAL_CONFIG))
    loop_parser.add_argument("--run-baseline", action=argparse.BooleanOptionalAction, default=True)
    loop_parser.add_argument("--python", default=sys.executable)
    loop_parser.add_argument("--family", default="promisys_twostep")
    loop_parser.add_argument("--cell-line", action="append", dest="cell_lines")
    loop_parser.add_argument("--literature-prior-json", default=str(DEFAULT_LITERATURE_PRIOR))
    loop_parser.add_argument("--normalizer-data-dir", default=str(DEFAULT_NORMALIZER_DATA_DIR))
    loop_parser.add_argument("--snpe-simulations", type=int, default=128)
    loop_parser.add_argument("--snpe-steps", type=int, default=200)
    loop_parser.add_argument("--posterior-samples", type=int, default=64)
    loop_parser.add_argument("--fit-steps", type=int, default=50)
    loop_parser.add_argument("--early-stopping-patience", type=int, default=10)
    loop_parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    loop_parser.add_argument("--eig-outer-samples", type=int, default=32)
    loop_parser.add_argument("--eig-inner-samples", type=int, default=5)
    loop_parser.add_argument("--mcmc-warmup", type=int, default=50)
    loop_parser.add_argument("--mcmc-samples", type=int, default=64)
    loop_parser.add_argument("--timeout-seconds", type=int, default=900)
    loop_parser.add_argument("--confirmation", action="store_true", default=False)
    loop_parser.set_defaults(func=_cmd_loop)

    args = parser.parse_args(argv)
    return int(args.func(args))


def score_run_dir(
    run_dir: str | Path,
    *,
    timeout_seconds: float | None = None,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    eig_path, fit_path = _find_summary_pair(root)
    reasons: list[str] = []
    if eig_path is None:
        return {
            "status": "crash",
            "best_eig": None,
            "best_cell_line": None,
            "best_dose": None,
            "final_snpe_loss": None,
            "final_likelihood_loss": None,
            "runtime_seconds": runtime_seconds,
            "reasons": ["missing eig_optimization_summary.json"],
            "run_dir": str(root),
        }

    eig_summary = _load_json(eig_path)
    fit_summary = _load_json(fit_path) if fit_path is not None else {}
    best_eig = _finite_float(eig_summary.get("best_eig"))
    best_dose = _finite_float(eig_summary.get("best_dose"))
    final_snpe_loss = _last_finite(fit_summary.get("snpe_loss_history"))
    final_likelihood_loss = _last_finite(fit_summary.get("likelihood_loss_history"))
    mcmc_counts = fit_summary.get("mcmc_sample_count_by_cell") or {}

    if best_eig is None:
        reasons.append("best_eig missing or non-finite")
    if final_snpe_loss is None:
        reasons.append("SNPE loss missing or non-finite")
    if final_likelihood_loss is None:
        reasons.append("likelihood loss missing or non-finite")
    if not mcmc_counts or any(int(value) <= 0 for value in mcmc_counts.values()):
        reasons.append("MCMC sample counts missing or zero")
    missing_artifacts = _missing_expected_artifacts(eig_path.parent)
    if missing_artifacts:
        reasons.append("missing artifacts: " + ", ".join(missing_artifacts))
    if runtime_seconds is not None and timeout_seconds is not None and runtime_seconds > timeout_seconds:
        reasons.append(f"runtime exceeded timeout ({runtime_seconds:.1f}s > {timeout_seconds:.1f}s)")

    return {
        "status": "completed" if not reasons else "failed",
        "best_eig": best_eig,
        "best_cell_line": eig_summary.get("best_cell_line"),
        "best_dose": best_dose,
        "final_snpe_loss": final_snpe_loss,
        "final_likelihood_loss": final_likelihood_loss,
        "runtime_seconds": runtime_seconds,
        "reasons": reasons,
        "eig_summary_path": str(eig_path),
        "fit_summary_path": str(fit_path) if fit_path is not None else None,
        "run_dir": str(root),
    }


def _cmd_init(args: argparse.Namespace) -> int:
    tag_dir = _tag_dir(args)
    tag_dir.mkdir(parents=True, exist_ok=True)
    _ensure_results_file(tag_dir / "results.tsv")
    best_config = tag_dir / "best_config.json"
    if not best_config.exists() and DEFAULT_BASELINE_CONFIG.exists():
        shutil.copyfile(DEFAULT_BASELINE_CONFIG, best_config)
    print(json.dumps({"tag_dir": str(tag_dir), "results": str(tag_dir / "results.tsv")}, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    source_config = DEFAULT_BASELINE_CONFIG if args.baseline else Path(args.config)
    if not source_config.exists():
        raise FileNotFoundError(f"Config not found: {source_config}")
    config_payload = _load_json(source_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_id = args.trial_id or ("baseline" if args.baseline else f"trial_{timestamp}")
    score = _execute_trial(
        args,
        config_payload=config_payload,
        trial_id=trial_id,
        description=args.description,
        baseline=bool(args.baseline),
        source_config=source_config,
        timestamp=timestamp,
    )
    print(json.dumps(score, indent=2))
    return 0 if score["status"] == "completed" else 1


def _cmd_loop(args: argparse.Namespace) -> int:
    tag_dir = _tag_dir(args)
    tag_dir.mkdir(parents=True, exist_ok=True)
    results_path = tag_dir / "results.tsv"
    _ensure_results_file(results_path)
    rng = random.Random(int(args.seed))
    started = time.perf_counter()
    no_improvement = 0
    completed_trials = 0
    kept_trials = 0
    stop_reason = "max_trials"

    if args.run_baseline and _previous_best_eig(results_path) is None:
        baseline_score = _execute_trial(
            args,
            config_payload=_load_json(DEFAULT_BASELINE_CONFIG),
            trial_id="baseline",
            description="loop baseline",
            baseline=True,
            source_config=DEFAULT_BASELINE_CONFIG,
        )
        print(json.dumps(baseline_score, indent=2))
        if baseline_score.get("status") != "completed":
            stop_reason = "baseline_failed"
            return _write_loop_summary(
                tag_dir,
                started=started,
                stop_reason=stop_reason,
                completed_trials=completed_trials,
                kept_trials=kept_trials,
            )

    for trial_index in range(1, max(int(args.max_trials), 0) + 1):
        if args.max_runtime_seconds is not None:
            elapsed = time.perf_counter() - started
            if elapsed >= float(args.max_runtime_seconds):
                stop_reason = "runtime_budget"
                break
        base_config = _loop_base_config(args, tag_dir)
        config_payload, mutation_description = _mutate_config(base_config, rng, args)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_id = f"auto_{timestamp}_{trial_index:03d}"
        score = _execute_trial(
            args,
            config_payload=config_payload,
            trial_id=trial_id,
            description=mutation_description,
            baseline=False,
            source_config=None,
            timestamp=timestamp,
        )
        print(json.dumps(score, indent=2))
        completed_trials += 1
        if score.get("decision") == "keep":
            kept_trials += 1
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= max(int(args.patience), 1):
            stop_reason = "patience"
            break
    else:
        stop_reason = "max_trials"

    return _write_loop_summary(
        tag_dir,
        started=started,
        stop_reason=stop_reason,
        completed_trials=completed_trials,
        kept_trials=kept_trials,
    )


def _execute_trial(
    args: argparse.Namespace,
    *,
    config_payload: dict[str, Any],
    trial_id: str,
    description: str,
    baseline: bool,
    source_config: Path | None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    tag_dir = _tag_dir(args)
    tag_dir.mkdir(parents=True, exist_ok=True)
    results_path = tag_dir / "results.tsv"
    _ensure_results_file(results_path)

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = tag_dir / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_copy = trial_dir / "trial_config.json"
    run_args = argparse.Namespace(**vars(args))
    if getattr(args, "confirmation", False):
        config_payload = _with_confirmation_budget(config_payload)
        run_args.snpe_simulations = 2000
        run_args.snpe_steps = 2000
        run_args.fit_steps = 100
        run_args.eig_outer_samples = 128
        run_args.mcmc_warmup = 200
        run_args.mcmc_samples = 256
    config_copy.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    output_dir = trial_dir / "output"
    log_path = trial_dir / "run.log"
    command = _bmp4_agent_command(run_args, output_dir=output_dir, config_path=config_copy)
    started = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n\n")
        log_handle.flush()
        try:
            proc = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(args.timeout_seconds),
                check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            log_handle.write(f"\n[bmp4_autoresearch] timed out after {args.timeout_seconds}s\n")
    runtime_seconds = time.perf_counter() - started

    score = score_run_dir(
        trial_dir,
        timeout_seconds=float(args.timeout_seconds),
        runtime_seconds=runtime_seconds,
    )
    if timed_out:
        score["status"] = "crash"
        score.setdefault("reasons", []).append("timeout")
    if returncode != 0 and score["status"] == "completed":
        score["status"] = "failed"
        score.setdefault("reasons", []).append(f"command returned {returncode}")

    decision = _decide_trial(score, results_path, baseline=baseline)
    score["decision"] = decision
    if decision == "keep":
        shutil.copyfile(config_copy, tag_dir / "best_config.json")
    elif (
        source_config is not None
        and source_config.resolve() == DEFAULT_TRIAL_CONFIG.resolve()
        and (tag_dir / "best_config.json").exists()
    ):
        shutil.copyfile(tag_dir / "best_config.json", DEFAULT_TRIAL_CONFIG)

    _append_result(
        results_path,
        trial_id=trial_id,
        timestamp=timestamp,
        config=config_copy,
        score=score,
        decision=decision,
        description=description,
        run_dir=trial_dir,
    )
    plot_path = write_progress_plot(results_path, title=_progress_title(args, tag_dir))
    if plot_path is not None:
        score["progress_plot_path"] = str(plot_path)
    return score


def _cmd_score(args: argparse.Namespace) -> int:
    score = score_run_dir(args.run_dir, timeout_seconds=args.timeout_seconds)
    if args.json:
        print(json.dumps(score, indent=2))
    else:
        print(
            "\t".join(
                [
                    str(score.get("status")),
                    str(score.get("best_eig")),
                    str(score.get("best_cell_line")),
                    str(score.get("best_dose")),
                    "; ".join(score.get("reasons", [])),
                ]
            )
        )
    return 0 if score["status"] == "completed" else 1


def _bmp4_agent_command(args: argparse.Namespace, *, output_dir: Path, config_path: Path) -> list[str]:
    command = [
        str(args.python),
        str(REPO_ROOT / "examples" / "agent" / "bmp4_gradient_agent.py"),
        "--family",
        str(args.family),
        "--literature-prior-json",
        str(args.literature_prior_json),
        "--normalizer-data-dir",
        str(args.normalizer_data_dir),
        "--output-dir",
        str(output_dir),
        "--snpe-simulations",
        str(args.snpe_simulations),
        "--snpe-steps",
        str(args.snpe_steps),
        "--posterior-samples",
        str(args.posterior_samples),
        "--fit-steps",
        str(args.fit_steps),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--eig-outer-samples",
        str(args.eig_outer_samples),
        "--eig-inner-samples",
        str(args.eig_inner_samples),
        "--mcmc-warmup",
        str(args.mcmc_warmup),
        "--mcmc-samples",
        str(args.mcmc_samples),
        "--promisys-hyperparams-json",
        str(config_path),
    ]
    for cell_line in args.cell_lines or []:
        command.extend(["--cell-line", str(cell_line)])
    return command


def _decide_trial(score: dict[str, Any], results_path: Path, *, baseline: bool) -> str:
    if score.get("status") != "completed" or score.get("best_eig") is None:
        return "crash" if score.get("status") == "crash" else "discard"
    if baseline:
        return "keep"
    previous_best = _previous_best_eig(results_path)
    if previous_best is None:
        return "keep"
    improvement = float(score["best_eig"]) - previous_best
    threshold = max(0.01, 0.02 * abs(previous_best))
    return "keep" if improvement >= threshold else "discard"


def _previous_best_eig(results_path: Path) -> float | None:
    if not results_path.exists():
        return None
    best: float | None = None
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("decision") != "keep" or row.get("status") != "completed":
                continue
            value = _finite_float(row.get("best_eig"))
            if value is not None and (best is None or value > best):
                best = value
    return best


def _append_result(
    results_path: Path,
    *,
    trial_id: str,
    timestamp: str,
    config: Path,
    score: dict[str, Any],
    decision: str,
    description: str,
    run_dir: Path,
) -> None:
    _ensure_results_file(results_path)
    row = {
        "trial_id": trial_id,
        "timestamp": timestamp,
        "config": str(config),
        "best_eig": _format_optional_float(score.get("best_eig")),
        "best_cell_line": score.get("best_cell_line") or "",
        "best_dose": _format_optional_float(score.get("best_dose")),
        "final_snpe_loss": _format_optional_float(score.get("final_snpe_loss")),
        "final_likelihood_loss": _format_optional_float(score.get("final_likelihood_loss")),
        "runtime_seconds": _format_optional_float(score.get("runtime_seconds")),
        "status": score.get("status") or "",
        "decision": decision,
        "description": description,
        "run_dir": str(run_dir),
        "reasons": "; ".join(score.get("reasons", [])),
    }
    with results_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writerow(row)


def _ensure_results_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(RESULTS_HEADER)


def write_progress_plot(
    results_path: str | Path,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path | None:
    results = Path(results_path)
    output = Path(output_path) if output_path is not None else results.with_name("progress.png")
    rows = _read_results_rows(results)
    finite = [row for row in rows if row["best_eig_float"] is not None]
    if not finite:
        return None
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "boed_agent_matplotlib"))
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    kept = [row for row in rows if row.get("decision") == "keep" and row["best_eig_float"] is not None]
    discarded = [
        row for row in rows
        if row.get("decision") == "discard" and row["best_eig_float"] is not None
    ]
    crashed = [row for row in rows if row["best_eig_float"] is None]
    running_best: list[float | None] = []
    best: float | None = None
    for row in rows:
        eig = row["best_eig_float"]
        if eig is not None and (best is None or eig > best):
            best = eig
        running_best.append(best)

    y_values = [row["best_eig_float"] for row in finite if row["best_eig_float"] is not None]
    y_min = min(y_values)
    y_max = max(y_values)
    span = max(y_max - y_min, abs(y_max) * 0.08, 0.01)
    crash_y = y_min - 0.18 * span
    x_values = [int(row["experiment"]) for row in rows]

    fig, axis = plt.subplots(figsize=(17, 7.5), dpi=180)
    axis.scatter(
        [row["experiment"] for row in discarded],
        [row["best_eig_float"] for row in discarded],
        s=34,
        color="#b8bcc2",
        alpha=0.55,
        label="Discarded",
        zorder=2,
    )
    axis.scatter(
        [row["experiment"] for row in kept],
        [row["best_eig_float"] for row in kept],
        s=78,
        color="#33d08a",
        edgecolor="#16784f",
        linewidth=0.9,
        label="Kept",
        zorder=4,
    )
    if crashed:
        axis.scatter(
            [row["experiment"] for row in crashed],
            [crash_y for _ in crashed],
            marker="x",
            s=70,
            color="#d65f5f",
            linewidth=1.7,
            label="Crashed",
            zorder=5,
        )
    step_x = [rows[index]["experiment"] for index, value in enumerate(running_best) if value is not None]
    step_y = [value for value in running_best if value is not None]
    axis.step(step_x, step_y, where="post", color="#73cfa0", linewidth=2.2, label="Running best", zorder=3)

    for row in rows:
        y = row["best_eig_float"] if row["best_eig_float"] is not None else crash_y
        label = _plot_label(row)
        color = "#2f9d66" if row.get("decision") == "keep" else (
            "#c64a4a" if row.get("decision") == "crash" else "#7b8087"
        )
        axis.annotate(
            label,
            (row["experiment"], y),
            xytext=(8, 8),
            textcoords="offset points",
            rotation=25,
            color=color,
            fontsize=8.5,
            ha="left",
            va="bottom",
        )

    axis.set_title(title or _default_progress_title(results, rows, kept), fontsize=16)
    axis.set_xlabel("Experiment #", fontsize=12)
    axis.set_ylabel("Best EIG (higher is better)", fontsize=12)
    axis.grid(True, color="#d9dde2", alpha=0.45, linewidth=0.8)
    axis.set_xlim(min(x_values) - 0.2, max(x_values) + 0.8)
    axis.set_ylim(crash_y - 0.18 * span, y_max + 0.45 * span)
    axis.legend(loc="best", frameon=True)
    axis.text(0.01, -0.12, "Source: " + str(results), transform=axis.transAxes, fontsize=8.5, color="#6b7280")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _read_results_rows(results_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results_path.exists():
        return rows
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            row["experiment"] = index
            row["best_eig_float"] = _finite_float(row.get("best_eig"))
            rows.append(row)
    return rows


def _plot_label(row: dict[str, Any]) -> str:
    label = (row.get("description") or row.get("trial_id") or f"experiment {row['experiment']}").strip()
    if row.get("decision") == "crash":
        reasons = str(row.get("reasons") or "").strip()
        if reasons:
            label = f"{label} ({reasons})"
    return label[:55] + "..." if len(label) > 58 else label


def _progress_title(args: argparse.Namespace, tag_dir: Path) -> str:
    family = str(getattr(args, "family", "promisys")).replace("_", " ")
    rows = _read_results_rows(tag_dir / "results.tsv")
    kept_count = sum(1 for row in rows if row.get("decision") == "keep")
    return f"BMP4 {family} Autoresearch Progress: {len(rows)} Experiments, {kept_count} Kept"


def _default_progress_title(results_path: Path, rows: list[dict[str, Any]], kept: list[dict[str, Any]]) -> str:
    tag = results_path.parent.name.replace("_", " ")
    return f"BMP4 Autoresearch Progress ({tag}): {len(rows)} Experiments, {len(kept)} Kept"


def _find_summary_pair(root: Path) -> tuple[Path | None, Path | None]:
    eig_paths = sorted(root.rglob("eig_optimization_summary.json"))
    if not eig_paths:
        return None, None
    for eig_path in eig_paths:
        fit_path = eig_path.with_name("fit_summary.json")
        if fit_path.exists():
            return eig_path, fit_path
    return eig_paths[0], None


def _missing_expected_artifacts(run_leaf: Path) -> list[str]:
    expected = [
        "snpe_posterior_samples.pt",
        "mcmc_posterior_samples.pt",
        "posterior_predictive.pt",
        "likelihood_checkpoint.pkl",
    ]
    return [filename for filename in expected if not (run_leaf / filename).exists()]


def _with_confirmation_budget(config: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(config))
    posterior = payload.setdefault("posterior_net", {})
    posterior["simulations"] = 2000
    posterior["steps"] = 2000
    objective = payload.setdefault("objective", {})
    objective["fit_steps"] = 100
    objective["eig_outer_samples"] = 128
    mcmc = payload.setdefault("mcmc", {})
    mcmc["warmup"] = 200
    mcmc["samples"] = 256
    return payload


def _loop_base_config(args: argparse.Namespace, tag_dir: Path) -> dict[str, Any]:
    best_config = tag_dir / "best_config.json"
    if best_config.exists():
        payload = _load_json(best_config)
        if payload:
            return payload
    seed_config = Path(args.seed_config)
    if seed_config.exists():
        payload = _load_json(seed_config)
        if payload:
            return payload
    return _default_search_config(args)


def _default_search_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "posterior_net": {
            "hidden_dim": 96,
            "layers": 2,
            "activation": "gelu",
            "batch_size": 64,
            "learning_rate": 0.001,
            "steps": int(args.snpe_steps),
            "simulations": int(args.snpe_simulations),
            "posterior_samples": int(args.posterior_samples),
        },
        "flow": {
            "num_layers": 4,
            "hidden_sizes": [96, 96],
            "num_bins": 8,
            "activation": "gelu",
            "use_resnet": True,
            "dropout_rate": 0.0,
            "standardize_theta": False,
        },
        "objective": {
            "fit_steps": int(args.fit_steps),
            "flow_learning_rate": 0.02,
            "design_learning_rate": 0.05,
            "eig_outer_samples": int(args.eig_outer_samples),
            "eig_inner_samples": int(args.eig_inner_samples),
            "infonce_lambda": 0.5,
            "design_dist_init_std": 0.3,
            "design_temperature_scale": 1.0,
            "early_stopping_patience": int(args.early_stopping_patience),
            "early_stopping_min_delta": float(args.early_stopping_min_delta),
        },
        "mcmc": {
            "warmup": int(args.mcmc_warmup),
            "samples": int(args.mcmc_samples),
            "proposal_scale": 0.03,
            "prior_std_floor": 0.03,
        },
    }


def _mutate_config(
    base_config: dict[str, Any],
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    candidate = _default_search_config(args)
    _deep_update(candidate, base_config)
    knobs: list[tuple[tuple[str, ...], list[Any], str]] = [
        (("posterior_net", "hidden_dim"), [64, 96, 128, 192], "posterior hidden dim"),
        (("posterior_net", "layers"), [1, 2, 3], "posterior layers"),
        (("posterior_net", "activation"), ["gelu", "silu", "relu"], "posterior activation"),
        (("posterior_net", "batch_size"), [32, 64, 128], "posterior batch"),
        (("posterior_net", "learning_rate"), [0.0003, 0.0005, 0.001, 0.002], "posterior LR"),
        (
            ("posterior_net", "steps"),
            _scaled_int_choices(int(args.snpe_steps), [0.5, 1.0, 1.5, 2.0], minimum=1),
            "SNPE steps",
        ),
        (
            ("posterior_net", "simulations"),
            _scaled_int_choices(int(args.snpe_simulations), [0.5, 1.0, 1.5, 2.0], minimum=2),
            "SNPE simulations",
        ),
        (("flow", "num_layers"), [2, 3, 4, 5, 6], "flow layers"),
        (
            ("flow", "hidden_sizes"),
            [[64, 64], [96, 96], [128, 128], [128, 128, 64], [192, 192]],
            "flow width",
        ),
        (("flow", "num_bins"), [6, 8, 10, 12, 16], "flow bins"),
        (("flow", "activation"), ["gelu", "silu", "relu"], "flow activation"),
        (("flow", "use_resnet"), [True, False], "flow resnet"),
        (("flow", "dropout_rate"), [0.0, 0.02, 0.05, 0.1], "flow dropout"),
        (("flow", "standardize_theta"), [False, True], "standardize theta"),
        (
            ("objective", "fit_steps"),
            _scaled_int_choices(int(args.fit_steps), [0.5, 1.0, 1.5, 2.0], minimum=1),
            "BOED fit steps",
        ),
        (("objective", "flow_learning_rate"), [0.002, 0.005, 0.01, 0.02, 0.04], "flow LR"),
        (("objective", "design_learning_rate"), [0.01, 0.02, 0.05, 0.08], "design LR"),
        (("objective", "eig_inner_samples"), [1, 3, 5, 10, 15], "InfoNCE negatives"),
        (("objective", "infonce_lambda"), [0.1, 0.25, 0.5, 0.75, 1.0], "InfoNCE lambda"),
        (("objective", "design_temperature_scale"), [0.5, 0.75, 1.0, 1.5, 2.0], "design temperature"),
        (("mcmc", "proposal_scale"), [0.015, 0.02, 0.03, 0.05], "MCMC proposal"),
        (("mcmc", "prior_std_floor"), [0.02, 0.03, 0.05, 0.08], "MCMC prior floor"),
    ]
    mutation_count = rng.randint(2, 4)
    changes: list[str] = []
    for path, values, label in rng.sample(knobs, k=mutation_count):
        current = _get_path(candidate, path)
        value = _choice_different(values, current, rng)
        _set_path(candidate, path, value)
        changes.append(f"{label}: {current}->{value}")
    return candidate, "; ".join(changes)


def _write_loop_summary(
    tag_dir: Path,
    *,
    started: float,
    stop_reason: str,
    completed_trials: int,
    kept_trials: int,
) -> int:
    progress_plot_path = write_progress_plot(tag_dir / "results.tsv")
    summary = {
        "stop_reason": stop_reason,
        "completed_trials": int(completed_trials),
        "kept_trials": int(kept_trials),
        "runtime_seconds": float(time.perf_counter() - started),
        "results_path": str(tag_dir / "results.tsv"),
        "best_config_path": str(tag_dir / "best_config.json"),
        "progress_plot_path": str(progress_plot_path) if progress_plot_path is not None else None,
    }
    path = tag_dir / "loop_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _scaled_int_choices(base: int, scales: list[float], *, minimum: int) -> list[int]:
    return sorted({max(int(round(base * scale)), minimum) for scale in scales})


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = json.loads(json.dumps(value))


def _get_path(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = config
    for key in path:
        cursor = cursor[key]
    return cursor


def _set_path(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor: dict[str, Any] = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = json.loads(json.dumps(value))


def _choice_different(values: list[Any], current: Any, rng: random.Random) -> Any:
    candidates = [value for value in values if value != current]
    return json.loads(json.dumps(rng.choice(candidates or values)))


def _tag_dir(args: argparse.Namespace) -> Path:
    return Path(args.root) / str(args.tag)


def _add_common_tag_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", default=str(DEFAULT_AUTORESEARCH_ROOT))


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def _last_finite(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    for value in reversed(values):
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return None


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _format_optional_float(value: Any) -> str:
    parsed = _finite_float(value)
    return "" if parsed is None else f"{parsed:.8g}"


if __name__ == "__main__":
    raise SystemExit(main())
