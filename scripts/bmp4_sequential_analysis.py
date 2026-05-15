#!/usr/bin/env python
"""Analyze BMP4 sequential Promisys comparison artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples.cases.bmp4_gradient.sequential_analysis import (  # noqa: E402
    analyze_comparison_run,
    plot_sequential_posterior_comparison,
    plot_trace_acquisition_history,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Comparison root or a single sequential_* run directory.")
    parser.add_argument("--normalizer-data-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-posterior-draws", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="Only regenerate acquisition-level final EIG/design plots from sequential_trace.json.",
    )
    parser.add_argument(
        "--posterior-comparison-only",
        action="store_true",
        help="Only regenerate sequential initial-prior vs final-posterior comparison plots.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON payload instead of key paths.")
    args = parser.parse_args(argv)

    if args.trace_only and args.posterior_comparison_only:
        parser.error("--trace-only and --posterior-comparison-only are mutually exclusive.")
    if args.trace_only:
        result = plot_trace_acquisition_history(args.run_dir)
    elif args.posterior_comparison_only:
        result = plot_sequential_posterior_comparison(args.run_dir)
    else:
        result = analyze_comparison_run(
            args.run_dir,
            normalizer_data_dir=args.normalizer_data_dir,
            max_posterior_draws=args.max_posterior_draws,
            data_path=args.data_path,
            seed=args.seed,
        )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_summary_text(result))
    return 0


def _summary_text(result: dict[str, Any]) -> str:
    if "comparison_metrics" in result:
        lines = [
            f"comparison_metrics: {result['comparison_metrics']}",
            f"comparison_plot: {result['comparison_plot']}",
        ]
        for run in result.get("runs", []):
            lines.append(
                "sequential_run: "
                + str(run.get("prior_mode"))
                + f" metrics={run.get('metrics_json')} plot={run.get('diagnostics_plot')}"
            )
        return "\n".join(lines)
    if "run_count" in result and "runs" in result:
        return "\n".join(
            [
                f"comparison_dir: {result['comparison_dir']}",
                *[
                    "sequential_run: "
                    + str(run.get("prior_mode"))
                    + (
                        f" plot={run.get('eig_optimization_plot')}"
                        if run.get("eig_optimization_plot")
                        else f" positive_plot={run.get('prior_posterior_positive_plot')}"
                    )
                    for run in result.get("runs", [])
                ],
            ]
        )
    if "eig_optimization_plot" in result:
        return f"eig_optimization_plot: {result['eig_optimization_plot']}"
    return "\n".join(
        [
            f"metrics_json: {result['metrics_json']}",
            f"metrics_tsv: {result['metrics_tsv']}",
            f"diagnostics_plot: {result['diagnostics_plot']}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
