"""LFIAX backend adapter backed by the external cli-anything-lfiax CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from boed_agent.backends.base import BackendAdapter
from boed_agent.models import (
    BackendDescriptor,
    EIGEstimate,
    ExperimentSpec,
    OptimizationResult,
    OptimizationStep,
    ValidationIssue,
    ValidationReport,
    field_is_missing,
)
from boed_agent.utils.trajectory import summarize_optimized_design_histories


class LFIAXBackend(BackendAdapter):
    name = "lfiax"
    description = "Simulator-first BOED adapter backed by cli-anything-lfiax."

    def supports(self, spec: ExperimentSpec) -> bool:
        return spec.backend == self.name or spec.simulator_ref is not None

    def required_fields(self, spec: ExperimentSpec | None = None) -> list[str]:
        return [
            "backend",
            "simulator_ref",
            "design_variables",
            "backend_options.design_mode",
            "differentiable",
            "objective.estimator",
            "prior_sampler_ref_or_latent_sampler_ref",
        ]

    def validate(self, spec: ExperimentSpec) -> ValidationReport:
        local_errors: list[ValidationIssue] = []
        missing = [
            path
            for path in self.required_fields(spec)
            if path != "prior_sampler_ref_or_latent_sampler_ref" and field_is_missing(spec, path)
        ]
        if spec.prior_sampler_ref is None and spec.latent_sampler_ref is None:
            missing.append("prior_sampler_ref_or_latent_sampler_ref")

        try:
            payload = self._run_cli_json(["oed", "validate"], spec)
        except RuntimeError as exc:
            return ValidationReport(
                valid=False,
                errors=[ValidationIssue(path="backend", message=str(exc))] + [
                    self._missing_field_issue(path) for path in sorted(set(missing))
                ],
                backend=self.name,
            )
        report = self._validation_from_payload(payload)
        for path in sorted(set(missing)):
            local_errors.append(self._missing_field_issue(path))
        report.errors.extend(local_errors)
        report.missing_fields = sorted(set(report.missing_fields + missing))
        report.valid = not report.errors
        return report

    def estimate_eig(self, spec: ExperimentSpec, design: list[float] | None = None) -> EIGEstimate:
        estimate_spec = spec.to_dict()
        chosen_design = [float(value) for value in (design or spec.effective_initial_design())]
        estimate_spec["initial_design"] = chosen_design
        estimate_spec.setdefault("backend_options", {})
        estimate_spec["backend_options"]["design_mode"] = "point"
        estimate_spec.setdefault("compute_budget", {})
        estimate_spec["compute_budget"]["num_optimization_steps"] = 1
        estimate_spec["compute_budget"]["design_learning_rate"] = 0.0

        try:
            payload = self._run_cli_json(
                ["oed", "optimize"],
                ExperimentSpec.from_dict(estimate_spec),
                extra_args=["--no-artifacts"],
            )
        except RuntimeError as exc:
            return EIGEstimate(
                backend=self.name,
                estimator=spec.objective.estimator,
                design=chosen_design,
                value=None,
                status="error",
                warnings=[str(exc)],
                metadata={},
            )
        warnings = list(payload.get("warnings", []))
        if payload.get("status") != "completed":
            warnings.append(payload.get("error", "LFIAX estimate run did not complete normally."))
        return EIGEstimate(
            backend=self.name,
            estimator=payload.get("estimator", spec.objective.estimator),
            design=[float(value) for value in payload.get("design", chosen_design)],
            value=payload.get("eig"),
            status=payload.get("status", "unknown"),
            warnings=warnings,
            metadata={
                "execution_path": payload.get("execution_path"),
                "artifacts": payload.get("artifacts", {}),
            },
        )

    def optimize(self, spec: ExperimentSpec) -> OptimizationResult:
        try:
            payload = self._run_cli_json(["oed", "optimize"], spec)
        except RuntimeError as exc:
            return OptimizationResult(
                backend=self.name,
                estimator=spec.objective.estimator,
                status="failed",
                design=spec.effective_initial_design(),
                eig=None,
                history=[],
                warnings=[str(exc)],
                artifacts={},
            )
        history = self._history_from_payload(payload, spec)
        warnings = list(payload.get("warnings", []))
        if payload.get("status") not in {"completed", "dry_run"} and payload.get("error"):
            warnings.append(payload["error"])
        result = OptimizationResult(
            backend=self.name,
            estimator=payload.get("estimator", spec.objective.estimator),
            status=self._normalize_status(payload.get("status")),
            design=[float(value) for value in payload.get("design", spec.effective_initial_design())],
            eig=payload.get("eig"),
            history=history,
            warnings=warnings,
            artifacts=dict(payload.get("artifacts", {})),
        )
        if payload.get("execution_path") is not None:
            result.artifacts.setdefault("execution_path", payload["execution_path"])
        if payload.get("xi_mu") is not None:
            result.artifacts.setdefault("xi_mu", payload["xi_mu"])
        if payload.get("xi_stddev") is not None:
            result.artifacts.setdefault("xi_stddev", payload["xi_stddev"])
        if spec.wants_recreated_trajectory() and result.history:
            result.artifacts["trajectory_recreated"] = True
            result.artifacts["optimized_design_histories"] = [
                [step.to_dict() for step in result.history]
            ]
            result.artifacts["optimized_design_history_summaries"] = summarize_optimized_design_histories(
                [result.history],
                spec.design_variables,
            )
        return result

    def describe(self, spec: ExperimentSpec | None = None) -> BackendDescriptor:
        try:
            payload = self._run_cli_json(["oed", "describe"])
        except RuntimeError as exc:
            return BackendDescriptor(
                name=self.name,
                description=self.description,
                capabilities={"error": str(exc)},
                required_fields=self.required_fields(spec),
                status="unavailable",
            )
        return BackendDescriptor(
            name=payload.get("name", self.name),
            description=payload.get("description", self.description),
            capabilities=dict(payload.get("capabilities", {})),
            required_fields=list(payload.get("required_fields", self.required_fields(spec))),
            status=payload.get("status", "available"),
        )

    def _resolve_cli(self) -> list[str]:
        cli_path = shutil.which("cli-anything-lfiax")
        if cli_path is None:
            raise RuntimeError(
                "cli-anything-lfiax not found on PATH. Install the lfiax agent harness "
                "(see https://github.com/vz415/lfiax — `pip install -e /path/to/lfiax/agent-harness`) "
                "and ensure the `cli-anything-lfiax` entry point is on your PATH."
            )
        return [cli_path]

    def _run_cli_json(
        self,
        args: list[str],
        spec: ExperimentSpec | None = None,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        command = self._resolve_cli() + ["--json"] + args

        # A TemporaryDirectory is portable across Linux/macOS/Windows and guarantees
        # cleanup even if the subprocess still holds a file handle on macOS, where
        # `NamedTemporaryFile(delete=False)` + manual unlink has historically been
        # fragile under concurrent access.
        with tempfile.TemporaryDirectory(prefix="boed_lfiax_") as tmpdir:
            if spec is not None:
                spec_path = Path(tmpdir) / "spec.json"
                spec_path.write_text(json.dumps(spec.to_dict()))
                command.append(str(spec_path))
            if extra_args:
                command.extend(extra_args)

            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

        if proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip() or "Unknown cli-anything-lfiax failure."
            raise RuntimeError(message)
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"cli-anything-lfiax returned invalid JSON: {proc.stdout!r}"
            ) from exc

    def _validation_from_payload(self, payload: dict[str, Any]) -> ValidationReport:
        return ValidationReport(
            valid=bool(payload.get("valid", False)),
            errors=[self._validation_issue(item, severity="error") for item in payload.get("errors", [])],
            warnings=[
                self._validation_issue(item, severity="warning") for item in payload.get("warnings", [])
            ],
            missing_fields=list(payload.get("missing_fields", [])),
            backend=payload.get("backend", self.name),
        )

    def _validation_issue(self, item: dict[str, Any], severity: str) -> ValidationIssue:
        return ValidationIssue(
            path=str(item.get("path", "backend")),
            message=str(item.get("message", "Unknown validation issue.")),
            severity=str(item.get("severity", severity)),
        )

    def _missing_field_issue(self, path: str) -> ValidationIssue:
        if path == "prior_sampler_ref_or_latent_sampler_ref":
            return ValidationIssue(
                path=path,
                message="Provide either `prior_sampler_ref` or `latent_sampler_ref` for simulator workflows.",
            )
        return ValidationIssue(path=path, message=f"Missing required field `{path}`.")

    def _history_from_payload(
        self,
        payload: dict[str, Any],
        spec: ExperimentSpec,
    ) -> list[OptimizationStep]:
        history: list[OptimizationStep] = []
        for item in payload.get("history", []):
            design = item.get("design") or item.get("xi_mu") or payload.get("design") or spec.effective_initial_design()
            history.append(
                OptimizationStep(
                    step=int(item.get("step", len(history))),
                    design=[float(value) for value in design],
                    eig=item.get("eig"),
                    notes=item.get("notes"),
                )
            )
        return history

    def _normalize_status(self, status: Any) -> str:
        normalized = str(status or "unknown")
        if normalized == "invalid_spec":
            return "failed"
        if normalized == "error":
            return "failed"
        return normalized
