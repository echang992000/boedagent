"""Pyro backend adapter for variational BOED estimators."""

from __future__ import annotations

import math
from dataclasses import asdict
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
from boed_agent.utils.imports import instantiate_reference, resolve_reference


class PyroBackend(BackendAdapter):
    name = "pyro"
    description = "Live backend for variational BOED estimators from pyro.contrib.oed."

    SUPPORTED_ESTIMATORS = {"vi_eig", "posterior_eig", "marginal_eig", "vnmc_eig"}

    def supports(self, spec: ExperimentSpec) -> bool:
        return spec.backend == self.name or spec.model_ref is not None

    def required_fields(self, spec: ExperimentSpec | None = None) -> list[str]:
        estimator = None if spec is None else spec.objective.estimator
        fields = [
            "backend",
            "model_ref",
            "design_variables",
            "observation_labels",
            "target_latent_labels",
            "objective.estimator",
        ]
        if estimator == "vi_eig":
            fields.extend(
                [
                    "guide_ref",
                    "loss_ref",
                    "optim_ref",
                    "compute_budget.guide_training_steps",
                    "compute_budget.num_outer_samples",
                ]
            )
        elif estimator in {"posterior_eig", "marginal_eig"}:
            fields.extend(
                [
                    "guide_ref",
                    "optim_ref",
                    "compute_budget.guide_training_steps",
                    "compute_budget.num_outer_samples",
                ]
            )
        elif estimator == "vnmc_eig":
            fields.extend(
                [
                    "guide_ref",
                    "optim_ref",
                    "compute_budget.guide_training_steps",
                    "compute_budget.num_outer_samples",
                    "compute_budget.num_inner_samples",
                ]
            )
        return fields

    def validate(self, spec: ExperimentSpec) -> ValidationReport:
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        missing = [path for path in self.required_fields(spec) if field_is_missing(spec, path)]

        if spec.objective.estimator and spec.objective.estimator not in self.SUPPORTED_ESTIMATORS:
            errors.append(
                ValidationIssue(
                    path="objective.estimator",
                    message=(
                        f"Unsupported Pyro estimator '{spec.objective.estimator}'. "
                        f"Supported estimators: {sorted(self.SUPPORTED_ESTIMATORS)}."
                    ),
                )
            )

        if spec.design_variables:
            for variable in spec.design_variables:
                if variable.lower >= variable.upper:
                    errors.append(
                        ValidationIssue(
                            path="design_variables",
                            message=f"Design variable '{variable.name}' must have lower < upper.",
                        )
                    )
                if variable.shape != 1:
                    warnings.append(
                        ValidationIssue(
                            path="design_variables",
                            message=(
                                f"Design variable '{variable.name}' has shape={variable.shape}. "
                                "The v1 optimizer assumes flattened scalar design coordinates."
                            ),
                            severity="warning",
                        )
                    )

        candidate_designs = spec.backend_options.get("candidate_designs")
        if candidate_designs is not None:
            if not isinstance(candidate_designs, list) or not candidate_designs:
                errors.append(
                    ValidationIssue(
                        path="backend_options.candidate_designs",
                        message="`candidate_designs` must be a non-empty list of design vectors.",
                    )
                )
            else:
                expected_dim = len(spec.design_variables)
                for index, candidate in enumerate(candidate_designs):
                    if not isinstance(candidate, (list, tuple)):
                        errors.append(
                            ValidationIssue(
                                path=f"backend_options.candidate_designs[{index}]",
                                message="Each candidate design must be a list or tuple of numeric values.",
                            )
                        )
                        continue
                    if expected_dim and len(candidate) != expected_dim:
                        errors.append(
                            ValidationIssue(
                                path=f"backend_options.candidate_designs[{index}]",
                                message=(
                                    f"Candidate design {index} has dimension {len(candidate)} but "
                                    f"expected {expected_dim} values."
                                ),
                            )
                        )

        initial_design = spec.effective_initial_design()
        if spec.design_variables and len(initial_design) != len(spec.design_variables):
            errors.append(
                ValidationIssue(
                    path="initial_design",
                    message="Initial design dimensionality must match `design_variables`.",
                )
            )

        try:
            self._import_pyro_stack()
        except RuntimeError as exc:
            warnings.append(
                ValidationIssue(
                    path="backend",
                    message=str(exc),
                    severity="warning",
                )
            )

        for ref_field in ("model_ref", "guide_ref", "loss_ref", "optim_ref"):
            reference = getattr(spec, ref_field)
            if reference is None:
                continue
            try:
                if ref_field in {"loss_ref", "optim_ref"}:
                    instantiate_reference(reference, spec)
                else:
                    resolve_reference(reference)
            except Exception as exc:  # pragma: no cover - exact import errors vary by environment
                errors.append(
                    ValidationIssue(
                        path=ref_field,
                        message=f"Unable to resolve `{ref_field}` reference '{reference}': {exc}",
                    )
                )

        for path in sorted(set(missing)):
            errors.append(ValidationIssue(path=path, message=f"Missing required field `{path}`."))

        return ValidationReport(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            missing_fields=sorted(set(missing)),
            backend=self.name,
        )

    def estimate_eig(self, spec: ExperimentSpec, design: list[float] | None = None) -> EIGEstimate:
        pyro, eig_module, torch = self._import_pyro_stack()
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError(f"Spec is not valid for Pyro backend: {validation.to_dict()}")

        estimator_name = spec.objective.estimator
        assert estimator_name is not None
        estimator = getattr(eig_module, estimator_name)
        pyro.clear_param_store()
        design_tensor = self._make_design_tensor(torch, spec, design, requires_grad=False)
        kwargs = self._build_estimator_kwargs(spec, design_tensor, return_history=False)
        raw_result = estimator(**kwargs)
        eig_value, history = self._split_estimator_result(raw_result)

        return EIGEstimate(
            backend=self.name,
            estimator=estimator_name,
            design=self._tensor_to_list(design_tensor),
            value=eig_value,
            status="ok",
            warnings=[],
            metadata={
                "history": history,
                "config": {
                    "objective": asdict(spec.objective),
                    "compute_budget": asdict(spec.compute_budget),
                    "backend_options": spec.backend_options,
                },
            },
        )

    def optimize(self, spec: ExperimentSpec) -> OptimizationResult:
        pyro, _, torch = self._import_pyro_stack()
        validation = self.validate(spec)
        if not validation.valid:
            raise ValueError(f"Spec is not valid for Pyro backend: {validation.to_dict()}")

        warnings: list[str] = []
        steps = spec.compute_budget.num_optimization_steps or 10
        strategy = spec.backend_options.get("optimization_strategy", "gradient_then_random_search")

        if strategy == "candidate_grid":
            result = self._optimize_candidate_grid(spec)
            result.warnings = warnings + result.warnings
            return self._attach_requested_trajectory(spec, result)

        if strategy in {"gradient", "gradient_then_random_search"}:
            try:
                result = self._optimize_gradient(spec, torch, pyro, steps)
                result.warnings.extend(warnings)
                return self._attach_requested_trajectory(spec, result)
            except Exception as exc:  # pragma: no cover - fallback path depends on installed Pyro
                if strategy == "gradient":
                    raise
                warnings.append(
                    f"Gradient-based design optimization failed and random search fallback was used: {exc}"
                )

        result = self._optimize_random_search(spec, torch, pyro, steps)
        result.warnings = warnings + result.warnings
        return self._attach_requested_trajectory(spec, result)

    def describe(self, spec: ExperimentSpec | None = None) -> BackendDescriptor:
        return BackendDescriptor(
            name=self.name,
            description=self.description,
            capabilities={
                "supported_estimators": sorted(self.SUPPORTED_ESTIMATORS),
                "optimization_strategies": [
                    "candidate_grid",
                    "gradient",
                    "random_search",
                    "gradient_then_random_search",
                ],
                "accepts_model_import_refs": True,
                "accepts_guide_import_refs": True,
            },
            required_fields=self.required_fields(spec),
            status="available",
        )

    def _optimize_gradient(
        self,
        spec: ExperimentSpec,
        torch: Any,
        pyro: Any,
        steps: int,
    ) -> OptimizationResult:
        _, eig_module, _ = self._import_pyro_stack()
        estimator_name = spec.objective.estimator
        assert estimator_name is not None
        estimator = getattr(eig_module, estimator_name)

        design = self._make_design_tensor(torch, spec, None, requires_grad=True)
        optimizer = torch.optim.Adam(
            [design],
            lr=spec.compute_budget.design_learning_rate
            or spec.backend_options.get("design_learning_rate", 0.05),
        )
        lower, upper = self._make_bounds_tensors(torch, spec)

        history: list[OptimizationStep] = []
        final_warnings: list[str] = []

        for step in range(steps):
            pyro.clear_param_store()
            optimizer.zero_grad()
            kwargs = self._build_estimator_kwargs(spec, design, return_history=False)
            raw = estimator(**kwargs)
            eig_tensor = self._extract_value_tensor(raw, torch)
            eig_value = self._coerce_float(eig_tensor)
            if eig_value is None or math.isnan(eig_value):
                raise RuntimeError("Estimator returned a non-finite EIG value.")
            loss = -eig_tensor.sum()
            loss.backward()
            if design.grad is None:
                raise RuntimeError("No design gradient was produced by the estimator.")
            optimizer.step()
            with torch.no_grad():
                design.clamp_(lower, upper)
            history.append(
                OptimizationStep(
                    step=step,
                    design=self._tensor_to_list(design),
                    eig=float(eig_value),
                )
            )

        estimate = self.estimate_eig(spec, design=self._tensor_to_list(design))
        return OptimizationResult(
            backend=self.name,
            estimator=estimator_name,
            status="completed",
            design=estimate.design,
            eig=estimate.value,
            history=history,
            warnings=final_warnings,
            artifacts={"estimate_metadata": estimate.metadata},
        )

    def _optimize_random_search(
        self,
        spec: ExperimentSpec,
        torch: Any,
        pyro: Any,
        steps: int,
    ) -> OptimizationResult:
        lower, upper = self._make_bounds_tensors(torch, spec)
        best_design = spec.effective_initial_design()
        best_estimate = self.estimate_eig(spec, design=best_design)
        history = [
            OptimizationStep(step=0, design=best_design, eig=best_estimate.value, notes="initial")
        ]

        for step in range(1, steps + 1):
            candidate_tensor = lower + (upper - lower) * torch.rand_like(lower)
            candidate = self._tensor_to_list(candidate_tensor)
            estimate = self.estimate_eig(spec, design=candidate)
            history.append(OptimizationStep(step=step, design=candidate, eig=estimate.value))
            if estimate.value is not None and (
                best_estimate.value is None or estimate.value > best_estimate.value
            ):
                best_design = candidate
                best_estimate = estimate

        return OptimizationResult(
            backend=self.name,
            estimator=spec.objective.estimator,
            status="completed_with_warnings",
            design=best_design,
            eig=best_estimate.value,
            history=history,
            warnings=["Random search was used for design optimization."],
            artifacts={"best_estimate": best_estimate.to_dict()},
        )

    def _optimize_candidate_grid(self, spec: ExperimentSpec) -> OptimizationResult:
        candidate_designs = spec.backend_options.get("candidate_designs", [])
        if not candidate_designs:
            raise ValueError(
                "Candidate-grid optimization requires `backend_options.candidate_designs`."
            )

        history: list[OptimizationStep] = []
        best_estimate: EIGEstimate | None = None
        best_design: list[float] = spec.effective_initial_design()
        candidate_evaluations: list[dict[str, Any]] = []

        for step, raw_candidate in enumerate(candidate_designs):
            candidate = [float(value) for value in raw_candidate]
            estimate = self.estimate_eig(spec, design=candidate)
            note = "candidate_grid"
            history.append(
                OptimizationStep(
                    step=step,
                    design=candidate,
                    eig=estimate.value,
                    notes=note,
                )
            )
            candidate_evaluations.append(
                {
                    "step": step,
                    "design": candidate,
                    "eig": estimate.value,
                }
            )
            if best_estimate is None or (
                estimate.value is not None
                and (best_estimate.value is None or estimate.value > best_estimate.value)
            ):
                best_design = candidate
                best_estimate = estimate

        assert best_estimate is not None
        return OptimizationResult(
            backend=self.name,
            estimator=spec.objective.estimator,
            status="completed",
            design=best_design,
            eig=best_estimate.value,
            history=history,
            warnings=[],
            artifacts={
                "best_estimate": best_estimate.to_dict(),
                "candidate_evaluations": candidate_evaluations,
                "optimization_strategy": "candidate_grid",
            },
        )

    def _import_pyro_stack(self) -> tuple[Any, Any, Any]:
        try:
            import pyro
            import torch
            from pyro.contrib.oed import eig as eig_module
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Pyro backend requires optional dependencies `pyro-ppl` and `torch`. Install with `.[pyro]`."
            ) from exc
        return pyro, eig_module, torch

    def _build_estimator_kwargs(
        self,
        spec: ExperimentSpec,
        design_tensor: Any,
        return_history: bool,
    ) -> dict[str, Any]:
        estimator = spec.objective.estimator
        assert estimator is not None
        kwargs: dict[str, Any] = {
            "model": resolve_reference(spec.model_ref),
            "design": design_tensor,
            "observation_labels": list(spec.observation_labels),
            "target_labels": list(spec.target_latent_labels),
        }
        objective_kwargs = dict(spec.objective.estimator_kwargs)
        backend_kwargs = dict(spec.backend_options.get("estimator_kwargs", {}))

        if estimator == "vi_eig":
            kwargs["vi_parameters"] = {
                "guide": resolve_reference(spec.guide_ref),
                "loss": instantiate_reference(spec.loss_ref, spec),
                "optim": instantiate_reference(spec.optim_ref, spec),
                "num_steps": spec.compute_budget.guide_training_steps,
            }
            kwargs["is_parameters"] = {
                "num_samples": spec.compute_budget.num_outer_samples,
            }
            kwargs.update(objective_kwargs)
            kwargs.update(backend_kwargs)
        elif estimator in {"posterior_eig", "marginal_eig"}:
            kwargs.update(
                {
                    "num_samples": spec.compute_budget.num_outer_samples,
                    "num_steps": spec.compute_budget.guide_training_steps,
                    "guide": resolve_reference(spec.guide_ref),
                    "optim": instantiate_reference(spec.optim_ref, spec),
                    "return_history": return_history,
                }
            )
            if "final_num_samples" in spec.backend_options:
                kwargs["final_num_samples"] = spec.backend_options["final_num_samples"]
            kwargs.update(objective_kwargs)
            kwargs.update(backend_kwargs)
        elif estimator == "vnmc_eig":
            kwargs.update(
                {
                    "num_samples": (
                        spec.compute_budget.num_outer_samples,
                        spec.compute_budget.num_inner_samples,
                    ),
                    "num_steps": spec.compute_budget.guide_training_steps,
                    "guide": resolve_reference(spec.guide_ref),
                    "optim": instantiate_reference(spec.optim_ref, spec),
                    "return_history": return_history,
                }
            )
            if "final_num_samples" in spec.backend_options:
                kwargs["final_num_samples"] = tuple(spec.backend_options["final_num_samples"])
            kwargs.update(objective_kwargs)
            kwargs.update(backend_kwargs)
        else:  # pragma: no cover - guarded by validation
            raise ValueError(f"Unsupported estimator {estimator!r}")

        return kwargs

    def _make_design_tensor(
        self,
        torch: Any,
        spec: ExperimentSpec,
        design: list[float] | None,
        requires_grad: bool,
    ) -> Any:
        values = design or spec.effective_initial_design()
        tensor = torch.tensor(values, dtype=torch.float32)
        tensor.requires_grad_(requires_grad)
        return tensor

    def _make_bounds_tensors(self, torch: Any, spec: ExperimentSpec) -> tuple[Any, Any]:
        lower = torch.tensor([variable.lower for variable in spec.design_variables], dtype=torch.float32)
        upper = torch.tensor([variable.upper for variable in spec.design_variables], dtype=torch.float32)
        return lower, upper

    def _split_estimator_result(self, result: Any) -> tuple[float | None, list[float] | None]:
        if isinstance(result, tuple):
            value = self._coerce_float(result[0])
            history = self._coerce_history(result[1])
            return value, history
        return self._coerce_float(result), None

    def _extract_value_tensor(self, result: Any, torch: Any) -> Any:
        value = result[0] if isinstance(result, tuple) else result
        if hasattr(value, "detach"):
            return value
        return torch.as_tensor(value, dtype=torch.float32)

    def _coerce_float(self, value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                detached = value.detach()
                if hasattr(detached, "mean"):
                    return float(detached.mean().cpu().item())
            return float(value)
        except Exception:
            return None

    def _coerce_history(self, history: Any) -> list[float] | None:
        if history is None:
            return None
        if hasattr(history, "detach"):
            return [float(item) for item in history.detach().cpu().reshape(-1).tolist()]
        if isinstance(history, (list, tuple)):
            return [float(item) for item in history]
        return None

    def _tensor_to_list(self, tensor: Any) -> list[float]:
        if hasattr(tensor, "detach"):
            return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]
        return [float(value) for value in tensor]

    def _attach_requested_trajectory(
        self,
        spec: ExperimentSpec,
        result: OptimizationResult,
    ) -> OptimizationResult:
        if not spec.wants_recreated_trajectory():
            return result
        result.artifacts["trajectory_recreated"] = True
        histories = [[step.to_dict() for step in result.history]]
        result.artifacts["optimized_design_histories"] = histories
        result.artifacts["optimized_design_history_summaries"] = summarize_optimized_design_histories(
            [result.history],
            spec.design_variables,
        )
        return result
