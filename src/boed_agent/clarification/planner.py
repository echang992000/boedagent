"""Clarification planner for incomplete BOED experiment specifications."""

from __future__ import annotations

from boed_agent.backends.registry import BackendRegistry
from boed_agent.clarification.profiles import QUESTION_TEMPLATES
from boed_agent.models import ClarificationQuestion, ExperimentSpec, field_is_missing


class ClarificationPlanner:
    def __init__(self, backend_registry: BackendRegistry) -> None:
        self.backend_registry = backend_registry

    def plan(self, spec: ExperimentSpec) -> list[ClarificationQuestion]:
        backend_name = self._infer_backend_name(spec)
        required_fields: list[str] = []
        if backend_name is None:
            required_fields.append("backend")
        else:
            required_fields.extend(self.backend_registry.get(backend_name).required_fields(spec))
        required_fields.extend(self._literature_fields(spec))

        questions: list[ClarificationQuestion] = []
        for field in self._ordered_fields(required_fields):
            if field == "backend" and backend_name is not None:
                continue
            if field == "backend" and not field_is_missing(spec, field):
                continue
            if field == "prior_sampler_ref_or_latent_sampler_ref":
                if spec.prior_sampler_ref or spec.latent_sampler_ref:
                    continue
                question = QUESTION_TEMPLATES["prior_sampler_ref"]
                questions.append(question)
                continue
            if field_is_missing(spec, field):
                template = QUESTION_TEMPLATES.get(field)
                if template is not None:
                    questions.append(template)
        return questions

    def _infer_backend_name(self, spec: ExperimentSpec) -> str | None:
        if spec.backend:
            return spec.backend
        if spec.model_ref:
            return "pyro"
        if spec.simulator_ref:
            return "lfiax"
        return None

    def _ordered_fields(self, fields: list[str]) -> list[str]:
        order = [
            "backend",
            "literature_source_mode",
            "literature_corpus_dir",
            "model_ref",
            "simulator_ref",
            "design_variables",
            "backend_options.design_mode",
            "observation_labels",
            "target_latent_labels",
            "objective.estimator",
            "guide_ref",
            "loss_ref",
            "optim_ref",
            "compute_budget.num_outer_samples",
            "compute_budget.num_inner_samples",
            "compute_budget.guide_training_steps",
            "prior_sampler_ref_or_latent_sampler_ref",
            "differentiable",
        ]
        rank = {name: index for index, name in enumerate(order)}
        return sorted(dict.fromkeys(fields), key=lambda field: rank.get(field, len(order) + 1))

    def _literature_fields(self, spec: ExperimentSpec) -> list[str]:
        if not spec.wants_literature():
            return []
        fields = []
        if field_is_missing(spec, "literature_source_mode"):
            fields.append("literature_source_mode")
            return fields
        source_mode = (spec.literature_source_mode or "").strip().lower()
        if source_mode in {"local", "both"} and field_is_missing(
            spec, "literature_corpus_dir"
        ):
            fields.append("literature_corpus_dir")
        return fields
