"""Interactive CLI intake for incomplete BOED experiment specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from boed_agent.clarification.planner import ClarificationPlanner
from boed_agent.demo.foster_variational import (
    PAPER_TITLE,
    PAPER_URL,
    REPRO_URL,
    build_foster_experiment_spec,
    get_foster_experiment_definition,
    list_foster_experiment_definitions,
)
from boed_agent.models import (
    ArtifactSettings,
    ClarificationQuestion,
    DesignVariable,
    ExperimentSpec,
    to_jsonable,
)


DEMO_LINEAR_SUMMARY = "Optimize a single scalar design using Pyro marginal EIG."
DEMO_LINEAR_MODEL_REF = "boed_agent.demo.pyro_linear:pyro_linear_model"
DEMO_LINEAR_GUIDE_REFS = {
    "marginal_eig": "boed_agent.demo.pyro_linear:pyro_linear_marginal_guide",
    "posterior_eig": "boed_agent.demo.pyro_linear:pyro_linear_posterior_guide",
    "vnmc_eig": "boed_agent.demo.pyro_linear:pyro_linear_posterior_guide",
    "vi_eig": "boed_agent.demo.pyro_linear:pyro_linear_vi_guide",
}
DEMO_LINEAR_LOSS_REF = "boed_agent.demo.pyro_linear:make_trace_elbo_loss"
DEMO_LINEAR_OPTIM_REF = "boed_agent.demo.pyro_linear:make_pyro_adam"
DEMO_LINEAR_OBSERVATION_LABELS = ["y"]
DEMO_LINEAR_TARGET_LABELS = ["theta"]
DEMO_LINEAR_DESIGN_VARIABLE = DesignVariable(name="design", lower=-2.0, upper=2.0, initial=0.25)
DEMO_FOSTER_TEMPLATE = "foster_variational"


@dataclass(frozen=True)
class InteractiveOption:
    label: str
    value: str
    description: str


def collect_interactive_spec(
    spec: ExperimentSpec,
    planner: ClarificationPlanner,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> tuple[ExperimentSpec, list[dict[str, Any]]]:
    input_fn = input_fn or input
    output_fn = output_fn or print
    transcript: list[dict[str, Any]] = []
    spec, template_transcript = _apply_demo_template(
        spec,
        input_fn,
        output_fn,
    )
    transcript.extend(template_transcript)

    while True:
        questions = planner.plan(spec)
        if not questions:
            break

        question = questions[0]
        answer = _answer_question(spec, question, input_fn, output_fn)
        _apply_answer(spec, question.field, answer)
        transcript.append(
            {
                "field": question.field,
                "prompt": question.prompt,
                "reason": question.reason,
                "answer": to_jsonable(answer),
            }
        )

    return spec, transcript


def _apply_demo_template(
    spec: ExperimentSpec,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> tuple[ExperimentSpec, list[dict[str, Any]]]:
    template = str(spec.metadata.get("demo_template", "")).strip().lower()
    if template != DEMO_FOSTER_TEMPLATE:
        return spec, []
    return _seed_foster_variational_spec(spec, input_fn, output_fn)


def _seed_foster_variational_spec(
    spec: ExperimentSpec,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> tuple[ExperimentSpec, list[dict[str, Any]]]:
    transcript: list[dict[str, Any]] = []
    paper_question = ClarificationQuestion(
        field="metadata.paper_experiment",
        prompt="Which Foster et al. paper experiment should be used?",
        reason="The paper defaults depend on the benchmark case and estimator family.",
        backend="pyro",
    )

    experiment_key = str(spec.metadata.get("paper_experiment", "")).strip()
    if not experiment_key:
        definitions = list_foster_experiment_definitions()
        experiment_option = _select_option(
            paper_question,
            [
                InteractiveOption(
                    label=definition.label,
                    value=definition.key,
                    description=definition.description,
                )
                for definition in definitions
            ],
            input_fn,
            output_fn,
        )
        experiment_key = experiment_option.value
        transcript.append(
            {
                "field": "metadata.paper_experiment",
                "prompt": paper_question.prompt,
                "reason": paper_question.reason,
                "answer": experiment_key,
            }
        )

    definition = get_foster_experiment_definition(experiment_key)
    estimator = spec.objective.estimator or str(spec.metadata.get("paper_estimator", "")).strip()
    if not estimator:
        estimator_question = ClarificationQuestion(
            field="objective.estimator",
            prompt=f"Which BOED estimator should be used for {definition.label}?",
            reason="Each estimator uses a different guide setup and paper-tuned compute budget.",
            backend="pyro",
        )
        estimator_option = _select_option(
            estimator_question,
            _foster_estimator_options(definition.key, definition.estimator_options),
            input_fn,
            output_fn,
        )
        estimator = estimator_option.value
        transcript.append(
            {
                "field": "objective.estimator",
                "prompt": estimator_question.prompt,
                "reason": estimator_question.reason,
                "answer": estimator,
            }
        )

    selected_spec = build_foster_experiment_spec(experiment_key, estimator)
    selected_spec.metadata = {
        **selected_spec.metadata,
        "demo_template": DEMO_FOSTER_TEMPLATE,
        "paper_title": PAPER_TITLE,
        "paper_url": PAPER_URL,
        "reproduction_url": REPRO_URL,
    }
    _apply_artifact_overrides(source=spec, destination=selected_spec)
    if spec.recreate_trajectory:
        selected_spec.recreate_trajectory = True
    if spec.provider is not None:
        selected_spec.provider = spec.provider
    return selected_spec, transcript


def _answer_question(
    spec: ExperimentSpec,
    question: ClarificationQuestion,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> Any:
    field = question.field

    if field == "backend":
        return _select_option(
            question,
            [
                InteractiveOption(
                    label="Use the Pyro backend",
                    value="pyro",
                    description="Explicit probabilistic model with variational OED estimators.",
                ),
                InteractiveOption(
                    label="Use the LFIAX backend",
                    value="lfiax",
                    description="Simulator-first workflow backed by the external cli-anything-lfiax optimizer.",
                ),
            ],
            input_fn,
            output_fn,
        ).value

    if field == "objective.estimator":
        return _select_option(
            question,
            [
                InteractiveOption(
                    label="marginal_eig",
                    value="marginal_eig",
                    description="Recommended for the bundled linear-regression demo.",
                ),
                InteractiveOption(
                    label="posterior_eig",
                    value="posterior_eig",
                    description="Posterior-guide estimator for explicit latent targets.",
                ),
                InteractiveOption(
                    label="vi_eig",
                    value="vi_eig",
                    description="VI estimator that requires an explicit loss callable.",
                ),
                InteractiveOption(
                    label="vnmc_eig",
                    value="vnmc_eig",
                    description="Nested Monte Carlo estimator with both outer and inner sample budgets.",
                ),
            ],
            input_fn,
            output_fn,
        ).value

    if field == "model_ref":
        option = _select_option(
            question,
            [
                InteractiveOption(
                    label="Use the bundled Pyro linear-regression demo model",
                    value=DEMO_LINEAR_MODEL_REF,
                    description=DEMO_LINEAR_MODEL_REF,
                ),
                InteractiveOption(
                    label="Enter a custom Python import reference",
                    value="__custom__",
                    description="Format: package.module:function_name",
                ),
            ],
            input_fn,
            output_fn,
        )
        if option.value != "__custom__":
            return option.value
        return _prompt_python_reference(
            "Model reference",
            input_fn,
            output_fn,
            example=DEMO_LINEAR_MODEL_REF,
        )

    if field == "guide_ref":
        recommended = _recommended_guide_ref(spec)
        options = []
        if recommended is not None:
            options.append(
                InteractiveOption(
                    label="Use the bundled guide for the selected estimator",
                    value=recommended,
                    description=recommended,
                )
            )
        options.append(
            InteractiveOption(
                label="Enter a custom Python import reference",
                value="__custom__",
                description="Format: package.module:function_name",
            )
        )
        option = _select_option(question, options, input_fn, output_fn)
        if option.value != "__custom__":
            return option.value
        return _prompt_python_reference(
            "Guide reference",
            input_fn,
            output_fn,
            example=recommended or "my_project.guides:my_guide",
        )

    if field == "loss_ref":
        option = _select_option(
            question,
            [
                InteractiveOption(
                    label="Use the bundled Trace_ELBO loss factory",
                    value=DEMO_LINEAR_LOSS_REF,
                    description=DEMO_LINEAR_LOSS_REF,
                ),
                InteractiveOption(
                    label="Enter a custom Python import reference",
                    value="__custom__",
                    description="Format: package.module:function_name",
                ),
            ],
            input_fn,
            output_fn,
        )
        if option.value != "__custom__":
            return option.value
        return _prompt_python_reference(
            "Loss reference",
            input_fn,
            output_fn,
            example=DEMO_LINEAR_LOSS_REF,
        )

    if field == "optim_ref":
        option = _select_option(
            question,
            [
                InteractiveOption(
                    label="Use the bundled Pyro Adam optimizer factory",
                    value=DEMO_LINEAR_OPTIM_REF,
                    description=DEMO_LINEAR_OPTIM_REF,
                ),
                InteractiveOption(
                    label="Enter a custom Python import reference",
                    value="__custom__",
                    description="Format: package.module:function_name",
                ),
            ],
            input_fn,
            output_fn,
        )
        if option.value != "__custom__":
            return option.value
        return _prompt_python_reference(
            "Optimizer reference",
            input_fn,
            output_fn,
            example=DEMO_LINEAR_OPTIM_REF,
        )

    if field == "simulator_ref":
        return _prompt_python_reference(
            "Simulator reference",
            input_fn,
            output_fn,
            example="my_project.simulators:simulate",
            question=question,
        )

    if field == "prior_sampler_ref":
        option = _select_option(
            question,
            [
                InteractiveOption(
                    label="Provide a prior sampler reference",
                    value="prior_sampler_ref",
                    description="The backend will call this to sample latent parameters before simulation.",
                ),
                InteractiveOption(
                    label="Provide a latent sampler reference",
                    value="latent_sampler_ref",
                    description="Use this if the project already exposes a latent-sampling callable.",
                ),
            ],
            input_fn,
            output_fn,
        )
        reference = _prompt_python_reference(
            "Sampler reference",
            input_fn,
            output_fn,
            example="my_project.simulators:sample_prior",
        )
        return {option.value: reference}

    if field == "design_variables":
        option = _select_option(
            question,
            [
                InteractiveOption(
                    label="Use the bundled scalar design variable",
                    value="__demo__",
                    description="`design` with bounds [-2.0, 2.0] and initial value 0.25.",
                ),
                InteractiveOption(
                    label="Define design variables manually",
                    value="__manual__",
                    description="Step through the variable names, bounds, and initials in the terminal.",
                ),
            ],
            input_fn,
            output_fn,
        )
        if option.value == "__demo__":
            return [DEMO_LINEAR_DESIGN_VARIABLE]
        return _prompt_design_variables(input_fn, output_fn)

    if field == "backend_options.design_mode":
        return _select_option(
            question,
            [
                InteractiveOption(
                    label="point",
                    value="point",
                    description="Optimize a single best design vector directly.",
                ),
                InteractiveOption(
                    label="distribution",
                    value="distribution",
                    description="Optimize an annealed design distribution over the design slots.",
                ),
            ],
            input_fn,
            output_fn,
        ).value

    if field == "observation_labels":
        option = _select_option(
            question,
            [
                InteractiveOption(
                    label="Use the bundled observation label",
                    value="__demo__",
                    description="`y` for the linear-regression demo.",
                ),
                InteractiveOption(
                    label="Enter custom observation labels",
                    value="__custom__",
                    description="Comma-separated labels or a JSON list.",
                ),
            ],
            input_fn,
            output_fn,
        )
        if option.value == "__demo__":
            return list(DEMO_LINEAR_OBSERVATION_LABELS)
        return _prompt_string_list(
            "Observation labels",
            input_fn,
            output_fn,
            example="y",
        )

    if field == "target_latent_labels":
        option = _select_option(
            question,
            [
                InteractiveOption(
                    label="Use the bundled target latent label",
                    value="__demo__",
                    description="`theta` for the linear-regression demo.",
                ),
                InteractiveOption(
                    label="Enter custom target latent labels",
                    value="__custom__",
                    description="Comma-separated labels or a JSON list.",
                ),
            ],
            input_fn,
            output_fn,
        )
        if option.value == "__demo__":
            return list(DEMO_LINEAR_TARGET_LABELS)
        return _prompt_string_list(
            "Target latent labels",
            input_fn,
            output_fn,
            example="theta",
        )

    if field == "compute_budget.num_outer_samples":
        return _prompt_budget_value(
            question,
            default=4,
            input_fn=input_fn,
            output_fn=output_fn,
        )

    if field == "compute_budget.num_inner_samples":
        return _prompt_budget_value(
            question,
            default=4,
            input_fn=input_fn,
            output_fn=output_fn,
        )

    if field == "compute_budget.guide_training_steps":
        return _prompt_budget_value(
            question,
            default=2,
            input_fn=input_fn,
            output_fn=output_fn,
        )

    if field == "differentiable":
        return _select_option(
            question,
            [
                InteractiveOption(
                    label="false",
                    value="false",
                    description="Treat the simulator as non-differentiable for now.",
                ),
                InteractiveOption(
                    label="true",
                    value="true",
                    description="Record that the simulator is differentiable for future optimization paths.",
                ),
            ],
            input_fn,
            output_fn,
        ).value == "true"

    return _prompt_text_value(question, input_fn, output_fn)


def _select_option(
    question: ClarificationQuestion,
    options: list[InteractiveOption],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> InteractiveOption:
    output_fn("")
    output_fn(f"Question: {question.prompt}")
    output_fn(f"Why this matters: {question.reason}")
    output_fn("Options:")
    for index, option in enumerate(options, start=1):
        output_fn(f"  {index}. {option.label}")
        output_fn(f"     {option.description}")

    accepted = {str(index): option for index, option in enumerate(options, start=1)}
    custom_option_indexes = [
        index
        for index, option in enumerate(options, start=1)
        if _looks_like_custom_option(option)
    ]
    while True:
        raw = input_fn("Selection (enter option number, default 1): ").strip()
        if raw == "":
            return options[0]
        option = accepted.get(raw)
        if option is not None:
            return option
        output_fn("Options:")
        output_fn("  - Enter one of the option numbers shown above.")
        if custom_option_indexes:
            indexes = ", ".join(str(index) for index in custom_option_indexes)
            output_fn(
                "  - If you want to type your own value, first choose the custom/manual "
                f"option number ({indexes})."
            )
        if any(raw == option.value for option in options):
            output_fn("  - This prompt expects the option number, not the option text or value.")


def _prompt_text_value(
    question: ClarificationQuestion,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    output_fn("")
    output_fn(f"Question: {question.prompt}")
    output_fn(f"Why this matters: {question.reason}")
    output_fn("Options:")
    output_fn("  - Enter a non-empty value.")
    while True:
        raw = input_fn("Value: ").strip()
        if raw:
            return raw
        output_fn("Options:")
        output_fn("  - A non-empty value is required.")


def _prompt_python_reference(
    label: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    example: str,
    question: ClarificationQuestion | None = None,
) -> str:
    if question is not None:
        output_fn("")
        output_fn(f"Question: {question.prompt}")
        output_fn(f"Why this matters: {question.reason}")
    while True:
        output_fn("Options:")
        output_fn("  - Enter a Python import reference in the form `package.module:function_name`.")
        output_fn(f"  - Example: `{example}`")
        raw = input_fn(f"{label}: ").strip()
        if raw and ":" in raw:
            return raw
        output_fn("Options:")
        output_fn("  - The reference must include a module path and a callable name separated by `:`.")


def _prompt_budget_value(
    question: ClarificationQuestion,
    default: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> int:
    option = _select_option(
        question,
        [
            InteractiveOption(
                label=f"Use the recommended demo value `{default}`",
                value=str(default),
                description="Small default that keeps the linear-regression demo fast.",
            ),
            InteractiveOption(
                label="Enter a custom positive integer",
                value="__custom__",
                description="Use this if you want a different compute budget.",
            ),
        ],
        input_fn,
        output_fn,
    )
    if option.value != "__custom__":
        return int(option.value)
    return _prompt_positive_int("Budget value", input_fn, output_fn, example=str(default))


def _prompt_positive_int(
    label: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    example: str,
) -> int:
    while True:
        output_fn("Options:")
        output_fn("  - Enter a positive integer.")
        output_fn(f"  - Example: `{example}`")
        raw = input_fn(f"{label}: ").strip()
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
        output_fn("Options:")
        output_fn("  - A positive integer is required.")


def _prompt_float(
    label: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    example: str,
    allow_empty: bool = False,
) -> float | None:
    while True:
        output_fn("Options:")
        output_fn("  - Enter a numeric value.")
        if allow_empty:
            output_fn("  - Press Enter to leave this value unset.")
        output_fn(f"  - Example: `{example}`")
        raw = input_fn(f"{label}: ").strip()
        if allow_empty and raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            output_fn("Options:")
            output_fn("  - A numeric value is required.")


def _prompt_string_list(
    label: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    example: str,
) -> list[str]:
    while True:
        output_fn("Options:")
        output_fn("  - Enter a comma-separated list.")
        output_fn("  - Or enter a JSON list like `[\"y\"]`.")
        output_fn(f"  - Example: `{example}`")
        raw = input_fn(f"{label}: ").strip()
        values = _parse_string_list(raw)
        if values:
            return values
        output_fn("Options:")
        output_fn("  - At least one label is required.")


def _parse_string_list(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _prompt_design_variables(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> list[DesignVariable]:
    count = _prompt_positive_int(
        "How many design variables should be defined?",
        input_fn,
        output_fn,
        example="1",
    )

    variables: list[DesignVariable] = []
    for index in range(count):
        output_fn("")
        output_fn(f"Design variable {index + 1} of {count}")

        name = _prompt_design_variable_name(index, input_fn, output_fn)
        while True:
            lower_value = _prompt_float(
                "Lower bound",
                input_fn,
                output_fn,
                example="-2.0",
            )
            upper_value = _prompt_float(
                "Upper bound",
                input_fn,
                output_fn,
                example="2.0",
            )
            assert lower_value is not None
            assert upper_value is not None
            if lower_value < upper_value:
                break
            output_fn("Options:")
            output_fn("  - The lower bound must be strictly smaller than the upper bound.")

        midpoint = (lower_value + upper_value) / 2.0
        initial_value = _prompt_float(
            "Initial value",
            input_fn,
            output_fn,
            example=str(midpoint),
            allow_empty=True,
        )
        variables.append(
            DesignVariable(
                name=name,
                lower=lower_value,
                upper=upper_value,
                initial=midpoint if initial_value is None else initial_value,
            )
        )

    return variables


def _prompt_design_variable_name(
    index: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    default = f"design_{index + 1}"
    while True:
        output_fn("Options:")
        output_fn("  - Enter a short variable name.")
        output_fn(f"  - Press Enter to use the default `{default}`.")
        raw = input_fn("Variable name: ").strip()
        if raw:
            return raw
        return default


def _apply_answer(spec: ExperimentSpec, field: str, answer: Any) -> None:
    if field == "backend":
        spec.backend = str(answer)
    elif field == "model_ref":
        spec.model_ref = str(answer)
    elif field == "guide_ref":
        spec.guide_ref = str(answer)
    elif field == "loss_ref":
        spec.loss_ref = str(answer)
    elif field == "optim_ref":
        spec.optim_ref = str(answer)
    elif field == "simulator_ref":
        spec.simulator_ref = str(answer)
    elif field == "prior_sampler_ref":
        if isinstance(answer, dict):
            spec.prior_sampler_ref = answer.get("prior_sampler_ref")
            spec.latent_sampler_ref = answer.get("latent_sampler_ref")
    elif field == "design_variables":
        spec.design_variables = list(answer)
    elif field == "backend_options.design_mode":
        spec.backend_options["design_mode"] = str(answer)
    elif field == "observation_labels":
        spec.observation_labels = list(answer)
    elif field == "target_latent_labels":
        spec.target_latent_labels = list(answer)
    elif field == "objective.estimator":
        spec.objective.estimator = str(answer)
    elif field == "compute_budget.num_outer_samples":
        spec.compute_budget.num_outer_samples = int(answer)
    elif field == "compute_budget.num_inner_samples":
        spec.compute_budget.num_inner_samples = int(answer)
    elif field == "compute_budget.guide_training_steps":
        spec.compute_budget.guide_training_steps = int(answer)
    elif field == "differentiable":
        spec.differentiable = bool(answer)
    else:
        raise ValueError(f"Unsupported interactive clarification field '{field}'.")

    if spec.problem_summary is None and spec.backend == "pyro" and spec.model_ref == DEMO_LINEAR_MODEL_REF:
        spec.problem_summary = DEMO_LINEAR_SUMMARY


def _recommended_guide_ref(spec: ExperimentSpec) -> str | None:
    estimator = spec.objective.estimator
    if estimator is None:
        return None
    return DEMO_LINEAR_GUIDE_REFS.get(estimator)


def _foster_estimator_options(
    experiment_key: str,
    estimator_names: tuple[str, ...],
) -> list[InteractiveOption]:
    options: list[InteractiveOption] = []
    for estimator in estimator_names:
        guide_ref = build_foster_experiment_spec(experiment_key, estimator).guide_ref
        options.append(
            InteractiveOption(
                label=estimator,
                value=estimator,
                description=_foster_estimator_description(estimator, guide_ref),
            )
        )
    return options


def _foster_estimator_description(estimator: str, guide_ref: str | None) -> str:
    if estimator == "posterior_eig":
        return f"Posterior-guide estimator using `{guide_ref}`."
    if estimator == "marginal_eig":
        return f"Marginal-guide estimator using `{guide_ref}`."
    if estimator == "vnmc_eig":
        return f"Nested Monte Carlo estimator using `{guide_ref}`."
    if estimator == "vi_eig":
        return (
            f"Variational estimator using `{guide_ref}` plus a Trace_ELBO loss, "
            "with the paper's longer training budget."
        )
    return f"Uses guide `{guide_ref}`."


def _apply_artifact_overrides(source: ExperimentSpec, destination: ExperimentSpec) -> None:
    default_artifacts = ArtifactSettings()
    if source.artifacts.output_dir != default_artifacts.output_dir:
        destination.artifacts.output_dir = source.artifacts.output_dir
    if source.artifacts.save_normalized_spec != default_artifacts.save_normalized_spec:
        destination.artifacts.save_normalized_spec = source.artifacts.save_normalized_spec
    if source.artifacts.save_transcript != default_artifacts.save_transcript:
        destination.artifacts.save_transcript = source.artifacts.save_transcript
    if source.artifacts.save_backend_summary != default_artifacts.save_backend_summary:
        destination.artifacts.save_backend_summary = source.artifacts.save_backend_summary
    if source.artifacts.save_result_payload != default_artifacts.save_result_payload:
        destination.artifacts.save_result_payload = source.artifacts.save_result_payload
    if source.artifacts.save_trajectory_plot != default_artifacts.save_trajectory_plot:
        destination.artifacts.save_trajectory_plot = source.artifacts.save_trajectory_plot
    if (
        source.artifacts.trajectory_plot_filename
        != default_artifacts.trajectory_plot_filename
    ):
        destination.artifacts.trajectory_plot_filename = source.artifacts.trajectory_plot_filename
    if (
        source.artifacts.trajectory_summary_filename
        != default_artifacts.trajectory_summary_filename
    ):
        destination.artifacts.trajectory_summary_filename = (
            source.artifacts.trajectory_summary_filename
        )


def _looks_like_custom_option(option: InteractiveOption) -> bool:
    label = option.label.lower()
    value = option.value.lower()
    return value.startswith("__custom__") or value.startswith("__manual__") or (
        "custom" in label or "manual" in label
    )
