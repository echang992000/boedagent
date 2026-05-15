from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from boed_agent.simulator_protocol import ParameterInfo, SimulatorMetadata

from . import priors
from . import promisys_onestep as promisys_onestep_model
from . import promisys_twostep as promisys_twostep_model
from .pyro import hill as hill_model
from .pyro import multireceptor as multireceptor_model
from .pyro import multireceptor_hierarchical as multireceptor_hierarchical_model


@dataclass(frozen=True)
class ModelFamilySpec:
    name: str
    candidate_name: str
    literature_parameters: tuple[str, ...]
    build_metadata: Callable[[Sequence[str]], SimulatorMetadata]
    build_problem_description: Callable[[str, Sequence[str]], str]
    translate_prior: Callable[..., priors.TranslatedPrior]
    make_fit_model: Callable[..., Any]
    predictive_draws: Callable[..., Any]
    scalar_log_likelihood: Callable[..., Any]
    execution_mode: str = "per_cell_line"


def get_model_family_registry() -> dict[str, ModelFamilySpec]:
    return {
        "hill": ModelFamilySpec(
            name="hill",
            candidate_name="hill_differentiable",
            literature_parameters=tuple(hill_model.LITERATURE_PARAMETER_NAMES),
            build_metadata=_hill_metadata,
            build_problem_description=lambda summary, receptor_names: hill_model.problem_description(summary),
            translate_prior=lambda augmented, cell_line_data=None, receptor_names=None: priors.build_hill_prior(augmented),
            make_fit_model=lambda translated, cell_line_data, receptor_names: hill_model.make_fit_model(translated),
            predictive_draws=lambda concentration, posterior_samples, cell_line_data, receptor_names: hill_model.predictive_draws(
                concentration,
                posterior_samples,
            ),
            scalar_log_likelihood=lambda y_value, concentration, posterior_samples, cell_line_data, receptor_names: hill_model.scalar_log_likelihood(
                y_value,
                concentration,
                posterior_samples,
            ),
        ),
        "multireceptor": ModelFamilySpec(
            name="multireceptor",
            candidate_name="multireceptor_differentiable",
            literature_parameters=tuple(multireceptor_model.LITERATURE_PARAMETER_NAMES),
            build_metadata=_multireceptor_metadata,
            build_problem_description=lambda summary, receptor_names: multireceptor_model.problem_description(
                summary,
                receptor_names,
            ),
            translate_prior=lambda augmented, cell_line_data, receptor_names: priors.build_multireceptor_prior(
                augmented,
                receptor_names=receptor_names,
                receptor_qpcr=cell_line_data.Rs,
            ),
            make_fit_model=lambda translated, cell_line_data, receptor_names: multireceptor_model.make_fit_model(
                translated,
                receptor_names=receptor_names,
            ),
            predictive_draws=lambda concentration, posterior_samples, cell_line_data, receptor_names: multireceptor_model.predictive_draws(
                concentration,
                posterior_samples,
                receptor_names=receptor_names,
            ),
            scalar_log_likelihood=lambda y_value, concentration, posterior_samples, cell_line_data, receptor_names: multireceptor_model.scalar_log_likelihood(
                y_value,
                concentration,
                posterior_samples,
                receptor_names=receptor_names,
            ),
        ),
        "multireceptor_hierarchical": ModelFamilySpec(
            name="multireceptor_hierarchical",
            candidate_name="multireceptor_hierarchical_pyro",
            literature_parameters=tuple(multireceptor_hierarchical_model.LITERATURE_PARAMETER_NAMES),
            build_metadata=_multireceptor_hierarchical_metadata,
            build_problem_description=lambda summary, receptor_names: multireceptor_hierarchical_model.problem_description(
                summary,
                receptor_names,
            ),
            translate_prior=lambda augmented, cell_line_data, receptor_names: priors.build_multireceptor_hierarchical_prior(
                augmented,
                receptor_names=receptor_names,
            ),
            make_fit_model=lambda translated, cell_line_data, receptor_names: multireceptor_hierarchical_model.make_fit_model(
                translated,
                q_obs=cell_line_data.q_obs,
                kd_prior_shift=cell_line_data.kd_prior_shift,
            ),
            predictive_draws=lambda concentration, posterior_samples, cell_line_data, receptor_names: multireceptor_hierarchical_model.predictive_draws(
                concentration,
                posterior_samples,
            ),
            scalar_log_likelihood=lambda y_value, concentration, posterior_samples, cell_line_data, receptor_names, **kwargs: multireceptor_hierarchical_model.scalar_log_likelihood(
                y_value,
                concentration,
                posterior_samples,
                **kwargs,
            ),
            execution_mode="joint_cell_lines",
        ),
        "promisys_onestep": ModelFamilySpec(
            name="promisys_onestep",
            candidate_name="promisys_onestep_lfiax",
            literature_parameters=(),
            build_metadata=_promisys_onestep_metadata,
            build_problem_description=lambda summary, receptor_names: promisys_onestep_model.problem_description(
                summary,
                receptor_names,
            ),
            translate_prior=lambda augmented, cell_line_data=None, receptor_names=None: priors.TranslatedPrior(
                family="promisys_onestep",
                sites={},
            ),
            make_fit_model=lambda translated, cell_line_data, receptor_names: None,
            predictive_draws=lambda concentration, posterior_samples, cell_line_data, receptor_names: None,
            scalar_log_likelihood=lambda y_value, concentration, posterior_samples, cell_line_data, receptor_names: None,
            execution_mode="promisys_onestep",
        ),
        "promisys_twostep": ModelFamilySpec(
            name="promisys_twostep",
            candidate_name="promisys_twostep_lfiax",
            literature_parameters=(),
            build_metadata=_promisys_twostep_metadata,
            build_problem_description=lambda summary, receptor_names: promisys_twostep_model.problem_description(
                summary,
                receptor_names,
            ),
            translate_prior=lambda augmented, cell_line_data=None, receptor_names=None: priors.TranslatedPrior(
                family="promisys_twostep",
                sites={},
            ),
            make_fit_model=lambda translated, cell_line_data, receptor_names: None,
            predictive_draws=lambda concentration, posterior_samples, cell_line_data, receptor_names: None,
            scalar_log_likelihood=lambda y_value, concentration, posterior_samples, cell_line_data, receptor_names: None,
            execution_mode="promisys_twostep",
        ),
    }


def _hill_metadata(receptor_names: Sequence[str]) -> SimulatorMetadata:
    _ = receptor_names
    return SimulatorMetadata(
        parameters=[
            ParameterInfo(name=name)
            for name in hill_model.LITERATURE_PARAMETER_NAMES
        ],
        observation_labels=["pSmad_response"],
        domain_tags=["bmp4_gradient", "hill", "dose_response"],
    )


def _multireceptor_metadata(receptor_names: Sequence[str]) -> SimulatorMetadata:
    extras = {"receptor_names": list(receptor_names)}
    return SimulatorMetadata(
        parameters=[
            ParameterInfo(name=name)
            for name in multireceptor_model.LITERATURE_PARAMETER_NAMES
        ],
        observation_labels=["pSmad_response"],
        domain_tags=["bmp4_gradient", "multireceptor", "dose_response"],
        extras=extras,
    )


def _multireceptor_hierarchical_metadata(receptor_names: Sequence[str]) -> SimulatorMetadata:
    extras = {
        "receptor_names": list(receptor_names),
        "execution_mode": "joint_cell_lines",
    }
    return SimulatorMetadata(
        parameters=[
            ParameterInfo(name=name)
            for name in multireceptor_hierarchical_model.LITERATURE_PARAMETER_NAMES
        ],
        observation_labels=["pSmad_response_by_cell_line", "qpcr_receptor_measurements"],
        domain_tags=["bmp4_gradient", "multireceptor", "hierarchical", "dose_response"],
        extras=extras,
    )


def _promisys_onestep_metadata(receptor_names: Sequence[str]) -> SimulatorMetadata:
    extras = {
        "receptor_names": list(receptor_names),
        "type_i_receptors": list(promisys_onestep_model.TYPE_I_RECEPTORS),
        "type_ii_receptors": list(promisys_onestep_model.TYPE_II_RECEPTORS),
        "complex_names": list(promisys_onestep_model.COMPLEX_NAMES),
        "execution_mode": "promisys_onestep",
    }
    return SimulatorMetadata(
        parameters=[
            ParameterInfo(name=item["name"], description=item["description"])
            for item in promisys_onestep_model.metadata_parameters()
        ],
        observation_labels=["normalized_pSmad_response"],
        domain_tags=["bmp4_gradient", "promisys", "onestep", "simulator_based"],
        extras=extras,
    )


def _promisys_twostep_metadata(receptor_names: Sequence[str]) -> SimulatorMetadata:
    extras = {
        "receptor_names": list(receptor_names),
        "type_i_receptors": list(promisys_twostep_model.TYPE_I_RECEPTORS),
        "type_ii_receptors": list(promisys_twostep_model.TYPE_II_RECEPTORS),
        "dimer_complex_names": list(promisys_twostep_model.DIMER_COMPLEX_NAMES),
        "complex_names": list(promisys_twostep_model.COMPLEX_NAMES),
        "execution_mode": "promisys_twostep",
    }
    return SimulatorMetadata(
        parameters=[
            ParameterInfo(name=item["name"], description=item["description"])
            for item in promisys_twostep_model.metadata_parameters()
        ],
        observation_labels=["normalized_pSmad_response"],
        domain_tags=["bmp4_gradient", "promisys", "twostep", "simulator_based"],
        extras=extras,
    )


__all__ = ["ModelFamilySpec", "get_model_family_registry"]
