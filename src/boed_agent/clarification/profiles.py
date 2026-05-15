"""Question templates and backend-specific clarification profiles."""

from __future__ import annotations

from boed_agent.models import ClarificationQuestion


QUESTION_TEMPLATES: dict[str, ClarificationQuestion] = {
    "backend": ClarificationQuestion(
        field="backend",
        prompt="Which BOED backend should be used: Pyro for an explicit probabilistic model, or LFIAX for a simulator-only workflow?",
        reason="Backend choice determines the required interfaces and tool path.",
        choices=["pyro", "lfiax"],
    ),
    "literature_source_mode": ClarificationQuestion(
        field="literature_source_mode",
        prompt="Should literature context come from online sources, a local paper directory, or both?",
        reason="Literature opt-in needs an explicit source mode before the advisory run can search for prior evidence.",
        choices=["online", "local", "both"],
    ),
    "literature_corpus_dir": ClarificationQuestion(
        field="literature_corpus_dir",
        prompt="Which local directory should be searched for relevant papers, notes, or PDFs?",
        reason="Local literature mode needs an explicit paper corpus directory.",
    ),
    "model_ref": ClarificationQuestion(
        field="model_ref",
        prompt="What probabilistic model callable should the Pyro backend import?",
        reason="Pyro execution requires a model reference.",
        backend="pyro",
    ),
    "guide_ref": ClarificationQuestion(
        field="guide_ref",
        prompt="What guide callable should be used for the selected variational estimator?",
        reason="Variational Pyro estimators require a guide family.",
        backend="pyro",
    ),
    "loss_ref": ClarificationQuestion(
        field="loss_ref",
        prompt="What loss callable should be used for the VI estimator?",
        reason="`vi_eig` requires an explicit VI loss callable.",
        backend="pyro",
    ),
    "optim_ref": ClarificationQuestion(
        field="optim_ref",
        prompt="What optimizer or optimizer factory should be used for variational training?",
        reason="Variational Pyro estimators require an optimizer.",
        backend="pyro",
    ),
    "simulator_ref": ClarificationQuestion(
        field="simulator_ref",
        prompt="What simulator callable should the LFIAX backend call?",
        reason="Simulator workflows need an executable simulator reference.",
        backend="lfiax",
    ),
    "prior_sampler_ref": ClarificationQuestion(
        field="prior_sampler_ref",
        prompt="How should latent parameters be sampled before simulation?",
        reason="The simulator backend needs either a prior sampler or latent sampler.",
        backend="lfiax",
    ),
    "design_variables": ClarificationQuestion(
        field="design_variables",
        prompt="What are the design variables and their bounds?",
        reason="Design optimization cannot proceed without a search space.",
    ),
    "backend_options.design_mode": ClarificationQuestion(
        field="backend_options.design_mode",
        prompt="Should the LFIAX backend optimize a point design or an annealed design distribution?",
        reason="This determines whether the design is a single vector or a tempered distribution over design slots.",
        backend="lfiax",
        choices=["point", "distribution"],
    ),
    "observation_labels": ClarificationQuestion(
        field="observation_labels",
        prompt="Which sample sites or outputs should be treated as future observations?",
        reason="The BOED objective needs to know what is observed.",
    ),
    "target_latent_labels": ClarificationQuestion(
        field="target_latent_labels",
        prompt="Which latent parameters are the design trying to learn about?",
        reason="The EIG target must be explicit.",
    ),
    "objective.estimator": ClarificationQuestion(
        field="objective.estimator",
        prompt="Which estimator family should be used for the BOED objective?",
        reason="Estimator choice changes backend requirements and runtime.",
        choices=["vi_eig", "posterior_eig", "marginal_eig", "vnmc_eig"],
    ),
    "compute_budget.num_outer_samples": ClarificationQuestion(
        field="compute_budget.num_outer_samples",
        prompt="What outer Monte Carlo sample budget should be used?",
        reason="BOED estimators need a concrete compute budget.",
    ),
    "compute_budget.num_inner_samples": ClarificationQuestion(
        field="compute_budget.num_inner_samples",
        prompt="What inner sample budget should be used?",
        reason="Nested estimators need an inner Monte Carlo budget.",
    ),
    "compute_budget.guide_training_steps": ClarificationQuestion(
        field="compute_budget.guide_training_steps",
        prompt="How many guide optimization steps are acceptable per EIG estimate?",
        reason="Variational BOED estimators require a training budget.",
    ),
    "differentiable": ClarificationQuestion(
        field="differentiable",
        prompt="Is the simulator differentiable with respect to design variables or latent parameters?",
        reason="This changes which optimization paths are possible later.",
        backend="lfiax",
        choices=["true", "false"],
    ),
}
