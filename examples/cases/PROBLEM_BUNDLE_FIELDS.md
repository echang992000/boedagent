# Problem Bundle Fields

This file describes the v1 `problem.json` shape for example problem bundles.

## Top-Level Fields

- `schema_version`
  Version string for the bundle format. Start with `"v1"`.
- `problem_summary`
  Short natural-language description of the BOED problem.
- `shared`
  Problem facts that apply across all candidate execution paths.
- `hints`
  Optional user beliefs, notes, or partial prior/literature guidance.
- `data`
  Optional observed-data summary or references to local artifacts.
- `candidates`
  List of candidate execution paths. Each candidate can later compile to one normal
  `ExperimentSpec`.
- `metadata`
  Free-form extras that do not belong in the stable schema.

## `shared`

Fields in `shared` should be backend-agnostic.

- `latent_spec`
  Structural information about the latent parameterization.
- `observation_spec`
  Structural information about the simulator outputs / observations.
- `design_variables`
  Design-space variables and bounds, using the same core shape as `ExperimentSpec`.

## `shared.latent_spec`

- `shape`
  Array-like shape for the latent parameter object, for example `[3]`.
- `labels`
  Optional parameter names if known.
- `notes`
  Optional explanation when the latent parameterization is only partially known.

## `shared.observation_spec`

- `shape`
  Array-like shape for one simulated observation, for example `[11]`.
- `labels`
  Optional labels for outputs or observation channels.
- `notes`
  Optional description of how to interpret the observation shape.

## `hints`

Use `hints` for partial user input that should not be treated as a hard runtime requirement.

- `candidate_preference`
  Optional preferred candidate name if the user already has one in mind.
- `prior_notes`
  Informal prior beliefs, parameter ranges, or model assumptions.
- `literature_corpus_dir`
  Optional local paper directory for literature advisory runs.
- `notes`
  Any other free-form problem notes.

## `data`

Use `data` to point at local files or summarize available observations.

- `observed_data_ref`
  Path or identifier for the main observed-data artifact.
- `observed_data_summary`
  Short human-readable summary of what is in the data.
- `sources`
  Optional list of additional paths, files, or notes tied to the problem.

## `candidates`

Each item in `candidates` describes one executable interpretation of the problem.
Only candidate-specific fields should go here.

Shared fields:

- `name`
  Stable candidate identifier.
- `kind`
  Either `"explicit"` or `"simulator"`.
- `description`
  Short explanation of what this candidate represents.

Explicit candidate fields:

- `model_ref`
- `guide_ref`
- `loss_ref`
- `optim_ref`

Simulator candidate fields:

- `simulator_ref`
- `prior_sampler_ref` or `latent_sampler_ref`
- `differentiable`

Optional candidate fields for either kind:

- `observation_labels`
- `target_latent_labels`
- `backend`
- `backend_options`
- `metadata`

## Sparsity Rules

- Problem bundles may be incomplete.
- Unknown fields should usually be omitted rather than filled with `null`.
- A sparse bundle is valid for planning and clarification.
- A compiled execution spec must still satisfy normal backend validation before it can run.
