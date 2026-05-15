# Problem Bundles

Problem bundles live under `examples/cases/<problem_name>/`.

Each bundle should contain:

- `problem.json` — the canonical problem-bundle manifest
- optional `README.md` — problem-specific notes, assumptions, and source links
- optional `simulators/` — user-provided simulator callables for candidate execution paths
- optional `pyro/` — Pyro model / guide / optimizer callables for explicit candidates
- optional `local_corpus/` — papers or notes used by the literature advisory path

The bundle is intentionally broader than an executable BOED spec. It stores shared
problem context plus one or more candidate execution paths. A later compile step
should select one candidate and emit a normal `ExperimentSpec` for runtime use.

See `PROBLEM_BUNDLE_FIELDS.md` for the field-level reference.
