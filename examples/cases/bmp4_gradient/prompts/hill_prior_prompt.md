Read these local files and directories:

- `data/bmp4_data/BMP4_GRADIENT_README.md`
- `examples/cases/bmp4_gradient/problem.json`
- `examples/cases/bmp4_gradient/prompts/PRIOR_OUTPUT_SCHEMA.md`
- `examples/agent/local_corpus/bmp4_gradient`

Task:

Synthesize literature-informed priors for the BMP4 gradient Hill model used in
the BMP4 example bundle.

Model-family context:

- Family name: `hill`
- Interpretation: four-parameter Hill dose-response curve with additive
  observation noise
- Parameters to synthesize:
  - `bottom`
  - `top`
  - `ec50`
  - `hill_n`
  - `sigma`

Requirements:

- Use only the local BMP4 corpus and the BMP4 readme/problem bundle listed
  above.
- Do not use outside knowledge, web search, or unstated biological assumptions.
- Keep parameter names exactly as listed.
- Prefer one of these distribution families:
  - `Normal`
  - `LogNormal`
  - `Gamma`
  - `Uniform`
  - `Beta`
- If evidence is weak, conflicting, or absent, choose a weak prior and set
  `fallback` to `true`.
- Keep `cited_papers` grounded in the local files only.
- Return JSON only, following the shared schema in
  `examples/cases/bmp4_gradient/prompts/PRIOR_OUTPUT_SCHEMA.md`.

Additional guidance:

- `bottom` should reflect baseline response behavior.
- `top` should reflect upper-plateau response behavior.
- `ec50` should reflect the half-maximal dose scale in raw BMP4 concentration
  units.
- `hill_n` should reflect response steepness.
- `sigma` should reflect observation-noise scale.

Output contract:

- `family` must be `"hill"`
- `corpus_scope` must be `"local_bmp4_corpus_only"`
- `parameter_order` must be `["bottom", "top", "ec50", "hill_n", "sigma"]`
- `priors` must contain exactly those five parameters and nothing else
