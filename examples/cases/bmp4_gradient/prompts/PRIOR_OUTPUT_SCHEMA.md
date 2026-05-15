# BMP4 Prior Output Schema

Return JSON only. No prose before or after the JSON blob.

Shared schema:

```json
{
  "family": "hill",
  "corpus_scope": "local_bmp4_corpus_only",
  "parameter_order": ["bottom", "top", "ec50", "hill_n", "sigma"],
  "priors": {
    "bottom": {
      "distribution": "Normal",
      "params": {
        "loc": 0.0,
        "scale": 1.0
      },
      "reasoning": "One or two sentences explaining the local-corpus evidence or fallback choice.",
      "cited_papers": ["filename_or_identifier"],
      "fallback": false
    }
  },
  "notes": [
    "Optional short notes about weak evidence, conflicting studies, or coverage gaps."
  ]
}
```

Rules:

- Keep `family` equal to the requested family name.
- Keep `corpus_scope` equal to `"local_bmp4_corpus_only"`.
- Keep `parameter_order` equal to the exact requested parameter list.
- `priors` must contain one object per requested parameter.
- Allowed distribution families:
  - `Normal`
  - `LogNormal`
  - `Gamma`
  - `Uniform`
  - `Beta`
- Use `fallback: true` when evidence is weak or missing.
- Do not invent citations. Use local filenames or clear local identifiers.
- Keep parameter names exactly as requested. Do not add extra parameters.
