You are working in the BOED agent repository at:

- `/Users/vincentzaballa/Projects/boed_agent`

Read and follow these local files:

- `examples/cases/bmp4_gradient/prompts/hill_prior_prompt.md`
- `examples/cases/bmp4_gradient/prompts/PRIOR_OUTPUT_SCHEMA.md`
- `examples/cases/bmp4_gradient/problem.json`
- `data/bmp4_data/BMP4_GRADIENT_README.md`
- `examples/agent/local_corpus/bmp4_gradient`

Task:

1. Synthesize the BMP4 Hill-family literature prior strictly from the local BMP4
   corpus and files above.
2. Follow the lower-level Hill prompt exactly.
3. Write the final JSON artifact to:
   - `artifacts/bmp4_gradient/priors/hill_literature_prior.json`
4. Overwrite that file if it already exists.
5. Do not modify any source code.
6. Do not browse the web.
7. Do not use outside biological knowledge beyond the provided local files.

Output requirements:

- The artifact file must contain valid JSON only.
- After writing the file, print a short final message containing:
  - the absolute path written
  - the family name
  - the parameter order

Nothing else is required.
