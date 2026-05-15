# BMP4 Literature Prompts

This directory holds reproducible prompt templates for the BMP4 gradient
literature-prior stage.

Intended use:

1. Pick the prompt file for the model family you want to test.
2. Paste it into a local LLM tool such as Codex CLI.
3. Keep the referenced files/corpus fixed.
4. Save the returned JSON artifact alongside the run so later reruns can replay it.

Files:

- `hill_prior_prompt.md` for the Hill family
- `multireceptor_prior_prompt.md` for the multireceptor family
- `PRIOR_OUTPUT_SCHEMA.md` for the shared JSON contract
- `codex_hill_session_prompt.md` for a higher-level reproducible Codex CLI run
- `codex_multireceptor_session_prompt.md` for a higher-level reproducible Codex CLI run

These prompts are intentionally example-local and should be treated as
versioned scientific inputs, not ad hoc chat instructions.
