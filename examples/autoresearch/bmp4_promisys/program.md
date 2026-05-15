# BMP4 Promisys Autoresearch

This folder is the editable surface for autoresearch-style BMP4 promisys tuning.

## Rules

- Edit only `examples/autoresearch/bmp4_promisys/trial.json`.
- Do not edit `baseline.json`; it represents the current implementation defaults.
- Do not edit the BMP4 model modules during a config-search run.
- Trial artifacts and `results.tsv` are written under `artifacts/bmp4_gradient/autoresearch/<tag>/`.

## Metric

The runner ranks trials by `best_eig` from `eig_optimization_summary.json`.
A trial is only valid when SNPE and likelihood losses are finite, required artifacts exist,
MCMC sample counts are nonzero, and the run stays under the timeout.

## Commands

Initialize a tag:

```bash
python scripts/bmp4_autoresearch.py init --tag bmp4-smoke
```

Run the empty baseline config:

```bash
python scripts/bmp4_autoresearch.py run --tag bmp4-smoke --baseline --description baseline
```

Run the editable trial config:

```bash
python scripts/bmp4_autoresearch.py run --tag bmp4-smoke --description "trial description"
```

Run autonomous config search until it reaches a stop criterion:

```bash
python scripts/bmp4_autoresearch.py loop \
  --tag bmp4-smoke \
  --family promisys_onestep \
  --max-trials 10 \
  --patience 4 \
  --max-runtime-seconds 7200
```

The loop mutates the current best config, records each trial in `results.tsv`,
updates `best_config.json` on kept improvements, and stops when it hits
`max-trials`, `patience`, or `max-runtime-seconds`.

Every `run` and `loop` update also writes `progress.png` beside `results.tsv`.

Confirm a promising config with the larger production budget:

```bash
python scripts/bmp4_autoresearch.py run --tag bmp4-confirm --confirmation --description "confirm best trial"
```

If the literature prior is not at the default local artifact path, pass:

```bash
--literature-prior-json /path/to/literature_prior.json
```
