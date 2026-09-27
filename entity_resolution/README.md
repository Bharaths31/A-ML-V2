# Business Entity Resolution — Amazon ML Challenge 2026

A precision-first, metric-aligned pipeline for multi-source business entity
resolution. Given records from three independent sources (S1 reference, S2/S3
candidates) it produces:

- `output/matching_results.tsv` — final matches (the leaderboard file)
- `output/candidate_pairs.tsv` — the exact blocking candidate set fed to the model

See `Documentation_template.md` at the repository root for the full methodology
write-up, and `SOLUTION_PLAN.md` for the design blueprint.

## Pipeline overview

```
sources -> [1] canonicalize -> [2] blocking cascade -> [3] pairwise features
        -> [4] calibrated LightGBM (+ singleton model) -> [5] per-entity
           expected-F0.5 decision engine -> [6] consistency + validated output
```

The decision engine (§5 of the plan) chooses, per Source 1 entity, between
abstaining and predicting the top-k prefix that maximises expected F0.5,
which is exactly the challenge metric (including the singleton 1.0/0.0
asymmetry). Country is used only as a string-similarity feature, never a filter,
so unseen countries (France) are handled by construction.

## Setup

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

The pipeline expects the challenge layout:

```
dataset/
├── train/{train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv}
└── test/{test_source1.tsv, test_source2.tsv, test_source3.tsv}
```

No dataset yet? Generate a challenge-shaped synthetic one (mimics the documented
noise and adds France only to the test split):

```bash
python -m src.cli.run --stage synthdata --data-root ../../dataset \
    --n-train 400 --n-test 200 --with-test-gt
```

## Run end-to-end

```bash
python -m src.cli.run --data-root ../../dataset
```

Outputs land in `output/`; reports and cached artifacts in `artifacts/`. The
pipeline prints validation macro-F0.5, blocking recall, mean candidates per S1
entity, and the submission validator result.

## CLI stages

| Command | Purpose |
|---|---|
| `--stage synthdata` | generate a synthetic dataset |
| `--stage all` (default) | run stages 1–6 end-to-end |
| `--stage validate` | run the submission validator on `output/` |
| `--stage score --gt <file>` | score `output/matching_results.tsv` against ground truth |

Useful flags: `--config <yaml>`, `--set decision.lambda=0.8`, `--seed`, `--out-dir`,
`--cache-dir`.

## Tests

```bash
python -m pytest -q
```

The test suite covers the exact F0.5 scorer (including the statement example),
prefix optimality of the decision engine, canonicalization, blocking recall,
output invariants, and a full end-to-end run.

## Reproducibility

- Every tunable lives in `src/config/default.yaml` (mirrored in
  `src/config/config.py`); nothing is hard-coded in the stages.
- Fixed seeds; LightGBM `deterministic=true`; deterministic MinHash/LSH.
- Pinned dependencies in `requirements.txt`.
- Stages cache versioned artifacts under `artifacts/`.

## Normalization assets & fair play

All normalization assets (`src/normalize/assets/`) are hand-curated from
common-language knowledge and shipped in-package. **No external database, API,
geocoding service, or internet lookup is used anywhere.** See
`src/normalize/assets/ASSETS.md` for provenance.

## License

Code is released for the challenge submission. Runtime dependencies are
permissive (BSD-3/MIT/Apache-2.0); see the license table in the methodology
document. Only MIT/Apache-2.0 licensed model weights (≤ 8B parameters) may be
added for the optional embedding feature.
