# V2 — Business Entity Resolution (Amazon ML Challenge 2026)

Precision-first, profiler-driven entity resolution across 3 sources.
Target metric: macro **F₀.₅** (precision weighted 2×).

## Project layout

```
entity_resolution/
├── config.yaml               # all tunables (paths, blocking, model, selection)
├── setup.py                  # create venv, install deps, GPU detect, stage data
├── master_profiler.py        # CLI: integrity | profile | select | report | all
├── master_trainer.py         # CLI: prepare | train | evaluate | predict | submit | all
├── run_train.py              # convenience: training entry point
├── run_test.py               # convenience: inference entry point
├── submit.py                 # validate + package the submission zip
├── profiler.py               # flat modules (wrap er_core)
├── blocker.py
├── features.py
├── trainer.py
├── tester.py
├── normalizer.py
├── metrics.py
├── io_utils.py
├── cuda_utils.py             # NVIDIA/CUDA detection + LightGBM GPU probe
├── er_core/                  # ported V1 engine (normalize → block → features → model → decide)
├── data/                     # unified train/ + test/ (from student_resource)
├── profile/                  # profile.json, selection.json, report.md
├── cache/                    # intermediate artifacts
├── models/ · output/         # model bundle · submission files
└── requirements.txt
```

## Setup

```bash
python setup.py            # auto CPU/GPU
python setup.py --gpu      # force GPU attempt
python setup.py --cpu      # force CPU
```

`setup.py` creates `.venv`, installs `requirements.txt`, detects NVIDIA/CUDA,
records the result in `cuda_status.json`, and copies the dataset into `data/`.

Activate the environment:
- Windows CMD: `.venv\Scripts\activate.bat`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

## Reproduce end-to-end

```bash
python master_profiler.py --stage all --data-root data --out profile
python master_trainer.py --stage all --config config.yaml --data-root data
```

- `master_profiler.py all` writes `profile/profile.json`, `profile/selection.json`,
  `profile/report.md`.
- `master_trainer.py all` runs canonicalization → blocking → featurization →
  training → calibration → validation → test inference → validator.

Scaling ladder (override the minimal fit set size):

```bash
python master_trainer.py --stage train --ladder 300k
```

## Validate + package the submission

```bash
python submit.py --team-name <team_name>            # validate, then zip
python submit.py --team-name <team_name> --validate-only
python submit.py --team-name <team_name> --package-only
```

`submit.py`:

1. Runs the official stdlib validator `data/utils/validate_submission.py`
   against `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and
   `data/test/`. It prints `PASS` (exit 0) or a numbered issue list (exit 1).
2. Builds `<team_name>_submission.zip` with the required layout:

```
<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md
```

Packaging is blocked unless validation passes (override with `--skip-validate`).

## Fair-play constraints

- No external data, APIs, or geocoding lookups.
- Normalization assets are hand-curated, in-package, and documented in
  `er_core/normalize/assets/ASSETS.md`.
- Model: LightGBM (MIT). CPU by default; GPU when available.

## Notes on scale

The ported engine is functional end-to-end. For the full 2.2M-entity dataset on
16 GB, run the chunked/staged path in `master_trainer.py` or use a machine with
≥32 GB for the first full pass. The profiler is streaming and cheap on any machine.
