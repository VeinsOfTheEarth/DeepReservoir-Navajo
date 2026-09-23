> **Snapshot scope:** Processed model inputs are bundled. The trained checkpoint and model-dependent results currently included are an archived pre-correction baseline; corrected aggregate search results are separate under `results/corrected-phase95-recovery/`.

# Data Availability

The repository bundles the processed daily inputs, calibrated physical
relationships, and archived pre-correction results used by the current model-dependent figures. Figure-specific
results from earlier experiments are stored beside their figure builders; those
figures do not require the original experiment directories.

## Physical Inputs

- Reservoir inflow, storage, release, elevation, and **reported historical
  evaporation**. Evaporation is not predicted by a learned model.
- San Juan and Animas streamflow records used for downstream flows and scoring.
- Historical NIIP delivery and annual demand targets.
- Elevation-storage relationships and hydropower parameters, with their
  calibration sources.
- Spring peak release parameters and snow-water-equivalent records used by
  retained opportunity-index diagnostics. SWE is **not** an input to the archived
  actor; its exact 20-input observation vector is defined in
  `src/deepreservoir/drl/observations.py`.

The Farmington correction scripts and their source/correction records in
`data/patch_sanjuan_at_farmington/` are retained so the processed historic
benchmark can be traced to its inputs. They are not run during evaluation.
Calibration sources are likewise retained for provenance, not loaded anew for
each rollout.

## Training Hydrology

The archived configuration applies annual holdout-mean scaling to the training
reservoir inflow, reported evaporation, and Animas Farmington flow. Evaluation
forcing is not rescaled for scoring. This calibration uses information from the
evaluation period; it is reproduced as an original study choice, not presented
as an independent forecast validation. Observation normalization also retains
the original full-record reference values.

## Bundled Results

`artifacts/legacy_phase95_policy/` contains the frozen checkpoint, evaluation rollout,
metrics, and selected-family seed results. Additional inputs for the included
figures live in their `paper/figures/<figure-name>/data/` directories.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for evaluation, training, and figure
commands, and [ARTIFACT_MANIFEST.md](ARTIFACT_MANIFEST.md) for checkpoint identity.
