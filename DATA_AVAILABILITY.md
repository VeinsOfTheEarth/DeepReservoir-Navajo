# Data Availability

The repository bundles the processed daily inputs, calibrated physical relationships, corrected selected-policy artifacts, and compact figure data needed to evaluate the model and rebuild the paper outputs. Earlier design-comparison inputs are stored beside their figure builders and do not require private experiment directories.

## Physical inputs

- Reservoir inflow, storage, release, elevation, and reported historical evaporation.
- San Juan and Animas streamflow records used for downstream flows and scoring.
- Historical NIIP delivery used as the daily demand proxy and annual demand targets.
- Elevation-storage relationships and hydropower parameters with calibration sources.
- Spring-peak parameters and snow-water-equivalent records used by opportunity-index diagnostics. Snow water equivalent is not one of the selected policy's 20 observation features.

Farmington correction scripts and their source records are retained under `data/patch_sanjuan_at_farmington/`. Calibration sources are retained for provenance and are not re-downloaded during evaluation.

## Training hydrology

The selected configuration applies annual holdout-mean scaling to training-period reservoir inflow, reported evaporation, and Animas Farmington flow. Evaluation forcing is unchanged. The selected policy uses the common 2019 storage datum and storage normalization fitted on the training window.

## Bundled results

`artifacts/selected_policy/` contains the checkpoint, resolved configuration, run manifest, evaluation rollout and metrics, selected-family seed metrics, training trace, and the two perturbation sweeps. `results/corrected-phase95-recovery/` contains aggregate records for all 128 corrected search policies. Figure-specific extracts live under `paper/figures/<figure-name>/data/`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands and [ARTIFACT_MANIFEST.md](ARTIFACT_MANIFEST.md) for checkpoint identity.
