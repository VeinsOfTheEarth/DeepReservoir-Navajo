> **Snapshot scope:** Frozen artifacts and figures preserve the archived pre-correction baseline. Evaluating that checkpoint now applies the corrected current code as a compatibility check and does not reproduce the frozen trajectory exactly. A final corrected checkpoint has not yet been selected or bundled. See [CORRECTED_SEARCH.md](CORRECTED_SEARCH.md).

# Reproducing the Prepublication Snapshot

Run commands from the repository root. Python 3.10, Stable-Baselines3 2.7.1, PyTorch 2.5.1, and Gymnasium 1.2.3 are specified in `environment.yml`. General scientific dependencies are not fully version-locked.

The environment uses the public `pytorch` and `conda-forge` channels only.
`nodefaults` prevents additional user-configured channels from being appended.

```powershell
conda env create -f environment.yml
conda activate deepreservoir
python -m pip install -e ".[test]"
python -m pip check
python -m pytest -q
```

## 1. Rebuild Figures and Tables

```powershell
python paper/build-figures.py
python paper/tables/selected_policy_comparison_with_historic/build.py
python paper/tables/selected_policy_metrics/build.py
```

The figure catalog is `paper/figure_catalog.json`. Eleven figures are rebuilt programmatically. The study-system map, objective-location schematic, and SPR flowchart use authoritative, manually edited PowerPoint/PDF files; the batch builder skips them.

Each figure's README documents its sources. Historical development comparisons use bundled CSV summaries and rollouts, not private experiment directories. These permit figure reproduction, not retraining every historical architecture or reward variant.

## 2. Reevaluate the Archived Baseline Policy

```powershell
python -m deepreservoir.drl.cli eval --outdir runs/paper-evaluation
```

This loads `artifacts/legacy_phase95_policy/navajo_reservoir_selected_policy.zip`, reads the frozen configuration, and evaluates 2014-01-01 through 2024-08-17. It writes the daily rollout and metrics. It does not start additional experiments or generate HTML reports.

The bundled metrics and rollout preserve the original pre-correction evaluation used by the archived figures. Re-evaluation exercises the current corrected code, so some metrics deliberately differ from those frozen artifacts. Tests verify the artifact checksums and document that expected divergence. PDF metadata may differ between builds even when plotted data and appearance match.

## 3. Retrain the Archived Baseline Configuration

```powershell
python -m deepreservoir.drl.cli train --outdir runs/paper-training
```

The command reads the configuration directly: seed 4, 1967-06-07 through 2013-12-31, 3,600-step episodes/PPO rollouts, batch size 60, gamma 0.999, ten PPO epochs, and the archived four-head policy/rewards. The requested budget of 3,159,000 is rounded down to 3,157,200 steps (877 complete rollouts), matching the original command's convention.

To recover richer logs for the training-reward-signals figure:

```powershell
python -m deepreservoir.drl.cli train --diagnostics --outdir runs/paper-training-diagnostics
```

For a short pipeline check:

```powershell
python -m deepreservoir.drl.cli train --diagnostics --timesteps 3600 --skip-eval --outdir runs/training-check
```

Training evaluates the final checkpoint unless `--skip-eval` is specified. `--seed`, `--device`, and `--torch-threads` are available for reproducing seeds or adapting execution to the machine. The default is CPU with one PyTorch thread. Resolved settings and library versions are saved with the run.

Retraining is not a promise of bit-for-bit recovery of the archived checkpoint. Hardware, numerical libraries, and stochastic optimization can change the result. The frozen artifact remains the reference for the archived figure inputs; it is not the final corrected paper policy. The bundled training-reward-signals data came from a separate diagnostic retraining with the same configuration.

## 4. Regenerate Included Perturbation Results

Optional if only rebuilding figures from bundled results:

```powershell
python scripts/run_selected_policy_inflow_scaling_sweep.py --help
python scripts/run_selected_policy_initial_storage_sweep.py --help
python paper/figures/policy-response/run_diagnostics.py --help
```

Use each command's output-directory option to write new results separately. The first two reproduce the archived baseline's inflow and initial-storage sweeps. Policy response compares adaptive decisions with replay of the unperturbed action schedule under the same perturbed inflows. Figure READMEs identify the exact bundled summaries and daily series consumed by their builders.

## Preserved Scientific Choices

Packaging does not alter the archived physical model, rewards, or performance definitions.

- Evaporation uses historical reported data, not a learned evaporation model.
- The archived training configuration scales San Juan inflow, historical evaporation, and Animas flow so their mean complete-water-year totals match the 2014-2024 reference period. The evaluation data are unchanged. This is a retrospective design choice using holdout climatology, not a wholly independent forecasting experiment.
- Storage standardization retains the original normalization over the joined record. Changing that would invalidate the checkpoint's inputs.
- The archived policy observes current conditions and SPR calendar/counter advice, not future-flow summaries. OI fields retained in evaluation diagnostics are not extra policy observations.
- Farmington streamflow corrections and hydropower/storage-elevation calibration sources are retained with the physical input data.

No private `runs/` folder is required for the bundled evaluation or figure builds. Paths named in old provenance records identify their origin; they are not dependencies.

## Release Checks

The public staging tree was checked on 2026-09-23 with Python 3.10 in the project Conda environment. `pip check` reported no broken requirements. The scientific suite reported 160 passed tests, 10 passed subtests, and one expected failure. That expected failure records the known difference between the frozen pre-correction rollout and the corrected current implementation.

All eleven cataloged programmatic main figures rebuilt successfully. Their PNG exports were byte-identical to the staged references; regenerated PDF hashes changed with PDF metadata. The three manually edited main figures were left untouched. All six appendix figures rebuilt from bundled runtime data, and all twelve appendix PDF/PNG exports were SHA-256-identical to the manuscript copies.

Both paper tables rebuilt successfully. The corrected Phase 95 specification validated as eight experiments by sixteen seeds, and the archived checkpoint completed a 3,882-day evaluation with the current code. These checks verify the public snapshot's packaging and executable workflows; they do not designate a final corrected paper policy.
