# Reproducing the Selected-Policy Results

Run commands from the repository root with Python 3.10. The environment file specifies Stable-Baselines3 2.7.1, PyTorch 2.5.1, and Gymnasium 1.2.3.

```powershell
conda env create -f environment.yml
conda activate deepreservoir
python -m pip install -e ".[test]"
python -m pip check
python -m pytest -q
```

## Evaluate the selected policy

```powershell
python -m deepreservoir.drl.cli eval --outdir runs/paper-evaluation
```

This evaluates the task-105 checkpoint over January 1, 2014 through August 17, 2024 with the corrected reward, storage, information-timing, and metric definitions.

## Rebuild figures and tables

```powershell
python paper/build-figures.py
python paper/tables/selected_policy_comparison_with_historic/build.py
python paper/tables/selected_policy_metrics/build.py
```

The batch builder regenerates eleven programmatic main figures and leaves the three manually edited PowerPoint figures untouched. Appendix build commands are listed in the root README.

## Rebuild perturbation diagnostics

```powershell
python scripts/run_selected_policy_initial_storage_sweep.py
python scripts/run_selected_policy_inflow_scaling_sweep.py
python paper/figures/policy-response/run_diagnostics.py
python paper/build-figures.py
```

The first two commands write to `artifacts/selected_policy/`. Copy their compact trajectory and summary outputs into `paper/figures/stress-testing/data/` before rebuilding that figure. The policy-response command writes directly to its figure-data directory.

## Retrain the configuration

```powershell
python -m deepreservoir.drl.cli train --outdir runs/paper-training
```

The command uses seed 13 and 3,157,200 complete PPO steps. Stochastic optimization and numerical-library differences can produce a different endpoint, so the bundled checkpoint remains the reference artifact.
