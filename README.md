# DeepReservoir-Navajo

Research code, processed inputs, trained-model artifacts, and figure builders for a study of deep-reinforcement-learning control of Navajo Reservoir in the San Juan River basin.

The repository contains the corrected Phase 95 search and the selected policy from recovery task 105 (`reward_jon_p95_oishift875to90_heff`, seed 13). The policy passed all ten historical screening criteria and led the three all-screen finalists in both objective alignment and viability. Model-dependent figures and tables have been rebuilt from this checkpoint.

## Repository contents

| Location | Contents |
| --- | --- |
| `src/deepreservoir/` | Reservoir environment, structured policy, rewards, metrics, training, and evaluation code |
| `config_files/` | Selected-policy and corrected-search configurations |
| `data/` | Processed model inputs and calibration products |
| `results/corrected-phase95-recovery/` | Aggregate output from all 128 corrected search policies |
| `artifacts/selected_policy/` | Selected checkpoint, evaluation, training trace, and perturbation results |
| `paper/figures/` | Main-text and appendix figure builders, data, editable sources, and exports |
| `paper/tables/` | Table builders and corrected selected-policy outputs |
| `tests/` | Scientific and reproducibility checks |

Training uses June 7, 1967 through December 31, 2013. Evaluation covers January 1, 2014 through August 17, 2024.

## Install and verify

```powershell
conda env create -f environment.yml
conda activate deepreservoir
python -m pip install -e ".[test]"
python -m pip check
python -m pytest -q
```

Rebuild figures and tables:

```powershell
python paper/build-figures.py
python paper/tables/selected_policy_comparison_with_historic/build.py
python paper/tables/selected_policy_metrics/build.py
python paper/figures/appendix/a-data-environment/build.py
python paper/figures/appendix/b-hydropower/build.py
python paper/figures/appendix/c-hydrologic-simplifications/build.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for evaluation, training, and perturbation commands. [CORRECTED_SEARCH.md](CORRECTED_SEARCH.md) documents the corrected search and selection.

## Data and use

[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md), [DATA_SOURCES.md](DATA_SOURCES.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) document the included data and their sources. Repository-controlled pickle files are trusted release artifacts; do not load replacements from untrusted sources.

This is research software, not an operational decision authority. Operational use would require independent validation, operator review, and integration with operational forecasts and procedures.

The project software license will be added after the approved copyright-holder wording is confirmed. Citation metadata are in [CITATION.cff](CITATION.cff).
