# DeepReservoir-Navajo

Research code, processed inputs, trained-model artifacts, and figure builders for a study of deep-reinforcement-learning control of Navajo Reservoir in the San Juan River basin.

> **Prepublication status:** The physical environment, metrics, and corrected Phase 95 search workflow include the manuscript-audit corrections through September 2026. The bundled model and model-dependent paper figures are an archived pre-correction baseline. They remain available for provenance and figure development, but they are not the final corrected paper policy or final results. Final policy selection and one coordinated result-figure refresh are still in progress.

## Repository contents

| Location | Contents |
| --- | --- |
| `src/deepreservoir/` | Reservoir environment, structured policy, rewards, metrics, training, and evaluation code |
| `config_files/` | Archived baseline and corrected-search configurations |
| `data/` | Processed model inputs and calibration products |
| `results/corrected-phase95-recovery/` | Compact aggregate output from the completed corrected 128-policy search |
| `artifacts/legacy_phase95_policy/` | Archived pre-correction checkpoint and evaluation artifacts |
| `paper/figures/` | Main-text and appendix figure builders, data, editable sources, and exports |
| `paper/tables/` | Table builders and current archived-baseline outputs |
| `tests/` | Scientific and reproducibility checks |

The training record begins on **June 7, 1967** and ends on December 31, 2013. Evaluation covers January 1, 2014 through August 17, 2024.

## Install and verify

```powershell
conda env create -f environment.yml
conda activate deepreservoir
python -m pip install -e ".[test]"
python -m pip check
python -m pytest -q
```

Rebuild the currently cataloged programmatic main-paper figures and the portable appendix figures:

```powershell
python paper/build-figures.py
python paper/figures/appendix/a-data-environment/build.py
python paper/figures/appendix/b-hydropower/build.py
python paper/figures/appendix/c-hydrologic-simplifications/build.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for evaluation, training, tables, and perturbation experiments. [CORRECTED_SEARCH.md](CORRECTED_SEARCH.md) describes the corrected Phase 95 search and explains why the archived policy is not the final paper policy.

## Data and attribution

[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) explains the included processed inputs. [DATA_SOURCES.md](DATA_SOURCES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) record source, attribution, and licensing information. Large source reports and workbooks that are not runtime dependencies are linked to their authoritative copies instead of being redistributed here.

The repository-controlled pickle files are trusted release artifacts. Do not load replacement pickle files from untrusted sources.

## Use and release status

This is research software and is not an operational decision authority. Reservoir deployment would require independent validation, operator review, and integration with operational forecasts and procedures.

The project software license will be added after the approved copyright-holder wording is confirmed. Third-party data, fonts, imagery, and editable assets remain subject to the terms listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For the source-snapshot lineage and exclusions, see [PROVENANCE.md](PROVENANCE.md). Citation metadata are in [CITATION.cff](CITATION.cff).
