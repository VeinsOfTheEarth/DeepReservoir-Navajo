# Corrected Phase 95 policy search

The corrected Phase 95 recovery repeated the original eight reward families across seeds 0--15 after the manuscript audit identified implementation and metric issues. The 128 endpoint policies used the same architecture, reward weights, hydrology transform, release caps, training budget, and evaluation window as the original search. The controlled changes were the Archuleta/Bluff flood reward, calendar-feasible spring-peak reward, previous-day decision hydrology, training-only irrigation fallback, common 2019 storage datum, training-window storage normalization, an 87.5% storage-budget target, and masking of incomplete initial spring seasons.

Training covered June 7, 1967 through December 31, 2013. Each run requested 3,159,000 proximal-policy-optimization steps, resolved to 3,157,200 complete steps. Deterministic endpoint evaluation covered January 1, 2014 through August 17, 2024.

All 128 tasks completed. Three endpoint policies met or exceeded the corrected historic benchmark on all ten screening criteria:

| Task | Reward family | Seed | Objective alignment | Viability |
| ---: | --- | ---: | ---: | ---: |
| 105 | `reward_jon_p95_oishift875to90_heff` | 13 | 0.898322 | 0.878266 |
| 14 | `reward_jon_p95_plateau90_96_heff` | 1 | 0.883300 | 0.852816 |
| 15 | `reward_jon_p95_oishift875to90_heff_niip275` | 1 | 0.882812 | 0.856460 |

These are candidates rather than a final ranking. Operational diagnostics and the separate priority experiment must be resolved before one policy is selected for the paper.

Corrected recovery task 36 used the same reward-family label and seed number as the pre-correction paper candidate, `reward_jon_p95_peak875_hdisceff/seed_004`, but it was trained anew after the implementation corrections. That corrected retraining passed seven of ten criteria and missed storage, minimum flow, and hydropower. It is not a rescore of the bundled pre-correction checkpoint. The bundled checkpoint, tables, and dependent figures remain legacy artifacts and must not be treated as final corrected results.

The portable workflow is implemented in `src/deepreservoir/drl/phase95_recovery.py`, with the search specification at `config_files/reward_jon_phase95_corrected_recovery.json`. Compact aggregate results and their hashes are under `results/corrected-phase95-recovery/`.

```powershell
python -B -m deepreservoir.drl.phase95_recovery validate `
  --spec config_files/reward_jon_phase95_corrected_recovery.json
```
