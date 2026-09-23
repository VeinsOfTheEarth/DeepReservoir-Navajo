# Corrected Phase 95 policy search

The corrected Phase 95 recovery evaluated eight reward families across seeds 0--15 after the manuscript audit corrected the flood locations, spring-peak calendar logic, decision-time hydrology, irrigation fallback, storage datum and normalization, storage-budget target, and incomplete initial spring-season handling. The 128 policies used the same architecture, reward weights, hydrology transform, release caps, training budget, and evaluation window.

Training covered June 7, 1967 through December 31, 2013. Each run requested 3,159,000 proximal-policy-optimization steps and resolved to 3,157,200 complete steps. Deterministic evaluation covered January 1, 2014 through August 17, 2024.

All 128 tasks completed. Three endpoint policies met or exceeded the corrected historic benchmark on all ten screening criteria:

| Task | Reward family | Seed | Objective alignment | Viability |
| ---: | --- | ---: | ---: | ---: |
| **105** | **`reward_jon_p95_oishift875to90_heff`** | **13** | **0.898322** | **0.878266** |
| 14 | `reward_jon_p95_plateau90_96_heff` | 1 | 0.883300 | 0.852816 |
| 15 | `reward_jon_p95_oishift875to90_heff_niip275` | 1 | 0.882812 | 0.856460 |

Task 105 was selected because it led both composite scores and retained substantially more variation in its four action outputs than the other finalists. Its checkpoint, rollout, metrics, training trace, perturbation results, and rebuilt figures are included under `artifacts/selected_policy/` and `paper/`.

The workflow is implemented in `src/deepreservoir/drl/phase95_recovery.py`, with the specification at `config_files/reward_jon_phase95_corrected_recovery.json`. Aggregate results and hashes are under `results/corrected-phase95-recovery/`.

```powershell
python -B -m deepreservoir.drl.phase95_recovery validate `
  --spec config_files/reward_jon_phase95_corrected_recovery.json
```
