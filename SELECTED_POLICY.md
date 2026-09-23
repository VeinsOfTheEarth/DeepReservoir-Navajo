# Selected Navajo Reservoir Policy

The paper policy is corrected Phase 95 recovery task 105, `reward_jon_p95_oishift875to90_heff/seed_013`. It was selected from three policies that passed all ten historical screens because it had the highest objective-alignment score (0.898322), the highest viability score (0.878266), and more usable action variation than the other finalists.

## Artifacts

| Artifact | Path |
| --- | --- |
| PPO checkpoint | `artifacts/selected_policy/navajo_reservoir_selected_policy.zip` |
| Evaluation metrics | `artifacts/selected_policy/selected_policy_eval_metrics.json` |
| Daily evaluation rollout | `artifacts/selected_policy/selected_policy_eval_rollout.parquet` |
| Training trace | `artifacts/selected_policy/train_update_metrics.parquet` |
| Selected-family metrics | `artifacts/selected_policy/selected_policy_family_seed_metrics.csv` |
| Initial-storage sweep | `artifacts/selected_policy/initial_storage_sweep/` |
| Inflow-scaling sweep | `artifacts/selected_policy/inflow_scaling_sweep/` |

## Configuration

| Setting | Value |
| --- | --- |
| Training period | 1967-06-07 to 2013-12-31 |
| Evaluation period | 2014-01-01 to 2024-08-17 |
| Seed | 13 |
| PPO steps requested/resolved | 3,159,000 / 3,157,200 |
| Episode length / PPO rollout | 3,600 days |
| Batch size; gamma; epochs | 60; 0.999; 10 |
| Policy | four split action heads, two 64-unit layers |
| Decision hydrology | previous day |
| Storage datum / normalization | 2019 elevation-capacity / training window |
| Storage-budget target | 87.5% of maximum storage |
| San Juan / NIIP release caps | 5,000 / 2,500 cfs |

The reward specification is:

```text
dam_safety:spill_guard_warn98@1.00,
storage_control:target_oishift875to90_concave0_softupper98@2.50,
hydropower:positive_efficiency@1.50,
flooding:penalty_caps_archuleta_bluff@0.25,
niip:delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon@3.00,
esa_min_flow:green_logistic_jon@2.50,
esa_spring_peak_release:farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar@2.00
```

## Corrected evaluation metrics

| Metric | Selected | Historic benchmark | Pass |
| --- | ---: | ---: | --- |
| Spill volume (acre-ft) | 0 | 0 | yes |
| Storage / maximum | 0.742550 | 0.685630 | yes |
| Minimum-flow days | 0.967285 | 0.958011 | yes |
| Flood-safe days | 0.999742 | 0.999742 | yes |
| SPR 10,000 cfs / 5 d | 0.090909 | 0.090909 | yes |
| SPR 8,000 cfs / 10 d | 0.272727 | 0.181818 | yes |
| SPR 5,000 cfs / 21 d | 0.272727 | 0.272727 | yes |
| SPR 2,500 cfs / 10 d | 0.727273 | 0.727273 | yes |
| Hydropower / maximum | 0.302581 | 0.300238 | yes |
| Annual NIIP volume / historic | 1.046757 | 1.000000 | yes |

NIIP daily demand attainment is 0.964429 and is reported diagnostically rather than used as a hard screen.

```python
from deepreservoir.drl import selected_policy
agent = selected_policy.load_selected_policy_model(device="cpu")
```
