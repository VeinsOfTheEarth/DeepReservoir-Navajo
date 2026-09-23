> **Prepublication artifact status:** This document describes the archived pre-correction Phase 95 seed-004 policy. Its weights were trained under superseded model definitions, so it is retained for provenance and is not the final corrected paper policy. See [CORRECTED_SEARCH.md](CORRECTED_SEARCH.md).

# Archived Pre-correction Navajo Reservoir Policy

This document records the former paper-candidate policy retained as a pre-correction baseline. It is derived from the Phase 95 run
`reward_jon_p95_peak875_hdisceff/seed_004`.

The archived public config is:

```text
config_files/selected_policy_navajo_reservoir.json
```

The archived tracked model artifact is:

```text
artifacts/legacy_phase95_policy/navajo_reservoir_selected_policy.zip
```

The frozen evaluation artifacts used by the archived model-dependent figures are:

```text
artifacts/legacy_phase95_policy/selected_policy_eval_metrics.csv
artifacts/legacy_phase95_policy/selected_policy_eval_metrics.json
artifacts/legacy_phase95_policy/selected_policy_eval_rollout.parquet
```

## Archived Policy Summary

The archived policy is a PPO controller trained for daily Navajo Reservoir operations. It uses a four-action release controller with split policy heads so that SPR-specific observations do not directly feed the non-SPR San Juan release branches.

The original hard evaluation target was to match or exceed historical management on:

- no spill,
- storage,
- ESA minimum flow,
- flood-safe days,
- all four SPR threshold frequencies,
- hydropower,
- NIIP annual volume.

NIIP daily timing remains a diagnostic, not a hard pass/fail target.

## Archived Run Artifact

The source run is local-only unless intentionally released:

```text
runs/reward_jon_p95_peak875_hdisceff/seed_004
```

Important files:

| Artifact | Path |
| --- | --- |
| Public selected model | `artifacts/legacy_phase95_policy/navajo_reservoir_selected_policy.zip` |
| Public selected metrics JSON | `artifacts/legacy_phase95_policy/selected_policy_eval_metrics.json` |
| Public selected metrics CSV | `artifacts/legacy_phase95_policy/selected_policy_eval_metrics.csv` |
| Public selected rollout | `artifacts/legacy_phase95_policy/selected_policy_eval_rollout.parquet` |
| Source selected model | `last_model.zip` |
| Training-best diagnostic model | `train_best_total_reward_model.zip` |
| Frozen resolved config | `resolved_config.json` |
| Run manifest | `run_manifest.json` |
| Holdout metrics | `eval__holdout_2014_2024_08_17/eval_metrics.json` |
| Holdout rollout | `eval__holdout_2014_2024_08_17/eval_rollout.parquet` |

The model can be loaded with:

```python
from deepreservoir.drl import selected_policy

agent = selected_policy.load_selected_policy_model(device="cpu")
```

## Archived Settings

| Setting | Value |
| --- | --- |
| Training period | 1967-06-07 to 2013-12-31 |
| Evaluation period | 2014-01-01 to 2024-08-17 |
| Seed | 4 |
| PPO timesteps requested | 3,159,000 |
| PPO episode length / n_steps | 3,600 |
| Batch size | 60 |
| Gamma | 0.999 |
| Policy type | `split_action_heads_sj_no_spr` |
| Policy architecture | two 64-unit layers for actor and critic |
| Observation context | `storage_niiphist_esa_req_sprall_advice_budget_spill` |
| Action mode | `esa_base_spr_proxy_4d_max` |
| SPR advice mode | `days_remaining` |
| Training hydrology transform | `match_holdout_annual_mean` |
| ESA deterministic floor | off |
| ESA/baseflow action cap | 1.5x daily deficit |
| SPR physical allocation priority | on |
| SPR owns San Juan window | off |
| Max controlled San Juan release | 5,000 cfs |
| Max NIIP release | 2,500 cfs |

## Reward Components

The selected reward specification is:

```text
dam_safety:spill_guard_warn98@1.00,
storage_control:target_peak875_concave0_softupper98@2.50,
hydropower:positive_discretionary_efficiency@1.50,
flooding:penalty_caps_jon@0.25,
niip:delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon@3.00,
esa_min_flow:green_logistic_jon@2.50,
esa_spring_peak_release:farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong@2.00
```

Plain-language interpretation:

- Dam safety discourages spill, with warning pressure beginning near 98% storage.
- Storage rewards a high operating target around 87.5% of peak storage with soft pressure near the upper guard.
- Hydropower rewards generation and shapes only discretionary San Juan releases toward more efficient generation.
- Flood control penalizes downstream flow-cap exceedance.
- NIIP rewards meeting annual historic-demand volume with softened overdelivery behavior.
- ESA minimum flow rewards maintaining the 500 cfs Farmington minimum flow.
- SPR rewards useful action-proxy threshold support for prescribed spring peak thresholds without using perfect future Animas knowledge.

### Flood-location correction

The archived Phase-95 checkpoint was trained with
`flooding:penalty_caps_jon`. In that historical implementation, the 5,000 cfs
term was evaluated with the Farmington proxy, which includes Animas flow. That
does not match the intended criterion at the San Juan River near Archuleta
gage. The legacy variant remains registered so the training configuration and
archived reward record are represented honestly.

Current code provides `flooding:penalty_caps_archuleta_bluff` for future
training. It applies the 5,000 cfs term to controlled San Juan release plus
spill, used as a daily outlet-flow proxy for Archuleta, and the 12,000 cfs term
to the existing two-day-lagged Farmington proxy for flow near Bluff. The proxy
does not model routing or local gains between Navajo Dam and Archuleta.

Rescoring the frozen evaluation trajectory against the intended locations
gives 100.000% flood-safe days for the selected policy on 3,880 days with both
modeled proxies available and 99.974% for historic management on 3,881 days
with paired Archuleta and Bluff observations. The frozen
metric artifact below retains the earlier 97.785% and 96.213% values because it
records the code used when the artifact was generated. Changing the reward in
current code does not retroactively retrain the checkpoint.

## Action Semantics

The selected action vector has four components:

1. ESA/baseflow action: requests controlled San Juan release to help meet the 500 cfs Farmington minimum.
2. NIIP action: requests NIIP delivery.
3. SPR proxy action: selects a spring peak threshold request; the environment computes the controlled San Juan bridge release needed with observed same-day Animas flow.
4. Discretionary San Juan action: requests additional controlled San Juan release.

The selected-policy controlled San Juan request is the max of ESA/baseflow need, SPR bridge need, and discretionary San Juan request. Physical availability limits are then applied, with SPR bridge release prioritized when water is scarce.

## Observation Semantics

The selected observation vector contains:

- storage state,
- storage budget state,
- historic NIIP demand fraction,
- ESA required release fraction,
- current Animas SPR contribution,
- remaining SPR needs for all four thresholds,
- SPR progress for all four thresholds,
- operational SPR advice fields,
- spill pressure and spill-avoidance information.

The non-SPR San Juan policy branches exclude all `spr_` fields and `animas_spr_frac`. The SPR branch and critic see the full observation vector.

## Frozen Holdout Metrics

Evaluation window: 2014-01-01 to 2024-08-17. These values preserve the original pre-correction metric artifacts and are not the corrected candidate-selection results.

| Metric | Selected policy | Historical benchmark | Hard target passed |
| --- | ---: | ---: | --- |
| Spill volume | 0 AF | 0 AF | yes |
| Storage score | 0.719817 | 0.709053 | yes |
| ESA min-flow days met | 1.000000 | 0.958011 | yes |
| Flood-safe days (archived Farmington-based definition) | 0.977846 | 0.962133 | yes |
| SPR 10,000 cfs / 5d frequency | 0.181818 | 0.090909 | yes |
| SPR 8,000 cfs / 10d frequency | 0.363636 | 0.181818 | yes |
| SPR 5,000 cfs / 21d frequency | 0.545455 | 0.272727 | yes |
| SPR 2,500 cfs / 10d frequency | 1.000000 | 0.727273 | yes |
| Hydropower score | 0.303697 | 0.303167 | yes |
| NIIP annual volume fraction | 1.055795 | 1.000000 target | yes |
| NIIP daily demand timing | 0.983964 | diagnostic only | not screened |

Composite diagnostic values:

- Objective alignment score: 0.933916
- Experiment diagnostic score: 0.916385
- Controlled San Juan unattributed volume: 1,209,436 AF

## Reproducibility Aliases

These legacy/internal aliases are retained for exact traceability:

| Public concept | Legacy/internal name |
| --- | --- |
| Selected policy config | `selected_policy_navajo_reservoir` |
| Selected policy family | `reward_jon_p95_peak875_hdisceff` |
| Selected seed | `seed_004` |
| Four-action controller | `esa_base_spr_proxy_4d_max` |
| Split policy | `split_action_heads_sj_no_spr` |
| Operational SPR advice | `days_remaining` |
| Selected observation preset | `storage_niiphist_esa_req_sprall_advice_budget_spill` |
