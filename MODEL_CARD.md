# Model Card: Selected Navajo Reservoir PPO Policy

## Model

- Family: `reward_jon_p95_oishift875to90_heff`
- Recovery task: 105
- Seed: 13
- Model: `artifacts/selected_policy/navajo_reservoir_selected_policy.zip`
- Config: `config_files/selected_policy_navajo_reservoir.json`

The model is a proximal-policy-optimization controller for research evaluation of daily Navajo Reservoir operations. It emits four normalized actions for environmental minimum flow/baseflow, irrigation delivery, spring-peak support, and discretionary San Juan release. Split actor heads prevent spring-specific observations from directly entering the other San Juan release branches.

## Training and evaluation

- Training: 1967-06-07 through 2013-12-31
- Requested/resolved steps: 3,159,000 / 3,157,200
- Evaluation: 2014-01-01 through 2024-08-17
- Decision hydrology: previous day's observed inflow and Animas flow
- Storage datum: 2019 elevation-capacity relationship
- Storage normalization: training window
- Spring advice: operational `days_remaining` fields without future-flow observations

Task 105 passed all ten historical screens. Selected-policy results include zero spill; storage score 0.742550; minimum-flow attainment 0.967285; flood-safe fraction 0.999742; hydropower score 0.302581; annual irrigation volume fraction 1.046757; and the four spring-frequency values recorded in `SELECTED_POLICY.md`.

## Intended use and limitations

The artifact supports paper reproduction, scientific inspection, stress testing, and follow-on research. It is not an operational decision authority. Deployment would require independent validation, operator and stakeholder review, forecast integration, and procedures for conditions outside the historical record. The checkpoint is one stochastic training outcome; the selected-family seed distribution is included to show variability.
