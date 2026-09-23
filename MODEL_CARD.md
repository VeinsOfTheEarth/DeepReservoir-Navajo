> **Prepublication artifact status:** This document describes the archived pre-correction Phase 95 seed-004 policy. Its weights were trained under superseded model definitions, so it is retained for provenance and is not the final corrected paper policy. See [CORRECTED_SEARCH.md](CORRECTED_SEARCH.md).

# Model Card: Archived Phase-95 Navajo Reservoir PPO Policy

## Model

Archived pre-correction policy candidate for Navajo Reservoir daily operations.

Internal provenance:

- Family: `reward_jon_p95_peak875_hdisceff`
- Seed: `004`
- Public model artifact: `artifacts/legacy_phase95_policy/navajo_reservoir_selected_policy.zip`
- Source model artifact: `last_model.zip`

Public config:

```text
config_files/selected_policy_navajo_reservoir.json
```

## Intended Use

Research evaluation of reinforcement-learning reservoir management under historical hydrology and specified operating objectives.

This artifact is released as research software. It is intended to be easy to run
and inspect for paper reproduction and follow-on research, but it is not a
supported official model release or operational decision product.

Primary use cases:

- reproduce the archived pre-correction analysis and figure inputs,
- generate figures,
- compare policy behavior against historical management,
- support stress-testing and sensitivity analyses.

## Not Intended For

This model is not an operational decision authority. It should not be used for real reservoir operations without independent review, validation, stakeholder approval, and operational forecasting integration. The repository is also not intended to provide long-term software support, service-level guarantees, or a turnkey production deployment.

## Inputs

The policy observes a daily state vector including storage, NIIP historic demand, ESA minimum-flow need, Animas contribution to SPR thresholds, SPR progress/advice fields, and spill-pressure features.

The non-SPR San Juan release branches do not directly observe SPR fields; the SPR action branch does.

## Outputs

The model emits four normalized actions interpreted by the environment as:

1. ESA/baseflow request,
2. NIIP release request,
3. SPR threshold-proxy request,
4. discretionary San Juan release request.

The environment combines controlled San Juan requests with a MAX rule and applies physical availability constraints.

## Training

- Algorithm: PPO
- Training period: 1967-06-07 to 2013-12-31
- Requested timesteps: 3,159,000
- Episode length: 3,600 days
- Bias-corrected training hydrology: annual holdout-mean matching for San Juan inflow, evaporation, and Animas Farmington flow

## Evaluation

Evaluation period: 2014-01-01 to 2024-08-17.

In its original pre-correction evaluation, the archived policy matched or exceeded historical management on the then-used hard screen:

- no spill,
- storage,
- ESA minimum flow,
- flood safety,
- SPR 10k/8k/5k/2.5k frequencies,
- hydropower,
- NIIP annual volume.

NIIP daily timing was reported diagnostically. A separate corrected retraining with the same family label and seed number passed seven of ten screening criteria; that run is not a rescore of this archived checkpoint. See `CORRECTED_SEARCH.md`.

## Known Limitations

- The Phase-95 checkpoint was trained with a flood reward whose 5,000 cfs term
  used a Farmington proxy that included Animas flow. The intended operating
  criterion is at Archuleta. Current code contains a corrected reward variant,
  and post hoc evaluation uses the intended Archuleta/Bluff locations, but the
  archived checkpoint itself has not been retrained with that correction.
- The policy is a single seed from a larger seed hunt; related families showed near-misses and some seed variability.
- The frozen checkpoint and metrics are the reference for reproducing the archived analysis. They are not the final corrected paper results.
- The policy still has substantial unattributed controlled San Juan release in diagnostic accounting.
- The policy uses an operational SPR advice feature, but not a full hydrologic forecast model.
- Generalization beyond the historical evaluation period must be tested separately.
