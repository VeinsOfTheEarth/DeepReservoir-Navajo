> **Prepublication artifact status:** This document describes the archived pre-correction Phase 95 seed-004 policy. Its weights were trained under superseded model definitions, so it is retained for provenance and is not the final corrected paper policy. See [CORRECTED_SEARCH.md](CORRECTED_SEARCH.md).

# Archived Phase-95 Policy Artifact Manifest

This manifest records the archived pre-correction model artifact and its local source artifacts.

The trained PPO model and frozen evaluation artifacts are included at:

```text
artifacts/legacy_phase95_policy/navajo_reservoir_selected_policy.zip
artifacts/legacy_phase95_policy/selected_policy_eval_metrics.csv
artifacts/legacy_phase95_policy/selected_policy_eval_metrics.json
artifacts/legacy_phase95_policy/selected_policy_eval_rollout.parquet
```

| Public artifact | SHA256 |
| --- | --- |
| `artifacts/legacy_phase95_policy/navajo_reservoir_selected_policy.zip` | `15A68C98679134E80B451497F7B910EF822F9BB04711414D0E48F437CFF22B09` |
| `artifacts/legacy_phase95_policy/selected_policy_eval_metrics.csv` | `95EDFB57B6EA5F7A959801ACA8EED4B66BC88874A0563675CB56D994BB9D493A` |
| `artifacts/legacy_phase95_policy/selected_policy_eval_metrics.json` | `5C8578033623753718A3447191AAB7E00C7A2DA234068157B692FF06E816D9CC` |
| `artifacts/legacy_phase95_policy/selected_policy_eval_rollout.parquet` | `D01F74C5E48519C4B8E32018618AA52956F95DCC8BCD4B273250A53A18CD27C6` |

The source run provenance is:

```text
runs/reward_jon_p95_peak875_hdisceff/seed_004
```

The `runs/` directory is ignored by git. These hashes are included so the stable public artifacts above can be checked against the local source run.

| Artifact | SHA256 |
| --- | --- |
| `last_model.zip` | `15A68C98679134E80B451497F7B910EF822F9BB04711414D0E48F437CFF22B09` |
| `resolved_config.json` | `239D8BA10FE31231CF61744134402732C32F60F2C560DE1DDF22305A08BA4A0B` |
| `run_manifest.json` | `CCFC296489BB8705173B9BDED5674EFD2A4651905FC397189340C293601DEE3D` |
| `eval__holdout_2014_2024_08_17/eval_metrics.json` | `5C8578033623753718A3447191AAB7E00C7A2DA234068157B692FF06E816D9CC` |
| `eval__holdout_2014_2024_08_17/eval_rollout.parquet` | `D01F74C5E48519C4B8E32018618AA52956F95DCC8BCD4B273250A53A18CD27C6` |

## Artifact Release Decision

The archived model weights, rollout, and metrics are tracked so the prepublication figures do not depend on an ignored local `runs/` directory. Larger report bundles should still live outside git unless the paper release needs them directly.

Recommended options:

1. GitHub Release assets or Zenodo for larger figure/report bundles.
2. Git LFS only if future model artifacts become too large for normal git history.

The public snapshot does not depend on an ignored local `runs/` directory for loading the archived policy or building the included figures. Public instructions should use `src/deepreservoir/drl/selected_policy.py` and the stable artifact paths above.
