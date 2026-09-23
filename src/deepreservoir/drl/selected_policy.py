"""Archived Phase-95 policy metadata and loading helpers.

The bundled checkpoint is retained as a reproducibility baseline for the earlier
Phase-95 analysis. It is provisional and is not the final paper-selected policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTED_CONFIG_PATH = REPO_ROOT / "config_files" / "selected_policy_navajo_reservoir.json"
SELECTED_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "legacy_phase95_policy"
SELECTED_MODEL_PATH = SELECTED_ARTIFACT_DIR / "navajo_reservoir_selected_policy.zip"
SELECTED_EVAL_METRICS_JSON_PATH = SELECTED_ARTIFACT_DIR / "selected_policy_eval_metrics.json"
SELECTED_EVAL_METRICS_CSV_PATH = SELECTED_ARTIFACT_DIR / "selected_policy_eval_metrics.csv"
SELECTED_EVAL_ROLLOUT_PATH = SELECTED_ARTIFACT_DIR / "selected_policy_eval_rollout.parquet"
SELECTED_INITIAL_STORAGE_SWEEP_DIR = SELECTED_ARTIFACT_DIR / "initial_storage_sweep"
SELECTED_INITIAL_STORAGE_SWEEP_STORAGE_PATH = (
    SELECTED_INITIAL_STORAGE_SWEEP_DIR / "initial_storage_sweep_storage.parquet"
)
SELECTED_INITIAL_STORAGE_SWEEP_SUMMARY_PATH = (
    SELECTED_INITIAL_STORAGE_SWEEP_DIR / "initial_storage_sweep_summary.csv"
)
SELECTED_INFLOW_SCALING_SWEEP_DIR = SELECTED_ARTIFACT_DIR / "inflow_scaling_sweep"
SELECTED_INFLOW_SCALING_SWEEP_STORAGE_PATH = (
    SELECTED_INFLOW_SCALING_SWEEP_DIR / "inflow_scaling_sweep_storage.parquet"
)
SELECTED_INFLOW_SCALING_SWEEP_SUMMARY_PATH = (
    SELECTED_INFLOW_SCALING_SWEEP_DIR / "inflow_scaling_sweep_summary.csv"
)

SELECTED_PUBLIC_NAME = "selected_policy_navajo_reservoir"
SELECTED_LEGACY_FAMILY = "reward_jon_p95_peak875_hdisceff"
SELECTED_LEGACY_SEED = 4
SELECTED_EVAL_WINDOW = "holdout_2014_2024_08_17"

SELECTED_REWARD_SPEC = (
    "dam_safety:spill_guard_warn98@1.00,"
    "storage_control:target_peak875_concave0_softupper98@2.50,"
    "hydropower:positive_discretionary_efficiency@1.50,"
    "flooding:penalty_caps_jon@0.25,"
    "niip:delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon@3.00,"
    "esa_min_flow:green_logistic_jon@2.50,"
    "esa_spring_peak_release:"
    "farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong@2.00"
)

SELECTED_OBSERVATION_COLUMNS = (
    "storage_af",
    "storage_budget_frac",
    "niip_historic_demand_frac",
    "esa_required_release_frac",
    "animas_spr_frac",
    "spr_needed_frac_10000cfs_5d",
    "spr_needed_frac_8000cfs_10d",
    "spr_needed_frac_5000cfs_21d",
    "spr_needed_frac_2500cfs_10d",
    "spr_progress_10000cfs_5d",
    "spr_progress_8000cfs_10d",
    "spr_progress_5000cfs_21d",
    "spr_progress_2500cfs_10d",
    "spr_advice_target_req05_frac",
    "spr_advice_threshold_frac",
    "spr_advice_progress_frac",
    "spr_advice_viability_frac",
    "spr_advice_active",
    "spill_pressure_frac",
    "spill_avoidance_sj_frac",
)

SELECTED_ACTION_MODE = "esa_base_spr_proxy_4d_max"
SELECTED_POLICY_TYPE = "split_action_heads_sj_no_spr"
SELECTED_OBS_CONTEXT = "storage_niiphist_esa_req_sprall_advice_budget_spill"
SELECTED_SPR_ADVICE_MODE = "days_remaining"
SELECTED_TRAIN_HYDROLOGY_TRANSFORM = "match_holdout_annual_mean"

SELECTED_ARTIFACT_SHA256 = {
    "navajo_reservoir_selected_policy.zip": "15A68C98679134E80B451497F7B910EF822F9BB04711414D0E48F437CFF22B09",
    "selected_policy_eval_metrics.csv": "95EDFB57B6EA5F7A959801ACA8EED4B66BC88874A0563675CB56D994BB9D493A",
    "selected_policy_eval_metrics.json": "5C8578033623753718A3447191AAB7E00C7A2DA234068157B692FF06E816D9CC",
    "selected_policy_eval_rollout.parquet": "D01F74C5E48519C4B8E32018618AA52956F95DCC8BCD4B273250A53A18CD27C6",
    "last_model.zip": "15A68C98679134E80B451497F7B910EF822F9BB04711414D0E48F437CFF22B09",
    "resolved_config.json": "239D8BA10FE31231CF61744134402732C32F60F2C560DE1DDF22305A08BA4A0B",
    "run_manifest.json": "CCFC296489BB8705173B9BDED5674EFD2A4651905FC397189340C293601DEE3D",
    "eval_metrics.json": "5C8578033623753718A3447191AAB7E00C7A2DA234068157B692FF06E816D9CC",
    "eval_rollout.parquet": "D01F74C5E48519C4B8E32018618AA52956F95DCC8BCD4B273250A53A18CD27C6",
}

SELECTED_METRICS = {
    "total_spill_af": 0.0,
    "storage_frac_of_max_possible": 0.7198171095386058,
    "esa_min_flow_frac_days_met": 1.0,
    "flooding_frac_days_met": 0.9778464708912932,
    "spr_freq_years_meeting_10000cfs_5d": 0.18181818181818182,
    "spr_freq_years_meeting_8000cfs_10d": 0.36363636363636365,
    "spr_freq_years_meeting_5000cfs_21d": 0.5454545454545454,
    "spr_freq_years_meeting_2500cfs_10d": 1.0,
    "hydropower_frac_of_max_possible": 0.30369730638142434,
    "niip_annual_volume_frac_of_contract": 1.055795140017504,
    "niip_frac_days_demand_met_in_window": 0.9839642616515096,
    "policy_objective_alignment_score": 0.933916479541438,
    "policy_experiment_diagnostic_score": 0.9163849717167568,
}


def load_selected_policy_config(path: str | Path = SELECTED_CONFIG_PATH) -> dict[str, Any]:
    """Load the archived pre-correction policy config."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def selected_model_path(path: str | Path = SELECTED_MODEL_PATH) -> Path:
    """Return the stable public path for the trained selected PPO model."""

    return Path(path)


def selected_eval_metrics_path(path: str | Path = SELECTED_EVAL_METRICS_JSON_PATH) -> Path:
    """Return the stable path for the archived metrics JSON."""

    return Path(path)


def selected_eval_rollout_path(path: str | Path = SELECTED_EVAL_ROLLOUT_PATH) -> Path:
    """Return the stable path for the archived rollout parquet."""

    return Path(path)


def load_selected_policy_model(
    path: str | Path = SELECTED_MODEL_PATH,
    *,
    env: Any | None = None,
    device: str = "auto",
) -> Any:
    """Load the trained selected PPO policy artifact.

    The import is intentionally local so importing :mod:`selected_policy` remains
    lightweight for docs/tests that only need metadata.
    """

    from stable_baselines3 import PPO

    return PPO.load(Path(path), env=env, device=device)


def selected_experiment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the single selected experiment entry from the public config."""

    cfg = load_selected_policy_config() if config is None else config
    experiments = cfg.get("experiments", [])
    if len(experiments) != 1:
        raise ValueError(
            f"Expected exactly one selected-policy experiment, found {len(experiments)}"
        )
    return dict(experiments[0])
