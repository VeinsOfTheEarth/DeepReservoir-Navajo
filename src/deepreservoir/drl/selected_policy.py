"""Metadata and loading helpers for the corrected selected Navajo policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTED_CONFIG_PATH = REPO_ROOT / "config_files" / "selected_policy_navajo_reservoir.json"
SELECTED_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "selected_policy"
SELECTED_MODEL_PATH = SELECTED_ARTIFACT_DIR / "navajo_reservoir_selected_policy.zip"
SELECTED_EVAL_METRICS_JSON_PATH = SELECTED_ARTIFACT_DIR / "selected_policy_eval_metrics.json"
SELECTED_EVAL_METRICS_CSV_PATH = SELECTED_ARTIFACT_DIR / "selected_policy_eval_metrics.csv"
SELECTED_EVAL_ROLLOUT_PATH = SELECTED_ARTIFACT_DIR / "selected_policy_eval_rollout.parquet"
SELECTED_INITIAL_STORAGE_SWEEP_DIR = SELECTED_ARTIFACT_DIR / "initial_storage_sweep"
SELECTED_INITIAL_STORAGE_SWEEP_STORAGE_PATH = SELECTED_INITIAL_STORAGE_SWEEP_DIR / "initial_storage_sweep_storage.parquet"
SELECTED_INITIAL_STORAGE_SWEEP_SUMMARY_PATH = SELECTED_INITIAL_STORAGE_SWEEP_DIR / "initial_storage_sweep_summary.csv"
SELECTED_INFLOW_SCALING_SWEEP_DIR = SELECTED_ARTIFACT_DIR / "inflow_scaling_sweep"
SELECTED_INFLOW_SCALING_SWEEP_STORAGE_PATH = SELECTED_INFLOW_SCALING_SWEEP_DIR / "inflow_scaling_sweep_storage.parquet"
SELECTED_INFLOW_SCALING_SWEEP_SUMMARY_PATH = SELECTED_INFLOW_SCALING_SWEEP_DIR / "inflow_scaling_sweep_summary.csv"

SELECTED_PUBLIC_NAME = "selected_policy_navajo_reservoir"
SELECTED_SOURCE_FAMILY = "reward_jon_p95_oishift875to90_heff"
SELECTED_TASK_ID = 105
SELECTED_SEED = 13
SELECTED_EVAL_WINDOW = "holdout_2014_2024_08_17"

SELECTED_REWARD_SPEC = (
    "dam_safety:spill_guard_warn98@1.00,"
    "storage_control:target_oishift875to90_concave0_softupper98@2.50,"
    "hydropower:positive_efficiency@1.50,"
    "flooding:penalty_caps_archuleta_bluff@0.25,"
    "niip:delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon@3.00,"
    "esa_min_flow:green_logistic_jon@2.50,"
    "esa_spring_peak_release:"
    "farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar@2.00"
)

SELECTED_OBSERVATION_COLUMNS = (
    "storage_af", "storage_budget_frac", "niip_historic_demand_frac",
    "esa_required_release_frac", "animas_spr_frac",
    "spr_needed_frac_10000cfs_5d", "spr_needed_frac_8000cfs_10d",
    "spr_needed_frac_5000cfs_21d", "spr_needed_frac_2500cfs_10d",
    "spr_progress_10000cfs_5d", "spr_progress_8000cfs_10d",
    "spr_progress_5000cfs_21d", "spr_progress_2500cfs_10d",
    "spr_advice_target_req05_frac", "spr_advice_threshold_frac",
    "spr_advice_progress_frac", "spr_advice_viability_frac",
    "spr_advice_active", "spill_pressure_frac", "spill_avoidance_sj_frac",
)

SELECTED_ACTION_MODE = "esa_base_spr_proxy_4d_max"
SELECTED_POLICY_TYPE = "split_action_heads_sj_no_spr"
SELECTED_OBS_CONTEXT = "storage_niiphist_esa_req_sprall_advice_budget_spill"
SELECTED_SPR_ADVICE_MODE = "days_remaining"
SELECTED_TRAIN_HYDROLOGY_TRANSFORM = "match_holdout_annual_mean"
SELECTED_DECISION_HYDROLOGY_TIMING = "previous_day"
SELECTED_NIIP_FALLBACK_MODE = "training_only"
SELECTED_STORAGE_DATUM_MODE = "elevation_2019"
SELECTED_STORAGE_NORMALIZATION = "train_window"
SELECTED_STORAGE_BUDGET_TARGET_FRAC = 0.875
SELECTED_MASK_INCOMPLETE_INITIAL_SPR = True


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


SELECTED_ARTIFACT_SHA256 = _load_json(SELECTED_ARTIFACT_DIR / "checksums.json")
SELECTED_METRICS = _load_json(SELECTED_EVAL_METRICS_JSON_PATH)


def load_selected_policy_config(path: str | Path = SELECTED_CONFIG_PATH) -> dict[str, Any]:
    """Load the corrected selected-policy configuration."""
    return _load_json(Path(path))


def selected_model_path(path: str | Path = SELECTED_MODEL_PATH) -> Path:
    return Path(path)


def selected_eval_metrics_path(path: str | Path = SELECTED_EVAL_METRICS_JSON_PATH) -> Path:
    return Path(path)


def selected_eval_rollout_path(path: str | Path = SELECTED_EVAL_ROLLOUT_PATH) -> Path:
    return Path(path)


def load_selected_policy_model(
    path: str | Path = SELECTED_MODEL_PATH,
    *,
    env: Any | None = None,
    device: str = "auto",
) -> Any:
    from stable_baselines3 import PPO
    return PPO.load(Path(path), env=env, device=device)


def selected_experiment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_selected_policy_config() if config is None else config
    experiments = cfg.get("experiments", [])
    if len(experiments) != 1:
        raise ValueError(f"Expected exactly one selected-policy experiment, found {len(experiments)}")
    return dict(experiments[0])
