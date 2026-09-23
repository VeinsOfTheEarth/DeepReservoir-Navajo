"""deepreservoir.drl.metrics

Registry-driven metrics computation for DeepReservoir experiment runs.

Design goals
------------
- **Low-touch integration**: compute everything from the test rollout dataframe
  produced by :func:`deepreservoir.drl.model.run_test_rollout`.
- **Composable**: metrics are registered by name and can be grouped.
- **Batch-friendly**: metrics return a single-row DataFrame so many runs can be
  concatenated for quick comparisons.

This module intentionally focuses on wiring and extensibility; metric
definitions can be expanded over time.

How to add a new metric
-----------------------
1) Write a function that accepts ``df_test: pd.DataFrame`` (and optional kwargs)
   and returns ``dict[str, float]``.
2) Register it in ``METRIC_REGISTRY`` with a short name.
3) Add it to one or more groups in ``METRIC_GROUPS`` (e.g. "core").
4) Document any emitted output keys in ``METRIC_DEFINITIONS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

import json
import numpy as np
import pandas as pd
from deepreservoir.define_env.storage_elevation.thresholds import (
    get_navajo_storage_thresholds,
)
from deepreservoir.define_env.spring_peak_release_curve import SpringPeakReleaseCurve


_CFS_DAY_TO_ACRE_FEET = 86400.0 / 43560.0  # 1 cfs sustained for 1 day -> acre-feet
_REPO_ROOT = Path(__file__).resolve().parents[3]
_THRESHOLD_METRIC_TOL = 1e-3

FLOOD_COMPARISON_PROFILE_CORRECTED = "archuleta_bluff_common_valid"
FLOOD_COMPARISON_PROFILE_LEGACY_PHASE95 = "legacy_phase95"
FLOOD_COMPARISON_PROFILES = (
    FLOOD_COMPARISON_PROFILE_CORRECTED,
    FLOOD_COMPARISON_PROFILE_LEGACY_PHASE95,
)


# -----------------------------------------------------------------------------
# Metric definitions (for documentation / CSV headers)
# -----------------------------------------------------------------------------
#
# Notes
# - All "fraction of days" metrics (e.g., *_frac_days_*) are fractions in [0, 1] (not 0–100).
# - `hydropower_frac_of_max_possible`, `storage_frac_of_max_possible`, and `niip_annual_volume_frac_of_contract` are fractions in [0, 1] (not 0–100).
# - Objective-aligned metrics return NaN if that objective was not active in the
#   experiment (detected by absence of `rc_<objective>.*` columns in df_test).
# - Reward-component summaries are dynamic because reward specs vary by run.

METRIC_DEFINITIONS: dict[str, str] = {
    # --- Rewards ---
    "total_reward": "Sum of per-timestep total reward over the test rollout (unitless).",
    "mean_reward": "Mean per-timestep total reward over the test rollout (unitless).",

    # --- Diagnostic / operations ---
    "frac_action_0_saturated": "Fraction of days action_0 is near a bound (|action_0| >= saturation threshold; default 0.99).",
    "frac_action_1_saturated": "Fraction of days action_1 is near a bound (|action_1| >= saturation threshold; default 0.99).",
    "frac_any_action_saturated": "Fraction of days any action dimension is near a bound (default threshold 0.99).",
    "frac_release_capped_or_limited": "Fraction of days requested controlled release exceeds actual controlled release.",
    "frac_release_cap_penalty_pos": "Fraction of days outlet-cap clipping was active.",
    "mean_release_cap_penalty": "Mean outlet-cap clipping penalty over the rollout.",
    "frac_release_phys_penalty_pos": "Fraction of days physical water-availability limiting was active.",
    "mean_release_phys_penalty": "Mean physical water-availability penalty over the rollout.",
    "frac_days_deadpool_blocked": "Fraction of days releases were blocked because the reservoir started the step at or below deadpool elevation.",
    "frac_days_spilling": "Fraction of rollout days with uncontrolled spill; unavailable if any daily spill value is missing.",
    "total_spill_af": "Total uncontrolled spill volume over the rollout (acre-feet); unavailable if any daily spill value is missing.",
    "mean_spill_af_when_spilling": "Mean uncontrolled spill volume on spill days only (acre-feet/day).",
    "controlled_sj_waste_af": (
        "Backward-compatible name for controlled_sj_unattributed_af. Diagnostic proxy for controlled San Juan "
        "release beyond apparent ESA/SPR need, summed over the rollout (acre-feet). Apparent need is max(ESA "
        "500-cfs mainstem need, SPR proxy/controller need if present, otherwise the SPR curve need) plus a small "
        "slack. This intentionally excludes uncontrolled spill and should be interpreted as unattributed release, "
        "not necessarily true waste."
    ),
    "controlled_sj_waste_frac_of_sj": (
        "Backward-compatible name for controlled_sj_unattributed_frac_of_sj. controlled_sj_unattributed_af divided "
        "by total controlled San Juan mainstem release volume. Lower values mean less controlled mainstem release "
        "that is not obviously buying ESA or SPR credit."
    ),
    "controlled_sj_unattributed_af": (
        "Controlled San Juan release beyond apparent ESA/SPR need, summed over the rollout (acre-feet). This is an "
        "attribution diagnostic: some of this water may be serving hydropower, storage/flood operations, or other "
        "implicit policy behavior, so it is not automatically true waste."
    ),
    "controlled_sj_unattributed_frac_of_sj": (
        "controlled_sj_unattributed_af divided by total controlled San Juan mainstem release volume. Lower values "
        "mean less controlled mainstem release outside apparent ESA/SPR need."
    ),

    # --- Dam safety (storage) ---
    "dam_safety_frac_days_within_storage_bounds": (
        "Fraction of days storage is within the default operating band of 3%-97% of available storage above "
        "deadpool. Uses storage_agent_af_end if available, else storage_agent_af. The band is computed from the "
        "rollout max_storage_af upper bound unless an explicit high_af override is provided."
    ),
    "dam_safety_frac_days_below_min_storage": (
        "Fraction of days storage is below the default lower operating bound of 3% of available storage above "
        "deadpool."
    ),
    "dam_safety_frac_days_above_max_storage": "Fraction of days storage is above the maximum storage bound.",
    "dam_safety_max_storage_range_water_year_af": "Maximum (max-min) storage range within any water year (AF). Water year starts Oct 1.",

    # --- ESA minimum flow ---
    "esa_min_flow_frac_days_met": (
        "Fraction of days ESA minimum flow is met: (animas_farmington_q_cfs + release_sj_main_cfs) >= threshold (default 500 cfs)."
    ),

    # --- Flooding ---
    "flooding_frac_days_met": (
        "Fraction of valid days flooding constraints are satisfied: sj_at_archuleta_proxy_cfs <= 5000 AND sj_at_bluff_proxy_cfs <= 12000."
    ),
    "agent_flooding_frac_days_met": (
        "Agent flood-safe fraction from compute_flooding_comparison_metrics, evaluated on the comparison profile's shared denominator."
    ),
    "historic_flooding_frac_days_met": (
        "Historic flood-safe fraction from compute_flooding_comparison_metrics, evaluated on the same denominator as the agent fraction."
    ),
    "flooding_comparison_days": (
        "Number of days in the shared flood-comparison denominator. Corrected comparisons require all two agent proxies and two observed historic gages to be finite."
    ),


    # --- ESA spring peak release (SPR) ---
    "spr_curve_mean_abs_error_cfs": (
        "Mean absolute error between controlled Farmington flow (Animas + controlled SJ mainstem release, excluding spill) and the SPR target curve, computed over SPR-window days only (cfs)."
    ),
    "spr_curve_mean_error_cfs": (
        "Mean signed error (controlled Farmington flow - target) vs the SPR target curve over SPR-window days (cfs). Positive means over-release."
    ),
    "spr_curve_frac_days_within_500cfs": (
        "Fraction of SPR-window days where |SPR metric proxy - target curve| <= 500 cfs."
    ),
    "spr_freq_years_meeting_*": (
        "Pattern: spr_freq_years_meeting_<thr>cfs_<dur>d = fraction of water years where the Farmington proxy "
        "(`animas_farmington_q_cfs + release_sj_main_cfs`, excluding spill) is >= <thr> for at least <dur> total days during the SPR window "
        "(not necessarily consecutive)."
    ),
    "spr_target_frequency_*": (
        "Pattern: spr_target_frequency_<thr>cfs_<dur>d = recommended annual frequency target for that threshold/duration pair."
    ),
    "spr_overachievement_*": (
        "Pattern: spr_overachievement_<thr>cfs_<dur>d = achieved frequency - target frequency (positive = overachieving, negative = underachieving)."
    ),
    "spr_mean_total_window_days_above_*": (
        "Pattern: spr_mean_total_window_days_above_<thr>cfs = mean across complete spring years of the total number of SPR-window days with controlled Farmington flow >= <thr> (not necessarily consecutive)."
    ),
    "spr_mean_max_consec_days_*": (
        "Legacy pattern retained for backward compatibility in reporting only. Historical CSVs using spr_mean_max_consec_days_<thr>cfs should be interpreted as the mean across water years of total SPR-window days above threshold, not a consecutive-day run length."
    ),
    "spr_mean_frac_window_days_above_*": (
        "Pattern: spr_mean_frac_window_days_above_<thr>cfs = mean across complete spring years of the fraction of SPR-window days with controlled Farmington flow >= <thr>."
    ),

    # --- Hydropower / storage relative to maximum possible ---
    "hydropower_frac_of_max_possible": (
        "Total agent generation as a fraction of a practical maximum generation over the test rollout: "
        "sum(hydro_agent_mwh) / (max_daily_hydropower_mwh * n_valid_days). The default max_daily_hydropower_mwh is 768 MWh/day, "
        "which corresponds to the 32 MW plant-capacity ceiling over 24 hours. Returns NaN if the agent hydropower column is missing or has no valid days."
    ),
    "storage_frac_of_max_possible": (
        "Total agent storage as a fraction of maximum possible storage over the test rollout: "
        "sum(storage_agent_af_end or storage_agent_af) / sum(max_storage_af). Returns NaN if the storage column is missing or no valid maximum-storage values are available."
    ),

    # --- NIIP ---
    "niip_frac_days_demand_met_in_window": (
        "Mean across complete calendar-year seasons of the NIIP daily shape-overlap score during the active window. For each year, "
        "daily demand and delivery are each normalized by their own seasonal totals, then overlap is computed as "
        "sum(min(demand_share_t, delivery_share_t)) over active days. Values are in [0, 1], where 1 means delivery "
        "followed the demand curve perfectly day-by-day (regardless of total seasonal volume) and 0 means no daily "
        "timing overlap. By default, the active window is DOY 50–300 (inclusive) with demand>0."
    ),
    "niip_annual_volume_frac_of_contract": (
        "Equal-weight mean across complete calendar-year seasons of (delivered seasonal volume / historic-delivery proxy volume), "
        "using positive-demand dates within DOY 50-300. Both volumes integrate daily CFS over the same dates. "
        "The internal key retains its historical name; the denominator is not an independently verified contract volume."
    ),
    "niip_mean_abs_daily_error_cfs": (
        "Mean absolute daily error between NIIP delivery and NIIP demand over the active NIIP window "
        "(default DOY 50-300, demand > 0), in cfs. Lower values mean the daily delivery curve is closer "
        "to the demand curve."
    ),

    # --- Composite policy scores ---
    "policy_objective_alignment_score": (
        "Priority-weighted score in [0, 1] for documented management objectives: dam safety/no spill (30%), "
        "Level-2 obligations (55% split evenly across NIIP annual volume, prescribed SPR frequencies, and ESA "
        "500-cfs minimum flow), and Level-3 objectives (15% split evenly across flooding and hydropower)."
    ),
    "policy_experiment_diagnostic_score": (
        "Diagnostic score in [0, 1] for comparing reward-design behavior during experiments: dam safety, NIIP "
        "volume/timing behavior, prescribed SPR frequencies, ESA 500-cfs minimum flow, storage, flooding, and hydropower."
    ),
    "policy_overall_score": (
        "Legacy alias for policy_objective_alignment_score, retained for older report artifacts."
    ),
    "policy_viability_score": (
        "Legacy alias for policy_experiment_diagnostic_score, retained for older report artifacts."
    ),
    "policy_efficiency_score": (
        "Score in [0, 1] derived from controlled_sj_unattributed_frac_of_sj "
        "(backward-compatible key: controlled_sj_waste_frac_of_sj). Higher values mean less controlled San Juan "
        "release outside apparent ESA/SPR need. Spill is not penalized in this score. This is diagnostic only and "
        "is not included in the two headline policy scores."
    ),
    "policy_dam_safety_score": "Dam-safety/no-spill component score used in policy_objective_alignment_score.",
    "policy_esa_score": "ESA 500-cfs minimum-flow component score used in policy scores.",
    "policy_spr_score": "Prescribed SPR frequency-matching component score used in policy scores.",
    "policy_niip_contract_score": "NIIP annual-volume component score used in policy_objective_alignment_score.",
    "policy_niip_behavior_score": "NIIP volume-plus-timing component score used in policy_experiment_diagnostic_score.",
    "policy_niip_score": "Legacy alias for policy_niip_behavior_score.",
    "policy_storage_score": "Storage component score used in policy_experiment_diagnostic_score.",
    "policy_hydropower_score": "Hydropower component score used in policy scores.",
    "policy_flooding_score": "Flooding component score used in policy scores.",

    # --- Dynamic reward-component summaries (pattern) ---
    "sum_rc_*": (
        "Pattern: sum_rc_<objective>.<variant> = sum of that reward component over the test rollout. Emitted for every "
        "column in df_test starting with 'rc_'."
    ),
    "mean_rc_*": (
        "Pattern: mean_rc_<objective>.<variant> = mean of that reward component over the test rollout. Emitted for every "
        "column in df_test starting with 'rc_'."
    ),
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def compute_metrics(
    df_test: pd.DataFrame,
    *,
    which: str | Sequence[str] | None = "core",
    metric_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Compute one or more metrics from a test rollout dataframe.

    Parameters
    ----------
    df_test
        Test-period rollout dataframe (from DRLModel.evaluate_test()).
    which
        - "all": all metrics in METRIC_REGISTRY
        - name of a single metric (e.g. "rewards_summary")
        - name of a group (e.g. "core", "storage")
        - list of metric and/or group names
    metric_kwargs
        Optional dict mapping metric-name -> kwargs dict passed to that metric
        function.
    validate
        If True, run basic checks on df_test and required columns.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame of scalar metrics.
    """
    if metric_kwargs is None:
        metric_kwargs = {}

    keys = _resolve_metric_keys(which)
    if validate:
        validate_rollout_df(df_test)
        _validate_required_columns(df_test, keys)

    out: dict[str, float] = {}
    for name in keys:
        spec = METRIC_REGISTRY[name]
        kw = dict(metric_kwargs.get(name, {}))
        res = spec.func(df_test, **kw)
        if not isinstance(res, dict):
            raise TypeError(f"Metric {name!r} must return dict[str, float], got {type(res)}")

        # Normalize to floats where possible (leave NaN as float)
        for k, v in res.items():
            try:
                out[k] = float(v)  # type: ignore[arg-type]
            except Exception:
                out[k] = float("nan")

    return pd.DataFrame([out])


def save_metrics(
    *,
    df_test: pd.DataFrame,
    outdir: Path | str,
    which: str | Sequence[str] | None = "core",
    metric_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    stem: str = "eval_metrics",
    validate: bool = True,
) -> dict[str, Path]:
    """Compute and save metrics to CSV (and JSON for convenience)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dfm = compute_metrics(
        df_test,
        which=which,
        metric_kwargs=metric_kwargs,
        validate=validate,
    )

    csv_path = outdir / f"{stem}.csv"
    json_path = outdir / f"{stem}.json"

    dfm.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dfm.iloc[0].to_dict(), f, indent=2, sort_keys=True)

    return {"csv": csv_path, "json": json_path}


def collect_run_metrics(
    run_dirs: Sequence[Path | str],
    *,
    metrics_filename: str = "eval_metrics.csv",
    rollout_filename: str = "eval_test_rollout.parquet",
    which_if_missing: str | Sequence[str] | None = "core",
) -> pd.DataFrame:
    """Collect metrics across many run directories."""
    rows: list[pd.DataFrame] = []
    for rd in run_dirs:
        run_dir = Path(rd)
        p_metrics = run_dir / metrics_filename
        if p_metrics.exists():
            dfm = pd.read_csv(p_metrics)
        else:
            p_roll = run_dir / rollout_filename
            if not p_roll.exists():
                continue
            df_roll = pd.read_parquet(p_roll)
            dfm = compute_metrics(df_roll, which=which_if_missing)

        dfm.insert(0, "run_dir", str(run_dir))
        rows.append(dfm)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def validate_rollout_df(df_test: pd.DataFrame) -> None:
    """Basic sanity checks on the rollout dataframe."""
    if not isinstance(df_test, pd.DataFrame):
        raise TypeError("df_test must be a pandas DataFrame")
    if not isinstance(df_test.index, pd.DatetimeIndex):
        raise ValueError("df_test.index must be a pandas.DatetimeIndex")
    if df_test.index.has_duplicates:
        raise ValueError("df_test.index contains duplicate timestamps")
    if not df_test.index.is_monotonic_increasing:
        raise ValueError("df_test.index must be monotonically increasing")


# -----------------------------------------------------------------------------
# Metric registry
# -----------------------------------------------------------------------------


MetricFunc = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class MetricSpec:
    func: MetricFunc
    requires: tuple[str, ...]


def _resolve_metric_keys(which: str | Sequence[str] | None) -> list[str]:
    if which is None:
        which = "core"

    if isinstance(which, str):
        if which == "all":
            return list(METRIC_REGISTRY.keys())
        items = [w.strip() for w in which.split(",") if w.strip()]
    else:
        items = list(which)

    selected: list[str] = []
    for item in items:
        if item == "all":
            selected.extend(list(METRIC_REGISTRY.keys()))
        elif item in METRIC_GROUPS:
            selected.extend(list(METRIC_GROUPS[item]))
        elif item in METRIC_REGISTRY:
            selected.append(item)
        else:
            raise KeyError(f"Unknown metric or group: {item!r}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for k in selected:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _validate_required_columns(df_test: pd.DataFrame, metric_keys: Sequence[str]) -> None:
    missing: dict[str, list[str]] = {}
    for name in metric_keys:
        req = METRIC_REGISTRY[name].requires
        if not req:
            continue
        miss = [c for c in req if c not in df_test.columns]
        if miss:
            missing[name] = miss
    if missing:
        parts = [f"{k}: {v}" for k, v in missing.items()]
        raise KeyError("Missing required columns for metrics: " + "; ".join(parts))


# -----------------------------------------------------------------------------
# Metric implementations (starter set)
# -----------------------------------------------------------------------------


def _metric_rewards_summary(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if "reward" in df.columns:
        out["total_reward"] = float(df["reward"].sum())
        out["mean_reward"] = float(df["reward"].mean())
    return out


def _metric_reward_components_summary(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    rc_cols = [c for c in df.columns if c.startswith("rc_")]
    for col in rc_cols:
        key = col[len("rc_") :]
        out[f"sum_rc_{key}"] = float(df[col].sum())
        out[f"mean_rc_{key}"] = float(df[col].mean())
    return out


def _metric_action_saturation(df: pd.DataFrame, *, sat: float = 0.99) -> dict[str, float]:
    out: dict[str, float] = {}
    a_cols = [c for c in ("action_0", "action_1") if c in df.columns]
    if not a_cols:
        return out
    for c in a_cols:
        s = df[c].astype(float)
        out[f"frac_{c}_saturated"] = float((np.abs(s) >= float(sat)).mean())
    if len(a_cols) == 2:
        s0 = df[a_cols[0]].astype(float)
        s1 = df[a_cols[1]].astype(float)
        out["frac_any_action_saturated"] = float(((np.abs(s0) >= sat) | (np.abs(s1) >= sat)).mean())
    return out


def _metric_release_constraint_binding(df: pd.DataFrame, *, eps: float = 1e-6) -> dict[str, float]:
    """How often the requested controlled release was capped or physically limited."""
    out: dict[str, float] = {}

    actual_col = "release_agent_controlled_cfs" if "release_agent_controlled_cfs" in df.columns else "release_agent_cfs"
    if "requested_total_release_cfs" in df.columns and actual_col in df.columns:
        req = df["requested_total_release_cfs"].astype(float)
        act = df[actual_col].astype(float)
        out["frac_release_capped_or_limited"] = float((req > (act + eps)).mean())

    if "release_cap_penalty" in df.columns:
        out["frac_release_cap_penalty_pos"] = float((df["release_cap_penalty"].astype(float) > 0.0).mean())
        out["mean_release_cap_penalty"] = float(df["release_cap_penalty"].astype(float).mean())
    if "release_phys_penalty" in df.columns:
        out["frac_release_phys_penalty_pos"] = float((df["release_phys_penalty"].astype(float) > 0.0).mean())
        out["mean_release_phys_penalty"] = float(df["release_phys_penalty"].astype(float).mean())

    return out


def _metric_operational_diagnostics(df: pd.DataFrame) -> dict[str, float]:
    """Compact operational diagnostics kept out of the default scoreboard."""
    out: dict[str, float] = {}

    if "deadpool_block" in df.columns:
        out["frac_days_deadpool_blocked"] = float(df["deadpool_block"].astype(bool).mean())
    if "spill_af" in df.columns:
        spill = pd.to_numeric(df["spill_af"], errors="coerce")
        if bool(spill.notna().all()) and not spill.empty:
            spilling = spill > 0.0
            out["frac_days_spilling"] = float(spilling.mean())
            out["total_spill_af"] = float(spill.sum())
            out["mean_spill_af_when_spilling"] = (
                float(spill[spilling].mean()) if bool(spilling.any()) else 0.0
            )
        else:
            out["frac_days_spilling"] = float("nan")
            out["total_spill_af"] = float("nan")
            out["mean_spill_af_when_spilling"] = float("nan")

    return out

def _objective_is_active(df: pd.DataFrame, objective: str) -> bool:
    """Return True if the rollout includes reward-component columns for an objective.

    We use presence of `rc_<objective>.*` columns as the indicator that the
    experiment's reward spec included that objective.
    """
    prefix = f"rc_{objective}."
    return any(str(c).startswith(prefix) for c in df.columns)


def _first_present_scalar(df: pd.DataFrame, *cols: str) -> float | None:
    """Return the first scalar value found among candidate columns."""
    for col in cols:
        if col in df.columns:
            try:
                return float(df[col].iloc[0])
            except Exception:
                continue
    return None


def _ge_threshold(values: pd.Series, threshold: float, *, tol: float = _THRESHOLD_METRIC_TOL) -> pd.Series:
    """Metric threshold comparison tolerant to tiny floating-point drift."""
    return values >= (float(threshold) - abs(float(tol)))


def _le_threshold(values: pd.Series, threshold: float, *, tol: float = _THRESHOLD_METRIC_TOL) -> pd.Series:
    """Metric threshold comparison tolerant to tiny floating-point drift."""
    return values <= (float(threshold) + abs(float(tol)))


def _lt_threshold(values: pd.Series, threshold: float, *, tol: float = _THRESHOLD_METRIC_TOL) -> pd.Series:
    """Strict threshold comparison with a tiny tolerance around equality."""
    return values < (float(threshold) + abs(float(tol)))


def _resolve_operating_storage_bounds(
    df: pd.DataFrame,
    *,
    high_af: float | None = None,
) -> tuple[float, float]:
    """Resolve the default operating band for storage metrics.

    Storage in this repo is measured above deadpool, so `0 AF` corresponds to
    deadpool rather than an empty reservoir basin. For evaluation we treat
    "in bounds" as occupying the interior operating band rather than hugging the
    physical edges. By default this band is 3%-97% of the available storage
    above deadpool.
    """
    if high_af is None:
        high_af = _first_present_scalar(df, "max_storage_af")

    if high_af is None:
        high_af = get_navajo_storage_thresholds().max_storage_af

    max_storage_af = float(high_af)
    lo = 0.03 * max_storage_af
    hi = 0.97 * max_storage_af
    return lo, hi


# -----------------------------------------------------------------------------
# Objective-aligned metrics (one-per-objective first pass)
# -----------------------------------------------------------------------------


def _metric_dam_safety_frac_days_within_storage_bounds(
    df: pd.DataFrame,
    *,
    high_af: float | None = None,
    storage_col: str = "storage_agent_af_end",
) -> dict[str, float]:
    """Dam safety: % of days storage is within the operating band."""
    out: dict[str, float] = {}

    # If the experiment did not include dam_safety, report NA.
    if not _objective_is_active(df, "dam_safety"):
        out["dam_safety_frac_days_within_storage_bounds"] = float("nan")
        return out

    col = storage_col if storage_col in df.columns else "storage_agent_af"
    if col not in df.columns:
        out["dam_safety_frac_days_within_storage_bounds"] = float("nan")
        return out

    s = df[col].astype(float)

    lo, hi = _resolve_operating_storage_bounds(df, high_af=high_af)
    valid = s.notna()
    within = _ge_threshold(s, lo) & _le_threshold(s, hi)
    out["dam_safety_frac_days_within_storage_bounds"] = (
        float(within[valid].mean()) if bool(valid.any()) else float("nan")
    )
    return out


def _metric_dam_safety_storage_detail(
    df: pd.DataFrame,
    *,
    high_af: float | None = None,
    storage_col: str = "storage_agent_af_end",
) -> dict[str, float]:
    """Extra dam-safety diagnostics (kept out of the default core set)."""
    out: dict[str, float] = {}

    # If the experiment did not include dam_safety, report NA.
    if not _objective_is_active(df, "dam_safety"):
        return {
            "dam_safety_frac_days_below_min_storage": float("nan"),
            "dam_safety_frac_days_above_max_storage": float("nan"),
            "dam_safety_max_storage_range_water_year_af": float("nan"),
        }

    col = storage_col if storage_col in df.columns else "storage_agent_af"
    if col not in df.columns:
        return out

    s = df[col].astype(float)

    lo, hi = _resolve_operating_storage_bounds(df, high_af=high_af)
    out["dam_safety_frac_days_below_min_storage"] = float((~_ge_threshold(s, lo)).mean())
    out["dam_safety_frac_days_above_max_storage"] = float((~_le_threshold(s, hi)).mean())

    # Max within-water-year storage range (AF)
    idx = df.index
    wy = idx.year + (idx.month >= 10).astype(int)
    ranges = s.groupby(wy).max() - s.groupby(wy).min()
    out["dam_safety_max_storage_range_water_year_af"] = float(ranges.max()) if len(ranges) else float("nan")

    return out


def _metric_esa_min_flow_frac_days_met(
    df: pd.DataFrame,
    *,
    threshold_cfs: float = 500.0,
    animas_col: str = "animas_farmington_q_cfs",
    release_col: str = "release_sj_main_cfs",
) -> dict[str, float]:
    """ESA minimum flow: % of days (Animas @ Farmington + San Juan mainstem release) >= threshold."""
    out: dict[str, float] = {}

    # If the experiment did not include esa_min_flow, report NA.
    if not _objective_is_active(df, "esa_min_flow"):
        out["esa_min_flow_frac_days_met"] = float("nan")
        return out

    if (animas_col not in df.columns) or (release_col not in df.columns):
        out["esa_min_flow_frac_days_met"] = float("nan")
        return out

    animas = df[animas_col].astype(float)
    release = df[release_col].astype(float)
    valid = animas.notna() & release.notna()
    met = _ge_threshold(animas + release, threshold_cfs)
    out["esa_min_flow_frac_days_met"] = (
        float(met[valid].mean()) if bool(valid.any()) else float("nan")
    )
    return out


def _metric_flooding_frac_days_met(
    df: pd.DataFrame,
    *,
    same_day_thresh_cfs: float = 5000.0,
    lag2_thresh_cfs: float = 12000.0,
    q0_col: str = "sj_at_archuleta_proxy_cfs",
    qlag2_col: str = "sj_at_bluff_proxy_cfs",
) -> dict[str, float]:
    """Flooding: % of days both flood constraints are satisfied.

    The same-day quantity is the modeled San Juan outlet flow at Archuleta,
    excluding the downstream Animas contribution. The second quantity retains
    the existing two-day-lagged Farmington proxy for flow near Bluff.
    Only days with both quantities available enter the denominator.
    """
    out: dict[str, float] = {}

    # If the experiment did not include flooding, report NA.
    if not _objective_is_active(df, "flooding"):
        out["flooding_frac_days_met"] = float("nan")
        return out

    if q0_col in df.columns:
        q0 = df[q0_col].astype(float)
    elif q0_col == "sj_at_archuleta_proxy_cfs" and "sj_main_flow_cfs" in df.columns:
        # Compatibility with rollouts written before the explicit proxy field
        # was added. ``sj_main_flow_cfs`` is controlled release plus spill.
        q0 = df["sj_main_flow_cfs"].astype(float)
    else:
        out["flooding_frac_days_met"] = float("nan")
        return out

    if qlag2_col in df.columns:
        qlag2 = df[qlag2_col].astype(float)
    elif qlag2_col == "sj_at_bluff_proxy_cfs" and "sj_at_farmington_lag2_cfs" in df.columns:
        # Compatibility with rollouts written before the Bluff-proxy alias.
        qlag2 = df["sj_at_farmington_lag2_cfs"].astype(float)
    else:
        out["flooding_frac_days_met"] = float("nan")
        return out

    valid = q0.notna() & qlag2.notna()
    safe_same = _lt_threshold(q0, same_day_thresh_cfs)
    safe_lag2 = _lt_threshold(qlag2, lag2_thresh_cfs)
    out["flooding_frac_days_met"] = (
        float((safe_same[valid] & safe_lag2[valid]).mean())
        if bool(valid.any())
        else float("nan")
    )
    return out


def compute_flooding_comparison_metrics(
    df_eval: pd.DataFrame,
    *,
    profile: str = FLOOD_COMPARISON_PROFILE_CORRECTED,
    same_day_thresh_cfs: float = 5000.0,
    downstream_thresh_cfs: float = 12000.0,
) -> dict[str, float | int | str]:
    """Compare agent and historic flood safety on an explicit denominator.

    The corrected profile compares the modeled Archuleta and Bluff proxies with
    the observed Archuleta and Bluff gages on one row-aligned, common finite-day
    mask.  The Bluff proxy already contains the model's two-day routing
    approximation, so the observed Bluff series is not shifted again.

    ``legacy_phase95`` exists only to reproduce the archived Phase-95 screen.
    It uses Farmington for the 5,000-cfs term, a two-day Farmington lag for the
    12,000-cfs term, treats unavailable lag values as safe, and retains every
    rollout row in the denominator.  New scientific comparisons should use the
    corrected profile.

    Corrected comparisons intentionally require the four explicit agent and
    historic location columns.  In particular, historic reservoir release is
    never silently substituted for an unavailable observed Archuleta gage.
    """
    validate_rollout_df(df_eval)
    resolved_profile = str(profile).strip()
    if resolved_profile not in FLOOD_COMPARISON_PROFILES:
        raise ValueError(
            f"Unknown flood comparison profile {profile!r}; choose from "
            f"{list(FLOOD_COMPARISON_PROFILES)}."
        )

    def require_numeric(columns: Sequence[str]) -> dict[str, pd.Series]:
        missing = [column for column in columns if column not in df_eval.columns]
        if missing:
            raise KeyError(
                f"Flood comparison profile {resolved_profile!r} requires "
                f"columns {missing}."
            )
        return {
            column: pd.to_numeric(df_eval[column], errors="coerce")
            for column in columns
        }

    def finite(series: pd.Series) -> pd.Series:
        return pd.Series(
            np.isfinite(series.to_numpy(dtype=float, copy=False)),
            index=df_eval.index,
            dtype=bool,
        )

    if resolved_profile == FLOOD_COMPARISON_PROFILE_CORRECTED:
        columns = require_numeric(
            (
                "sj_at_archuleta_proxy_cfs",
                "sj_at_bluff_proxy_cfs",
                "sj_archuleta_q_cfs",
                "sj_bluff_q_cfs",
            )
        )
        agent_same = columns["sj_at_archuleta_proxy_cfs"]
        agent_downstream = columns["sj_at_bluff_proxy_cfs"]
        historic_same = columns["sj_archuleta_q_cfs"]
        historic_downstream = columns["sj_bluff_q_cfs"]

        agent_available = finite(agent_same) & finite(agent_downstream)
        historic_available = finite(historic_same) & finite(historic_downstream)
        comparison_mask = agent_available & historic_available

        agent_safe = _lt_threshold(agent_same, same_day_thresh_cfs) & _lt_threshold(
            agent_downstream, downstream_thresh_cfs
        )
        historic_safe = _lt_threshold(
            historic_same, same_day_thresh_cfs
        ) & _lt_threshold(historic_downstream, downstream_thresh_cfs)
    else:
        columns = require_numeric(
            (
                "sj_at_farmington_cfs",
                "sj_at_farmington_lag2_cfs",
                "sj_farmington_q_cfs",
            )
        )
        agent_same = columns["sj_at_farmington_cfs"]
        agent_downstream = columns["sj_at_farmington_lag2_cfs"]
        historic_same = columns["sj_farmington_q_cfs"]
        historic_downstream = historic_same.shift(2)

        agent_available = finite(agent_same) & finite(agent_downstream)
        historic_available = finite(historic_same) & finite(historic_downstream)
        comparison_mask = pd.Series(True, index=df_eval.index, dtype=bool)

        agent_safe = _lt_threshold(agent_same, same_day_thresh_cfs) & (
            agent_downstream.isna()
            | _lt_threshold(agent_downstream, downstream_thresh_cfs)
        )
        historic_safe = _lt_threshold(historic_same, same_day_thresh_cfs) & (
            historic_downstream.isna()
            | _lt_threshold(historic_downstream, downstream_thresh_cfs)
        )

    comparison_days = int(comparison_mask.sum())
    agent_safe_days = int((agent_safe & comparison_mask).sum())
    historic_safe_days = int((historic_safe & comparison_mask).sum())

    return {
        "flooding_comparison_profile": resolved_profile,
        "flooding_comparison_days": comparison_days,
        "agent_flooding_available_days": int(agent_available.sum()),
        "historic_flooding_available_days": int(historic_available.sum()),
        "agent_flooding_safe_days": agent_safe_days,
        "historic_flooding_safe_days": historic_safe_days,
        "agent_flooding_frac_days_met": (
            float(agent_safe_days / comparison_days)
            if comparison_days > 0
            else float("nan")
        ),
        "historic_flooding_frac_days_met": (
            float(historic_safe_days / comparison_days)
            if comparison_days > 0
            else float("nan")
        ),
    }


def _metric_hydropower_frac_of_max_possible(
    df: pd.DataFrame,
    *,
    agent_col: str = "hydro_agent_mwh",
    max_daily_mwh: float = 768.0,
) -> dict[str, float]:
    """Hydropower: total generation relative to a practical maximum (fraction).

    The denominator is ``max_daily_mwh * n_valid_days``. By default
    ``max_daily_mwh`` is 768 MWh/day, which is the 32 MW plant-capacity
    ceiling over 24 hours already enforced by the hydropower model.
    """
    out: dict[str, float] = {"hydropower_frac_of_max_possible": float("nan")}

    if agent_col not in df.columns:
        return out

    a = pd.to_numeric(df[agent_col], errors="coerce")
    n_valid = int(a.notna().sum())
    if n_valid <= 0:
        return out

    denom = float(max_daily_mwh) * float(n_valid)
    if denom <= 0.0:
        return out

    out["hydropower_frac_of_max_possible"] = float(np.nansum(a.to_numpy(dtype=float)) / denom)
    return out


def _metric_storage_frac_of_max_possible(
    df: pd.DataFrame,
    *,
    agent_col: str = "storage_agent_af_end",
    fallback_agent_col: str = "storage_agent_af",
    max_col: str = "max_storage_af",
    default_max_storage_af: float | None = None,
) -> dict[str, float]:
    """Storage: total storage relative to maximum possible storage (fraction).

    Computed as ``sum(agent storage) / sum(max storage)`` over the rollout,
    which is equivalent to the mean daily storage fraction when ``max_storage_af``
    is constant. This metric is intentionally independent of the historical series.
    """
    out: dict[str, float] = {"storage_frac_of_max_possible": float("nan")}
    agent_name = agent_col if agent_col in df.columns else fallback_agent_col
    if agent_name not in df.columns:
        return out

    storage = pd.to_numeric(df[agent_name], errors="coerce")
    if default_max_storage_af is None:
        default_max_storage_af = get_navajo_storage_thresholds().max_storage_af
    if max_col in df.columns:
        max_storage = pd.to_numeric(df[max_col], errors="coerce")
    else:
        max_storage = pd.Series(float(default_max_storage_af), index=df.index, dtype=float)

    valid = storage.notna() & max_storage.notna() & (max_storage > 0.0)
    if not bool(valid.any()):
        return out

    denom = float(np.nansum(max_storage[valid].to_numpy(dtype=float)))
    if denom <= 0.0:
        return out

    out["storage_frac_of_max_possible"] = float(
        np.nansum(storage[valid].to_numpy(dtype=float)) / denom
    )
    return out


@lru_cache(maxsize=1)
def _load_niip_historic_delivery_series() -> pd.Series | None:
    """Load the historical NIIP delivery series (CFS) from the repo data dir."""
    path = _REPO_ROOT / "data" / "niip" / "NAVAJOINDIANIRRIGATIONPROJECT07-17-2025T13.21.47.csv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(
            path,
            usecols=["Date", "Flow (cfs)"],
            skipinitialspace=True,
        )
    except Exception:
        return None

    dates = pd.to_datetime(df.get("Date"), format="%d-%b-%y", errors="coerce")
    flows = pd.to_numeric(df.get("Flow (cfs)"), errors="coerce")
    series = pd.Series(flows.to_numpy(dtype=float), index=dates, name="niip_flow_cfs")
    series = series[series.index.notna() & series.notna()]
    if series.empty:
        return None

    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _niip_get_delivery_series(
    df: pd.DataFrame,
    *,
    delivery_col: str = "release_niip_cfs",
) -> pd.Series | None:
    """Return the NIIP delivery series in CFS, or None if not available."""
    if delivery_col in df.columns:
        return df[delivery_col].astype(float)
    return None



def _niip_get_demand_series(
    df: pd.DataFrame,
    *,
    demand_col: str = "niip_demand_cfs",
) -> pd.Series | None:
    """Return the NIIP demand series in CFS.

    Preference order:
      1) Use a precomputed column (default: niip_demand_cfs) if present.
      2) Compute from the NIIP demand curve (niip_daily_demand(doy)) using df.index.

    Returns None if demand cannot be obtained.
    """
    if demand_col in df.columns:
        try:
            return df[demand_col].astype(float)
        except Exception:
            return None

    # Compute from demand curve (import lazily; may fail if data files missing).
    try:
        from deepreservoir.define_env.niip.niip_demand import niip_daily_demand
    except Exception:
        return None

    doys = df.index.dayofyear.to_numpy()
    try:
        vals = niip_daily_demand(doys)  # supports ndarray in this repo
        arr = np.asarray(vals, dtype=float)
    except Exception:
        # fallback: scalar calls
        try:
            arr = np.asarray([float(niip_daily_demand(int(d))) for d in doys], dtype=float)
        except Exception:
            return None

    arr = np.clip(arr, 0.0, None)
    return pd.Series(arr, index=df.index, name=demand_col)


def _compute_niip_delivery_and_volume_from_series(
    *,
    index: pd.DatetimeIndex,
    demand: pd.Series,
    delivery: pd.Series,
    doy_start: int = 50,
    doy_end: int = 300,
    demand_positive_eps: float = 1e-9,
    tol_cfs: float = 0.0,
) -> dict[str, float]:
    """Compute NIIP timing-overlap and annual-volume metrics from aligned series.

    Annual summaries require all days from DOY 50 through 300 and valid demand
    and active-day delivery; partial or gapped seasons do not count as years.
    The daily-shape metric intentionally measures *when* NIIP water is delivered
    rather than *how much* was delivered over the whole season. For each active
    year, demand and delivery are normalized by their own seasonal totals, then
    their overlap is computed as the sum of the per-day minima. This keeps the
    metric distinct from the annual-volume metric.
    """
    out = {
        "niip_frac_days_demand_met_in_window": float("nan"),
        "niip_annual_volume_frac_of_contract": float("nan"),
        "niip_mean_abs_daily_error_cfs": float("nan"),
    }

    idx = pd.DatetimeIndex(index)
    demand = pd.Series(demand, index=idx, dtype=float)
    delivery = pd.Series(delivery, index=idx, dtype=float)

    doys = idx.dayofyear
    in_window = (doys >= int(doy_start)) & (doys <= int(doy_end))
    active = in_window & demand.notna() & (demand.astype(float) > float(demand_positive_eps))

    n_active = int(active.sum())
    if n_active == 0:
        return out

    years = idx.year
    overlaps: list[float] = []
    ratios: list[float] = []
    for y in np.unique(years):
        year_window = in_window & (years == y)
        expected = pd.date_range(
            pd.Timestamp(year=int(y), month=1, day=1) + pd.Timedelta(days=int(doy_start) - 1),
            pd.Timestamp(year=int(y), month=1, day=1) + pd.Timedelta(days=int(doy_end) - 1),
            freq="D",
        )
        window_dates = idx[year_window]
        if len(window_dates) != len(expected) or not bool(expected.isin(window_dates).all()):
            continue
        if not bool(demand[year_window].notna().all()):
            continue
        m = active & (years == y)
        if not bool(m.any()):
            continue
        if not bool(delivery[m].notna().all()):
            continue

        demand_year = demand[m].to_numpy(dtype=float)
        delivery_year = delivery[m].to_numpy(dtype=float)

        demand_total = float(np.nansum(demand_year))
        delivery_total = float(np.nansum(delivery_year))
        if demand_total > 0.0:
            if delivery_total > 0.0:
                demand_share = np.clip(demand_year, 0.0, None) / demand_total
                delivery_share = np.clip(delivery_year, 0.0, None) / delivery_total
                overlaps.append(float(np.minimum(demand_share, delivery_share).sum()))
            else:
                overlaps.append(0.0)

        delivered_af = float(np.nansum(delivery_year) * _CFS_DAY_TO_ACRE_FEET)
        contract_af = float(np.nansum(demand_year) * _CFS_DAY_TO_ACRE_FEET)
        if contract_af <= 0.0:
            continue
        ratios.append(delivered_af / contract_af)

    if overlaps:
        out["niip_frac_days_demand_met_in_window"] = float(np.mean(overlaps))

    if ratios:
        out["niip_annual_volume_frac_of_contract"] = float(np.mean(ratios))

    active_error = active & delivery.notna()
    if bool(active_error.any()):
        err = (
            delivery[active_error].astype(float).fillna(0.0)
            - demand[active_error].astype(float).fillna(0.0)
        )
        out["niip_mean_abs_daily_error_cfs"] = float(err.abs().mean())

    return out


def _metric_niip_delivery_and_volume(
    df: pd.DataFrame,
    *,
    doy_start: int = 50,
    doy_end: int = 300,
    demand_col: str = "niip_demand_cfs",
    delivery_col: str = "release_niip_cfs",
    demand_positive_eps: float = 1e-9,
    tol_cfs: float = 0.0,
) -> dict[str, float]:
    """NIIP metrics.

    Emits:
      - niip_frac_days_demand_met_in_window : mean annual NIIP daily shape-overlap score
      - niip_annual_volume_frac_of_contract : mean complete-season delivered-volume / historic-proxy-volume ratio

    The NIIP active window is defined as DOY in [doy_start, doy_end] (inclusive) and demand > 0.
    Demand is taken from `demand_col` if present; otherwise computed from the NIIP demand curve.
    """
    out = {
        "niip_frac_days_demand_met_in_window": float("nan"),
        "niip_annual_volume_frac_of_contract": float("nan"),
        "niip_mean_abs_daily_error_cfs": float("nan"),
    }

    # If the experiment did not include niip, report NA.
    if not _objective_is_active(df, "niip"):
        return out

    demand = _niip_get_demand_series(df, demand_col=demand_col)
    delivery = _niip_get_delivery_series(df, delivery_col=delivery_col)
    if demand is None or delivery is None:
        return out

    return _compute_niip_delivery_and_volume_from_series(
        index=pd.DatetimeIndex(df.index),
        demand=demand,
        delivery=delivery,
        doy_start=doy_start,
        doy_end=doy_end,
        demand_positive_eps=demand_positive_eps,
        tol_cfs=tol_cfs,
    )




# -----------------------------------------------------------------------------
# ESA Spring Peak Release (SPR) metrics
# -----------------------------------------------------------------------------

# Table-style targets derived from the SJRIP "spring peak release" guidance:
# Each tuple is (threshold_cfs, duration_days, target_frequency_per_year).
_SPR_THRESHOLD_SPECS: tuple[tuple[float, int, float], ...] = (
    (10_000.0, 5, 0.20),
    (8_000.0, 10, 0.33),
    (5_000.0, 21, 0.50),
    (2_500.0, 10, 0.80),
)


def _metric_spring_peak_release_scoreboard(
    df: pd.DataFrame,
    *,
    curve_tolerance_cfs: float = 500.0,
    threshold_specs: Sequence[tuple[float, int, float]] = _SPR_THRESHOLD_SPECS,
) -> dict[str, float]:
    """Compact SPR scoreboard: one curve metric plus the four threshold frequencies."""
    detail = _metric_spring_peak_release(
        df,
        curve_tolerance_cfs=curve_tolerance_cfs,
        threshold_specs=threshold_specs,
    )
    out = {
        "spr_curve_mean_abs_error_cfs": float(detail.get("spr_curve_mean_abs_error_cfs", float("nan"))),
        "spr_curve_frac_days_within_500cfs": float(detail.get("spr_curve_frac_days_within_500cfs", float("nan"))),
    }
    for thr, dur, _ in threshold_specs:
        out[f"spr_freq_years_meeting_{int(thr)}cfs_{int(dur)}d"] = float(
            detail.get(f"spr_freq_years_meeting_{int(thr)}cfs_{int(dur)}d", float("nan"))
        )
    return out


def _complete_spr_season_years(
    index: pd.DatetimeIndex,
    curve: SpringPeakReleaseCurve,
    *required_series: pd.Series,
) -> set[int]:
    """Calendar spring years with every scheduled day and required flow value."""
    first_month, first_day, _ = curve.cfg.points_md_cfs[0]
    last_month, last_day, _ = curve.cfg.points_md_cfs[-1]
    complete: set[int] = set()
    for year in np.unique(index.year):
        expected = pd.date_range(
            pd.Timestamp(year=int(year), month=first_month, day=first_day),
            pd.Timestamp(year=int(year), month=last_month, day=last_day),
            freq="D",
        )
        if not bool(expected.isin(index).all()):
            continue
        if all(bool(series.reindex(expected).notna().all()) for series in required_series):
            complete.add(int(year))
    return complete


def _metric_spring_peak_release(
    df: pd.DataFrame,
    *,
    animas_col: str = "animas_farmington_q_cfs",
    release_col: str = "release_sj_main_cfs",
    curve_tolerance_cfs: float = 500.0,
    threshold_specs: Sequence[tuple[float, int, float]] = _SPR_THRESHOLD_SPECS,
) -> dict[str, float]:
    """SPR metrics: threshold/duration frequencies + curve-matching summaries.

    - Uses SpringPeakReleaseCurve to define the SPR window (target>0).
    - Evaluates SPR using controlled Farmington flow only:
      Animas @ Farmington + agent SJ mainstem release. Uncontrolled spill is
      intentionally excluded from the SPR scorecard.
    - For each (threshold, duration, target_frequency):
        * frequency of water years meeting threshold for at least duration total days within the SPR window
        * over/underachievement relative to target_frequency
        * mean total number of SPR-window days above threshold
        * mean fraction of SPR-window days above threshold
    """
    out: dict[str, float] = {}

    if animas_col not in df.columns or release_col not in df.columns:
        return out

    curve = SpringPeakReleaseCurve()
    target = curve.targets_for_date_index(df.index).astype(float)
    spr_mask = target > 0.0

    if not bool(spr_mask.any()):
        out["spr_curve_mean_abs_error_cfs"] = float("nan")
        out["spr_curve_mean_error_cfs"] = float("nan")
        out["spr_curve_frac_days_within_500cfs"] = float("nan")
        for thr, dur, tf in threshold_specs:
            thr_i = int(thr)
            dur_i = int(dur)
            out[f"spr_freq_years_meeting_{thr_i}cfs_{dur_i}d"] = float("nan")
            out[f"spr_target_frequency_{thr_i}cfs_{dur_i}d"] = float(tf)
            out[f"spr_overachievement_{thr_i}cfs_{dur_i}d"] = float("nan")
            out[f"spr_mean_total_window_days_above_{thr_i}cfs"] = float("nan")
            out[f"spr_mean_frac_window_days_above_{thr_i}cfs"] = float("nan")
        return out

    rel = df[release_col].astype(float)
    animas = df[animas_col].astype(float)
    farm = animas + rel
    farm.name = "controlled_farmington_cfs"

    err = farm - target
    valid_spr = spr_mask & farm.notna()
    out["spr_curve_mean_abs_error_cfs"] = float(err.abs()[valid_spr].mean())
    out["spr_curve_mean_error_cfs"] = float(err[valid_spr].mean())
    out["spr_curve_frac_days_within_500cfs"] = float(
        _le_threshold(err.abs()[valid_spr], curve_tolerance_cfs).mean()
    )

    idx = df.index
    complete_years = _complete_spr_season_years(idx, curve, animas, rel)
    wy = idx.year + (idx.month >= 10).astype(int)
    wy_series = pd.Series(wy, index=idx, name="wy")

    for thr, dur_days, target_freq in threshold_specs:
        thr_i = int(thr)
        dur_i = int(dur_days)

        met_flags: list[bool] = []
        total_days_above: list[int] = []
        pct_days_above: list[float] = []

        for _wy_val, g_idx in wy_series.groupby(wy_series).groups.items():
            if int(_wy_val) not in complete_years:
                continue
            g_idx = pd.DatetimeIndex(g_idx)
            g_spr = spr_mask.loc[g_idx]
            spr_days = int(g_spr.sum())
            if spr_days <= 0:
                continue

            b = g_spr & _ge_threshold(farm.loc[g_idx], thr)
            days_above = int(b.sum())

            met_flags.append(bool(days_above >= int(dur_days)))
            total_days_above.append(int(days_above))
            pct_days_above.append(float(b.sum()) / float(spr_days))

        if not met_flags:
            out[f"spr_freq_years_meeting_{thr_i}cfs_{dur_i}d"] = float("nan")
            out[f"spr_target_frequency_{thr_i}cfs_{dur_i}d"] = float(target_freq)
            out[f"spr_overachievement_{thr_i}cfs_{dur_i}d"] = float("nan")
            out[f"spr_mean_total_window_days_above_{thr_i}cfs"] = float("nan")
            out[f"spr_mean_frac_window_days_above_{thr_i}cfs"] = float("nan")
            continue

        achieved = float(np.mean(met_flags))
        out[f"spr_freq_years_meeting_{thr_i}cfs_{dur_i}d"] = achieved
        out[f"spr_target_frequency_{thr_i}cfs_{dur_i}d"] = float(target_freq)
        out[f"spr_overachievement_{thr_i}cfs_{dur_i}d"] = achieved - float(target_freq)
        out[f"spr_mean_total_window_days_above_{thr_i}cfs"] = float(np.mean(total_days_above))
        out[f"spr_mean_frac_window_days_above_{thr_i}cfs"] = float(np.mean(pct_days_above))

    return out



_POLICY_OBJECTIVE_ALIGNMENT_WEIGHTS: dict[str, float] = {
    "policy_dam_safety_score": 0.30,
    "policy_niip_contract_score": 0.55 / 3.0,
    "policy_spr_score": 0.55 / 3.0,
    "policy_esa_score": 0.55 / 3.0,
    "policy_flooding_score": 0.15 / 2.0,
    "policy_hydropower_score": 0.15 / 2.0,
}


_POLICY_EXPERIMENT_DIAGNOSTIC_WEIGHTS: dict[str, float] = {
    "policy_dam_safety_score": 0.20,
    "policy_niip_behavior_score": 0.20,
    "policy_spr_score": 0.20,
    "policy_esa_score": 0.20,
    "policy_storage_score": 0.10,
    "policy_flooding_score": 0.05,
    "policy_hydropower_score": 0.05,
}


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _clip01(value: object, default: float = 0.0) -> float:
    v = _finite_float(value, default)
    if not np.isfinite(v):
        v = float(default)
    return float(np.clip(v, 0.0, 1.0))


def _series_float(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _spr_curve_need_series(df: pd.DataFrame, animas: pd.Series) -> pd.Series:
    try:
        curve = SpringPeakReleaseCurve()
        target = pd.Series(
            curve.targets_for_date_index(df.index).astype(float),
            index=df.index,
            dtype=float,
        )
    except Exception:
        target = pd.Series(0.0, index=df.index, dtype=float)
    return (target - animas).clip(lower=0.0)


def _controlled_sj_waste_stats(
    df: pd.DataFrame,
    *,
    release_col: str = "release_sj_main_cfs",
    slack_cfs: float = 100.0,
) -> dict[str, float]:
    """Return controlled-SJ unattributed-release diagnostics.

    The proxy intentionally focuses on controlled San Juan release, not spill.
    It asks how much controlled mainstem release exceeded apparent ESA or SPR
    need on a given day. The historical key names include "waste", but the
    concept is better interpreted as unattributed release.
    """
    release = _series_float(df, release_col)
    animas = _series_float(df, "animas_farmington_q_cfs")
    esa_need = (500.0 - animas).clip(lower=0.0)

    if "spr_proxy_controller_need_cfs" in df.columns:
        spr_need = _series_float(df, "spr_proxy_controller_need_cfs")
    else:
        spr_need = _spr_curve_need_series(df, animas)

    apparent_need = pd.concat([esa_need, spr_need], axis=1).max(axis=1)
    waste_cfs = (release - apparent_need - float(slack_cfs)).clip(lower=0.0)

    controlled_sj_af = float((release.clip(lower=0.0) * _CFS_DAY_TO_ACRE_FEET).sum())
    waste_af = float((waste_cfs * _CFS_DAY_TO_ACRE_FEET).sum())
    waste_frac = waste_af / max(controlled_sj_af, 1.0)
    return {
        "controlled_sj_waste_af": waste_af,
        "controlled_sj_waste_frac_of_sj": waste_frac,
        "controlled_sj_unattributed_af": waste_af,
        "controlled_sj_unattributed_frac_of_sj": waste_frac,
    }


def _weighted_score(components: Mapping[str, object], weights: Mapping[str, float]) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        value = _finite_float(components.get(key), float("nan"))
        if np.isfinite(value):
            weighted_sum += float(weight) * float(value)
            weight_sum += float(weight)
    return float(weighted_sum / weight_sum) if weight_sum > 0.0 else float("nan")


def _dam_safety_no_spill_score(metrics: Mapping[str, object]) -> float:
    """Score no-spill behavior without making one tiny spill day a hard zero."""
    spill_af = _finite_float(metrics.get("total_spill_af"), float("nan"))
    spill_days = _finite_float(metrics.get("frac_days_spilling"), float("nan"))
    has_spill_data = np.isfinite(spill_af) or np.isfinite(spill_days)

    if has_spill_data:
        if not np.isfinite(spill_af):
            spill_af = 0.0
        if not np.isfinite(spill_days):
            spill_days = 0.0

        # Priority Level 1 is "no spill", so this declines quickly. The volume
        # scale keeps small numerical/one-off spill from dominating the score,
        # while multi-hundred-kAF spill becomes a serious failure.
        volume_score = 1.0 / (1.0 + max(float(spill_af), 0.0) / 250_000.0)
        day_score = 1.0 - min(max(float(spill_days), 0.0) / 0.05, 1.0)
        return float(np.clip(min(volume_score, day_score), 0.0, 1.0))

    return _clip01(metrics.get("dam_safety_frac_days_within_storage_bounds"))


def compute_policy_composite_scores_from_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Compute composite screening scores from a metric dictionary.

    This helper is shared by live metric computation and report backfilling for
    older eval_metrics.csv files.
    """
    dam_safety_score = _dam_safety_no_spill_score(metrics)
    esa_score = _clip01(metrics.get("esa_min_flow_frac_days_met"))
    flooding_score = _clip01(metrics.get("flooding_frac_days_met"))

    # Normalize storage/hydropower to the practical range seen in this project,
    # while still keeping the components bounded and easy to reinterpret.
    storage_score = _clip01(_finite_float(metrics.get("storage_frac_of_max_possible"), 0.0) / 0.85)
    hydropower_score = _clip01(_finite_float(metrics.get("hydropower_frac_of_max_possible"), 0.0) / 0.35)

    niip_volume_score = _clip01(metrics.get("niip_annual_volume_frac_of_contract"))
    niip_shape_score = _clip01(metrics.get("niip_frac_days_demand_met_in_window"))
    niip_contract_score = niip_volume_score
    niip_behavior_score = float(0.65 * niip_volume_score + 0.35 * niip_shape_score)

    spr_scores: list[float] = []
    for thr, dur, target in _SPR_THRESHOLD_SPECS:
        key = f"spr_freq_years_meeting_{int(thr)}cfs_{int(dur)}d"
        achieved = _finite_float(metrics.get(key), float("nan"))
        if not np.isfinite(achieved):
            continue
        # A 25 percentage-point miss is a clear failure; smaller errors grade
        # smoothly. This intentionally penalizes over- and under-achievement.
        spr_scores.append(float(np.clip(1.0 - (abs(achieved - float(target)) / 0.25), 0.0, 1.0)))
    spr_score = float(np.mean(spr_scores)) if spr_scores else float("nan")
    if not np.isfinite(spr_score):
        spr_score = 0.0

    waste_frac = _finite_float(
        metrics.get(
            "controlled_sj_unattributed_frac_of_sj",
            metrics.get("controlled_sj_waste_frac_of_sj"),
        ),
        float("nan"),
    )
    if np.isfinite(waste_frac):
        efficiency_score = float(np.clip(1.0 - (waste_frac / 0.75), 0.0, 1.0))
    else:
        efficiency_score = float("nan")

    components = {
        "policy_dam_safety_score": dam_safety_score,
        "policy_esa_score": esa_score,
        "policy_spr_score": spr_score,
        "policy_niip_contract_score": niip_contract_score,
        "policy_niip_behavior_score": niip_behavior_score,
        "policy_niip_score": niip_behavior_score,
        "policy_storage_score": storage_score,
        "policy_hydropower_score": hydropower_score,
        "policy_flooding_score": flooding_score,
        "policy_efficiency_score": efficiency_score,
    }

    objective_alignment = _weighted_score(components, _POLICY_OBJECTIVE_ALIGNMENT_WEIGHTS)
    experiment_diagnostic = _weighted_score(components, _POLICY_EXPERIMENT_DIAGNOSTIC_WEIGHTS)

    return {
        **components,
        "policy_objective_alignment_score": float(objective_alignment),
        "policy_experiment_diagnostic_score": float(experiment_diagnostic),
        "policy_overall_score": float(objective_alignment),
        "policy_viability_score": float(experiment_diagnostic),
    }


def _metric_policy_composite_scores(df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for fn in (
        _metric_esa_min_flow_frac_days_met,
        _metric_flooding_frac_days_met,
        _metric_spring_peak_release,
        _metric_hydropower_frac_of_max_possible,
        _metric_storage_frac_of_max_possible,
        _metric_niip_delivery_and_volume,
        _metric_operational_diagnostics,
    ):
        try:
            metrics.update(fn(df))
        except Exception:
            pass

    waste = _controlled_sj_waste_stats(df)
    metrics.update(waste)
    return {
        **waste,
        **compute_policy_composite_scores_from_metrics(metrics),
    }


def compute_policy_composite_metrics_from_rollout(df_test: pd.DataFrame) -> dict[str, float]:
    """Compute policy score and controlled-SJ unattributed-release metrics."""
    return _metric_policy_composite_scores(df_test)




def compute_historic_summary_metrics(df_eval: pd.DataFrame) -> dict[str, float]:
    """Compute a compact historic benchmark row from an eval rollout dataframe.

    The returned metrics are aligned with the workbook summary columns and use
    historic data only (no agent-controlled releases).
    """
    out: dict[str, float] = {
        "dam_safety_frac_days_within_storage_bounds": float("nan"),
        "esa_min_flow_frac_days_met": float("nan"),
        "flooding_frac_days_met": float("nan"),
        "spr_curve_mean_abs_error_cfs": float("nan"),
        "spr_curve_frac_days_within_500cfs": float("nan"),
        "hydropower_frac_of_max_possible": float("nan"),
        "storage_frac_of_max_possible": float("nan"),
        "niip_frac_days_demand_met_in_window": float("nan"),
        "niip_annual_volume_frac_of_contract": float("nan"),
        "niip_mean_abs_daily_error_cfs": float("nan"),
        "controlled_sj_waste_af": float("nan"),
        "controlled_sj_waste_frac_of_sj": float("nan"),
        "controlled_sj_unattributed_af": float("nan"),
        "controlled_sj_unattributed_frac_of_sj": float("nan"),
        "frac_days_spilling": 0.0,
        "total_spill_af": 0.0,
        "policy_objective_alignment_score": float("nan"),
        "policy_experiment_diagnostic_score": float("nan"),
        "policy_overall_score": float("nan"),
        "policy_viability_score": float("nan"),
        "policy_efficiency_score": float("nan"),
        "policy_dam_safety_score": float("nan"),
        "policy_niip_contract_score": float("nan"),
        "policy_niip_behavior_score": float("nan"),
    }
    for thr, dur, _ in _SPR_THRESHOLD_SPECS:
        out[f"spr_freq_years_meeting_{int(thr)}cfs_{int(dur)}d"] = float("nan")

    if df_eval is None or df_eval.empty:
        return out

    # Storage against the same bounds used for the agent summaries.
    if "storage_hist_af" in df_eval.columns:
        low_af, high_af = _resolve_operating_storage_bounds(df_eval)
        s = df_eval["storage_hist_af"].astype(float)
        valid = s.notna()
        within = _ge_threshold(s, low_af) & _le_threshold(s, high_af)
        out["dam_safety_frac_days_within_storage_bounds"] = (
            float(within[valid].mean()) if bool(valid.any()) else float("nan")
        )
        max_storage_series = pd.to_numeric(df_eval.get("max_storage_af", float(high_af)), errors="coerce")
        if not isinstance(max_storage_series, pd.Series):
            max_storage_series = pd.Series(float(high_af), index=df_eval.index, dtype=float)
        valid_storage = s.notna() & max_storage_series.notna() & (max_storage_series > 0.0)
        denom_storage = float(np.nansum(max_storage_series[valid_storage].to_numpy(dtype=float)))
        if denom_storage > 0.0:
            out["storage_frac_of_max_possible"] = float(
                np.nansum(s[valid_storage].to_numpy(dtype=float)) / denom_storage
            )

    # ESA min flow on historical releases.
    if {"animas_farmington_q_cfs", "release_cfs"}.issubset(df_eval.columns):
        animas = df_eval["animas_farmington_q_cfs"].astype(float)
        release = df_eval["release_cfs"].astype(float)
        valid = animas.notna() & release.notna()
        met = _ge_threshold(animas + release, 500.0)
        out["esa_min_flow_frac_days_met"] = (
            float(met[valid].mean()) if bool(valid.any()) else float("nan")
        )
        out.update(_controlled_sj_waste_stats(df_eval, release_col="release_cfs"))

    # Flooding uses the observed Archuleta and Bluff gages. Legacy rollouts lack
    # the Archuleta column, so their reported reservoir release is used as a
    # close outlet-flow fallback; new rollouts carry the observed gage series.
    q0_col = (
        "sj_archuleta_q_cfs"
        if "sj_archuleta_q_cfs" in df_eval.columns
        else "release_cfs"
        if "release_cfs" in df_eval.columns
        else None
    )
    if q0_col is not None and "sj_bluff_q_cfs" in df_eval.columns:
        q0 = df_eval[q0_col].astype(float)
        qlag2 = df_eval["sj_bluff_q_cfs"].astype(float)
        safe_same = _lt_threshold(q0, 5000.0)
        safe_lag2 = _lt_threshold(qlag2, 12000.0)
        valid = q0.notna() & qlag2.notna()
        if bool(valid.any()):
            out["flooding_frac_days_met"] = float(
                (safe_same[valid] & safe_lag2[valid]).mean()
            )

    # SPR threshold frequencies use controlled Farmington flow only, matching
    # agent metrics: Animas @ Farmington + controlled Navajo mainstem release.
    if {"animas_farmington_q_cfs", "release_cfs"}.issubset(df_eval.columns):
        animas = pd.to_numeric(df_eval["animas_farmington_q_cfs"], errors="coerce")
        release = pd.to_numeric(df_eval["release_cfs"], errors="coerce")
        controlled_farmington = animas + release
        curve = SpringPeakReleaseCurve()
        target = curve.targets_for_date_index(df_eval.index).astype(float)
        spr_mask = target > 0.0
        if bool(spr_mask.any()):
            err = controlled_farmington - target
            valid_spr = spr_mask & controlled_farmington.notna()
            out["spr_curve_mean_abs_error_cfs"] = float(err.abs()[valid_spr].mean())
            out["spr_curve_frac_days_within_500cfs"] = float(
                _le_threshold(err.abs()[valid_spr], 500.0).mean()
            )

            idx = df_eval.index
            complete_years = _complete_spr_season_years(idx, curve, animas, release)
            wy = idx.year + (idx.month >= 10).astype(int)
            wy_series = pd.Series(wy, index=idx, name="wy")
            for thr, dur_days, _target_freq in _SPR_THRESHOLD_SPECS:
                thr_i = int(thr)
                dur_i = int(dur_days)
                met_flags: list[bool] = []
                for _wy_val, g_idx in wy_series.groupby(wy_series).groups.items():
                    if int(_wy_val) not in complete_years:
                        continue
                    g_idx = pd.DatetimeIndex(g_idx)
                    g_spr = spr_mask.loc[g_idx]
                    spr_days = int(g_spr.sum())
                    if spr_days <= 0:
                        continue
                    b = g_spr & _ge_threshold(controlled_farmington.loc[g_idx], thr)
                    met_flags.append(bool(int(b.sum()) >= int(dur_days)))
                if met_flags:
                    out[f"spr_freq_years_meeting_{thr_i}cfs_{dur_i}d"] = float(
                        np.mean(met_flags)
                    )

    # Historic hydropower relative to a practical plant-capacity maximum.
    if "hydro_hist_mwh" in df_eval.columns:
        hydro_hist = pd.to_numeric(df_eval["hydro_hist_mwh"], errors="coerce")
        n_valid_hydro = int(hydro_hist.notna().sum())
        denom_hydro = 768.0 * float(n_valid_hydro)
        if denom_hydro > 0.0:
            out["hydropower_frac_of_max_possible"] = float(
                np.nansum(hydro_hist.to_numpy(dtype=float)) / denom_hydro
            )

    # NIIP historic delivery benchmark from the observed NIIP diversion record.
    niip_hist = _load_niip_historic_delivery_series()
    if niip_hist is not None:
        eval_index = pd.DatetimeIndex(df_eval.index).normalize()
        delivery_hist = niip_hist.reindex(eval_index)
        # In the absence of an independent NIIP demand record, use observed
        # historic delivery as the benchmark target. This makes the historical
        # row answer "how well did actual operations match actual delivery?"
        demand_hist = delivery_hist.copy()
        if demand_hist is not None:
            out.update(
                _compute_niip_delivery_and_volume_from_series(
                    index=eval_index,
                    demand=demand_hist,
                    delivery=delivery_hist,
                )
            )

    out.update(compute_policy_composite_scores_from_metrics(out))
    return out


# -----------------------------------------------------------------------------
# Registry + groups
# -----------------------------------------------------------------------------


METRIC_REGISTRY: dict[str, MetricSpec] = {
    # Rewards
    "rewards_summary": MetricSpec(func=_metric_rewards_summary, requires=("reward",)),
    "reward_components_summary": MetricSpec(func=_metric_reward_components_summary, requires=()),

    # Objective-aligned scoreboard metrics
    "dam_safety": MetricSpec(func=_metric_dam_safety_frac_days_within_storage_bounds, requires=()),
    "esa_min_flow": MetricSpec(func=_metric_esa_min_flow_frac_days_met, requires=()),
    "flooding": MetricSpec(func=_metric_flooding_frac_days_met, requires=()),
    "spring_peak_release": MetricSpec(func=_metric_spring_peak_release_scoreboard, requires=("release_sj_main_cfs", "animas_farmington_q_cfs")),
    "spring_peak_release_detail": MetricSpec(func=_metric_spring_peak_release, requires=("release_sj_main_cfs", "animas_farmington_q_cfs")),
    "hydropower": MetricSpec(func=_metric_hydropower_frac_of_max_possible, requires=()),
    "storage_relative": MetricSpec(func=_metric_storage_frac_of_max_possible, requires=()),
    "niip": MetricSpec(func=_metric_niip_delivery_and_volume, requires=()),
    "policy_composite": MetricSpec(func=_metric_policy_composite_scores, requires=()),

    # Optional extra objective diagnostics
    "dam_safety_detail": MetricSpec(func=_metric_dam_safety_storage_detail, requires=()),

    # Actions / operations (diagnostic)
    "action_saturation": MetricSpec(func=_metric_action_saturation, requires=()),
    "release_constraint_binding": MetricSpec(func=_metric_release_constraint_binding, requires=()),
    "operational_diagnostics": MetricSpec(func=_metric_operational_diagnostics, requires=()),
}


METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    # Compact experiment scoreboard: objective-aligned comparison metrics only.
    "core": (
        "dam_safety",
        "esa_min_flow",
        "flooding",
        "spring_peak_release",
        "hydropower",
        "storage_relative",
        "niip",
        "policy_composite",
    ),
    "rewards": ("rewards_summary", "reward_components_summary"),
    "objectives": ("dam_safety", "esa_min_flow", "flooding", "spring_peak_release", "hydropower", "storage_relative", "niip", "policy_composite"),
    "dam_safety_detail": ("dam_safety_detail",),
    "actions": ("action_saturation", "release_constraint_binding"),
    "diagnostics": ("dam_safety_detail", "action_saturation", "release_constraint_binding", "operational_diagnostics"),
    "spr": ("spring_peak_release", "spring_peak_release_detail"),
    "all": tuple(),
}
