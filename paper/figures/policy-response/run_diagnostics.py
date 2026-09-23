"""Run paired policy-response and saved-action-schedule inflow diagnostics.

The saved-schedule baseline applies the selected policy's deterministic action
sequence from the unperturbed evaluation to every altered inflow series. The
environment still enforces the same controllers and physical limits, so paired
differences isolate the effect of updating policy actions as conditions change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepreservoir.drl import helpers  # noqa: E402
from deepreservoir.drl import metrics as drl_metrics  # noqa: E402
from deepreservoir.drl import model  # noqa: E402
from deepreservoir.drl import selected_policy  # noqa: E402


DEFAULT_WINDOW_START = "2014-01-01"
DEFAULT_WINDOW_END = "2024-08-17"
DEFAULT_OUTPUT_DIR = HERE / "data"
SCALE_VALUES = tuple(float(value) for value in np.round(np.arange(0.5, 1.51, 0.1), 10))
CFS_DAY_TO_AF = 86_400.0 / 43_560.0

METRIC_COLUMNS = (
    "storage_frac_of_max_possible",
    "esa_min_flow_frac_days_met",
    "flooding_frac_days_met",
    "policy_spr_score",
    "hydropower_frac_of_max_possible",
    "niip_annual_volume_frac_of_contract",
)

ACTION_LABELS = (
    "ESA/baseflow",
    "NIIP delivery",
    "SPR target",
    "Discretionary release",
)


class ActionReplayAgent:
    """Small duck-typed agent that returns a prescribed action sequence."""

    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)
        if self.actions.ndim != 2:
            raise ValueError("Replay actions must have shape (steps, actions).")
        self.position = 0

    def predict(self, _obs: np.ndarray, deterministic: bool = True):
        del deterministic
        if self.position >= len(self.actions):
            raise IndexError("Replay action sequence ended before the rollout.")
        action = self.actions[self.position].copy()
        self.position += 1
        return action, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_WINDOW_START)
    parser.add_argument("--end", default=DEFAULT_WINDOW_END)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _load_window(
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series | None]:
    all_data = model.load_all_model_data(
        storage_datum_mode=selected_policy.SELECTED_STORAGE_DATUM_MODE
    )
    raw, _ = helpers.slice_by_window(
        all_data["raw"],
        start_token=start,
        end_token=end,
        label="policy_response_inflow",
    )
    norm = all_data["norm"].loc[raw.index]
    norm_stats, _ = model._storage_norm_stats_for_evaluation(  # noqa: SLF001
        model_path=selected_policy.SELECTED_MODEL_PATH,
        full_record_norm_stats=all_data["norm_stats"],
        mode=selected_policy.SELECTED_STORAGE_NORMALIZATION,
    )
    prior_day = model._previous_calendar_hydrology_row(  # noqa: SLF001
        all_data["raw"], first_date=raw.index[0]
    )
    return raw, norm, norm_stats, prior_day


def _scaled_inflow_data(
    raw_nominal: pd.DataFrame,
    norm_nominal: pd.DataFrame,
    norm_stats: pd.DataFrame,
    scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw_nominal.copy()
    raw["inflow_cfs"] = (
        pd.to_numeric(raw["inflow_cfs"], errors="raise").astype(float) * float(scale)
    )

    norm = norm_nominal.copy()
    mean = float(norm_stats.loc["inflow_cfs", "mean"])
    std = float(norm_stats.loc["inflow_cfs", "std"])
    if std == 0.0:
        std = 1.0
    norm["inflow_cfs"] = (raw["inflow_cfs"] - mean) / std
    return raw, norm


def _make_eval_env(
    raw: pd.DataFrame,
    norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    prior_day: pd.Series | None,
):
    return model.make_env(
        data_raw=raw,
        data_norm=norm,
        norm_stats=norm_stats,
        reward_spec_str=selected_policy.SELECTED_REWARD_SPEC,
        episode_length=None,
        is_eval=True,
        max_release_sj_main_cfs=5000.0,
        max_release_niip_cfs=2500.0,
        obs_context=selected_policy.SELECTED_OBS_CONTEXT,
        action_scaling="linear",
        action_mode=selected_policy.SELECTED_ACTION_MODE,
        reward_balancing="none",
        esa_min_flow_floor=False,
        spr_proxy_priority_release=True,
        spr_proxy_owns_sj_window=False,
        spr_advice_mode=selected_policy.SELECTED_SPR_ADVICE_MODE,
        decision_hydrology_timing=selected_policy.SELECTED_DECISION_HYDROLOGY_TIMING,
        niip_fallback_mode=selected_policy.SELECTED_NIIP_FALLBACK_MODE,
        storage_datum_mode=selected_policy.SELECTED_STORAGE_DATUM_MODE,
        storage_normalization=selected_policy.SELECTED_STORAGE_NORMALIZATION,
        storage_budget_target_frac_of_max=selected_policy.SELECTED_STORAGE_BUDGET_TARGET_FRAC,
        mask_incomplete_initial_spr=selected_policy.SELECTED_MASK_INCOMPLETE_INITIAL_SPR,
        prior_day_hydrology=prior_day,
    )


def _run_adaptive(
    agent: PPO,
    raw: pd.DataFrame,
    norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    prior_day: pd.Series | None,
) -> pd.DataFrame:
    env = _make_eval_env(raw, norm, norm_stats, prior_day)
    return model._run_rollout_env(agent=agent, eval_env=env)  # noqa: SLF001


def _run_replay(
    nominal_actions: np.ndarray,
    raw: pd.DataFrame,
    norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    prior_day: pd.Series | None,
) -> pd.DataFrame:
    env = _make_eval_env(raw, norm, norm_stats, prior_day)
    replay_agent = ActionReplayAgent(nominal_actions)
    return model._run_rollout_env(agent=replay_agent, eval_env=env)  # type: ignore[arg-type]  # noqa: SLF001


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _sum_cfs_as_af(df: pd.DataFrame, *columns: str) -> float:
    total_cfs = sum(
        (_numeric(df, column) for column in columns),
        start=pd.Series(0.0, index=df.index, dtype=float),
    )
    return float(total_cfs.clip(lower=0.0).sum()) * CFS_DAY_TO_AF


def _rollout_summary(
    rollout: pd.DataFrame,
    *,
    mode: str,
    scale: float,
) -> dict[str, float | str]:
    metric_values = drl_metrics.compute_metrics(
        rollout,
        which="core",
        validate=True,
    ).iloc[0].to_dict()
    spill = _numeric(rollout, "spill_af").clip(lower=0.0)
    spill_days = float((spill > 1.0e-9).sum())

    niip_release_af = _sum_cfs_as_af(rollout, "release_niip_cfs")
    spr_release_af = _sum_cfs_as_af(
        rollout,
        "spr_attributed_sj_release_cfs",
        "spr_useful_unrequested_sj_release_cfs",
    )
    esa_release_af = _sum_cfs_as_af(rollout, "esa_attributed_sj_release_cfs")
    discretionary_sj_release_af = _sum_cfs_as_af(
        rollout,
        "discretionary_attributed_sj_release_cfs",
    )
    controlled_sj_release_af = _sum_cfs_as_af(rollout, "release_sj_main_cfs")
    other_sj_release_af = controlled_sj_release_af - (
        spr_release_af + esa_release_af + discretionary_sj_release_af
    )

    row: dict[str, float | str] = {
        "mode": mode,
        "inflow_scale": float(scale),
        "inflow_change_pct": 100.0 * (float(scale) - 1.0),
        "n_days": float(len(rollout)),
        "spill_days": spill_days,
        "spill_frac_days": spill_days / float(len(rollout)),
        "spill_free_days": 1.0 - spill_days / float(len(rollout)),
        "total_spill_af": float(spill.sum()),
        "niip_release_af": niip_release_af,
        "spr_release_af": spr_release_af,
        "esa_release_af": esa_release_af,
        "discretionary_sj_release_af": discretionary_sj_release_af,
        "other_sj_release_af": other_sj_release_af,
        "controlled_sj_release_af": controlled_sj_release_af,
        "ending_storage_af": float(_numeric(rollout, "storage_agent_af_end").iloc[-1]),
        "mean_storage_af": float(_numeric(rollout, "storage_agent_af_end").mean()),
    }
    for column in METRIC_COLUMNS:
        row[column] = float(metric_values[column])
    for action_index, action_label in enumerate(ACTION_LABELS):
        values = _numeric(rollout, f"action_{action_index}")
        row[f"action_{action_index}_label"] = action_label
        row[f"action_{action_index}_mean"] = float(values.mean())
    return row


def _paired_summary(
    adaptive: pd.DataFrame,
    replay: pd.DataFrame,
    *,
    scale: float,
) -> dict[str, float]:
    if not adaptive.index.equals(replay.index):
        raise ValueError("Adaptive and replay rollouts do not share the same dates.")

    row: dict[str, float] = {
        "inflow_scale": float(scale),
        "inflow_change_pct": 100.0 * (float(scale) - 1.0),
    }
    for action_index in range(len(ACTION_LABELS)):
        delta = _numeric(adaptive, f"action_{action_index}") - _numeric(
            replay, f"action_{action_index}"
        )
        row[f"action_{action_index}_mean_delta"] = float(delta.mean())
        row[f"action_{action_index}_mean_abs_delta"] = float(delta.abs().mean())

    adaptive_summary = _rollout_summary(adaptive, mode="adaptive", scale=scale)
    replay_summary = _rollout_summary(replay, mode="replay", scale=scale)
    volume_columns = (
        "niip_release_af",
        "spr_release_af",
        "esa_release_af",
        "discretionary_sj_release_af",
        "other_sj_release_af",
        "total_spill_af",
        "ending_storage_af",
        "mean_storage_af",
    )
    for column in volume_columns:
        row[f"delta_{column}"] = float(adaptive_summary[column]) - float(
            replay_summary[column]
        )

    release_and_storage = (
        "delta_niip_release_af",
        "delta_spr_release_af",
        "delta_esa_release_af",
        "delta_discretionary_sj_release_af",
        "delta_other_sj_release_af",
        "delta_total_spill_af",
        "delta_ending_storage_af",
    )
    row["water_balance_residual_af"] = float(
        sum(row[column] for column in release_and_storage)
    )
    return row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_nominal, norm_nominal, norm_stats, prior_day = _load_window(args.start, args.end)
    print("[policy-response] loading selected policy")
    agent = PPO.load(selected_policy.SELECTED_MODEL_PATH, device=args.device)

    print("[policy-response] building the unperturbed action schedule")
    nominal = _run_adaptive(agent, raw_nominal, norm_nominal, norm_stats, prior_day)
    action_columns = [f"action_{index}" for index in range(len(ACTION_LABELS))]
    nominal_actions = nominal[action_columns].to_numpy(dtype=np.float32)
    historic_metrics = drl_metrics.compute_historic_summary_metrics(nominal)

    rollout_rows: list[dict[str, float | str]] = []
    paired_rows: list[dict[str, float]] = []
    nominal_replay_max_storage_error_af = float("nan")
    nominal_replay_max_action_error = float("nan")

    for scale in SCALE_VALUES:
        change_pct = 100.0 * (scale - 1.0)
        print(f"[policy-response] inflow {change_pct:+.0f}%")
        raw, norm = _scaled_inflow_data(raw_nominal, norm_nominal, norm_stats, scale)
        scenario_prior_day = None if prior_day is None else prior_day.copy()
        if scenario_prior_day is not None:
            scenario_prior_day["inflow_cfs"] = (
                float(scenario_prior_day["inflow_cfs"]) * scale
            )
        adaptive = nominal if np.isclose(scale, 1.0) else _run_adaptive(
            agent, raw, norm, norm_stats, scenario_prior_day
        )
        replay = _run_replay(
            nominal_actions, raw, norm, norm_stats, scenario_prior_day
        )

        rollout_rows.append(_rollout_summary(adaptive, mode="adaptive", scale=scale))
        rollout_rows.append(_rollout_summary(replay, mode="replay", scale=scale))
        paired_rows.append(_paired_summary(adaptive, replay, scale=scale))

        if np.isclose(scale, 1.0):
            nominal_replay_max_storage_error_af = float(
                np.max(
                    np.abs(
                        _numeric(adaptive, "storage_agent_af_end").to_numpy()
                        - _numeric(replay, "storage_agent_af_end").to_numpy()
                    )
                )
            )
            nominal_replay_max_action_error = float(
                np.max(
                    np.abs(
                        adaptive[action_columns].to_numpy(dtype=float)
                        - replay[action_columns].to_numpy(dtype=float)
                    )
                )
            )

    rollout_summary = pd.DataFrame(rollout_rows).sort_values(
        ["inflow_scale", "mode"]
    )
    paired_summary = pd.DataFrame(paired_rows).sort_values("inflow_scale")
    rollout_summary.to_csv(
        output_dir / "rollout-summary.csv", index=False, float_format="%.10g"
    )
    paired_summary.to_csv(
        output_dir / "paired-differences.csv", index=False, float_format="%.10g"
    )

    metadata = {
        "selected_policy_model": str(
            selected_policy.SELECTED_MODEL_PATH.relative_to(REPO_ROOT)
        ),
        "selected_policy_model_sha256": _sha256(selected_policy.SELECTED_MODEL_PATH),
        "window_start": args.start,
        "window_end": args.end,
        "n_days": int(len(nominal)),
        "inflow_scales": list(SCALE_VALUES),
        "action_labels": list(ACTION_LABELS),
        "historic_metrics": {
            key: float(historic_metrics[key]) for key in METRIC_COLUMNS
        },
        "spill_free_days_historic_benchmark": 1.0,
        "action_replay_definition": (
            "The deterministic normalized action vector from each unperturbed evaluation day "
            "is used as the saved action schedule on the corresponding altered-inflow day; "
            "environment controllers and physical constraints remain active."
        ),
        "nominal_replay_max_storage_error_af": nominal_replay_max_storage_error_af,
        "nominal_replay_max_action_error": nominal_replay_max_action_error,
        "max_abs_water_balance_residual_af": float(
            paired_summary["water_balance_residual_af"].abs().max()
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[policy-response] wrote {output_dir}")
    print(
        "[policy-response] saved-schedule check: "
        f"storage={nominal_replay_max_storage_error_af:.6g} AF, "
        f"action={nominal_replay_max_action_error:.6g}"
    )
    print(
        "[policy-response] max paired water-balance residual: "
        f"{metadata['max_abs_water_balance_residual_af']:.6g} AF"
    )


if __name__ == "__main__":
    main()
