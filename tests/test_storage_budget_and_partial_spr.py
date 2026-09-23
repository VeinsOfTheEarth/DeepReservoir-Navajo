"""Corrected storage-budget and random-episode SPR initialization controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deepreservoir.define_env.spring_peak_release_curve import SpringPeakReleaseCurve
from deepreservoir.drl import model, selected_policy
from deepreservoir.drl.environs import NavajoReservoirEnv
from deepreservoir.drl.rewards import PRACTICAL_MIN_STORAGE_AF


SPR_REWARD_KEY = (
    "esa_spring_peak_release."
    "farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong"
)


@pytest.fixture(scope="module")
def model_data() -> dict[str, object]:
    return model.load_all_model_data()


def _spring_env(
    model_data: dict[str, object],
    *,
    mask_incomplete_initial_spr: bool,
) -> NavajoReservoirEnv:
    raw_all = model_data["raw"]
    norm_all = model_data["norm"]
    raw = raw_all.loc["2014-05-10":"2014-05-12"].copy()
    raw["animas_farmington_q_cfs"] = 6_000.0
    env = model.make_env(
        data_raw=raw,
        data_norm=norm_all.loc[raw.index].copy(),
        norm_stats=model_data["norm_stats"],
        reward_spec_str=selected_policy.SELECTED_REWARD_SPEC,
        episode_length=None,
        is_eval=True,
        storage_budget_target_frac_of_max=0.875,
        mask_incomplete_initial_spr=mask_incomplete_initial_spr,
    )
    assert isinstance(env, NavajoReservoirEnv)
    return env


def test_storage_budget_uses_configured_policy_target() -> None:
    env = NavajoReservoirEnv.__new__(NavajoReservoirEnv)
    env.deadpool_storage_af = 0.0
    env.max_storage_af = 1_647_936.17
    env.storage_af = 0.780 * env.max_storage_af

    env.storage_budget_target_frac_of_max = 0.780
    assert env._storage_budget_obs_value() == pytest.approx(1.0)

    env.storage_budget_target_frac_of_max = 0.875
    expected = (env.storage_af - PRACTICAL_MIN_STORAGE_AF) / (
        0.875 * env.max_storage_af - PRACTICAL_MIN_STORAGE_AF
    )
    assert env._storage_budget_obs_value() == pytest.approx(expected)
    env.storage_af = 0.875 * env.max_storage_af
    assert env._storage_budget_obs_value() == pytest.approx(1.0)


@pytest.mark.parametrize("target", [float("nan"), 1.01, 0.20])
def test_invalid_storage_budget_target_is_rejected(
    model_data: dict[str, object], target: float
) -> None:
    raw_all = model_data["raw"]
    norm_all = model_data["norm"]
    raw = raw_all.loc["2014-05-10":"2014-05-12"]
    with pytest.raises(ValueError, match="storage_budget_target_frac_of_max"):
        model.make_env(
            data_raw=raw,
            data_norm=norm_all.loc[raw.index],
            norm_stats=model_data["norm_stats"],
            reward_spec_str=selected_policy.SELECTED_REWARD_SPEC,
            episode_length=None,
            is_eval=True,
            storage_budget_target_frac_of_max=target,
        )


@pytest.mark.parametrize(
    ("date", "expected_masked"),
    [
        ("2020-01-01", False),
        ("2020-05-09", False),
        ("2020-05-10", True),
        ("2020-06-25", True),
        ("2020-06-26", False),
    ],
)
def test_incomplete_initial_spr_boundaries(date: str, expected_masked: bool) -> None:
    env = NavajoReservoirEnv.__new__(NavajoReservoirEnv)
    env.mask_incomplete_initial_spr = True
    env.date_index = pd.DatetimeIndex([date])
    env.n_steps = 1
    curve = SpringPeakReleaseCurve()
    env._spring_window_active_daily = (
        curve.targets_for_date_index(env.date_index).to_numpy(dtype=float) > 0.0
    )
    env._water_year = (
        env.date_index.year + (env.date_index.month >= 10)
    ).to_numpy(dtype=int)

    env._reset_initial_partial_spr_mask(idx=0)
    assert env._spr_initial_partial_season_masked(0) is expected_masked
    assert env._spr_window_active_for_episode(0) is bool(
        env._spring_window_active_daily[0] and not expected_masked
    )


def test_partial_spring_mask_clears_for_next_water_year() -> None:
    env = NavajoReservoirEnv.__new__(NavajoReservoirEnv)
    env.mask_incomplete_initial_spr = True
    env.date_index = pd.DatetimeIndex(["2020-05-10", "2021-05-09"])
    env.n_steps = 2
    env._spring_window_active_daily = np.array([True, True], dtype=bool)
    env._water_year = np.array([2020, 2021], dtype=int)

    env._reset_initial_partial_spr_mask(idx=0)
    assert env._spr_initial_partial_season_masked(0) is True
    assert env._spr_window_active_for_episode(1) is True

    env._reset_initial_partial_spr_mask(idx=1)
    assert env._spr_initial_partial_wy is None
    assert env._spr_window_active_for_episode(1) is True


def test_corrected_mode_masks_partial_spring_control_reward_and_state(
    model_data: dict[str, object],
) -> None:
    env = _spring_env(model_data, mask_incomplete_initial_spr=True)
    obs, _ = env.reset()
    obs_by_name = dict(zip(env.obs_cols, obs, strict=True))

    assert obs_by_name["animas_spr_frac"] == pytest.approx(0.6)
    assert all(
        float(value) == 0.0
        for name, value in obs_by_name.items()
        if name.startswith("spr_")
    )

    _, _, _, _, info = env.step(np.array([-1.0, -1.0, 1.0, -1.0]))
    assert info["spr_calendar_window_active"] is True
    assert info["spr_proxy_window_active"] is False
    assert info["spr_initial_partial_season_masked"] is True
    assert info["spr_reward_eligible"] is False
    assert info["spr_proxy_mask_applied"] is True
    assert info["spr_proxy_mask_mode"] == "incomplete_initial_spring"
    assert info["spr_proxy_raw_target_cfs"] == 10_000.0
    assert info["spr_proxy_target_cfs"] == 0.0
    assert info["release_sj_main_cfs"] == pytest.approx(0.0)
    assert info["spr_useful_unrequested_sj_release_cfs"] == pytest.approx(0.0)
    assert info["reward_components_step"][SPR_REWARD_KEY] == pytest.approx(0.0)
    assert all(count == 0 for count in env._spr_days_so_far_by_spec.values())
    assert info["storage_budget_target_frac_of_max"] == pytest.approx(0.875)
    assert info["storage_budget_target_af"] == pytest.approx(
        0.875 * env.max_storage_af
    )


def test_legacy_mode_keeps_partial_spring_behavior(
    model_data: dict[str, object],
) -> None:
    env = _spring_env(model_data, mask_incomplete_initial_spr=False)
    env.reset()
    _, _, _, _, info = env.step(np.array([-1.0, -1.0, 1.0, -1.0]))

    assert info["spr_calendar_window_active"] is True
    assert info["spr_proxy_window_active"] is True
    assert info["spr_initial_partial_season_masked"] is False
    assert info["spr_reward_eligible"] is True
    assert info["spr_proxy_mask_applied"] is False
    assert info["spr_proxy_target_cfs"] == 10_000.0
    assert info["release_sj_main_cfs"] == pytest.approx(4_000.0)
    assert info["reward_components_step"][SPR_REWARD_KEY] > 0.0
    assert all(count == 1 for count in env._spr_days_so_far_by_spec.values())
