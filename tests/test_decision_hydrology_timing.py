"""Decision-time hydrology and realized daily outcomes use distinct dates."""

import numpy as np
import pandas as pd
import pytest

from deepreservoir.drl import model, selected_policy
from deepreservoir.drl.environs import (
    CFS_TO_AF_PER_DAY,
    compute_esa_required_release_frac_signal,
    normalize_decision_hydrology_timing,
)
from deepreservoir.drl.rewards import (
    RewardContext,
    _spr_actionproxy_highest_justified_terms,
)


def _three_day_environment(*, timing: str):
    data = model.load_all_model_data()
    raw = data["raw"].loc["2014-05-09":"2014-05-11"].copy()
    raw.loc[pd.Timestamp("2014-05-09"), [
        "animas_farmington_q_cfs", "inflow_cfs", "evap_af",
    ]] = [100.0, 200.0, 100.0]
    raw.loc[pd.Timestamp("2014-05-10"), [
        "animas_farmington_q_cfs", "inflow_cfs", "evap_af",
    ]] = [9_000.0, 300.0, 110.0]

    prior = data["raw"].loc[pd.Timestamp("2014-05-08")].copy()
    prior.loc[["animas_farmington_q_cfs", "inflow_cfs", "evap_af"]] = [
        6_000.0, 1_000.0, 10.0,
    ]
    env = model.make_env(
        data_raw=raw,
        data_norm=data["norm"].loc[raw.index],
        norm_stats=data["norm_stats"],
        reward_spec_str=selected_policy.SELECTED_REWARD_SPEC,
        episode_length=None,
        is_eval=True,
        decision_hydrology_timing=timing,
        prior_day_hydrology=prior,
    )
    return env, raw, prior


def test_timing_mode_validation_and_legacy_default() -> None:
    assert normalize_decision_hydrology_timing(None) == "same_day"
    assert normalize_decision_hydrology_timing("previous_day") == "previous_day"
    with pytest.raises(ValueError, match="decision_hydrology_timing"):
        normalize_decision_hydrology_timing("forecast")


def test_previous_day_arrays_include_window_context_and_preserve_calendar() -> None:
    env, raw, prior = _three_day_environment(timing="previous_day")
    try:
        np.testing.assert_allclose(
            env._decision_animas_farmington_q_cfs[:2],
            [prior["animas_farmington_q_cfs"], raw.iloc[0]["animas_farmington_q_cfs"]],
        )
        np.testing.assert_allclose(
            env._decision_inflow_cfs[:2],
            [prior["inflow_cfs"], raw.iloc[0]["inflow_cfs"]],
        )
        np.testing.assert_allclose(
            env._decision_evap_af[:2],
            [prior["evap_af"], raw.iloc[0]["evap_af"]],
        )

        obs, _ = env.reset(options={"initial_storage_af": env.max_storage_af})
        assert env._spring_window_active_daily[0]
        assert obs[2] == pytest.approx(env._niip_historic_demand_cfs[0] / 2_500.0)
        assert obs[3] == pytest.approx(0.0)
        assert obs[4] == pytest.approx(0.6)
        assert obs[5] == pytest.approx(0.8)

        excess_af = prior["inflow_cfs"] * CFS_TO_AF_PER_DAY - prior["evap_af"]
        excess_cfs = excess_af / CFS_TO_AF_PER_DAY
        assert obs[18] == pytest.approx(excess_cfs / 7_500.0)
        assert obs[19] == pytest.approx(excess_cfs / 5_000.0)
    finally:
        env.close()


def test_requests_use_previous_day_but_transition_and_success_use_current_day() -> None:
    env, raw, _ = _three_day_environment(timing="previous_day")
    try:
        env.reset(options={"initial_storage_af": 1_000_000.0})
        storage_before = float(env.storage_af)
        action = np.array([-1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert info["decision_animas_farmington_q_cfs"] == pytest.approx(6_000.0)
        assert info["animas_farmington_q_cfs"] == pytest.approx(100.0)
        assert info["spr_proxy_controller_need_cfs"] == pytest.approx(4_000.0)
        assert info["spr_proxy_actual_bridge_need_cfs"] == pytest.approx(5_000.0)
        assert info["spr_proxy_target_reachable_at_decision"] is True
        assert info["spr_proxy_target_reachable"] is False
        assert info["spr_proxy_target_hit"] is False
        assert info["esa_floor_decision_required_cfs"] == pytest.approx(0.0)
        assert info["esa_floor_required_cfs"] == pytest.approx(400.0)
        assert info["esa_floor_met_after_release"] is True

        actual = raw.iloc[0]
        expected_storage = (
            storage_before
            + actual["inflow_cfs"] * CFS_TO_AF_PER_DAY
            - actual["evap_af"]
            - (info["release_sj_main_cfs"] + info["release_niip_cfs"])
            * CFS_TO_AF_PER_DAY
            - info["spill_af"]
        )
        assert env.storage_af == pytest.approx(expected_storage)
        assert env._spr_days_so_far_by_spec[(10_000.0, 5)] == 0
        assert env._spr_days_so_far_by_spec[(2_500.0, 10)] == 1
    finally:
        env.close()


def test_spr_preferred_target_uses_decision_animas_and_outcome_uses_realized_animas() -> None:
    info = {
        "action_mode": "esa_base_spr_proxy_4d_max",
        "decision_animas_farmington_q_cfs": 6_000.0,
        "animas_farmington_q_cfs": 100.0,
        "release_sj_main_cfs": 4_000.0,
        "max_release_sj_main_cfs": 5_000.0,
        "spr_proxy_window_active": True,
        "spr_calendar_days_left": 48,
    }
    for threshold, duration in ((10_000, 5), (8_000, 10), (5_000, 21), (2_500, 10)):
        info[f"spr_days_so_far_{threshold}cfs_{duration}d"] = 0
        info[f"spr_need_more_{threshold}cfs_{duration}d"] = True
    ctx = RewardContext(
        t=0,
        date=pd.Timestamp("2014-05-09"),
        obs=np.zeros(1),
        action=np.zeros(1),
        next_obs=np.zeros(1),
        info=info,
    )

    terms = _spr_actionproxy_highest_justified_terms(
        ctx,
        frequency_gate_mode="none",
        respect_calendar=True,
    )
    assert terms is not None
    assert terms.target_cfs == 10_000.0
    assert terms.decision_animas_cfs == 6_000.0
    assert terms.max_reachable_farmington_cfs == 11_000.0
    assert terms.animas_cfs == 100.0
    assert terms.farmington_total_cfs == 4_100.0


def test_same_day_mode_ignores_prior_context() -> None:
    env, raw, _ = _three_day_environment(timing="same_day")
    try:
        np.testing.assert_allclose(
            env._decision_animas_farmington_q_cfs,
            raw["animas_farmington_q_cfs"].to_numpy(),
        )
        expected = compute_esa_required_release_frac_signal(
            animas_cfs=raw["animas_farmington_q_cfs"].to_numpy(),
            max_release_sj_main_cfs=5_000.0,
        )
        np.testing.assert_allclose(env._obs_norm_arrays["esa_required_release_frac"], expected)
    finally:
        env.close()


def test_window_context_helper_requires_the_previous_calendar_day() -> None:
    frame = pd.DataFrame(
        {"inflow_cfs": [1.0, 2.0]},
        index=pd.to_datetime(["2013-12-31", "2014-01-01"]),
    )
    prior = model._previous_calendar_hydrology_row(
        frame,
        first_date=pd.Timestamp("2014-01-01"),
    )
    assert prior is not None and prior.name == pd.Timestamp("2013-12-31")
    assert model._previous_calendar_hydrology_row(
        frame.iloc[1:],
        first_date=pd.Timestamp("2014-01-01"),
    ) is None
