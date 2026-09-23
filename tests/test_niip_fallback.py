"""NIIP missing-day targets use an explicit, reproducible reference period."""

import numpy as np
import pandas as pd
import pytest

from deepreservoir.drl import model, selected_policy
from deepreservoir.drl.niip_targets import (
    NIIP_KNOWN_IN_RECORD_GAP_DATES,
    NIIP_TRAINING_REFERENCE_END,
    historic_niip_delivery_target_for_date,
    historic_niip_delivery_target_for_dates,
    historic_niip_month_day_fallback_lookup,
    load_historic_niip_delivery_series,
    normalize_niip_fallback_mode,
)
from deepreservoir.drl.rewards import (
    RewardContext,
    _niip_historic_day_demand_and_release,
)


def _month_day_median(series: pd.Series, date: pd.Timestamp) -> float:
    values = series.loc[
        (series.index.month == date.month) & (series.index.day == date.day)
    ]
    return float(values.median())


def test_fallback_mode_validation_and_legacy_default() -> None:
    assert normalize_niip_fallback_mode(None) == "legacy_full_series"
    assert normalize_niip_fallback_mode("training_only") == "training_only"
    with pytest.raises(ValueError, match="niip_fallback_mode"):
        normalize_niip_fallback_mode("evaluation")


def test_fallback_lookups_are_complete_immutable_leap_calendars() -> None:
    legacy = historic_niip_month_day_fallback_lookup("legacy_full_series")
    training = historic_niip_month_day_fallback_lookup("training_only")

    assert isinstance(legacy, tuple)
    assert isinstance(training, tuple)
    assert len(legacy) == len(training) == 366
    assert all(np.isfinite(legacy)) and all(value >= 0.0 for value in legacy)
    assert all(np.isfinite(training)) and all(value >= 0.0 for value in training)


def test_training_fallback_excludes_evaluation_observations() -> None:
    historic = load_historic_niip_delivery_series()
    training_reference = historic.loc[:NIIP_TRAINING_REFERENCE_END]
    early_date = pd.Timestamp("1969-04-22")

    legacy = historic_niip_delivery_target_for_dates(
        [early_date], fallback_mode="legacy_full_series"
    ).iloc[0]
    corrected = historic_niip_delivery_target_for_dates(
        [early_date], fallback_mode="training_only"
    ).iloc[0]

    assert legacy == pytest.approx(_month_day_median(historic, early_date))
    assert corrected == pytest.approx(
        _month_day_median(training_reference, early_date)
    )
    assert legacy != corrected


def test_exact_date_observations_override_both_fallback_tables() -> None:
    historic = load_historic_niip_delivery_series()
    observed_date = pd.Timestamp("2014-04-22")
    expected = float(historic.loc[observed_date])

    legacy = historic_niip_delivery_target_for_dates(
        [observed_date], fallback_mode="legacy_full_series"
    ).iloc[0]
    corrected = historic_niip_delivery_target_for_dates(
        [observed_date], fallback_mode="training_only"
    ).iloc[0]

    assert legacy == corrected == expected


def test_known_evaluation_gap_uses_training_only_month_day_value() -> None:
    historic = load_historic_niip_delivery_series()
    assert NIIP_KNOWN_IN_RECORD_GAP_DATES == (pd.Timestamp("2019-12-11"),)
    gap_date = NIIP_KNOWN_IN_RECORD_GAP_DATES[0]
    evaluation_dates = pd.date_range("2014-01-01", "2024-08-17", freq="D")
    assert tuple(evaluation_dates.difference(historic.index)) == (
        gap_date,
    )

    training_reference = historic.loc[:NIIP_TRAINING_REFERENCE_END]
    corrected = historic_niip_delivery_target_for_dates(
        [gap_date], fallback_mode="training_only"
    ).iloc[0]
    assert corrected == pytest.approx(
        _month_day_median(training_reference, gap_date)
    )


def test_reward_prefers_environment_precomputed_target() -> None:
    date = pd.Timestamp("1969-04-22")
    legacy_target = historic_niip_delivery_target_for_date(date)
    corrected_target = historic_niip_delivery_target_for_dates(
        [date], fallback_mode="training_only"
    ).iloc[0]
    assert legacy_target != corrected_target

    ctx = RewardContext(
        t=0,
        date=date,
        obs=np.zeros(1, dtype=np.float32),
        action=np.zeros(1, dtype=np.float32),
        next_obs=np.zeros(1, dtype=np.float32),
        info={"niip_demand_cfs": corrected_target, "release_niip_cfs": 100.0},
    )
    assert _niip_historic_day_demand_and_release(ctx) == pytest.approx(
        (corrected_target, 100.0)
    )


def test_environment_uses_one_corrected_target_for_observation_and_reward_info() -> None:
    data = model.load_all_model_data()
    date = pd.Timestamp("1969-04-22")
    raw = data["raw"].loc[[date]]
    expected = historic_niip_delivery_target_for_dates(
        [date], fallback_mode="training_only"
    ).iloc[0]
    env = model.make_env(
        data_raw=raw,
        data_norm=data["norm"].loc[raw.index],
        norm_stats=data["norm_stats"],
        reward_spec_str=selected_policy.SELECTED_REWARD_SPEC,
        episode_length=None,
        is_eval=True,
        niip_fallback_mode="training_only",
    )
    try:
        obs, _ = env.reset()
        assert env._niip_historic_demand_cfs[0] == pytest.approx(expected)
        assert obs[2] == pytest.approx(expected / env.max_release_niip_cfs)

        _, _, _, _, info = env.step(np.full(4, -1.0, dtype=np.float32))
        assert info["niip_fallback_mode"] == "training_only"
        assert info["niip_demand_cfs"] == pytest.approx(expected)
    finally:
        env.close()
