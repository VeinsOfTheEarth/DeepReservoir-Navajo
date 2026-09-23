from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepreservoir.drl.metrics import (
    FLOOD_COMPARISON_PROFILE_CORRECTED,
    FLOOD_COMPARISON_PROFILE_LEGACY_PHASE95,
    compute_flooding_comparison_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_corrected_comparison_uses_one_common_finite_day_mask() -> None:
    dates = pd.date_range("2014-01-01", periods=5, freq="D")
    rollout = pd.DataFrame(
        {
            # The first row is unavailable to the agent; the second is
            # unavailable to history. Neither may enter either fraction.
            "sj_at_archuleta_proxy_cfs": [6_000.0, 6_000.0, 4_900.0, 5_100.0, 5_000.0005],
            "sj_at_bluff_proxy_cfs": [np.nan, 9_000.0, 9_000.0, 9_000.0, 12_000.0005],
            "sj_archuleta_q_cfs": [4_900.0, np.nan, 5_100.0, 4_900.0, 4_900.0],
            "sj_bluff_q_cfs": [9_000.0, 9_000.0, 9_000.0, 9_000.0, 12_000.0005],
        },
        index=dates,
    )

    result = compute_flooding_comparison_metrics(rollout)

    assert result["flooding_comparison_profile"] == FLOOD_COMPARISON_PROFILE_CORRECTED
    assert result["agent_flooding_available_days"] == 4
    assert result["historic_flooding_available_days"] == 4
    assert result["flooding_comparison_days"] == 3
    assert result["agent_flooding_safe_days"] == 2
    assert result["historic_flooding_safe_days"] == 2
    assert result["agent_flooding_frac_days_met"] == pytest.approx(2.0 / 3.0)
    assert result["historic_flooding_frac_days_met"] == pytest.approx(2.0 / 3.0)


def test_corrected_comparison_applies_same_threshold_tolerance_to_both_series() -> None:
    dates = pd.date_range("2014-01-01", periods=2, freq="D")
    rollout = pd.DataFrame(
        {
            "sj_at_archuleta_proxy_cfs": [5_000.0005, 5_000.0015],
            "sj_at_bluff_proxy_cfs": [12_000.0005, 11_000.0],
            "sj_archuleta_q_cfs": [5_000.0005, 5_000.0015],
            "sj_bluff_q_cfs": [12_000.0005, 11_000.0],
        },
        index=dates,
    )

    result = compute_flooding_comparison_metrics(rollout)

    assert result["flooding_comparison_days"] == 2
    assert result["agent_flooding_safe_days"] == 1
    assert result["historic_flooding_safe_days"] == 1
    assert result["agent_flooding_frac_days_met"] == pytest.approx(0.5)
    assert result["historic_flooding_frac_days_met"] == pytest.approx(0.5)


def test_corrected_comparison_does_not_substitute_release_for_archuleta() -> None:
    dates = pd.date_range("2014-01-01", periods=1, freq="D")
    rollout = pd.DataFrame(
        {
            "sj_at_archuleta_proxy_cfs": [4_900.0],
            "sj_at_bluff_proxy_cfs": [9_000.0],
            "release_cfs": [4_900.0],
            "sj_bluff_q_cfs": [9_000.0],
        },
        index=dates,
    )

    with pytest.raises(KeyError, match="sj_archuleta_q_cfs"):
        compute_flooding_comparison_metrics(rollout)


def test_legacy_profile_retains_missing_lag_as_safe_and_all_rows() -> None:
    dates = pd.date_range("2014-01-01", periods=4, freq="D")
    rollout = pd.DataFrame(
        {
            "sj_at_farmington_cfs": [4_900.0, 5_100.0, 4_900.0, 4_900.0],
            "sj_at_farmington_lag2_cfs": [np.nan, np.nan, 13_000.0, 11_000.0],
            "sj_farmington_q_cfs": [4_900.0, 5_100.0, 4_900.0, 4_900.0],
        },
        index=dates,
    )

    result = compute_flooding_comparison_metrics(
        rollout,
        profile=FLOOD_COMPARISON_PROFILE_LEGACY_PHASE95,
    )

    assert result["flooding_comparison_days"] == 4
    assert result["agent_flooding_available_days"] == 2
    assert result["historic_flooding_available_days"] == 2
    assert result["agent_flooding_safe_days"] == 2
    assert result["historic_flooding_safe_days"] == 3
    assert result["agent_flooding_frac_days_met"] == pytest.approx(0.5)
    assert result["historic_flooding_frac_days_met"] == pytest.approx(0.75)


@pytest.fixture(scope="module")
def archived_phase95_rollout() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts" / "legacy_phase95_policy" / "selected_policy_eval_rollout.parquet"
    rollout = pd.read_parquet(path)
    rollout.index = pd.DatetimeIndex(rollout.index)
    return rollout.sort_index()


def test_legacy_profile_reproduces_frozen_phase95_flood_screen(
    archived_phase95_rollout: pd.DataFrame,
) -> None:
    result = compute_flooding_comparison_metrics(
        archived_phase95_rollout,
        profile=FLOOD_COMPARISON_PROFILE_LEGACY_PHASE95,
    )

    assert result["flooding_comparison_days"] == 3_882
    assert result["agent_flooding_frac_days_met"] == pytest.approx(
        0.9778464708912932
    )
    assert result["historic_flooding_frac_days_met"] == pytest.approx(
        0.9621329211746522
    )


def test_corrected_profile_has_expected_common_holdout_denominator(
    archived_phase95_rollout: pd.DataFrame,
) -> None:
    # The archived rollout predates the explicit proxy aliases and the joined
    # Archuleta series. Populate those columns explicitly for a post-hoc,
    # corrected-location comparison; the helper itself intentionally performs
    # no fallback or external data loading.
    rollout = archived_phase95_rollout.copy()
    rollout["sj_at_archuleta_proxy_cfs"] = rollout["sj_main_flow_cfs"]
    rollout["sj_at_bluff_proxy_cfs"] = rollout["sj_at_farmington_lag2_cfs"]

    archuleta = pd.read_csv(
        REPO_ROOT / "data" / "daily_flows" / "daily_sj_archuleta.csv",
        parse_dates=["time"],
    ).set_index("time")["value"]
    rollout["sj_archuleta_q_cfs"] = archuleta.reindex(rollout.index)

    result = compute_flooding_comparison_metrics(rollout)

    assert result["agent_flooding_available_days"] == 3_880
    assert result["historic_flooding_available_days"] == 3_881
    assert result["flooding_comparison_days"] == 3_879
    assert result["agent_flooding_safe_days"] == 3_879
    assert result["historic_flooding_safe_days"] == 3_878
    assert result["agent_flooding_frac_days_met"] == pytest.approx(1.0)
    assert result["historic_flooding_frac_days_met"] == pytest.approx(3_878 / 3_879)


def test_unknown_profile_is_rejected() -> None:
    rollout = pd.DataFrame(index=pd.date_range("2014-01-01", periods=1, freq="D"))

    with pytest.raises(ValueError, match="Unknown flood comparison profile"):
        compute_flooding_comparison_metrics(rollout, profile="implicit_fallback")
