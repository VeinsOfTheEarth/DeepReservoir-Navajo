"""Calendar feasibility for the corrected SPR reward variant."""

import numpy as np
import pandas as pd

from deepreservoir.drl.environs import NavajoReservoirEnv
from deepreservoir.drl.rewards import (
    RewardContext,
    _spr_actionproxy_highest_justified_terms,
    _spr_calendar_days_left,
    spr_farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong,
    spr_farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar,
)


def _context(
    date: str,
    *,
    days_left: int,
    counts: dict[str, int],
    target_cfs: float,
    release_cfs: float = 4_000.0,
) -> RewardContext:
    info = {
        "action_mode": "esa_base_spr_proxy_4d_max",
        "animas_farmington_q_cfs": 6_000.0,
        "release_sj_main_cfs": release_cfs,
        "max_release_sj_main_cfs": 5_000.0,
        "spr_proxy_window_active": True,
        "spr_proxy_target_cfs": target_cfs,
        "spr_proxy_target_reachable": True,
        "spr_proxy_target_hit": True,
        "spr_proxy_controller_need_cfs": release_cfs,
        "spr_calendar_days_left": days_left,
        "spr_completed_years_so_far": 0,
    }
    info.update({f"spr_days_so_far_{key}": count for key, count in counts.items()})
    return RewardContext(
        t=0,
        date=pd.Timestamp(date),
        obs=np.zeros(1),
        action=np.zeros(1),
        next_obs=np.zeros(1),
        info=info,
    )


def test_last_day_one_needed_is_creditable_but_two_needed_is_not() -> None:
    key = "10000cfs_5d"
    completable = _context(
        "2024-06-25", days_left=1, counts={key: 4}, target_cfs=10_000.0
    )
    infeasible = _context(
        "2024-06-25", days_left=1, counts={key: 3}, target_cfs=10_000.0
    )

    corrected = spr_farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar
    archived = spr_farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong
    assert corrected(completable) > 0.0
    assert corrected(infeasible) < 0.0
    assert archived(infeasible) > 0.0


def test_best_target_skips_infeasible_higher_threshold() -> None:
    ctx = _context(
        "2024-06-24",
        days_left=2,
        counts={"10000cfs_5d": 2, "8000cfs_10d": 9},
        target_cfs=8_000.0,
    )
    best = _spr_actionproxy_highest_justified_terms(ctx, respect_calendar=True)
    assert best is not None
    assert best.target_cfs == 8_000.0


def test_no_action_is_allowed_when_all_targets_are_infeasible() -> None:
    ctx = _context("2024-06-25", days_left=1, counts={}, target_cfs=0.0, release_cfs=0.0)
    corrected = spr_farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar
    archived = spr_farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong
    assert corrected(ctx) == 0.15
    assert archived(ctx) < 0.0


def test_calendar_count_is_inclusive_and_uses_available_rows() -> None:
    for year in (2023, 2024):
        for day, expected in (("05-09", 48), ("06-25", 1), ("06-26", 0)):
            ctx = _context(f"{year}-{day}", days_left=expected, counts={}, target_cfs=0.0)
            del ctx.info["spr_calendar_days_left"]
            assert _spr_calendar_days_left(ctx) == expected

    shortened = _context("2024-06-24", days_left=1, counts={}, target_cfs=0.0)
    assert _spr_calendar_days_left(shortened) == 1


def test_observed_advice_and_reward_use_the_same_available_row_count() -> None:
    env = NavajoReservoirEnv.__new__(NavajoReservoirEnv)
    env._spring_window_active_daily = np.array([True, True, False])
    env._water_year = np.array([2024, 2024, 2024])
    env._row_index = np.arange(3)
    env._raw_animas_farmington_q_cfs = np.array([6_000.0] * 3)
    env._decision_animas_farmington_q_cfs = env._raw_animas_farmington_q_cfs.copy()
    env.max_release_sj_main_cfs = 5_000.0
    env._spr_days_so_far_by_spec = {
        (10_000.0, 5): 2,
        (8_000.0, 10): 9,
        (5_000.0, 21): 0,
        (2_500.0, 10): 0,
    }

    assert env._spr_calendar_days_left(0) == 2
    assert env._spr_advice_candidate(0) == (8_000.0, 10, 1.0)
    ctx = _context(
        "2024-06-24",
        days_left=env._spr_calendar_days_left(0),
        counts={"10000cfs_5d": 2, "8000cfs_10d": 9},
        target_cfs=8_000.0,
    )
    best = _spr_actionproxy_highest_justified_terms(ctx, respect_calendar=True)
    assert best is not None and best.target_cfs == 8_000.0


def test_inactive_advice_fallback_is_not_a_reward_opportunity() -> None:
    env = NavajoReservoirEnv.__new__(NavajoReservoirEnv)
    env._spring_window_active_daily = np.array([True])
    env._water_year = np.array([2024])
    env._row_index = np.arange(1)
    env._raw_animas_farmington_q_cfs = np.array([6_000.0])
    env._decision_animas_farmington_q_cfs = env._raw_animas_farmington_q_cfs.copy()
    env.max_release_sj_main_cfs = 5_000.0
    env._spr_days_so_far_by_spec = {
        (10_000.0, 5): 3,
        (8_000.0, 10): 8,
        (5_000.0, 21): 0,
        (2_500.0, 10): 0,
    }

    assert env._spr_advice_candidate(0) == (10_000.0, 5, 0.5)
    assert env._spr_advice_obs_value("spr_advice_active", 0) == 0.0
    ctx = _context(
        "2024-06-25",
        days_left=env._spr_calendar_days_left(0),
        counts={"10000cfs_5d": 3, "8000cfs_10d": 8},
        target_cfs=0.0,
        release_cfs=0.0,
    )
    corrected = spr_farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar
    assert corrected(ctx) == 0.15
