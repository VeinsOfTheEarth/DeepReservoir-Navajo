from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from deepreservoir.define_env.spring_peak_release_curve import SpringPeakReleaseCurve
from deepreservoir.define_env.storage_elevation.thresholds import (
    get_navajo_storage_thresholds,
)
from deepreservoir.drl.metrics import (
    _SPR_THRESHOLD_SPECS,
    compute_historic_summary_metrics,
    compute_metrics,
)
from deepreservoir.drl.rewards import (
    RewardContext,
    flooding_penalty_caps_archuleta_bluff,
    flooding_penalty_caps_jon,
)


class NiipMetricTests(unittest.TestCase):
    def test_niip_daily_overlap_distinguishes_timing_from_volume(self) -> None:
        dates = pd.date_range("2014-01-01", "2014-12-31", freq="D")
        demand = pd.Series(0.0, index=dates)
        demand.loc["2014-06-01":"2014-06-03"] = [10.0, 20.0, 30.0]
        proportional_delivery = pd.Series(0.0, index=dates)
        proportional_delivery.loc["2014-06-01":"2014-06-03"] = [5.0, 10.0, 15.0]
        backloaded_delivery = pd.Series(0.0, index=dates)
        backloaded_delivery.loc["2014-06-03"] = 30.0

        df_proportional = pd.DataFrame(
            {
                "niip_demand_cfs": demand,
                "release_niip_cfs": proportional_delivery,
                "rc_niip.delivery_match_jon": 0.0,
            },
            index=dates,
        )
        df_backloaded = pd.DataFrame(
            {
                "niip_demand_cfs": demand,
                "release_niip_cfs": backloaded_delivery,
                "rc_niip.delivery_match_jon": 0.0,
            },
            index=dates,
        )

        proportional = compute_metrics(df_proportional, which="niip")
        backloaded = compute_metrics(df_backloaded, which="niip")

        self.assertAlmostEqual(
            float(proportional.loc[0, "niip_frac_days_demand_met_in_window"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(backloaded.loc[0, "niip_frac_days_demand_met_in_window"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(proportional.loc[0, "niip_annual_volume_frac_of_contract"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(backloaded.loc[0, "niip_annual_volume_frac_of_contract"]),
            0.5,
            places=6,
        )


class SprThresholdSpecTests(unittest.TestCase):
    def test_spr_threshold_specs_use_correct_durations(self) -> None:
        self.assertIn((8_000.0, 10, 0.33), _SPR_THRESHOLD_SPECS)
        self.assertIn((5_000.0, 21, 0.50), _SPR_THRESHOLD_SPECS)
        self.assertNotIn((8_000.0, 19, 0.33), _SPR_THRESHOLD_SPECS)
        self.assertNotIn((5_000.0, 20, 0.50), _SPR_THRESHOLD_SPECS)

    def test_spr_total_days_metric_is_nonconsecutive_and_renamed(self) -> None:
        dates = pd.date_range("2018-01-01", "2018-12-31", freq="D")
        curve = SpringPeakReleaseCurve()
        spr_dates = dates[curve.targets_for_date_index(dates).astype(float) > 0.0]
        hit_dates = spr_dates[[0, 2, 4, 6, 8]]

        df = pd.DataFrame(
            {
                "animas_farmington_q_cfs": 0.0,
                "release_sj_main_cfs": 0.0,
                "sj_at_farmington_lag2_cfs": 0.0,
            },
            index=dates,
        )
        df.loc[hit_dates, "release_sj_main_cfs"] = 10_500.0

        metrics = compute_metrics(df, which="spring_peak_release_detail")

        self.assertIn("spr_mean_total_window_days_above_10000cfs", metrics.columns)
        self.assertNotIn("spr_mean_max_consec_days_10000cfs", metrics.columns)
        self.assertAlmostEqual(
            float(metrics.loc[0, "spr_mean_total_window_days_above_10000cfs"]),
            5.0,
            places=6,
        )

    def test_spr_metrics_ignore_spill_in_lagged_total(self) -> None:
        dates = pd.date_range("2018-01-01", "2018-12-31", freq="D")
        curve = SpringPeakReleaseCurve()
        spr_dates = dates[curve.targets_for_date_index(dates).astype(float) > 0.0]
        hit_dates = spr_dates[:5]

        df = pd.DataFrame(
            {
                "animas_farmington_q_cfs": 0.0,
                "release_sj_main_cfs": 0.0,
                "sj_at_farmington_lag2_cfs": 12_000.0,
            },
            index=dates,
        )
        df.loc[hit_dates, "release_sj_main_cfs"] = 10_500.0

        metrics = compute_metrics(df, which="spring_peak_release_detail")

        self.assertAlmostEqual(
            float(metrics.loc[0, "spr_freq_years_meeting_10000cfs_5d"]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics.loc[0, "spr_mean_total_window_days_above_10000cfs"]),
            5.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics.loc[0, "spr_freq_years_meeting_10000cfs_5d"]),
            1.0,
            places=6,
        )


class StorageBoundMetricTests(unittest.TestCase):
    def test_storage_bounds_use_operating_band_not_zero_deadpool(self) -> None:
        dates = pd.date_range("2014-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "storage_agent_af_end": [0.0, 500_000.0, 750_000.0, 1_000_000.0],
                "deadpool_storage_af": [0.0, 0.0, 0.0, 0.0],
                "max_storage_af": [1_000_000.0] * 4,
                "rc_dam_safety.storage_band_jon": [0.0, 0.0, 0.0, 0.0],
            },
            index=dates,
        )

        summary = compute_metrics(df, which="dam_safety")
        detail = compute_metrics(df, which="dam_safety_detail")

        self.assertAlmostEqual(
            float(summary.loc[0, "dam_safety_frac_days_within_storage_bounds"]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(detail.loc[0, "dam_safety_frac_days_below_min_storage"]),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(detail.loc[0, "dam_safety_frac_days_above_max_storage"]),
            0.25,
            places=6,
        )


class StorageThresholdHelperTests(unittest.TestCase):
    def test_navajo_storage_thresholds_are_derived_from_current_curve(self) -> None:
        thresholds = get_navajo_storage_thresholds()

        self.assertAlmostEqual(thresholds.deadpool_elev_ft, 5775.0, places=6)
        self.assertAlmostEqual(thresholds.spill_elev_ft, 6085.0, places=6)
        self.assertAlmostEqual(thresholds.deadpool_storage_af, 0.0, places=6)
        self.assertAlmostEqual(thresholds.max_storage_af, 1_647_936.17, places=2)


class ThresholdMetricToleranceTests(unittest.TestCase):
    def test_esa_min_flow_metric_allows_tiny_threshold_drift(self) -> None:
        dates = pd.date_range("2014-01-01", periods=2, freq="D")
        df = pd.DataFrame(
            {
                "animas_farmington_q_cfs": [250.0, 250.0],
                "release_sj_main_cfs": [249.9995, 249.9985],
                "rc_esa_min_flow.baseline": [0.0, 0.0],
            },
            index=dates,
        )

        metrics = compute_metrics(df, which="esa_min_flow")

        self.assertAlmostEqual(
            float(metrics.loc[0, "esa_min_flow_frac_days_met"]),
            0.5,
            places=6,
        )

    def test_flooding_metric_allows_tiny_threshold_drift(self) -> None:
        dates = pd.date_range("2014-01-01", periods=2, freq="D")
        df = pd.DataFrame(
            {
                "sj_at_archuleta_proxy_cfs": [5000.0005, 5000.0015],
                "sj_at_farmington_cfs": [9000.0, 9000.0],
                "sj_at_bluff_proxy_cfs": [12000.0005, 11999.0],
                "rc_flooding.penalty_caps_jon": [0.0, 0.0],
            },
            index=dates,
        )

        metrics = compute_metrics(df, which="flooding")

        self.assertAlmostEqual(
            float(metrics.loc[0, "flooding_frac_days_met"]),
            0.5,
            places=6,
        )

    def test_flood_reward_excludes_downstream_animas_flow(self) -> None:
        ctx = RewardContext(
            t=0,
            date=pd.Timestamp("2014-01-01"),
            obs=np.zeros(1),
            action=np.zeros(1),
            next_obs=np.zeros(1),
            info={
                "sj_at_archuleta_proxy_cfs": 4_900.0,
                "sj_at_farmington_cfs": 9_000.0,
                "sj_at_bluff_proxy_cfs": 11_000.0,
            },
        )

        self.assertEqual(flooding_penalty_caps_archuleta_bluff(ctx), 0.0)
        self.assertEqual(flooding_penalty_caps_jon(ctx), -0.5)

    def test_agent_flood_metric_uses_paired_valid_proxy_days(self) -> None:
        dates = pd.date_range("2014-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "sj_at_archuleta_proxy_cfs": [5_100.0, 4_900.0, 4_900.0],
                "sj_at_bluff_proxy_cfs": [float("nan"), 9_000.0, 13_000.0],
                "rc_flooding.penalty_caps_jon": [0.0, 0.0, 0.0],
            },
            index=dates,
        )

        metrics = compute_metrics(df, which="flooding")

        self.assertAlmostEqual(
            float(metrics.loc[0, "flooding_frac_days_met"]),
            0.5,
            places=6,
        )

    def test_agent_flood_metric_requires_bluff_proxy(self) -> None:
        dates = pd.date_range("2014-01-01", periods=1, freq="D")
        df = pd.DataFrame(
            {
                "sj_at_archuleta_proxy_cfs": [4_900.0],
                "rc_flooding.penalty_caps_jon": [0.0],
            },
            index=dates,
        )

        metrics = compute_metrics(df, which="flooding")

        self.assertTrue(np.isnan(float(metrics.loc[0, "flooding_frac_days_met"])))

    def test_historic_flood_metric_uses_archuleta_and_excludes_missing_days(self) -> None:
        dates = pd.date_range("2014-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "sj_archuleta_q_cfs": [4_900.0, 5_100.0, float("nan")],
                "sj_bluff_q_cfs": [9_000.0, 9_000.0, 13_000.0],
            },
            index=dates,
        )

        metrics = compute_historic_summary_metrics(df)

        self.assertAlmostEqual(metrics["flooding_frac_days_met"], 0.5, places=6)

    def test_spr_threshold_metric_allows_tiny_threshold_drift(self) -> None:
        dates = pd.date_range("2018-01-01", "2018-12-31", freq="D")
        curve = SpringPeakReleaseCurve()
        spr_dates = dates[curve.targets_for_date_index(dates).astype(float) > 0.0]
        hit_dates = spr_dates[:5]

        df = pd.DataFrame(
            {
                "animas_farmington_q_cfs": 0.0,
                "release_sj_main_cfs": 0.0,
            },
            index=dates,
        )
        df.loc[hit_dates, "release_sj_main_cfs"] = 9999.9995

        metrics = compute_metrics(df, which="spring_peak_release_detail")

        self.assertAlmostEqual(
            float(metrics.loc[0, "spr_freq_years_meeting_10000cfs_5d"]),
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
