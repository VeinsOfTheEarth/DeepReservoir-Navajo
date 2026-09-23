"""Regression checks for complete seasons and evaluation denominators."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from deepreservoir.define_env.spring_peak_release_curve import SpringPeakReleaseCurve
from deepreservoir.drl.environs import NavajoReservoirEnv
from deepreservoir.drl.metrics import compute_metrics
from deepreservoir.drl.model import _complete_water_years, _mean_complete_water_year_total


class CalendarWindowTests(unittest.TestCase):
    def test_spring_curve_uses_calendar_dates_in_leap_years(self) -> None:
        curve = SpringPeakReleaseCurve()
        dates = pd.to_datetime(
            ["2024-05-08", "2024-05-09", "2024-06-25", "2024-06-26"]
        )
        values = curve.targets_for_date_index(dates)
        self.assertEqual(values.tolist(), [0.0, 500.0, 500.0, 0.0])
        for date in dates:
            self.assertEqual(curve.target_cfs_from_date(date), values.loc[date])

    def test_niip_annual_ratio_omits_partial_season_but_daily_error_uses_it(self) -> None:
        dates = pd.date_range("2014-01-01", "2015-08-17", freq="D")
        in_window = (dates.dayofyear >= 50) & (dates.dayofyear <= 300)
        demand = np.where(in_window, 1.0, 0.0)
        delivery = np.where(dates.year == 2014, demand, 0.0)
        df = pd.DataFrame(
            {
                "niip_demand_cfs": demand,
                "release_niip_cfs": delivery,
                "rc_niip.delivery_match_jon": 0.0,
            },
            index=dates,
        )
        result = compute_metrics(df, which="niip").iloc[0]
        self.assertEqual(result["niip_annual_volume_frac_of_contract"], 1.0)
        self.assertAlmostEqual(result["niip_frac_days_demand_met_in_window"], 1.0)
        self.assertGreater(result["niip_mean_abs_daily_error_cfs"], 0.0)

    def test_niip_annual_ratio_omits_gapped_or_missing_season(self) -> None:
        dates = pd.date_range("2014-01-01", "2014-12-31", freq="D")
        df = pd.DataFrame(
            {
                "niip_demand_cfs": 1.0,
                "release_niip_cfs": 1.0,
                "rc_niip.delivery_match_jon": 0.0,
            },
            index=dates,
        )
        for altered in (df.drop(pd.Timestamp("2014-06-01")), df.assign(
            release_niip_cfs=df["release_niip_cfs"].mask(df.index == pd.Timestamp("2014-06-01"))
        )):
            result = compute_metrics(altered, which="niip").iloc[0]
            self.assertTrue(np.isnan(result["niip_annual_volume_frac_of_contract"]))

    def test_spring_frequency_requires_complete_valid_window(self) -> None:
        dates = pd.date_range("2024-05-09", "2024-06-25", freq="D")
        df = pd.DataFrame(
            {
                "animas_farmington_q_cfs": 0.0,
                "release_sj_main_cfs": 10_500.0,
            },
            index=dates,
        )
        key = "spr_freq_years_meeting_10000cfs_5d"
        self.assertEqual(compute_metrics(df, which="spring_peak_release_detail").iloc[0][key], 1.0)
        for altered in (df.drop(pd.Timestamp("2024-05-09")), df.assign(
            animas_farmington_q_cfs=df["animas_farmington_q_cfs"].mask(
                df.index == pd.Timestamp("2024-05-09")
            )
        )):
            result = compute_metrics(altered, which="spring_peak_release_detail").iloc[0]
            self.assertTrue(np.isnan(result[key]))


class EpisodeAndDayDenominatorTests(unittest.TestCase):
    def test_partial_first_spring_year_is_not_recorded(self) -> None:
        for start_date, eligible in (
            ("2020-01-01", True),
            ("2020-05-09", True),
            ("2020-05-10", False),
            ("2020-07-01", False),
        ):
            with self.subTest(start_date=start_date):
                env = NavajoReservoirEnv.__new__(NavajoReservoirEnv)
                env._reset_spr_episode_frequency_tracking(
                    wy=2020, start_date=pd.Timestamp(start_date)
                )
                env._reset_spr_threshold_counters(wy=2020)
                env._spr_days_so_far_by_spec[(10_000.0, 5)] = 5
                env._sync_spr_threshold_counter_wy(2021)
                self.assertEqual(2020 in env._spr_recorded_years, eligible)
                env._spr_days_so_far_by_spec[(10_000.0, 5)] = 5
                env._sync_spr_threshold_counter_wy(2022)
                self.assertIn(2021, env._spr_recorded_years)

    def test_hydropower_uses_valid_days_including_partial_year(self) -> None:
        dates = pd.date_range("2014-01-01", "2015-08-17", freq="D")
        df = pd.DataFrame(
            {"hydro_agent_mwh": np.where(dates.year == 2014, 768.0, 0.0)},
            index=dates,
        )
        result = compute_metrics(df, which="hydropower").iloc[0]
        self.assertAlmostEqual(
            result["hydropower_frac_of_max_possible"],
            365.0 / float(len(dates)),
        )

    def test_missing_spill_cannot_be_called_spill_free(self) -> None:
        dates = pd.date_range("2014-01-01", periods=2, freq="D")
        df = pd.DataFrame({"spill_af": [0.0, np.nan]}, index=dates)
        result = compute_metrics(df, which="operational_diagnostics").iloc[0]
        self.assertTrue(np.isnan(result["total_spill_af"]))
        self.assertTrue(np.isnan(result["frac_days_spilling"]))

    def test_complete_water_year_rejects_missing_leap_day_or_flow(self) -> None:
        dates = pd.date_range("2019-10-01", "2020-09-30", freq="D")
        self.assertEqual(_complete_water_years(dates), {2020})
        self.assertEqual(
            _complete_water_years(dates.drop(pd.Timestamp("2020-03-01"))), set()
        )
        df = pd.DataFrame({"inflow_cfs": 1.0}, index=dates)
        df.loc["2020-03-01", "inflow_cfs"] = np.nan
        self.assertTrue(np.isnan(_mean_complete_water_year_total(df, "inflow_cfs", cfs_to_af=False)))


if __name__ == "__main__":
    unittest.main()
