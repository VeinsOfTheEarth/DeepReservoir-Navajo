# deepreservoir/define_env/spring_peak_release_curve.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpringPeakCurveConfig:
    """
    A DOY-based target release curve (cfs) for Navajo Dam spring peak operations.

    Built from the 2023 schedule you pasted (Coyote Gulch blog table).
    We represent the schedule as (month, day, target_cfs) points and linearly
    interpolate between points.
    """
    # Points: (month, day, target_release_cfs)
    # The curve's 5/25--6/13 plateau has 20 inclusive calendar days; the
    # separate 5,000-cfs threshold criterion requires 21 qualifying days.
    points_md_cfs: Tuple[Tuple[int, int, float], ...] = (
        (5, 9, 500.0),
        (5, 13, 800.0),
        (5, 15, 1200.0),
        (5, 18, 2000.0),
        (5, 19, 3000.0),
        (5, 22, 4000.0),
        (5, 23, 4600.0),
        (5, 24, 4800.0),
        (5, 25, 5000.0),   # start plateau
        (6, 13, 5000.0),   # end plateau
        (6, 14, 4800.0),   # begin ramp down
        (6, 15, 4500.0),
        (6, 16, 4000.0),
        (6, 17, 3000.0),
        (6, 18, 2800.0),
        (6, 19, 2500.0),
        (6, 20, 2000.0),
        (6, 21, 1500.0),
        (6, 22, 1200.0),
        (6, 23, 1000.0),
        (6, 24, 800.0),
        (6, 25, 500.0),
    )

    # If date is outside the spring window, return 0 (inactive).
    inactive_value_cfs: float = 0.0


def _md_to_doy(month: int, day: int, year: int = 2021) -> int:
    """
    Convert month/day -> day-of-year using a fixed non-leap "dummy" year.
    We use 2021 (non-leap) as a nominal month/day coordinate.
    """
    return int(pd.Timestamp(year=year, month=month, day=day).dayofyear)


class SpringPeakReleaseCurve:
    """
    Target release curve (cfs) as a function of date (Timestamp) or DOY.
    """

    def __init__(self, cfg: Optional[SpringPeakCurveConfig] = None):
        self.cfg = cfg or SpringPeakCurveConfig()

        # Convert point list to DOY and target arrays
        doys = np.array([_md_to_doy(m, d) for (m, d, _) in self.cfg.points_md_cfs], dtype=int)
        vals = np.array([float(v) for (_, _, v) in self.cfg.points_md_cfs], dtype=float)

        # Sanity: must be increasing DOY
        if np.any(np.diff(doys) < 0):
            raise ValueError("SpringPeakCurve points must be in increasing calendar order.")

        self._doys = doys
        self._vals = vals
        self._doy_start = int(doys[0])
        self._doy_end = int(doys[-1])

    @property
    def doy_start(self) -> int:
        return self._doy_start

    @property
    def doy_end(self) -> int:
        return self._doy_end

    def target_cfs_from_doy(self, doy: int) -> float:
        """
        Return target release (cfs) for a given DOY.
        Outside the defined window -> inactive_value_cfs.
        """
        doy = int(doy)
        if doy < self._doy_start or doy > self._doy_end:
            return float(self.cfg.inactive_value_cfs)

        # Linear interpolation on DOY points
        return float(np.interp(doy, self._doys, self._vals))

    def target_cfs_from_date(self, date: pd.Timestamp) -> float:
        """
        Return target release (cfs) for a given date (Timestamp-like).
        """
        date = pd.to_datetime(date)
        doy = int(date.dayofyear) - int(bool(date.is_leap_year) and date.month > 2)
        return self.target_cfs_from_doy(doy)

    def targets_for_date_index(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Vectorized helper: return a Series of targets (cfs) for a date index.
        """
        dates = pd.to_datetime(dates)
        doys = dates.dayofyear.values.astype(int) - (
            np.asarray(dates.is_leap_year, dtype=bool) & (dates.month > 2)
        ).astype(int)

        # Build result with inactive outside window
        out = np.full_like(doys, fill_value=float(self.cfg.inactive_value_cfs), dtype=float)
        mask = (doys >= self._doy_start) & (doys <= self._doy_end)
        out[mask] = np.interp(doys[mask], self._doys, self._vals)
        return pd.Series(out, index=dates, name="spring_peak_target_cfs")


def make_spring_peak_target_series_for_year(
    year: int,
    cfg: Optional[SpringPeakCurveConfig] = None,
) -> pd.Series:
    """
    Convenience function:
    Build a *daily* target series for a given year (from first point to last point).
    """
    curve = SpringPeakReleaseCurve(cfg=cfg)

    # Create real dates for that year using month/day points
    dates = [pd.Timestamp(year=year, month=m, day=d) for (m, d, _) in curve.cfg.points_md_cfs]
    idx = pd.date_range(min(dates), max(dates), freq="D")

    s = curve.targets_for_date_index(idx)
    s.name = f"spring_peak_target_cfs_{year}"
    return s
