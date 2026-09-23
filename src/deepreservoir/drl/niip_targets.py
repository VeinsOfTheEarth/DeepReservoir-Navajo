"""NIIP target helpers used by rewards, observations, and metrics."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from deepreservoir.data.metadata import project_metadata


NIIP_FALLBACK_MODE_CHOICES = ("legacy_full_series", "training_only")
NIIP_TRAINING_REFERENCE_END = pd.Timestamp("2013-12-31")
NIIP_KNOWN_IN_RECORD_GAP_DATES = (pd.Timestamp("2019-12-11"),)

# A leap-year calendar gives every month/day, including February 29, one fixed
# slot.  Keeping the lookup as an immutable tuple prevents later call sites
# from changing the fallback values after an environment has been configured.
_FALLBACK_CALENDAR = pd.date_range("2000-01-01", "2000-12-31", freq="D")
_FALLBACK_MONTH_DAY_KEYS = tuple(
    (int(date.month), int(date.day)) for date in _FALLBACK_CALENDAR
)
_FALLBACK_MONTH_DAY_TO_SLOT = {
    key: slot for slot, key in enumerate(_FALLBACK_MONTH_DAY_KEYS)
}


def normalize_niip_fallback_mode(value: str | None) -> str:
    """Normalize the reference record used only when an exact NIIP day is absent."""
    key = str(value or "legacy_full_series").strip().lower()
    if key not in NIIP_FALLBACK_MODE_CHOICES:
        raise ValueError(
            f"Unsupported niip_fallback_mode {value!r}; "
            f"expected {NIIP_FALLBACK_MODE_CHOICES}"
        )
    return key


@lru_cache(maxsize=1)
def load_historic_niip_delivery_series() -> pd.Series:
    """Return observed historic NIIP delivery as a daily CFS series."""
    m = project_metadata()
    path = m.path("niip_historic")
    raw = pd.read_csv(
        path,
        usecols=["Date", "Flow (cfs)"],
        comment="*",
        skipinitialspace=True,
    )
    dates = pd.to_datetime(raw["Date"], format="%d-%b-%y", errors="coerce")
    flow = pd.to_numeric(raw["Flow (cfs)"], errors="coerce")
    series = pd.Series(flow.to_numpy(dtype=float), index=dates)
    series = series[series.index.notna() & series.notna()]
    if series.empty:
        return pd.Series(dtype=float, name="niip_historic_delivery_cfs")
    series.index = pd.DatetimeIndex(series.index).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.clip(lower=0.0).rename("niip_historic_delivery_cfs")


@lru_cache(maxsize=None)
def historic_niip_month_day_fallback_lookup(
    fallback_mode: str = "legacy_full_series",
) -> tuple[float, ...]:
    """Return a deterministic 366-value calendar lookup for missing NIIP days.

    ``legacy_full_series`` reproduces the archived Phase-95 preprocessing by
    taking month/day medians over the complete NIIP file. ``training_only``
    calculates the same lookup from observations through 2013-12-31, before
    the paper evaluation period begins. Exact-date observations are never
    replaced by this lookup.
    """
    mode = normalize_niip_fallback_mode(fallback_mode)
    series = load_historic_niip_delivery_series()
    if series.empty:
        return (0.0,) * len(_FALLBACK_MONTH_DAY_KEYS)
    if mode == "training_only":
        series = series.loc[:NIIP_TRAINING_REFERENCE_END]
    if series.empty:
        return (0.0,) * len(_FALLBACK_MONTH_DAY_KEYS)
    df = pd.DataFrame(
        {
            "month": series.index.month,
            "day": series.index.day,
            "flow_cfs": series.to_numpy(dtype=float),
        }
    )
    grouped = df.groupby(["month", "day"], sort=True)["flow_cfs"].median()
    complete_index = pd.MultiIndex.from_tuples(
        _FALLBACK_MONTH_DAY_KEYS,
        names=["month", "day"],
    )
    values = grouped.reindex(complete_index).fillna(0.0).clip(lower=0.0)
    return tuple(float(value) for value in values.to_numpy(dtype=float))


def historic_niip_delivery_target_for_dates(
    dates: pd.DatetimeIndex | list[pd.Timestamp],
    *,
    fallback_mode: str = "legacy_full_series",
) -> pd.Series:
    """Return the NIIP target for each date.

    The primary target is the observed NIIP delivery on that exact date. For
    missing dates, use the selected month/day lookup so the reward remains
    active instead of silently dropping early training years. The historic file
    has one internal gap, 2019-12-11; it follows this same explicit rule.
    """
    mode = normalize_niip_fallback_mode(fallback_mode)
    idx = pd.DatetimeIndex(dates).normalize()
    historic = load_historic_niip_delivery_series()
    target = historic.reindex(idx) if not historic.empty else pd.Series(np.nan, index=idx)

    if target.isna().any():
        lookup = historic_niip_month_day_fallback_lookup(mode)
        fallback = np.asarray(
            [
                lookup[_FALLBACK_MONTH_DAY_TO_SLOT[(int(date.month), int(date.day))]]
                for date in idx
            ],
            dtype=float,
        )
        target = target.fillna(pd.Series(fallback, index=idx))

    return target.fillna(0.0).clip(lower=0.0).rename("niip_demand_cfs")


def historic_niip_delivery_target_for_date(date: pd.Timestamp) -> float:
    """Legacy scalar wrapper retained for archived reward-call compatibility."""
    return float(historic_niip_delivery_target_for_dates([pd.Timestamp(date)]).iloc[0])
