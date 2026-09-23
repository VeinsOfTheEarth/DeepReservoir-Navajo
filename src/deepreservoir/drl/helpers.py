"""deepreservoir.drl.helpers

Small, dependency-light helpers shared across DRL modules.

This file intentionally stays free of SB3/Gym imports (except type hints) so it
can be used by both CLI and core model code without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Water-year utilities
# -----------------------------------------------------------------------------


def water_year_from_timestamp(ts: pd.Timestamp) -> int:
    """USGS-style water year (Oct–Sep).

    Example:
      - 2015-10-01 -> WY 2016
      - 2016-09-30 -> WY 2016
    """
    ts = pd.to_datetime(ts)
    return int(ts.year + (1 if ts.month >= 10 else 0))


def water_year_start_end(wy: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start_date, end_date) for a water year, inclusive."""
    wy = int(wy)
    start = pd.Timestamp(year=wy - 1, month=10, day=1)
    end = pd.Timestamp(year=wy, month=9, day=30)
    return start, end


def split_train_test_by_water_year(df: pd.DataFrame, n_years_test: int = 10):
    """Split df into train/test by *water year* (Oct–Sep), taking the last
    `n_years_test` water years as test.

    Returns
    -------
    train_df, test_df
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex on df.")

    wy = pd.Series(idx.year + (idx.month >= 10).astype(int), index=idx, name="water_year")
    last_wy = int(wy.max())
    first_test_wy = last_wy - int(n_years_test) + 1
    test_mask = wy >= first_test_wy

    train_df = df.loc[~test_mask]
    test_df = df.loc[test_mask]

    print(
        f"[model_data] train {train_df.index.min().date()}–{train_df.index.max().date()}, "
        f"test {test_df.index.min().date()}–{test_df.index.max().date()} "
        f"({n_years_test} water years test; split between WY {first_test_wy-1} and {first_test_wy})."
    )

    return train_df, test_df


# -----------------------------------------------------------------------------
# Window parsing / slicing
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedWindow:
    """Resolved time window in absolute dates (inclusive)."""

    start: pd.Timestamp
    end: pd.Timestamp
    start_token: str
    end_token: str

    @property
    def n_days(self) -> int:
        try:
            return int((self.end - self.start).days) + 1
        except Exception:
            return 0


def _is_water_year_token(token: str) -> bool:
    return isinstance(token, str) and len(token) == 4 and token.isdigit()


def parse_window_bound(token: str, *, bound: str) -> pd.Timestamp:
    """Parse a window boundary token.

    Rules:
      - If token is 4 digits: interpret as WATER YEAR.
          * bound='start' -> WY start date (Oct 1 of prior calendar year)
          * bound='end'   -> WY end date   (Sep 30 of WY)
      - Else: interpret as YYYY-MM-DD.

    Parameters
    ----------
    token:
        Either 'YYYY' (water year) or 'YYYY-MM-DD'.
    bound:
        'start' or 'end'.
    """
    if bound not in {"start", "end"}:
        raise ValueError(f"bound must be 'start' or 'end', got: {bound!r}")

    if _is_water_year_token(token):
        wy = int(token)
        wy_start, wy_end = water_year_start_end(wy)
        return wy_start if bound == "start" else wy_end

    try:
        ts = pd.to_datetime(token)
    except Exception as e:
        raise ValueError(
            f"Invalid window token {token!r}. Expected 4-digit water year 'YYYY' "
            "or date 'YYYY-MM-DD'."
        ) from e

    if pd.isna(ts):
        raise ValueError(
            f"Invalid window token {token!r}. Expected 4-digit water year 'YYYY' "
            "or date 'YYYY-MM-DD'."
        )
    return pd.Timestamp(ts.date())


def resolve_window(start_token: str, end_token: str) -> ResolvedWindow:
    start = parse_window_bound(start_token, bound="start")
    end = parse_window_bound(end_token, bound="end")
    if end < start:
        raise ValueError(f"Window end ({end.date()}) is before start ({start.date()}).")
    return ResolvedWindow(start=start, end=end, start_token=start_token, end_token=end_token)


def slice_by_window(
    df: pd.DataFrame,
    *,
    start_token: Optional[str],
    end_token: Optional[str],
    label: str = "window",
) -> Tuple[pd.DataFrame, Optional[ResolvedWindow]]:
    """Return df sliced to [start:end] inclusive, along with the resolved window.

    If start_token or end_token is None, the full range is used for that side.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex on df.")

    if start_token is None:
        start = pd.Timestamp(df.index.min().date())
        start_token_eff = start.strftime("%Y-%m-%d")
    else:
        start = parse_window_bound(start_token, bound="start")
        start_token_eff = start_token

    if end_token is None:
        end = pd.Timestamp(df.index.max().date())
        end_token_eff = end.strftime("%Y-%m-%d")
    else:
        end = parse_window_bound(end_token, bound="end")
        end_token_eff = end_token

    if end < start:
        raise ValueError(f"{label} end ({end.date()}) is before start ({start.date()}).")

    out = df.loc[start:end]
    if out.empty:
        raise ValueError(
            f"{label} slice produced empty dataframe for {start.date()}–{end.date()}. "
            f"Available range is {df.index.min().date()}–{df.index.max().date()}."
        )

    return out, ResolvedWindow(start=start, end=end, start_token=start_token_eff, end_token=end_token_eff)


def available_date_range(df: pd.DataFrame) -> dict:
    """Return summary info about the available time range in df."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex on df.")
    start = pd.Timestamp(df.index.min().date())
    end = pd.Timestamp(df.index.max().date())
    return {
        "start": start,
        "end": end,
        "n_days": int((end - start).days) + 1,
        "min_water_year": water_year_from_timestamp(start),
        "max_water_year": water_year_from_timestamp(end),
    }


# -----------------------------------------------------------------------------
# Multi-segment training utilities
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedIndexSegment:
    """A resolved window plus its index bounds (inclusive) in a dataframe."""

    window: ResolvedWindow
    start_idx: int
    end_idx: int

    @property
    def n_steps(self) -> int:
        return int(self.end_idx - self.start_idx + 1)


def parse_range_spec(spec: str) -> tuple[Optional[str], Optional[str]]:
    """Parse a START:END range spec.

    START and/or END may be empty to indicate open-ended.

    Examples
    --------
    '2000:2010' -> ('2000','2010')
    '2011:'     -> ('2011', None)
    ':1999'     -> (None, '1999')

    Tokens follow the same rules as elsewhere:
      - 'YYYY' water year token
      - 'YYYY-MM-DD' explicit date
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"Invalid range spec {spec!r}. Expected format 'START:END' with ':' delimiter."
        )
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid range spec {spec!r}. Expected exactly one ':' delimiter (format 'START:END')."
        )
    start_token = parts[0].strip() or None
    end_token = parts[1].strip() or None
    return start_token, end_token


def resolve_window_in_df(
    df: pd.DataFrame,
    *,
    start_token: Optional[str],
    end_token: Optional[str],
    label: str = "window",
) -> ResolvedWindow:
    """Resolve a (start,end) token pair against df's available range."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex on df.")

    if start_token is None:
        start = pd.Timestamp(df.index.min().date())
        start_token_eff = start.strftime("%Y-%m-%d")
    else:
        start = parse_window_bound(start_token, bound="start")
        start_token_eff = start_token

    if end_token is None:
        end = pd.Timestamp(df.index.max().date())
        end_token_eff = end.strftime("%Y-%m-%d")
    else:
        end = parse_window_bound(end_token, bound="end")
        end_token_eff = end_token

    if end < start:
        raise ValueError(f"{label} end ({end.date()}) is before start ({start.date()}).")

    # Bounds check early for clearer errors.
    if start < pd.Timestamp(df.index.min().date()) or end > pd.Timestamp(df.index.max().date()):
        raise ValueError(
            f"{label} {start.date()}–{end.date()} falls outside available range "
            f"{df.index.min().date()}–{df.index.max().date()}."
        )

    return ResolvedWindow(start=start, end=end, start_token=start_token_eff, end_token=end_token_eff)


def window_to_index_range(df: pd.DataFrame, window: ResolvedWindow, *, label: str) -> tuple[int, int]:
    """Convert a ResolvedWindow (absolute dates) to inclusive index bounds in df."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex on df.")

    try:
        start_idx = int(df.index.get_loc(window.start))
        end_idx = int(df.index.get_loc(window.end))
    except KeyError as e:
        raise KeyError(
            f"{label} boundary date not found in dataframe index. "
            f"Tried {window.start.date()}–{window.end.date()}, but index spans "
            f"{df.index.min().date()}–{df.index.max().date()} (daily contiguous block)."
        ) from e

    if end_idx < start_idx:
        raise ValueError(f"{label} has inverted indices: {start_idx}..{end_idx}")
    return start_idx, end_idx


def resolve_range_specs(
    df: pd.DataFrame,
    specs: list[str],
    *,
    label: str,
) -> list[ResolvedIndexSegment]:
    """Resolve a list of 'START:END' specs into index-bounded segments."""
    out: list[ResolvedIndexSegment] = []
    for s in specs:
        st, en = parse_range_spec(s)
        w = resolve_window_in_df(df, start_token=st, end_token=en, label=label)
        i0, i1 = window_to_index_range(df, w, label=label)
        out.append(ResolvedIndexSegment(window=w, start_idx=i0, end_idx=i1))
    return out


def merge_index_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent inclusive index ranges."""
    if not ranges:
        return []
    rr = sorted([(int(a), int(b)) for a, b in ranges], key=lambda x: x[0])
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = rr[0]
    for s, e in rr[1:]:
        if s <= cur_e + 1:  # overlap or adjacency
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def complement_index_ranges(
    base: tuple[int, int],
    excludes: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return inclusive index ranges inside `base` not covered by `excludes`."""
    base_s, base_e = int(base[0]), int(base[1])
    if base_e < base_s:
        return []

    ex = []
    for s, e in excludes:
        s2 = max(int(s), base_s)
        e2 = min(int(e), base_e)
        if e2 >= s2:
            ex.append((s2, e2))
    ex = merge_index_ranges(ex)
    if not ex:
        return [(base_s, base_e)]

    out: list[tuple[int, int]] = []
    cur = base_s
    for s, e in ex:
        if cur <= s - 1:
            out.append((cur, s - 1))
        cur = e + 1
    if cur <= base_e:
        out.append((cur, base_e))
    return out
