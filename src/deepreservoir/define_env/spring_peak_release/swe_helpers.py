# swe_helpers.py — slim helpers for SPR exploration (Py3.9-safe)
from __future__ import annotations

import calendar
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from dataclasses import dataclass


# ---- small config used by these helpers
CFS_DAY_TO_ACFT = 1.983471      # 1 cfs for 1 day -> acre-feet
MIN_SPRING_DAY_FRAC = 0.60      # min Apr–Jul coverage for WY metrics
FREQ_15MIN_PER_DAY = 96         # for resampling sub-daily -> daily with coverage check

def _as_series(x, col: Optional[str], context: str) -> pd.Series:
    """Return a datetime-indexed numeric Series from Series/DF `x`."""
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if col and col in x.columns:
            s = x[col].copy()
        else:
            numcols = x.select_dtypes(include="number").columns
            if len(numcols) == 1:
                s = x[numcols[0]].copy()
            else:
                raise ValueError(
                    f"{context}: please specify column name. "
                    f"Found numeric columns: {list(numcols)}"
                )
    else:
        raise TypeError(f"{context}: expected Series or DataFrame, got {type(x)}")

    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.sort_index()

# ---------------------------------------------------------------------
# Core builders you asked to keep
# ---------------------------------------------------------------------

def build_daily_swe(swe_df: pd.DataFrame) -> pd.Series:
    """
    Make a daily SWE series from hourly (or sub-daily) data.
    Uses daily MAX (robust to time-of-day sampling), then a light 3-day smooth.
    Expects columns: ['date','snow_depth_water_equivalent'] (meters).
    """
    ser = swe_df.set_index(pd.to_datetime(swe_df['date']))['snow_depth_water_equivalent'].sort_index()
    daily = ser.resample('D').max()
    return daily.rolling(3, center=True, min_periods=1).mean()


def _safe_day(series: pd.Series, ts: pd.Timestamp) -> float:
    """Return value at ts; if missing, nearest within ±3 days (NaN if none)."""
    if ts in series.index and pd.notna(series.loc[ts]):
        return float(series.loc[ts])
    win = series.loc[ts - pd.Timedelta(days=3): ts + pd.Timedelta(days=3)].dropna()
    return float(win.iloc[(win.index - ts).abs().argmin()]) if not win.empty else np.nan


def assemble_wy_metrics(
    swe_daily,               # Series or DF
    q_daily,                 # Series or DF
    swe_col: Optional[str] = "animas_swe_m",
    q_col:   Optional[str] = "animas_farmington_q_cfs",
) -> pd.DataFrame:
    """
    Build WY-indexed metrics from daily SWE (meters) and Animas Q (cfs).

    New metric:
      - SWE_peak_by_Mar1_mm : max SWE between Nov 1 (wy-1) and Mar 1 (wy), inclusive
    """
    swe_daily = _as_series(swe_daily, swe_col, "assemble_wy_metrics(swe_daily)")
    q_daily   = _as_series(q_daily,   q_col,   "assemble_wy_metrics(q_daily)")

    wys_swe = (swe_daily.index.year + (swe_daily.index.month >= 10)).unique()
    wys_q   = (q_daily.dropna().index.year + (q_daily.dropna().index.month >= 10)).unique()
    wys = sorted(set(wys_swe).intersection(set(wys_q)))

    rows = []
    for wy in wys:
        # SWE windows
        swe_win = swe_daily.loc[pd.Timestamp(wy-1, 11, 1): pd.Timestamp(wy, 5, 31)]
        if swe_win.empty:
            continue

        feb_last = calendar.monthrange(wy, 2)[1]
        feb = swe_daily.loc[pd.Timestamp(wy, 2, 1): pd.Timestamp(wy, 2, feb_last)]
        pre_mar1 = swe_daily.loc[pd.Timestamp(wy-1, 11, 1): pd.Timestamp(wy, 3, 1)]  # Nov 1 .. Mar 1 inclusive

        # Scalars (avoid FutureWarning from float(Series))
        swe_peak_m  = swe_win.max() if np.isscalar(swe_win.max()) else swe_win.max().item()
        swe_peak_dt = swe_win.idxmax()

        swe_feb_mean = feb.mean() if np.isscalar(feb.mean()) else (feb.mean().item() if not feb.dropna().empty else np.nan)
        swe_feb_max  = feb.max()  if np.isscalar(feb.max())  else (feb.max().item()  if not feb.dropna().empty else np.nan)

        # NEW: peak by Mar 1 (forward-safe)
        pre_mar1_max = pre_mar1.max()
        swe_peak_by_mar1_m = pre_mar1_max if np.isscalar(pre_mar1_max) else (pre_mar1_max.item() if not pre_mar1.dropna().empty else np.nan)

        swe_mar1 = _safe_day(swe_daily, pd.Timestamp(wy, 3, 1))
        swe_apr1 = _safe_day(swe_daily, pd.Timestamp(wy, 4, 1))

        # Apr–Jul flow window & coverage
        q_win = q_daily.loc[pd.Timestamp(wy, 4, 1): pd.Timestamp(wy, 7, 31)]
        needed = (pd.Timestamp(wy, 7, 31) - pd.Timestamp(wy, 4, 1)).days + 1
        q_valid = q_win.dropna()
        if q_valid.shape[0] < MIN_SPRING_DAY_FRAC * needed:
            continue

        q_mean  = float(q_valid.mean())
        q_peak1 = float(q_valid.max())
        q_peak3 = float(q_valid.rolling(3, center=True, min_periods=2).mean().max())
        q_total_obs_acft = float((q_valid * CFS_DAY_TO_ACFT).sum())
        scale = needed / q_valid.shape[0] if q_valid.shape[0] else np.nan
        q_total_scaled_acft = float(q_total_obs_acft * scale) if np.isfinite(scale) else np.nan

        rows.append(dict(
            WY=wy,
            SWE_peak_mm=swe_peak_m * 1000.0, SWE_peak_date=swe_peak_dt,
            SWE_Feb_mean_mm=swe_feb_mean * 1000.0 if pd.notna(swe_feb_mean) else np.nan,
            SWE_Feb_max_mm=swe_feb_max * 1000.0 if pd.notna(swe_feb_max) else np.nan,
            SWE_Mar1_mm=swe_mar1 * 1000.0,
            SWE_Apr1_mm=swe_apr1 * 1000.0,
            # NEW export:
            SWE_peak_by_Mar1_mm=swe_peak_by_mar1_m * 1000.0,
            Q_AprJul_mean_cfs=q_mean, Q_AprJul_peak1d_cfs=q_peak1, Q_AprJul_peak3d_cfs=q_peak3,
            Q_AprJul_total_acft_obs=q_total_obs_acft, Q_AprJul_total_acft_scaled=q_total_scaled_acft,
            Q_days_present=int(q_valid.shape[0]), Q_days_expected=int(needed),
        ))

    df = pd.DataFrame(rows).set_index("WY").sort_index()
    if df.empty:
        print("assemble_wy_metrics: no qualifying years found.")
    return df


def prespring_storage_by_wy(
    usbr_df: pd.DataFrame,
    method: str = "feb_mean",     # "feb_mean" | "feb_max" | "mar_mean" | "window"
    window_days: Optional[int] = None,
    min_frac: float = 0.60
) -> pd.Series:
    """
    Compute a pre-SPR storage metric per WY from a USBR daily DataFrame with 'storage_af'.

    - feb_mean : mean over Feb 1..Feb end
    - feb_max  : max  over Feb 1..Feb end
    - mar_mean : mean over Mar 1..Mar 31   (for A/B tests)
    - window   : mean over the last N days ending Feb-end  (set window_days)

    Requires ≥ min_frac of days present in the window.
    """
    s = usbr_df["storage_af"].copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index().asfreq("D")

    years = range(s.index.min().year, s.index.max().year + 1)
    out = {}
    for wy in years:
        feb_start = pd.Timestamp(wy, 2, 1)
        feb_end   = pd.Timestamp(wy, 3, 1) - pd.Timedelta(days=1)
        mar_start = pd.Timestamp(wy, 3, 1)
        mar_end   = pd.Timestamp(wy, 3, 31)

        if method == "feb_mean":
            win = s.loc[feb_start:feb_end]; expected = (feb_end - feb_start).days + 1; val = win.mean()
        elif method == "feb_max":
            win = s.loc[feb_start:feb_end]; expected = (feb_end - feb_start).days + 1; val = win.max()
        elif method == "mar_mean":
            win = s.loc[mar_start:mar_end]; expected = (mar_end - mar_start).days + 1; val = win.mean()
        elif method == "window":
            if not window_days or window_days < 1:
                raise ValueError("For method='window', provide window_days >= 1.")
            win_end   = feb_end
            win_start = win_end - pd.Timedelta(days=window_days - 1)
            win = s.loc[win_start:win_end]; expected = window_days; val = win.mean()
        else:
            raise ValueError("method must be one of: feb_mean, feb_max, mar_mean, window")

        win = win.dropna()
        if win.shape[0] >= min_frac * expected and pd.notna(val):
            out[wy] = float(val)

    return pd.Series(out, name="preSPR_storage_af").sort_index()


def _storage_label_tag(method: str, window_days: Optional[int]) -> str:
    return (
        "Feb mean" if method == "feb_mean" else
        "Feb max"  if method == "feb_max"  else
        "Mar mean" if method == "mar_mean" else
        f"last {window_days} days ≤ Feb-end" if method == "window" else method
    )


# ---------------------------------------------------------------------
# Flexible scatter for quick discrimination tests
# ---------------------------------------------------------------------

def scatter_storage_vs_swe(
    wy_metrics: pd.DataFrame,      # from assemble_wy_metrics()
    usbr_df: pd.DataFrame,         # must have 'storage_af'
    spe_df: Optional[pd.DataFrame] = None,
    storage_method: str = "feb_mean",      # "feb_mean" | "feb_max" | "mar_mean" | "window"
    storage_window_days: Optional[int] = None,
    swe_col: str = "SWE_Mar1_mm",          # any wy_metrics column (e.g. SWE_Feb_mean_mm, SWE_peak_mm, ...)
    year_split: int = 2000,                # grey out years before this
    annotate_wy: bool = True,              # label WYs (post-year_split only)
    annotate_fontsize: int = 8,
    point_size: float = 80,
    colors: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    return_points: bool = False,
    # NEW:
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
):
    if colors is None:
        colors = {"pre": "#9ca3af", "no": "#f50b46", "spr": "#2563eb"}
    if swe_col not in wy_metrics.columns:
        raise KeyError(f"{swe_col!r} not in wy_metrics columns: {list(wy_metrics.columns)}")

    x_storage = prespring_storage_by_wy(
        usbr_df, method=storage_method, window_days=storage_window_days, min_frac=0.60
    )
    y_swe = wy_metrics[swe_col].rename("SWE_metric")
    df = pd.concat([x_storage, y_swe], axis=1).dropna()
    if df.empty:
        print("No overlapping WYs between storage metric and SWE metric.")
        return None

    if spe_df is not None and not spe_df.empty and "classified_SPE" in spe_df.columns:
        spr_flags = spe_df["classified_SPE"].reindex(df.index).fillna(False).astype(bool)
    else:
        spr_flags = pd.Series(False, index=df.index)
    df["cat"] = ["pre" if wy < year_split else ("spr" if bool(spr_flags.loc[wy]) else "no") for wy in df.index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
    for cat in ["pre", "no", "spr"]:
        dd = df[df["cat"] == cat]
        if dd.empty:
            continue
        ax.scatter(
            dd["preSPR_storage_af"], dd["SWE_metric"],
            s=point_size, alpha=0.9, color=colors[cat], edgecolor="white", linewidth=1.0,
            label=(f"WY < {year_split}" if cat=="pre" else ("SPR attempted" if cat=="spr" else f"No SPR (WY ≥ {year_split})"))
        )
        if annotate_wy and cat in ("no", "spr"):
            for wy, r in dd.iterrows():
                ax.annotate(
                    str(int(wy)), (r["preSPR_storage_af"], r["SWE_metric"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=annotate_fontsize,
                    color="#334155", path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    annotation_clip=False
                )

    # Labels / title with fallbacks
    ax.set_xlabel(x_label if x_label is not None
                  else f"Pre-Spring storage (af) [{_storage_label_tag(storage_method, storage_window_days)}]")
    ax.set_ylabel(y_label if y_label is not None else swe_col.replace("_", " "))
    ax.set_title(title if title is not None else f"{swe_col} vs pre-Spring storage")

    ax.minorticks_on(); ax.grid(True, alpha=0.25); [ax.spines[s].set_visible(False) for s in ("top","right")]

    handles, labels = ax.get_legend_handles_labels()
    wanted = [f"WY < {year_split}", f"No SPR (WY ≥ {year_split})", "SPR attempted"]
    order = [i for w in wanted for i, l in enumerate(labels) if l == w]
    if order:
        ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, loc="best")

    plt.tight_layout()
    if return_points:
        return df


# ---------------------------------------------------------------------
# Correlation helper (handy while you compare metrics)
# ---------------------------------------------------------------------

def correlations_and_fit(df: pd.DataFrame, y_col: str, x_col: str):
    """
    Return correlation stats + OLS line (in-sample and LOOCV R²/RMSE).
    """
    missing = [c for c in (x_col, y_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    d = df[[x_col, y_col]].dropna().copy()
    if d.empty:
        return None

    pearson = d[x_col].corr(d[y_col], method="pearson")
    spearman = d[x_col].corr(d[y_col], method="spearman")

    a, b = np.polyfit(d[x_col].values, d[y_col].values, 1)
    y_hat = a * d[x_col] + b
    r2_in = 1 - ((d[y_col] - y_hat).pow(2).sum() / ((d[y_col] - d[y_col].mean()).pow(2).sum()))

    # Leave-One-Out CV
    y_cv = pd.Series(index=d.index, dtype=float)
    x = d[x_col].to_numpy(); y = d[y_col].to_numpy()
    for i in range(len(d)):
        m = np.ones(len(d), dtype=bool); m[i] = False
        a_i, b_i = np.polyfit(x[m], y[m], 1)
        y_cv.iloc[i] = a_i * x[i] + b_i
    rmse_cv = float(np.sqrt(((d[y_col] - y_cv) ** 2).mean()))
    r2_cv = 1 - ((d[y_col] - y_cv).pow(2).sum() / ((d[y_col] - d[y_col].mean()).pow(2).sum()))

    return (
        {
            "pearson_r": float(pearson),
            "spearman_rho": float(spearman),
            "ols_slope": float(a),
            "ols_intercept": float(b),
            "r2_in_sample": float(r2_in),
            "r2_loocv": float(r2_cv),
            "rmse_loocv": rmse_cv,
            "n_years": int(len(d)),
        },
        y_hat, y_cv
    )


# ---------------------------------------------------------------------
# Back: timeline plot with shaded SPE years (works with daily or sub-daily)
# ---------------------------------------------------------------------

def _to_daily(x, daily_min_frac: float = 0.80, prefer_col: str = "release_cfs") -> pd.Series:
    """
    Accept Series or DataFrame; returns a daily Series.
    - If sub-daily: resamples to daily mean with coverage rule (>= daily_min_frac * 96).
    - If already daily: aligns to 'D' without filling.
    """
    # pick series
    if isinstance(x, pd.DataFrame):
        if prefer_col in x.columns:
            s = x[prefer_col].copy()
        else:
            numcols = x.select_dtypes(include="number").columns
            if len(numcols) == 1:
                s = x[numcols[0]].copy()
            else:
                raise ValueError(f"plot_spe_timeline: provide a Series or a DataFrame with '{prefer_col}'.")
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        raise TypeError("plot_spe_timeline: x must be a pandas Series or DataFrame.")

    # ensure datetime index
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    if s.empty:
        return s.asfreq("D")

    # cadence detect
    diffs = s.index.to_series().diff().dropna()
    med = diffs.median() if not diffs.empty else pd.Timedelta(days=1)

    if med < pd.Timedelta(hours=12):  # sub-daily -> resample with coverage rule
        cnt = s.resample("D").count()
        mean = s.resample("D").mean()
        good = cnt >= (FREQ_15MIN_PER_DAY * daily_min_frac)
        return mean.where(good)
    else:  # already daily
        return s.asfreq("D")


def plot_spe_timeline(
    x,                        # Series or DataFrame (15-min OR daily)
    spe_df: pd.DataFrame,     # WY-indexed with 'classified_SPE' (can be empty/None)
    prefer_col: str = "release_cfs",
    daily_min_frac: float = 0.80,
    shade_mode: str = "success_spring",    # "success_spring" | "success" | "spring" | "none"
    spring_start: tuple = (3, 1),
    spring_end:   tuple = (7, 31),
    color_shade: str = "#e9c46a",
    shade_alpha: float = 0.22,
    success_color: str = "tab:blue",
    line_color: str = "#264653",
    # NEW: easy time-window controls (override start/end if provided)
    year_min: int | None = None,
    year_max: int | None = None,
    start: str | None = None,      # optional explicit timestamps still supported
    end: str | None = None,
    title: str = "Spring Peak Events"
):
    q_daily = _to_daily(x, daily_min_frac=daily_min_frac, prefer_col=prefer_col)

    # ---- determine plot window
    if year_min is not None:
        plot_start = pd.Timestamp(year_min, 1, 1)
    else:
        plot_start = pd.to_datetime(start) if start is not None else (q_daily.index.min() if not q_daily.empty else pd.Timestamp.today())

    if year_max is not None:
        plot_end = pd.Timestamp(year_max, 12, 31)
    else:
        plot_end = pd.to_datetime(end) if end is not None else (q_daily.index.max() if not q_daily.empty else pd.Timestamp.today())

    # subset to window
    q_daily = q_daily.loc[plot_start:plot_end]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    if not q_daily.dropna().empty:
        ax.plot(q_daily.index, q_daily.values, lw=1.1, color=line_color, alpha=0.85)
    else:
        print("plot_spe_timeline: daily series is empty after preprocessing/windowing.")

    # visible WY range (for shading loops)
    if not q_daily.empty:
        wy_min = int((q_daily.index[0].year + (q_daily.index[0].month >= 10)))
        wy_max = int((q_daily.index[-1].year + (q_daily.index[-1].month >= 10)))
    else:
        if spe_df is not None and not spe_df.empty:
            wy_min, wy_max = int(spe_df.index.min()), int(spe_df.index.max())
        else:
            wy_min = wy_max = plot_start.year

    left_bound, right_bound = plot_start, plot_end

    # ---- shading
    if shade_mode == "spring":
        for wy in range(wy_min, wy_max + 1):
            s = pd.Timestamp(wy, spring_start[0], spring_start[1])
            e = pd.Timestamp(wy, spring_end[0],   spring_end[1])
            l, r = max(s, left_bound), min(e, right_bound)
            if r > l: ax.axvspan(l, r, color=color_shade, alpha=shade_alpha, zorder=0)

    elif shade_mode == "success" and spe_df is not None and not spe_df.empty and "classified_SPE" in spe_df.columns:
        for wy in spe_df.index[spe_df["classified_SPE"] == True].astype(int):
            d0, d1 = pd.Timestamp(wy-1, 10, 1), pd.Timestamp(wy, 9, 30)
            l, r = max(d0, left_bound), min(d1, right_bound)
            if r > l: ax.axvspan(l, r, color=color_shade, alpha=shade_alpha, zorder=0)

    elif shade_mode == "success_spring" and spe_df is not None and not spe_df.empty and "classified_SPE" in spe_df.columns:
        for wy in spe_df.index[spe_df["classified_SPE"] == True].astype(int):
            s = pd.Timestamp(wy, spring_start[0], spring_start[1])
            e = pd.Timestamp(wy, spring_end[0],   spring_end[1])
            l, r = max(s, left_bound), min(e, right_bound)
            if r > l: ax.axvspan(l, r, color=color_shade, alpha=shade_alpha, zorder=0)

    # ---- x-axis ticks (Jan 1 each year in the window)
    years = list(range(plot_start.year, plot_end.year + 1))
    ticks = [pd.Timestamp(y, 1, 1) for y in years]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(y) for y in years])
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")

    # blue labels for successful WYs
    success_set = set()
    if spe_df is not None and not spe_df.empty and "classified_SPE" in spe_df.columns:
        success_set = set(int(y) for y in spe_df.index[spe_df["classified_SPE"] == True])
    for y, lbl in zip(years, ax.get_xticklabels()):
        if y in success_set:
            lbl.set_color(success_color)

    ax.set_ylabel("Q (cfs)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.show()


def detect_spr_absolute(
    x,                                  # Series or DataFrame with releases
    threshold_cfs: float = 3500.0,      # absolute cut
    min_days: int = 1,                  # require ≥ this many days over threshold
    spring_start: tuple = (3, 1),       # (month, day) of WY; use (4,1) if you don't want March early-legs
    spring_end:   tuple = (7, 31),
    prefer_col: str = "release_cfs"
) -> pd.DataFrame:
    """
    Classify WY as SPR-attempted if daily release exceeds `threshold_cfs`
    on at least `min_days` days between spring_start and spring_end.
    Returns WY-indexed DataFrame with 'classified_SPE' plus a few diagnostics.
    """
    q = _to_daily(x, daily_min_frac=0.80, prefer_col=prefer_col)
    if q.empty:
        return pd.DataFrame(columns=["classified_SPE","days_above","first_above","last_above","max_cfs","max_date"])

    wys = (q.index.year + (q.index.month >= 10))
    rows = []
    for wy in range(int(wys.min()), int(wys.max()) + 1):
        s = q.loc[pd.Timestamp(wy, spring_start[0], spring_start[1]) :
                  pd.Timestamp(wy, spring_end[0],   spring_end[1])].dropna()
        if s.empty:
            rows.append(dict(WY=wy, classified_SPE=False, days_above=0,
                             first_above=pd.NaT, last_above=pd.NaT, max_cfs=np.nan, max_date=pd.NaT))
            continue

        above = s > threshold_cfs
        n = int(above.sum())
        first = s.index[above.argmax()] if n > 0 else pd.NaT
        last  = s.index[::-1][above.iloc[::-1].argmax()] if n > 0 else pd.NaT
        mdate = s.idxmax(); mval = float(s.loc[mdate]) if pd.notna(mdate) else np.nan

        rows.append(dict(
            WY=wy,
            classified_SPE=bool(n >= min_days),
            days_above=n,
            first_above=first,
            last_above=last,
            max_cfs=mval,
            max_date=mdate if pd.notna(mdate) else pd.NaT
        ))

    return pd.DataFrame(rows).set_index("WY").sort_index()

def plot_hyperbola_rule(
    x, y, attempted,
    tau: float = 1.0,
    S_ref: float = None,
    P_ref: float = None,
    annotate: bool = False,
    labels=None,              # Index/Series/list/array; default will use x.index
    ax=None,
    x_label: str = "Pre-Spring storage (af) [Feb mean]",
    y_label: str = "SWE metric (mm)",
    title: str = "Hyperbola go/no-go",
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    # --- Align everything on a single index ---
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    attempted = pd.Series(attempted, dtype="boolean")

    df = pd.concat(
        {"x": x, "y": y, "attempted": attempted},
        axis=1
    ).dropna(subset=["x", "y", "attempted"])

    # Labels handling (robust)
    if labels is None:
        lab = pd.Series(df.index.astype(str), index=df.index)
    elif isinstance(labels, pd.Series):
        lab = labels.astype(str).reindex(df.index)
    else:
        # list/array/Index → align by position to df.index
        lab = pd.Series(list(labels), index=df.index).astype(str)

    # Defaults for scales from positives if available, else global
    if S_ref is None:
        S_ref = float(np.median(df.loc[df.attempted.astype(bool), "x"])) \
                if (df.attempted.astype(bool)).any() else float(np.median(df["x"]))
    if P_ref is None:
        P_ref = float(np.median(df.loc[df.attempted.astype(bool), "y"])) \
                if (df.attempted.astype(bool)).any() else float(np.median(df["y"]))

    # Index & predictions
    I = (df["x"] / S_ref) * (df["y"] / P_ref)
    pred = I >= tau
    true = df["attempted"].astype(bool)

    tp = int(( pred &  true).sum())
    tn = int((~pred & ~true).sum())
    fp = int(( pred & ~true).sum())
    fn = int((~pred &  true).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(df.loc[~true, "x"], df.loc[~true, "y"], s=80,
               color="#f50b46", edgecolor="white", linewidth=1.0,
               label="No SPR (WY ≥ 2000)")
    ax.scatter(df.loc[ true, "x"], df.loc[ true, "y"], s=80,
               color="#2563eb", edgecolor="white", linewidth=1.0,
               label="SPR attempted")

    if annotate:
        for xx, yy, labtxt in zip(df["x"], df["y"], lab):
            ax.annotate(str(labtxt), (xx, yy), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color="#334155",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                        annotation_clip=False)

    # Boundary: y = (tau * S_ref * P_ref) / x
    xs = np.linspace(float(df["x"].min()*0.95), float(df["x"].max()*1.05), 400)
    ys = (tau * S_ref * P_ref) / xs
    ax.plot(xs, ys, color="#111827", lw=2.0, alpha=0.9,
            label=fr"$y=\frac{{\tau S_{{ref}} P_{{ref}}}}{{x}}$,  $\tau={tau:.2f}$")

    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_title(f"{title}\nacc={acc:.2f}  tp={tp} tn={tn} fp={fp} fn={fn}")
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best"); plt.tight_layout()

    plt.show()

    return {"tau": tau, "S_ref": S_ref, "P_ref": P_ref, "acc": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def plot_sigmoid_rule(
    x, y, attempted,
    s: float = 0.12,           # single knob: steepness (in "x" units)
    x0: float = None,          # knee (default=median x of blue, else median x)
    y_high: float = None,      # upper asymptote (default=85th pct of y)
    y_low: float = None,       # lower asymptote (default=15th pct of y)
    annotate: bool = False,
    labels=None,
    ax=None,
    x_label="Pre-Spring storage (af) — February mean",
    y_label="SWE metric (mm)",
    title="Sideways-sigmoid go/no-go",
):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    # Align on one index
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    attempted = pd.Series(attempted).astype(bool)
    df = pd.concat({"x": x, "y": y, "attempted": attempted}, axis=1).dropna()

    # Defaults: robust, label-aware anchors
    if x0 is None:
        x0 = float(np.median(df.loc[df.attempted, "x"])) if df.attempted.any() else float(np.median(df["x"]))
    if y_high is None:
        y_high = float(np.percentile(df["y"], 85))
    if y_low is None:
        y_low  = float(np.percentile(df["y"], 15))

    # Boundary
    def y_boundary(xx):
        return y_low + (y_high - y_low) / (1.0 + np.exp((xx - x0) / s))

    # Predictions
    yhat = y_boundary(df["x"].values)
    pred = df["y"].values >= yhat
    true = df["attempted"].values

    tp = int(( pred &  true).sum())
    tn = int((~pred & ~true).sum())
    fp = int(( pred & ~true).sum())
    fn = int((~pred &  true).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df.loc[~df.attempted, "x"], df.loc[~df.attempted, "y"], s=80,
               color="#f50b46", edgecolor="white", linewidth=1.0, label="No SPR (WY ≥ 2000)")
    ax.scatter(df.loc[ df.attempted, "x"], df.loc[ df.attempted, "y"], s=80,
               color="#2563eb", edgecolor="white", linewidth=1.0, label="SPR attempted")

    if annotate:
        if labels is None:
            labels = df.index.astype(str)
        elif not isinstance(labels, pd.Series):
            labels = pd.Series(labels, index=df.index)
        for xx, yy, lab in zip(df["x"], df["y"], labels):
            ax.annotate(str(lab), (xx, yy), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color="#334155",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                        annotation_clip=False)

    xs = np.linspace(float(df["x"].min()*0.95), float(df["x"].max()*1.05), 500)
    ax.plot(xs, y_boundary(xs), color="#111827", lw=2.0, alpha=0.95,
            label=fr"$y=y_{{low}}+\frac{{y_{{high}}-y_{{low}}}}{{1+e^{{(x-x_0)/s}}}}$")

    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_title(f"{title}\nacc={acc:.2f}  tp={tp} tn={tn} fp={fp} fn={fn}  |  x0={x0:.0f}, s={s:.3f}")
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best"); plt.tight_layout()

    return {"x0": x0, "s": s, "y_high": y_high, "y_low": y_low, "acc": acc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

from dataclasses import dataclass
import numpy as np
import pandas as pd

# --- your boundary params & helpers (reuse from earlier) ---
@dataclass
class SigmoidRuleParams:
    x0: float     # knee (af)
    s: float      # steepness (af)
    y_low: float  # lower asymptote (mm)
    y_high: float # upper asymptote (mm)

def sigmoid_boundary_y(x, p: SigmoidRuleParams):
    x = np.asarray(x, dtype="float64")
    return p.y_low + (p.y_high - p.y_low) / (1.0 + np.exp((x - p.x0) / p.s))

def sigmoid_margin(x, y, p: SigmoidRuleParams):
    return np.asarray(y, dtype="float64") - sigmoid_boundary_y(x, p)

# --- opportunity index mapping ---
def beta_from_target(omega_on_line=0.75, oi_at_pos_m0=0.90):
    """Choose beta so OI(+m0)=oi_at_pos_m0 (piecewise-exp mapper)."""
    return -np.log((1.0 - oi_at_pos_m0) / (1.0 - omega_on_line))

def opportunity_index_from_margin(margin, m0=None, beta=1.0, omega_on_line=0.75):
    """
    Piecewise-exponential OI in [0,1]. margin in mm; positive is GO side.
    m0: scale in mm (float). If None, uses robust scale from data (MAD or 1.0 to avoid div0).
    """
    m = np.asarray(margin, dtype="float64")
    if m0 is None:
        med = np.nanmedian(m)
        mad = np.nanmedian(np.abs(m - med))
        m0 = mad if (mad > 0) else (np.nanpercentile(np.abs(m), 67) or 1.0)
    mh = np.clip(m / m0, -50.0, 50.0)  # numeric safety
    oi = np.empty_like(mh)
    pos = mh >= 0
    oi[pos]  = 1.0 - (1.0 - omega_on_line) * np.exp(-beta * mh[pos])
    oi[~pos] = omega_on_line * np.exp( beta * mh[~pos])
    return np.clip(oi, 0.0, 1.0), m0

# --- convenience wrapper for your sigmoid boundary ---
def sigmoid_opportunity_index(x, y, params: SigmoidRuleParams,
                              m0=None, beta=1.0, omega_on_line=0.75,
                              return_margin=False):
    m = sigmoid_margin(x, y, params)
    oi, m0_used = opportunity_index_from_margin(m, m0=m0, beta=beta, omega_on_line=omega_on_line)
    return (oi, m, m0_used) if return_margin else oi


def plot_oi_scatter(
    x, y, params: SigmoidRuleParams,
    attempted=None,                  # optional bool Series (blue/red labels)
    m0=None, beta=1.0, omega_on_line=0.75,
    annotate=False, labels=None,
    show_boundary=True, contour_levels=(0.5, 0.75, 0.9),
    cmap="viridis", vmin=0.0, vmax=1.0,
    x_label="Pre-Spring storage (af) — February mean",
    y_label="SWE peak through Mar 1 (mm)",
    title="Opportunity Index map",
    ax=None,
):
    """
    Color each point by Opportunity Index (OI in [0,1]) computed from
    vertical margin to the sideways-sigmoid boundary.

    Returns dict with df (x,y,oi,margin), and the m0 actually used.
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    # --- align everything on one index ---
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    df = pd.concat({"x": x, "y": y}, axis=1).dropna()

    if attempted is not None:
        attempted = pd.Series(attempted).astype(bool).reindex(df.index)
        df["attempted"] = attempted
    else:
        df["attempted"] = np.nan

    if labels is None:
        lab = pd.Series(df.index.astype(str), index=df.index)
    elif isinstance(labels, pd.Series):
        lab = labels.reindex(df.index).astype(str)
    else:
        lab = pd.Series(list(labels), index=df.index).astype(str)

    # --- compute margin & OI ---
    margin = y.values - sigmoid_boundary_y(x.values, params)
    oi, m0_used = opportunity_index_from_margin(margin, m0=m0, beta=beta, omega_on_line=omega_on_line)
    df["oi"] = oi
    df["margin_mm"] = margin

    # --- plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.2, 6))

    # sort so low-OI points don't hide high-OI points
    df = df.sort_values("oi")

    sc = ax.scatter(df["x"], df["y"], s=85, c=df["oi"],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolor="white", linewidth=0.9)

    # Optional labels
    if annotate:
        for xx, yy, labtxt in zip(df["x"], df["y"], lab):
            ax.annotate(str(labtxt), (xx, yy), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color="#334155",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                        annotation_clip=False)

    # Boundary curve
    if show_boundary:
        xs = np.linspace(float(df["x"].min()*0.95), float(df["x"].max()*1.05), 500)
        ax.plot(xs, sigmoid_boundary_y(xs, params), color="#111827", lw=2.0, alpha=0.95, label="boundary")

    # Optional iso-OI contours
    if contour_levels:
        xs = np.linspace(float(df["x"].min()*0.95), float(df["x"].max()*1.05), 220)
        ys = np.linspace(float(df["y"].min()*0.95), float(df["y"].max()*1.05), 220)
        Xg, Yg = np.meshgrid(xs, ys)
        Mg = Yg - sigmoid_boundary_y(Xg, params)
        OIg, _ = opportunity_index_from_margin(Mg, m0=(m0 if m0 is not None else m0_used),
                                               beta=beta, omega_on_line=omega_on_line)
        cs = ax.contour(Xg, Yg, OIg, levels=contour_levels, colors="#111827", linewidths=1.0, alpha=0.6)
        ax.clabel(cs, fmt=lambda v: f"OI={v:.2f}", inline=True, fontsize=8)

    # Axis / colorbar
    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Opportunity Index")
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if show_boundary: ax.legend(frameon=False, loc="best")
    plt.tight_layout()

    return {"df": df, "m0_used": float(m0_used)}


def plot_oi_field(
    x, y, params: SigmoidRuleParams,
    attempted=None, labels=None, annotate=False,
    m0=None, beta=1.0, omega_on_line=0.75,
    gridsize=(450, 450), pad_frac=0.05,
    xlim=None, ylim=None,
    cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.90,
    # --- NEW presentation knobs ---
    legend_loc: str = "lower left",
    legend_fontsize: int = 10,
    axis_labelsize: int = 14,
    tick_labelsize: int | None = None,
    cbar: bool = True,
    cbar_kw: dict | None = None,
    cbar_label: str = "Opportunity Index",
    cbar_labelsize: int = 14,
    title: str | None = None,                 # None → no title
    boundary_label: str | None = None,        # default shows OI value
    # label styling from previous step (if you added them)
    label_fontsize: int = 11,
    label_weight: str = "bold",
    label_color: str = "black",
    label_outline_color: str = "white",
    label_outline_width: float = 3.0,
    show_boundary=True, scatter=True, scatter_size=85,
    x_label="Pre-Spring storage (af) — February mean",
    y_label="SWE peak through Mar 1 (mm)",
    ax=None,
):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    df = pd.concat({"x": x, "y": y}, axis=1).dropna()

    if attempted is not None:
        attempted = pd.Series(attempted).astype(bool).reindex(df.index)
        df["attempted"] = attempted
    else:
        df["attempted"] = np.nan

    if labels is None:
        lab = pd.Series(df.index.astype(str), index=df.index)
    elif isinstance(labels, pd.Series):
        lab = labels.reindex(df.index).astype(str)
    else:
        lab = pd.Series(list(labels), index=df.index).astype(str)

    margin_pts = df["y"].values - sigmoid_boundary_y(df["x"].values, params)
    oi_pts, m0_used = opportunity_index_from_margin(margin_pts, m0=m0, beta=beta, omega_on_line=omega_on_line)
    df["oi"] = oi_pts; df["margin_mm"] = margin_pts

    if xlim is None or ylim is None:
        xmin, xmax = float(df["x"].min()), float(df["x"].max())
        ymin, ymax = float(df["y"].min()), float(df["y"].max())
        px = (xmax - xmin) * pad_frac or (0.01 * max(1.0, xmax))
        py = (ymax - ymin) * pad_frac or (0.01 * max(1.0, ymax))
        xmin, xmax, ymin, ymax = xmin - px, xmax + px, ymin - py, ymax + py
    else:
        xmin, xmax = xlim; ymin, ymax = ylim

    nx, ny = int(gridsize[0]), int(gridsize[1])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    Xg, Yg = np.meshgrid(xs, ys)
    Mg = Yg - sigmoid_boundary_y(Xg, params)
    OIg, _ = opportunity_index_from_margin(Mg, m0=(m0 if m0 is not None else m0_used),
                                           beta=beta, omega_on_line=omega_on_line)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.6, 6.4))

    im = ax.imshow(
        OIg, origin="lower", extent=[xmin, xmax, ymin, ymax],
        aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation="bilinear"
    )
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    if show_boundary:
        xb = np.linspace(xmin, xmax, 700)
        ax.plot(
            xb, sigmoid_boundary_y(xb, params),
            color="#111827", lw=2.0, alpha=0.95,
            label=(boundary_label if boundary_label is not None else f"OI = {omega_on_line:g}")
        )

    if scatter:
        df_plot = df.sort_values("oi")
        if "attempted" in df_plot and df_plot["attempted"].notna().any():
            ms = scatter_size
            dd_no  = df_plot[df_plot["attempted"] == False]
            dd_yes = df_plot[df_plot["attempted"] == True]
            if not dd_no.empty:
                ax.scatter(dd_no["x"], dd_no["y"], s=ms, color="#f50b46", edgecolor="white", linewidth=0.9, label="No SPR")
            if not dd_yes.empty:
                ax.scatter(dd_yes["x"], dd_yes["y"], s=ms, color="#2563eb", edgecolor="white", linewidth=0.9, label="SPR attempted")
        else:
            ax.scatter(df_plot["x"], df_plot["y"], s=scatter_size, c=df_plot["oi"], cmap=cmap,
                       vmin=vmin, vmax=vmax, edgecolor="white", linewidth=0.9)

        if annotate:
            for xx, yy, labtxt in zip(df_plot["x"], df_plot["y"], lab.loc[df_plot.index]):
                ax.annotate(
                    str(labtxt), (xx, yy), xytext=(6, 6), textcoords="offset points",
                    fontsize=label_fontsize, fontweight=label_weight, color=label_color,
                    path_effects=[pe.withStroke(linewidth=label_outline_width,
                                                foreground=label_outline_color)],
                    annotation_clip=False, zorder=6
                )

    # axis labels & (optional) ticks size
    ax.set_xlabel(x_label, fontsize=axis_labelsize)
    ax.set_ylabel(y_label, fontsize=axis_labelsize)
    if tick_labelsize is not None:
        ax.tick_params(labelsize=tick_labelsize)

    # title (optional)
    if title:
        ax.set_title(title)

    # colorbar with bigger label
    if cbar:
        kw = dict(pad=0.01)
        if cbar_kw: kw.update(cbar_kw)
        cb = plt.colorbar(im, ax=ax, **kw)
        cb.set_label(cbar_label, fontsize=cbar_labelsize)
        if tick_labelsize is not None:
            cb.ax.tick_params(labelsize=tick_labelsize)

    # legend in lower-left (or wherever you set legend_loc)
    if show_boundary or (attempted is not None):
        ax.legend(frameon=False, loc=legend_loc, fontsize=legend_fontsize)

    plt.tight_layout()
    return {"df": df, "m0_used": float(m0_used), "beta_used": float(beta)}
