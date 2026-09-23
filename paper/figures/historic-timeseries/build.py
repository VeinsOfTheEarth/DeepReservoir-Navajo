"""Combined historical/training time-basis figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent / "figure-support"))
from figurestyle import add_panel_label
from common import (
    BLUE,
    CFS_DAY_TO_AF,
    COMMON_X_END,
    COMMON_X_START,
    GRAY,
    GREEN,
    ORANGE,
    PURPLE,
    SELECTED_EVAL_END,
    SELECTED_EVAL_START,
    SELECTED_TRAIN_END,
    SELECTED_TRAIN_START,
    TEXT,
    add_train_eval_shading,
    load_raw_model_data,
    save_figure,
    selected_policy_train_eval_split,
    set_common_time_axis,
    set_theme,
    soften_axes,
)

from deepreservoir.define_env.spring_peak_release_curve import SpringPeakReleaseCurve
from deepreservoir.data.storage_datum import apply_storage_datum_mode
from deepreservoir.drl.niip_targets import (
    historic_niip_delivery_target_for_dates,
    load_historic_niip_delivery_series,
)


OUTPUT_DIR = THIS_DIR
SPR_START = pd.Timestamp("2000-01-01")
THRESHOLDS = [
    (2500.0, 10, "#C4B5FD"),
    (5000.0, 21, "#A78BFA"),
    (8000.0, 10, PURPLE),
    (10000.0, 5, "#4C1D95"),
]


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    add_panel_label(ax, label)


def _calendar_year_midpoint(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=7, day=1)


def _complete_calendar_year_totals(df: pd.DataFrame) -> pd.Series:
    rows: dict[int, float] = {}
    for year in sorted({int(y) for y in df.index.year}):
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
        if df.index.min() <= start and df.index.max() >= end:
            mask = df.index.year == year
            expected = pd.date_range(start, end, freq="D")
            actual = df.index[mask]
            if len(actual) != len(expected) or not bool(expected.isin(actual).all()):
                continue
            inflow = pd.to_numeric(df.loc[mask, "inflow_cfs"], errors="coerce")
            if not bool(np.isfinite(inflow.to_numpy(dtype=float)).all()):
                continue
            rows[year] = float(
                inflow.sum()
                * CFS_DAY_TO_AF
                / 1_000_000.0
            )
    return pd.Series(rows, name="inflow_maf")


def _spr_yearly_counts(raw: pd.DataFrame) -> pd.DataFrame:
    q = pd.to_numeric(raw["sj_farmington_q_cfs"], errors="coerce")
    curve = SpringPeakReleaseCurve()
    spr_target = curve.targets_for_date_index(pd.DatetimeIndex(raw.index))
    in_window = spr_target > 0.0
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for year in sorted({int(y) for y in raw.index.year}):
        start = pd.Timestamp(year=year, month=5, day=9)
        end = pd.Timestamp(year=year, month=6, day=25)
        if raw.index.min() > start or raw.index.max() < end:
            continue
        mask = (raw.index.year == year) & in_window
        row: dict[str, float | int | pd.Timestamp] = {
            "year": year,
            "date": pd.Timestamp(year=year, month=7, day=1),
        }
        for threshold, _, _ in THRESHOLDS:
            row[f"days_ge_{int(threshold)}"] = int((q.loc[mask] >= threshold).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def _add_common_context(ax: plt.Axes, *, show_labels: bool = False) -> None:
    add_train_eval_shading(ax, label=False)
    ax.axvline(SPR_START, color=GRAY, lw=1.0, ls=":", alpha=0.82)
    if show_labels:
        ax.text(
            SELECTED_TRAIN_START + (SELECTED_TRAIN_END - SELECTED_TRAIN_START) * 0.46,
            1.035,
            "Training",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=BLUE,
        )
        ax.text(
            SELECTED_EVAL_START + (SELECTED_EVAL_END - SELECTED_EVAL_START) * 0.5,
            1.035,
            "Evaluation",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=ORANGE,
        )
        ax.text(
            SPR_START,
            0.97,
            "Post-recommendation period",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            rotation=90,
            fontsize=7.4,
            color=GRAY,
        )


def build() -> list[Path]:
    set_theme()
    raw, _ = apply_storage_datum_mode(load_raw_model_data(), mode="elevation_2019")
    train, eval_df = selected_policy_train_eval_split(raw)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(8.4, 8.9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 0.86, 0.9, 1.0]},
    )

    for i, ax in enumerate(axes):
        _add_common_context(ax, show_labels=(i == 0))
        ax.set_axisbelow(True)
        soften_axes(ax)
    for ax, label in zip(axes, ["A", "B", "C", "D"]):
        _add_panel_label(ax, label)

    storage = pd.to_numeric(raw["storage_af"], errors="coerce") / 1_000_000.0
    axes[0].plot(raw.index, storage, color=BLUE, lw=1.15, zorder=5)
    axes[0].set_ylabel("Storage\n(MAF)")

    train_annual = _complete_calendar_year_totals(train)
    eval_annual = _complete_calendar_year_totals(eval_df)
    train_mean = float(train_annual.mean())
    eval_mean = float(eval_annual.mean())
    train_dates = [_calendar_year_midpoint(int(year)) for year in train_annual.index]
    eval_dates = [_calendar_year_midpoint(int(year)) for year in eval_annual.index]
    axes[1].bar(
        train_dates,
        train_annual.to_numpy(dtype=float),
        color="#CBD5E1",
        edgecolor="#94A3B8",
        linewidth=0.3,
        width=250,
    )
    axes[1].bar(
        eval_dates,
        eval_annual.to_numpy(dtype=float),
        color="#FDBA74",
        edgecolor=ORANGE,
        linewidth=0.35,
        width=250,
    )
    axes[1].hlines(
        train_mean,
        _calendar_year_midpoint(int(train_annual.index.min())) - pd.Timedelta(days=180),
        _calendar_year_midpoint(int(train_annual.index.max())) + pd.Timedelta(days=180),
        color=BLUE,
        lw=0.95,
        ls="--",
        alpha=0.78,
    )
    axes[1].hlines(
        eval_mean,
        _calendar_year_midpoint(int(eval_annual.index.min())) - pd.Timedelta(days=180),
        _calendar_year_midpoint(int(eval_annual.index.max())) + pd.Timedelta(days=180),
        color=ORANGE,
        lw=0.95,
        ls="--",
        alpha=0.78,
    )
    axes[1].text(
        pd.Timestamp("1969-01-01"),
        train_mean + 0.055,
        f"{train_mean:.2f} MAF/yr",
        color=BLUE,
        fontsize=8.2,
        ha="left",
        va="bottom",
    )
    axes[1].annotate(
        f"{eval_mean:.2f} MAF/yr",
        xy=(pd.Timestamp("2016-06-01"), eval_mean),
        xytext=(pd.Timestamp("2014-03-01"), eval_mean + 0.42),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 0.75},
        color=ORANGE,
        fontsize=8.2,
        ha="left",
        va="bottom",
    )
    axes[1].set_ylabel("Inflow\n(MAF/yr)")
    axes[1].set_ylim(0, max(float(train_annual.max()), float(eval_annual.max())) * 1.18)

    demand = load_historic_niip_delivery_series()
    demand = pd.to_numeric(demand, errors="coerce").dropna().clip(lower=0.0)
    demand = demand.loc[:COMMON_X_END]
    record_start = pd.Timestamp(demand.index.min()) if not demand.empty else pd.NaT
    backfill_idx = pd.date_range(COMMON_X_START, record_start - pd.Timedelta(days=1), freq="D")
    backfill = (
        historic_niip_delivery_target_for_dates(
            backfill_idx,
            fallback_mode="training_only",
        )
        if len(backfill_idx)
        else pd.Series(dtype=float)
    )
    backfill_smoothed = backfill.rolling(30, center=True, min_periods=7).mean()
    if not backfill.empty:
        axes[2].plot(
            backfill_smoothed.index,
            backfill_smoothed.to_numpy(dtype=float),
            color=GRAY,
            lw=1.0,
            ls="--",
            alpha=0.78,
            label="Training-period median backfill",
            zorder=4,
        )
    axes[2].plot(
        demand.index,
        demand.to_numpy(dtype=float),
        color=GREEN,
        lw=0.72,
        alpha=0.7,
        label="Daily delivery",
        zorder=3,
    )
    if not demand.empty:
        axes[2].axvline(record_start, color=GRAY, lw=0.9, ls=":", alpha=0.82)
    axes[2].set_ylabel("NIIP delivery proxy\n(cfs)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_yticks([0, 300, 600, 900, 1200])
    axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
        ncol=2,
        fontsize=7.8,
        handlelength=1.45,
        columnspacing=0.95,
    )

    counts = _spr_yearly_counts(raw)
    offsets = [-75, -25, 25, 75]
    for offset, (threshold, required_days, color) in zip(offsets, THRESHOLDS):
        key = f"days_ge_{int(threshold)}"
        axes[3].bar(
            counts["date"] + pd.to_timedelta(offset, unit="D"),
            counts[key],
            color=color,
            edgecolor=color,
            linewidth=0.3,
            width=42,
            alpha=0.72,
            label=f"{int(threshold):,} cfs ({required_days} d)",
        )
    axes[3].set_ylabel("SPR days\nabove threshold")
    axes[3].set_ylim(0, 55)
    axes[3].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
        ncol=4,
        fontsize=7.7,
        handlelength=1.15,
        columnspacing=0.8,
    )
    axes[3].set_xlabel("Date")

    for ax in axes:
        set_common_time_axis(ax)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    return save_figure(fig, OUTPUT_DIR, "historic-timeseries")


def main() -> None:
    for path in build():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
