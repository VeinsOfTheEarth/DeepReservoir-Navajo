"""Shared helpers for selected-policy comparison paper figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from figurestyle import (
    OBJECTIVE_COLORS,
    add_panel_label,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepreservoir.drl import selected_policy
from deepreservoir.drl.metrics import compute_historic_summary_metrics
from deepreservoir.data.metadata import project_metadata
from deepreservoir.define_env.storage_elevation.thresholds import get_navajo_storage_thresholds


METADATA = project_metadata()

SELECTED_COLOR = "#2B6CB0"
SELECTED_FILL = "#BBD7F0"
HIST_COLOR = "#8B95A5"
HIST_FILL = "#E5E7EB"
TARGET_COLOR = "#C2410C"
NIIP_COLOR = OBJECTIVE_COLORS["niip"]
NIIP_FILL = "#BBEBCF"
HYDRO_COLOR = OBJECTIVE_COLORS["hydropower"]
HYDRO_FILL = "#FED7AA"
SPR_COLOR = OBJECTIVE_COLORS["spr"]
STORAGE_COLOR = OBJECTIVE_COLORS["storage"]
ESA_COLOR = OBJECTIVE_COLORS["esa"]
FLOOD_COLOR = OBJECTIVE_COLORS["flood"]
DAM_COLOR = OBJECTIVE_COLORS["dam_safety"]
SPR_FILL = "#DDD6FE"
SPR_DARK = "#5B21B6"
SPR_ANIMAS_FILL = "#F4A261"
SPR_NAVAJO_FILL = "#93C5FD"
GRID = "#E2E8F0"
SPINE = "#94A3B8"
TEXT = "#111827"
MUTED = "#6B7280"
BACKGROUND = "#F4F7FA"

SPR_THRESHOLD_SPECS = (
    (10_000.0, 5, 0.20),
    (8_000.0, 10, 0.33),
    (5_000.0, 21, 0.50),
    (2_500.0, 10, 0.80),
)

SPR_SPECS = (
    ("10,000 cfs / 5 d", "spr_freq_years_meeting_10000cfs_5d", "spr_target_frequency_10000cfs_5d"),
    ("8,000 cfs / 10 d", "spr_freq_years_meeting_8000cfs_10d", "spr_target_frequency_8000cfs_10d"),
    ("5,000 cfs / 21 d", "spr_freq_years_meeting_5000cfs_21d", "spr_target_frequency_5000cfs_21d"),
    ("2,500 cfs / 10 d", "spr_freq_years_meeting_2500cfs_10d", "spr_target_frequency_2500cfs_10d"),
)


def set_theme() -> None:
    font_dir = REPO_ROOT / "assets" / "fonts"
    if font_dir.is_dir():
        for font_path in font_dir.rglob("*.ttf"):
            try:
                fm.fontManager.addfont(str(font_path))
            except Exception:
                pass
    preferred = ["Inter", "Source Sans 3", "IBM Plex Sans", "DejaVu Sans"]
    available: list[str] = []
    for name in preferred:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            available.append(name)
        except Exception:
            pass
    if "DejaVu Sans" not in available:
        available.append("DejaVu Sans")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": available,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.72,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.4,
            "savefig.bbox": "tight",
        }
    )


def soften_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.9)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, dpi: int = 320) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem.replace("_", "-")
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def load_selected_rollout() -> pd.DataFrame:
    df = pd.read_parquet(selected_policy.SELECTED_EVAL_ROLLOUT_PATH)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def load_metric_rows() -> tuple[dict[str, float], dict[str, float]]:
    with selected_policy.SELECTED_EVAL_METRICS_JSON_PATH.open("r", encoding="utf-8") as f:
        selected = dict(json.load(f))
    historic = compute_historic_summary_metrics(load_selected_rollout())
    return selected, historic


def _load_navajo_historical_min_storage_af() -> float | None:
    try:
        df = pd.read_csv(
            METADATA.path("daily.reservoir"),
            usecols=["Storage (af)"],
        )
    except Exception:
        return None

    storage = pd.to_numeric(df["Storage (af)"], errors="coerce").dropna()
    if storage.empty:
        return None
    return float(storage.min())


def plot_storage_comparison(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    selected = pd.to_numeric(df["storage_agent_af_end"], errors="coerce") / 1_000_000.0
    historic = pd.to_numeric(df["storage_hist_af"], errors="coerce") / 1_000_000.0

    ax.plot(df.index, historic, color=HIST_COLOR, lw=1.15, ls=(0, (1.0, 1.1)), label="Historic management")
    ax.plot(df.index, selected, color=SELECTED_COLOR, lw=1.45, label="Selected policy")
    ax.set_ylabel("Storage (MAF)")
    ax.set_xlim(df.index.min(), df.index.max())
    ax.margins(x=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.tick_params(axis="x", which="minor", length=3.0, color=SPINE)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#D6DEE6", ncol=2)
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)


def plot_release_scatter(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    x = pd.to_numeric(df["release_cfs"], errors="coerce")
    y = pd.to_numeric(df["release_sj_main_cfs"], errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid].to_numpy(dtype=float)
    y = y[valid].to_numpy(dtype=float)
    max_v = float(np.nanmax([np.nanmax(x), np.nanmax(y), 1.0]))
    lim = max(5200.0, np.ceil(max_v / 500.0) * 500.0)

    ax.scatter(
        x,
        y,
        s=8.0,
        color=SELECTED_COLOR,
        alpha=0.18,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1.0, ls="--", alpha=0.85)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Historic release (cfs)")
    ax.set_ylabel("Selected-policy release (cfs)")
    ax.text(
        0.97,
        0.05,
        "Dashed line = 1:1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8.2,
    )
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)


def plot_spr_frequencies(
    ax: plt.Axes,
    selected_metrics: dict[str, float],
    historic_metrics: dict[str, float],
    *,
    panel: str | None = None,
) -> None:
    labels = [spec[0].replace(",000 cfs / ", "k\n").replace("2,500 cfs / ", "2.5k\n") for spec in SPR_SPECS]
    selected = np.asarray([selected_metrics[spec[1]] for spec in SPR_SPECS], dtype=float) * 100.0
    historic = np.asarray([historic_metrics[spec[1]] for spec in SPR_SPECS], dtype=float) * 100.0
    target = np.asarray([selected_metrics[spec[2]] for spec in SPR_SPECS], dtype=float) * 100.0

    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, historic, width=width, color=HIST_FILL, edgecolor=HIST_COLOR, lw=0.8, label="Historic")
    ax.bar(x + width / 2, selected, width=width, color=SPR_FILL, edgecolor=SPR_COLOR, lw=0.9, label="Selected policy")
    ax.scatter(x, target, marker="D", s=34, color=SPR_DARK, zorder=4, label="Prescribed frequency")
    for xi, yi in zip(x, target):
        ax.plot([xi - 0.36, xi + 0.36], [yi, yi], color=SPR_DARK, lw=0.75, alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Met target frequency (%)")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.10, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
    )
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)


def _climatology(values: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(values, errors="coerce")
    idx = pd.DatetimeIndex(values.index)
    pseudo = pd.to_datetime(
        {"year": np.full(len(idx), 2001), "month": idx.month, "day": idx.day},
        errors="coerce",
    )
    tmp = pd.DataFrame({"date": pseudo, "value": values.to_numpy(dtype=float)})
    tmp = tmp[tmp["date"].notna() & tmp["value"].notna()]
    grouped = tmp.groupby("date")["value"]
    return grouped.quantile([0.25, 0.5, 0.75]).unstack().rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})


def _plot_climatology_pair(
    ax: plt.Axes,
    selected: pd.Series,
    historic: pd.Series,
    *,
    selected_label: str = "Selected policy",
    historic_label: str = "Historic management",
) -> None:
    selected_stats = _climatology(selected)
    historic_stats = _climatology(historic)

    ax.fill_between(
        historic_stats.index,
        historic_stats["q25"].to_numpy(dtype=float),
        historic_stats["q75"].to_numpy(dtype=float),
        color=HIST_FILL,
        alpha=0.72,
        linewidth=0,
    )
    ax.plot(
        historic_stats.index,
        historic_stats["median"],
        color=HIST_COLOR,
        lw=1.2,
        ls=(0, (1.0, 1.1)),
        label=historic_label,
    )
    ax.fill_between(
        selected_stats.index,
        selected_stats["q25"].to_numpy(dtype=float),
        selected_stats["q75"].to_numpy(dtype=float),
        color=SELECTED_FILL,
        alpha=0.70,
        linewidth=0,
    )
    ax.plot(
        selected_stats.index,
        selected_stats["median"],
        color=SELECTED_COLOR,
        lw=1.45,
        label=selected_label,
    )
    ax.set_xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2001-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=2.8, color=SPINE)
    soften_axes(ax)


def _threshold_colors() -> dict[int, str]:
    return {
        10_000: MUTED,
        8_000: MUTED,
        5_000: MUTED,
        2_500: MUTED,
    }


def _threshold_styles() -> dict[int, tuple[int, tuple[int, int]]]:
    return {
        10_000: (0, (5, 3)),
        8_000: (0, (4, 3)),
        5_000: (0, (3, 3)),
        2_500: (0, (2, 3)),
    }


def _spring_marker_position(year: int, available_index: pd.DatetimeIndex) -> pd.Timestamp:
    marker_date = pd.Timestamp(year=int(year), month=5, day=22)
    if available_index.min() <= marker_date <= available_index.max():
        return marker_date
    return pd.Timestamp(available_index.min() + (available_index.max() - available_index.min()) / 2)


def _spr_threshold_year_markers(
    index: pd.DatetimeIndex,
    metric_proxy: pd.Series,
    threshold_specs: tuple[tuple[float, int, float], ...] = SPR_THRESHOLD_SPECS,
) -> list[tuple[pd.Timestamp, float, int]]:
    metric_proxy = pd.to_numeric(metric_proxy.reindex(index), errors="coerce")
    markers: list[tuple[pd.Timestamp, float, int]] = []
    for year in sorted(set(index.year)):
        spring_start = pd.Timestamp(year=int(year), month=5, day=9)
        spring_end = pd.Timestamp(year=int(year), month=6, day=25)
        g_idx = index[(index >= spring_start) & (index <= spring_end)]
        if len(g_idx) == 0:
            continue
        marker_x = _spring_marker_position(int(year), g_idx)
        for threshold_cfs, duration_days, _target_frequency in threshold_specs:
            threshold = float(threshold_cfs)
            days_above = int((metric_proxy.loc[g_idx] >= threshold).sum())
            if days_above >= int(duration_days):
                markers.append((marker_x, threshold, int(round(threshold))))
    return markers


def selected_farmington_total(df: pd.DataFrame) -> pd.Series:
    if "spr_proxy_farmington_after_release_cfs" in df.columns:
        return pd.to_numeric(df["spr_proxy_farmington_after_release_cfs"], errors="coerce")
    animas = pd.to_numeric(df["animas_farmington_q_cfs"], errors="coerce").fillna(0.0)
    release = pd.to_numeric(df["release_sj_main_cfs"], errors="coerce").fillna(0.0)
    return animas + release


def historical_farmington_total(df: pd.DataFrame) -> pd.Series:
    if "sj_farmington_q_cfs" in df.columns:
        return pd.to_numeric(df["sj_farmington_q_cfs"], errors="coerce")
    return pd.to_numeric(df["sj_at_farmington_cfs"], errors="coerce")


def plot_storage_timeseries(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    thresholds = get_navajo_storage_thresholds()
    deadpool_maf = thresholds.deadpool_storage_af / 1_000_000.0
    max_maf = thresholds.max_storage_af / 1_000_000.0
    selected = pd.to_numeric(df["storage_agent_af_end"], errors="coerce") / 1_000_000.0
    historic = pd.to_numeric(df["storage_hist_af"], errors="coerce") / 1_000_000.0
    ax.plot(df.index, historic, color=HIST_COLOR, lw=1.05, ls=(0, (1.0, 1.1)), label="Historic management")
    ax.plot(df.index, selected, color=SELECTED_COLOR, lw=1.35, label="Selected policy")
    hist_min_af = _load_navajo_historical_min_storage_af()
    if hist_min_af is not None:
        ax.axhline(
            hist_min_af / 1_000_000.0,
            color="#B45353",
            lw=1.0,
            ls=(0, (5, 3)),
            alpha=0.84,
            label="Historic minimum",
            zorder=1,
        )
    ax.axhline(
        deadpool_maf,
        color=MUTED,
        lw=0.95,
        ls=(0, (4, 3)),
        alpha=0.70,
        label="Deadpool/spill",
        zorder=1,
    )
    ax.axhline(
        max_maf,
        color=MUTED,
        lw=0.95,
        ls=(0, (4, 3)),
        alpha=0.70,
        zorder=1,
    )
    ax.set_ylabel("Storage (MAF)")
    ax.set_ylim(-0.03, max_maf * 1.035)
    ax.set_xlim(df.index.min(), df.index.max())
    ax.margins(x=0)
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)


def plot_storage_climatology(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    selected = pd.to_numeric(df["storage_agent_af_end"], errors="coerce") / 1_000_000.0
    historic = pd.to_numeric(df["storage_hist_af"], errors="coerce") / 1_000_000.0
    _plot_climatology_pair(ax, selected, historic)
    ax.set_ylabel("Storage (MAF)")
    if panel:
        add_panel_label(ax, panel)


def plot_spr_timeseries(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    x = pd.DatetimeIndex(df.index)
    animas = pd.to_numeric(df["animas_farmington_q_cfs"], errors="coerce").reindex(x).fillna(0.0).clip(lower=0.0)
    release = pd.to_numeric(df["release_sj_main_cfs"], errors="coerce").reindex(x).fillna(0.0).clip(lower=0.0)
    selected_total = selected_farmington_total(df).reindex(x).fillna(animas + release)
    historic_total = historical_farmington_total(df).reindex(x)

    ax.stackplot(
        x,
        animas.to_numpy(dtype=float),
        release.to_numpy(dtype=float),
        colors=[SPR_ANIMAS_FILL, SPR_NAVAJO_FILL],
        alpha=0.52,
        linewidth=0,
        zorder=1,
    )
    ax.plot(x, selected_total, color=SELECTED_COLOR, lw=0.72, alpha=0.9, zorder=3)
    ax.plot(x, historic_total, color=HIST_COLOR, lw=0.85, ls=(0, (1.0, 1.15)), alpha=0.82, zorder=2)

    threshold_colors = _threshold_colors()
    threshold_styles = _threshold_styles()
    ax.axhline(500.0, linestyle=(0, (1, 3)), linewidth=0.75, color="#64748B", alpha=0.42, zorder=2)
    label_x = min(pd.Timestamp("2023-12-01"), x.max() - pd.Timedelta(days=90))
    for threshold_cfs, duration_days, _ in SPR_THRESHOLD_SPECS:
        threshold_i = int(round(threshold_cfs))
        ax.axhline(
            threshold_cfs,
            linestyle=threshold_styles[threshold_i],
            linewidth=0.85,
            color=threshold_colors[threshold_i],
            alpha=0.55,
            zorder=2,
        )
        ax.text(
            label_x,
            threshold_cfs,
            f"{threshold_i:,}",
            ha="center",
            va="center",
            fontsize=7.5,
            color=MUTED,
            bbox={"facecolor": BACKGROUND, "edgecolor": "none", "alpha": 0.74, "pad": 0.5},
            clip_on=True,
        )

    ymax = max(float(np.nanmax(selected_total.to_numpy(dtype=float))), float(np.nanmax(historic_total.to_numpy(dtype=float))), 10_000.0)
    ax.set_ylim(0, ymax * 1.09)
    markers = _spr_threshold_year_markers(x, selected_total)
    for marker_x, threshold, threshold_i in markers:
        ax.plot(
            [marker_x],
            [threshold + ymax * 0.018],
            marker="x",
            markersize=6.2,
            markeredgewidth=1.15,
            linestyle="None",
            color=TEXT,
            zorder=7,
        )
    ax.set_ylabel("Farmington discharge (cfs)")
    ax.set_xlim(df.index.min(), df.index.max())
    ax.margins(x=0)
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)


def plot_spr_climatology(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    _plot_climatology_pair(ax, selected_farmington_total(df), historical_farmington_total(df))
    threshold_colors = _threshold_colors()
    threshold_styles = _threshold_styles()
    ax.axhline(500.0, linestyle=(0, (1, 3)), linewidth=0.75, color="#64748B", alpha=0.42, zorder=2)
    for threshold_cfs, _duration_days, _ in SPR_THRESHOLD_SPECS:
        threshold_i = int(round(threshold_cfs))
        ax.axhline(
            threshold_cfs,
            linestyle=threshold_styles[threshold_i],
            linewidth=0.75,
            color=threshold_colors[threshold_i],
            alpha=0.55,
            zorder=2,
        )
    ax.set_ylabel("Farmington discharge (cfs)")
    if panel:
        add_panel_label(ax, panel)


def plot_niip_timeseries(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    demand = pd.to_numeric(df["niip_demand_cfs"], errors="coerce")
    delivery = pd.to_numeric(df["release_niip_cfs"], errors="coerce")
    ax.plot(df.index, demand, color=HIST_COLOR, lw=0.9, ls=(0, (1.0, 1.15)), label="Demand proxy")
    ax.plot(df.index, delivery, color=NIIP_COLOR, lw=1.05, alpha=0.9, label="Selected policy")
    ax.set_ylabel("NIIP (cfs)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(df.index.min(), df.index.max())
    ax.margins(x=0)
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)


def plot_niip_climatology_compact(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    demand = _climatology(df["niip_demand_cfs"])
    delivery = _climatology(df["release_niip_cfs"])
    ax.fill_between(
        delivery.index,
        delivery["q25"].to_numpy(dtype=float),
        delivery["q75"].to_numpy(dtype=float),
        color=NIIP_FILL,
        alpha=0.70,
        linewidth=0,
    )
    ax.plot(delivery.index, delivery["median"], color=NIIP_COLOR, lw=1.28, alpha=0.82, label="Selected policy", zorder=3)
    ax.plot(demand.index, demand["median"], color=HIST_COLOR, lw=1.45, ls=(0, (1.0, 1.1)), label="Demand proxy", zorder=4)
    ax.set_xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2001-12-31"))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("NIIP (cfs)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=2.8, color=SPINE)
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)


def plot_hydropower_timeseries(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    historic = pd.to_numeric(df["hydro_hist_mwh"], errors="coerce")
    selected = pd.to_numeric(df["hydro_agent_mwh"], errors="coerce")
    ax.plot(df.index, historic, color=HIST_COLOR, lw=0.9, ls=(0, (1.0, 1.15)), label="Historic management")
    ax.plot(df.index, selected, color=HYDRO_COLOR, lw=1.05, alpha=0.92, label="Selected policy")
    ax.set_ylabel("Hydropower (MWh/day)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(df.index.min(), df.index.max())
    ax.margins(x=0)
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)


def plot_hydropower_climatology(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    selected_stats = _climatology(df["hydro_agent_mwh"])
    historic_stats = _climatology(df["hydro_hist_mwh"])
    ax.fill_between(
        historic_stats.index,
        historic_stats["q25"].to_numpy(dtype=float),
        historic_stats["q75"].to_numpy(dtype=float),
        color=HIST_FILL,
        alpha=0.72,
        linewidth=0,
    )
    ax.plot(
        historic_stats.index,
        historic_stats["median"],
        color=HIST_COLOR,
        lw=1.2,
        ls=(0, (1.0, 1.1)),
        label="Historic management",
    )
    ax.fill_between(
        selected_stats.index,
        selected_stats["q25"].to_numpy(dtype=float),
        selected_stats["q75"].to_numpy(dtype=float),
        color=HYDRO_FILL,
        alpha=0.78,
        linewidth=0,
    )
    ax.plot(
        selected_stats.index,
        selected_stats["median"],
        color=HYDRO_COLOR,
        lw=1.45,
        label="Selected policy",
    )
    ax.set_xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2001-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=2.8, color=SPINE)
    soften_axes(ax)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Hydropower (MWh/day)")
    if panel:
        add_panel_label(ax, panel)


def plot_metric_summary(
    ax: plt.Axes,
    selected_metrics: dict[str, float],
    historic_metrics: dict[str, float],
    *,
    panel: str | None = None,
) -> None:
    specs = [
        ("Storage", "storage_frac_of_max_possible", STORAGE_COLOR),
        ("ESA min flow", "esa_min_flow_frac_days_met", ESA_COLOR),
        ("Flood safety", "flooding_frac_days_met", FLOOD_COLOR),
        ("SPR", "policy_spr_score", SPR_COLOR),
        ("Hydropower", "hydropower_frac_of_max_possible", HYDRO_COLOR),
        ("NIIP volume", "niip_annual_volume_frac_of_contract", NIIP_COLOR),
    ]
    labels: list[str] = []
    ratios: list[float] = []
    colors: list[str] = []
    for label, key, color in specs:
        selected = float(selected_metrics.get(key, np.nan))
        historic = float(historic_metrics.get(key, np.nan))
        if not np.isfinite(selected) or not np.isfinite(historic) or historic == 0.0:
            continue
        labels.append(label)
        ratios.append(selected / historic)
        colors.append(color)

    x = np.arange(len(labels))
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (3, 3)), alpha=0.85, zorder=1)
    for xi, ratio, color in zip(x, ratios, colors):
        ax.plot([xi, xi], [1.0, ratio], color=color, lw=1.45, alpha=0.65, zorder=2)
        ax.scatter([xi], [ratio], s=42, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(
            xi,
            ratio + 0.035,
            f"{ratio:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.9,
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Selected policy / historic")
    ax.set_ylim(0.86, max(1.22, max(ratios, default=1.0) + 0.18))
    ax.set_xlim(-0.5, len(labels) - 0.5)
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)


def plot_niip_climatology(ax: plt.Axes, df: pd.DataFrame, *, panel: str | None = None) -> None:
    demand = _climatology(df["niip_demand_cfs"])
    delivery = _climatology(df["release_niip_cfs"])

    ax.fill_between(
        demand.index,
        demand["q25"].to_numpy(dtype=float),
        demand["q75"].to_numpy(dtype=float),
        color=HIST_FILL,
        alpha=0.78,
        linewidth=0,
    )
    ax.plot(demand.index, demand["median"], color=HIST_COLOR, lw=1.35, ls=(0, (1.0, 1.1)), label="Demand proxy")
    ax.fill_between(
        delivery.index,
        delivery["q25"].to_numpy(dtype=float),
        delivery["q75"].to_numpy(dtype=float),
        color=NIIP_FILL,
        alpha=0.72,
        linewidth=0,
    )
    ax.plot(delivery.index, delivery["median"], color=NIIP_COLOR, lw=1.6, label="Selected policy")
    ax.set_ylabel("NIIP release or demand (cfs)")
    ax.set_xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2001-12-31"))
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True, facecolor="white", edgecolor="#D6DEE6")
    soften_axes(ax)
    if panel:
        add_panel_label(ax, panel)
