"""Generate selected-policy release-attribution diagnostics.

The attribution accounting is diagnostic rather than causal: each daily release
is assigned to recognizable water-management purposes using the saved rollout
fields from the selected policy.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT / "paper" / "figure-support"))
from figurestyle import OBJECTIVE_COLORS, add_panel_label

from deepreservoir.drl import selected_policy


OUTPUT_DIR = Path(__file__).resolve().parent
CFS_DAY_TO_AF = 86_400.0 / 43_560.0

BLUE = "#2B6CB0"
ESA_BLUE = OBJECTIVE_COLORS["esa"]
GREEN = OBJECTIVE_COLORS["niip"]
ORANGE = OBJECTIVE_COLORS["hydropower"]
PURPLE = OBJECTIVE_COLORS["spr"]
DAM_COLOR = OBJECTIVE_COLORS["dam_safety"]
SLATE = "#64748B"
GRAY = "#6B7280"
GRID = "#E2E8F0"
SPINE = "#94A3B8"
TEXT = "#111827"
BACKGROUND = "#F4F7FA"


@dataclass(frozen=True)
class Component:
    key: str
    label: str
    color: str


ALL_RELEASE_COMPONENTS = (
    Component("niip", "NIIP release", GREEN),
    Component("spr", "SPR-attributed San Juan", PURPLE),
    Component("esa", "ESA/baseflow San Juan", ESA_BLUE),
    Component("hydropower", "Hydropower-attributed San Juan", ORANGE),
    Component("dam_safety", "Dam safety/headroom San Juan", DAM_COLOR),
    Component("unassigned", "Unassigned controlled San Juan", SLATE),
    Component("spill", "Spill", DAM_COLOR),
)

CONTROLLED_SJ_COMPONENTS = (
    Component("spr", "SPR-attributed San Juan", PURPLE),
    Component("esa", "ESA/baseflow San Juan", ESA_BLUE),
    Component("hydropower", "Hydropower-attributed San Juan", ORANGE),
    Component("dam_safety", "Dam safety/headroom San Juan", DAM_COLOR),
    Component("unassigned", "Unassigned controlled San Juan", SLATE),
)

MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _register_fonts() -> None:
    font_dir = REPO_ROOT / "assets" / "fonts"
    if font_dir.is_dir():
        for font_path in font_dir.rglob("*.ttf"):
            try:
                fm.fontManager.addfont(str(font_path))
            except Exception:
                pass
    try:
        fm._findfont_cached.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def _set_theme() -> None:
    _register_fonts()
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
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.2,
            "savefig.bbox": "tight",
        }
    )


def _soften_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.9)
    ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    add_panel_label(ax, label, x=0.018, y=0.965)


def _nonnegative_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0).clip(lower=0.0)


def _dam_safety_and_unassigned_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    discretionary = _nonnegative_series(df, "discretionary_attributed_sj_release_cfs")
    required_cols = {"storage_agent_af", "max_storage_af", "inflow_cfs"}
    if not required_cols.issubset(df.columns):
        zero = pd.Series(0.0, index=df.index, dtype=float)
        return zero, discretionary

    storage_af = pd.to_numeric(df["storage_agent_af"], errors="coerce").fillna(0.0)
    max_storage_af = pd.to_numeric(df["max_storage_af"], errors="coerce").fillna(np.inf)
    inflow_af = _nonnegative_series(df, "inflow_cfs") * CFS_DAY_TO_AF
    if "evap_af" in df.columns:
        evap_af = _nonnegative_series(df, "evap_af")
    else:
        evap_af = _nonnegative_series(df, "evap_cfs") * CFS_DAY_TO_AF

    niip_cfs = _nonnegative_series(df, "release_niip_cfs")
    spr_cfs = _nonnegative_series(df, "spr_attributed_sj_release_cfs")
    spr_useful_cfs = _nonnegative_series(df, "spr_useful_unrequested_sj_release_cfs")
    esa_cfs = _nonnegative_series(df, "esa_attributed_sj_release_cfs")
    hydro_cfs = _nonnegative_series(df, "hydropower_attributed_sj_release_cfs")

    no_control_excess_af = (storage_af + inflow_af - evap_af - max_storage_af).clip(lower=0.0)
    total_out_needed_cfs = no_control_excess_af / CFS_DAY_TO_AF
    mainstem_needed_cfs = (total_out_needed_cfs - niip_cfs).clip(lower=0.0)
    residual_after_named_cfs = (
        mainstem_needed_cfs - spr_cfs - spr_useful_cfs - esa_cfs - hydro_cfs
    ).clip(lower=0.0)
    dam_safety = pd.concat([discretionary, residual_after_named_cfs], axis=1).min(axis=1)
    dam_safety = dam_safety.fillna(0.0).clip(lower=0.0)
    unassigned = (discretionary - dam_safety).fillna(0.0).clip(lower=0.0)
    return dam_safety, unassigned


def _component_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    dam_safety, unassigned = _dam_safety_and_unassigned_series(df)
    return {
        "niip": _nonnegative_series(df, "release_niip_cfs"),
        "spr": (
            _nonnegative_series(df, "spr_attributed_sj_release_cfs")
            + _nonnegative_series(df, "spr_useful_unrequested_sj_release_cfs")
        ),
        "esa": _nonnegative_series(df, "esa_attributed_sj_release_cfs"),
        "hydropower": _nonnegative_series(df, "hydropower_attributed_sj_release_cfs"),
        "dam_safety": dam_safety,
        "unassigned": unassigned,
        "spill": _nonnegative_series(df, "spill_cfs"),
    }


def _active_components(
    components: tuple[Component, ...],
    series_by_key: dict[str, pd.Series],
) -> list[Component]:
    active: list[Component] = []
    for component in components:
        series = series_by_key[component.key]
        if float(series.sum()) > 1e-9:
            active.append(component)
    return active


def _plot_daily_timeseries(
    ax: plt.Axes,
    df: pd.DataFrame,
    series_by_key: dict[str, pd.Series],
    active_components: list[Component],
) -> None:
    x = pd.DatetimeIndex(df.index)
    values = [series_by_key[c.key].to_numpy(dtype=float) for c in active_components]
    ax.stackplot(
        x,
        *values,
        colors=[c.color for c in active_components],
        labels=[c.label for c in active_components],
        linewidth=0,
        alpha=0.82,
        zorder=3,
    )
    total_release = np.sum(np.vstack(values), axis=0) if values else np.zeros(len(x))
    ymax = max(1000.0, float(np.nanmax(total_release)) * 1.09)
    ax.set_ylim(0.0, ymax)
    ax.set_xlim(x.min(), x.max())
    ax.margins(x=0)
    ax.set_ylabel("Release\n(cfs)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="x", which="minor", length=2.8, color=SPINE)
    _soften_axes(ax)
    _panel_label(ax, "A")


def _plot_monthly_volumes(
    ax: plt.Axes,
    df: pd.DataFrame,
    series_by_key: dict[str, pd.Series],
    active_components: list[Component],
) -> None:
    months = np.arange(1, 13, dtype=float)
    bottom = np.zeros(12, dtype=float)
    for component in active_components:
        monthly = (
            series_by_key[component.key].groupby(df.index.month).sum().reindex(range(1, 13), fill_value=0.0)
            * CFS_DAY_TO_AF
            / 1000.0
        )
        values = monthly.to_numpy(dtype=float)
        ax.bar(
            months,
            values,
            width=0.72,
            bottom=bottom,
            color=component.color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
        bottom += values
    ax.set_xlim(0.35, 12.65)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Volume\n(kAF)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    _soften_axes(ax)
    _panel_label(ax, "B")


def _plot_annual_sj_fractions(
    ax: plt.Axes,
    df: pd.DataFrame,
    series_by_key: dict[str, pd.Series],
    active_components: list[Component],
) -> None:
    annual = pd.DataFrame(
        {component.label: series_by_key[component.key] for component in active_components},
        index=df.index,
    ).groupby(df.index.year).sum() * CFS_DAY_TO_AF
    annual = annual.loc[:, annual.sum(axis=0) > 1e-9]
    fractions = annual.div(annual.sum(axis=1).replace(0.0, np.nan), axis=0) * 100.0
    years = fractions.index.to_numpy(dtype=int)
    bottom = np.zeros(len(years), dtype=float)
    color_by_label = {component.label: component.color for component in active_components}
    for label in fractions.columns:
        values = fractions[label].fillna(0.0).to_numpy(dtype=float)
        ax.bar(
            years,
            values,
            width=0.78,
            bottom=bottom,
            color=color_by_label[label],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
        bottom += values
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(years.min() - 0.6, years.max() + 0.6)
    ax.set_xticks(years)
    ax.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    ax.set_ylabel("Share of controlled\nSan Juan release (%)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    _soften_axes(ax)
    _panel_label(ax, "C")


def _legend_components(active_components: list[Component]) -> tuple[list[Patch], list[str]]:
    handles = [
        Patch(facecolor=component.color, edgecolor="none", alpha=0.88)
        for component in active_components
    ]
    labels = [component.label for component in active_components]
    return handles, labels


def build(
    *,
    rollout_parquet: Path,
    output_dir: Path,
    dpi: int = 320,
) -> list[Path]:
    _set_theme()
    df = pd.read_parquet(rollout_parquet)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Selected-policy rollout must have a DatetimeIndex.")
    df = df.sort_index()

    series_by_key = _component_series(df)
    active_all = _active_components(ALL_RELEASE_COMPONENTS, series_by_key)
    active_sj = _active_components(CONTROLLED_SJ_COMPONENTS, series_by_key)

    fig = plt.figure(figsize=(8.4, 6.55), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 1.0])
    ax_daily = fig.add_subplot(gs[0, :])
    ax_monthly = fig.add_subplot(gs[1, 0])
    ax_annual = fig.add_subplot(gs[1, 1])

    _plot_daily_timeseries(ax_daily, df, series_by_key, active_all)
    _plot_monthly_volumes(ax_monthly, df, series_by_key, active_all)
    _plot_annual_sj_fractions(ax_annual, df, series_by_key, active_sj)

    handles, labels = _legend_components(active_all)
    ax_daily.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.985),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
        handlelength=1.3,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "release-attribution.png",
        output_dir / "release-attribution.pdf",
    ]
    fig.savefig(paths[0], dpi=dpi)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-parquet",
        type=Path,
        default=selected_policy.SELECTED_EVAL_ROLLOUT_PATH,
        help="Selected-policy holdout rollout parquet artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for attribution figure outputs.",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = build(
        rollout_parquet=args.rollout_parquet,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
