"""Build selected-policy stress-testing multi-panel figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SUPPORT_ROOT = HERE.parent.parent / "figure-support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

REPO_ROOT = HERE.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paperstyle import (  # noqa: E402
    DAM_COLOR,
    ESA_COLOR,
    FLOOD_COLOR,
    HIST_COLOR,
    HYDRO_COLOR,
    MUTED,
    NIIP_COLOR,
    SELECTED_COLOR,
    SPR_COLOR,
    STORAGE_COLOR,
    TEXT,
    add_panel_label,
    load_metric_rows,
    save_figure,
    set_theme,
    soften_axes,
)
from deepreservoir.define_env.storage_elevation.thresholds import (  # noqa: E402
    get_navajo_storage_thresholds,
)
from deepreservoir.drl import selected_policy  # noqa: E402


METRIC_SPECS = (
    ("Spill-free days", "no_spill_frac", DAM_COLOR),
    ("Storage", "storage_frac_of_max_possible", STORAGE_COLOR),
    ("ESA min flow", "esa_min_flow_frac_days_met", ESA_COLOR),
    ("Flood safety", "flooding_frac_days_met", FLOOD_COLOR),
    ("SPR", "policy_spr_score", SPR_COLOR),
    ("Hydropower", "hydropower_frac_of_max_possible", HYDRO_COLOR),
    ("NIIP volume", "niip_annual_volume_frac_of_contract", NIIP_COLOR),
)


def _format_time_axis(ax: plt.Axes, *, show_label: bool = False) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.tick_params(axis="x", which="minor", length=3.0, color="#94A3B8")
    if show_label:
        ax.set_xlabel("Date")
    else:
        ax.set_xlabel("")


def _load_initial_storage_trajectory() -> pd.DataFrame:
    path = HERE / "data" / "initial-storage-sweep-storage.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing initial-storage sweep artifact: {path}. "
            "Run scripts/run_selected_policy_initial_storage_sweep.py first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["initial_storage_maf", "date"])


def _load_inflow_scaling_trajectory() -> pd.DataFrame:
    path = HERE / "data" / "inflow-scaling-sweep-storage.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing inflow-scaling sweep artifact: {path}. "
            "Run scripts/run_selected_policy_inflow_scaling_sweep.py first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["inflow_change_pct", "date"])


def _load_initial_storage_summary() -> pd.DataFrame:
    path = HERE / "data" / "initial-storage-sweep-summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing initial-storage summary artifact: {path}. "
            "Run scripts/run_selected_policy_initial_storage_sweep.py first."
        )
    df = pd.read_csv(path)
    df["initial_storage_maf"] = pd.to_numeric(df["initial_storage_maf"], errors="coerce")
    return df.sort_values("initial_storage_maf").reset_index(drop=True)


def _load_inflow_scaling_summary() -> pd.DataFrame:
    path = HERE / "data" / "inflow-scaling-sweep-summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing inflow-scaling summary artifact: {path}. "
            "Run scripts/run_selected_policy_inflow_scaling_sweep.py first."
        )
    df = pd.read_csv(path)
    df["inflow_change_pct"] = pd.to_numeric(df["inflow_change_pct"], errors="coerce")
    return df.sort_values("inflow_change_pct").reset_index(drop=True)


def _ratio_values(summary: pd.DataFrame, historic: dict[str, float], key: str) -> np.ndarray:
    if key == "no_spill_frac":
        spill_frac = pd.to_numeric(summary["spill_frac_days"], errors="coerce").fillna(0.0)
        return (1.0 - spill_frac).to_numpy(dtype=float)
    baseline = float(historic[key])
    values = pd.to_numeric(summary[key], errors="coerce").to_numpy(dtype=float)
    return values / baseline if baseline else np.full_like(values, np.nan)


def _shared_ratio_axis_upper(
    summaries: tuple[pd.DataFrame, ...],
    historic: dict[str, float],
) -> float:
    max_ratio = 1.0
    for summary in summaries:
        for _label, key, _color in METRIC_SPECS:
            values = _ratio_values(summary, historic, key)
            finite = values[np.isfinite(values)]
            if len(finite):
                max_ratio = max(max_ratio, float(np.nanmax(finite)))
    return float(np.ceil(max_ratio * 1.08 * 10.0) / 10.0)


def _selected_initial_storage_maf() -> float:
    df = pd.read_parquet(
        selected_policy.SELECTED_EVAL_ROLLOUT_PATH,
        columns=["storage_agent_af"],
    )
    return float(pd.to_numeric(df["storage_agent_af"], errors="coerce").iloc[0]) / 1_000_000.0


def _add_storage_reference_lines(
    ax: plt.Axes,
    *,
    initial_storage_maf: float | None = None,
) -> None:
    thresholds = get_navajo_storage_thresholds()
    ax.axhline(
        thresholds.deadpool_storage_af / 1_000_000.0,
        color=HIST_COLOR,
        lw=0.85,
        ls=(0, (4, 3)),
        alpha=0.65,
        zorder=2,
    )
    ax.axhline(
        thresholds.max_storage_af / 1_000_000.0,
        color=HIST_COLOR,
        lw=0.85,
        ls=(0, (4, 3)),
        alpha=0.78,
        zorder=2,
    )
    if initial_storage_maf is not None:
        ax.axhline(
            initial_storage_maf,
            color=TEXT,
            lw=0.9,
            ls=(0, (4, 3)),
            alpha=0.72,
            zorder=2,
        )
    ax.text(
        pd.Timestamp("2022-01-01"),
        thresholds.max_storage_af / 1_000_000.0 + 0.018,
        "spill level",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=MUTED,
        clip_on=True,
    )


def _plot_initial_storage_trajectories(
    ax: plt.Axes,
    fig: plt.Figure,
    sweep: pd.DataFrame,
    selected_initial_storage_maf: float,
) -> None:
    cmap = mpl.colormaps["viridis"]
    vmin = float(sweep["initial_storage_maf"].min())
    vmax = float(sweep["initial_storage_maf"].max())
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for scenario_initial_storage_maf, group in sweep.groupby("initial_storage_maf", sort=True):
        ax.plot(
            group["date"],
            group["storage_maf"],
            color=cmap(norm(float(scenario_initial_storage_maf))),
            lw=0.82,
            alpha=0.82,
            zorder=3,
        )

    _add_storage_reference_lines(ax, initial_storage_maf=selected_initial_storage_maf)
    ax.set_xlim(sweep["date"].min(), sweep["date"].max())
    ax.set_ylim(0.0, 1.75)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 1.75])
    ax.tick_params(axis="y", labelsize=10.2)
    ax.margins(x=0)
    ax.set_ylabel("Storage (MAF)")
    _format_time_axis(ax)
    soften_axes(ax)
    add_panel_label(ax, "C")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.034, pad=0.016)
    cbar.set_label("Initial storage (MAF)", rotation=270, labelpad=15)
    cbar.set_ticks(np.arange(0.0, 1.8, 0.4))


def _plot_inflow_scaling_trajectories(
    ax: plt.Axes,
    fig: plt.Figure,
    sweep: pd.DataFrame,
) -> None:
    cmap = mpl.colormaps["coolwarm_r"]
    vmin = float(sweep["inflow_change_pct"].min())
    vmax = float(sweep["inflow_change_pct"].max())
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    for inflow_change_pct, group in sweep.groupby("inflow_change_pct", sort=True):
        ax.plot(
            group["date"],
            group["storage_maf"],
            color=cmap(norm(float(inflow_change_pct))),
            lw=0.82,
            alpha=0.82,
            zorder=3,
        )

    _add_storage_reference_lines(ax)
    ax.set_xlim(sweep["date"].min(), sweep["date"].max())
    ax.set_ylim(0.0, 1.75)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 1.75])
    ax.tick_params(axis="y", labelsize=10.2)
    ax.margins(x=0)
    ax.set_ylabel("Storage (MAF)")
    _format_time_axis(ax, show_label=True)
    soften_axes(ax)
    add_panel_label(ax, "A")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.034, pad=0.016)
    cbar.set_label("Reservoir inflow change (%)", rotation=270, labelpad=15)
    cbar.set_ticks(np.arange(-50.0, 151.0, 50.0))


def _plot_objective_response(
    ax: plt.Axes,
    summary: pd.DataFrame,
    historic: dict[str, float],
    x_col: str,
    x_label: str,
    x_ticks: np.ndarray,
    *,
    x_reference: float | None,
    reference_label: str,
    reference_text_xy: tuple[float, float] | None = None,
    y_upper: float,
    panel: str,
    show_legend: bool,
    annotate_first_spill: bool = False,
) -> None:
    x = summary[x_col].to_numpy(dtype=float)
    ax.axhline(1.0, color=MUTED, lw=0.95, ls=(0, (3, 3)), alpha=0.86, zorder=1)
    if x_reference is not None:
        ax.axvline(x_reference, color=TEXT, lw=0.85, ls=(0, (4, 3)), alpha=0.72, zorder=1)
        text_x, text_y = reference_text_xy if reference_text_xy is not None else (x_reference, 0.10)
        ax.text(
            text_x,
            text_y,
            reference_label,
            ha="left",
            va="bottom",
            fontsize=7.7,
            color=TEXT,
            clip_on=True,
        )

    for label, key, color in METRIC_SPECS:
        y = _ratio_values(summary, historic, key)
        ax.plot(x, y, color=color, lw=1.05, alpha=0.70, zorder=2)
        ax.scatter(
            x,
            y,
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.94,
            label=label,
            zorder=3,
        )

    if annotate_first_spill:
        spill_rows = summary[pd.to_numeric(summary["spill_days"], errors="coerce").fillna(0.0) > 0.0]
        if not spill_rows.empty:
            first_spill = spill_rows.sort_values(x_col).iloc[0]
            spill_x = float(first_spill[x_col])
            spill_days = int(round(float(first_spill["spill_days"])))
            spill_y = float(1.0 - float(first_spill["spill_frac_days"]))
            ax.annotate(
                f"first spill:\n{spill_days} d @ +{spill_x:.0f}%",
                xy=(spill_x, spill_y),
                xytext=(spill_x + 18.0, 0.34),
                textcoords="data",
                ha="left",
                va="bottom",
                fontsize=7.7,
                color=TEXT,
                arrowprops={
                    "arrowstyle": "->",
                    "color": TEXT,
                    "lw": 0.85,
                    "shrinkA": 2.0,
                    "shrinkB": 3.0,
                },
                zorder=5,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Stress run / historic")
    ax.set_xlim(min(x) - 0.03 * (max(x) - min(x)), max(x) + 0.03 * (max(x) - min(x)))
    ax.set_ylim(0.0, y_upper)
    ax.set_xticks(x_ticks)
    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.54, 1.0),
            ncol=4,
            frameon=True,
            facecolor="white",
            edgecolor="#D6DEE6",
            handlelength=1.25,
            columnspacing=1.3,
            alignment="center",
        )
    soften_axes(ax)
    add_panel_label(ax, panel)


def main() -> None:
    set_theme()
    initial_traj = _load_initial_storage_trajectory()
    inflow_traj = _load_inflow_scaling_trajectory()
    initial_summary = _load_initial_storage_summary()
    inflow_summary = _load_inflow_scaling_summary()
    _selected, historic = load_metric_rows()
    initial_storage_maf = _selected_initial_storage_maf()

    y_upper = _shared_ratio_axis_upper((initial_summary, inflow_summary), historic)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.9, 8.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.08, 1.0], "height_ratios": [1.0, 1.0]},
    )

    _plot_inflow_scaling_trajectories(axes[0, 0], fig, inflow_traj)
    axes[0, 0].set_xlabel("")
    _plot_objective_response(
        axes[0, 1],
        inflow_summary,
        historic,
        "inflow_change_pct",
        "Reservoir inflow change (%)",
        np.arange(-50.0, 151.0, 50.0),
        x_reference=0.0,
        reference_label="actual inflow",
        reference_text_xy=(2.0, 0.10),
        y_upper=y_upper,
        panel="B",
        show_legend=True,
        annotate_first_spill=True,
    )
    axes[0, 1].set_xlabel("")
    _plot_initial_storage_trajectories(axes[1, 0], fig, initial_traj, initial_storage_maf)
    axes[1, 0].set_xlabel("Date")
    _plot_objective_response(
        axes[1, 1],
        initial_summary,
        historic,
        "initial_storage_maf",
        "Initial storage (MAF)",
        np.arange(0.0, 1.7, 0.4),
        x_reference=initial_storage_maf,
        reference_label="initial storage",
        reference_text_xy=(initial_storage_maf + 0.025, 0.10),
        y_upper=y_upper,
        panel="D",
        show_legend=True,
    )

    save_figure(fig, HERE, "stress-testing")


if __name__ == "__main__":
    main()
