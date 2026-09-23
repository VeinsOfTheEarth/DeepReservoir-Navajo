"""Build the selected-policy performance dashboard figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SUPPORT_ROOT = HERE.parent.parent / "figure-support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

from paperstyle import (
    HIST_COLOR,
    HIST_FILL,
    MUTED,
    NIIP_COLOR,
    NIIP_FILL,
    HYDRO_COLOR,
    HYDRO_FILL,
    SELECTED_COLOR,
    SELECTED_FILL,
    SPR_ANIMAS_FILL,
    SPR_NAVAJO_FILL,
    Line2D,
    Patch,
    add_panel_label,
    load_metric_rows,
    load_selected_rollout,
    plot_hydropower_climatology,
    plot_metric_summary,
    plot_niip_climatology_compact,
    plot_release_scatter,
    plot_spr_frequencies,
    plot_spr_timeseries,
    plot_storage_timeseries,
    save_figure,
    set_theme,
)


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.tick_params(axis="x", which="minor", length=3.0, color="#94A3B8")


def _format_climatology_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=2.8, color="#94A3B8")


def main() -> None:
    set_theme()
    df = load_selected_rollout()
    selected_metrics, historic_metrics = load_metric_rows()

    fig = plt.figure(figsize=(11.6, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        width_ratios=(1.08, 1.08, 0.82),
        height_ratios=(1.0, 1.12, 1.0),
        hspace=0.08,
        wspace=0.04,
    )

    ax_storage_ts = fig.add_subplot(gs[0, 0:2])
    ax_scatter = fig.add_subplot(gs[0, 2])
    ax_spr_ts = fig.add_subplot(gs[1, 0:2], sharex=ax_storage_ts)
    ax_spr_freq = fig.add_subplot(gs[1, 2])
    ax_niip_clim = fig.add_subplot(gs[2, 0])
    ax_hydro_clim = fig.add_subplot(gs[2, 1])
    ax_metric = fig.add_subplot(gs[2, 2])

    plot_storage_timeseries(ax_storage_ts, df, panel="A")
    plot_release_scatter(ax_scatter, df, panel="B")
    plot_spr_timeseries(ax_spr_ts, df, panel="C")
    plot_spr_frequencies(ax_spr_freq, selected_metrics, historic_metrics, panel="D")
    plot_niip_climatology_compact(ax_niip_clim, df)
    add_panel_label(ax_niip_clim, "E", x=0.98, ha="right")
    plot_hydropower_climatology(ax_hydro_clim, df, panel="F")
    plot_metric_summary(ax_metric, selected_metrics, historic_metrics, panel="G")

    for ax in (ax_storage_ts, ax_spr_ts):
        _format_time_axis(ax)
    for ax in (ax_niip_clim, ax_hydro_clim):
        _format_climatology_axis(ax)

    ax_storage_ts.tick_params(labelbottom=False)
    ax_niip_clim.set_xlabel("Day of year")
    ax_hydro_clim.set_xlabel("Day of year")

    # Keep legends compact and use consistent colors across the paper.
    ax_storage_ts.legend(
        handles=[
            Line2D([0], [0], color=HIST_COLOR, lw=1.2, ls=(0, (1.0, 1.1)), label="Historic management"),
            Line2D([0], [0], color=SELECTED_COLOR, lw=1.45, label="Selected policy"),
            Line2D([0], [0], color="#B45353", lw=1.0, ls=(0, (5, 3)), label="Historic minimum"),
            Line2D([0], [0], color=MUTED, lw=0.95, ls=(0, (4, 3)), label="Deadpool/spill"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
    )
    ax_spr_ts.legend(
        handles=[
            Patch(facecolor=SPR_ANIMAS_FILL, edgecolor="none", alpha=0.52, label="Animas"),
            Patch(facecolor=SPR_NAVAJO_FILL, edgecolor="none", alpha=0.52, label="Navajo release"),
            Line2D([0], [0], color=MUTED, lw=0.95, ls=(0, (4, 3)), label="SPR threshold"),
            Line2D([0], [0], color="#111827", marker="x", lw=0, markersize=6.2, label="Annual target met"),
        ],
        loc="upper center",
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
    )
    ax_niip_clim.legend(
        handles=[
            Line2D([0], [0], color=HIST_COLOR, lw=1.2, ls=(0, (1.0, 1.1)), label="NIIP demand"),
            Patch(facecolor=NIIP_FILL, edgecolor="none", alpha=0.70, label="Selected policy IQR"),
            Line2D([0], [0], color=NIIP_COLOR, lw=1.45, label="Selected policy median"),
        ],
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
    )
    ax_hydro_clim.legend(
        handles=[
            Line2D([0], [0], color=HIST_COLOR, lw=1.2, ls=(0, (1.0, 1.1)), label="Historic median"),
            Patch(facecolor=HYDRO_FILL, edgecolor="none", alpha=0.78, label="Selected policy IQR"),
            Line2D([0], [0], color=HYDRO_COLOR, lw=1.45, label="Selected policy median"),
        ],
        loc="upper right",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
    )

    save_figure(fig, HERE, "selected-policy-comparison-with-historic")


if __name__ == "__main__":
    main()
