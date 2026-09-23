"""Build policy-response figures from paired inflow rollouts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


HERE = Path(__file__).resolve().parent
FIGURE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
for path in (FIGURE_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SUPPORT_ROOT = FIGURE_ROOT.parent / "figure-support"
sys.path.insert(0, str(SUPPORT_ROOT))

from paperstyle import (  # noqa: E402
    DAM_COLOR,
    ESA_COLOR,
    FLOOD_COLOR,
    HYDRO_COLOR,
    NIIP_COLOR,
    SELECTED_COLOR,
    SPINE,
    SPR_COLOR,
    STORAGE_COLOR,
    TEXT,
    add_panel_label,
    save_figure,
    set_theme,
    soften_axes,
)
from figurestyle import DISCRETIONARY_COLOR


DATA_DIR = HERE / "data"
ROLLOUT_PATH = DATA_DIR / "rollout-summary.csv"
PAIRED_PATH = DATA_DIR / "paired-differences.csv"
METADATA_PATH = DATA_DIR / "metadata.json"

LOW_COLOR = "#B2182B"
NEUTRAL_COLOR = "#F7F7F7"
HIGH_COLOR = "#2166AC"

OBJECTIVES = (
    ("spill_free_days", "Spill-free days", DAM_COLOR),
    ("storage_frac_of_max_possible", "Storage", STORAGE_COLOR),
    ("esa_min_flow_frac_days_met", "ESA min flow", ESA_COLOR),
    ("flooding_frac_days_met", "Flood safety", FLOOD_COLOR),
    ("hydropower_frac_of_max_possible", "Hydropower", HYDRO_COLOR),
    ("niip_annual_volume_frac_of_contract", "NIIP volume", NIIP_COLOR),
)

WATER_COMPONENTS = (
    ("delta_ending_storage_af", "Ending storage", STORAGE_COLOR, "-"),
    ("delta_niip_release_af", "NIIP delivery", NIIP_COLOR, "-"),
    ("delta_esa_release_af", "ESA release", ESA_COLOR, "-"),
    ("delta_spr_release_af", "SPR release", SPR_COLOR, (0, (3, 1.5))),
    (
        "delta_discretionary_sj_release_af",
        "Discretionary release",
        DISCRETIONARY_COLOR,
        "-",
    ),
    ("delta_total_spill_af", "Spill", DAM_COLOR, (0, (1.2, 1.2))),
)

ACTION_COMPONENTS = (
    ("action_0_mean_delta", "ESA/baseflow", ESA_COLOR),
    ("action_1_mean_delta", "NIIP delivery", NIIP_COLOR),
    ("action_2_mean_delta", "SPR target", SPR_COLOR),
    ("action_3_mean_delta", "Discretionary release", DISCRETIONARY_COLOR),
)


def _load() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    missing = [
        path for path in (ROLLOUT_PATH, PAIRED_PATH, METADATA_PATH) if not path.exists()
    ]
    if missing:
        paths = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing diagnostic data:\n{paths}\nRun run_diagnostics.py first."
        )
    rollout = pd.read_csv(ROLLOUT_PATH)
    paired = pd.read_csv(PAIRED_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return rollout, paired, metadata


def _performance_matrix(
    rollout: pd.DataFrame,
    metadata: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    adaptive = rollout.loc[rollout["mode"] == "adaptive"].sort_values(
        "inflow_change_pct"
    )
    replay = rollout.loc[rollout["mode"] == "replay"].sort_values(
        "inflow_change_pct"
    )
    if not np.allclose(adaptive["inflow_change_pct"], replay["inflow_change_pct"]):
        raise ValueError("Adaptive and replay summaries have different perturbations.")

    historic = dict(metadata["historic_metrics"])
    matrix: list[np.ndarray] = []
    for metric, _label, _color in OBJECTIVES:
        benchmark = 1.0 if metric == "spill_free_days" else float(historic[metric])
        adaptive_values = pd.to_numeric(
            adaptive[metric], errors="raise"
        ).to_numpy(dtype=float)
        replay_values = pd.to_numeric(replay[metric], errors="raise").to_numpy(
            dtype=float
        )
        matrix.append((adaptive_values - replay_values) / benchmark)
    return (
        adaptive["inflow_change_pct"].to_numpy(dtype=float),
        np.asarray(matrix, dtype=float),
    )


def _heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "policy_response",
        [LOW_COLOR, NEUTRAL_COLOR, HIGH_COLOR],
        N=256,
    )


def _plot_heatmap(
    ax: plt.Axes,
    rollout: pd.DataFrame,
    metadata: dict[str, object],
    *,
    panel: str | None = None,
) -> mpl.cm.ScalarMappable:
    perturbations, matrix = _performance_matrix(rollout, metadata)
    max_abs = max(0.01, float(np.nanmax(np.abs(matrix))))
    max_abs = float(np.ceil(max_abs * 100.0) / 100.0)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    x_edges = np.r_[
        perturbations[0] - 5.0,
        (perturbations[:-1] + perturbations[1:]) / 2.0,
        perturbations[-1] + 5.0,
    ]
    y_edges = np.arange(len(OBJECTIVES) + 1, dtype=float)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=_heatmap_cmap(),
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=1.0,
    )
    ax.axvline(0.0, color="#64748B", lw=1.0, ls="--", zorder=4)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(len(OBJECTIVES), 0.0)
    ax.set_xticks(perturbations)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:+.0f}"))
    ax.set_yticks(np.arange(len(OBJECTIVES), dtype=float) + 0.5)
    ax.set_yticklabels([label for _metric, label, _color in OBJECTIVES])
    for tick, (_metric, _label, color) in zip(ax.get_yticklabels(), OBJECTIVES):
        tick.set_color(color)
        tick.set_fontweight("semibold")
    ax.set_xlabel("Change in reservoir inflow (%)")
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = max_abs * 0.53
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if abs(value) >= threshold else TEXT
            ax.text(
                perturbations[col],
                row + 0.5,
                f"{value:+.2f}" if abs(value) >= 0.005 else "0.00",
                ha="center",
                va="center",
                fontsize=7.3,
                color=text_color,
                zorder=5,
            )
    if panel:
        add_panel_label(
            ax, panel,
            x=-0.18,
            y=1.0 - 0.5 / len(OBJECTIVES),
            ha="right",
            va="center",
        )
    return mesh


def _plot_water_reallocation(
    ax: plt.Axes,
    paired: pd.DataFrame,
    *,
    panel: str | None = None,
) -> None:
    data = paired.sort_values("inflow_change_pct")
    x = data["inflow_change_pct"].to_numpy(dtype=float)
    for column, label, color, linestyle in WATER_COMPONENTS:
        y = (
            pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
            / 1_000_000.0
        )
        ax.plot(
            x,
            y,
            color=color,
            ls=linestyle,
            lw=1.8 if column == "delta_ending_storage_af" else 1.45,
            marker="o",
            ms=3.2,
            label=label,
        )
    ax.axhline(0.0, color="#64748B", lw=0.9, ls="--")
    ax.axvline(0.0, color="#64748B", lw=0.9, ls="--")
    ax.set_xlim(x.min(), x.max())
    ax.margins(x=0)
    ax.set_xlabel("Change in reservoir inflow (%)")
    ax.set_ylabel("Water allocation change from\nsaved schedule (MAF)")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:+.0f}"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    soften_axes(ax)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
        columnspacing=1.0,
        handlelength=2.0,
    )
    if panel:
        add_panel_label(ax, panel)


def _plot_action_response(
    ax: plt.Axes,
    paired: pd.DataFrame,
    *,
    panel: str | None = None,
) -> None:
    data = paired.sort_values("inflow_change_pct")
    x = data["inflow_change_pct"].to_numpy(dtype=float)
    for column, label, color in ACTION_COMPONENTS:
        y = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
        ax.plot(x, y, color=color, lw=1.55, marker="o", ms=3.2, label=label)
    ax.axvline(0.0, color="#64748B", lw=0.9, ls="--")
    ax.axhline(0.0, color="#64748B", lw=0.9, ls="--")
    ax.set_xlim(x.min(), x.max())
    ax.margins(x=0)
    ax.set_xlabel("Change in reservoir inflow (%)")
    ax.set_ylabel(
        "Mean change in policy action from\nsaved schedule (normalized)"
    )
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:+.0f}"))
    soften_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
        columnspacing=1.0,
    )
    if panel:
        add_panel_label(ax, panel, x=0.92, ha="right")


def _add_colorbar(
    fig: plt.Figure,
    mesh: mpl.cm.ScalarMappable,
    ax: plt.Axes,
    *,
    orientation: str,
    pad: float,
    fraction: float,
) -> None:
    colorbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation=orientation,
        pad=pad,
        fraction=fraction,
    )
    colorbar.set_label("Performance change\n(fraction of historic)")
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.ax.tick_params(length=2.5)


def build_figures() -> list[Path]:
    set_theme()
    rollout, paired, metadata = _load()
    outputs: list[Path] = []

    fig = plt.figure(figsize=(11.0, 6.7), constrained_layout=False)
    action_ax = fig.add_axes([0.085, 0.565, 0.405, 0.385])
    water_ax = fig.add_axes([0.57, 0.565, 0.41, 0.385])
    heat_ax = fig.add_axes([0.20, 0.105, 0.60, 0.38])
    _plot_action_response(action_ax, paired, panel="A")
    _plot_water_reallocation(water_ax, paired, panel="B")
    mesh = _plot_heatmap(heat_ax, rollout, metadata, panel="C")
    colorbar_ax = fig.add_axes([0.825, 0.105, 0.016, 0.38])
    colorbar = fig.colorbar(mesh, cax=colorbar_ax, orientation="vertical")
    colorbar.set_label(
        "Performance change\n(fraction of historic)",
        labelpad=11.0,
        multialignment="center",
    )
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.ax.tick_params(length=2.5)
    outputs.extend(save_figure(fig, HERE, "policy-response"))
    return outputs


if __name__ == "__main__":
    for output in build_figures():
        print(output)
