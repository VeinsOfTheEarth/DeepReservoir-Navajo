"""Build the diagnostic-training reward-signal figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
FIGURE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
SUPPORT_ROOT = FIGURE_ROOT.parent / "figure-support"
if str(SUPPORT_ROOT) not in sys.path:
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


DATA_DIR = HERE / "data"
TRAINING_DATA = DATA_DIR / "training-reward-signals.csv"
EVALUATION_DATA = DATA_DIR / "selected-policy-evaluation-positive-rewards.csv"
METADATA_PATH = DATA_DIR / "training-reward-signals-metadata.json"

UPDATE_SOURCE = DATA_DIR / "train-update-metrics.csv"
COMPONENT_SOURCE = DATA_DIR / "train-update-component-diagnostics.csv"
EVALUATION_SOURCE = DATA_DIR / "selected-policy-evaluation-daily-rewards.csv"

ROLLING_UPDATES = 25
TOTAL_COLOR = "#334155"
REFERENCE_COLOR = "#64748B"
END_COLOR = "#D99A00"

COMPONENTS = (
    {
        "key": "dam_safety",
        "label": "Dam safety",
        "color": DAM_COLOR,
        "mean_column": "mean_dam_safety.spill_guard_warn98",
        "component": "dam_safety.spill_guard_warn98",
    },
    {
        "key": "flood_safety",
        "label": "Flood safety",
        "color": FLOOD_COLOR,
        "mean_column": "mean_flooding.penalty_caps_jon",
        "component": "flooding.penalty_caps_jon",
    },
    {
        "key": "niip_delivery",
        "label": "NIIP delivery",
        "color": NIIP_COLOR,
        "mean_column": "mean_niip.delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon",
        "component": "niip.delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon",
    },
    {
        "key": "hydropower",
        "label": "Hydropower",
        "color": HYDRO_COLOR,
        "mean_column": "mean_hydropower.positive_discretionary_efficiency",
        "component": "hydropower.positive_discretionary_efficiency",
    },
    {
        "key": "esa_min_flow",
        "label": "ESA min flow",
        "color": ESA_COLOR,
        "mean_column": "mean_esa_min_flow.green_logistic_jon",
        "component": "esa_min_flow.green_logistic_jon",
    },
    {
        "key": "storage",
        "label": "Storage",
        "color": STORAGE_COLOR,
        "mean_column": "mean_storage_control.target_peak875_concave0_softupper98",
        "component": "storage_control.target_peak875_concave0_softupper98",
    },
    {
        "key": "spring_peak_release",
        "label": "SPR",
        "color": SPR_COLOR,
        "mean_column": "mean_esa_spring_peak_release.farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong",
        "component": "esa_spring_peak_release.farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong",
    },
)
COMPONENT_BY_KEY = {item["key"]: item for item in COMPONENTS}
STACK_ORDER = (
    "spring_peak_release",
    "storage",
    "esa_min_flow",
    "niip_delivery",
    "hydropower",
    "flood_safety",
    "dam_safety",
)
LEGEND_ORDER = (
    "storage",
    "esa_min_flow",
    "niip_delivery",
    "hydropower",
    "spring_peak_release",
    "flood_safety",
    "dam_safety",
)


def _rolling(series: pd.Series) -> pd.Series:
    return series.rolling(
        window=ROLLING_UPDATES,
        center=True,
        min_periods=1,
    ).mean()


def _refresh_data() -> None:
    missing_sources = [
        path
        for path in (UPDATE_SOURCE, COMPONENT_SOURCE, EVALUATION_SOURCE)
        if not path.exists()
    ]
    if missing_sources:
        formatted = "\n".join(f"- {path}" for path in missing_sources)
        raise FileNotFoundError(f"Missing source data:\n{formatted}")

    updates = pd.read_csv(UPDATE_SOURCE)
    required_updates = {
        "update_idx",
        "timesteps",
        "mean_total_reward",
        *(item["mean_column"] for item in COMPONENTS),
    }
    missing = sorted(required_updates.difference(updates.columns))
    if missing:
        raise ValueError("Update diagnostics are missing columns: " + ", ".join(missing))

    component_data = pd.read_csv(COMPONENT_SOURCE)
    required_components = {
        "update_idx",
        "component",
        "steps",
        "reward_abs_share",
        "positive_steps",
        "positive_sum",
    }
    missing = sorted(required_components.difference(component_data.columns))
    if missing:
        raise ValueError(
            "Component diagnostics are missing columns: " + ", ".join(missing)
        )

    output = updates[["update_idx", "timesteps"]].copy()
    output["total_reward"] = pd.to_numeric(
        updates["mean_total_reward"], errors="raise"
    )
    indexed_components = component_data.set_index(["update_idx", "component"])
    for item in COMPONENTS:
        key = item["key"]
        output[f"mean_{key}"] = pd.to_numeric(
            updates[item["mean_column"]], errors="raise"
        )
        rows = indexed_components.xs(item["component"], level="component").reindex(
            output["update_idx"]
        )
        steps = pd.to_numeric(rows["steps"], errors="raise").to_numpy(dtype=float)
        positive_steps = pd.to_numeric(
            rows["positive_steps"], errors="raise"
        ).to_numpy(dtype=float)
        positive_sum = pd.to_numeric(
            rows["positive_sum"], errors="raise"
        ).to_numpy(dtype=float)
        output[f"absolute_share_{key}"] = (
            100.0
            * pd.to_numeric(rows["reward_abs_share"], errors="raise").to_numpy(
                dtype=float
            )
        )
        output[f"positive_days_{key}"] = np.divide(
            100.0 * positive_steps,
            steps,
            out=np.zeros_like(positive_steps),
            where=steps > 0.0,
        )
        output[f"mean_positive_{key}"] = np.divide(
            positive_sum,
            positive_steps,
            out=np.full_like(positive_sum, np.nan),
            where=positive_steps > 0.0,
        )

    evaluation = pd.read_csv(EVALUATION_SOURCE)
    evaluation_rows = []
    for key in ("spring_peak_release", "storage"):
        if key not in evaluation.columns:
            raise ValueError(f"Evaluation rewards are missing column: {key}")
        values = pd.to_numeric(evaluation[key], errors="raise")
        positive = values.loc[values > 0.0]
        evaluation_rows.append(
            {
                "objective": key,
                "positive_reward_days_pct": 100.0 * float((values > 0.0).mean()),
                "mean_positive_reward": float(positive.mean()),
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(TRAINING_DATA, index=False, float_format="%.9g")
    pd.DataFrame(evaluation_rows).to_csv(
        EVALUATION_DATA,
        index=False,
        float_format="%.9g",
    )
    METADATA_PATH.write_text(
        json.dumps(
            {
                "training_source": "same-configuration rich diagnostic retraining",
                "selected_legacy_family": "reward_jon_p95_peak875_hdisceff",
                "selected_seed": 4,
                "rolling_updates": ROLLING_UPDATES,
                "evaluation_reference": "archived selected-policy deterministic evaluation",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [path for path in (TRAINING_DATA, EVALUATION_DATA) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing tracked extracts:\n{formatted}\nRun build.py --refresh-data once."
        )
    training = pd.read_csv(TRAINING_DATA).sort_values("timesteps").reset_index(drop=True)
    evaluation = pd.read_csv(EVALUATION_DATA).set_index("objective")
    return training, evaluation


def _progress_cmap(name: str, start: float = 0.25) -> LinearSegmentedColormap:
    base = mpl.colormaps[name]
    return LinearSegmentedColormap.from_list(
        f"{name.lower()}_training_progress",
        base(np.linspace(start, 0.98, 256)),
    )


def _plot_training_rewards(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = data["timesteps"].to_numpy(dtype=float) / 1_000_000.0
    ax.axhline(0.0, color=REFERENCE_COLOR, lw=0.85, zorder=1)
    for item in COMPONENTS:
        values = data[f"mean_{item['key']}"]
        ax.plot(x, values, color=item["color"], lw=0.38, alpha=0.13, zorder=2)
        ax.plot(x, _rolling(values), color=item["color"], lw=1.22, zorder=3)
    total = data["total_reward"]
    ax.plot(x, total, color=TOTAL_COLOR, lw=0.45, alpha=0.17, zorder=2)
    ax.plot(x, _rolling(total), color=TOTAL_COLOR, lw=1.85, zorder=4)
    ax.set_xlim(0.0, float(x.max()))
    ax.set_ylim(-4.25, 6.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2.0))
    ax.set_xlabel("Timesteps (millions) [training]")
    ax.set_ylabel("Mean reward per step")
    ax.margins(x=0)
    soften_axes(ax)
    add_panel_label(ax, "A")


def _plot_absolute_mass(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = data["timesteps"].to_numpy(dtype=float) / 1_000_000.0
    shares = [
        _rolling(data[f"absolute_share_{key}"]).to_numpy(dtype=float)
        for key in STACK_ORDER
    ]
    ax.stackplot(
        x,
        *shares,
        colors=[COMPONENT_BY_KEY[key]["color"] for key in STACK_ORDER],
        alpha=0.90,
        linewidth=0.0,
        zorder=2,
    )
    ax.set_xlim(0.0, float(x.max()))
    ax.set_ylim(0.0, 100.0)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0, decimals=0))
    ax.set_xlabel("Timesteps (millions) [training]")
    ax.set_ylabel("Share of absolute reward mass")
    ax.margins(x=0)
    soften_axes(ax)
    add_panel_label(ax, "B")


def _plot_training_path(
    ax: plt.Axes,
    data: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    key: str,
    panel: str,
    cmap_name: str,
    x_label: str,
    y_label: str,
    show_legend: bool,
    colorbar_bounds: tuple[float, float, float, float],
) -> None:
    progress = data["timesteps"].to_numpy(dtype=float) / 1_000_000.0
    x = _rolling(data[f"positive_days_{key}"]).to_numpy(dtype=float)
    y = _rolling(data[f"mean_positive_{key}"]).to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(progress)
    x = x[valid]
    y = y[valid]
    progress = progress[valid]
    if len(x) < 2:
        raise ValueError(f"Not enough finite training-path values for {key}")

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = _progress_cmap(cmap_name)
    norm = Normalize(vmin=float(progress.min()), vmax=float(progress.max()))
    line = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=1.4,
        alpha=0.80,
        zorder=3,
    )
    line.set_array(progress[1:])
    ax.add_collection(line)
    ax.scatter(
        x,
        y,
        c=progress,
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=0.46,
        edgecolor="none",
        rasterized=True,
        zorder=4,
    )
    ax.scatter(
        [float(x[-1])],
        [float(y[-1])],
        marker="D",
        s=48,
        color=END_COLOR,
        edgecolor="#111827",
        linewidth=0.75,
        zorder=8,
        label="End of training",
    )
    eval_x = float(evaluation.loc[key, "positive_reward_days_pct"])
    eval_y = float(evaluation.loc[key, "mean_positive_reward"])
    ax.scatter(
        [eval_x],
        [eval_y],
        marker="*",
        s=118,
        color=SELECTED_COLOR,
        edgecolor="white",
        linewidth=0.9,
        zorder=8,
        label="Selected-policy evaluation",
    )
    if show_legend:
        legend = ax.legend(
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="#D6DEE6",
            borderpad=0.40,
            handletextpad=0.35,
            fontsize=7.2,
        )
        legend.get_frame().set_linewidth(0.8)

    color_ax = ax.inset_axes(colorbar_bounds)
    colorbar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_mappable.set_array([])
    colorbar = ax.figure.colorbar(
        colorbar_mappable,
        cax=color_ax,
        orientation="horizontal",
    )
    colorbar.set_ticks([float(progress.min()), float(progress.max())])
    colorbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    colorbar.ax.tick_params(labelsize=6.8, length=2.0, pad=1.0)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.6)
    colorbar.set_label("Training timestep (millions)", fontsize=7.0, labelpad=1.0)

    x_span = max(float(np.ptp(x)), abs(eval_x - float(np.mean(x))), 1.0)
    y_span = max(float(np.ptp(y)), abs(eval_y - float(np.mean(y))), 0.25)
    ax.set_xlim(min(float(x.min()), eval_x) - 0.08 * x_span, max(float(x.max()), eval_x) + 0.08 * x_span)
    ax.set_ylim(
        max(0.0, min(float(y.min()), eval_y) - 0.10 * y_span),
        max(float(y.max()), eval_y) + 0.12 * y_span,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    soften_axes(ax)
    add_panel_label(ax, panel)


def _component_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=COMPONENT_BY_KEY[key]["color"],
            lw=2.2,
            label=COMPONENT_BY_KEY[key]["label"],
        )
        for key in LEGEND_ORDER
    ]
    handles.append(Line2D([0], [0], color=TOTAL_COLOR, lw=2.5, label="Total"))
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=8,
        frameon=True,
        columnspacing=0.85,
        handlelength=1.25,
        handletextpad=0.34,
        borderpad=0.45,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D6DEE6")
    legend.get_frame().set_linewidth(0.8)


def build_figure() -> list[Path]:
    training, evaluation = _load_data()
    set_theme()
    mpl.rcParams.update(
        {
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8))
    _plot_training_rewards(axes[0, 0], training)
    _plot_absolute_mass(axes[0, 1], training)
    _plot_training_path(
        axes[1, 0],
        training,
        evaluation,
        key="spring_peak_release",
        panel="C",
        cmap_name="Purples",
        x_label="Positive SPR-reward days (%)",
        y_label="Mean positive SPR reward",
        show_legend=True,
        colorbar_bounds=(0.53, 0.90, 0.42, 0.027),
    )
    _plot_training_path(
        axes[1, 1],
        training,
        evaluation,
        key="storage",
        panel="D",
        cmap_name="Blues",
        x_label="Positive storage-reward days (%)",
        y_label="Mean positive storage reward",
        show_legend=False,
        colorbar_bounds=(0.53, 0.13, 0.42, 0.027),
    )
    _component_legend(fig)
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.16,
        top=0.975,
        wspace=0.22,
        hspace=0.25,
    )
    return save_figure(fig, HERE, "training-reward-signals", dpi=320)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh compact tracked extracts from the rich diagnostic run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_data:
        _refresh_data()
    for output in build_figure():
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
