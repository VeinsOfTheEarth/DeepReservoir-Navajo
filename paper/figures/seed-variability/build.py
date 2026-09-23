"""Build section-3.7 prototype figures for seed variability and design lessons."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "paper" / "figure-support"))
from figurestyle import OBJECTIVE_COLORS
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "data"

SELECTED_SEED = "seed_004"
SEED_AXIS_MIN = 0.5
SEED_AXIS_MAX = 2.55
SEED_FLOOR_Y = 0.56
SEED_CAP_Y = 2.51
DEFAULT_SEED_METRICS = DATA_DIR / "selected-policy-family-seed-metrics.csv"
DEFAULT_BENCHMARK_METRICS = DATA_DIR / "selected-policy-metrics.csv"

BLUE = "#2B6CB0"
BLUE_LIGHT = "#BBD7F0"
GRAY = "#8B95A5"
GRAY_LIGHT = "#E5E7EB"
GREEN = OBJECTIVE_COLORS["niip"]
PURPLE = OBJECTIVE_COLORS["spr"]
ORANGE = "#D97706"
RED = "#C44536"
TEXT = "#111827"
MUTED = "#6B7280"
GRID = "#E2E8F0"
SPINE = "#94A3B8"
BACKGROUND = "#F4F7FA"

SEED_LINE_COLORS = [
    "#4E79A7",
    "#59A14F",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#B07AA1",
    "#EDC948",
    "#9C755F",
    "#BAB0AC",
    "#6B6ECF",
    "#8CD17D",
    "#D37295",
    "#499894",
    "#FABFD2",
    "#A0CBE8",
    "#FFBE7D",
]


OBJECTIVES = [
    {
        "label": "SPR\n10k/5d",
        "column": "spr_freq_years_meeting_10000cfs_5d",
        "mode": "ratio",
        "objective": "Spring peak release",
        "metric": "10,000 cfs / 5 d frequency",
        "color": PURPLE,
    },
    {
        "label": "SPR\n8k/10d",
        "column": "spr_freq_years_meeting_8000cfs_10d",
        "mode": "ratio",
        "objective": "Spring peak release",
        "metric": "8,000 cfs / 10 d frequency",
        "color": PURPLE,
    },
    {
        "label": "SPR\n5k/21d",
        "column": "spr_freq_years_meeting_5000cfs_21d",
        "mode": "ratio",
        "objective": "Spring peak release",
        "metric": "5,000 cfs / 21 d frequency",
        "color": PURPLE,
    },
    {
        "label": "SPR\n2.5k/10d",
        "column": "spr_freq_years_meeting_2500cfs_10d",
        "mode": "ratio",
        "objective": "Spring peak release",
        "metric": "2,500 cfs / 10 d frequency",
        "color": PURPLE,
    },
    {
        "label": "Storage",
        "column": "storage_frac_of_max_possible",
        "mode": "ratio",
        "objective": "Storage",
        "metric": "Storage / maximum possible",
        "color": OBJECTIVE_COLORS["storage"],
    },
    {
        "label": "Spill-free\ndays",
        "column": "frac_days_spilling",
        "mode": "one_minus",
        "objective": "Dam safety",
        "metric": "Spill-free evaluation days",
        "color": OBJECTIVE_COLORS["dam_safety"],
    },
    {
        "label": "Flood\nsafety",
        "column": "flooding_frac_days_met",
        "mode": "ratio",
        "objective": "Flood control",
        "metric": "Flood-safe days",
        "color": OBJECTIVE_COLORS["flood"],
    },
    {
        "label": "ESA\nmin flow",
        "column": "esa_min_flow_frac_days_met",
        "mode": "ratio",
        "objective": "ESA minimum flow",
        "metric": "Days meeting 500 cfs Farmington minimum",
        "color": OBJECTIVE_COLORS["esa"],
    },
    {
        "label": "NIIP\nvolume",
        "column": "niip_annual_volume_frac_of_contract",
        "mode": "ratio",
        "objective": "NIIP",
        "metric": "Annual volume / contract",
        "color": GREEN,
    },
    {
        "label": "Hydro-\npower",
        "column": "hydropower_frac_of_max_possible",
        "mode": "ratio",
        "objective": "Hydropower",
        "metric": "Hydropower / maximum possible",
        "color": OBJECTIVE_COLORS["hydropower"],
    },
]


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
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = available


def _set_theme() -> None:
    _register_fonts()
    mpl.rcParams.update(
        {
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
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.80,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.4,
            "lines.dash_capstyle": "round",
            "lines.solid_capstyle": "round",
            "savefig.bbox": "tight",
        }
    )


def _soften_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.9)


def _seed_key(seed: object) -> int:
    match = re.search(r"(\d+)$", str(seed))
    return int(match.group(1)) if match else 10_000


def _load_benchmark_table(path: Path) -> dict[tuple[str, str], float]:
    table = pd.read_csv(path)
    out: dict[tuple[str, str], float] = {}
    for _, row in table.iterrows():
        out[(str(row["objective"]), str(row["metric"]))] = float(row["benchmark"])
    return out


def _objective_values(seed_metrics: pd.DataFrame, benchmarks: dict[tuple[str, str], float]) -> pd.DataFrame:
    values: dict[str, np.ndarray] = {}
    for spec in OBJECTIVES:
        column = str(spec["column"])
        label = str(spec["label"])
        series = pd.to_numeric(seed_metrics[column], errors="coerce").astype(float)
        if spec["mode"] == "one_minus":
            values[label] = (1.0 - series).clip(lower=0.0, upper=1.0).to_numpy()
            continue
        benchmark = benchmarks[(str(spec["objective"]), str(spec["metric"]))]
        if not math.isfinite(benchmark) or benchmark <= 0:
            raise ValueError(f"Benchmark for {spec['objective']} / {spec['metric']} must be positive")
        values[label] = (series / benchmark).to_numpy()
    return pd.DataFrame(values, index=seed_metrics["seed"])


def _save(fig: plt.Figure, output_dir: Path, stem: str, *, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    temp_dir = Path(tempfile.gettempdir())
    tmp_paths = [
        temp_dir / f"{stem}.{os.getpid()}.tmp.png",
        temp_dir / f"{stem}.{os.getpid()}.tmp.pdf",
    ]
    fig.savefig(tmp_paths[0], dpi=dpi)
    fig.savefig(tmp_paths[1])
    for tmp_path, path in zip(tmp_paths, paths):
        last_error: PermissionError | None = None
        for attempt in range(6):
            try:
                path.unlink(missing_ok=True)
                shutil.copyfile(tmp_path, path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.25)
        if last_error is not None:
            raise last_error
        tmp_path.unlink(missing_ok=True)
    plt.close(fig)
    return paths


def build_seed_objective_variability(
    *,
    seed_metrics_path: Path,
    benchmark_metrics_path: Path,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    seed_metrics = pd.read_csv(seed_metrics_path).sort_values("seed", key=lambda s: s.map(_seed_key))
    benchmarks = _load_benchmark_table(benchmark_metrics_path)
    objective_df = _objective_values(seed_metrics, benchmarks)

    labels = list(objective_df.columns)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.75), constrained_layout=True)

    q1 = objective_df.quantile(0.25, axis=0).to_numpy(dtype=float)
    q3 = objective_df.quantile(0.75, axis=0).to_numpy(dtype=float)
    med = objective_df.median(axis=0).to_numpy(dtype=float)
    ymin = objective_df.min(axis=0).to_numpy(dtype=float)
    ymax = objective_df.max(axis=0).to_numpy(dtype=float)
    seed_offsets = np.linspace(-0.085, 0.085, len(objective_df)) if len(objective_df) > 1 else np.array([0.0])
    has_below_axis = bool((objective_df < SEED_AXIS_MIN).to_numpy().any())
    has_above_axis = bool((objective_df > SEED_AXIS_MAX).to_numpy().any())

    for xi, spec, lo, hi, lo_iqr, hi_iqr, median in zip(x, OBJECTIVES, ymin, ymax, q1, q3, med):
        color = str(spec["color"])
        clipped_lo = max(float(lo), SEED_AXIS_MIN)
        clipped_hi = min(float(hi), SEED_CAP_Y)
        clipped_q1 = max(float(lo_iqr), SEED_AXIS_MIN)
        clipped_q3 = min(float(hi_iqr), SEED_CAP_Y)
        clipped_med = min(max(float(median), SEED_AXIS_MIN), SEED_CAP_Y)

        ax.vlines(xi, clipped_lo, clipped_hi, color=SPINE, linewidth=1.15, alpha=0.78, zorder=2)
        if clipped_q3 >= clipped_q1:
            ax.vlines(xi, clipped_q1, clipped_q3, color=color, linewidth=8.0, alpha=0.34, zorder=3)
        ax.hlines(clipped_med, xi - 0.13, xi + 0.13, color=TEXT, linewidth=2.0, zorder=6)

        if float(lo) < SEED_AXIS_MIN:
            ax.scatter(
                [xi],
                [SEED_FLOOR_Y],
                color=color,
                alpha=0.70,
                marker="v",
                s=46,
                facecolor="white",
                linewidth=1.0,
                zorder=7,
            )

        if float(hi) > SEED_AXIS_MAX:
            ax.scatter(
                [xi],
                [SEED_CAP_Y],
                color=color,
                alpha=0.70,
                marker="^",
                s=46,
                facecolor="white",
                linewidth=1.0,
                zorder=7,
            )

    for seed_offset, (_, row) in zip(seed_offsets, objective_df.iterrows()):
        values = row.to_numpy(dtype=float)
        low = values < SEED_AXIS_MIN
        high = values > SEED_AXIS_MAX
        in_range = ~(low | high)
        if bool(in_range.any()):
            ax.scatter(
                x[in_range] + seed_offset,
                values[in_range],
                s=16,
                marker="o",
                color="#B8C2D0",
                edgecolor=TEXT,
                linewidth=0.22,
                alpha=0.86,
                zorder=5,
            )
        if bool(low.any()):
            ax.scatter(
                x[low] + seed_offset,
                np.full(int(low.sum()), SEED_FLOOR_Y),
                s=20,
                marker="v",
                color="#B8C2D0",
                edgecolor=TEXT,
                linewidth=0.22,
                alpha=0.86,
                zorder=5,
            )
        if bool(high.any()):
            ax.scatter(
                x[high] + seed_offset,
                np.full(int(high.sum()), SEED_CAP_Y),
                s=20,
                marker="^",
                color="#B8C2D0",
                edgecolor=TEXT,
                linewidth=0.22,
                alpha=0.86,
                zorder=5,
            )

    if SELECTED_SEED in objective_df.index:
        selected = objective_df.loc[SELECTED_SEED].to_numpy(dtype=float)
        selected_y = np.where(selected < SEED_AXIS_MIN, SEED_FLOOR_Y, np.minimum(selected, SEED_CAP_Y))
        selected_markers = np.where(selected < SEED_AXIS_MIN, "v", np.where(selected > SEED_AXIS_MAX, "^", "o"))
        for marker in sorted(set(selected_markers)):
            mask = selected_markers == marker
            ax.scatter(
                x[mask],
                selected_y[mask],
                color=BLUE,
                marker=marker,
                s=32 if marker == "o" else 40,
                facecolor=TEXT,
                edgecolor="white",
                linewidth=0.35,
                zorder=8,
            )

    ax.axhline(1.0, color=TEXT, linewidth=0.95, linestyle=(0, (3.2, 2.4)), alpha=0.72, label="Benchmark")
    ax.axhline(2.0, color=SPINE, linewidth=0.8, linestyle=(0, (1.2, 2.0)), alpha=0.65)
    ax.set_ylim(SEED_AXIS_MIN, SEED_AXIS_MAX)
    ax.set_xlim(-0.30, len(labels) - 0.70)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Ratio to historic benchmark or target")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B8C2D0", markeredgecolor=TEXT, markeredgewidth=0.35, markersize=4.8, label="Seed"),
        Line2D([0], [0], color=SPINE, linewidth=1.35, label="Range"),
        Line2D([0], [0], color=BLUE, linewidth=5.0, alpha=0.34, label="IQR"),
        Line2D([0], [0], color=TEXT, linewidth=2.0, label="Median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEXT, markeredgecolor="white", markeredgewidth=0.35, markersize=5.9, label="Selected policy"),
        Line2D([0], [0], color=TEXT, linewidth=0.95, linestyle=(0, (3.2, 2.4)), alpha=0.72, label="Benchmark"),
    ]
    if has_below_axis:
        legend_handles.append(
            Line2D([0], [0], marker="v", color=SPINE, markerfacecolor="white", markeredgewidth=1.0, markersize=6.4, linestyle="none", label="Min below axis")
        )
    if has_above_axis:
        legend_handles.append(
            Line2D([0], [0], marker="^", color=SPINE, markerfacecolor="white", markeredgewidth=1.0, markersize=6.4, linestyle="none", label="Max above axis")
        )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=len(legend_handles),
        frameon=False,
        handlelength=1.0,
        columnspacing=0.60,
        handletextpad=0.35,
    )
    _soften_axes(ax)
    return _save(fig, output_dir, "seed-variability-groups", dpi=dpi)


def build_seed_objective_connected_lines(
    *,
    seed_metrics_path: Path,
    benchmark_metrics_path: Path,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    seed_metrics = pd.read_csv(seed_metrics_path).sort_values("seed", key=lambda s: s.map(_seed_key))
    benchmarks = _load_benchmark_table(benchmark_metrics_path)
    objective_df = _objective_values(seed_metrics, benchmarks)

    labels = list(objective_df.columns)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.75), constrained_layout=True)

    has_below_axis = bool((objective_df < SEED_AXIS_MIN).to_numpy().any())
    has_above_axis = bool((objective_df > SEED_AXIS_MAX).to_numpy().any())
    raw_values = objective_df.to_numpy(dtype=float)
    visible_y = np.where(raw_values < SEED_AXIS_MIN, SEED_FLOOR_Y, np.where(raw_values > SEED_AXIS_MAX, SEED_CAP_Y, raw_values))
    x_offsets = np.zeros_like(visible_y)
    for col_index in range(visible_y.shape[1]):
        rounded = np.round(visible_y[:, col_index], 8)
        for value in np.unique(rounded):
            tied = np.flatnonzero(rounded == value)
            if len(tied) > 1:
                width = min(0.18, 0.018 * (len(tied) - 1))
                x_offsets[tied, col_index] = np.linspace(-width, width, len(tied))

    for seed_index, (seed, row) in enumerate(objective_df.iterrows()):
        values = row.to_numpy(dtype=float)
        low = values < SEED_AXIS_MIN
        high = values > SEED_AXIS_MAX
        in_range = ~(low | high)
        y = visible_y[seed_index]
        xj = x + x_offsets[seed_index]

        is_selected = str(seed) == SELECTED_SEED
        seed_color = SEED_LINE_COLORS[seed_index % len(SEED_LINE_COLORS)]
        line_color = TEXT if is_selected else seed_color
        point_face = TEXT if is_selected else seed_color
        line_width = 1.55 if is_selected else 1.05
        alpha = 0.96 if is_selected else 0.58
        z = 6 if is_selected else 2

        ax.plot(xj, y, color=line_color, linewidth=line_width, alpha=alpha, zorder=z)
        if bool(in_range.any()):
            ax.scatter(
                xj[in_range],
                y[in_range],
                s=28 if is_selected else 14,
                marker="o",
                facecolor=point_face,
                edgecolor="white" if is_selected else TEXT,
                linewidth=0.35 if is_selected else 0.20,
                alpha=0.98 if is_selected else 0.82,
                zorder=z + 1,
            )
        if bool(low.any()):
            ax.scatter(
                xj[low],
                y[low],
                s=34 if is_selected else 20,
                marker="v",
                facecolor=point_face,
                edgecolor="white" if is_selected else TEXT,
                linewidth=0.35 if is_selected else 0.20,
                alpha=0.98 if is_selected else 0.82,
                zorder=z + 1,
            )
        if bool(high.any()):
            ax.scatter(
                xj[high],
                y[high],
                s=34 if is_selected else 20,
                marker="^",
                facecolor=point_face,
                edgecolor="white" if is_selected else TEXT,
                linewidth=0.35 if is_selected else 0.20,
                alpha=0.98 if is_selected else 0.82,
                zorder=z + 1,
            )

    ax.axhline(1.0, color=TEXT, linewidth=0.95, linestyle=(0, (3.2, 2.4)), alpha=0.72)
    ax.axhline(2.0, color=SPINE, linewidth=0.8, linestyle=(0, (1.2, 2.0)), alpha=0.65)
    ax.set_ylim(SEED_AXIS_MIN, SEED_AXIS_MAX)
    ax.set_xlim(-0.30, len(labels) - 0.70)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Ratio to historic benchmark or target")

    legend_handles = [
        Line2D([0], [0], marker="o", color=SEED_LINE_COLORS[0], markerfacecolor=SEED_LINE_COLORS[0], markeredgecolor=TEXT, markeredgewidth=0.25, linewidth=1.05, alpha=0.75, markersize=4.8, label="Seed"),
        Line2D([0], [0], marker="o", color=TEXT, markerfacecolor=TEXT, markeredgecolor="white", markeredgewidth=0.35, linewidth=1.35, markersize=5.4, label="Selected policy"),
        Line2D([0], [0], color=TEXT, linewidth=0.95, linestyle=(0, (3.2, 2.4)), alpha=0.72, label="Benchmark"),
    ]
    if has_below_axis:
        legend_handles.append(
            Line2D([0], [0], marker="v", color=SPINE, markerfacecolor="white", markeredgewidth=1.0, markersize=6.4, linestyle="none", label="Min below axis")
        )
    if has_above_axis:
        legend_handles.append(
            Line2D([0], [0], marker="^", color=SPINE, markerfacecolor="white", markeredgewidth=1.0, markersize=6.4, linestyle="none", label="Max above axis")
        )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=len(legend_handles),
        frameon=False,
        handlelength=1.0,
        columnspacing=0.70,
        handletextpad=0.40,
    )
    _soften_axes(ax)
    return _save(fig, output_dir, "seed-variability", dpi=dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-metrics", type=Path, default=DEFAULT_SEED_METRICS)
    parser.add_argument("--benchmark-metrics", type=Path, default=DEFAULT_BENCHMARK_METRICS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_theme()
    written: list[Path] = []
    written.extend(
        build_seed_objective_connected_lines(
            seed_metrics_path=args.seed_metrics,
            benchmark_metrics_path=args.benchmark_metrics,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
