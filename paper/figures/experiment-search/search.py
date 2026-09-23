"""Figure-specific helpers extracted without changing plot calculations."""

from __future__ import annotations

import argparse

import math

import os
import sys

import shutil

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
from figurestyle import add_panel_label

OUTPUT_DIR = Path(__file__).resolve().parent

DATA_DIR = OUTPUT_DIR / "data"

SELECTED_SEED = "seed_004"

DEFAULT_PHASE95_SEED_METRICS = DATA_DIR / "phase95-seed-metrics-with-pass-flags.csv"

DEFAULT_PHASE95_BENCHMARK = DATA_DIR / "phase95-historic-benchmark.csv"

ARCHIVE_REQUIRED_METRICS = (
    "storage_frac_of_max_possible",
    "hydropower_frac_of_max_possible",
    "esa_min_flow_frac_days_met",
    "flooding_frac_days_met",
    "spr_freq_years_meeting_10000cfs_5d",
    "spr_freq_years_meeting_8000cfs_10d",
    "spr_freq_years_meeting_5000cfs_21d",
    "spr_freq_years_meeting_2500cfs_10d",
    "niip_annual_volume_frac_of_contract",
)


BLUE = "#2B6CB0"

GREEN = "#2CA25F"

ORANGE = "#D97706"

TEXT = "#111827"

MUTED = "#6B7280"

GRID = "#E2E8F0"

SPINE = "#94A3B8"

BACKGROUND = "#F4F7FA"

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

def _add_panel_label(ax: plt.Axes, label: str, *, loc: str = "left") -> None:
    x = 0.98 if loc == "right" else 0.02
    ha = "right" if loc == "right" else "left"
    add_panel_label(ax, label, x=x, ha=ha, zorder=30)

def _save(fig: plt.Figure, output_dir: Path, stem: str, *, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem.replace("_", "-")
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
                try:
                    path.unlink(missing_ok=True)
                except PermissionError:
                    if path.exists():
                        raise
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

def _load_phase95_benchmark(path: Path) -> dict[str, float]:
    table = pd.read_csv(path)
    return {str(row["metric"]): float(row["historic"]) for _, row in table.iterrows()}

def _phase95_pass_columns(df: pd.DataFrame) -> list[str]:
    return [
        "no_spill",
        "pass_storage_frac_of_max_possible",
        "pass_esa_min_flow_frac_days_met",
        "pass_flooding_frac_days_met",
        "pass_spr_freq_years_meeting_10000cfs_5d",
        "pass_spr_freq_years_meeting_8000cfs_10d",
        "pass_spr_freq_years_meeting_5000cfs_21d",
        "pass_spr_freq_years_meeting_2500cfs_10d",
        "pass_hydropower_frac_of_max_possible",
        "pass_niip_annual_volume_frac_of_contract",
    ]

def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])

def _search_metric_frame(df: pd.DataFrame, benchmark: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ratio_columns = {
        "storage_ratio": "storage_frac_of_max_possible",
        "hydropower_ratio": "hydropower_frac_of_max_possible",
        "esa_ratio": "esa_min_flow_frac_days_met",
        "flood_ratio": "flooding_frac_days_met",
        "spr_10k_ratio": "spr_freq_years_meeting_10000cfs_5d",
        "spr_8k_ratio": "spr_freq_years_meeting_8000cfs_10d",
        "spr_5k_ratio": "spr_freq_years_meeting_5000cfs_21d",
        "spr_2p5k_ratio": "spr_freq_years_meeting_2500cfs_10d",
        "niip_annual_ratio": "niip_annual_volume_frac_of_contract",
    }
    for out_col, source_col in ratio_columns.items():
        out[out_col] = pd.to_numeric(df[source_col], errors="coerce") / benchmark[source_col]

    metric_pass = []
    if "total_spill_af" in df.columns:
        spill_pass = pd.to_numeric(df["total_spill_af"], errors="coerce").fillna(np.inf) <= 1e-6
    elif "frac_days_spilling" in df.columns:
        spill_pass = pd.to_numeric(df["frac_days_spilling"], errors="coerce").fillna(np.inf) <= 1e-12
    else:
        spill_pass = pd.Series(False, index=df.index)
    out["no_spill_score"] = spill_pass.astype(float)
    metric_pass.append(spill_pass)
    metric_pass.extend(
        [
            pd.to_numeric(df["storage_frac_of_max_possible"], errors="coerce")
            >= benchmark["storage_frac_of_max_possible"] - 1e-12,
            pd.to_numeric(df["esa_min_flow_frac_days_met"], errors="coerce")
            >= benchmark["esa_min_flow_frac_days_met"] - 1e-12,
            pd.to_numeric(df["flooding_frac_days_met"], errors="coerce")
            >= benchmark["flooding_frac_days_met"] - 1e-12,
            pd.to_numeric(df["spr_freq_years_meeting_10000cfs_5d"], errors="coerce")
            >= benchmark["spr_freq_years_meeting_10000cfs_5d"] - 1e-12,
            pd.to_numeric(df["spr_freq_years_meeting_8000cfs_10d"], errors="coerce")
            >= benchmark["spr_freq_years_meeting_8000cfs_10d"] - 1e-12,
            pd.to_numeric(df["spr_freq_years_meeting_5000cfs_21d"], errors="coerce")
            >= benchmark["spr_freq_years_meeting_5000cfs_21d"] - 1e-12,
            pd.to_numeric(df["spr_freq_years_meeting_2500cfs_10d"], errors="coerce")
            >= benchmark["spr_freq_years_meeting_2500cfs_10d"] - 1e-12,
            pd.to_numeric(df["hydropower_frac_of_max_possible"], errors="coerce")
            >= benchmark["hydropower_frac_of_max_possible"] - 1e-12,
            pd.to_numeric(df["niip_annual_volume_frac_of_contract"], errors="coerce")
            >= benchmark["niip_annual_volume_frac_of_contract"] - 1e-12,
        ]
    )
    pass_df = pd.concat(metric_pass, axis=1)
    out["pass_count_clean"] = pass_df.sum(axis=1)
    out["finish_line_clean"] = pass_df.all(axis=1)
    cap = 2.0
    spr_scores = out[["spr_10k_ratio", "spr_8k_ratio", "spr_5k_ratio", "spr_2p5k_ratio"]].clip(
        lower=0.0,
        upper=cap,
    )
    out["spr_composite_ratio"] = spr_scores.mean(axis=1)
    out["conservation_safety_score"] = out[["storage_ratio", "flood_ratio", "no_spill_score"]].clip(
        lower=0.0,
        upper=cap,
    ).mean(axis=1)
    out["beneficial_use_score"] = pd.concat(
        [
            out["spr_composite_ratio"],
            out["esa_ratio"],
            out["niip_annual_ratio"],
            out["hydropower_ratio"],
        ],
        axis=1,
    ).clip(lower=0.0, upper=cap).mean(axis=1)
    objective_scores = pd.concat(
        [
            out["no_spill_score"],
            out["storage_ratio"],
            out["esa_ratio"],
            out["flood_ratio"],
            out["spr_10k_ratio"],
            out["spr_8k_ratio"],
            out["spr_5k_ratio"],
            out["spr_2p5k_ratio"],
            out["hydropower_ratio"],
            out["niip_annual_ratio"],
        ],
        axis=1,
    ).clip(lower=0.0, upper=cap)
    out["min_objective_ratio"] = objective_scores.min(axis=1)
    out["total_objective_shortfall"] = (1.0 - objective_scores.clip(upper=1.0)).clip(lower=0.0).sum(axis=1)
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["storage_ratio", "hydropower_ratio"])


def _select_comparable_archive_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep comparable policy evaluations with all retained screen results."""
    phase = pd.to_numeric(raw["phase"], errors="coerce")
    window = raw["window"].astype("string")
    stage = raw["stage"].astype("string")

    comparable_period = ~phase.eq(78) | window.eq("holdout_2014_2024_08_17").fillna(False)
    policy_record = ~phase.eq(79) | ~stage.eq("historical").fillna(False)
    complete_metrics = (
        raw.loc[:, ARCHIVE_REQUIRED_METRICS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    spill_screen_recorded = pd.to_numeric(raw["total_spill_af"], errors="coerce").notna()
    keep = comparable_period & policy_record & complete_metrics & spill_screen_recorded
    return raw.loc[keep].copy()


def _load_archived_search_metrics(benchmark: dict[str, float]) -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "archived-search-metrics.csv", float_precision="round_trip", low_memory=False)
    raw = _select_comparable_archive_records(raw)
    archive = _search_metric_frame(raw, benchmark)
    for column in ("source", "phase", "report_run"):
        archive[column] = raw.loc[archive.index, column]
    return archive.reset_index(drop=True)


def build_experiment_search(
    *,
    seed_metrics_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    dpi: int,
    only_stems: set[str] | None = None,
) -> list[Path]:
    seeds = pd.read_csv(seed_metrics_path)
    benchmark = _load_phase95_benchmark(benchmark_path)
    archive = _load_archived_search_metrics(benchmark)
    pass_cols = _phase95_pass_columns(seeds)
    pass_df = pd.DataFrame({col: _as_bool(seeds[col]) for col in pass_cols})
    seeds["pass_count_clean"] = pass_df.sum(axis=1)
    seeds["finish_line_clean"] = pass_df.all(axis=1)
    seeds["storage_ratio"] = (
        pd.to_numeric(seeds["storage_frac_of_max_possible"], errors="coerce")
        / benchmark["storage_frac_of_max_possible"]
    )
    seeds["hydropower_ratio"] = (
        pd.to_numeric(seeds["hydropower_frac_of_max_possible"], errors="coerce")
        / benchmark["hydropower_frac_of_max_possible"]
    )
    seeds["phase"] = 95
    score_frame = _search_metric_frame(seeds, benchmark)
    score_cols = [
        "esa_ratio",
        "flood_ratio",
        "spr_10k_ratio",
        "spr_8k_ratio",
        "spr_5k_ratio",
        "spr_2p5k_ratio",
        "niip_annual_ratio",
        "no_spill_score",
        "spr_composite_ratio",
        "conservation_safety_score",
        "beneficial_use_score",
        "min_objective_ratio",
        "total_objective_shortfall",
    ]
    for col in score_cols:
        seeds[col] = score_frame[col]

    selected_mask = (seeds["family"] == "reward_jon_p95_peak875_hdisceff") & (seeds["seed"] == SELECTED_SEED)
    seeds["is_selected_policy"] = selected_mask
    total = len(seeds)
    finish_count = int(seeds["finish_line_clean"].sum())
    archive_finish_count = int(archive["finish_line_clean"].sum())
    all_finish_count = finish_count + archive_finish_count

    plot_cols = [
        "storage_ratio",
        "hydropower_ratio",
        "pass_count_clean",
        "phase",
        "is_phase95",
        "is_selected_policy",
        "finish_line_clean",
        *score_cols,
    ]
    archive_for_plot = archive.copy()
    if "phase" not in archive_for_plot.columns:
        archive_for_plot["phase"] = np.nan
    if not archive_for_plot.empty:
        archive_for_plot["is_phase95"] = False
        archive_for_plot["is_selected_policy"] = False
    seeds_for_plot = seeds.copy()
    seeds_for_plot["is_phase95"] = True
    combined = pd.concat([archive_for_plot[plot_cols], seeds_for_plot[plot_cols]], ignore_index=True)

    axis_specs = {
        "storage_hydro": {
            "x_col": "storage_ratio",
            "y_col": "hydropower_ratio",
            "xlabel": "Storage / historic",
            "ylabel": "Hydropower / historic",
            "x_ref": 1.0,
            "y_ref": 1.0,
            "upper_cap": 1.55,
            "minimum_upper": 1.24,
            "jitter_x": False,
        },
        "conservation_beneficial": {
            "x_col": "conservation_safety_score",
            "y_col": "beneficial_use_score",
            "xlabel": "Conservation/safety score",
            "ylabel": "Beneficial-use score",
            "x_ref": 1.0,
            "y_ref": 1.0,
            "upper_cap": 2.05,
            "minimum_upper": 1.24,
            "jitter_x": False,
            "xlim": (0.3, 1.1),
            "ylim": (0.6, 1.4),
        },
        "pass_min": {
            "x_col": "pass_count_clean",
            "y_col": "min_objective_ratio",
            "xlabel": "Objectives passing",
            "ylabel": "Weakest objective / historic",
            "x_ref": 10.0,
            "y_ref": 1.0,
            "upper_cap": 1.55,
            "minimum_upper": 1.24,
            "jitter_x": True,
        },
        "pass_shortfall": {
            "x_col": "pass_count_clean",
            "y_col": "total_objective_shortfall",
            "xlabel": "Objectives passing",
            "ylabel": "Total shortfall from historic",
            "x_ref": 10.0,
            "y_ref": 0.0,
            "upper_cap": None,
            "minimum_upper": 1.0,
            "jitter_x": True,
        },
    }

    def draw_histogram(ax_hist: plt.Axes, *, mode: str = "bars") -> None:
        bins = np.arange(0, len(pass_cols) + 1)
        visible_bins = np.arange(3, len(pass_cols) + 1)

        def phase_decade_group(phase: float) -> str:
            if phase < 68:
                return "58-67"
            if phase < 78:
                return "68-77"
            if phase < 88:
                return "78-87"
            if phase < 95:
                return "88-94"
            return "Phase 95"

        if mode == "phase_groups":
            grouped_hist = combined.dropna(subset=["phase", "pass_count_clean"]).copy()

            grouped_hist["phase_group"] = grouped_hist["phase"].astype(float).map(phase_decade_group)
            group_order = ["58-67", "68-77", "78-87", "88-94", "Phase 95"]
            group_colors = {
                "58-67": "#94A3B8",
                "68-77": BLUE,
                "78-87": GREEN,
                "88-94": ORANGE,
                "Phase 95": ORANGE,
            }
            for group in group_order:
                group_values = grouped_hist.loc[grouped_hist["phase_group"] == group, "pass_count_clean"]
                if group_values.empty:
                    continue
                share = group_values.value_counts().reindex(bins, fill_value=0) / len(group_values) * 100.0
                ax_hist.step(
                    bins,
                    share,
                    where="mid",
                    color=group_colors[group],
                    linewidth=1.8,
                    alpha=0.94,
                    label=f"Phase {group}",
                )
            ax_hist.set_xlabel("Objectives passing")
            ax_hist.set_ylabel("Seed share within group (%)")
            ax_hist.set_xlim(2.45, 10.55)
            ax_hist.set_xticks(range(3, len(pass_cols) + 1, 1))
            ax_hist.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.5)
            _soften_axes(ax_hist)
            return

        if mode == "stacked_phase_histograms":
            grouped_hist = combined.dropna(subset=["phase", "pass_count_clean"]).copy()
            grouped_hist["phase_group"] = grouped_hist["phase"].astype(float).map(phase_decade_group)
            group_order = ["58-67", "68-77", "78-87", "88-94", "Phase 95"]
            group_labels = {
                "88-94": "Phases 88-94",
                "78-87": "Phases 78-87",
                "68-77": "Phases 68-77",
                "58-67": "Phases 58-67",
                "Phase 95": "Phase 95",
            }
            row_gap = 1.0
            row_height = 0.70
            shares: dict[str, pd.Series] = {}
            max_share = 0.0
            for group in group_order:
                group_values = grouped_hist.loc[grouped_hist["phase_group"] == group, "pass_count_clean"]
                if group_values.empty:
                    continue
                share = group_values.value_counts().reindex(visible_bins, fill_value=0) / len(group_values) * 100.0
                shares[group] = share
                max_share = max(max_share, float(share.max()))
            max_share = max(max_share, 1.0)
            bar_color = "#64748B"
            for row_index, group in enumerate(group_order):
                if group not in shares:
                    continue
                baseline = (len(group_order) - 1 - row_index) * row_gap
                scaled = shares[group] / max_share * row_height
                ax_hist.bar(
                    visible_bins,
                    scaled,
                    bottom=baseline,
                    width=0.72,
                    color=bar_color,
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.45,
                )
                ax_hist.hlines(
                    baseline,
                    visible_bins.min() - 0.45,
                    visible_bins.max() + 0.45,
                    color=SPINE,
                    linewidth=0.65,
                    alpha=0.72,
                )
                label_x = visible_bins.max() + 0.32
                ax_hist.text(
                    label_x,
                    baseline + row_height * 0.88,
                    group_labels[group],
                    ha="right",
                    va="top",
                    fontsize=7.0,
                    color=TEXT,
                    bbox={
                        "boxstyle": "round,pad=0.16",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.70,
                    },
                )
            ax_hist.set_xlabel("Objectives passing")
            ax_hist.set_ylabel("Relative frequency", rotation=270, labelpad=15)
            ax_hist.yaxis.set_label_position("right")
            ax_hist.yaxis.tick_right()
            ax_hist.set_xlim(2.45, 10.55)
            ax_hist.set_xticks(range(3, len(pass_cols) + 1, 1))
            ax_hist.set_ylim(-0.12, (len(group_order) - 1) * row_gap + row_height + 0.15)
            ax_hist.set_yticks([])
            ax_hist.grid(axis="x", color=GRID, linestyle=":", linewidth=0.7, alpha=0.80)
            _soften_axes(ax_hist)
            ax_hist.spines["left"].set_visible(False)
            ax_hist.spines["right"].set_visible(True)
            ax_hist.spines["right"].set_color(SPINE)
            ax_hist.spines["right"].set_linewidth(0.9)
            return

        counts = seeds["pass_count_clean"].value_counts().reindex(bins, fill_value=0)
        phase95_share = counts / total * 100.0
        if not archive.empty:
            archive_counts = archive["pass_count_clean"].value_counts().reindex(bins, fill_value=0)
            archive_share = archive_counts / len(archive) * 100.0
            ax_hist.bar(
                bins - 0.20,
                archive_share,
                width=0.38,
                color="#CBD5E1",
                edgecolor="white",
                linewidth=0.55,
                label="All other phases",
            )
        ax_hist.bar(
            bins + 0.20,
            phase95_share,
            width=0.38,
            color=BLUE,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.55,
            label="Phase 95",
        )
        ax_hist.set_xlabel("Objectives passing")
        ax_hist.set_ylabel("Seed share (%)")
        ax_hist.set_xlim(2.45, 10.55)
        ax_hist.set_xticks(range(3, len(pass_cols) + 1, 1))
        ax_hist.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.1)
        _soften_axes(ax_hist)

    def axis_limits(spec: dict[str, object]) -> tuple[tuple[float, float], tuple[float, float]]:
        if "xlim" in spec and "ylim" in spec:
            return tuple(spec["xlim"]), tuple(spec["ylim"])  # type: ignore[return-value]
        x_col = str(spec["x_col"])
        y_col = str(spec["y_col"])
        if x_col == "pass_count_clean":
            xlim = (-0.55, 10.55)
        else:
            x_vals = pd.to_numeric(combined[x_col], errors="coerce").dropna()
            upper_cap = spec["upper_cap"]
            x_upper = max(float(spec["minimum_upper"]), float(x_vals.quantile(0.995)) + 0.06)
            if upper_cap is not None:
                x_upper = min(float(upper_cap), x_upper)
            xlim = (0.0, x_upper)

        y_vals = pd.to_numeric(combined[y_col], errors="coerce").dropna()
        if y_col == "total_objective_shortfall":
            y_upper = max(float(spec["minimum_upper"]), float(y_vals.quantile(0.995)) + 0.18)
            ylim = (-0.05, y_upper)
        else:
            upper_cap = spec["upper_cap"]
            y_upper = max(float(spec["minimum_upper"]), float(y_vals.quantile(0.995)) + 0.05)
            if upper_cap is not None:
                y_upper = min(float(upper_cap), y_upper)
            ylim = (0.0, y_upper)
        return xlim, ylim

    def draw_base_axes(ax: plt.Axes, spec: dict[str, object], *, show_count_text: bool = True) -> None:
        x_ref = spec.get("x_ref")
        y_ref = spec.get("y_ref")
        if x_ref is not None:
            ax.axvline(float(x_ref), color=TEXT, lw=0.9, ls=(0, (3.0, 2.4)), alpha=0.72)
        if y_ref is not None:
            ax.axhline(float(y_ref), color=TEXT, lw=0.9, ls=(0, (3.0, 2.4)), alpha=0.72)
        ax.set_xlabel(str(spec["xlabel"]))
        ax.set_ylabel(str(spec["ylabel"]))
        xlim, ylim = axis_limits(spec)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if spec["x_col"] == "pass_count_clean":
            ax.set_xticks(range(0, len(pass_cols) + 1, 2))
        if show_count_text:
            ax.text(
                0.03,
                0.96,
                f"{len(archive)} earlier policy/checkpoint evaluations + {total} Phase 95 seeds\n"
                f"{all_finish_count} all-screen policies ({archive_finish_count} earlier; {finish_count} Phase 95)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.7,
                color=MUTED,
            )
        _soften_axes(ax)

    def mark_selected(ax: plt.Axes, spec: dict[str, object], *, color: str = TEXT) -> None:
        if not selected_mask.any():
            return
        selected = seeds[selected_mask].iloc[0]
        ax.scatter(
            [selected[str(spec["x_col"])]],
            [selected[str(spec["y_col"])]],
            marker="*",
            s=185,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=6,
            label="Selected policy",
        )

    def mark_historic(ax: plt.Axes, spec: dict[str, object]) -> None:
        x_ref = spec.get("x_ref")
        y_ref = spec.get("y_ref")
        if x_ref is None or y_ref is None:
            return
        ax.scatter(
            [float(x_ref)],
            [float(y_ref)],
            marker="D",
            s=62,
            color="white",
            edgecolor=TEXT,
            linewidth=1.2,
            zorder=6,
            label="Historic",
        )

    def plot_xy(
        ax: plt.Axes,
        frame: pd.DataFrame,
        spec: dict[str, object],
        *,
        color_values: pd.Series | np.ndarray | None = None,
        cmap: mpl.colors.Colormap | None = None,
        norm: mpl.colors.Normalize | None = None,
        color: str | None = None,
        alpha: float = 0.70,
        label: str | None = None,
        zorder: int = 3,
        rasterized: bool = True,
    ) -> mpl.collections.PathCollection:
        x_col = str(spec["x_col"])
        y_col = str(spec["y_col"])
        plot_frame = frame.dropna(subset=[x_col, y_col]).copy()
        x_values = pd.to_numeric(plot_frame[x_col], errors="coerce").to_numpy(dtype=float)
        if bool(spec.get("jitter_x")):
            offsets = ((np.arange(len(plot_frame)) % 17) - 8) * 0.012
            x_values = x_values + offsets
        y_values = pd.to_numeric(plot_frame[y_col], errors="coerce").to_numpy(dtype=float)
        scatter_kwargs: dict[str, object] = {
            "s": 13,
            "edgecolor": "none",
            "linewidth": 0.0,
            "alpha": alpha,
            "zorder": zorder,
            "rasterized": rasterized,
            "label": label,
        }
        if color_values is None:
            scatter_kwargs["color"] = color
        else:
            scatter_kwargs["c"] = (
                pd.to_numeric(plot_frame[color_values.name], errors="coerce")
                if isinstance(color_values, pd.Series)
                else color_values
            )
            scatter_kwargs["cmap"] = cmap
            scatter_kwargs["norm"] = norm
        return ax.scatter(x_values, y_values, **scatter_kwargs)

    def make_figure(
        color_mode: str,
        stem: str,
        *,
        axis_key: str = "storage_hydro",
        cmap_name: str | None = None,
        hist_mode: str = "bars",
    ) -> list[Path]:
        if only_stems is not None and stem not in only_stems:
            return []
        figsize = (7.55, 3.85) if hist_mode == "stacked_phase_histograms" else (8.3, 3.85)
        width_ratios = [1.32, 0.72] if hist_mode == "stacked_phase_histograms" else [1.28, 0.88]
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": width_ratios},
            constrained_layout=True,
        )
        ax, ax_hist = axes
        spec = axis_specs[axis_key]

        if color_mode == "phase95_focus":
            if not archive.empty:
                plot_xy(
                    ax,
                    archive,
                    spec,
                    color="#9CA3AF",
                    alpha=0.16,
                    zorder=1,
                    label="Earlier experiment seed",
                )

            cmap = mpl.colormaps["viridis"]
            norm = mpl.colors.Normalize(vmin=0, vmax=len(pass_cols))
            scatter = plot_xy(
                ax,
                seeds,
                spec,
                color_values=seeds["pass_count_clean"],
                cmap=cmap,
                norm=norm,
                alpha=0.86,
                zorder=3,
            )
            cbar_label = "Objectives passing (of 10)"
            cbar_ticks = np.arange(0, len(pass_cols) + 1, 2)
        elif color_mode == "phase":
            cmap = mpl.colormaps[cmap_name or "viridis"]
            phase_min = float(combined["phase"].min())
            phase_max = float(combined["phase"].max())
            norm = mpl.colors.Normalize(vmin=phase_min, vmax=phase_max)
            scatter = plot_xy(
                ax,
                combined,
                spec,
                color_values=combined["phase"],
                cmap=cmap,
                norm=norm,
                alpha=0.68,
                zorder=3,
            )
            cbar_label = "Phase"
            cbar_ticks = None
        elif color_mode == "pass_count_all":
            cmap = mpl.colormaps["viridis"]
            norm = mpl.colors.Normalize(vmin=0, vmax=len(pass_cols))
            scatter = plot_xy(
                ax,
                combined,
                spec,
                color_values=combined["pass_count_clean"],
                cmap=cmap,
                norm=norm,
                alpha=0.64,
                zorder=3,
            )
            cbar_label = "Objectives passing (of 10)"
            cbar_ticks = np.arange(0, len(pass_cols) + 1, 2)
        elif color_mode == "phase_medians":
            cmap = mpl.colormaps[cmap_name or "viridis"]
            phase_min = float(combined["phase"].min())
            phase_max = float(combined["phase"].max())
            norm = mpl.colors.Normalize(vmin=phase_min, vmax=phase_max)
            plot_xy(
                ax,
                combined,
                spec,
                color="#9CA3AF",
                alpha=0.14,
                zorder=1,
                label="Experiment seed",
            )
            x_col = str(spec["x_col"])
            y_col = str(spec["y_col"])
            medians = (
                combined.dropna(subset=["phase", x_col, y_col])
                .groupby("phase", as_index=False)
                .agg(
                    x=(x_col, "median"),
                    y=(y_col, "median"),
                    n=(x_col, "size"),
                )
                .sort_values("phase")
            )
            ax.plot(
                medians["x"],
                medians["y"],
                color=TEXT,
                linewidth=1.25,
                alpha=0.58,
                zorder=3,
            )
            scatter = ax.scatter(
                medians["x"],
                medians["y"],
                c=medians["phase"],
                cmap=cmap,
                norm=norm,
                s=np.clip(np.sqrt(medians["n"].to_numpy(dtype=float)) * 9.0, 34.0, 90.0),
                edgecolor="white",
                linewidth=0.65,
                alpha=0.96,
                zorder=4,
                label="Phase median",
            )
            for _, row in medians.iterrows():
                ax.annotate(
                    f"{int(row['phase'])}",
                    (float(row["x"]), float(row["y"])),
                    xytext=(3.5, 2.5),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=MUTED,
                    zorder=5,
                )
            cbar_label = "Phase"
            cbar_ticks = None
        elif color_mode == "pass_count_phase_medians":
            pass_cmap = mpl.colors.ListedColormap(
                ["#E2E8F0", "#CBD5E1", "#94A3B8", "#64748B", "#334155"]
            )
            pass_norm = mpl.colors.BoundaryNorm(np.arange(4.5, 10.5, 1.0), pass_cmap.N)
            grouped = combined.copy()
            grouped["pass_count_color"] = grouped["pass_count_clean"].clip(lower=5, upper=9)
            grouped_background = grouped[~grouped["finish_line_clean"].astype(bool)]
            scatter = plot_xy(
                ax,
                grouped_background,
                spec,
                color_values=grouped_background["pass_count_color"],
                cmap=pass_cmap,
                norm=pass_norm,
                alpha=0.58,
                zorder=2,
            )
            x_col = str(spec["x_col"])
            y_col = str(spec["y_col"])
            finishers = combined[
                combined["finish_line_clean"].astype(bool) & ~combined["is_selected_policy"].astype(bool)
            ].dropna(subset=[x_col, y_col])
            if not finishers.empty:
                finishers = finishers.copy()
                finishers["_x_plot"] = pd.to_numeric(finishers[x_col], errors="coerce").astype(float)
                finishers["_y_plot"] = pd.to_numeric(finishers[y_col], errors="coerce").astype(float)
                finishers["_overlap_key"] = list(
                    zip(
                        np.round(finishers["_x_plot"].to_numpy(dtype=float), 3),
                        np.round(finishers["_y_plot"].to_numpy(dtype=float), 3),
                    )
                )
                for _, duplicate in finishers.groupby("_overlap_key", sort=False):
                    if len(duplicate) <= 1:
                        continue
                    offsets = np.linspace(-0.008, 0.008, len(duplicate))
                    finishers.loc[duplicate.index, "_x_plot"] = finishers.loc[duplicate.index, "_x_plot"] + offsets
                ax.scatter(
                    finishers["_x_plot"],
                    finishers["_y_plot"],
                    marker="*",
                    s=120,
                    color=TEXT,
                    edgecolor="white",
                    linewidth=0.70,
                    alpha=0.98,
                    zorder=7,
                    label="10/10 objectives met",
                )
            medians = (
                combined.dropna(subset=["phase", x_col, y_col])
                .groupby("phase", as_index=False)
                .agg(
                    x=(x_col, "median"),
                    y=(y_col, "median"),
                    n=(x_col, "size"),
                )
                .sort_values("phase")
            )
            ax.plot(
                medians["x"],
                medians["y"],
                color=TEXT,
                linewidth=1.35,
                alpha=0.68,
                zorder=4,
            )
            median_sizes = np.clip(np.sqrt(medians["n"].to_numpy(dtype=float)) * 8.5, 48.0, 98.0)
            for idx, ((_, row), marker_size) in enumerate(zip(medians.iterrows(), median_sizes)):
                if not (float(spec.get("xlim", (-np.inf, np.inf))[0]) <= float(row["x"]) <= float(spec.get("xlim", (-np.inf, np.inf))[1])):
                    continue
                if not (float(spec.get("ylim", (-np.inf, np.inf))[0]) <= float(row["y"]) <= float(spec.get("ylim", (-np.inf, np.inf))[1])):
                    continue
                z = 6.0 + idx * 0.025
                ax.scatter(
                    [float(row["x"])],
                    [float(row["y"])],
                    color=TEXT,
                    s=float(marker_size),
                    edgecolor="white",
                    linewidth=0.85,
                    alpha=0.98,
                    zorder=z,
                    label="Phase median" if idx == 0 else None,
                )
                ax.annotate(
                    f"{int(row['phase'])}",
                    (float(row["x"]), float(row["y"])),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    fontweight="bold",
                    color="white",
                    zorder=z + 0.01,
                )
            cbar_label = "Objectives passing (of 10)"
            cbar_ticks = np.arange(5, len(pass_cols), 1)
        else:
            raise ValueError(f"Unknown color mode: {color_mode}")

        draw_base_axes(ax, spec, show_count_text=color_mode != "pass_count_phase_medians")
        if color_mode == "pass_count_phase_medians" and hist_mode == "stacked_phase_histograms":
            _add_panel_label(ax, "A", loc="right")
            _add_panel_label(ax_hist, "B")
            x_col = str(spec["x_col"])
            y_col = str(spec["y_col"])
            n_shown = len(combined.dropna(subset=[x_col, y_col]))
            ax.text(
                0.025,
                0.035,
                f"N={n_shown:,}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.8,
                color=MUTED,
            )
        mark_historic(ax, spec)
        mark_selected(ax, spec, color="#F59E0B" if color_mode == "pass_count_phase_medians" else TEXT)
        if color_mode == "pass_count_phase_medians":
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    linestyle="none",
                    markerfacecolor="white",
                    markeredgecolor=TEXT,
                    markeredgewidth=1.35,
                    markersize=7.7,
                    label="Historic",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=TEXT,
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    markersize=8.0,
                    label="Phase median",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    linestyle="none",
                    markerfacecolor=TEXT,
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    markersize=10.0,
                    label="10/10 objectives met",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    linestyle="none",
                    markerfacecolor="#F59E0B",
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    markersize=11.6,
                    label="Selected policy",
                ),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(0.045, 1.0),
                frameon=False,
                handlelength=1.15,
                handletextpad=0.72,
                labelspacing=0.34,
                borderaxespad=0.25,
            )
        else:
            ax.legend(loc="lower right", frameon=False)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.016)
        cbar.set_label(cbar_label, rotation=270, labelpad=5)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
            if color_mode == "pass_count_phase_medians":
                cbar.set_ticklabels(["<=5", "6", "7", "8", "9"])
        else:
            cbar.locator = mpl.ticker.MaxNLocator(nbins=6, integer=True)
            cbar.update_ticks()

        draw_histogram(ax_hist, mode=hist_mode)
        return _save(fig, output_dir, stem, dpi=dpi)

    written: list[Path] = []
    written.extend(make_figure("phase95_focus", "experiment-search-phase95-focus"))
    written.extend(make_figure("phase", "experiment-search-by-phase-viridis", cmap_name="viridis"))
    written.extend(make_figure("phase", "experiment-search-by-phase-plasma", cmap_name="plasma"))
    written.extend(make_figure("phase", "experiment-search-by-phase-turbo", cmap_name="turbo"))
    written.extend(make_figure("pass_count_all", "experiment-search-by-pass-count-all"))
    written.extend(
        make_figure(
            "phase_medians",
            "experiment-search-phase-median-path",
            cmap_name="viridis",
        )
    )
    written.extend(
        make_figure(
            "phase",
            "experiment-search-conservation-beneficial-by-phase",
            axis_key="conservation_beneficial",
            cmap_name="viridis",
        )
    )
    written.extend(
        make_figure(
            "pass_count_phase_medians",
            "experiment-search-conservation-beneficial-median-path",
            axis_key="conservation_beneficial",
            cmap_name="turbo",
        )
    )
    written.extend(
        make_figure(
            "pass_count_phase_medians",
            "experiment-search-conservation-beneficial-median-path-phase-groups",
            axis_key="conservation_beneficial",
            cmap_name="turbo",
            hist_mode="phase_groups",
        )
    )
    written.extend(
        make_figure(
            "pass_count_phase_medians",
            "experiment-search",
            axis_key="conservation_beneficial",
            cmap_name="turbo",
            hist_mode="stacked_phase_histograms",
        )
    )
    written.extend(
        make_figure(
            "phase",
            "experiment-search-pass-min-objective-by-phase",
            axis_key="pass_min",
            cmap_name="viridis",
        )
    )
    written.extend(
        make_figure(
            "phase",
            "experiment-search-pass-shortfall-by-phase",
            axis_key="pass_shortfall",
            cmap_name="viridis",
        )
    )
    return written
