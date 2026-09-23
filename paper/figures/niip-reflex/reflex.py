"""Figure-specific helpers extracted without changing plot calculations."""

from __future__ import annotations

import argparse

import math

import os
import sys

import re

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
from figurestyle import OBJECTIVE_COLORS, add_panel_label

OUTPUT_DIR = Path(__file__).resolve().parent

DATA_DIR = OUTPUT_DIR / "data"

DEFAULT_PAIRED_REFLEX = DATA_DIR / "phase52-split-paired-reflex-reduction.csv"

DEFAULT_NIIP_REFLEX_ROLLOUT = DATA_DIR / "phase52-highreq10-niipband-tight-003-seed-002-eval-rollout.csv"

SPR_PULSE_THRESHOLD_CFS = 1000.0

BLUE = "#2B6CB0"

BLUE_LIGHT = "#BBD7F0"

GRAY = "#8B95A5"

GREEN = OBJECTIVE_COLORS["niip"]

PURPLE = OBJECTIVE_COLORS["spr"]

RED = "#C44536"

TEXT = "#111827"

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

def _load_reflex_rollout(path: Path) -> pd.DataFrame:
    columns = [
        "release_sj_main_cfs",
        "release_niip_cfs",
        "niip_demand_cfs",
        "requested_release_sj_main_cfs",
        "release_cfs",
        "spill_cfs",
    ]
    df = pd.read_csv(path, parse_dates=["date"])
    df["doy"] = df["date"].dt.dayofyear
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

def _doy_summary(df: pd.DataFrame, columns: list[str], *, doy_min: int = 50, doy_max: int = 300) -> pd.DataFrame:
    subset = df[df["doy"].between(doy_min, doy_max)].copy()
    grouped = subset.groupby("doy", sort=True)
    out = pd.DataFrame({"doy": sorted(subset["doy"].dropna().unique())})
    for column in columns:
        if column not in subset.columns:
            continue
        stats = grouped[column].agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        out = out.merge(stats.add_prefix(f"{column}_").reset_index(), on="doy", how="left")
    return out

def _smooth(values: pd.Series, window: int = 7) -> np.ndarray:
    return values.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)

def _dominant_span(curve: pd.DataFrame, column: str, *, threshold: float) -> tuple[int, int] | None:
    mask = curve[column].fillna(0.0) > threshold
    spans: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for doy, active in zip(curve["doy"].astype(int), mask):
        if bool(active):
            if start is None:
                start = int(doy)
            previous = int(doy)
        elif start is not None and previous is not None:
            spans.append((start, previous))
            start = None
            previous = None
    if start is not None and previous is not None:
        spans.append((start, previous))
    if not spans:
        return None
    return max(spans, key=lambda span: span[1] - span[0])

def _niip_bias_delta(df: pd.DataFrame) -> tuple[float, float, float]:
    spr_request = df["requested_release_sj_main_cfs"].fillna(0.0) > 1.0
    active_niip = df["niip_demand_cfs"].fillna(0.0) >= 100.0
    bias = df["release_niip_cfs"] - df["niip_demand_cfs"]
    spr_bias = float(bias[spr_request & active_niip].mean())
    non_spr_bias = float(bias[(~spr_request) & active_niip].mean())
    return spr_bias, non_spr_bias, spr_bias - non_spr_bias

def _shade_spr_span(ax: plt.Axes, span: tuple[int, int] | None, *, label: str | None = "SPR pulse window") -> None:
    if span is None:
        return
    ax.axvspan(span[0], span[1], color=PURPLE, alpha=0.08, linewidth=0, label=label)

def _plot_niip_zoom_panel(
    ax: plt.Axes,
    curve: pd.DataFrame,
    span: tuple[int, int] | None,
    *,
    bias_delta: tuple[float, float, float],
) -> plt.Axes:
    x = curve["doy"].to_numpy(dtype=float)
    demand = _smooth(curve["niip_demand_cfs_median"])
    niip = _smooth(curve["release_niip_cfs_median"])
    niip_q25 = _smooth(curve["release_niip_cfs_q25"])
    niip_q75 = _smooth(curve["release_niip_cfs_q75"])
    sj = _smooth(curve["release_sj_main_cfs_median"])
    under = niip < demand

    _shade_spr_span(ax, span, label=None)
    ax.plot(x, demand, color=GRAY, linewidth=1.55, linestyle=(0, (2.0, 1.8)), label="NIIP demand")
    ax.fill_between(x, niip_q25, niip_q75, color=GREEN, alpha=0.15, linewidth=0)
    ax.plot(x, niip, color=GREEN, linewidth=1.9, label="Median NIIP release")
    ax.fill_between(x, niip, demand, where=under, color=RED, alpha=0.18, interpolate=True, label="NIIP below demand")

    ax2 = ax.twinx()
    ax2.plot(x, sj, color=BLUE, linewidth=1.35, alpha=0.42, label="Median SJ release")
    ax2.set_ylim(0, max(4700.0, float(np.nanmax(sj)) * 1.08))
    ax2.set_ylabel("SJ release (cfs)", color=BLUE, rotation=270, labelpad=18)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.spines["right"].set_color(BLUE_LIGHT)
    ax2.spines["top"].set_visible(False)

    ax.set_xlim(50, 300)
    ax.set_ylim(0, max(860.0, float(np.nanmax([demand, niip_q75])) * 1.08))
    ax.set_xlabel("Day of year")
    ax.set_ylabel("NIIP release or demand (cfs)")
    _soften_axes(ax)
    return ax2

def _add_plain_panel_label(ax: plt.Axes, label: str) -> None:
    add_panel_label(ax, label, x=0.012, y=0.985, zorder=10)

def _plot_split_head_reflex_panel(ax: plt.Axes, paired: pd.DataFrame) -> None:
    paired = paired.dropna(subset=["shared_reflex_abs_cfs", "split_reflex_abs_cfs"]).copy()
    paired["comparison_label"] = paired["comparison"].map(
        {
            "base -> split": "Base reward",
            "niipband_mild -> split_niipband_mild": "Banded NIIP reward",
        }
    ).fillna(paired["comparison"])

    colors = {
        "Base reward": GRAY,
        "Banded NIIP reward": GREEN,
    }

    rng = np.random.default_rng(42)
    legend_used: set[str] = set()
    for _, row in paired.iterrows():
        label = str(row["comparison_label"])
        color = colors.get(label, BLUE)
        jitter = float(rng.uniform(-0.035, 0.035))
        xs = np.array([0.0 + jitter, 1.0 + jitter])
        ys = np.array([float(row["shared_reflex_abs_cfs"]), float(row["split_reflex_abs_cfs"])])
        ax.plot(xs, ys, color=color, alpha=0.30, linewidth=1.1, zorder=2)
        ax.scatter(
            xs,
            ys,
            s=28,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.86,
            label=label if label not in legend_used else None,
            zorder=3,
        )
        legend_used.add(label)

    med_shared = float(paired["shared_reflex_abs_cfs"].median())
    med_split = float(paired["split_reflex_abs_cfs"].median())
    ax.hlines(med_shared, -0.18, 0.18, color=TEXT, linewidth=2.4, zorder=4, label="Median")
    ax.hlines(med_split, 0.82, 1.18, color=TEXT, linewidth=2.4, zorder=4)
    ax.plot([0, 1], [med_shared, med_split], color=TEXT, linewidth=1.15, linestyle=(0, (3.0, 2.2)), alpha=0.55)

    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(0, max(640.0, float(paired[["shared_reflex_abs_cfs", "split_reflex_abs_cfs"]].to_numpy().max()) + 55.0))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Shared\nactor heads", "Split\nactor heads"])
    ax.set_ylabel("NIIP reflex magnitude (cfs)")
    ax.legend(loc="upper right", frameon=False)
    _soften_axes(ax)

def build_niip_reflex_design_lesson(
    *,
    rollout_path: Path,
    paired_reflex_path: Path,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    rollout = _load_reflex_rollout(rollout_path)
    curve = _doy_summary(
        rollout,
        [
            "release_sj_main_cfs",
            "release_niip_cfs",
            "niip_demand_cfs",
            "requested_release_sj_main_cfs",
            "release_cfs",
        ],
    )
    span = _dominant_span(curve, "requested_release_sj_main_cfs_median", threshold=SPR_PULSE_THRESHOLD_CFS)
    paired = pd.read_csv(paired_reflex_path)

    fig = plt.figure(figsize=(10.4, 4.15), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.38, 1.0])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    twin = _plot_niip_zoom_panel(ax_left, curve, span, bias_delta=_niip_bias_delta(rollout))
    handles, labels = ax_left.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    handles.extend(twin_handles)
    labels.extend(twin_labels)
    keep: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        keep.setdefault(label, handle)
    ax_left.legend(keep.values(), keep.keys(), loc="upper right", frameon=False)

    _plot_split_head_reflex_panel(ax_right, paired)

    _add_plain_panel_label(ax_left, "A")
    _add_plain_panel_label(ax_right, "B")
    return _save(fig, output_dir, "niip_reflex", dpi=dpi)
