"""Shared styling/data helpers for training-and-historic paper figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT / "paper" / "figure-support"))
from figurestyle import OBJECTIVE_COLORS

from deepreservoir.drl import helpers, model


CFS_DAY_TO_AF = 86_400.0 / 43_560.0
SELECTED_TRAIN_START = pd.Timestamp("1967-06-07")
SELECTED_TRAIN_END = pd.Timestamp("2013-12-31")
SELECTED_EVAL_START = pd.Timestamp("2014-01-01")
SELECTED_EVAL_END = pd.Timestamp("2024-08-17")
COMMON_X_START = SELECTED_TRAIN_START
# Use the selected-policy storage/evaluation record as the common endpoint.
COMMON_X_END = SELECTED_EVAL_END

BLUE = "#2B6CB0"
BLUE_LIGHT = "#BBD7F0"
GREEN = OBJECTIVE_COLORS["niip"]
GREEN_LIGHT = "#C7E9C0"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FED7AA"
TEAL = "#0F766E"
PURPLE = OBJECTIVE_COLORS["spr"]
GRAY = "#6B7280"
GRAY_LIGHT = "#E5E7EB"
GRID = "#E2E8F0"
SPINE = "#94A3B8"
TEXT = "#111827"
BACKGROUND = "#F4F7FA"


def set_theme() -> None:
    """Apply a compact paper-figure style."""
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
            "legend.fontsize": 8.6,
            "lines.dash_capstyle": "round",
            "lines.solid_capstyle": "round",
            "lines.dash_joinstyle": "round",
            "savefig.bbox": "tight",
        }
    )


def soften_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.9)


def add_train_eval_shading(ax: plt.Axes, *, label: bool = False) -> None:
    ax.axvspan(SELECTED_TRAIN_START, SELECTED_TRAIN_END, color=BLUE_LIGHT, alpha=0.18, linewidth=0)
    ax.axvspan(SELECTED_EVAL_START, SELECTED_EVAL_END, color=ORANGE, alpha=0.12, linewidth=0)
    ax.axvline(SELECTED_EVAL_START, color=GRAY, lw=1.15, ls="--", alpha=0.88)
    if label:
        ax.text(
            SELECTED_TRAIN_START + (SELECTED_TRAIN_END - SELECTED_TRAIN_START) * 0.43,
            0.93,
            "Training period",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=TEXT,
            fontsize=8.8,
        )
        ax.text(
            SELECTED_EVAL_START + (SELECTED_EVAL_END - SELECTED_EVAL_START) * 0.54,
            0.93,
            "Evaluation period",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=TEXT,
            fontsize=8.8,
        )


def set_common_time_axis(ax: plt.Axes) -> None:
    import matplotlib.dates as mdates

    ax.set_xlim(COMMON_X_START, COMMON_X_END)
    ax.margins(x=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.tick_params(axis="x", which="minor", length=2.8, color=SPINE)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem.replace("_", "-")
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=320)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def load_raw_model_data() -> pd.DataFrame:
    return model.load_all_model_data()["raw"]


def train_eval_split(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return helpers.split_train_test_by_water_year(raw, n_years_test=10)


def selected_policy_train_eval_split(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the selected-policy train/evaluation windows used in the paper."""
    train = raw.loc[SELECTED_TRAIN_START:SELECTED_TRAIN_END]
    eval_df = raw.loc[SELECTED_EVAL_START:SELECTED_EVAL_END]
    return train, eval_df


def water_year(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
        index.year + (index.month >= 10).astype(int),
        index=index,
        name="water_year",
    )


def water_year_midpoint(wy: int) -> pd.Timestamp:
    start = pd.Timestamp(year=int(wy) - 1, month=10, day=1)
    return start + pd.Timedelta(days=182)


def complete_water_years(index: pd.DatetimeIndex) -> set[int]:
    """Match the complete-water-year test used by the hydrology transform."""
    if index.empty:
        return set()
    wy_by_day = water_year(index)
    out: set[int] = set()
    for wy in sorted({int(v) for v in wy_by_day}):
        start = pd.Timestamp(year=wy - 1, month=10, day=1)
        end = pd.Timestamp(year=wy, month=9, day=30)
        if index.min() <= start and index.max() >= end:
            mask = wy_by_day == wy
            if int(mask.sum()) >= 365:
                out.add(int(wy))
    return out


def annual_total_af(df: pd.DataFrame, column: str, *, cfs_to_af: bool) -> pd.Series:
    """Annual totals by complete water year, matching the training transform."""
    complete = complete_water_years(pd.DatetimeIndex(df.index))
    if not complete:
        return pd.Series(dtype=float, name=column)
    wy_by_day = water_year(pd.DatetimeIndex(df.index))
    values = pd.to_numeric(df[column], errors="coerce").astype(float)
    if cfs_to_af:
        values = values * CFS_DAY_TO_AF
    rows: dict[int, float] = {}
    for wy in sorted(complete):
        rows[int(wy)] = float(values.loc[wy_by_day == wy].sum())
    return pd.Series(rows, name=column)
