"""Rebuild Appendix C evidence figures from bundled daily records.

The tributary and Farmington-infill plots are recomputed from repository data.
Travel-time estimates are transcribed from the historical analysis; its slide
deck may be supplied for provenance but is not required to build the figures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[3]
END = pd.Timestamp("2024-08-17")
STUDY_END = pd.Timestamp("2025-11-07")

SITE_FILES = {
    "San Juan near Bluff": "daily_sj_bluff.csv",
    "Animas at Farmington": "daily_animas_farmington.csv",
    "Chaco at Waterflow": "daily_chaco_waterflow.csv",
    "Mancos at Towaoc": "daily_mancos_towaoc.csv",
    "Chinle at Mexican Water": "daily_chinle_mexicanwater.csv",
    "La Plata at Farmington": "daily_laplata_farmington.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_record(path: Path, repo_root: Path) -> dict[str, str]:
    """Return a stable repository-relative provenance record."""
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    return {"path": relative, "sha256": sha256(path)}


def optional_record(path: Path | None, repo_root: Path) -> dict[str, str | bool]:
    """Describe an optional provenance file without exposing local paths."""
    if path is None:
        return {"provided": False}
    path = path.resolve()
    record: dict[str, str | bool] = {"provided": True, "available": path.is_file()}
    if path.is_file():
        try:
            record["path"] = path.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            record["filename"] = path.name
        record["sha256"] = sha256(path)
    else:
        record["filename"] = path.name
    return record


def load_daily(path: Path, *, clip_negative: bool = False) -> pd.Series:
    frame = pd.read_csv(path, usecols=["time", "value"])
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if clip_negative:
        frame["value"] = frame["value"].clip(lower=0)
    else:
        frame.loc[frame["value"] < 0, "value"] = np.nan
    if frame["time"].duplicated().any():
        raise ValueError(f"Duplicate daily dates: {path}")
    return frame.set_index("time")["value"].sort_index().rename(path.stem)


def set_style(repo_root: Path) -> object:
    style_path = repo_root / "paper/figure-support/figurestyle.py"
    spec = importlib.util.spec_from_file_location("navajo_figurestyle", style_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load figure style from {style_path}")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.register_fonts()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans"],
            "font.size": 9.1,
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.4,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    return style


def decorate(ax: plt.Axes, style: object, *, xgrid: bool = False) -> None:
    ax.set_facecolor(style.BACKGROUND)
    ax.grid(axis="x" if xgrid else "y", color=style.GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(style.SPINE)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.12,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(HERE / f"{stem}.png", dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_travel_time(style: object) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.1), layout="constrained")
    ax.set_xlim(-0.5, 4.3)
    ax.set_ylim(-0.28, 1.70)
    ax.axis("off")
    nodes = [(0, "Navajo\nDam"), (1.25, "Farmington"), (2.45, "Four\nCorners"), (4.0, "Bluff")]
    for x, label in nodes:
        ax.plot(x, 0.56, "o", ms=8, color=style.OBJECTIVE_COLORS["flood"], zorder=3)
        ax.text(x, 0.35, label, ha="center", va="top", color=style.TEXT)
    for x0, x1, hours in [(0, 1.25, "~10 h"), (1.25, 2.45, "19.9 h"), (2.45, 4.0, "")]:
        ax.annotate("", xy=(x1 - 0.09, 0.56), xytext=(x0 + 0.09, 0.56),
                    arrowprops={"arrowstyle": "->", "lw": 1.35, "color": style.MUTED if hasattr(style, "MUTED") else style.SPINE})
        if hours:
            ax.text((x0 + x1) / 2, 0.69, hours, ha="center", va="bottom", color=style.TEXT)
    ax.annotate("", xy=(3.96, 1.04), xytext=(1.27, 1.04),
                arrowprops={"arrowstyle": "->", "lw": 1.15, "color": style.SPINE})
    ax.text(2.61, 1.09, "38 h Farmington to Bluff", ha="center", va="bottom",
            color=style.TEXT, fontsize=8.8)
    ax.annotate("", xy=(3.96, 1.48), xytext=(0.04, 1.48),
                arrowprops={"arrowstyle": "->", "lw": 1.45,
                            "color": style.OBJECTIVE_COLORS["flood"]})
    ax.text(2.0, 1.53, "~48 h Navajo Dam to Bluff", ha="center", va="bottom",
            color=style.OBJECTIVE_COLORS["flood"], fontsize=9.4)
    ax.text(2.0, -0.15, "Daily flood proxy: two-step delay", ha="center",
            fontsize=8.5, color=style.TEXT)
    save(fig, "travel-time-evidence")


def annual_summary(series: pd.Series) -> pd.DataFrame:
    series = series.loc["1970-01-01":"2024-12-31"].dropna()
    years = series.groupby(series.index.year).agg(["mean", "max", "count"])
    return years.loc[years["count"] >= 330].copy()


def plot_tributaries(annual: dict[str, pd.DataFrame], style: object) -> None:
    labels = list(SITE_FILES)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.1), sharex=True,
                             sharey=True, layout="constrained")
    colors = [style.TEXT if i == 0 else
              style.OBJECTIVE_COLORS["esa"] if i == 1 else style.SPINE
              for i in range(len(labels))]
    for ax, field, title, panel in zip(
        axes,
        ["mean", "max"],
        ["Annual mean", "Annual daily peak"],
        ["A", "B"],
    ):
        decorate(ax, style, xgrid=True)
        for row, (label, color) in enumerate(zip(labels, colors)):
            values = annual[label][field].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            y = len(labels) - 1 - row
            lo, med, hi = np.quantile(values, [0.25, 0.5, 0.75])
            ax.plot([lo, hi], [y, y], color=color, lw=5, alpha=0.46, solid_capstyle="round")
            ax.scatter(values, np.full(len(values), y), s=9, alpha=0.32, color=color, linewidths=0)
            ax.scatter([med], [y], s=31, color=color, edgecolor="white", linewidth=0.6, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 40000)
        if panel == "B":
            ax.set_xlabel("Discharge (cfs, log scale)")
        ax.set_title(title, loc="left", pad=8)
        ax.set_yticks(range(len(labels)), labels[::-1])
        style.add_panel_label(ax, panel, fontsize=11)
    save(fig, "tributary-context")


def farmington_summary(arch: pd.Series, animas: pd.Series, observed: pd.Series,
                       patched: pd.Series, patch_log: pd.DataFrame) -> tuple[dict, pd.Series, pd.DataFrame]:
    dates = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
    raw_sum = arch.reindex(dates) + animas.reindex(dates)
    source = observed.reindex(dates)
    paired = pd.concat([raw_sum.rename("proxy"), source.rename("observed")], axis=1).dropna()
    bias = float((paired["proxy"] - paired["observed"]).mean())
    adjusted = (raw_sum - bias).clip(lower=0)
    paired["adjusted"] = adjusted.reindex(paired.index)
    predicted_patch = adjusted.reindex(patch_log.index)
    actual_patch = patched.reindex(patch_log.index)
    if not np.allclose(predicted_patch, actual_patch, rtol=0, atol=1e-7):
        difference = (predicted_patch - actual_patch).abs()
        raise ValueError(
            f"Regenerated patch differs from archived daily record: bias={bias}, "
            f"max difference={difference.max()}, first={difference.sort_values(ascending=False).head()}"
        )
    residual = adjusted.reindex(paired.index) - paired["observed"]
    r2 = 1.0 - float((residual ** 2).sum() / ((paired["observed"] - paired["observed"].mean()) ** 2).sum())
    pre = paired.loc[:"2020-08-19"]
    train = paired.loc[:"2013-12-31"]
    summary = {
        "n_overlap_days": int(len(paired)),
        "bias_cfs": bias,
        "pre_gap_bias_cfs": float((pre["proxy"] - pre["observed"]).mean()),
        "training_period_bias_cfs": float((train["proxy"] - train["observed"]).mean()),
        "adjusted_proxy_r2": r2,
        "adjusted_proxy_mae_cfs": float(residual.abs().mean()),
        "n_patched_days": int(len(patch_log)),
        "n_patched_evaluation_days": int(((patch_log.index >= "2014-01-01") & (patch_log.index <= END)).sum()),
    }
    return summary, adjusted, paired


def plot_farmington(observed: pd.Series, patched: pd.Series, adjusted: pd.Series,
                    paired: pd.DataFrame, patch_log: pd.DataFrame, summary: dict,
                    style: object) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), layout="constrained")
    ax = axes[0]
    decorate(ax, style)
    points = np.log10(paired[["adjusted", "observed"]].clip(lower=0) + 1.0)
    ax.hexbin(points["adjusted"], points["observed"], gridsize=54,
                     mincnt=1, bins="log", cmap="Blues", linewidths=0)
    lim = np.log10(30001)
    ax.plot([0, lim], [0, lim], ls="--", color=style.TEXT, lw=0.9)
    ticks = [0, 100, 1000, 10000]
    ax.set_xticks(np.log10(np.array(ticks) + 1), [f"{v:,}" for v in ticks])
    ax.set_yticks(np.log10(np.array(ticks) + 1), [f"{v:,}" for v in ticks])
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Bias-adjusted Archuleta + Animas (cfs)")
    ax.set_ylabel("Observed San Juan at Farmington (cfs)")
    ax.text(0.04, 0.94, f"{summary['n_overlap_days']:,} observed days\nAdjusted MAE: {summary['adjusted_proxy_mae_cfs']:.0f} cfs",
            transform=ax.transAxes, va="top", ha="left", color=style.TEXT,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": style.GRID})
    style.add_panel_label(ax, "A", x=0.0, y=1.04, va="bottom", fontsize=11)

    ax = axes[1]
    decorate(ax, style)
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    actual = observed.reindex(dates)
    reconstructed = patched.reindex(dates)
    missing = actual.isna()
    actual_smooth = actual.rolling(14, min_periods=1, center=True).median().where(~missing)
    reconstructed_smooth = reconstructed.where(missing).rolling(14, min_periods=1, center=True).median()
    ax.axvspan(pd.Timestamp("2020-08-20"), pd.Timestamp("2022-11-08"),
               color=style.OBJECTIVE_COLORS["flood"], alpha=0.075, lw=0)
    ax.plot(dates, reconstructed_smooth, color=style.OBJECTIVE_COLORS["flood"],
            lw=1.2, label="Infilled", zorder=3)
    ax.plot(dates, actual_smooth, color=style.TEXT, lw=1.1, label="Observed")
    ax.set_xlim(dates.min(), dates.max())
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("San Juan at Farmington (cfs)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=style.GRID)
    style.add_panel_label(ax, "B", x=0.0, y=1.04, va="bottom", fontsize=11)
    save(fig, "farmington-infill")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--slide-deck",
        type=Path,
        help="Optional historical analysis deck to hash for provenance",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    daily_root = repo_root / "data/daily_flows"
    patch_root = repo_root / "data/patch_sanjuan_at_farmington"
    style = set_style(repo_root)

    inputs = {name: daily_root / filename for name, filename in SITE_FILES.items()}
    inputs.update(
        {
            "archuleta": daily_root / "daily_sj_archuleta.csv",
            "farmington_original": patch_root / "daily_sj_farmington_original_backup.csv",
            "farmington_patched": daily_root / "daily_sj_farmington.csv",
            "patched_rows": patch_root / "patched_rows.csv",
            "figure_style": repo_root / "paper/figure-support/figurestyle.py",
        }
    )
    if any(not path.exists() for path in inputs.values()):
        raise FileNotFoundError([str(path) for path in inputs.values() if not path.exists()])

    arch = load_daily(inputs["archuleta"])
    animas = load_daily(inputs["Animas at Farmington"])
    animas_patch = load_daily(inputs["Animas at Farmington"], clip_negative=True)
    bluff = load_daily(inputs["San Juan near Bluff"])
    farm_original = load_daily(inputs["farmington_original"])
    farm_patched = load_daily(inputs["farmington_patched"])
    patch_log = pd.read_csv(inputs["patched_rows"], parse_dates=["time"]).set_index("time")

    plot_travel_time(style)

    annual = {name: annual_summary(load_daily(path)) for name, path in inputs.items()
              if name in SITE_FILES}
    plot_tributaries(annual, style)
    annual_info = {
        name: {
            "n_years_at_least_330_days": int(len(frame)),
            "first_year": int(frame.index.min()),
            "last_year": int(frame.index.max()),
            "median_annual_mean_cfs": float(frame["mean"].median()),
            "median_annual_daily_peak_cfs": float(frame["max"].median()),
            "maximum_recorded_daily_peak_cfs": float(frame["max"].max()),
        }
        for name, frame in annual.items()
    }

    farm_summary, adjusted, paired = farmington_summary(
        arch, animas_patch, farm_original, farm_patched, patch_log
    )
    plot_farmington(farm_original, farm_patched, adjusted, paired, patch_log,
                    farm_summary, style)

    common = pd.concat([arch.rename("arch"), animas.rename("animas"),
                        bluff.rename("bluff")], axis=1).loc[:STUDY_END].dropna()
    common = common.loc[common["bluff"] > 0]
    share = 100.0 * float((common["arch"] + common["animas"]).sum() / common["bluff"].sum())
    result = {
        "methods": {
            "event_lags_source": "Slide deck slides 8--10; original event-selection code unavailable",
            "flood_lag_basis": "Slide 10 reports approximately 48 h Navajo Reservoir to Bluff; retain two daily steps for the flood proxy",
            "annual_window": "1970--2024, site-specific years with at least 330 valid daily records",
            "farmington_patch": "max(0, Archuleta + Animas - full-overlap mean bias)",
        },
        "slide_travel_times_hours": {
            "archuleta_to_farmington": 8.6,
            "navajo_to_farmington": 10.0,
            "farmington_to_four_corners": 19.9,
            "farmington_to_bluff": 38.0,
            "navajo_to_bluff": 48.0,
        },
        "selected_daily_bluff_lag_days": 2,
        "annual_site_summary": annual_info,
        "farmington_gap_fill": farm_summary,
        "headwater_to_bluff_volume_comparison": {
            "common_days": int(len(common)),
            "first_day": str(common.index.min().date()),
            "last_day": str(common.index.max().date()),
            "archuleta_plus_animas_over_bluff_pct": share,
        },
        "inputs": {
            name: input_record(path, repo_root) for name, path in inputs.items()
        },
        "optional_provenance": {
            "historical_analysis_deck": optional_record(args.slide_deck, repo_root),
        },
    }
    (HERE / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
