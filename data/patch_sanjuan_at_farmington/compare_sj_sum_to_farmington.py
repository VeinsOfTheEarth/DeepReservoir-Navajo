from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = Path(__file__).resolve().parent


def _read_daily_csv(path: Path, date_col: str = "time") -> pd.DataFrame:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    value_cols = [c for c in df.columns if c != date_col]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].clip(lower=0)
    return df.set_index(date_col).sort_index()


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    obs = df["sj_farmington_q_cfs"].astype(float)
    proxy = df["proxy_sum_q_cfs"].astype(float)
    diff = proxy - obs
    abs_pct = (diff.abs() / obs.replace(0, np.nan)).dropna()
    sse = float((diff**2).sum())
    sst = float(((obs - obs.mean()) ** 2).sum())
    r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")

    return {
        "n_days": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "corr": float(obs.corr(proxy)),
        "r2": r2,
        "mean_obs_cfs": float(obs.mean()),
        "mean_proxy_cfs": float(proxy.mean()),
        "mean_bias_cfs": float(diff.mean()),
        "median_bias_cfs": float(diff.median()),
        "mae_cfs": float(diff.abs().mean()),
        "rmse_cfs": float(np.sqrt((diff**2).mean())),
        "mape_pct": float(abs_pct.mean() * 100.0),
    }


def _annual_volume_metrics(df_annual: pd.DataFrame) -> dict[str, float]:
    obs = df_annual["sj_farmington_af"].astype(float)
    proxy = df_annual["proxy_sum_af"].astype(float)
    diff = proxy - obs
    sse = float((diff**2).sum())
    sst = float(((obs - obs.mean()) ** 2).sum())
    r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")
    return {
        "n_years": int(len(df_annual)),
        "start_year": int(df_annual.index.min()),
        "end_year": int(df_annual.index.max()),
        "corr": float(obs.corr(proxy)),
        "r2": r2,
        "mean_obs_af": float(obs.mean()),
        "mean_proxy_af": float(proxy.mean()),
        "mean_bias_af": float(diff.mean()),
        "median_bias_af": float(diff.median()),
        "mae_af": float(diff.abs().mean()),
        "rmse_af": float(np.sqrt((diff**2).mean())),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    sj_archuleta = _read_daily_csv(REPO_ROOT / "data/daily_flows/daily_sj_archuleta.csv")
    animas = _read_daily_csv(REPO_ROOT / "data/daily_flows/daily_animas_farmington.csv")
    sj_farmington = _read_daily_csv(REPO_ROOT / "data/daily_flows/daily_sj_farmington.csv")

    df = pd.DataFrame(index=sj_archuleta.index.union(animas.index).union(sj_farmington.index))
    df["sj_archuleta_q_cfs"] = sj_archuleta["value"].astype(float)
    df["animas_farmington_q_cfs"] = animas["value"].astype(float)
    df["sj_farmington_q_cfs"] = sj_farmington["value"].astype(float)
    df["proxy_sum_q_cfs"] = df["sj_archuleta_q_cfs"] + df["animas_farmington_q_cfs"]
    df["sj_farmington_af"] = df["sj_farmington_q_cfs"] * 1.983471074
    df["proxy_sum_af"] = df["proxy_sum_q_cfs"] * 1.983471074

    overlap = df.dropna(subset=["proxy_sum_q_cfs", "sj_farmington_q_cfs"]).copy()
    gap_window = df.loc["2020-08-20":"2022-11-08"].copy()

    pre_gap = overlap.loc[: "2020-08-19"].copy()
    post_gap = overlap.loc["2022-11-09" :].copy()

    annual = overlap[["sj_farmington_af", "proxy_sum_af"]].copy()
    annual["year"] = annual.index.year
    annual["day_count"] = 1
    annual_summary = annual.groupby("year").agg(
        sj_farmington_af=("sj_farmington_af", "sum"),
        proxy_sum_af=("proxy_sum_af", "sum"),
        n_days=("day_count", "sum"),
    )
    full_year_mask = annual_summary.index.to_series().map(
        lambda year: 366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365
    )
    annual_complete = annual_summary.loc[annual_summary["n_days"] == full_year_mask].copy()
    annual_complete.to_csv(OUTDIR / "annual_volume_comparison.csv", index_label="year")

    summary = {
        "files": {
            "sj_archuleta": str(REPO_ROOT / "data/daily_flows/daily_sj_archuleta.csv"),
            "animas_farmington": str(REPO_ROOT / "data/daily_flows/daily_animas_farmington.csv"),
            "sj_farmington": str(REPO_ROOT / "data/daily_flows/daily_sj_farmington.csv"),
        },
        "coverage": {
            "sj_archuleta": {
                "start": str(sj_archuleta.index.min().date()),
                "end": str(sj_archuleta.index.max().date()),
                "n_days": int(len(sj_archuleta)),
            },
            "animas_farmington": {
                "start": str(animas.index.min().date()),
                "end": str(animas.index.max().date()),
                "n_days": int(len(animas)),
            },
            "sj_farmington": {
                "start": str(sj_farmington.index.min().date()),
                "end": str(sj_farmington.index.max().date()),
                "n_days": int(len(sj_farmington)),
            },
            "gap_window": {
                "start": "2020-08-20",
                "end": "2022-11-08",
                "n_days": int(len(gap_window)),
                "proxy_days_available": int(gap_window["proxy_sum_q_cfs"].notna().sum()),
                "farmington_days_available": int(gap_window["sj_farmington_q_cfs"].notna().sum()),
            },
        },
        "metrics": {
            "all_overlap": _metrics(overlap),
            "pre_gap_overlap": _metrics(pre_gap),
            "post_gap_overlap": _metrics(post_gap),
            "annual_complete_calendar_years": _annual_volume_metrics(annual_complete),
        },
    }

    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    preview = df.loc["2018-01-01":"2024-12-31", [
        "sj_farmington_q_cfs",
        "proxy_sum_q_cfs",
    ]].copy()
    preview.to_csv(OUTDIR / "comparison_2018_2024.csv", index_label="date")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False, constrained_layout=True)

    # Full overlap context
    ax = axes[0]
    full_plot = df.loc["2018-01-01":"2024-12-31"]
    ax.plot(full_plot.index, full_plot["sj_farmington_q_cfs"], label="SJ Farmington", linewidth=1.5)
    ax.plot(full_plot.index, full_plot["proxy_sum_q_cfs"], label="SJ Archuleta + Animas Farmington", linewidth=1.2, alpha=0.9)
    ax.axvspan(pd.Timestamp("2020-08-20"), pd.Timestamp("2022-11-08"), color="tab:red", alpha=0.12, label="Farmington gap")
    ax.set_title("SJ Farmington vs upstream proxy sum")
    ax.set_ylabel("Discharge (cfs)")
    ax.legend(loc="upper right")

    # Zoom near the gap
    ax = axes[1]
    zoom = df.loc["2019-01-01":"2023-12-31"]
    ax.plot(zoom.index, zoom["sj_farmington_q_cfs"], label="SJ Farmington", linewidth=1.6)
    ax.plot(zoom.index, zoom["proxy_sum_q_cfs"], label="SJ Archuleta + Animas Farmington", linewidth=1.2, alpha=0.9)
    ax.axvspan(pd.Timestamp("2020-08-20"), pd.Timestamp("2022-11-08"), color="tab:red", alpha=0.12)
    ax.set_title("Zoom around the 2020-08-20 to 2022-11-08 Farmington gap")
    ax.set_ylabel("Discharge (cfs)")
    ax.set_xlabel("Date")

    fig.savefig(OUTDIR / "sj_farmington_vs_proxy_sum.png", dpi=180)
    plt.close(fig)

    # Daily-value scatter plot
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.scatter(
        pre_gap["sj_farmington_q_cfs"],
        pre_gap["proxy_sum_q_cfs"],
        s=7,
        alpha=0.25,
        label="Pre-gap overlap",
        color="tab:blue",
        edgecolors="none",
    )
    ax.scatter(
        post_gap["sj_farmington_q_cfs"],
        post_gap["proxy_sum_q_cfs"],
        s=9,
        alpha=0.5,
        label="Post-gap overlap",
        color="tab:orange",
        edgecolors="none",
    )
    lim_max = float(
        max(
            overlap["sj_farmington_q_cfs"].max(),
            overlap["proxy_sum_q_cfs"].max(),
        )
    )
    ax.plot([0, lim_max], [0, lim_max], color="black", linestyle="--", linewidth=1.0, label="1:1 line")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("SJ Farmington (cfs)")
    ax.set_ylabel("SJ Archuleta + Animas Farmington (cfs)")
    ax.set_title("Daily-value scatter: Farmington vs proxy sum")
    overall = summary["metrics"]["all_overlap"]
    ax.text(
        0.03,
        0.97,
        f"R^2 = {overall['r2']:.4f}\nRMSE = {overall['rmse_cfs']:.1f} cfs\nN = {overall['n_days']:,}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    ax.legend(loc="lower right")
    fig.savefig(OUTDIR / "sj_farmington_vs_proxy_sum_scatter.png", dpi=180)
    plt.close(fig)

    # Annual total-volume scatter plot
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.scatter(
        annual_complete["sj_farmington_af"],
        annual_complete["proxy_sum_af"],
        s=28,
        alpha=0.65,
        color="tab:green",
        edgecolors="none",
        label="Complete calendar years",
    )
    annual_lim_max = float(
        max(
            annual_complete["sj_farmington_af"].max(),
            annual_complete["proxy_sum_af"].max(),
        )
    )
    ax.plot(
        [0, annual_lim_max],
        [0, annual_lim_max],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="1:1 line",
    )
    ax.set_xlim(0, annual_lim_max)
    ax.set_ylim(0, annual_lim_max)
    ax.set_xlabel("SJ Farmington annual volume (acre-ft)")
    ax.set_ylabel("Proxy annual volume (acre-ft)")
    ax.set_title("Calendar-year total volume: Farmington vs proxy sum")
    annual_metrics = summary["metrics"]["annual_complete_calendar_years"]
    ax.text(
        0.03,
        0.97,
        (
            f"R^2 = {annual_metrics['r2']:.4f}\n"
            f"RMSE = {annual_metrics['rmse_af']:.0f} acre-ft\n"
            f"N = {annual_metrics['n_years']:,} years"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    ax.legend(loc="lower right")
    fig.savefig(OUTDIR / "sj_farmington_vs_proxy_sum_annual_scatter.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote plot: {OUTDIR / 'sj_farmington_vs_proxy_sum.png'}")
    print(f"Wrote scatter plot: {OUTDIR / 'sj_farmington_vs_proxy_sum_scatter.png'}")
    print(f"Wrote annual scatter plot: {OUTDIR / 'sj_farmington_vs_proxy_sum_annual_scatter.png'}")
    print(f"Wrote summary: {OUTDIR / 'summary.json'}")
    print(f"Wrote preview CSV: {OUTDIR / 'comparison_2018_2024.csv'}")
    print(f"Wrote annual comparison CSV: {OUTDIR / 'annual_volume_comparison.csv'}")


if __name__ == "__main__":
    main()
