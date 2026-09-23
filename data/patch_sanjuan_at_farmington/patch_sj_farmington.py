from __future__ import annotations

"""Patch missing San Juan at Farmington daily flows with a bias-adjusted proxy sum.

The proxy is built as:
    San Juan at Archuleta + Animas at Farmington

We estimate a single additive daily bias over the overlap period so that the
proxy's total overlap volume matches the observed Farmington total volume, then
use that adjusted proxy to fill missing Farmington days.
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CFS_DAY_TO_AF = 1.983471074
PATCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = PATCH_DIR.parents[1]
DAILY_DIR = REPO_ROOT / "data" / "daily_flows"

PATH_SJ_ARCHULETA = DAILY_DIR / "daily_sj_archuleta.csv"
PATH_ANIMAS_FARMINGTON = DAILY_DIR / "daily_animas_farmington.csv"
PATH_SJ_FARMINGTON = DAILY_DIR / "daily_sj_farmington.csv"
PATH_SJ_FARMINGTON_BACKUP = PATCH_DIR / "daily_sj_farmington_original_backup.csv"

PATH_PATCHED_ROWS = PATCH_DIR / "patched_rows.csv"
PATH_PATCH_SUMMARY = PATCH_DIR / "patch_summary.json"
PATH_VALIDATION_PLOT = PATCH_DIR / "daily_sj_farmington_patched_2017_to_end.png"


def _read_daily_csv(path: Path, *, clip_negative: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if clip_negative:
        df["value"] = df["value"].clip(lower=0)
    return df.set_index("time").sort_index()


def _ensure_backup() -> Path:
    if not PATH_SJ_FARMINGTON_BACKUP.exists():
        shutil.copy2(PATH_SJ_FARMINGTON, PATH_SJ_FARMINGTON_BACKUP)
    return PATH_SJ_FARMINGTON_BACKUP


def _continuous_ranges(idx: pd.DatetimeIndex) -> list[dict[str, object]]:
    if len(idx) == 0:
        return []
    idx = pd.DatetimeIndex(sorted(idx.unique()))
    diffs = idx.to_series().diff().fillna(pd.Timedelta(days=1))
    groups = (diffs != pd.Timedelta(days=1)).cumsum()
    ranges: list[dict[str, object]] = []
    for _, dates in idx.to_series().groupby(groups):
        start = dates.iloc[0]
        end = dates.iloc[-1]
        ranges.append(
            {
                "start": str(start.date()),
                "end": str(end.date()),
                "n_days": int((end - start).days + 1),
            }
        )
    return ranges


def main() -> None:
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    backup_path = _ensure_backup()

    sj_archuleta = _read_daily_csv(PATH_SJ_ARCHULETA)
    animas_farmington = _read_daily_csv(PATH_ANIMAS_FARMINGTON)
    sj_farmington_source = _read_daily_csv(backup_path)

    proxy = pd.DataFrame(index=sj_archuleta.index.union(animas_farmington.index))
    proxy["sj_archuleta_q_cfs"] = sj_archuleta["value"]
    proxy["animas_farmington_q_cfs"] = animas_farmington["value"]
    proxy["proxy_sum_q_cfs"] = proxy["sj_archuleta_q_cfs"] + proxy["animas_farmington_q_cfs"]

    overlap = pd.DataFrame(index=sj_farmington_source.index.union(proxy.index))
    overlap["sj_farmington_q_cfs"] = sj_farmington_source["value"]
    overlap["proxy_sum_q_cfs"] = proxy["proxy_sum_q_cfs"]
    overlap = overlap.dropna(subset=["sj_farmington_q_cfs", "proxy_sum_q_cfs"]).copy()

    proxy_total_af = float(overlap["proxy_sum_q_cfs"].sum() * CFS_DAY_TO_AF)
    obs_total_af = float(overlap["sj_farmington_q_cfs"].sum() * CFS_DAY_TO_AF)
    volume_bias_af = float(proxy_total_af - obs_total_af)
    volume_bias_cfs_per_day = float(
        volume_bias_af / (len(overlap) * CFS_DAY_TO_AF)
    )

    full_index = pd.date_range(
        start=sj_farmington_source.index.min(),
        end=sj_farmington_source.index.max(),
        freq="D",
    )
    source_full = sj_farmington_source.reindex(full_index)
    proxy_full = proxy.reindex(full_index)

    source_full["proxy_sum_q_cfs"] = proxy_full["proxy_sum_q_cfs"]
    source_full["proxy_adjusted_q_cfs"] = (
        source_full["proxy_sum_q_cfs"] - volume_bias_cfs_per_day
    ).clip(lower=0)

    missing_mask = source_full["value"].isna()
    fill_mask = missing_mask & source_full["proxy_adjusted_q_cfs"].notna()

    unfilled_missing = source_full.index[missing_mask & (~fill_mask)]
    if len(unfilled_missing) > 0:
        raise RuntimeError(
            "Some missing San Juan at Farmington dates could not be patched because "
            f"the proxy was unavailable. First unfilled date: {unfilled_missing.min().date()}"
        )

    patched = source_full[["value"]].copy()
    patched.loc[fill_mask, "value"] = source_full.loc[fill_mask, "proxy_adjusted_q_cfs"]
    patched.index.name = "time"

    if patched["value"].isna().any():
        raise RuntimeError("Patched Farmington series still contains missing values.")

    patched_rows = pd.DataFrame(index=patched.index[fill_mask])
    patched_rows["patched_value_cfs"] = patched.loc[fill_mask, "value"]
    patched_rows["proxy_sum_q_cfs"] = source_full.loc[fill_mask, "proxy_sum_q_cfs"]
    patched_rows["proxy_adjusted_q_cfs"] = source_full.loc[fill_mask, "proxy_adjusted_q_cfs"]
    patched_rows.index.name = "time"
    patched_rows.to_csv(PATH_PATCHED_ROWS)

    patched.reset_index().to_csv(PATH_SJ_FARMINGTON, index=False)

    patched_ranges = _continuous_ranges(patched_rows.index)
    summary = {
        "source_file": str(backup_path),
        "patched_file": str(PATH_SJ_FARMINGTON),
        "n_overlap_days": int(len(overlap)),
        "proxy_total_overlap_af": proxy_total_af,
        "observed_total_overlap_af": obs_total_af,
        "volume_bias_af": volume_bias_af,
        "volume_bias_cfs_per_day": volume_bias_cfs_per_day,
        "n_patched_days": int(fill_mask.sum()),
        "patched_ranges": patched_ranges,
        "first_patched_date": str(patched_rows.index.min().date()),
        "last_patched_date": str(patched_rows.index.max().date()),
        "full_series_start": str(patched.index.min().date()),
        "full_series_end": str(patched.index.max().date()),
    }
    PATH_PATCH_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_df = patched.loc["2017-01-01":].copy()
    source_plot = source_full.loc["2017-01-01":, "value"]
    patched_points = patched_rows.loc[patched_rows.index >= pd.Timestamp("2017-01-01")]

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(plot_df.index, plot_df["value"], color="tab:blue", linewidth=1.4, label="Patched SJ Farmington")
    ax.plot(source_plot.index, source_plot, color="0.75", linewidth=1.0, alpha=0.8, label="Original SJ Farmington")
    if not patched_points.empty:
        ax.scatter(
            patched_points.index,
            patched_points["patched_value_cfs"],
            s=10,
            color="tab:red",
            alpha=0.9,
            label="Patched days",
            zorder=3,
        )
        for rng in patched_ranges:
            start = pd.Timestamp(rng["start"])
            end = pd.Timestamp(rng["end"])
            if end < pd.Timestamp("2017-01-01"):
                continue
            ax.axvspan(start, end, color="tab:red", alpha=0.08)

    ax.set_title("SJ Farmington patched series (2017 to end of record)")
    ax.set_ylabel("Discharge (cfs)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")
    fig.savefig(PATH_VALIDATION_PLOT, dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote patched file: {PATH_SJ_FARMINGTON}")
    print(f"Wrote backup file: {backup_path}")
    print(f"Wrote patched rows: {PATH_PATCHED_ROWS}")
    print(f"Wrote patch summary: {PATH_PATCH_SUMMARY}")
    print(f"Wrote validation plot: {PATH_VALIDATION_PLOT}")


if __name__ == "__main__":
    main()
