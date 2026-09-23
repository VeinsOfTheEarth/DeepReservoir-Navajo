"""Build the Appendix A elevation--capacity audit figure.

The builder uses the processed relation, active interpolation, historic export,
and shared figure style bundled with this repository.  An official source PDF
may be supplied for provenance, but it is not needed to rebuild the figure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[3]
TABLE_REL = Path("data/elevation_area_storage_relationships/elevation_storage_area_2019.csv")
PICKLE_REL = Path("data/elevation_area_storage_relationships/2019_elevation_area_capacity.pkl")
REPORT_REL = Path("data/elevation_area_storage_relationships/NavajoReservoir Area_Capacity Table_508-VI.pdf")
RESERVOIR_REL = Path(
    "data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv"
)
TABLE_CHANGE = pd.Timestamp("2021-10-01")
EVALUATION_START = pd.Timestamp("2014-01-01")
DEADPOOL_FT = 5775.0
SPILL_FT = 6085.0


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


def load_style(repo_root: Path) -> object:
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
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    return style


def load_inputs(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, Path]]:
    paths = {
        "table": repo_root / TABLE_REL,
        "interpolators": repo_root / PICKLE_REL,
        "historic_reservoir_export": repo_root / RESERVOIR_REL,
        "figure_style": repo_root / "paper/figure-support/figurestyle.py",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    table = pd.read_csv(
        paths["table"],
        usecols=["Elevation (ft)", "Capacity (ac-ft)", "Area (ac)"],
    ).rename(
        columns={
            "Elevation (ft)": "elev_ft",
            "Capacity (ac-ft)": "capacity_af",
            "Area (ac)": "area_ac",
        }
    )
    for column in table:
        table[column] = pd.to_numeric(table[column], errors="raise")
    if not table["elev_ft"].is_monotonic_increasing:
        raise ValueError("Elevation table is not increasing")
    if not table["capacity_af"].is_monotonic_increasing:
        raise ValueError("Capacity table is not increasing")

    with paths["interpolators"].open("rb") as stream:
        interpolators = pickle.load(stream)
    required = {"elevation_to_capacity", "capacity_to_elevation"}
    if not required.issubset(interpolators):
        raise KeyError(f"Interpolator pickle lacks {sorted(required - set(interpolators))}")

    historic = pd.read_csv(
        paths["historic_reservoir_export"],
        usecols=["Date", "Elevation (feet)", "Storage (af)"],
    ).rename(
        columns={"Elevation (feet)": "elev_ft", "Storage (af)": "reported_storage_af"}
    )
    historic["date"] = pd.to_datetime(historic.pop("Date"), format="%d-%b-%y", errors="raise")
    future = historic["date"].dt.year > 2024
    historic.loc[future, "date"] -= pd.DateOffset(years=100)
    historic = historic.sort_values("date").set_index("date")
    if historic.index.duplicated().any():
        raise ValueError("Duplicate dates in historic reservoir export")
    historic["curve_storage_af"] = np.asarray(
        interpolators["elevation_to_capacity"](historic["elev_ft"].to_numpy(float)),
        dtype=float,
    )
    historic["reported_minus_curve_af"] = (
        historic["reported_storage_af"] - historic["curve_storage_af"]
    )
    return table, historic, interpolators, paths


def verify(table: pd.DataFrame, interpolators: dict) -> dict[str, float | int | bool]:
    elev = table["elev_ft"].to_numpy(float)
    capacity = table["capacity_af"].to_numpy(float)
    implemented_capacity = np.asarray(interpolators["elevation_to_capacity"](elev), dtype=float)
    implemented_elev = np.asarray(interpolators["capacity_to_elevation"](capacity), dtype=float)
    max_capacity_error = float(np.max(np.abs(implemented_capacity - capacity)))
    max_elev_error = float(np.max(np.abs(implemented_elev - elev)))
    if max_capacity_error > 1e-8 or max_elev_error > 1e-10:
        raise ValueError(
            "Active interpolation object does not reproduce the 0.01-ft table: "
            f"{max_capacity_error=} {max_elev_error=}"
        )
    return {
        "table_rows": int(len(table)),
        "table_min_elevation_ft": float(elev.min()),
        "table_max_elevation_ft": float(elev.max()),
        "table_max_capacity_af": float(capacity.max()),
        "deadpool_elevation_ft": DEADPOOL_FT,
        "deadpool_capacity_af": float(interpolators["elevation_to_capacity"](DEADPOOL_FT)),
        "spill_elevation_ft": SPILL_FT,
        "spill_capacity_af": float(interpolators["elevation_to_capacity"](SPILL_FT)),
        "max_table_to_interpolator_capacity_error_af": max_capacity_error,
        "max_table_to_interpolator_elevation_error_ft": max_elev_error,
    }


def decorate(ax: plt.Axes, style: object) -> None:
    ax.set_facecolor(style.BACKGROUND)
    ax.grid(color=style.GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(style.SPINE)


def plot(table: pd.DataFrame, historic: pd.DataFrame, style: object) -> None:
    blue = style.OBJECTIVE_COLORS["storage"]
    orange = style.OBJECTIVE_COLORS["hydropower"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.55), layout="constrained")

    ax = axes[0]
    decorate(ax, style)
    pre = historic.loc[historic.index < TABLE_CHANGE]
    post = historic.loc[historic.index >= TABLE_CHANGE]
    # Plot observations first so the official curve remains visible.
    ax.scatter(
        pre["reported_storage_af"] / 1e6,
        pre["elev_ft"],
        s=4,
        color=style.SPINE,
        alpha=0.22,
        linewidths=0,
        label="Reported storage before 1 Oct 2021",
    )
    ax.scatter(
        post["reported_storage_af"] / 1e6,
        post["elev_ft"],
        s=7,
        color=orange,
        alpha=0.52,
        linewidths=0,
        label="Reported storage from 1 Oct 2021",
    )
    ax.plot(
        table["capacity_af"] / 1e6,
        table["elev_ft"],
        color=blue,
        linewidth=1.8,
        label="Processed 2019 capacity relation",
        zorder=5,
    )
    ax.axhline(SPILL_FT, color=style.OBJECTIVE_COLORS["dam_safety"], lw=0.9, ls="--")
    ax.text(
        0.16,
        SPILL_FT - 2.0,
        "Spillway crest: 6,085 ft",
        color=style.OBJECTIVE_COLORS["dam_safety"],
        fontsize=7.5,
        ha="left",
        va="top",
    )
    ax.set_xlim(-0.04, 1.82)
    ax.set_ylim(5768, 6097)
    ax.set_xlabel("Storage above deadpool (million acre-feet)")
    ax.set_ylabel("Reservoir elevation (ft, RPVD)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.94)
    panel_label = style.add_panel_label(ax, "A")
    panel_label.set_bbox({"facecolor": style.BACKGROUND, "edgecolor": "none", "pad": 0.2})

    ax = axes[1]
    decorate(ax, style)
    ax.plot(
        historic.index,
        historic["reported_minus_curve_af"] / 1000.0,
        color=blue,
        linewidth=0.85,
    )
    ax.axhline(0.0, color=style.TEXT, lw=0.8)
    ax.axvline(TABLE_CHANGE, color=orange, lw=1.1, ls="--")
    ax.annotate(
        "Reported storage aligns\nwith the 2019 relation",
        xy=(TABLE_CHANGE, 2.0),
        xytext=(pd.Timestamp("2005-01-01"), 17.0),
        ha="left",
        va="bottom",
        color=style.TEXT,
        fontsize=8.0,
        arrowprops={"arrowstyle": "->", "color": orange, "lw": 1.0},
    )
    ax.set_ylim(-6, 60)
    ax.set_ylabel("Reported minus 2019-curve storage\n(thousand acre-feet)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style.add_panel_label(ax, "B")

    fig.savefig(
        HERE / "elevation-capacity-audit.pdf",
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        HERE / "elevation-capacity-audit.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)


def summary(
    table: pd.DataFrame,
    historic: pd.DataFrame,
    interpolators: dict,
    paths: dict[str, Path],
    repo_root: Path,
    source_report: Path | None,
) -> dict:
    out = {
        "figure": "elevation-capacity-audit.pdf",
        "purpose": (
            "Verify the active elevation-capacity interpolation and document the abrupt shift "
            "in the historic Reclamation storage series."
        ),
        "official_sources": {
            "reservoir_surveys_index": "https://www.usbr.gov/tsc/techreferences/reservoir.html",
            "area_capacity_table": (
                "https://www.usbr.gov/tsc/techreferences/reservoir/"
                "NavajoReservoir2019AreaCapacityTables_final508VI.pdf"
            ),
            "sedimentation_survey": (
                "https://www.usbr.gov/tsc/techreferences/reservoir/"
                "NavajoReservoir2019SedimentationSurvey_final508VI.pdf"
            ),
            "historic_data_interface": "https://www.usbr.gov/rsvrWater/HistoricalApp.html",
        },
        "processed_relation_provenance": {
            "official_report_range_ft": [5775.0, 6110.0],
            "processed_range_ft": [float(table["elev_ft"].min()), float(table["elev_ft"].max())],
            "omitted_final_rows": 301,
            "omitted_rows_position": "above the modeled 6085-ft spill elevation",
            "verification_scope": (
                "processed CSV against active interpolation; source-PDF extraction not repeated"
            ),
        },
        "verification": verify(table, interpolators),
        "historic_export": {
            "first_date": str(historic.index.min().date()),
            "last_date": str(historic.index.max().date()),
            "rows": int(len(historic)),
            "apparent_relation_shift_date": str(TABLE_CHANGE.date()),
            "day_before": {
                key: float(value)
                for key, value in historic.loc[TABLE_CHANGE - pd.Timedelta(days=1)].items()
            },
            "change_date": {
                key: float(value) for key, value in historic.loc[TABLE_CHANGE].items()
            },
            "median_reported_minus_curve_before_change_af": float(
                historic.loc[historic.index < TABLE_CHANGE, "reported_minus_curve_af"].median()
            ),
            "median_reported_minus_curve_from_change_af": float(
                historic.loc[historic.index >= TABLE_CHANGE, "reported_minus_curve_af"].median()
            ),
            "evaluation_start": {
                key: float(value) for key, value in historic.loc[EVALUATION_START].items()
            },
        },
        "inputs": {
            name: input_record(path, repo_root) for name, path in paths.items()
        },
        "optional_provenance": {
            "area_capacity_report": optional_record(source_report, repo_root),
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--area-capacity-report",
        type=Path,
        help="Optional official source PDF to hash for provenance",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    table, historic, interpolators, paths = load_inputs(repo_root)
    style = load_style(repo_root)
    plot(table, historic, style)
    source_report = args.area_capacity_report
    if source_report is None:
        bundled_report = repo_root / REPORT_REL
        source_report = bundled_report if bundled_report.is_file() else None
    payload = summary(table, historic, interpolators, paths, repo_root, source_report)
    (HERE / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {HERE / 'elevation-capacity-audit.pdf'}")


if __name__ == "__main__":
    main()
