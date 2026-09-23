"""Rebuild Appendix B figures from the bundled hydropower calibration data.

The figure calculation uses only data, model parameters, runtime code, and
figure styling in this repository.  Files from the original analysis folder
may be supplied for provenance, but they are not required.  This script never
fits or exports parameters.
"""

from __future__ import annotations

import argparse
import ast
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


def runtime_constants(path: Path) -> dict:
    """Extract literal constants without importing the training environment."""
    wanted = {"_TAILWATER_Q_CFS", "_TAILWATER_ELEV_FT", "_RHO", "_G",
              "_CFS_TO_CMS", "_FT_TO_M", "_TURBINE_LIMIT_CFS", "_PLANT_CAPACITY_MW"}
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        expr = node.value
        if isinstance(expr, ast.Call):
            if not isinstance(expr.func, ast.Attribute) or expr.func.attr != "array":
                raise ValueError(f"Unexpected runtime declaration for {target.id}")
            expr = expr.args[0]
        values[target.id] = ast.literal_eval(expr)
    if set(values) != wanted:
        raise ValueError(f"Missing runtime constants: {sorted(wanted - set(values))}")
    return values


def load_model(paths: dict[str, Path]) -> dict:
    with paths["runtime_parameters"].open("rb") as stream:
        params = pickle.load(stream)
    if not isinstance(params, dict) or params.get("version") != "single_eta_v1":
        raise ValueError("Expected the active single_eta_v1 parameter dictionary")
    runtime = runtime_constants(paths["runtime_model"])
    q = [float(v) for v in runtime["_TAILWATER_Q_CFS"]]
    tw = [float(v) for v in runtime["_TAILWATER_ELEV_FT"]]
    if len(q) != len(tw) or not np.all(np.diff(q) > 0):
        raise ValueError("Invalid tailwater knots")
    if float(params["turbine_limit_cfs"]) != runtime["_TURBINE_LIMIT_CFS"]:
        raise ValueError("Parameter and runtime turbine-flow caps differ")
    if float(params["plant_capacity_MW"]) != runtime["_PLANT_CAPACITY_MW"]:
        raise ValueError("Parameter and runtime power caps differ")
    return {
        "version": params["version"], "eta_eff": float(params["eta_eff"]),
        "exported_at": params.get("exported_at"),
        "calibration_window_metadata": params.get("calibration_window"),
        "turbine_limit_cfs": float(runtime["_TURBINE_LIMIT_CFS"]),
        "configured_power_cap_MW": float(runtime["_PLANT_CAPACITY_MW"]),
        "density_kg_m3": float(runtime["_RHO"]),
        "gravity_m_s2": float(runtime["_G"]),
        "cfs_to_m3_s": float(runtime["_CFS_TO_CMS"]),
        "ft_to_m": float(runtime["_FT_TO_M"]),
        "tailwater_release_knots_cfs": q,
        "tailwater_elevation_knots_ft": tw,
    }


def energy_mwh(q_cfs: np.ndarray, elev_ft: np.ndarray, model: dict) -> np.ndarray:
    """Mirror the vector path in active hydropower_model.py (MWh per day)."""
    q = np.asarray(q_cfs, dtype=float)
    elev = np.asarray(elev_ft, dtype=float)
    tailwater = np.interp(q, model["tailwater_release_knots_cfs"],
                          model["tailwater_elevation_knots_ft"])
    head_m = np.maximum((elev - tailwater) * model["ft_to_m"], 0.0)
    q_turb_m3s = np.clip(q, 0.0, model["turbine_limit_cfs"]) * model["cfs_to_m3_s"]
    mw = (model["eta_eff"] * model["density_kg_m3"] * model["gravity_m_s2"]
          * q_turb_m3s * head_m / 1e6)
    return 24.0 * np.minimum(mw, model["configured_power_cap_MW"])


def load_usbr(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path, usecols=["Date", "Elevation (feet)", "Total Release (cfs)"])
    daily["Date"] = pd.to_datetime(daily["Date"], format="%d-%b-%y", errors="raise")
    future = daily["Date"].dt.year > 2024
    daily.loc[future, "Date"] -= pd.DateOffset(years=100)
    for column in ["Elevation (feet)", "Total Release (cfs)"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    if daily["Date"].duplicated().any():
        raise ValueError("Duplicate daily dates in USBR export")
    return daily.sort_values("Date").set_index("Date").loc["2000-01-01":"2022-12-31"]


def load_rectifhyd(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the bundled Navajo-only RectifHyd extract."""
    cols = ["plant", "year", "month", "RectifHyd_MWh", "recommended_data",
            "RectifHyd_method", "EIA_obs_freq", "smoothed", "scaled", "imputed"]
    navajo = pd.read_csv(path, usecols=cols)
    if set(navajo["plant"].dropna().unique()) != {"Navajo Dam"}:
        raise ValueError("Expected a Navajo Dam-only RectifHyd extract")
    navajo["date"] = pd.to_datetime(navajo["year"].astype(str) + " " + navajo["month"],
                                     format="%Y %b")
    navajo = navajo.sort_values("date").set_index("date")
    if navajo.index.has_duplicates:
        raise ValueError("Duplicate monthly RectifHyd dates for Navajo")
    flags = {}
    for column in ["recommended_data", "RectifHyd_method", "EIA_obs_freq"]:
        flags[column + "_counts"] = {str(k): int(v) for k, v in
                                     navajo[column].value_counts(dropna=False).items()}
    for column in ["smoothed", "scaled", "imputed"]:
        flags[column + "_months"] = int(navajo[column].fillna(False).astype(bool).sum())
    return navajo, flags


def set_style(code_root: Path) -> object:
    style_path = code_root / "paper/figure-support/figurestyle.py"
    spec = importlib.util.spec_from_file_location("navajo_figurestyle", style_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {style_path}")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.register_fonts()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans"],
        "font.size": 9.1, "axes.labelsize": 9.2, "axes.titlesize": 9.4,
        "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "legend.fontsize": 8.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    return style


def decorate(ax: plt.Axes, style: object) -> None:
    ax.set_facecolor(style.BACKGROUND)
    ax.grid(axis="y", color=style.GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(style.SPINE)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.12,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(HERE / f"{stem}.png", dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_model_response(daily: pd.DataFrame, model: dict, style: object) -> dict:
    valid = daily[["Total Release (cfs)", "Elevation (feet)"]].dropna()
    q_lo, q_hi = float(valid["Total Release (cfs)"].min()), float(valid["Total Release (cfs)"].max())
    elev_quantiles = valid["Elevation (feet)"].quantile([0.25, 0.5, 0.75]).to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), layout="constrained")
    q = np.linspace(0.0, 8000.0, 401)
    ax = axes[0]
    decorate(ax, style)
    ax.axvspan(q_lo, q_hi, color=style.OBJECTIVE_COLORS["hydropower"], alpha=0.085,
               lw=0, label="Historic release range")
    tailwater = np.interp(q, model["tailwater_release_knots_cfs"],
                          model["tailwater_elevation_knots_ft"])
    ax.plot(q, tailwater, color=style.OBJECTIVE_COLORS["hydropower"], lw=2.1,
            label="Implemented approximation")
    shown = np.asarray(model["tailwater_release_knots_cfs"]) <= 8000
    ax.scatter(np.asarray(model["tailwater_release_knots_cfs"])[shown],
               np.asarray(model["tailwater_elevation_knots_ft"])[shown],
               color=style.OBJECTIVE_COLORS["hydropower"], s=27,
               edgecolor="white", linewidth=0.7, zorder=4)
    ax.set(xlim=(0, 8000), ylim=(5711.5, 5714.75), xlabel="Total release (cfs)",
           ylabel="Tailwater elevation (ft)")
    ax.set_title("Implemented tailwater curve", loc="left", pad=8)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=style.GRID)
    style.add_panel_label(ax, "A", x=-0.025, y=1.06, ha="right", va="bottom", fontsize=11)

    ax = axes[1]
    decorate(ax, style)
    release = np.linspace(0.0, 5500.0, 551)
    colors = ["#94A3B8", style.OBJECTIVE_COLORS["hydropower"], "#9A3412"]
    for (quantile, elevation), color in zip(elev_quantiles.items(), colors):
        response = energy_mwh(release, np.full_like(release, elevation), model)
        ax.plot(release, response, lw=1.8, color=color,
                label=f"{int(quantile * 100)}th percentile: {elevation:.0f} ft")
    ax.axvline(model["turbine_limit_cfs"], color=style.TEXT, lw=1.1,
               ls=(0, (3, 2)), label="1,300-cfs turbine cap")
    ax.axhline(model["configured_power_cap_MW"] * 24, color=style.SPINE,
               lw=0.9, ls=(0, (1, 2)))
    ax.text(0.98, 0.98, "32-MW configured ceiling\n(768 MWh/day)",
            ha="right", va="top", transform=ax.transAxes, fontsize=7.9,
            color=style.TEXT)
    ax.set(xlim=(0, 5500), ylim=(0, 820), xlabel="Total release (cfs)",
           ylabel="Modeled energy (MWh/day)")
    ax.set_title("Daily response at historical elevations", loc="left", pad=8)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=style.GRID)
    style.add_panel_label(ax, "B", x=-0.025, y=1.06, ha="right", va="bottom", fontsize=11)
    save(fig, "model-response")
    return {
        "historic_daily_days_with_release_and_elevation": int(len(valid)),
        "historic_release_range_cfs": [q_lo, q_hi],
        "historic_elevation_quantiles_ft": {str(k): float(v) for k, v in elev_quantiles.items()},
        "response_figure_release_range_cfs": [0, 5500],
    }


def plot_monthly_fit(paired: pd.DataFrame, stats: dict, style: object) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4),
                             gridspec_kw={"height_ratios": [1.05, 1.0]}, layout="constrained")
    ax = axes[0]
    decorate(ax, style)
    ax.plot(paired.index, paired["rectifhyd_mwh"] / 1000, color=style.SPINE,
            lw=1.15, label="RectifHyd estimate", zorder=2)
    ax.plot(paired.index, paired["modeled_mwh"] / 1000,
            color=style.OBJECTIVE_COLORS["hydropower"], lw=1.15,
            label="Fitted model", zorder=3)
    ax.set_xlim(paired.index.min(), paired.index.max())
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Monthly energy (GWh)")
    ax.set_title("Monthly series, 2001–2022", loc="left", pad=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=style.GRID, ncol=2)
    style.add_panel_label(ax, "A", x=-0.025, y=1.06, ha="right", va="bottom", fontsize=11)

    ax = axes[1]
    decorate(ax, style)
    x = paired["rectifhyd_mwh"].to_numpy() / 1000
    y = paired["modeled_mwh"].to_numpy() / 1000
    limit = float(np.ceil(max(x.max(), y.max()) / 5) * 5)
    ax.plot([0, limit], [0, limit], color=style.TEXT, lw=1.0, ls=(0, (4, 2)))
    ax.scatter(x, y, s=21, color=style.OBJECTIVE_COLORS["hydropower"],
               alpha=0.62, edgecolors="white", linewidths=0.3, zorder=3)
    ax.set(xlim=(0, limit), ylim=(0, limit),
           xlabel="RectifHyd monthly estimate (GWh)", ylabel="Modeled monthly energy (GWh)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Paired months, fitted period", loc="left", pad=8)
    ax.text(0.97, 0.05,
            f"{stats['n_months']} months\nR² = {stats['r2']:.3f}\nRMSE = {stats['rmse_mwh']:,.0f} MWh",
            transform=ax.transAxes, ha="right", va="bottom", color=style.TEXT,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": style.GRID})
    style.add_panel_label(ax, "B", x=-0.025, y=1.06, ha="right", va="bottom", fontsize=11)
    save(fig, "monthly-fit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        help=(
            "Optional original Hydropower analysis directory. Its deck and scripts "
            "are hashed for provenance when present; they are not used to build figures."
        ),
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    paths = {
        "usbr_daily": repo / "data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv",
        "rectifhyd_navajo": repo / "data/hydropower/RectifYhd_v1.3_Navajo.csv",
        "runtime_parameters": repo / "data/hydropower/hydropower_parameters.pkl",
        "runtime_model": repo / "src/deepreservoir/define_env/hydropower_model.py",
        "digitized_knots": HERE / "tailwater-knots.csv",
        "figure_style": repo / "paper/figure-support/figurestyle.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    model = load_model(paths)
    digitized = pd.read_csv(paths["digitized_knots"])
    if (digitized["release_cfs"].astype(float).tolist()
            != model["tailwater_release_knots_cfs"]
            or digitized["tailwater_ft_ngvd"].astype(float).tolist()
            != model["tailwater_elevation_knots_ft"]):
        raise ValueError("Runtime tailwater knots differ from the audited plate reading")
    daily = load_usbr(paths["usbr_daily"])
    if (daily["Total Release (cfs)"].min() < model["tailwater_release_knots_cfs"][0]
            or daily["Total Release (cfs)"].max() > model["tailwater_release_knots_cfs"][-1]):
        raise ValueError("Calibration releases extend beyond the tailwater knots")
    rh, flags = load_rectifhyd(paths["rectifhyd_navajo"])
    daily = daily.assign(energy_mwh=energy_mwh(daily["Total Release (cfs)"].to_numpy(),
                                                  daily["Elevation (feet)"].to_numpy(), model))
    monthly = daily["energy_mwh"].resample("ME").sum(min_count=1)
    monthly.index = monthly.index.to_period("M").to_timestamp()
    paired = pd.concat([rh["RectifHyd_MWh"].rename("rectifhyd_mwh"),
                        monthly.rename("modeled_mwh")], axis=1).dropna()
    if (len(paired) != 264 or paired.index.min() != pd.Timestamp("2001-01-01")
            or paired.index.max() != pd.Timestamp("2022-12-01")):
        raise ValueError(f"Unexpected calibration months: {len(paired)}, {paired.index.min()}, {paired.index.max()}")
    diff = paired["modeled_mwh"] - paired["rectifhyd_mwh"]
    stats = {
        "first_month": str(paired.index.min().date()),
        "last_month": str(paired.index.max().date()),
        "n_months": int(len(paired)),
        "r2": float(1 - np.sum(diff.to_numpy() ** 2) /
                    np.sum((paired["rectifhyd_mwh"] - paired["rectifhyd_mwh"].mean()).to_numpy() ** 2)),
        "rmse_mwh": float(np.sqrt(np.mean(diff.to_numpy() ** 2))),
        "mae_mwh": float(np.mean(np.abs(diff.to_numpy()))),
        "mean_model_minus_rectifhyd_mwh": float(diff.mean()),
        "reference_is_independent_monthly_observation": False,
        "is_held_out_validation": False,
    }
    if not (np.isfinite(stats["r2"]) and np.isfinite(stats["rmse_mwh"])
            and 0 < stats["r2"] < 1 and stats["rmse_mwh"] > 0):
        raise ValueError(f"Invalid fit stats: {stats}")
    style = set_style(repo)
    response_stats = plot_model_response(daily, model, style)
    plot_monthly_fit(paired, stats, style)
    summary = {
        "model": model,
        "calibration_fit": stats,
        "rectifhyd_navajo_flags": flags,
        "historic_response": response_stats,
        "methods_and_limits": {
            "tailwater": "Nine points approximately digitized from the slide-2 copy of USACE 2010 draft Navajo water control manual Plate 7-4 and joined by linear interpolation; not an exact tracing",
            "tailwater_source_url": "https://water.usace.army.mil/cda/documents/wc/2560/Navajo_WCM_Draft_8-5-10Redacted.pdf",
            "monthly_reference": "RectifHyd v1.3 monthly estimates; all Navajo rows use release-based disaggregation",
            "runtime_elevation": "Runtime uses storage-derived end-of-step elevation; calibration reconstruction uses reported daily USBR elevation",
            "power_ceiling": "32 MW is the configured model cap, matching historical 2000 EIA net seasonal capability; 30 MW is EIA nameplate and current Reclamation capacity",
        },
        "inputs": {name: input_record(path, repo) for name, path in paths.items()},
        "optional_provenance": {
            name: optional_record(
                args.analysis_root / filename if args.analysis_root is not None else None,
                repo,
            )
            for name, filename in {
                "original_parameter_pickle": "parameters.pkl",
                "original_calibration_script": "eff_single_eta.py",
                "original_analysis_deck": "hydropower_model_navajo.pptx",
            }.items()
        },
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
