"""Refit Navajo's single efficiency after a documented tailwater-curve change.

Without --output, print the proposed fit and leave all source files untouched.
The calculation uses the Navajo-only RectifHyd extract bundled with this repo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from build import (DEFAULT_REPO, HERE, energy_mwh, load_model,
                   load_rectifhyd, load_usbr, sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, help="Explicit active parameter-pickle destination")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    paths = {
        "usbr_daily": repo / "data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv",
        "rectifhyd_navajo": repo / "data/hydropower/RectifYhd_v1.3_Navajo.csv",
        "runtime_parameters": repo / "data/hydropower/hydropower_parameters.pkl",
        "runtime_model": repo / "src/deepreservoir/define_env/hydropower_model.py",
    }
    model = load_model(paths)
    knots = pd.read_csv(HERE / "tailwater-knots.csv", comment="#")
    if (knots["release_cfs"].astype(float).tolist() != model["tailwater_release_knots_cfs"]
            or knots["tailwater_ft_ngvd"].astype(float).tolist()
            != model["tailwater_elevation_knots_ft"]):
        raise ValueError("Runtime tailwater knots differ from audited digitization CSV")
    daily = load_usbr(paths["usbr_daily"])
    if (daily["Total Release (cfs)"].min() < model["tailwater_release_knots_cfs"][0]
            or daily["Total Release (cfs)"].max() > model["tailwater_release_knots_cfs"][-1]):
        raise ValueError("Calibration release lies beyond tailwater chart")
    rh, _ = load_rectifhyd(paths["rectifhyd_navajo"])
    release = daily["Total Release (cfs)"].to_numpy()
    elevation = daily["Elevation (feet)"].to_numpy()
    observed = rh["RectifHyd_MWh"].rename("rectifhyd_mwh")

    def paired(eta: float) -> pd.DataFrame:
        candidate = dict(model, eta_eff=float(eta))
        modeled = pd.Series(energy_mwh(release, elevation, candidate), index=daily.index)
        monthly = modeled.resample("ME").sum(min_count=1)
        monthly.index = monthly.index.to_period("M").to_timestamp()
        result = pd.concat([observed, monthly.rename("modeled_mwh")], axis=1).dropna()
        if (len(result) != 264 or result.index.min() != pd.Timestamp("2001-01-01")
                or result.index.max() != pd.Timestamp("2022-12-01")):
            raise ValueError("Unexpected paired-month calibration window")
        return result

    def rmse(eta: float) -> float:
        frame = paired(eta)
        return float(np.sqrt(np.mean((frame["modeled_mwh"] - frame["rectifhyd_mwh"]) ** 2)))

    fit = minimize_scalar(rmse, method="bounded", bounds=(0.3, 0.95),
                          options={"xatol": 1e-12})
    if not fit.success:
        raise RuntimeError(f"Efficiency optimization failed: {fit.message}")
    old = float(model["eta_eff"])
    report = {
        "old_eta_eff": old,
        "old_eta_rmse_mwh": rmse(old),
        "new_eta_eff": float(fit.x),
        "new_eta_rmse_mwh": float(fit.fun),
        "paired_months": 264,
        "source_sha256": {
            path.resolve().relative_to(repo).as_posix(): sha256(path)
            for path in paths.values()
        },
        "knots_sha256": sha256(HERE / "tailwater-knots.csv"),
    }
    if args.output:
        if args.output.resolve() != paths["runtime_parameters"].resolve():
            raise ValueError("Output must be the active code repository parameter file")
        with paths["runtime_parameters"].open("rb") as stream:
            params = pickle.load(stream)
        if params.get("version") != "single_eta_v1":
            raise ValueError("Unexpected parameter format")
        params["eta_eff"] = float(fit.x)
        params["exported_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        params["notes"] = ("Single global efficiency refitted against 264 RectifHyd months "
                           "after correcting tailwater knots from USACE Plate 7-4; "
                           "the source curve credits USBR drawing 711-D-38.")
        params["tailwater_knots_sha256"] = report["knots_sha256"]
        params["calibration_input_sha256"] = {
            key: sha256(paths[key]) for key in ("usbr_daily", "rectifhyd_navajo")
        }
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(params, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(args.output)
        report["written_to"] = str(args.output)
        report["new_parameters_sha256"] = sha256(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
