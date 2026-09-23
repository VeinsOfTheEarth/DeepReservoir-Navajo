from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import pickle

import numpy as np

from deepreservoir.data.metadata import project_metadata


# Authoritative Navajo physical elevations. Storage thresholds are derived from
# the current elevation-capacity relationship built from the 2019 table.
NAVAJO_DEADPOOL_ELEV_FT = 5775.0
NAVAJO_SPILL_ELEV_FT = 6085.0


@dataclass(frozen=True)
class NavajoStorageThresholds:
    deadpool_elev_ft: float
    spill_elev_ft: float
    deadpool_storage_af: float
    max_storage_af: float


def _scalar_interp_value(fn, x: float) -> float:
    arr = np.asarray(fn(np.asarray([float(x)], dtype=float)), dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Interpolator returned no values.")
    return float(arr[0])


@lru_cache(maxsize=1)
def get_navajo_storage_thresholds() -> NavajoStorageThresholds:
    """Return Navajo deadpool/spill thresholds derived from the E-S curve.

    Storage in this repo follows the established "above deadpool" convention,
    so the derived deadpool storage is expected to be 0 AF. We still compute it
    from the current interpolator so every consumer uses the same source of
    truth and stays aligned with the active elevation-capacity table.
    """
    m = project_metadata()
    with open(m.path("elev_area_storage_pickle"), "rb") as f:
        models = pickle.load(f)

    elev_to_capacity = models.get("elevation_to_capacity")
    if elev_to_capacity is None:
        capacity_to_elev = models["capacity_to_elevation"]
        caps = np.linspace(0.0, 2_000_000.0, 20001, dtype=float)
        elevs = np.asarray(capacity_to_elev(caps), dtype=float)
        order = np.argsort(elevs)
        elevs_sorted = elevs[order]
        caps_sorted = caps[order]

        def elev_to_capacity(elev_ft):
            elev_arr = np.asarray(elev_ft, dtype=float)
            return np.interp(elev_arr, elevs_sorted, caps_sorted)

    deadpool_storage_af = max(
        0.0,
        _scalar_interp_value(elev_to_capacity, NAVAJO_DEADPOOL_ELEV_FT),
    )
    max_storage_af = max(
        deadpool_storage_af,
        _scalar_interp_value(elev_to_capacity, NAVAJO_SPILL_ELEV_FT),
    )
    return NavajoStorageThresholds(
        deadpool_elev_ft=float(NAVAJO_DEADPOOL_ELEV_FT),
        spill_elev_ft=float(NAVAJO_SPILL_ELEV_FT),
        deadpool_storage_af=float(deadpool_storage_af),
        max_storage_af=float(max_storage_af),
    )
