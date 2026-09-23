# hydropower_model.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from scipy.interpolate import interp1d

from deepreservoir.data.metadata import project_metadata


_META = project_metadata()
_PATH_MODEL_PARAMS: Path = _META.path("hydropower_eta")


def _load_eta_from_pickle(pkl_path: Path) -> float:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "eta_eff" in obj:
        return float(obj["eta_eff"])
    raise ValueError("Unsupported hydropower parameter format; expected dict with 'eta_eff'.")


_ETA_LOADED: Optional[float] = None
if _PATH_MODEL_PARAMS.exists():
    try:
        _ETA_LOADED = _load_eta_from_pickle(_PATH_MODEL_PARAMS)
    except Exception:
        _ETA_LOADED = None


# Approximate knots read from USACE Navajo WCM (2010), Plate 7-4; that plate
# credits USBR drawing 711-D-38. The 600-cfs knot represents its low-flow bend.
_TAILWATER_Q_CFS = np.array(
    [0.0, 600.0, 2000.0, 4000.0, 8000.0, 16000.0, 24000.0, 32000.0, 40000.0],
    dtype=float,
)
_TAILWATER_ELEV_FT = np.array(
    [5711.7, 5713.0, 5713.3, 5713.7, 5714.4, 5716.1, 5717.8, 5719.2, 5720.4],
    dtype=float,
)


def _create_tailwater_model():
    return interp1d(
        _TAILWATER_Q_CFS,
        _TAILWATER_ELEV_FT,
        kind="linear",
        bounds_error=False,
        fill_value=(_TAILWATER_ELEV_FT[0], _TAILWATER_ELEV_FT[-1]),
    )


_TAILWATER_MODEL = _create_tailwater_model()


def _tailwater_ft_scalar(q_cfs: float) -> float:
    """Fast scalar tailwater interpolation for the environment hot path."""
    return float(np.interp(float(q_cfs), _TAILWATER_Q_CFS, _TAILWATER_ELEV_FT))


_RHO = 1000.0
_G = 9.81
_CFS_TO_CMS = 0.0283168
_FT_TO_M = 0.3048
_TURBINE_LIMIT_CFS = 1300.0
_PLANT_CAPACITY_MW = 32.0
_ENERGY_COEFF_BASE = _RHO * _G * _CFS_TO_CMS * _FT_TO_M * 24.0 / 1e6


def _resolve_eta_and_coeff(eta_eff: Optional[float]) -> tuple[float, float]:
    if eta_eff is None:
        if _ETA_LOADED is None:
            raise RuntimeError("No eta provided and hydropower parameters could not be loaded.")
        eta = float(_ETA_LOADED)
    else:
        eta = float(eta_eff)
    return eta, eta * _ENERGY_COEFF_BASE


def navajo_power_generation_scalar(
    cfs_value: float,
    elevation_ft: float,
    eta_eff: Optional[float] = None,
) -> float:
    """Daily energy production in MWh for a single release/elevation pair."""
    _, energy_coeff = _resolve_eta_and_coeff(eta_eff)

    q_cfs = float(cfs_value)
    if q_cfs <= 0.0:
        return 0.0

    head_ft = float(elevation_ft) - _tailwater_ft_scalar(q_cfs)
    if head_ft <= 0.0:
        return 0.0

    q_cfs_capped = min(q_cfs, _TURBINE_LIMIT_CFS)
    energy_mwh = energy_coeff * q_cfs_capped * head_ft
    return float(min(energy_mwh, _PLANT_CAPACITY_MW * 24.0))


def navajo_power_generation_model(
    cfs_values: Union[float, Sequence[float]],
    elevation_ft: Union[float, Sequence[float]],
    eta_eff: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Predict daily energy production (MWh) from release and reservoir elevation.

    Releases are capped at the Navajo turbine flow limit before energy is
    calculated. Excess release can satisfy other objectives, but does not
    generate additional hydropower in this model.
    """
    if np.isscalar(cfs_values) and np.isscalar(elevation_ft):
        return navajo_power_generation_scalar(
            cfs_value=float(cfs_values),
            elevation_ft=float(elevation_ft),
            eta_eff=eta_eff,
        )

    eta_eff, _ = _resolve_eta_and_coeff(eta_eff)

    q_cfs = np.asarray(cfs_values, dtype=float)
    elev_ft = np.asarray(elevation_ft, dtype=float)

    tailwater_ft = _TAILWATER_MODEL(q_cfs)
    head_m = np.clip((elev_ft - tailwater_ft) * _FT_TO_M, 0.0, None)
    q_cms = np.clip(q_cfs, 0.0, _TURBINE_LIMIT_CFS) * _CFS_TO_CMS

    power_mw = eta_eff * _RHO * _G * q_cms * head_m / 1e6
    power_mw = np.minimum(power_mw, _PLANT_CAPACITY_MW)
    energy_mwh = power_mw * 24.0

    if energy_mwh.ndim > 0 and energy_mwh.size > 1:
        return energy_mwh
    return float(energy_mwh)
