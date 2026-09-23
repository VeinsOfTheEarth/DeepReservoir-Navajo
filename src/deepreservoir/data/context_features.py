from __future__ import annotations

import pandas as pd

from deepreservoir.data.metadata import project_metadata
from deepreservoir.define_env.spring_peak_release.opportunity_index import (
    OIParams,
    ThresholdOIParams,
    precompute_oi_by_wy,
    precompute_threshold_oi_by_wy,
)


SWE_CONTEXT_COLUMNS: tuple[str, ...] = (
    "animas_swe_m",
    "uppersj_swe_m",
)

OI_CONTEXT_COLUMNS: tuple[str, ...] = (
    "spr_oi",
    "spr_go",
)

THRESHOLD_OI_CONTEXT_COLUMNS: tuple[str, ...] = (
    "spring_oi_10000cfs_5d",
    "spring_oi_8000cfs_10d",
    "spring_oi_5000cfs_21d",
    "spring_oi_2500cfs_10d",
)

OPERATIONAL_CONTEXT_COLUMNS: tuple[str, ...] = (
    SWE_CONTEXT_COLUMNS
    + OI_CONTEXT_COLUMNS
    + THRESHOLD_OI_CONTEXT_COLUMNS
)


def _water_year_by_day(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
        (index.year + (index.month >= 10)).astype(int),
        index=index,
        name="water_year",
        dtype="int64",
    )


def add_operational_context_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct SPR opportunity-index fields retained in the recorded outputs.

    These diagnostic fields do not enter the selected policy's observations.
    Its actual input order is defined in drl/observations.py.
    """

    required = {
        "storage_af",
        "inflow_cfs",
        "animas_farmington_q_cfs",
        "animas_swe_m",
    }
    missing = sorted(required - set(model_df.columns))
    if missing:
        raise KeyError(
            "model_df is missing columns required for operational context features: "
            f"{missing}"
        )

    out = model_df.copy()
    wy = _water_year_by_day(out.index)

    pm = project_metadata()
    oi_params = OIParams.load(pm.path("params.spr_oi_params_json"))
    oi_by_wy = precompute_oi_by_wy(out, oi_params)
    out["spr_oi"] = wy.map(oi_by_wy["oi"].to_dict()).fillna(0.0).astype(float)
    out["spr_go"] = (
        wy.map(oi_by_wy["go"].to_dict())
        .astype("boolean")
        .fillna(False)
        .astype(float)
    )

    threshold_oi_params = ThresholdOIParams.load(
        pm.path("params.spr_threshold_oi_params_json")
    )
    threshold_oi_by_wy = precompute_threshold_oi_by_wy(out, threshold_oi_params)
    for key in threshold_oi_params.by_key():
        out[f"spring_oi_{key}"] = (
            wy.map(threshold_oi_by_wy[f"oi_{key}"].to_dict())
            .fillna(0.0)
            .astype(float)
        )
        out[f"spring_go_{key}"] = (
            wy.map(threshold_oi_by_wy[f"go_{key}"].to_dict())
            .astype("boolean")
            .fillna(False)
            .astype(float)
        )

    return out
