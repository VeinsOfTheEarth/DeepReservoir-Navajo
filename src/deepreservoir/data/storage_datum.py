"""Storage-datum preprocessing for the Navajo Reservoir record.

The Reclamation daily export changes its elevation--storage table on
2021-10-01.  ``reported`` preserves those delivered storage values for frozen
policy reproduction.  ``elevation_2019`` maps every reported elevation through
the bundled 2019 Reclamation area-capacity relationship so training resets and
historic comparisons share one storage datum.
"""

from __future__ import annotations

from functools import lru_cache
import pickle

import numpy as np
import pandas as pd

from deepreservoir.data.metadata import project_metadata


STORAGE_DATUM_MODE_CHOICES: tuple[str, ...] = (
    "reported",
    "elevation_2019",
)
STORAGE_RELATIONSHIP_SOURCE = (
    "data/elevation_area_storage_relationships/"
    "NavajoReservoir Area_Capacity Table_508-VI.pdf"
)
STORAGE_RELATIONSHIP_PARAMETER = (
    "data/elevation_area_storage_relationships/"
    "2019_elevation_area_capacity.pkl"
)


def normalize_storage_datum_mode(value: str | None) -> str:
    """Return the canonical storage preprocessing mode."""
    text = str(value or "reported").strip().lower()
    aliases = {
        "": "reported",
        "legacy": "reported",
        "raw": "reported",
        "as_reported": "reported",
        "2019": "elevation_2019",
        "common_2019": "elevation_2019",
        "reclamation_2019": "elevation_2019",
    }
    normalized = aliases.get(text, text)
    if normalized not in STORAGE_DATUM_MODE_CHOICES:
        raise ValueError(
            f"Unsupported storage_datum_mode {value!r}; choose from "
            f"{sorted(STORAGE_DATUM_MODE_CHOICES)}"
        )
    return normalized


@lru_cache(maxsize=1)
def _elevation_to_capacity_2019():
    metadata = project_metadata()
    with open(metadata.path("elev_area_storage_pickle"), "rb") as stream:
        models = pickle.load(stream)
    try:
        return models["elevation_to_capacity"]
    except KeyError as exc:
        raise KeyError(
            "The elevation-area-storage parameter file lacks "
            "'elevation_to_capacity'."
        ) from exc


def apply_storage_datum_mode(
    frame: pd.DataFrame,
    *,
    mode: str | None = "reported",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a model-data frame using the selected storage datum.

    ``elevation_2019`` preserves the source values as ``storage_reported_af``
    and replaces ``storage_af`` with the 2019 table value evaluated at each
    reported elevation.  Missing elevations fail explicitly because silently
    mixing storage datums would recreate the discontinuity this mode removes.
    """
    resolved = normalize_storage_datum_mode(mode)
    out = frame.copy()
    metadata: dict[str, object] = {
        "mode": resolved,
        "storage_column": "storage_af",
    }
    if resolved == "reported":
        metadata["source"] = "reported reservoir storage"
        return out, metadata

    required = {"storage_af", "elev_ft"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise KeyError(
            "elevation_2019 storage preprocessing requires columns "
            f"{sorted(required)}; missing {missing}"
        )

    reported = pd.to_numeric(out["storage_af"], errors="coerce")
    elevation = pd.to_numeric(out["elev_ft"], errors="coerce")
    reported_values = reported.to_numpy(dtype=float)
    elevation_values = elevation.to_numpy(dtype=float)
    if not bool(np.isfinite(reported_values).all()) or not bool(
        np.isfinite(elevation_values).all()
    ):
        raise ValueError(
            "elevation_2019 storage preprocessing requires finite reported "
            "storage and elevation on every retained model date."
        )

    elevation_to_capacity = _elevation_to_capacity_2019()
    converted = np.asarray(
        elevation_to_capacity(elevation_values),
        dtype=float,
    ).reshape(-1)
    if converted.size != len(out) or not bool(np.isfinite(converted).all()):
        raise ValueError("The 2019 elevation-capacity conversion was not finite.")

    out["storage_reported_af"] = reported_values
    out["storage_af"] = converted
    delta = reported_values - converted
    metadata.update(
        {
            "source": "2019 Reclamation area-capacity table",
            "source_document": STORAGE_RELATIONSHIP_SOURCE,
            "relationship_parameter": STORAGE_RELATIONSHIP_PARAMETER,
            "conversion": "storage_af = elevation_to_capacity_2019(elev_ft)",
            "interpolation": "piecewise linear with endpoint clamping",
            "preserved_reported_column": "storage_reported_af",
            "n_converted": int(len(out)),
            "reported_minus_2019_mean_af": float(np.mean(delta)),
            "reported_minus_2019_median_af": float(np.median(delta)),
        }
    )
    return out, metadata
