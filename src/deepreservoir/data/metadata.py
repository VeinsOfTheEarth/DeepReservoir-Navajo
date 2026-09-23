# metadata.py

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "src").exists() and (p / "data").exists():
            return p
    for p in here.parents:
        if p.name == "src":
            return p.parent
    return here.parent

@dataclass
class Metadata:
    base_dir: Path = field(default_factory=repo_root)

    # explicit, no-magic configs: edit these dicts directly
    daily_series: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    continuous_series: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # internal cache for flattened paths
    _paths_index: Dict[str, Path] = field(
        default_factory=dict, init=False, repr=False
    )

    def _abs(self, p: Path | str) -> Path:
        p = p if isinstance(p, Path) else Path(p)
        return p if p.is_absolute() else (self.base_dir / p)

    def resolve_paths(self) -> "Metadata":
        for group in (self.daily_series, self.continuous_series, self.tables, self.model_params):
            for _, cfg in group.items():
                cfg["path"] = self._abs(cfg["path"])
        # clear cache since underlying paths changed
        self._paths_index.clear()
        return self

    # --- convenience helpers -------------------------------------------------
    def _iter_groups(self):
        """Yield (prefix, group_dict) for building a flat index."""
        yield "daily", self.daily_series
        yield "continuous", self.continuous_series
        yield "tables", self.tables
        yield "params", self.model_params

    def build_paths_index(self, overwrite: bool = False) -> Dict[str, Path]:
        """
        Build or return a flattened mapping of name → Path.

        Keys look like:
          - 'daily.reservoir'
          - 'continuous.sj_farmington'
          - 'tables.elev_area_storage_data'
          - 'params.niip_demand_spline_pkl'
        """
        if self._paths_index and not overwrite:
            return self._paths_index

        idx: Dict[str, Path] = {}
        for prefix, group in self._iter_groups():
            for name, cfg in group.items():
                p = cfg.get("path")
                if p is None:
                    continue
                key = f"{prefix}.{name}"
                idx[key] = p

        self._paths_index = idx
        return idx

    @property
    def paths(self) -> Dict[str, Path]:
        """
        Flattened view of all known paths.

        Example:
            m = project_metadata()
            m.paths["daily.reservoir"]
            m.paths["params.niip_demand_spline_pkl"]
        """
        return self.build_paths_index()

    def path(self, key: str) -> Path:
        """
        Get a Path by name, with or without prefix.

        Examples:
            m.path("params.niip_demand_spline_pkl")
            m.path("niip_demand_spline_pkl")  # works if unique

        Raises KeyError if missing or ambiguous.
        """
        idx = self.build_paths_index()

        # 1) exact key first (e.g., 'params.niip_demand_spline_pkl')
        if key in idx:
            return idx[key]

        # 2) allow bare names (e.g., 'niip_demand_spline_pkl')
        matches = [k for k in idx if k.split(".", 1)[1] == key]
        if not matches:
            raise KeyError(f"No path named '{key}' in metadata.paths")
        if len(matches) > 1:
            raise KeyError(
                f"Ambiguous key '{key}'. Matches: {matches}. "
                "Use a fully-qualified key like 'daily.reservoir'."
            )
        return idx[matches[0]]


# Rename of former "build_scaffold" → descriptive:
def project_metadata() -> Metadata:
    """
    Return the project's default metadata (paths, index, columns) with
    all relative paths resolved against the repo root.
    """
    m = Metadata()

    # DAILY time series (single-day index; all already in imperial)
    m.daily_series = {

        # Historic reservor data
        "reservoir": {
            "path": "data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv",
            "reader": "csv",
            "index": {"name": "Date", "format": "%d-%b-%y", "two_digit_year_fix": "force_1900s"},
            "columns": {
                "Total Release (cfs)": "release_cfs",
                "Storage (af)": "storage_af",
                "Elevation (feet)": "elev_ft",
            },
            "required": ("release_cfs", "storage_af", "elev_ft"),
            "duplicates": "last",
        },
        "inflow": {
            "path": "data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv",
            "reader": "csv",
            "index": {"name": "Date", "format": "%d-%b-%y"},
            "columns": {"Inflow** (cfs)": "inflow_cfs"},
            "required": ("inflow_cfs",),
            "duplicates": "last",
        },
        "evaporation": {
            "path": "data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv",
            "reader": "csv",
            "index": {"name": "Date", "format": "%d-%b-%y"},
            "columns": {"Evaporation (af)": "evap_af"},
            "required": ("evap_af",),
            "duplicates": "last",
        },

        # Daily gages
        "sj_archuleta": {
            "path": "data/daily_flows/daily_sj_archuleta.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%Y-%m-%d"},
            "columns": {"value": "sj_archuleta_q_cfs"},
            "required": ("sj_archuleta_q_cfs",),
        },
        "sj_farmington": {
            "path": "data/daily_flows/daily_sj_farmington.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%Y-%m-%d"},
            "columns": {"value": "sj_farmington_q_cfs"},
            "required": ("sj_farmington_q_cfs",),
        },
        "sj_fourcorners": {
            "path": "data/daily_flows/daily_sj_fourcorners.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%Y-%m-%d"},
            "columns": {"value": "sj_fourcorners_q_cfs"},
            "required": ("sj_fourcorners_q_cfs",),
        },
        "sj_shiprock": {
            "path": "data/daily_flows/daily_sj_shiprock.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%Y-%m-%d"},
            "columns": {"value": "sj_shiprock_q_cfs"},
            "required": ("sj_shiprock_q_cfs",),
        },
        "sj_bluff": {
            "path": "data/daily_flows/daily_sj_bluff.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%Y-%m-%d"},
            "columns": {"value": "sj_bluff_q_cfs"},
            "required": ("sj_bluff_q_cfs",),
        },
        "animas_farmington": {
            "path": "data/daily_flows/daily_animas_farmington.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%m/%d/%Y"},
            "columns": {"value": "animas_farmington_q_cfs"},
            "required": ("animas_farmington_q_cfs",),
        },
        "chaco_waterflow": {
            "path": "data/daily_flows/daily_chaco_waterflow.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%Y-%m-%d"},
            "columns": {"value": "chaco_waterflow_q_cfs"},
            "required": ("chaco_waterflow_q_cfs",),
        },
        "chinle_mexicanwater": {
            "path": "data/daily_flows/daily_chinle_mexicanwater.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%Y-%m-%d"},
            "columns": {"value": "chinle_mexicanwater_q_cfs"},
            "required": ("chinle_mexicanwater_q_cfs",),
        },
        "laplata_farmington": {
            "path": "data/daily_flows/daily_laplata_farmington.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%Y-%m-%d"},
            "columns": {"value": "laplata_farmington_q_cfs"},
            "required": ("laplata_farmington_q_cfs",),
        },
        "mancos_towaoc": {
            "path": "data/daily_flows/daily_mancos_towaoc.csv",
            "reader": "csv",
            "index": {"name": "time", "format": "%Y-%m-%d"},
            "columns": {"value": "mancos_towaoc_q_cfs"},
            "required": ("mancos_towaoc_q_cfs",),
        },

         # --- SWE (daily) for model inputs ---
         # Animas basin SWE → required by the model
         "swe_animas": {
             "path": "data/snow_water_equivalent/Animas_swe_daily.csv",
             "reader": "csv",
             "index": {"name": "date", "format": "%Y-%m-%d"},
             "columns": {"snow_depth_water_equivalent": "animas_swe_m"},
             "required": ("animas_swe_m",),
         },
         # Upper San Juan SWE → loadable but not required by the model
         "swe_upper_sj": {
             "path": "data/snow_water_equivalent/UpperSJ_swe_daily.csv",
             "reader": "csv",
             "index": {"name": "date", "format": "%Y-%m-%d"},
             "columns": {"snow_depth_water_equivalent": "uppersj_swe_m"},
             "required": ("uppersj_swe_m",),
         },


        # NIIP
        "niip_historic": {
            "path": "data/niip/NAVAJOINDIANIRRIGATIONPROJECT07-17-2025T13.21.47.csv",
            "reader": "csv",
            "read_kwargs": {
                "usecols": ["Date", "Flow (cfs)"],  # keep only what we need
                "comment": "*",                      # drop footnote lines
                "skipinitialspace": True,            # harmless; trims after commas
            },
            "index": {"name": "Date", "format": "%d-%b-%y"},
            "columns": {"Flow (cfs)": "niip_flow_cfs"},
            "required": ("niip_flow_cfs",),
            "duplicates": "last",
    }
    }

    # Continuous/raw curation series are intentionally not part of the public
    # runtime metadata. The model uses processed daily series above.
    m.continuous_series = {}

    # Non-time tables (E–A–S relationship, ...)
    m.tables = {
        "elev_area_storage_data": {
            "path": "data/elevation_area_storage_relationships/elevation_storage_area_2019.csv",
            "reader": "csv",
            "index": {"name": None, "format": None},  # do not parse a datetime index
            "columns": {
                "Elevation (ft)": "elev_ft",
                "Area (ac)": "area_ac",
                "Capacity (ac-ft)": "capacity_af",
            },
            "required": ("elev_ft", "area_ac", "capacity_af"),
        },
    }

    # Model parameter files (pickles, etc.)
    m.model_params = {
        "hydropower_eta": {
            "path": "data/hydropower/hydropower_parameters.pkl",
            "kind": "pickle",
            "description": "Global efficiency eta for Navajo hydropower model",
        },
        "elev_area_storage_pickle": {
            "path": "data/elevation_area_storage_relationships/2019_elevation_area_capacity.pkl",
            "kind": "pickle",
            "description": "Piecewise linear E–A–S interpolators based on 2019 data",
        },
        "niip_demand_spline_pkl": {
            "path": "data/niip/niip_demand_spline.pkl",
            "kind": "pickle",
            "description": "Piecewise linear E–A–S interpolators based on 2019 data",
        },
        "spr_oi_params_json": {
            "path": "data/spring_peak_release/spr_oi_params.json",
            "kind": "json",
            "description": "Sigmoid boundary + OI mapping for SPR go/no-go",
        },
        "spr_threshold_oi_params_json": {
            "path": "data/spring_peak_release/spr_threshold_oi_params.json",
            "kind": "json",
            "description": "Threshold-specific SPR opportunity-index surfaces",
        },


    }

    return m.resolve_paths()
