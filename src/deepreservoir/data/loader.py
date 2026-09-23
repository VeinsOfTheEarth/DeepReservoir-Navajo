# loader.py

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import re
import pandas as pd

from .metadata import Metadata, project_metadata
from .context_features import add_operational_context_features
from .storage_datum import apply_storage_datum_mode

# timezone handling
_TZ_OFFSETS = {
    "UTC": 0, "GMT": 0,
    "HST": 10,
    "AKST": 9, "AKDT": 8,
    "PST": 8, "PDT": 7,
    "MST": 7, "MDT": 6,
    "CST": 6, "CDT": 5,
    "EST": 5, "EDT": 4,
}


def _read_usgs_continuous_file_irregular(
    path: Path,
    *,
    time_col_candidates=("datetime", "dateTime", "time"),
    tz_col_candidates=("tz_cd", "tz"),
    prefer_param="00060",
    drop_dupes="last",
    coerce_numeric=True,
) -> pd.DataFrame:
    """USGS tab file (row-wise time zone). Return DataFrame indexed by UTC with a single 'value' column."""
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    if df.empty:
        raise ValueError(f"{path} contained no rows after comments.")
    # Drop the "5s 15s ..." width-spec row if present
    if "agency_cd" in df.columns and df["agency_cd"].str.contains(r"\ds", na=False).any():
        df = df[~df["agency_cd"].str.fullmatch(r"\ds", na=False)]

    tcol = next((c for c in time_col_candidates if c in df.columns), None)
    if tcol is None:
        raise KeyError(f"{path}: datetime column not found.")

    zcol = next((c for c in tz_col_candidates if c in df.columns), None)

    patt = re.compile(rf".*_(?:{re.escape(prefer_param)})$")
    vcol = (next((c for c in df.columns if patt.fullmatch(c)), None)
            or next((c for c in df.columns if c.endswith("_00060")), None)
            or next((c for c in df.columns if c.lower() == "value"), None))
    if vcol is None:
        raise KeyError(f"{path}: flow/value column not found (*_00060 or 'value').")

    t = pd.to_datetime(df[tcol], errors="coerce")
    if zcol in df.columns:
        tz = df[zcol].astype(str).str.upper().map(_TZ_OFFSETS).fillna(0).astype(int)
        t = t + pd.to_timedelta(tz, unit="h")  # local -> UTC

    v = pd.to_numeric(df[vcol], errors="coerce") if coerce_numeric else df[vcol]
    s = pd.Series(v.values, index=t).dropna().sort_index()
    s = s[~s.index.duplicated(keep=drop_dupes)]
    return s.to_frame(name="value")


# --- Metadata-driven loader (no normalization; minimal checks) -------------------------------
@dataclass
class NavajoData:
    meta: Metadata = field(default_factory=project_metadata)  # fresh per instance
    tables: dict = field(default_factory=dict)               # no __post_init__ needed

    # Data required to build the dataframe needed by the model
    _MODEL_REQUIRED = {
        "reservoir": ("release_cfs", "storage_af", "elev_ft"),
        "inflow": ("inflow_cfs",),
        "evaporation": ("evap_af",),
        "sj_farmington": ("sj_farmington_q_cfs",),
        "sj_bluff": ("sj_bluff_q_cfs",),
        "animas_farmington": ("animas_farmington_q_cfs",),
        "swe_animas" : ("animas_swe_m",)
    }

    _MODEL_OPTIONAL = {
        "swe_upper_sj": ("uppersj_swe_m",),
    }

    def _contiguous_bounds(self, idx: pd.DatetimeIndex):
        """(start, end, block_index) for the longest strictly daily run in idx."""
        if len(idx) == 0:
            return pd.NaT, pd.NaT, idx
        idx = pd.DatetimeIndex(idx).sort_values()
        diffs = idx.to_series().diff().fillna(pd.Timedelta(days=1))
        groups = (diffs != pd.Timedelta(days=1)).cumsum()
        best = groups.value_counts().idxmax()
        block = groups[groups == best].index
        return block.min(), block.max(), block

    def _clip_model_contiguous(self, model_df: pd.DataFrame, used: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Clip model_df to its longest contiguous daily block and explain the limiters."""
        if model_df.empty:
            return model_df
        s, e, block = self._contiguous_bounds(model_df.index)
        clipped = model_df.loc[block]
        kept, dropped = len(clipped), len(model_df) - len(clipped)

        # who prevents extending by one day?
        s_prev = s - pd.Timedelta(days=1)
        e_next = e + pd.Timedelta(days=1)
        start_limiters = [n for n, df in used.items() if s_prev not in df.index]
        end_limiters   = [n for n, df in used.items() if e_next not in df.index]

        print(
            f"[model_data] clipped to contiguous block {s.date()}–{e.date()} "
            f"(kept {kept} days, dropped {dropped}). "
            f"Limited by start: {start_limiters}; end: {end_limiters}."
        )
        return clipped

    def _clip_negative_numeric(self, df: pd.DataFrame, *, name: str) -> pd.DataFrame:
        """
        Clip negative numeric values to zero.

        Most of our series (flow, storage, SWE, etc.) should be non-negative.
        To avoid bad sentinel values (e.g. -999999) wrecking plots, clamp any
        negative numeric entries to 0 and log a short message.
        """
        numeric_cols = df.select_dtypes(include="number").columns
        if not len(numeric_cols):
            return df

        neg_mask = (df[numeric_cols] < 0)
        if not neg_mask.any().any():
            return df

        counts = neg_mask.sum()
        total = int(counts.sum())
        cols_with_negs = list(counts[counts > 0].index)

        print(f"[{name}] clipped {total} negative values to 0 in columns {cols_with_negs}.")

        df = df.copy()
        df[numeric_cols] = df[numeric_cols].clip(lower=0)
        return df


    # Core executor: do exactly what the spec says, nothing more.
    def _load_from_spec(self, spec: Dict[str, Any], *, name: str) -> pd.DataFrame:
        path: Path = spec["path"]
        reader: str = spec["reader"]
        read_kwargs: Dict[str, Any] = spec.get("read_kwargs", {})
        index_cfg: Dict[str, Optional[str]] = spec.get("index", {"name": None, "format": None})
        idx_name = index_cfg.get("name")
        idx_fmt = index_cfg.get("format")
        columns_map: Dict[str, str] = spec.get("columns", {})
        required = tuple(spec.get("required", ()))
        dup_policy = spec.get("duplicates", "last")

        # 1) read
        if reader == "csv":
            df = pd.read_csv(path, **read_kwargs)  # <— pass through explicit kwargs
        elif reader == "parquet":
            df = pd.read_parquet(path)
        elif reader == "usgs_irregular":
            df = _read_usgs_continuous_file_irregular(path)
        else:
            raise ValueError(f"{name}: unknown reader '{reader}'")

        # 2) index (explicit only)
        idx_cfg  = spec.get("index", {}) or {}
        idx_name = idx_cfg.get("name")
        idx_fmt  = idx_cfg.get("format")

        if idx_name is not None:
            # Case-insensitive match to the actual column
            if idx_name not in df.columns:
                cols_lower = {c.lower(): c for c in df.columns}
                if idx_name.lower() in cols_lower:
                    idx_name = cols_lower[idx_name.lower()]
                else:
                    raise KeyError(f"{name}: index column '{idx_name}' not found.")

            raw = df[idx_name]

            # 1) Try the explicit format if provided
            dt = pd.to_datetime(raw, format=idx_fmt, errors="coerce") if idx_fmt else pd.to_datetime(raw, errors="coerce")

            # 2) If that failed entirely, retry with a tolerant parse (handles mixed formats)
            if dt.isna().all():
                dt = pd.to_datetime(raw, errors="coerce", dayfirst=True)

            # 3) Still nothing? Show a sample to debug and stop
            if dt.isna().all():
                sample = raw.astype(str).head(3).tolist()
                raise ValueError(
                    f"{name}: failed to parse any timestamps in '{idx_name}'. "
                    f"format={idx_fmt!r}. Sample values: {sample}"
                )

            # 4) Two-digit year correction (typical %y spillover → 2060s)
            # Only apply when we see both far-future and historical years.
            if (dt.dt.year >= 2060).any() and (dt.dt.year <= 1999).any():
                mask = dt.notna() & (dt.dt.year >= 2060) & (dt.dt.year <= 2099)
                dt.loc[mask] = dt.loc[mask] - pd.DateOffset(years=100)

            # 5) Set index
            df = df.set_index(dt).drop(columns=[idx_name])

        elif isinstance(df.index, pd.DatetimeIndex):
            # Already datetime-indexed; accept as-is
            pass

        # 3) rename exactly as declared (no inference)
        if columns_map:
            df = df.rename(columns=columns_map)

        # 4) keep only canonical columns we declared (quietly ignore extras)
        keep = list(columns_map.values()) if columns_map else list(df.columns)
        df = df[keep]

        # 5) required columns check (fail fast)
        if required:
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise KeyError(f"{name}: missing required columns after rename: {missing}")

        # 6) duplicates policy on index
        if dup_policy in ("first", "last"):
            df = df[~df.index.duplicated(keep=dup_policy)]
        elif dup_policy == "raise" and df.index.has_duplicates:
            raise ValueError(f"{name}: duplicate timestamps present.")
        # always sorted for sanity
        df = df.sort_index()

        # 7) clamp negative numeric values to zero
        df = self._clip_negative_numeric(df, name=name)

        self.tables[name] = df
        return df

    # --------- public API (thin wrappers over metadata) ------------------------------------
    def load_daily(self, name: str) -> pd.DataFrame:
        spec = self.meta.daily_series[name]
        return self._load_from_spec(spec, name=name)

    def load_continuous(self, name: str) -> pd.DataFrame:
        spec = self.meta.continuous_series[name]
        return self._load_from_spec(spec, name=name)

    def load_table(self, name: str) -> pd.DataFrame:
        spec = self.meta.tables[name]
        # tables usually aren't time-indexed; just do what the spec says
        return self._load_from_spec(spec, name=name)

    def load_all(
        self,
        *,
        include_cont_streamflow: bool = False,
        model_data: bool = False,
        storage_datum_mode: str = "reported",
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}

        # daily (reservoir, inflow, evaporation, swe, dsf_*, etc.)
        for name, spec in self.meta.daily_series.items():
            out[name] = self._load_from_spec(spec, name=name)

        # optional continuous (csf_* etc.), native time grid
        if include_cont_streamflow:
            for name, spec in self.meta.continuous_series.items():
                out[name] = self._load_from_spec(spec, name=name)

        # one inner-joined daily frame for the model (no resampling, no fills)
        if model_data:
            model_df = self.build_model_daily(
                storage_datum_mode=storage_datum_mode,
            )
            out["model_data"] = model_df
            out["model_data_norm"] = self.tables["model_data_norm"]
            out["model_norm_stats"] = self.tables["model_norm_stats"]

        return out

    def build_model_daily(
        self,
        *,
        require_optional: bool = False,
        storage_datum_mode: str = "reported",
    ) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        used: dict[str, pd.DataFrame] = {}

        # required datasets
        for name, cols in self._MODEL_REQUIRED.items():
            df = self.tables.get(name)
            if df is None:
                df = self._load_from_spec(self.meta.daily_series[name], name=name)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"{name}: missing required columns {missing}.")
            df_req = df[list(cols)]
            pieces.append(df_req)
            used[name] = df_req

        # optional datasets
        for name, cols in self._MODEL_OPTIONAL.items():
            if name in self.meta.daily_series:
                df = self.tables.get(name)
                if df is None:
                    df = self._load_from_spec(self.meta.daily_series[name], name=name)
                missing = [c for c in cols if c not in df.columns]
                if missing:
                    if require_optional:
                        raise KeyError(f"{name}: missing optional columns {missing}.")
                    continue
                df_opt = df[list(cols)]
                pieces.append(df_opt)
                used[name] = df_opt
            elif require_optional:
                raise KeyError(f"Optional dataset '{name}' not in metadata (require_optional=True).")

        model_df = pd.concat(pieces, axis=1, join="inner").sort_index()
        if model_df.isna().any(axis=None):
            model_df = model_df.dropna(axis=0, how="any")

        # clip only the JOINED frame; report limiting datasets
        model_df = self._clip_model_contiguous(model_df, used)
        # NOTE: SJ @ FARMINGTON is inexplicably missing 1.5 years of data (2020-2022).

        # --- add day-of-year (raw space) ---
        model_df["doy"] = model_df.index.dayofyear
        model_df = add_operational_context_features(model_df)
        model_df, storage_datum_meta = apply_storage_datum_mode(
            model_df,
            mode=storage_datum_mode,
        )

        # === NORMALIZATION ===
        # if you ever want train-only stats, slice model_df here
        train_df = model_df

        # all numeric columns (we'll normalize all of them)
        num_cols = train_df.select_dtypes(include="number").columns

        # standard normal stats for all numeric cols
        means = train_df[num_cols].mean()
        stds = train_df[num_cols].std()

        # special case for day-of-year: use (doy - 0) / 366
        if "doy" in num_cols:
            means["doy"] = 0.0
            stds["doy"] = 366.0

        # avoid division by zero in any weird constant columns
        stds_safe = stds.replace(0, 1.0)

        # normalized copy
        model_df_norm = model_df.copy()
        model_df_norm[num_cols] = (model_df_norm[num_cols] - means) / stds_safe

        # store stats as a DataFrame for reuse in env/testing
        norm_stats = pd.DataFrame({"mean": means, "std": stds_safe})

        self.tables["model_data"] = model_df
        self.tables["model_data_norm"] = model_df_norm
        self.tables["model_norm_stats"] = norm_stats
        self.tables["storage_datum_meta"] = storage_datum_meta

        return model_df
