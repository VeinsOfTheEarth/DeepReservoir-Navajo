from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .swe_helpers import (
    SigmoidRuleParams,
    sigmoid_boundary_y,
    opportunity_index_from_margin,
    assemble_wy_metrics,
    prespring_storage_by_wy,
)


def threshold_oi_label(threshold_cfs: float, duration_days: int) -> str:
    return f"{int(threshold_cfs)}cfs_{int(duration_days)}d"


@dataclass
class OIParams:
    boundary: SigmoidRuleParams
    m0: float = 40.0
    beta: float = 0.916290731874155  # ~beta_from_target(omega=0.75, oi_at_pos_m0=0.90)
    omega_on_line: float = 0.75
    storage_method: str = "feb_mean"            # Feb mean storage
    swe_metric: str = "SWE_peak_by_Mar1_mm"     # or "SWE_Feb_max_mm"

    @staticmethod
    def load(path: Path | str) -> "OIParams":
        with open(path, "r") as f:
            cfg = json.load(f)
        b = cfg["boundary"]
        return OIParams(
            boundary=SigmoidRuleParams(**b),
            m0=cfg.get("m0", 40.0),
            beta=cfg.get("beta", 0.916290731874155),
            omega_on_line=cfg.get("omega_on_line", 0.75),
            storage_method=cfg.get("storage_method", "feb_mean"),
            swe_metric=cfg.get("swe_metric", "SWE_peak_by_Mar1_mm"),
        )


@dataclass(frozen=True)
class ThresholdOISpec:
    threshold_cfs: float
    duration_days: int
    target_frequency: float
    boundary: SigmoidRuleParams
    margin_shift_mm: float = 0.0
    label: str = ""

    @property
    def key(self) -> str:
        return threshold_oi_label(self.threshold_cfs, self.duration_days)


@dataclass
class ThresholdOIParams:
    thresholds: tuple[ThresholdOISpec, ...]
    m0: float = 40.0
    beta: float = 0.916290731874155
    omega_on_line: float = 0.75
    storage_method: str = "feb_mean"
    swe_metric: str = "SWE_peak_by_Mar1_mm"
    calibration_wy_min: int = 2000

    @staticmethod
    def load(path: Path | str) -> "ThresholdOIParams":
        with open(path, "r") as f:
            cfg = json.load(f)

        common_boundary_cfg = cfg.get("boundary")
        threshold_cfg = cfg.get("thresholds", {})
        if isinstance(threshold_cfg, dict):
            items = threshold_cfg.items()
        elif isinstance(threshold_cfg, list):
            items = ((str(i), v) for i, v in enumerate(threshold_cfg))
        else:
            raise TypeError("thresholds must be a dict or list")

        specs: list[ThresholdOISpec] = []
        for key, raw_spec in items:
            if not isinstance(raw_spec, dict):
                raise TypeError(f"threshold spec {key!r} must be an object")
            boundary_cfg = raw_spec.get("boundary", common_boundary_cfg)
            if boundary_cfg is None:
                raise KeyError(f"threshold spec {key!r} is missing boundary params")
            specs.append(
                ThresholdOISpec(
                    threshold_cfs=float(raw_spec["threshold_cfs"]),
                    duration_days=int(raw_spec["duration_days"]),
                    target_frequency=float(raw_spec.get("target_frequency", np.nan)),
                    boundary=SigmoidRuleParams(**boundary_cfg),
                    margin_shift_mm=float(
                        raw_spec.get(
                            "margin_shift_mm",
                            raw_spec.get("margin_cut_mm", 0.0),
                        )
                    ),
                    label=str(raw_spec.get("label", "")),
                )
            )

        if not specs:
            raise ValueError(f"No threshold OI specs found in {path}")

        return ThresholdOIParams(
            thresholds=tuple(specs),
            m0=float(cfg.get("m0", 40.0)),
            beta=float(cfg.get("beta", 0.916290731874155)),
            omega_on_line=float(cfg.get("omega_on_line", 0.75)),
            storage_method=str(cfg.get("storage_method", "feb_mean")),
            swe_metric=str(cfg.get("swe_metric", "SWE_peak_by_Mar1_mm")),
            calibration_wy_min=int(cfg.get("calibration_wy_min", 2000)),
        )

    def by_key(self) -> dict[str, ThresholdOISpec]:
        return {spec.key: spec for spec in self.thresholds}


def _storage_swe_by_wy(
    model_df: pd.DataFrame,
    *,
    storage_method: str,
    swe_metric: str,
) -> pd.DataFrame:
    # x = pre-spring storage (e.g. Feb mean)
    x = prespring_storage_by_wy(model_df, method=storage_method)

    # y = SWE metric by WY (computed from daily animas SWE)
    swe_daily = model_df["animas_swe_m"]
    # an_daily not needed for this y metric; pass a placeholder series with same index
    dummy_q = model_df.get("animas_farmington_q_cfs", pd.Series(index=model_df.index, dtype=float))
    wy = assemble_wy_metrics(swe_daily, dummy_q)
    if swe_metric not in wy.columns:
        raise KeyError(f"Unknown SWE metric {swe_metric!r}. Have: {list(wy.columns)}")
    y = wy[swe_metric]

    df = pd.concat([x.rename("storage_x"), y.rename("swe_y")], axis=1).dropna()
    df.index.name = "WY"
    return df


def precompute_oi_by_wy(model_df: pd.DataFrame, oi: OIParams) -> pd.DataFrame:
    """
    Returns DataFrame indexed by water year with:
        ['storage_x', 'swe_y', 'margin_mm', 'oi', 'go']
    """
    df = _storage_swe_by_wy(
        model_df,
        storage_method=oi.storage_method,
        swe_metric=oi.swe_metric,
    )

    margin = df["swe_y"].values - sigmoid_boundary_y(df["storage_x"].values, oi.boundary)
    oi_vals, _ = opportunity_index_from_margin(margin, m0=oi.m0, beta=oi.beta, omega_on_line=oi.omega_on_line)

    df["margin_mm"] = margin
    df["oi"] = oi_vals
    df["go"] = df["oi"] >= oi.omega_on_line
    return df


def precompute_threshold_oi_by_wy(
    model_df: pd.DataFrame,
    oi: ThresholdOIParams,
) -> pd.DataFrame:
    """
    Returns WY-indexed threshold-specific SPR opportunity indices.

    For each configured threshold ``KEY`` the returned frame includes:
        - ``margin_KEY_mm``
        - ``boundary_KEY_mm``
        - ``oi_KEY``
        - ``go_KEY``
    """
    df = _storage_swe_by_wy(
        model_df,
        storage_method=oi.storage_method,
        swe_metric=oi.swe_metric,
    )
    x = df["storage_x"].to_numpy(dtype=float)
    y = df["swe_y"].to_numpy(dtype=float)

    for spec in oi.thresholds:
        key = spec.key
        boundary_y = sigmoid_boundary_y(x, spec.boundary) + float(spec.margin_shift_mm)
        margin = y - boundary_y
        oi_vals, _ = opportunity_index_from_margin(
            margin,
            m0=oi.m0,
            beta=oi.beta,
            omega_on_line=oi.omega_on_line,
        )
        df[f"boundary_{key}_mm"] = boundary_y
        df[f"margin_{key}_mm"] = margin
        df[f"oi_{key}"] = oi_vals
        df[f"go_{key}"] = df[f"oi_{key}"] >= oi.omega_on_line

    return df
