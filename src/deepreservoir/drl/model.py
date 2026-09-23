"""
High-level training and evaluation utilities for DeepReservoir DRL.

- loads data via NavajoData
- splits into train / test
- builds reward (from registry)
- builds Gymnasium envs
- trains & evaluates an SB3 agent
- provides helpers to roll out on the test period and compute metrics

Paper configuration:
- Training uses random-window episodes (episode_length_train=3600 by default)
- Testing rolls out the full test period deterministically
- Multi-action PPO: one agent, four continuous actions (handled inside environs.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import json
from datetime import datetime
import platform
import time

import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from deepreservoir.data import loader
from deepreservoir.data.context_features import (
    OPERATIONAL_CONTEXT_COLUMNS,
    add_operational_context_features,
)
from deepreservoir.data.storage_datum import normalize_storage_datum_mode
from deepreservoir.drl import helpers
from deepreservoir.drl import rewards as drl_rewards
from deepreservoir.drl import metrics as drl_metrics
from deepreservoir.drl.niip_targets import normalize_niip_fallback_mode
from deepreservoir.drl.policies import SplitActionHeadsActorCriticPolicy
from deepreservoir.drl.environs import (
    ACTION_MODE_CHOICES,
    ACTION_SCALING_CHOICES,
    NavajoReservoirEnv,
    OBS_CONTEXT_CHOICES,
    SPR_ADVICE_MODE_CHOICES,
    SELECTED_OBS_CONTEXT,
    obs_columns_for_context,
    normalize_action_mode,
    normalize_action_scaling,
    normalize_decision_hydrology_timing,
    normalize_spr_advice_mode,
)
from deepreservoir.define_env.hydropower_model import navajo_power_generation_scalar


REWARD_BALANCING_CHOICES = drl_rewards.REWARD_BALANCING_CHOICES
POLICY_TYPE_CHOICES: tuple[str, ...] = (
    "split_action_heads_sj_no_spr",
)
TRAIN_HYDROLOGY_TRANSFORM_CHOICES: tuple[str, ...] = (
    "none",
    "match_holdout_annual_mean",
)
STORAGE_NORMALIZATION_CHOICES: tuple[str, ...] = (
    "full_record",
    "train_window",
)
STORAGE_NORMALIZATION_FILENAME = "observation_normalization.json"
STORAGE_NORMALIZATION_SCHEMA_VERSION = 1
HYDROLOGY_TRANSFORM_REFERENCE_START = "2014-01-01"
HYDROLOGY_TRANSFORM_REFERENCE_END = "2024-08-17"
CFS_TO_AF_PER_DAY = 86400.0 / 43560.0

POLICY_NET_ARCH_ALIASES: dict[str, dict[str, list[int]] | None] = {
    "default": None,
    "sb3_default": None,
    "default64": {"pi": [64, 64], "vf": [64, 64]},
    "wide128": {"pi": [128, 128], "vf": [128, 128]},
    "wide256": {"pi": [256, 256], "vf": [256, 256]},
    "deep128": {"pi": [128, 128, 128], "vf": [128, 128, 128]},
    "critic256": {"pi": [128, 128], "vf": [256, 256]},
}


def _parse_layer_list(text: str) -> list[int]:
    vals: list[int] = []
    for token in str(text).replace("x", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Network layer sizes must be positive, got {value}.")
        vals.append(value)
    if not vals:
        raise ValueError(f"Network layer list is empty: {text!r}")
    return vals


def policy_net_arch_to_policy_kwargs(
    policy_net_arch: str | dict[str, list[int]] | None,
) -> dict[str, Any] | None:
    """Convert a public architecture spec to SB3 ``policy_kwargs``.

    Supported string forms:
      - ``default`` / ``sb3_default``: no explicit override
      - aliases: ``default64``, ``wide128``, ``wide256``, ``deep128``, ``critic256``
      - shared actor/critic layers: ``128,128`` or ``128x128``
      - explicit branches: ``pi:128,128;vf:256,256``
    """
    if policy_net_arch is None:
        return None

    if isinstance(policy_net_arch, dict):
        arch = {
            "pi": [int(v) for v in policy_net_arch.get("pi", [])],
            "vf": [int(v) for v in policy_net_arch.get("vf", [])],
        }
    else:
        spec = str(policy_net_arch).strip().lower()
        if not spec:
            return None
        if spec in POLICY_NET_ARCH_ALIASES:
            alias = POLICY_NET_ARCH_ALIASES[spec]
            if alias is None:
                return None
            arch = {"pi": list(alias["pi"]), "vf": list(alias["vf"])}
        elif ";" in spec or "pi:" in spec or "vf:" in spec:
            parts: dict[str, list[int]] = {}
            for part in spec.split(";"):
                if not part.strip():
                    continue
                if ":" not in part:
                    raise ValueError(
                        f"Invalid policy_net_arch part {part!r}; expected pi:... or vf:..."
                    )
                key, value = part.split(":", 1)
                key = key.strip()
                if key not in {"pi", "vf"}:
                    raise ValueError(
                        f"Invalid policy_net_arch branch {key!r}; expected 'pi' or 'vf'."
                    )
                parts[key] = _parse_layer_list(value)
            if "pi" not in parts or "vf" not in parts:
                raise ValueError(
                    f"policy_net_arch {policy_net_arch!r} must provide both pi and vf branches."
                )
            arch = {"pi": parts["pi"], "vf": parts["vf"]}
        else:
            layers = _parse_layer_list(spec)
            arch = {"pi": list(layers), "vf": list(layers)}

    if not arch["pi"] or not arch["vf"]:
        raise ValueError(f"policy_net_arch must contain non-empty pi and vf branches: {policy_net_arch!r}")
    if any(int(v) <= 0 for v in arch["pi"] + arch["vf"]):
        raise ValueError(f"policy_net_arch layer sizes must be positive: {policy_net_arch!r}")
    return {"net_arch": arch}


def normalize_policy_net_arch(policy_net_arch: str | dict[str, list[int]] | None) -> str | None:
    if policy_net_arch is None:
        return None
    if isinstance(policy_net_arch, str):
        text = policy_net_arch.strip()
        if not text:
            return None
        _ = policy_net_arch_to_policy_kwargs(text)
        return text
    kwargs = policy_net_arch_to_policy_kwargs(policy_net_arch)
    if kwargs is None:
        return None
    arch = kwargs["net_arch"]
    return "pi:" + ",".join(str(v) for v in arch["pi"]) + ";vf:" + ",".join(str(v) for v in arch["vf"])


def normalize_policy_type(policy_type: str | None) -> str:
    text = str(policy_type or "split_action_heads_sj_no_spr").strip().lower()
    aliases = {
        "": "mlp",
        "default": "mlp",
        "mlppolicy": "mlp",
        "mlp_policy": "mlp",
        "split": "split_action_heads",
        "split_heads": "split_action_heads",
        "split_action": "split_action_heads",
        "split_action_head": "split_action_heads",
        "split_action_heads": "split_action_heads",
        "split_sj_no_spr": "split_action_heads_sj_no_spr",
        "split_action_heads_sj_no_spr": "split_action_heads_sj_no_spr",
        "split_action_heads_no_spr_sj": "split_action_heads_sj_no_spr",
        "split_action_heads_sj_hide_spr": "split_action_heads_sj_no_spr",
    }
    normalized = aliases.get(text, text)
    if normalized not in POLICY_TYPE_CHOICES:
        raise ValueError(
            f"Unknown policy_type {policy_type!r}. Choose from: {sorted(POLICY_TYPE_CHOICES)}"
        )
    return normalized


def normalize_train_hydrology_transform(value: str | None) -> str:
    text = str(value or "none").strip().lower()
    aliases = {
        "": "none",
        "off": "none",
        "false": "none",
        "no": "none",
        "annual": "match_holdout_annual_mean",
        "annual_holdout": "match_holdout_annual_mean",
        "annual_holdout_match": "match_holdout_annual_mean",
        "monthly": "match_holdout_monthly_mean",
        "monthly_holdout": "match_holdout_monthly_mean",
        "monthly_holdout_match": "match_holdout_monthly_mean",
    }
    normalized = aliases.get(text, text)
    if normalized not in TRAIN_HYDROLOGY_TRANSFORM_CHOICES:
        raise ValueError(
            f"Unknown train_hydrology_transform {value!r}. "
            f"Choose from: {sorted(TRAIN_HYDROLOGY_TRANSFORM_CHOICES)}"
        )
    return normalized


def normalize_storage_normalization(value: str | None) -> str:
    """Resolve how the dynamic storage observation is standardized.

    ``full_record`` preserves the earlier full-record preprocessing
    checkpoint. ``train_window`` fits storage mean and sample standard deviation
    only on the configured training dates; those values are then stored beside
    the trained model for evaluation.
    """
    text = str(value or "full_record").strip().lower()
    aliases = {
        "": "full_record",
        "legacy": "full_record",
        "legacy_full_record": "full_record",
        "train": "train_window",
        "training": "train_window",
        "training_window": "train_window",
    }
    normalized = aliases.get(text, text)
    if normalized not in STORAGE_NORMALIZATION_CHOICES:
        raise ValueError(
            f"Unknown storage_normalization {value!r}. "
            f"Choose from: {sorted(STORAGE_NORMALIZATION_CHOICES)}"
        )
    return normalized


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_all_model_data(
    *,
    storage_datum_mode: str = "reported",
) -> Dict[str, Any]:
    """Load model data under the requested storage-datum convention."""
    resolved_storage_datum = normalize_storage_datum_mode(storage_datum_mode)
    nav_data = loader.NavajoData()
    alldata = nav_data.load_all(
        include_cont_streamflow=False,
        model_data=True,
        storage_datum_mode=resolved_storage_datum,
    )
    raw = alldata["model_data"].copy()

    # Archuleta is needed for the historical flood benchmark, but it has a few
    # missing dates. Left-join it after construction of the contiguous model
    # frame so those gaps do not truncate the training/evaluation record.
    archuleta = alldata.get("sj_archuleta")
    if archuleta is not None and "sj_archuleta_q_cfs" in archuleta.columns:
        raw = raw.join(archuleta[["sj_archuleta_q_cfs"]], how="left")

    return {
        "raw": raw,
        "norm": alldata["model_data_norm"],
        "norm_stats": alldata["model_norm_stats"],
        "storage_datum_meta": dict(nav_data.tables["storage_datum_meta"]),
    }


def _previous_calendar_hydrology_row(
    data_raw: pd.DataFrame,
    *,
    first_date: pd.Timestamp,
) -> pd.Series | None:
    """Return the row immediately before a sliced rollout, when available."""
    first = pd.Timestamp(first_date).normalize()
    previous = first - pd.Timedelta(days=1)
    if previous not in data_raw.index:
        return None
    return data_raw.loc[previous].copy()


def _date_for_water_year_month_day(water_year: int, month: int, day: int) -> pd.Timestamp:
    year = int(water_year) - 1 if int(month) >= 10 else int(water_year)
    return pd.Timestamp(year=year, month=int(month), day=int(day))


def _slice_training_window(data_raw, data_norm, *, train_start, train_end):
    """Slice the contiguous training period used by the paper."""
    train_raw, window = helpers.slice_by_window(
        data_raw, start_token=train_start, end_token=train_end, label="train_window")
    return {"train_raw": train_raw, "train_norm": data_norm.loc[train_raw.index]}, {
        "train": {"train_start": train_start, "train_end": train_end, "resolved": {
            "start": str(window.start.date()), "end": str(window.end.date()),
            "start_token": window.start_token, "end_token": window.end_token,
            "n_days": int(window.n_days),
        }}
    }


def _storage_normalization_for_training(
    *,
    train_raw: pd.DataFrame,
    all_raw: pd.DataFrame,
    full_record_norm_stats: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return normalization stats and provenance for the storage observation."""
    normalized_mode = normalize_storage_normalization(mode)
    if "storage_af" not in train_raw.columns or "storage_af" not in all_raw.columns:
        raise KeyError("storage_af is required to fit storage observation normalization.")
    if "storage_af" not in full_record_norm_stats.index:
        raise KeyError("Full-record normalization statistics are missing storage_af.")

    if normalized_mode == "train_window":
        fit_frame = train_raw
        values = pd.to_numeric(fit_frame["storage_af"], errors="coerce").to_numpy(
            dtype=float
        )
        if values.size < 2 or not bool(np.isfinite(values).all()):
            raise ValueError(
                "Training-window storage normalization requires at least two "
                "finite storage_af observations."
            )
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        source = "training_window"
    else:
        fit_frame = all_raw
        mean = float(full_record_norm_stats.loc["storage_af", "mean"])
        std = float(full_record_norm_stats.loc["storage_af", "std"])
        source = "loader_full_record"

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError(
            "Storage normalization requires finite mean and positive standard deviation; "
            f"received mean={mean!r}, std={std!r}."
        )

    resolved = full_record_norm_stats.copy()
    resolved.loc["storage_af", "mean"] = mean
    resolved.loc["storage_af", "std"] = std
    metadata: dict[str, object] = {
        "schema_version": STORAGE_NORMALIZATION_SCHEMA_VERSION,
        "mode": normalized_mode,
        "column": "storage_af",
        "source": source,
        "mean": mean,
        "std": std,
        "ddof": 1,
        "n_observations": int(len(fit_frame)),
        "fit_window": {
            "start": str(pd.Timestamp(fit_frame.index.min()).date()),
            "end": str(pd.Timestamp(fit_frame.index.max()).date()),
        },
    }
    return resolved, metadata


def _apply_storage_normalization_metadata(
    norm_stats: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    """Apply frozen storage mean/std metadata to a copy of normalization stats."""
    _, mean, std = _validate_storage_normalization_metadata(metadata)
    if "storage_af" not in norm_stats.index:
        raise KeyError("Normalization statistics are missing storage_af.")
    out = norm_stats.copy()
    out.loc["storage_af", "mean"] = mean
    out.loc["storage_af", "std"] = std
    return out


def _validate_storage_normalization_metadata(
    metadata: dict[str, object],
) -> tuple[str, float, float]:
    """Validate a frozen storage-normalization sidecar.

    Returning the normalized mode and numeric values keeps every loading path on
    the same schema and value checks.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Storage normalization artifact must contain a JSON object.")

    schema_version = metadata.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != STORAGE_NORMALIZATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "Storage normalization artifact has a missing or unsupported "
            f"schema_version {schema_version!r}; supported version is "
            f"{STORAGE_NORMALIZATION_SCHEMA_VERSION}."
        )
    if metadata.get("column") != "storage_af":
        raise ValueError("Storage normalization artifact must describe storage_af.")
    if "mode" not in metadata:
        raise ValueError("Storage normalization artifact is missing mode.")
    artifact_mode = normalize_storage_normalization(str(metadata["mode"]))
    try:
        mean = float(metadata["mean"])
        std = float(metadata["std"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Storage normalization artifact must contain numeric mean and std."
        ) from exc
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError(
            "Storage normalization artifact contains invalid statistics: "
            f"mean={mean!r}, std={std!r}."
        )
    return artifact_mode, mean, std


def _water_year_for_index(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
        (index.year + (index.month >= 10)).astype(int),
        index=index,
        name="water_year",
    )


def _complete_water_years(index: pd.DatetimeIndex) -> set[int]:
    if index.empty:
        return set()
    out: set[int] = set()
    wy_by_day = _water_year_for_index(index)
    for wy in sorted({int(v) for v in wy_by_day}):
        start = pd.Timestamp(year=wy - 1, month=10, day=1)
        end = pd.Timestamp(year=wy, month=9, day=30)
        if index.min() <= start and index.max() >= end:
            mask = wy_by_day == wy
            expected = pd.date_range(start, end, freq="D")
            actual = index[mask.to_numpy(dtype=bool)]
            if len(actual) == len(expected) and bool(expected.isin(actual).all()):
                out.add(int(wy))
    return out


def _mean_complete_water_year_total(
    df: pd.DataFrame,
    column: str,
    *,
    cfs_to_af: bool,
) -> float:
    if column not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
        return float("nan")
    complete_wys = _complete_water_years(df.index)
    if not complete_wys:
        return float("nan")
    wy_by_day = _water_year_for_index(df.index)
    values = pd.to_numeric(df[column], errors="coerce").astype(float)
    if cfs_to_af:
        values = values * CFS_TO_AF_PER_DAY
    totals: list[float] = []
    for wy in sorted(complete_wys):
        year_values = values.loc[wy_by_day == wy]
        if not bool(np.isfinite(year_values.to_numpy(dtype=float)).all()):
            continue
        total = float(year_values.sum())
        if np.isfinite(total):
            totals.append(total)
    if not totals:
        return float("nan")
    return float(np.mean(totals))


def _safe_scale_factor(reference_value: float, train_value: float) -> float:
    if not np.isfinite(reference_value) or not np.isfinite(train_value) or train_value <= 0.0:
        return 1.0
    return float(np.clip(reference_value / train_value, 0.05, 20.0))


def _renormalize_changed_columns(
    raw: pd.DataFrame,
    norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    out = norm.copy()
    for col in columns:
        if col not in raw.columns or col not in out.columns or col not in norm_stats.index:
            continue
        mean = float(norm_stats.loc[col, "mean"])
        std = float(norm_stats.loc[col, "std"])
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        out[col] = (pd.to_numeric(raw[col], errors="coerce").astype(float) - mean) / std
    return out


def _apply_train_hydrology_transform(
    *,
    train_raw: pd.DataFrame,
    train_norm: pd.DataFrame,
    all_raw: pd.DataFrame,
    norm_stats: pd.DataFrame,
    transform: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return training data with synthetic hydrology matched to holdout climate.

    Only the training window is changed. Evaluation and historical-comparison data
    remain untouched. The transforms are deliberately simple diagnostics:

    - ``match_holdout_annual_mean``: one scalar per hydrology column so the mean
      complete-water-year total matches the 2014-2024 holdout period.
    - ``match_holdout_monthly_mean``: one scalar per calendar month and column so
      the month-of-year mean daily values match the holdout period.
    """

    transform = normalize_train_hydrology_transform(transform)
    hydrology_transform_columns: list[tuple[str, bool]] = [
        ("inflow_cfs", True),
        ("evap_af", False),
        ("animas_farmington_q_cfs", True),
    ]
    changed_cols = [col for col, _ in hydrology_transform_columns]
    meta: dict[str, object] = {
        "name": transform,
        "reference_start": HYDROLOGY_TRANSFORM_REFERENCE_START,
        "reference_end": HYDROLOGY_TRANSFORM_REFERENCE_END,
        "columns": changed_cols,
    }
    if transform == "none":
        meta["applied"] = False
        return train_raw, train_norm, meta

    if not isinstance(train_raw.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex on train_raw for hydrology transform.")

    reference_raw = all_raw.loc[
        HYDROLOGY_TRANSFORM_REFERENCE_START:HYDROLOGY_TRANSFORM_REFERENCE_END
    ]
    if reference_raw.empty:
        raise ValueError(
            "Hydrology transform reference window is empty: "
            f"{HYDROLOGY_TRANSFORM_REFERENCE_START}..{HYDROLOGY_TRANSFORM_REFERENCE_END}"
        )

    raw_out = train_raw.copy()

    if transform == "match_holdout_annual_mean":
        factors: dict[str, float] = {}
        for col, is_cfs in hydrology_transform_columns:
            train_mean = _mean_complete_water_year_total(
                train_raw,
                col,
                cfs_to_af=is_cfs,
            )
            ref_mean = _mean_complete_water_year_total(
                reference_raw,
                col,
                cfs_to_af=is_cfs,
            )
            factor = _safe_scale_factor(ref_mean, train_mean)
            raw_out[col] = pd.to_numeric(raw_out[col], errors="coerce").astype(float) * factor
            factors[col] = factor
        meta["factors"] = factors
        meta["applied"] = True
    else:
        raise ValueError(f"Unsupported train_hydrology_transform: {transform}")

    # Recompute the recorded SPR diagnostics after changing training hydrology.
    # These fields are not additional observations of the selected policy.
    try:
        raw_out = add_operational_context_features(raw_out)
        changed_cols = list(dict.fromkeys(changed_cols + list(OPERATIONAL_CONTEXT_COLUMNS)))
    except Exception as exc:
        meta["context_refresh_warning"] = repr(exc)

    norm_out = _renormalize_changed_columns(raw_out, train_norm, norm_stats, changed_cols)
    return raw_out, norm_out, meta


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _write_storage_normalization_artifact(
    path: Path,
    metadata: dict[str, object],
) -> None:
    """Write immutable observation-scaling metadata beside a trained model."""
    _validate_storage_normalization_metadata(metadata)
    path = Path(path)
    if path.exists():
        existing = _read_json(path)
        if existing != metadata:
            raise ValueError(
                "Refusing to replace frozen storage normalization metadata at "
                f"{path}. Use a new run directory."
            )
        return
    _write_json(path, metadata)


def _storage_norm_stats_for_evaluation(
    *,
    model_path: Path | str,
    full_record_norm_stats: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Resolve storage scaling for evaluation from the model's frozen sidecar.

    Old model artifacts have no sidecar. They may therefore use only the legacy
    ``full_record`` mode. Corrected ``train_window`` models must carry the
    sidecar so evaluation cannot silently refit on a different data snapshot.
    """
    normalized_mode = normalize_storage_normalization(mode)
    run_dir = infer_run_dir_from_model_path(Path(model_path))
    path = run_dir / STORAGE_NORMALIZATION_FILENAME
    if not path.exists():
        if normalized_mode == "full_record":
            return full_record_norm_stats.copy(), None
        raise FileNotFoundError(
            "Training-window storage normalization requires the frozen artifact "
            f"{path}."
        )

    metadata = _read_json(path)
    artifact_mode, _, _ = _validate_storage_normalization_metadata(metadata)
    if artifact_mode != normalized_mode:
        raise ValueError(
            "Storage normalization mode does not match the model artifact: "
            f"requested {normalized_mode!r}, artifact records {artifact_mode!r}."
        )
    return _apply_storage_normalization_metadata(full_record_norm_stats, metadata), metadata


def _validate_storage_norm_stats_against_artifact(
    *,
    model_path: Path | str,
    norm_stats: pd.DataFrame,
    mode: str,
) -> dict[str, object] | None:
    """Require supplied storage statistics to match a model's frozen sidecar.

    Sidecar-free checkpoints predate this metadata and remain valid only with
    legacy ``full_record`` normalization.
    """
    normalized_mode = normalize_storage_normalization(mode)
    run_dir = infer_run_dir_from_model_path(Path(model_path))
    path = run_dir / STORAGE_NORMALIZATION_FILENAME
    if not path.exists():
        if normalized_mode == "full_record":
            return None
        raise FileNotFoundError(
            "Training-window storage normalization requires the frozen artifact "
            f"{path}."
        )

    metadata = _read_json(path)
    artifact_mode, expected_mean, expected_std = (
        _validate_storage_normalization_metadata(metadata)
    )
    if artifact_mode != normalized_mode:
        raise ValueError(
            "Storage normalization mode does not match the model artifact: "
            f"requested {normalized_mode!r}, artifact records {artifact_mode!r}."
        )
    if "storage_af" not in norm_stats.index:
        raise KeyError("Normalization statistics are missing storage_af.")
    try:
        supplied_mean = float(norm_stats.loc["storage_af", "mean"])
        supplied_std = float(norm_stats.loc["storage_af", "std"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Supplied normalization statistics must contain numeric storage_af "
            "mean and std values."
        ) from exc
    values_match = (
        np.isfinite(supplied_mean)
        and np.isfinite(supplied_std)
        and np.isclose(supplied_mean, expected_mean, rtol=1e-12, atol=1e-9)
        and np.isclose(supplied_std, expected_std, rtol=1e-12, atol=1e-9)
    )
    if not bool(values_match):
        raise ValueError(
            "Supplied storage normalization statistics do not match the model "
            f"artifact at {path}: supplied mean={supplied_mean!r}, "
            f"std={supplied_std!r}; artifact mean={expected_mean!r}, "
            f"std={expected_std!r}."
        )
    return metadata


def infer_run_dir_from_model_path(model_path: Path) -> Path:
    """Return a best-effort "run directory" for a model path.

    Heuristic:
      - If model is in a directory containing run_manifest.json -> that directory
      - Else, check parent directory
      - Else, return model_path.parent
    """
    model_path = Path(model_path)
    cand = model_path.parent
    if (cand / "run_manifest.json").exists():
        return cand
    if (cand.parent / "run_manifest.json").exists():
        return cand.parent
    return cand


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if callable(value):
        return repr(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _stable_baselines3_version() -> str | None:
    try:
        import stable_baselines3 as _sb3  # local import for robustness

        return str(getattr(_sb3, "__version__", None) or "") or None
    except Exception:
        try:
            from importlib.metadata import version

            return str(version("stable_baselines3"))
        except Exception:
            return None


def _torch_version() -> str | None:
    try:
        import torch as _torch  # local import for robustness

        return str(getattr(_torch, "__version__", None) or "") or None
    except Exception:
        try:
            from importlib.metadata import version

            return str(version("torch"))
        except Exception:
            return None


def _callable_name(obj: object) -> str | None:
    if obj is None:
        return None
    name = getattr(obj, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return obj.__class__.__name__


def _module_parameter_counts(module: object | None) -> dict[str, int] | None:
    if module is None or not hasattr(module, "parameters"):
        return None
    try:
        params = list(module.parameters())
    except Exception:
        return None
    return {
        "total": int(sum(p.numel() for p in params)),
        "trainable": int(sum(p.numel() for p in params if getattr(p, "requires_grad", False))),
    }


def _module_repr(module: object | None) -> str | None:
    if module is None:
        return None
    try:
        return str(module)
    except Exception:
        return repr(module)


def _collect_policy_summary(agent: PPO | None) -> dict[str, object]:
    if agent is None or getattr(agent, "policy", None) is None:
        return {}

    policy = agent.policy
    features_extractor = getattr(policy, "features_extractor", None)
    optimizer = getattr(policy, "optimizer", None)

    summary: dict[str, object] = {
        "net_arch": _json_safe(getattr(policy, "net_arch", None)),
        "activation_fn": _callable_name(getattr(policy, "activation_fn", None)),
        "ortho_init": _json_safe(getattr(policy, "ortho_init", None)),
        "features_extractor_class": (
            features_extractor.__class__.__name__ if features_extractor is not None else None
        ),
        "share_features_extractor": _json_safe(
            getattr(policy, "share_features_extractor", None)
        ),
        "normalize_images": _json_safe(getattr(policy, "normalize_images", None)),
        "optimizer_class": (
            optimizer.__class__.__name__ if optimizer is not None else None
        ),
        "parameter_counts": _module_parameter_counts(policy),
    }
    return {k: v for k, v in summary.items() if v is not None}


def _collect_resolved_sb3_config(agent: PPO | None) -> dict[str, object]:
    if agent is None:
        return {}

    policy_class = None
    try:
        if getattr(agent, "policy_class", None) is not None:
            policy_class = getattr(agent.policy_class, "__name__", repr(agent.policy_class))
        elif getattr(agent, "policy", None) is not None:
            policy_class = agent.policy.__class__.__name__
    except Exception:
        policy_class = None

    resolved: dict[str, object] = {
        "stable_baselines3_version": _stable_baselines3_version(),
        "torch_version": _torch_version(),
        "python_version": platform.python_version(),
        "algo_class": agent.__class__.__name__,
        "policy_class": policy_class,
        "policy_kwargs": _json_safe(getattr(agent, "policy_kwargs", None)),
        "device": _json_safe(getattr(agent, "device", None)),
        "policy_summary": _collect_policy_summary(agent),
    }

    for key in [
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "learning_rate",
        "gae_lambda",
        "clip_range",
        "clip_range_vf",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "use_sde",
        "sde_sample_freq",
        "stats_window_size",
    ]:
        if hasattr(agent, key):
            resolved[key] = _json_safe(getattr(agent, key))

    return {k: v for k, v in resolved.items() if v is not None}


def _collect_resolved_config_snapshot(agent: PPO | None) -> dict[str, object]:
    if agent is None or getattr(agent, "policy", None) is None:
        return {}

    policy = agent.policy
    optimizer = getattr(policy, "optimizer", None)

    module_names = [
        "features_extractor",
        "mlp_extractor",
        "action_net",
        "value_net",
        "pi_features_extractor",
        "vf_features_extractor",
    ]
    module_reprs: dict[str, object] = {}
    module_param_counts: dict[str, object] = {}
    for name in module_names:
        module = getattr(policy, name, None)
        if module is not None:
            module_reprs[name] = _module_repr(module)
            counts = _module_parameter_counts(module)
            if counts is not None:
                module_param_counts[name] = counts

    snapshot: dict[str, object] = {
        "library_versions": {
            "stable_baselines3": _stable_baselines3_version(),
            "torch": _torch_version(),
            "python": platform.python_version(),
        },
        "agent": _collect_resolved_sb3_config(agent),
        "policy": {
            "policy_repr": _module_repr(policy),
            "policy_kwargs": _json_safe(getattr(agent, "policy_kwargs", None)),
            "net_arch": _json_safe(getattr(policy, "net_arch", None)),
            "activation_fn": _callable_name(getattr(policy, "activation_fn", None)),
            "ortho_init": _json_safe(getattr(policy, "ortho_init", None)),
            "features_extractor_class": (
                getattr(getattr(policy, "features_extractor", None), "__class__", type(None)).__name__
                if getattr(policy, "features_extractor", None) is not None
                else None
            ),
            "share_features_extractor": _json_safe(
                getattr(policy, "share_features_extractor", None)
            ),
            "normalize_images": _json_safe(getattr(policy, "normalize_images", None)),
            "parameter_counts": _module_parameter_counts(policy),
            "module_parameter_counts": module_param_counts,
            "module_reprs": module_reprs,
            "optimizer": {
                "class": optimizer.__class__.__name__ if optimizer is not None else None,
                "defaults": _json_safe(
                    getattr(optimizer, "defaults", None)
                ) if optimizer is not None else None,
            },
        },
    }
    return _json_safe(snapshot)  # type: ignore[return-value]


def _apply_resolved_agent_config_to_manifest(
    manifest: dict[str, object],
    *,
    logdir: Path,
    agent: PPO | None,
) -> dict[str, object]:
    resolved_sb3 = _collect_resolved_sb3_config(agent)
    if not resolved_sb3:
        return manifest

    manifest["sb3"] = resolved_sb3
    manifest.setdefault("config", {})
    cfg = manifest["config"] if isinstance(manifest.get("config"), dict) else {}
    manifest["config"] = cfg
    cfg["device_resolved"] = resolved_sb3.get("device")
    for key in [
        "gamma",
        "gae_lambda",
        "learning_rate",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "n_steps",
        "batch_size",
        "n_epochs",
    ]:
        if key in resolved_sb3:
            cfg[key] = resolved_sb3.get(key)
    if resolved_sb3.get("policy_kwargs") is not None:
        cfg["policy_kwargs"] = resolved_sb3.get("policy_kwargs")
    policy_summary = resolved_sb3.get("policy_summary")
    if isinstance(policy_summary, dict) and policy_summary.get("net_arch") is not None:
        cfg["net_arch"] = policy_summary.get("net_arch")

    if manifest.get("train_invocations"):
        invs = manifest["train_invocations"]
        if isinstance(invs, list) and invs:
            invs[-1]["resolved_train_args"] = _manifest_resolved_train_args(resolved_sb3)
            invs[-1]["ppo_args"] = {
                "n_steps": resolved_sb3.get("n_steps"),
                "batch_size": resolved_sb3.get("batch_size"),
                "n_epochs": resolved_sb3.get("n_epochs"),
                "gamma": resolved_sb3.get("gamma"),
                "gae_lambda": resolved_sb3.get("gae_lambda"),
                "learning_rate": resolved_sb3.get("learning_rate"),
                "clip_range": resolved_sb3.get("clip_range"),
                "ent_coef": resolved_sb3.get("ent_coef"),
                "vf_coef": resolved_sb3.get("vf_coef"),
                "max_grad_norm": resolved_sb3.get("max_grad_norm"),
                "target_kl": resolved_sb3.get("target_kl"),
                "policy_kwargs": resolved_sb3.get("policy_kwargs"),
                "net_arch": (
                    policy_summary.get("net_arch")
                    if isinstance(policy_summary, dict)
                    else None
                ),
            }

    resolved_snapshot = _collect_resolved_config_snapshot(agent)
    if resolved_snapshot:
        resolved_path = logdir / "resolved_config.json"
        _write_json(resolved_path, resolved_snapshot)
        manifest.setdefault("artifacts", {})
        arts = manifest["artifacts"] if isinstance(manifest.get("artifacts"), dict) else {}
        manifest["artifacts"] = arts
        arts["resolved_config"] = str(resolved_path.resolve())
        sb3 = manifest["sb3"] if isinstance(manifest.get("sb3"), dict) else {}
        manifest["sb3"] = sb3
        sb3["resolved_config_path"] = str(resolved_path.resolve())

    return manifest


def _manifest_resolved_train_args(resolved_sb3: dict[str, object]) -> dict[str, object]:
    keys = [
        "device",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "learning_rate",
        "gae_lambda",
        "clip_range",
        "clip_range_vf",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "policy_kwargs",
        "max_grad_norm",
        "target_kl",
        "use_sde",
        "sde_sample_freq",
        "stats_window_size",
    ]
    return {k: resolved_sb3[k] for k in keys if k in resolved_sb3}


# ---------------------------------------------------------------------
# Agent / reward / env builders
# ---------------------------------------------------------------------
def build_agent(
    env: gym.Env,
    algo: str = "ppo",
    seed: int | None = None,
    *,
    device: str = "auto",
    policy_type: str = "split_action_heads_sj_no_spr",
    obs_columns: list[str] | tuple[str, ...] | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    gamma: float | None = None,
    gae_lambda: float | None = None,
    learning_rate: float | None = None,
    clip_range: float | None = None,
    ent_coef: float | None = None,
    vf_coef: float | None = None,
    max_grad_norm: float | None = None,
    target_kl: float | None = None,
    action_log_std_init: float | None = None,
    policy_net_arch: str | dict[str, list[int]] | None = None,
) -> PPO:
    algo = algo.lower()
    if algo != "ppo":
        raise ValueError(f"Unsupported algo: {algo}")

    ppo_kwargs: dict = {
        "verbose": 1,
        "seed": seed,
        "device": device,
    }
    if n_steps is not None:
        ppo_kwargs["n_steps"] = n_steps
    if batch_size is not None:
        ppo_kwargs["batch_size"] = batch_size
    if n_epochs is not None:
        ppo_kwargs["n_epochs"] = n_epochs
    if gamma is not None:
        ppo_kwargs["gamma"] = gamma
    if gae_lambda is not None:
        ppo_kwargs["gae_lambda"] = gae_lambda
    if learning_rate is not None:
        ppo_kwargs["learning_rate"] = learning_rate
    if clip_range is not None:
        ppo_kwargs["clip_range"] = clip_range
    if ent_coef is not None:
        ppo_kwargs["ent_coef"] = ent_coef
    if vf_coef is not None:
        ppo_kwargs["vf_coef"] = vf_coef
    if max_grad_norm is not None:
        ppo_kwargs["max_grad_norm"] = max_grad_norm
    if target_kl is not None:
        ppo_kwargs["target_kl"] = target_kl
    policy_kwargs = policy_net_arch_to_policy_kwargs(policy_net_arch)
    if action_log_std_init is not None:
        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs["log_std_init"] = float(action_log_std_init)
    policy_cls: object
    policy_type_eff = normalize_policy_type(policy_type)
    policy_cls = SplitActionHeadsActorCriticPolicy
    policy_kwargs = dict(policy_kwargs or {})
    policy_kwargs["obs_column_names"] = list(obs_columns or [])
    policy_kwargs["sj_exclude_prefixes"] = ("spr_",)
    policy_kwargs["sj_exclude_names"] = ("animas_spr_frac",)
    if policy_kwargs is not None:
        ppo_kwargs["policy_kwargs"] = policy_kwargs

    return PPO(policy_cls, env, **ppo_kwargs)


def build_reward(
    reward_spec_str: str,
    *,
    reward_balancing: str = "none",
    update_balancing: bool = True,
):
    spec = drl_rewards.parse_objective_spec(reward_spec_str)
    composite = drl_rewards.build_composite_reward(
        spec,
        weights=None,
        reward_balancing=reward_balancing,
        update_balancing=update_balancing,
    )
    return composite


def make_env(
    data_raw: pd.DataFrame,
    data_norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    reward_spec_str: str,
    *,
    episode_length: int | None,
    is_eval: bool = False,
    max_release_sj_main_cfs: float | None = None,
    max_release_niip_cfs: float | None = None,
    obs_context: str = SELECTED_OBS_CONTEXT,
    action_scaling: str = "linear",
    action_mode: str = "esa_base_spr_proxy_4d_max",
    reward_balancing: str = "none",
    esa_min_flow_floor: bool = False,
    esa_baseflow_max_multiplier: float = 1.5,
    spr_proxy_priority_release: bool = True,
    spr_proxy_owns_sj_window: bool = False,
    spr_advice_mode: str = "days_remaining",
    decision_hydrology_timing: str = "same_day",
    niip_fallback_mode: str = "legacy_full_series",
    storage_datum_mode: str = "reported",
    storage_normalization: str = "full_record",
    storage_budget_target_frac_of_max: float = 0.780,
    mask_incomplete_initial_spr: bool = False,
    prior_day_hydrology: pd.Series | None = None,
) -> gym.Env:
    """
    Build a single NavajoReservoirEnv.

    - Training: pass episode_length (e.g., 3600) -> random-window episodes inside env.reset
    - Eval: pass episode_length=None -> full series
    """
    reward_fn = build_reward(
        reward_spec_str,
        reward_balancing=reward_balancing,
        update_balancing=(not is_eval),
    )

    resolved_storage_normalization = normalize_storage_normalization(
        storage_normalization
    )
    resolved_storage_datum = normalize_storage_datum_mode(storage_datum_mode)
    env = NavajoReservoirEnv(
        data_raw=data_raw,
        data_norm=data_norm,
        norm_stats=norm_stats,
        reward_fn=reward_fn,
        episode_length=episode_length,
        is_eval=is_eval,
        obs_context=obs_context,
        action_scaling=action_scaling,
        action_mode=action_mode,
        esa_min_flow_floor=esa_min_flow_floor,
        esa_baseflow_max_multiplier=esa_baseflow_max_multiplier,
        spr_proxy_priority_release=spr_proxy_priority_release,
        spr_proxy_owns_sj_window=spr_proxy_owns_sj_window,
        spr_advice_mode=spr_advice_mode,
        decision_hydrology_timing=decision_hydrology_timing,
        niip_fallback_mode=niip_fallback_mode,
        storage_budget_target_frac_of_max=storage_budget_target_frac_of_max,
        mask_incomplete_initial_spr=mask_incomplete_initial_spr,
        prior_day_hydrology=prior_day_hydrology,
        **(
            {}
            if max_release_sj_main_cfs is None
            else {"max_release_sj_main_cfs": float(max_release_sj_main_cfs)}
        ),
        **(
            {}
            if max_release_niip_cfs is None
            else {"max_release_niip_cfs": float(max_release_niip_cfs)}
        ),
    )
    # The environment receives already-resolved numerical stats. Retain the
    # provenance mode for diagnostics without changing the frozen observation
    # schema or the environment constructor.
    env.storage_normalization = resolved_storage_normalization
    env.storage_datum_mode = resolved_storage_datum
    return env


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------
def _run_rollout_env(
    *,
    agent: PPO,
    eval_env: gym.Env,
    reset_options: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run a deterministic rollout on an already-constructed evaluation env."""
    obs, _ = eval_env.reset(options=reset_options)
    step = 0
    records: list[dict] = []

    def _to_float_or_nan(val: object) -> float:
        return np.nan if val is None else float(val)

    while True:
        # Date
        if hasattr(eval_env, "_current_date") and callable(eval_env._current_date):  # type: ignore[attr-defined]
            date = eval_env._current_date()  # type: ignore[attr-defined]
        else:
            t = getattr(eval_env, "t", step)
            start_idx = getattr(eval_env, "start_idx", 0)
            global_idx = start_idx + t
            date = eval_env.data_raw.index[global_idx]

        # Agent internal state before step
        storage_agent_af = float(eval_env.storage_af)
        elev_agent_ft = float(eval_env._capacity_to_elev_scalar(storage_agent_af)) if hasattr(eval_env, "_capacity_to_elev_scalar") else float(eval_env.capacity_to_elev(storage_agent_af))

        # Step
        action, _ = agent.predict(obs, deterministic=True)
        action_arr = np.asarray(action, dtype=float).reshape(-1)
        next_obs, reward, terminated, truncated, info_step = eval_env.step(action)
        done = bool(terminated or truncated)

        rec: dict[str, float | int | pd.Timestamp] = {
            "step": step,
            "date": pd.to_datetime(date),
            "reward": float(reward),
            "storage_agent_af": storage_agent_af,
            "elev_agent_ft": elev_agent_ft,
        }
        for i, val in enumerate(action_arr.tolist()):
            rec[f"action_{i}"] = float(val)

        # Reward components
        comps = info_step.get("reward_components", {})
        for k, v in comps.items():
            rec[f"rc_{k}"] = float(v)

        # State after step / operational diagnostics
        if "storage_af" in info_step:
            rec["storage_agent_af_end"] = _to_float_or_nan(info_step["storage_af"])
        if "elev_ft" in info_step:
            rec["elev_agent_ft_end"] = _to_float_or_nan(info_step["elev_ft"])
        elif "storage_agent_af_end" in rec:
            rec["elev_agent_ft_end"] = (
                float(eval_env._capacity_to_elev_scalar(rec["storage_agent_af_end"]))
                if hasattr(eval_env, "_capacity_to_elev_scalar")
                else float(eval_env.capacity_to_elev(rec["storage_agent_af_end"]))
            )

        # Component releases and operational fields (use env's native keys where possible)
        passthrough_keys = [
            "release_sj_main_cfs",
            "release_niip_cfs",
            "sj_main_flow_cfs",
            "sj_at_archuleta_proxy_cfs",
            "sj_at_farmington_cfs",
            "sj_at_farmington_lag2_cfs",
            "sj_at_bluff_proxy_cfs",
            "spill_cfs",
            "spill_af",
            "deadpool_block",
            "release_cap_penalty",
            "release_phys_penalty",
            "available_af",
            "total_release_af",
            "total_controlled_release_af",
            "prev_elev_ft",
            "deadpool_elev_ft",
            "spill_elev_ft",
            "raw_action_sj_frac",
            "scaled_action_sj_frac",
            "raw_action_niip_frac",
            "scaled_action_niip_frac",
            "raw_action_spr_proxy",
            "scaled_action_spr_proxy_frac",
            "spr_proxy_target_index",
            "spr_proxy_raw_target_index",
            "spr_proxy_raw_target_cfs",
            "spr_proxy_target_cfs",
            "spr_proxy_mask_applied",
            "spr_proxy_action_raw",
            "spr_proxy_action_frac",
            "spr_proxy_window_active",
            "spr_proxy_target_reachable",
            "spr_proxy_target_reachable_at_decision",
            "decision_animas_farmington_q_cfs",
            "decision_inflow_cfs",
            "decision_evap_af",
            "spr_proxy_controller_need_cfs",
            "spr_proxy_actual_bridge_need_cfs",
            "spr_proxy_baseline_request_sj_main_cfs",
            "spr_proxy_added_request_cfs",
            "spr_proxy_suppressed_baseline_request_cfs",
            "spr_proxy_priority_scaled",
            "spr_proxy_priority_release_cfs",
            "spr_proxy_farmington_after_release_cfs",
            "spr_proxy_target_hit",
            "spr_calendar_window_active",
            "spr_initial_partial_season_masked",
            "spr_reward_eligible",
            "mask_incomplete_initial_spr",
            "storage_budget_target_frac_of_max",
            "storage_budget_target_af",
            "esa_baseflow_action_mode",
            "esa_baseflow_max_multiplier",
            "raw_action_esa_base_frac",
            "scaled_action_esa_base_frac",
            "raw_action_discretionary_sj_frac",
            "scaled_action_discretionary_sj_frac",
            "esa_base_required_sj_cfs",
            "esa_floor_required_cfs",
            "esa_floor_decision_required_cfs",
            "esa_base_request_sj_cfs",
            "esa_base_multiplier",
            "discretionary_sj_request_cfs",
            "combined_sj_request_cfs",
            "spr_attributed_sj_release_cfs",
            "spr_useful_unrequested_sj_release_cfs",
            "spr_useful_unrequested_threshold_cfs",
            "spr_useful_unrequested_duration_days",
            "esa_attributed_sj_release_cfs",
            "hydropower_attributed_sj_release_cfs",
            "discretionary_attributed_sj_release_cfs",
            "esa_covered_by_controlled_release_cfs",
            "spr_covered_by_controlled_release_cfs",
            "release_beyond_esa_spr_need_cfs",
            "release_beyond_esa_spr_request_cfs",
            "spr_advice_target_req05_frac",
            "spr_advice_threshold_frac",
            "spr_advice_progress_frac",
            "spr_advice_viability_frac",
            "spr_advice_active",
            "niip_demand_cfs",
        ]
        for key in passthrough_keys:
            if key in info_step:
                val = info_step[key]
                if key in {
                    "deadpool_block",
                    "esa_baseflow_action_mode",
                    "spr_proxy_mask_applied",
                    "spr_proxy_window_active",
                    "spr_proxy_target_reachable",
                    "spr_proxy_target_reachable_at_decision",
                    "spr_proxy_priority_scaled",
                    "spr_proxy_target_hit",
                    "spr_calendar_window_active",
                    "spr_initial_partial_season_masked",
                    "spr_reward_eligible",
                    "mask_incomplete_initial_spr",
                    "spr_advice_active",
                }:
                    rec[key] = int(bool(val))
                else:
                    rec[key] = _to_float_or_nan(val)

        for key in [
            "sj_request_combination_rule",
            "spr_proxy_mask_mode",
            "action_mode",
            "decision_hydrology_timing",
        ]:
            if key in info_step:
                rec[key] = str(info_step[key])

        if "deadpool_storage_af" in info_step:
            rec["deadpool_storage_af"] = _to_float_or_nan(info_step["deadpool_storage_af"])
            rec.setdefault("min_storage_af", _to_float_or_nan(info_step["deadpool_storage_af"]))
        if "max_storage_af" in info_step:
            rec["max_storage_af"] = _to_float_or_nan(info_step["max_storage_af"])

        if "total_release_cfs" in info_step:
            rec["release_agent_cfs"] = _to_float_or_nan(info_step["total_release_cfs"])
        if "total_controlled_release_cfs" in info_step:
            rec["release_agent_controlled_cfs"] = _to_float_or_nan(info_step["total_controlled_release_cfs"])

        req_sj = info_step.get("requested_release_sj_main_cfs")
        req_niip = info_step.get("requested_release_niip_cfs")
        if req_sj is not None:
            rec["requested_release_sj_main_cfs"] = _to_float_or_nan(req_sj)
        if req_niip is not None:
            rec["requested_release_niip_cfs"] = _to_float_or_nan(req_niip)
        if req_sj is not None or req_niip is not None:
            rec["requested_total_release_cfs"] = float((req_sj or 0.0) + (req_niip or 0.0))

        # Historic row
        date_row = eval_env.data_raw.loc[rec["date"]]

        if "storage_af" in date_row.index:
            rec["storage_hist_af"] = float(date_row["storage_af"])
        if "storage_reported_af" in date_row.index:
            rec["storage_hist_reported_af"] = float(
                date_row["storage_reported_af"]
            )
            if "elev_ft" in date_row.index:
                rec["elev_hist_reported_ft"] = float(date_row["elev_ft"])

        for col in [
            "release_cfs",
            "inflow_cfs",
            "evap_af",
            "sj_archuleta_q_cfs",
            "sj_farmington_q_cfs",
            "animas_farmington_q_cfs",
            "sj_bluff_q_cfs",
        ]:
            if col in date_row.index:
                rec[col] = float(date_row[col])

        # Convenience conversion for plotting: evap_af (acre-feet/day) -> evap_cfs
        # 1 AF = 43,560 ft^3 ; 1 day = 86,400 s
        if "evap_af" in rec and "evap_cfs" not in rec:
            rec["evap_cfs"] = float(rec["evap_af"]) * (43560.0 / 86400.0)

        # Hydropower: prefer the env-computed value (controlled SJ release + end-of-step elevation)
        if "hydropower_mwh" in info_step:
            rec["hydro_agent_mwh"] = _to_float_or_nan(info_step["hydropower_mwh"])
        elif "release_sj_main_cfs" in rec:
            elev_for_hp = float(rec.get("elev_agent_ft_end", elev_agent_ft))
            hp_agent = navajo_power_generation_scalar(
                cfs_value=float(rec["release_sj_main_cfs"]),
                elevation_ft=elev_for_hp,
            )
            rec["hydro_agent_mwh"] = float(hp_agent)

        # Historic hydropower: use historic total release + elevation from historic storage
        if "storage_hist_af" in rec and "release_cfs" in rec:
            elev_hist_ft = float(eval_env._capacity_to_elev_scalar(rec["storage_hist_af"])) if hasattr(eval_env, "_capacity_to_elev_scalar") else float(eval_env.capacity_to_elev(rec["storage_hist_af"]))
            hp_hist = navajo_power_generation_scalar(
                cfs_value=float(rec["release_cfs"]),
                elevation_ft=elev_hist_ft,
            )
            rec["elev_hist_ft"] = elev_hist_ft
            rec["hydro_hist_mwh"] = float(hp_hist)

        records.append(rec)

        obs = next_obs
        step += 1
        if done:
            break

    df = pd.DataFrame.from_records(records).set_index("date").sort_index()
    return df


def run_rollout_data(
    *,
    model_path: Path | str,
    reward_spec: str,
    data_raw: pd.DataFrame,
    data_norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    device: str = "auto",
    reset_options: dict[str, object] | None = None,
    max_release_sj_main_cfs: float | None = None,
    max_release_niip_cfs: float | None = None,
    obs_context: str = SELECTED_OBS_CONTEXT,
    action_scaling: str = "linear",
    action_mode: str = "esa_base_spr_proxy_4d_max",
    reward_balancing: str = "none",
    esa_min_flow_floor: bool = False,
    esa_baseflow_max_multiplier: float = 1.5,
    spr_proxy_priority_release: bool = True,
    spr_proxy_owns_sj_window: bool = False,
    spr_advice_mode: str = "days_remaining",
    decision_hydrology_timing: str = "same_day",
    niip_fallback_mode: str = "legacy_full_series",
    storage_datum_mode: str = "reported",
    storage_normalization: str = "full_record",
    storage_budget_target_frac_of_max: float = 0.780,
    mask_incomplete_initial_spr: bool = False,
    prior_day_hydrology: pd.Series | None = None,
) -> pd.DataFrame:
    """Deterministic rollout over a provided raw/normalized evaluation slice."""
    model_path = Path(model_path)
    _validate_storage_norm_stats_against_artifact(
        model_path=model_path,
        norm_stats=norm_stats,
        mode=storage_normalization,
    )
    eval_env = make_env(
        data_raw=data_raw,
        data_norm=data_norm,
        norm_stats=norm_stats,
        reward_spec_str=reward_spec,
        episode_length=None,
        is_eval=True,
        max_release_sj_main_cfs=max_release_sj_main_cfs,
        max_release_niip_cfs=max_release_niip_cfs,
        obs_context=obs_context,
        action_scaling=action_scaling,
        action_mode=action_mode,
        reward_balancing=reward_balancing,
        esa_min_flow_floor=esa_min_flow_floor,
        esa_baseflow_max_multiplier=esa_baseflow_max_multiplier,
        spr_proxy_priority_release=spr_proxy_priority_release,
        spr_proxy_owns_sj_window=spr_proxy_owns_sj_window,
        spr_advice_mode=spr_advice_mode,
        decision_hydrology_timing=decision_hydrology_timing,
        niip_fallback_mode=niip_fallback_mode,
        storage_datum_mode=storage_datum_mode,
        storage_normalization=storage_normalization,
        storage_budget_target_frac_of_max=storage_budget_target_frac_of_max,
        mask_incomplete_initial_spr=mask_incomplete_initial_spr,
        prior_day_hydrology=prior_day_hydrology,
    )
    agent = PPO.load(model_path, device=device)
    return _run_rollout_env(agent=agent, eval_env=eval_env, reset_options=reset_options)


def run_rollout_window(
    *,
    model_path: Path | str,
    reward_spec: str,
    window_start: str | None = None,
    window_end: str | None = None,
    device: str = "auto",
    obs_context: str = SELECTED_OBS_CONTEXT,
    max_release_sj_main_cfs: float | None = None,
    max_release_niip_cfs: float | None = None,
    action_scaling: str = "linear",
    action_mode: str = "esa_base_spr_proxy_4d_max",
    reward_balancing: str = "none",
    esa_min_flow_floor: bool = False,
    esa_baseflow_max_multiplier: float = 1.5,
    spr_proxy_priority_release: bool = True,
    spr_proxy_owns_sj_window: bool = False,
    spr_advice_mode: str = "days_remaining",
    decision_hydrology_timing: str = "same_day",
    niip_fallback_mode: str = "legacy_full_series",
    storage_datum_mode: str = "reported",
    storage_normalization: str = "full_record",
    storage_budget_target_frac_of_max: float = 0.780,
    mask_incomplete_initial_spr: bool = False,
) -> pd.DataFrame:
    """Deterministic rollout over an arbitrary time window.

    Parameters
    ----------
    model_path:
        Path to the SB3 .zip file.
    reward_spec:
        Objective specification string used to build the env reward.
    window_start, window_end:
        Either 4-digit water year tokens (e.g. '2002') or 'YYYY-MM-DD'.
        If omitted, uses the full available model_data period.

    Returns
    -------
    DataFrame indexed by date.
    """
    all_data = load_all_model_data(storage_datum_mode=storage_datum_mode)
    raw_all = all_data["raw"]
    norm_all = all_data["norm"]
    norm_stats, _ = _storage_norm_stats_for_evaluation(
        model_path=model_path,
        full_record_norm_stats=all_data["norm_stats"],
        mode=storage_normalization,
    )

    raw_slice, _ = helpers.slice_by_window(
        raw_all,
        start_token=window_start,
        end_token=window_end,
        label="eval_window",
    )
    norm_slice = norm_all.loc[raw_slice.index]
    prior_day_hydrology = _previous_calendar_hydrology_row(
        raw_all,
        first_date=raw_slice.index[0],
    )

    return run_rollout_data(
        model_path=model_path,
        reward_spec=reward_spec,
        data_raw=raw_slice,
        data_norm=norm_slice,
        norm_stats=norm_stats,
        device=device,
        obs_context=obs_context,
        max_release_sj_main_cfs=max_release_sj_main_cfs,
        max_release_niip_cfs=max_release_niip_cfs,
        action_scaling=action_scaling,
        action_mode=action_mode,
        reward_balancing=reward_balancing,
        esa_min_flow_floor=esa_min_flow_floor,
        esa_baseflow_max_multiplier=esa_baseflow_max_multiplier,
        spr_proxy_priority_release=spr_proxy_priority_release,
        spr_proxy_owns_sj_window=spr_proxy_owns_sj_window,
        spr_advice_mode=spr_advice_mode,
        decision_hydrology_timing=decision_hydrology_timing,
        niip_fallback_mode=niip_fallback_mode,
        storage_datum_mode=storage_datum_mode,
        storage_normalization=storage_normalization,
        storage_budget_target_frac_of_max=storage_budget_target_frac_of_max,
        mask_incomplete_initial_spr=mask_incomplete_initial_spr,
        prior_day_hydrology=prior_day_hydrology,
    )


def evaluate_model_window(
    *,
    model_path: Path | str,
    reward_spec: str,
    window_start: str | None,
    window_end: str | None,
    outdir: Path | str,
    device: str = "auto",
    which_metrics: str = "core",
    save_rollout: bool = True,
    save_metrics: bool = True,
    obs_context: str = SELECTED_OBS_CONTEXT,
    max_release_sj_main_cfs: float | None = None,
    max_release_niip_cfs: float | None = None,
    action_scaling: str = "linear",
    action_mode: str = "esa_base_spr_proxy_4d_max",
    reward_balancing: str = "none",
    esa_min_flow_floor: bool = False,
    esa_baseflow_max_multiplier: float = 1.5,
    spr_proxy_priority_release: bool = True,
    spr_proxy_owns_sj_window: bool = False,
    spr_advice_mode: str = "days_remaining",
    decision_hydrology_timing: str = "same_day",
    niip_fallback_mode: str = "legacy_full_series",
    storage_datum_mode: str = "reported",
    storage_normalization: str = "full_record",
    storage_budget_target_frac_of_max: float = 0.780,
    mask_incomplete_initial_spr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a model on a specified window and write outputs to outdir."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = run_rollout_window(
        model_path=model_path,
        reward_spec=reward_spec,
        window_start=window_start,
        window_end=window_end,
        device=device,
        obs_context=obs_context,
        max_release_sj_main_cfs=max_release_sj_main_cfs,
        max_release_niip_cfs=max_release_niip_cfs,
        action_scaling=action_scaling,
        action_mode=action_mode,
        reward_balancing=reward_balancing,
        esa_min_flow_floor=esa_min_flow_floor,
        esa_baseflow_max_multiplier=esa_baseflow_max_multiplier,
        spr_proxy_priority_release=spr_proxy_priority_release,
        spr_proxy_owns_sj_window=spr_proxy_owns_sj_window,
        spr_advice_mode=spr_advice_mode,
        decision_hydrology_timing=decision_hydrology_timing,
        niip_fallback_mode=niip_fallback_mode,
        storage_datum_mode=storage_datum_mode,
        storage_normalization=storage_normalization,
        storage_budget_target_frac_of_max=storage_budget_target_frac_of_max,
        mask_incomplete_initial_spr=mask_incomplete_initial_spr,
    )

    if save_rollout:
        stem = "eval_rollout"
        out_path = outdir / f"{stem}.parquet"
        df.to_parquet(out_path)
        df.to_csv(out_path.with_suffix(".csv"), index=True)


    metrics_df = drl_metrics.compute_metrics(df, which=which_metrics, validate=True)
    if save_metrics:
        drl_metrics.save_metrics(
            df_test=df,
            outdir=outdir,
            which=which_metrics,
            stem="eval_metrics",
            validate=True,
        )

    return df, metrics_df

# ---------------------------------------------------------------------
# DRLModel class: single training entry point
# ---------------------------------------------------------------------
class DRLModel:
    """
    PPO training with contiguous random-window episodes and recorded diagnostics.
    The public CLI supplies the frozen paper configuration.
    """

    def __init__(
        self,
        reward_spec: str,
        *,
        train_start: str | None = None,
        train_end: str | None = None,
        train_hydrology_transform: str = "none",
        storage_datum_mode: str = "reported",
        storage_normalization: str = "full_record",
        algo: str = "ppo",
        logdir: str | Path = "runs/debug",
        seed: int | None = None,
        device: str = "auto",
        gamma: float | None = None,
        gae_lambda: float | None = None,
        learning_rate: float | None = None,
        clip_range: float | None = None,
        ent_coef: float | None = None,
        vf_coef: float | None = None,
        max_grad_norm: float | None = None,
        target_kl: float | None = None,
        action_log_std_init: float | None = None,
        policy_type: str = "split_action_heads_sj_no_spr",
        policy_net_arch: str | None = None,
        episode_length_train: int = 3600,
        launch_mode: str = "unknown",
        obs_context: str = SELECTED_OBS_CONTEXT,
        action_scaling: str = "linear",
        action_mode: str = "esa_base_spr_proxy_4d_max",
        max_release_sj_main_cfs: float | None = None,
        max_release_niip_cfs: float | None = None,
        reward_balancing: str = "none",
        esa_min_flow_floor: bool = False,
        esa_baseflow_max_multiplier: float = 1.5,
        spr_proxy_priority_release: bool = True,
        spr_proxy_owns_sj_window: bool = False,
        spr_advice_mode: str = "days_remaining",
        decision_hydrology_timing: str = "same_day",
        niip_fallback_mode: str = "legacy_full_series",
        storage_budget_target_frac_of_max: float = 0.780,
        mask_incomplete_initial_spr: bool = False,
    ) -> None:

        self.reward_spec = str(reward_spec)
        self.algo = algo.lower()
        self.logdir = Path(logdir)
        self.seed = seed
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.action_log_std_init = action_log_std_init
        self.policy_type = normalize_policy_type(policy_type)
        self.policy_net_arch = normalize_policy_net_arch(policy_net_arch)
        self.episode_length_train = int(episode_length_train)
        self.launch_mode = str(launch_mode)
        self.obs_context = str(obs_context or SELECTED_OBS_CONTEXT).strip().lower()
        self.obs_cols = list(obs_columns_for_context(self.obs_context))
        self.action_scaling = normalize_action_scaling(action_scaling)
        self.action_mode = normalize_action_mode(action_mode)
        self.reward_balancing = drl_rewards.normalize_reward_balancing(
            reward_balancing
        )
        self.esa_min_flow_floor = bool(esa_min_flow_floor)
        self.esa_baseflow_max_multiplier = max(float(esa_baseflow_max_multiplier), 0.0)
        self.spr_proxy_priority_release = bool(spr_proxy_priority_release)
        self.spr_proxy_owns_sj_window = bool(spr_proxy_owns_sj_window)
        self.spr_advice_mode = normalize_spr_advice_mode(spr_advice_mode)
        self.decision_hydrology_timing = normalize_decision_hydrology_timing(
            decision_hydrology_timing
        )
        self.niip_fallback_mode = normalize_niip_fallback_mode(niip_fallback_mode)
        self.storage_budget_target_frac_of_max = float(
            storage_budget_target_frac_of_max
        )
        self.mask_incomplete_initial_spr = bool(mask_incomplete_initial_spr)
        self.max_release_sj_main_cfs = (
            None
            if max_release_sj_main_cfs is None
            else float(max_release_sj_main_cfs)
        )
        self.max_release_niip_cfs = (
            None
            if max_release_niip_cfs is None
            else float(max_release_niip_cfs)
        )

        self.train_start = train_start
        self.train_end = train_end
        self.train_hydrology_transform = normalize_train_hydrology_transform(
            train_hydrology_transform
        )
        self.storage_datum_mode = normalize_storage_datum_mode(
            storage_datum_mode
        )
        self.storage_normalization = normalize_storage_normalization(
            storage_normalization
        )

        self._train_updates_cb: TrainUpdateRewardComponentsCallback | None = None
        self.train_update_metrics_: pd.DataFrame | None = None

        # Best-effort load previous metrics
        try:
            self.load_train_update_metrics()
        except Exception:
            pass

        # Load full data once, then slice into train/eval windows.
        self._all_data = load_all_model_data(
            storage_datum_mode=self.storage_datum_mode,
        )
        all_raw = self._all_data["raw"]
        all_norm = self._all_data["norm"]
        full_record_norm_stats = self._all_data["norm_stats"]

        self._data_range = helpers.available_date_range(all_raw)

        self.datasets, self._window_meta = _slice_training_window(
            all_raw,
            all_norm,
            train_start=self.train_start,
            train_end=self.train_end,
        )
        self.norm_stats, self.storage_normalization_meta = (
            _storage_normalization_for_training(
                train_raw=self.datasets["train_raw"],
                all_raw=all_raw,
                full_record_norm_stats=full_record_norm_stats,
                mode=self.storage_normalization,
            )
        )
        self._window_meta["storage_normalization"] = self.storage_normalization_meta
        self._window_meta["storage_datum"] = self._all_data["storage_datum_meta"]
        (
            self.datasets["train_raw"],
            self.datasets["train_norm"],
            hydrology_transform_meta,
        ) = _apply_train_hydrology_transform(
            train_raw=self.datasets["train_raw"],
            train_norm=self.datasets["train_norm"],
            all_raw=all_raw,
            norm_stats=self.norm_stats,
            transform=self.train_hydrology_transform,
        )
        self._window_meta["train_hydrology_transform"] = hydrology_transform_meta

        # TRAIN ENV (random 3600-step episodes)
        self.train_env = make_env(
            data_raw=self.datasets["train_raw"],
            data_norm=self.datasets["train_norm"],
            norm_stats=self.norm_stats,
            reward_spec_str=self.reward_spec,
            episode_length=self.episode_length_train,
            is_eval=False,
            obs_context=self.obs_context,
            action_scaling=self.action_scaling,
            action_mode=self.action_mode,
            reward_balancing=self.reward_balancing,
            max_release_sj_main_cfs=self.max_release_sj_main_cfs,
            max_release_niip_cfs=self.max_release_niip_cfs,
            esa_min_flow_floor=self.esa_min_flow_floor,
            esa_baseflow_max_multiplier=self.esa_baseflow_max_multiplier,
            spr_proxy_priority_release=self.spr_proxy_priority_release,
            spr_proxy_owns_sj_window=self.spr_proxy_owns_sj_window,
            spr_advice_mode=self.spr_advice_mode,
            decision_hydrology_timing=self.decision_hydrology_timing,
            niip_fallback_mode=self.niip_fallback_mode,
            storage_datum_mode=self.storage_datum_mode,
            storage_normalization=self.storage_normalization,
            storage_budget_target_frac_of_max=(
                self.storage_budget_target_frac_of_max
            ),
            mask_incomplete_initial_spr=self.mask_incomplete_initial_spr,
        )
        self.train_env = Monitor(self.train_env)


        self.agent: PPO | None = None

    def train(
        self,
        *,
        n_episodes: int = 350,
        total_timesteps: int | None = None,
        device: str | None = None,
        n_steps: int | None = None,
        batch_size: int | None = None,
        n_epochs: int | None = None,
        gamma: float | None = None,
        gae_lambda: float | None = None,
        learning_rate: float | None = None,
        clip_range: float | None = None,
        ent_coef: float | None = None,
        vf_coef: float | None = None,
        max_grad_norm: float | None = None,
        target_kl: float | None = None,
        action_log_std_init: float | None = None,
        policy_type: str | None = None,
        policy_net_arch: str | None = None,
        track_reward_components: bool = True,
        rich_training_diagnostics: bool = False,
        resume: bool = False,
    ) -> None:
        """
        Train PPO.

        Original code equivalence:
          total_timesteps = n_episodes * episode_length_train

        Training uses one environment, as in the selected run.
        """
        device_requested = device or self.device
        total_timesteps_eff = int(total_timesteps) if total_timesteps is not None else int(n_episodes * self.episode_length_train)
        gamma_eff = self.gamma if self.gamma is not None else gamma
        gae_lambda_eff = self.gae_lambda if self.gae_lambda is not None else gae_lambda
        learning_rate_eff = self.learning_rate if self.learning_rate is not None else learning_rate
        clip_range_eff = self.clip_range if self.clip_range is not None else clip_range
        ent_coef_eff = self.ent_coef if self.ent_coef is not None else ent_coef
        vf_coef_eff = self.vf_coef if self.vf_coef is not None else vf_coef
        max_grad_norm_eff = self.max_grad_norm if self.max_grad_norm is not None else max_grad_norm
        target_kl_eff = self.target_kl if self.target_kl is not None else target_kl
        action_log_std_init_eff = (
            self.action_log_std_init
            if self.action_log_std_init is not None
            else action_log_std_init
        )
        if policy_type is not None:
            self.policy_type = normalize_policy_type(policy_type)
        policy_net_arch_eff = (
            self.policy_net_arch
            if self.policy_net_arch is not None
            else normalize_policy_net_arch(policy_net_arch)
        )

        self.logdir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.logdir / "run_manifest.json"

        if not resume:
            try:
                has_existing_contents = any(self.logdir.iterdir())
            except Exception:
                has_existing_contents = False
            if has_existing_contents:
                if manifest_path.exists():
                    print(
                        f"[drl][warn] training into existing run directory without resume=True: {self.logdir} "
                        "(existing manifest will be updated and prior artifacts may be overwritten)."
                    )
                else:
                    print(
                        f"[drl][warn] training into non-empty directory without resume=True: {self.logdir} "
                        "(artifacts may mix across runs)."
                    )

        storage_normalization_path = self.logdir / STORAGE_NORMALIZATION_FILENAME
        _write_storage_normalization_artifact(
            storage_normalization_path,
            self.storage_normalization_meta,
        )

        # -----------------------------------------------------------------
        # Run manifest (written at train start; updated after agent setup and at train end)
        # -----------------------------------------------------------------
        if manifest_path.exists():
            try:
                manifest = _read_json(manifest_path)
            except Exception:
                manifest = {}
        else:
            manifest = {}

        manifest.setdefault("project", "DeepReservoir")
        manifest.setdefault("reservoir", "Navajo")
        manifest.setdefault("logdir", str(self.logdir))
        manifest.setdefault("created_at", manifest.get("created_at", _now_iso()))
        manifest["last_updated_at"] = _now_iso()
        manifest["status"] = "training"
        manifest["launch_mode"] = getattr(self, "launch_mode", "unknown")

        manifest["config"] = {
            "algo": self.algo,
            "reward_spec": self.reward_spec,
            "obs_context": self.obs_context,
            "observation_columns": list(self.obs_cols),
            "action_scaling": self.action_scaling,
            "action_mode": self.action_mode,
            "reward_balancing": self.reward_balancing,
            "esa_min_flow_floor": self.esa_min_flow_floor,
            "esa_baseflow_max_multiplier": self.esa_baseflow_max_multiplier,
            "spr_proxy_priority_release": self.spr_proxy_priority_release,
            "spr_proxy_owns_sj_window": self.spr_proxy_owns_sj_window,
            "spr_advice_mode": self.spr_advice_mode,
            "decision_hydrology_timing": self.decision_hydrology_timing,
            "niip_fallback_mode": self.niip_fallback_mode,
            "storage_datum_mode": self.storage_datum_mode,
            "storage_normalization": self.storage_normalization,
            "storage_normalization_meta": self.storage_normalization_meta,
            "storage_budget_target_frac_of_max": (
                self.storage_budget_target_frac_of_max
            ),
            "mask_incomplete_initial_spr": self.mask_incomplete_initial_spr,
            "max_release_sj_main_cfs": self.max_release_sj_main_cfs,
            "max_release_niip_cfs": self.max_release_niip_cfs,
            "seed": self.seed,
            "device": device_requested,
            "device_requested": device_requested,
            "device_resolved": None,
            "gamma": gamma_eff,
            "gae_lambda": gae_lambda_eff,
            "learning_rate": learning_rate_eff,
            "clip_range": clip_range_eff,
            "ent_coef": ent_coef_eff,
            "vf_coef": vf_coef_eff,
            "max_grad_norm": max_grad_norm_eff,
            "target_kl": target_kl_eff,
            "action_log_std_init": action_log_std_init_eff,
            "policy_type": self.policy_type,
            "policy_net_arch": policy_net_arch_eff,
            "episode_length_train": self.episode_length_train,
            "train_hydrology_transform": self.train_hydrology_transform,
            "train": self._window_meta.get("train", None),
            "train_hydrology_transform_meta": self._window_meta.get(
                "train_hydrology_transform",
                None,
            ),
            "storage_datum_meta": self._window_meta.get("storage_datum", None),
            "data_range": {
                "start": str(self._data_range["start"].date()),
                "end": str(self._data_range["end"].date()),
                "n_days": int(self._data_range["n_days"]),
                "min_water_year": int(self._data_range["min_water_year"]),
                "max_water_year": int(self._data_range["max_water_year"]),
            },
        }
        artifacts = (
            manifest.get("artifacts", {})
            if isinstance(manifest.get("artifacts"), dict)
            else {}
        )
        artifacts["observation_normalization"] = str(
            storage_normalization_path.resolve()
        )
        manifest["artifacts"] = artifacts

        inv = {
            "started_at": _now_iso(),
            "resume": bool(resume),
            "requested_total_timesteps": int(total_timesteps_eff),
            "requested_n_episodes": int(n_episodes),
            "track_reward_components": bool(track_reward_components),
            "rich_training_diagnostics": bool(rich_training_diagnostics),
            "requested_train_args": {
                "device": device_requested,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma_eff,
                "gae_lambda": gae_lambda_eff,
                "learning_rate": learning_rate_eff,
                "clip_range": clip_range_eff,
                "ent_coef": ent_coef_eff,
                "vf_coef": vf_coef_eff,
                "max_grad_norm": max_grad_norm_eff,
                "target_kl": target_kl_eff,
                "action_log_std_init": action_log_std_init_eff,
                "policy_type": self.policy_type,
                "policy_net_arch": policy_net_arch_eff,
                "action_mode": self.action_mode,
                "spr_advice_mode": self.spr_advice_mode,
                "decision_hydrology_timing": self.decision_hydrology_timing,
                "niip_fallback_mode": self.niip_fallback_mode,
                "storage_datum_mode": self.storage_datum_mode,
                "train_hydrology_transform": self.train_hydrology_transform,
                "storage_normalization": self.storage_normalization,
                "storage_budget_target_frac_of_max": (
                    self.storage_budget_target_frac_of_max
                ),
                "mask_incomplete_initial_spr": self.mask_incomplete_initial_spr,
            },
            # Backward-compatible subset retained for older readers.
            "ppo_args": {
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma_eff,
                "gae_lambda": gae_lambda_eff,
                "learning_rate": learning_rate_eff,
                "clip_range": clip_range_eff,
                "ent_coef": ent_coef_eff,
                "vf_coef": vf_coef_eff,
                "max_grad_norm": max_grad_norm_eff,
                "target_kl": target_kl_eff,
                "action_log_std_init": action_log_std_init_eff,
                "policy_type": self.policy_type,
                "policy_net_arch": policy_net_arch_eff,
                "action_mode": self.action_mode,
                "train_hydrology_transform": self.train_hydrology_transform,
                "storage_normalization": self.storage_normalization,
                "storage_budget_target_frac_of_max": (
                    self.storage_budget_target_frac_of_max
                ),
                "mask_incomplete_initial_spr": self.mask_incomplete_initial_spr,
            },
            "resolved_train_args": {},
        }
        manifest.setdefault("train_invocations", [])
        manifest["train_invocations"].append(inv)
        _write_json(manifest_path, manifest)

        # -----------------------------------------------------------------
        # Agent creation / resume
        # -----------------------------------------------------------------
        if self.agent is None or not resume:
            self.agent = build_agent(
                self.train_env,
                algo=self.algo,
                seed=self.seed,
                device=device_requested,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma_eff,
                gae_lambda=gae_lambda_eff,
                learning_rate=learning_rate_eff,
                clip_range=clip_range_eff,
                ent_coef=ent_coef_eff,
                vf_coef=vf_coef_eff,
                max_grad_norm=max_grad_norm_eff,
                target_kl=target_kl_eff,
                action_log_std_init=action_log_std_init_eff,
                policy_type=self.policy_type,
                obs_columns=self.obs_cols,
                policy_net_arch=policy_net_arch_eff,
            )
        else:
            # Resume training: keep the loaded agent hyperparams/optimizer.
            try:
                self.agent.set_env(self.train_env)
            except Exception:
                pass

        if manifest_path.exists():
            try:
                manifest = _read_json(manifest_path)
            except Exception:
                manifest = {}
        else:
            manifest = {}
        manifest["last_updated_at"] = _now_iso()
        manifest = _apply_resolved_agent_config_to_manifest(
            manifest,
            logdir=self.logdir,
            agent=self.agent,
        )
        _write_json(manifest_path, manifest)

        best_total_reward_model_path = self.logdir / "train_best_total_reward_model.zip"
        best_total_reward_metadata_path = self.logdir / "train_best_total_reward_model.json"

        cb_list: list[BaseCallback] = []
        self._train_updates_cb = None

        # If resuming and tracking reward components, append to existing file.
        prev_updates: pd.DataFrame | None = None
        update_idx_offset = 0
        if resume and track_reward_components:
            try:
                prev_updates = self.load_train_update_metrics()
                if prev_updates is not None and "update_idx" in prev_updates.columns:
                    update_idx_offset = int(prev_updates["update_idx"].max()) + 1
            except Exception:
                prev_updates = None
                update_idx_offset = 0

        if track_reward_components:
            self._train_updates_cb = TrainUpdateRewardComponentsCallback(
                start_update_idx=update_idx_offset,
                flush_path=self.logdir / "train_update_metrics.parquet",
                flush_every_updates=10,
                best_total_reward_model_path=best_total_reward_model_path,
                best_total_reward_metadata_path=best_total_reward_metadata_path,
                reward_balancing=self.reward_balancing,
                prior_updates=prev_updates,
                rich_diagnostics=rich_training_diagnostics,
                step_diagnostics_path=(
                    self.logdir / "train_step_diagnostics.parquet"
                    if rich_training_diagnostics
                    else None
                ),
                component_diagnostics_path=(
                    self.logdir / "train_update_component_diagnostics.parquet"
                    if rich_training_diagnostics
                    else None
                ),
                event_diagnostics_path=(
                    self.logdir / "train_update_event_diagnostics.parquet"
                    if rich_training_diagnostics
                    else None
                ),
            )
            cb_list.append(self._train_updates_cb)

        callback: BaseCallback | None
        if not cb_list:
            callback = None
        else:
            callback = cb_list[0] if len(cb_list) == 1 else CallbackList(cb_list)

        train_succeeded = False
        train_wall_start = time.perf_counter()
        try:
            self.agent.learn(
                total_timesteps=int(total_timesteps_eff),
                callback=callback,
                reset_num_timesteps=(not resume),
            )
            self.save_model("last_model")
            train_succeeded = True
        finally:
            if self._train_updates_cb is not None:
                self._train_updates_cb.close()
                df_upd = self._train_updates_cb.to_dataframe()
                self.train_update_metrics_ = df_upd
                out_path = self.logdir / "train_update_metrics.parquet"
                df_upd.to_parquet(out_path, index=False)
                df_upd.to_csv(out_path.with_suffix(".csv"), index=False)

            # Update manifest with end-of-run info (best-effort)
            try:
                manifest = _read_json(manifest_path) if manifest_path.exists() else {}
                manifest["last_updated_at"] = _now_iso()
                manifest["status"] = "completed" if train_succeeded else "failed"
                if manifest.get("train_invocations"):
                    manifest["train_invocations"][-1]["ended_at"] = _now_iso()
                    manifest["train_invocations"][-1]["train_wall_seconds"] = float(
                        time.perf_counter() - train_wall_start
                    )
                    if self.agent is not None:
                        manifest["train_invocations"][-1]["agent_num_timesteps"] = int(getattr(self.agent, "num_timesteps", 0))
                manifest = _apply_resolved_agent_config_to_manifest(
                    manifest,
                    logdir=self.logdir,
                    agent=self.agent,
                )
                artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}
                artifacts.update(
                    {
                        "last_model": str((self.logdir / "last_model.zip").resolve()),
                        "train_best_total_reward_model": str(best_total_reward_model_path.resolve()),
                        "train_best_total_reward_metadata": str(best_total_reward_metadata_path.resolve()),
                    }
                )
                manifest["artifacts"] = artifacts
                _write_json(manifest_path, manifest)
            except Exception:
                pass

    def save_model(self, name: str = "last_model") -> Path:
        if self.agent is None:
            raise RuntimeError("No agent to save (train or load a model first).")
        path = self.logdir / name
        if path.suffix == "":
            path = path.with_suffix(".zip")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path))
        return path

    def load_model(self, path_or_name: str = "last_model") -> PPO:
        """Load a saved SB3 model.

        Parameters
        ----------
        path_or_name:
            Either a path to a .zip model file, or a name resolved within self.logdir.
        """
        cand = Path(path_or_name)
        if cand.exists():
            path = cand
        else:
            path = self.logdir / path_or_name
        if path.suffix == "":
            path = path.with_suffix(".zip")
        _validate_storage_norm_stats_against_artifact(
            model_path=path,
            norm_stats=self.norm_stats,
            mode=self.storage_normalization,
        )
        self.agent = PPO.load(path, env=self.train_env, device=self.device)
        return self.agent

    def load_train_update_metrics(self) -> pd.DataFrame | None:
        path = self.logdir / "train_update_metrics.parquet"
        if not path.exists():
            return None
        df_upd = pd.read_parquet(path)
        self.train_update_metrics_ = df_upd
        return df_upd


class TrainUpdateRewardComponentsCallback(BaseCallback):
    """
    Per-rollout (per PPO update) mean reward component tracker.
    Expects env to expose:
      info["reward_components_step"] OR info["reward_components"].
    """

    SPR_COMPONENT_PREFIX = "esa_spring_peak_release."
    _SPR_EPS = 1e-12
    _STEP_FLOAT_INFO_FIELDS = (
        "storage_af",
        "prev_storage_af",
        "max_storage_af",
        "release_sj_main_cfs",
        "release_niip_cfs",
        "requested_release_sj_main_cfs",
        "requested_release_niip_cfs",
        "esa_base_required_sj_cfs",
        "esa_base_request_sj_cfs",
        "spr_proxy_target_cfs",
        "spr_proxy_controller_need_cfs",
        "spr_proxy_priority_release_cfs",
        "discretionary_sj_request_cfs",
        "combined_sj_request_cfs",
        "niip_demand_cfs",
        "inflow_cfs",
        "available_af",
        "spill_af",
        "spill_cfs",
    )
    _STEP_BOOL_INFO_FIELDS = (
        "spr_window_active",
        "spr_proxy_window_active",
        "spr_proxy_target_reachable",
        "spr_proxy_target_hit",
        "spr_proxy_priority_scaled",
    )

    def __init__(
        self,
        *,
        start_update_idx: int = 0,
        flush_path: Path | str | None = None,
        flush_every_updates: int = 10,
        best_total_reward_model_path: Path | str | None = None,
        best_total_reward_metadata_path: Path | str | None = None,
        reward_balancing: str = "none",
        prior_updates: pd.DataFrame | None = None,
        rich_diagnostics: bool = False,
        step_diagnostics_path: Path | str | None = None,
        component_diagnostics_path: Path | str | None = None,
        event_diagnostics_path: Path | str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.rollout_sums: dict[str, float] = {}
        self.rollout_reward_sum: float = 0.0
        self.rollout_count: int = 0
        self.component_abs_sum: float = 0.0
        self.spr_component_seen: bool = False
        self.spr_reward_sum: float = 0.0
        self.spr_reward_abs_sum: float = 0.0
        self.spr_reward_positive_sum: float = 0.0
        self.spr_reward_negative_sum: float = 0.0
        self.spr_reward_nonzero_steps: int = 0
        self.spr_reward_positive_steps: int = 0
        self.spr_reward_negative_steps: int = 0
        self.spr_window_steps: int = 0
        self.spr_window_reward_sum: float = 0.0
        self.spr_opportunity_steps: int = 0
        self.spr_opportunity_reward_sum: float = 0.0
        self.spr_reachable_opportunity_steps: int = 0
        self.spr_reachable_opportunity_reward_sum: float = 0.0
        self.update_history: list[dict] = []
        self._update_idx: int = 0
        self._start_update_idx: int = int(start_update_idx)
        self.flush_path = Path(flush_path) if flush_path is not None else None
        self.flush_every_updates = max(int(flush_every_updates), 1)
        self.best_total_reward_model_path = (
            Path(best_total_reward_model_path)
            if best_total_reward_model_path is not None
            else None
        )
        self.best_total_reward_metadata_path = (
            Path(best_total_reward_metadata_path)
            if best_total_reward_metadata_path is not None
            else None
        )
        self.best_mean_total_reward: float = float("-inf")
        self.best_total_reward_record: dict[str, object] | None = None
        self.reward_balancing = str(reward_balancing)
        self._prior_updates = (
            prior_updates.copy()
            if prior_updates is not None and not prior_updates.empty
            else None
        )
        self.rich_diagnostics = bool(rich_diagnostics)
        self.step_diagnostics_path = (
            Path(step_diagnostics_path)
            if step_diagnostics_path is not None
            else None
        )
        self.component_diagnostics_path = (
            Path(component_diagnostics_path)
            if component_diagnostics_path is not None
            else None
        )
        self.event_diagnostics_path = (
            Path(event_diagnostics_path)
            if event_diagnostics_path is not None
            else None
        )
        self.rollout_component_values: dict[str, list[float]] = {}
        self._step_records: list[dict[str, object]] = []
        self.component_update_history: list[dict[str, object]] = []
        self.event_update_history: list[dict[str, object]] = []
        self._step_writer: Any | None = None
        self._step_schema: Any | None = None
        self._step_tmp_path: Path | None = None
        self._closed = False

    def _on_training_start(self) -> None:
        self.rollout_sums = {}
        self.rollout_reward_sum = 0.0
        self.rollout_count = 0
        self._reset_spr_diagnostics()
        self.update_history = []
        self._update_idx = int(self._start_update_idx)
        self.best_mean_total_reward = float("-inf")
        self.best_total_reward_record = None
        self.rollout_component_values = {}
        self._step_records = []
        self.component_update_history = []
        self.event_update_history = []
        self._closed = False
        if self.rich_diagnostics and self.step_diagnostics_path is not None:
            self._step_tmp_path = self.step_diagnostics_path.with_suffix(
                self.step_diagnostics_path.suffix + ".inprogress"
            )
            existing = [
                path
                for path in (self.step_diagnostics_path, self._step_tmp_path)
                if path.exists()
            ]
            if existing:
                joined = ", ".join(str(path) for path in existing)
                raise FileExistsError(
                    "Rich step diagnostics will not overwrite an existing file: "
                    f"{joined}"
                )

    @classmethod
    def _is_spr_component(cls, key: object) -> bool:
        return str(key).startswith(cls.SPR_COMPONENT_PREFIX)

    @staticmethod
    def _truthy(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        try:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(float(value))
        except Exception:
            return bool(value)

    @classmethod
    def _spr_window_active(cls, info: dict) -> bool:
        return cls._truthy(info.get("spr_window_active", False))

    @classmethod
    def _spr_opportunity_active(cls, info: dict) -> bool:
        if not cls._spr_window_active(info):
            return False
        need_keys = [key for key in info if str(key).startswith("spr_need_more_")]
        if not need_keys:
            return True
        return any(cls._truthy(info.get(key)) for key in need_keys)

    @classmethod
    def _spr_reachable_opportunity_active(cls, info: dict) -> bool:
        if not cls._spr_opportunity_active(info):
            return False
        reachable_keys = [key for key in info if str(key).startswith("spr_reachable_")]
        if not reachable_keys:
            return False
        return any(cls._truthy(info.get(key)) for key in reachable_keys)

    def _reset_spr_diagnostics(self) -> None:
        self.component_abs_sum = 0.0
        self.spr_component_seen = False
        self.spr_reward_sum = 0.0
        self.spr_reward_abs_sum = 0.0
        self.spr_reward_positive_sum = 0.0
        self.spr_reward_negative_sum = 0.0
        self.spr_reward_nonzero_steps = 0
        self.spr_reward_positive_steps = 0
        self.spr_reward_negative_steps = 0
        self.spr_window_steps = 0
        self.spr_window_reward_sum = 0.0
        self.spr_opportunity_steps = 0
        self.spr_opportunity_reward_sum = 0.0
        self.spr_reachable_opportunity_steps = 0
        self.spr_reachable_opportunity_reward_sum = 0.0

    @staticmethod
    def _float_or_nan(value: object) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return result if np.isfinite(result) else float("nan")

    @staticmethod
    def _top_abs_mass_share(values: np.ndarray, fraction: float) -> float:
        abs_values = np.abs(np.asarray(values, dtype=float))
        total = float(np.sum(abs_values))
        if total <= 0.0 or abs_values.size == 0:
            return float("nan")
        n_top = max(1, int(np.ceil(float(fraction) * abs_values.size)))
        return float(np.sum(np.partition(abs_values, -n_top)[-n_top:]) / total)

    def _make_step_record(
        self,
        *,
        info: dict,
        comps: dict,
        reward_total: float,
        action: np.ndarray | None,
        spr_step_reward: float,
        info_idx: int,
    ) -> dict[str, object]:
        date_value = info.get("date")
        try:
            date_text = pd.Timestamp(date_value).isoformat()
        except Exception:
            date_text = str(date_value) if date_value is not None else None

        record: dict[str, object] = {
            "update_idx": int(self._update_idx),
            "global_step": int(self.num_timesteps - max(len(self.locals.get("infos", [])), 1) + info_idx + 1),
            "rollout_step": int(len(self._step_records)),
            "date": date_text,
            "reward_total": float(reward_total),
            "spr_reward": float(spr_step_reward),
            "spr_opportunity_active": bool(self._spr_opportunity_active(info)),
            "spr_reachable_opportunity_active": bool(
                self._spr_reachable_opportunity_active(info)
            ),
        }
        for key in self._STEP_FLOAT_INFO_FIELDS:
            record[key] = self._float_or_nan(info.get(key))
        for key in self._STEP_BOOL_INFO_FIELDS:
            record[key] = bool(self._truthy(info.get(key, False)))
        for key, value in comps.items():
            record[f"reward_{key}"] = self._float_or_nan(value)

        if action is not None:
            for idx, value in enumerate(np.asarray(action, dtype=float).reshape(-1)):
                record[f"action_{idx}"] = float(value)

        requests = {
            "esa": self._float_or_nan(info.get("esa_base_request_sj_cfs")),
            "spr": self._float_or_nan(info.get("spr_proxy_controller_need_cfs")),
            "discretionary": self._float_or_nan(
                info.get("discretionary_sj_request_cfs")
            ),
        }
        finite_requests = {
            key: max(value, 0.0)
            for key, value in requests.items()
            if np.isfinite(value)
        }
        max_request = max(finite_requests.values(), default=0.0)
        winners = [
            key
            for key, value in finite_requests.items()
            if max_request > self._SPR_EPS and np.isclose(value, max_request)
        ]
        record["sj_request_driver"] = "+".join(winners) if winners else "none"
        for key in requests:
            record[f"sj_driver_{key}"] = bool(key in winners)

        spr_need = requests["spr"]
        sj_release = self._float_or_nan(info.get("release_sj_main_cfs"))
        record["spr_bridge_fulfillment_frac"] = (
            float(np.clip(sj_release / spr_need, 0.0, 1.0))
            if np.isfinite(spr_need)
            and spr_need > self._SPR_EPS
            and np.isfinite(sj_release)
            else float("nan")
        )
        niip_request = self._float_or_nan(info.get("requested_release_niip_cfs"))
        niip_release = self._float_or_nan(info.get("release_niip_cfs"))
        record["niip_fulfillment_frac"] = (
            float(np.clip(niip_release / niip_request, 0.0, 1.0))
            if np.isfinite(niip_request)
            and niip_request > self._SPR_EPS
            and np.isfinite(niip_release)
            else float("nan")
        )
        return record

    def _record_rich_update(self) -> None:
        if not self.rich_diagnostics:
            return

        total_component_abs = float(
            sum(np.sum(np.abs(values)) for values in self.rollout_component_values.values())
        )
        for component, raw_values in self.rollout_component_values.items():
            values = np.asarray(raw_values, dtype=float)
            abs_values = np.abs(values)
            positive = values > self._SPR_EPS
            negative = values < -self._SPR_EPS
            self.component_update_history.append(
                {
                    "update_idx": int(self._update_idx),
                    "timesteps": int(self.num_timesteps),
                    "component": str(component),
                    "steps": int(values.size),
                    "reward_sum": float(np.sum(values)),
                    "reward_mean": float(np.mean(values)),
                    "reward_abs_sum": float(np.sum(abs_values)),
                    "reward_abs_mean": float(np.mean(abs_values)),
                    "reward_abs_share": (
                        float(np.sum(abs_values) / total_component_abs)
                        if total_component_abs > self._SPR_EPS
                        else float("nan")
                    ),
                    "positive_steps": int(np.sum(positive)),
                    "negative_steps": int(np.sum(negative)),
                    "positive_sum": float(np.sum(values[positive])),
                    "negative_sum": float(np.sum(values[negative])),
                    "max_reward": float(np.max(values)),
                    "min_reward": float(np.min(values)),
                    "max_abs_reward": float(np.max(abs_values)),
                    "q95_abs_reward": float(np.quantile(abs_values, 0.95)),
                    "q99_abs_reward": float(np.quantile(abs_values, 0.99)),
                    "top_1pct_abs_mass_share": self._top_abs_mass_share(values, 0.01),
                    "top_5pct_abs_mass_share": self._top_abs_mass_share(values, 0.05),
                }
            )

        n_records = len(self._step_records)
        if n_records == 0:
            return
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return
        advantages = np.asarray(rollout_buffer.advantages, dtype=float).reshape(-1)
        returns = np.asarray(rollout_buffer.returns, dtype=float).reshape(-1)
        n = min(n_records, advantages.size, returns.size)
        if n <= 0:
            return
        advantages = advantages[:n]
        returns = returns[:n]
        advantage_std = float(np.std(advantages))
        advantage_z = (
            (advantages - float(np.mean(advantages))) / advantage_std
            if advantage_std > self._SPR_EPS
            else np.zeros_like(advantages)
        )
        for idx, record in enumerate(self._step_records[:n]):
            record["advantage"] = float(advantages[idx])
            record["advantage_z"] = float(advantage_z[idx])
            record["return"] = float(returns[idx])
            record["value_estimate"] = float(returns[idx] - advantages[idx])

        spr_rewards = np.asarray(
            [self._float_or_nan(record.get("spr_reward")) for record in self._step_records[:n]],
            dtype=float,
        )
        group_masks: dict[str, np.ndarray] = {
            "all": np.ones(n, dtype=bool),
            "spr_positive": spr_rewards > self._SPR_EPS,
            "spr_negative": spr_rewards < -self._SPR_EPS,
            "spr_nonzero": np.abs(spr_rewards) > self._SPR_EPS,
            "spr_window": np.asarray(
                [bool(record.get("spr_window_active", False)) for record in self._step_records[:n]]
            ),
            "spr_opportunity": np.asarray(
                [bool(record.get("spr_opportunity_active", False)) for record in self._step_records[:n]]
            ),
            "spr_reachable_opportunity": np.asarray(
                [bool(record.get("spr_reachable_opportunity_active", False)) for record in self._step_records[:n]]
            ),
            "spr_priority_scaled": np.asarray(
                [bool(record.get("spr_proxy_priority_scaled", False)) for record in self._step_records[:n]]
            ),
        }
        for driver in ("esa", "spr", "discretionary"):
            group_masks[f"sj_driver_{driver}"] = np.asarray(
                [bool(record.get(f"sj_driver_{driver}", False)) for record in self._step_records[:n]]
            )

        total_rewards = np.asarray(
            [self._float_or_nan(record.get("reward_total")) for record in self._step_records[:n]],
            dtype=float,
        )
        for group, mask in group_masks.items():
            if not np.any(mask):
                continue
            self.event_update_history.append(
                {
                    "update_idx": int(self._update_idx),
                    "timesteps": int(self.num_timesteps),
                    "event_group": group,
                    "steps": int(np.sum(mask)),
                    "step_fraction": float(np.mean(mask)),
                    "advantage_mean": float(np.mean(advantages[mask])),
                    "advantage_median": float(np.median(advantages[mask])),
                    "advantage_q90": float(np.quantile(advantages[mask], 0.90)),
                    "advantage_z_mean": float(np.mean(advantage_z[mask])),
                    "return_mean": float(np.mean(returns[mask])),
                    "reward_total_mean": float(np.mean(total_rewards[mask])),
                    "spr_reward_mean": float(np.mean(spr_rewards[mask])),
                }
            )

    def _write_step_records(self) -> None:
        if not self.rich_diagnostics or not self._step_records:
            return
        if self._step_tmp_path is None:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._step_tmp_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self._step_records)
        if self._step_writer is None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            self._step_schema = table.schema
            self._step_writer = pq.ParquetWriter(
                self._step_tmp_path,
                self._step_schema,
                compression="zstd",
            )
        else:
            frame = frame.reindex(columns=self._step_schema.names)
            table = pa.Table.from_pandas(
                frame,
                schema=self._step_schema,
                preserve_index=False,
                safe=False,
            )
        self._step_writer.write_table(table)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        actions = self.locals.get("actions", None)
        reward_values = (
            np.asarray(rewards, dtype=float).reshape(-1)
            if rewards is not None
            else np.asarray([], dtype=float)
        )
        action_values = (
            np.asarray(actions, dtype=float)
            if actions is not None
            else None
        )

        if rewards is not None:
            try:
                self.rollout_reward_sum += float(np.sum(rewards))
            except Exception:
                self.rollout_reward_sum += float(rewards)

        if infos:
            for info_idx, info in enumerate(infos):
                comps = info.get("reward_components_step", info.get("reward_components", {}))
                if not comps:
                    continue
                spr_step_reward = 0.0
                for k, v in comps.items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    self.rollout_sums[k] = self.rollout_sums.get(k, 0.0) + fv
                    self.component_abs_sum += abs(fv)
                    if self.rich_diagnostics:
                        self.rollout_component_values.setdefault(str(k), []).append(fv)
                    if self._is_spr_component(k):
                        self.spr_component_seen = True
                        spr_step_reward += fv

                if self.spr_component_seen:
                    self.spr_reward_sum += spr_step_reward
                    self.spr_reward_abs_sum += abs(spr_step_reward)
                    if spr_step_reward > self._SPR_EPS:
                        self.spr_reward_positive_sum += spr_step_reward
                        self.spr_reward_positive_steps += 1
                        self.spr_reward_nonzero_steps += 1
                    elif spr_step_reward < -self._SPR_EPS:
                        self.spr_reward_negative_sum += spr_step_reward
                        self.spr_reward_negative_steps += 1
                        self.spr_reward_nonzero_steps += 1

                    if self._spr_window_active(info):
                        self.spr_window_steps += 1
                        self.spr_window_reward_sum += spr_step_reward
                    if self._spr_opportunity_active(info):
                        self.spr_opportunity_steps += 1
                        self.spr_opportunity_reward_sum += spr_step_reward
                    if self._spr_reachable_opportunity_active(info):
                        self.spr_reachable_opportunity_steps += 1
                        self.spr_reachable_opportunity_reward_sum += spr_step_reward

                if self.rich_diagnostics:
                    action_row = None
                    if action_values is not None:
                        if action_values.ndim <= 1:
                            action_row = action_values
                        elif info_idx < action_values.shape[0]:
                            action_row = action_values[info_idx]
                    reward_total = (
                        float(reward_values[info_idx])
                        if info_idx < reward_values.size
                        else float(sum(float(v) for v in comps.values()))
                    )
                    self._step_records.append(
                        self._make_step_record(
                            info=info,
                            comps=comps,
                            reward_total=reward_total,
                            action=action_row,
                            spr_step_reward=spr_step_reward,
                            info_idx=info_idx,
                        )
                    )

            self.rollout_count += len(infos)
        else:
            self.rollout_count += 1

        return True

    def _on_rollout_end(self) -> None:
        count = max(int(self.rollout_count), 1)

        self._record_rich_update()

        rec: dict[str, float | int] = {
            "update_idx": int(self._update_idx),
            "timesteps": int(self.num_timesteps),
            "rollout_steps": int(self.rollout_count),
            "mean_total_reward": float(self.rollout_reward_sum / count),
        }
        for k, total in self.rollout_sums.items():
            rec[f"mean_{k}"] = float(total / count)

        if self.spr_component_seen:
            rec.update(
                {
                    "spr_reward_sum": float(self.spr_reward_sum),
                    "spr_reward_abs_sum": float(self.spr_reward_abs_sum),
                    "spr_reward_positive_sum": float(self.spr_reward_positive_sum),
                    "spr_reward_negative_sum": float(self.spr_reward_negative_sum),
                    "spr_reward_mean_all_steps": float(self.spr_reward_sum / count),
                    "spr_reward_abs_mean_all_steps": float(
                        self.spr_reward_abs_sum / count
                    ),
                    "spr_reward_abs_share_of_components": (
                        float(self.spr_reward_abs_sum / self.component_abs_sum)
                        if self.component_abs_sum > self._SPR_EPS
                        else np.nan
                    ),
                    "spr_reward_nonzero_steps": int(self.spr_reward_nonzero_steps),
                    "spr_reward_positive_steps": int(self.spr_reward_positive_steps),
                    "spr_reward_negative_steps": int(self.spr_reward_negative_steps),
                    "spr_window_steps": int(self.spr_window_steps),
                    "spr_reward_mean_window_steps": (
                        float(self.spr_window_reward_sum / self.spr_window_steps)
                        if self.spr_window_steps > 0
                        else np.nan
                    ),
                    "spr_opportunity_steps": int(self.spr_opportunity_steps),
                    "spr_reward_mean_opportunity_steps": (
                        float(
                            self.spr_opportunity_reward_sum
                            / self.spr_opportunity_steps
                        )
                        if self.spr_opportunity_steps > 0
                        else np.nan
                    ),
                    "spr_reachable_opportunity_steps": int(
                        self.spr_reachable_opportunity_steps
                    ),
                    "spr_reward_mean_reachable_opportunity_steps": (
                        float(
                            self.spr_reachable_opportunity_reward_sum
                            / self.spr_reachable_opportunity_steps
                        )
                        if self.spr_reachable_opportunity_steps > 0
                        else np.nan
                    ),
                }
            )

        self.update_history.append(rec)
        self._maybe_save_best_total_reward_model(rec)
        self._update_idx += 1
        should_flush = (
            self.flush_path is not None
            and len(self.update_history) % self.flush_every_updates == 0
        )
        if should_flush:
            self.flush()

        self._write_step_records()

        # reset for next rollout
        self.rollout_sums = {}
        self.rollout_reward_sum = 0.0
        self.rollout_count = 0
        self._reset_spr_diagnostics()
        self.rollout_component_values = {}
        self._step_records = []

    def _maybe_save_best_total_reward_model(self, rec: dict[str, float | int]) -> None:
        if self.best_total_reward_model_path is None:
            return
        mean_total_reward = float(rec.get("mean_total_reward", float("nan")))
        if not np.isfinite(mean_total_reward):
            return
        if mean_total_reward <= self.best_mean_total_reward:
            return

        self.best_mean_total_reward = mean_total_reward
        self.best_total_reward_record = dict(rec)
        self.best_total_reward_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.best_total_reward_model_path))
        if self.best_total_reward_metadata_path is not None:
            payload = {
                "selection_metric": "mean_total_reward",
                "selection_scope": "training_rollout",
                "best_mean_total_reward": mean_total_reward,
                "update_idx": int(rec.get("update_idx", -1)),
                "timesteps": int(rec.get("timesteps", -1)),
                "rollout_steps": int(rec.get("rollout_steps", -1)),
                "model_path": str(self.best_total_reward_model_path.resolve()),
                "saved_at": _now_iso(),
                "record": {
                    str(k): (
                        None
                        if isinstance(v, float) and not np.isfinite(v)
                        else (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
                    )
                    for k, v in rec.items()
                },
            }
            _write_json(self.best_total_reward_metadata_path, payload)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.update_history)
        if self._prior_updates is not None:
            df = pd.concat([self._prior_updates, df], ignore_index=True)
        if df.empty:
            df = pd.DataFrame(
                columns=[
                    "update_idx",
                    "timesteps",
                    "rollout_steps",
                    "mean_total_reward",
                ]
            )
        df["reward_balancing"] = self.reward_balancing
        df["reward_component_space"] = (
            "effective" if self.reward_balancing != "none" else "raw"
        )
        return df

    def component_diagnostics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.component_update_history)

    def event_diagnostics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.event_update_history)

    def flush(self) -> None:
        if self.flush_path is not None:
            df = self.to_dataframe()
            self.flush_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.flush_path, index=False)
            df.to_csv(self.flush_path.with_suffix(".csv"), index=False)
        if self.rich_diagnostics and self.component_diagnostics_path is not None:
            component_df = self.component_diagnostics_dataframe()
            self.component_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            component_df.to_parquet(self.component_diagnostics_path, index=False)
            component_df.to_csv(
                self.component_diagnostics_path.with_suffix(".csv"),
                index=False,
            )
        if self.rich_diagnostics and self.event_diagnostics_path is not None:
            event_df = self.event_diagnostics_dataframe()
            self.event_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            event_df.to_parquet(self.event_diagnostics_path, index=False)
            event_df.to_csv(
                self.event_diagnostics_path.with_suffix(".csv"),
                index=False,
            )

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        if self._step_writer is not None:
            self._step_writer.close()
            self._step_writer = None
        if (
            self._step_tmp_path is not None
            and self._step_tmp_path.exists()
            and self.step_diagnostics_path is not None
        ):
            self._step_tmp_path.replace(self.step_diagnostics_path)
        self._closed = True

    def _on_training_end(self) -> None:
        self.close()
