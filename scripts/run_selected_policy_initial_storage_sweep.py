"""Run a selected-policy initial-storage sweep.

The script stores only the compact outputs needed for paper figures: a storage
trajectory table and a summary-metrics table. It does not write full per-scenario
rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepreservoir.define_env.storage_elevation.thresholds import (  # noqa: E402
    get_navajo_storage_thresholds,
)
from deepreservoir.drl import helpers  # noqa: E402
from deepreservoir.drl import metrics as drl_metrics  # noqa: E402
from deepreservoir.drl import model  # noqa: E402
from deepreservoir.drl import selected_policy  # noqa: E402
from deepreservoir.drl.rewards import PRACTICAL_MIN_STORAGE_AF  # noqa: E402


DEFAULT_WINDOW_START = "2014-01-01"
DEFAULT_WINDOW_END = "2024-08-17"
DEFAULT_OUTPUT_DIR = selected_policy.SELECTED_ARTIFACT_DIR / "initial_storage_sweep"
INCREMENT_AF = 100_000.0


def _initial_storage_values_af() -> list[float]:
    max_storage_af = float(get_navajo_storage_thresholds().max_storage_af)
    values = list(np.arange(0.0, np.floor(max_storage_af / INCREMENT_AF) * INCREMENT_AF + 0.5, INCREMENT_AF))
    if not np.isclose(values[-1], max_storage_af):
        values.append(max_storage_af)
    return [float(value) for value in values]


def _load_window(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_data = model.load_all_model_data()
    raw_slice, _ = helpers.slice_by_window(
        all_data["raw"],
        start_token=start,
        end_token=end,
        label="initial_storage_sweep",
    )
    norm_slice = all_data["norm"].loc[raw_slice.index]
    return raw_slice, norm_slice, all_data["norm_stats"]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _spill_metrics(rollout: pd.DataFrame) -> dict[str, float]:
    if "spill_af" not in rollout.columns:
        return {"spill_days": 0.0, "spill_frac_days": 0.0, "total_spill_af": 0.0}
    spill = pd.to_numeric(rollout["spill_af"], errors="coerce").fillna(0.0)
    spill_days = float((spill > 1.0e-9).sum())
    return {
        "spill_days": spill_days,
        "spill_frac_days": spill_days / float(len(spill)) if len(spill) else 0.0,
        "total_spill_af": float(spill.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_WINDOW_START)
    parser.add_argument("--end", default=DEFAULT_WINDOW_END)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_slice, norm_slice, norm_stats = _load_window(args.start, args.end)
    storage_values = _initial_storage_values_af()

    trajectory_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for initial_storage_af in storage_values:
        initial_storage_maf = initial_storage_af / 1_000_000.0
        label = f"{initial_storage_maf:.3f} MAF"
        print(f"[initial-storage-sweep] running {label}")
        rollout = model.run_rollout_data(
            model_path=selected_policy.SELECTED_MODEL_PATH,
            reward_spec=selected_policy.SELECTED_REWARD_SPEC,
            data_raw=raw_slice,
            data_norm=norm_slice,
            norm_stats=norm_stats,
            device=args.device,
            reset_options={"initial_storage_af": initial_storage_af},
            max_release_sj_main_cfs=5000.0,
            max_release_niip_cfs=2500.0,
            obs_context=selected_policy.SELECTED_OBS_CONTEXT,
            action_scaling="linear",
            action_mode=selected_policy.SELECTED_ACTION_MODE,
            reward_balancing="none",
            esa_min_flow_floor=False,
            spr_proxy_priority_release=True,
            spr_proxy_owns_sj_window=False,
            spr_advice_mode=selected_policy.SELECTED_SPR_ADVICE_MODE,
        )
        rollout = rollout.copy()
        rollout["min_storage_af"] = float(PRACTICAL_MIN_STORAGE_AF)

        storage_col = "storage_agent_af_end"
        traj = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(rollout.index),
                "initial_storage_af": initial_storage_af,
                "initial_storage_maf": initial_storage_maf,
                "storage_af": pd.to_numeric(rollout[storage_col], errors="coerce").to_numpy(dtype=float),
            }
        )
        traj["storage_maf"] = traj["storage_af"] / 1_000_000.0
        trajectory_frames.append(traj)

        metrics = drl_metrics.compute_metrics(rollout, which="core", validate=True).iloc[0].to_dict()
        metrics.update(_spill_metrics(rollout))
        metrics.update(
            {
                "initial_storage_af": initial_storage_af,
                "initial_storage_maf": initial_storage_maf,
                "scenario": f"init_storage_{initial_storage_maf:.1f}maf",
            }
        )
        metric_rows.append(metrics)

    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    summary = pd.DataFrame(metric_rows)

    trajectories.to_parquet(outdir / "initial_storage_sweep_storage.parquet", index=False)
    trajectories.to_csv(outdir / "initial_storage_sweep_storage.csv", index=False)
    summary.to_csv(outdir / "initial_storage_sweep_summary.csv", index=False)
    _write_json(outdir / "initial_storage_sweep_summary.json", summary.to_dict(orient="records"))
    _write_json(
        outdir / "initial_storage_sweep_metadata.json",
        {
            "selected_policy": {
                "public_name": selected_policy.SELECTED_PUBLIC_NAME,
                "legacy_family": selected_policy.SELECTED_LEGACY_FAMILY,
                "legacy_seed": selected_policy.SELECTED_LEGACY_SEED,
                "model_path": str(selected_policy.SELECTED_MODEL_PATH.relative_to(REPO_ROOT)),
                "reward_spec": selected_policy.SELECTED_REWARD_SPEC,
                "obs_context": selected_policy.SELECTED_OBS_CONTEXT,
                "action_mode": selected_policy.SELECTED_ACTION_MODE,
                "spr_advice_mode": selected_policy.SELECTED_SPR_ADVICE_MODE,
            },
            "sweep": {
                "window_start": args.start,
                "window_end": args.end,
                "increment_af": INCREMENT_AF,
                "max_storage_af": float(get_navajo_storage_thresholds().max_storage_af),
                "n_initial_storage_values": len(storage_values),
            },
        },
    )
    print(f"Wrote initial-storage sweep artifacts to {outdir}")
    print(summary[["initial_storage_maf", "storage_frac_of_max_possible", "esa_min_flow_frac_days_met", "hydropower_frac_of_max_possible", "spill_days"]].to_string(index=False))


if __name__ == "__main__":
    main()
