"""Run a selected-policy reservoir-inflow scaling sweep.

The sweep scales the evaluation-period reservoir inflow from -50% to +150%
relative to nominal (0.5x to 2.5x), then stores compact storage trajectories and
summary metrics for paper figures.
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

from deepreservoir.drl import helpers  # noqa: E402
from deepreservoir.drl import metrics as drl_metrics  # noqa: E402
from deepreservoir.drl import model  # noqa: E402
from deepreservoir.drl import selected_policy  # noqa: E402


DEFAULT_WINDOW_START = "2014-01-01"
DEFAULT_WINDOW_END = "2024-08-17"
DEFAULT_OUTPUT_DIR = selected_policy.SELECTED_ARTIFACT_DIR / "inflow_scaling_sweep"
MIN_SCALE = 0.5
MAX_SCALE = 2.5
SCALE_STEP = 0.1


def _scale_values() -> list[float]:
    values = np.arange(MIN_SCALE, MAX_SCALE + SCALE_STEP / 2.0, SCALE_STEP)
    return [float(np.round(value, 10)) for value in values]


def _load_window(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_data = model.load_all_model_data()
    raw_slice, _ = helpers.slice_by_window(
        all_data["raw"],
        start_token=start,
        end_token=end,
        label="inflow_scaling_sweep",
    )
    norm_slice = all_data["norm"].loc[raw_slice.index]
    return raw_slice, norm_slice, all_data["norm_stats"]


def _recompute_norm_inflow(
    raw: pd.DataFrame, norm: pd.DataFrame, norm_stats: pd.DataFrame
) -> pd.DataFrame:
    out = norm.copy()
    mean = float(norm_stats.loc["inflow_cfs", "mean"])
    std = float(norm_stats.loc["inflow_cfs", "std"])
    if std == 0.0:
        std = 1.0
    out["inflow_cfs"] = (
        (pd.to_numeric(raw["inflow_cfs"], errors="coerce") - mean) / std
    ).astype(float)
    return out


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
    scale_values = _scale_values()

    trajectory_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for inflow_scale in scale_values:
        inflow_change_pct = 100.0 * (inflow_scale - 1.0)
        print(
            "[inflow-scaling-sweep] running "
            f"scale={inflow_scale:.1f} ({inflow_change_pct:+.0f}%)"
        )
        raw = raw_slice.copy()
        raw["inflow_cfs"] = (
            pd.to_numeric(raw["inflow_cfs"], errors="coerce").astype(float)
            * inflow_scale
        )
        norm = _recompute_norm_inflow(raw, norm_slice, norm_stats)

        rollout = model.run_rollout_data(
            model_path=selected_policy.SELECTED_MODEL_PATH,
            reward_spec=selected_policy.SELECTED_REWARD_SPEC,
            data_raw=raw,
            data_norm=norm,
            norm_stats=norm_stats,
            device=args.device,
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

        traj = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(rollout.index),
                "inflow_scale": inflow_scale,
                "inflow_change_pct": inflow_change_pct,
                "storage_af": pd.to_numeric(
                    rollout["storage_agent_af_end"], errors="coerce"
                ).to_numpy(dtype=float),
            }
        )
        traj["storage_maf"] = traj["storage_af"] / 1_000_000.0
        trajectory_frames.append(traj)

        metrics = drl_metrics.compute_metrics(rollout, which="core", validate=True).iloc[0].to_dict()
        metrics.update(_spill_metrics(rollout))
        metrics.update(
            {
                "scenario": f"inflow_scale_{inflow_scale:.1f}x",
                "inflow_scale": inflow_scale,
                "inflow_change_pct": inflow_change_pct,
            }
        )
        metric_rows.append(metrics)

    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    summary = pd.DataFrame(metric_rows)

    trajectories.to_parquet(outdir / "inflow_scaling_sweep_storage.parquet", index=False)
    trajectories.to_csv(outdir / "inflow_scaling_sweep_storage.csv", index=False)
    summary.to_csv(outdir / "inflow_scaling_sweep_summary.csv", index=False)
    _write_json(outdir / "inflow_scaling_sweep_summary.json", summary.to_dict(orient="records"))
    _write_json(
        outdir / "inflow_scaling_sweep_metadata.json",
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
                "min_scale": MIN_SCALE,
                "max_scale": MAX_SCALE,
                "scale_step": SCALE_STEP,
                "n_scale_values": len(scale_values),
            },
        },
    )
    print(f"Wrote inflow-scaling sweep artifacts to {outdir}")
    print(
        summary[
            [
                "inflow_scale",
                "storage_frac_of_max_possible",
                "esa_min_flow_frac_days_met",
                "hydropower_frac_of_max_possible",
                "policy_spr_score",
                "spill_days",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
