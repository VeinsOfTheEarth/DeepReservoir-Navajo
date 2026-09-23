"""Build corrected selected-policy training reward diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SRC_ROOT = REPO_ROOT / "src"
SUPPORT_ROOT = HERE.parent.parent / "figure-support"
for path in (SRC_ROOT, SUPPORT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepreservoir.drl import selected_policy  # noqa: E402
from paperstyle import (  # noqa: E402
    DAM_COLOR, ESA_COLOR, FLOOD_COLOR, HYDRO_COLOR, NIIP_COLOR,
    SPR_COLOR, STORAGE_COLOR, TEXT, MUTED, add_panel_label,
    save_figure, set_theme, soften_axes,
)


ROLLING_UPDATES = 25
TRAIN_PATH = selected_policy.SELECTED_ARTIFACT_DIR / "train_update_metrics.csv"
ROLLOUT_PATH = selected_policy.SELECTED_EVAL_ROLLOUT_PATH
DATA_DIR = HERE / "data"

COMPONENTS = (
    {"key": "dam_safety", "label": "Dam safety", "column": "mean_dam_safety.spill_guard_warn98", "color": DAM_COLOR},
    {"key": "storage", "label": "Storage", "column": "mean_storage_control.target_oishift875to90_concave0_softupper98", "color": STORAGE_COLOR},
    {"key": "hydropower", "label": "Hydropower", "column": "mean_hydropower.positive_efficiency", "color": HYDRO_COLOR},
    {"key": "flood_safety", "label": "Flood safety", "column": "mean_flooding.penalty_caps_archuleta_bluff", "color": FLOOD_COLOR},
    {"key": "niip_delivery", "label": "NIIP delivery", "column": "mean_niip.delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon", "color": NIIP_COLOR},
    {"key": "esa_min_flow", "label": "ESA min flow", "column": "mean_esa_min_flow.green_logistic_jon", "color": ESA_COLOR},
    {"key": "spring_peak_release", "label": "SPR", "column": "mean_esa_spring_peak_release.farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar", "color": SPR_COLOR},
)
SPR_ROLLOUT_COLUMN = "rc_esa_spring_peak_release.farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar"


def rolling(values: pd.Series) -> pd.Series:
    return values.rolling(ROLLING_UPDATES, center=True, min_periods=1).mean()


def main() -> None:
    set_theme()
    updates = pd.read_csv(TRAIN_PATH).sort_values("timesteps").reset_index(drop=True)
    rollout = pd.read_parquet(ROLLOUT_PATH)
    x = pd.to_numeric(updates["timesteps"], errors="raise") / 1_000_000.0

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    total = pd.to_numeric(updates["mean_total_reward"], errors="raise")
    ax_a.plot(x, total, color=TEXT, alpha=0.16, lw=0.55)
    ax_a.plot(x, rolling(total), color=TEXT, lw=1.9, label="Total")
    for component in COMPONENTS:
        values = pd.to_numeric(updates[component["column"]], errors="raise")
        ax_a.plot(
            x,
            rolling(values),
            color=component["color"],
            lw=1.15,
            label=component["label"],
        )
    ax_a.axhline(0.0, color=MUTED, lw=0.8, ls=":")
    ax_a.set_xlabel("Training steps (millions)")
    ax_a.set_ylabel("Mean signed reward per step")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=False, fontsize=7.0)
    add_panel_label(ax_a, "A")
    soften_axes(ax_a)

    steps = pd.to_numeric(updates["rollout_steps"], errors="raise")
    positive_pct = 100.0 * pd.to_numeric(updates["spr_reward_positive_steps"], errors="raise") / steps
    nonzero_pct = 100.0 * pd.to_numeric(updates["spr_reward_nonzero_steps"], errors="raise") / steps
    absolute_share = 100.0 * pd.to_numeric(updates["spr_reward_abs_share_of_components"], errors="raise")
    ax_b.plot(x, rolling(absolute_share), color=SPR_COLOR, lw=1.8, label="Absolute reward share")
    ax_b.plot(x, rolling(nonzero_pct), color="#6B7280", lw=1.3, label="Nonzero-reward days")
    ax_b.plot(x, rolling(positive_pct), color="#D99A00", lw=1.3, label="Positive-reward days")
    ax_b.set_xlabel("Training steps (millions)")
    ax_b.set_ylabel("SPR signal (% per update)")
    ax_b.set_ylim(bottom=0.0)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=1, frameon=False, fontsize=7.2)
    add_panel_label(ax_b, "B")
    soften_axes(ax_b)

    positive_sum = pd.to_numeric(updates["spr_reward_positive_sum"], errors="raise").to_numpy(float)
    positive_steps = pd.to_numeric(updates["spr_reward_positive_steps"], errors="raise").to_numpy(float)
    positive_mean = np.divide(positive_sum, positive_steps, out=np.full_like(positive_sum, np.nan), where=positive_steps > 0)
    progress = np.linspace(0.0, 1.0, len(updates))
    scatter = ax_c.scatter(positive_pct, positive_mean, c=progress, cmap="viridis", s=13, alpha=0.72, edgecolor="none")
    eval_values = pd.to_numeric(rollout[SPR_ROLLOUT_COLUMN], errors="raise")
    eval_positive = eval_values[eval_values > 0]
    ax_c.scatter(
        [100.0 * float((eval_values > 0).mean())],
        [float(eval_positive.mean())],
        marker="*", s=130, color=TEXT, edgecolor="white", linewidth=0.7,
        label="Selected-policy evaluation", zorder=6,
    )
    ax_c.set_xlabel("Positive SPR-reward days per update (%)")
    ax_c.set_ylabel("Mean positive SPR reward")
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=7.2)
    cbar = fig.colorbar(scatter, ax=ax_c, pad=0.02, fraction=0.05)
    cbar.set_label("Training progress")
    cbar.set_ticks([0.0, 1.0], labels=["Start", "End"])
    add_panel_label(ax_c, "C")
    soften_axes(ax_c)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    updates.to_csv(DATA_DIR / "train-update-metrics.csv", index=False)
    pd.DataFrame({"spring_peak_release": eval_values}).to_csv(
        DATA_DIR / "selected-policy-evaluation-daily-rewards.csv", index=False
    )
    (DATA_DIR / "training-reward-signals-metadata.json").write_text(
        json.dumps({
            "source_family": selected_policy.SELECTED_SOURCE_FAMILY,
            "source_task_id": selected_policy.SELECTED_TASK_ID,
            "seed": selected_policy.SELECTED_SEED,
            "rolling_updates": ROLLING_UPDATES,
            "training_trace": str(TRAIN_PATH.relative_to(REPO_ROOT)),
            "evaluation_rollout": str(ROLLOUT_PATH.relative_to(REPO_ROOT)),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in save_figure(fig, HERE, "training-reward-signals"):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
