"""Build a stitched reward-function summary figure for the paper.

This figure intentionally excludes the SPR reward, which is documented with a
separate flowchart and scoring ledger because its event-ledger logic does not
fit naturally in a scalar curve panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


REWARDS_DIR = Path(__file__).resolve().parent
if str(REWARDS_DIR) not in sys.path:
    sys.path.insert(0, str(REWARDS_DIR))

from rewardstyle import ROLE_COLORS, _clean_axes
from figurestyle import SPINE, add_panel_label, register_fonts


OUTPUT_DIR = Path(__file__).resolve().parent


def _add_panel_label(ax: plt.Axes, label: str, *, loc: str = "left") -> None:
    x = 0.98 if loc == "right" else 0.02
    ha = "right" if loc == "right" else "left"
    add_panel_label(ax, label, x=x, ha=ha)


def _set_style() -> None:
    register_fonts()
    preferred = ["Inter", "Source Sans 3", "IBM Plex Sans", "DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": preferred,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "axes.facecolor": ROLE_COLORS["background"],
            "figure.facecolor": "white",
            "axes.edgecolor": SPINE,
            "axes.labelcolor": ROLE_COLORS["text"],
            "xtick.color": ROLE_COLORS["text"],
            "ytick.color": ROLE_COLORS["text"],
            "axes.titlecolor": ROLE_COLORS["text"],
            "axes.grid": True,
            "grid.color": ROLE_COLORS["grid"],
            "grid.alpha": 0.72,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
        }
    )


def _plot_storage(ax: plt.Axes) -> None:
    target = 0.875
    upper_warn = 0.98
    reward_at_warn = 0.5
    reward_at_full = -1.0
    frac = np.linspace(0.0, 1.0, 600)
    reward = np.empty_like(frac)

    lower = frac <= target
    reward[lower] = -1.0 + 2.0 * np.sqrt(np.clip(frac[lower] / target, 0, 1))

    mid = (frac > target) & (frac <= upper_warn)
    reward[mid] = 1.0 + (reward_at_warn - 1.0) * (
        (frac[mid] - target) / (upper_warn - target)
    )

    high = frac > upper_warn
    high_x = np.clip((frac[high] - upper_warn) / (1.0 - upper_warn), 0, 1)
    reward[high] = reward_at_warn + (reward_at_full - reward_at_warn) * high_x**2
    reward = np.clip(reward, -1.5, 1.0)

    ax.plot(frac * 100, reward, color=ROLE_COLORS["storage"], lw=2.0)
    target_line = ax.axvline(
        target * 100,
        color=ROLE_COLORS["storage"],
        lw=1.1,
        ls="--",
        label="Target",
    )
    upper_line = ax.axvline(
        upper_warn * 100,
        color=ROLE_COLORS["spill"],
        lw=1.1,
        ls="--",
        label="Soft upper warning",
    )
    ax.axhline(0, color="#5C6875", lw=1.0, ls=":", alpha=0.9)
    ax.set_title("Storage target")
    ax.set_xlabel("Storage (% max)")
    ax.set_ylabel("Reward")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-1.1, 1.08)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.legend(
        handles=[target_line, upper_line],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#D6DEE6",
    )
    _clean_axes(ax)


def _plot_dam_storage(ax: plt.Axes) -> None:
    frac = np.linspace(0.92, 1.0, 400)
    storage_penalty = np.where(frac <= 0.98, 0.0, -((frac - 0.98) / 0.02) ** 2)
    ax.plot(frac * 100, storage_penalty, color=ROLE_COLORS["dam_safety"], lw=2.0)
    ax.axvline(98, color=ROLE_COLORS["spill"], lw=1.1, ls="--")
    ax.axhline(0, color="#5C6875", lw=1.0, ls=":", alpha=0.9)
    ax.text(97.7, -0.9, "98% warning", color=ROLE_COLORS["spill"], ha="right")
    ax.set_title("Dam safety: storage warning")
    ax.set_xlabel("Storage (% max)")
    ax.set_ylabel("Reward")
    ax.set_xlim(91.8, 100.4)
    ax.set_ylim(-1.08, 0.08)
    _clean_axes(ax)


def _plot_spill(ax: plt.Axes) -> None:
    spill_cfs = np.linspace(0, 15000, 500)
    spill_penalty = np.where(
        spill_cfs <= 0,
        0.0,
        -1.0 - np.minimum(spill_cfs / 2500.0, 5.0),
    )
    ax.plot(spill_cfs, spill_penalty, color=ROLE_COLORS["spill"], lw=2.0)
    ax.axhline(0, color="#5C6875", lw=1.0, ls=":", alpha=0.9)
    ax.set_title("Dam safety: spill")
    ax.set_xlabel("Spill (cfs)")
    ax.set_ylabel("Reward")
    ax.set_xlim(-300, 15300)
    ax.set_ylim(-6.2, 0.2)
    ax.set_xticks([0, 5000, 10000, 15000])
    _clean_axes(ax)


def _plot_esa(ax: plt.Axes) -> None:
    delta = np.linspace(-700, 700, 700)
    z = (delta + 290.0) / 80.0
    reward = -1.65 + 2.65 / (1.0 + np.exp(-z))
    reward = np.clip(reward, -1.65, 1.0)

    ax.plot(delta, reward, color=ROLE_COLORS["esa"], lw=2.0)
    ax.axvline(0, color="#5C6875", lw=1.0, ls="--")
    ax.axhline(0, color="#5C6875", lw=1.0, ls=":", alpha=0.9)
    ax.text(20, -1.43, "500 cfs target", color="#5C6875", va="bottom")
    ax.set_title("ESA minimum flow")
    ax.set_xlabel("Surplus above ESA need (cfs)")
    ax.set_ylabel("Reward")
    ax.set_xlim(-700, 700)
    ax.set_ylim(-1.72, 1.08)
    _clean_axes(ax)


def _plot_flood(ax: plt.Axes) -> None:
    q = np.linspace(0, 17000, 800)

    def cap_component(flow: np.ndarray, lower: float, upper: float) -> np.ndarray:
        component = np.zeros_like(flow)
        active = flow > lower
        component[active] = -np.clip((flow[active] - lower) / (upper - lower), 0, 1)
        return 0.5 * component

    same_day = cap_component(q, 5000, 7000)
    lagged = cap_component(q, 12000, 16000)

    ax.plot(q, same_day, color=ROLE_COLORS["flood"], lw=2.0, label="Archuleta proxy")
    ax.plot(q, lagged, color="#F87171", lw=2.0, ls="-.", label="Bluff gage")
    ax.axvline(
        5000,
        color=ROLE_COLORS["flood"],
        lw=1.0,
        ls="--",
        alpha=0.9,
        label="Archuleta threshold",
    )
    ax.axvline(
        12000,
        color="#F87171",
        lw=1.0,
        ls="--",
        alpha=0.9,
        label="Bluff threshold",
    )
    ax.axhline(0, color="#5C6875", lw=1.0, ls=":", alpha=0.9)
    ax.set_title("Flood safety")
    ax.set_xlabel("San Juan discharge (cfs)")
    ax.set_ylabel("Reward")
    ax.set_xlim(0, 17000)
    ax.set_ylim(-0.56, 0.06)
    ax.set_xticks([0, 5000, 10000, 15000])
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#D6DEE6")
    _clean_axes(ax)


def _plot_hydropower(container_ax: plt.Axes) -> None:
    container_ax.set_title("Hydropower")
    container_ax.set_axis_off()
    ax = container_ax.inset_axes([0.0, 0.0, 0.76, 0.93])
    cax = container_ax.inset_axes([0.84, 0.16, 0.035, 0.66])

    release = np.linspace(0, 5000, 301)
    power_generation = np.linspace(0, 1, 241)
    release_grid, power_grid = np.meshgrid(release, power_generation)

    factor = np.ones_like(release_grid)
    above = release_grid > 1300
    factor[above] = np.sqrt(1300.0 / np.maximum(release_grid[above], 1.0))
    reward = np.clip(power_grid * factor, 0, 1)

    mesh = ax.pcolormesh(
        release,
        power_generation * 100,
        reward,
        shading="auto",
        cmap=LinearSegmentedColormap.from_list(
            "hydropower_reward", ["#FFF7ED", ROLE_COLORS["hydropower"]]
        ),
        vmin=0,
        vmax=1,
    )
    ax.axvline(1300, color="#5C6875", lw=1.0, ls="--")
    ax.text(1370, 7, "turbine-efficient\nband", color="#5C6875", va="bottom")
    ax.set_xlabel("San Juan release above ESA/SPR need (cfs)")
    ax.set_ylabel("Power generation (% max)")
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    cbar = ax.figure.colorbar(mesh, cax=cax)
    cbar.set_label("Reward")
    cbar.outline.set_edgecolor(SPINE)
    _clean_axes(ax)


def _plot_niip_active(ax: plt.Axes) -> None:
    demand = 500.0
    delivered_ratio = np.linspace(0, 2.2, 700)
    release = delivered_ratio * demand
    error = release - demand

    reward = np.empty_like(release)
    under = error < 0
    under_scale = max(175.0, 0.45 * demand)
    reward[under] = -(np.abs(error[under]) / under_scale)

    over = ~under
    over_cfs = error[over]
    full_credit = max(50.0, 0.10 * demand)
    zero_score = max(250.0, 0.45 * demand)
    reward[over] = np.where(
        over_cfs <= full_credit,
        1.0,
        1.0 - ((over_cfs - full_credit) / max(zero_score - full_credit, 1e-6)),
    )
    reward = np.clip(reward, -1.0, 1.0)

    ax.plot(delivered_ratio, reward, color=ROLE_COLORS["niip"], lw=2.0)
    ax.axvline(1.0, color="#5C6875", lw=1.0, ls="--")
    ax.axhline(0, color="#5C6875", lw=1.0, ls=":", alpha=0.9)
    ax.set_title("NIIP active demand")
    ax.set_xlabel("Delivery / daily demand")
    ax.set_ylabel("Reward")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    _clean_axes(ax)


def _plot_niip_no_demand(ax: plt.Axes) -> None:
    no_demand_release = np.linspace(0, 1800, 500)
    no_demand_penalty = -0.25 * np.log1p(no_demand_release / 250.0)
    ax.plot(no_demand_release, no_demand_penalty, color="#1F8A4C", lw=2.0)
    ax.axhline(0, color="#5C6875", lw=1.0, ls=":", alpha=0.9)
    ax.set_title("NIIP no demand")
    ax.set_xlabel("NIIP release (cfs)")
    ax.set_ylabel("Reward")
    ax.set_xlim(0, 1800)
    ax.set_ylim(-0.56, 0.06)
    ax.set_xticks([0, 600, 1200, 1800])
    _clean_axes(ax)


def build_reward_functions_diagrams() -> list[Path]:
    _set_style()
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(8.4, 13.2),
        constrained_layout=True,
    )
    plotters = [
        _plot_storage,
        _plot_esa,
        _plot_dam_storage,
        _plot_spill,
        _plot_niip_active,
        _plot_niip_no_demand,
        _plot_flood,
        _plot_hydropower,
    ]
    labels = list("ABCDEFGH")
    label_locs = {
        "C": "right",
        "H": "right",
    }
    for ax, plotter, label in zip(axes.flat, plotters, labels):
        plotter(ax)
        _add_panel_label(ax, label, loc=label_locs.get(label, "left"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        OUTPUT_DIR / "reward-functions-diagrams.png",
        OUTPUT_DIR / "reward-functions-diagrams.pdf",
    ]
    fig.savefig(paths[0], dpi=320)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def main() -> None:
    for path in build_reward_functions_diagrams():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
