"""Figure-specific helpers extracted without changing plot calculations."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.font_manager as fm

import matplotlib.pyplot as plt

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figure-support"))
from figurestyle import OBJECTIVE_COLORS, SPINE

ROLE_COLORS = {
    **OBJECTIVE_COLORS,
    "spill": OBJECTIVE_COLORS["dam_safety"],
    "background": "#F4F7FA",
    "grid": "#E2E8F0",
    "text": "#111827",
}

def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_axisbelow(True)
