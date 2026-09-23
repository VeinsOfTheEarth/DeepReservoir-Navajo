"""Lightweight visual conventions shared by the paper figure builders."""

from pathlib import Path

import matplotlib.font_manager as fm


ROOT = Path(__file__).resolve().parents[2]
FONT_ROOT = ROOT / "assets" / "fonts"
PANEL_FONT = Path(__file__).resolve().parent / "fonts" / "Inter-Bold.ttf"
PANEL_REFERENCE_WIDTH_IN = 8.4
PANEL_REFERENCE_SIZE_PT = 12.5

TEXT = "#111827"
BACKGROUND = "#F4F7FA"
GRID = "#E2E8F0"
SPINE = "#94A3B8"

OBJECTIVE_COLORS = {
    "storage": "#1D4ED8",
    "esa": "#0891B2",
    "niip": "#2CA25F",
    "hydropower": "#D97706",
    "spr": "#7C3AED",
    "flood": "#B91C1C",
    "dam_safety": "#111827",
}
DISCRETIONARY_COLOR = "#D97706"
SPILL_PRESSURE_COLOR = "#C44536"


def register_fonts() -> None:
    for path in FONT_ROOT.rglob("*.ttf"):
        fm.fontManager.addfont(str(path))


def add_panel_label(
    ax,
    label: str,
    *,
    x: float = 0.02,
    y: float = 0.98,
    ha: str = "left",
    va: str = "top",
    fontsize: float | None = None,
    zorder: int = 20,
):
    if fontsize is None:
        # Match apparent letter size when figures are placed at the same paper width.
        fontsize = PANEL_REFERENCE_SIZE_PT * ax.figure.get_figwidth() / PANEL_REFERENCE_WIDTH_IN
    # An explicit static font avoids variable-font fallback to regular weight.
    font = fm.FontProperties(fname=str(PANEL_FONT), size=fontsize, weight="bold", style="normal")
    return ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontproperties=font,
        color=TEXT,
        zorder=zorder,
        clip_on=False,
    )
