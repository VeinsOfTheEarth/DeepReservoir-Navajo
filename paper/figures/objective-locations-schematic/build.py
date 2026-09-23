"""Build the objective-and-evaluation-location schematic for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_STEM = "objective-locations-schematic"

COLORS = {
    "text": "#111827",
    "muted": "#64748B",
    "line": "#475569",
    "river": "#2B6CB0",
    "storage": "#1D4ED8",
    "dam": "#111827",
    "niip": "#2CA25F",
    "spr": "#7C3AED",
    "hydro": "#D97706",
    "esa": "#0891B2",
    "flood": "#B91C1C",
    "card": "#FFFFFF",
    "context": "#94A3B8",
    "guide": "#CBD5E1",
    "background": "#F4F7FA",
}


def configure_matplotlib() -> None:
    font_dir = REPO_ROOT / "assets" / "fonts" / "inter"
    for path in font_dir.glob("*.ttf"):
        fm.fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Source Sans 3", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": COLORS["text"],
        }
    )


def add_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    color: str,
    badge: str,
    dashed: bool = False,
    body_size: float = 7.5,
    title_size: float = 8.8,
) -> None:
    """Draw one objective card with a family-colored header and neutral body."""
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=COLORS["card"],
        edgecolor=color,
        linewidth=1.35,
        linestyle=(0, (4, 2)) if dashed else "solid",
        zorder=5,
    )
    ax.add_patch(card)

    header_h = 0.038
    header = FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=color,
        edgecolor=color,
        linewidth=0,
        zorder=6,
    )
    ax.add_patch(header)
    # Square off the lower corners of the rounded header.
    ax.add_patch(
        plt.Rectangle(
            (x, y + h - header_h),
            w,
            header_h * 0.48,
            facecolor=color,
            edgecolor="none",
            zorder=6,
        )
    )

    ax.text(
        x + 0.012,
        y + h - header_h / 2 + 0.001,
        title,
        ha="left",
        va="center",
        color="white",
        fontsize=title_size,
        fontweight=700,
        zorder=7,
    )

    badge_width = min(0.096, max(0.062, 0.0065 * len(badge)))
    badge_x = x + w - badge_width - 0.006
    badge_patch = FancyBboxPatch(
        (badge_x, y + h + 0.006),
        badge_width,
        0.024,
        boxstyle="round,pad=0.002,rounding_size=0.007",
        facecolor="#FFFFFF",
        edgecolor=color,
        linewidth=0.8,
        alpha=1.0,
        zorder=8,
    )
    ax.add_patch(badge_patch)
    ax.text(
        badge_x + badge_width / 2,
        y + h + 0.018,
        badge,
        ha="center",
        va="center",
        color=color,
        fontsize=6.2,
        fontweight=750,
        zorder=9,
    )

    n = len(lines)
    body_top = y + h - header_h - 0.014
    body_bottom = y + 0.012
    if n == 1:
        ys = [(body_top + body_bottom) / 2]
    else:
        step = (body_top - body_bottom) / n
        ys = [body_top - (i + 0.5) * step for i in range(n)]
    for i, (line, yy) in enumerate(zip(lines, ys)):
        ax.text(
            x + w / 2,
            yy,
            line,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["muted"] if i == n - 1 else COLORS["text"],
            fontweight=600 if i == 0 else 450,
            zorder=7,
        )


def connector(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str,
    arrow: bool = True,
    linewidth: float = 1.25,
) -> None:
    """Draw an orthogonal or diagonal objective-to-location connector."""
    for start, end in zip(points[:-1], points[1:]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=linewidth,
            alpha=0.82,
            solid_capstyle="round",
            zorder=2,
        )
    if arrow:
        start = points[-2]
        end = points[-1]
        arrow_patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=1.8,
            zorder=3,
        )
        ax.add_patch(arrow_patch)


def draw_system(ax: plt.Axes) -> dict[str, tuple[float, float]]:
    """Draw a compact, nongeographic San Juan operational network."""
    y = 0.49
    locations = {
        "powell": (0.075, y),
        "bluff": (0.205, y),
        "four_corners": (0.335, y),
        "shiprock": (0.455, y),
        "farmington": (0.585, y),
        "archuleta": (0.705, y),
        "dam": (0.817, y),
        "reservoir": (0.905, y),
        "niip": (0.925, 0.415),
        "animas": (0.585, 0.415),
    }

    # Main stem, with the arrow pointing downstream toward Lake Powell.
    ax.plot(
        [locations["powell"][0], locations["reservoir"][0]],
        [y, y],
        color=COLORS["river"],
        linewidth=3.0,
        solid_capstyle="round",
        zorder=3,
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.095, y),
            (0.071, y),
            arrowstyle="-|>",
            mutation_scale=13,
            color=COLORS["river"],
            linewidth=0,
            zorder=4,
        )
    )

    # Tributary and diversion branches.
    ax.plot(
        [locations["animas"][0], locations["farmington"][0]],
        [locations["animas"][1], locations["farmington"][1]],
        color=COLORS["river"],
        linewidth=2.2,
        zorder=3,
    )
    ax.plot(
        [locations["reservoir"][0], locations["niip"][0]],
        [locations["reservoir"][1], locations["niip"][1]],
        color=COLORS["niip"],
        linewidth=2.2,
        zorder=3,
    )

    # Lake Powell endpoint.
    lake = FancyBboxPatch(
        (0.015, y - 0.034),
        0.105,
        0.068,
        boxstyle="round,pad=0.004,rounding_size=0.032",
        facecolor="#DCEEFE",
        edgecolor=COLORS["river"],
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(lake)
    ax.text(
        0.0675,
        y,
        "Lake Powell",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight=650,
        color=COLORS["river"],
        zorder=5,
    )

    # Reservoir endpoint and dam symbol.
    reservoir = FancyBboxPatch(
        (0.87, y - 0.038),
        0.105,
        0.076,
        boxstyle="round,pad=0.004,rounding_size=0.036",
        facecolor="#BBD7F0",
        edgecolor=COLORS["storage"],
        linewidth=1.35,
        zorder=4,
    )
    ax.add_patch(reservoir)
    ax.text(
        0.9225,
        y,
        "Navajo\nReservoir",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight=700,
        color=COLORS["storage"],
        linespacing=0.95,
        zorder=5,
    )
    ax.plot([0.835, 0.835], [y - 0.041, y + 0.041], color=COLORS["text"], lw=3.0, zorder=5)
    ax.text(0.835, y - 0.057, "Navajo Dam", ha="center", va="top", fontsize=7.2, color=COLORS["muted"])

    gages = ["bluff", "four_corners", "shiprock", "farmington", "archuleta"]
    for key in gages:
        x0, y0 = locations[key]
        is_eval = key in {"bluff", "farmington"}
        ax.scatter(
            [x0],
            [y0],
            s=48 if is_eval else 34,
            facecolor="white",
            edgecolor=COLORS["text"] if is_eval else COLORS["context"],
            linewidth=1.2 if is_eval else 1.0,
            zorder=6,
        )

    labels = {
        "bluff": "SJ near\nBluff",
        "four_corners": "SJ at\nFour Corners",
        "shiprock": "SJ at\nShiprock",
        "farmington": "SJ at\nFarmington",
        "archuleta": "SJ near\nArchuleta",
    }
    for key, label in labels.items():
        x0, _ = locations[key]
        ax.text(
            x0,
            y - 0.054,
            label,
            ha="center",
            va="top",
            fontsize=6.9,
            color=COLORS["text"] if key in {"bluff", "farmington"} else COLORS["muted"],
            fontweight=650 if key in {"bluff", "farmington"} else 450,
            linespacing=0.96,
            zorder=6,
        )

    # Branch labels use short horizontal offsets to avoid the main-stem labels.
    ax.scatter(
        [locations["animas"][0]],
        [locations["animas"][1]],
        s=32,
        facecolor="white",
        edgecolor=COLORS["context"],
        linewidth=1.0,
        zorder=5,
    )
    ax.text(
        locations["animas"][0] - 0.030,
        locations["animas"][1] - 0.020,
        "Animas at Farmington",
        ha="right",
        va="top",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    ax.text(
        locations["niip"][0] - 0.012,
        locations["niip"][1] - 0.010,
        "NIIP diversion",
        ha="right",
        va="top",
        fontsize=6.8,
        color=COLORS["niip"],
        fontweight=600,
    )

    ax.text(
        0.50,
        y + 0.027,
        "San Juan River",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=COLORS["river"],
        fontweight=650,
    )
    ax.text(
        0.13,
        y + 0.021,
        "downstream",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=COLORS["muted"],
    )

    return locations


def build() -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    locations = draw_system(ax)

    # Top cards: objective constraints and the model-enabling storage target.
    add_card(
        ax,
        x=0.045,
        y=0.745,
        w=0.285,
        h=0.175,
        title="Flood safety",
        lines=[
            "Keep discharge below both caps",
            "SJ at Farmington ≤ 5,000 cfs",
            "SJ near Bluff ≤ 12,000 cfs",
        ],
        color=COLORS["flood"],
        badge="Priority 3",
    )
    add_card(
        ax,
        x=0.365,
        y=0.745,
        w=0.205,
        h=0.175,
        title="ESA minimum flow",
        lines=[
            "Maintain the daily minimum",
            "SJ at Farmington ≥ 500 cfs",
        ],
        color=COLORS["esa"],
        badge="Priority 2",
    )
    add_card(
        ax,
        x=0.605,
        y=0.745,
        w=0.185,
        h=0.175,
        title="Storage",
        lines=[
            "Maintain carryover storage",
            "87.5% of maximum target",
            "98% soft upper guard",
        ],
        color=COLORS["storage"],
        badge="Model-enabling",
        dashed=True,
        body_size=7.25,
    )
    add_card(
        ax,
        x=0.825,
        y=0.745,
        w=0.14,
        h=0.175,
        title="Dam safety",
        lines=[
            "Avoid spill",
            "Spill = 0",
            "Navajo Reservoir",
        ],
        color=COLORS["dam"],
        badge="Priority 1",
        body_size=7.25,
    )

    # Bottom cards: ecological event targets, power, and annual agricultural delivery.
    add_card(
        ax,
        x=0.235,
        y=0.075,
        w=0.355,
        h=0.245,
        title="Spring peak release",
        lines=[
            "Match annual event-attainment frequencies",
            "10,000 cfs / 5 d: 20%    8,000 cfs / 10 d: 33%",
            "5,000 cfs / 21 d: 50%    2,500 cfs / 10 d: 80%",
            "SJ at Farmington",
        ],
        color=COLORS["spr"],
        badge="Priority 2",
        body_size=7.15,
    )
    add_card(
        ax,
        x=0.625,
        y=0.075,
        w=0.17,
        h=0.205,
        title="Hydropower",
        lines=[
            "Maximize efficient generation",
            "Hydropower /",
            "maximum possible",
            "Navajo Dam",
        ],
        color=COLORS["hydro"],
        badge="Priority 3",
        body_size=7.2,
    )
    add_card(
        ax,
        x=0.825,
        y=0.075,
        w=0.15,
        h=0.205,
        title="NIIP delivery",
        lines=[
            "Meet annual agricultural",
            "delivery volume",
            "Annual volume / contract",
            "≥ 100%",
            "NIIP diversion",
        ],
        color=COLORS["niip"],
        badge="Priority 1",
        body_size=7.0,
        title_size=8.0,
    )

    # Objective-location connectors. Lines are routed away from card text and
    # terminate at the relevant gage, dam, reservoir, or diversion branch.
    flood_y = 0.725
    connector(
        ax,
        [(0.187, 0.745), (0.187, flood_y), (locations["bluff"][0], flood_y), locations["bluff"]],
        color=COLORS["flood"],
    )
    connector(
        ax,
        [(0.187, flood_y), (locations["farmington"][0] - 0.012, flood_y), (locations["farmington"][0] - 0.012, locations["farmington"][1] + 0.006)],
        color=COLORS["flood"],
    )
    connector(
        ax,
        [(0.468, 0.745), (0.468, 0.700), (locations["farmington"][0] + 0.012, 0.700), (locations["farmington"][0] + 0.012, locations["farmington"][1] + 0.006)],
        color=COLORS["esa"],
    )
    connector(
        ax,
        [(0.698, 0.745), (0.698, 0.680), (locations["reservoir"][0] - 0.020, 0.680), (locations["reservoir"][0] - 0.020, locations["reservoir"][1] + 0.026)],
        color=COLORS["storage"],
    )
    connector(
        ax,
        [(0.925, 0.745), (0.925, locations["reservoir"][1] + 0.030)],
        color=COLORS["dam"],
    )
    connector(
        ax,
        [(0.492, 0.320), (0.492, 0.350), (locations["farmington"][0], 0.350), (locations["farmington"][0], locations["farmington"][1] - 0.006)],
        color=COLORS["spr"],
    )
    connector(
        ax,
        [(0.710, 0.280), (0.710, 0.335), (locations["dam"][0], 0.335), (locations["dam"][0], locations["dam"][1] - 0.010)],
        color=COLORS["hydro"],
    )
    connector(
        ax,
        [(0.900, 0.280), (0.900, 0.345), (locations["niip"][0], 0.345), locations["niip"]],
        color=COLORS["niip"],
    )

    return fig


def save(fig: plt.Figure) -> None:
    for suffix in ("png", "pdf", "svg"):
        target = HERE / f"{OUTPUT_STEM}.{suffix}"
        kwargs: dict[str, object] = {
            "facecolor": "white",
            "bbox_inches": None,
            "pad_inches": 0,
        }
        if suffix == "png":
            kwargs["dpi"] = 360
        fig.savefig(target, **kwargs)


if __name__ == "__main__":
    figure = build()
    save(figure)
    plt.close(figure)
    print(f"Wrote {HERE / (OUTPUT_STEM + '.png')}")
