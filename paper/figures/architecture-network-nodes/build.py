"""Build more technical neural-network architecture mockups.

These are intentionally separate from ``build.py`` so the earlier conceptual
mockups are not overwritten while we iterate.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUTDIR = Path(__file__).resolve().parent
REPO_ROOT = OUTDIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "paper" / "figure-support"))
from figurestyle import DISCRETIONARY_COLOR, OBJECTIVE_COLORS, SPILL_PRESSURE_COLOR

FONT_STACK = ["Inter", "Source Sans 3", "IBM Plex Sans", "DejaVu Sans"]

COLORS = {
    "storage": OBJECTIVE_COLORS["storage"],
    "spill_pressure": SPILL_PRESSURE_COLOR,
    "sj_avoid": "#E07A5F",
    "niip": OBJECTIVE_COLORS["niip"],
    "spr": OBJECTIVE_COLORS["spr"],
    "esa": OBJECTIVE_COLORS["esa"],
    "discretionary": DISCRETIONARY_COLOR,
    "critic": "#111827",
    "text": "#111827",
    "muted": "#64748B",
    "spine": "#94A3B8",
    "grid": "#E2E8F0",
    "background": "#F4F7FA",
    "soft": "#F8FAFC",
}

FEATURE_GROUPS = [
    ("Storage\n+ budget", "storage"),
    ("NIIP\ndemand\nproxy", "niip"),
    ("ESA\nrequired\nrelease", "esa"),
    ("Spill\npressure", "spill_pressure"),
    ("SJ spill\navoid", "sj_avoid"),
    ("SPR +\nAnimas\nadvice", "spr"),
]

BRANCHES = [
    {
        "name": "ESA/baseflow",
        "role": "Actor",
        "short": "ESA",
        "color": "esa",
        "visible": ["storage", "niip", "esa", "spill_pressure", "sj_avoid"],
        "partial": [],
        "hidden": "64 -> 64",
        "action": r"$\mu_{\mathrm{ESA}}\rightarrow a_{\mathrm{ESA}}$",
        "interpretation": "ESA/baseflow\nrequest",
    },
    {
        "name": "NIIP",
        "role": "Actor",
        "short": "NIIP",
        "color": "niip",
        "visible": ["storage", "niip", "esa", "spill_pressure"],
        "partial": [],
        "hidden": "64 -> 64",
        "action": r"$\mu_{\mathrm{NIIP}}\rightarrow a_{\mathrm{NIIP}}$",
        "interpretation": "NIIP\nrelease",
    },
    {
        "name": "SPR target",
        "role": "Actor",
        "short": "SPR",
        "color": "spr",
        "visible": ["storage", "niip", "esa", "spill_pressure", "sj_avoid", "spr"],
        "partial": [],
        "hidden": "64 -> 64",
        "action": r"$\mu_{\mathrm{SPR}}\rightarrow a_{\mathrm{SPR}}$",
        "interpretation": "",
    },
    {
        "name": "Discretionary SJ\n(hydropower,\nstorage, flood safety)",
        "role": "Actor",
        "short": "SJ",
        "color": "discretionary",
        "visible": ["storage", "niip", "esa", "spill_pressure", "sj_avoid"],
        "partial": [],
        "hidden": "64 -> 64",
        "action": r"$\mu_{\mathrm{SJ}}\rightarrow a_{\mathrm{SJ}}$",
        "interpretation": "Discretionary\nSJ release",
    },
    {
        "name": "Value",
        "role": "Critic",
        "short": "V",
        "color": "critic",
        "visible": ["storage", "niip", "esa", "spill_pressure", "sj_avoid", "spr"],
        "partial": [],
        "hidden": "64 -> 64",
        "action": r"$V(s)$",
        "interpretation": "PPO value\nestimate",
    },
]


def configure_matplotlib() -> None:
    font_dir = REPO_ROOT / "assets" / "fonts"
    if font_dir.is_dir():
        for font_path in font_dir.rglob("*.ttf"):
            try:
                fm.fontManager.addfont(str(font_path))
            except Exception:
                pass
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": COLORS["text"],
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = stem.replace("_", "-")
    for ext in ("png", "pdf"):
        target = OUTDIR / f"{stem}.{ext}"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / target.name
            fig.savefig(tmp_path, dpi=300, bbox_inches="tight", facecolor="white")
            shutil.copy2(tmp_path, target)
    plt.close(fig)


def setup_ax(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 8.5,
    weight: str = "normal",
    color: str = COLORS["text"],
    ha: str = "center",
    va: str = "center",
    rotation: float = 0,
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        rotation=rotation,
        linespacing=1.12,
        zorder=6,
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    edge: str,
    face: str = "white",
    size: float = 8.5,
    weight: str = "normal",
    radius: float = 0.012,
    lw: float = 1.25,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(patch)
    add_text(ax, x + w / 2, y + h / 2, text, size=size, weight=weight)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 1.1,
    alpha: float = 0.85,
    rad: float = 0.0,
    zorder: int = 2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder,
        )
    )


def arrowhead(ax: plt.Axes, x: float, y: float, *, color: str) -> None:
    ax.scatter(
        [x],
        [y],
        marker=">",
        s=52,
        c=[color],
        edgecolors=color,
        linewidths=0.0,
        zorder=5,
    )


def node(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    color: str,
    size: float = 58,
    edge: str = "white",
    alpha: float = 1.0,
    zorder: int = 4,
    marker: str = "o",
) -> None:
    ax.scatter(
        [x],
        [y],
        s=size,
        c=[color],
        marker=marker,
        edgecolors=edge,
        linewidths=0.75,
        alpha=alpha,
        zorder=zorder,
    )


def draw_branch_network(
    ax: plt.Axes,
    *,
    branch: dict[str, object],
    y: float,
    input_x: float,
    h1_x: float,
    h2_x: float,
    action_x: float,
    action_w: float,
    node_scale: float = 1.0,
) -> None:
    color = COLORS[str(branch["color"])]
    visible = set(branch["visible"])
    partial = set(branch.get("partial", []))
    y_offsets = [0.050, 0.030, 0.010, -0.010, -0.030, -0.050]

    for (label, key), dy in zip(FEATURE_GROUPS, y_offsets):
        c = COLORS[key]
        is_visible = key in visible or key in partial
        alpha = 1.0 if is_visible else 0.10
        node(ax, input_x, y + dy, color=c, size=58 * node_scale, alpha=alpha, marker="s")
        if is_visible:
            for h_y in [y + 0.030, y, y - 0.030]:
                ax.plot(
                    [input_x + 0.010, h1_x - 0.010],
                    [y + dy, h_y],
                    color=c,
                    alpha=0.22,
                    lw=0.8,
                    zorder=1,
                )
    for x in (h1_x, h2_x):
        for h_y in [y + 0.030, y, y - 0.030]:
            node(ax, x, h_y, color=color, size=52 * node_scale, alpha=0.9)

    for h_y1 in [y + 0.030, y, y - 0.030]:
        for h_y2 in [y + 0.030, y, y - 0.030]:
            ax.plot(
                [h1_x + 0.010, h2_x - 0.010],
                [h_y1, h_y2],
                color=color,
                alpha=0.18,
                lw=0.65,
                zorder=1,
            )

    for h_y in [y + 0.030, y, y - 0.030]:
        ax.plot(
            [h2_x + 0.010, action_x - 0.012],
            [h_y, y],
            color=color,
            alpha=0.24,
            lw=0.75,
            zorder=1,
        )
    arrowhead(ax, action_x - 0.009, y, color=color)
    box_x = action_x + 0.016
    rounded_box(
        ax,
        box_x,
        y - 0.034,
        action_w,
        0.068,
        str(branch["action"]),
        edge=color,
        face="white",
        size=7.4,
        weight="bold" if branch["short"] in {"SPR", "V"} else "normal",
    )


def draw_technical_network() -> None:
    fig, ax = setup_ax((11.3, 6.6))

    header_y = 0.850
    add_text(ax, 0.092, header_y, "Branch", size=8.5, weight="bold")
    add_text(ax, 0.232, header_y, "Input mask", size=8.5, weight="bold")
    add_text(ax, 0.370, header_y, "Hidden 1\n(n=64)", size=7.7, weight="bold")
    add_text(ax, 0.480, header_y, "Hidden 2\n(n=64)", size=7.7, weight="bold")
    add_text(ax, 0.605, header_y, "Output", size=8.5, weight="bold")
    add_text(ax, 0.754, header_y, "Interpretation", size=8.5, weight="bold")

    ys = [0.760, 0.625, 0.490, 0.355, 0.205]
    add_text(ax, 0.030, 0.558, "Actor", size=8.1, weight="bold", color=COLORS["muted"], rotation=90)
    add_text(ax, 0.030, 0.205, "Critic", size=8.0, weight="bold", color=COLORS["muted"], rotation=90)
    ax.plot([0.040, 0.865], [0.288, 0.288], color=COLORS["grid"], lw=1.0, zorder=1)
    for i, (branch, y) in enumerate(zip(BRANCHES, ys)):
        color = COLORS[str(branch["color"])]
        face = "#F8FAFC" if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((0.045, y - 0.064), 0.820, 0.114, facecolor=face, edgecolor="none", zorder=0))
        add_text(ax, 0.064, y, str(branch["name"]), size=7.5, weight="bold", color=color, ha="left")
        draw_branch_network(
            ax,
            branch=branch,
            y=y,
            input_x=0.232,
            h1_x=0.370,
            h2_x=0.480,
            action_x=0.548,
            action_w=0.082,
            node_scale=0.92,
        )
        if branch["short"] == "SPR":
            rounded_box(
                ax,
                0.690,
                y - 0.034,
                0.128,
                0.068,
                "$a_{\\mathrm{SPR}}$ quantized\nnone / 2.5 / 5 / 8 / 10k",
                edge=color,
                face="white",
                size=6.2,
                lw=1.0,
            )
        else:
            rounded_box(
                ax,
                0.705,
                y - 0.034,
                0.095,
                0.068,
                str(branch["interpretation"]),
                edge=color,
                face="white",
                size=6.7,
                lw=1.0,
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS[key],
            markeredgecolor="white",
            markersize=7,
            label=label.replace("\n", " "),
        )
        for label, key in FEATURE_GROUPS
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=6,
        frameon=False,
        fontsize=7.0,
        handlelength=1.0,
        columnspacing=0.8,
    )

    save(fig, "architecture_network_nodes")


def draw_mask_matrix() -> None:
    fig, ax = setup_ax((12.6, 6.8))

    x0, y0 = 0.215, 0.22
    cell_w, cell_h = 0.095, 0.115
    branch_label_w = 0.19

    for j, (label_text, key) in enumerate(FEATURE_GROUPS):
        x = x0 + j * cell_w
        add_text(ax, x + cell_w / 2, y0 + 5.35 * cell_h, label_text, size=7.5, weight="bold", color=COLORS[key])

    for i, branch in enumerate(BRANCHES):
        y = y0 + (4 - i) * cell_h
        color = COLORS[str(branch["color"])]
        rounded_box(
            ax,
            x0 - branch_label_w - 0.015,
            y + 0.012,
            branch_label_w,
            cell_h - 0.024,
            str(branch["name"]),
            edge=color,
            face="white",
            size=7.6,
            weight="bold",
            lw=1.1,
        )
        for j, (_label_text, key) in enumerate(FEATURE_GROUPS):
            x = x0 + j * cell_w
            visible = key in set(branch["visible"])
            partial = key in set(branch.get("partial", []))
            face = COLORS[key] if (visible or partial) else "#FFFFFF"
            alpha = 0.86 if visible else (0.42 if partial else 1.0)
            edge = COLORS[key] if (visible or partial) else COLORS["grid"]
            ax.add_patch(
                Rectangle(
                    (x, y),
                    cell_w - 0.008,
                    cell_h - 0.008,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.0,
                    alpha=alpha,
                    zorder=1,
                )
            )
            if visible:
                add_text(ax, x + cell_w / 2 - 0.004, y + cell_h / 2 - 0.004, "yes", size=8.0, weight="bold", color="white")
            elif partial:
                add_text(ax, x + cell_w / 2 - 0.004, y + cell_h / 2 - 0.004, "partial", size=7.0, weight="bold", color=COLORS["text"])
            else:
                add_text(ax, x + cell_w / 2 - 0.004, y + cell_h / 2 - 0.004, "masked", size=6.5, color=COLORS["muted"])

        x_right = x0 + len(FEATURE_GROUPS) * cell_w + 0.035
        rounded_box(
            ax,
            x_right,
            y + 0.012,
            0.145,
            cell_h - 0.024,
            str(branch["action"]),
            edge=color,
            face="white",
            size=7.4,
            lw=1.1,
        )
        arrow(ax, (x0 + len(FEATURE_GROUPS) * cell_w + 0.010, y + cell_h / 2), (x_right - 0.008, y + cell_h / 2), color=color, lw=1.0)

    add_text(ax, x0 - branch_label_w - 0.015, y0 + 5.35 * cell_h, "Branch", size=8.3, weight="bold", ha="left")
    add_text(ax, x0 + len(FEATURE_GROUPS) * cell_w + 0.108, y0 + 5.35 * cell_h, "Output", size=8.3, weight="bold")

    rounded_box(
        ax,
        0.065,
        0.045,
        0.850,
        0.055,
        "Selected policy uses four actor branches plus a critic branch. ESA/baseflow, NIIP, and discretionary SJ actor branches do not see SPR-specific feature groups.\nNIIP receives general spill pressure but masks the SJ-specific spill-avoidance feature.",
        edge=COLORS["spine"],
        face="white",
        size=7.6,
        lw=1.0,
    )

    save(fig, "architecture_input_masks")


def main() -> None:
    configure_matplotlib()
    draw_technical_network()


if __name__ == "__main__":
    main()
