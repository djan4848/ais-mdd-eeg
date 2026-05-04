#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# =========================================================
# STYLE
# =========================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "figure.titlesize": 18,
})

# Colors
COL_INIT = "#d9dde1"
COL_INHIB = "#2f3e4e"

COL_FRONTAL = "#4c78a8"
COL_CACC = "#e15759"
COL_ARROW = "#2a9d8f"

COL_TEXT = "#222222"
COL_EDGE = "#444444"

# =========================================================
# HELPERS
# =========================================================
def draw_node(ax, xy, radius, facecolor, label, subtitle=None, alpha=1.0):
    circ = Circle(
        xy, radius=radius,
        facecolor=facecolor,
        edgecolor=COL_EDGE,
        linewidth=2,
        alpha=alpha
    )
    ax.add_patch(circ)
    ax.text(
        xy[0], xy[1] + 0.01,
        label,
        ha="center", va="center",
        fontsize=14,
        fontweight="bold",
        color="white" if facecolor != COL_INIT else COL_TEXT
    )
    if subtitle is not None:
        ax.text(
            xy[0], xy[1] - radius - 0.08,
            subtitle,
            ha="center", va="center",
            fontsize=10,
            color=COL_TEXT
        )

def draw_arrow(ax, start, end, text=None, color=COL_ARROW, lw=3.0, alpha=1.0, rad=0.0):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(arrow)

    if text is not None:
        xm = (start[0] + end[0]) / 2
        ym = (start[1] + end[1]) / 2 + 0.05
        ax.text(
            xm, ym, text,
            ha="center", va="center",
            fontsize=11,
            fontweight="bold",
            color=color
        )

def draw_panel(ax, title, frontal_radius, cacc_radius, arrow_strength=False):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.95, title,
        ha="center", va="top",
        fontsize=16,
        fontweight="bold",
        color=COL_TEXT
    )

    # Node positions
    frontal_xy = (0.30, 0.55)
    cacc_xy = (0.72, 0.55)

    # Nodes
    draw_node(
        ax, frontal_xy, frontal_radius,
        COL_FRONTAL,
        "Frontal",
        subtitle="Residual AIS"
    )
    draw_node(
        ax, cacc_xy, cacc_radius,
        COL_CACC,
        "cACC",
        subtitle="Residual AIS"
    )

    # Baseline directional relation
    if arrow_strength:
        draw_arrow(
            ax,
            (frontal_xy[0] + frontal_radius + 0.02, frontal_xy[1]),
            (cacc_xy[0] - cacc_radius - 0.02, cacc_xy[1]),
            text="TE ↑",
            color=COL_ARROW,
            lw=4.0,
            alpha=1.0
        )
    else:
        draw_arrow(
            ax,
            (frontal_xy[0] + frontal_radius + 0.02, frontal_xy[1]),
            (cacc_xy[0] - cacc_radius - 0.02, cacc_xy[1]),
            text=None,
            color="#9fb3c8",
            lw=2.0,
            alpha=0.5
        )

# =========================================================
# FIGURE
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
fig.suptitle(
    "Figure 4. Conceptual Summary of Fronto-Cingulate Reorganization During Semantic Inhibition",
    fontweight="bold",
    y=0.98
)

# Panel A: Initiation
draw_panel(
    axes[0],
    title="A. Initiation",
    frontal_radius=0.12,   # higher AIS
    cacc_radius=0.08,      # lower AIS
    arrow_strength=False
)

axes[0].text(
    0.5, 0.12,
    "Residual dynamics are dominated by higher local predictability in frontal regions.",
    ha="center", va="center",
    fontsize=11,
    color=COL_TEXT
)

# Panel B: Inhibition
draw_panel(
    axes[1],
    title="B. Inhibition",
    frontal_radius=0.08,   # lower AIS
    cacc_radius=0.12,      # higher AIS
    arrow_strength=True
)

axes[1].text(
    0.5, 0.12,
    "Semantic inhibition is associated with reduced frontal AIS,\n"
    "increased cACC AIS, and stronger residual transfer from frontal to cACC.",
    ha="center", va="center",
    fontsize=11,
    color=COL_TEXT
)

# Global note
fig.text(
    0.5, 0.03,
    "Node size represents the relative magnitude of residual Active Information Storage (AIS).\n"
    "The green arrow denotes the secondary increase in residual Transfer Entropy (TE) from frontal to cACC during inhibition.",
    ha="center",
    fontsize=10.5,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75")
)

plt.tight_layout(rect=[0, 0.08, 1, 0.93])

# Save
out_path = "Figure_4_Conceptual_FrontoCingulate_Reorganization.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"[ok] Saved conceptual figure to: {out_path}")
