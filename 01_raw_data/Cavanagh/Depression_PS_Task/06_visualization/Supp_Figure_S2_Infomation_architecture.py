#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

from dds_base.io.paths import DERIV_ROOT

# =========================================================
# PATHS
# =========================================================
AIS_PATH = DERIV_ROOT / "ais_n450_residual_r2pos" / "ais_n450_residual_r2pos_results.csv"
TE_PATH = DERIV_ROOT / "te_n450_residual_r2pos" / "te_n450_residual_r2pos_results.csv"

OUTDIR = DERIV_ROOT / "figures_paper"
OUTDIR.mkdir(exist_ok=True, parents=True)

# =========================================================
# LOAD
# =========================================================
ais = pd.read_csv(AIS_PATH)
te = pd.read_csv(TE_PATH)

cond_map = {"INIT": "Initiation", "INHIB": "Inhibition"}
ais["cond"] = ais["cond"].replace(cond_map)
te["cond"] = te["cond"].replace(cond_map)

# =========================================================
# STYLE
# =========================================================
sns.set_context("paper", font_scale=1.08)
sns.set_style("ticks")

COL_INIT = "#cfd4d9"
COL_INHIB = "#2f3e4e"

AIS_ROIS = ["frontal", "cacc", "lh", "rh"]
TE_DIRS = ["frontal->cacc", "cacc->frontal", "lh->frontal", "rh->frontal", "lh->cacc", "rh->cacc"]

AIS_TITLES = {
    "frontal": "AIS residual — Frontal",
    "cacc": "AIS residual — cACC",
    "lh": "AIS residual — LH",
    "rh": "AIS residual — RH",
}

TE_TITLES = {
    "frontal->cacc": "TE residual — Frontal → cACC",
    "cacc->frontal": "TE residual — cACC → Frontal",
    "lh->frontal": "TE residual — LH → Frontal",
    "rh->frontal": "TE residual — RH → Frontal",
    "lh->cacc": "TE residual — LH → cACC",
    "rh->cacc": "TE residual — RH → cACC",
}

# =========================================================
# HELPERS
# =========================================================
def paired_summary(df, value_col):
    piv = df.pivot(index="subject", columns="cond", values=value_col).dropna()
    init = piv["Initiation"].values
    inhib = piv["Inhibition"].values
    t, p = ttest_rel(inhib, init)
    return {
        "init_mean": float(np.mean(init)),
        "inhib_mean": float(np.mean(inhib)),
        "t": float(t),
        "p": float(p),
        "n": int(len(piv)),
    }

def add_panel(ax, df_plot, x_col, y_col, title, stat_text):
    sns.barplot(
        data=df_plot,
        x=x_col,
        y=y_col,
        order=["Initiation", "Inhibition"],
        palette=[COL_INIT, COL_INHIB],
        errorbar="sd",
        capsize=0.07,
        ax=ax
    )

    sns.stripplot(
        data=df_plot,
        x=x_col,
        y=y_col,
        order=["Initiation", "Inhibition"],
        color="black",
        alpha=0.30,
        size=2.8,
        ax=ax
    )

    ax.set_title(title, fontweight="bold", fontsize=11, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(
        0.5, 0.95,
        stat_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.8,
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="0.82", pad=2)
    )
    sns.despine(ax=ax)

# =========================================================
# SUBJECT-LEVEL SUMMARIES
# =========================================================
ais_subj = (
    ais.groupby(["subject", "roi", "cond"], as_index=False)
       .agg(ais_bits=("ais_bits", "mean"))
)

te_subj = (
    te.groupby(["subject", "direction", "cond"], as_index=False)
      .agg(te_bits=("te_bits", "mean"))
)

# =========================================================
# FIGURE
# =========================================================
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle(
    "Supplementary Figure S2. Full Residual Information Architecture Across ROIs and Directions",
    fontsize=17,
    fontweight="bold",
    y=0.985
)

# -----------------------------
# Top row: AIS (4 used, 5th blank)
# -----------------------------
for i, roi in enumerate(AIS_ROIS):
    ax = axes[0, i]
    df_plot = ais_subj[ais_subj["roi"] == roi].copy()
    stats = paired_summary(df_plot, "ais_bits")

    txt = (
        f"Init = {stats['init_mean']:.3f}\n"
        f"Inhib = {stats['inhib_mean']:.3f}\n"
        f"t({stats['n']-1}) = {stats['t']:.2f}\n"
        f"p = {stats['p']:.3f}"
    )

    add_panel(
        ax,
        df_plot,
        x_col="cond",
        y_col="ais_bits",
        title=AIS_TITLES[roi],
        stat_text=txt
    )

    if i == 0:
        ax.set_ylabel("AIS (bits)")

# blank last top panel
axes[0, 4].axis("off")
axes[0, 4].text(
    0.5, 0.5,
    "Top row:\nResidual AIS\nacross all ROIs",
    ha="center", va="center",
    fontsize=13, fontweight="bold"
)

# -----------------------------
# Bottom row: TE (5 first directions)
# -----------------------------
for i, direction in enumerate(TE_DIRS[:5]):
    ax = axes[1, i]
    df_plot = te_subj[te_subj["direction"] == direction].copy()
    stats = paired_summary(df_plot, "te_bits")

    txt = (
        f"Init = {stats['init_mean']:.3f}\n"
        f"Inhib = {stats['inhib_mean']:.3f}\n"
        f"t({stats['n']-1}) = {stats['t']:.2f}\n"
        f"p = {stats['p']:.3f}"
    )

    add_panel(
        ax,
        df_plot,
        x_col="cond",
        y_col="te_bits",
        title=TE_TITLES[direction],
        stat_text=txt
    )

    if i == 0:
        ax.set_ylabel("TE (bits)")

# Add the 6th TE direction as inset-like mini text in last panel or replace one panel strategy
# Here we repurpose the last bottom panel if needed; since grid is 2x5, we show the 6th in top-right note text.
# Better: overwrite bottom-right with the 6th if you prefer complete symmetry.
# We'll use bottom-right for 6th and shift the 5th into the previous slot already done.
axes[1, 4].cla()
direction = TE_DIRS[5]
df_plot = te_subj[te_subj["direction"] == direction].copy()
stats = paired_summary(df_plot, "te_bits")

txt = (
    f"Init = {stats['init_mean']:.3f}\n"
    f"Inhib = {stats['inhib_mean']:.3f}\n"
    f"t({stats['n']-1}) = {stats['t']:.2f}\n"
    f"p = {stats['p']:.3f}"
)

add_panel(
    axes[1, 4],
    df_plot,
    x_col="cond",
    y_col="te_bits",
    title=TE_TITLES[direction],
    stat_text=txt
)

# Need 5th TE direction as well: replace previous loop to first 4 and keep 5th+6th in last two
# So redraw properly:
for j in range(5):
    axes[1, j].cla()

for i, direction in enumerate(TE_DIRS):
    if i >= 5:
        break
    ax = axes[1, i]
    df_plot = te_subj[te_subj["direction"] == direction].copy()
    stats = paired_summary(df_plot, "te_bits")

    txt = (
        f"Init = {stats['init_mean']:.3f}\n"
        f"Inhib = {stats['inhib_mean']:.3f}\n"
        f"t({stats['n']-1}) = {stats['t']:.2f}\n"
        f"p = {stats['p']:.3f}"
    )

    add_panel(
        ax,
        df_plot,
        x_col="cond",
        y_col="te_bits",
        title=TE_TITLES[direction],
        stat_text=txt
    )

    if i == 0:
        ax.set_ylabel("TE (bits)")

# Add 6th direction as small inset note in the figure footer
direction6 = TE_DIRS[5]
df_plot6 = te_subj[te_subj["direction"] == direction6].copy()
stats6 = paired_summary(df_plot6, "te_bits")

footer_text = (
    f"Additional TE direction not shown as full panel: {TE_TITLES[direction6]} | "
    f"Init = {stats6['init_mean']:.3f}, Inhib = {stats6['inhib_mean']:.3f}, "
    f"t({stats6['n']-1}) = {stats6['t']:.2f}, p = {stats6['p']:.3f}"
)

# Footnote
n_subjects = ais["subject"].nunique()
fig.text(
    0.5, 0.02,
    (
        f"Residual analyses restricted to trials with meaningful DDS fits (R² > 0). "
        f"Sample: N = {n_subjects} subjects. "
        f"{footer_text}"
    ),
    ha="center",
    fontsize=10.2,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75")
)

plt.tight_layout(rect=[0, 0.07, 1, 0.95])
outfile = OUTDIR / "Supplementary_Figure_S2_Full_information_architecture.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()

print(f"[ok] Saved: {outfile}")
