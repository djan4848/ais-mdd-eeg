#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dds_base.io.paths import DERIV_ROOT

# =========================================================
# PATHS
# =========================================================
DDS_PATH = DERIV_ROOT / "dds_peak_aligned_n450" / "dds_n450_results.csv"
OUTDIR = DERIV_ROOT / "figures_paper"
OUTDIR.mkdir(exist_ok=True, parents=True)

# =========================================================
# LOAD
# =========================================================
df = pd.read_csv(DDS_PATH)
df = df[df["r2"] > 0].copy()

PARAMS = ["A1", "gamma1", "f1", "phi1", "A2", "gamma2", "f2", "phi2"]
ROIS = ["frontal", "cacc", "lh", "rh"]

# keep only rows with all params
df = df.dropna(subset=PARAMS + ["roi"])

# =========================================================
# STYLE
# =========================================================
sns.set_context("paper", font_scale=1.2)
sns.set_style("ticks")

palette = {
    "frontal": "#4c78a8",
    "cacc": "#e15759",
    "lh": "#72b7b2",
    "rh": "#f28e2b",
}

param_titles = {
    "A1": "A1 (Mode 1 amplitude)",
    "gamma1": "γ1 (Mode 1 damping)",
    "f1": "f1 (Mode 1 frequency)",
    "phi1": "φ1 (Mode 1 phase)",
    "A2": "A2 (Mode 2 amplitude)",
    "gamma2": "γ2 (Mode 2 damping)",
    "f2": "f2 (Mode 2 frequency)",
    "phi2": "φ2 (Mode 2 phase)",
}

# =========================================================
# FIGURE
# =========================================================
fig, axes = plt.subplots(2, 4, figsize=(19, 9))
axes = axes.flatten()

for ax, param in zip(axes, PARAMS):
    sns.violinplot(
        data=df,
        x="roi",
        y=param,
        order=ROIS,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.0,
        ax=ax
    )

    sns.boxplot(
        data=df,
        x="roi",
        y=param,
        order=ROIS,
        width=0.18,
        showfliers=False,
        boxprops=dict(facecolor="white", alpha=0.85),
        whiskerprops=dict(color="0.35"),
        capprops=dict(color="0.35"),
        medianprops=dict(color="black", linewidth=1.6),
        ax=ax
    )

    ax.set_title(param_titles[param], fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

    sns.despine(ax=ax)

# Global title
fig.suptitle(
    "Supplementary Figure S1. Distribution of DDS Parameters Across ROIs (Valid Fits Only)",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

# Footnote
n_subjects = df["subject"].nunique()
n_rows = len(df)

fig.text(
    0.5, 0.02,
    (
        f"DDS parameter distributions computed from valid fits only (R² > 0). "
        f"Sample: N = {n_subjects} subjects, n = {n_rows} trial×ROI fits."
    ),
    ha="center",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75")
)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
outfile = OUTDIR / "Supplementary_Figure_S1_DDS_parameter_distributions.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()

print(f"[ok] Saved: {outfile}")
