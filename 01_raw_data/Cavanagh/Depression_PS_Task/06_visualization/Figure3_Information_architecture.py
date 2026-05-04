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

# =========================================================
# BASIC FORMATTING
# =========================================================
cond_map = {"INIT": "Initiation", "INHIB": "Inhibition"}
ais["cond"] = ais["cond"].replace(cond_map)
te["cond"] = te["cond"].replace(cond_map)

# =========================================================
# COLORS
# =========================================================
COL_INIT = "#bdc3c7"
COL_INHIB = "#2c3e50"

COL_FRONTAL = "#4c78a8"
COL_CACC = "#e15759"
COL_TE = "#2a9d8f"

sns.set_context("paper", font_scale=1.35)
sns.set_style("ticks")

# =========================================================
# SUBJECT-LEVEL SUMMARIES
# =========================================================
# AIS
ais_subj = (
    ais.groupby(["subject", "roi", "cond"], as_index=False)
       .agg(ais_bits=("ais_bits", "mean"))
)

# TE frontal -> cacc only
te_fc = te[te["direction"] == "frontal->cacc"].copy()
te_fc_subj = (
    te_fc.groupby(["subject", "cond"], as_index=False)
         .agg(te_bits=("te_bits", "mean"))
)

# =========================================================
# STATS
# =========================================================
def paired_stats(df, value_col):
    piv = df.pivot(index="subject", columns="cond", values=value_col).dropna()
    init = piv["Initiation"].values
    inhib = piv["Inhibition"].values
    t, p = ttest_rel(inhib, init)
    return {
        "init_mean": float(np.mean(init)),
        "inhib_mean": float(np.mean(inhib)),
        "t": float(t),
        "p": float(p),
        "n": int(len(piv))
    }

stats_frontal = paired_stats(ais_subj[ais_subj["roi"] == "frontal"], "ais_bits")
stats_cacc = paired_stats(ais_subj[ais_subj["roi"] == "cacc"], "ais_bits")
stats_te = paired_stats(te_fc_subj, "te_bits")

# =========================================================
# FIGURE LAYOUT
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    "Residual Information Dynamics Reveal Fronto-Cingulate Reorganization During Semantic Inhibition",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

# =========================================================
# PANEL A — AIS frontal
# =========================================================
ax = axes[0]
df_plot = ais_subj[ais_subj["roi"] == "frontal"].copy()

sns.barplot(
    data=df_plot,
    x="cond",
    y="ais_bits",
    order=["Initiation", "Inhibition"],
    palette=[COL_INIT, COL_INHIB],
    ax=ax,
    errorbar="sd",
    capsize=0.08
)

sns.stripplot(
    data=df_plot,
    x="cond",
    y="ais_bits",
    order=["Initiation", "Inhibition"],
    color="black",
    alpha=0.35,
    size=3.5,
    ax=ax
)

ax.set_title("A. Residual AIS in Frontal ROI", fontweight="bold", pad=16)
ax.set_xlabel("")
ax.set_ylabel("AIS (bits)")
ax.text(
    0.5, 0.94,
    (
        f"Initiation: {stats_frontal['init_mean']:.3f}\n"
        f"Inhibition: {stats_frontal['inhib_mean']:.3f}\n"
        f"t({stats_frontal['n']-1}) = {stats_frontal['t']:.2f}, p = {stats_frontal['p']:.3f}"
    ),
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8", pad=3)
)
ax.text(
    0.5, 0.06,
    "Residual frontal predictability decreases during inhibition",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color=COL_FRONTAL
)
sns.despine(ax=ax)

# =========================================================
# PANEL B — AIS cACC
# =========================================================
ax = axes[1]
df_plot = ais_subj[ais_subj["roi"] == "cacc"].copy()

sns.barplot(
    data=df_plot,
    x="cond",
    y="ais_bits",
    order=["Initiation", "Inhibition"],
    palette=[COL_INIT, COL_INHIB],
    ax=ax,
    errorbar="sd",
    capsize=0.08
)

sns.stripplot(
    data=df_plot,
    x="cond",
    y="ais_bits",
    order=["Initiation", "Inhibition"],
    color="black",
    alpha=0.35,
    size=3.5,
    ax=ax
)

ax.set_title("B. Residual AIS in cACC ROI", fontweight="bold", pad=16)
ax.set_xlabel("")
ax.set_ylabel("AIS (bits)")
ax.text(
    0.5, 0.94,
    (
        f"Initiation: {stats_cacc['init_mean']:.3f}\n"
        f"Inhibition: {stats_cacc['inhib_mean']:.3f}\n"
        f"t({stats_cacc['n']-1}) = {stats_cacc['t']:.2f}, p = {stats_cacc['p']:.3f}"
    ),
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8", pad=3)
)
ax.text(
    0.5, 0.06,
    "Residual cACC predictability increases during inhibition",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color=COL_CACC
)
sns.despine(ax=ax)

# =========================================================
# PANEL C — TE frontal -> cacc
# =========================================================
ax = axes[2]
df_plot = te_fc_subj.copy()

sns.barplot(
    data=df_plot,
    x="cond",
    y="te_bits",
    order=["Initiation", "Inhibition"],
    palette=[COL_INIT, COL_INHIB],
    ax=ax,
    errorbar="sd",
    capsize=0.08
)

sns.stripplot(
    data=df_plot,
    x="cond",
    y="te_bits",
    order=["Initiation", "Inhibition"],
    color="black",
    alpha=0.35,
    size=3.5,
    ax=ax
)

ax.set_title("C. Residual TE: Frontal → cACC", fontweight="bold", pad=16)
ax.set_xlabel("")
ax.set_ylabel("TE (bits)")
ax.text(
    0.5, 0.94,
    (
        f"Initiation: {stats_te['init_mean']:.3f}\n"
        f"Inhibition: {stats_te['inhib_mean']:.3f}\n"
        f"t({stats_te['n']-1}) = {stats_te['t']:.2f}, p = {stats_te['p']:.3f}"
    ),
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8", pad=3)
)
ax.text(
    0.5, 0.06,
    "Directed residual transfer increases from frontal to cACC",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    color=COL_TE
)
sns.despine(ax=ax)

# =========================================================
# FOOTNOTE
# =========================================================
n_subjects = ais["subject"].nunique()
fig.text(
    0.5, 0.02,
    (
        f"Residual analyses were restricted to trials with meaningful DDS fits (R² > 0). "
        f"Sample: N = {n_subjects} subjects."
    ),
    ha="center",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7")
)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
outfile = OUTDIR / "Figure_3_Residual_Information_Dynamics.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()

print(f"[ok] Figure 3 saved: {outfile}")
