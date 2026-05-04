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
PID_PATH = DERIV_ROOT / "te_n450_residual_r2pos" / "pid_lh_rh_frontal_residual_r2pos.csv"

OUTDIR = DERIV_ROOT / "figures_paper"
OUTDIR.mkdir(exist_ok=True, parents=True)

# =========================================================
# LOAD
# =========================================================
pid = pd.read_csv(PID_PATH)

cond_map = {"INIT": "Initiation", "INHIB": "Inhibition"}
pid["cond"] = pid["cond"].replace(cond_map)

# =========================================================
# STYLE
# =========================================================
sns.set_context("paper", font_scale=1.12)
sns.set_style("ticks")

COL_INIT = "#cfd4d9"
COL_INHIB = "#2f3e4e"

METRICS = [
    ("redundancy", "Redundancy"),
    ("unique_s1", "Unique LH"),
    ("unique_s2", "Unique RH"),
    ("synergy", "Synergy"),
]

# =========================================================
# SUBJECT-LEVEL SUMMARIES
# =========================================================
pid_subj = (
    pid.groupby(["subject", "cond"], as_index=False)
       .agg(
           redundancy=("redundancy", "mean"),
           unique_s1=("unique_s1", "mean"),
           unique_s2=("unique_s2", "mean"),
           synergy=("synergy", "mean"),
       )
)

# =========================================================
# STATS
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
        "n": int(len(piv))
    }

# =========================================================
# FIGURE
# =========================================================
fig, axes = plt.subplots(1, 4, figsize=(19, 5.8))
fig.suptitle(
    "Supplementary Figure S3. Residual PID Results (Exploratory Analysis)",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

for ax, (metric, title) in zip(axes, METRICS):
    df_plot = pid_subj[["subject", "cond", metric]].copy()
    stats = paired_summary(df_plot, metric)

    sns.barplot(
        data=df_plot,
        x="cond",
        y=metric,
        order=["Initiation", "Inhibition"],
        palette=[COL_INIT, COL_INHIB],
        errorbar="sd",
        capsize=0.07,
        ax=ax
    )

    sns.stripplot(
        data=df_plot,
        x="cond",
        y=metric,
        order=["Initiation", "Inhibition"],
        color="black",
        alpha=0.30,
        size=3,
        ax=ax
    )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("PID (bits)")

    ax.text(
        0.5, 0.95,
        (
            f"Init = {stats['init_mean']:.3f}\n"
            f"Inhib = {stats['inhib_mean']:.3f}\n"
            f"t({stats['n']-1}) = {stats['t']:.2f}\n"
            f"p = {stats['p']:.3f}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="0.82", pad=2)
    )

    sns.despine(ax=ax)

# Footnote
n_subjects = pid["subject"].nunique()
fig.text(
    0.5, 0.02,
    (
        f"Residual PID computed for the LH + RH → Frontal configuration using only trials with meaningful DDS fits (R² > 0). "
        f"Sample: N = {n_subjects} subjects. No robust condition effects were observed."
    ),
    ha="center",
    fontsize=10.5,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75")
)

plt.tight_layout(rect=[0, 0.07, 1, 0.93])
outfile = OUTDIR / "Supplementary_Figure_S3_PID_results.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()

print(f"[ok] Saved: {outfile}")
