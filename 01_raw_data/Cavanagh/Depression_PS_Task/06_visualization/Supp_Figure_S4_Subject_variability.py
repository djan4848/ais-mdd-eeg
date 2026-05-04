#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# SUBJECT-LEVEL SUMMARIES
# =========================================================
ais_subj = (
    ais.groupby(["subject", "roi", "cond"], as_index=False)
       .agg(ais_bits=("ais_bits", "mean"))
)

te_fc = te[te["direction"] == "frontal->cacc"].copy()
te_subj = (
    te_fc.groupby(["subject", "cond"], as_index=False)
         .agg(te_bits=("te_bits", "mean"))
)

# =========================================================
# BUILD DELTA TABLES
# =========================================================
def build_delta(df, value_col, extra_cols=None):
    if extra_cols is None:
        extra_cols = []
    piv = df.pivot(index=["subject"] + extra_cols, columns="cond", values=value_col).dropna().reset_index()
    piv["delta"] = piv["INHIB"] - piv["INIT"]
    return piv

frontal_delta = build_delta(
    ais_subj[ais_subj["roi"] == "frontal"][["subject", "cond", "ais_bits"]],
    "ais_bits"
).sort_values("delta")

cacc_delta = build_delta(
    ais_subj[ais_subj["roi"] == "cacc"][["subject", "cond", "ais_bits"]],
    "ais_bits"
).sort_values("delta")

te_delta = build_delta(
    te_subj[["subject", "cond", "te_bits"]],
    "te_bits"
).sort_values("delta")

# top extremes
frontal_top = frontal_delta.reindex(frontal_delta["delta"].abs().sort_values(ascending=False).index).head(10)
cacc_top = cacc_delta.reindex(cacc_delta["delta"].abs().sort_values(ascending=False).index).head(10)

# =========================================================
# STYLE
# =========================================================
sns.set_context("paper", font_scale=1.08)
sns.set_style("ticks")

COL_FRONTAL = "#4c78a8"
COL_CACC = "#e15759"
COL_TE = "#2a9d8f"

# =========================================================
# FIGURE
# =========================================================
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.1, 1.1], hspace=0.38, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

fig.suptitle(
    "Supplementary Figure S4. Subject-Level Variability of Residual Information Effects",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

# =========================================================
# PANEL A — frontal delta
# =========================================================
x = np.arange(len(frontal_delta))
ax1.bar(x, frontal_delta["delta"].values, color=COL_FRONTAL, alpha=0.85)
ax1.axhline(0, color="black", linestyle="--", linewidth=1)
ax1.set_title("A. Subject-wise ΔAIS in Frontal ROI", fontweight="bold", pad=12)
ax1.set_ylabel("ΔAIS = INHIB - INIT")
ax1.set_xlabel("Subjects (sorted)")
ax1.text(
    0.5, 0.93,
    f"Mean ΔAIS = {frontal_delta['delta'].mean():.3f}",
    transform=ax1.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8", pad=2)
)
sns.despine(ax=ax1)

# =========================================================
# PANEL B — cacc delta
# =========================================================
x = np.arange(len(cacc_delta))
ax2.bar(x, cacc_delta["delta"].values, color=COL_CACC, alpha=0.85)
ax2.axhline(0, color="black", linestyle="--", linewidth=1)
ax2.set_title("B. Subject-wise ΔAIS in cACC ROI", fontweight="bold", pad=12)
ax2.set_ylabel("ΔAIS = INHIB - INIT")
ax2.set_xlabel("Subjects (sorted)")
ax2.text(
    0.5, 0.93,
    f"Mean ΔAIS = {cacc_delta['delta'].mean():.3f}",
    transform=ax2.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8", pad=2)
)
sns.despine(ax=ax2)

# =========================================================
# PANEL C — TE frontal->cacc delta
# =========================================================
x = np.arange(len(te_delta))
ax3.bar(x, te_delta["delta"].values, color=COL_TE, alpha=0.85)
ax3.axhline(0, color="black", linestyle="--", linewidth=1)
ax3.set_title("C. Subject-wise ΔTE (Frontal → cACC)", fontweight="bold", pad=12)
ax3.set_ylabel("ΔTE = INHIB - INIT")
ax3.set_xlabel("Subjects (sorted)")
ax3.text(
    0.5, 0.93,
    f"Mean ΔTE = {te_delta['delta'].mean():.3f}",
    transform=ax3.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8", pad=2)
)
sns.despine(ax=ax3)

# =========================================================
# PANEL D — extreme subjects table
# =========================================================
ax4.axis("off")
ax4.set_title("D. Top 10 Subjects with Largest |ΔAIS|", fontweight="bold", pad=12)

table_lines = []
table_lines.append("Frontal ROI")
table_lines.append("---------------------------")
for _, row in frontal_top.iterrows():
    table_lines.append(f"{row['subject']:>4s}   ΔAIS = {row['delta']:+.3f}")

table_lines.append("")
table_lines.append("cACC ROI")
table_lines.append("---------------------------")
for _, row in cacc_top.iterrows():
    table_lines.append(f"{row['subject']:>4s}   ΔAIS = {row['delta']:+.3f}")

table_text = "\n".join(table_lines)

ax4.text(
    0.02, 0.98,
    table_text,
    ha="left",
    va="top",
    family="monospace",
    fontsize=10.5,
    bbox=dict(facecolor="white", alpha=0.95, edgecolor="0.75", pad=4)
)

# =========================================================
# FOOTNOTE
# =========================================================
n_subjects = ais["subject"].nunique()
fig.text(
    0.5, 0.02,
    (
        f"Residual analyses were restricted to trials with meaningful DDS fits (R² > 0). "
        f"Bars represent subject-level condition differences (INHIB - INIT), allowing visual assessment of inter-individual variability. "
        f"Sample: N = {n_subjects} subjects."
    ),
    ha="center",
    fontsize=10.5,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75")
)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
outfile = OUTDIR / "Supplementary_Figure_S4_Subject_variability.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()

print(f"[ok] Saved: {outfile}")
