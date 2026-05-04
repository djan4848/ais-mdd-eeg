#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dds_base.io.paths import DERIV_ROOT, EXCLUDE_SUBJECTS, ROOT

# =========================================================
# PATHS
# =========================================================
AIS_PATH = DERIV_ROOT / "ais_n450" / "ais_n450_results_with_group.csv"
TE_PATH = DERIV_ROOT / "te_n450" / "te_n450_results_with_group.csv"
PID_PATH = DERIV_ROOT / "te_n450" / "pid_lh_rh_frontal_with_group.csv"

CLINICAL_PATH = ROOT / "../PARTICIPANTES_TODO_HAYLING.csv"

OUTDIR = DERIV_ROOT / "figures_paper"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUTFIG = OUTDIR / "Figure_5_Extreme_Subjects_Information_Architecture_v2.png"

# =========================================================
# CONFIG
# =========================================================
TE_LAG = 4
AIS_ROI = "frontal"

CLINICAL_SPECS = [
    ("BECK", "BDI"),
    ("RRS", "RRS"),
    ("SNAITH", "SHAPS"),
]

COND_LABELS = {"INIT": "Initiation", "INHIB": "Inhibition"}

PID_COMPONENTS = [
    ("unique_rh", "Unique RH"),
    ("unique_lh", "Unique LH"),
    ("redundancy", "Redundancy"),
    ("synergy", "Synergy"),
]

# =========================================================
# COLOR PALETTE (aligned with Figures 3 and 4)
# =========================================================
COL_INIT = "#bdc3c7"      # light grey
COL_INHIB = "#2c3e50"     # dark blue-grey

COL_UNIQUE_RH = "#9b59b6" # muted purple
COL_UNIQUE_LH = "#52b788" # muted green
COL_REDUND = "#5b8db8"    # muted blue
COL_SYNERGY = "#d36b6b"   # muted red

PID_COLORS = [COL_UNIQUE_RH, COL_UNIQUE_LH, COL_REDUND, COL_SYNERGY]

# =========================================================
# LOAD
# =========================================================
ais = pd.read_csv(AIS_PATH)
te = pd.read_csv(TE_PATH)
pid = pd.read_csv(PID_PATH)
clinical_raw = pd.read_csv(CLINICAL_PATH)

# =========================================================
# PREP CLINICAL
# =========================================================
clinical_raw["N. PARTICIPANTE"] = pd.to_numeric(clinical_raw["N. PARTICIPANTE"], errors="coerce")
clinical_raw = clinical_raw.dropna(subset=["N. PARTICIPANTE"]).copy()
clinical_raw["N. PARTICIPANTE"] = clinical_raw["N. PARTICIPANTE"].astype(int)
clinical_raw["subject"] = clinical_raw["N. PARTICIPANTE"].apply(lambda x: f"P{x}")

for col, _ in CLINICAL_SPECS:
    clinical_raw[col] = pd.to_numeric(clinical_raw[col], errors="coerce")

clinical = clinical_raw[["subject"] + [c for c, _ in CLINICAL_SPECS]].copy()
clinical = clinical.dropna(subset=[c for c, _ in CLINICAL_SPECS])

clinical = clinical[~clinical["subject"].isin(EXCLUDE_SUBJECTS)].copy()

subjects_available = set(ais["subject"].unique()) & set(te["subject"].unique()) & set(pid["subject"].unique())
clinical = clinical[clinical["subject"].isin(subjects_available)].copy()

# =========================================================
# EXTREME SUBJECTS
# =========================================================
extreme_rows = []
for raw_col, nice_label in CLINICAL_SPECS:
    tmp = clinical.dropna(subset=[raw_col]).copy()
    low_row = tmp.loc[tmp[raw_col].idxmin()]
    high_row = tmp.loc[tmp[raw_col].idxmax()]

    extreme_rows.append({
        "metric_raw": raw_col,
        "metric_label": nice_label,
        "subject_low": low_row["subject"],
        "value_low": low_row[raw_col],
        "subject_high": high_row["subject"],
        "value_high": high_row[raw_col],
    })

extremes = pd.DataFrame(extreme_rows)

print("\n=== Extreme subjects ===")
print(extremes.to_string(index=False))

# =========================================================
# PREP AIS / TE / PID summaries by subject
# =========================================================
ais_f = ais[ais["roi"] == AIS_ROI].copy()
ais_subj = (
    ais_f.groupby(["subject", "cond"], as_index=False)
    .agg(ais_bits=("ais_bits", "mean"))
)

te_f = te[(te["target_roi"] == "frontal") & (te["lag_samples"] == TE_LAG)].copy()
te_subj = (
    te_f.groupby(["subject", "cond"], as_index=False)
    .agg(te_bits=("te_bits", "mean"))
)

pid_subj = (
    pid.groupby(["subject", "cond"], as_index=False)
    .agg(
        unique_rh=("unique_rh", "mean"),
        unique_lh=("unique_lh", "mean"),
        redundancy=("redundancy", "mean"),
        synergy=("synergy", "mean"),
    )
)

# =========================================================
# FIGURE
# =========================================================
sns.set_context("paper", font_scale=1.12)
sns.set_style("ticks")

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle(
    "Extreme-Subject Visualization of Clinical Variation in N450 Information Architecture",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

for row_idx, row in extremes.iterrows():
    metric_label = row["metric_label"]
    subj_low = row["subject_low"]
    subj_high = row["subject_high"]
    val_low = row["value_low"]
    val_high = row["value_high"]

    subj_label_map = {
        subj_low: f"Low {metric_label}\n({subj_low}, {val_low:.0f})",
        subj_high: f"High {metric_label}\n({subj_high}, {val_high:.0f})",
    }

    # =====================================================
    # Panel 1: AIS
    # =====================================================
    ax = axes[row_idx, 0]

    ais_plot = ais_subj[ais_subj["subject"].isin([subj_low, subj_high])].copy()
    ais_plot["cond"] = ais_plot["cond"].replace(COND_LABELS)
    ais_plot["subject_label"] = ais_plot["subject"].map(subj_label_map)

    sns.barplot(
        data=ais_plot,
        x="subject_label",
        y="ais_bits",
        hue="cond",
        order=[subj_label_map[subj_low], subj_label_map[subj_high]],
        hue_order=["Initiation", "Inhibition"],
        palette=[COL_INIT, COL_INHIB],
        ax=ax,
        errorbar=None
    )

    ax.set_title(f"{chr(65 + row_idx*3)}. AIS Frontal", fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("AIS (bits)")
    if row_idx == 0:
        ax.legend(title="", frameon=False, loc="upper right")
    else:
        ax.legend_.remove()
    sns.despine(ax=ax)

    # =====================================================
    # Panel 2: TE to frontal
    # =====================================================
    ax = axes[row_idx, 1]

    te_plot = te_subj[te_subj["subject"].isin([subj_low, subj_high])].copy()
    te_plot["cond"] = te_plot["cond"].replace(COND_LABELS)
    te_plot["subject_label"] = te_plot["subject"].map(subj_label_map)

    sns.barplot(
        data=te_plot,
        x="subject_label",
        y="te_bits",
        hue="cond",
        order=[subj_label_map[subj_low], subj_label_map[subj_high]],
        hue_order=["Initiation", "Inhibition"],
        palette=[COL_INIT, COL_INHIB],
        ax=ax,
        errorbar=None
    )

    ax.set_title(f"{chr(66 + row_idx*3)}. TE to Frontal", fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("TE (bits)")
    if row_idx == 0:
        ax.legend(title="", frameon=False, loc="upper right")
    else:
        ax.legend_.remove()
    sns.despine(ax=ax)

    # =====================================================
    # Panel 3: PID in INHIB
    # =====================================================
    ax = axes[row_idx, 2]

    pid_plot = pid_subj[
        (pid_subj["subject"].isin([subj_low, subj_high])) &
        (pid_subj["cond"] == "INHIB")
    ].copy()

    pid_plot["subject_label"] = pid_plot["subject"].map(subj_label_map)
    pid_plot = pid_plot.set_index("subject_label").loc[
        [subj_label_map[subj_low], subj_label_map[subj_high]]
    ].reset_index()

    y_pos = np.arange(len(pid_plot))
    left = np.zeros(len(pid_plot))

    for (comp, comp_label), color in zip(PID_COMPONENTS, PID_COLORS):
        vals = pid_plot[comp].values
        ax.barh(y_pos, vals, left=left, color=color, label=comp_label, height=0.55)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pid_plot["subject_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("PID components (bits)")
    ax.set_title(f"{chr(67 + row_idx*3)}. PID in Inhibition", fontweight="bold", pad=10)

    if row_idx == 0:
        ax.legend(
            frameon=False,
            fontsize=8.8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.00),
            borderaxespad=0
        )

    sns.despine(ax=ax, left=False, bottom=False)

    # =====================================================
    # Row label
    # =====================================================
    axes[row_idx, 0].text(
        -0.33, 1.15, metric_label,
        transform=axes[row_idx, 0].transAxes,
        fontsize=14.5,
        fontweight="bold",
        va="top"
    )

# =========================================================
# FOOTNOTE
# =========================================================
fig.text(
    0.5, 0.015,
    "Descriptive extreme-subject visualization. Low/high cases selected independently for BDI, RRS, and SHAPS. "
    "N450 window: ±200 ms around frontal peak | TE lag = 16 ms | PID bins = 4",
    ha="center",
    fontsize=10.2,
    fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
)

plt.tight_layout(rect=[0.03, 0.04, 1, 0.95])
plt.savefig(OUTFIG, dpi=300, bbox_inches="tight")
plt.show()
