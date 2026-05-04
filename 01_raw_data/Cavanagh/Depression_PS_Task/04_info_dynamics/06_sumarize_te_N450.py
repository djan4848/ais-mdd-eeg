#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dds_base.io.paths import (
    DERIV_ROOT,
    hayling_epo_files,
    ROIS,
    EXCLUDE_SUBJECTS,
)
# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = DERIV_ROOT / "te_n450" / "te_n450_results.csv"

OUT_SUBJECT_SUMMARY = DERIV_ROOT / "te_n450" / "te_subject_summary.csv"
OUT_PAIR_SUMMARY = DERIV_ROOT / "te_n450" / "te_summary_by_pair_lag.csv"

# =========================================================
# LOAD
# =========================================================
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"No encontrado: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

required = [
    "subject", "cond", "source_roi", "target_roi",
    "lag_samples", "lag_ms", "bins", "te_bits"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Faltan columnas requeridas: {missing}")

print("[ok] loaded rows:", len(df))
print("[ok] subjects:", df["subject"].nunique())
print("[ok] conds:", sorted(df["cond"].unique().tolist()))
print("[ok] lags:", sorted(df["lag_samples"].unique().tolist()))

# =========================================================
# 1) SUBJECT-LEVEL SUMMARY
# =========================================================
# Promedio por sujeto, condición, par y lag
subject_summary = (
    df.groupby(["subject", "cond", "source_roi", "target_roi", "lag_samples", "lag_ms", "bins"], as_index=False)
      .agg(mean_te_bits=("te_bits", "mean"),
           std_te_bits=("te_bits", "std"),
           n_trials=("te_bits", "size"))
)

subject_summary.to_csv(OUT_SUBJECT_SUMMARY, index=False)
print("[ok] saved:", OUT_SUBJECT_SUMMARY)

# =========================================================
# 2) PAIR/LAG SUMMARY ACROSS SUBJECTS
# =========================================================
pair_summary = (
    subject_summary.groupby(["cond", "source_roi", "target_roi", "lag_samples", "lag_ms", "bins"], as_index=False)
    .agg(
        mean_te_bits=("mean_te_bits", "mean"),
        std_te_bits=("mean_te_bits", "std"),
        n_subjects=("subject", "nunique")
    )
)

# Pivot INIT / INHIB y delta
pivot = pair_summary.pivot_table(
    index=["source_roi", "target_roi", "lag_samples", "lag_ms", "bins"],
    columns="cond",
    values="mean_te_bits"
).reset_index()

# Asegurar columnas
if "INIT" not in pivot.columns:
    pivot["INIT"] = np.nan
if "INHIB" not in pivot.columns:
    pivot["INHIB"] = np.nan

pivot["delta_inhib_minus_init"] = pivot["INHIB"] - pivot["INIT"]

# Añadir también SD por condición si quieres
sd_pivot = pair_summary.pivot_table(
    index=["source_roi", "target_roi", "lag_samples", "lag_ms", "bins"],
    columns="cond",
    values="std_te_bits"
).reset_index()

if "INIT" not in sd_pivot.columns:
    sd_pivot["INIT"] = np.nan
if "INHIB" not in sd_pivot.columns:
    sd_pivot["INHIB"] = np.nan

sd_pivot = sd_pivot.rename(columns={
    "INIT": "std_INIT",
    "INHIB": "std_INHIB"
})

pair_out = pd.merge(
    pivot,
    sd_pivot,
    on=["source_roi", "target_roi", "lag_samples", "lag_ms", "bins"],
    how="left"
)

pair_out.to_csv(OUT_PAIR_SUMMARY, index=False)
print("[ok] saved:", OUT_PAIR_SUMMARY)

print("\n=== Mean TE by pair and lag ===")
print(pair_out.sort_values(["lag_samples", "delta_inhib_minus_init"], ascending=[True, False]).to_string(index=False))

# =========================================================
# 3) HEATMAPS OF DELTA BY LAG
# =========================================================
all_rois = sorted(set(df["source_roi"]).union(set(df["target_roi"])))

def make_delta_matrix(df_pair, lag_value):
    tmp = df_pair[df_pair["lag_samples"] == lag_value].copy()

    mat = pd.DataFrame(np.nan, index=all_rois, columns=all_rois)

    for _, row in tmp.iterrows():
        s = row["source_roi"]
        t = row["target_roi"]
        mat.loc[s, t] = row["delta_inhib_minus_init"]

    return mat


for lag in sorted(pair_out["lag_samples"].dropna().unique()):
    mat = make_delta_matrix(pair_out, lag)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(mat.values, aspect="auto", cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im, label="ΔTE = INHIB - INIT (bits)")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.title(f"TE delta matrix (lag = {lag} samples)")
    plt.tight_layout()

    out_png = DERIV_ROOT / "te_n450"/ f"te_delta_heatmap_lag{lag}.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print("[ok] saved:", out_png)

print("\n[done] TE summary and heatmaps generated.")
