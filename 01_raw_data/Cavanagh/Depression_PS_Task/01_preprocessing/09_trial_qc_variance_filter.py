#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09_trial_qc_variance_filter.py

Pre-QC de epochs trial-by-trial ANTES de DDS,
basado en z-score de la varianza dentro de cada:
    subject × cond × roi

Entrada:
    derivatives/trial_roi_timeseries/trial_roi_timeseries.csv

Salidas:
    derivatives/trial_roi_timeseries_qc/trial_roi_timeseries_qc.csv
    derivatives/trial_roi_timeseries_qc/trial_qc_variance_summary.csv
    derivatives/trial_roi_timeseries_qc/trial_qc_variance_flagged.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from dds_base.io.paths import DERIV_ROOT

# --------------------------------------------------
# Config
# --------------------------------------------------
INPUT_CSV = DERIV_ROOT / "trial_roi_timeseries" / "trial_roi_timeseries.csv"

OUTDIR = DERIV_ROOT / "trial_roi_timeseries_qc"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV_CLEAN = OUTDIR / "trial_roi_timeseries_qc.csv"
OUT_CSV_SUMMARY = OUTDIR / "trial_qc_variance_summary.csv"
OUT_CSV_FLAGGED = OUTDIR / "trial_qc_variance_flagged.csv"

Z_THRESH = 3.0

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x, ddof=1)

    if len(x) < 2 or np.isclose(sd, 0):
        return np.zeros_like(x, dtype=float)

    return (x - mu) / sd

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required = ["subject", "cond", "trial", "trial_uid", "roi", "time_ms", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # ----------------------------------------------
    # Trial-level variance
    # ----------------------------------------------
    trial_stats = (
        df.groupby(["subject", "cond", "trial", "trial_uid", "roi"], as_index=False)
          .agg(
              n_samples=("value", "size"),
              var_value=("value", "var"),
              mean_value=("value", "mean"),
              std_value=("value", "std"),
              ptp_value=("value", lambda x: np.max(x) - np.min(x)),
          )
    )

    # si alguna varianza sale NaN por pocos samples
    trial_stats["var_value"] = trial_stats["var_value"].fillna(0.0)
    trial_stats["std_value"] = trial_stats["std_value"].fillna(0.0)
    trial_stats["ptp_value"] = trial_stats["ptp_value"].fillna(0.0)

    # ----------------------------------------------
    # z-score within subject × cond × roi
    # ----------------------------------------------
    group_cols = ["subject", "cond", "roi"]

    z_list = []
    for keys, g in trial_stats.groupby(group_cols, sort=False):
        g = g.copy()
        g["z_var"] = safe_zscore(g["var_value"].values)
        g["flag_outlier_var"] = (np.abs(g["z_var"]) > Z_THRESH).astype(int)
        z_list.append(g)

    trial_stats = pd.concat(z_list, ignore_index=True)

    # flagged trials
    flagged = trial_stats[trial_stats["flag_outlier_var"] == 1].copy()

    # ----------------------------------------------
    # Clean trial list
    # ----------------------------------------------
    keep_trials = trial_stats[trial_stats["flag_outlier_var"] == 0][
        ["subject", "cond", "trial", "trial_uid", "roi"]
    ].copy()

    clean_df = df.merge(
        keep_trials,
        on=["subject", "cond", "trial", "trial_uid", "roi"],
        how="inner"
    )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------
    summary = (
        trial_stats.groupby(["subject", "cond", "roi"], as_index=False)
        .agg(
            n_trials_total=("trial_uid", "size"),
            n_trials_flagged=("flag_outlier_var", "sum"),
        )
    )
    summary["n_trials_kept"] = summary["n_trials_total"] - summary["n_trials_flagged"]
    summary["pct_flagged"] = 100 * summary["n_trials_flagged"] / summary["n_trials_total"]

    global_summary = pd.DataFrame([{
        "subject": "ALL",
        "cond": "ALL",
        "roi": "ALL",
        "n_trials_total": int(len(trial_stats)),
        "n_trials_flagged": int(trial_stats["flag_outlier_var"].sum()),
        "n_trials_kept": int((trial_stats["flag_outlier_var"] == 0).sum()),
        "pct_flagged": 100 * float(trial_stats["flag_outlier_var"].mean()),
    }])

    summary_out = pd.concat([summary, global_summary], ignore_index=True)

    # ----------------------------------------------
    # Save
    # ----------------------------------------------
    clean_df.to_csv(OUT_CSV_CLEAN, index=False)
    summary_out.to_csv(OUT_CSV_SUMMARY, index=False)
    flagged.to_csv(OUT_CSV_FLAGGED, index=False)

    # ----------------------------------------------
    # Print
    # ----------------------------------------------
    print("\n================ QC VARIANCE FILTER ================\n")
    print("Total trial×ROI units:", len(trial_stats))
    print("Flagged outliers      :", int(trial_stats["flag_outlier_var"].sum()))
    print("Kept                  :", int((trial_stats["flag_outlier_var"] == 0).sum()))
    print("Percent flagged       :", round(100 * trial_stats["flag_outlier_var"].mean(), 2), "%")

    print("\nSaved clean CSV:")
    print(OUT_CSV_CLEAN)

    print("\nSaved summary CSV:")
    print(OUT_CSV_SUMMARY)

    print("\nSaved flagged trials CSV:")
    print(OUT_CSV_FLAGGED)

    print("\nTop 10 subject/cond/roi with highest % flagged:")
    print(
        summary.sort_values("pct_flagged", ascending=False)
               .head(10)
               .to_string(index=False)
    )


if __name__ == "__main__":
    main()
