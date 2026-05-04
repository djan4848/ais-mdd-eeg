#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_extract_ais_n450_residual.py

Calcula AIS trial-by-trial sobre la señal residual:
    residual(t) = ERP(t) - DDS_fit(t)

Usa directamente:
    derivatives/trial_roi_timeseries_residual/trial_roi_timeseries_residual.csv

Mantiene la misma lógica metodológica del AIS original:
- solo N450
- peak-aligned (ya heredado del ajuste DDS)
- AIS por ROI y trial
- discretización por cuantiles
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from pathlib import Path

from dds_base.io.paths import DERIV_ROOT

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
INDIR = DERIV_ROOT / "trial_roi_timeseries_residual_r2pos"
INPUT_CSV = INDIR / "trial_roi_timeseries_residual_r2pos.csv"

OUTDIR = DERIV_ROOT / "ais_n450_residual_r2pos"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = OUTDIR / "ais_n450_residual_r2pos_results.csv"
LOG_WARNINGS = OUTDIR / "ais_n450_residual_r2pos_warnings.txt"
NBINS = 8
LAG = 1
MIN_SAMPLES = 20

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def safe_qcut(x, q):
    try:
        x_disc = pd.qcut(x, q=q, labels=False, duplicates="drop")
    except ValueError:
        return None

    if x_disc is None:
        return None

    x_disc = np.asarray(x_disc, dtype=float)

    if np.all(np.isnan(x_disc)):
        return None

    valid = ~np.isnan(x_disc)
    if valid.sum() < 2:
        return None

    return x_disc.astype(int)


def entropy_from_counts(arr):
    _, counts = np.unique(arr, return_counts=True, axis=0)
    return entropy(counts, base=2)


def calculate_ais_shannon(series, bins=8, lag=1):
    x = np.asarray(series, dtype=float)

    if len(x) <= lag + 1:
        return np.nan

    if np.allclose(np.std(x), 0):
        return np.nan

    x_disc = safe_qcut(x, q=bins)
    if x_disc is None:
        return np.nan

    past = x_disc[:-lag]
    current = x_disc[lag:]

    if len(past) < 2 or len(current) < 2:
        return np.nan

    h_past = entropy_from_counts(past)
    h_curr = entropy_from_counts(current)
    h_joint = entropy_from_counts(np.stack((past, current), axis=1))

    ais = h_past + h_curr - h_joint
    return max(0.0, float(ais))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not INPUT_CSV.exists():
        raise RuntimeError(f"Input file not found: {INPUT_CSV}")

    print(f"Loading residual trial ROI timeseries from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required_cols = [
        "subject", "cond", "trial", "trial_uid", "roi",
        "time_ms", "residual"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in residual CSV: {missing}")

    rows = []
    warnings = []

    group_cols = ["subject", "cond", "trial", "trial_uid", "roi"]

    for keys, g in df.groupby(group_cols, sort=False):
        subj, cond, trial, trial_uid, roi = keys

        g = g.sort_values("time_ms")
        y = g["residual"].to_numpy(dtype=float)

        n_samples = len(y)
        if n_samples < MIN_SAMPLES:
            warnings.append(f"{subj}\t{cond}\ttrial={trial}\t{roi}\tTOO_FEW_SAMPLES")
            continue

        ais_bits = calculate_ais_shannon(y, bins=NBINS, lag=LAG)

        peak_t_ms = np.nan
        peak_amp_ref = np.nan
        window_tmin_ms = float(g["time_ms"].min())
        window_tmax_ms = float(g["time_ms"].max())

        if "peak_t_ms" in g.columns:
            vals = g["peak_t_ms"].dropna().unique()
            if len(vals) > 0:
                peak_t_ms = float(vals[0])

        if "peak_amp_ref" in g.columns:
            vals = g["peak_amp_ref"].dropna().unique()
            if len(vals) > 0:
                peak_amp_ref = float(vals[0])

        rows.append({
            "subject": subj,
            "cond": cond,
            "trial": int(trial),
            "trial_uid": trial_uid,
            "component": "N450_residual",
            "roi": roi,
            "peak_t_ms": peak_t_ms,
            "peak_amp_ref": peak_amp_ref,
            "window_tmin_ms": window_tmin_ms,
            "window_tmax_ms": window_tmax_ms,
            "n_samples": int(n_samples),
            "ais_bits": float(ais_bits) if not np.isnan(ais_bits) else np.nan,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError("AIS residual output DataFrame is empty.")

    out.to_csv(OUT_CSV, index=False)

    if warnings:
        LOG_WARNINGS.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print(f"[warn] warnings written to: {LOG_WARNINGS}")
    elif LOG_WARNINGS.exists():
        LOG_WARNINGS.unlink()

    print("[ok] saved:", OUT_CSV)
    print("[ok] rows:", len(out))
    print("[ok] subjects:", out["subject"].nunique())
    print("[ok] conds:", sorted(out["cond"].dropna().unique().tolist()))
    print("[ok] rois:", sorted(out["roi"].dropna().unique().tolist()))
    print("[ok] mean AIS residual (bits):", round(out["ais_bits"].dropna().mean(), 4))


if __name__ == "__main__":
    main()
