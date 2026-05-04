#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_extract_te_n450_residual.py

Calcula Transfer Entropy trial-by-trial sobre la señal residual:
    residual(t) = ERP(t) - DDS_fit(t)

Entrada:
    derivatives/trial_roi_timeseries_residual/trial_roi_timeseries_residual.csv

Salida:
    derivatives/te_n450_residual/te_n450_residual_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy

from dds_base.io.paths import DERIV_ROOT

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
INDIR = DERIV_ROOT / "trial_roi_timeseries_residual_r2pos"
INPUT_CSV = INDIR / "trial_roi_timeseries_residual_r2pos.csv"

OUTDIR = DERIV_ROOT / "te_n450_residual_r2pos"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = OUTDIR / "te_n450_residual_r2pos_results.csv"
LOG_WARNINGS = OUTDIR / "te_n450_residual_r2pos_warnings.txt"

NBINS = 8
LAG = 1
MIN_SAMPLES = 20

# pares dirigidos que nos interesan
ROI_PAIRS = [
    ("frontal", "cacc"),
    ("cacc", "frontal"),
    ("lh", "frontal"),
    ("rh", "frontal"),
    ("lh", "cacc"),
    ("rh", "cacc"),
]

# ---------------------------------------------------------------------
# Utils
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


def transfer_entropy_discrete(source, target, lag=1, bins=8):
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)

    if len(x) != len(y):
        return np.nan

    if len(x) <= lag + 1:
        return np.nan

    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return np.nan

    x_disc = safe_qcut(x, q=bins)
    y_disc = safe_qcut(y, q=bins)

    if x_disc is None or y_disc is None:
        return np.nan

    # y_t, y_t-1, x_t-1
    y_t = y_disc[lag:]
    y_past = y_disc[:-lag]
    x_past = x_disc[:-lag]

    if len(y_t) < 2:
        return np.nan

    h_y_t_y_past = entropy_from_counts(np.stack((y_t, y_past), axis=1))
    h_y_past_x_past = entropy_from_counts(np.stack((y_past, x_past), axis=1))
    h_y_past = entropy_from_counts(y_past)
    h_y_t_y_past_x_past = entropy_from_counts(np.stack((y_t, y_past, x_past), axis=1))

    te = h_y_t_y_past + h_y_past_x_past - h_y_past - h_y_t_y_past_x_past
    return max(0.0, float(te))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not INPUT_CSV.exists():
        raise RuntimeError(f"Input file not found: {INPUT_CSV}")

    print(f"Loading residual trial ROI timeseries from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["subject", "cond", "trial", "trial_uid", "roi", "time_ms", "residual"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in residual CSV: {missing}")

    # guardar por trial para acceso rápido
    series_dict = {}
    base_group_cols = ["subject", "cond", "trial", "trial_uid", "roi"]

    for keys, g in df.groupby(base_group_cols, sort=False):
        subj, cond, trial, trial_uid, roi = keys
        g = g.sort_values("time_ms")
        y = g["residual"].to_numpy(dtype=float)
        t = g["time_ms"].to_numpy(dtype=float)

        series_dict[(subj, cond, trial, trial_uid, roi)] = {
            "time_ms": t,
            "y": y,
            "n_samples": len(y),
        }

    rows = []
    warnings = []

    # base trial identities
    trial_keys = df[["subject", "cond", "trial", "trial_uid"]].drop_duplicates()

    for _, row0 in trial_keys.iterrows():
        subj = row0["subject"]
        cond = row0["cond"]
        trial = row0["trial"]
        trial_uid = row0["trial_uid"]

        for src_roi, dst_roi in ROI_PAIRS:
            src_key = (subj, cond, trial, trial_uid, src_roi)
            dst_key = (subj, cond, trial, trial_uid, dst_roi)

            if src_key not in series_dict or dst_key not in series_dict:
                warnings.append(f"{subj}\t{cond}\ttrial={trial}\tmissing pair {src_roi}->{dst_roi}")
                continue

            src = series_dict[src_key]
            dst = series_dict[dst_key]

            if src["n_samples"] < MIN_SAMPLES or dst["n_samples"] < MIN_SAMPLES:
                warnings.append(f"{subj}\t{cond}\ttrial={trial}\ttoo few samples {src_roi}->{dst_roi}")
                continue

            # recorte común por seguridad
            n = min(len(src["y"]), len(dst["y"]))
            x = src["y"][:n]
            y = dst["y"][:n]
            t = dst["time_ms"][:n]

            te_bits = transfer_entropy_discrete(x, y, lag=LAG, bins=NBINS)

            rows.append({
                "subject": subj,
                "cond": cond,
                "trial": int(trial),
                "trial_uid": trial_uid,
                "component": "N450_residual",
                "source_roi": src_roi,
                "target_roi": dst_roi,
                "direction": f"{src_roi}->{dst_roi}",
                "lag": int(LAG),
                "window_tmin_ms": float(np.min(t)),
                "window_tmax_ms": float(np.max(t)),
                "n_samples": int(n),
                "te_bits": float(te_bits) if not np.isnan(te_bits) else np.nan,
            })

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError("TE residual output DataFrame is empty.")

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
    print("[ok] directions:", sorted(out["direction"].dropna().unique().tolist()))
    print("[ok] mean TE residual (bits):", round(out["te_bits"].dropna().mean(), 4))


if __name__ == "__main__":
    main()
