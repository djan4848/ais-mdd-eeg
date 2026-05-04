#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_pid_lh_rh_to_frontal_residual.py

PID exploratorio trial-by-trial sobre la señal residual:
    residual(t) = ERP(t) - DDS_fit(t)

Fuentes:
    lh, rh
Target:
    frontal

Salida:
    derivatives/te_n450_residual/pid_lh_rh_frontal_residual.csv
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

OUT_CSV = OUTDIR / "pid_lh_rh_frontal_residual_r2pos.csv"
LOG_WARNINGS = OUTDIR / "pid_lh_rh_frontal_residual_r2pos_warnings.txt"

NBINS = 8
LAG = 1
MIN_SAMPLES = 20

SRC1 = "lh"
SRC2 = "rh"
TGT = "frontal"

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


def mi_discrete(x, y):
    h_x = entropy_from_counts(x)
    h_y = entropy_from_counts(y)
    h_xy = entropy_from_counts(np.stack((x, y), axis=1))
    mi = h_x + h_y - h_xy
    return max(0.0, float(mi))


def cmi_discrete(x, y, z):
    h_xz = entropy_from_counts(np.stack((x, z), axis=1))
    h_yz = entropy_from_counts(np.stack((y, z), axis=1))
    h_z = entropy_from_counts(z)
    h_xyz = entropy_from_counts(np.stack((x, y, z), axis=1))
    cmi = h_xz + h_yz - h_z - h_xyz
    return max(0.0, float(cmi))


def pid_mmi_two_sources(src1, src2, target):
    """
    PID aproximada tipo MMI:
      redundancy = min(I(S1;T), I(S2;T))
      unique1    = I(S1;T) - redundancy
      unique2    = I(S2;T) - redundancy
      synergy    = I(S1,S2;T) - unique1 - unique2 - redundancy
    """
    i_s1_t = mi_discrete(src1, target)
    i_s2_t = mi_discrete(src2, target)

    h_s1s2 = entropy_from_counts(np.stack((src1, src2), axis=1))
    h_t = entropy_from_counts(target)
    h_s1s2_t = entropy_from_counts(np.stack((src1, src2, target), axis=1))
    i_joint = h_s1s2 + h_t - h_s1s2_t
    i_joint = max(0.0, float(i_joint))

    redundancy = min(i_s1_t, i_s2_t)
    unique1 = max(0.0, i_s1_t - redundancy)
    unique2 = max(0.0, i_s2_t - redundancy)
    synergy = max(0.0, i_joint - redundancy - unique1 - unique2)

    return {
        "mi_s1_t": i_s1_t,
        "mi_s2_t": i_s2_t,
        "mi_joint_t": i_joint,
        "redundancy": redundancy,
        "unique_s1": unique1,
        "unique_s2": unique2,
        "synergy": synergy,
    }


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

    series_dict = {}
    group_cols = ["subject", "cond", "trial", "trial_uid", "roi"]

    for keys, g in df.groupby(group_cols, sort=False):
        subj, cond, trial, trial_uid, roi = keys
        g = g.sort_values("time_ms")

        series_dict[(subj, cond, trial, trial_uid, roi)] = {
            "time_ms": g["time_ms"].to_numpy(dtype=float),
            "y": g["residual"].to_numpy(dtype=float),
            "n_samples": len(g),
        }

    rows = []
    warnings = []

    trial_keys = df[["subject", "cond", "trial", "trial_uid"]].drop_duplicates()

    for _, row0 in trial_keys.iterrows():
        subj = row0["subject"]
        cond = row0["cond"]
        trial = row0["trial"]
        trial_uid = row0["trial_uid"]

        k1 = (subj, cond, trial, trial_uid, SRC1)
        k2 = (subj, cond, trial, trial_uid, SRC2)
        kt = (subj, cond, trial, trial_uid, TGT)

        if k1 not in series_dict or k2 not in series_dict or kt not in series_dict:
            warnings.append(f"{subj}\t{cond}\ttrial={trial}\tmissing PID tuple")
            continue

        s1 = series_dict[k1]
        s2 = series_dict[k2]
        tg = series_dict[kt]

        if min(s1["n_samples"], s2["n_samples"], tg["n_samples"]) < MIN_SAMPLES:
            warnings.append(f"{subj}\t{cond}\ttrial={trial}\ttoo few samples PID tuple")
            continue

        n = min(len(s1["y"]), len(s2["y"]), len(tg["y"]))

        x1 = s1["y"][:n]
        x2 = s2["y"][:n]
        y = tg["y"][:n]
        t = tg["time_ms"][:n]

        # target actual, sources en el pasado
        if n <= LAG + 1:
            warnings.append(f"{subj}\t{cond}\ttrial={trial}\ttoo short after lag")
            continue

        x1_past = x1[:-LAG]
        x2_past = x2[:-LAG]
        y_curr = y[LAG:]

        x1_disc = safe_qcut(x1_past, q=NBINS)
        x2_disc = safe_qcut(x2_past, q=NBINS)
        y_disc = safe_qcut(y_curr, q=NBINS)

        if x1_disc is None or x2_disc is None or y_disc is None:
            warnings.append(f"{subj}\t{cond}\ttrial={trial}\tqcut failed")
            continue

        pid = pid_mmi_two_sources(x1_disc, x2_disc, y_disc)

        rows.append({
            "subject": subj,
            "cond": cond,
            "trial": int(trial),
            "trial_uid": trial_uid,
            "component": "N450_residual",
            "source1_roi": SRC1,
            "source2_roi": SRC2,
            "target_roi": TGT,
            "lag": int(LAG),
            "window_tmin_ms": float(np.min(t)),
            "window_tmax_ms": float(np.max(t)),
            "n_samples": int(n),
            "mi_s1_t": pid["mi_s1_t"],
            "mi_s2_t": pid["mi_s2_t"],
            "mi_joint_t": pid["mi_joint_t"],
            "redundancy": pid["redundancy"],
            "unique_s1": pid["unique_s1"],
            "unique_s2": pid["unique_s2"],
            "synergy": pid["synergy"],
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError("PID residual output DataFrame is empty.")

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
    print("[ok] mean redundancy:", round(out["redundancy"].dropna().mean(), 4))
    print("[ok] mean unique_s1:", round(out["unique_s1"].dropna().mean(), 4))
    print("[ok] mean unique_s2:", round(out["unique_s2"].dropna().mean(), 4))
    print("[ok] mean synergy:", round(out["synergy"].dropna().mean(), 4))


if __name__ == "__main__":
    main()
