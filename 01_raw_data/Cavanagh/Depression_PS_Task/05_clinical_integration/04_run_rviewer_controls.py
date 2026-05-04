#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import mne

from scipy.optimize import curve_fit
from scipy.stats import entropy, ttest_rel

from dds_base.io.paths import DERIV_ROOT, hayling_epo_files, ROIS, EXCLUDE_SUBJECTS

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
OUTDIR = DERIV_ROOT / "reviewer_controls"
OUTDIR.mkdir(parents=True, exist_ok=True)

DDS_MAIN_CSV = DERIV_ROOT / "dds_peak_aligned_n450" / "dds_n450_results.csv"
TRIAL_ROI_CSV = DERIV_ROOT / "trial_roi_timeseries" / "trial_roi_timeseries.csv"

R2_THRESHOLDS = [0.0, 0.3, 0.5]
BINS_TO_TEST = [3, 4, 5]

N450_TMIN_MS = 390
N450_TMAX_MS = 524
SEG_HALF_WIDTH_MS = 200

COND_INV_MAP = {
    "INIT": "ASOC",
    "INHIB": "NOASOC",
}

ALIGNMENT_ROIS = {
    "frontal": ["F3", "F4", "AF3", "AF4", "Fp1", "Fp2", "FC3", "FC4"],
    "cacc": ["FC2", "AFz", "F2"],
}

MAIN_DIRECTIONS = ["frontal->cacc", "cacc->frontal"]

# =========================================================
# DDS MODEL
# =========================================================
def dds_model(t, A1, gamma1, f1, phi1, A2, gamma2, f2, phi2):
    return (
        A1 * np.exp(-gamma1 * t) * np.sin(2 * np.pi * f1 * t + phi1)
        + A2 * np.exp(-gamma2 * t) * np.sin(2 * np.pi * f2 * t + phi2)
    )

def fit_dds_segment(t_sec, y):
    y = np.asarray(y, dtype=float)
    t_sec = np.asarray(t_sec, dtype=float)

    amp = max(np.ptp(y), 1e-7)
    p0 = [
        amp / 2,  5.0,  2.0, 0.0,
        amp / 4, 10.0, 10.0, 0.0
    ]

    lower = [-1e-3, 0.0, 0.5, -np.pi, -1e-3, 0.0, 0.5, -np.pi]
    upper = [ 1e-3, 200.0, 45.0,  np.pi,  1e-3, 200.0, 45.0,  np.pi]

    try:
        popt, _ = curve_fit(
            dds_model,
            t_sec,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000
        )
        y_hat = dds_model(t_sec, *popt)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            "A1": popt[0],
            "gamma1": popt[1],
            "f1": popt[2],
            "phi1": popt[3],
            "A2": popt[4],
            "gamma2": popt[5],
            "f2": popt[6],
            "phi2": popt[7],
            "r2": float(r2),
            "y_hat": y_hat
        }
    except Exception:
        return None

# =========================================================
# INFO THEORY HELPERS
# =========================================================
def safe_qcut(x, q):
    try:
        x_disc = pd.qcut(x, q=q, labels=False, duplicates="drop")
    except Exception:
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

def calculate_ais_shannon(series, bins=4, lag=4):
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

def transfer_entropy_discrete(source, target, lag=4, bins=4):
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)

    if len(x) != len(y) or len(x) <= lag + 1:
        return np.nan
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return np.nan

    x_disc = safe_qcut(x, q=bins)
    y_disc = safe_qcut(y, q=bins)
    if x_disc is None or y_disc is None:
        return np.nan

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

# =========================================================
# STATS HELPERS
# =========================================================
def paired_summary(df, value_col):
    piv = df.pivot(index="subject", columns="cond", values=value_col).dropna()
    if "INIT" not in piv.columns or "INHIB" not in piv.columns:
        return None

    init = piv["INIT"].values
    inhib = piv["INHIB"].values
    t, p = ttest_rel(inhib, init)

    return {
        "n_subjects": int(len(piv)),
        "init_mean": float(np.mean(init)),
        "inhib_mean": float(np.mean(inhib)),
        "delta_inhib_minus_init": float(np.mean(inhib - init)),
        "t": float(t),
        "p": float(p),
    }

# =========================================================
# PART A — SENSITIVITY TO R² THRESHOLD USING EXISTING MAIN FILES
# =========================================================
def build_residual_from_existing_csv(r2_thr=0.0):
    erp = pd.read_csv(TRIAL_ROI_CSV)
    dds = pd.read_csv(DDS_MAIN_CSV)

    dds = dds[dds["r2"] > r2_thr].copy()

    rows = []
    group_cols = ["subject", "cond", "trial", "roi"]

    for keys, g in erp.groupby(group_cols, sort=False):
        sub_val, cond_val, trial_val, roi_val = keys

        row = dds[
            (dds["subject"] == sub_val) &
            (dds["cond"] == cond_val) &
            (dds["trial"] == trial_val) &
            (dds["roi"] == roi_val)
        ]

        if len(row) == 0:
            continue
        row = row.iloc[0]

        t_ms = g["time_ms"].to_numpy(dtype=float)
        t_sec = t_ms / 1000.0
        y = g["value"].to_numpy(dtype=float)

        y_hat = dds_model(
            t_sec,
            row["A1"], row["gamma1"], row["f1"], row["phi1"],
            row["A2"], row["gamma2"], row["f2"], row["phi2"]
        )
        resid = y - y_hat

        tmp = g.copy()
        tmp["dds_r2"] = row["r2"]
        tmp["dds_fit"] = y_hat
        tmp["residual"] = resid
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)

def compute_main_metrics_from_residual(resid_df, bins=4, lag=4):
    if resid_df.empty:
        return None

    # AIS subject-level
    ais_rows = []
    for (subject, cond, roi, trial_uid), g in resid_df.groupby(["subject", "cond", "roi", "trial_uid"], sort=False):
        g = g.sort_values("time_ms")
        ais_val = calculate_ais_shannon(g["residual"].to_numpy(dtype=float), bins=bins, lag=lag)
        ais_rows.append({
            "subject": subject,
            "cond": cond,
            "roi": roi,
            "trial_uid": trial_uid,
            "ais_bits": ais_val
        })
    ais_df = pd.DataFrame(ais_rows)
    ais_subj = ais_df.groupby(["subject", "cond", "roi"], as_index=False)["ais_bits"].mean()

    # TE subject-level
    te_rows = []
    trial_keys = resid_df[["subject", "cond", "trial_uid"]].drop_duplicates()

    for _, tk in trial_keys.iterrows():
        subject = tk["subject"]
        cond = tk["cond"]
        trial_uid = tk["trial_uid"]

        by_roi = {}
        sub = resid_df[
            (resid_df["subject"] == subject) &
            (resid_df["cond"] == cond) &
            (resid_df["trial_uid"] == trial_uid)
        ]
        for roi, g in sub.groupby("roi", sort=False):
            by_roi[roi] = g.sort_values("time_ms")["residual"].to_numpy(dtype=float)

        for src, dst in [("frontal", "cacc"), ("cacc", "frontal")]:
            if src not in by_roi or dst not in by_roi:
                continue
            n = min(len(by_roi[src]), len(by_roi[dst]))
            te_val = transfer_entropy_discrete(by_roi[src][:n], by_roi[dst][:n], lag=lag, bins=bins)
            te_rows.append({
                "subject": subject,
                "cond": cond,
                "direction": f"{src}->{dst}",
                "trial_uid": trial_uid,
                "te_bits": te_val
            })

    te_df = pd.DataFrame(te_rows)
    te_subj = te_df.groupby(["subject", "cond", "direction"], as_index=False)["te_bits"].mean()

    # summaries
    out = []
    for roi in ["frontal", "cacc"]:
        s = paired_summary(ais_subj[ais_subj["roi"] == roi], "ais_bits")
        if s is not None:
            s.update({"metric": "AIS", "target": roi})
            out.append(s)

    for direction in MAIN_DIRECTIONS:
        s = paired_summary(te_subj[te_subj["direction"] == direction], "te_bits")
        if s is not None:
            s.update({"metric": "TE", "target": direction})
            out.append(s)

    return pd.DataFrame(out), ais_subj, te_subj

# =========================================================
# PART B — ALTERNATIVE PEAK ALIGNMENT CONTROL
# =========================================================
def pick_existing_channels(ch_names, wanted):
    return [ch for ch in wanted if ch in ch_names]

def extract_peak_aligned_timeseries_from_epochs(alignment_roi="frontal"):
    rows = []

    files = [f for f in hayling_epo_files() if f.parent.name not in EXCLUDE_SUBJECTS]
    for epo_file in files:
        subject = epo_file.parent.name
        epochs = mne.read_epochs(epo_file, preload=True, verbose="ERROR")

        align_chs = pick_existing_channels(epochs.ch_names, ALIGNMENT_ROIS[alignment_roi])
        if not align_chs:
            continue

        for cond, raw_cond in COND_INV_MAP.items():
            if raw_cond not in epochs.event_id:
                continue

            ep_cond = epochs[raw_cond]
            times_ms = ep_cond.times * 1000.0

            # window for peak search
            search_mask = (times_ms >= N450_TMIN_MS) & (times_ms <= N450_TMAX_MS)
            if search_mask.sum() < 5:
                continue

            for trial_idx in range(len(ep_cond)):
                ep_trial = ep_cond[trial_idx].copy()

                # alignment signal
                align_data = ep_trial.copy().pick(align_chs).get_data()[0].mean(axis=0)
                align_search = align_data[search_mask]
                search_times = times_ms[search_mask]
                peak_i = int(np.argmin(align_search))
                peak_t_ms = float(search_times[peak_i])

                seg_mask = (times_ms >= peak_t_ms - SEG_HALF_WIDTH_MS) & (times_ms <= peak_t_ms + SEG_HALF_WIDTH_MS)
                if seg_mask.sum() < 20:
                    continue

                for roi_name, roi_channels in ROIS.items():
                    roi_chs = pick_existing_channels(ep_cond.ch_names, roi_channels)
                    if not roi_chs:
                        continue

                    roi_data = ep_trial.copy().pick(roi_chs).get_data()[0].mean(axis=0)
                    seg_times_ms = times_ms[seg_mask]
                    seg_values = roi_data[seg_mask]

                    trial_uid = f"{subject}_{cond}_{trial_idx}_{alignment_roi}"

                    for sample_idx, (tm, val) in enumerate(zip(seg_times_ms, seg_values)):
                        rows.append({
                            "subject": subject,
                            "cond": cond,
                            "trial": trial_idx,
                            "trial_uid": trial_uid,
                            "roi": roi_name,
                            "sample_idx": sample_idx,
                            "time_ms": float(tm - (peak_t_ms - SEG_HALF_WIDTH_MS)),  # 0..400
                            "value": float(val),
                            "alignment_roi": alignment_roi,
                            "peak_t_ms_ref": peak_t_ms,
                        })

    return pd.DataFrame(rows)

def fit_dds_on_timeseries_df(ts_df):
    fit_rows = []

    for (subject, cond, trial, trial_uid, roi), g in ts_df.groupby(
        ["subject", "cond", "trial", "trial_uid", "roi"], sort=False
    ):
        g = g.sort_values("time_ms")
        t_sec = g["time_ms"].to_numpy(dtype=float) / 1000.0
        y = g["value"].to_numpy(dtype=float)

        fit = fit_dds_segment(t_sec, y)
        if fit is None:
            continue

        fit_rows.append({
            "subject": subject,
            "cond": cond,
            "trial": trial,
            "trial_uid": trial_uid,
            "roi": roi,
            "A1": fit["A1"],
            "gamma1": fit["gamma1"],
            "f1": fit["f1"],
            "phi1": fit["phi1"],
            "A2": fit["A2"],
            "gamma2": fit["gamma2"],
            "f2": fit["f2"],
            "phi2": fit["phi2"],
            "r2": fit["r2"],
        })

    return pd.DataFrame(fit_rows)

def build_residual_from_fit_df(ts_df, fit_df, r2_thr=0.0):
    fit_df = fit_df[fit_df["r2"] > r2_thr].copy()
    rows = []

    for (subject, cond, trial, trial_uid, roi), g in ts_df.groupby(
        ["subject", "cond", "trial", "trial_uid", "roi"], sort=False
    ):
        row = fit_df[
            (fit_df["subject"] == subject) &
            (fit_df["cond"] == cond) &
            (fit_df["trial"] == trial) &
            (fit_df["trial_uid"] == trial_uid) &
            (fit_df["roi"] == roi)
        ]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        g = g.sort_values("time_ms")
        t_sec = g["time_ms"].to_numpy(dtype=float) / 1000.0
        y = g["value"].to_numpy(dtype=float)
        y_hat = dds_model(
            t_sec,
            row["A1"], row["gamma1"], row["f1"], row["phi1"],
            row["A2"], row["gamma2"], row["f2"], row["phi2"]
        )
        resid = y - y_hat

        tmp = g.copy()
        tmp["dds_r2"] = row["r2"]
        tmp["dds_fit"] = y_hat
        tmp["residual"] = resid
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

# =========================================================
# RUN
# =========================================================
def main():
    summary = {}

    # -----------------------------------------------------
    # 1) Sensitivity to R² threshold
    # -----------------------------------------------------
    r2_tables = []
    print("\n[1] Running R²-threshold sensitivity...")
    for thr in R2_THRESHOLDS:
        resid_df = build_residual_from_existing_csv(r2_thr=thr)
        out = compute_main_metrics_from_residual(resid_df, bins=4, lag=4)
        if out is None:
            continue
        stats_df, ais_subj, te_subj = out
        stats_df["r2_threshold"] = thr
        r2_tables.append(stats_df)

        ais_subj.to_csv(OUTDIR / f"ais_subject_r2gt_{str(thr).replace('.','p')}.csv", index=False)
        te_subj.to_csv(OUTDIR / f"te_subject_r2gt_{str(thr).replace('.','p')}.csv", index=False)

    if r2_tables:
        r2_summary = pd.concat(r2_tables, ignore_index=True)
        r2_summary.to_csv(OUTDIR / "reviewer_r2_threshold_sensitivity_summary.csv", index=False)
        summary["r2_threshold_sensitivity"] = str(OUTDIR / "reviewer_r2_threshold_sensitivity_summary.csv")
        print(r2_summary)

    # -----------------------------------------------------
    # 2) Sensitivity to number of bins on main branch R² > 0
    # -----------------------------------------------------
    print("\n[2] Running bin sensitivity on R² > 0 branch...")
    resid_df_main = build_residual_from_existing_csv(r2_thr=0.0)
    bin_tables = []

    for nb in BINS_TO_TEST:
        out = compute_main_metrics_from_residual(resid_df_main, bins=nb, lag=4)
        if out is None:
            continue
        stats_df, _, _ = out
        stats_df["n_bins"] = nb
        bin_tables.append(stats_df)

    if bin_tables:
        bins_summary = pd.concat(bin_tables, ignore_index=True)
        bins_summary.to_csv(OUTDIR / "reviewer_bin_sensitivity_summary.csv", index=False)
        summary["bin_sensitivity"] = str(OUTDIR / "reviewer_bin_sensitivity_summary.csv")
        print(bins_summary)

    # -----------------------------------------------------
    # 3) Alternative peak alignment control
    # -----------------------------------------------------
    print("\n[3] Running alternative alignment control (this can be slow)...")

    # Main alignment reproduced from epochs
    for align_name in ["frontal", "cacc"]:
        print(f"    - extracting and fitting alignment: {align_name}")
        ts_df = extract_peak_aligned_timeseries_from_epochs(alignment_roi=align_name)
        ts_df.to_csv(OUTDIR / f"trial_roi_timeseries_{align_name}_aligned.csv", index=False)

        fit_df = fit_dds_on_timeseries_df(ts_df)
        fit_df.to_csv(OUTDIR / f"dds_fits_{align_name}_aligned.csv", index=False)

        resid_df = build_residual_from_fit_df(ts_df, fit_df, r2_thr=0.0)
        resid_df.to_csv(OUTDIR / f"residual_{align_name}_aligned_r2gt0.csv", index=False)

        out = compute_main_metrics_from_residual(resid_df, bins=4, lag=4)
        if out is None:
            continue
        stats_df, ais_subj, te_subj = out
        stats_df["alignment_reference"] = align_name
        stats_df.to_csv(OUTDIR / f"reviewer_alignment_control_{align_name}.csv", index=False)
        ais_subj.to_csv(OUTDIR / f"ais_subject_{align_name}_aligned.csv", index=False)
        te_subj.to_csv(OUTDIR / f"te_subject_{align_name}_aligned.csv", index=False)

    # Combine alignment summaries
    align_tables = []
    for align_name in ["frontal", "cacc"]:
        f = OUTDIR / f"reviewer_alignment_control_{align_name}.csv"
        if f.exists():
            align_tables.append(pd.read_csv(f))
    if align_tables:
        align_summary = pd.concat(align_tables, ignore_index=True)
        align_summary.to_csv(OUTDIR / "reviewer_alignment_control_summary.csv", index=False)
        summary["alignment_control"] = str(OUTDIR / "reviewer_alignment_control_summary.csv")
        print(align_summary)

    # -----------------------------------------------------
    # Save manifest
    # -----------------------------------------------------
    with open(OUTDIR / "reviewer_controls_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[done] Reviewer controls written to:", OUTDIR)

if __name__ == "__main__":
    main()
