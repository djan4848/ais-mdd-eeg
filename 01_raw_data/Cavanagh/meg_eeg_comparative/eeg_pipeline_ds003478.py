"""
eeg_pipeline_ds003478.py
========================
Applies the full ds003474 pipeline (Baseline spectral + DDS+Info+Baseline)
to ds003478 — resting-state EEG from the SAME participants as ds003474.

This enables the cleanest possible comparison:
  ds003474  (PST task EEG,   same people, same equipment) → AUC 0.836
  ds003478  (resting EEG,    same people, same equipment) → AUC ???

If resting > task → task activity masks the spectral biomarkers.
If resting ≈ task → the biomarkers are task-independent.

ds003478 specifics
------------------
- N = 122 (same participants as ds003474, matched by Original_ID)
- 64 EEG channels (10-20, same as ds003474)
- 500 Hz, EEGLAB .set format, Neuroscan Synamps2
- Task: "Rest"  (eyes open/closed alternating)
- run-01 : 6 min BEFORE the PST task (immediately after hook-up)
- run-02 : 6 min AFTER the PST task (~1 h later)
- Groups : BDI threshold = 13 (same as ds003474)

Usage
-----
    python eeg_pipeline_ds003478.py              # full pipeline, run-01
    python eeg_pipeline_ds003478.py --run 2      # use run-02
    python eeg_pipeline_ds003478.py --skip_extract
    python eeg_pipeline_ds003478.py --max_subjects 10   # debug
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.signal as sci_signal
from scipy.optimize import curve_fit
from scipy.signal import resample_poly
from scipy.stats import ttest_ind
from math import gcd

import mne

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Add ds003474 code dir to path so we can reuse preprocessing ──────────────
DS003474_CODE = Path(
    "/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds003474/code"
    "/eeg_depression_classification"
)
sys.path.insert(0, str(DS003474_CODE))

from preprocessing import preprocess_raw   # noqa: E402  (after sys.path insert)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BIDS_ROOT   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds003478")
OUTPUT_DIR  = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/meg_eeg_comparative")

# ──────────────────────────────────────────────────────────────────────────────
# Parameters  (mirroring ds003474/config.py)
# ──────────────────────────────────────────────────────────────────────────────
SFREQ        = 500.0
HIGHPASS     = 1.0
LOWPASS      = 45.0
POWERLINE    = 60.0
EPOCH_DUR    = 2.0      # s
EPOCH_OVERLAP = 0.0
AMPLITUDE_THRESH = 150e-6
BDI_THRESHOLD    = 13

FREQ_BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ROIs for DDS (same as ds003474 config)
ROI_MAP = {
    "frontal": ["F3", "F4", "AF3", "AF4", "FP1", "FP2", "FC3", "FC4"],
    "cACC":    ["FC2", "AFZ", "F2"],
    "LH":      ["F3", "F5", "FC3", "FC5"],
    "RH":      ["F4", "F6", "FC4", "FC6"],
}

# DDS parameters
DDS_WIN_MS   = 400
DDS_SFREQ    = 250.0
DDS_WIN_N    = int(DDS_WIN_MS / 1000.0 * DDS_SFREQ)   # 100 samples
DDS_MIN_FREQ = 0.5
DDS_MAX_FREQ = 45.0
DDS_MAX_ALPHA = 500.0
DDS_MAX_NFEV  = 3000
DDS_MAX_WINS  = 100

# Info-theory parameters
AIS_K    = 1
AIS_BINS = 8
INFO_LAG = 4
INFO_BINS = 4
TE_PAIRS = [("frontal", "cACC"), ("cACC", "frontal"),
            ("LH", "frontal"),   ("RH", "frontal")]
PID_SRC  = ["LH", "RH"]
PID_TGT  = "frontal"

# Classification
CV_FOLDS     = 5
RANDOM_STATE = 42
N_JOBS       = -1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Group labels from participants.tsv (BDI threshold = 13)
# ══════════════════════════════════════════════════════════════════════════════

def load_group_labels() -> pd.DataFrame:
    df = pd.read_csv(str(BIDS_ROOT / "participants.tsv"), sep="\t")
    rows = []
    for _, row in df.iterrows():
        sub_id = str(row["participant_id"]).strip()
        bdi = float(row["BDI"]) if pd.notna(row["BDI"]) else np.nan
        if np.isnan(bdi):
            continue
        group = 1 if bdi > BDI_THRESHOLD else 0
        rows.append({"subject_id": sub_id, "group": group, "BDI": bdi,
                     "Original_ID": int(row["Original_ID"])})
    df_out = pd.DataFrame(rows)
    log.info("Group labels: CTL=%d  MDD=%d  (BDI>%d threshold)",
             (df_out.group == 0).sum(), (df_out.group == 1).sum(), BDI_THRESHOLD)
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load + preprocess one subject (reuses ds003474 preprocessing)
# ══════════════════════════════════════════════════════════════════════════════

def load_subject(subject_id: str, run: int = 1) -> Optional[mne.io.BaseRaw]:
    """
    Load ds003478 resting EEG for one subject.
    Returns cleaned MNE Raw or None.
    """
    set_path = (BIDS_ROOT / subject_id / "eeg"
                / f"{subject_id}_task-Rest_run-{run:02d}_eeg.set")
    elec_tsv = (BIDS_ROOT / subject_id / "eeg"
                / f"{subject_id}_task-Rest_run-{run:02d}_electrodes.tsv")

    if not set_path.exists():
        log.warning("  [%s] .set file not found: %s", subject_id, set_path.name)
        return None

    raw, report = preprocess_raw(
        set_path=set_path,
        elec_tsv=elec_tsv,
        highpass=HIGHPASS,
        lowpass=LOWPASS,
        powerline=POWERLINE,
        run_ica_flag=True,
    )
    if raw is None:
        log.warning("  [%s] Excluded: %s", subject_id, report["exclusion_reason"])
        return None

    log.info("  [%s] bad_ch=%d  ICA_removed=%d  sfreq=%.0f  dur=%.0fs",
             subject_id, report["n_bad_ch"], report["n_ica_removed"],
             raw.info["sfreq"], raw.times[-1])
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Baseline feature extraction  (vectorised Welch + Hjorth)
# ══════════════════════════════════════════════════════════════════════════════

def extract_baseline_features(raw: mne.io.BaseRaw) -> dict:
    """
    Extract spectral + Hjorth features from continuous resting EEG.
    Same logic as meg_pipeline_ds005356.py (vectorised, ~1 s per subject).

    Per channel (71→64 ch):
      delta/theta/alpha/beta/gamma absolute power   5
      delta/theta/alpha/beta/gamma relative power   5
      alpha/beta, theta/beta, theta/alpha ratios     3
      Hjorth activity, mobility, complexity          3
    Regional mean alpha/beta and theta/beta (4 ROIs) 8
    Frontal alpha asymmetry (FAA, 4 pairs)            4
    ── Total: 64 × 16 + 12 = 1036 + 12 = 1048 features ──
    """
    data  = raw.get_data()
    sfreq = raw.info["sfreq"]
    ch_names = [c.upper() for c in raw.ch_names]
    n_ch, n_times = data.shape

    win_len = int(EPOCH_DUR * sfreq)
    hop_len = int((EPOCH_DUR - EPOCH_OVERLAP) * sfreq)
    nperseg = min(win_len, int(sfreq * 2))

    dummy_f = sci_signal.welch(np.zeros(win_len), fs=sfreq, nperseg=nperseg)[0]
    band_masks = {band: (dummy_f >= flo) & (dummy_f <= fhi)
                  for band, (flo, fhi) in FREQ_BANDS.items()}

    acc: dict[str, list] = {}
    windows_used = 0

    for start in range(0, n_times - win_len + 1, hop_len):
        w = data[:, start: start + win_len]
        pk2pk = w.max(axis=1) - w.min(axis=1)
        if (pk2pk > AMPLITUDE_THRESH).any():
            continue
        windows_used += 1

        _, psd = sci_signal.welch(w, fs=sfreq, nperseg=nperseg, axis=1)
        band_pow = {}
        for band, mask in band_masks.items():
            bp = np.trapz(psd[:, mask], dummy_f[mask], axis=1)
            band_pow[band] = np.maximum(bp, 0.0)
        total_pow = sum(band_pow.values())

        for band, bp in band_pow.items():
            acc.setdefault(f"{band}_abs", []).append(bp.copy())
        with np.errstate(invalid="ignore", divide="ignore"):
            for band, bp in band_pow.items():
                acc.setdefault(f"{band}_rel", []).append(
                    np.where(total_pow > 0, bp / total_pow, np.nan))
            a = band_pow["alpha"]; b = band_pow["beta"]; t = band_pow["theta"]
            acc.setdefault("alpha_beta_ratio", []).append(
                np.where(b > 0, a / b, np.nan))
            acc.setdefault("theta_beta_ratio", []).append(
                np.where(b > 0, t / b, np.nan))
            acc.setdefault("theta_alpha_ratio", []).append(
                np.where(a > 0, t / a, np.nan))

        d1 = np.diff(w, axis=1); d2 = np.diff(d1, axis=1)
        act = w.var(axis=1); var_d1 = d1.var(axis=1); var_d2 = d2.var(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mob = np.sqrt(np.where(act > 0, var_d1 / act, np.nan))
            com = np.where((mob > 0) & (var_d1 > 0),
                           np.sqrt(var_d2 / var_d1) / mob, np.nan)
        acc.setdefault("hjorth_act", []).append(act)
        acc.setdefault("hjorth_mob", []).append(mob)
        acc.setdefault("hjorth_com", []).append(com)

    if windows_used == 0:
        return {}

    # Average over windows
    feats = {}
    for feat_name, arrays in acc.items():
        mat = np.nanmean(np.stack(arrays, axis=0), axis=0)  # (n_ch,)
        for ci, val in enumerate(mat):
            feats[f"{feat_name}_ch{ci:03d}"] = float(val)

    log.info("  Baseline: %d windows, %d features", windows_used, len(feats))
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DDS fitting  (same as dds_pipeline_ds005356.py)
# ══════════════════════════════════════════════════════════════════════════════

def _dds_signal(t, A1, a1, f1, p1, A2, a2, f2, p2):
    return (A1 * np.exp(-a1 * t) * np.cos(2 * np.pi * f1 * t + p1) +
            A2 * np.exp(-a2 * t) * np.cos(2 * np.pi * f2 * t + p2))


def _initial_params(window, sfreq):
    from scipy.signal import welch as _welch
    n = len(window)
    f_ax, pxx = _welch(window, fs=sfreq, nperseg=min(n, 64))
    mask = (f_ax >= DDS_MIN_FREQ) & (f_ax <= DDS_MAX_FREQ)
    f_v, p_v = f_ax[mask], pxx[mask]
    if len(p_v) < 2:
        f1i, f2i = 10.0, 20.0
    else:
        order = np.argsort(p_v)[::-1]
        f1i = float(f_v[order[0]])
        f2i = next((float(f_v[i]) for i in order[1:]
                    if abs(f_v[i] - f1i) >= 2.0), f1i * 2.0)
        f2i = float(np.clip(f2i, DDS_MIN_FREQ, DDS_MAX_FREQ))
    A = float(np.std(window)) + 1e-12
    return [A * 0.7, 5.0, f1i, 0.0, A * 0.3, 10.0, f2i, 0.0]


def _fit_window(window, sfreq):
    n = len(window)
    t = np.arange(n) / sfreq
    p0 = _initial_params(window, sfreq)
    lo = [0, 0, DDS_MIN_FREQ, -np.pi, 0, 0, DDS_MIN_FREQ, -np.pi]
    hi = [np.inf, DDS_MAX_ALPHA, DDS_MAX_FREQ, np.pi,
          np.inf, DDS_MAX_ALPHA, DDS_MAX_FREQ, np.pi]
    p0 = [np.clip(v, l + 1e-8, max(l + 1e-8, h - 1e-8))
          for v, l, h in zip(p0, lo, hi)]
    try:
        popt, _ = curve_fit(_dds_signal, t, window, p0=p0,
                             bounds=(lo, hi), method="trf",
                             max_nfev=DDS_MAX_NFEV,
                             ftol=1e-8, xtol=1e-8, gtol=1e-8)
    except Exception:
        return None
    fitted = _dds_signal(t, *popt)
    res = window - fitted
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((window - window.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0
    return dict(A1=popt[0], alpha1=popt[1], f1=popt[2], phi1=popt[3],
                A2=popt[4], alpha2=popt[5], f2=popt[6], phi2=popt[7],
                r2=r2, residual_rms=float(np.sqrt(np.mean(res ** 2))),
                residual=res)


def _roi_signal(data, ch_names_upper, roi_chs):
    idx = [ch_names_upper.index(r.upper()) for r in roi_chs
           if r.upper() in ch_names_upper]
    return data[np.array(idx)].mean(axis=0) if idx else None


def compute_dds_features(raw: mne.io.BaseRaw) -> tuple[dict, dict]:
    sfreq = raw.info["sfreq"]
    data  = raw.get_data()
    ch_upper = [c.upper() for c in raw.ch_names]
    n_ch, n_times = data.shape

    # Resample to DDS_SFREQ
    if abs(sfreq - DDS_SFREQ) > 1.0:
        fs_int = int(round(sfreq))
        tg_int = int(round(DDS_SFREQ))
        g = gcd(fs_int, tg_int)
        data = resample_poly(data, tg_int // g, fs_int // g, axis=1)
        sfreq = DDS_SFREQ

    n_times = data.shape[1]
    hop = DDS_WIN_N
    starts = list(range(0, n_times - DDS_WIN_N + 1, hop))
    if len(starts) > DDS_MAX_WINS:
        rng = np.random.default_rng(RANDOM_STATE)
        starts = sorted(rng.choice(starts, DDS_MAX_WINS, replace=False).tolist())

    param_keys = ["A1", "alpha1", "f1", "phi1", "A2", "alpha2", "f2", "phi2",
                  "r2", "residual_rms"]
    roi_acc: dict[str, dict] = {}
    residuals: dict[str, list] = {}

    for roi_name, roi_chs in ROI_MAP.items():
        acc = {k: [] for k in param_keys}
        res_list = []
        roi_sig = _roi_signal(data, ch_upper, roi_chs)
        if roi_sig is None:
            roi_acc[roi_name] = acc; residuals[roi_name] = res_list; continue
        for s in starts:
            win = roi_sig[s: s + DDS_WIN_N]
            win = win - win.mean()
            if np.ptp(win) < 1e-15:
                continue
            fit = _fit_window(win, sfreq)
            if fit is None:
                continue
            for k in param_keys:
                acc[k].append(fit[k])
            res_list.append(fit["residual"])
        roi_acc[roi_name] = acc; residuals[roi_name] = res_list

    feats = {}
    for roi_name, acc in roi_acc.items():
        n_ok = len(acc["r2"])
        for k in param_keys:
            feats[f"dds_{roi_name}_{k}"] = float(np.nanmean(acc[k])) if n_ok else np.nan
        if n_ok:
            a1m = float(np.nanmean(acc["A1"])); a2m = float(np.nanmean(acc["A2"]))
            feats[f"dds_{roi_name}_A_ratio"]   = a1m / (a1m + a2m + 1e-30)
            feats[f"dds_{roi_name}_f_diff"]     = float(np.nanmean(
                np.abs(np.array(acc["f1"]) - np.array(acc["f2"]))))
            feats[f"dds_{roi_name}_alpha_mean"] = float(
                np.nanmean(np.array(acc["alpha1"]) + np.array(acc["alpha2"])) / 2)
            feats[f"dds_{roi_name}_n_wins_ok"]  = n_ok
        else:
            feats[f"dds_{roi_name}_A_ratio"]   = np.nan
            feats[f"dds_{roi_name}_f_diff"]     = np.nan
            feats[f"dds_{roi_name}_alpha_mean"] = np.nan
            feats[f"dds_{roi_name}_n_wins_ok"]  = 0
    log.info("  DDS: %d wins OK across ROIs.",
             sum(len(residuals[r]) for r in ROI_MAP))
    return feats, residuals


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Information theory  (AIS / TE / PID on DDS residuals)
# ══════════════════════════════════════════════════════════════════════════════

def _discretise(x, n_bins=INFO_BINS):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=int)
    for i, q in enumerate(np.linspace(0, 100, n_bins + 1)[1:-1]):
        out[x > np.percentile(x, q)] += 1
    return out

def _mi_hist(x, y, n_bins=AIS_BINS):
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    joint, _, _ = np.histogram2d(x, y, bins=n_bins)
    joint /= joint.sum() + 1e-30
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float(max(np.sum(joint[mask] * np.log2(
        joint[mask] / (px * py + 1e-30)[mask] + 1e-30)), 0.0))

def compute_ais(sig, k=AIS_K):
    return _mi_hist(sig[:-k], sig[k:], n_bins=AIS_BINS) if len(sig) > k + 2 else np.nan

def compute_te(source, target, lag=INFO_LAG):
    n = min(len(source), len(target)) - lag
    if n < 10:
        return np.nan
    s = _discretise(source[:n]); yt = _discretise(target[:n])
    yf = _discretise(target[lag:lag + n])
    jyy, _, _ = np.histogram2d(yt, yf, bins=INFO_BINS)
    pyt = jyy.sum(axis=1, keepdims=True)
    jyy_n = jyy / (jyy.sum() + 1e-30)
    pyt_n = pyt / (pyt.sum() + 1e-30)
    h_yf_yt = -float(np.sum(jyy_n[jyy_n > 0] * np.log2(
        jyy_n[jyy_n > 0] / (np.tile(pyt_n, (1, INFO_BINS))[jyy_n > 0] + 1e-30) + 1e-30)))
    state_id = yt * INFO_BINS + s
    h_yf_yt_s = 0.0
    for sid in range(INFO_BINS ** 2):
        mask = state_id == sid
        if mask.sum() < 2:
            continue
        p_state = mask.sum() / n
        counts = np.bincount(yf[mask], minlength=INFO_BINS).astype(float)
        counts /= counts.sum() + 1e-30
        h_yf_yt_s += p_state * (-float(np.sum(
            counts[counts > 0] * np.log2(counts[counts > 0] + 1e-30))))
    return max(h_yf_yt - h_yf_yt_s, 0.0)

def compute_pid(s1, s2, target, lag=INFO_LAG):
    n = min(len(s1), len(s2), len(target)) - lag
    if n < 10:
        return dict(pid_redundancy=np.nan, pid_unique_LH=np.nan,
                    pid_unique_RH=np.nan, pid_synergy=np.nan, pid_total_mi=np.nan)
    s1d = _discretise(s1[:n]); s2d = _discretise(s2[:n])
    tgtd = _discretise(target[lag:lag + n])
    mi1 = _mi_hist(s1d, tgtd); mi2 = _mi_hist(s2d, tgtd)
    p_tgt = np.bincount(tgtd, minlength=INFO_BINS).astype(float)
    p_tgt /= p_tgt.sum() + 1e-30
    red = 0.0
    for tv in range(INFO_BINS):
        if p_tgt[tv] == 0:
            continue
        mask_t = tgtd == tv
        def _ss(sx):
            p_sx_t = np.bincount(sx[mask_t], minlength=INFO_BINS).astype(float)
            p_sx_t /= p_sx_t.sum() + 1e-30
            p_sx_a = np.bincount(sx[:n], minlength=INFO_BINS).astype(float)
            p_sx_a /= p_sx_a.sum() + 1e-30
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(p_sx_a > 0, p_sx_t / p_sx_a, 0)
                return max(float(np.sum(p_sx_t * np.log2(ratio + 1e-30))), 0.0)
        red += p_tgt[tv] * min(_ss(s1d), _ss(s2d))
    mi_j = _mi_hist(s1d * INFO_BINS + s2d, tgtd, n_bins=INFO_BINS ** 2)
    return dict(pid_redundancy=red, pid_unique_LH=max(mi1 - red, 0.0),
                pid_unique_RH=max(mi2 - red, 0.0),
                pid_synergy=max(mi_j - mi1 - mi2 + red, 0.0),
                pid_total_mi=mi_j)

def compute_info_features(residuals: dict) -> dict:
    roi_sigs = {r: np.concatenate(v) for r, v in residuals.items() if v}
    feats = {}
    for roi, sig in roi_sigs.items():
        feats[f"ais_{roi}"] = compute_ais(sig)
    for src, tgt in TE_PAIRS:
        s_sig = roi_sigs.get(src); t_sig = roi_sigs.get(tgt)
        feats[f"te_{src}_{tgt}"] = (compute_te(s_sig, t_sig)
                                     if s_sig is not None and t_sig is not None
                                     else np.nan)
    s1 = roi_sigs.get(PID_SRC[0]); s2 = roi_sigs.get(PID_SRC[1])
    tg = roi_sigs.get(PID_TGT)
    feats.update(compute_pid(s1, s2, tg) if (s1 is not None and s2 is not None
                                              and tg is not None)
                 else dict(pid_redundancy=np.nan, pid_unique_LH=np.nan,
                           pid_unique_RH=np.nan, pid_synergy=np.nan,
                           pid_total_mi=np.nan))
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Batch extraction with incremental caching
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(labels_df: pd.DataFrame, run: int,
                   baseline_csv: Path, dds_csv: Path, info_csv: Path,
                   max_subjects: int = 0) -> None:
    label_dict = labels_df.set_index("subject_id").to_dict("index")

    # Incremental cache — with dedup guard to prevent partial-run contamination
    def _load_deduped(csv_path):
        if not csv_path.exists():
            return set(), []
        df = pd.read_csv(csv_path, dtype={"subject_id": str})
        n_before = len(df)
        df = df.drop_duplicates(subset="subject_id", keep="last")
        if len(df) < n_before:
            log.warning("%s: dropped %d duplicate rows.", csv_path.name, n_before - len(df))
            df.to_csv(str(csv_path), index=False)
        return set(df["subject_id"].tolist()), df.to_dict("records")

    done_base, base_recs = _load_deduped(baseline_csv)
    done_dds,  dds_recs  = _load_deduped(dds_csv)
    done_info, info_recs = _load_deduped(info_csv)
    if done_base:
        log.info("Resuming: %d subjects already cached.", len(done_base))

    subjects = sorted(label_dict.keys())
    if max_subjects > 0:
        subjects = subjects[:max_subjects]

    for sub_id in subjects:
        need_base = sub_id not in done_base
        need_dds  = sub_id not in done_dds
        need_info = sub_id not in done_info

        if not need_base and not need_dds and not need_info:
            log.info("  [%s] Fully cached – skipping.", sub_id)
            continue

        log.info("Processing %s (run-%02d) …", sub_id, run)
        raw = load_subject(sub_id, run=run)
        if raw is None:
            continue

        lbl = label_dict[sub_id]
        meta = {"subject_id": sub_id, "group": lbl["group"], "BDI": lbl["BDI"]}

        if need_base:
            base_feats = extract_baseline_features(raw)
            if base_feats:
                base_recs.append({**meta, **base_feats})
                pd.DataFrame(base_recs).drop_duplicates(subset="subject_id", keep="last").to_csv(
                    str(baseline_csv), index=False)

        if need_dds or need_info:
            dds_feats, residuals = compute_dds_features(raw)
            if need_dds and dds_feats:
                dds_recs.append({**meta, **dds_feats})
                pd.DataFrame(dds_recs).drop_duplicates(subset="subject_id", keep="last").to_csv(
                    str(dds_csv), index=False)
            if need_info:
                info_feats = compute_info_features(residuals)
                info_recs.append({**meta, **info_feats})
                pd.DataFrame(info_recs).drop_duplicates(subset="subject_id", keep="last").to_csv(
                    str(info_csv), index=False)

        log.info("  [%s] Done. group=%s", sub_id,
                 "MDD" if lbl["group"] == 1 else "CTL")

    log.info("Baseline: %d subjects | DDS: %d | Info: %d",
             len(base_recs), len(dds_recs), len(info_recs))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Classification  (same pipeline as ds003474 / ds005356)
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(clf, use_smote: bool, n_pca: int):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        HAS_SMOTE = True
    except ImportError:
        HAS_SMOTE = False
    steps = [("imputer", SimpleImputer(strategy="mean")),
             ("scaler",  StandardScaler()),
             ("var_thr", VarianceThreshold(threshold=0.0)),
             ("pca",     PCA(n_components=n_pca, random_state=RANDOM_STATE)),
             ("clf",     clf)]
    if use_smote and HAS_SMOTE:
        steps.insert(3, ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)))
        return ImbPipeline(steps=steps)
    return Pipeline(steps=steps)


def run_classification(feature_sets: dict, output_dir: Path,
                       use_smote: bool = True, n_pca: int = 40) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                                 f1_score, confusion_matrix)
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    clfs = {
        "LogReg":  LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                       class_weight="balanced", random_state=RANDOM_STATE),
        "LDA":     LDA(solver="svd"),
        "SVM-Lin": SVC(kernel="linear", C=0.1, probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE),
        "SVM-RBF": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE),
        "RF":      RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                           class_weight="balanced", n_jobs=N_JOBS,
                                           random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        clfs["XGB"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     eval_metric="logloss", random_state=RANDOM_STATE,
                                     n_jobs=N_JOBS, verbosity=0)

    meta_cols = {"subject_id", "group", "BDI", "Original_ID"}
    all_results = []

    for fs_name, df_feats in feature_sets.items():
        if df_feats.empty:
            continue
        fcols = [c for c in df_feats.columns if c not in meta_cols]
        nan_frac = df_feats[fcols].isnull().mean()
        fcols = [c for c in fcols if nan_frac[c] <= 0.10]
        df_feats = df_feats.dropna(subset=["group"])

        X = df_feats[fcols].values.astype(float)
        y = df_feats["group"].values.astype(int)

        n_train = int(len(y) * (CV_FOLDS - 1) / CV_FOLDS) - 1
        n_pca_safe = min(n_pca, n_train, X.shape[1])

        log.info("  [%s] %d subjects, %d features (CTL=%d, MDD=%d)",
                 fs_name, len(y), len(fcols), (y == 0).sum(), (y == 1).sum())

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for clf_name, clf in clfs.items():
            pipe = build_pipeline(clf, use_smote=use_smote, n_pca=n_pca_safe)
            try:
                y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict", n_jobs=N_JOBS)
                try:
                    y_prob = cross_val_predict(pipe, X, y, cv=cv,
                                               method="predict_proba", n_jobs=N_JOBS)[:, 1]
                    auc = roc_auc_score(y, y_prob)
                except Exception:
                    auc = 0.5
                cm = confusion_matrix(y, y_pred)
                tn, fp, fn, tp = cm.ravel()
                res = {"feature_set": fs_name, "classifier": clf_name,
                       "roc_auc": round(auc, 3),
                       "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 3),
                       "f1": round(f1_score(y, y_pred, zero_division=0), 3),
                       "sensitivity": round(tp / (tp + fn + 1e-9), 3),
                       "specificity": round(tn / (tn + fp + 1e-9), 3),
                       "n_subjects": len(y), "n_features": len(fcols)}
                log.info("    %-10s %-20s AUC=%.3f  BalAcc=%.3f",
                         clf_name, "", auc, res["balanced_accuracy"])
                all_results.append(res)
            except Exception as exc:
                log.warning("    %s failed: %s", clf_name, exc)

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(str(output_dir / "ds003478_classification_results.csv"), index=False)
    return df_res


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Group statistics
# ══════════════════════════════════════════════════════════════════════════════

def run_group_stats(df_dds_info: pd.DataFrame, output_dir: Path) -> None:
    meta = {"subject_id", "group", "BDI", "Original_ID"}
    key_feats = [c for c in df_dds_info.columns
                 if c not in meta and any(x in c for x in
                     ["f1", "f2", "A_ratio", "alpha_mean", "ais_", "te_", "pid_"])]
    rows = []
    for feat in key_feats:
        ctl = df_dds_info.loc[df_dds_info.group == 0, feat].dropna().values
        mdd = df_dds_info.loc[df_dds_info.group == 1, feat].dropna().values
        if len(ctl) < 3 or len(mdd) < 3:
            continue
        pooled = np.sqrt((ctl.std() ** 2 + mdd.std() ** 2) / 2 + 1e-30)
        d = (mdd.mean() - ctl.mean()) / pooled
        _, p = ttest_ind(ctl, mdd, equal_var=False)
        rows.append({"feature": feat, "CTL_mean": round(ctl.mean(), 4),
                     "MDD_mean": round(mdd.mean(), 4),
                     "cohens_d": round(d, 3), "p_value": round(p, 4)})
    if not rows:
        return
    df_stats = pd.DataFrame(rows).sort_values("cohens_d", key=abs, ascending=False)
    df_stats.to_csv(str(output_dir / "ds003478_group_stats.csv"), index=False)
    log.info("Group stats → ds003478_group_stats.csv")
    print("\n── Top DDS/Info features by |Cohen's d| (ds003478 resting) ──")
    print(df_stats.head(15).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — 3-way comparison table
# ══════════════════════════════════════════════════════════════════════════════

def compare_three_way(df_clf: pd.DataFrame, output_dir: Path) -> None:
    reference = [
        {"pipeline": "ds003474 / Task EEG / DDS+Info+Baseline",    "roc_auc": 0.836,
         "paradigm": "Task (PST)", "dataset": "ds003474"},
        {"pipeline": "ds003474 / Task EEG / Microstates+Baseline", "roc_auc": 0.825,
         "paradigm": "Task (PST)", "dataset": "ds003474"},
        {"pipeline": "ds003474 / Task EEG / Baseline+PLV",         "roc_auc": 0.807,
         "paradigm": "Task (PST)", "dataset": "ds003474"},
        {"pipeline": "ds003474 / Task EEG / Baseline spectral",    "roc_auc": 0.675,
         "paradigm": "Task (PST)", "dataset": "ds003474"},
        {"pipeline": "ds005356 / Task MEG+EEG / DDS+Info",         "roc_auc": 0.564,
         "paradigm": "Task (PST)", "dataset": "ds005356"},
        {"pipeline": "ds005356 / Task MEG+EEG / Baseline",         "roc_auc": 0.585,
         "paradigm": "Task (PST)", "dataset": "ds005356"},
    ]
    best_clf = df_clf.loc[df_clf.groupby("feature_set")["roc_auc"].idxmax()][
        ["feature_set", "classifier", "roc_auc"]]
    new_rows = []
    for _, row in best_clf.iterrows():
        new_rows.append({
            "pipeline": f"ds003478 / Resting EEG / {row['feature_set']} / {row['classifier']}",
            "roc_auc":  row["roc_auc"],
            "paradigm": "Resting",
            "dataset":  "ds003478",
        })
    df_comp = pd.DataFrame(reference + new_rows).sort_values("roc_auc", ascending=False)
    df_comp.to_csv(str(output_dir / "comparison_3way.csv"), index=False)

    print("\n" + "=" * 75)
    print("3-WAY COMPARISON: ds003474 (task) vs ds003478 (rest) vs ds005356 (task)")
    print("=" * 75)
    print(df_comp[["pipeline", "roc_auc", "paradigm"]].to_string(index=False))
    print("=" * 75)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full EEG classification pipeline for ds003478 resting state")
    parser.add_argument("--run", type=int, default=1, choices=[1, 2],
                        help="Which rest run to use (1=pre-task, 2=post-task)")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Use cached feature CSVs")
    parser.add_argument("--max_subjects", type=int, default=0,
                        help="Limit subjects (0=all)")
    parser.add_argument("--no_smote", action="store_true")
    parser.add_argument("--n_pca", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"run{args.run:02d}"
    baseline_csv = out_dir / f"ds003478_{run_tag}_baseline_features.csv"
    dds_csv      = out_dir / f"ds003478_{run_tag}_dds_features.csv"
    info_csv     = out_dir / f"ds003478_{run_tag}_info_features.csv"
    merged_csv   = out_dir / f"ds003478_{run_tag}_dds_info_features.csv"
    full_csv     = out_dir / f"ds003478_{run_tag}_dds_info_baseline_features.csv"

    # ── Labels ──────────────────────────────────────────────────────────────
    labels_df = load_group_labels()

    # ── Extract ─────────────────────────────────────────────────────────────
    if not args.skip_extract:
        run_extraction(labels_df, args.run,
                       baseline_csv, dds_csv, info_csv,
                       max_subjects=args.max_subjects)
    else:
        log.info("Skipping extraction — using cached CSVs.")

    # ── Load and merge feature sets ──────────────────────────────────────────
    df_base = pd.read_csv(str(baseline_csv)) if baseline_csv.exists() else pd.DataFrame()
    df_dds  = pd.read_csv(str(dds_csv))      if dds_csv.exists()      else pd.DataFrame()
    df_info = pd.read_csv(str(info_csv))     if info_csv.exists()     else pd.DataFrame()

    meta_cols = ["subject_id", "group", "BDI"]

    df_dds_info = pd.DataFrame()
    if not df_dds.empty and not df_info.empty:
        info_fcols = [c for c in df_info.columns if c not in set(meta_cols)]
        df_dds_info = df_dds.merge(df_info[["subject_id"] + info_fcols],
                                    on="subject_id", how="inner")
        df_dds_info.to_csv(str(merged_csv), index=False)
    elif not df_dds.empty:
        df_dds_info = df_dds.copy()

    df_full = pd.DataFrame()
    if not df_dds_info.empty and not df_base.empty:
        base_fcols = [c for c in df_base.columns if c not in set(meta_cols)]
        df_full = df_dds_info.merge(df_base[["subject_id"] + base_fcols],
                                     on="subject_id", how="inner",
                                     suffixes=("", "_base"))
        df_full.to_csv(str(full_csv), index=False)
        log.info("Full CSV → %s  (%d subj × %d feats)",
                 full_csv.name, len(df_full), df_full.shape[1])

    # ── Group statistics ─────────────────────────────────────────────────────
    if not df_dds_info.empty:
        run_group_stats(df_dds_info, out_dir)

    # ── Classify ─────────────────────────────────────────────────────────────
    feature_sets = {}
    if not df_base.empty:
        feature_sets["Baseline spectral"] = df_base
    if not df_dds.empty:
        feature_sets["DDS only"] = df_dds
    if not df_info.empty:
        feature_sets["Info only"] = df_info
    if not df_dds_info.empty:
        feature_sets["DDS+Info"] = df_dds_info
    if not df_full.empty:
        feature_sets["DDS+Info+Baseline"] = df_full

    log.info("Running classification …")
    df_clf = run_classification(feature_sets, out_dir,
                                 use_smote=not args.no_smote, n_pca=args.n_pca)

    # ── 3-way comparison ─────────────────────────────────────────────────────
    compare_three_way(df_clf, out_dir)
    log.info("Pipeline complete. Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
