"""
ds003478_ec_pipeline.py — Resting EEG, Eyes-Closed, BDI continuo
=================================================================
Dataset : ds003478  (Cavanagh / Allen lab, ~2008-2010)
Subjects: 70 (after RMS quality screen: RMS_RAW < 500 µV)
Signal  : Eyes-Closed segments only (3 blocks × ~66 s = ~3.3 min)
Labels  : BDI as continuous variable (no binarisation)
Features:
  1. Spectral — relative band power, ratios, IAF, FAA  (per channel + ROI)
  2. DDS      — dual damped sinusoid per ROI (spectral initialisation)
  3. Info     — AIS, TE (frontal↔cACC), PID redundancy/synergy
Analysis: Spearman r vs BDI, FDR correction, scatter plots

Pipeline:
  STEP 0  — quality screen (RMS < 500 µV on raw unfiltered data)
  STEP 1  — preprocessing per subject (bandpass + ICA)
  STEP 2  — EC segment extraction and concatenation
  STEP 3  — feature extraction (spectral + DDS + info)
  STEP 4  — correlation analysis (Spearman + FDR)
  STEP 5  — figures

Usage:
  python ds003478_ec_pipeline.py            # full run
  python ds003478_ec_pipeline.py --check    # validate one subject only
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
from scipy import signal as sci_signal
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BIDS_ROOT   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds003478")
OUT_DIR     = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/meg_eeg_comparative")

# Quality screen
RMS_THRESH_UV = 500.0      # µV — subjects above this are excluded (bad amplitude scale)

# Preprocessing
HIGHPASS     = 1.0         # Hz
LOWPASS      = 45.0        # Hz
POWERLINE    = 60.0        # Hz (US recording)
ICA_N_COMP   = 20          # ICA components to compute
ICA_EOG_CH   = "VEOG"      # EOG channel name for automated removal
ICA_EMG_TH   = 3.0         # z-score threshold for EMG component detection

# EC segment extraction
EC_VALUES_2S = {11, 13, 15}   # 2000-ms marker values for EC blocks
EC_TRIM_S    = 2.0            # trim first 2 s of each EC block (settling)

# Spectral features
FREQ_BANDS = {
    "delta":     (1.0,  4.0),
    "theta":     (4.0,  8.0),
    "lo_alpha":  (8.0, 10.0),
    "hi_alpha":  (10.0, 13.0),
    "alpha":     (8.0, 13.0),
    "beta":      (13.0, 30.0),
    "gamma":     (30.0, 45.0),
}
EPOCH_DUR      = 2.0    # s per window for Welch
EPOCH_OVERLAP  = 0.5    # s overlap
AMP_REJECT_UV  = 400.0  # µV peak-to-peak amplitude rejection per window
                         # (post-ICA resting alpha can easily reach 150-250 µV pk-pk;
                         #  400 µV catches residual artifacts without over-rejecting)

# ROIs
ROI_MAP = {
    "frontal":   ["F3", "F4", "F1", "F2", "FZ", "AF3", "AF4"],
    "cACC":      ["FC2", "FCZ", "F2"],
    "LH":        ["F3", "F5", "FC3", "FC5"],
    "RH":        ["F4", "F6", "FC4", "FC6"],
    "occipital": ["O1", "O2", "OZ", "PO3", "POZ", "PO4"],
}

# Individual Alpha Frequency
IAF_RANGE = (7.0, 13.0)   # Hz search range for alpha peak

# Frontal Alpha Asymmetry
FAA_PAIRS = [("F4","F3"), ("FC4","FC3"), ("F6","F5"), ("AF4","AF3")]

# DDS fitting
DDS_WIN_MS    = 400
DDS_SFREQ     = 250.0
DDS_WIN_N     = int(DDS_WIN_MS / 1000.0 * DDS_SFREQ)   # 100 samples
DDS_MIN_FREQ  = 0.5
DDS_MAX_FREQ  = 45.0
DDS_MAX_ALPHA = 500.0
DDS_MAX_NFEV  = 3000
DDS_MAX_WINS  = 120     # max windows per ROI

# Info-theory
AIS_K    = 1
AIS_BINS = 8
INFO_LAG = 4
INFO_BINS = 4
TE_PAIRS  = [("frontal","cACC"), ("cACC","frontal"),
             ("LH","frontal"),   ("RH","frontal")]
PID_SRC   = ["LH", "RH"]
PID_TGT   = "frontal"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 0 — QUALITY SCREEN
# ═══════════════════════════════════════════════════════════════════════════

NON_EEG_CH = {"HEOG", "VEOG", "M1", "M2", "CB1", "CB2"}

def quality_screen() -> pd.DataFrame:
    """
    Load participants.tsv, exclude NaN-BDI subjects (sub-038/544),
    compute raw RMS for each subject, flag BAD if RMS > RMS_THRESH_UV.
    Returns DataFrame with columns: subject_id, BDI, rms_uV, quality.
    """
    df = pd.read_csv(BIDS_ROOT / "participants.tsv", sep="\t")
    rows = []
    for _, row in df.iterrows():
        sub  = str(row["participant_id"]).strip()
        bdi  = row["BDI"]
        if pd.isna(bdi):
            log.info("  %s: BDI missing — excluded", sub)
            continue

        fpath = BIDS_ROOT / sub / "eeg" / f"{sub}_task-Rest_run-01_eeg.set"
        if not fpath.exists():
            log.warning("  %s: run-01 file not found — excluded", sub)
            continue

        raw = mne.io.read_raw_eeglab(str(fpath), preload=True, verbose=False)
        eeg_ch = [c for c in raw.ch_names if c.upper() not in NON_EEG_CH]
        raw.pick_channels(eeg_ch)
        rms_uV = float(np.sqrt(np.mean(raw.get_data() ** 2))) * 1e6

        quality = "OK" if rms_uV <= RMS_THRESH_UV else "BAD"
        rows.append({"subject_id": sub, "BDI": float(bdi),
                     "rms_uV": rms_uV, "quality": quality})

    result = pd.DataFrame(rows)
    n_ok  = (result.quality == "OK").sum()
    n_bad = (result.quality == "BAD").sum()
    log.info("Quality screen: OK=%d  BAD=%d  (threshold=%.0f µV)",
             n_ok, n_bad, RMS_THRESH_UV)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(subject_id: str) -> Optional[mne.io.BaseRaw]:
    """
    Load run-01, apply bandpass filter, run ICA to remove EOG components.
    Returns cleaned Raw or None on failure.
    """
    fpath = BIDS_ROOT / subject_id / "eeg" / f"{subject_id}_task-Rest_run-01_eeg.set"
    raw = mne.io.read_raw_eeglab(str(fpath), preload=True, verbose=False)

    # Drop non-EEG channels (keep VEOG for ICA)
    eeg_plus_eog = [c for c in raw.ch_names
                    if c.upper() not in {"M1", "M2", "CB1", "CB2"}]
    raw.pick_channels(eeg_plus_eog)

    # Standardise channel names to upper-case
    raw.rename_channels({c: c.upper() for c in raw.ch_names})

    # Set channel types
    for ch in raw.ch_names:
        if ch in {"HEOG", "VEOG"}:
            raw.set_channel_types({ch: "eog"})

    # Bandpass
    raw.filter(HIGHPASS, LOWPASS, method="fir", verbose=False)
    raw.notch_filter(POWERLINE, verbose=False)

    # Resample to DDS_SFREQ if needed
    if abs(raw.info["sfreq"] - DDS_SFREQ) > 1.0:
        raw.resample(int(DDS_SFREQ), verbose=False)

    # ICA
    eeg_ch = [c for c in raw.ch_names if raw.get_channel_types([c])[0] == "eeg"]
    if len(eeg_ch) < 10:
        log.warning("  %s: only %d EEG channels — skipping", subject_id, len(eeg_ch))
        return None

    ica = mne.preprocessing.ICA(
        n_components=min(ICA_N_COMP, len(eeg_ch) - 1),
        method="fastica",
        random_state=42,
        max_iter=800,
    )
    try:
        ica.fit(raw, picks="eeg", verbose=False)
    except Exception as exc:
        log.warning("  %s: ICA failed (%s) — skipping", subject_id, exc)
        return None

    # Remove EOG components
    eog_inds = []
    if ICA_EOG_CH in raw.ch_names:
        try:
            eog_inds, _ = ica.find_bads_eog(raw, ch_name=ICA_EOG_CH,
                                             threshold=3.0, verbose=False)
        except Exception:
            pass

    if eog_inds:
        ica.exclude = eog_inds
        log.info("  %s: removed %d ICA EOG components", subject_id, len(eog_inds))

    ica.apply(raw, verbose=False)

    # Drop EOG/non-EEG channels after ICA
    raw.pick_types(eeg=True, verbose=False)

    return raw


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — EC SEGMENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_ec_segments(subject_id: str,
                        raw: mne.io.BaseRaw) -> Optional[np.ndarray]:
    """
    Parse events TSV to find Eyes-Closed blocks (values 11, 13, 15).
    Extract and concatenate EC signal across all 3 blocks.
    Returns array (n_channels, n_times) in µV, or None if < 30 s usable.
    """
    ev_path = (BIDS_ROOT / subject_id / "eeg"
               / f"{subject_id}_task-Rest_run-01_events.tsv")
    ev = pd.read_csv(str(ev_path), sep="\t")
    sfreq = raw.info["sfreq"]
    data_V = raw.get_data()           # (n_ch, n_times) in Volts
    data_uV = data_V * 1e6            # convert to µV

    ec_2000 = ev[ev["trial_type"].str.contains("Eyes Closed", na=False) &
                 ev["trial_type"].str.contains("2000", na=False)]

    segments = []
    total_sec = 0.0

    for val in sorted(EC_VALUES_2S):
        block = ec_2000[ec_2000["value"].astype(str) == str(val)]
        if len(block) == 0:
            continue
        t_start = float(block["onset"].min()) + EC_TRIM_S
        t_end   = float(block["onset"].max()) + 2.0   # last marker + 2 s

        i_start = int(t_start * sfreq)
        i_end   = int(t_end   * sfreq)
        i_start = max(0, i_start)
        i_end   = min(data_uV.shape[1], i_end)

        if i_end - i_start < int(sfreq * 5):   # < 5 s → skip block
            log.warning("  %s: EC block val=%s too short (%.1f s)",
                        subject_id, val, (i_end - i_start) / sfreq)
            continue

        seg = data_uV[:, i_start:i_end]
        segments.append(seg)
        total_sec += (i_end - i_start) / sfreq

    if not segments:
        log.warning("  %s: no valid EC segments", subject_id)
        return None

    if total_sec < 30.0:
        log.warning("  %s: only %.1f s of EC data — skipping", subject_id, total_sec)
        return None

    ec_signal = np.concatenate(segments, axis=1)
    log.info("  %s: %d EC blocks → %.1f s (%.0f samples)",
             subject_id, len(segments), total_sec, ec_signal.shape[1])
    return ec_signal


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3a — SPECTRAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _roi_signal(data_uV: np.ndarray, ch_names: list[str],
                roi_chs: list[str]) -> Optional[np.ndarray]:
    """Mean signal across ROI channels present in the data."""
    ch_up = [c.upper() for c in ch_names]
    idx = [ch_up.index(r.upper()) for r in roi_chs if r.upper() in ch_up]
    if not idx:
        return None
    return data_uV[idx].mean(axis=0)


def extract_spectral_features(ec_signal: np.ndarray,
                               ch_names: list[str]) -> dict:
    """
    Compute per-channel and per-ROI spectral features from EC signal.

    Per channel (up to 64 EEG):
      - relative band power × 7 bands
      - ratios: alpha/beta, theta/alpha, theta/beta, delta/alpha
      - → 11 features × n_channels

    ROI features:
      - mean relative alpha, theta per ROI
      - IAF (individual alpha frequency) from occipital
      - FAA (frontal alpha asymmetry) — 4 electrode pairs

    All features based on RELATIVE power → robust to amplitude scale.
    """
    sfreq   = DDS_SFREQ
    n_ch, n_times = ec_signal.shape
    ch_up   = [c.upper() for c in ch_names]

    win_len = int(EPOCH_DUR * sfreq)
    hop_len = int((EPOCH_DUR - EPOCH_OVERLAP) * sfreq)
    nperseg = min(win_len, int(sfreq * 2))

    # Accumulate band powers across non-artifact windows
    acc_bp: dict[str, list] = {}   # band → list of arrays (n_ch,)
    n_win_used = 0

    for start in range(0, n_times - win_len + 1, hop_len):
        w = ec_signal[:, start: start + win_len]
        pk2pk = w.max(axis=1) - w.min(axis=1)
        if (pk2pk > AMP_REJECT_UV).any():
            continue
        n_win_used += 1

        f, psd = sci_signal.welch(w, fs=sfreq, nperseg=nperseg, axis=1)

        band_pow = {}
        for band, (flo, fhi) in FREQ_BANDS.items():
            mask = (f >= flo) & (f <= fhi)
            bp = np.trapz(psd[:, mask], f[mask], axis=1)
            band_pow[band] = np.maximum(bp, 0.0)

        total = sum(band_pow[b] for b in
                    ["delta", "theta", "alpha", "beta", "gamma"])

        for band in FREQ_BANDS:
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.where(total > 0, band_pow[band] / total, np.nan)
            acc_bp.setdefault(f"{band}_rel", []).append(rel)

    if n_win_used == 0:
        return {}

    # Average over windows → shape (n_ch,)
    mean_bp = {k: np.nanmean(np.stack(v, axis=0), axis=0)
               for k, v in acc_bp.items()}

    feats = {}

    # 1. Per-channel relative band power and ratios
    for band in FREQ_BANDS:
        key = f"{band}_rel"
        vals = mean_bp.get(key, np.full(n_ch, np.nan))
        for i, ch in enumerate(ch_up):
            if ch in NON_EEG_CH:
                continue
            feats[f"{ch}_{key}"] = float(vals[i])

    for i, ch in enumerate(ch_up):
        if ch in NON_EEG_CH:
            continue
        a = mean_bp.get("alpha_rel", np.full(n_ch, np.nan))[i]
        b = mean_bp.get("beta_rel",  np.full(n_ch, np.nan))[i]
        t = mean_bp.get("theta_rel", np.full(n_ch, np.nan))[i]
        d = mean_bp.get("delta_rel", np.full(n_ch, np.nan))[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"{ch}_alpha_beta_ratio"] = float(a / b)  if b > 0 else np.nan
            feats[f"{ch}_theta_alpha_ratio"] = float(t / a) if a > 0 else np.nan
            feats[f"{ch}_theta_beta_ratio"]  = float(t / b) if b > 0 else np.nan
            feats[f"{ch}_delta_alpha_ratio"] = float(d / a) if a > 0 else np.nan

    # 2. ROI mean relative alpha and theta
    for roi, roi_chs in ROI_MAP.items():
        roi_idx = [ch_up.index(r.upper()) for r in roi_chs
                   if r.upper() in ch_up]
        if not roi_idx:
            continue
        for band in ["delta", "theta", "lo_alpha", "hi_alpha", "alpha", "beta"]:
            key = f"{band}_rel"
            if key in mean_bp:
                feats[f"roi_{roi}_{key}"] = float(
                    np.nanmean(mean_bp[key][roi_idx]))
        # ROI ratios
        a_m = feats.get(f"roi_{roi}_alpha_rel", np.nan)
        t_m = feats.get(f"roi_{roi}_theta_rel", np.nan)
        b_m = feats.get(f"roi_{roi}_beta_rel",  np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"roi_{roi}_alpha_beta_ratio"]  = (a_m / b_m if b_m > 0
                                                      else np.nan)
            feats[f"roi_{roi}_theta_alpha_ratio"] = (t_m / a_m if a_m > 0
                                                      else np.nan)
            feats[f"roi_{roi}_theta_beta_ratio"]  = (t_m / b_m if b_m > 0
                                                      else np.nan)

    # 3. Individual Alpha Frequency (IAF) from occipital ROI
    occ_idx = [ch_up.index(c.upper()) for c in ROI_MAP["occipital"]
               if c.upper() in ch_up]
    if occ_idx:
        # Compute PSD over all EC data for better frequency resolution
        occ_sig = ec_signal[occ_idx].mean(axis=0)
        f_iaf, psd_iaf = sci_signal.welch(
            occ_sig, fs=sfreq, nperseg=int(sfreq * 4))
        iaf_mask = (f_iaf >= IAF_RANGE[0]) & (f_iaf <= IAF_RANGE[1])
        if iaf_mask.any():
            feats["iaf_occipital"] = float(
                f_iaf[iaf_mask][np.argmax(psd_iaf[iaf_mask])])

    # 4. Frontal Alpha Asymmetry (FAA = log(right_alpha) - log(left_alpha))
    for rch, lch in FAA_PAIRS:
        if rch.upper() in ch_up and lch.upper() in ch_up:
            ri = ch_up.index(rch.upper())
            li = ch_up.index(lch.upper())
            a_r = mean_bp.get("alpha_rel", np.full(n_ch, np.nan))[ri]
            a_l = mean_bp.get("alpha_rel", np.full(n_ch, np.nan))[li]
            if a_r > 0 and a_l > 0:
                feats[f"faa_{rch}_{lch}"] = float(np.log(a_r) - np.log(a_l))

    feats["n_windows_used"] = n_win_used
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3b — DDS FEATURES (spectral initialisation — correct version)
# ═══════════════════════════════════════════════════════════════════════════

def _dds_signal(t, A1, a1, f1, p1, A2, a2, f2, p2):
    return (A1 * np.exp(-a1 * t) * np.cos(2 * np.pi * f1 * t + p1) +
            A2 * np.exp(-a2 * t) * np.cos(2 * np.pi * f2 * t + p2))


def _dds_initial_params(window: np.ndarray, sfreq: float) -> list:
    """
    Spectral initialisation: use two dominant Welch PSD peaks as
    initial frequency guesses. Avoids the fixed-initialisation bug
    (f1=10 Hz, f2=25 Hz) that caused MODMA/TDBRAIN artifacts.
    """
    n = len(window)
    nperseg = min(n, 64)
    f_ax, pxx = sci_signal.welch(window, fs=sfreq, nperseg=nperseg)
    mask = (f_ax >= DDS_MIN_FREQ) & (f_ax <= DDS_MAX_FREQ)
    f_v, p_v = f_ax[mask], pxx[mask]

    if len(p_v) < 2:
        f1i, f2i = 10.0, 20.0
    else:
        order = np.argsort(p_v)[::-1]
        f1i = float(f_v[order[0]])
        f2i = next(
            (float(f_v[i]) for i in order[1:]
             if abs(f_v[i] - f1i) >= 2.0),
            f1i * 2.0,
        )
        f2i = float(np.clip(f2i, DDS_MIN_FREQ, DDS_MAX_FREQ))

    A = float(np.std(window)) + 1e-12
    return [A * 0.7, 5.0, f1i, 0.0, A * 0.3, 10.0, f2i, 0.0]


def _fit_dds_window(window: np.ndarray, sfreq: float) -> Optional[dict]:
    """
    Fit one DDS window. Returns parameter dict or None on failure.
    Uses spectral initialisation to avoid fixed-parameter artifacts.
    """
    n = len(window)
    t = np.arange(n) / sfreq
    y = window - window.mean()
    if np.std(y) < 1e-30:
        return None

    p0 = _dds_initial_params(y, sfreq)
    lo = [0, 0, DDS_MIN_FREQ, -np.pi, 0, 0, DDS_MIN_FREQ, -np.pi]
    hi = [np.inf, DDS_MAX_ALPHA, DDS_MAX_FREQ, np.pi,
          np.inf, DDS_MAX_ALPHA, DDS_MAX_FREQ, np.pi]
    # Clip p0 inside bounds
    p0 = [float(np.clip(v, l + 1e-8, max(l + 1e-8, h - 1e-8)))
          for v, l, h in zip(p0, lo, hi)]
    try:
        popt, _ = curve_fit(
            _dds_signal, t, y, p0=p0,
            bounds=(lo, hi), method="trf",
            max_nfev=DDS_MAX_NFEV,
        )
        fitted = _dds_signal(t, *popt)
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-30
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        param_names = ("A1", "alpha1", "f1", "phi1", "A2", "alpha2", "f2", "phi2")
        return {p: float(v) for p, v in zip(param_names, popt)} | {"r2": r2}
    except Exception:
        return None


def extract_dds_features(ec_signal: np.ndarray,
                          ch_names: list[str]) -> dict:
    """
    Fit DDS on DDS_MAX_WINS random 400-ms windows per ROI.
    Aggregate: mean and std of each parameter across windows.

    Returns features prefixed with dds_{roi}_{param}_{mean|std}.
    Also returns dds_{roi}_f_diff (mean f2-f1), dds_{roi}_A_ratio (A1/(A1+A2)).
    """
    rng = np.random.default_rng(42)
    ch_up  = [c.upper() for c in ch_names]
    sfreq  = DDS_SFREQ
    feats  = {}
    n_ok_total = 0

    # Resample EC signal to DDS_SFREQ if needed (already done in preprocess)
    # ec_signal is (n_ch, n_times) in µV, sfreq = DDS_SFREQ

    param_names = ("A1", "alpha1", "f1", "phi1", "A2", "alpha2", "f2", "phi2", "r2")

    for roi_name, roi_chs in ROI_MAP.items():
        roi_sig = _roi_signal(ec_signal, ch_names, roi_chs)
        if roi_sig is None:
            continue

        n_times = len(roi_sig)
        max_start = n_times - DDS_WIN_N
        if max_start < 1:
            continue

        starts = rng.integers(0, max_start, size=DDS_MAX_WINS * 3)
        acc: dict[str, list] = {p: [] for p in param_names}
        n_ok = 0

        for s in starts:
            if n_ok >= DDS_MAX_WINS:
                break
            seg = roi_sig[int(s): int(s) + DDS_WIN_N]
            result = _fit_dds_window(seg, sfreq)
            if result is None:
                continue
            for p in param_names:
                acc[p].append(result[p])
            n_ok += 1

        if n_ok == 0:
            continue

        n_ok_total += n_ok

        for p in param_names:
            vals = np.array(acc[p])
            feats[f"dds_{roi_name}_{p}_mean"] = float(np.nanmean(vals))
            feats[f"dds_{roi_name}_{p}_std"]  = float(np.nanstd(vals))

        # Derived: frequency difference and amplitude ratio
        f1m = np.nanmean(acc["f1"])
        f2m = np.nanmean(acc["f2"])
        A1m = np.nanmean(acc["A1"])
        A2m = np.nanmean(acc["A2"])
        feats[f"dds_{roi_name}_f_diff"]  = float(f2m - f1m)
        feats[f"dds_{roi_name}_A_ratio"] = float(A1m / (A1m + A2m + 1e-30))

    log.info("  DDS: %d windows OK across %d ROIs", n_ok_total, len(ROI_MAP))
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3c — INFO-THEORY FEATURES (AIS, TE, PID)
# ═══════════════════════════════════════════════════════════════════════════

def _quantise(signal: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.percentile(signal, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-10
    edges[-1] += 1e-10
    return np.digitize(signal, edges) - 1


def _ais(x: np.ndarray, k: int = AIS_K,
         n_bins: int = AIS_BINS) -> float:
    """Active Information Storage: I(x_t+1 ; x_t^{(k)})."""
    xq = _quantise(x, n_bins)
    n  = len(xq)
    src = xq[k:]
    cnd = xq[: n - k]
    px  = np.bincount(src, minlength=n_bins) / (n - k)
    pxy = np.zeros((n_bins, n_bins))
    for a, b in zip(src, cnd):
        pxy[a, b] += 1
    pxy /= pxy.sum() + 1e-30
    py = pxy.sum(axis=0) + 1e-30
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i, j] > 0 and py[j] > 0 and px[i] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    return float(mi)


def _te(src: np.ndarray, tgt: np.ndarray,
        lag: int = INFO_LAG, n_bins: int = INFO_BINS) -> float:
    """Transfer entropy TE(src → tgt)."""
    sq = _quantise(src, n_bins)
    tq = _quantise(tgt, n_bins)
    n  = len(sq) - lag
    if n < 20:
        return np.nan
    t_f   = tq[lag:]
    t_p   = tq[:n]
    s_p   = sq[:n]
    bins  = n_bins
    h_tf_tp = 0.0
    h_tp    = 0.0
    h_tf_tp_sp = 0.0
    h_tp_sp    = 0.0
    joint_tp_sp = np.zeros((bins, bins))
    joint_tf_tp = np.zeros((bins, bins))
    joint_all   = np.zeros((bins, bins, bins))
    for a, b, c in zip(t_f, t_p, s_p):
        joint_tp_sp[b, c] += 1
        joint_tf_tp[a, b] += 1
        joint_all[a, b, c] += 1
    total = n
    joint_tp_sp  /= total
    joint_tf_tp  /= total
    joint_all    /= total
    p_tp = joint_tp_sp.sum(axis=1) + 1e-30
    for b in range(bins):
        if p_tp[b] > 0:
            h_tp -= p_tp[b] * np.log2(p_tp[b])
    for a, b in np.ndindex(bins, bins):
        p = joint_tf_tp[a, b]
        if p > 0:
            h_tf_tp -= p * np.log2(p)
    for b, c in np.ndindex(bins, bins):
        p = joint_tp_sp[b, c]
        if p > 0:
            h_tp_sp -= p * np.log2(p)
    for a, b, c in np.ndindex(bins, bins, bins):
        p = joint_all[a, b, c]
        if p > 0:
            h_tf_tp_sp -= p * np.log2(p)
    te = h_tf_tp + h_tp_sp - h_tp - h_tf_tp_sp
    return float(max(0.0, te))


def _pid(src1: np.ndarray, src2: np.ndarray,
         tgt: np.ndarray, n_bins: int = INFO_BINS) -> dict:
    """
    Partial Information Decomposition (approximate).
    Redundancy ≈ min(I(S1;T), I(S2;T)).
    Synergy ≈ I(S1,S2;T) - max(I(S1;T), I(S2;T)).
    """
    s1q = _quantise(src1, n_bins)
    s2q = _quantise(src2, n_bins)
    tq  = _quantise(tgt,  n_bins)
    n   = len(tq)

    def _mi(xq, yq):
        pxy = np.zeros((n_bins, n_bins))
        for a, b in zip(xq, yq):
            pxy[a, b] += 1
        pxy /= n
        px = pxy.sum(axis=1) + 1e-30
        py = pxy.sum(axis=0) + 1e-30
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if pxy[i, j] > 0:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
        return float(max(0.0, mi))

    i_s1t  = _mi(s1q, tq)
    i_s2t  = _mi(s2q, tq)
    # Joint MI: I(S1,S2;T)
    s12q = s1q * n_bins + s2q
    i_12t = _mi(s12q, tq)

    redundancy = min(i_s1t, i_s2t)
    synergy    = max(0.0, i_12t - max(i_s1t, i_s2t))
    return {"redundancy": redundancy, "synergy": synergy,
            "i_s1t": i_s1t, "i_s2t": i_s2t, "i_joint": i_12t}


def extract_info_features(ec_signal: np.ndarray,
                           ch_names: list[str]) -> dict:
    """
    Extract AIS per ROI, TE for frontal↔cACC and LH/RH→frontal,
    PID (redundancy/synergy) for LH+RH → frontal.
    Uses downsampled ROI-mean signal for computational efficiency.
    """
    sfreq = DDS_SFREQ
    # Downsample to 100 Hz for info theory (reduce computation)
    ds_factor = max(1, int(sfreq // 100))
    feats = {}

    roi_sigs: dict[str, Optional[np.ndarray]] = {}
    for roi, roi_chs in ROI_MAP.items():
        sig = _roi_signal(ec_signal, ch_names, roi_chs)
        if sig is not None:
            roi_sigs[roi] = sig[::ds_factor]   # downsample
        else:
            roi_sigs[roi] = None

    # AIS per ROI
    for roi, sig in roi_sigs.items():
        if sig is None or len(sig) < 50:
            continue
        feats[f"ais_{roi}"] = _ais(sig)

    # TE
    for src_roi, tgt_roi in TE_PAIRS:
        s = roi_sigs.get(src_roi)
        t = roi_sigs.get(tgt_roi)
        if s is None or t is None:
            continue
        feats[f"te_{src_roi}_to_{tgt_roi}"] = _te(s, t)

    # PID: LH + RH → frontal
    s1 = roi_sigs.get("LH")
    s2 = roi_sigs.get("RH")
    tg = roi_sigs.get("frontal")
    if s1 is not None and s2 is not None and tg is not None:
        pid_res = _pid(s1, s2, tg)
        feats["pid_redundancy"] = pid_res["redundancy"]
        feats["pid_synergy"]    = pid_res["synergy"]
        feats["pid_i_LH"]       = pid_res["i_s1t"]
        feats["pid_i_RH"]       = pid_res["i_s2t"]
        feats["pid_i_joint"]    = pid_res["i_joint"]

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def process_subject(subject_id: str) -> Optional[dict]:
    """Full pipeline for one subject. Returns feature dict or None."""
    log.info("Processing %s …", subject_id)

    raw = preprocess(subject_id)
    if raw is None:
        return None

    ec_signal = extract_ec_segments(subject_id, raw)
    if ec_signal is None:
        return None

    ch_names = list(raw.ch_names)
    feats: dict = {"subject_id": subject_id}

    spec = extract_spectral_features(ec_signal, ch_names)
    if not spec:
        log.warning("  %s: spectral features empty", subject_id)
        return None
    feats.update(spec)

    dds = extract_dds_features(ec_signal, ch_names)
    feats.update(dds)

    info = extract_info_features(ec_signal, ch_names)
    feats.update(info)

    log.info("  %s: %d features", subject_id, len(feats) - 1)
    return feats


def run_pipeline(subjects: list[str], bdi_map: dict[str, float],
                 out_csv: Path) -> pd.DataFrame:
    """Run full pipeline on all subjects, save features CSV."""
    all_feats = []
    n_ok = 0

    for sub in subjects:
        feats = process_subject(sub)
        if feats is None:
            log.warning("  %s: EXCLUDED (pipeline failure)", sub)
            continue
        feats["BDI"] = bdi_map[sub]
        all_feats.append(feats)
        n_ok += 1

    df = pd.DataFrame(all_feats)
    df.to_csv(out_csv, index=False)
    log.info("Features saved → %s  (N=%d)", out_csv, n_ok)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def correlation_analysis(df: pd.DataFrame,
                         out_csv: Path) -> pd.DataFrame:
    """
    Spearman r between each feature and BDI.
    FDR correction (Benjamini-Hochberg) across all features.
    Returns sorted DataFrame of results.
    """
    bdi = df["BDI"].values
    skip = {"subject_id", "BDI", "n_windows_used"}
    feat_cols = [c for c in df.columns if c not in skip
                 and df[c].notna().sum() >= 20]

    rows = []
    for col in feat_cols:
        vals = df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 20:
            continue
        r, p = spearmanr(vals[mask], bdi[mask])
        rows.append({"feature": col, "r": r, "p": p, "n": int(mask.sum())})

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    _, q, _, _ = multipletests(result["p"].values, method="fdr_bh")
    result["q"] = q
    result = result.sort_values("p").reset_index(drop=True)

    result.to_csv(out_csv, index=False)
    n_sig = (result["q"] < 0.05).sum()
    n_nom = (result["p"] < 0.05).sum()
    log.info("Correlations: %d features, %d nominal (p<0.05), %d FDR-significant",
             len(result), n_nom, n_sig)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ds003478 EC pipeline — BDI correlations")
    parser.add_argument("--check", action="store_true",
                        help="Validate pipeline on first OK subject only")
    parser.add_argument("--subjects", nargs="+",
                        help="Process specific subjects only")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("ds003478 — Eyes-Closed Resting EEG Pipeline")
    log.info("=" * 65)

    # Quality screen
    qc = quality_screen()
    ok_df = qc[qc["quality"] == "OK"].copy()
    bdi_map = dict(zip(ok_df["subject_id"], ok_df["BDI"]))

    if args.subjects:
        subjects = [s for s in args.subjects if s in bdi_map]
        log.info("Running on %d specified subjects", len(subjects))
    elif args.check:
        subjects = [ok_df["subject_id"].iloc[0]]
        log.info("CHECK mode: single subject %s", subjects[0])
    else:
        subjects = ok_df["subject_id"].tolist()
        log.info("Full run: %d subjects", len(subjects))

    # Run pipeline
    feat_csv = OUT_DIR / "ds003478_ec_features.csv"
    df = run_pipeline(subjects, bdi_map, feat_csv)

    if df.empty:
        log.error("No subjects processed successfully.")
        return

    # Correlation analysis
    corr_csv = OUT_DIR / "ds003478_ec_bdi_correlations.csv"
    corr = correlation_analysis(df, corr_csv)

    log.info("\nTop 20 features correlated with BDI (Spearman r):")
    cols = ["feature", "r", "p", "q", "n"]
    log.info("\n%s", corr[cols].head(20).to_string(index=False))

    log.info("\nDone.")


if __name__ == "__main__":
    main()
