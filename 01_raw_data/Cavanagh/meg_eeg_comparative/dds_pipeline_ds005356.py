"""
dds_pipeline_ds005356.py
========================
Dual Damped Sine (DDS) + Information Theory (AIS / TE / PID) pipeline
applied to ds005356 concurrent EEG channels.

Mirrors the DDS+Info+Baseline pipeline from ds003474 so results are
directly comparable. Key adaptation: renames EEGxxx → 10-20 names via
the mgh70 montage (exact 0 mm match to standard_1020).

Pipeline stages
---------------
1. Load preprocessed EEG (uses already-cleaned FIF via the same
   preprocessing path as meg_pipeline_ds005356.py)
2. Rename EEGxxx → 10-20 names (EEG013→Fz, EEG011→F3, etc.)
3. DDS fit: two damped sinusoids on each ROI-averaged window
4. Info theory: AIS / TE / PID on DDS residuals
5. Merge with baseline features CSV (already computed)
6. Classification: DDS only | Info only | DDS+Info | DDS+Info+Baseline

Usage
-----
    python dds_pipeline_ds005356.py                    # full pipeline
    python dds_pipeline_ds005356.py --skip_extract     # use cached CSVs
    python dds_pipeline_ds005356.py --n_jobs 4
    python dds_pipeline_ds005356.py --max_subjects 10  # debug
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
from scipy.optimize import curve_fit
from scipy.signal import welch, resample_poly
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

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
BIDS_ROOT   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds005356")
EXCEL_FILE  = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds005356/Code/MEG MDD IDs and Quex.xlsx")
OUTPUT_DIR  = SCRIPT_DIR
BASELINE_CSV = OUTPUT_DIR / "meg_features.csv"   # from meg_pipeline_ds005356.py

# ──────────────────────────────────────────────────────────────────────────────
# EEGxxx → 10-20 name mapping  (from mgh70 montage, 0 mm distance to std_1020)
# ──────────────────────────────────────────────────────────────────────────────
EEG_TO_1020 = {
    'EEG001': 'Fp1',  'EEG002': 'Fpz',  'EEG003': 'Fp2',
    'EEG004': 'AF7',  'EEG005': 'AF3',  'EEG006': 'AFz',
    'EEG007': 'AF4',  'EEG008': 'AF8',
    'EEG009': 'F7',   'EEG010': 'F5',   'EEG011': 'F3',
    'EEG012': 'F1',   'EEG013': 'Fz',   'EEG014': 'F2',
    'EEG015': 'F4',   'EEG016': 'F6',   'EEG017': 'F8',
    'EEG018': 'FT9',  'EEG019': 'FT7',  'EEG020': 'FC5',
    'EEG021': 'FC3',  'EEG022': 'FC1',  'EEG023': 'FCz',
    'EEG024': 'FC2',  'EEG025': 'FC4',  'EEG026': 'FC6',
    'EEG027': 'FT8',  'EEG028': 'FT10',
    'EEG029': 'T9',   'EEG030': 'T7',   'EEG031': 'C5',
    'EEG032': 'C3',   'EEG033': 'C1',   'EEG034': 'Cz',
    'EEG035': 'C2',   'EEG036': 'C4',   'EEG037': 'C6',
    'EEG038': 'T8',   'EEG039': 'T10',
    'EEG040': 'TP9',  'EEG041': 'TP7',  'EEG042': 'CP5',
    'EEG043': 'CP3',  'EEG044': 'CP1',  'EEG045': 'CPz',
    'EEG046': 'CP2',  'EEG047': 'CP4',  'EEG048': 'CP6',
    'EEG049': 'TP8',  'EEG050': 'TP10',
    'EEG051': 'P9',   'EEG052': 'P7',   'EEG053': 'P5',
    'EEG054': 'P3',   'EEG055': 'P1',   'EEG056': 'Pz',
    'EEG057': 'P2',   'EEG058': 'P4',   'EEG059': 'P6',
    'EEG060': 'P8',
    'EEG065': 'P10',  'EEG066': 'PO7',  'EEG067': 'PO3',
    'EEG068': 'POz',  'EEG069': 'PO4',  'EEG070': 'PO8',
    'EEG071': 'O1',   'EEG072': 'Oz',   'EEG073': 'O2',
    'EEG074': 'Iz',
}
# Reverse mapping: 10-20 → EEGxxx (for reference)
_1020_TO_EEG = {v: k for k, v in EEG_TO_1020.items()}

# ──────────────────────────────────────────────────────────────────────────────
# DDS parameters  (same as ds003474 config.py)
# ──────────────────────────────────────────────────────────────────────────────
DDS_WIN_MS     = 400          # ms
DDS_SFREQ      = 250.0        # Hz (resample to this)
DDS_WIN_N      = int(DDS_WIN_MS / 1000.0 * DDS_SFREQ)   # 100 samples
DDS_MIN_FREQ   = 0.5
DDS_MAX_FREQ   = 45.0
DDS_MAX_ALPHA  = 500.0
DDS_MAX_NFEV   = 3000
DDS_MAX_WINS   = 100          # max windows per subject (speed)

# ROI definitions — same as ds003474 (case-insensitive match)
ROI_MAP = {
    "frontal": ["F3", "F4", "AF3", "AF4", "Fp1", "Fp2", "FC3", "FC4"],
    "cACC":    ["FC2", "AFz", "F2"],
    "LH":      ["F3", "F5", "FC3", "FC5"],
    "RH":      ["F4", "F6", "FC4", "FC6"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Info-theory parameters  (same as ds003474)
# ──────────────────────────────────────────────────────────────────────────────
AIS_K       = 1
AIS_BINS    = 8
INFO_LAG    = 4              # samples at DDS_SFREQ → 16 ms
INFO_BINS   = 4
TE_PAIRS    = [("frontal", "cACC"), ("cACC", "frontal"),
               ("LH", "frontal"),   ("RH", "frontal")]
PID_SRC     = ["LH", "RH"]
PID_TGT     = "frontal"

# ──────────────────────────────────────────────────────────────────────────────
# Classification parameters
# ──────────────────────────────────────────────────────────────────────────────
CV_FOLDS     = 5
RANDOM_STATE = 42
N_JOBS       = -1

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing parameters  (same as meg_pipeline_ds005356.py)
# ──────────────────────────────────────────────────────────────────────────────
SFREQ_TARGET    = 500.0
HIGHPASS        = 1.0
LOWPASS         = 45.0
NOTCH_FREQ      = 60.0
N_ICA_COMPONENTS = 20
AMPLITUDE_THRESH = 150e-6


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Group labels
# ══════════════════════════════════════════════════════════════════════════════

def load_group_labels(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(str(excel_path))
    df.columns = df.columns.str.strip()
    ursi_col = next(c for c in df.columns if "URSI" in c.upper())
    grp_col  = next((c for c in df.columns if "GROUP" in c.upper()
                     or "DIAGNOSIS" in c.upper() or "MDD" in c.upper()), None)
    bdi_col  = next((c for c in df.columns if "BDI" in c.upper()), None)

    rows = []
    for _, row in df.iterrows():
        try:
            ursi = int(float(str(row[ursi_col]).strip()))
        except (ValueError, TypeError):
            continue
        sub_id = f"sub-M87{100000 + ursi}"
        grp = 0
        if grp_col:
            val = str(row[grp_col]).strip().upper()
            if any(x in val for x in ("MDD", "PATIENT", "1", "DEP")):
                grp = 1
        bdi = float(row[bdi_col]) if bdi_col and pd.notna(row[bdi_col]) else np.nan
        rows.append({"subject_id": sub_id, "group": grp, "BDI": bdi})

    return pd.DataFrame(rows).drop_duplicates("subject_id")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Find FIF files  (reused from meg_pipeline_ds005356.py)
# ══════════════════════════════════════════════════════════════════════════════

def find_fif_files(bids_root: Path) -> dict:
    fif_map = {}
    for sub_dir in sorted(bids_root.glob("sub-M*")):
        sub_id = sub_dir.name
        pattern = f"{sub_id}_ses-01_task-pst_run-1_split-01_meg.fif"
        candidates = list(sub_dir.glob(f"ses-01/meg/{pattern}"))
        if candidates:
            fif_map[sub_id] = candidates[0]
        else:
            all_fifs = sorted(sub_dir.glob("ses-01/meg/*_meg.fif"))
            split01 = [f for f in all_fifs if "split-01" in f.name]
            if split01:
                fif_map[sub_id] = split01[0]
            elif all_fifs:
                fif_map[sub_id] = all_fifs[0]
    log.info("Found %d subjects with FIF files.", len(fif_map))
    return fif_map


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Preprocessing  (same logic as meg_pipeline_ds005356.py)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_and_rename(fif_path: Path) -> Optional[mne.io.BaseRaw]:
    """
    Load FIF → clean EEG → rename EEGxxx to 10-20 names.
    Returns Raw with 10-20 channel names, or None on failure.
    """
    try:
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    except Exception as exc:
        log.warning("  Load error: %s", exc)
        return None

    # Keep only EEG channels
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    if len(eeg_picks) < 10:
        log.warning("  Too few EEG channels (%d).", len(eeg_picks))
        return None
    raw.pick(eeg_picks)

    # Resample to 500 Hz
    if raw.info["sfreq"] != SFREQ_TARGET:
        raw.resample(SFREQ_TARGET, verbose=False)

    # Bad channel detection (relative thresholds, before average reference)
    data_pre = raw.get_data()
    stds_pre = data_pre.std(axis=1)
    med_std  = float(np.median(stds_pre))
    flat_bads  = [raw.ch_names[i] for i, s in enumerate(stds_pre)
                  if s < 0.10 * med_std]
    noisy_bads = [raw.ch_names[i] for i, s in enumerate(stds_pre)
                  if s > 5.0 * med_std]
    bads = list(set(flat_bads + noisy_bads))
    if 0 < len(bads) < int(0.25 * len(raw.ch_names)):
        raw.info["bads"] = bads
        try:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        except Exception:
            raw.info["bads"] = []

    # Average reference & filter
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    raw.filter(HIGHPASS, LOWPASS, verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    # ICA
    raw.info["bads"] = []
    all_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    n_comps = min(N_ICA_COMPONENTS, len(all_picks) - 1)
    raw_1hz = raw.copy().filter(1.0, None, verbose=False)
    raw_ica_fit = raw_1hz.copy().crop(
        tmax=min(300.0, raw_1hz.times[-1])) if raw_1hz.n_times > int(300 * raw.info["sfreq"]) else raw_1hz
    try:
        from mne.preprocessing import ICA
        ica = ICA(n_components=n_comps, method="fastica",
                  fit_params={"tol": 1e-4, "max_iter": 300},
                  random_state=RANDOM_STATE, verbose=False)
        ica.fit(raw_ica_fit, picks=all_picks, verbose=False)
        eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
        mus_idx, _ = ica.find_bads_muscle(raw, verbose=False)
        excl = list(set(eog_idx[:2] + mus_idx[:2]))
        ica.exclude = excl
        ica.apply(raw, verbose=False)
    except Exception:
        pass

    # Rename EEGxxx → 10-20
    rename_dict = {ch: EEG_TO_1020[ch] for ch in raw.ch_names if ch in EEG_TO_1020}
    raw.rename_channels(rename_dict)

    return raw


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DDS fitting
# ══════════════════════════════════════════════════════════════════════════════

def _dds_signal(t, A1, alpha1, f1, phi1, A2, alpha2, f2, phi2):
    return (A1 * np.exp(-alpha1 * t) * np.cos(2 * np.pi * f1 * t + phi1) +
            A2 * np.exp(-alpha2 * t) * np.cos(2 * np.pi * f2 * t + phi2))


def _initial_params(window: np.ndarray, sfreq: float) -> list:
    n = len(window)
    nperseg = min(n, 64)
    f_ax, pxx = welch(window, fs=sfreq, nperseg=nperseg)
    mask = (f_ax >= DDS_MIN_FREQ) & (f_ax <= DDS_MAX_FREQ)
    f_v, p_v = f_ax[mask], pxx[mask]
    if len(p_v) < 2:
        f1_init, f2_init = 10.0, 20.0
    else:
        order = np.argsort(p_v)[::-1]
        f1_init = float(f_v[order[0]])
        f2_init = next((float(f_v[i]) for i in order[1:]
                        if abs(f_v[i] - f1_init) >= 2.0), f1_init * 2.0)
        f2_init = float(np.clip(f2_init, DDS_MIN_FREQ, DDS_MAX_FREQ))
    A_rms = float(np.std(window)) + 1e-12
    return [A_rms * 0.70, 5.0, f1_init, 0.0, A_rms * 0.30, 10.0, f2_init, 0.0]


def _fit_window(window: np.ndarray, sfreq: float):
    """Fit DDS to a single window. Returns dict or None on failure."""
    n = len(window)
    t = np.arange(n) / sfreq
    p0 = _initial_params(window, sfreq)
    lower = [0, 0, DDS_MIN_FREQ, -np.pi, 0, 0, DDS_MIN_FREQ, -np.pi]
    upper = [np.inf, DDS_MAX_ALPHA, DDS_MAX_FREQ, np.pi,
             np.inf, DDS_MAX_ALPHA, DDS_MAX_FREQ, np.pi]
    p0 = [np.clip(v, lo + 1e-8, max(lo + 1e-8, hi - 1e-8))
          for v, lo, hi in zip(p0, lower, upper)]
    try:
        popt, _ = curve_fit(_dds_signal, t, window, p0=p0,
                             bounds=(lower, upper), method="trf",
                             max_nfev=DDS_MAX_NFEV,
                             ftol=1e-8, xtol=1e-8, gtol=1e-8)
    except Exception:
        return None
    fitted   = _dds_signal(t, *popt)
    residual = window - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((window - window.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else 0.0
    return dict(A1=popt[0], alpha1=popt[1], f1=popt[2], phi1=popt[3],
                A2=popt[4], alpha2=popt[5], f2=popt[6], phi2=popt[7],
                r2=r2, residual_rms=float(np.sqrt(np.mean(residual ** 2))),
                residual=residual)


def _roi_signal(data: np.ndarray, ch_names: list, roi_chs: list) -> Optional[np.ndarray]:
    ch_upper = [c.upper() for c in ch_names]
    idx = [ch_upper.index(r.upper()) for r in roi_chs if r.upper() in ch_upper]
    return data[np.array(idx)].mean(axis=0) if idx else None


def compute_dds_features(raw: mne.io.BaseRaw) -> tuple[dict, dict]:
    """
    Fit DDS model on sliding windows from the continuous recording.
    Returns (features_dict, residuals_dict).
    residuals_dict: {roi_name: [residual_array, ...]}
    """
    # Resample to DDS_SFREQ (250 Hz)
    sfreq = raw.info["sfreq"]
    data  = raw.get_data()   # (n_ch, n_times)
    ch_names = raw.ch_names

    if abs(sfreq - DDS_SFREQ) > 1.0:
        fs_int = int(round(sfreq))
        tg_int = int(round(DDS_SFREQ))
        g = gcd(fs_int, tg_int)
        up, down = tg_int // g, fs_int // g
        data = resample_poly(data, up, down, axis=1)
        sfreq = DDS_SFREQ

    n_ch, n_times = data.shape
    win_n = DDS_WIN_N
    hop   = win_n   # non-overlapping

    # Collect all valid start positions then subsample to DDS_MAX_WINS
    starts = list(range(0, n_times - win_n + 1, hop))
    if len(starts) > DDS_MAX_WINS:
        rng = np.random.default_rng(RANDOM_STATE)
        starts = sorted(rng.choice(starts, DDS_MAX_WINS, replace=False).tolist())

    param_keys = ["A1", "alpha1", "f1", "phi1",
                  "A2", "alpha2", "f2", "phi2",
                  "r2", "residual_rms"]

    roi_acc:  dict[str, dict[str, list]] = {}
    residuals: dict[str, list] = {}

    for roi_name, roi_chs in ROI_MAP.items():
        acc = {k: [] for k in param_keys}
        res_list = []

        roi_sig = _roi_signal(data, ch_names, roi_chs)
        if roi_sig is None:
            roi_acc[roi_name]  = acc
            residuals[roi_name] = res_list
            continue

        for s in starts:
            win = roi_sig[s: s + win_n] - roi_sig[s: s + win_n].mean()
            if np.ptp(win) < 1e-15:
                continue
            fit = _fit_window(win, sfreq)
            if fit is None:
                continue
            for k in param_keys:
                acc[k].append(fit[k])
            res_list.append(fit["residual"])

        roi_acc[roi_name]  = acc
        residuals[roi_name] = res_list

    # Aggregate
    feats = {}
    for roi_name, acc in roi_acc.items():
        n_ok = len(acc["r2"])
        for k in param_keys:
            feats[f"dds_{roi_name}_{k}"] = float(np.nanmean(acc[k])) if n_ok else np.nan
        if n_ok:
            a1_m = float(np.nanmean(acc["A1"]))
            a2_m = float(np.nanmean(acc["A2"]))
            f1_arr = np.array(acc["f1"])
            f2_arr = np.array(acc["f2"])
            feats[f"dds_{roi_name}_A_ratio"]    = a1_m / (a1_m + a2_m + 1e-30)
            feats[f"dds_{roi_name}_f_diff"]      = float(np.nanmean(np.abs(f1_arr - f2_arr)))
            feats[f"dds_{roi_name}_alpha_mean"]  = float(
                np.nanmean(np.array(acc["alpha1"]) + np.array(acc["alpha2"])) / 2.0)
            feats[f"dds_{roi_name}_n_wins_ok"]   = n_ok
        else:
            feats[f"dds_{roi_name}_A_ratio"]   = np.nan
            feats[f"dds_{roi_name}_f_diff"]     = np.nan
            feats[f"dds_{roi_name}_alpha_mean"] = np.nan
            feats[f"dds_{roi_name}_n_wins_ok"]  = 0

    return feats, residuals


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Information theory  (AIS / TE / PID)
# ══════════════════════════════════════════════════════════════════════════════

def _discretise(x: np.ndarray, n_bins: int = INFO_BINS) -> np.ndarray:
    """Uniform-quantile discretisation to n_bins integer symbols."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=int)
    for i, q in enumerate(np.linspace(0, 100, n_bins + 1)[1:-1]):
        out[x > np.percentile(x, q)] += 1
    return out


def _mi_histogram(x: np.ndarray, y: np.ndarray, n_bins: int = AIS_BINS) -> float:
    """Mutual information I(X;Y) via histogram estimator."""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    joint, _, _ = np.histogram2d(x, y, bins=n_bins)
    joint = joint / (joint.sum() + 1e-30)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log2(joint[mask] / (px * py + 1e-30)[mask] + 1e-30)))
    return max(mi, 0.0)


def compute_ais(signal: np.ndarray, k: int = AIS_K) -> float:
    """AIS = I(X_{t-k:t-1} ; X_t) via histogram MI, k=1."""
    n = len(signal)
    if n < k + 2:
        return np.nan
    x_past = signal[:-k]
    x_fut  = signal[k:]
    return _mi_histogram(x_past, x_fut, n_bins=AIS_BINS)


def compute_te(source: np.ndarray, target: np.ndarray,
               lag: int = INFO_LAG) -> float:
    """TE(source → target) = H(Y_{t+lag} | Y_t) − H(Y_{t+lag} | X_t, Y_t)."""
    n = min(len(source), len(target)) - lag
    if n < 10:
        return np.nan
    s  = _discretise(source[:n])
    yt = _discretise(target[:n])
    yf = _discretise(target[lag:lag + n])
    # H(yf | yt)
    joint_yy, _, _ = np.histogram2d(yt, yf, bins=INFO_BINS)
    pyt = joint_yy.sum(axis=1, keepdims=True)
    joint_yy = joint_yy / (joint_yy.sum() + 1e-30)
    pyt_n = pyt / (pyt.sum() + 1e-30)
    h_yf_yt = -float(np.sum(
        joint_yy[joint_yy > 0] * np.log2(
            joint_yy[joint_yy > 0] /
            (np.tile(pyt_n, (1, INFO_BINS))[joint_yy > 0] + 1e-30) + 1e-30)))
    # H(yf | yt, s)
    stacked = np.stack([yt, s], axis=1)
    n_states = INFO_BINS ** 2
    state_id = stacked[:, 0] * INFO_BINS + stacked[:, 1]
    h_yf_yt_s = 0.0
    for sid in range(n_states):
        mask = state_id == sid
        if mask.sum() < 2:
            continue
        p_state = mask.sum() / n
        yf_given = yf[mask]
        counts   = np.bincount(yf_given, minlength=INFO_BINS).astype(float)
        counts  /= counts.sum() + 1e-30
        h_cond   = -float(np.sum(counts[counts > 0] * np.log2(counts[counts > 0] + 1e-30)))
        h_yf_yt_s += p_state * h_cond
    return max(h_yf_yt - h_yf_yt_s, 0.0)


def compute_pid(s1: np.ndarray, s2: np.ndarray, target: np.ndarray,
                lag: int = INFO_LAG) -> dict:
    """
    Partial Information Decomposition (I_min redundancy, Williams & Beer 2010).
    Computes: redundancy, unique_s1, unique_s2, synergy, total_mi.
    """
    n = min(len(s1), len(s2), len(target)) - lag
    if n < 10:
        return dict(pid_redundancy=np.nan, pid_unique_LH=np.nan,
                    pid_unique_RH=np.nan, pid_synergy=np.nan, pid_total_mi=np.nan)
    s1d  = _discretise(s1[:n])
    s2d  = _discretise(s2[:n])
    tgtd = _discretise(target[lag:lag + n])

    mi1  = _mi_histogram(s1d, tgtd)
    mi2  = _mi_histogram(s2d, tgtd)

    # I_min redundancy
    n_tgt = INFO_BINS
    red = 0.0
    p_tgt = np.bincount(tgtd, minlength=n_tgt).astype(float)
    p_tgt /= p_tgt.sum() + 1e-30
    for tv in range(n_tgt):
        if p_tgt[tv] == 0:
            continue
        mask_t = tgtd == tv
        if mask_t.sum() == 0:
            continue
        def _specific_surprise(sx):
            p_sx_t = np.bincount(sx[mask_t], minlength=INFO_BINS).astype(float)
            p_sx_t /= p_sx_t.sum() + 1e-30
            p_sx_all = np.bincount(sx[:n], minlength=INFO_BINS).astype(float)
            p_sx_all /= p_sx_all.sum() + 1e-30
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(p_sx_all > 0, p_sx_t / p_sx_all, 0)
                ss = float(np.sum(p_sx_t * np.log2(ratio + 1e-30)))
            return max(ss, 0.0)
        red += p_tgt[tv] * min(_specific_surprise(s1d),
                                _specific_surprise(s2d))

    mi_joint = _mi_histogram(
        s1d * INFO_BINS + s2d, tgtd, n_bins=INFO_BINS ** 2)
    uniq1   = max(mi1 - red, 0.0)
    uniq2   = max(mi2 - red, 0.0)
    synergy = max(mi_joint - mi1 - mi2 + red, 0.0)
    return dict(pid_redundancy=red, pid_unique_LH=uniq1,
                pid_unique_RH=uniq2, pid_synergy=synergy,
                pid_total_mi=mi_joint)


def compute_info_features(residuals: dict) -> dict:
    """
    Compute AIS / TE / PID from DDS residuals dict
    {roi_name: [residual_array, ...]}.
    """
    # Concatenate residual arrays per ROI into one long signal
    roi_signals = {}
    for roi_name, res_list in residuals.items():
        if res_list:
            roi_signals[roi_name] = np.concatenate(res_list)

    feats = {}

    # AIS per ROI
    for roi_name, sig in roi_signals.items():
        feats[f"ais_{roi_name}"] = compute_ais(sig)

    # TE between ROI pairs
    for src, tgt in TE_PAIRS:
        s_sig = roi_signals.get(src)
        t_sig = roi_signals.get(tgt)
        if s_sig is not None and t_sig is not None:
            feats[f"te_{src}_{tgt}"] = compute_te(s_sig, t_sig)
        else:
            feats[f"te_{src}_{tgt}"] = np.nan

    # PID
    s1_sig = roi_signals.get(PID_SRC[0])
    s2_sig = roi_signals.get(PID_SRC[1])
    tg_sig = roi_signals.get(PID_TGT)
    if s1_sig is not None and s2_sig is not None and tg_sig is not None:
        feats.update(compute_pid(s1_sig, s2_sig, tg_sig))
    else:
        feats.update(dict(pid_redundancy=np.nan, pid_unique_LH=np.nan,
                          pid_unique_RH=np.nan, pid_synergy=np.nan,
                          pid_total_mi=np.nan))

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Batch extraction
# ══════════════════════════════════════════════════════════════════════════════

def run_dds_extraction(fif_map: dict, labels_df: pd.DataFrame,
                       dds_csv: Path, info_csv: Path,
                       max_subjects: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_dict = labels_df.set_index("subject_id").to_dict("index")

    # Incremental caching — with dedup guard
    def _load_deduped_dds(csv_path):
        if not csv_path.exists():
            return set(), []
        df = pd.read_csv(csv_path, dtype={"subject_id": str})
        n_before = len(df)
        df = df.drop_duplicates(subset="subject_id", keep="last")
        if len(df) < n_before:
            log.warning("%s: dropped %d duplicate rows.", csv_path.name, n_before - len(df))
            df.to_csv(str(csv_path), index=False)
        return set(df["subject_id"].tolist()), df.to_dict("records")

    done_dds,  dds_records  = _load_deduped_dds(dds_csv)
    done_info, info_records = _load_deduped_dds(info_csv)

    if done_dds:
        log.info("Resuming DDS: %d already cached.", len(done_dds))

    subjects = sorted(fif_map.keys())
    if max_subjects > 0:
        subjects = subjects[:max_subjects]

    for sub_id in subjects:
        if sub_id not in label_dict:
            continue
        if sub_id in done_dds and sub_id in done_info:
            log.info("  [%s] Already cached – skipping.", sub_id)
            continue

        log.info("Processing %s …", sub_id)
        raw = preprocess_and_rename(fif_path=fif_map[sub_id])
        if raw is None:
            log.warning("  [%s] Preprocessing failed – skipping.", sub_id)
            continue

        lbl = label_dict[sub_id]
        meta = {"subject_id": sub_id, "group": lbl["group"],
                "BDI": lbl.get("BDI", np.nan)}

        if sub_id not in done_dds:
            dds_feats, residuals = compute_dds_features(raw)
            n_ok = sum(dds_feats.get(f"dds_{r}_n_wins_ok", 0) for r in ROI_MAP)
            log.info("  DDS: %d successful windows across ROIs.", n_ok)
            row_dds = {**meta, **dds_feats}
            dds_records.append(row_dds)
            # Save incrementally
            pd.DataFrame(dds_records).drop_duplicates(subset="subject_id", keep="last").to_csv(
                str(dds_csv), index=False)
        else:
            # Need residuals for info features — recompute DDS quickly
            _, residuals = compute_dds_features(raw)

        if sub_id not in done_info:
            info_feats = compute_info_features(residuals)
            row_info = {**meta, **info_feats}
            info_records.append(row_info)
            pd.DataFrame(info_records).drop_duplicates(subset="subject_id", keep="last").to_csv(
                str(info_csv), index=False)

        log.info("  [%s] Done. group=%s", sub_id,
                 "MDD" if lbl["group"] == 1 else "CTL")

    df_dds  = pd.read_csv(str(dds_csv))  if dds_csv.exists()  else pd.DataFrame()
    df_info = pd.read_csv(str(info_csv)) if info_csv.exists() else pd.DataFrame()
    log.info("DDS features:  %d subjects × %d features",
             len(df_dds),  df_dds.shape[1] if not df_dds.empty else 0)
    log.info("Info features: %d subjects × %d features",
             len(df_info), df_info.shape[1] if not df_info.empty else 0)
    return df_dds, df_info


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Classification
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

    steps = [
        ("imputer",   SimpleImputer(strategy="mean")),
        ("scaler",    StandardScaler()),
        ("var_thr",   VarianceThreshold(threshold=0.0)),
        ("pca",       PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("clf",       clf),
    ]
    if use_smote and HAS_SMOTE:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        steps.insert(3, ("smote", smote))
        return ImbPipeline(steps=steps)
    return Pipeline(steps=steps)


def run_classification(feature_sets: dict, labels_df: pd.DataFrame,
                       output_dir: Path, use_smote: bool = True,
                       n_pca: int = 40) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                                 f1_score, confusion_matrix, accuracy_score)
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    meta_cols = {"subject_id", "group", "BDI"}
    label_dict = labels_df.set_index("subject_id")["group"].to_dict()

    clfs = {
        "LogReg":  LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE),
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
        clfs["XGB"] = XGBClassifier(n_estimators=200, max_depth=4,
                                     learning_rate=0.05, eval_metric="logloss",
                                     random_state=RANDOM_STATE, n_jobs=N_JOBS,
                                     verbosity=0)

    all_results = []

    for fs_name, df_feats in feature_sets.items():
        if df_feats.empty:
            continue
        # Drop NaN-majority columns
        fcols = [c for c in df_feats.columns if c not in meta_cols]
        nan_frac = df_feats[fcols].isnull().mean()
        fcols = [c for c in fcols if nan_frac[c] <= 0.10]

        # Align labels
        df_feats = df_feats[df_feats["subject_id"].isin(label_dict)].copy()
        df_feats["group"] = df_feats["subject_id"].map(label_dict)
        df_feats = df_feats.dropna(subset=["group"])

        X = df_feats[fcols].values.astype(float)
        y = df_feats["group"].values.astype(int)

        n_train = int(len(y) * (CV_FOLDS - 1) / CV_FOLDS) - 1
        n_pca_safe = min(n_pca, n_train, X.shape[1])

        log.info("  [%s] %d subjects, %d features (CTL=%d, MDD=%d)",
                 fs_name, len(y), len(fcols), (y == 0).sum(), (y == 1).sum())

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)

        for clf_name, clf in clfs.items():
            pipe = build_pipeline(clf, use_smote=use_smote, n_pca=n_pca_safe)
            try:
                y_pred = cross_val_predict(pipe, X, y, cv=cv,
                                           method="predict", n_jobs=N_JOBS)
                try:
                    y_prob = cross_val_predict(pipe, X, y, cv=cv,
                                               method="predict_proba",
                                               n_jobs=N_JOBS)[:, 1]
                    auc = roc_auc_score(y, y_prob)
                except Exception:
                    auc = 0.5

                cm = confusion_matrix(y, y_pred)
                tn, fp, fn, tp = cm.ravel()
                res = {
                    "feature_set": fs_name,
                    "classifier":  clf_name,
                    "roc_auc":     round(auc, 3),
                    "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 3),
                    "f1":          round(f1_score(y, y_pred, zero_division=0), 3),
                    "sensitivity": round(tp / (tp + fn + 1e-9), 3),
                    "specificity": round(tn / (tn + fp + 1e-9), 3),
                    "n_subjects":  len(y),
                    "n_features":  len(fcols),
                }
                log.info("    %-12s %-10s AUC=%.3f  BalAcc=%.3f",
                         clf_name, "", auc, res["balanced_accuracy"])
                all_results.append(res)
            except Exception as exc:
                log.warning("    %-12s failed: %s", clf_name, exc)

    df_res = pd.DataFrame(all_results)
    out_csv = output_dir / "dds_classification_results.csv"
    df_res.to_csv(str(out_csv), index=False)
    log.info("Classification results → %s", out_csv)
    return df_res


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Group statistics
# ══════════════════════════════════════════════════════════════════════════════

def run_group_stats(df: pd.DataFrame, labels_df: pd.DataFrame,
                    output_dir: Path) -> None:
    """Print Cohen's d and t-test for key DDS features."""
    label_dict = labels_df.set_index("subject_id")["group"].to_dict()
    df = df[df["subject_id"].isin(label_dict)].copy()
    df["group"] = df["subject_id"].map(label_dict)

    meta = {"subject_id", "group", "BDI"}
    key_feats = [c for c in df.columns
                 if c not in meta and any(x in c for x in
                     ["f1", "f2", "alpha", "A_ratio", "f_diff",
                      "ais_", "te_", "pid_"])]

    rows = []
    for feat in key_feats:
        ctl = df.loc[df.group == 0, feat].dropna().values
        mdd = df.loc[df.group == 1, feat].dropna().values
        if len(ctl) < 3 or len(mdd) < 3:
            continue
        pooled_sd = np.sqrt((ctl.std() ** 2 + mdd.std() ** 2) / 2 + 1e-30)
        d = (mdd.mean() - ctl.mean()) / pooled_sd
        _, p = ttest_ind(ctl, mdd, equal_var=False)
        rows.append({"feature": feat, "CTL_mean": ctl.mean(), "MDD_mean": mdd.mean(),
                     "cohens_d": d, "p_value": p})

    df_stats = pd.DataFrame(rows)
    if df_stats.empty:
        log.warning("Not enough subjects per group for statistics.")
        return
    df_stats = df_stats.sort_values("cohens_d", key=abs, ascending=False)
    stats_csv = output_dir / "dds_group_stats.csv"
    df_stats.to_csv(str(stats_csv), index=False)
    log.info("Group stats → %s", stats_csv)

    top = df_stats.head(15)
    print("\n── Top DDS/Info features by |Cohen's d| ──")
    print(top[["feature", "CTL_mean", "MDD_mean", "cohens_d", "p_value"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def compare_all(df_clf: pd.DataFrame, output_dir: Path) -> None:
    # ds003474 reference results
    ds003474 = [
        {"pipeline": "ds003474 / Baseline spectral",        "roc_auc": 0.675, "dataset": "ds003474"},
        {"pipeline": "ds003474 / Baseline+Connectivity",    "roc_auc": 0.807, "dataset": "ds003474"},
        {"pipeline": "ds003474 / Microstates+Baseline",     "roc_auc": 0.825, "dataset": "ds003474"},
        {"pipeline": "ds003474 / DDS+Info+Baseline",        "roc_auc": 0.836, "dataset": "ds003474"},
    ]
    # Best AUC per feature_set from ds005356
    best = df_clf.groupby("feature_set")["roc_auc"].max().reset_index()
    best_clf = df_clf.loc[df_clf.groupby("feature_set")["roc_auc"].idxmax()][
        ["feature_set", "classifier", "roc_auc"]]

    ds005356_rows = []
    for _, row in best_clf.iterrows():
        ds005356_rows.append({
            "pipeline": f"ds005356 / {row['feature_set']} / {row['classifier']}",
            "roc_auc":  row["roc_auc"],
            "dataset":  "ds005356",
        })

    df_comp = pd.DataFrame(ds003474 + ds005356_rows).sort_values("roc_auc", ascending=False)
    comp_csv = output_dir / "comparison_full_ds003474_vs_ds005356.csv"
    df_comp.to_csv(str(comp_csv), index=False)

    print("\n" + "=" * 70)
    print("FULL COMPARATIVE RESULTS: ds003474 vs ds005356 (DDS+Info+Baseline)")
    print("=" * 70)
    print(df_comp.to_string(index=False))
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DDS+Info pipeline for ds005356 concurrent EEG")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip DDS/Info extraction; use cached CSVs")
    parser.add_argument("--max_subjects", type=int, default=0,
                        help="Limit N subjects (0=all; use 5-10 for debug)")
    parser.add_argument("--no_smote", action="store_true")
    parser.add_argument("--n_pca", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dds_csv     = out_dir / "dds_features.csv"
    info_csv    = out_dir / "info_features.csv"
    merged_csv  = out_dir / "dds_info_features.csv"
    full_csv    = out_dir / "dds_info_baseline_features.csv"

    # ── Labels ──────────────────────────────────────────────────────────────
    log.info("Loading group labels …")
    labels_df = load_group_labels(EXCEL_FILE)
    log.info("  CTL=%d  MDD=%d", (labels_df.group == 0).sum(), (labels_df.group == 1).sum())

    # ── Extract ─────────────────────────────────────────────────────────────
    if not args.skip_extract:
        log.info("Scanning BIDS root for FIF files …")
        fif_map = find_fif_files(BIDS_ROOT)
        log.info("Starting DDS+Info extraction …")
        df_dds, df_info = run_dds_extraction(
            fif_map, labels_df, dds_csv, info_csv,
            max_subjects=args.max_subjects)
    else:
        if not dds_csv.exists():
            raise FileNotFoundError(f"DDS cache not found: {dds_csv}")
        df_dds  = pd.read_csv(str(dds_csv))
        df_info = pd.read_csv(str(info_csv)) if info_csv.exists() else pd.DataFrame()
        log.info("Loaded cached DDS (%d subj) + Info (%d subj)",
                 len(df_dds), len(df_info) if not df_info.empty else 0)

    # ── Merge feature sets ───────────────────────────────────────────────────
    log.info("Merging feature sets …")
    meta_cols = ["subject_id", "group", "BDI"]

    # DDS + Info
    if not df_info.empty:
        info_feat_cols = [c for c in df_info.columns
                          if c not in {"subject_id", "group", "BDI"}]
        df_dds_info = df_dds.merge(
            df_info[["subject_id"] + info_feat_cols],
            on="subject_id", how="inner")
        df_dds_info.to_csv(str(merged_csv), index=False)
    else:
        df_dds_info = df_dds.copy()

    # DDS + Info + Baseline
    if BASELINE_CSV.exists():
        df_base = pd.read_csv(str(BASELINE_CSV))
        base_feat_cols = [c for c in df_base.columns
                          if c not in {"subject_id", "group", "BDI"}]
        df_full = df_dds_info.merge(
            df_base[["subject_id"] + base_feat_cols],
            on="subject_id", how="inner", suffixes=("", "_base"))
        df_full.to_csv(str(full_csv), index=False)
        log.info("Full merged CSV → %s  (%d subj × %d feats)",
                 full_csv, len(df_full), df_full.shape[1])
    else:
        df_full = df_dds_info.copy()
        log.warning("Baseline CSV not found at %s — DDS+Info only.", BASELINE_CSV)

    # ── Group statistics ─────────────────────────────────────────────────────
    log.info("Computing group statistics …")
    run_group_stats(df_dds_info, labels_df, out_dir)

    # ── Classify ─────────────────────────────────────────────────────────────
    log.info("Running classification …")
    feature_sets = {"DDS only": df_dds}
    if not df_info.empty:
        feature_sets["Info only"] = df_info
        feature_sets["DDS+Info"]  = df_dds_info
    if not df_full.empty and df_full.shape[1] > df_dds_info.shape[1]:
        feature_sets["DDS+Info+Baseline"] = df_full

    df_clf = run_classification(
        feature_sets, labels_df, out_dir,
        use_smote=not args.no_smote, n_pca=args.n_pca)

    # ── Compare with ds003474 ─────────────────────────────────────────────────
    compare_all(df_clf, out_dir)

    log.info("Pipeline complete. Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
