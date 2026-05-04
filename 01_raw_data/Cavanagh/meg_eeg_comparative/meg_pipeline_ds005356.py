"""
meg_pipeline_ds005356.py
========================
Adaptation of the ds003474 EEG depression classification pipeline to ds005356.

Dataset: ds005356 – "MEG: Major Depression & Probabilistic Learning Task"
         Pirrung, Singh, Hogeveen, Quinn & Cavanagh (2025)
         Biol. Psychiatry: CNNI
         90 subjects (38 CTL, 52 MDD), 306-sensor Elekta Neuromag MEG
         + simultaneous 71-channel EEG, 1000 Hz, Probabilistic Selection Task.

Strategy
--------
The ds005356 recordings include SIMULTANEOUS EEG (71 channels) co-registered
with the MEG data.  Because ds003474 features were computed from 64-channel
EEG, the most direct comparison reuses the same EEG feature extraction on the
concurrent EEG channels embedded in every .fif file.  This avoids the need to
re-derive channel atlases from MEG sensor space.

Key adaptations vs. ds003474 EEG pipeline
------------------------------------------
1. File format: MNE reads .fif directly (no EEGLAB .set loader needed).
2. EEG channels: Named EEG001–EEG074 (not 10-10 labels). Positions are
   embedded in the FIF digitization block (71-channel DigMontage).
   Feature computation uses channel-index-based region groupings derived from
   anterior / posterior sensor order rather than named-channel lookups.
3. Sampling rate: 1000 Hz → resample to 500 Hz for consistency with ds003474.
4. Group labels: Derived from the Excel spreadsheet via URSI→BIDS mapping:
       sub-M87100XXX  →  URSI = int(sub_id[7:]) → CTL/MDD from Excel.
5. Task epochs: ds005356 uses the same Probabilistic Selection Task. Events
   are encoded as cue onset (trial_type "cue/*") and feedback ("FB/win",
   "FB/loss").  Two analysis modes are supported:
       (a) Continuous: treat the full recording as quasi-resting (same as
           ds003474 which used a PST recording in quasi-resting mode).
       (b) Epoch: segment around feedback onset (−0.2 s → +0.8 s) and
           analyse each epoch independently.
   Default: continuous mode for direct comparability.
6. No mastoid reference channels (M1/M2) in this dataset.
   Average reference is applied to EEG channels.
7. Power-line notch: 60 Hz (US recording, same as ds003474).
8. ICA: run on EEG channels only; fallback EOG detection uses first two
   frontal EEG channels (EEG001, EEG002 ≈ anterior).

Usage
-----
    # Full pipeline (extract + classify)
    python meg_pipeline_ds005356.py

    # Skip EEG extraction, classify from cached CSV
    python meg_pipeline_ds005356.py --skip_extract

    # Epoch mode (peri-feedback epochs instead of continuous)
    python meg_pipeline_ds005356.py --epoch_mode

    # Limit to N subjects for debugging
    python meg_pipeline_ds005356.py --max_subjects 10

    # Custom output directory
    python meg_pipeline_ds005356.py --output_dir results_test

Dependencies
------------
    pip install mne numpy scipy pandas scikit-learn matplotlib seaborn
    pip install antropy nolds imbalanced-learn xgboost openpyxl
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from scipy import signal as sci_signal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, confusion_matrix, roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import antropy as ant
    HAS_ANT = True
except ImportError:
    HAS_ANT = False

try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS & PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

BIDS_ROOT   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds005356")
EXCEL_FILE  = BIDS_ROOT / "Code" / "MEG MDD IDs and Quex.xlsx"
OUTPUT_DIR  = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/meg_eeg_comparative")

# Preprocessing
SFREQ_TARGET     = 500       # Hz – resample to match ds003474
HIGHPASS_HZ      = 1.0
LOWPASS_HZ       = 45.0
POWERLINE_HZ     = 60
AMPLITUDE_THRESH = 500e-6   # V – epoch rejection threshold

# Epoch parameters (epoch mode)
EPOCH_TMIN       = -0.2     # s before feedback onset
EPOCH_TMAX       = 0.8      # s after feedback onset
EPOCH_BASELINE   = (-0.2, 0.0)

# Continuous-mode window
WINDOW_DURATION  = 2.0      # s per analysis window
WINDOW_OVERLAP   = 0.0      # s overlap

# Frequency bands (same as ds003474)
FREQ_BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# EEG anterior / posterior channel groups (by index, 71-channel system)
# EEG001-EEG020  ≈ frontal / fronto-central
# EEG021-EEG040  ≈ central / temporal
# EEG041-EEG060  ≈ parietal / centroparietal
# EEG061-EEG071  ≈ occipital
EEG_REGIONS = {
    "frontal":          list(range(0,  20)),   # EEG001–EEG020
    "central":          list(range(20, 40)),   # EEG021–EEG040
    "parietal":         list(range(40, 60)),   # EEG041–EEG060
    "occipital":        list(range(60, 71)),   # EEG061–EEG071
}

# Frontal alpha asymmetry: approximate left vs. right based on digitization
# (EEG001 is typically anterior-left, EEG002 anterior-right for this system;
#  without named channels we use the first two frontal pairs)
ASYMMETRY_PAIRS_IDX = [(1, 0), (3, 2), (5, 4), (7, 6)]  # (right_idx, left_idx)

# Classification
CV_FOLDS      = 5
RANDOM_STATE  = 42
N_JOBS        = -1

# ICA
N_ICA_COMPONENTS = 20

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: GROUP LABEL EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def load_group_labels(excel_path: Path) -> pd.DataFrame:
    """
    Load CTL/MDD group labels from the Excel spreadsheet.

    The BIDS subject ID encodes the URSI:
        sub-M87100058 → URSI = int("100058") - 100000 = 58  (for URSI < 10000)
    or more generally:
        sub-M87XXXXXXX → URSI = int(sub_id[3:])  (drop "M87")
    The URSI column in Excel matches this interpretation.

    Returns a DataFrame with columns: ['subject_id', 'group', 'BDI', 'SHAPS']
    with group 0=CTL, 1=MDD.
    """
    df = pd.read_excel(str(excel_path), sheet_name="Sheet1", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # Map URSI → BIDS subject ID
    # sub-M87100058 → URSI 58 → M87 prefix + 6-digit pad → M87100058
    def ursi_to_bids(ursi: int) -> str:
        return f"sub-M87{100000 + int(ursi)}"

    records = []
    for _, row in df.iterrows():
        try:
            ursi  = int(row["URSI"])
            group = 1 if str(row["Group"]).strip() == "MDD" else 0
            bdi   = float(row["BDI"]) if pd.notna(row.get("BDI")) else np.nan
            shaps = float(row["SHAPS"]) if pd.notna(row.get("SHAPS")) else np.nan
            records.append({
                "subject_id": ursi_to_bids(ursi),
                "group":      group,
                "BDI":        bdi,
                "SHAPS":      shaps,
            })
        except Exception:
            continue

    result = pd.DataFrame(records)
    n_ctl = (result["group"] == 0).sum()
    n_mdd = (result["group"] == 1).sum()
    log.info("Group labels: CTL=%d  MDD=%d  (total=%d)", n_ctl, n_mdd, len(result))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: FIND SUBJECT FILES
# ──────────────────────────────────────────────────────────────────────────────

def find_fif_files(bids_root: Path) -> dict:
    """
    Scan BIDS root for *_meg.fif files (split-01 only to avoid duplicates).

    Returns {subject_id: Path} for the first split of each subject's run-1.
    """
    fif_map = {}
    for sub_dir in sorted(bids_root.glob("sub-M*")):
        sub_id = sub_dir.name
        # Find first split of run-1
        pattern = f"{sub_id}_ses-01_task-pst_run-1_split-01_meg.fif"
        candidates = list(sub_dir.glob(f"ses-01/meg/{pattern}"))
        if candidates:
            fif_map[sub_id] = candidates[0]
        else:
            # Fallback: prefer split-01, then any single-run fif
            all_fifs = sorted(sub_dir.glob("ses-01/meg/*_meg.fif"))
            split01 = [f for f in all_fifs if "split-01" in f.name]
            if split01:
                fif_map[sub_id] = split01[0]
            elif all_fifs:
                # Single-run subjects (no split suffix)
                fif_map[sub_id] = all_fifs[0]

    log.info("Found %d subjects with FIF files.", len(fif_map))
    return fif_map


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_meg_eeg(
    fif_path: Path,
    resample_hz: float = SFREQ_TARGET,
) -> tuple:
    """
    Load MEG+EEG .fif and return cleaned EEG-only Raw.

    Adaptations vs. ds003474:
    - Loads .fif (not .set); EEG channel positions already embedded.
    - Picks EEG channels only (71 channels).
    - Resamples 1000 → 500 Hz.
    - Average reference (same as ds003474).
    - ICA with EOG fallback (no separate EOG .tsv here).
    - No mastoid channels to drop.
    """
    report = {
        "subject_id": fif_path.parent.parent.parent.name,
        "n_eeg_ch": 0,
        "n_ica_removed": 0,
        "excluded": False,
        "exclusion_reason": "",
    }

    try:
        # Load full recording; read_raw_fif auto-concatenates if split
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    except Exception as exc:
        # Try concatenating splits manually
        try:
            fif_dir = fif_path.parent
            splits = sorted(fif_dir.glob(
                fif_path.name.replace("split-01", "split-*")
            ))
            raws = [mne.io.read_raw_fif(str(s), preload=True, verbose=False)
                    for s in splits]
            raw = mne.concatenate_raws(raws)
        except Exception as exc2:
            report["excluded"] = True
            report["exclusion_reason"] = f"Load error: {exc2}"
            return None, report

    # ── Pick EEG channels only ────────────────────────────────────────────────
    raw.pick_types(meg=False, eeg=True, eog=False, verbose=False)
    n_eeg = len(raw.ch_names)
    report["n_eeg_ch"] = n_eeg
    log.info("[%s]  EEG channels: %d, sfreq: %.0f Hz, duration: %.1f s",
             report["subject_id"], n_eeg, raw.info["sfreq"], raw.times[-1])

    if n_eeg < 30:
        report["excluded"] = True
        report["exclusion_reason"] = f"Too few EEG channels: {n_eeg}"
        return None, report

    # ── Resample to target frequency ─────────────────────────────────────────
    if abs(raw.info["sfreq"] - resample_hz) > 1:
        raw.resample(resample_hz, npad="auto", verbose=False)
        log.info("  Resampled to %.0f Hz.", resample_hz)

    # ── Band-pass + notch filter ──────────────────────────────────────────────
    raw.filter(l_freq=HIGHPASS_HZ, h_freq=LOWPASS_HZ,
               method="fir", fir_window="hamming", verbose=False)
    raw.notch_filter(freqs=[POWERLINE_HZ], method="fir", verbose=False)
    log.info("  Filtered: [%.1f–%.1f Hz], notch @ %.0f Hz.",
             HIGHPASS_HZ, LOWPASS_HZ, POWERLINE_HZ)

    # ── Bad channel detection on PRE-REFERENCE data (relative thresholds) ────
    # IMPORTANT: bad-channel detection must happen BEFORE average reference.
    # In MEG+EEG FIFs the EEG is typically stored with an original reference
    # (often a linked-mastoid or a single electrode).  One artifact channel
    # (e.g. the physical reference, std >> 50 µV) dominates the common-mode
    # signal.  If average reference is applied first, that artifact redistributes
    # ~1/n_ch of its amplitude to every channel, making all "good" channels
    # appear near-flat when compared against a fixed µV threshold.
    #
    # We therefore use RELATIVE thresholds on the pre-reference filtered data:
    #   - flat   : std < 10% of the median std across channels
    #   - noisy  : std > 5× the median std across channels
    # This is hardware-agnostic and works regardless of amplifier scale.
    data_pre = raw.get_data()           # (n_eeg, n_times)
    stds_pre  = data_pre.std(axis=1)
    med_std   = float(np.median(stds_pre))

    flat_bads  = [raw.ch_names[i] for i, s in enumerate(stds_pre)
                  if s < 0.10 * med_std]
    noisy_bads = [raw.ch_names[i] for i, s in enumerate(stds_pre)
                  if s > 5.0 * med_std]

    bad_chs = list(set(flat_bads + noisy_bads))
    log.info("  Bad channels (pre-ref, med_std=%.2e V): "
             "%d flat (<10%% med), %d noisy (>5x med) → %d total",
             med_std, len(flat_bads), len(noisy_bads), len(bad_chs))

    if bad_chs and len(bad_chs) < n_eeg * 0.25:
        raw.info["bads"] = bad_chs
        try:
            raw.interpolate_bads(reset_bads=True, verbose=False)
            log.info("  Interpolated %d bad channels.", len(bad_chs))
        except Exception as exc:
            log.warning("  Interpolation failed (%s) – clearing bads.", exc)
            raw.info["bads"] = []
    elif bad_chs:
        log.warning("  Too many bad channels (%d/%d) – skipping interpolation.",
                    len(bad_chs), n_eeg)
        raw.info["bads"] = []

    # ── Average reference (AFTER bad-channel cleanup) ─────────────────────────
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

    # Always clear bads before ICA so ICA sees ALL channels.
    # (picks="eeg" in MNE respects raw.info["bads"], which is why the original
    # code saw only 10 channels when 61 were marked bad.)
    raw.info["bads"] = []

    # ── ICA ───────────────────────────────────────────────────────────────────
    # Use explicit picks array (exclude=[]) to guarantee all EEG channels are
    # passed to ICA regardless of the bads list (which we just cleared, but
    # being explicit is safer across MNE versions).
    all_eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    n_comps = min(N_ICA_COMPONENTS, len(all_eeg_picks) - 1)
    log.info("  Running ICA: %d components on %d EEG channels.",
             n_comps, len(all_eeg_picks))

    ica = ICA(n_components=n_comps, method="fastica",
              fit_params={"tol": 1e-4, "max_iter": 300},
              random_state=RANDOM_STATE, verbose=False)
    try:
        raw_1hz = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        # Subsample to at most 300 s for ICA fitting speed.
        # FastICA on 677 000 samples (1355 s × 500 Hz) can take > 5 min;
        # 150 000 samples (300 s) converges in < 60 s with identical results.
        max_ica_samples = int(300 * raw.info["sfreq"])
        if raw_1hz.n_times > max_ica_samples:
            raw_ica_fit = raw_1hz.copy().crop(tmax=300.0)
            log.info("  ICA fit on first 300 s (%d samples) for speed.",
                     max_ica_samples)
        else:
            raw_ica_fit = raw_1hz
        # Pass explicit picks so bads cannot interfere
        ica.fit(raw_ica_fit, picks=all_eeg_picks, verbose=False)

        # Fallback EOG: use first two anterior EEG channels
        eog_chs = [raw.ch_names[p] for p in all_eeg_picks[:2]]
        try:
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=eog_chs,
                                             threshold=3.5, verbose=False)
        except Exception:
            eog_idx = []

        # Muscle detection: high HF/LF power ratio in ICA activations
        src = ica.get_sources(raw_1hz).get_data()
        muscle_idx = []
        for ci in range(src.shape[0]):
            if ci in eog_idx:
                continue
            freqs, psd = sci_signal.welch(src[ci], fs=raw.info["sfreq"],
                                           nperseg=int(raw.info["sfreq"]))
            lf = np.trapz(psd[(freqs >= 1) & (freqs < 20)],
                           freqs[(freqs >= 1) & (freqs < 20)])
            hf = np.trapz(psd[(freqs >= 20) & (freqs <= 45)],
                           freqs[(freqs >= 20) & (freqs <= 45)])
            if lf > 0 and (hf / lf) > 5.0:
                muscle_idx.append(ci)

        exclude = sorted(set(eog_idx + muscle_idx))
        ica.exclude = exclude
        if exclude:
            ica.apply(raw, verbose=False)
        report["n_ica_removed"] = len(exclude)
        log.info("  ICA: removed %d components (eog=%d, muscle=%d).",
                 len(exclude), len(eog_idx), len(muscle_idx))
    except Exception as exc:
        log.warning("  ICA failed: %s – proceeding without ICA.", exc)

    return raw, report


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_features_continuous(
    raw: mne.io.BaseRaw,
    window_s: float = WINDOW_DURATION,
    overlap_s: float = WINDOW_OVERLAP,
    compute_entropy: bool = False,
) -> dict:
    """
    Extract fast spectral + Hjorth features from cleaned EEG using sliding windows.

    Feature set is aligned with the Boruta-confirmed discriminators from ds003474:
      α/β ratio, θ/β ratio, θ/α ratio (the 5 confirmed + 6 tentative features
      are all spectral ratios at specific electrodes).

    Per channel (vectorised Welch PSD — one scipy call for all channels):
      - Band power absolute (δ/θ/α/β/γ)            5
      - Band power relative                          5
      - α/β, θ/β, θ/α ratios                        3
      - Hjorth activity, mobility, complexity        3
      - Spectral entropy (optional, compute_entropy) 1
    Regional aggregates (frontal / central / parietal / occipital):
      - mean α/β ratio, mean θ/β ratio              2 × 4 = 8
    Frontal alpha asymmetry (4 index-based pairs):  4

    Total (default): 71 ch × 16 = 1136 + 8 + 4 = ~1148 features
    With entropy:    71 ch × 17 = 1207 + 12 = ~1219 features

    Omitted vs. original: SampEn (O(n²)), DFA, AIS, PLV, coherence.
    All omitted features can be re-added with compute_entropy=True or a
    separate advanced run if needed.

    Speed: ~1–3 s per subject on a modern CPU (vs. ~100 min with SampEn).
    """
    data  = raw.get_data()           # (n_ch, n_times)
    sfreq = raw.info["sfreq"]
    n_ch, n_times = data.shape

    win_len  = int(window_s * sfreq)
    hop_len  = int((window_s - overlap_s) * sfreq)
    nperseg  = min(win_len, int(sfreq * 2))

    # Pre-compute band frequency masks (same for every window)
    dummy_f = sci_signal.welch(np.zeros(win_len), fs=sfreq, nperseg=nperseg)[0]
    band_masks = {
        band: (dummy_f >= flo) & (dummy_f <= fhi)
        for band, (flo, fhi) in FREQ_BANDS.items()
    }

    # Accumulators: (n_windows, n_ch) arrays per feature type
    # Collect lists per window, convert to array at the end for fast nanmean
    acc: dict[str, list] = {}

    def _add_vec(name: str, arr: np.ndarray):
        """Append a (n_ch,) vector for the current window."""
        acc.setdefault(name, []).append(arr.copy())

    def _add_scalar(name: str, val: float):
        acc.setdefault(name, []).append(np.array([val]))

    windows_used = 0

    for start in range(0, n_times - win_len + 1, hop_len):
        w = data[:, start:start + win_len]   # (n_ch, win_len)

        # Amplitude-based artifact rejection
        pk2pk = w.max(axis=1) - w.min(axis=1)
        if (pk2pk > AMPLITUDE_THRESH).any():
            continue
        windows_used += 1

        # ── Vectorised Welch PSD: one call for ALL channels ──────────────────
        _, psd = sci_signal.welch(w, fs=sfreq, nperseg=nperseg, axis=1)
        # psd shape: (n_ch, n_freqs)

        band_pow = {}  # band → (n_ch,) absolute power
        for band, mask in band_masks.items():
            bp = np.trapz(psd[:, mask], dummy_f[mask], axis=1)
            band_pow[band] = np.maximum(bp, 0.0)

        total_pow = sum(band_pow.values())                # (n_ch,)

        # Absolute band powers
        for band, bp in band_pow.items():
            _add_vec(f"{band}_abs", bp)

        # Relative band powers
        with np.errstate(invalid="ignore", divide="ignore"):
            for band, bp in band_pow.items():
                _add_vec(f"{band}_rel", np.where(total_pow > 0, bp / total_pow, np.nan))

        # Spectral ratios
        a = band_pow["alpha"]
        b = band_pow["beta"]
        t = band_pow["theta"]
        with np.errstate(invalid="ignore", divide="ignore"):
            _add_vec("alpha_beta_ratio", np.where(b > 0, a / b, np.nan))
            _add_vec("theta_beta_ratio", np.where(b > 0, t / b, np.nan))
            _add_vec("theta_alpha_ratio", np.where(a > 0, t / a, np.nan))

        # ── Vectorised Hjorth parameters ──────────────────────────────────────
        d1 = np.diff(w, axis=1)                    # (n_ch, win_len-1)
        d2 = np.diff(d1, axis=1)                   # (n_ch, win_len-2)
        act = w.var(axis=1)                         # (n_ch,)
        var_d1 = d1.var(axis=1)
        var_d2 = d2.var(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mob = np.sqrt(np.where(act > 0, var_d1 / act, np.nan))
            com = np.where(
                (mob > 0) & (var_d1 > 0),
                np.sqrt(var_d2 / var_d1) / mob,
                np.nan,
            )
        _add_vec("hjorth_act", act)
        _add_vec("hjorth_mob", mob)
        _add_vec("hjorth_com", com)

        # ── Optional spectral entropy ─────────────────────────────────────────
        if compute_entropy:
            psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-30)
            sp_ent   = -(psd_norm * np.log2(psd_norm + 1e-30)).sum(axis=1)
            _add_vec("spec_entropy", sp_ent)

        # ── Frontal alpha asymmetry (index-based) ─────────────────────────────
        for pair_idx, (r_idx, l_idx) in enumerate(ASYMMETRY_PAIRS_IDX):
            if r_idx < n_ch and l_idx < n_ch:
                with np.errstate(invalid="ignore"):
                    faa = np.log(a[r_idx] + 1e-30) - np.log(a[l_idx] + 1e-30)
                _add_scalar(f"faa_pair{pair_idx}", float(faa))

        # ── Regional mean spectral ratios ─────────────────────────────────────
        for reg, idxs in EEG_REGIONS.items():
            valid = [i for i in idxs if i < n_ch]
            if valid:
                a_m = a[valid].mean()
                b_m = b[valid].mean()
                t_m = t[valid].mean()
                with np.errstate(invalid="ignore", divide="ignore"):
                    _add_scalar(f"alpha_beta_ratio_{reg}",
                                float(a_m / b_m) if b_m > 0 else np.nan)
                    _add_scalar(f"theta_beta_ratio_{reg}",
                                float(t_m / b_m) if b_m > 0 else np.nan)

    if windows_used == 0:
        log.warning("  No valid windows found after artifact rejection.")
        return {}

    # ── Flatten accumulators: average over windows, then over channels ─────────
    feats = {}
    for name, arrays in acc.items():
        mat = np.array(arrays)           # (n_windows, n_ch) or (n_windows, 1)
        mean_over_windows = np.nanmean(mat, axis=0)   # (n_ch,) or (1,)
        if mean_over_windows.size == 1:
            feats[name] = float(mean_over_windows[0])
        else:
            for c, v in enumerate(mean_over_windows):
                feats[f"{name}_ch{c:03d}"] = float(v)

    log.info("  Extracted %d features from %d windows.", len(feats), windows_used)
    return feats


def extract_features_epochs(
    raw: mne.io.BaseRaw,
    events_tsv: Path,
    event_type: str = "FB",
) -> dict:
    """
    Extract features from peri-feedback epochs (−0.2 → +0.8 s).

    event_type: 'FB' (all feedback), 'FB/win', or 'FB/loss'.
    If no events file or insufficient epochs, falls back to continuous mode.
    """
    if not events_tsv.exists():
        log.warning("  Events TSV not found – using continuous mode.")
        return extract_features_continuous(raw)

    try:
        ev_df = pd.read_csv(str(events_tsv), sep="\t")
        mask  = ev_df["trial_type"].str.startswith(event_type, na=False)
        ev_df = ev_df[mask].copy()

        if len(ev_df) < 10:
            log.warning("  Fewer than 10 %s events – using continuous mode.", event_type)
            return extract_features_continuous(raw)

        # Build MNE events array [sample, 0, event_id]
        sfreq = raw.info["sfreq"]
        events_arr = np.column_stack([
            (ev_df["onset"].values * sfreq).astype(int),
            np.zeros(len(ev_df), dtype=int),
            np.ones(len(ev_df), dtype=int),
        ])

        epochs = mne.Epochs(
            raw, events_arr,
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=EPOCH_BASELINE,
            preload=True, verbose=False,
        )
        epochs.drop_bad(reject={"eeg": AMPLITUDE_THRESH}, verbose=False)

        if len(epochs) < 10:
            log.warning("  Fewer than 10 clean epochs after rejection – continuous.")
            return extract_features_continuous(raw)

        # Treat epochs as pseudo-continuous windows and reuse the fast extractor.
        # Create a temporary Raw from the epoch data so extract_features_continuous
        # can handle it without duplicating the vectorised logic.
        log.info("  Using %d clean epochs (%.2f s each).", len(epochs),
                 EPOCH_TMAX - EPOCH_TMIN)
        epoch_sfreq = epochs.info["sfreq"]
        data_ep     = epochs.get_data()          # (n_ep, n_ch, n_times)
        n_ep, n_ch, n_ep_times = data_ep.shape

        # Stack epochs into a single pseudo-continuous array and wrap in Raw
        data_concat = data_ep.reshape(n_ch, n_ep * n_ep_times)
        info_ep = mne.create_info(ch_names=epochs.ch_names,
                                   sfreq=epoch_sfreq, ch_types="eeg")
        raw_ep = mne.io.RawArray(data_concat, info_ep, verbose=False)
        # Each epoch is n_ep_times samples → use that as the window length
        return extract_features_continuous(
            raw_ep,
            window_s=(n_ep_times / epoch_sfreq),
            overlap_s=0.0,
        )

    except Exception as exc:
        log.warning("  Epoch extraction failed (%s) – falling back to continuous.", exc)
        return extract_features_continuous(raw)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: BATCH PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def run_extraction(
    fif_map: dict,
    labels_df: pd.DataFrame,
    output_csv: Path,
    epoch_mode: bool = False,
    max_subjects: int = 0,
    compute_entropy: bool = False,
) -> pd.DataFrame:
    """
    Preprocess + extract features for all subjects.
    Saves features.csv with columns: subject_id, group, BDI, feat1, feat2, ...

    compute_entropy : add spectral entropy per channel (marginal extra time).
                      Sample entropy / DFA remain disabled (too slow).
    """
    label_dict = labels_df.set_index("subject_id").to_dict("index")

    # Incremental caching: load already-processed subjects to skip them
    already_done = set()
    existing_records = []
    if output_csv.exists():
        cached_df = pd.read_csv(str(output_csv))
        n_before = len(cached_df)
        cached_df = cached_df.drop_duplicates(subset="subject_id", keep="last")
        if len(cached_df) < n_before:
            log.warning("Dropped %d duplicate rows from cached CSV.", n_before - len(cached_df))
            cached_df.to_csv(str(output_csv), index=False)
        already_done = set(cached_df["subject_id"].tolist())
        existing_records = cached_df.to_dict("records")
        log.info("Resuming: %d subjects already in cache, will skip them.", len(already_done))

    records = list(existing_records)

    subjects = sorted(fif_map.keys())
    if max_subjects > 0:
        subjects = subjects[:max_subjects]

    for sub_id in subjects:
        if sub_id in already_done:
            log.info("  [%s] Already cached – skipping.", sub_id)
            continue
        if sub_id not in label_dict:
            log.info("  [%s] No group label – skipping.", sub_id)
            continue

        fif_path = fif_map[sub_id]
        log.info("Processing %s …", sub_id)

        raw, rpt = preprocess_meg_eeg(fif_path)
        if raw is None:
            log.warning("  [%s] Excluded: %s", sub_id, rpt["exclusion_reason"])
            continue

        if epoch_mode:
            # Strip split suffix if present; works for both split and single-run files
            stem = fif_path.name
            for suffix in ("_split-01_meg.fif", "_split-02_meg.fif", "_meg.fif"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            ev_tsv = fif_path.parent / f"{stem}_events.tsv"
            feats = extract_features_epochs(raw, ev_tsv)
        else:
            feats = extract_features_continuous(raw, compute_entropy=compute_entropy)

        if not feats:
            log.warning("  [%s] No features extracted – skipping.", sub_id)
            continue

        lbl = label_dict[sub_id]
        row = {
            "subject_id": sub_id,
            "group":       lbl["group"],
            "BDI":         lbl.get("BDI", np.nan),
        }
        row.update(feats)
        records.append(row)
        log.info("  [%s] Done. group=%s, n_feats=%d",
                 sub_id, "MDD" if lbl["group"] == 1 else "CTL", len(feats))

    df = pd.DataFrame(records).drop_duplicates(subset="subject_id", keep="last").reset_index(drop=True)
    df.to_csv(str(output_csv), index=False)
    log.info("Features saved → %s  (%d subjects)", output_csv, len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def get_classifiers():
    clfs = {
        "LogReg": LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                      class_weight="balanced",
                                      random_state=RANDOM_STATE),
        "LDA":    LDA(solver="svd"),
        "SVM-RBF": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                       class_weight="balanced", random_state=RANDOM_STATE),
        "SVM-Lin": SVC(kernel="linear", C=0.1, probability=True,
                       class_weight="balanced", random_state=RANDOM_STATE),
        "RF":     RandomForestClassifier(n_estimators=300, max_depth=None,
                                          min_samples_leaf=2,
                                          class_weight="balanced",
                                          n_jobs=N_JOBS, random_state=RANDOM_STATE),
        "MLP":    MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                 activation="relu", solver="adam",
                                 alpha=1e-3, max_iter=500,
                                 early_stopping=True, validation_fraction=0.15,
                                 random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        clfs["XGB"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", scale_pos_weight=52/38,
            n_jobs=N_JOBS, random_state=RANDOM_STATE, verbosity=0,
        )
    return clfs


def build_pipeline(clf, use_smote: bool = True, n_pca: int = 40):
    steps = [
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("var_thresh", VarianceThreshold(threshold=1e-6)),
        ("pca",        PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("clf",        clf),
    ]
    if use_smote and HAS_SMOTE:
        smote = SMOTE(sampling_strategy="minority", k_neighbors=5,
                      random_state=RANDOM_STATE)
        steps.insert(3, ("smote", smote))
        return ImbPipeline(steps=steps)
    return Pipeline(steps=steps)


def run_classification(
    features_csv: Path,
    output_dir: Path,
    use_smote: bool = True,
    n_pca: int = 40,
) -> pd.DataFrame:
    df  = pd.read_csv(str(features_csv))
    meta = {"subject_id", "group", "BDI"}
    fcols = [c for c in df.columns if c not in meta]

    # Drop feature columns where >10% of subjects have NaN
    # (caused by a minority of subjects having more EEG channels than the standard 71)
    nan_frac = df[fcols].isnull().mean()
    fcols = [c for c in fcols if nan_frac[c] <= 0.10]
    if len(fcols) < len([c for c in df.columns if c not in meta]):
        log.info("Dropped %d feature columns with >10%% NaN; using %d clean features.",
                 len([c for c in df.columns if c not in meta]) - len(fcols), len(fcols))

    X = df[fcols].values.astype(float)
    y = df["group"].values.astype(int)

    log.info("Classification: %d subjects, %d features  (CTL=%d, MDD=%d)",
             X.shape[0], X.shape[1], (y == 0).sum(), (y == 1).sum())

    # Adapt PCA components to training fold size (N * (k-1) / k samples per fold)
    n_train_fold = int(X.shape[0] * (CV_FOLDS - 1) / CV_FOLDS) - 1
    n_pca_safe = min(n_pca, n_train_fold)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    clfs       = get_classifiers()
    all_results = []

    for clf_name, clf in clfs.items():
        pipe   = build_pipeline(clf, use_smote=use_smote, n_pca=n_pca_safe)
        y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict", n_jobs=N_JOBS)
        try:
            y_prob = cross_val_predict(pipe, X, y, cv=cv,
                                        method="predict_proba", n_jobs=N_JOBS)[:, 1]
        except Exception:
            y_prob = y_pred.astype(float)

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        res = {
            "classifier":        clf_name,
            "accuracy":          accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "roc_auc":           roc_auc_score(y, y_prob),
            "f1":                f1_score(y, y_pred, pos_label=1),
            "sensitivity":       tp / max(tp + fn, 1),
            "specificity":       tn / max(tn + fp, 1),
            "y_prob":            y_prob,
        }
        all_results.append(res)
        log.info("%-10s  AUC=%.3f  BalAcc=%.3f  F1=%.3f  Sens=%.3f  Spec=%.3f",
                 clf_name, res["roc_auc"], res["balanced_accuracy"],
                 res["f1"], res["sensitivity"], res["specificity"])

    scalar_keys = [k for k in all_results[0] if k != "y_prob"]
    results_df  = pd.DataFrame([{k: r[k] for k in scalar_keys} for r in all_results])
    results_csv = output_dir / "meg_classification_results.csv"
    results_df.to_csv(str(results_csv), index=False)
    log.info("Results saved → %s", results_csv)

    # ── ROC curves ────────────────────────────────────────────────────────────
    figs_dir = output_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap    = plt.cm.get_cmap("tab10", len(all_results))
    for i, res in enumerate(all_results):
        fpr, tpr, _ = roc_curve(y, res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=cmap(i),
                label=f"{res['classifier']}  (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – MEG/EEG MDD vs CTL (ds005356)\n5-fold Stratified CV")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(figs_dir / "meg_roc_curves.png"), dpi=150)
    plt.close(fig)

    # ── Confusion matrices ────────────────────────────────────────────────────
    n = len(all_results)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i, res in enumerate(all_results):
        cm = confusion_matrix(y, res["y_prob"] > 0.5, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=["CTL", "MDD"], yticklabels=["CTL", "MDD"],
                    ax=axes[i], vmin=0, vmax=1, cbar=False)
        axes[i].set_title(f"{res['classifier']}\nAUC={res['roc_auc']:.2f}")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(figs_dir / "meg_confusion_matrices.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7: COMPARATIVE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def compare_with_ds003474(
    meg_results: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate comparison table and bar chart vs. ds003474 EEG results.
    ds003474 best results are hardcoded from INTEGRATED_DYNAMICS_REPORT.md.
    """
    # ds003474 reference results (from INTEGRATED_DYNAMICS_REPORT.md, April 2026)
    eeg_ref = pd.DataFrame([
        {"pipeline": "Baseline spectral+entropy",   "roc_auc": 0.675, "balanced_accuracy": 0.64},
        {"pipeline": "Baseline+Connectivity (PLV)",  "roc_auc": 0.807, "balanced_accuracy": 0.75},
        {"pipeline": "Microstates+Baseline",          "roc_auc": 0.825, "balanced_accuracy": 0.77},
        {"pipeline": "Boruta confirmed (n=5)",         "roc_auc": 0.788, "balanced_accuracy": 0.74},
        {"pipeline": "DDS+Info+Baseline",              "roc_auc": 0.836, "balanced_accuracy": 0.78},
    ])
    eeg_ref["dataset"] = "ds003474 (EEG, PST-resting)"

    meg_best = meg_results.sort_values("roc_auc", ascending=False).iloc[0]
    meg_comp = pd.DataFrame([
        {
            "pipeline":          f"MEG-concurrent EEG / {meg_best['classifier']}",
            "roc_auc":           round(meg_best["roc_auc"], 3),
            "balanced_accuracy": round(meg_best["balanced_accuracy"], 3),
            "dataset":           "ds005356 (MEG+EEG, PST-task)",
        }
    ])

    comp_df = pd.concat([eeg_ref, meg_comp], ignore_index=True)
    comp_df["improvement_vs_eeg_baseline"] = comp_df["roc_auc"] - 0.675
    comp_csv = output_dir / "comparison_ds003474_vs_ds005356.csv"
    comp_df.to_csv(str(comp_csv), index=False)
    log.info("Comparison table saved → %s", comp_csv)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#2196F3" if "ds003474" in r else "#E91E63"
               for r in comp_df["dataset"]]
    bars = ax.barh(
        range(len(comp_df)),
        comp_df["roc_auc"],
        color=colors, edgecolor="black", linewidth=0.5,
    )
    ax.set_yticks(range(len(comp_df)))
    ax.set_yticklabels(comp_df["pipeline"], fontsize=9)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("MDD vs CTL Classification AUC\nds003474 (EEG) vs ds005356 (MEG+EEG)")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.axvline(0.675, color="#2196F3", linestyle=":", linewidth=1.5,
               label="EEG baseline (0.675)")
    ax.set_xlim(0.3, 1.0)

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2196F3", label="ds003474 (EEG)"),
        Patch(facecolor="#E91E63", label="ds005356 (MEG+EEG)"),
    ], loc="lower right")
    fig.tight_layout()
    fig.savefig(str(output_dir / "figures" / "comparison_auc_barplot.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("COMPARATIVE RESULTS: ds003474 vs ds005356")
    print("=" * 60)
    print(comp_df[["pipeline", "roc_auc", "balanced_accuracy", "dataset"]].to_string(index=False))

    return comp_df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="MEG depression classification pipeline (ds005356 → ds003474 comparison)"
    )
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip EEG extraction, use cached features CSV")
    parser.add_argument("--epoch_mode", action="store_true",
                        help="Use peri-feedback epochs instead of continuous windows")
    parser.add_argument("--max_subjects", type=int, default=0,
                        help="Limit to N subjects (0 = all, useful for debugging)")
    parser.add_argument("--output_dir", type=str,
                        default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--no_smote", action="store_true",
                        help="Disable SMOTE oversampling")
    parser.add_argument("--entropy", action="store_true",
                        help="Add spectral entropy per channel (adds ~10%% time, "
                             "not needed for Boruta-confirmed features)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    features_csv = out_dir / "meg_features.csv"

    # ── Extract features ──────────────────────────────────────────────────────
    if not args.skip_extract:
        log.info("Loading group labels from Excel …")
        labels_df = load_group_labels(EXCEL_FILE)

        log.info("Scanning BIDS root for FIF files …")
        fif_map = find_fif_files(BIDS_ROOT)

        log.info("Starting feature extraction (epoch_mode=%s, entropy=%s) …",
                 args.epoch_mode, args.entropy)
        feat_df = run_extraction(
            fif_map, labels_df, features_csv,
            epoch_mode=args.epoch_mode,
            max_subjects=args.max_subjects,
            compute_entropy=args.entropy,
        )
    else:
        if not features_csv.exists():
            raise FileNotFoundError(
                f"Features CSV not found: {features_csv}\n"
                "Run without --skip_extract first."
            )
        feat_df = pd.read_csv(str(features_csv))
        log.info("Loaded cached features: %d subjects", len(feat_df))

    if len(feat_df) < 10:
        log.error("Too few subjects (%d) – aborting classification.", len(feat_df))
        return

    # ── Classify ──────────────────────────────────────────────────────────────
    log.info("Running classification …")
    results_df = run_classification(
        features_csv, out_dir,
        use_smote=not args.no_smote,
        n_pca=40,
    )

    # ── Compare with ds003474 ─────────────────────────────────────────────────
    compare_with_ds003474(results_df, out_dir)

    log.info("Pipeline complete. Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
