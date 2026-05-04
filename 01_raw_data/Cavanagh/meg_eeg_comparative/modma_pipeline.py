"""
modma_pipeline.py — DDS+Info+Baseline pipeline for the MODMA dataset
=====================================================================
Dataset: EEG_128channels_resting_lanzhou_2015 (resting, .mat)
         EEG_128channels_ERP_lanzhou_2015    (dot-probe task, .raw EGI)

System   : EGI HydroCel GSN-128, 128 channels, 250 Hz
Groups   : MDD=24, HC=29 — clinical diagnosis (explicit label, no threshold)
Scale    : PHQ-9 (Patient Health Questionnaire-9)
Notch    : 50 Hz (China power line)

Usage:
    python modma_pipeline.py              # resting mode (default)
    python modma_pipeline.py --mode erp   # ERP continuous task mode
    python modma_pipeline.py --max_subjects 5   # quick test
"""

import sys, os, logging, argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as ss
from scipy.optimize import curve_fit

import mne
mne.set_log_level("WARNING")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── optional info-theory deps ──────────────────────────────────────────────────
try:
    import antropy
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False

try:
    from idtxl.multivariate_te import MultivariateTE
    from idtxl.data import Data as IDTxlData
    HAS_IDTXL = True
except ImportError:
    HAS_IDTXL = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.base import clone

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
MODMA_ROOT   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/MODMA")
REST_DIR     = MODMA_ROOT / "EEG_128channels_resting_lanzhou_2015"
ERP_DIR      = MODMA_ROOT / "EEG_128channels_ERP_lanzhou_2015"
META_XLSX    = REST_DIR   / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"
OUT_DIR      = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/meg_eeg_comparative")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SFREQ_TARGET  = 250          # Hz — already 250 Hz, no resampling needed
L_FREQ        = 1.0
H_FREQ        = 45.0
NOTCH_FREQ    = 50.0         # China power line

DDS_WIN_MS    = 400          # ms — window for DDS fitting
DDS_WIN_N     = int(DDS_WIN_MS * SFREQ_TARGET / 1000)   # 100 samples
DDS_MAX_WINS  = 100          # windows per ROI

WELCH_WIN_S   = 4.0          # s
WELCH_WIN_N   = int(WELCH_WIN_S * SFREQ_TARGET)

CV_FOLDS      = 5
N_PCA         = 40
SEED          = 42

# ── GSN-HydroCel-128 → 10-20 mapping (closest channels, verified by distance) ─
# Format: '10-20 name': 'EGI channel'
GSN128_ROI = {
    # Frontal ROI (bilateral)
    "frontal": ["E20",  # F3
                "E118", # F4
                "E23",  # AF3
                "E3",   # AF4
                "E22",  # Fp1
                "E9",   # Fp2
                "E29",  # FC3
                "E111"],# FC4
    # cACC (supplementary motor / anterior cingulate area)
    "cACC":   ["E11",  # AFz
               "E5",   # Fz
               "E111"],# FC4 (proxy for FC2)
    # Left hemisphere fronto-lateral
    "LH":     ["E20",  # F3
               "E27",  # F5
               "E29",  # FC3
               "E35"], # FC5
    # Right hemisphere fronto-lateral
    "RH":     ["E118", # F4
               "E123", # F6
               "E111", # FC4
               "E110"],# FC6
}

# E129 is Cz (vertex reference in EGI) — kept as EEG electrode
# E8 is used as EOG proxy (frontal polar in GSN-128)

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. METADATA
# ══════════════════════════════════════════════════════════════════════════════
def load_metadata():
    """Return DataFrame with subject_id, group (0=HC, 1=MDD), PHQ-9 and other scales."""
    df = pd.read_excel(META_XLSX, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    # Rename for convenience
    rename = {
        df.columns[0]: "subject_id",
        df.columns[1]: "type",
        df.columns[2]: "age",
        df.columns[3]: "gender",
        df.columns[4]: "education",
        df.columns[5]: "PHQ9",
        df.columns[6]: "CTQ_SF",
        df.columns[7]: "LES",
        df.columns[8]: "SSRS",
        df.columns[9]: "GAD7",
        df.columns[10]: "PSQI",
    }
    df = df.rename(columns=rename)
    df = df[df["subject_id"].notna() & df["type"].isin(["MDD", "HC"])].copy()
    df["group"]      = (df["type"] == "MDD").astype(int)
    # Excel stores IDs as integers (e.g. 2010002); pad to 8 digits to match filenames
    df["subject_id"] = df["subject_id"].astype(str).str.strip().str.zfill(8)
    log.info("Metadata: %d subjects  HC=%d  MDD=%d",
             len(df), (df.group==0).sum(), (df.group==1).sum())
    return df.set_index("subject_id")


def find_files(mode: str):
    """Return {subject_id: Path} for resting .mat or ERP .raw files."""
    file_map = {}
    if mode == "rest":
        for f in REST_DIR.glob("*.mat"):
            # Filename starts with subject_id (8 digits)
            sub_id = f.name[:8]
            if sub_id.isdigit():
                file_map[sub_id] = f
    else:  # erp
        for f in ERP_DIR.glob("*.raw"):
            sub_id = f.name[:8].rstrip("_")
            if sub_id.isdigit():
                file_map[sub_id] = f
    log.info("Found %d %s files.", len(file_map), mode)
    return file_map


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_resting_mat(path: Path):
    """Load a MODMA resting .mat file → MNE Raw (E1-E128 + E129)."""
    mat = sio.loadmat(str(path))
    data_keys = [k for k in mat if not k.startswith("_")
                 and k not in ("samplingRate", "Impedances_0")]
    if not data_keys:
        raise ValueError(f"No data key found in {path.name}")
    data = mat[data_keys[0]]          # shape (129, n_samples)
    sfreq = float(mat["samplingRate"].flat[0])

    # Channel names: E1..E129 (E129 = Cz vertex reference in EGI nets)
    ch_names = [f"E{i+1}" for i in range(data.shape[0])]
    ch_types = ["eeg"] * data.shape[0]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # mat data in µV → convert to V
    return raw


def load_erp_raw(path: Path):
    """Load a MODMA ERP .raw (EGI) file → MNE Raw."""
    raw = mne.io.read_raw_egi(str(path), preload=True, verbose=False)
    # Keep only EEG channels (drop stim channels for preprocessing)
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_modma(raw: mne.io.BaseRaw, sub_id: str, mode: str):
    """
    Standard preprocessing for MODMA EEG:
    1. Set GSN-HydroCel-128 montage
    2. Drop non-EEG channels (stim/misc in ERP mode)
    3. Filter 1-45 Hz + notch 50 Hz
    4. Detect and interpolate bad channels (>4× median std)
    5. Average reference
    6. ICA (20 components, ICLabel)
    Returns preprocessed Raw.
    """
    raw = raw.copy().load_data(verbose=False)

    # Drop non-EEG channels in ERP mode
    if mode == "erp":
        non_eeg = [c for c in raw.ch_names
                   if raw.get_channel_types([c])[0] not in ("eeg",)]
        if non_eeg:
            raw.drop_channels(non_eeg)

    # Set montage
    try:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    except Exception as e:
        log.warning("[%s] Montage error: %s", sub_id, e)

    # Mark channels with no position (NaN locs) as misc so they skip interpolation
    for ch_dict in raw.info["chs"]:
        if ch_dict["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH:
            if np.isnan(ch_dict["loc"][:3]).any():
                raw.set_channel_types({ch_dict["ch_name"]: "misc"}, verbose=False)

    log.info("[%s]  Loaded: %d EEG ch, %.0f s", sub_id,
             len(mne.pick_types(raw.info, eeg=True)), raw.times[-1])

    # Filter
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    log.info("[%s]  Filtered [%.1f–%.1f Hz], notch @ %.0f Hz",
             sub_id, L_FREQ, H_FREQ, NOTCH_FREQ)

    # Bad channel detection: channels with std > 4× median
    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude=[])
    if len(eeg_picks) == 0:
        raise ValueError(f"[{sub_id}] No EEG channels after type filtering.")
    data_eeg = raw.get_data(picks=eeg_picks)
    data_std  = data_eeg.std(axis=1)
    median_std = np.median(data_std)
    eeg_names  = [raw.ch_names[i] for i in eeg_picks]
    bads = [eeg_names[i] for i, s in enumerate(data_std) if s > 4 * median_std]
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True, verbose=False)
    log.info("[%s]  Bad channels: %d interpolated", sub_id, len(bads))

    # Average reference
    raw.set_eeg_reference("average", verbose=False)

    # ICA
    log.info("[%s]  Running ICA…", sub_id)
    ica = mne.preprocessing.ICA(n_components=20, random_state=SEED,
                                  max_iter="auto", verbose=False)
    raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(raw_hp, verbose=False)

    # EOG detection via E8 (frontal polar in GSN-128 ≈ Fp1/Fp2)
    n_excluded = 0
    eog_proxy = "E8" if "E8" in raw.ch_names else None
    if eog_proxy:
        try:
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=eog_proxy,
                                            threshold=2.5, verbose=False)
            ica.exclude = eog_idx
            n_excluded  = len(eog_idx)
        except Exception:
            pass

    # ICLabel fallback if mne-icalabel available and EOG found nothing
    if n_excluded == 0:
        try:
            from mne_icalabel import label_components
            raw_tmp = raw.copy().filter(1.0, 100.0, verbose=False)
            ica_tmp = mne.preprocessing.ICA(n_components=20, random_state=SEED,
                                             max_iter="auto", verbose=False)
            ica_tmp.fit(raw_tmp, verbose=False)
            labels = label_components(raw_tmp, ica_tmp, method="iclabel")
            bad_types = {"muscle artifact", "eye blink", "heart beat", "other"}
            excl = [i for i, (lbl, prob) in enumerate(
                        zip(labels["labels"], labels["y_pred_proba"]))
                    if lbl in bad_types and prob > 0.70]
            ica.exclude = excl
            n_excluded  = len(excl)
        except Exception:
            pass

    ica.apply(raw, verbose=False)
    log.info("[%s]  ICA: %d components removed", sub_id, n_excluded)

    return raw, len(bads), n_excluded


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. Baseline spectral (Welch + Hjorth) ────────────────────────────────────
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13),
         ("beta",  13, 30), ("gamma", 30, 45)]

def extract_baseline_spectral(raw: mne.io.BaseRaw):
    """
    Vectorised Welch PSD over sliding windows → per-channel band power + Hjorth.
    Returns dict of feature_name → mean-across-windows.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=eeg_picks)       # (n_ch, n_times)
    sfreq = raw.info["sfreq"]
    ch_names = [raw.ch_names[i] for i in eeg_picks]

    step = WELCH_WIN_N // 2
    n_ch, n_t = data.shape
    starts = range(0, n_t - WELCH_WIN_N + 1, step)

    band_acc = {f"{ch}_{b}": [] for ch in ch_names for b, _, _ in BANDS}
    hjorth_acc = {f"{ch}_hjorth_{h}": []
                  for ch in ch_names for h in ("mob", "comp")}

    for s in starts:
        seg = data[:, s:s + WELCH_WIN_N]
        freqs, psd = ss.welch(seg, fs=sfreq, nperseg=WELCH_WIN_N, axis=1)
        df_freq = freqs[1] - freqs[0]
        for bi, (bname, fl, fh) in enumerate(BANDS):
            idx = np.where((freqs >= fl) & (freqs < fh))[0]
            bp = psd[:, idx].sum(axis=1) * df_freq
            for ci, ch in enumerate(ch_names):
                band_acc[f"{ch}_{bname}"].append(bp[ci])

        # Hjorth over segment
        d1 = np.diff(seg, axis=1)
        d2 = np.diff(d1, axis=1)
        var0 = seg.var(axis=1) + 1e-30
        var1 = d1.var(axis=1) + 1e-30
        var2 = d2.var(axis=1) + 1e-30
        mob  = np.sqrt(var1 / var0)
        comp = np.sqrt(var2 / var1) / (mob + 1e-30)
        for ci, ch in enumerate(ch_names):
            hjorth_acc[f"{ch}_hjorth_mob"].append(mob[ci])
            hjorth_acc[f"{ch}_hjorth_comp"].append(comp[ci])

    feats = {}
    for k, vals in {**band_acc, **hjorth_acc}.items():
        feats[k] = float(np.mean(vals)) if vals else np.nan

    n_wins = len(list(starts))
    return feats, n_wins


# ── 4b. DDS (Dual Damped Sine) ────────────────────────────────────────────────
def _dds_model(t, A1, alpha1, f1, phi1, A2, alpha2, f2, phi2):
    return (A1 * np.exp(-alpha1 * t) * np.cos(2*np.pi*f1*t + phi1) +
            A2 * np.exp(-alpha2 * t) * np.cos(2*np.pi*f2*t + phi2))

_DDS_PARAMS = ("A1","alpha1","f1","phi1","A2","alpha2","f2","phi2")

def _fit_dds_window(segment: np.ndarray, sfreq: float):
    """Fit DDS to one window. Returns dict of params or None."""
    t = np.arange(len(segment)) / sfreq
    y = segment - segment.mean()
    amp = float(np.std(y)) * 1.4
    if amp < 1e-30:
        return None
    p0 = [amp, 5.0, 10.0, 0.0, amp * 0.5, 8.0, 25.0, 0.0]
    bounds = ([0,0,1,-np.pi, 0,0,1,-np.pi],
              [amp*10,100,45,np.pi, amp*10,100,45,np.pi])
    try:
        popt, _ = curve_fit(_dds_model, t, y, p0=p0, bounds=bounds,
                            method="trf", max_nfev=1200)
        residual = y - _dds_model(t, *popt)
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((y - y.mean())**2)) + 1e-30
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        return {p: float(v) for p, v in zip(_DDS_PARAMS, popt)} | {"r2": r2, "residual": residual}
    except Exception:
        return None


def extract_dds_features(raw: mne.io.BaseRaw, rng=None):
    """
    Fit DDS on DDS_MAX_WINS random windows per ROI.
    Returns dict of feature_name → scalar + residuals array per ROI.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data      = raw.get_data(picks=eeg_picks)
    ch_names  = [raw.ch_names[i] for i in eeg_picks]
    sfreq     = raw.info["sfreq"]
    n_times   = data.shape[1]

    feats = {}
    residuals_by_roi = {}
    n_ok_total = 0

    for roi_name, roi_chs in GSN128_ROI.items():
        avail = [c for c in roi_chs if c in ch_names]
        if not avail:
            log.warning("  ROI %s: no channels available.", roi_name)
            continue

        ch_idx = [ch_names.index(c) for c in avail]
        roi_signal = data[ch_idx, :].mean(axis=0)   # average across ROI

        max_start = n_times - DDS_WIN_N
        if max_start <= 0:
            log.warning("  ROI %s: signal too short for DDS windows.", roi_name)
            continue

        starts = rng.integers(0, max_start, size=DDS_MAX_WINS * 3)
        results, residuals = [], []
        for s in starts:
            if len(results) >= DDS_MAX_WINS:
                break
            seg = roi_signal[s:s + DDS_WIN_N]
            r   = _fit_dds_window(seg, sfreq)
            if r is not None:
                residuals.append(r.pop("residual"))
                results.append(r)

        n_ok_total += len(results)
        if not results:
            continue

        residuals_by_roi[roi_name] = np.array(residuals)   # (n_ok, win_N)

        # Aggregate: mean ± std of each DDS parameter
        for p in _DDS_PARAMS + ("r2",):
            vals = [res[p] for res in results]
            feats[f"dds_{roi_name}_{p}_mean"] = float(np.mean(vals))
            feats[f"dds_{roi_name}_{p}_std"]  = float(np.std(vals))

    log.info("  DDS: %d successful windows across ROIs.", n_ok_total)
    return feats, residuals_by_roi


# ── 4c. Information-theory (AIS, TE, PID) ─────────────────────────────────────
def _ais(x: np.ndarray, k: int = 1, bins: int = 16) -> float:
    """Approximate AIS via discretised MI(X_{t-k}, X_t)."""
    try:
        x_q = np.digitize(x, np.linspace(x.min(), x.max() + 1e-9, bins + 1)) - 1
        past, pres = x_q[:-k], x_q[k:]
        joint_hist, _, _ = np.histogram2d(past, pres,
                                          bins=[bins, bins],
                                          range=[[0, bins-1],[0, bins-1]])
        joint_hist = joint_hist / (joint_hist.sum() + 1e-30)
        px = joint_hist.sum(axis=1, keepdims=True) + 1e-30
        py = joint_hist.sum(axis=0, keepdims=True) + 1e-30
        mi = float(np.sum(joint_hist * np.log2(joint_hist / (px * py) + 1e-30)))
        return max(0.0, mi)
    except Exception:
        return np.nan


def _te(x: np.ndarray, y: np.ndarray, k: int = 1, bins: int = 12) -> float:
    """Approximate TE(X→Y) via discretised conditional MI."""
    try:
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max()) + 1e-9
        edges = np.linspace(lo, hi, bins + 1)
        xq = np.digitize(x, edges) - 1
        yq = np.digitize(y, edges) - 1
        # TE(X→Y) = MI(X_past; Y_pres | Y_past)
        # = H(Y_pres | Y_past) - H(Y_pres | X_past, Y_past)
        yp, yf = yq[:-k], yq[k:]
        xp      = xq[:-k]

        def _cond_ent(a, b):
            ab, _, _ = np.histogram2d(a, b, bins=[bins, bins],
                                      range=[[0,bins-1],[0,bins-1]])
            ab = ab / (ab.sum() + 1e-30)
            pb = ab.sum(axis=0, keepdims=True) + 1e-30
            ca_b = ab / pb
            return -float(np.sum(ab * np.log2(ca_b + 1e-30)))

        def _cond_ent3(a, b, c):
            n = len(a)
            joint_idx = b * bins + c
            result = 0.0
            for bc_val in np.unique(joint_idx):
                mask = joint_idx == bc_val
                p_bc = mask.mean()
                if p_bc == 0:
                    continue
                a_sub = a[mask]
                h, _ = np.histogram(a_sub, bins=bins, range=(0, bins-1))
                h = h / (h.sum() + 1e-30)
                result += p_bc * (-np.sum(h * np.log2(h + 1e-30)))
            return result

        h_y_given_ypast  = _cond_ent(yp, yf)
        h_y_given_xypast = _cond_ent3(yf, xp, yp)
        return max(0.0, h_y_given_ypast - h_y_given_xypast)
    except Exception:
        return np.nan


def _pid_approx(s1: np.ndarray, s2: np.ndarray, t: np.ndarray,
                bins: int = 10) -> dict:
    """
    Approximate PID (redundancy/unique_s1/unique_s2/synergy) using
    I_min definition (Williams & Beer 2010, discretised).
    """
    try:
        lo = min(s1.min(), s2.min(), t.min())
        hi = max(s1.max(), s2.max(), t.max()) + 1e-9
        edges = np.linspace(lo, hi, bins + 1)
        s1q = np.digitize(s1, edges) - 1
        s2q = np.digitize(s2, edges) - 1
        tq  = np.digitize(t,  edges) - 1

        def _mi2(a, b):
            h, _, _ = np.histogram2d(a, b, bins=[bins, bins],
                                     range=[[0,bins-1],[0,bins-1]])
            h = h / (h.sum() + 1e-30)
            pa = h.sum(axis=1, keepdims=True) + 1e-30
            pb = h.sum(axis=0, keepdims=True) + 1e-30
            return float(np.sum(h * np.log2(h / (pa * pb) + 1e-30)))

        i1  = max(0.0, _mi2(s1q, tq))
        i2  = max(0.0, _mi2(s2q, tq))
        # I_min (redundancy) ≈ min(I(S1;T), I(S2;T))
        redundancy = min(i1, i2)
        unique_s1  = max(0.0, i1 - redundancy)
        unique_s2  = max(0.0, i2 - redundancy)

        # 3-way joint MI
        n = len(s1q)
        idx = s1q * bins**2 + s2q * bins + tq
        counts = np.bincount(idx, minlength=bins**3).astype(float)
        p3 = counts / (counts.sum() + 1e-30)
        # Synergy ≈ I(S1,S2;T) - max(I(S1;T), I(S2;T))
        p_s1s2 = p3.reshape(bins**2, bins).sum(axis=1, keepdims=True)
        p_t    = p3.reshape(bins**2, bins).sum(axis=0, keepdims=True)
        joint_s1s2_t = p3.reshape(bins**2, bins) / (p3.sum() + 1e-30)
        i12t = float(np.sum(joint_s1s2_t * np.log2(
            joint_s1s2_t / ((p_s1s2 * p_t / (p3.sum()+1e-30)) + 1e-30) + 1e-30)))
        synergy = max(0.0, i12t - max(i1, i2))

        return {"redundancy": redundancy, "unique_s1": unique_s1,
                "unique_s2": unique_s2,   "synergy":   synergy}
    except Exception:
        return {"redundancy": np.nan, "unique_s1": np.nan,
                "unique_s2": np.nan,  "synergy": np.nan}


def extract_info_features(residuals_by_roi: dict) -> dict:
    """
    Compute AIS, TE (LH→frontal, RH→frontal, LH↔RH), and PID (LH,RH→frontal)
    on the mean DDS residual time-series per ROI.
    """
    feats = {}

    # Mean residual series per ROI (concatenate windows)
    series = {}
    for roi, res_arr in residuals_by_roi.items():
        series[roi] = res_arr.ravel()  # flatten all windows

    # AIS per ROI
    for roi, sig in series.items():
        feats[f"info_AIS_{roi}"] = _ais(sig)

    # TE pairs
    pairs_te = [("LH", "frontal"), ("RH", "frontal"),
                ("LH", "RH"),      ("cACC", "frontal")]
    for src, tgt in pairs_te:
        if src in series and tgt in series:
            n = min(len(series[src]), len(series[tgt]))
            feats[f"info_TE_{src}_to_{tgt}"] = _te(series[src][:n], series[tgt][:n])

    # PID: LH + RH → frontal
    if "LH" in series and "RH" in series and "frontal" in series:
        n = min(len(series["LH"]), len(series["RH"]), len(series["frontal"]))
        pid = _pid_approx(series["LH"][:n], series["RH"][:n], series["frontal"][:n])
        for k, v in pid.items():
            feats[f"info_PID_{k}"] = v

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN EXTRACTION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_extraction(meta_df: pd.DataFrame, file_map: dict, mode: str,
                   output_csv: Path, max_subjects: int = None):
    """Load, preprocess, extract features for all subjects. Incrementally cached."""
    # Incremental caching
    already_done = set()
    existing_records = []
    if output_csv.exists():
        cached = pd.read_csv(str(output_csv), dtype={"subject_id": str})
        # Normalize IDs to 8-digit zero-padded strings to match meta_df.index
        cached["subject_id"] = cached["subject_id"].str.zfill(8)
        n_before = len(cached)
        cached = cached.drop_duplicates(subset="subject_id", keep="last")
        if len(cached) < n_before:
            log.warning("Dropped %d duplicate rows from cached CSV.", n_before - len(cached))
        cached.to_csv(str(output_csv), index=False)   # always re-save with normalized IDs
        already_done = set(cached["subject_id"].tolist())  # 8-digit strings, match meta_df.index
        existing_records = cached.to_dict("records")
        log.info("Resuming: %d subjects already cached.", len(already_done))

    records = list(existing_records)
    subjects = sorted(set(meta_df.index) & set(file_map.keys()))
    if max_subjects:
        subjects = subjects[:max_subjects]

    for sub_id in subjects:
        if sub_id in already_done:
            log.info("  [%s] Cached — skipping.", sub_id)
            continue

        row_meta = meta_df.loc[sub_id]
        group  = int(row_meta["group"])
        phq9   = float(row_meta.get("PHQ9", np.nan))
        label  = row_meta["type"]

        log.info("Processing %s (%s, PHQ-9=%s) …", sub_id, label, phq9)

        try:
            # Load
            if mode == "rest":
                raw = load_resting_mat(file_map[sub_id])
            else:
                raw = load_erp_raw(file_map[sub_id])

            # Preprocess
            raw_clean, n_bad, n_ica = preprocess_modma(raw, sub_id, mode)

            # Baseline spectral
            baseline_feats, n_wins = extract_baseline_spectral(raw_clean)
            log.info("  Baseline: %d windows, %d features", n_wins, len(baseline_feats))

            # DDS
            dds_feats, residuals_by_roi = extract_dds_features(raw_clean)

            # Info
            info_feats = extract_info_features(residuals_by_roi) if residuals_by_roi else {}

            record = {
                "subject_id": sub_id,
                "group":      group,
                "label":      label,
                "PHQ9":       phq9,
                "age":        float(row_meta.get("age", np.nan)),
                "gender":     str(row_meta.get("gender", "")),
                "n_bad_ch":   n_bad,
                "n_ica_removed": n_ica,
                **baseline_feats,
                **dds_feats,
                **info_feats,
            }
            records.append(record)
            log.info("  [%s] Done. group=%s", sub_id, label)

        except Exception as e:
            log.error("  [%s] FAILED: %s", sub_id, e)
            import traceback; traceback.print_exc()
            continue

        # Save incrementally (dedup guard in case of prior partial runs)
        pd.DataFrame(records).drop_duplicates(subset="subject_id", keep="last").to_csv(
            str(output_csv), index=False)

    df = pd.DataFrame(records).drop_duplicates(subset="subject_id", keep="last").reset_index(drop=True)
    log.info("Extracted: %d subjects × %d features", len(df), df.shape[1])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def get_classifiers():
    clfs = {
        "LogReg":  LogisticRegression(max_iter=2000, random_state=SEED, C=0.1),
        "LDA":     LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "SVM-Lin": SVC(kernel="linear", probability=True, random_state=SEED, C=0.5),
        "SVM-RBF": SVC(kernel="rbf",    probability=True, random_state=SEED),
        "RF":      RandomForestClassifier(n_estimators=200, random_state=SEED,
                                          class_weight="balanced"),
        "MLP":     MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                 random_state=SEED),
    }
    if HAS_XGB:
        clfs["XGB"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=SEED, verbosity=0,
            use_label_encoder=False)
    return clfs


def build_pipeline(clf, n_features: int, n_subjects: int):
    n_train = int(n_subjects * (CV_FOLDS - 1) / CV_FOLDS) - 1
    n_pca   = min(N_PCA, n_train, n_features)
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("var",     VarianceThreshold(threshold=1e-6)),
        ("pca",     PCA(n_components=n_pca, random_state=SEED)),
        ("clf",     clf),
    ]
    return Pipeline(steps)


def run_classification(df: pd.DataFrame, mode_label: str):
    """
    Classify on 5 feature sets:
      1. Baseline spectral
      2. DDS only
      3. Info only
      4. DDS + Info
      5. DDS + Info + Baseline
    """
    meta_cols = {"subject_id", "group", "label", "PHQ9", "age", "gender",
                 "n_bad_ch", "n_ica_removed"}
    all_feat_cols = [c for c in df.columns if c not in meta_cols]

    baseline_cols = [c for c in all_feat_cols if not c.startswith(("dds_", "info_"))]
    dds_cols      = [c for c in all_feat_cols if c.startswith("dds_")]
    info_cols     = [c for c in all_feat_cols if c.startswith("info_")]

    # Drop >10% NaN columns
    def clean_cols(cols):
        nan_frac = df[cols].isnull().mean()
        return [c for c in cols if nan_frac[c] <= 0.10]

    baseline_cols = clean_cols(baseline_cols)
    dds_cols      = clean_cols(dds_cols)
    info_cols     = clean_cols(info_cols)

    feature_sets = {
        "Baseline spectral": baseline_cols,
        "DDS only":          dds_cols,
        "Info only":         info_cols,
        "DDS+Info":          dds_cols + info_cols,
        "DDS+Info+Baseline": dds_cols + info_cols + baseline_cols,
    }

    labels   = df["group"].values
    n_mdd    = int(labels.sum())
    n_ctl    = int((labels == 0).sum())
    N        = len(labels)
    clfs     = get_classifiers()
    cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    results = []
    for fs_name, fcols in feature_sets.items():
        if not fcols:
            log.warning("  [%s] No features — skipping.", fs_name)
            continue
        X = df[fcols].values.astype(float)
        log.info("  [%s] %d subjects, %d features (HC=%d, MDD=%d)",
                 fs_name, N, len(fcols), n_ctl, n_mdd)

        for clf_name, clf_obj in clfs.items():
            aucs, bals = [], []
            for fold, (tr, te) in enumerate(cv.split(X, labels)):
                X_tr, y_tr = X[tr], labels[tr]
                X_te, y_te = X[te], labels[te]
                try:
                    if HAS_SMOTE and len(set(y_tr)) > 1 and y_tr.sum() >= 2:
                        k_sm = min(5, y_tr.sum() - 1)
                        sm = SMOTE(k_neighbors=k_sm, random_state=SEED)
                        X_tr, y_tr = sm.fit_resample(
                            SimpleImputer(strategy="median").fit_transform(X_tr), y_tr)
                    pipe = build_pipeline(clone(clf_obj), X_tr.shape[1], len(y_tr))
                    pipe.fit(X_tr, y_tr)
                    prob = pipe.predict_proba(X_te)[:, 1]
                    aucs.append(roc_auc_score(y_te, prob))
                    bals.append(balanced_accuracy_score(y_te, pipe.predict(X_te)))
                except Exception as e:
                    log.debug("  fold %d %s/%s: %s", fold, fs_name, clf_name, e)
                    continue

            if aucs:
                auc = float(np.mean(aucs))
                bal = float(np.mean(bals))
                log.info("    %-30s AUC=%.3f  BalAcc=%.3f", clf_name, auc, bal)
                results.append({
                    "dataset":    "MODMA",
                    "mode":       mode_label,
                    "feature_set": fs_name,
                    "classifier": clf_name,
                    "roc_auc":    auc,
                    "roc_auc_std": float(np.std(aucs)),
                    "bal_acc":    bal,
                    "N":          N,
                    "n_HC":       n_ctl,
                    "n_MDD":      n_mdd,
                    "n_features": len(fcols),
                })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# 7. PHQ-9 CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_phq9_correlation(df: pd.DataFrame, results_df: pd.DataFrame,
                         mode_label: str, out_dir: Path):
    """Spearman correlation between PHQ-9 and top DDS/Info features (MDD+HC combined)."""
    from scipy.stats import spearmanr
    if "PHQ9" not in df.columns or df["PHQ9"].isna().all():
        return

    dds_info_cols = [c for c in df.columns
                     if c.startswith(("dds_", "info_")) and df[c].notna().mean() > 0.9]
    if not dds_info_cols:
        return

    phq9 = df["PHQ9"].values.astype(float)
    rows = []
    for col in dds_info_cols:
        x = df[col].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(phq9))
        if mask.sum() < 10:
            continue
        rho, pval = spearmanr(x[mask], phq9[mask])
        rows.append({"feature": col, "spearman_rho": rho, "pval": pval})

    if not rows:
        return

    corr_df = pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)
    out_path = out_dir / f"modma_{mode_label}_phq9_correlations.csv"
    corr_df.to_csv(str(out_path), index=False)
    log.info("PHQ-9 correlations saved → %s", out_path)

    top5 = corr_df.head(5)
    log.info("Top 5 PHQ-9 correlates:")
    for _, r in top5.iterrows():
        log.info("  %-45s  ρ=%+.3f  p=%.4f", r["feature"], r["spearman_rho"], r["pval"])


# ══════════════════════════════════════════════════════════════════════════════
# 8. 4-WAY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def print_4way_comparison(modma_results: pd.DataFrame, mode_label: str):
    """Print comparison table: Cavanagh ds003474/ds003478/ds005356 + MODMA."""
    # Reference values from prior runs
    reference = [
        {"dataset": "ds003474", "paradigm": "Task EEG (PST)",
         "best_pipeline": "DDS+Info+Baseline", "roc_auc": 0.836},
        {"dataset": "ds003478", "paradigm": "Resting EEG",
         "best_pipeline": "DDS+Info+Baseline", "roc_auc": 0.715},
        {"dataset": "ds005356", "paradigm": "Task MEG+EEG (PST)",
         "best_pipeline": "Baseline", "roc_auc": 0.585},
    ]

    # Best result from MODMA per feature set
    if not modma_results.empty:
        for fs in modma_results["feature_set"].unique():
            sub = modma_results[modma_results["feature_set"] == fs]
            best_row = sub.loc[sub["roc_auc"].idxmax()]
            reference.append({
                "dataset":      f"MODMA ({mode_label})",
                "paradigm":     "ERP task" if mode_label == "erp" else "Resting EEG",
                "best_pipeline": f"{fs} / {best_row['classifier']}",
                "roc_auc":      best_row["roc_auc"],
            })

    cmp_df = pd.DataFrame(reference).sort_values("roc_auc", ascending=False)

    sep = "=" * 78
    print(f"\n{sep}")
    print("4-WAY COMPARISON: Cavanagh (ds003474/ds003478/ds005356) + MODMA")
    print(sep)
    print(f"{'Dataset + Paradigm':35s}  {'Best pipeline':30s}  AUC")
    print("-" * 78)
    for _, r in cmp_df.iterrows():
        tag = f"{r['dataset']} / {r['paradigm']}"
        print(f"  {tag:33s}  {r['best_pipeline']:30s}  {r['roc_auc']:.3f}")
    print(sep)

    return cmp_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="MODMA DDS+Info+Baseline pipeline")
    parser.add_argument("--mode", choices=["rest", "erp"], default="rest",
                        help="rest = resting .mat files; erp = emotional face .raw files")
    parser.add_argument("--max_subjects", type=int, default=None,
                        help="Limit to N subjects (for testing)")
    args = parser.parse_args()

    mode       = args.mode
    mode_label = mode
    log.info("MODMA pipeline  mode=%s", mode)

    # 1. Metadata
    meta_df  = load_metadata()

    # 2. File discovery
    file_map = find_files(mode)

    # 3. Feature extraction
    out_csv  = OUT_DIR / f"modma_{mode_label}_features.csv"
    df       = run_extraction(meta_df, file_map, mode, out_csv,
                              max_subjects=args.max_subjects)

    if len(df) < 10:
        log.error("Too few subjects (%d) for classification. Exiting.", len(df))
        return

    log.info("\nGroup distribution: HC=%d  MDD=%d",
             (df["group"] == 0).sum(), (df["group"] == 1).sum())

    # 4. Classification
    log.info("\nRunning classification …")
    results_df = run_classification(df, mode_label)
    results_csv = OUT_DIR / f"modma_{mode_label}_classification.csv"
    results_df.to_csv(str(results_csv), index=False)
    log.info("Classification results → %s", results_csv)

    # 5. PHQ-9 correlations
    run_phq9_correlation(df, results_df, mode_label, OUT_DIR)

    # 6. 4-way comparison
    cmp_df = print_4way_comparison(results_df, mode_label)
    cmp_csv = OUT_DIR / f"comparison_4way_modma_{mode_label}.csv"
    cmp_df.to_csv(str(cmp_csv), index=False)

    log.info("\nPipeline complete. Outputs in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
