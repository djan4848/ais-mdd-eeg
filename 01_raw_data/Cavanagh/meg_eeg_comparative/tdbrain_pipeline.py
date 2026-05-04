"""
tdbrain_pipeline.py — DDS+Info+Baseline pipeline for the TDBRAIN dataset
=========================================================================
Dataset : TDBRAIN (van Dijk et al. 2021, BrainClinics/neuroCare, Netherlands)
          DOI: 10.7303/syn25671079
Subjects: MDD=320, HC=47  (DISCOVERY set, ses-1 only)
EEG     : 26 channels 10-20, 500 Hz → resample 250 Hz, BrainVision format
Paradigm: Resting state eyes-open (restEO, 2 min) + eyes-closed (restEC, 2 min)
Scale   : BDI_pre (available ~40% of MDD); primary labels are clinical (MDD/HC)
Notch   : 50 Hz (Netherlands)
Reference: Linked Mastoids (pre-applied) → re-ref to average in pipeline

Usage:
    python tdbrain_pipeline.py               # restEO mode (default)
    python tdbrain_pipeline.py --task restEC # eyes-closed
    python tdbrain_pipeline.py --max_subjects 5  # quick test
"""

import sys, os, logging, argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
from scipy.optimize import curve_fit

import mne
mne.set_log_level("WARNING")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── optional deps ──────────────────────────────────────────────────────────────
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.base import clone

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
TDBRAIN_ROOT = Path("/media/neuraldyn/PortableSSD/TDBRAIN_derivatives")
META_TSV     = TDBRAIN_ROOT / "TDBRAIN_participants_V2.tsv"
OUT_DIR      = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/meg_eeg_comparative")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SFREQ_TARGET  = 250          # resample from 500 → 250 Hz
L_FREQ        = 1.0
H_FREQ        = 45.0
NOTCH_FREQ    = 50.0         # Netherlands power line

DDS_WIN_MS    = 400
DDS_WIN_N     = int(DDS_WIN_MS * SFREQ_TARGET / 1000)   # 100 samples
DDS_MAX_WINS  = 80           # reduced from 100 (only 120 s of data)

WELCH_WIN_S   = 4.0
WELCH_WIN_N   = int(WELCH_WIN_S * SFREQ_TARGET)         # 1000 samples

CV_FOLDS      = 5
N_PCA         = 40
SEED          = 42

# Artifact / non-EEG channels to drop
ART_CHANNELS  = {"VPVA", "VNVB", "HPHL", "HNHR", "Erbs", "OrbOcc", "Mass"}

# ROIs — direct 10-20 names (26-channel subset of full 10-20)
# Note: AF3/AF4/F5/F6/FC2/AFz not present in 26-ch TDBRAIN montage
TDBRAIN_ROIS = {
    "frontal": ["F3", "F4", "Fp1", "Fp2", "FC3", "FC4", "Fz"],
    "cACC":    ["FCz", "Fz"],          # FCz = proxy for AFz/FC2
    "LH":      ["F3", "F7", "FC3"],
    "RH":      ["F4", "F8", "FC4"],
}

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
    """
    Return DataFrame indexed by (participants_ID, sessID=1) with:
    group=0 (HC), group=1 (MDD), BDI_pre (continuous, may be NaN).
    Uses DISCOVERY subjects only, session 1 only.
    """
    df = pd.read_csv(META_TSV, sep="\t", low_memory=False)
    df = df[df["DISC/REP"] == "DISCOVERY"].copy()
    df = df[df["indication"].isin(["MDD", "HEALTHY"])].copy()
    df = df[df["sessID"] == 1.0].copy()
    df["group"]   = (df["indication"] == "MDD").astype(int)
    df["BDI_pre"] = pd.to_numeric(df["BDI_pre"], errors="coerce")
    df = df.set_index("participants_ID")
    n_hc  = (df.group == 0).sum()
    n_mdd = (df.group == 1).sum()
    log.info("Metadata: %d subjects  HC=%d  MDD=%d", len(df), n_hc, n_mdd)
    bdi_valid = df["BDI_pre"].notna().sum()
    log.info("BDI_pre available: %d/%d  mean=%.1f ± %.1f",
             bdi_valid, n_mdd,
             df.loc[df.group==1, "BDI_pre"].mean(),
             df.loc[df.group==1, "BDI_pre"].std())
    return df


def find_eeg_files(meta_df: pd.DataFrame, task: str = "restEO"):
    """Return {sub_id: Path} for .vhdr files of the given task."""
    file_map = {}
    for sub_id in meta_df.index:
        vhdr = (TDBRAIN_ROOT / sub_id / "ses-1" / "eeg"
                / f"{sub_id}_ses-1_task-{task}_eeg.vhdr")
        if vhdr.exists():
            file_map[sub_id] = vhdr
    log.info("Found %d EEG files for task=%s.", len(file_map), task)
    return file_map


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess(vhdr_path: Path, sub_id: str):
    """
    Load BrainVision, drop artifact channels, filter, ICA, re-reference.
    Returns (raw_clean, n_bad_ch, n_ica_removed).
    """
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    # Drop known artifact/aux channels
    to_drop = [c for c in ART_CHANNELS if c in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)

    # Mark bad channels from channels.tsv
    tsv_path = str(vhdr_path).replace("_eeg.vhdr", "_channels.tsv")
    if os.path.exists(tsv_path):
        ch_df = pd.read_csv(tsv_path, sep="\t")
        bads_tsv = ch_df.loc[ch_df.get("status", pd.Series()) == "bad", "name"].tolist()
        bads_tsv = [c for c in bads_tsv if c in raw.ch_names]
        if bads_tsv:
            raw.info["bads"] = bads_tsv
    n_tsv_bad = len(raw.info.get("bads", []))

    log.info("[%s]  Loaded: %d EEG ch, %.0f s, %d bad from TSV",
             sub_id, len(raw.ch_names), raw.times[-1], n_tsv_bad)

    # Resample to 250 Hz
    if raw.info["sfreq"] != SFREQ_TARGET:
        raw.resample(SFREQ_TARGET, verbose=False)

    # Bandpass + notch
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    # Additional bad channel detection: std > 4× median
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_picks) > 0:
        data_eeg  = raw.get_data(picks=eeg_picks)
        data_std  = data_eeg.std(axis=1)
        med_std   = np.median(data_std)
        eeg_names = [raw.ch_names[i] for i in eeg_picks]
        extra_bad = [eeg_names[i] for i, s in enumerate(data_std)
                     if s > 4 * med_std and eeg_names[i] not in raw.info.get("bads",[])]
        if extra_bad:
            raw.info["bads"] = list(raw.info.get("bads", [])) + extra_bad

    n_bad_total = len(raw.info.get("bads", []))
    if n_bad_total > 0:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                        on_missing="ignore", verbose=False)
        raw.interpolate_bads(reset_bads=True, verbose=False)
    log.info("[%s]  Bad channels interpolated: %d", sub_id, n_bad_total)

    # Re-reference to average (was linked mastoids)
    raw.set_eeg_reference("average", verbose=False)

    # ICA — use Fp1 as EOG proxy (frontopolar, closest to eye)
    log.info("[%s]  Running ICA…", sub_id)
    ica = mne.preprocessing.ICA(n_components=15, random_state=SEED,
                                  max_iter="auto", verbose=False)
    raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(raw_hp, verbose=False)

    n_excluded = 0
    for eog_proxy in ["Fp1", "Fp2"]:
        if eog_proxy in raw.ch_names:
            try:
                idx, _ = ica.find_bads_eog(raw, ch_name=eog_proxy,
                                            threshold=3.0, verbose=False)
                ica.exclude = list(set(list(ica.exclude) + idx))
                n_excluded = len(ica.exclude)
            except Exception:
                pass
            break

    # ICLabel fallback
    if n_excluded == 0:
        try:
            from mne_icalabel import label_components
            raw_tmp = raw.copy().filter(1.0, 100.0, verbose=False)
            ica_tmp = mne.preprocessing.ICA(n_components=15, random_state=SEED,
                                             max_iter="auto", verbose=False)
            ica_tmp.fit(raw_tmp, verbose=False)
            labels = label_components(raw_tmp, ica_tmp, method="iclabel")
            bad_types = {"muscle artifact", "eye blink", "heart beat", "other"}
            excl = [i for i, (lbl, prob) in enumerate(
                        zip(labels["labels"], labels["y_pred_proba"]))
                    if lbl in bad_types and prob > 0.70]
            ica.exclude = excl
            n_excluded = len(excl)
        except Exception:
            pass

    ica.apply(raw, verbose=False)
    log.info("[%s]  ICA: %d components removed", sub_id, n_excluded)

    return raw, n_bad_total, n_excluded


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE EXTRACTION  (reused from modma_pipeline.py logic)
# ══════════════════════════════════════════════════════════════════════════════
BANDS = [("delta",1,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,45)]

def extract_baseline_spectral(raw):
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data  = raw.get_data(picks=eeg_picks)
    sfreq = raw.info["sfreq"]
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    step  = WELCH_WIN_N // 2
    n_ch, n_t = data.shape
    starts = list(range(0, n_t - WELCH_WIN_N + 1, step))
    band_acc   = {f"{ch}_{b}": []   for ch in ch_names for b, _, _ in BANDS}
    hjorth_acc = {f"{ch}_hjorth_{h}": [] for ch in ch_names for h in ("mob","comp")}
    for s in starts:
        seg   = data[:, s:s+WELCH_WIN_N]
        freqs, psd = ss.welch(seg, fs=sfreq, nperseg=WELCH_WIN_N, axis=1)
        df_f  = freqs[1] - freqs[0]
        for _, bname, fl, fh in [(i, *b) for i, b in enumerate(BANDS)]:
            idx = np.where((freqs >= fl) & (freqs < fh))[0]
            bp  = psd[:, idx].sum(axis=1) * df_f
            for ci, ch in enumerate(ch_names):
                band_acc[f"{ch}_{bname}"].append(bp[ci])
        d1  = np.diff(seg, axis=1)
        d2  = np.diff(d1, axis=1)
        var0 = seg.var(axis=1) + 1e-30
        var1 = d1.var(axis=1) + 1e-30
        var2 = d2.var(axis=1) + 1e-30
        mob  = np.sqrt(var1 / var0)
        comp = np.sqrt(var2 / var1) / (mob + 1e-30)
        for ci, ch in enumerate(ch_names):
            hjorth_acc[f"{ch}_hjorth_mob"].append(mob[ci])
            hjorth_acc[f"{ch}_hjorth_comp"].append(comp[ci])
    feats = {}
    for k, v in {**band_acc, **hjorth_acc}.items():
        feats[k] = float(np.mean(v)) if v else np.nan
    return feats, len(starts)


def _dds_model(t, A1, a1, f1, p1, A2, a2, f2, p2):
    return (A1*np.exp(-a1*t)*np.cos(2*np.pi*f1*t+p1) +
            A2*np.exp(-a2*t)*np.cos(2*np.pi*f2*t+p2))

_DDS_PARAMS = ("A1","alpha1","f1","phi1","A2","alpha2","f2","phi2")

def _fit_dds_window(seg, sfreq):
    t   = np.arange(len(seg)) / sfreq
    y   = seg - seg.mean()
    amp = float(np.std(y)) * 1.4
    if amp < 1e-30:
        return None
    p0  = [amp, 5.0, 10.0, 0.0, amp*0.5, 8.0, 25.0, 0.0]
    bnd = ([0,0,1,-np.pi, 0,0,1,-np.pi], [amp*10,100,45,np.pi, amp*10,100,45,np.pi])
    try:
        popt, _ = curve_fit(_dds_model, t, y, p0=p0, bounds=bnd,
                            method="trf", max_nfev=1200)
        res = y - _dds_model(t, *popt)
        ss_tot = float(np.sum((y-y.mean())**2)) + 1e-30
        r2  = max(0.0, 1.0 - np.sum(res**2)/ss_tot)
        return {p: float(v) for p, v in zip(_DDS_PARAMS, popt)} | {"r2": r2, "residual": res}
    except Exception:
        return None


def extract_dds_features(raw, rng=None):
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
    for roi_name, roi_chs in TDBRAIN_ROIS.items():
        avail = [c for c in roi_chs if c in ch_names]
        if not avail:
            continue
        ch_idx     = [ch_names.index(c) for c in avail]
        roi_signal = data[ch_idx, :].mean(axis=0)
        max_start  = n_times - DDS_WIN_N
        if max_start <= 0:
            continue
        starts  = rng.integers(0, max_start, size=DDS_MAX_WINS * 3)
        results, residuals = [], []
        for s in starts:
            if len(results) >= DDS_MAX_WINS:
                break
            r = _fit_dds_window(roi_signal[s:s+DDS_WIN_N], sfreq)
            if r is not None:
                residuals.append(r.pop("residual"))
                results.append(r)
        n_ok_total += len(results)
        if not results:
            continue
        residuals_by_roi[roi_name] = np.array(residuals)
        for p in _DDS_PARAMS + ("r2",):
            vals = [res[p] for res in results]
            feats[f"dds_{roi_name}_{p}_mean"] = float(np.mean(vals))
            feats[f"dds_{roi_name}_{p}_std"]  = float(np.std(vals))
    log.info("  DDS: %d successful windows across ROIs.", n_ok_total)
    return feats, residuals_by_roi


def _ais(x, k=1, bins=16):
    try:
        xq = np.digitize(x, np.linspace(x.min(), x.max()+1e-9, bins+1)) - 1
        past, pres = xq[:-k], xq[k:]
        jh, _, _ = np.histogram2d(past, pres, bins=[bins,bins],
                                   range=[[0,bins-1],[0,bins-1]])
        jh = jh / (jh.sum() + 1e-30)
        px = jh.sum(axis=1, keepdims=True) + 1e-30
        py = jh.sum(axis=0, keepdims=True) + 1e-30
        return max(0.0, float(np.sum(jh * np.log2(jh / (px*py) + 1e-30))))
    except Exception:
        return np.nan

def _te(x, y, k=1, bins=12):
    try:
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max()) + 1e-9
        edges = np.linspace(lo, hi, bins+1)
        xq, yq = np.digitize(x, edges)-1, np.digitize(y, edges)-1
        yp, yf, xp = yq[:-k], yq[k:], xq[:-k]
        def _ce(a, b):
            ab, _, _ = np.histogram2d(a, b, bins=[bins,bins], range=[[0,bins-1],[0,bins-1]])
            ab = ab/(ab.sum()+1e-30); pb = ab.sum(axis=0,keepdims=True)+1e-30
            return -float(np.sum(ab * np.log2(ab/pb + 1e-30)))
        def _ce3(a, b, c):
            idx = b*bins+c; res = 0.0
            for v in np.unique(idx):
                m = idx==v; pb = m.mean()
                if pb == 0: continue
                h, _ = np.histogram(a[m], bins=bins, range=(0,bins-1))
                h = h/(h.sum()+1e-30)
                res += pb * (-np.sum(h * np.log2(h+1e-30)))
            return res
        return max(0.0, _ce(yp,yf) - _ce3(yf,xp,yp))
    except Exception:
        return np.nan

def _pid_approx(s1, s2, t, bins=10):
    try:
        lo = min(s1.min(),s2.min(),t.min()); hi = max(s1.max(),s2.max(),t.max())+1e-9
        edges = np.linspace(lo, hi, bins+1)
        s1q = np.digitize(s1,edges)-1; s2q = np.digitize(s2,edges)-1
        tq  = np.digitize(t, edges)-1
        def _mi2(a,b):
            h,_,_ = np.histogram2d(a,b,bins=[bins,bins],range=[[0,bins-1],[0,bins-1]])
            h = h/(h.sum()+1e-30); pa = h.sum(axis=1,keepdims=True)+1e-30
            pb = h.sum(axis=0,keepdims=True)+1e-30
            return float(np.sum(h * np.log2(h/(pa*pb)+1e-30)))
        i1 = max(0.0, _mi2(s1q,tq)); i2 = max(0.0, _mi2(s2q,tq))
        red = min(i1,i2)
        idx = s1q*bins**2 + s2q*bins + tq
        p3 = np.bincount(idx, minlength=bins**3).astype(float)
        p3 = p3/(p3.sum()+1e-30)
        p_s1s2 = p3.reshape(bins**2,bins).sum(axis=1,keepdims=True)
        p_t    = p3.reshape(bins**2,bins).sum(axis=0,keepdims=True)
        j12t   = p3.reshape(bins**2,bins)/(p3.sum()+1e-30)
        i12t   = float(np.sum(j12t * np.log2(j12t/((p_s1s2*p_t/(p3.sum()+1e-30))+1e-30)+1e-30)))
        return {"redundancy": red, "unique_s1": max(0.0,i1-red),
                "unique_s2": max(0.0,i2-red), "synergy": max(0.0,i12t-max(i1,i2))}
    except Exception:
        return {"redundancy": np.nan, "unique_s1": np.nan,
                "unique_s2": np.nan,  "synergy":   np.nan}

def extract_info_features(residuals_by_roi):
    feats = {}
    series = {roi: arr.ravel() for roi, arr in residuals_by_roi.items()}
    for roi, sig in series.items():
        feats[f"info_AIS_{roi}"] = _ais(sig)
    for src, tgt in [("LH","frontal"),("RH","frontal"),("LH","RH"),("cACC","frontal")]:
        if src in series and tgt in series:
            n = min(len(series[src]), len(series[tgt]))
            feats[f"info_TE_{src}_to_{tgt}"] = _te(series[src][:n], series[tgt][:n])
    if all(r in series for r in ("LH","RH","frontal")):
        n = min(len(series["LH"]), len(series["RH"]), len(series["frontal"]))
        pid = _pid_approx(series["LH"][:n], series["RH"][:n], series["frontal"][:n])
        for k, v in pid.items():
            feats[f"info_PID_{k}"] = v
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXTRACTION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_extraction(meta_df, file_map, task, output_csv, max_subjects=None):
    already_done = set(); existing_records = []
    if output_csv.exists():
        cached = pd.read_csv(str(output_csv), dtype={"subject_id": str})
        n_before = len(cached)
        cached = cached.drop_duplicates(subset="subject_id", keep="last")
        if len(cached) < n_before:
            log.warning("Dropped %d duplicate rows from cached CSV.", n_before - len(cached))
            cached.to_csv(str(output_csv), index=False)
        already_done = set(cached["subject_id"].tolist())
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

        row  = meta_df.loc[sub_id]
        grp  = int(row["group"])
        bdi  = float(row.get("BDI_pre", np.nan)) if pd.notna(row.get("BDI_pre")) else np.nan
        label = "MDD" if grp == 1 else "HC"
        log.info("Processing %s (%s) …", sub_id, label)

        try:
            raw, n_bad, n_ica = load_and_preprocess(file_map[sub_id], sub_id)

            baseline_feats, n_wins = extract_baseline_spectral(raw)
            log.info("  Baseline: %d windows, %d features", n_wins, len(baseline_feats))

            dds_feats, residuals_by_roi = extract_dds_features(raw)
            info_feats = extract_info_features(residuals_by_roi) if residuals_by_roi else {}

            record = {
                "subject_id": sub_id, "group": grp, "label": label,
                "BDI_pre": bdi,
                "age":  float(str(row.get("age","")).replace(",",".")) if pd.notna(row.get("age")) else np.nan,
                "gender": float(row.get("gender", np.nan)) if pd.notna(row.get("gender")) else np.nan,
                "n_bad_ch": n_bad, "n_ica_removed": n_ica,
                **baseline_feats, **dds_feats, **info_feats,
            }
            records.append(record)
            log.info("  [%s] Done. group=%s", sub_id, label)

        except Exception as e:
            log.error("  [%s] FAILED: %s", sub_id, e)
            import traceback; traceback.print_exc()
            continue

        pd.DataFrame(records).drop_duplicates(subset="subject_id", keep="last").to_csv(
            str(output_csv), index=False)

    df = pd.DataFrame(records).drop_duplicates(subset="subject_id", keep="last").reset_index(drop=True)
    log.info("Extracted: %d subjects × %d features", len(df), df.shape[1])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def get_classifiers():
    clfs = {
        "LogReg":  LogisticRegression(max_iter=2000, random_state=SEED, C=0.1),
        "LDA":     LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "SVM-Lin": SVC(kernel="linear", probability=True, random_state=SEED, C=0.5),
        "SVM-RBF": SVC(kernel="rbf",    probability=True, random_state=SEED,
                       class_weight="balanced"),
        "RF":      RandomForestClassifier(n_estimators=200, random_state=SEED,
                                          class_weight="balanced"),
        "MLP":     MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500,
                                 random_state=SEED),
    }
    if HAS_XGB:
        n_pos = 1; n_neg = 1  # will be updated per call
        clfs["XGB"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=SEED, verbosity=0,
            use_label_encoder=False)
    return clfs


def build_pipeline(clf, n_features, n_subjects):
    n_train = int(n_subjects * (CV_FOLDS-1) / CV_FOLDS) - 1
    n_pca   = min(N_PCA, n_train, n_features)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("var",     VarianceThreshold(threshold=1e-6)),
        ("pca",     PCA(n_components=n_pca, random_state=SEED)),
        ("clf",     clf),
    ])


def run_classification(df, task_label):
    meta_cols = {"subject_id","group","label","BDI_pre","age","gender",
                 "n_bad_ch","n_ica_removed"}
    all_fcols = [c for c in df.columns if c not in meta_cols]
    baseline  = [c for c in all_fcols if not c.startswith(("dds_","info_"))]
    dds_cols  = [c for c in all_fcols if c.startswith("dds_")]
    info_cols = [c for c in all_fcols if c.startswith("info_")]

    def clean(cols):
        return [c for c in cols if df[cols].isnull().mean()[c] <= 0.10]

    baseline  = clean(baseline)
    dds_cols  = clean(dds_cols)
    info_cols = clean(info_cols)

    feature_sets = {
        "Baseline spectral": baseline,
        "DDS only":          dds_cols,
        "Info only":         info_cols,
        "DDS+Info":          dds_cols + info_cols,
        "DDS+Info+Baseline": dds_cols + info_cols + baseline,
    }

    labels  = df["group"].values
    n_mdd   = int(labels.sum())
    n_hc    = int((labels==0).sum())
    N       = len(labels)
    clfs    = get_classifiers()
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    results = []

    for fs_name, fcols in feature_sets.items():
        if not fcols:
            continue
        X = df[fcols].values.astype(float)
        log.info("  [%s] %d subjects, %d features (HC=%d, MDD=%d)",
                 fs_name, N, len(fcols), n_hc, n_mdd)

        for clf_name, clf_obj in clfs.items():
            aucs, bals = [], []
            for tr, te in cv.split(X, labels):
                X_tr, y_tr = X[tr], labels[tr]
                X_te, y_te = X[te], labels[te]
                try:
                    if HAS_SMOTE and y_tr.sum() >= 2 and (y_tr==0).sum() >= 2:
                        k_sm = min(5, int(min(y_tr.sum(), (y_tr==0).sum())) - 1)
                        if k_sm >= 1:
                            sm = SMOTE(k_neighbors=k_sm, random_state=SEED)
                            X_tr, y_tr = sm.fit_resample(
                                SimpleImputer(strategy="median").fit_transform(X_tr), y_tr)
                    pipe = build_pipeline(clone(clf_obj), X_tr.shape[1], len(y_tr))
                    pipe.fit(X_tr, y_tr)
                    prob = pipe.predict_proba(X_te)[:, 1]
                    aucs.append(roc_auc_score(y_te, prob))
                    bals.append(balanced_accuracy_score(y_te, pipe.predict(X_te)))
                except Exception as e:
                    log.debug("  fold skip %s/%s: %s", fs_name, clf_name, e)
                    continue

            if aucs:
                auc = float(np.mean(aucs)); bal = float(np.mean(bals))
                log.info("    %-30s AUC=%.3f  BalAcc=%.3f", clf_name, auc, bal)
                results.append({
                    "dataset": "TDBRAIN", "task": task_label,
                    "feature_set": fs_name, "classifier": clf_name,
                    "roc_auc": auc, "roc_auc_std": float(np.std(aucs)),
                    "bal_acc": bal, "N": N, "n_HC": n_hc, "n_MDD": n_mdd,
                    "n_features": len(fcols),
                })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# 6. BDI CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
def run_bdi_correlation(df, task_label, out_dir):
    from scipy.stats import spearmanr
    bdi = df["BDI_pre"].values.astype(float)
    if np.isnan(bdi).all():
        return
    dds_info = [c for c in df.columns
                if c.startswith(("dds_","info_")) and df[c].notna().mean() > 0.9]
    if not dds_info:
        return
    rows = []
    for col in dds_info:
        x = df[col].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(bdi))
        if mask.sum() < 10:
            continue
        rho, pval = spearmanr(x[mask], bdi[mask])
        rows.append({"feature": col, "spearman_rho": rho, "pval": pval})
    if not rows:
        return
    corr_df = pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)
    out = out_dir / f"tdbrain_{task_label}_bdi_correlations.csv"
    corr_df.to_csv(str(out), index=False)
    log.info("BDI correlations → %s", out)
    for _, r in corr_df.head(5).iterrows():
        log.info("  %-45s  ρ=%+.3f  p=%.4f", r["feature"], r["spearman_rho"], r["pval"])


# ══════════════════════════════════════════════════════════════════════════════
# 7. 5-WAY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def print_5way_comparison(tdbrain_results, task_label):
    reference = [
        {"dataset":"ds003474","paradigm":"EEG Task PST","diagnosis":"BDI≥13",
         "best_pipeline":"DDS+Info+Baseline","roc_auc":0.836},
        {"dataset":"MODMA","paradigm":"EEG Resting","diagnosis":"PHQ-9 clinical",
         "best_pipeline":"DDS+Info / LogReg","roc_auc":0.790},
        {"dataset":"ds003478","paradigm":"EEG Resting","diagnosis":"BDI≥13",
         "best_pipeline":"DDS+Info+Baseline","roc_auc":0.715},
        {"dataset":"ds005356","paradigm":"MEG+EEG Task PST","diagnosis":"SCID",
         "best_pipeline":"Baseline","roc_auc":0.585},
    ]
    if not tdbrain_results.empty:
        for fs in tdbrain_results["feature_set"].unique():
            sub = tdbrain_results[tdbrain_results["feature_set"]==fs]
            br  = sub.loc[sub["roc_auc"].idxmax()]
            reference.append({
                "dataset": f"TDBRAIN ({task_label})",
                "paradigm": "EEG Resting", "diagnosis": "Clinical MDD/HC",
                "best_pipeline": f"{fs} / {br['classifier']}",
                "roc_auc": br["roc_auc"],
            })
    cmp = pd.DataFrame(reference).sort_values("roc_auc", ascending=False)
    sep = "=" * 82
    print(f"\n{sep}")
    print("5-WAY COMPARISON: ds003474 / ds003478 / ds005356 / MODMA / TDBRAIN")
    print(sep)
    print(f"{'Dataset + Paradigm':37s}  {'Best pipeline':32s}  AUC")
    print("-" * 82)
    for _, r in cmp.iterrows():
        tag = f"{r['dataset']} / {r['paradigm']}"
        print(f"  {tag:35s}  {r['best_pipeline']:32s}  {r['roc_auc']:.3f}")
    print(sep)
    return cmp


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="TDBRAIN DDS+Info+Baseline pipeline")
    parser.add_argument("--task", choices=["restEO","restEC"], default="restEO",
                        help="restEO = eyes open (default); restEC = eyes closed")
    parser.add_argument("--max_subjects", type=int, default=None)
    args = parser.parse_args()

    task = args.task
    log.info("TDBRAIN pipeline  task=%s", task)

    meta_df  = load_metadata()
    file_map = find_eeg_files(meta_df, task)

    out_csv  = OUT_DIR / f"tdbrain_{task}_features.csv"
    df       = run_extraction(meta_df, file_map, task, out_csv, args.max_subjects)

    if len(df) < 15:
        log.error("Too few subjects (%d). Exiting.", len(df))
        return

    log.info("Group distribution: HC=%d  MDD=%d",
             (df["group"]==0).sum(), (df["group"]==1).sum())

    log.info("\nRunning classification …")
    results_df = run_classification(df, task)
    res_csv    = OUT_DIR / f"tdbrain_{task}_classification.csv"
    results_df.to_csv(str(res_csv), index=False)
    log.info("Results → %s", res_csv)

    run_bdi_correlation(df, task, OUT_DIR)
    cmp = print_5way_comparison(results_df, task)
    cmp.to_csv(OUT_DIR / f"comparison_5way_tdbrain_{task}.csv", index=False)

    log.info("\nPipeline complete. Outputs in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
