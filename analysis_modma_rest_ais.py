"""
MODMA resting-state AIS_rest: HC vs MDD
128-ch EEG, .mat format, 250Hz, ~5 min
E6 = FCz equivalent (GSN-HydroCel-128, 4.7mm from FCz)
Same pipeline as TDBRAIN and ds003478:
  1-40Hz filter, 2s windows, lag=1, bins=4, min 10 windows
"""

import warnings; warnings.filterwarnings('ignore')
import json, re
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as ssig
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr

BASE    = Path("/media/neuraldyn/PortableSSD/DEPRESSION")
MODMA   = BASE / "01_raw_data/MODMA"
REST    = MODMA / "EEG_128channels_resting_lanzhou_2015"
PST     = BASE / "01_raw_data/Cavanagh/Depression_PS_Task"
ASSETS  = BASE / "06_manuscript_assets"

E6_IDX      = 5        # row index in (129, samples) mat data
SFREQ       = 250.0    # confirmed from file
WINDOW_SEC  = 2.0
MIN_WINDOWS = 10
LAG         = 1
N_BINS      = 4

# ── helpers ──────────────────────────────────────────────────────────────────

def safe_ais(x, lag=1, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < 2*lag + 10: return np.nan
    if np.std(x) < 1e-12:   return np.nan
    try:
        edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins+1)))
        if len(edges) < 3: return np.nan
        bins = np.digitize(x, edges[1:-1])
        x_t, x_lag = bins[lag:], bins[:-lag]
        joint = np.zeros((n_bins, n_bins))
        for a, b in zip(x_t, x_lag):
            joint[a-1, b-1] += 1
        joint /= joint.sum() + 1e-10
        px_t, px_lag = joint.sum(axis=1), joint.sum(axis=0)
        mi = sum(joint[i,j] * np.log2(joint[i,j] / (px_t[i]*px_lag[j]))
                 for i in range(n_bins) for j in range(n_bins)
                 if joint[i,j]>0 and px_t[i]>0 and px_lag[j]>0)
        return float(mi) if np.isfinite(mi) else np.nan
    except Exception:
        return np.nan

def bandpass(sig, lo=1., hi=40., fs=250.):
    sos = ssig.butter(4, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return ssig.sosfiltfilt(sos, sig)

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1))
                 / (len(a)+len(b)-2) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / sp)

# ── AIS validation ────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
ar1 = np.zeros(500)
for i in range(1, 500): ar1[i] = 0.9*ar1[i-1] + 0.1*rng.standard_normal()
wn  = rng.standard_normal(500)
assert safe_ais(ar1) > safe_ais(wn), "AIS validation failed"
print("AIS validation OK ✅")

# ── STEP 0 — Load subjects info ───────────────────────────────────────────────
info_file = REST / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"
df_info   = pd.read_excel(info_file)
df_info   = df_info[df_info['type'].isin(['MDD','HC'])].copy()
df_info['sub_str'] = df_info['subject id'].astype(str).str.zfill(7)

print(f"\nSubjects: HC={( df_info['type']=='HC').sum()}, "
      f"MDD={(df_info['type']=='MDD').sum()}")
print(f"PHQ-9 HC:  {df_info[df_info['type']=='HC']['PHQ-9'].mean():.1f} ± "
      f"{df_info[df_info['type']=='HC']['PHQ-9'].std():.1f}")
print(f"PHQ-9 MDD: {df_info[df_info['type']=='MDD']['PHQ-9'].mean():.1f} ± "
      f"{df_info[df_info['type']=='MDD']['PHQ-9'].std():.1f}")

# ── STEP 1 — Build file index ─────────────────────────────────────────────────
# Filenames like: "02010002rest 20150416 1017..mat"
# Subject ID in info: 2010002 → padded to "2010002", filename prefix "0201002.."
# Extract ID from filename: strip leading zeros and 'rest' suffix

mat_files = list(REST.glob("*.mat"))
file_map  = {}   # sub_str → Path
for fp in mat_files:
    m = re.match(r'0?(\d{7})', fp.name)
    if m:
        key = m.group(1)
        file_map[key] = fp

print(f"\nMat files found: {len(mat_files)}")
print(f"Matched to IDs:  {len(file_map)}")

# ── STEP 2 — Compute AIS_rest ─────────────────────────────────────────────────
records = []
win_len = int(WINDOW_SEC * SFREQ)

for _, row in df_info.iterrows():
    sub = row['sub_str']
    grp = row['type']
    phq = row['PHQ-9']

    fp = file_map.get(sub)
    if fp is None:
        print(f"  {sub} ({grp}): no file found")
        continue

    try:
        mat      = sio.loadmat(str(fp))
        data_key = [k for k in mat.keys()
                    if not k.startswith('_') and k not in ('samplingRate','Impedances_0')][0]
        raw_data = mat[data_key].astype(float)   # shape (129, samples)

        sig = raw_data[E6_IDX]                   # E6 channel

        if np.std(sig) < 1e-8 or np.any(np.isnan(sig)) or np.any(np.isinf(sig)):
            print(f"  {sub}: bad signal")
            continue

        sig = bandpass(sig, 1., 40., SFREQ)

        n_wins  = len(sig) // win_len
        if n_wins < MIN_WINDOWS:
            print(f"  {sub}: too short ({n_wins} windows)")
            continue

        ais_vals = [safe_ais(sig[i*win_len:(i+1)*win_len], LAG, N_BINS)
                    for i in range(n_wins)]
        valid    = [v for v in ais_vals if np.isfinite(v)]

        if len(valid) < MIN_WINDOWS:
            print(f"  {sub}: too few valid windows ({len(valid)})")
            continue

        mean_ais = float(np.mean(valid))
        records.append({
            'subject_id': sub,
            'group':      grp,
            'AIS_rest':   mean_ais,
            'PHQ9':       phq,
            'n_windows':  len(valid),
        })
        print(f"  {sub} ({grp}): AIS_rest={mean_ais:.4f}  ({len(valid)} windows)")

    except Exception as e:
        print(f"  {sub}: ERROR — {e}")

df = pd.DataFrame(records)
print(f"\nProcessed: {len(df)}  (HC={( df['group']=='HC').sum()}, "
      f"MDD={(df['group']=='MDD').sum()})")
print(df.groupby('group')['AIS_rest'].describe().round(4).to_string())

# ── STEP 3 — HC vs MDD ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("PRIMARY: AIS_rest HC vs MDD  (MODMA resting)")
print("="*55)

hc  = df[df['group']=='HC' ]['AIS_rest'].dropna()
mdd = df[df['group']=='MDD']['AIS_rest'].dropna()

_, p_hm = mannwhitneyu(hc, mdd, alternative='two-sided')
d_hm    = cohens_d(hc.values, mdd.values)

print(f"\nHC:  {hc.mean():.4f} ± {hc.std():.4f}  (N={len(hc)})")
print(f"MDD: {mdd.mean():.4f} ± {mdd.std():.4f}  (N={len(mdd)})")
print(f"d = {d_hm:+.3f}, p = {p_hm:.4f}")
print(f"Direction HC>MDD: {'✅' if hc.mean()>mdd.mean() else '❌'}")

# ── STEP 4 — PHQ-9 continuous correlation ─────────────────────────────────────
print("\n" + "="*55)
print("PHQ-9 correlation")
print("="*55)
all_phq = df[['AIS_rest','PHQ9']].dropna()
r_phq, p_phq = pearsonr(all_phq['AIS_rest'], all_phq['PHQ9'])
print(f"r(AIS_rest, PHQ-9) = {r_phq:+.3f}, p={p_phq:.4f}  (N={len(all_phq)})")

# ── STEP 5 — Cross-dataset evidence table ─────────────────────────────────────
print("\n" + "="*65)
print("UPDATED EVIDENCE TABLE")
print("="*65)
print(f"""
┌─────────────────────────────┬───────┬────────┬──────────┐
│ Finding                     │   d   │   p    │ Dataset  │
├─────────────────────────────┼───────┼────────┼──────────┤
│ AIS_pre: CTL>MDD (task PST) │+0.817 │ 0.0003 │ Cav EEG  │
│ AIS_rest: CTL>>MDD (rest)   │+2.024 │  ≈0    │ TDBRAIN  │
│ AIS_rest: CTL>MDD  (rest)   │+0.703 │ 0.0031 │ ds003478 │
│ AIS_rest: HC>MDD   (rest)   │{d_hm:+.3f} │ {p_hm:.4f} │ MODMA    │ ← NEW
│ AIS_pre scar: cur>past      │+0.965 │ 0.017  │ Cav EEG  │
│ Boundary: MODMA task null   │+0.091 │ 0.675  │ MODMA    │
│ Boundary: Hayling null      │+0.018 │ 0.873  │ Hayling  │
└─────────────────────────────┴───────┴────────┴──────────┘
""")

# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "06_manuscript_assets/modma_rest_ais_subjects.csv"
df.to_csv(out, index=False)
print(f"Saved: {out}")

# Update frozen state
freeze = PST / "derivatives/RESEARCH_STATE_FROZEN.json"
with open(freeze) as f:
    state = json.load(f)

state['modma_resting_ais'] = {
    'date':        '2026-05-03',
    'N_HC':        int(len(hc)),
    'N_MDD':       int(len(mdd)),
    'HC_mean':     round(float(hc.mean()), 4),
    'MDD_mean':    round(float(mdd.mean()), 4),
    'd_hc_vs_mdd': round(d_hm, 3),
    'p_hc_vs_mdd': round(float(p_hm), 4),
    'direction':   'HC>MDD' if hc.mean() > mdd.mean() else 'MDD>HC',
    'r_PHQ9':      round(float(r_phq), 3),
    'p_PHQ9':      round(float(p_phq), 4),
    'channel':     'E6 (FCz equivalent, GSN-HydroCel-128)',
    'method':      '1-40Hz butter, 2s windows, lag=1, bins=4',
}

with open(freeze, 'w') as f:
    json.dump(state, f, indent=2)
print("Frozen state updated.")
