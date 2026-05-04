"""
MEG ds005356 — AIS_rest from pre-task resting segment
Channel : EEG007 (FCz equivalent, x=-5.3mm, y=47.4mm; same as 14a_v2)
Resting : t=0 to first_event (varies ~40–160s per subject)
Method  : 1-40Hz butter, 2s windows, lag=1, bins=4, min 10 windows
Groups  : CTL / MDD  from  Code/MEG MDD IDs and Quex.xlsx
Excluded: sub-M87121835 (DC-drift EEG007, flagged in 14a_v2)
"""

import warnings; warnings.filterwarnings('ignore')
import json
import numpy as np
import pandas as pd
import mne; mne.set_log_level('ERROR')
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr
import scipy.signal as ssig
import glob

BASE    = Path("/media/neuraldyn/PortableSSD/DEPRESSION")
MEG     = BASE / "01_raw_data/Cavanagh/ds005356"
PST     = BASE / "01_raw_data/Cavanagh/Depression_PS_Task"
ASSETS  = BASE / "06_manuscript_assets"

CHANNEL    = 'EEG007'
WINDOW_SEC = 2.0
MIN_WIN    = 10
LAG        = 1
N_BINS     = 4
EXCLUDE    = {'sub-M87121835'}   # DC-drift outlier from 14a_v2

# ── helpers ──────────────────────────────────────────────────────────────────

def safe_ais(x, lag=1, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < 2*lag + 10: return np.nan
    if np.std(x) < 1e-12:   return np.nan
    try:
        edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins+1)))
        if len(edges) < 3: return np.nan
        bins = np.digitize(x, edges[1:-1])   # values 0..n_bins-1
        x_t, x_lag = bins[lag:], bins[:-lag]
        flat  = x_t * n_bins + x_lag
        joint = np.bincount(flat, minlength=n_bins*n_bins).reshape(n_bins, n_bins).astype(float)
        joint /= joint.sum() + 1e-10
        px_t  = joint.sum(axis=1, keepdims=True)
        px_lag = joint.sum(axis=0, keepdims=True)
        mask  = (joint > 0) & (px_t > 0) & (px_lag > 0)
        mi    = np.sum(joint[mask] * np.log2(joint[mask] / (px_t * px_lag)[mask]))
        return float(mi) if np.isfinite(mi) else np.nan
    except Exception:
        return np.nan

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1))
                 / (len(a)+len(b)-2) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / sp)

# ── AIS validation ────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
ar1 = np.zeros(500)
for i in range(1, 500): ar1[i] = 0.9*ar1[i-1] + 0.1*rng.standard_normal()
assert safe_ais(ar1) > safe_ais(rng.standard_normal(500))
print("AIS validation OK ✅")

# ── STEP 0 — Clinical data ────────────────────────────────────────────────────
clinical = pd.read_excel(MEG / "Code/MEG MDD IDs and Quex.xlsx")
clinical['bids_id'] = 'sub-M87' + (100000 + clinical['URSI'].astype(int)).astype(str)
print(f"\nClinical: N={len(clinical)}  "
      f"CTL={sum(clinical['Group']=='CTL')}  "
      f"MDD={sum(clinical['Group']=='MDD')}")

# ── STEP 1 — Build file map ───────────────────────────────────────────────────
single_files = sorted(MEG.glob("sub-M*/ses-01/meg/sub-M*_task-pst_run-1_meg.fif"))
split_files  = sorted(MEG.glob("sub-M*/ses-01/meg/sub-M*_task-pst_run-1_split-01_meg.fif"))

subject_map = {}   # bids_id → first fif file (pre-task rest is always in file 1)
for fp in single_files:
    sid = 'sub-' + fp.name.split('sub-')[1].split('_')[0]
    subject_map[sid] = fp
for fp in split_files:
    sid = 'sub-' + fp.name.split('sub-')[1].split('_')[0]
    subject_map[sid] = fp

# Keep only subjects with clinical data
has_clin = set(clinical['bids_id']) & set(subject_map.keys())
print(f"File+clinical: {len(has_clin)}")

# ── STEP 2 — Compute AIS_rest on pre-task segment ────────────────────────────
records = []
win_len = int(WINDOW_SEC * 250)    # 250Hz after resample

for sid in sorted(has_clin):
    if sid in EXCLUDE:
        print(f"  {sid}: EXCLUDED (DC-drift)")
        continue

    row   = clinical[clinical['bids_id'] == sid].iloc[0]
    group = row['Group']
    bdi   = row.get('BDI', np.nan)
    fp    = subject_map[sid]

    try:
        # ── Get first TASK event from TSV (no FIF reading needed) ──
        # events TSV name never has split suffix
        tsv_name = f"{sid}_ses-01_task-pst_run-1_events.tsv"
        tsv_path = fp.parent / tsv_name
        if not tsv_path.exists():
            print(f"  {sid}: events TSV missing")
            continue
        ev_df    = pd.read_csv(tsv_path, sep='\t')
        # Keep only real task events (cue / feedback), skip system triggers
        task_evs = ev_df[ev_df['trial_type'].str.startswith(('cue','FB'), na=False)]
        if task_evs.empty:
            print(f"  {sid}: no task events found in TSV")
            continue
        rest_end = float(task_evs['onset'].min())   # seconds from recording start

        if rest_end < MIN_WIN * WINDOW_SEC:
            print(f"  {sid}: pre-task too short ({rest_end:.0f}s)")
            continue

        # ── Load only EEG007 for the pre-task window ──
        raw = mne.io.read_raw_fif(fp, preload=False, verbose=False)

        if CHANNEL not in raw.ch_names:
            print(f"  {sid}: {CHANNEL} not found")
            continue

        # pick BEFORE load_data: reduces 396 ch → 1 ch before any disk read
        raw.pick([CHANNEL])
        raw.crop(tmin=0., tmax=rest_end - 0.001)
        raw.load_data(verbose=False)
        # resample to 250Hz before filtering — reduces data 4x, consistent with other datasets
        raw.resample(250, verbose=False)
        raw.filter(1., 40., method='fir', verbose=False)

        sig = raw.get_data()[0]

        if np.std(sig) < 1e-12 or np.any(~np.isfinite(sig)):
            print(f"  {sid}: bad signal")
            continue

        n_wins   = len(sig) // win_len
        ais_vals = [safe_ais(sig[i*win_len:(i+1)*win_len], LAG, N_BINS)
                    for i in range(n_wins)]
        valid    = [v for v in ais_vals if np.isfinite(v)]

        if len(valid) < MIN_WIN:
            print(f"  {sid}: only {len(valid)} valid windows")
            continue

        mean_ais = float(np.mean(valid))
        records.append({
            'subject_id':  sid,
            'group':       group,
            'AIS_rest':    mean_ais,
            'rest_dur_s':  rest_end,
            'n_windows':   len(valid),
            'BDI':         bdi,
        })
        print(f"  {sid} ({group}): AIS_rest={mean_ais:.4f}  "
              f"rest={rest_end:.0f}s  N_win={len(valid)}")

    except Exception as e:
        print(f"  {sid}: ERROR — {e}")

df = pd.DataFrame(records)
print(f"\nProcessed: {len(df)}  "
      f"(CTL={(df['group']=='CTL').sum()}  "
      f"MDD={(df['group']=='MDD').sum()})")
print(df.groupby('group')['AIS_rest'].describe().round(4).to_string())

# ── STEP 3 — CTL vs MDD ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("PRIMARY: AIS_rest CTL vs MDD  (MEG ds005356, EEG007)")
print("="*55)

ctl = df[df['group']=='CTL']['AIS_rest'].dropna()
mdd = df[df['group']=='MDD']['AIS_rest'].dropna()

_, p_cm = mannwhitneyu(ctl, mdd, alternative='two-sided')
d_cm    = cohens_d(ctl.values, mdd.values)

print(f"\nCTL: {ctl.mean():.4f} ± {ctl.std():.4f}  (N={len(ctl)})")
print(f"MDD: {mdd.mean():.4f} ± {mdd.std():.4f}  (N={len(mdd)})")
print(f"d = {d_cm:+.3f}, p = {p_cm:.4f}")
print(f"Direction CTL>MDD: {'✅' if ctl.mean()>mdd.mean() else '❌'}")

# ── STEP 4 — BDI continuous correlation ──────────────────────────────────────
v = df[['AIS_rest','BDI']].dropna()
if len(v) > 10:
    r_bdi, p_bdi = pearsonr(v['AIS_rest'], v['BDI'])
    print(f"\nr(AIS_rest, BDI) = {r_bdi:+.3f}, p={p_bdi:.4f}  (N={len(v)})")

# Rest duration as possible confound
r_dur, p_dur = pearsonr(df['AIS_rest'], df['rest_dur_s'])
print(f"r(AIS_rest, rest_duration) = {r_dur:+.3f}, p={p_dur:.4f}  "
      f"[confound check — should be near 0]")

# ── STEP 5 — Updated evidence table ──────────────────────────────────────────
print("\n" + "="*65)
print("COMPLETE EVIDENCE TABLE — AIS IN MDD")
print("="*65)

# Load frozen MODMA result
with open(PST / "derivatives/RESEARCH_STATE_FROZEN.json") as f:
    state = json.load(f)
modma = state.get('modma_resting_ais', {})
d_modma = modma.get('d_hc_vs_mdd', '?')
p_modma = modma.get('p_hc_vs_mdd', '?')

print(f"""
┌─────────────────────────────┬───────┬────────┬──────────────┐
│ Finding                     │   d   │   p    │ Dataset      │
├─────────────────────────────┼───────┼────────┼──────────────┤
│ AIS_pre: CTL>MDD (task PST) │+0.817 │ 0.0003 │ Cav EEG      │
│ AIS_rest: CTL>>MDD          │+2.024 │  ≈0    │ TDBRAIN      │
│ AIS_rest: CTL>MDD           │+0.703 │ 0.0031 │ ds003478 EEG │
│ AIS_rest: HC>MDD  (n.s.)    │{d_modma:+.3f} │ {p_modma:.4f} │ MODMA EEG    │
│ AIS_rest: CTL>MDD           │{d_cm:+.3f} │ {p_cm:.4f} │ MEG ds005356 │ ← NEW
│ AIS_pre scar: cur>past      │+0.965 │ 0.017  │ Cav EEG      │
│ Boundary: MODMA task null   │+0.091 │ 0.675  │ MODMA task   │
│ Boundary: Hayling null      │+0.018 │ 0.873  │ Hayling      │
└─────────────────────────────┴───────┴────────┴──────────────┘
NOTE: MEG rest = pre-task segment (variable duration, ~40-160s)
""")

# ── Save ──────────────────────────────────────────────────────────────────────
out = ASSETS / "meg_rest_ais_subjects.csv"
df.to_csv(out, index=False)
print(f"Saved: {out}")

state['meg_pretask_rest_ais'] = {
    'date':         '2026-05-03',
    'channel':      CHANNEL,
    'rest_type':    'pre-task segment (t=0 to first_event)',
    'N_CTL':        int(len(ctl)),
    'N_MDD':        int(len(mdd)),
    'CTL_mean':     round(float(ctl.mean()), 4),
    'MDD_mean':     round(float(mdd.mean()), 4),
    'd_ctl_mdd':    round(d_cm, 3),
    'p_ctl_mdd':    round(float(p_cm), 4),
    'direction':    'CTL>MDD' if ctl.mean() > mdd.mean() else 'MDD>CTL',
    'note':         'Not a standardized rest block; duration confound checked',
}

with open(PST / "derivatives/RESEARCH_STATE_FROZEN.json", 'w') as f:
    json.dump(state, f, indent=2)
print("Frozen state updated.")
