"""
TDBRAIN AIS_rest: MDD vs CTL comparison (missing control)
  - MDD: 121 subjects from tdbrain_chaos_measures.csv (already computed)
  - CTL: 47 HEALTHY subjects (indication='HEALTHY') — compute fresh
  - Three-way: CTL vs Remitters vs Non-remitters
"""

import warnings; warnings.filterwarnings('ignore')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
import mne; mne.set_log_level('ERROR')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path("/media/neuraldyn/PortableSSD/DEPRESSION")
PST     = BASE / "01_raw_data/Cavanagh/Depression_PS_Task"
TDBRAIN = Path("/media/neuraldyn/PortableSSD/TDBRAIN_derivatives")
ASSETS  = BASE / "06_manuscript_assets"
freeze_file = PST / "derivatives/RESEARCH_STATE_FROZEN.json"

# ── AIS function (identical to primary pipeline) ───────────────────────────────
def safe_ais(x, lag=1, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < 2*lag + 10: return np.nan
    if np.std(x) < 1e-12: return np.nan
    try:
        edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins+1)))
        if len(edges) < 3: return np.nan
        bins = np.digitize(x, edges[1:-1])
        x_t, x_lag = bins[lag:], bins[:-lag]
        joint = np.zeros((n_bins, n_bins))
        for a, b in zip(x_t, x_lag): joint[a-1, b-1] += 1
        joint /= joint.sum() + 1e-10
        px_t = joint.sum(axis=1); px_lag = joint.sum(axis=0)
        mi = sum(joint[i,j] * np.log2(joint[i,j]/(px_t[i]*px_lag[j]))
                 for i in range(n_bins) for j in range(n_bins)
                 if joint[i,j]>0 and px_t[i]>0 and px_lag[j]>0)
        return float(mi) if np.isfinite(mi) else np.nan
    except Exception:
        return np.nan

# ── Cohen's d ──────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1))
                 / (len(a)+len(b)-2) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / sp)

# ── STEP 0 — Load existing MDD results ────────────────────────────────────────
chaos_file = TDBRAIN / "derivatives/tdbrain_chaos_measures.csv"
df_mdd = pd.read_csv(chaos_file)
df_mdd['group'] = 'MDD'
ais_col = 'ais_rest'
remit_col = 'remitter'

print("=== STEP 0: MDD data loaded ===")
print(f"  N_MDD = {len(df_mdd)}")
print(f"  AIS_rest MDD = {df_mdd[ais_col].mean():.4f} ± {df_mdd[ais_col].std():.4f}")
print(f"  Remitters: {(df_mdd[remit_col]==1).sum()}, "
      f"Non-remitters: {(df_mdd[remit_col]==0).sum()}")

# ── STEP 1 — Identify HEALTHY CTL subjects ────────────────────────────────────
parts = pd.read_csv(TDBRAIN / "TDBRAIN_participants_V2.tsv", sep='\t')
# Column is 'participants_ID', not 'participant_id'
id_col = 'participants_ID'

healthy = parts[parts['indication'] == 'HEALTHY']
print(f"\n=== STEP 1: CTL identification ===")
print(f"  HEALTHY subjects: {len(healthy)}")
print(f"  Example IDs: {healthy[id_col].head(3).tolist()}")
ctl_ids = set(healthy[id_col].astype(str))

# ── STEP 2 — Compute AIS_rest for CTL subjects ────────────────────────────────
WINDOW_SEC = 2.0   # exactly matching MDD pipeline
MIN_WIN    = 10
FCZ        = 'FCz'

print(f"\n=== STEP 2: Computing AIS_rest for {len(ctl_ids)} CTL subjects ===")

ctl_records = []
for sub_id in sorted(ctl_ids):
    fpath = (TDBRAIN / sub_id / "ses-1" / "eeg" /
             f"{sub_id}_ses-1_task-restEC_eeg.vhdr")
    if not fpath.exists():
        print(f"  {sub_id}: file missing")
        continue
    try:
        raw   = mne.io.read_raw_brainvision(fpath, preload=True, verbose='ERROR')
        raw.filter(1., 40., method='fir', verbose='ERROR')
        if FCZ not in raw.ch_names: continue
        sig   = raw.get_data(picks=[FCZ])[0]
        sfreq = raw.info['sfreq']

        if np.std(sig) < 1e-8: continue
        if np.isnan(sig).any() or np.isinf(sig).any(): continue

        win_len  = int(WINDOW_SEC * sfreq)
        n_wins   = len(sig) // win_len
        if n_wins < MIN_WIN: continue

        ais_vals = [safe_ais(sig[i*win_len:(i+1)*win_len]) for i in range(n_wins)]
        valid    = [v for v in ais_vals if np.isfinite(v)]
        if len(valid) < MIN_WIN: continue

        mean_ais = float(np.mean(valid))
        ctl_records.append({'subject_id': sub_id, 'group': 'CTL',
                            ais_col: mean_ais, 'n_windows': len(valid)})
        print(f"  {sub_id}: AIS_rest={mean_ais:.4f}  ({len(valid)} windows)")
    except Exception as e:
        print(f"  {sub_id}: ERROR — {e}")

df_ctl = pd.DataFrame(ctl_records)
print(f"\nCTL processed: {len(df_ctl)}/{len(ctl_ids)}")
print(f"AIS_rest CTL = {df_ctl[ais_col].mean():.4f} ± {df_ctl[ais_col].std():.4f}")

# ── STEP 3 — PRIMARY: MDD vs CTL ──────────────────────────────────────────────
print("\n" + "="*55)
print("PRIMARY: AIS_rest MDD vs CTL (TDBRAIN)")
print("="*55)

mdd_ais = df_mdd[ais_col].dropna()
ctl_ais = df_ctl[ais_col].dropna()

_, p_mc = mannwhitneyu(mdd_ais, ctl_ais, alternative='two-sided')
d_mc    = cohens_d(mdd_ais.values, ctl_ais.values)
dirn_mc = 'MDD>CTL' if mdd_ais.mean() > ctl_ais.mean() else 'CTL>MDD'

print(f"\nMDD: {mdd_ais.mean():.4f} ± {mdd_ais.std():.4f}  (N={len(mdd_ais)})")
print(f"CTL: {ctl_ais.mean():.4f} ± {ctl_ais.std():.4f}  (N={len(ctl_ais)})")
print(f"d = {d_mc:+.3f}, p = {p_mc:.4f}  [{dirn_mc}]")

print(f"\nCross-reference:")
print(f"  AIS_pre (Cavanagh task, CTL>MDD):  d=+0.817 — reward anticipation deficit")
print(f"  AIS_rest (TDBRAIN, ?):             d={d_mc:+.3f} — resting temporal dynamics")
if mdd_ais.mean() > ctl_ais.mean():
    print(f"\n  MDD has HIGHER AIS_rest than CTL — MORE temporally rigid at rest")
    print(f"  Consistent with deep attractor model:")
    print(f"    Resting: MDD more rigid (↑AIS_rest)")
    print(f"    Task:    MDD less prepared (↓AIS_pre)")
    print(f"    Same attractor — different observables")
else:
    print(f"\n  MDD has LOWER AIS_rest than CTL — LESS temporally regular at rest")
    print(f"  Inconsistent with rigidity hypothesis")
    print(f"  Possible: MDD resting is more fragmented/variable")

# ── STEP 4 — THREE-WAY: CTL / Remitters / Non-remitters ──────────────────────
print("\n" + "="*55)
print("THREE-WAY: CTL vs Remitters vs Non-remitters")
print("="*55)

rem  = df_mdd[df_mdd[remit_col] == 1][ais_col].dropna()
nrem = df_mdd[df_mdd[remit_col] == 0][ais_col].dropna()
ctl  = df_ctl[ais_col].dropna()

print(f"\nCTL:              {ctl.mean():.4f} ± {ctl.std():.4f}  (N={len(ctl)})")
print(f"MDD Remitters:    {rem.mean():.4f} ± {rem.std():.4f}  (N={len(rem)})")
print(f"MDD Non-rem:      {nrem.mean():.4f} ± {nrem.std():.4f}  (N={len(nrem)})")

pairs = [('CTL',     ctl,  'Remitters', rem),
         ('CTL',     ctl,  'Non-rem',   nrem),
         ('Rem',     rem,  'Non-rem',   nrem)]
print()
for n1, v1, n2, v2 in pairs:
    _, p12 = mannwhitneyu(v1, v2, alternative='two-sided')
    d12    = cohens_d(v1.values, v2.values)
    print(f"  {n1:<12} vs {n2:<12}: d={d12:+.3f}, p={p12:.4f}")

# Pattern analysis
gradient_up   = ctl.mean() < rem.mean() < nrem.mean()
gradient_down = ctl.mean() > rem.mean() > nrem.mean()
treatment_eff = (ctl.mean() < rem.mean()) and abs(rem.mean() - nrem.mean()) < 0.03
diag_eff      = (nrem.mean() - ctl.mean()) > 0.05

print(f"\nPattern check:")
print(f"  CTL={ctl.mean():.4f} → Rem={rem.mean():.4f} → Non-rem={nrem.mean():.4f}")
if gradient_up:
    print("  ✅ GRADIENT (CTL < Rem < Non-rem): continuous spectrum")
    print("     Healthy → treatable MDD → treatment-resistant MDD")
    print("     AIS_rest = severity marker along a single axis")
elif treatment_eff:
    print("  ✅ TREATMENT EFFECT (CTL≈Rem < Non-rem): rTMS normalises")
    print("     Remitters return to CTL-like resting dynamics")
    print("     Non-remitters remain rigid")
elif diag_eff:
    print("  → DIAGNOSIS EFFECT: MDD elevated regardless of remission")
    print("     AIS_rest distinguishes MDD from healthy more than remission")
else:
    print("  → NULL/MIXED: no clear gradient or group pattern")

# ── Save ──────────────────────────────────────────────────────────────────────
df_combined = pd.concat([
    df_ctl[['subject_id', 'group', ais_col]].assign(remitter=np.nan),
    df_mdd[['subject_id', 'group', ais_col, remit_col]].rename(
        columns={remit_col: 'remitter'})
], ignore_index=True)

out_file = TDBRAIN / "derivatives/tdbrain_ais_mdd_ctl.csv"
df_combined.to_csv(out_file, index=False)
print(f"\nSaved: {out_file.name}  (N={len(df_combined)})")

# ── Update frozen state ────────────────────────────────────────────────────────
with open(freeze_file) as f:
    state = json.load(f)

state['tdbrain_mdd_vs_ctl'] = {
    'date': '2026-05-03',
    'N_mdd': int(len(mdd_ais)),
    'N_ctl': int(len(ctl_ais)),
    'MDD_mean': round(float(mdd_ais.mean()), 4),
    'CTL_mean': round(float(ctl_ais.mean()), 4),
    'd_mdd_vs_ctl': round(d_mc, 3),
    'p_mdd_vs_ctl': round(float(p_mc), 4),
    'direction': dirn_mc,
    'gradient_confirmed': bool(gradient_up),
    'three_way': {
        'CTL_mean':  round(float(ctl.mean()), 4),
        'Rem_mean':  round(float(rem.mean()), 4),
        'Nrem_mean': round(float(nrem.mean()), 4),
    }
}

with open(freeze_file, 'w') as f:
    json.dump(state, f, indent=2)
print("Frozen state updated.")
