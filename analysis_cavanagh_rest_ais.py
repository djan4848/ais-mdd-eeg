"""
ds003478 AIS_rest — SAME subjects as PST task (sub-507..628)
Four questions: Q1 CTL>MDD, Q2 scar, Q3 r(rest,task), Q4 incremental
- sub-NNN ↔ Original_ID ↔ PST subject 507..628
- Channel: FCZ (uppercase, 67-ch cap)
- Runs: run-01 + run-02 concatenated (~1068s / ~534 2s-windows)
"""

import warnings; warnings.filterwarnings('ignore')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
import mne; mne.set_log_level('ERROR')
from numpy.linalg import lstsq

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE  = Path("/media/neuraldyn/PortableSSD/DEPRESSION")
PST   = BASE / "01_raw_data/Cavanagh/Depression_PS_Task"
REST  = BASE / "01_raw_data/Cavanagh/ds003478"
DERIV = PST / "derivatives"
freeze_file = DERIV / "RESEARCH_STATE_FROZEN.json"

# ── Frozen state ───────────────────────────────────────────────────────────────
with open(freeze_file) as f:
    state = json.load(f)
print("=== KEY CONTEXT ===")
print("AIS_pre (PST task, same subjects): CTL>MDD, d=+0.817, p=0.0003")
print("AIS_rest (TDBRAIN, different N):   CTL>>MDD, d=+2.024, p≈0")
print("Target: AIS_rest in ds003478 with SAME subjects as PST\n")

# ── Subject ID mapping ─────────────────────────────────────────────────────────
# ds003478 participant_id (sub-001..sub-122) → Original_ID (507..628) → SCID group
parts = pd.read_csv(REST / "participants.tsv", sep='\t')
clin  = pd.read_csv(DERIV / "clinical_lookup_ps_task.csv")
ais_task = pd.read_csv(DERIV / "erp_it_cavanagh/delta_ais_aggregated.csv")

# Merge to get SCID group for each ds003478 subject
mapping = parts.merge(
    clin[['subject_id', 'scid_group', 'analysis_group_broad',
          'excluded', 'BDI']],
    left_on='Original_ID', right_on='subject_id', how='left')

# Exclude excluded and ANX_other subjects
mapping = mapping[
    (mapping['excluded'] == False) &
    (mapping['scid_group'] != 'ANX_other')
].copy()

# Normalize group names
mapping['group'] = mapping['scid_group'].replace(
    {'CTL': 'CTL', 'MDD_current': 'MDD_current', 'MDD_past': 'MDD_past'})

print(f"Subjects after exclusions: {len(mapping)}")
print(f"Groups: {mapping['group'].value_counts().to_dict()}")

# Build lookup: ds003478 participant_id → {group, orig_id}
sub_info = {row['participant_id']: {'group': row['group'],
                                     'orig_id': int(row['Original_ID'])}
            for _, row in mapping.iterrows()}

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

rng = np.random.default_rng(42)
ar1 = np.zeros(500); [ar1.__setitem__(i, 0.9*ar1[i-1]+0.1*rng.standard_normal()) for i in range(1,500)]
assert safe_ais(ar1) > safe_ais(rng.standard_normal(500)), "AIS validation FAILED"
print("AIS validation OK\n")

FCZ = 'FCZ'   # uppercase in this dataset
WINDOW_SEC = 2.0
MIN_WIN    = 10

# ── Main loop: run-01 + run-02 concatenated ────────────────────────────────────
print(f"=== COMPUTING AIS_rest — {len(sub_info)} subjects ===")
records = []

for ds_id, info in sorted(sub_info.items()):
    group   = info['group']
    orig_id = info['orig_id']

    # Collect available run files
    run_files = sorted((REST / ds_id / "eeg").glob(f"{ds_id}_task-Rest_run-*_eeg.set"))
    if not run_files:
        continue

    try:
        # Load runs; concatenate only if channel counts match
        raws = []
        for rf in run_files:
            r = mne.io.read_raw_eeglab(rf, preload=True, verbose='ERROR')
            raws.append(r)
        if len(raws) == 1:
            raw = raws[0]
        elif all(r.info['nchan'] == raws[0].info['nchan'] for r in raws):
            raw = mne.concatenate_raws(raws)
        else:
            # Channel count mismatch across runs — use the longest run
            raw = max(raws, key=lambda r: r.n_times)

        # Filter
        raw.filter(1., 40., method='fir', verbose='ERROR')

        if FCZ not in raw.ch_names:
            print(f"  {ds_id}: FCZ missing — channels: {raw.ch_names[:5]}")
            continue

        sig   = raw.get_data(picks=[FCZ])[0]
        sfreq = raw.info['sfreq']

        if np.std(sig) < 1e-8: continue
        if np.isnan(sig).any() or np.isinf(sig).any(): continue

        win_len = int(WINDOW_SEC * sfreq)
        n_wins  = len(sig) // win_len
        if n_wins < MIN_WIN: continue

        ais_vals = [safe_ais(sig[i*win_len:(i+1)*win_len]) for i in range(n_wins)]
        valid    = [v for v in ais_vals if np.isfinite(v)]
        if len(valid) < MIN_WIN: continue

        mean_ais = float(np.mean(valid))
        records.append({'participant_id': ds_id, 'orig_id': orig_id,
                        'group': group, 'AIS_rest': mean_ais,
                        'n_windows': len(valid)})
        print(f"  {ds_id} (orig={orig_id}, {group}): AIS_rest={mean_ais:.4f}  [{len(valid)} win]")
    except Exception as e:
        print(f"  {ds_id} ERROR: {e}")

df = pd.DataFrame(records)
print(f"\nProcessed: {len(df)} subjects")
print(df.groupby('group')['AIS_rest'].agg(['mean','std','count']).round(4))

# ── Cohen's d ──────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1))
                 / (len(a)+len(b)-2) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / sp)

# ══ Q1: CTL vs MDD ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Q1: CTL vs MDD in AIS_rest (ds003478, SAME subjects as PST)")
print("Expected direction: CTL > MDD (same as TDBRAIN)")
print("="*60)

ctl_r  = df[df.group == 'CTL']['AIS_rest'].dropna()
mdd_r  = df[df.group.isin(['MDD_current','MDD_past'])]['AIS_rest'].dropna()

_, p_q1 = mannwhitneyu(ctl_r, mdd_r, alternative='two-sided')
d_q1    = cohens_d(ctl_r.values, mdd_r.values)

print(f"\nCTL: {ctl_r.mean():.4f} ± {ctl_r.std():.4f}  (N={len(ctl_r)})")
print(f"MDD: {mdd_r.mean():.4f} ± {mdd_r.std():.4f}  (N={len(mdd_r)})")
print(f"d = {d_q1:+.3f}, p = {p_q1:.4f}  "
      f"[{'CTL>MDD ✅' if ctl_r.mean()>mdd_r.mean() else 'MDD>CTL ❌'}]")

print(f"\nCross-reference:")
print(f"  AIS_pre (task, same subjects):  CTL>MDD, d=+0.817, p=0.0003")
print(f"  AIS_rest (TDBRAIN, diff cohort): CTL>MDD, d=+2.024, p≈0")
print(f"  AIS_rest (rest, same subjects):  CTL>MDD, d={d_q1:+.3f}, p={p_q1:.4f}")

# ══ Q2: SCID phase pattern ═════════════════════════════════════════════════════
print("\n" + "="*60)
print("Q2: SCID phase — scar in resting state?")
print("="*60)

curr_r = df[df.group == 'MDD_current']['AIS_rest'].dropna()
past_r = df[df.group == 'MDD_past']['AIS_rest'].dropna()

print(f"\nCTL:         {ctl_r.mean():.4f} ± {ctl_r.std():.4f}  (N={len(ctl_r)})")
print(f"MDD_current: {curr_r.mean():.4f} ± {curr_r.std():.4f}  (N={len(curr_r)})")
print(f"MDD_past:    {past_r.mean():.4f} ± {past_r.std():.4f}  (N={len(past_r)})")

if len(curr_r) >= 3 and len(past_r) >= 3:
    _, p_cp = mannwhitneyu(curr_r, past_r, alternative='two-sided')
    d_cp    = cohens_d(curr_r.values, past_r.values)
    _, p_cp_ctl = mannwhitneyu(ctl_r, past_r, alternative='two-sided')
    d_cp_ctl = cohens_d(ctl_r.values, past_r.values)
    print(f"\nMDD_current vs MDD_past:  d={d_cp:+.3f}, p={p_cp:.4f}")
    print(f"CTL vs MDD_past:          d={d_cp_ctl:+.3f}, p={p_cp_ctl:.4f}")
    print(f"\nTask AIS_pre scar: CTL(1.112)>current(1.072)>past(0.989), d=+0.965")
    print(f"Rest AIS_rest: CTL({ctl_r.mean():.3f}) vs current({curr_r.mean():.3f}) vs past({past_r.mean():.3f})")
    if ctl_r.mean() > curr_r.mean() > past_r.mean():
        print("→ GRADIENT in rest too: same scar pattern")
    elif curr_r.mean() > past_r.mean():
        print("→ current > past in rest (same direction as task scar)")
    else:
        print("→ current < past in rest (OPPOSITE direction: scar absent in rest)")

# ══ Q3: Within-subject correlation ════════════════════════════════════════════
print("\n" + "="*60)
print("Q3: r(AIS_rest, AIS_pre_task) — same subjects")
print("Critical test: are they measuring the same thing?")
print("="*60)

# Merge on orig_id → PST subject_id
df_merged = df.merge(
    ais_task[['subject_id', 'mean_AIS_pre']].rename(
        columns={'subject_id': 'orig_id'}),
    on='orig_id', how='inner')

print(f"\nMatched subjects: {len(df_merged)}")
valid_corr = df_merged[['AIS_rest', 'mean_AIS_pre', 'group']].dropna()

r_all, p_all = pearsonr(valid_corr.AIS_rest, valid_corr.mean_AIS_pre)
rs_all, _    = spearmanr(valid_corr.AIS_rest, valid_corr.mean_AIS_pre)
print(f"\nAll subjects (N={len(valid_corr)}): r={r_all:+.3f}, p={p_all:.4f}  (ρ={rs_all:+.3f})")
print(f"Shared variance: {r_all**2*100:.0f}%  ('same underlying trait' threshold: >25%)")

for grp in ('CTL', 'MDD_current', 'MDD_past'):
    sub = valid_corr[valid_corr.group == grp]
    if len(sub) < 5: continue
    r_g, p_g = pearsonr(sub.AIS_rest, sub.mean_AIS_pre)
    print(f"  {grp} (N={len(sub)}): r={r_g:+.3f}, p={p_g:.4f}")

print(f"\nInterpretation guide:")
if abs(r_all) > 0.50:
    print(f"  r={r_all:.3f} → HIGH: AIS_rest and AIS_pre reflect same trait")
    print(f"  Resting temporal coherence predicts task temporal coherence")
    print(f"  One unified biomarker — reward context modulates magnitude")
elif abs(r_all) > 0.25:
    print(f"  r={r_all:.3f} → MODERATE: partially shared (~{r_all**2*100:.0f}%)")
    print(f"  Task context adds genuine specificity beyond resting state")
elif abs(r_all) > 0.10:
    print(f"  r={r_all:.3f} → LOW: mostly independent measures")
    print(f"  Reward anticipation (AIS_pre) is NOT just resting state AIS")
    print(f"  The PST finding is NOT explained by resting brain baseline")
    print(f"  Boundary condition (reward context) is real and necessary")
else:
    print(f"  r={r_all:.3f} → NEAR ZERO: fully independent measures")
    print(f"  AIS_pre is a unique task-specific biomarker")

# ══ Q4: Incremental validity ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("Q4: Incremental validity — does AIS_rest add beyond AIS_pre?")
print("="*60)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

df_bin = df_merged[df_merged.group.isin(
    ['CTL', 'MDD_current', 'MDD_past'])].copy()
df_bin['is_mdd'] = (df_bin.group != 'CTL').astype(int)
valid_bin = df_bin[['AIS_rest', 'mean_AIS_pre', 'is_mdd']].dropna()

if len(valid_bin) >= 20:
    X   = valid_bin[['AIS_rest', 'mean_AIS_pre']].values
    y   = valid_bin['is_mdd'].values
    sc  = StandardScaler()
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr  = LogisticRegression(C=1.0, random_state=42, max_iter=200)

    auc_pre  = cross_val_score(lr, sc.fit_transform(X[:, 1:2]), y, cv=cv, scoring='roc_auc').mean()
    auc_rest = cross_val_score(lr, sc.fit_transform(X[:, 0:1]), y, cv=cv, scoring='roc_auc').mean()
    auc_both = cross_val_score(lr, sc.fit_transform(X),         y, cv=cv, scoring='roc_auc').mean()

    print(f"\n  AUC (AIS_pre  alone):  {auc_pre:.3f}")
    print(f"  AUC (AIS_rest alone):  {auc_rest:.3f}")
    print(f"  AUC (both combined):   {auc_both:.3f}")
    delta = auc_both - max(auc_pre, auc_rest)
    print(f"  Additive gain:         {delta:+.3f}")
    if delta > 0.03:
        print("  → COMPLEMENTARY: rest + task together improve CTL/MDD discrimination")
    elif delta > 0:
        print("  → MARGINAL gain from combining")
    else:
        print("  → REDUNDANT: one measure is sufficient")

# ══ Summary table ══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("COMPLETE AIS EVIDENCE TABLE")
print("="*65)
print(f"""
  Measure/Dataset          d        p       Direction   N
  ─────────────────────────────────────────────────────
  AIS_pre task (Cav EEG)  +0.817   0.0003  CTL>MDD     87+23
  AIS_rest TDBRAIN        +2.024   ≈0      CTL>MDD     47+121
  AIS_rest ds003478       {d_q1:+.3f}   {p_q1:.4f}  {"CTL>MDD ✅" if ctl_r.mean()>mdd_r.mean() else "MDD>CTL ❌"}  {len(ctl_r)}+{len(mdd_r)}
  AIS_pre scar (cur>past) +0.965   0.017   cur>past    11+12
  Hayling null            +0.018   0.873   —           22+25
  MODMA null              +0.091   0.675   —           29+24
""")

# ── Save and update state ──────────────────────────────────────────────────────
df.to_csv(DERIV / "cavanagh_rest_ais_subjects.csv", index=False)
df_merged.to_csv(DERIV / "cavanagh_rest_task_merged.csv", index=False)

state['cavanagh_resting_ds003478'] = {
    'date': '2026-05-03',
    'same_subjects_as_PST': True,
    'N_CTL': int(len(ctl_r)), 'N_MDD': int(len(mdd_r)),
    'CTL_mean': round(float(ctl_r.mean()), 4),
    'MDD_mean': round(float(mdd_r.mean()), 4),
    'Q1_d': round(d_q1, 3), 'Q1_p': round(float(p_q1), 4),
    'Q1_direction': 'CTL>MDD' if ctl_r.mean() > mdd_r.mean() else 'MDD>CTL',
    'Q2_current_vs_past_d': round(d_cp, 3) if len(curr_r)>=3 and len(past_r)>=3 else None,
    'Q3_r_rest_task': round(r_all, 3), 'Q3_p': round(p_all, 4),
    'Q3_R2_shared_pct': round(r_all**2*100, 1),
    'Q4_AUC_pre': round(auc_pre, 3) if 'auc_pre' in dir() else None,
    'Q4_AUC_rest': round(auc_rest, 3) if 'auc_rest' in dir() else None,
    'Q4_AUC_both': round(auc_both, 3) if 'auc_both' in dir() else None,
}
with open(freeze_file, 'w') as f:
    json.dump(state, f, indent=2)
print("Saved. Frozen state updated.")
