"""
TE Follow-up: Three interpretive tests for MDD > CTL absolute TE elevation
  TEST 1 — Reward vs Loss specificity
  TEST 2 — Signal amplitude confound check
  TEST 3 — BDI continuous + AIS_pre anticorrelation
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, wilcoxon, spearmanr
import mne; mne.set_log_level('ERROR')

# ── Paths ────────────────────────────────────────────────────────────────────
PST   = Path("/media/neuraldyn/PortableSSD/DEPRESSION"
             "/01_raw_data/Cavanagh/Depression_PS_Task")
DERIV = PST / "derivatives"
EPO   = DERIV / "epochs"

# ── Load existing data ────────────────────────────────────────────────────────
df_te  = pd.read_csv(DERIV / "te_f1_fcz_subject_level.csv")
df_te['subject_id'] = df_te['subject_id'].astype(int)

ais = pd.read_csv(DERIV / "erp_it_cavanagh/delta_ais_aggregated.csv")
clin = pd.read_csv(DERIV / "clinical_lookup_ps_task.csv")

# Normalize MDD_any → MDD in clin
clin['group'] = clin['analysis_group_broad'].replace({'MDD_any': 'MDD'})

df_te = df_te.merge(ais[['subject_id', 'mean_AIS_pre', 'BDI_Anh']],
                    on='subject_id', how='left')
df_te = df_te.merge(clin[['subject_id', 'BDI']].rename(columns={'BDI': 'BDI_clin'}),
                    on='subject_id', how='left')

print(f"Merged: {len(df_te)} subjects  "
      f"CTL={len(df_te[df_te.group=='CTL'])}, MDD={len(df_te[df_te.group=='MDD'])}")

PRIMARY_TE = 'TE_fwd_F1_to_FCz'

# ── TE function (same vectorised version) ─────────────────────────────────────
def safe_te(source, target, lag=1, n_bins=4):
    x, y = np.asarray(source, float), np.asarray(target, float)
    n = len(x)
    if n != len(y) or n < 3 * lag + 10: return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12: return np.nan
    try:
        xe = np.unique(np.percentile(x, np.linspace(0, 100, n_bins+1)))
        ye = np.unique(np.percentile(y, np.linspace(0, 100, n_bins+1)))
        if len(xe) < 3 or len(ye) < 3: return np.nan
        xb = np.digitize(x, xe[1:-1])
        yb = np.digitize(y, ye[1:-1])
        nb = n_bins
        yt = yb[lag:]; ytlag = yb[:n-lag]; xtlag = xb[:n-lag]
        p3 = np.bincount(yt*nb*nb + ytlag*nb + xtlag,
                         minlength=nb**3).reshape(nb,nb,nb).astype(float)
        p3 /= (p3.sum() + 1e-12)
        p_yy = p3.sum(axis=2)
        p_yl = p_yy.sum(axis=0)
        denom_yyx = p3.sum(axis=0, keepdims=True) + 1e-12
        denom_yy  = p_yl[np.newaxis, :, np.newaxis] + 1e-12
        ratio = (p3 / denom_yyx) / (p_yy[:,:,np.newaxis] / denom_yy + 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.where(ratio > 0, np.log2(ratio), 0.0)
        te = float(np.sum(p3 * log_r))
        return te if np.isfinite(te) else np.nan
    except Exception:
        return np.nan

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1))
                 / (len(a)+len(b)-2) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / sp)

PRE  = (-0.200, 0.000)
F1, FCZ = 'F1', 'FCz'
REWARD_CODE, LOSS_CODE = 94, 104

epo_files = sorted(EPO.glob("sub-*_task-ps_epo.fif"))

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1: REWARD vs LOSS SPECIFICITY")
print("="*60)

records1 = []
for fpath in epo_files:
    sub_id = int(fpath.name.split('_')[0].replace('sub-', ''))
    row_c = clin[clin['subject_id'] == sub_id]
    if len(row_c) == 0: continue
    if row_c.iloc[0].get('excluded', False): continue
    group = row_c.iloc[0]['group']
    if group not in ('CTL', 'MDD'): continue

    try:
        epo  = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        times = epo.times
        pre_m = (times >= PRE[0]) & (times < PRE[1])
        if F1 not in epo.ch_names or FCZ not in epo.ch_names: continue
        fi, ci = epo.ch_names.index(F1), epo.ch_names.index(FCZ)

        rec = {'subject_id': sub_id, 'group': group}
        for label, code in [('reward', REWARD_CODE),
                             ('loss',   LOSS_CODE),
                             ('all',    None)]:
            ev_mask = (epo.events[:, 2] == code) if code else np.ones(len(epo), bool)
            idx = np.where(ev_mask)[0]
            if len(idx) < 5: continue
            data = epo.get_data()[idx]   # (n_sel, n_ch, n_t)
            tf = [safe_te(t[fi, pre_m], t[ci, pre_m]) for t in data]
            tb = [safe_te(t[ci, pre_m], t[fi, pre_m]) for t in data]
            vf = [v for v in tf if np.isfinite(v)]
            vb = [v for v in tb if np.isfinite(v)]
            if len(vf) >= 5:
                rec[f'fwd_{label}'] = np.mean(vf)
                rec[f'bwd_{label}'] = np.mean(vb) if len(vb) >= 5 else np.nan
                rec[f'net_{label}'] = (np.mean(vf) - np.mean(vb)
                                       if len(vb) >= 5 else np.nan)
                rec[f'n_{label}']   = len(vf)
        records1.append(rec)
        if len(records1) % 20 == 0:
            print(f"  {len(records1)} subjects done …")
    except Exception as e:
        print(f"  sub-{sub_id}: {e}")

df1 = pd.DataFrame(records1)
print(f"\nCondition-split TE: {len(df1)} subjects")

# Between-group comparison per condition
print("\n  Condition   d       p        N_CTL  N_MDD  direction")
print("  " + "-"*55)
for label in ('reward', 'loss', 'all'):
    col = f'fwd_{label}'
    if col not in df1.columns: continue
    ctl = df1[df1.group == 'CTL'][col].dropna()
    mdd = df1[df1.group == 'MDD'][col].dropna()
    if len(ctl) < 5 or len(mdd) < 5: continue
    _, p = mannwhitneyu(ctl, mdd, alternative='two-sided')
    d = cohens_d(ctl.values, mdd.values)
    dirn = 'MDD>CTL' if mdd.mean() > ctl.mean() else 'CTL>MDD'
    print(f"  {label:<10}  {d:+.3f}   {p:.4f}   {len(ctl):<5}  {len(mdd):<5}  {dirn}")

# Within-group reward vs loss (paired)
print("\n  REWARD vs LOSS within group (Wilcoxon paired):")
for grp in ('CTL', 'MDD'):
    sub = df1[df1.group == grp][['fwd_reward', 'fwd_loss']].dropna()
    if len(sub) < 5: continue
    diff_mean = (sub.fwd_reward - sub.fwd_loss).mean()
    try:
        _, pw = wilcoxon(sub.fwd_reward, sub.fwd_loss)
    except Exception:
        pw = np.nan
    print(f"    {grp}: reward−loss = {diff_mean:+.5f}, p={pw:.4f}  (N={len(sub)})")

# Specificity index: is the CTL−MDD gap larger in reward or loss?
print("\n  GROUP × CONDITION interaction (reward advantage):")
for grp in ('CTL', 'MDD'):
    sub = df1[df1.group == grp][['fwd_reward', 'fwd_loss']].dropna()
    if len(sub) < 5: continue
    print(f"    {grp}: reward={sub.fwd_reward.mean():.5f}, loss={sub.fwd_loss.mean():.5f}, "
          f"diff={sub.fwd_reward.mean()-sub.fwd_loss.mean():+.5f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: AMPLITUDE CONFOUND")
print("="*60)

amp_records = []
for fpath in epo_files:
    sub_id = int(fpath.name.split('_')[0].replace('sub-', ''))
    try:
        epo  = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        times = epo.times
        pre_m = (times >= PRE[0]) & (times < PRE[1])
        if F1 not in epo.ch_names or FCZ not in epo.ch_names: continue
        fi, ci = epo.ch_names.index(F1), epo.ch_names.index(FCZ)
        data = epo.get_data()   # (n_trials, n_ch, n_t)
        amp_f1  = float(np.abs(data[:, fi, :][:, pre_m]).mean())
        amp_fcz = float(np.abs(data[:, ci, :][:, pre_m]).mean())
        # Also RMS (power proxy)
        rms_f1  = float(np.sqrt((data[:, fi, :][:, pre_m]**2).mean()))
        rms_fcz = float(np.sqrt((data[:, ci, :][:, pre_m]**2).mean()))
        amp_records.append({'subject_id': sub_id,
                            'amp_F1': amp_f1, 'amp_FCz': amp_fcz,
                            'amp_mean': (amp_f1 + amp_fcz) / 2,
                            'rms_mean': (rms_f1 + rms_fcz) / 2})
    except Exception:
        continue

df_amp = pd.DataFrame(amp_records)
df_check = df_te.merge(df_amp, on='subject_id', how='inner')
valid_amp = df_check[[PRIMARY_TE, 'amp_mean', 'rms_mean', 'group']].dropna()

r_amp,  p_amp  = pearsonr(valid_amp.amp_mean, valid_amp[PRIMARY_TE])
r_rms,  p_rms  = pearsonr(valid_amp.rms_mean, valid_amp[PRIMARY_TE])
r_s,    p_s    = spearmanr(valid_amp.amp_mean, valid_amp[PRIMARY_TE])

print(f"\n  r(|amplitude| F1+FCz, TE_fwd) = {r_amp:.3f}, p={p_amp:.4f}  (Pearson)")
print(f"  r(RMS F1+FCz,         TE_fwd) = {r_rms:.3f}, p={p_rms:.4f}  (Pearson)")
print(f"  ρ(|amplitude|,        TE_fwd) = {r_s:.3f}, p={p_s:.4f}  (Spearman)")

# Within-group correlations
for grp in ('CTL', 'MDD'):
    sub = valid_amp[valid_amp.group == grp]
    if len(sub) < 5: continue
    r, p = pearsonr(sub.amp_mean, sub[PRIMARY_TE])
    print(f"    Within {grp}: r={r:.3f}, p={p:.4f}  (N={len(sub)})")

# Median amplitude CTL vs MDD
ctl_amp = valid_amp[valid_amp.group=='CTL']['amp_mean']
mdd_amp = valid_amp[valid_amp.group=='MDD']['amp_mean']
_, p_grp = mannwhitneyu(ctl_amp, mdd_amp, alternative='two-sided')
d_amp = cohens_d(ctl_amp.values, mdd_amp.values)
print(f"\n  CTL amp={ctl_amp.mean()*1e6:.2f}µV vs MDD amp={mdd_amp.mean()*1e6:.2f}µV")
print(f"  Group amplitude diff: d={d_amp:.3f}, p={p_grp:.4f}")

if abs(r_amp) > 0.5:
    print("\n  *** HIGH correlation — TE elevation may be amplitude-driven ***")
    print("  Recommendation: partial out amplitude or normalise signal")
elif abs(r_amp) > 0.3:
    print("\n  ⚠  Moderate correlation — check amplitude-partialled TE")
else:
    print("\n  Low correlation — TE elevation is not simply amplitude artifact")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3: BDI CONTINUOUS + AIS_pre ANTICORRELATION")
print("="*60)

valid3 = df_te[[PRIMARY_TE, 'BDI', 'BDI_Anh', 'mean_AIS_pre', 'group']].dropna(
    subset=[PRIMARY_TE, 'BDI', 'mean_AIS_pre'])

print(f"\n  N for correlations: {len(valid3)}")

r_bdi,  p_bdi  = pearsonr(valid3[PRIMARY_TE], valid3.BDI)
r_banh, p_banh = pearsonr(valid3[PRIMARY_TE],
                          valid3.BDI_Anh.fillna(0))
r_ais,  p_ais  = pearsonr(valid3[PRIMARY_TE], valid3.mean_AIS_pre)
# Spearman too
rs_bdi, ps_bdi = spearmanr(valid3[PRIMARY_TE], valid3.BDI)
rs_ais, ps_ais = spearmanr(valid3[PRIMARY_TE], valid3.mean_AIS_pre)

print(f"\n  r(TE_fwd_F1→FCz, BDI)       = {r_bdi:+.3f}, p={p_bdi:.4f}  (ρ={rs_bdi:+.3f}, p={ps_bdi:.4f})")
print(f"  r(TE_fwd_F1→FCz, BDI_Anh)   = {r_banh:+.3f}, p={p_banh:.4f}")
print(f"  r(TE_fwd_F1→FCz, AIS_pre)   = {r_ais:+.3f}, p={p_ais:.4f}  (ρ={rs_ais:+.3f}, p={ps_ais:.4f})")

# AIS_pre direction check: low AIS = high TE?
print(f"\n  [Expected if scenario 4: r(TE, AIS_pre) < 0]")
if r_ais < -0.2 and p_ais < 0.1:
    print(f"  ✅ ANTICORRELATION: low AIS_pre ↔ high TE_fwd  (r={r_ais:.3f})")
    print(f"     Interpretation: same subjects = low local coherence + high spatial coupling")
    print(f"     These are COMPLEMENTARY faces of the same dysfunction")
elif r_ais > 0.2:
    print(f"  → POSITIVE: high AIS + high TE (correlated measures, not independent)")
else:
    print(f"  → NEAR ZERO: AIS_pre and TE_fwd are orthogonal measures")

# Within MDD (dimensional test)
mdd_sub = valid3[valid3.group == 'MDD']
if len(mdd_sub) >= 8:
    r_m, p_m = pearsonr(mdd_sub[PRIMARY_TE], mdd_sub.BDI)
    r_am, p_am = pearsonr(mdd_sub[PRIMARY_TE], mdd_sub.mean_AIS_pre)
    print(f"\n  Within MDD only (N={len(mdd_sub)}):")
    print(f"    r(TE_fwd, BDI)     = {r_m:+.3f}, p={p_m:.4f}")
    print(f"    r(TE_fwd, AIS_pre) = {r_am:+.3f}, p={p_am:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("INTEGRATED INTERPRETATION")
print("="*60)

# Determine dominant scenario
print(f"\nKey numbers:")
print(f"  Primary TE effect:  d=-0.601, p=0.009 (MDD>CTL)")
print(f"  Net TE (asymmetry): d=+0.392, p=0.176 (null)")
print(f"  r(TE, amplitude)  = {r_amp:.3f} (p={p_amp:.4f})")
print(f"  r(TE, AIS_pre)    = {r_ais:.3f} (p={p_ais:.4f})")
print(f"  r(TE, BDI)        = {r_bdi:.3f} (p={p_bdi:.4f})")

# Reward vs loss summary
if 'fwd_reward' in df1.columns and 'fwd_loss' in df1.columns:
    ctl_r = df1[df1.group=='CTL'][['fwd_reward','fwd_loss']].dropna()
    mdd_r = df1[df1.group=='MDD'][['fwd_reward','fwd_loss']].dropna()
    # interaction: CTL-MDD gap in reward vs loss
    if len(ctl_r) > 3 and len(mdd_r) > 3:
        gap_rew = ctl_r.fwd_reward.mean() - mdd_r.fwd_reward.mean()
        gap_los = ctl_r.fwd_loss.mean()   - mdd_r.fwd_loss.mean()
        print(f"  CTL−MDD gap in reward = {gap_rew:+.5f}")
        print(f"  CTL−MDD gap in loss   = {gap_los:+.5f}")
        interaction = abs(gap_rew) - abs(gap_los)
        print(f"  Reward-specificity index = {interaction:+.5f}  "
              f"({'reward-specific' if interaction > 0.001 else 'non-specific'})")

print("\nScenario verdict:")
if abs(r_amp) > 0.5:
    print("  → SCENARIO 2 (amplitude confound) — TE elevation is artifactual")
elif r_ais < -0.25 and p_ais < 0.05:
    print("  → SCENARIO 4 (anticorrelation) — low AIS + high TE = single neural phenotype")
    print("     Local temporal incoherence + global spatial synchrony in MDD")
elif r_bdi > 0.25 and p_bdi < 0.05:
    print("  → SCENARIO 3 (dimensional) — chronic frontal hypersynchrony, BDI-linked")
else:
    print("  → SCENARIO 3 (condition-independent) — inspect reward vs loss gap above")
    print("     If reward-specific: DMN suppression failure during reward anticipation")
    print("     If non-specific: chronic ruminative synchrony, not reward-contingent")
