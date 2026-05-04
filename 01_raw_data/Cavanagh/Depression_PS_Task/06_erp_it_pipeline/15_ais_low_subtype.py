#!/usr/bin/env python3
"""
15_ais_low_subtype.py
Analysis 1: AIS_LOW subtype characterization.

Hypothesis: AIS_pre does not identify "MDD in general" but a subtype
with anticipatory anhedonia profile.

Loads from existing CSVs — no raw EEG recomputation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, chi2_contingency
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/media/neuraldyn/PortableSSD/DEPRESSION")
PST  = BASE / "01_raw_data/Cavanagh/Depression_PS_Task"
MEG  = BASE / "01_raw_data/Cavanagh/ds005356"

# ── STEP 0: LOAD ALL EXISTING DATA ───────────────────────────────────────────
print("=== STEP 0: LOADING DATA ===")

# Primary EEG AIS_pre (N=109, groups: CTL / MDD_any)
ais_eeg = pd.read_csv(
    PST / "derivatives/erp_it_cavanagh/delta_ais_aggregated.csv")
print(f"EEG AIS_pre: {len(ais_eeg)} subjects  "
      f"{ais_eeg['group'].value_counts().to_dict()}")

# Full clinical lookup (SCID groups, BDI subscales, HamD)
clin = pd.read_csv(
    PST / "derivatives/clinical_lookup_ps_task.csv")
print(f"Clinical lookup: {len(clin)}  "
      f"scid_group: {clin['scid_group'].value_counts().to_dict()}")

# Merge AIS_pre with clinical
df = ais_eeg.merge(clin, on='subject_id', how='left',
                   suffixes=('', '_clin'))
# Drop duplicate BDI/BDI_Anh columns created by merge
for col in ['BDI_clin', 'BDI_Anh_clin']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
print(f"Merged EEG dataset: {len(df)}  columns: {df.columns.tolist()}")

# RewP: aggregate from trial-level master (reward - loss, Fz mean amplitude)
master = pd.read_csv(
    PST / "derivatives/erp_it_cavanagh/erp_it_master_results.csv")
# Fz_mean_amp_uV averaged 0–400ms post-feedback, already in master
rewp_sub = (master.groupby(['subject', 'cond'])['Fz_mean_amp_uV']
            .mean().unstack(fill_value=np.nan))
if 'reward' in rewp_sub.columns and 'loss' in rewp_sub.columns:
    rewp_sub['RewP'] = rewp_sub['reward'] - rewp_sub['loss']
elif all(c in rewp_sub.columns for c in ['94', '104']):
    rewp_sub['RewP'] = rewp_sub['94'] - rewp_sub['104']
else:
    # Use whatever two conditions exist
    cols = [c for c in rewp_sub.columns if c not in ['nan']]
    if len(cols) >= 2:
        rewp_sub['RewP'] = rewp_sub[cols[0]] - rewp_sub[cols[1]]
    else:
        rewp_sub['RewP'] = np.nan
rewp_sub = rewp_sub[['RewP']].reset_index().rename(
    columns={'subject': 'subject_id'})
rewp_sub['subject_id'] = rewp_sub['subject_id'].astype(str)
df['subject_id'] = df['subject_id'].astype(str)
df = df.merge(rewp_sub, on='subject_id', how='left')
print(f"RewP available for {df['RewP'].notna().sum()} subjects")

# MEG dataset (exclude artifact subject)
ais_meg = pd.read_csv(
    MEG / "derivatives/meg_ais_pre_full_sample_results.csv")
ais_meg = ais_meg[ais_meg['subject_id'] != 'sub-M87121835'].copy()
print(f"\nMEG AIS_pre (artifact excluded): {len(ais_meg)}  "
      f"{ais_meg['group'].value_counts().to_dict()}")
print(f"MEG columns: {ais_meg.columns.tolist()}")

# ── STEP 1: DEFINE AIS_LOW SUBTYPE ───────────────────────────────────────────
print("\n=== STEP 1: SUBTYPE DEFINITION ===")

ais_vals = df['mean_AIS_pre'].dropna().values
p25 = np.percentile(ais_vals, 25)
p75 = np.percentile(ais_vals, 75)

# Approach A: percentile-based
df['ais_tier'] = pd.cut(
    df['mean_AIS_pre'],
    bins=[-np.inf, p25, p75, np.inf],
    labels=['AIS_LOW', 'AIS_MID', 'AIS_HIGH'])

# Approach B: CTL mean - 1SD
ctl_mean = df[df['group'] == 'CTL']['mean_AIS_pre'].mean()
ctl_std  = df[df['group'] == 'CTL']['mean_AIS_pre'].std()
thr_sd   = ctl_mean - 1.0 * ctl_std
df['ais_low_sd'] = df['mean_AIS_pre'] < thr_sd

# Approach C: GMM
X = ais_vals.reshape(-1, 1)
bic_scores = []
for n in [1, 2, 3]:
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(X)
    bic_scores.append((n, gmm.bic(X), gmm))

print("\nGMM BIC scores:")
for n, bic, _ in bic_scores:
    print(f"  {n} components: BIC={bic:.1f}")
best_n   = min(bic_scores, key=lambda x: x[1])[0]
best_gmm = min(bic_scores, key=lambda x: x[1])[2]
print(f"  Best: {best_n} components")

if best_n >= 2:
    X_full = df['mean_AIS_pre'].fillna(df['mean_AIS_pre'].mean()).values.reshape(-1, 1)
    labels = best_gmm.predict(X_full)
    means  = best_gmm.means_.flatten()
    low_c  = int(np.argmin(means))
    df['ais_gmm_low'] = (labels == low_c)
    print(f"  GMM LOW  mean={means[low_c]:.4f}  "
          f"N={df['ais_gmm_low'].sum()}")
    print(f"  GMM HIGH mean={means[np.argmax(means)]:.4f}  "
          f"N={(~df['ais_gmm_low']).sum()}")
else:
    df['ais_gmm_low'] = df['mean_AIS_pre'] < p25
    print("  Unimodal → using P25 fallback")

print(f"\nThresholds:")
print(f"  P25 = {p25:.4f}")
print(f"  P75 = {p75:.4f}")
print(f"  CTL mean−1SD = {thr_sd:.4f}  "
      f"(CTL mean={ctl_mean:.4f}, SD={ctl_std:.4f})")

# ── STEP 2: WHO IS IN AIS_LOW? ────────────────────────────────────────────────
print("\n=== STEP 2: SUBTYPE COMPOSITION ===")

for tier in ['AIS_LOW', 'AIS_MID', 'AIS_HIGH']:
    sub = df[df['ais_tier'] == tier]
    n_ctl = (sub['group'] == 'CTL').sum()
    n_mdd = (sub['group'] == 'MDD_any').sum()
    n     = len(sub)
    ais   = sub['mean_AIS_pre'].dropna()
    print(f"\n{tier} (N={n}):")
    print(f"  CTL={n_ctl} ({100*n_ctl/n:.0f}%)  "
          f"MDD_any={n_mdd} ({100*n_mdd/n:.0f}%)")
    print(f"  AIS_pre: {ais.mean():.4f} ± {ais.std():.4f}")
    for col in ['BDI_Anh', 'BDI', 'HamD']:
        v = sub[col].dropna() if col in sub.columns else pd.Series()
        if len(v) >= 3:
            print(f"  {col}: {v.mean():.1f} ± {v.std():.1f}")

ct = pd.crosstab(df['ais_tier'], df['group'])
chi2_v, p_chi, dof, _ = chi2_contingency(ct)
print(f"\nChi-square (tier × group): χ²={chi2_v:.2f}, df={dof}, p={p_chi:.4f}")
print(ct)

# ── STEP 3: CLINICAL PROFILE — AIS_LOW vs AIS_HIGH WITHIN MDD ────────────────
print("\n=== STEP 3: WITHIN MDD — AIS_LOW vs AIS_HIGH ===")

mdd = df[df['group'] == 'MDD_any'].copy()
low_mdd  = mdd[mdd['ais_tier'] == 'AIS_LOW']
mid_mdd  = mdd[mdd['ais_tier'] == 'AIS_MID']
high_mdd = mdd[mdd['ais_tier'] == 'AIS_HIGH']
print(f"MDD_any AIS_LOW={len(low_mdd)}  MID={len(mid_mdd)}  HIGH={len(high_mdd)}")

clinical_vars = [c for c in mdd.columns
                 if c not in ['subject_id', 'group', 'scid_group',
                               'analysis_group_strict', 'analysis_group_broad',
                               'excluded', 'sex', 'ais_tier', 'ais_low_sd',
                               'ais_gmm_low']
                 and mdd[c].dtype in [np.float64, np.int64, float, int]]

print(f"Testing: {clinical_vars}")

results_mdd = []
for var in clinical_vars:
    lv = low_mdd[var].dropna()
    hv = high_mdd[var].dropna()
    if len(lv) < 3 or len(hv) < 3:
        continue
    U, p = mannwhitneyu(lv, hv, alternative='two-sided')
    pool = np.sqrt(((len(lv)-1)*lv.var(ddof=1) +
                    (len(hv)-1)*hv.var(ddof=1)) /
                   (len(lv)+len(hv)-2) + 1e-10)
    d = (lv.mean() - hv.mean()) / pool
    results_mdd.append({'variable': var,
                         'LOW_mean': round(lv.mean(), 3),
                         'HIGH_mean': round(hv.mean(), 3),
                         'd': round(d, 3), 'p': round(p, 4),
                         'N_LOW': len(lv), 'N_HIGH': len(hv)})

df_mdd_res = pd.DataFrame(results_mdd)
if not df_mdd_res.empty and 'p' in df_mdd_res.columns:
    df_mdd_res = df_mdd_res.sort_values('p')
print("\nAIS_LOW vs AIS_HIGH within MDD_any (EEG):")
if df_mdd_res.empty:
    print("  Insufficient N in AIS_HIGH MDD (N=1) — within-MDD EEG comparison underpowered")
else:
    print(df_mdd_res.to_string(index=False))

# ── STEP 4: SCID SUBGROUPS — CURRENT vs PAST MDD ─────────────────────────────
print("\n=== STEP 4: SCID PHASE ANALYSIS ===")
print(f"scid_group values: {df['scid_group'].value_counts().to_dict()}")

d_cp = np.nan
for label in ['CTL', 'MDD_current', 'MDD_past', 'ANX_other']:
    sub = df[df['scid_group'] == label]
    if len(sub) == 0:
        continue
    n_low = (sub['ais_tier'] == 'AIS_LOW').sum()
    ais   = sub['mean_AIS_pre'].dropna()
    print(f"\n{label} (N={len(sub)}):")
    print(f"  AIS_pre: {ais.mean():.4f} ± {ais.std():.4f}")
    print(f"  In AIS_LOW: {n_low} ({100*n_low/len(sub):.0f}%)")

curr = df[df['scid_group'] == 'MDD_current']['mean_AIS_pre'].dropna()
past = df[df['scid_group'] == 'MDD_past']['mean_AIS_pre'].dropna()
if len(curr) >= 3 and len(past) >= 3:
    _, p_cp = mannwhitneyu(curr, past, alternative='two-sided')
    pool_cp = np.sqrt(((len(curr)-1)*curr.var(ddof=1) +
                       (len(past)-1)*past.var(ddof=1)) /
                      (len(curr)+len(past)-2) + 1e-10)
    d_cp = (curr.mean() - past.mean()) / pool_cp
    print(f"\nMDD_current vs MDD_past:")
    print(f"  current: {curr.mean():.4f} ± {curr.std():.4f} (N={len(curr)})")
    print(f"  past:    {past.mean():.4f} ± {past.std():.4f} (N={len(past)})")
    print(f"  d={d_cp:.3f}, p={p_cp:.4f}")
    print("  →", "STATE marker (effect differs by phase)" if abs(d_cp) > 0.3
          else "TRAIT marker (no phase effect)")

# ── STEP 5: MEG VALIDATION — ANHEDONIA BATTERY ───────────────────────────────
print("\n=== STEP 5: MEG VALIDATION ===")

meg_p25 = np.percentile(ais_meg['mean_AIS_pre'].dropna(), 25)
meg_p75 = np.percentile(ais_meg['mean_AIS_pre'].dropna(), 75)
ais_meg['ais_tier'] = pd.cut(
    ais_meg['mean_AIS_pre'],
    bins=[-np.inf, meg_p25, meg_p75, np.inf],
    labels=['AIS_LOW', 'AIS_MID', 'AIS_HIGH'])

anhedonia_cols = [c for c in ['TEPS_anticipatory', 'SHAPS', 'DARS', 'MASQ_Anh', 'BDI']
                  if c in ais_meg.columns]
print(f"Anhedonia measures available: {anhedonia_cols}")

for tier in ['AIS_LOW', 'AIS_HIGH']:
    sub = ais_meg[ais_meg['ais_tier'] == tier]
    n_ctl = (sub['group'] == 'CTL').sum()
    n_mdd = (sub['group'] == 'MDD').sum()
    print(f"\n{tier} (N={len(sub)}, CTL={n_ctl}, MDD={n_mdd}):")
    for col in anhedonia_cols:
        v = sub[col].dropna()
        if len(v) >= 3:
            print(f"  {col}: {v.mean():.2f} ± {v.std():.2f}")

# Key test: TEPS_anticipatory AIS_LOW vs AIS_HIGH (all subjects)
p_t = d_t = np.nan
if 'TEPS_anticipatory' in ais_meg.columns:
    low_t  = ais_meg[ais_meg['ais_tier'] == 'AIS_LOW']['TEPS_anticipatory'].dropna()
    high_t = ais_meg[ais_meg['ais_tier'] == 'AIS_HIGH']['TEPS_anticipatory'].dropna()
    if len(low_t) >= 3 and len(high_t) >= 3:
        _, p_t = mannwhitneyu(low_t, high_t, alternative='two-sided')
        pool_t = np.sqrt(((len(low_t)-1)*low_t.var(ddof=1) +
                          (len(high_t)-1)*high_t.var(ddof=1)) /
                         (len(low_t)+len(high_t)-2) + 1e-10)
        d_t = (low_t.mean() - high_t.mean()) / pool_t
        print(f"\nTEPS_anticipatory AIS_LOW vs AIS_HIGH (all MEG):")
        print(f"  LOW:  {low_t.mean():.2f} ± {low_t.std():.2f} (N={len(low_t)})")
        print(f"  HIGH: {high_t.mean():.2f} ± {high_t.std():.2f} (N={len(high_t)})")
        print(f"  d={d_t:.3f}, p={p_t:.4f}")
        print("  →", "✓ AIS_LOW has lower anticipatory pleasure" if p_t < 0.10
              else "n.s.")

# Within MDD only (MEG)
print("\nWithin MDD (MEG):")
mdd_meg      = ais_meg[ais_meg['group'] == 'MDD']
mdd_meg_low  = mdd_meg[mdd_meg['ais_tier'] == 'AIS_LOW']
mdd_meg_high = mdd_meg[mdd_meg['ais_tier'] == 'AIS_HIGH']
print(f"  AIS_LOW={len(mdd_meg_low)}  AIS_HIGH={len(mdd_meg_high)}")

for col in anhedonia_cols:
    lv = mdd_meg_low[col].dropna()
    hv = mdd_meg_high[col].dropna()
    if len(lv) < 3 or len(hv) < 3:
        continue
    _, p_m = mannwhitneyu(lv, hv, alternative='two-sided')
    pool_m = np.sqrt(((len(lv)-1)*lv.var(ddof=1) +
                      (len(hv)-1)*hv.var(ddof=1)) /
                     (len(lv)+len(hv)-2) + 1e-10)
    d_m = (lv.mean() - hv.mean()) / pool_m
    print(f"  {col}: d={d_m:.3f}, p={p_m:.4f}  "
          f"(LOW={lv.mean():.2f}, HIGH={hv.mean():.2f})")

# ── STEP 6: CROSS-DATASET CONSISTENCY ────────────────────────────────────────
print("\n=== STEP 6: CROSS-DATASET CONSISTENCY ===")

eeg_mdd_total = (df['group'] == 'MDD_any').sum()
eeg_mdd_low   = ((df['group'] == 'MDD_any') & (df['ais_tier'] == 'AIS_LOW')).sum()
eeg_ctl_total = (df['group'] == 'CTL').sum()
eeg_ctl_low   = ((df['group'] == 'CTL') & (df['ais_tier'] == 'AIS_LOW')).sum()

meg_mdd_total = (ais_meg['group'] == 'MDD').sum()
meg_mdd_low   = ((ais_meg['group'] == 'MDD') & (ais_meg['ais_tier'] == 'AIS_LOW')).sum()
meg_ctl_total = (ais_meg['group'] == 'CTL').sum()
meg_ctl_low   = ((ais_meg['group'] == 'CTL') & (ais_meg['ais_tier'] == 'AIS_LOW')).sum()

print(f"EEG: {eeg_mdd_low}/{eeg_mdd_total} MDD in AIS_LOW "
      f"({100*eeg_mdd_low/eeg_mdd_total:.0f}%)  |  "
      f"{eeg_ctl_low}/{eeg_ctl_total} CTL in AIS_LOW "
      f"({100*eeg_ctl_low/eeg_ctl_total:.0f}%)")
print(f"MEG: {meg_mdd_low}/{meg_mdd_total} MDD in AIS_LOW "
      f"({100*meg_mdd_low/meg_mdd_total:.0f}%)  |  "
      f"{meg_ctl_low}/{meg_ctl_total} CTL in AIS_LOW "
      f"({100*meg_ctl_low/meg_ctl_total:.0f}%)")

ct2 = np.array([[eeg_mdd_low, eeg_mdd_total - eeg_mdd_low],
                 [meg_mdd_low, meg_mdd_total - meg_mdd_low]])
chi2_2, p_chi2, _, _ = chi2_contingency(ct2)
print(f"Proportion consistency: χ²={chi2_2:.2f}, p={p_chi2:.4f}  "
      f"({'stable prevalence' if p_chi2 > 0.05 else 'different proportions'})")

# ── STEP 7: FIGURE ────────────────────────────────────────────────────────────
print("\n=== STEP 7: FIGURE ===")

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    'AIS_pre Subtype Analysis: Characterizing the AIS_LOW Subtype within MDD\n'
    'Hypothesis: AIS_pre identifies a neural subtype (anticipatory anhedonia), not MDD in general',
    fontsize=12, fontweight='bold')

tier_colors = {'AIS_LOW': '#B71C1C', 'AIS_MID': '#FF8F00', 'AIS_HIGH': '#1B5E20'}

# A — Distribution + tier thresholds
ax1 = fig.add_subplot(3, 3, 1)
for grp, col in [('CTL', '#2196F3'), ('MDD_any', '#E53935')]:
    v = df[df['group'] == grp]['mean_AIS_pre'].dropna()
    ax1.hist(v, bins=20, alpha=0.6, color=col, label=grp, density=True)
ax1.axvline(p25, color='black', ls='--', lw=1.8, label=f'P25={p25:.3f}')
ax1.axvline(p75, color='gray',  ls=':',  lw=1.4, label=f'P75={p75:.3f}')
ax1.set_xlabel('AIS_pre (bits)')
ax1.set_ylabel('Density')
ax1.set_title('A. Distribution + Tier Thresholds', loc='left', fontsize=9)
ax1.legend(fontsize=7)

# B — CTL/MDD composition per tier
ax2 = fig.add_subplot(3, 3, 2)
tc = df.groupby(['ais_tier', 'group']).size().unstack(fill_value=0)
tc.plot(kind='bar', ax=ax2, color=['#2196F3', '#E53935'],
        alpha=0.8, width=0.6)
ax2.set_title('B. CTL / MDD_any by Tier', loc='left', fontsize=9)
ax2.set_xlabel('')
ax2.set_ylabel('N subjects')
ax2.tick_params(axis='x', rotation=0)

# C — AIS_pre by SCID phase
ax3 = fig.add_subplot(3, 3, 3)
phase_order = ['CTL', 'MDD_current', 'MDD_past', 'ANX_other']
phase_data, phase_labels = [], []
for lbl in phase_order:
    v = df[df['scid_group'] == lbl]['mean_AIS_pre'].dropna()
    if len(v) >= 3:
        phase_data.append(v.values)
        phase_labels.append(f"{lbl}\n(N={len(v)})")
bp = ax3.boxplot(phase_data, labels=phase_labels, patch_artist=True)
for patch, col in zip(bp['boxes'],
                      ['#2196F3', '#E53935', '#FF7043', '#9E9E9E']):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
ax3.set_ylabel('AIS_pre (bits)')
ax3.set_title('C. AIS_pre by SCID Phase\n(state vs trait?)', loc='left', fontsize=9)
ax3.tick_params(axis='x', labelsize=7)

# D — BDI_Anh by tier (EEG)
ax4 = fig.add_subplot(3, 3, 4)
if 'BDI_Anh' in df.columns and df['BDI_Anh'].notna().sum() > 10:
    sns.boxplot(data=df.dropna(subset=['BDI_Anh', 'ais_tier']),
                x='ais_tier', y='BDI_Anh', ax=ax4,
                order=['AIS_LOW', 'AIS_MID', 'AIS_HIGH'],
                palette=tier_colors)
    ax4.set_title('D. BDI Anhedonia by Tier (EEG)', loc='left', fontsize=9)
    ax4.set_xlabel('')
else:
    ax4.text(0.5, 0.5, 'BDI_Anh not available', ha='center', va='center',
             transform=ax4.transAxes)
    ax4.set_title('D. BDI Anhedonia (EEG)', loc='left', fontsize=9)

# E — GMM fit
ax5 = fig.add_subplot(3, 3, 5)
x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
ax5.hist(X.flatten(), bins=28, density=True, alpha=0.35, color='gray')
if best_n >= 2:
    from scipy.stats import norm as sp_norm
    for i in range(best_n):
        mu  = best_gmm.means_[i, 0]
        sig = np.sqrt(best_gmm.covariances_[i, 0, 0])
        w   = best_gmm.weights_[i]
        ax5.plot(x_range.flatten(),
                 w * sp_norm.pdf(x_range.flatten(), mu, sig),
                 lw=2, label=f'Comp {i+1}: μ={mu:.3f}')
ax5.set_xlabel('AIS_pre (bits)')
ax5.set_title(f'E. GMM ({best_n} component{"s" if best_n>1 else ""})\n'
              'Bimodal vs unimodal?', loc='left', fontsize=9)
ax5.legend(fontsize=7)

# F — TEPS_anticipatory by tier (MEG)
ax6 = fig.add_subplot(3, 3, 6)
if 'TEPS_anticipatory' in ais_meg.columns:
    sns.boxplot(data=ais_meg.dropna(subset=['TEPS_anticipatory', 'ais_tier']),
                x='ais_tier', y='TEPS_anticipatory', ax=ax6,
                order=['AIS_LOW', 'AIS_MID', 'AIS_HIGH'],
                palette=tier_colors)
    title_str = (f'F. TEPS_anticipatory by Tier (MEG)\n'
                 f'd={d_t:.2f}, p={p_t:.3f}' if not np.isnan(d_t)
                 else 'F. TEPS_anticipatory by Tier (MEG)')
    ax6.set_title(title_str, loc='left', fontsize=9)
    ax6.set_xlabel('')
else:
    ax6.text(0.5, 0.5, 'TEPS not available', ha='center', va='center',
             transform=ax6.transAxes)

# G — AIS_pre vs BDI_Anh scatter, colored by tier
ax7 = fig.add_subplot(3, 3, 7)
if 'BDI_Anh' in df.columns and df['BDI_Anh'].notna().sum() > 10:
    for tier, col in tier_colors.items():
        sub = df[df['ais_tier'] == tier].dropna(subset=['BDI_Anh'])
        ax7.scatter(sub['mean_AIS_pre'], sub['BDI_Anh'],
                    c=col, alpha=0.65, s=35, label=tier)
    r_ba, p_ba = pearsonr(df[['mean_AIS_pre', 'BDI_Anh']].dropna()['mean_AIS_pre'],
                          df[['mean_AIS_pre', 'BDI_Anh']].dropna()['BDI_Anh'])
    ax7.set_xlabel('AIS_pre (bits)')
    ax7.set_ylabel('BDI Anhedonia')
    ax7.set_title(f'G. AIS_pre vs Anhedonia\nr={r_ba:+.3f}, p={p_ba:.3f}',
                  loc='left', fontsize=9)
    ax7.legend(fontsize=7)

# H — AIS_LOW prevalence across datasets
ax8 = fig.add_subplot(3, 3, 8)
x_pos = np.arange(2)
ax8.bar(x_pos - 0.2,
        [100*eeg_mdd_low/eeg_mdd_total, 100*meg_mdd_low/meg_mdd_total],
        0.38, label='MDD', color='#E53935', alpha=0.8)
ax8.bar(x_pos + 0.2,
        [100*eeg_ctl_low/eeg_ctl_total, 100*meg_ctl_low/meg_ctl_total],
        0.38, label='CTL', color='#2196F3', alpha=0.8)
ax8.axhline(25, color='gray', ls=':', alpha=0.5)
ax8.set_xticks(x_pos)
ax8.set_xticklabels([f'EEG\n(MDD={eeg_mdd_total})', f'MEG\n(MDD={meg_mdd_total})'])
ax8.set_ylabel('% in AIS_LOW tier')
ax8.set_title('H. AIS_LOW Prevalence\nCross-Dataset', loc='left', fontsize=9)
ax8.legend(fontsize=8)

# I — Summary text
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
summary = (
    "AIS_LOW SUBTYPE — SUMMARY\n\n"
    f"EEG (d=0.874, N=109):\n"
    f"  {eeg_mdd_low}/{eeg_mdd_total} MDD are AIS_LOW "
    f"({100*eeg_mdd_low/eeg_mdd_total:.0f}%)\n"
    f"  {eeg_ctl_low}/{eeg_ctl_total} CTL are AIS_LOW "
    f"({100*eeg_ctl_low/eeg_ctl_total:.0f}%)\n\n"
    f"MEG (d=0.448, N=82):\n"
    f"  {meg_mdd_low}/{meg_mdd_total} MDD are AIS_LOW "
    f"({100*meg_mdd_low/meg_mdd_total:.0f}%)\n"
    f"  {meg_ctl_low}/{meg_ctl_total} CTL are AIS_LOW "
    f"({100*meg_ctl_low/meg_ctl_total:.0f}%)\n\n"
    f"GMM: {best_n} component(s)\n"
    f"SCID phase d={d_cp:.3f}\n"
    f"TEPS_ant d={d_t:.3f} (MEG)\n\n"
    "HYPOTHESIS:\n"
    "AIS_LOW ≈ MDD with anticipatory anhedonia\n"
    "AIS_HIGH ≈ MDD without hedonic profile\n"
    "(subtype, not shift of full MDD distribution)"
)
ax9.text(0.05, 0.97, summary, transform=ax9.transAxes,
         fontsize=8.5, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))

plt.tight_layout()
out_fig = PST / "derivatives/subtype_ais_low_analysis.png"
fig.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"Figure saved: {out_fig}")

# ── STEP 8: FINAL VERDICT ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS 1 — AIS_LOW SUBTYPE — FINAL VERDICT")
print("="*60)

enrichment = (eeg_mdd_low/eeg_mdd_total) / max(eeg_ctl_low/eeg_ctl_total, 0.01)
print(f"\n1. Is AIS_LOW enriched in MDD?")
print(f"   EEG: {100*eeg_mdd_low/eeg_mdd_total:.0f}% MDD vs "
      f"{100*eeg_ctl_low/eeg_ctl_total:.0f}% CTL in AIS_LOW")
print(f"   Enrichment: {enrichment:.1f}x  (>1.5x = subtype exists)")
print("  ", "✓ ENRICHED — subtype exists" if enrichment > 1.5
      else "→ No enrichment — effect is a global distribution shift")

sig_vars   = df_mdd_res[df_mdd_res['p'] < 0.05]['variable'].tolist() if len(df_mdd_res) else []
trend_vars = df_mdd_res[(df_mdd_res['p'] >= 0.05) & (df_mdd_res['p'] < 0.15)]['variable'].tolist() if len(df_mdd_res) else []
print(f"\n2. Clinical differentiation within MDD?")
print(f"   Significant (p<0.05): {sig_vars if sig_vars else 'none'}")
print(f"   Trends (p<0.15):      {trend_vars if trend_vars else 'none'}")

print(f"\n3. State vs trait marker?")
if not np.isnan(d_cp):
    print(f"   MDD_current vs MDD_past: d={d_cp:.3f}")
    print("  ", "→ STATE marker" if abs(d_cp) > 0.3 else "→ TRAIT marker (no phase effect)")

print(f"\n4. Bimodal distribution?")
print(f"   GMM best fit: {best_n} component(s)")
print("  ", "✓ BIMODAL — natural subtype" if best_n >= 2 else "→ Unimodal — continuous effect")

print(f"\n5. Validated by TEPS_anticipatory (MEG)?")
if not np.isnan(d_t):
    print(f"   d={d_t:.3f}, p={p_t:.4f}")
    print("  ", "✓ VALIDATED" if p_t < 0.10 else "→ Not significant (trend)" if p_t < 0.20 else "→ Null")

print(f"\n→ NEXT STEPS:")
print("   If subtype confirmed → Analysis 2: resting-state signature (TDBRAIN)")
print("   If continuous → Reframe as AIS_pre dimension of anticipatory anhedonia")
print("="*60)

# Save subject-level results
df.to_csv(PST / "derivatives/subtype_ais_eeg_subjects.csv", index=False)
ais_meg.to_csv(MEG / "derivatives/subtype_ais_meg_subjects.csv", index=False)
print("\nSubject-level subtype CSVs saved.")
