import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE     = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/Depression_PS_Task")
EPO_DIR  = BASE / "derivatives/epochs"
OUT_DIR  = BASE / "derivatives/erp_it_cavanagh"
CLINICAL = BASE / "derivatives/clinical_lookup_ps_task.csv"

TMIN, TMAX  = 0.350, 0.450   # seconds post-feedback
MIN_TRIALS  = 5
GROUP_COL   = 'analysis_group_broad'
INCLUDE     = ['CTL', 'MDD_any']

# ── STEP 1: Load clinical ─────────────────────────────────────────────────
clinical = pd.read_csv(CLINICAL)
print("analysis_group_broad unique:", clinical['analysis_group_broad'].unique())
print("analysis_group_strict unique:", clinical['analysis_group_strict'].unique())

clinical_main = clinical[
    clinical[GROUP_COL].isin(INCLUDE) & (~clinical['excluded'])
].copy()
print(f"Main analysis N: {clinical_main[GROUP_COL].value_counts().to_dict()}")

# ── STEP 2: Inspect one epoch file ───────────────────────────────────────
sample_epo = mne.read_epochs(
    sorted(EPO_DIR.glob('*_task-ps_epo.fif'))[0],
    preload=False, verbose='ERROR'
)
print("Channels[:10]:", sample_epo.ch_names[:10], "...")
print("event_id:", sample_epo.event_id)
print("Times:", sample_epo.times[0], "to", sample_epo.times[-1], "| sfreq:", sample_epo.info['sfreq'])

ch_target = None
for candidate in ['FCz', 'FC z', 'fcz', 'FCZ']:
    if candidate in sample_epo.ch_names:
        ch_target = candidate
        break
if ch_target is None:
    for fallback in ['FC1', 'Fz']:
        if fallback in sample_epo.ch_names:
            ch_target = fallback
            print(f"FCz not found — using fallback: {ch_target}")
            break
assert ch_target is not None, f"No frontal midline channel found in {sample_epo.ch_names}"
print(f"Target channel: {ch_target}")

# event_id = {'Reward': 94, 'Loss': 104} — access by name
assert 'Reward' in sample_epo.event_id, f"Reward key missing. event_id: {sample_epo.event_id}"
assert 'Loss'   in sample_epo.event_id, f"Loss key missing. event_id: {sample_epo.event_id}"

# ── STEP 3: Extract per-subject CV ───────────────────────────────────────
records = []
epoch_files = sorted(EPO_DIR.glob('*_task-ps_epo.fif'))
print(f"\nProcessing {len(epoch_files)} epoch files...")

for fpath in epoch_files:
    try:
        sub_id = int(fpath.stem.split('-')[1].split('_')[0])
    except (IndexError, ValueError):
        print(f"  Cannot parse ID from {fpath.name} — skip")
        continue

    row = clinical_main[clinical_main['subject_id'] == sub_id]
    if row.empty:
        continue
    group = row[GROUP_COL].values[0]

    try:
        epo = mne.read_epochs(fpath, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"  sub-{sub_id}: load error — {e}")
        continue

    tmask  = (epo.times >= TMIN) & (epo.times <= TMAX)
    ch_idx = epo.ch_names.index(ch_target)

    for cond_name in ('Reward', 'Loss'):
        try:
            cond_epo = epo[cond_name]
        except KeyError:
            print(f"  sub-{sub_id}: '{cond_name}' not found — skip")
            continue

        n_trials = len(cond_epo)
        if n_trials < MIN_TRIALS:
            print(f"  sub-{sub_id} {cond_name}: {n_trials} trials < {MIN_TRIALS} — skip")
            continue

        # shape: (n_trials, n_channels, n_times) → per-trial mean amplitude in window
        trial_amps = cond_epo.get_data()[:, ch_idx, :][:, tmask].mean(axis=1)

        mean_amp = np.mean(trial_amps)
        std_amp  = np.std(trial_amps, ddof=1)
        cv = std_amp / np.abs(mean_amp) if np.abs(mean_amp) > 1e-10 else np.nan

        records.append({
            'subject_id': sub_id,
            'group':      group,
            'condition':  cond_name,
            'CV':         cv,
            'mean_amp_uV': mean_amp,
            'std_amp_uV':  std_amp,
            'n_trials':    n_trials,
        })

df_cv = pd.DataFrame(records)
print(f"\nRecords collected: {len(df_cv)}")
print(df_cv.groupby(['group', 'condition'])['CV'].describe().round(3))

# ── STEP 4: Statistical comparison ───────────────────────────────────────
def cohens_d(a, b):
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(
        ((n_a - 1) * np.std(a, ddof=1)**2 + (n_b - 1) * np.std(b, ddof=1)**2)
        / (n_a + n_b - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

results = []
for cond in ('Reward', 'Loss'):
    sub = df_cv[df_cv['condition'] == cond]
    ctl = sub[sub['group'] == 'CTL']['CV'].dropna()
    mdd = sub[sub['group'] != 'CTL']['CV'].dropna()
    U, p = mannwhitneyu(ctl, mdd, alternative='two-sided')
    d    = cohens_d(mdd.values, ctl.values)
    results.append({
        'comparison': f'CTL vs MDD — CV_{cond}',
        'N_CTL': len(ctl), 'N_MDD': len(mdd),
        'CTL_mean': ctl.mean(), 'CTL_std': ctl.std(),
        'MDD_mean': mdd.mean(), 'MDD_std': mdd.std(),
        'U': U, 'p_raw': p, 'd': d,
    })

# Reward–Loss asymmetry per subject
df_wide = df_cv.pivot_table(
    index=['subject_id', 'group'], columns='condition', values='CV'
).reset_index()
df_wide['CV_asymmetry'] = df_wide['Reward'] - df_wide['Loss']

ctl_asym = df_wide[df_wide['group'] == 'CTL']['CV_asymmetry'].dropna()
mdd_asym = df_wide[df_wide['group'] != 'CTL']['CV_asymmetry'].dropna()
U_a, p_a = mannwhitneyu(ctl_asym, mdd_asym, alternative='two-sided')
d_a      = cohens_d(mdd_asym.values, ctl_asym.values)
results.append({
    'comparison': 'CTL vs MDD — CV_Asymmetry',
    'N_CTL': len(ctl_asym), 'N_MDD': len(mdd_asym),
    'CTL_mean': ctl_asym.mean(), 'CTL_std': ctl_asym.std(),
    'MDD_mean': mdd_asym.mean(), 'MDD_std': mdd_asym.std(),
    'U': U_a, 'p_raw': p_a, 'd': d_a,
})

df_res = pd.DataFrame(results)
_, p_fdr, _, _ = multipletests(df_res['p_raw'], method='fdr_bh')
df_res['p_fdr'] = p_fdr

print("\n=== CV RESULTS ===")
print(df_res[['comparison', 'N_CTL', 'N_MDD',
              'CTL_mean', 'MDD_mean', 'd', 'p_raw', 'p_fdr']].to_string(index=False))

for _, row in df_res.iterrows():
    direction = "MDD > CTL" if row['MDD_mean'] > row['CTL_mean'] else "CTL > MDD"
    sig = "SIGNIFICANT (FDR<0.05)" if row['p_fdr'] < 0.05 else "ns"
    print(f"  {row['comparison']}: {direction}, d={row['d']:.3f}, "
          f"p_raw={row['p_raw']:.4f}, p_fdr={row['p_fdr']:.4f} [{sig}]")

# ── STEP 5: Sensitivity — strict MDD (SCID=1 only) ───────────────────────
strict_ids = set(
    clinical[
        clinical['analysis_group_strict'].isin(['CTL', 'MDD_current']) &
        (~clinical['excluded'])
    ]['subject_id']
)
df_cv_strict = df_cv[df_cv['subject_id'].isin(strict_ids)].copy()

for cond in ('Reward', 'Loss'):
    sub_s = df_cv_strict[df_cv_strict['condition'] == cond]
    ctl_s = sub_s[sub_s['group'] == 'CTL']['CV'].dropna()
    mdd_s = sub_s[sub_s['group'] != 'CTL']['CV'].dropna()
    if len(mdd_s) > 3:
        U_s, p_s = mannwhitneyu(ctl_s, mdd_s, alternative='two-sided')
        d_s      = cohens_d(mdd_s.values, ctl_s.values)
        print(f"Sensitivity strict MDD (N={len(mdd_s)}) {cond}: "
              f"d={d_s:.3f}, p={p_s:.4f}")

# ── STEP 6: Visualization ────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
palette = {'CTL': '#2196F3', 'MDD_any': '#F44336'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Neural Rigidity Index: CV at FCz (350–450ms post-feedback)',
             fontsize=13, fontweight='bold')

for ax, cond in zip(axes[:2], ('Reward', 'Loss')):
    sub = df_cv[df_cv['condition'] == cond]
    sns.boxplot(data=sub, x='group', y='CV', ax=ax,
                palette=palette, width=0.5, order=['CTL', 'MDD_any'])
    sns.stripplot(data=sub, x='group', y='CV', ax=ax,
                  color='black', alpha=0.4, size=3, jitter=True,
                  order=['CTL', 'MDD_any'])
    row_r = df_res[df_res['comparison'].str.contains(cond)].iloc[0]
    ax.set_title(f'CV {cond}\nd={row_r["d"]:.2f}, p_fdr={row_r["p_fdr"]:.3f}')
    ax.set_xlabel('')
    ax.set_ylabel('CV (std/|mean|)')

ax3 = axes[2]
for grp, color in [('CTL', '#2196F3'), ('MDD_any', '#F44336')]:
    sub = df_wide[df_wide['group'] == grp]
    ax3.scatter(sub['Reward'], sub['Loss'],
                c=color, alpha=0.6, label=grp, s=50, edgecolors='white', linewidths=0.5)
ax3.set_xlabel('CV Reward')
ax3.set_ylabel('CV Loss')
ax3.set_title(f'CV Reward vs Loss\nAsymmetry: d={d_a:.2f}, p_fdr={df_res.iloc[-1]["p_fdr"]:.3f}')
ax3.legend()
lims = [0, max(df_wide[['Reward', 'Loss']].max().max() * 1.05, 1)]
ax3.plot(lims, lims, 'k--', alpha=0.3, lw=1)
ax3.set_xlim(lims); ax3.set_ylim(lims)

plt.tight_layout()
plt.savefig(OUT_DIR / 'cv_fcz_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUT_DIR / 'cv_fcz_results.png'}")

# ── STEP 7: Save results ─────────────────────────────────────────────────
out_csv = OUT_DIR / 'cv_fcz_rewloss.csv'
df_cv.to_csv(out_csv, index=False)

out_stats = OUT_DIR / 'cv_fcz_stats.csv'
df_res.to_csv(out_stats, index=False)

print(f"\nData saved:  {out_csv}  ({len(df_cv)} rows)")
print(f"Stats saved: {out_stats}  ({len(df_res)} rows)")
print("\n=== FINAL COUNTS ===")
print(f"  Subjects processed (broad): {df_cv['subject_id'].nunique()}")
for grp in ('CTL', 'MDD_any'):
    n = df_cv[df_cv['group'] == grp]['subject_id'].nunique()
    print(f"  N {grp}: {n}")
