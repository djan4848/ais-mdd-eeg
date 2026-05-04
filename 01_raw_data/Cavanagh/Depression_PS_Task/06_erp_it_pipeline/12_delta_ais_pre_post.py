import sys
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE     = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/Depression_PS_Task")
EPO_DIR  = BASE / "derivatives/epochs"
OUT_DIR  = BASE / "derivatives/erp_it_cavanagh"
CLINICAL = BASE / "derivatives/clinical_lookup_ps_task.csv"

GROUP_COL = 'analysis_group_broad'
CTL_LABEL = 'CTL'
MDD_LABEL = 'MDD_any'

clinical = pd.read_csv(CLINICAL)
clinical_main = clinical[
    clinical[GROUP_COL].isin([CTL_LABEL, MDD_LABEL]) & (~clinical['excluded'])
].copy()
print(f"N CTL: {(clinical_main[GROUP_COL]==CTL_LABEL).sum()}, "
      f"N MDD: {(clinical_main[GROUP_COL]==MDD_LABEL).sum()}")

# ── STEP 0: Import or implement AIS ──────────────────────────────────────
INFO_THEORY_PATH = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/"
                        "ds003474/code/eeg_depression_classification")
sys.path.insert(0, str(INFO_THEORY_PATH))

try:
    from info_theory import compute_ais
    print("compute_ais imported from info_theory.py")
except ImportError:
    print("info_theory not found — using inline AIS")
    def compute_ais(x, lag=4, n_bins=4):
        x = np.asarray(x, dtype=float)
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return np.nan
        bins  = np.digitize(x, edges[1:-1])
        x_t   = bins[lag:]
        x_lag = bins[:-lag]
        joint = np.zeros((n_bins, n_bins))
        for a, b in zip(x_t, x_lag):
            joint[min(a, n_bins) - 1, min(b, n_bins) - 1] += 1
        joint /= joint.sum() + 1e-10
        px_t  = joint.sum(axis=1)
        px_lag = joint.sum(axis=0)
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and px_t[i] > 0 and px_lag[j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (px_t[i] * px_lag[j]))
        return float(mi)

# Validate on synthetic data
rng   = np.random.default_rng(42)
ar1   = np.zeros(200)
for i in range(1, 200):
    ar1[i] = 0.9 * ar1[i-1] + 0.1 * rng.standard_normal()
wn        = rng.standard_normal(200)
ais_ar1   = compute_ais(ar1, lag=1, n_bins=4)
ais_wn    = compute_ais(wn,  lag=1, n_bins=4)
assert ais_ar1 > ais_wn, f"AIS validation FAILED: AR1={ais_ar1:.4f} <= WN={ais_wn:.4f}"
print(f"AIS validation OK: AR1={ais_ar1:.4f} > WN={ais_wn:.4f}")

# ── STEP 1: Safe wrapper ──────────────────────────────────────────────────
def safe_ais(x, lag=1, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * lag + 10:
        return np.nan
    if np.std(x) < 1e-12:
        return np.nan
    try:
        val = compute_ais(x, lag=lag, n_bins=n_bins)
        return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0

def perm_r(x, y, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    r0  = pearsonr(x, y)[0]
    null = [pearsonr(rng.permutation(x), y)[0] for _ in range(n)]
    return r0, float(np.mean(np.abs(null) >= np.abs(r0))), np.percentile(null, [2.5, 97.5])

# ── STEP 2: Lag sensitivity preview (5 subjects) ─────────────────────────
print("\n=== LAG SENSITIVITY PREVIEW (5 subjects, post window 0–400ms) ===")
sample_files = sorted(EPO_DIR.glob('*_task-ps_epo.fif'))[:5]
for lag_test in [1, 2, 4, 6]:
    vals = []
    for fpath in sample_files:
        epo   = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        ch_idx = epo.ch_names.index('FCz')
        # Use first trial of Reward condition
        seg = epo['Reward'].get_data()[0, ch_idx, :]
        mask = (epo.times >= 0.0) & (epo.times < 0.4)
        vals.append(safe_ais(seg[mask], lag=lag_test, n_bins=4))
    print(f"  lag={lag_test}: mean={np.nanmean(vals):.4f} bits (N={sum(np.isfinite(v) for v in vals)})")

# ── Window definitions ─────────────────────────────────────────────────────
WINDOWS = {
    'pre':  {'tmin': -0.200, 'tmax': 0.000, 'lag': 1, 'n_bins': 4},
    'post': {'tmin':  0.000, 'tmax': 0.400, 'lag': 4, 'n_bins': 4},
}
MIN_TRIALS = 5

# ── STEP 3: Main extraction loop ──────────────────────────────────────────
records     = []
nan_log     = []
epoch_files = sorted(EPO_DIR.glob('*_task-ps_epo.fif'))
print(f"\nProcessing {len(epoch_files)} epoch files...")

for idx, fpath in enumerate(epoch_files):
    try:
        sub_id = int(fpath.stem.split('-')[1].split('_')[0])
    except (IndexError, ValueError):
        continue

    row_c = clinical_main[clinical_main['subject_id'] == sub_id]
    if row_c.empty:
        continue
    row_c   = row_c.iloc[0]
    group   = row_c[GROUP_COL]
    bdi_anh = row_c.get('BDI_Anh', np.nan)
    bdi     = row_c.get('BDI', np.nan)

    try:
        epo = mne.read_epochs(fpath, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"  sub-{sub_id}: load error — {e}")
        continue

    ch_idx = epo.ch_names.index('FCz')
    times  = epo.times

    for cond_name, event_key in (('Reward', 'Reward'), ('Loss', 'Loss')):
        try:
            cond_epo = epo[event_key]
        except KeyError:
            print(f"  sub-{sub_id}: '{event_key}' not found — skip")
            continue
        if len(cond_epo) < MIN_TRIALS:
            continue

        data_3d = cond_epo.get_data()   # (n_trials, n_ch, n_t)

        for trial_idx in range(len(cond_epo)):
            signal = data_3d[trial_idx, ch_idx, :]
            row = {
                'subject_id': sub_id,
                'group':      group,
                'BDI_Anh':   bdi_anh,
                'BDI':        bdi,
                'condition':  cond_name,
                'trial':      trial_idx,
            }
            for win_name, wp in WINDOWS.items():
                mask = (times >= wp['tmin']) & (times < wp['tmax'])
                val  = safe_ais(signal[mask], lag=wp['lag'], n_bins=wp['n_bins'])
                row[f'AIS_{win_name}'] = val
                if np.isnan(val):
                    nan_log.append({'sub': sub_id, 'cond': cond_name,
                                    'trial': trial_idx, 'window': win_name})
            row['delta_AIS'] = row['AIS_pre'] - row['AIS_post']
            records.append(row)

    if (idx + 1) % 25 == 0:
        print(f"  [{idx+1}/{len(epoch_files)}]")

df_trial = pd.DataFrame(records)
print(f"\nTotal trial records: {len(df_trial):,}")
print(f"NaN AIS_pre:   {df_trial['AIS_pre'].isna().mean():.1%}")
print(f"NaN AIS_post:  {df_trial['AIS_post'].isna().mean():.1%}")
print(f"NaN delta_AIS: {df_trial['delta_AIS'].isna().mean():.1%}")

# ── STEP 4: Subject-level aggregation ────────────────────────────────────
df_subj = (
    df_trial
    .groupby(['subject_id', 'group', 'BDI_Anh', 'BDI', 'condition'])
    .agg(
        mean_AIS_pre   = ('AIS_pre',   'mean'),
        mean_AIS_post  = ('AIS_post',  'mean'),
        mean_delta_AIS = ('delta_AIS', 'mean'),
        std_delta_AIS  = ('delta_AIS', 'std'),
        n_valid_trials = ('delta_AIS', 'count'),
    )
    .reset_index()
)

df_all = (
    df_trial
    .groupby(['subject_id', 'group', 'BDI_Anh', 'BDI'])
    .agg(
        mean_AIS_pre   = ('AIS_pre',   'mean'),
        mean_AIS_post  = ('AIS_post',  'mean'),
        mean_delta_AIS = ('delta_AIS', 'mean'),
    )
    .reset_index()
)

# ── STEP 5: Primary hypothesis test ──────────────────────────────────────
print("\n=== PRIMARY HYPOTHESIS: ΔAIS_CTL > ΔAIS_MDD ===")
results = []

for label, data_df, col in [
    ('All_conditions', df_all,                                    'mean_delta_AIS'),
    ('Reward_only',    df_subj[df_subj['condition']=='Reward'],   'mean_delta_AIS'),
    ('Loss_only',      df_subj[df_subj['condition']=='Loss'],     'mean_delta_AIS'),
]:
    ctl = data_df[data_df['group'] == CTL_LABEL][col].dropna()
    mdd = data_df[data_df['group'] == MDD_LABEL][col].dropna()
    if len(ctl) < 3 or len(mdd) < 3:
        continue
    U, p = mannwhitneyu(ctl, mdd, alternative='two-sided')
    d    = cohens_d(ctl.values, mdd.values)
    h1   = ctl.mean() > mdd.mean()
    results.append({
        'comparison':   label,
        'N_CTL':        len(ctl),  'N_MDD':        len(mdd),
        'CTL_AIS_pre':  data_df[data_df['group']==CTL_LABEL]['mean_AIS_pre'].mean(),
        'MDD_AIS_pre':  data_df[data_df['group']==MDD_LABEL]['mean_AIS_pre'].mean(),
        'CTL_AIS_post': data_df[data_df['group']==CTL_LABEL]['mean_AIS_post'].mean(),
        'MDD_AIS_post': data_df[data_df['group']==MDD_LABEL]['mean_AIS_post'].mean(),
        'CTL_dAIS': ctl.mean(), 'CTL_dAIS_std': ctl.std(),
        'MDD_dAIS': mdd.mean(), 'MDD_dAIS_std': mdd.std(),
        'U': U, 'p_raw': p, 'd': d,
        'H1_confirmed': '✓' if h1 else '✗',
    })
    print(f"\n{label}:")
    print(f"  CTL ΔAIS = {ctl.mean():.4f} ± {ctl.std():.4f}  (N={len(ctl)})")
    print(f"  MDD ΔAIS = {mdd.mean():.4f} ± {mdd.std():.4f}  (N={len(mdd)})")
    print(f"  d={d:.3f}, p={p:.4f}, H1 (CTL>MDD): {'CONFIRMED ✓' if h1 else 'REJECTED ✗'}")
    # Also print pre and post separately
    ctl_pre  = data_df[data_df['group']==CTL_LABEL]['mean_AIS_pre'].mean()
    mdd_pre  = data_df[data_df['group']==MDD_LABEL]['mean_AIS_pre'].mean()
    ctl_post = data_df[data_df['group']==CTL_LABEL]['mean_AIS_post'].mean()
    mdd_post = data_df[data_df['group']==MDD_LABEL]['mean_AIS_post'].mean()
    print(f"  CTL: AIS_pre={ctl_pre:.4f}, AIS_post={ctl_post:.4f}, Δ={ctl_pre-ctl_post:.4f}")
    print(f"  MDD: AIS_pre={mdd_pre:.4f}, AIS_post={mdd_post:.4f}, Δ={mdd_pre-mdd_post:.4f}")

df_res = pd.DataFrame(results)
if len(df_res) > 0:
    _, p_fdr, _, _ = multipletests(df_res['p_raw'], method='fdr_bh')
    df_res['p_fdr'] = p_fdr

# ── STEP 6: Anhedonia correlation ─────────────────────────────────────────
print("\n=== BDI_Anh CORRELATION ===")
valid = df_all[['mean_delta_AIS', 'BDI_Anh', 'BDI']].dropna()
r_anh, p_anh, ci_anh = perm_r(valid['mean_delta_AIS'].values, valid['BDI_Anh'].values)
r_bdi, p_bdi, ci_bdi = perm_r(valid['mean_delta_AIS'].values, valid['BDI'].values)
rs_anh = spearmanr(valid['mean_delta_AIS'], valid['BDI_Anh'])[0]
print(f"  ΔAIS vs BDI_Anh: r={r_anh:.3f}, p_perm={p_anh:.4f}, "
      f"95%CI_null=[{ci_anh[0]:.3f},{ci_anh[1]:.3f}], rho={rs_anh:.3f}")
print(f"  ΔAIS vs BDI:     r={r_bdi:.3f}, p_perm={p_bdi:.4f}")
print(f"  Expected direction: r < 0 (more anhedonia → less ΔAIS)")
print(f"  Confirmed: {'✓' if r_anh < 0 else '✗'}  (N={len(valid)})")

# ── STEP 7: Lag sensitivity (15-subject subsample) ────────────────────────
print("\n=== LAG SENSITIVITY (15-subject subsample) ===")
lag_files = sorted(EPO_DIR.glob('*_task-ps_epo.fif'))[:15]
lag_results = {}

for lag_test in [1, 2, 4, 6, 8]:
    lag_rows = []
    for fpath in lag_files:
        try:
            sub_id = int(fpath.stem.split('-')[1].split('_')[0])
        except (IndexError, ValueError):
            continue
        row_c = clinical_main[clinical_main['subject_id'] == sub_id]
        if row_c.empty:
            continue
        grp = row_c.iloc[0][GROUP_COL]
        try:
            epo = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        except Exception:
            continue
        ch_idx = epo.ch_names.index('FCz')
        times  = epo.times
        pre_mask  = (times >= -0.2) & (times < 0.0)
        post_mask = (times >= 0.0)  & (times < 0.4)
        for cond_key in ('Reward', 'Loss'):
            try:
                cond_data = epo[cond_key].get_data()[:, ch_idx, :]
            except KeyError:
                continue
            for trial in cond_data:
                pre_v  = safe_ais(trial[pre_mask],  lag=1,        n_bins=4)
                post_v = safe_ais(trial[post_mask], lag=lag_test, n_bins=4)
                if np.isfinite(pre_v) and np.isfinite(post_v):
                    lag_rows.append({'sub': sub_id, 'group': grp,
                                     'delta': pre_v - post_v})
    df_lag = pd.DataFrame(lag_rows)
    if len(df_lag) < 10:
        continue
    ctl_l = df_lag[df_lag['group'] == CTL_LABEL]['delta'].dropna()
    mdd_l = df_lag[df_lag['group'] == MDD_LABEL]['delta'].dropna()
    if len(mdd_l) < 3:
        continue
    U_l, p_l = mannwhitneyu(ctl_l, mdd_l, alternative='two-sided')
    d_l = cohens_d(ctl_l.values, mdd_l.values)
    lag_results[lag_test] = {'d': d_l, 'p': p_l,
                              'CTL': ctl_l.mean(), 'MDD': mdd_l.mean()}
    print(f"  lag={lag_test}: d={d_l:.3f}, p={p_l:.4f}, "
          f"CTL_Δ={ctl_l.mean():.4f}, MDD_Δ={mdd_l.mean():.4f}")

# ── STEP 8: Visualization ─────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
palette = {CTL_LABEL: '#2196F3', MDD_LABEL: '#F44336'}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'ΔAIS (Active Information Storage: Pre − Post Feedback)\n'
    'Hypothesis: ΔAIS_CTL > ΔAIS_MDD (feedback disrupts autocorrelation more in CTL)',
    fontsize=11, fontweight='bold'
)

# Panel A: ΔAIS by group × condition
ax = axes[0, 0]
sns.boxplot(data=df_subj, x='condition', y='mean_delta_AIS',
            hue='group', ax=ax, palette=palette, width=0.5)
sns.stripplot(data=df_subj, x='condition', y='mean_delta_AIS',
              hue='group', ax=ax, dodge=True, alpha=0.4, size=3, palette=palette,
              legend=False)
ax.axhline(0, color='gray', ls='--', alpha=0.5, lw=1)
ax.set_title('A. ΔAIS by Group × Condition')
ax.set_ylabel('ΔAIS = AIS_pre − AIS_post [bits]')
ax.set_xlabel('')
ax.get_legend().set_title('')

# Panel B: AIS_pre vs AIS_post scatter
ax = axes[0, 1]
lim_all = pd.concat([df_all['mean_AIS_pre'], df_all['mean_AIS_post']]).dropna()
lims    = [lim_all.min() * 0.95, lim_all.max() * 1.05]
for grp, color in [(CTL_LABEL, '#2196F3'), (MDD_LABEL, '#F44336')]:
    sub = df_all[df_all['group'] == grp]
    ax.scatter(sub['mean_AIS_pre'], sub['mean_AIS_post'],
               c=color, alpha=0.55, s=45, label=grp)
    ax.scatter(sub['mean_AIS_pre'].mean(), sub['mean_AIS_post'].mean(),
               c=color, s=200, marker='*', edgecolors='black', zorder=5)
ax.plot(lims, lims, 'k--', alpha=0.3, lw=1, label='No change (ΔAIS=0)')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Mean AIS_pre [bits]')
ax.set_ylabel('Mean AIS_post [bits]')
ax.set_title('B. AIS Pre vs Post\n(below diagonal → ΔAIS > 0)')
ax.legend(fontsize=8)

# Panel C: BDI_Anh vs ΔAIS
ax = axes[0, 2]
for grp, color in [(CTL_LABEL, '#2196F3'), (MDD_LABEL, '#F44336')]:
    sub = df_all[df_all['group'] == grp].dropna(subset=['BDI_Anh', 'mean_delta_AIS'])
    ax.scatter(sub['BDI_Anh'], sub['mean_delta_AIS'],
               c=color, alpha=0.55, s=45, label=grp)
valid2 = df_all[['BDI_Anh', 'mean_delta_AIS']].dropna()
if len(valid2) > 5:
    xl = np.linspace(valid2['BDI_Anh'].min(), valid2['BDI_Anh'].max(), 100)
    z  = np.polyfit(valid2['BDI_Anh'], valid2['mean_delta_AIS'], 1)
    ax.plot(xl, np.polyval(z, xl), 'k-', alpha=0.6, lw=1.5,
            label=f'r={r_anh:.2f}, p={p_anh:.3f}')
ax.axhline(0, color='gray', ls='--', alpha=0.3, lw=1)
ax.set_xlabel('BDI_Anh (anhedonia score 0–8)')
ax.set_ylabel('Mean ΔAIS [bits]')
ax.set_title('C. Anhedonia vs ΔAIS')
ax.legend(fontsize=8)

# Panel D: Lag sensitivity bar
ax = axes[1, 0]
if lag_results:
    lags     = sorted(lag_results.keys())
    ds       = [lag_results[l]['d'] for l in lags]
    ps       = [lag_results[l]['p'] for l in lags]
    bar_cols = ['#4CAF50' if p < 0.05 else '#FF9800' if p < 0.10 else '#9E9E9E' for p in ps]
    ax.bar([str(l) for l in lags], ds, color=bar_cols, edgecolor='black', lw=0.5)
    ax.axhline(0,   color='gray', lw=0.8)
    ax.axhline(0.3, color='green', ls='--', alpha=0.5, lw=1, label="d=0.3")
    ax.set_xlabel('AIS lag (post window)')
    ax.set_ylabel("Cohen's d (CTL > MDD)")
    ax.set_title("D. Lag Sensitivity\n(green=p<0.05, orange=p<0.10, grey=ns)")
    ax.legend(fontsize=8)

# Panel E: ΔAIS distribution
ax = axes[1, 1]
for grp, color in [(CTL_LABEL, '#2196F3'), (MDD_LABEL, '#F44336')]:
    vals = df_all[df_all['group'] == grp]['mean_delta_AIS'].dropna()
    ax.hist(vals, bins=20, alpha=0.55, color=color, label=f'{grp} (N={len(vals)})',
            edgecolor='white', lw=0.3)
ax.axvline(0, color='black', ls='--', alpha=0.5, lw=1)
ax.set_xlabel('Mean ΔAIS [bits]')
ax.set_ylabel('Count')
ax.set_title('E. ΔAIS Distribution by Group')
ax.legend()

# Panel F: Text summary
ax = axes[1, 2]
ax.axis('off')
if len(df_res) > 0:
    lines  = ["HYPOTHESIS TEST SUMMARY", "=" * 34]
    for _, row in df_res.iterrows():
        lines += [
            f"\n{row['comparison']}:",
            f"  CTL={row['CTL_dAIS']:.4f}±{row['CTL_dAIS_std']:.4f}",
            f"  MDD={row['MDD_dAIS']:.4f}±{row['MDD_dAIS_std']:.4f}",
            f"  d={row['d']:.3f}, p_fdr={row.get('p_fdr', row['p_raw']):.4f} {row['H1_confirmed']}",
        ]
    lines += [
        "", "ANHEDONIA CORRELATION:",
        f"  r(ΔAIS, BDI_Anh)={r_anh:.3f}",
        f"  p_perm={p_anh:.4f}",
        f"  Confirmed (r<0): {'✓' if r_anh < 0 else '✗'}",
    ]
    ax.text(0.04, 0.97, "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUT_DIR / 'delta_ais_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUT_DIR / 'delta_ais_results.png'}")

# ── STEP 9: Save outputs ──────────────────────────────────────────────────
df_trial.to_csv(OUT_DIR / 'delta_ais_trial_level.csv',   index=False)
df_subj.to_csv( OUT_DIR / 'delta_ais_subject_level.csv', index=False)
df_all.to_csv(  OUT_DIR / 'delta_ais_aggregated.csv',    index=False)
df_res.to_csv(  OUT_DIR / 'delta_ais_stats.csv',         index=False)
print(f"CSVs saved to: {OUT_DIR}")

# ── STEP 10: Final summary ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ΔAIS ANALYSIS — COMPLETE")
print("=" * 55)
print(f"Trial records:    {len(df_trial):,}")
print(f"N subjects CTL:   {(df_all['group']==CTL_LABEL).sum()}")
print(f"N subjects MDD:   {(df_all['group']==MDD_LABEL).sum()}")
print(f"NaN AIS_pre:      {df_trial['AIS_pre'].isna().mean():.1%}")
print(f"NaN AIS_post:     {df_trial['AIS_post'].isna().mean():.1%}")
print(f"NaN delta_AIS:    {df_trial['delta_AIS'].isna().mean():.1%}")

if len(df_res) > 0:
    main_rows = df_res[df_res['comparison'] == 'All_conditions']
    if len(main_rows):
        main = main_rows.iloc[0]
        print(f"\nPrimary result (All conditions):")
        print(f"  ΔAIS CTL={main['CTL_dAIS']:.4f}  MDD={main['MDD_dAIS']:.4f}")
        print(f"  d={main['d']:.3f},  p_fdr={main.get('p_fdr', main['p_raw']):.4f}")
        print(f"  H1 (CTL>MDD): {main['H1_confirmed']}")

print(f"\nAnhedonia: r(ΔAIS, BDI_Anh)={r_anh:.3f}, p_perm={p_anh:.4f}")
print("=" * 55)
print("Next: run 13_ksg_te_asymmetry.py")
