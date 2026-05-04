import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE     = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/Depression_PS_Task")
OUT_DIR  = BASE / "derivatives/erp_it_cavanagh"
CLINICAL = BASE / "derivatives/clinical_lookup_ps_task.csv"

df       = pd.read_csv(OUT_DIR / 'delta_ais_aggregated.csv')
df_subj  = pd.read_csv(OUT_DIR / 'delta_ais_subject_level.csv')
clinical = pd.read_csv(CLINICAL)

print("Columns:", df.columns.tolist())
print("Groups:",  df['group'].unique())
print("N per group:", df['group'].value_counts().to_dict())

CTL_LABEL = 'CTL'
MDD_LABEL = 'MDD_any'

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0

def perm_r(x, y, n=5000, seed=42):
    rng  = np.random.default_rng(seed)
    r0   = pearsonr(x, y)[0]
    null = [pearsonr(rng.permutation(x), y)[0] for _ in range(n)]
    return r0, float(np.mean(np.abs(null) >= np.abs(r0)))

# ── TEST 1: AIS_pre ───────────────────────────────────────────────────────
print("\n=== AIS_pre: ANTICIPATORY PREPARATION ===")
ctl_pre = df[df['group'] == CTL_LABEL]['mean_AIS_pre'].dropna()
mdd_pre = df[df['group'] == MDD_LABEL]['mean_AIS_pre'].dropna()
U_pre, p_pre = mannwhitneyu(ctl_pre, mdd_pre, alternative='two-sided')
d_pre = cohens_d(ctl_pre.values, mdd_pre.values)
print(f"CTL AIS_pre = {ctl_pre.mean():.4f} ± {ctl_pre.std():.4f}  (N={len(ctl_pre)})")
print(f"MDD AIS_pre = {mdd_pre.mean():.4f} ± {mdd_pre.std():.4f}  (N={len(mdd_pre)})")
print(f"d={d_pre:.3f}, p={p_pre:.4f}")
print(f"H (CTL>MDD): {'✓' if ctl_pre.mean() > mdd_pre.mean() else '✗'}")

# ── TEST 2: AIS_post ──────────────────────────────────────────────────────
print("\n=== AIS_post: POST-FEEDBACK DYNAMICS ===")
ctl_post = df[df['group'] == CTL_LABEL]['mean_AIS_post'].dropna()
mdd_post = df[df['group'] == MDD_LABEL]['mean_AIS_post'].dropna()
U_post, p_post = mannwhitneyu(ctl_post, mdd_post, alternative='two-sided')
d_post = cohens_d(ctl_post.values, mdd_post.values)
print(f"CTL AIS_post = {ctl_post.mean():.4f} ± {ctl_post.std():.4f}  (N={len(ctl_post)})")
print(f"MDD AIS_post = {mdd_post.mean():.4f} ± {mdd_post.std():.4f}  (N={len(mdd_post)})")
print(f"d={d_post:.3f}, p={p_post:.4f}")
print(f"H (CTL>MDD): {'✓' if ctl_post.mean() > mdd_post.mean() else '✗'}")

# ── TEST 3: Comparative table ─────────────────────────────────────────────
print("\n=== COMPARATIVE TABLE ===")
comparisons = {
    'AIS_pre  (anticipation)':  (ctl_pre,  mdd_pre),
    'AIS_post (post-feedback)': (ctl_post, mdd_post),
    'delta_AIS (disruption)': (
        df[df['group'] == CTL_LABEL]['mean_delta_AIS'].dropna(),
        df[df['group'] == MDD_LABEL]['mean_delta_AIS'].dropna(),
    ),
}
rows = []
for name, (a, b) in comparisons.items():
    U, p = mannwhitneyu(a, b, alternative='two-sided')
    d    = cohens_d(a.values, b.values)
    rows.append({'measure': name, 'CTL': a.mean(), 'MDD': b.mean(),
                 'd': d, 'p': p, 'H_confirmed': a.mean() > b.mean()})

df_comp             = pd.DataFrame(rows)
_, p_fdr, _, _      = multipletests(df_comp['p'], method='fdr_bh')
df_comp['p_fdr']    = p_fdr
print(df_comp.to_string(index=False))

best = df_comp.loc[df_comp['d'].abs().idxmax()]
print(f"\nLargest effect: {best['measure'].strip()}  (d={best['d']:.3f})")

# ── TEST 4: BDI_Anh correlations ──────────────────────────────────────────
print("\n=== ANHEDONIA CORRELATIONS ===")
anh_rows = []
for col, name in [
    ('mean_AIS_pre',   'AIS_pre'),
    ('mean_AIS_post',  'AIS_post'),
    ('mean_delta_AIS', 'delta_AIS'),
]:
    valid = df[['BDI_Anh', col]].dropna()
    if len(valid) < 10:
        continue
    r, p_perm = perm_r(valid[col].values, valid['BDI_Anh'].values)
    print(f"  {name:<18} vs BDI_Anh: r={r:+.3f}, p_perm={p_perm:.4f}  (N={len(valid)})")
    anh_rows.append({'measure': name, 'r': r, 'p_perm': p_perm})

# ── TEST 5: AIS_pre by condition ──────────────────────────────────────────
print("\n=== AIS_pre BY CONDITION ===")
cond_rows = []
for cond in ('Reward', 'Loss'):
    sub = df_subj[df_subj['condition'] == cond]
    a   = sub[sub['group'] == CTL_LABEL]['mean_AIS_pre'].dropna()
    b   = sub[sub['group'] == MDD_LABEL]['mean_AIS_pre'].dropna()
    if len(a) < 3 or len(b) < 3:
        continue
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    d    = cohens_d(a.values, b.values)
    print(f"  {cond}: CTL={a.mean():.4f} ± {a.std():.4f}, "
          f"MDD={b.mean():.4f} ± {b.std():.4f}, d={d:.3f}, p={p:.4f}")
    cond_rows.append({'condition': cond, 'CTL': a.mean(), 'MDD': b.mean(), 'd': d, 'p': p})

print("Prediction: AIS_pre_Reward shows larger CTL>MDD effect than Loss")
print("(CTL anticipates reward structure more actively)")
if cond_rows:
    df_cond_stat = pd.DataFrame(cond_rows)
    rew_d = df_cond_stat[df_cond_stat['condition']=='Reward']['d'].values[0]
    los_d = df_cond_stat[df_cond_stat['condition']=='Loss']['d'].values[0]
    print(f"Confirmed: {'✓' if rew_d > los_d else '✗'}  (d_Reward={rew_d:.3f} vs d_Loss={los_d:.3f})")

# ── VISUALIZATION ─────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
palette = {CTL_LABEL: '#2196F3', MDD_LABEL: '#F44336'}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    'AIS Components: Anticipatory Preparation vs Post-Feedback Dynamics\n'
    'Hypothesis: CTL shows higher pre-feedback self-predictability (AIS_pre)',
    fontsize=11, fontweight='bold'
)

# Panel A: All three measures
rename_map = {
    'mean_AIS_pre':   'AIS_pre\n(anticipation)',
    'mean_AIS_post':  'AIS_post\n(post-feedback)',
    'mean_delta_AIS': 'ΔAIS\n(disruption)',
}
df_melt = pd.melt(
    df[['group'] + list(rename_map.keys())].rename(columns=rename_map),
    id_vars='group', var_name='measure', value_name='value'
)
measure_order = list(rename_map.values())
sns.boxplot(data=df_melt, x='measure', y='value', hue='group',
            ax=axes[0], palette=palette, width=0.5, order=measure_order)
axes[0].set_title('A. All AIS Components')
axes[0].set_xlabel('')
axes[0].set_ylabel('bits')
axes[0].get_legend().set_title('')

y_top = df_melt['value'].quantile(0.97)
for i, (_, row) in enumerate(df_comp.iterrows()):
    axes[0].text(i, y_top * 1.01, f"d={row['d']:.2f}",
                 ha='center', fontsize=8.5, color='black')

# Panel B: AIS_pre by condition
df_cond_plot = df_subj.groupby(['subject_id', 'group', 'condition'])['mean_AIS_pre'].mean().reset_index()
sns.boxplot(data=df_cond_plot, x='condition', y='mean_AIS_pre',
            hue='group', ax=axes[1], palette=palette, width=0.5)
sns.stripplot(data=df_cond_plot, x='condition', y='mean_AIS_pre',
              hue='group', ax=axes[1], dodge=True, alpha=0.3, size=3,
              palette=palette, legend=False)
axes[1].set_title('B. AIS_pre by Condition\n(anticipatory preparation)')
axes[1].set_ylabel('AIS_pre [bits]')
axes[1].set_xlabel('')
axes[1].get_legend().set_title('')

# Panel C: BDI_Anh vs AIS_pre
valid_plot = df[['BDI_Anh', 'mean_AIS_pre', 'group']].dropna()
for grp, color in [(CTL_LABEL, '#2196F3'), (MDD_LABEL, '#F44336')]:
    sub = valid_plot[valid_plot['group'] == grp]
    axes[2].scatter(sub['BDI_Anh'], sub['mean_AIS_pre'],
                    c=color, alpha=0.55, s=45, label=grp)
if len(valid_plot) > 5:
    z  = np.polyfit(valid_plot['BDI_Anh'], valid_plot['mean_AIS_pre'], 1)
    xl = np.linspace(valid_plot['BDI_Anh'].min(), valid_plot['BDI_Anh'].max(), 100)
    r_pre, p_pre_r = perm_r(valid_plot['mean_AIS_pre'].values, valid_plot['BDI_Anh'].values)
    axes[2].plot(xl, np.polyval(z, xl), 'k-', alpha=0.6, lw=1.5,
                 label=f'r={r_pre:.2f}, p={p_pre_r:.3f}')
axes[2].set_xlabel('BDI_Anh (anhedonia score 0–8)')
axes[2].set_ylabel('AIS_pre [bits]')
axes[2].set_title('C. Anhedonia vs AIS_pre')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / 'ais_pre_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUT_DIR / 'ais_pre_analysis.png'}")

df_comp.to_csv(OUT_DIR / 'ais_pre_stats.csv', index=False)

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("SCRIPT 12b COMPLETE")
print("=" * 50)
print(f"{'Measure':<28}  {'d':>6}  {'p_fdr':>7}  H")
for _, row in df_comp.iterrows():
    print(f"  {row['measure']:<26}  {row['d']:>+6.3f}  {row['p_fdr']:>7.4f}  "
          f"{'✓' if row['H_confirmed'] else '✗'}")
print(f"\nLargest effect: {best['measure'].strip()}  (d={best['d']:.3f})")
print("If AIS_pre d > delta_AIS d → AIS_pre is the primary biomarker")
print("Next: run 13_ksg_te_asymmetry.py")
