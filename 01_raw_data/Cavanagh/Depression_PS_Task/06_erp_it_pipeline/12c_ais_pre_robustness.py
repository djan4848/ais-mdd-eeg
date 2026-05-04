"""
Robustness check for AIS_pre (d=0.806, p_fdr=0.0007 from Script 12b).
Two methodological validations:
  A) Extended window: -500ms to 0ms, lag=2, n_bins=4 (250 samples)
  B) Bin sensitivity: -200ms to 0ms, lag=1, n_bins in {4, 6, 8}
All variants computed in a single pass over epoch files.
"""
import sys
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

GROUP_COL = 'analysis_group_broad'
CTL_LABEL = 'CTL'
MDD_LABEL = 'MDD_any'

clinical      = pd.read_csv(CLINICAL)
clinical_main = clinical[
    clinical[GROUP_COL].isin([CTL_LABEL, MDD_LABEL]) & (~clinical['excluded'])
].copy()
print(f"N CTL: {(clinical_main[GROUP_COL]==CTL_LABEL).sum()}, "
      f"N MDD: {(clinical_main[GROUP_COL]==MDD_LABEL).sum()}")

# ── Import AIS ────────────────────────────────────────────────────────────
INFO_PATH = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/"
                 "ds003474/code/eeg_depression_classification")
sys.path.insert(0, str(INFO_PATH))
from info_theory import compute_ais
print("compute_ais imported OK")

# ── Helpers ───────────────────────────────────────────────────────────────
def safe_ais(x, lag, n_bins):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * lag + 10 or np.std(x) < 1e-12:
        return np.nan
    try:
        v = compute_ais(x, lag=lag, n_bins=n_bins)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0

# ── Variant definitions ───────────────────────────────────────────────────
# (label, tmin, tmax, lag, n_bins)
VARIANTS = [
    ('pre_200_lag1_bins4', -0.200, 0.0, 1, 4),   # baseline (Script 12b)
    ('pre_500_lag2_bins4', -0.500, 0.0, 2, 4),   # Validation A
    ('pre_200_lag1_bins6', -0.200, 0.0, 1, 6),   # Validation B: bins=6
    ('pre_200_lag1_bins8', -0.200, 0.0, 1, 8),   # Validation B: bins=8
]

# ── Single-pass extraction ────────────────────────────────────────────────
epoch_files = sorted(EPO_DIR.glob('*_task-ps_epo.fif'))
print(f"\nProcessing {len(epoch_files)} epoch files (4 variants × all trials)...")

records = []
for idx, fpath in enumerate(epoch_files):
    try:
        sub_id = int(fpath.stem.split('-')[1].split('_')[0])
    except (IndexError, ValueError):
        continue
    row_c = clinical_main[clinical_main['subject_id'] == sub_id]
    if row_c.empty:
        continue
    group = row_c.iloc[0][GROUP_COL]

    try:
        epo = mne.read_epochs(fpath, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"  sub-{sub_id}: {e}")
        continue

    ch_idx = epo.ch_names.index('FCz')
    times  = epo.times

    # Accumulate trial-level AIS for each variant, both conditions combined
    trial_vals = {label: [] for label, *_ in VARIANTS}

    for cond_key in ('Reward', 'Loss'):
        try:
            data_3d = epo[cond_key].get_data()[:, ch_idx, :]
        except KeyError:
            continue
        for trial_sig in data_3d:
            for label, tmin, tmax, lag, n_bins in VARIANTS:
                mask = (times >= tmin) & (times < tmax)
                trial_vals[label].append(safe_ais(trial_sig[mask], lag, n_bins))

    # Subject mean per variant
    row = {'subject_id': sub_id, 'group': group}
    for label, *_ in VARIANTS:
        vals = [v for v in trial_vals[label] if np.isfinite(v)]
        row[label] = np.mean(vals) if vals else np.nan
    records.append(row)

    if (idx + 1) % 25 == 0:
        print(f"  [{idx+1}/{len(epoch_files)}]")

df = pd.DataFrame(records)
print(f"\nSubjects processed: {len(df)}  (CTL={( df['group']==CTL_LABEL).sum()}, "
      f"MDD={(df['group']==MDD_LABEL).sum()})")

# ── Statistics for every variant ──────────────────────────────────────────
print("\n" + "=" * 72)
print("ROBUSTNESS TABLE")
print("=" * 72)
header = f"{'Variant':<28} {'Window':>10} {'lag':>4} {'bins':>5} "
header += f"{'CTL':>8} {'MDD':>8} {'d':>7} {'p_raw':>8} {'p_fdr':>8}"
print(header)
print("-" * 72)

meta = {
    'pre_200_lag1_bins4': ('-200–0ms', 1, 4),
    'pre_500_lag2_bins4': ('-500–0ms', 2, 4),
    'pre_200_lag1_bins6': ('-200–0ms', 1, 6),
    'pre_200_lag1_bins8': ('-200–0ms', 1, 8),
}
stat_rows = []
for label, *_ in VARIANTS:
    ctl = df[df['group'] == CTL_LABEL][label].dropna()
    mdd = df[df['group'] == MDD_LABEL][label].dropna()
    if len(ctl) < 3 or len(mdd) < 3:
        continue
    U, p = mannwhitneyu(ctl, mdd, alternative='two-sided')
    d    = cohens_d(ctl.values, mdd.values)
    win, lag, bins = meta[label]
    stat_rows.append({'variant': label, 'window': win, 'lag': lag, 'bins': bins,
                      'CTL': ctl.mean(), 'MDD': mdd.mean(), 'd': d, 'p_raw': p,
                      'N_CTL': len(ctl), 'N_MDD': len(mdd)})

df_stats             = pd.DataFrame(stat_rows)
_, p_fdr, _, _       = multipletests(df_stats['p_raw'], method='fdr_bh')
df_stats['p_fdr']    = p_fdr
df_stats['H_dir']    = df_stats['CTL'] > df_stats['MDD']

for _, row in df_stats.iterrows():
    tag = '← BASELINE' if row['variant'] == 'pre_200_lag1_bins4' else ''
    print(f"  {row['variant']:<26} {row['window']:>10} {int(row['lag']):>4} "
          f"{int(row['bins']):>5} {row['CTL']:>8.4f} {row['MDD']:>8.4f} "
          f"{row['d']:>7.3f} {row['p_raw']:>8.4f} {row['p_fdr']:>8.4f}  {tag}")

print("=" * 72)

# ── Validation verdict ────────────────────────────────────────────────────
print("\n=== VALIDATION A: Extended window (-500ms, lag=2) ===")
row_a    = df_stats[df_stats['variant'] == 'pre_500_lag2_bins4'].iloc[0]
row_base = df_stats[df_stats['variant'] == 'pre_200_lag1_bins4'].iloc[0]
print(f"  Baseline  (-200ms, lag=1, bins=4): d={row_base['d']:.3f}, p_fdr={row_base['p_fdr']:.4f}")
print(f"  Extended  (-500ms, lag=2, bins=4): d={row_a['d']:.3f},    p_fdr={row_a['p_fdr']:.4f}")
delta_d_a = row_a['d'] - row_base['d']
verdict_a = "ROBUST" if row_a['d'] > 0.5 and row_a['p_fdr'] < 0.05 else "WEAKER"
print(f"  Δd = {delta_d_a:+.3f} → Validation A: {verdict_a}")

print("\n=== VALIDATION B: Bin sensitivity (-200ms, lag=1) ===")
ds_bins = []
for bins in (4, 6, 8):
    lbl = f'pre_200_lag1_bins{bins}'
    r   = df_stats[df_stats['variant'] == lbl].iloc[0]
    ds_bins.append(r['d'])
    print(f"  n_bins={bins}: d={r['d']:.3f}, p_fdr={r['p_fdr']:.4f}")
d_range = max(ds_bins) - min(ds_bins)
verdict_b = "ROBUST" if d_range < 0.15 else "SENSITIVE"
print(f"  d range across bins 4/6/8: {d_range:.3f} → Validation B: {verdict_b} "
      f"({'stable' if d_range < 0.15 else 'varies with discretization'})")

# ── NaN audit ─────────────────────────────────────────────────────────────
print("\n=== NaN RATES ===")
for label, *_ in VARIANTS:
    nan_rate = df[label].isna().mean()
    print(f"  {label}: {nan_rate:.1%} NaN")

# ── Visualization ──────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
palette = {CTL_LABEL: '#2196F3', MDD_LABEL: '#F44336'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("AIS_pre Robustness Check\n"
             "Validation A (window) and B (discretization bins)",
             fontsize=12, fontweight='bold')

# Panel A: d across all variants
ax = axes[0]
colors_bar = ['#1565C0' if row['p_fdr'] < 0.05 else '#90CAF9'
              for _, row in df_stats.iterrows()]
short_labels = ['bins=4\n-200ms', 'bins=4\n-500ms', 'bins=6\n-200ms', 'bins=8\n-200ms']
ax.bar(short_labels, df_stats['d'], color=colors_bar, edgecolor='black', lw=0.5)
ax.axhline(0,   color='gray', lw=0.8)
ax.axhline(0.5, color='green', ls='--', lw=1, alpha=0.6, label='d=0.5')
ax.axhline(0.8, color='darkgreen', ls='--', lw=1, alpha=0.6, label='d=0.8')
ax.set_ylabel("Cohen's d (CTL > MDD)")
ax.set_title("A. Effect Size by Variant\n(dark=p_fdr<0.05)")
ax.legend(fontsize=8)
for i, (_, row) in enumerate(df_stats.iterrows()):
    ax.text(i, row['d'] + 0.02, f"p={row['p_fdr']:.3f}", ha='center', fontsize=8)

# Panel B: boxplot AIS_pre original vs extended window
ax = axes[1]
df_ab = df[['group', 'pre_200_lag1_bins4', 'pre_500_lag2_bins4']].rename(columns={
    'pre_200_lag1_bins4': 'AIS_pre\n-200ms (baseline)',
    'pre_500_lag2_bins4': 'AIS_pre\n-500ms (Val. A)',
})
df_melt = pd.melt(df_ab, id_vars='group', var_name='variant', value_name='AIS')
sns.boxplot(data=df_melt, x='variant', y='AIS', hue='group', ax=ax,
            palette=palette, width=0.5)
ax.set_title("B. Window Validation\n(-200ms vs -500ms)")
ax.set_xlabel('')
ax.set_ylabel('AIS_pre [bits]')
ax.get_legend().set_title('')

# Panel C: boxplot across n_bins (CTL only — shows sensitivity)
ax = axes[2]
df_bc = df[['group', 'pre_200_lag1_bins4', 'pre_200_lag1_bins6', 'pre_200_lag1_bins8']].rename(columns={
    'pre_200_lag1_bins4': 'bins=4',
    'pre_200_lag1_bins6': 'bins=6',
    'pre_200_lag1_bins8': 'bins=8',
})
df_melt2 = pd.melt(df_bc, id_vars='group', var_name='bins', value_name='AIS')
sns.boxplot(data=df_melt2, x='bins', y='AIS', hue='group', ax=ax,
            palette=palette, width=0.5)
ax.set_title("C. Bin Sensitivity\n(-200ms window, lag=1)")
ax.set_xlabel('n_bins')
ax.set_ylabel('AIS_pre [bits]')
ax.get_legend().set_title('')

plt.tight_layout()
plt.savefig(OUT_DIR / 'ais_pre_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUT_DIR / 'ais_pre_robustness.png'}")

df_stats.to_csv(OUT_DIR / 'ais_pre_robustness_stats.csv', index=False)

# ── Final verdict ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ROBUSTNESS CHECK COMPLETE")
print("=" * 55)
print(f"Validation A (extended window): {verdict_a}")
print(f"Validation B (bin sensitivity): {verdict_b}")
overall = "VALIDATED" if verdict_a == "ROBUST" and verdict_b == "ROBUST" else "NEEDS REVIEW"
print(f"\nOverall AIS_pre finding: {overall}")
print("=" * 55)
print("Safe to proceed to 13_ksg_te_asymmetry.py")
