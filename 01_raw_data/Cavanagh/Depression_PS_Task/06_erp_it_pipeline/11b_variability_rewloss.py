import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from scipy.signal import hilbert
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE     = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/Depression_PS_Task")
EPO_DIR  = BASE / "derivatives/epochs"
OUT_DIR  = BASE / "derivatives/erp_it_cavanagh"
CLINICAL = BASE / "derivatives/clinical_lookup_ps_task.csv"

MIN_TRIALS = 5
GROUP_COL  = 'analysis_group_broad'
CTL_LABEL  = 'CTL'
MDD_LABEL  = 'MDD_any'

# ── STEP 1: Clinical ──────────────────────────────────────────────────────
clinical = pd.read_csv(CLINICAL)
print("Group values:", clinical[GROUP_COL].unique())

clinical_main = clinical[
    clinical[GROUP_COL].isin([CTL_LABEL, MDD_LABEL]) & (~clinical['excluded'])
].copy()
print(f"N CTL: {(clinical_main[GROUP_COL]==CTL_LABEL).sum()}")
print(f"N MDD: {(clinical_main[GROUP_COL]==MDD_LABEL).sum()}")

# ── STEP 2: Inspect one file ──────────────────────────────────────────────
sample_f = sorted(EPO_DIR.glob('*_task-ps_epo.fif'))[0]
epo_s    = mne.read_epochs(sample_f, preload=False, verbose='ERROR')
print("Event IDs:", epo_s.event_id)
print("Times:", epo_s.times[0], "to", epo_s.times[-1], "s | sfreq:", epo_s.info['sfreq'])

FCZ = 'FCz' if 'FCz' in epo_s.ch_names else 'Fz'
print(f"Using channel: {FCZ}")

REWARD_KEY = None
LOSS_KEY   = None
for k, v in epo_s.event_id.items():
    if v == 94  or 'reward'  in k.lower(): REWARD_KEY = k
    if v == 104 or 'loss'    in k.lower(): LOSS_KEY   = k
print(f"Reward key: {REWARD_KEY!r}, Loss key: {LOSS_KEY!r}")
assert REWARD_KEY and LOSS_KEY, "Could not identify event keys"

# ── HELPERS ───────────────────────────────────────────────────────────────
def cohens_d(a, b):
    n1, n2   = len(a), len(b)
    pooled   = np.sqrt(
        ((n1-1)*np.std(a, ddof=1)**2 + (n2-1)*np.std(b, ddof=1)**2)
        / (n1 + n2 - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def get_rewp_amp(data, times, tmin=0.200, tmax=0.400):
    """Per-trial mean amplitude in window. data: (n_trials, n_times)."""
    mask = (times >= tmin) & (times <= tmax)
    return data[:, mask].mean(axis=1)

def get_itc_theta(data, times, sfreq, tmin=0.0, tmax=0.6, fmin=4.0, fmax=8.0):
    """
    ITC in theta band via Hilbert on bandpassed signal.
    data: (n_trials, n_times). Returns scalar ITC.
    """
    data_filt = mne.filter.filter_data(
        data.astype(float), sfreq=sfreq, l_freq=fmin, h_freq=fmax, verbose=False
    )
    tmask    = (times >= tmin) & (times <= tmax)
    data_win = data_filt[:, tmask]
    analytic = hilbert(data_win, axis=1)
    phase    = np.angle(analytic)
    itc      = np.abs(np.mean(np.exp(1j * phase), axis=0)).mean()
    return float(itc)

def perm_r(x, y, n=5000, seed=42):
    rng  = np.random.default_rng(seed)
    r0   = pearsonr(x, y)[0]
    null = [pearsonr(rng.permutation(x), y)[0] for _ in range(n)]
    return r0, float(np.mean(np.abs(null) >= np.abs(r0)))

# ── STEP 3: Main extraction loop ──────────────────────────────────────────
records     = []
epoch_files = sorted(EPO_DIR.glob('*_task-ps_epo.fif'))
print(f"\nProcessing {len(epoch_files)} epoch files...")

for idx, fpath in enumerate(epoch_files):
    try:
        sub_id = int(fpath.stem.split('-')[1].split('_')[0])
    except (IndexError, ValueError):
        continue

    row_clin = clinical_main[clinical_main['subject_id'] == sub_id]
    if row_clin.empty:
        continue
    row_clin = row_clin.iloc[0]
    group    = row_clin[GROUP_COL]
    bdi_anh  = row_clin['BDI_Anh']
    bdi      = row_clin['BDI']

    try:
        epo = mne.read_epochs(fpath, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"  sub-{sub_id}: load error — {e}")
        continue

    ch_idx = epo.ch_names.index(FCZ)
    sfreq  = epo.info['sfreq']
    times  = epo.times

    for cond_name, event_key in (('Reward', REWARD_KEY), ('Loss', LOSS_KEY)):
        try:
            cond_epo = epo[event_key]
        except KeyError:
            continue
        if len(cond_epo) < MIN_TRIALS:
            continue

        data_2d = cond_epo.get_data()[:, ch_idx, :]   # (n_trials, n_times)

        # Measure 1 & 3: RewP/FRN window mean and std
        rewp_amps  = get_rewp_amp(data_2d, times, 0.200, 0.400)
        rewp_mean  = float(np.mean(rewp_amps))
        rewp_std   = float(np.std(rewp_amps, ddof=1))

        # Measure 3: per-trial peak (max for Reward/RewP, min for Loss/FRN)
        peak_mask = (times >= 0.200) & (times <= 0.400)
        if cond_name == 'Reward':
            peaks = data_2d[:, peak_mask].max(axis=1)
        else:
            peaks = data_2d[:, peak_mask].min(axis=1)
        peak_mean = float(np.mean(peaks))
        peak_std  = float(np.std(peaks, ddof=1))

        # Measure 2: ITC theta
        itc_theta = get_itc_theta(data_2d, times, sfreq, 0.0, 0.6, 4.0, 8.0)

        records.append({
            'subject_id':   sub_id,
            'group':        group,
            'BDI_Anh':      bdi_anh,
            'BDI':          bdi,
            'condition':    cond_name,
            'n_trials':     len(cond_epo),
            'rewp_mean_uV': rewp_mean,
            'rewp_std_uV':  rewp_std,
            'itc_theta':    itc_theta,
            'peak_mean_uV': peak_mean,
            'peak_std_uV':  peak_std,
        })

    if (idx + 1) % 20 == 0:
        print(f"  [{idx+1}/{len(epoch_files)}] processed")

df = pd.DataFrame(records)
print(f"\nTotal records: {len(df)}")
print(df.groupby(['group', 'condition'])[
    ['rewp_mean_uV', 'rewp_std_uV', 'itc_theta', 'peak_mean_uV', 'peak_std_uV']
].mean().round(3))

# ── STEP 4: Statistical comparisons ──────────────────────────────────────
print("\n=== STATISTICAL RESULTS ===")
all_results = []

# (metric_col, label, hypothesis_direction)
METRICS = [
    ('rewp_mean_uV', 'RewP_mean',  'CTL>MDD'),
    ('rewp_std_uV',  'RewP_std',   'CTL>MDD'),
    ('itc_theta',    'ITC_theta',  'MDD>CTL'),
    ('peak_mean_uV', 'Peak_mean',  'CTL>MDD'),
    ('peak_std_uV',  'Peak_std',   'CTL>MDD'),
]

for cond in ('Reward', 'Loss'):
    sub = df[df['condition'] == cond]
    ctl = sub[sub['group'] == CTL_LABEL]
    mdd = sub[sub['group'] == MDD_LABEL]

    for metric, label, h_dir in METRICS:
        a = ctl[metric].dropna().values
        b = mdd[metric].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        U, p = mannwhitneyu(a, b, alternative='two-sided')
        d    = cohens_d(a, b)
        confirmed = (
            (h_dir == 'CTL>MDD' and np.mean(a) > np.mean(b)) or
            (h_dir == 'MDD>CTL' and np.mean(b) > np.mean(a))
        )
        all_results.append({
            'condition': cond, 'metric': label, 'hypothesis': h_dir,
            'CTL_mean': np.mean(a), 'CTL_std': np.std(a, ddof=1),
            'MDD_mean': np.mean(b), 'MDD_std': np.std(b, ddof=1),
            'N_CTL': len(a), 'N_MDD': len(b),
            'd': d, 'p_raw': p, 'H_dir_confirmed': '✓' if confirmed else '✗',
        })

df_res               = pd.DataFrame(all_results)
_, p_fdr, _, _       = multipletests(df_res['p_raw'], method='fdr_bh')
df_res['p_fdr']      = p_fdr
df_res['sig_fdr']    = df_res['p_fdr'] < 0.05
df_res['trend_raw']  = df_res['p_raw'] < 0.10

print(df_res[[
    'condition', 'metric', 'N_CTL', 'N_MDD',
    'CTL_mean', 'MDD_mean', 'd', 'p_raw', 'p_fdr', 'H_dir_confirmed'
]].to_string(index=False))

sig   = df_res[df_res['sig_fdr']]
trend = df_res[df_res['trend_raw'] & ~df_res['sig_fdr']]

if len(sig) > 0:
    print(f"\n*** {len(sig)} SIGNIFICANT (p_fdr < 0.05) ***")
    print(sig[['condition', 'metric', 'd', 'p_raw', 'p_fdr', 'H_dir_confirmed']].to_string(index=False))
else:
    print("\nNo results survive FDR correction.")

if len(trend) > 0:
    print(f"Trend-level (p_raw < 0.10, not FDR-corrected):")
    print(trend[['condition', 'metric', 'd', 'p_raw', 'H_dir_confirmed']].to_string(index=False))

# ── STEP 5: BDI_Anh correlations ─────────────────────────────────────────
print("\n=== BDI_Anh CORRELATIONS (Reward condition) ===")
df_rew = df[df['condition'] == 'Reward'].copy()
anh_results = []
for metric in ('rewp_mean_uV', 'rewp_std_uV', 'itc_theta', 'peak_mean_uV', 'peak_std_uV'):
    valid = df_rew[['BDI_Anh', metric]].dropna()
    if len(valid) < 10:
        continue
    r, p_perm = perm_r(valid[metric].values, valid['BDI_Anh'].values)
    rs, _     = spearmanr(valid[metric], valid['BDI_Anh'])
    print(f"  {metric:<18} vs BDI_Anh: r={r:+.3f}, p_perm={p_perm:.4f}, rho={rs:+.3f}  (N={len(valid)})")
    anh_results.append({'metric': metric, 'r': r, 'p_perm': p_perm, 'rho': rs, 'N': len(valid)})

# ── STEP 6: Visualization ─────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
palette = {CTL_LABEL: '#2196F3', MDD_LABEL: '#F44336'}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Trial-by-Trial ERP Variability: MDD vs CTL\n(FCz, 200–400ms)',
             fontsize=13, fontweight='bold')

plot_spec = [
    ('rewp_mean_uV', 'RewP Mean Amp (µV)',   'Reward', axes[0, 0]),
    ('rewp_std_uV',  'RewP Trial Std (µV)',  'Reward', axes[0, 1]),
    ('itc_theta',    'ITC Theta (0–0.6s)',   'Reward', axes[0, 2]),
    ('rewp_mean_uV', 'FRN Mean Amp (µV)',    'Loss',   axes[1, 0]),
    ('rewp_std_uV',  'FRN Trial Std (µV)',   'Loss',   axes[1, 1]),
    ('itc_theta',    'ITC Theta — Loss',     'Loss',   axes[1, 2]),
]

metric_label_map = {
    'rewp_mean_uV': 'RewP_mean', 'rewp_std_uV': 'RewP_std', 'itc_theta': 'ITC_theta',
}

for metric, ylabel, cond, ax in plot_spec:
    sub = df[df['condition'] == cond]
    sns.boxplot(data=sub, x='group', y=metric, ax=ax,
                palette=palette, width=0.5, order=[CTL_LABEL, MDD_LABEL])
    sns.stripplot(data=sub, x='group', y=metric, ax=ax,
                  color='black', alpha=0.35, size=3, jitter=True,
                  order=[CTL_LABEL, MDD_LABEL])
    ax.set_title(f'{ylabel}\n({cond})')
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)

    lbl = metric_label_map.get(metric)
    if lbl:
        row = df_res[(df_res['condition'] == cond) & (df_res['metric'] == lbl)]
        if len(row):
            r  = row.iloc[0]
            ax.set_xlabel(f"d={r['d']:.2f}, p_fdr={r['p_fdr']:.3f} {r['H_dir_confirmed']}")

plt.tight_layout()
fig.savefig(OUT_DIR / 'variability_rewloss.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUT_DIR / 'variability_rewloss.png'}")

# ── STEP 7: Save results ──────────────────────────────────────────────────
df.to_csv(OUT_DIR / 'variability_subject_level.csv', index=False)
df_res.to_csv(OUT_DIR / 'variability_stats.csv', index=False)

print(f"Data saved:  {OUT_DIR / 'variability_subject_level.csv'}  ({len(df)} rows)")
print(f"Stats saved: {OUT_DIR / 'variability_stats.csv'}  ({len(df_res)} rows)")
print("\nScript 11b complete.")
print("Next: run 12_delta_ais_pre_post.py")
