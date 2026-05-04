#!/usr/bin/env python3
"""
17_meg_searchlight_ais.py
AIS_pre searchlight across EEG and MEG channels.

For each channel in the search set, compute AIS_pre CTL vs MDD Cohen's d.
Answers:
  1. Is there a better EEG channel than EEG007 for the FCz-like effect?
  2. Do MEG gradiometers near FCz show larger d?
  3. Do Cavanagh 2025 sensors (MEG0511/0921, frontal-temporal) replicate?
  4. Does averaging a neighbourhood of channels improve p?

Exclusions: sub-M87121835 excluded from ALL channels.
  Reason: EEG007 on this subject shows sustained rhythmic oscillation
  making it 3x more self-predictable than any other subject; the channel
  appears to be recording a non-representative signal on EEG007, and
  until a per-channel audit is complete the subject is excluded wholesale.

Signal quality: safe_ais() returns NaN for std<1e-12 (flat channels).
  Additional check: per-channel outlier flagging (>median + 5*IQR → NaN).
"""

import sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr
from joblib import Parallel, delayed
import mne
mne.set_log_level('ERROR')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MEG  = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/ds005356")
CLIN_PATH = BASE_MEG / "Code/MEG MDD IDs and Quex.xlsx"
DERIV_DIR = BASE_MEG / "derivatives"

# ── Hard exclusions ───────────────────────────────────────────────────────────
EXCLUDE_SUBJECTS = {'sub-M87121835'}  # EEG007 sustained oscillation artifact

# ── AIS function ──────────────────────────────────────────────────────────────
def safe_ais(x, lag=1, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * lag + 10 or np.std(x) < 1e-12:
        return np.nan
    try:
        edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
        if len(edges) < 3:
            return np.nan
        bins = np.digitize(x, edges[1:-1])
        xt, xl = bins[lag:], bins[:-lag]
        j = np.zeros((n_bins, n_bins))
        for a, b in zip(xt, xl):
            j[min(a-1, n_bins-1), min(b-1, n_bins-1)] += 1
        j /= j.sum() + 1e-10
        pt, pl = j.sum(1), j.sum(0)
        mi = sum(j[i,k] * np.log2(j[i,k] / (pt[i]*pl[k]))
                 for i in range(n_bins) for k in range(n_bins)
                 if j[i,k]>0 and pt[i]>0 and pl[k]>0)
        return float(mi) if np.isfinite(mi) else np.nan
    except Exception:
        return np.nan

# ── Statistical helpers ────────────────────────────────────────────────────────
def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1-1)*np.var(a,ddof=1) + (n2-1)*np.var(b,ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.

def perm_r(x, y, n_perm=5000, seed=42):
    rng   = np.random.default_rng(seed)
    r_obs = pearsonr(x, y)[0]
    r_null = np.array([pearsonr(rng.permutation(x), y)[0] for _ in range(n_perm)])
    return r_obs, (np.sum(np.abs(r_null) >= abs(r_obs)) + 1) / (n_perm + 1)

def flag_outliers(vals):
    """Replace per-channel outliers (> median + 5*IQR) with NaN."""
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) < 4:
        return vals
    q25, q75 = np.percentile(finite, [25, 75])
    iqr = q75 - q25
    upper = np.median(finite) + 5 * iqr
    vals[vals > upper] = np.nan
    return vals

# ── Clinical data ─────────────────────────────────────────────────────────────
clinical = pd.read_excel(CLIN_PATH)
clinical['bids_id'] = 'sub-M87' + (100000 + clinical['URSI'].astype(int)).astype(str)
clinical = clinical[~clinical['bids_id'].isin(EXCLUDE_SUBJECTS)].reset_index(drop=True)
print(f"Clinical: N={len(clinical)} (CTL={sum(clinical['Group']=='CTL')}, "
      f"MDD={sum(clinical['Group']=='MDD')})  — excl. {EXCLUDE_SUBJECTS}")

# ── Build subject file map ────────────────────────────────────────────────────
subject_map = {}
for sub_dir in sorted(BASE_MEG.glob("sub-M87*")):
    sub_id = sub_dir.name
    if sub_id in EXCLUDE_SUBJECTS:
        continue
    meg_dir = sub_dir / "ses-01" / "meg"
    if not meg_dir.exists():
        continue
    tsv = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_events.tsv"
    if not tsv.exists():
        continue
    single = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_meg.fif"
    if single.exists():
        subject_map[sub_id] = {'type': 'single', 'files': [single], 'tsv': tsv}
        continue
    split01 = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_split-01_meg.fif"
    split02 = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_split-02_meg.fif"
    splits  = [f for f in [split01, split02] if f.exists()]
    if splits:
        subject_map[sub_id] = {'type': 'split', 'files': splits, 'tsv': tsv}

valid_ids = sorted(set(subject_map) & set(clinical['bids_id']))
print(f"Subjects for searchlight: {len(valid_ids)}")

# ── Define search channels ────────────────────────────────────────────────────
# EEG channels: all 71
EEG_CHANNELS = [f'EEG{str(i).zfill(3)}' for i in range(1, 72)]

# MEG channels near FCz (distance from FCz in mm, from audit):
#   MEG1041/42/43 triplet: 30.2mm  (right fronto-central)
#   MEG0631/32/33 triplet: 30.6mm  (left fronto-central)
#   MEG0721/22/23 triplet: 32.5mm  (right frontal)
#   MEG0711/12/13 triplet: 32.9mm  (left frontal)
#   MEG0621/22/23 triplet: 50.8mm  (frontal midline)
MEG_NEAR_FCZ = [
    'MEG1041','MEG1042','MEG1043',   # right fronto-central, 30.2mm
    'MEG0631','MEG0632','MEG0633',   # left fronto-central,  30.6mm
    'MEG0721','MEG0722','MEG0723',   # right frontal,        32.5mm
    'MEG0711','MEG0712','MEG0713',   # left frontal,         32.9mm
    'MEG0621','MEG0622','MEG0623',   # frontal midline,      50.8mm
]

# Cavanagh 2025 / user-specified sensors (frontal-temporal, ~114-139mm from FCz)
MEG_CAVANAGH = ['MEG0511','MEG0521','MEG0921','MEG0931']

SEARCH_CHANNELS = EEG_CHANNELS + MEG_NEAR_FCZ + MEG_CAVANAGH
print(f"Search channels: {len(EEG_CHANNELS)} EEG + {len(MEG_NEAR_FCZ)} MEG near FCz "
      f"+ {len(MEG_CAVANAGH)} MEG Cavanagh = {len(SEARCH_CHANNELS)} total")

# ── Per-subject worker ────────────────────────────────────────────────────────
TARGET_SFREQ = 250
TMIN_EPO     = -1.0
TMAX_EPO     =  1.5
BASELINE     = (-0.5, -0.2)
PRE_WIN      = (-0.200, 0.000)
WIN_CODE, LOSS_CODE = 8, 9
MIN_TRIALS   = 10

def process_one_subject(sub_id, sub_info, search_channels):
    """
    Load raw, epoch, compute AIS_pre for every search channel.
    Returns dict: {channel: AIS_pre_value}
    """
    try:
        # Load raw
        files = sub_info['files']
        if sub_info['type'] == 'single':
            raw = mne.io.read_raw_fif(files[0], preload=True, verbose='ERROR')
        else:
            raws = [mne.io.read_raw_fif(f, preload=True, verbose='ERROR') for f in files]
            raw  = mne.concatenate_raws(raws, verbose='ERROR')

        if raw.info['sfreq'] > TARGET_SFREQ + 10:
            raw.resample(TARGET_SFREQ, verbose='ERROR')

        # Keep only channels we need (speeds up epoching significantly)
        available = [c for c in search_channels if c in raw.ch_names]
        if not available:
            return None
        raw.pick_channels(available, ordered=False)

        # Events from TSV
        ev_df  = pd.read_csv(str(sub_info['tsv']), sep='\t')
        fb_df  = ev_df[ev_df['trial_type'].isin(['FB/win', 'FB/loss'])].copy()
        if len(fb_df) < MIN_TRIALS:
            return None
        code_map  = {'FB/win': WIN_CODE, 'FB/loss': LOSS_CODE}
        samples   = np.round(fb_df['onset'].values * raw.info['sfreq']).astype(int)
        codes     = fb_df['trial_type'].map(code_map).values.astype(int)
        fb_events = np.column_stack([samples, np.zeros(len(samples), int), codes])

        epo = mne.Epochs(
            raw, fb_events,
            event_id={'FB/win': WIN_CODE, 'FB/loss': LOSS_CODE},
            tmin=TMIN_EPO, tmax=TMAX_EPO,
            baseline=BASELINE,
            picks=available,
            preload=True, verbose='ERROR',
            event_repeated='drop',
        )
        epo.drop_bad(verbose='ERROR')
        if len(epo) < MIN_TRIALS:
            return None

        times    = epo.times
        pre_mask = (times >= PRE_WIN[0]) & (times < PRE_WIN[1])
        data_all = epo.get_data()   # shape: (n_epochs, n_channels, n_times)

        result = {}
        for ch_name in available:
            ch_idx = epo.ch_names.index(ch_name)
            ch_data = data_all[:, ch_idx, :]
            trial_ais = [safe_ais(t[pre_mask], lag=1, n_bins=4) for t in ch_data]
            valid = [v for v in trial_ais if np.isfinite(v)]
            result[ch_name] = np.mean(valid) if len(valid) >= MIN_TRIALS else np.nan

        del raw, epo
        return result

    except Exception as e:
        return None

# ── Run searchlight in parallel ───────────────────────────────────────────────
print(f"\n=== SEARCHLIGHT ({len(valid_ids)} subjects × {len(SEARCH_CHANNELS)} channels) ===")
t0 = time.time()

results_list = Parallel(n_jobs=-1, verbose=3)(
    delayed(process_one_subject)(sub_id, subject_map[sub_id], SEARCH_CHANNELS)
    for sub_id in valid_ids
)

elapsed = time.time() - t0
print(f"\nSearchlight done in {elapsed/60:.1f} min")

# Build subject × channel DataFrame
rows = []
for sub_id, res in zip(valid_ids, results_list):
    if res is None:
        continue
    row_c  = clinical[clinical['bids_id'] == sub_id]
    if len(row_c) == 0:
        continue
    row_c  = row_c.iloc[0]
    entry  = {'subject_id': sub_id, 'group': row_c['Group'],
              'BDI': row_c.get('BDI', np.nan),
              'TEPS_anticipatory': row_c.get('TEPS_anticipatory', np.nan)}
    entry.update(res)
    rows.append(entry)

df_full = pd.DataFrame(rows)
n_ok    = len(df_full)
print(f"Subjects with data: {n_ok}  (CTL={sum(df_full['group']=='CTL')}, "
      f"MDD={sum(df_full['group']=='MDD')})")

# ── Per-channel statistics ────────────────────────────────────────────────────
print("\n=== PER-CHANNEL STATISTICS ===")
results = []

for ch in SEARCH_CHANNELS:
    if ch not in df_full.columns:
        continue
    vals_raw = df_full[ch].values.astype(float)
    # Outlier flagging per channel (>median+5*IQR → NaN)
    vals = flag_outliers(vals_raw.copy())
    df_full[ch] = vals

    ctl = vals[df_full['group'].values == 'CTL']
    mdd = vals[df_full['group'].values == 'MDD']
    ctl = ctl[np.isfinite(ctl)]
    mdd = mdd[np.isfinite(mdd)]

    if len(ctl) < 5 or len(mdd) < 5:
        continue

    d    = cohen_d(ctl, mdd)
    _, p = mannwhitneyu(ctl, mdd, alternative='two-sided')
    n_ctl, n_mdd = len(ctl), len(mdd)

    results.append({
        'channel': ch,
        'ch_type': 'EEG' if ch.startswith('EEG') else 'MEG',
        'd': d, 'p': p,
        'n_ctl': n_ctl, 'n_mdd': n_mdd,
        'CTL_mean': ctl.mean(), 'MDD_mean': mdd.mean(),
    })

df_res = pd.DataFrame(results).sort_values('d', ascending=False)

# Print top 20 by d
print(f"\n{'Channel':<12} {'Type':<5} {'d':>6} {'p':>7} {'CTL_N':>5} {'MDD_N':>5}")
print("-" * 50)
for _, row in df_res.head(20).iterrows():
    marker = ' ◄' if row['channel'] == 'EEG007' else ''
    print(f"{row['channel']:<12} {row['ch_type']:<5} {row['d']:>+6.3f}  "
          f"{row['p']:>7.4f}  {row['n_ctl']:>5}  {row['n_mdd']:>5}{marker}")

# EEG007 reference
eeg007_row = df_res[df_res['channel'] == 'EEG007']
print(f"\nEEG007 (reference): d={eeg007_row['d'].values[0]:+.3f}, "
      f"p={eeg007_row['p'].values[0]:.4f}")

# Best EEG channel
best_eeg = df_res[df_res['ch_type']=='EEG'].iloc[0]
print(f"Best EEG channel:   {best_eeg['channel']}  d={best_eeg['d']:+.3f}, "
      f"p={best_eeg['p']:.4f}")

# Best MEG channel
best_meg = df_res[df_res['ch_type']=='MEG'].iloc[0]
print(f"Best MEG channel:   {best_meg['channel']}  d={best_meg['d']:+.3f}, "
      f"p={best_meg['p']:.4f}")

# Cavanagh 2025 sensor results
print("\nCavanagh 2025 sensors (MEG0511/0521/0921/0931):")
for ch in MEG_CAVANAGH:
    row = df_res[df_res['channel']==ch]
    if len(row):
        print(f"  {ch}: d={row['d'].values[0]:+.3f}, p={row['p'].values[0]:.4f}")

# ── Neighbourhood average: top-3 EEG channels ─────────────────────────────────
top3_eeg = df_res[df_res['ch_type']=='EEG']['channel'].head(3).tolist()
print(f"\n=== NEIGHBOURHOOD AVERAGE: top-3 EEG ({top3_eeg}) ===")
neigh_vals = df_full[top3_eeg].mean(axis=1).values
df_full['EEG_neigh'] = flag_outliers(neigh_vals)
ctl_n = df_full.loc[df_full['group']=='CTL', 'EEG_neigh'].dropna().values
mdd_n = df_full.loc[df_full['group']=='MDD', 'EEG_neigh'].dropna().values
d_n   = cohen_d(ctl_n, mdd_n)
_, p_n = mannwhitneyu(ctl_n, mdd_n, alternative='two-sided')
print(f"Neighbourhood avg: d={d_n:+.3f}, p={p_n:.4f}  "
      f"(CTL={len(ctl_n)}, MDD={len(mdd_n)})")

# ── Full stats for best channel ────────────────────────────────────────────────
best_ch = df_res.iloc[0]['channel']
print(f"\n=== FULL STATS — BEST CHANNEL: {best_ch} ===")
ctl_b = df_full.loc[df_full['group']=='CTL', best_ch].dropna().values
mdd_b = df_full.loc[df_full['group']=='MDD', best_ch].dropna().values
d_b   = cohen_d(ctl_b, mdd_b)
_, p_b = mannwhitneyu(ctl_b, mdd_b, alternative='two-sided')
print(f"CTL: {ctl_b.mean():.4f} ± {ctl_b.std(ddof=1):.4f} (N={len(ctl_b)})")
print(f"MDD: {mdd_b.mean():.4f} ± {mdd_b.std(ddof=1):.4f} (N={len(mdd_b)})")
print(f"d={d_b:+.3f}, p={p_b:.4f}")

# Anhedonia for best channel
for col, name in [('TEPS_anticipatory','TEPS_ant'), ('BDI','BDI')]:
    valid = df_full[[best_ch, col]].dropna()
    if len(valid) >= 10:
        r, pr = perm_r(valid[best_ch].values, valid[col].values)
        print(f"r(AIS_pre, {name}) = {r:+.3f}, p_perm={pr:.4f}, N={len(valid)}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: d-value bar chart, top 25 channels
ax = axes[0]
top25 = df_res.head(25)
colors = ['#2196F3' if r['ch_type']=='EEG' else '#E53935' for _, r in top25.iterrows()]
bars = ax.barh(range(len(top25)), top25['d'].values, color=colors, alpha=0.8)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25['channel'].values, fontsize=8)
ax.axvline(0, color='black', lw=0.8, ls='--')
ax.set_xlabel("Cohen's d  (CTL > MDD)")
ax.set_title(f'A.  Searchlight — top 25 channels by d\n'
             f'EEG=blue  MEG=red  (excl. {list(EXCLUDE_SUBJECTS)[0]})',
             loc='left', fontsize=10)
ax.invert_yaxis()
# Mark EEG007
for i, (_, row) in enumerate(top25.iterrows()):
    if row['channel'] == 'EEG007':
        ax.axhline(i, color='orange', lw=1.5, ls=':', alpha=0.8, label='EEG007 (ref.)')
        ax.legend(fontsize=8, frameon=False)
# Mark significant p<0.05
for i, (_, row) in enumerate(top25.iterrows()):
    if row['p'] < 0.05:
        ax.text(row['d'] + 0.01, i, '*', va='center', fontsize=12, color='black')

# Panel B: scatter EEG007 vs best channel
ax = axes[1]
if best_ch != 'EEG007' and best_ch in df_full.columns and 'EEG007' in df_full.columns:
    for grp, col in [('CTL','#2196F3'), ('MDD','#E53935')]:
        sub = df_full[df_full['group']==grp][[best_ch,'EEG007']].dropna()
        ax.scatter(sub['EEG007'], sub[best_ch], color=col, alpha=0.6, s=25,
                   linewidths=0, label=grp)
    ax.set_xlabel(f'EEG007 AIS_pre [bits]')
    ax.set_ylabel(f'{best_ch} AIS_pre [bits]')
    r_cmp = pearsonr(df_full[['EEG007',best_ch]].dropna()['EEG007'],
                     df_full[['EEG007',best_ch]].dropna()[best_ch])[0]
    ax.text(0.95, 0.05, f'r = {r_cmp:.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9)
    ax.set_title(f'B.  EEG007 vs best channel ({best_ch})\n'
                 f'EEG007: d={eeg007_row["d"].values[0]:+.3f}  '
                 f'{best_ch}: d={d_b:+.3f}', loc='left', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
else:
    ax.text(0.5, 0.5, f'Best channel = EEG007\n(no comparison needed)',
            ha='center', va='center', transform=ax.transAxes)

for a in axes:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.tight_layout()
out_fig = DERIV_DIR / 'meg_searchlight_ais.png'
fig.savefig(out_fig, dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {out_fig}")

# ── Save full results ─────────────────────────────────────────────────────────
df_res.to_csv(DERIV_DIR / 'meg_searchlight_results.csv', index=False)
df_full.to_csv(DERIV_DIR / 'meg_searchlight_subject_channel.csv', index=False)
print(f"Saved: meg_searchlight_results.csv")

# ── FINAL VERDICT ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SEARCHLIGHT — FINAL VERDICT")
print("="*65)
print(f"Reference EEG007:   d={eeg007_row['d'].values[0]:+.3f}, "
      f"p={eeg007_row['p'].values[0]:.4f}")
print(f"Best EEG channel:   {best_eeg['channel']}  d={best_eeg['d']:+.3f}, "
      f"p={best_eeg['p']:.4f}")
print(f"Best MEG channel:   {best_meg['channel']}  d={best_meg['d']:+.3f}, "
      f"p={best_meg['p']:.4f}")
print(f"EEG neighbourhood:  d={d_n:+.3f}, p={p_n:.4f}  ({top3_eeg})")
print("="*65)
