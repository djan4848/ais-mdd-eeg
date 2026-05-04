#!/usr/bin/env python3
"""
14a_v2_meg_ais_pre_full.py
MEG AIS_pre replication — FULL SAMPLE (all 85 subjects on disk).

Fix vs 14a: the original script only globbed *_split-01_meg.fif, missing
all 55 single-file subjects. This version handles both:
  - single:  *_task-pst_run-1_meg.fif
  - split:   *_task-pst_run-1_split-01_meg.fif + split-02_meg.fif

Events are always in *_task-pst_run-1_events.tsv (no split suffix).
"""

import sys
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

mne.set_log_level('ERROR')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MEG   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/ds005356")
CLIN_PATH  = BASE_MEG / "Code/MEG MDD IDs and Quex.xlsx"
DERIV_DIR  = BASE_MEG / "derivatives"
CACHE_DIR  = DERIV_DIR / "epochs_ais"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
FCZ_CHAN     = 'EEG007'      # confirmed nearest midline channel (x=−5.3mm)
TARGET_SFREQ = 250
TMIN_EPO     = -1.0
TMAX_EPO     =  1.5
BASELINE     = (-0.5, -0.2)  # avoids zeroing the AIS_pre window
PRE_WINDOW   = (-0.200, 0.000)
WIN_CODE, LOSS_CODE = 8, 9
MIN_TRIALS   = 10

# ── AIS function (identical to Cavanagh EEG pipeline) ─────────────────────────
def safe_ais(x, lag=1, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * lag + 10 or np.std(x) < 1e-12:
        return np.nan
    try:
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return np.nan
        bins = np.digitize(x, edges[1:-1])
        x_t, x_lag = bins[lag:], bins[:-lag]
        joint = np.zeros((n_bins, n_bins))
        for a, b in zip(x_t, x_lag):
            ai = min(a - 1, n_bins - 1)
            bi = min(b - 1, n_bins - 1)
            joint[ai, bi] += 1
        joint /= joint.sum() + 1e-10
        px_t  = joint.sum(axis=1)
        px_lag = joint.sum(axis=0)
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and px_t[i] > 0 and px_lag[j] > 0:
                    mi += joint[i, j] * np.log2(
                        joint[i, j] / (px_t[i] * px_lag[j]))
        return float(mi) if np.isfinite(mi) else np.nan
    except Exception:
        return np.nan

rng = np.random.default_rng(42)
ar1 = np.zeros(300)
for i in range(1, 300):
    ar1[i] = 0.9 * ar1[i-1] + 0.1 * rng.standard_normal()
assert safe_ais(ar1) > safe_ais(rng.standard_normal(300)), "AIS sanity failed"
print("AIS sanity check OK")

# ── Statistical helpers ────────────────────────────────────────────────────────
def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def perm_r(x, y, n_perm=5000, seed=42):
    rng   = np.random.default_rng(seed)
    r_obs = pearsonr(x, y)[0]
    r_null = np.array([pearsonr(rng.permutation(x), y)[0] for _ in range(n_perm)])
    p = (np.sum(np.abs(r_null) >= abs(r_obs)) + 1) / (n_perm + 1)
    return r_obs, p

# ── Clinical data ─────────────────────────────────────────────────────────────
clinical = pd.read_excel(CLIN_PATH)
clinical['bids_id'] = 'sub-M87' + (100000 + clinical['URSI'].astype(int)).astype(str)
print(f"Clinical: N={len(clinical)} (CTL={sum(clinical['Group']=='CTL')}, "
      f"MDD={sum(clinical['Group']=='MDD')})")

# ── Build subject file map ────────────────────────────────────────────────────
print("\n=== MEG FILE AUDIT ===")
subject_map = {}   # bids_id → {'type': 'single'|'split', 'files': [...], 'tsv': Path}

for sub_dir in sorted(BASE_MEG.glob("sub-M87*")):
    sub_id  = sub_dir.name
    meg_dir = sub_dir / "ses-01" / "meg"
    if not meg_dir.exists():
        continue

    # Events TSV (always without split suffix)
    tsv = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_events.tsv"
    if not tsv.exists():
        continue   # no events → cannot epoch

    # Single file
    single = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_meg.fif"
    if single.exists():
        subject_map[sub_id] = {'type': 'single', 'files': [single], 'tsv': tsv}
        continue

    # Split files
    split01 = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_split-01_meg.fif"
    split02 = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_split-02_meg.fif"
    splits  = [f for f in [split01, split02] if f.exists()]
    if splits:
        subject_map[sub_id] = {'type': 'split', 'files': splits, 'tsv': tsv}

n_single = sum(1 for v in subject_map.values() if v['type'] == 'single')
n_split  = sum(1 for v in subject_map.values() if v['type'] == 'split')
print(f"Subjects with MEG + events TSV: {len(subject_map)}")
print(f"  Single-file: {n_single}")
print(f"  Split-file:  {n_split}")

# Cross-reference with clinical
has_clinical = {k for k in subject_map if k in set(clinical['bids_id'])}
print(f"With clinical data: {len(has_clinical)}")

# ── PHASE 1: Epoching (with cache) ────────────────────────────────────────────
print("\n=== PHASE 1: EPOCHING ===")

def load_and_epoch(sub_id, sub_info):
    """Load raw MEG, epoch around feedback, return cached path or None."""
    cache_path = CACHE_DIR / f"{sub_id}-epo.fif"
    if cache_path.exists():
        return cache_path, "cached"

    try:
        # Load raw (concatenate splits if needed)
        files = sub_info['files']
        if sub_info['type'] == 'single':
            raw = mne.io.read_raw_fif(files[0], preload=True, verbose='ERROR')
        else:
            raws = [mne.io.read_raw_fif(f, preload=True, verbose='ERROR') for f in files]
            raw  = mne.concatenate_raws(raws, verbose='ERROR')

        # Downsample
        if raw.info['sfreq'] > TARGET_SFREQ + 10:
            raw.resample(TARGET_SFREQ, verbose='ERROR')

        # Check FCz channel present
        if FCZ_CHAN not in raw.ch_names:
            return None, f"no {FCZ_CHAN}"

        # Read events from TSV (annotations are always empty in this dataset)
        ev_df   = pd.read_csv(str(sub_info['tsv']), sep='\t')
        fb_df   = ev_df[ev_df['trial_type'].isin(['FB/win', 'FB/loss'])].copy()
        if len(fb_df) == 0:
            return None, "no FB events in TSV"

        code_map = {'FB/win': WIN_CODE, 'FB/loss': LOSS_CODE}
        sfreq    = raw.info['sfreq']
        samples  = np.round(fb_df['onset'].values * sfreq).astype(int)
        codes    = fb_df['trial_type'].map(code_map).values.astype(int)
        fb_events = np.column_stack([samples, np.zeros(len(samples), int), codes])

        n_win  = int((codes == WIN_CODE).sum())
        n_loss = int((codes == LOSS_CODE).sum())

        if n_win + n_loss < MIN_TRIALS:
            return None, f"only {n_win+n_loss} feedback events"

        epo = mne.Epochs(
            raw, fb_events,
            event_id={'FB/win': WIN_CODE, 'FB/loss': LOSS_CODE},
            tmin=TMIN_EPO, tmax=TMAX_EPO,
            baseline=BASELINE,
            picks=[FCZ_CHAN],
            preload=True, verbose='ERROR',
            event_repeated='drop',
        )
        epo.drop_bad(verbose='ERROR')

        if len(epo) < MIN_TRIALS:
            return None, f"only {len(epo)} epochs after drop_bad"

        epo.save(cache_path, overwrite=True, verbose='ERROR')
        del raw, epo
        return cache_path, f"win={n_win}, loss={n_loss}"

    except Exception as e:
        return None, f"ERROR: {e}"

skipped = []
for sub_id in sorted(has_clinical):
    if sub_id not in subject_map:
        continue
    cache_path, status = load_and_epoch(sub_id, subject_map[sub_id])
    if cache_path is None:
        skipped.append((sub_id, status))
        print(f"  {sub_id}: skip — {status}")
    else:
        if status != "cached":
            print(f"  {sub_id}: epoched ({status})")

print(f"\nEpoching complete. Skipped: {len(skipped)}")

# ── PHASE 2: AIS_pre computation ──────────────────────────────────────────────
print("\n=== PHASE 2: AIS_PRE COMPUTATION ===")

records = []
cached_files = sorted(CACHE_DIR.glob("sub-M87*-epo.fif"))

for cache_path in cached_files:
    sub_id = cache_path.name.replace("-epo.fif", "")

    # Get clinical row
    row = clinical[clinical['bids_id'] == sub_id]
    if len(row) == 0:
        continue
    row = row.iloc[0]

    group = row['Group']   # 'CTL' or 'MDD'

    epo   = mne.read_epochs(cache_path, preload=True, verbose='ERROR')
    times = epo.times
    pre_mask = (times >= PRE_WINDOW[0]) & (times < PRE_WINDOW[1])

    data = epo.get_data()[:, 0, :]   # single channel (FCZ)

    all_ais = [safe_ais(t[pre_mask], lag=1, n_bins=4) for t in data]
    valid   = [v for v in all_ais if np.isfinite(v)]
    if len(valid) < MIN_TRIALS:
        continue

    n_win  = len(epo['FB/win'])  if 'FB/win'  in epo.event_id else 0
    n_loss = len(epo['FB/loss']) if 'FB/loss' in epo.event_id else 0

    # By condition AIS
    win_mask_ep  = epo.events[:, 2] == WIN_CODE
    loss_mask_ep = epo.events[:, 2] == LOSS_CODE
    win_ais  = [safe_ais(t[pre_mask], lag=1, n_bins=4) for t in data[win_mask_ep]]
    loss_ais = [safe_ais(t[pre_mask], lag=1, n_bins=4) for t in data[loss_mask_ep]]

    print(f"  {sub_id} ({group}): AIS_pre={np.mean(valid):.4f}, N={len(valid)}")

    records.append({
        'subject_id':       sub_id,
        'group':            group,
        'mean_AIS_pre':     np.mean(valid),
        'AIS_pre_win':      np.nanmean([v for v in win_ais  if np.isfinite(v)]) if win_ais  else np.nan,
        'AIS_pre_loss':     np.nanmean([v for v in loss_ais if np.isfinite(v)]) if loss_ais else np.nan,
        'n_trials':         len(valid),
        'n_win':            n_win,
        'n_loss':           n_loss,
        'BDI':              row.get('BDI', np.nan),
        'TEPS_anticipatory': row.get('TEPS_anticipatory', np.nan),
        'SHAPS':            row.get('SHAPS', np.nan),
        'DARS':             row.get('DARS_TOTAL_SCALE', np.nan),
        'MASQ_Anh':         row.get('MASQ_Anhedonic_Depression', np.nan),
    })

if not records:
    print("ERROR: no subjects processed"); sys.exit(1)

df = pd.DataFrame(records)
print(f"\nTotal: {len(df)}  (CTL={sum(df['group']=='CTL')}, MDD={sum(df['group']=='MDD')})")

# ── PHASE 3: Statistics ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("MEG AIS_PRE — FULL SAMPLE PRIMARY REPLICATION RESULT")
print("="*65)

ctl = df.loc[df['group'] == 'CTL', 'mean_AIS_pre'].dropna().values
mdd = df.loc[df['group'] == 'MDD', 'mean_AIS_pre'].dropna().values

U, p_mwu = mannwhitneyu(ctl, mdd, alternative='two-sided')
d         = cohen_d(ctl, mdd)

print(f"\nCTL AIS_pre: {ctl.mean():.4f} ± {ctl.std(ddof=1):.4f}  (N={len(ctl)})")
print(f"MDD AIS_pre: {mdd.mean():.4f} ± {mdd.std(ddof=1):.4f}  (N={len(mdd)})")
print(f"Cohen's d   = {d:.3f}")
print(f"MWU p       = {p_mwu:.4f}")
print(f"Direction CTL>MDD: {'✓' if ctl.mean() > mdd.mean() else '✗'}")

# Outlier check — sub-M87121835 was 0.427 in N=27 run
outlier_id = 'sub-M87121835'
if outlier_id in df['subject_id'].values:
    out_val = df.loc[df['subject_id'] == outlier_id, 'mean_AIS_pre'].values[0]
    out_grp = df.loc[df['subject_id'] == outlier_id, 'group'].values[0]
    print(f"\nOutlier {outlier_id} ({out_grp}): AIS_pre={out_val:.4f}")
    if out_grp == 'CTL':
        ctl_no = ctl[ctl != out_val]
        _, p_no = mannwhitneyu(ctl_no, mdd, alternative='two-sided')
        d_no    = cohen_d(ctl_no, mdd)
        print(f"  With outlier:    d={d:.3f}, p={p_mwu:.4f}")
        print(f"  Without outlier: d={d_no:.3f}, p={p_no:.4f}")

# By condition
print("\n=== BY FEEDBACK CONDITION ===")
for cond_label, cond_col in [('win (reward)', 'AIS_pre_win'), ('loss', 'AIS_pre_loss')]:
    a = df.loc[df['group'] == 'CTL', cond_col].dropna().values
    b = df.loc[df['group'] == 'MDD', cond_col].dropna().values
    if len(a) < 5 or len(b) < 5:
        continue
    _, p_c = mannwhitneyu(a, b, alternative='two-sided')
    d_c    = cohen_d(a, b)
    print(f"  {cond_label}: CTL={a.mean():.4f}, MDD={b.mean():.4f}, "
          f"d={d_c:.3f}, p={p_c:.4f}")

# Anhedonia correlations
print("\n=== ANHEDONIA CORRELATIONS ===")
corr_spec = [
    ('TEPS_anticipatory', 'TEPS_ant',   'positive'),
    ('SHAPS',             'SHAPS',      'positive'),
    ('DARS',              'DARS',       'positive'),
    ('MASQ_Anh',          'MASQ_Anh',   'negative'),
    ('BDI',               'BDI',        'negative'),
]
for col, name, expected in corr_spec:
    valid = df[['mean_AIS_pre', col]].dropna()
    if len(valid) < 10:
        continue
    r, p_r = perm_r(valid['mean_AIS_pre'].values, valid[col].values)
    rs, _  = spearmanr(valid['mean_AIS_pre'], valid[col])
    ok     = (expected == 'positive' and r > 0) or (expected == 'negative' and r < 0)
    print(f"  AIS_pre vs {name:<12s}: r={r:+.3f}, p_perm={p_r:.4f}, "
          f"rho={rs:+.3f}  {'✓' if ok else '✗'}  (N={len(valid)})")

# Robustness
print("\n=== ROBUSTNESS (parameter variants) ===")
variants = [
    ('−200ms/lag=1/bins=4', -0.200, 0.000, 1, 4),
    ('−500ms/lag=2/bins=4', -0.500, 0.000, 2, 4),
    ('−200ms/lag=1/bins=6', -0.200, 0.000, 1, 6),
    ('−200ms/lag=1/bins=8', -0.200, 0.000, 1, 8),
]
for label, t0, t1, lag, nb in variants:
    rob_vals = []
    for cache_path in cached_files:
        sub_id = cache_path.name.replace("-epo.fif", "")
        row = clinical[clinical['bids_id'] == sub_id]
        if len(row) == 0:
            continue
        grp = row.iloc[0]['Group']
        epo = mne.read_epochs(cache_path, preload=True, verbose='ERROR')
        mask = (epo.times >= t0) & (epo.times < t1)
        data = epo.get_data()[:, 0, :]
        ais  = [safe_ais(t[mask], lag=lag, n_bins=nb) for t in data]
        v    = [x for x in ais if np.isfinite(x)]
        if v:
            rob_vals.append({'group': grp, 'ais': np.mean(v)})
    df_r = pd.DataFrame(rob_vals)
    if len(df_r) < 10:
        continue
    a = df_r[df_r['group'] == 'CTL']['ais'].values
    b = df_r[df_r['group'] == 'MDD']['ais'].values
    _, p_r = mannwhitneyu(a, b, alternative='two-sided')
    d_r    = cohen_d(a, b)
    print(f"  {label}: CTL={a.mean():.4f}, MDD={b.mean():.4f}, d={d_r:.3f}, p={p_r:.4f}")

# ── PHASE 4: Figure ───────────────────────────────────────────────────────────
CTL_C = '#2196F3'
MDD_C = '#E53935'
GRAY  = '#555555'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    f'MEG AIS$_{{pre}}$ — Full Sample Replication (N={len(df)})\n'
    'Cavanagh PS Task · EEG007 (virtual FCz) · −200 ms to 0 ms pre-feedback',
    fontsize=11)

rng_j = np.random.default_rng(7)

# Panel A: Group comparison
ax = axes[0]
for grp, arr, x in [('CTL', ctl, 0), ('MDD', mdd, 1)]:
    col = CTL_C if grp == 'CTL' else MDD_C
    jit = rng_j.uniform(-0.14, 0.14, len(arr))
    ax.scatter(x + jit, arr, color=col, alpha=0.5, s=18, linewidths=0)
    m  = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    ax.errorbar(x, m, yerr=[[1.96*se],[1.96*se]], fmt='o', color='black',
                capsize=4, capthick=1.2, elinewidth=1.2, markersize=6, zorder=5)

y_top = max(ctl.max(), mdd.max()) * 1.05
ax.annotate('', xy=(1, y_top), xytext=(0, y_top),
            arrowprops=dict(arrowstyle='<->', color='black', lw=0.9))
p_str = f'p = {p_mwu:.3f}' if p_mwu >= 0.001 else 'p < 0.001'
ax.text(0.5, y_top * 1.01, f'd = {d:.2f},  {p_str}',
        ha='center', va='bottom', fontsize=8.5)
ax.set_xticks([0, 1])
ax.set_xticklabels([f'CTL\n(n={len(ctl)})', f'MDD\n(n={len(mdd)})'])
ax.set_ylabel('AIS$_{pre}$  [bits]')
ax.set_title(f'A.  Group comparison (full sample)\n(Previous N=27 run: d=+0.437)',
             loc='left', fontsize=10)

# Panel B: AIS_pre vs TEPS_ant
ax = axes[1]
valid_t = df[['mean_AIS_pre', 'TEPS_anticipatory', 'group']].dropna()
if len(valid_t) >= 10:
    for grp, col in [('CTL', CTL_C), ('MDD', MDD_C)]:
        sub = valid_t[valid_t['group'] == grp]
        ax.scatter(sub['TEPS_anticipatory'], sub['mean_AIS_pre'],
                   color=col, alpha=0.6, s=25, linewidths=0, label=grp)
    z  = np.polyfit(valid_t['TEPS_anticipatory'], valid_t['mean_AIS_pre'], 1)
    xl = np.linspace(valid_t['TEPS_anticipatory'].min(), valid_t['TEPS_anticipatory'].max(), 200)
    ax.plot(xl, np.polyval(z, xl), color=GRAY, lw=1.4)
    r_t, p_t = perm_r(valid_t['mean_AIS_pre'].values, valid_t['TEPS_anticipatory'].values)
    p_str = f'p = {p_t:.3f}' if p_t >= 0.001 else 'p < 0.001'
    ax.text(0.97, 0.95, f'r = {r_t:+.3f}\n{p_str}\nN = {len(valid_t)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.7))
ax.set_xlabel('TEPS Anticipatory')
ax.set_ylabel('AIS$_{pre}$  [bits]')
ax.set_title('B.  Anticipatory anhedonia\n(TEPS_ant, correct direction: r > 0)',
             loc='left', fontsize=10)
ax.legend(fontsize=8, frameon=False)

# Panel C: AIS_pre vs BDI
ax = axes[2]
valid_b = df[['mean_AIS_pre', 'BDI', 'group']].dropna()
if len(valid_b) >= 10:
    for grp, col in [('CTL', CTL_C), ('MDD', MDD_C)]:
        sub = valid_b[valid_b['group'] == grp]
        ax.scatter(sub['BDI'], sub['mean_AIS_pre'],
                   color=col, alpha=0.6, s=25, linewidths=0, label=grp)
    z  = np.polyfit(valid_b['BDI'], valid_b['mean_AIS_pre'], 1)
    xl = np.linspace(valid_b['BDI'].min(), valid_b['BDI'].max(), 200)
    ax.plot(xl, np.polyval(z, xl), color=GRAY, lw=1.4)
    r_b, p_b = perm_r(valid_b['mean_AIS_pre'].values, valid_b['BDI'].values)
    p_str = f'p = {p_b:.3f}' if p_b >= 0.001 else 'p < 0.001'
    ax.text(0.97, 0.95, f'r = {r_b:+.3f}\n{p_str}\nN = {len(valid_b)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.7))
ax.set_xlabel('BDI total')
ax.set_ylabel('AIS$_{pre}$  [bits]')
ax.set_title('B.  Depression severity\n(BDI, correct direction: r < 0)',
             loc='left', fontsize=10)
ax.legend(fontsize=8, frameon=False)

plt.tight_layout()
out_fig = DERIV_DIR / 'meg_ais_pre_full_sample_replication.png'
fig.savefig(out_fig, dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {out_fig}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_csv = DERIV_DIR / 'meg_ais_pre_full_sample_results.csv'
df.to_csv(out_csv, index=False)
print(f"Results saved: {out_csv}")

# ── FINAL VERDICT ─────────────────────────────────────────────────────────────
if   d >= 0.5 and ctl.mean() > mdd.mean() and p_mwu < 0.05:  verdict = "STRONG REPLICATION"
elif d >= 0.3 and ctl.mean() > mdd.mean():                    verdict = "PARTIAL REPLICATION"
elif ctl.mean() > mdd.mean():                                  verdict = "DIRECTIONAL REPLICATION"
else:                                                          verdict = "FAILURE TO REPLICATE"

print("\n" + "="*65)
print("MEG CAVANAGH — FULL SAMPLE FINAL VERDICT")
print("="*65)
print(f"EEG primary   (N={86}+{23}):   d=+0.874, p=0.0003")
print(f"MEG previous  (N=14+13):   d=+0.437, p=0.698  [N=27, underpowered]")
print(f"MEG full sample (N={len(ctl)}+{len(mdd)}):  d={d:+.3f}, p={p_mwu:.4f}")
print(f"Direction CTL>MDD: {'✓' if ctl.mean() > mdd.mean() else '✗'}")
print(f"VERDICT: {verdict}")
print("="*65)
