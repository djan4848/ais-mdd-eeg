"""
Script 14a: AIS_pre replication — Cavanagh MEG (ds005356)
Same PST paradigm, independent cohort (N=90: CTL=38, MDD=52).
FCz equivalent: EEG007 (x=-5.3mm, essentially midline, dist=28mm from FCz).
Primary target: replicate CTL>MDD AIS_pre (d=0.874, p=0.0003) from EEG.
"""
import sys
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_MEG  = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/ds005356")
EPO_OUT   = BASE_MEG / "derivatives/epochs_ais"
OUT_DIR   = BASE_MEG / "derivatives"
CLIN_PATH = BASE_MEG / "Code/MEG MDD IDs and Quex.xlsx"

EPO_OUT.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FCz equivalent confirmed by position (x=-5.3mm, y=47.4mm, z=114.7mm)
FCZ_CHAN = 'EEG007'

# Import AIS (Shannon estimator validated by KSG r=0.962 in Script 13)
INFO_PATH = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/"
                 "ds003474/code/eeg_depression_classification")
sys.path.insert(0, str(INFO_PATH))
from info_theory import compute_ais
print("compute_ais imported OK")

def safe_ais(x, lag=1, n_bins=4):
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

def perm_r(x, y, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    r0 = pearsonr(x, y)[0]
    null = [pearsonr(rng.permutation(x), y)[0] for _ in range(n)]
    return r0, float(np.mean(np.abs(null) >= np.abs(r0)))

# AIS sanity check
rng0 = np.random.default_rng(99)
ar1  = np.zeros(200)
for i in range(1, 200):
    ar1[i] = 0.9 * ar1[i-1] + 0.1 * rng0.standard_normal()
wn = rng0.standard_normal(200)
assert safe_ais(ar1) > safe_ais(wn), "AIS sanity check FAILED"
print("AIS sanity check OK")

# ── Clinical ──────────────────────────────────────────────────────────────
clinical = pd.read_excel(CLIN_PATH)
clinical['bids_id'] = clinical['URSI'].apply(lambda u: f"sub-M87{100000 + int(u)}")
clinical_main = clinical[clinical['Group'].isin(['CTL', 'MDD'])].copy()
print(f"Clinical: CTL={( clinical_main['Group']=='CTL').sum()}, "
      f"MDD={(clinical_main['Group']=='MDD').sum()}")

# ── PHASE 1: Epoching ─────────────────────────────────────────────────────
print("\n=== PHASE 1: EPOCHING ===")
meg_files = sorted(BASE_MEG.glob(
    "sub-*/ses-01/meg/*_task-pst_run-1_split-01_meg.fif"))
print(f"Found {len(meg_files)} MEG raw files")

TMIN, TMAX   = -1.0, 1.5
BASELINE     = (-0.5, -0.2)   # avoids zeroing the -200ms to 0ms AIS_pre window
TARGET_SFREQ = 250.0

for fpath in meg_files:
    sub_id   = fpath.name.split('_')[0]
    out_file = EPO_OUT / f"{sub_id}_task-pst_epo.fif"

    if out_file.exists():
        print(f"  {sub_id}: cached")
        continue

    row_c = clinical_main[clinical_main['bids_id'] == sub_id]
    if row_c.empty:
        print(f"  {sub_id}: no clinical match, skip")
        continue

    try:
        raw = mne.io.read_raw_fif(str(fpath), preload=True, verbose='ERROR')
    except Exception as e:
        print(f"  {sub_id}: read error — {e}")
        continue

    if raw.info['sfreq'] > 300:
        raw.resample(TARGET_SFREQ, verbose='ERROR')

    if FCZ_CHAN not in raw.ch_names:
        print(f"  {sub_id}: {FCZ_CHAN} not found, skip")
        continue

    # Events are not in annotations — read from BIDS sidecar TSV (at 1000Hz original)
    events_tsv = fpath.parent / fpath.name.replace('_split-01_meg.fif', '_events.tsv')
    if not events_tsv.exists():
        print(f"  {sub_id}: no events.tsv at {events_tsv.name}, skip")
        continue
    try:
        ev_df  = pd.read_csv(str(events_tsv), sep='\t')
        fb_df  = ev_df[ev_df['trial_type'].isin(['FB/win', 'FB/loss'])].copy()
        if len(fb_df) < 20:
            print(f"  {sub_id}: only {len(fb_df)} feedback events in TSV, skip")
            continue
        WIN_CODE, LOSS_CODE = 8, 9
        ev_map   = {'win': WIN_CODE, 'loss': LOSS_CODE}
        code_map = {'FB/win': WIN_CODE, 'FB/loss': LOSS_CODE}
        # Convert onset (seconds) → samples at TARGET_SFREQ (raw already resampled)
        samples   = np.round(fb_df['onset'].values * TARGET_SFREQ).astype(int)
        codes     = fb_df['trial_type'].map(code_map).values.astype(int)
        fb_events = np.column_stack([samples, np.zeros(len(samples), int), codes])
    except Exception as e:
        print(f"  {sub_id}: events TSV error — {e}")
        continue

    if len(fb_events) < 20:
        print(f"  {sub_id}: only {len(fb_events)} feedback events, skip")
        continue

    try:
        epo = mne.Epochs(
            raw, fb_events, event_id=ev_map,
            tmin=TMIN, tmax=TMAX,
            baseline=BASELINE,
            picks=[FCZ_CHAN],
            preload=True, verbose='ERROR'
        )
        epo.drop_bad(verbose='ERROR')
    except Exception as e:
        print(f"  {sub_id}: epoch error — {e}")
        continue

    if len(epo) < 20:
        print(f"  {sub_id}: only {len(epo)} epochs after drop_bad, skip")
        continue

    epo.save(str(out_file), overwrite=True, verbose='ERROR')
    print(f"  {sub_id}: {len(epo)} epochs "
          f"(win={len(epo['win'])}, loss={len(epo['loss'])})")

epo_files = sorted(EPO_OUT.glob('*_task-pst_epo.fif'))
print(f"\nEpoching complete: {len(epo_files)} subjects cached")

# ── PHASE 2: AIS_pre computation ──────────────────────────────────────────
print("\n=== PHASE 2: AIS_PRE COMPUTATION ===")
PRE_TMIN, PRE_TMAX = -0.200, 0.000
MIN_TRIALS = 10

records = []
for fpath in epo_files:
    sub_id = fpath.stem.split('_')[0]
    row_c  = clinical_main[clinical_main['bids_id'] == sub_id]
    if row_c.empty:
        continue
    rc    = row_c.iloc[0]
    group = rc['Group']

    epo      = mne.read_epochs(str(fpath), preload=True, verbose='ERROR')
    times    = epo.times
    pre_mask = (times >= PRE_TMIN) & (times < PRE_TMAX)

    cond_ais = {}
    for cond_label in ('win', 'loss', 'combined'):
        if cond_label == 'combined':
            data_c = epo.get_data()[:, 0, pre_mask]
        else:
            try:
                data_c = epo[cond_label].get_data()[:, 0, pre_mask]
            except KeyError:
                continue
        vals = [safe_ais(t, lag=1, n_bins=4) for t in data_c]
        valid = [v for v in vals if np.isfinite(v)]
        cond_ais[cond_label] = np.mean(valid) if len(valid) >= MIN_TRIALS else np.nan

    rec = {
        'subject_id':   sub_id,
        'group':        group,
        'mean_AIS_pre': cond_ais.get('combined', np.nan),
        'AIS_pre_win':  cond_ais.get('win',      np.nan),
        'AIS_pre_loss': cond_ais.get('loss',     np.nan),
        'n_trials':     len(epo),
        'BDI':          rc.get('BDI',                      np.nan),
        'SHAPS':        rc.get('SHAPS',                    np.nan),
        'TEPS_ant':     rc.get('TEPS_anticipatory',        np.nan),
        'DARS':         rc.get('DARS_TOTAL_SCALE',         np.nan),
        'MASQ_Anh':     rc.get('MASQ_Anhedonic_Depression',np.nan),
    }
    records.append(rec)
    print(f"  {sub_id} ({group}): AIS_pre={rec['mean_AIS_pre']:.4f}, N={len(epo)}")

df = pd.DataFrame(records) if records else pd.DataFrame(
    columns=['subject_id','group','mean_AIS_pre','AIS_pre_win','AIS_pre_loss',
             'n_trials','BDI','SHAPS','TEPS_ant','DARS','MASQ_Anh'])
print(f"\nTotal: {len(df)}  "
      f"(CTL={(df['group']=='CTL').sum()}, MDD={(df['group']=='MDD').sum()})")
if df.empty:
    print("ERROR: No subjects processed — check epoching phase above.")
    import sys; sys.exit(1)

# ── PHASE 3: Statistics ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MEG AIS_PRE — PRIMARY REPLICATION RESULT")
print("=" * 65)

ctl = df[df['group'] == 'CTL']['mean_AIS_pre'].dropna()
mdd = df[df['group'] == 'MDD']['mean_AIS_pre'].dropna()

if len(ctl) >= 3 and len(mdd) >= 3:
    U, p = mannwhitneyu(ctl, mdd, alternative='two-sided')
    d    = cohens_d(ctl.values, mdd.values)
    print(f"\nCTL AIS_pre: {ctl.mean():.4f} ± {ctl.std():.4f}  (N={len(ctl)})")
    print(f"MDD AIS_pre: {mdd.mean():.4f} ± {mdd.std():.4f}  (N={len(mdd)})")
    print(f"Cohen's d   = {d:.3f}")
    print(f"p           = {p:.4f}")
    print(f"Direction CTL>MDD: {'✓' if ctl.mean() > mdd.mean() else '✗'}")
else:
    print("Insufficient subjects for group comparison")
    d, p = np.nan, np.nan

# By feedback condition
print("\n=== BY FEEDBACK CONDITION ===")
for cond, col in [('win (reward)', 'AIS_pre_win'), ('loss', 'AIS_pre_loss')]:
    a = df[df['group'] == 'CTL'][col].dropna()
    b = df[df['group'] == 'MDD'][col].dropna()
    if len(a) < 3 or len(b) < 3:
        continue
    _, p_c = mannwhitneyu(a, b, alternative='two-sided')
    d_c    = cohens_d(a.values, b.values)
    print(f"  {cond}: CTL={a.mean():.4f}, MDD={b.mean():.4f}, "
          f"d={d_c:.3f}, p={p_c:.4f}")

# Anhedonia correlations
print("\n=== ANHEDONIA CORRELATIONS ===")
# expected_sign: TEPS_ant/DARS r>0 (more pleasure → more AIS)
#                SHAPS/BDI/MASQ r<0 (more anhedonia/depression → less AIS)
anhedonia_map = [
    ('TEPS_ant',  'TEPS_anticipatory',        '+'),
    ('SHAPS',     'SHAPS',                    '-'),
    ('DARS',      'DARS_TOTAL_SCALE',         '+'),
    ('MASQ_Anh',  'MASQ_Anhedonic_Depress.',  '-'),
    ('BDI',       'BDI',                      '-'),
]
for df_col, label, expected_sign in anhedonia_map:
    if df_col not in df.columns:
        continue
    valid = df[['mean_AIS_pre', df_col]].dropna()
    if len(valid) < 8:
        continue
    r, p_r = perm_r(valid['mean_AIS_pre'].values, valid[df_col].values)
    rs     = spearmanr(valid['mean_AIS_pre'], valid[df_col])[0]
    match  = '✓' if (expected_sign == '+' and r > 0) or \
                    (expected_sign == '-' and r < 0) else '✗'
    print(f"  AIS_pre vs {label:<28}: r={r:+.3f}, p_perm={p_r:.4f}, "
          f"rho={rs:+.3f}  {match}  (N={len(valid)})")

# Robustness
print("\n=== ROBUSTNESS (parameter variants) ===")
for label, tmin, tmax, lag, bins in [
    ('−200ms/lag=1/bins=4', -0.200, 0.000, 1, 4),
    ('−500ms/lag=2/bins=4', -0.500, 0.000, 2, 4),
    ('−200ms/lag=1/bins=6', -0.200, 0.000, 1, 6),
    ('−200ms/lag=1/bins=8', -0.200, 0.000, 1, 8),
]:
    rob_vals = []
    for fp in epo_files:
        sid  = fp.stem.split('_')[0]
        rc   = clinical_main[clinical_main['bids_id'] == sid]
        if rc.empty:
            continue
        grp  = rc.iloc[0]['Group']
        epo  = mne.read_epochs(str(fp), preload=True, verbose='ERROR')
        mask = (epo.times >= tmin) & (epo.times < tmax)
        data = epo.get_data()[:, 0, mask]
        vals = [safe_ais(t, lag=lag, n_bins=bins) for t in data]
        valid = [v for v in vals if np.isfinite(v)]
        if valid:
            rob_vals.append({'group': grp, 'ais': np.mean(valid)})
    df_r = pd.DataFrame(rob_vals)
    if len(df_r) < 4:
        continue
    a = df_r[df_r['group'] == 'CTL']['ais'].dropna()
    b = df_r[df_r['group'] == 'MDD']['ais'].dropna()
    if len(a) < 2 or len(b) < 2:
        continue
    _, p_r = mannwhitneyu(a, b, alternative='two-sided')
    d_r    = cohens_d(a.values, b.values)
    print(f"  {label}: CTL={a.mean():.4f}, MDD={b.mean():.4f}, "
          f"d={d_r:.3f}, p={p_r:.4f}")

# Visualization
sns.set_theme(style='whitegrid', font_scale=1.1)
palette = {'CTL': '#2196F3', 'MDD': '#F44336'}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    'AIS_pre Replication — Cavanagh MEG (PST, N=90)\n'
    f'FCz equiv: {FCZ_CHAN} (x=−5.3mm, midline)',
    fontsize=11, fontweight='bold'
)

# Panel A: main group comparison
ax = axes[0]
sns.boxplot(data=df, x='group', y='mean_AIS_pre',
            palette=palette, width=0.5, ax=ax, order=['CTL','MDD'])
sns.stripplot(data=df, x='group', y='mean_AIS_pre',
              palette=palette, alpha=0.5, size=4, ax=ax,
              order=['CTL','MDD'], dodge=False)
if np.isfinite(d):
    ax.set_title(f'A. AIS_pre CTL vs MDD\nd={d:.3f}, p={p:.4f}')
ax.set_ylabel('AIS_pre [bits]')
ax.set_xlabel('')

# Panel B: by feedback condition
ax = axes[1]
df_melt = df[['group', 'AIS_pre_win', 'AIS_pre_loss']].rename(
    columns={'AIS_pre_win': 'win', 'AIS_pre_loss': 'loss'}
).melt(id_vars='group', var_name='condition', value_name='AIS_pre')
sns.boxplot(data=df_melt, x='condition', y='AIS_pre',
            hue='group', palette=palette, width=0.5,
            ax=ax, hue_order=['CTL','MDD'])
ax.set_title('B. AIS_pre by Condition\n(win=reward, loss=loss)')
ax.set_ylabel('AIS_pre [bits]')
ax.set_xlabel('')
ax.get_legend().set_title('')

# Panel C: TEPS_anticipatory vs AIS_pre
ax = axes[2]
valid_teps = df[['TEPS_ant', 'mean_AIS_pre', 'group']].dropna()
for grp, color in [('CTL', '#2196F3'), ('MDD', '#F44336')]:
    sub = valid_teps[valid_teps['group'] == grp]
    ax.scatter(sub['TEPS_ant'], sub['mean_AIS_pre'],
               c=color, alpha=0.6, s=50, label=grp)
if len(valid_teps) > 5:
    z  = np.polyfit(valid_teps['TEPS_ant'], valid_teps['mean_AIS_pre'], 1)
    xl = np.linspace(valid_teps['TEPS_ant'].min(),
                     valid_teps['TEPS_ant'].max(), 50)
    r_teps, p_teps = perm_r(valid_teps['mean_AIS_pre'].values,
                             valid_teps['TEPS_ant'].values)
    ax.plot(xl, np.polyval(z, xl), 'k-', alpha=0.6, lw=1.5,
            label=f'r={r_teps:+.2f}, p={p_teps:.3f}')
ax.set_xlabel('TEPS_anticipatory (higher=more pleasure)')
ax.set_ylabel('AIS_pre [bits]')
ax.set_title('C. TEPS_anticipatory vs AIS_pre\n(key anhedonia link)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / 'meg_ais_pre_replication.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUT_DIR / 'meg_ais_pre_replication.png'}")

# Final verdict
print("\n" + "=" * 60)
print("MEG CAVANAGH REPLICATION — FINAL VERDICT")
print("=" * 60)
print(f"Primary (EEG, N=87CTL/23MDD):   d=+0.874, p=0.0003")
if np.isfinite(d):
    print(f"MEG replication (N={len(ctl)}CTL/{len(mdd)}MDD): d={d:+.3f}, p={p:.4f}")
    print(f"Direction CTL>MDD: {'✓' if ctl.mean() > mdd.mean() else '✗'}")
    if d >= 0.5 and ctl.mean() > mdd.mean():
        verdict = "STRONG REPLICATION"
    elif d >= 0.3 and ctl.mean() > mdd.mean():
        verdict = "PARTIAL REPLICATION"
    elif d > 0.0 and ctl.mean() > mdd.mean():
        verdict = "DIRECTIONAL REPLICATION"
    else:
        verdict = "FAILURE TO REPLICATE"
    print(f"\nVERDICT: {verdict}")
print("=" * 60)

df.to_csv(OUT_DIR / 'meg_ais_pre_results.csv', index=False)
print(f"\nResults saved: {OUT_DIR / 'meg_ais_pre_results.csv'}")
print("Next: run 14b_modma_ais_pre_replication.py")
