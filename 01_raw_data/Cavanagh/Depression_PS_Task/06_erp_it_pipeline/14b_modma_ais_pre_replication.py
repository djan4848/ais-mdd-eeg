"""
Script 14b: AIS_pre cross-paradigm generalization — MODMA EEG
Emotional dot-probe task, N=53 (HC=29, MDD=24).
Channel: E6 (4.7mm from FCz, EGI HydroCel-128).
Window: -200ms to 0ms relative to probe dot onset (DURING face viewing).
Hypothesis: CTL shows higher temporal neural coherence during face processing.
NOTE: This tests a related but distinct construct from anticipatory AIS_pre.
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

BASE_MODMA = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/MODMA")
EPO_DIR    = BASE_MODMA / "DDS-MODMA/derivatives/epochs"
OUT_DIR    = BASE_MODMA / "DDS-MODMA/derivatives"
CLIN_FILE  = BASE_MODMA / ("EEG_128channels_ERP_lanzhou_2015/"
                            "subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx")

FCZ_CHAN = 'E6'   # confirmed 4.7mm from FCz in GSN-HydroCel-128

# Import AIS
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
rng0 = np.random.default_rng(42)
ar1 = np.zeros(200)
for i in range(1, 200):
    ar1[i] = 0.9 * ar1[i-1] + 0.1 * rng0.standard_normal()
wn = rng0.standard_normal(200)
assert safe_ais(ar1) > safe_ais(wn), "AIS sanity check FAILED"
print("AIS sanity check OK")

# ── Clinical ──────────────────────────────────────────────────────────────
clin = pd.read_excel(CLIN_FILE)
print(f"Clinical loaded: {len(clin)} subjects")
print(f"Groups: {clin['type'].value_counts().to_dict()}")
print(f"PHQ-9 HC  : {clin[clin['type']=='HC']['PHQ-9'].mean():.1f} ± "
      f"{clin[clin['type']=='HC']['PHQ-9'].std():.1f}")
print(f"PHQ-9 MDD : {clin[clin['type']=='MDD']['PHQ-9'].mean():.1f} ± "
      f"{clin[clin['type']=='MDD']['PHQ-9'].std():.1f}")

# ── Quick data check on one subject ──────────────────────────────────────
sample_file = next(EPO_DIR.glob('*.fif'))
epo_s = mne.read_epochs(str(sample_file), preload=False, verbose='ERROR')
print(f"\nSample epoch: {sample_file.name}")
print(f"  sfreq={epo_s.info['sfreq']} Hz, "
      f"tmin={epo_s.tmin:.3f}s, tmax={epo_s.tmax:.3f}s")
print(f"  Events: {epo_s.event_id}")
assert FCZ_CHAN in epo_s.ch_names, f"{FCZ_CHAN} not in ch_names"
print(f"  {FCZ_CHAN} confirmed present ✓")
pre_mask_check = (epo_s.times >= -0.200) & (epo_s.times < 0.000)
print(f"  Pre-window samples: {pre_mask_check.sum()} "
      f"(= {pre_mask_check.sum()/epo_s.info['sfreq']*1000:.0f}ms at "
      f"{epo_s.info['sfreq']:.0f}Hz)")

# ── AIS_pre computation ───────────────────────────────────────────────────
print(f"\n=== AIS_PRE COMPUTATION ===")
PRE_TMIN, PRE_TMAX = -0.200, 0.000
MIN_TRIALS = 10

# Emotion conditions
COND_MAP = {'happy': 'hdot', 'fearful': 'fdot', 'sad': 'sdot'}

records = []
epo_files = sorted(EPO_DIR.glob('*.fif'))
print(f"Processing {len(epo_files)} subjects...")

for fpath in epo_files:
    # Parse subject ID: '02010002-epo.fif' → '02010002' → clinical 2010002
    stem   = fpath.stem                          # '02010002-epo' or '02010008_-epo'
    id_str = stem.split('-')[0].rstrip('_')      # '02010002' or '02010008'
    try:
        subj_int = int(id_str)                   # 2010002
    except ValueError:
        print(f"  Cannot parse ID from {fpath.name}, skip")
        continue

    # Match clinical (clinical 'subject id' is integer, e.g. 2010002)
    row_c = clin[clin['subject id'] == subj_int]
    if row_c.empty:
        print(f"  {id_str}: no clinical match, skip")
        continue
    rc    = row_c.iloc[0]
    group = rc['type']              # 'HC' or 'MDD'
    phq9  = rc.get('PHQ-9', np.nan)
    gad7  = rc.get('GAD-7', np.nan)

    epo      = mne.read_epochs(str(fpath), preload=True, verbose='ERROR')
    times    = epo.times
    pre_mask = (times >= PRE_TMIN) & (times < PRE_TMAX)
    ch_idx   = epo.ch_names.index(FCZ_CHAN)

    cond_ais = {}

    # All trials combined
    all_data = epo.get_data()[:, ch_idx, pre_mask]
    trial_ais_all = [safe_ais(t, lag=1, n_bins=4) for t in all_data]
    valid_all = [v for v in trial_ais_all if np.isfinite(v)]
    cond_ais['combined'] = np.mean(valid_all) if len(valid_all) >= MIN_TRIALS else np.nan

    # Per-emotion condition
    for emo_name, event_key in COND_MAP.items():
        try:
            cond_data = epo[event_key].get_data()[:, ch_idx, pre_mask]
        except KeyError:
            continue
        vals = [safe_ais(t, lag=1, n_bins=4) for t in cond_data]
        valid = [v for v in vals if np.isfinite(v)]
        cond_ais[emo_name] = np.mean(valid) if len(valid) >= 5 else np.nan

    rec = {
        'subject_id':    id_str,
        'group':         group,
        'mean_AIS_pre':  cond_ais.get('combined', np.nan),
        'AIS_pre_happy': cond_ais.get('happy',    np.nan),
        'AIS_pre_fear':  cond_ais.get('fearful',  np.nan),
        'AIS_pre_sad':   cond_ais.get('sad',       np.nan),
        'n_trials':      len(epo),
        'PHQ9':          phq9,
        'GAD7':          gad7,
    }
    records.append(rec)
    print(f"  {id_str} ({group}): AIS_pre={rec['mean_AIS_pre']:.4f}, N={len(epo)}")

df = pd.DataFrame(records)
print(f"\nTotal: {len(df)}  "
      f"(HC={(df['group']=='HC').sum()}, MDD={(df['group']=='MDD').sum()})")
if df['mean_AIS_pre'].isna().any():
    print(f"NaN rate: {df['mean_AIS_pre'].isna().mean():.1%}")

# ── Statistics ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODMA EEG — AIS_PRE CROSS-PARADIGM RESULTS")
print("=" * 60)

hc  = df[df['group'] == 'HC']['mean_AIS_pre'].dropna()
mdd = df[df['group'] == 'MDD']['mean_AIS_pre'].dropna()

U, p = mannwhitneyu(hc, mdd, alternative='two-sided')
d    = cohens_d(hc.values, mdd.values)

print(f"\nHC  AIS_pre: {hc.mean():.4f} ± {hc.std():.4f}  (N={len(hc)})")
print(f"MDD AIS_pre: {mdd.mean():.4f} ± {mdd.std():.4f}  (N={len(mdd)})")
print(f"Cohen's d   = {d:.3f}")
print(f"p           = {p:.4f}")
print(f"Direction HC>MDD: {'✓' if hc.mean() > mdd.mean() else '✗'}")

# By emotion condition
print("\n=== BY EMOTION CONDITION ===")
for emo, col in [('happy', 'AIS_pre_happy'),
                 ('fearful', 'AIS_pre_fear'),
                 ('sad',  'AIS_pre_sad')]:
    a = df[df['group'] == 'HC'][col].dropna()
    b = df[df['group'] == 'MDD'][col].dropna()
    if len(a) < 3 or len(b) < 3:
        continue
    _, p_c = mannwhitneyu(a, b, alternative='two-sided')
    d_c    = cohens_d(a.values, b.values)
    print(f"  {emo:8s}: HC={a.mean():.4f}, MDD={b.mean():.4f}, "
          f"d={d_c:.3f}, p={p_c:.4f}")
print("Prediction: happy faces show largest effect "
      "(reward-salient stimulus)")

# PHQ-9 correlation
print("\n=== PHQ-9 CORRELATION ===")
valid_phq = df[['mean_AIS_pre', 'PHQ9']].dropna()
if len(valid_phq) >= 10:
    r_phq, p_phq = perm_r(valid_phq['mean_AIS_pre'].values,
                           valid_phq['PHQ9'].values)
    rs_phq = spearmanr(valid_phq['mean_AIS_pre'], valid_phq['PHQ9'])[0]
    match = '✓' if r_phq < 0 else '✗'
    print(f"  AIS_pre vs PHQ-9: r={r_phq:+.3f}, p_perm={p_phq:.4f}, "
          f"rho={rs_phq:+.3f}  {match}  (N={len(valid_phq)})")
    print(f"  Expected: negative r (more depressed = less AIS_pre)")

# Robustness
print("\n=== ROBUSTNESS ===")
for label, lag, bins in [
    ('lag=1 bins=4 (baseline)', 1, 4),
    ('lag=1 bins=6',            1, 6),
    ('lag=1 bins=8',            1, 8),
]:
    rob_vals = []
    for fp in epo_files:
        stem2  = fp.stem.split('-')[0].rstrip('_')
        try:
            sint = int(stem2)
        except ValueError:
            continue
        row2 = clin[clin['subject id'] == sint]
        if row2.empty:
            continue
        grp2 = row2.iloc[0]['type']
        ep2  = mne.read_epochs(str(fp), preload=True, verbose='ERROR')
        mask = (ep2.times >= PRE_TMIN) & (ep2.times < PRE_TMAX)
        cidx = ep2.ch_names.index(FCZ_CHAN)
        data = ep2.get_data()[:, cidx, mask]
        vals = [safe_ais(t, lag=lag, n_bins=bins) for t in data]
        valid = [v for v in vals if np.isfinite(v)]
        if valid:
            rob_vals.append({'group': grp2, 'ais': np.mean(valid)})
    df_r = pd.DataFrame(rob_vals)
    if len(df_r) < 4:
        continue
    a = df_r[df_r['group'] == 'HC']['ais'].dropna()
    b = df_r[df_r['group'] == 'MDD']['ais'].dropna()
    if len(a) < 2 or len(b) < 2:
        continue
    _, p_r = mannwhitneyu(a, b, alternative='two-sided')
    d_r    = cohens_d(a.values, b.values)
    print(f"  {label}: d={d_r:.3f}, p={p_r:.4f}")

# Visualization
sns.set_theme(style='whitegrid', font_scale=1.1)
palette = {'HC': '#2196F3', 'MDD': '#F44336'}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    'AIS_pre Cross-Paradigm Test — MODMA EEG (Dot-Probe, N=53)\n'
    'Channel: E6 (FCz equiv.), Window: −200ms to 0ms before probe onset',
    fontsize=10, fontweight='bold'
)

# Panel A: main comparison
ax = axes[0]
sns.boxplot(data=df, x='group', y='mean_AIS_pre',
            palette=palette, width=0.5, ax=ax, order=['HC','MDD'])
sns.stripplot(data=df, x='group', y='mean_AIS_pre',
              palette=palette, alpha=0.5, size=4, ax=ax,
              order=['HC','MDD'])
ax.set_title(f'A. AIS_pre HC vs MDD\nd={d:.3f}, p={p:.4f}')
ax.set_ylabel('AIS_pre [bits]')
ax.set_xlabel('')

# Panel B: by emotion condition
ax = axes[1]
df_emo = df[['group','AIS_pre_happy','AIS_pre_fear','AIS_pre_sad']].rename(
    columns={'AIS_pre_happy':'happy','AIS_pre_fear':'fearful','AIS_pre_sad':'sad'}
).melt(id_vars='group', var_name='emotion', value_name='AIS_pre')
sns.boxplot(data=df_emo, x='emotion', y='AIS_pre',
            hue='group', palette=palette, width=0.5, ax=ax,
            hue_order=['HC','MDD'])
ax.set_title('B. AIS_pre by Emotion\n(during face processing)')
ax.set_ylabel('AIS_pre [bits]')
ax.set_xlabel('')
ax.get_legend().set_title('')

# Panel C: PHQ-9 vs AIS_pre
ax = axes[2]
for grp, color in [('HC', '#2196F3'), ('MDD', '#F44336')]:
    sub = df[df['group'] == grp].dropna(subset=['PHQ9','mean_AIS_pre'])
    ax.scatter(sub['PHQ9'], sub['mean_AIS_pre'],
               c=color, alpha=0.6, s=50, label=grp)
if len(valid_phq) > 5:
    z  = np.polyfit(valid_phq['PHQ9'], valid_phq['mean_AIS_pre'], 1)
    xl = np.linspace(valid_phq['PHQ9'].min(), valid_phq['PHQ9'].max(), 50)
    ax.plot(xl, np.polyval(z, xl), 'k-', alpha=0.6, lw=1.5,
            label=f'r={r_phq:+.2f}, p={p_phq:.3f}')
ax.set_xlabel('PHQ-9 score')
ax.set_ylabel('AIS_pre [bits]')
ax.set_title('C. PHQ-9 vs AIS_pre')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / 'modma_ais_pre_replication.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUT_DIR / 'modma_ais_pre_replication.png'}")

# Final report
print("\n" + "=" * 60)
print("MODMA CROSS-PARADIGM GENERALIZATION — FINAL VERDICT")
print("=" * 60)
print(f"Cavanagh EEG (PS Task):   d=+0.874, p=0.0003")
print(f"MODMA EEG (dot-probe):    d={d:+.3f}, p={p:.4f}")
print(f"Direction HC>MDD:         {'✓' if hc.mean() > mdd.mean() else '✗'}")
if d >= 0.5 and hc.mean() > mdd.mean():
    verdict = "STRONG GENERALIZATION"
elif d >= 0.3 and hc.mean() > mdd.mean():
    verdict = "PARTIAL GENERALIZATION"
elif d > 0.0 and hc.mean() > mdd.mean():
    verdict = "DIRECTIONAL GENERALIZATION"
else:
    verdict = "DOES NOT GENERALIZE"
print(f"VERDICT: {verdict}")
print("="*60)
print("\nInterpretation note: MODMA window is DURING face processing,")
print("not blank anticipation. Effect (if present) reflects temporal")
print("neural coherence during emotional stimulus processing, not")
print("anticipatory preparation before feedback.")

df.to_csv(OUT_DIR / 'modma_ais_pre_results.csv', index=False)
print(f"\nResults saved: {OUT_DIR / 'modma_ais_pre_results.csv'}")
print("Both replication scripts complete.")
