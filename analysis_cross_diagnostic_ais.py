"""
AIS_rest cross-diagnostic comparison — TDBRAIN
MDD (already done) vs ADHD, SMC, OCD vs CTL (already done)
"""
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, mne
mne.set_log_level('ERROR')
from pathlib import Path
from scipy.stats import mannwhitneyu, kruskal, pearsonr
from numpy.linalg import lstsq

BASE    = Path("/media/neuraldyn/PortableSSD/DEPRESSION")
TDBRAIN = Path("/media/neuraldyn/PortableSSD/TDBRAIN_derivatives")
ASSETS  = BASE / "06_manuscript_assets"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── helpers ────────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1))
                 / (len(a)+len(b)-2) + 1e-10)
    return float((np.mean(a) - np.mean(b)) / sp)

def safe_ais(x, lag=1, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < 2*lag + 10: return np.nan
    if np.std(x) < 1e-12:   return np.nan
    try:
        edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins+1)))
        if len(edges) < 3: return np.nan
        bins = np.digitize(x, edges[1:-1])
        flat = bins[lag:] * n_bins + bins[:-lag]
        joint = np.bincount(flat, minlength=n_bins*n_bins).reshape(n_bins, n_bins).astype(float)
        joint /= joint.sum() + 1e-10
        px_t   = joint.sum(axis=1, keepdims=True)
        px_lag = joint.sum(axis=0, keepdims=True)
        mask   = (joint > 0) & (px_t > 0) & (px_lag > 0)
        mi = np.sum(joint[mask] * np.log2(joint[mask] / (px_t * px_lag)[mask]))
        return float(mi) if np.isfinite(mi) else np.nan
    except Exception:
        return np.nan

# ── STEP 0 — Load existing CTL + MDD ──────────────────────────────────────────
df_existing = pd.read_csv(TDBRAIN / "derivatives/tdbrain_ais_mdd_ctl.csv")
# columns: subject_id, group (CTL/MDD), ais_rest, remitter
AIS_COL = 'ais_rest'
already_done = set(df_existing['subject_id'].astype(str))
print("=== EXISTING DATA ===")
print(df_existing['group'].value_counts().to_string())

# ── STEP 1 — Participants file ─────────────────────────────────────────────────
parts = pd.read_csv(TDBRAIN / "TDBRAIN_participants_V2.tsv", sep='\t')
# columns: participants_ID, indication, age (comma-decimal), gender, sessID, ...
# Keep one row per subject (take sessID=1 when multiple sessions exist)
parts1 = parts[parts['sessID'] == 1].copy()
# Age is stored as European decimal string e.g. "9,00" → convert
parts1['age_num'] = pd.to_numeric(
    parts1['age'].astype(str).str.replace(',', '.', regex=False),
    errors='coerce')

# Build subject → (indication, age, gender) lookup
sub_info = {}
for _, row in parts1.iterrows():
    sid = str(row['participants_ID'])
    sub_info[sid] = {
        'indication': str(row['indication']),
        'age':        float(row['age_num']) if pd.notna(row['age_num']) else np.nan,
        'gender':     float(row['gender'])  if pd.notna(row['gender'])  else np.nan,
    }

print(f"\nParticipants (sessID=1): {len(parts1)}")
print("indication distribution:")
print(parts1['indication'].value_counts().to_string())

# ── STEP 2 — Compute AIS_rest for ADHD, SMC, OCD ─────────────────────────────
# Use ses-1 files only; CTL and MDD already in existing CSV
restEC_files = sorted(TDBRAIN.glob("sub-*/ses-1/eeg/*task-restEC_eeg.vhdr"))
print(f"\nrestEC ses-1 files: {len(restEC_files)}")

WINDOW_SEC  = 2.0
MIN_WINDOWS = 10
TARGET_GROUPS = {'ADHD', 'SMC', 'OCD'}

new_records = []
n_proc = 0
n_skip = 0
counts = {g: 0 for g in TARGET_GROUPS}

for fpath in restEC_files:
    sub_id = fpath.name.split('_')[0]   # "sub-XXXXXXXX"

    if sub_id in already_done:
        n_skip += 1
        continue

    info = sub_info.get(sub_id)
    if info is None:
        continue

    ind = info['indication'].upper()
    if 'ADHD' in ind or 'ATTENTION' in ind:
        group = 'ADHD'
    elif 'SMC' in ind or 'MEMORY' in ind or 'SUBJECTIVE' in ind:
        group = 'SMC'
    elif 'OCD' in ind or 'OBSESSIVE' in ind or 'COMPULSIVE' in ind:
        group = 'OCD'
    else:
        continue

    try:
        raw = mne.io.read_raw_brainvision(fpath, preload=True, verbose='ERROR')
        raw.filter(1., 40., verbose='ERROR')
        if 'FCz' not in raw.ch_names: continue
        sig   = raw.get_data(picks=['FCz'])[0]
        sfreq = raw.info['sfreq']
        if np.std(sig) < 1e-10: continue

        win_len = int(WINDOW_SEC * sfreq)
        n_win   = len(sig) // win_len
        if n_win < MIN_WINDOWS: continue

        ais_vals = [safe_ais(sig[i*win_len:(i+1)*win_len]) for i in range(n_win)]
        valid    = [v for v in ais_vals if np.isfinite(v)]
        if len(valid) < MIN_WINDOWS: continue

        new_records.append({
            'subject_id': sub_id,
            'group':      group,
            AIS_COL:      float(np.mean(valid)),
            'n_windows':  len(valid),
            'age':        info['age'],
            'gender':     info['gender'],
        })
        counts[group] += 1
        n_proc += 1
        if n_proc % 25 == 0:
            print(f"  [{n_proc}] " + "  ".join(f"{g}={counts[g]}" for g in TARGET_GROUPS))

    except Exception:
        continue

print(f"\nNew subjects processed: {n_proc}  "
      + "  ".join(f"{g}={counts[g]}" for g in TARGET_GROUPS))
print(f"Skipped (already done): {n_skip}")

# Add age/gender to existing subjects from participants file
df_existing_aug = df_existing[['subject_id','group',AIS_COL]].copy()
df_existing_aug['age']    = df_existing_aug['subject_id'].map(
    {k: v['age']    for k,v in sub_info.items()})
df_existing_aug['gender'] = df_existing_aug['subject_id'].map(
    {k: v['gender'] for k,v in sub_info.items()})

df_new = pd.DataFrame(new_records)
df_all = pd.concat([df_existing_aug, df_new], ignore_index=True)

print(f"\nCombined dataset ({len(df_all)} subjects):")
print(df_all['group'].value_counts().to_string())

df_all.to_csv(TDBRAIN / "derivatives/tdbrain_ais_all_disorders.csv", index=False)
print("Saved: tdbrain_ais_all_disorders.csv")

# ── STEP 3 — Cross-diagnostic comparison vs CTL ────────────────────────────────
print("\n" + "="*60)
print("AIS_rest CROSS-DIAGNOSTIC COMPARISON vs CTL")
print("="*60)

ctl = df_all[df_all['group']=='CTL'][AIS_COL].dropna()
print(f"\nCTL: {ctl.mean():.4f} ± {ctl.std():.4f}  (N={len(ctl)})")

GROUPS = ['MDD','ADHD','SMC','OCD']
results = []
for grp in GROUPS:
    data = df_all[df_all['group']==grp][AIS_COL].dropna()
    if len(data) < 5: continue
    U, p = mannwhitneyu(ctl, data, alternative='two-sided')
    d    = cohens_d(ctl.values, data.values)
    auc  = max(U / (len(ctl)*len(data)), 1 - U/(len(ctl)*len(data)))
    results.append({'group': grp, 'N': len(data), 'mean': round(data.mean(),4),
                    'std': round(data.std(),4), 'd_vs_CTL': round(d,3),
                    'p_vs_CTL': round(p,5), 'AUC': round(auc,3),
                    'direction': 'CTL>' if ctl.mean()>data.mean() else '<CTL'})
    print(f"\n{grp} (N={len(data)}): {data.mean():.4f} ± {data.std():.4f}")
    print(f"  vs CTL: d={d:+.3f}, p={p:.4f}, AUC={auc:.3f}")
    print(f"  {'CTL > ' + grp if ctl.mean() > data.mean() else grp + ' > CTL'}")

df_res = pd.DataFrame(results)

# FDR correction
try:
    from statsmodels.stats.multitest import multipletests
    _, p_fdr, _, _ = multipletests(df_res['p_vs_CTL'].values, method='fdr_bh')
    df_res['p_fdr'] = np.round(p_fdr, 5)
except ImportError:
    df_res['p_fdr'] = df_res['p_vs_CTL']

print("\nAfter FDR correction:")
for _, row in df_res.iterrows():
    sig = '✅' if row['p_fdr'] < 0.05 else '⚠️ '
    print(f"  {sig} {row['group']}: d={row['d_vs_CTL']:+.3f}, "
          f"p_fdr={row['p_fdr']:.4f}, AUC={row['AUC']:.3f}")

# ── STEP 4 — Pairwise (MDD vs each disorder) ───────────────────────────────────
print("\n" + "="*60)
print("PAIRWISE: MDD vs each other disorder")
print("="*60)
mdd_data = df_all[df_all['group']=='MDD'][AIS_COL].dropna()
for grp in ['ADHD','SMC','OCD']:
    data = df_all[df_all['group']==grp][AIS_COL].dropna()
    if len(data) < 5: continue
    U, p = mannwhitneyu(mdd_data, data, alternative='two-sided')
    d    = cohens_d(mdd_data.values, data.values)
    print(f"  MDD vs {grp}: d={d:+.3f}, p={p:.4f}  "
          f"({'MDD>' if mdd_data.mean()>data.mean() else grp+'>MDD'})")

# ── STEP 5 — Age confound ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("DEMOGRAPHIC CONFOUND CHECK")
print("="*60)
for grp in ['CTL'] + GROUPS:
    sub = df_all[df_all['group']==grp]
    if 'age' in sub.columns:
        ag = sub['age'].dropna()
        if len(ag) > 3:
            print(f"  {grp}: age={ag.mean():.1f}±{ag.std():.1f}  "
                  f"N_age={len(ag)}")

age_ais = df_all[['age', AIS_COL]].dropna()
if len(age_ais) > 20:
    r_age, p_age = pearsonr(age_ais['age'], age_ais[AIS_COL])
    print(f"\nr(age, AIS_rest) = {r_age:+.3f}, p={p_age:.4f}  (N={len(age_ais)})")
    if abs(r_age) > 0.3:
        print("⚠️  Age correlates with AIS_rest — computing age-adjusted d values:")
        for grp in GROUPS:
            data_grp = df_all[df_all['group'].isin(['CTL', grp])][
                ['group', AIS_COL, 'age']].dropna()
            if len(data_grp) < 10: continue
            X_aug = np.column_stack([np.ones(len(data_grp)), data_grp['age'].values])
            coef, _, _, _ = lstsq(X_aug, data_grp[AIS_COL].values, rcond=None)
            resid = data_grp[AIS_COL].values - X_aug @ coef
            ctl_r  = resid[data_grp['group'].values == 'CTL']
            diag_r = resid[data_grp['group'].values == grp]
            if len(ctl_r) < 3 or len(diag_r) < 3: continue
            _, p_adj = mannwhitneyu(ctl_r, diag_r, alternative='two-sided')
            d_adj = cohens_d(ctl_r, diag_r)
            print(f"  {grp}: age-adjusted d={d_adj:+.3f}, p={p_adj:.4f}")
    else:
        print(f"✅ Age does not confound AIS_rest")

# ── STEP 6 — Kruskal-Wallis omnibus ──────────────────────────────────────────
print("\n" + "="*60)
print("KRUSKAL-WALLIS OMNIBUS TEST (all groups)")
print("="*60)
all_gdata = [df_all[df_all['group']==g][AIS_COL].dropna().values
             for g in ['CTL']+GROUPS if len(df_all[df_all['group']==g][AIS_COL].dropna()) >= 5]
H, p_kw = kruskal(*all_gdata)
print(f"H = {H:.2f}, p = {p_kw:.6f}")
print(f"{'✅ Significant group differences' if p_kw<0.05 else '⚠️  No global difference'}")

# ── STEP 7 — Binary AUC classifier ────────────────────────────────────────────
print("\n" + "="*60)
print("SINGLE-FEATURE AUC: AIS_rest (each disorder vs CTL)")
print("="*60)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    sc = StandardScaler()
    for grp in GROUPS:
        df_bin = df_all[df_all['group'].isin(['CTL', grp])][[AIS_COL,'group']].dropna()
        if len(df_bin) < 20: continue
        X = sc.fit_transform(df_bin[AIS_COL].values.reshape(-1,1))
        y = (df_bin['group']==grp).astype(int).values
        auc_cv = cross_val_score(LogisticRegression(random_state=42), X, y,
                                  cv=5, scoring='roc_auc').mean()
        print(f"  {grp} vs CTL:  CV-AUC = {auc_cv:.3f}")

    # Multi-class
    grps_avail = ['CTL'] + [g for g in GROUPS if len(df_all[df_all['group']==g]) >= 5]
    df_mc = df_all[df_all['group'].isin(grps_avail)][[AIS_COL,'group']].dropna()
    if len(df_mc) > 50:
        from sklearn.multiclass import OneVsRestClassifier
        X_mc = sc.fit_transform(df_mc[AIS_COL].values.reshape(-1,1))
        y_mc = pd.Categorical(df_mc['group']).codes
        auc_mc = cross_val_score(
            OneVsRestClassifier(LogisticRegression(random_state=42)),
            X_mc, y_mc, cv=5, scoring='roc_auc_ovr').mean()
        print(f"\nMulti-class AUC (OvR, {len(grps_avail)} classes): {auc_mc:.3f}")
        print(f"  {'✅ Discriminative' if auc_mc > 0.65 else '⚠️  Limited multi-class utility'}")
except ImportError:
    print("sklearn not available — skipping classifier evaluation")

# ── STEP 8 — Figure ───────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

PALETTE = {'CTL':'#2166AC','MDD':'#D6604D','ADHD':'#4DAF4A',
           'SMC':'#984EA3','OCD':'#FF7F00'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')
rng = np.random.default_rng(42)

ORDER = [g for g in ['CTL','MDD','ADHD','SMC','OCD'] if g in df_all['group'].unique()]

# ── Panel A: Boxplot all groups
ax = axes[0]
plot_data   = [df_all[df_all['group']==g][AIS_COL].dropna().values for g in ORDER]
plot_labels = [f'{g}\n(N={len(d)})' for g, d in zip(ORDER, plot_data)]
plot_cols   = [PALETTE[g] for g in ORDER]

bp = ax.boxplot(plot_data, patch_artist=True,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                flierprops=dict(marker='o', markersize=3, alpha=0.3))
for patch, c in zip(bp['boxes'], plot_cols):
    patch.set_facecolor(c); patch.set_alpha(0.75)
for i, (d, c) in enumerate(zip(plot_data, plot_cols)):
    ax.scatter(rng.normal(i+1, 0.08, len(d)), d, color=c, alpha=0.25, s=6, zorder=3)

ctl_mean = ctl.mean()
ax.axhline(ctl_mean, color=PALETTE['CTL'], linestyle='--', linewidth=1, alpha=0.4)
ax.set_xticks(range(1, len(ORDER)+1))
ax.set_xticklabels(plot_labels, fontsize=8)
ax.set_ylabel('AIS$_\\mathrm{rest}$ (bits)')
ax.set_title('A.  AIS_rest by diagnosis', fontweight='bold', loc='left')

# ── Panel B: Effect sizes d vs CTL
ax = axes[1]
grps_r = df_res['group'].tolist()
d_vals = df_res['d_vs_CTL'].tolist()
p_fdr_v= df_res['p_fdr'].tolist()
bar_cols = [PALETTE.get(g,'#888') for g in grps_r]
bars = ax.bar(range(len(grps_r)), d_vals, color=bar_cols, width=0.6, edgecolor='white')
for bar, p_v in zip(bars, p_fdr_v):
    bar.set_alpha(0.85 if p_v < 0.05 else 0.30)

ax.axhline(0,   color='black', linewidth=0.8)
ax.axhline(0.5, color='gray',  linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(-0.5,color='gray',  linestyle='--', linewidth=0.8, alpha=0.5)
for i, (d_v, p_v) in enumerate(zip(d_vals, p_fdr_v)):
    stars = '***' if p_v<0.001 else '**' if p_v<0.01 else '*' if p_v<0.05 else 'n.s.'
    ax.text(i, d_v + (0.05 if d_v>=0 else -0.12), stars, ha='center', fontsize=9)
ax.set_xticks(range(len(grps_r))); ax.set_xticklabels(grps_r)
ax.set_ylabel("Cohen's d  (CTL > diagnosis)")
ax.set_title("B.  Effect sizes vs CTL\n    (transparent = p_fdr>0.05)",
             fontweight='bold', loc='left')

# ── Panel C: Violin normalized to CTL mean
ax = axes[2]
for i, grp in enumerate(ORDER):
    d_g = df_all[df_all['group']==grp][AIS_COL].dropna().values
    if len(d_g) < 3: continue
    norm_d = d_g / ctl_mean
    vp = ax.violinplot(norm_d, positions=[i], widths=0.6,
                        showmedians=True, showextrema=False)
    for pc in vp['bodies']:
        pc.set_facecolor(PALETTE[grp]); pc.set_alpha(0.6)
    vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2)

ax.axhline(1.0, color=PALETTE['CTL'], linestyle='--', linewidth=1, alpha=0.5)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, fontsize=9)
ax.set_ylabel('AIS$_\\mathrm{rest}$ (normalized to CTL=1.0)')
ax.set_title('C.  Distribution by diagnosis\n    (normalized to CTL = 1.0)',
             fontweight='bold', loc='left')

fig.suptitle('AIS_rest cross-diagnostic comparison — TDBRAIN restEC (FCz)\n'
             'Active Information Storage across psychiatric disorders',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()

out_png = ASSETS / "cross_diagnostic_ais_rest.png"
fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n✅ Figure saved: {out_png}")

# ── STEP 9 — Final interpretive summary ───────────────────────────────────────
print("\n" + "="*65)
print("CROSS-DIAGNOSTIC AIS_rest — FINAL SUMMARY")
print("="*65)
print(f"\nCTL reference: {ctl.mean():.4f} bits")
print(f"\n{'Group':<8} {'N':>5} {'Mean':>8} {'d_CTL':>8} {'p_fdr':>8} {'AUC':>6}")
print("-"*48)
for _, r in df_res.iterrows():
    print(f"{r['group']:<8} {r['N']:>5} {r['mean']:>8.4f} "
          f"{r['d_vs_CTL']:>8.3f} {r['p_fdr']:>8.4f} {r['AUC']:>6.3f}")

print("\nKEY QUESTIONS:")
mdd_r  = df_res[df_res['group']=='MDD']
adhd_r = df_res[df_res['group']=='ADHD']
if len(mdd_r) and len(adhd_r):
    d_mdd  = mdd_r.iloc[0]['d_vs_CTL']
    d_adhd = adhd_r.iloc[0]['d_vs_CTL']
    print(f"\nQ1 — AIS_rest specificity to MDD?")
    if d_mdd * d_adhd < 0 and abs(d_adhd) > 0.2:
        print(f"  ✅ DISSOCIATION: MDD d={d_mdd:+.2f}, ADHD d={d_adhd:+.2f}")
        print(f"  AIS_rest differentiates MDD from ADHD — diagnostic value")
    elif abs(d_adhd) < 0.2:
        print(f"  ✅ MDD-SPECIFIC: ADHD near null (d={d_adhd:+.2f})")
    else:
        print(f"  ⚠️  BOTH affected: MDD d={d_mdd:+.2f}, ADHD d={d_adhd:+.2f}")
        print(f"  AIS_rest reflects general psychopathology, not MDD-specific")

df_res.to_csv(TDBRAIN / "derivatives/tdbrain_cross_diagnostic_results.csv", index=False)
print("\n✅ Results saved: tdbrain_cross_diagnostic_results.csv")
print("\nIMPLICATIONS:")
print("  MDD-specific: AIS_rest = depression marker → targeted use")
print("  Multi-disorder ↓: AIS_rest = mental health index → broad use")
print("  Dissociation MDD≠ADHD: AIS_rest = diagnostic classifier → BCI discriminator")
