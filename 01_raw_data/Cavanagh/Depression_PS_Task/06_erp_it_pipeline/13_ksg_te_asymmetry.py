"""
Script 13: KSG Transfer Entropy Fz → FCz (post-feedback)
+ KSG AIS validation of Shannon AIS_pre finding.
Secondary analysis to AIS_pre (d=0.806, p=0.0003).
"""
import sys
import time
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
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

DECIMATE_FACTOR = 4     # 500 → 125 Hz
TAU_KSG         = 5     # 5 samples @ 125Hz = 40ms
MIN_TRIALS      = 20
N_KSG_SAMPLES   = 3000  # max pairs fed to KSG per subject×condition

# ── Import Shannon AIS ────────────────────────────────────────────────────
INFO_PATH = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/"
                 "ds003474/code/eeg_depression_classification")
sys.path.insert(0, str(INFO_PATH))
from info_theory import compute_ais
print("compute_ais imported OK")

# ── KSG primitives (inline, not imported, so joblib workers are self-contained) ──

def _ksg_cmi_core(X, Y, Z, k=4, seed=42):
    """KSG CMI in nats then convert to bits. All arrays shape (N,d)."""
    N = len(X)
    rng = np.random.default_rng(seed)
    X = X + 1e-10 * rng.standard_normal(X.shape)
    Y = Y + 1e-10 * rng.standard_normal(Y.shape)
    Z = Z + 1e-10 * rng.standard_normal(Z.shape)
    XYZ = np.hstack((X, Y, Z))
    dist, _ = cKDTree(XYZ).query(XYZ, k=k+1, p=np.inf)
    eps = dist[:, k]
    XZ = np.hstack((X, Z)); YZ = np.hstack((Y, Z))
    radius = eps + 1e-12
    t_xz = cKDTree(XZ); t_yz = cKDTree(YZ); t_z = cKDTree(Z)
    n_xz = np.array([len(t_xz.query_ball_point(XZ[i], radius[i], p=np.inf)) for i in range(N)])
    n_yz = np.array([len(t_yz.query_ball_point(YZ[i], radius[i], p=np.inf)) for i in range(N)])
    n_z  = np.array([len(t_z.query_ball_point(Z[i],  radius[i], p=np.inf)) for i in range(N)])
    return max(0.0, (digamma(k) + np.mean(digamma(n_z))
                     - np.mean(digamma(n_xz)) - np.mean(digamma(n_yz))) / np.log(2))

def _ksg_mi_core(X, Y, k=4, seed=42):
    """KSG MI Algorithm 1 in bits."""
    N = len(X)
    rng = np.random.default_rng(seed)
    X = X + 1e-10 * rng.standard_normal(X.shape)
    Y = Y + 1e-10 * rng.standard_normal(Y.shape)
    XY = np.hstack((X, Y))
    dist, _ = cKDTree(XY).query(XY, k=k+1, p=np.inf)
    eps = dist[:, k]
    t_x = cKDTree(X); t_y = cKDTree(Y)
    radius = eps + 1e-12
    n_x = np.array([len(t_x.query_ball_point(X[i], radius[i], p=np.inf)) - 1 for i in range(N)])
    n_y = np.array([len(t_y.query_ball_point(Y[i], radius[i], p=np.inf)) - 1 for i in range(N)])
    return max(0.0, (digamma(k) - np.mean(digamma(n_x+1))
                     - np.mean(digamma(n_y+1)) + digamma(N)) / np.log(2))

def ksg_ais(x, tau=2, k=4, N_samples=2000, seed=42):
    """KSG AIS = MI(X_t; X_{t-tau}). Proper unconditioned MI."""
    x = np.asarray(x, dtype=float)
    Xp = x[tau:].reshape(-1, 1)
    Xl = x[:-tau].reshape(-1, 1)
    N  = len(Xp)
    if N < 20:
        return np.nan
    if N > N_samples:
        idx = np.random.default_rng(seed).choice(N, N_samples, replace=False)
        Xp, Xl = Xp[idx], Xl[idx]
    return _ksg_mi_core(Xp, Xl, k=k, seed=seed)

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0

# ── Clinical ──────────────────────────────────────────────────────────────
clinical      = pd.read_csv(CLINICAL)
clinical_main = clinical[
    clinical[GROUP_COL].isin([CTL_LABEL, MDD_LABEL]) & (~clinical['excluded'])
].copy()
print(f"N CTL: {(clinical_main[GROUP_COL]==CTL_LABEL).sum()}, "
      f"N MDD: {(clinical_main[GROUP_COL]==MDD_LABEL).sum()}")

epoch_files = sorted(EPO_DIR.glob('*_task-ps_epo.fif'))

# ── SECTION 1: KSG vs Shannon AIS_pre validation (N=20) ──────────────────
rng_sel = np.random.default_rng(42)
ctl_ids = clinical_main[clinical_main[GROUP_COL]==CTL_LABEL]['subject_id'].values
mdd_ids = clinical_main[clinical_main[GROUP_COL]==MDD_LABEL]['subject_id'].values
ctl_samp = rng_sel.choice(ctl_ids, size=min(10, len(ctl_ids)), replace=False)
mdd_samp = rng_sel.choice(mdd_ids, size=min(10, len(mdd_ids)), replace=False)
val_ids  = set(np.concatenate([ctl_samp, mdd_samp]))

print(f"\n=== VALIDATION: KSG vs Shannon AIS_pre (N={len(val_ids)} subjects) ===")
val_rows = []
for fpath in epoch_files:
    try:
        sub_id = int(fpath.stem.split('-')[1].split('_')[0])
    except (IndexError, ValueError):
        continue
    if sub_id not in val_ids:
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

    ch_idx   = epo.ch_names.index('FCz')
    pre_mask = (epo.times >= -0.5) & (epo.times < 0.0)

    # Concatenate all pre-window trials (Reward + Loss)
    segs = []
    for ckey in ('Reward', 'Loss'):
        try:
            segs.append(epo[ckey].get_data()[:, ch_idx, :][:, pre_mask])
        except KeyError:
            pass
    if not segs:
        continue
    pre_sig = np.concatenate(segs).flatten()

    # Shannon AIS (lag=2, n_bins=4) — same as Script 12c
    try:
        shan = compute_ais(pre_sig, lag=2, n_bins=4)
        shan = shan if np.isfinite(shan) else np.nan
    except Exception:
        shan = np.nan

    # KSG AIS (unconditioned MI, lag=2)
    ksg = ksg_ais(pre_sig, tau=2, k=4, N_samples=2000, seed=sub_id)

    val_rows.append({'subject_id': sub_id, 'group': group,
                     'shannon_ais': shan, 'ksg_ais': ksg})
    print(f"  sub-{sub_id} ({group}): Shannon={shan:.3f}, KSG={ksg:.3f}")

df_val = pd.DataFrame(val_rows).dropna(subset=['shannon_ais', 'ksg_ais'])
if len(df_val) >= 4:
    r_val, p_val = pearsonr(df_val['shannon_ais'], df_val['ksg_ais'])
    rho_val      = spearmanr(df_val['shannon_ais'], df_val['ksg_ais'])[0]
    print(f"\nKSG vs Shannon AIS_pre: r={r_val:.3f}, rho={rho_val:.3f}, "
          f"p={p_val:.4f}  (N={len(df_val)})")
    if r_val > 0.7:
        print("→ Shannon estimator VALIDATED by KSG")
    elif r_val > 0.5:
        print("→ Moderate agreement — Shannon estimator cautiously supported")
    else:
        print("→ WARNING: Estimators diverge — interpret Shannon AIS_pre with caution")
else:
    print("  Too few valid subjects for correlation")
    r_val = np.nan

# ── SECTION 2: KSG TE worker (module-level, picklable) ───────────────────
def _worker(fpath_str, sub_id, group, bdi_anh, bdi,
            tau=TAU_KSG, min_trials=MIN_TRIALS, n_samp=N_KSG_SAMPLES):
    """Compute KSG TE(Fz→FCz) and TE(FCz→Fz) for Reward and Loss."""
    import mne
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    def _cmi(X, Y, Z, k=4):
        N = len(X)
        if X.ndim == 1: X = X[:, None]
        if Y.ndim == 1: Y = Y[:, None]
        if Z.ndim == 1: Z = Z[:, None]
        rng = np.random.default_rng(sub_id % 10000)
        X = X + 1e-10 * rng.standard_normal(X.shape)
        Y = Y + 1e-10 * rng.standard_normal(Y.shape)
        Z = Z + 1e-10 * rng.standard_normal(Z.shape)
        XYZ = np.hstack((X, Y, Z))
        dist, _ = cKDTree(XYZ).query(XYZ, k=k+1, p=np.inf)
        eps     = dist[:, k]
        XZ = np.hstack((X, Z)); YZ = np.hstack((Y, Z))
        r   = eps + 1e-12
        txz = cKDTree(XZ); tyz = cKDTree(YZ); tz = cKDTree(Z)
        n_xz = np.array([len(txz.query_ball_point(XZ[i], r[i], p=np.inf)) for i in range(N)])
        n_yz = np.array([len(tyz.query_ball_point(YZ[i], r[i], p=np.inf)) for i in range(N)])
        n_z  = np.array([len(tz.query_ball_point( Z[i],  r[i], p=np.inf)) for i in range(N)])
        ans  = digamma(k) + np.mean(digamma(n_z)) - np.mean(digamma(n_xz)) - np.mean(digamma(n_yz))
        return max(0.0, ans / np.log(2))

    out = {'subject_id': sub_id, 'group': group, 'BDI_Anh': bdi_anh, 'BDI': bdi}
    try:
        epo = mne.read_epochs(fpath_str, preload=True, verbose='ERROR')
        epo.decimate(4)         # 500 → 125 Hz (anti-aliased internally)
    except Exception as e:
        out['error'] = str(e)
        return out

    times     = epo.times
    post_mask = (times >= 0.0) & (times < 0.4)
    try:
        fz_idx  = epo.ch_names.index('Fz')
        fcz_idx = epo.ch_names.index('FCz')
    except ValueError as e:
        out['error'] = str(e)
        return out

    for cond_name in ('Reward', 'Loss'):
        for key in (f'TE_fwd_{cond_name}', f'TE_bwd_{cond_name}',
                    f'TE_net_{cond_name}', f'n_trials_{cond_name}'):
            out[key] = np.nan
        try:
            cdata = epo[cond_name].get_data()   # (n_trials, n_ch, n_t)
        except KeyError:
            continue
        n_trials = cdata.shape[0]
        out[f'n_trials_{cond_name}'] = n_trials
        if n_trials < min_trials:
            continue

        # Build temporal pairs WITHIN each trial to avoid stitching artifacts
        Xf, Yf, Zf = [], [], []
        Xb, Yb, Zb = [], [], []
        for t in range(n_trials):
            fz  = cdata[t, fz_idx,  :][post_mask]
            fcz = cdata[t, fcz_idx, :][post_mask]
            if len(fz) <= tau:
                continue
            Xf.append(fz[:-tau]);  Yf.append(fcz[tau:]); Zf.append(fcz[:-tau])
            Xb.append(fcz[:-tau]); Yb.append(fz[tau:]);  Zb.append(fz[:-tau])

        if not Xf:
            continue

        Xf = np.concatenate(Xf); Yf = np.concatenate(Yf); Zf = np.concatenate(Zf)
        Xb = np.concatenate(Xb); Yb = np.concatenate(Yb); Zb = np.concatenate(Zb)

        N = len(Xf)
        if N > n_samp:
            idx = np.random.default_rng(sub_id + hash(cond_name) % 100).choice(N, n_samp, replace=False)
            idx.sort()
            Xf, Yf, Zf = Xf[idx], Yf[idx], Zf[idx]
            Xb, Yb, Zb = Xb[idx], Yb[idx], Zb[idx]

        te_fwd = _cmi(Xf, Yf, Zf)
        te_bwd = _cmi(Xb, Yb, Zb)
        out[f'TE_fwd_{cond_name}'] = te_fwd
        out[f'TE_bwd_{cond_name}'] = te_bwd
        out[f'TE_net_{cond_name}'] = te_fwd - te_bwd

    return out

# ── Build job list and run ────────────────────────────────────────────────
jobs = []
for fpath in epoch_files:
    try:
        sub_id = int(fpath.stem.split('-')[1].split('_')[0])
    except (IndexError, ValueError):
        continue
    row_c = clinical_main[clinical_main['subject_id'] == sub_id]
    if row_c.empty:
        continue
    rc = row_c.iloc[0]
    jobs.append((str(fpath), sub_id, rc[GROUP_COL],
                 rc.get('BDI_Anh', np.nan), rc.get('BDI', np.nan)))

print(f"\n=== KSG TE ANALYSIS: {len(jobs)} subjects, joblib n_jobs=-1 ===")
t0 = time.time()

raw_results = Parallel(n_jobs=-1, verbose=3)(
    delayed(_worker)(*job) for job in jobs
)

elapsed = time.time() - t0
print(f"\nTotal computation time: {elapsed/60:.1f} min")

errors = [r for r in raw_results if 'error' in r]
if errors:
    print(f"Errors: {len(errors)} subjects")
    for e in errors[:5]:
        print(f"  sub-{e['subject_id']}: {e['error']}")

df_te = pd.DataFrame([r for r in raw_results if 'error' not in r])
print(f"Processed: {len(df_te)} subjects")
print(f"NaN rates — TE_fwd_Reward: {df_te['TE_fwd_Reward'].isna().mean():.1%}, "
      f"TE_fwd_Loss: {df_te['TE_fwd_Loss'].isna().mean():.1%}")

# ── SECTION 3: Statistics ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KSG TE RESULTS")
print("=" * 70)

stat_rows = []
metrics = [
    ('TE_fwd_Reward',  'TE Fz→FCz (Reward)',    'CTL>MDD'),
    ('TE_fwd_Loss',    'TE Fz→FCz (Loss)',       'CTL>MDD'),
    ('TE_bwd_Reward',  'TE FCz→Fz (Reward)',     'no_hyp'),
    ('TE_bwd_Loss',    'TE FCz→Fz (Loss)',       'no_hyp'),
    ('TE_net_Reward',  'Net TE Reward (fwd-bwd)', 'CTL>MDD'),
    ('TE_net_Loss',    'Net TE Loss (fwd-bwd)',   'CTL>MDD'),
]

# Add asymmetry columns
df_te['TE_fwd_asym'] = df_te['TE_fwd_Reward'] - df_te['TE_fwd_Loss']
df_te['TE_net_asym'] = df_te['TE_net_Reward']  - df_te['TE_net_Loss']
metrics += [
    ('TE_fwd_asym', 'TE_fwd Reward−Loss asymmetry', 'CTL>MDD'),
    ('TE_net_asym', 'Net TE Reward−Loss asymmetry',  'CTL>MDD'),
]

for col, label, h_dir in metrics:
    if col not in df_te.columns:
        continue
    ctl = df_te[df_te['group'] == CTL_LABEL][col].dropna()
    mdd = df_te[df_te['group'] == MDD_LABEL][col].dropna()
    if len(ctl) < 3 or len(mdd) < 3:
        continue
    U, p = mannwhitneyu(ctl, mdd, alternative='two-sided')
    d    = cohens_d(ctl.values, mdd.values)
    h1   = ctl.mean() > mdd.mean() if h_dir == 'CTL>MDD' else None
    stat_rows.append({'metric': label, 'col': col,
                      'N_CTL': len(ctl), 'N_MDD': len(mdd),
                      'CTL_mean': ctl.mean(), 'CTL_std': ctl.std(),
                      'MDD_mean': mdd.mean(), 'MDD_std': mdd.std(),
                      'd': d, 'p_raw': p, 'H_dir': h_dir,
                      'H_confirmed': h1})

df_stats = pd.DataFrame(stat_rows)
_, p_fdr, _, _ = multipletests(df_stats['p_raw'], method='fdr_bh')
df_stats['p_fdr'] = p_fdr

print(f"\n{'Metric':<38} {'CTL':>8} {'MDD':>8} {'d':>7} {'p_raw':>8} {'p_fdr':>8} H")
print("-" * 90)
for _, row in df_stats.iterrows():
    h_sym = '✓' if row['H_confirmed'] else ('✗' if row['H_confirmed'] is False else '—')
    sig   = '***' if row['p_fdr'] < 0.001 else ('**' if row['p_fdr'] < 0.01 else
            ('*' if row['p_fdr'] < 0.05 else ('.' if row['p_raw'] < 0.10 else '')))
    print(f"  {row['metric']:<36} {row['CTL_mean']:>8.4f} {row['MDD_mean']:>8.4f} "
          f"{row['d']:>7.3f} {row['p_raw']:>8.4f} {row['p_fdr']:>8.4f} {h_sym} {sig}")

# ── SECTION 4: Correlation matrix (AIS_pre × TE × BDI_Anh) ──────────────
print("\n=== CORRELATION MATRIX: AIS_pre × TE_fwd × BDI_Anh ===")
df_ais = pd.read_csv(OUT_DIR / 'delta_ais_aggregated.csv')
df_merge = df_te[['subject_id', 'group', 'BDI_Anh', 'BDI',
                   'TE_fwd_Reward', 'TE_fwd_Loss', 'TE_fwd_asym']].merge(
    df_ais[['subject_id', 'mean_AIS_pre', 'mean_AIS_post', 'mean_delta_AIS']],
    on='subject_id', how='inner'
)
df_merge['TE_fwd_mean'] = df_merge[['TE_fwd_Reward', 'TE_fwd_Loss']].mean(axis=1)

corr_cols = ['mean_AIS_pre', 'TE_fwd_mean', 'BDI_Anh']
corr_labels = ['AIS_pre', 'TE_fwd_mean', 'BDI_Anh']
valid = df_merge[corr_cols + ['group']].dropna()

print(f"\nCorrelation matrix (N={len(valid)}, Pearson r / p_perm below):")
print(f"{'':>20}", end='')
for lbl in corr_labels:
    print(f"  {lbl:>14}", end='')
print()

rng_perm = np.random.default_rng(42)
corr_matrix = {}
for i, (c1, l1) in enumerate(zip(corr_cols, corr_labels)):
    print(f"  {l1:<18}", end='')
    for j, (c2, l2) in enumerate(zip(corr_cols, corr_labels)):
        if i == j:
            print(f"  {'1.000':>14}", end='')
        else:
            x, y = valid[c1].values, valid[c2].values
            r    = pearsonr(x, y)[0]
            null = [pearsonr(rng_perm.permutation(x), y)[0] for _ in range(2000)]
            p    = np.mean(np.abs(null) >= np.abs(r))
            key  = tuple(sorted([l1, l2]))
            corr_matrix[key] = (r, p)
            star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {f'r={r:+.3f}{star}':>14}", end='')
    print()

print("\nDetailed p-values:")
for (l1, l2), (r, p) in corr_matrix.items():
    print(f"  {l1} vs {l2}: r={r:+.3f}, p_perm={p:.4f}")

# ── SECTION 5: TE vs AIS_pre independence test ───────────────────────────
print("\n=== INDEPENDENCE: TE_fwd is orthogonal to AIS_pre? ===")
key_pair = tuple(sorted(['AIS_pre', 'TE_fwd_mean']))
if key_pair in corr_matrix:
    r_ind, p_ind = corr_matrix[key_pair]
    print(f"  r(AIS_pre, TE_fwd_mean) = {r_ind:+.3f}, p_perm = {p_ind:.4f}")
    if abs(r_ind) < 0.20:
        print("  → Measures are largely INDEPENDENT (|r| < 0.20) ✓")
    else:
        print("  → Moderate correlation — measures share variance")

# ── SECTION 6: Visualization ──────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
palette = {CTL_LABEL: '#2196F3', MDD_LABEL: '#F44336'}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('KSG Transfer Entropy Fz → FCz (post-feedback 0–400ms, 125Hz)\n'
             'Secondary biomarker analysis', fontsize=11, fontweight='bold')

# A: TE_fwd by condition and group
ax = axes[0, 0]
df_melt = df_te[['group', 'TE_fwd_Reward', 'TE_fwd_Loss']].rename(
    columns={'TE_fwd_Reward': 'Reward', 'TE_fwd_Loss': 'Loss'}
).melt(id_vars='group', var_name='condition', value_name='TE_fwd')
sns.boxplot(data=df_melt, x='condition', y='TE_fwd', hue='group',
            ax=ax, palette=palette, width=0.5)
sns.stripplot(data=df_melt, x='condition', y='TE_fwd', hue='group',
              ax=ax, dodge=True, alpha=0.3, size=3, palette=palette, legend=False)
ax.set_title('A. TE Fz→FCz by Condition')
ax.set_ylabel('TE [bits]')
ax.get_legend().set_title('')

# B: TE asymmetry (Reward - Loss)
ax = axes[0, 1]
sns.boxplot(data=df_te, x='group', y='TE_fwd_asym', ax=ax, palette=palette, width=0.5)
sns.stripplot(data=df_te, x='group', y='TE_fwd_asym', ax=ax,
              color='black', alpha=0.4, size=3)
row_asym = df_stats[df_stats['col'] == 'TE_fwd_asym']
title_d  = f"d={row_asym.iloc[0]['d']:.2f}, p_fdr={row_asym.iloc[0]['p_fdr']:.3f}" if len(row_asym) else ""
ax.axhline(0, color='gray', ls='--', alpha=0.5, lw=1)
ax.set_title(f'B. TE Asymmetry (Reward−Loss)\n{title_d}')
ax.set_ylabel('ΔTE Reward−Loss [bits]')
ax.set_xlabel('')

# C: KSG vs Shannon AIS_pre (validation scatter)
ax = axes[0, 2]
for grp, color in [(CTL_LABEL, '#2196F3'), (MDD_LABEL, '#F44336')]:
    sub = df_val[df_val['group'] == grp]
    ax.scatter(sub['shannon_ais'], sub['ksg_ais'], c=color, alpha=0.7, s=60, label=grp)
if len(df_val) >= 4:
    xl = np.linspace(df_val['shannon_ais'].min(), df_val['shannon_ais'].max(), 50)
    z  = np.polyfit(df_val['shannon_ais'], df_val['ksg_ais'], 1)
    ax.plot(xl, np.polyval(z, xl), 'k-', alpha=0.6, lw=1.5,
            label=f'r={r_val:.2f}' if np.isfinite(r_val) else '')
ax.set_xlabel('Shannon AIS_pre [bits]')
ax.set_ylabel('KSG AIS_pre [bits]')
ax.set_title('C. KSG vs Shannon AIS_pre\n(Estimator validation, N=20)')
ax.legend(fontsize=8)

# D: AIS_pre vs TE_fwd scatter
ax = axes[1, 0]
for grp, color in [(CTL_LABEL, '#2196F3'), (MDD_LABEL, '#F44336')]:
    sub = df_merge[df_merge['group'] == grp].dropna(subset=['mean_AIS_pre', 'TE_fwd_mean'])
    ax.scatter(sub['mean_AIS_pre'], sub['TE_fwd_mean'], c=color, alpha=0.55, s=45, label=grp)
if len(valid) > 5:
    v = valid.dropna()
    z = np.polyfit(v['mean_AIS_pre'], v['TE_fwd_mean'], 1)
    xl = np.linspace(v['mean_AIS_pre'].min(), v['mean_AIS_pre'].max(), 50)
    ax.plot(xl, np.polyval(z, xl), 'k-', alpha=0.5, lw=1.5)
key_ai_te = tuple(sorted(['AIS_pre', 'TE_fwd_mean']))
r_ai_te = corr_matrix.get(key_ai_te, (np.nan, np.nan))[0]
ax.set_xlabel('Mean AIS_pre [bits]')
ax.set_ylabel('Mean TE_fwd [bits]')
ax.set_title(f'D. AIS_pre vs TE_fwd\nr={r_ai_te:.3f} (independence check)')
ax.legend(fontsize=8)

# E: TE_fwd vs BDI_Anh
ax = axes[1, 1]
valid_te_anh = df_merge[['BDI_Anh', 'TE_fwd_mean', 'group']].dropna()
for grp, color in [(CTL_LABEL, '#2196F3'), (MDD_LABEL, '#F44336')]:
    sub = valid_te_anh[valid_te_anh['group'] == grp]
    ax.scatter(sub['BDI_Anh'], sub['TE_fwd_mean'], c=color, alpha=0.55, s=45, label=grp)
if len(valid_te_anh) > 5:
    z  = np.polyfit(valid_te_anh['BDI_Anh'], valid_te_anh['TE_fwd_mean'], 1)
    xl = np.linspace(valid_te_anh['BDI_Anh'].min(), valid_te_anh['BDI_Anh'].max(), 50)
    ax.plot(xl, np.polyval(z, xl), 'k-', alpha=0.5, lw=1.5)
key_te_anh = tuple(sorted(['TE_fwd_mean', 'BDI_Anh']))
r_te_anh = corr_matrix.get(key_te_anh, (np.nan, np.nan))[0]
ax.set_xlabel('BDI_Anh (anhedonia)')
ax.set_ylabel('Mean TE_fwd [bits]')
ax.set_title(f'E. TE_fwd vs Anhedonia\nr={r_te_anh:.3f}')
ax.legend(fontsize=8)

# F: Summary panel
ax = axes[1, 2]
ax.axis('off')
sig_rows = df_stats[df_stats['p_fdr'] < 0.05]
trend_rows = df_stats[(df_stats['p_raw'] < 0.10) & (df_stats['p_fdr'] >= 0.05)]
lines = ["KSG TE SUMMARY", "=" * 30, ""]
lines.append(f"Validation: KSG vs Shannon r={r_val:.2f}" if np.isfinite(r_val) else "Validation: N/A")
lines.append("")
lines.append(f"Significant (p_fdr<0.05): N={len(sig_rows)}")
for _, row in sig_rows.iterrows():
    lines.append(f"  {row['metric']}: d={row['d']:.2f}")
lines.append(f"\nTrend (p_raw<0.10): N={len(trend_rows)}")
for _, row in trend_rows.iterrows():
    lines.append(f"  {row['metric']}: d={row['d']:.2f}")
lines.append("")
lines.append("Correlation matrix:")
for (l1, l2), (r, p) in sorted(corr_matrix.items()):
    star = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    lines.append(f"  {l1[:8]}×{l2[:8]}: r={r:+.3f}{star}")
ax.text(0.04, 0.97, "\n".join(lines), transform=ax.transAxes,
        fontsize=8, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUT_DIR / 'ksg_te_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUT_DIR / 'ksg_te_results.png'}")

# ── Save ──────────────────────────────────────────────────────────────────
df_te.to_csv(    OUT_DIR / 'ksg_te_subject_level.csv', index=False)
df_stats.to_csv( OUT_DIR / 'ksg_te_stats.csv',         index=False)
df_merge.to_csv( OUT_DIR / 'ksg_te_ais_merged.csv',    index=False)
df_val.to_csv(   OUT_DIR / 'ksg_ais_validation.csv',   index=False)

print(f"\n{'='*60}")
print("SCRIPT 13 COMPLETE")
print(f"{'='*60}")
print(f"  Subjects processed:   {len(df_te)}")
print(f"  Computation time:     {elapsed/60:.1f} min")
print(f"  KSG vs Shannon r:     {r_val:.3f}" if np.isfinite(r_val) else "  KSG vs Shannon:      N/A")
primary = df_stats[df_stats['col'].isin(['TE_fwd_Reward', 'TE_fwd_Loss'])]
for _, row in primary.iterrows():
    print(f"  {row['metric']}: d={row['d']:.3f}, p_fdr={row['p_fdr']:.4f}")
print(f"\nNext: Script 14 — integrated results plot (IRP)")
