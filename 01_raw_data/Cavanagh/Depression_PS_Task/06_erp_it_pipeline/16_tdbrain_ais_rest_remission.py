#!/usr/bin/env python3
"""
16_tdbrain_ais_rest_remission.py
TDBRAIN MDD-rTMS: AIS_restEC as pre-treatment predictor of remission.

Hypothesis: lower resting AIS at FCz (scar marker) → worse rTMS response.

Signal:  restEC, FCz, 1Hz HP filtered, 2s non-overlapping windows.
AIS:     lag=1, bins=4 (identical to PST analysis), min 10 valid windows.
Outcomes: Remitter (BDI_post ≤ 12), Responder (≥50% BDI reduction),
          BDI_change (continuous).
Controls: BDI_pre, age, sex in logistic regression.
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
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('ERROR')

BASE    = Path("/media/neuraldyn/PortableSSD/TDBRAIN_derivatives")
OUT_DIR = Path("/media/neuraldyn/PortableSSD/DEPRESSION/06_manuscript_assets")

# ── Parameters ────────────────────────────────────────────────────────────────
SFREQ_EXPECTED = 500
WIN_SAMPLES    = 1000          # 2s × 500Hz
MIN_WINS       = 10
HP_FREQ        = 1.0           # Hz — removes DC, keeps EEG band
CHAN           = 'FCz'
DC_THRESH      = 15.0          # µV
FLAT_THRESH    = 2.0           # µV std

ROBUSTNESS_VARIANTS = [
    ('2s/lag=1/bins=4',  1000, 1, 4),
    ('1s/lag=1/bins=4',   500, 1, 4),
    ('2s/lag=1/bins=6',  1000, 1, 6),
    ('2s/lag=1/bins=8',  1000, 1, 8),
]

# ── AIS (identical to PST pipeline) ──────────────────────────────────────────
def safe_ais(x, lag=1, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * lag + 10 or np.std(x) < 1e-12:
        return np.nan
    try:
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return np.nan
        bins    = np.digitize(x, edges[1:-1])
        x_t     = bins[lag:]
        x_lag   = bins[:-lag]
        joint   = np.zeros((n_bins, n_bins))
        for a, b in zip(x_t, x_lag):
            joint[min(a-1, n_bins-1), min(b-1, n_bins-1)] += 1
        joint  /= joint.sum() + 1e-10
        px_t    = joint.sum(axis=1)
        px_lag  = joint.sum(axis=0)
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
_ar = np.zeros(300)
for i in range(1, 300):
    _ar[i] = 0.9 * _ar[i-1] + 0.1 * rng.standard_normal()
assert safe_ais(_ar) > safe_ais(rng.standard_normal(300)), "AIS sanity failed"
print("AIS sanity check OK")


def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pool = np.sqrt(((n1-1)*np.var(a, ddof=1) +
                    (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pool if pool > 0 else 0.0


# ── Load participants — MDD-rTMS, session 1 only ──────────────────────────────
print("\n=== LOADING PARTICIPANTS ===")
parts = pd.read_csv(BASE / "TDBRAIN_participants_V2.tsv", sep='\t')

# Keep MDD-rTMS arm, sessID=1 (pre-treatment recording)
mdd = parts[(parts['Dataset'] == 'MDD-rTMS') &
            (parts['sessID'].astype(str) == '1.0')].copy()

# Parse BDI (European comma decimals in some cells)
for col in ['BDI_pre', 'BDI_post']:
    mdd[col] = pd.to_numeric(
        mdd[col].astype(str).str.replace(',', '.'), errors='coerce')
mdd['age'] = pd.to_numeric(
    mdd['age'].astype(str).str.replace(',', '.'), errors='coerce')
mdd['gender'] = pd.to_numeric(mdd['gender'], errors='coerce')

# Define outcome variables
mdd['BDI_change']      = mdd['BDI_post'] - mdd['BDI_pre']
mdd['BDI_pct_change']  = 100 * mdd['BDI_change'] / mdd['BDI_pre'].replace(0, np.nan)
mdd['remitter_bdi']    = (mdd['BDI_post'] <= 12).astype(int)
mdd['responder_50pct'] = (mdd['BDI_pct_change'] <= -50).astype(int)
# Also use pre-coded labels
mdd['remitter_coded']  = pd.to_numeric(mdd['Remitter'], errors='coerce')
mdd['responder_coded'] = pd.to_numeric(mdd['Responder'], errors='coerce')

print(f"MDD-rTMS sessID=1: N={len(mdd)}")
print(f"BDI_pre:  {mdd['BDI_pre'].mean():.1f} ± {mdd['BDI_pre'].std():.1f}")
print(f"BDI_post: {mdd['BDI_post'].mean():.1f} ± {mdd['BDI_post'].std():.1f}")
print(f"Remitters (BDI≤12): {mdd['remitter_bdi'].sum()} / {mdd['remitter_bdi'].notna().sum()}")
print(f"Responders (≥50%):  {mdd['responder_50pct'].sum()} / {mdd['responder_50pct'].notna().sum()}")

# ── PHASE 1: Compute AIS_rest for each subject ────────────────────────────────
print("\n=== PHASE 1: AIS_rest COMPUTATION ===")

records    = []
quality_log = []

for i, row in mdd.iterrows():
    sub_id = row['participants_ID']
    eeg_dir = BASE / sub_id / "ses-1" / "eeg"
    if not eeg_dir.exists():
        quality_log.append({'subject': sub_id, 'reason': 'no eeg dir'})
        continue

    vhdr_files = list(eeg_dir.glob("*restEC*.vhdr"))
    if not vhdr_files:
        quality_log.append({'subject': sub_id, 'reason': 'no restEC vhdr'})
        continue

    try:
        raw = mne.io.read_raw_brainvision(
            vhdr_files[0], preload=True, verbose='ERROR')

        if raw.info['sfreq'] != SFREQ_EXPECTED:
            raw.resample(SFREQ_EXPECTED, verbose='ERROR')

        # Drop non-EEG channels
        drop = [c for c in raw.ch_names
                if c in ['VEOG', 'HEOG', 'EMG', 'ECG', 'EKG', 'Status']]
        if drop:
            raw.drop_channels(drop)

        if CHAN not in raw.ch_names:
            quality_log.append({'subject': sub_id, 'reason': f'no {CHAN}'})
            continue

        # 1Hz HP filter to remove DC
        raw.filter(l_freq=HP_FREQ, h_freq=None, verbose='ERROR')

        sig = raw.get_data(picks=[CHAN])[0] * 1e6   # V → µV
        n_wins = len(sig) // WIN_SAMPLES

        win_ais, win_failed = [], 0
        for w in range(n_wins):
            s = sig[w * WIN_SAMPLES:(w+1) * WIN_SAMPLES]
            # Per-window quality check
            if abs(s.mean()) > DC_THRESH:
                win_failed += 1
                quality_log.append({'subject': sub_id,
                                    'reason': f'win{w} DC {s.mean():.1f}µV'})
                continue
            if s.std() < FLAT_THRESH:
                win_failed += 1
                quality_log.append({'subject': sub_id,
                                    'reason': f'win{w} flat std={s.std():.2f}µV'})
                continue
            ais_val = safe_ais(s, lag=1, n_bins=4)
            if np.isfinite(ais_val):
                win_ais.append(ais_val)

        if len(win_ais) < MIN_WINS:
            quality_log.append({'subject': sub_id,
                                 'reason': f'only {len(win_ais)} valid windows'})
            continue

        n_idx = len(records) + 1
        if n_idx % 20 == 0 or n_idx == 1:
            print(f"  [{n_idx}] {sub_id}: "
                  f"AIS_rest={np.mean(win_ais):.4f}  "
                  f"valid_wins={len(win_ais)}/{n_wins}")

        records.append({
            'subject_id':      sub_id,
            'AIS_rest':        np.mean(win_ais),
            'AIS_rest_std':    np.std(win_ais),
            'n_valid_wins':    len(win_ais),
            'n_total_wins':    n_wins,
            'BDI_pre':         row['BDI_pre'],
            'BDI_post':        row['BDI_post'],
            'BDI_change':      row['BDI_change'],
            'BDI_pct_change':  row['BDI_pct_change'],
            'remitter_bdi':    row['remitter_bdi'],
            'responder_50pct': row['responder_50pct'],
            'remitter_coded':  row['remitter_coded'],
            'responder_coded': row['responder_coded'],
            'age':             row['age'],
            'gender':          row['gender'],
        })

    except Exception as e:
        quality_log.append({'subject': sub_id, 'reason': f'error: {str(e)[:60]}'})

df = pd.DataFrame(records)
print(f"\nProcessed: {len(df)} subjects")
print(f"Excluded:  {len(quality_log)} entries")
print(f"AIS_rest:  {df['AIS_rest'].mean():.4f} ± {df['AIS_rest'].std():.4f}")

df.to_csv(OUT_DIR / "tdbrain_ais_rest_results.csv", index=False)
pd.DataFrame(quality_log).to_csv(
    OUT_DIR / "tdbrain_ais_rest_quality_log.csv", index=False)

if len(df) < 20:
    print("ERROR: too few subjects to analyse"); sys.exit(1)

# ── PHASE 2: PRIMARY ANALYSES ─────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2: PRIMARY ANALYSES")
print("="*60)

rem  = df[df['remitter_bdi'] == 1]['AIS_rest'].dropna().values
nrem = df[df['remitter_bdi'] == 0]['AIS_rest'].dropna().values
resp = df[df['responder_50pct'] == 1]['AIS_rest'].dropna().values
nrsp = df[df['responder_50pct'] == 0]['AIS_rest'].dropna().values

# 1. Remitters vs Non-remitters
print("\n--- 1. AIS_rest: Remitters vs Non-remitters (BDI≤12) ---")
if len(rem) >= 5 and len(nrem) >= 5:
    _, p_rem = mannwhitneyu(rem, nrem, alternative='two-sided')
    d_rem    = cohen_d(rem, nrem)
    print(f"Remitter     (N={len(rem)}):  {rem.mean():.4f} ± {rem.std():.4f}")
    print(f"Non-remitter (N={len(nrem)}): {nrem.mean():.4f} ± {nrem.std():.4f}")
    print(f"Cohen's d = {d_rem:.3f}  MWU p = {p_rem:.4f}")
    print(f"Direction (remitter > non-remitter): {'✓' if rem.mean() > nrem.mean() else '✗'}")
else:
    print("  Insufficient N"); p_rem = d_rem = np.nan

# 2. Responders vs Non-responders
print("\n--- 2. AIS_rest: Responders vs Non-responders (≥50% BDI reduction) ---")
if len(resp) >= 5 and len(nrsp) >= 5:
    _, p_rsp = mannwhitneyu(resp, nrsp, alternative='two-sided')
    d_rsp    = cohen_d(resp, nrsp)
    print(f"Responder    (N={len(resp)}):  {resp.mean():.4f} ± {resp.std():.4f}")
    print(f"Non-responder(N={len(nrsp)}): {nrsp.mean():.4f} ± {nrsp.std():.4f}")
    print(f"Cohen's d = {d_rsp:.3f}  MWU p = {p_rsp:.4f}")
else:
    print("  Insufficient N"); p_rsp = d_rsp = np.nan

# 3. r(AIS_rest, BDI_change) — continuous
print("\n--- 3. r(AIS_rest, BDI_change) ---")
valid_chg = df[['AIS_rest', 'BDI_change']].dropna()
r_chg, p_chg = pearsonr(valid_chg['AIS_rest'], valid_chg['BDI_change'])
rs_chg, _    = spearmanr(valid_chg['AIS_rest'], valid_chg['BDI_change'])
print(f"N={len(valid_chg)}")
print(f"Pearson  r={r_chg:+.3f}, p={p_chg:.4f}")
print(f"Spearman r={rs_chg:+.3f}")
print(f"Interpretation: {'higher AIS → less BDI reduction (anti-scar)' if r_chg > 0 else 'higher AIS → more BDI reduction (scar-consistent)'}")

# 4. CRITICAL CONTROL: r(AIS_rest, BDI_pre)
print("\n--- 4. CONTROL: r(AIS_rest, BDI_pre) ---")
valid_pre = df[['AIS_rest', 'BDI_pre']].dropna()
r_pre, p_pre = pearsonr(valid_pre['AIS_rest'], valid_pre['BDI_pre'])
print(f"N={len(valid_pre)}")
print(f"Pearson r={r_pre:+.3f}, p={p_pre:.4f}")
if abs(r_pre) > 0.4 and p_pre < 0.05:
    print("  ⚠ STRONG correlation with BDI_pre — severity confound likely")
elif abs(r_pre) < 0.2 or p_pre > 0.05:
    print("  ✓ Weak/null correlation — AIS_rest is NOT just a severity marker")
else:
    print("  → Moderate correlation — logistic regression will partial out")

# 5. Logistic regression: AIS_rest → remission, controlling for BDI_pre
print("\n--- 5. Logistic regression: AIS_rest + BDI_pre → remission ---")
lr_data = df[['AIS_rest', 'BDI_pre', 'age', 'gender', 'remitter_bdi']].dropna()
if len(lr_data) >= 20:
    X_lr = lr_data[['AIS_rest', 'BDI_pre']].values
    y_lr = lr_data['remitter_bdi'].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_lr)

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_sc, y_lr)
    proba = lr.predict_proba(X_sc)[:, 1]
    auc_full = roc_auc_score(y_lr, proba)

    lr_bdi = LogisticRegression(random_state=42, max_iter=1000)
    lr_bdi.fit(X_sc[:, [1]], y_lr)
    auc_bdi_only = roc_auc_score(y_lr, lr_bdi.predict_proba(X_sc[:, [1]])[:, 1])

    lr_ais = LogisticRegression(random_state=42, max_iter=1000)
    lr_ais.fit(X_sc[:, [0]], y_lr)
    auc_ais_only = roc_auc_score(y_lr, lr_ais.predict_proba(X_sc[:, [0]])[:, 1])

    print(f"N={len(lr_data)}")
    print(f"AUC (AIS_rest only):          {auc_ais_only:.3f}")
    print(f"AUC (BDI_pre only):           {auc_bdi_only:.3f}")
    print(f"AUC (AIS_rest + BDI_pre):     {auc_full:.3f}")
    print(f"Coefs (standardized): AIS={lr.coef_[0][0]:+.3f}  BDI={lr.coef_[0][1]:+.3f}")
    inc_value = auc_full - auc_bdi_only
    print(f"Incremental AUC from AIS_rest: {inc_value:+.3f}")
    print(f"Interpretation: {'✓ AIS_rest adds predictive value beyond severity' if inc_value > 0.02 else '→ AIS_rest adds little beyond BDI_pre'}")

# ── PHASE 3: ROBUSTNESS ───────────────────────────────────────────────────────
print("\n=== PHASE 3: ROBUSTNESS ===")

for label, win_samp, lag, n_bins in ROBUSTNESS_VARIANTS:
    rob_recs = []
    for _, row in mdd.iterrows():
        sub_id  = row['participants_ID']
        eeg_dir = BASE / sub_id / "ses-1" / "eeg"
        vhdr_f  = list(eeg_dir.glob("*restEC*.vhdr")) if eeg_dir.exists() else []
        if not vhdr_f:
            continue
        try:
            r = mne.io.read_raw_brainvision(vhdr_f[0], preload=True, verbose='ERROR')
            r.drop_channels([c for c in r.ch_names
                             if c in ['VEOG','HEOG','EMG','ECG','EKG','Status']],
                            on_missing='ignore')
            r.filter(l_freq=HP_FREQ, h_freq=None, verbose='ERROR')
            if CHAN not in r.ch_names:
                continue
            s_full = r.get_data(picks=[CHAN])[0] * 1e6
            n_w    = len(s_full) // win_samp
            ais_v  = []
            for w in range(n_w):
                s = s_full[w*win_samp:(w+1)*win_samp]
                if abs(s.mean()) > DC_THRESH or s.std() < FLAT_THRESH:
                    continue
                v = safe_ais(s, lag=lag, n_bins=n_bins)
                if np.isfinite(v):
                    ais_v.append(v)
            if len(ais_v) >= MIN_WINS:
                rob_recs.append({'subject_id': sub_id,
                                 'AIS_rob': np.mean(ais_v),
                                 'remitter_bdi': row['remitter_bdi']})
        except Exception:
            pass

    df_r = pd.DataFrame(rob_recs).dropna()
    if len(df_r) < 10:
        print(f"  {label}: too few subjects")
        continue
    a = df_r[df_r['remitter_bdi']==1]['AIS_rob'].values
    b = df_r[df_r['remitter_bdi']==0]['AIS_rob'].values
    if len(a) < 3 or len(b) < 3:
        continue
    _, p_r = mannwhitneyu(a, b, alternative='two-sided')
    d_r    = cohen_d(a, b)
    print(f"  {label}: d={d_r:.3f}, p={p_r:.4f}  "
          f"(rem={a.mean():.4f}, non-rem={b.mean():.4f}, N={len(df_r)})")

# ── PHASE 4: FIGURE ───────────────────────────────────────────────────────────
print("\n=== PHASE 4: FIGURE ===")

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(
    'TDBRAIN MDD-rTMS: AIS_restEC (FCz) as Pre-treatment Predictor of Remission\n'
    'Hypothesis: lower resting AIS = neural scar → worse response',
    fontsize=11, fontweight='bold')

REM_C  = '#1B5E20'
NREM_C = '#B71C1C'
GRAY   = '#555555'

rng_j = np.random.default_rng(7)

# Panel A: Remitter vs Non-remitter
ax = axes[0]
for grp, arr, x, col, lbl in [
        ('Remitter',     rem,  0, REM_C,  f'Remitter\n(N={len(rem)})'),
        ('Non-remitter', nrem, 1, NREM_C, f'Non-rem.\n(N={len(nrem)})')]:
    jit = rng_j.uniform(-0.12, 0.12, len(arr))
    ax.scatter(x + jit, arr, color=col, alpha=0.45, s=16, linewidths=0)
    m, se = arr.mean(), arr.std(ddof=1)/np.sqrt(len(arr))
    ax.errorbar(x, m, yerr=[[1.96*se],[1.96*se]], fmt='o', color='black',
                capsize=4, capthick=1.2, elinewidth=1.2, markersize=6, zorder=5)
y_top = max(np.nanmax(rem), np.nanmax(nrem)) * 1.03
ax.annotate('', xy=(1, y_top), xytext=(0, y_top),
            arrowprops=dict(arrowstyle='<->', color='black', lw=0.9))
p_str = f'p={p_rem:.3f}' if not np.isnan(p_rem) else ''
ax.text(0.5, y_top*1.005,
        f'd={d_rem:.2f},  {p_str}' if not np.isnan(d_rem) else '',
        ha='center', va='bottom', fontsize=8.5)
ax.set_xticks([0, 1])
ax.set_xticklabels([f'Remitter\n(N={len(rem)})', f'Non-rem.\n(N={len(nrem)})'])
ax.set_ylabel('AIS_rest  [bits]')
ax.set_title('A. Remission\n(BDI_post ≤ 12)', loc='left', fontsize=10)

# Panel B: Responder vs Non-responder
ax = axes[1]
for grp, arr, x, col in [
        ('Resp',  resp, 0, '#2196F3'),
        ('NResp', nrsp, 1, '#E53935')]:
    if len(arr) == 0: continue
    jit = rng_j.uniform(-0.12, 0.12, len(arr))
    ax.scatter(x + jit, arr, color=col, alpha=0.45, s=16, linewidths=0)
    m, se = arr.mean(), arr.std(ddof=1)/np.sqrt(len(arr))
    ax.errorbar(x, m, yerr=[[1.96*se],[1.96*se]], fmt='o', color='black',
                capsize=4, capthick=1.2, elinewidth=1.2, markersize=6, zorder=5)
ax.set_xticks([0, 1])
ax.set_xticklabels([f'Responder\n(N={len(resp)})', f'Non-resp.\n(N={len(nrsp)})'])
ax.set_ylabel('AIS_rest  [bits]')
d_str = f'd={d_rsp:.2f},  p={p_rsp:.3f}' if not np.isnan(d_rsp) else ''
ax.set_title(f'B. Response\n(≥50% BDI reduction)\n{d_str}', loc='left', fontsize=10)

# Panel C: AIS_rest vs BDI_change scatter
ax = axes[2]
valid_plt = df[['AIS_rest', 'BDI_change', 'remitter_bdi']].dropna()
for rem_v, col in [(1, REM_C), (0, NREM_C)]:
    sub = valid_plt[valid_plt['remitter_bdi'] == rem_v]
    ax.scatter(sub['AIS_rest'], sub['BDI_change'],
               color=col, alpha=0.5, s=20, linewidths=0,
               label=f'{"Remitter" if rem_v else "Non-remitter"} (N={len(sub)})')
z  = np.polyfit(valid_plt['AIS_rest'], valid_plt['BDI_change'], 1)
xl = np.linspace(valid_plt['AIS_rest'].min(), valid_plt['AIS_rest'].max(), 200)
ax.plot(xl, np.polyval(z, xl), color=GRAY, lw=1.5)
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.set_xlabel('AIS_rest (bits)')
ax.set_ylabel('BDI change (post − pre)')
p_str2 = f'p={p_chg:.3f}' if not np.isnan(p_chg) else ''
ax.set_title(f'C. AIS_rest vs BDI change\nr={r_chg:+.3f}, {p_str2}', loc='left', fontsize=10)
ax.legend(fontsize=7, frameon=False)

# Panel D: Control — AIS_rest vs BDI_pre
ax = axes[3]
valid_ctrl = df[['AIS_rest', 'BDI_pre']].dropna()
ax.scatter(valid_ctrl['BDI_pre'], valid_ctrl['AIS_rest'],
           color='#78909C', alpha=0.5, s=20, linewidths=0)
z2  = np.polyfit(valid_ctrl['BDI_pre'], valid_ctrl['AIS_rest'], 1)
xl2 = np.linspace(valid_ctrl['BDI_pre'].min(), valid_ctrl['BDI_pre'].max(), 200)
ax.plot(xl2, np.polyval(z2, xl2), color=GRAY, lw=1.5)
ax.set_xlabel('BDI_pre (severity at baseline)')
ax.set_ylabel('AIS_rest (bits)')
p_str3 = f'p={p_pre:.3f}' if not np.isnan(p_pre) else ''
ax.set_title(f'D. CONTROL: AIS_rest vs BDI_pre\nr={r_pre:+.3f}, {p_str3}\n'
             f'{"⚠ Severity confound" if abs(r_pre)>0.4 else "✓ Independent of severity"}',
             loc='left', fontsize=10)

plt.tight_layout()
out_fig = OUT_DIR / 'tdbrain_ais_rest_remission.png'
fig.savefig(out_fig, dpi=200, bbox_inches='tight')
print(f"Figure saved: {out_fig}")

# ── FINAL VERDICT ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TDBRAIN — AIS_restEC REMISSION PREDICTION — VERDICT")
print("="*60)
print(f"\nN processed: {len(df)}")
print(f"AIS_rest: {df['AIS_rest'].mean():.4f} ± {df['AIS_rest'].std():.4f}")
print(f"\n1. Remitter vs Non-remitter:")
if not np.isnan(d_rem):
    print(f"   d={d_rem:.3f}, p={p_rem:.4f}  "
          f"({'✓ remitters have higher AIS_rest' if rem.mean()>nrem.mean() else '✗ wrong direction'})")
print(f"\n2. r(AIS_rest, BDI_change) = {r_chg:+.3f}, p={p_chg:.4f}")
print(f"   {'(higher AIS → more improvement: scar-consistent)' if r_chg > 0 else '(higher AIS → less improvement)'}")
print(f"\n3. CONTROL r(AIS_rest, BDI_pre) = {r_pre:+.3f}, p={p_pre:.4f}")
if abs(r_pre) < 0.2 or p_pre > 0.05:
    print("   ✓ AIS_rest is NOT a severity proxy — scar interpretation viable")
elif abs(r_pre) > 0.4 and p_pre < 0.05:
    print("   ⚠ AIS_rest correlates with severity — scar claim needs regression control")

print(f"\n4. Logistic regression:")
if 'auc_full' in dir():
    print(f"   AUC (AIS only)={auc_ais_only:.3f}  "
          f"BDI only={auc_bdi_only:.3f}  "
          f"Both={auc_full:.3f}")
    print(f"   Incremental AUC: {auc_full-auc_bdi_only:+.3f}")

print(f"\n→ SCAR HYPOTHESIS:")
if not np.isnan(d_rem) and d_rem > 0.2 and p_rem < 0.10 and abs(r_pre) < 0.3:
    print("   SUPPORTED — lower resting AIS predicts non-remission,")
    print("   independent of baseline severity")
elif not np.isnan(d_rem) and d_rem > 0.2 and p_rem < 0.10:
    print("   PARTIAL — AIS predicts response but confounded with severity")
    print("   Use regression-adjusted result")
else:
    print("   NOT SUPPORTED at current threshold")
    print("   AIS_rest may not capture same construct as AIS_pre (task-evoked)")
print("="*60)
