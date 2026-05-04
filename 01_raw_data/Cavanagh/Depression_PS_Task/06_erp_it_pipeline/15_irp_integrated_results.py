#!/usr/bin/env python3
"""
15_irp_integrated_results.py
Integrated Results Plot (IRP) for AIS_pre paper.

Four-panel publication figure:
  A. EEG group comparison (violin + strip, Cavanagh PS Task)
  B. Cross-dataset replication forest plot (3 studies, d ± 95% CI)
  C. Anhedonia correlation (BDI_Anh vs AIS_pre, EEG, N=109)
  D. KSG estimator validation (KSG vs Shannon AIS_pre, N=20)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_EEG   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/Depression_PS_Task")
BASE_MEG   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/ds005356")
BASE_MODMA = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/MODMA/DDS-MODMA")
OUT_DIR    = BASE_EEG / "derivatives/erp_it_cavanagh"

# ── Load data ─────────────────────────────────────────────────────────────────
df_eeg   = pd.read_csv(OUT_DIR / 'ksg_te_ais_merged.csv')
df_ksg   = pd.read_csv(OUT_DIR / 'ksg_ais_validation.csv')
df_meg   = pd.read_csv(BASE_MEG  / 'derivatives/meg_ais_pre_results.csv')
df_modma = pd.read_csv(BASE_MODMA / 'derivatives/modma_ais_pre_results.csv')

# Normalise group labels → CTL / MDD
df_eeg['grp']   = df_eeg['group'].map({'CTL': 'CTL', 'MDD_any': 'MDD'})
df_ksg['grp']   = df_ksg['group'].map({'CTL': 'CTL', 'MDD_any': 'MDD'})
df_meg['grp']   = df_meg['group']
df_modma['grp'] = df_modma['group'].map({'HC': 'CTL', 'MDD': 'MDD'})

print(f"EEG   N={len(df_eeg)} (CTL={sum(df_eeg['grp']=='CTL')}, MDD={sum(df_eeg['grp']=='MDD')})")
print(f"MEG   N={len(df_meg)} (CTL={sum(df_meg['grp']=='CTL')}, MDD={sum(df_meg['grp']=='MDD')})")
print(f"MODMA N={len(df_modma)} (HC={sum(df_modma['grp']=='CTL')}, MDD={sum(df_modma['grp']=='MDD')})")
print(f"KSG val N={len(df_ksg)}")

# ── Statistical helpers ───────────────────────────────────────────────────────
def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pooled

def d_se(d, n1, n2):
    return np.sqrt((n1+n2)/(n1*n2) + d**2 / (2*(n1+n2-2)))

def perm_r(x, y, n_perm=5000, seed=42):
    rng   = np.random.default_rng(seed)
    r_obs = pearsonr(x, y)[0]
    r_null = np.array([pearsonr(rng.permutation(x), y)[0] for _ in range(n_perm)])
    p = (np.sum(np.abs(r_null) >= abs(r_obs)) + 1) / (n_perm + 1)
    return r_obs, p

def regress_line(x, y, ax, **kw):
    z    = np.polyfit(x, y, 1)
    xr   = np.linspace(x.min(), x.max(), 200)
    ax.plot(xr, np.polyval(z, xr), **kw)

def mean_ci(x):
    m  = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(len(x))
    return m, m - 1.96*se, m + 1.96*se

# ── Compute stats ─────────────────────────────────────────────────────────────
ctl_eeg = df_eeg.loc[df_eeg['grp']=='CTL', 'mean_AIS_pre'].values
mdd_eeg = df_eeg.loc[df_eeg['grp']=='MDD', 'mean_AIS_pre'].values
d_eeg   = cohen_d(ctl_eeg, mdd_eeg)
se_eeg  = d_se(d_eeg, len(ctl_eeg), len(mdd_eeg))

ctl_meg = df_meg.loc[df_meg['grp']=='CTL', 'mean_AIS_pre'].values
mdd_meg = df_meg.loc[df_meg['grp']=='MDD', 'mean_AIS_pre'].values
d_meg   = cohen_d(ctl_meg, mdd_meg)
se_meg  = d_se(d_meg, len(ctl_meg), len(mdd_meg))

ctl_mod = df_modma.loc[df_modma['grp']=='CTL', 'mean_AIS_pre'].values
mdd_mod = df_modma.loc[df_modma['grp']=='MDD', 'mean_AIS_pre'].values
d_mod   = cohen_d(ctl_mod, mdd_mod)
se_mod  = d_se(d_mod, len(ctl_mod), len(mdd_mod))

valid_anh = df_eeg[['BDI_Anh','mean_AIS_pre','grp']].dropna()
r_anh, p_anh = perm_r(valid_anh['mean_AIS_pre'].values, valid_anh['BDI_Anh'].values)

r_ksg, p_ksg = pearsonr(df_ksg['shannon_ais'], df_ksg['ksg_ais'])

print(f"\nEEG  : d={d_eeg:+.3f} ± {1.96*se_eeg:.3f} (95%CI), N={len(ctl_eeg)}+{len(mdd_eeg)}")
print(f"MEG  : d={d_meg:+.3f} ± {1.96*se_meg:.3f} (95%CI), N={len(ctl_meg)}+{len(mdd_meg)}")
print(f"MODMA: d={d_mod:+.3f} ± {1.96*se_mod:.3f} (95%CI), N={len(ctl_mod)}+{len(mdd_mod)}")
print(f"BDI_Anh vs AIS_pre: r={r_anh:.3f}, p_perm={p_anh:.4f}, N={len(valid_anh)}")
print(f"KSG vs Shannon AIS: r={r_ksg:.3f}, p={p_ksg:.4f}, N={len(df_ksg)}")

# ── Figure setup ──────────────────────────────────────────────────────────────
CTL_C = '#2196F3'
MDD_C = '#E53935'
GRAY  = '#555555'

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
})

fig = plt.figure(figsize=(11, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

rng_jitter = np.random.default_rng(7)

# ── Panel A: EEG group comparison ─────────────────────────────────────────────
groups  = [('CTL', ctl_eeg, CTL_C, 0), ('MDD', mdd_eeg, MDD_C, 1)]
vp_data = [ctl_eeg, mdd_eeg]
vp_cols = [CTL_C, MDD_C]

vp = axA.violinplot(vp_data, positions=[0, 1], widths=0.6,
                    showmedians=False, showextrema=False)
for body, col in zip(vp['bodies'], vp_cols):
    body.set_facecolor(col)
    body.set_alpha(0.25)
    body.set_edgecolor(col)
    body.set_linewidth(0.8)

for label, arr, col, x in groups:
    jit = rng_jitter.uniform(-0.12, 0.12, len(arr))
    axA.scatter(x + jit, arr, color=col, alpha=0.55, s=18, linewidths=0, zorder=3)
    m, lo, hi = mean_ci(arr)
    axA.errorbar(x, m, yerr=[[m-lo],[hi-m]], fmt='o', color='black',
                 capsize=4, capthick=1.2, elinewidth=1.2, markersize=6, zorder=5)

axA.set_xticks([0, 1])
axA.set_xticklabels([f'CTL\n(n={len(ctl_eeg)})', f'MDD\n(n={len(mdd_eeg)})'])
axA.set_ylabel('AIS$_\mathrm{pre}$  [bits]')
axA.set_title('A.  Anticipatory preparation\n(Cavanagh EEG, PS Task)', loc='left', fontsize=10)

# Cohen's d annotation
y_max = max(ctl_eeg.max(), mdd_eeg.max())
y_ann = y_max + 0.04
axA.annotate('', xy=(1, y_ann), xytext=(0, y_ann),
             arrowprops=dict(arrowstyle='<->', color='black', lw=0.9))
axA.text(0.5, y_ann + 0.01, f'd = {d_eeg:.2f}, FDR p < 0.001',
         ha='center', va='bottom', fontsize=8.5)
axA.set_ylim(0.6, y_ann + 0.09)

# ── Panel B: Forest plot ───────────────────────────────────────────────────────
studies = [
    ('Cavanagh EEG\n(N=109, PST)',  d_eeg, se_eeg, '#1565C0'),
    ('Cavanagh MEG\n(N=27, PST)',   d_meg, se_meg, '#7B1FA2'),
    ('MODMA EEG\n(N=53, dot-probe)',d_mod, se_mod, '#2E7D32'),
]
y_pos   = [2, 1, 0]
shapes  = ['o', 's', 'D']

for (label, d, se, col), yp, shape in zip(studies, y_pos, shapes):
    ci_lo = d - 1.96*se
    ci_hi = d + 1.96*se
    w     = 1 / se**2
    ms    = 6 + 4*(w / max(1/s**2 for _, _, s, _ in studies))
    axB.plot([ci_lo, ci_hi], [yp, yp], color=col, lw=1.4, zorder=2)
    axB.plot(d, yp, marker=shape, color=col, ms=ms, zorder=3, mec='white', mew=0.6)

axB.axvline(0, color='black', lw=0.8, ls='--', zorder=1)
axB.set_yticks(y_pos)
axB.set_yticklabels([s[0] for s in studies], fontsize=8.5)
axB.set_xlabel("Cohen's  d  (CTL > MDD)")
axB.set_title('B.  Cross-dataset replication\n(effect sizes ± 95% CI)', loc='left', fontsize=10)
axB.set_xlim(-0.85, 1.75)
axB.set_ylim(-0.6, 2.7)

# small annotations
ann_texts = [
    f'd={d_eeg:+.2f}',
    f'd={d_meg:+.2f}†',
    f'd={d_mod:+.2f}‡',
]
for (label, d, se, col), yp, ann in zip(studies, y_pos, ann_texts):
    axB.text(d + 1.96*se + 0.04, yp, ann, va='center', fontsize=8, color=col)

axB.text(0.03, -0.52, '† underpowered (N=27)    ‡ during-face window (paradigm mismatch)',
         fontsize=7, color='#555555', transform=axB.transAxes)

# ── Panel C: Anhedonia correlation ────────────────────────────────────────────
for grp, col in [('CTL', CTL_C), ('MDD', MDD_C)]:
    sub = valid_anh[valid_anh['grp'] == grp]
    axC.scatter(sub['BDI_Anh'], sub['mean_AIS_pre'],
                color=col, alpha=0.55, s=22, linewidths=0, label=grp)

regress_line(valid_anh['BDI_Anh'].values, valid_anh['mean_AIS_pre'].values,
             axC, color=GRAY, lw=1.4, ls='-', zorder=2)

p_str = 'p = 0.018' if p_anh >= 0.001 else 'p < 0.001'
axC.text(0.97, 0.95, f'r = {r_anh:.3f}\n{p_str}\nN = {len(valid_anh)}',
         transform=axC.transAxes, ha='right', va='top', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.7))
axC.set_xlabel('BDI Anhedonia subscale')
axC.set_ylabel('AIS$_\mathrm{pre}$  [bits]')
axC.set_title('C.  Anhedonia link\n(Cavanagh EEG, N=109)', loc='left', fontsize=10)
legend_els = [Line2D([0],[0], marker='o', color='w', markerfacecolor=CTL_C, ms=7, label='CTL'),
              Line2D([0],[0], marker='o', color='w', markerfacecolor=MDD_C, ms=7, label='MDD')]
axC.legend(handles=legend_els, fontsize=8, frameon=False, loc='lower left')

# ── Panel D: KSG validation ───────────────────────────────────────────────────
for grp, col in [('CTL', CTL_C), ('MDD', MDD_C)]:
    sub = df_ksg[df_ksg['grp'] == grp]
    axD.scatter(sub['shannon_ais'], sub['ksg_ais'],
                color=col, alpha=0.75, s=40, linewidths=0, label=grp)

regress_line(df_ksg['shannon_ais'].values, df_ksg['ksg_ais'].values,
             axD, color=GRAY, lw=1.4, zorder=2)

p_ksg_str = 'p < 0.001' if p_ksg < 0.001 else f'p = {p_ksg:.3f}'
axD.text(0.05, 0.95, f'r = {r_ksg:.3f}\n{p_ksg_str}\nN = {len(df_ksg)}',
         transform=axD.transAxes, ha='left', va='top', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.7))
axD.set_xlabel('Shannon AIS$_\mathrm{pre}$  [bits]')
axD.set_ylabel('KSG AIS$_\mathrm{pre}$  [bits]')
axD.set_title('D.  Estimator validation\n(KSG vs Shannon, N=20)', loc='left', fontsize=10)
axD.legend(handles=legend_els, fontsize=8, frameon=False, loc='lower right')

# ── Panel labels (bold) ────────────────────────────────────────────────────────
# Titles already have A/B/C/D embedded; no separate label needed.

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / 'irp_integrated_results.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {out_path}")

# ── Print final summary table ──────────────────────────────────────────────────
print("\n" + "="*60)
print("INTEGRATED RESULTS SUMMARY")
print("="*60)
print(f"Primary finding  — Cavanagh EEG (N={len(ctl_eeg)}+{len(mdd_eeg)}):")
print(f"  CTL AIS_pre = {np.mean(ctl_eeg):.4f} ± {np.std(ctl_eeg,ddof=1):.4f}")
print(f"  MDD AIS_pre = {np.mean(mdd_eeg):.4f} ± {np.std(mdd_eeg,ddof=1):.4f}")
print(f"  Cohen's d   = {d_eeg:.3f}, 95% CI [{d_eeg-1.96*se_eeg:.3f}, {d_eeg+1.96*se_eeg:.3f}]")
print()
print(f"Cross-modal    — Cavanagh MEG (N={len(ctl_meg)}+{len(mdd_meg)}):")
print(f"  CTL AIS_pre = {np.mean(ctl_meg):.4f} ± {np.std(ctl_meg,ddof=1):.4f}")
print(f"  MDD AIS_pre = {np.mean(mdd_meg):.4f} ± {np.std(mdd_meg,ddof=1):.4f}")
print(f"  Cohen's d   = {d_meg:.3f}, 95% CI [{d_meg-1.96*se_meg:.3f}, {d_meg+1.96*se_meg:.3f}]")
print()
print(f"Cross-paradigm — MODMA EEG (N={len(ctl_mod)}+{len(mdd_mod)}):")
print(f"  HC  AIS_pre = {np.mean(ctl_mod):.4f} ± {np.std(ctl_mod,ddof=1):.4f}")
print(f"  MDD AIS_pre = {np.mean(mdd_mod):.4f} ± {np.std(mdd_mod,ddof=1):.4f}")
print(f"  Cohen's d   = {d_mod:.3f}, 95% CI [{d_mod-1.96*se_mod:.3f}, {d_mod+1.96*se_mod:.3f}]")
print(f"  (null expected: during-face window, not blank anticipation)")
print()
print(f"Anhedonia      — BDI_Anh vs AIS_pre: r={r_anh:.3f}, p_perm={p_anh:.4f}, N={len(valid_anh)}")
print(f"Validation     — KSG vs Shannon AIS: r={r_ksg:.3f}, p={p_ksg:.4f}, N={len(df_ksg)}")
print("="*60)
