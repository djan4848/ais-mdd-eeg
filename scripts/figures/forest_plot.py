#!/usr/bin/env python3
# Last updated: 2026-06-15
# Manuscript: AIS_MDD_manuscript_v14
# Repository: https://github.com/djan4848/ais-mdd-eeg
"""
forest_plot.py — Random-effects meta-analytic forest plot of
AIS effect sizes across EEG and MEG datasets (HC > MDD).

Generates: figures/forest_plot.png (300 dpi, 9x7 in)

Usage:
    python scripts/figures/forest_plot.py
    python scripts/figures/forest_plot.py --output path/to/forest.png

Dependencies: numpy, scipy, matplotlib
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

# ── Dataset constants ──────────────────────────────────────────────
# n_hc = healthy controls (HC); consistent with manuscript terminology
DATASETS = [
    {
        # Excluded from pooled estimate: same participants as ds003478 (REST)
        "label":        "Cavanagh PST\n(EEG task, n=109)\n[‡ same cohort as REST]",
        "d":            0.81,
        "n_hc":         86,
        "n_mdd":        23,
        "confirmatory": False,
        "modality":     "PST_same",
    },
    {
        "label":        "ds003478\n(EEG rest, n=113)",
        "d":            0.703,
        "n_hc":         90,
        "n_mdd":        23,
        "confirmatory": True,
        "modality":     "EEG",
    },
    {
        "label":        "TDBRAIN\n(EEG rest, n=168)",
        "d":            0.527,
        "n_hc":         47,
        "n_mdd":        121,
        "confirmatory": True,
        "modality":     "EEG",
    },
    {
        "label":        "MODMA\n(EEG rest, n=53)\n[250 Hz; underpowered]",
        "d":            0.32,
        "n_hc":         29,
        "n_mdd":        24,
        "confirmatory": False,
        "modality":     "EEG",
    },
    {
        "label":        "Cavanagh MEG PST\n(MEG task, n=82)\n[cross-modal; n.s.]",
        "d":            0.37,
        "n_hc":         35,
        "n_mdd":        47,
        "confirmatory": False,
        "modality":     "MEG",
    },
]

# ── Palette ────────────────────────────────────────────────────────
COL_EEG_CONF = '#4C72B0'   # confirmatory EEG (pooled)
COL_PST_SAME = '#89B4D4'   # Cavanagh PST (same cohort as REST; not pooled)
COL_GREY     = '#8E8E8E'   # exploratory EEG
COL_MEG      = '#DD8452'   # MEG cross-modal
COL_POOL     = '#1A3A6B'   # RE pooled diamond
FONT         = 'DejaVu Sans'


# ── Statistical functions ──────────────────────────────────────────
def compute_se(d, n_hc, n_mdd):
    """Standard error of Cohen's d."""
    n = n_hc + n_mdd
    return np.sqrt(n / (n_hc * n_mdd) + d**2 / (2 * n))


def dearsimonian_laird(ds, ses):
    """
    Random-effects pooled estimate via DerSimonian-Laird.
    Returns: pooled_d, pooled_se, I2, Q, p_Q, tau2, weights_pct
    """
    vi  = ses**2
    w   = 1.0 / vi
    d_f = np.sum(w * ds) / np.sum(w)           # fixed-effects estimate
    Q   = np.sum(w * (ds - d_f)**2)
    df  = len(ds) - 1
    c   = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / c)
    I2   = max(0.0, (Q - df) / Q * 100) if Q > df else 0.0
    p_Q  = 1.0 - stats.chi2.cdf(Q, df)
    w_re      = 1.0 / (vi + tau2)
    pooled_d  = np.sum(w_re * ds) / np.sum(w_re)
    pooled_se = np.sqrt(1.0 / np.sum(w_re))
    weights_pct = w_re / np.sum(w_re) * 100
    return pooled_d, pooled_se, I2, Q, p_Q, tau2, weights_pct


# ── Main plot function ─────────────────────────────────────────────
def make_forest_plot(output_path):
    # ── Compute SEs and CIs for every dataset ─────────────────────
    for ds in DATASETS:
        ds['se']    = compute_se(ds['d'], ds['n_hc'], ds['n_mdd'])
        ds['ci_lo'] = ds['d'] - 1.96 * ds['se']
        ds['ci_hi'] = ds['d'] + 1.96 * ds['se']

    # ── Random-effects meta-analysis (confirmatory EEG only) ──────
    conf    = [ds for ds in DATASETS if ds['confirmatory']]
    d_arr   = np.array([ds['d']  for ds in conf])
    se_arr  = np.array([ds['se'] for ds in conf])

    (pooled_d, pooled_se, I2, Q, p_Q,
     tau2, weight_pct) = dearsimonian_laird(d_arr, se_arr)

    pooled_ci_lo = pooled_d - 1.96 * pooled_se
    pooled_ci_hi = pooled_d + 1.96 * pooled_se

    # ── Layout constants ──────────────────────────────────────────
    N_DS     = len(DATASETS)
    Y_TOP    = N_DS - 1          # 4  — top dataset row
    Y_SEP    = -0.60             # separator below datasets
    DY       = -1.60             # diamond centre
    D_H      = 0.30              # diamond half-height
    I2_Y     = DY - D_H - 0.22  # I² label (below diamond)
    FOOT_Y   = I2_Y - 0.72      # footnote (below I²)
    POOL_TXT_Y = (Y_SEP + DY + D_H) / 2   # pooled CI text between sep. and diamond

    XLIM_L = -0.60
    XLIM_R =  1.65

    # ── Figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(XLIM_L, XLIM_R)
    ax.set_ylim(FOOT_Y - 0.50, Y_TOP + 0.80)

    # ── Reference lines ───────────────────────────────────────────
    ax.axvline(0,        color='black',       lw=0.8, zorder=1)
    ax.axvline(pooled_d, color=COL_EEG_CONF,  lw=0.8, ls='--', alpha=0.55, zorder=1)

    # ── Column header ─────────────────────────────────────────────
    ax.text(XLIM_R - 0.02, Y_TOP + 0.65,
            "d [95% CI]   weight",
            ha='right', va='center', fontsize=8,
            fontfamily=FONT, color='#333333', fontweight='bold')
    ax.axhline(Y_TOP + 0.45, color='#999999', lw=0.5, xmin=0.60, xmax=0.98)

    # ── Dataset rows ──────────────────────────────────────────────
    y_pos    = list(range(N_DS - 1, -1, -1))   # [4, 3, 2, 1, 0]
    conf_idx = 0

    for ds, yp in zip(DATASETS, y_pos):
        d, ci_lo, ci_hi = ds['d'], ds['ci_lo'], ds['ci_hi']

        if ds['confirmatory'] and ds['modality'] == 'EEG':
            col, ls = COL_EEG_CONF, '-'
            w_str = f"{weight_pct[conf_idx]:.1f}%"
            conf_idx += 1
        elif ds['modality'] == 'PST_same':
            col, ls, w_str = COL_PST_SAME, '-', "—"
        elif ds['modality'] == 'MEG':
            col, ls, w_str = COL_MEG, '--', "—"
        else:
            col, ls, w_str = COL_GREY, '--', "—"

        # CI line + marker
        ax.plot([ci_lo, ci_hi], [yp, yp], color=col, lw=1.5, ls=ls, zorder=2)
        ax.plot(d, yp, 'o', color=col, ms=8, zorder=3)

        # Left label
        ax.text(XLIM_L + 0.02, yp, ds['label'],
                ha='left', va='center', fontsize=8,
                fontfamily=FONT, color='black')

        # Right column — right-aligned, fixed x to avoid clipping
        right_text = f"d = {d:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]  {w_str}"
        ax.text(XLIM_R - 0.02, yp, right_text,
                ha='right', va='center', fontsize=7.5,
                fontfamily=FONT, color='black')

    # ── Separator ─────────────────────────────────────────────────
    ax.axhline(Y_SEP, color='#888888', lw=0.6, xmin=0.26, xmax=0.96)

    # ── Diamond ───────────────────────────────────────────────────
    diam_x = [pooled_ci_lo, pooled_d, pooled_ci_hi, pooled_d, pooled_ci_lo]
    diam_y = [DY, DY + D_H, DY, DY - D_H, DY]
    ax.fill(diam_x, diam_y, color=COL_POOL, zorder=4)
    ax.plot(diam_x, diam_y, color=COL_POOL, lw=1.0, zorder=5)

    # ── Pooled estimate text — between separator and diamond ──────
    pool_text = (f"RE pooled (2 independent datasets): d = {pooled_d:.2f} "
                 f"[{pooled_ci_lo:.2f}, {pooled_ci_hi:.2f}]")
    ax.text(XLIM_R - 0.02, POOL_TXT_Y, pool_text,
            ha='right', va='center', fontsize=7.5,
            fontfamily=FONT, color=COL_POOL, style='italic')

    # ── I² / Q / p annotation — below diamond ─────────────────────
    ax.text(pooled_d, I2_Y,
            f"I² = {I2:.0f}%    Q = {Q:.2f}    p = {p_Q:.2f}",
            ha='center', va='top', fontsize=8,
            fontfamily=FONT, color=COL_POOL)

    # ── Footnote — below I² ───────────────────────────────────────
    footnote = (
        "‡ Cavanagh PST and REST involve the same participants (N=109); PST shown for task vs. rest\n"
        "comparison only — excluded from pooled estimate. RE pooled based on 2 independent EEG datasets\n"
        "(ds003478 and TDBRAIN). MODMA (250 Hz) and MEG PST (cross-modal) shown for completeness."
    )
    ax.text(XLIM_L + 0.05, FOOT_Y,
            footnote, ha='left', va='top',
            fontsize=7, fontfamily=FONT, color='#555555', style='italic')

    # ── Legend — upper left, above dataset rows ───────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COL_EEG_CONF, ms=8, label='Confirmatory EEG (pooled)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COL_PST_SAME, ms=8,
               label='Cavanagh PST (‡ same cohort as REST; not pooled)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COL_GREY, ms=8, label='Exploratory EEG (underpowered)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COL_MEG, ms=8, label='MEG (cross-modal; n.s.)'),
        mpatches.Patch(facecolor=COL_POOL, label='RE pooled estimate (2 independent datasets)'),
    ]
    ax.legend(handles=legend_elements,
              loc='upper left',
              bbox_to_anchor=(1.02, 1.0),
              fontsize=8,
              frameon=True,
              framealpha=0.95,
              edgecolor='#CCCCCC',
              borderpad=0.8)

    # ── Axes cosmetics ────────────────────────────────────────────
    ax.set_xlabel("Cohen's d  (positive = HC > MDD)", fontsize=10, fontfamily=FONT)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────
    fig.subplots_adjust(right=0.72)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {output_path}')


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generate forest plot (Figure 2)')
    parser.add_argument('--output', default='figures/forest_plot.png',
                        help='Output path (default: figures/forest_plot.png)')
    args = parser.parse_args()
    make_forest_plot(args.output)


if __name__ == '__main__':
    main()
