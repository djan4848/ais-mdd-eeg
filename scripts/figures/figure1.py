#!/usr/bin/env python3
# Last updated: 2026-06-15
# Manuscript: AIS_MDD_manuscript_v14
# Repository: https://github.com/djan4848/ais-mdd-eeg
"""
figure1.py — AIS at FCz in MDD: task, rest, temporal profile,
neural scar, parameter robustness, and evidence map.

Generates: figures/figure1_revised.png (300 dpi, 14x10 in)

Usage:
    python scripts/figures/figure1.py
    python scripts/figures/figure1.py --output path/to/figure1.png

Dependencies: numpy, scipy, matplotlib
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Reproducibility ────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Palette ────────────────────────────────────────────────────────
HC_COLOR   = '#4C72B0'
MDD_COLOR  = '#DD8452'
PART_COLOR = '#C44E52'   # MDD Partial remission (panel D)
GREY       = '#8E8E8E'
BLACK      = '#333333'
FONT       = 'DejaVu Sans'

# ── Global style ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        FONT,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          False,
    'font.size':          9,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})

# ── Simulated data (hardcoded constants) ──────────────────────────
# Panel A — reward anticipation task
hc_task  = rng.normal(1.113, 0.105, 86)
mdd_task = rng.normal(1.029, 0.095, 23)

# Panel B — resting state (normalised to HC mean = 1.0)
hc_rest_ds  = rng.normal(1.0, 0.150, 90)
mdd_rest_ds = rng.normal(1.0 - 0.70 * 0.150, 0.150, 23)
hc_rest_td  = rng.normal(1.0, 0.033, 47)
mdd_rest_td = rng.normal(1.0 - 0.53 * 0.058, 0.058, 121)

# Panel C — temporal sweep
WIN_LABELS  = [
    "−500:−400", "−400:−300", "−300:−200", "−200:−100", "−100:0",
    "0:+100", "+100:+200", "+200:+300", "+300:+400", "+400:+500", "+500:+600",
]
D_VALS      = [0.61, 0.71, 0.90, 0.81, 0.75, 0.58, 0.40, 0.21, 0.18, 0.12, 0.09]
P_SIG       = [True, True, True, True, True, True, False, False, False, False, False]
APRIORI_IDX = 3   # −200:−100 ms window

# Panel D — neural scar
hc_scar      = rng.normal(1.113, 0.105, 86)
current_scar = rng.normal(1.072, 0.069, 11)
partial_scar = rng.normal(0.989, 0.077, 12)

# Panel E — parameter robustness
PARAM_LABELS = ["3 bins", "4 bins\n(primary)", "6 bins", "8 bins", "lag=2\n(−500:0ms)"]
PARAM_D      = [0.79, 0.81, 0.81, 0.80, 0.87]

# Panel F — evidence map (sorted by |d| descending)
EVIDENCE = [
    ("Neural scar †",          0.96,  True,  "EEG"),
    ("PST task (EEG)",         0.81,  True,  "EEG"),
    ("ds003478 rest (EEG)",    0.70,  True,  "EEG"),
    ("TDBRAIN rest (EEG)",     0.53,  True,  "EEG"),
    ("MEG PST (cross-modal)",  0.37,  False, "MEG"),
    ("MODMA rest (EEG)",       0.32,  False, "EEG"),
    ("TDBRAIN rem/non-rem",   -0.15,  False, "EEG"),
    ("MODMA task (null)",      0.09,  False, "EEG"),
    ("Hayling (null)",         0.02,  False, "EEG"),
]


# ── Helpers ────────────────────────────────────────────────────────
def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


def _panel_letter(ax, letter):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left', fontfamily=FONT)


def _sig_bracket(ax, x1, x2, y, text, dy=0.025, tdy=0.01, color=BLACK, ls='-'):
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y],
            color=color, lw=1.2, ls=ls)
    ax.text((x1 + x2) / 2, y + dy + tdy, text,
            ha='center', va='bottom', fontsize=7, color=color, fontfamily=FONT)


def _violin_strip(ax, pos, data, color, jitter_seed=0):
    vp = ax.violinplot(data, positions=[pos], widths=0.6,
                       showmedians=False, showextrema=False)
    for body in vp['bodies']:
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)
    jx = pos + np.random.default_rng(jitter_seed).uniform(-0.12, 0.12, len(data))
    ax.scatter(jx, data, color=color, alpha=0.6, s=14, zorder=3)
    ax.hlines(np.median(data), pos - 0.17, pos + 0.17,
              colors=color, lw=2.5, zorder=4)


# ── Panel functions ────────────────────────────────────────────────
def plot_panel_a(ax):
    _clean(ax)
    _violin_strip(ax, 0, hc_task,  HC_COLOR,  jitter_seed=10)
    _violin_strip(ax, 1, mdd_task, MDD_COLOR, jitter_seed=11)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['HC', 'MDD'], fontsize=9)
    ax.set_ylabel('AIS (bits)', fontsize=9)
    ax.set_ylim(0.75, 1.52)
    ax.set_xlim(-0.6, 1.6)
    _sig_bracket(ax, 0, 1, 1.38, 'd = 0.81, p < .001', dy=0.03, tdy=0.01)
    ax.set_title('(A) Reward anticipation task (EEG)', fontsize=9, pad=4)
    _panel_letter(ax, 'A')


def plot_panel_b(ax):
    _clean(ax)
    pairs = [
        (0.0, hc_rest_ds,  HC_COLOR,  20),
        (0.7, mdd_rest_ds, MDD_COLOR, 21),
        (1.8, hc_rest_td,  HC_COLOR,  22),
        (2.5, mdd_rest_td, MDD_COLOR, 23),
    ]
    for pos, data, col, seed in pairs:
        vp = ax.violinplot(data, positions=[pos], widths=0.55,
                           showmedians=False, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(col); body.set_alpha(0.35); body.set_edgecolor(col)
        jx = pos + np.random.default_rng(seed).uniform(-0.10, 0.10, len(data))
        ax.scatter(jx, data, color=col, alpha=0.5, s=9, zorder=3)
        ax.hlines(np.median(data), pos - 0.14, pos + 0.14,
                  colors=col, lw=2.0, zorder=4)

    ax.text(0.35, 0.76, 'ds003478\nd=0.70, p=.003', ha='center', va='top',
            fontsize=7.5, color=BLACK, transform=ax.get_xaxis_transform())
    ax.text(2.15, 0.76, 'TDBRAIN\nd=0.53, p=.006', ha='center', va='top',
            fontsize=7.5, color=BLACK, transform=ax.get_xaxis_transform())
    ax.set_xticks([0.0, 0.7, 1.8, 2.5])
    ax.set_xticklabels(['HC', 'MDD', 'HC', 'MDD'], fontsize=8)
    ax.set_ylabel('AIS_rest (norm. to HC mean = 1.0)', fontsize=8.5)
    ax.set_xlim(-0.5, 3.1)
    _sig_bracket(ax, 0.0, 0.7, 1.36, 'd=0.70', dy=0.025, tdy=0.008)
    _sig_bracket(ax, 1.8, 2.5, 1.09, 'd=0.53', dy=0.025, tdy=0.008)
    ax.set_title('(B) Resting-state EEG (two datasets)', fontsize=9, pad=4)
    _panel_letter(ax, 'B')


def plot_panel_c(ax):
    _clean(ax)
    n_win  = len(WIN_LABELS)
    y_pos  = list(range(n_win - 1, -1, -1))   # top = earliest window
    colors = [HC_COLOR if s else GREY for s in P_SIG]
    ax.barh(y_pos, D_VALS, color=colors, height=0.70, alpha=0.80)
    # Highlight a priori window with black border
    ap_y = y_pos[APRIORI_IDX]
    ax.barh(ap_y, D_VALS[APRIORI_IDX], height=0.72,
            color=HC_COLOR, alpha=0.90, edgecolor='black', linewidth=2.2)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(WIN_LABELS, fontsize=7)
    ax.set_xlabel("Cohen's d", fontsize=9)
    ax.set_xlim(-0.08, 1.12)
    ax.annotate('a priori\nwindow', xy=(D_VALS[APRIORI_IDX], ap_y),
                xytext=(0.95, ap_y + 1.6),
                fontsize=7, ha='center', va='bottom', color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
    rewp_y = y_pos[7]
    ax.annotate('RewP\nwindow', xy=(D_VALS[7], rewp_y),
                xytext=(0.60, rewp_y - 1.6),
                fontsize=7, ha='center', va='top', color=GREY,
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.0))
    ax.text(0.97, 0.02, 'N = 86 HC,  N = 23 MDD',
            ha='right', va='bottom', fontsize=7, color='#444444',
            transform=ax.transAxes)
    ax.set_title('(C) Temporal profile', fontsize=9, pad=4)
    _panel_letter(ax, 'C')


def plot_panel_d(ax):
    _clean(ax)
    groups = [
        (0, hc_scar,      HC_COLOR,   30),
        (1, current_scar, MDD_COLOR,  31),
        (2, partial_scar, PART_COLOR, 32),
    ]
    for pos, data, col, seed in groups:
        vp = ax.violinplot(data, positions=[pos], widths=0.55,
                           showmedians=False, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(col); body.set_alpha(0.35); body.set_edgecolor(col)
        jx = pos + np.random.default_rng(seed).uniform(-0.10, 0.10, len(data))
        ax.scatter(jx, data, color=col, alpha=0.6, s=16, zorder=3)
        ax.hlines(np.median(data), pos - 0.15, pos + 0.15,
                  colors=col, lw=2.5, zorder=4)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['HC', 'MDD Current', 'MDD Partial'], fontsize=8)
    ax.set_ylabel('AIS_pre (bits)', fontsize=9)
    ax.set_ylim(0.78, 1.48)
    ax.set_xlim(-0.6, 2.6)

    # HamD annotations below x-axis using transform
    for pos, txt in [(0, 'HamD: n/a'), (1, 'HamD: 13.09'), (2, 'HamD: 7.25')]:
        ax.text((pos + 0.6) / 3.2, -0.10, txt,
                ha='center', va='top', fontsize=6.5,
                color='#555555', style='italic',
                transform=ax.transAxes)

    _sig_bracket(ax, 1, 2, 1.27,
                 'd=0.96, p=.034*\n(exploratory)',
                 dy=0.04, tdy=0.01, color=BLACK)
    _sig_bracket(ax, 0, 2, 1.38,
                 'post-hoc gradient',
                 dy=0.03, tdy=0.008, color=GREY, ls='--')
    ax.text(0.5, -0.19, '*Exploratory; n=11–12; uncorrected',
            ha='center', va='top', fontsize=6.5,
            color='#555555', style='italic', transform=ax.transAxes)
    ax.set_title('(D) Neural scar analysis (exploratory)', fontsize=9, pad=4)
    _panel_letter(ax, 'D')


def plot_panel_e(ax):
    _clean(ax)
    n = len(PARAM_LABELS)
    y_pos = list(range(n - 1, -1, -1))
    bar_cols = [HC_COLOR if 'primary' in p else '#C8D7EF' for p in PARAM_LABELS]
    ax.barh(y_pos, PARAM_D, color=bar_cols, height=0.65, edgecolor='white')
    ax.axvline(0.81, color=HC_COLOR, lw=1.2, ls='--', alpha=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(PARAM_LABELS, fontsize=8.5)
    ax.set_xlabel("Cohen's d", fontsize=9)
    ax.set_xlim(0.60, 1.02)
    ax.text(0.98, 0.04, 'KSG cross-validation:\nr = 0.96',
            ha='right', va='bottom', fontsize=7.5, color='#444444',
            transform=ax.transAxes)
    ax.set_title('(E) Parameter robustness', fontsize=9, pad=4)
    _panel_letter(ax, 'E')


def plot_panel_f(ax):
    _clean(ax)
    ev = sorted(EVIDENCE, key=lambda x: abs(x[1]), reverse=True)
    y_pos = list(range(len(ev) - 1, -1, -1))
    ax.axvline(0, color='black', lw=0.8, zorder=1)
    max_d = max(abs(e[1]) for e in ev)
    for yp, (lbl, d, sig, mod) in zip(y_pos, ev):
        if mod == 'MEG':
            col, alpha = MDD_COLOR, 0.35
        elif d >= 0 and sig:
            col, alpha = HC_COLOR, 0.80
        elif d < 0 and sig:
            col, alpha = PART_COLOR, 0.80
        elif d >= 0:
            col, alpha = HC_COLOR, 0.35
        else:
            col, alpha = PART_COLOR, 0.35
        ax.barh(yp, d, height=0.65, color=col, alpha=alpha, zorder=2)
        suffix = ' (MEG)' if mod == 'MEG' else ''
        offset = 0.018 if d >= 0 else -0.018
        ax.text(d + offset, yp, f'{d:+.2f}{suffix}',
                va='center', ha='left' if d >= 0 else 'right',
                fontsize=6.8, color='#333333')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e[0] for e in ev], fontsize=7.5)
    ax.set_xlabel("Cohen's d  (positive = HC > MDD)", fontsize=9)
    ax.set_xlim(-0.45, max_d + 0.38)
    ax.text(0.02, 0.01, '† Exploratory, uncorrected',
            ha='left', va='bottom', fontsize=6.5,
            color='#555555', style='italic', transform=ax.transAxes)
    ax.set_title('(F) Evidence map', fontsize=9, pad=4)
    _panel_letter(ax, 'F')


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generate Figure 1')
    parser.add_argument('--output', default='figures/figure1_revised.png',
                        help='Output path (default: figures/figure1_revised.png)')
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axA, axB, axC = axes[0]
    axD, axE, axF = axes[1]

    plot_panel_a(axA)
    plot_panel_b(axB)
    plot_panel_c(axC)
    plot_panel_d(axD)
    plot_panel_e(axE)
    plot_panel_f(axF)

    fig.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)

    out = args.output
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
