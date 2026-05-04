#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dds_base.io.paths import (
    DERIV_ROOT,
    hayling_epo_files,
    ROIS,
    EXCLUDE_SUBJECTS,
)

# =========================================================
# SETTINGS
# =========================================================
DDS_PATH = DERIV_ROOT / "dds_peak_aligned_n450" / "dds_n450_results.csv"
OUTDIR = DERIV_ROOT / "figures_paper"
OUTDIR.mkdir(exist_ok=True, parents=True)

TARGET_SUBJECT = "P49"   # cámbialo si quieres
TARGET_ROI = "cacc"      # "frontal", "cacc", "lh", "rh"
FALLBACK_TO_BEST_GLOBAL = True

PARAMS = ["A1", "gamma1", "f1", "phi1", "A2", "gamma2", "f2", "phi2"]

COND_INV_MAP = {
    "INIT": "ASOC",
    "INHIB": "NOASOC",
}

# =========================================================
# LOAD DDS TABLE
# =========================================================
df = pd.read_csv(DDS_PATH)
df = df[df["component"] == "N450"].copy()
df = df[df["subject"].isin([f.parent.name for f in hayling_epo_files()
                            if f.parent.name not in EXCLUDE_SUBJECTS])].copy()

df_all = df.copy()
df_r2pos = df[df["r2"] > 0].copy()

n_subjects = df_all["subject"].nunique()
n_trials = len(df_all)
pct_valid = 100 * len(df_r2pos) / len(df_all)

# =========================================================
# DDS MODEL WITH PHASES
# =========================================================
def dds_reconstruct(t, p):
    s1 = p.A1 * np.exp(-p.gamma1 * t) * np.sin(2 * np.pi * p.f1 * t + p.phi1)
    s2 = p.A2 * np.exp(-p.gamma2 * t) * np.sin(2 * np.pi * p.f2 * t + p.phi2)
    return s1 + s2

# =========================================================
# FIND REPRESENTATIVE REAL TRIAL
# =========================================================
cand = df_r2pos[(df_r2pos["subject"] == TARGET_SUBJECT) & (df_r2pos["roi"] == TARGET_ROI)].copy()

if cand.empty and FALLBACK_TO_BEST_GLOBAL:
    print(f"[warn] No rows found for subject={TARGET_SUBJECT}, roi={TARGET_ROI}. Falling back to best global row in ROI.")
    cand = df_r2pos[df_r2pos["roi"] == TARGET_ROI].copy()

if cand.empty:
    raise RuntimeError(f"No DDS rows found for ROI={TARGET_ROI}")

best_idx = cand["r2"].idxmax()
best_p = df_r2pos.loc[best_idx]

print("[ok] Representative trial selected:")
print(best_p[["subject", "cond", "trial", "roi", "r2", "window_tmin_ms", "window_tmax_ms"]])

# =========================================================
# LOAD ORIGINAL EPOCH FILE FOR THAT SUBJECT
# =========================================================
file_map = {f.parent.name: f for f in hayling_epo_files() if f.parent.name not in EXCLUDE_SUBJECTS}
if best_p.subject not in file_map:
    raise FileNotFoundError(f"No epoch file found for subject {best_p.subject}")

epo_file = file_map[best_p.subject]
epochs = mne.read_epochs(epo_file, preload=True, verbose="ERROR")

raw_cond = COND_INV_MAP[best_p.cond]
if raw_cond not in epochs.event_id:
    raise KeyError(f"Condition {raw_cond} not found in epochs for subject {best_p.subject}")

ep_cond = epochs[raw_cond]
if int(best_p.trial) >= len(ep_cond):
    raise IndexError(f"Trial {best_p.trial} out of range for subject {best_p.subject}, condition {best_p.cond}")

roi_channels = [ch for ch in ROIS[best_p.roi] if ch in ep_cond.ch_names]
if not roi_channels:
    raise RuntimeError(f"No channels available for ROI {best_p.roi} in subject {best_p.subject}")

trial_data = ep_cond[int(best_p.trial)].copy().pick(roi_channels).get_data().mean(axis=1).squeeze()
times = ep_cond.times

tmin = best_p.window_tmin_ms / 1000.0
tmax = best_p.window_tmax_ms / 1000.0
mask = (times >= tmin) & (times <= tmax)

if mask.sum() < 10:
    raise RuntimeError("Too few samples in selected window")

t_abs = times[mask]
t_model = t_abs - tmin
y_raw = trial_data[mask]
y_fit = dds_reconstruct(t_model, best_p)

y_raw_uv = y_raw * 1e6
y_fit_uv = y_fit * 1e6
t_ms = t_model * 1000.0

# =========================================================
# PANEL B DATA: R² summary
# =========================================================
r2_all = df_all["r2"].dropna().values
r2_pos = df_r2pos["r2"].dropna().values

def q25(x): return np.quantile(x, 0.25)
def q75(x): return np.quantile(x, 0.75)

summary_rows = [
    {
        "subset": "All fits",
        "n": len(r2_all),
        "mean": np.mean(r2_all),
        "median": np.median(r2_all),
        "q25": q25(r2_all),
        "q75": q75(r2_all),
    },
    {
        "subset": "R² > 0",
        "n": len(r2_pos),
        "mean": np.mean(r2_pos),
        "median": np.median(r2_pos),
        "q25": q25(r2_pos),
        "q75": q75(r2_pos),
    }
]
r2_summary = pd.DataFrame(summary_rows)

# =========================================================
# PANEL C DATA: PCA on valid fits only
# =========================================================
X = df_r2pos[PARAMS].dropna().copy()
Xz = StandardScaler().fit_transform(X)
pca = PCA()
pca.fit(Xz)

evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)

n80 = int(np.argmax(cum >= 0.80) + 1)
n90 = int(np.argmax(cum >= 0.90) + 1)
n95 = int(np.argmax(cum >= 0.95) + 1)

# =========================================================
# FIGURE
# =========================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

fig = plt.figure(figsize=(20, 6.8))

# ---------------------------------------------------------
# Panel A: Representative Trial Fit
# ---------------------------------------------------------
ax_a = plt.subplot(1, 3, 1)
ax_a.plot(
    t_ms, y_raw_uv,
    color="#bfc5cc",
    linewidth=1.3,
    label="Single-trial ROI signal"
)
ax_a.plot(
    t_ms, y_fit_uv,
    color="#1f2d3a",
    linewidth=2.8,
    label=f"DDS fit ($R^2$ = {best_p.r2:.2f})"
)

ax_a.set_title("A. Representative DDS Fit", fontweight="bold", pad=14)
ax_a.set_xlabel("Time from window start (ms)")
ax_a.set_ylabel("Amplitude (µV)")

subtitle = f"{best_p.subject} | {best_p.cond} | ROI: {best_p.roi.upper()} | trial {int(best_p.trial)}"
ax_a.text(
    0.03, 0.95,
    subtitle,
    transform=ax_a.transAxes,
    ha="left",
    va="top",
    fontsize=9.5,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8", pad=2)
)
ax_a.legend(frameon=False, loc="lower right")

# ---------------------------------------------------------
# Panel B: Fit quality summary
# ---------------------------------------------------------
ax_b = plt.subplot(1, 3, 2)

positions = [1, 2]
box_data = [r2_all, r2_pos]

bp = ax_b.boxplot(
    box_data,
    positions=positions,
    widths=0.5,
    patch_artist=True,
    showfliers=False
)

box_colors = ["#c7ced6", "#5b7fa3"]
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

for whisker in bp["whiskers"]:
    whisker.set_color("0.4")
for cap in bp["caps"]:
    cap.set_color("0.4")

ax_b.set_xticks(positions)
ax_b.set_xticklabels(["All fits", "R² > 0"])
ax_b.set_ylabel("DDS goodness-of-fit ($R^2$)")
ax_b.set_title("B. DDS Fit Quality", fontweight="bold", pad=14)
ax_b.axhline(0, color="0.6", linestyle="--", linewidth=1)

for i, row in enumerate(summary_rows, start=1):
    txt = (
        f"n = {row['n']}\n"
        f"mean = {row['mean']:.2f}\n"
        f"median = {row['median']:.2f}\n"
        f"IQR = [{row['q25']:.2f}, {row['q75']:.2f}]"
    )
    ax_b.text(
        i, ax_b.get_ylim()[1] * 0.93,
        txt,
        ha="center", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.85", pad=2)
    )

# ---------------------------------------------------------
# Panel C: PCA cumulative explained variance
# ---------------------------------------------------------
ax_c = plt.subplot(1, 3, 3)

components = np.arange(1, len(cum) + 1)

ax_c.plot(
    components, cum,
    marker="o",
    color="#264653",
    linewidth=2.4
)

ax_c.axhline(0.80, color="#2a9d8f", linestyle="--", linewidth=1.2)
ax_c.axhline(0.90, color="#e9c46a", linestyle="--", linewidth=1.2)
ax_c.axhline(0.95, color="#e76f51", linestyle="--", linewidth=1.2)

ax_c.axvline(n80, color="#2a9d8f", linestyle=":", linewidth=1.2)
ax_c.axvline(n90, color="#e9c46a", linestyle=":", linewidth=1.2)
ax_c.axvline(n95, color="#e76f51", linestyle=":", linewidth=1.2)

ax_c.set_xlim(1, len(cum))
ax_c.set_ylim(0, 1.02)
ax_c.set_xticks(components)
ax_c.set_xlabel("Number of principal components")
ax_c.set_ylabel("Cumulative explained variance")
ax_c.set_title("C. DDS Parameter Space Complexity", fontweight="bold", pad=14)

ax_c.text(
    0.05, 0.22,
    f"80% variance: {n80} PCs\n"
    f"90% variance: {n90} PCs\n"
    f"95% variance: {n95} PCs",
    transform=ax_c.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.85", pad=3)
)

# ---------------------------------------------------------
# Footnote
# ---------------------------------------------------------
fig.text(
    0.5, 0.02,
    (
        f"DDS evidence summary: N = {n_subjects} subjects | "
        f"n = {n_trials} trial-ROI fits | "
        f"valid fits (R² > 0) = {pct_valid:.1f}% | "
        f"PCA computed on valid fits only"
    ),
    ha="center",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7")
)

for ax in [ax_a, ax_b, ax_c]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.subplots_adjust(bottom=0.18, wspace=0.32)
outfile = OUTDIR / "Figure_2_DDS_Evidence.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()

print(f"[ok] Figure 2 saved: {outfile}")
