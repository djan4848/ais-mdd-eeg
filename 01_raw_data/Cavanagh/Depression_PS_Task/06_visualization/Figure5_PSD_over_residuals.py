#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT = Path(".")
INFILE = ROOT / "derivatives" / "trial_roi_timeseries_residual_r2pos" / "trial_roi_timeseries_residual_r2pos.csv"
OUTDIR = ROOT / "derivatives" / "figures_paper"
OUTDIR.mkdir(parents=True, exist_ok=True)

FS = 250.0  # 4 ms per sample
ROIS_TO_PLOT = ["frontal", "cacc"]
MAX_FREQ = 30.0

df = pd.read_csv(INFILE)

required = {"subject", "cond", "trial_uid", "roi", "time_ms", "value", "residual"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns in input file: {missing}")

fig, axes = plt.subplots(1, len(ROIS_TO_PLOT), figsize=(12, 4.5), sharey=True)
if len(ROIS_TO_PLOT) == 1:
    axes = [axes]

for ax, roi in zip(axes, ROIS_TO_PLOT):
    sub = df[df["roi"] == roi].copy()
    psd_orig = []
    psd_resid = []
    f_ref = None

    for _, g in sub.groupby(["subject", "cond", "trial_uid"], sort=False):
        g = g.sort_values("time_ms")
        x = g["value"].to_numpy(dtype=float)
        r = g["residual"].to_numpy(dtype=float)

        if len(x) < 32:
            continue

        f, pxx = welch(x, fs=FS, nperseg=min(len(x), 128))
        _, prr = welch(r, fs=FS, nperseg=min(len(r), 128))

        keep = f <= MAX_FREQ
        f = f[keep]
        pxx = pxx[keep]
        prr = prr[keep]

        psd_orig.append(pxx)
        psd_resid.append(prr)
        f_ref = f

    psd_orig = np.array(psd_orig)
    psd_resid = np.array(psd_resid)

    mean_orig = psd_orig.mean(axis=0)
    mean_resid = psd_resid.mean(axis=0)
    sem_orig = psd_orig.std(axis=0, ddof=1) / np.sqrt(len(psd_orig))
    sem_resid = psd_resid.std(axis=0, ddof=1) / np.sqrt(len(psd_resid))

    ax.plot(f_ref, mean_orig, linewidth=2, label="Original ERP segment")
    ax.fill_between(f_ref, mean_orig - sem_orig, mean_orig + sem_orig, alpha=0.2)

    ax.plot(f_ref, mean_resid, linewidth=2, label="DDS residual")
    ax.fill_between(f_ref, mean_resid - sem_resid, mean_resid + sem_resid, alpha=0.2)

    ax.set_title(f"{roi.upper()} ROI")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(0, MAX_FREQ)
    ax.set_ylabel("Power spectral density")

axes[0].legend(frameon=False)
fig.suptitle("Supplementary Figure S5. Spectral characterization of original ERP segments and DDS residuals", fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])

outfile = OUTDIR / "Supplementary_Figure_S5_Residual_PSD.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.show()

print(f"[ok] Saved: {outfile}")
