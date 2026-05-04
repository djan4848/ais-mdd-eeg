#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_extract_ais_n450_only.py

Calcula Active Information Storage (AIS) trial-by-trial
sobre segmentos peak-aligned del N450.

Decisiones metodológicas:
- solo N450
- detección de pico en ROI frontal de referencia
- ventana ±200 ms alrededor del pico detectado
- misma lógica temporal que el script 03
- AIS expresado en bits (base 2)
"""

import numpy as np
import pandas as pd
import mne
from scipy.stats import entropy

from dds_base.io.paths import (
    DERIV_ROOT,
    hayling_epo_files,
    ROIS,
    EXCLUDE_SUBJECTS,
)

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
OUTDIR = DERIV_ROOT / "ais_n450"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = OUTDIR / "ais_n450_results.csv"
LOG_SKIPPED = OUTDIR / "skipped_subjects_ais_n450.txt"
LOG_WARNINGS = OUTDIR / "ais_n450_warnings.txt"

COND_MAP = {
    "ASOC": "INIT",
    "NOASOC": "INHIB",
}

PEAK_REFERENCE_ROI = "frontal"

N450_SEARCH_WINDOW = (0.390, 0.524)   # segundos
DELTA = 0.200                         # ±200 ms = 400 ms total
NBINS = 8
LAG = 1
MIN_SAMPLES = 20


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def detect_negative_peak(y, t, search_window):
    """Detecta el mínimo dentro de la ventana N450."""
    mask = (t >= search_window[0]) & (t <= search_window[1])
    if not np.any(mask):
        return None, None

    y_seg = y[mask]
    t_seg = t[mask]
    if len(y_seg) == 0:
        return None, None

    idx = int(np.argmin(y_seg))
    return float(t_seg[idx]), float(y_seg[idx])


def safe_qcut(x, q):
    """
    Discretización robusta por cuantiles.
    Devuelve enteros 0..k-1 o None si no se puede discretizar.
    """
    try:
        x_disc = pd.qcut(x, q=q, labels=False, duplicates="drop")
    except ValueError:
        return None

    if x_disc is None:
        return None

    x_disc = np.asarray(x_disc, dtype=float)

    if np.all(np.isnan(x_disc)):
        return None

    # Convertir a enteros válidos
    valid = ~np.isnan(x_disc)
    if valid.sum() < 2:
        return None

    return x_disc.astype(int)


def entropy_from_counts(arr):
    """Entropía Shannon en bits."""
    _, counts = np.unique(arr, return_counts=True, axis=0)
    return entropy(counts, base=2)


def calculate_ais_shannon(series, bins=8, lag=1):
    """
    AIS = I(X_t ; X_{t-lag}) usando discretización por cuantiles.
    Devuelve AIS en bits.
    """
    x = np.asarray(series, dtype=float)

    if len(x) <= lag + 1:
        return np.nan

    if np.allclose(np.std(x), 0):
        return np.nan

    x_disc = safe_qcut(x, q=bins)
    if x_disc is None:
        return np.nan

    past = x_disc[:-lag]
    current = x_disc[lag:]

    if len(past) < 2 or len(current) < 2:
        return np.nan

    h_past = entropy_from_counts(past)
    h_curr = entropy_from_counts(current)
    h_joint = entropy_from_counts(np.stack((past, current), axis=1))

    ais = h_past + h_curr - h_joint
    return max(0.0, float(ais))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    files = sorted(hayling_epo_files())
    if not files:
        raise RuntimeError("No Hayling *-epo.fif files found. Check paths.py.")

    files = [f for f in files if f.parent.name not in EXCLUDE_SUBJECTS]

    rows = []
    skipped = []
    warnings = []

    for f in files:
        subj = f.parent.name
        print(f"-> Processing AIS for {subj}...")

        try:
            epochs = mne.read_epochs(f, preload=True, verbose="ERROR")
        except Exception as e:
            skipped.append(f"{subj}\tREAD_ERROR={repr(e)}")
            continue

        times = epochs.times

        for raw_cond, paper_cond in COND_MAP.items():
            if raw_cond not in epochs.event_id:
                warnings.append(f"{subj}\tmissing_condition={raw_cond}")
                continue

            ep_cond = epochs[raw_cond]
            if len(ep_cond) == 0:
                warnings.append(f"{subj}\t{paper_cond}\tEMPTY")
                continue

            # ROI frontal de referencia para detectar el pico N450
            ref_channels = [ch for ch in ROIS[PEAK_REFERENCE_ROI] if ch in ep_cond.ch_names]
            if not ref_channels:
                warnings.append(f"{subj}\t{paper_cond}\tNO_REF_CHANNELS_{PEAK_REFERENCE_ROI}")
                continue

            data_ref = ep_cond.copy().pick(ref_channels).get_data().mean(axis=1)

            # Cachear datos por ROI
            roi_data = {}
            for roi_name, roi_channels in ROIS.items():
                actual_channels = [ch for ch in roi_channels if ch in ep_cond.ch_names]
                if not actual_channels:
                    warnings.append(f"{subj}\t{paper_cond}\t{roi_name}\tNO_CHANNELS")
                    continue
                roi_data[roi_name] = ep_cond.copy().pick(actual_channels).get_data().mean(axis=1)

            for trial_idx in range(data_ref.shape[0]):
                y_ref = data_ref[trial_idx, :]

                peak_t, peak_amp = detect_negative_peak(y_ref, times, N450_SEARCH_WINDOW)
                if peak_t is None:
                    warnings.append(f"{subj}\t{paper_cond}\ttrial={trial_idx}\tNO_N450_PEAK")
                    continue

                tmin = peak_t - DELTA
                tmax = peak_t + DELTA
                if tmin < times[0] or tmax > times[-1]:
                    warnings.append(f"{subj}\t{paper_cond}\ttrial={trial_idx}\tWINDOW_OUT_OF_BOUNDS")
                    continue

                mask = (times >= tmin) & (times <= tmax)
                n_samples = int(np.sum(mask))
                if n_samples < MIN_SAMPLES:
                    warnings.append(f"{subj}\t{paper_cond}\ttrial={trial_idx}\tTOO_FEW_SAMPLES")
                    continue

                for roi_name, y_all in roi_data.items():
                    y_roi = y_all[trial_idx, mask]
                    ais_bits = calculate_ais_shannon(y_roi, bins=NBINS, lag=LAG)

                    rows.append({
                        "subject": subj,
                        "cond": paper_cond,
                        "trial": int(trial_idx),
                        "trial_uid": f"{subj}_{paper_cond}_{trial_idx}",
                        "component": "N450",
                        "roi": roi_name,
                        "peak_t_ms": float(peak_t * 1000.0),
                        "peak_amp_ref": float(peak_amp),
                        "window_tmin_ms": float(tmin * 1000.0),
                        "window_tmax_ms": float(tmax * 1000.0),
                        "n_samples": n_samples,
                        "ais_bits": float(ais_bits) if not np.isnan(ais_bits) else np.nan,
                    })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("AIS output DataFrame is empty. Check inputs, ROIs, and peak detection.")

    df.to_csv(OUT_CSV, index=False)

    if skipped:
        LOG_SKIPPED.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"[warn] skipped cases written to: {LOG_SKIPPED}")
    elif LOG_SKIPPED.exists():
        LOG_SKIPPED.unlink()

    if warnings:
        LOG_WARNINGS.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print(f"[warn] warnings written to: {LOG_WARNINGS}")
    elif LOG_WARNINGS.exists():
        LOG_WARNINGS.unlink()

    print("[ok] saved:", OUT_CSV)
    print("[ok] rows:", len(df))
    print("[ok] subjects:", df["subject"].nunique())
    print("[ok] conds:", sorted(df["cond"].dropna().unique().tolist()))
    print("[ok] rois:", sorted(df["roi"].dropna().unique().tolist()))
    print("[ok] mean AIS (bits):", round(df["ais_bits"].dropna().mean(), 4) if "ais_bits" in df else "NA")


if __name__ == "__main__":
    main()
