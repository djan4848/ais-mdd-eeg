#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_dds_peak_aligned_trial_by_tiral.py

Ajuste DDS trial-by-trial sobre segmentos peak-aligned centrados en el N450.

Decisiones metodológicas:
- solo analiza N450
- detección de pico en ROI frontal de referencia
- ventana simétrica ±150 ms alrededor del pico detectado
- tiempo del modelo definido desde el borde izquierdo de la ventana
- modelo DDS original (senos amortiguados) con fases phi1 y phi2
"""

import numpy as np
import pandas as pd
import mne
from scipy.optimize import curve_fit

from dds_base.io.paths import (
    DERIV_ROOT,
    hayling_epo_files,
    ROIS,
    EXCLUDE_SUBJECTS,
)

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
OUTDIR = DERIV_ROOT / "dds_peak_aligned_n450"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = OUTDIR / "dds_n450_results.csv"
LOG_SKIPPED = OUTDIR / "skipped_subjects_dds_n450.txt"
LOG_WARNINGS = OUTDIR / "dds_n450_warnings.txt"

COND_MAP = {
    "ASOC": "INIT",
    "NOASOC": "INHIB",
}

PEAK_REFERENCE_ROI = "frontal"

N450_SEARCH_WINDOW = (0.390, 0.524)   # segundos
DELTA = 0.200                        # ±150 ms
MIN_SAMPLES = 40

# ---------------------------------------------------------------------
# Modelo DDS original + fases
# ---------------------------------------------------------------------
def dds2_phi(t, A1, g1, f1, phi1, A2, g2, f2, phi2):
    """
    Dual Damped Sine model with phases.
    t comienza en el borde izquierdo de la ventana (t >= 0).
    """
    return (
        A1 * np.exp(-g1 * t) * np.sin(2 * np.pi * f1 * t + phi1)
        + A2 * np.exp(-g2 * t) * np.sin(2 * np.pi * f2 * t + phi2)
    )


def guess_f(t, y, fmin, fmax, fallback):
    """Estimación FFT simple para inicializar frecuencia."""
    if len(t) < 3:
        return fallback

    dt = np.median(np.diff(t))
    if dt <= 0:
        return fallback

    y0 = y - np.mean(y)
    Y = np.fft.rfft(y0)
    freqs = np.fft.rfftfreq(len(y0), d=dt)

    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return fallback

    k = np.argmax(np.abs(Y[band]))
    return float(freqs[band][k])


def wrap_phase(phi):
    """Normaliza fase a [-pi, pi]."""
    return float((phi + np.pi) % (2 * np.pi) - np.pi)


def fit_dds_phi(t, y):
    """
    Ajuste robusto del modelo DDS con fases.
    """
    nan_res = {
        k: np.nan for k in
        ["A1", "gamma1", "f1", "phi1", "A2", "gamma2", "f2", "phi2", "r2"]
    }

    if len(t) < MIN_SAMPLES or np.allclose(np.std(y), 0):
        return nan_res

    amp = float(np.ptp(y))
    A0 = amp / 2.0 if amp > 0 else max(abs(np.mean(y)), 1e-6)

    f1_0 = guess_f(t, y, 0.5, 8.0, 3.0)
    f2_0 = guess_f(t, y, 8.0, 30.0, 12.0)

    # Inicialización razonable: el pico suele caer aproximadamente en el centro
    # de la ventana (~150 ms), por lo que phi no debe fijarse rígidamente.
    p0 = [A0, 8.0, f1_0, 0.0, A0 / 2.0, 20.0, f2_0, 0.0]

    lb = [-300.0, 0.001, 0.5, -2 * np.pi, -300.0, 0.001, 8.0, -2 * np.pi]
    ub = [300.0, 150.0, 10.0,  2 * np.pi,  300.0, 200.0, 45.0,  2 * np.pi]

    try:
        popt, _ = curve_fit(
            dds2_phi,
            t,
            y,
            p0=p0,
            bounds=(lb, ub),
            maxfev=80000,
        )

        yhat = dds2_phi(t, *popt)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-18
        r2 = 1.0 - (ss_res / ss_tot)

        return {
            "A1": float(popt[0]),
            "gamma1": float(popt[1]),
            "f1": float(popt[2]),
            "phi1": wrap_phase(popt[3]),
            "A2": float(popt[4]),
            "gamma2": float(popt[5]),
            "f2": float(popt[6]),
            "phi2": wrap_phase(popt[7]),
            "r2": float(r2),
        }
    except Exception:
        return nan_res


def detect_negative_peak(y, t, search_window):
    """Detecta el mínimo en la ventana N450."""
    mask = (t >= search_window[0]) & (t <= search_window[1])
    if not np.any(mask):
        return None, None

    y_seg = y[mask]
    t_seg = t[mask]
    if len(y_seg) == 0:
        return None, None

    idx = int(np.argmin(y_seg))
    return float(t_seg[idx]), float(y_seg[idx])


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
        print(f"-> Processing {subj}...")

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

            # ROI frontal de referencia para detección del pico N450
            ref_channels = [ch for ch in ROIS[PEAK_REFERENCE_ROI] if ch in ep_cond.ch_names]
            if not ref_channels:
                warnings.append(f"{subj}\t{paper_cond}\tNO_REF_CHANNELS_{PEAK_REFERENCE_ROI}")
                continue

            data_ref = ep_cond.copy().pick(ref_channels).get_data().mean(axis=1)

            # Cachear señales por ROI
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
                if np.sum(mask) < MIN_SAMPLES:
                    warnings.append(f"{subj}\t{paper_cond}\ttrial={trial_idx}\tTOO_FEW_SAMPLES")
                    continue

                # Tiempo del modelo: t=0 en el borde izquierdo de la ventana
                t_win = times[mask] - tmin

                for roi_name, y_all in roi_data.items():
                    y_roi = y_all[trial_idx, mask]

                    res = fit_dds_phi(t_win, y_roi)

                    row = {
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
                        "n_samples": int(np.sum(mask)),
                    }
                    row.update(res)
                    rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("DDS output DataFrame is empty. Check inputs, ROIs, and peak detection.")

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
    print("[ok] mean r2:", round(df["r2"].dropna().mean(), 4) if "r2" in df else "NA")


if __name__ == "__main__":
    main()
