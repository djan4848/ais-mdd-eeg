#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_extract_te_n450_direct.py

Calcula Transfer Entropy (TE) bivariada directa entre ROIs
sobre segmentos N450 peak-aligned.

Salida:
    derivatives/te_n450/te_n450_results.csv

Estructura de salida alineada con DDS/AIS:
    subject, cond, trial, trial_uid, component,
    peak_t_ms, peak_amp_ref, window_tmin_ms, window_tmax_ms, n_samples,
    source_roi, target_roi, lag_samples, lag_ms, bins, te_bits
"""

import numpy as np
import pandas as pd
import mne
from collections import Counter

from dds_base.io.paths import (
    DERIV_ROOT,
    hayling_epo_files,
    ROIS,
    EXCLUDE_SUBJECTS,
)

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
OUTDIR = DERIV_ROOT / "te_n450"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = OUTDIR / "te_n450_results.csv"
LOG_SKIPPED = OUTDIR / "skipped_subjects_te_n450.txt"
LOG_WARNINGS = OUTDIR / "te_n450_warnings.txt"

COND_MAP = {
    "ASOC": "INIT",
    "NOASOC": "INHIB",
}

PEAK_REFERENCE_ROI = "frontal"

N450_SEARCH_WINDOW = (0.390, 0.524)   # segundos
DELTA = 0.200                         # ±200 ms
NBINS = 4                             # cuartiles
LAGS = [1, 4]                         # 4 ms y 16 ms a 250 Hz
MIN_SAMPLES = 20

# Pares dirigidos a analizar
ROI_PAIRS = [
    ("cacc", "frontal"),
    ("frontal", "cacc"),
    ("cacc", "lh"),
    ("cacc", "rh"),
    ("lh", "rh"),
    ("rh", "lh"),
]


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


def safe_qcut_int(x, q=4):
    """
    Discretización robusta por cuantiles.
    Devuelve enteros 0..k-1 o None si no se puede discretizar.
    """
    x = np.asarray(x, dtype=float)

    if len(x) < q:
        return None

    # Si la serie es prácticamente constante, no tiene sentido discretizar
    if np.allclose(np.std(x), 0):
        return None

    try:
        x_disc = pd.qcut(x, q=q, labels=False, duplicates="drop")
    except Exception:
        return None

    if x_disc is None:
        return None

    x_disc = np.asarray(x_disc, dtype=float)

    if np.all(np.isnan(x_disc)):
        return None

    valid = ~np.isnan(x_disc)
    if valid.sum() < 2:
        return None

    return x_disc.astype(int)


def entropy_from_counter(counter):
    """Entropía de Shannon en bits a partir de un Counter."""
    counts = np.array(list(counter.values()), dtype=float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def transfer_entropy_discrete(x, y, lag=1):
    """
    TE_{X->Y} = I(Y_t ; X_{t-lag} | Y_{t-lag})
    Implementación empírica discreta en bits.
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) <= lag + 1:
        return np.nan

    x_past = x[:-lag]
    y_past = y[:-lag]
    y_curr = y[lag:]

    if len(y_curr) < 2:
        return np.nan

    # Contadores empíricos
    c_ycurr_ypast = Counter(zip(y_curr, y_past))
    c_ypast_xpast = Counter(zip(y_past, x_past))
    c_ycurr_ypast_xpast = Counter(zip(y_curr, y_past, x_past))
    c_ypast = Counter(y_past)

    # Fórmula equivalente:
    # TE = H(Y_t, Y_past) + H(Y_past, X_past) - H(Y_t, Y_past, X_past) - H(Y_past)
    h_ycurr_ypast = entropy_from_counter(c_ycurr_ypast)
    h_ypast_xpast = entropy_from_counter(c_ypast_xpast)
    h_ycurr_ypast_xpast = entropy_from_counter(c_ycurr_ypast_xpast)
    h_ypast = entropy_from_counter(c_ypast)

    te = h_ycurr_ypast + h_ypast_xpast - h_ycurr_ypast_xpast - h_ypast

    # por robustez numérica
    return max(0.0, float(te))


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
        print(f"-> Processing TE for {subj}...")

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

                # Señales por ROI para este trial
                trial_signals = {}
                for roi_name, arr in roi_data.items():
                    trial_signals[roi_name] = arr[trial_idx, mask]

                # Calcular TE para pares y lags
                for source_roi, target_roi in ROI_PAIRS:
                    if source_roi not in trial_signals or target_roi not in trial_signals:
                        warnings.append(
                            f"{subj}\t{paper_cond}\ttrial={trial_idx}\tPAIR_MISSING\t{source_roi}->{target_roi}"
                        )
                        continue

                    x_cont = trial_signals[source_roi]
                    y_cont = trial_signals[target_roi]

                    # discretización por trial y ROI
                    x_disc = safe_qcut_int(x_cont, q=NBINS)
                    y_disc = safe_qcut_int(y_cont, q=NBINS)

                    if x_disc is None or y_disc is None:
                        warnings.append(
                            f"{subj}\t{paper_cond}\ttrial={trial_idx}\tDISCRETIZATION_FAIL\t{source_roi}->{target_roi}"
                        )
                        continue

                    for lag in LAGS:
                        if len(x_disc) <= lag + 1 or len(y_disc) <= lag + 1:
                            warnings.append(
                                f"{subj}\t{paper_cond}\ttrial={trial_idx}\tLAG_TOO_LONG\tlag={lag}\t{source_roi}->{target_roi}"
                            )
                            continue

                        te_bits = transfer_entropy_discrete(x_disc, y_disc, lag=lag)

                        rows.append({
                            "subject": subj,
                            "cond": paper_cond,
                            "trial": int(trial_idx),
                            "trial_uid": f"{subj}_{paper_cond}_{trial_idx}",
                            "component": "N450",
                            "peak_t_ms": float(peak_t * 1000.0),
                            "peak_amp_ref": float(peak_amp),
                            "window_tmin_ms": float(tmin * 1000.0),
                            "window_tmax_ms": float(tmax * 1000.0),
                            "n_samples": n_samples,
                            "source_roi": source_roi,
                            "target_roi": target_roi,
                            "lag_samples": int(lag),
                            "lag_ms": float(lag * 4.0),   # 250 Hz -> 4 ms/sample
                            "bins": int(NBINS),
                            "te_bits": float(te_bits) if not np.isnan(te_bits) else np.nan,
                        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("TE output DataFrame is empty. Check inputs, ROIs, peak detection, and discretization.")

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
    print("[ok] pairs:", sorted(set(zip(df["source_roi"], df["target_roi"]))))
    print("[ok] lags:", sorted(df["lag_samples"].unique().tolist()))
    print("[ok] mean TE (bits):", round(df["te_bits"].dropna().mean(), 6) if "te_bits" in df else "NA")


if __name__ == "__main__":
    main()
