#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_extract_trial_roi_timeseries.py

Extrae series temporales trial-by-trial promediadas por ROI a partir de epochs
de Hayling ya preprocesados.

Salida:
    derivatives/trial_roi_timeseries/trial_roi_timeseries.csv

Columnas:
    subject, cond, trial, trial_uid, roi, sample_idx, time_ms, value
"""

from pathlib import Path
import mne
import pandas as pd

from dds_base.io.paths import (
    DERIV_ROOT,
    hayling_epo_files,
    ROIS,
    EXCLUDE_SUBJECTS,
)

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
APPLY_BASELINE = False          # Activar solo si confirmas que los .epo NO vienen baselinados
BASELINE = (-0.2, 0.0)

COND_MAP = {
    "ASOC": "INIT",
    "NOASOC": "INHIB",
}

OUTDIR = DERIV_ROOT / "trial_roi_timeseries"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = OUTDIR / "trial_roi_timeseries.csv"
LOG_SKIPPED = OUTDIR / "skipped_subjects_trial_roi_timeseries.txt"
LOG_ROI_WARNINGS = OUTDIR / "roi_channel_warnings_trial_roi_timeseries.txt"


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def extract_roi_signal(epoch_data, ch_names, roi_channels):
    """
    Promedia los canales disponibles de una ROI para un trial.
    Devuelve:
        signal: np.ndarray (n_times,) o None si no hay canales
        used_channels: lista de canales usados
    """
    idx = [ch_names.index(ch) for ch in roi_channels if ch in ch_names]
    if not idx:
        return None, []
    signal = epoch_data[idx].mean(axis=0)
    used_channels = [ch_names[i] for i in idx]
    return signal, used_channels


def append_condition_rows(rows, roi_warnings, subj, cond_label, epochs_cond):
    """
    Añade a `rows` todas las muestras de todos los trials de una condición.
    """
    times_ms = epochs_cond.times * 1000.0
    ch_names = epochs_cond.ch_names
    data = epochs_cond.get_data()  # (n_trials, n_channels, n_times)

    for trial_idx in range(data.shape[0]):
        trial_data = data[trial_idx]
        trial_uid = f"{subj}_{cond_label}_{trial_idx}"

        for roi_name, roi_channels in ROIS.items():
            signal, used_channels = extract_roi_signal(trial_data, ch_names, roi_channels)

            if signal is None:
                roi_warnings.append(
                    f"{subj}\t{cond_label}\ttrial={trial_idx}\troi={roi_name}\tNO_CHANNELS_FOUND"
                )
                continue

            if len(used_channels) != len(roi_channels):
                missing = sorted(set(roi_channels) - set(used_channels))
                roi_warnings.append(
                    f"{subj}\t{cond_label}\ttrial={trial_idx}\troi={roi_name}\t"
                    f"used={used_channels}\tmissing={missing}"
                )

            for sample_idx, (t_ms, value) in enumerate(zip(times_ms, signal)):
                rows.append({
                    "subject": subj,
                    "cond": cond_label,
                    "trial": int(trial_idx),          # índice dentro de la condición
                    "trial_uid": trial_uid,           # identificador único robusto
                    "roi": roi_name,
                    "sample_idx": int(sample_idx),
                    "time_ms": round(float(t_ms), 3),
                    "value": float(value),
                })


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    files = sorted(hayling_epo_files())
    if not files:
        raise RuntimeError(
            "No se encontraron archivos *-epo.fif vía hayling_epo_files(). "
            "Revisa HAYLING_EEG_ROOT y HAYLING_EPO_GLOB en paths.py."
        )

    rows = []
    skipped = []
    roi_warnings = []

    # Exclusión centralizada
    files = [f for f in files if f.parent.name not in EXCLUDE_SUBJECTS]

    for f in files:
        subj = f.parent.name

        try:
            epochs = mne.read_epochs(f, preload=True, verbose="ERROR")
        except Exception as e:
            skipped.append(f"{subj}\t{f.name}\tREAD_ERROR={repr(e)}")
            continue

        if APPLY_BASELINE:
            epochs.apply_baseline(BASELINE)

        missing_conds = [raw_cond for raw_cond in COND_MAP if raw_cond not in epochs.event_id]
        if missing_conds:
            skipped.append(f"{subj}\t{f.name}\tmissing_in_event_id={missing_conds}")
            continue

        for raw_cond, paper_cond in COND_MAP.items():
            ep_cond = epochs[raw_cond]

            if len(ep_cond) == 0:
                skipped.append(f"{subj}\t{f.name}\t{raw_cond}_EMPTY")
                continue

            append_condition_rows(rows, roi_warnings, subj, paper_cond, ep_cond)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "El DataFrame de salida quedó vacío. "
            "Revisa etiquetas de condición, exclusiones y archivos de entrada."
        )

    df.to_csv(OUT_CSV, index=False)

    if skipped:
        LOG_SKIPPED.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"[warn] Casos omitidos: {len(skipped)} -> {LOG_SKIPPED}")
    elif LOG_SKIPPED.exists():
        LOG_SKIPPED.unlink()

    if roi_warnings:
        LOG_ROI_WARNINGS.write_text("\n".join(roi_warnings) + "\n", encoding="utf-8")
        print(f"[warn] Inconsistencias ROI registradas en: {LOG_ROI_WARNINGS}")
    elif LOG_ROI_WARNINGS.exists():
        LOG_ROI_WARNINGS.unlink()

    print("[ok] Guardado:", OUT_CSV)
    print("[ok] Filas:", len(df))
    print("[ok] Sujetos:", df["subject"].nunique())
    print("[ok] ROIs:", sorted(df["roi"].unique().tolist()))
    print("[ok] Condiciones:", sorted(df["cond"].unique().tolist()))
    print("[ok] Sujetos excluidos:", sorted(EXCLUDE_SUBJECTS))


if __name__ == "__main__":
    main()
