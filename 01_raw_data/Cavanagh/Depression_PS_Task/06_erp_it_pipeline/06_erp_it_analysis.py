#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_erp_it_analysis.py

Pipeline integrado de ERP Clásico y Teoría de la Información (IT)
para el dataset de Cavanagh (PRED-CT).
Extrae características Trial-by-Trial de amplitud (FRN/RewP) y dinámicas 
temporales no lineales (AIS, TE, PID) optimizando parámetros de embedding.

Autor: Antigravity Code Assistant
"""

import numpy as np
import pandas as pd
import mne
import sys
from scipy.stats import entropy
from pathlib import Path

# ---------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
EPO_DIR = ROOT / "derivatives" / "epochs"
OUTDIR = ROOT / "derivatives" / "erp_it_cavanagh"
OUTDIR.mkdir(exist_ok=True, parents=True)
OUT_CSV = OUTDIR / "erp_it_master_results.csv"

# Parámetros ERP
BASELINE = (-0.2, 0.0)
FILTER_BANDS = (1.0, 40.0)
ERP_WINDOW = (0.250, 0.450) # 250ms a 450ms

# Parámetros IT
IT_WINDOW = (0.0, 0.600)    # Ventana de feedback entera para IT
NBINS = 8
MIN_SAMPLES = 40
MAX_TAU_SEARCH = 30         # Máximo lag para buscar el mínimo de AMI
N_SURROGATES = 50           # Permutaciones para significancia

# Canales
SOURCE_CH = "Fz"
SINK_CH_PREF = ["Cz", "FCz"]

# ---------------------------------------------------------------------
# Funciones Básicas de Teoría de la Información (Binned Shannon)
# ---------------------------------------------------------------------
def safe_qcut(x, q):
    try:
        x_disc = pd.qcut(x, q=q, labels=False, duplicates="drop")
    except ValueError:
        return None
    x_disc = np.asarray(x_disc, dtype=float)
    if np.all(np.isnan(x_disc)): return None
    if (~np.isnan(x_disc)).sum() < 2: return None
    return x_disc.astype(int)

def entropy_from_counts(arr):
    _, counts = np.unique(arr, return_counts=True, axis=0)
    return entropy(counts, base=2)

def calc_mi(x, y):
    h_x = entropy_from_counts(x)
    h_y = entropy_from_counts(y)
    h_xy = entropy_from_counts(np.stack((x, y), axis=1))
    return max(0.0, float(h_x + h_y - h_xy))

# ---------------------------------------------------------------------
# Estimación Dinámica de TAU (Primer Mínimo de AMI)
# ---------------------------------------------------------------------
def estimate_tau(series, max_lag=30, bins=8):
    """
    Estima el delay temporal óptimo (tau) buscando el 
    primer mínimo local de la Auto-Información Mutua (AMI).
    """
    x = np.asarray(series, dtype=float)
    x_disc = safe_qcut(x, q=bins)
    if x_disc is None or len(x) < max_lag + 2:
        return 1 # Fallback
        
    ami_vals = []
    for lag in range(1, max_lag + 1):
        past = x_disc[:-lag]
        curr = x_disc[lag:]
        ami_vals.append(calc_mi(past, curr))
        
    # Primer mínimo local
    for i in range(1, len(ami_vals) - 1):
        if ami_vals[i] < ami_vals[i-1] and ami_vals[i] < ami_vals[i+1]:
            return i + 1 # el index 0 es lag 1
            
    # Si no hay mínimo local, retornar donde cae al < 1/e del inicial o el lag=1
    return 1

# ---------------------------------------------------------------------
# Core IT Metrics (AIS, TE, PID)
# ---------------------------------------------------------------------
def calc_ais(series, lag, bins=8):
    x = np.asarray(series)
    x_disc = safe_qcut(x, q=bins)
    if x_disc is None or len(x) <= lag: return np.nan
    past, curr = x_disc[:-lag], x_disc[lag:]
    return calc_mi(past, curr) # AIS en embed 1 es la auto-mi a lag tau.

def calc_te(source, target, lag, bins=8):
    x, y = np.asarray(source), np.asarray(target)
    x_disc, y_disc = safe_qcut(x, q=bins), safe_qcut(y, q=bins)
    if x_disc is None or y_disc is None or len(x) <= lag: return np.nan
    
    y_t, y_past, x_past = y_disc[lag:], y_disc[:-lag], x_disc[:-lag]
    h_y_t_y_past = entropy_from_counts(np.stack((y_t, y_past), axis=1))
    h_y_past_x_past = entropy_from_counts(np.stack((y_past, x_past), axis=1))
    h_y_past = entropy_from_counts(y_past)
    h_y_t_y_past_x_past = entropy_from_counts(np.stack((y_t, y_past, x_past), axis=1))
    return max(0.0, float(h_y_t_y_past + h_y_past_x_past - h_y_past - h_y_t_y_past_x_past))

def calc_pid_mmi(src1, src2, target):
    """ MMI PID: Redundancy, Unique1, Unique2, Synergy """
    i_s1_t = calc_mi(src1, target)
    i_s2_t = calc_mi(src2, target)

    h_s1s2 = entropy_from_counts(np.stack((src1, src2), axis=1))
    h_t = entropy_from_counts(target)
    h_s1s2_t = entropy_from_counts(np.stack((src1, src2, target), axis=1))
    i_joint = max(0.0, float(h_s1s2 + h_t - h_s1s2_t))

    redun = min(i_s1_t, i_s2_t)
    unq1 = max(0.0, i_s1_t - redun)
    unq2 = max(0.0, i_s2_t - redun)
    syn = max(0.0, i_joint - redun - unq1 - unq2)
    return redun, unq1, unq2, syn

def permute_surrogates_te(source, target, lag, bins=8, iterations=N_SURROGATES):
    x, y = np.asarray(source), np.asarray(target)
    x_disc, y_disc = safe_qcut(x, q=bins), safe_qcut(y, q=bins)
    if x_disc is None or y_disc is None: return []
    
    y_t, y_past, x_past = y_disc[lag:], y_disc[:-lag], x_disc[:-lag]
    surr = []
    
    h_y_past = entropy_from_counts(y_past)
    h_y_t_y_past = entropy_from_counts(np.stack((y_t, y_past), axis=1))
    
    for _ in range(iterations):
        x_past_sh = np.random.permutation(x_past)
        h_y_past_x_past = entropy_from_counts(np.stack((y_past, x_past_sh), axis=1))
        h_joint = entropy_from_counts(np.stack((y_t, y_past, x_past_sh), axis=1))
        surr.append(max(0.0, float(h_y_t_y_past + h_y_past_x_past - h_y_past - h_joint)))
    return surr

def get_p_val(real_val, surr_dist):
    if len(surr_dist) == 0 or np.isnan(real_val): return np.nan
    return float((np.sum(np.array(surr_dist) >= real_val) + 1) / (len(surr_dist) + 1))


# ---------------------------------------------------------------------
# Ejecución Principal
# ---------------------------------------------------------------------
def main():
    epochs_files = sorted(EPO_DIR.glob("*_task-ps_epo.fif"))
    if not epochs_files:
        print(f"[ERROR] No epoch files found in {EPO_DIR}")
        sys.exit(1)
        
    # Crear archivo si no existe
    if not OUT_CSV.exists():
        pd.DataFrame(columns=[
            "subject", "cond", "trial", "source_ch", "sink_ch",
            "Fz_mean_amp_uV", "Fz_min_peak_uV", "Fz_min_lat_ms", 
            "Fz_max_peak_uV", "Fz_max_lat_ms", "tau_lag", "AIS_Fz", 
            "TE_Fz_to_Sink", "TE_Fz_to_Sink_pval", "PID_redundancy", 
            "PID_synergy", "PID_unique_Fz", "PID_unique_Sink"
        ]).to_csv(OUT_CSV, index=False)
        
    for epo_file in epochs_files:
        subj = epo_file.stem.split('_')[0].replace('sub-', '')
        master_rows = [] # Reset rows list per subject
        
        # Check if already processed
        try:
            df_exist = pd.read_csv(OUT_CSV)
            if subj in df_exist['subject'].astype(str).values:
                print(f"-> Subject {subj} already processed. Skipping.")
                continue
        except pd.errors.EmptyDataError:
            pass

        print(f"-> Processing Subject {subj} ERP-IT Pipeline...")
        
        try:
            epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
        except Exception as e:
            print(f"Skipping {subj}: {e}")
            continue
            
        # Limpieza base (Fase 1 Requirements)
        epochs.filter(l_freq=FILTER_BANDS[0], h_freq=FILTER_BANDS[1], verbose=False)
        epochs.apply_baseline(BASELINE, verbose=False)
        
        # Validar Canales
        if SOURCE_CH not in epochs.ch_names:
            print(f"  [WARN] Missing Source {SOURCE_CH}. Skipping.")
            continue
            
        sink_ch = next((ch for ch in SINK_CH_PREF if ch in epochs.ch_names), None)
        if not sink_ch:
            print(f"  [WARN] Missing Sink {SINK_CH_PREF}. Skipping.")
            continue
            
        times = epochs.times
        mask_erp = (times >= ERP_WINDOW[0]) & (times <= ERP_WINDOW[1])
        t_erp = times[mask_erp]
        
        mask_it = (times >= IT_WINDOW[0]) & (times <= IT_WINDOW[1])
        t_it = times[mask_it]
        
        for cond in ['Reward', 'Loss']:
            if cond not in epochs.event_id: continue
            ep_cond = epochs[cond]
            
            for trial_idx in range(len(ep_cond)):
                trial_data = ep_cond.get_data(item=trial_idx, copy=False)[0] # Shape: (channels, times)
                
                # ERP Extraction (Window 250-450)
                ch_idx_fz = epochs.ch_names.index(SOURCE_CH)
                fz_erp_win = trial_data[ch_idx_fz, mask_erp] * 1e6 # microvoltios
                
                mean_amp = np.mean(fz_erp_win)
                peak_min = np.min(fz_erp_win)
                peak_max = np.max(fz_erp_win)
                
                peak_min_lat = t_erp[np.argmin(fz_erp_win)] * 1000 # ms
                peak_max_lat = t_erp[np.argmax(fz_erp_win)] * 1000 # ms
                
                # IT Extraction (Window 0-600)
                fz_it_win = trial_data[ch_idx_fz, mask_it] * 1e6
                sink_it_win = trial_data[epochs.ch_names.index(sink_ch), mask_it] * 1e6
                
                # 1. Estimation tau of Fz
                tau = estimate_tau(fz_it_win, max_lag=MAX_TAU_SEARCH, bins=NBINS)
                
                # 2. AIS (Memoria en Fz)
                ais_fz = calc_ais(fz_it_win, lag=tau, bins=NBINS)
                
                # 3. TE (Comunicacion Fz -> Sink)
                te_src_snk = calc_te(fz_it_win, sink_it_win, lag=tau, bins=NBINS)
                te_surr = permute_surrogates_te(fz_it_win, sink_it_win, lag=tau, bins=NBINS)
                te_pval = get_p_val(te_src_snk, te_surr)
                
                # 4. Temporal Spatial PID (Predicting Sink based on past Source and past Sink)
                # s1 = source_past, s2 = sink_past, tgt = sink_curr
                if len(fz_it_win) > tau + 1:
                    s1_past = safe_qcut(fz_it_win[:-tau], q=NBINS)
                    s2_past = safe_qcut(sink_it_win[:-tau], q=NBINS)
                    tgt_curr = safe_qcut(sink_it_win[tau:], q=NBINS)
                    
                    if s1_past is not None and s2_past is not None and tgt_curr is not None:
                        redun, unq1, unq2, syn = calc_pid_mmi(s1_past, s2_past, tgt_curr)
                    else:
                        redun, unq1, unq2, syn = np.nan, np.nan, np.nan, np.nan
                else:
                    redun, unq1, unq2, syn = np.nan, np.nan, np.nan, np.nan
                    
                master_rows.append({
                    "subject": subj,
                    "cond": cond,
                    "trial": trial_idx,
                    "source_ch": SOURCE_CH,
                    "sink_ch": sink_ch,
                    # ERP Features
                    "Fz_mean_amp_uV": mean_amp,
                    "Fz_min_peak_uV": peak_min,
                    "Fz_min_lat_ms": peak_min_lat,
                    "Fz_max_peak_uV": peak_max,
                    "Fz_max_lat_ms": peak_max_lat,
                    # IT Features
                    "tau_lag": tau,
                    "AIS_Fz": ais_fz,
                    "TE_Fz_to_Sink": te_src_snk,
                    "TE_Fz_to_Sink_pval": te_pval,
                    "PID_redundancy": redun,
                    "PID_synergy": syn,
                    "PID_unique_Fz": unq1,
                    "PID_unique_Sink": unq2
                })
        
        # Save per subject
        if master_rows:
            df_subj = pd.DataFrame(master_rows)
            df_subj.to_csv(OUT_CSV, mode='a', header=False, index=False)
            print(f"  [OK] Saved {len(df_subj)} trials for Subject {subj}.")
                
    print(f"\n[OK] Pipeline completed! Results available in {OUT_CSV.name}")
    

if __name__ == "__main__":
    main()
