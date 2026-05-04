#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_information_dynamics_cavanagh.py

Calcula Active Information Storage (AIS) y Transfer Entropy (TE)
trial-by-trial basándose en el análisis de residuos del DDS adaptado
para la tarea de Cavanagh (PRED-CT).
Incluye test basados en distribuciones Surrogate para descontar bias estadísticos.
"""

import numpy as np
import pandas as pd
import mne
import sys
from scipy.stats import entropy
from pathlib import Path

# Configuración de Rutas
ROOT = Path(__file__).resolve().parents[1]
EPO_DIR = ROOT / "derivatives" / "epochs"
DDS_RESULTS_CSV = ROOT / "derivatives" / "dds_cavanagh" / "dds_cavanagh_results.csv"

OUTDIR = ROOT / "derivatives" / "info_dynamics_cavanagh"
OUTDIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = OUTDIR / "information_dynamics_results.csv"

# Hyperparametros de discretización y lógica TE/AIS
NBINS = 8
LAG = 1
MIN_SAMPLES = 40
N_SURROGATES = 50

# Variables extraidas y pares direccionales requeridos
FIXED_WINDOW = (0.0, 0.600)   # ms

ROIS = {
    "frontal": ["Fz", "F3", "F4", "AF3", "AF4", "Fp1", "Fp2", "FC3", "FC4"],
    "cacc": ["Fz", "FCz", "FC2", "AFz", "F2"], 
    "lh": ["F3", "F5", "FC3", "FC5"],
    "rh": ["F4", "F6", "FC4", "FC6"],
}

# ---------------------------------------------------------------------
# Reconstrucción del Modelo DDS
# ---------------------------------------------------------------------
def dds_model_free(t, A1, g1, f1, phi1, A2, g2, f2, phi2, C):
    return (A1 * np.exp(-g1 * t) * np.sin(2 * np.pi * f1 * t + phi1) +
            A2 * np.exp(-g2 * t) * np.sin(2 * np.pi * f2 * t + phi2) + C)

# ---------------------------------------------------------------------
# Entropía Discreta SHANNON
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

def calc_ais(series, bins=8, lag=1):
    x = np.asarray(series, dtype=float)
    if len(x) <= lag + 1 or np.allclose(np.std(x), 0): return np.nan
    x_disc = safe_qcut(x, q=bins)
    if x_disc is None: return np.nan
    
    past = x_disc[:-lag]
    current = x_disc[lag:]
    if len(past) < 2: return np.nan
    
    h_past = entropy_from_counts(past)
    h_curr = entropy_from_counts(current)
    h_joint = entropy_from_counts(np.stack((past, current), axis=1))

    return max(0.0, float(h_past + h_curr - h_joint))

def calc_te(source, target, lag=1, bins=8):
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    if len(x) != len(y) or len(x) <= lag + 1 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return np.nan
    
    x_disc = safe_qcut(x, q=bins)
    y_disc = safe_qcut(y, q=bins)
    if x_disc is None or y_disc is None: return np.nan
    
    y_t = y_disc[lag:]
    y_past = y_disc[:-lag]
    x_past = x_disc[:-lag]
    if len(y_t) < 2: return np.nan
    
    h_y_t_y_past = entropy_from_counts(np.stack((y_t, y_past), axis=1))
    h_y_past_x_past = entropy_from_counts(np.stack((y_past, x_past), axis=1))
    h_y_past = entropy_from_counts(y_past)
    h_y_t_y_past_x_past = entropy_from_counts(np.stack((y_t, y_past, x_past), axis=1))

    return max(0.0, float(h_y_t_y_past + h_y_past_x_past - h_y_past - h_y_t_y_past_x_past))

# ---------------------------------------------------------------------
# Generación Mágica de P-values (Surrogates)
# ---------------------------------------------------------------------
def permute_surrogates_ais(series, bins=8, lag=1, iterations=N_SURROGATES):
    x = np.asarray(series, dtype=float)
    if len(x) <= lag + 1 or np.allclose(np.std(x), 0): return np.nan
    x_disc = safe_qcut(x, q=bins)
    if x_disc is None: return np.nan
    
    past = x_disc[:-lag]
    current = x_disc[lag:]
    
    surr_ais = []
    for _ in range(iterations):
        past_shuffled = np.random.permutation(past)
        h_past = entropy_from_counts(past_shuffled)
        h_curr = entropy_from_counts(current)
        h_joint = entropy_from_counts(np.stack((past_shuffled, current), axis=1))
        surr_ais.append(max(0.0, float(h_past + h_curr - h_joint)))
        
    return np.mean(surr_ais), np.std(surr_ais), surr_ais

def permute_surrogates_te(source, target, lag=1, bins=8, iterations=N_SURROGATES):
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    x_disc, y_disc = safe_qcut(x, q=bins), safe_qcut(y, q=bins)
    if x_disc is None or y_disc is None: return np.nan, np.nan, []
    
    y_t = y_disc[lag:]
    y_past = y_disc[:-lag]
    x_past = x_disc[:-lag]
    
    surr_te = []
    for _ in range(iterations):
        x_past_shuffled = np.random.permutation(x_past) # Cortar el link causal predictivo
        
        h_y_t_y_past = entropy_from_counts(np.stack((y_t, y_past), axis=1))
        h_y_past_x_past = entropy_from_counts(np.stack((y_past, x_past_shuffled), axis=1))
        h_y_past = entropy_from_counts(y_past)
        h_y_t_y_past_x_past = entropy_from_counts(np.stack((y_t, y_past, x_past_shuffled), axis=1))
        
        surr_te.append(max(0.0, float(h_y_t_y_past + h_y_past_x_past - h_y_past - h_y_t_y_past_x_past)))

    return np.mean(surr_te), np.std(surr_te), surr_te

def get_p_val(real_val, surr_dist):
    if len(surr_dist) == 0: return np.nan
    return float((np.sum(np.array(surr_dist) >= real_val) + 1) / (len(surr_dist) + 1))


# ---------------------------------------------------------------------
# Flujo Principal
# ---------------------------------------------------------------------
def main():
    if not DDS_RESULTS_CSV.exists():
        print(f"[ERROR] No DDS results found in {DDS_RESULTS_CSV}.")
        sys.exit(1)
        
    print(f"-> Loading DDS fit configurations from {DDS_RESULTS_CSV.name}")
    df_dds = pd.read_csv(DDS_RESULTS_CSV)
    
    epochs_files = sorted(EPO_DIR.glob("*_task-ps_epo.fif"))
    rows = []
    
    for epo_file in epochs_files:
        subj = epo_file.stem.split('_')[0].replace('sub-', '')
        
        if str(subj) not in df_dds['subject'].astype(str).values:
            continue
            
        print(f"-> Processing Subject {subj} Info Dynamics...")
        epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
        times = epochs.times
        mask = (times >= FIXED_WINDOW[0]) & (times <= FIXED_WINDOW[1])
        t_win = times[mask] - FIXED_WINDOW[0]
        
        df_subj = df_dds[df_dds['subject'].astype(str) == str(subj)]
        
        for cond in ['Reward', 'Loss']:
            if cond not in epochs.event_id: continue
            df_cond = df_subj[df_subj['cond'] == cond]
            if df_cond.empty: continue
            
            ep_cond = epochs[cond]
            
            # Cargar canales y promediar a demanda
            # Al no guardar todas en diccionario global ahorramos memoria por epoca
            for trial_idx in range(ep_cond.get_data().shape[0]):
                df_trial = df_cond[df_cond['trial'] == trial_idx]
                if df_trial.empty: continue
                
                res_dict = {}
                
                # 1. Recuperar los residuos para todas las ROIs necesarias
                for roi_name in ROIS.keys():
                    row_roi = df_trial[df_trial['roi'] == roi_name]
                    if row_roi.empty: continue
                    
                    actual_ch = [ch for ch in ROIS[roi_name] if ch in ep_cond.ch_names]
                    if not actual_ch: continue
                    
                    real_y = np.squeeze(ep_cond.copy().pick(actual_ch).get_data())
                    if real_y.ndim > 1: real_y = real_y.mean(axis=0) # ROI Average
                    if real_y.ndim == 1: real_y = real_y[np.newaxis, :] # Make trials iteratable if lonely
                    
                    real_y = real_y[trial_idx, mask] * 1e6 # microvalts scale
                    
                    # Reconstruccion fit()
                    pnts = list(row_roi.iloc[0][['A1', 'gamma1', 'f1', 'phi1', 'A2', 'gamma2', 'f2', 'phi2', 'C']])
                    if np.isnan(pnts).any(): 
                        res_dict[roi_name] = None
                        continue
                        
                    fit_y = dds_model_free(t_win, *pnts)
                    
                    # Residuo Neto 
                    residual = real_y - fit_y
                    # Estandarización -> zscore temporal trial
                    residual = (residual - np.mean(residual)) / np.std(residual)
                    res_dict[roi_name] = residual

                # 2. Computar las Métricas sobre los Residuos Extraídos
                if "cacc" not in res_dict or res_dict["cacc"] is None: continue
                res_cacc = res_dict["cacc"]
                
                # AIS - Capacidad predictiva residual local en Fz
                ais_val = calc_ais(res_cacc)
                sur_mean, sur_std, sur_dist = permute_surrogates_ais(res_cacc)
                ais_p_val = get_p_val(ais_val, sur_dist)
                
                # TE - Traspaso efectivo LH -> cACC y RH -> cACC
                te_lh_val, te_lh_p, te_lh_surr = np.nan, np.nan, np.nan
                if "lh" in res_dict and res_dict["lh"] is not None:
                    te_lh_val = calc_te(res_dict["lh"], res_cacc)
                    s_m, s_st, s_d = permute_surrogates_te(res_dict["lh"], res_cacc)
                    te_lh_p = get_p_val(te_lh_val, s_d)
                    
                te_rh_val, te_rh_p, te_rh_surr = np.nan, np.nan, np.nan
                if "rh" in res_dict and res_dict["rh"] is not None:
                    te_rh_val = calc_te(res_dict["rh"], res_cacc)
                    s_m, s_st, s_d = permute_surrogates_te(res_dict["rh"], res_cacc)
                    te_rh_p = get_p_val(te_rh_val, s_d)
                    
                rows.append({
                    "subject": subj,
                    "cond": cond, 
                    "trial": trial_idx,
                    "n_samples": len(t_win),
                    "AIS_cacc": ais_val,
                    "AIS_cacc_pvalue": ais_p_val,
                    "TE_lh_to_cacc": te_lh_val,
                    "TE_lh_to_cacc_pvalue": te_lh_p,
                    "TE_rh_to_cacc": te_rh_val,
                    "TE_rh_to_cacc_pvalue": te_rh_p
                })
                
    if not rows:
        print("[ERROR] No calculation arrays created.")
        sys.exit(1)
        
    df_out = pd.DataFrame(rows)
    
    # Merge meta informatica y parametros DDS
    df_merge = pd.merge(df_dds, df_out, on=['subject', 'cond', 'trial'], how='inner')
    df_merge.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Información Dinámica y DDS procesados y guardados en {OUT_CSV.name}")
    print(df_merge[['subject', 'cond', 'AIS_cacc', 'TE_lh_to_cacc', 'TE_rh_to_cacc']].describe())
    
if __name__ == "__main__":
    main()
