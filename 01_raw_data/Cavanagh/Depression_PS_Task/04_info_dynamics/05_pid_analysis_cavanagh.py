#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_pid_analysis_cavanagh.py

Calcula Partial Information Decomposition (PID) trial-by-trial
basado en la métrica MMI (Minimum Mutual Information) sobre la señal
residual (DDS_fit - real) para evaluar la integración lateral-medial
durante el procesamiento de feedback (Loss/Reward).

Fuentes (Sources):
    lh, rh
Destino (Target):
    cacc (prioridad Fz)

Salida:
    derivatives/info_dynamics_cavanagh/pid_analysis_results.csv
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

OUT_CSV = OUTDIR / "pid_analysis_results.csv"

# Hyperparametros de discretización y lógica PID
NBINS = 8
LAG = 1
MIN_SAMPLES = 40
N_SURROGATES = 50

FIXED_WINDOW = (0.0, 0.600)   # ms

# Reducimos los ROIs solo a los necesarios
ROIS = {
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
# PID (MMI Approximation)
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

def mi_discrete(x, y):
    h_x = entropy_from_counts(x)
    h_y = entropy_from_counts(y)
    h_xy = entropy_from_counts(np.stack((x, y), axis=1))
    return max(0.0, float(h_x + h_y - h_xy))

def pid_mmi_two_sources(src1, src2, target):
    """
    PID aproximada tipo MMI:
      redundancy = min(I(S1;T), I(S2;T))
      unique1    = I(S1;T) - redundancy
      unique2    = I(S2;T) - redundancy
      synergy    = I(S1,S2;T) - unique1 - unique2 - redundancy
    """
    i_s1_t = mi_discrete(src1, target)
    i_s2_t = mi_discrete(src2, target)

    h_s1s2 = entropy_from_counts(np.stack((src1, src2), axis=1))
    h_t = entropy_from_counts(target)
    h_s1s2_t = entropy_from_counts(np.stack((src1, src2, target), axis=1))
    
    i_joint = max(0.0, float(h_s1s2 + h_t - h_s1s2_t))

    redundancy = min(i_s1_t, i_s2_t)
    unique1 = max(0.0, i_s1_t - redundancy)
    unique2 = max(0.0, i_s2_t - redundancy)
    synergy = max(0.0, i_joint - redundancy - unique1 - unique2)

    return i_s1_t, i_s2_t, i_joint, redundancy, unique1, unique2, synergy

def permute_surrogates_pid_synergy(src1, src2, target, iterations=N_SURROGATES):
    surr_syn = []
    for _ in range(iterations):
        # Desacoplar T de S1/S2 manteniendo la relación S1-S2 
        # para testear si la sinergia es genuina al target
        target_shuffled = np.random.permutation(target)
        
        i_s1_t_s = mi_discrete(src1, target_shuffled)
        i_s2_t_s = mi_discrete(src2, target_shuffled)

        h_s1s2 = entropy_from_counts(np.stack((src1, src2), axis=1))
        h_t_s = entropy_from_counts(target_shuffled)
        h_s1s2_t_s = entropy_from_counts(np.stack((src1, src2, target_shuffled), axis=1))
        
        i_joint_s = max(0.0, float(h_s1s2 + h_t_s - h_s1s2_t_s))

        redundancy_s = min(i_s1_t_s, i_s2_t_s)
        unique1_s = max(0.0, i_s1_t_s - redundancy_s)
        unique2_s = max(0.0, i_s2_t_s - redundancy_s)
        synergy_s = max(0.0, i_joint_s - redundancy_s - unique1_s - unique2_s)
        
        surr_syn.append(synergy_s)
        
    return np.mean(surr_syn), np.std(surr_syn), surr_syn

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
            
        print(f"-> Processing Subject {subj} PID Analysis...")
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
            
            for trial_idx in range(ep_cond.get_data().shape[0]):
                df_trial = df_cond[df_cond['trial'] == trial_idx]
                if df_trial.empty: continue
                
                res_dict = {}
                
                # 1. Recuperar los residuos predictivos
                for roi_name in ROIS.keys():
                    row_roi = df_trial[df_trial['roi'] == roi_name]
                    if row_roi.empty: continue
                    
                    actual_ch = [ch for ch in ROIS[roi_name] if ch in ep_cond.ch_names]
                    if not actual_ch: continue
                    
                    real_y = np.squeeze(ep_cond.copy().pick(actual_ch).get_data())
                    if real_y.ndim > 1: real_y = real_y.mean(axis=0)
                    if real_y.ndim == 1: real_y = real_y[np.newaxis, :]
                    
                    real_y = real_y[trial_idx, mask] * 1e6 # microvolts
                    
                    pnts = list(row_roi.iloc[0][['A1', 'gamma1', 'f1', 'phi1', 'A2', 'gamma2', 'f2', 'phi2', 'C']])
                    if np.isnan(pnts).any(): 
                        res_dict[roi_name] = None
                        continue
                        
                    fit_y = dds_model_free(t_win, *pnts)
                    residual = real_y - fit_y
                    residual = (residual - np.mean(residual)) / np.std(residual) # Zscore para discretización estable
                    res_dict[roi_name] = residual

                # 2. Computar PID (LH, RH) -> cACC
                if "cacc" not in res_dict or res_dict["cacc"] is None: continue
                if "lh" not in res_dict or res_dict["lh"] is None: continue
                if "rh" not in res_dict or res_dict["rh"] is None: continue
                
                s1 = res_dict["lh"]
                s2 = res_dict["rh"]
                tg = res_dict["cacc"]
                
                if min(len(s1), len(s2), len(tg)) <= LAG + 1: continue

                # Discretizar para TE (Métrica Transferencial)
                s1_past = safe_qcut(s1[:-LAG], q=NBINS)
                s2_past = safe_qcut(s2[:-LAG], q=NBINS)
                tg_curr = safe_qcut(tg[LAG:], q=NBINS)
                
                if s1_past is None or s2_past is None or tg_curr is None: continue
                
                # Calcular Metricas PID Reales
                mi_s1, mi_s2, mi_joint, redun, unq_s1, unq_s2, syn = pid_mmi_two_sources(s1_past, s2_past, tg_curr)
                
                # Extraer Significancia Unicamente del Termino de Sinergia (El mas crítico teóricamente para Integration)
                s_mean, s_std, s_dist = permute_surrogates_pid_synergy(s1_past, s2_past, tg_curr)
                syn_pval = get_p_val(syn, s_dist)
                    
                rows.append({
                    "subject": subj,
                    "cond": cond, 
                    "trial": trial_idx,
                    "n_samples": len(t_win),
                    "PID_redundancy": redun,
                    "PID_unique_lh": unq_s1,
                    "PID_unique_rh": unq_s2,
                    "PID_synergy": syn,
                    "PID_synergy_pvalue": syn_pval
                })
                
    if not rows:
        print("[ERROR] No PID arrays created.")
        sys.exit(1)
        
    df_out = pd.DataFrame(rows)
    
    # Merge meta informatica y parametros DDS
    # Limpiamos duplicaciones de DDS ya que los ROIs no importan, 
    # queremos una linea unica por ensayo, conservamos solo CACC model values para union
    df_dds_cacc = df_dds[df_dds['roi'] == 'cacc'].drop(columns=['roi'])
    
    df_merge = pd.merge(df_dds_cacc, df_out, on=['subject', 'cond', 'trial'], how='inner')
    df_merge.to_csv(OUT_CSV, index=False)
    
    print(f"\n[OK] PID Analysis completado y guardado en {OUT_CSV.name}")
    print(df_merge[['subject', 'cond', 'PID_redundancy', 'PID_synergy', 'PID_synergy_pvalue']].describe())
    
if __name__ == "__main__":
    main()
