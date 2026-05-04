import mne
import numpy as np
import pandas as pd
from pathlib import Path
import os
import joblib
import warnings
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

DIR_ASSETS = Path("06_manuscript_assets")
DIR_TDBRAIN = Path("/media/neuraldyn/PortableSSD/TDBRAIN_derivatives")

from scipy.stats import iqr
from scipy import signal

def _get_fd_bins(x):
    iqr_x = iqr(x)
    if iqr_x == 0:
        return 10
    h = 2 * iqr_x * (len(x) ** (-1/3))
    if h == 0:
        return 10
    num_bins = int(np.ceil((np.max(x) - np.min(x)) / h))
    return min(max(num_bins, 5), 100) # Prevents memory explosion

def _shannon_entropy(x):
    bins = _get_fd_bins(x)
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def _joint_entropy(x, y):
    bins_x = _get_fd_bins(x)
    bins_y = _get_fd_bins(y)
    hist, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def _conditional_entropy(x, y):
    return _joint_entropy(x, y) - _shannon_entropy(y)

def calc_te(source, target, tau=5):
    source_past = source[:-tau]
    target_past = target[:-tau]
    target_future = target[tau:]
    
    b_tf = _get_fd_bins(target_future)
    b_tp = _get_fd_bins(target_past)
    b_sp = _get_fd_bins(source_past)
    
    h_tf_tp = _conditional_entropy(target_future, target_past)
    
    # H(X,Y,Z)
    hist_3d, _ = np.histogramdd((target_future, target_past, source_past), bins=(b_tf, b_tp, b_sp), density=True)
    hist_3d = hist_3d[hist_3d > 0]
    h_xyz = -np.sum(hist_3d * np.log2(hist_3d))
    
    h_tp_sp = _joint_entropy(target_past, source_past)
    h_tf_tp_sp = h_xyz - h_tp_sp
    
    te = h_tf_tp - h_tf_tp_sp
    return te

def _conditional_te(source, target, cond, tau=5):
    # cTE = TE(X->Y | Z)
    source_past = source[:-tau]
    target_past = target[:-tau]
    cond_past = cond[:-tau]
    target_future = target[tau:]
    
    b_tf = _get_fd_bins(target_future)
    b_tp = _get_fd_bins(target_past)
    b_sp = _get_fd_bins(source_past)
    b_cp = _get_fd_bins(cond_past)
    
    # H(Y_f, Y_p, Z_p)
    hist_3d_cond, _ = np.histogramdd((target_future, target_past, cond_past), bins=(b_tf, b_tp, b_cp), density=True)
    hist_3d_cond = hist_3d_cond[hist_3d_cond > 0]
    h_yf_yp_zp = -np.sum(hist_3d_cond * np.log2(hist_3d_cond))
    
    h_yp_zp = _joint_entropy(target_past, cond_past)
    h_yf_given_yp_zp = h_yf_yp_zp - h_yp_zp
    
    # H(Y_f, Y_p, X_p, Z_p)
    hist_4d, _ = np.histogramdd((target_future, target_past, source_past, cond_past), bins=(b_tf, b_tp, b_sp, b_cp), density=True)
    hist_4d = hist_4d[hist_4d > 0]
    h_4d = -np.sum(hist_4d * np.log2(hist_4d))
    
    # H(Y_p, X_p, Z_p)
    hist_3d_past, _ = np.histogramdd((target_past, source_past, cond_past), bins=(b_tp, b_sp, b_cp), density=True)
    hist_3d_past = hist_3d_past[hist_3d_past > 0]
    h_yp_xp_zp = -np.sum(hist_3d_past * np.log2(hist_3d_past))
    
    h_yf_given_yp_xp_zp = h_4d - h_yp_xp_zp
    
    cte = h_yf_given_yp_zp - h_yf_given_yp_xp_zp
    return cte

def run_tdbrain_audit():
    print("--- EXTERNAL OOD VALIDATION: TDBRAIN ---")
    
    # 1. Load Clinical Info
    clinical_file = DIR_TDBRAIN / "TDBRAIN_participants_V2.tsv"
    if not clinical_file.exists():
        print("[!] TDBRAIN Clinical metadata not found.")
        return
        
    clin_df = pd.read_csv(clinical_file, sep='\t')
    # Use only clean baseline sessions
    valid_pats = clin_df[(clin_df['indication'].isin(['HEALTHY', 'MDD']))].dropna(subset=['BDI_pre'])
    
    # Assign Unified Severity Index (USI) based on BDI_pre max = 63 (standard)
    valid_pats['USI'] = valid_pats['BDI_pre'].astype(float) / 63.0
    
    # Map to severity
    def assign_sev(row):
        if row['indication'] == 'HEALTHY':
            return 'Healthy'
        elif row['USI'] >= 0.40: # Standard MODMA/Cavanagh Severe cut-off
            return 'Severe MDD'
        else:
            return 'Moderate MDD'
            
    valid_pats['Severity'] = valid_pats.apply(assign_sev, axis=1)
    
    subjects_to_run = valid_pats.drop_duplicates(subset=['participants_ID'])
    print(f"[*] Found {len(subjects_to_run)} valid external TDBRAIN subjects to audit.")
    
    # 2. Loading Scalers & ML Models
    try:
        scaler = joblib.load(DIR_ASSETS / "ThreeBody_Scaler.pkl")
        rf = joblib.load(DIR_ASSETS / "ThreeBody_RF_Model.pkl")
        winz_limits = joblib.load(DIR_ASSETS / "ThreeBody_WinzLimits.pkl")
    except Exception as e:
        print("[!] Required ML binary models missing from Phase 4.")
        print(e)
        return
        
    results = []
    audits = []
    
    for idx, row in subjects_to_run.iterrows():
        sub_id = row['participants_ID']
        sev = row['Severity']
        usi = row['USI']
        
        search_path = DIR_TDBRAIN / sub_id / "ses-1" / "eeg"
        if not search_path.exists():
            continue
            
        vhdr_files = list(search_path.glob("*task-restEC*.vhdr"))
        if not vhdr_files:
            continue
            
        vhdr = vhdr_files[0]
        
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)
            
            ch_names = raw.ch_names
            
            # Simple Topology Maps for TDBRAIN 10-20
            proxy_A = [c for c in ch_names if c.upper() in ['F3', 'F4', 'FP1', 'FP2']] # Frontal (PFC)
            proxy_B = [c for c in ch_names if c.upper() in ['T7', 'T8', 'T3', 'T4']] # Temporal/Insula Proxy
            proxy_C = [c for c in ch_names if c.upper() in ['FZ', 'CZ', 'FCZ']] # Midline Cingulate Proxy
            
            if not proxy_A or not proxy_B or not proxy_C:
                continue
                
            sig_A = np.mean(raw.get_data(picks=proxy_A), axis=0)
            sig_B = np.mean(raw.get_data(picks=proxy_B), axis=0)
            sig_C = np.mean(raw.get_data(picks=proxy_C), axis=0)
            
            # Downsample spatially for calculation speed to a standardized length (avoiding long arrays)
            max_len = 10000 
            if len(sig_A) > max_len:
                sig_A = sig_A[:max_len]
                sig_B = sig_B[:max_len]
                sig_C = sig_C[:max_len]
                
            # MANDATORY Z-Score & Detrend Stabilization
            sig_A = signal.detrend(sig_A)
            sig_B = signal.detrend(sig_B)
            sig_C = signal.detrend(sig_C)
            
            sig_A = (sig_A - np.mean(sig_A)) / (np.std(sig_A) + 1e-9)
            sig_B = (sig_B - np.mean(sig_B)) / (np.std(sig_B) + 1e-9)
            sig_C = (sig_C - np.mean(sig_C)) / (np.std(sig_C) + 1e-9)
                
            # Compute Transfer Entropies
            te_b_a = calc_te(sig_B, sig_A)
            te_a_b = calc_te(sig_A, sig_B)
            
            delta_te = te_b_a - te_a_b
            
            cte_b_a = _conditional_te(sig_B, sig_A, sig_C)
            c_factor = ((te_b_a - cte_b_a) / te_b_a) * 100 if te_b_a > 1e-4 else 0.0
            
            # Interaction Information II
            II = te_b_a - cte_b_a
            
            # Log purely math execution prior to sanity checks for auditory histogram
            audits.append({'Subject': sub_id, 'TE_Insula_PFC': te_b_a, 'Interaction_Information_II': II})
            
            # Bit Sanity Checks
            if te_b_a > 10 or te_a_b > 10 or te_b_a < 0 or te_a_b < 0:
                print(f"[!] Rejected {sub_id} due to pathological divergence in Entropic Bits.")
                continue
            
            # Final sanity check for numerical explosions (e.g., c_factor > 1e4)
            if np.abs(II) > 20 or np.abs(c_factor) > 1e4:
                continue
            
            results.append({
                'Subject': sub_id,
                'Severity': sev,
                'USI': usi,
                'TE_Insula_PFC': te_b_a,
                'TE_PFC_Insula': te_a_b,
                'Delta_TE_Ins_PFC': delta_te,
                'Conditioning_Factor_Perc': c_factor,
                'Interaction_Information_II': II
            })
            
        except Exception as e:
            continue
            
    df_audit = pd.DataFrame(audits)
    if not df_audit.empty:
        # Metric Fidelity Histogram
        plt.figure(figsize=(10, 5))
        sns.histplot(df_audit['TE_Insula_PFC'], color='blue', alpha=0.5, label='TE (Insula -> PFC)', kde=True)
        sns.histplot(df_audit['Interaction_Information_II'], color='red', alpha=0.5, label='Interaction Info (II)', kde=True)
        plt.axvline(0, color='black', linestyle='--')
        plt.xlabel('Bits (Entropia Estabilizada O Cruda FDR)')
        plt.ylabel('Frecuencia OOD')
        plt.title('TDBRAIN Metric Stability Audit\n(Post Z-Score & FDR Binning)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(DIR_ASSETS / "Figure_TDBRAIN_Metric_Stability.png", dpi=300)
        plt.close()
            
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[!] No results parsed due to failures or missing topology.")
        return
        
    df_res = df_res.dropna().copy()
    if len(df_res) == 0:
        print("[!] All subjects were pathologically rejected (N=0). OOD pipeline aborted gracefully.")
        return
    
    # 3. Predict & OOD Generalization Test
    print(f"\n[+] Evaluated N={len(df_res)} TDBRAIN subjects via Mathematical Proxy.")
    
    # Apply exactly identical winsorization mapping from Cavanagh training memory
    for feat in ['Interaction_Information_II', 'Conditioning_Factor_Perc', 'Delta_TE_Ins_PFC']:
        lower, upper = winz_limits[feat]
        df_res[f'{feat}_Winz'] = np.clip(df_res[feat], lower, upper)
        
    X_ext = df_res[['Interaction_Information_II_Winz', 'Conditioning_Factor_Perc_Winz', 'Delta_TE_Ins_PFC_Winz']]
    X_ext_scaled = scaler.transform(X_ext)
    
    preds = rf.predict(X_ext_scaled)
    df_res['Latent_Prediction'] = preds
    
    # Binary Eval
    df_bin = df_res[df_res['Severity'].isin(['Healthy', 'Severe MDD'])]
    if len(df_bin) > 0:
        y_true = df_bin['Severity'].map({'Healthy': 0, 'Severe MDD': 1}).values
        y_pred = df_bin['Latent_Prediction'].values
        acc = (y_true == y_pred).mean() * 100
        print(f"OOD Generalization Accuracy (Subsets HC vs Severe): {acc:.2f}%")
        
        cm = np.zeros((2,2), dtype=int)
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
            
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Healthy', 'Severe'], yticklabels=['Healthy', 'Severe'])
        plt.ylabel('TDBRAIN Ground Truth')
        plt.xlabel('Cavanagh RF Prediction')
        plt.title(f'TDBRAIN Zero-Retraining Paradigm\nExternal Generalization Score: {acc:.2f}%')
        plt.tight_layout()
        plt.savefig(DIR_ASSETS / "TDBRAIN_Confusion_Matrix.png", dpi=300)
        plt.close()
        
    # Bimodal Moderate Audit
    df_mod = df_res[df_res['Severity'] == 'Moderate MDD']
    if len(df_mod) > 0:
        y_mod_pred = df_mod['Latent_Prediction'].values
        r_hc = np.sum(y_mod_pred == 0) / len(y_mod_pred)
        r_sev = np.sum(y_mod_pred == 1) / len(y_mod_pred)
        
    # Latent Scatter Overlay Match
    try:
        df_old = pd.read_csv(DIR_ASSETS / "Three_Body_Orbit_Decay.csv")
        df_old['Dataset'] = 'MODMA/Cavanagh (Train)'
        df_res['Dataset'] = 'TDBRAIN (OOD)'
        
        # Merge for plotting
        df_plot = pd.concat([df_old[['Interaction_Information_II', 'Conditioning_Factor_Perc', 'Dataset']],
                             df_res[['Interaction_Information_II', 'Conditioning_Factor_Perc', 'Dataset']]])
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_plot, x='Interaction_Information_II', y='Conditioning_Factor_Perc',
                        hue='Dataset', style='Dataset', palette={'MODMA/Cavanagh (Train)': 'gray', 'TDBRAIN (OOD)': 'blue'},
                        alpha=0.6, s=100, edgecolor='black')
                        
        plt.axvline(0, color='red', linestyle='--', alpha=0.3)
        plt.axhline(0, color='red', linestyle='--', alpha=0.3)
        
        plt.xlim(winz_limits['Interaction_Information_II'][0], winz_limits['Interaction_Information_II'][1])
        plt.ylim(winz_limits['Conditioning_Factor_Perc'][0], winz_limits['Conditioning_Factor_Perc'][1])
        
        plt.title('Latent Space Overlay:\nTDBRAIN OOD vs Neural Training Geometries')
        plt.xlabel('Interaction Information (Synergy)')
        plt.ylabel('Cingulate Intercept Influence ($C_{factor}$ %)')
        plt.tight_layout()
        plt.savefig(DIR_ASSETS / "TDBRAIN_Latent_Space.png", dpi=300)
        plt.close()
    except Exception as e:
        print("[!] Could not overlay scatter plots:", e)
        
    df_res.to_csv(DIR_ASSETS / "tdbrain_generalization_results_STABLE.csv", index=False)
    print("\n[+] TDBRAIN Verification Concluded. Data exported.")
    
if __name__ == "__main__":
    run_tdbrain_audit()
