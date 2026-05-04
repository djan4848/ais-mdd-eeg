import numpy as np
import pandas as pd
import mne
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from mne_icalabel import label_components

warnings.filterwarnings('ignore')

DIR_DATA = Path("01_raw_data")
DIR_ASSETS = Path("06_manuscript_assets")
FILE_LOOKUP = DIR_DATA / "clinical_lookup.csv"

FS = 256.0
T_LENGTH = 2.0
TE_LAG = 5
N_BINS = 4

# Strict non-overlapping proxies to ensure pure Synergy/Redundancy separation
REGIONS_1020 = {
    'Frontal': ['F3', 'F4', 'Fp1', 'Fp2'],
    'Temporal': ['T7', 'T8'],
    'Central': ['Fz', 'Cz', 'FCz']
}

REGIONS_EGI = {
    'Frontal': ['E24', 'E124', 'E22', 'E9'],
    'Temporal': ['E45', 'E108'],
    'Central': ['E11', 'E129', 'E6']
}

def digitize_ts(ts, bins=N_BINS):
    try:
        return pd.qcut(ts, q=bins, labels=False, duplicates='drop').astype(np.int64)
    except:
        return np.zeros_like(ts, dtype=np.int64)

def shannon_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-10))

def H2(x, y):
    j = x * 100 + y
    return shannon_entropy(j)

def H3(x, y, z):
    j = x * 10000 + y * 100 + z
    return shannon_entropy(j)

def H4(w, x, y, z):
    j = w * 1000000 + x * 10000 + y * 100 + z
    return shannon_entropy(j)

def I2(x, y):
    return max(0.0, shannon_entropy(x) + shannon_entropy(y) - H2(x, y))

def calc_te(source, target, lag=TE_LAG):
    if len(source) <= lag: return 0.0
    sp = source[:-lag]
    tp = target[:-lag]
    tf = target[lag:]
    
    # TE(S->T) = I(Tf ; Sp | Tp) = H(Tf | Tp) - H(Tf | Tp, Sp)
    h_Tf_Tp = H2(tf, tp) - shannon_entropy(tp)
    h_Tf_TpSp = H3(tf, tp, sp) - H2(tp, sp)
    return max(0.0, h_Tf_Tp - h_Tf_TpSp)

def calc_conditional_te(source, target, cond, lag=TE_LAG):
    if len(source) <= lag: return 0.0
    sp = source[:-lag]
    tp = target[:-lag]
    cp = cond[:-lag]
    tf = target[lag:]
    
    # cTE = TE(S -> T | C) = I(Tf ; Sp | Tp, Cp) 
    # = H(Tf | Tp, Cp) - H(Tf | Tp, Cp, Sp)
    h_Tf_cd_TpCp = H3(tf, tp, cp) - H2(tp, cp)
    h_Tf_cd_TpCpSp = H4(tf, tp, cp, sp) - H3(tp, cp, sp)
    return max(0.0, h_Tf_cd_TpCp - h_Tf_cd_TpCpSp)

def calc_interaction_information(a, b, c):
    # II(A;B;C) = H(A) + H(B) + H(C) - H(A,B) - H(B,C) - H(A,C) + H(A,B,C)
    return (shannon_entropy(a) + shannon_entropy(b) + shannon_entropy(c)
            - H2(a, b) - H2(b, c) - H2(a, c) + H3(a, b, c))

def get_smooth_trajectory(ts, window=10):
    return np.convolve(ts, np.ones(window)/window, mode='valid')

def run_three_body():
    print("--- THE THREE-BODY PROBLEM (CINGULATE AUDIT) ---")
    lookup_db = pd.read_csv(FILE_LOOKUP)
    lookup_db['Subject_ID'] = lookup_db['Subject_ID'].astype(str)
    
    hc_phq9 = lookup_db[(lookup_db['Standardized_Severity']=='Healthy') & (lookup_db['Score_Type']=='PHQ-9')]['Raw_Score'].mean()
    hc_bdi = lookup_db[(lookup_db['Standardized_Severity']=='Healthy') & (lookup_db['Score_Type']=='BDI')]['Raw_Score'].mean()
    
    all_files = list((DIR_DATA / "Cavanagh/Depression_PS_Task/derivatives/epochs").glob("*.fif")) + \
                list((DIR_DATA / "MODMA/DDS-MODMA/derivatives/epochs").glob("*.fif"))
                
    records = []
    
    best_hc_usi, best_hc_orbit = 999.0, None
    worst_mdd_usi, worst_mdd_orbit = -999.0, None
    
    for count, f in enumerate(all_files):
        subj_name = f.name.split('-')[0].split('_')[0]
        subj_id = 'sub' if 'Cavanagh' in str(f) else subj_name
        clin = lookup_db[lookup_db['Subject_ID'] == subj_id]
        if clin.empty: continue
        clin = clin.iloc[0]
        score_type, raw_score, sev = clin['Score_Type'], clin['Raw_Score'], clin['Standardized_Severity']
        
        usi = (raw_score - hc_phq9)/(27.0 - hc_phq9) if score_type=='PHQ-9' else (raw_score - hc_bdi)/(63.0 - hc_bdi)
        
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        m_type = 'GSN-HydroCel-128' if 'E1' in epochs.ch_names else 'standard_1020'
        epochs.set_montage(m_type, match_case=False, on_missing='ignore')
        
        ica = mne.preprocessing.ICA(n_components=15, random_state=42, method='fastica', max_iter=200)
        ica.fit(epochs, verbose=False)
        try:
            ic_labels = label_components(epochs, ica, method='iclabel')
            drop_idx = [i for i, (lbl, prb) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])) if lbl in ['eye', 'heart'] and prb>0.8]
            ica.exclude = drop_idx
        except: pass
        
        ep_c = ica.apply(epochs.copy(), exclude=ica.exclude, verbose=False)
        regs = REGIONS_EGI if m_type == 'GSN-HydroCel-128' else REGIONS_1020
        
        ch_f = [ch for ch in regs['Frontal'] if ch in ep_c.ch_names]
        ch_t = [ch for ch in regs['Temporal'] if ch in ep_c.ch_names]
        ch_c = [ch for ch in regs['Central'] if ch in ep_c.ch_names]
        
        if not ch_f or not ch_t or not ch_c: continue
        
        v_A = ep_c.get_data()[:, [ep_c.ch_names.index(c) for c in ch_f], :].mean(axis=(0, 1)) # Collapse epochs to mean ERP for orbit
        v_B = ep_c.get_data()[:, [ep_c.ch_names.index(c) for c in ch_t], :].mean(axis=(0, 1))
        v_C = ep_c.get_data()[:, [ep_c.ch_names.index(c) for c in ch_c], :].mean(axis=(0, 1))
        
        # Raw long array for information metrics (all epochs concatenated)
        long_A = ep_c.get_data()[:, [ep_c.ch_names.index(c) for c in ch_f], :].mean(axis=1).flatten()
        long_B = ep_c.get_data()[:, [ep_c.ch_names.index(c) for c in ch_t], :].mean(axis=1).flatten()
        long_C = ep_c.get_data()[:, [ep_c.ch_names.index(c) for c in ch_c], :].mean(axis=1).flatten()
        
        bA, bB, bC = digitize_ts(long_A), digitize_ts(long_B), digitize_ts(long_C)
        
        # Math Implementation
        te_BA = calc_te(bB, bA, TE_LAG)
        cte_BA_C = calc_conditional_te(bB, bA, bC, TE_LAG)
        
        cond_factor = 0.0
        if te_BA > 1e-5:
            cond_factor = ((te_BA - cte_BA_C) / te_BA) * 100.0
            
        inter_info = calc_interaction_information(bA, bB, bC)
        
        records.append({
            'Subject': subj_name, 'USI': usi, 'Severity': sev,
            'TE_Insula_PFC': te_BA,
            'cTE_Conditioned_by_Cingulate': cte_BA_C,
            'Conditioning_Factor_Perc': cond_factor,
            'Interaction_Information_II': inter_info
        })
        
        print(f"[{count+1}/{len(all_files)}] Eval: USI={usi:.2f} | CondFact={cond_factor:.1f}% | II={inter_info:.3f}")
        
        # State tracking for Orbits
        if sev == 'Healthy' and usi < best_hc_usi:
            best_hc_usi, best_hc_orbit = usi, (get_smooth_trajectory(v_A), get_smooth_trajectory(v_B), get_smooth_trajectory(v_C))
        if sev == 'Severe MDD' and usi > worst_mdd_usi:
            worst_mdd_usi, worst_mdd_orbit = usi, (get_smooth_trajectory(v_A), get_smooth_trajectory(v_B), get_smooth_trajectory(v_C))
            
    df = pd.DataFrame(records)
    df.to_csv(DIR_ASSETS / "Three_Body_Orbit_Decay.csv", index=False)
    
    # === PLOT PHASE SPACE ===
    if best_hc_orbit and worst_mdd_orbit:
        fig = plt.figure(figsize=(14, 6))
        
        # HC Axis
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(best_hc_orbit[0], best_hc_orbit[1], best_hc_orbit[2], color='teal', linewidth=1.5, alpha=0.8)
        ax1.set_title(f'Harmonic Orbit (Healthy Control)\nUSI: {best_hc_usi:.2f}', fontweight='bold')
        ax1.set_xlabel('PFC Transducer (A)')
        ax1.set_ylabel('Insula Transducer (B)')
        ax1.set_zlabel('sgACC Cingulate (C)')
        
        # Severe MDD Axis
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(worst_mdd_orbit[0], worst_mdd_orbit[1], worst_mdd_orbit[2], color='maroon', linewidth=1.5, alpha=0.8)
        ax2.set_title(f'Chaotic Path (Severe MDD)\nUSI: {worst_mdd_usi:.2f}', fontweight='bold')
        ax2.set_xlabel('PFC Transducer (A)')
        ax2.set_ylabel('Insula Transducer (B)')
        ax2.set_zlabel('sgACC Cingulate (C)')
        
        plt.tight_layout()
        plt.savefig(DIR_ASSETS / "Figure_3Body_Phase_Space.png", dpi=300)
        plt.close()
    
    print("[+] 3-Body Problem Simulation Complete.")

if __name__ == "__main__":
    run_three_body()
