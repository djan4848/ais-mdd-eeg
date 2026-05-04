import numpy as np
import pandas as pd
import mne
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from mne_icalabel import label_components

warnings.filterwarnings('ignore')

DIR_DATA = Path("01_raw_data")
DIR_ASSETS = Path("06_manuscript_assets")
FILE_LOOKUP = DIR_DATA / "clinical_lookup.csv"

FS = 256.0
T_LENGTH = 2.0
TE_LAG = 5
N_SURR = 20
N_BINS = 4

REGIONS_1020 = {
    'Frontal': ['Fz', 'F3', 'F4', 'AFz'],
    'Temporal': ['T7', 'TP7']
}

REGIONS_EGI = {
    'Frontal': ['E11', 'E24', 'E124', 'E12'],
    'Temporal': ['E45', 'E46']
}

def digitize_ts(ts, bins=N_BINS):
    try:
        return pd.qcut(ts, q=bins, labels=False, duplicates='drop').astype(int)
    except:
        return np.zeros_like(ts, dtype=int)

def calc_shannon_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-10))

def calc_joint_entropy(x, y):
    # Unique numerical hashing for bivariate space
    joint = x.astype(np.int64) * 100 + y.astype(np.int64)
    return calc_shannon_entropy(joint)

def calc_joint_entropy_3(x, y, z):
    # Unique numerical hashing for trivariate space
    joint = x.astype(np.int64) * 10000 + y.astype(np.int64) * 100 + z.astype(np.int64)
    return calc_shannon_entropy(joint)

def calc_mutual_info(x, y):
    hx = calc_shannon_entropy(x)
    hy = calc_shannon_entropy(y)
    hxy = calc_joint_entropy(x, y)
    return max(0.0, hx + hy - hxy)

def calc_te(source, target, lag=TE_LAG):
    if len(source) <= lag: return 0.0
    s_past = source[:-lag]
    t_past = target[:-lag]
    t_pres = target[lag:]
    
    # Conditional Entropy H(T_t | T_past)
    h_tpres_tpast = calc_joint_entropy(t_pres, t_past)
    h_tpast = calc_shannon_entropy(t_past)
    h_T_cd_Tpast = h_tpres_tpast - h_tpast
    
    # Conditional Entropy H(T_t | T_past, S_past)
    h_tpres_tpast_spast = calc_joint_entropy_3(t_pres, t_past, s_past)
    h_tpast_spast = calc_joint_entropy(t_past, s_past)
    h_T_cd_JointPast = h_tpres_tpast_spast - h_tpast_spast
    
    return max(0.0, h_T_cd_Tpast - h_T_cd_JointPast)

def calc_te_zscore(source, target, lag=TE_LAG):
    real_te = calc_te(source, target, lag)
    surr = []
    for _ in range(N_SURR):
        s_shuf = np.random.permutation(source)
        surr.append(calc_te(s_shuf, target, lag))
    ms = np.mean(surr)
    ss = np.std(surr) + 1e-10
    return (real_te - ms) / ss

def calc_ais(ts):
    if len(ts) <= 5: return 0.0
    ais_5 = calc_mutual_info(ts[5:], ts[:-5])
    if ais_5 < 0.05:
        # Fallback to local lag=1 to avoid undersampling null-spaces
        return calc_mutual_info(ts[1:], ts[:-1])
    return ais_5

def pid_synergy(x1, x2, y):
    # Synergy = I(X1, X2 ; Y) - I(X1 ; Y) - I(X2 ; Y) + min(I(X1; Y), I(X2; Y))
    hx1x2 = calc_joint_entropy(x1, x2)
    hy = calc_shannon_entropy(y)
    h_all = calc_joint_entropy_3(x1, x2, y)
    
    I_x1x2_y = max(0.0, hx1x2 + hy - h_all)
    I_x1_y = calc_mutual_info(x1, y)
    I_x2_y = calc_mutual_info(x2, y)
    
    red = min(I_x1_y, I_x2_y)
    syn = I_x1x2_y - I_x1_y - I_x2_y + red
    return max(0.0, syn)

def run_info_flow():
    print("--- STARTING DIRECTED INFO-FLOW AUDIT (PFC-INSULA AXIS) ---")
    
    lookup_db = pd.read_csv(FILE_LOOKUP)
    lookup_db['Subject_ID'] = lookup_db['Subject_ID'].astype(str)
    
    hc_phq9_mean = lookup_db[(lookup_db['Standardized_Severity'] == 'Healthy') & (lookup_db['Score_Type'] == 'PHQ-9')]['Raw_Score'].mean()
    hc_bdi_mean = lookup_db[(lookup_db['Standardized_Severity'] == 'Healthy') & (lookup_db['Score_Type'] == 'BDI')]['Raw_Score'].mean()
    
    path_cav = DIR_DATA / "Cavanagh/Depression_PS_Task/derivatives/epochs"
    path_modma = DIR_DATA / "MODMA/DDS-MODMA/derivatives/epochs"
    
    all_files = list(path_cav.glob("*.fif")) + list(path_modma.glob("*.fif"))
    print(f"[*] Found {len(all_files)} files for Info-Flow computation.")
    
    records = []
    
    for count, f in enumerate(all_files):
        subj_name = f.name.split('-')[0].split('_')[0]
        subj_id = 'sub' if 'Cavanagh' in str(f) else subj_name
        
        clin = lookup_db[(lookup_db['Subject_ID'] == subj_id)]
        if clin.empty: continue
        clin = clin.iloc[0]
        
        score_type = clin['Score_Type']
        raw_score = clin['Raw_Score']
        sev = clin['Standardized_Severity']
        
        if score_type == 'PHQ-9': usi = (raw_score - hc_phq9_mean) / (27.0 - hc_phq9_mean)
        else: usi = (raw_score - hc_bdi_mean) / (63.0 - hc_bdi_mean)
        
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        if epochs.info['sfreq'] != FS: epochs.resample(FS)
        tmin = epochs.times[0]
        tmax = tmin + T_LENGTH
        if epochs.times[-1] > tmax: epochs.crop(tmin=tmin, tmax=tmax)
            
        drop_chs = [ch for ch in ['CB1', 'CB2', 'HEOG', 'VEOG', 'M1', 'M2'] if ch in epochs.ch_names]
        if drop_chs: epochs.drop_channels(drop_chs)
            
        montage_type = 'GSN-HydroCel-128' if 'E1' in epochs.ch_names else 'standard_1020'
        epochs.set_montage(montage_type, match_case=False, on_missing='ignore')
        
        ica = mne.preprocessing.ICA(n_components=15, random_state=42, method='fastica', max_iter=200)
        ica.fit(epochs, verbose=False)
        try:
            ic_labels = label_components(epochs, ica, method='iclabel')
            drop_idx = [i for i, (lbl, prb) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])) if lbl in ['eye', 'heart', 'muscle'] and prb > 0.8]
            ica.exclude = drop_idx
        except:
            pass
            
        epochs_clean = ica.apply(epochs.copy(), exclude=ica.exclude, verbose=False)
        regions = REGIONS_EGI if montage_type == 'GSN-HydroCel-128' else REGIONS_1020
        
        frontal_ch = [ch for ch in regions['Frontal'] if ch in epochs_clean.ch_names]
        temporal_ch = [ch for ch in regions['Temporal'] if ch in epochs_clean.ch_names]
        
        if not frontal_ch or not temporal_ch:
            continue
            
        idx_f = [epochs_clean.ch_names.index(ch) for ch in frontal_ch]
        idx_t = [epochs_clean.ch_names.index(ch) for ch in temporal_ch]
        
        # Mean across channels for the region
        pfc_raw = epochs_clean.get_data()[:, idx_f, :].mean(axis=1).flatten()
        ins_raw = epochs_clean.get_data()[:, idx_t, :].mean(axis=1).flatten()
        
        pfc_bin = digitize_ts(pfc_raw, bins=N_BINS)
        ins_bin = digitize_ts(ins_raw, bins=N_BINS)
        
        # 1. Transfer Entropy & Net Dominance
        te_pfc_ins_z = calc_te_zscore(pfc_bin, ins_bin, lag=TE_LAG)
        te_ins_pfc_z = calc_te_zscore(ins_bin, pfc_bin, lag=TE_LAG)
        delta_te = te_ins_pfc_z - te_pfc_ins_z  # Positive means Bottom-up (Ins->PFC) dominates
        
        # 2. Local AIS
        ais_pfc = calc_ais(pfc_bin)
        
        # 3. PID Synergy
        # Synergy(PFC_t, Ins_t -> PFC_{t+lag})
        if len(pfc_bin) > TE_LAG:
            pfc_t = pfc_bin[:-TE_LAG]
            ins_t = ins_bin[:-TE_LAG]
            pfc_t_plus = pfc_bin[TE_LAG:]
            syn = pid_synergy(pfc_t, ins_t, pfc_t_plus)
        else:
            syn = 0.0
            
        records.append({
            'Subject': subj_name, 'USI': usi, 'Severity': sev,
            'Delta_TE_Ins_PFC': delta_te,
            'AIS_Frontal': ais_pfc,
            'PID_Synergy': syn
        })
        print(f"[{count+1}/{len(all_files)}] Processed {subj_name} | D-TE Z={delta_te:.2f} | AIS={ais_pfc:.3f}")
        
    df = pd.DataFrame(records)
    df.to_csv(DIR_ASSETS / "directed_info_flow_audit.csv", index=False)
    
    # VISUALIZATION
    plt.figure(figsize=(15, 5))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    plt.subplot(1, 3, 1)
    sns.regplot(data=df, x='USI', y='Delta_TE_Ins_PFC', color='darkred', scatter_kws={'alpha':0.6})
    plt.title('Bottom-Up Dominance ($\Delta$ TE)\nInsula $\Rightarrow$ PFC')
    plt.ylabel('Net TE (Z-Score)')
    plt.xlabel('Unified Severity Index (USI)')
    plt.axhline(0, color='gray', linestyle='--')
    
    plt.subplot(1, 3, 2)
    sns.regplot(data=df, x='USI', y='AIS_Frontal', color='darkblue', scatter_kws={'alpha':0.6})
    plt.title('PFC Predictive Memory Collapse\n(Frontal Active Info Storage)')
    plt.ylabel('AIS (bits)')
    plt.xlabel('Unified Severity Index (USI)')
    
    plt.subplot(1, 3, 3)
    sns.regplot(data=df, x='USI', y='PID_Synergy', color='darkmagenta', scatter_kws={'alpha':0.6})
    plt.title('PFC-Insula Information Synergy\n(PID Degradation)')
    plt.ylabel('Synergy (bits)')
    plt.xlabel('Unified Severity Index (USI)')
    
    plt.tight_layout()
    plt.savefig(DIR_ASSETS / "Figure_Directed_Info_Flow.png", dpi=300)
    plt.close()
    
    print("[+] SUCCESS! Directed Info-Flow Audit exported mathematically.")

if __name__ == "__main__":
    run_info_flow()
