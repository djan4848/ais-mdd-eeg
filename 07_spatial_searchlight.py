import numpy as np
import pandas as pd
import mne
import warnings
from pathlib import Path
from scipy.signal import welch
from specparam import SpectralModel
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mne_icalabel import label_components
import time

warnings.filterwarnings('ignore')

DIR_DATA = Path("01_raw_data")
DIR_ASSETS = Path("06_manuscript_assets")
FILE_LOOKUP = DIR_DATA / "clinical_lookup.csv"

FS = 256.0
T_LENGTH = 2.0
FREQ_RANGE = [2.0, 40.0]

REGIONS_1020 = {
    'Frontal': ['Fz', 'F3', 'F4', 'AFz'],
    'Central': ['Cz', 'C3', 'C4'],
    'Parietal': ['Pz', 'P3', 'P4', 'POz'],
    'L-Temporal': ['T7', 'TP7'],
    'R-Temporal': ['T8', 'TP8']
}

REGIONS_EGI = {
    'Frontal': ['E11', 'E24', 'E124', 'E12'],
    'Central': ['E129', 'E36', 'E104'],
    'Parietal': ['E62', 'E52', 'E92', 'E72'],
    'L-Temporal': ['E45', 'E46'],
    'R-Temporal': ['E108', 'E102']
}

def lempel_ziv_complexity(binary_sequence):
    s = "".join(map(str, binary_sequence))
    n = len(s)
    if n <= 1: return 0
    i, c, l = 0, 1, 1
    while i + l <= n:
        if s[i:i+l] not in s[:i]:
            c += 1
            l = 1
            i += 1
        else:
            l += 1
    return c / (n / np.log2(n))

def calc_var_crash(data_2d):
    try:
        if data_2d.shape[1] < 2: return 1
        model = VAR(data_2d)
        res = model.fit(maxlags=1)
        return 0
    except:
        return 1

def run_spatial_searchlight():
    print("--- STARTING SPATIAL SEARCHLIGHT (BLIND SINGULARITY SWEEP) ---")
    
    lookup_db = pd.read_csv(FILE_LOOKUP)
    lookup_db['Subject_ID'] = lookup_db['Subject_ID'].astype(str)
    
    # Precompute baseline HC thresholds for USI mappings
    hc_phq9_mean = lookup_db[(lookup_db['Standardized_Severity'] == 'Healthy') & (lookup_db['Score_Type'] == 'PHQ-9')]['Raw_Score'].mean()
    hc_bdi_mean = lookup_db[(lookup_db['Standardized_Severity'] == 'Healthy') & (lookup_db['Score_Type'] == 'BDI')]['Raw_Score'].mean()
    
    path_cav = DIR_DATA / "Cavanagh/Depression_PS_Task/derivatives/epochs"
    path_modma = DIR_DATA / "MODMA/DDS-MODMA/derivatives/epochs"
    
    all_files = list(path_cav.glob("*.fif")) + list(path_modma.glob("*.fif"))
    print(f"[*] Found {len(all_files)} brains for spatial decomposition.")
    
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
        
        res = {'Subject': subj_name, 'USI': usi, 'Severity': sev, 'System': montage_type}
        valid_regions = 0
        
        for r_name, r_chs in regions.items():
            avail = [ch for ch in r_chs if ch in epochs_clean.ch_names]
            if len(avail) < 1:
                res[f'{r_name}_R2'] = np.nan
                res[f'{r_name}_LZC'] = np.nan
                res[f'{r_name}_VAR'] = np.nan
                continue
                
            idx = [epochs_clean.ch_names.index(ch) for ch in avail]
            data_3d = epochs_clean.get_data()[:, idx, :]
            
            vsens = data_3d.mean(axis=1)
            
            # spectral R2
            nperseg = int(2*FS) if vsens.shape[-1] >= int(2*FS) else vsens.shape[-1]
            freqs, psds = welch(vsens, FS, nperseg=nperseg, axis=-1)
            fm = SpectralModel(peak_width_limits=[1, 8], min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
            fm.fit(freqs, psds.mean(axis=0), FREQ_RANGE)
            res[f'{r_name}_R2'] = fm.get_metrics('gof', 'rsquared')
            
            # LZC: Cap to 5 epochs (approx 2560 samples) to prevent exponential CPU hang
            max_eps = min(5, len(vsens))
            flat = vsens[:max_eps].flatten()
            if len(flat) > 0:
                binary = (flat > np.median(flat)).astype(int)
                res[f'{r_name}_LZC'] = lempel_ziv_complexity(binary)
            else:
                res[f'{r_name}_LZC'] = np.nan
            
            # VAR Crash
            if len(avail) >= 2:
                flat_multivar = np.swapaxes(data_3d, 1, 2).reshape(-1, len(avail))
                res[f'{r_name}_VAR'] = calc_var_crash(flat_multivar)
            else:
                res[f'{r_name}_VAR'] = 1
                
            valid_regions += 1
            
        if valid_regions == len(regions):
            records.append(res)
            print(f"[{count+1}/{len(all_files)}] Processed {subj_name} | USI: {usi:.2f}")
            
    df = pd.DataFrame(records)
    print(f"\n[*] Extracted topological features for {len(df)} subjects. Proceeding to Unsupervised Classification.")
    
    # Extract only the R2 features for the unsupervised clustering
    feature_cols = [f'{r}_R2' for r in ['Frontal', 'Central', 'Parietal', 'L-Temporal', 'R-Temporal']]
    df_clean = df.dropna(subset=feature_cols).copy()
    
    # Standardize
    X = StandardScaler().fit_transform(df_clean[feature_cols])
    
    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    df_clean['Blind_Cluster_k2'] = kmeans.fit_predict(X)
    
    # Feature Importance via Cluster Separation Capability (Variance logic)
    centroids = kmeans.cluster_centers_
    separations = np.abs(centroids[0] - centroids[1])
    most_important = feature_cols[np.argmax(separations)]
    
    print("\n--- NO-LABEL TEST RESULTS ---")
    print("Blind Clustering was executed strictly on spectral integrity layout across the topology.")
    print(f"Most prominent structural variance between clusters identified at: {most_important}")
    for c in [0, 1]:
        subset = df_clean[df_clean['Blind_Cluster_k2'] == c]
        mean_usi = subset['USI'].mean()
        maj_sev = subset['Severity'].mode()[0]
        print(f"Cluster {c}: Mean USI={mean_usi:.2f} | Majority Class={maj_sev}")
        
    df_clean.to_csv(DIR_ASSETS / "spatial_searchlight_results.csv", index=False)
    
    # HEATMAP GENERATION ========================
    # Sort subjects by Unified Severity Index
    df_sorted = df_clean.sort_values(by='USI', ascending=False)
    heatmap_mat = df_sorted[feature_cols].values
    
    # Y-labels formatting for clean visualization
    severity_labels = df_sorted['Severity'].values
    y_ticks_idx = np.arange(0, len(severity_labels), max(1, len(severity_labels)//15))
    y_tick_labels = [severity_labels[i] for i in y_ticks_idx]
    
    plt.figure(figsize=(10, 12))
    sns.set_theme(style="white", font_scale=1.1)
    
    # Color bar correlates to predictability (Higher R2 = Better determinism)
    ax = sns.heatmap(heatmap_mat, cmap='viridis', cbar_kws={'label': r'$1/f$ Spectral Determinism ($R^2$)'},
                     xticklabels=[r.split('_')[0] for r in feature_cols], yticklabels=False, vmin=0, vmax=1.0)
                     
    ax.set_yticks(y_ticks_idx)
    ax.set_yticklabels(y_tick_labels, rotation=0)
    
    plt.xlabel('Cortical Spatial Clusters', fontweight='bold', fontsize=14)
    plt.ylabel('Subjects (Descending USI - Max Severity to Health)', fontweight='bold', fontsize=14)
    plt.title('The Singularity Map\nPhase Transition Across Distal Topologies', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(DIR_ASSETS / "The_Singularity_Map.png", dpi=300)
    plt.close()
    print("[+] SUCCESS! Spatial logic completed and The Singularity Map generated.")

if __name__ == "__main__":
    run_spatial_searchlight()
