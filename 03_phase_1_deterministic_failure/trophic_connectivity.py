import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.api import VAR
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# 1. CORE MATH: EFFECTIVE CONNECTIVITY (VAR) & TROPHIC LEVELS
# -------------------------------------------------------------------

def compute_effective_connectivity(data_array, p=1):
    """
    Fits a VAR(p) model on a (T_trials, N_ROIs) continuous matrix.
    Returns W matrix of shape (ROIs, ROIs), where W[i,j] is the 
    Granger causal influence from ROI_i -> ROI_j.
    """
    if len(data_array) < 20: 
        return None  # Too few trials
        
    try:
        model = VAR(data_array)
        # Suppress warnings if maxlags is close to data length, though p=1 is safe
        results = model.fit(maxlags=p, ic=None, trend='c')
        
        # results.coefs shape is (p, N_ROIs, N_ROIs)
        # A_ij describes how variable j at lag p affects variable i at t=0.
        # So influence J -> I is coefs[0, i, j].
        A = results.coefs[0]
        
        # We want W[i,j] = influence from I to J.
        # Therefore: W[i, j] = |A_ji|.
        W = np.abs(A).T
        
        # Clear exact self-loops for network analysis if desired, but trophic logic can use them.
        # We will zero out the diagonal for cleaner graph metrics:
        np.fill_diagonal(W, 0)
        return W
        
    except Exception as e:
        print(f"[!] VAR model failed: {e}")
        return None

def compute_trophic_levels(W):
    """
    Computes Trophic Level (Deco & Kringelbach, 2024).
    W[i,j] = out-flow from i to j.
    """
    # 1. Degrees
    out_deg = np.sum(W, axis=1) # Outflow from node i
    in_deg  = np.sum(W, axis=0) # Inflow to node i
    
    # 2. Trophic Level
    # Nodes with high in_deg are sinks (T ~ 2.0). Nodes with high out_deg are sources (T ~ 0.0)
    eps = 1e-9
    T = 1.0 + (in_deg - out_deg) / (in_deg + out_deg + eps)
    
    return T, in_deg, out_deg

def compute_network_metrics(W):
    """
    Computes NetworkX global directed efficiency and weighted clustering.
    """
    try:
        # Create a directed graph from the adjacency matrix W
        G = nx.from_numpy_array(W, create_using=nx.DiGraph)
        
        # Efficiency (using inverse weights for shortest path distance approximation)
        # NetworkX calculates global efficiency by default on unweighted graphs, 
        # or we can write a custom weighted shortest path loop. For broad robustness,
        # we threshold weak connections or just use standard efficiency over the 3-node graph.
        
        # Weighted clustering (directed) works natively in nx
        cl_dict = nx.clustering(G, weight='weight')
        clustering = np.nanmean(list(cl_dict.values()))
        
        # Global efficiency is not implemented for directed graphs in networkx
        # We approximate topological integration by casting to undirected structure
        global_eff = nx.global_efficiency(G.to_undirected())
        
        return global_eff, clustering
    except Exception as e:
        print(f"NetworkX error: {e}")
        return np.nan, np.nan

# -------------------------------------------------------------------
# 2. DATA EXTRACTION
# -------------------------------------------------------------------

def get_roi_trial_matrix(csv_path, roi_list, time_window=(200, 400), condition=None):
    """
    Converts raw TS data into a (Trials x ROIs) matrix per subject.
    Extracts power proxy (variance within latency window).
    """
    print(f"Reading: {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    if condition:
        df = df[df['cond'] == condition]
    
    # Filter for given ROIs
    df = df[df['roi'].isin(roi_list)]
    
    # Filter time latency
    df = df[(df['time_ms'] >= time_window[0]) & (df['time_ms'] <= time_window[1])]
    
    # Calculate power proxy per trial
    power_df = df.groupby(['subject', 'cond', 'trial', 'roi'])['value'].var().reset_index()
    
    subject_matrices = {}
    
    for subj in power_df['subject'].unique():
        subj_data = power_df[power_df['subject'] == subj]
        # Pivot table to shape: Trials (rows) x ROIs (columns)
        pivot = subj_data.pivot_table(index='trial', columns='roi', values='value').dropna()
        
        if len(pivot) >= 20: 
            # Ensure column order matches roi_list
            valid = True
            for r in roi_list:
                if r not in pivot.columns:
                    valid = False
            if valid:
                mat = pivot[roi_list].values
                # Standardize to prevent VAR underflow on 10^-22 variance arrays
                mat = (mat - np.mean(mat, axis=0)) / (np.std(mat, axis=0) + 1e-12)
                subject_matrices[subj] = mat
                
    return subject_matrices

def load_cavanagh(root_path):
    root = Path(root_path)
    ts_file = root / "DDS-ds005456/derivatives/trial_roi_timeseries/trial_roi_timeseries.csv"
    clin_file = root / "DDS-ds005456/derivatives/cavanagh_clinical.csv"
    
    if not ts_file.exists(): return None, None, None
        
    roi_list = ['vmPFC', 'DMN', 'ECN']
    subj_mats = get_roi_trial_matrix(ts_file, roi_list, time_window=(200,400))
    
    clin = pd.read_csv(clin_file)
    clin.rename(columns={'Subject': 'subject'}, inplace=True)
    return subj_mats, clin, roi_list

def load_modma(root_path):
    root = Path(root_path)
    ts_file = root / "DDS-MODMA/derivatives/trial_roi_timeseries/trial_roi_timeseries.csv"
    clin_file = root / "EEG_128channels_ERP_lanzhou_2015/subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx"
    
    if not ts_file.exists(): return None, None, None
        
    roi_list = ['cACC', 'DMN', 'ECN']
    subj_mats = get_roi_trial_matrix(ts_file, roi_list, time_window=(200,400))
    
    clin = pd.read_excel(clin_file)
    clin['subject'] = clin['subject id'].astype(str).str.zfill(8)
    clin['Group'] = clin['PHQ-9'].apply(lambda x: "Severe" if x >= 15 else "Healthy" if x <= 5 else "Mild")
    return subj_mats, clin, roi_list


# -------------------------------------------------------------------
# 3. PIPELINE & VISUALIZATION
# -------------------------------------------------------------------

def process_dataset(subj_mats, clin, roi_list, clinical_col, group_col, target_roi, dataset_name, out_dir):
    print(f"\n=============================================")
    print(f" TROPHIC LEVELS ANALYSIS: {dataset_name}")
    print(f" ROIs: {roi_list}  | Target ROI: {target_roi}")
    print(f"=============================================")
    
    results = []
    matrix_store = {'Healthy': [], 'Severe': []} # For group average heatmaps
    
    for subj, data_matrix in subj_mats.items():
        clin_row = clin[clin['subject'] == subj]
        if clin_row.empty: continue
            
        score = clin_row.iloc[0][clinical_col]
        group = clin_row.iloc[0][group_col]
        
        # A) Effective Connectivity -> Adjacency Matrix W
        W = compute_effective_connectivity(data_matrix, p=1)
        if W is None: continue
            
        if group in ['Healthy', 'Severe']:
            matrix_store[group].append(W)
            
        # B) Trophic Levels Deco
        T_array, _, _ = compute_trophic_levels(W)
        trophic_coherence = np.var(T_array)
        mean_trophic = np.mean(T_array)
        
        # C) Network Metrics
        eff, clu = compute_network_metrics(W)
        
        row = {
            'subject': subj,
            'group': group,
            'clinical_score': score,
            'trophic_coherence': trophic_coherence,
            'mean_trophic': mean_trophic,
            'global_eff': eff,
            'clustering': clu
        }
        
        # Add pure ROIs Trophic metrics
        for i, r in enumerate(roi_list):
            row[f'T_{r}'] = T_array[i]
            
        results.append(row)
        
    df_res = pd.DataFrame(results)
    
    # ----- 4. GROUP COMPARISONS (T-TESTS) -----
    if not df_res.empty:
        healthy = df_res[df_res['group'] == 'Healthy']
        severe = df_res[df_res['group'] == 'Severe'] # Exclude 'Mild' or intermediate for clean extremes
        if severe.empty: severe = df_res[df_res['group'] != 'Healthy'] # fallback
            
        metrics = ['trophic_coherence', 'global_eff'] + [f'T_{r}' for r in roi_list]
        print("\n--- Group T-Tests (Healthy vs Severely Depressed) ---")
        for m in metrics:
            if healthy.empty or severe.empty: continue
            t, p = stats.ttest_ind(healthy[m].dropna(), severe[m].dropna(), equal_var=False)
            d = (healthy[m].mean() - severe[m].mean()) / np.sqrt((healthy[m].std()**2 + severe[m].std()**2)/2 + 1e-9)
            print(f"> {m}: T={t:.3f}, p={p:.3f}, Cohen's d={d:.3f}")
            
    # ----- 5. PLOTS -----
    sns.set_theme(style="whitegrid")
    
    # FILTER FOR CLEAN PLOTS
    plot_df = df_res[df_res['group'].isin(['Healthy', 'Severe'])]
    if plot_df.empty: plot_df = df_res
        
    # Fig 1: Bar Plots of T_ROI
    fig, axes = plt.subplots(1, len(roi_list), figsize=(14, 5))
    if len(roi_list) == 1: axes = [axes]
    
    for idx, r in enumerate(roi_list):
        sns.barplot(data=plot_df, x='group', y=f'T_{r}', ax=axes[idx], palette='muted', capsize=.1)
        axes[idx].set_title(f"Trophic Level: {r}")
        axes[idx].set_ylabel("T Level (0=Source, 2=Sink)")
    plt.suptitle(f"Fig 1: Trophic Hierarchy Shifts by Group ({dataset_name})")
    plt.tight_layout()
    plt.savefig(out_dir / f"Fig1_TrophicLevels_{dataset_name}.png", dpi=300)
    plt.close()
    
    # Fig 2: Scatter / Regression (Target ROI Trophic vs Clinical)
    md_df = df_res[df_res['group'] != 'Healthy']
    plt.figure(figsize=(7, 6))
    sns.regplot(data=md_df, x=f'T_{target_roi}', y='clinical_score', color='crimson')
    plt.title(f"Fig 2: Severe MDD correlation - Clinical vs {target_roi} Trophic")
    plt.xlabel(f"Trophic Level ({target_roi})")
    plt.ylabel(f"Severity ({clinical_col})")
    
    # Calculate Spearman
    if not md_df.empty:
        rho, p = stats.spearmanr(md_df[f'T_{target_roi}'], md_df['clinical_score'])
        plt.annotate(f"Spearman rho = {rho:.3f}\np={p:.3f}", xy=(0.05, 0.95), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"))
        
    plt.tight_layout()
    plt.savefig(out_dir / f"Fig2_Scatter_{dataset_name}.png", dpi=300)
    plt.close()
    
    # Fig 3: Boxplots of Global Topology
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=plot_df, x='group', y='trophic_coherence', ax=axes[0], palette='pastel')
    sns.swarmplot(data=plot_df, x='group', y='trophic_coherence', ax=axes[0], color=".25")
    axes[0].set_title("Trophic Coherence (Structural Order)")
    
    sns.boxplot(data=plot_df, x='group', y='global_eff', ax=axes[1], palette='pastel')
    sns.swarmplot(data=plot_df, x='group', y='global_eff', ax=axes[1], color=".25")
    axes[1].set_title("Directed Global Efficiency")
    
    plt.suptitle(f"Fig 3: Brain Network Topology ({dataset_name})")
    plt.tight_layout()
    plt.savefig(out_dir / f"Fig3_Topology_{dataset_name}.png", dpi=300)
    plt.close()
    
    # Fig 4: Average Directed Adjacency Heatmaps W
    if matrix_store['Healthy'] and matrix_store['Severe']:
        W_H = np.mean(matrix_store['Healthy'], axis=0)
        W_S = np.mean(matrix_store['Severe'], axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.heatmap(W_H, annot=True, xticklabels=roi_list, yticklabels=roi_list, cmap="Blues", ax=axes[0])
        axes[0].set_title(f"Healthy: Av Granger Flow")
        axes[0].set_xlabel("Target (Inflow to)")
        axes[0].set_ylabel("Source (Outflow from)")
        
        sns.heatmap(W_S, annot=True, xticklabels=roi_list, yticklabels=roi_list, cmap="Reds", ax=axes[1])
        axes[1].set_title(f"Severe MDD: Av Granger Flow")
        axes[1].set_xlabel("Target (Inflow to)")
        
        plt.suptitle(f"Fig 4: Hierarchical Effective Connectivity (VAR[1])")
        plt.tight_layout()
        plt.savefig(out_dir / f"Fig4_Heatmap_{dataset_name}.png", dpi=300)
        plt.close()


def run():
    p_cav = "/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds005356"
    p_modma = "/media/neuraldyn/PortableSSD/DEPRESSION/MODMA"
    
    out_dir = Path("/media/neuraldyn/PortableSSD/DEPRESSION/UNIVERSAL_BIFURCATION_LAB/trophic_results")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n[+] Starting Hierarchical Effective Connectivity Engine (Deco & Kringelbach) [+]")
    
    # Cavanagh
    subj_c, clin_c, rois_c = load_cavanagh(p_cav)
    if subj_c:
        process_dataset(subj_c, clin_c, rois_c, 'SHAPS', 'Group_SHAPS', 'vmPFC', 'CAVANAGH_MEG', out_dir)
        
    # MODMA
    subj_m, clin_m, rois_m = load_modma(p_modma)
    if subj_m:
        process_dataset(subj_m, clin_m, rois_m, 'PHQ-9', 'Group', 'cACC', 'MODMA_EEG', out_dir)
        
    print("\n[✔] Trophic Levels Extracted and Rendered successfully. Assets written to SSD.")

if __name__ == "__main__":
    run()
