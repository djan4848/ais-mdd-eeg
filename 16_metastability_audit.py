import mne
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

warnings.filterwarnings('ignore')

DIR_ASSETS = Path("06_manuscript_assets")
DIR_TDBRAIN = Path("/media/neuraldyn/PortableSSD/TDBRAIN_derivatives")

# --- KSG-kNN Core (k=3 for N=500) ---
def ksg_entropy(X, k=3):
    if X.ndim == 1: X = X[:, None]
    N, d = X.shape
    rng = np.random.default_rng(seed=42)
    # Essential Jitter for exact coincident points zero-distances
    X = X + 1e-10 * rng.standard_normal(X.shape)
    
    tree = cKDTree(X)
    distances, _ = tree.query(X, k=k+1, p=np.inf)
    eps = np.maximum(distances[:, k], 1e-15)
    ans = digamma(N) - digamma(k) + np.log(2**d) + (d / N) * np.sum(np.log(eps))
    return ans / np.log(2)

def compute_o_info(A, B, C, D, k=3):
    H_A, H_B = ksg_entropy(A, k), ksg_entropy(B, k)
    H_C, H_D = ksg_entropy(C, k), ksg_entropy(D, k)
    H_BCD = ksg_entropy(np.column_stack((B, C, D)), k)
    H_ACD = ksg_entropy(np.column_stack((A, C, D)), k)
    H_ABD = ksg_entropy(np.column_stack((A, B, D)), k)
    H_ABC = ksg_entropy(np.column_stack((A, B, C)), k)
    H_All = ksg_entropy(np.column_stack((A, B, C, D)), k)
    
    return 2 * H_All + (H_A + H_B + H_C + H_D) - (H_BCD + H_ACD + H_ABD + H_ABC)

def extract_kinetics(states, slide_s=1.0):
    if len(states) == 0:
        return 0, 0, 0, 0
        
    transitions = np.diff((states > 0).astype(int))
    flips = np.sum(np.abs(transitions))
    mins = (len(states) * slide_s) / 60.0
    lability = flips / mins if mins > 0 else 0
    
    # State runs logic
    r_runs = []
    current_r_len = 0
    for s in states:
        if s > 0: # State R
            current_r_len += 1
        else:
            if current_r_len > 0:
                r_runs.append(current_r_len)
                current_r_len = 0
    if current_r_len > 0:
        r_runs.append(current_r_len)
        
    mean_dwell = (np.mean(r_runs) * slide_s) if r_runs else 0
    
    n_S, n_R = np.sum(states < 0), np.sum(states > 0)
    S_to_R = np.sum(transitions == 1)
    R_to_S = np.sum(transitions == -1)
    
    p_s_to_r = S_to_R / n_S if n_S > 0 else 0
    p_r_to_s = R_to_S / n_R if n_R > 0 else 0
    
    return mean_dwell, p_s_to_r, p_r_to_s, lability

def run_metastability_tdbrain():
    print("--- META-STATE TRANSITION AUDIT (TDBRAIN) ---")
    
    clin_tdbrain = pd.read_csv(DIR_TDBRAIN / "TDBRAIN_participants_V2.tsv", sep='\t')
    df = clin_tdbrain[clin_tdbrain['indication'].isin(['HEALTHY', 'MDD'])].copy()
    
    df['BDI_pre'] = pd.to_numeric(df['BDI_pre'], errors='coerce')
    hc_bdi = df[df['indication'] == 'HEALTHY']['BDI_pre'].mean()
    if pd.isna(hc_bdi): hc_bdi = 2.0
    
    df['USI'] = (df['BDI_pre'] - hc_bdi) / (63.0 - hc_bdi)
    df = df.dropna(subset=['participants_ID']).drop_duplicates(subset=['participants_ID'])
    
    SFREQ = 250.0
    win_len = int(2.0 * SFREQ) # 500
    hop_len = int(1.0 * SFREQ) # 250
    
    results = []
    processed = 0
    
    for idx, row in df.iterrows():
        sub_id = row['participants_ID']
        usi = row['USI']
        sev = 'Healthy' if row['indication'] == 'HEALTHY' else ('Severe MDD' if usi >= 0.4 else 'Moderate MDD')
        
        search = DIR_TDBRAIN / sub_id / "ses-1" / "eeg"
        if not search.exists(): continue
        vhdrs = list(search.glob("*task-restEC*.vhdr"))
        if not vhdrs: continue
        
        try:
            raw = mne.io.read_raw_brainvision(vhdrs[0], preload=True, verbose=False)
            raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)
            
            pA = [c for c in raw.ch_names if c.upper() in ['F3', 'F4']]
            pB = [c for c in raw.ch_names if c.upper() in ['T7', 'T8']]
            pC = [c for c in raw.ch_names if c.upper() in ['FZ', 'CZ']]
            pD = [c for c in raw.ch_names if c.upper() in ['FP1', 'FP2', 'F7', 'F8']]
            
            if not pA or not pB or not pC or not pD: continue
            
            # Universal extraction and baseline prep
            sA = np.mean(raw.get_data(picks=pA), axis=0)
            sB = np.mean(raw.get_data(picks=pB), axis=0)
            sC = np.mean(raw.get_data(picks=pC), axis=0)
            sD_raw = np.mean(raw.get_data(picks=pD), axis=0)
            
            b, a = signal.butter(4, [30.0/(raw.info['sfreq']/2), 45.0/(raw.info['sfreq']/2)], btype='bandpass')
            sD = signal.filtfilt(b, a, sD_raw)
            
            # Crop length just in case it's huge, keep it reasonable (like max 60 seconds)
            max_p = int(60 * raw.info['sfreq'])
            if len(sA) > max_p:
                sA, sB, sC, sD = sA[:max_p], sB[:max_p], sC[:max_p], sD[:max_p]
            
            # We must resample arrays to 250Hz locally
            if raw.info['sfreq'] != 250.0:
                sA = signal.resample(sA, int(len(sA) * 250 / raw.info['sfreq']))
                sB = signal.resample(sB, int(len(sB) * 250 / raw.info['sfreq']))
                sC = signal.resample(sC, int(len(sC) * 250 / raw.info['sfreq']))
                sD = signal.resample(sD, int(len(sD) * 250 / raw.info['sfreq']))
            
            # Sliding extraction
            raw_omegas = []
            N_tot = len(sA)
            
            for start_i in range(0, N_tot - win_len, hop_len):
                wA, wB = sA[start_i:start_i+win_len], sB[start_i:start_i+win_len]
                wC, wD = sC[start_i:start_i+win_len], sD[start_i:start_i+win_len]
                
                for w in [wA, wB, wC, wD]:
                    w[:] = signal.detrend(w)
                    w[:] = (w - np.mean(w)) / (np.std(w) + 1e-9)
                    
                val = compute_o_info(wA, wB, wC, wD, k=3)
                raw_omegas.append(val)
                
            if len(raw_omegas) < 10:
                continue
                
            raw_signs = np.sign(raw_omegas)
            
            # The Median Filter smoothing (Majority Voting) on exactly 5 length
            filtered_signs = signal.medfilt(raw_signs, kernel_size=5)
            
            mean_dt, p_sr, p_rs, lab = extract_kinetics(filtered_signs, slide_s=1.0)
            
            results.append({
                'Subject': sub_id, 'Severity': sev, 'USI': usi,
                'Mean_Dwell_R': mean_dt,
                'P_S_to_R': p_sr,
                'P_R_to_S': p_rs,
                'Lability_per_m': lab
            })
            
            processed += 1
            if processed % 5 == 0:
                print(f"Metastability Engine Processed: {processed}")
                
        except Exception as e:
            continue
            
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[!] Execution failed on TDBRAIN Metastability.")
        return
        
    df_res.to_csv(DIR_ASSETS / "Metastability_Metrics.csv", index=False)
    
    # -----------------------------
    # ML & STATS VALIDATION
    # -----------------------------
    df_bin = df_res[df_res['Severity'].isin(['Healthy', 'Severe MDD'])]
    
    if len(df_bin) > 0:
        X = df_bin[['Mean_Dwell_R', 'P_S_to_R', 'P_R_to_S', 'Lability_per_m']]
        y = df_bin['Severity'].map({'Healthy': 0, 'Severe MDD': 1})
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        acc = cross_val_score(rf, X, y, cv=cv, scoring='accuracy').mean()
        print(f"\n[+] Meta-Kinetics CV Accuracy (HC vs Severe MDD): {acc * 100:.2f}%")
        
    # Correlation Check
    df_reg = df_res.dropna(subset=['USI', 'Mean_Dwell_R'])
    X_ols = sm.add_constant(df_reg['USI'])
    model = sm.OLS(df_reg['Mean_Dwell_R'], X_ols).fit()
    print(f"[+] Dwell_Time vs USI -> P-Value: {model.pvalues.get('USI', 1.0):.4f}")
    
    # -----------------------------
    # PLOTS
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Dwell Time
    sns.kdeplot(data=df_res, x='Mean_Dwell_R', hue='Severity', fill=True, 
                palette={'Healthy': 'darkgreen', 'Moderate MDD': 'orange', 'Severe MDD': 'darkred'}, 
                ax=axes[0], alpha=0.5)
    axes[0].set_title('Dwell Time Distribution (State R: Redundancy)')
    axes[0].set_xlabel('Mean Seconds Trapped in Synergy-Failure')
    
    # Plot 2: State Transitions P
    sns.scatterplot(data=df_res, x='P_S_to_R', y='P_R_to_S', hue='Severity', style='Severity',
                    palette={'Healthy': 'darkgreen', 'Moderate MDD': 'orange', 'Severe MDD': 'darkred'},
                    ax=axes[1], s=100, alpha=0.8)
    axes[1].set_title('Asymmetric State Transitions\nFalling In vs Waking Up')
    axes[1].set_xlabel('P(Synergy -> Redundancy)')
    axes[1].set_ylabel('P(Redundancy -> Synergy)')
    
    plt.tight_layout()
    plt.savefig(DIR_ASSETS / "Metastability_Dynamics.png", dpi=300)
    plt.close()
    print("[+] Assets Generated.")

if __name__ == "__main__":
    run_metastability_tdbrain()
