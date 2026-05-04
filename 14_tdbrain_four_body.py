import mne
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

DIR_ASSETS = Path("06_manuscript_assets")
DIR_TDBRAIN = Path("/media/neuraldyn/PortableSSD/TDBRAIN_derivatives")

# --- CORE MATH: Kozachenko-Leonenko Continuous Entropy & KSG ---
def ksg_entropy(X, k=4):
    if X.ndim == 1: X = X[:, None]
    N, d = X.shape
    rng = np.random.default_rng(seed=42)
    X = X + 1e-10 * rng.standard_normal(X.shape)
    
    tree = cKDTree(X)
    distances, _ = tree.query(X, k=k+1, p=np.inf)
    eps = np.maximum(distances[:, k], 1e-15)
    
    ans = digamma(N) - digamma(k) + np.log(2**d) + (d / N) * np.sum(np.log(eps))
    return ans / np.log(2)

def ksg_cmi(X, Y, Z, k=4):
    N = len(X)
    if X.ndim == 1: X = X[:, None]
    if Y.ndim == 1: Y = Y[:, None]
    if Z.ndim == 1: Z = Z[:, None]

    rng = np.random.default_rng(seed=42)
    X = X + 1e-10 * rng.standard_normal(X.shape)
    Y = Y + 1e-10 * rng.standard_normal(Y.shape)
    Z = Z + 1e-10 * rng.standard_normal(Z.shape)

    XYZ = np.hstack((X, Y, Z))
    tree_xyz = cKDTree(XYZ)
    distances, _ = tree_xyz.query(XYZ, k=k+1, p=np.inf)
    eps = distances[:, k] 
    
    XZ = np.hstack((X, Z))
    YZ = np.hstack((Y, Z))
    
    tree_xz = cKDTree(XZ)
    tree_yz = cKDTree(YZ)
    tree_z = cKDTree(Z)
    
    radius = eps + 1e-12
    n_xz = np.array([len(tree_xz.query_ball_point(XZ[i], radius[i], p=np.inf)) for i in range(N)])
    n_yz = np.array([len(tree_yz.query_ball_point(YZ[i], radius[i], p=np.inf)) for i in range(N)])
    n_z = np.array([len(tree_z.query_ball_point(Z[i], radius[i], p=np.inf)) for i in range(N)])
    
    ans = digamma(k) + np.mean(digamma(n_z)) - np.mean(digamma(n_xz)) - np.mean(digamma(n_yz))
    return max(0, ans / np.log(2))

def compute_o_information(A, B, C, D, k=4):
    H_A, H_B = ksg_entropy(A, k), ksg_entropy(B, k)
    H_C, H_D = ksg_entropy(C, k), ksg_entropy(D, k)
    
    H_BCD = ksg_entropy(np.column_stack((B, C, D)), k=k)
    H_ACD = ksg_entropy(np.column_stack((A, C, D)), k=k)
    H_ABD = ksg_entropy(np.column_stack((A, B, D)), k=k)
    H_ABC = ksg_entropy(np.column_stack((A, B, C)), k=k)
    
    H_All = ksg_entropy(np.column_stack((A, B, C, D)), k=k)
    o_info = 2 * H_All + (H_A + H_B + H_C + H_D) - (H_BCD + H_ACD + H_ABD + H_ABC)
    return o_info

def extract_4body_metrics(raw_A, raw_B, raw_C, raw_D, tau=5, N_samples=10000):
    N_total = len(raw_A)
    best_start = 0
    if N_total > N_samples:
        min_var = np.inf
        stride = 500
        for i in range(0, N_total - N_samples, stride):
            var_comb = np.var(raw_A[i:i+N_samples]) + np.var(raw_B[i:i+N_samples]) + \
                       np.var(raw_C[i:i+N_samples]) + np.var(raw_D[i:i+N_samples])
            if var_comb < min_var:
                min_var = var_comb
                best_start = i
                
    sig_A = raw_A[best_start : best_start + N_samples]
    sig_B = raw_B[best_start : best_start + N_samples]
    sig_C = raw_C[best_start : best_start + N_samples]
    sig_D = raw_D[best_start : best_start + N_samples]
    
    sigs = [sig_A, sig_B, sig_C, sig_D]
    sigs_z = []
    for s in sigs:
        s = signal.detrend(s)
        s = (s - np.mean(s)) / (np.std(s) + 1e-9)
        sigs_z.append(s)
        
    sig_A, sig_B, sig_C, sig_D = sigs_z
    
    past_A, past_B, past_C, past_D = sig_A[:-tau], sig_B[:-tau], sig_C[:-tau], sig_D[:-tau]
    fut_A, fut_B, fut_C, fut_D = sig_A[tau:], sig_B[tau:], sig_C[tau:], sig_D[tau:]
    
    te_b_a = ksg_cmi(past_B, fut_A, past_A)
    te_d_c = ksg_cmi(past_D, fut_C, past_C)
    
    cond_past = np.column_stack((past_A, past_C, past_D))
    cte_4body = ksg_cmi(past_B, fut_A, cond_past)
    
    o_info = compute_o_information(sig_A, sig_B, sig_C, sig_D)
    
    return te_b_a, te_d_c, cte_4body, o_info

def run_tdbrain_four_body():
    print("--- 4-BODY OOD SYSTEM AUDIT (TDBRAIN N=10,000) ---")
    
    # Train primary Random Forest silently via dataset proxy
    try:
        df_train = pd.read_csv("06_manuscript_assets/FourBody_Audit.csv")
    except:
        print("[!] Needs FourBody_Audit.csv")
        return
        
    train_bin = df_train[df_train['Severity'].isin(['Healthy', 'Severe MDD'])]
    X_train = train_bin[['TE_Ins_PFC', 'TE_Amig_Cing', 'cTE_4Body', 'O_Information']]
    y_train = train_bin['Severity'].map({'Healthy': 0, 'Severe MDD': 1})
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # -----------------------------
    # PREPARE TDBRAIN EXTRACTION
    # -----------------------------
    clinical_file = DIR_TDBRAIN / "TDBRAIN_participants_V2.tsv"
    clin_tdbrain = pd.read_csv(clinical_file, sep='\t')
    subjects_to_run = clin_tdbrain[clin_tdbrain['indication'].isin(['HEALTHY', 'MDD'])].copy()
    
    # Use 'BDI_pre'
    subjects_to_run['BDI_pre'] = pd.to_numeric(subjects_to_run['BDI_pre'], errors='coerce')
    hc_bdi_mean = subjects_to_run[subjects_to_run['indication'] == 'HEALTHY']['BDI_pre'].mean()
    if pd.isna(hc_bdi_mean): hc_bdi_mean = 2.0
    
    subjects_to_run['USI'] = (subjects_to_run['BDI_pre'] - hc_bdi_mean) / (63.0 - hc_bdi_mean)
    
    def assign_severity(row):
        if row['indication'] == 'HEALTHY': return 'Healthy'
        if row['indication'] == 'MDD':
            return 'Severe MDD' if row['USI'] >= 0.4 else 'Moderate MDD'
        return None
        
    subjects_to_run['Severity'] = subjects_to_run.apply(assign_severity, axis=1)
    
    # Needs to match `participants_ID` precisely
    subjects_to_run = subjects_to_run.dropna(subset=['participants_ID']).drop_duplicates(subset=['participants_ID'])
    
    results = []
    
    print(f"[*] Extracting 4-Body contingency over {len(subjects_to_run)} registered subjects.")

    processed = 0
    for idx, row in subjects_to_run.iterrows():
        sub_id = row['participants_ID']
        sev = row['Severity']
        usi = row['USI']
        
        search_path = DIR_TDBRAIN / sub_id / "ses-1" / "eeg"
        if not search_path.exists(): continue
            
        vhdr_files = list(search_path.glob("*task-restEC*.vhdr"))
        if not vhdr_files: continue
            
        vhdr = vhdr_files[0]
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            # Universal Bandpass 1-45 Hz overall
            raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)
            ch_names = raw.ch_names
            
            # TDBRAIN Explicit Topology Nodes
            proxy_A = [c for c in ch_names if c.upper() in ['F3', 'F4']]
            proxy_B = [c for c in ch_names if c.upper() in ['T7', 'T8']]
            proxy_C = [c for c in ch_names if c.upper() in ['FZ', 'CZ']]
            proxy_D = [c for c in ch_names if c.upper() in ['FP1', 'FP2', 'F7', 'F8']]
            
            if not proxy_A or not proxy_B or not proxy_C or not proxy_D:
                continue
                
            sig_A = np.mean(raw.get_data(picks=proxy_A), axis=0)
            sig_B = np.mean(raw.get_data(picks=proxy_B), axis=0)
            sig_C = np.mean(raw.get_data(picks=proxy_C), axis=0)
            
            # Amygdala Proxy (D) uses Gamma 30-45 Hz Bandpass 
            node_d_raw = np.mean(raw.get_data(picks=proxy_D), axis=0)
            b, a = signal.butter(4, [30.0/(raw.info['sfreq']/2), 45.0/(raw.info['sfreq']/2)], btype='bandpass')
            sig_D = signal.filtfilt(b, a, node_d_raw)
            
            te_ba, te_dc, cte_4b, o_info = extract_4body_metrics(sig_A, sig_B, sig_C, sig_D)
            
            results.append({
                'Subject': sub_id, 'Severity': sev, 'USI': usi,
                'TE_Ins_PFC': te_ba, 'TE_Amig_Cing': te_dc,
                'cTE_4Body': cte_4b, 'O_Information': o_info
            })
            
            processed += 1
            if processed % 10 == 0:
                print(f"KSG Engine Processed: {processed}")
                
        except Exception as e:
            continue
            
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("[!] No results parsed due to topological/KSG mismatch on TDBRAIN.")
        return
        
    df_res.to_csv(DIR_ASSETS / "FourBody_TDBRAIN_Audit.csv", index=False)
    
    # -----------------------------
    # ML PERFORMANCE SCORE
    # -----------------------------
    df_test_bin = df_res[df_res['Severity'].isin(['Healthy', 'Severe MDD'])]
    if len(df_test_bin) < 20:
        print("[!] Not enough binary bounds found to validate RF in TDBRAIN.")
        return
        
    X_test = df_test_bin[['TE_Ins_PFC', 'TE_Amig_Cing', 'cTE_4Body', 'O_Information']]
    y_test = df_test_bin['Severity'].map({'Healthy': 0, 'Severe MDD': 1})
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- OOD GENERALIZATION SUMMARY (TDBRAIN) ---")
    print(f"Accuracy de Clasificación Externa: {acc * 100:.2f}%")
    
    if acc < 0.75:
        print("[CRITICAL] TDBRAIN Classification failed minimal thresholds. OOD broken.")
        return
        
    print("[+] GENERALIZATION SUCCESS! Amygdala High-Frequency proxy confirmed as transversal biological rule.")
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Severe'], yticklabels=['Healthy', 'Severe'])
    plt.ylabel('Ground Truth (TDBRAIN)')
    plt.xlabel('Random Forest 4-Body Predictions')
    plt.title(f'TDBRAIN Transversal 4-Body Confusion Matrix\nAccuracy: {acc*100:.2f}%')
    plt.tight_layout()
    plt.savefig(DIR_ASSETS / "FourBody_TDBRAIN_Confusion.png", dpi=300)
    plt.close()
    
    # MAP_MODERADOS (El Gran Test)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_res, x='TE_Amig_Cing', y='O_Information',
                    hue='Severity', palette={'Healthy': 'darkgreen', 'Moderate MDD': 'orange', 'Severe MDD': 'darkred'},
                    alpha=0.8, s=100, edgecolor='black')
    
    plt.title('Modo de Activación Transversal\n¿Dónde caen los Moderados de TDBRAIN?')
    plt.xlabel('TE Amygdala (Gamma) $\Rightarrow$ Cingulate')
    plt.ylabel('O-Information (System Global Synergy/Redundancy)')
    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(DIR_ASSETS / "FourBody_TDBRAIN_LatentSpace.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_tdbrain_four_body()
