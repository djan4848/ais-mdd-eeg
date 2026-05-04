import mne
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy import signal
import os

warnings.filterwarnings('ignore')

DIR_MEG = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/ds005356")
DIR_ASSETS = Path("06_manuscript_assets")

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

def extract_4body_metrics_meg(raw_A, raw_B, raw_C, raw_D, tau=5, N_samples=10000):
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

def get_meg_proxies(raw):
    # Retrieve explicitly only Neuromag magnetometers
    mag_picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False)
    ch_names = [raw.ch_names[i] for i in mag_picks]
    
    # Use spatial names directly by Neuromag groupings
    # MNE includes selections like "Left-frontal", "Right-frontal"
    lf = mne.read_vectorview_selection('Left-frontal', info=raw.info)
    rf = mne.read_vectorview_selection('Right-frontal', info=raw.info)
    pfc_chs = list(set([ch for ch in lf + rf if ch in ch_names]))
    
    lt = mne.read_vectorview_selection('Left-temporal', info=raw.info)
    rt = mne.read_vectorview_selection('Right-temporal', info=raw.info)
    insula_chs = list(set([ch for ch in lt + rf if ch in ch_names])) # actually lt/rt is insula approx
    insula_chs = list(set([ch for ch in lt + rt if ch in ch_names]))
    
    # Frontal Midline (Cingulate). We'll manually grab sensors containing "011", "012", "013" (usually midline sensors in neuromag mag array)
    cing_chs = [c for c in ch_names if any(x in c for x in ['0111', '0121', '0131', '1411', '1421', '1431', '1441'])]
    
    # Amygdala (Temporal anterior and frontopolar) -> Neuromag ~ 09, 10, 02 (lower frontal/anterior temporal)
    # We will grab all mags from "Right-temporal" and "Left-temporal" but explicitly those near the pole, 
    # e.g., 0241, 0941, 1011, 0911, 1041. 
    amig_chs = [c for c in ch_names if any(x in c for x in ['0241', '0231', '0941', '0911', '1011', '1041', '1311', '1341'])]
    
    return pfc_chs, insula_chs, cing_chs, amig_chs

def run_meg_extraction():
    print("--- INDEPENDENT D2 EXTRACTION: CAVANAGH MEG (N=10k, Gamma-Amig) ---")
    
    excel_file = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/ds005356/Code/MEG MDD IDs and Quex.xlsx")
    if not excel_file.exists():
        print("[!] Needs MEG MDD IDs and Quex.xlsx")
        return
        
    df_labels = pd.read_excel(excel_file, sheet_name="Sheet1", engine="openpyxl")
    df_labels.columns = [str(c).strip() for c in df_labels.columns]
    
    # Need USI
    hc_bdi = df_labels[df_labels['Group'] == 'CTL']['BDI'].mean()
    if pd.isna(hc_bdi): hc_bdi = 2.0
    
    results = []
    meg_files = list(DIR_MEG.rglob("*_meg.fif"))
    # Filter splits
    meg_files = [f for f in meg_files if 'split-02' not in f.name]
    
    processed = 0
    for f in meg_files:
        try:
            sub_id = f.name.split('_')[0]
            try:
                base_val = int(str(sub_id).replace("sub-M87", ""))
                ursi_val = base_val - 100000
            except:
                continue
            
            row = df_labels[df_labels['URSI'] == ursi_val]
            if row.empty: continue
            
            row = row.iloc[0]
            grp = 'Healthy' if str(row['Group']).strip() == 'CTL' else 'MDD'
            
            bdi = float(row['BDI']) if not pd.isna(row['BDI']) else hc_bdi
            usi = (bdi - hc_bdi) / (63.0 - hc_bdi)
            sev = 'Healthy' if grp == 'Healthy' else ('Severe MDD' if usi >= 0.4 else 'Moderate MDD')
            
            raw = mne.io.read_raw_fif(str(f), preload=True, verbose=False)
            
            # Subsample strictly to 10k max limit to avoid memory explosions before we filter
            raw.crop(tmax=30.0) 
            raw.resample(250.0, verbose=False) # standardizing to our training freq
            
            raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)
            
            pA, pB, pC, pD = get_meg_proxies(raw)
            if not pA or not pB or not pC or not pD:
                continue
                
            sig_A = np.mean(raw.get_data(picks=pA), axis=0)
            sig_B = np.mean(raw.get_data(picks=pB), axis=0)
            sig_C = np.mean(raw.get_data(picks=pC), axis=0)
            
            # Gamma just for Amygdala (30-45)
            node_d_raw = np.mean(raw.get_data(picks=pD), axis=0)
            b, a = signal.butter(4, [30.0/(raw.info['sfreq']/2), 45.0/(raw.info['sfreq']/2)], btype='bandpass')
            sig_D = signal.filtfilt(b, a, node_d_raw)
            
            te_ba, te_dc, cte_4b, o_info = extract_4body_metrics_meg(sig_A, sig_B, sig_C, sig_D, N_samples=int(20*250)) # 20*250 = 5000 points because MEG is cropped to 30.0s (7500 pts max)
            
            results.append({
                'Subject': sub_id, 'Severity': sev, 'USI': usi,
                'TE_Ins_PFC': te_ba, 'TE_Amig_Cing': te_dc,
                'cTE_4Body': cte_4b, 'O_Information': o_info
            })
            
            processed += 1
            if processed % 5 == 0:
                print(f"MEG D2 Extracted: {processed}")
                
        except Exception as e:
            continue
            
    df_res = pd.DataFrame(results)
    if df_res.empty: return
    df_res.to_csv(DIR_ASSETS / "FourBody_MEG_Audit.csv", index=False)
    print(f"\n[+] D2 (MEG) Extracted. N = {len(df_res)} subjects.")

if __name__ == "__main__":
    run_meg_extraction()
