import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.optimize import curve_fit

from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------
# 1. CORE MATH: CSD & NON-LINEAR MODELING
# ----------------------------------------------------

def compute_csd(series, window_fraction=0.2):
    """
    Computes rolling Variance and AR(1) over a detrended time series of trials.
    Returns the mean/max of variance and AR(1), and the Kendall tau for trends.
    """
    if len(series) < 10:
        return None
        
    # Detrend the series to satisfy stationarity assumptions before CSD
    detrended = signal.detrend(series, type='linear')
    
    # Calculate window size
    w_size = max(10, int(len(series) * window_fraction))
    
    # Rolling calculations
    variances = []
    ar1s = []
    
    for i in range(len(detrended) - w_size + 1):
        window = detrended[i:i+w_size]
        variances.append(np.var(window))
        # AR(1) calculation using Pearson correlation with lag-1
        if np.std(window[:-1]) == 0 or np.std(window[1:]) == 0:
            ar1s.append(0)
        else:
            r, _ = stats.pearsonr(window[:-1], window[1:])
            ar1s.append(r)
            
    # Trend calculations using Kendall's Tau
    time_points = np.arange(len(variances))
    tau_var, p_var = stats.kendalltau(time_points, variances)
    tau_ar1, p_ar1 = stats.kendalltau(time_points, ar1s)
    
    # Peaks
    max_var = np.max(variances)
    max_ar1 = np.max(ar1s)
    idx_max_ar1 = np.argmax(ar1s)
    
    return {
        'detrended': detrended,
        'rolling_var': np.array(variances),
        'rolling_ar1': np.array(ar1s),
        'mean_var': np.mean(variances),
        'mean_ar1': np.mean(ar1s),
        'max_var': max_var,
        'max_ar1': max_ar1,
        'idx_max_ar1': idx_max_ar1,
        'tau_var': tau_var,
        'tau_ar1': tau_ar1
    }

def fit_sigmoid(x, y):
    """
    Fits y = L / (1 + exp(-k*(x - x0))) with L=max(y) or 1.
    Calculates AIC and returns parameters. 
    """
    def sigmoid(x, k, x0):
        # We assume responses scale 0 to 1 (Accuracy)
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))
        
    def linear(x, a, b):
        return a + b * x
        
    # Standardize X for optimization stability
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-9)
    
    # Fit Sigmoid
    try:
        popt_sig, _ = curve_fit(sigmoid, x_norm, y, p0=[1.0, 0.0], maxfev=10000)
        y_pred_sig = sigmoid(x_norm, *popt_sig)
        mse_sig = np.mean((y - y_pred_sig)**2)
        k_est = len(popt_sig)
        aic_sig = len(y) * np.log(mse_sig) + 2 * k_est
    except:
        popt_sig = [np.nan, np.nan]
        aic_sig = np.inf
        y_pred_sig = np.zeros_like(y)
        
    # Fit Linear
    try:
        popt_lin, _ = curve_fit(linear, x_norm, y, p0=[np.mean(y), 0.0])
        y_pred_lin = linear(x_norm, *popt_lin)
        mse_lin = np.mean((y - y_pred_lin)**2)
        k_lin = len(popt_lin)
        aic_lin = len(y) * np.log(mse_lin) + 2 * k_lin
    except:
        popt_lin = [np.nan, np.nan]
        aic_lin = np.inf
        y_pred_lin = np.zeros_like(y)
        
    return {
        'x_norm': x_norm,
        'aic_sig': aic_sig,
        'aic_lin': aic_lin,
        'k_sig': popt_sig[0],
        'x0_sig': popt_sig[1],
        'y_pred_sig': y_pred_sig,
        'y_pred_lin': y_pred_lin
    }

# ----------------------------------------------------
# 2. DATA LOADING & EXTRACTION (NO DDS)
# ----------------------------------------------------

def extract_theta_trial_series(csv_path, roi='vmPFC', time_window=(250, 400)):
    """
    Reads trial_roi_timeseries.csv.
    Extracts the physiological power (variance) in the 250-400ms window as 
    a proxy for Frontomedial Theta / ERP amplitude for each trial.
    """
    print(f"Loading timeseries: {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    if 'roi' in df.columns:
        df = df[df['roi'] == roi]
        
    # Filter time window for component of interest
    df = df[(df['time_ms'] >= time_window[0]) & (df['time_ms'] <= time_window[1])]
    
    # Calculate signal variance (power) within the window for each trial
    # Using variance of the time-series window as a proxy for raw band power
    trials_power = df.groupby(['subject', 'cond', 'trial'])['value'].var().reset_index()
    trials_power.rename(columns={'value': 'theta_power'}, inplace=True)
    
    # Sort chronologically by trial index
    trials_power.sort_values(by=['subject', 'cond', 'trial'], inplace=True)
    return trials_power

def load_cavanagh_data(root_path):
    root = Path(root_path)
    ts_file = root / "DDS-ds005456/derivatives/trial_roi_timeseries/trial_roi_timeseries.csv"
    clin_file = root / "DDS-ds005456/derivatives/cavanagh_clinical.csv"
    
    if not ts_file.exists():
        print("Cavanagh TS not found!")
        return None, None
        
    # Extract
    tp = extract_theta_trial_series(ts_file, roi='vmPFC', time_window=(200, 400))
    # Keep high-conflict loss trials (the "punishment" state)
    tp = tp[tp['cond'] == 'loss']
    
    clin = pd.read_csv(clin_file)
    clin.rename(columns={'Subject': 'subject'}, inplace=True)
    
    return tp, clin

def load_modma_data(root_path):
    root = Path(root_path)
    ts_file = root / "DDS-MODMA/derivatives/trial_roi_timeseries/trial_roi_timeseries.csv"
    clin_file = root / "EEG_128channels_ERP_lanzhou_2015/subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx"
    
    if not ts_file.exists():
        print("MODMA TS not found!")
        return None, None
        
    tp = extract_theta_trial_series(ts_file, roi='cACC', time_window=(200, 400))
    tp = tp[tp['cond'] == 'happy'] # Using available task if loss isn't mapped
    
    clin = pd.read_excel(clin_file)
    clin['subject'] = clin['subject id'].astype(str).str.zfill(8)
    # Binary group for boxplots
    clin['Group'] = clin['PHQ-9'].apply(lambda x: "Severe" if x >= 15 else "Healthy" if x <= 5 else "Mild")
    
    return tp, clin

# ----------------------------------------------------
# 3. STATISTICAL ANALYSIS & PLOTTING
# ----------------------------------------------------

def run_dataset_csd(tp, clin, clinical_col, group_col, dataset_name, out_dir):
    print(f"\n=========================================")
    print(f" CSD ANALYSIS: {dataset_name}")
    print(f"=========================================")
    
    # 1. Run CSD per subject
    csd_results = []
    
    subjects = tp['subject'].unique()
    
    # Save an example subject for Figure 1
    figure1_subjects = []
    
    for subj in subjects:
        series = tp[tp['subject'] == subj]['theta_power'].values
        res = compute_csd(series, window_fraction=0.20)
        
        if res is not None:
            # Check if this subject is in clinical
            subj_clin = clin[clin['subject'] == subj]
            if not subj_clin.empty:
                score = subj_clin.iloc[0][clinical_col]
                group = subj_clin.iloc[0][group_col]
                
                csd_results.append({
                    'subject': subj,
                    'group': group,
                    'clinical_score': score,
                    'mean_var': res['mean_var'],
                    'max_var': res['max_var'],
                    'mean_ar1': res['mean_ar1'],
                    'max_ar1': res['max_ar1'],
                    'tau_var': res['tau_var'],
                    'tau_ar1': res['tau_ar1']
                })
                
                # Snag 1 Healthy and 1 MDD for plotting
                if len(figure1_subjects) < 2:
                    if len(figure1_subjects) == 0 and group == "Healthy":
                        figure1_subjects.append((subj, group, res))
                    elif len(figure1_subjects) == 1 and group != "Healthy" and figure1_subjects[0][1] != group:
                        figure1_subjects.append((subj, group, res))
                        
    df_csd = pd.DataFrame(csd_results)
    
    # 2. Group Statistics & Figure 2
    print("\n--- Group Comparisons (T-Tests) ---")
    healthy = df_csd[df_csd['group'] == 'Healthy']
    severe = df_csd[df_csd['group'] != 'Healthy']
    
    for metric in ['mean_var', 'mean_ar1', 'max_ar1', 'tau_ar1']:
        t, p = stats.ttest_ind(healthy[metric].dropna(), severe[metric].dropna(), equal_var=False)
        d = (healthy[metric].mean() - severe[metric].mean()) / np.sqrt((healthy[metric].std()**2 + severe[metric].std()**2)/2)
        print(f"{metric}: T = {t:.3f}, p = {p:.3f}, Cohen's d = {d:.3f}")
        
    # Figure 2: Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df_csd, x='group', y='mean_var', ax=axes[0], palette='pastel')
    sns.swarmplot(data=df_csd, x='group', y='mean_var', ax=axes[0], color='.25')
    axes[0].set_title(f"{dataset_name}: Mean Variance (Rolling)")
    
    sns.boxplot(data=df_csd, x='group', y='mean_ar1', ax=axes[1], palette='pastel')
    sns.swarmplot(data=df_csd, x='group', y='mean_ar1', ax=axes[1], color='.25')
    axes[1].set_title(f"{dataset_name}: Mean AR(1)")
    
    plt.tight_layout()
    plt.savefig(out_dir / f"Fig2_GroupComparisons_{dataset_name}.png", dpi=300)
    plt.close()
    
    # Figure 1: Individual Examples
    if len(figure1_subjects) == 2:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        for col_idx, (subj, group, res) in enumerate(figure1_subjects):
            axes[0, col_idx].plot(res['detrended'], color='black')
            axes[0, col_idx].set_title(f"{group} Individual: {subj}\nDetrended Theta Sequence")
            
            axes[1, col_idx].plot(res['rolling_var'], color='red')
            axes[1, col_idx].set_title(f"Rolling Variance (Tau={res['tau_var']:.2f})")
            
            axes[2, col_idx].plot(res['rolling_ar1'], color='blue')
            axes[2, col_idx].axvline(res['idx_max_ar1'], color='black', linestyle='--', alpha=0.5, label='Max AR1')
            axes[2, col_idx].set_title(f"Rolling AR(1) Lag-1 (Tau={res['tau_ar1']:.2f})")
            axes[2, col_idx].legend()
            
        plt.tight_layout()
        plt.savefig(out_dir / f"Fig1_CSD_Individuals_{dataset_name}.png", dpi=300)
        plt.close()
        
    return df_csd

def threshold_analysis(df_csd, dataset_name, clinical_col, out_dir):
    print(f"\n--- Non-Linear Threshold Analysis & Bifurcation Check ---")
    
    # We test if the severity/accuracy score drops geometrically according to AR(1) or Theta Variance
    x = df_csd['mean_ar1'].values
    y = df_csd['clinical_score'].values
    
    # Normalize Y to 0-1 for Sigmoid fit comparison
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-9)
    # Invert so 1 is healthy (loss of function) if clinical is higher
    # Actually, let's keep it as is, sigmoid can have negative k
    
    fit_res = fit_sigmoid(x, y_norm)
    
    print(f"Sigmoid AIC: {fit_res['aic_sig']:.2f}")
    print(f"Linear AIC : {fit_res['aic_lin']:.2f}")
    
    if fit_res['aic_sig'] < fit_res['aic_lin'] - 2:
        print("[*] RESULTS: The Non-Linear Sigmoid model is a superior fit. A phase transition/threshold exists.")
    else:
        print("[-] RESULTS: The Linear model is preferred. No distinct critical boundary (topological phase transition) confirmed.")
        
    # Figure 3: Scatter
    plt.figure(figsize=(9, 6))
    
    # Plot real data
    plt.scatter(fit_res['x_norm'], y_norm, c='gray', alpha=0.7, label='Patients')
    
    # Sort for smooth lines
    sort_idx = np.argsort(fit_res['x_norm'])
    plt.plot(fit_res['x_norm'][sort_idx], fit_res['y_pred_sig'][sort_idx], color='red', lw=2, label=f'Sigmoid (AIC={fit_res["aic_sig"]:.1f})')
    plt.plot(fit_res['x_norm'][sort_idx], fit_res['y_pred_lin'][sort_idx], color='blue', lw=2, linestyle='--', label=f'Linear (AIC={fit_res["aic_lin"]:.1f})')
    
    plt.title(f"Fig 3: Phase Transition Threshold Check ({dataset_name})\nPredicting Clinical Severity via Autocorrelation (AR1)")
    plt.xlabel("Standardized Mean AR(1) [CSD Indicator]")
    plt.ylabel(f"Normalized Severity ({clinical_col})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"Fig3_Threshold_Analysis_{dataset_name}.png", dpi=300)
    plt.close()

def run_pipeline():
    p_cavanagh = "/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds005356"
    p_modma = "/media/neuraldyn/PortableSSD/DEPRESSION/MODMA"
    
    out_dir = Path("/media/neuraldyn/PortableSSD/DEPRESSION/UNIVERSAL_BIFURCATION_LAB/csd_results")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. CAVANAGH
    tp_c, clin_c = load_cavanagh_data(p_cavanagh)
    if tp_c is not None and not tp_c.empty:
        df_csd_c = run_dataset_csd(tp_c, clin_c, clinical_col='SHAPS', group_col='Group_SHAPS', dataset_name="CAVANAGH_MEG", out_dir=out_dir)
        threshold_analysis(df_csd_c, "CAVANAGH_MEG", 'SHAPS', out_dir)
        
    # 2. MODMA
    tp_m, clin_m = load_modma_data(p_modma)
    if tp_m is not None and not tp_m.empty:
        df_csd_m = run_dataset_csd(tp_m, clin_m, clinical_col='PHQ-9', group_col='Group', dataset_name="MODMA_EEG", out_dir=out_dir)
        threshold_analysis(df_csd_m, "MODMA_EEG", 'PHQ-9', out_dir)
        
    print("\n[OK] CSD Forensics and Non-Linear Threshold modeling completed. Assets saved.")

if __name__ == "__main__":
    run_pipeline()
