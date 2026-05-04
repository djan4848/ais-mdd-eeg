import numpy as np
import pandas as pd
import scipy.stats as stats
import mne
from mne_icalabel import label_components
from scipy.signal import welch
from specparam import SpectralModel
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

FS = 256.0
T_LENGTH = 2.0
FREQ_RANGE = [2.0, 40.0]
CH_FRONTAL = ['Fz', 'AFz', 'FCz', 'E11', 'E12', 'E5']
CH_POSTERIOR = ['Pz', 'POz', 'E62', 'E72']

def compute_spectrals(epo_path):
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    if epochs.info['sfreq'] != FS: epochs.resample(FS)
    
    tmin = epochs.times[0]
    tmax_target = tmin + T_LENGTH
    if epochs.times[-1] > tmax_target:
        epochs = epochs.copy().crop(tmin=tmin, tmax=tmax_target)
        
    channels_to_drop = [ch for ch in ['CB1', 'CB2', 'HEOG', 'VEOG', 'M1', 'M2'] if ch in epochs.ch_names]
    if channels_to_drop: epochs.drop_channels(channels_to_drop)
    
    if 'E1' in epochs.ch_names:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    else:
        montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage, match_case=False, on_missing='ignore')
    
    ica = mne.preprocessing.ICA(n_components=15, random_state=42, method='fastica', max_iter=200)
    ica.fit(epochs, verbose=False)
    
    try:
        ic_labels = label_components(epochs, ica, method='iclabel')
        brain_ics = [i for i, (lbl, prb) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])) if lbl == 'brain' and prb > 0.70]
    except:
        return None, None
        
    if len(brain_ics) < 1: brain_ics = [0, 1]
    
    mixing_matrix = ica.get_components()
    ch_names = epochs.info['ch_names']
    post_idx = [ch_names.index(ch) for ch in CH_POSTERIOR if ch in ch_names]
    
    if not post_idx: return None, None
    best_p = max(brain_ics, key=lambda ic: np.sum(np.abs(mixing_matrix[post_idx, ic])))
    
    ica_s = ica.get_sources(epochs).get_data()
    post_epochs = ica_s[:, best_p, :]
    
    nperseg = int(2*FS) if post_epochs.shape[-1] >= int(2*FS) else post_epochs.shape[-1]
    freqs, psds = welch(post_epochs, FS, nperseg=nperseg, axis=-1)
    avg_psd = np.mean(psds, axis=0)
    
    fm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=4, 
                       min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
    fm.fit(freqs, avg_psd, FREQ_RANGE)
    r2 = fm.get_metrics('gof', 'rsquared')
    chi = fm.get_params('aperiodic', 'exponent')
    
    return r2, chi

def sigmoid(x, L, x0, k, b):
    return b + L / (1.0 + np.exp(-k * (x - x0)))

def generate_assets():
    print("--- GENERATING FINAL MANUSCRIPT ASSETS ---")
    df_bio = pd.read_csv("final_mdd_biomarker_matrix.csv")
    
    path_modma = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/MODMA/DDS-MODMA/derivatives/epochs")
    all_files = list(path_modma.glob("*epo.fif"))
    
    results = []
    
    for idx, row in df_bio.iterrows():
        subj_str = str(row['subject']).zfill(8)
        # Find matching fif
        matching_file = next((f for f in all_files if subj_str in f.name), None)
        
        if matching_file is None:
            continue
            
        r2, chi = compute_spectrals(matching_file)
        if r2 is not None:
            results.append({
                'subject': row['subject'],
                'PHQ-9': row['PHQ-9'],
                'Synergy': row['synergy'],
                'Type': row['type'],
                'R2': r2,
                'Chi': chi
            })
            print(f"Processed {subj_str}: R2={r2:.3f}, Chi={chi:.3f}, PHQ-9={row['PHQ-9']}")
            
    df_res = pd.DataFrame(results)
    if len(df_res) == 0:
        print("No valid data exported.")
        return
        
    out_dir = Path("06_manuscript_assets")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # 1. Phase Transition Plot (PHQ-9 vs R2)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_res, x='PHQ-9', y='R2', hue='Type', palette=['blue', 'red'], s=100, alpha=0.8)
    
    # Sigmoid fit attempt
    try:
        xdata = df_res['PHQ-9'].values
        ydata = df_res['R2'].values
        # Initial guess
        p0 = [max(ydata)-min(ydata), np.median(xdata), -0.5, min(ydata)]
        popt, _ = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox', maxfev=10000)
        x_fit = np.linspace(min(xdata), max(xdata), 100)
        y_fit = sigmoid(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'k--', label='Logistic Phase Transition Fit', alpha=0.7)
    except Exception as e:
        print("Sigmoid fit failed:", e)
        sns.regplot(data=df_res, x='PHQ-9', y='R2', scatter=False, color='k', label='Linear Trend')
        
    plt.axhline(0.90, color='gray', linestyle=':', label='HC Base Threshold')
    plt.title("Aperiodic Breakdown as a Phase Transition in MDD")
    plt.ylabel("$R^2$ (SpecParam Goodness of Fit)")
    plt.xlabel("Depression Severity (PHQ-9)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "Figure_Phase_Transition.png", dpi=300)
    plt.close()
    
    # 2. Synergy-Noise Coupling (Chi vs Synergy)
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df_res, x='Chi', y='Synergy', scatter_kws={'alpha':0.8, 's':100}, color='purple')
    rho, p_val = stats.spearmanr(df_res['Chi'], df_res['Synergy'])
    plt.title(f"Information Synergy vs Aperiodic Noise Coupling\nSpearman Rho = {rho:.3f} (p = {p_val:.4f})")
    plt.ylabel("PID Synergy (Bits)")
    plt.xlabel("Aperiodic Exponent $\chi$ (1/f drop)")
    plt.tight_layout()
    plt.savefig(out_dir / "Figure_Synergy_Noise_Coupling.png", dpi=300)
    plt.close()

    # 3. Paper Assets Table
    def categorize_severity(phq):
        if phq <= 4: return "Healthy Control"
        elif phq <= 14: return "Moderate MDD"
        else: return "Severe MDD"
        
    df_res['Group'] = df_res['PHQ-9'].apply(categorize_severity)
    summary = df_res.groupby('Group')[['Chi', 'R2', 'Synergy']].agg(['mean', 'std'])
    
    # Format table exactly for markdown
    md_table = "### Table 1: Neurodynamic & Information Biomarkers by Severity Group\n\n"
    md_table += "| Clinical Group | Aperiodic Exponent ($\chi$) | SpecParam Fit ($R^2$) | Information Synergy (PID) |\n"
    md_table += "| --- | --- | --- | --- |\n"
    
    for group in ["Healthy Control", "Moderate MDD", "Severe MDD"]:
        if group in summary.index:
            chi_m, chi_s = summary.loc[group, 'Chi']
            r2_m, r2_s = summary.loc[group, 'R2']
            syn_m, syn_s = summary.loc[group, 'Synergy']
            md_table += f"| {group} | {chi_m:.3f} ± {chi_s:.3f} | {r2_m:.3f} ± {r2_s:.3f} | {syn_m:.3f} ± {syn_s:.3f} |\n"

    with open(out_dir / "Table_1_Biomarkers.md", "w") as f:
        f.write(md_table)
        
    print("\n[+] SUCCESS! Phase plot, Correlation plot, and Final Table generated in 06_manuscript_assets/")

if __name__ == "__main__":
    generate_assets()
