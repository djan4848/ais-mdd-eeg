import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne
from mne_icalabel import label_components
from scipy.signal import welch
from specparam import SpectralModel
import warnings
warnings.filterwarnings('ignore')

FS = 256.0
CHANNELS_FRONTAL = ['Fz', 'AFz', 'FCz']
CHANNELS_POSTERIOR = ['Pz', 'POz']

out_dir = Path("/media/neuraldyn/PortableSSD/DEPRESSION/UNIVERSAL_BIFURCATION_LAB/forensic_audit")
out_dir.mkdir(parents=True, exist_ok=True)

path_modma = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/MODMA/DDS-MODMA/derivatives/epochs") 
modma_files = list(path_modma.glob("*epo.fif"))

# Select 3 HC subjects
hc_files = []
for f in modma_files:
    subj_id = f.name.split('-')[0].split('_')[0]
    is_mdd = ('4' in subj_id or '6' in subj_id or '10' in subj_id or 'M' in subj_id)
    if not is_mdd:
        hc_files.append(f)
    if len(hc_files) == 3:
        break

print(f"Selected HC files: {[f.name for f in hc_files]}")

def compute_psd_fooof(signal, fs, freq_range):
    freqs, psd = welch(signal, fs, nperseg=int(2*fs))
    fm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=4, 
                       min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
    fm.fit(freqs, psd, freq_range)
    r2 = fm.get_metrics('gof', 'rsquared')
    return fm, freqs, psd, r2

def process_subject_config(epo_path, t_len, ic_threshold, freq_range):
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    if epochs.info['sfreq'] != FS:
         epochs.resample(FS)
         
    tmin = epochs.times[0]
    if t_len is not None:
        tmax_target = tmin + t_len
        if epochs.times[-1] > tmax_target:
             epochs = epochs.copy().crop(tmin=tmin, tmax=tmax_target)
             
    # Droppear channels
    channels_to_drop = [ch for ch in ['CB1', 'CB2', 'HEOG', 'VEOG', 'M1', 'M2'] if ch in epochs.ch_names]
    if channels_to_drop:
        epochs.drop_channels(channels_to_drop)
        
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage, match_case=False, on_missing='ignore')
    
    ica = mne.preprocessing.ICA(n_components=15, random_state=42, method='fastica', max_iter=200)
    ica.fit(epochs, verbose=False)
    
    brain_ics = list(range(15)) # fallback todo
    if ic_threshold is not None:
        try:
             ic_labels = label_components(epochs, ica, method='iclabel')
             labels = ic_labels['labels']
             probs = ic_labels['y_pred_proba']
             brain_ics = [i for i, (lbl, prb) in enumerate(zip(labels, probs)) if lbl == 'brain' and prb > ic_threshold]
        except Exception as e:
             pass
            
    if len(brain_ics) < 2: 
        # Fallback si el umbral mató todo
        brain_ics = [0, 1] 
        
    mixing_matrix = ica.get_components()
    ch_names = epochs.info['ch_names']
    
    frontal_idx = [ch_names.index(ch) for ch in CHANNELS_FRONTAL if ch in ch_names]
    posterior_idx = [ch_names.index(ch) for ch in CHANNELS_POSTERIOR if ch in ch_names]
    
    best_posterior_ic = brain_ics[0]
    max_p = -1
    for ic in brain_ics:
         power = np.sum(np.abs(mixing_matrix[posterior_idx, ic]))
         if power > max_p:
             max_p = power
             best_posterior_ic = ic
             
    ica_sources = ica.get_sources(epochs).get_data()
    posterior_ts = ica_sources[0, best_posterior_ic, :]
    
    fm, freqs, psd, r2 = compute_psd_fooof(posterior_ts, FS, freq_range)
    return fm, freqs, psd, r2, posterior_ts

# Evaluar
results = []
for idx, f in enumerate(hc_files):
    subj_id = f.name.split('-')[0].split('_')[0]
    print(f"Processing {subj_id}...")
    
    # 1. Pipeline V3 (Actual) -> t_len: 2.0s, ICLabel > 0.70, freq_range [4, 40]
    fm_v3, freqs_v3, psd_v3, r2_v3, ts_v3 = process_subject_config(f, 2.0, 0.70, [4.0, 40.0])
    
    # 2. Pipeline "Poubelle/Archeological" (Aproximación) -> t_len: 5.0 (o None/Full), ICLabel: None (raw ICA components), freq_range [2, 40]
    fm_old, freqs_old, psd_old, r2_old, ts_old = process_subject_config(f, 5.0, None, [2.0, 40.0])
    
    # Sensibilidad specific check:
    # 2a. Solo freq range [2,40] con el resto V3
    _, _, _, r2_freq, _ = process_subject_config(f, 2.0, 0.70, [2.0, 40.0])
    
    # 2b. Solo t_len = 5.0 con el resto V3
    _, _, _, r2_tlen, _ = process_subject_config(f, 5.0, 0.70, [4.0, 40.0])
    
    # 2c. Solo ICLabel = None con el resto V3
    _, _, _, r2_ica, _ = process_subject_config(f, 2.0, None, [4.0, 40.0])
    
    results.append({
        'Subject': subj_id,
        'R2_V3': r2_v3,
        'R2_Old': r2_old,
        'R2_OnlyFreq[2,40]': r2_freq,
        'R2_OnlyTlen=5s': r2_tlen,
        'R2_OnlyIC=None': r2_ica
    })
    
    # Visual Trace
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    fm_old.plot(plot_peaks='shade', add_legend=True, ax=plt.gca())
    plt.title(f"Poubelle/Old Version (R2={r2_old:.2f})")
    
    plt.subplot(1, 2, 2)
    fm_v3.plot(plot_peaks='shade', add_legend=True, ax=plt.gca())
    plt.title(f"V3/Zero-Absolute Version (R2={r2_v3:.2f})")
    
    plt.tight_layout()
    plt.savefig(out_dir / f"trace_cmp_{subj_id}.png")
    plt.close()

df = pd.DataFrame(results)
print("\n--- RESULTS OF FORENSIC SWEEP ---")
print(df)
df.to_csv(out_dir / "audit_parameters.csv", index=False)
