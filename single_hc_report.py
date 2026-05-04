import numpy as np
import scipy.stats as stats
import mne
from mne_icalabel import label_components
from scipy.signal import welch
from scipy.signal.windows import hann
from specparam import SpectralModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

FS = 256.0
T_LENGTH = 2.0
FREQ_RANGE = [2.0, 40.0]

def plot_single_subject(f_path):
    epochs = mne.read_epochs(f_path, preload=True, verbose=False)
    
    # Check if there is an aggressive high-pass filter applied already
    print(f"Epochs Info: {epochs.info['highpass']} Hz HPF, {epochs.info['lowpass']} Hz LPF")
    
    if epochs.info['sfreq'] != FS: epochs.resample(FS)
    
    tmin = epochs.times[0]
    tmax_target = tmin + T_LENGTH
    if epochs.times[-1] > tmax_target:
         epochs = epochs.copy().crop(tmin=tmin, tmax=tmax_target)
         
    channels_to_drop = [ch for ch in ['CB1', 'CB2', 'HEOG', 'VEOG', 'M1', 'M2'] if ch in epochs.ch_names]
    if channels_to_drop: epochs.drop_channels(channels_to_drop)
        
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage, match_case=False, on_missing='ignore')
    
    ica = mne.preprocessing.ICA(n_components=15, random_state=42, method='fastica', max_iter=200)
    ica.fit(epochs, verbose=False)
    
    try:
         ic_labels = label_components(epochs, ica, method='iclabel')
         labels = ic_labels['labels']
         probs = ic_labels['y_pred_proba']
         brain_ics = [i for i, (lbl, prb) in enumerate(zip(labels, probs)) if lbl == 'brain' and prb > 0.70]
    except Exception as e:
         brain_ics = list(range(15))
            
    if len(brain_ics) < 1: brain_ics = [0, 1]
    
    # We just want a posterior component to see Alpha
    mixing_matrix = ica.get_components()
    ch_names = epochs.info['ch_names']
    posterior_idx = [ch_names.index(ch) for ch in ['Pz', 'POz'] if ch in ch_names]
    
    max_p = -1; best_posterior_ic = brain_ics[0]
    for ic in brain_ics:
         power = np.sum(np.abs(mixing_matrix[posterior_idx, ic]))
         if power > max_p:
             max_p = power; best_posterior_ic = ic
             
    ica_sources = ica.get_sources(epochs).get_data()
    posterior_epochs = ica_sources[:, best_posterior_ic, :]
    
    # Method 1: Welch average over epochs
    nperseg = int(2*FS) if posterior_epochs.shape[-1] >= int(2*FS) else posterior_epochs.shape[-1]
    freqs_avg, psds = welch(posterior_epochs, FS, nperseg=nperseg, axis=-1)
    avg_psd = np.mean(psds, axis=0)
    
    # Method 2: Hanning window + Flatten (User's suggested method to verify jump removal)
    n_epochs, n_times = posterior_epochs.shape
    window = hann(n_times)
    tapered_epochs = np.zeros_like(posterior_epochs)
    for i in range(n_epochs):
        epoch_mean = np.mean(posterior_epochs[i, :])
        tapered_epochs[i, :] = (posterior_epochs[i, :] - epoch_mean) * window
        
    flat_ts = tapered_epochs.flatten()
    freqs_flat, psd_flat = welch(flat_ts, FS, nperseg=int(2*FS))
    
    # FOOOF Fit on Average (Method 1)
    fm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=4, 
                       min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
    fm.fit(freqs_avg, avg_psd, FREQ_RANGE)
    r2_avg = fm.get_metrics('gof', 'rsquared')
    
    # Ploteo
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.loglog(freqs_avg, avg_psd, label='Averaged Epoch PSD')
    plt.loglog(freqs_flat, psd_flat, label='Flattened Tapered PSD', alpha=0.7)
    plt.axvline(10, color='r', linestyle='--', alpha=0.5, label='Alpha Peak (10Hz)')
    plt.xlim(2, 40)
    plt.title("PSD Comparison: Ave vs Flat")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    fm.plot(plot_peaks='shade', add_legend=True, ax=plt.gca())
    plt.title(f"FOOOF Fit (Avg PSD) R2={r2_avg:.4f}")
    
    plt.tight_layout()
    plt.savefig('hc_alpha_validation.png')
    
    print(f"R2 using Average Epochs: {r2_avg:.4f}")

path_modma = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/MODMA/DDS-MODMA/derivatives/epochs") 
hc_files = [f for f in path_modma.glob("*epo.fif") if not ('4' in f.name.split('-')[0].split('_')[0] or '6' in f.name.split('-')[0].split('_')[0] or '10' in f.name.split('-')[0].split('_')[0] or 'M' in f.name.split('-')[0].split('_')[0])]

plot_single_subject(hc_files[0])
