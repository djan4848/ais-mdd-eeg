import numpy as np
import scipy.stats as stats
import mne
from mne_icalabel import label_components
from scipy.signal import welch
from scipy.signal.windows import tukey
from specparam import SpectralModel
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

FS = 256.0
T_LENGTH = 2.0
FREQ_RANGE = [2.0, 40.0]
CHANNELS_FRONTAL = ['Fz', 'AFz', 'FCz']
CHANNELS_POSTERIOR = ['Pz', 'POz']

def compute_psd_fooof(signal, fs):
    freqs, psd = welch(signal, fs, nperseg=int(2*fs))
    fm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=4, 
                       min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
    fm.fit(freqs, psd, FREQ_RANGE)
    return fm.get_metrics('gof', 'rsquared')

def process_subject_preflight(epo_path):
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
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
            
    if len(brain_ics) < 2: brain_ics = [0, 1]
        
    mixing_matrix = ica.get_components()
    ch_names = epochs.info['ch_names']
    
    frontal_idx = [ch_names.index(ch) for ch in CHANNELS_FRONTAL if ch in ch_names]
    posterior_idx = [ch_names.index(ch) for ch in CHANNELS_POSTERIOR if ch in ch_names]
    
    max_p = -1; best_posterior_ic = brain_ics[0]
    for ic in brain_ics:
         power = np.sum(np.abs(mixing_matrix[posterior_idx, ic]))
         if power > max_p:
             max_p = power; best_posterior_ic = ic
             
    max_f = -1; best_frontal_ic = brain_ics[0]
    for ic in brain_ics:
        if ic == best_posterior_ic: continue
        power = np.sum(np.abs(mixing_matrix[frontal_idx, ic]))
        if power > max_f:
             max_f = power; best_frontal_ic = ic
             
    ica_sources = ica.get_sources(epochs).get_data()
    frontal_epochs = ica_sources[:, best_frontal_ic, :]
    posterior_epochs = ica_sources[:, best_posterior_ic, :]
    
    freqs, psds = welch(posterior_epochs, FS, nperseg=int(2*FS) if posterior_epochs.shape[1] >= int(2*FS) else posterior_epochs.shape[1], axis=-1)
    avg_psd = np.mean(psds, axis=0)
    
    fm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=4, 
                       min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
    fm.fit(freqs, avg_psd, FREQ_RANGE)
    r2 = fm.get_metrics('gof', 'rsquared')
    
    # Flatten just for VAR to satisfy the VAR requirements
    frontal_ts = frontal_epochs.flatten()
    posterior_ts = posterior_epochs.flatten()

    # Check VAR 
    data = np.column_stack([frontal_ts, posterior_ts])
    try:
        model = VAR(data)
        res = model.fit(maxlags=1)
        var_crash = np.isnan(res.params).any()
    except:
        var_crash = True
        
    return r2, var_crash

path_modma = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/MODMA/DDS-MODMA/derivatives/epochs") 
hc_files = [f for f in path_modma.glob("*epo.fif") if not ('4' in f.name.split('-')[0].split('_')[0] or '6' in f.name.split('-')[0].split('_')[0] or '10' in f.name.split('-')[0].split('_')[0] or 'M' in f.name.split('-')[0].split('_')[0])]

for f in hc_files[:2]:
    r2, var_crash = process_subject_preflight(f)
    print(f"Subject {f.name}: R2 = {r2:.4f}, VAR_Crash = {var_crash}")
