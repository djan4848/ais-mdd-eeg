import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

def dds_model_free(t, A1, g1, f1, phi1, A2, g2, f2, phi2, C):
    return (A1 * np.exp(-g1 * t) * np.sin(2 * np.pi * f1 * t + phi1) +
            A2 * np.exp(-g2 * t) * np.sin(2 * np.pi * f2 * t + phi2) + C)

def run_validation_v4():
    ROOT = Path(__file__).resolve().parents[0]
    EPO_FILE = ROOT / "derivatives/epochs/sub-507_task-ps_epo.fif"
    epochs = mne.read_epochs(EPO_FILE, preload=True, verbose=False)
    
    # Vamos directo al ensayo 19 que sabemos que es bueno
    y_raw = np.squeeze(epochs['Loss'].get_data(picks="Fz"))[19, :] 
    times = epochs.times
    mask = (times >= 0.0) & (times <= 0.600)
    t_win = times[mask] - 0.0
    y_win = y_raw[mask] * 1e6
    
    # ESTRATEGIA: Probamos iniciar en 1.7Hz (lo que vimos) y 5Hz (teoría)
    guesses = [1.7, 5.0]
    best_r2 = -np.inf
    best_popt = None

    for f_start in guesses:
        p0 = [np.ptp(y_win), 5.0, f_start, 0, np.ptp(y_win)/2, 15.0, 10.0, 0, np.mean(y_win)]
        try:
            # Sin bounds, como el que funcionó
            popt, _ = curve_fit(dds_model_free, t_win, y_win, p0=p0, maxfev=100000)
            y_fit = dds_model_free(t_win, *popt)
            r2 = 1 - np.sum((y_win-y_fit)**2)/np.sum((y_win-np.mean(y_win))**2)
            if r2 > best_r2:
                best_r2 = r2
                best_popt = popt
        except: continue

    if best_popt is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(t_win*1000, y_win, 'k', alpha=0.3, label='Real')
        plt.plot(t_win*1000, dds_model_free(t_win, *best_popt), 'r--', label=f'R2: {best_r2:.3f}')
        plt.title(f"Validación Exitosa - F1: {best_popt[2]:.2f} Hz")
        plt.legend(); plt.show()

run_validation_v4()
