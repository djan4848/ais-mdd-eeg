import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# --- MODELO CON OFFSET (C) ---
def dds_model_free(t, A1, g1, f1, phi1, A2, g2, f2, phi2, C):
    return (A1 * np.exp(-g1 * t) * np.sin(2 * np.pi * f1 * t + phi1) +
            A2 * np.exp(-g2 * t) * np.sin(2 * np.pi * f2 * t + phi2) + C)

def plot_best_trial_unconstrained(condition="Loss", n_trials_to_scan=20):
    ROOT = Path(__file__).resolve().parents[0]
    # Usamos la ruta verificada en tus trazas
    EPO_FILE = ROOT / "derivatives/epochs/sub-507_task-ps_epo.fif"
    
    epochs = mne.read_epochs(EPO_FILE, preload=True, verbose=False)
    evoked = epochs[condition]
    times = evoked.times
    mask = (times >= 0.0) & (times <= 0.600)
    t_win = times[mask] - 0.0
    data = np.squeeze(evoked.get_data(picks="Fz"))
    
    best_r2 = -np.inf
    best_res = None

    print(f"Buscando en {n_trials_to_scan} ensayos sin restricciones...")
    
    for i in range(min(n_trials_to_scan, data.shape[0])):
        y_real = data[i, mask] * 1e6
        
        # p0 sin boundaries: [A1, g1, f1, phi1, A2, g2, f2, phi2, C]
        # Usamos valores genéricos pero razonables
        p0 = [np.ptp(y_real), 5.0, 5.0, 0.0, np.ptp(y_real)/2, 15.0, 15.0, 0.0, np.mean(y_real)]
        
        try:
            # AJUSTE SIN BOUNDS
            popt, _ = curve_fit(dds_model_free, t_win, y_real, p0=p0, maxfev=100000)
            y_fit = dds_model_free(t_win, *popt)
            r2 = 1 - np.sum((y_real - y_fit)**2) / (np.sum((y_real - np.mean(y_real))**2) + 1e-10)
            
            if r2 > best_r2:
                best_r2 = r2
                best_res = (y_real, y_fit, popt, i)
        except:
            continue

    if best_res:
        y_real, y_fit, popt, idx = best_res
        print(f"Mejor R2: {best_r2:.4f} en ensayo {idx}")
        print(f"Frecuencia detectada f1: {popt[2]:.2f} Hz | f2: {popt[6]:.2f} Hz")
        
        plt.figure(figsize=(10, 5))
        plt.plot(t_win*1000, y_real, 'k', alpha=0.3, label='Real')
        plt.plot(t_win*1000, y_fit, 'r--', label=f'DDS Free (R2: {best_r2:.3f})')
        plt.title(f"Ajuste Sin Restricciones - Sujeto 507 - Ensayo {idx}")
        plt.legend(); plt.show()
    else:
        print("Error: No se encontró convergencia incluso sin límites.")

if __name__ == "__main__":
    plot_best_trial_unconstrained()
