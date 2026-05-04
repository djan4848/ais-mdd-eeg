import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# --- CONFIGURACIÓN DEL TEST ---
SUBJ = "507"
FILE_PATH = f"/media/neuraldyn/PortableSSD/Cavanagh_DEP/Depression_PS_Task/derivatives/epochs/sub-{SUBJ}_task-ps_epo.fif"
T_START, T_END = 0.200, 0.450  # Ventana del componente FRN/RewP

def dds_model_ordered(tr, A1, g1, f1, p1, A2, g2, df, p2, C):
    f2 = f1 + df
    return (A1 * np.exp(-g1 * tr) * np.sin(2 * np.pi * f1 * tr + p1) +
            A2 * np.exp(-g2 * tr) * np.sin(2 * np.pi * f2 * tr + p2) + C)

def run_evoked_test():
    print(f"*** INICIANDO TEST DE EVOCADO: SUJETO {SUBJ} ***")
    
    # 1. Carga y Pre-procesado
    epochs = mne.read_epochs(FILE_PATH, preload=True, verbose=False)
    epochs.filter(l_freq=1.0, h_freq=40.0, l_trans_bandwidth=0.5, verbose=False)
    
    times = epochs.times
    m_win = (times >= T_START) & (times <= T_END)
    m_base = (times >= -0.2) & (times <= 0)
    t_win = times[m_win]
    tr = t_win - t_win[0]

    for cond in ["Reward", "Loss"]:
        if cond not in epochs.event_id: continue
        
        # 2. PROMEDIADO (Grand Average del Sujeto)
        evoked = epochs[cond].average(picks="Fz")
        y_uv = evoked.data[0, :] * 1e6
        
        # Baseline correction sobre el promedio
        y_corr = y_uv[m_win] - np.mean(y_uv[m_base])
        
        # 3. AJUSTE DDS
        ptp_val = np.ptp(y_corr)
        # p0: [A1, g1, f1, p1, A2, g2, df, p2, C]
        p0 = [ptp_val, 5.0, 4.0, 0, ptp_val/2, 10.0, 10.0, 0, 0.0]
        lb = [-1000, 0.1, 0.5, -np.pi, -1000, 0.1, 0.5, -np.pi, -200]
        ub = [ 1000, 150, 12.0, np.pi,  1000, 150, 40.0, np.pi,  200]
        
        try:
            popt, _ = curve_fit(dds_model_ordered, tr, y_corr, p0=p0, bounds=(lb, ub), maxfev=50000)
            yhat = dds_model_ordered(tr, *popt)
            r2 = 1.0 - (np.sum((y_corr - yhat)**2) / (np.sum((y_corr - np.mean(y_corr))**2) + 1e-18))
            
            print(f"\nRESULTADO {cond.upper()}:")
            print(f"  R2: {r2:.3f}")
            print(f"  f1: {popt[2]:.2f} Hz | f2: {popt[2]+popt[6]:.2f} Hz")
            print(f"  A1: {popt[0]:.2f} uV | A2: {popt[4]:.2f} uV")
            
            # 4. VISUALIZACIÓN CRÍTICA
            plt.figure(figsize=(10, 5))
            plt.plot(t_win*1000, y_corr, label='Dato Real (Evocado)', color='blue', linewidth=2)
            plt.plot(t_win*1000, yhat, label=f'Ajuste DDS (R2={r2:.3f})', color='red', linestyle='--')
            plt.title(f"Sujeto {SUBJ} | {cond} | Evoked Fit")
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"  [FALLO] No se pudo ajustar {cond}: {e}")

if __name__ == "__main__":
    run_evoked_test()
