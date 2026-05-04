import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# --- CONFIGURACIÓN DE VENTANA ---
# Acotamos a la zona del FRN/RewP para mejorar la convergencia
T_START, T_END = 0.200, 0.450 

ROOT = Path(__file__).resolve().parents[1]
EPO_DIR = ROOT / "derivatives" / "epochs"
OUTDIR = ROOT / "derivatives" / "dds_cavanagh"
OUTDIR.mkdir(exist_ok=True, parents=True)
OUT_CSV = OUTDIR / "dds_cavanagh_windowed.csv"

# Modelo ordenado: f2 = f1 + df (Evita colisiones espectrales)
def dds_model_ordered(tr, A1, g1, f1, p1, A2, g2, df, p2, C):
    f2 = f1 + df
    return (A1 * np.exp(-g1 * tr) * np.sin(2 * np.pi * f1 * tr + p1) +
            A2 * np.exp(-g2 * tr) * np.sin(2 * np.pi * f2 * tr + p2) + C)

def guess_f_fft(tr, y, fmin, fmax, fallback):
    dt = tr[1] - tr[0]
    y_det = y - np.linspace(y[0], y[-1], len(y))
    Y = np.fft.rfft(y_det)
    freqs = np.fft.rfftfreq(len(y), d=dt)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band): return fallback
    return freqs[band][np.argmax(np.abs(Y[band]))]

def main():
    files = sorted(EPO_DIR.glob("*_task-ps_epo.fif"))
    all_rows = []

    print(f"*** DDS VENTANA ACOTADA ({int(T_START*1000)}-{int(T_END*1000)}ms) ***")

    for f in files:
        subj = f.stem.split('_')[0].replace('sub-', '')
        try:
            epochs = mne.read_epochs(f, preload=True, verbose=False)
            # Filtro 1-40Hz para limpiar el camino al optimizador
            epochs_filt = epochs.copy().filter(l_freq=1.0, h_freq=40.0, l_trans_bandwidth=0.5, verbose=False)
            
            times = epochs_filt.times
            m_base = (times >= -0.2) & (times <= 0)
            m_win = (times >= T_START) & (times <= T_END)
            t_win = times[m_win]
            
            for cond in ["Reward", "Loss"]:
                if cond not in epochs_filt.event_id: continue
                data = epochs_filt[cond].get_data(picks="Fz")

                print(f"\n-> Sujeto {subj} | {cond}:")

                for i in range(data.shape[0]):
                    y_full = data[i, 0, :] * 1e6
                    y_win = y_full[m_win]
                    # Baseline individual por trial
                    y_corr = y_win - np.mean(y_full[m_base])
                    tr = t_win - t_win[0]

                    # Semillas por FFT (Lógica Stroop)
                    f1_start = guess_f_fft(tr, y_corr, 1.0, 8.0, 4.0)
                    f2_start = guess_f_fft(tr, y_corr, 8.0, 30.0, 12.0)
                    df_start = max(1.0, f2_start - f1_start)
                    ptp_val = np.ptp(y_corr)

                    # Semillas con escaneo de fase para morder el pico
                    p0 = [ptp_val, 5.0, f1_start, 0, ptp_val/2, 10.0, df_start, 0, 0.0]
                    
                    # Límites de Realidad Física
                    lb = [-1000, 0.1, 0.5, -np.pi, -1000, 0.1, 0.5, -np.pi, -200]
                    ub = [ 1000, 150, 12.0, np.pi,  1000, 150, 40.0, np.pi,  200]

                    try:
                        p0 = np.clip(p0, lb, ub)
                        popt, _ = curve_fit(dds_model_ordered, tr, y_corr, p0=p0, bounds=(lb, ub), maxfev=50000)
                        yhat = dds_model_ordered(tr, *popt)
                        r2 = 1.0 - (np.sum((y_corr - yhat)**2) / (np.sum((y_corr - np.mean(y_corr))**2) + 1e-18))
                        
                        status = "OK" if r2 > 0.15 else "DEBIL"
                        print(f"  [{status}] T{i:02d} | R2:{r2:6.3f} | f1:{popt[2]:5.2f}Hz | A1:{popt[0]:6.1f}uV")

                        all_rows.append({
                            "subject": subj, "cond": cond, "trial": i, "r2": r2,
                            "A1": popt[0], "f1": popt[2], "gamma1": popt[1],
                            "A2": popt[4], "f2": popt[2] + popt[6], "C": popt[8]
                        })
                    except: continue

        except Exception as e:
            print(f"Error en {subj}: {e}")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        print(f"\n[FIN] CSV guardado en {OUT_CSV}")

if __name__ == "__main__":
    main()
