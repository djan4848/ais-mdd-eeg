import numpy as np
import pandas as pd
import mne
from scipy.optimize import curve_fit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPO_DIR = ROOT / "derivatives" / "epochs"
OUTDIR = ROOT / "derivatives" / "dds_cavanagh"
OUTDIR.mkdir(exist_ok=True, parents=True)
OUT_CSV = OUTDIR / "dds_cavanagh_results.csv"

ROIS = {"frontal": ["Fz", "F3", "F4", "AF3", "AF4", "Fp1", "Fp2", "FC3", "FC4"], "cacc": ["Fz", "FCz", "FC2", "AFz", "F2"]}
COND_MAP = {"Reward": "Reward", "Loss": "Loss"}

def dds_model_free(t, A1, g1, f1, phi1, A2, g2, f2, phi2, C):
    return (A1 * np.exp(-g1 * t) * np.sin(2 * np.pi * f1 * t + phi1) +
            A2 * np.exp(-g2 * t) * np.sin(2 * np.pi * f2 * t + phi2) + C)

def fit_dds_blind(t, y):
    # Intentamos dos semillas: una lenta (Delta) y una media (Theta)
    seeds = [
        [np.ptp(y), 5.0, 1.7, 0, np.ptp(y)/2, 15.0, 5.0, 0, np.mean(y)],
        [np.ptp(y), 8.0, 5.0, 0, np.ptp(y)/4, 20.0, 12.0, 0, np.mean(y)]
    ]
    best_res = {k: np.nan for k in ["A1", "gamma1", "f1", "phi1", "A2", "gamma2", "f2", "phi2", "C", "r2", "rmse"]}
    best_r2 = -np.inf

    for p0 in seeds:
        try:
            # AJUSTE TOTALMENTE LIBRE (SIN BOUNDS)
            popt, _ = curve_fit(dds_model_free, t, y, p0=p0, maxfev=100000)
            yhat = dds_model_free(t, *popt)
            r2 = 1.0 - (np.sum((y - yhat)**2) / (np.sum((y - np.mean(y))**2) + 1e-18))
            if r2 > best_r2:
                best_r2 = r2
                best_res = {
                    "A1": popt[0], "gamma1": popt[1], "f1": popt[2], "phi1": popt[3],
                    "A2": popt[4], "gamma2": popt[5], "f2": popt[6], "phi2": popt[7],
                    "C": popt[8], "r2": r2, "rmse": np.sqrt(np.mean((y - yhat)**2))
                }
        except: continue
    return best_res

def main():
    files = sorted(EPO_DIR.glob("*_task-ps_epo.fif"))
    rows = []
    for f in files:
        subj = f.stem.split('_')[0].replace('sub-', '')
        print(f"-> Processing {subj}...")
        try:
            epochs = mne.read_epochs(f, preload=True, verbose=False)
            times = epochs.times
            mask = (times >= 0.0) & (times <= 0.600)
            t_win = times[mask] - 0.0
            
            for raw_cond, paper_cond in COND_MAP.items():
                if raw_cond not in epochs.event_id: continue
                ep_data = np.squeeze(epochs[raw_cond].get_data(picks="Fz"))
                if ep_data.ndim == 1: ep_data = ep_data[np.newaxis, :]

                for trial_idx in range(ep_data.shape[0]):
                    y_trial = ep_data[trial_idx, mask] * 1e6
                    res = fit_dds_blind(t_win, y_trial)
                    res.update({"subject": subj, "cond": paper_cond, "trial": trial_idx})
                    rows.append(res)
        except: continue

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[OK] CSV guardado en {OUT_CSV}")

if __name__ == "__main__":
    main()
