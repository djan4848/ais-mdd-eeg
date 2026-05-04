import numpy as np
import mne
import tensorly as tl
from tensorly.decomposition import parafac
from pathlib import Path

# --- CONFIGURACIÓN ---
ROOT = Path(__file__).resolve().parents[1]
EPO_DIR = ROOT / "derivatives" / "epochs"
OUTDIR = ROOT / "derivatives" / "parafac"
OUTDIR.mkdir(exist_ok=True, parents=True)

# Parámetros
N_COMPONENTS = 3          # Número de componentes (ajustar según necesidad)
RANK = N_COMPONENTS
CHANNELS_TO_USE = ['Fz','Cz'  ]  # Si quieres forzar una lista concreta de canales EEG (ej. ['Fz','Cz'])

def build_tensor(epochs, tmin=0, tmax=0.6):
    """Construye tensor (trials × canales × tiempo) usando solo canales EEG."""
    epochs.crop(tmin, tmax)
    # Seleccionar únicamente canales de tipo EEG (excluye EOG, STIM, etc.)
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False, stim=False)
    data = epochs.get_data(picks=picks)  # forma: (n_trials, n_channels_eeg, n_times)
    
    if CHANNELS_TO_USE is not None:
        # Si se especifica una lista, restrigir a esos (deben existir entre los EEG)
        ch_names = [epochs.ch_names[i] for i in picks]
        ch_idx = [ch_names.index(ch) for ch in CHANNELS_TO_USE]
        data = data[:, ch_idx, :]
        picks = [picks[i] for i in ch_idx]  # actualizar picks para info reducido
    
    return data, picks

def main():
    files = sorted(EPO_DIR.glob("*_task-ps_epo.fif"))
    all_results = {}

    for f in files:
        subj = f.stem.split('_')[0].replace('sub-', '')
        print(f"\nProcesando {subj}...")
        epochs = mne.read_epochs(f, preload=True, verbose=False)

        # Filtro pasa banda con configuración que evita el warning
        epochs_filt = epochs.copy().filter(
            1, 40, method='fir', phase='zero',
            fir_design='firwin', l_trans_bandwidth='auto',
            verbose=False
        )

        # Corregir línea base
        epochs_filt.apply_baseline((-0.2, 0))

        for cond in ['Reward', 'Loss']:
            if cond not in epochs_filt.event_id:
                continue
            epochs_cond = epochs_filt[cond]
            
            # Construir tensor y obtener los picks (canales EEG utilizados)
            tensor, picks = build_tensor(epochs_cond, tmin=0, tmax=0.6)
            
            # Normalización opcional (comentar si no se desea)
            # tensor = (tensor - tensor.mean(axis=(1,2), keepdims=True)) / tensor.std(axis=(1,2), keepdims=True)

            # Aplicar PARAFAC
            cp = parafac(tensor, rank=RANK, init='random', tol=1e-6, n_iter_max=500)
            factors = cp.factors  # lista: [trials x rank, canales x rank, tiempo x rank]

            # Guardar factores (opcional, se pueden exportar a .pkl)
            all_results[f"{subj}_{cond}"] = factors

            # Visualización
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, RANK, figsize=(15, 6))
            times = epochs_cond.times * 1000  # convertir a ms
            
            # Crear info reducido para topoplot (solo los canales usados)
            info_reducido = mne.pick_info(epochs_cond.info, sel=picks)
            
            for r in range(RANK):
                # Perfil temporal
                axes[0, r].plot(times, factors[2][:, r])
                axes[0, r].set_title(f'Comp {r+1} - tiempo')
                axes[0, r].set_xlabel('ms')
                
                # Perfil espacial (topografía)
                mne.viz.plot_topomap(factors[1][:, r], info_reducido, axes=axes[1, r], show=False)
                axes[1, r].set_title(f'Comp {r+1} - mapa')
            
            plt.suptitle(f'{subj} {cond}')
            plt.tight_layout()
            plt.savefig(OUTDIR / f'{subj}_{cond}_parafac.png', dpi=100)
            plt.close()

    # Opcional: guardar todos los factores en un archivo para análisis posteriores
    # import pickle
    # with open(OUTDIR / 'factores_parafac.pkl', 'wb') as f:
    #     pickle.dump(all_results, f)

if __name__ == "__main__":
    main()
