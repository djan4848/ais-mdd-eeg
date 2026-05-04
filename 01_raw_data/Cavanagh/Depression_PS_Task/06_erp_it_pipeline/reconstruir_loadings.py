import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN ---
ROOT = Path(__file__).resolve().parents[1]
FACTORS_PKL = ROOT / "derivatives" / "parafac" / "parafac_factors.pkl"
PARTICIPANTS_FILE = ROOT / "participants.tsv"
OUT_CSV = ROOT / "derivatives" / "parafac" / "parafac_loadings_con_grupos.csv"
BDI_THRESHOLD = 14

def load_group_map():
    """Carga BDI desde Original_ID y asigna grupo según umbral."""
    df = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    # Convertir Original_ID a string (sin ceros a la izquierda)
    df['Original_ID'] = df['Original_ID'].astype(str)
    group_map = {}
    for _, row in df.iterrows():
        pid = row['Original_ID']          # ej. "507"
        bdi = row['BDI']
        if pd.isna(bdi):
            group_map[pid] = 'unknown'
        else:
            try:
                bdi = float(bdi)
                if bdi < BDI_THRESHOLD:
                    group_map[pid] = 'CTL'
                else:
                    group_map[pid] = 'DEP'
            except:
                group_map[pid] = 'unknown'
    return group_map

# Cargar grupos usando Original_ID
group_map = load_group_map()
print("Distribución de grupos según BDI (usando Original_ID):")
print(pd.Series(group_map).value_counts())

# Cargar factores
with open(FACTORS_PKL, 'rb') as f:
    factors = pickle.load(f)

# Reconstruir DataFrame de loadings
all_rows = []
for key, data in factors.items():
    # key es algo como "507_Reward"
    subj, cond = key.split('_')
    n_trials, n_comp = data['trials_loadings'].shape
    for comp in range(n_comp):
        for trial in range(n_trials):
            all_rows.append({
                'subject': subj,
                'group': group_map.get(subj, 'unknown'),
                'condition': cond,
                'trial': trial,
                'component': comp + 1,
                'loading': data['trials_loadings'][trial, comp]
            })

df_new = pd.DataFrame(all_rows)
df_new.to_csv(OUT_CSV, index=False)
print(f"\nNuevo archivo guardado en: {OUT_CSV}")
print(f"Total filas: {len(df_new)}")
print(f"Sujetos únicos: {df_new['subject'].nunique()}")
print(f"Grupos en el CSV: {df_new['group'].unique()}")
