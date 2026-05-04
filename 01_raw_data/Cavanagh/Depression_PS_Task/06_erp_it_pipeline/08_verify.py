import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path

# ---------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
EPO_DIR = ROOT / "derivatives" / "epochs"
INPUT = ROOT / "derivatives" / "erp_it_cavanagh"

FILE = INPUT / "erp_it_master_results_fixed.csv"
# Cargamos los hechos ya procesados (con escala corregida)
df = pd.read_csv(FILE)

def verify_cavanagh_with_it(df):
    results = []
    # Analizamos por sujeto para evitar pseudo-replicación
    for subj in df['subject'].unique():
        subj_data = df[df['subject'] == subj]
        group = 'Control' if subj < 600 else 'Depression'
        
        # 1. MI entre Condición (Reward/Loss) y Amplitud Media
        # ¿Cuánto "sabe" el voltaje sobre el tipo de premio?
        mi = mutual_info_classif(subj_data[['Fz_mean_amp_uV']], 
                                 subj_data['cond'].factorize()[0])[0]
        
        results.append({
            'subject': subj,
            'group': group,
            'Stimulus_Info_MI': mi,
            'AIS_mean': subj_data['AIS_Fz'].mean(),
            'Synergy_mean': subj_data['PID_synergy'].mean()
        })
    return pd.DataFrame(results)

# Ejecución de verificación
verif_df = verify_cavanagh_with_it(df)
