import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN ---
ROOT = Path(__file__).resolve().parents[1]
DDS_CSV = ROOT / "derivatives/dds_cavanagh/dds_cavanagh_results_stroop_logic.csv"
OUT_REPORT = ROOT / "derivatives/dds_cavanagh/survival_report.csv"

# Umbral de exclusión de sujetos: si tiene menos de N ensayos, se marca para exclusión.
MIN_TRIALS_PER_COND = 10 

def run_survival_audit():
    if not DDS_CSV.exists():
        print(f"[ERROR] No se encuentra el archivo: {DDS_CSV}")
        return

    # 1. Cargar resultados
    df = pd.read_csv(DDS_CSV)
    
    # 2. Contabilidad por Sujeto y Condición
    # Contamos cuántos ensayos (trials) pasaron el filtro de R2 > 0.2
    survival = df.groupby(['subject', 'cond']).size().unstack(fill_value=0)
    
    # 3. Cálculo de Estadísticas de Calidad
    survival['Total_OK'] = survival['Reward'] + survival['Loss']
    survival['Avg_R2'] = df.groupby('subject')['r2'].mean()
    survival['Avg_F1'] = df.groupby('subject')['f1'].mean()
    
    # 4. Criterio de Exclusión (Hecho empírico)
    # Marcamos sujetos que podrían debilitar la potencia estadística del estudio
    survival['Status'] = np.where(
        (survival['Reward'] < MIN_TRIALS_PER_COND) | (survival['Loss'] < MIN_TRIALS_PER_COND),
        "REVISAR (Pocos ensayos)",
        "OK"
    )

    # 5. Informe por pantalla (Trazas de Auditoría)
    print("\n" + "="*50)
    print("INFORME DE SUPERVIVENCIA DE ENSAYOS (DDS)")
    print("="*50)
    print(f"Total de ensayos válidos en el dataset: {len(df)}")
    print(f"Sujetos procesados: {len(survival)}")
    print(f"Media de R2 global: {df['r2'].mean():.3f}")
    print("\nResumen de Sujetos Críticos (Pocos datos):")
    criticos = survival[survival['Status'] != "OK"]
    if not criticos.empty:
        print(criticos[['Reward', 'Loss', 'Avg_R2', 'Status']])
    else:
        print("Todos los sujetos superan el umbral de supervivencia.")
    
    # 6. Guardar reporte
    survival.to_csv(OUT_REPORT)
    print(f"\n[OK] Reporte detallado guardado en: {OUT_REPORT}")

if __name__ == "__main__":
    run_survival_audit()
