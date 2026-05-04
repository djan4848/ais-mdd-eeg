import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# --- CARGA Y LIMPIEZA ---
FILE = "erp_it_master_results.csv"
df = pd.read_csv(FILE)

# 1. Corrección de Escala (uV reales)
amp_cols = [c for c in df.columns if 'amp' in c or 'peak' in c]
df[amp_cols] = df[amp_cols] / 1e6

# 2. Asignación de Grupos (Deducción por ID)
# 500-599: Control (HC) | 600-699: Depresión (MDD)
df['group'] = np.where(df['subject'] < 600, 'Control', 'Depression')

# 3. Promediado por Sujeto (Evitar sesgo de número de trials)
# Agrupamos por sujeto y condición para tener una métrica por persona
subj_df = df.groupby(['subject', 'group', 'cond']).agg({
    'Fz_mean_amp_uV': 'mean',
    'AIS_Fz': 'mean',
    'TE_Fz_to_Sink': 'mean',
    'PID_synergy': 'mean'
}).reset_index()

# --- ANÁLISIS ESTADÍSTICO Y VISUALIZACIÓN ---
metrics = ['Fz_mean_amp_uV', 'AIS_Fz', 'TE_Fz_to_Sink', 'PID_synergy']
titles = ['Amplitud ERP (Fz)', 'Almacenamiento Info (AIS)', 'Transferencia Info (TE)', 'Sinergia (PID)']

plt.figure(figsize=(16, 10))

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    
    # Solo comparamos en la condición de "Loss" (típico de Cavanagh para FRN)
    data_loss = subj_df[subj_df['cond'] == 'Loss']
    
    sns.boxplot(x='group', y=metric, data=data_loss, palette='Set2', showfliers=False)
    sns.stripplot(x='group', y=metric, data=data_loss, color='black', alpha=0.3)
    
    # Estadística
    hc = data_loss[data_loss['group'] == 'Control'][metric]
    mdd = data_loss[data_loss['group'] == 'Depression'][metric]
    u_stat, p_val = mannwhitneyu(hc, mdd)
    
    plt.title(f"{titles[i]}\n(p-value: {p_val:.4f})")
    plt.ylabel("Valor")
    plt.xlabel("Grupo")

plt.tight_layout()
plt.savefig("group_comparison_erp_vs_it.png", dpi=300)
print("\n[OK] Gráfica de comparación guardada como 'group_comparison_erp_vs_it.png'")

# --- RESUMEN DE RESULTADOS ---
print("\nRESUMEN DE RESULTADOS (Promedios por Grupo):")
summary = subj_df.groupby(['group', 'cond'])[metrics].mean().round(4)
print(summary)
