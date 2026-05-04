import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURACIÓN ---
ROOT = Path(__file__).resolve().parents[1]
LOADINGS_FILE = ROOT / "derivatives" / "parafac" / "parafac_loadings_con_grupos.csv"
OUTDIR = ROOT / "derivatives" / "parafac" / "estadisticos"
OUTDIR.mkdir(exist_ok=True, parents=True)

# Cargar datos
df = pd.read_csv(LOADINGS_FILE)
print(f"Archivo cargado: {LOADINGS_FILE}")
print(f"Total de filas: {len(df)}")
print(f"Sujetos únicos: {df['subject'].nunique()}")
print(f"Grupos: {df['group'].unique()}")
print(f"Condiciones: {df['condition'].unique()}")
print(f"Componentes: {df['component'].unique()}")

# Excluir sujetos con grupo 'unknown'
df = df[df['group'] != 'unknown'].copy()
print(f"\nDespués de excluir 'unknown': {df['subject'].nunique()} sujetos")

# Calcular promedio por sujeto, condición y componente
df_avg = df.groupby(['subject', 'group', 'condition', 'component'])['loading'].mean().reset_index()

# --- 1. COMPARACIÓN REWARD VS LOSS (pareado) ---
print("\n" + "="*60)
print("📊 COMPARACIÓN REWARD vs LOSS (t-test pareado)")
print("="*60)

resultados_cond = []
for comp in df_avg['component'].unique():
    # Crear tabla pivote: sujeto x condición
    pivot = df_avg[df_avg['component'] == comp].pivot(index='subject', columns='condition', values='loading').dropna()
    if len(pivot) < 2:
        print(f"Componente {comp}: ❌ datos insuficientes")
        continue
    t_stat, p_val = stats.ttest_rel(pivot['Reward'], pivot['Loss'])
    sig = "🌟" if p_val < 0.05 else ""
    print(f"Componente {comp}: t({len(pivot)-1}) = {t_stat:.3f}, p = {p_val:.4f} {sig}")
    resultados_cond.append({
        'componente': comp,
        't_stat': t_stat,
        'df': len(pivot)-1,
        'p_val': p_val,
        'significativo': p_val < 0.05
    })

    # Boxplot
    plt.figure(figsize=(6,4))
    data_plot = df_avg[df_avg['component'] == comp]
    sns.boxplot(x='condition', y='loading', data=data_plot)
    plt.title(f'Componente {comp} - Reward vs Loss')
    plt.tight_layout()
    plt.savefig(OUTDIR / f'boxplot_comp{comp}_reward_loss.png', dpi=100)
    plt.close()

# Guardar resultados condiciones
pd.DataFrame(resultados_cond).to_csv(OUTDIR / 'test_reward_loss.csv', index=False)

# --- 2. COMPARACIÓN CTL VS DEP (independiente) ---
print("\n" + "="*60)
print("📊 COMPARACIÓN CTL vs DEP (t-test independiente)")
print("="*60)

grupos = df_avg['group'].unique()
if 'CTL' in grupos and 'DEP' in grupos:
    resultados_grupo = []
    for cond in df_avg['condition'].unique():
        print(f"\n--- Condición: {cond} ---")
        for comp in df_avg['component'].unique():
            data_cond = df_avg[(df_avg['condition'] == cond) & (df_avg['component'] == comp)]
            ctl = data_cond[data_cond['group'] == 'CTL']['loading'].values
            dep = data_cond[data_cond['group'] == 'DEP']['loading'].values
            if len(ctl) < 2 or len(dep) < 2:
                print(f"Componente {comp}: ❌ n insuficiente (CTL={len(ctl)}, DEP={len(dep)})")
                continue
            t_stat, p_val = stats.ttest_ind(ctl, dep, equal_var=False)
            sig = "🌟" if p_val < 0.05 else ""
            print(f"Componente {comp}: t = {t_stat:.3f}, p = {p_val:.4f} {sig}")
            resultados_grupo.append({
                'condicion': cond,
                'componente': comp,
                't_stat': t_stat,
                'p_val': p_val,
                'n_CTL': len(ctl),
                'n_DEP': len(dep),
                'significativo': p_val < 0.05
            })

            # Boxplot
            plt.figure(figsize=(6,4))
            sns.boxplot(x='group', y='loading', data=data_cond)
            plt.title(f'Componente {comp} - {cond} (CTL vs DEP)')
            plt.tight_layout()
            plt.savefig(OUTDIR / f'boxplot_comp{comp}_{cond}_CTLvsDEP.png', dpi=100)
            plt.close()

    # Guardar resultados grupos
    pd.DataFrame(resultados_grupo).to_csv(OUTDIR / 'test_CTL_vs_DEP.csv', index=False)
else:
    print("\n⚠️  No se encontraron ambos grupos (CTL y DEP).")

print("\n" + "="*60)
print("✅ Análisis completado. Resultados guardados en:")
print(f"   {OUTDIR}")
print("="*60)
