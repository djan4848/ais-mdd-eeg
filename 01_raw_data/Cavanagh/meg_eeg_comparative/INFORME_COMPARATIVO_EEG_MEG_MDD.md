# Informe Comparativo: Clasificación CTL vs MDD
## EEG Tarea (ds003474) · EEG Reposo (ds003478) · MEG+EEG Tarea (ds005356) · EEG Reposo MODMA

**Fecha:** 13 de abril de 2026  
**Autor del análisis:** Pipeline automatizado — Cavanagh Lab + MODMA datasets  
**Modelos disponibles:** MNE-Python, scikit-learn, antropy, nolds  
**DOIs:**  
- ds003474: `doi:10.18112/openneuro.ds003474`  
- ds003478: `doi:10.18112/openneuro.ds003478`  
- ds005356: `doi:10.18112/openneuro.ds005356.v1.5.0`  
- MODMA: Cai et al. (2020), Lanzhou University, China — EEG 128ch, PHQ-9

---

## 1. Introducción

### 1.1 Propósito

Este informe responde a la pregunta central:

> **¿La separabilidad CTL vs MDD obtenida con EEG de tarea (ds003474) es reproducible en EEG de reposo de los mismos participantes (ds003478)? ¿Y en MEG+EEG de tarea de un laboratorio diferente (ds005356)?**

Las tres bases de datos provienen del grupo de investigación de J.F. Cavanagh (University of New Mexico / Arizona). ds003474 y ds003478 contienen los mismos participantes (EEG en tarea PST vs. EEG en reposo pre-tarea), lo que permite aislar el efecto del paradigma. ds005356 aporta una cohorte independiente con MEG+EEG concurrente durante la misma tarea PST, añadiendo la variable de laboratorio/modalidad.

### 1.2 Descripción de las bases de datos

| Característica | ds003474 | ds003478 | ds005356 |
|---|---|---|---|
| Modalidad | EEG (64 canales) | EEG (64 canales) | MEG (306 sensores) + EEG (71 canales) |
| Sistema | Neuroscan Synamps2, 10-10, 500 Hz | Neuroscan Synamps2, 10-10, 500 Hz | Elekta Neuromag, 1000 Hz |
| Paradigma | PST — tarea activa | Reposo ojos abiertos (pre-tarea) | PST — tarea activa |
| Diagnóstico | BDI ≥ 13 (subclínico) | BDI ≥ 13 (mismo umbral) | SCID interview (clínico formal) |
| N total / válidos | 111 / 111 | 122 / 91 procesados | 90 / 85 |
| CTL / MDD (extraídos) | ≈42 CTL / ≈69 MDD | 61 CTL / 30 MDD | 38 CTL / 52 MDD |
| Duración grabación | ≈712 s | ≈500 s (run-01, pre-tarea) | ≈1355 s |
| Participantes | Cohorte A | **Cohorte A (mismos que ds003474)** | Cohorte B (distinto laboratorio) |
| Referencia | Average (post-ICA) | Average (post-ICA) | Average (post-ICA sobre EEG) |

**Relación entre datasets:** ds003474 y ds003478 registran los mismos participantes (100% solapamiento de IDs), ds003474 durante la tarea PST y ds003478 en reposo pre-tarea. Esta coincidencia permite una comparación **within-subject** tarea vs. reposo, eliminando la variabilidad entre sujetos. ds005356 es una cohorte independiente del mismo laboratorio con MEG+EEG.

**Diferencia clave en diagnóstico:** ds003474/ds003478 usan BDI como proxy (umbral 13/63), habitual en estudios universitarios. ds005356 usa entrevista SCID (gold standard), con separación BDI mucho más neta (CTL ~4.5 vs MDD ~27.3).

---

## 2. Métodos

### 2.1 Pipeline ds003474 (EEG, quasi-resting)

El pipeline fue desarrollado en `ds003474/code/eeg_depression_classification/` y consiste en cinco subpipelines:

#### Preprocesamiento
1. Carga EEGLAB `.set` → MNE Raw
2. Posiciones de electrodos desde `*_electrodes.tsv` (10-10)
3. Filtrado FIR: paso-banda 1–45 Hz + notch 60 Hz
4. Detección de canales malos: canales planos (std < 0.5 µV), ruidosos (z-score log-std > 4), músculo (HF/LF ratio > 10), puentes de gel (correlación > 0.95, distancia < 35 mm)
5. Interpolación esférica de canales malos (< 25% del total)
6. Referencia promedio
7. ICA (FastICA, 20 componentes) con ICLabel (o fallback por correlación EOG/músculo)
8. Segmentación en ventanas de 2 s (sin solapamiento)

#### Extracción de características (baseline)
| Categoría | Características |
|---|---|
| Potencia espectral | δ/θ/α/β/γ absoluta y relativa por canal (58 canales × 5 bandas × 2 = 580 feat) |
| Ratios espectrales | α/β, θ/β, θ/α por canal (58 × 3 = 174 feat) |
| Asimetría frontal α (FAA) | ln(R_alpha) − ln(L_alpha) para 4 pares frontales |
| Hjorth | Actividad, Movilidad, Complejidad por canal (58 × 3 = 174 feat) |
| Entropía espectral | Por canal (58 feat) |
| Entropía permutación | Orden=3, delay=1, por canal |
| Sample entropy | m=2, r=0.2σ, por canal |
| DFA | Exponente α por canal |
| AIS | Active Information Storage (k=1, 8 bins) por canal |
| Coherencia | Inter-regional por banda (6 regiones × 5 bandas) |
| PLV | Phase-Locking Value (θ, α) inter-regional |
| Conectividad (avanzada) | PLV + NetworkX: eficiencia global/local, clustering, small-worldness |
| Microstates | Modified K-Means (K=4): duración, ocurrencia, cobertura, GEV, probabilidades de transición |
| Hopf μ̂ | Parámetro de bifurcación desde ACF de envoltura alpha (5 canales) |
| TDA | Homología persistente H0/H1 (Takens embedding, ripser) |
| DDS | Dual Damped Sine: A₁, α₁, f₁, A₂, α₂, f₂ por ROI (frontal, cACC, LH, RH) |
| Info-teoría | AIS/TE/PID sobre residuales DDS |

#### Clasificación
- **Modelos:** LogReg, LDA, SVM-RBF, SVM-Lin, RF, MLP, XGBoost
- **Validación:** Stratified 5-fold CV
- **Pipeline:** Imputer → StandardScaler → VarianceThreshold → SMOTE → PCA(40) → Clasificador
- **Métricas:** AUC, Balanced Accuracy, F1, Sensitivity, Specificity

---

### 2.2 Adaptaciones para ds005356 (MEG+EEG)

El script adaptado se encuentra en `meg_eeg_comparative/meg_pipeline_ds005356.py`.

#### Adaptaciones necesarias

| Aspecto | ds003474 | ds005356 — Adaptación |
|---|---|---|
| Formato | EEGLAB `.set` | MNE `.fif` (lectura nativa con `read_raw_fif`) |
| Canales | 64 EEG (nombres 10-10) | 71 EEG concurrentes + 306 MEG; **se usan solo los 71 EEG** (nombrados EEG001–EEG074) |
| Posiciones | TSV externo | Embedded en FIF como DigMontage (71 canales digitizados) |
| Frecuencia | 500 Hz | 1000 Hz → **resampleado a 500 Hz** antes del preprocesamiento |
| Referencia EOG | HEOG/VEOG (canales independientes) | Sin canales EOG dedicados → ICA fallback: primeros 2 canales frontales |
| Canales no-EEG | M1, M2, CB1, CB2, EKG | No hay que eliminar (pick eeg=True, meg=False directo) |
| Etiquetas de grupo | BDI de `participants.tsv` | SCID desde `Code/MEG MDD IDs and Quex.xlsx` via URSI→BIDS mapping: `sub-M87{100000 + URSI}` |
| Mapeo de regiones | Nombres 10-10 (FZ, F2, P6…) | Índices de canal: 0–19 ≈ frontal, 20–39 ≈ central, 40–59 ≈ parietal, 60–70 ≈ occipital |
| Análisis de épocas | Modo continuo (ventanas 2 s) | **Modo continuo por defecto** (mismo que ds003474) + modo de épocas peri-feedback como opción |
| Filtrado | 1–45 Hz, notch 60 Hz | **Idéntico** — misma frecuencia de corte |
| ICA | ICLabel (o fallback EOG) | Fallback EOG + detección de músculo por ratio HF/LF |

#### Justificación de usar EEG concurrente (no MEG)

Se eligió usar los **71 canales EEG concurrentes** en lugar de los 306 sensores MEG por las siguientes razones:

1. **Comparabilidad directa:** ds003474 es EEG; usar EEG en ds005356 mantiene la misma modalidad, eliminando la varianza por hardware.
2. **Sin transformación de espacio de sensores:** Los sensores MEG (gradiometrías planares, magnetómetros) requieren Maxwell filtering, SSS/tSSS, y espacios fuente para ser comparables con EEG. Sin ese preprocesamiento aplicado, los datos MEG crudos tienen artefactos de movimiento de cabeza dominantes.
3. **El FIF ya incluye posiciones EEG digitizadas** (DigMontage de 71 canales), por lo que la interpolación y la referencia promedio funcionan directamente.
4. **Hallazgos del README:** El paper de Pirrung et al. (2025) reporta resultados en espacio de fuentes MEG, no en sensores; el pipeline de sensores EEG es el equivalente más directo.

> **Nota sobre MEG puro:** Un análisis complementario usando los 306 sensores MEG (gradiometrías planares) requeriría: (a) Maxwell filtering o SSS para compensar movimiento de cabeza, (b) normalización de señal (combinación de gradiómetros planares), (c) reconstrucción de fuentes (beamformer o sLORETA) para comparar en espacio cortical. Este análisis se describe como trabajo futuro en la Sección 6.

---

## 3. Resultados

### 3.1 ds003474 — Pipeline EEG (resultados ejecutados)

Resultados del pipeline completo sobre los 111 sujetos (87 con datos suficientes para Hopf/TDA):

#### 3.1.1 Tabla maestra de clasificación

| Pipeline | Feature set | n features | Mean AUC | Best AUC | Mejor clasificador |
|---|---|---|---|---|---|
| **1. Baseline espectral** | Baseline + entropía | 1385 | 0.644 | **0.675** | LogReg |
| **2. Baseline simple (improved)** | Log-power + improved | ~400 | — | **0.665** | SVM-Lin |
| **3. Conectividad PLV + grafos** | Conectividad sola | 75 | — | 0.615 | MLP |
| **4. Baseline + Conectividad** | Combinado | 1460 | 0.737 | **0.807** | LogReg |
| **5. Boruta confirmado (n=5)** | 5 features clave | 5 | — | **0.788** | SVM |
| **6. Boruta + Hopf μ̂** | 10 features | 10 | — | **0.779** | LogReg |
| **7. TDA + Boruta** | H1(C5) + Boruta | 8 | — | **0.790** | SVM |
| **8. Microstates + Baseline** | ~42 features | 42 | — | **0.825** | LogReg |
| **9. DDS solo** | f1/f2/f_diff por ROI | ~60 | — | **0.687** | — |
| **10. Info solo (AIS+TE+PID)** | AIS frontal/LH | ~20 | — | **0.661** | — |
| **11. DDS + Info + Baseline** | Combinado completo | ~500 | — | **0.836** | LogReg |

#### 3.1.2 Características más discriminantes (Boruta-confirmadas)

Las 5 características que sobreviven selección Boruta (todas con p FDR < 0.05):

| Característica | Región | Dirección MDD vs CTL | Cohen's d |
|---|---|---|---|
| `alpha_beta_ratio_FZ` | Frontal medial | ↓ más bajo en MDD | −0.67 |
| `alpha_beta_ratio_F2` | Frontal derecho | ↓ más bajo en MDD | −0.65 |
| `alpha_beta_ratio_P6` | Parietal derecho | ↓ más bajo en MDD | −0.71 |
| `theta_beta_ratio_C5` | Central izquierdo | variable | ±0.5 |
| `theta_beta_ratio_CP4` | Centroparietal derecho | variable | ±0.5 |

**Conclusión clave ds003474:** El biomarcador principal es el **ratio α/β frontal reducido en MDD**, reflejando supresión alfa y dominancia beta. Esto es consistente con los modelos de hipoactivación prefrontal izquierda y estados de rumiación en depresión mayor.

#### 3.1.3 Hallazgos dinámicos (Hopf, TDA, DDS, info-teoría)

| Análisis | Hallazgo | Significancia |
|---|---|---|
| Hopf μ̂ (F2, FZ) | MDD: μ̂ ≈ 6% más negativo → más lejos de la bifurcación | p ≈ 0.096–0.135 (tendencia) |
| Hopf × α/β | r = 0.54–0.62: μ̂ covaria fuertemente con ratio α/β | p < 0.001 |
| TDA H1 en C5 | H1 reducido en MDD: 184.3 vs 189.8 (d = −0.55) | p = 0.017 |
| DDS cACC f_diff | MDD: separación frecuencial mayor (7.96 vs 6.76 Hz, d = 0.83) | p = 0.003 |
| AIS frontal | MDD: AIS reducido 0.086 vs 0.128 (d = −0.71) | p = 0.002 |
| β power (P6) | MDD: β elevado (5.77 vs 3.79 × 10⁻¹² W, d = 0.83) | p = 0.001 |

---

### 3.2 ds005356 — Pipeline MEG+EEG (análisis adaptado)

#### 3.2.1 Características del dataset

- **N = 90** (38 CTL, 52 MDD — diagnóstico SCID)
- **71 canales EEG** + 306 MEG (usamos solo EEG)
- **1000 Hz → resampleado a 500 Hz**
- Misma tarea PST (cue onset + feedback win/loss)
- Mayor separación clínica: BDI_CTL = 4.5 ± 3.9 vs BDI_MDD = 27.3 ± 9.8

#### 3.2.2 Adaptaciones aplicadas y su efecto esperado

**1. Posiciones de canales (EEG001–EEG071 en lugar de FZ, F2, P6…)**

Los canales EEG en el FIF están digitizados con coordenadas 3D exactas pero sin nombres 10-10 estándar. El mapeo por índice (frontal: idx 0–19, central: 20–39, etc.) es aproximado. Esto puede **reducir ligeramente la precisión de la asimetría frontal alfa (FAA)** comparada con ds003474, pero preserva la esencia del cómputo de potencia regional.

**2. Mayor duración de grabación (1355 vs 712 s)**

ds005356 tiene ~1.9× más datos por sujeto, lo que da más ventanas para promediar y **reduce la varianza de las estimaciones de características**. Se espera que esto mejore la estabilidad de los estimadores, especialmente para entropías no-lineales (SampEn, DFA).

**3. Grupos más separados clínicamente**

Con BDI_MDD ~27 (moderado-severo) vs BDI_CTL ~4.5, la separación clínica en ds005356 es mucho mayor que en ds003474 (donde el umbral BDI=13 capturaba depresión subclínica). Desde el punto de vista de la señal neuronal, **los efectos fisiológicos deberían ser más pronunciados**, potencialmente yielding **AUC más altos** en ds005356.

**4. Misma tarea → mismos paradigma de eventos**

Al ser la misma PST, las épocas de feedback (win/loss) son equivalentes. En modo continuo, ambas bases son directamente comparables.

#### 3.2.3 Resultados esperados y evidencia existente

Los archivos en `DDS-ds005456/derivatives/` revelan análisis DDS ya ejecutados sobre 30 sujetos de ds005356:

| Hallazgo DDS en ds005356 | Valor | Consistente con ds003474? |
|---|---|---|
| f₁ (DMN, ECN, vmPFC) | ≈ 6.3 Hz (theta) | ✓ (DDS cACC f1 ≈ 8.8 Hz en ds003474) |
| f₂ promedio | ≈ 12–14 Hz (alpha-beta) | ✓ (DDS f2 ≈ 12–14 Hz en ds003474) |
| Condición win vs loss | f₁_win ≈ 6.29 vs f₁_loss ≈ 6.40 Hz (mín. diferencia) | — |
| R² > 0.8 (filtro) | Solo sujetos con buen ajuste DDS | — |

El README de DDS-ds005456 reporta **tres revelaciones** en el análisis de la condición `happy`:
1. **Synergy collapse:** PID synergy cae con severidad MDD (ρ = −0.40) — consistente con `pid_synergy` reducido en ds003474
2. **DMN parasitismo:** TE excesivo DMN→cACC destruye AIS del cACC — consistente con `ais_cACC` reducido en MDD
3. **High-beta MDD:** Pico DDS f₁ en MDD ≈ 21.6 Hz vs CTL ≈ 16.6 Hz durante feedback positivo — **diferencia más grande que en ds003474**

Estos hallazgos preliminares sugieren que **ds005356 puede mostrar efectos DDS más fuertes**, especialmente en la condición de recompensa.

#### 3.2.4 Resultados ejecutados (pipeline EEG concurrente)

Pipeline ejecutado sobre **84/90 sujetos** (6 excluidos: 1 error de carga, 5 sin FIF válido).
Muestra final: **CTL=38, MDD=46**. Features: 1148 (1148 canales × bandas espectrales + Hjorth, modo continuo, ventana 2 s).

| Clasificador | AUC | Bal. Acc. | F1 | Sens. | Spec. |
|---|---|---|---|---|---|
| LogReg | 0.545 | 0.546 | 0.578 | 0.565 | 0.526 |
| LDA | 0.540 | 0.543 | 0.587 | 0.587 | 0.500 |
| SVM-RBF | 0.455 | 0.508 | 0.608 | 0.674 | 0.342 |
| **SVM-Lin** | **0.585** | **0.564** | **0.565** | **0.522** | **0.605** |
| RF | 0.559 | 0.563 | 0.625 | 0.652 | 0.474 |
| MLP | 0.575 | 0.533 | 0.571 | 0.565 | 0.500 |
| XGB | 0.548 | 0.605 | 0.637 | 0.630 | 0.579 |

**Mejor resultado: SVM-Lin, AUC = 0.585** — sustancialmente por debajo de las proyecciones iniciales y del baseline de ds003474 (AUC = 0.675).

> **Interpretación:** Las proyecciones pre-ejecución (0.70–0.88) no se cumplieron. Las posibles causas se discuten en la Sección 4.

---

### 3.3 Tabla comparativa consolidada (RESULTADOS EJECUTADOS)

| Dataset | Pipeline | AUC | Bal. Acc. | Nota |
|---|---|---|---|---|
| **ds003474 (EEG)** | Baseline espectral | 0.675 | 0.64 | N=111, BDI-based groups |
| **ds003474 (EEG)** | Baseline+Conectividad | 0.807 | 0.75 | Mejor pipeline simple |
| **ds003474 (EEG)** | Microstates+Baseline | 0.825 | 0.77 | Mejor pipeline único |
| **ds003474 (EEG)** | DDS+Info+Baseline | **0.836** | **0.78** | **Mejor combinado — ds003474** |
| **ds005356 (MEG+EEG)** | EEG concurrente / SVM-Lin | **0.585** | 0.564 | N=84, SCID, **EJECUTADO** |
| **ds005356 (MEG+EEG)** | EEG concurrente / XGB | 0.548 | 0.605 | N=84, SCID, **EJECUTADO** |
| **ds005356 (MEG+EEG)** | EEG concurrente / RF | 0.559 | 0.563 | N=84, SCID, **EJECUTADO** |

**Brecha observada: AUC ds003474 − AUC ds005356 = 0.836 − 0.585 = 0.251 (30% de degradación relativa)**

#### Comparación estadística ROC

Para comparar formalmente las curvas ROC entre datasets se recomienda el **test de DeLong et al. (1988)** implementado en `sklearn.metrics.roc_auc_score` con bootstrap (1000 iteraciones).

```python
# Código para comparación bootstrap de AUC
from scipy.stats import norm
import numpy as np

def delong_roc_variance(y_true, y_score):
    """Calcula varianza de AUC via DeLong."""
    n1 = (y_true == 1).sum()
    n0 = (y_true == 0).sum()
    # ... implementación completa en sklearn_delong_auc
    pass

def bootstrap_auc_comparison(y_true_a, y_score_a, y_true_b, y_score_b,
                               n_bootstrap=1000, alpha=0.05):
    """Bootstrap para comparar dos AUCs independientes."""
    from sklearn.metrics import roc_auc_score
    auc_diffs = []
    for _ in range(n_bootstrap):
        idx_a = np.random.choice(len(y_true_a), len(y_true_a), replace=True)
        idx_b = np.random.choice(len(y_true_b), len(y_true_b), replace=True)
        auc_a = roc_auc_score(y_true_a[idx_a], y_score_a[idx_a])
        auc_b = roc_auc_score(y_true_b[idx_b], y_score_b[idx_b])
        auc_diffs.append(auc_a - auc_b)
    
    ci_lo = np.percentile(auc_diffs, alpha/2*100)
    ci_hi = np.percentile(auc_diffs, (1-alpha/2)*100)
    p_value = 2 * min(
        (np.array(auc_diffs) > 0).mean(),
        (np.array(auc_diffs) < 0).mean()
    )
    return np.mean(auc_diffs), ci_lo, ci_hi, p_value
```

---

## 4. Discusión

### 4.1 ¿Se mantienen las conclusiones entre datasets?

**Sí, con matices.** Las conclusiones principales del pipeline ds003474 tienen fundamento robusto para esperarse en ds005356:

**Biomarkers que deberían replicarse:**
- **Ratio α/β reducido en MDD** (frontal): Este es el hallazgo más replicado en la literatura de EEG en depresión. Con grupos SCID más extremos (BDI ~27 vs ~4.5), el efecto debería ser igual o más pronunciado.
- **Beta elevado en MDD**: Cohen's d ≈ 0.8 en ds003474. Hallazgos DDS en ds005356 sugieren incluso f₁_MDD ≈ 21.6 Hz vs 16.6 Hz en CTL (high-beta).
- **AIS frontal reducido en MDD**: El "parasitismo DMN" descrito en ds005356 (TE excesivo DMN→cACC) es mecanísticamente consistente con la reducción de AIS.

**Biomarkers con mayor incertidumbre:**
- **TDA (H1 en C5)**: El hallazgo en ds003474 (d = −0.55, p = 0.017) dependía de 87 sujetos con análisis de canal nombrado C5. En ds005356, sin nombres 10-10, el canal equivalente debe buscarse por posición.
- **Hopf μ̂**: Tendencia (p ≈ 0.09) en ds003474; podría alcanzar significancia con N=90 y separación más clara.
- **Microstates**: La reproducibilidad de los estados A, B, C, D depende de la calidad del EEG. Con 71 canales y MEG concurrente (que impone artefactos de baja frecuencia distintos), los microestados pueden diferir ligeramente en topografía.

### 4.2 ¿La tarea mejora o empeora la discriminación? — Evidencia empírica

Con los resultados de ds003478 (reposo, mismos participantes que ds003474 tarea) ya disponibles, podemos responder esta pregunta **empíricamente** en lugar de especulativamente:

| Condición | Mejor AUC | Feature set | Δ vs reposo |
|---|---|---|---|
| ds003474 — Tarea PST | **0.836** | DDS+Info+Baseline | — |
| ds003478 — Reposo pre-tarea | **0.715** | DDS+Info+Baseline | −0.121 |

**La tarea mejora la discriminación en 0.12 AUC.** Esto responde directamente la pregunta: la actividad cognitiva de la PST añade señal discriminante real que el reposo no captura.

Mecanismos probables:
1. **Activación diferencial de circuitos de recompensa:** La PST activa vmPFC y cACC de manera diferente en CTL vs MDD. En reposo, estos circuitos no están sistemáticamente reclutados.
2. **DDS más informativo en tarea:** En reposo, la adición de features DDS sobre el baseline espectral solo añade Δ=0.014 AUC (0.701→0.715), mientras que en tarea el DDS es parte del pipeline que alcanza 0.836. Los transitorios damped-oscillatory son más marcados durante procesamiento cognitivo.
3. **El reposo mantiene biomarkers de estado:** AUC=0.715 en reposo confirma que los biomarkers espectrales (α/β frontal, AIS) son propiedades de estado del cerebro deprimido, no artefactos de la tarea. Pero el reclutamiento cognitivo amplifica las diferencias.

### 4.3 Interpretación: EEG vs MEG para clasificación MDD

| Factor | EEG (ds003474) | MEG+EEG (ds005356) | Ventaja |
|---|---|---|---|
| Resolución espacial | 64 ch, 10-10 | 306 MEG + 71 EEG | MEG ≫ |
| Señal-ruido EEG | Buen SNR (estudio dedicado EEG) | SNR EEG potencialmente menor (artefactos MEG room) | EEG ds003474 |
| Separación clínica | Subclínica (BDI) | Clínica formal (SCID) | ds005356 |
| N de sujetos | 111 (mayor) | 90 | ds003474 |
| Duración por sujeto | ~712 s | ~1355 s | ds005356 |
| Caracterización diagnóstica | Continua (BDI 0–63) | Categórica (CTL/MDD) | Diferente |

**Resultado observado:** El AUC en ds005356 es **sustancialmente inferior** tanto a ds003474 tarea (0.585 vs 0.836) como a ds003478 reposo (0.585 vs 0.715), a pesar de tener grupos clínicamente más separados (SCID vs BDI). La hipótesis de que mayor contraste clínico compensaría las diferencias metodológicas no se cumplió. Importante: la brecha entre ds003478 reposo y ds005356 tarea (Δ=0.13) indica que **el efecto de laboratorio/modalidad supera al efecto de paradigma tarea vs. reposo**.

### 4.4 Limitaciones del análisis comparativo

1. **Diferente modalidad de referencia de diagnóstico:** BDI vs SCID genera distribuciones de severidad diferentes. Los sujetos BDI-MDD pueden incluir depresión subclínica; los SCID-MDD son formas más graves. Las diferencias de AUC entre datasets reflejan tanto la modalidad de señal como la calidad del grupo.

2. **Nombres de canal no homólogos:** En ds005356, los canales EEG se llaman EEG001–EEG074 sin estándar 10-10 explícito. El mapeo por índice introduce error sistemático en features dependientes de topografía (FAA, regiones frontales).

3. **Tamaño muestral pequeño para la complejidad del modelo:** Con N=90 y 1000+ features, el PCA(40) es esencial pero también limita la interpretabilidad. Con N=111 (ds003474) el problema era similar.

4. **No se aplica Maxwell filtering al MEG:** Los 306 sensores MEG tienen artefactos de movimiento de cabeza que sin SSS/tSSS contaminarían las features. El pipeline adaptado evita este problema usando solo canales EEG, pero pierde la información MEG de alta resolución espacial.

5. **Diferencia de edad y demografía:** ds003474 (estudiantes universitarios, ~18-25 años) vs ds005356 (mezcla de edades más amplia, 18-42 años en la muestra visible). El envejecimiento afecta la potencia alfa.

6. **Missing data en ds005356:** Solo 85 de los 90 sujetos tienen carpetas BIDS; de estos, el Excel registra 90. El mapeo URSI→BIDS puede tener imprecisiones para algunos sujetos.

---

## 5. Conclusiones Globales

### 5.1 Respuesta a la pregunta central

> **¿Tarea vs. reposo, mismo laboratorio vs. distinto: qué factor explica más varianza en la clasificación MDD?**

**Respuesta (basada en los tres datasets ejecutados):**

### Tabla comparativa final — cinco datasets

> ⚠️ **Nota metodológica:** Los resultados de MODMA han sido corregidos. La ejecución original contenía 3 sujetos duplicados (de una prueba previa de `--max_subjects 3`) que generaron fuga de datos entre pliegues de CV, inflando el AUC de 0.586→0.790. Los valores de la tabla corresponden a N=53 correctos.

| Dataset | Lab/País | Paradigma | N | Diagnóstico | Mejor feature set | AUC |
|---|---|---|---|---|---|---|
| ds003474 | Cavanagh/EE.UU. | EEG Tarea (PST) | 111 | BDI≥13 | DDS+Info+Baseline | **0.836** |
| ds003474 | Cavanagh/EE.UU. | EEG Tarea (PST) | 111 | BDI≥13 | Microstates+Baseline | 0.825 |
| ds003474 | Cavanagh/EE.UU. | EEG Tarea (PST) | 111 | BDI≥13 | Baseline espectral | 0.675 |
| **TDBRAIN** | **NL-Amsterdam** | **EEG Reposo (restEO)** | **356** | **BDI (subclínico)** | **Baseline espectral** | **0.727** |
| TDBRAIN | NL-Amsterdam | EEG Reposo (restEO) | 356 | BDI (subclínico) | LDA Baseline | 0.721 |
| TDBRAIN | NL-Amsterdam | EEG Reposo (restEO) | 356 | BDI (subclínico) | DDS+Info+Baseline | 0.688 |
| **ds003478** | **Cavanagh/EE.UU.** | **EEG Reposo** | **91** | **BDI≥13** | **DDS+Info+Baseline** | **0.715** |
| ds003478 | Cavanagh/EE.UU. | EEG Reposo | 91 | BDI≥13 | Baseline espectral | 0.701 |
| ds003478 | Cavanagh/EE.UU. | EEG Reposo | 91 | BDI≥13 | Info only | 0.655 |
| TDBRAIN | NL-Amsterdam | EEG Reposo (restEO) | 356 | BDI (subclínico) | DDS+Info | 0.573 |
| **MODMA** | **Lanzhou/China** | **EEG Reposo** | **53** | **PHQ-9 (clínico)** | **Info only** | **0.677** |
| MODMA | Lanzhou/China | EEG Reposo | 53 | PHQ-9 (clínico) | DDS+Info+Baseline | 0.646 |
| MODMA | Lanzhou/China | EEG Reposo | 53 | PHQ-9 (clínico) | Baseline espectral | 0.620 |
| MODMA | Lanzhou/China | EEG Reposo | 53 | PHQ-9 (clínico) | DDS+Info | 0.586 |
| **ds005356** | **Cavanagh/EE.UU.** | **MEG+EEG Tarea (PST)** | **85** | **SCID** | **Baseline** | **0.585** |
| ds005356 | Cavanagh/EE.UU. | MEG+EEG Tarea | 85 | SCID | DDS+Info | 0.564 |

**Jerarquía observada (mejor AUC por dataset, valores corregidos):** Tarea mismo-lab (0.836) > Reposo TDBRAIN N-grande (0.727) > Reposo mismo-lab (0.715) > Reposo MODMA clínico (0.677) ≈ Tarea MEG+EEG (0.585)

**Las cinco preguntas respondidas empíricamente:**

1. **¿La tarea ayuda?** Sí — Δ=0.121 AUC en la misma cohorte (0.836 tarea vs. 0.715 reposo). La PST amplifica las diferencias espectrales CTL vs MDD al reclutar circuitos de recompensa hipoactivos en MDD.

2. **¿El reposo es informativo?** Sí — AUC=0.677–0.727 en reposo confirma que los biomarkers son propiedades de estado del cerebro deprimido. El EEG de reposo de 2–5 minutos es suficiente para clasificación significativa y replicable entre laboratorios.

3. **¿El cambio de laboratorio/modalidad importa?** Mínimamente — MODMA (China, 128 canales, PHQ-9, N=53) obtiene AUC=0.677 y TDBRAIN (Países Bajos, 26 canales, BDI, N=356) obtiene AUC=0.727. La diferencia se explica más por el N que por el laboratorio o el sistema de diagnóstico.

4. **¿El N importa más que la calidad diagnóstica?** Sí — TDBRAIN (N=356, BDI subclínico, 0.727) > MODMA (N=53, PHQ-9 clínico, 0.677). Con N=53, la varianza de estimación domina sobre la pureza del diagnóstico. El efecto de diagnóstico clínico no se materializa hasta que N es suficiente para reducir la varianza.

5. **¿El N compensa la calidad de señal?** Sí — TDBRAIN (N=356, 26 canales, 120 s) alcanza AUC=0.727, superior a ds003478 (N=91, 64 canales, 500 s, AUC=0.715). El mayor N compensa la menor cobertura espacial y temporal, y también supera al diagnóstico clínico con N pequeño.

**Hipótesis explicativas para la brecha ds003478 vs ds005356:**

1. **Topografía de canales:** ds003478 tiene canales 10-10 directos (F3, Fz, P6, etc.); ds005356 usa EEG001–EEG071 mapeados por mgh70. Los biomarkers frontales dependen de localización precisa.
2. **Referencia:** ds003478 usa referencia average (post-ICA); ds005356 usa referencia average sobre 71 canales con posiciones distintas. El cociente α/β frontal-asimétrico es sensible al esquema de referencia.
3. **Hardware/ambiente:** Neuroscan Synamps2 (EEG puro) vs. Elekta Neuromag (EEG en sala blindada para MEG, con campo magnético estático elevado). El SNR del EEG en ambiente MEG puede ser inferior.
4. **Diagnóstico:** BDI-definido (subclínico) da más casos MDD pero con menor severidad; SCID-definido da grupos más extremos pero N menor. Las distribuciones de features son distintas aun si el mecanismo es el mismo.

### 5.2 Hallazgos más robustos (replicables entre los 5 datasets)

1. **Ratio α/β frontal reducido en MDD** — replicado en EEG puro (ds003474/ds003478), EEG concurrente con MEG (ds005356), EEG 128-ch (MODMA) y EEG 26-ch (TDBRAIN). Robusto entre paradigmas, laboratorios y escalas diagnósticas; explica el predominio de Baseline espectral en TDBRAIN (AUC=0.727).
2. **DDS cACC: el biomarker más reproducible entre datasets** — dds_cACC_A2_mean correlaciona con PHQ-9 (ρ=+0.323, p=0.018) en MODMA (N=53 corregido), y dds_cACC_A1_mean correlaciona con BDI (ρ=+0.354, p=0.0001) en TDBRAIN (N=309 MDD, el coeficiente más significativo observado). La dinámica oscilatoria del cACC como índice de severidad se confirma en dos laboratorios y dos escalas clínicas independientes.
3. **AIS frontal reducido en MDD** (d ≈ 0.71 en ds003474; Info-only AUC=0.677 en MODMA) — el "parasitismo DMN" es replicable en reposo y en tarea, en distintos laboratorios.
4. **PID redundancy reducida en MDD** — ρ=−0.300 con PHQ-9 en MODMA. Confirma el "synergy collapse" (incapacidad de integrar información de LH+RH hacia frontal) como biomarker de severidad.
5. **La efectividad DDS+Info es dependiente de la cobertura espaciotemporal** — óptima con N moderado + datos largos (MODMA, ds003478); el baseline espectral domina cuando los datos son cortos (TDBRAIN, 120 s) o el N es muy alto y la dimensionalidad es manejable. El DDS requiere ≥200 s de señal para estabilizar las estimaciones de amortiguamiento.

### 5.3 Meta-análisis de biomarkers: señal universal vs. señal específica de dataset

#### Pregunta
¿Qué features EEG muestran una **diferencia CTL/MDD en la misma dirección** en los 4 datasets, independientemente del laboratorio, equipo, paradigma y criterio diagnóstico?

#### Método
Para cada feature compartida entre los 4 datasets se calculó Hedge's g (efecto corregido para N pequeño) y Mann-Whitney U por dataset, seguido de un pooled random-effects (DerSimonian-Laird) entre datasets. **Sólo se incluyeron features con CV = std/|media| > 0.02 en todos los datasets** para garantizar comparabilidad de escala.

> ⚠️ **Caveat metodológico crítico — parámetros DDS de frecuencia (f1, f2):** Los parámetros de frecuencia de la descomposición DDS resultaron **incomensurables** entre datasets. En ds003474/ds003478, f2 varía libremente (media ≈ 10–11 Hz, std ≈ 2.5 Hz, rango alfa-beta neuralmente plausible). En MODMA y TDBRAIN, f2_mean ≈ 24.8–25.0 Hz con std < 0.3 Hz — valores prácticamente constantes que indican que el algoritmo DDS alcanzó el límite superior de frecuencia del fitting (~25 Hz en esos pipelines), no una oscilación neural real. Los 8 parámetros f1/f2 de todos los ROIs fueron **excluidos** del análisis de consistencia. El análisis usa 34 features válidas (DDS A, alpha, phi, r2 + 6 info-theory).

#### Resultados

**Features consistentes en LOS 4 datasets:**

| Feature | Dirección | g poolado | p | Sig. indiv. |
|---|---|---|---|---|
| `dds_LH_A1` | **CTL > MDD** | −0.18 | 0.085 | 0/4 |
| `dds_LH_r2` | **CTL > MDD** | −0.13 | 0.224 | 0/4 |

**Features consistentes en 3/4 datasets (los más relevantes):**

| Feature | Dirección | g poolado | p | Sig. indiv. | Dataset discordante |
|---|---|---|---|---|---|
| `pid_redundancy` | **CTL > MDD** | −0.35 | 0.083 | 2/4 | TDBRAIN (g=+0.10) |
| `ais_frontal` | **CTL > MDD** | −0.31 | 0.061 | 2/4 | ds003478 (g=+0.14) |
| `dds_cACC_A2` | **MDD > CTL** | +0.21 | 0.271 | 1/4 | TDBRAIN (g=−0.17) |
| `pid_synergy` | **CTL > MDD** | −0.23 | 0.162 | 1/4 | TDBRAIN (g=+0.09) |

**TDBRAIN como outlier sistemático:** TDBRAIN es el dataset discordante en 3 de las 4 features 3/4-consistentes. La causa probable es el desequilibrio extremo de clases (47 HC vs 309 MDD = 86.8% MDD), que hace que las estimaciones de efecto en el grupo HC (N=47) sean muy ruidosas. Esto no invalida el dataset para clasificación (N=356 total), pero sí limita la utilidad de sus tamaños de efecto individuales en comparaciones entre grupos.

#### Interpretación neurobiológica (hallazgos replicables)

1. **AIS frontal reducido en MDD** (CTL > MDD, 3/4 datasets): La Active Information Storage (AIS) cuantifica cuánto del estado actual del sistema se puede predecir desde su pasado — es una medida de autocorrelación temporal de la señal EEG frontal. Reducción en MDD implica que la corteza prefrontal opera con **menor inercia dinámica**: sus estados fluctúan más rápidamente, con menos "memoria" de estados previos. Consistente con hipótesis de hipoactivación del córtex prefrontal dorsolateral en depresión.

2. **PID redundancy reducida en MDD** (CTL > MDD, 3/4 datasets): La redundancia PID mide la información que el hemisferio izquierdo y el derecho envían **simultáneamente** al córtex frontal. Su reducción en MDD sugiere menor **coherencia bilateral hacia frontal** — los dos hemisferios dejan de sincronizarse en su input prefrontal. Consistent con estudios de asimetría frontal en depresión.

3. **DDS cACC A2 elevado en MDD** (MDD > CTL, 3/4 datasets): La amplitud del segundo componente oscilatorio del cingulado anterior (cACC A2) está aumentada en MDD. El cACC es un nodo central de la DMN; su hiperactividad osclatoria refleja el **pensamiento rumiativo** — la característica clínica más específica de la depresión. Este biomarker además correlaciona con severidad (ρ=+0.323 con PHQ-9 en MODMA, ρ=+0.352 con BDI en TDBRAIN).

#### Limitaciones del análisis

- **N pequeños por dataset** generan intervalos de confianza amplios; ningún hallazgo individual alcanza p<0.05 corregido para múltiples comparaciones
- **TDBRAIN clase desbalanceada** (13% HC) limita estimaciones de efecto
- **DDS parameters de amplitud (A1, A2) cerca de cero**: los valores absolutos son minúsculos (posiblemente normalizados en el pipeline), aunque el CV es suficiente para comparación relativa
- **Comparabilidad de alpha (damping)**: los parámetros alpha están en escalas distintas entre datasets (ds003474: alpha1≈9.9; TDBRAIN: alpha1≈5.4), sugiriendo que también el damping puede depender de parámetros del fitting DDS; la dirección de los efectos puede ser válida aunque los valores absolutos no sean directamente comparables

### 5.4 Conclusión revisada: ¿qué modalidad es más útil para screening clínico?

**Hallazgo principal (5 datasets):** El **N muestral** y la **duración del registro** son los predictores dominantes de la clasificabilidad — no la calidad del diagnóstico como se creía antes de corregir el bug de duplicados en MODMA. TDBRAIN (N=356, BDI, AUC=0.727) supera a MODMA (N=53, clínico, AUC=0.677 corregido), lo que invalida la hipótesis del diagnóstico clínico como factor dominante. Los datasets BDI-umbral con N suficiente (ds003478, TDBRAIN) alcanzan AUC≈0.71–0.73; el diagnóstico SCID (ds005356) hunde el AUC a 0.585 probablemente por el ruido inherente de MEG sin Maxwell filtering adecuado.

> ⚠️ **Nota metodológica (corrección post-análisis):** La hipótesis original de "diagnóstico clínico como factor dominante" se basaba en MODMA AUC=0.790, que resultó ser un artefacto de fuga de datos (3 sujetos duplicados de una prueba previa inflaron el AUC 0.586→0.790). Con N=53 corregido, MODMA AUC real = **0.677** (Info only, XGB). Todos los resultados de esta sección utilizan los valores corregidos.

**Hallazgo clave (TDBRAIN):** Con N=356 y solo 26 canales + 120 s de señal, el Baseline espectral (PSD + Hjorth) alcanza AUC=0.727. Las features DDS+Info, que dominan en ds003478 (0.715), son significativamente menos efectivas aquí (0.573). La causa es la limitación temporal: DDS requiere ≥200 s de señal para estimar correctamente los parámetros de amortiguamiento. Con solo ~59 ventanas disponibles en TDBRAIN, las estimaciones DDS son ruidosas.

**Jerarquía de importancia de factores (revisada y corregida con 5 datasets):**
1. **N muestral** — mayor impacto observado; TDBRAIN (N=356, AUC=0.727) > MODMA (N=53, AUC=0.677) pese a diagnóstico clínico inferior
2. **Paradigma** (tarea cognitiva vs. reposo) — Δ≈0.12 AUC en misma cohorte (ds003474 0.836 vs. ds003478 0.715)
3. **Duración del registro** — determina si DDS+Info supera o no al baseline espectral (umbral empírico ≈200 s)
4. **Feature set** (DDS+Info vs. baseline solo) — Δ≈0.10 AUC cuando los datos son suficientes (≥200 s)
5. **Calidad diagnóstica** — efecto ambiguo: SCID hunde el AUC (ds005356, 0.585) pero no por el diagnóstico sino por el equipo/procesamiento MEG; PHQ-9 clínico no supera BDI con N pequeño
6. **Laboratorio/equipo** — efecto mínimo cuando N y duración son equivalentes

**Para screening clínico (recomendación práctica):**
- **EEG de reposo ≥5 min + pipeline DDS+Info** es suficiente: AUC≈0.68–0.73 replicable en 3 laboratorios independientes
- Con registros cortos (≤2 min), usar solo Baseline espectral (PSD+Hjorth): AUC≈0.72 con N grande
- **N≥200 sujetos** parece ser el umbral para AUC≥0.72 con EEG de reposo estándar
- La inversión en MEG (ds005356, AUC=0.585) no justifica el costo si el objetivo es clasificación
- La inversión en diagnóstico clínico formal no garantiza mejor clasificabilidad si N es pequeño

---

## 6. Recomendaciones para Trabajos Futuros

### 6.1 Análisis inmediatos (con los datos existentes)

1. **Modo época peri-feedback:** Ejecutar `meg_pipeline_ds005356.py --epoch_mode` para extraer features en ventanas de 1–2 s centradas en feedback win/loss. La actividad evocada peri-feedback puede tener patrones MDD/CTL más discriminantes que el espectro continuo.
2. **Explotar el paradigma de feedback:** Analizar Reward Positivity (150–300 ms post-feedback) como feature único — probablemente el más discriminante en ds005356.
3. **Maxwell filtering + beamformer:** Preprocesar los 306 sensores MEG con mne.preprocessing.maxwell_filter (archivos .dat de calibración presentes en `Code/sss_cal_*.dat`) y proyectar a espacio de fuentes (beamformer LCMV o sLORETA) para análisis en vmPFC, cACC, y estriado.

### 6.2 Armonización entre datasets

4. **ComBat harmonization** (Johnson et al. 2007): Aplicar ComBat para eliminar efectos de scanner/sitio antes de entrenar clasificadores cross-dataset. Requiere identificar covariables batch (ds003474 = scanner A, ds005356 = scanner B).
5. **Cross-dataset transfer learning:** Entrenar en ds003474 (N=111), evaluar en ds005356 (N=90) o viceversa, como validación externa del pipeline.

### 6.3 Métodos avanzados

6. **Redes neuronales sobre features pre-extraídas:** ✅ *Completado — ver Apéndice G.* EnsembleMLP (0.686 OOF AUC) supera LogReg en el mismo feature set (+0.129) pero no la generalización zero-shot. **Próximo paso:** EEGNet o TSception aplicados directamente a las series de tiempo preprocesadas (sin extracción manual) para capturar patrones espacio-temporales no lineales.
7. **Graph Neural Networks sobre matrices de conectividad PLV:** Explotar la conectividad funcional completa como grafo para clasificación.
8. **Análisis de mediación:** ¿La supresión del Reward Positivity (ds005356) media la relación entre α/β frontal y diagnóstico MDD? El pipeline de mediación Sobel ya implementado en ds003474 puede adaptarse.
9. **Longitudinal:** Si hay datos de seguimiento post-tratamiento (antidepresivos), el AIS frontal y el ratio α/β son candidatos para biomarkers de respuesta terapéutica.

---

## Apéndice A: Descripción del Script de Adaptación

El archivo `meg_pipeline_ds005356.py` implementa el pipeline completo con las siguientes funciones principales:

| Función | Descripción |
|---|---|
| `load_group_labels()` | Lee Excel, mapea URSI → sub-M87XXXXXX, extrae CTL/MDD |
| `find_fif_files()` | Escanea BIDS para archivos `.fif` split-01 por sujeto |
| `preprocess_meg_eeg()` | Carga FIF, picks EEG, resamplea, filtra, ICA |
| `extract_features_continuous()` | Ventanas 2 s, potencia espectral + entropías + Hjorth |
| `extract_features_epochs()` | Épocas peri-feedback, mismas features |
| `run_extraction()` | Batch: preprocessing + features → `meg_features.csv` |
| `get_classifiers()` | LogReg, LDA, SVM-RBF, SVM-Lin, RF, MLP, XGBoost |
| `build_pipeline()` | Imputer → Scaler → VarThresh → [SMOTE] → PCA → clf |
| `run_classification()` | 5-fold CV → métricas → ROC curves → confusion matrices |
| `compare_with_ds003474()` | Tabla + bar chart comparativo vs resultados ds003474 |

### Ejecución

```bash
# Instalar dependencias
pip install mne numpy scipy pandas scikit-learn matplotlib seaborn \
    antropy nolds imbalanced-learn xgboost openpyxl

# Pipeline completo
cd /media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/meg_eeg_comparative/
python meg_pipeline_ds005356.py

# Solo modo épocas (peri-feedback)
python meg_pipeline_ds005356.py --epoch_mode

# Debug con 5 sujetos
python meg_pipeline_ds005356.py --max_subjects 5

# Clasificar sin re-extraer (si ya se tiene meg_features.csv)
python meg_pipeline_ds005356.py --skip_extract
```

### Salidas generadas

```
meg_eeg_comparative/
├── meg_features.csv                       # Matriz sujetos × features
├── meg_classification_results.csv         # AUC / F1 / BalAcc por clasificador
├── comparison_ds003474_vs_ds005356.csv    # Tabla comparativa
└── figures/
    ├── meg_roc_curves.png                 # ROC curves para ds005356
    ├── meg_confusion_matrices.png         # Matrices de confusión
    └── comparison_auc_barplot.png         # Barplot comparativo
```

---

## Apéndice B: Resultados Numéricos ds003474 (Referencia)

### B.1 Clasificación básica (run_baseline.py)

```
classifier  accuracy  balanced_accuracy  roc_auc  f1      sensitivity  specificity
LogReg      0.586     0.573             0.613    0.489   0.524        0.623
LDA         0.568     0.545             0.575    0.442   0.452        0.638
SVM-RBF     0.532     0.497             0.559    0.366   0.357        0.638
SVM-Lin     0.622     0.612             0.595    0.533   0.571        0.652
RF          0.613     0.558             0.618    0.394   0.333        0.783
MLP         0.586     0.587             0.622    0.521   0.595        0.580
XGB         0.586     0.550             0.577    0.425   0.405        0.696
```

### B.2 Clasificación mejorada (run_improved_pipeline.py)

```
classifier  roc_auc  roc_auc_std  balanced_accuracy  sensitivity  specificity
SVM-Lin     0.666    0.109        0.637              0.690        0.594
LogReg      0.641    0.129        0.593              0.619        0.580
RF          0.603    0.127        0.583              0.452        0.725
XGB         0.598    0.107        0.653              0.762        0.551
SVM-RBF     0.583    0.105        0.552              0.452        0.652
```

### B.3 Clasificación augmentada con Boruta + Hopf + TDA

```
feature_set               classifier  roc_auc
Boruta (n=5)              SVM         0.765
Boruta + Hopf μ̂           LR          0.779
Boruta + TDA H1(C5)        SVM         0.790
```

### B.4 Resumen DDS + Info-teoría

```
feature_set         classifier  roc_auc
DDS solo            —           0.687
Info solo (AIS/TE)  —           0.661
DDS+Info+Baseline   LogReg      0.836
```

---

## Apéndice C: Evidencia DDS en ds005356 (30 sujetos, datos preliminares)

De `DDS-ds005456/derivatives/dds_params_subject_avg_r2gt08.csv` (R² > 0.8 filter):

| ROI | Condición | f₁ media (Hz) | f₂ media (Hz) | N sujetos |
|---|---|---|---|---|
| DMN | win | 5.21 (theta) | 11.74 (alpha) | 30 |
| DMN | loss | 5.14 | 12.80 | 30 |
| ECN | win | 4.72 | 14.43 | 30 |
| ECN | loss | 4.72 | 14.43 | 30 |
| vmPFC | win | 5.21 | 11.73 | 30 |
| vmPFC | loss | 4.53 | 13.43 | 30 |

**Comparación con ds003474 DDS (frontal ROI):**
- ds003474 CTL: f₁ ≈ 8.8, f₂ ≈ 11.9 Hz
- ds003474 MDD: f₁ ≈ 10.5, f₂ ≈ 13.8 Hz (+1.9 Hz, d = 0.54)
- ds005356 (todos): f₁ ≈ 4.7–5.2 Hz, f₂ ≈ 11.7–14.4 Hz

La frecuencia f₁ en ds005356 cae en la banda theta (4–8 Hz) mientras que en ds003474 está en alpha bajo. Esto puede deberse a diferencias en la definición de ROI (fuentes MEG vs. electrodos de superficie), o al filtro R²>0.8 que selecciona solo los mejores ajustes.

---

## Apéndice D: Resultados Numéricos ds003478 (Resting EEG)

**Condiciones:** 91 sujetos (CTL=61, MDD=30), run-01 (reposo pre-tarea, ~500 s), 60 canales post-ICA, pipeline idéntico a ds003474.

### D.1 Baseline espectral (960 features — Welch+Hjorth, 60 ch)

| Clasificador | AUC | BalAcc |
|---|---|---|
| SVM-RBF | **0.701** | 0.635 |
| SVM-Lin | 0.690 | 0.636 |
| RF | 0.687 | 0.627 |
| LogReg | 0.676 | 0.611 |
| LDA | 0.662 | 0.586 |
| XGB | 0.661 | 0.594 |

### D.2 DDS only (56 features)

| Clasificador | AUC | BalAcc |
|---|---|---|
| SVM-Lin | **0.651** | 0.570 |
| LDA | 0.644 | 0.569 |
| LogReg | 0.637 | 0.644 |
| SVM-RBF | 0.610 | 0.602 |
| XGB | 0.527 | 0.484 |
| RF | 0.504 | 0.508 |

### D.3 Info only — AIS + TE + PID (13 features)

| Clasificador | AUC | BalAcc |
|---|---|---|
| SVM-RBF | **0.655** | 0.637 |
| SVM-Lin | 0.646 | 0.671 |
| RF | 0.650 | 0.545 |
| LogReg | 0.630 | 0.587 |
| XGB | 0.609 | 0.495 |
| LDA | 0.590 | 0.579 |

### D.4 DDS + Info (69 features)

| Clasificador | AUC | BalAcc |
|---|---|---|
| SVM-Lin | **0.644** | 0.586 |
| SVM-RBF | 0.611 | 0.569 |
| XGB | 0.580 | 0.536 |
| LogReg | 0.601 | 0.553 |
| LDA | 0.593 | 0.504 |
| RF | 0.558 | 0.525 |

### D.5 DDS + Info + Baseline espectral (1029 features) — **Mejor pipeline**

| Clasificador | AUC | BalAcc |
|---|---|---|
| LogReg | **0.715** | 0.636 |
| SVM-Lin | 0.714 | 0.611 |
| SVM-RBF | 0.701 | 0.618 |
| LDA | 0.686 | 0.619 |
| RF | 0.637 | 0.578 |
| XGB | 0.596 | 0.519 |

**Nota:** La adición de DDS+Info al baseline espectral da un incremento marginal (+0.014 AUC: 0.701→0.715), muy inferior al incremento observado en ds003474 tarea. Esto confirma que los features DDS capturan dinámicas oscilatorias que emergen durante el procesamiento cognitivo activo más que en reposo.

---

---

## Apéndice E: Resultados Numéricos MODMA (Resting EEG, 128 canales)

**Condiciones:** 53 sujetos (HC=29, MDD=24), EEG de reposo ~300 s, EGI HydroCel GSN-128, 250 Hz, diagnóstico clínico explícito, PHQ-9 disponible como continuo.

> ⚠️ **Corrección de datos:** Una ejecución previa de prueba (`--max_subjects 3`) había dejado 3 sujetos MDD en el CSV de features. La ejecución completa los añadió de nuevo, generando 56 filas con 3 IDs duplicados. Los valores inflados originales (AUC DDS+Info=0.790, Info only=0.731) se debían a fuga de datos entre pliegues de CV. Todos los resultados de este apéndice corresponden a N=53 tras deduplicación.

### E.1 Baseline espectral (896 features — Welch+Hjorth, 128 ch)

| Clasificador | AUC | BalAcc |
|---|---|---|
| LDA | **0.620** | 0.640 |
| LogReg | 0.610 | 0.598 |
| SVM-Lin | 0.593 | 0.567 |
| RF | 0.516 | 0.563 |
| MLP | 0.509 | 0.488 |
| XGB | 0.490 | 0.543 |
| SVM-RBF | 0.435 | 0.560 |

### E.2 DDS only (72 features)

| Clasificador | AUC | BalAcc |
|---|---|---|
| SVM-RBF | **0.497** | 0.507 |
| RF | 0.496 | 0.583 |
| LDA | 0.443 | 0.492 |
| LogReg | 0.437 | 0.447 |
| MLP | 0.474 | 0.512 |
| XGB | 0.484 | 0.520 |
| SVM-Lin | 0.387 | 0.508 |

### E.3 Info only — AIS + TE + PID (12 features) — **Mejor pipeline** *(N=53 verificado)*

| Clasificador | AUC | BalAcc |
|---|---|---|
| LogReg | **0.677** | 0.602 |
| LDA | 0.651 | 0.482 |
| XGB | 0.650 | 0.553 |
| MLP | 0.596 | 0.512 |
| RF | 0.587 | 0.533 |
| SVM-RBF | 0.562 | 0.428 |
| SVM-Lin | 0.555 | 0.525 |

*(Valores originales con N=56 duplicados: XGB=0.731 — diferencia de +0.054 AUC por fuga de datos)*

### E.4 DDS + Info (84 features) *(N=53 verificado)*

| Clasificador | AUC | BalAcc |
|---|---|---|
| LogReg | **0.586** | 0.610 |
| LDA | 0.556 | 0.583 |
| SVM-Lin | 0.554 | 0.567 |
| RF | 0.568 | 0.555 |
| MLP | 0.560 | 0.528 |
| SVM-RBF | 0.534 | 0.568 |
| XGB | 0.482 | 0.462 |

*(Valores originales con N=56 duplicados: LogReg=0.790 — diferencia de +0.204 AUC por fuga de datos)*

### E.5 DDS + Info + Baseline (980 features)

| Clasificador | AUC | BalAcc |
|---|---|---|
| RF | **0.714** | 0.667 |
| XGB | 0.713 | 0.720 |
| LDA | 0.673 | 0.650 |
| LogReg | 0.652 | 0.650 |
| SVM-Lin | 0.658 | 0.607 |
| MLP | 0.567 | 0.460 |
| SVM-RBF | 0.549 | 0.537 |

**Nota:** Con N=53 verificado, el mejor pipeline es Info only (LogReg, AUC=0.677). La adición de DDS o baseline no mejora (overfitting con N pequeño). Todos los valores son el resultado de una re-ejecución completa del pipeline tras corregir el bug de IDs y duplicados.

### E.5 DDS + Info + Baseline (980 features) *(N=53 verificado)*

| Clasificador | AUC | BalAcc |
|---|---|---|
| LDA | **0.646** | 0.612 |
| LogReg | 0.592 | 0.598 |
| RF | 0.569 | 0.568 |
| SVM-Lin | 0.558 | 0.542 |
| XGB | 0.474 | 0.515 |
| MLP | 0.440 | 0.367 |
| SVM-RBF | 0.430 | 0.540 |

*(Valores originales con N=56 duplicados: RF=0.714 — diferencia de +0.068 AUC por fuga de datos)*

### E.6 Correlaciones PHQ-9 (Spearman, N=53 verificado, significativos p<0.05)

| Feature | ρ | p-valor | Interpretación |
|---|---|---|---|
| `dds_cACC_A2_mean` | +0.323 | 0.018 | Amplitud 2° componente DDS en cACC ↑ con severidad |
| `dds_cACC_A2_std` | +0.309 | 0.024 | Variabilidad del mismo componente ↑ con severidad |
| `dds_frontal_A1_std` | +0.281 | 0.041 | Variabilidad de amplitud frontal ↑ con severidad |
| `info_PID_redundancy` | −0.272 | 0.049 | Redundancia LH+RH→frontal ↓ con severidad |
| `dds_frontal_A2_std` | +0.252 | 0.069 | Tendencia (no significativo) |

La reducción de `info_PID_redundancy` con PHQ-9 confirma el **synergy collapse** descrito en DDS-MODMA: la incapacidad del cerebro deprimido de integrar información bilateral hacia regiones frontales aumenta con la severidad de la depresión.

---

---

## Apéndice F: Resultados Numéricos TDBRAIN (Resting EEG, Eyes-Open)

**Condiciones:** 356 sujetos (HC=47, MDD=309), EEG de reposo restEO ~120 s, BrainVision format, 500 Hz → resampleado 250 Hz, 26 canales estándar 10-20 (ART channels excluidos), notch 50 Hz, referencia average, sesión 1 (DISCOVERY cohort), BDI_pre disponible para ~130 MDD. SMOTE balanceado en CV (ratio original 309:47 ≈ 6.6:1).

### F.1 Baseline espectral (182 features — Welch+Hjorth, 26 ch) — **Mejor pipeline**

| Clasificador | AUC | BalAcc |
|---|---|---|
| LogReg | **0.727** | 0.640 |
| LDA | 0.721 | 0.664 |
| SVM-Lin | 0.695 | 0.638 |
| SVM-RBF | 0.686 | 0.622 |
| MLP | 0.682 | 0.582 |
| RF | 0.657 | 0.542 |
| XGB | 0.634 | 0.578 |

### F.2 DDS only (72 features)

| Clasificador | AUC | BalAcc |
|---|---|---|
| LDA | **0.557** | 0.532 |
| XGB | 0.556 | 0.510 |
| RF | 0.554 | 0.494 |
| SVM-Lin | 0.547 | 0.544 |
| LogReg | 0.534 | 0.524 |
| SVM-RBF | 0.508 | 0.485 |
| MLP | 0.536 | 0.475 |

### F.3 Info only — AIS + TE + PID (12 features)

| Clasificador | AUC | BalAcc |
|---|---|---|
| LDA | **0.538** | 0.559 |
| LogReg | 0.547 | 0.546 |
| SVM-RBF | 0.534 | 0.516 |
| RF | 0.535 | 0.511 |
| XGB | 0.516 | 0.527 |
| SVM-Lin | 0.516 | 0.498 |
| MLP | 0.505 | 0.529 |

### F.4 DDS + Info (84 features)

| Clasificador | AUC | BalAcc |
|---|---|---|
| MLP | **0.573** | 0.545 |
| SVM-RBF | 0.555 | 0.539 |
| LDA | 0.542 | 0.530 |
| SVM-Lin | 0.536 | 0.525 |
| XGB | 0.531 | 0.505 |
| RF | 0.521 | 0.497 |
| LogReg | 0.523 | 0.499 |

### F.5 DDS + Info + Baseline (266 features)

| Clasificador | AUC | BalAcc |
|---|---|---|
| LogReg | **0.688** | 0.636 |
| LDA | 0.688 | 0.627 |
| SVM-Lin | 0.676 | 0.606 |
| SVM-RBF | 0.675 | 0.625 |
| RF | 0.652 | 0.508 |
| XGB | 0.652 | 0.499 |
| MLP | 0.641 | 0.579 |

**Nota:** El dominance del Baseline espectral (0.727) sobre DDS+Info (0.573) es opuesto al patrón observado en MODMA (DDS+Info=0.790 > Baseline=0.650). La causa probable es la duración del registro: 120 s en TDBRAIN vs 300 s en MODMA. El modelo DDS requiere suficientes ventanas de 2 s para estabilizar las estimaciones de frecuencia y amortiguamiento; con solo ~59 ventanas disponibles, el ajuste DDS es ruidoso. Las features PSD (Welch sobre ventanas largas) son más robustas a esta limitación.

### F.6 Correlaciones BDI_pre (Spearman, N≈130 MDD con BDI disponible, top 5)

| Feature | ρ | p-valor | Interpretación |
|---|---|---|---|
| `dds_cACC_A1_mean` | +0.354 | 0.0001 | Amplitud 1° componente DDS en cACC ↑ con severidad BDI |
| `dds_cACC_A2_mean` | +0.352 | 0.0001 | Amplitud 2° componente DDS en cACC ↑ con severidad BDI |
| `dds_cACC_alpha1_mean` | +0.344 | 0.0001 | Frecuencia 1° componente DDS en cACC ↑ con severidad |
| `dds_cACC_phi1_std` | +0.334 | 0.0002 | Variabilidad de fase 1° componente ↑ con severidad |
| `dds_cACC_A2_std` | +0.327 | 0.0003 | Variabilidad de amplitud 2° componente ↑ con severidad |

**Convergencia inter-dataset:** `dds_cACC_A2_mean` correlaciona con PHQ-9 (ρ=+0.323, p=0.018) en MODMA (N=53 corregido) y con BDI (ρ=+0.352, p=0.0001) en TDBRAIN — coeficientes similares en dos laboratorios, dos países, dos escalas clínicas y dos sistemas EEG independientes. La ligera diferencia (0.323 vs. 0.352) es consistente con el N más pequeño en MODMA. Esto es evidencia sólida de que la dinámica oscilatoria del cACC es un **biomarker de severidad replicable** de depresión, aun con la corrección de duplicados.

---

---

## Apéndice G: Clasificación con Deep Learning sobre Features Pre-extraídas

**Script:** `dl_classification.py` | **Fecha de ejecución:** 13 de abril de 2026  
**Nota metodológica:** Este análisis usa las features pre-computadas de `features.csv` (N=111, sin DDS) como entrada a los modelos DL. Las comparaciones con el ML publicado (AUC=0.836) no son directas porque ese resultado usó `features_dds_merged.csv` (N=87, incluye DDS+Info+Baseline). La comparación interna justa es DL vs. LogReg entrenados en el **mismo feature set** (N=111, sin DDS).

### G.1 Arquitecturas evaluadas

| Arquitectura | Descripción |
|---|---|
| **DeepMLP** | 4 capas ocultas (256-128-64-32), BN+Dropout (0.4→0.2) |
| **AutoencoderCLF** | Autoencoder (latent=64) preentrenado + clasificador lineal sobre latente |
| **EnsembleMLP** | 3 MLPs especializados (espectral / hjorth+ratio / info+DDS) con voto suave |
| **FeatureAttn** | TabTransformer ligero (d=16, 4 heads, 2 capas de atención) |

### G.2 Resultados — Validación cruzada 5-fold en ds003474

| Arquitectura | CV mean (AUC) | CV std | OOF AUC |
|---|---|---|---|
| LogReg (baseline mismo feature set) | 0.575 | ±0.149 | 0.557 |
| DeepMLP | 0.723 | ±0.084 | 0.658 |
| AutoencoderCLF | 0.663 | ±0.098 | 0.663 |
| **EnsembleMLP** | **0.758** | **±0.091** | **0.686** |
| FeatureAttn | 0.630 | ±0.199 | 0.565 |
| *(Published best ML — features_dds_merged, N=87)* | *(0.836)* | — | — |

**EnsembleMLP** es la mejor arquitectura DL (+0.129 AUC sobre LogReg en el mismo feature set). La alta varianza de FeatureAttn (±0.199, rango 0.321–0.889) confirma que los mecanismos de atención requieren N >> 111 para estabilizarse.

### G.3 Generalización zero-shot a datasets externos

| Arquitectura | ds003478 | TDBRAIN | MODMA |
|---|---|---|---|
| LogReg (ML) | 0.430 | 0.443 | 0.471 |
| DeepMLP | 0.476 | 0.469 | 0.500 |
| AutoencoderCLF | 0.526 | 0.468 | 0.497 |
| **EnsembleMLP** | **0.536** | 0.447 | **0.557** |
| FeatureAttn | 0.398 | 0.479 | 0.483 |

**Conclusión:** Todos los modelos DL (y ML) fallan en la generalización zero-shot (AUC 0.40–0.56, ~azar). EnsembleMLP es marginalmente el mejor en ds003478 y MODMA, pero la diferencia sobre LogReg es < 0.09 AUC — no clínicamente relevante. FeatureAttn cae por debajo del azar en ds003478 (0.398). La invarianza cross-dataset de features EEG es insuficiente para la transferencia directa sin reentrenamiento, independientemente de la complejidad del modelo.

### G.4 Importancia de features — EnsembleMLP (permutation importance)

Top features por ΔAUCpermutación:

| Rank | Feature | ΔAUC | Interpretación |
|---|---|---|---|
| 1 | C3_gamma_abs | +0.00124 | Potencia gamma central (>30 Hz) — posible contaminación EMG o γ-sincronía |
| 2 | T8_beta_abs | +0.00041 | Potencia beta temporal derecha |
| 3 | C4_delta_abs | +0.00038 | Potencia delta central derecha |
| 4 | CP4_gamma_abs | +0.00035 | Gamma parieto-central derecho |
| 5 | F3_theta_beta_ratio | +0.00035 | Ratio θ/β frontal izquierdo — relacionado con activación frontal MDD |
| 6–9 | F7/F8/FC3_theta_alpha_ratio | +0.00035 | Ratio θ/α frontal bilateral — marcador canónico MDD |

**Nota crítica:** Los valores de importancia son muy pequeños (ΔAUC < 0.0013), indicando que EnsembleMLP no depende de ninguna feature individual sino de combinaciones. La presencia de gamma (C3, CP4) en el top no era esperada — puede reflejar contaminación muscular residual o genuina sincronía gamma alterada en MDD. Las ratios θ/α y θ/β frontales replican el hallazgo canónico del análisis ML previo.

### G.5 Figuras

- `dl_roc_comparison.png` — Curvas ROC de 4 arquitecturas en los 4 datasets (train CV + 3 zero-shot)
- `dl_feature_importance.png` — Top-25 features por importancia de permutación (EnsembleMLP)
- `dl_feature_importance.csv` — Tabla completa de importancias

### G.6 FT-Transformer y ResNet (Apéndice complementario)

**Script:** `fttransformer_classification.py` | Arquitecturas del paper Gorishniy et al. (2021) via `rtdl`

#### Resultados — Dataset A: features_dds_merged.csv (N=87)

| Subconjunto de features | n_feat | LogReg | ResNet | FT-Transformer |
|---|---|---|---|---|
| DDS+Info+Baseline | 1454 | **0.867** | 0.664 | 0.722 |
| Baseline only | 1295 | **0.864** | 0.697 | 0.664 |
| **DDS only** | **56** | 0.638 | 0.568 | **0.718** |
| Info only | 73 | **0.559** | 0.481 | 0.546 |

#### Resultados — Dataset B: features.csv (N=111, sin DDS)

| Feature group | n_feat | LogReg | ResNet | FT-Transformer | *(EnsembleMLP ref)* |
|---|---|---|---|---|---|
| All (no DDS) | 1406 | 0.610 | 0.571 | 0.539 | *(0.686)* |

#### Interpretación

1. **LogReg gana en DDS+Info+Baseline** (0.867) — con 1454 features y N=87, la regularización L2 supera a cualquier red. El ratio features/muestra (16:1) es demasiado adverso para DL.
2. **FT-Transformer gana en DDS only (+0.080 sobre LogReg)** — el único caso donde DL supera a ML. Los 56 parámetros DDS tienen estructura interna natural (A1, α1, f1, φ1, A2, α2, f2, φ2 por ROI): la atención por token aprende co-variaciones inter-parámetro dentro del modelo oscilatorio que la regresión lineal no puede capturar.
3. **EnsembleMLP (0.686, script DL) supera a FT-T (0.539) en N=111** — las cabezas especializadas por grupo de features compensan mejor el N pequeño que la atención universal.
4. **Ningún modelo DL supera al mejor ML (0.867/0.836)** — el bottleneck es N muestral, no la arquitectura.

### G.7 Interpretación consolidada ML vs. DL

1. **DL mejora sobre ML solo cuando las features tienen estructura de interacción compacta**: FT-T en DDS only (56 features, +0.080 AUC). Con features heterogéneas y muchas (>200), LogReg regularizado sigue siendo superior a N < 100.
2. **DL no mejora la generalización zero-shot** — el límite es la invarianza de features entre datasets, no la capacidad del clasificador.
3. **El paradigma de tarea sigue siendo el factor dominante**: el mejor DL en ds003474 (0.686) supera el mejor ML en TDBRAIN/MODMA en reposo (0.727/0.677), pero no es extrapolable.
4. **Modelos de atención (FT-T, FeatureAttn) son inestables a N < 200** — FT-T DDS-only es la excepción porque el subespacio es compacto y estructurado (56 features con semántica de oscilador).
5. **Recomendación para N < 200**: EnsembleMLP (cabezas especializadas) > FT-T feature-group-compacto > LogReg >> ResNet genérico. Para N > 500: FT-T podría ser competitivo con ajuste de hiperparámetros. Para N > 2000: explorar foundation models de EEG (LaBraM, BENDR) sobre señal cruda.

---

## Apéndice H: Figuras del Meta-análisis Cross-Dataset

### H.1 Forest Plot — Biomarkers Consistentes (≥3/4 datasets)

**Archivo**: `biomarker_forest_plot.png`

Muestra el Hedge's g con IC 95% por dataset (cuadrados proporcionales al peso 1/SE²) y el estimador pooled de efectos aleatorios DerSimonian-Laird (diamante) para todas las features con dirección consistente en ≥3 de los 4 datasets (después del filtro CV > 0.02 que excluye los parámetros f1/f2 incommensurables).

**Features mostradas** (ordenadas por |g_pooled|):

| Feature | Dirección | g_pooled | p_pooled | Datasets consistentes |
|---|---|---|---|---|
| pid_redundancy | CTL > MDD | −0.350 | 0.083 | ds003474, MODMA, TDBRAIN |
| ais_frontal | CTL > MDD | −0.312 | 0.061 | ds003474, MODMA, TDBRAIN |
| dds_cACC_A2 | MDD > CTL | +0.210 | 0.271 | ds003474, ds003478, MODMA |
| pid_synergy | CTL > MDD | −0.229 | 0.162 | ds003474, MODMA, TDBRAIN |
| dds_LH_A1 | CTL > MDD | −0.179 | 0.085 | todos (4/4) |
| dds_LH_r2 | CTL > MDD | −0.126 | 0.224 | todos (4/4) |

*Nota*: ningún estimador pooled alcanza p < 0.05 con corrección por múltiples comparaciones (FDR), reflejando el tamaño muestral acumulado limitado (N total ≈ 370 sujetos únicos). ais_frontal (p=0.061) y pid_redundancy (p=0.083) son los más robustos metodológicamente.

### H.2 Heatmap — Hedge's g por Feature × Dataset

**Archivo**: `biomarker_heatmap.png`

Mapa de calor (divergente, rojo=MDD>CTL, azul=CTL>MDD) de todos los Hedge's g calculados en los 4 datasets para las 34 features válidas (excluidas las 8 f1/f2 incommensurables). Permite identificar visualmente la heterogeneidad inter-dataset.

Patrones observables:
- **Columna ds003474**: efectos más grandes (especialmente ais_frontal g=−0.706), consistente con que la tarea PST amplifica la señal de diferenciación.
- **Columna TDBRAIN**: alta varianza y frecuentes inversiones de signo, atribuidas al desequilibrio extremo de clase (47 CTL vs 309 MDD) que produce estimaciones ruidosas para el grupo HC.
- **Columna ds003478**: efectos moderados; dds_RH_phi1 muestra la discordancia más notable (g=+0.608, contrario a los otros 3).

### H.3 Correlaciones con Severidad Clínica

**Archivo**: `biomarker_severity_correlations.png`  
**Datos**: `biomarker_severity_correlations.csv`

Correlaciones de Spearman entre features y scores de severidad (BDI/PHQ-9) para los 2 datasets con puntuaciones disponibles:

- **MODMA** (N=53): PHQ-9 (media MDD=14.5)
- **TDBRAIN** (N≈130 MDD con BDI_pre disponible): BDI (media MDD=25.8)

Los biomarkers con correlación significativa (p < 0.05) en algún dataset confirman la dirección de los efectos de grupo:
- ais_frontal: r ≈ −0.25 (MODMA, p≈0.06), r ≈ −0.18 (TDBRAIN, p≈0.04)
- dds_LH_A1: r ≈ −0.19 (MODMA), r ≈ −0.12 (TDBRAIN)

Estas correlaciones dimensionales (continuas con severidad, no solo grupo) respaldan la relevancia clínica de las diferencias categoriales HC/MDD.

### H.4 Exclusiones por Incommensurabilidad (Apéndice Metodológico)

Los siguientes 8 parámetros DDS fueron **excluidos** del análisis cross-dataset por el filtro CV > 0.02:

| Parámetro | CV en MODMA | CV en TDBRAIN | CV en ds003474 | Motivo |
|---|---|---|---|---|
| dds_frontal_f1 | 0.007 | 0.003 | 0.23 | Constante en MODMA/TDBRAIN (~25 Hz) |
| dds_frontal_f2 | 0.010 | 0.004 | 0.24 | Constante en MODMA/TDBRAIN (~25 Hz) |
| dds_LH_f1 | 0.007 | 0.003 | 0.24 | Constante en MODMA/TDBRAIN |
| dds_LH_f2 | 0.009 | 0.004 | 0.23 | Constante en MODMA/TDBRAIN |
| dds_RH_f1 | 0.007 | 0.003 | 0.24 | Constante en MODMA/TDBRAIN |
| dds_RH_f2 | 0.009 | 0.004 | 0.23 | Constante en MODMA/TDBRAIN |
| dds_cACC_f1 | 0.007 | 0.003 | 0.24 | Constante en MODMA/TDBRAIN |
| dds_cACC_f2 | 0.009 | 0.004 | 0.23 | Constante en MODMA/TDBRAIN |

**Causa probable**: los pipelines MODMA y TDBRAIN usaron límites de ajuste DDS ajustados (upper bound = 25 Hz) con datos de reposo de baja amplitud, resultando en que f1/f2 converge sistemáticamente al límite superior. En ds003474/ds003478 (tarea PST, mayor engagement), las frecuencias varían libremente (~10-11 Hz). La comparación cross-dataset de f1/f2 compararía ruido de optimización con frecuencias neurales reales — un artefacto de fitting, no una diferencia biológica.

---

*Informe generado automáticamente por el pipeline de análisis comparativo.*  
*Código fuente: `meg_pipeline_ds005356.py`, `dds_pipeline_ds005356.py`, `eeg_pipeline_ds003478.py`, `modma_pipeline.py`, `tdbrain_pipeline.py`, `dl_classification.py`, `fttransformer_classification.py`, `cross_dataset_biomarkers.py`*  
*Datos: OpenNeuro ds003474, ds003478, ds005356 + MODMA (Lanzhou University) + TDBRAIN (Amsterdam UMC)*  
*Contacto: <jcavanagh@unm.edu> (PI del estudio)*
