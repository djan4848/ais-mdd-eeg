import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import statsmodels.api as sm
import joblib

warnings.filterwarnings('ignore')

DIR_ASSETS = Path("06_manuscript_assets")

def run_phenotype_separation():
    print("--- THE ORBITAL TRANSITION MAP ---")
    
    file_3body = DIR_ASSETS / "Three_Body_Orbit_Decay.csv"
    file_dir = DIR_ASSETS / "directed_info_flow_audit.csv"
    
    if not file_3body.exists() or not file_dir.exists():
        print("[!] Critical Failure: Required CSV pipeline outputs missing.")
        return
        
    df_3b = pd.read_csv(file_3body)
    df_dir = pd.read_csv(file_dir)
    
    # 1. Feature Inter-dimensional Assignment (To avoid Subject Key Multiplications)
    df = df_3b.copy()
    df['Delta_TE_Ins_PFC'] = df_dir['Delta_TE_Ins_PFC']
    
    n_pre = len(df)
    
    # 2. Base Noise Filtering (TE < 1e-4 excluded for C-Factor)
    # The conditional logic is undefined if the denominator (TE) is practically zero.
    df.loc[df['TE_Insula_PFC'] < 1e-4, 'Conditioning_Factor_Perc'] = np.nan
    df = df.dropna().copy()
    print(f"[*] Filtered {n_pre - len(df)} subjects suffering from absolute null-TE conditions.")
    
    # 3. Winsorization (5th to 95th Percentile to preserve pathology while trimming numeric explosions)
    features = ['Interaction_Information_II', 'Conditioning_Factor_Perc', 'Delta_TE_Ins_PFC']
    winz_limits = {}
    for feat in features:
        lower = df[feat].quantile(0.05)
        upper = df[feat].quantile(0.95)
        winz_limits[feat] = (lower, upper)
        df[f'{feat}_Winz'] = np.clip(df[feat].values, lower, upper)
        
    joblib.dump(winz_limits, DIR_ASSETS / "ThreeBody_WinzLimits.pkl")
        
    # 4. Supervised Definition
    df_binary = df[df['Severity'].isin(['Healthy', 'Severe MDD'])].copy()
    df_moderate = df[df['Severity'] == 'Moderate MDD'].copy()
    
    print(f"[*] Extracted Binary Subsets: {len(df_binary)} (HC/Sev), {len(df_moderate)} (Mod) \n")
    
    X_bin = df_binary[[f'{f}_Winz' for f in features]]
    y_bin = df_binary['Severity'].map({'Healthy': 0, 'Severe MDD': 1})
    
    # Scale via Quartiles
    scaler = RobustScaler()
    X_bin_scaled = scaler.fit_transform(X_bin)
    
    # 5. Random Forest & Cross Validation
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(rf, X_bin_scaled, y_bin, cv=cv, scoring='accuracy')
    y_pred_cv = cross_val_predict(rf, X_bin_scaled, y_bin, cv=cv)
    
    acc = accuracy_score(y_bin, y_pred_cv)
    cm = confusion_matrix(y_bin, y_pred_cv)
    
    print("--- SUPERVISED BINARY SEPARABILITY (HC vs Severe) ---")
    print(f"Random Forest Stratified 5-Fold Accuracy: {acc*100:.2f}%")
    print(f"Cross-Validation Stability: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_bin, y_pred_cv, target_names=['Healthy', 'Severe MDD']))
    
    if acc < 0.85:
        print("[!] WARNING: Global Accuracy < 85%. The topology might not be fully singular, or variance is too high.")
    else:
        print("[+] CLASSIFICATION ROBUSTNESS VERIFIED (>85% achieved).")
        
    # Heatmap Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Severe'], yticklabels=['Healthy', 'Severe'])
    plt.ylabel('Ground Truth (Clinical)')
    plt.xlabel('Topological RF Prediction')
    plt.title('Clinical Separability Confusion Matrix\n(3-Body Dynamics Only)')
    plt.tight_layout()
    plt.savefig(DIR_ASSETS / "Figure_Confusion_Matrix.png", dpi=300)
    plt.close()
    
    # 6. Moderate Latent Position
    rf.fit(X_bin_scaled, y_bin) # Fit on full binary vector
    
    # Save parameters for Zero-Retraining pipeline on TDBRAIN
    joblib.dump(rf, DIR_ASSETS / "ThreeBody_RF_Model.pkl")
    joblib.dump(scaler, DIR_ASSETS / "ThreeBody_Scaler.pkl")
    
    if len(df_moderate) > 0:
        X_mod = df_moderate[[f'{f}_Winz' for f in features]]
        X_mod_scaled = scaler.transform(X_mod)
        pred_mod = rf.predict(X_mod_scaled)
        prob_mod = rf.predict_proba(X_mod_scaled)[:, 1] # Probability of being Severe
        
        ratio_hc = np.sum(pred_mod == 0) / len(pred_mod)
        ratio_sev = np.sum(pred_mod == 1) / len(pred_mod)
        
        print("\n--- LATENT MAPPING: MODERATE MDD ---")
        print("We tested Moderate subjects blind against the HC vs Severe Hyperplane:")
        print(f"-> {ratio_hc*100:.1f}% structurally collapsed into the Healthy Atractor.")
        print(f"-> {ratio_sev*100:.1f}% structurally collapsed into the Chaotic Severe Atractor.")
        
        # Merge Probabilities for visualization mapping
        df_moderate['Machine_Probability_Severe'] = prob_mod
        df_moderate['Latent_Class'] = np.where(pred_mod == 1, 'Mod (Severe-like)', 'Mod (Healthy-like)')
    
    # 7. Dispersiones Bi-dimensional (The Decay Trajectory)
    # Eje X: Interaction Information II (Winsorized)
    # Eje Y: C-Factor (Winsorized)
    
    # Join dfs back for consistent plotting
    df_plot = df.copy()
    if len(df_moderate) > 0:
        df_plot.loc[df_moderate.index, 'Latent_Class'] = df_moderate['Latent_Class']
    df_plot['Latent_Class'] = df_plot['Latent_Class'].fillna(df_plot['Severity'])
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Palette logic
    pal = {'Healthy': 'darkgreen', 'Severe MDD': 'darkred', 'Mod (Healthy-like)': 'lightgreen', 'Mod (Severe-like)': 'lightcoral'}
    
    # We plot the true values (Winsorized to not ruin axis limits)
    sns.scatterplot(data=df_plot, x='Interaction_Information_II_Winz', y='Conditioning_Factor_Perc_Winz', 
                    hue='Latent_Class', palette=pal, size='Delta_TE_Ins_PFC_Winz', sizes=(30, 250), alpha=0.8, edgecolor='black')
                    
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Interaction Information PID ($II$) - Synergy vs Redundancy')
    plt.ylabel('Cingulate Intercept Influence ($C_{factor}$ %)')
    plt.title('The Orbital Transition Map\nDecay Trajectory of Predictive Topology', fontweight='bold', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(DIR_ASSETS / "Figure_Orbital_Transition.png", dpi=300)
    plt.close()
    
    # 8. Statistical Sensitivity Audit against USI
    print("\n--- SENSITIVITY AUDIT: USI PREDICTORS ---")
    y_usi = df['USI']
    X_all_scaled = scaler.transform(df[[f'{f}_Winz' for f in features]])
    X_ols = sm.add_constant(X_all_scaled)
    
    model = sm.OLS(y_usi, X_ols).fit()
    print(model.summary(yname="USI", xname=['const', 'II', 'C-Factor', 'Delta-TE']))
    
    print("\n[+] SUCCESS! Orbital Map Extracted and Validation Logs Displayed.")

if __name__ == "__main__":
    run_phenotype_separation()
