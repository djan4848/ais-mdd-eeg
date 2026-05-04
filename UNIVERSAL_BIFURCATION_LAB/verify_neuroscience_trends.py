import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
from pathlib import Path

def test_hypothesis_1_synergy(df, clinical_col, name, out_dir):
    print(f"\n--- H1: The Synergy Trend ({name}) ---")
    
    # We want mean synergy per subject vs clinical_col
    subj_df = df.groupby(['subject', clinical_col])['synergy'].mean().reset_index()
    # Drop NaNs
    subj_df = subj_df.dropna(subset=[clinical_col, 'synergy'])
    
    r_spear, p_spear = spearmanr(subj_df[clinical_col], subj_df['synergy'])
    r_pear, p_pear = pearsonr(subj_df[clinical_col], subj_df['synergy'])
    print(f"Spearman: rho={r_spear:.3f}, p={p_spear:.3f}")
    print(f"Pearson:  r={r_pear:.3f}, p={p_pear:.3f}")
    
    plt.figure(figsize=(8, 6))
    sns.regplot(data=subj_df, x=clinical_col, y='synergy', scatter_kws={'alpha':0.6})
    plt.title(f"H1: Information Integration Decline ({name})\nSpearman Rho = {r_spear:.3f} (p={p_spear:.3f})")
    plt.xlabel("Current Clinical Severity")
    plt.ylabel("PID Synergy (bits)")
    plt.tight_layout()
    plt.savefig(out_dir / f"H1_Synergy_Trend_{name}.png", dpi=300)
    plt.close()

def test_hypothesis_2_decoupling(df, clinical_col, name, out_dir):
    print(f"\n--- H2: Mechanism-Function Decoupling ({name}) ---")
    
    # Filter only valid fits to see real alignment
    # Wait, the hypothesis was: the *difference* between physical fit (R2) and Synergy is the biomarker.
    # Trial-by-trial decoupling!
    # For every trial, normalise R2 and S. Then calculate Decoupling Index = abs(Z(R2) - Z(S)).
    # We average the decoupling index per subject.
    
    # We need trial-by-trial level. df is already trial-level.
    df_valid = df.dropna(subset=['r2', 'synergy']).copy()
    
    # We can handle outliers but we just use MinMax for pure alignment 0-1
    df_valid['r2_norm'] = (df_valid['r2'] - df_valid['r2'].min()) / (df_valid['r2'].max() - df_valid['r2'].min() + 1e-9)
    df_valid['s_norm'] = (df_valid['synergy'] - df_valid['synergy'].min()) / (df_valid['synergy'].max() - df_valid['synergy'].min() + 1e-9)
    
    # Decoupling Index: High alignment = 0. Decoupled = 1.
    df_valid['decoupling_idx'] = np.abs(df_valid['r2_norm'] - df_valid['s_norm'])
    
    subj_df = df_valid.groupby(['subject', clinical_col])['decoupling_idx'].mean().reset_index()
    
    r_spear, p_spear = spearmanr(subj_df[clinical_col], subj_df['decoupling_idx'])
    print(f"Spearman (Decoupling vs Clinical): rho={r_spear:.3f}, p={p_spear:.3f}")
    
    plt.figure(figsize=(8, 6))
    sns.regplot(data=subj_df, x=clinical_col, y='decoupling_idx', scatter_kws={'alpha':0.6}, color='darkorange')
    plt.title(f"H2: Mechanism-Function Decoupling ({name})\nTrial-by-Trial $|Z(R^2) - Z(S)|$\nSpearman Rho = {r_spear:.3f} (p={p_spear:.3f})")
    plt.xlabel("Current Clinical Severity")
    plt.ylabel("Decoupling Index")
    plt.tight_layout()
    plt.savefig(out_dir / f"H2_Decoupling_{name}.png", dpi=300)
    plt.close()

def test_hypothesis_3_compensation(df, clinical_col, name, out_dir):
    print(f"\n--- H3: Compensation Trade-off (f1 vs A1) ({name}) ---")
    
    # Biological valid trials only
    df_bio = df[df['r2'] >= 0.6].copy()
    
    # Subject mean of f1 and A1
    subj_df = df_bio.groupby(['subject', clinical_col])[['f1', 'A1']].mean().reset_index()
    subj_df = subj_df.dropna(subset=[clinical_col, 'f1', 'A1'])
    
    r_f1, p_f1 = spearmanr(subj_df[clinical_col], subj_df['f1'])
    r_a1, p_a1 = spearmanr(subj_df[clinical_col], subj_df['A1'])
    print(f"f1 vs Clinic: rho={r_f1:.3f}, p={p_f1:.3f}")
    print(f"A1 vs Clinic: rho={r_a1:.3f}, p={p_a1:.3f}")
    
    # Compensation Ratio (f1 / A1)
    subj_df['compensation_ratio'] = subj_df['f1'] / (subj_df['A1'] + 1e-9)
    r_comp, p_comp = spearmanr(subj_df[clinical_col], subj_df['compensation_ratio'])
    print(f"Compensation Ratio vs Clinic: rho={r_comp:.3f}, p={p_comp:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.regplot(data=subj_df, x='A1', y='f1', ax=axes[0], color='purple', scatter_kws={'alpha':0.6})
    axes[0].set_title(f"Endogenous Trade-off ($f_1$ vs $A_1$)")
    axes[0].set_xlabel("Amplitude Capacity ($\\mu V$)")
    axes[0].set_ylabel("Resonance Frequency (Hz)")
    
    sns.regplot(data=subj_df, x=clinical_col, y='compensation_ratio', ax=axes[1], color='brown', scatter_kws={'alpha':0.6})
    axes[1].set_title(f"H3: Clinical Compensation Shift\nSpearman Rho = {r_comp:.3f} (p={p_comp:.3f})")
    axes[1].set_xlabel("Clinical Severity")
    axes[1].set_ylabel("Structural Compensation Ratio ($f_1/A_1$)")
    
    plt.tight_layout()
    plt.savefig(out_dir / f"H3_Compensation_{name}.png", dpi=300)
    plt.close()

def load_modma():
    print("Loading raw MODMA data...")
    ROOT = Path("/media/neuraldyn/PortableSSD/DEPRESSION/MODMA")
    dds_file = ROOT / "DDS-MODMA/derivatives/dds_peak_aligned_component_erp/dds_component_erp_results.csv"
    pid_file = ROOT / "DDS-MODMA/derivatives/pid_ecn_dmn_to_cacc_happy/pid_happy_results.csv"
    clin_file = ROOT / "EEG_128channels_ERP_lanzhou_2015/subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx"
    
    dds = pd.read_csv(dds_file, dtype={"subject": str})
    dds["subject"] = dds["subject"].str.replace("_", "").str.zfill(8)
    # Filter target ROI & cond
    dds = dds[(dds['cond'] == 'happy') & (dds['roi'] == 'cACC')]
    
    pid = pd.read_csv(pid_file, dtype={"subject": str})
    pid["subject"] = pid["subject"].str.replace("_", "").str.zfill(8)
    
    clin = pd.read_excel(clin_file)
    clin["subject"] = clin["subject id"].astype(str).str.zfill(8)
    
    # Merge on trial-level! pid_file has trial column for Synergy?
    # Wait, PID has 1 row per subject (averaged)?
    # Let me check pid_happy_results.csv. Ah! Williams Beer calculates one synergy value per subject across ALL trials.
    # We must merge subject-level PID to trial-level DDS.
    
    df = dds.merge(clin[['subject', 'PHQ-9']], on='subject', how='inner')
    df = df.merge(pid[['subject', 'trial', 'synergy']], on=['subject', 'trial'], how='inner')
    
    return df, 'PHQ-9'

def load_cavanagh():
    print("Loading raw CAVANAGH data...")
    ROOT = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds005356")
    dds_file = ROOT / "DDS-ds005456/derivatives/dds_cavanagh_fixed_window/dds_component_erp_results.csv"
    pid_file = ROOT / "DDS-ds005456/derivatives/pid_ecn_dmn_to_vmpfc_loss/pid_loss_results.csv"
    clin_file = ROOT / "DDS-ds005456/derivatives/cavanagh_clinical.csv"
    
    dds = pd.read_csv(dds_file, dtype={"subject": str})
    dds = dds[(dds['cond'] == 'loss') & (dds['roi'] == 'vmPFC')]
    
    pid = pd.read_csv(pid_file, dtype={"subject": str})
    
    clin = pd.read_csv(clin_file)
    clin.rename(columns={'Subject': 'subject'}, inplace=True)
    
    df = dds.merge(clin[['subject', 'SHAPS']], on='subject', how='inner')
    df = df.merge(pid[['subject', 'trial', 'synergy']], on=['subject', 'trial'], how='inner')
    
    return df, 'SHAPS'

def run():
    out_dir = Path("/media/neuraldyn/PortableSSD/DEPRESSION/UNIVERSAL_BIFURCATION_LAB/neuroscience_trends")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # MODMA
    modma_df, modma_clin = load_modma()
    test_hypothesis_1_synergy(modma_df, modma_clin, "MODMA", out_dir)
    test_hypothesis_2_decoupling(modma_df, modma_clin, "MODMA", out_dir)
    test_hypothesis_3_compensation(modma_df, modma_clin, "MODMA", out_dir)
    
    print("---------------------------------------------------------")
    
    # CAVANAGH
    cav_df, cav_clin = load_cavanagh()
    test_hypothesis_1_synergy(cav_df, cav_clin, "CAVANAGH", out_dir)
    test_hypothesis_2_decoupling(cav_df, cav_clin, "CAVANAGH", out_dir)
    test_hypothesis_3_compensation(cav_df, cav_clin, "CAVANAGH", out_dir)
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run()
