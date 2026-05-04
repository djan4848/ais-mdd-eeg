import pandas as pd
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def investigate_ais_te():
    print("--- INVESTIGATING AIS & TE GROUND TRUTH ---")
    
    # ------------------
    # MODMA AIS/TE
    # ------------------
    print("\n--- MODMA (EEG) ---")
    m_root = Path("/media/neuraldyn/PortableSSD/DEPRESSION/MODMA")
    ais_modma_file = m_root / "DDS-MODMA/derivatives/ais_component_erp_residual_happy/ais_component_erp_residual_happy_results.csv"
    te_modma_file = m_root / "DDS-MODMA/derivatives/te_component_erp_residual_happy/te_component_erp_residual_happy_results.csv"
    clin_modma_file = m_root / "EEG_128channels_ERP_lanzhou_2015/subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx"
    
    clin_m = pd.read_excel(clin_modma_file)
    clin_m["subject"] = clin_m["subject id"].astype(str).str.zfill(8)
    
    if ais_modma_file.exists():
        ais_m = pd.read_csv(ais_modma_file, dtype={"subject": str})
        ais_m["subject"] = ais_m["subject"].str.replace("_", "").str.zfill(8)
        # Average AIS over ROIs and Trials
        ais_agg = ais_m.groupby("subject")["ais_bits"].mean().reset_index()
        df_ais = ais_agg.merge(clin_m[["subject", "PHQ-9"]], on="subject").dropna()
        
        r, p = spearmanr(df_ais["PHQ-9"], df_ais["ais_bits"])
        print(f"AIS vs PHQ-9 (Spearman): rho = {r:.3f}, p = {p:.3f} (N={len(df_ais)})")
    else:
        print("MODMA AIS file not found.")
        
    if te_modma_file.exists():
        te_m = pd.read_csv(te_modma_file, dtype={"subject": str})
        te_m["subject"] = te_m["subject"].str.replace("_", "").str.zfill(8)
        # Average TE
        te_agg = te_m.groupby("subject")["te_bits"].mean().reset_index()
        df_te = te_agg.merge(clin_m[["subject", "PHQ-9"]], on="subject").dropna()
        
        r, p = spearmanr(df_te["PHQ-9"], df_te["te_bits"])
        print(f"TE vs PHQ-9 (Spearman) : rho = {r:.3f}, p = {p:.3f} (N={len(df_te)})")
    else:
        print("MODMA TE file not found.")

    # ------------------
    # CAVANAGH AIS
    # ------------------
    print("\n--- CAVANAGH (MEG) ---")
    c_root = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds005356")
    ais_cav_file = c_root / "DDS-ds005456/derivatives/ais_component_erp_residual_r2pos/ais_component_erp_residual_r2pos_results.csv"
    clin_cav_file = c_root / "DDS-ds005456/derivatives/cavanagh_clinical.csv"
    
    if ais_cav_file.exists():
        clin_c = pd.read_csv(clin_cav_file)
        clin_c.rename(columns={"Subject": "subject"}, inplace=True)
        
        ais_c = pd.read_csv(ais_cav_file, dtype={"subject": str})
        ais_agg = ais_c.groupby("subject")["ais_bits"].mean().reset_index()
        df_ais_c = ais_agg.merge(clin_c[["subject", "SHAPS"]], on="subject").dropna()
        
        r, p = spearmanr(df_ais_c["SHAPS"], df_ais_c["ais_bits"])
        print(f"AIS vs SHAPS (Spearman): rho = {r:.3f}, p = {p:.3f} (N={len(df_ais_c)})")
    else:
        print("CAVANAGH AIS file not found.")
        
if __name__ == "__main__":
    investigate_ais_te()
