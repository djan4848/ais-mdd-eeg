import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def universal_dynamic_dissection():
    print("======================================================")
    print(" UNIVERSAL BIFURCATION LAB: Attractor & Stability     ")
    print("======================================================")

    LAB_DIR = Path(".")
    
    # 1. Look for all exported dynamic data files
    data_files = list(LAB_DIR.glob("dynamic_*_ready.csv"))
    
    if not data_files:
        print("No 'dynamic_*_ready.csv' files found in the lab.")
        return
        
    print(f"Detected {len(data_files)} consolidated dynamic datasets.")

    df_list = []
    
    for file_path in data_files:
        dataset_name = file_path.stem.replace("dynamic_", "").replace("_ready", "")
        df = pd.read_csv(file_path)
        df["source_dataset"] = dataset_name
        df_list.append(df)
        
    master_df = pd.concat(df_list, ignore_index=True)
    
    # Ensure necessary columns exist
    req_cols = ["clinical_norm", "area_norm", "f1_norm", "source_dataset"]
    for c in req_cols:
        if c not in master_df.columns:
            print(f"Missing required column {c} in datasets. Stopping.")
            return

    sns.set_theme(style="whitegrid", context="talk")
    colors = {"CAVANAGH": "coral", "MODMA": "teal"}
    
    # ====================================================
    # PLOT 1: Universal Exploration Capacity Collapse
    # ====================================================
    print("Plotting Universal Exploration Capacity Collapse...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.scatterplot(
        data=master_df, 
        x="clinical_norm", 
        y="area_norm", 
        hue="source_dataset", 
        palette=colors,
        alpha=0.6,
        s=80,
        edgecolor="k",
        ax=ax
    )
    
    # Add smoothing 
    sns.regplot(data=master_df, x="clinical_norm", y="area_norm", scatter=False, color="darkred", order=2, ax=ax, label="Universal Exploration Collapse")
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title("Universal Dynamic Invariance:\nTransient Manifold Exploration Capacity", fontweight="bold")
    ax.set_xlabel("Normalized Clinical Pathophysiology Severity [0-1]")
    ax.set_ylabel("Standardized Exploration Capacity [Z-Score]")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(LAB_DIR / "Universal_Attractor_Area_Collapse.png", dpi=300)
    plt.close()
    
    # ====================================================
    # PLOT 2: Universal Inverse Hopf Bifurcation (f1 vs Exploration)
    # ====================================================
    print("Plotting Universal Inverse Hopf Shift...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.scatterplot(
        data=master_df, 
        x="f1_norm", 
        y="area_norm", 
        hue="source_dataset", 
        palette=colors,
        alpha=0.6,
        s=80,
        edgecolor="k",
        ax=ax
    )
    
    sns.regplot(data=master_df, x="f1_norm", y="area_norm", scatter=False, color="darkred", order=2, ax=ax, label="Meta-Shift")
    
    ax.set_title("Universal Inverse Hopf Shift:\nResonance Decoupling vs Exploration Collapse", fontweight="bold")
    ax.set_xlabel("Standardized Frequency Shift ($f_1$ Z-Score)")
    ax.set_ylabel("Standardized Exploration Capacity [Z-Score]")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(LAB_DIR / "Universal_Inverse_Hopf_Bifurcation.png", dpi=300)
    plt.close()

    print("\n[OK] Universal Dynamic Validations saved.")
    
if __name__ == "__main__":
    universal_dynamic_dissection()
