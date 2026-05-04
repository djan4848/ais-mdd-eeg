import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def universal_bifurcation_overlay():
    print("==================================================")
    print(" UNIVERSAL BIFURCATION LAB: Cross-Dataset Mapping ")
    print("==================================================")

    LAB_DIR = Path(".")
    
    # 1. Look for all exported data files
    data_files = list(LAB_DIR.glob("data_*_ready.csv"))
    json_files = list(LAB_DIR.glob("critical_*.json"))
    
    if not data_files:
        print("No 'data_*_ready.csv' files found in the lab.")
        return
        
    print(f"Detected {len(data_files)} consolidated datasets:")
    for f in data_files:
        print(f" - {f.name}")

    # Load all datasets into a single dataframe
    df_list = []
    crit_dict = {}
    
    # Preload critical JSONs
    for jc in json_files:
        with open(jc, "r") as f:
            d = json.load(f)
            crit_dict[d["dataset"]] = d

    for file_path in data_files:
        dataset_name = file_path.stem.replace("data_", "").replace("_ready", "")
        df = pd.read_csv(file_path)
        
        # Intra-dataset Synergy normalization (Z-score)
        # This isolates the relative collapse geometry from absolute dataset ranges
        mean_s = df["synergy_mean"].mean()
        std_s = df["synergy_mean"].std() if df["synergy_mean"].std() > 0 else 1.0
        df["synergy_normalized"] = (df["synergy_mean"] - mean_s) / std_s
        
        df["source_dataset"] = dataset_name
        df_list.append(df)
        
    master_df = pd.concat(df_list, ignore_index=True)
    
    # 2. Universal Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set_theme(style="whitegrid", context="talk")
    
    # Colors for different datasets
    colors = {"CAVANAGH": "coral", "MODMA": "teal"}
    
    # Plot empirical dots
    sns.scatterplot(
        data=master_df, 
        x="clinical_norm", 
        y="synergy_normalized", 
        hue="source_dataset", 
        palette=colors,
        alpha=0.6,
        s=80,
        edgecolor="k",
        ax=ax
    )
    
    # Plot Universal Sigmoid (Logistic regression) over ALL points combined
    # Or just overlay their respective fits. Let's do a cross-dataset meta-fit
    from scipy.optimize import curve_fit
    
    x = master_df["clinical_norm"].values
    y = master_df["synergy_normalized"].values
    
    def boltzmann_meta(x, A, B, x0, dx):
        return A + (B - A) / (1 + np.exp((x - x0) / dx))
        
    try:
        # Initial guess for meta fit
        meta_popt, _ = curve_fit(boltzmann_meta, x, y, p0=[y.max(), y.min(), 0.5, 0.1])
        x_dense = np.linspace(0, 1, 200)
        y_meta_fit = boltzmann_meta(x_dense, *meta_popt)
        
        ax.plot(x_dense, y_meta_fit, color="darkred", lw=3, label="Universal Meta-Sigmoid")
        meta_b_c = meta_popt[2]
        ax.axvline(x=meta_b_c, color="red", linestyle=":", label=f"Universal Critical Phase ($b_c$={meta_b_c:.2f})")
    except Exception as e:
        print(f"Meta-fit failed: {e}")

    # Plot individual theoretical asymptotes if available
    for ds_name, crit_data in crit_dict.items():
        # Retrieve the critical point (raw SHAPS/PHQ9 scale!). 
        # But our x-axis is normalized [0-1]. We need to normalize the critical point.
        ds_subset = master_df[master_df["source_dataset"] == ds_name]
        try:
            raw_min = ds_subset["clinical_score"].min()
            raw_max = ds_subset["clinical_score"].max()
            norm_b_c = (crit_data["critical_point_bc"] - raw_min) / (raw_max - raw_min + 1e-9)
            
            c = colors.get(ds_name, "black")
            ax.axvline(x=norm_b_c, color=c, linestyle="--", alpha=0.5, 
                       label=f"{ds_name} Critical Point ($b_c$={norm_b_c:.2f})")
        except:
            pass
            
    ax.set_title("Universal Scale Invariance: Synergy Phase Transition", fontweight="bold")
    ax.set_xlabel("Normalized Clinical Pathophysiology Severity [0-1]")
    ax.set_ylabel("Standardized Synergy [Z-Score]")
    
    # Place legend outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_fig = LAB_DIR / "Universal_Bifurcation_Collapse.png"
    plt.savefig(out_fig, dpi=300)
    print(f"\n[OK] Universal Bifurcation plot saved to: {out_fig}")

if __name__ == "__main__":
    universal_bifurcation_overlay()
