#!/usr/bin/env python3
"""
Cross-Dataset MDD Generalization Study
=======================================
Train on MODMA (N=53, PHQ-9 clinical diagnosis — cleanest labels)
Zero-shot test on:
  ds003478  — EEG resting,    Cavanagh lab USA,    BDI labels,  N=91
  TDBRAIN   — EEG resting,    Amsterdam UMC NL,   BDI labels,  N=356
  ds005356  — MEG+EEG task,   Cavanagh lab USA,    SCID labels, N=84

Feature set — 43 ROI-based features (no per-electrode dependence):
  DDS × 4 ROIs × 8 params  = 32 features
  Info (AIS×4 + TE×3 + PID×4) = 11 features

Normalization: StandardScaler fitted ONLY on MODMA training data,
applied unchanged to all test sets (true zero-shot transfer).

Scientific question: do the cross-dataset biomarkers
  dds_cACC_A2, info_PID_redundancy, info_AIS_frontal
generalise across labs, countries, hardware and paradigms?
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/meg_eeg_comparative")

# ─────────────────────────────────────────────────────────────────────────────
# Feature schema
# ─────────────────────────────────────────────────────────────────────────────
ROIS       = ["cACC", "frontal", "LH", "RH"]
DDS_PARAMS = ["A1", "A2", "alpha1", "alpha2", "f1", "f2", "phi1", "phi2"]

DDS_FEATS  = [f"dds_{roi}_{p}" for roi in ROIS for p in DDS_PARAMS]   # 32
INFO_FEATS = [
    "ais_frontal", "ais_cACC", "ais_LH", "ais_RH",              # 4 AIS
    "te_LH_frontal", "te_RH_frontal", "te_cACC_frontal",        # 3 TE
    "pid_redundancy", "pid_unique_s1", "pid_unique_s2", "pid_synergy",  # 4 PID
]
ALL_FEATS = DDS_FEATS + INFO_FEATS                                      # 43

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders — normalise to canonical column names
# ─────────────────────────────────────────────────────────────────────────────

def load_modma_tdbrain(path: Path) -> pd.DataFrame:
    """
    Loader for MODMA and TDBRAIN CSVs.
    Column naming:  dds_{roi}_{param}_mean  /  info_AIS_*  /  info_PID_*
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="subject_id").reset_index(drop=True)

    rows = {"subject_id": df["subject_id"].values,
            "group":      df["group"].values.astype(int)}

    # DDS: strip the trailing _mean
    for roi in ROIS:
        for p in DDS_PARAMS:
            rows[f"dds_{roi}_{p}"] = df[f"dds_{roi}_{p}_mean"].values

    # Info
    rows["ais_frontal"]     = df["info_AIS_frontal"].values
    rows["ais_cACC"]        = df["info_AIS_cACC"].values
    rows["ais_LH"]          = df["info_AIS_LH"].values
    rows["ais_RH"]          = df["info_AIS_RH"].values
    rows["te_LH_frontal"]   = df["info_TE_LH_to_frontal"].values
    rows["te_RH_frontal"]   = df["info_TE_RH_to_frontal"].values
    rows["te_cACC_frontal"] = df["info_TE_cACC_to_frontal"].values
    rows["pid_redundancy"]  = df["info_PID_redundancy"].values
    rows["pid_unique_s1"]   = df["info_PID_unique_s1"].values
    rows["pid_unique_s2"]   = df["info_PID_unique_s2"].values
    rows["pid_synergy"]     = df["info_PID_synergy"].values

    return pd.DataFrame(rows)


def load_ds003478_ds005356(path: Path) -> pd.DataFrame:
    """
    Loader for ds003478 and ds005356 CSVs.
    Column naming:  dds_{roi}_{param}  /  ais_*  /  pid_*
    """
    df = pd.read_csv(path)

    rows = {"subject_id": df["subject_id"].values,
            "group":      df["group"].values.astype(int)}

    # DDS: columns already lack _mean suffix
    for roi in ROIS:
        for p in DDS_PARAMS:
            rows[f"dds_{roi}_{p}"] = df[f"dds_{roi}_{p}"].values

    # Info (different prefix convention)
    rows["ais_frontal"]     = df["ais_frontal"].values
    rows["ais_cACC"]        = df["ais_cACC"].values
    rows["ais_LH"]          = df["ais_LH"].values
    rows["ais_RH"]          = df["ais_RH"].values
    rows["te_LH_frontal"]   = df["te_LH_frontal"].values
    rows["te_RH_frontal"]   = df["te_RH_frontal"].values
    rows["te_cACC_frontal"] = df["te_cACC_frontal"].values
    rows["pid_redundancy"]  = df["pid_redundancy"].values
    rows["pid_unique_s1"]   = df["pid_unique_LH"].values   # rename to canonical
    rows["pid_unique_s2"]   = df["pid_unique_RH"].values   # rename to canonical
    rows["pid_synergy"]     = df["pid_synergy"].values

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame):
    """Return (X float array, y int array); impute column-median for NaN."""
    X = df[ALL_FEATS].values.astype(float)
    y = df["group"].values.astype(int)
    col_med = np.nanmedian(X, axis=0)
    r, c = np.where(np.isnan(X))
    X[r, c] = col_med[c]
    # Clip extreme outliers at ±10 IQR
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    med = np.median(X, axis=0)
    X = np.clip(X, med - 10 * iqr, med + 10 * iqr)
    return X, y


def bootstrap_auc(y_true, y_prob, n_boot=2000, ci=0.95, seed=42):
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return float(np.mean(aucs)), float(lo), float(hi)


def make_pipe(clf_name: str, n_pca: int):
    if clf_name == "LogReg":
        clf = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
    elif clf_name == "LDA":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif clf_name == "SVM-Lin":
        clf = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    else:
        raise ValueError(clf_name)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=42)),
        ("clf",    clf),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load datasets ──────────────────────────────────────────────────────
    log.info("Loading feature CSVs …")

    datasets = {
        "MODMA":    load_modma_tdbrain(
                        BASE / "modma_rest_features.csv"),
        "ds003478": load_ds003478_ds005356(
                        BASE / "ds003478_run01_dds_info_baseline_features.csv"),
        "TDBRAIN":  load_modma_tdbrain(
                        BASE / "tdbrain_restEO_features.csv"),
        "ds005356": load_ds003478_ds005356(
                        BASE / "dds_info_baseline_features.csv"),
    }

    log.info("Dataset sizes after deduplication:")
    for name, df in datasets.items():
        g = df["group"].value_counts().to_dict()
        log.info(f"  {name:<12}: N={len(df):>4}  HC={g.get(0,0):>3}  MDD={g.get(1,0):>3}")

    # ── Training set ───────────────────────────────────────────────────────
    df_train   = datasets["MODMA"]
    X_train, y_train = prepare_xy(df_train)
    N_train, N_feats = X_train.shape
    n_pca = min(18, N_train - 2, N_feats)

    log.info(f"\nTrain: MODMA  N={N_train}  features={N_feats}  PCA components={n_pca}")
    log.info(f"Features: {len(DDS_FEATS)} DDS + {len(INFO_FEATS)} Info = {len(ALL_FEATS)} total")

    classifiers = ["LogReg", "LDA", "SVM-Lin"]
    results     = []
    roc_data    = {}   # dataset → (fpr, tpr, auc) for LogReg
    pipes_fit   = {}   # trained pipelines

    for clf_name in classifiers:
        pipe = make_pipe(clf_name, n_pca)
        pipe.fit(X_train, y_train)
        pipes_fit[clf_name] = pipe

        # Cross-validated AUC on MODMA (5-fold) as internal baseline
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        prob_cv = cross_val_predict(make_pipe(clf_name, n_pca), X_train, y_train,
                                    cv=skf, method="predict_proba")[:, 1]
        auc_cv, lo_cv, hi_cv = bootstrap_auc(y_train, prob_cv)
        log.info(f"\n[{clf_name}] MODMA 5-fold CV AUC = {auc_cv:.3f} [{lo_cv:.3f}–{hi_cv:.3f}]")

        results.append({
            "Classifier":    clf_name,
            "Test dataset":  "MODMA (5-fold CV)",
            "N_test":        N_train,
            "HC":            int((y_train == 0).sum()),
            "MDD":           int((y_train == 1).sum()),
            "AUC":           round(auc_cv, 3),
            "AUC_lo":        round(lo_cv, 3),
            "AUC_hi":        round(hi_cv, 3),
            "BalAcc":        round(balanced_accuracy_score(y_train, prob_cv >= 0.5), 3),
            "Transfer":      "internal",
        })

        # ── Zero-shot test on every other dataset ─────────────────────────
        for ds_name, df_test in datasets.items():
            if ds_name == "MODMA":
                continue

            X_test, y_test = prepare_xy(df_test)
            prob_test = pipe.predict_proba(X_test)[:, 1]
            pred_test = pipe.predict(X_test)

            auc_m, auc_lo, auc_hi = bootstrap_auc(y_test, prob_test)
            bal_acc = balanced_accuracy_score(y_test, pred_test)

            log.info(f"  → {ds_name:<12}: AUC={auc_m:.3f} [{auc_lo:.3f}–{auc_hi:.3f}]"
                     f"  BalAcc={bal_acc:.3f}")

            results.append({
                "Classifier":    clf_name,
                "Test dataset":  ds_name,
                "N_test":        len(y_test),
                "HC":            int((y_test == 0).sum()),
                "MDD":           int((y_test == 1).sum()),
                "AUC":           round(auc_m, 3),
                "AUC_lo":        round(auc_lo, 3),
                "AUC_hi":        round(auc_hi, 3),
                "BalAcc":        round(bal_acc, 3),
                "Transfer":      "zero-shot",
            })

            if clf_name == "LogReg":
                fpr, tpr, _ = roc_curve(y_test, prob_test)
                roc_data[ds_name] = (fpr, tpr, auc_m)

    # ── Feature importance (LogReg back-projection through PCA) ───────────
    log.info("\n── Feature importance (LogReg, back-projected from PCA space) ──")
    pipe_lr = pipes_fit["LogReg"]
    scaler  = pipe_lr.named_steps["scaler"]
    pca     = pipe_lr.named_steps["pca"]
    coef_lr = pipe_lr.named_steps["clf"].coef_[0]

    # Coeff in PCA space → original (scaled) space → original feature space
    coef_orig = pca.components_.T @ coef_lr        # (n_features,)
    coef_phys = coef_orig / scaler.scale_           # undo z-scoring

    feat_imp = (pd.Series(np.abs(coef_phys), index=ALL_FEATS)
                .sort_values(ascending=False))
    print("\nTop 15 features (|LogReg coefficient|, original scale):")
    for feat, val in feat_imp.head(15).items():
        print(f"  {feat:<42s} {val:.5f}")

    # ── Summary table ─────────────────────────────────────────────────────
    df_res = pd.DataFrame(results)

    print("\n" + "=" * 82)
    print("CROSS-DATASET GENERALIZATION  (Train: MODMA N=53, PHQ-9 clinical labels)")
    print("Feature set: 43 ROI features (DDS×4ROIs×8params + Info AIS/TE/PID)")
    print("Normalization: StandardScaler fitted on MODMA, applied unchanged to test sets")
    print("=" * 82)

    labels = {
        "MODMA (5-fold CV)": "MODMA train 5-CV",
        "ds003478":          "ds003478  (rest/BDI/USA)",
        "TDBRAIN":           "TDBRAIN   (rest/BDI/NL) ",
        "ds005356":          "ds005356  (task/SCID/USA)",
    }

    for clf_name in classifiers:
        df_c = df_res[df_res["Classifier"] == clf_name]
        print(f"\n  ─── {clf_name} ───")
        print(f"  {'Dataset':<30} {'N':>5} {'AUC':>6}  {'95% CI':>16}  {'BalAcc':>7}  Transfer")
        print("  " + "-" * 75)
        for ds_raw, ds_label in labels.items():
            row = df_c[df_c["Test dataset"] == ds_raw]
            if row.empty:
                continue
            r = row.iloc[0]
            ci_str = f"[{r.AUC_lo:.3f}–{r.AUC_hi:.3f}]"
            tf = r.get("Transfer", "")
            print(f"  {ds_label:<30} {r.N_test:>5} {r.AUC:>6.3f}  {ci_str:<16}  {r.BalAcc:>7.3f}  {tf}")

    # ── Scientific interpretation ──────────────────────────────────────────
    lr_results = df_res[df_res["Classifier"] == "LogReg"]
    auc_478  = lr_results[lr_results["Test dataset"]=="ds003478"]["AUC"].values
    auc_td   = lr_results[lr_results["Test dataset"]=="TDBRAIN"]["AUC"].values
    auc_556  = lr_results[lr_results["Test dataset"]=="ds005356"]["AUC"].values
    auc_cv   = lr_results[lr_results["Test dataset"]=="MODMA (5-fold CV)"]["AUC"].values

    print("\n" + "─" * 82)
    print("INTERPRETATION:")
    if len(auc_cv) and len(auc_478) and len(auc_td) and len(auc_556):
        drop_478 = float(auc_cv[0] - auc_478[0])
        drop_td  = float(auc_cv[0] - auc_td[0])
        drop_556 = float(auc_cv[0] - auc_556[0])
        print(f"  Internal CV AUC (MODMA): {float(auc_cv[0]):.3f}")
        print(f"  ΔAUC to ds003478  (rest, BDI, same lab): {drop_478:+.3f}  →  {'GENERALIZA' if auc_478[0]>0.60 else 'FALLA'}")
        print(f"  ΔAUC to TDBRAIN   (rest, BDI, NL):      {drop_td:+.3f}  →  {'GENERALIZA' if auc_td[0]>0.60 else 'FALLA'}")
        print(f"  ΔAUC to ds005356  (task, SCID, MEG):    {drop_556:+.3f}  →  {'GENERALIZA' if auc_556[0]>0.60 else 'FALLA'}")
        n_gen = sum(1 for x in [auc_478[0], auc_td[0], auc_556[0]] if x > 0.60)
        print(f"\n  Biomarkers generalizan en {n_gen}/3 datasets independientes")
        if n_gen >= 2:
            print("  → Los biomarcadores DDS+Info son replicables entre laboratorios")
        if auc_478[0] > 0.60 and auc_td[0] > 0.60:
            print("  → Convergencia cross-laboratorio confirmada para EEG de reposo")
    print("─" * 82)

    # ── Save results CSV ───────────────────────────────────────────────────
    out_csv = BASE / "cross_dataset_generalization_results.csv"
    df_res.to_csv(out_csv, index=False)
    log.info(f"\nResults saved → {out_csv}")

    # ── ROC curve plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: ROC curves for zero-shot transfer
    ax = axes[0]
    palette = {
        "ds003478": ("steelblue",   "ds003478 resting EEG (BDI/USA)"),
        "TDBRAIN":  ("darkorange",  "TDBRAIN resting EEG (BDI/NL)"),
        "ds005356": ("forestgreen", "ds005356 task MEG+EEG (SCID/USA)"),
    }
    for ds_name, (fpr, tpr, auc) in roc_data.items():
        color, label = palette.get(ds_name, ("gray", ds_name))
        ax.plot(fpr, tpr, lw=2.2, color=color, label=f"{label}   AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Zero-shot generalization (LogReg)\nTrain: MODMA N=53, PHQ-9 clinical", fontsize=11)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True, alpha=0.25)

    # Right: Feature importance bar chart (top 20)
    ax2 = axes[1]
    top_feats = feat_imp.head(20)
    colors_bar = ["#d62728" if "dds_cACC" in f else
                  "#2ca02c" if "pid" in f or "ais" in f or "te_" in f else
                  "#1f77b4" for f in top_feats.index]
    ax2.barh(range(len(top_feats)), top_feats.values[::-1],
             color=colors_bar[::-1], edgecolor="white", height=0.7)
    ax2.set_yticks(range(len(top_feats)))
    ax2.set_yticklabels(top_feats.index[::-1], fontsize=8)
    ax2.set_xlabel("|LogReg coefficient| (original scale)", fontsize=10)
    ax2.set_title("Feature importance\n(back-projected from PCA)", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.25)
    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(color="#d62728", label="DDS cACC"),
                  Patch(color="#2ca02c", label="Info (AIS/TE/PID)"),
                  Patch(color="#1f77b4", label="DDS frontal/LH/RH")]
    ax2.legend(handles=legend_els, loc="lower right", fontsize=8.5)

    plt.tight_layout()
    out_fig = BASE / "cross_dataset_roc_and_importance.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    log.info(f"Figure saved → {out_fig}")

    print(f"\nOutputs:")
    print(f"  {out_csv}")
    print(f"  {out_fig}")
    print("\nDone.")


if __name__ == "__main__":
    main()
