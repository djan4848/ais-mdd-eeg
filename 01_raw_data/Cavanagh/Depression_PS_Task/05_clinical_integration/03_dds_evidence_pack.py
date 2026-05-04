#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA

# =========================================================
# Config
# =========================================================
ROOT = Path(".")
DERIV = ROOT / "derivatives"

DDS_CSV = DERIV / "dds_peak_aligned_n450" / "dds_n450_results.csv"

OUTDIR = DERIV / "dds_evidence_pack"
OUTDIR.mkdir(parents=True, exist_ok=True)

PARAMS = ["A1", "gamma1", "f1", "phi1", "A2", "gamma2", "f2", "phi2"]
ROIS = ["frontal", "cacc", "lh", "rh"]
N_SPLITS = 5
N_PERM = 200  # subir a 1000 si luego quieres más robustez

# =========================================================
# Group assignment
# =========================================================
subjects_CTL = [
    'P1/HYL_01_01','P2/HYL_02_01','P3/HYL_03_01',
    'P9/HYL_09_01','P10/HYL_10_01','P11/HYL_11_01',
    'P12/HYL_12_01','P13/HYL_13_01','P14/HYL_14_01',
    'P15/HYL_15_01','P16/HYL_16_01','P17/HYL_17_01',
    'P18/HYL_18_01','P22/HYL_22_01','P26/HYL_26_01',
    'P27/HYL_27_01','P30/HYL_30_01','P31/HYL_31_01',
    'P33/HYL_33_01','P38/HYL_38_01','P49/HYL_49_01',
    'P50/HYL_50_01'
]

subjects_DEP = [
    'P6/HYL_06_01','P7/HYL_07_01','P8/HYL_08_01',
    'P20/HYL_20_01','P21/HYL_21_01','P23/HYL_23_01',
    'P24/HYL_24_01','P25/HYL_25_01','P28/HYL_28_01',
    'P29/HYL_29_01','P32/HYL_32_01','P34/HYL_34_01',
    'P35/HYL_35_01','P36/HYL_36_01','P37/HYL_37_01',
    'P39/HYL_39_01','P40/HYL_40_01','P41/HYL_41_01',
    'P42/HYL_42_01','P43/HYL_43_01','P44/HYL_44_01',
    'P45/HYL_45_01','P46/HYL_46_01','P47/HYL_47_01',
    'P48/HYL_48_01'
]

CTL_IDS = {s.split("/")[0] for s in subjects_CTL}
DEP_IDS = {s.split("/")[0] for s in subjects_DEP}

def assign_group(subject: str) -> str:
    if subject in CTL_IDS:
        return "CTL"
    elif subject in DEP_IDS:
        return "DEP"
    else:
        return "UNKNOWN"

# =========================================================
# Helpers
# =========================================================
def cohens_d_paired(x, y):
    diff = np.asarray(x) - np.asarray(y)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return np.mean(diff) / sd

def hedges_g_ind(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / sp
    correction = 1 - (3 / (4 * (nx + ny) - 9))
    return d * correction

def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def fit_summary_table(df, label):
    rows = []

    def add_row(subdf, roi_name):
        r2 = subdf["r2"].dropna().values
        if len(r2) == 0:
            return
        rows.append({
            "subset": label,
            "roi": roi_name,
            "n_rows": int(len(r2)),
            "n_subjects": int(subdf["subject"].nunique()),
            "mean_r2": float(np.mean(r2)),
            "median_r2": float(np.median(r2)),
            "q25_r2": float(np.quantile(r2, 0.25)),
            "q75_r2": float(np.quantile(r2, 0.75)),
            "min_r2": float(np.min(r2)),
            "max_r2": float(np.max(r2)),
            "pct_r2_le_0": float(100.0 * np.mean(r2 <= 0)),
        })

    add_row(df, "ALL")
    for roi in ROIS:
        add_row(df[df["roi"] == roi], roi)

    return pd.DataFrame(rows)

def run_grouped_classifier(df, feature_cols, label_col="cond_bin", group_col="subject", n_splits=5):
    df = df.dropna(subset=feature_cols + [label_col, group_col]).copy()

    X = df[feature_cols].values
    y = df[label_col].values
    groups = df[group_col].values

    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=n_splits)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ))
    ])

    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict")
    y_prob = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]

    bal_acc = balanced_accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    auc = safe_auc(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    return {
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(acc),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(df)),
        "n_subjects": int(df[group_col].nunique()),
    }

def permutation_test_grouped(df, feature_cols, n_perm=200, label_col="cond_bin", group_col="subject", n_splits=5):
    df = df.dropna(subset=feature_cols + [label_col, group_col]).copy()
    observed = run_grouped_classifier(df, feature_cols, label_col, group_col, n_splits)["balanced_accuracy"]

    perm_scores = []
    rng = np.random.default_rng(42)

    for _ in range(n_perm):
        df_perm = df.copy()
        y_perm = []

        for _, g in df.groupby(group_col, sort=False):
            vals = g[label_col].values.copy()
            rng.shuffle(vals)
            y_perm.extend(vals.tolist())

        df_perm[label_col] = y_perm
        score = run_grouped_classifier(df_perm, feature_cols, label_col, group_col, n_splits)["balanced_accuracy"]
        perm_scores.append(score)

    perm_scores = np.asarray(perm_scores, dtype=float)
    p_perm = (1 + np.sum(perm_scores >= observed)) / (1 + len(perm_scores))

    return {
        "observed_balanced_accuracy": float(observed),
        "perm_mean": float(np.mean(perm_scores)),
        "perm_std": float(np.std(perm_scores, ddof=1)),
        "p_perm": float(p_perm),
    }

def summarize_r2_condition(df):
    rows = []

    # global
    subj_cond = df.groupby(["subject", "cond"], as_index=False)["r2"].mean()
    piv = subj_cond.pivot(index="subject", columns="cond", values="r2").dropna()

    if {"INIT", "INHIB"}.issubset(piv.columns):
        t, p = ttest_rel(piv["INHIB"], piv["INIT"])
        dz = cohens_d_paired(piv["INHIB"], piv["INIT"])
        rows.append({
            "roi": "ALL",
            "mean_r2_INIT": float(piv["INIT"].mean()),
            "mean_r2_INHIB": float(piv["INHIB"].mean()),
            "median_r2_INIT": float(np.median(piv["INIT"])),
            "median_r2_INHIB": float(np.median(piv["INHIB"])),
            "t": float(t),
            "p": float(p),
            "cohens_dz": float(dz) if not np.isnan(dz) else np.nan,
            "n_subjects": int(len(piv)),
        })

    for roi in ROIS:
        tmp = df[df["roi"] == roi].groupby(["subject", "cond"], as_index=False)["r2"].mean()
        piv = tmp.pivot(index="subject", columns="cond", values="r2").dropna()
        if {"INIT", "INHIB"}.issubset(piv.columns):
            t, p = ttest_rel(piv["INHIB"], piv["INIT"])
            dz = cohens_d_paired(piv["INHIB"], piv["INIT"])
            rows.append({
                "roi": roi,
                "mean_r2_INIT": float(piv["INIT"].mean()),
                "mean_r2_INHIB": float(piv["INHIB"].mean()),
                "median_r2_INIT": float(np.median(piv["INIT"])),
                "median_r2_INHIB": float(np.median(piv["INHIB"])),
                "t": float(t),
                "p": float(p),
                "cohens_dz": float(dz) if not np.isnan(dz) else np.nan,
                "n_subjects": int(len(piv)),
            })

    return pd.DataFrame(rows)

def run_pca(df):
    X = df[PARAMS].dropna().copy()
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(Xz)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    rows = []
    for i, (e, c) in enumerate(zip(evr, cum), start=1):
        rows.append({
            "component": i,
            "explained_variance_ratio": float(e),
            "cumulative_explained_variance": float(c),
        })

    out = pd.DataFrame(rows)

    summary = {
        "n_components_80pct": int(np.argmax(cum >= 0.80) + 1),
        "n_components_90pct": int(np.argmax(cum >= 0.90) + 1),
        "n_components_95pct": int(np.argmax(cum >= 0.95) + 1),
    }
    return out, summary

def prepare_classification_table(df):
    out = df.copy()
    out["cond_bin"] = (out["cond"] == "INHIB").astype(int)
    return out

def make_subject_aggregated(df):
    agg = df.groupby(["subject", "group", "cond", "roi"], as_index=False)[PARAMS + ["r2"]].mean()
    agg["cond_bin"] = (agg["cond"] == "INHIB").astype(int)
    return agg

def classification_suite(df, level_name):
    results = []

    # full model all rois pooled
    res = run_grouped_classifier(df, PARAMS)
    perm = permutation_test_grouped(df, PARAMS, n_perm=N_PERM)
    results.append({
        "level": level_name,
        "analysis": "full_all_rois",
        **res,
        **perm
    })

    # roi-specific
    for roi in ROIS:
        sub = df[df["roi"] == roi].copy()
        if sub.empty:
            continue
        res = run_grouped_classifier(sub, PARAMS)
        perm = permutation_test_grouped(sub, PARAMS, n_perm=N_PERM)
        results.append({
            "level": level_name,
            "analysis": f"full_{roi}",
            **res,
            **perm
        })

    # ablation on pooled
    full_bacc = [r for r in results if r["analysis"] == "full_all_rois"][0]["balanced_accuracy"]
    for p in PARAMS:
        feats = [x for x in PARAMS if x != p]
        res = run_grouped_classifier(df, feats)
        perm = permutation_test_grouped(df, feats, n_perm=N_PERM)
        results.append({
            "level": level_name,
            "analysis": f"drop_{p}",
            "delta_bal_acc_vs_full": float(res["balanced_accuracy"] - full_bacc),
            **res,
            **perm
        })

    return pd.DataFrame(results)

def r2_group_cost_stats(df):
    subj_cond_roi = df.groupby(["subject", "group", "roi", "cond"], as_index=False)["r2"].mean()
    cost_rows = []

    for roi in ["ALL"] + ROIS:
        if roi != "ALL":
            tmp = subj_cond_roi[subj_cond_roi["roi"] == roi].copy()
        else:
            tmp = df.groupby(["subject", "group", "cond"], as_index=False)["r2"].mean()

        piv = tmp.pivot_table(index=["subject", "group"], columns="cond", values="r2").dropna()

        if {"INIT", "INHIB"}.issubset(piv.columns):
            piv["cost"] = piv["INHIB"] - piv["INIT"]
            reset = piv.reset_index()
            ctl = reset.query("group == 'CTL'")["cost"].values
            dep = reset.query("group == 'DEP'")["cost"].values

            if len(ctl) > 1 and len(dep) > 1:
                t, p = ttest_ind(dep, ctl, equal_var=False)
                g = hedges_g_ind(dep, ctl)
                cost_rows.append({
                    "roi": roi,
                    "mean_cost_CTL": float(np.mean(ctl)),
                    "mean_cost_DEP": float(np.mean(dep)),
                    "t": float(t),
                    "p": float(p),
                    "hedges_g_DEP_minus_CTL": float(g) if not np.isnan(g) else np.nan
                })

    return pd.DataFrame(cost_rows)

def save_subset_outputs(df, subset_name):
    # summaries
    fit_tbl = fit_summary_table(df, subset_name)
    fit_tbl.to_csv(OUTDIR / f"dds_fit_summary_{subset_name}.csv", index=False)

    r2_cond = summarize_r2_condition(df)
    r2_cond.to_csv(OUTDIR / f"dds_r2_condition_stats_{subset_name}.csv", index=False)

    group_cost = r2_group_cost_stats(df)
    group_cost.to_csv(OUTDIR / f"dds_r2_group_cost_stats_{subset_name}.csv", index=False)

    # classification
    trial_df = prepare_classification_table(df)
    subj_df = make_subject_aggregated(df)

    trial_cls = classification_suite(trial_df, f"trial_level_{subset_name}")
    subj_cls = classification_suite(subj_df, f"subject_aggregated_{subset_name}")

    trial_cls.to_csv(OUTDIR / f"dds_classification_trial_level_{subset_name}.csv", index=False)
    subj_cls.to_csv(OUTDIR / f"dds_classification_subject_aggregated_{subset_name}.csv", index=False)

    best_trial = trial_cls.sort_values(
        ["observed_balanced_accuracy", "p_perm"],
        ascending=[False, True]
    ).iloc[0].to_dict()

    best_subj = subj_cls.sort_values(
        ["observed_balanced_accuracy", "p_perm"],
        ascending=[False, True]
    ).iloc[0].to_dict()

    return {
        "fit_summary": fit_tbl,
        "r2_condition_stats": r2_cond,
        "group_cost_stats": group_cost,
        "trial_cls": trial_cls,
        "subj_cls": subj_cls,
        "best_trial": best_trial,
        "best_subj": best_subj,
    }

# =========================================================
# Main
# =========================================================
def main():
    if not DDS_CSV.exists():
        raise FileNotFoundError(f"Missing DDS file: {DDS_CSV}")

    df = pd.read_csv(DDS_CSV)

    if "group" not in df.columns:
        df["group"] = df["subject"].astype(str).apply(assign_group)

    print("\n=== group counts (rows) ===")
    print(df["group"].value_counts(dropna=False))

    print("\n=== unique subjects by group ===")
    print(df.groupby("group")["subject"].nunique())

    unknown_subjects = sorted(df.loc[df["group"] == "UNKNOWN", "subject"].unique().tolist())
    if unknown_subjects:
        print("\n[warn] UNKNOWN subjects detected:")
        for s in unknown_subjects:
            print("  ", s)

    required = ["subject", "group", "cond", "trial", "roi", "r2"] + PARAMS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in DDS file: {missing}")

    # define subsets
    df_all = df.copy()
    df_r2pos = df[df["r2"] > 0].copy()

    # save global subset counts
    subset_counts = pd.DataFrame([
        {
            "subset": "all",
            "n_rows": int(len(df_all)),
            "n_subjects": int(df_all["subject"].nunique()),
            "pct_rows": 100.0,
        },
        {
            "subset": "r2pos",
            "n_rows": int(len(df_r2pos)),
            "n_subjects": int(df_r2pos["subject"].nunique()),
            "pct_rows": 100.0 * len(df_r2pos) / len(df_all),
        }
    ])
    subset_counts.to_csv(OUTDIR / "dds_subset_counts.csv", index=False)

    # outputs for all
    out_all = save_subset_outputs(df_all, "all")

    # outputs for r2 positive only
    out_r2pos = save_subset_outputs(df_r2pos, "r2pos")

    # PCA only on r2pos
    pca_table, pca_summary = run_pca(df_r2pos)
    pca_table.to_csv(OUTDIR / "dds_pca_explained_variance_r2pos.csv", index=False)

    corr = df_r2pos[PARAMS].corr()
    corr.to_csv(OUTDIR / "dds_param_correlation_matrix_r2pos.csv")

    # summary json
    summary = {
        "input_file": str(DDS_CSV),
        "n_rows_all": int(len(df_all)),
        "n_rows_r2pos": int(len(df_r2pos)),
        "n_subjects": int(df_all["subject"].nunique()),
        "pct_r2pos": 100.0 * len(df_r2pos) / len(df_all),
        "rois": sorted(df_all["roi"].dropna().unique().tolist()),
        "params": PARAMS,
        "pca_summary_r2pos": pca_summary,
        "best_trial_level_all": out_all["best_trial"],
        "best_subject_aggregated_all": out_all["best_subj"],
        "best_trial_level_r2pos": out_r2pos["best_trial"],
        "best_subject_aggregated_r2pos": out_r2pos["best_subj"],
        "n_perm": N_PERM
    }

    with open(OUTDIR / "dds_evidence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # terminal report
    print("\n================ DDS EVIDENCE PACK ================\n")

    print("0) SUBSET COUNTS")
    print(subset_counts.to_string(index=False))

    print("\n1) FIT QUALITY SUMMARY — ALL")
    print(out_all["fit_summary"].to_string(index=False))

    print("\n2) FIT QUALITY SUMMARY — R2 > 0")
    print(out_r2pos["fit_summary"].to_string(index=False))

    print("\n3) R² CONDITION STATS — ALL")
    print(out_all["r2_condition_stats"].to_string(index=False))

    print("\n4) R² CONDITION STATS — R2 > 0")
    print(out_r2pos["r2_condition_stats"].to_string(index=False))

    print("\n5) PCA SUMMARY — R2 > 0")
    print(json.dumps(pca_summary, indent=2))

    print("\n6) BEST TRIAL-LEVEL CLASSIFICATION — ALL")
    print(json.dumps(out_all["best_trial"], indent=2))

    print("\n7) BEST SUBJECT-AGGREGATED CLASSIFICATION — ALL")
    print(json.dumps(out_all["best_subj"], indent=2))

    print("\n8) BEST TRIAL-LEVEL CLASSIFICATION — R2 > 0")
    print(json.dumps(out_r2pos["best_trial"], indent=2))

    print("\n9) BEST SUBJECT-AGGREGATED CLASSIFICATION — R2 > 0")
    print(json.dumps(out_r2pos["best_subj"], indent=2))

    print(f"\nSaved outputs in: {OUTDIR}\n")

if __name__ == "__main__":
    main()
