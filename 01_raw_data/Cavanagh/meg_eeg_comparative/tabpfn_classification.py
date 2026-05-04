"""
TabPFN classification on pre-extracted EEG features.
Compares against LogReg and EnsembleMLP baselines using the same feature sets.

Datasets:
  A) features_dds_merged.csv  — N=87, DDS+Info+Baseline (best ML: LogReg AUC=0.836)
  B) features.csv (ds003474)  — N=111, no DDS (DL baseline: EnsembleMLP OOF=0.686)
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh")
MEG_DIR = BASE / "meg_eeg_comparative"
DDS_CSV = BASE / "ds003474/code/eeg_depression_classification/results/dds/features_dds_merged.csv"
FEAT_CSV = MEG_DIR / "dds_info_baseline_features.csv"   # ds003474 N=111 no-DDS fallback

# ── feature sets used in best ML pipeline ──────────────────────────────────
DDS_FEAT_GROUPS = {
    "DDS+Info+Baseline": None,   # all columns
    "Baseline only":     lambda cols: [c for c in cols if not c.startswith(("dds_","ais_","pid_","te_","info_"))],
    "DDS only":          lambda cols: [c for c in cols if c.startswith("dds_")],
    "Info only":         lambda cols: [c for c in cols if c.startswith(("ais_","pid_","te_","info_"))],
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── loaders ────────────────────────────────────────────────────────────────

def load_dds_merged():
    df = pd.read_csv(DDS_CSV)
    y = df["group"].values.astype(int)
    drop = [c for c in ["subject_id", "group", "BDI"] if c in df.columns]
    X = df.drop(columns=drop)
    feat_names = X.columns.tolist()
    X = X.values.astype(float)
    log.info("DDS merged: N=%d  features=%d  (CTL=%d MDD=%d)", len(y), X.shape[1],
             (y==0).sum(), (y==1).sum())
    return X, y, feat_names


def load_features_csv():
    """Fallback: ds003474 features.csv — N=111, no DDS."""
    csv = MEG_DIR / "dds_info_baseline_features.csv"
    if not csv.exists():
        # try the ds003474 results dir
        csv = BASE / "ds003474/code/eeg_depression_classification/results/baseline/features.csv"
    df = pd.read_csv(csv)
    label_col = next(c for c in ["label", "group", "MDD"] if c in df.columns)
    y = df[label_col].values.astype(int)
    drop = [c for c in df.columns if c in ["subject_id","subject","label","group","MDD","BDI"]]
    X = df.drop(columns=drop).values.astype(float)
    feat_names = [c for c in df.columns if c not in drop]
    log.info("features.csv: N=%d  features=%d  (CTL=%d MDD=%d)", len(y), X.shape[1],
             (y==0).sum(), (y==1).sum())
    return X, y, feat_names


# ── preprocessing ──────────────────────────────────────────────────────────

def preprocess(X_tr, X_te):
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return X_tr, X_te


def select_features(X, y, k=200):
    """ANOVA F-test top-k selection (fit on training set in CV loop)."""
    sel = SelectKBest(f_classif, k=min(k, X.shape[1]))
    return sel


# ── CV evaluation ──────────────────────────────────────────────────────────

def cv_logreg(X, y, label="LogReg"):
    scores = []
    for fold, (tr, te) in enumerate(CV.split(X, y), 1):
        X_tr, X_te = preprocess(X[tr], X[te])
        clf = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs")
        clf.fit(X_tr, y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(X_te)[:, 1])
        scores.append(auc)
        log.info("  %s  fold=%d  AUC=%.3f", label, fold, auc)
    mean, std = np.mean(scores), np.std(scores)
    log.info("%s  CV mean=%.3f±%.3f", label, mean, std)
    return mean, std, scores


def cv_tabpfn(X, y, label="TabPFN", max_features=500):
    """
    TabPFN v2 CV.  If n_features > max_features, apply ANOVA top-k selection
    per fold to stay within a reasonable range.
    """
    from tabpfn import TabPFNClassifier

    use_selection = X.shape[1] > max_features
    k = max_features if use_selection else X.shape[1]
    log.info("  %s: n_features=%d  selection=%s (k=%d)", label, X.shape[1], use_selection, k)

    scores = []
    for fold, (tr, te) in enumerate(CV.split(X, y), 1):
        X_tr_raw, X_te_raw = X[tr], X[te]

        # impute + scale
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr_raw)
        X_te_imp = imp.transform(X_te_raw)

        # optional feature selection (fit on train only)
        if use_selection:
            sel = SelectKBest(f_classif, k=k)
            X_tr_s = sel.fit_transform(X_tr_imp, y[tr])
            X_te_s = sel.transform(X_te_imp)
        else:
            X_tr_s, X_te_s = X_tr_imp, X_te_imp

        # TabPFN does its own normalisation internally — no StandardScaler needed
        clf = TabPFNClassifier(
            n_estimators=16,
            device="cpu",
            ignore_pretraining_limits=True,
            random_state=42,
        )
        clf.fit(X_tr_s, y[tr])
        proba = clf.predict_proba(X_te_s)[:, 1]
        auc = roc_auc_score(y[te], proba)
        scores.append(auc)
        log.info("  %s  fold=%d  AUC=%.3f", label, fold, auc)

    mean, std = np.mean(scores), np.std(scores)
    log.info("%s  CV mean=%.3f±%.3f", label, mean, std)
    return mean, std, scores


# ── feature-group sweep on DDS merged ──────────────────────────────────────

def sweep_feature_groups(X_all, y, feat_names):
    """Run LogReg + TabPFN on each feature subset, replicating the ML pipeline."""
    results = []
    feat_names = np.array(feat_names)

    for group_name, selector in DDS_FEAT_GROUPS.items():
        if selector is None:
            idx = np.arange(len(feat_names))
        else:
            selected = selector(feat_names.tolist())
            idx = np.array([i for i, f in enumerate(feat_names) if f in set(selected)])

        if len(idx) == 0:
            log.warning("No features for group '%s' — skipping", group_name)
            continue

        X = X_all[:, idx]
        n_feat = len(idx)
        log.info("\n── %s  (n_feat=%d) ──", group_name, n_feat)

        lr_mean, lr_std, _ = cv_logreg(X, y, label=f"LogReg [{group_name}]")
        pfn_mean, pfn_std, _ = cv_tabpfn(X, y, label=f"TabPFN [{group_name}]", max_features=500)

        results.append({"feature_group": group_name, "n_features": n_feat,
                        "LogReg_AUC": lr_mean, "LogReg_std": lr_std,
                        "TabPFN_AUC": pfn_mean, "TabPFN_std": pfn_std})

    return pd.DataFrame(results)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("TabPFN vs LogReg — EEG depression classification")
    log.info("=" * 70)

    results_all = []

    # ── Dataset A: DDS merged (N=87, full pipeline) ──────────────────────
    log.info("\n[A] DDS merged (N=87) — replicating best ML pipeline")
    X_dds, y_dds, feat_dds = load_dds_merged()
    df_a = sweep_feature_groups(X_dds, y_dds, feat_dds)
    df_a["dataset"] = "ds003474_dds_merged (N=87)"
    results_all.append(df_a)

    # ── Dataset B: features.csv (N=111, no DDS) for DL comparison ────────
    log.info("\n[B] features.csv (N=111) — comparison with DL results")
    try:
        X_f, y_f, feat_f = load_features_csv()
        lr_mean, lr_std, _ = cv_logreg(X_f, y_f, label="LogReg [N=111]")
        pfn_mean, pfn_std, _ = cv_tabpfn(X_f, y_f, label="TabPFN [N=111]", max_features=400)
        df_b = pd.DataFrame([{
            "dataset": "ds003474_features (N=111)",
            "feature_group": "All features (no DDS)",
            "n_features": X_f.shape[1],
            "LogReg_AUC": lr_mean, "LogReg_std": lr_std,
            "TabPFN_AUC": pfn_mean, "TabPFN_std": pfn_std,
        }])
        results_all.append(df_b)
    except Exception as e:
        log.warning("Could not load features.csv: %s", e)

    # ── Summary ───────────────────────────────────────────────────────────
    results = pd.concat(results_all, ignore_index=True)
    log.info("\n" + "=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)
    log.info("\n%s", results[["dataset","feature_group","n_features",
                               "LogReg_AUC","LogReg_std",
                               "TabPFN_AUC","TabPFN_std"]].to_string(index=False))

    log.info("\nReference points:")
    log.info("  Published best ML (LogReg, DDS+Info+Baseline, N=87):  AUC = 0.836")
    log.info("  Best DL (EnsembleMLP, N=111, no DDS):                 AUC = 0.686")

    out = MEG_DIR / "tabpfn_results.csv"
    results.to_csv(out, index=False)
    log.info("\nResults saved → %s", out)


if __name__ == "__main__":
    main()
