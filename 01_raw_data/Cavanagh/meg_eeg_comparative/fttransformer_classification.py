"""
FT-Transformer (Feature Tokenizer + Transformer) classification on pre-extracted EEG features.
Gorishniy et al. 2021 — "Revisiting Deep Learning Models for Tabular Data"

Key idea: instead of concatenating all features into one vector (MLP),
each feature gets its own learned d-dimensional token. A Transformer encoder
then attends over the N_features tokens, and the CLS token is used to classify.

Datasets:
  A) features_dds_merged.csv  — N=87, DDS+Info+Baseline  (best ML LogReg AUC=0.836)
  B) features.csv (N=111, no DDS)                        (DL EnsembleMLP OOF=0.686)

Comparison table output:
  LogReg | ResNet (rtdl) | FT-Transformer  × each dataset/feature-group
"""

import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import rtdl

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Device: %s", DEVICE)

BASE     = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh")
MEG_DIR  = BASE / "meg_eeg_comparative"
DDS_CSV  = BASE / "ds003474/code/eeg_depression_classification/results/dds/features_dds_merged.csv"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── loaders ────────────────────────────────────────────────────────────────

def load_dds_merged():
    df  = pd.read_csv(DDS_CSV)
    y   = df["group"].values.astype(int)
    drop = [c for c in ["subject_id", "group", "BDI"] if c in df.columns]
    X   = df.drop(columns=drop).values.astype(float)
    feat = [c for c in df.columns if c not in drop]
    log.info("DDS merged: N=%d  features=%d  (CTL=%d MDD=%d)",
             len(y), X.shape[1], (y==0).sum(), (y==1).sum())
    return X, y, feat


def load_features_csv():
    for csv in [BASE / "ds003474/code/eeg_depression_classification/results/features.csv",
                BASE / "ds003474/code/eeg_depression_classification/results/baseline/features.csv"]:
        if csv.exists():
            break
    df  = pd.read_csv(csv)
    lc  = next(c for c in ["label","group","MDD"] if c in df.columns)
    y   = df[lc].values.astype(int)
    drop = [c for c in df.columns if c in ["subject_id","subject","label","group","MDD","BDI"]]
    X   = df.drop(columns=drop).values.astype(float)
    feat = [c for c in df.columns if c not in drop]
    log.info("features.csv: N=%d  features=%d  (CTL=%d MDD=%d)",
             len(y), X.shape[1], (y==0).sum(), (y==1).sum())
    return X, y, feat

# ── preprocessing ──────────────────────────────────────────────────────────

def preprocess_fold(X_tr, X_te, n_features_out=None):
    """Impute → optional ANOVA selection → StandardScale."""
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    if n_features_out and X_tr.shape[1] > n_features_out:
        sel  = SelectKBest(f_classif, k=n_features_out)
        X_tr = sel.fit_transform(X_tr, np.zeros(len(X_tr)))  # labels not used at test time
        X_te = sel.transform(X_te)

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return X_tr.astype(np.float32), X_te.astype(np.float32)


def preprocess_fold_supervised(X_tr, y_tr, X_te, n_features_out=None):
    """Same but uses y_tr for feature selection (correct: only train labels)."""
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    if n_features_out and X_tr.shape[1] > n_features_out:
        sel  = SelectKBest(f_classif, k=n_features_out)
        X_tr = sel.fit_transform(X_tr, y_tr)
        X_te = sel.transform(X_te)

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return X_tr.astype(np.float32), X_te.astype(np.float32)

# ── LogReg CV ───────────────────────────────────────────────────────────────

def cv_logreg(X, y, label="LogReg"):
    scores = []
    for fold, (tr, te) in enumerate(CV.split(X, y), 1):
        X_tr, X_te = preprocess_fold_supervised(X[tr], y[tr], X[te])
        clf = LogisticRegression(max_iter=2000, C=0.1, solver="lbfgs")
        clf.fit(X_tr, y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(X_te)[:, 1])
        scores.append(auc)
        log.info("  %s  fold=%d  AUC=%.3f", label, fold, auc)
    mean, std = np.mean(scores), np.std(scores)
    log.info("%s  CV mean=%.3f±%.3f", label, mean, std)
    return mean, std

# ── generic PyTorch trainer ─────────────────────────────────────────────────

def _forward(model, X_t):
    """Call model handling both (x,) and (x_num, x_cat) signatures."""
    try:
        return model(X_t)
    except TypeError:
        return model(X_t, None)


def train_eval_torch(model, X_tr, y_tr, X_te, y_te,
                     epochs=150, lr=3e-4, batch=32, patience=20):
    """Train a model and return test AUC. Uses early stopping on train loss."""
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(X_te, device=DEVICE)

    n = len(X_tr_t)
    best_loss, wait, best_state = 1e9, 0, None

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            out = _forward(model, X_tr_t[idx]).squeeze(-1)
            loss = loss_fn(out, y_tr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n
        if ep_loss < best_loss - 1e-4:
            best_loss, wait = ep_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = _forward(model, X_te_t).squeeze(-1).cpu().numpy()
    proba = 1 / (1 + np.exp(-logits))
    return roc_auc_score(y_te, proba)

# ── ResNet (rtdl) CV ─────────────────────────────────────────────────────────

def cv_resnet(X, y, label="ResNet", max_feat=200):
    scores = []
    for fold, (tr, te) in enumerate(CV.split(X, y), 1):
        X_tr, X_te = preprocess_fold_supervised(X[tr], y[tr], X[te], n_features_out=max_feat)
        d_in = X_tr.shape[1]
        model = rtdl.ResNet.make_baseline(
            d_in=d_in,
            d_main=128,
            d_hidden=256,
            dropout_first=0.3,
            dropout_second=0.1,
            n_blocks=3,
            d_out=1,
        )
        auc = train_eval_torch(model, X_tr, y[tr], X_te, y[te])
        scores.append(auc)
        log.info("  %s  fold=%d  AUC=%.3f", label, fold, auc)
    mean, std = np.mean(scores), np.std(scores)
    log.info("%s  CV mean=%.3f±%.3f", label, mean, std)
    return mean, std

# ── FT-Transformer (rtdl) CV ────────────────────────────────────────────────

def cv_fttransformer(X, y, label="FT-Transformer", max_feat=150):
    """
    FT-Transformer: each feature → learnable d_token-dim embedding,
    then Transformer encoder, CLS token for classification.
    max_feat: reduce via ANOVA to stay at manageable size (attention is O(n_feat²))
    """
    scores = []
    for fold, (tr, te) in enumerate(CV.split(X, y), 1):
        X_tr, X_te = preprocess_fold_supervised(X[tr], y[tr], X[te], n_features_out=max_feat)
        d_in = X_tr.shape[1]

        model = rtdl.FTTransformer.make_baseline(
            n_num_features=d_in,
            cat_cardinalities=None,      # no categorical features
            d_token=64,
            n_blocks=3,
            attention_dropout=0.2,
            ffn_d_hidden=128,
            ffn_dropout=0.1,
            residual_dropout=0.0,
            d_out=1,
        )
        auc = train_eval_torch(model, X_tr, y[tr], X_te, y[te],
                               epochs=200, lr=1e-4, batch=16, patience=25)
        scores.append(auc)
        log.info("  %s  fold=%d  AUC=%.3f", label, fold, auc)
    mean, std = np.mean(scores), np.std(scores)
    log.info("%s  CV mean=%.3f±%.3f", label, mean, std)
    return mean, std

# ── feature-group sweep ──────────────────────────────────────────────────────

FEAT_GROUPS = {
    "DDS+Info+Baseline": None,
    "Baseline only":     lambda cols: [c for c in cols if not c.startswith(("dds_","ais_","pid_","te_","info_","plv_"))],
    "DDS only":          lambda cols: [c for c in cols if c.startswith("dds_")],
    "Info only":         lambda cols: [c for c in cols if c.startswith(("ais_","pid_","te_","info_"))],
}

def sweep(X_all, y, feat_names, dataset_label, resnet=True, ftt=True):
    results = []
    feat_arr = np.array(feat_names)

    for gname, selector in FEAT_GROUPS.items():
        if selector is None:
            idx = np.arange(len(feat_arr))
        else:
            sel_set = set(selector(feat_arr.tolist()))
            idx = np.array([i for i, f in enumerate(feat_arr) if f in sel_set])

        if len(idx) < 5:
            log.warning("Skipping '%s' — only %d features", gname, len(idx))
            continue

        X = X_all[:, idx]
        log.info("\n── %s | %s  (n_feat=%d) ──", dataset_label, gname, len(idx))

        lr_m, lr_s  = cv_logreg(X, y, label=f"LogReg [{gname}]")
        row = {"dataset": dataset_label, "feature_group": gname,
               "n_features": len(idx), "LogReg_AUC": lr_m, "LogReg_std": lr_s}

        if resnet:
            rn_m, rn_s = cv_resnet(X, y, label=f"ResNet [{gname}]")
            row.update({"ResNet_AUC": rn_m, "ResNet_std": rn_s})

        if ftt:
            ft_m, ft_s = cv_fttransformer(X, y, label=f"FT-T [{gname}]")
            row.update({"FTTransformer_AUC": ft_m, "FTTransformer_std": ft_s})

        results.append(row)
    return pd.DataFrame(results)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("FT-Transformer vs ResNet vs LogReg — EEG depression features")
    log.info("=" * 70)

    all_results = []

    # ── A: DDS merged N=87 — ALREADY RUN, load from CSV ──────────────────
    log.info("\n[A] DDS merged (N=87) — loading previous results from CSV")
    prev = pd.read_csv(MEG_DIR / "fttransformer_results.csv")
    df_a = prev[prev["dataset"] == "ds003474 DDS+N=87"].copy()
    all_results.append(df_a)
    log.info("  Loaded %d rows from previous run.", len(df_a))

    # ── B: features.csv N=111 ─────────────────────────────────────────────
    log.info("\n[B] features.csv (N=111) — comparison with DL EnsembleMLP=0.686")
    try:
        X_f, y_f, feat_f = load_features_csv()
        lr_m, lr_s  = cv_logreg(X_f, y_f, label="LogReg [N=111]")
        rn_m, rn_s  = cv_resnet(X_f, y_f, label="ResNet [N=111]")
        ft_m, ft_s  = cv_fttransformer(X_f, y_f, label="FT-T [N=111]")
        all_results.append(pd.DataFrame([{
            "dataset": "ds003474 N=111", "feature_group": "All (no DDS)",
            "n_features": X_f.shape[1],
            "LogReg_AUC": lr_m, "LogReg_std": lr_s,
            "ResNet_AUC": rn_m, "ResNet_std": rn_s,
            "FTTransformer_AUC": ft_m, "FTTransformer_std": ft_s,
        }]))
    except Exception as e:
        log.warning("Could not load features.csv: %s", e)

    # ── Summary ───────────────────────────────────────────────────────────
    results = pd.concat(all_results, ignore_index=True)

    log.info("\n" + "=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)
    cols = ["dataset", "feature_group", "n_features",
            "LogReg_AUC", "ResNet_AUC", "FTTransformer_AUC"]
    cols = [c for c in cols if c in results.columns]
    log.info("\n%s", results[cols].to_string(index=False))

    log.info("\nReference (published results):")
    log.info("  Best ML   (LogReg, DDS+Info+Baseline, N=87):  AUC = 0.836")
    log.info("  Best DL   (EnsembleMLP, N=111, no DDS):       AUC = 0.686 (OOF)")

    out = MEG_DIR / "fttransformer_results.csv"
    results.to_csv(out, index=False)
    log.info("\nResults saved → %s", out)
    log.info("Done.")


if __name__ == "__main__":
    main()
