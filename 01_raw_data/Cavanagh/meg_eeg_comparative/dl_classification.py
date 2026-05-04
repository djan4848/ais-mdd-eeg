"""
dl_classification.py
====================
Deep-learning classification of MDD vs CTL using pre-computed EEG features.
No re-extraction: reads existing CSV outputs from the ML pipeline.

Architecture 1 — Deep MLP with BatchNorm + Dropout
Architecture 2 — Autoencoder pre-train → latent classifier
Architecture 3 — Ensemble of group-specialized MLPs
Architecture 4 — Feature-attention network (lightweight TabTransformer)

Validation:
  • Internal: 5-fold stratified CV on ds003474 (N=111)
  • Zero-shot: ds003478 (resting, same subjects), TDBRAIN (N=356), MODMA (N=53)
"""

import sys, os, logging, warnings, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression          # ML baseline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE474 = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/ds003474"
               "/code/eeg_depression_classification/results")
MEG_DIR = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh/meg_eeg_comparative")
OUT_DIR = MEG_DIR

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SEED        = 42
N_FOLDS     = 5
EPOCHS      = 200
PATIENCE    = 25          # early stopping
LR          = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE  = 32

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# 18 standard 10-20 channels common across ds003474 / ds003478 / TDBRAIN / MODMA
COMMON_CH = ["C3","C4","CP3","CP4","F3","F4","F7","F8","FC3","FC4",
             "O1","O2","P3","P4","P7","P8","T7","T8"]

# ds003478 channel order (ch000 = FP1, ch007 = F3, …)
CH478 = ["FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
         "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1",
         "CZ","C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6",
         "TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ",
         "PO4","PO6","PO8","O1","OZ","O2"]
CH478_idx = {ch: i for i, ch in enumerate(CH478)}

# MODMA EGI → standard 10-20 mapping (subset of COMMON_CH)
MODMA_MAP = {
    "E20":"F3","E118":"F4","E36":"F7","E124":"F8",
    "E47":"FC3","E106":"FC4","E58":"T7","E96":"T8",
    "E52":"C3","E92":"C4","E62":"CP3","E85":"CP4",
    "E70":"P3","E83":"P4","E75":"O1","E82":"O2",
}
MODMA_INV = {v: k for k, v in MODMA_MAP.items() if v in COMMON_CH}

# ROI → channels (for aggregating AIS from per-channel to ROI)
ROIS = {
    "frontal": ["F3","F4","F7","F8","FC3","FC4"],
    "cacc":    ["C3","C4"],
    "lh":      ["F3","C3","P3","T7"],
    "rh":      ["F4","C4","P4","T8"],
}

META_COLS = {"subject","subject_id","group","label","BDI","BDI_pre","PHQ9",
             "age","gender","n_bad_ch","n_ica_removed","sfreq","dur","run",
             "id_len"}

torch.manual_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — canonical feature normalisation
# ══════════════════════════════════════════════════════════════════════════════
def _add(out: dict, key: str, series):
    if series is not None and not series.isna().all():
        out[key] = series


def load_ds003474() -> pd.DataFrame:
    df = pd.read_csv(BASE474 / "features.csv")
    out = {}
    for ch in COMMON_CH:
        for band in BANDS:
            for t in ["abs", "rel"]:
                c = f"{band}_{t}_{ch}"
                _add(out, f"{ch}_{band}_{t}", df.get(c))
            for ratio in ["alpha_beta_ratio", "theta_beta_ratio", "theta_alpha_ratio"]:
                _add(out, f"{ch}_{ratio}", df.get(f"{ratio}_{ch}"))
        for h_src, h_dst in [("hjorth_activity","hjorth_act"),
                              ("hjorth_mobility","hjorth_mob"),
                              ("hjorth_complexity","hjorth_com")]:
            _add(out, f"{ch}_{h_dst}", df.get(f"{h_src}_{ch}"))
        for e_src, e_dst in [("spectral_entropy","spec_ent"),("perm_entropy","perm_ent"),
                              ("sample_entropy","samp_ent"),("dfa","dfa")]:
            _add(out, f"{ch}_{e_dst}", df.get(f"{e_src}_{ch}"))
        _add(out, f"{ch}_ais", df.get(f"ais_{ch}"))
    # FAA (F4-F3 pair only)
    for band in BANDS:
        _add(out, f"FAA_{band}_F4_F3", df.get(f"FAA_{band}_F4_F3"))
    # AIS aggregated to ROIs
    for roi, chs in ROIS.items():
        vals = [df[f"ais_{ch}"] for ch in chs if f"ais_{ch}" in df.columns]
        if vals:
            out[f"ais_{roi}"] = pd.concat(vals, axis=1).mean(axis=1)

    result = pd.DataFrame(out, index=df.index)
    result["group"] = df["group"].values
    log.info("ds003474: N=%d  features=%d  (CTL=%d MDD=%d)",
             len(result), result.shape[1]-1,
             (result.group==0).sum(), (result.group==1).sum())
    return result


def load_ds003478() -> pd.DataFrame:
    df = pd.read_csv(MEG_DIR / "ds003478_run01_dds_info_baseline_features.csv")
    out = {}
    for ch in COMMON_CH:
        idx = CH478_idx[ch]; sfx = f"ch{idx:03d}"
        for band in BANDS:
            for t in ["abs", "rel"]:
                _add(out, f"{ch}_{band}_{t}", df.get(f"{band}_{t}_{sfx}"))
            for ratio in ["alpha_beta_ratio", "theta_beta_ratio", "theta_alpha_ratio"]:
                _add(out, f"{ch}_{ratio}", df.get(f"{ratio}_{sfx}"))
        for h in ["hjorth_act", "hjorth_mob", "hjorth_com"]:
            _add(out, f"{ch}_{h}", df.get(f"{h}_{sfx}"))
    # DDS ROI (no _mean suffix in 478)
    for c in df.columns:
        if c.startswith("dds_"):
            out[c] = df[c]
    # Info-theory ROI
    for c in df.columns:
        if c.startswith(("ais_", "pid_", "te_")):
            out[c] = df[c]

    result = pd.DataFrame(out, index=df.index)
    result["group"] = df["group"].values
    log.info("ds003478: N=%d  features=%d  (CTL=%d MDD=%d)",
             len(result), result.shape[1]-1,
             (result.group==0).sum(), (result.group==1).sum())
    return result


def load_tdbrain() -> pd.DataFrame:
    df = pd.read_csv(MEG_DIR / "tdbrain_restEO_features.csv")
    out = {}
    for ch in COMMON_CH:
        # TDBRAIN uses standard case (F3, Cz, etc.)
        ch_tdb = next((c for c in [ch, ch.capitalize()]
                       if any(col.startswith(c+"_") for col in df.columns)), None)
        if not ch_tdb:
            continue
        for band in BANDS:
            _add(out, f"{ch}_{band}_abs", df.get(f"{ch_tdb}_{band}"))
        for h in ["hjorth_mob", "hjorth_com"]:
            _add(out, f"{ch}_{h}", df.get(f"{ch_tdb}_{h}"))
    # DDS ROI (drop _mean suffix)
    for c in df.columns:
        if c.startswith("dds_") and c.endswith("_mean"):
            out[c[:-5]] = df[c]
    # Info-theory
    for c in df.columns:
        if c.startswith("info_"):
            new = (c.replace("info_AIS_", "ais_")
                    .replace("info_PID_", "pid_")
                    .replace("info_TE_",  "te_")
                    .lower())
            out[new] = df[c]

    result = pd.DataFrame(out, index=df.index)
    result["group"] = df["group"].values
    log.info("TDBRAIN: N=%d  features=%d  (CTL=%d MDD=%d)",
             len(result), result.shape[1]-1,
             (result.group==0).sum(), (result.group==1).sum())
    return result


def load_modma() -> pd.DataFrame:
    df = pd.read_csv(MEG_DIR / "modma_rest_features.csv", dtype={"subject_id": str})
    out = {}
    for ch in COMMON_CH:
        e_ch = MODMA_INV.get(ch)
        if not e_ch:
            continue
        for band in BANDS:
            _add(out, f"{ch}_{band}_abs", df.get(f"{e_ch}_{band}"))
        for h in ["hjorth_mob", "hjorth_com"]:
            _add(out, f"{ch}_{h}", df.get(f"{e_ch}_{h}"))
    # DDS ROI
    for c in df.columns:
        if c.startswith("dds_") and c.endswith("_mean"):
            out[c[:-5]] = df[c]
    # Info-theory
    for c in df.columns:
        if c.startswith("info_"):
            new = (c.replace("info_AIS_", "ais_")
                    .replace("info_PID_", "pid_")
                    .replace("info_TE_",  "te_")
                    .lower())
            out[new] = df[c]

    result = pd.DataFrame(out, index=df.index)
    result["group"] = df["group"].values
    log.info("MODMA: N=%d  features=%d  (CTL=%d MDD=%d)",
             len(result), result.shape[1]-1,
             (result.group==0).sum(), (result.group==1).sum())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_XY(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c != "group"]
    X = df[feat_cols].values.astype(np.float32)
    y = df["group"].values.astype(np.float32)
    return X, y, feat_cols


def align_features(X_train, feat_train: list, X_test, feat_test: list):
    """Return X_test aligned to feat_train columns.
    Missing columns → imputed with column mean from X_train.
    Extra test columns → dropped.
    """
    common = [f for f in feat_train if f in feat_test]
    log.info("  Feature alignment: train=%d  test=%d  common=%d",
             len(feat_train), len(feat_test), len(common))
    # Build aligned test matrix
    feat_test_idx  = {f: i for i, f in enumerate(feat_test)}
    feat_train_idx = {f: i for i, f in enumerate(feat_train)}
    X_out = np.full((len(X_test), len(feat_train)), np.nan, dtype=np.float32)
    for f in common:
        X_out[:, feat_train_idx[f]] = X_test[:, feat_test_idx[f]]
    # Impute missing columns with training column mean (already scaled)
    for i, f in enumerate(feat_train):
        if f not in feat_test:
            col_mean = np.nanmean(X_train[:, i])
            X_out[:, i] = col_mean
    return X_out


def preprocess(X_tr, X_te=None):
    """Impute NaN → StandardScaler. Returns numpy arrays."""
    imp = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_tr = imp.fit_transform(X_tr)
    X_tr = scaler.fit_transform(X_tr)
    if X_te is not None:
        X_te = imp.transform(X_te)
        X_te = scaler.transform(X_te)
        return X_tr, X_te, imp, scaler
    return X_tr, imp, scaler


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════════════════════
def to_tensors(X, y=None):
    Xt = torch.tensor(X, dtype=torch.float32)
    if y is None:
        return Xt
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return Xt, yt


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE 1: Deep MLP
# ══════════════════════════════════════════════════════════════════════════════
class DeepMLP(nn.Module):
    def __init__(self, n_in: int, hidden=(256, 128, 64, 32), drop=(0.4, 0.4, 0.3, 0.2)):
        super().__init__()
        layers = []
        prev = n_in
        for h, d in zip(hidden, drop):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(d)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE 2: Autoencoder + Latent Classifier
# ══════════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self, n_in: int, latent: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, latent), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 128), nn.ReLU(),
            nn.Linear(128, 256),    nn.ReLU(),
            nn.Linear(256, n_out),
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, n_in: int, latent: int = 64):
        super().__init__()
        self.encoder = Encoder(n_in, latent)
        self.decoder = Decoder(latent, n_in)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class LatentClassifier(nn.Module):
    def __init__(self, latent: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16),     nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, z):
        return self.net(z)


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE 3: Ensemble of Group-Specialized MLPs
# ══════════════════════════════════════════════════════════════════════════════
def make_feature_groups(feat_names: list) -> dict:
    """Split features into 3 groups for ensemble."""
    spectral, hjorth_ratio, info_dds = [], [], []
    for i, f in enumerate(feat_names):
        if any(f.endswith(f"_{b}_{t}") for b in BANDS for t in ["abs","rel"]):
            spectral.append(i)
        elif any(tok in f for tok in ["hjorth","ratio","spec_ent","perm_ent","samp_ent","dfa","FAA"]):
            hjorth_ratio.append(i)
        else:   # ais, pid, te, dds
            info_dds.append(i)
    return {"spectral": spectral, "hjorth_ratio": hjorth_ratio, "info_dds": info_dds}


class GroupMLP(nn.Module):
    def __init__(self, n_in: int, latent: int = 32, drop: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, latent), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class EnsembleMLP(nn.Module):
    def __init__(self, groups: dict, latent: int = 32):
        super().__init__()
        self.groups = groups
        self.group_nets = nn.ModuleDict({
            name: GroupMLP(len(idx), latent) if idx else None
            for name, idx in groups.items()
        })
        n_active = sum(1 for idx in groups.values() if idx)
        self.fusion = nn.Sequential(
            nn.Linear(latent * n_active, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
        self._register_groups()

    def _register_groups(self):
        """Store group indices as buffers so they move with the module."""
        for name, idx in self.groups.items():
            if idx:
                self.register_buffer(f"idx_{name}", torch.tensor(idx, dtype=torch.long))

    def forward(self, x):
        parts = []
        for name, net in self.group_nets.items():
            if net is None:
                continue
            idx = getattr(self, f"idx_{name}")
            parts.append(net(x[:, idx]))
        return self.fusion(torch.cat(parts, dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE 4: Feature-Attention Network (Lightweight TabTransformer)
# ══════════════════════════════════════════════════════════════════════════════
class FeatureAttentionNet(nn.Module):
    """
    Each feature is embedded to dim `d_model`.
    Multi-head self-attention over the feature dimension.
    Mean-pooled → MLP classifier.
    """
    def __init__(self, n_feat: int, d_model: int = 16, n_heads: int = 4,
                 n_layers: int = 2, drop: float = 0.3):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=64,
            dropout=drop, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, F) → (B, F, 1) → embed → (B, F, d_model)
        x = self.embedding(x.unsqueeze(-1))
        x = self.transformer(x)     # (B, F, d_model)
        x = x.mean(dim=1)           # (B, d_model)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def compute_class_weight(y: np.ndarray) -> torch.Tensor:
    n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    return pos_weight


def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY,
                batch_size=BATCH_SIZE, patience=PATIENCE,
                ae_pretrain=False, ae_epochs=100) -> float:
    """Train any binary classification model. Returns best val AUC."""
    device = torch.device("cpu")
    model.to(device)

    pos_w   = compute_class_weight(y_tr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Xtr_t, ytr_t = to_tensors(X_tr, y_tr)
    Xvl_t        = to_tensors(X_val)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    best_auc, best_state, wait = 0.0, None, 0

    for ep in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xvl_t.to(device)).cpu().numpy().ravel()
        probs  = 1.0 / (1.0 + np.exp(-logits))
        try:
            auc = roc_auc_score(y_val, probs)
        except Exception:
            auc = 0.5

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_auc


def train_autoencoder(model: Autoencoder, X_all: np.ndarray,
                      ae_epochs: int = 100, lr: float = 1e-3) -> None:
    """Pretrain autoencoder unsupervised on X_all."""
    device   = torch.device("cpu")
    model.to(device)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt       = torch.tensor(X_all, dtype=torch.float32)
    loader   = DataLoader(TensorDataset(Xt), batch_size=32, shuffle=True)
    for _ in range(ae_epochs):
        model.train()
        for (Xb,) in loader:
            Xb = Xb.to(device)
            recon, _ = model(Xb)
            loss = F.mse_loss(recon, Xb)
            opt.zero_grad(); loss.backward(); opt.step()


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
    return torch.sigmoid(logits).cpu().numpy().ravel()


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION ON ds003474
# ══════════════════════════════════════════════════════════════════════════════
def cv_evaluate(arch_name: str, X: np.ndarray, y: np.ndarray,
                feat_names: list) -> dict:
    """5-fold stratified CV. Returns per-fold and mean AUC."""
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs   = []
    all_probs, all_labels = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        X_tr_s, X_va_s, imp, scl = preprocess(X_tr, X_va)

        n_in = X_tr_s.shape[1]
        groups = make_feature_groups(feat_names)

        if arch_name == "DeepMLP":
            model = DeepMLP(n_in)
        elif arch_name == "AutoencoderCLF":
            ae      = Autoencoder(n_in, latent=64)
            clf_net = LatentClassifier(latent=64)
            # Pretrain AE on training fold
            train_autoencoder(ae, X_tr_s, ae_epochs=80)
            ae.eval()
            # Encode training and val
            with torch.no_grad():
                _, z_tr = ae(torch.tensor(X_tr_s, dtype=torch.float32))
                _, z_va = ae(torch.tensor(X_va_s, dtype=torch.float32))
            X_tr_s = z_tr.numpy();  X_va_s = z_va.numpy()
            n_in   = 64
            model  = clf_net
        elif arch_name == "EnsembleMLP":
            model = EnsembleMLP(groups, latent=32)
        elif arch_name == "FeatureAttn":
            model = FeatureAttentionNet(n_in, d_model=16, n_heads=4, n_layers=2)
        else:
            raise ValueError(arch_name)

        best_auc = train_model(model, X_tr_s, y_tr, X_va_s, y_va)
        probs    = predict_proba(model, X_va_s)
        auc_val  = roc_auc_score(y_va, probs)
        aucs.append(auc_val)
        all_probs.append(probs)
        all_labels.append(y_va)
        log.info("  %s  fold=%d  AUC=%.3f", arch_name, fold+1, auc_val)

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    oof_auc    = roc_auc_score(all_labels, all_probs)

    log.info("%s  CV mean=%.3f±%.3f  OOF AUC=%.3f",
             arch_name, np.mean(aucs), np.std(aucs), oof_auc)

    return {
        "arch": arch_name,
        "fold_aucs": aucs,
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "oof_auc":  float(oof_auc),
        "probs":    all_probs,
        "labels":   all_labels,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FINAL MODEL TRAINING (full ds003474) + ZERO-SHOT EVAL
# ══════════════════════════════════════════════════════════════════════════════
def train_final(arch_name: str, X: np.ndarray, y: np.ndarray,
                feat_names: list, n_in_override: int = None):
    """Train on full ds003474, return (model, imp, scaler, encoded_X)."""
    X_s, imp, scl = preprocess(X)
    n_in   = n_in_override or X_s.shape[1]
    groups = make_feature_groups(feat_names)

    if arch_name == "DeepMLP":
        model = DeepMLP(n_in)
        train_model(model, X_s, y, X_s, y, epochs=EPOCHS, patience=9999)
    elif arch_name == "AutoencoderCLF":
        ae  = Autoencoder(n_in, latent=64)
        clf = LatentClassifier(latent=64)
        train_autoencoder(ae, X_s, ae_epochs=120)
        ae.eval()
        with torch.no_grad():
            _, Z = ae(torch.tensor(X_s, dtype=torch.float32))
        X_s = Z.numpy()
        n_in = 64
        train_model(clf, X_s, y, X_s, y, epochs=EPOCHS, patience=9999)
        return (ae, clf), imp, scl, X_s
    elif arch_name == "EnsembleMLP":
        model = EnsembleMLP(groups, latent=32)
        train_model(model, X_s, y, X_s, y, epochs=EPOCHS, patience=9999)
    elif arch_name == "FeatureAttn":
        model = FeatureAttentionNet(n_in)
        train_model(model, X_s, y, X_s, y, epochs=EPOCHS, patience=9999)
    else:
        raise ValueError(arch_name)
    return model, imp, scl, X_s


def zero_shot_eval(arch_name: str, model_bundle, imp, scl,
                   feat_train: list,
                   X_ext: np.ndarray, y_ext: np.ndarray,
                   feat_ext: list) -> float:
    """Align features, preprocess, predict, return AUC."""
    X_ext_imp, _, _ = preprocess(X_ext)   # impute NaN in ext data
    # Align to training columns
    feat_ext_imp = feat_ext   # after preprocess columns unchanged
    X_aligned = align_features(
        imp.transform(np.zeros((1, len(feat_train)))),   # dummy for column means
        feat_train, X_ext_imp, feat_ext)
    X_scaled  = scl.transform(X_aligned)

    if arch_name == "AutoencoderCLF":
        ae, clf = model_bundle
        ae.eval(); clf.eval()
        with torch.no_grad():
            _, Z = ae(torch.tensor(X_scaled.astype(np.float32)))
        probs = predict_proba(clf, Z.numpy())
    else:
        probs = predict_proba(model_bundle, X_scaled.astype(np.float32))

    try:
        auc = roc_auc_score(y_ext, probs)
    except Exception:
        auc = float("nan")
    return auc


# ══════════════════════════════════════════════════════════════════════════════
# ML BASELINE (LogReg, replicate published results)
# ══════════════════════════════════════════════════════════════════════════════
def ml_baseline_cv(X: np.ndarray, y: np.ndarray) -> dict:
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs  = []
    all_p, all_l = [], []
    for tr, va in skf.split(X, y):
        X_tr_s, X_va_s, *_ = preprocess(X[tr], X[va])
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                 class_weight="balanced", random_state=SEED)
        clf.fit(X_tr_s, y[tr])
        probs = clf.predict_proba(X_va_s)[:, 1]
        auc   = roc_auc_score(y[va], probs)
        aucs.append(auc)
        all_p.append(probs); all_l.append(y[va])
    all_p = np.concatenate(all_p); all_l = np.concatenate(all_l)
    oof   = roc_auc_score(all_l, all_p)
    log.info("LogReg baseline  CV mean=%.3f±%.3f  OOF=%.3f",
             np.mean(aucs), np.std(aucs), oof)
    return {"arch":"LogReg","fold_aucs":aucs,"mean_auc":np.mean(aucs),
            "std_auc":np.std(aucs),"oof_auc":oof,"probs":all_p,"labels":all_l}


def ml_baseline_zeroshot(X_tr, y_tr, feat_tr, X_ext, y_ext, feat_ext) -> float:
    X_tr_s, imp, scl = preprocess(X_tr)
    X_ext_i, _, _    = preprocess(X_ext)
    X_ext_a = align_features(
        imp.transform(np.zeros((1, len(feat_tr)))), feat_tr, X_ext_i, feat_ext)
    X_ext_s = scl.transform(X_ext_a)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                             class_weight="balanced", random_state=SEED)
    clf.fit(X_tr_s, y_tr)
    probs = clf.predict_proba(X_ext_s)[:, 1]
    try:
        return roc_auc_score(y_ext, probs)
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# SHAP FEATURE IMPORTANCE (best DL model)
# ══════════════════════════════════════════════════════════════════════════════
def compute_shap(model, X_tr_s: np.ndarray, X_va_s: np.ndarray,
                 feat_names: list, arch_name: str, top_n: int = 20):
    try:
        import shap
        log.info("Computing SHAP values for %s …", arch_name)
        background = torch.tensor(X_tr_s[:50], dtype=torch.float32)
        test_data  = torch.tensor(X_va_s[:40], dtype=torch.float32)

        def predict_fn(x):
            xt = torch.tensor(x, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                logits = model(xt).numpy().ravel()
            return 1.0 / (1.0 + np.exp(-logits))

        explainer = shap.KernelExplainer(predict_fn, background.numpy())
        shap_vals = explainer.shap_values(test_data.numpy(), nsamples=100)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        top_idx   = np.argsort(mean_abs)[::-1][:top_n]
        top_feats = [(feat_names[i], float(mean_abs[i])) for i in top_idx]
        return top_feats
    except Exception as e:
        log.warning("SHAP failed: %s", e)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# PERMUTATION IMPORTANCE (fast fallback)
# ══════════════════════════════════════════════════════════════════════════════
def permutation_importance(model, X: np.ndarray, y: np.ndarray,
                           feat_names: list, n_repeats: int = 10,
                           top_n: int = 20) -> list:
    base_probs = predict_proba(model, X)
    base_auc   = roc_auc_score(y, base_probs)
    importances = np.zeros(len(feat_names))
    rng = np.random.default_rng(SEED)
    for i in range(len(feat_names)):
        drops = []
        for _ in range(n_repeats):
            X_perm     = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            p   = predict_proba(model, X_perm)
            auc = roc_auc_score(y, p)
            drops.append(base_auc - auc)
        importances[i] = np.mean(drops)
    top_idx = np.argsort(importances)[::-1][:top_n]
    return [(feat_names[i], float(importances[i])) for i in top_idx]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_comparison(cv_results: list, zeroshot: dict,
                        ml_cv: dict, ml_zs: dict, out_path: Path):
    """4-panel figure: ROC for each dataset."""
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    datasets = [
        ("ds003474\n(train, 5-fold CV)", "oof", None),
        ("ds003478\n(resting, same subjects)", "ds003478", None),
        ("TDBRAIN\n(external lab, N=356)",  "TDBRAIN",  None),
        ("MODMA\n(external lab, N=53)",     "MODMA",    None),
    ]

    ARCH_COLORS = {"DeepMLP":"tab:blue","AutoencoderCLF":"tab:orange",
                   "EnsembleMLP":"tab:green","FeatureAttn":"tab:red","LogReg":"black"}

    for ax, (title, key, _) in zip(axes, datasets):
        # ML baseline
        if key == "oof":
            fpr, tpr, _ = roc_curve(ml_cv["labels"], ml_cv["probs"])
            ax.plot(fpr, tpr, color="black", lw=2, ls="--",
                    label=f"LogReg (AUC={ml_cv['oof_auc']:.3f})")
        else:
            ml_auc = ml_zs.get(key, float("nan"))
            # Dummy line placeholder for ML
            ax.axhline(0, color="white", lw=0, label=f"LogReg zero-shot AUC={ml_auc:.3f}")

        # DL architectures
        for res in cv_results:
            arch = res["arch"]
            c    = ARCH_COLORS.get(arch, "gray")
            if key == "oof":
                fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
                auc_lbl = f'{res["oof_auc"]:.3f}'
                ax.plot(fpr, tpr, color=c, lw=2, label=f"{arch} (AUC={auc_lbl})")
            else:
                auc_val = zeroshot.get(arch, {}).get(key, float("nan"))
                ax.axhline(0, color="white", lw=0,
                           label=f"{arch} zero-shot={auc_val:.3f}")

        if key == "oof":
            ax.plot([0,1],[0,1],"k:",lw=0.8)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    plt.suptitle("ML vs DL — ROC comparison across datasets", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("ROC figure saved → %s", out_path)


def plot_feature_importance(top_feats: list, arch_name: str, out_path: Path):
    if not top_feats:
        return
    names  = [f[0] for f in top_feats]
    values = [f[1] for f in top_feats]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#d73027" if v > 0 else "#4575b4" for v in values]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean ΔAUC (permutation importance)")
    ax.set_title(f"Feature importance — {arch_name}", fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Feature importance figure → %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 72)
    log.info("DL Classification on EEG features (no re-extraction)")
    log.info("=" * 72)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("\n[1] Loading datasets …")
    d474 = load_ds003474()
    d478 = load_ds003478()
    dTDB = load_tdbrain()
    dMOD = load_modma()

    X474, y474, f474 = get_XY(d474)
    X478, y478, f478 = get_XY(d478)
    XTDB, yTDB, fTDB = get_XY(dTDB)
    XMOD, yMOD, fMOD = get_XY(dMOD)

    log.info("\n[2] Feature overlap summary")
    s474 = set(f474); s478 = set(f478); sTDB = set(fTDB); sMOD = set(fMOD)
    log.info("  474∩478=%d  474∩TDB=%d  474∩MOD=%d  ALL4=%d",
             len(s474&s478), len(s474&sTDB), len(s474&sMOD),
             len(s474&s478&sTDB&sMOD))

    # ── 2. ML baseline CV ────────────────────────────────────────────────────
    log.info("\n[3] ML baseline (LogReg, 5-fold CV on ds003474) …")
    ml_cv = ml_baseline_cv(X474, y474)

    # ── 3. DL CV on ds003474 ─────────────────────────────────────────────────
    log.info("\n[4] DL architectures — 5-fold CV on ds003474 …")
    archs = ["DeepMLP", "AutoencoderCLF", "EnsembleMLP", "FeatureAttn"]
    cv_results = []
    for arch in archs:
        log.info("  Architecture: %s", arch)
        res = cv_evaluate(arch, X474, y474, f474)
        cv_results.append(res)

    # ── 4. Train final models on full ds003474 ───────────────────────────────
    log.info("\n[5] Training final models on full ds003474 …")
    final_models = {}
    for arch in archs:
        log.info("  Fitting final %s …", arch)
        bundle, imp, scl, X_enc = train_final(arch, X474, y474, f474)
        final_models[arch] = (bundle, imp, scl, f474, X_enc)

    # ── 5. Zero-shot evaluation ───────────────────────────────────────────────
    log.info("\n[6] Zero-shot evaluation …")
    ext_datasets = {
        "ds003478": (X478, y478, f478),
        "TDBRAIN":  (XTDB, yTDB, fTDB),
        "MODMA":    (XMOD, yMOD, fMOD),
    }
    zeroshot = {arch: {} for arch in archs}
    for arch in archs:
        bundle, imp, scl, feat_tr, X_tr_enc = final_models[arch]
        for ds_name, (X_ext, y_ext, f_ext) in ext_datasets.items():
            auc = zero_shot_eval(arch, bundle, imp, scl, feat_tr, X_ext, y_ext, f_ext)
            zeroshot[arch][ds_name] = auc
            log.info("  %s → %s  AUC=%.3f", arch, ds_name, auc)

    # ML zero-shot
    ml_zs = {}
    X474_s, imp474, scl474 = preprocess(X474)
    for ds_name, (X_ext, y_ext, f_ext) in ext_datasets.items():
        auc = ml_baseline_zeroshot(X474, y474, f474, X_ext, y_ext, f_ext)
        ml_zs[ds_name] = auc
        log.info("  LogReg → %s  AUC=%.3f", ds_name, auc)

    # ── 6. Feature importance (best DL model) ────────────────────────────────
    log.info("\n[7] Permutation feature importance …")
    best_res  = max(cv_results, key=lambda r: r["oof_auc"])
    best_arch = best_res["arch"]
    log.info("  Best DL arch: %s  OOF AUC=%.3f", best_arch, best_res["oof_auc"])

    bundle_b, imp_b, scl_b, feat_b, X_b_enc = final_models[best_arch]
    X_s_b = scl_b.transform(imp_b.transform(X474))
    if best_arch == "AutoencoderCLF":
        ae_b, clf_b = bundle_b
        ae_b.eval()
        with torch.no_grad():
            _, Z_b = ae_b(torch.tensor(X_s_b.astype(np.float32)))
        top_feats = permutation_importance(clf_b, Z_b.numpy(), y474,
                                           [f"z{i}" for i in range(64)], top_n=20)
    else:
        top_feats = permutation_importance(bundle_b, X_s_b.astype(np.float32), y474,
                                           feat_b, top_n=25)

    # ── 7. Results table ─────────────────────────────────────────────────────
    log.info("\n" + "="*72)
    log.info("RESULTS SUMMARY")
    log.info("="*72)

    published_ml = {"ds003474": 0.836, "ds003478": 0.715, "TDBRAIN": 0.727, "MODMA": 0.677}

    rows = []
    # ML row
    ml_row = {"arch": "LogReg (ML baseline)",
              "ds003474_cv_auc": f"{ml_cv['oof_auc']:.3f}",
              "ds003478_zs":     f"{ml_zs['ds003478']:.3f}",
              "TDBRAIN_zs":      f"{ml_zs['TDBRAIN']:.3f}",
              "MODMA_zs":        f"{ml_zs['MODMA']:.3f}"}
    rows.append(ml_row)

    for res in cv_results:
        arch = res["arch"]
        row  = {"arch": arch,
                "ds003474_cv_auc": f"{res['oof_auc']:.3f}",
                "ds003478_zs":     f"{zeroshot[arch]['ds003478']:.3f}",
                "TDBRAIN_zs":      f"{zeroshot[arch]['TDBRAIN']:.3f}",
                "MODMA_zs":        f"{zeroshot[arch]['MODMA']:.3f}"}
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_csv = OUT_DIR / "dl_classification_results.csv"
    results_df.to_csv(str(results_csv), index=False)

    # Console table
    header = f"{'Architecture':<22} {'ds003474 (CV)':>14} {'ds003478 (0-shot)':>18} {'TDBRAIN (0-shot)':>18} {'MODMA (0-shot)':>16}"
    sep    = "-" * len(header)
    log.info(sep)
    log.info(header)
    log.info(sep)
    pub = f"{'Published best ML':<22} {'0.836':>14} {'0.715':>18} {'0.727':>18} {'0.677':>16}"
    log.info(pub)
    log.info(sep)
    for row in rows:
        line = (f"{row['arch']:<22} {row['ds003474_cv_auc']:>14} "
                f"{row['ds003478_zs']:>18} {row['TDBRAIN_zs']:>18} {row['MODMA_zs']:>16}")
        log.info(line)
    log.info(sep)

    # ── 8. Figures ────────────────────────────────────────────────────────────
    log.info("\n[8] Generating figures …")
    plot_roc_comparison(cv_results, zeroshot, ml_cv, ml_zs,
                        OUT_DIR / "dl_roc_comparison.png")
    plot_feature_importance(top_feats, best_arch,
                             OUT_DIR / "dl_feature_importance.png")

    # Save feature importance
    fi_df  = pd.DataFrame(top_feats, columns=["feature", "importance_delta_auc"])
    fi_csv = OUT_DIR / "dl_feature_importance.csv"
    fi_df.to_csv(str(fi_csv), index=False)

    log.info("\nOutputs:")
    log.info("  %s", results_csv)
    log.info("  %s", OUT_DIR / "dl_roc_comparison.png")
    log.info("  %s", OUT_DIR / "dl_feature_importance.png")
    log.info("  %s", fi_csv)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
