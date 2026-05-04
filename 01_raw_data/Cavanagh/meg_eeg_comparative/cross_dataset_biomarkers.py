"""
Cross-Dataset EEG Biomarker Consistency Analysis
=================================================
Scientific question: which EEG features show a CONSISTENT direction of
difference (CTL vs MDD) across all 4 datasets, independent of lab,
hardware, paradigm, and diagnostic criteria?

Method: for each shared feature compute per dataset —
  • Cohen's d  (effect size, signed: positive = MDD > CTL)
  • Hedge's g  (small-N corrected Cohen's d)
  • Mann-Whitney U p-value (non-parametric)
  • Spearman ρ with severity (BDI or PHQ-9) where available

Then:
  1. Heatmap:  feature × dataset, colour = Hedge's g
  2. Consistency score per feature: how many datasets agree on direction
  3. Forest plot: pooled random-effects meta-analysis for top features
  4. Severity correlation comparison across datasets
  5. Summary table saved to CSV

Datasets:
  ds003474 — EEG task PST,    N=87,  BDI≥13    (Cavanagh, USA)
  ds003478 — EEG resting,     N=91,  BDI≥13    (same participants as ds003474)
  MODMA    — EEG resting,     N=53,  PHQ-9 clinical (Lanzhou, China)
  TDBRAIN  — EEG resting,     N=356, BDI subthreshold (Amsterdam, NL)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE = Path("/media/neuraldyn/PortableSSD/DEPRESSION/Cavanagh")
MEG  = BASE / "meg_eeg_comparative"

# ── feature name normalisation ────────────────────────────────────────────

def normalize_cols(df):
    """Map MODMA/TDBRAIN naming conventions → ds003474 canonical names."""
    rename = {}
    for c in df.columns:
        nc = c
        # dds_frontal_A1_mean → dds_frontal_A1
        if nc.endswith("_mean") and nc.startswith("dds_"):
            nc = nc[:-5]
        # info_AIS_frontal → ais_frontal
        if nc.startswith("info_AIS_"):
            nc = "ais_" + nc[9:].lower()
        # info_TE_LH_to_frontal → te_LH_frontal
        if nc.startswith("info_TE_") and "_to_" in nc:
            parts = nc[8:].split("_to_")
            nc = f"te_{parts[0]}_{parts[1].lower()}"
        # info_PID_redundancy → pid_redundancy
        if nc.startswith("info_PID_"):
            nc = "pid_" + nc[9:].lower()
        rename[c] = nc
    return df.rename(columns=rename)


META_COLS = {"subject_id", "group", "label", "MDD", "BDI", "BDI_pre",
             "PHQ9", "age", "gender", "dataset", "task",
             "feature_set", "classifier", "roc_auc", "roc_auc_std"}

# ── effect size helpers ───────────────────────────────────────────────────

def cohens_d(a, b):
    """Signed Cohen's d: positive = group_a > group_b (MDD > CTL)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_sd = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled_sd


def hedges_g(a, b):
    """Hedge's g = Cohen's d × correction factor for small N."""
    d = cohens_d(a, b)
    if np.isnan(d):
        return np.nan
    n = len(a) + len(b)
    cf = 1 - (3 / (4*n - 9))   # correction factor
    return d * cf


def hedges_g_se(na, nb, g):
    """Standard error of Hedge's g."""
    if np.isnan(g):
        return np.nan
    return np.sqrt((na+nb)/(na*nb) + g**2/(2*(na+nb)))


def random_effects_meta(gs, ses):
    """DerSimonian-Laird random-effects pooled estimate."""
    valid = [(g, s) for g, s in zip(gs, ses) if not (np.isnan(g) or np.isnan(s) or s == 0)]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan
    gs_v, ses_v = zip(*valid)
    gs_v, ses_v = np.array(gs_v), np.array(ses_v)
    wi   = 1 / ses_v**2
    Q    = np.sum(wi * (gs_v - np.sum(wi*gs_v)/np.sum(wi))**2)
    k    = len(gs_v)
    C    = np.sum(wi) - np.sum(wi**2)/np.sum(wi)
    tau2 = max(0, (Q - (k-1)) / C)
    wi_re = 1 / (ses_v**2 + tau2)
    pooled = np.sum(wi_re * gs_v) / np.sum(wi_re)
    se_pooled = np.sqrt(1 / np.sum(wi_re))
    z = pooled / se_pooled
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return pooled, se_pooled, p

# ── loader ────────────────────────────────────────────────────────────────

def load_dataset(path, label_col, label_map=None, severity_col=None):
    df = pd.read_csv(path, dtype={"subject_id": str})
    df = normalize_cols(df)
    if label_map:
        df[label_col] = df[label_col].map(label_map)
    y = df[label_col].astype(int).values
    feat_cols = [c for c in df.columns if c not in META_COLS and df[c].dtype != object]
    X = df[feat_cols]
    sev = df[severity_col].values if severity_col and severity_col in df.columns else None
    return X, y, sev, feat_cols

# ── main analysis ─────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Cross-Dataset EEG Biomarker Consistency Analysis")
    log.info("=" * 70)

    # ── Load datasets ─────────────────────────────────────────────────────
    specs = {
        "ds003474\n(task PST\nN=87 BDI)": dict(
            path=BASE/"ds003474/code/eeg_depression_classification/results/dds/features_dds_merged.csv",
            label_col="group", label_map=None, severity_col="BDI"),
        "ds003478\n(resting\nN=91 BDI)": dict(
            path=MEG/"ds003478_run01_dds_info_baseline_features.csv",
            label_col="group", label_map=None, severity_col="BDI"),
        "MODMA\n(resting\nN=53 PHQ9)": dict(
            path=MEG/"modma_rest_features.csv",
            label_col="label", label_map={"MDD":1,"HC":0}, severity_col="PHQ9"),
        "TDBRAIN\n(resting\nN=356 BDI)": dict(
            path=MEG/"tdbrain_restEO_features.csv",
            label_col="label", label_map={"MDD":1,"HC":0}, severity_col="BDI_pre"),
    }

    datasets = {}
    for name, kw in specs.items():
        X, y, sev, cols = load_dataset(**kw)
        n_ctl, n_mdd = (y==0).sum(), (y==1).sum()
        log.info("%-30s N=%d  CTL=%d  MDD=%d  features=%d",
                 name.replace('\n',' '), len(y), n_ctl, n_mdd, len(cols))
        datasets[name] = dict(X=X, y=y, sev=sev, cols=set(cols),
                               n_ctl=n_ctl, n_mdd=n_mdd)

    # ── Universal feature set ─────────────────────────────────────────────
    universal = set.intersection(*[d["cols"] for d in datasets.values()])

    # Validity filter: exclude features with CV (std/|mean|) < 0.02 in ANY dataset
    # This removes DDS frequency parameters (f1, f2) that hit fitting boundaries
    # in MODMA/TDBRAIN (~25 Hz, std<0.3) making them incommensurable with
    # ds003474/ds003478 where they vary freely (~8-11 Hz, std~2).
    def _cv(vals):
        m = abs(np.nanmean(vals))
        s = np.nanstd(vals)
        return s / m if m > 1e-10 else (s if s > 0 else 0)

    valid_features = []
    excluded = []
    for f in universal:
        cvs = [_cv(d["X"][f].values.astype(float)) for d in datasets.values()
               if f in d["X"].columns]
        if all(cv > 0.02 for cv in cvs):
            valid_features.append(f)
        else:
            excluded.append(f)

    log.info("\nUniversal features: %d total  →  %d valid after CV filter  (%d excluded — near-constant)",
             len(universal), len(valid_features), len(excluded))
    if excluded:
        log.info("  Excluded (incommensurable scale/boundary): %s", excluded)

    dds_u  = sorted(f for f in valid_features if f.startswith("dds_"))
    info_u = sorted(f for f in valid_features if any(f.startswith(x)
                    for x in ["ais_","te_","pid_"]))
    spec_u = sorted(f for f in valid_features
                    if any(b in f for b in ["delta","theta","alpha","beta","gamma"]))
    features = dds_u + info_u + spec_u
    log.info("Valid set: DDS=%d  Info=%d  Spectral=%d  Total=%d",
             len(dds_u), len(info_u), len(spec_u), len(features))

    # ── Compute effect sizes ──────────────────────────────────────────────
    log.info("\nComputing effect sizes (Hedge's g) per feature per dataset …")
    records = []
    ds_names = list(datasets.keys())

    for feat in features:
        row = {"feature": feat}
        gs, ses, ps = [], [], []
        for ds_name, d in datasets.items():
            if feat not in d["X"].columns:
                g, se, p = np.nan, np.nan, np.nan
            else:
                vals = d["X"][feat].values.astype(float)
                mdd  = vals[d["y"] == 1]
                ctl  = vals[d["y"] == 0]
                mdd  = mdd[~np.isnan(mdd)]
                ctl  = ctl[~np.isnan(ctl)]
                g    = hedges_g(mdd, ctl)
                se   = hedges_g_se(len(mdd), len(ctl), g)
                _, p = stats.mannwhitneyu(mdd, ctl, alternative="two-sided") \
                       if len(mdd) >= 3 and len(ctl) >= 3 else (np.nan, np.nan)
            short = ds_name.split('\n')[0]
            row[f"g_{short}"]  = g
            row[f"se_{short}"] = se
            row[f"p_{short}"]  = p
            gs.append(g)
            ses.append(se)
            ps.append(p)

        # consistency: number of datasets with same-sign effect
        signs = [np.sign(g) for g in gs if not np.isnan(g)]
        n_pos = signs.count(1)
        n_neg = signs.count(-1)
        row["n_consistent"] = max(n_pos, n_neg)
        row["direction"]    = "MDD>" if n_pos >= n_neg else "CTL>"
        row["n_sig_p05"]    = sum(p < 0.05 for p in ps if not np.isnan(p))
        row["n_sig_p10"]    = sum(p < 0.10 for p in ps if not np.isnan(p))

        # pooled random-effects
        g_pool, se_pool, p_pool = random_effects_meta(gs, ses)
        row["g_pooled"] = g_pool
        row["se_pooled"] = se_pool
        row["p_pooled"]  = p_pool

        records.append(row)

    results = pd.DataFrame(records).sort_values(
        ["n_consistent", "n_sig_p05", "g_pooled"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # ── Severity correlations ─────────────────────────────────────────────
    log.info("Computing severity correlations …")
    sev_records = []
    sev_labels = {
        "ds003474\n(task PST\nN=87 BDI)": "BDI",
        "ds003478\n(resting\nN=91 BDI)": "BDI",
        "MODMA\n(resting\nN=53 PHQ9)": "PHQ-9",
        "TDBRAIN\n(resting\nN=356 BDI)": "BDI",
    }
    for feat in features:
        row = {"feature": feat}
        for ds_name, d in datasets.items():
            if feat not in d["X"].columns or d["sev"] is None:
                continue
            vals = d["X"][feat].values.astype(float)
            sev  = d["sev"].astype(float)
            mask = ~(np.isnan(vals) | np.isnan(sev))
            if mask.sum() < 10:
                continue
            rho, p = stats.spearmanr(vals[mask], sev[mask])
            short = ds_name.split('\n')[0]
            row[f"rho_{short}"]  = rho
            row[f"rsp_{short}"] = p
        sev_records.append(row)
    sev_df = pd.DataFrame(sev_records)

    # ── Save tables ───────────────────────────────────────────────────────
    results.to_csv(MEG / "biomarker_consistency.csv", index=False)
    sev_df.to_csv(MEG / "biomarker_severity_correlations.csv", index=False)
    log.info("Tables saved.")

    # ── Print top findings ────────────────────────────────────────────────
    top = results[results["n_consistent"] >= 3].head(20)
    log.info("\nTop features consistent in ≥3/4 datasets:")
    log.info("%-35s %5s %5s %7s %5s %6s",
             "Feature", "n_con", "n_p05", "g_pool", "p_pool", "dir")
    for _, r in top.iterrows():
        log.info("%-35s %5d %5d %7.3f %5.3f %6s",
                 r.feature, r.n_consistent, r.n_sig_p05,
                 r.g_pooled if not np.isnan(r.g_pooled) else 0,
                 r.p_pooled if not np.isnan(r.p_pooled) else 1,
                 r.direction)

    # ── Figure 1: Heatmap ─────────────────────────────────────────────────
    log.info("\nGenerating figures …")
    ds_short = [n.split('\n')[0] for n in ds_names]
    g_cols   = [f"g_{s}" for s in ds_short]

    # sort features by group then pooled g
    feat_order = results["feature"].tolist()
    G = results[g_cols].values.astype(float)

    fig, ax = plt.subplots(figsize=(10, max(8, len(feat_order)*0.3)))
    vmax = np.nanpercentile(np.abs(G), 95)
    im = ax.imshow(G.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_order, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ds_short)))
    ax.set_yticklabels([n.replace('\n', ' ') for n in ds_names], fontsize=9)
    ax.set_title("Hedge's g (MDD vs CTL) across datasets\nBlue = MDD < CTL | Red = MDD > CTL",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label="Hedge's g")
    # mark significant cells
    for xi, feat in enumerate(feat_order):
        for yi, ds_name in enumerate(ds_names):
            short = ds_name.split('\n')[0]
            p = results.loc[results.feature == feat, f"p_{short}"].values
            if len(p) and not np.isnan(p[0]) and p[0] < 0.05:
                ax.text(xi, yi, "*", ha="center", va="center", fontsize=9, color="white")
    plt.tight_layout()
    fig.savefig(MEG / "biomarker_heatmap.png", dpi=150)
    plt.close()

    # ── Figure 2: Forest plot — all features consistent in ≥3/4 datasets ──
    consistent = results[results["n_consistent"] >= 3].head(15).copy()

    n_feat = len(consistent)
    fig, axes = plt.subplots(1, 1, figsize=(12, max(6, n_feat * 0.55 + 2)))
    ax = axes
    colors = ["#2166ac", "#4dac26", "#d01c8b", "#f1a340"]  # one per dataset
    y_base = np.arange(n_feat) * (len(ds_names) + 2)

    for fi, (_, row) in enumerate(consistent.iterrows()):
        feat = row["feature"]
        y0 = y_base[fi]
        for di, (ds_name, col) in enumerate(zip(ds_names, g_cols)):
            short = ds_name.split('\n')[0]
            g  = row[f"g_{short}"]
            se = row[f"se_{short}"]
            p  = row[f"p_{short}"]
            if np.isnan(g):
                continue
            y = y0 + di
            ax.errorbar(g, y, xerr=1.96*se, fmt="o", color=colors[di],
                        markersize=5, capsize=3, linewidth=1.2,
                        markeredgewidth=0.5, markeredgecolor="white",
                        label=ds_name.replace('\n',' ') if fi == 0 else "")
            if not np.isnan(p) and p < 0.05:
                ax.text(g + 1.96*se + 0.02, y, "*", va="center", fontsize=9,
                        color=colors[di])

        # pooled effect
        g_p  = row["g_pooled"]
        se_p = row["se_pooled"]
        p_p  = row["p_pooled"]
        if not np.isnan(g_p):
            y_pool = y0 + len(ds_names) + 0.5
            ax.errorbar(g_p, y_pool, xerr=1.96*se_p, fmt="D", color="black",
                        markersize=7, capsize=4, linewidth=2,
                        label="Pooled (RE)" if fi == 0 else "")
            sig_str = "**" if not np.isnan(p_p) and p_p < 0.01 else \
                      "*"  if not np.isnan(p_p) and p_p < 0.05 else ""
            ax.text(g_p + 1.96*se_p + 0.02, y_pool,
                    f"{sig_str} g={g_p:.2f} p={p_p:.3f}" if not np.isnan(p_p) else f"g={g_p:.2f}",
                    va="center", fontsize=8)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvspan(-0.2, 0.2, alpha=0.05, color="gray")  # negligible effect band
    ax.set_yticks(y_base + (len(ds_names)-1)/2)
    ax.set_yticklabels(consistent["feature"].tolist(), fontsize=8)
    ax.set_xlabel("Hedge's g  (positive = MDD > CTL)", fontsize=10)
    ax.set_title("Forest plot — features consistent across ≥3/4 datasets\n"
                 "Error bars = 95% CI  |  ◆ = pooled random-effects estimate  |  * p<0.05",
                 fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    # deduplicate
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(MEG / "biomarker_forest_plot.png", dpi=150)
    plt.close()

    # ── Figure 3: Severity correlation comparison ─────────────────────────
    # Top features by |pooled g| that also have severity data
    sev_top = results[results["n_consistent"] >= 3].head(20)
    sev_merged = sev_top.merge(sev_df, on="feature", how="left")

    rho_cols = [c for c in sev_df.columns if c.startswith("rho_")]
    if rho_cols:
        fig, ax = plt.subplots(figsize=(10, max(5, len(sev_merged)*0.4 + 1)))
        ds_colors = dict(zip(ds_short, colors))
        y_pos = np.arange(len(sev_merged))
        for di, rc in enumerate(rho_cols):
            short = rc[4:]
            rho_vals = sev_merged[rc].values.astype(float)
            p_col = f"rsp_{short}"
            p_vals = sev_merged[p_col].values.astype(float) \
                     if p_col in sev_merged.columns else np.full(len(rho_vals), np.nan)
            ax.scatter(rho_vals, y_pos + di*0.2 - 0.3,
                       color=ds_colors.get(short, "gray"),
                       s=40, label=short, zorder=3,
                       alpha=0.85)
            # mark significant
            for yi, (rho, p) in enumerate(zip(rho_vals, p_vals)):
                if not np.isnan(p) and p < 0.05:
                    ax.text(rho + 0.02, yi + di*0.2 - 0.3, "*",
                            va="center", fontsize=9,
                            color=ds_colors.get(short, "gray"))
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvspan(-0.1, 0.1, alpha=0.05, color="gray")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sev_merged["feature"].tolist(), fontsize=8)
        ax.set_xlabel("Spearman ρ with severity (BDI or PHQ-9)", fontsize=10)
        ax.set_title("Severity correlation — top consistent features\n* p<0.05",
                     fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(MEG / "biomarker_severity_correlations.png", dpi=150)
        plt.close()

    log.info("Figures saved:")
    log.info("  %s", MEG / "biomarker_heatmap.png")
    log.info("  %s", MEG / "biomarker_forest_plot.png")
    log.info("  %s", MEG / "biomarker_severity_correlations.png")
    log.info("  %s", MEG / "biomarker_consistency.csv")

    # ── Final narrative summary ───────────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("BIOMARKER CONSISTENCY SUMMARY")
    log.info("="*70)
    all4 = results[results["n_consistent"] == 4]
    log.info("Features consistent in ALL 4 datasets: %d", len(all4))
    for _, r in all4.iterrows():
        log.info("  %-35s dir=%-6s g_pool=%+.3f p_pool=%.3f n_sig=%d/4",
                 r.feature, r.direction,
                 r.g_pooled if not np.isnan(r.g_pooled) else 0,
                 r.p_pooled if not np.isnan(r.p_pooled) else 1,
                 r.n_sig_p05)

    log.info("\nFeatures consistent in 3/4 datasets:")
    three = results[results["n_consistent"] == 3]
    for _, r in three.iterrows():
        log.info("  %-35s dir=%-6s g_pool=%+.3f p_pool=%.3f n_sig=%d/4",
                 r.feature, r.direction,
                 r.g_pooled if not np.isnan(r.g_pooled) else 0,
                 r.p_pooled if not np.isnan(r.p_pooled) else 1,
                 r.n_sig_p05)

    log.info("\nDone. Results → %s", MEG)


if __name__ == "__main__":
    main()
