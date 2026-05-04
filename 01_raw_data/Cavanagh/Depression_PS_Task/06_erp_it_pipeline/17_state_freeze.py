"""
17_state_freeze.py — Research program state snapshot (2026-05-02)
Verifies key files exist, recomputes key statistics, and saves JSON + TXT summary.
"""

import json, datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/Depression_PS_Task")
DERIV   = BASE / "derivatives"
ASSETS  = Path("/media/neuraldyn/PortableSSD/DEPRESSION/06_manuscript_assets")
MEG_D   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/ds005356/derivatives")

KEY_FILES = {
    "ais_eeg":        DERIV / "erp_it_cavanagh/delta_ais_aggregated.csv",
    "ais_meg_full":   MEG_D / "meg_ais_pre_full_sample_results.csv",
    "clinical":       DERIV / "clinical_lookup_ps_task.csv",
    "ais_rest_tdb":   ASSETS / "tdbrain_ais_rest_results.csv",
    "robustness":     DERIV / "erp_it_cavanagh/ais_pre_robustness_stats.csv",
    "subtype_ais":    DERIV / "subtype_ais_eeg_subjects.csv",
    "searchlight":    MEG_D / "channel_searchlight_results.csv",
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    """Pooled-SD Cohen's d (a > b direction)."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan

def mwu(a, b):
    """Mann-Whitney U p-value."""
    _, p = stats.mannwhitneyu(a, b, alternative='greater')
    return float(p)

# ── 1. File presence ──────────────────────────────────────────────────────────
print("=" * 60)
print("STATE FREEZE: AIS_pre MDD Research Program")
print(f"Date: {datetime.date.today()}")
print("=" * 60)

file_status = {}
for name, path in KEY_FILES.items():
    exists = path.exists()
    file_status[name] = {"path": str(path), "exists": exists}
    tag = "OK" if exists else "MISSING"
    print(f"  [{tag}] {name}: {path.name}")

# ── 2. Primary EEG finding ────────────────────────────────────────────────────
print("\n--- PRIMARY EEG FINDING (Cavanagh) ---")
eeg = pd.read_csv(KEY_FILES["ais_eeg"])
ctl_ais = eeg.loc[eeg["group"] == "CTL", "mean_AIS_pre"].dropna().values
mdd_ais = eeg.loc[eeg["group"] == "MDD_any", "mean_AIS_pre"].dropna().values
d_eeg   = cohens_d(ctl_ais, mdd_ais)
p_eeg   = mwu(ctl_ais, mdd_ais)
print(f"  CTL: {np.mean(ctl_ais):.4f} ± {np.std(ctl_ais, ddof=1):.4f}  (N={len(ctl_ais)})")
print(f"  MDD: {np.mean(mdd_ais):.4f} ± {np.std(mdd_ais, ddof=1):.4f}  (N={len(mdd_ais)})")
print(f"  d = {d_eeg:.3f},  p = {p_eeg:.4f}")
expected_d_eeg = 0.817  # canonical: -200ms, lag=1, bins=4 (not the -500ms robustness variant)
flag_eeg = "OK" if abs(d_eeg - expected_d_eeg) < 0.05 else f"DRIFT (expected ~{expected_d_eeg})"
print(f"  Expected d ≈ {expected_d_eeg}  →  {flag_eeg}")

# ── 3. MEG replication ────────────────────────────────────────────────────────
print("\n--- MEG REPLICATION (artifact excluded) ---")
meg = pd.read_csv(KEY_FILES["ais_meg_full"])
meg = meg[meg["subject_id"] != "sub-M87121835"]  # DC drift artifact excluded
meg_ctl = meg.loc[meg["group"] == "CTL", "mean_AIS_pre"].dropna().values
meg_mdd = meg.loc[meg["group"] == "MDD", "mean_AIS_pre"].dropna().values
d_meg   = cohens_d(meg_ctl, meg_mdd)
p_meg   = mwu(meg_ctl, meg_mdd)
print(f"  CTL: {np.mean(meg_ctl):.4f} ± {np.std(meg_ctl, ddof=1):.4f}  (N={len(meg_ctl)})")
print(f"  MDD: {np.mean(meg_mdd):.4f} ± {np.std(meg_mdd, ddof=1):.4f}  (N={len(meg_mdd)})")
print(f"  d = {d_meg:.3f},  p = {p_meg:.4f}")
expected_d_meg = 0.372  # sub-M87121835 excluded (artifact inflated CTL mean)
flag_meg = "OK" if abs(d_meg - expected_d_meg) < 0.05 else f"DRIFT (expected ~{expected_d_meg})"
print(f"  Expected d ≈ {expected_d_meg}  →  {flag_meg}")

# ── 4. Scar hypothesis (MDD_past > MDD_current dissociation) ─────────────────
print("\n--- SCAR HYPOTHESIS (EEG, SCID dissociation) ---")
clin = pd.read_csv(KEY_FILES["clinical"])
# Merge AIS_pre (EEG) with clinical
eeg_ids = eeg[["subject_id", "mean_AIS_pre", "group"]].copy()
merged  = pd.merge(eeg_ids, clin[["subject_id", "scid_group", "HamD"]], on="subject_id", how="inner")
cur  = merged.loc[merged["scid_group"] == "MDD_current", "mean_AIS_pre"].dropna().values
past = merged.loc[merged["scid_group"] == "MDD_past",    "mean_AIS_pre"].dropna().values
print(f"  MDD_current: N={len(cur)}  mean={np.mean(cur):.4f}")
print(f"  MDD_past:    N={len(past)}  mean={np.mean(past):.4f}")
if len(cur) > 1 and len(past) > 1:
    d_scar = cohens_d(past, cur)   # past LOWER than current → negative d
    # Direction: past has lower AIS → cur > past → cohens_d(cur, past) positive
    d_scar_dir = cohens_d(cur, past)
    _, p_scar = stats.mannwhitneyu(cur, past, alternative='greater')
    print(f"  d (cur > past) = {d_scar_dir:.3f},  p = {p_scar:.4f}")
    expected_d_scar = 0.965
    flag_scar = "OK" if abs(d_scar_dir - expected_d_scar) < 0.05 else f"DRIFT (expected ~{expected_d_scar})"
    print(f"  Expected d ≈ {expected_d_scar}  →  {flag_scar}")
    # HamD dissociation
    hamd_cur  = merged.loc[merged["scid_group"] == "MDD_current", "HamD"].dropna().values
    hamd_past = merged.loc[merged["scid_group"] == "MDD_past",    "HamD"].dropna().values
    print(f"  HamD current: {np.mean(hamd_cur):.1f}  vs  past: {np.mean(hamd_past):.1f}  (past lower = expected)")
else:
    d_scar_dir = np.nan
    p_scar     = np.nan
    flag_scar  = "INSUFFICIENT DATA"
    print("  Insufficient data for scar test")

# ── 5. TDBRAIN AIS_rest ───────────────────────────────────────────────────────
print("\n--- TDBRAIN AIS_rest REMISSION PREDICTOR ---")
tdb = pd.read_csv(KEY_FILES["ais_rest_tdb"])
rem    = tdb.loc[tdb["remitter_bdi"] == 1, "AIS_rest"].dropna().values
nonrem = tdb.loc[tdb["remitter_bdi"] == 0, "AIS_rest"].dropna().values
d_tdb  = cohens_d(rem, nonrem)
_, p_tdb = stats.mannwhitneyu(nonrem, rem, alternative='greater')  # non-remitters HIGHER
print(f"  Remitters:     N={len(rem)}   mean={np.mean(rem):.4f}")
print(f"  Non-remitters: N={len(nonrem)}  mean={np.mean(nonrem):.4f}")
print(f"  d (rem > nonrem) = {d_tdb:.3f}  [EXPECTED NEGATIVE — wrong direction]")
print(f"  p (nonrem > rem, one-sided) = {p_tdb:.4f}")
expected_d_tdb = -0.336
flag_tdb = "OK" if abs(d_tdb - expected_d_tdb) < 0.05 else f"DRIFT (expected ~{expected_d_tdb})"
print(f"  Expected d ≈ {expected_d_tdb}  →  {flag_tdb}")

# ── 6. MEG channel searchlight key result ─────────────────────────────────────
print("\n--- MEG SEARCHLIGHT (EEG064) ---")
if KEY_FILES["searchlight"].exists():
    sl = pd.read_csv(KEY_FILES["searchlight"])
    if "channel" in sl.columns:
        row = sl.loc[sl["channel"] == "EEG064"]
        if not row.empty:
            print(f"  EEG064 d={row['d'].values[0]:.3f},  p={row['p'].values[0]:.4f}")
            print(f"  N_CTL={row['n_ctl'].values[0]},  N_MDD={row['n_mdd'].values[0]}")
        else:
            print("  EEG064 not in searchlight CSV")
    else:
        print("  Searchlight CSV columns:", list(sl.columns))
else:
    print("  Searchlight CSV not found")

# ── 7. Compile state dict ──────────────────────────────────────────────────────
state = {
    "freeze_date": str(datetime.date.today()),
    "files": file_status,
    "primary_eeg": {
        "N_CTL": int(len(ctl_ais)), "N_MDD": int(len(mdd_ais)),
        "CTL_mean": float(np.mean(ctl_ais)), "MDD_mean": float(np.mean(mdd_ais)),
        "cohens_d": float(d_eeg), "mwu_p": float(p_eeg),
        "expected_d": expected_d_eeg, "flag": flag_eeg,
    },
    "meg_replication": {
        "N_CTL": int(len(meg_ctl)), "N_MDD": int(len(meg_mdd)),
        "CTL_mean": float(np.mean(meg_ctl)), "MDD_mean": float(np.mean(meg_mdd)),
        "cohens_d": float(d_meg), "mwu_p": float(p_meg),
        "expected_d": expected_d_meg, "flag": flag_meg,
    },
    "scar_hypothesis": {
        "N_current": int(len(cur)), "N_past": int(len(past)),
        "d_cur_over_past": float(d_scar_dir) if not np.isnan(d_scar_dir) else None,
        "mwu_p": float(p_scar) if not np.isnan(p_scar) else None,
        "expected_d": expected_d_scar, "flag": flag_scar,
    },
    "tdbrain_ais_rest": {
        "N_remitters": int(len(rem)), "N_nonremitters": int(len(nonrem)),
        "rem_mean": float(np.mean(rem)), "nonrem_mean": float(np.mean(nonrem)),
        "cohens_d_rem_over_nonrem": float(d_tdb), "mwu_p_nonrem_greater": float(p_tdb),
        "expected_d": expected_d_tdb, "flag": flag_tdb,
        "interpretation": "WRONG DIRECTION — non-remitters have higher AIS_rest; resting AIS ≠ task AIS_pre",
    },
    "pending": [
        "Magnetometer searchlight for MEG0511/MEG0921 (Cavanagh 2025 sensors)",
        "Verify EEG064 anatomical location (posterior temporal — mismatch with FCz hypothesis)",
        "Final manuscript Figure 1 (AIS_pre CTL vs MDD violin + ROC curve)",
        "Formal paper section: two-axis model (low task AIS_pre + high resting AIS_rest)",
        "Optional: contact jcavanagh@unm.edu for PST data with episode history",
    ],
    "key_conclusions": {
        "primary":   "AIS_pre (FCz, −200→0ms pre-feedback) CTL > MDD, d=0.817, p<0.001 (N=109; robustness −500ms variant: d=0.874)",
        "meg":       "MEG partial replication EEG064, d=0.558 (ungated); EEG007 unreliable dataset-wide",
        "scar":      "MDD_past lower AIS_pre than MDD_current despite lower HamD — scar hypothesis supported",
        "tdbrain":   "AIS_rest does NOT predict rTMS remission (wrong direction, d=−0.336) — construct dissociation",
        "subtype":   "AIS_LOW enriched 2.8× in MDD but not discrete (GMM weakly bimodal, TEPS null)",
    },
}

# ── 8. Save ────────────────────────────────────────────────────────────────────
out_dir = DERIV
json_path = out_dir / "RESEARCH_STATE_FROZEN.json"
txt_path  = out_dir / "RESEARCH_STATE_SUMMARY.txt"

with open(json_path, "w") as f:
    json.dump(state, f, indent=2)

with open(txt_path, "w") as f:
    f.write(f"AIS_pre MDD Research Program — State Freeze {datetime.date.today()}\n")
    f.write("=" * 70 + "\n\n")
    f.write("KEY CONCLUSIONS\n")
    for k, v in state["key_conclusions"].items():
        f.write(f"  {k.upper():10s}: {v}\n")
    f.write("\nKEY STATISTICS\n")
    f.write(f"  Primary EEG  d={d_eeg:.3f}  p={p_eeg:.4f}  [{flag_eeg}]\n")
    f.write(f"  MEG replic.  d={d_meg:.3f}  p={p_meg:.4f}  [{flag_meg}]\n")
    if not np.isnan(d_scar_dir):
        f.write(f"  Scar hyp.    d={d_scar_dir:.3f}  p={p_scar:.4f}  [{flag_scar}]\n")
    f.write(f"  TDBRAIN rest d={d_tdb:.3f}  p={p_tdb:.4f}  [{flag_tdb}]\n")
    f.write("\nPENDING TASKS\n")
    for i, task in enumerate(state["pending"], 1):
        f.write(f"  {i}. {task}\n")
    f.write(f"\nFiles saved:\n  {json_path}\n  {txt_path}\n")

print("\n" + "=" * 60)
print("STATE FREEZE COMPLETE")
print(f"  JSON: {json_path}")
print(f"  TXT:  {txt_path}")
print("=" * 60)
