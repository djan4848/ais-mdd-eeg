import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(".")

trial_file = ROOT / "derivatives/trial_roi_timeseries/trial_roi_timeseries.csv"
dds_file = ROOT / "derivatives/dds_peak_aligned_n450/dds_n450_results.csv"

out_dir = ROOT / "derivatives/trial_roi_timeseries_residual_r2pos"
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading ERP trial timeseries...")
erp = pd.read_csv(trial_file)

print("Loading DDS parameters...")
dds = pd.read_csv(dds_file)

print("\nERP columns:", list(erp.columns))
print("DDS columns:", list(dds.columns))

# --------------------------------------------------
# Filter DDS fits
# --------------------------------------------------
if "r2" not in dds.columns:
    raise ValueError("DDS file does not contain an 'r2' column.")

n_dds_before = len(dds)
dds = dds[dds["r2"] > 0].copy()
n_dds_after = len(dds)

print("\nDDS filtering:")
print(f"Rows before r2 filter : {n_dds_before}")
print(f"Rows after r2 > 0     : {n_dds_after}")
print(f"Rows removed          : {n_dds_before - n_dds_after}")

# ---------------------------
# Helpers
# ---------------------------
def find_first_existing(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a column for {label}. Tried: {candidates}")

def dds_model(t, A1, gamma1, f1, phi1, A2, gamma2, f2, phi2):
    return (
        A1 * np.exp(-gamma1 * t) * np.sin(2 * np.pi * f1 * t + phi1)
        + A2 * np.exp(-gamma2 * t) * np.sin(2 * np.pi * f2 * t + phi2)
    )

# ERP columns
sub_col_erp   = find_first_existing(erp, ["subject", "subj", "participant"], "ERP subject")
cond_col_erp  = find_first_existing(erp, ["cond", "condition"], "ERP condition")
trial_col_erp = find_first_existing(erp, ["trial"], "ERP trial")
roi_col_erp   = find_first_existing(erp, ["roi"], "ERP roi")
time_col_erp  = find_first_existing(erp, ["time_ms", "time", "t", "times"], "ERP time")
val_col_erp   = find_first_existing(erp, ["value", "signal", "erp", "amplitude"], "ERP signal")

# DDS columns
sub_col_dds   = find_first_existing(dds, ["subject", "subj", "participant"], "DDS subject")
roi_col_dds   = find_first_existing(dds, ["roi"], "DDS roi")
trial_col_dds = find_first_existing(dds, ["trial"], "DDS trial")

# cond puede no estar en DDS
cond_col_dds = None
for c in ["cond", "condition"]:
    if c in dds.columns:
        cond_col_dds = c
        break

# parámetros DDS
A1_col     = find_first_existing(dds, ["A1", "a1"], "DDS A1")
gamma1_col = find_first_existing(dds, ["gamma1", "g1"], "DDS gamma1")
f1_col     = find_first_existing(dds, ["f1"], "DDS f1")
phi1_col   = find_first_existing(dds, ["phi1", "phase1"], "DDS phi1")
A2_col     = find_first_existing(dds, ["A2", "a2"], "DDS A2")
gamma2_col = find_first_existing(dds, ["gamma2", "g2"], "DDS gamma2")
f2_col     = find_first_existing(dds, ["f2"], "DDS f2")
phi2_col   = find_first_existing(dds, ["phi2", "phase2"], "DDS phi2")

print("\nDetected mapping:")
print(f"ERP -> subject={sub_col_erp}, cond={cond_col_erp}, trial={trial_col_erp}, roi={roi_col_erp}, time={time_col_erp}, value={val_col_erp}")
print(f"DDS -> subject={sub_col_dds}, cond={cond_col_dds}, trial={trial_col_dds}, roi={roi_col_dds}")

# convertir tiempo a segundos para el modelo DDS
residual_rows = []
n_total = 0
n_ok = 0
n_missing = 0
n_multi = 0
n_error = 0

group_cols = [sub_col_erp, cond_col_erp, trial_col_erp, roi_col_erp]

for keys, g in erp.groupby(group_cols, sort=False):
    n_total += 1

    sub_val, cond_val, trial_val, roi_val = keys

    mask = (
        (dds[sub_col_dds] == sub_val) &
        (dds[trial_col_dds] == trial_val) &
        (dds[roi_col_dds] == roi_val)
    )

    if cond_col_dds is not None:
        mask = mask & (dds[cond_col_dds] == cond_val)

    row = dds.loc[mask]

    if len(row) == 0:
        n_missing += 1
        continue

    if len(row) > 1:
        n_multi += 1
        row = row.iloc[[0]]

    row = row.iloc[0]

    try:
        t_ms = g[time_col_erp].to_numpy(dtype=float)
        t = t_ms / 1000.0
        y = g[val_col_erp].to_numpy(dtype=float)

        y_hat = dds_model(
            t,
            float(row[A1_col]), float(row[gamma1_col]), float(row[f1_col]), float(row[phi1_col]),
            float(row[A2_col]), float(row[gamma2_col]), float(row[f2_col]), float(row[phi2_col])
        )

        residual = y - y_hat

        tmp = g.copy()
        tmp["dds_fit"] = y_hat
        tmp["residual"] = residual
        tmp["dds_r2"] = float(row["r2"])

        residual_rows.append(tmp)
        n_ok += 1

    except Exception as e:
        n_error += 1
        print(f"[warn] failed for subject={sub_val}, cond={cond_val}, trial={trial_val}, roi={roi_val}: {e}")

if len(residual_rows) == 0:
    raise RuntimeError("No residual rows were generated. Check the key mapping between ERP and DDS tables.")

residual_df = pd.concat(residual_rows, ignore_index=True)

out_file = out_dir / "trial_roi_timeseries_residual_r2pos.csv"
residual_df.to_csv(out_file, index=False)

print("\nDone.")
print(f"Groups total (ERP)        : {n_total}")
print(f"Matched groups kept       : {n_ok}")
print(f"Missing after r2 > 0      : {n_missing}")
print(f"Multi matches             : {n_multi}")
print(f"Errors                    : {n_error}")
print("Residual file saved to    :", out_file)

# pequeño resumen de cuántos trials quedaron
trial_keep = residual_df[[sub_col_erp, cond_col_erp, trial_col_erp, roi_col_erp]].drop_duplicates()
print(f"Unique kept trial×ROI rows: {len(trial_keep)}")
