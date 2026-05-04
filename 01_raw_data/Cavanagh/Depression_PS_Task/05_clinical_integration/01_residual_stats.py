import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
from pathlib import Path

ROOT = Path("derivatives")

ais_file = ROOT / "ais_n450_residual_r2pos/ais_n450_residual_r2pos_results.csv"
te_file = ROOT / "te_n450_residual_r2pos/te_n450_residual_r2pos_results.csv"
pid_file = ROOT / "te_n450_residual_r2pos/pid_lh_rh_frontal_residual_r2pos.csv"

print("Loading data...")

ais = pd.read_csv(ais_file)
te = pd.read_csv(te_file)
pid = pd.read_csv(pid_file)

# --------------------------------------------------
# AIS STATS
# --------------------------------------------------

print("\n===== AIS residual =====")

for roi in ["frontal", "cacc"]:

    df = ais[ais.roi == roi]

    pivot = df.pivot_table(
        index="subject",
        columns="cond",
        values="ais_bits",
        aggfunc="mean"
    )

    init = pivot["INIT"]
    inhib = pivot["INHIB"]

    t, p = ttest_rel(inhib, init)

    print(f"\nROI: {roi}")
    print("mean INIT :", init.mean())
    print("mean INHIB:", inhib.mean())
    print("t =", t)
    print("p =", p)


# --------------------------------------------------
# TE STATS
# --------------------------------------------------

print("\n===== TE residual =====")

for direction in ["cacc->frontal", "frontal->cacc"]:

    df = te[te.direction == direction]

    pivot = df.pivot_table(
        index="subject",
        columns="cond",
        values="te_bits",
        aggfunc="mean"
    )

    init = pivot["INIT"]
    inhib = pivot["INHIB"]

    t, p = ttest_rel(inhib, init)

    print(f"\nDirection: {direction}")
    print("mean INIT :", init.mean())
    print("mean INHIB:", inhib.mean())
    print("t =", t)
    print("p =", p)


# --------------------------------------------------
# PID STATS
# --------------------------------------------------

print("\n===== PID residual =====")

for metric in ["redundancy", "synergy"]:

    pivot = pid.pivot_table(
        index="subject",
        columns="cond",
        values=metric,
        aggfunc="mean"
    )

    init = pivot["INIT"]
    inhib = pivot["INHIB"]

    t, p = ttest_rel(inhib, init)

    print(f"\nMetric: {metric}")
    print("mean INIT :", init.mean())
    print("mean INHIB:", inhib.mean())
    print("t =", t)
    print("p =", p)
