import pandas as pd

# ===============================
# cargar resultados previos
# ===============================

ais = pd.read_csv("derivatives/ais_n450/ais_n450_results.csv")
te  = pd.read_csv("derivatives/te_n450/te_n450_results.csv")
pid = pd.read_csv("derivatives/te_n450/pid_lh_rh_frontal.csv")

# ===============================
# AIS frontal
# ===============================

ais_frontal = (
    ais[ais["roi"]=="frontal"]
    .groupby("cond")["ais_bits"]
    .mean()
)

# ===============================
# TE hacia frontal
# ===============================

te_frontal = (
    te[te["target_roi"]=="frontal"]
    .groupby("cond")["te_bits"]
    .mean()
)

# ===============================
# PID sinergia
# ===============================

pid_syn = (
    pid.groupby("cond")["synergy"]
    .mean()
)

# ===============================
# combinar
# ===============================

summary = pd.DataFrame({

    "AIS_frontal": ais_frontal,
    "TE_to_frontal": te_frontal,
    "PID_synergy": pid_syn

})

summary["integration_index"] = (
    summary["TE_to_frontal"] + summary["PID_synergy"]
) / summary["AIS_frontal"]

print("\n=== INFORMATION ROLE ANALYSIS ===\n")
print(summary)

print("\nInterpretation:")
print("integration_index < 0.3   → generator-like")
print("0.3 – 0.7                 → mixed")
print("> 0.7                     → integrator")

