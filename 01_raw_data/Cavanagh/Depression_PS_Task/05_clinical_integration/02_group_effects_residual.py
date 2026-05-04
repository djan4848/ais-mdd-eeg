import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# --------------------------------------------------
# Subject groups
# --------------------------------------------------

subjects_CTL = [
'P1','P2','P3','P9','P10','P11','P12','P13','P14','P15',
'P16','P17','P18','P22','P26','P27','P30','P31','P33',
'P38','P49','P50'
]

subjects_DEP = [
'P6','P7','P8','P20','P21','P23','P24','P25','P28','P29',
'P32','P34','P35','P36','P37','P39','P40','P41','P42',
'P43','P44','P45','P46','P47','P48'
]

CTL = set(subjects_CTL)
DEP = set(subjects_DEP)

# --------------------------------------------------
# Load AIS residual
# --------------------------------------------------

df = pd.read_csv(
"derivatives/ais_n450_residual/ais_n450_residual_results.csv"
)

print("\n===== GROUP EFFECTS (AIS residual) =====")

for roi in ["frontal","cacc"]:

    sub = df[df.roi == roi]

    pivot = sub.pivot_table(
        index="subject",
        columns="cond",
        values="ais_bits",
        aggfunc="mean"
    )

    pivot["cost"] = pivot["INHIB"] - pivot["INIT"]

    ctl_vals = pivot.loc[pivot.index.isin(CTL),"cost"]
    dep_vals = pivot.loc[pivot.index.isin(DEP),"cost"]

    t,p = ttest_ind(ctl_vals,dep_vals)

    print("\nROI:",roi)
    print("CTL mean cost:",ctl_vals.mean())
    print("DEP mean cost:",dep_vals.mean())
    print("t =",t)
    print("p =",p)
