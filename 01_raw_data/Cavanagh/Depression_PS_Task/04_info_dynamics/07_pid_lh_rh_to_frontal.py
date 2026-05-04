#!/usr/bin/env python3

import numpy as np
import pandas as pd
import mne
from sklearn.metrics import mutual_info_score

from dds_base.io.paths import (
    hayling_epo_files,
    ROIS,
    EXCLUDE_SUBJECTS,
)

NBINS = 4
LAG = 4

COND_MAP = {
    "ASOC": "INIT",
    "NOASOC": "INHIB",
}

results = []

files = [f for f in hayling_epo_files() if f.parent.name not in EXCLUDE_SUBJECTS]

for f in files:

    subj = f.parent.name
    epochs = mne.read_epochs(f, preload=True, verbose="ERROR")

    for raw_cond, paper_cond in COND_MAP.items():

        if raw_cond not in epochs.event_id:
            continue

        ep = epochs[raw_cond]

        roi_data = {}

        for roi, chans in ROIS.items():
            actual = [c for c in chans if c in ep.ch_names]
            if not actual:
                continue
            roi_data[roi] = ep.copy().pick(actual).get_data().mean(axis=1)

        if not all(k in roi_data for k in ["lh", "rh", "frontal"]):
            continue

        for trial in range(len(ep)):

            lh = roi_data["lh"][trial]
            rh = roi_data["rh"][trial]
            fr = roi_data["frontal"][trial]

            if len(lh) <= LAG:
                continue

            lh = lh[:-LAG]
            rh = rh[:-LAG]
            fr = fr[LAG:]

            try:
                lh = pd.qcut(lh, NBINS, labels=False, duplicates="drop")
                rh = pd.qcut(rh, NBINS, labels=False, duplicates="drop")
                fr = pd.qcut(fr, NBINS, labels=False, duplicates="drop")
            except Exception:
                continue

            lh = np.asarray(lh, dtype=int)
            rh = np.asarray(rh, dtype=int)
            fr = np.asarray(fr, dtype=int)

            I_lh = mutual_info_score(lh, fr)
            I_rh = mutual_info_score(rh, fr)

            joint = lh * NBINS + rh
            I_joint = mutual_info_score(joint, fr)

            R = min(I_lh, I_rh)
            U_lh = I_lh - R
            U_rh = I_rh - R
            S = I_joint - U_lh - U_rh - R

            results.append({
                "subject": subj,
                "cond": paper_cond,
                "trial": trial,
                "trial_uid": f"{subj}_{paper_cond}_{trial}",
                "component": "N450",
                "lag_samples": LAG,
                "lag_ms": LAG * 4.0,
                "bins": NBINS,
                "redundancy": R,
                "unique_lh": U_lh,
                "unique_rh": U_rh,
                "synergy": S
            })

pid = pd.DataFrame(results)

summary = pid.groupby("cond").mean(numeric_only=True)
print(summary)

pid.to_csv(
    "derivatives/te_n450/pid_lh_rh_frontal.csv",
    index=False
)

print("saved PID results")
print("excluded subjects:", sorted(EXCLUDE_SUBJECTS))
