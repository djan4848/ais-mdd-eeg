#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd

# =========================================================
# GROUP MAPPING
# =========================================================
subjects_CTL = [
    'P1/HYL_01_01','P2/HYL_02_01','P3/HYL_03_01',
    'P9/HYL_09_01','P10/HYL_10_01','P11/HYL_11_01',
    'P12/HYL_12_01','P13/HYL_13_01','P14/HYL_14_01',
    'P15/HYL_15_01','P16/HYL_16_01','P17/HYL_17_01',
    'P18/HYL_18_01','P22/HYL_22_01','P26/HYL_26_01',
    'P27/HYL_27_01','P30/HYL_30_01','P31/HYL_31_01',
    'P33/HYL_33_01','P38/HYL_38_01','P49/HYL_49_01',
    'P50/HYL_50_01'
]

subjects_DEP = [
    'P6/HYL_06_01','P7/HYL_07_01','P8/HYL_08_01',
    'P20/HYL_20_01','P21/HYL_21_01','P23/HYL_23_01',
    'P24/HYL_24_01','P25/HYL_25_01','P28/HYL_28_01',
    'P29/HYL_29_01','P32/HYL_32_01','P34/HYL_34_01',
    'P35/HYL_35_01','P36/HYL_36_01','P37/HYL_37_01',
    'P39/HYL_39_01','P40/HYL_40_01','P41/HYL_41_01',
    'P42/HYL_42_01','P43/HYL_43_01','P44/HYL_44_01',
    'P45/HYL_45_01','P46/HYL_46_01','P47/HYL_47_01',
    'P48/HYL_48_01'
]

CTL_IDS = {s.split("/")[0] for s in subjects_CTL}
DEP_IDS = {s.split("/")[0] for s in subjects_DEP}


def assign_group(subject: str) -> str:
    if subject in CTL_IDS:
        return "CTL"
    elif subject in DEP_IDS:
        return "DEP"
    else:
        return "UNKNOWN"


def add_group_column(input_csv: Path, output_csv: Path | None = None) -> None:
    df = pd.read_csv(input_csv)

    if "subject" not in df.columns:
        raise KeyError(f"'subject' column not found in {input_csv}")

    df["group"] = df["subject"].astype(str).apply(assign_group)

    print("\n=== group counts (rows) ===")
    print(df["group"].value_counts(dropna=False))

    print("\n=== unique subjects by group ===")
    print(df.groupby("group")["subject"].nunique())

    unknown_subjects = sorted(df.loc[df["group"] == "UNKNOWN", "subject"].unique().tolist())
    if unknown_subjects:
        print("\n[warn] UNKNOWN subjects detected:")
        for s in unknown_subjects:
            print("  ", s)

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_with_group.csv")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 add_group_to_table.py <input_csv> [output_csv]")
        sys.exit(1)

    input_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    add_group_column(input_csv, output_csv)
