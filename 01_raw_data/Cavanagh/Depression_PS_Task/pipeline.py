#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline.py

Executes the entire DDS-Hayling pipeline sequentially.
"""

import os
import sys
import subprocess
from pathlib import Path

# The root directory of the project
ROOT = Path(__file__).resolve().parent

# Ensure we are in the project root
os.chdir(ROOT)

# List of scripts to execute in order
PIPELINE_SCRIPTS = [
    #ROOT / "01_preprocessing" / "01_make_evokeds.py",
    #ROOT /  "01_preprocessing" / "02_extract_trial_roi_timeseries.py",
    #ROOT / "02_dds_modeling" / "03_dds_peak_aligned_trial_by_tiral.py",
    #ROOT / "04_info_dynamics" / "04_extract_ais_n450_only.py",
    ROOT / "05_clinical_integration" / "05_mixed_effects_n450_clinical.py",
    #ROOT / "06_visualization" / "Figure1_ERP_N450_explanation.py",
    #ROOT / "06_visualization" / "Figure2_DDS_N450_explanation.py",
    #ROOT / "06_visualization" / "Figure3_Clinical_explanation.py",
]

def main():
    print("========================================")
    print(" DDS-HAYLING PIPELINE EXECUTION STARTED")
    print("========================================\n")

    for script in PIPELINE_SCRIPTS:
        if not script.exists():
            print(f"[ERROR] Script not found: {script.relative_to(ROOT)}")
            sys.exit(1)

        print("----------------------------------------")
        print(f"[{script.parent.name}] Running: {script.name}")
        print("----------------------------------------")
        
        try:
            # We use subprocess.run to execute the script and stream its output
            # We set PYTHONPATH so imports like `dds_base.io` resolve correctly
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + (f":{env['PYTHONPATH']}" if "PYTHONPATH" in env else "")

            result = subprocess.run(
                [sys.executable, str(script)],
                env=env,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Pipeline failed at step: {script.relative_to(ROOT)}")
            print(f"Exit code: {e.returncode}")
            sys.exit(e.returncode)

    print("\n========================================")
    print(" DDS-HAYLING PIPELINE COMPLETED SUCCESSFULLY")
    print("========================================")

if __name__ == "__main__":
    main()
