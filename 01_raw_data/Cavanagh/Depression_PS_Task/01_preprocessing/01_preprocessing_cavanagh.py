#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_preprocessing_cavanagh.py

Adapts Cavanagh depression dataset (.mat files) into MNE Epochs format
compatible with the DDS-Hayling pipeline.
"""

import os
import numpy as np
import mne
import scipy.io as sio
from pathlib import Path
import sys

# Paths setup (adapt to the directory structure)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
DERIV_ROOT = ROOT / "derivatives" / "epochs"
DERIV_ROOT.mkdir(exist_ok=True, parents=True)

# Important frontal channels for the DDS ROI
EXPECTED_FRONTAL_ROI = ["Fz", "FCz", "F3", "F4"]

# Event mapping
# Based on original manuscript scripts (94 = Correct/Reward, 104 = Incorrect/Loss)
EVENT_ID = {
    "Reward": 94,
    "Loss": 104
}

def main():
    mat_files = sorted(DATA_DIR.glob("*.mat"))
    if not mat_files:
        print(f"[ERROR] No .mat files found in {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(mat_files)} files to process.")

    for mat_path in mat_files:
        subject_id = mat_path.stem
        out_file = DERIV_ROOT / f"sub-{subject_id}_task-ps_epo.fif"
        
        if out_file.exists():
            print(f"[{subject_id}] Already processed. Skipping...")
            continue
            
        print(f"\n--- Processing {subject_id} ---")
        
        try:
            mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        except Exception as e:
            print(f"[{subject_id}] Error loading .mat file: {e}")
            continue

        if 'EEG' not in mat:
            print(f"[{subject_id}] 'EEG' key not found in .mat")
            continue
            
        EEG = mat['EEG']
        
        # 1. Extract Data
        if not hasattr(EEG, 'data'):
            print(f"[{subject_id}] No 'data' attribute in EEG struct.")
            continue
        data = EEG.data  # Expected shape (channels, times)
        
        # NaN / Inf Check
        if np.isnan(data).any():
            print(f"[{subject_id}] WARNING: NaN values detected in EEG data! Interpolating or continuing.")
            data = np.nan_to_num(data)
        if np.isinf(data).any():
            print(f"[{subject_id}] WARNING: Inf values detected in EEG data! Replacing.")
            data = np.nan_to_num(data)
            
        # 2. Extract Channel Info
        if hasattr(EEG, 'chanlocs') and hasattr(EEG.chanlocs, '__iter__'):
            ch_names = [c.labels for c in EEG.chanlocs if hasattr(c, 'labels')]
        else:
            print(f"[{subject_id}] Missing channel names.")
            continue
            
        # Upper/lower case standardization to match MNE 10-20
        ch_names = [ch.upper() if ch.lower() != 'fz' else 'Fz' for ch in ch_names]
        ch_names = [ch.replace('Z', 'z') if ch.endswith('Z') else ch for ch in ch_names]
        # Fp1/Fp2 formatting
        ch_names = [ch.replace('FP1', 'Fp1').replace('FP2', 'Fp2') for ch in ch_names]

        # Validation of Frontal ROI
        missing_roi = [c for c in EXPECTED_FRONTAL_ROI if c not in ch_names]
        if missing_roi:
            print(f"[{subject_id}] ERROR: Missing essential frontal ROI channels: {missing_roi}. Available: {ch_names[:5]}...")
            continue
            
        # 3. Extract Sampling Rate
        sfreq = getattr(EEG, 'srate', 500.0)
        
        # Report Reference (for awareness)
        ref = getattr(EEG, 'ref', 'unknown')
        print(f"[{subject_id}] File indicates reference: {ref}")
        
        # 4. MNE Structure Creation
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Set standard montage
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage, match_case=False, on_missing='ignore')
        except Exception as e:
            print(f"[{subject_id}] Montage issue: {e}")

        raw = mne.io.RawArray(data, info, verbose='ERROR')
        
        # 5. Extract Events and Epoching
        events = []
        if hasattr(EEG, 'event'):
            for ev in EEG.event:
                ev_type = getattr(ev, 'type', None)
                idx = getattr(ev, 'latency', 0)
                
                # Clean up event string format
                try:
                    if isinstance(ev_type, str):
                        code = int(ev_type)
                    else:
                        code = int(ev_type)
                except:
                    continue
                    
                if code in EVENT_ID.values():
                    events.append([int(idx), 0, code])
                    
        if not events:
            print(f"[{subject_id}] No matching events found for {EVENT_ID}.")
            continue
            
        events = np.array(events)
        
        # Epoch settings (Using -2 to 2 like manuscript script STEP1, or -1 to 2)
        tmin, tmax = -1.0, 2.0
        epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=tmin, tmax=tmax, 
                            baseline=(None, 0), preload=True, verbose='ERROR')
        
        print(f"[{subject_id}] Epoched {len(epochs)} trials.")
        
        # 6. Export to FIF
        epochs.save(out_file, overwrite=True, verbose='ERROR')
        print(f"[{subject_id}] Saved to {out_file.name}")

if __name__ == "__main__":
    main()
