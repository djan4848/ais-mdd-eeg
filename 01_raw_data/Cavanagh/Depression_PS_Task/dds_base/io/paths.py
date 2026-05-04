from pathlib import Path
import yaml

# ---------------------------------------------------------------------
# Project root
# Assumes this file lives in: <repo>/dds_base/io/paths.py
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------
CFG_PATH = ROOT / "config" / "analysis.yaml"
if not CFG_PATH.exists():
    raise FileNotFoundError(f"Config file not found: {CFG_PATH}")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

if "paths" not in CFG:
    raise KeyError(f"'paths' key not found in config file: {CFG_PATH}")


def p(key: str) -> Path:
    """Return a project path defined in config/analysis.yaml."""
    if key not in CFG["paths"]:
        raise KeyError(f"Path key '{key}' not found in config/analysis.yaml")
    return ROOT / CFG["paths"][key]


# ---------------------------------------------------------------------
# Raw/preprocessed EEG inputs
# ---------------------------------------------------------------------
# Adjust this path if your actual project layout differs
HAYLING_EEG_ROOT = ROOT / "../EEG"

# Final epoched files
HAYLING_EPO_GLOB = "P*/HYL_*_90_Hz-2-ar-den-epo.fif"


def hayling_epo_files():
    """Return sorted list of final Hayling epoch files."""
    return sorted(HAYLING_EEG_ROOT.glob(HAYLING_EPO_GLOB))


# ---------------------------------------------------------------------
# Derivatives / outputs
# ---------------------------------------------------------------------
DERIV_ROOT = ROOT / "derivatives"
DERIV_ROOT.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------
# Canonical ROI definitions for the CURRENT frontocentral DDS-Hayling pipeline
# Conservative choice: keep ROIs close to the frontocentral inhibitory phenomenon
# ---------------------------------------------------------------------
ROIS = {
    "frontal": ["F3", "F4", "AF3", "AF4", "Fp1", "Fp2", "FC3", "FC4"],
    "cacc": ["FC2", "AFz", "F2"],
    "lh": ["F3", "F5", "FC3", "FC5"],
    "rh": ["F4", "F6", "FC4", "FC6"],
}

ROI_DISPLAY_NAMES = {
    "frontal": "Frontal",
    "cacc": "cACC",
    "lh": "LH",
    "rh": "RH",
}

# ---------------------------------------------------------------------
# Subject exclusions
# Keep centralized so all scripts use the same exclusion policy
# ---------------------------------------------------------------------
EXCLUDE_SUBJECTS = {"P4", "P5", "P19"}


def get_roi_channels(roi_name: str, info_ch_names):
    """Return available channels for a ROI, preserving ROI order."""
    if roi_name not in ROIS:
        raise KeyError(f"ROI '{roi_name}' not found. Available: {list(ROIS.keys())}")
    return [ch for ch in ROIS[roi_name] if ch in info_ch_names]


def validate_rois_against_channels(info_ch_names):
    """
    Validate ROI coverage against a channel list.
    Returns a dict with available and missing channels per ROI.
    """
    report = {}
    for roi_name, roi_channels in ROIS.items():
        available = [ch for ch in roi_channels if ch in info_ch_names]
        missing = [ch for ch in roi_channels if ch not in info_ch_names]
        report[roi_name] = {
            "available": available,
            "missing": missing,
            "n_available": len(available),
            "n_expected": len(roi_channels),
        }
    return report
