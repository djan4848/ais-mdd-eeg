# Active Information Storage as a Pre-Feedback Neural Marker in MDD

Analysis pipeline for: *Active Information Storage at FCz as an anticipatory neural marker in major depressive disorder* (Jan et al., 2026).

## Primary finding

Active Information Storage (AIS) at electrode FCz in the 200ms window immediately preceding reward feedback is significantly reduced in MDD:

| Comparison | d | p (FDR) | N |
|---|---|---|---|
| CTL > MDD (Cavanagh EEG, primary) | **0.81** | 0.0002 | 87 CTL, 23 MDD |
| CTL > MDD (Cavanagh resting-state) | 0.70 | 0.003 | 90 CTL, 23 MDD |
| CTL > MDD (TDBRAIN resting-state) | 2.02 | <0.001 | large N |
| MDD_current > MDD_past (neural scar) | 0.97 | 0.034 | 11 cur, 12 past |

Effect is specific to blank anticipatory periods in outcome-contingent contexts (null in resting-state task-irrelevant conditions, null at RewP window +200–+400ms).

## Repository structure

```
ais_estimator.py              # Canonical AIS estimator (import this)
analysis_manuscript_figure.py # Main figure (4 panels)
analysis_forensic_audit.py    # Risk audit for primary dataset
analysis_forensic_audit_all_datasets.py  # Audit for TDBRAIN, MODMA, ds003478
analysis_cavanagh_rest_ais.py # Resting-state replication (ds003478)
analysis_ais_rest_mdd_ctl.py  # TDBRAIN resting-state analysis
analysis_modma_rest_ais.py    # MODMA replication (null, paradigm mismatch)
analysis_meg_rest_ais.py      # MEG resting-state
analysis_te_followup.py       # Transfer entropy follow-up
analysis_b_te_f1_fcz.py       # TE F1→FCz (shadow of AIS_pre)
analysis_cross_diagnostic_ais # Cross-diagnostic comparison

01_raw_data/Cavanagh/Depression_PS_Task/
  01_preprocessing/           # EEGLAB→MNE conversion
  06_erp_it_pipeline/         # Scripts 12–17: AIS extraction and replication
  derivatives/erp_it_cavanagh/  # Computed results (CSV)
```

## AIS estimator

`ais_estimator.py` provides three functions:

```python
from ais_estimator import safe_ais, compute_ais_windowed, compute_ais_rest

# Single window (1-D array of EEG samples)
ais = safe_ais(signal, lag=1, n_bins=4)

# Epoched task EEG (n_trials × n_times), window in seconds
ais = compute_ais_windowed(epoch_data, times, tmin=-0.2, tmax=0.0)

# Resting-state continuous signal
ais = compute_ais_rest(signal, sfreq=500, window_sec=2.0)
```

The estimator uses equiquantile (percentile) binning, making it **amplitude-invariant** by construction. Validated against KSG estimator: r=0.962 (Script 13).

**Canonical parameters** (primary finding): FCz electrode, lag=1, n_bins=4, window −200 to 0ms pre-feedback.

## Datasets

Raw EEG data are not included. Access via:

| Dataset | Source | DOI |
|---|---|---|
| Cavanagh PST EEG | OpenNeuro ds005356 | https://openneuro.org/datasets/ds005356 |
| Cavanagh resting-state | OpenNeuro ds003478 | https://openneuro.org/datasets/ds003478 |
| Cavanagh MEG | OpenNeuro ds003474 | https://openneuro.org/datasets/ds003474 |
| TDBRAIN | van Dijk et al. (2021) | https://doi.org/10.1038/s41597-022-01409-z |
| MODMA | Cai et al. (2022) | https://doi.org/10.1038/s41597-022-01173-0 |

## Preprocessing

Original Cavanagh PST data were preprocessed using the **APPLE pipeline** (Cavanagh, 2013) in EEGLAB 12.0.2.1b:
- Bad channels: FASTER (z>3 on correlation, variance, Hurst) + spherical interpolation
- ICA: extended infomax (`runica`), dimensions = n_channels − n_bad_channels per subject
- Ocular component rejection: VEOG correlation |z|>3 and frontopolar template |z|>3
- Epoch rejection: FASTER (z>3 on mean deviation, variance, max amplitude)
- Reference: linked mastoids (M1+M2); 500 Hz; epochs −2 to +2s

## Forensic audit (pre-submission)

Seven a priori risks tested — all pass:

| Risk | Result |
|---|---|
| Baseline leakage | r(baseline, no-baseline) = 1.000 |
| Channel selection | All 4 FCz neighbors significant, same direction |
| Binning amplification | r(variance, AIS) = +0.065, p=0.50 |
| Window selection | Primary (−200,0ms) is NOT maximum (max at −300,−100ms) |
| TDBRAIN amplitude | No between-group amplitude difference (d=−0.041, p=0.756) |
| Preprocessing exclusion | All 3 excluded subjects are CTL (0% MDD excluded) |
| Estimator reliability | Split-half r=0.969; Spearman-Brown r=0.984 |

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Citation

Jan, D. (2026). *Active Information Storage at FCz as a pre-feedback anticipatory neural marker in major depressive disorder*. Preprint.

## License

MIT — see [LICENSE](LICENSE).
