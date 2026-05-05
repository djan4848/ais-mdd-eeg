"""
ais_estimator.py
----------------
Canonical implementation of Active Information Storage (AIS) for EEG signals,
as used in: "Active Information Storage at FCz as a pre-feedback anticipatory
neural marker in major depressive disorder" (Jan et al., 2026).

Primary finding: CTL > MDD at FCz, window −200 to 0ms pre-feedback,
d=0.81, p=0.0002 (N=87 CTL, N=23 MDD; Cavanagh PST dataset).

Usage
-----
    from ais_estimator import safe_ais, compute_ais_windowed, compute_ais_rest

    # Single time series
    ais_val = safe_ais(signal, lag=1, n_bins=4)

    # Epoched EEG (shape: n_trials × n_times), sliding window
    subject_ais = compute_ais_windowed(epoch_data, sfreq=500,
                                       tmin=-0.2, tmax=0.0, lag=1, n_bins=4)

    # Resting-state (continuous signal), 2-second windows
    rest_ais = compute_ais_rest(raw_signal, sfreq=500,
                                window_sec=2.0, lag=1, n_bins=4)
"""

import numpy as np
from typing import Optional


def safe_ais(x: np.ndarray, lag: int = 1, n_bins: int = 4) -> float:
    """
    Compute Active Information Storage via percentile-binned mutual information.

    AIS(lag) = MI( X_t ; X_{t-lag} )

    Estimator is amplitude-invariant by construction (equiquantile binning).
    Returns NaN instead of raising on degenerate input.

    Parameters
    ----------
    x       : 1-D array of EEG samples (already in the target window)
    lag     : autoregressive lag in samples (default 1)
    n_bins  : number of equiprobable bins (default 4)

    Returns
    -------
    AIS value in bits, or np.nan if input is degenerate
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * lag + 10:
        return np.nan
    if np.std(x) < 1e-12:
        return np.nan
    try:
        edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
        if len(edges) < 3:
            return np.nan
        bins = np.digitize(x, edges[1:-1])
        x_t = bins[lag:]
        x_lag = bins[:-lag]
        joint = np.zeros((n_bins, n_bins))
        for a, b in zip(x_t, x_lag):
            joint[a - 1, b - 1] += 1
        joint /= joint.sum() + 1e-10
        px_t = joint.sum(axis=1)
        px_lag = joint.sum(axis=0)
        mi = sum(
            joint[i, j] * np.log2(joint[i, j] / (px_t[i] * px_lag[j]))
            for i in range(n_bins)
            for j in range(n_bins)
            if joint[i, j] > 0 and px_t[i] > 0 and px_lag[j] > 0
        )
        return float(mi) if np.isfinite(mi) else np.nan
    except Exception:
        return np.nan


def compute_ais_windowed(
    epoch_data: np.ndarray,
    times: np.ndarray,
    tmin: float = -0.2,
    tmax: float = 0.0,
    lag: int = 1,
    n_bins: int = 4,
    min_trials: int = 5,
) -> float:
    """
    Compute subject-level AIS from an epoched array by averaging over trials.

    Parameters
    ----------
    epoch_data  : array of shape (n_trials, n_times) — single channel
    times       : 1-D array of time points in seconds (aligned with epoch_data axis 1)
    tmin, tmax  : window boundaries in seconds (default: −200 to 0ms)
    lag         : AIS lag in samples
    n_bins      : number of equiprobable bins
    min_trials  : minimum valid trials required (returns NaN if below)

    Returns
    -------
    Mean AIS across trials (bits), or np.nan
    """
    mask = (times >= tmin) & (times < tmax)
    if mask.sum() < 2 * lag + 10:
        return np.nan
    vals = [safe_ais(epoch_data[i, mask], lag=lag, n_bins=n_bins)
            for i in range(epoch_data.shape[0])]
    valid = [v for v in vals if np.isfinite(v)]
    if len(valid) < min_trials:
        return np.nan
    return float(np.mean(valid))


def compute_ais_rest(
    signal: np.ndarray,
    sfreq: float,
    window_sec: float = 2.0,
    lag: int = 1,
    n_bins: int = 4,
    min_windows: int = 10,
) -> float:
    """
    Compute AIS from a continuous resting-state signal using non-overlapping windows.

    Parameters
    ----------
    signal      : 1-D array of EEG samples
    sfreq       : sampling frequency in Hz
    window_sec  : window length in seconds (default 2.0)
    lag         : AIS lag in samples
    n_bins      : number of equiprobable bins
    min_windows : minimum valid windows required (returns NaN if below)

    Returns
    -------
    Mean AIS across windows (bits), or np.nan
    """
    win_samples = int(window_sec * sfreq)
    n_windows = len(signal) // win_samples
    if n_windows < min_windows:
        return np.nan
    vals = [
        safe_ais(signal[i * win_samples:(i + 1) * win_samples], lag=lag, n_bins=n_bins)
        for i in range(n_windows)
    ]
    valid = [v for v in vals if np.isfinite(v)]
    if len(valid) < min_windows:
        return np.nan
    return float(np.mean(valid))


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(42)
    sig = rng.standard_normal(1000)
    print(f"safe_ais (lag=1, bins=4): {safe_ais(sig):.4f} bits")
    print(f"compute_ais_rest (2s @ 500Hz): {compute_ais_rest(sig, sfreq=500):.4f} bits")

    # Verify amplitude invariance
    sig_scaled = sig * 1000
    assert abs(safe_ais(sig) - safe_ais(sig_scaled)) < 1e-10, "Not amplitude-invariant!"
    print("Amplitude invariance: OK")
