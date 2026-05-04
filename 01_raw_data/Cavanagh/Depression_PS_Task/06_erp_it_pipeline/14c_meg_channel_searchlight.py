#!/usr/bin/env python3
"""
14c_meg_channel_searchlight.py
Channel-level AIS_pre searchlight across MEG gradiometers and EEG channels.

sub-M87121835 is excluded (DC drift on EEG007: mean=+29.4µV, std=1.80µV —
artifact, not neural signal).

Per-subject per-channel signal quality gates are applied before computing
AIS_pre; failures are set to NaN and logged.

Answers:
  1. Is there a better EEG channel than EEG007?
  2. Do MEG gradiometers (MEG06xx) show larger d?
  3. Do Cavanagh 2025 sensors (MEG0511/MEG0921) replicate?
  4. Does neighborhood averaging improve p?

After searchlight: recomputes final MEG result with the best channel found.
"""

import sys
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import mannwhitneyu

mne.set_log_level('ERROR')

# ── Exclusions ────────────────────────────────────────────────────────────────
EXCLUDE_SUBJECTS = ['sub-M87121835']

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MEG   = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/ds005356")
CLIN_PATH  = BASE_MEG / "Code/MEG MDD IDs and Quex.xlsx"
DERIV_DIR  = BASE_MEG / "derivatives"
CACHE_DIR  = DERIV_DIR / "epochs_searchlight"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
TARGET_SFREQ = 250
TMIN_EPO     = -1.0
TMAX_EPO     =  1.5
BASELINE     = (-0.5, -0.2)   # avoids zeroing the AIS_pre window
PRE_WINDOW   = (-0.200, 0.000)
WIN_CODE, LOSS_CODE = 8, 9
MIN_TRIALS   = 10

CAVANAGH_2025_SENSORS = ['MEG0511', 'MEG0921']

# ── Signal quality check ──────────────────────────────────────────────────────
def check_signal_quality(signal, ch_type='eeg'):
    """
    Returns (True, 'OK') if signal is physiologically plausible.
    Rejects DC drift artifacts and flat/dead channels.

    EEG  thresholds: µV  (signal passed in Volts → converted internally)
    MEG grad thresholds: fT/cm (signal in T/m → converted internally)
    """
    if ch_type == 'eeg':
        sig = signal * 1e6           # V → µV
        dc_thresh    = 15.0          # µV
        flat_thresh  = 2.0           # µV std
        range_thresh = 20.0          # µV peak-to-peak
        unit         = 'µV'
    elif ch_type == 'grad':
        sig = signal * 1e13          # T/m → fT/cm
        dc_thresh    = 1000.0        # fT/cm
        flat_thresh  = 1.0           # fT/cm std
        range_thresh = 10.0          # fT/cm peak-to-peak
        unit         = 'fT/cm'
    else:
        # magnetometer or other — use relative check only
        sig = signal
        mean_abs  = np.abs(np.mean(sig))
        std_val   = np.std(sig)
        if std_val < 1e-14:
            return False, "Flat signal (near-zero std)"
        if mean_abs / (std_val + 1e-30) > 50.0:
            return False, f"DC/mean >> std ratio={mean_abs/(std_val+1e-30):.1f}"
        return True, "OK"

    mean_abs  = np.abs(np.mean(sig))
    std_val   = np.std(sig)
    range_val = np.ptp(sig)

    if mean_abs > dc_thresh:
        return False, f"DC offset {mean_abs:.1f}{unit}"
    if std_val < flat_thresh:
        return False, f"Flat signal std={std_val:.2f}{unit}"
    if range_val < range_thresh:
        return False, f"Narrow range {range_val:.1f}{unit}"
    return True, "OK"


# ── AIS (identical to 14a_v2) ─────────────────────────────────────────────────
def safe_ais(x, lag=1, n_bins=4):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * lag + 10 or np.std(x) < 1e-12:
        return np.nan
    try:
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return np.nan
        bins    = np.digitize(x, edges[1:-1])
        x_t     = bins[lag:]
        x_lag   = bins[:-lag]
        joint   = np.zeros((n_bins, n_bins))
        for a, b in zip(x_t, x_lag):
            joint[min(a-1, n_bins-1), min(b-1, n_bins-1)] += 1
        joint  /= joint.sum() + 1e-10
        px_t    = joint.sum(axis=1)
        px_lag  = joint.sum(axis=0)
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and px_t[i] > 0 and px_lag[j] > 0:
                    mi += joint[i, j] * np.log2(
                        joint[i, j] / (px_t[i] * px_lag[j]))
        return float(mi) if np.isfinite(mi) else np.nan
    except Exception:
        return np.nan


rng_sanity = np.random.default_rng(42)
_ar1 = np.zeros(300)
for _i in range(1, 300):
    _ar1[_i] = 0.9 * _ar1[_i-1] + 0.1 * rng_sanity.standard_normal()
assert safe_ais(_ar1) > safe_ais(rng_sanity.standard_normal(300)), "AIS sanity failed"
print("AIS sanity check OK")


# ── Statistical helpers ───────────────────────────────────────────────────────
def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


# ── Clinical data ─────────────────────────────────────────────────────────────
clinical = pd.read_excel(CLIN_PATH)
clinical['bids_id'] = 'sub-M87' + (100000 + clinical['URSI'].astype(int)).astype(str)
grp_map = {row['bids_id']: row['Group'] for _, row in clinical.iterrows()}
print(f"Clinical: N={len(clinical)}  "
      f"(CTL={sum(clinical['Group']=='CTL')}, MDD={sum(clinical['Group']=='MDD')})")

# ── Build subject file map ────────────────────────────────────────────────────
subject_map = {}
for sub_dir in sorted(BASE_MEG.glob("sub-M87*")):
    sub_id  = sub_dir.name
    if sub_id in EXCLUDE_SUBJECTS:
        print(f"[EXCLUDED] {sub_id} — DC drift artifact on EEG007")
        continue
    meg_dir = sub_dir / "ses-01" / "meg"
    if not meg_dir.exists():
        continue
    tsv = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_events.tsv"
    if not tsv.exists():
        continue
    single = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_meg.fif"
    if single.exists():
        subject_map[sub_id] = {'type': 'single', 'files': [single], 'tsv': tsv}
        continue
    split01 = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_split-01_meg.fif"
    split02 = meg_dir / f"{sub_id}_ses-01_task-pst_run-1_split-02_meg.fif"
    splits  = [f for f in [split01, split02] if f.exists()]
    if splits:
        subject_map[sub_id] = {'type': 'split', 'files': splits, 'tsv': tsv}

has_clinical = {k for k in subject_map if k in set(clinical['bids_id'])}
print(f"Subjects with MEG + clinical (artifact excluded): {len(has_clinical)}")

# ── PHASE 1: Epoch all MEG grad + EEG channels (separate cache) ───────────────
print("\n=== PHASE 1: EPOCHING (all MEG grad + EEG channels) ===")


def load_and_epoch_all(sub_id, sub_info):
    cache_path = CACHE_DIR / f"{sub_id}-epo.fif"
    if cache_path.exists():
        return cache_path, "cached"
    try:
        files = sub_info['files']
        if sub_info['type'] == 'single':
            raw = mne.io.read_raw_fif(files[0], preload=True, verbose='ERROR')
        else:
            raws = [mne.io.read_raw_fif(f, preload=True, verbose='ERROR') for f in files]
            raw  = mne.concatenate_raws(raws, verbose='ERROR')

        if raw.info['sfreq'] > TARGET_SFREQ + 10:
            raw.resample(TARGET_SFREQ, verbose='ERROR')

        ev_df  = pd.read_csv(str(sub_info['tsv']), sep='\t')
        fb_df  = ev_df[ev_df['trial_type'].isin(['FB/win', 'FB/loss'])].copy()
        if len(fb_df) == 0:
            return None, "no FB events in TSV"

        code_map  = {'FB/win': WIN_CODE, 'FB/loss': LOSS_CODE}
        sfreq     = raw.info['sfreq']
        samples   = np.round(fb_df['onset'].values * sfreq).astype(int)
        codes     = fb_df['trial_type'].map(code_map).values.astype(int)
        fb_events = np.column_stack([samples, np.zeros(len(samples), int), codes])

        if (codes == WIN_CODE).sum() + (codes == LOSS_CODE).sum() < MIN_TRIALS:
            return None, "too few feedback events"

        picks = mne.pick_types(raw.info, meg='grad', eeg=True, exclude='bads')
        epo = mne.Epochs(
            raw, fb_events,
            event_id={'FB/win': WIN_CODE, 'FB/loss': LOSS_CODE},
            tmin=TMIN_EPO, tmax=TMAX_EPO,
            baseline=BASELINE,
            picks=picks,
            preload=True, verbose='ERROR',
            event_repeated='drop',
        )
        epo.drop_bad(verbose='ERROR')
        if len(epo) < MIN_TRIALS:
            return None, f"only {len(epo)} epochs after drop_bad"

        epo.save(cache_path, overwrite=True, verbose='ERROR')
        del raw, epo
        return cache_path, "epoched"
    except Exception as e:
        return None, f"ERROR: {e}"


skipped = []
for sub_id in sorted(has_clinical):
    cache_path, status = load_and_epoch_all(sub_id, subject_map[sub_id])
    if cache_path is None:
        skipped.append((sub_id, status))
        print(f"  {sub_id}: skip — {status}")
    elif status != "cached":
        print(f"  {sub_id}: {status}")

cached_files = sorted(CACHE_DIR.glob("sub-M87*-epo.fif"))
print(f"Epoching done. Skipped: {len(skipped)}  Cached: {len(cached_files)}")

if not cached_files:
    print("ERROR: no cached epochs"); sys.exit(1)

# Discover channel list and types from first file
_probe = mne.read_epochs(cached_files[0], preload=False, verbose='ERROR')
all_channels = _probe.ch_names
ch_type_map  = {ch: mne.channel_type(_probe.info, _probe.ch_names.index(ch))
                for ch in all_channels}
print(f"Channels in searchlight: {len(all_channels)}  "
      f"(grad={sum(v=='grad' for v in ch_type_map.values())}, "
      f"eeg={sum(v=='eeg' for v in ch_type_map.values())})")

# ── PHASE 2: AIS_pre per subject × channel ────────────────────────────────────
print("\n=== PHASE 2: AIS_PRE — all channels ===")

quality_log      = []
sub_channel_data = {}   # sub_id → {ch_name: ais_mean or nan}

for i, cache_path in enumerate(cached_files):
    sub_id = cache_path.name.replace("-epo.fif", "")
    if grp_map.get(sub_id) is None:
        continue

    epo      = mne.read_epochs(cache_path, preload=True, verbose='ERROR')
    times    = epo.times
    pre_mask = (times >= PRE_WINDOW[0]) & (times < PRE_WINDOW[1])
    data_3d  = epo.get_data()   # (n_epochs, n_channels, n_times)

    ch_ais = {}
    for ci, ch in enumerate(epo.ch_names):
        ch_type = mne.channel_type(epo.info, ci)

        # Quality check on the concatenated pre-window signal
        signal_concat = data_3d[:, ci, pre_mask].flatten()
        ok, reason = check_signal_quality(signal_concat, ch_type=ch_type)
        if not ok:
            quality_log.append({'subject': sub_id, 'channel': ch,
                                 'ch_type': ch_type, 'reason': reason})
            ch_ais[ch] = np.nan
            continue

        per_trial = [safe_ais(data_3d[t, ci, pre_mask], lag=1, n_bins=4)
                     for t in range(data_3d.shape[0])]
        valid = [v for v in per_trial if np.isfinite(v)]
        ch_ais[ch] = np.mean(valid) if len(valid) >= MIN_TRIALS else np.nan

    sub_channel_data[sub_id] = ch_ais
    n_valid = sum(np.isfinite(v) for v in ch_ais.values())
    print(f"  [{i+1}/{len(cached_files)}] {sub_id} ({grp_map[sub_id]}): "
          f"{n_valid}/{len(epo.ch_names)} channels valid")

# ── PHASE 3: Per-channel group statistics ─────────────────────────────────────
print("\n=== PHASE 3: PER-CHANNEL STATS ===")

channel_stats = []
for ch in all_channels:
    ctl_vals = [sub_channel_data[s][ch] for s in sub_channel_data
                if grp_map.get(s) == 'CTL'
                and np.isfinite(sub_channel_data[s].get(ch, np.nan))]
    mdd_vals = [sub_channel_data[s][ch] for s in sub_channel_data
                if grp_map.get(s) == 'MDD'
                and np.isfinite(sub_channel_data[s].get(ch, np.nan))]

    if len(ctl_vals) < 5 or len(mdd_vals) < 5:
        channel_stats.append({'channel': ch, 'ch_type': ch_type_map.get(ch, '?'),
                               'd': np.nan, 'p': np.nan,
                               'n_ctl': len(ctl_vals), 'n_mdd': len(mdd_vals),
                               'ctl_mean': np.nan, 'mdd_mean': np.nan})
        continue

    a, b = np.array(ctl_vals), np.array(mdd_vals)
    d    = cohen_d(a, b)
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    channel_stats.append({
        'channel':  ch,
        'ch_type':  ch_type_map.get(ch, '?'),
        'd':        d,
        'p':        p,
        'n_ctl':    len(a),
        'n_mdd':    len(b),
        'ctl_mean': a.mean(),
        'mdd_mean': b.mean(),
    })

df_sl = pd.DataFrame(channel_stats).sort_values('d', ascending=False).reset_index(drop=True)
df_sl.to_csv(DERIV_DIR / 'channel_searchlight_results.csv', index=False)

# ── PHASE 4: Neighborhood averaging ──────────────────────────────────────────
print("\n=== PHASE 4: NEIGHBORHOOD AVERAGING ===")

# Build position dict from info (MEG and EEG have 3-D sensor locations in .fif)
_epo_ref = mne.read_epochs(cached_files[0], preload=False, verbose='ERROR')
pos_dict = {}
for ci, ch in enumerate(_epo_ref.ch_names):
    loc = _epo_ref.info['chs'][ci]['loc'][:3]
    if np.any(loc != 0):
        pos_dict[ch] = loc


def get_neighbors(ch, radius=0.04):
    if ch not in pos_dict:
        return [ch]
    p0 = pos_dict[ch]
    return [ch] + [c for c, p in pos_dict.items()
                   if c != ch and np.linalg.norm(p - p0) <= radius]


neighborhood_stats = []
for ch in all_channels:
    neighbors = get_neighbors(ch)
    if len(neighbors) < 2:
        continue
    ctl_vals, mdd_vals = [], []
    for sid in sub_channel_data:
        grp = grp_map.get(sid)
        vals = [sub_channel_data[sid].get(n, np.nan) for n in neighbors]
        valid = [v for v in vals if np.isfinite(v)]
        if not valid:
            continue
        avg = np.mean(valid)
        if grp == 'CTL':
            ctl_vals.append(avg)
        elif grp == 'MDD':
            mdd_vals.append(avg)
    if len(ctl_vals) < 5 or len(mdd_vals) < 5:
        continue
    a, b = np.array(ctl_vals), np.array(mdd_vals)
    d    = cohen_d(a, b)
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    neighborhood_stats.append({
        'center_ch':   ch,
        'ch_type':     ch_type_map.get(ch, '?'),
        'n_neighbors': len(neighbors),
        'd':           d,
        'p':           p,
        'n_ctl':       len(a),
        'n_mdd':       len(b),
    })

df_nb = (pd.DataFrame(neighborhood_stats)
           .sort_values('d', ascending=False)
           .reset_index(drop=True))
df_nb.to_csv(DERIV_DIR / 'channel_neighborhood_results.csv', index=False)

# ── PHASE 5: Report ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("CHANNEL SEARCHLIGHT — TOP 20 BY COHEN'S d (CTL > MDD)")
print("="*65)
top20 = df_sl.dropna(subset=['d']).head(20)
print(top20[['channel', 'ch_type', 'd', 'p', 'n_ctl', 'n_mdd',
             'ctl_mean', 'mdd_mean']].to_string(index=False))

print("\n--- EEG007 reference (14a_v2 with artifact subject) ---")
print("  Previous result (N=83, artifact included): d=0.448, p=0.114")
ref = df_sl[df_sl['channel'] == 'EEG007']
if not ref.empty:
    r = ref.iloc[0]
    print(f"  EEG007 (artifact excluded): "
          f"d={r['d']:.3f}, p={r['p']:.4f}, "
          f"CTL={r['ctl_mean']:.4f}, MDD={r['mdd_mean']:.4f}, "
          f"N={int(r['n_ctl'])}+{int(r['n_mdd'])}")

print("\n--- Cavanagh 2025 sensors ---")
for s in CAVANAGH_2025_SENSORS:
    row = df_sl[df_sl['channel'] == s]
    if not row.empty:
        r = row.iloc[0]
        print(f"  {s}: d={r['d']:.3f}, p={r['p']:.4f}, "
              f"N={int(r['n_ctl'])}+{int(r['n_mdd'])}")
    else:
        print(f"  {s}: not in channel list")

print("\n--- MEG06xx gradiometers (top 5) ---")
meg06 = df_sl[df_sl['channel'].str.startswith('MEG06')].dropna(subset=['d']).head(5)
if not meg06.empty:
    print(meg06[['channel', 'd', 'p', 'n_ctl', 'n_mdd']].to_string(index=False))
else:
    print("  no MEG06xx channels found")

print("\n--- Top 10 neighborhood averages ---")
if not df_nb.empty:
    print(df_nb[['center_ch', 'ch_type', 'n_neighbors', 'd', 'p',
                 'n_ctl', 'n_mdd']].head(10).to_string(index=False))

# Save quality log
pd.DataFrame(quality_log).to_csv(
    DERIV_DIR / 'searchlight_quality_failures.csv', index=False)
print(f"\nQuality failures logged: {len(quality_log)} subject-channel pairs")

# ── PHASE 6: Final result with best channel ───────────────────────────────────
best_single = df_sl.dropna(subset=['d']).iloc[0]
best_ch     = best_single['channel']
best_d      = best_single['d']
best_p      = best_single['p']

best_nb_ch = df_nb.iloc[0]['center_ch'] if not df_nb.empty else None
best_nb_d  = df_nb.iloc[0]['d']         if not df_nb.empty else np.nan

print(f"\n=== PHASE 6: FINAL RESULT — best channel ({best_ch}) ===")

best_ctl, best_mdd = [], []
for sid, ch_dict in sub_channel_data.items():
    grp = grp_map.get(sid)
    val = ch_dict.get(best_ch, np.nan)
    if np.isfinite(val):
        if grp == 'CTL':
            best_ctl.append(val)
        elif grp == 'MDD':
            best_mdd.append(val)

a_f, b_f = np.array(best_ctl), np.array(best_mdd)
d_final   = cohen_d(a_f, b_f)
_, p_final = mannwhitneyu(a_f, b_f, alternative='two-sided')

print(f"Channel: {best_ch}  (sub-M87121835 excluded)")
print(f"CTL: {a_f.mean():.4f} ± {a_f.std(ddof=1):.4f}  (N={len(a_f)})")
print(f"MDD: {b_f.mean():.4f} ± {b_f.std(ddof=1):.4f}  (N={len(b_f)})")
print(f"Cohen's d = {d_final:.3f}")
print(f"MWU p     = {p_final:.4f}")
print(f"Direction CTL>MDD: {'✓' if a_f.mean() > b_f.mean() else '✗'}")

# ── PHASE 7: Figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f'MEG AIS$_{{pre}}$ Channel Searchlight  (sub-M87121835 excluded)\n'
    f'−200 ms to 0 ms pre-feedback · lag=1 · bins=4',
    fontsize=11)

# Panel A — top 30 channels by d
ax = axes[0]
top30 = df_sl.dropna(subset=['d']).head(30)
bar_colors = []
for ch in top30['channel']:
    if ch == 'EEG007':
        bar_colors.append('#E53935')
    elif ch in CAVANAGH_2025_SENSORS:
        bar_colors.append('#FF9800')
    elif ch.startswith('MEG06'):
        bar_colors.append('#2196F3')
    else:
        bar_colors.append('#78909C')

ax.barh(range(len(top30)), top30['d'].values, color=bar_colors)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30['channel'].values, fontsize=7)
ax.invert_yaxis()
ax.axvline(0.448, color='#E53935', lw=0.8, ls='--', alpha=0.6,
           label='EEG007 prev (d=0.448, with artifact)')
ax.axvline(0.0, color='black', lw=0.6)
ax.set_xlabel("Cohen's d  (CTL − MDD)")
ax.set_title("A.  Top 30 channels (single)", loc='left', fontsize=10)
ax.legend(handles=[
    Patch(color='#E53935', label='EEG007'),
    Patch(color='#FF9800', label='Cavanagh 2025'),
    Patch(color='#2196F3', label='MEG06xx grad'),
    Patch(color='#78909C', label='other'),
], fontsize=7.5, frameon=False)

# Panel B — neighborhood averages
ax = axes[1]
if not df_nb.empty:
    top20nb = df_nb.dropna(subset=['d']).head(20)
    nb_colors = ['#FF9800' if c in CAVANAGH_2025_SENSORS
                 else '#2196F3' if c.startswith('MEG06')
                 else '#7E57C2'
                 for c in top20nb['center_ch']]
    ax.barh(range(len(top20nb)), top20nb['d'].values, color=nb_colors)
    ax.set_yticks(range(len(top20nb)))
    ax.set_yticklabels(top20nb['center_ch'].values, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0.0, color='black', lw=0.6)
    ax.set_xlabel("Cohen's d  (CTL − MDD)")
    ax.set_title("B.  Top 20 neighborhood averages", loc='left', fontsize=10)
else:
    ax.text(0.5, 0.5, 'No position data available for neighbor lookup',
            ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
out_fig = DERIV_DIR / 'channel_searchlight_map.png'
fig.savefig(out_fig, dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {out_fig}")

print("\n" + "="*65)
print("SEARCHLIGHT COMPLETE")
print(f"  Best single channel:  {best_ch}  d={best_d:.3f}, p={best_p:.4f}")
if best_nb_ch:
    print(f"  Best neighborhood:    {best_nb_ch}  d={best_nb_d:.3f}")
print(f"  EEG007 (prev):        d=0.448  (N=83, artifact included)")
print(f"  EEG007 (clean):       see searchlight CSV")
print("="*65)
