import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE     = Path("/media/neuraldyn/PortableSSD/DEPRESSION/01_raw_data/Cavanagh/Depression_PS_Task")
EPO_DIR  = BASE / "derivatives/epochs"
OUT_DIR  = BASE / "derivatives/erp_it_cavanagh"
CLINICAL = BASE / "derivatives/clinical_lookup_ps_task.csv"

clinical    = pd.read_csv(CLINICAL)
GROUP_COL   = 'analysis_group_broad'
CTL_LABEL   = 'CTL'
MDD_LABEL   = 'MDD_any'

clinical_main = clinical[
    clinical[GROUP_COL].isin([CTL_LABEL, MDD_LABEL]) & (~clinical['excluded'])
].copy()
print(f"N CTL: {(clinical_main[GROUP_COL]==CTL_LABEL).sum()}, "
      f"N MDD: {(clinical_main[GROUP_COL]==MDD_LABEL).sum()}")

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1-1)*np.std(a, ddof=1)**2 + (n2-1)*np.std(b, ddof=1)**2) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0

# ── Probe one file ────────────────────────────────────────────────────────
epoch_files = sorted(EPO_DIR.glob('*_task-ps_epo.fif'))
sample_epo  = mne.read_epochs(epoch_files[0], preload=False, verbose='ERROR')
times       = sample_epo.times
sfreq       = sample_epo.info['sfreq']

FCZ = 'FCz' if 'FCz' in sample_epo.ch_names else 'Fz'
print(f"Primary channel: {FCZ}")
print(f"All channels: {sample_epo.ch_names}")
print(f"Event IDs: {sample_epo.event_id}")

REWARD_KEY = None; LOSS_KEY = None
for k, v in sample_epo.event_id.items():
    if v == 94:  REWARD_KEY = k
    if v == 104: LOSS_KEY   = k
print(f"Reward key: {REWARD_KEY!r}, Loss key: {LOSS_KEY!r}")
assert REWARD_KEY and LOSS_KEY

CHANNELS_TO_TEST = [c for c in ['Fz','FCz','Cz','FC1','FC2','F3','F4']
                    if c in sample_epo.ch_names]
print(f"Channels to compare: {CHANNELS_TO_TEST}")

# ── STEP 4: Event verification on first 5 subjects ───────────────────────
print("\n=== STEP 4: EVENT VERIFICATION (first 5 subjects) ===")
for fpath in epoch_files[:5]:
    sub_id = int(fpath.stem.split('-')[1].split('_')[0])
    epo    = mne.read_epochs(fpath, preload=False, verbose='ERROR')
    n_rew  = len(epo[REWARD_KEY])
    n_los  = len(epo[LOSS_KEY])
    total  = n_rew + n_los
    print(f"  sub-{sub_id}: N_Reward={n_rew}, N_Loss={n_los}, "
          f"Reward_ratio={n_rew/total:.2f}  (expected ~0.50 for balanced design)")

# ── Main extraction loop ──────────────────────────────────────────────────
ctl_rew_waves, ctl_los_waves = [], []
mdd_rew_waves, mdd_los_waves = [], []

# For individual subject plots (Step 3)
ctl_sample_ids, mdd_sample_ids = [], []
ctl_sample_rew, ctl_sample_los = [], []
mdd_sample_rew, mdd_sample_los = [], []

records          = []
baseline_records = []

print(f"\nProcessing {len(epoch_files)} files...")
for idx, fpath in enumerate(epoch_files):
    try:
        sub_id = int(fpath.stem.split('-')[1].split('_')[0])
    except (IndexError, ValueError):
        continue
    row_c = clinical_main[clinical_main['subject_id'] == sub_id]
    if row_c.empty:
        continue
    group = row_c.iloc[0][GROUP_COL]

    try:
        epo = mne.read_epochs(fpath, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"  sub-{sub_id}: load error — {e}")
        continue

    ch_idx  = epo.ch_names.index(FCZ)
    rew_epo = epo[REWARD_KEY]
    los_epo = epo[LOSS_KEY]
    if len(rew_epo) < 3 or len(los_epo) < 3:
        continue

    # Per-subject mean waveforms at FCz
    rew_wave = rew_epo.get_data()[:, ch_idx, :].mean(axis=0)
    los_wave = los_epo.get_data()[:, ch_idx, :].mean(axis=0)

    if group == CTL_LABEL:
        ctl_rew_waves.append(rew_wave)
        ctl_los_waves.append(los_wave)
        if len(ctl_sample_ids) < 3:
            ctl_sample_ids.append(sub_id)
            ctl_sample_rew.append(rew_wave)
            ctl_sample_los.append(los_wave)
    else:
        mdd_rew_waves.append(rew_wave)
        mdd_los_waves.append(los_wave)
        if len(mdd_sample_ids) < 3:
            mdd_sample_ids.append(sub_id)
            mdd_sample_rew.append(rew_wave)
            mdd_sample_los.append(los_wave)

    # STEP 6: Baseline check (−200ms to 0ms)
    bl_mask = (times >= -0.200) & (times < 0.0)
    baseline_records.append({
        'subject_id':          sub_id,
        'group':               group,
        'baseline_reward_mean': rew_wave[bl_mask].mean(),
        'baseline_loss_mean':   los_wave[bl_mask].mean(),
    })

    # STEP 2: Difference wave in 4 windows at FCz
    for tmin, tmax, label in [
        (0.200, 0.350, 'RewP_200_350'),
        (0.250, 0.350, 'RewP_250_350'),
        (0.300, 0.400, 'RewP_300_400'),
        (0.350, 0.450, 'RewP_350_450'),
    ]:
        mask = (times >= tmin) & (times <= tmax)
        records.append({
            'subject_id': sub_id, 'group': group, 'window': label,
            'rewp_diff':  rew_wave[mask].mean() - los_wave[mask].mean(),
            'reward_amp': rew_wave[mask].mean(),
            'loss_amp':   los_wave[mask].mean(),
        })

    # STEP 5: Channel comparison at 350–450ms
    mask_450 = (times >= 0.350) & (times <= 0.450)
    for ch in CHANNELS_TO_TEST:
        cidx    = epo.ch_names.index(ch)
        rew_ch  = rew_epo.get_data()[:, cidx, :].mean(axis=0)[mask_450].mean()
        los_ch  = los_epo.get_data()[:, cidx, :].mean(axis=0)[mask_450].mean()
        records.append({
            'subject_id': sub_id, 'group': group, 'window': f'CHAN_{ch}',
            'rewp_diff':  rew_ch - los_ch,
            'reward_amp': rew_ch, 'loss_amp': los_ch,
        })

    if (idx + 1) % 25 == 0:
        print(f"  [{idx+1}/{len(epoch_files)}]")

df      = pd.DataFrame(records)
df_base = pd.DataFrame(baseline_records)
ms      = times * 1000  # for plotting

# ── STEP 6: Baseline check ────────────────────────────────────────────────
print("\n=== STEP 6: BASELINE CHECK (mean amplitude −200 to 0ms) ===")
for grp in (CTL_LABEL, MDD_LABEL):
    sub = df_base[df_base['group'] == grp]
    print(f"  {grp}:")
    print(f"    Reward baseline: mean={sub['baseline_reward_mean'].mean():.4f}, "
          f"std={sub['baseline_reward_mean'].std():.4f}")
    print(f"    Loss   baseline: mean={sub['baseline_loss_mean'].mean():.4f}, "
          f"std={sub['baseline_loss_mean'].std():.4f}")
grand_bl = df_base[['baseline_reward_mean','baseline_loss_mean']].abs().mean().mean()
print(f"  Grand mean |baseline|: {grand_bl:.5f}")
print("  OK: baseline near zero." if grand_bl < 0.5 else "  WARNING: baseline NOT near zero.")

# ── STEP 2: Difference wave statistics ───────────────────────────────────
print("\n=== STEP 2: REWP DIFFERENCE WAVE (Reward − Loss) ===")
window_results = []
for win in ['RewP_200_350','RewP_250_350','RewP_300_400','RewP_350_450']:
    sub  = df[df['window'] == win]
    ctl  = sub[sub['group'] == CTL_LABEL]['rewp_diff'].dropna().values
    mdd  = sub[sub['group'] == MDD_LABEL]['rewp_diff'].dropna().values
    if len(ctl) < 3 or len(mdd) < 3:
        continue
    U, p = mannwhitneyu(ctl, mdd, alternative='two-sided')
    d    = cohens_d(ctl, mdd)
    confirmed = np.mean(ctl) > np.mean(mdd)
    print(f"  {win}: CTL={np.mean(ctl):+.4f} MDD={np.mean(mdd):+.4f}  "
          f"d={d:+.3f}  p={p:.4f}  CTL>MDD={'YES ✓' if confirmed else 'NO ✗'}")
    window_results.append({'window': win, 'CTL_mean': np.mean(ctl),
                           'MDD_mean': np.mean(mdd), 'd': d, 'p': p, 'confirmed': confirmed})

# ── STEP 5: Channel comparison ────────────────────────────────────────────
print("\n=== STEP 5: CHANNEL COMPARISON (350–450ms, Reward−Loss) ===")
chan_results = []
for ch in CHANNELS_TO_TEST:
    sub  = df[df['window'] == f'CHAN_{ch}']
    ctl  = sub[sub['group'] == CTL_LABEL]['rewp_diff'].dropna().values
    mdd  = sub[sub['group'] == MDD_LABEL]['rewp_diff'].dropna().values
    if len(ctl) < 3 or len(mdd) < 3:
        continue
    U, p = mannwhitneyu(ctl, mdd, alternative='two-sided')
    d    = cohens_d(ctl, mdd)
    print(f"  {ch:5s}: CTL={np.mean(ctl):+.4f}  MDD={np.mean(mdd):+.4f}  "
          f"d={d:+.3f}  p={p:.4f}")
    chan_results.append({'channel': ch, 'd': d, 'p': p,
                         'CTL_mean': np.mean(ctl), 'MDD_mean': np.mean(mdd)})

df_chan = pd.DataFrame(chan_results)
if len(df_chan) > 0:
    best = df_chan.loc[df_chan['d'].abs().idxmax()]
    print(f"\n  Best channel: {best['channel']}  (|d|={abs(best['d']):.3f})")
    if best['channel'] != FCZ:
        print(f"  NOTE: {best['channel']} outperforms {FCZ} — consider using it downstream.")
    else:
        print(f"  {FCZ} is confirmed as the optimal channel.")

# ── STEP 1: Grand average plot ────────────────────────────────────────────
n_ctl = len(ctl_rew_waves)
n_mdd = len(mdd_rew_waves)

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 (top, full width): grand average
ax1 = fig.add_subplot(gs[0, :])
for waves, label, color, ls in [
    (ctl_rew_waves, f'CTL Reward (N={n_ctl})', '#1565C0', '-'),
    (ctl_los_waves, f'CTL Loss',               '#1565C0', '--'),
    (mdd_rew_waves, f'MDD Reward (N={n_mdd})', '#C62828', '-'),
    (mdd_los_waves, f'MDD Loss',               '#C62828', '--'),
]:
    if not waves:
        continue
    grand = np.array(waves).mean(axis=0)
    sem   = np.array(waves).std(axis=0) / np.sqrt(len(waves))
    ax1.plot(ms, grand, color=color, linestyle=ls, lw=2, label=label)
    ax1.fill_between(ms, grand - sem, grand + sem, color=color, alpha=0.12)

ax1.axvline(0, color='black', ls=':', lw=1.2, alpha=0.6)
ax1.axhline(0, color='gray', lw=0.5)
ax1.axvspan(250, 350, alpha=0.10, color='green',  label='RewP 250–350ms')
ax1.axvspan(350, 450, alpha=0.10, color='orange', label='Cavanagh t-ROI 350–450ms')
ax1.set_xlim(-500, 1000)
ax1.set_xlabel('Time post-feedback (ms)')
ax1.set_ylabel('Amplitude (µV)')
ax1.set_title(f'Grand Average ERPs at {FCZ} — All Groups and Conditions')
ax1.legend(fontsize=8, ncol=3, loc='upper right')

# Panel 2 (middle row): difference waves CTL vs MDD
ax2 = fig.add_subplot(gs[1, :2])
ctl_diff = np.array(ctl_rew_waves).mean(axis=0) - np.array(ctl_los_waves).mean(axis=0)
mdd_diff = np.array(mdd_rew_waves).mean(axis=0) - np.array(mdd_los_waves).mean(axis=0)
ctl_diff_sem = (np.array(ctl_rew_waves) - np.array(ctl_los_waves)).std(axis=0) / np.sqrt(n_ctl)
mdd_diff_sem = (np.array(mdd_rew_waves) - np.array(mdd_los_waves)).std(axis=0) / np.sqrt(n_mdd)
ax2.plot(ms, ctl_diff, color='#1565C0', lw=2, label=f'CTL diff (N={n_ctl})')
ax2.fill_between(ms, ctl_diff - ctl_diff_sem, ctl_diff + ctl_diff_sem, color='#1565C0', alpha=0.15)
ax2.plot(ms, mdd_diff, color='#C62828', lw=2, label=f'MDD diff (N={n_mdd})')
ax2.fill_between(ms, mdd_diff - mdd_diff_sem, mdd_diff + mdd_diff_sem, color='#C62828', alpha=0.15)
ax2.axvline(0, color='black', ls=':', alpha=0.5)
ax2.axhline(0, color='gray', lw=0.5)
ax2.axvspan(250, 350, alpha=0.10, color='green')
ax2.axvspan(350, 450, alpha=0.10, color='orange')
ax2.set_xlim(-500, 1000)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Reward − Loss (µV)')
ax2.set_title('RewP Difference Wave (Reward − Loss)')
ax2.legend(fontsize=9)

# Panel 3 (middle right): channel comparison bar
ax3 = fig.add_subplot(gs[1, 2])
if len(df_chan) > 0:
    colors_bar = ['#C62828' if d < 0 else '#1565C0' for d in df_chan['d']]
    ax3.barh(df_chan['channel'], df_chan['d'], color=colors_bar, edgecolor='black', lw=0.5)
    ax3.axvline(0, color='black', lw=0.8)
    ax3.set_xlabel("Cohen's d (CTL vs MDD)")
    ax3.set_title('RewP d by Channel\n(350–450ms)')
    ax3.invert_yaxis()

# Panels 4–6 (bottom row): 3 CTL single subjects
for i, (sid, rw, lw) in enumerate(zip(ctl_sample_ids, ctl_sample_rew, ctl_sample_los)):
    ax = fig.add_subplot(gs[2, i])
    ax.plot(ms, rw, color='#1565C0', lw=1.5, label='Reward')
    ax.plot(ms, lw, color='#1565C0', lw=1.5, ls='--', label='Loss')
    ax.axvline(0, color='black', ls=':', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_xlim(-300, 800)
    ax.set_title(f'CTL sub-{sid}', fontsize=9)
    ax.set_xlabel('Time (ms)', fontsize=8)
    if i == 0:
        ax.set_ylabel('µV', fontsize=8)
    ax.legend(fontsize=7)

fig.suptitle('RewP Canonical Validation — Cavanagh PS Task EEG', fontsize=13, fontweight='bold')
plt.savefig(OUT_DIR / 'validation_rewp_canonical.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {OUT_DIR / 'validation_rewp_canonical.png'}")

# ── STEP 3: Single-subject MDD plot (separate file) ───────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle('Individual Subject ERPs at FCz (3 CTL, 3 MDD)', fontsize=11, fontweight='bold')

for i, (sid, rw, lw) in enumerate(zip(ctl_sample_ids, ctl_sample_rew, ctl_sample_los)):
    ax = axes2[0, i]
    ax.plot(ms, rw, color='#1565C0', lw=1.5, label='Reward')
    ax.plot(ms, lw, color='#1565C0', lw=1.5, ls='--', label='Loss')
    ax.axvline(0, color='k', ls=':', alpha=0.5); ax.axhline(0, color='gray', lw=0.4)
    ax.axvspan(250, 350, alpha=0.12, color='green')
    ax.set_xlim(-300, 800); ax.set_title(f'CTL sub-{sid}', fontsize=9)
    ax.set_xlabel('ms'); ax.legend(fontsize=7)
    if i == 0: ax.set_ylabel('µV')

for i, (sid, rw, lw) in enumerate(zip(mdd_sample_ids, mdd_sample_rew, mdd_sample_los)):
    ax = axes2[1, i]
    ax.plot(ms, rw, color='#C62828', lw=1.5, label='Reward')
    ax.plot(ms, lw, color='#C62828', lw=1.5, ls='--', label='Loss')
    ax.axvline(0, color='k', ls=':', alpha=0.5); ax.axhline(0, color='gray', lw=0.4)
    ax.axvspan(250, 350, alpha=0.12, color='green')
    ax.set_xlim(-300, 800); ax.set_title(f'MDD sub-{sid}', fontsize=9)
    ax.set_xlabel('ms'); ax.legend(fontsize=7)
    if i == 0: ax.set_ylabel('µV')

plt.tight_layout()
plt.savefig(OUT_DIR / 'validation_single_subjects.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure saved: {OUT_DIR / 'validation_single_subjects.png'}")

# ── Summary verdict ───────────────────────────────────────────────────────
print("\n=== VALIDATION VERDICT ===")
wr = [r for r in window_results if r['window'] == 'RewP_250_350']
if wr:
    w    = wr[0]
    ok   = w['confirmed'] and w['p'] < 0.10
    flag = "PASS" if ok else "FAIL / INVESTIGATE"
    print(f"  RewP_250_350: CTL={w['CTL_mean']:+.4f}  MDD={w['MDD_mean']:+.4f}  "
          f"d={w['d']:+.3f}  p={w['p']:.4f}  → {flag}")
print("  If FAIL: check event labels, baseline, channel polarity before AIS.")
print("\nDone.")
