"""
Analysis B: Transfer Entropy F1→FCz (DLPFC→ACC)
Pre-feedback window (-200ms to 0ms), PST dataset

Key fix: F1→FCz instead of Fz→FCz (too close, d=-0.007 null)
"""

import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
import mne

mne.set_log_level('ERROR')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/media/neuraldyn/PortableSSD/DEPRESSION")
PST  = BASE / "01_raw_data/Cavanagh/Depression_PS_Task"
DERIV = PST / "derivatives"

epo_path   = DERIV / "epochs"
clin_file  = DERIV / "clinical_lookup_ps_task.csv"
freeze_file = DERIV / "RESEARCH_STATE_FROZEN.json"

pst_epo_files = sorted(epo_path.glob("sub-*_task-ps_epo.fif"))
print(f"PST epoch files: {len(pst_epo_files)}")

# ── Clinical ─────────────────────────────────────────────────────────────────
clin = pd.read_csv(clin_file)
# subject_id is integer in file (507, 508 …)
clin['subject_id'] = clin['subject_id'].astype(str)
clin_lookup = clin.set_index('subject_id')
print(f"Clinical rows: {len(clin)}  groups: {clin['analysis_group_broad'].value_counts().to_dict()}")

# ── Frozen state ─────────────────────────────────────────────────────────────
with open(freeze_file) as f:
    state = json.load(f)

print("\n=== RESEARCH STATE ===")
pe = state['primary_eeg']
print(f"Primary EEG (AIS_pre): d={pe['cohens_d']:.3f}, p={pe['mwu_p']:.5f}, "
      f"N_CTL={pe['N_CTL']}, N_MDD={pe['N_MDD']}")
sc = state['scar_hypothesis']
print(f"Scar hypothesis: d={sc['d_cur_over_past']:.3f}, p={sc['mwu_p']:.4f}")
print(f"tdbrain: {state['tdbrain_ais_rest']['interpretation']}")

# ── TE function (vectorised, bin-based) ──────────────────────────────────────
def safe_te(source: np.ndarray, target: np.ndarray,
            lag: int = 1, n_bins: int = 4) -> float:
    """
    TE(source→target) = I(Y_t ; X_{t-lag} | Y_{t-lag})
    Uses percentile-based uniform binning.
    Returns float, or nan on failure.
    """
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    n = len(x)
    if n != len(y) or n < 3 * lag + 10:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    try:
        xe = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
        ye = np.unique(np.percentile(y, np.linspace(0, 100, n_bins + 1)))
        if len(xe) < 3 or len(ye) < 3:
            return np.nan
        xb = np.digitize(x, xe[1:-1])      # 0-indexed bins → 0..n_bins-1
        yb = np.digitize(y, ye[1:-1])

        # Triplet (Y_t, Y_{t-lag}, X_{t-lag}) for t=lag..n-1
        yt    = yb[lag:]     # shape (n-lag,)
        ytlag = yb[:n - lag]
        xtlag = xb[:n - lag]

        nb = n_bins
        # 3-D joint: shape (nb, nb, nb) → (yt, ytlag, xtlag)
        # Using linear index for speed
        idx3 = yt * nb * nb + ytlag * nb + xtlag
        p3 = np.bincount(idx3, minlength=nb**3).reshape(nb, nb, nb).astype(float)
        p3 /= (p3.sum() + 1e-12)

        # Marginals
        p_yy  = p3.sum(axis=2)   # P(Y_t, Y_{t-lag})
        p_yyl = p3.sum(axis=0)   # P(Y_{t-lag}, X_{t-lag}) — not needed
        p_yl  = p_yy.sum(axis=0) # P(Y_{t-lag})

        # TE = Σ p(yt,ytl,xtl) log [ p(yt|ytl,xtl) / p(yt|ytl) ]
        # p(yt|ytl,xtl) = p3[yt,ytl,xtl] / sum_yt p3[yt,ytl,xtl]
        # p(yt|ytl)     = p_yy[yt,ytl]   / p_yl[ytl]
        denom_yyx = p3.sum(axis=0, keepdims=True) + 1e-12  # (1, nb, nb)
        denom_yy  = p_yl[np.newaxis, :, np.newaxis] + 1e-12  # (1, nb, 1)

        ratio = (p3 / denom_yyx) / (p_yy[:, :, np.newaxis] / denom_yy + 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio = np.where(ratio > 0, np.log2(ratio), 0.0)
        te = float(np.sum(p3 * log_ratio))
        return te if np.isfinite(te) else np.nan
    except Exception:
        return np.nan


# Validate TE
rng = np.random.default_rng(123)
driver = np.zeros(600)
for i in range(1, 600):
    driver[i] = 0.8 * driver[i-1] + 0.2 * rng.standard_normal()
driven = 0.5 * driver + 0.5 * rng.standard_normal(600)
te_fwd = safe_te(driver, driven)
te_bwd = safe_te(driven, driver)
assert te_fwd > te_bwd, f"TE validation FAILED: fwd={te_fwd:.4f} bwd={te_bwd:.4f}"
print(f"\nTE validation OK  fwd={te_fwd:.4f} > bwd={te_bwd:.4f}")


# ── Channel pairs ─────────────────────────────────────────────────────────────
TE_PAIRS = [
    ('F1',  'FCz', 'DLPFC_L→ACC'),     # PRIMARY
    ('F2',  'FCz', 'DLPFC_R→ACC'),     # PRIMARY
    ('FC1', 'FCz', 'premotor_L→ACC'),
    ('FC2', 'FCz', 'premotor_R→ACC'),
    ('Fz',  'FCz', 'mesial_F→ACC'),    # null control
]

PRE_WINDOW = (-0.200, 0.000)   # same as AIS_pre primary finding
MIN_TRIALS = 10

print(f"\nPre-feedback window: {PRE_WINDOW[0]*1000:.0f}ms to {PRE_WINDOW[1]*1000:.0f}ms")
print(f"Pairs: {[p[2] for p in TE_PAIRS]}")
print("\n" + "="*60)
print("ANALYSIS B: TRANSFER ENTROPY — Processing subjects")
print("="*60)

records = []

for fpath in pst_epo_files:
    sub_str = fpath.name.split('_')[0]   # e.g. "sub-507"
    sub_id  = sub_str.replace('sub-', '') # "507"

    if sub_id not in clin_lookup.index:
        continue
    row = clin_lookup.loc[sub_id]
    if row.get('excluded', False):
        continue
    group_raw = row['analysis_group_broad']
    # Normalize: MDD_any → MDD
    if group_raw == 'CTL':
        group = 'CTL'
    elif group_raw == 'MDD_any':
        group = 'MDD'
    else:
        continue

    try:
        epo   = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        times = epo.times
        pre_m = (times >= PRE_WINDOW[0]) & (times < PRE_WINDOW[1])

        avail = epo.ch_names
        data  = epo.get_data()   # (n_trials, n_ch, n_times)

        rec = {'subject_id': sub_id, 'group': group,
               'BDI': row.get('BDI', np.nan)}

        any_pair_ok = False
        for src_ch, tgt_ch, label in TE_PAIRS:
            if src_ch not in avail or tgt_ch not in avail:
                continue
            si = avail.index(src_ch)
            ti = avail.index(tgt_ch)
            col_key = f"{src_ch}_to_{tgt_ch}"

            tf_list, tb_list = [], []
            for trial in range(len(epo)):
                seg_s = data[trial, si, pre_m]
                seg_t = data[trial, ti, pre_m]
                tf_list.append(safe_te(seg_s, seg_t))
                tb_list.append(safe_te(seg_t, seg_s))

            vf = [v for v in tf_list if np.isfinite(v)]
            vb = [v for v in tb_list if np.isfinite(v)]

            if len(vf) >= MIN_TRIALS:
                rec[f'TE_fwd_{col_key}'] = float(np.mean(vf))
                rec[f'TE_bwd_{col_key}'] = float(np.mean(vb)) if len(vb) >= MIN_TRIALS else np.nan
                rec[f'TE_net_{col_key}'] = (rec[f'TE_fwd_{col_key}'] -
                                            (rec[f'TE_bwd_{col_key}']
                                             if np.isfinite(rec[f'TE_bwd_{col_key}']) else 0))
                rec[f'n_trials_{col_key}'] = len(vf)
                any_pair_ok = True

        if any_pair_ok:
            records.append(rec)
            # Brief progress every 10 subjects
            if len(records) % 10 == 0:
                print(f"  {len(records)} subjects done …")

    except Exception as e:
        print(f"  {sub_id} ERROR: {e}")

df_te = pd.DataFrame(records)
print(f"\nTE computed: {len(df_te)} subjects "
      f"(CTL={len(df_te[df_te.group=='CTL'])}, MDD={len(df_te[df_te.group=='MDD'])})")

# ── Cohen's d ─────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a)-1)*np.var(a, ddof=1) +
                  (len(b)-1)*np.var(b, ddof=1)) /
                 (len(a) + len(b) - 2) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / sp)

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRANSFER ENTROPY RESULTS (CTL vs MDD)")
print("Reference: Fz→FCz d=-0.007 (too close, null)")
print("="*60)

te_rows = []
for src_ch, tgt_ch, label in TE_PAIRS:
    col_key = f"{src_ch}_to_{tgt_ch}"
    for measure_type in ('fwd', 'net'):
        col = f'TE_{measure_type}_{col_key}'
        if col not in df_te.columns:
            continue
        ctl_v = df_te[df_te.group == 'CTL'][col].dropna()
        mdd_v = df_te[df_te.group == 'MDD'][col].dropna()
        if len(ctl_v) < 5 or len(mdd_v) < 5:
            continue
        U, p = mannwhitneyu(ctl_v, mdd_v, alternative='two-sided')
        d    = cohens_d(ctl_v.values, mdd_v.values)
        dirn = 'CTL>MDD' if ctl_v.mean() > mdd_v.mean() else 'MDD>CTL'
        te_rows.append(dict(pair=label, measure=measure_type,
                            d=round(d, 3), p=round(p, 4),
                            direction=dirn,
                            CTL_mean=round(ctl_v.mean(), 5),
                            MDD_mean=round(mdd_v.mean(), 5),
                            N_CTL=len(ctl_v), N_MDD=len(mdd_v)))

        primary_flag = (src_ch in ('F1', 'F2') and tgt_ch == 'FCz'
                        and measure_type == 'fwd')
        marker = '  ← PRIMARY' if primary_flag else ''
        print(f"\n  {label} [{measure_type}]{marker}")
        print(f"    CTL={ctl_v.mean():.5f}±{ctl_v.std():.5f} (N={len(ctl_v)})")
        print(f"    MDD={mdd_v.mean():.5f}±{mdd_v.std():.5f} (N={len(mdd_v)})")
        print(f"    d={d:.3f}, p={p:.4f} [{dirn}]")
        if primary_flag:
            if abs(d) >= 0.30 and p < 0.05:
                print(f"    ✅ FINDING: DLPFC→ACC flow differs CTL vs MDD")
            elif abs(d) >= 0.20:
                print(f"    ⚠  TREND: d={d:.3f} but p={p:.4f}")
            else:
                print(f"    → NULL: DLPFC→ACC TE not informative")

df_results = pd.DataFrame(te_rows).sort_values('p').reset_index(drop=True)
print("\n\nAll TE results sorted by |d|:")
print(df_results.sort_values('d', key=abs, ascending=False).to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
out_sub   = DERIV / "te_f1_fcz_subject_level.csv"
out_stats = DERIV / "te_f1_fcz_results.csv"
df_te.to_csv(out_sub, index=False)
df_results.to_csv(out_stats, index=False)
print(f"\nSaved: {out_sub.name}, {out_stats.name}")

# ── Update frozen state ───────────────────────────────────────────────────────
# Extract primary result: F1→FCz fwd
primary_row = df_results[
    (df_results.pair == 'DLPFC_L→ACC') &
    (df_results.measure == 'fwd')
]
primary_d = float(primary_row['d'].iloc[0]) if len(primary_row) else None
primary_p = float(primary_row['p'].iloc[0]) if len(primary_row) else None

state['te_f1_fcz'] = {
    'analysis_date': '2026-05-03',
    'window': '-200ms to 0ms (pre-feedback)',
    'primary_pair': 'F1→FCz (left DLPFC→ACC)',
    'primary_d': primary_d,
    'primary_p': primary_p,
    'reference_null': {'pair': 'Fz→FCz', 'd': -0.007, 'p': 0.845},
    'pairs_tested': [p[2] for p in TE_PAIRS],
    'full_results': te_rows,
    'verdict': ('FINDING' if primary_d is not None and abs(primary_d) >= 0.30 and primary_p < 0.05
                else 'TREND' if primary_d is not None and abs(primary_d) >= 0.20
                else 'NULL'),
}

# Hayling boundary note
state['hayling_pre500_blocked'] = {
    'analysis_date': '2026-05-03',
    'reason': 'Hayling epochs tmin=-0.2s; -500ms window not captured in any processed file',
    'solution': 'Re-epoch from raw HYL_*_90_Hz-raw.fif with tmin=-0.6s, then re-apply ICA/AR',
    'status': 'BLOCKED — re-epoch required'
}

with open(freeze_file, 'w') as f:
    json.dump(state, f, indent=2)
print("Frozen state updated.")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Analysis A (Hayling -500 to -200ms):")
print(f"  BLOCKED — all Hayling processed epochs have tmin=-0.2s")
print(f"  Resolution: re-epoch from raw with tmin=-0.6s")
print(f"\nAnalysis B (TE F1→FCz):")
if primary_d is not None:
    print(f"  F1→FCz: d={primary_d:.3f}, p={primary_p:.4f}")
    print(f"  Verdict: {state['te_f1_fcz']['verdict']}")
else:
    print("  No result computed")
