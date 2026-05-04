"""
TDBRAIN restEC chaos/nonlinear measures → remission prediction
Measures: AC1, Variance, SampEn, PermEnt, HFD, DFA-α, CSD index
Test: (a) remission prediction, (b) independence from AIS_rest
"""

import warnings; warnings.filterwarnings('ignore')
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from scipy.signal import decimate
import mne; mne.set_log_level('ERROR')
import antropy as ant
import nolds
from numpy.linalg import lstsq

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path("/media/neuraldyn/PortableSSD/DEPRESSION")
PST     = BASE / "01_raw_data/Cavanagh/Depression_PS_Task"
TDBRAIN = Path("/media/neuraldyn/PortableSSD/TDBRAIN_derivatives")
ASSETS  = BASE / "06_manuscript_assets"
freeze_file = PST / "derivatives/RESEARCH_STATE_FROZEN.json"

# ── Reload frozen state ────────────────────────────────────────────────────────
with open(freeze_file) as f:
    state = json.load(f)
print("=== FROZEN STATE ===")
print(f"Date: {state['freeze_date']}")
tdb = state['tdbrain_ais_rest']
print(f"AIS_rest: d={tdb['cohens_d_rem_over_nonrem']:.3f}, "
      f"p={tdb['mwu_p_nonrem_greater']:.4f} (Non-rem > Rem)")
te_r  = state.get('te_f1_fcz', {})
print(f"TE follow-up: r(TE, AIS_pre)=-0.883 — TE is mathematical shadow of AIS")

# ── Load existing AIS_rest results (subject list + remission labels) ───────────
ais_rest = pd.read_csv(ASSETS / "tdbrain_ais_rest_results.csv")
ais_rest['subject_id'] = ais_rest['subject_id'].astype(str)
remit_col = 'remitter_coded'
ais_rest_lookup = ais_rest.set_index('subject_id')
print(f"\nAIS_rest loaded: {len(ais_rest)} subjects")
print(f"Remitters: {(ais_rest[remit_col]==1).sum()}, "
      f"Non-remitters: {(ais_rest[remit_col]==0).sum()}")

# ── Search for previous chaos work (STEP 0B summary) ─────────────────────────
CHAOS_KEYWORDS = ['lyapunov','chaos','fractal','dfa','sampen','hfd','nonlinear']
found_py = [f for f in BASE.rglob("*.py")
            if any(k in f.name.lower() for k in CHAOS_KEYWORDS)]
print(f"\nExisting chaos scripts: {len(found_py)}")
if found_py:
    for f in found_py: print(f"  {f.name}")

# ── Measure functions (using antropy + nolds) ─────────────────────────────────
def compute_ac1(x):
    x = x - x.mean()
    sd = np.std(x)
    if sd < 1e-10: return np.nan
    n = len(x)
    return float(np.dot(x[1:], x[:-1]) / (n * sd**2))

def compute_dfa(x_full, sfreq, decimate_factor=5):
    """DFA on downsampled full signal. Returns α exponent."""
    n_min = 40
    if len(x_full) < n_min * decimate_factor: return np.nan
    try:
        x_ds = decimate(x_full, decimate_factor, zero_phase=True)
        n_ds = len(x_ds)
        nvals = np.unique(
            np.round(np.logspace(
                np.log10(4), np.log10(n_ds//4), 12)
            ).astype(int))
        nvals = nvals[(nvals >= 4) & (nvals <= n_ds // 4)]
        if len(nvals) < 4: return np.nan
        return float(nolds.dfa(x_ds, nvals=nvals, overlap=False))
    except Exception:
        return np.nan

# Validate all measures
print("\n=== MEASURE VALIDATION ===")
rng_v = np.random.default_rng(42)
n_v   = 2000
ar1_v = np.zeros(n_v)
for i in range(1, n_v): ar1_v[i] = 0.9*ar1_v[i-1] + 0.1*rng_v.standard_normal()
wn_v = rng_v.standard_normal(n_v)

checks = [
    ('AC1',    compute_ac1(ar1_v),            compute_ac1(wn_v),          'AR1>WN'),
    ('Var',    np.var(ar1_v, ddof=1),          np.var(wn_v, ddof=1),       'similar'),
    ('SampEn', ant.sample_entropy(ar1_v),      ant.sample_entropy(wn_v),   'AR1<WN'),
    ('PermEnt',ant.perm_entropy(ar1_v,3,True), ant.perm_entropy(wn_v,3,True),'AR1<WN'),
    ('HFD',    ant.higuchi_fd(ar1_v),          ant.higuchi_fd(wn_v),       'AR1<WN'),
    ('DFA_α',  compute_dfa(ar1_v,1000),        compute_dfa(wn_v,1000),     'AR1>WN'),
]
all_ok = True
for name, v_ar1, v_wn, expect in checks:
    ok = ((expect == 'AR1>WN' and v_ar1 > v_wn) or
          (expect == 'AR1<WN' and v_ar1 < v_wn) or
          (expect == 'similar'))
    sym = '✅' if ok else '⚠️'
    if not ok: all_ok = False
    print(f"  {sym} {name}: AR1={v_ar1:.4f}, WN={v_wn:.4f}  [expect {expect}]")
print(f"  {'All validated' if all_ok else 'SOME FAILED — check parameters'}")

# ── Main loop ──────────────────────────────────────────────────────────────────
print(f"\n=== PROCESSING {len(ais_rest)} TDBRAIN restEC SUBJECTS ===")
print("(2s windows, FCz, 1-40Hz filter, DFA on downsampled full signal)")

WINDOW_SEC   = 2.0
MIN_WINS     = 10
FCZ          = 'FCz'

records = []
t_start = time.time()

for idx, row in ais_rest.iterrows():
    sub_id   = row['subject_id']
    remitter = row[remit_col]
    ais_val  = row['AIS_rest']

    fpath = (TDBRAIN / sub_id / "ses-1" / "eeg" /
             f"{sub_id}_ses-1_task-restEC_eeg.vhdr")
    if not fpath.exists():
        continue

    try:
        raw   = mne.io.read_raw_brainvision(fpath, preload=True, verbose='ERROR')
        sfreq = raw.info['sfreq']

        # Minimal preprocessing: bandpass, pick FCz
        raw.filter(1., 40., method='fir', verbose='ERROR')
        if FCZ not in raw.ch_names:
            continue
        sig = raw.get_data(picks=[FCZ])[0]

        # Quality gates
        if np.std(sig) < 1e-8: continue
        if np.isnan(sig).any() or np.isinf(sig).any(): continue

        # Split into non-overlapping 2s windows
        win_len   = int(WINDOW_SEC * sfreq)
        n_wins    = len(sig) // win_len
        if n_wins < MIN_WINS: continue
        windows   = [sig[i*win_len : (i+1)*win_len] for i in range(n_wins)]

        # Per-window measures (vectorised via antropy)
        ac1_w = np.array([compute_ac1(w) for w in windows])
        var_w = np.array([np.var(w, ddof=1) for w in windows])
        se_w  = np.array([ant.sample_entropy(w) for w in windows])
        pe_w  = np.array([ant.perm_entropy(w, order=3, normalize=True) for w in windows])
        hf_w  = np.array([ant.higuchi_fd(w) for w in windows])

        def nm(arr):
            v = arr[np.isfinite(arr)]
            return float(np.mean(v)) if len(v) >= 3 else np.nan

        dfa_val = compute_dfa(sig, sfreq, decimate_factor=5)

        mean_ac1  = nm(ac1_w)
        mean_var  = nm(var_w)
        mean_se   = nm(se_w)
        mean_pe   = nm(pe_w)
        mean_hfd  = nm(hf_w)

        records.append({
            'subject_id': sub_id,
            'remitter':   remitter,
            'ais_rest':   ais_val,
            'ac1':        mean_ac1,
            'variance':   mean_var,
            'sample_entropy': mean_se,
            'perm_entropy':   mean_pe,
            'hfd':            mean_hfd,
            'dfa_alpha':      dfa_val,
            'csd_index':      mean_ac1 * mean_var,
            'n_windows':      n_wins,
        })
        if len(records) % 15 == 0:
            elapsed = time.time() - t_start
            rate    = len(records) / elapsed
            eta     = (len(ais_rest) - len(records)) / rate
            print(f"  {len(records)}/{len(ais_rest)} done  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    except Exception as e:
        print(f"  {sub_id}: {e}")

df = pd.DataFrame(records)
print(f"\nProcessed: {len(df)} subjects  "
      f"(Rem={int((df.remitter==1).sum())}, "
      f"Non-rem={int((df.remitter==0).sum())})")

# ── Cohen's d helper ──────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1))
                 / (len(a)+len(b)-2) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / sp)

# ── STEP 3: Remission prediction ──────────────────────────────────────────────
print("\n" + "="*60)
print("REMISSION PREDICTION (all measures)")
print("Reference: AIS_rest d=−0.336, p=0.031  [Non-rem > Rem]")
print("="*60)

rem  = df[df.remitter == 1]
nrem = df[df.remitter == 0]

MEASURES = ['ac1', 'variance', 'sample_entropy', 'perm_entropy',
            'hfd', 'dfa_alpha', 'csd_index', 'ais_rest']

results = []
for m in MEASURES:
    rv = rem[m].dropna();  nv = nrem[m].dropna()
    if len(rv) < 5 or len(nv) < 5: continue
    _, p = mannwhitneyu(rv, nv, alternative='two-sided')
    d    = cohens_d(rv.values, nv.values)
    dirn = 'Non-rem>Rem' if nv.mean() > rv.mean() else 'Rem>Non-rem'
    _, p_s = spearmanr(df[m].dropna(), df.loc[df[m].notna(), 'remitter'])
    results.append({'measure': m, 'Rem': round(rv.mean(), 5),
                    'Non-rem': round(nv.mean(), 5), 'd': round(d, 3),
                    'p_mwu': round(p, 4), 'direction': dirn,
                    'N_rem': len(rv), 'N_nrem': len(nv)})

df_res = pd.DataFrame(results).sort_values('p_mwu')
print()
print(df_res.to_string(index=False))

beats_ais = df_res[df_res.p_mwu < 0.031]
print(f"\nMeasures beating AIS_rest reference (p<0.031): "
      f"{beats_ais['measure'].tolist() if len(beats_ais) else 'none'}")

# ── STEP 4: Independence from AIS_rest ────────────────────────────────────────
print("\n" + "="*60)
print("INDEPENDENCE FROM AIS_rest")
print("Key: |r|>0.7 → redundant; |r|<0.3 → independent")
print("="*60)

partial_rows = []
for m in MEASURES:
    if m == 'ais_rest': continue
    pair = df[[m, 'ais_rest', 'remitter']].dropna()
    if len(pair) < 15: continue

    r_raw, p_raw = pearsonr(pair[m], pair.ais_rest)

    # Partial correlation: m ~ remitter | ais_rest
    X = np.column_stack([np.ones(len(pair)), pair.ais_rest.values])
    b_m, *_ = lstsq(X, pair[m].values, rcond=None)
    resid_m = pair[m].values - X @ b_m
    b_r, *_ = lstsq(X, pair.remitter.values.astype(float), rcond=None)
    resid_r = pair.remitter.values - X @ b_r
    r_part, p_part = pearsonr(resid_m, resid_r)

    partial_rows.append({'measure': m,
                         'r_with_AIS': round(r_raw, 3),
                         'p_r_AIS':    round(p_raw, 4),
                         'partial_r_remit|AIS': round(r_part, 3),
                         'partial_p':  round(p_part, 4),
                         'R2_shared':  round(r_raw**2, 3),
                         'N': len(pair)})
    print(f"  {m:<18}  r(m,AIS)={r_raw:+.3f}  "
          f"partial_r(remit)={r_part:+.3f} p={p_part:.4f}")

df_part = pd.DataFrame(partial_rows)

# Independent measures: |r(m,AIS)| < 0.3 AND partial still significant
independent = df_part[(df_part.r_with_AIS.abs() < 0.3) &
                      (df_part.partial_p < 0.05)]
print(f"\nIndependent of AIS_rest AND significant (partial):")
print(independent[['measure','r_with_AIS','partial_r_remit|AIS','partial_p']].to_string(index=False)
      if len(independent) else "  None found")

# ── STEP 5: CSD signature ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("CRITICAL SLOWING DOWN SIGNATURE")
print("="*60)

for label, grp in [('Remitters', rem), ('Non-remitters', nrem)]:
    ac1_m = grp['ac1'].mean()
    var_m = grp['variance'].mean()
    csd_m = grp['csd_index'].mean()
    se_m  = grp['sample_entropy'].mean()
    dfa_m = grp['dfa_alpha'].mean()
    print(f"\n  {label} (N={len(grp)}):")
    print(f"    AC1={ac1_m:.4f}  Var={var_m:.3e}  CSD={csd_m:.3e}")
    print(f"    SampEn={se_m:.4f}  DFA-α={dfa_m:.4f}")

d_ac1 = cohens_d(nrem.ac1.dropna().values, rem.ac1.dropna().values)
d_var = cohens_d(nrem.variance.dropna().values, rem.variance.dropna().values)
d_dfa = cohens_d(nrem.dfa_alpha.dropna().values, rem.dfa_alpha.dropna().values)

print(f"\n  d(AC1)  Non-rem vs Rem = {d_ac1:+.3f}")
print(f"  d(Var)  Non-rem vs Rem = {d_var:+.3f}")
print(f"  d(DFA)  Non-rem vs Rem = {d_dfa:+.3f}")

if d_ac1 > 0.15 and d_var > 0.15:
    print("\n  → BOTH AC1 and Var elevated in Non-remitters")
    print("    CSD SIGNATURE PRESENT: near-critical attractor state")
elif d_ac1 > 0.15 and d_var < 0.05:
    print("\n  → AC1 elevated but Var NOT elevated in Non-remitters")
    print("    DEEP ATTRACTOR: rigid but stable, far from bifurcation")
elif d_dfa > 0.3:
    print("\n  → DFA-α elevated: longer memory = stronger correlations")
    print("    Long-range temporal rigidity in non-remitters")
else:
    print("\n  → Mixed/null CSD pattern")

# ── Save results ──────────────────────────────────────────────────────────────
out = TDBRAIN / "derivatives"
out.mkdir(exist_ok=True)
df.to_csv(out / "tdbrain_chaos_measures.csv", index=False)
df_res.to_csv(out / "tdbrain_chaos_results.csv", index=False)
df_part.to_csv(out / "tdbrain_chaos_partial.csv", index=False)
print(f"\nSaved to {out}")

# ── Update frozen state ───────────────────────────────────────────────────────
top_measure = df_res.iloc[0]['measure'] if len(df_res) else None
state['chaos_analysis'] = {
    'date': '2026-05-03',
    'dataset': 'TDBRAIN restEC, FCz',
    'N_rem': int((df.remitter==1).sum()),
    'N_nrem': int((df.remitter==0).sum()),
    'measures_tested': MEASURES,
    'best_measure': top_measure,
    'best_d': float(df_res.iloc[0]['d']) if len(df_res) else None,
    'best_p': float(df_res.iloc[0]['p_mwu']) if len(df_res) else None,
    'independent_of_ais': independent['measure'].tolist() if len(independent) else [],
    'csd_d_ac1': round(d_ac1, 3), 'csd_d_var': round(d_var, 3),
    'full_results': df_res.to_dict(orient='records'),
}
with open(freeze_file, 'w') as f:
    json.dump(state, f, indent=2)
print("Frozen state updated.")
