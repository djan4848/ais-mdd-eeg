# 03_make_evokeds_diff.py  (DDS-Hayling)
import mne
from pathlib import Path

from dds_base.io.paths import DERIV_ROOT, hayling_epo_files

COND_A = "ASOC"
COND_B = "NOASOC"
BASELINE = (-0.2, 0.0)
DROP = {"M1", "M2"}  # mastoides

OUTDIR = DERIV_ROOT / "evokeds"
OUTDIR.mkdir(exist_ok=True, parents=True)

LOG_SKIPPED = OUTDIR / "skipped_subjects_empty_conditions.txt"

OUT_GRAND_A = OUTDIR / "grand_evoked_asoc-ave.fif"
OUT_GRAND_B = OUTDIR / "grand_evoked_noasoc-ave.fif"
OUT_GRAND_DIFF = OUTDIR / "grand_evoked_diff_asoc_minus_noasoc-ave.fif"
OUT_GRAND_DIFF_NOM = OUTDIR / "grand_evoked_diff_asoc_minus_noasoc_noM-ave.fif"


def _drop_mastoids_if_present(ev: mne.Evoked) -> mne.Evoked:
    to_drop = [ch for ch in DROP if ch in ev.ch_names]
    return ev.copy().drop_channels(to_drop) if to_drop else ev


def main():
    files = hayling_epo_files()
    if not files:
        raise RuntimeError("No Hayling *-epo.fif files found via hayling_epo_files(). Check paths.py.")

    per_subj_a = []
    per_subj_b = []
    per_subj_diff = []

    skipped = []
    EXCLUDE_SUBJECTS = {"P19"}  # NOASOC vacío (len_NOASOC=0)

    files = [f for f in files if f.parent.name not in EXCLUDE_SUBJECTS]
    for f in files:
        subj = f.parent.name  # e.g., P10

        epochs = mne.read_epochs(f, preload=True, verbose="ERROR")
        epochs.apply_baseline(BASELINE)

        # Ensure conditions exist in event_id (names)
        missing_names = [c for c in (COND_A, COND_B) if c not in epochs.event_id]
        if missing_names:
            skipped.append(f"{subj}\t{f.name}\tmissing_in_event_id={missing_names}")
            continue

        ep_a = epochs[COND_A]
        ep_b = epochs[COND_B]

        # Guard: empty after selection
        if len(ep_a) == 0 or len(ep_b) == 0:
            skipped.append(f"{subj}\t{f.name}\tlen_ASOC={len(ep_a)}\tlen_NOASOC={len(ep_b)}")
            continue

        ev_a = ep_a.average()
        ev_b = ep_b.average()
        ev_diff = mne.combine_evoked([ev_a, ev_b], weights=[1, -1])  # ASOC - NOASOC

        # Save per-subject evokeds
        out_a = OUTDIR / f"{subj}_evoked_asoc-ave.fif"
        out_b = OUTDIR / f"{subj}_evoked_noasoc-ave.fif"
        out_d = OUTDIR / f"{subj}_evoked_diff_asoc_minus_noasoc-ave.fif"
        out_dn = OUTDIR / f"{subj}_evoked_diff_asoc_minus_noasoc_noM-ave.fif"

        mne.write_evokeds(out_a, ev_a, overwrite=True)
        mne.write_evokeds(out_b, ev_b, overwrite=True)
        mne.write_evokeds(out_d, ev_diff, overwrite=True)

        ev_diff_nom = _drop_mastoids_if_present(ev_diff)
        mne.write_evokeds(out_dn, ev_diff_nom, overwrite=True)

        per_subj_a.append(ev_a)
        per_subj_b.append(ev_b)
        per_subj_diff.append(ev_diff)

    # Write skip log
    if skipped:
        LOG_SKIPPED.write_text("\n".join(skipped) + "\n")
        print(f"[warn] skipped {len(skipped)} subjects (empty/missing conditions). Log: {LOG_SKIPPED}")
    else:
        if LOG_SKIPPED.exists():
            LOG_SKIPPED.unlink()

    if len(per_subj_a) == 0:
        raise RuntimeError("No valid subjects left after filtering. Check skipped log.")

    grand_a = mne.grand_average(per_subj_a)
    grand_b = mne.grand_average(per_subj_b)
    grand_diff = mne.combine_evoked([grand_a, grand_b], weights=[1, -1])
    grand_diff_nom = _drop_mastoids_if_present(grand_diff)
    grand_a.nave = len(per_subj_a)
    grand_b.nave = len(per_subj_b)
    grand_diff.nave = len(per_subj_diff)
    grand_diff_nom.nave = len(per_subj_diff)
    mne.write_evokeds(OUT_GRAND_A, grand_a, overwrite=True)
    mne.write_evokeds(OUT_GRAND_B, grand_b, overwrite=True)
    mne.write_evokeds(OUT_GRAND_DIFF, grand_diff, overwrite=True)
    mne.write_evokeds(OUT_GRAND_DIFF_NOM, grand_diff_nom, overwrite=True)

    print("[ok] total files:", len(files))
    print("[ok] valid subjects:", len(per_subj_a))
    print("[ok] per-subject evokeds in:", OUTDIR)
    print("[ok] saved grand:", OUT_GRAND_A.name, OUT_GRAND_B.name, OUT_GRAND_DIFF.name, OUT_GRAND_DIFF_NOM.name)
    print("[ok] excluded subjects:", sorted(EXCLUDE_SUBJECTS))

if __name__ == "__main__":
    main()
