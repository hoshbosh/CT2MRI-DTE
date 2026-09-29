"""Phase 2 sanity check: how much of the reported Dice is processing noise?

The Phase 2 baseline reports Dice between SynthSeg-on-real-MR and
SynthSeg-on-synthetic-MR and gets 0.909 for the thalamus. Read naively that is
a good number. But SynthSeg is heavily prior-driven: it will happily snap to a
plausible thalamus from a mediocre image, so a high Dice may say more about the
segmenter's prior than about the synthesis. If so, Dice is pinned near its
ceiling and cannot rank one checkpoint above another.

The way to tell is to measure the ceiling. Take two label maps that describe
the SAME underlying anatomy and differ only in how they were produced:

  seg/<pid>_real_synthseg.nii.gz   SynthSeg run directly on the real MR, in
                                   evaluation space (raw aseg codes)
  <pid>_labels_phase1.nii.gz       SynthSeg run on the NATIVE-grid MR, then
                                   transported onto this grid by the Phase 1
                                   geometry chain (compact codes 1..12)

No synthesis is involved in that pair. Whatever Dice they disagree by is the
noise floor: resampling, interpolation, grid differences, and run-to-run
SynthSeg variability. Call it dice_floor.

dice_syn -- the already-reported real-vs-synthetic Dice -- is recomputed here
so both land in one table. The quantity of interest is the headroom,
dice_floor - dice_syn. Near zero means the metric is saturated and a better
model cannot show up in it. Clearly positive (say 0.97 against 0.91) means the
synthesis gap is real and there is room to improve.

Fails loudly on anything missing or mis-shaped. A quietly skipped subject or a
structure defaulted to zero would bias the very number this check exists to
establish.

Usage:
    python runners/check_dice_noise_floor.py \
        --export_dir /blue/.../phase2/export \
        --manifest   ./datasets/label_qc/label_manifest.csv
"""
import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_dataset_utils.geometry import (  # noqa: E402
    ASEG_LABEL_MAP,
    BILATERAL_GROUPS,
    STRUCTURE_NAMES,
)
from brain_dataset_utils.make_labels import load_manifest  # noqa: E402

# compact code -> the aseg code SynthSeg actually writes. Same inversion as
# eval_structures.py, so the dice_syn column here is comparable to the numbers
# already reported there rather than a second, subtly different computation.
CODE_TO_ASEG = {v: k for k, v in ASEG_LABEL_MAP.items()}


def dice(m1, m2):
    s = m1.sum() + m2.sum()
    if s == 0:
        return np.nan
    return float(2.0 * np.logical_and(m1, m2).sum() / s)


def load_int(path, pid, what):
    if not os.path.exists(path):
        raise SystemExit(
            f'{pid}: missing {what} at {path}. This check compares three label '
            f'maps per subject and reports a cohort-level floor; refusing to '
            f'run on a partial cohort.'
        )
    return np.asanyarray(nib.load(path).dataobj).astype(np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--export_dir', required=True,
                    help='Output of export_eval_volumes.py (holds subjects.csv and seg/)')
    ap.add_argument('--manifest', default=None, help='Phase 1 QC manifest')
    ap.add_argument('--out_csv', default=None,
                    help='Per-subject-per-structure CSV (default: <export_dir>/dice_noise_floor.csv)')
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    seg_dir = os.path.join(args.export_dir, 'seg')
    index_path = os.path.join(args.export_dir, 'subjects.csv')
    out_csv = args.out_csv or os.path.join(args.export_dir, 'dice_noise_floor.csv')

    if not os.path.exists(index_path):
        raise SystemExit(f'No subject index at {index_path}')
    with open(index_path) as fh:
        subjects = list(csv.DictReader(fh))
    if not subjects:
        raise SystemExit(f'No subjects listed in {index_path}')

    rows = []
    for entry in subjects:
        pid = entry['pid']
        _, excluded = manifest.get(pid, ('default', frozenset()))

        real_path = os.path.join(seg_dir, f'{pid}_real_synthseg.nii.gz')
        syn_path = os.path.join(seg_dir, f'{pid}_syn_synthseg.nii.gz')
        phase1_path = os.path.join(args.export_dir, f'{pid}_labels_phase1.nii.gz')

        seg_real = load_int(real_path, pid, 'SynthSeg-on-real segmentation')
        seg_syn = load_int(syn_path, pid, 'SynthSeg-on-synthetic segmentation')
        labels_phase1 = load_int(phase1_path, pid, 'Phase 1 transported labels')

        # All three are supposed to be on one evaluation grid. If they are not,
        # voxelwise overlap is meaningless and any "floor" derived from it is an
        # artefact of the mismatch, so stop rather than report it.
        if not (seg_real.shape == seg_syn.shape == labels_phase1.shape):
            raise SystemExit(
                f'{pid}: shape mismatch -- '
                f'{real_path} {seg_real.shape}, '
                f'{syn_path} {seg_syn.shape}, '
                f'{phase1_path} {labels_phase1.shape}'
            )

        for code in sorted(STRUCTURE_NAMES):
            name = STRUCTURE_NAMES[code]
            aseg = CODE_TO_ASEG[code]

            # seg_real and seg_syn carry raw aseg values; the Phase 1 volume was
            # remapped to the compact 1..12 coding by make_labels.py.
            mask_real = seg_real == aseg
            mask_phase1 = labels_phase1 == code
            mask_syn = seg_syn == aseg

            row = {
                'pid': pid,
                'structure': name,
                'n_vox_real': int(mask_real.sum()),
                'n_vox_phase1': int(mask_phase1.sum()),
                'n_vox_syn': int(mask_syn.sum()),
            }

            if code in excluded:
                # QC rejected this structure for this subject in Phase 1, so the
                # phase1 arm has no trustworthy reference. Recorded, not dropped.
                row.update(dice_floor=np.nan, dice_syn=np.nan,
                           status='excluded_manifest')
            else:
                row.update(dice_floor=dice(mask_real, mask_phase1),
                           dice_syn=dice(mask_real, mask_syn),
                           status='ok')
            rows.append(row)

        print(f'  {pid}  12 structures compared', flush=True)

    fields = ['pid', 'structure', 'dice_floor', 'dice_syn',
              'n_vox_real', 'n_vox_phase1', 'n_vox_syn']
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    report(rows, out_csv, len(subjects))


def report(rows, out_csv, n_subjects):
    def stat(sel, key):
        vals = [r[key] for r in sel
                if isinstance(r[key], float) and np.isfinite(r[key])]
        if not vals:
            return np.nan, np.nan, 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    print(f'\nPer-structure results written to {out_csv}')
    print(f'\nNoise floor vs synthesis Dice, N={n_subjects} subjects:')
    print(f'{"structure":14s} {"dice_floor":>18s} {"dice_syn":>18s} '
          f'{"headroom":>9s} {"n":>4s}')

    floors, syns = [], []
    for group, members in BILATERAL_GROUPS.items():
        names = {STRUCTURE_NAMES[m] for m in members}
        sel = [r for r in rows if r['structure'] in names and r['status'] == 'ok']
        if not sel:
            continue
        fm, fs, nf = stat(sel, 'dice_floor')
        sm, ss, ns = stat(sel, 'dice_syn')
        # Only groups that actually scored contribute to the overall mean; a
        # nan from an entirely excluded group would otherwise poison it.
        if np.isfinite(fm) and np.isfinite(sm):
            floors.append(fm)
            syns.append(sm)
        print(f'{group:14s} {fm:9.3f} +/-{fs:5.3f} {sm:9.3f} +/-{ss:5.3f} '
              f'{fm - sm:9.3f} {min(nf, ns):4d}')

    if not floors:
        raise SystemExit('No structures were scorable; nothing to interpret.')

    overall_floor = float(np.mean(floors))
    overall_syn = float(np.mean(syns))
    gap = overall_floor - overall_syn

    print(f'\n{"MEAN":14s} {overall_floor:9.3f} {"":11s}{overall_syn:9.3f} '
          f'{"":11s}{gap:9.3f}')

    print('\nHow to read this:')
    print('  dice_floor  real-MR SynthSeg vs Phase 1 transported labels. Same')
    print('              anatomy, no synthesis -- pure processing-chain and')
    print('              SynthSeg run-to-run disagreement. This is the ceiling')
    print('              any synthesis Dice could possibly reach.')
    print('  dice_syn    real-MR SynthSeg vs synthetic-MR SynthSeg, the number')
    print('              the Phase 2 baseline reports.')
    print('  headroom    dice_floor - dice_syn.')
    print('')
    print('  Decision rule: if dice_floor is close to dice_syn, Dice is')
    print('  saturated by processing noise and cannot discriminate synthesis')
    print('  quality here -- a better model has nowhere to move the number. If')
    print('  dice_floor is substantially higher (e.g. ~0.97 against ~0.91), the')
    print('  synthesis gap is real and there is headroom for improvement.')
    print('')
    print(f'  Observed: dice_floor {overall_floor:.3f}, dice_syn '
          f'{overall_syn:.3f}, headroom {gap:.3f}.')


if __name__ == '__main__':
    main()
