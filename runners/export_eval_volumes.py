"""Phase 2, step 1: export real and synthetic MR volumes for SynthSeg.

`sample_to_eval` writes `{pid}.nii` with an identity affine, which throws away
the voxel spacing. SynthSeg is resolution-aware and centroid displacement is
reported in mm, so both need the true spacing from the geometry record.

Both the real and the synthetic volume are exported and segmented, rather than
reusing the Phase 1 labels as the reference. The Phase 1 labels came from
native-grid SynthSeg followed by a transport; running SynthSeg on the real
volume in *this* space instead means reference and prediction traverse an
identical path, so Dice measures synthesis error rather than the difference
between two processing chains. The transported labels are still written out
alongside, as an independent cross-check.

Slice alignment is asserted, not assumed: the HDF5 and the sampler both drop
blank slices via the same `filter_blank_slices_thick` selection, so a subject
whose synthetic volume has a different slice count means the sample directory
was produced from a different HDF5 build and the comparison would be silently
misaligned.
"""
import argparse
import os
import sys
from collections import OrderedDict

import h5py
import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_dataset_utils.geometry import GeometryParams  # noqa: E402
from brain_dataset_utils.make_labels import load_manifest  # noqa: E402


# transpose_LPS_to_ITKSNAP_position applies this axis permutation per plane.
# Only axial is supported here: the coronal and sagittal paths also flip and
# rot90, and deriving those by hand is exactly the kind of guess that produced
# an unparseable volume the first time.
PLANE_PERM = {'axial': [1, 0, 2]}


def evaluation_affine(params, plane):
    """World-space affine for a volume on the evaluation grid.

    SynthSeg is orientation-aware: it uses the affine to decide where anatomy
    is. A plain diagonal affine asserts RAS, which is wrong for these volumes --
    they have been through the in-plane resize and the axial transpose -- and
    SynthSeg responded by finding cortex and CSF but no subcortical structures
    at all. This composes the real transform instead:

      1. crop_affine is RAS for the resampled+cropped, PRE-resize volume, so its
         columns carry target_spacing. Rescale each to the true post-resize
         spacing_mm.
      2. transpose_LPS_to_ITKSNAP_position permutes the axes, so permute the
         affine's columns to match. The origin voxel is unchanged, so the
         translation carries over untouched.
    """
    if plane not in PLANE_PERM:
        raise ValueError(
            f"no affine derivation for plane {plane!r}; only "
            f"{sorted(PLANE_PERM)} is implemented"
        )
    A = np.asarray(params.crop_affine, dtype=np.float64).copy()
    ts = float(params.target_spacing)
    for i, s in enumerate(params.spacing_mm):
        A[:3, i] *= float(s) / ts

    out = A.copy()
    out[:3, :3] = A[:3, PLANE_PERM[plane]]
    return out


def group_by_subject(subjects):
    """Contiguous slice ranges per subject, in file order."""
    groups = OrderedDict()
    for i, raw in enumerate(subjects):
        pid = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
        if pid not in groups:
            groups[pid] = [i, i]
        else:
            if i != groups[pid][1] + 1:
                raise RuntimeError(
                    f"{pid}: slices are not contiguous in the HDF5 (index {i} follows "
                    f"{groups[pid][1]}). The volume assembly below assumes file order."
                )
            groups[pid][1] = i
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hdf5', required=True, help='Test HDF5 (must carry LABEL_dataset)')
    ap.add_argument('--sample_dir', required=True,
                    help='sample_to_eval output containing <pid>.nii')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--geom_dir', required=True,
                    help='out-fine directory holding <pid>/geometry.json, used to '
                         'reconstruct the world affine the evaluation grid sits in')
    ap.add_argument('--plane', default='axial')
    ap.add_argument('--manifest', default=None,
                    help='Phase 1 QC manifest; recorded per subject so the SynthSeg '
                         'array can use the same mode the reference labels used')
    ap.add_argument('--syn_suffix', default='.nii',
                    help="Synthetic filename suffix; '_mean.nii' for a UQ ensemble run")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.hdf5, 'r') as hf:
        for name in ('MR_dataset', 'LABEL_dataset', 'spacing_dataset', 'subject'):
            if name not in hf:
                raise SystemExit(
                    f"{args.hdf5} has no {name}. Phase 2 needs the fine_v2 build; "
                    "the older HDF5s carry neither labels nor spacing."
                )
        subjects = hf['subject'][:]
        groups = group_by_subject(subjects)

        mr = hf['MR_dataset']
        lab = hf['LABEL_dataset']
        spacing = hf['spacing_dataset'][:]

        rows, failures = [], []
        for pid, (lo, hi) in groups.items():
            n = hi - lo + 1
            syn_path = os.path.join(args.sample_dir, f'{pid}{args.syn_suffix}')
            if not os.path.exists(syn_path):
                failures.append(f'{pid}: no synthetic volume at {syn_path}')
                continue

            syn = np.asanyarray(nib.load(syn_path).dataobj).astype(np.float32)
            real = np.asarray(mr[:, :, lo:hi + 1], dtype=np.float32)
            labels = np.asarray(lab[:, :, lo:hi + 1], dtype=np.uint8)

            if syn.shape != real.shape:
                failures.append(
                    f'{pid}: synthetic volume is {syn.shape} but the HDF5 holds '
                    f'{real.shape} for this subject -- the sample directory was built '
                    f'from a different HDF5 and the slices do not correspond'
                )
                continue

            sp = spacing[lo:hi + 1]
            if not np.allclose(sp, sp[0], equal_nan=True):
                failures.append(f'{pid}: spacing varies across slices, which should be impossible')
                continue
            sx, sy, sz = (float(v) for v in sp[0])
            if not np.isfinite([sx, sy, sz]).all():
                failures.append(f'{pid}: spacing is not finite ({sx}, {sy}, {sz})')
                continue

            geom_path = os.path.join(args.geom_dir, pid, 'geometry.json')
            if not os.path.exists(geom_path):
                failures.append(f'{pid}: no geometry.json at {geom_path}')
                continue
            params = GeometryParams.from_json(geom_path)
            affine = evaluation_affine(params, args.plane)

            # The affine must agree with the spacing the HDF5 recorded, or one of
            # the two is describing a different volume.
            got = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
            if not np.allclose(got, [sx, sy, sz], atol=1e-4):
                failures.append(
                    f'{pid}: affine spacing {got.round(4).tolist()} disagrees with '
                    f'spacing_dataset ({sx:.4f}, {sy:.4f}, {sz:.4f})'
                )
                continue
            nib.save(nib.Nifti1Image(real, affine),
                     os.path.join(args.out_dir, f'{pid}_real.nii.gz'))
            nib.save(nib.Nifti1Image(syn, affine),
                     os.path.join(args.out_dir, f'{pid}_syn.nii.gz'))
            nib.save(nib.Nifti1Image(labels, affine),
                     os.path.join(args.out_dir, f'{pid}_labels_phase1.nii.gz'))

            mode, excluded = manifest.get(pid, ('default', frozenset()))
            rows.append((pid, n, sx, sy, sz, mode, len(excluded)))
            print(f'  OK   {pid}  slices={n:4d}  spacing=({sx:.3f},{sy:.3f},{sz:.3f})  '
                  f'mode={mode}' + (f'  excluded={len(excluded)}' if excluded else ''),
                  flush=True)

    index_path = os.path.join(args.out_dir, 'subjects.csv')
    with open(index_path, 'w') as fh:
        fh.write('pid,slices,spacing_x,spacing_y,spacing_z,synthseg_mode,n_excluded\n')
        for r in rows:
            fh.write(','.join(str(x) for x in r) + '\n')

    print(f'\nExported {len(rows)} subjects to {args.out_dir}')
    print(f'Subject index: {index_path}')
    if failures:
        print('\nFAILURES:')
        for msg in failures:
            print(f'  {msg}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
