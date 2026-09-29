"""Boundary and centroid error per structure, beyond what Dice can say.

Dice is volumetric overlap: a structure whose outline is off by a consistent
rind reports one number, with no indication of how far the boundary actually
moved or in which direction. This measures the surface directly and decomposes
the centroid error into signed components, which separates two failure modes
Dice conflates:

  * a systematic shift or size bias -- the structure is drawn in the wrong
    place or at the wrong scale, and the centroid moves with the boundary
  * random boundary noise -- the outline is ragged but centred, so surface
    distance is large while the centroid barely moves

Metrics, per structure per side (never bilaterally pooled, since a systematic
shift can cancel between left and right):

  assd_mm      average symmetric surface distance: mean over both surfaces of
               the distance from each boundary voxel to the nearest boundary
               voxel of the other. The honest 'how far did the boundary move'
  hd95_mm      95th-percentile Hausdorff distance -- worst-case boundary error,
               with the top 5% trimmed so a single stray voxel cannot dominate
  centroid_dx/dy/dz_mm
               SIGNED per-axis centroid displacement, synthetic minus real, in
               world axes. Averaged across subjects a systematic bias survives
               and random error cancels, which is the test for directional bias
  vol_ratio    synthetic volume / real volume. <1 means the model's structures
               are systematically shrunk, >1 dilated

Surfaces are extracted with a 6-connected erosion so a boundary voxel is one
with at least one face-neighbour outside the mask, and distances are computed
on the anisotropic voxel grid via the sampling argument -- these volumes have
~0.9mm in-plane and 1.0mm through-plane spacing, so an isotropic distance
transform would misreport every number here.
"""
import argparse
import csv
import functools
import multiprocessing as mp
import os
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_dataset_utils.geometry import (  # noqa: E402
    ASEG_LABEL_MAP,
    STRUCTURE_NAMES,
)
from brain_dataset_utils.make_labels import load_manifest  # noqa: E402

CODE_TO_ASEG = {v: k for k, v in ASEG_LABEL_MAP.items()}

# 6-connectivity: a surface voxel is one with a face-neighbour outside the mask.
_STRUCT = np.zeros((3, 3, 3), dtype=bool)
_STRUCT[1, 1, :] = True
_STRUCT[1, :, 1] = True
_STRUCT[:, 1, 1] = True


def surface(mask):
    return mask & ~binary_erosion(mask, structure=_STRUCT, border_value=0)


def _union_box(a, b, pad=2):
    """Bounding box containing both masks, padded.

    The distance transform is the expensive step and scales with the volume it
    runs on, so running it over the full 256x256x173 grid for a structure of a
    few thousand voxels is almost all wasted work. Cropping to a box that
    contains BOTH surfaces is exact, not an approximation: every distance
    measured is from a voxel of `a` to a voxel of `b`, and both lie inside.
    """
    idx = np.argwhere(a | b)
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + pad + 1, a.shape)
    return tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))


def surface_distances(a, b, spacing, box=None):
    """Distances from each surface voxel of `a` to the nearest surface of `b`."""
    if box is None:
        box = _union_box(a, b)
    a, b = a[box], b[box]
    sb = surface(b)
    sa = surface(a)
    if not sb.any() or not sa.any():
        return None
    # distance_transform_edt gives distance to the nearest ZERO, so invert.
    dt = distance_transform_edt(~sb, sampling=spacing)
    return dt[sa]


def centroid(mask, spacing):
    idx = np.argwhere(mask).astype(np.float64)
    return idx.mean(axis=0) * np.asarray(spacing, dtype=np.float64)


def process_subject(entry, seg_dir, manifest):
    """All structures for one subject. Module-level so Pool can pickle it."""
    pid = entry['pid']
    spacing = (float(entry['spacing_x']), float(entry['spacing_y']),
               float(entry['spacing_z']))
    _, excluded = manifest.get(pid, ('default', frozenset()))

    paths = {k: os.path.join(seg_dir, f'{pid}_{k}_synthseg.nii.gz')
             for k in ('real', 'syn')}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(f'{pid}: missing {k} segmentation at {p}')

    seg_real = np.asanyarray(nib.load(paths['real']).dataobj).astype(np.int32)
    seg_syn = np.asanyarray(nib.load(paths['syn']).dataobj).astype(np.int32)
    if seg_real.shape != seg_syn.shape:
        raise SystemExit(f'{pid}: shape mismatch {seg_real.shape} vs {seg_syn.shape}')

    out = []
    for code in sorted(STRUCTURE_NAMES):
        name = STRUCTURE_NAMES[code]
        if code in excluded:
            continue
        aseg = CODE_TO_ASEG[code]
        mr, ms = seg_real == aseg, seg_syn == aseg
        if not mr.any() or not ms.any():
            # Recorded, not silently skipped: an absent structure is a real
            # failure and eval_structures.py already scores it as dice 0.
            out.append({'pid': pid, 'structure': name, 'status': 'missing',
                        'assd_mm': np.nan, 'hd95_mm': np.nan,
                        'centroid_dx_mm': np.nan, 'centroid_dy_mm': np.nan,
                        'centroid_dz_mm': np.nan, 'centroid_mm': np.nan,
                        'vol_ratio': np.nan})
            continue

        # One box per structure, shared by both directions.
        box = _union_box(mr, ms)
        d_rs = surface_distances(mr, ms, spacing, box=box)
        d_sr = surface_distances(ms, mr, spacing, box=box)
        assd = float((d_rs.mean() + d_sr.mean()) / 2.0)
        hd95 = float(max(np.percentile(d_rs, 95), np.percentile(d_sr, 95)))

        cr, cs = centroid(mr, spacing), centroid(ms, spacing)
        d = cs - cr

        out.append({
            'pid': pid, 'structure': name, 'status': 'ok',
            'assd_mm': assd, 'hd95_mm': hd95,
            'centroid_dx_mm': float(d[0]), 'centroid_dy_mm': float(d[1]),
            'centroid_dz_mm': float(d[2]),
            'centroid_mm': float(np.linalg.norm(d)),
            'vol_ratio': float(ms.sum() / mr.sum()),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--export_dir', required=True)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--workers', type=int, default=1,
                    help='subjects processed in parallel; each is independent')
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    seg_dir = os.path.join(args.export_dir, 'seg')
    out_csv = args.out_csv or os.path.join(args.export_dir, 'structure_boundary.csv')

    with open(os.path.join(args.export_dir, 'subjects.csv')) as fh:
        subjects = list(csv.DictReader(fh))

    work = functools.partial(process_subject, seg_dir=seg_dir, manifest=manifest)
    rows = []
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(work, subjects), 1):
                rows.extend(res)
                print(f'  {i}/{len(subjects)} subjects done', flush=True)
    else:
        for i, entry in enumerate(subjects, 1):
            rows.extend(work(entry))
            print(f'  {i}/{len(subjects)} subjects done', flush=True)

    fields = ['pid', 'structure', 'status', 'assd_mm', 'hd95_mm',
              'centroid_dx_mm', 'centroid_dy_mm', 'centroid_dz_mm',
              'centroid_mm', 'vol_ratio']
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    report(rows, out_csv)


def report(rows, out_csv):
    ok = [r for r in rows if r['status'] == 'ok']
    print(f'\nPer-structure boundary metrics written to {out_csv}')

    print('\nBoundary error, per side (mm):')
    print(f'{"structure":16s} {"assd":>13s} {"hd95":>13s} {"centroid":>13s} '
          f'{"vol ratio":>13s}  {"n":>3s}')
    for code in sorted(STRUCTURE_NAMES):
        name = STRUCTURE_NAMES[code]
        sel = [r for r in ok if r['structure'] == name]
        if not sel:
            continue
        def ms(key):
            v = np.array([r[key] for r in sel], dtype=float)
            return f'{v.mean():6.2f}+/-{v.std():4.2f}'
        print(f'{name:16s} {ms("assd_mm"):>13s} {ms("hd95_mm"):>13s} '
              f'{ms("centroid_mm"):>13s} {ms("vol_ratio"):>13s}  {len(sel):3d}')

    print('\nSigned centroid displacement (synthetic minus real, mm).')
    print('A mean far from 0 relative to its own std is a SYSTEMATIC shift;')
    print('a mean near 0 with a large std is random error that cancels.')
    print(f'{"structure":16s} {"dx":>14s} {"dy":>14s} {"dz":>14s}')
    for code in sorted(STRUCTURE_NAMES):
        name = STRUCTURE_NAMES[code]
        sel = [r for r in ok if r['structure'] == name]
        if not sel:
            continue
        cells = []
        for key in ('centroid_dx_mm', 'centroid_dy_mm', 'centroid_dz_mm'):
            v = np.array([r[key] for r in sel], dtype=float)
            cells.append(f'{v.mean():+6.2f}+/-{v.std():4.2f}')
        print(f'{name:16s} ' + ' '.join(f'{c:>14s}' for c in cells))


if __name__ == '__main__':
    main()
