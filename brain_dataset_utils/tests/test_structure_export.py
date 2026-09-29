"""Guards for the per-structure NIfTI export and the voxel-change columns.

End-to-end: builds a synthetic export_dir where every structure is a 6x6x6 cube
in the real segmentation and a 5x6x6 cube shifted one voxel in x in the
synthetic one, runs eval_structures.py over it, and checks the three requested
quantities against their analytic values:

    dice      = 2*180/(216+180) = 0.9091
    vol_ratio = 180/216         = 0.8333
    centroid  = 0.5 voxel * 0.8 mm = 0.40 mm

A dumped mask is also reloaded to confirm it is binary and carries the real
voxel spacing -- a mask written with an identity affine would overlay wrongly
in a viewer, which is the failure mode that cost Phase 2 a SynthSeg run.
"""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import nibabel as nib

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from brain_dataset_utils.geometry import ASEG_LABEL_MAP, STRUCTURE_NAMES  # noqa: E402

SPACING = (0.8, 0.8, 1.2)
SHAPE = (64, 64, 24)
PID = 'TEST001'
N_REAL, N_SYN = 6 * 6 * 6, 5 * 6 * 6


def build_fixture(root):
    seg_dir = os.path.join(root, 'seg')
    os.makedirs(seg_dir, exist_ok=True)
    affine = np.diag(list(SPACING) + [1.0])

    seg_real = np.zeros(SHAPE, np.int32)
    seg_syn = np.zeros(SHAPE, np.int32)
    code_to_aseg = {v: k for k, v in ASEG_LABEL_MAP.items()}

    for i, code in enumerate(sorted(STRUCTURE_NAMES)):
        aseg = code_to_aseg[code]
        x = 4 + (i % 4) * 14
        y = 4 + (i // 4) * 18
        seg_real[x:x + 6, y:y + 6, 8:14] = aseg
        seg_syn[x + 1:x + 6, y:y + 6, 8:14] = aseg

    nib.save(nib.Nifti1Image(seg_real, affine),
             os.path.join(seg_dir, f'{PID}_real_synthseg.nii.gz'))
    nib.save(nib.Nifti1Image(seg_syn, affine),
             os.path.join(seg_dir, f'{PID}_syn_synthseg.nii.gz'))

    rng = np.random.default_rng(0)
    real = rng.random(SHAPE).astype(np.float32)
    nib.save(nib.Nifti1Image(real, affine), os.path.join(root, f'{PID}_real.nii.gz'))
    nib.save(nib.Nifti1Image((real * 0.9).astype(np.float32), affine),
             os.path.join(root, f'{PID}_syn.nii.gz'))

    with open(os.path.join(root, 'subjects.csv'), 'w') as fh:
        fh.write('pid,slices,spacing_x,spacing_y,spacing_z,synthseg_mode,n_excluded\n')
        fh.write(f'{PID},{SHAPE[2]},{SPACING[0]},{SPACING[1]},{SPACING[2]},default,0\n')


def test_export_and_voxel_change():
    with tempfile.TemporaryDirectory() as tmp:
        export_dir = os.path.join(tmp, 'export')
        dump_dir = os.path.join(tmp, 'structs')
        os.makedirs(export_dir, exist_ok=True)
        build_fixture(export_dir)

        subprocess.run(
            [sys.executable, os.path.join(REPO, 'runners', 'eval_structures.py'),
             '--export_dir', export_dir, '--dump_structures', dump_dir],
            check=True, cwd=REPO, stdout=subprocess.DEVNULL,
        )

        # one real + one syn mask for every structure
        written = sorted(os.listdir(os.path.join(dump_dir, PID)))
        expected = 2 * len(STRUCTURE_NAMES)
        assert len(written) == expected, \
            f'expected {expected} dumped masks, got {len(written)}'
        print(f'  dumped {len(written)} per-structure masks for {PID}')

        with open(os.path.join(export_dir, 'structure_metrics.csv')) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(STRUCTURE_NAMES)

        for col in ('d_vox', 'vol_mm3_real', 'vol_mm3_syn', 'vol_ratio'):
            assert col in rows[0], f'{col} missing from structure_metrics.csv'

        vox_mm3 = float(np.prod(SPACING))
        for r in rows:
            assert int(r['n_vox_real']) == N_REAL, r
            assert int(r['n_vox_syn']) == N_SYN, r
            assert int(r['d_vox']) == N_SYN - N_REAL, r
            assert abs(float(r['vol_ratio']) - N_SYN / N_REAL) < 1e-9, r
            assert abs(float(r['vol_mm3_real']) - N_REAL * vox_mm3) < 0.01, r
            assert abs(float(r['dice']) - 2 * N_SYN / (N_REAL + N_SYN)) < 1e-9, r
            assert abs(float(r['centroid_mm']) - 0.5 * SPACING[0]) < 1e-6, r
        print(f"  dice={float(rows[0]['dice']):.4f} "
              f"vol_ratio={float(rows[0]['vol_ratio']):.4f} "
              f"d_vox={rows[0]['d_vox']} "
              f"centroid={float(rows[0]['centroid_mm']):.2f} mm  (all analytic)")

        # A mask with the wrong affine silently misaligns in a viewer.
        name = STRUCTURE_NAMES[sorted(STRUCTURE_NAMES)[0]]
        img = nib.load(os.path.join(dump_dir, PID, f'{PID}_{name}_real.nii.gz'))
        data = np.asanyarray(img.dataobj)
        assert set(np.unique(data)).issubset({0, 1}), 'dumped mask is not binary'
        assert int(data.sum()) == N_REAL, 'dumped mask voxel count disagrees with CSV'
        zooms = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
        assert zooms == SPACING, f'dumped mask spacing {zooms} != {SPACING}'
        print(f'  {name} mask: binary, {int(data.sum())} voxels, zooms {zooms}')


if __name__ == '__main__':
    test_export_and_voxel_change()
    print('PASS')
