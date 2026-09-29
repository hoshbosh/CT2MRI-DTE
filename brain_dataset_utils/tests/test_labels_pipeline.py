import tempfile
"""Smoke-test make_labels.py + verify_labels.py on a synthetic cohort:
native MR -> fake SynthSeg output (on a DIFFERENT grid, as SynthSeg really does)
-> transported labels -> QC report."""
import os, sys, tempfile, shutil, subprocess, csv
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO)
import numpy as np, nibabel as nib
from brain_dataset_utils.geometry import derive_geometry, ASEG_LABEL_MAP

W = os.path.join(tempfile.mkdtemp(prefix='ct2mri_test_'), 'labpipe')
shutil.rmtree(W, ignore_errors=True)
NATIVE, SEGD, OUT = [os.path.join(W, d) for d in ('native', 'synthseg', 'out-fine')]
for d in (NATIVE, SEGD, OUT): os.makedirs(d)

rng = np.random.default_rng(3)
PIDS = ['S001', 'S002', 'S003']
shape = (60, 70, 50)
# Native grid is anisotropic, so SynthSeg's 1mm output really is a different grid.
native_affine = np.diag([0.9, 0.9, 1.6, 1.0])

for pid in PIDS:
    d = os.path.join(NATIVE, pid); os.makedirs(d)
    mr = rng.random(shape).astype(np.float32) * 500
    mask = np.zeros(shape, bool); mask[6:54, 8:62, 5:45] = True
    mr[~mask] = 0
    nib.save(nib.Nifti1Image(mr, native_affine), os.path.join(d, 'mr.nii.gz'))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), native_affine), os.path.join(d, 'mask.nii.gz'))

    # Fake SynthSeg output: 1mm isotropic grid, aseg label values, 12 blobs.
    seg_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    seg_shape = (54, 63, 80)
    seg = np.zeros(seg_shape, dtype=np.int16)
    for i, aseg in enumerate(sorted(ASEG_LABEL_MAP)):
        x = 12 + (i % 4) * 8; y = 14 + (i // 4) * 12; z = 22 + (i % 3) * 10
        seg[x:x+6, y:y+7, z:z+7] = aseg
    nib.save(nib.Nifti1Image(seg, seg_affine), os.path.join(SEGD, f'{pid}_synthseg.nii.gz'))

    # out-fine: geometry.json + a preprocessed mr.nii for the overlay
    od = os.path.join(OUT, pid); os.makedirs(od)
    mr_img = nib.load(os.path.join(d, 'mr.nii.gz'))
    params = derive_geometry(mr_img, mask, pid, 1.0, 4, 256, 256)
    params.to_json(os.path.join(od, 'geometry.json'))
    from brain_dataset_utils.geometry import apply_geometry
    g = apply_geometry(mr, params, order=1)
    nib.save(nib.Nifti1Image(g.astype(np.float32), np.asarray(params.crop_affine)),
             os.path.join(od, 'mr.nii'))

with open(os.path.join(OUT, 'data.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pid','set'])
    for pid, s in zip(PIDS, ['train','valid','test']): w.writerow([pid, s])

def run(cmd):
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(r.stdout.strip()[-2500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2500:]); raise SystemExit(f"failed: {' '.join(cmd)}")

print("--- make_labels ---")
run([sys.executable, '-u', 'brain_dataset_utils/make_labels.py',
     '--synthseg_dir', SEGD, '--native_dir', NATIVE, '--out_dir', OUT])

for pid in PIDS:
    seg = np.asanyarray(nib.load(os.path.join(OUT, pid, 'seg.nii')).dataobj)
    mr = np.asanyarray(nib.load(os.path.join(OUT, pid, 'mr.nii')).dataobj)
    assert seg.shape == mr.shape, f"{pid}: seg {seg.shape} != mr {mr.shape}"
    vals = set(np.unique(seg).tolist()) - {0}
    assert vals == set(range(1, 13)), f"{pid}: got codes {sorted(vals)}"
assert set(np.unique(seg).tolist()) <= set(range(13)), "labels outside the compact range"
print("  all 12 structures present, on the mr.nii grid, compact codes only")

print("\n--- verify_labels ---")
run([sys.executable, '-u', 'brain_dataset_utils/verify_labels.py',
     '--out_dir', OUT, '--data_csv', os.path.join(OUT, 'data.csv'),
     '--report_dir', os.path.join(W, 'qc'), '--n_overlay', '3'])

pngs = os.listdir(os.path.join(W, 'qc', 'overlays'))
assert len(pngs) >= 3, f"expected overlays, got {pngs}"
print(f"\n  overlays written: {sorted(pngs)}")
print("label pipeline smoke test passed")
