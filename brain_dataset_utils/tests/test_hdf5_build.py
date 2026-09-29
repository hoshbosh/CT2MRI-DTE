import tempfile
"""End-to-end: build an HDF5 from synthetic subjects (one with 260 slices, the
case that used to desynchronise), then assert image/index/label/spacing alignment."""
import os, sys, tempfile, shutil, subprocess, csv
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO)
import numpy as np, nibabel as nib, h5py
from brain_dataset_utils.geometry import GeometryParams

WORK = os.path.join(tempfile.mkdtemp(prefix='ct2mri_test_'), 'hdf5build')
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
H = W = 8

# subject -> (total slices, leading blank slices)
SUBJECTS = {'SUB_A': (40, 3), 'SUB_BIG': (260, 2), 'SUB_C': (35, 4)}

for si, (pid, (nslices, nblank)) in enumerate(SUBJECTS.items()):
    d = os.path.join(WORK, pid); os.makedirs(d)
    ct = np.ones((H, W, nslices), dtype=np.float32)
    ct[:, :, :nblank] = 0.0                       # blank slices the filter must drop
    mr = np.zeros((H, W, nslices), dtype=np.float32)
    seg = np.zeros((H, W, nslices), dtype=np.uint8)
    for z in range(nslices):
        mr[:, :, z] = si * 1000 + z               # encode identity into voxel values
        seg[0, 0, z] = (z % 12) + 1               # and into the label map
    nib.save(nib.Nifti1Image(ct, np.eye(4)), os.path.join(d, 'ct.nii'))
    nib.save(nib.Nifti1Image(mr, np.eye(4)), os.path.join(d, 'mr.nii'))
    nib.save(nib.Nifti1Image(seg, np.eye(4)), os.path.join(d, 'seg.nii'))
    GeometryParams(
        subject=pid, orig_shape=(H, W, nslices), ornt_transform=[[0,1],[1,1],[2,1]],
        zoom_factors=[1,1,1], resample_shape=[H,W,nslices], crop_mins=[0,0,0],
        crop_maxs=[H,W,nslices], height=H, width=W, target_spacing=1.0,
        crop_affine=np.eye(4).tolist(), spacing_mm=[0.5 + si, 0.7 + si, 1.0],
    ).to_json(os.path.join(d, 'geometry.json'))

with open(os.path.join(WORK, 'data.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['pid', 'set'])
    for pid in SUBJECTS: w.writerow([pid, 'train'])

out = os.path.join(WORK, 'out.hdf5')
r = subprocess.run([sys.executable, '-u', 'brain_dataset_utils/generate_total_hdf5_csv.py',
    '--plane', 'axial', '--which_set', 'train', '--height', str(H), '--width', str(W),
    '--hdf5_name', out, '--data_dir', WORK, '--data_csv', os.path.join(WORK, 'data.csv'),
    '--CT_name', 'ct.nii', '--MR_name', 'mr.nii', '--SEG_name', 'seg.nii'],
    cwd=REPO, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:]); print(r.stderr[-3000:]); raise SystemExit("builder failed")

with h5py.File(out, 'r') as hf:
    MR = np.array(hf['MR_dataset']); LAB = np.array(hf['LABEL_dataset'])
    IDX = np.array(hf['index_dataset']); SP = np.array(hf['spacing_dataset'])
    SUBJ = np.array(hf['subject'])

expected_total = sum(n - b for n, b in SUBJECTS.values())
assert MR.shape[2] == expected_total, f"{MR.shape[2]} slices != expected {expected_total}"
assert IDX.shape[0] == MR.shape[2], f"index rows {IDX.shape[0]} != image slices {MR.shape[2]}"
assert LAB.shape[2] == MR.shape[2], "label/image slice count mismatch"
assert IDX.dtype == np.int32, f"index dtype is {IDX.dtype}, expected int32"
assert SP.shape == (MR.shape[2], 3), f"spacing shape {SP.shape}"

# Walk every global index and check it points at the slice its metadata claims.
pos = 0
for si, (pid, (nslices, nblank)) in enumerate(SUBJECTS.items()):
    kept = nslices - nblank
    for k in range(kept):
        g = pos + k
        assert SUBJ[g].decode() == pid, f"index {g}: subject {SUBJ[g]} != {pid}"
        assert IDX[g, 0] == k, f"index {g}: slice_number {IDX[g,0]} != {k}"
        assert IDX[g, 1] == kept - 1, f"index {g}: max_slice {IDX[g,1]} != {kept-1}"
        src_z = nblank + k
        assert MR[0, 0, g] == si * 1000 + src_z, (
            f"index {g}: image says {MR[0,0,g]}, metadata implies {si*1000+src_z}")
        # label axis-0/1 swap: seg[0,0,z] lands at LAB[0,0,g] after the axial transpose
        assert LAB[0, 0, g] == (src_z % 12) + 1, (
            f"index {g}: label {LAB[0,0,g]} != {(src_z % 12) + 1} -- label/image misaligned")
        assert np.allclose(SP[g], [0.7 + si, 0.5 + si, 1.0]), f"index {g}: spacing {SP[g]}"
    pos += kept

print(f"  {expected_total} slices across {len(SUBJECTS)} subjects (incl. one with 260)")
print("  image / index / subject / label / spacing all aligned at every global index")
print("HDF5 build verified")
