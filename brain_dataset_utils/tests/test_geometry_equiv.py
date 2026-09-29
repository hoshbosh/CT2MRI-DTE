"""Assert the extracted chain in geometry.py is bit-identical to the original
inline chain in finetune_preprocess.py, on both an intensity volume and a mask."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np
import nibabel as nib

from brain_dataset_utils import geometry as fp
from brain_dataset_utils.geometry import derive_geometry, apply_geometry

rng = np.random.default_rng(0)

def original_chain(mr_data, mask_data, affine, target_spacing, padding, height, width):
    """Verbatim replication of finetune_preprocess.preprocess_subject's geometry steps."""
    orig_ornt = nib.io_orientation(affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    mr_data = nib.orientations.apply_orientation(mr_data, transform)
    mask_data = nib.orientations.apply_orientation(mask_data, transform)
    aff = affine @ nib.orientations.inv_ornt_aff(transform, mr_data.shape[:3])

    mr_resampled, new_affine = fp.resample_volume(mr_data, aff, target_spacing, order=1)
    mask_resampled, _ = fp.resample_volume(mask_data.astype(np.float64), aff, target_spacing, order=0)
    mask_resampled = mask_resampled > 0.5

    coords = np.argwhere(mask_resampled)
    mins = np.maximum(coords.min(axis=0) - padding, 0)
    maxs = np.minimum(coords.max(axis=0) + 1 + padding, np.array(mr_resampled.shape[:3]))
    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    mr_cropped = mr_resampled[slices]
    return fp.resize_volume(mr_cropped, height, width)

cases = [
    ("RAS 1mm",      np.diag([1.0, 1.0, 1.0, 1.0])),
    ("LAS 0.8/0.8/2", np.diag([-0.8, 0.8, 2.0, 1.0])),
    ("PSR anisotropic", np.array([[0.0, 0.9, 0.0, -10.0],
                                  [0.0, 0.0, 1.5,  20.0],
                                  [0.7, 0.0, 0.0,  -5.0],
                                  [0.0, 0.0, 0.0,   1.0]])),
]

for name, affine in cases:
    shape = (44, 52, 37)
    mr = rng.random(shape) * 1000.0
    mask = np.zeros(shape, dtype=bool)
    mask[8:36, 10:44, 6:31] = True

    expected = original_chain(mr.copy(), mask.copy().astype(np.float64),
                              affine.copy(), 1.0, 4, 256, 256)

    img = nib.Nifti1Image(mr.astype(np.float32), affine)
    params = derive_geometry(img, mask, subject="TEST", target_spacing=1.0,
                             padding=4, height=256, width=256)
    got = apply_geometry(mr.copy(), params, order=1)

    assert got.shape == expected.shape, f"{name}: shape {got.shape} != {expected.shape}"
    maxdiff = np.abs(got - expected).max()
    assert maxdiff == 0.0, f"{name}: max abs diff {maxdiff}"

    # A label map must survive the same chain with no invented values.
    lab = np.zeros(shape, dtype=np.uint8)
    lab[12:20, 14:24, 10:18] = 7
    lab[24:32, 26:36, 12:22] = 11
    got_lab = apply_geometry(lab, params, order=0)
    vals = set(np.unique(got_lab).tolist())
    assert vals <= {0.0, 7.0, 11.0}, f"{name}: label chain invented values {vals}"
    assert 7.0 in vals and 11.0 in vals, f"{name}: label chain lost a structure: {vals}"

    print(f"  OK  {name:18s} shape={got.shape} spacing={[round(s,4) for s in params.spacing_mm]}")

print("geometry extraction is bit-identical to the original chain")
