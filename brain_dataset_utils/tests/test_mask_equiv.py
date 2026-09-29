"""The refactor moved mask binarization from mid-chain to end-of-chain.
With order=0 that must be a no-op. Prove it."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import numpy as np, nibabel as nib
from brain_dataset_utils.geometry import derive_geometry, apply_geometry, resample_volume, resize_volume

rng = np.random.default_rng(1)
for name, affine in [("RAS 1mm", np.diag([1.,1.,1.,1.])),
                     ("LAS aniso", np.diag([-0.8,0.8,2.0,1.])),
                     ("PSR", np.array([[0,0.9,0,-10.],[0,0,1.5,20.],[0.7,0,0,-5.],[0,0,0,1.]]))]:
    shape = (44, 52, 37)
    mask = np.zeros(shape, dtype=bool)
    mask[8:36, 10:44, 6:31] = True
    mask[rng.random(shape) > 0.97] = True   # speckle, to stress nearest-neighbour

    img = nib.Nifti1Image(rng.random(shape).astype(np.float32), affine)
    params = derive_geometry(img, mask, "TEST", 1.0, 4, 256, 256)

    # ORIGINAL: reorient -> resample(order=0) -> >0.5 -> crop -> resize(order=0) -> >0.5
    tr = np.asarray(params.ornt_transform, dtype=float)
    m = nib.orientations.apply_orientation(mask.astype(np.float64), tr)
    aff = affine @ nib.orientations.inv_ornt_aff(tr, shape)
    m, _ = resample_volume(m, aff, 1.0, order=0)
    m = m > 0.5
    sl = tuple(slice(a, b) for a, b in zip(params.crop_mins, params.crop_maxs))
    m = m[sl]
    expected = resize_volume(m.astype(np.float64), 256, 256, order=0) > 0.5

    # REFACTORED: single apply_geometry(order=0), binarize once at the end
    got = apply_geometry(mask.astype(np.float64), params, order=0) > 0.5

    assert got.shape == expected.shape, f"{name}: {got.shape} != {expected.shape}"
    ndiff = int((got != expected).sum())
    assert ndiff == 0, f"{name}: {ndiff} voxels differ"
    print(f"  OK  {name:10s} mask identical ({got.sum()} voxels in mask)")
print("mask binarization move is a no-op")
