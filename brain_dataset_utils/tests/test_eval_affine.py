"""Guard for the evaluation-grid affine.

SynthSeg places anatomy using the affine, so an affine that disagrees with how
the voxels are actually laid out yields a segmentation with cortex and CSF but
no subcortical structures -- which is what a plain diagonal affine produced.

The test is a round trip: pick a voxel in the pre-transpose volume, note its
world coordinate under crop_affine (corrected for the in-plane resize), then
find that same voxel after the transpose and check the derived affine maps it
to the same world point.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from brain_dataset_utils.geometry import GeometryParams  # noqa: E402
from runners.export_eval_volumes import PLANE_PERM, evaluation_affine  # noqa: E402


def make_params(crop_affine, spacing_mm, target_spacing=1.0, height=256, width=256):
    return GeometryParams(
        subject='TEST', orig_shape=(200, 200, 150), ornt_transform=[],
        zoom_factors=[1.0, 1.0, 1.0], resample_shape=[200, 200, 150],
        crop_mins=[0, 0, 0], crop_maxs=[200, 200, 150],
        height=height, width=width, target_spacing=target_spacing,
        crop_affine=np.asarray(crop_affine).tolist(),
        spacing_mm=list(spacing_mm),
    )


def test_world_point_survives_the_transpose():
    """A voxel's world coordinate must not move when the axes are permuted."""
    # An oblique-ish RAS affine with unit columns, as crop_affine has.
    base = np.array([
        [0.99, -0.10, 0.05, -84.0],
        [0.10, 0.99, -0.02, -110.0],
        [-0.05, 0.03, 1.00, -60.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    spacing = [0.8, 0.7, 1.0]
    params = make_params(base, spacing)

    # The pre-transpose volume after the in-plane resize: columns scaled.
    pre = base.copy()
    for i, s in enumerate(spacing):
        pre[:3, i] *= s / params.target_spacing

    post = evaluation_affine(params, 'axial')
    perm = PLANE_PERM['axial']

    # A voxel at index v in the pre-transpose volume sits at index v[perm] in
    # np.transpose(data, perm). The affine columns are permuted the same way, so
    # both must map to the same world point.
    rng = np.random.default_rng(0)
    for _ in range(25):
        v_pre = rng.integers(0, 150, size=3).astype(float)
        v_post = v_pre[np.asarray(perm)]
        w_pre = pre[:3, :3] @ v_pre + pre[:3, 3]
        w_post = post[:3, :3] @ v_post + post[:3, 3]
        assert np.allclose(w_pre, w_post, atol=1e-9), (
            f'world point moved: {w_pre} vs {w_post}')
    print('  world coordinates identical across the axial transpose')


def test_affine_spacing_matches_spacing_mm():
    base = np.eye(4)
    base[:3, 3] = [-90.0, -120.0, -70.0]
    spacing = [0.914, 0.832, 1.0]
    params = make_params(base, spacing)
    A = evaluation_affine(params, 'axial')
    got = np.sqrt((A[:3, :3] ** 2).sum(axis=0))
    # axial permutation is [1, 0, 2], so the reported spacing is reordered
    want = [spacing[1], spacing[0], spacing[2]]
    assert np.allclose(got, want, atol=1e-9), f'{got} != {want}'
    print(f'  column norms {got.round(4).tolist()} match permuted spacing_mm')


def test_unknown_plane_raises():
    params = make_params(np.eye(4), [1.0, 1.0, 1.0])
    for plane in ('coronal', 'sagittal'):
        try:
            evaluation_affine(params, plane)
        except ValueError:
            continue
        raise AssertionError(f'{plane} should raise rather than return a guess')
    print('  coronal/sagittal refuse rather than guessing')


if __name__ == '__main__':
    test_world_point_survives_the_transpose()
    test_affine_spacing_matches_spacing_mm()
    test_unknown_plane_raises()
    print('PASS')
