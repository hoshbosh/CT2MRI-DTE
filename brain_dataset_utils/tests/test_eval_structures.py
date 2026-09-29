"""Guards for the Phase 2 metrics.

Checks the arithmetic against cases with a known answer rather than trusting
the implementation: a structure shifted by a known number of voxels must report
exactly that displacement in mm, and CW-SSIM must actually be more tolerant of
translation than plain SSIM -- otherwise the SSIM/CW-SSIM gap means nothing.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from runners.eval_structures import (  # noqa: E402
    centroid_mm, cw_ssim, dice, patch_image_metrics,
)
from skimage.metrics import structural_similarity  # noqa: E402


def make_blob(shape, centre, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    d2 = ((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2)
    return d2 <= radius ** 2


def test_dice_identical_and_disjoint():
    m = make_blob((40, 40, 20), (20, 20, 10), 6)
    assert abs(dice(m, m) - 1.0) < 1e-12, 'identical masks must score dice 1'
    far = make_blob((40, 40, 20), (5, 5, 3), 2)
    assert dice(m, far) == 0.0, 'disjoint masks must score dice 0'
    print('  dice identical=1, disjoint=0')


def test_centroid_displacement_is_exact():
    """A blob shifted by a known voxel offset must report shift * spacing mm."""
    shape = (40, 40, 20)
    spacing = (0.7, 0.9, 1.3)
    a = make_blob(shape, (20, 20, 10), 5)
    shift = (3, -2, 1)
    b = make_blob(shape, (20 + shift[0], 20 + shift[1], 10 + shift[2]), 5)

    ca, cb = centroid_mm(a, spacing), centroid_mm(b, spacing)
    got = float(np.linalg.norm(ca - cb))
    want = float(np.linalg.norm(np.array(shift, dtype=float) * np.array(spacing)))
    assert abs(got - want) < 1e-9, f'centroid displacement {got} != expected {want}'
    print(f'  centroid displacement {got:.4f} mm matches analytic {want:.4f} mm')


def test_centroid_none_when_absent():
    assert centroid_mm(np.zeros((5, 5, 5), bool), (1, 1, 1)) is None
    print('  absent structure yields no centroid (not a silent zero)')


def test_cwssim_more_translation_tolerant_than_ssim():
    """The whole point of reporting CW-SSIM alongside SSIM."""
    rng = np.random.default_rng(0)
    base = rng.random((64, 64)).astype(np.float32)
    # smooth it so it has structure rather than being pure noise
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(base, 2.0)
    base = (base - base.min()) / (base.max() - base.min())
    shifted = np.roll(base, 2, axis=0)

    s_plain = structural_similarity(base, shifted, data_range=1.0)
    s_cw = cw_ssim(base, shifted)
    assert s_cw > s_plain, (
        f'CW-SSIM ({s_cw:.3f}) should exceed SSIM ({s_plain:.3f}) under a pure '
        'translation; if it does not, the reported gap is meaningless'
    )
    same = cw_ssim(base, base)
    assert same > 0.99, f'CW-SSIM of a volume with itself should be ~1, got {same:.4f}'
    print(f'  translation by 2px: ssim={s_plain:.3f} < cw_ssim={s_cw:.3f}; self={same:.4f}')


def test_patch_metrics_perfect_reconstruction():
    shape = (48, 48, 12)
    rng = np.random.default_rng(1)
    real = rng.random(shape).astype(np.float32)
    mask = make_blob(shape, (24, 24, 6), 6)
    ssim, cws, psnr = patch_image_metrics(real, real.copy(), mask)
    assert ssim > 0.999, f'identical volumes should give ssim ~1, got {ssim}'
    assert np.isinf(psnr), f'identical volumes should give infinite psnr, got {psnr}'
    print(f'  perfect reconstruction: ssim={ssim:.4f} cw_ssim={cws:.4f} psnr=inf')


if __name__ == '__main__':
    test_dice_identical_and_disjoint()
    test_centroid_displacement_is_exact()
    test_centroid_none_when_absent()
    test_cwssim_more_translation_tolerant_than_ssim()
    test_patch_metrics_perfect_reconstruction()
    print('PASS')
