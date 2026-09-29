"""Single source of truth for the CT/MR/label geometric transform chain.

`finetune_preprocess.py` originally inlined this chain. It is extracted here so
that segmentation label maps can be pushed through the *identical* sequence with
nearest-neighbour interpolation instead of a second, subtly different
reimplementation -- which is the main way label/image misalignment happens.

Chain (native NIfTI -> training grid):
  1. reorient to RAS+
  2. resample to isotropic voxels
  3. crop to the brain-mask bounding box (+ padding)
  4. resize the two in-plane dimensions to (height, width)

Steps 1-4 leave a volume on the exact grid stored in `out-fine/<pid>/mr.nii`.
The axial transpose and the blank-slice filter happen later, in the HDF5
builder, where CT / MR / labels are all filtered together with one mask.

Interpolation order is the caller's choice: 1 (linear) for intensity images,
0 (nearest) for label maps. Everything else is fixed by the params record.
"""

from dataclasses import dataclass, asdict
import json

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, zoom


# FreeSurfer aseg label values kept for the deep-gray analysis, mapped to a
# compact contiguous range so the label volume fits in uint8 and indexes
# directly into STRUCTURE_NAMES.
ASEG_LABEL_MAP = {
    10: 1,   # Left-Thalamus
    49: 2,   # Right-Thalamus
    11: 3,   # Left-Caudate
    50: 4,   # Right-Caudate
    12: 5,   # Left-Putamen
    51: 6,   # Right-Putamen
    13: 7,   # Left-Pallidum
    52: 8,   # Right-Pallidum
    17: 9,   # Left-Hippocampus
    53: 10,  # Right-Hippocampus
    18: 11,  # Left-Amygdala
    54: 12,  # Right-Amygdala
}

STRUCTURE_NAMES = {
    1: 'thalamus_L', 2: 'thalamus_R',
    3: 'caudate_L', 4: 'caudate_R',
    5: 'putamen_L', 6: 'putamen_R',
    7: 'pallidum_L', 8: 'pallidum_R',
    9: 'hippocampus_L', 10: 'hippocampus_R',
    11: 'amygdala_L', 12: 'amygdala_R',
}

# Bilateral merges, used for reporting alongside the lateralized labels.
BILATERAL_GROUPS = {
    'thalamus': (1, 2),
    'caudate': (3, 4),
    'putamen': (5, 6),
    'pallidum': (7, 8),
    'hippocampus': (9, 10),
    'amygdala': (11, 12),
}


@dataclass
class GeometryParams:
    """Everything needed to replay the chain on another volume of the same subject."""
    subject: str
    orig_shape: tuple          # native volume shape, pre-reorient
    ornt_transform: list       # nibabel orientation transform, native -> RAS+
    zoom_factors: list         # per-axis resample factor (current spacing / target)
    resample_shape: list       # shape after step 2
    crop_mins: list            # bounding-box start, in resampled voxels
    crop_maxs: list            # bounding-box stop, in resampled voxels
    height: int                # step-4 target, dim 0
    width: int                 # step-4 target, dim 1
    target_spacing: float      # isotropic spacing after step 2, in mm
    crop_affine: list          # affine of the cropped volume (pre-resize; see note)
    spacing_mm: list           # true per-voxel mm on the final grid, dims (0, 1, 2)

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path):
        with open(path) as f:
            return GeometryParams(**json.load(f))

    def spacing_for_plane(self, plane):
        """Per-voxel mm in HDF5 axis order, after `transpose_LPS_to_ITKSNAP_position`.

        Axial transposes (0, 1, 2) -> (1, 0, 2), so the stored H axis is the
        volume's dim 1 and the stored W axis is its dim 0.
        """
        sx, sy, sz = self.spacing_mm
        if plane == 'axial':
            return [sy, sx, sz]
        raise NotImplementedError(
            f"spacing_for_plane is only derived for 'axial'; got {plane!r}. "
            "Coronal/sagittal reorder axes differently and need their own case."
        )


def _reorient_to_ras(data, transform):
    return nib.orientations.apply_orientation(data, transform)


def resample_volume(data, affine, target_spacing=1.0, order=1):
    """Resample a volume to isotropic voxel spacing."""
    current_spacing = np.array(nib.affines.voxel_sizes(affine))
    zoom_factors = current_spacing / target_spacing
    new_shape = np.round(np.array(data.shape[:3]) * zoom_factors).astype(int)

    resample_matrix = np.diag(1.0 / zoom_factors)
    resampled = affine_transform(
        data, matrix=resample_matrix, output_shape=tuple(new_shape),
        order=order, mode="constant", cval=0.0,
    )

    new_affine = affine.copy()
    for i in range(3):
        new_affine[:3, i] = affine[:3, i] / zoom_factors[i]

    return resampled, new_affine


def resize_volume(data, height, width, order=1):
    """Resize the first two spatial dimensions of a volume to (height, width)."""
    zoom_factors = (height / data.shape[0], width / data.shape[1], 1.0)
    return zoom(data, zoom_factors, order=order)


def derive_geometry(mr_img, mask_data, subject, target_spacing=1.0, padding=4,
                    height=256, width=256):
    """Compute the transform params for one subject from its MR and brain mask.

    mr_img:    nibabel image, native (used for its affine and shape only)
    mask_data: native-space brain mask array, same grid as mr_img
    """
    affine = mr_img.affine.copy()
    orig_shape = tuple(mr_img.shape[:3])

    orig_ornt = nib.io_orientation(affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    affine = affine @ nib.orientations.inv_ornt_aff(transform, orig_shape)

    mask_ras = _reorient_to_ras(np.asarray(mask_data, dtype=np.float64), transform)
    mask_resampled, new_affine = resample_volume(mask_ras, affine, target_spacing, order=0)
    mask_resampled = mask_resampled > 0.5

    coords = np.argwhere(mask_resampled)
    if len(coords) == 0:
        raise ValueError(f"{subject}: brain mask is empty after resampling; refusing to guess a crop box")
    mins = np.maximum(coords.min(axis=0) - padding, 0)
    maxs = np.minimum(coords.max(axis=0) + 1 + padding, np.array(mask_resampled.shape[:3]))

    crop_affine = new_affine.copy()
    crop_affine[:3, 3] = new_affine[:3, :3] @ mins + new_affine[:3, 3]

    # The in-plane resize changes voxel size but the saved affine is not updated
    # (preserved for byte-compatibility with the existing out-fine volumes), so
    # the true spacing is recorded separately here.
    extent = maxs - mins
    spacing_mm = [
        float(extent[0]) * target_spacing / height,
        float(extent[1]) * target_spacing / width,
        float(target_spacing),
    ]

    current_spacing = np.array(nib.affines.voxel_sizes(affine))
    return GeometryParams(
        subject=subject,
        orig_shape=orig_shape,
        ornt_transform=transform.tolist(),
        zoom_factors=(current_spacing / target_spacing).tolist(),
        resample_shape=list(np.asarray(mask_resampled.shape[:3], dtype=int).tolist()),
        crop_mins=mins.tolist(),
        crop_maxs=maxs.tolist(),
        height=height,
        width=width,
        target_spacing=float(target_spacing),
        crop_affine=crop_affine.tolist(),
        spacing_mm=spacing_mm,
    )


def apply_geometry(data, params, order):
    """Push a native-space volume through the chain onto the training grid.

    order=1 for intensity images, order=0 for label maps. No default: picking
    the wrong one silently blends label values into nonexistent structures.
    """
    data = np.asarray(data, dtype=np.float64)
    if tuple(data.shape[:3]) != tuple(params.orig_shape):
        raise ValueError(
            f"{params.subject}: volume shape {data.shape[:3]} does not match the "
            f"geometry record's {tuple(params.orig_shape)}; wrong subject or wrong grid"
        )

    transform = np.asarray(params.ornt_transform, dtype=float)
    data = _reorient_to_ras(data, transform)

    zoom_factors = np.asarray(params.zoom_factors, dtype=float)
    resample_matrix = np.diag(1.0 / zoom_factors)
    data = affine_transform(
        data, matrix=resample_matrix, output_shape=tuple(params.resample_shape),
        order=order, mode="constant", cval=0.0,
    )

    slices = tuple(slice(mn, mx) for mn, mx in zip(params.crop_mins, params.crop_maxs))
    data = data[slices]

    return resize_volume(data, params.height, params.width, order=order)
