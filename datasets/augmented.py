"""
Augmented paired dataset for CT-to-MRI translation.

Spatial augmentations (flip, rotation, scale, translation) are applied
identically to both CT and MRI to preserve alignment.
Intensity augmentations (noise, brightness) are applied independently.

Usage: set dataset_name to 'ct2mr_aligned_augmented' in your config YAML.
"""

import math
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Register import Registers
from datasets.base import multi_ch_nifti_default_Dataset


# ---------------------------------------------------------------------------
# Affine helpers
# ---------------------------------------------------------------------------

def _build_affine_matrix(angle_deg, scale, tx, ty):
    """Build a 2x3 affine matrix combining rotation, scale, and translation."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[scale * cos_a, -scale * sin_a, tx],
         [scale * sin_a,  scale * cos_a, ty]],
        dtype=torch.float32,
    )


def _apply_affine(img, theta):
    """Apply a 2x3 affine transform to a (C, H, W) tensor."""
    x = img.unsqueeze(0)  # (1, C, H, W)
    grid = F.affine_grid(theta.unsqueeze(0), x.size(), align_corners=False)
    out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros',
                        align_corners=False)
    return out.squeeze(0)


# ---------------------------------------------------------------------------
# Augmented paired dataset
# ---------------------------------------------------------------------------

@Registers.datasets.register_with_name('ct2mr_aligned_augmented')
class AugmentedCT2MR_Paired_Dataset(Dataset):
    """CT2MR paired dataset with online data augmentation.

    Replaces the deterministic flip doubling with random spatial and intensity
    augmentations applied on-the-fly, giving more diversity per epoch without
    increasing disk usage.
    """

    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        self.augment = (stage == 'train')
        self.to_normal = dataset_config.to_normal

        hdf5_path = os.path.join(
            dataset_config.dataset_path,
            f"{dataset_config.image_size}_{stage}_{dataset_config.plane}.hdf5",
        )
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('MR_dataset'), dtype=np.float16)
            B_dataset = np.array(hf.get('CT_dataset'), dtype=np.float16)
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))

        # Disable flip in base — all augmentation handled here
        self.imgs_ori = multi_ch_nifti_default_Dataset(
            A_dataset, index_dataset, subjects, self.radius, self.image_size,
            flip=False, to_normal=self.to_normal,
        )
        self.imgs_cond = multi_ch_nifti_default_Dataset(
            B_dataset, index_dataset, subjects, self.radius, self.image_size,
            flip=False, to_normal=self.to_normal,
        )

        # Augmentation hyperparameters
        self.flip_p = 0.5
        self.rot_range = 10.0        # ±10 degrees
        self.scale_range = 0.05      # ±5%
        self.translate_range = 0.03  # ±3% of image size
        self.noise_std = 0.02
        self.brightness_range = 0.05  # ±5%

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        img_ori, subj_ori = self.imgs_ori[i]
        img_cond, subj_cond = self.imgs_cond[i]

        if not self.augment:
            return (img_ori, subj_ori), (img_cond, subj_cond)

        # --- Spatial augmentations (synchronized for both modalities) ---

        # Random horizontal flip (skip for sagittal — would flip L/R anatomy)
        if self.plane != 'sagittal' and torch.rand(1).item() < self.flip_p:
            img_ori = torch.flip(img_ori, [-1])
            img_cond = torch.flip(img_cond, [-1])

        # Random affine: rotation + scale + translation
        angle = (torch.rand(1).item() - 0.5) * 2 * self.rot_range
        scale = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.scale_range
        tx = (torch.rand(1).item() - 0.5) * 2 * self.translate_range
        ty = (torch.rand(1).item() - 0.5) * 2 * self.translate_range

        if abs(angle) > 0.5 or abs(scale - 1.0) > 0.005:
            theta = _build_affine_matrix(angle, scale, tx, ty)
            img_ori = _apply_affine(img_ori, theta)
            img_cond = _apply_affine(img_cond, theta)

        # --- Intensity augmentations (independent per modality) ---

        # Gaussian noise
        if torch.rand(1).item() < 0.5:
            img_ori = img_ori + torch.randn_like(img_ori) * self.noise_std
            img_cond = img_cond + torch.randn_like(img_cond) * self.noise_std

        # Random brightness shift
        if torch.rand(1).item() < 0.5:
            shift_ori = (torch.rand(1).item() - 0.5) * 2 * self.brightness_range
            shift_cond = (torch.rand(1).item() - 0.5) * 2 * self.brightness_range
            img_ori = img_ori + shift_ori
            img_cond = img_cond + shift_cond

        # Clamp to valid range
        if self.to_normal:
            img_ori.clamp_(-1.0, 1.0)
            img_cond.clamp_(-1.0, 1.0)
        else:
            img_ori.clamp_(0.0, 1.0)
            img_cond.clamp_(0.0, 1.0)

        return (img_ori, subj_ori), (img_cond, subj_cond)
