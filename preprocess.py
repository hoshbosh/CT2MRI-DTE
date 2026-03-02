#!/usr/bin/env python3
"""
Preprocess MRI NIfTI volumes for CT2MRI pretraining using FreeSurfer SynthStrip.

Pipeline per volume:
  1. Skull strip via FreeSurfer mri_synthstrip
  2. Resample to 1mm isotropic voxels
  3. Crop to brain bounding box
  4. Clip outlier intensities (99.5th percentile)
  5. Min-max normalize to [0, 1]
  6. Save as subject_dir/ct.nii.gz and subject_dir/mr.nii.gz
  7. Generate split.csv with 70/30 train/test split

Output structure:
  output_dir/
  ├── split.csv
  ├── subject1/
  │   ├── ct.nii.gz
  │   └── mr.nii.gz
  ├── subject2/
  │   ├── ct.nii.gz
  │   └── mr.nii.gz
  └── ...

Usage:
    python preprocess_mri.py --input_dir /path/to/niftis --output_dir /path/to/output

Requirements:
    pip install nibabel numpy scipy tqdm
    FreeSurfer installed with mri_synthstrip available on PATH
    (set FREESURFER_HOME and source $FREESURFER_HOME/SetUpFreeSurfer.sh)
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Skull stripping via FreeSurfer SynthStrip
# ---------------------------------------------------------------------------

def check_synthstrip():
    """Verify mri_synthstrip is available."""
    if shutil.which("mri_synthstrip") is None:
        print("ERROR: mri_synthstrip not found on PATH.")
        print("Make sure FreeSurfer is installed and configured:")
        print("  export FREESURFER_HOME=/path/to/freesurfer")
        print("  source $FREESURFER_HOME/SetUpFreeSurfer.sh")
        sys.exit(1)


def skull_strip(input_path: str, work_dir: str):
    """
    Run mri_synthstrip on a NIfTI file.

    Returns:
        stripped_data (np.ndarray): skull-stripped volume
        mask_data (np.ndarray): binary brain mask
        orig_img (nib.Nifti1Image): original loaded image
    """
    stripped_path = os.path.join(work_dir, "stripped.nii.gz")
    mask_path = os.path.join(work_dir, "mask.nii.gz")

    cmd = [
        "mri_synthstrip",
        "-i", input_path,
        "-o", stripped_path,
        "-m", mask_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"mri_synthstrip failed (exit {result.returncode}):\n{result.stderr}"
        )

    orig_img = nib.load(input_path)
    stripped_img = nib.load(stripped_path)
    mask_img = nib.load(mask_path)

    return (
        stripped_img.get_fdata().astype(np.float64),
        mask_img.get_fdata().astype(bool),
        orig_img,
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_volume(data: np.ndarray, affine: np.ndarray,
                    target_spacing: float = 1.0, order: int = 1):
    """
    Resample a volume to isotropic voxel spacing.

    Args:
        order: interpolation order (1=trilinear for images, 0=nearest for masks)

    Returns:
        resampled_data, new_affine
    """
    current_spacing = np.array(nib.affines.voxel_sizes(affine))
    zoom_factors = current_spacing / target_spacing
    new_shape = np.round(np.array(data.shape[:3]) * zoom_factors).astype(int)

    resample_matrix = np.diag(1.0 / zoom_factors)
    resampled = affine_transform(
        data,
        matrix=resample_matrix,
        output_shape=tuple(new_shape),
        order=order,
        mode="constant",
        cval=0.0,
    )

    new_affine = affine.copy()
    for i in range(3):
        new_affine[:3, i] = affine[:3, i] / zoom_factors[i]

    return resampled, new_affine


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def crop_to_brain(data: np.ndarray, mask: np.ndarray,
                  affine: np.ndarray, padding: int = 4):
    """Crop volume and mask to the brain bounding box with padding."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return data, mask, affine

    mins = np.maximum(coords.min(axis=0) - padding, 0)
    maxs = np.minimum(coords.max(axis=0) + 1 + padding, np.array(data.shape[:3]))

    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    cropped_data = data[slices]
    cropped_mask = mask[slices]

    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, :3] @ mins + affine[:3, 3]

    return cropped_data, cropped_mask, new_affine


# ---------------------------------------------------------------------------
# Intensity normalization
# ---------------------------------------------------------------------------

def clip_and_normalize(data: np.ndarray, mask: np.ndarray,
                       percentile: float = 99.5):
    """
    Clip outlier intensities at the given percentile within the brain mask,
    then min-max normalize to [0, 1]. Non-brain voxels are set to 0.
    """
    brain_voxels = data[mask]
    if len(brain_voxels) == 0:
        return data

    high = np.percentile(brain_voxels, percentile)
    data = np.clip(data, 0.0, high)

    if high > 0:
        data = data / high

    data[~mask] = 0.0
    return data


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_single(nii_path: str, output_dir: str, target_spacing: float = 1.0,
                      padding: int = 4, clip_percentile: float = 99.5):
    """Run the full preprocessing pipeline on a single NIfTI file."""
    basename = Path(nii_path).name.replace(".nii.gz", "").replace(".nii", "")

    # Create subject directory
    subject_dir = os.path.join(output_dir, basename)
    os.makedirs(subject_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as work_dir:
        # 1. Skull strip with SynthStrip
        stripped_data, mask_data, orig_img = skull_strip(nii_path, work_dir)

    affine = orig_img.affine.copy()

    # 2. Resample to isotropic
    resampled_data, new_affine = resample_volume(
        stripped_data, affine, target_spacing, order=1
    )
    resampled_mask, _ = resample_volume(
        mask_data.astype(np.float64), affine, target_spacing, order=0
    )
    resampled_mask = resampled_mask > 0.5

    # 3. Crop to brain bounding box
    cropped_data, cropped_mask, crop_affine = crop_to_brain(
        resampled_data, resampled_mask, new_affine, padding
    )

    # 4 & 5. Clip outliers and normalize to [0, 1]
    normalized_data = clip_and_normalize(cropped_data, cropped_mask, clip_percentile)

    # Save as ct.nii.gz and mr.nii.gz in subject directory
    out_img = nib.Nifti1Image(normalized_data.astype(np.float32), crop_affine)
    nib.save(out_img, os.path.join(subject_dir, "ct.nii"))
    nib.save(out_img, os.path.join(subject_dir, "mr.nii"))

    return basename, normalized_data.shape


def find_nifti_files(input_dir: str):
    """Recursively find all .nii and .nii.gz files."""
    nifti_files = []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if f.endswith(".nii.gz") or (f.endswith(".nii") and not f.endswith(".nii.gz")):
                nifti_files.append(os.path.join(root, f))
    return nifti_files


def write_split_csv(output_dir: str, subject_names: list, test_ratio: float = 0.3,
                    seed: int = 42):
    """Write a CSV with pid and train/test split."""
    random.seed(seed)
    shuffled = subject_names.copy()
    random.shuffle(shuffled)

    n_test = round(len(shuffled) * test_ratio)
    test_set = set(shuffled[:n_test])

    csv_path = os.path.join(output_dir, "split.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "split"])
        for name in sorted(subject_names):
            split = "test" if name in test_set else "train"
            writer.writerow([name, split])

    n_train = len(subject_names) - n_test
    print(f"Split CSV: {csv_path} ({n_train} train, {n_test} test)")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MRI NIfTI volumes for CT2MRI pretraining (FreeSurfer SynthStrip)"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing .nii/.nii.gz files (searched recursively)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save preprocessed subject folders")
    parser.add_argument("--target_spacing", type=float, default=1.0,
                        help="Isotropic voxel size in mm (default: 1.0)")
    parser.add_argument("--padding", type=int, default=4,
                        help="Padding voxels around brain bounding box (default: 4)")
    parser.add_argument("--clip_percentile", type=float, default=99.5,
                        help="Percentile for outlier clipping (default: 99.5)")
    parser.add_argument("--test_ratio", type=float, default=0.3,
                        help="Fraction of subjects for test set (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split (default: 42)")
    args = parser.parse_args()

    check_synthstrip()
    os.makedirs(args.output_dir, exist_ok=True)

    nifti_files = find_nifti_files(args.input_dir)
    if not nifti_files:
        print(f"No NIfTI files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(nifti_files)} NIfTI files")
    print(f"Target spacing: {args.target_spacing}mm isotropic")
    print(f"Clip percentile: {args.clip_percentile}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Output: {args.output_dir}\n")

    successful_subjects = []
    failed = []

    for nii_path in tqdm(nifti_files, desc="Preprocessing"):
        try:
            basename, shape = preprocess_single(
                nii_path, args.output_dir,
                target_spacing=args.target_spacing,
                padding=args.padding,
                clip_percentile=args.clip_percentile,
            )
            successful_subjects.append(basename)
            tqdm.write(f"  OK  {basename} -> {shape}")
        except Exception as e:
            tqdm.write(f"  FAIL {Path(nii_path).name}: {e}")
            failed.append((nii_path, str(e)))

    # Generate train/test split CSV
    if successful_subjects:
        write_split_csv(args.output_dir, successful_subjects,
                        test_ratio=args.test_ratio, seed=args.seed)

    print(f"\nDone. {len(successful_subjects)}/{len(nifti_files)} succeeded.")
    if failed:
        print("\nFailed files:")
        for path, err in failed:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
