#!/usr/bin/env python3
"""
Preprocess SynthRAD Task1 Brain dataset for CT2MRI fine-tuning.

Input structure (SynthRAD):
  input_dir/
  ├── subject1/
  │   ├── ct.nii.gz
  │   ├── mr.nii.gz
  │   └── mask.nii.gz
  ├── subject2/
  │   ├── ct.nii.gz
  │   ├── mr.nii.gz
  │   └── mask.nii.gz
  └── ...

Pipeline per subject:
  1. Load CT, MR, and provided brain mask
  2. Reorient to RAS+
  3. Resample to 1mm isotropic voxels
  4. Crop to brain bounding box
  5. Resize to target spatial dimensions
  6. Clip outlier intensities (99.5th percentile)
  7. Min-max normalize to [0, 1]

Output structure:
  output_dir/
  ├── data.csv
  ├── subject1/
  │   ├── ct.nii
  │   └── mr.nii
  └── ...

Usage:
    python finetune_preprocess.py --input_dir /path/to/synthrad --output_dir /path/to/output
    python finetune_preprocess.py --input_dir /path/to/synthrad --output_dir /path/to/output --workers 8

Requirements:
    pip install nibabel numpy scipy tqdm
"""

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, zoom
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def crop_to_brain(data, mask, affine, padding=4):
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
# Resizing
# ---------------------------------------------------------------------------

def resize_volume(data, height, width, order=1):
    """Resize the first two spatial dimensions of a volume to (height, width)."""
    zoom_factors = (height / data.shape[0], width / data.shape[1], 1.0)
    return zoom(data, zoom_factors, order=order)


# ---------------------------------------------------------------------------
# Intensity normalization
# ---------------------------------------------------------------------------

def clip_and_normalize(data, mask, percentile=99.5):
    """Clip outlier intensities then min-max normalize to [0, 1]."""
    brain_voxels = data[mask]
    if len(brain_voxels) == 0:
        return data

    high = np.percentile(brain_voxels, percentile)
    data = np.clip(data, 0.0, high)

    if high > 0:
        data = data / high

    data[~mask] = 0.0
    return data


def clip_and_normalize_ct(data, mask, window_center=40, window_width=80):
    """
    Clip CT to a soft-tissue brain window then normalize to [0, 1].
    Default window: center=40 HU, width=80 HU -> range [0, 80] HU.
    This prevents bone (1000+ HU) from compressing brain tissue into a
    narrow dark range.
    """
    low = window_center - window_width / 2
    high = window_center + window_width / 2
    data = np.clip(data, low, high)

    data = (data - low) / (high - low)

    data[~mask] = 0.0
    return data


# ---------------------------------------------------------------------------
# Single subject pipeline
# ---------------------------------------------------------------------------

def preprocess_subject(args_tuple):
    """Run the full preprocessing pipeline on a single SynthRAD subject."""
    subject_dir, output_dir, target_spacing, padding, clip_percentile, height, width, \
        ct_name, mr_name, mask_name = args_tuple

    subject_name = Path(subject_dir).name
    out_subject_dir = os.path.join(output_dir, subject_name)
    os.makedirs(out_subject_dir, exist_ok=True)

    # Load CT, MR, and mask
    ct_img = nib.load(os.path.join(subject_dir, ct_name))
    mr_img = nib.load(os.path.join(subject_dir, mr_name))
    mask_img = nib.load(os.path.join(subject_dir, mask_name))

    ct_data = ct_img.get_fdata().astype(np.float64)
    mr_data = mr_img.get_fdata().astype(np.float64)
    mask_data = mask_img.get_fdata().astype(bool)
    affine = mr_img.affine.copy()

    # Reorient to RAS+
    orig_ornt = nib.io_orientation(affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)

    ct_data = nib.orientations.apply_orientation(ct_data, transform)
    mr_data = nib.orientations.apply_orientation(mr_data, transform)
    mask_data = nib.orientations.apply_orientation(mask_data, transform)
    affine = affine @ nib.orientations.inv_ornt_aff(transform, mr_img.shape[:3])

    # Resample to isotropic
    ct_resampled, new_affine = resample_volume(ct_data, affine, target_spacing, order=1)
    mr_resampled, _ = resample_volume(mr_data, affine, target_spacing, order=1)
    mask_resampled, _ = resample_volume(mask_data.astype(np.float64), affine, target_spacing, order=0)
    mask_resampled = mask_resampled > 0.5

    # Crop to brain bounding box (use mask to define bounds, apply to all)
    ct_cropped, mask_cropped, crop_affine = crop_to_brain(ct_resampled, mask_resampled, new_affine, padding)
    # Crop MR with same bounds
    coords = np.argwhere(mask_resampled)
    mins = np.maximum(coords.min(axis=0) - padding, 0)
    maxs = np.minimum(coords.max(axis=0) + 1 + padding, np.array(mr_resampled.shape[:3]))
    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    mr_cropped = mr_resampled[slices]

    # Resize to target spatial dimensions
    ct_cropped = resize_volume(ct_cropped, height, width)
    mr_cropped = resize_volume(mr_cropped, height, width)
    mask_cropped = resize_volume(mask_cropped.astype(np.float64), height, width, order=0) > 0.5

    # Clip and normalize (CT uses brain window, MR uses percentile)
    ct_normalized = clip_and_normalize_ct(ct_cropped, mask_cropped)
    mr_normalized = clip_and_normalize(mr_cropped, mask_cropped, clip_percentile)

    # Save
    ct_out = nib.Nifti1Image(ct_normalized.astype(np.float32), crop_affine)
    mr_out = nib.Nifti1Image(mr_normalized.astype(np.float32), crop_affine)
    nib.save(ct_out, os.path.join(out_subject_dir, "ct.nii"))
    nib.save(mr_out, os.path.join(out_subject_dir, "mr.nii"))

    return subject_name, ct_normalized.shape, None


def preprocess_subject_safe(args_tuple):
    """Wrapper that catches exceptions so one failure doesn't kill the pool."""
    try:
        return preprocess_subject(args_tuple)
    except Exception as e:
        subject_name = Path(args_tuple[0]).name
        return subject_name, None, str(e)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_subject_dirs(input_dir, ct_name, mr_name, mask_name):
    """Find all subject directories containing ct, mr, and mask files."""
    subject_dirs = []
    for entry in sorted(os.listdir(input_dir)):
        subject_path = os.path.join(input_dir, entry)
        if not os.path.isdir(subject_path):
            continue
        ct_path = os.path.join(subject_path, ct_name)
        mr_path = os.path.join(subject_path, mr_name)
        mask_path = os.path.join(subject_path, mask_name)
        if os.path.exists(ct_path) and os.path.exists(mr_path) and os.path.exists(mask_path):
            subject_dirs.append(subject_path)
    return subject_dirs


def write_data_csv(output_dir, subject_names, train_ratio=0.7, test_ratio=0.2,
                   seed=42):
    """Write data.csv with pid,set columns (70/20/10 train/test/valid)."""
    random.seed(seed)
    shuffled = subject_names.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = round(n_total * train_ratio)
    n_test = round(n_total * test_ratio)

    train_set = set(shuffled[:n_train])
    test_set = set(shuffled[n_train:n_train + n_test])
    # Remainder is valid

    csv_path = os.path.join(output_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "set"])
        for name in sorted(subject_names):
            if name in train_set:
                split = "train"
            elif name in test_set:
                split = "test"
            else:
                split = "valid"
            writer.writerow([name, split])

    n_valid = n_total - n_train - n_test
    print(f"Data CSV: {csv_path} ({n_train} train, {n_test} test, {n_valid} valid)")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SynthRAD Task1 Brain dataset for CT2MRI fine-tuning"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Root directory containing subject folders")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save preprocessed subject folders")
    parser.add_argument("--target_spacing", type=float, default=1.0,
                        help="Isotropic voxel size in mm (default: 1.0)")
    parser.add_argument("--padding", type=int, default=4,
                        help="Padding voxels around brain bounding box (default: 4)")
    parser.add_argument("--clip_percentile", type=float, default=99.5,
                        help="Percentile for outlier clipping (default: 99.5)")
    parser.add_argument("--height", type=int, default=180,
                        help="Target slice height in pixels (default: 180)")
    parser.add_argument("--width", type=int, default=180,
                        help="Target slice width in pixels (default: 180)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: number of CPU cores)")
    parser.add_argument("--ct_name", default="ct.nii.gz",
                        help="CT filename in each subject dir (default: ct.nii.gz)")
    parser.add_argument("--mr_name", default="mr.nii.gz",
                        help="MR filename in each subject dir (default: mr.nii.gz)")
    parser.add_argument("--mask_name", default="mask.nii.gz",
                        help="Mask filename in each subject dir (default: mask.nii.gz)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subject_dirs = find_subject_dirs(args.input_dir, args.ct_name, args.mr_name, args.mask_name)
    if not subject_dirs:
        print(f"No valid subject directories found in {args.input_dir}")
        print(f"Expected each subject dir to contain: {args.ct_name}, {args.mr_name}, {args.mask_name}")
        sys.exit(1)

    n_workers = args.workers or os.cpu_count()
    print(f"Found {len(subject_dirs)} subjects")
    print(f"Workers: {n_workers}")
    print(f"Target spacing: {args.target_spacing}mm isotropic")
    print(f"Resize: {args.height}x{args.width}")
    print(f"Clip percentile: {args.clip_percentile}")
    print(f"Output: {args.output_dir}\n")

    task_args = [
        (sd, args.output_dir, args.target_spacing, args.padding, args.clip_percentile,
         args.height, args.width, args.ct_name, args.mr_name, args.mask_name)
        for sd in subject_dirs
    ]

    successful_subjects = []
    failed = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(preprocess_subject_safe, ta): ta[0] for ta in task_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            name, shape, error = future.result()
            if error is None:
                successful_subjects.append(name)
                tqdm.write(f"  OK  {name} -> {shape}")
            else:
                failed.append((name, error))
                tqdm.write(f"  FAIL {name}: {error}")

    if successful_subjects:
        write_data_csv(args.output_dir, successful_subjects, seed=args.seed)

    print(f"\nDone. {len(successful_subjects)}/{len(subject_dirs)} succeeded.")
    if failed:
        print("\nFailed subjects:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
