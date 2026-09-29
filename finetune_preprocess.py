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

from brain_dataset_utils.geometry import derive_geometry, apply_geometry
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# The reorient / resample / crop / resize chain lives in
# brain_dataset_utils/geometry.py so that segmentation label maps can be pushed
# through the identical steps with nearest-neighbour interpolation.

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
        ct_name, mr_name, mask_name, geometry_only = args_tuple

    subject_name = Path(subject_dir).name
    out_subject_dir = os.path.join(output_dir, subject_name)
    os.makedirs(out_subject_dir, exist_ok=True)

    # Load CT, MR, and mask
    mr_img = nib.load(os.path.join(subject_dir, mr_name))
    mask_img = nib.load(os.path.join(subject_dir, mask_name))
    mask_data = mask_img.get_fdata().astype(bool)
    # Derive the transform chain once from the MR grid, then replay it on every
    # volume. Labels reuse the same record via geometry.GeometryParams.from_json.
    params = derive_geometry(mr_img, mask_data, subject_name,
                             target_spacing=target_spacing, padding=padding,
                             height=height, width=width)
    params.to_json(os.path.join(out_subject_dir, "geometry.json"))

    if geometry_only:
        # Emit only the transform record, leaving existing ct.nii/mr.nii untouched.
        # Used to backfill geometry.json for a cohort preprocessed before it existed.
        return subject_name, None, None

    ct_data = nib.load(os.path.join(subject_dir, ct_name)).get_fdata().astype(np.float64)
    mr_data = mr_img.get_fdata().astype(np.float64)

    ct_cropped = apply_geometry(ct_data, params, order=1)
    mr_cropped = apply_geometry(mr_data, params, order=1)
    mask_cropped = apply_geometry(mask_data.astype(np.float64), params, order=0) > 0.5
    crop_affine = np.asarray(params.crop_affine)

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
    parser.add_argument("--height", type=int, default=256,
                        help="Target slice height in pixels (default: 256)")
    parser.add_argument("--width", type=int, default=256,
                        help="Target slice width in pixels (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--geometry_only", action="store_true",
                        help="Write geometry.json only; do not touch existing ct.nii/mr.nii")
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
    print(f"Output: {args.output_dir}")
    if args.geometry_only:
        print("Mode: geometry_only -- writing geometry.json, leaving ct.nii/mr.nii untouched")
    print()

    task_args = [
        (sd, args.output_dir, args.target_spacing, args.padding, args.clip_percentile,
         args.height, args.width, args.ct_name, args.mr_name, args.mask_name,
         args.geometry_only)
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
