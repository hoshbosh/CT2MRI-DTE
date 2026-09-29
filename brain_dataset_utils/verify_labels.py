#!/usr/bin/env python3
"""Verification pass for transported SynthSeg labels.

Three checks, none optional:
  1. PNG overlays of labels on the MR for a sample of subjects across all splits,
     so a human can confirm the structures land on the right anatomy.
  2. Per-structure volume statistics across the cohort.
  3. Outlier flagging -- both robust (median/MAD across this cohort) and against
     coarse published adult volume ranges.

Outliers are reported, not dropped. A subject failing a check is a finding.

Usage:
    python brain_dataset_utils/verify_labels.py \
        --out_dir /blue/.../out-fine \
        --data_csv /blue/.../out-fine/data.csv \
        --report_dir ./datasets/label_qc
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain_dataset_utils.make_labels import load_manifest  # noqa: E402
from brain_dataset_utils.geometry import (  # noqa: E402
    BILATERAL_GROUPS, STRUCTURE_NAMES, GeometryParams,
)

# Coarse per-hemisphere adult volume ranges in mm3, used only to flag
# implausible segmentations. Deliberately wide: these bound "obviously wrong",
# not "normal". Sources vary by method and cohort, so a flag here means look at
# the overlay, not that the subject is bad.
PLAUSIBLE_MM3 = {
    'thalamus': (4000, 11000),
    'caudate': (2000, 6500),
    'putamen': (2500, 7500),
    'pallidum': (800, 3200),
    'hippocampus': (2000, 6000),
    'amygdala': (700, 3000),
}

OVERLAY_COLORS = np.array([
    [0, 0, 0], [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
    [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
    [153, 153, 153], [26, 188, 156], [190, 174, 212], [253, 192, 134],
], dtype=np.uint8)


def load_subject(out_dir, subject):
    seg_path = os.path.join(out_dir, subject, "seg.nii")
    mr_path = os.path.join(out_dir, subject, "mr.nii")
    geom_path = os.path.join(out_dir, subject, "geometry.json")
    for path in (seg_path, mr_path, geom_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{subject}: missing {path}")
    seg = np.asanyarray(nib.load(seg_path).dataobj).astype(np.uint8)
    mr = np.asanyarray(nib.load(mr_path).dataobj).astype(np.float32)
    params = GeometryParams.from_json(geom_path)
    if seg.shape != mr.shape:
        raise ValueError(f"{subject}: seg {seg.shape} != mr {mr.shape}")
    return mr, seg, params


def subject_volumes(seg, params):
    voxel_mm3 = float(np.prod(params.spacing_mm))
    row = {}
    for code, name in STRUCTURE_NAMES.items():
        row[name] = float((seg == code).sum()) * voxel_mm3
    for group, codes in BILATERAL_GROUPS.items():
        row[group] = sum(row[STRUCTURE_NAMES[c]] for c in codes)
    row['voxel_mm3'] = voxel_mm3
    return row


def mask_coverage(mr, seg, close_kernel=9):
    """Compare the old and corrected training brain masks, and how much deep gray each covers.

    The training mask thresholded x0 in [-1, 1] against a value meant for [0, 1],
    which is equivalent to a [0,1] threshold of 0.525. The corrected mask uses
    0.05 in [0, 1], matching runners/eval.py. Deep-gray coverage is the number
    that matters: loss weighting only reaches structures inside the mask.
    """
    struct = np.ones((close_kernel, close_kernel), dtype=bool)

    def build(thr):
        m = mr > thr
        for z in range(m.shape[2]):
            m[:, :, z] = binary_closing(m[:, :, z], structure=struct)
        return m

    old = build(0.525)   # what the code actually did before the unit fix
    new = build(0.05)    # what it was meant to do, and what eval reports on
    deep = seg > 0
    n_deep = int(deep.sum())
    return {
        'mask_frac_old': float(old.mean()),
        'mask_frac_new': float(new.mean()),
        'deepgray_covered_old': float((old & deep).sum() / n_deep) if n_deep else np.nan,
        'deepgray_covered_new': float((new & deep).sum() / n_deep) if n_deep else np.nan,
    }


def write_overlay(mr, seg, subject, split, report_dir):
    """Three axial slices spanning the labelled extent, MR in grey + labels in colour."""
    labelled_z = np.where((seg > 0).any(axis=(0, 1)))[0]
    if len(labelled_z) == 0:
        raise ValueError(f"{subject}: no labelled slices to overlay")
    picks = labelled_z[[len(labelled_z) // 4, len(labelled_z) // 2, 3 * len(labelled_z) // 4]]

    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4.4))
    for ax, z in zip(np.atleast_1d(axes), picks):
        base = mr[:, :, z]
        denom = base.max() if base.max() > 0 else 1.0
        rgb = np.repeat((base / denom)[:, :, None], 3, axis=2)
        lab = seg[:, :, z]
        colored = OVERLAY_COLORS[lab] / 255.0
        alpha = (lab > 0)[:, :, None] * 0.55
        ax.imshow(np.clip(rgb * (1 - alpha) + colored * alpha, 0, 1), origin="lower")
        ax.set_title(f"z={z}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{subject}  ({split})  labels on MR", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(report_dir, "overlays", f"{subject}_{split}.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Verify transported SynthSeg labels")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--report_dir", default="./datasets/label_qc")
    ap.add_argument("--manifest", default=None,
                    help="Phase 1 QC manifest. Structures it excludes are zeroed in "
                         "seg.nii by make_labels.py, so they are held out of the cohort "
                         "statistics and never re-flagged as outliers.")
    ap.add_argument("--n_overlay", type=int, default=12,
                    help="Subjects to render overlays for, spread across splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.report_dir, "overlays"), exist_ok=True)
    splits = pd.read_csv(args.data_csv).set_index("pid")["set"].to_dict()

    excluded = {}
    if args.manifest:
        for subject, (_, codes) in load_manifest(args.manifest).items():
            groups = {g for g, members in BILATERAL_GROUPS.items()
                      if set(members) & set(codes)}
            if groups:
                excluded[subject] = groups

    rows, failures = [], []
    total = len(splits)
    # Reading two ~190-slice volumes per subject off /blue takes minutes; report
    # progress so a slow run is distinguishable from a hung one.
    for i, subject in enumerate(sorted(splits), 1):
        if i % 20 == 0 or i == total:
            print(f"  read {i}/{total} subjects", flush=True)
        try:
            mr, seg, params = load_subject(args.out_dir, subject)
        except Exception as exc:
            failures.append((subject, str(exc)))
            print(f"  FAIL  {subject}: {exc}", flush=True)
            continue
        row = subject_volumes(seg, params)
        row.update(mask_coverage(mr, seg))
        row.update(pid=subject, split=splits[subject],
                   spacing_x=params.spacing_mm[0], spacing_y=params.spacing_mm[1],
                   spacing_z=params.spacing_mm[2])
        rows.append(row)

    if not rows:
        raise SystemExit("No subjects could be read; nothing to verify.")
    df = pd.DataFrame(rows).set_index("pid")

    # Manifest-excluded structures are zeroed in seg.nii. Holding them out of
    # both the statistics and the flagging keeps a deliberate exclusion from
    # masquerading as a fresh 0mm3 outlier, and stops the zeros from dragging
    # the cohort median down.
    held_out = pd.DataFrame(False, index=df.index, columns=list(BILATERAL_GROUPS))
    for subject, groups in excluded.items():
        if subject in held_out.index:
            held_out.loc[subject, list(groups)] = True

    # Robust outlier detection: |x - median| / (1.4826 * MAD) > 3.5
    flags = {}
    for group in BILATERAL_GROUPS:
        keep = ~held_out[group]
        v = df.loc[keep, group]
        med = v.median()
        mad = np.median(np.abs(v - med))
        scale = 1.4826 * mad if mad > 0 else np.nan
        z = np.abs(v - med) / scale if scale and np.isfinite(scale) else pd.Series(0.0, index=v.index)
        df[f"{group}_robust_z"] = z
        lo, hi = PLAUSIBLE_MM3[group]
        # Bilateral totals are compared against a doubled per-hemisphere range.
        implausible = (v < 2 * lo) | (v > 2 * hi)
        for pid in v.index[(z > 3.5) | implausible]:
            flags.setdefault(pid, []).append(group)

    df.to_csv(os.path.join(args.report_dir, "label_volumes.csv"))

    stats_frame = df[list(BILATERAL_GROUPS)].mask(held_out)
    summary = stats_frame.agg(["mean", "std", "min", "median", "max"]).T
    summary.to_csv(os.path.join(args.report_dir, "label_volume_summary.csv"))

    print("\nPer-structure bilateral volumes across the cohort (mm3):")
    print(summary.round(1).to_string())
    if excluded:
        n = sum(len(g) for g in excluded.values())
        print(f"  ({n} structure-subject pair(s) held out per {args.manifest})")

    print(f"\nVoxel size: {df['voxel_mm3'].min():.4f} - {df['voxel_mm3'].max():.4f} mm3 "
          f"(median {df['voxel_mm3'].median():.4f})")

    print("\nBrain-mask comparison (old = pre-fix [-1,1]/[0,1] unit mismatch, new = corrected):")
    print(f"  image fraction in mask:      old {df['mask_frac_old'].mean():.3f}   "
          f"new {df['mask_frac_new'].mean():.3f}")
    print(f"  deep-gray voxels in mask:    old {df['deepgray_covered_old'].mean():.3f}   "
          f"new {df['deepgray_covered_new'].mean():.3f}")
    print("  (deep-gray coverage is the share of labelled structure voxels the "
          "loss weighting actually reaches)")

    rng = np.random.default_rng(args.seed)
    sample = []
    for split in sorted(set(splits.values())):
        pool = df.index[df["split"] == split].tolist()
        take = min(len(pool), max(1, round(args.n_overlay * len(pool) / len(df))))
        sample += rng.choice(pool, size=take, replace=False).tolist()
    # Always render the flagged subjects, they are the ones worth eyeballing.
    sample = sorted(set(sample) | set(flags))

    print(f"\nRendering {len(sample)} overlays to {args.report_dir}/overlays/")
    for subject in sample:
        mr, seg, _ = load_subject(args.out_dir, subject)
        write_overlay(mr, seg, subject, splits[subject], args.report_dir)

    if excluded:
        print("\nEXCLUDED per manifest (zeroed in seg.nii, held out of the stats above):")
        for subject, groups in sorted(excluded.items()):
            print(f"  {subject}: {', '.join(sorted(groups))}")

    if flags:
        print(f"\nOUTLIERS -- {len(flags)} subject(s) flagged, inspect their overlays:")
        for pid, groups in sorted(flags.items()):
            vols = ", ".join(f"{g}={df.loc[pid, g]:.0f}mm3" for g in groups)
            print(f"  {pid} ({df.loc[pid, 'split']}): {vols}")
    else:
        print("\nNo volume outliers flagged.")

    print(f"\n{len(df)}/{len(splits)} subjects verified.")
    if failures:
        print("UNREADABLE:")
        for subject, msg in failures:
            print(f"  {subject}: {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
