#!/usr/bin/env python3
"""Push SynthSeg subcortical labels onto the CT2MRI training grid.

Run AFTER mri_synthseg has produced a segmentation of each subject's ORIGINAL
(native-geometry) MR. SynthSeg is resolution-aware and the preprocessed
out-fine/<pid>/mr.nii carries a stale affine (the in-plane resize is not
reflected in it), so segmenting the native volume and transporting the labels
is the only correct order of operations.

Per subject:
  1. resample the SynthSeg label map onto the native MR voxel grid (nearest)
  2. remap FreeSurfer aseg values to the compact 1..12 deep-gray coding
  3. replay the exact preprocessing geometry from geometry.json (nearest)
  4. write out-fine/<pid>/seg.nii as uint8, on the same grid as mr.nii

Fails loudly. A subject with a missing input, a shape mismatch, an empty label
map, or a missing structure is reported as a failure and produces no output --
never a silently empty label map that later gets treated as ground truth.

Usage:
    python brain_dataset_utils/make_labels.py \
        --synthseg_dir /blue/.../synthseg \
        --native_dir   /blue/.../synthrad/brain \
        --out_dir      /blue/.../out-fine \
        --subject 1BA001
    # or omit --subject to process every subject with a geometry.json
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain_dataset_utils.geometry import (  # noqa: E402
    ASEG_LABEL_MAP, STRUCTURE_NAMES, GeometryParams, apply_geometry,
)


def build_lut():
    """Dense lookup table: FreeSurfer aseg value -> compact code, 0 elsewhere."""
    lut = np.zeros(max(ASEG_LABEL_MAP) + 1, dtype=np.uint8)
    for aseg_value, compact in ASEG_LABEL_MAP.items():
        lut[aseg_value] = compact
    return lut


NAME_TO_CODE = {name: code for code, name in STRUCTURE_NAMES.items()}


def load_manifest(path):
    """Read the Phase 1 QC manifest: per-subject SynthSeg mode and exclusions.

    Absent subjects get mode 'default' and no exclusions. Returns
    {subject: (mode, frozenset(codes))}. Unknown structure names and unknown
    modes are hard errors -- a typo here would silently un-exclude a structure
    that QC found unusable.
    """
    manifest = {}
    if path is None:
        return manifest
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith('#') or line.startswith('subject,'):
                continue
            fields = line.split(',')
            if len(fields) < 3:
                raise ValueError(f"{path}:{lineno}: expected >=3 comma-separated fields")
            subject, mode, excluded = fields[0].strip(), fields[1].strip(), fields[2].strip()
            if mode not in ('default', 'robust'):
                raise ValueError(f"{path}:{lineno}: unknown synthseg_mode {mode!r}")
            codes = set()
            for name in (n.strip() for n in excluded.split(';') if n.strip()):
                if name not in NAME_TO_CODE:
                    raise ValueError(
                        f"{path}:{lineno}: unknown structure {name!r}; "
                        f"expected one of {sorted(NAME_TO_CODE)}"
                    )
                codes.add(NAME_TO_CODE[name])
            manifest[subject] = (mode, frozenset(codes))
    return manifest


def make_subject_labels(subject, synthseg_dir, native_dir, out_dir,
                        mr_name="mr.nii.gz", overwrite=False, excluded=frozenset()):
    out_subject_dir = os.path.join(out_dir, subject)
    seg_out_path = os.path.join(out_subject_dir, "seg.nii")
    geom_path = os.path.join(out_subject_dir, "geometry.json")

    if os.path.exists(seg_out_path) and not overwrite:
        return {"subject": subject, "status": "skipped", "reason": "seg.nii exists"}

    if not os.path.exists(geom_path):
        raise FileNotFoundError(
            f"{subject}: no geometry.json in {out_subject_dir}. Re-run "
            "finetune_preprocess.py so the transform record exists."
        )
    params = GeometryParams.from_json(geom_path)

    seg_path = os.path.join(synthseg_dir, f"{subject}_synthseg.nii.gz")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"{subject}: SynthSeg output missing at {seg_path}")

    native_mr_path = os.path.join(native_dir, subject, mr_name)
    if not os.path.exists(native_mr_path):
        raise FileNotFoundError(f"{subject}: native MR missing at {native_mr_path}")

    seg_img = nib.load(seg_path)
    mr_img = nib.load(native_mr_path)

    # SynthSeg writes its output on an internally-resampled 1mm grid, not
    # necessarily the input grid, so put it back on the native MR grid first.
    if seg_img.shape[:3] != mr_img.shape[:3] or not np.allclose(seg_img.affine, mr_img.affine):
        seg_img = resample_from_to(seg_img, (mr_img.shape[:3], mr_img.affine), order=0)
    seg_native = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int32)

    lut = build_lut()
    seg_native = np.where(seg_native < len(lut), seg_native, 0)
    compact = lut[seg_native]

    present = set(np.unique(compact).tolist()) - {0}
    # Excluded structures were judged unusable by the Phase 1 QC pass, so their
    # absence is expected rather than a failure. They are zeroed below so the
    # loss and the per-structure metrics never see them.
    missing = sorted(set(STRUCTURE_NAMES) - present - set(excluded))
    if missing:
        raise ValueError(
            f"{subject}: SynthSeg produced no voxels for "
            f"{[STRUCTURE_NAMES[m] for m in missing]} -- segmentation is incomplete, refusing to write"
        )

    if excluded:
        compact = np.where(np.isin(compact, list(excluded)), 0, compact)
        present = present - set(excluded)

    seg_grid = apply_geometry(compact, params, order=0)
    seg_grid = np.rint(seg_grid).astype(np.uint8)

    present_after = set(np.unique(seg_grid).tolist()) - {0}
    lost = sorted(present - present_after)
    if lost:
        raise ValueError(
            f"{subject}: {[STRUCTURE_NAMES[l] for l in lost]} survived segmentation but "
            "vanished in the crop/resize -- the structure fell outside the brain-mask "
            "bounding box or was resampled away"
        )

    voxel_mm3 = float(np.prod(params.spacing_mm))
    volumes = {
        STRUCTURE_NAMES[code]: float((seg_grid == code).sum()) * voxel_mm3
        for code in sorted(STRUCTURE_NAMES)
    }

    os.makedirs(out_subject_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(seg_grid, np.asarray(params.crop_affine)), seg_out_path)

    return {
        "subject": subject,
        "status": "ok",
        "excluded": sorted(STRUCTURE_NAMES[c] for c in excluded),
        "shape": tuple(seg_grid.shape),
        "voxel_mm3": voxel_mm3,
        "spacing_mm": params.spacing_mm,
        "volumes_mm3": volumes,
    }


def main():
    ap = argparse.ArgumentParser(description="Transport SynthSeg labels onto the training grid")
    ap.add_argument("--synthseg_dir", required=True, help="Directory of <pid>_synthseg.nii.gz")
    ap.add_argument("--robust_dir", default=None,
                    help="Directory of --robust SynthSeg output, for subjects the "
                         "manifest marks synthseg_mode=robust")
    ap.add_argument("--manifest", default=None,
                    help="Phase 1 QC manifest (subject,synthseg_mode,excluded_structures,...)")
    ap.add_argument("--native_dir", required=True, help="Original SynthRAD subject directories")
    ap.add_argument("--out_dir", required=True, help="Preprocessed out-fine directory")
    ap.add_argument("--mr_name", default="mr.nii.gz", help="Native MR filename")
    ap.add_argument("--subject", default=None, help="Single subject; omit to process all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if any(mode == 'robust' for mode, _ in manifest.values()) and not args.robust_dir:
        raise SystemExit(
            f"{args.manifest} marks subjects as synthseg_mode=robust but --robust_dir "
            "was not given; refusing to silently fall back to the default pass"
        )

    if args.subject:
        subjects = [args.subject]
    else:
        subjects = sorted(
            d for d in os.listdir(args.out_dir)
            if os.path.exists(os.path.join(args.out_dir, d, "geometry.json"))
        )
    if not subjects:
        raise SystemExit(f"No subjects with geometry.json under {args.out_dir}")

    failures = []
    for subject in subjects:
        mode, excluded = manifest.get(subject, ('default', frozenset()))
        source_dir = args.robust_dir if mode == 'robust' else args.synthseg_dir
        try:
            result = make_subject_labels(subject, source_dir, args.native_dir,
                                         args.out_dir, args.mr_name, args.overwrite,
                                         excluded=excluded)
        except Exception as exc:
            failures.append((subject, str(exc)))
            print(f"  FAIL  {subject}: {exc}", flush=True)
            continue
        if result["status"] == "skipped":
            print(f"  SKIP  {subject}: {result['reason']}", flush=True)
        else:
            vols = result["volumes_mm3"]
            note = ''
            if mode != 'default':
                note += f"  mode={mode}"
            if result['excluded']:
                note += f"  EXCLUDED={','.join(result['excluded'])}"
            print(f"  OK    {subject}  shape={result['shape']}  "
                  f"voxel={result['voxel_mm3']:.4f}mm3  "
                  f"thal={vols['thalamus_L']:.0f}/{vols['thalamus_R']:.0f} "
                  f"pall={vols['pallidum_L']:.0f}/{vols['pallidum_R']:.0f}{note}", flush=True)

    print(f"\nDone. {len(subjects) - len(failures)}/{len(subjects)} succeeded.")
    if failures:
        print("FAILURES:")
        for subject, msg in failures:
            print(f"  {subject}: {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
