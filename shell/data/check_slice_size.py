import argparse
import os
import nibabel as nib


def check_subject(subject_dir, expected_size=180):
    issues = []
    for fname in ("ct.nii", "mr.nii"):
        fpath = os.path.join(subject_dir, fname)
        if not os.path.exists(fpath):
            issues.append(f"  MISSING: {fname}")
            continue
        shape = nib.load(fpath).shape
        if len(shape) < 2 or shape[0] != expected_size or shape[1] != expected_size:
            issues.append(f"  BAD SHAPE {fname}: {shape} (expected {expected_size}x{expected_size} in dims 0,1)")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Check that all ct.nii/mr.nii slices are expected_size x expected_size")
    parser.add_argument("input_dir", help="Directory containing subject subdirectories")
    parser.add_argument("--size", type=int, default=180, help="Expected slice size (default: 180)")
    args = parser.parse_args()

    subjects = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])

    if not subjects:
        print(f"No subdirectories found in {args.input_dir}")
        return

    print(f"Checking {len(subjects)} subjects for {args.size}x{args.size} slices...\n")

    passed = 0
    failed = 0
    for subject in subjects:
        subject_dir = os.path.join(args.input_dir, subject)
        issues = check_subject(subject_dir, args.size)
        if issues:
            print(f"FAIL {subject}")
            for issue in issues:
                print(issue)
            failed += 1
        else:
            passed += 1

    print(f"\n{passed}/{len(subjects)} passed, {failed}/{len(subjects)} failed")


if __name__ == "__main__":
    main()
