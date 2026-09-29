"""Dump tensorboard scalars from the Phase 3 arms to CSV / stdout.

Answers three things the job logs cannot:
  * the validation-loss curve, so we can tell convergence from overfitting
    (only one top_model survives, so the log alone cannot show the shape)
  * whether deepgray_frac was non-zero, i.e. the weighting actually engaged
    rather than silently seeing empty label batches
  * how the aux loss terms moved per arm

Reads event files directly; no tensorboard server needed.
"""
import argparse
import os
import sys
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    sys.exit("needs tensorboard: pip install tensorboard (or conda activate ct2mri)")


def load(run_dir):
    """All scalars under run_dir, merged across event files, sorted by step."""
    series = defaultdict(list)
    found = False
    for root, _, files in os.walk(run_dir):
        for f in files:
            if 'tfevents' not in f:
                continue
            found = True
            ea = event_accumulator.EventAccumulator(
                os.path.join(root, f),
                size_guidance={event_accumulator.SCALARS: 0})
            ea.Reload()
            for tag in ea.Tags().get('scalars', []):
                for ev in ea.Scalars(tag):
                    series[tag].append((ev.step, ev.value))
    if not found:
        raise FileNotFoundError(f"no tfevents files under {run_dir}")
    for tag in series:
        series[tag].sort(key=lambda t: t[0])
    return series


def summarise(name, series, n_show):
    print(f"\n=== {name}")
    if not series:
        print("  (no scalars)")
        return
    for tag in sorted(series):
        pts = series[tag]
        vals = [v for _, v in pts]
        lo_i = min(range(len(vals)), key=lambda i: vals[i])
        print(f"  {tag:28s} n={len(pts):6d}  first={vals[0]:.5f}  "
              f"last={vals[-1]:.5f}  min={vals[lo_i]:.5f} @step {pts[lo_i][0]}")
        if n_show and 'val' in tag.lower():
            step = max(1, len(pts) // n_show)
            trail = [f"{v:.4f}" for _, v in pts[::step]][:n_show]
            print(f"  {'':28s} curve: {' '.join(trail)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256')
    ap.add_argument('--arms', nargs='*',
                    default=['armA_control', 'armB_uniform5x', 'armE_shell5x'])
    ap.add_argument('--prefix', default='241213_256_BBDM_axial_DDIM_MR_p3_')
    ap.add_argument('--csv', default=None, help='write every scalar to this CSV')
    ap.add_argument('--show', type=int, default=12, help='points to print for val curves')
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        run = os.path.join(args.root, args.prefix + arm)
        try:
            series = load(run)
        except FileNotFoundError as e:
            print(f"\n=== {arm}\n  SKIP: {e}")
            continue
        summarise(arm, series, args.show)
        for tag, pts in series.items():
            for step, val in pts:
                rows.append((arm, tag, step, val))

        dg = [v for t, p in series.items() if 'deepgray' in t for _, v in p]
        if dg:
            print(f"  deepgray_frac: mean={sum(dg)/len(dg):.5f} max={max(dg):.5f} "
                  f"-> weighting ENGAGED")
        elif arm != 'armA_control':
            print(f"  WARNING: no deepgray_frac scalar -- weighting may not have engaged")

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['arm', 'tag', 'step', 'value'])
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == '__main__':
    main()
