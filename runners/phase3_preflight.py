"""Pre-launch checks for the Phase 3 structure-weighted arms.

Runs on the HPC against the real training HDF5. Answers three questions that
are expensive to discover after a 20-hour job:

  1. Does the training HDF5 actually carry LABEL_dataset?
  2. What fraction of pixels are deep gray, uniform vs shell?
  3. By how much does the deep-gray term shift the global loss scale?

(3) matters because the weight map is not normalised -- recloss is
`(diff * loss_weight).mean()` -- so a large shift would change the effective
learning rate and stop arm A from being a clean control.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.BrownianBridge.BrownianBridgeModel import deepgray_mask_from_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdf5', required=True, help='Training HDF5 (must carry LABEL_dataset)')
    ap.add_argument('--mask_weight', type=float, default=3.0)
    ap.add_argument('--deepgray_weight', type=float, default=5.0)
    ap.add_argument('--mask_threshold', type=float, default=0.05)
    ap.add_argument('--dilate', type=int, default=2)
    ap.add_argument('--erode', type=int, default=1)
    ap.add_argument('--n_slices', type=int, default=2000, help='total slices to sample')
    ap.add_argument('--n_blocks', type=int, default=25,
                    help='read the sample as this many contiguous runs rather than '
                         'scattered single slices; the HDF5 is gzip-chunked, so '
                         'scattered reads decompress a chunk per slice')
    args = ap.parse_args()

    with h5py.File(args.hdf5, 'r') as hf:
        keys = list(hf.keys())
        print(f"datasets in {os.path.basename(args.hdf5)}: {keys}")
        if 'LABEL_dataset' not in hf:
            raise SystemExit(
                "FAIL: no LABEL_dataset. Rebuild with generate_total_hdf5_csv.py --SEG_name "
                "before launching any Phase 3 arm."
            )
        n = hf['MR_dataset'].shape[2]
        if hf['LABEL_dataset'].shape[2] != n:
            raise SystemExit(
                f"FAIL: {hf['LABEL_dataset'].shape[2]} label slices vs {n} image slices"
            )
        print(f"OK: {n} slices, labels present and synchronised")

        # Contiguous runs spread across the stack: same coverage as a scattered
        # sample, but each run is one sequential read instead of one gzip chunk
        # decompression per slice. Slices are flat-concatenated across subjects,
        # so spreading the run starts evenly still spans many subjects.
        n_take = min(args.n_slices, n)
        n_blocks = max(1, min(args.n_blocks, n_take))
        run = max(1, n_take // n_blocks)
        starts = np.linspace(0, max(n - run, 0), n_blocks).astype(int)
        print(f"\nsampling {n_blocks} runs of {run} contiguous slices "
              f"({n_blocks * run} total) ...", flush=True)

        mr_parts, lab_parts = [], []
        for j, st in enumerate(starts):
            sl = slice(st, min(st + run, n))
            mr_parts.append(np.asarray(hf['MR_dataset'][:, :, sl], dtype=np.float32))
            lab_parts.append(np.asarray(hf['LABEL_dataset'][:, :, sl], dtype=np.int64))
            print(f"  run {j + 1}/{n_blocks}  slices {sl.start}:{sl.stop}", flush=True)
        mr = np.concatenate(mr_parts, axis=2).transpose(2, 0, 1)
        lab = np.concatenate(lab_parts, axis=2).transpose(2, 0, 1)
        print(f"read {mr.shape[0]} slices", flush=True)

    codes = np.unique(lab)
    print(f"\nlabel codes present: {codes.tolist()}")
    if codes.max() > 12:
        raise SystemExit(f"FAIL: label code {codes.max()} exceeds the 12 compact codes")
    missing = sorted(set(range(1, 13)) - set(codes.tolist()))
    if missing:
        print(f"  NOTE: codes absent from this sample: {missing}")

    lab_t = torch.from_numpy(lab)[:, None]
    mr_t = torch.from_numpy(mr)[:, None]

    # MR is stored in [0,1]; p_losses thresholds x0 rescaled to the same units.
    brain = (mr_t > args.mask_threshold).float()

    print(f"\n{'':16s}{'frac of all px':>16s}{'frac of in-brain':>18s}")
    frac_brain = brain.mean().item()
    print(f"{'in-brain':16s}{frac_brain:16.4f}{1.0:18.4f}")

    results = {}
    for mode in ('uniform', 'shell'):
        dg = deepgray_mask_from_labels(lab_t, mode, args.dilate, args.erode)
        f_all = dg.mean().item()
        f_brain = (dg * brain).sum().item() / max(brain.sum().item(), 1)
        results[mode] = (dg, f_all, f_brain)
        print(f"{'deep gray ' + mode:16s}{f_all:16.4f}{f_brain:18.4f}")

    print(f"\nloss-scale shift (weights are NOT normalised, so this is the "
          f"effective LR change):")
    base_w = 1.0 + (args.mask_weight - 1.0) * brain
    base_mean = base_w.mean().item()
    print(f"  arm A (control)          mean weight {base_mean:.4f}   (reference)")
    for mode, (dg, _, _) in results.items():
        w = base_w * (1.0 + (args.deepgray_weight - 1.0) * dg)
        m = w.mean().item()
        arm = 'B' if mode == 'uniform' else 'E'
        print(f"  arm {arm} ({mode:7s})        mean weight {m:.4f}   "
              f"{100 * (m / base_mean - 1):+.1f}% vs control")

    print("\nper-slice deep-gray coverage (uniform), so an empty-label batch is visible:")
    dg = results['uniform'][0]
    per = dg.flatten(1).mean(1)
    n_empty = int((per == 0).sum())
    print(f"  slices with no structures: {n_empty}/{len(per)} "
          f"({100 * n_empty / len(per):.1f}%)  <- expected: many, deep gray spans "
          f"only part of the axial stack")
    nz = per[per > 0]
    if len(nz):
        print(f"  on slices that have structures: mean {nz.mean():.4f}, max {nz.max():.4f}")

    print("\nPREFLIGHT OK")


if __name__ == '__main__':
    main()
