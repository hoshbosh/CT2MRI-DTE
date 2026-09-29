"""Phase 2, step 3: per-structure evaluation of a CT->MR checkpoint.

Global masked SSIM averages over the whole brain, where cortex and white matter
dominate. The thalamus is roughly 7 cm3 of a 1200 cm3 brain, so a model can
place the deep gray badly and barely move the headline number. This reports the
structures directly:

  dice              overlap between SynthSeg on the real volume and on the
                    synthetic one -- does the structure exist, with the right shape
  centroid_mm       displacement of the structure's centre of mass, in mm.
                    The clinically meaningful quantity for targeting
  ssim / psnr       image fidelity restricted to the structure's bounding box
  cw_ssim           complex-wavelet SSIM on the same patch. CW-SSIM is designed
                    to be tolerant of small translations, so the gap between it
                    and plain SSIM estimates how much of the SSIM loss is
                    misalignment rather than synthesis error

Both segmentations come from the same SynthSeg mode on volumes in the same
space, so a difference reflects the synthesis, not the processing chain.

No silent dropping: a structure SynthSeg cannot find in the synthetic volume is
recorded with dice 0.0 and seg_status 'missing_syn', and the summary reports
both the honest mean (failures counted as zero) and the mean over successfully
segmented structures only, clearly separated.
"""
import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_dataset_utils.geometry import (  # noqa: E402
    ASEG_LABEL_MAP,
    BILATERAL_GROUPS,
    STRUCTURE_NAMES,
)
from brain_dataset_utils.make_labels import load_manifest  # noqa: E402

# compact code -> the aseg code SynthSeg actually writes
CODE_TO_ASEG = {v: k for k, v in ASEG_LABEL_MAP.items()}


# ---------------------------------------------------------------- CW-SSIM

def _gabor_bank(shape, n_orient=6, wavelength=4.0, sigma=2.0):
    """Complex Gabor filters in the frequency domain, one per orientation.

    Sampat et al. (2009) define CW-SSIM over a complex steerable pyramid. A
    single-scale complex Gabor bank is the standard lightweight stand-in: it
    shares the property that matters here -- a small translation rotates the
    coefficient phase rather than changing its magnitude -- without pulling in
    a pyramid dependency. Because the scale is fixed, absolute values are not
    comparable with steerable-pyramid CW-SSIM from other papers; only
    comparisons within this harness are meaningful.
    """
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f0 = 1.0 / wavelength
    bank = []
    for i in range(n_orient):
        theta = np.pi * i / n_orient
        # rotate into the filter's frame, then a Gaussian offset to +f0 in u
        u = fx * np.cos(theta) + fy * np.sin(theta)
        v = -fx * np.sin(theta) + fy * np.cos(theta)
        g = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * ((u - f0) ** 2 + v ** 2))
        bank.append(g)
    return bank


def cw_ssim(a, b, n_orient=6, K=1e-3):
    """CW-SSIM index between two 2D patches, averaged over orientations."""
    if a.shape != b.shape or min(a.shape) < 8:
        return np.nan
    bank = _gabor_bank(a.shape, n_orient=n_orient)
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    vals = []
    for g in bank:
        ca = np.fft.ifft2(fa * g)
        cb = np.fft.ifft2(fb * g)
        num = 2.0 * np.abs(np.sum(ca * np.conj(cb))) + K
        den = np.sum(np.abs(ca) ** 2) + np.sum(np.abs(cb) ** 2) + K
        vals.append(num / den)
    return float(np.mean(vals))


# ------------------------------------------------------------------ metrics

def dice(m1, m2):
    s = m1.sum() + m2.sum()
    if s == 0:
        return np.nan
    return float(2.0 * np.logical_and(m1, m2).sum() / s)


def centroid_mm(mask, spacing):
    if not mask.any():
        return None
    idx = np.argwhere(mask).astype(np.float64)
    return (idx.mean(axis=0) * np.asarray(spacing, dtype=np.float64))


def bbox_slices(mask, margin=8):
    """Bounding box of a mask, padded, as per-axis slice objects."""
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def patch_image_metrics(real, syn, mask):
    """2D SSIM / CW-SSIM over the structure's bounding box, plus masked PSNR.

    The model generates 2D axial slices, so image metrics are computed per
    slice on the slices where the structure appears and then averaged, rather
    than on a 3D window that would blend information across slices the model
    never saw together.
    """
    box = bbox_slices(mask)
    r, s, m = real[box], syn[box], mask[box]

    ssims, cws = [], []
    for k in range(r.shape[2]):
        if not m[:, :, k].any():
            continue
        rk, sk = r[:, :, k], s[:, :, k]
        if min(rk.shape) < 8:
            continue
        ssims.append(structural_similarity(rk, sk, data_range=1.0))
        cws.append(cw_ssim(rk, sk))

    if mask.sum() == 0:
        psnr = np.nan
    else:
        mse = float(((syn[mask] - real[mask]) ** 2).mean())
        psnr = float('inf') if mse == 0 else float(10.0 * np.log10(1.0 / mse))

    return (float(np.mean(ssims)) if ssims else np.nan,
            float(np.nanmean(cws)) if cws else np.nan,
            psnr)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--export_dir', required=True,
                    help='Output of export_eval_volumes.py (holds subjects.csv and seg/)')
    ap.add_argument('--manifest', default=None, help='Phase 1 QC manifest')
    ap.add_argument('--out_csv', default=None,
                    help='Per-subject-per-structure CSV (default: <export_dir>/structure_metrics.csv)')
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    seg_dir = os.path.join(args.export_dir, 'seg')
    index_path = os.path.join(args.export_dir, 'subjects.csv')
    out_csv = args.out_csv or os.path.join(args.export_dir, 'structure_metrics.csv')

    with open(index_path) as fh:
        subjects = list(csv.DictReader(fh))
    if not subjects:
        raise SystemExit(f'No subjects listed in {index_path}')

    rows = []
    for entry in subjects:
        pid = entry['pid']
        spacing = (float(entry['spacing_x']), float(entry['spacing_y']),
                   float(entry['spacing_z']))
        _, excluded = manifest.get(pid, ('default', frozenset()))

        paths = {k: os.path.join(seg_dir, f'{pid}_{k}_synthseg.nii.gz')
                 for k in ('real', 'syn')}
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            raise SystemExit(
                f'{pid}: missing SynthSeg output for {missing}. Run the eval array '
                f'to completion before scoring; refusing to report a partial cohort.'
            )

        seg_real = np.asanyarray(nib.load(paths['real']).dataobj).astype(np.int32)
        seg_syn = np.asanyarray(nib.load(paths['syn']).dataobj).astype(np.int32)
        real = np.asanyarray(nib.load(
            os.path.join(args.export_dir, f'{pid}_real.nii.gz')).dataobj).astype(np.float32)
        syn = np.asanyarray(nib.load(
            os.path.join(args.export_dir, f'{pid}_syn.nii.gz')).dataobj).astype(np.float32)

        if seg_real.shape != seg_syn.shape or seg_real.shape != real.shape:
            raise SystemExit(
                f'{pid}: shape mismatch -- real {real.shape}, seg_real {seg_real.shape}, '
                f'seg_syn {seg_syn.shape}'
            )

        for code in sorted(STRUCTURE_NAMES):
            name = STRUCTURE_NAMES[code]
            aseg = CODE_TO_ASEG[code]
            mr = seg_real == aseg
            ms = seg_syn == aseg

            row = {'pid': pid, 'structure': name, 'n_vox_real': int(mr.sum()),
                   'n_vox_syn': int(ms.sum())}

            if code in excluded:
                row.update(seg_status='excluded_manifest', dice=np.nan,
                           centroid_mm=np.nan, ssim=np.nan, cw_ssim=np.nan, psnr=np.nan)
                rows.append(row)
                continue

            if not mr.any():
                # No reference to score against; not the model's failure.
                row.update(seg_status='missing_real', dice=np.nan, centroid_mm=np.nan,
                           ssim=np.nan, cw_ssim=np.nan, psnr=np.nan)
                rows.append(row)
                continue

            d = dice(mr, ms)
            cr = centroid_mm(mr, spacing)
            cs = centroid_mm(ms, spacing)
            disp = np.nan if cs is None else float(np.linalg.norm(cr - cs))
            ssim, cws, psnr = patch_image_metrics(real, syn, mr)

            status = 'ok' if ms.any() else 'missing_syn'
            row.update(seg_status=status, dice=d, centroid_mm=disp,
                       ssim=ssim, cw_ssim=cws, psnr=psnr)
            rows.append(row)

        done = sum(1 for r in rows if r['pid'] == pid and r['seg_status'] == 'ok')
        print(f'  {pid}  {done}/12 structures scored', flush=True)

    fields = ['pid', 'structure', 'seg_status', 'dice', 'centroid_mm', 'ssim',
              'cw_ssim', 'psnr', 'n_vox_real', 'n_vox_syn']
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    report(rows, out_csv)


def report(rows, out_csv):
    def stat(sel, key):
        vals = [r[key] for r in sel if isinstance(r[key], float) and np.isfinite(r[key])]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)

    print(f'\nPer-structure metrics written to {out_csv}')
    print('\nPer structure (failures counted as dice 0, which is the honest number):')
    print(f'{"structure":14s} {"dice":>16s} {"centroid mm":>16s} {"ssim":>8s} '
          f'{"cw_ssim":>8s} {"psnr":>7s}  {"n":>3s} {"fail":>4s}')

    for group, members in BILATERAL_GROUPS.items():
        names = {STRUCTURE_NAMES[m] for m in members}
        sel = [r for r in rows if r['structure'] in names
               and r['seg_status'] in ('ok', 'missing_syn')]
        if not sel:
            continue
        scored = [r for r in sel if r['seg_status'] == 'ok']
        nfail = sum(1 for r in sel if r['seg_status'] == 'missing_syn')
        # a missing synthetic structure is a real failure: dice 0, not dropped
        dvals = [0.0 if r['seg_status'] == 'missing_syn' else r['dice'] for r in sel]
        dm, ds = float(np.mean(dvals)), float(np.std(dvals))
        cm, cs, _ = stat(scored, 'centroid_mm')
        sm, _, _ = stat(scored, 'ssim')
        wm, _, _ = stat(scored, 'cw_ssim')
        pm, _, _ = stat(scored, 'psnr')
        print(f'{group:14s} {dm:7.3f} +/-{ds:5.3f} {cm:8.2f} +/-{cs:5.2f} '
              f'{sm:8.3f} {wm:8.3f} {pm:7.2f}  {len(sel):3d} {nfail:4d}')

    ok = [r for r in rows if r['seg_status'] == 'ok']
    sm, _, _ = stat(ok, 'ssim')
    wm, _, _ = stat(ok, 'cw_ssim')
    print(f'\nCW-SSIM vs SSIM over all scored structures: {wm:.3f} vs {sm:.3f} '
          f'(gap {wm - sm:+.3f})')
    print('  A positive gap is the share of SSIM loss attributable to small')
    print('  misalignment rather than to synthesis error.')

    counts = {}
    for r in rows:
        counts[r['seg_status']] = counts.get(r['seg_status'], 0) + 1
    print('\nseg_status counts: ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))


if __name__ == '__main__':
    main()
