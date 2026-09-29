"""Phase 3 guards: label/image flip parity and the deep-gray weight map.

Runs offline on synthetic data. The failures these catch are all silent at
runtime -- a label that flips the wrong way, or a weight map that is quietly
all-ones, produces a training run that completes normally and reports a
negative result for the wrong reason.
"""
import os
import sys
import tempfile

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets.base import multi_ch_nifti_default_Dataset, multi_ch_nifti_label_Dataset
from model.BrownianBridge.BrownianBridgeModel import deepgray_mask_from_labels


def _fake(n_slices=7, H=16, W=16):
    """One subject, asymmetric content so a missed flip cannot hide."""
    rng = np.random.default_rng(0)
    img = rng.random((H, W, n_slices)).astype(np.float16)
    img[:, :W // 3, :] += 0.5           # bright left edge -> flip is visible
    lab = np.zeros((H, W, n_slices), dtype=np.uint8)
    lab[4:8, 2:6, :] = 5                # left putamen, off-centre
    lab[9:12, 11:14, :] = 8             # right pallidum
    index = np.stack([np.arange(n_slices), np.full(n_slices, n_slices - 1)], axis=1).astype(np.int32)
    subs = np.array([b'1BA001'] * n_slices)
    return img, lab, index, subs


def test_flip_parity():
    img, lab, index, subs = _fake()
    imgs = multi_ch_nifti_default_Dataset(img, index, subs, radius=1, flip=True, to_normal=True)
    labs = multi_ch_nifti_label_Dataset(lab, index, subs, flip=True)
    assert len(imgs) == len(labs) == 14

    n = index.shape[0]
    for i in range(len(labs)):
        im = imgs[i][0][1]                      # centre channel, (H, W)
        lb = labs[i][0][0]                      # (H, W)
        base = lab[:, :, i % n]
        expect = base[:, ::-1] if i >= n else base
        assert torch.equal(lb, torch.from_numpy(np.ascontiguousarray(expect)).long()), \
            f"label flip mismatch at index {i}"
        # and the image really did flip on the same indices
        flipped = im[:, :im.shape[1] // 3].mean() < im[:, -im.shape[1] // 3:].mean()
        assert flipped == (i >= n), f"image flip disagrees with label flip at {i}"
    print("  flip parity: image and label flip on the same indices")


def test_labels_are_not_normalised():
    _, lab, index, subs = _fake()
    labs = multi_ch_nifti_label_Dataset(lab, index, subs, flip=False)
    out = labs[0][0]
    assert out.dtype == torch.long, f"labels must stay integer, got {out.dtype}"
    assert set(out.unique().tolist()) == {0, 5, 8}, \
        f"label codes corrupted: {out.unique().tolist()}"
    print("  labels keep integer codes, no [-1,1] normalisation")


def test_uniform_mask():
    lab = torch.zeros(2, 1, 16, 16, dtype=torch.long)
    lab[:, :, 4:8, 4:8] = 7
    m = deepgray_mask_from_labels(lab, 'uniform')
    assert m.sum().item() == 2 * 16, m.sum()
    assert m[0, 0, 4, 4] == 1 and m[0, 0, 0, 0] == 0
    print("  uniform mask covers exactly the labelled voxels")


def test_shell_mask_straddles_boundary():
    lab = torch.zeros(1, 1, 32, 32, dtype=torch.long)
    lab[:, :, 12:20, 12:20] = 7            # 8x8 square
    m = deepgray_mask_from_labels(lab, 'shell', dilate=2, erode=1)

    # Exact analytic expectation: dilate(2) grows the square to 12x12,
    # erode(1) shrinks it to 6x6, so the shell is 12^2 - 6^2 = 108 px.
    assert m.sum().item() == 12 * 12 - 6 * 6, f"shell area {m.sum().item()}, expected 108"
    assert m[0, 0, 16, 16] == 0, "shell must be hollow at the structure centre"
    assert m[0, 0, 12, 16] == 1, "shell must include the boundary row"
    assert m[0, 0, 11, 16] == 1, "shell must reach outward into neighbouring tissue"
    assert m[0, 0, 9, 16] == 0, "shell must not reach beyond the dilate radius"
    print("  shell mask straddles the boundary with exact analytic area")


def test_empty_labels_give_empty_mask():
    lab = torch.zeros(1, 1, 16, 16, dtype=torch.long)
    for mode in ('uniform', 'shell'):
        assert deepgray_mask_from_labels(lab, mode).sum().item() == 0
    print("  a slice with no structures contributes no extra weight")


def test_bad_input_fails_loudly():
    for bad, mode in [
        (torch.zeros(1, 3, 8, 8, dtype=torch.long), 'uniform'),   # multi-channel
        (torch.zeros(1, 1, 8, 8, dtype=torch.long), 'nonsense'),  # bad mode
    ]:
        try:
            deepgray_mask_from_labels(bad, mode)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for shape={tuple(bad.shape)} mode={mode}")
    print("  malformed input raises instead of silently producing zeros")


def test_weight_composition():
    """1.0 background / 3.0 in-brain / 15.0 deep gray, as documented."""
    lab = torch.zeros(1, 1, 8, 8, dtype=torch.long)
    lab[:, :, 2:4, 2:4] = 5
    brain = torch.zeros(1, 1, 8, 8)
    brain[:, :, 1:6, 1:6] = 1.0

    loss_weight = 1.0 + (3.0 - 1.0) * brain
    dg = deepgray_mask_from_labels(lab, 'uniform')
    loss_weight = loss_weight * (1.0 + (5.0 - 1.0) * dg)

    assert loss_weight[0, 0, 0, 0].item() == 1.0, "background must stay at 1.0"
    assert loss_weight[0, 0, 5, 5].item() == 3.0, "in-brain non-structure must stay at 3.0"
    assert loss_weight[0, 0, 2, 2].item() == 15.0, "deep gray must be 3 x 5"
    print("  weight composition: 1.0 / 3.0 / 15.0")


def test_dataset_returns_four_tuple(tmp=None):
    """End-to-end: the registered dataset yields (ori, cond, hist, label).

    This is the integration the BBDMRunner.py:187 unpacking bug would have
    broken silently -- a 4th element used to be swallowed into a list and
    discarded, so training ran to completion with no weighting and no error.
    """
    import pickle
    import shutil
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from datasets.custom import hist_context_label_CT2MR_Paired_Dataset

    d = tempfile.mkdtemp()
    try:
        H = W = 32
        img, lab, index, subs = _fake(n_slices=12, H=H, W=W)
        ct = np.asarray(img[::-1, :, :], dtype=np.float16)
        with h5py.File(os.path.join(d, f'{H}_train_axial.hdf5'), 'w') as hf:
            hf.create_dataset('MR_dataset', data=img)
            hf.create_dataset('CT_dataset', data=ct)
            hf.create_dataset('LABEL_dataset', data=lab)
            hf.create_dataset('index_dataset', data=index)
            hf.create_dataset('subject', data=subs)
        with open(os.path.join(d, f'MR_hist_global_{H}_train_axial_.pkl'), 'wb') as f:
            pickle.dump({'1BA001': np.ones(128, dtype=np.float32)}, f)

        cfg = SimpleNamespace(dataset_path=d, image_size=H, plane='axial', channels=3,
                              to_normal=True, flip=True, hist_type=None)
        ds = hist_context_label_CT2MR_Paired_Dataset(cfg, stage='train')
        batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))

        assert len(batch) == 4, f"expected a 4-tuple, got {len(batch)}"
        (x, _), (x_cond, _), hist, labels = batch
        assert x.shape == (4, 3, H, W), x.shape
        assert labels.shape == (4, 1, H, W), labels.shape
        assert labels.dtype == torch.long, labels.dtype
        assert hist.shape == (4, 128), hist.shape
        assert set(labels.unique().tolist()) <= {0, 5, 8}
        assert labels.sum() > 0, "labels arrived empty"
        print("  dataset yields (ori, cond, hist, label) with labels intact")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_missing_labels_fail_loudly():
    """An HDF5 without LABEL_dataset must raise, not fall back to no weighting."""
    import pickle
    import shutil
    from types import SimpleNamespace
    from datasets.custom import hist_context_label_CT2MR_Paired_Dataset

    d = tempfile.mkdtemp()
    try:
        H = W = 32
        img, lab, index, subs = _fake(n_slices=12, H=H, W=W)
        with h5py.File(os.path.join(d, f'{H}_train_axial.hdf5'), 'w') as hf:
            hf.create_dataset('MR_dataset', data=img)
            hf.create_dataset('CT_dataset', data=img)
            hf.create_dataset('index_dataset', data=index)
            hf.create_dataset('subject', data=subs)
        with open(os.path.join(d, f'MR_hist_global_{H}_train_axial_.pkl'), 'wb') as f:
            pickle.dump({'1BA001': np.ones(128, dtype=np.float32)}, f)
        cfg = SimpleNamespace(dataset_path=d, image_size=H, plane='axial', channels=3,
                              to_normal=True, flip=True, hist_type=None)
        try:
            hist_context_label_CT2MR_Paired_Dataset(cfg, stage='train')
        except KeyError as e:
            assert 'LABEL_dataset' in str(e)
            print("  an HDF5 without labels raises instead of training unweighted")
            return
        raise AssertionError("expected KeyError for an HDF5 with no LABEL_dataset")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    for fn in [test_flip_parity, test_labels_are_not_normalised, test_uniform_mask,
               test_shell_mask_straddles_boundary, test_empty_labels_give_empty_mask,
               test_bad_input_fails_loudly, test_weight_composition,
               test_dataset_returns_four_tuple, test_missing_labels_fail_loudly]:
        fn()
    print("deep-gray loss guards pass")
