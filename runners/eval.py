import os
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import binary_fill_holes, binary_erosion

def cal_mae(syn_img, raw_img):
    mae = np.abs(syn_img - raw_img).mean()
    return mae

def cal_mse(syn_img, raw_img):
    mse = ((syn_img - raw_img) ** 2).mean()
    return mse

def cal_rmse(syn_img, raw_img):
    rmse = np.sqrt(cal_mse(syn_img, raw_img))
    return rmse

def cal_nrmse(syn_img, raw_img):
    mse = cal_mse(syn_img, raw_img)
    data_range = np.max(syn_img) - np.min(syn_img)
    nrmse = np.sqrt(mse) / data_range
    return nrmse

def cal_psnr(syn_img, raw_img):
    psnr = peak_signal_noise_ratio(syn_img, raw_img, data_range=1.0)
    return psnr

def cal_ssim(syn_img, raw_img):
    ssim_index = structural_similarity(syn_img, raw_img,
                                       data_range=1.0)
    return ssim_index


def foreground_mask(raw_img, thr=0.05, erode_iter=0):
    """Brain mask from GT volume: threshold + per-slice hole-fill + optional erosion.

    Hole-filling closes CSF gaps (ventricles, sulci) that a bare threshold marks
    as background, making ssim_mask measure actual brain-interior fidelity.
    erode_iter > 0 strips skull fringe (use if data is NOT already skull-stripped).

    raw_img: (H, W, S) numpy array in [0, 1]
    """
    coarse = raw_img > thr
    filled = np.zeros_like(coarse, dtype=bool)
    for s in range(coarse.shape[2]):
        filled[:, :, s] = binary_fill_holes(coarse[:, :, s])
    if erode_iter > 0:
        struct = np.ones((3, 3, 1), dtype=bool)  # 2D per-slice erosion
        filled = binary_erosion(filled, structure=struct, iterations=erode_iter)
    return filled


def cal_masked_psnr(syn_img, raw_img, mask):
    """PSNR computed only over masked voxels (data_range=1.0)."""
    if mask.sum() == 0:
        return np.nan
    mse = ((syn_img[mask] - raw_img[mask]) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


# LPIPS is perceptual and sharpness-sensitive but pulls in an extra dependency
# and downloads its backbone weights on first use; lazy-load so a missing
# package (or offline compute node) degrades to NaN instead of killing eval.
_LPIPS_MODEL = None
_LPIPS_DISABLED = False

def _get_lpips_model(device):
    global _LPIPS_MODEL, _LPIPS_DISABLED
    if _LPIPS_DISABLED:
        return None
    if _LPIPS_MODEL is None:
        try:
            import lpips as lpips_lib
            _LPIPS_MODEL = lpips_lib.LPIPS(net='alex').to(device).eval()
        except Exception as e:
            print(f"[eval] LPIPS unavailable ({e}); reporting NaN. "
                  f"Run `pip install lpips` in the env (instantiate once on a "
                  f"login node so the backbone weights cache before compute).")
            _LPIPS_DISABLED = True
            return None
    return _LPIPS_MODEL


def cal_lpips(syn_img, raw_img, device='cpu'):
    """Mean LPIPS over the slices of a [H, W, S] volume in [0,1].
    Empty (all-background) slices are skipped. Returns NaN if lpips is absent."""
    model = _get_lpips_model(device)
    if model is None:
        return np.nan
    import torch
    H, W, S = syn_img.shape
    vals = []
    with torch.no_grad():
        for s in range(S):
            syn_s = syn_img[:, :, s]
            raw_s = raw_img[:, :, s]
            if raw_s.max() < 1e-3 and syn_s.max() < 1e-3:
                continue
            def to_t(a):
                t = torch.from_numpy(np.ascontiguousarray(a)).float().mul(2.0).sub(1.0)
                return t[None, None].repeat(1, 3, 1, 1).to(device)
            vals.append(model(to_t(syn_s), to_t(raw_s)).item())
    if not vals:
        return np.nan
    return float(np.mean(vals))


def calcul_metrics(metrics_dict, pa_id, syn_img, raw_img, mask=None, device='cpu'):
    metrics_dict[pa_id]['nrmse'] = cal_nrmse(syn_img, raw_img)
    metrics_dict[pa_id]['psnr'] = cal_psnr(syn_img, raw_img)

    # Whole-volume SSIM plus SSIM restricted to the brain/ROI mask. The masked
    # score strips the easy background that dominates whole-volume SSIM, so it
    # tracks fidelity where it matters for DBS targeting.
    ssim_global, ssim_map = structural_similarity(syn_img, raw_img, data_range=1.0, full=True)
    metrics_dict[pa_id]['ssim'] = ssim_global

    if mask is None:
        mask = foreground_mask(raw_img)
    metrics_dict[pa_id]['ssim_mask'] = float(ssim_map[mask].mean()) if mask.any() else np.nan
    metrics_dict[pa_id]['psnr_mask'] = cal_masked_psnr(syn_img, raw_img, mask)
    metrics_dict[pa_id]['lpips'] = cal_lpips(syn_img, raw_img, device=device)

    print(f"{pa_id} : ")
    print(metrics_dict[pa_id])

def add_result(results_file, result_data):
    try:
        existing_data = pd.read_csv(results_file)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['name', 'date', 'size', 'baseline', 'plane', 'sampling', 'exp_type', 'nrmse', 'psnr', 'ssim'])

    result_data_columns = ['name', 'date', 'size', 'baseline', 'plane', 'sampling', 'exp_type', 'nrmse', 'psnr', 'ssim']
    new_row = pd.DataFrame([result_data], columns=result_data_columns)
    
    updated_data = pd.concat([existing_data, new_row], ignore_index=True)
    updated_data.to_csv(results_file, index=False)

def save_exp_result(results_file, config, means):
    name, checkpoint, inference_type = config.model.model_name, config.model.model_load_path, config.model.BB.params.inference_type
    checkpoint = checkpoint.split('/')[-1]

    # Columns follow whatever metrics `means` carries, so adding/removing a
    # metric in calcul_metrics doesn't desync this row.
    metric_names = list(means.index)
    result_data_columns = ['name', 'checkpoint', 'inference_type'] + metric_names
    result_data = [name, checkpoint, inference_type] + [means[m] for m in metric_names]

    try:
        existing_data = pd.read_csv(results_file)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=result_data_columns)

    new_row = pd.DataFrame([result_data], columns=result_data_columns)
    updated_data = pd.concat([existing_data, new_row], ignore_index=True)
    updated_data.to_csv(results_file, index=False)
    
    