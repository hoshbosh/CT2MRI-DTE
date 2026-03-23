#!/usr/bin/env python3
"""
Standalone inference script: generate a synthetic MRI from a single CT NIfTI file.

Usage:
    python infer_single.py \
        --input /path/to/ct.nii.gz \
        --output /path/to/synthetic_mr.nii \
        --config configs/fine-tune.yaml \
        --checkpoint /path/to/latest_model.pth \
        --gpu 0

    # If your model uses histogram conditioning and you have an MR reference:
    python infer_single.py \
        --input /path/to/ct.nii.gz \
        --output /path/to/synthetic_mr.nii \
        --config configs/fine-tune.yaml \
        --checkpoint /path/to/latest_model.pth \
        --hist_ref /path/to/reference_mr.nii \
        --gpu 0

    # Or use average histogram from a pickle:
    python infer_single.py \
        --input /path/to/ct.nii.gz \
        --output /path/to/synthetic_mr.nii \
        --config configs/fine-tune.yaml \
        --checkpoint /path/to/latest_model.pth \
        --hist_pkl /path/to/MR_hist_global_160_test_axial_avg.pkl \
        --gpu 0
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch
import yaml
from scipy.ndimage import affine_transform, zoom
from tqdm import tqdm


# ============================================================================
# SECTION 1: CONFIG LOADING
# ============================================================================
# The model needs a config object (an argparse.Namespace tree) that mirrors
# the YAML structure. This section loads the YAML and converts it into nested
# Namespace objects so the model code can access values like
# config.model.BB.params.num_timesteps.
# ============================================================================

def dict_to_namespace(d):
    """Recursively convert a dict to argparse.Namespace for attribute access."""
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = dict_to_namespace(v)
        return argparse.Namespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d


def load_config(config_path):
    """Load YAML config and convert to nested Namespace."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)


# ============================================================================
# SECTION 2: PREPROCESSING
# ============================================================================
# Before the model can process a CT volume, it needs to go through the same
# preprocessing pipeline used during training:
#
#   1. Reorient to RAS+ (Right-Anterior-Superior) so axes are consistent
#   2. Resample to 1mm isotropic voxels
#   3. Crop to brain bounding box (using intensity thresholding since we
#      don't have a mask for arbitrary input CTs)
#   4. Resize to the model's expected spatial dimensions (e.g., 160x160)
#   5. Apply CT brain window (center=40, width=80 HU) to normalize to [0,1]
#
# The spatial metadata (original affine, crop bounds, original shape) is
# saved so we can map the output back to the original space.
# ============================================================================

def resample_volume(data, affine, target_spacing=1.0, order=1):
    """
    Resample a volume to isotropic voxel spacing.

    The current voxel sizes are read from the affine matrix. A resampling
    matrix is computed as the ratio of current spacing to target spacing,
    then scipy's affine_transform warps the data to the new grid.

    Args:
        data: 3D numpy array (the volume)
        affine: 4x4 affine matrix mapping voxel indices to world coordinates
        target_spacing: desired voxel size in mm (default 1.0)
        order: interpolation order (1=trilinear for images, 0=nearest for masks)

    Returns:
        resampled_data: volume at new resolution
        new_affine: updated affine matrix
    """
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


def crop_to_nonzero(data, affine, padding=4):
    """
    Crop volume to the bounding box of non-zero voxels with padding.

    Since we don't have a brain mask for arbitrary input CTs, we use
    non-zero voxels as a proxy. After CT windowing, background is 0.

    Returns:
        cropped_data, new_affine, (mins, maxs) for later uncropping
    """
    coords = np.argwhere(data > 0)
    if len(coords) == 0:
        return data, affine, (np.array([0, 0, 0]), np.array(data.shape))

    mins = np.maximum(coords.min(axis=0) - padding, 0)
    maxs = np.minimum(coords.max(axis=0) + 1 + padding, np.array(data.shape[:3]))

    slices = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    cropped = data[slices]

    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, :3] @ mins + affine[:3, 3]

    return cropped, new_affine, (mins, maxs)


def resize_volume(data, height, width, order=1):
    """Resize the first two spatial dimensions to (height, width)."""
    zoom_factors = (height / data.shape[0], width / data.shape[1], 1.0)
    return zoom(data, zoom_factors, order=order)


def ct_brain_window(data, window_center=40, window_width=80):
    """
    Apply a CT brain soft-tissue window and normalize to [0, 1].

    CT Hounsfield units for brain soft tissue are roughly 20-80 HU.
    Bone is ~1000 HU. If we just do percentile-based normalization,
    bone dominates and brain tissue gets compressed to near-zero (dark).

    A standard brain window (center=40, width=80) clips to [0, 80] HU,
    making brain tissue fill the full [0, 1] range.

    Args:
        data: CT volume in Hounsfield units
        window_center: center of the window in HU (default 40)
        window_width: width of the window in HU (default 80)

    Returns:
        Normalized volume in [0, 1]
    """
    low = window_center - window_width / 2
    high = window_center + window_width / 2
    data = np.clip(data, low, high)
    data = (data - low) / (high - low)
    return data


def preprocess_ct(ct_path, image_size, target_spacing=1.0, padding=4):
    """
    Full preprocessing pipeline for a single CT NIfTI file.

    Steps:
        1. Load the NIfTI file
        2. Reorient to RAS+ so the axes are (Right, Anterior, Superior)
           This ensures consistency regardless of how the scanner saved the data
        3. Apply CT brain window BEFORE cropping (so thresholding works on
           normalized values)
        4. Resample to isotropic voxels
        5. Crop to brain bounding box
        6. Resize to model's expected dimensions

    Returns:
        processed_volume: [H, W, D] numpy array, values in [0, 1]
        spatial_metadata: dict with everything needed to map output back
                          to original space (affine, shapes, crop bounds, etc.)
    """
    # Step 1: Load
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata().astype(np.float64)
    affine = ct_img.affine.copy()
    original_shape = ct_data.shape

    # Step 2: Reorient to RAS+
    # nib.io_orientation reads the current orientation from the affine.
    # We compute a transform to go from current orientation to RAS+,
    # then apply it to both the data and the affine.
    orig_ornt = nib.io_orientation(affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    ct_data = nib.orientations.apply_orientation(ct_data, transform)
    affine = affine @ nib.orientations.inv_ornt_aff(transform, original_shape)

    # Step 3: Apply CT brain window to normalize to [0, 1]
    ct_data = ct_brain_window(ct_data)

    # Step 4: Resample to isotropic
    ct_data, affine = resample_volume(ct_data, affine, target_spacing, order=1)

    # Step 5: Crop to brain (non-zero) region
    ct_data, crop_affine, (crop_mins, crop_maxs) = crop_to_nonzero(ct_data, affine, padding)
    pre_resize_shape = ct_data.shape

    # Step 6: Resize to model dimensions
    ct_data = resize_volume(ct_data, image_size, image_size)

    spatial_metadata = {
        'original_img': ct_img,           # for header/affine when saving output
        'crop_affine': crop_affine,        # affine after cropping
        'pre_resize_shape': pre_resize_shape,  # shape before resize (for inverse)
    }

    return ct_data, spatial_metadata


# ============================================================================
# SECTION 3: SLICE PREPARATION
# ============================================================================
# The model doesn't process the whole 3D volume at once. It works on 2D
# slices, but with multi-channel context: for each target slice, it also
# takes neighboring slices as input (e.g., slice-1, slice, slice+1 for
# channels=3).
#
# transpose_LPS_to_ITKSNAP_position: The training data was stored in a
# specific axis ordering after this transform. We must apply the same
# transform so the slicing axis matches what the model learned.
#
# The slices are then normalized to [-1, 1] (if to_normal=True in config)
# to match training normalization.
# ============================================================================

def transpose_LPS_to_ITKSNAP_position(data, plane):
    """
    Transform volume axes to match the orientation used during HDF5 creation.

    During training data preparation, volumes are transposed so that the
    slicing dimension (axis 2) corresponds to the chosen plane:
      - axial: slicing through Superior-Inferior axis
      - coronal: slicing through Anterior-Posterior axis
      - sagittal: slicing through Left-Right axis

    The model learned on data in this specific axis arrangement, so inference
    input must match exactly.
    """
    if plane == 'axial':
        data = np.transpose(data, [1, 0, 2])
    elif plane == 'coronal':
        data = np.flip(data, axis=2)
        data = np.transpose(data, [2, 0, 1])
    elif plane == 'sagittal':
        data = np.rot90(data, axes=(1, 2))
        data = np.transpose(data, [1, 2, 0])
    return data


def inverse_transpose_LPS_to_ITKSNAP_position(data, plane):
    """
    Undo the axis transform applied before slicing, so the output volume
    is back in the original RAS+ orientation for saving as NIfTI.
    """
    if plane == 'axial':
        data = np.transpose(data, [1, 0, 2])
    elif plane == 'coronal':
        data = np.transpose(data, [1, 2, 0])
        data = np.flip(data, axis=2)
    elif plane == 'sagittal':
        data = np.transpose(data, [2, 0, 1])
        data = np.rot90(data, k=-1, axes=(1, 2))
    return data


def prepare_slices(volume, channels):
    """
    Convert a 3D volume [H, W, D] into batched multi-channel slices for
    the model.

    For each slice index i, we extract a window of neighboring slices:
        [i - radius, ..., i, ..., i + radius]
    where radius = channels // 2.

    Boundary slices are zero-padded (same as training).

    Returns:
        torch.Tensor of shape [num_slices, channels, H, W]
    """
    H, W, D = volume.shape
    radius = channels // 2
    slices = []

    for i in range(D):
        # Extract multi-channel context window
        if i < radius:
            # Near the start: need zero-padding on the left
            chunk = volume[:, :, 0:i + radius + 1]
            pad_width = radius - i
            chunk = np.pad(chunk, ((0, 0), (0, 0), (pad_width, 0)), mode='constant')
        elif i > D - 1 - radius:
            # Near the end: need zero-padding on the right
            chunk = volume[:, :, i - radius:D]
            pad_width = radius - (D - 1 - i)
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        else:
            chunk = volume[:, :, i - radius:i + radius + 1]

        # chunk shape: [H, W, channels] -> transpose to [channels, H, W]
        slices.append(chunk.transpose(2, 0, 1))

    return np.stack(slices, axis=0)  # [D, channels, H, W]


# ============================================================================
# SECTION 4: HISTOGRAM CONTEXT
# ============================================================================
# If the model was trained with histogram conditioning
# (condition_key="hist_context_y_concat"), it expects a histogram feature
# vector as additional input at each diffusion step. This histogram
# describes the target MR intensity distribution and guides the model
# toward producing output with realistic MR contrast.
#
# Three options for providing this at inference:
#   1. --hist_ref: a reference MR NIfTI — compute its histogram directly
#   2. --hist_pkl: a pre-computed pickle file (e.g., training set average)
#   3. Neither: generate a flat/uniform histogram as fallback
#
# The histogram is a [128, 3, 1] array with 3 channels:
#   Channel 0: normalized histogram (128 bins, range [0.001, 1])
#   Channel 1: cumulative histogram
#   Channel 2: histogram gradient (np.diff)
# ============================================================================

def compute_histogram_from_nifti(nifti_path, plane='axial'):
    """
    Compute the 3-channel histogram feature from a reference MR NIfTI.

    This replicates the exact computation from generate_total_hist_global.py:
      1. Load and reorient the MR volume
      2. Compute 128-bin histogram on intensity range [0.001, 1]
      3. Normalize and scale by 10
      4. Compute cumulative sum
      5. Compute gradient (diff)
      6. Stack into [128, 3, 1] array
    """
    num_bins = 128
    scale = 10

    mr_img = nib.load(nifti_path)
    mr_data = np.asanyarray(mr_img.dataobj)
    mr_data = np.nan_to_num(mr_data)
    mr_data = transpose_LPS_to_ITKSNAP_position(mr_data, plane)

    histograms, _ = np.histogram(mr_data, bins=num_bins, range=(0.001, 1))
    normalized = histograms / histograms.sum(keepdims=True)
    normalized *= scale

    cum_hist = np.cumsum(normalized)

    hist_diff = np.diff(normalized)
    hist_diff = np.insert(hist_diff, 0, hist_diff[0])
    hist_diff *= scale

    combined = np.stack((normalized, cum_hist, hist_diff), axis=1)
    combined = np.expand_dims(combined, axis=-1)  # [128, 3, 1]
    return combined


def load_histogram_from_pkl(pkl_path):
    """
    Load a pre-computed histogram from a pickle file.

    The pickle contains a dict: {subject_name: histogram_array}.
    We just grab the first entry since for avg/colin pickles all entries
    are identical.
    """
    import pickle
    with open(pkl_path, 'rb') as f:
        hist_dict = pickle.load(f)
    # Return the first histogram in the dict
    first_key = next(iter(hist_dict))
    return hist_dict[first_key]


def make_uniform_histogram():
    """
    Create a uniform histogram as fallback when no reference is available.

    This is a reasonable default — it tells the model "I don't have a
    specific target distribution preference." The model still produces
    valid output; it just won't be guided toward a particular MR contrast.
    """
    num_bins = 128
    scale = 10
    uniform = np.ones(num_bins) / num_bins * scale
    cum = np.cumsum(uniform)
    diff = np.zeros(num_bins)
    combined = np.stack((uniform, cum, diff), axis=1)
    return np.expand_dims(combined, axis=-1)  # [128, 3, 1]


# ============================================================================
# SECTION 5: MODEL LOADING
# ============================================================================
# The model is a BrownianBridgeModel containing a UNet for denoising.
# We load the checkpoint (which contains the model state_dict, EMA weights,
# epoch/step info) and optionally apply EMA weights for better quality.
#
# EMA (Exponential Moving Average) maintains a running average of model
# weights during training. At inference time, EMA weights typically produce
# smoother, higher-quality results than the raw training weights.
# ============================================================================

def load_model(config, checkpoint_path, device):
    """
    Instantiate the BrownianBridgeModel and load checkpoint weights.

    Steps:
        1. Create model from config (this builds the UNet architecture)
        2. Load checkpoint file
        3. Load state_dict with strict=False (handles architecture mismatches
           from pretraining vs fine-tuning)
        4. If EMA weights exist, swap them in (better inference quality)
        5. Move to device and set to eval mode
    """
    from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel

    # Create model
    net = BrownianBridgeModel(config.model)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    states = torch.load(checkpoint_path, map_location='cpu')

    # Filter out shape-mismatched keys (same logic as BaseRunner)
    current_state = net.state_dict()
    filtered_state = {}
    for k, v in states['model'].items():
        if k in current_state and current_state[k].shape != v.shape:
            print(f"  Skipping shape-mismatched key: {k}")
        else:
            filtered_state[k] = v
    missing, unexpected = net.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # Use EMA weights if available (smoother, higher quality output)
    if 'ema' in states and states['ema']:
        print("  Applying EMA weights")
        ema_shadow = states['ema']
        for name, param in net.named_parameters():
            if name in ema_shadow and ema_shadow[name].shape == param.shape:
                param.data.copy_(ema_shadow[name])

    net = net.to(device)
    net.eval()

    print(f"  Model loaded (epoch {states.get('epoch', '?')}, step {states.get('step', '?')})")
    return net


# ============================================================================
# SECTION 6: DIFFUSION SAMPLING
# ============================================================================
# The Brownian Bridge Diffusion Model generates MRI by starting from the
# CT input (the "condition" y) and iteratively denoising toward the target
# distribution (MRI).
#
# Unlike standard diffusion models that start from pure noise, BBDM starts
# from the condition image y and walks a "bridge" toward x (the target MRI).
# At each timestep, the UNet predicts the gradient/noise to remove.
#
# The sampling uses DDIM-style skip steps for efficiency:
# instead of 1000 steps, we typically use 50-100 steps by skipping.
#
# For memory efficiency with large batches, we process in sub-batches.
# ============================================================================

def run_diffusion(net, ct_slices, hist_context, config, device, batch_size=48):
    """
    Run the diffusion sampling process on all CT slices.

    Args:
        net: loaded BrownianBridgeModel
        ct_slices: [num_slices, channels, H, W] tensor of preprocessed CT
        hist_context: [128, 3, 1] histogram array (or None if no conditioning)
        config: model config namespace
        device: torch device
        batch_size: max slices per forward pass (for GPU memory)

    Returns:
        output_slices: [num_slices, channels, H, W] tensor of synthetic MRI

    The key steps:
        1. Normalize CT to [-1, 1] (matching training normalization)
        2. Initialize diffusion chain at y (the CT condition)
        3. Loop through timesteps, predicting and removing noise at each step
        4. Denormalize output back to [0, 1]
    """
    num_slices = ct_slices.shape[0]
    channels = config.data.dataset_config.channels
    mid_slice = channels // 2
    condition_key = config.model.BB.params.UNetParams.condition_key

    # Collect all output mid-slices
    all_outputs = []

    # Process in batches to fit in GPU memory
    for start in range(0, num_slices, batch_size):
        end = min(start + batch_size, num_slices)
        batch = ct_slices[start:end]

        # The model expects x (target, MRI) and y (condition, CT).
        # At inference, we don't have x, so we pass y as both x and y.
        # The sampling loop starts from y and walks toward the target.
        y = batch.to(device)
        x = y.clone()  # placeholder — not used in sampling, only y matters

        # Prepare context based on conditioning type
        context = None
        if condition_key == 'hist_context_y_concat' and hist_context is not None:
            # Repeat histogram for each slice in the batch
            hist_tensor = torch.from_numpy(hist_context).float().to(device)
            hist_batch = hist_tensor.unsqueeze(0).expand(y.shape[0], -1, -1, -1)
            context = hist_batch

        # Run the diffusion sampling loop
        # net.sample internally calls p_sample_loop which:
        #   - Starts from img = y (the CT)
        #   - At each timestep t, predicts the denoised x0
        #   - Computes x_{t-1} using the DDIM update rule
        #   - Returns the final denoised result
        sample = net.sample(
            x, y,
            context=context,
            clip_denoised=False,
            path=None,
            save=False,
            config=config,
            device=device,
        )

        # Extract the middle slice from the multi-channel output
        # and denormalize from [-1, 1] back to [0, 1]
        sample = sample[:, mid_slice].detach().cpu().mul_(0.5).add_(0.5).clamp_(0, 1.)
        all_outputs.append(sample)

    return torch.cat(all_outputs, dim=0)  # [num_slices, H, W]


# ============================================================================
# SECTION 7: POSTPROCESSING & SAVING
# ============================================================================
# After the model produces 2D slices, we need to:
#   1. Stack them back into a 3D volume
#   2. Undo the axis transpose (inverse of transpose_LPS_to_ITKSNAP_position)
#   3. Resize back to the pre-resize shape (undo the model's fixed HxW resize)
#   4. Save as NIfTI using the cropped affine from preprocessing
# ============================================================================

def postprocess_and_save(output_slices, spatial_metadata, output_path, plane, image_size):
    """
    Reassemble 2D slices into a 3D NIfTI volume and save.

    Args:
        output_slices: [num_slices, H, W] tensor from the model
        spatial_metadata: dict from preprocess_ct with spatial info
        output_path: where to save the output NIfTI file
        plane: 'axial', 'coronal', or 'sagittal'
        image_size: model's H/W dimension
    """
    # Stack slices into volume: [H, W, D]
    volume = output_slices.numpy()
    volume = np.transpose(volume, (1, 2, 0))  # [H, W, num_slices]

    # Undo the ITKSNAP axis transform to get back to RAS+ orientation
    volume = inverse_transpose_LPS_to_ITKSNAP_position(volume, plane)

    # Resize back to pre-resize spatial dimensions
    pre_shape = spatial_metadata['pre_resize_shape']
    if volume.shape != pre_shape:
        zoom_factors = (
            pre_shape[0] / volume.shape[0],
            pre_shape[1] / volume.shape[1],
            pre_shape[2] / volume.shape[2],
        )
        volume = zoom(volume, zoom_factors, order=1)

    # Save as NIfTI with the crop affine
    out_img = nib.Nifti1Image(volume.astype(np.float32), spatial_metadata['crop_affine'])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nib.save(out_img, output_path)
    print(f"Saved synthetic MRI to: {output_path}")
    print(f"  Volume shape: {volume.shape}")
    print(f"  Intensity range: [{volume.min():.3f}, {volume.max():.3f}]")


# ============================================================================
# SECTION 8: MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic MRI from a single CT NIfTI file"
    )
    parser.add_argument("--input", required=True,
                        help="Path to input CT NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--output", required=True,
                        help="Path to save output synthetic MRI NIfTI")
    parser.add_argument("--config", required=True,
                        help="Path to model config YAML (e.g., configs/fine-tune.yaml)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint .pth file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (default: 0). Use -1 for CPU.")
    parser.add_argument("--sample_step", type=int, default=None,
                        help="Number of diffusion steps (default: from config)")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="Max slices per forward pass (default: 48)")
    parser.add_argument("--hist_ref", default=None,
                        help="Path to a reference MR NIfTI for histogram conditioning")
    parser.add_argument("--hist_pkl", default=None,
                        help="Path to a pre-computed histogram pickle file")
    args = parser.parse_args()

    # --- Load config ---
    config = load_config(args.config)
    image_size = config.data.dataset_config.image_size
    channels = config.data.dataset_config.channels
    plane = config.data.dataset_config.plane
    condition_key = config.model.BB.params.UNetParams.condition_key
    to_normal = config.data.dataset_config.to_normal

    # Override sample_step if specified
    if args.sample_step is not None:
        config.model.BB.params.sample_step = args.sample_step

    # We need config.args.train = False so the model uses sub-batch sampling
    if not hasattr(config, 'args'):
        config.args = argparse.Namespace()
    config.args.train = False

    # --- Device setup ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Preprocess CT ---
    print(f"\nPreprocessing: {args.input}")
    print(f"  Target size: {image_size}x{image_size}, plane: {plane}")
    ct_volume, spatial_metadata = preprocess_ct(args.input, image_size)
    print(f"  Preprocessed shape: {ct_volume.shape}")

    # Apply the same axis transform used during training data creation
    ct_volume = transpose_LPS_to_ITKSNAP_position(ct_volume, plane)
    print(f"  After {plane} transform: {ct_volume.shape}")

    # --- Prepare slices ---
    # Convert volume to batched multi-channel slices
    ct_slices = prepare_slices(ct_volume, channels)  # [D, C, H, W]
    ct_slices = torch.from_numpy(ct_slices).float()

    # Normalize to [-1, 1] if the model was trained that way
    if to_normal:
        ct_slices = (ct_slices - 0.5) * 2.0
        ct_slices.clamp_(-1., 1.)

    print(f"  Prepared {ct_slices.shape[0]} slices, shape per slice: {ct_slices.shape[1:]}")

    # --- Prepare histogram context ---
    hist_context = None
    if condition_key == 'hist_context_y_concat':
        if args.hist_ref:
            print(f"\nComputing histogram from reference MR: {args.hist_ref}")
            hist_context = compute_histogram_from_nifti(args.hist_ref, plane)
        elif args.hist_pkl:
            print(f"\nLoading histogram from pickle: {args.hist_pkl}")
            hist_context = load_histogram_from_pkl(args.hist_pkl)
        else:
            print("\nWARNING: Model uses histogram conditioning but no --hist_ref or --hist_pkl provided.")
            print("  Using uniform histogram as fallback. Results may be suboptimal.")
            print("  Consider providing --hist_ref (a reference MR) or --hist_pkl (avg histogram from training).")
            hist_context = make_uniform_histogram()
        print(f"  Histogram shape: {hist_context.shape}")

    # --- Load model ---
    print(f"\nLoading model...")
    net = load_model(config, args.checkpoint, device)

    # --- Run diffusion ---
    print(f"\nRunning diffusion sampling ({config.model.BB.params.sample_step} steps)...")
    with torch.no_grad():
        output_slices = run_diffusion(
            net, ct_slices, hist_context, config, device,
            batch_size=args.batch_size
        )
    print(f"  Generated {output_slices.shape[0]} slices")

    # --- Postprocess and save ---
    print(f"\nPostprocessing and saving...")
    postprocess_and_save(output_slices, spatial_metadata, args.output, plane, image_size)
    print("\nDone!")


if __name__ == "__main__":
    main()
