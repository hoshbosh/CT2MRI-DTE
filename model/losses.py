import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss on the mid-channel of (B, C, H, W) inputs in [-1, 1].

    Returns 1 - SSIM so it can be added directly to a training objective.
    Uses a Gaussian window matched to skimage's structural_similarity defaults
    so the training-time signal aligns with the eval-time metric.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        # Build separable Gaussian kernel once and register as a buffer.
        coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g[:, None] * g[None, :]
        self.register_buffer('kernel', kernel_2d[None, None])  # (1,1,W,W)
        self.window_size = window_size

    def _prepare(self, x):
        # Extract mid-channel and rescale [-1,1] -> [0,1] to match SSIM convention.
        mid = x.shape[1] // 2
        x = x[:, mid:mid + 1]
        return x * 0.5 + 0.5

    def forward(self, pred, target):
        pred = self._prepare(pred)
        target = self._prepare(target)

        pad = self.window_size // 2
        mu1 = F.conv2d(pred, self.kernel, padding=pad)
        mu2 = F.conv2d(target, self.kernel, padding=pad)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.kernel, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.kernel, padding=pad) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.kernel, padding=pad) - mu1_mu2

        # Constants for data range = 1.0
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """VGG-16 perceptual loss. Extracts features at relu1_2, relu2_2, relu3_3, relu4_3
    and computes L1 distance between predicted and target features.

    Input images are expected in [-1, 1] range (single or multi-channel).
    The mid-channel slice is extracted, repeated to 3 channels, and normalized
    to ImageNet stats before feeding into VGG.
    """

    # VGG layer indices for relu1_2, relu2_2, relu3_3, relu4_3
    LAYER_INDICES = [4, 9, 16, 23]

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Only keep layers up to relu4_3 (index 23)
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self.LAYER_INDICES:
            self.slices.append(nn.Sequential(*list(vgg.features.children())[prev:idx + 1]))
            prev = idx + 1

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Register normalization buffers
        self.register_buffer('mean', torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _prepare(self, x):
        """Convert model output (B, C, H, W) in [-1, 1] to VGG input (B, 3, H, W) ImageNet-normalized."""
        # Extract mid-channel slice -> (B, 1, H, W)
        mid = x.shape[1] // 2
        x = x[:, mid:mid + 1]
        # [-1, 1] -> [0, 1]
        x = x * 0.5 + 0.5
        # Repeat to 3 channels
        x = x.expand(-1, 3, -1, -1)
        # ImageNet normalize
        x = (x - self.mean) / self.std
        return x

    def _extract_features(self, x):
        feats = []
        h = x
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return feats

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) predicted image in [-1, 1]
            target: (B, C, H, W) ground truth image in [-1, 1]
        Returns:
            Scalar perceptual loss (mean L1 across layers)
        """
        pred_prep = self._prepare(pred)
        target_prep = self._prepare(target)

        pred_feats = self._extract_features(pred_prep)
        target_feats = self._extract_features(target_prep)

        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pf, tf)
        return loss / len(pred_feats)


class FrequencyLoss(nn.Module):
    """Focal Frequency Loss (Jiang et al., ICCV 2021) on the mid-channel.

    L1/perceptual losses are mean-seeking and systematically under-weight high
    spatial frequencies, which is the direct cause of diffusion-output blur.
    This compares the 2D Fourier spectra of pred vs target and focuses on the
    frequencies the model is currently getting most wrong (hard-frequency
    weighting), so it explicitly pushes back on missing fine detail.

    Inputs are (B, C, H, W) in [-1, 1]; only the mid channel is used to match
    the other auxiliary losses.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def _mid(self, x):
        mid = x.shape[1] // 2
        return x[:, mid:mid + 1]

    def forward(self, pred, target):
        pred = self._mid(pred)
        target = self._mid(target)

        # Orthonormal FFT so the loss scale is resolution-independent.
        fp = torch.fft.fft2(pred, norm='ortho')
        ft = torch.fft.fft2(target, norm='ortho')

        # Squared distance between complex spectra, per frequency bin.
        diff_real = fp.real - ft.real
        diff_imag = fp.imag - ft.imag
        dist = diff_real ** 2 + diff_imag ** 2  # (B, 1, H, W)

        # Focal weight: emphasize high-error frequencies. Detached so the weight
        # is a static spectral mask, not a second gradient path. Normalized per
        # image to [0, 1] for scale stability.
        weight = dist.detach() ** self.alpha
        weight = weight / (weight.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        return (weight * dist).mean()
