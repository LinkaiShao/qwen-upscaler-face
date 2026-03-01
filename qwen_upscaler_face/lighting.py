"""Differentiable light lock — frequency-separated luminance transfer."""

import math
import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(sigma, kernel_size):
    """Create 1D Gaussian kernel."""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_blur(x, radius):
    """Apply separable Gaussian blur to (B, 1, H, W) tensor."""
    if radius <= 0:
        return x
    sigma = radius / 2.0
    kernel_size = int(math.ceil(radius * 3)) * 2 + 1
    k1d = _gaussian_kernel_1d(sigma, kernel_size).to(x.device, x.dtype)
    pad = kernel_size // 2

    # Horizontal
    kh = k1d.view(1, 1, 1, -1)
    out = F.conv2d(x, kh, padding=(0, pad))
    # Vertical
    kv = k1d.view(1, 1, -1, 1)
    out = F.conv2d(out, kv, padding=(pad, 0))
    return out


# BT.601 luma coefficients
_Y_R, _Y_G, _Y_B = 0.299, 0.587, 0.114


def lock_lighting(pred, guide, y_blend=0.80, blur_radius=7.0):
    """Anchor low-frequency luminance of pred to guide, preserving pred's detail.

    Args:
        pred:  (B, 3, H, W) tensor in [-1, 1]
        guide: (B, 3, H, W) tensor in [-1, 1]  (LR upscaled to same resolution)
        y_blend:     blending strength (0 = keep pred lighting, 1 = fully follow guide)
        blur_radius: Gaussian blur radius for separating low/high frequency

    Returns:
        (B, 3, H, W) tensor with corrected lighting, same range as input
    """
    # Extract Y (luma) channel
    y_pred = _Y_R * pred[:, 0:1] + _Y_G * pred[:, 1:2] + _Y_B * pred[:, 2:3]  # (B,1,H,W)
    y_guide = _Y_R * guide[:, 0:1] + _Y_G * guide[:, 1:2] + _Y_B * guide[:, 2:3]

    # Separate low/high frequency
    y_pred_base = _gaussian_blur(y_pred, blur_radius)
    y_guide_base = _gaussian_blur(y_guide, blur_radius)
    y_pred_detail = y_pred - y_pred_base

    # Blend: keep pred detail, anchor base illumination to guide
    y_out = (1.0 - y_blend) * y_pred + y_blend * (y_guide_base + y_pred_detail)

    # Apply luminance ratio to RGB channels (preserves color)
    eps = 1e-6
    ratio = y_out / (y_pred + eps)
    out = pred * ratio
    return out.clamp(-1.0, 1.0)
