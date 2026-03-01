#!/usr/bin/env python3
"""Run PatchGAN discriminator on images and produce multi-panel heatmap figures.

Usage:
    python -m qwen_upscaler_face.patch_gan.infer_disc \
        --lr image_lr.png --target image_hr.png pred.png \
        --disc_ckpt /path/to/disc_step11500.pt \
        --output_dir /path/to/output
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/vton upscaling")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qwen_upscaler_face.patch_gan.discriminator import NLayerDiscriminator
from qwen_upscaler_face.patch_gan.dataset import load_clothing_mask_tensor

BASE = "/home/link/Desktop/Code/fashion gen testing"
STEP_DIR = f"{BASE}/qwen_upscale_face_out/qwen_face_20260224_001227/step-6500"
DEFAULT_DISC = f"{STEP_DIR}/patch_gan_disc_clothing/disc_final.pt"


def load_image(path):
    """Load PNG → (1, 3, H, W) tensor in [-1, 1]."""
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    img = img / 127.5 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def to_display(tensor_1chw):
    """(1, 3, H, W) tensor [-1,1] → (H, W, 3) uint8."""
    return ((tensor_1chw[0].cpu().float().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255.0
            ).clip(0, 255).astype(np.uint8)


def run(lr_path, target_path, disc, device, clothing_mask=None):
    """Run discriminator, return logits grid and image numpy."""
    lr = load_image(lr_path).to(device)
    target = load_image(target_path).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = disc(torch.cat([lr, target], dim=1))

    logits_np = logits[0, 0].float().cpu().numpy()  # (126, 94) native grid
    prob_np = 1.0 / (1.0 + np.exp(-logits_np))      # sigmoid
    target_np = to_display(target)

    return logits_np, prob_np, target_np


def _overlay(target_np, logits_np, patch_mask_np=None, cmap_name="RdBu",
             shared_vmin=None, shared_vmax=None):
    """Nearest-neighbor upscale of logit grid onto image, clothing regions only.

    Uses shared scale if provided, otherwise per-image normalization.
    """
    h, w = target_np.shape[:2]
    cmap = plt.get_cmap(cmap_name)

    if shared_vmin is not None and shared_vmax is not None:
        lo, hi = shared_vmin, shared_vmax
    elif patch_mask_np is not None:
        valid = patch_mask_np > 0.1
        valid_logits = logits_np[valid] if valid.any() else logits_np.ravel()
        lo, hi = valid_logits.min(), valid_logits.max()
    else:
        lo, hi = logits_np.min(), logits_np.max()

    if hi - lo < 1e-6:
        normed = np.full_like(logits_np, 0.5)
    else:
        normed = (logits_np - lo) / (hi - lo)
    colored = cmap(normed)[:, :, :3]  # (126, 94, 3)
    colored_up = np.array(Image.fromarray((colored * 255).astype(np.uint8)).resize(
        (w, h), Image.NEAREST)).astype(np.float32)

    # Blend only on clothing regions
    if patch_mask_np is not None:
        # Upscale patch mask to pixel resolution
        mask_up = np.array(Image.fromarray((patch_mask_np * 255).clip(0, 255).astype(np.uint8)).resize(
            (w, h), Image.NEAREST)).astype(np.float32) / 255.0
        alpha = mask_up[:, :, np.newaxis] * 0.5  # blend strength on clothing
        overlay = (target_np.astype(np.float32) * (1 - alpha) + colored_up * alpha
                   ).clip(0, 255).astype(np.uint8)
    else:
        overlay = (target_np.astype(np.float32) * 0.5 + colored_up * 0.5
                   ).clip(0, 255).astype(np.uint8)
    return overlay, normed


def make_panel_figure(name, target_np, logits_np, out_path, mean_p=None, patch_mask_np=None):
    """Single-target 3-panel figure: image | logit grid (clothing only) | overlay."""
    if mean_p is None:
        prob_np = 1.0 / (1.0 + np.exp(-logits_np))
        mean_p = prob_np.mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Panel 1: Original image
    axes[0].imshow(target_np)
    label = "REAL" if mean_p > 0.5 else "FAKE"
    axes[0].set_title(f"{name}\nMean P(real)={mean_p:.4f} → {label}", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Patch logit grid — masked to clothing, per-image colorscale
    display_logits = logits_np.copy()
    if patch_mask_np is not None:
        valid = patch_mask_np > 0.1
        display_logits = np.where(valid, logits_np, np.nan)
        vmin, vmax = logits_np[valid].min(), logits_np[valid].max()
    else:
        vmin, vmax = logits_np.min(), logits_np.max()
    im = axes[1].imshow(display_logits, cmap="inferno", interpolation="nearest", aspect="auto",
                        vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Patch Logits (clothing only)\n"
                      f"range [{vmin:.2f}, {vmax:.2f}]", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="logit (bright=more real)")

    # Panel 3: Overlay (clothing regions only)
    overlay, _ = _overlay(target_np, logits_np, patch_mask_np)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (bright=more real, dark=more fake)", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_comparison_figure(results, out_path, patch_mask_np=None):
    """Comparison figure: top = GT & pred side by side, bottom = logit grids + pred overlay."""
    assert len(results) >= 2, "Need at least 2 targets for comparison"

    # Shared colorscale across all: symmetric around 0
    if patch_mask_np is not None:
        valid = patch_mask_np > 0.1
        all_valid = np.concatenate([r["logits"][valid].ravel() for r in results])
    else:
        valid = None
        all_valid = np.concatenate([r["logits"].ravel() for r in results])
    abs_max = max(abs(all_valid.min()), abs(all_valid.max()), 0.1)
    shared_vmin, shared_vmax = -abs_max, abs_max

    # 2×6 grid: top row each image spans 3 cols, bottom row each panel spans 2 cols
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.12, wspace=0.08)

    # Top row: GT and pred images side by side (2 panels, each spanning 3 cols)
    top_axes = []
    for i, r in enumerate(results[:2]):
        ax = fig.add_subplot(gs[0, i * 3:(i + 1) * 3])
        top_axes.append(ax)
        mean_p = r.get("mean_p")
        if mean_p is None:
            prob_np = 1.0 / (1.0 + np.exp(-r["logits"]))
            mean_p = prob_np.mean()
        label = "REAL" if mean_p > 0.5 else "FAKE"
        ax.imshow(r["target_np"])
        ax.set_title(f"{r['name']}\nP(real)={mean_p:.4f} → {label}", fontsize=12)
        ax.axis("off")

    # Bottom row: GT logits | Pred logits | Pred overlay (3 panels, each spanning 2 cols)
    bot_axes = []
    logit_axes = []
    im = None
    for i, r in enumerate(results[:2]):
        ax = fig.add_subplot(gs[1, i * 2:(i + 1) * 2])
        bot_axes.append(ax)
        logit_axes.append(ax)

        logits_np = r["logits"]
        display_logits = logits_np.copy()
        if valid is not None:
            display_logits = np.where(valid, logits_np, np.nan)
            vl = logits_np[valid]
            subtitle = f"{r['name']} logits [{vl.min():.2f}, {vl.max():.2f}]"
        else:
            subtitle = f"{r['name']} logits [{logits_np.min():.2f}, {logits_np.max():.2f}]"

        im = ax.imshow(display_logits, cmap="RdBu", interpolation="nearest", aspect="auto",
                       vmin=shared_vmin, vmax=shared_vmax)
        ax.set_title(subtitle, fontsize=10)
        ax.axis("off")

    # 3rd bottom panel: pred overlay
    pred_r = results[-1]
    ax = fig.add_subplot(gs[1, 4:6])
    bot_axes.append(ax)
    overlay, _ = _overlay(pred_r["target_np"], pred_r["logits"], patch_mask_np,
                          shared_vmin=shared_vmin, shared_vmax=shared_vmax)
    ax.imshow(overlay)
    ax.set_title("Pred overlay (blue=real, red=fake)", fontsize=10)
    ax.axis("off")

    # Shared colorbar on logit panels
    if im is not None:
        fig.colorbar(im, ax=logit_axes, fraction=0.046, pad=0.04,
                     label="logit (>0 = real, <0 = fake)")

    plt.suptitle("PatchGAN Discriminator — Clothing Only", fontsize=14, y=0.98)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run PatchGAN on images and produce heatmaps")
    parser.add_argument("--lr", type=str, required=True, help="LR image path")
    parser.add_argument("--target", type=str, nargs="+", required=True, help="Target image(s) to evaluate")
    parser.add_argument("--disc_ckpt", type=str, default=DEFAULT_DISC, help="Discriminator checkpoint")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for heatmaps")
    parser.add_argument("--image_id", type=str, default=None,
                        help="Image ID for clothing mask (e.g. 02202_00). Auto-detected from target filename if not set.")
    parser.add_argument("--full_image", action="store_true",
                        help="Disable clothing masking (run on full image)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    # Load discriminator
    disc = NLayerDiscriminator(input_nc=6).to(device)
    state = torch.load(args.disc_ckpt, map_location=device, weights_only=True)
    if "model" in state and "optimizer" in state:
        disc.load_state_dict(state["model"])
    elif "model_state_dict" in state:
        disc.load_state_dict(state["model_state_dict"])
    else:
        disc.load_state_dict(state)
    disc.eval()
    print(f"Loaded discriminator from {args.disc_ckpt}")

    # Load clothing mask
    clothing_mask = None
    if not args.full_image:
        # Auto-detect image_id from first target filename (e.g. "02202_00_pred.png" → "02202_00")
        image_id = args.image_id
        if image_id is None:
            # Extract base image_id: strip known suffixes
            base = os.path.splitext(os.path.basename(args.target[0]))[0]
            for suffix in ("_pred", "_hr_gt", "_hr", "_lr"):
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    break
            image_id = base
        clothing_mask = load_clothing_mask_tensor(image_id, h=1024, w=768)
        if clothing_mask is not None:
            print(f"Clothing mask: {image_id} ({clothing_mask.sum().int().item()} / {clothing_mask.numel()} pixels)")
        else:
            print(f"WARNING: No clothing mask found for {image_id}, running full image")

    # Downscale clothing mask to patch grid for accurate stats
    patch_mask_np = None
    if clothing_mask is not None:
        pm = F.interpolate(clothing_mask.unsqueeze(0), size=(126, 94), mode="area")
        patch_mask_np = pm[0, 0].numpy()  # (126, 94), values in [0,1]

    results = []
    for target_path in args.target:
        name = os.path.splitext(os.path.basename(target_path))[0]
        print(f"\n--- {name} ---")

        logits_np, prob_np, target_np = run(args.lr, target_path, disc, device, clothing_mask)

        # Average P(real) only over clothing patches
        if patch_mask_np is not None:
            valid = patch_mask_np > 0.1
            mean_p = prob_np[valid].mean() if valid.any() else prob_np.mean()
        else:
            mean_p = prob_np.mean()

        label = "REAL" if mean_p > 0.5 else "FAKE"
        print(f"  Mean P(real): {mean_p:.4f} → {label}")
        print(f"  Logits: shape={logits_np.shape}, min={logits_np.min():.2f}, max={logits_np.max():.2f}")

        results.append({"name": name, "logits": logits_np, "prob": prob_np,
                        "target_np": target_np, "mean_p": mean_p})

        # Per-target panel figure
        panel_path = os.path.join(args.output_dir, f"{name}_disc_panel.png")
        make_panel_figure(name, target_np, logits_np, panel_path, mean_p=mean_p, patch_mask_np=patch_mask_np)
        print(f"  Saved {panel_path}")

    # Comparison figure if multiple targets
    if len(results) > 1:
        compare_path = os.path.join(args.output_dir, "disc_compare.png")
        make_comparison_figure(results, compare_path, patch_mask_np=patch_mask_np)
        print(f"\nSaved comparison → {compare_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
