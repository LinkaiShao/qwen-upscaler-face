#!/usr/bin/env python3
"""Top-k mining evaluation: find the k worst spatial patches within the
clothing region region of val[0] for the step-4500 checkpoint."""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/vton upscaling")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

from qwen_upscaler_face.config import Args
from qwen_upscaler_face.models import (
    load_transformer, load_vae,
    pack_latents, unpack_latents, denormalize_latents,
)
from qwen_upscaler_face.checkpoint import load_lora_weights

VITON_BASE = "/home/link/Desktop/Code/fashion gen testing/VITON-HD-dataset"
PARSE_DIR = os.path.join(VITON_BASE, "train/image-parse-v3")
# Clothing labels in VITON-HD ATR format
CLOTHING_LABELS = {5, 6, 7, 9, 12}  # upper-clothes, dress, coat, pants, skirt
DEFAULT_CKPT = (
    "/home/link/Desktop/Code/fashion gen testing/"
    "qwen_upscale_face_out/qwen_face_20260222_010533/step-4500"
)

device = torch.device("cuda")
weight_dtype = torch.bfloat16


def cache_key(row):
    return hashlib.md5(f"{row['model_lr']}_{row['model_hr']}".encode()).hexdigest()


def image_id_from_row(row):
    return Path(row["model_hr"]).stem


def to_uint8(img_chw):
    arr = ((img_chw.detach().cpu().float().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255.0)
    return arr.clip(0, 255).astype(np.uint8)


@torch.no_grad()
def single_step(transformer, lr_packed, prompt_embeds, prompt_mask, img_shapes, txt_seq_lens):
    B = lr_packed.shape[0]
    t_batch = torch.zeros(B, device=device, dtype=weight_dtype)
    hidden = torch.cat([lr_packed, lr_packed], dim=1)
    pred = transformer(
        hidden_states=hidden, timestep=t_batch,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        img_shapes=img_shapes, txt_seq_lens=txt_seq_lens,
        return_dict=False,
    )[0]
    v_pred = pred[:, :lr_packed.size(1), :]
    return lr_packed + v_pred


def decode_to_np(packed, H, W, vae):
    unpacked = unpack_latents(packed, H, W)
    denorm = denormalize_latents(unpacked.to(vae.dtype), vae)
    decoded = vae.decode(denorm, return_dict=False)[0][:, :, 0].clamp(-1, 1)
    return to_uint8(decoded[0])


def load_clothing_mask(image_id, target_w, target_h):
    """Load full parse map and return mask for clothing-only pixels."""
    path = os.path.join(PARSE_DIR, f"{image_id}.png")
    if not os.path.exists(path):
        return None
    parse = np.array(Image.open(path).resize((target_w, target_h), Image.NEAREST))
    mask = np.isin(parse, list(CLOTHING_LABELS))
    return mask


def mine_top_k_patches(error_map, mask, k=10, patch_size=32):
    """Find the k patches with highest mean error within the mask."""
    h, w = error_map.shape
    ps = patch_size
    patches = []
    for py in range(0, h - ps + 1, ps // 2):  # 50% overlap
        for px in range(0, w - ps + 1, ps // 2):
            pmask = mask[py:py + ps, px:px + ps]
            if pmask.sum() < ps * ps * 0.3:
                continue
            patch_err = error_map[py:py + ps, px:px + ps]
            # Mean error only over masked pixels
            mean_err = patch_err[pmask].mean()
            patches.append((float(mean_err), py, px))
    patches.sort(reverse=True)

    # Non-max suppression: skip patches that overlap too much with already-selected ones
    selected = []
    for err, py, px in patches:
        overlap = False
        for _, sy, sx in selected:
            if abs(py - sy) < ps * 0.6 and abs(px - sx) < ps * 0.6:
                overlap = True
                break
        if not overlap:
            selected.append((err, py, px))
        if len(selected) >= k:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Top-k mining: find worst patches on val[0] in clothing region region",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--patch_size", type=int, default=32)
    cli = parser.parse_args()

    cfg = Args()
    df = pd.read_parquet(cfg.data_parquet)
    val_rows = df[df["split"] == "val"].reset_index(drop=True)
    ld = Path(cfg.latent_cache_dir)

    row = val_rows.iloc[0]
    key_str = cache_key(row)
    image_id = image_id_from_row(row)
    print(f"Val[0]: {image_id}  |  k={cli.k}  patch={cli.patch_size}px")
    print(f"Checkpoint: {cli.ckpt}")

    # Auto-detect LoRA rank from checkpoint
    from safetensors.torch import load_file as safe_load
    lora_path = os.path.join(cli.ckpt, "lora_weights.safetensors")
    ckpt_state = safe_load(lora_path)
    for key, val in ckpt_state.items():
        if "lora_A" in key and val.ndim == 2:
            cfg.rank = val.shape[0]
            cfg.alpha = cfg.rank
            break
    del ckpt_state
    print(f"Detected LoRA rank={cfg.rank}")

    # Load model
    print("Loading transformer + LoRA...")
    transformer = load_transformer(cfg, device=device)
    load_lora_weights(transformer, cli.ckpt)
    transformer.eval()

    print("Loading VAE...")
    vae = load_vae(cfg, device=device)

    # Load val[0] cached latents
    hr_latent = torch.load(ld / f"{key_str}_hr_latent.pt", weights_only=True).unsqueeze(0).to(device, dtype=weight_dtype)
    lr_latent = torch.load(ld / f"{key_str}_lr_latent.pt", weights_only=True).unsqueeze(0).to(device, dtype=weight_dtype)
    prompt_embeds = torch.load(ld / f"{key_str}_prompt_embeds.pt", weights_only=True).unsqueeze(0).to(device, dtype=weight_dtype)
    prompt_mask = torch.load(ld / f"{key_str}_prompt_mask.pt", weights_only=True).unsqueeze(0).to(device, dtype=torch.long)

    B, C, H, W = hr_latent.shape
    hr_packed = pack_latents(hr_latent, B, C, H, W)
    lr_packed = pack_latents(lr_latent, B, C, H, W)
    img_shapes = [[(1, H // 2, W // 2), (1, H // 2, W // 2)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    # Run inference
    print("Running single-step inference...")
    z_pred = single_step(transformer, lr_packed, prompt_embeds, prompt_mask, img_shapes, txt_seq_lens)

    # Decode
    pred_np = decode_to_np(z_pred, H, W, vae)
    hr_np = decode_to_np(hr_packed, H, W, vae)
    lr_np = decode_to_np(lr_packed, H, W, vae)

    # Load agnostic mask
    mask = load_clothing_mask(image_id, cfg.target_width, cfg.target_height)
    if mask is None:
        print(f"ERROR: No agnostic map found for {image_id}")
        return

    # Per-pixel error (mean across RGB channels)
    error = np.abs(pred_np.astype(np.float32) - hr_np.astype(np.float32)).mean(axis=2)
    masked_error = error * mask

    # Mine top-k worst patches
    top_patches = mine_top_k_patches(masked_error, mask, k=cli.k, patch_size=cli.patch_size)

    print(f"\nTop-{len(top_patches)} worst patches (clothing region, {cli.patch_size}x{cli.patch_size}):")
    for i, (err, py, px) in enumerate(top_patches):
        print(f"  {i + 1:2d}. ({px:4d}, {py:4d})  MAE={err:.1f}")

    # ── Visualization ──
    ps = cli.patch_size
    colors = plt.cm.tab10(np.linspace(0, 1, cli.k))

    fig, axes = plt.subplots(2, 3, figsize=(18, 13))

    # Row 0: full images
    axes[0, 0].imshow(hr_np)
    axes[0, 0].set_title("HR Ground Truth", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_np)
    axes[0, 1].set_title("LoRA Prediction (step-4500)", fontsize=11)
    axes[0, 1].axis("off")

    # Prediction with top-k patches highlighted
    axes[0, 2].imshow(pred_np)
    for i, (err, py, px) in enumerate(top_patches):
        rect = plt.Rectangle(
            (px, py), ps, ps,
            linewidth=2, edgecolor=colors[i], facecolor="none",
        )
        axes[0, 2].add_patch(rect)
        axes[0, 2].text(
            px + ps // 2, py - 4, f"{i + 1}",
            color=colors[i], fontsize=8, fontweight="bold",
            ha="center", va="bottom",
        )
    axes[0, 2].set_title(f"Top-{cli.k} Worst Patches", fontsize=11)
    axes[0, 2].axis("off")

    # Row 1: error map + zoomed patches
    nonzero = masked_error[masked_error > 0]
    vmax = float(np.percentile(nonzero, 95)) if len(nonzero) > 0 else 1.0
    axes[1, 0].imshow(masked_error, cmap="hot", vmin=0, vmax=max(vmax, 1.0))
    for i, (err, py, px) in enumerate(top_patches):
        rect = plt.Rectangle(
            (px, py), ps, ps,
            linewidth=2, edgecolor="cyan", facecolor="none",
        )
        axes[1, 0].add_patch(rect)
    axes[1, 0].set_title("Error Heatmap (clothing only)", fontsize=11)
    axes[1, 0].axis("off")

    # Zoomed crops of top-5 worst: GT vs Pred side by side
    n_zoom = min(5, len(top_patches))
    crop_w = ps * 3  # show a wider context around the patch
    for i in range(n_zoom):
        err, py, px = top_patches[i]
        cy, cx = py + ps // 2, px + ps // 2
        y1 = max(0, cy - crop_w // 2)
        y2 = min(hr_np.shape[0], y1 + crop_w)
        x1 = max(0, cx - crop_w // 2)
        x2 = min(hr_np.shape[1], x1 + crop_w)

        gt_crop = hr_np[y1:y2, x1:x2]
        pred_crop = pred_np[y1:y2, x1:x2]
        # Side by side
        combined = np.concatenate([gt_crop, pred_crop], axis=1)

        ax_idx = 1 if i < 3 else None  # use remaining subplot space
        if i < 2:
            row_ax = axes[1, 1 + i]
            row_ax.imshow(combined)
            row_ax.set_title(f"#{i + 1} MAE={err:.1f}  (GT | Pred)", fontsize=9)
            row_ax.axvline(x=gt_crop.shape[1], color="white", linewidth=1)
            row_ax.axis("off")

    # If fewer than 2 zoom crops, hide unused axes
    if n_zoom < 2:
        for j in range(n_zoom, 2):
            axes[1, 1 + j].axis("off")

    ckpt_name = os.path.basename(cli.ckpt)
    mae_overall = float(masked_error.sum() / max(mask.sum(), 1))
    plt.suptitle(
        f"Top-{cli.k} Mining on val[0] ({image_id})  |  {ckpt_name}\n"
        f"Overall agnostic MAE={mae_overall:.2f}",
        fontsize=12,
    )
    plt.tight_layout()

    out_dir = os.path.dirname(cli.ckpt)
    out_path = os.path.join(out_dir, "top_k_mining.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
