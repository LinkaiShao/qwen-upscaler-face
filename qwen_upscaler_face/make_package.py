#!/usr/bin/env python3
"""Generate a full comparison package for a validation image at a given checkpoint.

Outputs (saved to checkpoint folder):
    {image_id}_hr_gt.png        - HR ground truth (VAE decode)
    {image_id}_lr.png           - LR baseline (VAE decode)
    {image_id}_pred.png         - LoRA prediction (768x1024)
    {image_id}_pred_1440.png    - LoRA prediction (1088x1440)
    {image_id}_top_k_mining.png - Top-k worst clothing patches visualization
    {image_id}_blobs.png        - Blob regions from top-k seeds

Usage:
    python -m qwen_upscaler_face.make_package --val_idx 3 --ckpt /path/to/step-6500
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/vton upscaling")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from scipy import ndimage
from safetensors.torch import load_file as safe_load
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from qwen_upscaler_face.config import Args
from qwen_upscaler_face.models import (
    load_transformer, load_vae, encode_vae_image,
    pack_latents, unpack_latents, denormalize_latents,
)
from qwen_upscaler_face.checkpoint import load_lora_weights
from qwen_upscaler_face.top_k_mining_eval import mine_top_k_patches, load_clothing_mask
from qwen_upscaler_face.blob_regions import seeds_to_blob_regions

PARSE_DIR = "/home/link/Desktop/Code/fashion gen testing/VITON-HD-dataset/train/image-parse-v3"
DEFAULT_CKPT = (
    "/home/link/Desktop/Code/fashion gen testing/"
    "qwen_upscale_face_out/qwen_face_20260224_001227/step-6500"
)

device = torch.device("cuda")
weight_dtype = torch.bfloat16


def to_uint8(img_chw):
    return ((img_chw.detach().cpu().float().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255.0
            ).clip(0, 255).astype(np.uint8)


def decode(packed, H, W, vae):
    with torch.no_grad():
        unpacked = unpack_latents(packed, H, W)
        denorm = denormalize_latents(unpacked.to(vae.dtype), vae)
        decoded = vae.decode(denorm, return_dict=False)[0][:, :, 0].clamp(-1, 1)
    return to_uint8(decoded[0])


def make_top_k_plot(pred_np, hr_np, clothing_mask, top_patches, image_id, ckpt_name,
                    psnr_pred, ssim_pred, patch_size, out_path):
    error = np.abs(pred_np.astype(np.float32) - hr_np.astype(np.float32)).mean(axis=2)
    masked_error = error * clothing_mask
    ps = patch_size
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 3, figsize=(18, 13))

    axes[0, 0].imshow(hr_np)
    axes[0, 0].set_title(f"HR GT: {image_id}", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_np)
    axes[0, 1].set_title(f"Pred (PSNR={psnr_pred:.1f} SSIM={ssim_pred:.4f})", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_np)
    for i, (err, py, px) in enumerate(top_patches):
        rect = plt.Rectangle((px, py), ps, ps, linewidth=2, edgecolor=colors[i], facecolor="none")
        axes[0, 2].add_patch(rect)
        axes[0, 2].text(px + ps // 2, py - 4, f"{i + 1}", color=colors[i],
                        fontsize=8, fontweight="bold", ha="center")
    axes[0, 2].set_title(f"Top-10 Worst Clothing Patches", fontsize=11)
    axes[0, 2].axis("off")

    nonzero = masked_error[masked_error > 0]
    vmax = float(np.percentile(nonzero, 95)) if len(nonzero) > 0 else 1.0
    axes[1, 0].imshow(masked_error, cmap="hot", vmin=0, vmax=max(vmax, 1.0))
    for i, (err, py, px) in enumerate(top_patches):
        rect = plt.Rectangle((px, py), ps, ps, linewidth=2, edgecolor="cyan", facecolor="none")
        axes[1, 0].add_patch(rect)
    axes[1, 0].set_title("Error Heatmap (clothing only)", fontsize=11)
    axes[1, 0].axis("off")

    for j in range(min(2, len(top_patches))):
        err, py, px = top_patches[j]
        cw = ps * 3
        cy, cx = py + ps // 2, px + ps // 2
        y1, y2 = max(0, cy - cw // 2), min(pred_np.shape[0], max(0, cy - cw // 2) + cw)
        x1, x2 = max(0, cx - cw // 2), min(pred_np.shape[1], max(0, cx - cw // 2) + cw)
        combined = np.concatenate([hr_np[y1:y2, x1:x2], pred_np[y1:y2, x1:x2]], axis=1)
        axes[1, 1 + j].imshow(combined)
        axes[1, 1 + j].axvline(x=(x2 - x1), color="white", linewidth=1)
        axes[1, 1 + j].set_title(f"#{j + 1} MAE={err:.1f} (GT | Pred)", fontsize=9)
        axes[1, 1 + j].axis("off")

    mae_overall = float(masked_error.sum() / max(clothing_mask.sum(), 1))
    plt.suptitle(f"Top-10 Mining: {image_id} | {ckpt_name} | Clothing MAE={mae_overall:.2f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_blob_plot(pred_np, hr_np, clothing_mask, top_patches, label_map, n_regions,
                   blob_mask, image_id, ckpt_name, patch_size, out_path):
    masked_error = (np.abs(pred_np.astype(np.float32) - hr_np.astype(np.float32)).mean(axis=2)
                    * clothing_mask)
    ps = patch_size
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_regions, 1)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Prediction with blob overlay
    axes[0].imshow(pred_np)
    overlay = np.zeros((*pred_np.shape[:2], 4), dtype=np.float32)
    for rid in range(1, n_regions + 1):
        c = colors[rid - 1]
        overlay[label_map == rid] = [c[0], c[1], c[2], 0.35]
    axes[0].imshow(overlay)
    for i, (err, py, px) in enumerate(top_patches):
        cy, cx = py + ps // 2, px + ps // 2
        rid_at = label_map[min(cy, clothing_mask.shape[0] - 1),
                           min(cx, clothing_mask.shape[1] - 1)]
        mc = "white" if rid_at > 0 else "gray"
        axes[0].plot(cx, cy, "x", color=mc, markersize=8, markeredgewidth=2)
        axes[0].text(cx + 6, cy, f"{i + 1}", color=mc, fontsize=7, fontweight="bold")
    axes[0].set_title(f"Blobs ({n_regions} regions, min_seeds=3)", fontsize=11)
    axes[0].axis("off")

    # Error heatmap with blob boundaries
    nonzero = masked_error[masked_error > 0]
    vmax = float(np.percentile(nonzero, 95)) if len(nonzero) > 0 else 1.0
    axes[1].imshow(masked_error, cmap="hot", vmin=0, vmax=max(vmax, 1.0))
    for rid in range(1, n_regions + 1):
        region = (label_map == rid).astype(float)
        axes[1].contour(region, [0.5], colors=[colors[rid - 1][:3]], linewidths=1.5)
    axes[1].set_title("Error + Blob Boundaries", fontsize=11)
    axes[1].axis("off")

    # Label map with discarded clothing grayed out
    label_vis = np.zeros((*label_map.shape, 3), dtype=np.float32)
    for rid in range(1, n_regions + 1):
        label_vis[label_map == rid] = colors[rid - 1][:3]
    label_vis[clothing_mask & ~blob_mask] = [0.15, 0.15, 0.15]
    axes[2].imshow(label_vis)
    axes[2].set_title("Kept Regions (gray=discarded)", fontsize=11)
    axes[2].axis("off")

    plt.suptitle(f"Blob Regions: {image_id} | {ckpt_name} | {n_regions} regions", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate full comparison package for a val image")
    parser.add_argument("--val_idx", type=int, default=0, help="Validation set index")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Checkpoint path")
    parser.add_argument("--k", type=int, default=10, help="Top-k patches")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--blob_radius", type=int, default=64)
    parser.add_argument("--min_seeds", type=int, default=3)
    cli = parser.parse_args()

    cfg = Args()
    df = pd.read_parquet(cfg.data_parquet)
    val_rows = df[df["split"] == "val"].reset_index(drop=True)
    ld = Path(cfg.latent_cache_dir)

    row = val_rows.iloc[cli.val_idx]
    image_id = Path(row["model_hr"]).stem
    key_str = hashlib.md5(f"{row['model_lr']}_{row['model_hr']}".encode()).hexdigest()
    ckpt_name = os.path.basename(cli.ckpt)

    print(f"=== Package: val[{cli.val_idx}] = {image_id} | {ckpt_name} ===")

    # Auto-detect LoRA rank
    st = safe_load(os.path.join(cli.ckpt, "lora_weights.safetensors"))
    for k, v in st.items():
        if "lora_A" in k and v.ndim == 2:
            cfg.rank = v.shape[0]
            cfg.alpha = cfg.rank
            break
    del st
    print(f"LoRA rank={cfg.rank}")

    # Load model
    print("Loading transformer + LoRA...")
    transformer = load_transformer(cfg, device=device)
    load_lora_weights(transformer, cli.ckpt)
    transformer.eval()

    print("Loading VAE...")
    vae = load_vae(cfg, device=device)

    # ── 1. Standard res: pred, LR, HR GT ──
    print("\n[1/4] Standard res (768x1024)...")
    hr_latent = torch.load(ld / f"{key_str}_hr_latent.pt", weights_only=True).unsqueeze(0).to(device, dtype=weight_dtype)
    lr_latent = torch.load(ld / f"{key_str}_lr_latent.pt", weights_only=True).unsqueeze(0).to(device, dtype=weight_dtype)
    prompt_embeds = torch.load(ld / f"{key_str}_prompt_embeds.pt", weights_only=True).unsqueeze(0).to(device, dtype=weight_dtype)
    prompt_mask = torch.load(ld / f"{key_str}_prompt_mask.pt", weights_only=True).unsqueeze(0).to(device, dtype=torch.long)

    B, C, H, W = hr_latent.shape
    hr_packed = pack_latents(hr_latent, B, C, H, W)
    lr_packed = pack_latents(lr_latent, B, C, H, W)
    img_shapes = [[(1, H // 2, W // 2), (1, H // 2, W // 2)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    with torch.no_grad():
        t_batch = torch.zeros(B, device=device, dtype=weight_dtype)
        hidden = torch.cat([lr_packed, lr_packed], dim=1)
        pred = transformer(
            hidden_states=hidden, timestep=t_batch,
            encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes, txt_seq_lens=txt_seq_lens, return_dict=False,
        )[0]
        z_pred = lr_packed + pred[:, :lr_packed.size(1), :]

    pred_np = decode(z_pred, H, W, vae)
    hr_np = decode(hr_packed, H, W, vae)
    lr_np = decode(lr_packed, H, W, vae)

    Image.fromarray(pred_np).save(os.path.join(cli.ckpt, f"{image_id}_pred.png"))
    Image.fromarray(hr_np).save(os.path.join(cli.ckpt, f"{image_id}_hr_gt.png"))
    Image.fromarray(lr_np).save(os.path.join(cli.ckpt, f"{image_id}_lr.png"))

    psnr_pred = peak_signal_noise_ratio(hr_np, pred_np)
    ssim_pred = structural_similarity(hr_np, pred_np, channel_axis=2)
    psnr_lr = peak_signal_noise_ratio(hr_np, lr_np)
    print(f"  Pred: PSNR={psnr_pred:.2f} SSIM={ssim_pred:.4f}")
    print(f"  LR:   PSNR={psnr_lr:.2f}")
    print(f"  Saved _pred.png, _hr_gt.png, _lr.png")

    # ── 2. 1440p prediction ──
    print("\n[2/4] 1440p (1088x1440)...")
    TARGET_W, TARGET_H = 1088, 1440
    lr_pil = Image.open(row["model_lr"]).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
    lr_arr = np.array(lr_pil).astype(np.float32) / 127.5 - 1.0
    lr_t = torch.from_numpy(lr_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device, dtype=weight_dtype)

    with torch.no_grad():
        lr_latent_1440 = encode_vae_image(vae, lr_t)
    B2, C2, H2, W2 = lr_latent_1440.shape
    lr_packed_1440 = pack_latents(lr_latent_1440.to(weight_dtype), B2, C2, H2, W2)
    img_shapes_1440 = [[(1, H2 // 2, W2 // 2), (1, H2 // 2, W2 // 2)]] * B2

    with torch.no_grad():
        t_batch = torch.zeros(B2, device=device, dtype=weight_dtype)
        hidden = torch.cat([lr_packed_1440, lr_packed_1440], dim=1)
        pred = transformer(
            hidden_states=hidden, timestep=t_batch,
            encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes_1440, txt_seq_lens=txt_seq_lens, return_dict=False,
        )[0]
        z_1440 = lr_packed_1440 + pred[:, :lr_packed_1440.size(1), :]

    pred_1440_np = decode(z_1440, H2, W2, vae)
    Image.fromarray(pred_1440_np).save(os.path.join(cli.ckpt, f"{image_id}_pred_1440.png"))
    print(f"  Saved _pred_1440.png ({pred_1440_np.shape[1]}x{pred_1440_np.shape[0]})")

    # ── 3. Top-k mining ──
    print("\n[3/4] Top-k mining...")
    clothing_mask = load_clothing_mask(image_id, cfg.target_width, cfg.target_height)
    if clothing_mask is None:
        print(f"  ERROR: No parse map for {image_id}")
        return

    error = np.abs(pred_np.astype(np.float32) - hr_np.astype(np.float32)).mean(axis=2)
    masked_error = error * clothing_mask
    top_patches = mine_top_k_patches(masked_error, clothing_mask, k=cli.k, patch_size=cli.patch_size)

    print(f"  Top-{len(top_patches)} worst patches:")
    for i, (err, py, px) in enumerate(top_patches):
        print(f"    {i + 1:2d}. ({px:4d}, {py:4d}) MAE={err:.1f}")

    make_top_k_plot(
        pred_np, hr_np, clothing_mask, top_patches, image_id, ckpt_name,
        psnr_pred, ssim_pred, cli.patch_size,
        os.path.join(cli.ckpt, f"{image_id}_top_k_mining.png"),
    )
    print(f"  Saved _top_k_mining.png")

    # ── 4. Blob regions ──
    print("\n[4/4] Blob regions...")
    label_map, n_regions, blob_mask = seeds_to_blob_regions(
        top_patches, clothing_mask,
        patch_size=cli.patch_size, radius=cli.blob_radius, min_seeds=cli.min_seeds,
    )

    print(f"  {n_regions} regions (min_seeds={cli.min_seeds}, R={cli.blob_radius}):")
    ps = cli.patch_size
    for rid in range(1, n_regions + 1):
        n_pix = (label_map == rid).sum()
        n_s = sum(1 for _, py, px in top_patches
                  if label_map[min(py + ps // 2, clothing_mask.shape[0] - 1),
                               min(px + ps // 2, clothing_mask.shape[1] - 1)] == rid)
        print(f"    Region {rid}: {n_pix:,} px ({n_s} seeds)")

    make_blob_plot(
        pred_np, hr_np, clothing_mask, top_patches, label_map, n_regions,
        blob_mask, image_id, ckpt_name, cli.patch_size,
        os.path.join(cli.ckpt, f"{image_id}_blobs.png"),
    )
    print(f"  Saved _blobs.png")

    print(f"\n=== Done! All outputs in {cli.ckpt} ===")


if __name__ == "__main__":
    main()
