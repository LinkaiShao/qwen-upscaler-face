#!/usr/bin/env python3
"""Run one-sample inference with the exact same rollout as validation.py."""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Ensure local package imports work when run from repository root.
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/vton upscaling")

from qwen_upscaler_face.config import Args
from qwen_upscaler_face.models import (
    denormalize_latents,
    load_transformer,
    load_vae,
    pack_latents,
    unpack_latents,
)
from qwen_upscaler_face.checkpoint import load_lora_weights
from qwen_upscaler_face.lighting import lock_lighting
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel


def _cache_key(row) -> str:
    key_str = f"{row['model_lr']}_{row['model_hr']}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    arr = ((img_chw.detach().cpu().float().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255.0)
    return arr.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Qwen face LoRA one-sample inference (val rollout)")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint dir containing lora_weights.safetensors")
    parser.add_argument("--no_lora", action="store_true", help="Run zero-shot (do not load LoRA)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=1, help="Number of Euler steps (1 = single-step residual)")
    parser.add_argument("--rank", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="Override LoRA alpha")
    parser.add_argument("--lock_lighting", action="store_true", help="Force-enable lighting lock")
    parser.add_argument("--no_lock_lighting", action="store_true", help="Force-disable lighting lock")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/link/Desktop/Code/fashion gen testing/vton upscaling/qwen_lora_comparison",
    )
    parser.add_argument("--tag", type=str, default="face_rollout", help="Output filename prefix")
    args_cli = parser.parse_args()

    cfg = Args()
    if args_cli.rank is not None:
        cfg.rank = args_cli.rank
    if args_cli.alpha is not None:
        cfg.alpha = args_cli.alpha
    if args_cli.lock_lighting and args_cli.no_lock_lighting:
        raise ValueError("Use only one of --lock_lighting or --no_lock_lighting.")
    if args_cli.lock_lighting:
        cfg.lock_lighting = True
    if args_cli.no_lock_lighting:
        cfg.lock_lighting = False

    os.makedirs(args_cli.output_dir, exist_ok=True)

    device = torch.device("cuda")
    weight_dtype = torch.bfloat16

    print("Loading model + VAE...")
    if args_cli.no_lora:
        # True zero-shot path: load base transformer without PEFT wrappers.
        transformer = QwenImageTransformer2DModel.from_pretrained(
            cfg.pretrained_model,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to(device)
    else:
        transformer = load_transformer(cfg, device=device)
        if not args_cli.checkpoint:
            raise ValueError("--checkpoint is required unless --no_lora is set.")
        if not load_lora_weights(transformer, args_cli.checkpoint):
            raise FileNotFoundError(f"No LoRA weights found in: {args_cli.checkpoint}")
    vae = load_vae(cfg, device=device)

    print(f"Loading sample split={args_cli.split} index={args_cli.index}...")
    df = pd.read_parquet(cfg.data_parquet)
    df_split = df[df["split"] == args_cli.split].reset_index(drop=True)
    if len(df_split) == 0:
        raise ValueError(f"No rows found for split={args_cli.split}")
    if args_cli.index < 0 or args_cli.index >= len(df_split):
        raise IndexError(f"index={args_cli.index} out of range for split={args_cli.split} (n={len(df_split)})")

    row = df_split.iloc[args_cli.index]
    key = _cache_key(row)

    hr_latent = torch.load(f"{cfg.latent_cache_dir}/{key}_hr_latent.pt", weights_only=True).unsqueeze(0).to(device, dtype=weight_dtype)
    lr_latent = torch.load(f"{cfg.latent_cache_dir}/{key}_lr_latent.pt", weights_only=True).unsqueeze(0).to(device, dtype=weight_dtype)
    prompt_embeds = torch.load(f"{cfg.latent_cache_dir}/{key}_prompt_embeds.pt", weights_only=True).unsqueeze(0).to(device, dtype=weight_dtype)
    prompt_mask = torch.load(f"{cfg.latent_cache_dir}/{key}_prompt_mask.pt", weights_only=True).unsqueeze(0).to(device, dtype=torch.long)

    B, C, H, W = hr_latent.shape
    hr_packed = pack_latents(hr_latent, B, C, H, W)
    lr_packed = pack_latents(lr_latent, B, C, H, W)

    img_shapes = [[(1, H // 2, W // 2), (1, H // 2, W // 2)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    dt = 1.0 / args_cli.n_steps
    z = lr_packed.clone()

    print("Running LR->HR Euler rollout...")
    transformer.eval()
    with torch.no_grad():
        for k in range(args_cli.n_steps):
            t = k * dt
            t_batch = torch.full((B,), t, device=device, dtype=weight_dtype)
            hidden = torch.cat([z, lr_packed], dim=1)
            pred = transformer(
                hidden_states=hidden,
                timestep=t_batch,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]
            v_pred = pred[:, :z.size(1), :]
            z = z + dt * v_pred

    mse_latent = torch.mean((z.float() - hr_packed.float()) ** 2).item()

    print("Decoding images...")
    pred_unpacked = unpack_latents(z, H, W)
    pred_denorm = denormalize_latents(pred_unpacked.to(vae.dtype), vae)
    pred_decoded = vae.decode(pred_denorm, return_dict=False)[0][:, :, 0].clamp(-1, 1)

    hr_unpacked = unpack_latents(hr_packed, H, W)
    hr_denorm = denormalize_latents(hr_unpacked.to(vae.dtype), vae)
    hr_decoded = vae.decode(hr_denorm, return_dict=False)[0][:, :, 0].clamp(-1, 1)

    lr_unpacked = lr_latent.unsqueeze(2)
    lr_denorm = denormalize_latents(lr_unpacked.to(vae.dtype), vae)
    lr_decoded = vae.decode(lr_denorm, return_dict=False)[0][:, :, 0].clamp(-1, 1)

    if cfg.lock_lighting:
        pred_decoded = lock_lighting(pred_decoded, lr_decoded, cfg.lighting_y_blend, cfg.lighting_blur_radius)

    pred_np = _to_uint8(pred_decoded[0])
    hr_np = _to_uint8(hr_decoded[0])
    lr_np = _to_uint8(lr_decoded[0])

    Image.fromarray(lr_np).save(os.path.join(args_cli.output_dir, f"{args_cli.tag}_lr.png"))
    Image.fromarray(hr_np).save(os.path.join(args_cli.output_dir, f"{args_cli.tag}_hr_gt.png"))
    Image.fromarray(pred_np).save(os.path.join(args_cli.output_dir, f"{args_cli.tag}_pred.png"))

    mse_px = float(np.mean((pred_np.astype(np.float32) - hr_np.astype(np.float32)) ** 2))
    print(f"Saved outputs to: {args_cli.output_dir}")
    print(f"Latent MSE: {mse_latent:.6f}")
    print(f"Pixel MSE : {mse_px:.6f}")


if __name__ == "__main__":
    main()
