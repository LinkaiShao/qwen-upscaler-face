"""Precompute pixel cache: run frozen LoRA generator on all samples, save PNGs."""

import hashlib
import os
import time

import torch
from PIL import Image
from safetensors.torch import load_file as safe_load

from qwen_upscaler_face.config import Args
from qwen_upscaler_face.models import (
    denormalize_latents,
    load_transformer,
    load_vae,
    pack_latents,
    unpack_latents,
)
from qwen_upscaler_face.checkpoint import load_lora_weights

from .dataset import to_uint8


def _cache_key(row) -> str:
    key_str = f"{row['model_lr']}_{row['model_hr']}"
    return hashlib.md5(key_str.encode()).hexdigest()


def precompute(args_cli):
    """Run frozen LoRA generator on every train+val sample, save PNGs."""
    import pandas as pd

    cfg = Args()

    # Auto-detect LoRA rank from checkpoint (same as make_package.py)
    st = safe_load(os.path.join(args_cli.checkpoint, "lora_weights.safetensors"))
    for k, v in st.items():
        if "lora_A" in k and v.ndim == 2:
            cfg.rank = v.shape[0]
            cfg.alpha = cfg.rank
            break
    del st
    print(f"Auto-detected LoRA rank={cfg.rank}")

    # CLI overrides (if explicitly set)
    if args_cli.rank is not None:
        cfg.rank = args_cli.rank
    if args_cli.alpha is not None:
        cfg.alpha = args_cli.alpha

    cache_dir = args_cli.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device("cuda")
    weight_dtype = torch.bfloat16

    # Load generator + VAE
    print("Loading LoRA generator...")
    transformer = load_transformer(cfg, device=device)
    if not load_lora_weights(transformer, args_cli.checkpoint):
        raise FileNotFoundError(f"No LoRA weights found in: {args_cli.checkpoint}")
    transformer.eval()

    print("Loading VAE...")
    vae = load_vae(cfg, device=device)

    # Load dataframe
    df = pd.read_parquet(cfg.data_parquet)
    splits = getattr(args_cli, "splits", None)
    if splits:
        df_work = df[df["split"].isin(splits)].reset_index(drop=True)
    else:
        df_work = df.reset_index(drop=True)

    # Optional: limit to N random samples
    max_samples = getattr(args_cli, "max_samples", None)
    if max_samples and max_samples < len(df_work):
        df_work = df_work.sample(n=max_samples, random_state=42).reset_index(drop=True)

    total = len(df_work)
    print(f"Precomputing {total} samples → {cache_dir}")

    skipped = 0
    t0 = time.time()

    for idx in range(total):
        row = df_work.iloc[idx]
        key = _cache_key(row)
        split = row["split"]

        # Skip if all three PNGs already exist
        pred_path = os.path.join(cache_dir, f"{key}_pred.png")
        hr_path = os.path.join(cache_dir, f"{key}_hr.png")
        lr_path = os.path.join(cache_dir, f"{key}_lr.png")
        split_path = os.path.join(cache_dir, f"{key}_split.txt")
        if os.path.exists(pred_path) and os.path.exists(hr_path) and os.path.exists(lr_path):
            skipped += 1
            continue

        # Load latents (skip if missing)
        hr_latent_path = f"{cfg.latent_cache_dir}/{key}_hr_latent.pt"
        if not os.path.exists(hr_latent_path):
            print(f"  SKIP {key}: missing latent files")
            skipped += 1
            continue
        hr_latent = torch.load(
            hr_latent_path, weights_only=True
        ).unsqueeze(0).to(device, dtype=weight_dtype)
        lr_latent = torch.load(
            f"{cfg.latent_cache_dir}/{key}_lr_latent.pt", weights_only=True
        ).unsqueeze(0).to(device, dtype=weight_dtype)
        prompt_embeds = torch.load(
            f"{cfg.latent_cache_dir}/{key}_prompt_embeds.pt", weights_only=True
        ).unsqueeze(0).to(device, dtype=weight_dtype)
        prompt_mask = torch.load(
            f"{cfg.latent_cache_dir}/{key}_prompt_mask.pt", weights_only=True
        ).unsqueeze(0).to(device, dtype=torch.long)

        B, C, H, W = hr_latent.shape
        hr_packed = pack_latents(hr_latent, B, C, H, W)
        lr_packed = pack_latents(lr_latent, B, C, H, W)

        img_shapes = [[(1, H // 2, W // 2), (1, H // 2, W // 2)]] * B
        txt_seq_lens = prompt_mask.sum(dim=1).tolist()

        # Single-step inference (t=0)
        with torch.no_grad():
            t_batch = torch.zeros(B, device=device, dtype=weight_dtype)
            hidden = torch.cat([lr_packed, lr_packed], dim=1)
            pred = transformer(
                hidden_states=hidden,
                timestep=t_batch,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]
            v_pred = pred[:, :lr_packed.size(1), :]
            z = lr_packed + v_pred

            # Decode pred
            pred_unpacked = unpack_latents(z, H, W)
            pred_denorm = denormalize_latents(pred_unpacked.to(vae.dtype), vae)
            pred_decoded = vae.decode(pred_denorm, return_dict=False)[0][:, :, 0].clamp(-1, 1)

            # Decode HR
            hr_unpacked = unpack_latents(hr_packed, H, W)
            hr_denorm = denormalize_latents(hr_unpacked.to(vae.dtype), vae)
            hr_decoded = vae.decode(hr_denorm, return_dict=False)[0][:, :, 0].clamp(-1, 1)

            # Decode LR (same pixel domain as HR)
            lr_5d = lr_latent.unsqueeze(2)  # (B, 16, 1, H, W)
            lr_denorm = denormalize_latents(lr_5d.to(vae.dtype), vae)
            lr_decoded = vae.decode(lr_denorm, return_dict=False)[0][:, :, 0].clamp(-1, 1)

        # Save PNGs
        Image.fromarray(to_uint8(pred_decoded[0])).save(pred_path)
        Image.fromarray(to_uint8(hr_decoded[0])).save(hr_path)
        Image.fromarray(to_uint8(lr_decoded[0])).save(lr_path)

        # Save split label
        with open(split_path, "w") as f:
            f.write(split)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1 - skipped) / max(elapsed, 1)
            eta = (total - idx - 1) / max(rate, 1e-6)
            print(f"  [{idx+1}/{total}] {rate:.1f} samples/s  ETA {eta/60:.0f}m  (skipped {skipped})")

    print(f"Done. {total - skipped} generated, {skipped} skipped (already cached).")
