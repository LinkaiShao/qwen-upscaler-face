#!/usr/bin/env python3
"""Test loss balance across different hyperparameter settings.

Runs N samples through the GAN training step and reports the relative
contribution of each loss term. Lets you tune weights before committing
to a full training run.

Usage:
    python -m qwen_upscaler_face.patch_gan.test_loss_percentage
    python -m qwen_upscaler_face.patch_gan.test_loss_percentage --lambda_id 0.2 --lambda_gan 0.1
    python -m qwen_upscaler_face.patch_gan.test_loss_percentage --n_samples 200 --topk_ratio 0.1
"""

import argparse
import sys

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/vton upscaling")

import numpy as np
import torch

from qwen_upscaler_face_gan.config import Args
from qwen_upscaler_face_gan.dataset import create_dataloaders
from qwen_upscaler_face_gan.training import train_step
from qwen_upscaler_face_gan.models import load_discriminator
from qwen_upscaler_face.models import load_transformer, load_vae
from qwen_upscaler_face.face import load_face_models


def main():
    parser = argparse.ArgumentParser(description="Test loss balance with different hyperparameters")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to test")

    # Tunable weights
    parser.add_argument("--lambda_id", type=float, default=None)
    parser.add_argument("--lambda_anchor", type=float, default=None)
    parser.add_argument("--lambda_gan", type=float, default=None)
    parser.add_argument("--lambda_topk", type=float, default=None)
    parser.add_argument("--face_weight", type=float, default=None)

    # Schedule params (simulates a specific step)
    parser.add_argument("--topk_ratio", type=float, default=0.2)
    parser.add_argument("--lambda_gan_eff", type=float, default=None,
                        help="Effective GAN weight (overrides lambda_gan for testing ramp)")
    parser.add_argument("--compute_id", action="store_true", default=True,
                        help="Compute id loss (default: True)")
    parser.add_argument("--no_id", action="store_true", help="Skip id loss")

    cli = parser.parse_args()

    args = Args()
    # Apply overrides
    for key in ["lambda_id", "lambda_anchor", "lambda_gan", "lambda_topk", "face_weight"]:
        val = getattr(cli, key)
        if val is not None:
            setattr(args, key, val)

    lambda_gan_eff = cli.lambda_gan_eff if cli.lambda_gan_eff is not None else args.lambda_gan
    compute_id = not cli.no_id

    device = torch.device("cuda")
    weight_dtype = torch.bfloat16

    print("Loading models...")
    transformer = load_transformer(args, device=device)
    vae = load_vae(args, device=device)

    # Load LoRA weights using the same path as training
    from qwen_upscaler_face.checkpoint import load_lora_weights
    if not load_lora_weights(transformer, args.lora_ckpt):
        raise FileNotFoundError(f"No LoRA weights in {args.lora_ckpt}")
    for p in transformer.parameters():
        p.requires_grad_(False)
    for n, p in transformer.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)

    _, face_encoder = load_face_models(device)
    disc = load_discriminator(args.disc_ckpt, device)
    disc.requires_grad_(False)

    print("Loading data...")
    train_loader, _ = create_dataloaders(args)

    # Collect stats
    keys = ["flow", "id", "anchor", "gan", "topk_mse"]
    stats = {k: [] for k in keys}
    cos_sims = []
    flow_scales = []

    print(f"Running {cli.n_samples} samples...")
    for i, batch in enumerate(train_loader):
        if i >= cli.n_samples:
            break
        loss_dict = train_step(
            batch, transformer, vae, face_encoder, disc,
            args, device, weight_dtype,
            phase=1,
            global_step=i,
            compute_id=compute_id,
            lambda_gan_eff=lambda_gan_eff,
            topk_ratio=cli.topk_ratio,
        )
        for k in keys:
            stats[k].append(loss_dict[k])
        cos_sims.append(loss_dict["cos_sim"])
        flow_scales.append(loss_dict.get("flow_scale", 1.0))

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{cli.n_samples}")

    # Weights map
    weights = {
        "flow": 1.0,
        "id": args.lambda_id,
        "anchor": args.lambda_anchor,
        "gan": lambda_gan_eff,
        "topk_mse": args.lambda_topk,
    }

    print("\n" + "=" * 65)
    print("CONFIG")
    print("=" * 65)
    for k in keys:
        print(f"  lambda_{k:10s} = {weights[k]:.4f}")
    print(f"  topk_ratio       = {cli.topk_ratio}")
    print(f"  compute_id       = {compute_id}")
    print(f"  face_weight      = {args.face_weight}")
    print(f"  min_flow_frac    = {args.min_flow_frac}")

    print("\n" + "=" * 65)
    print("RAW LOSS MAGNITUDES")
    print("=" * 65)
    for k in keys:
        v = np.array(stats[k])
        print(f"  {k:10s}  mean={v.mean():.6f}  std={v.std():.6f}  min={v.min():.6f}  max={v.max():.6f}")

    print(f"\n  cos_sim     mean={np.mean(cos_sims):.4f}  min={np.min(cos_sims):.4f}  max={np.max(cos_sims):.4f}")

    fs = np.array(flow_scales)
    print(f"  flow_scale  mean={fs.mean():.4f}  min={fs.min():.4f}  max={fs.max():.4f}")

    # After-scale contributions
    mean_scale = fs.mean()
    print("\n" + "=" * 65)
    print(f"WEIGHTED CONTRIBUTIONS (after flow floor, avg scale={mean_scale:.3f})")
    print("=" * 65)
    flow_contrib = np.mean(stats["flow"])
    other_contribs = {k: weights[k] * np.mean(stats[k]) * mean_scale for k in keys if k != "flow"}
    total = flow_contrib + sum(other_contribs.values())

    # Flow
    pct = flow_contrib / total * 100 if total > 0 else 0
    bar = "#" * int(pct / 2)
    print(f"  1.0000 * {'flow':10s} = {flow_contrib:.6f}  ({pct:5.1f}%)  {bar}")
    # Others
    for k in keys:
        if k == "flow":
            continue
        contrib = other_contribs[k]
        pct = contrib / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {weights[k]:.4f} * {k:10s} = {contrib:.6f}  ({pct:5.1f}%)  {bar}")

    print(f"\n  TOTAL = {total:.6f}")

    # Checks
    flow_pct = flow_contrib / total * 100 if total > 0 else 0
    id_pct = other_contribs.get("id", 0) / total * 100 if total > 0 else 0
    print(f"\n  flow: {flow_pct:.1f}% {'OK (>= 30%)' if flow_pct >= 30 else 'WARNING: below 30% floor'}")
    print(f"  id:   {id_pct:.1f}% {'OK (< 20%)' if id_pct < 20 else 'WARNING: above 20% cap'}")


if __name__ == "__main__":
    main()
