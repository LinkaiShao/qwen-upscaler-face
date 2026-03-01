#!/usr/bin/env python3
"""PatchGAN discriminator: precompute pixel cache + train.

Two modes:
  --precompute   Run frozen generator on all train+val samples, save pred/hr/lr PNGs.
  (default)      Train discriminator from precomputed PNGs.

Example:
  # Step 1: precompute pixel cache (~1-2 hours)
  python -m qwen_upscaler_face.patch_gan.train_discriminator --precompute \\
      --checkpoint /path/to/lora_checkpoint

  # Step 2: train discriminator (~20 min)
  python -m qwen_upscaler_face.patch_gan.train_discriminator \\
      --disc_output_dir /path/to/disc_checkpoints
"""

import argparse
import sys

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/vton upscaling")

from qwen_upscaler_face.patch_gan.precompute import precompute
from qwen_upscaler_face.patch_gan.training import train

BASE = "/home/link/Desktop/Code/fashion gen testing"
STEP_DIR = f"{BASE}/qwen_upscale_face_out/qwen_face_20260224_001227/step-6500"
DEFAULT_CACHE_DIR = f"{STEP_DIR}/pred_pixel_cache"
DEFAULT_DISC_OUT = f"{STEP_DIR}/patch_gan_disc_clothing_v2"


def main():
    parser = argparse.ArgumentParser(description="PatchGAN discriminator: precompute + train")

    # Mode
    parser.add_argument("--precompute", action="store_true", help="Precompute pixel cache from frozen generator")

    # Shared
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Directory for precomputed pred/hr/lr PNGs")

    # Precompute args
    parser.add_argument("--checkpoint", type=str, default="",
                        help="LoRA checkpoint dir (required for --precompute)")
    parser.add_argument("--rank", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="Override LoRA alpha")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit precompute to N random samples")
    parser.add_argument("--splits", type=str, nargs="+", default=None,
                        help="Which splits to precompute (default: all)")

    # Training args
    parser.add_argument("--disc_output_dir", type=str, default=DEFAULT_DISC_OUT,
                        help="Output dir for discriminator checkpoints")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--validate_every", type=int, default=200)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default="",
                        help="Resume training from discriminator checkpoint .pt file")
    parser.add_argument("--clothing_only", action="store_true",
                        help="Train/eval only on clothing-mask pixels (recommended)")
    parser.add_argument("--full_image", action="store_true",
                        help="Disable clothing-mask-only behavior")
    parser.add_argument("--n_val", type=int, default=None,
                        help="Custom val set size (ignores parquet splits, default: use _split.txt)")

    args_cli = parser.parse_args()

    # Default to clothing-only unless user explicitly requests full-image behavior.
    if not args_cli.full_image:
        args_cli.clothing_only = True

    # Override output dir for clothing-only mode if using default.
    if args_cli.clothing_only and args_cli.disc_output_dir == DEFAULT_DISC_OUT:
        args_cli.disc_output_dir = f"{STEP_DIR}/patch_gan_disc_clothing"

    if args_cli.precompute:
        if not args_cli.checkpoint:
            raise ValueError("--checkpoint is required for --precompute mode")
        precompute(args_cli)
    else:
        train(args_cli)


if __name__ == "__main__":
    main()
