"""Configuration for Qwen upscaler with face loss."""

import math
from dataclasses import dataclass

BASE = "/home/link/Desktop/Code/fashion gen testing"


@dataclass
class Args:
    pretrained_model: str = f"{BASE}/Qwen-Image-Edit-2511"
    data_parquet: str = f"{BASE}/vton upscaling/viton_hd_upscale_data.parquet"
    latent_cache_dir: str = f"{BASE}/vton upscaling/qwen_latent_cache"
    face_cache_dir: str = f"{BASE}/vton upscaling/face_cache"
    output_dir: str = f"{BASE}/qwen_upscale_face_out"

    target_width: int = 768
    target_height: int = 1024

    # LoRA
    rank: int = 16
    alpha: int = 16
    init_lora_weights: str = "gaussian"

    # Training
    lr: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 4
    num_epochs: int = 3
    face_weight: float = 5.0
    lambda_id: float = 0.5
    lambda_anchor: float = 0.1
    prompt: str = "upscale this image to high resolution, preserve all details exactly"

    # Lighting lock
    lock_lighting: bool = True
    lighting_y_blend: float = 0.80
    lighting_blur_radius: float = 7.0

    # Optimizer
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Other
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    checkpoint_every: int = 500
    validate_every: int = 250
    seed: int = 42
    logging_steps: int = 10

    # Resume
    resume_from: str = ""


def calculate_dimensions(target_area: int, ratio: float):
    """Calculate width/height for a target area preserving aspect ratio (from Qwen pipeline)."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def parse_args() -> Args:
    import argparse
    parser = argparse.ArgumentParser(description="Train Qwen Upscaler with Face Loss")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--face_weight", type=float, default=None)
    parser.add_argument("--lambda_id", type=float, default=None)
    parser.add_argument("--lambda_anchor", type=float, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--validate_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--lock_lighting", action="store_true", help="Enable lighting lock for face loss/validation")
    parser.add_argument("--no_lock_lighting", action="store_true", help="Disable lighting lock for face loss/validation")
    parser.add_argument("--lighting_y_blend", type=float, default=None)
    parser.add_argument("--lighting_blur_radius", type=float, default=None)

    cli_args = parser.parse_args()
    args = Args()

    for key, value in vars(cli_args).items():
        if value is not None and hasattr(args, key):
            setattr(args, key, value)

    # Resolve mutually exclusive lighting lock toggles.
    if cli_args.lock_lighting and cli_args.no_lock_lighting:
        raise ValueError("Use only one of --lock_lighting or --no_lock_lighting.")
    if cli_args.lock_lighting:
        args.lock_lighting = True
    if cli_args.no_lock_lighting:
        args.lock_lighting = False

    return args
