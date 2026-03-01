"""Configuration for GAN-augmented Qwen upscaler training."""

from dataclasses import dataclass

BASE = "/home/link/Desktop/Code/fashion gen testing"
STEP_DIR = f"{BASE}/qwen_upscale_face_out/qwen_face_20260224_001227/step-6500"


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

    # Starting checkpoints
    lora_ckpt: str = STEP_DIR  # LoRA checkpoint to resume from
    disc_ckpt: str = f"{STEP_DIR}/patch_gan_disc_clothing/disc_final.pt"

    # Training — epoch-based (total_steps computed from num_epochs)
    lr: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 4  # effective batch size 4 (same as original training)
    num_epochs: int = 2

    # --- Phase boundaries ---
    # Phase 0 (0 to phase0_end): top-k only, no GAN, D frozen
    # Phase 1 (phase0_end to phase1_end): GAN ramps in, D updates every 2 steps
    # Phase 2 (phase1_end+): tight mining, GAN stable, D every 4 steps
    phase0_end: int = 500
    phase1_end: int = 2000

    # Loss weights
    face_weight: float = 5.0
    lambda_id: float = 0.15  # capped: face already trained, keep id < 20% of total
    lambda_anchor: float = 0.1
    lambda_gan: float = 0.002  # max GAN weight (reached at end of phase 1)
    lambda_topk: float = 1.0  # top-k clothing MSE penalty
    min_flow_frac: float = 0.30  # hard floor: flow is always >= 30% of total loss

    # Top-k mining schedule
    topk_ratio_phase0: float = 0.2   # phase 0: worst 20% of clothing pixels
    topk_ratio_phase1: float = 0.2   # phase 1 start (anneals to 0.1)
    topk_ratio_phase2: float = 0.1   # phase 2: worst 10%

    # ID loss gating
    id_every_n: int = 4              # compute id loss every N steps
    id_cos_threshold: float = 0.95   # skip id if running cos_sim > this

    # D update frequency
    d_update_every_p1: int = 2       # phase 1: D updates every 2 G steps
    d_update_every_p2: int = 4       # phase 2: D updates every 4 G steps

    prompt: str = "upscale this image to high resolution, preserve all details exactly"

    # Lighting lock
    lock_lighting: bool = True
    lighting_y_blend: float = 0.80
    lighting_blur_radius: float = 7.0

    # Generator optimizer
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05  # shorter warmup since fine-tuning

    # Discriminator optimizer
    disc_lr: float = 2e-5  # lower than G LR for stability

    # Other
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    checkpoint_every: int = 500
    validate_every: int = 250
    seed: int = 42
    logging_steps: int = 10

    # Resume
    resume_from: str = ""


def parse_args() -> Args:
    import argparse
    parser = argparse.ArgumentParser(description="Train Qwen Upscaler with GAN Loss")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--phase0_end", type=int, default=None)
    parser.add_argument("--phase1_end", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--face_weight", type=float, default=None)
    parser.add_argument("--lambda_id", type=float, default=None)
    parser.add_argument("--lambda_anchor", type=float, default=None)
    parser.add_argument("--lambda_gan", type=float, default=None)
    parser.add_argument("--lambda_topk", type=float, default=None)
    parser.add_argument("--min_flow_frac", type=float, default=None)
    parser.add_argument("--topk_ratio_phase0", type=float, default=None)
    parser.add_argument("--topk_ratio_phase1", type=float, default=None)
    parser.add_argument("--topk_ratio_phase2", type=float, default=None)
    parser.add_argument("--id_every_n", type=int, default=None)
    parser.add_argument("--id_cos_threshold", type=float, default=None)
    parser.add_argument("--d_update_every_p1", type=int, default=None)
    parser.add_argument("--d_update_every_p2", type=int, default=None)
    parser.add_argument("--disc_lr", type=float, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--validate_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--disc_ckpt", type=str, default=None)
    parser.add_argument("--lock_lighting", action="store_true")
    parser.add_argument("--no_lock_lighting", action="store_true")

    cli_args = parser.parse_args()
    args = Args()

    for key, value in vars(cli_args).items():
        if value is not None and hasattr(args, key):
            setattr(args, key, value)

    if cli_args.lock_lighting and cli_args.no_lock_lighting:
        raise ValueError("Use only one of --lock_lighting or --no_lock_lighting.")
    if cli_args.lock_lighting:
        args.lock_lighting = True
    if cli_args.no_lock_lighting:
        args.lock_lighting = False

    return args
