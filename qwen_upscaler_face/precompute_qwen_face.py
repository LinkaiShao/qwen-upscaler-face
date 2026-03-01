#!/usr/bin/env python3
"""Pre-compute VAE latents, text embeddings, and face data for Qwen upscaler training."""

from qwen_upscaler_face.config import parse_args
from qwen_upscaler_face.precompute import (
    precompute_vae_latents,
    precompute_face_data,
    precompute_text_embeddings,
)


def main():
    args = parse_args()

    print("=" * 60)
    print("Qwen Upscaler Face — Pre-computation")
    print(f"  Data:         {args.data_parquet}")
    print(f"  Latent cache: {args.latent_cache_dir}")
    print(f"  Face cache:   {args.face_cache_dir}")
    print(f"  Target size:  {args.target_width}x{args.target_height}")
    print("=" * 60)

    # Step 1: VAE encode HR + LR images
    print("\n[1/3] VAE latent encoding...")
    precompute_vae_latents(args)

    # Step 2: Face detection + identity embeddings
    print("\n[2/3] Face detection + embedding...")
    precompute_face_data(args)

    # Step 3: Text encoding with condition images
    print("\n[3/3] Text encoding...")
    precompute_text_embeddings(args)

    print("\nAll pre-computation complete.")


if __name__ == "__main__":
    main()
