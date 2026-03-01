"""Model loading utilities for Qwen upscaler."""

import sys

# Use local diffusers dev version with Qwen support
sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/diffusers/src")

import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from peft import LoraConfig, get_peft_model


def load_transformer(args, device="cuda"):
    """Load Qwen transformer with LoRA."""
    print(f"Loading Qwen transformer from {args.pretrained_model}/transformer...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        init_lora_weights=args.init_lora_weights,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.to(device)
    return transformer


def load_vae(args, device="cuda"):
    """Load Qwen VAE."""
    print(f"Loading Qwen VAE from {args.pretrained_model}/vae...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def load_text_encoder(args, device="cuda"):
    """Load Qwen2.5-VL text encoder + processor."""
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

    print(f"Loading Qwen2.5-VL text encoder from {args.pretrained_model}/text_encoder...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )
    text_encoder.to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    print(f"Loading processor from {args.pretrained_model}/processor...")
    processor = Qwen2VLProcessor.from_pretrained(
        args.pretrained_model,
        subfolder="processor",
    )

    return text_encoder, processor


def encode_vae_image(vae, image_tensor):
    """Encode image tensor to normalized Qwen latent.

    Args:
        vae: Qwen VAE
        image_tensor: (B, 3, 1, H, W) tensor in [-1, 1] range (5D with temporal dim)

    Returns:
        Normalized latent (B, 16, H//8, W//8)
    """
    with torch.no_grad():
        raw_latent = vae.encode(image_tensor).latent_dist.mode()
        # Normalize with channel-wise mean/std
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, 16, 1, 1, 1)
            .to(raw_latent.device, raw_latent.dtype)
        )
        latents_std = (
            torch.tensor(vae.config.latents_std)
            .view(1, 16, 1, 1, 1)
            .to(raw_latent.device, raw_latent.dtype)
        )
        normalized = (raw_latent - latents_mean) / latents_std
        # Remove temporal dimension: (B, 16, 1, H//8, W//8) → (B, 16, H//8, W//8)
        if normalized.dim() == 5:
            normalized = normalized.squeeze(2)
        return normalized


def pack_latents(latents, B, C, H, W):
    """Pack latents for Qwen transformer (2x2 patchify).

    (B, C, H, W) → (B, (H//2)*(W//2), C*4)
    """
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(B, (H // 2) * (W // 2), C * 4)


def unpack_latents(latents, H, W):
    """Unpack latents from Qwen transformer format.

    (B, (H//2)*(W//2), C*4) → (B, C, 1, H, W) with temporal dim for VAE decode
    """
    B = latents.shape[0]
    C4 = latents.shape[2]
    C = C4 // 4
    latents = latents.reshape(B, H // 2, W // 2, C, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(B, C, H, W)
    # Add temporal dim for VAE decode
    return latents.unsqueeze(2)


def load_scheduler(args):
    """Load the FlowMatchEulerDiscreteScheduler from model config."""
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler",
    )


def denormalize_latents(latents, vae):
    """Reverse the normalization for VAE decoding.

    Args:
        latents: (B, 16, 1, H, W) normalized latents
        vae: Qwen VAE for config values

    Returns:
        Denormalized latents ready for vae.decode()
    """
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, 16, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(vae.config.latents_std)
        .view(1, 16, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    return latents / latents_std + latents_mean
