"""Model loading for GAN-augmented Qwen upscaler."""

import torch

# Re-export generator/VAE utilities from base package
from qwen_upscaler_face.models import (  # noqa: F401
    load_transformer,
    load_vae,
    load_text_encoder,
    encode_vae_image,
    pack_latents,
    unpack_latents,
    denormalize_latents,
    load_scheduler,
)

from .discriminator import NLayerDiscriminator


def load_discriminator(disc_ckpt, device="cuda"):
    """Load pre-trained PatchGAN discriminator.

    Args:
        disc_ckpt: Path to discriminator checkpoint (.pt file).
        device: Target device.

    Returns:
        Discriminator in eval mode with requires_grad=False.
    """
    disc = NLayerDiscriminator(input_nc=6, ndf=64, n_layers=3).to(device)

    state = torch.load(disc_ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        disc.load_state_dict(state["model"])
    elif isinstance(state, dict) and "model_state_dict" in state:
        disc.load_state_dict(state["model_state_dict"])
    else:
        disc.load_state_dict(state)

    disc.eval()
    disc.requires_grad_(False)
    return disc
