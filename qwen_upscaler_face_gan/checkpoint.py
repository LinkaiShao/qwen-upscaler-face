"""Checkpoint utilities for GAN-augmented training.

Extends base checkpoint functions with discriminator state save/load.
"""

import os

import torch

# Re-export base checkpoint functions
from qwen_upscaler_face.checkpoint import (  # noqa: F401
    save_final,
    load_resume_state,
    load_lora_weights,
    load_optimizer_scheduler,
)
from qwen_upscaler_face.checkpoint import save_checkpoint as _save_checkpoint_base


def save_checkpoint(transformer, g_optimizer, scheduler, epoch, global_step, path,
                    disc=None, d_optimizer=None):
    """Save LoRA weights + training state + discriminator state."""
    # Save LoRA + G optimizer + scheduler via base function
    _save_checkpoint_base(transformer, g_optimizer, scheduler, epoch, global_step, path)

    # Save discriminator state
    if disc is not None:
        disc_state = {"model": disc.state_dict()}
        if d_optimizer is not None:
            disc_state["optimizer"] = d_optimizer.state_dict()
        torch.save(disc_state, os.path.join(path, "disc_state.pt"))


def load_disc_state(disc, path, d_optimizer=None, device="cuda"):
    """Load discriminator weights and optimizer state from checkpoint.

    Args:
        disc: NLayerDiscriminator instance.
        path: Checkpoint directory.
        d_optimizer: Optional optimizer to restore state.
        device: Target device.

    Returns:
        True if loaded successfully, False otherwise.
    """
    disc_path = os.path.join(path, "disc_state.pt")
    if not os.path.exists(disc_path):
        return False

    state = torch.load(disc_path, map_location=device, weights_only=False)
    disc.load_state_dict(state["model"])
    if d_optimizer is not None and "optimizer" in state:
        d_optimizer.load_state_dict(state["optimizer"])
    return True
