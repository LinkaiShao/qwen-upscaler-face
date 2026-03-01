"""Checkpoint save/load utilities for Qwen upscaler training."""

import os

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import save_file, load_file


def _to_pipeline_format(peft_state):
    """Convert PEFT state dict to Diffusers pipeline-friendly key format."""
    out = {}
    for k, v in peft_state.items():
        nk = k
        nk = nk.replace("base_model.model.", "")
        nk = nk.replace(".lora_A.default.weight", ".lora_A.weight")
        nk = nk.replace(".lora_B.default.weight", ".lora_B.weight")
        out[f"transformer.{nk}"] = v
    return out


def _to_peft_format(state):
    """Normalize loaded state keys to PEFT key format expected by set_peft_model_state_dict."""
    out = {}
    for k, v in state.items():
        nk = k
        # Accept both training/internal and pipeline-export formats.
        if nk.startswith("transformer."):
            nk = nk[len("transformer."):]
        if not nk.startswith("base_model.model."):
            nk = f"base_model.model.{nk}"
        # Keep lora_A/lora_B key suffix as-is. set_peft_model_state_dict will
        # route them under the active adapter name internally.
        out[nk] = v
    return out


def save_checkpoint(transformer, optimizer, scheduler, epoch, global_step, path):
    """Save LoRA weights + training state."""
    os.makedirs(path, exist_ok=True)

    # LoRA weights (PEFT-native, for training resume / custom inference).
    lora_state = get_peft_model_state_dict(transformer)
    save_file(lora_state, os.path.join(path, "lora_weights.safetensors"))
    # Optional pipeline-export variant for Diffusers pipeline.load_lora_weights.
    save_file(_to_pipeline_format(lora_state), os.path.join(path, "lora_weights_pipeline.safetensors"))

    # Training state
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
    }, os.path.join(path, "training_state.pt"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))


def save_final(transformer, path):
    """Save final LoRA weights only (no optimizer state)."""
    os.makedirs(path, exist_ok=True)
    lora_state = get_peft_model_state_dict(transformer)
    save_file(lora_state, os.path.join(path, "lora_weights.safetensors"))
    save_file(_to_pipeline_format(lora_state), os.path.join(path, "lora_weights_pipeline.safetensors"))


def load_resume_state(resume_from):
    """Load training state from checkpoint dir. Returns dict or None."""
    state_path = os.path.join(resume_from, "training_state.pt")
    if os.path.exists(state_path):
        return torch.load(state_path, weights_only=False)
    return None


def load_lora_weights(transformer, resume_from):
    """Load LoRA weights from checkpoint dir."""
    lora_path = os.path.join(resume_from, "lora_weights.safetensors")
    if os.path.exists(lora_path):
        lora_state = load_file(lora_path)
        peft_state = _to_peft_format(lora_state)
        set_peft_model_state_dict(transformer, peft_state)
        return True
    # Backward compatibility if only pipeline export exists.
    pipe_path = os.path.join(resume_from, "lora_weights_pipeline.safetensors")
    if os.path.exists(pipe_path):
        lora_state = load_file(pipe_path)
        peft_state = _to_peft_format(lora_state)
        set_peft_model_state_dict(transformer, peft_state)
        return True
    return False


def load_optimizer_scheduler(optimizer, scheduler, resume_from):
    """Load optimizer and scheduler state from checkpoint dir."""
    opt_path = os.path.join(resume_from, "optimizer.pt")
    sched_path = os.path.join(resume_from, "scheduler.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, weights_only=False))
    if os.path.exists(sched_path):
        scheduler.load_state_dict(torch.load(sched_path, weights_only=False))
