#!/usr/bin/env python3
"""Train Qwen upscaler with GAN loss (three-phase adversarial training).

Phase 0 (0 to phase0_end):
    Top-k clothing MSE only. No GAN. D frozen/off.
    ID loss periodic (every id_every_n steps) + threshold gated.

Phase 1 (phase0_end to phase1_end):
    GAN ramps from 0 to lambda_gan. D updates every d_update_every_p1 steps.
    Top-k ratio at 0.2, anneals toward 0.1.

Phase 2 (phase1_end+):
    GAN weight stable. D updates every d_update_every_p2 steps.
    Top-k tightens to 0.05-0.1. ID as guardrail only.

Usage:
    python -m qwen_upscaler_face_gan
"""

import os
import sys
import traceback
from datetime import datetime

sys.path.insert(0, "/home/link/Desktop/Code/fashion gen testing/vton upscaling")

import torch
from tqdm.auto import tqdm

from qwen_upscaler_face_gan.config import parse_args
from qwen_upscaler_face_gan.models import load_transformer, load_vae, load_discriminator
from qwen_upscaler_face_gan.dataset import create_dataloaders
from qwen_upscaler_face_gan.training import train_step, disc_step
from qwen_upscaler_face_gan.checkpoint import (
    save_checkpoint, save_final, load_resume_state,
    load_lora_weights, load_disc_state,
)
from qwen_upscaler_face.validation import validate
from qwen_upscaler_face.logging_utils import setup_logging, mem_stats


def _get_phase(global_step, args):
    """Determine current phase from step count."""
    if global_step < args.phase0_end:
        return 0
    elif global_step < args.phase1_end:
        return 1
    else:
        return 2


def _get_lambda_gan(global_step, args):
    """GAN weight: 0 in phase 0, linear ramp in phase 1, stable in phase 2."""
    if global_step < args.phase0_end:
        return 0.0
    elif global_step < args.phase1_end:
        # Linear ramp from 0 to lambda_gan over phase 1
        progress = (global_step - args.phase0_end) / max(1, args.phase1_end - args.phase0_end)
        return args.lambda_gan * progress
    else:
        return args.lambda_gan


def _get_topk_ratio(global_step, args):
    """Top-k ratio schedule: 0.2 in phases 0-1, anneals to final in phase 2."""
    if global_step < args.phase0_end:
        return args.topk_ratio_phase0
    elif global_step < args.phase1_end:
        # Anneal from phase1 ratio toward phase2 ratio
        progress = (global_step - args.phase0_end) / max(1, args.phase1_end - args.phase0_end)
        return args.topk_ratio_phase1 + (args.topk_ratio_phase2 - args.topk_ratio_phase1) * progress
    else:
        return args.topk_ratio_phase2


def _should_compute_id(global_step, phase, args, running_cos_sim):
    """ID loss: periodic every N steps, gated by cos_sim threshold."""
    if phase == 0 or phase == 1:
        # Periodic: every id_every_n steps
        if global_step % args.id_every_n != 0:
            return False
        # Threshold gate: skip if face is already good
        if running_cos_sim > args.id_cos_threshold:
            return False
        return True
    else:
        # Phase 2: id as guardrail only (same gating, less frequent)
        if global_step % (args.id_every_n * 2) != 0:
            return False
        if running_cos_sim > args.id_cos_threshold:
            return False
        return True


def _get_d_update_every(phase, args):
    """D update frequency per phase."""
    if phase <= 0:
        return 0  # no D updates in phase 0
    elif phase == 1:
        return args.d_update_every_p1
    else:
        return args.d_update_every_p2


def main():
    args = parse_args()
    device = torch.device("cuda")
    weight_dtype = torch.bfloat16

    if not args.disc_ckpt or not os.path.exists(args.disc_ckpt):
        raise FileNotFoundError(f"Discriminator checkpoint not found: {args.disc_ckpt}")

    # Resume?
    resume_state = None
    if args.resume_from:
        resume_state = load_resume_state(args.resume_from)

    # Output directory
    if resume_state is not None:
        args.output_dir = os.path.dirname(args.resume_from.rstrip("/"))
    else:
        run_name = f"qwen_face_gan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    log, log_path = setup_logging(args.output_dir)
    log.info("=" * 60)
    if resume_state:
        log.info(f"RESUMING from step {resume_state['global_step']}")
    else:
        log.info(f"GAN Upscaler Training (3-phase) | Output: {args.output_dir}")
    log.info(f"Log: {log_path}")
    log.info("=" * 60)

    torch.manual_seed(args.seed)

    # Load transformer + LoRA
    transformer = load_transformer(args, device=device)
    log.info(f"[mem] after transformer: {mem_stats()}")

    # Load starting LoRA weights
    if args.resume_from:
        if not load_lora_weights(transformer, args.resume_from):
            raise FileNotFoundError(
                f"Resume checkpoint does not contain LoRA weights: {args.resume_from}"
            )
        log.info("Loaded LoRA weights from resume checkpoint")
    elif args.lora_ckpt:
        if not load_lora_weights(transformer, args.lora_ckpt):
            raise FileNotFoundError(
                f"Starting LoRA checkpoint does not contain LoRA weights: {args.lora_ckpt}"
            )
        log.info(f"Loaded starting LoRA weights from {args.lora_ckpt}")
        base_state = load_resume_state(args.lora_ckpt)
        if base_state and "global_step" in base_state:
            log.info(f"Base LoRA source global_step={base_state['global_step']}")
    else:
        raise ValueError("Provide --lora_ckpt (or --resume_from) to fine-tune an existing Qwen Face model.")

    # Load VAE
    vae = load_vae(args, device=device)
    log.info(f"[mem] after vae: {mem_stats()}")

    # Load frozen face encoder
    from facenet_pytorch import InceptionResnetV1
    face_encoder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    face_encoder.requires_grad_(False)
    log.info(f"[mem] after face_encoder: {mem_stats()}")

    # Load discriminator
    disc = load_discriminator(args.disc_ckpt, device=device)
    n_disc_params = sum(p.numel() for p in disc.parameters())
    log.info(f"Discriminator: {n_disc_params/1e6:.2f}M params from {args.disc_ckpt}")
    log.info(f"[mem] after disc: {mem_stats()}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    log.info(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # Compute total steps from epochs
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs

    # Generator optimizer
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    g_optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # LR scheduler: warmup + cosine decay
    num_warmup = int(args.warmup_ratio * total_steps)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        g_optimizer, start_factor=1e-3, total_iters=max(1, num_warmup),
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer, T_max=max(1, total_steps - num_warmup), eta_min=0,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        g_optimizer, [warmup_sched, cosine_sched], milestones=[num_warmup],
    )

    # D optimizer — created at phase 1 transition
    d_optimizer = None

    # Resume state
    global_step = 0
    start_epoch = 0
    if resume_state is not None:
        global_step = resume_state["global_step"]
        opt_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            g_optimizer.load_state_dict(torch.load(opt_path, weights_only=False))
        remaining_steps = total_steps - global_step
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            g_optimizer, T_max=max(1, remaining_steps), eta_min=0,
        )
        # If resuming in phase 1+, set up D optimizer
        if global_step >= args.phase0_end:
            disc.train()
            disc.requires_grad_(True)
            d_optimizer = torch.optim.Adam(
                disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.999),
            )
            if load_disc_state(disc, args.resume_from, d_optimizer, device):
                log.info("Resumed discriminator state + optimizer")
            else:
                log.info("No disc_state.pt in resume checkpoint; using provided disc_ckpt weights")
        log.info(f"Resumed at step {global_step}, {remaining_steps} steps remaining")

    # Log config
    log.info(f"Epochs: {args.num_epochs} | Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    log.info(f"Phase 0: 0-{args.phase0_end} | Phase 1: {args.phase0_end}-{args.phase1_end} | Phase 2: {args.phase1_end}+")
    log.info(f"LoRA rank={args.rank} alpha={args.alpha} | G lr={args.lr} | D lr={args.disc_lr}")
    log.info(f"lambda_gan={args.lambda_gan} | lambda_id={args.lambda_id} | lambda_anchor={args.lambda_anchor} | lambda_topk={args.lambda_topk}")
    log.info(f"topk: p0={args.topk_ratio_phase0} p1={args.topk_ratio_phase1} p2={args.topk_ratio_phase2}")
    log.info(f"id: every {args.id_every_n} steps, gate at cos>{args.id_cos_threshold}")
    log.info(f"D update: p1 every {args.d_update_every_p1} | p2 every {args.d_update_every_p2}")
    log.info(f"Batch={args.batch_size} | Grad accum={args.grad_accum} | Effective batch={args.batch_size * args.grad_accum}")

    # Tracking
    step_hist, flow_hist, gan_hist = [], [], []
    val_steps_hist, val_mse_hist = [], []
    running_cos_sim = 0.98  # start optimistic
    cos_ema_alpha = 0.1

    progress = tqdm(total=total_steps, initial=global_step)
    micro_step = 0
    transformer.train()
    prev_phase = _get_phase(global_step, args)

    try:
        for epoch in range(start_epoch, args.num_epochs):
            log.info(f"--- Epoch {epoch + 1}/{args.num_epochs} ---")

            for batch in train_loader:
                if global_step >= total_steps:
                    break
                micro_step += 1
                phase = _get_phase(global_step, args)

                # === Phase transitions ===
                if phase != prev_phase:
                    if phase == 1 and d_optimizer is None:
                        disc.train()
                        disc.requires_grad_(True)
                        d_optimizer = torch.optim.Adam(
                            disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.999),
                        )
                        log.info(f"=== PHASE 1 at step {global_step}: D created, GAN ramping, lr={args.disc_lr} ===")
                    elif phase == 2:
                        log.info(f"=== PHASE 2 at step {global_step}: GAN stable, tight mining ===")
                    prev_phase = phase

                # Compute schedule values
                lambda_gan_eff = _get_lambda_gan(global_step, args)
                topk_ratio = _get_topk_ratio(global_step, args)
                compute_id = _should_compute_id(global_step, phase, args, running_cos_sim)

                # === Generator update ===
                disc.requires_grad_(False)
                loss_dict = train_step(
                    batch, transformer, vae, face_encoder, disc,
                    args, device, weight_dtype,
                    phase=phase,
                    global_step=global_step,
                    compute_id=compute_id,
                    lambda_gan_eff=lambda_gan_eff,
                    topk_ratio=topk_ratio,
                )
                g_loss = loss_dict["loss"] / args.grad_accum
                g_loss.backward()

                # Update running cos_sim EMA
                if compute_id and loss_dict["cos_sim"] < 1.0:
                    running_cos_sim = (1 - cos_ema_alpha) * running_cos_sim + cos_ema_alpha * loss_dict["cos_sim"]

                # === Discriminator update (phase 1+) ===
                d_loss_val = 0.0
                p_real_val = 0.0
                p_fake_val = 0.0
                gap_p_val = 0.0
                d_update_every = _get_d_update_every(phase, args)
                update_d = phase >= 1 and d_update_every > 0 and (global_step + 1) % d_update_every == 0
                if phase >= 1:
                    disc.requires_grad_(True)
                    d_dict = disc_step(
                        batch, transformer, vae, disc,
                        args, device, weight_dtype,
                    )
                    d_loss_val = d_dict["d_loss"].item()
                    p_real_val = d_dict["p_real"]
                    p_fake_val = d_dict["p_fake"]
                    gap_p_val = d_dict["gap_p"]
                    if update_d:
                        d_loss = d_dict["d_loss"] / args.grad_accum
                        d_loss.backward()

                # === Optimizer steps ===
                if micro_step % args.grad_accum == 0:
                    # G step
                    torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                    g_optimizer.step()
                    scheduler.step()
                    g_optimizer.zero_grad()

                    # D step
                    if update_d and d_optimizer is not None:
                        d_optimizer.step()
                        d_optimizer.zero_grad()

                    global_step += 1
                    progress.update(1)

                    # Logging
                    if global_step % args.logging_steps == 0:
                        step_hist.append(global_step)
                        flow_hist.append(loss_dict["flow"])
                        gan_hist.append(loss_dict["gan"])

                        phase_str = f"P{phase}"
                        d_info = ""
                        if phase >= 1:
                            d_info = (f" d_loss={d_loss_val:.4f}"
                                      f" p_real={p_real_val:.3f}"
                                      f" p_fake={p_fake_val:.3f}"
                                      f" gap={gap_p_val:.3f}")

                        id_flag = "Y" if compute_id else "n"
                        progress.set_postfix(
                            flow=loss_dict["flow"], gan=loss_dict["gan"],
                            phase=phase, step=global_step,
                        )
                        log.info(
                            f"[{phase_str}] step {global_step}/{total_steps} "
                            f"flow={loss_dict['flow']:.4f} id={loss_dict['id']:.4f}({id_flag}) "
                            f"anchor={loss_dict['anchor']:.4f} gan={loss_dict['gan']:.4f} "
                            f"topk={loss_dict['topk_mse']:.4f} "
                            f"lgan={lambda_gan_eff:.4f} rho={topk_ratio:.2f} "
                            f"fscale={loss_dict.get('flow_scale', 1.0):.2f} "
                            f"cos={running_cos_sim:.4f}"
                            f"{d_info} "
                            f"lr={scheduler.get_last_lr()[0]:.2e} | {mem_stats()}"
                        )

                    # Validation
                    if global_step % args.validate_every == 0:
                        log.info("Running validation...")
                        val_result = validate(
                            transformer, val_loader, vae, face_encoder, args, device, weight_dtype,
                        )
                        val_steps_hist.append(global_step)
                        val_mse_hist.append(val_result["mse"])
                        log.info(
                            f"[Val] step {global_step}: mse={val_result['mse']:.4f} "
                            f"psnr={val_result['psnr']:.2f} ssim={val_result['ssim']:.4f} "
                            f"face_cos={val_result['face_cos_sim']:.4f}"
                        )
                        transformer.train()

                    # Checkpoint
                    if global_step % args.checkpoint_every == 0:
                        ckpt_path = os.path.join(args.output_dir, f"step-{global_step}")
                        save_checkpoint(
                            transformer, g_optimizer, scheduler, epoch, global_step, ckpt_path,
                            disc=disc, d_optimizer=d_optimizer,
                        )
                        log.info(f"Saved checkpoint -> {ckpt_path}")

            if global_step >= total_steps:
                break

            # End-of-epoch checkpoint
            epoch_path = os.path.join(args.output_dir, f"epoch-{epoch}")
            save_checkpoint(
                transformer, g_optimizer, scheduler, epoch, global_step, epoch_path,
                disc=disc, d_optimizer=d_optimizer,
            )
            log.info(f"Epoch {epoch + 1} done. Saved -> {epoch_path}")

    except Exception as e:
        log.error(f"CRASH at epoch={epoch} step={global_step}")
        log.error(f"{e}\n{traceback.format_exc()}")
        log.error(f"[mem] {mem_stats()}")

        try:
            crash_path = os.path.join(args.output_dir, f"crash-epoch{epoch}-step{global_step}")
            save_checkpoint(
                transformer, g_optimizer, scheduler, epoch, global_step, crash_path,
                disc=disc, d_optimizer=d_optimizer,
            )
            log.error(f"Emergency checkpoint -> {crash_path}")
        except Exception:
            log.error("Failed to save emergency checkpoint")
        raise

    # Final save
    final_path = os.path.join(args.output_dir, "final")
    save_checkpoint(
        transformer, g_optimizer, scheduler, 0, global_step, final_path,
        disc=disc, d_optimizer=d_optimizer,
    )
    save_final(transformer, final_path)
    log.info(f"Training complete! Final checkpoint -> {final_path}")

    # Final validation
    log.info("Running final validation...")
    val_result = validate(transformer, val_loader, vae, face_encoder, args, device, weight_dtype)
    log.info(
        f"[Final Val] mse={val_result['mse']:.4f} "
        f"psnr={val_result['psnr']:.2f} ssim={val_result['ssim']:.4f} "
        f"face_cos={val_result['face_cos_sim']:.4f}"
    )

    progress.close()
    log.info(f"[mem] {mem_stats()}")


if __name__ == "__main__":
    main()
