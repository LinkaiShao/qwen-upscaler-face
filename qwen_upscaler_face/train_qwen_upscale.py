#!/usr/bin/env python3
"""Train Qwen-Image-Edit LoRA upscaler with face-weighted flow matching loss."""

import os
import traceback
from datetime import datetime

import torch
from tqdm.auto import tqdm

from qwen_upscaler_face.config import parse_args
from qwen_upscaler_face.models import load_transformer, load_vae
from qwen_upscaler_face.dataset import create_dataloaders
from qwen_upscaler_face.training import train_step
from qwen_upscaler_face.validation import validate
from qwen_upscaler_face.checkpoint import (
    save_checkpoint, save_final, load_resume_state,
    load_lora_weights,
)
from qwen_upscaler_face.plotting import save_training_plots
from qwen_upscaler_face.logging_utils import setup_logging, mem_stats


def main():
    args = parse_args()
    device = torch.device("cuda")
    weight_dtype = torch.bfloat16

    # Resume?
    resume_state = None
    if args.resume_from:
        resume_state = load_resume_state(args.resume_from)

    # Output directory
    if resume_state is not None:
        args.output_dir = os.path.dirname(args.resume_from.rstrip("/"))
    else:
        run_name = f"qwen_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    log, log_path = setup_logging(args.output_dir)
    log.info("=" * 60)
    if resume_state:
        log.info(f"RESUMING from epoch {resume_state['epoch']} step {resume_state['global_step']}")
    else:
        log.info(f"Qwen Face Upscaler Training | Output: {args.output_dir}")
    log.info(f"Log: {log_path}")
    log.info("=" * 60)

    torch.manual_seed(args.seed)

    # Load transformer + LoRA
    transformer = load_transformer(args, device=device)
    log.info(f"[mem] after transformer: {mem_stats()}")

    # Load LoRA weights if resuming
    if args.resume_from:
        if load_lora_weights(transformer, args.resume_from):
            log.info("Loaded LoRA weights from checkpoint")

    # Load VAE (on GPU — needed for identity loss during training + validation)
    vae = load_vae(args, device=device)
    log.info(f"[mem] after vae: {mem_stats()}")

    # Load frozen face encoder for identity loss
    from facenet_pytorch import InceptionResnetV1
    face_encoder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    face_encoder.requires_grad_(False)
    log.info(f"[mem] after face_encoder: {mem_stats()}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    log.info(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # LR scheduler: warmup + cosine decay
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    num_warmup = int(args.warmup_ratio * total_steps)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=num_warmup,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - num_warmup, eta_min=0,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[num_warmup],
    )

    # Resume optimizer/scheduler
    start_epoch = 0
    global_step = 0
    if resume_state is not None:
        start_epoch = resume_state["epoch"] + 1
        global_step = resume_state["global_step"]
        # Load optimizer state (Adam momentum/variance buffers).
        # Scheduler is NOT loaded — we build a fresh cosine schedule for the
        # remaining epochs so LR doesn't start near zero.
        opt_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, weights_only=False))
        remaining_steps = total_steps - global_step
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, remaining_steps), eta_min=0,
        )
        log.info(f"Resuming from epoch {start_epoch}, global_step {global_step}")
        log.info(f"Fresh cosine schedule for {remaining_steps} remaining steps")

    log.info(f"Epochs: {args.num_epochs} (from {start_epoch}) | Steps/epoch: {steps_per_epoch} | Total: {total_steps}")
    log.info(f"LoRA rank={args.rank} alpha={args.alpha} | lr={args.lr} | face_weight={args.face_weight} | lambda_id={args.lambda_id}")
    log.info(f"Batch={args.batch_size} | Grad accum={args.grad_accum} | Warmup={num_warmup}")

    # Metrics
    step_hist, flow_hist, id_hist = [], [], []
    val_steps_hist, val_mse_hist = [], []

    progress = tqdm(total=total_steps, initial=global_step)
    micro_step = 0
    transformer.train()

    try:
        for epoch in range(start_epoch, args.num_epochs):
            log.info(f"--- Epoch {epoch + 1}/{args.num_epochs} ---")

            for batch in train_loader:
                micro_step += 1

                loss, flow_val, id_val = train_step(
                    batch, transformer, vae, face_encoder, args, device, weight_dtype,
                )

                # Scale loss for gradient accumulation
                scaled_loss = loss / args.grad_accum
                scaled_loss.backward()

                if micro_step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in transformer.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress.update(1)

                    if global_step % args.logging_steps == 0:
                        step_hist.append(global_step)
                        flow_hist.append(flow_val)
                        id_hist.append(id_val)
                        progress.set_postfix(
                            flow=flow_val, id=id_val, epoch=epoch + 1,
                        )
                        log.info(
                            f"step {global_step}/{total_steps} (epoch={epoch+1}) "
                            f"flow={flow_val:.4f} id={id_val:.4f} "
                            f"lr={scheduler.get_last_lr()[0]:.2e} | {mem_stats()}"
                        )

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

                    if global_step % args.checkpoint_every == 0:
                        ckpt_path = os.path.join(args.output_dir, f"step-{global_step}")
                        save_checkpoint(transformer, optimizer, scheduler, epoch, global_step, ckpt_path)
                        log.info(f"Saved checkpoint -> {ckpt_path}")

            # End-of-epoch checkpoint
            epoch_path = os.path.join(args.output_dir, f"epoch-{epoch}")
            save_checkpoint(transformer, optimizer, scheduler, epoch, global_step, epoch_path)
            log.info(f"Epoch {epoch + 1} done. Saved -> {epoch_path}")

            # Epoch validation
            log.info("Running epoch validation...")
            val_result = validate(transformer, val_loader, vae, face_encoder, args, device, weight_dtype)
            val_steps_hist.append(global_step)
            val_mse_hist.append(val_result["mse"])
            log.info(
                f"[Val] epoch {epoch+1}: mse={val_result['mse']:.4f} "
                f"psnr={val_result['psnr']:.2f} ssim={val_result['ssim']:.4f} "
                f"face_cos={val_result['face_cos_sim']:.4f}"
            )
            transformer.train()

            save_training_plots(step_hist, flow_hist, id_hist, val_steps_hist, val_mse_hist, args.output_dir)

    except Exception as e:
        log.error(f"CRASH at epoch={epoch} step={global_step}")
        log.error(f"{e}\n{traceback.format_exc()}")
        log.error(f"[mem] {mem_stats()}")

        try:
            crash_path = os.path.join(args.output_dir, f"crash-epoch{epoch}-step{global_step}")
            save_checkpoint(transformer, optimizer, scheduler, epoch, global_step, crash_path)
            log.error(f"Emergency checkpoint -> {crash_path}")
        except Exception:
            log.error("Failed to save emergency checkpoint")
        raise

    # Final save
    final_path = os.path.join(args.output_dir, "final")
    save_final(transformer, final_path)
    log.info(f"Training complete! Final model -> {final_path}")
    log.info(f"[mem] {mem_stats()}")

    progress.close()


if __name__ == "__main__":
    main()
