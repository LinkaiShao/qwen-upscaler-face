"""Discriminator training loop and validation."""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .discriminator import NLayerDiscriminator
from .dataset import DiscriminatorDataset
from .visualization import save_heatmap_viz


def _masked_bce_logits(logits, labels, bce_none, mask):
    """BCE on logits weighted by patch mask coverage."""
    # Resize pixel mask to discriminator patch grid using area averaging.
    patch_mask = F.interpolate(mask, size=logits.shape[-2:], mode="area")
    loss_map = bce_none(logits, labels)
    weighted = loss_map * patch_mask
    denom = patch_mask.sum().clamp_min(1e-6)
    return weighted.sum() / denom


def validate(disc, val_loader, bce, device):
    """Run validation, return loss and mean sigmoid metrics."""
    disc.eval()
    total_loss = 0.0
    sum_p_real = 0.0
    sum_p_fake = 0.0
    total_weight = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            pred = batch["pred"].to(device)
            clothing_mask = batch["clothing_mask"].to(device)

            real_input = torch.cat([lr, hr], dim=1)
            fake_input = torch.cat([lr, pred], dim=1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                real_logits = disc(real_input)
                fake_logits = disc(fake_input)

                real_labels = torch.full_like(real_logits, 0.9)
                fake_labels = torch.zeros_like(fake_logits)
                loss_real = _masked_bce_logits(real_logits, real_labels, bce, clothing_mask)
                loss_fake = _masked_bce_logits(fake_logits, fake_labels, bce, clothing_mask)
                loss = 0.5 * (loss_real + loss_fake)

            total_loss += loss.item()

            patch_mask = F.interpolate(clothing_mask, size=real_logits.shape[-2:], mode="area")
            w = patch_mask.sum().item()
            sum_p_real += (torch.sigmoid(real_logits.float()) * patch_mask).sum().item()
            sum_p_fake += (torch.sigmoid(fake_logits.float()) * patch_mask).sum().item()
            total_weight += w
            n_batches += 1

    p_real = sum_p_real / max(total_weight, 1e-6)
    p_fake = sum_p_fake / max(total_weight, 1e-6)
    return {
        "loss": total_loss / max(n_batches, 1),
        "p_real": p_real,
        "p_fake": p_fake,
        "gap_p": p_real - p_fake,
    }


def train(args_cli):
    """Train PatchGAN discriminator from precomputed PNGs."""
    cache_dir = args_cli.cache_dir
    output_dir = args_cli.disc_output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda")

    # Datasets
    clothing_only = getattr(args_cli, "clothing_only", False)
    n_val = getattr(args_cli, "n_val", None)
    train_ds = DiscriminatorDataset(cache_dir, split="train", augment=True,
                                    clothing_only=clothing_only, n_val=n_val)
    val_ds = DiscriminatorDataset(cache_dir, split="val", augment=False,
                                  clothing_only=clothing_only, n_val=n_val)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {cache_dir}. Run --precompute first.")

    train_loader = DataLoader(
        train_ds, batch_size=args_cli.batch_size, shuffle=True,
        num_workers=args_cli.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args_cli.batch_size, shuffle=False,
        num_workers=args_cli.num_workers, pin_memory=True,
    )

    # Model
    disc = NLayerDiscriminator(input_nc=6, ndf=64, n_layers=3).to(device)
    n_params = sum(p.numel() for p in disc.parameters())
    print(f"NLayerDiscriminator: {n_params/1e6:.2f}M params")

    # Optimizer
    optimizer = torch.optim.Adam(
        disc.parameters(), lr=2e-4, betas=(0.5, 0.999),
    )

    # Loss
    bce = nn.BCEWithLogitsLoss(reduction="none")

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Resume
    start_epoch = 0
    global_step = 0
    if args_cli.resume_from and os.path.exists(args_cli.resume_from):
        ckpt = torch.load(args_cli.resume_from, weights_only=False, map_location=device)
        disc.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {args_cli.resume_from} (epoch={start_epoch}, step={global_step})")

    print(f"\nTraining: {args_cli.epochs} epochs, batch_size={args_cli.batch_size}")
    print(f"  train: {len(train_ds)} samples, val: {len(val_ds)} samples")
    print(f"  steps/epoch: {len(train_loader)}")

    for epoch in range(start_epoch, args_cli.epochs):
        disc.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in train_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            pred = batch["pred"].to(device)
            clothing_mask = batch["clothing_mask"].to(device)

            # Real pair: [LR, HR]   Fake pair: [LR, pred]
            real_input = torch.cat([lr, hr], dim=1)   # (B, 6, H, W)
            fake_input = torch.cat([lr, pred], dim=1)  # (B, 6, H, W)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                real_logits = disc(real_input)
                fake_logits = disc(fake_input)

                # Label smoothing on real only (0.9 instead of 1.0)
                real_labels = torch.full_like(real_logits, 0.9)
                fake_labels = torch.zeros_like(fake_logits)

                loss_real = _masked_bce_logits(real_logits, real_labels, bce, clothing_mask)
                loss_fake = _masked_bce_logits(fake_logits, fake_labels, bce, clothing_mask)
                loss = 0.5 * (loss_real + loss_fake)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % 50 == 0:
                print(f"  [epoch {epoch+1} step {global_step}] loss={loss.item():.4f}")

            # Validate
            if global_step % args_cli.validate_every == 0:
                val_metrics = validate(disc, val_loader, bce, device)
                print(f"  VAL step={global_step}: loss={val_metrics['loss']:.4f} "
                      f"p_real={val_metrics['p_real']:.3f} "
                      f"p_fake={val_metrics['p_fake']:.3f} "
                      f"gap={val_metrics['gap_p']:.3f}")
                disc.train()

            # Checkpoint
            if global_step % args_cli.checkpoint_every == 0:
                ckpt_path = os.path.join(output_dir, f"disc_step{global_step}.pt")
                torch.save({
                    "model": disc.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                }, ckpt_path)
                print(f"  Saved checkpoint → {ckpt_path}")

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1}/{args_cli.epochs}: avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")

        # Epoch-end checkpoint
        epoch_ckpt_path = os.path.join(output_dir, f"disc_epoch{epoch+1}.pt")
        torch.save({
            "model": disc.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
        }, epoch_ckpt_path)
        print(f"  Saved epoch checkpoint → {epoch_ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(output_dir, "disc_final.pt")
    torch.save({
        "model": disc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": args_cli.epochs,
        "global_step": global_step,
    }, final_path)
    print(f"Training complete. Final checkpoint → {final_path}")

    # Final validation
    val_metrics = validate(disc, val_loader, bce, device)
    print(f"Final VAL: loss={val_metrics['loss']:.4f} "
          f"p_real={val_metrics['p_real']:.3f} "
          f"p_fake={val_metrics['p_fake']:.3f} "
          f"gap={val_metrics['gap_p']:.3f}")

    # Save a heatmap visualization
    save_heatmap_viz(disc, val_ds, device, output_dir)
