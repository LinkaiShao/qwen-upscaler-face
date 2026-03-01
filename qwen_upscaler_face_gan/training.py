"""Training step with GAN loss for the Qwen upscaler.

Extends the base training step with adversarial loss from a PatchGAN discriminator.
The VAE decode is shared between the face identity loss and GAN loss.

Three-phase training:
  Phase 0: top-k MSE only, no GAN, D frozen. id periodic + gated.
  Phase 1: GAN ramps in, D updates every 2 steps, top-k at 20%.
  Phase 2: GAN stable, D every 4 steps, top-k tightens to 10%.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_upscaler_face.models import pack_latents, unpack_latents, denormalize_latents
from qwen_upscaler_face.face import pack_weight_mask
from qwen_upscaler_face.lighting import lock_lighting


def _masked_bce(logits, labels, clothing_mask):
    """BCE on logits weighted by clothing mask coverage."""
    bce = nn.BCEWithLogitsLoss(reduction="none")
    patch_mask = F.interpolate(clothing_mask, size=logits.shape[-2:], mode="area")
    loss_map = bce(logits, labels)
    weighted = loss_map * patch_mask
    denom = patch_mask.sum().clamp_min(1e-6)
    return weighted.sum() / denom


def _decode_pixels(latent_packed, lr_latent, vae, H, W, with_grad=True):
    """Decode packed latent tokens to pixel space."""
    unpacked = unpack_latents(latent_packed, H, W)  # (B, C, 1, H, W)
    denorm = denormalize_latents(unpacked.to(vae.dtype), vae)
    if with_grad:
        decoded = vae.decode(denorm, return_dict=False)[0][:, :, 0]
    else:
        with torch.no_grad():
            decoded = vae.decode(denorm, return_dict=False)[0][:, :, 0]
    return decoded.clamp(-1, 1).float()


def train_step(batch, transformer, vae, face_encoder, disc, args, device, weight_dtype,
               *, phase, global_step, compute_id=True, lambda_gan_eff=0.0, topk_ratio=0.2):
    """Generator training step with flow + face id + anchor + GAN + top-k MSE.

    Args:
        phase: current training phase (0, 1, or 2).
        global_step: current optimizer step count.
        compute_id: whether to compute face identity loss this step.
        lambda_gan_eff: effective GAN weight (0 in phase 0, ramped in phase 1).
        topk_ratio: fraction of worst clothing pixels for top-k mining.

    Returns:
        dict with loss components and cos_sim for id gating.
    """
    hr_latent = batch["hr_latent"].to(device, dtype=weight_dtype).contiguous()
    lr_latent = batch["lr_latent"].to(device, dtype=weight_dtype).contiguous()
    prompt_embeds = batch["prompt_embeds"].to(device, dtype=weight_dtype).contiguous()
    prompt_mask = batch["prompt_mask"].to(device, dtype=torch.long).contiguous()
    face_weight_mask = batch["face_weight_mask"].to(device, dtype=torch.float32).contiguous()
    face_bbox = batch["face_bbox"]
    face_embed_gt = batch["face_embed"].to(device, dtype=torch.float32)
    has_face = batch["has_face"]
    clothing_mask = batch["clothing_mask"].to(device, dtype=torch.float32)

    B, C, H, W = hr_latent.shape

    # Pack latents
    hr_packed = pack_latents(hr_latent, B, C, H, W)
    lr_packed = pack_latents(lr_latent, B, C, H, W)

    # Pack face weight mask
    face_w_list = []
    for i in range(B):
        fw = pack_weight_mask(face_weight_mask[i], H, W)
        face_w_list.append(fw)
    face_w = torch.cat(face_w_list, dim=0).to(device)

    # Single-step residual: t=0, input=LR
    t = torch.zeros(B, device=device, dtype=weight_dtype)
    v_target = (hr_packed - lr_packed).to(torch.float32)

    hidden_states = torch.cat([lr_packed, lr_packed], dim=1)
    img_shapes = [[(1, H // 2, W // 2), (1, H // 2, W // 2)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    # Transformer forward
    pred = transformer(
        hidden_states=hidden_states,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        return_dict=False,
    )[0].to(torch.float32)

    v_pred = pred[:, :hr_packed.size(1), :]

    # === Latent-space losses ===
    per_token_mse = (v_pred - v_target).pow(2).mean(dim=-1, keepdim=True)
    flow_loss = (per_token_mse * face_w).mean()
    anchor_loss = v_pred.pow(2).mean()

    # === Decode to pixel space ===
    x1_pred = lr_packed + v_pred
    need_pixels = compute_id or lambda_gan_eff > 0 or topk_ratio > 0
    if need_pixels:
        pred_pixels = _decode_pixels(x1_pred, lr_latent, vae, H, W, with_grad=True)
        lr_pixels = _decode_pixels(lr_packed, lr_latent, vae, H, W, with_grad=False)
        clothing_mask_pixel = F.interpolate(clothing_mask, size=pred_pixels.shape[-2:], mode="nearest")

    # === Face identity loss (periodic + threshold gated) ===
    id_loss = torch.tensor(0.0, device=device)
    cos_sim_val = 1.0  # default: face is fine
    if compute_id and need_pixels and has_face.any() and args.lambda_id > 0:
        pred_for_face = pred_pixels
        if args.lock_lighting:
            pred_for_face = lock_lighting(
                pred_pixels, lr_pixels,
                args.lighting_y_blend, args.lighting_blur_radius,
            )

        id_losses = []
        cos_sims = []
        for b in range(B):
            if not has_face[b]:
                continue
            bbox = face_bbox[b]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            img_h, img_w = pred_for_face.shape[2], pred_for_face.shape[3]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            pred_crop = F.interpolate(
                pred_for_face[b:b+1, :, y1:y2, x1:x2], size=(160, 160),
                mode="bilinear", align_corners=False,
            )
            pred_crop = pred_crop * (255.0 / 256.0)
            pred_embed = face_encoder(pred_crop)
            gt_embed = face_embed_gt[b:b+1]
            cs = F.cosine_similarity(pred_embed, gt_embed, dim=1)
            cos_sims.append(cs.item())
            id_losses.append(1.0 - cs)
        if id_losses:
            id_loss = torch.cat(id_losses).mean()
            cos_sim_val = sum(cos_sims) / len(cos_sims)

    # === GAN loss (phase 1+2 only, weight controlled by caller) ===
    gan_loss = torch.tensor(0.0, device=device)
    if lambda_gan_eff > 0 and need_pixels:
        # Detach non-clothing pixels
        masked_pred = pred_pixels * clothing_mask_pixel + pred_pixels.detach() * (1 - clothing_mask_pixel)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            disc_input = torch.cat([lr_pixels, masked_pred], dim=1)
            fake_logits = disc(disc_input)
        real_labels = torch.ones_like(fake_logits)
        gan_loss = _masked_bce(fake_logits.float(), real_labels, clothing_mask)

    # === Top-k clothing pixel MSE (spatial loss map with hard mining) ===
    # E(i) = per-pixel MSE, spatial map (B,1,H,W)
    # m(i) = binary mask selecting worst top-k clothing pixels
    # loss = sum(E(i) * m(i)) / k  — gradients flow only from top-k patches
    topk_loss = torch.tensor(0.0, device=device)
    if topk_ratio > 0 and need_pixels:
        hr_pixels = _decode_pixels(hr_packed, lr_latent, vae, H, W, with_grad=False)
        E = (pred_pixels - hr_pixels).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)
        valid = (clothing_mask_pixel > 0.5).flatten()
        E_flat = E.flatten()
        valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
        n_clothing = int(valid_idx.numel())
        if n_clothing > 0:
            k = min(max(1, int(n_clothing * topk_ratio)), n_clothing)
            E_valid = E_flat[valid_idx]
            _, topk_local_idx = E_valid.topk(k)
            topk_idx = valid_idx[topk_local_idx]
            m = torch.zeros_like(E_flat)
            m[topk_idx] = 1.0
            topk_loss = (E_flat * m).sum() / k

    # === Total loss with flow floor guarantee ===
    # Compute other losses first, then scale them so flow is at least 30%.
    flow_val = flow_loss.item()
    other_loss = (args.lambda_id * id_loss
                  + args.lambda_anchor * anchor_loss
                  + lambda_gan_eff * gan_loss
                  + args.lambda_topk * topk_loss)
    other_val = other_loss.item()

    min_flow_frac = getattr(args, "min_flow_frac", 0.30)
    # flow / (flow + scale * other) >= min_flow_frac
    # => scale <= flow * (1 - min_flow_frac) / (other * min_flow_frac)
    if other_val > 1e-8 and flow_val > 1e-8:
        max_scale = flow_val * (1.0 - min_flow_frac) / (other_val * min_flow_frac)
        scale = min(1.0, max_scale)
    else:
        scale = 1.0

    loss = flow_loss + scale * other_loss

    return {
        "loss": loss,
        "flow": flow_val,
        "id": id_loss.item(),
        "anchor": anchor_loss.item(),
        "gan": gan_loss.item(),
        "topk_mse": topk_loss.item(),
        "cos_sim": cos_sim_val,
        "flow_scale": scale,
    }


def disc_step(batch, transformer, vae, disc, args, device, weight_dtype):
    """Discriminator training step.

    Runs transformer + VAE decode under no_grad. Only disc has gradients.
    """
    hr_latent = batch["hr_latent"].to(device, dtype=weight_dtype).contiguous()
    lr_latent = batch["lr_latent"].to(device, dtype=weight_dtype).contiguous()
    prompt_embeds = batch["prompt_embeds"].to(device, dtype=weight_dtype).contiguous()
    prompt_mask = batch["prompt_mask"].to(device, dtype=torch.long).contiguous()
    clothing_mask = batch["clothing_mask"].to(device, dtype=torch.float32)

    B, C, H, W = hr_latent.shape

    with torch.no_grad():
        hr_packed = pack_latents(hr_latent, B, C, H, W)
        lr_packed = pack_latents(lr_latent, B, C, H, W)

        t = torch.zeros(B, device=device, dtype=weight_dtype)
        hidden_states = torch.cat([lr_packed, lr_packed], dim=1)
        img_shapes = [[(1, H // 2, W // 2), (1, H // 2, W // 2)]] * B
        txt_seq_lens = prompt_mask.sum(dim=1).tolist()

        pred = transformer(
            hidden_states=hidden_states,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        v_pred = pred[:, :hr_packed.size(1), :]
        x1_pred = lr_packed + v_pred

        pred_pixels = _decode_pixels(x1_pred, lr_latent, vae, H, W, with_grad=False)
        lr_pixels = _decode_pixels(lr_packed, lr_latent, vae, H, W, with_grad=False)
        hr_pixels = _decode_pixels(hr_packed, lr_latent, vae, H, W, with_grad=False)

    # Disc forward (with grad on disc params)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        real_input = torch.cat([lr_pixels, hr_pixels], dim=1)
        fake_input = torch.cat([lr_pixels, pred_pixels], dim=1)
        real_logits = disc(real_input)
        fake_logits = disc(fake_input)

    # Masked BCE with label smoothing on real
    real_labels = torch.full_like(real_logits, 0.9)
    fake_labels = torch.zeros_like(fake_logits)
    d_loss_real = _masked_bce(real_logits.float(), real_labels, clothing_mask)
    d_loss_fake = _masked_bce(fake_logits.float(), fake_labels, clothing_mask)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    # Mean sigmoid probabilities on clothing patches
    patch_mask = F.interpolate(clothing_mask, size=real_logits.shape[-2:], mode="area")
    denom = patch_mask.sum().clamp_min(1e-6)
    p_real = (torch.sigmoid(real_logits.float()) * patch_mask).sum().item() / denom.item()
    p_fake = (torch.sigmoid(fake_logits.float()) * patch_mask).sum().item() / denom.item()

    return {
        "d_loss": d_loss,
        "d_loss_real": d_loss_real.item(),
        "d_loss_fake": d_loss_fake.item(),
        "p_real": p_real,
        "p_fake": p_fake,
        "gap_p": p_real - p_fake,
    }
