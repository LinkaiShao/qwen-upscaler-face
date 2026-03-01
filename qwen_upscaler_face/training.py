"""Training step with single-step LR→HR residual prediction + face identity loss."""

import torch
import torch.nn.functional as F

from .models import pack_latents, unpack_latents, denormalize_latents
from .face import pack_weight_mask
from .lighting import lock_lighting


def train_step(batch, transformer, vae, face_encoder, args, device, weight_dtype):
    """Execute single training step with single-step residual prediction.

    Model predicts v = HR - LR at t=0. Inference: HR_pred = LR + v_pred.

    Returns: (loss, flow_loss_value, id_loss_value)
    """
    hr_latent = batch["hr_latent"].to(device, dtype=weight_dtype).contiguous()
    lr_latent = batch["lr_latent"].to(device, dtype=weight_dtype).contiguous()
    prompt_embeds = batch["prompt_embeds"].to(device, dtype=weight_dtype).contiguous()
    prompt_mask = batch["prompt_mask"].to(device, dtype=torch.long).contiguous()
    face_weight_mask = batch["face_weight_mask"].to(device, dtype=torch.float32).contiguous()
    face_bbox = batch["face_bbox"]  # (B, 4)
    face_embed_gt = batch["face_embed"].to(device, dtype=torch.float32)  # (B, 512)
    has_face = batch["has_face"]  # (B,) bool

    B, C, H, W = hr_latent.shape  # B, 16, 128, 96

    # Pack latents
    hr_packed = pack_latents(hr_latent, B, C, H, W)   # (B, 3072, 64)
    lr_packed = pack_latents(lr_latent, B, C, H, W)    # (B, 3072, 64)

    # Pack face weight mask per sample
    face_w_list = []
    for i in range(B):
        fw = pack_weight_mask(face_weight_mask[i], H, W)  # (1, 3072, 1)
        face_w_list.append(fw)
    face_w = torch.cat(face_w_list, dim=0).to(device)  # (B, 3072, 1)

    # Single-step: always t=0, input is LR
    t = torch.zeros(B, device=device, dtype=weight_dtype)

    # Residual target: HR - LR
    v_target = (hr_packed - lr_packed).to(torch.float32)

    # Concat [LR, LR] — at t=0, current state = LR = condition
    hidden_states = torch.cat([lr_packed, lr_packed], dim=1)  # (B, 6144, 64)

    # img_shapes: [(output_shape, source_shape)] per batch
    img_shapes = [
        [(1, H // 2, W // 2), (1, H // 2, W // 2)]
    ] * B

    # txt_seq_lens from prompt mask
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    # Forward through transformer
    pred = transformer(
        hidden_states=hidden_states,
        timestep=t,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        return_dict=False,
    )[0].to(torch.float32)

    # Slice to velocity prediction (first N tokens = output latent)
    v_pred = pred[:, :hr_packed.size(1), :]

    # Face-weighted flow loss
    per_token_mse = (v_pred - v_target).pow(2).mean(dim=-1, keepdim=True)  # (B, 3072, 1)
    flow_loss = (per_token_mse * face_w).mean()

    # --- Face identity loss ---
    id_loss = torch.tensor(0.0, device=device)

    if has_face.any() and args.lambda_id > 0:
        # Predicted HR = LR + v_pred (add residual to LR, valid at all timesteps)
        x1_pred = lr_packed + v_pred  # (B, seq, C*4), float32

        # Unpack to spatial latent: (B, C, 1, H, W)
        x1_unpacked = unpack_latents(x1_pred, H, W)

        # Denormalize + VAE decode predicted HR (gradients flow through frozen VAE)
        x1_denorm = denormalize_latents(x1_unpacked.to(vae.dtype), vae)
        decoded = vae.decode(x1_denorm, return_dict=False)[0][:, :, 0]  # (B, 3, H*8, W*8)
        decoded = decoded.clamp(-1, 1).float()

        # Light lock: anchor lighting to LR guide before face identity comparison
        if args.lock_lighting:
            lr_for_guide = lr_latent.unsqueeze(2)  # (B, 16, 1, H, W)
            lr_denorm = denormalize_latents(lr_for_guide.to(vae.dtype), vae)
            lr_decoded = vae.decode(lr_denorm, return_dict=False)[0][:, :, 0]
            lr_decoded = lr_decoded.clamp(-1, 1).float()
            decoded = lock_lighting(decoded, lr_decoded, args.lighting_y_blend, args.lighting_blur_radius)

        id_losses = []
        for b in range(B):
            if not has_face[b]:
                continue

            bbox = face_bbox[b]  # (x1, y1, x2, y2) pixel coords
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Clamp to image bounds
            img_h, img_w = decoded.shape[2], decoded.shape[3]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            # Crop + resize to 160x160 for face encoder
            pred_crop = F.interpolate(
                decoded[b:b+1, :, y1:y2, x1:x2], size=(160, 160),
                mode="bilinear", align_corners=False,
            )  # (1, 3, 160, 160), values in [-1, 1]
            # Convert [-1,1] to MTCNN normalization: (pixel-127.5)/128
            pred_crop = pred_crop * (255.0 / 256.0)

            pred_embed = face_encoder(pred_crop)  # (1, 512)
            gt_embed = face_embed_gt[b:b+1]  # (1, 512) precomputed

            cos_sim = F.cosine_similarity(pred_embed, gt_embed, dim=1)  # (1,)
            id_losses.append(1.0 - cos_sim)

        if id_losses:
            id_loss = torch.cat(id_losses).mean()

    # Anchor loss: penalize large deviations from LR (keeps output near baseline)
    anchor_loss = v_pred.pow(2).mean()

    loss = flow_loss + args.lambda_id * id_loss + args.lambda_anchor * anchor_loss

    return loss, flow_loss.item(), id_loss.item()
