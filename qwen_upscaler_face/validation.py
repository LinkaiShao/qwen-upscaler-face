"""Validation with PSNR/SSIM and face identity metric (single-step residual)."""

import numpy as np
import torch
import torch.nn.functional as F

from .models import pack_latents, unpack_latents, denormalize_latents
from .lighting import lock_lighting


@torch.no_grad()
def validate(
    transformer,
    val_loader,
    vae,
    face_encoder,
    args,
    device,
    weight_dtype,
    max_samples=8,
):
    """Single-step residual prediction on validation samples.

    HR_pred = LR + model(LR, t=0). Returns dict with: mse, psnr, ssim, face_cos_sim.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    transformer.eval()
    metrics = {"mse": [], "psnr": [], "ssim": [], "face_cos_sim": []}
    n_seen = 0
    decoded_samples = []

    for batch in val_loader:
        if n_seen >= max_samples:
            break

        hr_latent = batch["hr_latent"].to(device, dtype=weight_dtype)
        lr_latent = batch["lr_latent"].to(device, dtype=weight_dtype)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype=weight_dtype)
        prompt_mask = batch["prompt_mask"].to(device, dtype=torch.long)
        face_embed_gt = batch["face_embed"].to(device, dtype=torch.float32)  # (B, 512)
        face_bbox = batch["face_bbox"]  # (B, 4)
        has_face = batch["has_face"]  # (B,) bool

        B, C, H, W = hr_latent.shape

        # Pack latents
        hr_packed = pack_latents(hr_latent, B, C, H, W)
        lr_packed = pack_latents(lr_latent, B, C, H, W)

        img_shapes = [[(1, H // 2, W // 2), (1, H // 2, W // 2)]] * B
        txt_seq_lens = prompt_mask.sum(dim=1).tolist()

        # Single-step: predict residual at t=0, apply once
        t_batch = torch.zeros(B, device=device, dtype=weight_dtype)
        hidden = torch.cat([lr_packed, lr_packed], dim=1)
        pred = transformer(
            hidden_states=hidden,
            timestep=t_batch,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        v_pred = pred[:, :lr_packed.size(1), :]
        z = lr_packed + v_pred

        # Latent MSE
        mse = F.mse_loss(z.float(), hr_packed.float()).item()
        metrics["mse"].append(mse)

        # Decode for pixel-space metrics
        z_unpacked = unpack_latents(z, H, W)  # (B, C, 1, H, W)
        z_denorm = denormalize_latents(z_unpacked.to(vae.dtype), vae)
        decoded = vae.decode(z_denorm, return_dict=False)[0][:, :, 0]  # (B, 3, H*8, W*8)
        decoded = decoded.clamp(-1, 1)

        # Light lock: anchor lighting to LR guide (matches inference post-processing)
        if args.lock_lighting:
            lr_unpacked = lr_latent.unsqueeze(2)  # (B, 16, 1, H, W)
            lr_denorm = denormalize_latents(lr_unpacked.to(vae.dtype), vae)
            lr_decoded = vae.decode(lr_denorm, return_dict=False)[0][:, :, 0]
            lr_decoded = lr_decoded.clamp(-1, 1)
            decoded = lock_lighting(decoded, lr_decoded, args.lighting_y_blend, args.lighting_blur_radius)

        # Also decode HR for comparison
        hr_unpacked = unpack_latents(hr_packed, H, W)
        hr_denorm = denormalize_latents(hr_unpacked.to(vae.dtype), vae)
        hr_decoded = vae.decode(hr_denorm, return_dict=False)[0][:, :, 0]
        hr_decoded = hr_decoded.clamp(-1, 1)

        # Pixel-space metrics
        for b in range(B):
            pred_np = ((decoded[b].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            hr_np = ((hr_decoded[b].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

            psnr = peak_signal_noise_ratio(hr_np, pred_np)
            ssim = structural_similarity(hr_np, pred_np, channel_axis=2)
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)

            # Face identity cosine similarity
            if has_face[b]:
                bbox = face_bbox[b]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                img_h, img_w = decoded.shape[2], decoded.shape[3]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    pred_crop = F.interpolate(
                        decoded[b:b+1, :, y1:y2, x1:x2].float(),
                        size=(160, 160), mode="bilinear", align_corners=False,
                    )
                    # Convert [-1,1] to MTCNN normalization: (pixel-127.5)/128
                    pred_crop = pred_crop * (255.0 / 256.0)
                    pred_embed = face_encoder(pred_crop)  # (1, 512)
                    gt_embed = face_embed_gt[b:b+1]  # (1, 512)
                    cos_sim = F.cosine_similarity(pred_embed, gt_embed, dim=1).item()
                    metrics["face_cos_sim"].append(cos_sim)

            if n_seen < 4:
                decoded_samples.append((pred_np, hr_np))

        n_seen += B

    transformer.train()

    result = {}
    for k, v in metrics.items():
        result[k] = float(np.mean(v)) if v else 0.0

    result["decoded_samples"] = decoded_samples
    return result
