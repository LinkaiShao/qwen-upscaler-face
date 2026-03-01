"""Discriminator heatmap visualization utilities."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import to_uint8


def save_heatmap_viz(disc, val_ds, device, output_dir, n_samples=4):
    """Save discriminator heatmap visualizations for a few val samples."""
    disc.eval()
    viz_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(viz_dir, exist_ok=True)

    n = min(n_samples, len(val_ds))
    if n == 0:
        return

    for i in range(n):
        sample = val_ds[i]
        lr = sample["lr"].unsqueeze(0).to(device)
        hr = sample["hr"].unsqueeze(0).to(device)
        pred = sample["pred"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            real_logits = disc(torch.cat([lr, hr], dim=1))
            fake_logits = disc(torch.cat([lr, pred], dim=1))

        # Convert logits to probabilities
        real_prob = torch.sigmoid(real_logits[0, 0]).float().cpu().numpy()
        fake_prob = torch.sigmoid(fake_logits[0, 0]).float().cpu().numpy()

        # Upsample fake heatmap to image size for overlay
        h, w = hr.shape[2], hr.shape[3]
        fake_up = F.interpolate(
            torch.sigmoid(fake_logits.float()), size=(h, w), mode="bilinear", align_corners=False,
        )[0, 0].cpu().numpy()

        # Save raw heatmaps as grayscale images
        Image.fromarray((real_prob * 255).clip(0, 255).astype(np.uint8)).save(
            os.path.join(viz_dir, f"sample{i}_real_heatmap.png")
        )
        Image.fromarray((fake_prob * 255).clip(0, 255).astype(np.uint8)).save(
            os.path.join(viz_dir, f"sample{i}_fake_heatmap.png")
        )

        # Overlay on pred image: blend fake heatmap (red channel) onto pred
        pred_np = to_uint8(pred[0])  # (H, W, 3)
        overlay = pred_np.copy().astype(np.float32)
        # Red = areas the discriminator thinks are fake (low prob = more fake)
        fake_mask = (1.0 - fake_up) * 255.0  # Invert: 1=real, 0=fake → high=fake
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] * 0.5 + fake_mask * 0.5, 0, 255)
        Image.fromarray(overlay.astype(np.uint8)).save(
            os.path.join(viz_dir, f"sample{i}_pred_overlay.png")
        )

    print(f"Saved {n} heatmap visualizations → {viz_dir}")
