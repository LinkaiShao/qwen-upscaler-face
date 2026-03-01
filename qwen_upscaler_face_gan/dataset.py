"""Dataset with clothing mask for GAN-augmented training."""

import hashlib
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

from qwen_upscaler_face.dataset import CachedQwenUpscaleDataset

PARSE_DIR = "/home/link/Desktop/Code/fashion gen testing/VITON-HD-dataset/train/image-parse-v3"
CLOTHING_LABELS = {5, 6, 7, 9, 12}  # upper-clothes, dress, coat, pants, skirt
MIN_REGION_PIXELS = 500  # discard tiny stray regions


def _load_clothing_mask(image_id, h=1024, w=768):
    """Load clothing mask keeping only the largest contiguous region per label."""
    path = os.path.join(PARSE_DIR, f"{image_id}.png")
    if not os.path.exists(path):
        return None
    parse = np.array(Image.open(path).resize((w, h), Image.NEAREST))

    mask = np.zeros((h, w), dtype=bool)
    for label in CLOTHING_LABELS:
        label_mask = parse == label
        if not label_mask.any():
            continue
        labeled, n_components = ndimage.label(label_mask)
        if n_components == 0:
            continue
        # Keep largest connected component for this label
        sizes = ndimage.sum(label_mask, labeled, range(1, n_components + 1))
        largest = np.argmax(sizes) + 1
        region = labeled == largest
        if region.sum() >= MIN_REGION_PIXELS:
            mask |= region

    if mask.sum() / mask.size < 0.01:
        return None
    return mask


class GanDataset(Dataset):
    """Wraps CachedQwenUpscaleDataset to add clothing mask for GAN loss.

    Returns all original fields plus 'clothing_mask' (1, H, W) float tensor.
    """

    def __init__(self, parquet_path, split, latent_cache_dir, face_cache_dir,
                 augment=False, target_width=768):
        self.inner = CachedQwenUpscaleDataset(
            parquet_path, split, latent_cache_dir, face_cache_dir,
            augment=False,  # we handle augment ourselves to include mask
            target_width=target_width,
        )
        self.augment = augment
        self.target_width = target_width

        # Build key → image_id mapping for clothing mask lookup
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.key_to_image_id = {}
        for _, row in df.iterrows():
            key = hashlib.md5(f"{row['model_lr']}_{row['model_hr']}".encode()).hexdigest()
            image_id = os.path.splitext(os.path.basename(row["model_hr"]))[0]
            self.key_to_image_id[key] = image_id

        print(f"GanDataset: {len(self.inner)} {split} samples (augment={augment})")

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        data = self.inner[idx]

        # Get image_id for this sample
        row = self.inner.df.iloc[idx]
        key = hashlib.md5(f"{row['model_lr']}_{row['model_hr']}".encode()).hexdigest()
        image_id = self.key_to_image_id.get(key)

        # Load clothing mask
        if image_id is not None:
            mask = _load_clothing_mask(image_id)
        else:
            mask = None

        if mask is not None:
            clothing_mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)
        else:
            clothing_mask = torch.ones(1, data["hr_latent"].shape[1] * 8,
                                       data["hr_latent"].shape[2] * 8)

        # Horizontal flip augmentation (latent space + clothing mask)
        if self.augment and torch.rand(1).item() < 0.5:
            data["hr_latent"] = data["hr_latent"].flip(-1)
            data["lr_latent"] = data["lr_latent"].flip(-1)
            data["face_weight_mask"] = data["face_weight_mask"].flip(-1)
            clothing_mask = clothing_mask.flip(-1)
            if data["has_face"]:
                bbox = data["face_bbox"]
                x1, y1, x2, y2 = bbox.tolist()
                w_pixel = float(self.target_width)
                data["face_bbox"] = torch.tensor(
                    [w_pixel - x2, y1, w_pixel - x1, y2], dtype=torch.float32)

        data["clothing_mask"] = clothing_mask
        return data


def create_dataloaders(args):
    """Create train and validation dataloaders with clothing masks."""
    train_dataset = GanDataset(
        args.data_parquet, "train", args.latent_cache_dir, args.face_cache_dir,
        augment=True, target_width=args.target_width,
    )
    val_dataset = GanDataset(
        args.data_parquet, "val", args.latent_cache_dir, args.face_cache_dir,
        augment=False, target_width=args.target_width,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
    )

    return train_loader, val_loader
