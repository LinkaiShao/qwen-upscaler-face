"""Dataset loading pre-computed latents, text embeddings, and face data."""

import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _cache_key(row) -> str:
    """Deterministic hash key for a sample."""
    key_str = f"{row['model_lr']}_{row['model_hr']}"
    return hashlib.md5(key_str.encode()).hexdigest()


class CachedQwenUpscaleDataset(Dataset):
    """Loads pre-computed latents, text embeddings, and face data from disk."""

    def __init__(self, parquet_path, split, latent_cache_dir, face_cache_dir, augment=False, target_width=768):
        df = pd.read_parquet(parquet_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.latent_dir = Path(latent_cache_dir)
        self.face_dir = Path(face_cache_dir)
        self.augment = augment
        self.target_width = target_width
        print(f"CachedQwenUpscaleDataset: {len(self.df)} {split} samples (augment={augment})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = _cache_key(row)

        hr_latent = torch.load(self.latent_dir / f"{key}_hr_latent.pt", weights_only=True)
        lr_latent = torch.load(self.latent_dir / f"{key}_lr_latent.pt", weights_only=True)
        prompt_embeds = torch.load(self.latent_dir / f"{key}_prompt_embeds.pt", weights_only=True)
        prompt_mask = torch.load(self.latent_dir / f"{key}_prompt_mask.pt", weights_only=True)

        face_weight_mask = torch.load(self.face_dir / f"{key}_face_weight_mask.pt", weights_only=True)
        face_embed_path = self.face_dir / f"{key}_face_embed.pt"
        if face_embed_path.exists():
            face_data = torch.load(face_embed_path, weights_only=True)
            has_face = face_data is not None
            face_embed = face_data if has_face else torch.zeros(512)
        else:
            has_face = False
            face_embed = torch.zeros(512)

        face_bbox_path = self.face_dir / f"{key}_face_bbox.pt"
        if face_bbox_path.exists():
            bbox_data = torch.load(face_bbox_path, weights_only=False)
            face_bbox = torch.tensor(bbox_data, dtype=torch.float32) if bbox_data is not None else torch.zeros(4)
        else:
            face_bbox = torch.zeros(4)

        # Horizontal flip augmentation (latent space)
        if self.augment and torch.rand(1).item() < 0.5:
            hr_latent = hr_latent.flip(-1)
            lr_latent = lr_latent.flip(-1)
            face_weight_mask = face_weight_mask.flip(-1)
            if has_face:
                x1, y1, x2, y2 = face_bbox.tolist()
                w_pixel = float(self.target_width)
                face_bbox = torch.tensor([w_pixel - x2, y1, w_pixel - x1, y2], dtype=torch.float32)

        return {
            "hr_latent": hr_latent,
            "lr_latent": lr_latent,
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "face_weight_mask": face_weight_mask,
            "face_embed": face_embed,
            "face_bbox": face_bbox,
            "has_face": has_face,
        }


def create_dataloaders(args):
    """Create train and validation dataloaders."""
    train_dataset = CachedQwenUpscaleDataset(
        args.data_parquet, "train", args.latent_cache_dir, args.face_cache_dir,
        augment=True, target_width=args.target_width,
    )
    val_dataset = CachedQwenUpscaleDataset(
        args.data_parquet, "val", args.latent_cache_dir, args.face_cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader
