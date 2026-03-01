"""Dataset and image utilities for discriminator training."""

import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

PARSE_DIR = "/home/link/Desktop/Code/fashion gen testing/VITON-HD-dataset/train/image-parse-v3"
CLOTHING_LABELS = {5, 6, 7, 9, 12}  # upper-clothes, dress, coat, pants, skirt
MIN_CLOTHING_FRAC = 0.01  # skip samples with < 1% clothing pixels


def to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """(3, H, W) tensor in [-1, 1] → (H, W, 3) uint8 numpy."""
    arr = ((img_chw.detach().cpu().float().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255.0)
    return arr.clip(0, 255).astype(np.uint8)


def _build_key_to_image_id(parquet_path):
    """Build MD5 cache key → image_id mapping from parquet."""
    df = pd.read_parquet(parquet_path)
    mapping = {}
    for _, row in df.iterrows():
        key = hashlib.md5(f"{row['model_lr']}_{row['model_hr']}".encode()).hexdigest()
        image_id = os.path.splitext(os.path.basename(row["model_hr"]))[0]
        mapping[key] = image_id
    return mapping


def _load_clothing_mask(image_id, h=1024, w=768):
    """Load clothing-only boolean mask from parse map. Returns None if not found."""
    path = os.path.join(PARSE_DIR, f"{image_id}.png")
    if not os.path.exists(path):
        return None
    parse = np.array(Image.open(path).resize((w, h), Image.NEAREST))
    mask = np.isin(parse, list(CLOTHING_LABELS))
    if mask.sum() / mask.size < MIN_CLOTHING_FRAC:
        return None
    return mask


def load_clothing_mask_tensor(image_id, h, w):
    """Return clothing mask tensor (1, H, W) float in {0,1}, or None."""
    mask = _load_clothing_mask(image_id, h=h, w=w)
    if mask is None:
        return None
    return torch.from_numpy(mask).unsqueeze(0).float()


class DiscriminatorDataset(Dataset):
    """Loads precomputed pred/hr/lr PNGs for discriminator training.

    Args:
        cache_dir: Directory with precomputed PNGs.
        split: "train" or "val".
        augment: Whether to apply random horizontal flip.
        clothing_only: Zero out non-clothing pixels using parse maps.
        n_val: Number of val samples (ignored parquet splits, uses random split).
            If None, uses _split.txt files (legacy behavior).
    """

    def __init__(self, cache_dir, split="train", augment=False,
                 clothing_only=False, n_val=None):
        self.cache_dir = cache_dir
        self.augment = augment
        self.clothing_only = clothing_only
        self.key_to_image_id = {}

        # Discover all valid samples
        all_keys = []
        for fname in sorted(os.listdir(cache_dir)):
            if fname.endswith("_pred.png"):
                key = fname[:-len("_pred.png")]
                if (os.path.exists(os.path.join(cache_dir, f"{key}_hr.png")) and
                        os.path.exists(os.path.join(cache_dir, f"{key}_lr.png"))):
                    all_keys.append(key)

        # Build key → image_id mapping if needed for clothing masking
        if clothing_only:
            from qwen_upscaler_face.config import Args
            cfg = Args()
            self.key_to_image_id = _build_key_to_image_id(cfg.data_parquet)

            # Filter to samples with valid clothing masks
            valid_keys = []
            skipped = 0
            for key in all_keys:
                image_id = self.key_to_image_id.get(key)
                if image_id is None:
                    skipped += 1
                    continue
                mask = _load_clothing_mask(image_id)
                if mask is None:
                    skipped += 1
                    continue
                valid_keys.append(key)
            print(f"Clothing filter: {len(valid_keys)} valid, {skipped} skipped (no mask or <1% clothing)")
            all_keys = valid_keys

        # Split: custom n_val or legacy _split.txt
        if n_val is not None:
            # Custom split: shuffle with fixed seed, last n_val → val, rest → train
            shuffled = sorted(all_keys)
            rng = random.Random(42)
            rng.shuffle(shuffled)
            if n_val >= len(shuffled):
                n_val = len(shuffled) // 5  # fallback: 20% val
            if split == "val":
                self.keys = shuffled[-n_val:]
            else:
                self.keys = shuffled[:-n_val]
        else:
            # Legacy: use _split.txt files
            self.keys = []
            for key in all_keys:
                split_file = os.path.join(cache_dir, f"{key}_split.txt")
                if os.path.exists(split_file):
                    with open(split_file) as f:
                        s = f.read().strip()
                    if s != split:
                        continue
                elif split != "train":
                    continue
                self.keys.append(key)

        mode = "clothing-only" if clothing_only else "full-image"
        print(f"DiscriminatorDataset: {len(self.keys)} {split} samples ({mode}, augment={augment})")

    def __len__(self):
        return len(self.keys)

    def _load_png(self, path):
        """Load PNG → (3, H, W) tensor in [-1, 1]."""
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
        t = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1.0
        return t

    def __getitem__(self, idx):
        key = self.keys[idx]
        pred = self._load_png(os.path.join(self.cache_dir, f"{key}_pred.png"))
        hr = self._load_png(os.path.join(self.cache_dir, f"{key}_hr.png"))
        lr = self._load_png(os.path.join(self.cache_dir, f"{key}_lr.png"))
        clothing_mask = torch.ones(1, pred.shape[1], pred.shape[2], dtype=torch.float32)

        # Load clothing mask for loss weighting (inputs stay unmasked)
        if self.clothing_only:
            image_id = self.key_to_image_id.get(key)
            if image_id is not None:
                mask_t = load_clothing_mask_tensor(image_id, h=pred.shape[1], w=pred.shape[2])
                if mask_t is not None:
                    clothing_mask = mask_t

        # Horizontal flip augmentation
        if self.augment and torch.rand(1).item() < 0.5:
            pred = pred.flip(-1)
            hr = hr.flip(-1)
            lr = lr.flip(-1)
            clothing_mask = clothing_mask.flip(-1)

        return {"pred": pred, "hr": hr, "lr": lr, "clothing_mask": clothing_mask}
