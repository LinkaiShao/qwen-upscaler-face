"""Pre-compute VAE latents, text embeddings, and face data to disk."""

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .config import calculate_dimensions, CONDITION_IMAGE_SIZE


def _cache_key(row) -> str:
    key_str = f"{row['model_lr']}_{row['model_hr']}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _validate_cache(cache_dir, meta_name, params, suffixes):
    """Check cache metadata against current params. Delete stale files if changed."""
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, f".{meta_name}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            stored = json.load(f)
        if stored == params:
            return  # cache is fresh
        # params changed — purge stale files
        print(f"[{meta_name}] Config changed ({stored} -> {params}), clearing stale cache...")
        count = 0
        for suffix in suffixes:
            for p in glob.glob(os.path.join(cache_dir, f"*{suffix}")):
                os.remove(p)
                count += 1
        print(f"[{meta_name}] Deleted {count} stale files.")
    # write current params
    with open(meta_path, "w") as f:
        json.dump(params, f, indent=2)


def precompute_vae_latents(args, splits=("train", "val")):
    """Load Qwen VAE, encode all HR/LR images, save normalized latents to cache."""
    from .models import load_vae, encode_vae_image

    _validate_cache(
        args.latent_cache_dir, "vae",
        {"target_width": args.target_width, "target_height": args.target_height},
        ["_hr_latent.pt", "_lr_latent.pt"],
    )
    df = pd.read_parquet(args.data_parquet)
    df = df[df["split"].isin(splits)].reset_index(drop=True)

    vae = load_vae(args, device="cuda")

    skipped = 0
    for idx in tqdm(range(len(df)), desc="VAE encoding"):
        row = df.iloc[idx]
        key = _cache_key(row)

        hr_path = os.path.join(args.latent_cache_dir, f"{key}_hr_latent.pt")
        lr_path = os.path.join(args.latent_cache_dir, f"{key}_lr_latent.pt")
        if os.path.exists(hr_path) and os.path.exists(lr_path):
            skipped += 1
            continue

        # Encode HR image
        hr_pil = Image.open(row["model_hr"]).convert("RGB")
        hr_pil = hr_pil.resize((args.target_width, args.target_height), Image.LANCZOS)
        hr_tensor = torch.from_numpy(np.array(hr_pil)).permute(2, 0, 1).float() / 255.0
        hr_tensor = hr_tensor * 2.0 - 1.0
        # Add batch + temporal dims: (1, 3, 1, H, W)
        hr_tensor = hr_tensor.unsqueeze(0).unsqueeze(2).to("cuda", dtype=torch.bfloat16)
        hr_latent = encode_vae_image(vae, hr_tensor).cpu().squeeze(0)  # (16, H/8, W/8)
        torch.save(hr_latent, hr_path)

        # Encode LR image (resized to target size)
        lr_pil = Image.open(row["model_lr"]).convert("RGB")
        lr_pil = lr_pil.resize((args.target_width, args.target_height), Image.LANCZOS)
        lr_tensor = torch.from_numpy(np.array(lr_pil)).permute(2, 0, 1).float() / 255.0
        lr_tensor = lr_tensor * 2.0 - 1.0
        lr_tensor = lr_tensor.unsqueeze(0).unsqueeze(2).to("cuda", dtype=torch.bfloat16)
        lr_latent = encode_vae_image(vae, lr_tensor).cpu().squeeze(0)
        torch.save(lr_latent, lr_path)

    del vae
    torch.cuda.empty_cache()
    print(f"VAE encoding done. Skipped {skipped} already cached.")


def precompute_face_data(args, splits=("train", "val")):
    """Load MTCNN + InceptionResnetV1, detect faces, compute embeddings and weight masks."""
    from .face import load_face_models, detect_face, embed_face, make_face_weight_mask

    _validate_cache(
        args.face_cache_dir, "face",
        {"target_width": args.target_width, "target_height": args.target_height, "face_weight": args.face_weight},
        ["_face_weight_mask.pt", "_face_embed.pt", "_face_bbox.pt"],
    )
    df = pd.read_parquet(args.data_parquet)
    df = df[df["split"].isin(splits)].reset_index(drop=True)

    mtcnn, resnet = load_face_models(device="cuda")

    H_latent = args.target_height // 8  # 128
    W_latent = args.target_width // 8   # 96

    skipped = 0
    detected = 0
    for idx in tqdm(range(len(df)), desc="Face detection"):
        row = df.iloc[idx]
        key = _cache_key(row)

        mask_path = os.path.join(args.face_cache_dir, f"{key}_face_weight_mask.pt")
        embed_path = os.path.join(args.face_cache_dir, f"{key}_face_embed.pt")
        bbox_path = os.path.join(args.face_cache_dir, f"{key}_face_bbox.pt")
        if os.path.exists(mask_path) and os.path.exists(bbox_path) and os.path.exists(embed_path):
            skipped += 1
            continue

        hr_pil = Image.open(row["model_hr"]).convert("RGB")
        hr_pil = hr_pil.resize((args.target_width, args.target_height), Image.LANCZOS)

        bbox, confidence = detect_face(hr_pil, mtcnn)

        if bbox is not None:
            detected += 1
            face_embedding = embed_face(hr_pil, bbox, resnet)
            weight_mask = make_face_weight_mask(bbox, H_latent, W_latent, args.face_weight)
            torch.save(face_embedding, embed_path)
        else:
            weight_mask = make_face_weight_mask(None, H_latent, W_latent, args.face_weight)
            torch.save(None, embed_path)

        torch.save(bbox, bbox_path)
        torch.save(weight_mask, mask_path)

    del mtcnn, resnet
    torch.cuda.empty_cache()
    total = len(df) - skipped
    print(f"Face detection done. Detected {detected}/{total} new faces. Skipped {skipped} cached.")


def precompute_text_embeddings(args, splits=("train", "val")):
    """Load Qwen2.5-VL text encoder, encode prompts with LR condition images."""
    from .models import load_text_encoder

    _validate_cache(
        args.latent_cache_dir, "text",
        {"prompt": args.prompt},
        ["_prompt_embeds.pt", "_prompt_mask.pt"],
    )
    df = pd.read_parquet(args.data_parquet)
    df = df[df["split"].isin(splits)].reset_index(drop=True)

    text_encoder, processor = load_text_encoder(args, device="cuda")

    # Prompt template from pipeline
    prompt_template = (
        "<|im_start|>system\n"
        "Describe the key features of the input image (color, shape, size, texture, objects, background), "
        "then explain how the user's text instruction should alter or modify the image. "
        "Generate a new image that meets the user's requirements while maintaining consistency "
        "with the original input where appropriate.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    drop_idx = 64  # from pipeline_qwenimage_edit_plus.py

    skipped = 0
    for idx in tqdm(range(len(df)), desc="Text encoding"):
        row = df.iloc[idx]
        key = _cache_key(row)

        embeds_path = os.path.join(args.latent_cache_dir, f"{key}_prompt_embeds.pt")
        mask_path = os.path.join(args.latent_cache_dir, f"{key}_prompt_mask.pt")
        if os.path.exists(embeds_path) and os.path.exists(mask_path):
            skipped += 1
            continue

        # Load LR as condition image (resized per pipeline convention)
        lr_pil = Image.open(row["model_lr"]).convert("RGB")
        img_w, img_h = lr_pil.size
        cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, img_w / img_h)
        lr_condition = lr_pil.resize((cond_w, cond_h), Image.LANCZOS)

        # Build prompt text with image placeholder
        img_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        txt = prompt_template.format(img_prompt + args.prompt)

        model_inputs = processor(
            text=[txt],
            images=[lr_condition],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]

        # Extract masked hidden states (same logic as pipeline)
        bool_mask = model_inputs.attention_mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        # Drop initial system tokens
        split_hidden = [e[drop_idx:] for e in split_result]
        prompt_embeds = split_hidden[0].cpu()  # (seq_len, hidden_dim)
        prompt_mask = torch.ones(prompt_embeds.size(0), dtype=torch.long)

        torch.save(prompt_embeds, embeds_path)
        torch.save(prompt_mask, mask_path)

    del text_encoder, processor
    torch.cuda.empty_cache()
    print(f"Text encoding done. Skipped {skipped} already cached.")
