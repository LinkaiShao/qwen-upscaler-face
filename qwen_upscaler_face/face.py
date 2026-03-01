"""Face detection and identity embedding utilities."""

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


def load_face_models(device="cuda"):
    """Load frozen MTCNN face detector + InceptionResnetV1 identity encoder."""
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        device=device,
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    resnet.requires_grad_(False)
    return mtcnn, resnet


@torch.no_grad()
def detect_face(image_pil: Image.Image, mtcnn: MTCNN, min_confidence: float = 0.9):
    """Detect face in a PIL image.

    Returns:
        (bbox, confidence) where bbox is (x1, y1, x2, y2) in pixel coords,
        or (None, 0.0) if no face detected.
    """
    boxes, probs = mtcnn.detect(image_pil)
    if boxes is not None and len(boxes) > 0 and probs[0] >= min_confidence:
        return boxes[0].tolist(), float(probs[0])
    return None, 0.0


@torch.no_grad()
def embed_face(
    image_pil: Image.Image,
    bbox,
    resnet: InceptionResnetV1,
):
    """Extract 512-dim face identity embedding using bbox crop.

    Uses raw bbox crop (no MTCNN alignment) so the embedding pipeline matches
    training-time crops from decoded latents.

    Returns:
        (512,) tensor or None if crop is degenerate.
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    face_crop = image_pil.crop((x1, y1, x2, y2)).resize((160, 160), Image.LANCZOS)

    # MTCNN-style normalization: (pixel - 127.5) / 128.0
    arr = np.array(face_crop, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 160, 160)
    tensor = (tensor - 127.5) / 128.0
    tensor = tensor.unsqueeze(0).to(next(resnet.parameters()).device)

    embedding = resnet(tensor)  # (1, 512)
    return embedding.squeeze(0).cpu()


def make_face_weight_mask(
    bbox,
    H_latent: int,
    W_latent: int,
    face_weight: float = 5.0,
    vae_scale: int = 8,
):
    """Create spatial weight mask with upweighted face region.

    Args:
        bbox: (x1, y1, x2, y2) in pixel coords, or None
        H_latent, W_latent: latent spatial dimensions (e.g., 128, 96)
        face_weight: multiplier for face region
        vae_scale: VAE spatial compression factor

    Returns:
        (H_latent, W_latent) float32 tensor, normalized so mean ≈ 1.0
    """
    mask = torch.ones(H_latent, W_latent, dtype=torch.float32)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Convert pixel coords to latent coords
        lx1 = max(0, int(x1 / vae_scale))
        ly1 = max(0, int(y1 / vae_scale))
        lx2 = min(W_latent, int(x2 / vae_scale) + 1)
        ly2 = min(H_latent, int(y2 / vae_scale) + 1)
        if ly2 > ly1 and lx2 > lx1:
            mask[ly1:ly2, lx1:lx2] = face_weight

    # Normalize so mean weight ≈ 1.0
    mask = mask / mask.mean()
    return mask


def pack_weight_mask(mask: torch.Tensor, H: int, W: int):
    """Pack spatial weight mask to match transformer packed token layout.

    Args:
        mask: (H, W) weight mask
        H, W: latent spatial dims (must match mask shape)

    Returns:
        (1, (H//2)*(W//2), 1) packed mask for per-token weighting
    """
    # Average 2x2 patches (matches latent packing)
    packed = mask.view(H // 2, 2, W // 2, 2).mean(dim=(1, 3))  # (H//2, W//2)
    return packed.reshape(1, -1, 1)
