"""PatchGAN discriminator for Qwen upscaler adversarial training."""

from .discriminator import NLayerDiscriminator
from .dataset import DiscriminatorDataset, to_uint8
from .visualization import save_heatmap_viz
from .training import train, validate

# precompute imports heavy model-loading code; import on demand.
# Usage: from qwen_upscaler_face.patch_gan.precompute import precompute
