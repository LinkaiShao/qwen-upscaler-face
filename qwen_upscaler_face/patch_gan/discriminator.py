"""PatchGAN discriminator (70x70 receptive field, conditional on LR input).

Architecture follows pix2pix NLayerDiscriminator with InstanceNorm
(works with batch_size=1, unlike BatchNorm).

Input:  cat([LR_upsampled, HR_or_pred], dim=1) → (B, 6, 1024, 768)
Output: (B, 1, 126, 94) patch logits
"""

import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with 70x70 receptive field.

    Parameters:
        input_nc (int): Number of input channels (default 6 = 3 LR + 3 HR/pred).
        ndf (int): Base number of filters (default 64).
        n_layers (int): Number of conv layers in the middle (default 3).
    """

    def __init__(self, input_nc=6, ndf=64, n_layers=3):
        super().__init__()

        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Stride-1 layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final 1-channel output
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Gaussian weight initialization (std=0.02) matching pix2pix."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 6, H, W) concatenated [LR, HR_or_pred] in [-1, 1].

        Returns:
            (B, 1, H', W') patch logits (not sigmoided).
        """
        return self.model(x)
