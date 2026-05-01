"""
Inpainting generator: 4-layer U-Net with skip connections.

Input:  4 channels (source image 3ch + binary mask 1ch)
Output: 3 channels (inpainted image)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _down_block(
    in_feat: int, out_feat: int, normalize: bool = True
) -> List[nn.Module]:
    layers: List[nn.Module] = [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_feat))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


def _up_block(
    in_feat: int, out_feat: int, dropout: float = 0.0
) -> List[nn.Module]:
    layers: List[nn.Module] = [
        nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_feat),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(dropout))
    return layers


class InpaintGenerator(nn.Module):
    """4-layer U-Net generator for clock hand inpainting."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()

        # Encoder
        self.down1 = nn.Sequential(*_down_block(in_channels, 64, normalize=False))
        self.down2 = nn.Sequential(*_down_block(64, 128))
        self.down3 = nn.Sequential(*_down_block(128, 256))
        self.down4 = nn.Sequential(*_down_block(256, 512, normalize=False))

        # Decoder
        self.up1 = nn.Sequential(*_up_block(512, 256))
        self.up2 = nn.Sequential(*_up_block(512, 128))
        self.up3 = nn.Sequential(*_up_block(256, 64))

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))

        return self.final(torch.cat([u3, d1], 1))


# ============================================================================
# Discriminator with Spectral Normalization (anti-vanishing-gradient)
# ============================================================================

def _sn_conv(in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1) -> nn.Module:
    """Spectrally-normalised Conv2d."""
    return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p))


class InpaintDiscriminator(nn.Module):
    """
    PatchGAN discriminator for inpainting (7-channel input).

    Anti-vanishing-gradient measures:
    - Spectral normalization on all conv layers
    - No InstanceNorm (conflicts with spectral norm)
    - LeakyReLU(0.2) throughout
    """

    def __init__(self, in_channels: int = 7) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _sn_conv(in_channels, 64),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            _sn_conv(256, 1, k=4, s=1, p=1),
        )

    def forward(
        self, inp: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.model(torch.cat((inp, target), 1))
