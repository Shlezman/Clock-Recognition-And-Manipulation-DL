"""
Sketch-cGAN generator: Pix2Pix-style 8-layer U-Net.

Input:  6 channels (source image 3ch + sketch 3ch)
Output: 3 channels (target image)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """Encoder block: Conv → [InstanceNorm] → LeakyReLU → [Dropout]."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        normalize: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose → InstanceNorm → ReLU → [Dropout] + skip-cat."""

    def __init__(
        self, in_size: int, out_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class GeneratorUNet(nn.Module):
    """8-layer U-Net generator for sketch-guided clock image translation."""

    def __init__(self, in_channels: int = 6, out_channels: int = 3) -> None:
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

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
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


# ============================================================================
# Discriminator with Spectral Normalization (anti-vanishing-gradient)
# ============================================================================

def _sn_conv(in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1) -> nn.Module:
    """Spectrally-normalised Conv2d."""
    return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p))


class SketchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for sketch-cGAN (9-channel input).

    Anti-vanishing-gradient measures:
    - Spectral normalization on all conv layers (constrains Lipschitz constant)
    - No InstanceNorm (conflicts with spectral norm)
    - LeakyReLU(0.2) throughout
    """

    def __init__(self, in_channels: int = 9) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _sn_conv(in_channels, 64),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            _sn_conv(512, 1, k=4, s=1, p=1),
        )

    def forward(
        self, src: torch.Tensor, sketch: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.model(torch.cat((src, sketch, target), 1))
