"""
ViT-based clock generation model.

Architecture:
  - Encoder: pretrained vit_small_patch16_224 (timm) — extracts 196 patch tokens
  - Time conditioning: sinusoidal embedding for (HH, MM) → AdaIN scale/shift in decoder
  - Decoder: 4-block progressive upsampling CNN (14→28→56→112→224)
  - Output: Tanh-normalised 224×224 RGB image

Training loss: L1 + perceptual (VGG16 relu_1_2 / relu_2_2 / relu_3_3).
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding for a scalar tensor t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    """Encode (hour, minute) integers into a *cond_dim*-dimensional vector."""

    def __init__(self, cond_dim: int = 256, raw_dim: int = 128) -> None:
        super().__init__()
        # raw_dim*2 for each of hour & minute → raw_dim*4 total
        self.proj = nn.Sequential(
            nn.Linear(raw_dim * 4, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.raw_dim = raw_dim

    def forward(self, hh: torch.Tensor, mm: torch.Tensor) -> torch.Tensor:
        t_h = (hh % 12).float() / 12.0   # 12-hour cycle
        t_m = mm.float() / 60.0
        emb_h = _sinusoidal(t_h, self.raw_dim * 2)   # (B, raw_dim*2)
        emb_m = _sinusoidal(t_m, self.raw_dim * 2)
        return self.proj(torch.cat([emb_h, emb_m], dim=-1))  # (B, cond_dim)


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """ConvTranspose upsample (x2) with AdaIN-style time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=False),
            nn.ReLU(inplace=True),
        )
        # AdaIN: predict per-channel scale and shift from conditioning vector
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)   # each (B, C)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ViTClockGenerator(nn.Module):
    """
    ViT encoder + time-conditioned CNN decoder for analog clock image translation.

    Args:
        vit_name:  timm model name (default: vit_small_patch16_224).
        cond_dim:  dimensionality of the time conditioning vector.
        pretrained: load ImageNet weights for the ViT encoder.
    """

    VIT_FEAT_DIMS = {
        "vit_small_patch16_224": 384,
        "vit_base_patch16_224":  768,
        "vit_tiny_patch16_224":  192,
    }

    def __init__(
        self,
        vit_name: str = "vit_small_patch16_224",
        cond_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        vit_dim = self.VIT_FEAT_DIMS.get(vit_name, self.encoder.embed_dim)

        self.time_emb = TimeEmbedding(cond_dim=cond_dim)

        # Project ViT patch tokens → initial spatial feature map (14x14)
        self.feat_proj = nn.Sequential(
            nn.Linear(vit_dim, 512),
            nn.ReLU(inplace=True),
        )

        # Decoder: 14→28→56→112→224
        self.dec1 = DecoderBlock(512, 256, cond_dim)
        self.dec2 = DecoderBlock(256, 128, cond_dim)
        self.dec3 = DecoderBlock(128, 64,  cond_dim)
        self.dec4 = DecoderBlock(64,  32,  cond_dim)

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        hh: torch.Tensor,
        mm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:  (B, 3, 224, 224) source clock image, normalised to [-1, 1].
            hh: (B,) integer hour values [0, 23].
            mm: (B,) integer minute values [0, 59].

        Returns:
            (B, 3, 224, 224) generated clock image in [-1, 1].
        """
        B = x.shape[0]
        cond = self.time_emb(hh, mm)                    # (B, cond_dim)

        tokens = self.encoder.forward_features(x)       # (B, 197, D) — CLS + 196 patches
        patches = tokens[:, 1:]                          # drop CLS → (B, 196, D)

        n = int(patches.shape[1] ** 0.5)                # 14 for 224/16
        feat = self.feat_proj(patches)                   # (B, 196, 512)
        feat = feat.permute(0, 2, 1).reshape(B, 512, n, n)  # (B, 512, 14, 14)

        x = self.dec1(feat, cond)    # (B, 256, 28,  28)
        x = self.dec2(x,    cond)    # (B, 128, 56,  56)
        x = self.dec3(x,    cond)    # (B,  64, 112, 112)
        x = self.dec4(x,    cond)    # (B,  32, 224, 224)
        return self.out_conv(x)      # (B,   3, 224, 224)


# ---------------------------------------------------------------------------
# Perceptual loss (VGG16 relu features)
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """L1 distance on VGG16 feature activations (relu_1_2 / relu_2_2 / relu_3_3)."""

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as tv_models
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        feats = vgg.features
        self.s1 = nn.Sequential(*list(feats.children())[:4]).train(False)
        self.s2 = nn.Sequential(*list(feats.children())[:9]).train(False)
        self.s3 = nn.Sequential(*list(feats.children())[:16]).train(False)
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x is in [-1, 1]; convert to [0, 1] then normalise to ImageNet stats
        x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p, t = self._preprocess(pred), self._preprocess(target)
        loss = (
            nn.functional.l1_loss(self.s1(p), self.s1(t))
            + nn.functional.l1_loss(self.s2(p), self.s2(t))
            + nn.functional.l1_loss(self.s3(p), self.s3(t))
        )
        return loss / 3.0
