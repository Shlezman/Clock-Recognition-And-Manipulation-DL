"""
CNN for predicting clock time from a 256x256 binary hand-mask.

Outputs 4 values: (sin_h, cos_h, sin_m, cos_m) representing the hour and
minute hand angles via sin/cos encoding.  This avoids discontinuities at
the 0/360-degree boundary that plague naive regression or classification.

Angle recovery:
    hour_angle  = atan2(sin_h, cos_h)          # radians, [0, 2*pi)
    minute_angle = atan2(sin_m, cos_m)

    hour   = hour_angle   / (2*pi) * 12        # [0, 12)
    minute = minute_angle / (2*pi) * 60         # [0, 60)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> optional MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ClockHandCNN(nn.Module):
    """
    Lightweight CNN for clock-hand angle regression.

    Input : (B, 1, 256, 256) binary mask
    Output: (B, 4) -> [sin_h, cos_h, sin_m, cos_m]

    Architecture (6 conv blocks with pooling):
        256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
    followed by global average pooling and two FC layers.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32),    # 256 -> 128
            ConvBlock(32, 64),   # 128 -> 64
            ConvBlock(64, 128),  # 64  -> 32
            ConvBlock(128, 256), # 32  -> 16
            ConvBlock(256, 256), # 16  -> 8
            ConvBlock(256, 256), # 8   -> 4
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, 256, 4, 4) -> (B, 256, 1, 1)
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),       # sin_h, cos_h, sin_m, cos_m
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
