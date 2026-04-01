"""Tests for GAN model architectures (forward pass shape checks)."""

from __future__ import annotations

import pytest
import torch

from analog_clock.GAN.inpainting.generator_model import InpaintDiscriminator, InpaintGenerator
from analog_clock.GAN.sketch.generator_model import GeneratorUNet, SketchDiscriminator


class TestGeneratorUNet:
    def test_output_shape(self) -> None:
        model = GeneratorUNet(in_channels=6, out_channels=3)
        x = torch.randn(1, 6, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 256, 256)

    def test_output_range_tanh(self) -> None:
        model = GeneratorUNet()
        x = torch.randn(1, 6, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestInpaintGenerator:
    def test_output_shape(self) -> None:
        model = InpaintGenerator(in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 256, 256)

    def test_output_range_tanh(self) -> None:
        model = InpaintGenerator()
        x = torch.randn(1, 4, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_batch_dimension(self) -> None:
        model = InpaintGenerator()
        x = torch.randn(4, 4, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 4


class TestSketchDiscriminator:
    def test_output_shape(self) -> None:
        model = SketchDiscriminator(in_channels=9)
        src = torch.randn(1, 3, 256, 256)
        skc = torch.randn(1, 3, 256, 256)
        tgt = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(src, skc, tgt)
        # PatchGAN output
        assert out.dim() == 4
        assert out.shape[0] == 1

    def test_has_spectral_norm(self) -> None:
        model = SketchDiscriminator()
        sn_count = sum(
            1 for m in model.modules()
            if hasattr(m, "weight_orig")  # spectral_norm adds weight_orig
        )
        assert sn_count >= 4, f"Expected >=4 spectrally-normed layers, got {sn_count}"


class TestInpaintDiscriminator:
    def test_output_shape(self) -> None:
        model = InpaintDiscriminator(in_channels=7)
        inp = torch.randn(1, 4, 256, 256)
        tgt = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(inp, tgt)
        assert out.dim() == 4
        assert out.shape[0] == 1

    def test_has_spectral_norm(self) -> None:
        model = InpaintDiscriminator()
        sn_count = sum(
            1 for m in model.modules()
            if hasattr(m, "weight_orig")
        )
        assert sn_count >= 3, f"Expected >=3 spectrally-normed layers, got {sn_count}"
