"""Tests for analog_clock.shared.config."""

from __future__ import annotations

import pytest

from analog_clock.shared.config import BaseConfig, InpaintConfig, SketchConfig


class TestBaseConfig:
    def test_frozen(self) -> None:
        cfg = BaseConfig()
        with pytest.raises(AttributeError):
            cfg.N_SAMPLES = 999  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = BaseConfig()
        assert cfg.N_SAMPLES == 20000
        assert cfg.CROP_SIZE == 256
        assert 0 < cfg.TRAIN_SPLIT < 1


class TestSketchConfig:
    def test_inherits_base(self) -> None:
        cfg = SketchConfig()
        assert cfg.N_SAMPLES == 20000
        assert cfg.CROP_SIZE == 256

    def test_overrides(self) -> None:
        cfg = SketchConfig()
        assert cfg.INCLUDE_SECOND_HAND_PROB == 0.0
        assert cfg.SOLID_FACE_PROB == 0.30
        assert hasattr(cfg, "CGAN_DIR")


class TestInpaintConfig:
    def test_overrides(self) -> None:
        cfg = InpaintConfig()
        assert cfg.INCLUDE_SECOND_HAND_PROB == 0.50
        assert cfg.SOLID_FACE_PROB == 0.50
        assert hasattr(cfg, "INPAINT_DIR")

    def test_ultra_clean_augmentation(self) -> None:
        cfg = InpaintConfig()
        base = BaseConfig()
        # Inpainting should have tighter augmentation ranges
        assert cfg.ROTATION_RANGE[1] <= base.ROTATION_RANGE[1]
        assert cfg.GAUSSIAN_NOISE_VAR[1] <= base.GAUSSIAN_NOISE_VAR[1]
