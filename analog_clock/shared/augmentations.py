"""
Augmentation pipelines for GAN dataset generation.

Unified version that supports:
- Shared geometric + visual transforms for sketch-cGAN (source ↔ target)
- Image + target + multi-mask transforms for inpainting pipeline
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np

from analog_clock.shared.config import BaseConfig


class AugmentationPipeline:
    """Albumentations-based augmentation that applies identical transforms to paired images."""

    def __init__(self, config: BaseConfig | None = None) -> None:
        self.config = config or BaseConfig()
        self._pipeline = self._build_pipeline()

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> A.Compose:
        cfg = self.config
        return A.Compose(
            [
                # --- Geometric (applied to image + all additional targets + masks) ---
                A.Perspective(scale=cfg.PERSPECTIVE_SCALE, keep_size=True, p=0.4),
                A.Affine(
                    rotate=cfg.ROTATION_RANGE,
                    scale=cfg.SCALE_RANGE,
                    p=0.5,
                ),
                # --- Visual / quality ---
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=cfg.BRIGHTNESS_LIMIT,
                            contrast_limit=cfg.CONTRAST_LIMIT,
                            p=1,
                        ),
                        A.RandomGamma(p=1),
                        A.HueSaturationValue(
                            hue_shift_limit=cfg.HUE_SHIFT_LIMIT,
                            sat_shift_limit=cfg.SATURATION_SHIFT_LIMIT,
                            val_shift_limit=0,
                            p=1,
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=cfg.GAUSSIAN_NOISE_VAR, p=1),
                        A.ISONoise(intensity=cfg.ISO_NOISE_INTENSITY, p=1),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=cfg.GAUSSIAN_BLUR_LIMIT, p=1),
                        A.MotionBlur(blur_limit=5, p=1),
                        A.ImageCompression(
                            quality_lower=cfg.JPEG_QUALITY_RANGE[0],
                            quality_upper=cfg.JPEG_QUALITY_RANGE[1],
                            p=1,
                        ),
                    ],
                    p=0.3,
                ),
                # --- "In the wild" colour shifts ---
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2.0, p=1),
                        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=1),
                        A.Emboss(alpha=(0.1, 0.2), strength=(0.3, 0.6), p=1),
                    ],
                    p=0.15,
                ),
                A.RandomShadow(
                    shadow_roi=(0.0, 0.0, 1.0, 1.0),
                    num_shadows_limit=(1, 2),
                    shadow_dimension=5,
                    p=0.15,
                ),
            ],
            additional_targets={"target": "image"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_paired(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply identical transforms to *source* and *target* images (sketch-cGAN)."""
        src_rgb = self._ensure_rgb(source)
        tgt_rgb = self._ensure_rgb(target)
        out = self._pipeline(image=src_rgb, target=tgt_rgb)
        return out["image"], out["target"]

    def apply_with_masks(
        self,
        image: np.ndarray,
        target: np.ndarray,
        masks: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Apply transforms to image + target + a list of binary masks (inpainting)."""
        img_rgb = self._ensure_rgb(image)
        tgt_rgb = self._ensure_rgb(target)
        out = self._pipeline(image=img_rgb, target=tgt_rgb, masks=masks)
        return out["image"], out["target"], out["masks"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def to_bgr(img_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
