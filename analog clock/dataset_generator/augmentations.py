# ==============================================================================
# FILE 2: augmentations.py
# Augmentation pipelines for source and target images
# ==============================================================================

import albumentations as A
import cv2
import numpy as np
import random
from config import Config


class AugmentationPipeline:
    """Manages augmentation pipelines for dataset generation"""

    def __init__(self):
        self.config = Config()
        self._init_pipelines()

    def _init_pipelines(self):
        """Initialize augmentation pipelines"""

        # Geometric augmentations (apply to BOTH source and target for alignment)
        self.geometric_aug = A.Compose([
            A.Perspective(
                scale=self.config.PERSPECTIVE_SCALE,
                keep_size=True,
                p=0.5
            ),
            A.Affine(
                rotate=self.config.ROTATION_RANGE,
                scale=self.config.SCALE_RANGE,
                p=0.5
            ),
        ])

        # Visual augmentations (apply mainly to source for difficulty)
        self.visual_aug = A.Compose([
            # Lighting/Contrast variations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1
                ),
                A.RandomToneCurve(scale=0.3, p=1),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=1
                ),
            ], p=0.7),

            # Noise (simulating camera sensor noise)
            A.OneOf([
                A.GaussNoise(var_limit=self.config.GAUSSIAN_NOISE_VAR, p=1),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=self.config.ISO_NOISE_INTENSITY,
                    p=1
                ),
            ], p=0.8),

            # Shadows (simulating cast shadows)
            A.RandomShadow(
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                shadow_roi=(0, 0.5, 1, 1),
                p=0.5
            ),

            # Blur (simulating focus issues)
            A.OneOf([
                A.GaussianBlur(blur_limit=self.config.GAUSSIAN_BLUR_LIMIT, p=1),
                A.MotionBlur(blur_limit=self.config.MOTION_BLUR_LIMIT, p=1),
                A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=1),
            ], p=0.4),

            # JPEG compression artifacts
            A.ImageCompression(
                quality_lower=self.config.JPEG_QUALITY_RANGE[0],
                quality_upper=self.config.JPEG_QUALITY_RANGE[1],
                p=0.5
            ),
        ])

        # Mild augmentations for target (optional, minimal noise)
        self.target_aug = A.Compose([
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        ])

    def add_glare_overlay(self, img: np.ndarray) -> np.ndarray:
        """
        Add random glare/light leak effect to simulate glass reflections.
        This is a custom augmentation not available in albumentations.
        """
        h, w = img.shape[:2]
        overlay = img.copy()

        if random.random() < self.config.GLARE_PROBABILITY:
            # Random position (more likely in upper half)
            center_x = random.randint(0, w)
            center_y = random.randint(0, h // 2)
            radius = random.randint(*self.config.GLARE_RADIUS_RANGE)

            # Create glare mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (99, 99), 0)

            # Blend white glare
            white = np.ones_like(img) * 255
            alpha = mask[:, :, np.newaxis] / 255.0
            alpha *= random.uniform(*self.config.GLARE_ALPHA_RANGE)
            overlay = (img * (1 - alpha) + white * alpha).astype(np.uint8)

        return overlay

    def apply_to_source(self, img: np.ndarray) -> np.ndarray:
        """Apply full augmentation pipeline to source image"""
        # Convert BGR to RGB for albumentations
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Geometric augmentations
        img_rgb = self.geometric_aug(image=img_rgb)['image']

        # Add custom glare effect
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = self.add_glare_overlay(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Visual augmentations
        img_rgb = self.visual_aug(image=img_rgb)['image']

        return img_rgb

    def apply_to_target(self, img: np.ndarray) -> np.ndarray:
        """Apply alignment-preserving augmentations to target image"""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Same geometric augmentations as source for alignment
        img_rgb = self.geometric_aug(image=img_rgb)['image']

        # Optional: very light noise
        img_rgb = self.target_aug(image=img_rgb)['image']

        return img_rgb