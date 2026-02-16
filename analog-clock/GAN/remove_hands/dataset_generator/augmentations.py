# ==============================================================================
# FILE: augmentations.py
# Augmentation pipeline supporting Images + Multiple Masks
# ==============================================================================

import albumentations as A
import cv2
import numpy as np
from config import Config


class AugmentationPipeline:
    def __init__(self):
        self.config = Config()
        self._init_pipelines()

    def _init_pipelines(self):
        # Combined pipeline: Geometric + Visual
        # Designed to work on (Image, Target, Mask1, Mask2, Mask3...)
        self.pipeline = A.Compose([
            # Geometric (Applied to Image and ALL Masks)
            A.Perspective(scale=self.config.PERSPECTIVE_SCALE, keep_size=True, p=0.4),
            A.Affine(rotate=self.config.ROTATION_RANGE, scale=self.config.SCALE_RANGE, p=0.5),

            # Visual (Applied ONLY to the Image, not masks)
            # Ultra mild settings
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config.BRIGHTNESS_LIMIT,
                    contrast_limit=self.config.CONTRAST_LIMIT,
                    p=1),
                A.RandomGamma(p=1),
            ], p=0.3),

            A.OneOf([
                A.GaussNoise(var_limit=self.config.GAUSSIAN_NOISE_VAR, p=1),
                A.ISONoise(intensity=self.config.ISO_NOISE_INTENSITY, p=1),
            ], p=0.2),

            A.OneOf([
                A.GaussianBlur(blur_limit=self.config.GAUSSIAN_BLUR_LIMIT, p=1),
                A.ImageCompression(
                    quality_lower=self.config.JPEG_QUALITY_RANGE[0],
                    quality_upper=self.config.JPEG_QUALITY_RANGE[1],
                    p=1),
            ], p=0.2),

        ], additional_targets={'target': 'image'})
        # Note: 'masks' are handled automatically by A.Compose if passed as a list

    def apply(self, image, target, masks_list):
        """
        Apply augmentations to:
        1. Image (Visual + Geometric)
        2. Target (Geometric only - it's the clean image)
        3. Masks List (Geometric only)
        """
        # Convert to RGB
        if image.shape[2] == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if target.shape[2] == 3: target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Albumentations expects masks to be passed as 'masks' argument
        # We assume masks_list contains [mask_combined, mask_hour, mask_minute]
        res = self.pipeline(image=image, target=target, masks=masks_list)

        return res['image'], res['target'], res['masks']

    def to_bgr(self, img_rgb):
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)