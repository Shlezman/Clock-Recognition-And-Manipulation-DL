# ==============================================================================
# FILE: augmentations.py
# Augmentation pipelines - MILD SHARED EFFECTS (Fixes UserWarnings)
# ==============================================================================

import albumentations as A
import cv2
import numpy as np
from config import Config


class AugmentationPipeline:
    """
    Manages augmentations.
    Applies both Geometric AND Visual augmentations identically to Source and Target.
    """

    def __init__(self):
        self.config = Config()
        self._init_pipelines()

    def _init_pipelines(self):
        """Initialize augmentation pipelines"""

        # Combined pipeline: Geometric + Visual (Mild)
        self.shared_aug = A.Compose([
            # 1. Geometric (Alignment)
            A.Perspective(
                scale=self.config.PERSPECTIVE_SCALE,
                keep_size=True,
                p=0.4
            ),
            # Removed 'cval=0' as it is the default and caused warnings in some versions
            A.Affine(
                rotate=self.config.ROTATION_RANGE,
                scale=self.config.SCALE_RANGE,
                p=0.5
            ),

            # 2. Visual / Quality (Mild Noise & Blur)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config.BRIGHTNESS_LIMIT,
                    contrast_limit=self.config.CONTRAST_LIMIT,
                    p=1
                ),
                A.RandomGamma(p=1),
            ], p=0.5),

            # Note: var_limit and intensity are standard args.
            # If warnings persist, check albumentations version in requirements.
            A.OneOf([
                A.GaussNoise(var_limit=self.config.GAUSSIAN_NOISE_VAR, p=1),
                A.ISONoise(intensity=self.config.ISO_NOISE_INTENSITY, p=1),
            ], p=0.3),

            A.OneOf([
                A.GaussianBlur(blur_limit=self.config.GAUSSIAN_BLUR_LIMIT, p=1),
                A.ImageCompression(
                    quality_lower=self.config.JPEG_QUALITY_RANGE[0],
                    quality_upper=self.config.JPEG_QUALITY_RANGE[1],
                    p=1
                ),
            ], p=0.3),

        ], additional_targets={'target': 'image'})

    def apply_shared_geometric(self, source_img: np.ndarray, target_img: np.ndarray):
        """
        Apply EXACTLY the same transforms (Geometric + Visual) to both.
        """
        if source_img.shape[2] == 3:
            src_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            tgt_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        else:
            src_rgb = source_img
            tgt_rgb = target_img

        transformed = self.shared_aug(image=src_rgb, target=tgt_rgb)

        return transformed['image'], transformed['target']

    def to_bgr(self, img_rgb):
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)