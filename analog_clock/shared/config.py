"""
Shared configuration for GAN dataset generation.

BaseConfig holds defaults shared by both sketch-cGAN and inpainting pipelines.
Each pipeline subclasses it to override only what differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class BaseConfig:
    """Immutable configuration shared across both GAN dataset pipelines."""

    # --- Paths ---
    DATA_DIR: str = "./data"
    BACKGROUNDS_DIR: str = "./data/backgrounds"
    OUTPUT_DIR: str = "./dataset"

    # --- Asset download ---
    DOWNLOAD_KAGGLE: bool = True
    DOWNLOAD_PICSUM: bool = True
    NUM_TEXTURES_TO_DOWNLOAD: int = 50
    KAGGLE_DATASET: str = "roustoumabdelmoula/textures-dataset"

    # --- Dataset parameters ---
    N_SAMPLES: int = 20000
    TRAIN_SPLIT: float = 0.8

    # --- Scene geometry ---
    SCENE_WIDTH_RANGE: Tuple[int, int] = (480, 1024)
    SCENE_HEIGHT_RANGE: Tuple[int, int] = (480, 1024)
    CROP_SIZE: int = 256

    # --- Clock geometry ---
    MIN_RADIUS_RATIO: float = 0.15
    MAX_RADIUS_RATIO: float = 0.35
    CROP_PADDING_RATIO: float = 0.15
    MAX_CENTER_OFFSET: int = 0
    SHOW_NUMBERS_PROB: float = 0.85

    # --- Clock face ---
    SOLID_FACE_PROB: float = 0.50

    # --- Second hand ---
    INCLUDE_SECOND_HAND_PROB: float = 0.0  # overridden by inpainting config

    # --- Augmentation (geometric) ---
    PERSPECTIVE_SCALE: Tuple[float, float] = (0.01, 0.03)
    ROTATION_RANGE: Tuple[int, int] = (-10, 10)
    SCALE_RANGE: Tuple[float, float] = (0.95, 1.0)

    # --- Augmentation (visual) ---
    GAUSSIAN_NOISE_VAR: Tuple[float, float] = (1.0, 3.5)
    ISO_NOISE_INTENSITY: Tuple[float, float] = (0.05, 0.07)
    GAUSSIAN_BLUR_LIMIT: int = 3
    JPEG_QUALITY_RANGE: Tuple[int, int] = (88, 100)
    BRIGHTNESS_LIMIT: float = 0.10
    CONTRAST_LIMIT: float = 0.10
    GLARE_PROBABILITY: float = 0.001

    # --- Color jitter ---
    HUE_SHIFT_LIMIT: int = 10
    SATURATION_SHIFT_LIMIT: int = 15

    # --- "In the wild" realism ---
    SHADOW_PROB: float = 0.40
    GLARE_SPOT_PROB: float = 0.25
    LIGHTING_GRADIENT_PROB: float = 0.30
    FRAME_SHADOW_PROB: float = 0.35
    GLASS_REFLECTION_PROB: float = 0.15
    FACE_YELLOWING_PROB: float = 0.20
    CENTER_JITTER_PX: int = 3


@dataclass(frozen=True)
class SketchConfig(BaseConfig):
    """Config overrides for the sketch-cGAN pipeline."""

    OUTPUT_DIR: str = "./dataset"
    YOLO_DIR: str = "./dataset/yolo"
    CGAN_DIR: str = "./dataset/cgan"

    INCLUDE_SECOND_HAND_PROB: float = 0.0
    SOLID_FACE_PROB: float = 0.30


@dataclass(frozen=True)
class InpaintConfig(BaseConfig):
    """Config overrides for the inpainting pipeline."""

    OUTPUT_DIR: str = "./dataset"
    YOLO_DIR: str = "./dataset/yolo_seg"
    INPAINT_DIR: str = "./dataset/inpainting"

    INCLUDE_SECOND_HAND_PROB: float = 0.50
    SOLID_FACE_PROB: float = 0.50

    # Ultra-clean augmentation for inpainting (needs pixel-precise masks)
    PERSPECTIVE_SCALE: Tuple[float, float] = (0.005, 0.015)
    ROTATION_RANGE: Tuple[int, int] = (-5, 5)
    SCALE_RANGE: Tuple[float, float] = (0.98, 1.0)
    GAUSSIAN_NOISE_VAR: Tuple[float, float] = (0.0, 1.5)
    ISO_NOISE_INTENSITY: Tuple[float, float] = (0.01, 0.04)
    JPEG_QUALITY_RANGE: Tuple[int, int] = (95, 100)
    BRIGHTNESS_LIMIT: float = 0.05
    CONTRAST_LIMIT: float = 0.05
    GLARE_PROBABILITY: float = 0.0
