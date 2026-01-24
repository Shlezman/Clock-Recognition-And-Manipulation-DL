# ==============================================================================
# FILE 1: config.py
# Configuration settings for the dataset generator
# ==============================================================================

from pathlib import Path


class Config:
    """Configuration parameters for clock dataset generation"""

    # Paths
    BACKGROUNDS_DIR = "./data/textures-dataset"  # Kaggle textures
    CLEAN_CLOCKS_DIR = "./data/clean_clocks"  # Inpainted clock faces
    HANDS_DIR = "./data/clock_hands"  # Hand PNG assets
    OUTPUT_DIR = "./dataset"  # Output directory

    # Reference repository integration
    REFERENCE_REPO_PATH = "./Reading-analog-clocks-with-neural-networks"

    # Dataset parameters
    N_SAMPLES = 20000
    TRAIN_SPLIT = 0.9
    IMAGE_SIZE = 256
    USE_TEXTURES_AS_BG = True  # Use texture dataset as backgrounds

    # Augmentation parameters
    PERSPECTIVE_SCALE = (0.05, 0.1)
    ROTATION_RANGE = (-15, 15)
    SCALE_RANGE = (0.95, 1.05)

    # Noise parameters
    GAUSSIAN_NOISE_VAR = (10.0, 50.0)
    ISO_NOISE_INTENSITY = (0.1, 0.5)

    # Blur parameters
    GAUSSIAN_BLUR_LIMIT = (3, 7)
    MOTION_BLUR_LIMIT = 7

    # JPEG compression
    JPEG_QUALITY_RANGE = (60, 95)

    # Glare parameters
    GLARE_PROBABILITY = 0.5
    GLARE_RADIUS_RANGE = (50, 150)
    GLARE_ALPHA_RANGE = (0.3, 0.7)