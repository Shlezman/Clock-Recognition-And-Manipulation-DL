# ==============================================================================
# FILE: config.py
# ==============================================================================

class Config:
    """Configuration parameters for Dual-Stage (YOLO + cGAN) dataset generation"""

    # Paths
    DATA_DIR = "./data"
    BACKGROUNDS_DIR = "./data/backgrounds"

    # Output Directories
    OUTPUT_DIR = "./dataset"
    YOLO_DIR = "./dataset/yolo"  # For Object Detection
    CGAN_DIR = "./dataset/cgan"  # For Image-to-Image Translation

    SKETCH_CREATOR_PATH = "."

    # Generation Flags
    DOWNLOAD_KAGGLE = True
    DOWNLOAD_PICSUM = True
    NUM_TEXTURES_TO_DOWNLOAD = 50
    KAGGLE_DATASET = "roustoumabdelmoula/textures-dataset"

    # Dataset parameters
    N_SAMPLES = 20000
    TRAIN_SPLIT = 0.8

    # === SIZES (DYNAMIC YOLO SCENES) ===
    # Randomly pick dimensions for each YOLO training image to support robustness
    SCENE_WIDTH_RANGE = (480, 1024)
    SCENE_HEIGHT_RANGE = (480, 1024)

    # CROP_SIZE: The input size for the cGAN model (Fixed Standard)
    CROP_SIZE = 256

    # Procedural Clock Parameters
    # Radius is relative to the *smallest dimension* of the current scene
    MIN_RADIUS_RATIO = 0.15
    MAX_RADIUS_RATIO = 0.35

    CROP_PADDING_RATIO = 0.15
    MAX_CENTER_OFFSET = 0  # Offset logic is handled by the random placement
    SHOW_NUMBERS_PROB = 0.85

    # === Augmentation parameters (MILD) ===
    PERSPECTIVE_SCALE = (0.01, 0.03)
    ROTATION_RANGE = (-10, 10)
    SCALE_RANGE = (0.95, 1.0)

    # Visual Quality
    GAUSSIAN_NOISE_VAR = (1.0, 3.5)
    ISO_NOISE_INTENSITY = (0.05, 0.07)
    GAUSSIAN_BLUR_LIMIT = 2
    JPEG_QUALITY_RANGE = (88, 100)

    BRIGHTNESS_LIMIT = 0.1
    CONTRAST_LIMIT = 0.1
    GLARE_PROBABILITY = 0.001