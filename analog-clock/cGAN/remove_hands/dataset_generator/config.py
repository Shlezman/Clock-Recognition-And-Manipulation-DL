# ==============================================================================
# FILE: config.py
# ==============================================================================

class Config:
    """Configuration for YOLOv8-seg and Inpainting (cGAN) dataset generation"""

    # Paths
    DATA_DIR = "./data"
    BACKGROUNDS_DIR = "./data/backgrounds"

    # Output Directories
    OUTPUT_DIR = "./dataset"
    YOLO_DIR = "./dataset/yolo_seg"
    INPAINT_DIR = "./dataset/inpainting"

    SKETCH_CREATOR_PATH = "."

    # Generation Flags
    DOWNLOAD_KAGGLE = True
    DOWNLOAD_PICSUM = True
    NUM_TEXTURES_TO_DOWNLOAD = 50
    KAGGLE_DATASET = "roustoumabdelmoula/textures-dataset"

    # Dataset parameters
    N_SAMPLES = 25000
    TRAIN_SPLIT = 0.8

    # === SIZES ===
    SCENE_WIDTH_RANGE = (480, 1024)
    SCENE_HEIGHT_RANGE = (480, 1024)
    CROP_SIZE = 256

    # Procedural Clock Parameters
    MIN_RADIUS_RATIO = 0.15
    MAX_RADIUS_RATIO = 0.35
    CROP_PADDING_RATIO = 0.15
    MAX_CENTER_OFFSET = 0
    SHOW_NUMBERS_PROB = 0.85

    # New: 40% Chance for a solid color clock face instead of texture
    SOLID_FACE_PROB = 0.40

    # === Augmentation parameters (ULTRA CLEAN) ===
    # Reduced upper limits significantly as requested

    PERSPECTIVE_SCALE = (0.005, 0.015)
    ROTATION_RANGE = (-5, 5)
    SCALE_RANGE = (0.98, 1.0)

    # Visual Quality - Barely visible noise
    # Was (1.0, 5.0) -> Now (0.0, 2.0)
    GAUSSIAN_NOISE_VAR = (0.0, 1.0)

    # Was (0.02, 0.08) -> Now (0.01, 0.04)
    ISO_NOISE_INTENSITY = (0.01, 0.025)

    # Kernel 3 is minimal. Limit stays low.
    GAUSSIAN_BLUR_LIMIT = 3

    # Very High Quality JPEGs only
    JPEG_QUALITY_RANGE = (95, 100)

    BRIGHTNESS_LIMIT = 0.05
    CONTRAST_LIMIT = 0.05
    GLARE_PROBABILITY = 0.05