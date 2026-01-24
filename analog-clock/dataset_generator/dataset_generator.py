# ==============================================================================
# FILE: dataset_generator.py
# Dual-Output Generator with DYNAMIC SCENE RESOLUTION
# ==============================================================================

import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm
from config import Config
from procedural_generator import AssetManager, ProceduralClockGenerator
from clock_renderer import ClockRenderer
from sketch_generator import SketchGenerator
from augmentations import AugmentationPipeline


class ClockDatasetGenerator:
    def __init__(self):
        self.config = Config()
        self.output_dir = Path(self.config.OUTPUT_DIR)
        self.yolo_dir = Path(self.config.YOLO_DIR)
        self.cgan_dir = Path(self.config.CGAN_DIR)

        print(f"Initializing Dual-Stage Generator...")
        print(f"  - YOLO Scene Range: {self.config.SCENE_WIDTH_RANGE} x {self.config.SCENE_HEIGHT_RANGE}")
        print(f"  - cGAN Crop Size:   {self.config.CROP_SIZE}x{self.config.CROP_SIZE}")

        self.asset_mgr = AssetManager()
        self.asset_mgr.prepare_assets()
        self.textures = self.asset_mgr.get_all_images()

        # Generators (No fixed size in init anymore for clock_gen)
        self.clock_gen = ProceduralClockGenerator()
        self.renderer = ClockRenderer()  # Renderer is now stateless regarding size
        self.augmentor = AugmentationPipeline()

        # Sketch generator uses fixed CROP_SIZE because sketches are for the cGAN part
        self.sketch_gen = SketchGenerator(self.config.CROP_SIZE)

        self._create_output_dirs()

    def _create_output_dirs(self):
        # YOLO
        for split in ['train', 'val']:
            (self.yolo_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        # cGAN
        for split in ['train', 'val']:
            for subdir in ['source', 'target', 'sketch']:
                (self.cgan_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    def _get_yolo_bbox(self, cx, cy, radius, img_w, img_h):
        """Returns normalized YOLO bbox: x_center y_center width height"""
        w = radius * 2
        h = radius * 2
        nx = cx / img_w
        ny = cy / img_h
        nw = w / img_w
        nh = h / img_h
        return 0, nx, ny, nw, nh

    def _crop_and_resize(self, img, cx, cy, radius):
        """Crops the clock with padding and resizes to CROP_SIZE"""
        padding = int(radius * self.config.CROP_PADDING_RATIO)
        crop_r = radius + padding
        d = crop_r * 2

        x1 = max(0, cx - crop_r)
        y1 = max(0, cy - crop_r)
        x2 = min(img.shape[1], cx + crop_r)
        y2 = min(img.shape[0], cy + crop_r)

        crop = img[y1:y2, x1:x2]

        # Pad to square if clipped
        h, w = crop.shape[:2]
        if h != w or h != d:
            square = np.zeros((d, d, 3), dtype=np.uint8)
            sy = (d - h) // 2
            sx = (d - w) // 2
            square[sy:sy + h, sx:sx + w] = crop
            crop = square

        resized = cv2.resize(crop, (self.config.CROP_SIZE, self.config.CROP_SIZE), interpolation=cv2.INTER_AREA)
        return resized

    def generate_sample(self, idx: int, split: str = 'train'):
        # === STEP 1: DYNAMIC SCENE GENERATION ===

        # 1. Randomize Scene Dimensions
        w_range = self.config.SCENE_WIDTH_RANGE
        h_range = self.config.SCENE_HEIGHT_RANGE

        scene_w = random.randint(w_range[0], w_range[1])
        scene_h = random.randint(h_range[0], h_range[1])
        scene_size = (scene_w, scene_h)

        # 2. Randomize Clock Geometry relative to this scene size
        min_dim = min(scene_w, scene_h)
        r_min = int(min_dim * self.config.MIN_RADIUS_RATIO)
        r_max = int(min_dim * self.config.MAX_RADIUS_RATIO)
        radius = random.randint(r_min, r_max)

        # Position
        margin = int(radius * 1.2)
        cx = random.randint(margin, scene_w - margin)
        cy = random.randint(margin, scene_h - margin)
        center = (cx, cy)

        # 3. Textures
        tex_wall = random.choice(self.textures)
        tex_face = random.choice(self.textures)

        # 4. Generate Base & Hands (Pass scene_size!)
        base_img = self.clock_gen.create_clock_on_wall(tex_wall, tex_face, center, radius, scene_size)
        h_hand_img, m_hand_img = self.clock_gen.generate_hand_set(center, radius, scene_size)

        # 5. Times
        t1_h, t1_m = random.randint(0, 23), random.randint(0, 59)
        t2_h, t2_m = random.randint(0, 23), random.randint(0, 59)

        # 6. Render Full Scenes
        scene_source_clean = self.renderer.composite_hands(base_img, h_hand_img, m_hand_img, t1_h, t1_m, center)
        scene_target_clean = self.renderer.composite_hands(base_img, h_hand_img, m_hand_img, t2_h, t2_m, center)

        # 7. Apply Augmentation (To full scene)
        scene_source_aug, scene_target_aug = self.augmentor.apply_shared_geometric(scene_source_clean,
                                                                                   scene_target_clean)

        # === STEP 2: YOLO OUTPUT ===
        fname = f"{idx:06d}"
        yolo_img_path = self.yolo_dir / 'images' / split / f"{fname}.jpg"
        yolo_lbl_path = self.yolo_dir / 'labels' / split / f"{fname}.txt"

        cv2.imwrite(str(yolo_img_path), self.augmentor.to_bgr(scene_source_aug))

        cls, nx, ny, nw, nh = self._get_yolo_bbox(cx, cy, radius, scene_w, scene_h)
        with open(yolo_lbl_path, 'w') as f:
            f.write(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

        # === STEP 3: cGAN OUTPUT (Fixed 256x256) ===
        crop_source = self._crop_and_resize(scene_source_aug, cx, cy, radius)
        crop_target = self._crop_and_resize(scene_target_aug, cx, cy, radius)

        sketch = self.sketch_gen.draw_analog_clock(t2_h, t2_m)
        if len(sketch.shape) == 2:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(self.cgan_dir / split / 'source' / f"{fname}.png"), self.augmentor.to_bgr(crop_source))
        cv2.imwrite(str(self.cgan_dir / split / 'target' / f"{fname}.png"), self.augmentor.to_bgr(crop_target))
        cv2.imwrite(str(self.cgan_dir / split / 'sketch' / f"{fname}.png"), sketch)

        return {
            'filename': fname,
            'split': split,
            'scene_w': scene_w, 'scene_h': scene_h,
            'bbox': (nx, ny, nw, nh)
        }

    def generate_dataset(self):
        total = self.config.N_SAMPLES
        n_train = int(total * self.config.TRAIN_SPLIT)
        n_val = total - n_train

        print(f"\n🚀 Starting Dual-Stage Generation (Total: {total})")
        print(f"   [1] YOLO: Dynamic Sizes ({self.config.SCENE_WIDTH_RANGE[0]}-{self.config.SCENE_WIDTH_RANGE[1]})")
        print(f"   [2] cGAN: Fixed Crop ({self.config.CROP_SIZE}x{self.config.CROP_SIZE})")
        print("-" * 50)

        meta = []
        for i in tqdm(range(n_train), desc="Train"):
            meta.append(self.generate_sample(i, 'train'))
        for i in tqdm(range(n_val), desc="Val  "):
            meta.append(self.generate_sample(i, 'val'))

        pd.DataFrame(meta).to_csv(self.output_dir / 'metadata_full.csv', index=False)
        self._create_yolo_yaml()
        print(f"\n✅ Generation Complete!")

    def _create_yolo_yaml(self):
        yaml_content = f"""
path: {self.yolo_dir.absolute()}
train: images/train
val: images/val
nc: 1
names: ['analog_clock']
"""
        with open(self.yolo_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)