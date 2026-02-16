# ==============================================================================
# FILE: dataset_generator.py
# Generators for:
# 1. YOLOv8-seg (Centered Clock, Dynamic Res, Safe Polygons)
# 2. cGAN Inpainting (Centered Crop 256x256)
# Supports: 3 Classes (Hour, Minute, Second)
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
from augmentations import AugmentationPipeline


class ClockDatasetGenerator:
    def __init__(self):
        self.config = Config()

        # Directories
        self.yolo_dir = Path(self.config.YOLO_DIR)
        self.inpaint_dir = Path(self.config.INPAINT_DIR)

        print(f"Initializing Generator...")
        print(f"  - YOLO-seg: Centered Clock, Safe Polygons")
        print(f"  - Inpainting: Fixed {self.config.CROP_SIZE}x{self.config.CROP_SIZE} Crop")

        self.asset_mgr = AssetManager()
        self.asset_mgr.prepare_assets()
        self.textures = self.asset_mgr.get_all_images()

        self.clock_gen = ProceduralClockGenerator()
        self.renderer = ClockRenderer()
        self.augmentor = AugmentationPipeline()

        self._create_output_dirs()

    def _create_output_dirs(self):
        # YOLO-seg
        for split in ['train', 'val']:
            (self.yolo_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        # Inpainting
        for split in ['train', 'val']:
            for subdir in ['source', 'mask', 'target']:
                (self.inpaint_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    def _mask_to_polygons(self, mask):
        """Converts binary mask to normalized YOLO polygons"""
        height, width = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                poly_norm = []
                for point in approx:
                    x, y = point[0]
                    nx = max(0.0, min(1.0, x / width))
                    ny = max(0.0, min(1.0, y / height))
                    poly_norm.append(nx)
                    poly_norm.append(ny)
                if len(poly_norm) >= 6:
                    polygons.append(poly_norm)
        return polygons

    def _crop_and_resize(self, img, cx, cy, radius, interpolation=cv2.INTER_AREA):
        padding = int(radius * self.config.CROP_PADDING_RATIO)
        crop_r = radius + padding
        d = crop_r * 2
        x1 = max(0, cx - crop_r)
        y1 = max(0, cy - crop_r)
        x2 = min(img.shape[1], cx + crop_r)
        y2 = min(img.shape[0], cy + crop_r)
        crop = img[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h != w or h != d:
            if len(img.shape) == 3:
                square = np.zeros((d, d, 3), dtype=np.uint8)
            else:
                square = np.zeros((d, d), dtype=np.uint8)
            sy = (d - h) // 2
            sx = (d - w) // 2
            square[sy:sy + h, sx:sx + w] = crop
            crop = square
        resized = cv2.resize(crop, (self.config.CROP_SIZE, self.config.CROP_SIZE), interpolation=interpolation)
        if len(img.shape) == 2:
            _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        return resized

    def generate_sample(self, idx: int, split: str = 'train'):
        # 1. Randomize Scene
        scene_w = random.randint(*self.config.SCENE_WIDTH_RANGE)
        scene_h = random.randint(*self.config.SCENE_HEIGHT_RANGE)
        scene_size = (scene_w, scene_h)

        # 2. Geometry: CENTERED
        min_dim = min(scene_w, scene_h)
        r_min = int(min_dim * self.config.MIN_RADIUS_RATIO)
        r_max = int(min_dim * self.config.MAX_RADIUS_RATIO)
        radius = random.randint(r_min, r_max)
        cx, cy = scene_w // 2, scene_h // 2
        center = (cx, cy)

        # 3. Base Image
        tex_wall = random.choice(self.textures)
        tex_face = random.choice(self.textures)
        clean_bg = self.clock_gen.create_clock_on_wall(tex_wall, tex_face, center, radius, scene_size)

        # 4. Hands (Now includes Second Hand)
        h_hand_img, m_hand_img, s_hand_img = self.clock_gen.generate_hand_set(center, radius, scene_size)
        t_h, t_m = random.randint(0, 23), random.randint(0, 59)
        t_s = random.randint(0, 59)

        # 5. Render Scene
        scene_with_hands = self.renderer.composite_hands(
            clean_bg, h_hand_img, m_hand_img, t_h, t_m, center, s_hand_img, t_s
        )

        # 6. Extract Masks
        # Hour
        rot_h = self.renderer.rotate_hand(h_hand_img, (t_h % 12) * 30 + t_m * 0.5, center)
        mask_h = rot_h[:, :, 3]
        # Minute
        rot_m = self.renderer.rotate_hand(m_hand_img, t_m * 6, center)
        mask_m = rot_m[:, :, 3]
        # Second (Handle if exists)
        if s_hand_img is not None:
            rot_s = self.renderer.rotate_hand(s_hand_img, t_s * 6, center)
            mask_s = rot_s[:, :, 3]
            # Combined for Inpainting (All hands)
            mask_combined = cv2.bitwise_or(mask_h, mask_m)
            mask_combined = cv2.bitwise_or(mask_combined, mask_s)
        else:
            mask_s = np.zeros_like(mask_h)  # Empty mask
            mask_combined = cv2.bitwise_or(mask_h, mask_m)

        kernel = np.ones((3, 3), np.uint8)
        mask_combined_dilated = cv2.dilate(mask_combined, kernel, iterations=1)

        # 7. Augmentation
        # We pass 4 masks: [H, M, S, Combined]
        aug_img, aug_clean, aug_masks = self.augmentor.apply(
            scene_with_hands,
            clean_bg,
            [mask_h, mask_m, mask_s, mask_combined_dilated]
        )
        aug_mask_h = aug_masks[0]
        aug_mask_m = aug_masks[1]
        aug_mask_s = aug_masks[2]
        aug_mask_combined = aug_masks[3]

        # === DATASET 1: YOLO-seg ===
        fname = f"{idx:06d}"
        yolo_img_path = self.yolo_dir / 'images' / split / f"{fname}.jpg"
        cv2.imwrite(str(yolo_img_path), self.augmentor.to_bgr(aug_img))

        yolo_lbl_path = self.yolo_dir / 'labels' / split / f"{fname}.txt"

        polys_h = self._mask_to_polygons(aug_mask_h)
        polys_m = self._mask_to_polygons(aug_mask_m)
        polys_s = self._mask_to_polygons(aug_mask_s)

        with open(yolo_lbl_path, 'w') as f:
            # Class 0: Hour
            for poly in polys_h:
                f.write(f"0 {' '.join([f'{c:.6f}' for c in poly])}\n")
            # Class 1: Minute
            for poly in polys_m:
                f.write(f"1 {' '.join([f'{c:.6f}' for c in poly])}\n")
            # Class 2: Second (Only if exists)
            if s_hand_img is not None:
                for poly in polys_s:
                    f.write(f"2 {' '.join([f'{c:.6f}' for c in poly])}\n")

        # === DATASET 2: Inpainting ===
        crop_source = self._crop_and_resize(aug_img, cx, cy, radius)
        crop_target = self._crop_and_resize(aug_clean, cx, cy, radius)
        crop_mask = self._crop_and_resize(aug_mask_combined, cx, cy, radius, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(self.inpaint_dir / split / 'source' / f"{fname}.png"), self.augmentor.to_bgr(crop_source))
        cv2.imwrite(str(self.inpaint_dir / split / 'target' / f"{fname}.png"), self.augmentor.to_bgr(crop_target))
        cv2.imwrite(str(self.inpaint_dir / split / 'mask' / f"{fname}.png"), crop_mask)

        return {}

    def generate_dataset(self):
        total = self.config.N_SAMPLES
        n_train = int(total * self.config.TRAIN_SPLIT)
        n_val = total - n_train

        print(f"\n🚀 Starting Generation (Total: {total})")
        print(f"   [1] YOLO-seg (3 Classes: H, M, S)")
        print(f"   [2] Inpainting (Hands Removed)")
        print("-" * 50)

        for i in tqdm(range(n_train), desc="Train"):
            self.generate_sample(i, 'train')
        for i in tqdm(range(n_val), desc="Val  "):
            self.generate_sample(i, 'val')

        self._create_yolo_yaml()
        print(f"\n✅ Generation Complete!")

    def _create_yolo_yaml(self):
        yaml_content = f"""
path: {self.yolo_dir.absolute()}
train: images/train
val: images/val
nc: 3
names: ['hour_hand', 'minute_hand', 'second_hand']
"""
        with open(self.yolo_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)