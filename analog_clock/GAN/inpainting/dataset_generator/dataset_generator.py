"""
Inpainting dataset generator.

Produces:
  1. YOLOv8-seg polygon labels  (images + txt, 3 classes: hour/minute/second)
  2. Inpainting triplets         (source / mask / target  –  256×256)

Imports all shared logic from ``analog_clock.shared``.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``analog_clock.shared`` resolves
# when this file is executed as a standalone script.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analog_clock.shared.asset_manager import AssetManager
from analog_clock.shared.augmentations import AugmentationPipeline
from analog_clock.shared.clock_renderer import ClockRenderer
from analog_clock.shared.config import InpaintConfig
from analog_clock.shared.procedural_generator import ProceduralClockGenerator

logger = logging.getLogger(__name__)


class ClockDatasetGenerator:
    """Generate inpainting + YOLO-seg datasets with hour/minute/second hands."""

    def __init__(self, config: InpaintConfig | None = None) -> None:
        self.config = config or InpaintConfig()

        self.yolo_dir = Path(self.config.YOLO_DIR)
        self.inpaint_dir = Path(self.config.INPAINT_DIR)

        logger.info(
            "Inpainting dataset generator — YOLO-seg 3-class, inpaint crop %d×%d",
            self.config.CROP_SIZE,
            self.config.CROP_SIZE,
        )

        self.asset_mgr = AssetManager(self.config)
        self.asset_mgr.prepare_assets()
        self.textures = self.asset_mgr.get_all_images()

        self.clock_gen = ProceduralClockGenerator(self.config)
        self.renderer = ClockRenderer()
        self.augmentor = AugmentationPipeline(self.config)

        self._create_output_dirs()

    # ------------------------------------------------------------------
    # Directory scaffolding
    # ------------------------------------------------------------------

    def _create_output_dirs(self) -> None:
        for split in ("train", "val"):
            (self.yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            for subdir in ("source", "mask", "target"):
                (self.inpaint_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
        """Convert a binary mask to normalised YOLO polygon coordinates."""
        height, width = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 20:
                continue
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            poly_norm: list[float] = []
            for point in approx:
                x, y = point[0]
                poly_norm.append(max(0.0, min(1.0, x / width)))
                poly_norm.append(max(0.0, min(1.0, y / height)))
            if len(poly_norm) >= 6:
                polygons.append(poly_norm)
        return polygons

    def _crop_and_resize(
        self,
        img: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        interpolation: int = cv2.INTER_AREA,
    ) -> np.ndarray:
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
            channels = 3 if img.ndim == 3 else 0
            if channels:
                square = np.zeros((d, d, 3), dtype=np.uint8)
            else:
                square = np.zeros((d, d), dtype=np.uint8)
            sy, sx = (d - h) // 2, (d - w) // 2
            square[sy : sy + h, sx : sx + w] = crop
            crop = square

        resized = cv2.resize(
            crop,
            (self.config.CROP_SIZE, self.config.CROP_SIZE),
            interpolation=interpolation,
        )
        # Binarise masks after resize to avoid interpolation artefacts
        if img.ndim == 2:
            _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        return resized

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def generate_sample(self, idx: int, split: str = "train") -> None:
        cfg = self.config

        # 1. Random scene
        scene_w = random.randint(*cfg.SCENE_WIDTH_RANGE)
        scene_h = random.randint(*cfg.SCENE_HEIGHT_RANGE)
        scene_size = (scene_w, scene_h)

        # 2. Clock geometry (centered)
        min_dim = min(scene_w, scene_h)
        radius = random.randint(
            int(min_dim * cfg.MIN_RADIUS_RATIO),
            int(min_dim * cfg.MAX_RADIUS_RATIO),
        )
        cx, cy = scene_w // 2, scene_h // 2
        center = (cx, cy)

        # 3. Base image
        tex_wall = random.choice(self.textures)
        tex_face = random.choice(self.textures)
        clean_bg = self.clock_gen.create_clock_on_wall(
            tex_wall, tex_face, center, radius, scene_size
        )

        # 4. Hands (hour + minute + optional second)
        hand_set = self.clock_gen.generate_hand_set(center, radius, scene_size)
        h_hand, m_hand = hand_set[0], hand_set[1]
        s_hand = hand_set[2] if len(hand_set) == 3 else None

        t_h, t_m = random.randint(0, 23), random.randint(0, 59)
        t_s = random.randint(0, 59)

        # 5. Render scene
        scene_with_hands = self.renderer.composite_hands(
            clean_bg, h_hand, m_hand, t_h, t_m, center, s_hand, t_s
        )

        # 6. Per-hand masks
        rot_h = self.renderer.rotate_hand(h_hand, (t_h % 12) * 30 + t_m * 0.5, center)
        mask_h = rot_h[:, :, 3]

        rot_m = self.renderer.rotate_hand(m_hand, t_m * 6, center)
        mask_m = rot_m[:, :, 3]

        if s_hand is not None:
            rot_s = self.renderer.rotate_hand(s_hand, t_s * 6, center)
            mask_s = rot_s[:, :, 3]
            mask_combined = cv2.bitwise_or(cv2.bitwise_or(mask_h, mask_m), mask_s)
        else:
            mask_s = np.zeros_like(mask_h)
            mask_combined = cv2.bitwise_or(mask_h, mask_m)

        kernel = np.ones((3, 3), np.uint8)
        mask_combined_dilated = cv2.dilate(mask_combined, kernel, iterations=1)

        # 7. Augmentation (image + clean bg + 4 masks)
        aug_img, aug_clean, aug_masks = self.augmentor.apply_with_masks(
            scene_with_hands,
            clean_bg,
            [mask_h, mask_m, mask_s, mask_combined_dilated],
        )
        aug_mask_h, aug_mask_m, aug_mask_s, aug_mask_combined = aug_masks

        # === YOLO-seg ===
        fname = f"{idx:06d}"
        cv2.imwrite(
            str(self.yolo_dir / "images" / split / f"{fname}.jpg"),
            self.augmentor.to_bgr(aug_img),
        )

        lbl_path = self.yolo_dir / "labels" / split / f"{fname}.txt"
        lines: list[str] = []
        for cls_id, mask in ((0, aug_mask_h), (1, aug_mask_m)):
            for poly in self._mask_to_polygons(mask):
                lines.append(f"{cls_id} {' '.join(f'{c:.6f}' for c in poly)}")
        if s_hand is not None:
            for poly in self._mask_to_polygons(aug_mask_s):
                lines.append(f"2 {' '.join(f'{c:.6f}' for c in poly)}")
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        # === Inpainting ===
        crop_src = self._crop_and_resize(aug_img, cx, cy, radius)
        crop_tgt = self._crop_and_resize(aug_clean, cx, cy, radius)
        crop_mask = self._crop_and_resize(
            aug_mask_combined, cx, cy, radius, interpolation=cv2.INTER_NEAREST
        )

        cv2.imwrite(
            str(self.inpaint_dir / split / "source" / f"{fname}.png"),
            self.augmentor.to_bgr(crop_src),
        )
        cv2.imwrite(
            str(self.inpaint_dir / split / "target" / f"{fname}.png"),
            self.augmentor.to_bgr(crop_tgt),
        )
        cv2.imwrite(
            str(self.inpaint_dir / split / "mask" / f"{fname}.png"),
            crop_mask,
        )

    # ------------------------------------------------------------------
    # Full dataset
    # ------------------------------------------------------------------

    def generate_dataset(self) -> None:
        total = self.config.N_SAMPLES
        n_train = int(total * self.config.TRAIN_SPLIT)
        n_val = total - n_train

        logger.info(
            "Generating %d samples (%d train / %d val)", total, n_train, n_val
        )

        for i in tqdm(range(n_train), desc="Train"):
            self.generate_sample(i, "train")
        for i in tqdm(range(n_val), desc="Val  "):
            self.generate_sample(i, "val")

        self._create_yolo_yaml()
        logger.info("Inpainting dataset generation complete.")

    def _create_yolo_yaml(self) -> None:
        yaml_content = (
            f"path: {self.yolo_dir.absolute()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 3\n"
            f"names: ['hour_hand', 'minute_hand', 'second_hand']\n"
        )
        (self.yolo_dir / "dataset.yaml").write_text(yaml_content)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = ClockDatasetGenerator()
    gen.generate_dataset()
