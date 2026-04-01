"""
Sketch-cGAN dataset generator.

Produces:
  1. YOLO bounding-box labels  (images + txt)
  2. cGAN triplets             (source / target / sketch  –  256×256)

Imports all shared logic from ``analog_clock.shared``.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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
from analog_clock.shared.config import SketchConfig
from analog_clock.shared.procedural_generator import ProceduralClockGenerator
from analog_clock.shared.sketch_generator import SketchGenerator

logger = logging.getLogger(__name__)


class ClockDatasetGenerator:
    """Generate paired sketch-cGAN + YOLO datasets."""

    def __init__(self, config: SketchConfig | None = None) -> None:
        self.config = config or SketchConfig()
        self.output_dir = Path(self.config.OUTPUT_DIR)
        self.yolo_dir = Path(self.config.YOLO_DIR)
        self.cgan_dir = Path(self.config.CGAN_DIR)

        logger.info(
            "Sketch dataset generator — YOLO scenes %s–%s, cGAN crop %d×%d",
            self.config.SCENE_WIDTH_RANGE,
            self.config.SCENE_HEIGHT_RANGE,
            self.config.CROP_SIZE,
            self.config.CROP_SIZE,
        )

        self.asset_mgr = AssetManager(self.config)
        self.asset_mgr.prepare_assets()
        self.textures = self.asset_mgr.get_all_images()

        self.clock_gen = ProceduralClockGenerator(self.config)
        self.renderer = ClockRenderer()
        self.augmentor = AugmentationPipeline(self.config)
        self.sketch_gen = SketchGenerator(self.config.CROP_SIZE)

        self._create_output_dirs()

    # ------------------------------------------------------------------
    # Directory scaffolding
    # ------------------------------------------------------------------

    def _create_output_dirs(self) -> None:
        for split in ("train", "val"):
            (self.yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            for subdir in ("source", "target", "sketch"):
                (self.cgan_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_yolo_bbox(
        cx: int, cy: int, radius: int, img_w: int, img_h: int
    ) -> tuple[int, float, float, float, float]:
        w = radius * 2
        h = radius * 2
        return 0, cx / img_w, cy / img_h, w / img_w, h / img_h

    def _crop_and_resize(
        self, img: np.ndarray, cx: int, cy: int, radius: int
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
            square = np.zeros((d, d, 3), dtype=np.uint8)
            sy, sx = (d - h) // 2, (d - w) // 2
            square[sy : sy + h, sx : sx + w] = crop
            crop = square

        return cv2.resize(
            crop,
            (self.config.CROP_SIZE, self.config.CROP_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def generate_sample(self, idx: int, split: str = "train") -> dict:
        cfg = self.config

        # 1. Random scene dimensions
        scene_w = random.randint(*cfg.SCENE_WIDTH_RANGE)
        scene_h = random.randint(*cfg.SCENE_HEIGHT_RANGE)
        scene_size = (scene_w, scene_h)

        # 2. Clock geometry
        min_dim = min(scene_w, scene_h)
        radius = random.randint(
            int(min_dim * cfg.MIN_RADIUS_RATIO),
            int(min_dim * cfg.MAX_RADIUS_RATIO),
        )
        margin = int(radius * 1.2)
        cx = random.randint(margin, scene_w - margin)
        cy = random.randint(margin, scene_h - margin)
        center = (cx, cy)

        # 3. Textures
        tex_wall = random.choice(self.textures)
        tex_face = random.choice(self.textures)

        # 4. Base image + hands
        base_img = self.clock_gen.create_clock_on_wall(
            tex_wall, tex_face, center, radius, scene_size
        )
        hand_set = self.clock_gen.generate_hand_set(center, radius, scene_size)
        h_hand, m_hand = hand_set[0], hand_set[1]

        # 5. Two random times
        t1_h, t1_m = random.randint(0, 23), random.randint(0, 59)
        t2_h, t2_m = random.randint(0, 23), random.randint(0, 59)

        # 6. Render both scenes
        scene_src = self.renderer.composite_hands(
            base_img, h_hand, m_hand, t1_h, t1_m, center
        )
        scene_tgt = self.renderer.composite_hands(
            base_img, h_hand, m_hand, t2_h, t2_m, center
        )

        # 7. Augment (shared geometric + visual on the pair)
        scene_src_aug, scene_tgt_aug = self.augmentor.apply_paired(scene_src, scene_tgt)

        # === YOLO output ===
        fname = f"{idx:06d}"
        cv2.imwrite(
            str(self.yolo_dir / "images" / split / f"{fname}.jpg"),
            self.augmentor.to_bgr(scene_src_aug),
        )

        cls, nx, ny, nw, nh = self._get_yolo_bbox(cx, cy, radius, scene_w, scene_h)
        lbl_path = self.yolo_dir / "labels" / split / f"{fname}.txt"
        lbl_path.write_text(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

        # === cGAN output (256×256) ===
        crop_src = self._crop_and_resize(scene_src_aug, cx, cy, radius)
        crop_tgt = self._crop_and_resize(scene_tgt_aug, cx, cy, radius)
        sketch = self.sketch_gen.draw_analog_clock(t2_h, t2_m)
        if sketch.ndim == 2:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(
            str(self.cgan_dir / split / "source" / f"{fname}.png"),
            self.augmentor.to_bgr(crop_src),
        )
        cv2.imwrite(
            str(self.cgan_dir / split / "target" / f"{fname}.png"),
            self.augmentor.to_bgr(crop_tgt),
        )
        cv2.imwrite(
            str(self.cgan_dir / split / "sketch" / f"{fname}.png"),
            sketch,
        )

        return {
            "filename": fname,
            "split": split,
            "scene_w": scene_w,
            "scene_h": scene_h,
            "bbox": (nx, ny, nw, nh),
        }

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

        meta: list[dict] = []
        for i in tqdm(range(n_train), desc="Train"):
            meta.append(self.generate_sample(i, "train"))
        for i in tqdm(range(n_val), desc="Val  "):
            meta.append(self.generate_sample(i, "val"))

        pd.DataFrame(meta).to_csv(self.output_dir / "metadata_full.csv", index=False)
        self._create_yolo_yaml()
        logger.info("Sketch dataset generation complete.")

    def _create_yolo_yaml(self) -> None:
        yaml_content = (
            f"path: {self.yolo_dir.absolute()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 1\n"
            f"names: ['analog_clock']\n"
        )
        (self.yolo_dir / "dataset.yaml").write_text(yaml_content)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = ClockDatasetGenerator()
    gen.generate_dataset()
