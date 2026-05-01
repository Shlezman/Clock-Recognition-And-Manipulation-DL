"""
ViT clock generation dataset generator.

Produces paired (source, target) 224x224 clock images where:
  - source: analog clock showing a random time
  - target: the same clock face showing a *different* target time
  - labels.csv: maps filename → (target_hh, target_mm)

Output layout:
    dataset/vit_clock_gen/
    ├── train/
    │   ├── source/     # 224x224 PNG
    │   ├── target/     # 224x224 PNG
    │   └── labels.csv
    └── val/
        ├── source/
        ├── target/
        └── labels.csv

Run from the project root:
    python analog_clock/vit_clock_gen/dataset_generator.py
    python analog_clock/vit_clock_gen/dataset_generator.py --n_samples 5000 --output_dir ./my_dataset
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analog_clock.shared.asset_manager import AssetManager
from analog_clock.shared.augmentations import AugmentationPipeline
from analog_clock.shared.clock_renderer import ClockRenderer
from analog_clock.shared.config import SketchConfig
from analog_clock.shared.procedural_generator import ProceduralClockGenerator

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224  # ViT patch16 expects 224×224


class ViTClockDatasetGenerator:
    """Generate paired source/target clock images for ViT training."""

    def __init__(self, config: SketchConfig | None = None, output_dir: str = "./dataset/vit_clock_gen") -> None:
        self.config = config or SketchConfig()
        self.output_dir = Path(output_dir)

        self.asset_mgr = AssetManager(self.config)
        self.asset_mgr.prepare_assets()
        self.textures = self.asset_mgr.get_all_images()

        self.clock_gen = ProceduralClockGenerator(self.config)
        self.renderer = ClockRenderer()
        self.augmentor = AugmentationPipeline(self.config)

        self._create_dirs()
        logger.info("ViT clock dataset generator initialised — output: %s", self.output_dir)

    def _create_dirs(self) -> None:
        for split in ("train", "val"):
            for subdir in ("source", "target"):
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-sample generation
    # ------------------------------------------------------------------

    def _render_clock(
        self,
        tex_wall: np.ndarray,
        tex_face: np.ndarray,
        center: tuple[int, int],
        radius: int,
        scene_size: tuple[int, int],
        hh: int,
        mm: int,
        h_hand: np.ndarray,
        m_hand: np.ndarray,
    ) -> np.ndarray:
        base = self.clock_gen.create_clock_on_wall(tex_wall, tex_face, center, radius, scene_size)
        scene = self.renderer.composite_hands(base, h_hand, m_hand, hh, mm, center)
        return scene

    def _crop_and_resize(
        self,
        img: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> np.ndarray:
        padding = int(radius * self.config.CROP_PADDING_RATIO)
        crop_r = radius + padding
        x1 = max(0, cx - crop_r)
        y1 = max(0, cy - crop_r)
        x2 = min(img.shape[1], cx + crop_r)
        y2 = min(img.shape[0], cy + crop_r)
        crop = img[y1:y2, x1:x2]

        # Pad to square if needed
        h, w = crop.shape[:2]
        d = crop_r * 2
        if h != d or w != d:
            square = np.zeros((d, d, 3), dtype=np.uint8)
            sy, sx = (d - h) // 2, (d - w) // 2
            square[sy: sy + h, sx: sx + w] = crop
            crop = square

        return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    def generate_sample(self, idx: int, split: str = "train") -> dict:
        cfg = self.config
        scene_w = random.randint(*cfg.SCENE_WIDTH_RANGE)
        scene_h = random.randint(*cfg.SCENE_HEIGHT_RANGE)

        min_dim = min(scene_w, scene_h)
        radius = random.randint(
            int(min_dim * cfg.MIN_RADIUS_RATIO),
            int(min_dim * cfg.MAX_RADIUS_RATIO),
        )
        margin = int(radius * 1.2)
        cx = random.randint(margin, scene_w - margin)
        cy = random.randint(margin, scene_h - margin)
        center = (cx, cy)

        tex_wall = random.choice(self.textures)
        tex_face = random.choice(self.textures)
        hand_set = self.clock_gen.generate_hand_set(center, radius, (scene_w, scene_h))
        h_hand, m_hand = hand_set[0], hand_set[1]

        src_hh, src_mm = random.randint(0, 23), random.randint(0, 59)
        tgt_hh, tgt_mm = random.randint(0, 23), random.randint(0, 59)

        scene_src = self._render_clock(
            tex_wall, tex_face, center, radius, (scene_w, scene_h),
            src_hh, src_mm, h_hand, m_hand,
        )
        scene_tgt = self._render_clock(
            tex_wall, tex_face, center, radius, (scene_w, scene_h),
            tgt_hh, tgt_mm, h_hand, m_hand,
        )

        scene_src, scene_tgt = self.augmentor.apply_paired(scene_src, scene_tgt)

        crop_src = self._crop_and_resize(scene_src, cx, cy, radius)
        crop_tgt = self._crop_and_resize(scene_tgt, cx, cy, radius)

        fname = f"{idx:06d}"
        cv2.imwrite(
            str(self.output_dir / split / "source" / f"{fname}.png"),
            self.augmentor.to_bgr(crop_src),
        )
        cv2.imwrite(
            str(self.output_dir / split / "target" / f"{fname}.png"),
            self.augmentor.to_bgr(crop_tgt),
        )

        return {"filename": fname, "split": split, "target_hh": tgt_hh, "target_mm": tgt_mm}

    # ------------------------------------------------------------------
    # Full dataset
    # ------------------------------------------------------------------

    def generate_dataset(self, n_samples: int = 10_000, train_split: float = 0.85) -> None:
        n_train = int(n_samples * train_split)
        n_val = n_samples - n_train
        logger.info(
            "Generating %d samples (%d train / %d val) at %dx%d",
            n_samples, n_train, n_val, IMAGE_SIZE, IMAGE_SIZE,
        )

        train_meta: list[dict] = []
        for i in tqdm(range(n_train), desc="Train"):
            train_meta.append(self.generate_sample(i, "train"))
        pd.DataFrame(train_meta).to_csv(self.output_dir / "train" / "labels.csv", index=False)

        val_meta: list[dict] = []
        for i in tqdm(range(n_val), desc="Val  "):
            val_meta.append(self.generate_sample(i, "val"))
        pd.DataFrame(val_meta).to_csv(self.output_dir / "val" / "labels.csv", index=False)

        logger.info("ViT clock dataset generation complete — %s", self.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ViT clock generation training dataset")
    p.add_argument("--n_samples", type=int, default=10_000, help="Total samples (default: 10000)")
    p.add_argument("--train_split", type=float, default=0.85, help="Fraction for training (default: 0.85)")
    p.add_argument("--output_dir", type=str, default="./dataset/vit_clock_gen", help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    gen = ViTClockDatasetGenerator(output_dir=args.output_dir)
    gen.generate_dataset(n_samples=args.n_samples, train_split=args.train_split)
