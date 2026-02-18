"""
yolo_export.py
==============
Handles exporting annotated images to a YOLO-Seg compatible dataset structure.

Output layout:
    <output_dir>/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml

YOLO-Seg label format (per .txt file):
    <class_id> x1 y1 x2 y2 ... xn yn
    - One line per object instance.
    - Coordinates are normalized [0, 1].
    - Polygon points ordered (clockwise or counter-clockwise).

data.yaml structure (YOLOv8-compatible):
    path: /absolute/path/to/dataset
    train: images/train
    val: images/val
    nc: 3
    names:
      0: hour
      1: minute
      2: second
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from annotation_logic import AnnotationManager


CLASS_NAMES = {0: "hour", 1: "minute", 2: "second"}


class YOLOExporter:
    """
    Converts annotated images into a YOLO-Seg dataset.

    Parameters
    ----------
    ann_manager : AnnotationManager
        The manager used to read existing annotations.
    """

    def __init__(self, ann_manager: "AnnotationManager"):
        self._ann = ann_manager

    def export(
        self,
        image_files: list[Path],
        output_dir: Path,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> dict:
        """
        Export dataset to *output_dir*.

        Parameters
        ----------
        image_files : list[Path]
            All images in the annotation folder.
        output_dir : Path
            Root directory for the exported dataset.
        train_ratio : float
            Fraction of annotated images used for training (0-1).
        seed : int
            Random seed for reproducible splits.

        Returns
        -------
        dict with keys: 'train', 'val', 'skipped'
        """
        # Collect images that actually have annotations
        annotated = [p for p in image_files if self._ann.has_annotations(p)]
        skipped = len(image_files) - len(annotated)

        if not annotated:
            raise RuntimeError(
                "No annotated images found. "
                "Please annotate at least one image before exporting."
            )

        # Reproducible shuffle → split
        rng = random.Random(seed)
        shuffled = list(annotated)
        rng.shuffle(shuffled)

        split_idx = max(1, int(len(shuffled) * train_ratio))
        train_imgs = shuffled[:split_idx]
        val_imgs   = shuffled[split_idx:] or [shuffled[-1]]  # always at least 1 val

        # Create directory structure
        dirs = {
            "img_train": output_dir / "images" / "train",
            "img_val":   output_dir / "images" / "val",
            "lbl_train": output_dir / "labels" / "train",
            "lbl_val":   output_dir / "labels" / "val",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Copy images + write labels
        self._copy_split(train_imgs, dirs["img_train"], dirs["lbl_train"])
        self._copy_split(val_imgs,   dirs["img_val"],   dirs["lbl_val"])

        # Generate data.yaml
        self._write_yaml(output_dir)

        return {
            "train":   len(train_imgs),
            "val":     len(val_imgs),
            "skipped": skipped,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _copy_split(
        self,
        image_paths: list[Path],
        img_dir: Path,
        lbl_dir: Path,
    ):
        """Copy images and write YOLO label files for one split."""
        for img_path in image_paths:
            # Copy image
            dest_img = img_dir / img_path.name
            shutil.copy2(img_path, dest_img)

            # Write label file
            lines = self._ann.to_yolo_lines(img_path)
            lbl_name = img_path.stem + ".txt"
            dest_lbl = lbl_dir / lbl_name
            dest_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_yaml(self, output_dir: Path):
        """Write a YOLOv8-compatible data.yaml."""
        data = {
            "path":  str(output_dir.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "nc":    3,
            "names": {i: name for i, name in CLASS_NAMES.items()},
        }
        yaml_path = output_dir / "data.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
