"""
annotation_logic.py
====================
Handles persistence of annotations as JSON sidecar files.

Each image gets a companion  <image_stem>.ann.json  stored in the same folder.
Format:
{
    "image": "clock01.jpg",
    "width": 640,
    "height": 480,
    "annotations": {
        "0": [[0.12, 0.34], [0.15, 0.40], ...],   // hour hand polygon
        "1": [[0.50, 0.20], ...],                   // minute hand polygon
        "2": [[0.55, 0.30], ...]                    // second hand polygon
    }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Annotation:
    """Stores polygon points (normalized) for one class."""
    class_id: int
    points: list[tuple[float, float]] = field(default_factory=list)

    def is_valid(self) -> bool:
        return len(self.points) >= 3

    def to_yolo_line(self) -> str:
        """Returns one YOLO-seg line: class_id x1 y1 x2 y2 ..."""
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        return f"{self.class_id} {coords}"


class AnnotationManager:
    """
    Handles read/write of annotation sidecar files (.ann.json).

    The sidecar lives next to the source image:
        images/clock01.jpg  →  images/clock01.ann.json
    """

    SIDECAR_SUFFIX = ".ann.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, image_path: Path) -> dict[int, list[tuple[float, float]]]:
        """
        Load annotations for *image_path*.

        Returns a dict  {class_id: [(nx, ny), ...]}
        If no sidecar exists, returns {}.
        """
        sidecar = self._sidecar_path(image_path)
        if not sidecar.exists():
            return {}
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            result: dict[int, list[tuple[float, float]]] = {}
            for cls_str, pts in data.get("annotations", {}).items():
                cls_id = int(cls_str)
                result[cls_id] = [tuple(p) for p in pts]
            return result
        except (json.JSONDecodeError, KeyError, ValueError):
            return {}

    def save(self, image_path: Path, annotations: dict[int, list[tuple[float, float]]]):
        """
        Persist *annotations* for *image_path*.

        annotations: {class_id: [(nx, ny), ...]}
        Clears the sidecar if annotations is empty.
        """
        sidecar = self._sidecar_path(image_path)
        if not annotations:
            if sidecar.exists():
                sidecar.unlink()
            return

        payload = {
            "image": image_path.name,
            "annotations": {
                str(cls_id): [[round(x, 8), round(y, 8)] for x, y in pts]
                for cls_id, pts in annotations.items()
                if pts  # skip empty
            },
        }
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def delete(self, image_path: Path):
        """Remove the annotation sidecar for *image_path*."""
        sidecar = self._sidecar_path(image_path)
        if sidecar.exists():
            sidecar.unlink()

    def has_annotations(self, image_path: Path) -> bool:
        sidecar = self._sidecar_path(image_path)
        if not sidecar.exists():
            return False
        data = self.load(image_path)
        return bool(data)

    def get_all_annotated(self, image_files: list[Path]) -> list[Path]:
        """Return only those image paths that have non-empty annotations."""
        return [p for p in image_files if self.has_annotations(p)]

    def to_yolo_lines(self, image_path: Path) -> list[str]:
        """
        Return YOLO-seg formatted lines for the given image.
        Each line: '<class_id> x1 y1 x2 y2 ... xn yn'
        Returns [] if no annotations or annotations have < 3 points.
        """
        annotations = self.load(image_path)
        lines = []
        for cls_id in sorted(annotations.keys()):
            pts = annotations[cls_id]
            if len(pts) < 3:
                continue
            ann = Annotation(class_id=cls_id, points=pts)
            lines.append(ann.to_yolo_line())
        return lines

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sidecar_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(self.SIDECAR_SUFFIX)
