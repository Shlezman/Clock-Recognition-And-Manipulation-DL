#!/usr/bin/env python3
"""
Create a cropped hand-annotation dataset using a clock detector.

Input:
  - Images in source folder
  - Sidecar annotations with same stem:
      <image>.ann.json or <image>.json

Output:
  - Cropped images + remapped sidecars in output folder
  - Metadata CSV + summary JSON in output folder/_meta
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass
class CropResult:
    src_image: Path
    src_ann: Path
    out_image: Path
    out_ann: Path
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    det_conf: float
    kept_classes: int


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "analog_clock").exists() and (p / "digital_clock").exists():
            return p
    raise FileNotFoundError("Could not locate repository root")


def sidecar_for_image(image_path: Path) -> Path | None:
    for p in (image_path.with_suffix(".ann.json"), image_path.with_suffix(".json")):
        if p.exists() and p.is_file():
            return p
    return None


def load_annotations(ann_path: Path) -> dict[int, list[tuple[float, float]]]:
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    ann_obj = payload.get("annotations", {})
    if not isinstance(ann_obj, dict):
        return {}

    out: dict[int, list[tuple[float, float]]] = {}
    for cls_key, points in ann_obj.items():
        try:
            cls_id = int(cls_key)
        except (TypeError, ValueError):
            continue
        if cls_id < 0 or cls_id > 2 or not isinstance(points, list):
            continue

        cleaned: list[tuple[float, float]] = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            try:
                x = float(pt[0])
                y = float(pt[1])
            except (TypeError, ValueError):
                continue
            if np.isnan(x) or np.isnan(y):
                continue
            cleaned.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
        if len(cleaned) >= 3:
            out[cls_id] = cleaned
    return out


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def dedup_points(points: list[tuple[float, float]], eps: float = 1e-8) -> list[tuple[float, float]]:
    if not points:
        return []
    out: list[tuple[float, float]] = [points[0]]
    for x, y in points[1:]:
        px, py = out[-1]
        if abs(x - px) > eps or abs(y - py) > eps:
            out.append((x, y))
    if len(out) > 1:
        fx, fy = out[0]
        lx, ly = out[-1]
        if abs(fx - lx) <= eps and abs(fy - ly) <= eps:
            out.pop()
    return out


def remap_polygon_to_crop(
    points_norm: list[tuple[float, float]],
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> list[tuple[float, float]]:
    cw = max(1, x2 - x1)
    ch = max(1, y2 - y1)
    remapped: list[tuple[float, float]] = []
    for nx, ny in points_norm:
        px = nx * width
        py = ny * height
        cx = (px - x1) / cw
        cy = (py - y1) / ch
        # Clip so partially outside polygons still remain valid in crop frame.
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        remapped.append((cx, cy))

    remapped = dedup_points(remapped)
    if len(remapped) < 3:
        return []
    if len({(round(x, 8), round(y, 8)) for x, y in remapped}) < 3:
        return []
    if polygon_area(remapped) < 1e-6:
        return []
    return remapped


def select_clock_bbox(
    result,
    img_w: int,
    img_h: int,
    margin_ratio: float,
    min_conf: float,
) -> tuple[int, int, int, int, float] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), dtype=np.float32)
    cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros((len(xyxy),), dtype=np.float32)

    candidates: list[tuple[float, np.ndarray]] = []
    for i, box in enumerate(xyxy):
        c = float(conf[i]) if i < len(conf) else 0.0
        k = int(cls[i]) if i < len(cls) else 0
        if c < min_conf:
            continue
        if k != 0:
            continue
        candidates.append((c, box))
    if not candidates:
        # Fallback: highest-confidence box regardless of class id.
        for i, box in enumerate(xyxy):
            c = float(conf[i]) if i < len(conf) else 0.0
            if c < min_conf:
                continue
            candidates.append((c, box))
    if not candidates:
        return None

    best_conf, best_box = max(candidates, key=lambda t: t[0])
    x1, y1, x2, y2 = [float(v) for v in best_box.tolist()]
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    pad_x = bw * margin_ratio
    pad_y = bh * margin_ratio

    x1 = int(max(0, np.floor(x1 - pad_x)))
    y1 = int(max(0, np.floor(y1 - pad_y)))
    x2 = int(min(img_w, np.ceil(x2 + pad_x)))
    y2 = int(min(img_h, np.ceil(y2 + pad_y)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2, best_conf)


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path.cwd().resolve())
    default_source = repo_root / "analog_clock" / "data" / "from_internet"
    default_output = repo_root / "analog_clock" / "data" / "from_internet_cropped"
    default_model = repo_root / "analog_clock" / "yolo_detect_clock" / "analog_clock_yolo_model.pt"
    default_device = "0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Crop clock images and remap hand annotations")
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--clock-model", type=Path, default=default_model)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--max-images", type=int, default=0, help="0 means no limit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    meta_dir = output_dir / "_meta"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    if not args.clock_model.exists():
        raise FileNotFoundError(f"Clock model not found: {args.clock_model}")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.clock_model))

    images = sorted(
        [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    )
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    processed: list[CropResult] = []
    skipped_no_sidecar: list[Path] = []
    skipped_bad_ann: list[Path] = []
    skipped_no_detection: list[Path] = []
    skipped_empty_after_crop: list[Path] = []

    out_index = 0
    for image_path in images:
        sidecar = sidecar_for_image(image_path)
        if sidecar is None:
            skipped_no_sidecar.append(image_path)
            continue

        try:
            anns = load_annotations(sidecar)
        except Exception:
            skipped_bad_ann.append(sidecar)
            continue
        if not anns:
            skipped_bad_ann.append(sidecar)
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            skipped_bad_ann.append(sidecar)
            continue
        h, w = img.shape[:2]

        pred = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        bbox = select_clock_bbox(pred, img_w=w, img_h=h, margin_ratio=args.margin, min_conf=args.conf)
        if bbox is None:
            skipped_no_detection.append(image_path)
            continue
        x1, y1, x2, y2, det_conf = bbox

        remapped: dict[int, list[tuple[float, float]]] = {}
        for cls_id, points in anns.items():
            mapped = remap_polygon_to_crop(points, width=w, height=h, x1=x1, y1=y1, x2=x2, y2=y2)
            if len(mapped) >= 3:
                remapped[cls_id] = mapped
        if not remapped:
            skipped_empty_after_crop.append(image_path)
            continue

        crop = img[y1:y2, x1:x2]
        out_stem = f"clock_crop_{out_index:05d}"
        out_image = output_dir / f"{out_stem}{image_path.suffix.lower()}"
        out_ann = output_dir / f"{out_stem}.ann.json"
        out_index += 1

        if not cv2.imwrite(str(out_image), crop):
            skipped_bad_ann.append(sidecar)
            continue

        payload = {
            "image": out_image.name,
            "annotations": {
                str(cls_id): [[round(x, 8), round(y, 8)] for x, y in pts]
                for cls_id, pts in sorted(remapped.items())
            },
            "meta": {
                "source_image": str(image_path),
                "source_annotation": str(sidecar),
                "crop_bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": float(det_conf),
            },
        }
        out_ann.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        processed.append(
            CropResult(
                src_image=image_path,
                src_ann=sidecar,
                out_image=out_image,
                out_ann=out_ann,
                bbox_x1=int(x1),
                bbox_y1=int(y1),
                bbox_x2=int(x2),
                bbox_y2=int(y2),
                det_conf=float(det_conf),
                kept_classes=len(remapped),
            )
        )

    rows = [
        {
            "source_image": str(r.src_image),
            "source_annotation": str(r.src_ann),
            "output_image": str(r.out_image),
            "output_annotation": str(r.out_ann),
            "bbox_x1": r.bbox_x1,
            "bbox_y1": r.bbox_y1,
            "bbox_x2": r.bbox_x2,
            "bbox_y2": r.bbox_y2,
            "det_conf": r.det_conf,
            "kept_classes": r.kept_classes,
        }
        for r in processed
    ]
    pd.DataFrame(rows).to_csv(meta_dir / "mapping.csv", index=False)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "clock_model": str(args.clock_model.resolve()),
        "device": str(args.device),
        "source_images_total": len(images),
        "processed_crops": len(processed),
        "skipped_no_matching_json": len(skipped_no_sidecar),
        "skipped_invalid_json_or_annotations": len(skipped_bad_ann),
        "skipped_no_clock_detected": len(skipped_no_detection),
        "skipped_empty_after_crop": len(skipped_empty_after_crop),
        "mapping_csv": str((meta_dir / "mapping.csv").resolve()),
    }
    (meta_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

