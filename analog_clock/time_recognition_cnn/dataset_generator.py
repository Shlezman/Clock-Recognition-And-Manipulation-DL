"""
Synthetic dataset generator for time-recognition from binary clock-hand masks.

Produces 256x256 single-channel binary images of clock hands at random times,
plus CSV labels with hour, minute, and their sin/cos angle encodings.

Usage:
    python dataset_generator.py --n_samples 20000 --output_dir ./dataset
"""

import cv2
import csv
import math
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hand drawing helpers (simplified from inpainting procedural_generator)
# ---------------------------------------------------------------------------

HAND_STYLES = [
    "pointed", "rectangle", "modern", "arrow", "diamond",
    "tapered", "sword", "lollipop", "baton",
    "leaf", "pencil", "dauphine", "breguet", "spade",
    "cathedral", "alpha", "feuille", "lance",
    "plongeur", "syringe", "flamme",
]

IMG_SIZE = 256
CENTER = (IMG_SIZE // 2, IMG_SIZE // 2)


def _draw_hand_polygon(img: np.ndarray, cx: int, cy: int,
                       length: int, width: int, style: str) -> None:
    """Draw a single hand polygon onto *img* (white on black, single channel)."""
    color = 255  # white foreground

    if style == "pointed":
        pts = np.array([[cx, cy - length], [cx + width, cy],
                        [cx, cy + width], [cx - width, cy]], np.int32)
    elif style == "arrow":
        ah = int(width * 2)
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + ah, cy - length + ah],
            [cx + hw, cy - length + ah], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, cy - length + ah],
            [cx - ah, cy - length + ah],
        ], np.int32)
    elif style == "diamond":
        ds = int(width * 1.5)
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + ds, cy - length + ds],
            [cx + hw, cy - length + ds * 2], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, cy - length + ds * 2],
            [cx - ds, cy - length + ds],
        ], np.int32)
    elif style == "tapered":
        pts = np.array([
            [cx, cy - length],
            [cx + width // 4, int(cy - length * 0.7)],
            [cx + width // 2, cy],
            [cx - width // 2, cy],
            [cx - width // 4, int(cy - length * 0.7)],
        ], np.int32)
    elif style == "sword":
        bw = width // 3
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + bw, int(cy - length * 0.85)],
            [cx + hw, int(cy - length * 0.85)], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, int(cy - length * 0.85)],
            [cx - bw, int(cy - length * 0.85)],
        ], np.int32)
    elif style == "lollipop":
        hw = width // 2
        pts = np.array([
            [cx - hw, cy], [cx - hw, cy - length + width * 2],
            [cx + hw, cy - length + width * 2], [cx + hw, cy],
        ], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - length + width), width, color, -1, cv2.LINE_AA)
        return
    elif style == "baton":
        tw = width // 3
        pts = np.array([
            [cx - tw, cy], [cx - tw, cy - length],
            [cx + tw, cy - length], [cx + tw, cy],
        ], np.int32)
    elif style == "leaf":
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + width, int(cy - length * 0.6)],
            [cx + hw, cy], [cx - hw, cy],
            [cx - width, int(cy - length * 0.6)],
        ], np.int32)
    elif style == "pencil":
        hw = width // 2
        tw = width // 3
        pts = np.array([
            [cx, cy - length], [cx + tw, int(cy - length * 0.9)],
            [cx + hw, cy], [cx - hw, cy],
            [cx - tw, int(cy - length * 0.9)],
        ], np.int32)
    elif style == "dauphine":
        hw = width // 2
        tw = width // 3
        pts = np.array([
            [cx, cy - length], [cx + tw, int(cy - length * 0.7)],
            [cx + width, int(cy - length * 0.5)],
            [cx + hw, cy], [cx - hw, cy],
            [cx - width, int(cy - length * 0.5)],
            [cx - tw, int(cy - length * 0.7)],
        ], np.int32)
    elif style == "breguet":
        hw = width // 2
        pts = np.array([
            [cx - hw, cy], [cx - hw, cy - length],
            [cx + hw, cy - length], [cx + hw, cy],
        ], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        hole_r = int(width * 0.8)
        hole_y = cy - int(length * 0.75)
        cv2.circle(img, (cx, hole_y), hole_r, 0, -1, cv2.LINE_AA)
        return
    elif style == "spade":
        sw = int(width * 1.8)
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + sw, int(cy - length * 0.8)],
            [int(cx + sw * 0.7), int(cy - length * 0.65)],
            [cx + hw, int(cy - length * 0.7)], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, int(cy - length * 0.7)],
            [int(cx - sw * 0.7), int(cy - length * 0.65)],
            [cx - sw, int(cy - length * 0.8)],
        ], np.int32)
    elif style == "cathedral":
        hw = width // 2
        qw = width // 4
        pts = np.array([
            [cx, cy - length], [cx + qw, int(cy - length * 0.92)],
            [int(cx + width * 0.6), int(cy - length * 0.8)],
            [cx + hw, int(cy - length * 0.65)], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, int(cy - length * 0.65)],
            [int(cx - width * 0.6), int(cy - length * 0.8)],
            [cx - qw, int(cy - length * 0.92)],
        ], np.int32)
    elif style == "alpha":
        hw = width // 2
        pts = np.array([
            [cx, cy - length],
            [int(cx + width * 1.5), int(cy - length * 0.6)],
            [cx + hw, cy], [cx - hw, cy],
            [int(cx - width * 1.5), int(cy - length * 0.6)],
        ], np.int32)
    elif style == "feuille":
        tw = width // 3
        pts = np.array([
            [cx, cy - length],
            [int(cx + width * 0.8), int(cy - length * 0.5)],
            [cx + tw, cy], [cx - tw, cy],
            [int(cx - width * 0.8), int(cy - length * 0.5)],
        ], np.int32)
    elif style == "lance":
        qw = width // 4
        pts = np.array([
            [cx, cy - length],
            [int(cx + width * 0.6), int(cy - length * 0.9)],
            [cx + qw, int(cy - length * 0.8)], [cx + qw, cy],
            [cx - qw, cy], [cx - qw, int(cy - length * 0.8)],
            [int(cx - width * 0.6), int(cy - length * 0.9)],
        ], np.int32)
    elif style == "plongeur":
        ow = int(width * 0.7)
        pts = np.array([
            [cx - ow, cy], [cx - ow, int(cy - length * 0.9)],
            [cx, cy - length],
            [cx + ow, int(cy - length * 0.9)], [cx + ow, cy],
        ], np.int32)
    elif style == "syringe":
        tw = width // 4
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + tw, int(cy - length * 0.95)],
            [cx + hw, int(cy - length * 0.9)], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, int(cy - length * 0.9)],
            [cx - tw, int(cy - length * 0.95)],
        ], np.int32)
    elif style == "flamme":
        hw = width // 2
        pts = np.array([
            [cx, cy - length],
            [int(cx + width * 0.3), int(cy - length * 0.9)],
            [int(cx + width * 0.7), int(cy - length * 0.7)],
            [int(cx + width * 0.5), int(cy - length * 0.5)],
            [cx + hw, cy], [cx - hw, cy],
            [int(cx - width * 0.5), int(cy - length * 0.5)],
            [int(cx - width * 0.7), int(cy - length * 0.7)],
            [int(cx - width * 0.3), int(cy - length * 0.9)],
        ], np.int32)
    else:  # rectangle / modern / default
        hw = width // 2
        pts = np.array([
            [cx - hw, cy], [cx - hw, cy - length],
            [cx + hw, cy - length], [cx + hw, cy],
        ], np.int32)

    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def _rotate(img: np.ndarray, angle_deg: float, center: tuple[int, int]) -> np.ndarray:
    """Rotate *img* around *center* by *angle_deg* (CW from 12-o'clock)."""
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------------------
# Mask rendering
# ---------------------------------------------------------------------------

def render_mask(hour: int, minute: int, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Render a 256x256 binary mask with clock hands pointing to *hour*:*minute*.

    Returns a uint8 image where 255 = hand pixels, 0 = background.
    """
    cx, cy = img_size // 2, img_size // 2
    center = (cx, cy)
    radius = int(img_size * 0.45)  # leave a small border

    # Randomise hand appearance per sample
    style = random.choice(HAND_STYLES)

    minute_length = int(radius * random.uniform(0.75, 0.95))
    hour_length = int(radius * random.uniform(0.50, 0.65))
    minute_width = max(2, int(radius * random.uniform(0.03, 0.06)))
    hour_width = max(3, int(radius * random.uniform(0.06, 0.10)))

    # Draw hands pointing UP (12 o'clock), then rotate to target angle
    h_img = np.zeros((img_size, img_size), dtype=np.uint8)
    m_img = np.zeros((img_size, img_size), dtype=np.uint8)

    _draw_hand_polygon(h_img, cx, cy, hour_length, hour_width, style)
    _draw_hand_polygon(m_img, cx, cy, minute_length, minute_width, style)

    # Angles: 0 deg = 12 o'clock, CW positive
    angle_h = (hour % 12) * 30.0 + minute * 0.5
    angle_m = minute * 6.0

    h_rot = _rotate(h_img, angle_h, center)
    m_rot = _rotate(m_img, angle_m, center)

    mask = cv2.bitwise_or(h_rot, m_rot)

    # Add center pin
    cv2.circle(mask, center, max(2, int(radius * 0.03)), 255, -1, cv2.LINE_AA)

    return mask


# ---------------------------------------------------------------------------
# Label encoding helpers
# ---------------------------------------------------------------------------

def time_to_angles(hour: int, minute: int) -> tuple[float, float]:
    """Return (hour_angle_rad, minute_angle_rad) in [0, 2*pi)."""
    hour_angle = ((hour % 12) * 30.0 + minute * 0.5) * math.pi / 180.0
    minute_angle = (minute * 6.0) * math.pi / 180.0
    return hour_angle, minute_angle


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_samples: int, output_dir: str, train_split: float = 0.8) -> None:
    """
    Generate *n_samples* binary-mask images + a CSV of labels.

    Directory layout:
        output_dir/
            train/
                images/  000000.png ...
            val/
                images/  000000.png ...
            train_labels.csv
            val_labels.csv
    """
    out = Path(output_dir)
    n_train = int(n_samples * train_split)

    splits = [
        ("train", 0, n_train),
        ("val", n_train, n_samples),
    ]

    for split_name, start, end in splits:
        img_dir = out / split_name / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out / f"{split_name}_labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "hour", "minute",
                "hour_sin", "hour_cos", "minute_sin", "minute_cos",
            ])

            for idx in tqdm(range(end - start), desc=split_name):
                hour = random.randint(0, 11)
                minute = random.randint(0, 59)

                mask = render_mask(hour, minute)

                fname = f"{idx:06d}.png"
                cv2.imwrite(str(img_dir / fname), mask)

                h_angle, m_angle = time_to_angles(hour, minute)
                writer.writerow([
                    fname, hour, minute,
                    f"{math.sin(h_angle):.6f}", f"{math.cos(h_angle):.6f}",
                    f"{math.sin(m_angle):.6f}", f"{math.cos(m_angle):.6f}",
                ])

    print(f"Dataset saved to {out}  ({n_train} train / {n_samples - n_train} val)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary-mask time dataset")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--output_dir", type=str, default="./dataset")
    parser.add_argument("--train_split", type=float, default=0.8)
    args = parser.parse_args()

    generate_dataset(args.n_samples, args.output_dir, args.train_split)
