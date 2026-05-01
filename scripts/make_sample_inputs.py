"""
Generate the two sample input images expected by full-pipeline.ipynb.

Produces:
    sample_inputs/digital_clock_image.png   — a 7-segment-style HH:MM display
    sample_inputs/analog_clock_image.jpeg   — a rendered analog clock face

The originals were never in git, so the demo failed on fresh clones with
FileNotFoundError. This script regenerates plausible substitutes.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "sample_inputs"
DIGITAL_PATH = OUTPUT_DIR / "digital_clock_image.png"
ANALOG_PATH = OUTPUT_DIR / "analog_clock_image.jpeg"

DEFAULT_HOUR = 10
DEFAULT_MINUTE = 32


SEG_LAYOUT = {
    "0": (1, 1, 1, 1, 1, 1, 0),
    "1": (0, 1, 1, 0, 0, 0, 0),
    "2": (1, 1, 0, 1, 1, 0, 1),
    "3": (1, 1, 1, 1, 0, 0, 1),
    "4": (0, 1, 1, 0, 0, 1, 1),
    "5": (1, 0, 1, 1, 0, 1, 1),
    "6": (1, 0, 1, 1, 1, 1, 1),
    "7": (1, 1, 1, 0, 0, 0, 0),
    "8": (1, 1, 1, 1, 1, 1, 1),
    "9": (1, 1, 1, 1, 0, 1, 1),
}


def draw_seven_segment_digit(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    digit: str,
    width: int,
    height: int,
    thickness: int,
    color,
) -> None:
    half = height // 2
    gap = thickness // 2
    coords = [
        (x + gap, y, x + width - gap, y + thickness),
        (x + width - thickness, y + gap, x + width, y + half - gap),
        (x + width - thickness, y + half + gap, x + width, y + height - gap),
        (x + gap, y + height - thickness, x + width - gap, y + height),
        (x, y + half + gap, x + thickness, y + height - gap),
        (x, y + gap, x + thickness, y + half - gap),
        (x + gap, y + half - thickness // 2, x + width - gap, y + half + thickness // 2),
    ]
    for is_on, box in zip(SEG_LAYOUT[digit], coords):
        if is_on:
            draw.rectangle(box, fill=color)


def render_digital_clock(hour: int, minute: int) -> Image.Image:
    img_w, img_h = 640, 320
    img = Image.new("RGB", (img_w, img_h), (16, 16, 22))
    draw = ImageDraw.Draw(img)

    seg_w, seg_h = 70, 140
    thickness = 14
    digit_color = (90, 230, 255)
    spacing = 18
    colon_gap = 36

    digits = f"{hour:02d}{minute:02d}"
    total_w = 4 * seg_w + 3 * spacing + colon_gap
    start_x = (img_w - total_w) // 2
    start_y = (img_h - seg_h) // 2

    cursor = start_x
    for i, ch in enumerate(digits):
        draw_seven_segment_digit(draw, cursor, start_y, ch, seg_w, seg_h, thickness, digit_color)
        cursor += seg_w + spacing
        if i == 1:
            dot = thickness + 2
            mid_y = start_y + seg_h // 2
            draw.ellipse(
                (cursor, mid_y - dot - 8, cursor + dot, mid_y - 8),
                fill=digit_color,
            )
            draw.ellipse(
                (cursor, mid_y + 8, cursor + dot, mid_y + dot + 8),
                fill=digit_color,
            )
            cursor += colon_gap

    return img


def render_analog_clock(hour: int, minute: int) -> Image.Image:
    size = 640
    bg = Image.new("RGB", (size, size), (210, 200, 185))
    rng = random.Random(42)
    for _ in range(2400):
        px = rng.randint(0, size - 1)
        py = rng.randint(0, size - 1)
        jitter = rng.randint(-12, 12)
        bg.putpixel(
            (px, py),
            (max(0, min(255, 210 + jitter)),
             max(0, min(255, 200 + jitter)),
             max(0, min(255, 185 + jitter))),
        )
    draw = ImageDraw.Draw(bg)

    cx = cy = size // 2
    radius = 240

    draw.ellipse(
        (cx - radius - 18, cy - radius - 18, cx + radius + 18, cy + radius + 18),
        fill=(40, 30, 25),
    )
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(248, 246, 240),
    )

    for tick in range(60):
        angle = math.radians(tick * 6)
        outer = radius - 8
        inner = radius - (28 if tick % 5 == 0 else 14)
        thickness = 4 if tick % 5 == 0 else 2
        x1 = cx + outer * math.sin(angle)
        y1 = cy - outer * math.cos(angle)
        x2 = cx + inner * math.sin(angle)
        y2 = cy - inner * math.cos(angle)
        draw.line((x1, y1, x2, y2), fill=(30, 30, 30), width=thickness)

    hour_angle = math.radians((hour % 12) * 30 + minute * 0.5)
    minute_angle = math.radians(minute * 6)

    def hand(angle: float, length: int, width: int, color):
        tx = cx + length * math.sin(angle)
        ty = cy - length * math.cos(angle)
        draw.line((cx, cy, tx, ty), fill=color, width=width)

    hand(hour_angle, int(radius * 0.55), 12, (25, 25, 25))
    hand(minute_angle, int(radius * 0.80), 8, (25, 25, 25))

    pin = 12
    draw.ellipse((cx - pin, cy - pin, cx + pin, cy + pin), fill=(180, 30, 30))

    return bg


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    digital = render_digital_clock(DEFAULT_HOUR, DEFAULT_MINUTE)
    digital.save(DIGITAL_PATH)
    print(f"Wrote {DIGITAL_PATH}  ({digital.size[0]}x{digital.size[1]})")

    analog = render_analog_clock(DEFAULT_HOUR, DEFAULT_MINUTE)
    analog.save(ANALOG_PATH, quality=92)
    print(f"Wrote {ANALOG_PATH}  ({analog.size[0]}x{analog.size[1]})")


if __name__ == "__main__":
    main()
