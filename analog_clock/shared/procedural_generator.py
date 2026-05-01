"""
Procedural clock generator: creates synthetic clock faces with hands.

Unified from sketch and inpainting versions. Supports:
- 23 hand styles (all from inpainting + originals)
- Optional second hand (controlled by config)
- Solid and textured face modes
- Dynamic scene sizes
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from analog_clock.shared.config import BaseConfig

logger = logging.getLogger(__name__)

# All supported hand styles
HAND_STYLES = [
    "pointed", "rectangle", "modern", "arrow", "diamond",
    "tapered", "sword", "lollipop", "skeleton", "baton",
    "leaf", "pencil", "dauphine", "breguet", "spade",
    "anchor", "cathedral", "alpha", "feuille", "lance",
    "plongeur", "syringe", "flamme",
]

SECOND_HAND_COLORS = [(0, 0, 200), (0, 0, 0), (200, 0, 0), (180, 0, 0)]

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_ITALIC,
]

ROMAN = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI",
    7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII",
}


class ProceduralClockGenerator:
    """Generate synthetic clock images with diverse hand styles."""

    def __init__(self, config: BaseConfig | None = None) -> None:
        self.config = config or BaseConfig()
        self.hand_colors = [
            (0, 0, 0), (255, 255, 255), (40, 40, 40),
            (20, 20, 80), (80, 20, 20), (200, 150, 20),
            (50, 50, 50), (100, 80, 60),
        ]

    # ------------------------------------------------------------------
    # Clock face generation
    # ------------------------------------------------------------------

    def create_clock_on_wall(
        self,
        wall_path: Path,
        face_path: Path,
        center: Tuple[int, int],
        radius: int,
        scene_size: Tuple[int, int],
    ) -> np.ndarray:
        """Composite a clock face onto a wall texture."""
        w, h = scene_size

        wall = cv2.imread(str(wall_path))
        if wall is None:
            wall = np.zeros((h, w, 3), np.uint8)
        wall = cv2.resize(wall, (w, h))

        d = radius * 2

        # Face: solid colour or texture
        if random.random() < self.config.SOLID_FACE_PROB:
            face_tex = np.zeros((d, d, 3), dtype=np.uint8)
            if random.random() < 0.70:
                val = random.randint(235, 255)
                face_tex[:] = (val, val, val)
            else:
                face_tex[:] = tuple(random.randint(40, 240) for _ in range(3))
        else:
            face_tex = cv2.imread(str(face_path))
            if face_tex is None:
                face_tex = np.full((d, d, 3), 200, dtype=np.uint8)
            else:
                face_tex = cv2.resize(face_tex, (d, d))

        # Circular mask
        mask = np.zeros((d, d), dtype=np.uint8)
        cv2.circle(mask, (radius, radius), radius, 255, -1, cv2.LINE_AA)

        face_circular = cv2.bitwise_and(face_tex, face_tex, mask=mask)
        _decorate_face(face_circular, radius, self.config.SHOW_NUMBERS_PROB)

        # Composite onto wall
        x1, y1 = center[0] - radius, center[1] - radius
        x2, y2 = x1 + d, y1 + d

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return wall

        roi = wall[y1:y2, x1:x2]
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(face_circular, face_circular, mask=mask)
        wall[y1:y2, x1:x2] = cv2.add(bg, fg)

        # Frame ring
        frame_clr = tuple(random.randint(20, 80) for _ in range(3))
        frame_thickness = random.randint(2, 6)
        cv2.circle(wall, center, radius, frame_clr, frame_thickness, cv2.LINE_AA)

        # "In the wild" realism effects
        wall = self._apply_wild_effects(wall, center, radius)

        return wall

    # ------------------------------------------------------------------
    # "In the wild" realism effects
    # ------------------------------------------------------------------

    def _apply_wild_effects(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        """Apply realistic imperfections to make synthetic clocks look natural."""
        cfg = self.config
        cx, cy = center

        # 1. Drop shadow beneath clock (wall-mounted shadow)
        if random.random() < cfg.SHADOW_PROB:
            img = _add_drop_shadow(img, cx, cy, radius)

        # 2. Frame/bezel inner shadow (depth effect)
        if random.random() < cfg.FRAME_SHADOW_PROB:
            img = _add_frame_shadow(img, cx, cy, radius)

        # 3. Lighting gradient across the face (directional illumination)
        if random.random() < cfg.LIGHTING_GRADIENT_PROB:
            img = _add_lighting_gradient(img, cx, cy, radius)

        # 4. Glare / specular highlight on glass
        if random.random() < cfg.GLARE_SPOT_PROB:
            img = _add_glare_spot(img, cx, cy, radius)

        # 5. Glass reflection stripe
        if random.random() < cfg.GLASS_REFLECTION_PROB:
            img = _add_glass_reflection(img, cx, cy, radius)

        # 6. Aged/yellowed face tint
        if random.random() < cfg.FACE_YELLOWING_PROB:
            img = _add_face_yellowing(img, cx, cy, radius)

        return img

    # ------------------------------------------------------------------
    # Hand generation
    # ------------------------------------------------------------------

    def generate_hand_set(
        self,
        center: Tuple[int, int],
        radius: int,
        scene_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return (hour_hand, minute_hand, second_hand_or_None) as RGBA images."""
        color = random.choice(self.hand_colors)
        style = random.choice(HAND_STYLES)

        minute_length = radius * random.uniform(0.75, 0.95)
        hour_length = radius * random.uniform(0.50, 0.65)
        hour_width = radius * random.uniform(0.06, 0.09)
        minute_width = radius * random.uniform(0.03, 0.05)

        hour_hand = _draw_single_hand(center, hour_length, hour_width, color, style, scene_size)
        minute_hand = _draw_single_hand(center, minute_length, minute_width, color, style, scene_size)

        # Optional second hand
        second_hand: Optional[np.ndarray] = None
        if random.random() < self.config.INCLUDE_SECOND_HAND_PROB:
            sec_color = random.choice(SECOND_HAND_COLORS)
            sec_length = radius * random.uniform(0.8, 0.95)
            sec_width = max(1, int(radius * 0.015))
            second_hand = _draw_second_hand(center, sec_length, sec_width, sec_color, scene_size)

        return hour_hand, minute_hand, second_hand


# ======================================================================
# "In the wild" effect helpers
# ======================================================================

def _add_drop_shadow(
    img: np.ndarray, cx: int, cy: int, radius: int
) -> np.ndarray:
    """Draw a soft drop shadow behind the clock (wall-mounted look)."""
    h, w = img.shape[:2]
    shadow = np.zeros((h, w), dtype=np.uint8)
    offset_x = random.randint(3, max(4, radius // 8))
    offset_y = random.randint(3, max(4, radius // 6))
    cv2.circle(shadow, (cx + offset_x, cy + offset_y), radius + 2, 255, -1, cv2.LINE_AA)
    # Don't shadow inside the clock itself
    cv2.circle(shadow, (cx, cy), radius - 1, 0, -1, cv2.LINE_AA)
    blur_k = max(11, radius // 3) | 1  # must be odd
    shadow = cv2.GaussianBlur(shadow, (blur_k, blur_k), 0)
    alpha = (shadow.astype(np.float32) / 255.0 * random.uniform(0.15, 0.40))[..., None]
    result = img.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def _add_frame_shadow(
    img: np.ndarray, cx: int, cy: int, radius: int
) -> np.ndarray:
    """Inner shadow from the bezel, giving depth."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # Thin ring just inside the frame
    cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (cx, cy), int(radius * 0.92), 0, -1, cv2.LINE_AA)
    blur_k = max(5, radius // 10) | 1
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    alpha = mask.astype(np.float32) / 255.0 * random.uniform(0.10, 0.30)
    result = img.astype(np.float32)
    result -= result * alpha[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def _add_lighting_gradient(
    img: np.ndarray, cx: int, cy: int, radius: int
) -> np.ndarray:
    """Directional lighting gradient across the clock face."""
    h, w = img.shape[:2]
    # Random light direction
    angle = random.uniform(0, 2 * np.pi)
    dx = np.cos(angle)
    dy = np.sin(angle)

    yy, xx = np.mgrid[0:h, 0:w]
    dot = (xx - cx) * dx + (yy - cy) * dy
    dot = dot / (radius + 1e-6)
    dot = np.clip(dot, -1, 1)

    # Circular mask to limit to clock face
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    face_mask = (dist < radius).astype(np.float32)

    strength = random.uniform(0.05, 0.20)
    adjustment = (dot * strength * face_mask)[..., None]
    result = img.astype(np.float32) + adjustment * 255
    return np.clip(result, 0, 255).astype(np.uint8)


def _add_glare_spot(
    img: np.ndarray, cx: int, cy: int, radius: int
) -> np.ndarray:
    """Circular specular highlight (glass reflection)."""
    h, w = img.shape[:2]
    glare = np.zeros((h, w), dtype=np.float32)
    # Random position on the face (offset from center)
    gx = cx + random.randint(-radius // 3, radius // 3)
    gy = cy + random.randint(-radius // 3, radius // 3)
    gr = random.randint(radius // 6, radius // 3)
    cv2.circle(glare, (gx, gy), gr, 1.0, -1, cv2.LINE_AA)
    blur_k = max(15, gr) | 1
    glare = cv2.GaussianBlur(glare, (blur_k, blur_k), 0)
    alpha = glare * random.uniform(0.10, 0.35)
    white = np.full_like(img, 255, dtype=np.float32)
    result = img.astype(np.float32) * (1.0 - alpha[..., None]) + white * alpha[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def _add_glass_reflection(
    img: np.ndarray, cx: int, cy: int, radius: int
) -> np.ndarray:
    """Diagonal stripe reflection across the glass."""
    h, w = img.shape[:2]
    reflection = np.zeros((h, w), dtype=np.float32)
    # Diagonal stripe
    angle = random.uniform(20, 70)
    stripe_w = random.randint(radius // 8, radius // 4)
    offset = random.randint(-radius // 2, radius // 2)

    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.deg2rad(angle)
    proj = (xx - cx) * np.cos(rad) + (yy - cy) * np.sin(rad) - offset
    stripe = np.exp(-0.5 * (proj / max(1, stripe_w)) ** 2)

    # Limit to clock face
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    face_mask = (dist < radius * 0.95).astype(np.float32)
    stripe *= face_mask

    alpha = stripe * random.uniform(0.08, 0.20)
    white = np.full_like(img, 255, dtype=np.float32)
    result = img.astype(np.float32) * (1.0 - alpha[..., None]) + white * alpha[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def _add_face_yellowing(
    img: np.ndarray, cx: int, cy: int, radius: int
) -> np.ndarray:
    """Slight warm/yellow tint simulating an aged clock face."""
    h, w = img.shape[:2]
    dist = np.sqrt((np.mgrid[0:h, 0:w][1] - cx) ** 2 + (np.mgrid[0:h, 0:w][0] - cy) ** 2)
    face_mask = (dist < radius * 0.95).astype(np.float32)
    strength = random.uniform(0.03, 0.12)
    tint = np.zeros_like(img, dtype=np.float32)
    tint[:, :, 0] = 20   # B (less)
    tint[:, :, 1] = 30   # G
    tint[:, :, 2] = 45   # R (warm)
    result = img.astype(np.float32) + tint * (face_mask[..., None] * strength)
    return np.clip(result, 0, 255).astype(np.uint8)


# ======================================================================
# Module-level drawing helpers (stateless)
# ======================================================================

def _decorate_face(img: np.ndarray, radius: int, number_prob: float = 0.85) -> None:
    """Draw ticks and numbers on a circular clock face."""
    center = (radius, radius)

    # Contrast-aware colour
    if np.mean(img) > 127:
        color = tuple(random.randint(0, 60) for _ in range(3))
    else:
        color = tuple(random.randint(200, 255) for _ in range(3))

    tick_style = random.choice(
        ["lines", "thick_lines", "dots", "squares", "triangles", "rings", "minimal"]
    )

    for i in range(60):
        angle = i * 6 * (np.pi / 180)
        is_hour = i % 5 == 0
        if tick_style == "minimal" and not is_hour:
            continue

        r_out = radius - radius * 0.03

        if tick_style == "dots":
            r_pos = r_out - 5
            x = int(center[0] + r_pos * np.cos(angle - np.pi / 2))
            y = int(center[1] + r_pos * np.sin(angle - np.pi / 2))
            cv2.circle(img, (x, y), 3 if is_hour else 1, color, -1, cv2.LINE_AA)

        elif tick_style in ("squares", "triangles") and is_hour:
            r_pos = r_out - 8
            x = int(center[0] + r_pos * np.cos(angle - np.pi / 2))
            y = int(center[1] + r_pos * np.sin(angle - np.pi / 2))
            sz = 4
            if tick_style == "squares":
                cv2.rectangle(img, (x - sz, y - sz), (x + sz, y + sz), color, -1)
            else:
                fill = -1 if random.random() > 0.5 else 2
                cv2.circle(img, (x, y), sz + 1, color, fill, cv2.LINE_AA)

        elif tick_style == "rings":
            if i == 0:
                cv2.circle(img, center, int(radius * 0.95), color, 1, cv2.LINE_AA)
            if is_hour:
                r_in = radius * 0.85
                x1 = int(center[0] + r_out * np.cos(angle - np.pi / 2))
                y1 = int(center[1] + r_out * np.sin(angle - np.pi / 2))
                x2 = int(center[0] + r_in * np.cos(angle - np.pi / 2))
                y2 = int(center[1] + r_in * np.sin(angle - np.pi / 2))
                cv2.line(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        else:
            length = radius * 0.12 if is_hour else radius * 0.04
            thickness = 2 if (is_hour or tick_style == "thick_lines") else 1
            if is_hour and tick_style == "thick_lines":
                thickness = 3
            r_in = r_out - length
            x1 = int(center[0] + r_out * np.cos(angle - np.pi / 2))
            y1 = int(center[1] + r_out * np.sin(angle - np.pi / 2))
            x2 = int(center[0] + r_in * np.cos(angle - np.pi / 2))
            y2 = int(center[1] + r_in * np.sin(angle - np.pi / 2))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Numbers
    if random.random() < number_prob:
        font = random.choice(FONTS)
        use_roman = random.random() < 0.3
        font_scale = radius * random.uniform(0.003, 0.005)
        thickness = random.randint(1, 2)
        if font in (cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX):
            font_scale *= 1.5
        r_num = radius * 0.75

        for n in range(1, 13):
            angle = n * 30 * (np.pi / 180)
            text = ROMAN[n] if use_roman else str(n)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = int(center[0] + r_num * np.cos(angle - np.pi / 2)) - tw // 2
            y = int(center[1] + r_num * np.sin(angle - np.pi / 2)) + th // 2
            cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_single_hand(
    center: Tuple[int, int],
    length: float,
    width: float,
    color: Tuple[int, int, int],
    style: str,
    scene_size: Tuple[int, int],
) -> np.ndarray:
    """Draw one hand (pointing straight up / 12-o'clock) on an RGBA canvas."""
    w, h = scene_size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    cx, cy = center
    length = int(length)
    width = max(1, int(width))
    rgba = list(color) + [255]

    pts: Optional[np.ndarray] = None

    if style == "pointed":
        pts = np.array([[cx, cy - length], [cx + width, cy], [cx, cy + width], [cx - width, cy]])
    elif style == "arrow":
        ah = int(width * 2)
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + ah, cy - length + ah],
            [cx + hw, cy - length + ah], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, cy - length + ah],
            [cx - ah, cy - length + ah],
        ])
    elif style == "diamond":
        ds = int(width * 1.5)
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + ds, cy - length + ds],
            [cx + hw, cy - length + ds * 2], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, cy - length + ds * 2],
            [cx - ds, cy - length + ds],
        ])
    elif style == "tapered":
        pts = np.array([
            [cx, cy - length],
            [cx + width // 4, int(cy - length * 0.7)],
            [cx + width // 2, cy],
            [cx - width // 2, cy],
            [cx - width // 4, int(cy - length * 0.7)],
        ])
    elif style == "sword":
        bw = width // 3
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + bw, int(cy - length * 0.85)],
            [cx + hw, int(cy - length * 0.85)], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, int(cy - length * 0.85)],
            [cx - bw, int(cy - length * 0.85)],
        ])
    elif style == "lollipop":
        hw = width // 2
        shaft = np.array([
            [cx - hw, cy], [cx - hw, cy - length + width * 2],
            [cx + hw, cy - length + width * 2], [cx + hw, cy],
        ], dtype=np.int32)
        cv2.fillPoly(img, [shaft], rgba, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - length + width), width, rgba, -1, cv2.LINE_AA)
        return img
    elif style == "skeleton":
        hw = width // 2
        iw = width // 3
        outer = np.array([
            [cx - hw, cy], [cx - hw, cy - length],
            [cx + hw, cy - length], [cx + hw, cy],
        ], dtype=np.int32)
        inner = np.array([
            [cx - iw // 2, cy - width], [cx - iw // 2, cy - length + width],
            [cx + iw // 2, cy - length + width], [cx + iw // 2, cy - width],
        ], dtype=np.int32)
        cv2.fillPoly(img, [outer], rgba, cv2.LINE_AA)
        cv2.fillPoly(img, [inner], [0, 0, 0, 0], cv2.LINE_AA)
        return img
    elif style == "baton":
        tw = width // 3
        pts = np.array([
            [cx - tw, cy], [cx - tw, cy - length],
            [cx + tw, cy - length], [cx + tw, cy],
        ])
    elif style == "leaf":
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + width, int(cy - length * 0.6)],
            [cx + hw, cy], [cx - hw, cy],
            [cx - width, int(cy - length * 0.6)],
        ])
    elif style == "pencil":
        hw = width // 2
        tw = width // 3
        pts = np.array([
            [cx, cy - length], [cx + tw, int(cy - length * 0.9)],
            [cx + hw, cy], [cx - hw, cy],
            [cx - tw, int(cy - length * 0.9)],
        ])
    elif style == "dauphine":
        hw = width // 2
        tw = width // 3
        pts = np.array([
            [cx, cy - length], [cx + tw, int(cy - length * 0.7)],
            [cx + width, int(cy - length * 0.5)],
            [cx + hw, cy], [cx - hw, cy],
            [cx - width, int(cy - length * 0.5)],
            [cx - tw, int(cy - length * 0.7)],
        ])
    elif style == "breguet":
        hw = width // 2
        rect = np.array([
            [cx - hw, cy], [cx - hw, cy - length],
            [cx + hw, cy - length], [cx + hw, cy],
        ], dtype=np.int32)
        cv2.fillPoly(img, [rect], rgba, cv2.LINE_AA)
        hole_r = int(width * 0.8)
        cv2.circle(img, (cx, cy - int(length * 0.75)), hole_r, [0, 0, 0, 0], -1, cv2.LINE_AA)
        return img
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
        ])
    elif style == "anchor":
        tw = width // 3
        ew = int(width * 1.5)
        pts = np.array([
            [cx - tw, cy], [cx - tw, int(cy - length * 0.85)],
            [cx - ew, int(cy - length * 0.85)], [cx - ew, int(cy - length * 0.7)],
            [cx - tw, int(cy - length * 0.7)], [cx - tw, cy - length],
            [cx + tw, cy - length], [cx + tw, int(cy - length * 0.7)],
            [cx + ew, int(cy - length * 0.7)], [cx + ew, int(cy - length * 0.85)],
            [cx + tw, int(cy - length * 0.85)], [cx + tw, cy],
        ])
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
        ])
    elif style == "alpha":
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [int(cx + width * 1.5), int(cy - length * 0.6)],
            [cx + hw, cy], [cx - hw, cy],
            [int(cx - width * 1.5), int(cy - length * 0.6)],
        ])
    elif style == "feuille":
        tw = width // 3
        pts = np.array([
            [cx, cy - length], [int(cx + width * 0.8), int(cy - length * 0.5)],
            [cx + tw, cy], [cx - tw, cy],
            [int(cx - width * 0.8), int(cy - length * 0.5)],
        ])
    elif style == "lance":
        qw = width // 4
        pts = np.array([
            [cx, cy - length], [int(cx + width * 0.6), int(cy - length * 0.9)],
            [cx + qw, int(cy - length * 0.8)], [cx + qw, cy],
            [cx - qw, cy], [cx - qw, int(cy - length * 0.8)],
            [int(cx - width * 0.6), int(cy - length * 0.9)],
        ])
    elif style == "plongeur":
        ow = int(width * 0.7)
        pts = np.array([
            [cx - ow, cy], [cx - ow, int(cy - length * 0.9)],
            [cx, cy - length],
            [cx + ow, int(cy - length * 0.9)], [cx + ow, cy],
        ])
    elif style == "syringe":
        tw = width // 4
        hw = width // 2
        pts = np.array([
            [cx, cy - length], [cx + tw, int(cy - length * 0.95)],
            [cx + hw, int(cy - length * 0.9)], [cx + hw, cy],
            [cx - hw, cy], [cx - hw, int(cy - length * 0.9)],
            [cx - tw, int(cy - length * 0.95)],
        ])
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
        ])
    else:  # rectangle / modern / default
        hw = width // 2
        pts = np.array([
            [cx - hw, cy], [cx - hw, cy - length],
            [cx + hw, cy - length], [cx + hw, cy],
        ])

    if pts is not None:
        cv2.fillPoly(img, [pts.astype(np.int32)], rgba, cv2.LINE_AA)
    return img


def _draw_second_hand(
    center: Tuple[int, int],
    length: float,
    width: int,
    color: Tuple[int, int, int],
    scene_size: Tuple[int, int],
) -> np.ndarray:
    """Draw a thin second hand with tail, pointing straight up."""
    w, h = scene_size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    cx, cy = center
    length = int(length)
    tail = int(length * 0.2)
    rgba = list(color) + [255]

    pts = np.array([
        [cx - width // 2, cy + tail],
        [cx - width // 2, cy - length],
        [cx + width // 2, cy - length],
        [cx + width // 2, cy + tail],
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts], rgba, cv2.LINE_AA)
    cv2.circle(img, center, int(width * 3), rgba, -1, cv2.LINE_AA)
    return img
