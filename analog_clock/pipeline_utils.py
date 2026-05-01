"""
Pipeline utilities for the full clock pipeline.

Provides:
- Time recognition from binary masks using ClockHandCNN
- Hybrid clock-center finding (Hough + PCA + fitLine + fallback)
- Hand angle measurement via skeletonisation
- Improved hand recomposition with multi-scale feathered alpha blending
- GIF animation generation for both analog clock paths
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from analog_clock.time_recognition_cnn.model import ClockHandCNN

logger = logging.getLogger(__name__)


# ============================================================================
# Time Recognition
# ============================================================================

def decode_sincos(pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode sin/cos encoded predictions to hours [0,12) and minutes [0,60)."""
    h_angle = torch.atan2(pred[:, 0], pred[:, 1]) % (2 * math.pi)
    m_angle = torch.atan2(pred[:, 2], pred[:, 3]) % (2 * math.pi)
    hours = h_angle / (2 * math.pi) * 12.0
    minutes = m_angle / (2 * math.pi) * 60.0
    return hours, minutes


def load_time_recognition_cnn(
    weights_path: str,
    device: torch.device,
) -> ClockHandCNN:
    """Load trained ClockHandCNN weights."""
    model = ClockHandCNN().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def recognize_time_from_masks(
    masks: np.ndarray,
    model: ClockHandCNN,
    device: torch.device,
    img_size: int = 256,
) -> Tuple[int, int]:
    """
    Recognize the current time from YOLO-seg binary masks.

    Returns (hours, minutes) as integers, 12-hour format.
    """
    mask_uint8 = (masks > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(
        mask_uint8, (img_size, img_size), interpolation=cv2.INTER_NEAREST
    )

    tensor = (
        torch.from_numpy(mask_resized)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    ).to(device)

    with torch.no_grad():
        pred = model(tensor)

    hours, minutes = decode_sincos(pred)
    h = int(round(hours.item())) % 12
    m = int(round(minutes.item())) % 60
    return h, m


# ============================================================================
# Clock-Center Finding  (hybrid: Hough → PCA → fitLine → fallback)
# ============================================================================

def _get_orientation_pca(
    mask: np.ndarray,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """PCA on mask pixels → (centroid, principal_direction) or (None, None)."""
    y_idxs, x_idxs = np.where(mask > 0)
    if len(x_idxs) < 10:
        return None, None
    pts = np.empty((len(x_idxs), 2), dtype=np.float64)
    pts[:, 0] = x_idxs
    pts[:, 1] = y_idxs
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    return (mean[0, 0], mean[0, 1]), (eigenvectors[0, 0], eigenvectors[0, 1])


def _get_line_params(
    mask: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """cv2.fitLine on mask → (vx, vy, x0, y0) or None."""
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None or len(points) < 10:
        return None
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(line[0][0]), float(line[1][0]), float(line[2][0]), float(line[3][0])


def _intersect_lines(
    line1: Optional[Tuple[float, float, float, float]],
    line2: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[int, int]]:
    """Intersect two (vx, vy, x0, y0) lines. Returns pixel coords or None."""
    if line1 is None or line2 is None:
        return None
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2
    det = vx1 * vy2 - vy1 * vx2
    if abs(det) < 1e-6:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det
    return int(x1 + t * vx1), int(y1 + t * vy1)


def _intersect_pca_lines(
    line1: Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]],
    line2: Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]],
) -> Optional[Tuple[int, int]]:
    """Intersect two PCA lines ((point, direction), …). Returns pixel coords."""
    p1, v1 = line1
    p2, v2 = line2
    if p1 is None or p2 is None or v1 is None or v2 is None:
        return None
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    t = x[0]
    return int(p1[0] + t * v1[0]), int(p1[1] + t * v1[1])


def _find_center_hough(
    img_rgb: np.ndarray,
    *,
    min_radius_ratio: float = 0.1,
    max_radius_ratio: float = 0.9,
    max_dist_ratio: float = 0.15,
) -> Optional[Tuple[int, int]]:
    """Detect the clock circle via HoughCircles and return the best center."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        h / 4,
        param1=100,
        param2=35,
        minRadius=int(h * min_radius_ratio),
        maxRadius=int(h * max_radius_ratio),
    )
    if circles is None:
        return None

    img_center = np.array([w / 2, h / 2])
    best_center: Optional[Tuple[int, int]] = None
    min_dist = float("inf")
    for c in np.uint16(np.around(circles))[0]:
        dist = np.linalg.norm(np.array([c[0], c[1]], dtype=float) - img_center)
        if dist < min_dist and dist < w * max_dist_ratio:
            min_dist = dist
            best_center = (int(c[0]), int(c[1]))
    return best_center


def _find_center_fitline(
    masks: List[np.ndarray],
    image_shape: Tuple[int, ...],
    max_dist_ratio: float = 0.25,
) -> Optional[Tuple[int, int]]:
    """Center via cv2.fitLine intersection of hand masks."""
    h, w = image_shape[:2]
    geo_center = np.array([w / 2, h / 2])
    max_px = min(h, w) * max_dist_ratio

    lines = [_get_line_params(m) for m in masks]
    intersections: List[Tuple[int, int]] = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = _intersect_lines(lines[i], lines[j])
            if pt is not None:
                dist = np.linalg.norm(np.array(pt, dtype=float) - geo_center)
                if dist < max_px:
                    intersections.append(pt)

    if intersections:
        avg_x = int(np.mean([p[0] for p in intersections]))
        avg_y = int(np.mean([p[1] for p in intersections]))
        return avg_x, avg_y
    return None


def _find_center_pca(
    masks: List[np.ndarray],
    image_shape: Tuple[int, ...],
    max_dist_ratio: float = 0.20,
) -> Optional[Tuple[int, int]]:
    """Center via PCA intersection of hand masks."""
    h, w = image_shape[:2]
    geo_center = np.array([w / 2, h / 2])
    max_px = min(h, w) * max_dist_ratio

    pca_lines = [_get_orientation_pca(m) for m in masks]
    intersections: List[Tuple[int, int]] = []
    for i in range(len(pca_lines)):
        for j in range(i + 1, len(pca_lines)):
            pt = _intersect_pca_lines(pca_lines[i], pca_lines[j])
            if pt is not None:
                dist = np.linalg.norm(np.array(pt, dtype=float) - geo_center)
                if dist < max_px:
                    intersections.append(pt)

    if intersections:
        return (
            int(np.mean([p[0] for p in intersections])),
            int(np.mean([p[1] for p in intersections])),
        )
    return None


def _find_center_dynamic(masks: List[np.ndarray], image_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Last-resort: hand pixel closest to image centre."""
    h, w = image_shape[:2]
    geo = np.array([w // 2, h // 2])
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m.shape[:2] == (h, w):
            combined = np.maximum(combined, m)
    y_idxs, x_idxs = np.where(combined > 0)
    if len(x_idxs) == 0:
        return w // 2, h // 2
    pts = np.stack((x_idxs, y_idxs), axis=1)
    dists = np.sum((pts - geo) ** 2, axis=1)
    best = pts[np.argmin(dists)]
    return int(best[0]), int(best[1])


def find_clock_center(
    img_rgb: np.ndarray,
    mask_h: np.ndarray,
    mask_m: np.ndarray,
    mask_s: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """
    Hybrid clock-center finder.

    Priority:
      1. Hough circle detection
      2. cv2.fitLine intersection (robust to thickness)
      3. PCA intersection
      4. Dynamic centroid (closest mask pixel to image centre)

    When Hough and another method agree (< 5% of width), their average is used
    for sub-pixel stability.
    """
    masks = [m for m in (mask_h, mask_m, mask_s) if m is not None and np.sum(m) > 0]
    h, w = img_rgb.shape[:2]

    hough = _find_center_hough(img_rgb)
    fitline = _find_center_fitline(masks, img_rgb.shape) if len(masks) >= 2 else None
    pca = _find_center_pca(masks, img_rgb.shape) if len(masks) >= 2 else None

    # If Hough and a line-based method agree, average for best accuracy
    if hough is not None:
        for alt in (fitline, pca):
            if alt is not None:
                if np.linalg.norm(np.array(hough) - np.array(alt)) < w * 0.05:
                    logger.debug("Hough + line-based average center")
                    return (hough[0] + alt[0]) // 2, (hough[1] + alt[1]) // 2
        logger.debug("Using Hough center")
        return hough

    if fitline is not None:
        logger.debug("Using fitLine intersection center")
        return fitline
    if pca is not None:
        logger.debug("Using PCA intersection center")
        return pca

    logger.debug("Falling back to dynamic centroid")
    return _find_center_dynamic(masks, img_rgb.shape)


# ============================================================================
# Hand Angle Measurement
# ============================================================================

def get_angle_clockwise_from_12(
    mask: np.ndarray,
    center: Tuple[int, int],
) -> float:
    """
    Measure a hand's angle (0-360, clockwise from 12 o'clock).

    Uses skeleton thinning to find the hand's centre-line, then locates the
    tip (farthest skeleton pixel from *center*) for a robust angle.
    """
    if np.sum(mask) == 0:
        return 0.0

    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    try:
        skeleton = cv2.ximgproc.thinning(mask_uint8)
    except (cv2.error, AttributeError):
        skeleton = mask_uint8

    y_idxs, x_idxs = np.where(skeleton > 0)
    if len(x_idxs) == 0:
        return 0.0

    dists = (x_idxs - center[0]) ** 2 + (y_idxs - center[1]) ** 2
    tip_idx = np.argmax(dists)
    vec_x = x_idxs[tip_idx] - center[0]
    vec_y = y_idxs[tip_idx] - center[1]

    # atan2(x, -y) gives 0 at 12 o'clock, increasing clockwise
    angle_deg = math.degrees(math.atan2(vec_x, -vec_y))
    return angle_deg % 360


# ============================================================================
# Improved Hand Recomposition  (multi-scale feathering + edge refinement)
# ============================================================================

def extract_hand_rgba(
    mask: np.ndarray,
    img_rgb: np.ndarray,
    feather_radius: int = 5,
) -> np.ndarray:
    """
    Extract a hand as RGBA with multi-scale feathered alpha.

    Two-pass Gaussian blur (fine + coarse) creates a smooth, natural
    falloff that reduces jagged artefacts after rotation.
    """
    mask_uint8 = ((mask > 0).astype(np.uint8) * 255) if mask.max() <= 1 else mask.astype(np.uint8)

    # Fine edge feathering
    k_fine = max(3, feather_radius * 2 + 1)
    alpha_fine = cv2.GaussianBlur(mask_uint8, (k_fine, k_fine), 0)

    # Coarser pass for the outermost fringe
    k_coarse = max(5, feather_radius * 4 + 1)
    alpha_coarse = cv2.GaussianBlur(mask_uint8, (k_coarse, k_coarse), 0)

    # Blend: use fine inside, coarse on the outer fringe
    alpha = np.where(mask_uint8 > 0, alpha_fine, alpha_coarse).astype(np.uint8)

    b, g, r = cv2.split(img_rgb)
    return cv2.merge([b, g, r, alpha])


def rotate_hand_rgba(
    hand_rgba: np.ndarray,
    rotation_deg: float,
    center: Tuple[int, int],
) -> np.ndarray:
    """Rotate an RGBA hand image around *center* by *rotation_deg* CW."""
    h, w = hand_rgba.shape[:2]
    M = cv2.getRotationMatrix2D(center, -rotation_deg, 1.0)
    return cv2.warpAffine(
        hand_rgba,
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def blend_rgba_onto_rgb(
    background: np.ndarray,
    overlay_rgba: np.ndarray,
) -> np.ndarray:
    """Alpha-blend an RGBA overlay onto an RGB background."""
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = overlay_rgba[:, :, :3].astype(np.float32)
    bg = background.astype(np.float32)
    blended = fg * alpha + bg * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def recompose_hand(
    mask: np.ndarray,
    img_rgb: np.ndarray,
    background: np.ndarray,
    center: Tuple[int, int],
    current_angle_cw: float,
    target_angle_cw: float,
    feather_radius: int = 5,
) -> np.ndarray:
    """
    Extract a hand from *img_rgb* using *mask*, rotate it from
    *current_angle_cw* to *target_angle_cw*, and blend onto *background*.
    """
    if np.sum(mask) == 0:
        return background

    hand_rgba = extract_hand_rgba(mask, img_rgb, feather_radius=feather_radius)
    rotation_diff = target_angle_cw - current_angle_cw
    rotated = rotate_hand_rgba(hand_rgba, rotation_diff, center)
    return blend_rgba_onto_rgb(background, rotated)


def recompose_hands_full(
    img_rgb: np.ndarray,
    clean_bg: np.ndarray,
    center: Tuple[int, int],
    mask_h: np.ndarray,
    mask_m: np.ndarray,
    current_hh: int,
    current_mm: int,
    target_hh: int,
    target_mm: int,
    feather_radius: int = 5,
    mask_s: Optional[np.ndarray] = None,
    current_ss: int = 0,
    target_ss: int = 0,
) -> np.ndarray:
    """
    High-level: recompose all hands from current time to target time.

    1. Compute current/target angles for each hand
    2. Extract RGBA with feathered alpha
    3. Rotate by angle difference
    4. Blend onto clean background
    """
    result = clean_bg.copy()

    # Hour
    cur_h_angle = time_to_degrees_cw(current_hh, current_mm, "hour")
    tgt_h_angle = time_to_degrees_cw(target_hh, target_mm, "hour")
    result = recompose_hand(
        mask_h, img_rgb, result, center, cur_h_angle, tgt_h_angle, feather_radius
    )

    # Minute
    cur_m_angle = time_to_degrees_cw(current_hh, current_mm, "minute")
    tgt_m_angle = time_to_degrees_cw(target_hh, target_mm, "minute")
    result = recompose_hand(
        mask_m, img_rgb, result, center, cur_m_angle, tgt_m_angle, feather_radius
    )

    # Second (optional)
    if mask_s is not None and np.sum(mask_s) > 0:
        cur_s_angle = current_ss * 6.0
        tgt_s_angle = target_ss * 6.0
        result = recompose_hand(
            mask_s, img_rgb, result, center, cur_s_angle, tgt_s_angle, feather_radius
        )

    return result


# ============================================================================
# Time Utilities
# ============================================================================

def time_to_degrees_cw(hh: int, mm: int, hand_type: str) -> float:
    """Convert HH:MM to clockwise degrees from 12 o'clock."""
    if hand_type == "hour":
        return ((hh % 12) * 30) + (mm * 0.5)
    if hand_type == "minute":
        return (mm % 60) * 6
    return 0.0


def generate_intermediate_times(
    start_hh: int,
    start_mm: int,
    end_hh: int,
    end_mm: int,
    step_minutes: int = 2,
) -> List[Tuple[int, int]]:
    """
    Generate (hh, mm) tuples from start to end (12-hour), stepping by
    *step_minutes*.  Handles wrap-around past 12:00.
    """
    start_total = (start_hh % 12) * 60 + start_mm
    end_total = (end_hh % 12) * 60 + end_mm
    if end_total <= start_total:
        end_total += 12 * 60

    times: List[Tuple[int, int]] = []
    t = start_total
    while t <= end_total:
        times.append(((t // 60) % 12, t % 60))
        t += step_minutes

    final = ((end_total // 60) % 12, end_total % 60)
    if not times or times[-1] != final:
        times.append(final)
    return times


# ============================================================================
# GIF Generation
# ============================================================================

def save_gif(
    frames: List[np.ndarray],
    output_path: str,
    duration_ms: int = 200,
    loop: int = 0,
) -> str:
    """Save a list of RGB numpy frames as an animated GIF."""
    if not frames:
        raise ValueError("No frames to save")
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
    )
    return output_path


def generate_sketch_cgan_gif(
    original_img_pil: Image.Image,
    crop_coords: Tuple[int, int, int, int],
    generator: nn.Module,
    transform,
    start_hh: int,
    start_mm: int,
    target_hh: int,
    target_mm: int,
    draw_analog_clock_fn,
    device: torch.device,
    step_minutes: int = 2,
    output_path: str = "sketch_cgan_animation.gif",
    duration_ms: int = 200,
) -> str:
    """
    Animate the sketch-cGAN path from start to target time.

    Each frame: render sketch → cGAN inference → paste back into scene.
    """
    times = generate_intermediate_times(
        start_hh, start_mm, target_hh, target_mm, step_minutes
    )

    x1, y1, x2, y2 = crop_coords
    cropped = original_img_pil.crop(crop_coords)
    src_tensor = transform(cropped).unsqueeze(0).to(device)

    frames: List[np.ndarray] = []
    for hh, mm in times:
        sketch_arr = draw_analog_clock_fn(hh, mm, return_array=True)
        sketch_pil = Image.fromarray(sketch_arr).convert("RGB").resize((256, 256))
        skc_tensor = transform(sketch_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            gen_tensor = generator(torch.cat((src_tensor, skc_tensor), 1))

        gen_img = (gen_tensor.cpu().squeeze(0) * 0.5 + 0.5).permute(1, 2, 0).numpy()
        gen_img = np.clip(gen_img * 255, 0, 255).astype(np.uint8)
        gen_pil = Image.fromarray(gen_img)

        gen_resized = gen_pil.resize((x2 - x1, y2 - y1), Image.LANCZOS)
        frame = original_img_pil.copy()
        frame.paste(gen_resized, (x1, y1))
        frames.append(np.array(frame))

    return save_gif(frames, output_path, duration_ms=duration_ms)


def generate_inpainting_gif(
    img_cv: np.ndarray,
    clean_bg: np.ndarray,
    center: Tuple[int, int],
    mask_h: np.ndarray,
    mask_m: np.ndarray,
    current_angle_h: float,
    current_angle_m: float,
    start_hh: int,
    start_mm: int,
    target_hh: int,
    target_mm: int,
    step_minutes: int = 2,
    output_path: str = "inpainting_animation.gif",
    duration_ms: int = 200,
    feather_radius: int = 5,
) -> str:
    """
    Animate the inpainting path from start to target time.

    Each frame: rotate extracted hands to intermediate angles, blend onto
    the clean (hand-removed) background.
    """
    times = generate_intermediate_times(
        start_hh, start_mm, target_hh, target_mm, step_minutes
    )

    hand_h_rgba = (
        extract_hand_rgba(mask_h, img_cv, feather_radius)
        if np.sum(mask_h) > 0
        else None
    )
    hand_m_rgba = (
        extract_hand_rgba(mask_m, img_cv, feather_radius)
        if np.sum(mask_m) > 0
        else None
    )

    frames: List[np.ndarray] = []
    for hh, mm in times:
        frame = clean_bg.copy()

        if hand_h_rgba is not None:
            target_h = time_to_degrees_cw(hh, mm, "hour")
            rotated_h = rotate_hand_rgba(hand_h_rgba, target_h - current_angle_h, center)
            frame = blend_rgba_onto_rgb(frame, rotated_h)

        if hand_m_rgba is not None:
            target_m = time_to_degrees_cw(hh, mm, "minute")
            rotated_m = rotate_hand_rgba(hand_m_rgba, target_m - current_angle_m, center)
            frame = blend_rgba_onto_rgb(frame, rotated_m)

        frames.append(frame)

    return save_gif(frames, output_path, duration_ms=duration_ms)
