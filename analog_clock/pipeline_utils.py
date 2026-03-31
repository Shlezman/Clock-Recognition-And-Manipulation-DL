"""
Pipeline utilities for the full clock pipeline.

Provides:
- Time recognition from binary masks using ClockHandCNN
- Improved hand recomposition with anti-aliased alpha blending
- GIF animation generation for both analog clock paths
"""

import math
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from analog_clock.time_recognition_cnn.model import ClockHandCNN


# ---------------------------------------------------------------------------
# Time Recognition
# ---------------------------------------------------------------------------

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

    Parameters
    ----------
    masks : np.ndarray
        Combined binary mask of all detected hands (H, W) with values in {0, 1}.
    model : ClockHandCNN
        Loaded and eval-mode CNN model.
    device : torch.device
    img_size : int
        Expected input size for the CNN (default 256).

    Returns
    -------
    (hours, minutes) as integers, 12-hour format.
    """
    # Ensure binary uint8 then resize
    mask_uint8 = (masks > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_uint8, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    # To tensor: (1, 1, H, W) float in [0, 1]
    tensor = torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        pred = model(tensor)

    hours, minutes = decode_sincos(pred)
    h = int(round(hours.item())) % 12
    m = int(round(minutes.item())) % 60
    return h, m


# ---------------------------------------------------------------------------
# Improved Hand Recomposition
# ---------------------------------------------------------------------------

def extract_hand_rgba(
    mask: np.ndarray,
    img_rgb: np.ndarray,
    feather_radius: int = 5,
) -> np.ndarray:
    """
    Extract a hand from the image as RGBA with feathered (anti-aliased) alpha.

    Uses Gaussian blur on the mask to create smooth edges, reducing jagged
    artefacts after rotation.
    """
    h, w = img_rgb.shape[:2]
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Feathered alpha via Gaussian blur
    ksize = feather_radius * 2 + 1
    alpha = cv2.GaussianBlur(mask_uint8, (ksize, ksize), 0)

    b, g, r = cv2.split(img_rgb)
    hand_rgba = cv2.merge([b, g, r, alpha])
    return hand_rgba


def rotate_hand_rgba(
    hand_rgba: np.ndarray,
    rotation_deg: float,
    center: Tuple[int, int],
) -> np.ndarray:
    """Rotate an RGBA hand image around *center* by *rotation_deg* (positive = CW on screen)."""
    h, w = hand_rgba.shape[:2]
    M = cv2.getRotationMatrix2D(center, -rotation_deg, 1.0)
    rotated = cv2.warpAffine(
        hand_rgba, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


def blend_rgba_onto_rgb(
    background: np.ndarray,
    overlay_rgba: np.ndarray,
) -> np.ndarray:
    """Alpha-blend an RGBA overlay onto an RGB background."""
    alpha = overlay_rgba[:, :, 3].astype(np.float32) / 255.0
    alpha_3 = np.dstack([alpha, alpha, alpha])
    fg = overlay_rgba[:, :, :3].astype(np.float32)
    bg = background.astype(np.float32)
    blended = fg * alpha_3 + bg * (1.0 - alpha_3)
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

    Returns the updated background.
    """
    if np.sum(mask) == 0:
        return background

    hand_rgba = extract_hand_rgba(mask, img_rgb, feather_radius=feather_radius)
    rotation_diff = target_angle_cw - current_angle_cw
    rotated = rotate_hand_rgba(hand_rgba, rotation_diff, center)
    return blend_rgba_onto_rgb(background, rotated)


# ---------------------------------------------------------------------------
# Time Utilities
# ---------------------------------------------------------------------------

def time_to_degrees_cw(hh: int, mm: int, hand_type: str) -> float:
    """Convert HH:MM to clockwise degrees from 12 o'clock."""
    if hand_type == "hour":
        return ((hh % 12) * 30) + (mm * 0.5)
    elif hand_type == "minute":
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
    Generate a list of (hh, mm) tuples from start to end time (12-hour),
    stepping forward by *step_minutes*.  Always includes the end time.

    Handles wrap-around (e.g. 11:50 → 0:10).
    """
    start_total = (start_hh % 12) * 60 + start_mm
    end_total = (end_hh % 12) * 60 + end_mm

    # Handle wrap-around: if end <= start, assume we pass 12:00
    if end_total <= start_total:
        end_total += 12 * 60

    times: List[Tuple[int, int]] = []
    t = start_total
    while t <= end_total:
        hh = (t // 60) % 12
        mm = t % 60
        times.append((hh, mm))
        t += step_minutes

    # Ensure the very last frame is the target time
    final = ((end_total // 60) % 12, end_total % 60)
    if not times or times[-1] != final:
        times.append(final)

    return times


# ---------------------------------------------------------------------------
# GIF Generation
# ---------------------------------------------------------------------------

def save_gif(
    frames: List[np.ndarray],
    output_path: str,
    duration_ms: int = 200,
    loop: int = 0,
) -> str:
    """
    Save a list of RGB numpy frames as an animated GIF.

    Parameters
    ----------
    frames : list of np.ndarray (H, W, 3) uint8
    output_path : str
    duration_ms : int  – per-frame duration
    loop : int – 0 = infinite loop

    Returns
    -------
    The output path.
    """
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
    Generate a GIF animating the sketch-cGAN path from start_time to target_time.

    For each intermediate time step:
      1. Render sketch at that time
      2. Run cGAN (source clock + sketch → generated clock)
      3. Paste back into original image
      4. Collect frame

    Returns the saved GIF path.
    """
    from analog_clock.analog_sketch_creator import draw_analog_clock

    times = generate_intermediate_times(start_hh, start_mm, target_hh, target_mm, step_minutes)

    # Pre-compute source tensor (doesn't change per frame)
    x1, y1, x2, y2 = crop_coords
    cropped = original_img_pil.crop(crop_coords)
    src_tensor = transform(cropped).unsqueeze(0).to(device)

    frames: List[np.ndarray] = []
    for hh, mm in times:
        sketch_arr = draw_analog_clock_fn(hh, mm, return_array=True)
        sketch_pil = Image.fromarray(sketch_arr).convert("RGB").resize((256, 256))
        skc_tensor = transform(sketch_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            inp = torch.cat((src_tensor, skc_tensor), 1)
            gen_tensor = generator(inp)

        # Denormalize
        gen_img = gen_tensor.cpu().squeeze(0) * 0.5 + 0.5
        gen_img = gen_img.permute(1, 2, 0).numpy()
        gen_img = np.clip(gen_img * 255, 0, 255).astype(np.uint8)
        gen_pil = Image.fromarray(gen_img)

        # Paste back
        target_w = x2 - x1
        target_h = y2 - y1
        gen_resized = gen_pil.resize((target_w, target_h), Image.LANCZOS)
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
    Generate a GIF animating the inpainting path from start_time to target_time.

    For each intermediate time step:
      1. Start from cleaned background (hands removed)
      2. Rotate extracted hour/minute hands to the step's angles
      3. Alpha-blend onto clean background
      4. Collect frame

    Returns the saved GIF path.
    """
    times = generate_intermediate_times(start_hh, start_mm, target_hh, target_mm, step_minutes)

    # Pre-extract hand RGBA images (once)
    hand_h_rgba = extract_hand_rgba(mask_h, img_cv, feather_radius=feather_radius) if np.sum(mask_h) > 0 else None
    hand_m_rgba = extract_hand_rgba(mask_m, img_cv, feather_radius=feather_radius) if np.sum(mask_m) > 0 else None

    frames: List[np.ndarray] = []
    for hh, mm in times:
        frame = clean_bg.copy()

        # Hour hand
        if hand_h_rgba is not None:
            target_angle_h = time_to_degrees_cw(hh, mm, "hour")
            rot_diff_h = target_angle_h - current_angle_h
            rotated_h = rotate_hand_rgba(hand_h_rgba, rot_diff_h, center)
            frame = blend_rgba_onto_rgb(frame, rotated_h)

        # Minute hand
        if hand_m_rgba is not None:
            target_angle_m = time_to_degrees_cw(hh, mm, "minute")
            rot_diff_m = target_angle_m - current_angle_m
            rotated_m = rotate_hand_rgba(hand_m_rgba, rot_diff_m, center)
            frame = blend_rgba_onto_rgb(frame, rotated_m)

        frames.append(frame)

    return save_gif(frames, output_path, duration_ms=duration_ms)
