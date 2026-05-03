"""
Pipeline runner functions for the full clock pipeline.

High-level orchestrators that load models, run inference, and display results.
Import this module in full-pipeline.ipynb to keep notebook cells short.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analog_clock.analog_sketch_creator import draw_analog_clock
from analog_clock.GAN.inpainting.generator_model import InpaintGenerator
from analog_clock.GAN.sketch.generator_model import GeneratorUNet
from analog_clock.pipeline_utils import (
    generate_inpainting_gif,
    generate_sketch_cgan_gif,
    get_angle_clockwise_from_12,
    load_time_recognition_cnn,
    recognize_time_from_masks,
    recompose_hand,
    time_to_degrees_cw,
)
from digital_clock.svhn_digit_recognition_cnn.svhn_cnn_model import SVHNModel


# ── center-finding (preserved exactly from original notebook) ────────────────
# pipeline_utils._find_center_hough uses minRadius=h*0.10 / maxRadius=h*0.90,
# which is too loose for clock faces and produces wrong centres.
# These functions use the original tight window (35-55 % of height).

def _pca_line(mask: np.ndarray):
    """Return (centre_point, direction_vector) via PCA, or (None, None)."""
    y_idx, x_idx = np.where(mask > 0)
    if len(x_idx) == 0:
        return None, None
    pts = np.column_stack([x_idx.astype(np.float64), y_idx.astype(np.float64)])
    mean, vecs = cv2.PCACompute(pts, mean=None)
    return (mean[0, 0], mean[0, 1]), (vecs[0, 0], vecs[0, 1])


def _pca_intersect(line1, line2):
    """Intersection of two (point, vector) lines; returns None for parallel."""
    if line1[0] is None or line2[0] is None:
        return None
    p1, v1 = line1
    p2, v2 = line2
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    try:
        t = np.linalg.solve(A, b)[0]
        return (int(p1[0] + t * v1[0]), int(p1[1] + t * v1[1]))
    except np.linalg.LinAlgError:
        return None


def _find_clock_center(img_rgb: np.ndarray, mask_h: np.ndarray,
                       mask_m: np.ndarray, mask_s: np.ndarray) -> tuple:
    """
    Hybrid centre finder identical to the original notebook's find_best_center_hybrid.
    Uses Hough with clock-specific radius bounds (35–55 % of height), then falls
    back to PCA intersection of the three hand masks.
    """
    h, w = img_rgb.shape[:2]
    geo = np.array([w / 2, h / 2])

    # 1. Hough Circles with tight clock-face radius window
    gray = cv2.medianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), 7)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, h / 4,
        param1=100, param2=35,
        minRadius=int(h * 0.35), maxRadius=int(h * 0.55),
    )
    hough = None
    if circles is not None:
        for c in np.uint16(np.around(circles))[0]:
            d = np.linalg.norm(np.array([c[0], c[1]], float) - geo)
            if d < w * 0.15 and (hough is None or d < np.linalg.norm(np.array(hough, float) - geo)):
                hough = (int(c[0]), int(c[1]))

    # 2. PCA intersection of all three hand lines
    line_h = _pca_line(mask_h)
    line_m = _pca_line(mask_m)
    line_s = _pca_line(mask_s)
    pts = [_pca_intersect(a, b) for a, b in [(line_h, line_m), (line_m, line_s), (line_s, line_h)]]
    valid = [p for p in pts if p and np.linalg.norm(np.array(p, float) - geo) < w * 0.2]
    pca = (int(np.mean([p[0] for p in valid])), int(np.mean([p[1] for p in valid]))) if valid else None

    if hough and pca and np.linalg.norm(np.array(hough, float) - np.array(pca, float)) < w * 0.05:
        return (hough[0] + pca[0]) // 2, (hough[1] + pca[1]) // 2
    return hough or pca or (w // 2, h // 2)


# ── shared transforms ────────────────────────────────────────────────────────

_PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

_CGAN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# ── private helpers ───────────────────────────────────────────────────────────

def _decode_digits(predictions) -> int:
    digits = [
        str(torch.argmax(head, dim=1).item())
        for head in predictions
        if torch.argmax(head, dim=1).item() != 10
    ]
    return int("".join(digits)) if digits else 0


def _crop_bbox(image: Image.Image, bbox, pad_pct: float = 0.1) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = bbox
    pw, ph = (x2 - x1) * pad_pct, (y2 - y1) * pad_pct
    return image.crop((
        max(0, int(x1 - pw)), max(0, int(y1 - ph)),
        min(w, int(x2 + pw)), min(h, int(y2 + ph)),
    ))


def _get_clock_crop(image: Image.Image, yolo_clock, padding_pct: float = 0.1):
    results = yolo_clock(image, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return image, (0, 0, image.width, image.height)
    x1, y1, x2, y2 = results[0].boxes[0].xyxy[0].tolist()
    pw, ph = (x2 - x1) * padding_pct, (y2 - y1) * padding_pct
    box = (
        max(0, int(x1 - pw)), max(0, int(y1 - ph)),
        min(image.width, int(x2 + pw)), min(image.height, int(y2 + ph)),
    )
    return image.crop(box), box


def _denormalize(tensor) -> Image.Image:
    img = (tensor.cpu().squeeze(0) * 0.5 + 0.5).permute(1, 2, 0).numpy()
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def _paste_generated(original: Image.Image, generated: Image.Image, crop_coords) -> Image.Image:
    x1, y1, x2, y2 = crop_coords
    result = original.copy()
    result.paste(generated.resize((x2 - x1, y2 - y1), Image.LANCZOS), (x1, y1))
    return result


# ── Stage 1: Digital Clock ────────────────────────────────────────────────────

def load_digital_models(yolo_path: str, cnn_path: str, device: torch.device):
    """Return (yolo_digit, cnn_model)."""
    yolo = YOLO(yolo_path)
    model = SVHNModel(num_channels=1).to(device)
    ckpt = torch.load(cnn_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    print("Digital models loaded.")
    return yolo, model


def process_clock_image(img_path: str, yolo, cnn_model, device: torch.device):
    """Run digital recognition; display 4-panel figure; return (hh, mm, sketch_array)."""
    original = Image.open(img_path).convert("RGB")
    results = yolo(original, verbose=False)
    hh = mm = None
    crops: dict = {}

    for box in results[0].boxes:
        label = results[0].names[int(box.cls[0])]
        crop = _crop_bbox(original, box.xyxy[0].tolist())
        crops[label] = crop
        with torch.no_grad():
            num = _decode_digits(cnn_model(_PREPROCESS(crop).unsqueeze(0).to(device)))
        if label in ("hours", "hh", "0"):
            hh = num
        elif label in ("minutes", "mm", "1"):
            mm = num

    if hh is None or mm is None:
        print("Could not detect both hours and minutes.")
        return None, None, None

    print(f"Detected time: {hh}:{mm:02d}")
    sketch = draw_analog_clock(hh, mm, return_array=True)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(original);
    axes[0].set_title("Original");      axes[0].axis("off")
    axes[1].imshow(crops.get("hours") or crops.get("hh") or list(crops.values())[0])
    axes[1].set_title(f"Hours: {hh}");  axes[1].axis("off")
    axes[2].imshow(crops.get("minutes") or crops.get("mm") or list(crops.values())[-1])
    axes[2].set_title(f"Minutes: {mm:02d}"); axes[2].axis("off")
    axes[3].imshow(sketch);
    axes[3].set_title("Generated Sketch"); axes[3].axis("off")
    plt.tight_layout(); plt.show()
    return hh, mm, sketch


# ── Stage 2: Sketch-cGAN ──────────────────────────────────────────────────────

def load_sketch_cgan_models(yolo_clock_path: str, generator_path: str, device: torch.device):
    """Return (yolo_clock, generator)."""
    yolo_clock = YOLO(yolo_clock_path)
    gen = GeneratorUNet(in_channels=6, out_channels=3).to(device)
    gen.load_state_dict(torch.load(generator_path, map_location=device))
    gen.eval()
    print("Sketch-cGAN models loaded.")
    return yolo_clock, gen


def run_sketch_cgan(img_path: str, hh: int, mm: int, yolo_clock, generator, device: torch.device):
    """Run Sketch-cGAN; display 4-panel result; return (result, original, crop_coords)."""
    original = Image.open(img_path).convert("RGB")
    crop, crop_coords = _get_clock_crop(original, yolo_clock)

    src_t = _CGAN_TRANSFORM(crop).unsqueeze(0).to(device)
    sketch_arr = draw_analog_clock(hh, mm, return_array=True)
    skc_t = _CGAN_TRANSFORM(
        Image.fromarray(sketch_arr).convert("RGB").resize((256, 256))
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        gen_t = generator(torch.cat((src_t, skc_t), 1))

    gen_img = _denormalize(gen_t)
    result = _paste_generated(original, gen_img, crop_coords)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, title in zip(
        axes,
        [original, crop, Image.fromarray(sketch_arr).resize((256, 256)), result],
        ["Original", "Detected Clock", f"Target Sketch ({hh}:{mm:02d})", "Final Result"],
    ):
        ax.imshow(img); ax.set_title(title); ax.axis("off")
    plt.tight_layout(); plt.show()
    return result, original, crop_coords


def recognize_original_time(
    img_path: str, yolo_hands_path: str, time_cnn_path: str, device: torch.device
):
    """Detect original time from an analog clock image. Returns (hh, mm)."""
    yolo_tmp = YOLO(yolo_hands_path)
    img_arr = np.array(Image.open(img_path).convert("RGB").resize((256, 256)))
    res = yolo_tmp(img_arr, conf=0.05, verbose=False)
    if res[0].masks is not None:
        combined = np.zeros((256, 256), dtype=np.float32)
        for m in res[0].masks.data.cpu().numpy():
            combined = np.maximum(combined, cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST))
        time_cnn = load_time_recognition_cnn(time_cnn_path, device)
        hh, mm = recognize_time_from_masks(combined, time_cnn, device)
    else:
        print("No hands detected — defaulting to 12:00")
        hh, mm = 0, 0
    print(f"Recognised original time: {hh}:{mm:02d}")
    return hh, mm


# ── Stage 3: Inpainting ───────────────────────────────────────────────────────

def load_inpainting_models(yolo_hands_path: str, igan_path: str, device: torch.device):
    """Return (yolo_hands, igan)."""
    yolo_hands = YOLO(yolo_hands_path)
    igan = InpaintGenerator().to(device)
    try:
        igan.load_state_dict(torch.load(igan_path, map_location=device))
    except RuntimeError:
        igan.load_state_dict(torch.load(igan_path, map_location="cpu"))
        igan.to(device)
    igan.eval()
    print("Inpainting models loaded.")
    return yolo_hands, igan


def run_inpainting(img_path: str, hh: int, mm: int, igan, yolo_hands, device: torch.device) -> dict:
    """Run inpainting pipeline; display 4-panel result; return state dict for GIF generation."""
    img_pil = Image.open(img_path).convert("RGB").resize((256, 256))
    img_cv = np.array(img_pil)

    results = yolo_hands(img_cv, conf=0.05, verbose=False)
    dets_h, dets_m, dets_s = [], [], []
    if results[0].masks is not None:
        for mask, cls, conf in zip(
            results[0].masks.data.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
        ):
            m = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            if cls == 0:
                dets_h.append((m, conf))
            elif cls == 1:
                dets_m.append((m, conf))
            elif cls == 2:
                dets_s.append((m, conf))

    if not dets_m and dets_s:
        dets_m = dets_s

    def _best(dets):
        if not dets:
            return np.zeros((256, 256), dtype=np.uint8)
        return max(dets, key=lambda x: x[1])[0]

    def _union(dets):
        out = np.zeros((256, 256), dtype=np.uint8)
        for m, _ in dets:
            out = np.maximum(out, m)
        return out

    mask_h = _best(dets_h)
    mask_m = _best(dets_m)
    mask_s = _best(dets_s)
    full_bin        = (_union(dets_h + dets_m) > 0.5).astype(np.float32)
    dilated_cgan    = cv2.dilate(full_bin, np.ones((7,  7),  np.uint8), iterations=1)
    dilated_inpaint = cv2.dilate(full_bin, np.ones((11, 11), np.uint8), iterations=1)

    # GAN inpainting pass
    _norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    img_t  = _norm(img_pil).to(device)
    mask_t = (torch.from_numpy(dilated_cgan).unsqueeze(0).to(device) - 0.5) / 0.5
    with torch.no_grad():
        clean_cgan = (
            igan(torch.cat([img_t, mask_t], 0).unsqueeze(0))
            .squeeze().permute(1, 2, 0).cpu().numpy()
        )
    clean_cgan = ((clean_cgan * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    clean_bg = cv2.inpaint(
        clean_cgan, (dilated_inpaint * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA
    )

    # Center + angles — uses original notebook's tight Hough radius window
    center  = _find_clock_center(img_cv, mask_h, mask_m, mask_s)
    angle_h = get_angle_clockwise_from_12(mask_h, center)
    angle_m = get_angle_clockwise_from_12(mask_m, center)

    result = recompose_hand(mask_h, img_cv, clean_bg, center,
                            angle_h, time_to_degrees_cw(hh, mm, "hour"),   feather_radius=5)
    result = recompose_hand(mask_m, img_cv, result,   center,
                            angle_m, time_to_degrees_cw(hh, mm, "minute"), feather_radius=5)
    cv2.circle(result, center, 4, (255, 0, 0), -1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, title, gray in zip(
        axes,
        [img_cv, (dilated_cgan * 255).astype(np.uint8), clean_bg, result],
        ["Original Input", "Detected Mask", "Cleaned Background", f"New Time ({hh}:{mm:02d})"],
        [False, True, False, False],
    ):
        ax.imshow(img, cmap="gray" if gray else None)
        ax.set_title(title); ax.axis("off")
    plt.tight_layout(); plt.show()

    return {
        "img_cv":       img_cv,
        "clean_bg":     clean_bg,
        "result":       result,
        "center":       center,
        "best_mask_h":  mask_h,
        "best_mask_m":  mask_m,
        "angle_h_orig": angle_h,
        "angle_m_orig": angle_m,
        "mask_display": (dilated_cgan * 255).astype(np.uint8),
    }
