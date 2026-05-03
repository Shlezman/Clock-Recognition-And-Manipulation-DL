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
    find_clock_center,
    generate_inpainting_gif,
    generate_sketch_cgan_gif,
    get_angle_clockwise_from_12,
    load_time_recognition_cnn,
    recognize_time_from_masks,
    recompose_hand,
    time_to_degrees_cw,
)
from digital_clock.svhn_digit_recognition_cnn.svhn_cnn_model import SVHNModel


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

    mask_h, mask_m = _best(dets_h), _best(dets_m)
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

    # Center + angles
    center  = find_clock_center(img_cv, mask_h, mask_m)
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
